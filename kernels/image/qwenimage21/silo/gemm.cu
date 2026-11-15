// gemm.cu: Qwen-Image 2.1's products in its silo - out [M, N] = A [M, K] . W [N, K]^T (+ bias), half operands, on
// mma.sync m16n8k16 with a cp.async pipeline (128 x 128 tiles, 32 deep, 4 stages, ldmatrix from swizzled shared memory,
// the tiles walked in groups of 8 rows so a wave's operands stay in L2). The engine's linears (after a slice of the
// stored weights is made half) and the VAE's convolutions (A gathered from the picture: an implicit GEMM, no im2col,
// a nearest 2x upsample folded into the gather) go through it.
//
// Accumulation: `f16acc` = 0 is float32 throughout (cuBLAS's numbers); 1 sums each 32-deep slice of K in half on the
// tensor cores (twice their float32 rate on a GeForce: 143 against 71.5 TFLOPS on the 5070, 51.6 against 25.8 on the
// 3060) and adds the slice's sum to float32 accumulators - a half rounding a slice of 32 products, not one on a
// running total over all of K.
#include "ns.h"
#include "ns_internal.hpp"

#include <cuda_fp16.h>
#include <cstdint>

namespace {

// the 16-deep steps a half sum spans before it is added to float32 (2: 32 products; 4 and 8 timed no faster)
#ifndef QI_KCH
#define QI_KCH 2
#endif
constexpr int KCH = QI_KCH;
// row tiles a group (the tile order: 4 and 16 timed the same)
#ifndef QI_GROUP
#define QI_GROUP 8
#endif

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t) __cvta_generic_to_shared(p); }

// a 16-byte copy, zero-filled when `ok` is false (src is then not read)
__device__ __forceinline__ void cp16(uint32_t dst, const void* src, bool ok) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

__device__ __forceinline__ void mma_f32(float c[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_f16(uint32_t c[2], const uint32_t a[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(c[0]), "+r"(c[1])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// A as rows of a matrix: row m, halves k..k+7
struct RowsA {
    const __half* a;
    long lda;
    int M, K;
    struct Row { const __half* p; bool ok; };
    __device__ Row row(int m) const { return {a + (long) min(m, M - 1) * lda, m < M}; }
    __device__ const __half* at(const Row& r, int k, bool& ok) const {
        ok = r.ok && k < K;
        return ok ? r.p + k : a;
    }
};

// A gathered from a channels-last picture x [H, W, Ci] for a 3 x 3 convolution (padding 1) of its nearest `up` x upsample
// (1 or 2): row m = output pixel (y, x) of [H * up, W * up], k = (dy * 3 + dx) * Ci + ci
struct ConvA {
    const __half* x;
    int H, W, Ci, up, M, K;
    struct Row { int y, x; bool ok; };
    __device__ Row row(int m) const {
        const int Wo = W * up;
        return {m / Wo, m % Wo, m < M};
    }
    __device__ const __half* at(const Row& r, int k, bool& ok) const {
        const int tap = k / Ci, ci = k - tap * Ci;
        const int yy = r.y + tap / 3 - 1, xx = r.x + tap % 3 - 1;
        ok = r.ok && k < K && yy >= 0 && yy < H * up && xx >= 0 && xx < W * up;
        if (!ok) return x;
        return x + ((long) (yy / up) * W + xx / up) * Ci + ci;
    }
};

// A tile shape: BM x BN outputs a block, BK deep a stage, warps WR x WC (each (BM / WR) x (BN / WC) outputs)
template <int BM_, int BN_, int BK_, int WR_, int WC_, int STAGES_>
struct Shape {
    static constexpr int BM = BM_, BN = BN_, BK = BK_, WR = WR_, WC = WC_, STAGES = STAGES_;
    static constexpr int THREADS = WR * WC * 32;
    static constexpr int TM = BM / WR, TN = BN / WC;          // a warp's tile
    static constexpr int MI = TM / 16, NI = TN / 8;           // its mma tiles
    static constexpr int CH = BK / 8;                          // 16-byte chunks a row of a stage
    static constexpr int A_HALVES = BM * BK, B_HALVES = BN * BK;
    static constexpr int ACH = BM * CH / THREADS, BCH = BN * CH / THREADS;   // chunks a thread loads a stage
    static_assert(BM * CH % THREADS == 0 && BN * CH % THREADS == 0, "whole chunks a thread");
    static_assert(MI >= 1 && NI % 2 == 0, "warp tile");
    // the byte offset of (row, chunk) in an operand's stage: the chunk XORed with the row's low bits, so ldmatrix's 8
    // rows at one chunk land on 8 distinct bank groups (rows of 64 bytes: bits 1-2; of 128 bytes: bits 0-2)
    static __device__ __forceinline__ uint32_t swz(int row, int chunk) {
        if (CH == 4) return (uint32_t) (row * 64 + ((chunk ^ ((row >> 1) & 3)) << 4));
        return (uint32_t) (row * (CH * 16) + ((chunk ^ (row & 7)) << 4));
    }
};

// Q8: W as Q8_0 split for the tensor cores (ns_image_qi21_silo_q8_split): int8 rows [N, K], each 32 values of a row
// reordered so a thread's fragment bytes are together (t * 8 + half * 4 + {2t, 2t + 1, 2t + 8, 2t + 9} of each 16), and
// the blocks' half scales [K / 32, N]; a fragment is made half on its way to the tensor cores (q * d rounded once: the
// shared dequant's numbers)
template <class S, bool Q8> struct Stage {
    static constexpr int B_BYTES = Q8 ? S::BN * S::BK + (S::BK / 32) * S::BN * 2 : S::B_HALVES * 2;
    static constexpr int BYTES = S::A_HALVES * 2 + B_BYTES;
    static constexpr int SMEM = S::STAGES * BYTES;
};

template <class S, class A, bool F16ACC, int EPI, bool Q8>
__global__ void __launch_bounds__(S::THREADS) gemm_k(A a, const void* __restrict__ wv, const __half* __restrict__ wd, int N, int K, const float* __restrict__ bias,
                                                     void* out, long ldo) {
    extern __shared__ __align__(128) unsigned char smem[];
    constexpr int BM = S::BM, BN = S::BN, BK = S::BK, STAGES = S::STAGES, MI = S::MI, NI = S::NI, CH = S::CH;
    constexpr int STAGE_BYTES = Stage<S, Q8>::BYTES;
    static_assert(!Q8 || BK % 32 == 0, "whole Q8_0 blocks a stage");
    constexpr int NB = BK / 32;
    const __half* w = (const __half*) wv;
    const int8_t* wq = (const int8_t*) wv;
    const uint32_t sbase = smem_u32(smem);
    const int M = a.M;
    // tile order: groups of 8 row tiles, the column tiles of a group walked with its rows (L2 keeps the group's operands)
    const int mt = (M + BM - 1) / BM, nt = (N + BN - 1) / BN;
    const int bid = blockIdx.x, width = QI_GROUP * nt, group = bid / width, first = group * QI_GROUP, gsz = min(mt - first, QI_GROUP);
    const int tm = first + (bid % width) % gsz, tn = (bid % width) / gsz;
    const int m0 = tm * BM, n0 = tn * BN;

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    // the loads: chunk idx = tid + i * THREADS of a stage is row idx / CH, chunk idx % CH (the same each stage)
    const int lc = tid % CH;
    typename A::Row ar[S::ACH];
#pragma unroll
    for (int i = 0; i < S::ACH; ++i) ar[i] = a.row(m0 + (tid + i * S::THREADS) / CH);
    const __half* wp[S::BCH];
    bool wok[S::BCH];
#pragma unroll
    for (int i = 0; i < S::BCH; ++i) {
        const int n = n0 + (tid + i * S::THREADS) / CH;
        wok[i] = n < N;
        wp[i] = w + (long) min(n, N - 1) * K;
    }
    // Q8: a stage's W is NB blocks of BN rows of 32 bytes (2 chunks each), then NB x BN half scales (BN / 8 chunks each)
    constexpr int QCH = Q8 ? (NB * BN * 2 + S::THREADS - 1) / S::THREADS : 1;
    const int KT = (K + BK - 1) / BK;

    auto load = [&](int stage, int kt) {
        const uint32_t sa = sbase + stage * STAGE_BYTES, sb = sa + S::A_HALVES * 2;
        const int k = kt * BK + lc * 8;
#pragma unroll
        for (int i = 0; i < S::ACH; ++i) {
            bool ok;
            const __half* p = a.at(ar[i], k, ok);
            cp16(sa + S::swz((tid + i * S::THREADS) / CH, lc), p, ok);
        }
        if constexpr (Q8) {
#pragma unroll
            for (int i = 0; i < QCH; ++i) {
                const int idx = tid + i * S::THREADS, b = idx / (BN * 2), rem = idx % (BN * 2), n = n0 + (rem >> 1);
                if (idx < NB * BN * 2) cp16(sb + idx * 16, n < N ? wq + (long) n * K + (kt * NB + b) * 32 + (rem & 1) * 16 : wq, n < N);
            }
            if (tid < NB * BN / 8) {
                const int b = tid / (BN / 8), c = tid % (BN / 8), n = n0 + c * 8;
                cp16(sb + NB * BN * 32 + tid * 16, n < N ? wd + (long) (kt * NB + b) * N + n : wd, n < N);
            }
        } else {
            const bool kin = k < K;
#pragma unroll
            for (int i = 0; i < S::BCH; ++i) cp16(sb + S::swz((tid + i * S::THREADS) / CH, lc), kin && wok[i] ? wp[i] + k : w, kin && wok[i]);
        }
    };

    const int wm = (warp / S::WC) * S::TM, wn = (warp % S::WC) * S::TN;
    float acc[MI][NI][4];
#pragma unroll
    for (int i = 0; i < MI; ++i)
#pragma unroll
        for (int j = 0; j < NI; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;
    uint32_t h[F16ACC ? MI : 1][F16ACC ? NI : 1][2];

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < KT) load(s, s);
        cp_commit();
    }
    for (int kt = 0; kt < KT; ++kt) {
        cp_wait<STAGES - 2>();
        __syncthreads();
        if (kt + STAGES - 1 < KT) load((kt + STAGES - 1) % STAGES, kt + STAGES - 1);
        cp_commit();
        const uint32_t sa = sbase + (kt % STAGES) * STAGE_BYTES, sb = sa + S::A_HALVES * 2;
#pragma unroll
        for (int ks = 0; ks < BK / 16; ++ks) {
            // half accumulators: zeroed at the start of each 32-deep slice, added to float32 at its end
            const int kk = kt * (BK / 16) + ks;
            if (F16ACC && kk % KCH == 0) {
#pragma unroll
                for (int i = 0; i < MI; ++i)
#pragma unroll
                    for (int j = 0; j < NI; ++j) h[i][j][0] = h[i][j][1] = 0u;
            }
            uint32_t af[MI][4], bf[NI][2];
#pragma unroll
            for (int i = 0; i < MI; ++i)
                ldsm_x4(af[i][0], af[i][1], af[i][2], af[i][3], sa + S::swz(wm + i * 16 + (lane & 15), ks * 2 + (lane >> 4)));
            if constexpr (Q8) {
                // the W fragments of this 16-deep step made half: row wn + j * 8 + g of block ks / 2, bytes 8t + 4 (ks % 2)
                const unsigned char* sq = smem + (sb - sbase) + (ks / 2) * BN * 32;
                const __half* sd = (const __half*) (smem + (sb - sbase) + NB * BN * 32) + (ks / 2) * BN;
                const int g = lane >> 2, t = lane & 3;
                const __half2 off = __float2half2_rn(1152.0f);
#pragma unroll
                for (int j = 0; j < NI; ++j) {
                    const int row = wn + j * 8 + g;
                    const uint32_t x = *reinterpret_cast<const uint32_t*>(sq + row * 32 + t * 8 + (ks % 2) * 4) ^ 0x80808080u;
                    const __half2 d2 = __half2half2(sd[row]);
                    // 0x64XX is half 1024 + XX: the byte (q + 128) as a half, less 1152, is q exactly
                    uint32_t lo = __byte_perm(x, 0x64646464u, 0x4140), hi = __byte_perm(x, 0x64646464u, 0x4342);
                    __half2 l2 = __hmul2(__hsub2(*reinterpret_cast<__half2*>(&lo), off), d2);
                    __half2 h2 = __hmul2(__hsub2(*reinterpret_cast<__half2*>(&hi), off), d2);
                    bf[j][0] = *reinterpret_cast<uint32_t*>(&l2);
                    bf[j][1] = *reinterpret_cast<uint32_t*>(&h2);
                }
            } else {
#pragma unroll
                for (int j = 0; j < NI / 2; ++j)
                    ldsm_x4(bf[2 * j][0], bf[2 * j][1], bf[2 * j + 1][0], bf[2 * j + 1][1],
                            sb + S::swz(wn + j * 16 + (lane & 7) + ((lane >> 4) << 3), ks * 2 + ((lane >> 3) & 1)));
            }
#pragma unroll
            for (int i = 0; i < MI; ++i)
#pragma unroll
                for (int j = 0; j < NI; ++j) {
                    if constexpr (F16ACC) mma_f16(h[i][j], af[i], bf[j][0], bf[j][1]);
                    else mma_f32(acc[i][j], af[i], bf[j][0], bf[j][1]);
                }
            if (F16ACC && (kk % KCH == KCH - 1 || kk == KT * (BK / 16) - 1)) {
#pragma unroll
                for (int i = 0; i < MI; ++i)
#pragma unroll
                    for (int j = 0; j < NI; ++j) {
                        const float2 lo = __half22float2(*reinterpret_cast<__half2*>(&h[i][j][0]));
                        const float2 hi = __half22float2(*reinterpret_cast<__half2*>(&h[i][j][1]));
                        acc[i][j][0] += lo.x; acc[i][j][1] += lo.y; acc[i][j][2] += hi.x; acc[i][j][3] += hi.y;
                    }
            }
        }
    }
    cp_wait<0>();

    // out: row g / g + 8 of each 16, columns 2t, 2t + 1 of each 8
    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int j = 0; j < NI; ++j) {
        const int col = n0 + wn + j * 8 + 2 * t;
        if (col >= N) continue;
        // EPI 2: `bias` is a gate, out float32 rows the product is added into (rounded to half first: the separate
        // half output and gated add's numbers)
        const float b0 = bias && EPI != 2 ? bias[col] : 0.0f, b1 = bias && EPI != 2 ? bias[col + 1] : 0.0f;
        const float g0 = EPI == 2 ? bias[col] : 0.0f, g1 = EPI == 2 ? bias[col + 1] : 0.0f;
#pragma unroll
        for (int i = 0; i < MI; ++i) {
#pragma unroll
            for (int hf = 0; hf < 2; ++hf) {
                const int row = m0 + wm + i * 16 + g + hf * 8;
                if (row >= M) continue;
                const float v0 = acc[i][j][2 * hf] + b0, v1 = acc[i][j][2 * hf + 1] + b1;
                if constexpr (EPI == 1) {
                    *reinterpret_cast<float2*>((float*) out + (long) row * ldo + col) = make_float2(v0, v1);
                } else if constexpr (EPI == 2) {
                    float2* xo = reinterpret_cast<float2*>((float*) out + (long) row * ldo + col);
                    const float2 hv = __half22float2(__floats2half2_rn(v0, v1));
                    float2 cur = *xo;
                    cur.x += g0 * hv.x;
                    cur.y += g1 * hv.y;
                    *xo = cur;
                } else {
                    *reinterpret_cast<__half2*>((__half*) out + (long) row * ldo + col) = __floats2half2_rn(v0, v1);
                }
            }
        }
    }
}

template <class S, class A, bool F16ACC, int EPI, bool Q8 = false>
int launch(ns_gpu* g, const A& a, const void* w, int64_t N, int64_t K, const float* bias, void* out, int64_t ldo, const void* wd = nullptr) {
    static int set_on[16] = {};
    constexpr int SM = Stage<S, Q8>::SMEM;
    // the shared memory a block needs past 48 KiB, asked for once a device
    if (g->dev < 16 && !set_on[g->dev]) {
        NS_CU(cudaFuncSetAttribute(gemm_k<S, A, F16ACC, EPI, Q8>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM));
        set_on[g->dev] = 1;
    }
    const int64_t tiles = ((a.M + S::BM - 1) / S::BM) * ((N + S::BN - 1) / S::BN);
    gemm_k<S, A, F16ACC, EPI, Q8><<<(unsigned) tiles, S::THREADS, SM, g->q>>>(a, w, (const __half*) wd, (int) N, (int) K, bias, out, (long) ldo);
    NS_LAUNCH_CHECK();
    return 0;
}

// the tile shapes, by timing (a 16384 x 4096 by 4096 x 8192 product, half sums): the 5070 (sm_120) 128 x 128 with 4
// warps of 64 x 64 and 3 stages (106 TFLOPS of its 143; cuBLAS's float32 sums 70); the 3060 (sm_86) 128 x 256 with
// 8 warps of 64 x 64 (34 of its 51.6; cuBLAS 27)
using Sm120 = Shape<128, 128, 32, 2, 2, 3>;
using Sm86 = Shape<128, 256, 32, 2, 4, 3>;
#ifdef QI_Q8BM
using Q8Sm120 = Shape<QI_Q8BM, QI_Q8BN, QI_Q8BK, QI_Q8WR, QI_Q8WC, QI_Q8ST>;
using Q8Sm86 = Q8Sm120;
#else
// the Q8 operand's stages are 12.25 KiB: four still leave room for two blocks an SM (99.7 TFLOPS on the 5070; the
// 3060 runs its products on dequantized slices instead - its Q8 fragments cost more than the dequant)
using Q8Sm120 = Shape<128, 128, 32, 2, 2, 4>;
using Q8Sm86 = Sm86;
#endif

int major_of(int dev) {
    static int major[16] = {};
    if (dev < 0 || dev >= 16) return 0;
    if (!major[dev]) cudaDeviceGetAttribute(&major[dev], cudaDevAttrComputeCapabilityMajor, dev);
    return major[dev];
}

template <class S, class A>
int dispatch_s(ns_gpu* g, const A& a, const void* w, int64_t N, int64_t K, const float* bias, void* out, int epi, int64_t ldo, int f16acc) {
    if (f16acc) {
        if (epi == 2) return launch<S, A, true, 2>(g, a, w, N, K, bias, out, ldo);
        return epi ? launch<S, A, true, 1>(g, a, w, N, K, bias, out, ldo) : launch<S, A, true, 0>(g, a, w, N, K, bias, out, ldo);
    }
    if (epi == 2) return launch<S, A, false, 2>(g, a, w, N, K, bias, out, ldo);
    return epi ? launch<S, A, false, 1>(g, a, w, N, K, bias, out, ldo) : launch<S, A, false, 0>(g, a, w, N, K, bias, out, ldo);
}

template <class S>
int dispatch_q8_s(ns_gpu* g, const RowsA& a, const void* q, const void* d, int64_t N, int64_t K, const float* gate, void* out, int epi, int64_t ldo,
                  int f16acc) {
    if (f16acc) {
        if (epi == 2) return launch<S, RowsA, true, 2, true>(g, a, q, N, K, gate, out, ldo, d);
        return epi ? launch<S, RowsA, true, 1, true>(g, a, q, N, K, nullptr, out, ldo, d) : launch<S, RowsA, true, 0, true>(g, a, q, N, K, nullptr, out, ldo, d);
    }
    if (epi == 2) return launch<S, RowsA, false, 2, true>(g, a, q, N, K, gate, out, ldo, d);
    return epi ? launch<S, RowsA, false, 1, true>(g, a, q, N, K, nullptr, out, ldo, d) : launch<S, RowsA, false, 0, true>(g, a, q, N, K, nullptr, out, ldo, d);
}

// a Q8_0 matrix's blocks [N, K / 32] (34 bytes: a half scale, 32 int8) -> the split form the Q8 products read
__global__ void q8_split_k(const uint8_t* __restrict__ src, int N, int K, int8_t* __restrict__ q, __half* __restrict__ d) {
    const long id = (long) blockIdx.x * blockDim.x + threadIdx.x;
    const int nb = K / 32;
    if (id >= (long) N * nb) return;
    const int n = (int) (id / nb), b = (int) (id % nb);
    const uint8_t* blk = src + id * 34;
    d[(long) b * N + n] = __ushort_as_half((unsigned short) (blk[0] | (blk[1] << 8)));
    int8_t* o = q + (long) n * K + b * 32;
#pragma unroll
    for (int t = 0; t < 4; ++t)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int base = hh * 16 + 2 * t;
            o[t * 8 + hh * 4 + 0] = (int8_t) blk[2 + base];
            o[t * 8 + hh * 4 + 1] = (int8_t) blk[2 + base + 1];
            o[t * 8 + hh * 4 + 2] = (int8_t) blk[2 + base + 8];
            o[t * 8 + hh * 4 + 3] = (int8_t) blk[2 + base + 9];
        }
}

template <class A>
int dispatch(ns_gpu* g, const A& a, const void* w, int64_t N, int64_t K, const float* bias, void* out, int out_f32, int64_t ldo, int f16acc) {
#ifdef QI_BM
    return dispatch_s<Shape<QI_BM, QI_BN, QI_BK, QI_WR, QI_WC, QI_ST>>(g, a, w, N, K, bias, out, out_f32, ldo, f16acc);
#else
    if (major_of(g->dev) >= 12) return dispatch_s<Sm120>(g, a, w, N, K, bias, out, out_f32, ldo, f16acc);
    return dispatch_s<Sm86>(g, a, w, N, K, bias, out, out_f32, ldo, f16acc);
#endif
}

}  // namespace

extern "C" {

// out [M, N] (row stride ldo; out_f32 0: half, 1: float32, 2: float32 += bias[col] * the product rounded to half, bias
// then being a gate) = x [M, K] (row stride lda) . w [N, K]^T + bias (float32
// [N] or NULL). K, lda a multiple of 8 (whole 16-byte rows), N and ldo even.
int ns_image_qi21_silo_gemm(ns_gpu* g, const void* x, int64_t M, int64_t K, int64_t lda, const void* w, int64_t N, const float* bias, void* out,
                            int out_f32, int64_t ldo, int f16acc) {
    if (M <= 0 || N <= 0) return 0;
    if (K % 8 || lda % 8 || N % 2 || ldo % 2) return ns_fail("silo gemm: K and lda multiples of 8, N and ldo even");
    if (M > INT32_MAX / 2 || N > INT32_MAX / 2) return ns_fail("silo gemm: too many rows");
    NS_CU(cudaSetDevice(g->dev));
    RowsA a{(const __half*) x, (long) lda, (int) M, (int) K};
    return dispatch(g, a, w, N, K, bias, out, out_f32, ldo, f16acc);
}

// A Q8_0 matrix src [N, K / 32 blocks of 34 bytes] split for ns_image_qi21_silo_gemm_q8: q int8 [N, K] (each 32's
// bytes reordered for the fragments), d half [K / 32, N]. K a multiple of 32, N of 8.
int ns_image_qi21_silo_q8_split(ns_gpu* g, const void* src, int64_t N, int64_t K, void* q, void* d) {
    if (K % 32 || N % 8) return ns_fail("silo q8 split: K a multiple of 32, N of 8");
    NS_CU(cudaSetDevice(g->dev));
    const long n = (long) N * (K / 32);
    q8_split_k<<<(unsigned) ((n + 255) / 256), 256, 0, g->q>>>((const uint8_t*) src, (int) N, (int) K, (int8_t*) q, (__half*) d);
    NS_LAUNCH_CHECK();
    return 0;
}

// out [M, N] = x [M, K] half . W^T with W a split Q8_0 matrix (q, d from ns_image_qi21_silo_q8_split)
// `epi`: 0 out half, 1 float32, 2 out float32 += gate[col] * the product rounded to half (gate float32 [N])
int ns_image_qi21_silo_gemm_q8(ns_gpu* g, const void* x, int64_t M, int64_t K, int64_t lda, const void* q, const void* d, int64_t N, const float* gate,
                               void* out, int epi, int64_t ldo, int f16acc) {
    if (epi == 2 && !gate) return ns_fail("silo gemm q8: a gated add without its gate");
    if (M <= 0 || N <= 0) return 0;
    if (K % 32 || lda % 8 || N % 8 || ldo % 2) return ns_fail("silo gemm q8: K a multiple of 32, lda and N of 8, ldo even");
    NS_CU(cudaSetDevice(g->dev));
    RowsA a{(const __half*) x, (long) lda, (int) M, (int) K};
#ifdef QI_BM
    return dispatch_q8_s<Shape<QI_BM, QI_BN, QI_BK, QI_WR, QI_WC, QI_ST>>(g, a, q, d, N, K, gate, out, epi, ldo, f16acc);
#else
    if (major_of(g->dev) >= 12) return dispatch_q8_s<Q8Sm120>(g, a, q, d, N, K, gate, out, epi, ldo, f16acc);
    return dispatch_q8_s<Q8Sm86>(g, a, q, d, N, K, gate, out, epi, ldo, f16acc);
#endif
}

// A 3 x 3 convolution (padding 1) of the nearest `up` x upsample (1 or 2) of x [H, W, Ci] half, channels last:
// out [H * up, W * up, Co] half = conv + bias. w [Co, 3, 3, Ci] half (the taps' order: row, column, channel). Ci a
// multiple of 8, Co even.
int ns_image_qi21_silo_conv3x3(ns_gpu* g, const void* x, int64_t H, int64_t W, int64_t Ci, int up, const void* w, int64_t Co, const float* bias,
                               void* out, int f16acc) {
    if (Ci % 8 || Co % 2) return ns_fail("silo conv3x3: channels in a multiple of 8, out even");
    if (up != 1 && up != 2) return ns_fail("silo conv3x3: upsample 1 or 2");
    if (H * W * up * up > INT32_MAX / 2 || 9 * Ci > INT32_MAX / 2) return ns_fail("silo conv3x3: too large");
    NS_CU(cudaSetDevice(g->dev));
    ConvA a{(const __half*) x, (int) H, (int) W, (int) Ci, up, (int) (H * W * up * up), (int) (9 * Ci)};
    return dispatch(g, a, w, Co, 9 * Ci, bias, out, 0, Co, f16acc);
}

}  // extern "C"
