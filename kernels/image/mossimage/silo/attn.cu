// attn.cu: MOSS-Image's attention (head size 128, half in and out), built once a card. FlashAttention-2's scheme on
// mma.sync: a block of queries (16 a warp) against 64-key tiles that never leave the chip, the softmax online. Over
// nsd_flash_attention (the same scheme, Qwen-Image 2.1's silo kernel): the key and value tiles double-buffered with
// cp.async (the next tile loads while this one is used), their fragments by ldmatrix (the values' transposed by
// ldmatrix.trans, where that kernel built each from four 16-bit reads), and the block's width per card.
//
// The scores accumulated in half (fp8 -1, the engine's default): GeForce cards run that product at twice the float32
// accumulation's rate - the kernel 20% faster, the reference checks unchanged (2026-10-26).
//
// FP8 (fp8 >= 1; built only for a card with FP8 tensor cores, sm_89 and up): the scores from e4m3 copies of q and k,
// each 128-value row scaled to its own maximum (m16n8k32: twice the features a product); fp8 2: the probabilities and
// the values in e4m3 too, each value column scaled. Measured on the 5070 it is no faster (a GeForce card's FP8 product
// with a float32 accumulation runs at half precision's rate, and the copies cost a pass) and less exact (8 guided steps
// at cosine 0.977 against 0.996): kept to be chosen, not the default.
#include "ns.h"
#include "ns_internal.hpp"
#include "silo.h"

#include <cuda_fp16.h>
#if NS_SILO_SM >= 89
#include <cuda_fp8.h>
#endif
#include <algorithm>
#include <cfloat>
#include <string>
#include <cstdint>

#ifndef NS_SILO_SM
#define NS_SILO_SM 0
#endif

namespace {

constexpr int D = 128;     // head size
constexpr int BK = 64;     // keys a tile
constexpr int LDK = 136;   // smem row stride in halves: 272 bytes, so ldmatrix's 8 rows land on distinct banks

__device__ __forceinline__ void mma16816(float c[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// the same product accumulated in half (GeForce cards run it at twice the rate of a float32 accumulation)
__device__ __forceinline__ void mma16816h(uint32_t c[2], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(c[0]), "+r"(c[1])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pack(float a, float b) {
    __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ void cp16(void* smem, const void* g, bool pred) {
    const unsigned s = (unsigned) __cvta_generic_to_shared(smem);
    const int n = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(g), "r"(n));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldsm4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* smem) {
    const unsigned s = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(s));
}
__device__ __forceinline__ void ldsm4t(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* smem) {
    const unsigned s = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(s));
}

// the online softmax of one tile's scores s (already in log2 units, masked), into o, m, l
__device__ __forceinline__ void online(float s[8][4], float o[16][4], float& m0, float& m1, float& l0, float& l1) {
    float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
    for (int nb = 0; nb < 8; ++nb) {
        mx0 = fmaxf(mx0, fmaxf(s[nb][0], s[nb][1]));
        mx1 = fmaxf(mx1, fmaxf(s[nb][2], s[nb][3]));
    }
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
    const float n0 = fmaxf(m0, mx0), n1 = fmaxf(m1, mx1);
    const float u0 = n0 == -INFINITY ? 0.0f : n0, u1 = n1 == -INFINITY ? 0.0f : n1;
    const float a0 = exp2f(m0 - u0), a1 = exp2f(m1 - u1);
    m0 = n0;
    m1 = n1;
    l0 *= a0;
    l1 *= a1;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        o[i][0] *= a0; o[i][1] *= a0;
        o[i][2] *= a1; o[i][3] *= a1;
    }
#pragma unroll
    for (int nb = 0; nb < 8; ++nb) {
        s[nb][0] = exp2f(s[nb][0] - u0);
        s[nb][1] = exp2f(s[nb][1] - u0);
        s[nb][2] = exp2f(s[nb][2] - u1);
        s[nb][3] = exp2f(s[nb][3] - u1);
        l0 += s[nb][0] + s[nb][1];
        l1 += s[nb][2] + s[nb][3];
    }
}

// O += P V for one tile, the values half in smem (Vs[key][dim]), P from the score accumulators
__device__ __forceinline__ void pv_half(float s[8][4], float o[16][4], const __half* Vs, int lane) {
    const int jj = lane >> 3, rr = lane & 7;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint32_t pa0 = pack(s[2 * j][0], s[2 * j][1]), pa1 = pack(s[2 * j][2], s[2 * j][3]);
        const uint32_t pa2 = pack(s[2 * j + 1][0], s[2 * j + 1][1]), pa3 = pack(s[2 * j + 1][2], s[2 * j + 1][3]);
#pragma unroll
        for (int nd2 = 0; nd2 < 8; ++nd2) {
            uint32_t r0, r1, r2, r3;
            ldsm4t(r0, r1, r2, r3, &Vs[(j * 16 + (jj & 1) * 8 + rr) * LDK + nd2 * 16 + (jj >> 1) * 8]);
            mma16816(o[2 * nd2], pa0, pa1, pa2, pa3, r0, r1);
            mma16816(o[2 * nd2 + 1], pa0, pa1, pa2, pa3, r2, r3);
        }
    }
}

// the normalized output of a warp's 16 rows
__device__ __forceinline__ void store(float o[16][4], float l0, float l1, __half* out, int r0, int r1, int Sq, int H, int h, int t) {
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    const float i0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, i1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
#pragma unroll
    for (int nd = 0; nd < 16; ++nd) {
        const int d = nd * 8 + 2 * t;
        if (r0 < Sq) *reinterpret_cast<uint32_t*>(out + ((long) r0 * H + h) * D + d) = pack(o[nd][0] * i0, o[nd][1] * i0);
        if (r1 < Sq) *reinterpret_cast<uint32_t*>(out + ((long) r1 * H + h) * D + d) = pack(o[nd][2] * i1, o[nd][3] * i1);
    }
}

template <int WARPS, bool QK16>
__global__ void __launch_bounds__(WARPS * 32) fa_half_k(const __half* __restrict__ q, const __half* __restrict__ k, const __half* __restrict__ v,
                                                        __half* __restrict__ out, int Sq, int Skv, int H, long qs, long kvs, float scale_log2) {
    constexpr int BQ = WARPS * 16, NT = WARPS * 32;
    extern __shared__ __half smem[];
    __half* Ks = smem;                     // [2][BK][LDK]
    __half* Vs = smem + 2 * BK * LDK;      // [2][BK][LDK]
    const int h = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int qrow0 = blockIdx.x * BQ + warp * 16;
    const int r0 = qrow0 + g, r1 = r0 + 8;

    auto load = [&](int stage, int kt0) {
        for (int i = threadIdx.x; i < BK * D / 8; i += NT) {
            const int row = i / (D / 8), c = (i % (D / 8)) * 8;
            const int key = kt0 + row;
            const bool in = key < Skv;
            const long at = (long) (in ? key : 0) * kvs + (long) h * D + c;
            cp16(&Ks[(stage * BK + row) * LDK + c], k + at, in);
            cp16(&Vs[(stage * BK + row) * LDK + c], v + at, in);
        }
        cp_commit();
    };
    load(0, 0);

    uint32_t qa[8][4];
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
        const int d = kk * 16 + 2 * t;
        qa[kk][0] = r0 < Sq ? *reinterpret_cast<const uint32_t*>(q + r0 * qs + (long) h * D + d) : 0u;
        qa[kk][1] = r1 < Sq ? *reinterpret_cast<const uint32_t*>(q + r1 * qs + (long) h * D + d) : 0u;
        qa[kk][2] = r0 < Sq ? *reinterpret_cast<const uint32_t*>(q + r0 * qs + (long) h * D + d + 8) : 0u;
        qa[kk][3] = r1 < Sq ? *reinterpret_cast<const uint32_t*>(q + r1 * qs + (long) h * D + d + 8) : 0u;
    }
    float o[16][4];
#pragma unroll
    for (int i = 0; i < 16; ++i) { o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f; }
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;

    const int tiles = (Skv + BK - 1) / BK;
    const int jj = lane >> 3, rr = lane & 7;
    for (int it = 0; it < tiles; ++it) {
        if (it + 1 < tiles) {
            load((it + 1) & 1, (it + 1) * BK);
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();
        const __half* K = Ks + (it & 1) * BK * LDK;
        const __half* V = Vs + (it & 1) * BK * LDK;
        float s[8][4];
        if (QK16) {
            // the scores accumulated in half (q and k are normed: a score stays well inside half's range)
            uint32_t sh[8][2];
#pragma unroll
            for (int nb = 0; nb < 8; ++nb) { sh[nb][0] = sh[nb][1] = 0u; }
#pragma unroll
            for (int kk = 0; kk < 8; ++kk) {
#pragma unroll
                for (int nb2 = 0; nb2 < 4; ++nb2) {
                    uint32_t b0, b1, b2, b3;
                    ldsm4(b0, b1, b2, b3, &K[((nb2 * 2 + (jj >> 1)) * 8 + rr) * LDK + kk * 16 + (jj & 1) * 8]);
                    mma16816h(sh[2 * nb2], qa[kk][0], qa[kk][1], qa[kk][2], qa[kk][3], b0, b1);
                    mma16816h(sh[2 * nb2 + 1], qa[kk][0], qa[kk][1], qa[kk][2], qa[kk][3], b2, b3);
                }
            }
#pragma unroll
            for (int nb = 0; nb < 8; ++nb) {
                const float2 lo = __half22float2(*reinterpret_cast<const __half2*>(&sh[nb][0]));
                const float2 hi = __half22float2(*reinterpret_cast<const __half2*>(&sh[nb][1]));
                s[nb][0] = lo.x; s[nb][1] = lo.y; s[nb][2] = hi.x; s[nb][3] = hi.y;
            }
        } else {
#pragma unroll
            for (int nb = 0; nb < 8; ++nb) { s[nb][0] = s[nb][1] = s[nb][2] = s[nb][3] = 0.0f; }
#pragma unroll
            for (int kk = 0; kk < 8; ++kk) {
#pragma unroll
                for (int nb2 = 0; nb2 < 4; ++nb2) {
                    uint32_t b0, b1, b2, b3;
                    ldsm4(b0, b1, b2, b3, &K[((nb2 * 2 + (jj >> 1)) * 8 + rr) * LDK + kk * 16 + (jj & 1) * 8]);
                    mma16816(s[2 * nb2], qa[kk][0], qa[kk][1], qa[kk][2], qa[kk][3], b0, b1);
                    mma16816(s[2 * nb2 + 1], qa[kk][0], qa[kk][1], qa[kk][2], qa[kk][3], b2, b3);
                }
            }
        }
        const int kt0 = it * BK;
#pragma unroll
        for (int nb = 0; nb < 8; ++nb) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int col = kt0 + nb * 8 + 2 * t + (e & 1);
                s[nb][e] = col < Skv ? s[nb][e] * scale_log2 : -INFINITY;
            }
        }
        online(s, o, m0, m1, l0, l1);
        pv_half(s, o, V, lane);
        __syncthreads();
    }
    store(o, l0, l1, out, r0, r1, Sq, H, h, t);
}

#if NS_SILO_SM >= 89
constexpr int LD8 = 144;   // an e4m3 row of 128 in smem: 144 bytes (8 rows a quad of 4-byte reads on distinct banks)

__device__ __forceinline__ void mma_e4m3(float c[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pack8(float a, float b, float c, float d) {
    const __nv_fp8x4_e4m3 x(make_float4(a, b, c, d));
    return *reinterpret_cast<const uint32_t*>(&x);
}

// rows of [S, H, 128] half (row stride rs) to e4m3 [S, H, 128] with a scale each (absmax / 448), a warp a row
__global__ void quant8_k(const __half* x, int S, int H, long rs, uint8_t* q, float* sc) {
    const long row = (long) blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    if (row >= (long) S * H) return;
    const long s = row / H, h = row % H;
    const __half* src = x + s * rs + h * D + lane * 4;
    const float2 a = __half22float2(*reinterpret_cast<const __half2*>(src));
    const float2 b = __half22float2(*reinterpret_cast<const __half2*>(src + 2));
    float m = fmaxf(fmaxf(fabsf(a.x), fabsf(a.y)), fmaxf(fabsf(b.x), fabsf(b.y)));
#pragma unroll
    for (int off = 16; off; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
    const float scale = fmaxf(m / 448.0f, 1e-30f), inv = 1.0f / scale;
    *reinterpret_cast<uint32_t*>(q + row * D + lane * 4) = pack8(a.x * inv, a.y * inv, b.x * inv, b.y * inv);
    if (lane == 0) sc[row] = scale;
}

// a value column's scale over all keys of a head ([H * 128], absmax / 448) and the values in e4m3, transposed to
// [H, 128, Skv8] (Skv8: Skv rounded up to 64, zeros past it) for the P V product's B operand
__global__ void vscale_k(const __half* v, int Skv, int H, long kvs, float* vsc) {
    const int col = blockIdx.x;   // h * 128 + d
    float m = 0.0f;
    for (int s = threadIdx.x; s < Skv; s += blockDim.x) m = fmaxf(m, fabsf(__half2float(v[(long) s * kvs + col])));
    __shared__ float red[256];
    red[threadIdx.x] = m;
    __syncthreads();
    for (int w = blockDim.x / 2; w; w >>= 1) {
        if ((int) threadIdx.x < w) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + w]);
        __syncthreads();
    }
    if (threadIdx.x == 0) vsc[col] = fmaxf(red[0] / 448.0f, 1e-30f);
}

__global__ void vquant_k(const __half* v, int Skv, int Skv8, int H, long kvs, const float* vsc, uint8_t* vt) {
    const long i = (long) blockIdx.x * blockDim.x + threadIdx.x;   // over H * 128 * Skv8 / 4
    const long n = (long) H * D * Skv8 / 4;
    if (i >= n) return;
    const long col = i / (Skv8 / 4), s0 = (i % (Skv8 / 4)) * 4;
    const float inv = 1.0f / vsc[col];
    float f[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) f[e] = s0 + e < Skv ? __half2float(v[(s0 + e) * kvs + col]) * inv : 0.0f;
    *reinterpret_cast<uint32_t*>(vt + col * Skv8 + s0) = pack8(f[0], f[1], f[2], f[3]);
}

// the scores from e4m3 q and k (q8 / k8 [S, H, 128], qsc / ksc [S, H]); pv8: P and V in e4m3 too (vt [H, 128, Skv8],
// vsc [H * 128]), else V half from v
template <int WARPS, bool PV8>
__global__ void __launch_bounds__(WARPS * 32) fa_fp8_k(const uint8_t* __restrict__ q8, const float* __restrict__ qsc, const uint8_t* __restrict__ k8,
                                                       const float* __restrict__ ksc, const __half* __restrict__ v, const uint8_t* __restrict__ vt,
                                                       const float* __restrict__ vsc, __half* __restrict__ out, int Sq, int Skv, int Skv8, int H, long kvs,
                                                       float scale_log2) {
    constexpr int BQ = WARPS * 16, NT = WARPS * 32;
    extern __shared__ __align__(16) unsigned char sm8[];
    uint8_t* Ks = sm8;                                          // [2][BK][LD8]
    float* Kc = reinterpret_cast<float*>(sm8 + 2 * BK * LD8);   // [2][BK] the keys' scales
    unsigned char* rest = sm8 + 2 * BK * LD8 + 2 * BK * 4;
    // values: half [2][BK][LDK], or e4m3 transposed [2][128][BK + 16]
    constexpr int LDV8 = BK + 16;
    __half* Vs = reinterpret_cast<__half*>(rest);
    uint8_t* Vt = rest;
    const int h = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int qrow0 = blockIdx.x * BQ + warp * 16;
    const int r0 = qrow0 + g, r1 = r0 + 8;

    auto load = [&](int stage, int kt0) {
        for (int i = threadIdx.x; i < BK * D / 16; i += NT) {
            const int row = i / (D / 16), c = (i % (D / 16)) * 16;
            const int key = kt0 + row;
            const bool in = key < Skv;
            cp16(&Ks[(stage * BK + row) * LD8 + c], k8 + ((long) (in ? key : 0) * H + h) * D + c, in);
        }
        for (int i = threadIdx.x; i < BK; i += NT) Kc[stage * BK + i] = kt0 + i < Skv ? ksc[(long) (kt0 + i) * H + h] : 0.0f;
        if (PV8) {
            for (int i = threadIdx.x; i < D * BK / 16; i += NT) {
                const int d = i / (BK / 16), c = (i % (BK / 16)) * 16;
                cp16(&Vt[(stage * D + d) * LDV8 + c], vt + ((long) h * D + d) * Skv8 + kt0 + c, true);
            }
        } else {
            for (int i = threadIdx.x; i < BK * D / 8; i += NT) {
                const int row = i / (D / 8), c = (i % (D / 8)) * 8;
                const int key = kt0 + row;
                const bool in = key < Skv;
                cp16(&Vs[(stage * BK + row) * LDK + c], v + (long) (in ? key : 0) * kvs + (long) h * D + c, in);
            }
        }
        cp_commit();
    };
    load(0, 0);

    // q's rows as A fragments of the m16n8k32 product: 4 steps of 32 features
    uint32_t qa[4][4];
    const float sq0 = r0 < Sq ? qsc[(long) r0 * H + h] : 0.0f, sq1 = r1 < Sq ? qsc[(long) r1 * H + h] : 0.0f;
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
        const int d = kk * 32 + 4 * t;
        qa[kk][0] = r0 < Sq ? *reinterpret_cast<const uint32_t*>(q8 + ((long) r0 * H + h) * D + d) : 0u;
        qa[kk][1] = r1 < Sq ? *reinterpret_cast<const uint32_t*>(q8 + ((long) r1 * H + h) * D + d) : 0u;
        qa[kk][2] = r0 < Sq ? *reinterpret_cast<const uint32_t*>(q8 + ((long) r0 * H + h) * D + d + 16) : 0u;
        qa[kk][3] = r1 < Sq ? *reinterpret_cast<const uint32_t*>(q8 + ((long) r1 * H + h) * D + d + 16) : 0u;
    }
    float o[16][4];
#pragma unroll
    for (int i = 0; i < 16; ++i) { o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f; }
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;

    const int tiles = (Skv + BK - 1) / BK;
    for (int it = 0; it < tiles; ++it) {
        if (it + 1 < tiles) {
            load((it + 1) & 1, (it + 1) * BK);
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();
        const uint8_t* K = Ks + (it & 1) * BK * LD8;
        const float* kc = Kc + (it & 1) * BK;
        float s[8][4];
#pragma unroll
        for (int nb = 0; nb < 8; ++nb) { s[nb][0] = s[nb][1] = s[nb][2] = s[nb][3] = 0.0f; }
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
            for (int nb = 0; nb < 8; ++nb) {
                const uint8_t* kr = &K[(nb * 8 + g) * LD8 + kk * 32 + 4 * t];
                mma_e4m3(s[nb], qa[kk][0], qa[kk][1], qa[kk][2], qa[kk][3], *reinterpret_cast<const uint32_t*>(kr),
                         *reinterpret_cast<const uint32_t*>(kr + 16));
            }
        }
        const int kt0 = it * BK;
#pragma unroll
        for (int nb = 0; nb < 8; ++nb) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int cl = nb * 8 + 2 * t + (e & 1);
                const float sc = ((e >> 1) ? sq1 : sq0) * kc[cl] * scale_log2;
                s[nb][e] = kt0 + cl < Skv ? s[nb][e] * sc : -INFINITY;
            }
        }
        online(s, o, m0, m1, l0, l1);
        if (PV8) {
            // P (probabilities, at most 1: scaled to 448) and V in e4m3: k = 32 keys a product; B[k = key][n = dim]
            const uint8_t* V8 = Vt + (it & 1) * D * LDV8;
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const uint32_t pa0 = pack8(s[4 * j][0] * 448.0f, s[4 * j][1] * 448.0f, s[4 * j + 1][0] * 448.0f, s[4 * j + 1][1] * 448.0f);
                const uint32_t pa1 = pack8(s[4 * j][2] * 448.0f, s[4 * j][3] * 448.0f, s[4 * j + 1][2] * 448.0f, s[4 * j + 1][3] * 448.0f);
                const uint32_t pa2 = pack8(s[4 * j + 2][0] * 448.0f, s[4 * j + 2][1] * 448.0f, s[4 * j + 3][0] * 448.0f, s[4 * j + 3][1] * 448.0f);
                const uint32_t pa3 = pack8(s[4 * j + 2][2] * 448.0f, s[4 * j + 2][3] * 448.0f, s[4 * j + 3][2] * 448.0f, s[4 * j + 3][3] * 448.0f);
#pragma unroll
                for (int nd = 0; nd < 16; ++nd) {
                    const uint8_t* vr = &V8[(nd * 8 + g) * LDV8 + j * 32 + 4 * t];
                    float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    mma_e4m3(c, pa0, pa1, pa2, pa3, *reinterpret_cast<const uint32_t*>(vr), *reinterpret_cast<const uint32_t*>(vr + 16));
                    // the value columns' scales (each thread's two columns of this block) and P's 448
                    const float v0 = vsc[h * D + nd * 8 + 2 * t] / 448.0f, v1 = vsc[h * D + nd * 8 + 2 * t + 1] / 448.0f;
                    o[nd][0] += c[0] * v0; o[nd][1] += c[1] * v1;
                    o[nd][2] += c[2] * v0; o[nd][3] += c[3] * v1;
                }
            }
        } else {
            pv_half(s, o, Vs + (it & 1) * BK * LDK, lane);
        }
        __syncthreads();
    }
    store(o, l0, l1, out, r0, r1, Sq, H, h, t);
}

// m16n8k32's A operand pairs keys (4t..4t+3) where the score accumulators hold (2t, 2t+1) of two 8-key blocks: the
// packing above takes keys 2t, 2t+1 of block 4j and of block 4j+1 for a0 - which is the A layout when the B rows
// (keys) are permuted the same way; V8's columns are loaded in that order (vquant writes keys in order, so the
// permutation is applied here instead: see vt_perm_k)
__global__ void vt_perm_k(uint8_t* vt, int rows, int Skv8) {
    // per 32-key chunk of a row: key order 0..31 -> the order the P fragment's bytes take (8-key blocks b0 b1 b2 b3:
    // a0 bytes = b0[2t, 2t+1], b1[2t, 2t+1] for t = 0..3 -> position 4t..4t+3 holds keys 2t, 2t+1, 8 + 2t, 8 + 2t + 1;
    // a2 likewise for blocks b2 b3 at positions 16 + 4t ...)
    const long i = (long) blockIdx.x * blockDim.x + threadIdx.x;   // one 32-key chunk
    const long n = (long) rows * (Skv8 / 32);
    if (i >= n) return;
    uint8_t* c = vt + (i / (Skv8 / 32)) * Skv8 + (i % (Skv8 / 32)) * 32;
    uint8_t src[32];
#pragma unroll
    for (int e = 0; e < 32; ++e) src[e] = c[e];
#pragma unroll
    for (int half = 0; half < 2; ++half)
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int base = half * 16;
            c[base + 4 * t + 0] = src[base + 2 * t];
            c[base + 4 * t + 1] = src[base + 2 * t + 1];
            c[base + 4 * t + 2] = src[base + 8 + 2 * t];
            c[base + 4 * t + 3] = src[base + 8 + 2 * t + 1];
        }
}
#endif

// the FP8 copies' buffers, grown and kept (one model a process)
struct Fp8Bufs {
    uint8_t* q8 = nullptr;
    uint8_t* k8 = nullptr;
    uint8_t* vt = nullptr;
    float* sc = nullptr;
    size_t cap8 = 0, capv = 0, capsc = 0;
};

}  // namespace

extern "C" {

int ns_image_moss_silo_info(int* sm, int* fp8) {
    *sm = NS_SILO_SM;
    *fp8 = NS_SILO_SM >= 89;
    return 0;
}

int ns_image_moss_silo_attention(ns_gpu* gp, const void* q, int64_t S, int64_t qs, const void* k, const void* v, int64_t Skv, int64_t kvs,
                                 int64_t H, void* out, int warps, int fp8) {
    if (qs % 8 || kvs % 8) return ns_fail("moss attention: rows of whole 16-byte groups (strides multiples of 8)");
    NS_CU(cudaSetDevice(gp->dev));
    // 8 warps (128 queries) a block by default: the faster on both cards, by 4% on the 3060 and the 5070 alike at
    // 1024 x 1024 (measured, 2026-10-26); 4 kept for a card where it is not
    if (warps == 0) warps = 8;
    if (warps != 4 && warps != 8) return ns_fail("moss attention: 4 or 8 warps a block");
    const float scale_log2 = (1.0f / sqrtf((float) D)) * 1.4426950408889634f;
    const dim3 grid((unsigned) ((S + warps * 16 - 1) / (warps * 16)), (unsigned) H);
    if (fp8 <= 0) {
        // fp8 -1: the scores accumulated in half
        const size_t smem = (size_t) 4 * BK * LDK * sizeof(__half);
        auto launch = [&](auto kern, int threads) -> int {
            NS_CU(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem));
            kern<<<grid, threads, smem, gp->q>>>((const __half*) q, (const __half*) k, (const __half*) v, (__half*) out, (int) S, (int) Skv, (int) H, (long) qs,
                                                 (long) kvs, scale_log2);
            NS_LAUNCH_CHECK();
            return 0;
        };
        if (warps == 4) return fp8 < 0 ? launch(fa_half_k<4, true>, 128) : launch(fa_half_k<4, false>, 128);
        return fp8 < 0 ? launch(fa_half_k<8, true>, 256) : launch(fa_half_k<8, false>, 256);
    }
#if NS_SILO_SM >= 89
    static Fp8Bufs b;
    const int Skv8 = (int) ((Skv + BK - 1) / BK * BK);
    const size_t n8 = (size_t) std::max(S, Skv) * H * D, nv = (size_t) H * D * Skv8, nsc = (size_t) (S + Skv) * H + H * D;
    if (b.cap8 < n8) {
        if (b.q8) { cudaFree(b.q8); cudaFree(b.k8); }
        NS_CU(cudaMalloc(&b.q8, n8));
        NS_CU(cudaMalloc(&b.k8, n8));
        b.cap8 = n8;
    }
    if (fp8 == 2 && b.capv < nv) {
        if (b.vt) cudaFree(b.vt);
        NS_CU(cudaMalloc(&b.vt, nv));
        b.capv = nv;
    }
    if (b.capsc < nsc) {
        if (b.sc) cudaFree(b.sc);
        NS_CU(cudaMalloc(&b.sc, nsc * 4));
        b.capsc = nsc;
    }
    float* qsc = b.sc;
    float* ksc = b.sc + S * H;
    float* vsc = b.sc + (S + Skv) * H;
    quant8_k<<<(unsigned) ((S * H + 7) / 8), 256, 0, gp->q>>>((const __half*) q, (int) S, (int) H, (long) qs, b.q8, qsc);
    quant8_k<<<(unsigned) ((Skv * H + 7) / 8), 256, 0, gp->q>>>((const __half*) k, (int) Skv, (int) H, (long) kvs, b.k8, ksc);
    NS_LAUNCH_CHECK();
    const bool pv8 = fp8 == 2;
    if (pv8) {
        vscale_k<<<(unsigned) (H * D), 256, 0, gp->q>>>((const __half*) v, (int) Skv, (int) H, (long) kvs, vsc);
        const long nq = (long) H * D * Skv8 / 4;
        vquant_k<<<(unsigned) ((nq + 255) / 256), 256, 0, gp->q>>>((const __half*) v, (int) Skv, Skv8, (int) H, (long) kvs, vsc, b.vt);
        const long nc = (long) H * D * (Skv8 / 32);
        vt_perm_k<<<(unsigned) ((nc + 255) / 256), 256, 0, gp->q>>>(b.vt, (int) (H * D), Skv8);
        NS_LAUNCH_CHECK();
    }
    const size_t kbytes = (size_t) 2 * BK * LD8 + 2 * BK * 4;
    const size_t vbytes = pv8 ? (size_t) 2 * D * (BK + 16) : (size_t) 2 * BK * LDK * sizeof(__half);
    const size_t smem = kbytes + vbytes;
    auto launch = [&](auto kern, int threads) -> int {
        NS_CU(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem));
        kern<<<grid, threads, smem, gp->q>>>(b.q8, qsc, b.k8, ksc, (const __half*) v, b.vt, vsc, (__half*) out, (int) S, (int) Skv, Skv8, (int) H, (long) kvs,
                                            scale_log2);
        NS_LAUNCH_CHECK();
        return 0;
    };
    if (warps == 4) return pv8 ? launch(fa_fp8_k<4, true>, 128) : launch(fa_fp8_k<4, false>, 128);
    return pv8 ? launch(fa_fp8_k<8, true>, 256) : launch(fa_fp8_k<8, false>, 256);
#else
    return ns_fail("moss attention: FP8 needs a card with FP8 tensor cores (sm_89 and up); this silo is for sm_" + std::to_string(NS_SILO_SM));
#endif
}

}  // extern "C"
