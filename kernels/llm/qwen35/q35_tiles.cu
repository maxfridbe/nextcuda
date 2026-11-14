// q35_tiles.cu: nextcuda's own products of several rows on the int8 tensor cores, beside ggml-cuda's MMQ (q35_mmq.cu)
// and the verify products (q35_vq8.cu), taken instead of them with NS_Q35_TILES=1 - not the default yet: measured
// 2026-11-13 on 512 rows of Qwen3.8-27B's matrices (17408 x 5120), ggml-cuda's time / these, the 5070: IQ2_XXS 952 /
// 1116 us, IQ3_XXS 999 / 1221, IQ3_S 1031 / 1226, IQ4_XS 909 / 1219, Q6_K 1333 / 1699, Q2_K 1597 / 2074, Q4_K 1036 /
// 1724; the 3060 1.25-1.8x; verify passes of 8 rows ~2.6x. As close to the float64 product as MMQ's or closer.
//
// On nemotronh's block reader (nh_blocks.cuh): a thread block takes 128 matrix rows and 64 or 128 input rows; a chunk
// of each row's blocks (one 256-value block; four 32s) is copied into shared memory (cp.async) while the last chunk's
// steps run, and each step decodes 128 x 64 weights from there into int8 once (no half copy, no repacking) and
// multiplies them with the input rows' int8 (read a step ahead, from their own layout: xtile_k's) by mma m16n8k32 - or
// m16n8k16 a half where a type's halves of 32 have their own scales. The integer sums are exact; each is scaled by its
// weight scale and the input row's, the offsets (K-quants' mins) taken against the input rows' sums. 1..8 rows (a
// verify pass): a warp a 16-row tile over all of K, the input rows straight from their Q8_1 blocks.
#include "ns.h"
#include "ns_internal.hpp"
#include "q35.h"
#include "nh.h"
#include "nh_blocks.cuh"

#include <cuda_fp16.h>
#include <cstdlib>
#include <string>

namespace {

using nsq::Fmt;
using nsq::Q8_1;
using nsq::Sub;

constexpr int BM = 128;        // matrix rows a thread block
constexpr int KS = 64;         // values a step (two pieces of 32)
constexpr int LD = KS + 16;    // shared row stride in bytes: 80 keeps a fragment's 8 rows on distinct banks
constexpr int NW = 8;          // warps a thread block

__device__ __forceinline__ void mma16(int (&c)[4], int a0, int a1, int b0) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a0), "r"(a1), "r"(b0));
}
__device__ __forceinline__ void mma32(int (&c)[4], int a0, int a1, int a2, int a3, int b0, int b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// an input row's piece for the scaling: its Q8_1 scale, the values' stored sum, the halves' sums (quantized, scaled)
template<int T> __device__ __forceinline__ float4 x_scales(const Q8_1& b, const int (&v)[8]) {
    const float2 ds = __half22float2(b.ds);
    int h0 = 0, h1 = 0;
    if (Fmt<T>::mins >= 2) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            h0 = __dp4a(0x01010101, v[k], h0);
            h1 = __dp4a(0x01010101, v[k + 4], h1);
        }
    }
    return make_float4(ds.x, ds.y, ds.x * (float) h0, ds.x * (float) h1);
}

// the offsets' part of an output: row piece w against input piece x
template<int T> __device__ __forceinline__ float offsets(const float4& w, const float4& x) {
    if (Fmt<T>::mins == 1) return w.z * x.y;
    if (Fmt<T>::mins == 3) return w.z * (x.z + x.w);
    if (Fmt<T>::mins == 2) return w.z * x.z + w.w * x.w;
    return 0.f;
}

// x [T][K] -> the tile products' layout: piece j of row t as 32 int8 at (j T + t) x 32, then (after all of them) its
// scales at [j T + t] as a float4 (the Q8_1 scale, the values' sum, the halves' quantized sums times the scale). A
// warp a piece of a row; the scale the largest magnitude / 127, kept as float
__global__ void xtile_k(const float* __restrict__ x, uint8_t* __restrict__ xt, int64_t T, int64_t K) {
    const int64_t w = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int64_t nsub = K / 32;
    if (w >= T * nsub) return;
    const int64_t t = w / nsub, j = w % nsub;
    const float v = x[t * K + j * 32 + lane];
    float amax = fabsf(v), sum = v;
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
    }
    const float d = amax / 127.f;
    const int qv = amax == 0.f ? 0 : (int) roundf(v / d);
    xt[(j * T + t) * 32 + lane] = (uint8_t) (int8_t) qv;
    int h = qv;
    for (int o = 8; o > 0; o >>= 1) h += __shfl_xor_sync(0xffffffffu, h, o);   // the sum over each half of 16
    const int h1 = __shfl_sync(0xffffffffu, h, 16);
    if (lane == 0) ((float4*) (xt + nsub * T * 32))[j * T + t] = make_float4(d, sum, d * (float) h, d * (float) h1);
}

// the warps' layout over the 128 x BN tile: WM along the rows, WN along the columns
template<int BN> struct Lay;
template<> struct Lay<64> { static constexpr int WM = 4, WN = 2; };
template<> struct Lay<128> { static constexpr int WM = 4, WN = 2; };

__device__ __forceinline__ void cp_async4(void* smem, const void* gmem) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"((unsigned) __cvta_generic_to_shared(smem)), "l"(gmem));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;"); }
template<int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N)); }

// a chunk: CH values of each matrix row (whole blocks: one of a 256-value type, four 32s of the 32-value ones), CB
// bytes, staged in RW words a row (a row's bytes start 0 or 2 into its first word); NP pieces of 32 a step (four
// measured slower: more registers, spills for the K-quants), within 99 KB of shared memory
template<int T> struct Chunk {
    static constexpr int CH = Fmt<T>::qk == 256 ? 256 : 128;
    static constexpr int CB = CH / Fmt<T>::qk * Fmt<T>::bytes;
    static constexpr int RW = (CB + 2 + 3) / 4;
    static constexpr int NP = 2;
    static constexpr int LDX = NP * 32 + 16;   // the int8 rows' stride: a fragment's 8 rows on distinct banks
};

template<int T, int BN> constexpr size_t xmm_smem() {
    using C = Chunk<T>;
    return (size_t) 2 * BM * C::RW * 4 + (size_t) (BM + BN) * C::LDX + (size_t) (BM + BN) * C::NP * sizeof(float4);
}

// y [ntok][n_out] = x [ntok][K] (the tile layout, xtile_k's) . W [n_out][K]^T (rows of rb bytes in T's blocks).
// Pipelined: the next chunk's block bytes are copied into shared memory (cp.async) while this chunk's steps run, each
// step decoding its weights from there, and the next step's input rows are read into registers during this one
template<int T, int BN>
__global__ void __launch_bounds__(32 * NW, 1) xmm_k(const uint8_t* __restrict__ w, size_t rb, const uint8_t* __restrict__ xt, float* __restrict__ y, int K,
                                                    int n_out, int ntok) {
    constexpr int WM = Lay<BN>::WM, WN = Lay<BN>::WN;
    constexpr int MT = BM / WM / 16, NT = BN / WN / 8;   // a warp's m16 and n8 tiles
    constexpr int spb = Fmt<T>::qk / 32;
    using C = Chunk<T>;
    constexpr int NP = C::NP, LDX = C::LDX;
    constexpr int XU = (BN * NP * 2 + 32 * NW - 1) / (32 * NW), SU = (BN * NP + 32 * NW - 1) / (32 * NW);
    extern __shared__ __align__(16) uint8_t sm[];
    uint32_t* raw = (uint32_t*) sm;                                   // [2][BM][RW]
    uint8_t* ws = sm + (size_t) 2 * BM * C::RW * 4;                   // [BM][LD]
    uint8_t* xs = ws + BM * LDX;                                      // [BN][LDX]
    float4* wsc = (float4*) (xs + BN * LDX);                          // [BM][NP]: a row's piece: the halves' scales, offsets
    float4* xsc = wsc + BM * NP;                                      // [BN][NP]: an input row's piece: scale, sum, halves' sums
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, q = lane & 3;
    const int wm = warp % WM, wn = warp / WM;
    const int row0 = blockIdx.x * BM, tok0 = blockIdx.y * BN;
    const int nsub = K / 32, nch = K / C::CH;
    const float4* xsc_g = (const float4*) (xt + (size_t) nsub * ntok * 32);
    // where row r's bytes of chunk c start in global memory, and how far into their first word
    auto src = [&](int r, int c) { return w + (size_t) (row0 + r) * rb + (size_t) c * C::CB; };
    auto fetch = [&](int c, int b) {
        for (int i = tid; i < BM * C::RW; i += 32 * NW) {
            const int r = i / C::RW, k = i % C::RW;
            if (row0 + r >= n_out) continue;
            const uint8_t* p = src(r, c);
            const int mis = (int) ((uintptr_t) p & 3);
            if (4 * k < mis + C::CB) cp_async4(raw + ((size_t) b * BM + r) * C::RW + k, p - mis + 4 * k);
        }
        cp_commit();
    };
    // the next step's input rows, in registers: 16-byte chunks i (piece, row, half) and scales (row, piece)
    int4 xr[XU];
    float4 xsr[SU];
    auto xload = [&](int k0) {
#pragma unroll
        for (int u = 0; u < XU; ++u) {
            const int i = tid + u * 32 * NW;
            xr[u] = make_int4(0, 0, 0, 0);
            if (i < BN * NP * 2) {
                const int c = i & 1, t = (i >> 1) % BN, p = (i >> 1) / BN, j = k0 + p;
                if (tok0 + t < ntok && j < nsub) xr[u] = *(const int4*) (xt + ((size_t) j * ntok + tok0 + t) * 32 + c * 16);
            }
        }
#pragma unroll
        for (int u = 0; u < SU; ++u) {
            const int i = tid + u * 32 * NW;
            xsr[u] = make_float4(0.f, 0.f, 0.f, 0.f);
            if (i < BN * NP) {
                const int t = i % BN, p = i / BN, j = k0 + p;
                if (tok0 + t < ntok && j < nsub) xsr[u] = xsc_g[(size_t) j * ntok + tok0 + t];
            }
        }
    };
    float acc[MT][NT][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int k = 0; k < 4; ++k) acc[i][j][k] = 0.f;

    fetch(0, 0);
    xload(0);
    for (int ch = 0; ch < nch; ++ch) {
        if (ch + 1 < nch) {
            fetch(ch + 1, (ch + 1) & 1);
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();
#pragma unroll 1
        for (int st = 0; st < C::CH / (32 * NP); ++st) {
            const int k0 = ch * (C::CH / 32) + st * NP;
            // the weights, from the staged bytes: a thread a (row, piece), NP / 2 of them
#pragma unroll
            for (int u = 0; u < NP / 2; ++u) {
                const int item = tid + u * 32 * NW, r = item / NP, p = item % NP, jc = st * NP + p;
                Sub sb;
                if (row0 + r < n_out) {
                    const uint8_t* rowb = (const uint8_t*) (raw + ((size_t) (ch & 1) * BM + r) * C::RW) + ((uintptr_t) src(r, ch) & 3);
                    nsq::sub<T>(rowb + (jc / spb) * Fmt<T>::bytes, jc % spb, sb);
                } else {
#pragma unroll
                    for (int i = 0; i < 8; ++i) sb.v[i] = 0;
                    sb.s0 = sb.s1 = sb.m0 = sb.m1 = 0.f;
                }
                int4* d = (int4*) (ws + r * LDX + p * 32);
                d[0] = make_int4(sb.v[0], sb.v[1], sb.v[2], sb.v[3]);
                d[1] = make_int4(sb.v[4], sb.v[5], sb.v[6], sb.v[7]);
                wsc[r * NP + p] = make_float4(sb.s0, sb.s1, sb.m0, sb.m1);
            }
            // the input rows read during the last step, to shared memory; then the next step's
#pragma unroll
            for (int u = 0; u < XU; ++u) {
                const int i = tid + u * 32 * NW;
                if (i < BN * NP * 2) {
                    const int c = i & 1, t = (i >> 1) % BN, p = (i >> 1) / BN;
                    *(int4*) (xs + t * LDX + p * 32 + c * 16) = xr[u];
                }
            }
#pragma unroll
            for (int u = 0; u < SU; ++u) {
                const int i = tid + u * 32 * NW;
                if (i < BN * NP) xsc[(i % BN) * NP + i / BN] = xsr[u];
            }
            if (k0 + NP < nsub) xload(k0 + NP);
            __syncthreads();
    #pragma unroll
            for (int p = 0; p < NP; ++p) {
                if (nsq::Halves<T>::v) {
                    // the halves' scales differ: a k16 product a half, each scaled on its own
    #pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int ko = p * 32 + h * 16 + q * 4;
                        int a[MT][2], b[NT];
    #pragma unroll
                        for (int mt = 0; mt < MT; ++mt) {
                            const int r = (wm * MT + mt) * 16 + g;
                            a[mt][0] = *(const int*) (ws + r * LDX + ko);
                            a[mt][1] = *(const int*) (ws + (r + 8) * LDX + ko);
                        }
    #pragma unroll
                        for (int nt = 0; nt < NT; ++nt) b[nt] = *(const int*) (xs + ((wn * NT + nt) * 8 + g) * LDX + ko);
    #pragma unroll
                        for (int mt = 0; mt < MT; ++mt) {
                            const int r = (wm * MT + mt) * 16 + g;
                            const float s0 = h ? wsc[r * NP + p].y : wsc[r * NP + p].x, s1 = h ? wsc[(r + 8) * NP + p].y : wsc[(r + 8) * NP + p].x;
    #pragma unroll
                            for (int nt = 0; nt < NT; ++nt) {
                                int c[4] = {0, 0, 0, 0};
                                mma16(c, a[mt][0], a[mt][1], b[nt]);
                                const int t = (wn * NT + nt) * 8 + q * 2;
                                const float d0 = xsc[t * NP + p].x, d1 = xsc[(t + 1) * NP + p].x;
                                acc[mt][nt][0] += (d0 * s0) * (float) c[0];
                                acc[mt][nt][1] += (d1 * s0) * (float) c[1];
                                acc[mt][nt][2] += (d0 * s1) * (float) c[2];
                                acc[mt][nt][3] += (d1 * s1) * (float) c[3];
                            }
                        }
                    }
                } else {
                    // one scale for the 32: one k32 product, scaled once
                    const int ko = p * 32 + q * 4;
                    int a[MT][4], b[NT][2];
    #pragma unroll
                    for (int mt = 0; mt < MT; ++mt) {
                        const int r = (wm * MT + mt) * 16 + g;
                        a[mt][0] = *(const int*) (ws + r * LDX + ko);
                        a[mt][1] = *(const int*) (ws + (r + 8) * LDX + ko);
                        a[mt][2] = *(const int*) (ws + r * LDX + ko + 16);
                        a[mt][3] = *(const int*) (ws + (r + 8) * LDX + ko + 16);
                    }
    #pragma unroll
                    for (int nt = 0; nt < NT; ++nt) {
                        const uint8_t* xr = xs + ((wn * NT + nt) * 8 + g) * LDX + ko;
                        b[nt][0] = *(const int*) xr;
                        b[nt][1] = *(const int*) (xr + 16);
                    }
    #pragma unroll
                    for (int mt = 0; mt < MT; ++mt) {
                        const int r = (wm * MT + mt) * 16 + g;
                        const float s0 = wsc[r * NP + p].x, s1 = wsc[(r + 8) * NP + p].x;
    #pragma unroll
                        for (int nt = 0; nt < NT; ++nt) {
                            int c[4] = {0, 0, 0, 0};
                            mma32(c, a[mt][0], a[mt][1], a[mt][2], a[mt][3], b[nt][0], b[nt][1]);
                            const int t = (wn * NT + nt) * 8 + q * 2;
                            const float d0 = xsc[t * NP + p].x, d1 = xsc[(t + 1) * NP + p].x;
                            acc[mt][nt][0] += (d0 * s0) * (float) c[0];
                            acc[mt][nt][1] += (d1 * s0) * (float) c[1];
                            acc[mt][nt][2] += (d0 * s1) * (float) c[2];
                            acc[mt][nt][3] += (d1 * s1) * (float) c[3];
                        }
                    }
                }
                // the offsets (K-quants' mins, IQ1_S's delta) against the input rows' sums
                if (Fmt<T>::mins != 0) {
    #pragma unroll
                    for (int mt = 0; mt < MT; ++mt) {
                        const int r = (wm * MT + mt) * 16 + g;
                        const float4 w0 = wsc[r * NP + p], w1 = wsc[(r + 8) * NP + p];
    #pragma unroll
                        for (int nt = 0; nt < NT; ++nt) {
                            const int t = (wn * NT + nt) * 8 + q * 2;
                            const float4 x0 = xsc[t * NP + p], x1 = xsc[(t + 1) * NP + p];
                            acc[mt][nt][0] += offsets<T>(w0, x0);
                            acc[mt][nt][1] += offsets<T>(w0, x1);
                            acc[mt][nt][2] += offsets<T>(w1, x0);
                            acc[mt][nt][3] += offsets<T>(w1, x1);
                        }
                    }
                }
            }
            __syncthreads();
        }
    }
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const int r = row0 + (wm * MT + mt) * 16 + g, t = tok0 + (wn * NT + nt) * 8 + q * 2;
            if (r < n_out) {
                if (t < ntok) y[(size_t) t * n_out + r] = acc[mt][nt][0];
                if (t + 1 < ntok) y[(size_t) (t + 1) * n_out + r] = acc[mt][nt][1];
            }
            if (r + 8 < n_out) {
                if (t < ntok) y[(size_t) t * n_out + r + 8] = acc[mt][nt][2];
                if (t + 1 < ntok) y[(size_t) (t + 1) * n_out + r + 8] = acc[mt][nt][3];
            }
        }
}


// 1..8 input rows (a verify pass): a warp takes 16 matrix rows and all of K, two pieces of 32 a step - each lane
// decodes one (row, piece) into the warp's own shared rows, the input rows' int8 come straight from their Q8_1 blocks
// as the B fragments - no block-wide syncs; 4 warps a thread block
constexpr int VW = 4;
template<int T>
__global__ void __launch_bounds__(32 * VW) vq_k(const uint8_t* __restrict__ w, size_t rb, const Q8_1* __restrict__ x, float* __restrict__ y, int K, int n_out,
                                                int ntok) {
    __shared__ __align__(16) uint8_t wsa[VW][16 * LD];
    __shared__ float4 wsca[VW][16][2];
    __shared__ float4 xsca[VW][8][2];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, q = lane & 3;
    uint8_t* ws = wsa[warp];
    float4 (*wsc)[2] = wsca[warp];
    float4 (*xsc)[2] = xsca[warp];
    const int row0 = (blockIdx.x * VW + warp) * 16;
    if (row0 >= n_out) return;
    constexpr int spb = Fmt<T>::qk / 32;
    const int nsub = K / 32;
    const bool has0 = 2 * q < ntok, has1 = 2 * q + 1 < ntok, hasg = g < ntok;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int k0 = 0; k0 < nsub; k0 += 2) {
        {
            const int r = lane >> 1, p = lane & 1, j = k0 + p;
            Sub sb;
            if (row0 + r < n_out && j < nsub) {
                nsq::sub<T>(w + (size_t) (row0 + r) * rb + (size_t) (j / spb) * Fmt<T>::bytes, j % spb, sb);
            } else {
#pragma unroll
                for (int i = 0; i < 8; ++i) sb.v[i] = 0;
                sb.s0 = sb.s1 = sb.m0 = sb.m1 = 0.f;
            }
            int4* d = (int4*) (ws + r * LD + p * 32);
            d[0] = make_int4(sb.v[0], sb.v[1], sb.v[2], sb.v[3]);
            d[1] = make_int4(sb.v[4], sb.v[5], sb.v[6], sb.v[7]);
            wsc[r][p] = make_float4(sb.s0, sb.s1, sb.m0, sb.m1);
        }
        if (lane < 16) {
            const int t = lane >> 1, p = lane & 1, j = k0 + p;
            float4 sc = make_float4(0.f, 0.f, 0.f, 0.f);
            if (t < ntok && j < nsub) {
                const Q8_1& b = x[(size_t) t * nsub + j];
                int v[8];
#pragma unroll
                for (int k = 0; k < 8; ++k) v[k] = ((const int*) b.qs)[k];
                sc = x_scales<T>(b, v);
            }
            xsc[t][p] = sc;
        }
        __syncwarp();
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            const int j = k0 + p;
            if (j >= nsub) break;
            const int* xg = (const int*) x[(size_t) (hasg ? g : 0) * nsub + j].qs;
            const float d0 = xsc[2 * q][p].x, d1 = xsc[2 * q + 1][p].x;
            if (nsq::Halves<T>::v) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int ko = p * 32 + h * 16 + q * 4;
                    int c[4] = {0, 0, 0, 0};
                    mma16(c, *(const int*) (ws + g * LD + ko), *(const int*) (ws + (g + 8) * LD + ko), hasg ? xg[h * 4 + q] : 0);
                    const float s0 = h ? wsc[g][p].y : wsc[g][p].x, s1 = h ? wsc[g + 8][p].y : wsc[g + 8][p].x;
                    acc[0] += (d0 * s0) * (float) c[0];
                    acc[1] += (d1 * s0) * (float) c[1];
                    acc[2] += (d0 * s1) * (float) c[2];
                    acc[3] += (d1 * s1) * (float) c[3];
                }
            } else {
                const int ko = p * 32 + q * 4;
                int c[4] = {0, 0, 0, 0};
                mma32(c, *(const int*) (ws + g * LD + ko), *(const int*) (ws + (g + 8) * LD + ko), *(const int*) (ws + g * LD + ko + 16),
                      *(const int*) (ws + (g + 8) * LD + ko + 16), hasg ? xg[q] : 0, hasg ? xg[4 + q] : 0);
                const float s0 = wsc[g][p].x, s1 = wsc[g + 8][p].x;
                acc[0] += (d0 * s0) * (float) c[0];
                acc[1] += (d1 * s0) * (float) c[1];
                acc[2] += (d0 * s1) * (float) c[2];
                acc[3] += (d1 * s1) * (float) c[3];
            }
            if (Fmt<T>::mins != 0) {
                acc[0] += offsets<T>(wsc[g][p], xsc[2 * q][p]);
                acc[1] += offsets<T>(wsc[g][p], xsc[2 * q + 1][p]);
                acc[2] += offsets<T>(wsc[g + 8][p], xsc[2 * q][p]);
                acc[3] += offsets<T>(wsc[g + 8][p], xsc[2 * q + 1][p]);
            }
        }
        __syncwarp();
    }
    const int r = row0 + g, t = 2 * q;
    if (r < n_out) {
        if (has0) y[(size_t) t * n_out + r] = acc[0];
        if (has1) y[(size_t) (t + 1) * n_out + r] = acc[1];
    }
    if (r + 8 < n_out) {
        if (has0) y[(size_t) t * n_out + r + 8] = acc[2];
        if (has1) y[(size_t) (t + 1) * n_out + r + 8] = acc[3];
    }
}

template<int T>
void vq_launch(const void* w, int64_t n_out, int64_t K, const Q8_1* x, int64_t ntok, float* y, cudaStream_t s) {
    const size_t rb = (size_t) (K / Fmt<T>::qk) * Fmt<T>::bytes;
    vq_k<T><<<(unsigned) ((n_out + 16 * VW - 1) / (16 * VW)), 32 * VW, 0, s>>>((const uint8_t*) w, rb, x, y, (int) K, (int) n_out, (int) ntok);
}

template<int T, int BN>
void xmm_go(const uint8_t* W, size_t rb, int64_t n_out, int64_t K, const uint8_t* x, int64_t ntok, float* y, cudaStream_t s) {
    constexpr size_t smem = xmm_smem<T, BN>();
    cudaFuncSetAttribute(xmm_k<T, BN>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
    xmm_k<T, BN><<<dim3((unsigned) ((n_out + BM - 1) / BM), (unsigned) ((ntok + BN - 1) / BN)), 32 * NW, smem, s>>>(W, rb, x, y, (int) K, (int) n_out, (int) ntok);
}

template<int T>
void xmm_launch(const void* w, int64_t n_out, int64_t K, const uint8_t* x, int64_t ntok, float* y, cudaStream_t s) {
    const size_t rb = (size_t) (K / Fmt<T>::qk) * Fmt<T>::bytes;
    if (ntok <= 64) xmm_go<T, 64>((const uint8_t*) w, rb, n_out, K, x, ntok, y, s);
    else xmm_go<T, 128>((const uint8_t*) w, rb, n_out, K, x, ntok, y, s);
}

// the types the tile products are built for (Qwen3.8-27B GSQ-RCO's file holds them; the rest go through the decode
// products or are expanded to half by the engine)
bool xmm_type(int t) {
    switch (t) {
        case nsq::Q2_K: case nsq::Q4_K: case nsq::Q6_K: case nsq::IQ2_XXS: case nsq::IQ2_XS: case nsq::IQ2_S: case nsq::IQ3_XXS:
        case nsq::IQ3_S: case nsq::IQ4_XS: case nsq::Q8_0:
            return true;
        default:
            return false;
    }
}

// the tile products (x in the tile layout) or, `rows`, the per-warp ones for 1..8 rows (x the Q8_1 rows)
int xmm(ns_gpu* g, int type, const void* w, int64_t n_out, int64_t K, const void* xq, int64_t ntok, float* y, bool rows, const char* who) {
    if (!xmm_type(type)) return ns_fail(std::string(who) + ": type " + std::to_string(type) + " has no tile product here");
    // whole chunks of the tile products (256 values; 128 for Q8_0), whole pairs of 32s for the per-warp ones
    if (K % (rows ? (nsq::fmt_of(type).x == 256 ? 256 : 64) : (nsq::fmt_of(type).x == 256 ? 256 : 128)) != 0)
        return ns_fail(std::string(who) + ": K not whole chunks of the products");
    if (ntok <= 0 || n_out <= 0) return 0;
    switch (type) {
#define Q35_XMM(T) case T: if (rows) vq_launch<T>(w, n_out, K, (const Q8_1*) xq, ntok, y, g->q); else xmm_launch<T>(w, n_out, K, (const uint8_t*) xq, ntok, y, g->q); break;
        Q35_XMM(nsq::Q2_K) Q35_XMM(nsq::Q4_K) Q35_XMM(nsq::Q6_K) Q35_XMM(nsq::IQ2_XXS) Q35_XMM(nsq::IQ2_XS) Q35_XMM(nsq::IQ2_S)
        Q35_XMM(nsq::IQ3_XXS) Q35_XMM(nsq::IQ3_S) Q35_XMM(nsq::IQ4_XS) Q35_XMM(nsq::Q8_0)
#undef Q35_XMM
    }
    return 0;
}

}  // namespace

extern "C" {

size_t ns_q35_tiles_q8_bytes(int64_t K, int64_t T) { return (size_t) (K / 32) * (size_t) T * 48; }

int ns_q35_tiles_quantize(ns_gpu* g, int type, const float* x, int64_t T, int64_t K, void* xq) {
    NS_TRY
    NS_CU(cudaSetDevice(g->dev));
    if (!xmm_type(type) || K % 32 != 0) return ns_fail("ns_q35_tiles_quantize: a type without tile products, or K not a multiple of 32");
    const int64_t n = T * (K / 32) * 32;
    if (n > 0) xtile_k<<<(unsigned) ((n + 255) / 256), 256, 0, g->q>>>(x, (uint8_t*) xq, T, K);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_q35_tiles_mmq(ns_gpu* g, int type, const void* w, int64_t n_out, int64_t K, const void* xq, int64_t T, float* y) {
    NS_TRY
    NS_CU(cudaSetDevice(g->dev));
    if (xmm(g, type, w, n_out, K, xq, T, y, false, "ns_q35_tiles_mmq") != 0) return -1;
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_q35_tiles_vq8(ns_gpu* g, int type, const void* w, const void* xq, float* y, int64_t n_in, int64_t n_out, int64_t ncols) {
    NS_TRY
    NS_CU(cudaSetDevice(g->dev));
    if (ncols < 1 || ncols > 8) return ns_fail("ns_q35_tiles_vq8: 1..8 input rows");
    if (xmm(g, type, w, n_out, n_in, xq, ncols, y, true, "ns_q35_tiles_vq8") != 0) return -1;
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

}  // extern "C"
