// q35_tiles.cu: nextcuda's own products of several rows on the int8 tensor cores, beside ggml-cuda's MMQ (q35_mmq.cu)
// and the verify products (q35_vq8.cu), taken instead of them with NS_Q35_TILES=1 - not the default: measured
// 2026-11-14 on Qwen3.8-27B GSQ-RCO's matrices (ffn gate / down, attn qkv / k), the sum of their times, these / MMQ's:
// 512 rows 0.955 on the 5070, 1.009 on the 3060 (Q6_K up to 1.19 there, Q2_K 0.83); 128 rows 1.036 / 0.984; 47 rows
// 1.42 / 1.33 (16 rows 1.16 / 1.35). Through the engine (5070 + 3060) a 3119-token prompt 6 % faster, 16.6K 2.6 %, the
// same tokens on the 3k prompt; the story's KL against MMQ's logits 0.00145, perplexity over 3000 tokens +0.25 %.
//
// The 256-value block types go through the chunk products (namespace k2, below): x quantized a scale a 128, the
// sums integer within it, producer warps decoding while others multiply. Q8_0 (the DSpine draft) through the older
// tile products: a thread block takes 128 matrix rows and 64 or 128 input rows; a chunk of each row's blocks is copied
// into shared memory (cp.async) while the last chunk's steps run, each step decoding 128 x 64 weights into int8 and
// multiplying them with the input rows' int8 by mma m16n8k32, each 32's sum scaled. 1..8 rows (a verify pass): a warp
// a 16-row tile over all of K, the input rows straight from their Q8_1 blocks.
#include "ns.h"
#include "ns_internal.hpp"
#include "q35.h"
#include "nh.h"
#include "nh_blocks.cuh"

#include <cuda_fp16.h>
#include <algorithm>
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


// ---- the chunk products (x's layout 2): the 256-value block types, on the int8 tensor cores. x comes quantized G
// values (128, or a chunk's 256) at a time with one float scale, so within a group the products are summed as
// integers: each mma's sums times the piece's integer scale (the format's own: IQ3_S's 1 + 2s, IQ4_XS's ls - 32, the
// K-quants' 4- / 6-bit scales, the i-quants' 2s + 1 with the eighth or quarter in d), the K-quants' mins against x's
// sums of 16 by dp2a, and the group's sums scaled once (the block's d x x's scale) - not every 32 (or 16) values'
// sums each converted and scaled. A thread block takes 64 matrix rows and BN = 8 NT WN input rows; 4 WN warps multiply
// (4 along the rows, 16 each: their A fragments by ldmatrix, reused over the BN; WN along the inputs) while PW
// producer warps copy the blocks (cp.async, a chunk ahead) and x (two steps of 128 ahead) into shared memory and decode
// each step's 64 x 128 weights into int8 there - the multipliers never wait on a decode, only on the step being
// ready (named barriers a stage: full, the producers arrive and the multipliers wait; empty, the other way)
namespace k2 {
constexpr int CH = 256, SV = 128;        // a chunk's values (one block a row); a step's
constexpr int BM = 64;
constexpr int TP = 128;                  // x's rows padded to a multiple of this (the products read whole tiles)
inline int64_t tpad(int64_t t) { return (t + TP - 1) / TP * TP; }

// x [T][K] -> layout 2: per chunk c the int8 values [c][Tp][256], a scale every G values [c][256 / G][Tp] (float),
// each 16 values' sum of the int8 [c][Tp][16] (int16: the K-quants' mins meet them). Padded rows are zero
inline size_t xbytes(int64_t K, int64_t T, int G) { return (size_t) (K / CH) * (size_t) tpad(T) * (CH + 4 * (CH / G) + 32); }

// a warp a (chunk, row): a lane 8 values
template<int G>
__global__ void xq_k(const float* __restrict__ x, uint8_t* __restrict__ xq, int64_t T, int64_t tp, int64_t K) {
    const int64_t wid = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int64_t nch = K / CH;
    if (wid >= nch * tp) return;
    const int64_t c = wid / tp, t = wid % tp;
    float v[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = t < T ? x[t * K + c * CH + lane * 8 + i] : 0.f;
    float am = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) am = fmaxf(am, fabsf(v[i]));
#pragma unroll
    for (int o = 1; o < G / 8; o <<= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
    const float d = am / 127.f;
    uint32_t w0 = 0, w1 = 0;
    int sum = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int qv = am == 0.f ? 0 : (int) roundf(v[i] / d);
        sum += qv;
        if (i < 4) w0 |= (uint32_t) (qv & 255) << (8 * i);
        else w1 |= (uint32_t) (qv & 255) << (8 * (i - 4));
    }
    *(uint2*) (xq + (c * tp + t) * CH + lane * 8) = make_uint2(w0, w1);
    float* D = (float*) (xq + nch * tp * CH);
    int16_t* S = (int16_t*) (D + nch * (CH / G) * tp);
    if (lane % (G / 8) == 0) D[(c * (CH / G) + lane / (G / 8)) * tp + t] = d;
    sum += __shfl_xor_sync(0xffffffffu, sum, 1);
    if ((lane & 1) == 0) S[(c * tp + t) * 16 + lane / 2] = (int16_t) sum;
}

// a piece's integer scales (per half of 16: s0, s1; the mins m0, m1) and the block's d and dmin, value = d s q - dmin m
template<int T> __device__ __forceinline__ void iscale(const uint8_t* b, int s, int& s0, int& s1, int& m0, int& m1, float& d, float& dm);
template<> __device__ __forceinline__ void iscale<nsq::Q2_K>(const uint8_t* b, int s, int& s0, int& s1, int& m0, int& m1, float& d, float& dm) {
    const int a = b[2 * s], c = b[2 * s + 1];
    s0 = a & 15; s1 = c & 15; m0 = a >> 4; m1 = c >> 4;
    d = nsq::half_at(b + 80); dm = nsq::half_at(b + 82);
}
template<> __device__ __forceinline__ void iscale<nsq::Q4_K>(const uint8_t* b, int s, int& s0, int& s1, int& m0, int& m1, float& d, float& dm) {
    int sc, m;
    nsq::scale_min_k4(b + 4, s, sc, m);
    s0 = s1 = sc; m0 = m1 = m;
    d = nsq::half_at(b); dm = nsq::half_at(b + 2);
}
template<> __device__ __forceinline__ void iscale<nsq::Q6_K>(const uint8_t* b, int s, int& s0, int& s1, int& m0, int& m1, float& d, float& dm) {
    s0 = (int8_t) b[192 + 2 * s]; s1 = (int8_t) b[193 + 2 * s]; m0 = m1 = 0;
    d = nsq::half_at(b + 208); dm = 0.f;
}
// the i-quants' (0.5 + s) / 4 and / 2 as (2 s + 1) / 8 and / 4: the eighth or quarter in d
template<> __device__ __forceinline__ void iscale<nsq::IQ2_XXS>(const uint8_t* b, int s, int& s0, int& s1, int& m0, int& m1, float& d, float& dm) {
    s0 = s1 = 2 * (int) (nsq::u32_at(b + 6 + 8 * s) >> 28) + 1; m0 = m1 = 0;
    d = 0.125f * nsq::half_at(b); dm = 0.f;
}
template<> __device__ __forceinline__ void iscale<nsq::IQ2_XS>(const uint8_t* b, int s, int& s0, int& s1, int& m0, int& m1, float& d, float& dm) {
    const int sc = b[66 + s];
    s0 = 2 * (sc & 15) + 1; s1 = 2 * (sc >> 4) + 1; m0 = m1 = 0;
    d = 0.125f * nsq::half_at(b); dm = 0.f;
}
template<> __device__ __forceinline__ void iscale<nsq::IQ2_S>(const uint8_t* b, int s, int& s0, int& s1, int& m0, int& m1, float& d, float& dm) {
    const int sc = b[74 + s];
    s0 = 2 * (sc & 15) + 1; s1 = 2 * (sc >> 4) + 1; m0 = m1 = 0;
    d = 0.125f * nsq::half_at(b); dm = 0.f;
}
template<> __device__ __forceinline__ void iscale<nsq::IQ3_XXS>(const uint8_t* b, int s, int& s0, int& s1, int& m0, int& m1, float& d, float& dm) {
    s0 = s1 = 2 * (int) (nsq::u32_at(b + 66 + 4 * s) >> 28) + 1; m0 = m1 = 0;
    d = 0.25f * nsq::half_at(b); dm = 0.f;
}
template<> __device__ __forceinline__ void iscale<nsq::IQ3_S>(const uint8_t* b, int s, int& s0, int& s1, int& m0, int& m1, float& d, float& dm) {
    s0 = s1 = 1 + 2 * ((b[106 + s / 2] >> (4 * (s & 1))) & 15); m0 = m1 = 0;
    d = nsq::half_at(b); dm = 0.f;
}
template<> __device__ __forceinline__ void iscale<nsq::IQ4_XS>(const uint8_t* b, int s, int& s0, int& s1, int& m0, int& m1, float& d, float& dm) {
    s0 = s1 = (((b[4 + s / 2] >> (4 * (s & 1))) & 15) | (int) (((nsq::u16_at(b + 2) >> (2 * s)) & 3) << 4)) - 32; m0 = m1 = 0;
    d = nsq::half_at(b); dm = 0.f;
}

// the grid types' codebooks, copied into shared memory for the decode's lookups (spread over banks there, where the
// L1 serves a warp's scattered reads line by line); the readers as nh_blocks.cuh's, with the table passed
template<int T> struct Grid { static constexpr int N = 0; using E = uint32_t; static __device__ const E* src() { return nullptr; } };
template<> struct Grid<nsq::IQ3_S> { static constexpr int N = 512; using E = uint32_t; static __device__ const E* src() { return nsq_iq3s_grid; } };
template<> struct Grid<nsq::IQ3_XXS> { static constexpr int N = 256; using E = uint32_t; static __device__ const E* src() { return nsq_iq3xxs_grid; } };
template<> struct Grid<nsq::IQ2_XXS> { static constexpr int N = 256; using E = uint64_t; static __device__ const E* src() { return nsq_iq2xxs_grid; } };
template<> struct Grid<nsq::IQ2_XS> { static constexpr int N = 512; using E = uint64_t; static __device__ const E* src() { return nsq_iq2xs_grid; } };
template<> struct Grid<nsq::IQ2_S> { static constexpr int N = 1024; using E = uint64_t; static __device__ const E* src() { return nsq_iq2s_grid; } };

template<int T> __device__ __forceinline__ void gsub(const uint8_t* b, int s, Sub& o, const typename Grid<T>::E*) { nsq::sub<T>(b, s, o); }
template<> __device__ __forceinline__ void gsub<nsq::IQ3_S>(const uint8_t* b, int s, Sub& o, const uint32_t* grid) {
    using namespace nsq;
    const int qh = b[66 + s];
    const uint32_t q[2] = {u32_at(b + 2 + 8 * s), u32_at(b + 6 + 8 * s)}, signs = u32_at(b + 74 + 4 * s);
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const uint32_t sg = signs >> (8 * l);
        const uint32_t w = q[l / 2] >> (16 * (l & 1));
        const int i1 = (w & 255) | ((qh << (8 - 2 * l)) & 256);
        const int i2 = ((w >> 8) & 255) | ((qh << (7 - 2 * l)) & 256);
        o.v[2 * l] = signed4(grid[i1], sg);
        o.v[2 * l + 1] = signed4(grid[i2], sg >> 4);
    }
}
template<> __device__ __forceinline__ void gsub<nsq::IQ3_XXS>(const uint8_t* b, int s, Sub& o, const uint32_t* grid) {
    using namespace nsq;
    const uint32_t aux = u32_at(b + 66 + 4 * s);
    const uint32_t q[2] = {u32_at(b + 2 + 8 * s), u32_at(b + 6 + 8 * s)};
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const uint32_t sg = signs7(aux >> (7 * l));
        const uint32_t w = q[l / 2] >> (16 * (l & 1));
        o.v[2 * l] = signed4(grid[w & 255], sg);
        o.v[2 * l + 1] = signed4(grid[(w >> 8) & 255], sg >> 4);
    }
}
template<> __device__ __forceinline__ void gsub<nsq::IQ2_XXS>(const uint8_t* b, int s, Sub& o, const uint64_t* grid) {
    using namespace nsq;
    const uint32_t idx = u32_at(b + 2 + 8 * s), aux = u32_at(b + 6 + 8 * s);
#pragma unroll
    for (int l = 0; l < 4; ++l) grid8(grid[(idx >> (8 * l)) & 255], signs7(aux >> (7 * l)), o.v[2 * l], o.v[2 * l + 1]);
}
template<> __device__ __forceinline__ void gsub<nsq::IQ2_XS>(const uint8_t* b, int s, Sub& o, const uint64_t* grid) {
    using namespace nsq;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const uint32_t q = u16_at(b + 2 + 8 * s + 2 * l);
        grid8(grid[q & 511], signs7(q >> 9), o.v[2 * l], o.v[2 * l + 1]);
    }
}
template<> __device__ __forceinline__ void gsub<nsq::IQ2_S>(const uint8_t* b, int s, Sub& o, const uint64_t* grid) {
    using namespace nsq;
    const int qh = b[66 + s];
    const uint32_t qs = u32_at(b + 2 + 4 * s), sg = u32_at(b + 34 + 4 * s);
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const int i = ((qs >> (8 * l)) & 255) | ((qh << (8 - 2 * l)) & 0x300);
        grid8(grid[i], sg >> (8 * l), o.v[2 * l], o.v[2 * l + 1]);
    }
}

__device__ __forceinline__ void ldsm4(const void* p, int& r0, int& r1, int& r2, int& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"((unsigned) __cvta_generic_to_shared(p)));
}
__device__ __forceinline__ void cp16(void* smem, const void* gmem) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"((unsigned) __cvta_generic_to_shared(smem)), "l"(gmem));
}
__device__ __forceinline__ void imma32(int (&c)[4], int a0, int a1, int a2, int a3, int b0, int b1) {
    c[0] = c[1] = c[2] = c[3] = 0;
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void imma16(int (&c)[4], int a0, int a1, int b0) {
    c[0] = c[1] = c[2] = c[3] = 0;
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a0), "r"(a1), "r"(b0));
}
__device__ __forceinline__ void bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void bar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(n) : "memory"); }

// a step's tiles' rows are 128 bytes, their 16-byte chunks swizzled by the row (chunk k of row r at k ^ (r & 7)): the
// 8 rows an ldmatrix reads, and the copies' rows, on distinct banks without padding
__device__ __forceinline__ int swz(int r, int k) { return r * SV + ((k ^ (r & 7)) << 4); }

// the staged blocks: a row's block copied as the 16-byte units it overlaps (RU of them: the block 0 .. 14 bytes into
// the first, blocks being 2-byte aligned), RW words a row
template<int T> struct Raw { static constexpr int CB = Fmt<T>::bytes, RU = (CB + 14 + 15) / 16, RW = 4 * RU; };

// the weights' stages (two): the step's int8 [BM][128], their scales and mins a 16 [BM][8] (int8), d and dmin [BM];
// x's (three: copied two steps ahead): int8 [BN][128], the scale [BN], (the K-quants') sums of 16 [BN][8]
struct WSt { static constexpr int W = 0, WS = W + BM * SV, WM = WS + BM * 8, WD = WM + BM * 8, BYTES = WD + BM * 8; };
template<int BN, bool SUMS> struct XSt { static constexpr int X = 0, XD = X + BN * SV, XS = XD + BN * 4, BYTES = XS + (SUMS ? BN * 16 : 0); };
template<int T, int BN> constexpr size_t smem() {
    return (size_t) 2 * WSt::BYTES + (size_t) 3 * XSt<BN, Fmt<T>::mins != 0>::BYTES + (size_t) 2 * BM * Raw<T>::RW * 4 +
           (size_t) Grid<T>::N * sizeof(typename Grid<T>::E);
}

template<int T, int NT, int WN, int PW, int G>
__global__ void __launch_bounds__(128 * WN + 32 * PW, 1) kmm_k(const uint8_t* __restrict__ w, size_t rb, const uint8_t* __restrict__ xq, float* __restrict__ y,
                                                             int K, int n_out, int ntok, int tp, size_t ystride) {
    constexpr int BN = 8 * NT * WN, CT = 128 * WN, NTH = 32 * PW, ALL = CT + NTH;
    constexpr bool HV = nsq::Halves<T>::v;
    constexpr int MN = Fmt<T>::mins;
    using XS_ = XSt<BN, MN != 0>;
    using R = Raw<T>;
    using GE = typename Grid<T>::E;
    extern __shared__ __align__(16) uint8_t sm[];
    uint8_t* xs0 = sm + 2 * WSt::BYTES;
    uint32_t* raw = (uint32_t*) (xs0 + 3 * XS_::BYTES);   // [2][BM][RW]
    GE* grid = (GE*) (raw + 2 * BM * R::RW);
    auto wst = [&](int s) { return sm + (s & 1) * WSt::BYTES; };
    auto xst = [&](int s) { return xs0 + (s % 3) * XS_::BYTES; };
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, q = lane & 3;
    const int wm = warp & 3, wn = warp >> 2;
    const int tid = (int) threadIdx.x - CT;   // a producer's index (negative for the multipliers)
    // the input tiles fastest: the thread blocks running at once share their rows' blocks (read once into L2)
    const int row0 = blockIdx.y * BM, tok0 = blockIdx.x * BN, R0 = wm * 16, T0 = wn * 8 * NT;
    // split K (blockIdx.z of gridDim.z: a share of the chunks, its sums to its own output, added up after)
    const int nall = K / CH, c0 = (int) (blockIdx.z * nall / gridDim.z), nch = (int) ((blockIdx.z + 1) * nall / gridDim.z) - c0, nst = 2 * nch;
    y += blockIdx.z * ystride;
    const float* xD = (const float*) (xq + (size_t) nall * tp * CH);
    const int16_t* xS = (const int16_t*) (xD + (size_t) nall * (CH / G) * tp);
    auto src = [&](int r, int c) { return w + (size_t) min(row0 + r, n_out - 1) * rb + (size_t) (c0 + c) * R::CB; };

    if (tid >= 0) {
        // ---- the producers
        // chunk c's block bytes into raw buffer c & 1: whole 16-byte units, the last one's words past the block not read
        auto rfetch = [&](int c) {
            for (int i = tid; i < BM * R::RU; i += NTH) {
                const int r = i / R::RU, k = i % R::RU;
                const uint8_t* p = src(r, c);
                const int mis = (int) ((uintptr_t) p & 15);
                const uint8_t* u = p - mis + 16 * k;
                uint32_t* d = raw + ((c & 1) * BM + r) * R::RW + 4 * k;
                if (16 * k + 16 <= mis + R::CB) cp16(d, u);
                else
                    for (int j = 0; j < 4 && 16 * k + 4 * j < mis + R::CB; ++j) cp_async4(d + j, u + 4 * j);
            }
            cp_commit();
        };
        // step s's x into its stage (the scale of the step's group)
        auto xfetch = [&](int s) {
            uint8_t* st = xst(s);
            const int c = c0 + (s >> 1), hh = s & 1;
            for (int i = tid; i < BN * 8; i += NTH) {
                const int t = i >> 3, k = i & 7;
                cp16(st + XS_::X + swz(t, k), xq + ((size_t) c * tp + tok0 + t) * CH + hh * SV + k * 16);
            }
            for (int i = tid; i < BN / 4; i += NTH)
                cp16(st + XS_::XD + i * 16, xD + ((size_t) c * (CH / G) + (G == CH ? 0 : hh)) * tp + tok0 + i * 4);
            if (MN != 0)
                for (int i = tid; i < BN; i += NTH) cp16(st + XS_::XS + i * 16, xS + ((size_t) c * tp + tok0 + i) * 16 + hh * 8);
            cp_commit();
        };
        // step s's weights: a (row, piece) of the 64 x 4, from the staged bytes into stage s & 1
        auto decode = [&](int s, int item) {
            uint8_t* st = wst(s);
            const int c = s >> 1, r = item >> 2, pl = item & 3, p = 4 * (s & 1) + pl;
            const int mis = (int) ((uintptr_t) src(r, c) & 15);
            const uint8_t* blk = (const uint8_t*) (raw + ((c & 1) * BM + r) * R::RW) + mis;
            Sub sb;
            int s0, s1, m0, m1;
            float d, dm;
            // a block at a 4- or 16-byte boundary read in words or wider (the reader's 2-byte loads merged)
            auto read = [&](const uint8_t* b) {
                gsub<T>(b, p, sb, grid);
                iscale<T>(b, p, s0, s1, m0, m1, d, dm);
            };
            if (mis == 0) read((const uint8_t*) __builtin_assume_aligned(blk, 16));
            else if ((mis & 3) == 0) read((const uint8_t*) __builtin_assume_aligned(blk, 4));
            else read(blk);
            *(int4*) (st + WSt::W + swz(r, 2 * pl)) = make_int4(sb.v[0], sb.v[1], sb.v[2], sb.v[3]);
            *(int4*) (st + WSt::W + swz(r, 2 * pl + 1)) = make_int4(sb.v[4], sb.v[5], sb.v[6], sb.v[7]);
            *(char2*) (st + WSt::WS + r * 8 + 2 * pl) = make_char2((char) s0, (char) s1);
            if (MN != 0) *(char2*) (st + WSt::WM + r * 8 + 2 * pl) = make_char2((char) m0, (char) m1);
            if (pl == 0) *(float2*) (st + WSt::WD + r * 8) = make_float2(d, dm);
        };
        xfetch(0);
        rfetch(0);
        for (int i = tid; i < Grid<T>::N; i += NTH) grid[i] = Grid<T>::src()[i];   // seen after the first bar 5
#pragma unroll 1
        for (int s = 0; s < nst + 2; ++s) {
            if (s >= 2) bar_sync(3 + (s & 1), ALL);   // empty: step s - 2 multiplied
            if (s >= nst) continue;
            int issued = 0;
            if (s + 1 < nst) {
                xfetch(s + 1);
                ++issued;
            }
            if ((s & 1) == 0 && s / 2 + 1 < nch) {
                rfetch(s / 2 + 1);
                ++issued;
            }
            // this step's x and its chunk's bytes in (all but what was just issued), seen by every producer
            if (issued == 2) cp_wait<2>();
            else if (issued == 1) cp_wait<1>();
            else cp_wait<0>();
            bar_sync(5, NTH);
            for (int item = tid; item < BM * 4; item += NTH) decode(s, item);
            bar_arrive(1 + (s & 1), ALL);             // full: step s ready
        }
        return;
    }

    // ---- the multipliers: a warp's 16 rows x 8 NT inputs; the integer sums of a group of x (ai, and the mins' am)
    // and the float sums
    float acc[NT][4];
    int ai[NT][4], am[NT][4];
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            acc[i][k] = 0.f;
            ai[i][k] = am[i][k] = 0;
        }
#pragma unroll 1
    for (int s = 0; s < nst; ++s) {
        bar_sync(1 + (s & 1), ALL);
        const uint8_t* st = wst(s);
        const uint8_t* xt = xst(s);
        // 4 pieces: each the n-tiles' mma (k32, or two k16 for the halves' own scales) issued together, their sums
        // times the pieces' scales
#pragma unroll 1
        for (int p = 0; p < 4; ++p) {
            int a[4];
            ldsm4(st + WSt::W + swz(R0 + (lane & 7) + ((lane >> 3) & 1) * 8, 2 * p + (lane >> 4)), a[0], a[1], a[2], a[3]);
            const char2 sA = *(const char2*) (st + WSt::WS + (R0 + g) * 8 + 2 * p), sB = *(const char2*) (st + WSt::WS + (R0 + g + 8) * 8 + 2 * p);
            if (!HV) {
                const int sg = (int8_t) sA.x, s8 = (int8_t) sB.x;
                int bf[NT][2];
#pragma unroll
                for (int n2 = 0; n2 < NT / 2; ++n2)
                    ldsm4(xt + XS_::X + swz(T0 + n2 * 16 + (lane & 7) + (lane >> 4) * 8, 2 * p + ((lane >> 3) & 1)), bf[2 * n2][0], bf[2 * n2][1],
                          bf[2 * n2 + 1][0], bf[2 * n2 + 1][1]);
                int cc[NT][4];
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) imma32(cc[nt], a[0], a[1], a[2], a[3], bf[nt][0], bf[nt][1]);
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    ai[nt][0] += cc[nt][0] * sg;
                    ai[nt][1] += cc[nt][1] * sg;
                    ai[nt][2] += cc[nt][2] * s8;
                    ai[nt][3] += cc[nt][3] * s8;
                }
            } else {
                // a half at a time: 4 n-tiles' k16 B fragments an ldmatrix, their 4 mma, the half's scale
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int sg = (int8_t) (h ? sA.y : sA.x), s8 = (int8_t) (h ? sB.y : sB.x);
#pragma unroll
                    for (int n4 = 0; n4 < NT / 4; ++n4) {
                        int bf[4];
                        ldsm4(xt + XS_::X + swz(T0 + n4 * 32 + (lane & 7) + (lane >> 3) * 8, 2 * p + h), bf[0], bf[1], bf[2], bf[3]);
                        int cc[4][4];
#pragma unroll
                        for (int k = 0; k < 4; ++k) imma16(cc[k], a[2 * h], a[2 * h + 1], bf[k]);
#pragma unroll
                        for (int k = 0; k < 4; ++k) {
                            const int nt = 4 * n4 + k;
                            ai[nt][0] += cc[k][0] * sg;
                            ai[nt][1] += cc[k][1] * sg;
                            ai[nt][2] += cc[k][2] * s8;
                            ai[nt][3] += cc[k][3] * s8;
                        }
                    }
                }
            }
        }
        // the mins against x's sums of 16 (the step's 8: dp2a, two a word against two of the mins' bytes)
        if (MN != 0) {
            const int2 u = *(const int2*) (st + WSt::WM + (R0 + g) * 8), v = *(const int2*) (st + WSt::WM + (R0 + g + 8) * 8);
            auto mins = [](const int4& xs, int m0, int m1, int r) {
                r = __dp2a_lo(xs.x, m0, r);
                r = __dp2a_hi(xs.y, m0, r);
                r = __dp2a_lo(xs.z, m1, r);
                return __dp2a_hi(xs.w, m1, r);
            };
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const int t = T0 + nt * 8 + 2 * q;
                const int4 x0 = *(const int4*) (xt + XS_::XS + t * 16), x1 = *(const int4*) (xt + XS_::XS + (t + 1) * 16);
                am[nt][0] = mins(x0, u.x, u.y, am[nt][0]);
                am[nt][1] = mins(x1, u.x, u.y, am[nt][1]);
                am[nt][2] = mins(x0, v.x, v.y, am[nt][2]);
                am[nt][3] = mins(x1, v.x, v.y, am[nt][3]);
            }
        }
        // a group of x done: its sums scaled once - d x x's scale (less dmin x it x the mins)
        if (G == SV || (s & 1)) {
            const float2 da = *(const float2*) (st + WSt::WD + (R0 + g) * 8), db = *(const float2*) (st + WSt::WD + (R0 + g + 8) * 8);
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const float2 dx = *(const float2*) (xt + XS_::XD + (T0 + nt * 8 + 2 * q) * 4);
                if (MN != 0) {
                    acc[nt][0] += dx.x * (da.x * (float) ai[nt][0] - da.y * (float) am[nt][0]);
                    acc[nt][1] += dx.y * (da.x * (float) ai[nt][1] - da.y * (float) am[nt][1]);
                    acc[nt][2] += dx.x * (db.x * (float) ai[nt][2] - db.y * (float) am[nt][2]);
                    acc[nt][3] += dx.y * (db.x * (float) ai[nt][3] - db.y * (float) am[nt][3]);
                } else {
                    acc[nt][0] += (dx.x * da.x) * (float) ai[nt][0];
                    acc[nt][1] += (dx.y * da.x) * (float) ai[nt][1];
                    acc[nt][2] += (dx.x * db.x) * (float) ai[nt][2];
                    acc[nt][3] += (dx.y * db.x) * (float) ai[nt][3];
                }
#pragma unroll
                for (int k = 0; k < 4; ++k) ai[nt][k] = am[nt][k] = 0;
            }
        }
        bar_arrive(3 + (s & 1), ALL);
    }
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
        const int r = row0 + R0 + g, t = tok0 + T0 + nt * 8 + 2 * q;
        if (t < ntok && r < n_out) y[(size_t) t * n_out + r] = acc[nt][0];
        if (t < ntok && r + 8 < n_out) y[(size_t) t * n_out + r + 8] = acc[nt][2];
        if (t + 1 < ntok && r < n_out) y[(size_t) (t + 1) * n_out + r] = acc[nt][1];
        if (t + 1 < ntok && r + 8 < n_out) y[(size_t) (t + 1) * n_out + r + 8] = acc[nt][3];
    }
}

// the split products' sums added up, in order (the same result run to run)
__global__ void ksum_k(const float* __restrict__ ws, float* __restrict__ y, size_t n, int parts) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = ws[i];
    for (int z = 1; z < parts; ++z) v += ws[(size_t) z * n + i];
    y[i] = v;
}

// the split products' room on each GPU, allocated outside any graph capture (ns_q35_tiles_reserve)
struct Ws {
    float* p = nullptr;
    size_t floats = 0;
};
inline Ws& wspace(int dev) {
    static Ws w[64];
    return w[dev & 63];
}
constexpr size_t WS_FLOATS = (size_t) 8 << 20;   // 32 MiB

template<int T, int NT, int WN, int PW, int G>
void kmm_go(const uint8_t* W, size_t rb, int64_t n_out, int64_t K, const uint8_t* x, int64_t ntok, float* y, cudaStream_t s) {
    constexpr int BN = 8 * NT * WN;
    constexpr size_t sm = smem<T, BN>();
    static bool set[64] = {};
    static int nsm[64] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    if (!set[dev & 63]) {
        cudaFuncSetAttribute(kmm_k<T, NT, WN, PW, G>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sm);
        cudaDeviceGetAttribute(&nsm[dev & 63], cudaDevAttrMultiProcessorCount, dev);
        set[dev & 63] = true;
    }
    // too few tiles for the GPU (short prompts, narrow matrices): K split until about two a multiprocessor, each
    // share at least two chunks, its sums in the reserved room
    const unsigned gx = (unsigned) ((ntok + BN - 1) / BN), gy = (unsigned) ((n_out + BM - 1) / BM);
    const int nall = (int) (K / CH);
    int parts = 1;
    const Ws& w = wspace(dev);
    const size_t out = (size_t) ntok * n_out;
    while ((size_t) gx * gy * parts < (size_t) 2 * nsm[dev & 63] && nall / (parts + 1) >= 2 && (size_t) (parts + 1) * out <= w.floats) ++parts;
    float* dst = parts > 1 ? w.p : y;
    kmm_k<T, NT, WN, PW, G><<<dim3(gx, gy, (unsigned) parts), 128 * WN + 32 * PW, sm, s>>>(W, rb, x, dst, (int) K, (int) n_out, (int) ntok,
                                                                                          (int) tpad(ntok), out);
    if (parts > 1) ksum_k<<<(unsigned) ((out + 255) / 256), 256, 0, s>>>(w.p, y, out, parts);
}
// the input tile: 32 or 64 rows for short prompts (4 multiplying warps, 8 decoding: the decode the larger share), 128
template<int T, int G>
void kmm_g(const uint8_t* W, size_t rb, int64_t n_out, int64_t K, const uint8_t* x, int64_t ntok, float* y, cudaStream_t s) {
    if (ntok <= 32) kmm_go<T, 4, 1, 8, G>(W, rb, n_out, K, x, ntok, y, s);
    else if (ntok <= 64) kmm_go<T, 8, 1, 8, G>(W, rb, n_out, K, x, ntok, y, s);
    else kmm_go<T, 8, 2, 8, G>(W, rb, n_out, K, x, ntok, y, s);
}
template<int T>
void kmm_launch(const void* w, int64_t n_out, int64_t K, const uint8_t* x, int64_t ntok, float* y, cudaStream_t s) {
    const size_t rb = (size_t) (K / Fmt<T>::qk) * Fmt<T>::bytes;
    kmm_g<T, SV>((const uint8_t*) w, rb, n_out, K, x, ntok, y, s);
}
}  // namespace k2

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

// x's layout for a type's prompt products: 2 the chunk products' (the 256-value block types), 1 the tile products'
// (Q8_0), -1 none
int layout(int t) { return !xmm_type(t) ? -1 : nsq::fmt_of(t).x == 256 ? 2 : 1; }

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

int ns_q35_tiles_layout(int type) { return layout(type); }

int ns_q35_tiles_reserve(ns_gpu* g) {
    NS_TRY
    NS_CU(cudaSetDevice(g->dev));
    k2::Ws& w = k2::wspace(g->dev);
    if (!w.p) {
        NS_CU(cudaMalloc(&w.p, k2::WS_FLOATS * sizeof(float)));
        w.floats = k2::WS_FLOATS;
    }
    return 0;
    NS_CATCH
}

// room for either layout (the chunk products' with its smallest groups)
size_t ns_q35_tiles_q8_bytes(int64_t K, int64_t T) { return std::max((size_t) (K / 32) * (size_t) T * 48, K % 256 == 0 ? k2::xbytes(K, T, k2::SV) : 0); }

int ns_q35_tiles_quantize(ns_gpu* g, int type, const float* x, int64_t T, int64_t K, void* xq) {
    NS_TRY
    NS_CU(cudaSetDevice(g->dev));
    if (!xmm_type(type) || K % 32 != 0) return ns_fail("ns_q35_tiles_quantize: a type without tile products, or K not a multiple of 32");
    if (layout(type) == 2) {
        if (K % 256 != 0) return ns_fail("ns_q35_tiles_quantize: K not a multiple of 256");
        const int64_t tp = k2::tpad(T), n = (K / 256) * tp * 32;
        const unsigned blocks = (unsigned) ((n + 255) / 256);
        k2::xq_k<k2::SV><<<blocks, 256, 0, g->q>>>(x, (uint8_t*) xq, T, tp, K);
        NS_LAUNCH_CHECK();
        return 0;
    }
    const int64_t n = T * (K / 32) * 32;
    if (n > 0) xtile_k<<<(unsigned) ((n + 255) / 256), 256, 0, g->q>>>(x, (uint8_t*) xq, T, K);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_q35_tiles_mmq(ns_gpu* g, int type, const void* w, int64_t n_out, int64_t K, const void* xq, int64_t T, float* y) {
    NS_TRY
    NS_CU(cudaSetDevice(g->dev));
    if (layout(type) == 2) {
        if (K % 256 != 0 || K > 65536) return ns_fail("ns_q35_tiles_mmq: K not a multiple of 256, or past 65536");
        if (T <= 0 || n_out <= 0) return 0;
        switch (type) {
#define Q35_KMM(TY) case TY: k2::kmm_launch<TY>(w, n_out, K, (const uint8_t*) xq, T, y, g->q); break;
            Q35_KMM(nsq::Q2_K) Q35_KMM(nsq::Q4_K) Q35_KMM(nsq::Q6_K) Q35_KMM(nsq::IQ2_XXS) Q35_KMM(nsq::IQ2_XS) Q35_KMM(nsq::IQ2_S)
            Q35_KMM(nsq::IQ3_XXS) Q35_KMM(nsq::IQ3_S) Q35_KMM(nsq::IQ4_XS)
#undef Q35_KMM
        }
    } else if (xmm(g, type, w, n_out, K, xq, T, y, false, "ns_q35_tiles_mmq") != 0) {
        return -1;
    }
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
