// q35_tiles.cu: nextcuda's own products of several rows on the int8 tensor cores - Qwen3.8's prompt products (the
// default since 2026-11-14, through q35_mmq.cu's entry points) and the opt-in verify products.
//
// The 256-value block types (namespace k2): x quantized a scale a 128 values so the sums stay integer within it (each
// mma's sums times the format's own integer scale, the K-quants' mins by dp2a), the group scaled once. Two kernels:
// for short prompts (64 rows or fewer) and the grid i-quants, ggml-cuda MMQ's structure (mq_k: load_tiles /
// process_tile, stream-k with fixup, mmq_x as MMQ picks it - ported, ggml MIT); for the rest a producer / multiplier
// pipeline (kmm_k). Measured 2026-11-14 on Qwen3.8-27B GSQ-RCO's matrices (9 roles, every type), the sum of the times
// these / ggml-cuda MMQ's: 16 rows 0.984 on the 5070, 0.953 on the 3060; 47 rows 0.970 / 0.920; 128 rows 0.982 / 0.991;
// 512 rows 0.951 / 0.989. Through the engine (5070 + 3060): a 181-token prompt 461 tok/s against 382, 703 tokens 740
// against 662, 3119 1168 against 1028, 16.6K 1343 against 1266; KL to MMQ's logits 0.00150, perplexity over 3000
// tokens 7.5821 against 7.5747.
//
// Q8_0 (the DSpine draft's layers, made at load) through mq_k at every prompt length, x with a scale a 32 (layout 3:
// MMQ's q8_1 numerics, each piece's sums scaled on their own as MMQ's Q8_0 case); rows not whole chunks of 256 through
// the older tile products (128 matrix rows x 64 or 128 input rows, cp.async staging, mma m16n8k32). 1..8 rows (a
// verify pass, opt-in): a warp a 16-row tile over all of K, the input rows straight from their Q8_1 blocks.
#include "ns.h"
#include "ns_internal.hpp"
#include "q35.h"
#include "nh.h"
#include "nh_blocks.cuh"

#include <cuda_fp16.h>
#include <algorithm>
#include <type_traits>
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
// (short prompts padded only to the input tile they are read in: 32 or 64)
inline int64_t tpad(int64_t t) { return t <= 32 ? 32 : t <= 64 ? 64 : (t + TP - 1) / TP * TP; }

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

// (Q8_0: its scale is a float a piece, applied in mq_k's own Q8_0 step; this only for the generic path's instantiation)
template<> __device__ __forceinline__ void iscale<nsq::Q8_0>(const uint8_t* b, int, int& s0, int& s1, int& m0, int& m1, float& d, float& dm) {
    s0 = s1 = 1; m0 = m1 = 0;
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
template<int T, int BN, int NR> constexpr size_t smem() {
    return (size_t) 2 * WSt::BYTES + (size_t) 3 * XSt<BN, Fmt<T>::mins != 0>::BYTES + (size_t) NR * BM * Raw<T>::RW * 4 +
           (size_t) Grid<T>::N * sizeof(typename Grid<T>::E);
}

// NR buffers of the blocks' bytes: copied NR - 1 chunks ahead (3 for short prompts, whose steps are short against the
// copies' latency: the weights stream from memory once)
template<int T, int NT, int WN, int PW, int G, int NR>
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
    uint32_t* raw = (uint32_t*) (xs0 + 3 * XS_::BYTES);   // [NR][BM][RW]
    GE* grid = (GE*) (raw + NR * BM * R::RW);
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
                uint32_t* d = raw + ((c % NR) * BM + r) * R::RW + 4 * k;
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
            const uint8_t* blk = (const uint8_t*) (raw + ((c % NR) * BM + r) * R::RW) + mis;
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
        // the copies' commit groups counted: a step's x and its chunk's bytes are in once all but the groups
        // committed after them are (x one step ahead, the bytes NR - 1 chunks)
        int groups = 0, gx = 0, gr0 = 0, gr1 = 0, gr2 = 0;
        auto rid = [&](int c) { const int k = c % NR; return k == 0 ? gr0 : k == 1 ? gr1 : gr2; };
        auto rset = [&](int c, int v) { const int k = c % NR; if (k == 0) gr0 = v; else if (k == 1) gr1 = v; else gr2 = v; };
        xfetch(0);
        gx = groups++;
        for (int k = 0; k < NR - 1 && k < nch; ++k) { rfetch(k); rset(k, groups++); }
        for (int i = tid; i < Grid<T>::N; i += NTH) grid[i] = Grid<T>::src()[i];   // seen after the first bar 5
#pragma unroll 1
        for (int s = 0; s < nst + 2; ++s) {
            if (s >= 2) bar_sync(3 + (s & 1), ALL);   // empty: step s - 2 multiplied
            if (s >= nst) continue;
            int gxn = gx;
            if (s + 1 < nst) {
                xfetch(s + 1);
                gxn = groups++;
            }
            if ((s & 1) == 0 && s / 2 + NR - 1 < nch) {
                rfetch(s / 2 + NR - 1);
                rset(s / 2 + NR - 1, groups++);
            }
            // this step's x and its chunk's bytes in, seen by every producer
            const int newer = groups - 1 - max(gx, rid(s / 2));
            if (newer >= 3) cp_wait<3>();
            else if (newer == 2) cp_wait<2>();
            else if (newer == 1) cp_wait<1>();
            else cp_wait<0>();
            gx = gxn;
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

template<int T, int NT, int WN, int PW, int G, int NR>
void kmm_go(const uint8_t* W, size_t rb, int64_t n_out, int64_t K, const uint8_t* x, int64_t ntok, float* y, cudaStream_t s) {
    constexpr int BN = 8 * NT * WN;
    constexpr size_t sm = smem<T, BN, NR>();
    static bool set[64] = {};
    static int nsm[64] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    if (!set[dev & 63]) {
        cudaFuncSetAttribute(kmm_k<T, NT, WN, PW, G, NR>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sm);
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
    kmm_k<T, NT, WN, PW, G, NR><<<dim3(gx, gy, (unsigned) parts), 128 * WN + 32 * PW, sm, s>>>(W, rb, x, dst, (int) K, (int) n_out, (int) ntok,
                                                                                          (int) tpad(ntok), out);
    if (parts > 1) ksum_k<<<(unsigned) ((out + 255) / 256), 256, 0, s>>>(w.p, y, out, parts);
}
// ---- short prompts (up to 64 input rows): ggml-cuda MMQ's structure (mmq.cuh's mul_mat_q_process_tile and
// load_tiles_*; ggml, MIT) - 128 matrix rows a thread block, 8 warps of 16 rows each over all J input rows, a chunk of
// 256 values a step decoded by every thread straight from global memory into shared int8 (4 pieces a thread, their
// loads in flight together), x copied in a half (128 values) at a time - load, barrier, multiply, barrier; mmq_x (J)
// the fewest tiles' as MMQ picks it. Two thread blocks an SM (MMQ runs one): one's decode beside the other's products.
// Our formats' integer scales and x's scale a 128 as above
constexpr int MI = 128;
__device__ __forceinline__ int swz2(int r, int k) { return r * CH + ((k ^ (r & 7)) << 4); }   // 256-byte rows

// (XN: x's scales a half - 1, a scale a 128; Q8_0's 4, a scale a 32. Q8_0's own scales, a float a piece, in WS..WD)
template<int J, bool SUMS, int XN = 1> struct MSt {
    static constexpr int W = 0, WS = W + MI * CH, WM = WS + MI * 16, WD = WM + MI * 16, X = WD + MI * 8, XD = X + J * SV, XS = XD + J * 4 * XN,
                         BYTES = XS + (SUMS ? J * 16 : 0);
};
// Q8_0's 4 pieces a half unrolled, from 33 input rows, by this many: all 4 on the 5070 (its loads, products and scaling
// interleaved - 6% faster at 512 rows), 2 on the 3060 (all 4 spill there: 3-5% slower); all 4 at any J of one block an
// SM (MMQ's own loop is unrolled whole)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
constexpr int Q8U = 4;
#else
constexpr int Q8U = 2;
#endif
// x's values a scale for a type in mq_k: Q8_0's 32 (its weights scaled a piece at a time anyway: MMQ's q8_1 numerics)
template<int T> __host__ __device__ constexpr int mqg() { return T == nsq::Q8_0 ? 32 : SV; }
// (the codebook in shared memory where it is 4 KB or less: IQ2_S's 8 KB would leave room for one block an SM)
template<int T> __host__ __device__ constexpr bool msgrid() { return Grid<T>::N * sizeof(typename Grid<T>::E) <= 4096; }
template<int T, int J> constexpr size_t msmem() {
    return (size_t) MSt<J, Fmt<T>::mins != 0, SV / mqg<T>()>::BYTES + (msgrid<T>() ? (size_t) Grid<T>::N * sizeof(typename Grid<T>::E) : 0);
}

// Stream-k (mmq.cuh's mul_mat_q stream-k path and mul_mat_q_stream_k_fixup; ggml, MIT): the thread blocks share the
// (tile, chunk) iterations evenly, a tile at a time; a tile finished in a block goes to y, a block's last, unfinished
// tile to its own fixup room, added into y after (in block order) by the block that finishes that tile
template<int T, int J>
__global__ void __launch_bounds__(256, J <= 64 ? 2 : 1) mq_k(const uint8_t* __restrict__ w, size_t rb, const uint8_t* __restrict__ xq, float* __restrict__ y,
                                               int K, int n_out, int ntok, int tp, float* __restrict__ fixup) {
    constexpr int NT = J / 8;
    constexpr bool HV = nsq::Halves<T>::v;
    constexpr int MN = Fmt<T>::mins;
    constexpr bool Q8 = T == nsq::Q8_0;
    constexpr int XN = SV / mqg<T>();
    using S = MSt<J, MN != 0, XN>;
    using GE = typename Grid<T>::E;
    extern __shared__ __align__(16) uint8_t sm[];
    const GE* grid = (const GE*) (sm + S::BYTES);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, q = lane & 3;
    const int R0 = warp * 16;
    // Q8_0 from 64 input rows: MMQ's mma layout (vec_dot_q8_0_q8_1_mma; ggml, MIT) - a warp 32 rows (2 m-tiles) x half
    // the n-tiles, against 16 rows x all of them: a third less shared memory read a product. The accumulators: entry
    // i of acc the warp's m-tile i / NTW, n-tile NB + i % NTW
    constexpr int MW = Q8 && J >= 64 ? 2 : 1, NTW = NT / MW;
    const int RB = (warp / MW) * 16 * MW, NB = (warp % MW) * NTW;
    uint32_t i4t[4];
    nsq::iq4nl_table(i4t);
    const int nch = K / CH, ntx = (ntok + J - 1) / J, nty = (n_out + MI - 1) / MI;
    const int total = ntx * nty * nch;
    int kbc = (int) ((int64_t) blockIdx.x * total / gridDim.x);
    const int kbc_stop = (int) ((int64_t) (blockIdx.x + 1) * total / gridDim.x);
    const float* xD = (const float*) (xq + (size_t) nch * tp * CH);
    const int16_t* xS = (const int16_t*) (xD + (size_t) nch * 2 * XN * tp);
    if (msgrid<T>())
        for (int i = tid; i < Grid<T>::N; i += 256) ((GE*) (sm + S::BYTES))[i] = Grid<T>::src()[i];
    else
        grid = (GE*) Grid<T>::src();
    __syncthreads();
#pragma unroll 1
    while (kbc < kbc_stop) {
    const int tile = kbc / nch, c0 = kbc % nch, c1 = min(nch, c0 + kbc_stop - kbc);
    kbc += c1 - c0;
    const int row0 = (tile / ntx) * MI, tok0 = (tile % ntx) * J;
    float acc[NT][4];
    int ai[NT][4];
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int k = 0; k < 4; ++k) acc[i][k] = 0.f;
    // x a half ahead in registers: read during the products, stored after the barrier that ends them
    constexpr int XU = (J * 8 + 255) / 256, DU = (J * XN + 255) / 256;
    int4 xr[XU], xs4 = make_int4(0, 0, 0, 0);
    float xd[DU];
    auto xload = [&](int c, int hh) {
#pragma unroll
        for (int u = 0; u < XU; ++u) {
            const int i = tid + 256 * u, t = i >> 3, k = i & 7;
            if (i < J * 8) xr[u] = *(const int4*) (xq + ((size_t) c * tp + tok0 + t) * CH + hh * SV + k * 16);
        }
        // its scales [XN][J]: scale pp of the half for input row t
#pragma unroll
        for (int u = 0; u < DU; ++u) {
            const int i = tid + 256 * u;
            if (i < J * XN) xd[u] = xD[((size_t) c * 2 * XN + hh * XN + i / J) * tp + tok0 + i % J];
        }
        if (MN != 0 && tid < J) xs4 = *(const int4*) (xS + ((size_t) c * tp + tok0 + tid) * 16 + hh * 8);
    };
    xload(c0, 0);
#pragma unroll 1
    for (int c = c0; c < c1; ++c) {
        // load_tiles: a thread a (row, piece), 4 of them, from the blocks in global memory - their loads in flight
        // together: the blocks' alignment (the same for every row where a row is a multiple of 16 bytes) chosen once a
        // chunk, not a branch a piece
        // a row's block: its word k into the int8 row at byte 4j of piece p (both nibble halves for the 4-bit types)
        auto put4 = [&](int r, int p, int j, uint32_t v) { *(uint32_t*) (sm + S::W + swz2(r, 2 * p + (j >> 2)) + 4 * (j & 3)) = v; };
        auto rowp = [&](int r) { return w + (size_t) min(row0 + r, n_out - 1) * rb + (size_t) c * Fmt<T>::bytes; };
        auto load = [&](auto al) {
            constexpr int AL = decltype(al)::value;
            if constexpr (Q8) {
                // load_tiles_q8_0 (ggml, MIT): a thread a (row, piece), 4 of them - the piece's 34-byte block from 4-byte
                // words (a block starts 0 or 2 into one; the words past its end stay in the word its last bytes are
                // in), its half scale as a float a piece. (A lane a word of a block, MMQ's own mapping, measured slower
                // here: its 32 rows' loads in flight cost the registers of two blocks an SM)
                (void) AL;
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const int item = tid + 256 * u, r = item >> 3, p = item & 7;
                    const uint8_t* blk = w + (size_t) min(row0 + r, n_out - 1) * rb + (size_t) (c * 8 + p) * 34;
                    const uint32_t* wd = (const uint32_t*) ((uintptr_t) blk & ~(uintptr_t) 3);
                    const bool odd = ((uintptr_t) blk & 2) != 0;
                    uint32_t v[9];
#pragma unroll
                    for (int k = 0; k < 9; ++k) v[k] = wd[k];
                    uint32_t q[8];
#pragma unroll
                    for (int k = 0; k < 8; ++k) q[k] = odd ? v[k + 1] : __funnelshift_r(v[k], v[k + 1], 16);
                    const uint16_t hd = odd ? (uint16_t) (v[0] >> 16) : (uint16_t) (v[0] & 0xffffu);
                    *(int4*) (sm + S::W + swz2(r, 2 * p)) = make_int4((int) q[0], (int) q[1], (int) q[2], (int) q[3]);
                    *(int4*) (sm + S::W + swz2(r, 2 * p + 1)) = make_int4((int) q[4], (int) q[5], (int) q[6], (int) q[7]);
                    ((float*) (sm + S::WS))[r * 8 + p] = __half2float(__ushort_as_half(hd));
                }
                return;
            } else if constexpr (T == nsq::IQ4_XS || T == nsq::Q6_K) {
                // load_tiles_iq4_xs / _q6_K's mapping (ggml, MIT): a warp a row, a lane a word of its values -
                // 16 rows a thread, each word read once, the rows' 128 bytes a coalesced read
#pragma unroll
                for (int i0 = 0; i0 < MI; i0 += 8) {
                    const int r = i0 + warp;
                    const uint8_t* b = (const uint8_t*) __builtin_assume_aligned(rowp(r), AL);
                    if constexpr (T == nsq::Q4_K) {
                        const uint32_t q = *(const uint32_t*) (b + 16 + 4 * lane);   // pair k = lane / 8, word lane % 8
                        put4(r, 2 * (lane >> 3), lane & 7, q & 0x0f0f0f0fu);
                        put4(r, 2 * (lane >> 3) + 1, lane & 7, (q >> 4) & 0x0f0f0f0fu);
                    } else if constexpr (T == nsq::IQ4_XS) {
                        const int2 v = nsq::levels8(*(const uint32_t*) (b + 8 + 4 * lane), i4t);   // piece lane / 4, word lane % 4
                        put4(r, lane >> 2, lane & 3, (uint32_t) v.x);
                        put4(r, lane >> 2, 4 + (lane & 3), (uint32_t) v.y);
                    } else {
                        // ql word lane: half n = lane / 16, w = lane % 16: piece 4n + w / 8 (low nibbles) and + 2 (high),
                        // its bytes 4 (w % 8); the high bits from qh word 8n + w % 8
                        const int n = lane >> 4, wv = lane & 15, k1 = wv >> 3, j = wv & 7;
                        const uint32_t ql = nsq::u32_at(b + 4 * lane), qh = nsq::u32_at(b + 128 + 4 * (8 * n + j));
                        put4(r, 4 * n + k1, j, __vsub4((ql & 0x0f0f0f0fu) | (((qh >> (2 * k1)) & 0x03030303u) << 4), 0x20202020u));
                        put4(r, 4 * n + k1 + 2, j, __vsub4(((ql >> 4) & 0x0f0f0f0fu) | (((qh >> (2 * k1 + 4)) & 0x03030303u) << 4), 0x20202020u));
                    }
                }
                // the scales (and mins): a thread a (row, piece), 4 of them
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const int item = tid + 256 * u, r = item >> 3, p = item & 7;
                    const uint8_t* b = (const uint8_t*) __builtin_assume_aligned(rowp(r), AL);
                    int s0, s1, m0, m1;
                    float d, dm;
                    iscale<T>(b, p, s0, s1, m0, m1, d, dm);
                    *(char2*) (sm + S::WS + r * 16 + 2 * p) = make_char2((char) s0, (char) s1);
                    if (MN != 0) *(char2*) (sm + S::WM + r * 16 + 2 * p) = make_char2((char) m0, (char) m1);
                    if (p == 0) *(float2*) (sm + S::WD + r * 8) = make_float2(d, dm);
                }
                return;
            }
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int item = tid + 256 * u, r = item >> 3, p = item & 7;
                const uint8_t* blk = (const uint8_t*) __builtin_assume_aligned(
                    w + (size_t) min(row0 + r, n_out - 1) * rb + (size_t) c * Fmt<T>::bytes, AL);
                Sub sb;
                int s0, s1, m0, m1;
                float d, dm;
                gsub<T>(blk, p, sb, grid);
                iscale<T>(blk, p, s0, s1, m0, m1, d, dm);
                *(int4*) (sm + S::W + swz2(r, 2 * p)) = make_int4(sb.v[0], sb.v[1], sb.v[2], sb.v[3]);
                *(int4*) (sm + S::W + swz2(r, 2 * p + 1)) = make_int4(sb.v[4], sb.v[5], sb.v[6], sb.v[7]);
                *(char2*) (sm + S::WS + r * 16 + 2 * p) = make_char2((char) s0, (char) s1);
                if (MN != 0) *(char2*) (sm + S::WM + r * 16 + 2 * p) = make_char2((char) m0, (char) m1);
                if (p == 0) *(float2*) (sm + S::WD + r * 8) = make_float2(d, dm);
            }
        };
        {
            const uintptr_t a0 = (uintptr_t) (w + (size_t) row0 * rb + (size_t) c * Fmt<T>::bytes) | (rb & 15);
            if ((a0 & 15) == 0) load(std::integral_constant<int, 16>());
            else if ((a0 & 3) == 0) load(std::integral_constant<int, 4>());
            else load(std::integral_constant<int, 2>());
        }
#pragma unroll 1
        for (int hh = 0; hh < 2; ++hh) {
            // x's half (read a half ago) into shared memory; then the next half's read
#pragma unroll
            for (int u = 0; u < XU; ++u) {
                const int i = tid + 256 * u, t = i >> 3, k = i & 7;
                if (i < J * 8) *(int4*) (sm + S::X + swz(t, k)) = xr[u];
            }
#pragma unroll
            for (int u = 0; u < DU; ++u)
                if (tid + 256 * u < J * XN) ((float*) (sm + S::XD))[tid + 256 * u] = xd[u];
            if (MN != 0 && tid < J) *(int4*) (sm + S::XS + tid * 16) = xs4;
            constexpr int o = 0;
            __syncthreads();
            if (hh == 0) xload(c, 1);
            else if (c + 1 < c1) xload(c + 1, 0);
            if constexpr (Q8) {
                // 4 pieces: the warp's MW m-tiles' A fragments, its NTW n-tiles' B, each piece's sums scaled on their own
#pragma unroll (J <= 32 || J > 64 ? 4 : Q8U)
                for (int pl = 0; pl < 4; ++pl) {
                    const int p = 4 * hh + pl;
                    int a[MW][4];
#pragma unroll
                    for (int m = 0; m < MW; ++m)
                        ldsm4(sm + S::W + swz2(RB + m * 16 + (lane & 7) + ((lane >> 3) & 1) * 8, 2 * p + (lane >> 4)), a[m][0], a[m][1], a[m][2], a[m][3]);
                    int bf[NTW][2];
#pragma unroll
                    for (int n2 = 0; n2 + 1 < NTW; n2 += 2)
                        ldsm4(sm + S::X + swz((NB + n2) * 8 + (lane & 7) + (lane >> 4) * 8, 2 * pl + ((lane >> 3) & 1)), bf[n2][0], bf[n2][1], bf[n2 + 1][0],
                              bf[n2 + 1][1]);
                    if (NTW & 1)
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
                                     : "=r"(bf[NTW - 1][0]), "=r"(bf[NTW - 1][1])
                                     : "r"((unsigned) __cvta_generic_to_shared(sm + S::X + swz((NB + NTW - 1) * 8 + (lane & 7), 2 * pl + ((lane >> 3) & 1)))));
                    float2 dx[NTW];
#pragma unroll
                    for (int j = 0; j < NTW; ++j) dx[j] = *(const float2*) (sm + S::XD + (pl * J + (NB + j) * 8 + 2 * q) * 4);
#pragma unroll
                    for (int m = 0; m < MW; ++m) {
                        const float wa = ((const float*) (sm + S::WS))[(RB + m * 16 + g) * 8 + p], wb = ((const float*) (sm + S::WS))[(RB + m * 16 + g + 8) * 8 + p];
                        int cc[NTW][4];
#pragma unroll
                        for (int j = 0; j < NTW; ++j) imma32(cc[j], a[m][0], a[m][1], a[m][2], a[m][3], bf[j][0], bf[j][1]);
#pragma unroll
                        for (int j = 0; j < NTW; ++j) {
                            float* A = acc[m * NTW + j];
                            A[0] += (dx[j].x * wa) * (float) cc[j][0];
                            A[1] += (dx[j].y * wa) * (float) cc[j][1];
                            A[2] += (dx[j].x * wb) * (float) cc[j][2];
                            A[3] += (dx[j].y * wb) * (float) cc[j][3];
                        }
                    }
                }
            } else {
            // vec_dot: 4 pieces, the warp's 16 rows' A fragments by ldmatrix against the NT n-tiles
#pragma unroll
            for (int i = 0; i < NT; ++i)
#pragma unroll
                for (int k = 0; k < 4; ++k) ai[i][k] = 0;
#pragma unroll (Q8 && J <= 32 ? 4 : 1)
            for (int pl = 0; pl < 4; ++pl) {
                const int p = 4 * hh + pl;
                int a[4];
                ldsm4(sm + S::W + swz2(R0 + (lane & 7) + ((lane >> 3) & 1) * 8, 2 * p + (lane >> 4)), a[0], a[1], a[2], a[3]);
                char2 sA = make_char2(1, 1), sB = make_char2(1, 1);
                if constexpr (!Q8) {
                    sA = *(const char2*) (sm + S::WS + (R0 + g) * 16 + 2 * p);
                    sB = *(const char2*) (sm + S::WS + (R0 + g + 8) * 16 + 2 * p);
                }
#pragma unroll
                for (int h = 0; h < (HV ? 2 : 1); ++h) {
                    const int sg = (int8_t) (h ? sA.y : sA.x), s8 = (int8_t) (h ? sB.y : sB.x);
                    // all the n-tiles' B fragments first (ldmatrix x4: two n-tiles' 16-byte halves, or for k16 four
                    // n-tiles' half h), then their mma, then the scaling - no load-to-mma-to-add chain an n-tile
                    int bf[NT][2];
                    if (!HV) {
#pragma unroll
                        for (int n2 = 0; n2 + 1 < NT; n2 += 2)
                            ldsm4(sm + S::X + o + swz(n2 * 8 + (lane & 7) + (lane >> 4) * 8, 2 * pl + ((lane >> 3) & 1)), bf[n2][0], bf[n2][1], bf[n2 + 1][0],
                                  bf[n2 + 1][1]);
                        if (NT & 1)
                            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
                                         : "=r"(bf[NT - 1][0]), "=r"(bf[NT - 1][1])
                                         : "r"((unsigned) __cvta_generic_to_shared(sm + S::X + o + swz((NT - 1) * 8 + (lane & 7), 2 * pl + ((lane >> 3) & 1)))));
                    } else {
#pragma unroll
                        for (int n4 = 0; n4 + 3 < NT; n4 += 4)
                            ldsm4(sm + S::X + o + swz(n4 * 8 + (lane & 7) + (lane >> 3) * 8, 2 * pl + h), bf[n4][0], bf[n4 + 1][0], bf[n4 + 2][0], bf[n4 + 3][0]);
#pragma unroll
                        for (int nt = NT & ~3; nt < NT; ++nt)
                            asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];"
                                         : "=r"(bf[nt][0])
                                         : "r"((unsigned) __cvta_generic_to_shared(sm + S::X + o + swz(nt * 8 + (lane & 7), 2 * pl + h))));
                    }
                    int cc[NT][4];
#pragma unroll
                    for (int nt = 0; nt < NT; ++nt) {
                        if (!HV) imma32(cc[nt], a[0], a[1], a[2], a[3], bf[nt][0], bf[nt][1]);
                        else imma16(cc[nt], a[2 * h], a[2 * h + 1], bf[nt][0]);
                    }
                    if constexpr (Q8) {
                        // a piece's sums scaled on their own (x's scale and the block's, both a 32's): as MMQ's Q8_0
                        (void) sg; (void) s8;
                        const float wa = ((const float*) (sm + S::WS))[(R0 + g) * 8 + p], wb = ((const float*) (sm + S::WS))[(R0 + g + 8) * 8 + p];
#pragma unroll
                        for (int nt = 0; nt < NT; ++nt) {
                            const float2 dx = *(const float2*) (sm + S::XD + (pl * J + nt * 8 + 2 * q) * 4);
                            acc[nt][0] += (dx.x * wa) * (float) cc[nt][0];
                            acc[nt][1] += (dx.y * wa) * (float) cc[nt][1];
                            acc[nt][2] += (dx.x * wb) * (float) cc[nt][2];
                            acc[nt][3] += (dx.y * wb) * (float) cc[nt][3];
                        }
                    } else {
#pragma unroll
                        for (int nt = 0; nt < NT; ++nt) {
                            ai[nt][0] += cc[nt][0] * sg;
                            ai[nt][1] += cc[nt][1] * sg;
                            ai[nt][2] += cc[nt][2] * s8;
                            ai[nt][3] += cc[nt][3] * s8;
                        }
                    }
                }
            }
            if constexpr (!Q8) {
            const float2 da = *(const float2*) (sm + S::WD + (R0 + g) * 8), db = *(const float2*) (sm + S::WD + (R0 + g + 8) * 8);
            int2 u = make_int2(0, 0), v = make_int2(0, 0);
            if (MN != 0) {
                u = *(const int2*) (sm + S::WM + (R0 + g) * 16 + 8 * hh);
                v = *(const int2*) (sm + S::WM + (R0 + g + 8) * 16 + 8 * hh);
            }
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const int t = nt * 8 + 2 * q;
                const float2 dx = *(const float2*) (sm + S::XD + o + t * 4);
                if (MN != 0) {
                    const int4 x0 = *(const int4*) (sm + S::XS + o + t * 16), x1 = *(const int4*) (sm + S::XS + o + (t + 1) * 16);
                    auto mins = [](const int4& xs, int m0, int m1) {
                        int r = __dp2a_lo(xs.x, m0, 0);
                        r = __dp2a_hi(xs.y, m0, r);
                        r = __dp2a_lo(xs.z, m1, r);
                        return __dp2a_hi(xs.w, m1, r);
                    };
                    acc[nt][0] += dx.x * (da.x * (float) ai[nt][0] - da.y * (float) mins(x0, u.x, u.y));
                    acc[nt][1] += dx.y * (da.x * (float) ai[nt][1] - da.y * (float) mins(x1, u.x, u.y));
                    acc[nt][2] += dx.x * (db.x * (float) ai[nt][2] - db.y * (float) mins(x0, v.x, v.y));
                    acc[nt][3] += dx.y * (db.x * (float) ai[nt][3] - db.y * (float) mins(x1, v.x, v.y));
                } else {
                    acc[nt][0] += (dx.x * da.x) * (float) ai[nt][0];
                    acc[nt][1] += (dx.y * da.x) * (float) ai[nt][1];
                    acc[nt][2] += (dx.x * db.x) * (float) ai[nt][2];
                    acc[nt][3] += (dx.y * db.x) * (float) ai[nt][3];
                }
            }
            }
            }   // (not Q8_0)
            __syncthreads();
        }
    }
    if (c1 == nch) {
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const int r = row0 + RB + (nt / NTW) * 16 + g, t = tok0 + (NB + nt % NTW) * 8 + 2 * q;
            if (t < ntok && r < n_out) y[(size_t) t * n_out + r] = acc[nt][0];
            if (t < ntok && r + 8 < n_out) y[(size_t) t * n_out + r + 8] = acc[nt][2];
            if (t + 1 < ntok && r < n_out) y[(size_t) (t + 1) * n_out + r] = acc[nt][1];
            if (t + 1 < ntok && r + 8 < n_out) y[(size_t) (t + 1) * n_out + r + 8] = acc[nt][3];
        }
    } else {
        // the unfinished tile (the block's last): [input in tile][row in tile], as y's rows run
        float* f = fixup + (size_t) blockIdx.x * MI * J;
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const int r = RB + (nt / NTW) * 16 + g, t = (NB + nt % NTW) * 8 + 2 * q;
            f[t * MI + r] = acc[nt][0];
            f[(t + 1) * MI + r] = acc[nt][1];
            f[t * MI + r + 8] = acc[nt][2];
            f[(t + 1) * MI + r + 8] = acc[nt][3];
        }
    }
    }
}

// a block that finished a tile it did not start adds the earlier blocks' parts of it, in block order. A thread an
// entry of the tile (blockIdx.y the tile's slice of 256): one loop over the parts a thread, their loads in flight
// together - a thread looping over a whole tile's entries waits out each load in turn (30 us for a 128 x 128 tile)
template<int J>
__global__ void mq_fixup_k(float* __restrict__ y, const float* __restrict__ fixup, int n_out, int ntok, int nch, int nb) {
    const int b = blockIdx.x, ntx = (ntok + J - 1) / J, total = ntx * ((n_out + MI - 1) / MI) * nch;
    const int s = (int) ((int64_t) b * total / nb), e = (int) ((int64_t) (b + 1) * total / nb);
    if (s % nch == 0 || e < (s / nch + 1) * nch) return;   // starts a tile, or finishes none it started in
    const int tile = s / nch, row0 = (tile / ntx) * MI, tok0 = (tile % ntx) * J;
    const int i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= MI * J) return;
    const int t = i / MI, r = i % MI;   // (consecutive threads along y's row)
    if (row0 + r >= n_out || tok0 + t >= ntok) return;
    float v = 0.f;
    for (int j = b - 1; j >= 0; --j) {
        v += fixup[(size_t) j * MI * J + i];
        if ((int) ((int64_t) j * total / nb) <= tile * nch) break;   // j started at or before the tile's start
    }
    y[(size_t) (tok0 + t) * n_out + row0 + r] += v;
}

template<int T, int J>
void mq_go(const uint8_t* W, size_t rb, int64_t n_out, int64_t K, const uint8_t* x, int64_t ntok, float* y, cudaStream_t s) {
    constexpr size_t sm = msmem<T, J>();
    static bool set[64] = {};
    static int nsm[64] = {}, major[64] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    if (!set[dev & 63]) {
        cudaFuncSetAttribute(mq_k<T, J>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sm);
        cudaDeviceGetAttribute(&nsm[dev & 63], cudaDevAttrMultiProcessorCount, dev);
        cudaDeviceGetAttribute(&major[dev & 63], cudaDevAttrComputeCapabilityMajor, dev);
        set[dev & 63] = true;
    }
    // two thread blocks an SM over all the (tile, chunk) iterations; tiling alone where it already fills the waves
    // to 90 % (as MMQ: no fixup then)
    const int ntiles = (int) (((ntok + J - 1) / J) * ((n_out + MI - 1) / MI)), slots = (J <= 64 ? 2 : 1) * nsm[dev & 63];
    const int waves = (ntiles + slots - 1) / slots;
    const Ws& wsp = wspace(dev);
    const int total = ntiles * (int) (K / CH);
    // (fewer tiles than slots: a block for each slot, 8 chunks a block at least. More, for Q8_0 on the 3060: a block a
    // slot, each its share of the iterations across the tiles - MMQ's stream-k - not a last wave of few tiles; measured
    // slower on the 5070, and for the other types the plain tiles kept as they were measured)
    const bool spread = T == nsq::Q8_0 && major[dev & 63] < 9;
    // (Q8_0's blocks at least 2 chunks: the narrow k / v matrices' short prompts on every slot, as MMQ - 3-5% there)
    const int minch = T == nsq::Q8_0 ? 2 : 8;
    int nb = 100 * ntiles / (slots * waves) >= 90 ? ntiles : spread && ntiles >= slots ? slots : std::max(ntiles, std::min(slots, total / minch));
    if ((size_t) nb * MI * J > wsp.floats) nb = ntiles;   // (no room reserved: plain tiling)
    mq_k<T, J><<<nb, 256, sm, s>>>(W, rb, x, y, (int) K, (int) n_out, (int) ntok, (int) tpad(ntok), wsp.p);
    if (nb != ntiles) mq_fixup_k<J><<<dim3((unsigned) nb, (unsigned) (MI * J / 256)), 256, 0, s>>>(y, wsp.p, (int) n_out, (int) ntok, (int) (K / CH), nb);
}
// mmq_x the way MMQ picks it (mul_mat_q_switch_J): the fewest column tiles, then the narrowest; 8 .. 64 here
template<int T>
void mq_launch(const uint8_t* W, size_t rb, int64_t n_out, int64_t K, const uint8_t* x, int64_t ntok, float* y, cudaStream_t s) {
    // (Q8_0 past 64 rows in tiles of 64: two blocks an SM, one's loads beside the other's products - measured faster
    // than MMQ's 96 / 128 here, on both cards)
    if (T == nsq::Q8_0 && ntok > 64) {
        mq_go<T, 64>(W, rb, n_out, K, x, ntok, y, s);
        return;
    }
    if (ntok <= 8) mq_go<T, 8>(W, rb, n_out, K, x, ntok, y, s);
    else if (ntok <= 16) mq_go<T, 16>(W, rb, n_out, K, x, ntok, y, s);
    else if (ntok <= 24) mq_go<T, 24>(W, rb, n_out, K, x, ntok, y, s);
    else if (ntok <= 32) mq_go<T, 32>(W, rb, n_out, K, x, ntok, y, s);
    else if (ntok <= 40) mq_go<T, 40>(W, rb, n_out, K, x, ntok, y, s);
    else if (ntok <= 48) mq_go<T, 48>(W, rb, n_out, K, x, ntok, y, s);
    else if (ntok <= 56) mq_go<T, 56>(W, rb, n_out, K, x, ntok, y, s);
    else if (ntok <= 64) mq_go<T, 64>(W, rb, n_out, K, x, ntok, y, s);
    else if (ntok <= 96) mq_go<T, 96>(W, rb, n_out, K, x, ntok, y, s);
    else mq_go<T, 128>(W, rb, n_out, K, x, ntok, y, s);
}

// the input tile: 32 or 64 rows for short prompts (4 multiplying warps, 8 decoding: the decode the larger share), 128
template<int T, int G>
void kmm_g(const uint8_t* W, size_t rb, int64_t n_out, int64_t K, const uint8_t* x, int64_t ntok, float* y, cudaStream_t s) {
    if (ntok <= 32) kmm_go<T, 4, 1, 8, G, 3>(W, rb, n_out, K, x, ntok, y, s);
    else if (ntok <= 64) kmm_go<T, 8, 1, 8, G, 3>(W, rb, n_out, K, x, ntok, y, s);
    else kmm_go<T, 8, 2, 8, G, 2>(W, rb, n_out, K, x, ntok, y, s);
}
template<int T>
void kmm_launch(const void* w, int64_t n_out, int64_t K, const uint8_t* x, int64_t ntok, float* y, cudaStream_t s) {
    const size_t rb = (size_t) (K / Fmt<T>::qk) * Fmt<T>::bytes;
    // which of the two kernels, by timing both on Qwen3.8-27B's matrices (16 / 47 / 128 / 512 input rows, the 5070 and
    // the 3060): MMQ's structure (mq_k) for short prompts and, past them, for the grid i-quants (their decode the larger
    // share, every warp at it); the producer / multiplier pipeline (kmm_k) for the rest and for narrow matrices
    // (2048 rows or fewer)
    int dev = 0, major = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    bool mq = n_out > 2048;
    if (mq && ntok > 64) {
        const bool grid = T == nsq::IQ2_XXS || T == nsq::IQ3_XXS || T == nsq::IQ3_S;
        mq = grid || (T == nsq::IQ4_XS && (major >= 9 ? ntok <= 128 : ntok > 128));
    }
    if (mq) {
        mq_launch<T>((const uint8_t*) w, rb, n_out, K, x, ntok, y, s);
        return;
    }
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

// x's layout for a type's prompt products: 2 the chunk products' (the 256-value block types), 3 theirs with a scale a
// 32 (Q8_0, through mq_k), -1 none. (Q8_0 rows not whole chunks of 256 go through the older tile products, x in their
// layout 1: quantize and the products both choose by K)
int layout(int t) { return !xmm_type(t) ? -1 : t == nsq::Q8_0 ? 3 : 2; }
bool q8_chunks(int t, int64_t K) { return t == nsq::Q8_0 && K % 256 == 0; }

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
size_t ns_q35_tiles_q8_bytes(int64_t K, int64_t T) { return std::max((size_t) (K / 32) * (size_t) T * 48, K % 256 == 0 ? k2::xbytes(K, T, 32) : 0); }

int ns_q35_tiles_quantize(ns_gpu* g, int type, const float* x, int64_t T, int64_t K, void* xq) {
    NS_TRY
    NS_CU(cudaSetDevice(g->dev));
    if (!xmm_type(type) || K % 32 != 0) return ns_fail("ns_q35_tiles_quantize: a type without tile products, or K not a multiple of 32");
    if (layout(type) == 2 || q8_chunks(type, K)) {
        if (K % 256 != 0) return ns_fail("ns_q35_tiles_quantize: K not a multiple of 256");
        const int64_t tp = k2::tpad(T), n = (K / 256) * tp * 32;
        const unsigned blocks = (unsigned) ((n + 255) / 256);
        if (type == nsq::Q8_0) k2::xq_k<32><<<blocks, 256, 0, g->q>>>(x, (uint8_t*) xq, T, tp, K);
        else k2::xq_k<k2::SV><<<blocks, 256, 0, g->q>>>(x, (uint8_t*) xq, T, tp, K);
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
    } else if (q8_chunks(type, K)) {
        if (K > 65536) return ns_fail("ns_q35_tiles_mmq: K past 65536");
        if (T <= 0 || n_out <= 0) return 0;
        k2::mq_launch<nsq::Q8_0>((const uint8_t*) w, (size_t) (K / 32) * 34, n_out, K, (const uint8_t*) xq, T, y, g->q);
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
