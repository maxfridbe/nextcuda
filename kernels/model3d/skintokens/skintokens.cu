// skintokens.cu: SkinTokens / TokenRig's kernels (skintokens.h): layer and RMS norms, activations, the Qwen3 decoder's
// per-head norm and rotary embedding (writing the keys and values to the beams' slot pool), attention (dense, and a decode
// step's over a slot pool shared by the beams; grouped-query), gathers - all float32 as the reference runs; the linear
// layers are the shared diffusion kernels' (cuBLAS, float32). nextsycl's skintokens.cpp in CUDA, the same names, arguments
// and arithmetic: its work-group reductions are block reductions (warp shuffles, then shared memory), its sub-group row
// dot a warp's, its local accessors shared memory, on the GPU's stream. Built without --use_fast_math (kernels/build.sh):
// the rotary angles reach hundreds of radians, where the fast sine is far off.
#include "ns.h"
#include "ns_internal.hpp"
#include "skintokens.h"

#include <algorithm>
#include <cmath>

namespace {

unsigned blocks(int64_t n, int t = 256) { return (unsigned) ((n + t - 1) / t); }
#define ID() const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x

__device__ inline float gelu_erf(float v) {
    return 0.5f * v * (1.0f + erff(v * 0.7071067811865476f));
}

// the sum / max over the block (blockDim.x a multiple of 32, at most 1024), every thread gets it; red holds 32 floats
__device__ float block_sum(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = 0.0f;
    for (int i = 0; i < (int) (blockDim.x >> 5); ++i) t += red[i];
    __syncthreads();
    return t;
}

__device__ float block_max(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = red[0];
    for (int i = 1; i < (int) (blockDim.x >> 5); ++i) t = fmaxf(t, red[i]);
    __syncthreads();
    return t;
}

constexpr int LN_WG = 256, LN_PER = 32;

__global__ void ln_k(const float* x, int64_t x_row, int64_t C, float eps, const float* w, const float* b, float* out) {
    __shared__ float red[32];
    const int64_t r = blockIdx.x;
    const int l = threadIdx.x;
    const float* row = x + r * x_row;
    float v[LN_PER];
    float s = 0.0f;
#pragma unroll
    for (int j = 0; j < LN_PER; ++j) {
        const int64_t i = (int64_t) j * LN_WG + l;
        v[j] = i < C ? row[i] : 0.0f;
        s += v[j];
    }
    const float mean = block_sum(s, red) / (float) C;
    float s2 = 0.0f;
#pragma unroll
    for (int j = 0; j < LN_PER; ++j) {
        const int64_t i = (int64_t) j * LN_WG + l;
        const float d = i < C ? v[j] - mean : 0.0f;
        s2 += d * d;
    }
    const float inv = rsqrtf(block_sum(s2, red) / (float) C + eps);
#pragma unroll
    for (int j = 0; j < LN_PER; ++j) {
        const int64_t i = (int64_t) j * LN_WG + l;
        if (i < C) {
            float y = (v[j] - mean) * inv;
            if (w) y = y * w[i];
            if (b) y = y + b[i];
            out[r * C + i] = y;
        }
    }
}

__global__ void rms_k(const float* x, int64_t x_row, int64_t C, float eps, const float* w, float* out) {
    __shared__ float red[32];
    const int64_t r = blockIdx.x;
    const int l = threadIdx.x;
    const float* row = x + r * x_row;
    float v[LN_PER];
    float s2 = 0.0f;
#pragma unroll
    for (int j = 0; j < LN_PER; ++j) {
        const int64_t i = (int64_t) j * LN_WG + l;
        v[j] = i < C ? row[i] : 0.0f;
        s2 += v[j] * v[j];
    }
    const float inv = rsqrtf(block_sum(s2, red) / (float) C + eps);
#pragma unroll
    for (int j = 0; j < LN_PER; ++j) {
        const int64_t i = (int64_t) j * LN_WG + l;
        if (i < C) out[r * C + i] = v[j] * inv * (w ? w[i] : 1.0f);
    }
}

__global__ void act_k(float* x, int64_t n, int kind) {
    ID();
    if (i >= n) return;
    const float v = x[i];
    x[i] = kind == 0 ? gelu_erf(v) : 1.0f / (1.0f + expf(-v));
}

__global__ void add_k(float* x, const float* y, int64_t n) { ID(); if (i < n) x[i] = x[i] + y[i]; }

__global__ void swiglu_k(const float* gu, int64_t I, float* out, int64_t n) {
    ID();
    if (i >= n) return;
    const int64_t m = i / I, c = i % I;
    const float a = gu[m * 2 * I + c], b = gu[m * 2 * I + I + c];
    out[i] = a / (1.0f + expf(-a)) * b;
}

// a warp a row
__global__ void rowdot_k(const float* x, int64_t M, int64_t C, const float* w, float b, int act, float* out) {
    const int l = threadIdx.x & 31;
    const int64_t r = (int64_t) blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (r >= M) return;
    float s = 0.0f;
    for (int64_t i = l; i < C; i += 32) s += x[r * C + i] * w[i];
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    s += b;
    if (l == 0) out[r] = act == 1 ? 1.0f / (1.0f + expf(-s)) : s;
}

__global__ void gather_k(const float* x, int64_t x_row, const int32_t* idx, int64_t C, float* out, int64_t n) {
    ID();
    if (i >= n) return;
    const int64_t r = i / C, c = i % C;
    out[i] = x[(int64_t) idx[r] * x_row + c];
}

constexpr int QK_MAXD = 128;

__global__ void qkv_post_k(float* qkv, int64_t M, int64_t Hq, int64_t Hkv, int64_t D, float eps, const float* qw, const float* kw, const int32_t* pos,
                           const float* inv_freq, const int32_t* slot, float* kpool, float* vpool) {
    const int64_t t = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nh = Hq + Hkv;
    if (t >= M * nh) return;
    const int64_t m = t / nh, h = t % nh;
    const int64_t row = (Hq + 2 * Hkv) * D, half = D / 2, kvrow = Hkv * D;
    const bool is_q = h < Hq;
    float* p = qkv + m * row + h * D;
    const float* w = is_q ? qw : kw;
    float v[QK_MAXD];
    float s2 = 0.0f;
    for (int64_t i = 0; i < D; ++i) {
        v[i] = p[i];
        s2 += v[i] * v[i];
    }
    const float inv = rsqrtf(s2 / (float) D + eps);
    for (int64_t i = 0; i < D; ++i) v[i] = v[i] * inv * w[i];
    const float tp = (float) pos[m];
    for (int64_t i = 0; i < half; ++i) {
        const float a = tp * inv_freq[i];
        float s, c;
        sincosf(a, &s, &c);
        const float x1 = v[i], x2 = v[i + half];
        p[i] = x1 * c - x2 * s;
        p[i + half] = x2 * c + x1 * s;
    }
    if (!is_q) {
        const int64_t kh = h - Hq;
        const int64_t at = (int64_t) slot[m] * kvrow + kh * D;
        const float* pv = qkv + m * row + (Hq + Hkv + kh) * D;
        for (int64_t i = 0; i < D; ++i) {
            kpool[at + i] = p[i];
            vpool[at + i] = pv[i];
        }
    }
}

constexpr int AT_WG = 256, AT_MAXD = 1024;

// a block a (batch, head, query row) of this pass; the row's scores in the scratch
__global__ void attention_k(const float* q, int64_t q_row, int64_t q_head, const float* k, int64_t k_row, int64_t k_head, const float* v,
                            int64_t v_row, int64_t v_head, int64_t Sq, int64_t Skv, int64_t H, int64_t group, int64_t D, int64_t Dv, float scale,
                            int causal, float* scratch, float* out, int64_t out_row, int64_t g0) {
    __shared__ float qv[AT_MAXD];
    __shared__ float red[32];
    const int64_t gr = blockIdx.x, gi = g0 + gr;
    const int64_t i = gi % Sq, hh = (gi / Sq) % H, b = gi / (Sq * H);
    const int64_t hk = hh / group;
    const int l = threadIdx.x;
    const float* qr = q + (b * Sq + i) * q_row + hh * q_head;
    for (int64_t d = l; d < D; d += AT_WG) qv[d] = qr[d];
    __syncthreads();
    const int64_t nk = causal ? i + 1 : Skv;
    float* sc = scratch + gr * Skv;
    float mx = -INFINITY;
    for (int64_t j = l; j < nk; j += AT_WG) {
        const float* kr = k + (b * Skv + j) * k_row + hk * k_head;
        float s = 0.0f;
        for (int64_t d = 0; d < D; ++d) s += qv[d] * kr[d];
        s *= scale;
        sc[j] = s;
        mx = fmaxf(mx, s);
    }
    mx = block_max(mx, red);
    float sum = 0.0f;
    for (int64_t j = l; j < nk; j += AT_WG) {
        const float e = expf(sc[j] - mx);
        sc[j] = e;
        sum += e;
    }
    sum = block_sum(sum, red);
    __threadfence_block();
    __syncthreads();
    const float inv = 1.0f / sum;
    for (int64_t d = l; d < Dv; d += AT_WG) {
        const float* vc = v + b * Skv * v_row + hk * v_head + d;
        float acc = 0.0f;
        for (int64_t j = 0; j < nk; ++j) acc += sc[j] * vc[j * v_row];
        out[(b * Sq + i) * out_row + hh * Dv + d] = acc * inv;
    }
}

constexpr int SL_WG = 256, SL_MAXL = 8192, SL_MAXD = 128;

__global__ void attn_slots_k(const float* q, int64_t q_row, const float* kpool, const float* vpool, const int32_t* slots, int64_t stride, int64_t len,
                             int64_t H, int64_t group, int64_t row, int64_t D, float scale, float* out) {
    __shared__ float sc[SL_MAXL];
    __shared__ float qv[SL_MAXD];
    __shared__ float red[32];
    const int64_t gi = blockIdx.x, b = gi / H, hh = gi % H, hk = hh / group;
    const int l = threadIdx.x;
    for (int64_t d = l; d < D; d += SL_WG) qv[d] = q[b * q_row + hh * D + d];
    __syncthreads();
    const int32_t* st = slots + b * stride;
    float mx = -INFINITY;
    for (int64_t t = l; t < len; t += SL_WG) {
        const float* kr = kpool + (int64_t) st[t] * row + hk * D;
        float s = 0.0f;
        for (int64_t d = 0; d < D; ++d) s += qv[d] * kr[d];
        s *= scale;
        sc[t] = s;
        mx = fmaxf(mx, s);
    }
    mx = block_max(mx, red);
    float sum = 0.0f;
    for (int64_t t = l; t < len; t += SL_WG) {
        const float e = expf(sc[t] - mx);
        sc[t] = e;
        sum += e;
    }
    sum = block_sum(sum, red);
    __syncthreads();
    const float inv = 1.0f / sum;
    for (int64_t d = l; d < D; d += SL_WG) {
        float acc = 0.0f;
        for (int64_t t = 0; t < len; ++t) acc += sc[t] * vpool[(int64_t) st[t] * row + hk * D + d];
        out[b * H * D + hh * D + d] = acc * inv;
    }
}

}  // namespace

extern "C" {

int ns_model3d_skintokens_ln(ns_gpu* g, const float* x, int64_t x_row, int64_t M, int64_t C, float eps, const float* w, const float* b, float* out) {
    if (C > (int64_t) LN_WG * LN_PER) return ns_fail("ns_model3d_skintokens_ln: rows longer than 8192");
    if (M <= 0 || C <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    ln_k<<<(unsigned) M, LN_WG, 0, g->q>>>(x, x_row, C, eps, w, b, out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_skintokens_rms(ns_gpu* g, const float* x, int64_t x_row, int64_t M, int64_t C, float eps, const float* w, float* out) {
    if (C > (int64_t) LN_WG * LN_PER) return ns_fail("ns_model3d_skintokens_rms: rows longer than 8192");
    if (M <= 0 || C <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    rms_k<<<(unsigned) M, LN_WG, 0, g->q>>>(x, x_row, C, eps, w, out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_skintokens_act(ns_gpu* g, float* x, int64_t n, int kind) {
    if (kind < 0 || kind > 1) return ns_fail("ns_model3d_skintokens_act: kind 0 (gelu) or 1 (sigmoid)");
    if (n <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    act_k<<<blocks(n), 256, 0, g->q>>>(x, n, kind);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_skintokens_add(ns_gpu* g, float* x, const float* y, int64_t n) {
    if (n <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    add_k<<<blocks(n), 256, 0, g->q>>>(x, y, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_skintokens_swiglu(ns_gpu* g, const float* gu, int64_t M, int64_t I, float* out) {
    if (M <= 0 || I <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    swiglu_k<<<blocks(M * I), 256, 0, g->q>>>(gu, I, out, M * I);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_skintokens_rowdot(ns_gpu* g, const float* x, int64_t M, int64_t C, const float* w, float b, int act, float* out) {
    if (M <= 0 || C <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    constexpr int ROWS = 8;  // warps a block
    rowdot_k<<<(unsigned) ((M + ROWS - 1) / ROWS), ROWS * 32, 0, g->q>>>(x, M, C, w, b, act, out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_skintokens_gather(ns_gpu* g, const float* x, int64_t x_row, const int32_t* idx, int64_t R, int64_t C, float* out) {
    if (R <= 0 || C <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    gather_k<<<blocks(R * C), 256, 0, g->q>>>(x, x_row, idx, C, out, R * C);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_skintokens_qkv_post(ns_gpu* g, float* qkv, int64_t M, int64_t Hq, int64_t Hkv, int64_t D, float eps, const float* qw, const float* kw,
                                   const int32_t* pos, const float* inv_freq, const int32_t* slot, float* kpool, float* vpool) {
    if (D <= 0 || D > QK_MAXD || (D & 1)) return ns_fail("ns_model3d_skintokens_qkv_post: D even, at most 128");
    if (M <= 0 || Hq + Hkv <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    // a thread a (row, head): few threads, each with a head's 128 values in registers (64 a block keeps them there)
    qkv_post_k<<<blocks(M * (Hq + Hkv), 64), 64, 0, g->q>>>(qkv, M, Hq, Hkv, D, eps, qw, kw, pos, inv_freq, slot, kpool, vpool);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_skintokens_attention(ns_gpu* g, const float* q, int64_t q_row, int64_t q_head, const float* k, int64_t k_row, int64_t k_head,
                                    const float* v, int64_t v_row, int64_t v_head, int64_t B, int64_t Sq, int64_t Skv, int64_t H, int64_t Hkv,
                                    int64_t D, int64_t Dv, float scale, int causal, float* scratch, int64_t scratch_n, float* out, int64_t out_row) {
    if (B <= 0 || Sq <= 0 || Skv <= 0 || H <= 0) return 0;
    if (Hkv <= 0 || H % Hkv) return ns_fail("ns_model3d_skintokens_attention: the heads must be a multiple of the key / value heads");
    if (D > AT_MAXD || D <= 0 || Dv <= 0) return ns_fail("ns_model3d_skintokens_attention: 0 < D <= 1024, Dv > 0");
    if (scratch_n < Skv) return ns_fail("ns_model3d_skintokens_attention: the scratch holds less than one row of scores");
    if (causal && Sq != Skv) return ns_fail("ns_model3d_skintokens_attention: a causal mask needs Sq = Skv");
    NS_CU(cudaSetDevice(g->dev));
    const int64_t total = B * H * Sq;
    const int64_t per = std::max<int64_t>(1, std::min<int64_t>({total, scratch_n / Skv, (int64_t) 0x7fffffff}));
    for (int64_t g0 = 0; g0 < total; g0 += per) {
        const int64_t n = std::min(per, total - g0);
        attention_k<<<(unsigned) n, AT_WG, 0, g->q>>>(q, q_row, q_head, k, k_row, k_head, v, v_row, v_head, Sq, Skv, H, H / Hkv, D, Dv, scale, causal,
                                                     scratch, out, out_row, g0);
        NS_LAUNCH_CHECK();
    }
    return 0;
}

int ns_model3d_skintokens_attn_slots(ns_gpu* g, const float* q, int64_t q_row, const float* kpool, const float* vpool, const int32_t* slots, int64_t stride,
                                     int64_t len, int64_t B, int64_t H, int64_t Hkv, int64_t D, float scale, float* out) {
    if (len <= 0 || B <= 0) return 0;
    if (len > SL_MAXL || D > SL_MAXD || D <= 0) return ns_fail("ns_model3d_skintokens_attn_slots: len <= 8192, 0 < D <= 128");
    if (Hkv <= 0 || H % Hkv) return ns_fail("ns_model3d_skintokens_attn_slots: the heads must be a multiple of the key / value heads");
    if (H <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    attn_slots_k<<<(unsigned) (B * H), SL_WG, 0, g->q>>>(q, q_row, kpool, vpool, slots, stride, len, H, H / Hkv, Hkv * D, D, scale, out);
    NS_LAUNCH_CHECK();
    return 0;
}

}  // extern "C"
