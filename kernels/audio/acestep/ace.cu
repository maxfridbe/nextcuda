// ace.cu: the ACE-Step 1.5 engine's own kernels - what the shared diffusion kernels (nsd) do not have: the DiT's and
// its encoders' sliding-window attention (bidirectional: a row reads the keys within 128 of it on either side), and
// the Oobleck VAE's snake, whose divisor is a parameter of its own (beta) rather than the frequency (alpha).
#include "ns.h"
#include "ns_internal.hpp"
#include "ace.h"

#include <cuda_bf16.h>

#include <cmath>

namespace {

constexpr int WARP = 32;
constexpr int HD = 128;          // the head size of every ACE-Step attention
constexpr int ROWS = 4;          // query rows a block (a warp each)

__device__ inline float bf(uint16_t b) { return __uint_as_float((uint32_t) b << 16); }

__device__ inline uint16_t to_bf(float f) {
    // round to nearest even, NaN kept a NaN
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t) ((u >> 16) | 0x40);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}

__device__ inline float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

unsigned blocks(int64_t n, int t = 256) { return (unsigned) ((n + t - 1) / t); }

// a warp a query row, each lane 4 of the head's 128 features; online softmax over the window's keys - or, seg > 0,
// over the keys of the row's own segment (rows [seg i/seg, seg i/seg + seg): sequences of seg rows side by side)
__global__ void attn_window_k(const uint16_t* q, int64_t qs, const uint16_t* k, const uint16_t* v, int64_t kvs, int64_t S, int64_t Hq, int64_t Hkv,
                              int64_t w, int64_t seg, float scale, uint16_t* out, int64_t os) {
    const int lane = threadIdx.x & (WARP - 1), warp = threadIdx.x / WARP;
    const int64_t i = (int64_t) blockIdx.x * ROWS + warp, h = blockIdx.y;
    if (i >= S) return;
    const int64_t hk = h / (Hq / Hkv);
    const uint16_t* qp = q + i * qs + h * HD + lane * 4;
    float qv[4];
#pragma unroll
    for (int d = 0; d < 4; ++d) qv[d] = bf(qp[d]) * scale;
    int64_t j0 = w < 0 ? 0 : (i - w > 0 ? i - w : 0);
    int64_t j1 = w < 0 ? S - 1 : (i + w < S - 1 ? i + w : S - 1);
    if (seg > 0) {
        j0 = i / seg * seg;
        j1 = j0 + seg - 1 < S - 1 ? j0 + seg - 1 : S - 1;
    }
    float m = -INFINITY, l = 0.0f, o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int64_t j = j0; j <= j1; ++j) {
        const uint16_t* kp = k + j * kvs + hk * HD + lane * 4;
        const uint2 kr = *reinterpret_cast<const uint2*>(kp);
        const uint16_t* kk = reinterpret_cast<const uint16_t*>(&kr);
        float s = qv[0] * bf(kk[0]) + qv[1] * bf(kk[1]) + qv[2] * bf(kk[2]) + qv[3] * bf(kk[3]);
        s = warp_sum(s);
        const float mn = fmaxf(m, s);
        const float corr = __expf(m - mn), p = __expf(s - mn);
        const uint2 vr = *reinterpret_cast<const uint2*>(v + j * kvs + hk * HD + lane * 4);
        const uint16_t* vv = reinterpret_cast<const uint16_t*>(&vr);
        l = l * corr + p;
#pragma unroll
        for (int d = 0; d < 4; ++d) o[d] = o[d] * corr + p * bf(vv[d]);
        m = mn;
    }
    const float inv = l > 0.0f ? 1.0f / l : 0.0f;
    uint16_t* op = out + i * os + h * HD + lane * 4;
#pragma unroll
    for (int d = 0; d < 4; ++d) op[d] = to_bf(o[d] * inv);
}

__global__ void snake_k(const float* x, int64_t L, const float* alpha, const float* inv_beta, float* out, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t c = i / L;
    const float v = x[i], s = sinf(alpha[c] * v);
    out[i] = v + inv_beta[c] * s * s;
}

}  // namespace

extern "C" {

int ns_audio_ace_attn_window(ns_gpu* g, const void* q, int64_t qs, const void* k, const void* v, int64_t kvs, int64_t S, int64_t Hq, int64_t Hkv,
                             int64_t w, float scale, void* out, int64_t os) {
    if (S <= 0) return 0;
    if (Hkv <= 0 || Hq % Hkv) return ns_fail("ns_audio_ace_attn_window: query heads are not a multiple of the key heads");
    if (qs % 4 || kvs % 4 || os % 4) return ns_fail("ns_audio_ace_attn_window: row strides must be multiples of 4 values");
    NS_CU(cudaSetDevice(g->dev));
    const dim3 grid((unsigned) ((S + ROWS - 1) / ROWS), (unsigned) Hq);
    attn_window_k<<<grid, ROWS * WARP, 0, g->q>>>((const uint16_t*) q, qs, (const uint16_t*) k, (const uint16_t*) v, kvs, S, Hq, Hkv, w, 0, scale,
                                                 (uint16_t*) out, os);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_ace_attn_segments(ns_gpu* g, const void* q, int64_t qs, const void* k, const void* v, int64_t kvs, int64_t S, int64_t Hq, int64_t Hkv,
                               int64_t seg, float scale, void* out, int64_t os) {
    if (S <= 0) return 0;
    if (seg <= 0) return ns_fail("ns_audio_ace_attn_segments: a segment of at least one row");
    if (Hkv <= 0 || Hq % Hkv) return ns_fail("ns_audio_ace_attn_segments: query heads are not a multiple of the key heads");
    if (qs % 4 || kvs % 4 || os % 4) return ns_fail("ns_audio_ace_attn_segments: row strides must be multiples of 4 values");
    NS_CU(cudaSetDevice(g->dev));
    const dim3 grid((unsigned) ((S + ROWS - 1) / ROWS), (unsigned) Hq);
    attn_window_k<<<grid, ROWS * WARP, 0, g->q>>>((const uint16_t*) q, qs, (const uint16_t*) k, (const uint16_t*) v, kvs, S, Hq, Hkv, -1, seg, scale,
                                                 (uint16_t*) out, os);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_ace_snake(ns_gpu* g, const float* x, int64_t C, int64_t L, const float* alpha, const float* inv_beta, float* out) {
    const int64_t n = C * L;
    if (n <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    snake_k<<<blocks(n), 256, 0, g->q>>>(x, L, alpha, inv_beta, out, n);
    NS_LAUNCH_CHECK();
    return 0;
}

}  // extern "C"
