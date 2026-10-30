// lx.cu: the LTX-2.5 engine's own small kernels - what the shared diffusion kernels (nsd) do not have: rows to half at a
// scale (the products past half's range), the gated GELUs (Gemma's MLP, the DiT's feed-forwards), the attention's
// per-head gates, the prompt's modulation and the DiT's output norm. Elementwise or one block a row, on the GPU's stream.
#include "ns.h"
#include "ns_internal.hpp"
#include "lx.h"

#include <cuda_fp16.h>

namespace {

constexpr int WG = 256;
constexpr int PER = 32;  // a row of up to 8,192 values held in registers

__device__ inline __half f2h(float f) { return __float2half_rn(f); }

__device__ float block_sum(float v) {
    __shared__ float s[WG / 32];
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0) s[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = 0.0f;
    for (int i = 0; i < WG / 32; ++i) t += s[i];
    __syncthreads();
    return t;
}

// gelu with the tanh approximation, in float32 (torch's gelu(approximate="tanh"), transformers' gelu_pytorch_tanh)
__device__ inline float gelu_tanh(float x) {
    const float k = 0.7978845608028654f;  // sqrt(2 / pi)
    return 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
}

__global__ void to_half_rows_k(const float* x, int64_t C, int64_t ldx, int64_t n, float s, __half* o, int64_t ldo) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t r = i / C, c = i % C;
    o[r * ldo + c] = f2h(x[r * ldx + c] * s);
}

__global__ void to_float_k(const __half* x, float* o, int64_t n, float s) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = __half2float(x[i]) * s;
}

__global__ void geglu_k(const float* a, int64_t lda, const float* b, int64_t ldb, int64_t C, int64_t n, float s, __half* o) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t r = i / C, c = i % C;
    float v = gelu_tanh(a[r * lda + c]);
    if (b) v *= b[r * ldb + c];
    o[i] = f2h(v * s);
}

__global__ void axpy_k(float* x, const float* y, int64_t n, float a) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += a * y[i];
}

__global__ void head_gate_k(__half* x, int64_t H, int64_t D, int64_t n, const float* logits) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t rh = i / D;  // (row, head)
    const float g = 2.0f / (1.0f + expf(-logits[rh]));
    x[i] = f2h(__half2float(x[i]) * g);
}

__global__ void mod_rows_k(const float* x, int64_t C, int64_t n, const float* scale, const float* shift, float s, __half* o) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t c = i % C;
    float v = x[i] * (1.0f + (scale ? scale[c] : 0.0f)) + (shift ? shift[c] : 0.0f);
    o[i] = f2h(v * s);
}

__global__ void ln_mod_k(const float* x, int64_t C, float eps, const int32_t* rows, const float* scale, const float* shift, float* o) {
    const size_t r = blockIdx.x;
    const int l = threadIdx.x;
    const float* row = x + r * C;
    float v[PER];
    float s = 0.0f;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int64_t i = (int64_t) j * WG + l;
        v[j] = i < C ? row[i] : 0.0f;
        s += v[j];
    }
    const float mean = block_sum(s) / (float) C;
    float s2 = 0.0f;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int64_t i = (int64_t) j * WG + l;
        const float d = i < C ? v[j] - mean : 0.0f;
        s2 += d * d;
    }
    const float inv = rsqrtf(block_sum(s2) / (float) C + eps);
    const int64_t t = rows ? rows[r] : 0;
    const float* sc = scale ? scale + t * C : nullptr;
    const float* sh = shift ? shift + t * C : nullptr;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int64_t i = (int64_t) j * WG + l;
        if (i < C) {
            float y = (v[j] - mean) * inv;
            if (sc) y *= 1.0f + sc[i];
            if (sh) y += sh[i];
            o[r * C + i] = y;
        }
    }
}

__device__ inline float ld(const __half* p, int64_t i) { return __half2float(p[i]); }
__device__ inline float ld(const float* p, int64_t i) { return p[i]; }
__device__ inline void st(__half* p, int64_t i, float v) { p[i] = f2h(v); }
__device__ inline void st(float* p, int64_t i, float v) { p[i] = v; }

template <typename T>
__global__ void rms_silu_k(const T* x, int64_t C, float eps, int silu, __half* o) {
    const size_t r = blockIdx.x;
    const T* row = x + r * C;
    // the mean square in float32 over values that may pass half's range: divided as it goes
    float s = 0.0f;
    for (int64_t i = threadIdx.x; i < C; i += WG) {
        const float v = ld(row, i);
        s += v * v / (float) C;
    }
    const float inv = 1.0f / sqrtf(block_sum(s) + eps);
    for (int64_t i = threadIdx.x; i < C; i += WG) {
        float v = ld(row, i) * inv;
        if (silu) v = v / (1.0f + expf(-v));
        o[r * C + i] = f2h(v);
    }
}

__global__ void d2s_k(const __half* x, int64_t T, int64_t H, int64_t W, int64_t C, int s0, int s1, int s2, int drop, int64_t n, __half* o) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // i over the output [To, Ho, Wo, C]
    const int64_t Ho = H * s1, Wo = W * s2;
    int64_t c = i % C, r = i / C;
    const int64_t wo = r % Wo; r /= Wo;
    const int64_t ho = r % Ho;
    const int64_t to = r / Ho + drop;
    const int64_t t = to / s0, a = to % s0, h = ho / s1, b = ho % s1, w = wo / s2, k = wo % s2;
    const int64_t ci = ((c * s0 + a) * s1 + b) * s2 + k;
    o[i] = x[((t * H + h) * W + w) * (C * s0 * s1 * s2) + ci];
}

__global__ void up2_k(const __half* x, int64_t H, int64_t W, int64_t C, int64_t n, __half* o) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t c = i % C, r = i / C, w = r % (2 * W), h = r / (2 * W);
    o[i] = x[((h / 2) * W + w / 2) * C + c];
}

template <typename T>
__global__ void s2d_k(const T* x, int64_t T_, int64_t H, int64_t W, int64_t C, int s0, int s1, int s2, int G, int64_t n, T* o) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t Ho = H / s1, Wo = W / s2, Co = C * s0 * s1 * s2 / G;
    int64_t j = i % Co, r = i / Co;
    const int64_t wo = r % Wo; r /= Wo;
    const int64_t ho = r % Ho;
    const int64_t to = r / Ho;
    float acc = 0.0f;
    for (int g = 0; g < G; ++g) {
        int64_t q = j * G + g;
        const int64_t k = q % s2; q /= s2;
        const int64_t b = q % s1; q /= s1;
        const int64_t a = q % s0;
        const int64_t c = q / s0;
        acc += ld(x, (((to * s0 + a) * H + ho * s1 + b) * W + wo * s2 + k) * C + c);
    }
    st(o, i, acc / (float) G);
}

__global__ void add_bias_k(float* x, int64_t C, int64_t n, const float* b) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += b[i % C];
}

// one block a group: its mean and 1 / std over N x (C / G) values (two passes, float32)
__global__ void gn_stat_k(const __half* x, int64_t N, int64_t C, int64_t cg, float eps, float* stat) {
    const int64_t g = blockIdx.x;
    const int64_t n = N * cg;
    float s = 0.0f;
    for (int64_t i = threadIdx.x; i < n; i += WG) s += __half2float(x[(i / cg) * C + g * cg + i % cg]);
    const float mean = block_sum(s) / (float) n;
    float s2 = 0.0f;
    for (int64_t i = threadIdx.x; i < n; i += WG) {
        const float d = __half2float(x[(i / cg) * C + g * cg + i % cg]) - mean;
        s2 += d * d;
    }
    const float var = block_sum(s2) / (float) n;
    if (threadIdx.x == 0) {
        stat[g * 2] = mean;
        stat[g * 2 + 1] = rsqrtf(var + eps);
    }
}

__global__ void gn_apply_k(const __half* x, int64_t C, int64_t cg, int64_t n, const float* stat, const float* w, const float* b, const __half* res,
                           int silu, __half* o) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t c = i % C, g = c / cg;
    float v = (__half2float(x[i]) - stat[g * 2]) * stat[g * 2 + 1] * w[c] + b[c];
    if (res) v += __half2float(res[i]);
    if (silu) v = v / (1.0f + expf(-v));
    o[i] = f2h(v);
}

inline unsigned blocks(int64_t n) { return (unsigned) ((n + 255) / 256); }

}  // namespace

extern "C" {

int ns_video_lx_to_half_rows(ns_gpu* g, const float* x, int64_t M, int64_t C, int64_t ldx, float s, void* out, int64_t ldo) {
    if (M * C == 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    to_half_rows_k<<<blocks(M * C), 256, 0, g->q>>>(x, C, ldx, M * C, s, (__half*) out, ldo);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_video_lx_to_float(ns_gpu* g, const void* x, float* out, int64_t n, float s) {
    if (n == 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    to_float_k<<<blocks(n), 256, 0, g->q>>>((const __half*) x, out, n, s);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_video_lx_geglu(ns_gpu* g, const float* a, int64_t lda, const float* b, int64_t ldb, int64_t M, int64_t C, float s, void* out) {
    if (M * C == 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    geglu_k<<<blocks(M * C), 256, 0, g->q>>>(a, lda, b, ldb, C, M * C, s, (__half*) out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_video_lx_axpy(ns_gpu* g, float* x, const float* y, int64_t n, float a) {
    if (n == 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    axpy_k<<<blocks(n), 256, 0, g->q>>>(x, y, n, a);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_video_lx_head_gate(ns_gpu* g, void* x, int64_t M, int64_t H, int64_t D, const float* logits) {
    if (M * H * D == 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    head_gate_k<<<blocks(M * H * D), 256, 0, g->q>>>((__half*) x, H, D, M * H * D, logits);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_video_lx_mod_rows(ns_gpu* g, const float* x, int64_t M, int64_t C, const float* scale, const float* shift, float s, void* out) {
    if (M * C == 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    mod_rows_k<<<blocks(M * C), 256, 0, g->q>>>(x, C, M * C, scale, shift, s, (__half*) out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_video_lx_ln_mod(ns_gpu* g, const float* x, int64_t M, int64_t C, float eps, const int32_t* rows, const float* scale,
                       const float* shift, float* out) {
    if (C > (int64_t) WG * PER) return ns_fail("ns_video_lx_ln_mod: rows longer than 8192");
    if (M == 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    ln_mod_k<<<(unsigned) M, WG, 0, g->q>>>(x, C, eps, rows, scale, shift, out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_video_lx_rms_silu(ns_gpu* g, const void* x, int64_t M, int64_t C, float eps, int silu, void* out) {
    if (M == 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    rms_silu_k<<<(unsigned) M, WG, 0, g->q>>>((const __half*) x, C, eps, silu, (__half*) out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_video_lx_rms_silu_f32(ns_gpu* g, const float* x, int64_t M, int64_t C, float eps, int silu, void* out) {
    if (M == 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    rms_silu_k<<<(unsigned) M, WG, 0, g->q>>>(x, C, eps, silu, (__half*) out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_video_lx_add_bias(ns_gpu* g, float* x, int64_t M, int64_t C, const float* b) {
    if (M * C == 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    add_bias_k<<<blocks(M * C), 256, 0, g->q>>>(x, C, M * C, b);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_video_lx_d2s(ns_gpu* g, const void* x, int64_t T, int64_t H, int64_t W, int64_t C, int s0, int s1, int s2, int drop, void* out) {
    const int64_t n = (T * s0 - drop) * H * s1 * W * s2 * C;
    if (n <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    d2s_k<<<blocks(n), 256, 0, g->q>>>((const __half*) x, T, H, W, C, s0, s1, s2, drop, n, (__half*) out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_video_lx_s2d(ns_gpu* g, const void* x, int64_t T, int64_t H, int64_t W, int64_t C, int s0, int s1, int s2, int G, void* out) {
    const int64_t n = (T / s0) * (H / s1) * (W / s2) * (C * s0 * s1 * s2 / G);
    if (n <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    s2d_k<<<blocks(n), 256, 0, g->q>>>((const __half*) x, T, H, W, C, s0, s1, s2, G, n, (__half*) out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_video_lx_s2d_f32(ns_gpu* g, const float* x, int64_t T, int64_t H, int64_t W, int64_t C, int s0, int s1, int s2, int G, float* out) {
    const int64_t n = (T / s0) * (H / s1) * (W / s2) * (C * s0 * s1 * s2 / G);
    if (n <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    s2d_k<<<blocks(n), 256, 0, g->q>>>(x, T, H, W, C, s0, s1, s2, G, n, out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_video_lx_group_norm(ns_gpu* g, const void* x, int64_t N, int64_t C, int64_t G, const float* w, const float* b, float eps, const void* res,
                           int silu, void* out) {
    if (N * C == 0) return 0;
    if (C % G) return ns_fail("ns_video_lx_group_norm: channels not a multiple of the groups");
    NS_CU(cudaSetDevice(g->dev));
    float* stat = nullptr;
    NS_CU(cudaMallocAsync((void**) &stat, (size_t) G * 2 * sizeof(float), g->q));
    gn_stat_k<<<(unsigned) G, WG, 0, g->q>>>((const __half*) x, N, C, C / G, eps, stat);
    gn_apply_k<<<blocks(N * C), 256, 0, g->q>>>((const __half*) x, C, C / G, N * C, stat, w, b, (const __half*) res, silu, (__half*) out);
    NS_LAUNCH_CHECK();
    NS_CU(cudaFreeAsync(stat, g->q));
    return 0;
}

int ns_video_lx_up2(ns_gpu* g, const void* x, int64_t H, int64_t W, int64_t C, void* out) {
    const int64_t n = 4 * H * W * C;
    if (n == 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    up2_k<<<blocks(n), 256, 0, g->q>>>((const __half*) x, H, W, C, n, (__half*) out);
    NS_LAUNCH_CHECK();
    return 0;
}

}  // extern "C"
