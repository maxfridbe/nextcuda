// zimage.cu: Z-Image's DiT's own small kernels (zimage.h) - the modulated norms and tanh-gated residuals, the scaled
// activations, the sampler's step. They came from Ming-Image's library (kernels/image/mingimage) when its DiT moved to
// the foundation so Z-Image-Turbo could share it. Elementwise or one block a row, on the GPU's stream.
#include "ns.h"
#include "ns_internal.hpp"
#include "zimage.h"

#include <cuda_fp16.h>

namespace {

constexpr int WG = 256;
constexpr int PER = 32;  // a row of up to 8,192 values held in registers

__device__ inline float h2f(__half h) { return __half2float(h); }
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

__global__ void ln_mod_k(const float* x, int64_t C, float eps, const float* scale, __half* o) {
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
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int64_t i = (int64_t) j * WG + l;
        if (i < C) o[r * C + i] = f2h((v[j] - mean) * inv * (scale ? 1.0f + scale[i] : 1.0f));
    }
}

__global__ void rms_mod_k(const float* x, int64_t C, const float* w, float eps, const float* scale, __half* o) {
    const size_t r = blockIdx.x;
    const int l = threadIdx.x;
    const float* row = x + r * C;
    float v[PER];
    float s = 0.0f;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int64_t i = (int64_t) j * WG + l;
        v[j] = i < C ? row[i] : 0.0f;
        s += v[j] * v[j];
    }
    const float inv = rsqrtf(block_sum(s) / (float) C + eps);
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int64_t i = (int64_t) j * WG + l;
        if (i < C) o[r * C + i] = f2h(v[j] * inv * (w ? w[i] : 1.0f) * (scale ? 1.0f + scale[i] : 1.0f));
    }
}

__global__ void gate_add_k(float* x, int64_t C, int64_t n, const __half* o, const float* gate) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] += h2f(o[i]) * (gate ? gate[i % C] : 1.0f);
}

__global__ void swiglu_scaled_k(const __half* x, int64_t C, int64_t n, float s, __half* o) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t r = i / C, c = i % C;
    const float a = h2f(x[r * 2 * C + c]), b = h2f(x[r * 2 * C + C + c]);
    o[i] = f2h(a / (1.0f + expf(-a)) * b * s);
}

__global__ void to_half_rows_k(const float* x, int64_t C, int64_t n, float s, __half* o, int64_t ld) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    o[(i / C) * ld + i % C] = f2h(x[i] * s);
}

__global__ void to_float_k(const __half* p, float* o, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = h2f(p[i]);
}

__global__ void axpy_k(float* x, const float* y, int64_t n, float a) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += a * y[i];
}

inline unsigned blocks(int64_t n) { return (unsigned) ((n + 255) / 256); }

}  // namespace

extern "C" {

int ns_zimage_ln_mod(ns_gpu* g, const float* x, int64_t M, int64_t C, float eps, const float* scale, void* out) {
    if (C > (int64_t) WG * PER) return ns_fail("ns_zimage_ln_mod: rows longer than 8192");
    if (M <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    ln_mod_k<<<(unsigned) M, WG, 0, g->q>>>(x, C, eps, scale, (__half*) out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_zimage_rms_mod(ns_gpu* g, const float* x, int64_t M, int64_t C, const float* w, float eps, const float* scale, void* out) {
    if (C > (int64_t) WG * PER) return ns_fail("ns_zimage_rms_mod: rows longer than 8192");
    if (M <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    rms_mod_k<<<(unsigned) M, WG, 0, g->q>>>(x, C, w, eps, scale, (__half*) out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_zimage_gate_add(ns_gpu* g, float* x, int64_t M, int64_t C, const void* other, const float* gate) {
    if (M * C <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    gate_add_k<<<blocks(M * C), 256, 0, g->q>>>(x, C, M * C, (const __half*) other, gate);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_zimage_swiglu_scaled(ns_gpu* g, const void* x, int64_t M, int64_t C, float s, void* out) {
    if (M * C <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    swiglu_scaled_k<<<blocks(M * C), 256, 0, g->q>>>((const __half*) x, C, M * C, s, (__half*) out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_zimage_to_half_rows(ns_gpu* g, const float* x, int64_t M, int64_t C, float s, void* out, int64_t ld) {
    if (M * C <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    to_half_rows_k<<<blocks(M * C), 256, 0, g->q>>>(x, C, M * C, s, (__half*) out, ld);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_zimage_to_float(ns_gpu* g, const void* x, float* out, int64_t n) {
    if (n <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    to_float_k<<<blocks(n), 256, 0, g->q>>>((const __half*) x, out, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_zimage_axpy(ns_gpu* g, float* x, const float* y, int64_t n, float a) {
    if (n <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    axpy_k<<<blocks(n), 256, 0, g->q>>>(x, y, n, a);
    NS_LAUNCH_CHECK();
    return 0;
}

}  // extern "C"
