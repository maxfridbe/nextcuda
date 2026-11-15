// vae.cu: the VAE's normalization in Qwen-Image 2.1's silo - out = silu(rms norm(x) * gamma) in one pass over the
// pixels (the shared kernels make the norm's half output, then read it again for the silu: twice the traffic at
// 2048 x 2048, where these passes are memory bound). Channels last, half in and out, a warp a pixel.
#include "ns.h"
#include "ns_internal.hpp"

#include <cuda_fp16.h>
#include <cstdint>

namespace {

__global__ void rms_silu_k(const __half* __restrict__ x, long P, int C, const float* __restrict__ gamma, float eps, __half* __restrict__ out) {
    const long p = (long) blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    if (p >= P) return;
    const __half2* row = reinterpret_cast<const __half2*>(x + p * C);
    float ss = 0.0f;
    for (int c = lane; c < C / 2; c += 32) {
        const float2 v = __half22float2(row[c]);
        ss += v.x * v.x + v.y * v.y;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    const float inv = rsqrtf(ss / (float) C + eps);
    __half2* orow = reinterpret_cast<__half2*>(out + p * C);
    for (int c = lane; c < C / 2; c += 32) {
        const float2 v = __half22float2(row[c]);
        float a = v.x * inv * gamma[2 * c], b = v.y * inv * gamma[2 * c + 1];
        a = a / (1.0f + __expf(-a));
        b = b / (1.0f + __expf(-b));
        orow[c] = __floats2half2_rn(a, b);
    }
}

}  // namespace

extern "C" {

// out [P, C] half = silu(x * rsqrt(mean(x^2) + eps) * gamma) per pixel of x [P, C] half; C even
int ns_image_qi21_silo_rms_silu(ns_gpu* g, const void* x, int64_t P, int64_t C, const float* gamma, float eps, void* out) {
    if (C % 2) return ns_fail("silo rms_silu: an even channel count");
    NS_CU(cudaSetDevice(g->dev));
    if (P <= 0) return 0;
    rms_silu_k<<<(unsigned) ((P + 7) / 8), 256, 0, g->q>>>((const __half*) x, (long) P, (int) C, gamma, eps, (__half*) out);
    NS_LAUNCH_CHECK();
    return 0;
}

}  // extern "C"
