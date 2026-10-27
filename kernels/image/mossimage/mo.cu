// mo.cu: the MOSS-Image engine's own kernels - what the shared diffusion kernels (nsd) and Ming-Image's (mi) do not
// have: the MLPs' clipped-gate activation and the VAE attention's group norm without its SiLU. On the GPU's stream.
#include "ns.h"
#include "ns_internal.hpp"
#include "mo.h"

#include <cuda_fp16.h>

namespace {

constexpr int WG = 256;

unsigned blocks(int64_t n) {
    return (unsigned) ((n + WG - 1) / WG);
}

__global__ void situ_k(const float* x, int64_t F, int64_t ld, int64_t n, __half* o) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t r = i / F, c = i % F;
    const float h = x[r * ld + c], g = x[r * ld + F + c];
    const float gate = 4.0f * tanhf(g * 0.25f) / (1.0f + expf(-g));
    o[i] = __float2half_rn(gate * 25.0f * tanhf(h * 0.04f));
}

// one block a group: its mean and variance over P pixels x C / G channels (double sums: a group at 1024 x 1024 is
// a few million values), then the group normalized
__global__ void group_norm_k(const __half* x, int64_t P, int64_t C, int64_t G, const float* w, const float* b, float eps, __half* o) {
    const int64_t gi = blockIdx.x;
    const int64_t cg = C / G;
    const int64_t n = P * cg;
    double s = 0.0, s2 = 0.0;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        const int64_t p = i / cg, c = gi * cg + i % cg;
        const double v = __half2float(x[p * C + c]);
        s += v;
        s2 += v * v;
    }
    __shared__ double a[WG], q[WG];
    a[threadIdx.x] = s;
    q[threadIdx.x] = s2;
    __syncthreads();
    for (int k = WG / 2; k > 0; k >>= 1) {
        if (threadIdx.x < k) {
            a[threadIdx.x] += a[threadIdx.x + k];
            q[threadIdx.x] += q[threadIdx.x + k];
        }
        __syncthreads();
    }
    const double mean = a[0] / (double) n;
    const double var = q[0] / (double) n - mean * mean;
    const float inv = (float) (1.0 / sqrt((var > 0.0 ? var : 0.0) + (double) eps));
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        const int64_t p = i / cg, c = gi * cg + i % cg;
        const float v = (__half2float(x[p * C + c]) - (float) mean) * inv;
        o[p * C + c] = __float2half_rn(v * (w ? w[c] : 1.0f) + (b ? b[c] : 0.0f));
    }
}

}  // namespace

extern "C" {

int ns_image_mo_situ(ns_gpu* g, const float* x, int64_t M, int64_t F, int64_t ld, void* out) {
    NS_CU(cudaSetDevice(g->dev));
    situ_k<<<blocks(M * F), WG, 0, g->q>>>(x, F, ld, M * F, (__half*) out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_mo_group_norm(ns_gpu* g, const void* x, int64_t P, int64_t C, int64_t G, const float* w, const float* b, float eps, void* out) {
    NS_CU(cudaSetDevice(g->dev));
    if (G <= 0 || C % G) return ns_fail("group_norm: channels not a multiple of the groups");
    group_norm_k<<<(unsigned) G, WG, 0, g->q>>>((const __half*) x, P, C, G, w, b, eps, (__half*) out);
    NS_LAUNCH_CHECK();
    return 0;
}

}  // extern "C"
