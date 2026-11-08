// zi.cu: the Z-Image-Turbo engine's own small kernels (zi.h) - its VAE decoder's group norm before the middle attention
// and its nearest 2x upsampling, on the GPU's stream.
#include "ns.h"
#include "ns_internal.hpp"
#include "zi.h"

#include <cuda_fp16.h>

namespace {

constexpr int WG = 256;

// one block a group: the sums over the group's P x C / G values (double: a group holds up to millions), then the
// normalized values
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

__global__ void up2_k(const __half* s, __half* o, int64_t H, int64_t W, int64_t C) {
    const int64_t n = 2 * H * 2 * W * C;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t c = i % C, xx = (i / C) % (2 * W), y = i / (C * 2 * W);
    o[i] = s[((y / 2) * W + xx / 2) * C + c];
}

inline unsigned blocks(int64_t n) { return (unsigned) ((n + 255) / 256); }

}  // namespace

extern "C" {

int ns_image_zi_group_norm(ns_gpu* g, const void* x, int64_t P, int64_t C, int64_t G, const float* w, const float* b, float eps, void* out) {
    if (G <= 0 || C % G) return ns_fail("ns_image_zi_group_norm: channels not a multiple of the groups");
    NS_CU(cudaSetDevice(g->dev));
    group_norm_k<<<(unsigned) G, WG, 0, g->q>>>((const __half*) x, P, C, G, w, b, eps, (__half*) out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_zi_up2(ns_gpu* g, const void* x, int64_t H, int64_t W, int64_t C, void* out) {
    if (H * W * C <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    up2_k<<<blocks(4 * H * W * C), 256, 0, g->q>>>((const __half*) x, (__half*) out, H, W, C);
    NS_LAUNCH_CHECK();
    return 0;
}

}  // extern "C"
