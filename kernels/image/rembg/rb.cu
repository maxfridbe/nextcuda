// rb.cu: the background removers' own kernels: IS-Net's ceil-mode max-pool, its half-pixel bilinear upsampling, a
// bias with ReLU, the sigmoid. Elementwise, on the GPU's stream.
#include "ns.h"
#include "ns_internal.hpp"
#include "rb.h"

#include <cfloat>
#include <cmath>

namespace {

constexpr int WG = 256;
inline unsigned nblk(int64_t n) { return (unsigned) ((n + WG - 1) / WG); }

__global__ void maxpool2_k(const float* x, int64_t H, int64_t W, int64_t C, int64_t Ho, int64_t Wo, float* out) {
    const int64_t n = Ho * Wo * C;
    const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    const int64_t c = id % C, p = id / C, xo = p % Wo, yo = p / Wo;
    float m = -FLT_MAX;
    for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
            const int64_t y = 2 * yo + dy, xx = 2 * xo + dx;
            if (y < H && xx < W) m = fmaxf(m, x[(y * W + xx) * C + c]);
        }
    }
    out[id] = m;
}

__global__ void resize_k(const float* x, int64_t Hi, int64_t Wi, int64_t C, int64_t Ho, int64_t Wo, float* out, int64_t os) {
    const int64_t n = Ho * Wo * C;
    const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    const int64_t c = id % C, p = id / C, xo = p % Wo, yo = p / Wo;
    const float sy = fmaxf(((float) yo + 0.5f) * (float) Hi / (float) Ho - 0.5f, 0.0f);
    const float sx = fmaxf(((float) xo + 0.5f) * (float) Wi / (float) Wo - 0.5f, 0.0f);
    const int64_t y0 = min((int64_t) sy, Hi - 1), x0 = min((int64_t) sx, Wi - 1);
    const int64_t y1 = min(y0 + 1, Hi - 1), x1 = min(x0 + 1, Wi - 1);
    const float fy = sy - (float) y0, fx = sx - (float) x0;
    const float a = x[(y0 * Wi + x0) * C + c], b = x[(y0 * Wi + x1) * C + c];
    const float d = x[(y1 * Wi + x0) * C + c], e = x[(y1 * Wi + x1) * C + c];
    out[(yo * Wo + xo) * os + c] = (a * (1.0f - fx) + b * fx) * (1.0f - fy) + (d * (1.0f - fx) + e * fx) * fy;
}

__global__ void bias_act_k(float* x, int64_t n, int64_t C, const float* bias, int relu) {
    const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    float v = x[id] + (bias ? bias[id % C] : 0.0f);
    x[id] = relu ? fmaxf(v, 0.0f) : v;
}

__global__ void sigmoid_k(float* x, int64_t n) {
    const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) x[id] = 1.0f / (1.0f + expf(-x[id]));
}

}  // namespace

#define RB_LAUNCH(k, n, ...) do { NS_CU(cudaSetDevice(g->dev)); if ((n) > 0) { k<<<nblk(n), WG, 0, g->q>>>(__VA_ARGS__); NS_LAUNCH_CHECK(); } return 0; } while (0)

extern "C" {

int ns_image_rb_maxpool2(ns_gpu* g, const float* x, int64_t H, int64_t W, int64_t C, float* out) {
    const int64_t Ho = (H + 1) / 2, Wo = (W + 1) / 2;
    RB_LAUNCH(maxpool2_k, Ho * Wo * C, x, H, W, C, Ho, Wo, out);
}

int ns_image_rb_resize(ns_gpu* g, const float* x, int64_t Hi, int64_t Wi, int64_t C, int64_t Ho, int64_t Wo, float* out, int64_t out_stride) {
    RB_LAUNCH(resize_k, Ho * Wo * C, x, Hi, Wi, C, Ho, Wo, out, out_stride);
}

int ns_image_rb_bias_act(ns_gpu* g, float* x, int64_t M, int64_t C, const float* bias, int relu) {
    RB_LAUNCH(bias_act_k, M * C, x, M * C, C, bias, relu);
}

int ns_image_rb_sigmoid(ns_gpu* g, float* x, int64_t n) { RB_LAUNCH(sigmoid_k, n, x, n); }

}  // extern "C"
