// up.cu: the upscalers' own kernels (up.h): what Real-ESRGAN's and waifu2x's ncnn graphs need beside the shared
// convolutions - the tile's pre- and post-processing as the two ncnn programs do them (mirrored or clamped borders, the
// +0.5 rounding), zero padding and dilation (a transposed convolution as a convolution), the activations, PixelShuffle,
// ncnn's Interp (nearest, bilinear, bicubic), Concat's channel copies, Crop, the squeeze-and-excitation blocks' mean,
// dense layers and channel scale. Channels-last half, float32 arithmetic, on the GPU's queue.
#include "ns.h"
#include "ns_internal.hpp"
#include "up.h"

#include <cuda_fp16.h>

namespace {

__device__ inline float h2f(__half h) { return __half2float(h); }
__device__ inline __half f2h(float f) { return __float2half_rn(f); }

inline unsigned blocks(int64_t n) { return (unsigned) ((n + 255) / 256); }

__device__ inline int64_t mirror(int64_t i, int64_t n) {
    i = i < 0 ? -i : i;
    return (n - 1) - llabs(i - (n - 1));
}

__device__ inline int64_t clampi(int64_t i, int64_t n) { return i < 0 ? 0 : (i >= n ? n - 1 : i); }

__global__ void pre_k(const uint8_t* src, int64_t H, int64_t W, int64_t ch, int64_t x0, int64_t y0, int64_t tw, int64_t th, int border,
                      __half* out) {
    const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= tw * th * 3) return;
    const int64_t c = id % 3, x = (id / 3) % tw, y = id / 3 / tw;
    int64_t sx = x0 + x, sy = y0 + y;
    if (border == 0) {
        sx = mirror(sx, W);
        sy = mirror(sy, H);
    } else {
        sx = clampi(sx, W);
        sy = clampi(sy, H);
    }
    out[id] = f2h((float) src[(sy * W + sx) * ch + c] * (1.0f / 255.0f));
}

__global__ void pre_alpha_k(const uint8_t* src, int64_t H, int64_t W, int64_t x0, int64_t y0, int64_t tw, int64_t th, __half* out) {
    const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= tw * th) return;
    const int64_t x = id % tw, y = id / tw;
    const int64_t sx = clampi(x0 + x, W), sy = clampi(y0 + y, H);
    out[id] = f2h((float) src[(sy * W + sx) * 4 + 3] * (1.0f / 255.0f));
}

__device__ inline uint8_t to_u8(float v) {
    v = floorf(v * 255.0f + 0.5f);
    return (uint8_t) (v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v));
}

__global__ void post_k(const __half* src, int64_t SW, int64_t sx, int64_t sy, const __half* alpha, uint8_t* dst, int64_t DW, int64_t dch,
                       int64_t dx, int64_t dy, int64_t w, int64_t h) {
    const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= w * h * dch) return;
    const int64_t c = id % dch, x = (id / dch) % w, y = id / dch / w;
    const float v = c < 3 ? h2f(src[((sy + y) * SW + sx + x) * 3 + c]) : (alpha ? h2f(alpha[y * w + x]) : 1.0f);
    dst[((dy + y) * DW + dx + x) * dch + c] = to_u8(v);
}

__global__ void pad_k(const __half* x, int64_t H, int64_t W, int64_t C, int64_t t, int64_t l, int64_t Wo, int64_t n, __half* out) {
    const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    const int64_t c = id % C, ox = (id / C) % Wo, oy = id / C / Wo;
    const int64_t ix = ox - l, iy = oy - t;
    out[id] = (ix >= 0 && ix < W && iy >= 0 && iy < H) ? x[(iy * W + ix) * C + c] : __float2half(0.0f);
}

__global__ void dilate_k(const __half* x, int64_t W, int64_t C, int64_t s, int64_t Wo, int64_t n, __half* out) {
    const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    const int64_t c = id % C, ox = (id / C) % Wo, oy = id / C / Wo;
    out[id] = (ox % s == 0 && oy % s == 0) ? x[((oy / s) * W + ox / s) * C + c] : __float2half(0.0f);
}

__device__ inline float act(float v, int kind, float a, float slope) {
    switch (kind) {
        case 1: return v > 0.0f ? v : 0.0f;
        case 2: return v > 0.0f ? v : v * a;
        case 3: return v > 0.0f ? v : v * slope;
        case 4: return 1.0f / (1.0f + __expf(-v));
        default: return v;
    }
}

__global__ void act_k(__half* x, int64_t C, int kind, float a, const float* slopes, int64_t n_slopes, int64_t n) {
    const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    const float slope = slopes ? slopes[n_slopes == 1 ? 0 : id % C] : 0.0f;
    x[id] = f2h(act(h2f(x[id]), kind, a, slope));
}

__global__ void pixel_shuffle_k(const __half* x, int64_t W, int64_t C, int64_t r, int64_t Wo, int64_t n, __half* out) {
    const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    const int64_t c = id % C, ox = (id / C) % Wo, oy = id / C / Wo;
    const int64_t ix = ox / r, iy = oy / r, dx = ox % r, dy = oy % r;
    out[id] = x[(iy * W + ix) * (C * r * r) + c * r * r + dy * r + dx];
}

__device__ inline void cubic(float x, float* w) {
    const float A = -0.75f;
    w[0] = ((A * (x + 1) - 5 * A) * (x + 1) + 8 * A) * (x + 1) - 4 * A;
    w[1] = ((A + 2) * x - (A + 3)) * x * x + 1;
    w[2] = ((A + 2) * (1 - x) - (A + 3)) * (1 - x) * (1 - x) + 1;
    w[3] = 1.0f - w[0] - w[1] - w[2];
}

__global__ void resize_k(const __half* x, int64_t H, int64_t W, int64_t C, int64_t Ho, int64_t Wo, int kind, int64_t n, __half* out) {
    const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    const int64_t c = id % C, ox = (id / C) % Wo, oy = id / C / Wo;
    // the library builds with --use_fast_math: a float division W / Wo is approximate (220 / 880 came out under 0.25
    // and the nearest 4x took the pixel before - 2026-10-19), so nearest is integer and the others' positions double
    if (kind == 1) {
        const int64_t ix = min(ox * W / Wo, W - 1), iy = min(oy * H / Ho, H - 1);
        out[id] = x[(iy * W + ix) * C + c];
        return;
    }
    const double ws = (double) W / (double) Wo, hs = (double) H / (double) Ho;
    const double dx = (ox + 0.5) * ws - 0.5, dy = (oy + 0.5) * hs - 0.5;
    int64_t sx = (int64_t) floor(dx), sy = (int64_t) floor(dy);
    float fx = (float) (dx - (double) sx), fy = (float) (dy - (double) sy);
    if (kind == 2) {
        if (sx < 0) { sx = 0; fx = 0.0f; }
        if (sx >= W - 1) { sx = W > 1 ? W - 2 : 0; fx = W > 1 ? 1.0f : 0.0f; }
        if (sy < 0) { sy = 0; fy = 0.0f; }
        if (sy >= H - 1) { sy = H > 1 ? H - 2 : 0; fy = H > 1 ? 1.0f : 0.0f; }
        const int64_t sx1 = min(sx + 1, W - 1), sy1 = min(sy + 1, H - 1);
        const float a = h2f(x[(sy * W + sx) * C + c]), b = h2f(x[(sy * W + sx1) * C + c]);
        const float d = h2f(x[(sy1 * W + sx) * C + c]), e = h2f(x[(sy1 * W + sx1) * C + c]);
        out[id] = f2h((a * (1 - fx) + b * fx) * (1 - fy) + (d * (1 - fx) + e * fx) * fy);
        return;
    }
    float wx[4], wy[4];
    cubic(fx, wx);
    cubic(fy, wy);
    float v = 0.0f;
    for (int j = 0; j < 4; ++j) {
        const int64_t yy = clampi(sy - 1 + j, H);
        float row = 0.0f;
        for (int i = 0; i < 4; ++i) row += wx[i] * h2f(x[(yy * W + clampi(sx - 1 + i, W)) * C + c]);
        v += wy[j] * row;
    }
    out[id] = f2h(v);
}

__global__ void add_k(const __half* a, const __half* b, int64_t n, int op, float ca, float cb, __half* out) {
    const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    const float x = h2f(a[id]), y = h2f(b[id]);
    out[id] = f2h(op == 1 ? x * y : (op == 2 ? fmaxf(x, y) : ca * x + cb * y));
}

__global__ void copy_channels_k(const __half* src, int64_t Cs, __half* dst, int64_t Cd, int64_t off, int64_t n) {
    const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    const int64_t c = id % Cs, p = id / Cs;
    dst[p * Cd + off + c] = src[id];
}

__global__ void crop_k(const __half* x, int64_t W, int64_t C, int64_t x0, int64_t y0, int64_t w, int64_t n, __half* out) {
    const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    const int64_t c = id % C, ox = (id / C) % w, oy = id / C / w;
    out[id] = x[((y0 + oy) * W + x0 + ox) * C + c];
}

// one block a channel: the mean over every pixel
__global__ void mean_k(const __half* x, int64_t P, int64_t C, float* out) {
    __shared__ float s[256];
    const int64_t c = blockIdx.x;
    float acc = 0.0f;
    for (int64_t p = threadIdx.x; p < P; p += blockDim.x) acc += h2f(x[p * C + c]);
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) {
        if ((int) threadIdx.x < o) s[threadIdx.x] += s[threadIdx.x + o];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[c] = s[0] / (float) P;
}

__global__ void dense_k(const float* x, int64_t I, const float* w, const float* b, int64_t O, int kind, float a, float* out) {
    const int64_t o = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= O) return;
    float v = b ? b[o] : 0.0f;
    for (int64_t i = 0; i < I; ++i) v += w[o * I + i] * x[i];
    out[o] = act(v, kind, a, 0.0f);
}

__global__ void scale_k(__half* x, int64_t C, const float* s, int64_t n) {
    const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    x[id] = f2h(h2f(x[id]) * s[id % C]);
}

}  // namespace

extern "C" {

int ns_image_up_pre(ns_gpu* g, const uint8_t* src, int64_t H, int64_t W, int64_t ch, int64_t x0, int64_t y0, int64_t tw, int64_t th,
                    int border, void* out) {
    NS_CU(cudaSetDevice(g->dev));
    const int64_t n = tw * th * 3;
    pre_k<<<blocks(n), 256, 0, g->q>>>(src, H, W, ch, x0, y0, tw, th, border, (__half*) out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_up_pre_alpha(ns_gpu* g, const uint8_t* src, int64_t H, int64_t W, int64_t x0, int64_t y0, int64_t tw, int64_t th, void* out) {
    NS_CU(cudaSetDevice(g->dev));
    pre_alpha_k<<<blocks(tw * th), 256, 0, g->q>>>(src, H, W, x0, y0, tw, th, (__half*) out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_up_post(ns_gpu* g, const void* src, int64_t SW, int64_t sx, int64_t sy, const void* alpha, uint8_t* dst, int64_t DW,
                     int64_t dch, int64_t dx, int64_t dy, int64_t w, int64_t h) {
    NS_CU(cudaSetDevice(g->dev));
    post_k<<<blocks(w * h * dch), 256, 0, g->q>>>((const __half*) src, SW, sx, sy, (const __half*) alpha, dst, DW, dch, dx, dy, w, h);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_up_pad(ns_gpu* g, const void* x, int64_t H, int64_t W, int64_t C, int64_t t, int64_t l, int64_t b, int64_t r, void* out) {
    NS_CU(cudaSetDevice(g->dev));
    const int64_t Wo = W + l + r, n = (H + t + b) * Wo * C;
    pad_k<<<blocks(n), 256, 0, g->q>>>((const __half*) x, H, W, C, t, l, Wo, n, (__half*) out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_up_dilate(ns_gpu* g, const void* x, int64_t H, int64_t W, int64_t C, int64_t s, void* out) {
    NS_CU(cudaSetDevice(g->dev));
    const int64_t Wo = (W - 1) * s + 1, n = ((H - 1) * s + 1) * Wo * C;
    dilate_k<<<blocks(n), 256, 0, g->q>>>((const __half*) x, W, C, s, Wo, n, (__half*) out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_up_act(ns_gpu* g, void* x, int64_t N, int64_t C, int kind, float a, const float* slopes, int64_t n_slopes) {
    NS_CU(cudaSetDevice(g->dev));
    const int64_t n = N * C;
    act_k<<<blocks(n), 256, 0, g->q>>>((__half*) x, C, kind, a, slopes, n_slopes, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_up_pixel_shuffle(ns_gpu* g, const void* x, int64_t H, int64_t W, int64_t C, int64_t r, void* out) {
    NS_CU(cudaSetDevice(g->dev));
    const int64_t Wo = W * r, n = H * r * Wo * C;
    pixel_shuffle_k<<<blocks(n), 256, 0, g->q>>>((const __half*) x, W, C, r, Wo, n, (__half*) out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_up_resize(ns_gpu* g, const void* x, int64_t H, int64_t W, int64_t C, int64_t Ho, int64_t Wo, int kind, void* out) {
    NS_CU(cudaSetDevice(g->dev));
    const int64_t n = Ho * Wo * C;
    resize_k<<<blocks(n), 256, 0, g->q>>>((const __half*) x, H, W, C, Ho, Wo, kind, n, (__half*) out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_up_add(ns_gpu* g, const void* a, const void* b, int64_t n, int op, float ca, float cb, void* out) {
    NS_CU(cudaSetDevice(g->dev));
    add_k<<<blocks(n), 256, 0, g->q>>>((const __half*) a, (const __half*) b, n, op, ca, cb, (__half*) out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_up_copy_channels(ns_gpu* g, const void* src, int64_t P, int64_t Cs, void* dst, int64_t Cd, int64_t off) {
    NS_CU(cudaSetDevice(g->dev));
    const int64_t n = P * Cs;
    copy_channels_k<<<blocks(n), 256, 0, g->q>>>((const __half*) src, Cs, (__half*) dst, Cd, off, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_up_crop(ns_gpu* g, const void* x, int64_t W, int64_t C, int64_t x0, int64_t y0, int64_t w, int64_t h, void* out) {
    NS_CU(cudaSetDevice(g->dev));
    const int64_t n = w * h * C;
    crop_k<<<blocks(n), 256, 0, g->q>>>((const __half*) x, W, C, x0, y0, w, n, (__half*) out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_up_mean(ns_gpu* g, const void* x, int64_t P, int64_t C, float* out) {
    NS_CU(cudaSetDevice(g->dev));
    mean_k<<<(unsigned) C, 256, 0, g->q>>>((const __half*) x, P, C, out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_up_dense(ns_gpu* g, const float* x, int64_t I, const float* w, const float* b, int64_t O, int kind, float a, float* out) {
    NS_CU(cudaSetDevice(g->dev));
    dense_k<<<blocks(O), 256, 0, g->q>>>(x, I, w, b, O, kind, a, out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_up_scale(ns_gpu* g, void* x, int64_t P, int64_t C, const float* s) {
    NS_CU(cudaSetDevice(g->dev));
    const int64_t n = P * C;
    scale_k<<<blocks(n), 256, 0, g->q>>>((__half*) x, C, s, n);
    NS_LAUNCH_CHECK();
    return 0;
}

}  // extern "C"
