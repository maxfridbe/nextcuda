// ph3.cu: P-H3-Image's deblock filter (image/ph3image/src/deblock.rs): the high-pass, the per-tile and per-position sums
// that give the decoder's fixed tile pattern, its subtraction, and the boundary pass. On the GPU's stream.
#include "ns.h"
#include "ns_internal.hpp"
#include "ph3.h"

#include <cmath>

namespace {

constexpr int WG = 256;
constexpr int T = 16;
inline unsigned nblk(int64_t n) { return (unsigned) ((n + WG - 1) / WG); }
__device__ inline int64_t clampi(int64_t v, int64_t lo, int64_t hi) { return v < lo ? lo : (v > hi ? hi : v); }

__global__ void box_rows_k(const float* x, int64_t H, int64_t W, int64_t r, float* tmp) {
    const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= H * W) return;
    const int64_t y = id / W, xx = id % W;
    const float* row = x + y * W;
    float s = 0.0f;
    for (int64_t k = -r; k <= r; ++k) s += row[clampi(xx + k, 0, W - 1)];
    tmp[id] = s / (float) (2 * r + 1);
}

__global__ void box_cols_hp_k(const float* x, const float* tmp, int64_t H, int64_t W, int64_t r, float* hp) {
    const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= H * W) return;
    const int64_t y = id / W, xx = id % W;
    float s = 0.0f;
    for (int64_t k = -r; k <= r; ++k) s += tmp[clampi(y + k, 0, H - 1) * W + xx];
    hp[id] = x[id] - s / (float) (2 * r + 1);
}

__device__ inline float block_sum(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if ((int) threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o];
        __syncthreads();
    }
    const float s = sh[0];
    __syncthreads();
    return s;
}

// one block a position in the tile; its threads over the whole tiles (tw_all: the tiles a row, a partial one counted)
__global__ void tile_sums_k(const float* hp, int64_t W, int64_t th, int64_t tw, int64_t tw_all, const float* wts, float* sums) {
    __shared__ float sh[WG];
    const int p = blockIdx.x, i = p / T, j = p % T;
    float s = 0.0f, q = 0.0f;
    for (int64_t t = threadIdx.x; t < th * tw; t += blockDim.x) {
        const int64_t ty = t / tw, tx = t % tw;
        const float v = hp[(ty * T + i) * W + tx * T + j], w = wts[ty * tw_all + tx];
        s += w * v;
        q += w * v * v;
    }
    s = block_sum(s, sh);
    q = block_sum(q, sh);
    if (threadIdx.x == 0) {
        sums[2 * p] = s;
        sums[2 * p + 1] = q;
    }
}

// one block a tile, a thread a position
__global__ void tile_stats_k(const float* hp, int64_t H, int64_t W, int64_t tw, const float* pat, float* energy, float* dots) {
    __shared__ float sh[T * T];
    const int64_t t = blockIdx.x;
    const int i = threadIdx.x / T, j = threadIdx.x % T;
    const int64_t y = (t / tw) * T + i, xx = (t % tw) * T + j;
    const float v = (y < H && xx < W) ? hp[y * W + xx] : 0.0f;
    const float e = block_sum(v * v, sh);
    const float d = pat ? block_sum(v * pat[threadIdx.x], sh) : 0.0f;
    if (threadIdx.x == 0) {
        energy[t] = e;
        dots[t] = d;
    }
}

__global__ void subtract_k(float* x, int64_t H, int64_t W, int64_t tw, const float* a, const float* pat) {
    const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= H * W) return;
    const int64_t y = id / W, xx = id % W;
    x[id] -= a[(y / T) * tw + xx / T] * pat[(y % T) * T + xx % T];
}

// a thread a line across one boundary: `pos` the boundary's index (1.. along the axis crossed), `line` the line
__global__ void edges_k(float* x, int64_t H, int64_t W, int vertical, int n, float alpha, float beta, float strength, int64_t nb) {
    const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t lines = vertical ? H : W;
    if (id >= nb * lines) return;
    const int64_t b = (id / lines + 1) * T, line = id % lines;
    // p_k = the k-th pixel before the boundary, q_k the k-th after, as offsets into a plane
    const int64_t step = vertical ? 1 : W;
    const int64_t at = vertical ? line * W + b : b * W + line;
    const int64_t plane = H * W;
    auto luma = [&](int64_t o) { return 0.299f * x[o] + 0.587f * x[plane + o] + 0.114f * x[2 * plane + o]; };
    const float p0 = luma(at - step), p1 = luma(at - 2 * step), p2 = luma(at - 3 * step);
    const float q0 = luma(at), q1 = luma(at + step), q2 = luma(at + 2 * step);
    const float act = fmaxf(fmaxf(fabsf(p0 - p1), fabsf(p1 - p2)), fmaxf(fabsf(q0 - q1), fabsf(q1 - q2)));
    const float g = strength * fmaxf(1.0f - act / beta, 0.0f) * fmaxf(1.0f - fabsf(q0 - p0) / alpha, 0.0f);
    if (g <= 0.0f) return;
    for (int c = 0; c < 3; ++c) {
        float* v = x + c * plane + at;
        const float P0 = v[-step], P1 = v[-2 * step], Q0 = v[0], Q1 = v[step];
        const float e = g * ((Q0 - P0) - ((P0 - P1) + (Q1 - Q0)) * 0.5f);
        for (int k = 0; k < n; ++k) {
            const float w = ((float) (n - k) - 0.5f) / (float) (2 * n);
            v[-(k + 1) * step] += e * w;
            v[k * step] -= e * w;
        }
    }
}

}  // namespace

#define PH3_LAUNCH(k, blocks, threads, ...) do { NS_CU(cudaSetDevice(g->dev)); if ((blocks) > 0) { k<<<(blocks), (threads), 0, g->q>>>(__VA_ARGS__); NS_LAUNCH_CHECK(); } } while (0)

extern "C" {

int ns_image_ph3_highpass(ns_gpu* g, const float* x, int64_t H, int64_t W, int64_t r, float* tmp, float* hp) {
    PH3_LAUNCH(box_rows_k, nblk(H * W), WG, x, H, W, r, tmp);
    PH3_LAUNCH(box_cols_hp_k, nblk(H * W), WG, x, tmp, H, W, r, hp);
    return 0;
}

int ns_image_ph3_tile_stats(ns_gpu* g, const float* hp, int64_t H, int64_t W, const float* pat, float* energy, float* dots) {
    const int64_t th = (H + T - 1) / T, tw = (W + T - 1) / T;
    PH3_LAUNCH(tile_stats_k, (unsigned) (th * tw), T * T, hp, H, W, tw, pat, energy, dots);
    return 0;
}

int ns_image_ph3_tile_sums(ns_gpu* g, const float* hp, int64_t H, int64_t W, const float* wts, float* sums) {
    if (H < T || W < T) return ns_fail("ns_image_ph3_tile_sums: a picture smaller than one tile");
    PH3_LAUNCH(tile_sums_k, T * T, WG, hp, W, H / T, W / T, (W + T - 1) / T, wts, sums);
    return 0;
}

int ns_image_ph3_subtract(ns_gpu* g, float* x, int64_t H, int64_t W, const float* a, const float* pat) {
    PH3_LAUNCH(subtract_k, nblk(H * W), WG, x, H, W, (W + T - 1) / T, a, pat);
    return 0;
}

int ns_image_ph3_edges(ns_gpu* g, float* x, int64_t H, int64_t W, int vertical, int n, float alpha, float beta, float strength) {
    if (n < 3 || n > T / 2) return ns_fail("ns_image_ph3_edges: n is 3 to 8");
    const int64_t len = vertical ? W : H;
    // the boundaries with n pixels (and 3 for the test) on both sides
    int64_t nb = 0;
    while ((nb + 1) * T + n <= len && (nb + 1) * T - n >= 0) ++nb;
    const int64_t lines = vertical ? H : W;
    PH3_LAUNCH(edges_k, nblk(nb * lines), WG, x, H, W, vertical, n, alpha, beta, strength, nb);
    return 0;
}

}  // extern "C"
