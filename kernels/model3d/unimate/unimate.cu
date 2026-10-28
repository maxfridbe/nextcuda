// unimate.cu: UniMate's kernels (unimate.h): RMS norms with the adaLN modulation, gated residuals, SwiGLU / gated GELU,
// the rotary embeddings, attention (dense, a bias, a key mask) and the row shuffles of the token grid, all float32 as the
// reference runs; the linear layers are the shared diffusion kernels' (cuBLAS, float32). nextsycl's unimate.cpp in CUDA,
// the same names, arguments and arithmetic: its work-group reductions are block reductions (warp shuffles, then shared
// memory), on the GPU's stream. Built without --use_fast_math (kernels/build.sh), as the reference's float32.
#include "ns.h"
#include "ns_internal.hpp"
#include "unimate.h"

#include <algorithm>
#include <cmath>

namespace {

unsigned blocks(int64_t n, int t = 256) { return (unsigned) ((n + t - 1) / t); }
#define ID() const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x

__device__ inline float silu(float v) {
    return v / (1.0f + expf(-v));
}
__device__ inline float gelu_erf(float v) {
    return 0.5f * v * (1.0f + erff(v * 0.7071067811865476f));
}
__device__ inline float gelu_tanh(float v) {  // T5's gelu_new
    return 0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
}

// the sum over the block (blockDim.x a multiple of 32), every thread gets it; red holds 32 floats
__device__ float block_sum(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = 0.0f;
    for (int i = 0; i < (int) (blockDim.x >> 5); ++i) t += red[i];
    __syncthreads();
    return t;
}

constexpr int RMS_WG = 128, RMS_PER = 8;  // a row of up to 1024 values held in registers

__global__ void rms_k(const float* x, int64_t x_row, int64_t H, int64_t C, float eps, const float* w, const float* scale, const float* shift, float* out,
                      int64_t out_row) {
    __shared__ float red[32];
    const int64_t r = blockIdx.x;
    const int64_t m = r / H, h = r % H;
    const int l = threadIdx.x;
    const float* row = x + m * x_row + h * C;
    float v[RMS_PER];
    float s = 0.0f;
#pragma unroll
    for (int j = 0; j < RMS_PER; ++j) {
        const int64_t i = (int64_t) j * RMS_WG + l;
        v[j] = i < C ? row[i] : 0.0f;
        s += v[j] * v[j];
    }
    // block_sum's barriers: every lane has read its row before any writes it (out may be x)
    const float inv = rsqrtf(block_sum(s, red) / (float) C + eps);
    float* o = out + m * out_row + h * C;
#pragma unroll
    for (int j = 0; j < RMS_PER; ++j) {
        const int64_t i = (int64_t) j * RMS_WG + l;
        if (i < C) {
            float y = w[i] * (v[j] * inv);
            if (scale) y = y * (1.0f + scale[i]);
            if (shift) y = y + shift[i];
            o[i] = y;
        }
    }
}

__global__ void gated_add_k(float* x, const float* y, int64_t C, const float* gate, int64_t n) {
    ID();
    if (i >= n) return;
    x[i] += (gate ? gate[i % C] : 1.0f) * y[i];
}

__global__ void act_k(float* x, int64_t n, int kind) {
    ID();
    if (i >= n) return;
    x[i] = kind == 0 ? silu(x[i]) : gelu_erf(x[i]);
}

__global__ void glu_k(const float* x, int64_t C, int kind, float* out, int64_t n) {
    ID();
    if (i >= n) return;
    const int64_t r = i / C, c = i % C;
    const float a = x[r * 2 * C + c];
    out[i] = (kind == 0 ? silu(a) : gelu_tanh(a)) * x[r * 2 * C + C + c];
}

__global__ void rope_k(float* x, int64_t row, int64_t seg, int64_t M, int64_t H, int64_t half, int64_t div, int64_t mod, const float* cos,
                       const float* sin, int64_t n) {
    ID();
    if (i >= n) return;
    const int64_t ii = i % half, h = (i / half) % H, r = i / half / H;
    const int64_t sg = r / M, m = r % M, D = 2 * half;
    float* p = x + sg * seg + m * row + h * D;
    const int64_t t = ((m / div) % mod) * D;
    const float a = p[ii], b = p[ii + half];
    p[ii] = a * cos[t + ii] - b * sin[t + ii];
    p[ii + half] = b * cos[t + ii + half] + a * sin[t + ii + half];
}

constexpr int AT_MAXD = 128, AT_MAXS = 128;

// a thread a (group, head, query)
__global__ void attention_k(const float* q, int64_t q_row, const float* k, int64_t k_row, const float* v, int64_t v_row, int64_t G, int64_t Sq,
                            int64_t Skv, int64_t H, int64_t D, int64_t q_gs, int64_t q_ss, int64_t kv_gs, int64_t kv_ss, int64_t nv, float scale,
                            const float* bias, float* out, int64_t out_row) {
    const int64_t t = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= G * H * Sq) return;
    const int64_t i = t % Sq, h = (t / Sq) % H, gi = t / (Sq * H);
    const float* qp = q + (gi * q_gs + i * q_ss) * q_row + h * D;
    float s[AT_MAXS];
    float mx = -INFINITY;
    for (int64_t j = 0; j < nv; ++j) {
        const float* kp = k + (gi * kv_gs + j * kv_ss) * k_row + h * D;
        float d = 0.0f;
        for (int64_t c = 0; c < D; ++c) d += qp[c] * kp[c];
        d *= scale;
        if (bias) d += bias[(h * Sq + i) * Skv + j];
        s[j] = d;
        mx = fmaxf(mx, d);
    }
    float sum = 0.0f;
    for (int64_t j = 0; j < nv; ++j) {
        s[j] = expf(s[j] - mx);
        sum += s[j];
    }
    float acc[AT_MAXD];
    for (int64_t c = 0; c < D; ++c) acc[c] = 0.0f;
    for (int64_t j = 0; j < nv; ++j) {
        const float p = s[j] / sum;
        const float* vp = v + (gi * kv_gs + j * kv_ss) * v_row + h * D;
        for (int64_t c = 0; c < D; ++c) acc[c] += p * vp[c];
    }
    float* op = out + (gi * q_gs + i * q_ss) * out_row + h * D;
    for (int64_t c = 0; c < D; ++c) op[c] = acc[c];
}

__global__ void copy_rows_k(const float* src, int64_t src_off, int64_t src_stride, float* dst, int64_t dst_off, int64_t dst_stride, int64_t C, int64_t n) {
    ID();
    if (i >= n) return;
    const int64_t r = i / C, c = i % C;
    dst[(dst_off + r * dst_stride) * C + c] = src[(src_off + r * src_stride) * C + c];
}

__global__ void cat2_k(const float* a, int64_t C1, const float* b, int64_t C2, float* out, int64_t n) {
    ID();
    if (i >= n) return;
    const int64_t r = i / (C1 + C2), c = i % (C1 + C2);
    out[i] = c < C1 ? a[r * C1 + c] : b[r * C2 + (c - C1)];
}

__global__ void tokens_k(const float* tpos, const float* joints, const float* root, const float* depth, const float* name, int64_t J, int64_t D,
                         float* out, int64_t n) {
    ID();
    if (i >= n) return;
    const int64_t r = i / D, c = i % D;
    const int64_t f = r / J, j = r % J;
    float t;
    if (f == 0) t = tpos[j * D + c];
    else if (j == 0) t = root[(f - 1) * D + c];
    else t = joints[((f - 1) * J + j) * D + c];
    out[i] = (t + depth[j * D + c]) + name[j * D + c];
}

}  // namespace

extern "C" {

int ns_model3d_unimate_rms(ns_gpu* g, const float* x, int64_t x_row, int64_t M, int64_t H, int64_t C, float eps, const float* w,
                           const float* scale, const float* shift, float* out, int64_t out_row) {
    if (C > (int64_t) RMS_WG * RMS_PER) return ns_fail("ns_model3d_unimate_rms: rows longer than 1024");
    if (!w) return ns_fail("ns_model3d_unimate_rms: no weight");
    if (M <= 0 || H <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    rms_k<<<(unsigned) (M * H), RMS_WG, 0, g->q>>>(x, x_row, H, C, eps, w, scale, shift, out, out_row);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_unimate_gated_add(ns_gpu* g, float* x, const float* y, int64_t M, int64_t C, const float* gate) {
    if (M <= 0 || C <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    gated_add_k<<<blocks(M * C), 256, 0, g->q>>>(x, y, C, gate, M * C);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_unimate_act(ns_gpu* g, float* x, int64_t n, int kind) {
    if (n <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    act_k<<<blocks(n), 256, 0, g->q>>>(x, n, kind);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_unimate_glu(ns_gpu* g, const float* x, int64_t M, int64_t C, int kind, float* out) {
    if (M <= 0 || C <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    glu_k<<<blocks(M * C), 256, 0, g->q>>>(x, C, kind, out, M * C);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_unimate_rope(ns_gpu* g, float* x, int64_t row, int64_t seg, int64_t nseg, int64_t M, int64_t H, int64_t D, int64_t div,
                            int64_t mod, const float* cos, const float* sin) {
    if (M <= 0 || D % 2 != 0) return 0;
    const int64_t half = D / 2, n = nseg * M * H * half;
    if (n <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    rope_k<<<blocks(n), 256, 0, g->q>>>(x, row, seg, M, H, half, div, mod, cos, sin, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_unimate_attention(ns_gpu* g, const float* q, int64_t q_row, const float* k, int64_t k_row, const float* v, int64_t v_row,
                                 int64_t G, int64_t Sq, int64_t Skv, int64_t H, int64_t D, int64_t q_gs, int64_t q_ss, int64_t kv_gs,
                                 int64_t kv_ss, int64_t nvalid, float scale, const float* bias, float* out, int64_t out_row) {
    if (D > AT_MAXD || Skv > AT_MAXS) return ns_fail("ns_model3d_unimate_attention: head size or key count past 128");
    if (G <= 0 || Sq <= 0 || Skv <= 0 || H <= 0) return 0;
    const int64_t nv = std::min(nvalid, Skv);
    NS_CU(cudaSetDevice(g->dev));
    attention_k<<<blocks(G * H * Sq, 64), 64, 0, g->q>>>(q, q_row, k, k_row, v, v_row, G, Sq, Skv, H, D, q_gs, q_ss, kv_gs, kv_ss, nv, scale, bias,
                                                        out, out_row);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_unimate_copy_rows(ns_gpu* g, const float* src, int64_t src_off, int64_t src_stride, float* dst, int64_t dst_off,
                                 int64_t dst_stride, int64_t n, int64_t C) {
    if (n <= 0 || C <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    copy_rows_k<<<blocks(n * C), 256, 0, g->q>>>(src, src_off, src_stride, dst, dst_off, dst_stride, C, n * C);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_unimate_cat2(ns_gpu* g, const float* a, int64_t C1, const float* b, int64_t C2, int64_t M, float* out) {
    if (M <= 0 || C1 + C2 <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    cat2_k<<<blocks(M * (C1 + C2)), 256, 0, g->q>>>(a, C1, b, C2, out, M * (C1 + C2));
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_model3d_unimate_tokens(ns_gpu* g, const float* tpos, const float* joints, const float* root, const float* depth, const float* name,
                              int64_t F, int64_t J, int64_t D, float* out) {
    if (F <= 0 || J <= 0 || D <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    tokens_k<<<blocks((F + 1) * J * D), 256, 0, g->q>>>(tpos, joints, root, depth, name, J, D, out, (F + 1) * J * D);
    NS_LAUNCH_CHECK();
    return 0;
}

}  // extern "C"
