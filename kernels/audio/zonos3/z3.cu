// z3.cu: the ZONOS3-Mini engine's own kernels (audio/zonos3) - what the shared ones do not have: the backbone's
// embeddings, Full AttnRes, the gated DeltaNet's convolution and recurrence, the latent MoE's Q4_0 experts read from
// their blocks, a small-head attention (causal, windowed or full), the codec decoder's table lookups, causal ConvNeXt
// convolution, patch upsampling, iSTFT spectrum, overlap-add, FIR mixer and resampler, and the speaker encoder's grouped
// convolutions and pooling.
//
// Origin: the arithmetic of Zyphra's ZONOS3 reference runtime (github.com/Zyphra/ZONOS3-Mini: zonos3.reference.lm,
// servers/ggml/src/zonos3-graph.cpp, zac.cpp, spk-encoder.cpp and the MLX port; MIT licence, the model Apache-2.0),
// written anew for CUDA. The gated delta rule's per-step order (decay, read, correct, write, read out) is the one of
// the reference's kernel.
//
// Precise arithmetic (kernels/build.sh builds audio/ without --use_fast_math).
#include "ns.h"
#include "ns_internal.hpp"
#include "z3.h"

#include <cuda_fp16.h>

#include <cmath>

namespace {

constexpr int WARP = 32;

__device__ inline float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// a block's sum (blockDim.x a multiple of 32, at most 1024); every thread gets it
__device__ float block_sum(float v) {
    __shared__ float part[32];
    const int lane = threadIdx.x & 31, wi = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) part[wi] = v;
    __syncthreads();
    float s = lane < nw ? part[lane] : 0.0f;
    s = warp_sum(s);
    return s;
}

unsigned blocks(int64_t n, int t = 256) { return (unsigned) ((n + t - 1) / t); }

__device__ inline float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ inline float siluf_(float x) { return x / (1.0f + expf(-x)); }

// ---------------------------------------------------------------------------------------------------- backbone

__global__ void embed_k(const __half* audio, int ncb, int64_t va, const __half* text, int64_t vt, const int32_t* ids, int64_t rows, int64_t D, float* out) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * D) return;
    const int64_t r = i / D, d = i % D;
    const int32_t* id = ids + r * (ncb + 1);
    float s = 0.0f;
    for (int c = 0; c < ncb; ++c) {
        int64_t t = id[c];
        t = t < 0 ? 0 : (t >= va ? va - 1 : t);
        s += __half2float(audio[((int64_t) c * va + t) * D + d]);
    }
    int64_t t = id[ncb];
    t = t < 0 ? 0 : (t >= vt ? vt - 1 : t);
    out[i] = s + __half2float(text[t * D + d]);
}

__global__ void rms_k(const float* x, int64_t xs, int64_t C, const float* w, float eps, float* out) {
    const int64_t r = blockIdx.x;
    const float* p = x + r * xs;
    float ss = 0.0f;
    for (int64_t c = threadIdx.x; c < C; c += blockDim.x) ss += p[c] * p[c];
    const float inv = 1.0f / sqrtf(block_sum(ss) / (float) C + eps);
    for (int64_t c = threadIdx.x; c < C; c += blockDim.x) out[r * C + c] = w ? p[c] * inv * w[c] : p[c] * inv;
}

__global__ void attnres_score_k(const float* src, int64_t D, const float* q, int64_t Q, float eps, float* scores) {
    const int64_t r = blockIdx.x;
    const float* p = src + r * D;
    float ss = 0.0f;
    for (int64_t c = threadIdx.x; c < D; c += blockDim.x) ss += p[c] * p[c];
    const float inv = 1.0f / sqrtf(block_sum(ss) / (float) D + eps);
    const int lane = threadIdx.x & 31, wi = threadIdx.x >> 5, nw = blockDim.x >> 5;
    for (int64_t j = wi; j < Q; j += nw) {
        float s = 0.0f;
        for (int64_t c = lane; c < D; c += 32) s += (p[c] * inv) * q[j * D + c];
        s = warp_sum(s);
        if (lane == 0) scores[r * Q + j] = s;
    }
}

__global__ void attnres_mix_k(const float* bank, const float* scores, int S, int64_t cap, int64_t D, int64_t Q, int64_t qi, float* out) {
    __shared__ float w[64];
    const int64_t r = blockIdx.x;
    if (threadIdx.x == 0) {
        float m = -INFINITY;
        for (int s = 0; s < S; ++s) m = fmaxf(m, scores[((int64_t) s * cap + r) * Q + qi]);
        float z = 0.0f;
        for (int s = 0; s < S; ++s) {
            w[s] = expf(scores[((int64_t) s * cap + r) * Q + qi] - m);
            z += w[s];
        }
        for (int s = 0; s < S; ++s) w[s] /= z;
    }
    __syncthreads();
    for (int64_t d = threadIdx.x; d < D; d += blockDim.x) {
        float a = 0.0f;
        for (int s = 0; s < S; ++s) a += w[s] * bank[((int64_t) s * cap + r) * D + d];
        out[r * D + d] = a;
    }
}

__global__ void add_vec_k(float* x, int64_t C, const float* v, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += v[i % C];
}

// one thread a channel, the rows in order (the window of the last K inputs in registers)
constexpr int MAXK = 8;
__global__ void gdn_conv_k(const float* proj, int64_t rows, int64_t P, int64_t C, const float* w, int K, float* hist, float* out) {
    const int64_t c = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float win[MAXK], wc[MAXK];
    for (int j = 0; j < K; ++j) wc[j] = w[c * K + j];
    for (int j = 0; j < K - 1; ++j) win[j] = hist[c * (K - 1) + j];
    for (int64_t t = 0; t < rows; ++t) {
        win[K - 1] = proj[t * P + c];
        float s = 0.0f;
        for (int j = 0; j < K; ++j) s += wc[j] * win[j];
        out[t * C + c] = siluf_(s);
        for (int j = 0; j < K - 1; ++j) win[j] = win[j + 1];
    }
    for (int j = 0; j < K - 1; ++j) hist[c * (K - 1) + j] = win[j];
}

// a block a (row, head), S threads
__global__ void gdn_prep_k(const float* conv, const float* proj, int64_t P, int64_t ab0, int H, int S, const float* a_log, const float* dt_bias, float* q,
                           float* k, float* v, float* decay, float* beta) {
    const int64_t r = blockIdx.x / H;
    const int h = (int) (blockIdx.x % H), i = threadIdx.x;
    const int64_t inner = (int64_t) H * S;
    const float* cr = conv + r * 3 * inner;
    const float qv = cr[h * S + i], kv = cr[inner + h * S + i], vv = cr[2 * inner + h * S + i];
    const float qn = block_sum(qv * qv), kn = block_sum(kv * kv);
    const int64_t o = (r * H + h) * S + i;
    q[o] = qv / sqrtf(qn + 1e-6f) * (1.0f / sqrtf((float) S));
    k[o] = kv / sqrtf(kn + 1e-6f);
    v[o] = vv;
    if (i == 0) {
        const float a = proj[r * P + ab0 + h] + dt_bias[h];
        const float sp = a > 0.0f ? a + log1pf(expf(-a)) : log1pf(expf(a));
        decay[r * H + h] = expf(-expf(a_log[h]) * sp);
        beta[r * H + h] = sigmoidf_(proj[r * P + ab0 + H + h]);
    }
}

// a warp a (head, value row); lane l holds the state's keys l + 32 j (S = 128: j < 4)
constexpr int GS = 128, GJ = GS / WARP, G_NW = 4;
__global__ void gdn_scan_k(const float* q, const float* k, const float* v, const float* decay, const float* beta, int64_t rows, int H, float* state,
                           float* y) {
    const int h = blockIdx.x, lane = threadIdx.x & 31;
    const int vr = blockIdx.y * G_NW + (threadIdx.x >> 5);
    float* st = state + ((int64_t) h * GS + vr) * GS;
    float s[GJ];
#pragma unroll
    for (int j = 0; j < GJ; ++j) s[j] = st[lane + 32 * j];
    for (int64_t t = 0; t < rows; ++t) {
        const int64_t o = (t * H + h) * GS;
        const float g = decay[t * H + h], b = beta[t * H + h];
        float kv = 0.0f, kk[GJ], qq[GJ];
#pragma unroll
        for (int j = 0; j < GJ; ++j) {
            kk[j] = k[o + lane + 32 * j];
            qq[j] = q[o + lane + 32 * j];
            s[j] *= g;
            kv += s[j] * kk[j];
        }
        kv = warp_sum(kv);
        const float d = (v[o + vr] - kv) * b;
        float out = 0.0f;
#pragma unroll
        for (int j = 0; j < GJ; ++j) {
            s[j] += kk[j] * d;
            out += s[j] * qq[j];
        }
        out = warp_sum(out);
        if (lane == 0) y[o + vr] = out;
    }
#pragma unroll
    for (int j = 0; j < GJ; ++j) st[lane + 32 * j] = s[j];
}

__global__ void gdn_out_k(const float* y, const float* proj, int64_t P, int64_t z0, int H, int S, const float* w, float eps, float* out) {
    const int64_t r = blockIdx.x / H;
    const int h = (int) (blockIdx.x % H), i = threadIdx.x;
    const float yv = y[(r * H + h) * S + i];
    const float inv = 1.0f / sqrtf(block_sum(yv * yv) / (float) S + eps);
    const float z = proj[r * P + z0 + h * S + i];
    out[(r * H + h) * S + i] = yv * inv * w[i] * siluf_(z);
}

// a warp a (row, head)
__global__ void head_rms_k(float* x, int64_t xs, int64_t rows, int H, int D, const float* scale, float eps) {
    const int64_t gw = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) / WARP;
    const int lane = threadIdx.x & 31;
    if (gw >= rows * H) return;
    const int64_t r = gw / H;
    const int h = (int) (gw % H);
    float* p = x + r * xs + (int64_t) h * D;
    float ss = 0.0f;
    for (int i = lane; i < D; i += 32) ss += p[i] * p[i];
    const float inv = 1.0f / sqrtf(warp_sum(ss) / (float) D + eps);
    const float sc = scale ? scale[h] : 1.0f;
    for (int i = lane; i < D; i += 32) p[i] = p[i] * inv * sc;
}

__global__ void rope_k(float* x, int64_t xs, int64_t rows, int H, int D, float theta, int64_t pos0) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int half = D / 2;
    if (i >= rows * H * half) return;
    const int64_t r = i / ((int64_t) H * half);
    const int h = (int) ((i / half) % H), p = (int) (i % half);
    float* e = x + r * xs + (int64_t) h * D + 2 * p;
    const float inv_freq = powf(theta, -2.0f * (float) p / (float) D);
    const float a = (float) (pos0 + r) * inv_freq, c = cosf(a), s = sinf(a);
    const float x0 = e[0], x1 = e[1];
    e[0] = x0 * c - x1 * s;
    e[1] = x0 * s + x1 * c;
}

// a block of QB query rows of one head; keys in tiles of KT through shared memory, each thread its own online softmax
constexpr int QB = 64, KT = 64;
template <int D>
__global__ void attention_k(const float* q, int64_t qs, int64_t rows, const float* k, const float* v, int64_t kvs, int64_t keys, int Hq, int Hk,
                            float scale, int64_t pos0, int causal, int64_t window, float* out) {
    __shared__ float ks[KT][D], vs[KT][D];
    const int h = blockIdx.y, hk = h / (Hq / Hk);
    const int64_t r0 = (int64_t) blockIdx.x * QB, r = r0 + threadIdx.x;
    const bool live = r < rows;
    const int64_t rl = (r0 + QB < rows ? r0 + QB : rows) - 1;
    const int64_t qp = pos0 + r;
    int64_t kend = keys, kbeg = 0;
    if (causal) kend = min(keys, pos0 + rl + 1);
    if (window > 0) kbeg = max((int64_t) 0, pos0 + r0 - window + 1);
    float qv[D], acc[D];
    float m = -INFINITY, l = 0.0f;
#pragma unroll
    for (int d = 0; d < D; ++d) {
        qv[d] = live ? q[r * qs + (int64_t) h * D + d] * scale : 0.0f;
        acc[d] = 0.0f;
    }
    for (int64_t t0 = kbeg; t0 < kend; t0 += KT) {
        __syncthreads();
        for (int e = threadIdx.x; e < KT * D; e += blockDim.x) {
            const int kk = e / D, d = e % D;
            const int64_t kp = t0 + kk;
            ks[kk][d] = kp < kend ? k[kp * kvs + (int64_t) hk * D + d] : 0.0f;
            vs[kk][d] = kp < kend ? v[kp * kvs + (int64_t) hk * D + d] : 0.0f;
        }
        __syncthreads();
        if (!live) continue;
        const int n = (int) min((int64_t) KT, kend - t0);
        for (int kk = 0; kk < n; ++kk) {
            const int64_t kp = t0 + kk;
            if (causal && kp > qp) break;
            if (window > 0 && qp - kp >= window) continue;
            float s = 0.0f;
#pragma unroll
            for (int d = 0; d < D; ++d) s += qv[d] * ks[kk][d];
            if (s > m) {
                const float c = expf(m - s);
                l *= c;
#pragma unroll
                for (int d = 0; d < D; ++d) acc[d] *= c;
                m = s;
            }
            const float p = expf(s - m);
            l += p;
#pragma unroll
            for (int d = 0; d < D; ++d) acc[d] += p * vs[kk][d];
        }
    }
    if (!live) return;
    const float inv = l > 0.0f ? 1.0f / l : 0.0f;
#pragma unroll
    for (int d = 0; d < D; ++d) out[r * (int64_t) Hq * D + (int64_t) h * D + d] = acc[d] * inv;
}

__global__ void head_gate_k(float* x, const float* gate, int64_t n, int64_t H, int64_t D) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= sigmoidf_(gate[i / D]);
}

__global__ void softcap_k(float* x, int64_t n, float cap) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = cap * tanhf(x[i] / cap);
}

__global__ void silu_mul_k(const float* a, const float* b, float* out, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = siluf_(a[i]) * b[i];
}

__global__ void topk_k(const float* logits, const float* bias, int64_t rows, int E, int k, int32_t* idx, float* w) {
    const int64_t r = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const float* lg = logits + r * E;
    int sel[8];
    for (int s = 0; s < k; ++s) {
        int best = -1;
        float bv = INFINITY;
        for (int e = 0; e < E; ++e) {
            bool used = false;
            for (int u = 0; u < s; ++u) used |= sel[u] == e;
            if (used) continue;
            const float val = bias[e] - lg[e];
            if (val < bv) {
                bv = val;
                best = e;
            }
        }
        sel[s] = best < 0 ? 0 : best;
    }
    float m = -INFINITY;
    for (int s = 0; s < k; ++s) m = fmaxf(m, lg[sel[s]]);
    float z = 0.0f, ex[8];
    for (int s = 0; s < k; ++s) {
        ex[s] = expf(lg[sel[s]] - m);
        z += ex[s];
    }
    for (int s = 0; s < k; ++s) {
        idx[r * k + s] = sel[s];
        w[r * k + s] = ex[s] / z;
    }
}

// Q4_0: a half scale and 16 bytes, value j of a block (q[j] & 15) - 8 for j < 16, (q[j - 16] >> 4) - 8 after
constexpr int Q4B = 18;
// 8 consecutive values (j0 a multiple of 8) of a Q4_0 row, dotted with x[0..8)
__device__ inline float q4_dot8(const uint8_t* row, int64_t j0, const float* x) {
    const uint8_t* blk = row + (j0 / 32) * Q4B;
    const float d = __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t*>(blk)));
    const int j = (int) (j0 % 32);
    const uint8_t* qs = blk + 2;
    float s = 0.0f;
    if (j < 16) {
#pragma unroll
        for (int i = 0; i < 8; ++i) s += (float) ((int) (qs[j + i] & 15) - 8) * x[i];
    } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) s += (float) ((int) (qs[j - 16 + i] >> 4) - 8) * x[i];
    }
    return s * d;
}

constexpr int M_NW = 8;
// a warp an output f of a (row, slot): gate and up together
__global__ void moe_up_k(const float* x, int64_t L, const uint8_t* gate, const uint8_t* up, const int32_t* idx, int k, int64_t F, float* h) {
    const int64_t rs = blockIdx.x;  // row * k + slot
    const int64_t f = (int64_t) blockIdx.y * M_NW + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (f >= F) return;
    const int64_t e = idx[rs];
    const int64_t row_bytes = L / 32 * Q4B;
    const uint8_t* gr = gate + (e * F + f) * row_bytes;
    const uint8_t* ur = up + (e * F + f) * row_bytes;
    const float* xr = x + (rs / k) * L;
    float sg = 0.0f, su = 0.0f;
    for (int64_t c = lane; c < L / 8; c += 32) {
        sg += q4_dot8(gr, c * 8, xr + c * 8);
        su += q4_dot8(ur, c * 8, xr + c * 8);
    }
    sg = warp_sum(sg);
    su = warp_sum(su);
    if (lane == 0) h[rs * F + f] = siluf_(sg) * su;
}

__global__ void moe_down_k(const float* h, int k, int64_t F, const uint8_t* down, const int32_t* idx, const float* w, int64_t L, float* out) {
    const int64_t r = blockIdx.x;
    const int64_t l = (int64_t) blockIdx.y * M_NW + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (l >= L) return;
    const int64_t row_bytes = F / 32 * Q4B;
    float acc = 0.0f;
    for (int s = 0; s < k; ++s) {
        const int64_t e = idx[r * k + s];
        const uint8_t* dr = down + (e * L + l) * row_bytes;
        const float* hr = h + (r * k + s) * F;
        float d = 0.0f;
        for (int64_t c = lane; c < F / 8; c += 32) d += q4_dot8(dr, c * 8, hr + c * 8);
        acc += w[r * k + s] * warp_sum(d);
    }
    if (lane == 0) out[r * L + l] = acc;
}

// ---------------------------------------------------------------------------------------------------- decoder

__global__ void lookup_k(const int32_t* codes, int64_t T, int64_t ncol, int64_t col, const float* table, int64_t C, const float* bias, int acc,
                         float* out) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * C) return;
    const int64_t t = i / C, c = i % C;
    const float v = table[(int64_t) codes[t * ncol + col] * C + c] + (bias ? bias[c] : 0.0f);
    out[i] = acc ? out[i] + v : v;
}

__global__ void dwconv_causal_k(const float* x, int64_t T, int64_t C, const float* w, int K, const float* bias, float* out) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * C) return;
    const int64_t t = i / C, c = i % C;
    float s = bias ? bias[c] : 0.0f;
    for (int j = 0; j < K; ++j) {
        const int64_t tt = t + j - (K - 1);
        if (tt >= 0) s += w[c * K + j] * x[tt * C + c];
    }
    out[i] = s;
}

__global__ void dwconv_same_k(const float* x, int64_t T, int64_t C, const float* w, int K, const float* bias, float* out) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * C) return;
    const int64_t t = i / C, c = i % C;
    float s = bias ? bias[c] : 0.0f;
    for (int j = 0; j < K; ++j) {
        const int64_t tt = t + j - K / 2;
        if (tt >= 0 && tt < T) s += w[c * K + j] * x[tt * C + c];
    }
    out[i] = s;
}

__global__ void patch_up_k(const float* x, int64_t T, int64_t C, int64_t h, float* out) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * h * C) return;
    const int64_t o = i / C, c = i % C;  // out row o = t h + j
    const int64_t t = o / h, j = o % h;
    out[i] = x[t * C * h + c * h + j];
}

__global__ void spec_k(const float* y, int64_t F, int64_t bins, float maxlog, float* ri) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= F * bins) return;
    const int64_t f = i / bins, b = i % bins;
    const float mag = expf(fminf(y[f * 2 * bins + b], maxlog));
    const float ph = y[f * 2 * bins + bins + b];
    ri[f * 2 * bins + b] = mag * cosf(ph);
    ri[f * 2 * bins + bins + b] = (b == 0 || b == bins - 1) ? 0.0f : mag * sinf(ph);
}

__global__ void ola_k(const float* frames, int64_t F, int64_t n, int64_t hop, const float* win, float* out) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= F * hop) return;
    const int64_t p = i + n / 2;
    int64_t f0 = p - n + 1 <= 0 ? 0 : (p - n + 1 + hop - 1) / hop;
    int64_t f1 = p / hop;
    if (f1 > F - 1) f1 = F - 1;
    float s = 0.0f, env = 0.0f;
    for (int64_t f = f0; f <= f1; ++f) {
        const int64_t j = p - f * hop;
        s += frames[f * n + j] * win[j];
        env += win[j] * win[j];
    }
    out[i] = env > 1e-11f ? s / env : s;
}

__global__ void mixer_k(const float* x0, const float* x1, int64_t L, const float* w, int K, float* out) {
    const int64_t t = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= L) return;
    const int pad = (K - 1) / 2;
    float s = 0.0f;
    for (int j = 0; j < K; ++j) {
        const int64_t tt = t + j - pad;
        if (tt < 0 || tt >= L) continue;
        s += w[j] * x0[tt] + w[K + j] * x1[tt];
    }
    out[t] = s;
}

__global__ void resample_k(const float* x, int64_t L, const float* kern, int64_t orig, int64_t nw, int64_t width, float* out, int64_t total) {
    const int64_t o = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= total) return;
    const int64_t n = o / nw, i = o % nw, KL = 2 * width + orig;
    const float* kr = kern + i * KL;
    float s = 0.0f;
    for (int64_t j = 0; j < KL; ++j) {
        const int64_t src = n * orig + j - width;
        if (src >= 0 && src < L) s += kr[j] * x[src];
    }
    out[o] = s;
}

// ---------------------------------------------------------------------------------------------------- speaker encoder

__global__ void conv3x3_k(const float* x, int64_t Fi, int64_t Ti, int64_t Ci, const float* w, int64_t Co, int64_t a, int64_t sf, int64_t st,
                          const float* bias, int64_t Fo, int64_t To, float* out) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Fo * To * Co) return;
    const int64_t co = i % Co, to = (i / Co) % To, fo = i / (Co * To);
    const int64_t groups = Ci / a, per = Co / groups;
    const int64_t base = (co / per) * a;
    float s = bias ? bias[co] : 0.0f;
    for (int kh = 0; kh < 3; ++kh) {
        const int64_t fi = fo * sf + kh - 1;
        if (fi < 0 || fi >= Fi) continue;
        for (int kw = 0; kw < 3; ++kw) {
            const int64_t ti = to * st + kw - 1;
            if (ti < 0 || ti >= Ti) continue;
            const float* xp = x + (fi * Ti + ti) * Ci + base;
            const float* wp = w + co * a * 9 + kh * 3 + kw;
            for (int64_t j = 0; j < a; ++j) s += wp[j * 9] * xp[j];
        }
    }
    out[i] = s;
}

__global__ void affine_k(const float* x, int64_t n, int64_t C, const float* scale, const float* shift, int relu, const float* add, float* out) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * C) return;
    const int64_t c = i % C;
    float v = x[i] * scale[c] + shift[c];
    if (relu) v = fmaxf(v, 0.0f);
    if (add) v += add[i];
    out[i] = v;
}

__global__ void subsample_k(const float* x, int64_t F, int64_t T, int64_t C, int64_t sf, int64_t st, int64_t Fo, int64_t To, float* out) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Fo * To * C) return;
    const int64_t c = i % C, t = (i / C) % To, f = i / (C * To);
    out[i] = x[((f * sf) * T + t * st) * C + c];
}

__global__ void swap01_k(const float* x, int64_t A, int64_t B, int64_t C, float* out) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= A * B * C) return;
    const int64_t c = i % C, a = (i / C) % A, b = i / (C * A);  // out [B, A, C]
    out[i] = x[(a * B + b) * C + c];
}

__global__ void repeat_rows_k(const float* x, int64_t rep, int64_t T, int64_t C, const float* add, float* out) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * C) return;
    const int64_t t = i / C, c = i % C;
    out[i] = x[(t / rep) * C + c] + (add ? add[i] : 0.0f);
}

__global__ void softmax_rows_k(float* x, int64_t n, int64_t C) {
    const int64_t r = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    float* p = x + r * C;
    float m = -INFINITY;
    for (int64_t c = 0; c < C; ++c) m = fmaxf(m, p[c]);
    float z = 0.0f;
    for (int64_t c = 0; c < C; ++c) {
        p[c] = expf(p[c] - m);
        z += p[c];
    }
    for (int64_t c = 0; c < C; ++c) p[c] /= z;
}

__global__ void mix_experts_k(const float* x, const float* r, int64_t T, int64_t E, int64_t J, float* out) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * J) return;
    const int64_t t = i / J, j = i % J;
    float s = 0.0f;
    for (int64_t e = 0; e < E; ++e) s += r[t * E + e] * x[(t * E + e) * J + j];
    out[i] = s;
}

__global__ void fuse4_k(const float* x0, const float* x1, const float* x2, const float* x3, const float* w, int64_t T, int64_t C, float* out) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * C) return;
    const int64_t c = i % C;
    out[i] = x0[i] * w[c] + x1[i] * w[C + c] + x2[i] * w[2 * C + c] + x3[i] * w[3 * C + c];
}

__global__ void astp_k(const float* x, const float* att, int64_t T, int64_t C, float floor_, float* out) {
    const int64_t c = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float m = -INFINITY;
    for (int64_t t = 0; t < T; ++t) m = fmaxf(m, att[t * C + c]);
    double z = 0.0, s1 = 0.0, s2 = 0.0;
    for (int64_t t = 0; t < T; ++t) {
        const double a = exp((double) (att[t * C + c] - m));
        const double v = x[t * C + c];
        z += a;
        s1 += a * v;
        s2 += a * v * v;
    }
    const double mean = s1 / z, var = s2 / z - mean * mean;
    out[c] = (float) mean;
    out[C + c] = (float) sqrt(var > floor_ ? var : (double) floor_);
}

__global__ void tanh_k(float* x, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = tanhf(x[i]);
}

// ---------------------------------------------------------------------------------------------------- reference encoder

__global__ void snake_cf_k(const float* x, int64_t C, int64_t L, const float* alpha, const float* inv_alpha, float* out) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= C * L) return;
    const int64_t c = i / L;
    const float sn = sinf(alpha[c] * x[i]);
    out[i] = x[i] + inv_alpha[c] * sn * sn;
}

__global__ void add_k(float* x, const float* y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

__global__ void patch_down_k(const float* x, int64_t T, int64_t C, int64_t h, float* out) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t To = T / h;
    if (i >= To * C * h) return;
    const int64_t t = i / (C * h), r = i % (C * h), c = r / h, j = r % h;
    out[i] = x[(t * h + j) * C + c];
}

__global__ void glu_k(const float* x, int64_t T, int64_t C, float* out) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * C) return;
    const int64_t t = i / C, c = i % C;
    out[i] = x[t * 2 * C + c] * sigmoidf_(x[t * 2 * C + C + c]);
}

__global__ void silu_k(float* x, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = siluf_(x[i]);
}

// full attention with Wav2Vec2-BERT's relative positions: score = scale q . k_j + scale q . dist[clip(j - i, -left,
// right) + left], keys with valid[j] == 0 masked; q, k, v [T, H, 64] (rows qs apart), dist [left + right + 1, 64].
// Two passes over the keys (the row's max and sum, then the normalized probabilities times v), so that with bf != 0
// the scores, the probabilities and the products round to bfloat16 where the reference's BF16 recipe rounds them
// (ssl-encoder.cpp: content and relative scores each after the matmul and after the scale, their sum, the
// probabilities).
constexpr int RD = 64, RQ = 64, RK = 32;
__device__ __forceinline__ float bf16r(float x) {
    unsigned u = __float_as_uint(x);
    if ((u & 0x7f800000u) == 0x7f800000u) return x;  // inf, nan
    u += 0x7fffu + ((u >> 16) & 1u);
    return __uint_as_float(u & 0xffff0000u);
}
__global__ void attn_rel_k(const float* q, const float* k, const float* v, int64_t qs, int64_t T, int H, const float* dist, int left, int right,
                           const float* valid, float scale, int bf, float* out) {
    __shared__ float ks[RK][RD], vs[RK][RD];
    const int h = blockIdx.y;
    const int64_t i = (int64_t) blockIdx.x * RQ + threadIdx.x;
    const bool live = i < T;
    float qv[RD], acc[RD];
    float m = -INFINITY, l = 0.0f;
    for (int d = 0; d < RD; ++d) {
        qv[d] = live ? q[i * qs + (int64_t) h * RD + d] : 0.0f;
        acc[d] = 0.0f;
    }
    auto R = [bf](float x) { return bf ? bf16r(x) : x; };
    for (int pass = 0; pass < 2; ++pass) {
        const float inv = l > 0.0f ? 1.0f / l : 0.0f;
        for (int64_t t0 = 0; t0 < T; t0 += RK) {
            __syncthreads();
            for (int e = threadIdx.x; e < RK * RD; e += blockDim.x) {
                const int kk = e / RD, d = e % RD;
                const int64_t j = t0 + kk;
                ks[kk][d] = j < T ? k[j * qs + (int64_t) h * RD + d] : 0.0f;
                vs[kk][d] = j < T ? v[j * qs + (int64_t) h * RD + d] : 0.0f;
            }
            __syncthreads();
            if (!live) continue;
            const int n = (int) min((int64_t) RK, T - t0);
            for (int kk = 0; kk < n; ++kk) {
                const int64_t j = t0 + kk;
                if (valid && valid[j] == 0.0f) continue;
                int64_t r = j - i;
                r = r < -left ? -left : (r > right ? right : r);
                const float* dr = dist + (r + left) * RD;
                float c = 0.0f, rr = 0.0f;
#pragma unroll
                for (int d = 0; d < RD; ++d) {
                    c += qv[d] * ks[kk][d];
                    rr += qv[d] * dr[d];
                }
                const float s = R(R(R(c) * scale) + R(R(rr) * scale));
                if (pass == 0) {
                    if (s > m) {
                        l *= expf(m - s);
                        m = s;
                    }
                    l += expf(s - m);
                } else {
                    const float p = R(expf(s - m) * inv);
#pragma unroll
                    for (int d = 0; d < RD; ++d) acc[d] += p * vs[kk][d];
                }
            }
        }
    }
    if (!live) return;
    for (int d = 0; d < RD; ++d) out[i * (int64_t) H * RD + (int64_t) h * RD + d] = R(acc[d]);
}

__global__ void bf16_k(float* x, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = bf16r(x[i]);
}

// channels-first x [rows, cols] += bias[row]
__global__ void add_rows_k(float* x, const float* bias, int64_t rows, int64_t cols) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * cols) x[i] += bias[i / cols];
}

}  // namespace

extern "C" {

#define Z3_BEGIN(n) do { if ((n) <= 0) return 0; NS_CU(cudaSetDevice(g->dev)); } while (0)

int ns_audio_z3_embed(ns_gpu* g, const void* audio, int64_t ncb, int64_t va, const void* text, int64_t vt, const int32_t* ids, int64_t rows, int64_t D,
                      float* out) {
    Z3_BEGIN(rows * D);
    embed_k<<<blocks(rows * D), 256, 0, g->q>>>((const __half*) audio, (int) ncb, va, (const __half*) text, vt, ids, rows, D, out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_z3_rms(ns_gpu* g, const float* x, int64_t xs, int64_t rows, int64_t C, const float* w, float eps, float* out) {
    Z3_BEGIN(rows * C);
    rms_k<<<(unsigned) rows, 256, 0, g->q>>>(x, xs, C, w, eps, out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_z3_attnres_score(ns_gpu* g, const float* src, int64_t rows, int64_t D, const float* q, int64_t Q, float eps, float* scores) {
    Z3_BEGIN(rows);
    attnres_score_k<<<(unsigned) rows, 256, 0, g->q>>>(src, D, q, Q, eps, scores);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_z3_attnres_mix(ns_gpu* g, const float* bank, const float* scores, int64_t S, int64_t cap, int64_t rows, int64_t D, int64_t Q, int64_t qi,
                            float* out) {
    Z3_BEGIN(rows);
    if (S < 1 || S > 64) return ns_fail("ns_audio_z3_attnres_mix: 1 to 64 sources");
    attnres_mix_k<<<(unsigned) rows, 256, 0, g->q>>>(bank, scores, (int) S, cap, D, Q, qi, out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_z3_add_vec(ns_gpu* g, float* x, int64_t rows, int64_t C, const float* v) {
    Z3_BEGIN(rows * C);
    add_vec_k<<<blocks(rows * C), 256, 0, g->q>>>(x, C, v, rows * C);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_z3_gdn_conv(ns_gpu* g, const float* proj, int64_t rows, int64_t P, int64_t C, const float* w, int64_t K, float* hist, float* out) {
    Z3_BEGIN(rows * C);
    if (K < 2 || K > MAXK) return ns_fail("ns_audio_z3_gdn_conv: a kernel of 2 to 8 taps");
    gdn_conv_k<<<blocks(C, 128), 128, 0, g->q>>>(proj, rows, P, C, w, (int) K, hist, out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_z3_gdn_prep(ns_gpu* g, const float* conv, const float* proj, int64_t rows, int64_t P, int64_t ab0, int64_t H, int64_t S, const float* a_log,
                         const float* dt_bias, float* q, float* k, float* v, float* decay, float* beta) {
    Z3_BEGIN(rows);
    if (S % 32 || S > 1024) return ns_fail("ns_audio_z3_gdn_prep: heads of a multiple of 32 up to 1024");
    gdn_prep_k<<<(unsigned) (rows * H), (unsigned) S, 0, g->q>>>(conv, proj, P, ab0, (int) H, (int) S, a_log, dt_bias, q, k, v, decay, beta);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_z3_gdn_scan(ns_gpu* g, const float* q, const float* k, const float* v, const float* decay, const float* beta, int64_t rows, int64_t H,
                         int64_t S, float* state, float* y) {
    Z3_BEGIN(rows);
    if (S != GS) return ns_fail("ns_audio_z3_gdn_scan: heads of 128");
    gdn_scan_k<<<dim3((unsigned) H, GS / G_NW), WARP * G_NW, 0, g->q>>>(q, k, v, decay, beta, rows, (int) H, state, y);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_z3_gdn_out(ns_gpu* g, const float* y, const float* proj, int64_t P, int64_t z0, int64_t rows, int64_t H, int64_t S, const float* w, float eps,
                        float* out) {
    Z3_BEGIN(rows);
    if (S % 32 || S > 1024) return ns_fail("ns_audio_z3_gdn_out: heads of a multiple of 32 up to 1024");
    gdn_out_k<<<(unsigned) (rows * H), (unsigned) S, 0, g->q>>>(y, proj, P, z0, (int) H, (int) S, w, eps, out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_z3_head_rms(ns_gpu* g, float* x, int64_t xs, int64_t rows, int64_t H, int64_t D, const float* scale, float eps) {
    Z3_BEGIN(rows * H);
    head_rms_k<<<blocks(rows * H * WARP), 256, 0, g->q>>>(x, xs, rows, (int) H, (int) D, scale, eps);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_z3_rope(ns_gpu* g, float* x, int64_t xs, int64_t rows, int64_t H, int64_t D, float theta, int64_t pos0) {
    Z3_BEGIN(rows * H * D / 2);
    rope_k<<<blocks(rows * H * D / 2), 256, 0, g->q>>>(x, xs, rows, (int) H, (int) D, theta, pos0);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_z3_attention(ns_gpu* g, const float* q, int64_t qs, int64_t rows, const float* k, const float* v, int64_t kvs, int64_t keys, int64_t Hq,
                          int64_t Hk, int64_t D, float scale, int64_t pos0, int causal, int64_t window, float* out) {
    Z3_BEGIN(rows);
    if (keys <= 0) return ns_fail("ns_audio_z3_attention: no keys");
    if (Hk <= 0 || Hq % Hk) return ns_fail("ns_audio_z3_attention: query heads a multiple of the key heads");
    const dim3 grid((unsigned) ((rows + QB - 1) / QB), (unsigned) Hq);
    switch (D) {
    case 16: attention_k<16><<<grid, QB, 0, g->q>>>(q, qs, rows, k, v, kvs, keys, (int) Hq, (int) Hk, scale, pos0, causal, window, out); break;
    case 32: attention_k<32><<<grid, QB, 0, g->q>>>(q, qs, rows, k, v, kvs, keys, (int) Hq, (int) Hk, scale, pos0, causal, window, out); break;
    case 64: attention_k<64><<<grid, QB, 0, g->q>>>(q, qs, rows, k, v, kvs, keys, (int) Hq, (int) Hk, scale, pos0, causal, window, out); break;
    default: return ns_fail("ns_audio_z3_attention: heads of 16, 32 or 64");
    }
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_z3_head_gate(ns_gpu* g, float* x, const float* gate, int64_t rows, int64_t H, int64_t D) {
    Z3_BEGIN(rows * H * D);
    head_gate_k<<<blocks(rows * H * D), 256, 0, g->q>>>(x, gate, rows * H * D, H, D);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_z3_softcap(ns_gpu* g, float* x, int64_t n, float cap) {
    Z3_BEGIN(n);
    softcap_k<<<blocks(n), 256, 0, g->q>>>(x, n, cap);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_z3_silu_mul(ns_gpu* g, const float* a, const float* b, float* out, int64_t n) {
    Z3_BEGIN(n);
    silu_mul_k<<<blocks(n), 256, 0, g->q>>>(a, b, out, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_z3_topk(ns_gpu* g, const float* logits, const float* bias, int64_t rows, int64_t E, int64_t k, int32_t* idx, float* w) {
    Z3_BEGIN(rows);
    if (k < 1 || k > 8 || k > E) return ns_fail("ns_audio_z3_topk: 1 to 8 experts a row");
    topk_k<<<blocks(rows, 64), 64, 0, g->q>>>(logits, bias, rows, (int) E, (int) k, idx, w);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_z3_moe_up(ns_gpu* g, const float* x, int64_t rows, int64_t L, const void* gate, const void* up, const int32_t* idx, int64_t k, int64_t F,
                       float* h) {
    Z3_BEGIN(rows);
    if (L % 32) return ns_fail("ns_audio_z3_moe_up: rows of whole Q4_0 blocks");
    moe_up_k<<<dim3((unsigned) (rows * k), (unsigned) ((F + M_NW - 1) / M_NW)), WARP * M_NW, 0, g->q>>>(x, L, (const uint8_t*) gate, (const uint8_t*) up, idx,
                                                                                                        (int) k, F, h);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_z3_moe_down(ns_gpu* g, const float* h, int64_t rows, int64_t k, int64_t F, const void* down, const int32_t* idx, const float* w, int64_t L,
                         float* out) {
    Z3_BEGIN(rows);
    if (F % 32) return ns_fail("ns_audio_z3_moe_down: rows of whole Q4_0 blocks");
    moe_down_k<<<dim3((unsigned) rows, (unsigned) ((L + M_NW - 1) / M_NW)), WARP * M_NW, 0, g->q>>>(h, (int) k, F, (const uint8_t*) down, idx, w, L, out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_z3_lookup(ns_gpu* g, const int32_t* codes, int64_t T, int64_t ncol, int64_t col, const float* table, int64_t C, const float* bias, int acc,
                       float* out) {
    Z3_BEGIN(T * C);
    lookup_k<<<blocks(T * C), 256, 0, g->q>>>(codes, T, ncol, col, table, C, bias, acc, out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_z3_dwconv_causal(ns_gpu* g, const float* x, int64_t T, int64_t C, const float* w, int64_t K, const float* bias, float* out) {
    Z3_BEGIN(T * C);
    dwconv_causal_k<<<blocks(T * C), 256, 0, g->q>>>(x, T, C, w, (int) K, bias, out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_z3_dwconv_same(ns_gpu* g, const float* x, int64_t T, int64_t C, const float* w, int64_t K, const float* bias, float* out) {
    Z3_BEGIN(T * C);
    if (K % 2 == 0) return ns_fail("ns_audio_z3_dwconv_same: an odd kernel");
    dwconv_same_k<<<blocks(T * C), 256, 0, g->q>>>(x, T, C, w, (int) K, bias, out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_z3_patch_up(ns_gpu* g, const float* x, int64_t T, int64_t C, int64_t h, float* out) {
    Z3_BEGIN(T * C * h);
    patch_up_k<<<blocks(T * C * h), 256, 0, g->q>>>(x, T, C, h, out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_z3_spec(ns_gpu* g, const float* y, int64_t F, int64_t bins, float maxlog, float* ri) {
    Z3_BEGIN(F * bins);
    spec_k<<<blocks(F * bins), 256, 0, g->q>>>(y, F, bins, maxlog, ri);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_z3_ola(ns_gpu* g, const float* frames, int64_t F, int64_t n, int64_t hop, const float* win, float* out) {
    Z3_BEGIN(F * hop);
    ola_k<<<blocks(F * hop), 256, 0, g->q>>>(frames, F, n, hop, win, out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_z3_mixer(ns_gpu* g, const float* x0, const float* x1, int64_t L, const float* w, int64_t K, float* out) {
    Z3_BEGIN(L);
    mixer_k<<<blocks(L), 256, 0, g->q>>>(x0, x1, L, w, (int) K, out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_z3_resample(ns_gpu* g, const float* x, int64_t L, const float* kern, int64_t orig, int64_t nw, int64_t width, float* out, int64_t total) {
    Z3_BEGIN(total);
    resample_k<<<blocks(total), 256, 0, g->q>>>(x, L, kern, orig, nw, width, out, total);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_z3_conv3x3(ns_gpu* g, const float* x, int64_t Fi, int64_t Ti, int64_t Ci, const float* w, int64_t Co, int64_t a, int64_t sf, int64_t st,
                        const float* bias, float* out) {
    const int64_t Fo = (Fi - 1) / sf + 1, To = (Ti - 1) / st + 1;
    Z3_BEGIN(Fo * To * Co);
    if (a <= 0 || Ci % a || Co % (Ci / a)) return ns_fail("ns_audio_z3_conv3x3: the groups do not divide the channels");
    conv3x3_k<<<blocks(Fo * To * Co), 256, 0, g->q>>>(x, Fi, Ti, Ci, w, Co, a, sf, st, bias, Fo, To, out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_z3_affine(ns_gpu* g, const float* x, int64_t n, int64_t C, const float* scale, const float* shift, int relu, const float* add, float* out) {
    Z3_BEGIN(n * C);
    affine_k<<<blocks(n * C), 256, 0, g->q>>>(x, n, C, scale, shift, relu, add, out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_z3_subsample(ns_gpu* g, const float* x, int64_t F, int64_t T, int64_t C, int64_t sf, int64_t st, float* out) {
    const int64_t Fo = (F + sf - 1) / sf, To = (T + st - 1) / st;
    Z3_BEGIN(Fo * To * C);
    subsample_k<<<blocks(Fo * To * C), 256, 0, g->q>>>(x, F, T, C, sf, st, Fo, To, out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_z3_swap01(ns_gpu* g, const float* x, int64_t A, int64_t B, int64_t C, float* out) {
    Z3_BEGIN(A * B * C);
    swap01_k<<<blocks(A * B * C), 256, 0, g->q>>>(x, A, B, C, out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_z3_repeat_rows(ns_gpu* g, const float* x, int64_t rep, int64_t T, int64_t C, const float* add, float* out) {
    Z3_BEGIN(T * C);
    repeat_rows_k<<<blocks(T * C), 256, 0, g->q>>>(x, rep, T, C, add, out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_z3_softmax(ns_gpu* g, float* x, int64_t n, int64_t C) {
    Z3_BEGIN(n);
    softmax_rows_k<<<blocks(n), 256, 0, g->q>>>(x, n, C);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_z3_mix_experts(ns_gpu* g, const float* x, const float* r, int64_t T, int64_t E, int64_t J, float* out) {
    Z3_BEGIN(T * J);
    mix_experts_k<<<blocks(T * J), 256, 0, g->q>>>(x, r, T, E, J, out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_z3_fuse4(ns_gpu* g, const float* x0, const float* x1, const float* x2, const float* x3, const float* w, int64_t T, int64_t C, float* out) {
    Z3_BEGIN(T * C);
    fuse4_k<<<blocks(T * C), 256, 0, g->q>>>(x0, x1, x2, x3, w, T, C, out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_z3_astp(ns_gpu* g, const float* x, const float* att, int64_t T, int64_t C, float floor, float* out) {
    Z3_BEGIN(C);
    if (T <= 0) return ns_fail("ns_audio_z3_astp: no frames");
    astp_k<<<blocks(C, 128), 128, 0, g->q>>>(x, att, T, C, floor, out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_z3_tanh(ns_gpu* g, float* x, int64_t n) {
    Z3_BEGIN(n);
    tanh_k<<<blocks(n), 256, 0, g->q>>>(x, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_z3_snake_cf(ns_gpu* g, const float* x, int64_t C, int64_t L, const float* alpha, const float* inv_alpha, float* out) {
    Z3_BEGIN(C * L);
    snake_cf_k<<<blocks(C * L), 256, 0, g->q>>>(x, C, L, alpha, inv_alpha, out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_z3_add(ns_gpu* g, float* x, const float* y, int64_t n) {
    Z3_BEGIN(n);
    add_k<<<blocks(n), 256, 0, g->q>>>(x, y, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_z3_patch_down(ns_gpu* g, const float* x, int64_t T, int64_t C, int64_t h, float* out) {
    Z3_BEGIN(T / h * C * h);
    patch_down_k<<<blocks(T / h * C * h), 256, 0, g->q>>>(x, T, C, h, out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_z3_glu(ns_gpu* g, const float* x, int64_t T, int64_t C, float* out) {
    Z3_BEGIN(T * C);
    glu_k<<<blocks(T * C), 256, 0, g->q>>>(x, T, C, out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_z3_silu(ns_gpu* g, float* x, int64_t n) {
    Z3_BEGIN(n);
    silu_k<<<blocks(n), 256, 0, g->q>>>(x, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_z3_attn_rel(ns_gpu* g, const float* q, const float* k, const float* v, int64_t qs, int64_t T, int64_t H, int64_t D, const float* dist,
                         int64_t left, int64_t right, const float* valid, float scale, int bf, float* out) {
    Z3_BEGIN(T);
    if (D != RD) return ns_fail("ns_audio_z3_attn_rel: heads of 64");
    attn_rel_k<<<dim3((unsigned) ((T + RQ - 1) / RQ), (unsigned) H), RQ, 0, g->q>>>(q, k, v, qs, T, (int) H, dist, (int) left, (int) right, valid, scale, (int) bf, out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_z3_bf16(ns_gpu* g, float* x, int64_t n) {
    Z3_BEGIN(n);
    bf16_k<<<blocks(n), 256, 0, g->q>>>(x, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_z3_add_rows(ns_gpu* g, float* x, const float* bias, int64_t rows, int64_t cols) {
    Z3_BEGIN(rows * cols);
    add_rows_k<<<blocks(rows * cols), 256, 0, g->q>>>(x, bias, rows, cols);
    NS_LAUNCH_CHECK();
    return 0;
}

}  // extern "C"
