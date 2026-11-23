// vi.cu: the Voice Isolator engine's own kernels (audio/voiceisolator) - what the shared ones do not have: the token
// transposes between the time and band paths, the band embedding and the decoder's conditioning, rotary positions,
// exclusive self attention, attention within a frame's few bands, and the Mamba layers' convolution, step sizes' softplus and
// selective scan.
//
// Origin: the arithmetic of Inworld's voice isolator (github.com/inworld-ai/voice-isolator: src/inworld/voice_isolation
// model.py and scan.py, Apache-2.0), written anew for CUDA. The scan keeps a channel's states in registers as the
// reference's Metal kernel does, with the reference's per-step order (decay, drive, read out, skip, gate).
//
// Precise arithmetic (kernels/build.sh builds audio/ without --use_fast_math).
#include "ns.h"
#include "ns_internal.hpp"
#include "vi.h"

#include <cmath>

namespace {

unsigned blocks(int64_t n, int t = 256) { return (unsigned) ((n + t - 1) / t); }

__device__ inline float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ inline float siluf_(float x) { return x / (1.0f + expf(-x)); }

__global__ void swap01_k(const float* x, int64_t A, int64_t B, int64_t W, float* out, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t w = i % W, r = i / W, a = r % A, b = r / A;  // out row (b, a)
    out[i] = x[(a * B + b) * W + w];
}

__global__ void add_seq_k(float* x, int64_t L, int64_t W, const float* v, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t w = i % W, s = i / W / L;
    x[i] += v[s * W + w];
}

__global__ void film_k(float* x, int64_t W, const float* scale, const float* shift, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t w = i % W;
    x[i] = x[i] * (1.0f + scale[w]) + shift[w];
}

// one thread a (row, q|k, head, pair)
__global__ void rope_k(float* qkv, int64_t L, int64_t H, int64_t D, const float* cs, const float* sn, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t half = D / 2, p = i % half, h = (i / half) % H, which = (i / half / H) % 2, r = i / half / H / 2;
    float* x = qkv + r * 3 * H * D + which * H * D + h * D;
    const int64_t pos = r % L;
    const float c = cs[pos * half + p], s = sn[pos * half + p];
    const float a = x[p], b = x[half + p];
    x[p] = a * c - b * s;
    x[half + p] = a * s + b * c;
}

// a warp a (row, head)
__global__ void exclusive_k(float* o, const float* v, int64_t vs, int64_t rows, int64_t H, int64_t D) {
    const int64_t wi = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x & 31;
    if (wi >= rows * H) return;
    const int64_t r = wi / H, h = wi % H;
    float* op = o + r * H * D + h * D;
    const float* vp = v + r * vs + h * D;
    float ov = 0.0f, vv = 0.0f;
    for (int64_t d = lane; d < D; d += 32) {
        ov += op[d] * vp[d];
        vv += vp[d] * vp[d];
    }
    ov = warp_sum(ov);
    vv = warp_sum(vv);
    const float coef = ov / fmaxf(vv, 1e-6f);
    for (int64_t d = lane; d < D; d += 32) op[d] = op[d] - coef * vp[d];
}

// a block a (sequence, head): q, k, v and the score table in shared memory
__global__ void short_attention_k(const float* qkv, int64_t S, int64_t H, int64_t D, float scale, float* out) {
    extern __shared__ float sm[];
    // q and k rows padded by one: a warp's threads reading one column of different rows hit different banks
    const int64_t P = D + 1;
    float* q = sm;
    float* k = q + S * P;
    float* v = k + S * P;
    float* p = v + S * D;
    const int64_t seq = blockIdx.x / H, h = blockIdx.x % H;
    const int64_t ld = 3 * H * D;
    const float* base = qkv + seq * S * ld + h * D;
    for (int64_t i = threadIdx.x; i < S * D; i += blockDim.x) {
        const int64_t s = i / D, d = i % D;
        q[s * P + d] = base[s * ld + d];
        k[s * P + d] = base[s * ld + H * D + d];
        v[i] = base[s * ld + 2 * H * D + d];
    }
    __syncthreads();
    for (int64_t i = threadIdx.x; i < S * S; i += blockDim.x) {
        const int64_t a = i / S, b = i % S;
        float acc = 0.0f;
        for (int64_t d = 0; d < D; ++d) acc += q[a * P + d] * k[b * P + d];
        p[i] = acc * scale;
    }
    __syncthreads();
    for (int64_t a = threadIdx.x; a < S; a += blockDim.x) {
        float m = -INFINITY;
        for (int64_t b = 0; b < S; ++b) m = fmaxf(m, p[a * S + b]);
        float sum = 0.0f;
        for (int64_t b = 0; b < S; ++b) {
            const float e = expf(p[a * S + b] - m);
            p[a * S + b] = e;
            sum += e;
        }
        const float inv = 1.0f / sum;
        for (int64_t b = 0; b < S; ++b) p[a * S + b] *= inv;
    }
    __syncthreads();
    for (int64_t i = threadIdx.x; i < S * D; i += blockDim.x) {
        const int64_t a = i / D, d = i % D;
        float acc = 0.0f;
        for (int64_t b = 0; b < S; ++b) acc += p[a * S + b] * v[b * D + d];
        out[(seq * S + a) * H * D + h * D + d] = acc;
    }
}

__global__ void dup_k(const float* x, int64_t W, float* cat, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t r = i / W, w = i % W;
    const float v = x[i];
    cat[r * 2 * W + w] = v;
    cat[r * 2 * W + W + w] = v;
}

__global__ void conv_silu_k(const float* xz, int64_t P, int64_t L, int64_t C, const float* w, int K, const float* b, int reverse, float* out,
                            int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t c = i % C, r = i / C, t = r % L, s = r / L;
    const float* u = xz + s * L * P + c;
    float acc = b[c];
    for (int k = 0; k < K; ++k) {
        const int64_t tt = reverse ? t + (K - 1) - k : t - (K - 1) + k;
        if (tt >= 0 && tt < L) acc += w[c * K + k] * u[tt * P];
    }
    out[i] = siluf_(acc);
}

__global__ void softplus_k(float* x, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    x[i] = v > 20.0f ? v : log1pf(expf(v));
}

constexpr int MAXN = 16;

__global__ void scan_k(const float* u, const float* delta, const float* a_log, const float* dd, const float* xdbl, int64_t xs, int64_t bc,
                       const float* xz, int64_t zs, int64_t zc, int64_t S, int64_t L, int64_t C, int N, int reverse, float* y) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= S * C) return;
    const int64_t s = i / C, c = i % C;
    float h[MAXN], a[MAXN];
    for (int n = 0; n < N; ++n) {
        a[n] = -expf(a_log[c * N + n]);
        h[n] = 0.0f;
    }
    const float skip = dd[c];
    for (int64_t k = 0; k < L; ++k) {
        const int64_t t = reverse ? L - 1 - k : k;
        const int64_t r = s * L + t;
        const float step = delta[r * C + c], x = u[r * C + c];
        const float drive = step * x;
        const float* B = xdbl + r * xs + bc;
        const float* Cc = B + N;
        float acc = 0.0f;
        for (int n = 0; n < N; ++n) {
            h[n] = expf(step * a[n]) * h[n] + drive * B[n];
            acc += h[n] * Cc[n];
        }
        acc += x * skip;
        y[r * C + c] = acc * siluf_(xz[r * zs + zc + c]);
    }
}

}  // namespace

#define VI_BEGIN(n) do { if ((n) <= 0) return 0; NS_CU(cudaSetDevice(g->dev)); } while (0)

int ns_audio_vi_swap01(ns_gpu* g, const float* x, int64_t A, int64_t B, int64_t W, float* out) {
    const int64_t n = A * B * W;
    VI_BEGIN(n);
    swap01_k<<<blocks(n), 256, 0, g->q>>>(x, A, B, W, out, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_vi_add_seq(ns_gpu* g, float* x, int64_t S, int64_t L, int64_t W, const float* v) {
    const int64_t n = S * L * W;
    VI_BEGIN(n);
    add_seq_k<<<blocks(n), 256, 0, g->q>>>(x, L, W, v, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_vi_film(ns_gpu* g, float* x, int64_t rows, int64_t W, const float* scale, const float* shift) {
    const int64_t n = rows * W;
    VI_BEGIN(n);
    film_k<<<blocks(n), 256, 0, g->q>>>(x, W, scale, shift, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_vi_rope(ns_gpu* g, float* qkv, int64_t rows, int64_t L, int64_t H, int64_t D, const float* cs, const float* sn) {
    const int64_t n = rows * 2 * H * (D / 2);
    VI_BEGIN(n);
    if (D % 2 || L < 1) return ns_fail("ns_audio_vi_rope: an even head size and a positive length");
    rope_k<<<blocks(n), 256, 0, g->q>>>(qkv, L, H, D, cs, sn, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_vi_exclusive(ns_gpu* g, float* o, const float* v, int64_t vs, int64_t rows, int64_t H, int64_t D) {
    VI_BEGIN(rows * H);
    exclusive_k<<<blocks(rows * H * 32), 256, 0, g->q>>>(o, v, vs, rows, H, D);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_vi_short_attention(ns_gpu* g, const float* qkv, int64_t seqs, int64_t S, int64_t H, int64_t D, float* out) {
    VI_BEGIN(seqs * S);
    const size_t shm = (size_t) (2 * S * (D + 1) + S * D + S * S) * sizeof(float);
    if (shm > 48 * 1024) return ns_fail("ns_audio_vi_short_attention: a sequence too long for shared memory (" + std::to_string(S) + " tokens)");
    short_attention_k<<<(unsigned) (seqs * H), 128, shm, g->q>>>(qkv, S, H, D, 1.0f / sqrtf((float) D), out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_vi_dup(ns_gpu* g, const float* x, int64_t rows, int64_t W, float* cat) {
    const int64_t n = rows * W;
    VI_BEGIN(n);
    dup_k<<<blocks(n), 256, 0, g->q>>>(x, W, cat, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_vi_conv_silu(ns_gpu* g, const float* xz, int64_t P, int64_t S, int64_t L, int64_t C, const float* w, int64_t K, const float* b,
                          int reverse, float* out) {
    const int64_t n = S * L * C;
    VI_BEGIN(n);
    conv_silu_k<<<blocks(n), 256, 0, g->q>>>(xz, P, L, C, w, (int) K, b, reverse, out, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_vi_softplus(ns_gpu* g, float* x, int64_t n) {
    VI_BEGIN(n);
    softplus_k<<<blocks(n), 256, 0, g->q>>>(x, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_vi_scan(ns_gpu* g, const float* u, const float* delta, const float* a_log, const float* d, const float* xdbl, int64_t xs, int64_t bc,
                     const float* xz, int64_t zs, int64_t zc, int64_t S, int64_t L, int64_t C, int64_t N, int reverse, float* y) {
    VI_BEGIN(S * C);
    if (N < 1 || N > MAXN) return ns_fail("ns_audio_vi_scan: 1 to 16 states");
    scan_k<<<blocks(S * C, 128), 128, 0, g->q>>>(u, delta, a_log, d, xdbl, xs, bc, xz, zs, zc, S, L, C, (int) N, reverse, y);
    NS_LAUNCH_CHECK();
    return 0;
}
