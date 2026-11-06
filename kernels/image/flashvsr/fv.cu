// fv.cu: the FlashVSR engine's own kernels - what the shared diffusion kernels (nsd) do not have: the DiT's tokens in
// window order (patches in and out, the LQ projection's residual), its norms and gated residuals, the block-sparse
// attention (the draft scores between block means, the top-k threshold, the lists of selected key blocks, and the
// attention over them: nsd_flash.inc's FlashAttention-2 kernel walking a list), the LQ projection's causal 3-D
// convolution unfolded, the tiny decoder's unfolded 3 x 3 convolutions (a nearest 2x upsample and a MemBlock's past
// read in place), the colour fix and the noise. On the GPU's stream.
#include "ns.h"
#include "ns_internal.hpp"
#include "fv.h"

#include <cuda_fp16.h>
#include <cfloat>
#include <cmath>

namespace {

constexpr int WG = 256;

__device__ inline float h2f(__half h) { return __half2float(h); }
__device__ inline __half f2h(float f) { return __float2half_rn(f); }

inline unsigned nblk(int64_t n) { return (unsigned) ((n + WG - 1) / WG); }

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

__device__ float block_max(float v) {
    __shared__ float s[WG / 32];
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    if ((threadIdx.x & 31) == 0) s[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = s[0];
    for (int i = 1; i < WG / 32; ++i) t = fmaxf(t, s[i]);
    __syncthreads();
    return t;
}

// token index (window order) of patch (frame t, y, x) in a grid of f x hp x wp patches
__device__ inline int64_t win_index(int64_t t, int64_t y, int64_t x, int64_t hp, int64_t wp) {
    const int64_t nbw = wp / 8, nb = (hp / 8) * nbw;
    const int64_t g = t >> 1, ft = t & 1;
    const int64_t b = (y >> 3) * nbw + (x >> 3);
    return ((g * nb + b) << 7) + ft * 64 + (y & 7) * 8 + (x & 7);
}

__global__ void patchify_k(const float* lat, int64_t f, int64_t h8, int64_t w8, __half* out) {
    const int64_t hp = h8 / 2, wp = w8 / 2, n = f * hp * wp * 64;
    const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    const int64_t k = id % 64, p = id / 64;
    const int64_t x = p % wp, y = (p / wp) % hp, t = p / (wp * hp);
    const int64_t c = k / 4, ky = (k / 2) % 2, kx = k % 2;
    const float v = lat[((c * f + t) * h8 + 2 * y + ky) * w8 + 2 * x + kx];
    out[win_index(t, y, x, hp, wp) * 64 + k] = f2h(v);
}

__global__ void unpatchify_k(const float* pred, const float* noise, int64_t f, int64_t h8, int64_t w8, float* out) {
    const int64_t n = 16 * f * h8 * w8;
    const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    const int64_t xx = id % w8, yy = (id / w8) % h8, t = (id / (w8 * h8)) % f, c = id / (w8 * h8 * f);
    const int64_t hp = h8 / 2, wp = w8 / 2;
    // the head's features are (ky, kx, c): Wan's unpatchify, '(x y z c)'
    const int64_t k = ((yy % 2) * 2 + (xx % 2)) * 16 + c;
    out[id] = noise[id] - pred[win_index(t, yy / 2, xx / 2, hp, wp) * 64 + k];
}

__global__ void add_lq_k(float* x, const float* lq, int64_t f, int64_t hp, int64_t wp, int64_t C) {
    const int64_t n = f * hp * wp * C;
    const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    const int64_t c = id % C, p = id / C;
    const int64_t xx = p % wp, yy = (p / wp) % hp, t = p / (wp * hp);
    x[win_index(t, yy, xx, hp, wp) * C + c] += lq[id];
}

__global__ void ln_k(const float* x, int64_t C, float eps, const float* a, const float* b, __half* out) {
    const float* row = x + (size_t) blockIdx.x * C;
    float s = 0.0f;
    for (int64_t i = threadIdx.x; i < C; i += WG) s += row[i];
    const float mean = block_sum(s) / (float) C;
    float s2 = 0.0f;
    for (int64_t i = threadIdx.x; i < C; i += WG) { const float d = row[i] - mean; s2 += d * d; }
    const float inv = rsqrtf(block_sum(s2) / (float) C + eps);
    __half* o = out + (size_t) blockIdx.x * C;
    for (int64_t i = threadIdx.x; i < C; i += WG) {
        float v = (row[i] - mean) * inv;
        if (a) v *= a[i];
        if (b) v += b[i];
        o[i] = f2h(v);
    }
}

__global__ void gate_add_k(float* x, const float* y, int64_t n, int64_t C, const float* gate) {
    const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    x[id] += gate ? y[id] * gate[id % C] : y[id];
}

__global__ void block_mean_k(const __half* x, int64_t stride, int64_t C, float* out) {
    const int64_t blk = blockIdx.x;
    for (int64_t c = threadIdx.x; c < C; c += blockDim.x) {
        float s = 0.0f;
        for (int r = 0; r < 128; ++r) s += h2f(x[(blk * 128 + r) * stride + c]);
        out[blk * C + c] = s / 128.0f;
    }
}

__global__ void block_probs_k(const float* qm, const float* km, int64_t nq, int64_t nk, int64_t H, int64_t nbh, int64_t nbw, int64_t lr,
                              float* probs) {
    extern __shared__ float sc[];   // nk scores
    __shared__ float qv[128];
    const int64_t i = blockIdx.x, h = blockIdx.y, C = H * 128, nb = nbh * nbw;
    if (threadIdx.x < 128) qv[threadIdx.x] = qm[i * C + h * 128 + threadIdx.x];
    __syncthreads();
    const int64_t qb = i % nb, qr = qb / nbw, qc = qb % nbw, half = lr / 2;
    const int64_t r0 = qr - half, c0 = qc - half;
    const float scale = 1.0f / sqrtf(128.0f);
    float mx = -INFINITY;
    for (int64_t j = threadIdx.x; j < nk; j += WG) {
        const int64_t kb = j % nb, kr = kb / nbw, kc = kb % nbw;
        float s = -INFINITY;
        if (kr >= r0 && kr <= r0 + lr - 1 && kc >= c0 && kc <= c0 + lr - 1) {
            const float* kv = km + j * C + h * 128;
            float d = 0.0f;
            for (int e = 0; e < 128; ++e) d += qv[e] * kv[e];
            s = d * scale;
        }
        sc[j] = s;
        mx = fmaxf(mx, s);
    }
    mx = block_max(mx);
    float sum = 0.0f;
    for (int64_t j = threadIdx.x; j < nk; j += WG) {
        const float e = sc[j] == -INFINITY ? 0.0f : expf(sc[j] - mx);
        sc[j] = e;
        sum += e;
    }
    sum = block_sum(sum);
    float* o = probs + (h * nq + i) * nk;
    for (int64_t j = threadIdx.x; j < nk; j += WG) o[j] = sc[j] / sum;
}

// the rank-th largest of a segment's values (all >= 0, so their bits order as they do): a radix select, 8 bits a pass
__global__ void topk_thr_k(const float* p, int64_t n, int64_t rank, float* thr) {
    __shared__ unsigned hist[256];
    __shared__ unsigned s_prefix;
    __shared__ long long s_rank;
    const float* d = p + (size_t) blockIdx.x * n;
    unsigned prefix = 0, mask = 0;
    long long r = rank;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
        __syncthreads();
        for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
            const unsigned u = __float_as_uint(fmaxf(d[i], 0.0f));
            if ((u & mask) == prefix) atomicAdd(&hist[(u >> shift) & 255u], 1u);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            long long acc = 0;
            int dg = 0;
            for (dg = 255; dg > 0; --dg) {
                if (acc + hist[dg] >= r) break;
                acc += hist[dg];
            }
            s_prefix = prefix | ((unsigned) dg << shift);
            s_rank = r - acc;
        }
        __syncthreads();
        prefix = s_prefix;
        r = s_rank;
        mask |= 255u << shift;
        __syncthreads();
    }
    if (threadIdx.x == 0) thr[blockIdx.x] = __uint_as_float(prefix);
}

__global__ void block_lists_k(const float* probs, const float* thr, int64_t H, int64_t nq, int64_t nk, int64_t nb, int32_t* lists, int32_t* counts) {
    const int64_t w = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (w >= H * nq) return;
    const int64_t h = w / nq, i = w % nq, G = nq / nb;
    const float t = thr[h * G + i / nb];
    const float* row = probs + w * nk;
    int32_t* out = lists + w * nk;
    int cnt = 0;
    for (int64_t j0 = 0; j0 < nk; j0 += 32) {
        const int64_t j = j0 + lane;
        const bool s = j < nk && row[j] > t;
        const unsigned b = __ballot_sync(0xffffffffu, s);
        if (s) out[cnt + __popc(b & ((1u << lane) - 1u))] = (int32_t) j;
        cnt += __popc(b);
    }
    if (lane == 0) counts[w] = cnt;
}

// ---- block-sparse attention: nsd_flash.inc's kernel, its key tiles taken from a list of 128-key blocks ----

constexpr int FA_D = 128, FA_BQ = 64, FA_BK = 64, FA_LDK = 136;

__device__ __forceinline__ void fa_mma16816(float c[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t fa_pack(float a, float b) {
    __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<uint32_t*>(&h);
}

__global__ void __launch_bounds__(128) sparse_k(const __half* __restrict__ q, int Sq, int H, const __half* __restrict__ kc, const __half* __restrict__ vc,
                                                const __half* __restrict__ kt, const __half* __restrict__ vt, int split, const int32_t* __restrict__ lists,
                                                const int32_t* __restrict__ counts, int nk, int nq, __half* __restrict__ out, float scale_log2) {
    extern __shared__ __half smem[];
    __half* Ks = smem;
    __half* Vs = smem + FA_BK * FA_LDK;
    const int h = blockIdx.y;
    const long ld = (long) H * FA_D;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int qrow0 = blockIdx.x * FA_BQ + warp * 16;
    const int r0 = qrow0 + g, r1 = r0 + 8;
    const int qb = (blockIdx.x * FA_BQ) / 128;

    uint32_t qa[8][4];
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
        const int d = kk * 16 + 2 * t;
        qa[kk][0] = r0 < Sq ? *reinterpret_cast<const uint32_t*>(q + r0 * ld + (long) h * FA_D + d) : 0u;
        qa[kk][1] = r1 < Sq ? *reinterpret_cast<const uint32_t*>(q + r1 * ld + (long) h * FA_D + d) : 0u;
        qa[kk][2] = r0 < Sq ? *reinterpret_cast<const uint32_t*>(q + r0 * ld + (long) h * FA_D + d + 8) : 0u;
        qa[kk][3] = r1 < Sq ? *reinterpret_cast<const uint32_t*>(q + r1 * ld + (long) h * FA_D + d + 8) : 0u;
    }
    float o[16][4];
#pragma unroll
    for (int i = 0; i < 16; ++i) { o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f; }
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;

    const int32_t* list = lists + ((long) h * nq + qb) * nk;
    const int count = qb < nq ? counts[(long) h * nq + qb] : 0;
    for (int li = 0; li < 2 * count; ++li) {
        const int j = list[li >> 1];
        const __half* kb = j < split ? kc + (long) j * 128 * ld : kt + (long) (j - split) * 128 * ld;
        const __half* vb = j < split ? vc + (long) j * 128 * ld : vt + (long) (j - split) * 128 * ld;
        const int half = (li & 1) * FA_BK;
        __syncthreads();
        for (int i = threadIdx.x; i < FA_BK * FA_D / 8; i += 128) {
            const int row = i / (FA_D / 8), c = (i % (FA_D / 8)) * 8;
            *reinterpret_cast<uint4*>(&Ks[row * FA_LDK + c]) = *reinterpret_cast<const uint4*>(kb + (long) (half + row) * ld + (long) h * FA_D + c);
            *reinterpret_cast<uint4*>(&Vs[row * FA_LDK + c]) = *reinterpret_cast<const uint4*>(vb + (long) (half + row) * ld + (long) h * FA_D + c);
        }
        __syncthreads();
        float s[8][4];
#pragma unroll
        for (int nb = 0; nb < 8; ++nb) { s[nb][0] = s[nb][1] = s[nb][2] = s[nb][3] = 0.0f; }
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
#pragma unroll
            for (int nb = 0; nb < 8; ++nb) {
                const __half* kr = &Ks[(nb * 8 + g) * FA_LDK + kk * 16 + 2 * t];
                fa_mma16816(s[nb], qa[kk][0], qa[kk][1], qa[kk][2], qa[kk][3], *reinterpret_cast<const uint32_t*>(kr),
                            *reinterpret_cast<const uint32_t*>(kr + 8));
            }
        }
        float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
        for (int nb = 0; nb < 8; ++nb) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float x = s[nb][e] * scale_log2;
                s[nb][e] = x;
                if (e < 2) mx0 = fmaxf(mx0, x); else mx1 = fmaxf(mx1, x);
            }
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const float n0 = fmaxf(m0, mx0), n1 = fmaxf(m1, mx1);
        const float a0 = exp2f(m0 - n0), a1 = exp2f(m1 - n1);
        m0 = n0;
        m1 = n1;
        l0 *= a0;
        l1 *= a1;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            o[i][0] *= a0; o[i][1] *= a0;
            o[i][2] *= a1; o[i][3] *= a1;
        }
#pragma unroll
        for (int nb = 0; nb < 8; ++nb) {
            s[nb][0] = exp2f(s[nb][0] - n0);
            s[nb][1] = exp2f(s[nb][1] - n0);
            s[nb][2] = exp2f(s[nb][2] - n1);
            s[nb][3] = exp2f(s[nb][3] - n1);
            l0 += s[nb][0] + s[nb][1];
            l1 += s[nb][2] + s[nb][3];
        }
#pragma unroll
        for (int j2 = 0; j2 < 4; ++j2) {
            const uint32_t pa0 = fa_pack(s[2 * j2][0], s[2 * j2][1]), pa1 = fa_pack(s[2 * j2][2], s[2 * j2][3]);
            const uint32_t pa2 = fa_pack(s[2 * j2 + 1][0], s[2 * j2 + 1][1]), pa3 = fa_pack(s[2 * j2 + 1][2], s[2 * j2 + 1][3]);
#pragma unroll
            for (int nd = 0; nd < 16; ++nd) {
                const int d = nd * 8 + g;
                const __half* v0 = &Vs[(16 * j2 + 2 * t) * FA_LDK + d];
                __half2 hb0 = __halves2half2(v0[0], v0[FA_LDK]);
                __half2 hb1 = __halves2half2(v0[8 * FA_LDK], v0[9 * FA_LDK]);
                fa_mma16816(o[nd], pa0, pa1, pa2, pa3, *reinterpret_cast<uint32_t*>(&hb0), *reinterpret_cast<uint32_t*>(&hb1));
            }
        }
    }
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    const float i0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, i1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
#pragma unroll
    for (int nd = 0; nd < 16; ++nd) {
        const int d = nd * 8 + 2 * t;
        if (r0 < Sq) *reinterpret_cast<uint32_t*>(out + (long) r0 * ld + (long) h * FA_D + d) = fa_pack(o[nd][0] * i0, o[nd][1] * i0);
        if (r1 < Sq) *reinterpret_cast<uint32_t*>(out + (long) r1 * ld + (long) h * FA_D + d) = fa_pack(o[nd][2] * i1, o[nd][3] * i1);
    }
}

__global__ void unshuffle_k(const __half* x, int64_t T, int64_t H, int64_t W, int64_t ff, int64_t r, __half* out) {
    const int64_t Ho = H / r, Wo = W / r, Co = 3 * ff * r * r, n = (T / ff) * Ho * Wo * Co;
    const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    const int64_t ch = id % Co, p = id / Co;
    const int64_t xo = p % Wo, yo = (p / Wo) % Ho, to = p / (Wo * Ho);
    const int64_t xx = ch % r, yy = (ch / r) % r, tt = (ch / (r * r)) % ff, c = ch / (r * r * ff);
    out[id] = x[(((to * ff + tt) * H) + yo * r + yy) * W * 3 + (xo * r + xx) * 3 + c];
}

__global__ void im2col3_k(const __half* x, int64_t T, int64_t H, int64_t W, int64_t C, int64_t kt, int64_t st, int64_t p0, int64_t n, __half* col) {
    const int64_t K = kt * 9 * C;
    const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n * K) return;
    const int64_t k = id % K, p = p0 + id / K;
    const int64_t c = k % C, tap = k / C;
    const int64_t dx = tap % 3, dy = (tap / 3) % 3, dt = tap / 9;
    const int64_t xx = p % W, yy = (p / W) % H, to = p / (W * H);
    const int64_t ti = to * st + dt;
    const int64_t yi = min(max(yy + dy - 1, (int64_t) 0), H - 1), xi = min(max(xx + dx - 1, (int64_t) 0), W - 1);
    col[id] = x[((ti * H + yi) * W + xi) * C + c];
}

__global__ void im2col2_k(const __half* a, int64_t lda, int64_t Ca, const __half* mem, int64_t Cb, int64_t H, int64_t W, int up, int64_t p0, int64_t n,
                          __half* col) {
    const int64_t Ct = Ca + Cb, K = 9 * Ct;
    const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n * K) return;
    const int64_t k = id % K, p = p0 + id / K;
    const int64_t c = k % Ct, tap = k / Ct;
    const int64_t dx = tap % 3, dy = tap / 3;
    const int64_t xx = p % W, yy = (p / W) % H, fr = p / (W * H);
    const int64_t yi = yy + dy - 1, xi = xx + dx - 1;
    __half v = __float2half(0.0f);
    if (yi >= 0 && yi < H && xi >= 0 && xi < W) {
        const int64_t Hs = up ? H / 2 : H, Ws = up ? W / 2 : W;
        const int64_t ys = up ? yi / 2 : yi, xs = up ? xi / 2 : xi;
        const int64_t pix = ys * Ws + xs;
        if (c < Ca) {
            v = a[(fr * Hs * Ws + pix) * lda + c];
        } else if (fr > 0) {
            v = a[((fr - 1) * Hs * Ws + pix) * lda + (c - Ca)];
        } else if (mem) {
            v = mem[pix * lda + (c - Ca)];
        }
    }
    col[id] = v;
}

__global__ void rms_silu_k(const __half* x, int64_t C, const float* gamma, __half* out) {
    const __half* row = x + (size_t) blockIdx.x * C;
    float s = 0.0f;
    for (int64_t i = threadIdx.x; i < C; i += WG) { const float v = h2f(row[i]); s += v * v; }
    const float nrm = fmaxf(sqrtf(block_sum(s)), 1e-12f);
    const float k = sqrtf((float) C) / nrm;
    __half* o = out + (size_t) blockIdx.x * C;
    for (int64_t i = threadIdx.x; i < C; i += WG) {
        const float v = h2f(row[i]) * k * gamma[i];
        o[i] = f2h(v / (1.0f + expf(-v)));
    }
}

__global__ void relu_k(__half* x, int64_t n) {
    const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) x[id] = f2h(fmaxf(h2f(x[id]), 0.0f));
}

__global__ void add_relu_k(__half* x, const __half* y, int64_t n) {
    const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) x[id] = f2h(fmaxf(h2f(x[id]) + h2f(y[id]), 0.0f));
}

__global__ void tcd_input_k(const __half* cond, const float* lat, int64_t f, int64_t t, int64_t h8, int64_t w8, __half* out) {
    const int64_t n = h8 * w8 * 784;
    const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    const int64_t ch = id % 784, p = id / 784;
    const int64_t x = p % w8, y = p / w8;
    float v;
    if (ch < 768) {
        const int64_t xx = ch % 8, yy = (ch / 8) % 8, tt = (ch / 64) % 4, c = ch / 256;
        const int64_t H = h8 * 8, W = w8 * 8;
        v = h2f(cond[((tt * H + y * 8 + yy) * W + x * 8 + xx) * 3 + c]);
    } else {
        v = lat[(((ch - 768) * f + t) * h8 + y) * w8 + x];
    }
    out[id] = f2h(tanhf(v / 3.0f) * 3.0f);
}

__global__ void u8_to_half_k(const uint8_t* x, int64_t n, __half* out) {
    const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) out[id] = f2h((float) x[id] / 255.0f * 2.0f - 1.0f);
}

__global__ void to_half_k(const float* x, int64_t n, __half* out) {
    const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) out[id] = f2h(x[id]);
}

__global__ void to_float_k(const __half* x, int64_t n, float* out) {
    const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) out[id] = h2f(x[id]);
}

__global__ void stats_k(const __half* x, int64_t W, int64_t x0, int64_t y0, int64_t w, int64_t h, float a, float b, double* sums) {
    double s[6] = {0, 0, 0, 0, 0, 0};
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < w * h; i += (int64_t) gridDim.x * blockDim.x) {
        const int64_t xx = x0 + i % w, yy = y0 + i / w;
        for (int c = 0; c < 3; ++c) {
            const double v = (double) h2f(x[(yy * W + xx) * 3 + c]) * a + b;
            s[c] += v;
            s[3 + c] += v * v;
        }
    }
    for (int c = 0; c < 6; ++c) {
        double v = s[c];
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if ((threadIdx.x & 31) == 0) atomicAdd(&sums[c], v);
    }
}

__global__ void to_u8_k(const __half* dec, int64_t W, int64_t x0, int64_t y0, int64_t w, int64_t h, const float* ms, uint8_t* out) {
    const int64_t n = w * h * 3;
    const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    const int64_t c = id % 3, p = id / 3;
    const int64_t xx = x0 + p % w, yy = y0 + p / w;
    float v = h2f(dec[(yy * W + xx) * 3 + c]) * 2.0f - 1.0f;
    if (ms) v = (v - ms[c * 4]) / ms[c * 4 + 1] * ms[c * 4 + 3] + ms[c * 4 + 2];
    v = fminf(fmaxf(v, -1.0f), 1.0f);
    out[id] = (uint8_t) fminf(fmaxf((v + 1.0f) * 127.5f, 0.0f), 255.0f);
}

__global__ void adain_k(const double* s, double n, float* ms) {
    const int c = threadIdx.x;
    if (c >= 3) return;
    const double md = s[c] / n, ml = s[6 + c] / n;
    const double vd = fmax(s[3 + c] / n - md * md, 0.0), vl = fmax(s[9 + c] / n - ml * ml, 0.0);
    ms[c * 4] = (float) md;
    ms[c * 4 + 1] = (float) sqrt(vd + 1e-5);
    ms[c * 4 + 2] = (float) ml;
    ms[c * 4 + 3] = (float) sqrt(vl + 1e-5);
}

__global__ void conv3_small_k(const __half* x, int64_t H, int64_t W, int64_t ci, const float* w, int64_t co, const float* bias, __half* out) {
    const int64_t p = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= H * W) return;
    const int64_t xx = p % W, yy = p / W;
    float acc[4] = {0, 0, 0, 0};
    for (int dy = 0; dy < 3; ++dy) {
        const int64_t y = yy + dy - 1;
        if (y < 0 || y >= H) continue;
        for (int dx = 0; dx < 3; ++dx) {
            const int64_t x0 = xx + dx - 1;
            if (x0 < 0 || x0 >= W) continue;
            const __half2* row = reinterpret_cast<const __half2*>(x + (y * W + x0) * ci);
            const int tap = dy * 3 + dx;
            for (int64_t c = 0; c < ci / 2; ++c) {
                const float2 v = __half22float2(row[c]);
                for (int o = 0; o < 4; ++o) {
                    if (o >= co) break;
                    const float* wr = w + (o * 9 + tap) * ci + 2 * c;
                    acc[o] += v.x * wr[0] + v.y * wr[1];
                }
            }
        }
    }
    for (int o = 0; o < co; ++o) out[p * co + o] = f2h(acc[o] + (bias ? bias[o] : 0.0f));
}

__device__ inline uint64_t splitmix(uint64_t z) {
    z += 0x9e3779b97f4a7c15ull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

__global__ void randn_k(float* out, int64_t n, uint64_t seed, uint64_t offset) {
    const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    const uint64_t r = splitmix(seed * 0x2545f4914f6cdd1dull ^ splitmix((uint64_t) id + offset));
    const double u1 = ((double) (r >> 11) + 1.0) / 9007199254740993.0;
    const double u2 = (double) (splitmix(r) >> 11) / 9007199254740992.0;
    out[id] = (float) (sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2));
}

}  // namespace

#define FV_LAUNCH(k, n, ...) do { NS_CU(cudaSetDevice(g->dev)); if ((n) > 0) { k<<<nblk(n), WG, 0, g->q>>>(__VA_ARGS__); NS_LAUNCH_CHECK(); } return 0; } while (0)

extern "C" {

int ns_image_fv_patchify(ns_gpu* g, const float* lat, int64_t f, int64_t h8, int64_t w8, void* out) {
    if (f % 2 || h8 % 16 || w8 % 16) return ns_fail("ns_image_fv_patchify: frames in pairs, sides multiples of 16 latents");
    FV_LAUNCH(patchify_k, f * (h8 / 2) * (w8 / 2) * 64, lat, f, h8, w8, (__half*) out);
}

int ns_image_fv_unpatchify_sub(ns_gpu* g, const float* pred, const float* noise, int64_t f, int64_t h8, int64_t w8, float* out) {
    FV_LAUNCH(unpatchify_k, 16 * f * h8 * w8, pred, noise, f, h8, w8, out);
}

int ns_image_fv_add_lq(ns_gpu* g, float* x, const float* lq, int64_t f, int64_t hp, int64_t wp, int64_t C) {
    FV_LAUNCH(add_lq_k, f * hp * wp * C, x, lq, f, hp, wp, C);
}

int ns_image_fv_ln(ns_gpu* g, const float* x, int64_t M, int64_t C, float eps, const float* a, const float* b, void* out) {
    NS_CU(cudaSetDevice(g->dev));
    if (M > 0) { ln_k<<<(unsigned) M, WG, 0, g->q>>>(x, C, eps, a, b, (__half*) out); NS_LAUNCH_CHECK(); }
    return 0;
}

int ns_image_fv_gate_add(ns_gpu* g, float* x, const float* y, int64_t M, int64_t C, const float* gate) {
    FV_LAUNCH(gate_add_k, M * C, x, y, M * C, C, gate);
}

int ns_image_fv_block_mean(ns_gpu* g, const void* x, int64_t n, int64_t stride, int64_t C, float* out) {
    NS_CU(cudaSetDevice(g->dev));
    if (n > 0) { block_mean_k<<<(unsigned) n, WG, 0, g->q>>>((const __half*) x, stride, C, out); NS_LAUNCH_CHECK(); }
    return 0;
}

int ns_image_fv_block_probs(ns_gpu* g, const float* qm, const float* km, int64_t nq, int64_t nk, int64_t H, int64_t nbh, int64_t nbw, int64_t lr,
                            float* probs) {
    NS_CU(cudaSetDevice(g->dev));
    const size_t smem = (size_t) nk * sizeof(float);
    if (smem > 48 * 1024) return ns_fail("ns_image_fv_block_probs: more than 12,288 key blocks");
    dim3 grid((unsigned) nq, (unsigned) H);
    block_probs_k<<<grid, WG, smem, g->q>>>(qm, km, nq, nk, H, nbh, nbw, lr, probs);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_fv_topk_thr(ns_gpu* g, const float* p, int64_t S, int64_t n, int64_t rank, float* thr) {
    if (rank < 1 || rank > n) return ns_fail("ns_image_fv_topk_thr: the rank is outside the segment");
    NS_CU(cudaSetDevice(g->dev));
    topk_thr_k<<<(unsigned) S, 1024, 0, g->q>>>(p, n, rank, thr);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_fv_block_lists(ns_gpu* g, const float* probs, const float* thr, int64_t H, int64_t nq, int64_t nk, int64_t nb, int32_t* lists,
                            int32_t* counts) {
    NS_CU(cudaSetDevice(g->dev));
    const int64_t threads = H * nq * 32;
    block_lists_k<<<nblk(threads), WG, 0, g->q>>>(probs, thr, H, nq, nk, nb, lists, counts);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_fv_sparse_attn(ns_gpu* g, const void* q, int64_t Sq, int64_t H, const void* kc, const void* vc, const void* kt, const void* vt,
                            int64_t split, const int32_t* lists, const int32_t* counts, int64_t nk, void* out) {
    NS_CU(cudaSetDevice(g->dev));
    if (Sq % 128) return ns_fail("ns_image_fv_sparse_attn: query rows in whole blocks of 128");
    const size_t smem = (size_t) 2 * FA_BK * FA_LDK * sizeof(__half);
    static int set_on[64] = {0};
    if (g->dev < 64 && !set_on[g->dev]) {
        NS_CU(cudaFuncSetAttribute(sparse_k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem));
        set_on[g->dev] = 1;
    }
    dim3 grid((unsigned) (Sq / FA_BQ), (unsigned) H);
    const float scale_log2 = (1.0f / sqrtf((float) FA_D)) * 1.4426950408889634f;
    sparse_k<<<grid, 128, smem, g->q>>>((const __half*) q, (int) Sq, (int) H, (const __half*) kc, (const __half*) vc, (const __half*) kt,
                                        (const __half*) vt, (int) split, lists, counts, (int) nk, (int) (Sq / 128), (__half*) out, scale_log2);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_fv_unshuffle(ns_gpu* g, const void* x, int64_t T, int64_t H, int64_t W, int64_t ff, int64_t r, void* out) {
    if (T % ff || H % r || W % r) return ns_fail("ns_image_fv_unshuffle: sizes not multiples of the factors");
    FV_LAUNCH(unshuffle_k, T * H * W * 3, (const __half*) x, T, H, W, ff, r, (__half*) out);
}

int ns_image_fv_im2col3(ns_gpu* g, const void* x, int64_t T, int64_t H, int64_t W, int64_t C, int64_t kt, int64_t st, int64_t p0, int64_t n,
                        void* col) {
    FV_LAUNCH(im2col3_k, n * kt * 9 * C, (const __half*) x, T, H, W, C, kt, st, p0, n, (__half*) col);
}

int ns_image_fv_im2col2(ns_gpu* g, const void* a, int64_t lda, int64_t Ca, const void* mem, int64_t Cb, int64_t N, int64_t H, int64_t W, int up,
                        int64_t p0, int64_t n, void* col) {
    (void) N;
    FV_LAUNCH(im2col2_k, n * 9 * (Ca + Cb), (const __half*) a, lda, Ca, (const __half*) mem, Cb, H, W, up, p0, n, (__half*) col);
}

int ns_image_fv_rms_silu(ns_gpu* g, const void* x, int64_t M, int64_t C, const float* gamma, void* out) {
    NS_CU(cudaSetDevice(g->dev));
    if (M > 0) { rms_silu_k<<<(unsigned) M, WG, 0, g->q>>>((const __half*) x, C, gamma, (__half*) out); NS_LAUNCH_CHECK(); }
    return 0;
}

int ns_image_fv_relu(ns_gpu* g, void* x, int64_t n) { FV_LAUNCH(relu_k, n, (__half*) x, n); }

int ns_image_fv_add_relu(ns_gpu* g, void* x, const void* y, int64_t n) { FV_LAUNCH(add_relu_k, n, (__half*) x, (const __half*) y, n); }

int ns_image_fv_tcd_input(ns_gpu* g, const void* cond, const float* lat, int64_t f, int64_t t, int64_t h8, int64_t w8, void* out) {
    FV_LAUNCH(tcd_input_k, h8 * w8 * 784, (const __half*) cond, lat, f, t, h8, w8, (__half*) out);
}

int ns_image_fv_u8_to_half(ns_gpu* g, const uint8_t* x, int64_t n, void* out) { FV_LAUNCH(u8_to_half_k, n, x, n, (__half*) out); }

int ns_image_fv_to_half(ns_gpu* g, const float* x, int64_t n, void* out) { FV_LAUNCH(to_half_k, n, x, n, (__half*) out); }

int ns_image_fv_to_float(ns_gpu* g, const void* x, int64_t n, float* out) { FV_LAUNCH(to_float_k, n, (const __half*) x, n, out); }

int ns_image_fv_stats(ns_gpu* g, const void* x, int64_t W, int64_t x0, int64_t y0, int64_t w, int64_t h, float a, float b, double* sums) {
    NS_CU(cudaSetDevice(g->dev));
    stats_k<<<256, WG, 0, g->q>>>((const __half*) x, W, x0, y0, w, h, a, b, sums);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_fv_to_u8(ns_gpu* g, const void* dec, int64_t W, int64_t x0, int64_t y0, int64_t w, int64_t h, const float* ms, uint8_t* out) {
    FV_LAUNCH(to_u8_k, w * h * 3, (const __half*) dec, W, x0, y0, w, h, ms, out);
}

int ns_image_fv_adain(ns_gpu* g, const double* sums, double n, float* ms) {
    NS_CU(cudaSetDevice(g->dev));
    adain_k<<<1, 32, 0, g->q>>>(sums, n, ms);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_fv_conv3_small(ns_gpu* g, const void* x, int64_t H, int64_t W, int64_t ci, const float* w, int64_t co, const float* bias, void* out) {
    if (co > 4 || ci % 2) return ns_fail("ns_image_fv_conv3_small: at most 4 outputs, an even number of inputs");
    FV_LAUNCH(conv3_small_k, H * W, (const __half*) x, H, W, ci, w, co, bias, (__half*) out);
}

int ns_image_fv_randn(ns_gpu* g, float* out, int64_t n, uint64_t seed, uint64_t offset) { FV_LAUNCH(randn_k, n, out, n, seed, offset); }

}  // extern "C"
