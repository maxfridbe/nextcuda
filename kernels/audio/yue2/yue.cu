// yue.cu: the YuE2 engine's own kernels - what the shared diffusion kernels (nsd) do not have: the decode's product
// read from the Q8_0 blocks themselves (nsd_linear_q makes the whole matrix half first: right for the thousands of rows
// of a prefill or a flow pass, five times the traffic for the decode's one or two), the embedding rows from the Q8_0
// table, the attention's inputs (q / k norm, RoPE, the cache's rows) in one pass, the decode's attention against the
// half cache, and the flow stage's few element-wise steps.
//
// Precise arithmetic (kernels/build.sh builds audio/ without --use_fast_math): the rotary angles are upstream's float32
// ones, so the positions agree with the reference to the last bit of their inputs.
#include "ns.h"
#include "ns_internal.hpp"
#include "yue.h"

#include <cuda_fp16.h>

#include <cmath>

namespace {

constexpr int WARP = 32;

__device__ inline float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

unsigned blocks(int64_t n, int t = 256) { return (unsigned) ((n + t - 1) / t); }

// ---- the Q8_0 product: each warp R output rows, a lane a block of 32 at a time across K (the lanes' blocks side by
// side: a warp reads 32 x 34 contiguous bytes of a row a step)
constexpr int Q_NW = 8, Q_R = 2, Q_MM = 8;
constexpr int QB = 34;  // bytes of a Q8_0 block

__device__ inline void q8_block(const uint8_t* p, float* o) {
    // the block's bytes are 2-aligned (34 a block): the scale, then 16 pairs of int8
    const uint16_t* h = reinterpret_cast<const uint16_t*>(p);
    const float d = __half2float(__ushort_as_half(h[0]));
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const uint16_t u = h[1 + i];
        o[2 * i] = d * (float) (int8_t) (u & 0xff);
        o[2 * i + 1] = d * (float) (int8_t) (u >> 8);
    }
}

__global__ void gemv_q8_k(const float* x, int M, int64_t K, const uint8_t* w, int64_t N, float* out, int64_t ldo, bool acc) {
    const int lane = threadIdx.x & 31, wi = threadIdx.x >> 5;
    const int64_t n0 = ((int64_t) blockIdx.x * Q_NW + wi) * Q_R;
    const int64_t nb = K / 32, row_bytes = nb * QB;
    float a[Q_MM][Q_R];
#pragma unroll
    for (int m = 0; m < Q_MM; ++m)
#pragma unroll
        for (int r = 0; r < Q_R; ++r) a[m][r] = 0.0f;
    for (int64_t b = lane; b < nb; b += WARP) {
        float wf[Q_R][32];
#pragma unroll
        for (int r = 0; r < Q_R; ++r) {
            const int64_t n = n0 + r < N ? n0 + r : N - 1;
            q8_block(w + n * row_bytes + b * QB, wf[r]);
        }
#pragma unroll
        for (int m = 0; m < Q_MM; ++m) {
            if (m < M) {
                const float4* xp = reinterpret_cast<const float4*>(x + m * K + b * 32);
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float4 xv = xp[i];
#pragma unroll
                    for (int r = 0; r < Q_R; ++r)
                        a[m][r] += wf[r][4 * i] * xv.x + wf[r][4 * i + 1] * xv.y + wf[r][4 * i + 2] * xv.z + wf[r][4 * i + 3] * xv.w;
                }
            }
        }
    }
#pragma unroll
    for (int m = 0; m < Q_MM; ++m) {
        if (m < M) {
#pragma unroll
            for (int r = 0; r < Q_R; ++r) {
                const float s = warp_sum(a[m][r]);
                const int64_t n = n0 + r;
                if (lane == 0 && n < N) {
                    float* o = out + m * ldo + n;
                    *o = acc ? *o + s : s;
                }
            }
        }
    }
}

__global__ void embed_q8_k(const uint8_t* t, int64_t K, const int32_t* ids, int64_t n, float* out) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nb = K / 32;
    if (i >= n * nb) return;
    const int64_t r = i / nb, b = i % nb;
    float v[32];
    q8_block(t + (int64_t) ids[r] * nb * QB + b * QB, v);
#pragma unroll
    for (int j = 0; j < 32; ++j) out[r * K + b * 32 + j] = v[j];
}

// ---- q / k norm + RoPE + the cache's rows: one warp a (row, head) of the 16 q, 8 k and 8 v heads; lane l holds features
// l + 32 j (j < 4), the rotated pairs (i, 64 + i) are j and j + 2
constexpr int HQ = 16, HKV = 8, HD = 128;

__global__ void qkv_k(const float* qkv, int64_t T, const float* qn, const float* kn, float eps, float theta, int64_t pos0, __half* qo, __half* kc,
                      __half* vc, int64_t row0) {
    const int l = threadIdx.x & 31;
    const int64_t gw = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) / WARP;
    const int64_t t = gw / (HQ + 2 * HKV);
    const int hh = (int) (gw % (HQ + 2 * HKV));
    if (t >= T) return;
    const float* p = qkv + t * (HQ + 2 * HKV) * HD + hh * HD;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = p[l + 32 * j];
    if (hh >= HQ + HKV) {
        __half* o = vc + (row0 + t) * HKV * HD + (hh - HQ - HKV) * HD;
#pragma unroll
        for (int j = 0; j < 4; ++j) o[l + 32 * j] = __float2half_rn(v[j]);
        return;
    }
    const float* w = hh < HQ ? qn : kn;
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) ss += v[j] * v[j];
    const float inv = rsqrtf(warp_sum(ss) / (float) HD + eps);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = (v[j] * inv) * w[l + 32 * j];
    const float pos = (float) (pos0 + t);
    float o[4];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int i = l + 32 * j;
        // upstream: 1 / base ** (arange(0, 128, 2) / 128) in float32, angle = position * inv_freq
        const float inv_freq = 1.0f / powf(theta, (float) (2 * i) / (float) HD);
        const float a = pos * inv_freq, c = cosf(a), s = sinf(a);
        o[j] = v[j] * c - v[j + 2] * s;
        o[j + 2] = v[j + 2] * c + v[j] * s;
    }
    __half* dst = hh < HQ ? qo + t * HQ * HD + hh * HD : kc + (row0 + t) * HKV * HD + (hh - HQ) * HD;
#pragma unroll
    for (int j = 0; j < 4; ++j) dst[l + 32 * j] = __float2half_rn(o[j]);
}

// ---- one query row against the cache: a block per (key head, split of the keys), NW_A warps taking the split's keys in
// turn, a lane DL = 4 features of the G = 2 query heads that share the key head; the warps' running (max, sum, values)
// merged in shared memory, the splits by a second kernel (mm3.cu's scheme, the cache rows-major here)
constexpr int64_t KB = 64;  // keys a split: a thousand-key decode spreads over 128 blocks
constexpr int NW_A = 4, DL = 4, G = 2;

int64_t splits(int64_t n) { return n > KB ? (n + KB - 1) / KB : 1; }

__global__ void attn_k(const __half* q, const __half* kc, const __half* vc, int64_t n, int64_t ns, float* out, float* part) {
    extern __shared__ float slm[];
    const int lane = threadIdx.x & 31, sgi = threadIdx.x >> 5;
    const float scale = rsqrtf((float) HD);
    const int64_t sp = blockIdx.x % ns, hk = blockIdx.x / ns;
    const int64_t j0 = sp * (ns > 1 ? KB : n), j1 = min(n, ns > 1 ? (sp + 1) * KB : n);
    float qv[G][DL], acc[G][DL], m[G], l[G];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        const __half* qr = q + (hk * G + g) * HD + lane * DL;
#pragma unroll
        for (int i = 0; i < DL; ++i) {
            qv[g][i] = __half2float(qr[i]) * scale;
            acc[g][i] = 0.0f;
        }
        m[g] = -INFINITY;
        l[g] = 0.0f;
    }
    for (int64_t j = j0 + sgi; j < j1; j += NW_A) {
        const __half* kr = kc + (j * HKV + hk) * HD + lane * DL;
        const __half* vr = vc + (j * HKV + hk) * HD + lane * DL;
        float kv[DL], vv[DL];
#pragma unroll
        for (int i = 0; i < DL; ++i) {
            kv[i] = __half2float(kr[i]);
            vv[i] = __half2float(vr[i]);
        }
#pragma unroll
        for (int g = 0; g < G; ++g) {
            float d = 0.0f;
#pragma unroll
            for (int i = 0; i < DL; ++i) d += qv[g][i] * kv[i];
            const float sc = warp_sum(d);
            const float mn = fmaxf(m[g], sc);
            const float c = expf(m[g] - mn), e = expf(sc - mn);
#pragma unroll
            for (int i = 0; i < DL; ++i) acc[g][i] = acc[g][i] * c + e * vv[i];
            l[g] = l[g] * c + e;
            m[g] = mn;
        }
    }
#pragma unroll
    for (int g = 0; g < G; ++g) {
        float* o = &slm[(sgi * G + g) * (HD + 2)];
#pragma unroll
        for (int i = 0; i < DL; ++i) o[lane * DL + i] = acc[g][i];
        if (lane == 0) {
            o[HD] = m[g];
            o[HD + 1] = l[g];
        }
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < G * HD; idx += WARP * NW_A) {
        const int g = idx / HD, d = idx % HD;
        float mx = -INFINITY;
        for (int k = 0; k < NW_A; ++k) mx = fmaxf(mx, slm[(k * G + g) * (HD + 2) + HD]);
        float L = 0.0f, A = 0.0f;
        if (mx != -INFINITY) {
            for (int k = 0; k < NW_A; ++k) {
                const float* o = &slm[(k * G + g) * (HD + 2)];
                const float e = expf(o[HD] - mx);
                L += o[HD + 1] * e;
                A += o[d] * e;
            }
        }
        const int64_t hq = hk * G + g;
        if (ns == 1) {
            out[hq * HD + d] = L > 0.0f ? A / L : 0.0f;
        } else {
            float* p = part + (hq * ns + sp) * (HD + 2);
            p[d] = A;
            if (d == 0) {
                p[HD] = mx;
                p[HD + 1] = L;
            }
        }
    }
}

__global__ void attn_merge_k(const float* part, int64_t ns, float* out) {
    const int64_t h = blockIdx.x, d = threadIdx.x;
    const float* p = part + h * ns * (HD + 2);
    float mx = -INFINITY;
    for (int64_t k = 0; k < ns; ++k) mx = fmaxf(mx, p[k * (HD + 2) + HD]);
    float L = 0.0f, A = 0.0f;
    if (mx != -INFINITY) {
        for (int64_t k = 0; k < ns; ++k) {
            const float e = expf(p[k * (HD + 2) + HD] - mx);
            L += p[k * (HD + 2) + HD + 1] * e;
            A += p[k * (HD + 2) + d] * e;
        }
    }
    out[h * HD + d] = L > 0.0f ? A / L : 0.0f;
}

__global__ void add_vec_rows_k(float* x, int64_t C, const float* a, const float* rows, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] += a[i % C] + rows[i];
}

__global__ void axpy_k(float* out, const float* a, const float* b, int64_t n, float s) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + s * b[i];
}

}  // namespace

extern "C" {

int ns_audio_yue2_gemv_q8(ns_gpu* g, const float* x, int64_t M, int64_t K, const void* w, int64_t N, float* out, int64_t ldo, int acc) {
    if (M <= 0 || N <= 0) return 0;
    if (M > Q_MM || K % 32 != 0) return ns_fail("ns_audio_yue2_gemv_q8: at most 8 rows, K a multiple of 32");
    NS_CU(cudaSetDevice(g->dev));
    const unsigned grid = (unsigned) ((N + Q_NW * Q_R - 1) / (Q_NW * Q_R));
    gemv_q8_k<<<grid, WARP * Q_NW, 0, g->q>>>(x, (int) M, K, (const uint8_t*) w, N, out, ldo ? ldo : N, acc != 0);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_yue2_embed_q8(ns_gpu* g, const void* table, int64_t K, const int32_t* ids, int64_t n, float* out) {
    if (n <= 0) return 0;
    if (K % 32 != 0) return ns_fail("ns_audio_yue2_embed_q8: K a multiple of 32");
    NS_CU(cudaSetDevice(g->dev));
    embed_q8_k<<<blocks(n * (K / 32)), 256, 0, g->q>>>((const uint8_t*) table, K, ids, n, out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_yue2_qkv(ns_gpu* g, const float* qkv, int64_t T, const float* qn, const float* kn, float eps, float theta, int64_t pos0, void* q_out,
                      void* kc, void* vc, int64_t row0) {
    if (T <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    const int64_t threads = T * (HQ + 2 * HKV) * WARP;
    qkv_k<<<blocks(threads, 256), 256, 0, g->q>>>(qkv, T, qn, kn, eps, theta, pos0, (__half*) q_out, (__half*) kc, (__half*) vc, row0);
    NS_LAUNCH_CHECK();
    return 0;
}

int64_t ns_audio_yue2_attn_scratch(int64_t n) {
    const int64_t ns = splits(n);
    return ns > 1 ? HQ * ns * (HD + 2) : 0;
}

int ns_audio_yue2_attn(ns_gpu* g, const void* q, const void* kc, const void* vc, int64_t n, float* out, float* part) {
    if (n <= 0) return ns_fail("ns_audio_yue2_attn: no keys");
    NS_CU(cudaSetDevice(g->dev));
    const int64_t ns = splits(n);
    if (ns > 1 && !part) return ns_fail("ns_audio_yue2_attn: the splits' scratch is missing");
    const size_t shm = (size_t) NW_A * G * (HD + 2) * sizeof(float);
    attn_k<<<(unsigned) (HKV * ns), WARP * NW_A, shm, g->q>>>((const __half*) q, (const __half*) kc, (const __half*) vc, n, ns, out, part);
    NS_LAUNCH_CHECK();
    if (ns > 1) {
        attn_merge_k<<<(unsigned) HQ, HD, 0, g->q>>>(part, ns, out);
        NS_LAUNCH_CHECK();
    }
    return 0;
}

int ns_audio_yue2_add_vec_rows(ns_gpu* g, float* x, int64_t R, int64_t C, const float* a, const float* rows) {
    if (R * C <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    add_vec_rows_k<<<blocks(R * C), 256, 0, g->q>>>(x, C, a, rows, R * C);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_yue2_axpy(ns_gpu* g, float* out, const float* a, const float* b, int64_t n, float s) {
    if (n <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    axpy_k<<<blocks(n), 256, 0, g->q>>>(out, a, b, n, s);
    NS_LAUNCH_CHECK();
    return 0;
}

}  // extern "C"
