// ink.cu: the inkling2 engine's own kernels (ink.h) - Inkling-2-Mini's weights (Q8_0, BF16, NVFP4 decoded from the
// GGUF block layouts), its MoE (sigmoid router, SiTU), the depth attention and loop controllers, the short
// convolutions, the clipped attention with its relative bias and sink. nextsycl's kernels/llm/inkling2/ink.cpp written
// for CUDA (a warp of 32 where it had a sub-group of 16): the same names, arguments and float32 arithmetic in the same
// order. After llama.cpp's src/models/inkling2.cpp and inkling2-loop.cpp; the arithmetic is written from the formats
// and the graph, not taken from ggml. The plain, exact path: the silo (kernels/llm/inkling2/silo) is the fast one.
#include "ns.h"
#include "ns_internal.hpp"
#include "ink.h"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstring>
#include <mutex>
#include <string>

namespace {

constexpr int WS = 32;

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// a block's sum / max / min (every thread gets it); sh: 32 floats of shared memory, blockDim a multiple of 32
template <typename T, typename Op>
__device__ __forceinline__ T block_reduce(T v, T* sh, Op op) {
    const int lane = threadIdx.x % WS, w = threadIdx.x / WS, nw = blockDim.x / WS;
    for (int o = 16; o > 0; o >>= 1) v = op(v, __shfl_xor_sync(0xffffffffu, v, o));
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    T r = sh[0];
    for (int i = 1; i < nw; ++i) r = op(r, sh[i]);
    return r;
}
struct Add {
    template <typename T> __device__ T operator()(T a, T b) const { return a + b; }
};
struct Max {
    __device__ float operator()(float a, float b) const { return fmaxf(a, b); }
};
struct MinI {
    __device__ long long operator()(long long a, long long b) const { return a < b ? a : b; }
};

// ---- the block formats ----
// Q8_0: a half scale, then 32 int8 values. NVFP4: 64 values - four UE4M3 scales (one a 16), then 32 bytes of E2M1
// codes, sub-block s's value j in the low nibble of byte 8s + j and value j + 8 in its high nibble; a code is twice
// its E2M1 value (the table), the scale half its UE4M3 value. BF16: the top half of a float32.
constexpr int VB_Q8 = 32, BB_Q8 = 34, VB_FP4 = 64, BB_FP4 = 36;

__device__ __forceinline__ float ue4m3(uint8_t x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    const int e = (x >> 3) & 0xF, m = x & 0x7;
    // (1 + m / 8) 2^(e - 7) / 2 = (8 + m) 2^(e - 11); subnormal m 2^-9 / 2
    return e == 0 ? ldexpf((float) m, -10) : ldexpf((float) (8 + m), e - 11);
}

__device__ __forceinline__ float e2m1x2(int c) {
    // 0, 0.5, 1, 1.5, 2, 3, 4, 6 doubled, then the negatives
    const float m = (float) ((c & 7) < 4 ? (c & 7) : (c & 7) == 4 ? 4 : (c & 7) == 5 ? 6 : (c & 7) == 6 ? 8 : 12);
    return (c & 8) && m > 0.0f ? -m : m;   // code 8 (-0) as +0, as ggml's table has it
}

__device__ __forceinline__ float bf16f(uint16_t v) {
    return __uint_as_float((uint32_t) v << 16);
}

__device__ __forceinline__ float f16f(const uint8_t* p) {
    const uint16_t b = (uint16_t) (p[0] | (p[1] << 8));
    return __half2float(__ushort_as_half(b));
}

// values and bytes of a block of a type (BF16 / F32: 32 values a "block")
__host__ __device__ inline int vb(int type) {
    return type == 40 ? VB_FP4 : VB_Q8;
}
__host__ __device__ inline int bb(int type) {
    switch (type) {
        case 0: return 128;
        case 30: return 64;
        case 8: return BB_Q8;
        case 40: return BB_FP4;
        default: return 0;
    }
}

// block b of a row decoded into v (vb(T) values)
template <int T>
__device__ __forceinline__ void decode(const uint8_t* row, int64_t b, float* v) {
    if constexpr (T == 8) {
        const uint8_t* p = row + b * BB_Q8;
        const float d = f16f(p);
        const int8_t* q = (const int8_t*) (p + 2);
#pragma unroll
        for (int i = 0; i < 32; ++i) v[i] = d * (float) q[i];
    } else if constexpr (T == 40) {
        const uint8_t* p = row + b * BB_FP4;
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            const float d = ue4m3(p[s]);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint8_t c = p[4 + s * 8 + j];
                v[s * 16 + j] = e2m1x2(c & 15) * d;
                v[s * 16 + j + 8] = e2m1x2(c >> 4) * d;
            }
        }
    } else if constexpr (T == 30) {
        const uint16_t* p = (const uint16_t*) row + b * 32;
#pragma unroll
        for (int i = 0; i < 32; ++i) v[i] = bf16f(p[i]);
    } else {
        const float* p = (const float*) row + b * 32;
#pragma unroll
        for (int i = 0; i < 32; ++i) v[i] = p[i];
    }
}

constexpr int MAXT = 16;
constexpr int MV_W = 4;   // rows (warps) a block

// y rows: a warp a row; its lanes take the row's blocks in turn, each block against the T inputs
template <int TY>
__global__ void mv_rows_k(const uint8_t* w, size_t stride, size_t row_bytes, const float* scale, const int32_t* ids, const float* x, int per_pair,
                          float* y, int64_t T, int64_t K, int64_t n_in, int64_t n_out, bool moe) {
    constexpr int V = TY == 40 ? VB_FP4 : 32;
    const int64_t nb = n_in / V;
    const int lane = threadIdx.x % WS;
    const int64_t p = blockIdx.y;
    const int64_t row = (int64_t) blockIdx.x * MV_W + threadIdx.x / WS;
    if (row >= n_out) return;
    int64_t e = 0, nt = T, x0 = 0;
    if (moe) {
        e = ids[p];
        nt = 1;
        x0 = per_pair ? p : p / K;
    }
    const uint8_t* wr = w + (size_t) e * stride + (size_t) row * row_bytes;
    float acc[MAXT];
#pragma unroll
    for (int t = 0; t < MAXT; ++t) acc[t] = 0.0f;
    float v[V];
    for (int64_t b = lane; b < nb; b += WS) {
        decode<TY>(wr, b, v);
#pragma unroll
        for (int t = 0; t < MAXT; ++t) {
            if (t >= nt) break;
            const float* xr = x + (x0 + t) * n_in + b * V;
            float s = 0.0f;
#pragma unroll
            for (int i = 0; i < V; ++i) s += v[i] * xr[i];
            acc[t] += s;
        }
    }
#pragma unroll
    for (int t = 0; t < MAXT; ++t) {
        if (t >= nt) break;
        const float r = warp_sum(acc[t]);
        if (lane == 0) {
            if (moe) y[p * n_out + row] = scale ? r * scale[e] : r;
            else y[t * n_out + row] = r;
        }
    }
}

int mv_any(ns_gpu* g, int type, const void* w, size_t stride, const float* scale, const int32_t* ids, const float* x, int per_pair, float* y, int64_t T,
           int64_t K, int64_t n_in, int64_t n_out, bool moe, const char* who) {
    if (!moe && (T < 1 || T > MAXT)) return ns_fail(std::string(who) + ": 1..16 columns");
    if (bb(type) == 0 || n_in % vb(type)) return ns_fail(std::string(who) + ": a type it does not take, or rows of part blocks");
    if (T <= 0 || n_out <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    const size_t rb = (size_t) (n_in / vb(type)) * bb(type);
    auto* wp = (const uint8_t*) w;
    const dim3 grid((unsigned) ((n_out + MV_W - 1) / MV_W), (unsigned) (moe ? T * K : 1));
    switch (type) {
        case 0: mv_rows_k<0><<<grid, MV_W * WS, 0, g->q>>>(wp, stride, rb, scale, ids, x, per_pair, y, T, K, n_in, n_out, moe); break;
        case 8: mv_rows_k<8><<<grid, MV_W * WS, 0, g->q>>>(wp, stride, rb, scale, ids, x, per_pair, y, T, K, n_in, n_out, moe); break;
        case 30: mv_rows_k<30><<<grid, MV_W * WS, 0, g->q>>>(wp, stride, rb, scale, ids, x, per_pair, y, T, K, n_in, n_out, moe); break;
        case 40: mv_rows_k<40><<<grid, MV_W * WS, 0, g->q>>>(wp, stride, rb, scale, ids, x, per_pair, y, T, K, n_in, n_out, moe); break;
    }
    NS_LAUNCH_CHECK();
    return 0;
}

template <int TY, typename O>
__global__ void expand_k(const uint8_t* src, int64_t nblocks, O* dst) {
    constexpr int V = TY == 40 ? VB_FP4 : 32;
    const int64_t b = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    float v[V];
    decode<TY>(src, b, v);
#pragma unroll
    for (int i = 0; i < V; ++i) {
        if constexpr (sizeof(O) == 2) dst[b * V + i] = __half_as_ushort(__float2half_rn(v[i]));
        else dst[b * V + i] = v[i];
    }
}

template <typename O>
int expand_any(ns_gpu* g, int type, const void* src, int64_t n, O* dst, const char* who) {
    if (bb(type) == 0 || n % vb(type)) return ns_fail(std::string(who) + ": a type it does not take, or part blocks");
    if (n <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    auto* s = (const uint8_t*) src;
    const int64_t nbk = n / vb(type);
    const unsigned grid = (unsigned) ((nbk + 127) / 128);
    switch (type) {
        case 0: expand_k<0><<<grid, 128, 0, g->q>>>(s, nbk, dst); break;
        case 8: expand_k<8><<<grid, 128, 0, g->q>>>(s, nbk, dst); break;
        case 30: expand_k<30><<<grid, 128, 0, g->q>>>(s, nbk, dst); break;
        case 40: expand_k<40><<<grid, 128, 0, g->q>>>(s, nbk, dst); break;
    }
    NS_LAUNCH_CHECK();
    return 0;
}

// ---- the attention ----
constexpr int D_ATT = 128, DL = D_ATT / WS, KT = 32;   // a head's dims, a lane's, keys a tile
constexpr int64_t KB = 256;                             // decode rows' keys a split (absolute positions)

// rows [r0, r0 + RT) x the G query heads of key head hk, keys of split sp (KB keys at absolute [(s0 + sp) KB, ..); KB 0:
// all): a warp a (row, head), lanes over the dims; the keys a tile in shared memory
template <int RT, int G>
__global__ void attn_k(const float* qp, const __half* kc, const __half* vc, const float* r, const float* relb, int64_t E, int64_t R,
                       const float* sink, float* out, int64_t T, int64_t Hq, int64_t Hkv, int64_t cap, int64_t p0, int64_t win, float scale, int64_t kb,
                       int64_t s0, int64_t ns, float* part) {
    constexpr int WG = RT * G * WS;
    __shared__ __half ks[KT * D_ATT];
    __shared__ __half vs[KT * D_ATT];
    const int tid = threadIdx.x, lane = tid % WS, wi = tid / WS;
    int64_t gi = blockIdx.x;
    const int64_t sp = gi % ns;
    gi /= ns;
    const int64_t hk = gi % Hkv, tile = gi / Hkv;
    const int64_t rr = tile * RT + wi / G, hq = hk * G + wi % G;
    const bool live = rr < T;
    // the tile's keys: the union of its rows' [first, end), cut to the split
    const int64_t r_lo = tile * RT, r_hi = (T < r_lo + RT ? T : r_lo + RT) - 1;
    int64_t lo = win > 0 ? (p0 + r_lo + 1 - win > 0 ? p0 + r_lo + 1 - win : 0) : 0, hi = p0 + r_hi + 1;
    if (kb > 0) {
        lo = lo > (s0 + sp) * kb ? lo : (s0 + sp) * kb;
        hi = hi < (s0 + sp + 1) * kb ? hi : (s0 + sp + 1) * kb;
    }
    const int64_t pos = p0 + rr, first = win > 0 ? (pos + 1 - win > 0 ? pos + 1 - win : 0) : 0, end = pos + 1;
    float qv[DL];
    float rv = 0.0f;
    if (live) {
#pragma unroll
        for (int i = 0; i < DL; ++i) qv[i] = qp[(rr * Hq + hq) * D_ATT + lane * DL + i] * scale;
        if (lane < R) rv = r[(rr * Hq + hq) * R + lane];
    }
    // with no split the sink is the first logit (value 0); split rows merge it later
    float m = -INFINITY, s = 0.0f;
    if (live && kb == 0) {
        m = sink[rr * Hq + hq];
        s = 1.0f;
    }
    float acc[DL];
#pragma unroll
    for (int i = 0; i < DL; ++i) acc[i] = 0.0f;
    for (int64_t j0 = lo; j0 < hi; j0 += KT) {
        const int64_t nk = KT < hi - j0 ? KT : hi - j0;
        __syncthreads();
        for (int i = tid; i < nk * D_ATT; i += WG) {
            const int64_t j = j0 + i / D_ATT, slot = j % cap;
            ks[i] = kc[(hk * cap + slot) * D_ATT + i % D_ATT];
            vs[i] = vc[(hk * cap + slot) * D_ATT + i % D_ATT];
        }
        __syncthreads();
        if (!live) continue;
        for (int jj = 0; jj < nk; ++jj) {
            const int64_t j = j0 + jj;
            if (j < first || j >= end) continue;
            const int64_t d = pos - j;
            float part_l = 0.0f;
#pragma unroll
            for (int i = 0; i < DL; ++i) part_l += qv[i] * __half2float(ks[jj * D_ATT + lane * DL + i]);
            if (d < E && lane < R) part_l += rv * relb[d * R + lane];
            const float lg = warp_sum(part_l);
            const float mn = fmaxf(m, lg);
            const float c = expf(m - mn), p = expf(lg - mn);
            s = s * c + p;
#pragma unroll
            for (int i = 0; i < DL; ++i) acc[i] = acc[i] * c + p * __half2float(vs[jj * D_ATT + lane * DL + i]);
            m = mn;
        }
    }
    if (!live) return;
    if (kb == 0) {
#pragma unroll
        for (int i = 0; i < DL; ++i) out[(rr * Hq + hq) * D_ATT + lane * DL + i] = acc[i] / s;
    } else {
        float* pp = part + ((rr * Hq + hq) * ns + sp) * (D_ATT + 2);
#pragma unroll
        for (int i = 0; i < DL; ++i) pp[lane * DL + i] = acc[i];
        if (lane == 0) {
            pp[D_ATT] = m;
            pp[D_ATT + 1] = s;
        }
    }
}

// the splits of a decode row merged in order, with its sink
__global__ void attn_merge_k(const float* part, const float* sink, float* out, int64_t T, int64_t Hq, int64_t ns) {
    const int64_t rh = blockIdx.x, dd = threadIdx.x;
    const float* pp = part + rh * ns * (D_ATT + 2);
    float M = sink[rh];
    for (int64_t i = 0; i < ns; ++i) M = fmaxf(M, pp[i * (D_ATT + 2) + D_ATT]);
    float S = expf(sink[rh] - M), A = 0.0f;
    for (int64_t i = 0; i < ns; ++i) {
        const float c = expf(pp[i * (D_ATT + 2) + D_ATT] - M);
        S += pp[i * (D_ATT + 2) + D_ATT + 1] * c;
        A += pp[i * (D_ATT + 2) + dd] * c;
    }
    out[rh * D_ATT + dd] = A / S;
}

// decode rows' splits: [s0, s0 + ns) over every row's keys
void splits(int64_t T, int64_t p0, int64_t win, int64_t& s0, int64_t& ns) {
    const int64_t lo = win > 0 ? (p0 + 1 - win > 0 ? p0 + 1 - win : 0) : 0, hi = p0 + T;
    s0 = lo / KB;
    ns = (hi - 1) / KB - s0 + 1;
}

struct Ptrs {
    const float* p[8];
};

// ---- elementwise and per-row kernels ----
__global__ void route_k(const float* logits, const float* bias, int32_t* ids, float* w, int64_t E, int64_t K, float scale) {
    // a warp a row: lane l holds experts l, l + 32, ...; K rounds of the largest p + bias (the lowest index of equals)
    constexpr int PER = 256 / WS;
    const int lane = threadIdx.x;
    const int64_t t = blockIdx.x;
    const float* l = logits + t * E;
    float p[PER], v[PER];
    uint32_t used = 0;
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int e = lane + i * WS;
        p[i] = e < E ? 1.0f / (1.0f + expf(-l[e])) : 0.0f;
        v[i] = e < E ? (bias ? p[i] + bias[e] : p[i]) : -INFINITY;
    }
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        float bv = -INFINITY;
        int bi = 1 << 30;
#pragma unroll
        for (int i = 0; i < PER; ++i) {
            const int e = lane + i * WS;
            if (!(used >> i & 1) && e < E && (v[i] > bv || (v[i] == bv && e < bi))) {
                bv = v[i];
                bi = e;
            }
        }
        const float m = warp_max(bv);
        int cand = bv == m ? bi : (1 << 30);
        for (int o = 16; o > 0; o >>= 1) cand = min(cand, __shfl_xor_sync(0xffffffffu, cand, o));
        const int best = cand;
        float pb = 0.0f;
        if (best % WS == lane) {
            const int i = best / WS;
#pragma unroll
            for (int j = 0; j < PER; ++j)
                if (j == i) {
                    used |= 1u << j;
                    pb = p[j];
                }
        }
        pb = warp_sum(pb);
        if (lane == 0) {
            ids[t * K + k] = best;
            w[t * K + k] = pb;
        }
        sum += pb;
    }
    __syncwarp();
    sum += 1e-20f;
    if (lane < K) w[t * K + lane] = w[t * K + lane] / sum * scale;
}

__global__ void situ_k(const float* gu, const float* rs, float* act, uint16_t* act_h, int64_t n, int64_t F, float beta, float lb) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t r = i / F, c = i % F;
    const float sc = rs ? rs[r] : 1.0f;
    const float gt = gu[r * 2 * F + c] * sc, up = gu[r * 2 * F + F + c] * sc;
    const float a = beta * tanhf(gt / beta) * (1.0f / (1.0f + expf(-gt)));
    const float u = lb > 0.0f ? lb * tanhf(up / lb) : up;
    const float y = a * u;
    if (act) act[i] = y;
    if (act_h) act_h[i] = __half_as_ushort(__float2half_rn(y));
}

__global__ void combine_k(const float* e, const float* w, float* y, int64_t n, int64_t K, int64_t C) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t t = i / C, c = i % C;
    float s = 0.0f;
    for (int64_t k = 0; k < K; ++k) s += w[t * K + k] * e[(t * K + k) * C + c];
    y[t * C + c] = s;
}

__global__ void combine_rows_k(const float* rows, const int32_t* t_ptr, const int32_t* ent, const float* w, float* y, int64_t n, int64_t C) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t t = i / C, c = i % C;
    float s = 0.0f;
    for (int32_t j = t_ptr[t]; j < t_ptr[t + 1]; ++j) s += w[j] * rows[(int64_t) ent[j] * C + c];
    y[t * C + c] += s;
}

__global__ void gather_f16_k(const float* x, const int32_t* idx, uint16_t* out, int64_t n, int64_t C) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __half_as_ushort(__float2half_rn(x[(int64_t) idx[i / C] * C + i % C]));
}

constexpr int RW = 256;   // a row's threads (mix, the controller's input, the RMS norm)

__global__ void mix_k(Ptrs st, int64_t ns, const float* query, int64_t nq, float* out, int64_t T, int64_t C, float eps) {
    __shared__ float sh[32];
    const int64_t t = blockIdx.x;
    const int tid = threadIdx.x;
    float inv[8], pr[4][8];
    for (int s = 0; s < ns; ++s) {
        const float* x = st.p[s] + t * C;
        float ss = 0.0f;
        for (int64_t c = tid; c < C; c += RW) ss += x[c] * x[c];
        inv[s] = rsqrtf(block_reduce(ss, sh, Add()) / (float) C + eps);
    }
    const float sc = 1.0f / sqrtf((float) C);
    for (int qi = 0; qi < nq; ++qi) {
        float mx = -INFINITY;
        for (int s = 0; s < ns; ++s) {
            const float* x = st.p[s] + t * C;
            float d = 0.0f;
            for (int64_t c = tid; c < C; c += RW) d += x[c] * inv[s] * query[qi * C + c];
            pr[qi][s] = block_reduce(d, sh, Add()) * sc;
            mx = fmaxf(mx, pr[qi][s]);
        }
        float sum = 0.0f;
        for (int s = 0; s < ns; ++s) {
            pr[qi][s] = expf(pr[qi][s] - mx);
            sum += pr[qi][s];
        }
        for (int s = 0; s < ns; ++s) pr[qi][s] /= sum;
    }
    for (int64_t c = tid; c < C; c += RW) {
        for (int qi = 0; qi < nq; ++qi) {
            float y = 0.0f;
            for (int s = 0; s < ns; ++s) y += pr[qi][s] * (st.p[s][t * C + c] * inv[s]);
            out[(qi * T + t) * C + c] = y;
        }
    }
}

__global__ void ctrl_z_k(const float* cur, const float* msg, float* z, int64_t C, float eps) {
    __shared__ float sh[32];
    const int64_t t = blockIdx.x;
    const int tid = threadIdx.x;
    const float* a = cur + t * C;
    const float* b = msg + t * C;
    float sa = 0.0f, sb = 0.0f;
    for (int64_t c = tid; c < C; c += RW) {
        sa += a[c] * a[c];
        sb += b[c] * b[c];
    }
    const float ia = rsqrtf(block_reduce(sa, sh, Add()) / (float) C + eps);
    const float ib = rsqrtf(block_reduce(sb, sh, Add()) / (float) C + eps);
    for (int64_t c = tid; c < C; c += RW) {
        z[t * 2 * C + c] = a[c] * ia;
        z[t * 2 * C + C + c] = b[c] * ib;
    }
}

__global__ void ctrl_apply_k(const float* cur, const float* msg, const float* f, const float* gn, float* out, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = cur[i];
    out[i] = (x - tanhf(f[i]) * x) + tanhf(gn[i]) * msg[i];
}

__global__ void conv_rows_k(const float* x, const float* state, const float* w, const float* dyn, float* out, int64_t n, int64_t C, int64_t H, int64_t hc,
                            float inv) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t t = i / C, c = i % C;
    auto at = [&](int64_t j) { return j < 3 ? state[j * C + c] : x[(j - 3) * C + c]; };   // the extended rows
    float s = 0.0f;
    for (int k = 0; k < 4; ++k) s += at(t + k) * w[c * 4 + k];
    if (dyn) {
        const float* dt = dyn + (t * H + c / hc) * 4;
        for (int l = 0; l < 4; ++l) s += dt[l] * at(t + 3 - l);
    }
    out[t * C + c] = (x[t * C + c] + s) * inv;
}

__global__ void conv_state_k(const float* x, float* state, float* snap, int64_t T, int64_t C) {
    const int64_t c = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float s0 = state[c], s1 = state[C + c], s2 = state[2 * C + c];
    if (snap) {
        snap[c] = s0;
        snap[C + c] = s1;
        snap[2 * C + c] = s2;
        for (int64_t t = 0; t < T; ++t) snap[(3 + t) * C + c] = x[t * C + c];
    }
    for (int64_t t = T - 3 > 0 ? T - 3 : 0; t < T; ++t) {
        s0 = s1;
        s1 = s2;
        s2 = x[t * C + c];
    }
    state[c] = s0;
    state[C + c] = s1;
    state[2 * C + c] = s2;
}

__global__ void conv_keep_k(const float* snap, float* state, int64_t keep, int64_t n, int64_t C) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) state[i] = snap[keep * C + i];
}

struct Slots {
    const float* snap[64];
    float* state[64];
    int64_t g[4];
};

__global__ void conv_keep_slots_k(Slots sl, int64_t keep, int64_t total) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 3 * total) return;
    const int s = blockIdx.y;
    int64_t off = 0, j = i;
    for (int k = 0; k < 4; ++k) {
        const int64_t ch = sl.g[k];
        if (j < 3 * ch) {
            sl.state[s][3 * off + j] = sl.snap[s][(3 + MAXT) * off + keep * ch + j];
            return;
        }
        j -= 3 * ch;
        off += ch;
    }
}

__global__ void clip_k(float* x, const float* scale, int64_t n, int64_t H, int64_t D, float thr, float log_alpha, float log_floor, int64_t p0) {
    const int lane = threadIdx.x % WS;
    const int64_t rh = (int64_t) blockIdx.x * 4 + threadIdx.x / WS;
    if (rh >= n) return;
    const int64_t t = rh / H, h = rh % H;
    float* xr = x + rh * D;
    float ss = 0.0f;
    for (int64_t d = lane; d < D; d += WS) {
        const float v = scale ? xr[d] * scale[h * D + d] : xr[d];
        ss += v * v;
    }
    const float norm = sqrtf(warp_sum(ss)) + 1e-6f;
    const float f = fminf(fmaxf(thr / norm, 0.0f), 1.0f);
    float tau = 1.0f;
    if (log_alpha > 0.0f) tau = log_alpha * logf(fmaxf((float) (p0 + t + 1) * (1.0f / log_floor), 1.0f)) + 1.0f;
    for (int64_t d = lane; d < D; d += WS) {
        const float v = scale ? xr[d] * scale[h * D + d] : xr[d];
        xr[d] = log_alpha > 0.0f ? (v * f) * tau : v * f;
    }
}

__global__ void rms_rows_k(const float* x, float* y, int64_t rows, int64_t D, float eps) {
    const int lane = threadIdx.x % WS;
    const int64_t r = (int64_t) blockIdx.x * 4 + threadIdx.x / WS;
    if (r >= rows) return;
    float ss = 0.0f;
    for (int64_t d = lane; d < D; d += WS) ss += x[r * D + d] * x[r * D + d];
    const float inv = rsqrtf(warp_sum(ss) / (float) D + eps);
    for (int64_t d = lane; d < D; d += WS) y[r * D + d] = x[r * D + d] * inv;
}

__global__ void rms_norm_k(const float* x, const float* w, float* y, int64_t n, float eps) {
    __shared__ float sh[32];
    const int64_t r = blockIdx.x;
    const float* xr = x + r * n;
    float ss = 0.0f;
    for (int64_t c = threadIdx.x; c < n; c += RW) ss += xr[c] * xr[c];
    const float inv = rsqrtf(block_reduce(ss, sh, Add()) / (float) n + eps);
    for (int64_t c = threadIdx.x; c < n; c += RW) y[r * n + c] = xr[c] * inv * w[c];
}

__global__ void kv_store_k(const float* k, const float* v, int64_t n, int64_t Hkv, int64_t D, int64_t cap, int64_t p0, __half* kh, __half* vh) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t d = i % D, h = (i / D) % Hkv, t = i / (D * Hkv);
    const int64_t slot = (p0 + t) % cap;
    kh[(h * cap + slot) * D + d] = __float2half_rn(k[i]);
    vh[(h * cap + slot) * D + d] = __float2half_rn(v[i]);
}

// a block a (row, head): 4 warps, a block of 32 dims each
__global__ void kv_store_q8_k(const float* k, const float* v, int64_t Hkv, int64_t cap, int64_t p0, uint8_t* kc, uint8_t* vc) {
    const int64_t rh = blockIdx.x, t = rh / Hkv, h = rh % Hkv;
    const int d = threadIdx.x, lane = d % WS, w = d / WS;
    const int64_t slot = (p0 + t) % cap;
    for (int which = 0; which < 2; ++which) {
        const float x = (which ? v : k)[rh * 128 + d];
        uint8_t* row = (which ? vc : kc) + (h * cap + slot) * 136;
        const float amax = warp_max(fabsf(x));
        const float dd = amax / 127.0f;
        const float id = dd != 0.0f ? 1.0f / dd : 0.0f;
        row[d] = (uint8_t) (int8_t) roundf(x * id);
        if (lane == 0) ((__half*) (row + 128))[w] = __float2half_rn(dd);
    }
}

__global__ void gate_k(float* x, const float* gate, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= 1.0f / (1.0f + expf(-gate[i]));
}

__global__ void add_k(const float* a, const float* b, float* y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] + b[i];
}

__global__ void to_f16_k(const float* x, uint16_t* y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __half_as_ushort(__float2half_rn(x[i]));
}

__global__ void softcap_k(const float* x, float* out, int64_t total, int64_t n, int64_t ld, float cap) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int64_t r = i / n, c = i % n;
    const float v = x[r * ld + c];
    out[i] = cap > 0.0f ? cap * tanhf(v * (1.0f / cap)) : v;
}

constexpr int AW = 1024;   // a row's threads (argmax, the draw)

__global__ void argmax_k(const float* x, int32_t* out, int64_t n) {
    __shared__ float sh[32];
    __shared__ long long shi[32];
    const int64_t r = blockIdx.x;
    const int tid = threadIdx.x;
    const float* xr = x + r * n;
    float best = -INFINITY;
    long long at = n;
    for (int64_t i = tid; i < n; i += AW) {
        if (xr[i] > best) {
            best = xr[i];
            at = i;
        }
    }
    const float m = block_reduce(best, sh, Max());
    const long long first = block_reduce<long long>(best == m ? at : (long long) n, shi, MinI());
    if (tid == 0) out[r] = (int32_t) (first < n ? first : 0);
}

// a block's exclusive prefix sum (blockDim AW)
__device__ long long block_exclusive_scan(long long v, long long* sh) {
    const int lane = threadIdx.x % WS, w = threadIdx.x / WS;
    long long inc = v;
    for (int o = 1; o < WS; o <<= 1) {
        const long long y = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o) inc += y;
    }
    __syncthreads();
    if (lane == WS - 1) sh[w] = inc;
    __syncthreads();
    long long base = 0;
    for (int i = 0; i < w; ++i) base += sh[i];
    return base + inc - v;
}

__device__ __forceinline__ uint32_t key_of(float v) {
    const uint32_t b = __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// nextsycl's GPU draw (its museglimmer engine's, mg.cpp): the 256 largest, temperature, top-p, one uniform a row
__global__ void sample_k(const float* x, int32_t* out, int64_t n, float temp, float top_p, uint64_t seed, int64_t pos0) {
    constexpr int K = 256, CAP = 8192;
    __shared__ float cv[K];
    __shared__ int32_t ci[K];
    __shared__ int32_t cnt;
    __shared__ uint32_t cand[CAP];
    __shared__ float shf[32];
    __shared__ long long shl[32];
    const int64_t r = blockIdx.x;
    const int tid = threadIdx.x;
    const float* xr = x + r * n;
    // the K-th largest key: the largest c with K keys at or above it. The row's values within d of its largest, for the
    // smallest d of 1/8 .. 22.6 (factors of sqrt 2) or 64 that holds K of them, hold every key at or above it, so the
    // bit-by-bit search over them alone (in shared memory) finds the same c; past CAP of them, over the whole row
    float mx = -INFINITY;
    for (int64_t i = tid; i < n; i += AW) mx = fmaxf(mx, xr[i]);
    mx = block_reduce(mx, shf, Max());
    constexpr int ND = 17;
    auto win = [](int j) { return j == ND - 1 ? 64.0f : 0.125f * exp2f(0.5f * (float) j); };
    long long cd[ND];
#pragma unroll
    for (int j = 0; j < ND; ++j) cd[j] = 0;
    for (int64_t i = tid; i < n; i += AW) {
        const float v = xr[i];
#pragma unroll
        for (int j = 0; j < ND; ++j) cd[j] += v >= mx - win(j);
    }
    int dsel = -1;
    long long csel = 0;
#pragma unroll
    for (int j = 0; j < ND; ++j) {
        const long long c = block_reduce(cd[j], shl, Add());
        if (dsel < 0 && c >= K) {
            dsel = j;
            csel = c;
        }
    }
    uint32_t kth = 0;
    if (dsel >= 0 && csel <= CAP) {
        const float lo = mx - win(dsel);
        if (tid == 0) cnt = 0;
        __syncthreads();
        for (int64_t i = tid; i < n; i += AW) {
            if (xr[i] >= lo) cand[atomicAdd(&cnt, 1)] = key_of(xr[i]);
        }
        __syncthreads();
        const int nc = (int) csel;
        for (int bit = 31; bit >= 0; --bit) {
            const uint32_t c = kth | (1u << bit);
            long long m = 0;
            for (int i = tid; i < nc; i += AW) m += cand[i] >= c;
            if (block_reduce(m, shl, Add()) >= K) kth = c;
        }
    } else {
        for (int bit = 31; bit >= 0; --bit) {
            const uint32_t c = kth | (1u << bit);
            long long m = 0;
            for (int64_t i = tid; i < n; i += AW) m += key_of(xr[i]) >= c;
            if (block_reduce(m, shl, Add()) >= K) kth = c;
        }
    }
    __syncthreads();
    if (tid == 0) cnt = 0;
    __syncthreads();
    long long gt = 0;
    for (int64_t i = tid; i < n; i += AW) gt += key_of(xr[i]) > kth;
    const long long n_gt = block_reduce(gt, shl, Add());
    for (int64_t i = tid; i < n; i += AW) {
        if (key_of(xr[i]) > kth) {
            const int at = atomicAdd(&cnt, 1);
            cv[at] = xr[i];
            ci[at] = (int32_t) i;
        }
    }
    const int64_t chunk = (n + AW - 1) / AW, b0 = tid * chunk, b1 = n < b0 + chunk ? n : b0 + chunk;
    long long eq = 0;
    for (int64_t i = b0; i < b1; ++i) eq += key_of(xr[i]) == kth;
    long long before = block_exclusive_scan(eq, shl);
    for (int64_t i = b0; i < b1 && n_gt + before < K; ++i) {
        if (key_of(xr[i]) == kth) {
            cv[n_gt + before] = xr[i];
            ci[n_gt + before] = (int32_t) i;
            ++before;
        }
    }
    __syncthreads();
    for (int size = 2; size <= K; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            if (tid < K) {
                const int partner = tid ^ stride;
                if (partner > tid) {
                    const bool up = (tid & size) == 0;
                    const float va = cv[tid], vb2 = cv[partner];
                    const int32_t ia = ci[tid], ib = ci[partner];
                    const bool a_first = va > vb2 || (va == vb2 && ia < ib);
                    if (a_first != up) {
                        cv[tid] = vb2;
                        cv[partner] = va;
                        ci[tid] = ib;
                        ci[partner] = ia;
                    }
                }
            }
            __syncthreads();
        }
    }
    if (tid == 0) {
        uint64_t z = (uint64_t) seed + (uint64_t) (pos0 + r) * 0x9E3779B97F4A7C15ull;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z ^= z >> 31;
        float u = (float) (z >> 40) / (float) (1u << 24);
        const float m0 = cv[0];
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) sum += expf((cv[i] - m0) / temp);
        float acc = 0.0f;
        int keep = K;
        for (int i = 0; i < K; ++i) {
            acc += expf((cv[i] - m0) / temp) / sum;
            if (acc >= top_p) {
                keep = i + 1;
                break;
            }
        }
        float total = 0.0f;
        for (int i = 0; i < keep; ++i) total += expf((cv[i] - m0) / temp) / sum;
        int32_t pick = ci[keep - 1];
        for (int i = 0; i < keep; ++i) {
            u -= expf((cv[i] - m0) / temp) / sum / total;
            if (u <= 0.0f) {
                pick = ci[i];
                break;
            }
        }
        out[r] = pick;
    }
}

unsigned blocks(int64_t n, int t = 256) {
    return (unsigned) ((n + t - 1) / t);
}

// a cuBLAS handle a device, on the GPU's stream
cublasHandle_t blas(ns_gpu* g) {
    static std::mutex mu;
    static cublasHandle_t h[64] = {};
    std::lock_guard<std::mutex> lk(mu);
    if (g->dev < 0 || g->dev >= 64) return nullptr;
    if (!h[g->dev] && cublasCreate(&h[g->dev]) != CUBLAS_STATUS_SUCCESS) return nullptr;
    cublasSetStream(h[g->dev], g->q);
    return h[g->dev];
}

}  // namespace

extern "C" {

int ns_ink2_type_supported(int type, int64_t n_in) {
    return bb(type) > 0 && n_in % vb(type) == 0;
}

int ns_ink2_dequant(ns_gpu* g, int type, const void* src, int64_t n, float* dst) {
    NS_TRY
    return expand_any<float>(g, type, src, n, dst, "ns_ink2_dequant");
    NS_CATCH
}

int ns_ink2_dequant_f16(ns_gpu* g, int type, const void* src, int64_t n, uint16_t* dst) {
    NS_TRY
    return expand_any<uint16_t>(g, type, src, n, dst, "ns_ink2_dequant_f16");
    NS_CATCH
}

int ns_ink2_mv(ns_gpu* g, int type, const void* w, const float* x, float* y, int64_t n_in, int64_t n_out, int64_t T) {
    NS_TRY
    return mv_any(g, type, w, 0, nullptr, nullptr, x, 0, y, T, 1, n_in, n_out, false, "ns_ink2_mv");
    NS_CATCH
}

int ns_ink2_moe_mv(ns_gpu* g, int type, const void* w, size_t stride, const float* scale, const int32_t* ids, const float* x, int per_pair, float* y,
                   int64_t T, int64_t K, int64_t n_in, int64_t n_out) {
    NS_TRY
    return mv_any(g, type, w, stride, scale, ids, x, per_pair, y, T, K, n_in, n_out, true, "ns_ink2_moe_mv");
    NS_CATCH
}

int ns_ink2_route(ns_gpu* g, const float* logits, const float* bias, int32_t* ids, float* w, int64_t T, int64_t E, int64_t K, float scale) {
    NS_TRY
    if (E > 256 || K > 16) return ns_fail("ns_ink2_route: at most 256 experts, 16 chosen");
    if (T <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    route_k<<<(unsigned) T, WS, 0, g->q>>>(logits, bias, ids, w, E, K, scale);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_ink2_situ(ns_gpu* g, const float* gu, const float* rs, float* act, uint16_t* act_h, int64_t rows, int64_t F, float beta, float lb) {
    NS_TRY
    if (rows <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    situ_k<<<blocks(rows * F), 256, 0, g->q>>>(gu, rs, act, act_h, rows * F, F, beta, lb);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_ink2_combine(ns_gpu* g, const float* e, const float* w, float* y, int64_t T, int64_t K, int64_t C) {
    NS_TRY
    if (T <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    combine_k<<<blocks(T * C), 256, 0, g->q>>>(e, w, y, T * C, K, C);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_ink2_combine_rows(ns_gpu* g, const float* rows, const int32_t* t_ptr, const int32_t* ent, const float* w, float* y, int64_t T, int64_t C) {
    NS_TRY
    if (T <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    combine_rows_k<<<blocks(T * C), 256, 0, g->q>>>(rows, t_ptr, ent, w, y, T * C, C);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_ink2_gather_f16(ns_gpu* g, const float* x, const int32_t* idx, uint16_t* out, int64_t n, int64_t C) {
    NS_TRY
    if (n <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    gather_f16_k<<<blocks(n * C), 256, 0, g->q>>>(x, idx, out, n * C, C);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_ink2_mix(ns_gpu* g, const float* const* states, int64_t ns, const float* query, int64_t nq, float* out, int64_t T, int64_t C, float eps) {
    NS_TRY
    if (ns < 1 || ns > 8 || nq < 1 || nq > 4) return ns_fail("ns_ink2_mix: 1..8 states, 1..4 queries");
    if (T <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    Ptrs st{};
    for (int i = 0; i < ns; ++i) st.p[i] = states[i];
    mix_k<<<(unsigned) T, RW, 0, g->q>>>(st, ns, query, nq, out, T, C, eps);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_ink2_ctrl_z(ns_gpu* g, const float* cur, const float* msg, float* z, int64_t T, int64_t C, float eps) {
    NS_TRY
    if (T <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    ctrl_z_k<<<(unsigned) T, RW, 0, g->q>>>(cur, msg, z, C, eps);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_ink2_ctrl_apply(ns_gpu* g, const float* cur, const float* msg, const float* f, const float* gn, float* out, int64_t T, int64_t C) {
    NS_TRY
    if (T <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    ctrl_apply_k<<<blocks(T * C), 256, 0, g->q>>>(cur, msg, f, gn, out, T * C);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_ink2_conv(ns_gpu* g, const float* x, float* state, const float* w, const float* dyn, float* out, float* snap, int64_t T, int64_t C, int64_t H,
                 float inv) {
    NS_TRY
    if (T <= 0) return 0;
    if (dyn && (H <= 0 || C % H)) return ns_fail("ns_ink2_conv: heads that divide the channels");
    NS_CU(cudaSetDevice(g->dev));
    const int64_t hc = dyn ? C / H : 1;
    // the rows (the state's three before them), then the state (and the snapshot): a thread a channel
    conv_rows_k<<<blocks(T * C), 256, 0, g->q>>>(x, state, w, dyn, out, T * C, C, H, hc, inv);
    NS_LAUNCH_CHECK();
    conv_state_k<<<blocks(C), 256, 0, g->q>>>(x, state, snap, T, C);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_ink2_conv_keep(ns_gpu* g, const float* snap, float* state, int64_t keep, int64_t C) {
    NS_TRY
    NS_CU(cudaSetDevice(g->dev));
    conv_keep_k<<<blocks(3 * C), 256, 0, g->q>>>(snap, state, keep, 3 * C, C);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_ink2_conv_keep_slots(ns_gpu* g, int n, const float* const* snaps, float* const* states, int64_t keep, const int64_t* groups) {
    NS_TRY
    if (n < 1 || n > 64) return ns_fail("ns_ink2_conv_keep_slots: 1..64 slots");
    NS_CU(cudaSetDevice(g->dev));
    Slots sl{};
    for (int i = 0; i < n; ++i) {
        sl.snap[i] = snaps[i];
        sl.state[i] = states[i];
    }
    int64_t total = 0;
    for (int k = 0; k < 4; ++k) {
        sl.g[k] = groups[k];
        total += groups[k];
    }
    conv_keep_slots_k<<<dim3(blocks(3 * total), (unsigned) n), 256, 0, g->q>>>(sl, keep, total);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_ink2_clip(ns_gpu* g, float* x, const float* scale, int64_t T, int64_t H, int64_t D, float thr, float log_alpha, float log_floor, int64_t p0) {
    NS_TRY
    const int64_t n = T * H;
    if (n <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    clip_k<<<(unsigned) ((n + 3) / 4), 4 * WS, 0, g->q>>>(x, scale, n, H, D, thr, log_alpha, log_floor, p0);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_ink2_rms_rows(ns_gpu* g, const float* x, float* y, int64_t rows, int64_t D, float eps) {
    NS_TRY
    if (rows <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    rms_rows_k<<<(unsigned) ((rows + 3) / 4), 4 * WS, 0, g->q>>>(x, y, rows, D, eps);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_ink2_kv_store(ns_gpu* g, const float* k, const float* v, int64_t T, int64_t Hkv, int64_t D, int64_t cap, int64_t p0, void* kc, void* vc) {
    NS_TRY
    if (T <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    kv_store_k<<<blocks(T * Hkv * D), 256, 0, g->q>>>(k, v, T * Hkv * D, Hkv, D, cap, p0, (__half*) kc, (__half*) vc);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_ink2_kv_store_q8(ns_gpu* g, const float* k, const float* v, int64_t T, int64_t Hkv, int64_t D, int64_t cap, int64_t p0, void* kc, void* vc) {
    NS_TRY
    if (D != 128) return ns_fail("ns_ink2_kv_store_q8: heads of 128");
    if (T <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    kv_store_q8_k<<<(unsigned) (T * Hkv), 128, 0, g->q>>>(k, v, Hkv, cap, p0, (uint8_t*) kc, (uint8_t*) vc);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int64_t ns_ink2_attn_scratch(int64_t T, int64_t Hq, int64_t D, int64_t p0, int64_t win) {
    if (T > MAXT) return 0;
    int64_t s0, ns;
    splits(T, p0, win, s0, ns);
    return T * Hq * ns * (D + 2);
}

int ns_ink2_attn(ns_gpu* g, const float* q, const void* kc, const void* vc, const float* r, const float* relb, int64_t E, int64_t R, const float* sink,
                 float* out, int64_t T, int64_t Hq, int64_t Hkv, int64_t D, int64_t cap, int64_t p0, int64_t win, float scale, float* part) {
    NS_TRY
    if (D != D_ATT || Hq != 3 * Hkv || R > WS) return ns_fail("ns_ink2_attn: heads of 128, 3 query heads a key head, a bias of at most 32");
    if (T <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    auto* kh = (const __half*) kc;
    auto* vh = (const __half*) vc;
    if (T > MAXT) {
        const int64_t tiles = (T + 7) / 8;
        attn_k<8, 3><<<(unsigned) (tiles * Hkv), 8 * 3 * WS, 0, g->q>>>(q, kh, vh, r, relb, E, R, sink, out, T, Hq, Hkv, cap, p0, win, scale, 0, 0, 1, nullptr);
        NS_LAUNCH_CHECK();
        return 0;
    }
    if (!part) return ns_fail("ns_ink2_attn: decode rows need the scratch");
    int64_t s0, ns;
    splits(T, p0, win, s0, ns);
    attn_k<1, 3><<<(unsigned) (T * Hkv * ns), 3 * WS, 0, g->q>>>(q, kh, vh, r, relb, E, R, sink, out, T, Hq, Hkv, cap, p0, win, scale, KB, s0, ns, part);
    NS_LAUNCH_CHECK();
    attn_merge_k<<<(unsigned) (T * Hq), D_ATT, 0, g->q>>>(part, sink, out, T, Hq, ns);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_ink2_gate(ns_gpu* g, float* x, const float* gate, int64_t n) {
    NS_TRY
    if (n <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    gate_k<<<blocks(n), 256, 0, g->q>>>(x, gate, n);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_ink2_add(ns_gpu* g, const float* a, const float* b, float* y, int64_t n) {
    NS_TRY
    if (n <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    add_k<<<blocks(n), 256, 0, g->q>>>(a, b, y, n);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_ink2_softcap(ns_gpu* g, const float* x, float* out, int64_t rows, int64_t n, int64_t ld, float cap) {
    NS_TRY
    if (rows <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    softcap_k<<<blocks(rows * n), 256, 0, g->q>>>(x, out, rows * n, n, ld, cap);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_ink2_argmax(ns_gpu* g, const float* x, int32_t* out, int64_t T, int64_t n) {
    NS_TRY
    if (T <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    argmax_k<<<(unsigned) T, AW, 0, g->q>>>(x, out, n);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_ink2_sample(ns_gpu* g, const float* x, int32_t* out, int64_t T, int64_t n, float temp, float top_p, uint64_t seed, int64_t pos0) {
    NS_TRY
    if (T <= 0) return 0;
    if (n < 256) return ns_fail("ns_ink2_sample: fewer than 256 values a row");
    NS_CU(cudaSetDevice(g->dev));
    sample_k<<<(unsigned) T, AW, 0, g->q>>>(x, out, n, temp, top_p, seed, pos0);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_ink2_rms_norm(ns_gpu* g, const float* x, const float* w, float* y, int64_t rows, int64_t n, float eps) {
    NS_TRY
    if (rows <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    rms_norm_k<<<(unsigned) rows, RW, 0, g->q>>>(x, w, y, n, eps);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_ink2_to_f16(ns_gpu* g, const float* x, uint16_t* y, int64_t n) {
    NS_TRY
    if (n <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    to_f16_k<<<blocks(n), 256, 0, g->q>>>(x, y, n);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_ink2_gemm(ns_gpu* g, int64_t m, int64_t n, int64_t k, const float* x, int64_t ldx, const float* w, float* y, int64_t ldy, int accumulate) {
    NS_TRY
    if (m <= 0 || n <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    cublasHandle_t h = blas(g);
    if (!h) return ns_fail("ns_ink2_gemm: no cuBLAS handle");
    const float one = 1.0f, beta = accumulate ? 1.0f : 0.0f;
    // column-major: Y^T [n, m] = W^T (w is k x n column-major) . X^T [k, m]
    const cublasStatus_t st = cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, (int) n, (int) m, (int) k, &one, w, (int) k, x, (int) ldx, &beta, y, (int) ldy);
    if (st != CUBLAS_STATUS_SUCCESS) return ns_fail("ns_ink2_gemm: cublasSgemm failed (" + std::to_string((int) st) + ")");
    return 0;
    NS_CATCH
}

int ns_ink2_gemm_f16(ns_gpu* g, int64_t m, int64_t n, int64_t k, const uint16_t* x, int64_t ldx, const uint16_t* w, float* y, int64_t ldy, int accumulate) {
    NS_TRY
    if (m <= 0 || n <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    cublasHandle_t h = blas(g);
    if (!h) return ns_fail("ns_ink2_gemm_f16: no cuBLAS handle");
    const float one = 1.0f, beta = accumulate ? 1.0f : 0.0f;
    const cublasStatus_t st = cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, (int) n, (int) m, (int) k, &one, w, CUDA_R_16F, (int) k, x, CUDA_R_16F, (int) ldx, &beta,
                                           y, CUDA_R_32F, (int) ldy, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (st != CUBLAS_STATUS_SUCCESS) return ns_fail("ns_ink2_gemm_f16: cublasGemmEx failed (" + std::to_string((int) st) + ")");
    return 0;
    NS_CATCH
}

}  // extern "C"
