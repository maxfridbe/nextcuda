// dspine.cu: the qwen35 engine's kernels for the DSpine draft (q35.h: ns_q35_dsp_*; crates/qwen35/src/dspine.rs says
// what the draft is). After nextsycl's kernels/llm/qwen35/silo/dspine.cpp: the draft's products are the engine's own;
// what is new is the code bridge between its layers (a handful of small row-wise steps), the serial refinement of the
// last layer's rows (each row's 16 candidates scored against the previous row's choice), each row's 16 best logits, and
// an attention for the draft's heads of 128 (the engine's is for the target's 256) over its ring cache. All small: rows
// at most 8, a vector 512 to 5,120 wide.
#include "ns.h"
#include "ns_internal.hpp"
#include "q35.h"

#include <cuda_fp16.h>
#include <cfloat>
#include <climits>
#include <cmath>
#include <string>

namespace {

constexpr int WG = 256;
constexpr float F32_TINY = 1.1754943508222875e-38f;

__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// a block's sum (WG threads); every thread gets it
__device__ float block_sum(float v, float* red) {
    v = warp_sum(v);
    const int w = threadIdx.x / 32, l = threadIdx.x % 32;
    __syncthreads();
    if (l == 0) red[w] = v;
    __syncthreads();
    float s = 0.f;
    for (int i = 0; i < (int) (blockDim.x / 32); ++i) s += red[i];
    return s;
}

__global__ void bridge_norm_k(const float* x, float* y, float* rms, int64_t C, float eps) {
    __shared__ float red[32];
    const int64_t r = blockIdx.x;
    const float* xr = x + r * C;
    float s = 0.f;
    for (int64_t i = threadIdx.x; i < C; i += blockDim.x) s += xr[i] * xr[i];
    s = block_sum(s, red);
    const float rm = sqrtf(fmaxf(s / (float) C, F32_TINY));
    if (threadIdx.x == 0) rms[r] = rm;
    const float inv = 1.0f / (rm + eps);
    for (int64_t i = threadIdx.x; i < C; i += blockDim.x) y[r * C + i] = xr[i] * inv;
}

__global__ void codes_k(const float* proj, const float* anchor_code, float* codes, int64_t C, float sqrt_c, float eps) {
    __shared__ float red[32];
    const int64_t r = blockIdx.x;
    if (r == 0) {
        for (int64_t i = threadIdx.x; i < C; i += blockDim.x) codes[i] = anchor_code[i];
        return;
    }
    const float* pr = proj + r * C;
    float s = 0.f;
    for (int64_t i = threadIdx.x; i < C; i += blockDim.x) s += pr[i] * pr[i];
    s = block_sum(s, red);
    const float sc = sqrt_c / (sqrtf(s) + eps);
    for (int64_t i = threadIdx.x; i < C; i += blockDim.x) codes[r * C + i] = pr[i] * sc;
}

__global__ void message_k(const float* packed, const float* gate_bias, float* msg, int64_t rows, int64_t R) {
    const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= rows * R) return;
    const int64_t i = id / R, r = id % R;
    if (i == 0) {
        msg[id] = 0.f;
        return;
    }
    const float* cur = packed + i * 3 * R;
    const float* prev = packed + (i - 1) * 3 * R;
    msg[id] = sigmoidf_(cur[r] + prev[R + r] + gate_bias[r]) * prev[2 * R + r];
}

__global__ void write_k(const float* raw, const float* delta, const float* rms, float* out, int64_t rows, int64_t H) {
    const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= rows * H) return;
    const int64_t i = id / H;
    out[id] = i == 0 ? raw[id] : raw[id] + rms[i] * delta[id];
}

__global__ void base_k(const float* raw, const float* norm_w, const __half* rows, float* base, int64_t K, int64_t H) {
    __shared__ float red[32];
    const int64_t nk = blockIdx.x, r = nk / K;
    float s = 0.f;
    for (int64_t h = threadIdx.x; h < H; h += blockDim.x) s += raw[r * H + h] * norm_w[h] * __half2float(rows[nk * H + h]);
    s = block_sum(s, red);
    if (threadIdx.x == 0) base[nk] = s;
}

// (row r, sender k): m = sigmoid(q[r] + tsv[r][k][:R] + bias) * tsv[r][k][R:], then the best k2 of
// base[r][k2] + rms[r] (m . tg[r][k2])
constexpr int PWG = 128;
__global__ void pick_k(const float* base, const float* rms, const float* q, int64_t qs, const float* tsv, const float* tg, const float* gate_bias,
                       int32_t* idx, int64_t K, int64_t R) {
    __shared__ float m[512];
    __shared__ float red[32];
    const int64_t nk = blockIdx.x, r = nk / K;
    const float* ts = tsv + nk * 2 * R;
    for (int64_t i = threadIdx.x; i < R; i += blockDim.x) m[i] = sigmoidf_(q[r * qs + i] + ts[i] + gate_bias[i]) * ts[R + i];
    __syncthreads();
    float best = 0.f;
    int bi = 0;
    for (int k2 = 0; k2 < K; ++k2) {
        const float* t2 = tg + (r * K + k2) * R;
        float s = 0.f;
        for (int64_t i = threadIdx.x; i < R; i += blockDim.x) s += m[i] * t2[i];
        s = block_sum(s, red);
        const float score = base[r * K + k2] + rms[r] * s;
        if (k2 == 0 || score > best) {
            best = score;
            bi = k2;
        }
    }
    if (threadIdx.x == 0) idx[nk] = bi;
}

// ---- the K best of each row (descending; the lower index first on a tie): a part of the row a block (its K best by K
// rounds of a block arg-max over the part in shared memory), then the parts' K x P a row the same way
constexpr int TK_PARTS = 128, TK_MAX_K = 32, TK_PART_MAX = 2048;

__device__ __forceinline__ bool better(float v, int i, float bv, int bi) { return v > bv || (v == bv && i < bi); }

// the block's best (value, index) of vals[0..n) (their indices idxs, or the position itself when null); every thread gets it
__device__ void block_best(const float* vals, const int* idxs, int n, float* rv, int* ri, int* rp, float& bv, int& bi, int& bp) {
    float v = -FLT_MAX;
    int i = INT_MAX, p = -1;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        const int ij = idxs ? idxs[j] : j;
        if (p < 0 || better(vals[j], ij, v, i)) {
            v = vals[j];
            i = ij;
            p = j;
        }
    }
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, v, o);
        const int oi = __shfl_xor_sync(0xffffffffu, i, o), op = __shfl_xor_sync(0xffffffffu, p, o);
        if (op >= 0 && (p < 0 || better(ov, oi, v, i))) {
            v = ov;
            i = oi;
            p = op;
        }
    }
    const int w = threadIdx.x / 32, l = threadIdx.x % 32;
    __syncthreads();
    if (l == 0) {
        rv[w] = v;
        ri[w] = i;
        rp[w] = p;
    }
    __syncthreads();
    bv = -FLT_MAX;
    bi = INT_MAX;
    bp = -1;
    for (int k = 0; k < (int) (blockDim.x / 32); ++k) {
        if (rp[k] >= 0 && (bp < 0 || better(rv[k], ri[k], bv, bi))) {
            bv = rv[k];
            bi = ri[k];
            bp = rp[k];
        }
    }
}

__global__ void topk_parts_k(const float* x, int64_t n, int K, int per, float* pv, int* pi) {
    __shared__ float v[TK_PART_MAX];
    __shared__ float rv[32];
    __shared__ int ri[32], rp[32];
    const int64_t row = blockIdx.y, part = blockIdx.x;
    const int64_t a = part * per;
    const int cnt = (int) max((int64_t) 0, min((int64_t) per, n - a));
    for (int j = threadIdx.x; j < cnt; j += blockDim.x) v[j] = x[row * n + a + j];
    __syncthreads();
    for (int k = 0; k < K; ++k) {
        float bv;
        int bi, bp;
        block_best(v, nullptr, cnt, rv, ri, rp, bv, bi, bp);
        if (threadIdx.x == 0) {
            const int64_t o = (row * TK_PARTS + part) * K + k;
            pv[o] = bp >= 0 ? bv : -FLT_MAX;
            pi[o] = bp >= 0 ? (int) (a + bi) : INT_MAX;
            if (bp >= 0) v[bp] = -FLT_MAX;
        }
        __syncthreads();
        // (a taken value is -FLT_MAX: it ties the part's empty slots, which a later round may then take - their
        // index is past every real one only when the part is all taken; K is at most the part's length)
    }
}

__global__ void topk_merge_k(const float* pv, const int* pi, int K, int32_t* out) {
    __shared__ float v[TK_PARTS * TK_MAX_K];
    __shared__ int ix[TK_PARTS * TK_MAX_K];
    __shared__ float rv[32];
    __shared__ int ri[32], rp[32];
    const int64_t row = blockIdx.x;
    const int n = TK_PARTS * K;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        v[j] = pv[row * n + j];
        ix[j] = pi[row * n + j];
    }
    __syncthreads();
    for (int k = 0; k < K; ++k) {
        float bv;
        int bi, bp;
        block_best(v, ix, n, rv, ri, rp, bv, bi, bp);
        if (threadIdx.x == 0) {
            out[row * K + k] = bi;
            v[bp] = -FLT_MAX;
            ix[bp] = INT_MAX;
        }
        __syncthreads();
    }
}

// ---- attention for heads of 128 (the draft's): rows q [T, Hq x 128] (qs floats apart) at the block's positions,
// against keys [0, pa + T) of a half cache [Hkv][cap][128] (from the window's first key: the caller's pointers), causal
// (row t sees keys [0, pa + t]); scale 1 / sqrt(128). A block a (key head, run of DA_KEYS keys): the key head's G x T
// queries scored against the run's keys (a lane a query), then the run's softmax share and values; the runs merged after.
constexpr int DA_KEYS = 64, DA_D = 128, DA_MAXQ = 32;

__global__ void __launch_bounds__(128) dsp_attn_k(const float* q, int64_t qs, const __half* kc, const __half* vc, int T, int Hq, int Hkv, int64_t cap,
                                                    int64_t pa, float* part) {
    __shared__ float qsh[DA_MAXQ][DA_D + 1];
    __shared__ __half ksh[DA_KEYS][DA_D];
    __shared__ float s[DA_MAXQ][DA_KEYS + 1];
    __shared__ float mrow[DA_MAXQ], lrow[DA_MAXQ];
    const int h = blockIdx.x, run = blockIdx.y, runs = gridDim.y;
    const int G = Hq / Hkv, NQ = G * T;
    const int64_t nk = pa + T, k0 = (int64_t) run * DA_KEYS;
    const int kn = (int) min((int64_t) DA_KEYS, nk - k0);
    const float scale = rsqrtf((float) DA_D);
    // the queries (query qi: row qi / G, head h G + qi % G)
    for (int i = threadIdx.x; i < NQ * DA_D; i += blockDim.x) {
        const int qi = i / DA_D, d = i % DA_D;
        qsh[qi][d] = q[(int64_t) (qi / G) * qs + (int64_t) (h * G + qi % G) * DA_D + d] * scale;
    }
    // the run's keys
    const __half* kh = kc + ((int64_t) h * cap + k0) * DA_D;
    for (int i = threadIdx.x; i < kn * DA_D / 2; i += blockDim.x) reinterpret_cast<__half2*>(&ksh[0][0])[i] = reinterpret_cast<const __half2*>(kh)[i];
    __syncthreads();
    // scores: warp w takes keys w, w + 4, ...; lane = query
    const int w = threadIdx.x / 32, l = threadIdx.x % 32;
    for (int j = w; j < kn; j += 4) {
        if (l < NQ) {
            float a = 0.f;
            #pragma unroll 8
            for (int d = 0; d < DA_D; d += 2) {
                const float2 kk = __half22float2(*reinterpret_cast<const __half2*>(&ksh[j][d]));
                a += qsh[l][d] * kk.x + qsh[l][d + 1] * kk.y;
            }
            const int t = l / G;
            s[l][j] = (k0 + j <= pa + t) ? a : -FLT_MAX;
        }
    }
    __syncthreads();
    // each query's max and sum over the run (a warp a query)
    for (int qi = w; qi < NQ; qi += 4) {
        float m = -FLT_MAX;
        for (int j = l; j < kn; j += 32) m = fmaxf(m, s[qi][j]);
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        float sum = 0.f;
        for (int j = l; j < kn; j += 32) {
            const float p = s[qi][j] == -FLT_MAX ? 0.f : expf(s[qi][j] - m);
            s[qi][j] = p;
            sum += p;
        }
        sum = warp_sum(sum);
        if (l == 0) {
            mrow[qi] = m;
            lrow[qi] = sum;
        }
    }
    __syncthreads();
    // the values: thread d accumulates every query's feature d over the run
    const int d = threadIdx.x;
    float acc[DA_MAXQ];
    #pragma unroll
    for (int qi = 0; qi < DA_MAXQ; ++qi) acc[qi] = 0.f;
    const __half* vh = vc + ((int64_t) h * cap + k0) * DA_D;
    for (int j = 0; j < kn; ++j) {
        const float vv = __half2float(vh[(int64_t) j * DA_D + d]);
        #pragma unroll
        for (int qi = 0; qi < DA_MAXQ; ++qi)
            if (qi < NQ) acc[qi] += s[qi][j] * vv;
    }
    // part [Hkv][runs][NQ][D + 2]: the run's (max, sum, values)
    float* pp = part + (((int64_t) h * runs + run) * NQ) * (DA_D + 2);
    for (int qi = 0; qi < NQ; ++qi) {
        float* o = pp + (int64_t) qi * (DA_D + 2);
        o[2 + d] = acc[qi];
        if (d == 0) {
            o[0] = mrow[qi];
            o[1] = lrow[qi];
        }
    }
}

__global__ void dsp_attn_merge_k(const float* part, int T, int Hq, int Hkv, int runs, float* out) {
    // a block a query (row t, head hq), thread d a feature
    const int hq = blockIdx.x % Hq, t = blockIdx.x / Hq, d = threadIdx.x;
    const int G = Hq / Hkv, h = hq / G, qi = t * G + hq % G, NQ = G * T;
    float m = -FLT_MAX;
    for (int r = 0; r < runs; ++r) m = fmaxf(m, part[(((int64_t) h * runs + r) * NQ + qi) * (DA_D + 2)]);
    float l = 0.f, a = 0.f;
    for (int r = 0; r < runs; ++r) {
        const float* o = part + (((int64_t) h * runs + r) * NQ + qi) * (DA_D + 2);
        if (o[1] <= 0.f) continue;
        const float w = expf(o[0] - m);
        l += w * o[1];
        a += w * o[2 + d];
    }
    out[(int64_t) t * Hq * DA_D + (int64_t) hq * DA_D + d] = l > 0.f ? a / l : 0.f;
}

}  // namespace

extern "C" {

int ns_q35_dsp_bridge_norm(ns_gpu* g, const float* x, float* y, float* rms, int64_t rows, int64_t C, float eps) {
    NS_TRY
    NS_CU(cudaSetDevice(g->dev));
    if (rows <= 0) return 0;
    bridge_norm_k<<<(unsigned) rows, WG, 0, g->q>>>(x, y, rms, C, eps);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_q35_dsp_codes(ns_gpu* g, const float* proj, const float* anchor_code, float* codes, int64_t rows, int64_t C, float sqrt_c, float eps) {
    NS_TRY
    NS_CU(cudaSetDevice(g->dev));
    if (rows <= 0) return 0;
    codes_k<<<(unsigned) rows, WG, 0, g->q>>>(proj, anchor_code, codes, C, sqrt_c, eps);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_q35_dsp_message(ns_gpu* g, const float* packed, const float* gate_bias, float* msg, int64_t rows, int64_t R) {
    NS_TRY
    NS_CU(cudaSetDevice(g->dev));
    if (rows <= 0) return 0;
    message_k<<<(unsigned) ((rows * R + WG - 1) / WG), WG, 0, g->q>>>(packed, gate_bias, msg, rows, R);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_q35_dsp_write(ns_gpu* g, const float* raw, const float* delta, const float* rms, float* out, int64_t rows, int64_t H) {
    NS_TRY
    NS_CU(cudaSetDevice(g->dev));
    if (rows <= 0) return 0;
    write_k<<<(unsigned) ((rows * H + WG - 1) / WG), WG, 0, g->q>>>(raw, delta, rms, out, rows, H);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_q35_dsp_base(ns_gpu* g, const float* raw, const float* norm_w, const void* rows_f16, float* base, int64_t n, int64_t K, int64_t H) {
    NS_TRY
    NS_CU(cudaSetDevice(g->dev));
    if (n <= 0) return 0;
    base_k<<<(unsigned) (n * K), WG, 0, g->q>>>(raw, norm_w, static_cast<const __half*>(rows_f16), base, K, H);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_q35_dsp_pick(ns_gpu* g, const float* base, const float* rms, const float* q, int64_t qs, const float* tsv, const float* tg, const float* gate_bias,
                    int32_t* idx, int64_t n, int64_t K, int64_t R) {
    NS_TRY
    NS_CU(cudaSetDevice(g->dev));
    if (n <= 0) return 0;
    if (R > 512 || K > TK_MAX_K) return ns_fail("ns_q35_dsp_pick: rank at most 512, at most 32 candidates");
    pick_k<<<(unsigned) (n * K), PWG, 0, g->q>>>(base, rms, q, qs, tsv, tg, gate_bias, idx, K, R);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int64_t ns_q35_dsp_topk_scratch(int64_t rows, int64_t K) { return rows * TK_PARTS * K * 2; }

int ns_q35_dsp_topk(ns_gpu* g, const float* x, int64_t rows, int64_t n, int64_t K, int32_t* out, float* scratch) {
    NS_TRY
    NS_CU(cudaSetDevice(g->dev));
    if (rows <= 0) return 0;
    const int64_t per = (n + TK_PARTS - 1) / TK_PARTS;
    if (K < 1 || K > TK_MAX_K || per > TK_PART_MAX || K > per) return ns_fail("ns_q35_dsp_topk: 1..32 of a row of at most 262,144 (and K at most a part)");
    float* pv = scratch;
    int* pi = reinterpret_cast<int*>(scratch + rows * TK_PARTS * K);
    topk_parts_k<<<dim3(TK_PARTS, (unsigned) rows), WG, 0, g->q>>>(x, n, (int) K, (int) per, pv, pi);
    NS_LAUNCH_CHECK();
    topk_merge_k<<<(unsigned) rows, WG, 0, g->q>>>(pv, pi, (int) K, out);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int64_t ns_q35_dsp_attn_scratch(int64_t T, int64_t Hq, int64_t Hkv, int64_t pa) {
    const int64_t runs = (pa + T + DA_KEYS - 1) / DA_KEYS;
    return Hkv * runs * (Hq / Hkv) * T * (DA_D + 2);
}

int ns_q35_dsp_attn(ns_gpu* g, const float* q, int64_t qs, const void* kc, const void* vc, int64_t T, int64_t Hq, int64_t Hkv, int64_t D, int64_t cap,
                    int64_t pa, float* out, float* scratch) {
    NS_TRY
    NS_CU(cudaSetDevice(g->dev));
    if (T <= 0) return 0;
    if (D != DA_D || Hkv <= 0 || Hq % Hkv || (Hq / Hkv) * T > DA_MAXQ || pa + T > cap)
        return ns_fail("ns_q35_dsp_attn: heads of 128, at most 32 queries a key head, the keys within the cache");
    const int64_t runs = (pa + T + DA_KEYS - 1) / DA_KEYS;
    dsp_attn_k<<<dim3((unsigned) Hkv, (unsigned) runs), 128, 0, g->q>>>(q, qs, (const __half*) kc, (const __half*) vc, (int) T, (int) Hq, (int) Hkv, cap, pa,
                                                                         scratch);
    NS_LAUNCH_CHECK();
    dsp_attn_merge_k<<<(unsigned) (T * Hq), DA_D, 0, g->q>>>(scratch, (int) T, (int) Hq, (int) Hkv, (int) runs, out);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

}  // extern "C"
