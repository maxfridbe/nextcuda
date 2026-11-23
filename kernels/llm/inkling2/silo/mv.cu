// mv.cu: the inkling2 silo's products (silo.h) - its repacked Q8_0 / NVFP4 matrices against Q8_0-quantized
// activations at decode width (1..16 rows): one matrix, several in one launch, the routed experts (gate | up with SiTU and
// the activations' quantization made in the launch; down weighted and summed), the union forms of a verify pass; the
// float32 / BF16 products; the activations' quantization (with the RMS norm); the repacking itself and its expansion to
// half for prompt products. nextsycl's mv.cpp / fmv.cpp API, written for CUDA: 8 lanes a row (4 rows a warp), each
// lane a 32-value line at a time (16 bytes of NVFP4 codes or 32 of Q8_0 quants), the line decoded once and dotted (dp4a)
// with each row of activations; the lanes' sums added in a fixed order whatever the rows, so a row's result does not
// depend on what shares its pass.
#include "common.cuh"

using namespace ink2s;

namespace {

constexpr int MAXT = 16;
constexpr int OCT = 8;            // lanes a matrix row
constexpr int RPW = WS / OCT;     // rows a warp
constexpr int MV_WARPS = 4;       // warps a block
constexpr int RPB = RPW * MV_WARPS;

// NVFP4's UE4M3 scale byte as a float, halved (its codes below are twice their E2M1 values)
__device__ __forceinline__ float ue4m3h(uint32_t x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    const uint32_t e = (x >> 3) & 0xF, m = x & 7;
    return e == 0 ? (float) m * (1.0f / 1024.0f) : __uint_as_float(((e + 119) << 23) | (m << 20));
}

// four E2M1 codes (the selector's four nibbles) as four int8 values twice their E2M1 values: the magnitudes from one
// byte table, their negations from another, each byte picked by its code's sign bit
__device__ __forceinline__ int dec4(uint32_t sel) {
    const uint32_t s = sel & 0x7777u;
    const uint32_t pos = __byte_perm(0x03020100u, 0x0C080604u, s);
    const uint32_t neg = __byte_perm(0xFDFEFF00u, 0xF4F8FAFCu, s);
    return (int) __byte_perm(pos, neg, ((sel & 0x8888u) >> 1) | 0x3210u);
}

// a repacked matrix: the quants of all rows, then the scales
struct Packed {
    const uint8_t* q;
    const uint8_t* s;
};
__host__ __device__ inline size_t scales_at(int type, int64_t rows, int64_t n) {
    return align256(type == 40 ? (size_t) (rows * n / 2) : (size_t) (rows * n));
}
__device__ __forceinline__ Packed packed(int type, const void* w, int64_t rows, int64_t n) {
    return Packed{(const uint8_t*) w, (const uint8_t*) w + scales_at(type, rows, n)};
}

// row `row` of a repacked matrix (of n values) against activation rows xr[0..nt) (each its quants, its scales): this
// lane's share of each row's dot, lines lane, lane + 8, ...; the octet's sums after
template <int TY, int TT>
__device__ __forceinline__ void row_dot(Packed w, int64_t row, int64_t n, const int8_t* const* xq, const float* const* xd, int nt, int lane, float* acc) {
    const int64_t L = n / 32;
#pragma unroll
    for (int t = 0; t < TT; ++t) acc[t] = 0.0f;
    if constexpr (TY == 40) {
        const uint4* cq = (const uint4*) (w.q + row * (n / 2));
        const uint16_t* cs = (const uint16_t*) (w.s + row * (n / 16));
        for (int64_t l = lane; l < L; l += OCT) {
            const uint4 c = __ldg(cq + l);
            const uint32_t sb = __ldg(cs + l);
            const float sa = ue4m3h(sb & 0xFF), sbb = ue4m3h(sb >> 8);
            const int d0 = dec4(c.x), d1 = dec4(c.x >> 16), d2 = dec4(c.y), d3 = dec4(c.y >> 16);
            const int e0 = dec4(c.z), e1 = dec4(c.z >> 16), e2 = dec4(c.w), e3 = dec4(c.w >> 16);
#pragma unroll
            for (int t = 0; t < TT; ++t) {
                if (t >= nt) break;
                const int4* xp = (const int4*) (xq[t] + l * 32);
                const int4 x0 = __ldg(xp), x1 = __ldg(xp + 1);
                const int ia = __dp4a(d3, x0.w, __dp4a(d2, x0.z, __dp4a(d1, x0.y, __dp4a(d0, x0.x, 0))));
                const int ib = __dp4a(e3, x1.w, __dp4a(e2, x1.z, __dp4a(e1, x1.y, __dp4a(e0, x1.x, 0))));
                acc[t] = __fmaf_rn(__ldg(xd[t] + l), __fmaf_rn(sa, (float) ia, __fmul_rn(sbb, (float) ib)), acc[t]);
            }
        }
    } else {
        const uint4* wq = (const uint4*) (w.q + row * n);
        const __half* ws = (const __half*) (w.s) + row * L;
        for (int64_t l = lane; l < L; l += OCT) {
            const uint4 a = __ldg(wq + 2 * l), b = __ldg(wq + 2 * l + 1);
            const float wd = __half2float(ws[l]);
#pragma unroll
            for (int t = 0; t < TT; ++t) {
                if (t >= nt) break;
                const int4* xp = (const int4*) (xq[t] + l * 32);
                const int4 x0 = __ldg(xp), x1 = __ldg(xp + 1);
                int s = __dp4a((int) a.x, x0.x, 0);
                s = __dp4a((int) a.y, x0.y, s);
                s = __dp4a((int) a.z, x0.z, s);
                s = __dp4a((int) a.w, x0.w, s);
                s = __dp4a((int) b.x, x1.x, s);
                s = __dp4a((int) b.y, x1.y, s);
                s = __dp4a((int) b.z, x1.z, s);
                s = __dp4a((int) b.w, x1.w, s);
                acc[t] = __fmaf_rn(__ldg(xd[t] + l), __fmul_rn(wd, (float) s), acc[t]);
            }
        }
    }
    // the octet's sums (lanes 8k .. 8k + 7), a fixed order
#pragma unroll
    for (int t = 0; t < TT; ++t) {
        float v = acc[t];
        v += __shfl_xor_sync(0xffffffffu, v, 4);
        v += __shfl_xor_sync(0xffffffffu, v, 2);
        v += __shfl_xor_sync(0xffffffffu, v, 1);
        acc[t] = v;
    }
}

struct MvJob {
    int ty;
    const void* w;
    const void* xq;
    float* y;
    int64_t rows, n, blocks0;
};
struct MvJobs {
    MvJob j[8];
    int n;
};

// y [T, rows] = W . x for each job (its blocks of RPB rows after the earlier jobs'), T rows of activations
template <int TT>
__global__ void mv_k(MvJobs jobs, int64_t T) {
    int ji = 0;
    while (ji + 1 < jobs.n && (int64_t) blockIdx.x >= jobs.j[ji + 1].blocks0) ++ji;
    const MvJob jb = jobs.j[ji];
    const int lane = threadIdx.x % OCT;
    const int64_t rw = ((int64_t) blockIdx.x - jb.blocks0) * RPB + threadIdx.x / OCT;
    // a row past the matrix computes row 0 and writes nothing (every lane of the warp stays for the shuffles)
    const bool live = rw < jb.rows;
    const int64_t row = live ? rw : 0;
    const int8_t* xq[TT];
    const float* xd[TT];
#pragma unroll
    for (int t = 0; t < TT; ++t) {
        xq[t] = xq_q((void*) jb.xq) + t * jb.n;
        xd[t] = xq_d((void*) jb.xq, T, jb.n) + t * (jb.n / 32);
    }
    float acc[TT];
    const Packed w = packed(jb.ty, jb.w, jb.rows, jb.n);
    if (jb.ty == 40) row_dot<40, TT>(w, row, jb.n, xq, xd, (int) T, lane, acc);
    else row_dot<8, TT>(w, row, jb.n, xq, xd, (int) T, lane, acc);
    if (lane == 0 && live) {
#pragma unroll
        for (int t = 0; t < TT; ++t)
            if (t < T) jb.y[t * jb.rows + row] = acc[t];
    }
}

int mv_launch(ns_gpu* g, MvJobs& jobs, int64_t T) {
    int64_t total = 0;
    for (int i = 0; i < jobs.n; ++i) {
        jobs.j[i].blocks0 = total;
        total += (jobs.j[i].rows + RPB - 1) / RPB;
    }
    if (total <= 0) return 0;
    const unsigned grid = (unsigned) total, thr = MV_WARPS * WS;
    if (T == 1) mv_k<1><<<grid, thr, 0, g->q>>>(jobs, T);
    else if (T == 2) mv_k<2><<<grid, thr, 0, g->q>>>(jobs, T);
    else if (T <= 4) mv_k<4><<<grid, thr, 0, g->q>>>(jobs, T);
    else if (T <= 8) mv_k<8><<<grid, thr, 0, g->q>>>(jobs, T);
    else mv_k<16><<<grid, thr, 0, g->q>>>(jobs, T);
    NS_LAUNCH_CHECK();
    return 0;
}

bool supported(int type, int64_t n) {
    return (type == 8 || type == 40) && n % 64 == 0 && n > 0;
}

// ---- the routed experts ----
__device__ __forceinline__ float situ(float g, float u, float beta, float lb) {
    const float a = beta * tanhf(g / beta) * (1.0f / (1.0f + expf(-g)));
    const float v = lb > 0.0f ? lb * tanhf(u / lb) : u;
    return a * v;
}

// gate | up: a block 32 of the F activations of `np` pairs that share an expert (np 1: one pair) - gate rows [j0, j0 + 32)
// and up rows [F + j0, ..) of the expert, 16 warps; each pair's 32 values through SiTU, quantized into act_q's row p
// the pairs of block (blockIdx.x: the 32-column group; blockIdx.y: a pair, or a distinct expert of uni)
template <int TY, int NP>
__global__ void __launch_bounds__(512) moe_gu_k(const void* wv, int64_t E, const float* scale, const int32_t* ids, const int32_t* uni, const void* xqv, void* actv, int64_t T,
                         int64_t K, int64_t n, int64_t F, float beta, float lb) {
    __shared__ float vals[NP][64];
    const int64_t P = T * K;
    int np = 1, e;
    int pairs[NP];
    if (uni) {
        const int u = (int) blockIdx.y;
        if (u >= uni[0]) return;
        e = uni[1 + u];
        np = uni[1 + P + u];
        for (int i = 0; i < NP; ++i) pairs[i] = i < np ? uni[1 + 2 * P + u * 16 + i] : 0;
    } else {
        pairs[0] = (int) blockIdx.y;
        e = ids[pairs[0]];
    }
    const int lane = threadIdx.x % OCT;
    const int r = threadIdx.x / OCT;                  // 0..63
    const int64_t j0 = (int64_t) blockIdx.x * 32;
    const int64_t row = (int64_t) e * 2 * F + (r < 32 ? j0 + r : F + j0 + r - 32);
    const int8_t* xq[NP];
    const float* xd[NP];
    for (int i = 0; i < NP; ++i) {
        const int64_t t = pairs[i] / K;
        xq[i] = xq_q((void*) xqv) + t * n;
        xd[i] = xq_d((void*) xqv, T, n) + t * (n / 32);
    }
    float acc[NP];
    const Packed w = packed(TY, wv, E * 2 * F, n);
    row_dot<TY, NP>(w, row, n, xq, xd, np, lane, acc);
    const float sc = scale ? scale[e] : 1.0f;
    if (lane == 0)
        for (int i = 0; i < NP; ++i)
            if (i < np) vals[i][r] = acc[i] * sc;
    __syncthreads();
    // each pair's 32 activations: a warp a pair
    const int wi = threadIdx.x / WS, wl = threadIdx.x % WS;
    for (int i = wi; i < np; i += 16) {
        const float a = situ(vals[i][wl], vals[i][32 + wl], beta, lb);
        const int64_t p = pairs[i];
        quant32(a, wl, xq_q(actv) + p * F + j0, xq_d(actv, P, F) + p * (F / 32) + j0 / 32);
    }
}

// down: the K experts of row t (blockIdx.y), each row of the block's RPB output rows weighted and summed in order
template <int TY>
__global__ void moe_down_k(const void* wv, int64_t E, const float* scale, const int32_t* ids, const float* wts, const void* actv, float* y, int64_t T,
                           int64_t K, int64_t F, int64_t n_out) {
    const int lane = threadIdx.x % OCT;
    const int64_t rw = (int64_t) blockIdx.x * RPB + threadIdx.x / OCT, t = blockIdx.y;
    const bool live = rw < n_out;
    const int64_t row = live ? rw : 0;
    const int64_t P = T * K;
    const Packed w = packed(TY, wv, E * n_out, F);
    float sum = 0.0f;
    for (int64_t k = 0; k < K; ++k) {
        const int64_t p = t * K + k;
        const int e = ids[p];
        const int8_t* xq[1] = {xq_q((void*) actv) + p * F};
        const float* xd[1] = {xq_d((void*) actv, P, F) + p * (F / 32)};
        float acc[1];
        row_dot<TY, 1>(w, (int64_t) e * n_out + row, F, xq, xd, 1, lane, acc);
        const float part = __fmul_rn(scale ? scale[e] : 1.0f, acc[0]);
        sum = __fmaf_rn(wts[p], part, sum);
    }
    if (lane == 0 && live) y[t * n_out + row] = sum;
}

// down, the union form: each distinct expert (blockIdx.y) against its pairs: part [P, n_out] = scale (W . act)
template <int TY>
__global__ void moe_down_union_k(const void* wv, int64_t E, const float* scale, const int32_t* uni, const void* actv, float* part, int64_t T, int64_t K,
                                 int64_t F, int64_t n_out) {
    const int u = (int) blockIdx.y;
    if (u >= uni[0]) return;
    const int lane = threadIdx.x % OCT;
    const int64_t rw = (int64_t) blockIdx.x * RPB + threadIdx.x / OCT;
    const bool live = rw < n_out;
    const int64_t row = live ? rw : 0;
    const int64_t P = T * K;
    const int e = uni[1 + u], np = uni[1 + P + u];
    const int8_t* xq[MAXT];
    const float* xd[MAXT];
    for (int i = 0; i < MAXT; ++i) {
        const int64_t p = i < np ? uni[1 + 2 * P + u * 16 + i] : 0;
        xq[i] = xq_q((void*) actv) + p * F;
        xd[i] = xq_d((void*) actv, P, F) + p * (F / 32);
    }
    const Packed w = packed(TY, wv, E * n_out, F);
    float acc[MAXT];
    row_dot<TY, MAXT>(w, (int64_t) e * n_out + row, F, xq, xd, np, lane, acc);
    if (lane == 0 && live) {
        const float sc = scale ? scale[e] : 1.0f;
        for (int i = 0; i < np; ++i) part[(int64_t) uni[1 + 2 * P + u * 16 + i] * n_out + row] = __fmul_rn(sc, acc[i]);
    }
}

__global__ void moe_combine_k(const float* part, const float* wts, float* y, int64_t T, int64_t K, int64_t n_out) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * n_out) return;
    const int64_t t = i / n_out, row = i % n_out;
    float sum = 0.0f;
    for (int64_t k = 0; k < K; ++k) sum = __fmaf_rn(wts[t * K + k], part[(t * K + k) * n_out + row], sum);
    y[i] = sum;
}

// a pass's distinct experts in order of first use, each one's pairs
__global__ void moe_union_k(const int32_t* ids, int64_t P, int32_t* uni) {
    if (threadIdx.x != 0) return;
    int n = 0;
    for (int64_t p = 0; p < P; ++p) {
        const int e = ids[p];
        int u = 0;
        while (u < n && uni[1 + u] != e) ++u;
        if (u == n) {
            uni[1 + u] = e;
            uni[1 + P + u] = 0;
            ++n;
        }
        const int c = uni[1 + P + u]++;
        if (c < 16) uni[1 + 2 * P + u * 16 + c] = (int) p;
    }
    uni[0] = n;
}

// ---- the activations ----
// rows x [T, n] (normed by w with eps when norm is set) quantized; y (nullable) the normed rows
struct P4 {
    const float* x[4];
    const float* w[4];
    float* y[4];
    void* q[4];
};

__global__ void rms_xq4_k(P4 p, int64_t T, int64_t n, float eps, int norm) {
    __shared__ float sh[32];
    const int64_t t = blockIdx.x;
    const int m = blockIdx.y;
    const float* x = p.x[m] + t * n;
    const float* w = p.w[m];
    float* y = p.y[m];
    void* q = p.q[m];
    float inv = 1.0f;
    if (norm) {
        float ss = 0.0f;
        for (int64_t c = threadIdx.x; c < n; c += blockDim.x) ss += x[c] * x[c];
        inv = rsqrtf(block_sum(ss, sh) / (float) n + eps);
    }
    const int wi = threadIdx.x / WS, lane = threadIdx.x % WS, nw = blockDim.x / WS;
    for (int64_t b = wi; b < n / 32; b += nw) {
        const int64_t c = b * 32 + lane;
        const float v = norm ? x[c] * inv * w[c] : x[c];
        if (y) y[t * n + c] = v;
        quant32(v, lane, xq_q(q) + t * n + b * 32, xq_d(q, T, n) + t * (n / 32) + b);
    }
}

// ---- the repacking ----
// a thread a 32-value line of a row: its quants into the 0 8 1 9 ... order (Q8_0) or its codes as they are (NVFP4), its
// scales after all the rows' quants
__global__ void pack_k(int type, const uint8_t* raw, int64_t rows, int64_t n, uint8_t* dst) {
    const int64_t L = n / 32;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * L) return;
    const int64_t r = i / L, l = i % L;
    uint8_t* sc = dst + scales_at(type, rows, n);
    if (type == 8) {
        const uint8_t* b = raw + (r * L + l) * 34;
        int8_t* q = (int8_t*) dst + r * n + l * 32;
        for (int v = 0; v < 32; ++v) q[qpos(v)] = (int8_t) b[2 + v];
        ((uint16_t*) sc)[r * L + l] = (uint16_t) (b[0] | (b[1] << 8));
    } else {
        // NVFP4 blocks of 64: line l is half `l & 1` of block l / 2 (its two sub-blocks' scales and 16 code bytes)
        const uint8_t* b = raw + (r * (n / 64) + l / 2) * 36;
        const int h = (int) (l & 1);
        uint8_t* q = dst + r * (n / 2) + l * 16;
        for (int k = 0; k < 16; ++k) q[k] = b[4 + h * 16 + k];
        sc[r * (n / 16) + l * 2] = b[h * 2];
        sc[r * (n / 16) + l * 2 + 1] = b[h * 2 + 1];
    }
}

__device__ __forceinline__ float ue4m3_full(uint32_t x) { return ue4m3h(x); }

// rows [row0, row0 + nr) of a repacked matrix back to half, in the values' own order
__global__ void unpack_k(int type, const uint8_t* w, int64_t rows, int64_t row0, int64_t nr, int64_t n, __half* dst) {
    const int64_t L = n / 32;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nr * L) return;
    const int64_t r = row0 + i / L, l = i % L;
    const uint8_t* sc = w + scales_at(type, rows, n);
    __half* o = dst + (i / L) * n + l * 32;
    if (type == 8) {
        const int8_t* q = (const int8_t*) w + r * n + l * 32;
        const float d = __half2float(__ushort_as_half(((const uint16_t*) sc)[r * L + l]));
        for (int v = 0; v < 32; ++v) o[v] = __float2half_rn(d * (float) q[qpos(v)]);
    } else {
        const uint8_t* q = w + r * (n / 2) + l * 16;
        for (int s = 0; s < 2; ++s) {
            const float d = ue4m3_full(sc[r * (n / 16) + l * 2 + s]);
            for (int j = 0; j < 8; ++j) {
                const uint8_t c = q[s * 8 + j];
                const int lo = c & 15, hi = c >> 4;
                const float m[8] = {0, 1, 2, 3, 4, 6, 8, 12};
                o[s * 16 + j] = __float2half_rn((lo & 8 ? -m[lo & 7] : m[lo & 7]) * d);
                o[s * 16 + j + 8] = __float2half_rn((hi & 8 ? -m[hi & 7] : m[hi & 7]) * d);
            }
        }
    }
}

// ---- float32 / BF16 products ----
template <int TY, int TT>
__global__ void fmv_k(const void* wv, const float* x, float* y, int64_t n, int64_t rows, int64_t T) {
    const int lane = threadIdx.x % WS;
    const int64_t row = (int64_t) blockIdx.x * 4 + threadIdx.x / WS;
    if (row >= rows) return;
    float acc[TT];
#pragma unroll
    for (int t = 0; t < TT; ++t) acc[t] = 0.0f;
    for (int64_t i = lane * 8; i < n; i += WS * 8) {
        float wv8[8];
        if constexpr (TY == 30) {
            const uint4 u = __ldg((const uint4*) ((const uint16_t*) wv + row * n + i));
            const uint32_t uu[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                wv8[2 * k] = __uint_as_float(uu[k] << 16);
                wv8[2 * k + 1] = __uint_as_float(uu[k] & 0xFFFF0000u);
            }
        } else {
            const float4 a = __ldg((const float4*) ((const float*) wv + row * n + i)), b = __ldg((const float4*) ((const float*) wv + row * n + i) + 1);
            wv8[0] = a.x; wv8[1] = a.y; wv8[2] = a.z; wv8[3] = a.w; wv8[4] = b.x; wv8[5] = b.y; wv8[6] = b.z; wv8[7] = b.w;
        }
#pragma unroll
        for (int t = 0; t < TT; ++t) {
            if (t >= T) break;
            const float4 a = __ldg((const float4*) (x + t * n + i)), b = __ldg((const float4*) (x + t * n + i) + 1);
            float s = acc[t];
            s = fmaf(wv8[0], a.x, s); s = fmaf(wv8[1], a.y, s); s = fmaf(wv8[2], a.z, s); s = fmaf(wv8[3], a.w, s);
            s = fmaf(wv8[4], b.x, s); s = fmaf(wv8[5], b.y, s); s = fmaf(wv8[6], b.z, s); s = fmaf(wv8[7], b.w, s);
            acc[t] = s;
        }
    }
#pragma unroll
    for (int t = 0; t < TT; ++t) {
        if (t >= T) break;
        const float v = warp_sum(acc[t]);
        if (lane == 0) y[t * rows + row] = v;
    }
}

template <int TY>
void fmv_launch(ns_gpu* g, const void* w, const float* x, float* y, int64_t n, int64_t rows, int64_t T) {
    const unsigned grid = (unsigned) ((rows + 3) / 4);
    if (T == 1) fmv_k<TY, 1><<<grid, 4 * WS, 0, g->q>>>(w, x, y, n, rows, T);
    else if (T <= 4) fmv_k<TY, 4><<<grid, 4 * WS, 0, g->q>>>(w, x, y, n, rows, T);
    else fmv_k<TY, 16><<<grid, 4 * WS, 0, g->q>>>(w, x, y, n, rows, T);
}

}  // namespace

extern "C" {

int ns_ink2s_supported(int type, int64_t n_in) {
    return supported(type, n_in) ? 1 : 0;
}

size_t ns_ink2s_scales_at(int type, int64_t rows, int64_t n_in) {
    return scales_at(type, rows, n_in);
}

size_t ns_ink2s_packed_bytes(int type, int64_t rows, int64_t n_in) {
    return scales_at(type, rows, n_in) + (size_t) (type == 40 ? rows * n_in / 16 : rows * (n_in / 32) * 2);
}

int ns_ink2s_pack(ns_gpu* g, int type, const void* raw, int64_t rows, int64_t n_in, void* dst) {
    NS_TRY
    if (!supported(type, n_in)) return ns_fail("ns_ink2s_pack: Q8_0 or NVFP4 rows of a multiple of 64");
    NS_CU(cudaSetDevice(g->dev));
    pack_k<<<blocks(rows * (n_in / 32)), 256, 0, g->q>>>(type, (const uint8_t*) raw, rows, n_in, (uint8_t*) dst);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_ink2s_unpack_f16(ns_gpu* g, int type, const void* w, int64_t rows, int64_t row0, int64_t n, int64_t n_in, uint16_t* dst) {
    NS_TRY
    if (!supported(type, n_in)) return ns_fail("ns_ink2s_unpack_f16: Q8_0 or NVFP4");
    if (n <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    unpack_k<<<blocks(n * (n_in / 32)), 256, 0, g->q>>>(type, (const uint8_t*) w, rows, row0, n, n_in, (__half*) dst);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

size_t ns_ink2s_xq_bytes(int64_t T, int64_t n) {
    return align256((size_t) (T * n)) + (size_t) (T * (n / 32)) * 4;
}

size_t ns_ink2s_xq_scales_at(int64_t T, int64_t n) {
    return align256((size_t) (T * n));
}

int ns_ink2s_xq(ns_gpu* g, const float* x, void* xq, int64_t T, int64_t n) {
    NS_TRY
    if (n % 32) return ns_fail("ns_ink2s_xq: rows of whole blocks of 32");
    if (T <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    P4 p{};
    p.x[0] = x;
    p.q[0] = xq;
    rms_xq4_k<<<dim3((unsigned) T, 1), 256, 0, g->q>>>(p, T, n, 0.0f, 0);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_ink2s_rms_xq(ns_gpu* g, const float* x, const float* w, float* y, void* xq, int64_t T, int64_t n, float eps) {
    return ns_ink2s_rms_xq_many(g, 1, &x, &w, &y, &xq, T, n, eps);
}

int ns_ink2s_rms_xq_many(ns_gpu* g, int m, const float* const* x, const float* const* w, float* const* y, void* const* xq, int64_t T, int64_t n, float eps) {
    NS_TRY
    if (m < 1 || m > 4 || n % 32) return ns_fail("ns_ink2s_rms_xq_many: 1..4 inputs, rows of whole blocks of 32");
    if (T <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    P4 p{};
    for (int i = 0; i < m; ++i) {
        p.x[i] = x[i];
        p.w[i] = w[i];
        p.y[i] = y ? y[i] : nullptr;
        p.q[i] = xq[i];
    }
    rms_xq4_k<<<dim3((unsigned) T, (unsigned) m), 256, 0, g->q>>>(p, T, n, eps, 1);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_ink2s_mv(ns_gpu* g, int type, const void* w, const void* xq, float* y, int64_t n_in, int64_t rows, int64_t T) {
    NS_TRY
    if (!supported(type, n_in) || T < 1 || T > MAXT) return ns_fail("ns_ink2s_mv: Q8_0 / NVFP4 rows of a multiple of 64, 1..16 rows of activations");
    NS_CU(cudaSetDevice(g->dev));
    MvJobs jobs{};
    jobs.n = 1;
    jobs.j[0] = MvJob{type, w, xq, y, rows, n_in, 0};
    return mv_launch(g, jobs, T);
    NS_CATCH
}

int ns_ink2s_mv_many(ns_gpu* g, int n, const int* ty, const void* const* w, const void* const* xq, float* const* y, const int64_t* rows, const int64_t* n_in,
                     int64_t T) {
    NS_TRY
    if (n < 1 || n > 8 || T < 1 || T > MAXT) return ns_fail("ns_ink2s_mv_many: 1..8 products, 1..16 rows");
    NS_CU(cudaSetDevice(g->dev));
    MvJobs jobs{};
    jobs.n = n;
    for (int i = 0; i < n; ++i) {
        if (!supported(ty[i], n_in[i])) return ns_fail("ns_ink2s_mv_many: Q8_0 / NVFP4 rows of a multiple of 64");
        jobs.j[i] = MvJob{ty[i], w[i], xq[i], y[i], rows[i], n_in[i], 0};
    }
    return mv_launch(g, jobs, T);
    NS_CATCH
}

int ns_ink2s_fmv_supported(int type, int64_t n_in) {
    return (type == 0 || type == 30) && n_in % 256 == 0 ? 1 : 0;
}

int ns_ink2s_fmv(ns_gpu* g, int type, const void* w, const float* x, float* y, int64_t n_in, int64_t rows, int64_t T) {
    NS_TRY
    if (!ns_ink2s_fmv_supported(type, n_in) || T < 1 || T > MAXT) return ns_fail("ns_ink2s_fmv: float32 / BF16 rows of a multiple of 256, 1..16 rows");
    if (rows <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    if (type == 30) fmv_launch<30>(g, w, x, y, n_in, rows, T);
    else fmv_launch<0>(g, w, x, y, n_in, rows, T);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_ink2s_moe_gu(ns_gpu* g, int type, const void* w, int64_t E, const float* scale, const int32_t* ids, const void* xq, void* act_q, int64_t T, int64_t K,
                    int64_t n_in, int64_t F, float beta, float lb) {
    NS_TRY
    if (!supported(type, n_in) || F % 32) return ns_fail("ns_ink2s_moe_gu: Q8_0 / NVFP4, F a multiple of 32");
    if (T <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    const dim3 grid((unsigned) (F / 32), (unsigned) (T * K));
    if (type == 40) moe_gu_k<40, 1><<<grid, 64 * OCT, 0, g->q>>>(w, E, scale, ids, nullptr, xq, act_q, T, K, n_in, F, beta, lb);
    else moe_gu_k<8, 1><<<grid, 64 * OCT, 0, g->q>>>(w, E, scale, ids, nullptr, xq, act_q, T, K, n_in, F, beta, lb);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_ink2s_moe_union(ns_gpu* g, const int32_t* ids, int64_t P, int32_t* uni) {
    NS_TRY
    if (P < 1 || P > 256) return ns_fail("ns_ink2s_moe_union: 1..256 pairs");
    NS_CU(cudaSetDevice(g->dev));
    moe_union_k<<<1, 32, 0, g->q>>>(ids, P, uni);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_ink2s_moe_gu_union(ns_gpu* g, int type, const void* w, int64_t E, const float* scale, const int32_t* uni, const void* xq, void* act_q, int64_t T,
                          int64_t K, int64_t n_in, int64_t F, float beta, float lb) {
    NS_TRY
    if (!supported(type, n_in) || F % 32 || T > MAXT) return ns_fail("ns_ink2s_moe_gu_union: Q8_0 / NVFP4, F a multiple of 32, at most 16 rows");
    if (T <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    const dim3 grid((unsigned) (F / 32), (unsigned) (T * K));
    // an expert's pairs: at most one a row
    if (type == 40) moe_gu_k<40, MAXT><<<grid, 64 * OCT, 0, g->q>>>(w, E, scale, nullptr, uni, xq, act_q, T, K, n_in, F, beta, lb);
    else moe_gu_k<8, MAXT><<<grid, 64 * OCT, 0, g->q>>>(w, E, scale, nullptr, uni, xq, act_q, T, K, n_in, F, beta, lb);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_ink2s_moe_down(ns_gpu* g, int type, const void* w, int64_t E, const float* scale, const int32_t* ids, const float* wts, const void* act_q, float* y,
                      int64_t T, int64_t K, int64_t F, int64_t n_out) {
    NS_TRY
    if (!supported(type, F)) return ns_fail("ns_ink2s_moe_down: Q8_0 / NVFP4, F a multiple of 64");
    if (T <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    const dim3 grid((unsigned) ((n_out + RPB - 1) / RPB), (unsigned) T);
    if (type == 40) moe_down_k<40><<<grid, MV_WARPS * WS, 0, g->q>>>(w, E, scale, ids, wts, act_q, y, T, K, F, n_out);
    else moe_down_k<8><<<grid, MV_WARPS * WS, 0, g->q>>>(w, E, scale, ids, wts, act_q, y, T, K, F, n_out);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_ink2s_moe_down_union(ns_gpu* g, int type, const void* w, int64_t E, const float* scale, const int32_t* uni, const float* wts, const void* act_q,
                            float* part, float* y, int64_t T, int64_t K, int64_t F, int64_t n_out) {
    NS_TRY
    if (!supported(type, F) || T > MAXT) return ns_fail("ns_ink2s_moe_down_union: Q8_0 / NVFP4, F a multiple of 64, at most 16 rows");
    if (T <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    const dim3 grid((unsigned) ((n_out + RPB - 1) / RPB), (unsigned) (T * K));
    if (type == 40) moe_down_union_k<40><<<grid, MV_WARPS * WS, 0, g->q>>>(w, E, scale, uni, act_q, part, T, K, F, n_out);
    else moe_down_union_k<8><<<grid, MV_WARPS * WS, 0, g->q>>>(w, E, scale, uni, act_q, part, T, K, F, n_out);
    NS_LAUNCH_CHECK();
    moe_combine_k<<<blocks(T * n_out), 256, 0, g->q>>>(part, wts, y, T, K, n_out);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

}  // extern "C"
