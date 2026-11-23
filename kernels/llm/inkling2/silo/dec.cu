// dec.cu: the inkling2 silo's fused decode steps (silo.h) - a layer's small steps in few launches: the depth attention
// with the norms and the activations' quantization after it; the convolution of k and v with their clip / norm and the
// cache write, with q's and r's clips (and the full layers' log scale) and the BF16 sink / tap projections; the merge of
// the attention's splits with the sink, the output gate and the quantization for the output product; the convolutions
// of the attention's and the FFN's outputs with the residual; the loop controllers in two launches; the router's product
// and choice. nextsycl's dec.cpp, written for CUDA. Every row's arithmetic is the same whatever rows share a pass.
#include "common.cuh"

using namespace ink2s;

namespace {

constexpr int MAXS = 8, MAXQ = 4, MAXC = 2304;

struct MixArgs {
    const float* st[MAXS];
    int ns, nq;
    const float* query;
    const float* nw[MAXQ];   // a query's norm weight (null: the mix unnormed)
    float* out;              // [nq, T, C]
    void* xq[MAXQ];          // a normed query's quantized rows (nullable)
    int64_t T, C;
    float eps_mix, eps_norm;
};

constexpr int MIXW = 576;    // C / 4 for 2304

// a block's sums of NV values (each thread's partials), every thread gets them; sh [nwarps * NV + NV]
template <int NV>
__device__ __forceinline__ void block_sums(float* part, float* sh) {
    const int lane = threadIdx.x % WS, w = threadIdx.x / WS, nw = blockDim.x / WS;
#pragma unroll
    for (int v = 0; v < NV; ++v) {
        const float s = warp_sum(part[v]);
        if (lane == 0) sh[w * NV + v] = s;
    }
    __syncthreads();
    if ((int) threadIdx.x < NV) {
        float s = 0.0f;
        for (int i = 0; i < nw; ++i) s += sh[i * NV + threadIdx.x];
        sh[nw * NV + threadIdx.x] = s;
    }
    __syncthreads();
#pragma unroll
    for (int v = 0; v < NV; ++v) part[v] = sh[nw * NV + v];
    __syncthreads();
}

// a block a row: one pass over the channels (16-byte loads) for the states' sums of squares and the (query, state)
// dots; every query's mix in a second (into shared memory) with its sum of squares; then each normed, written and
// quantized
template <int S, int Q>
__global__ void mix_k(MixArgs a) {
    constexpr int NV = S + Q * S;
    __shared__ float sh[(MIXW / WS) * NV + NV + 8];
    __shared__ __align__(16) float rb[Q * MAXC];
    const int64_t t = blockIdx.x;
    const int tid = threadIdx.x;
    const int64_t C = a.C, c4 = C / 4;
    float part[NV];
#pragma unroll
    for (int v = 0; v < NV; ++v) part[v] = 0.0f;
    for (int64_t i = tid; i < c4; i += MIXW) {
        float4 qv[Q];
#pragma unroll
        for (int qi = 0; qi < Q; ++qi) qv[qi] = ((const float4*) (a.query + qi * C))[i];
#pragma unroll
        for (int s = 0; s < S; ++s) {
            const float4 x = ((const float4*) (a.st[s] + t * C))[i];
            part[s] += x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w;
#pragma unroll
            for (int qi = 0; qi < Q; ++qi) part[S + qi * S + s] += x.x * qv[qi].x + x.y * qv[qi].y + x.z * qv[qi].z + x.w * qv[qi].w;
        }
    }
    block_sums<NV>(part, sh);
    float inv[S], pr[Q][S];
#pragma unroll
    for (int s = 0; s < S; ++s) inv[s] = rsqrtf(part[s] / (float) C + a.eps_mix);
    const float sc = 1.0f / sqrtf((float) C);
#pragma unroll
    for (int qi = 0; qi < Q; ++qi) {
        float mx = -INFINITY;
#pragma unroll
        for (int s = 0; s < S; ++s) {
            pr[qi][s] = part[S + qi * S + s] * inv[s] * sc;
            mx = fmaxf(mx, pr[qi][s]);
        }
        float sum = 0.0f;
#pragma unroll
        for (int s = 0; s < S; ++s) {
            pr[qi][s] = expf(pr[qi][s] - mx);
            sum += pr[qi][s];
        }
#pragma unroll
        for (int s = 0; s < S; ++s) pr[qi][s] = pr[qi][s] / sum * inv[s];
    }
    float ssq[Q];
#pragma unroll
    for (int qi = 0; qi < Q; ++qi) ssq[qi] = 0.0f;
    for (int64_t i = tid; i < c4; i += MIXW) {
        float4 y[Q];
#pragma unroll
        for (int qi = 0; qi < Q; ++qi) y[qi] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
        for (int s = 0; s < S; ++s) {
            const float4 x = ((const float4*) (a.st[s] + t * C))[i];
#pragma unroll
            for (int qi = 0; qi < Q; ++qi) {
                y[qi].x += pr[qi][s] * x.x;
                y[qi].y += pr[qi][s] * x.y;
                y[qi].z += pr[qi][s] * x.z;
                y[qi].w += pr[qi][s] * x.w;
            }
        }
#pragma unroll
        for (int qi = 0; qi < Q; ++qi) {
            ((float4*) (rb + qi * C))[i] = y[qi];
            ssq[qi] += y[qi].x * y[qi].x + y[qi].y * y[qi].y + y[qi].z * y[qi].z + y[qi].w * y[qi].w;
        }
    }
    block_sums<Q>(ssq, sh);
    for (int64_t i = tid; i < c4; i += MIXW) {
#pragma unroll
        for (int qi = 0; qi < Q; ++qi) {
            float4 y = ((const float4*) (rb + qi * C))[i];
            if (a.nw[qi]) {
                const float ninv = rsqrtf(ssq[qi] / (float) C + a.eps_norm);
                const float4 w = ((const float4*) a.nw[qi])[i];
                y = make_float4(y.x * ninv * w.x, y.y * ninv * w.y, y.z * ninv * w.z, y.w * ninv * w.w);
                ((float4*) (rb + qi * C))[i] = y;
            }
            ((float4*) (a.out + (qi * a.T + t) * C))[i] = y;
        }
    }
    __syncthreads();
    // the normed rows quantized: a warp a block of 32 in turn, over the queries
    const int lane = tid % WS, wi = tid / WS;
    const int64_t nb = C / 32;
    for (int64_t bq = wi; bq < Q * nb; bq += MIXW / WS) {
        const int qi = (int) (bq / nb);
        const int64_t b = bq % nb;
        if (!a.nw[qi] || !a.xq[qi]) continue;
        quant32(rb[qi * C + b * 32 + lane], lane, xq_q(a.xq[qi]) + t * C + b * 32, xq_d(a.xq[qi], a.T, C) + t * nb + b);
    }
}

__global__ void conv_res_k(const float* x, float* state, const float* w, float* snap, const float* base, float* dst, float* out, int64_t T, int64_t C, float inv) {
    const int64_t c = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float h0 = state[c], h1 = state[C + c], h2 = state[2 * C + c];
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    if (snap) {
        snap[c] = h0;
        snap[C + c] = h1;
        snap[2 * C + c] = h2;
    }
    for (int64_t t = 0; t < T; ++t) {
        const float xc = x[t * C + c];
        if (snap) snap[(3 + t) * C + c] = xc;
        float s = 0.0f;
        s += h0 * w0;
        s += h1 * w1;
        s += h2 * w2;
        s += xc * w3;
        const float o = (xc + s) * inv;
        if (out) out[t * C + c] = o;
        dst[t * C + c] = base ? base[t * C + c] + o : o;
        h0 = h1;
        h1 = h2;
        h2 = xc;
    }
    state[c] = h0;
    state[C + c] = h1;
    state[2 * C + c] = h2;
}

// blocks: Hkv for k, Hkv for v (a thread a channel, its rows in order), Hq for q, Hq for r; 128 threads
__global__ void post_qkv_k(Ink2sQkv p) {
    __shared__ float rows[16 * 128];
    __shared__ float dloc[16 * 4];
    __shared__ float sh[32];
    const int64_t gi = blockIdx.x;
    const int tid = threadIdx.x;
    const int64_t T = p.T, KD = p.Hkv * 128;
    auto tau = [&](int64_t t) { return p.alpha > 0.0f ? p.alpha * logf(fmaxf((float) (p.p0 + t + 1) * (1.0f / p.floor), 1.0f)) + 1.0f : 1.0f; };
    // a BF16 row of C against a float row: the block's sum
    auto bdot = [&](const uint16_t* wr, const float* xr) {
        float acc = 0.0f;
        for (int64_t i = tid; i < p.C; i += 128) acc += bf16f(wr[i]) * xr[i];
        return block_sum(acc, sh);
    };
    if (gi < 2 * p.Hkv) {
        const bool isv = gi >= p.Hkv;
        const int64_t hh = isv ? gi - p.Hkv : gi, c = hh * 128 + tid;
        const float* x = isv ? p.vraw : p.kraw;
        const float* dyn = isv ? p.vdyn : p.kdyn;
        // the head's input-dependent taps, made here: [T][4] in shared memory
        const uint16_t* dw = isv ? p.vdw : p.kdw;
        if (dw) {
            const float* nx = isv ? p.n2 : p.n1;
            for (int64_t t = 0; t < T; ++t)
                for (int l = 0; l < 4; ++l) {
                    const float v = bdot(dw + (hh * 4 + l) * p.C, nx + t * p.C);
                    if (tid == 0) dloc[t * 4 + l] = v;
                }
            __syncthreads();
            dyn = dloc;
        }
        const int64_t dstride = dw ? 4 : p.Hkv * 4, doff = dw ? 0 : hh * 4;
        float* st = isv ? p.vstate : p.kstate;
        float* sn = isv ? p.vsnap : p.ksnap;
        const float* w = isv ? p.vw : p.kw;
        float h0 = st[c], h1 = st[KD + c], h2 = st[2 * KD + c];
        if (sn) {
            sn[c] = h0;
            sn[KD + c] = h1;
            sn[2 * KD + c] = h2;
        }
        for (int64_t t = 0; t < T; ++t) {
            const float xc = x[t * KD + c];
            if (sn) sn[(3 + t) * KD + c] = xc;
            float s = 0.0f;
            s += h0 * w[c * 4];
            s += h1 * w[c * 4 + 1];
            s += h2 * w[c * 4 + 2];
            s += xc * w[c * 4 + 3];
            if (dyn) {
                const float* dt = dyn + t * dstride + doff;
                s += dt[0] * xc;
                s += dt[1] * h2;
                s += dt[2] * h1;
                s += dt[3] * h0;
            }
            rows[t * 128 + tid] = (xc + s) * (dyn ? p.inv_dyn : p.inv);
            h0 = h1;
            h1 = h2;
            h2 = xc;
        }
        st[c] = h0;
        st[KD + c] = h1;
        st[2 * KD + c] = h2;
        __syncthreads();
        // each row: k clipped to an L2 norm of at most thr, v RMS-normed; into the caches (and the float rows)
        for (int64_t t = 0; t < T; ++t) {
            const float v = rows[t * 128 + tid];
            const float ss = block_sum(v * v, sh);
            float y;
            if (isv) {
                y = v * rsqrtf(ss / 128.0f + p.eps);
            } else {
                const float f = fminf(fmaxf(p.thr / (sqrtf(ss) + 1e-6f), 0.0f), 1.0f);
                y = v * f;
            }
            (isv ? p.vout : p.kout)[t * KD + c] = y;
            const int64_t slot = (p.p0 + t) % p.cap;
            if (p.kvq) {
                // a q8 row (ns_ink2_kv_store_q8's): a warp a block of 32
                uint8_t* row = (uint8_t*) (isv ? p.vc : p.kc) + (hh * p.cap + slot) * 136;
                const float amax = warp_max(fabsf(y));
                const float dd = amax / 127.0f;
                const float id = dd != 0.0f ? 1.0f / dd : 0.0f;
                row[tid] = (uint8_t) (int8_t) roundf(y * id);
                if (tid % WS == 0) ((__half*) (row + 128))[tid / WS] = __float2half_rn(dd);
            } else {
                __half* cache = (__half*) (isv ? p.vc : p.kc);
                cache[(hh * p.cap + slot) * 128 + tid] = __float2half_rn(y);
            }
        }
        return;
    }
    // q (heads of 128) or r (heads of R), clipped after their scale, the full layers' log scale after
    const bool isr = gi >= 2 * p.Hkv + p.Hq;
    const int64_t hh = gi - 2 * p.Hkv - (isr ? p.Hq : 0);
    const int64_t D = isr ? p.R : 128;
    float* x = isr ? p.r : p.q;
    const float* scl = isr ? p.rscale : p.qscale;
    const float thr = isr ? p.rthr : p.thr;
    if (!isr && p.sinkw) {
        // the row's sink logit for this head
        for (int64_t t = 0; t < T; ++t) {
            const float v = bdot(p.sinkw + hh * p.C, p.n0 + t * p.C);
            if (tid == 0) p.sink[t * p.Hq + hh] = v;
        }
    }
    for (int64_t t = 0; t < T; ++t) {
        float* xr = x + (t * p.Hq + hh) * D;
        const float v = tid < D ? xr[tid] * scl[hh * D + tid] : 0.0f;
        const float ss = block_sum(v * v, sh);
        const float f = fminf(fmaxf(thr / (sqrtf(ss) + 1e-6f), 0.0f), 1.0f);
        if (tid < D) xr[tid] = p.alpha > 0.0f ? (v * f) * tau(t) : v * f;
    }
}

// a block a (row, head): its splits merged in order with the sink, gated, the head's four blocks quantized
__global__ void merge_k(const float* part, const float* sink, const float* gate, float* out, void* xq, int64_t T, int64_t Hq, int64_t ns) {
    constexpr int D = 128;
    const int64_t rh = blockIdx.x, t = rh / Hq, hq = rh % Hq, n = Hq * D;
    const int dd = threadIdx.x;
    const float* pp = part + rh * ns * (D + 2);
    const float sk = sink ? sink[rh] : -INFINITY;
    float M = sk;
    for (int64_t i = 0; i < ns; ++i) M = fmaxf(M, pp[i * (D + 2) + D]);
    float S = sink ? expf(sk - M) : 0.0f, A = 0.0f;
    for (int64_t i = 0; i < ns; ++i) {
        const float c = expf(pp[i * (D + 2) + D] - M);
        S += pp[i * (D + 2) + D + 1] * c;
        A += pp[i * (D + 2) + dd] * c;
    }
    const int64_t at = t * n + hq * D + dd;
    const float y = gate ? A / S * (1.0f / (1.0f + expf(-gate[at]))) : A / S;
    out[at] = y;
    // the warp's 32 dims are one block of 32
    const int wi = dd / WS, lane = dd % WS;
    quant32(y, lane, xq_q(xq) + t * n + hq * D + wi * 32, xq_d(xq, T, n) + t * (n / 32) + hq * 4 + wi);
}

// up to 4 controllers in one launch (the three attention inputs of a looped layer)
struct CtrlJobs {
    const float* cur[4];
    const float* msg[4];
    const float* down[4];
    const float* gain[4];
    const float* forget[4];
    const float* u[4];
    float* uo[4];
    float* out[4];
    int n;
};

constexpr int CDW = 256, CRPW = 2;

// a block a (controller, row, 2 of the rank's rows): z = [rms(cur) | rms(msg)] into shared memory once, then each rank
// row's dot over all the block's threads (16-byte loads), the sums added in a fixed order
__global__ void ctrl_down_k(CtrlJobs jb, int64_t T, int64_t C, int64_t rank, float eps, int64_t nblk) {
    __shared__ __align__(16) float z[8192];
    __shared__ float sl[CDW / WS * CRPW + 8];
    __shared__ float sh[32];
    int64_t gi = blockIdx.x;
    const int64_t r0 = (gi % nblk) * CRPW;
    gi /= nblk;
    const int64_t t = gi % T;
    const int k = (int) (gi / T);
    const float* down = jb.down[k];
    const int tid = threadIdx.x, lane = tid % WS, wi = tid / WS;
    const float4* a = (const float4*) (jb.cur[k] + t * C);
    const float4* b = (const float4*) (jb.msg[k] + t * C);
    const int64_t c4 = C / 4;
    float pa = 0.0f, pb = 0.0f;
    for (int64_t i = tid; i < c4; i += CDW) {
        const float4 va = a[i], vb = b[i];
        pa += va.x * va.x + va.y * va.y + va.z * va.z + va.w * va.w;
        pb += vb.x * vb.x + vb.y * vb.y + vb.z * vb.z + vb.w * vb.w;
        ((float4*) z)[i] = va;
        ((float4*) (z + C))[i] = vb;
    }
    const float ia = rsqrtf(block_sum(pa, sh) / (float) C + eps);
    const float ib = rsqrtf(block_sum(pb, sh) / (float) C + eps);
    __syncthreads();
    float acc[CRPW];
#pragma unroll
    for (int r = 0; r < CRPW; ++r) acc[r] = 0.0f;
    const int64_t n4 = 2 * C / 4;
    for (int64_t i = tid; i < n4; i += CDW) {
        const float4 zv = ((const float4*) z)[i];
        const float f = i < c4 ? ia : ib;
#pragma unroll
        for (int r = 0; r < CRPW; ++r) {
            if (r0 + r >= rank) break;
            const float4 dv = ((const float4*) (down + (r0 + r) * 2 * C))[i];
            acc[r] += (dv.x * zv.x + dv.y * zv.y + dv.z * zv.z + dv.w * zv.w) * f;
        }
    }
#pragma unroll
    for (int r = 0; r < CRPW; ++r) {
        const float v = warp_sum(acc[r]);
        if (lane == 0) sl[wi * CRPW + r] = v;
    }
    __syncthreads();
    if (tid < CRPW && r0 + tid < rank) {
        float v = 0.0f;
        for (int i = 0; i < CDW / WS; ++i) v += sl[i * CRPW + tid];
        jb.uo[k][t * rank + r0 + tid] = v;
    }
}

constexpr int CAW = 128, CPW = CAW / WS * 4;

// a warp 4 channels: their gain and forget rows against u (the lanes over the rank)
__global__ void ctrl_apply_k(CtrlJobs jb, int64_t T, int64_t C, int64_t rank, int64_t nblk) {
    const int lane = threadIdx.x % WS, wi = threadIdx.x / WS;
    int64_t gi = blockIdx.x;
    const int64_t c0 = (gi % nblk) * CPW + wi * 4;
    gi /= nblk;
    const int64_t t = gi % T;
    const int k = (int) (gi / T);
    const float* u = jb.u[k];
    const float* gain = jb.gain[k];
    const float* forget = jb.forget[k];
    float uv[2];
    for (int i = 0; i < 2; ++i) uv[i] = lane + i * WS < rank ? u[t * rank + lane + i * WS] : 0.0f;
    for (int kk = 0; kk < 4; ++kk) {
        const int64_t c = c0 + kk;
        if (c >= C) break;
        float gs = 0.0f, fs = 0.0f;
        for (int i = 0; i < 2; ++i) {
            const int64_t j = lane + i * WS;
            if (j < rank) {
                gs = fmaf(gain[c * rank + j], uv[i], gs);
                fs = fmaf(forget[c * rank + j], uv[i], fs);
            }
        }
        gs = warp_sum(gs);
        fs = warp_sum(fs);
        if (lane == 0) {
            const float x = jb.cur[k][t * C + c];
            jb.out[k][t * C + c] = (x - tanhf(fs) * x) + tanhf(gs) * jb.msg[k][t * C + c];
        }
    }
}

// the router's logits: a block a (row, 8 experts), a warp an expert's dot (lanes over the row, 16-byte loads)
__global__ void router_logits_k(const float* x, const float* w, float* logits, int64_t E, int64_t C, int64_t nwg) {
    const int lane = threadIdx.x % WS, wi = threadIdx.x / WS;
    const int64_t t = blockIdx.x / nwg, e = (blockIdx.x % nwg) * 8 + wi;
    const float4* xr = (const float4*) (x + t * C);
    const float4* wr = (const float4*) (w + e * C);
    float acc = 0.0f;
    for (int64_t i = lane; i < C / 4; i += WS) {
        const float4 a = __ldg(xr + i), b = __ldg(wr + i);
        acc = fmaf(a.x, b.x, acc);
        acc = fmaf(a.y, b.y, acc);
        acc = fmaf(a.z, b.z, acc);
        acc = fmaf(a.w, b.w, acc);
    }
    acc = warp_sum(acc);
    if (lane == 0) logits[t * E + e] = acc;
}

// the choice, a thread an expert: sigmoid, p + bias; an expert's rank the count of those larger (the lower index first
// among equals); the K first by rank, their weights p / (sum in rank order + 1e-20) x scale
__global__ void router_choice_k(const float* logits, const float* bias, int32_t* ids, float* wts, int64_t E, int64_t K, float scale) {
    __shared__ float vv[256], selp[16];
    __shared__ int seli[16];
    const int64_t t = blockIdx.x;
    const int e2 = threadIdx.x;
    float pe = 0.0f, ve = -INFINITY;
    if (e2 < E) {
        pe = 1.0f / (1.0f + expf(-logits[t * E + e2]));
        ve = bias ? pe + bias[e2] : pe;
    }
    vv[e2] = ve;
    __syncthreads();
    if (e2 < E) {
        int rank = 0;
        for (int j = 0; j < E; ++j) rank += vv[j] > ve || (vv[j] == ve && j < e2);
        if (rank < K) {
            seli[rank] = e2;
            selp[rank] = pe;
        }
    }
    __syncthreads();
    if (e2 < K) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) sum += selp[k];
        sum += 1e-20f;
        ids[t * K + e2] = seli[e2];
        wts[t * K + e2] = selp[e2] / sum * scale;
    }
}

}  // namespace

extern "C" {

int ns_ink2s_mix_norm(ns_gpu* g, const float* const* states, int64_t ns, const float* query, int64_t nq, const float* const* norm_w, float* out,
                      void* const* xq, int64_t T, int64_t C, float eps_mix, float eps_norm) {
    NS_TRY
    if (ns < 1 || ns > MAXS || nq < 1 || nq > MAXQ || C > MAXC || C % 128) return ns_fail("ns_ink2s_mix_norm: 1..8 states, 1..4 queries, rows of a multiple of 128 to 2304");
    if (T <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    MixArgs a{};
    for (int i = 0; i < ns; ++i) a.st[i] = states[i];
    for (int i = 0; i < nq; ++i) {
        a.nw[i] = norm_w ? norm_w[i] : nullptr;
        a.xq[i] = xq ? xq[i] : nullptr;
    }
    a.ns = (int) ns;
    a.nq = (int) nq;
    a.query = query;
    a.out = out;
    a.T = T;
    a.C = C;
    a.eps_mix = eps_mix;
    a.eps_norm = eps_norm;
    // the kernel sized to the call: states and queries as template arguments
    switch (ns * 8 + nq) {
#define MIXCASE(S_, Q_) case S_ * 8 + Q_: mix_k<S_, Q_><<<(unsigned) T, MIXW, 0, g->q>>>(a); break;
        MIXCASE(1, 1) MIXCASE(2, 1) MIXCASE(3, 1) MIXCASE(4, 1) MIXCASE(5, 1) MIXCASE(6, 1) MIXCASE(7, 1) MIXCASE(8, 1)
        MIXCASE(1, 3) MIXCASE(2, 3) MIXCASE(3, 3) MIXCASE(4, 3) MIXCASE(5, 3) MIXCASE(6, 3) MIXCASE(7, 3) MIXCASE(8, 3)
        MIXCASE(1, 4) MIXCASE(2, 4) MIXCASE(3, 4) MIXCASE(4, 4) MIXCASE(5, 4) MIXCASE(6, 4) MIXCASE(7, 4) MIXCASE(8, 4)
        MIXCASE(1, 2) MIXCASE(2, 2) MIXCASE(3, 2) MIXCASE(4, 2) MIXCASE(5, 2) MIXCASE(6, 2) MIXCASE(7, 2) MIXCASE(8, 2)
#undef MIXCASE
        default: return ns_fail("ns_ink2s_mix_norm: no kernel for this many states and queries");
    }
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_ink2s_conv_res(ns_gpu* g, const float* x, float* state, const float* w, float* snap, const float* base, float* dst, float* out, int64_t T, int64_t C,
                      float inv) {
    NS_TRY
    if (T < 1 || T > 16) return ns_fail("ns_ink2s_conv_res: 1..16 rows");
    NS_CU(cudaSetDevice(g->dev));
    conv_res_k<<<blocks(C), 256, 0, g->q>>>(x, state, w, snap, base, dst, out, T, C, inv);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_ink2s_post_qkv(ns_gpu* g, const Ink2sQkv* pp) {
    NS_TRY
    const Ink2sQkv p = *pp;
    if (p.T < 1 || p.T > 16 || p.D != 128 || p.R > 128) return ns_fail("ns_ink2s_post_qkv: 1..16 rows, heads of 128");
    NS_CU(cudaSetDevice(g->dev));
    post_qkv_k<<<(unsigned) (2 * p.Hkv + 2 * p.Hq), 128, 0, g->q>>>(p);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_ink2s_merge_gate_xq(ns_gpu* g, const float* part, const float* sink, const float* gate, float* out, void* xq, int64_t T, int64_t Hq, int64_t ns) {
    NS_TRY
    if (T <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    merge_k<<<(unsigned) (T * Hq), 128, 0, g->q>>>(part, sink, gate, out, xq, T, Hq, ns);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_ink2s_ctrl_down_many(ns_gpu* g, int n, const float* const* cur, const float* const* msg, const float* const* down, float* const* u, int64_t T,
                            int64_t C, int64_t rank, float eps) {
    NS_TRY
    if (T <= 0) return 0;
    if (n < 1 || n > 4 || C % 4 || 2 * C > 8192) return ns_fail("ns_ink2s_ctrl_down_many: 1..4 controllers, rows of a multiple of 4, at most 4096");
    NS_CU(cudaSetDevice(g->dev));
    CtrlJobs jb{};
    for (int k = 0; k < n; ++k) {
        jb.cur[k] = cur[k];
        jb.msg[k] = msg[k];
        jb.down[k] = down[k];
        jb.uo[k] = u[k];
    }
    jb.n = n;
    const int64_t nblk = (rank + CRPW - 1) / CRPW;
    ctrl_down_k<<<(unsigned) (n * T * nblk), CDW, 0, g->q>>>(jb, T, C, rank, eps, nblk);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_ink2s_ctrl_down(ns_gpu* g, const float* cur, const float* msg, const float* down, float* u, int64_t T, int64_t C, int64_t rank, float eps) {
    return ns_ink2s_ctrl_down_many(g, 1, &cur, &msg, &down, &u, T, C, rank, eps);
}

int ns_ink2s_ctrl_apply_many(ns_gpu* g, int n, const float* const* cur, const float* const* msg, const float* const* u, const float* const* gain,
                             const float* const* forget, float* const* out, int64_t T, int64_t C, int64_t rank) {
    NS_TRY
    if (T <= 0) return 0;
    if (n < 1 || n > 4 || rank > 64) return ns_fail("ns_ink2s_ctrl_apply_many: 1..4 controllers, a rank of at most 64");
    NS_CU(cudaSetDevice(g->dev));
    CtrlJobs jb{};
    for (int k = 0; k < n; ++k) {
        jb.cur[k] = cur[k];
        jb.msg[k] = msg[k];
        jb.u[k] = u[k];
        jb.gain[k] = gain[k];
        jb.forget[k] = forget[k];
        jb.out[k] = out[k];
    }
    jb.n = n;
    const int64_t nblk = (C + CPW - 1) / CPW;
    ctrl_apply_k<<<(unsigned) (n * T * nblk), CAW, 0, g->q>>>(jb, T, C, rank, nblk);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_ink2s_ctrl_apply(ns_gpu* g, const float* cur, const float* msg, const float* u, const float* gain, const float* forget, float* out, int64_t T, int64_t C,
                        int64_t rank) {
    return ns_ink2s_ctrl_apply_many(g, 1, &cur, &msg, &u, &gain, &forget, &out, T, C, rank);
}

int ns_ink2s_router(ns_gpu* g, const float* x, const float* w, const float* bias, float* logits, int32_t* ids, float* wts, int64_t T, int64_t E,
                    int64_t C, int64_t K, float scale) {
    NS_TRY
    if (T <= 0) return 0;
    if (E > 256 || E % 8 || K > 16 || C % 4) return ns_fail("ns_ink2s_router: at most 256 experts (a multiple of 8), 16 chosen, rows of a multiple of 4");
    NS_CU(cudaSetDevice(g->dev));
    const int64_t nwg = E / 8;
    router_logits_k<<<(unsigned) (T * nwg), 8 * WS, 0, g->q>>>(x, w, logits, E, C, nwg);
    NS_LAUNCH_CHECK();
    router_choice_k<<<(unsigned) T, 256, 0, g->q>>>(logits, bias, ids, wts, E, K, scale);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

}  // extern "C"
