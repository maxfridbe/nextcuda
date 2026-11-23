// draft.cu: the inkling2 silo's DSpark drafter steps (silo.h) - Inkling 2 Mini's 3-layer block drafter (its
// mtp.safetensors; SGLang's models/inkling2_mtp.py, models/dspark.py, kernels/ops/speculative/inkling2_mtp.py): the
// target's features kept a row, their per-layer mixes normed, the block's SwiGLU quantized, the q / k norms with NEOX
// rotation, and the Markov head's recurrent step and its bias on the logits. nextsycl's draft.cpp, written for CUDA.
#include "common.cuh"

using namespace ink2s;

namespace {

__global__ void feat_put_k(const float* src, float* feats, int64_t T, int64_t C, int64_t nf, int64_t k) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * C) return;
    const int64_t t = i / C, c = i % C;
    feats[(t * nf + k) * C + c] = rbf(src[t * C + c]);
}

__global__ void feat_norm_k(const float* feats, int64_t nf, const float* alpha, const float* w, float eps, float* y, void* xq, int64_t T, int64_t C) {
    __shared__ float rp[4096];
    __shared__ float sh[32];
    const int64_t t = blockIdx.x;
    const int tid = threadIdx.x;
    float ss = 0.0f;
    for (int64_t c = tid; c < C; c += blockDim.x) {
        float m = 0.0f;
        for (int64_t k = 0; k < nf; ++k) m += feats[(t * nf + k) * C + c] * alpha[k];
        rp[c] = m;
        ss += m * m;
    }
    const float inv = rsqrtf(block_sum(ss, sh) / (float) C + eps);
    for (int64_t c = tid; c < C; c += blockDim.x) {
        const float v = rp[c] * inv * w[c];
        rp[c] = v;
        if (y) y[t * C + c] = v;
    }
    __syncthreads();
    if (xq) {
        const int lane = tid % WS, wi = tid / WS, nw = blockDim.x / WS;
        for (int64_t b = wi; b < C / 32; b += nw) quant32(rp[b * 32 + lane], lane, xq_q(xq) + t * C + b * 32, xq_d(xq, T, C) + t * (C / 32) + b);
    }
}

// a warp a block of 32: silu(gate) up, quantized
__global__ void swiglu_xq_k(const float* gate, const float* up, void* xq, int64_t T, int64_t F) {
    const int64_t nb = T * (F / 32);
    const int64_t b = (int64_t) blockIdx.x * 4 + threadIdx.x / WS;
    const int lane = threadIdx.x % WS;
    if (b >= nb) return;
    const int64_t at = b * 32 + lane;
    const float gv = gate[at];
    const float v = gv / (1.0f + expf(-gv)) * up[at];
    const int64_t t = b / (F / 32), bb = b % (F / 32);
    quant32(v, lane, xq_q(xq) + t * F + bb * 32, xq_d(xq, T, F) + t * (F / 32) + bb);
}

__global__ void to_bf16_k(const float* x, uint16_t* y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = to_bf16(x[i]);
}

// [state | w1[token] | hidden | f_rnn], each rounded to BF16 (the reference's inputs are BF16)
__global__ void rnn_inputs_k(const float* state, const int32_t* tok, const uint16_t* w1, int64_t vocab, int64_t rank, const float* hidden, const float* f_rnn,
                             int64_t C, float* out) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 2 * rank + 2 * C) return;
    float v;
    if (i < rank) v = state[i];
    else if (i < 2 * rank) {
        // a token past w1's rows (never written: a stopped block) reads row 0
        const int64_t tk = tok[0] >= 0 && tok[0] < vocab ? tok[0] : 0;
        v = bf16f(w1[tk * rank + (i - rank)]);
    } else if (i < 2 * rank + C) v = hidden[i - 2 * rank];
    else v = f_rnn[i - 2 * rank - C];
    out[i] = rbf(v);
}

// the projection (+ bias) rounded to BF16, then each op rounded as the reference's eager BF16 ops: gate = sigmoid(g),
// state' = gate state + (1 - gate) tanh(cand), latent = tanh(out)
__global__ void rnn_gates_k(const float* proj, const float* bias, float* state, float* latent, int64_t rank) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rank) return;
    const float gr = rbf(proj[i] + bias[i]), cr = rbf(proj[rank + i] + bias[rank + i]), orw = rbf(proj[2 * rank + i] + bias[2 * rank + i]);
    const float gate = rbf(1.0f / (1.0f + expf(-gr)));
    const float kept = rbf(gate * state[i]);
    const float blend = rbf(rbf(1.0f - gate) * rbf(tanhf(cr)));
    state[i] = rbf(kept + blend);
    latent[i] = rbf(tanhf(orw));
}

constexpr int BW = 1024;

// the step's logits: softcap(base + BF16(fold_b + bias)) over the first n; the largest (the lowest index of equals) and
// its probability (softmax at temperature 1: the online max and sum of exponentials, merged in a fixed order)
__global__ void bias_argmax_k(const float* base, const float* bias, const float* fold_b, int64_t n, float cap, float* logits, int32_t* out, float* prob,
                              int32_t* stop, float pmin) {
    __shared__ float bv[BW / WS], bs[BW / WS];
    __shared__ int bi[BW / WS];
    const int tid = threadIdx.x;
    // a block already ended (an earlier draft below p_min): nothing to do but leave a valid token for the next step's
    // Markov input (it indexes w1 by it)
    if (stop && stop[0]) {
        if (tid == 0) out[0] = 0;
        return;
    }
    float best = -INFINITY, se = 0.0f;
    int at = 0x7FFFFFFF;
    for (int64_t i = tid; i < n; i += BW) {
        float v = base[i] + rbf(fold_b[i] + bias[i]);
        if (cap > 0.0f) v = cap * tanhf(v / cap);
        if (logits) logits[i] = v;
        if (v > best) {
            se = se * expf(best - v) + 1.0f;
            best = v;
            at = (int) i;
        } else {
            se += expf(v - best);
        }
    }
    const float m = warp_max(best);
    int a = best == m ? at : 0x7FFFFFFF;
    for (int o = 16; o > 0; o >>= 1) a = min(a, __shfl_xor_sync(0xffffffffu, a, o));
    const float sm = warp_sum(best > -INFINITY ? se * expf(best - m) : 0.0f);
    if (tid % WS == 0) {
        bv[tid / WS] = m;
        bs[tid / WS] = sm;
        bi[tid / WS] = a;
    }
    __syncthreads();
    if (tid == 0) {
        float m2 = -INFINITY;
        int a2 = 0x7FFFFFFF;
        for (int s = 0; s < BW / WS; ++s)
            if (bv[s] > m2 || (bv[s] == m2 && bi[s] < a2)) {
                m2 = bv[s];
                a2 = bi[s];
            }
        float st = 0.0f;
        for (int s = 0; s < BW / WS; ++s)
            if (bv[s] > -INFINITY) st += bs[s] * expf(bv[s] - m2);
        out[0] = a2;
        if (prob) prob[0] = 1.0f / st;
        if (stop && 1.0f / st < pmin) stop[0] = 1;
    }
}

// a warp a (row, head): RMS-normed with w [D] (null: not normed), then NEOX-rotated (pairs i, i + D / 2) at position
// p0 + t, angle pos theta^(-2i / D)
__global__ void qk_norm_rope_k(float* x, int64_t H, const float* w, float eps, float theta, int64_t p0) {
    constexpr int D = 128, DL = D / WS;
    const int lane = threadIdx.x;
    const int64_t rh = blockIdx.x, t = rh / H;
    float* xr = x + rh * D;
    float v[DL];
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < DL; ++i) {
        v[i] = xr[lane + i * WS];
        ss += v[i] * v[i];
    }
    const float inv = w ? rsqrtf(warp_sum(ss) / (float) D + eps) : 1.0f;
#pragma unroll
    for (int i = 0; i < DL; ++i) v[i] = w ? v[i] * inv * w[lane + i * WS] : v[i];
    const float pos = (float) (p0 + t);
#pragma unroll
    for (int i = 0; i < DL / 2; ++i) {
        const int d = lane + i * WS;
        const float ang = pos * powf(theta, -2.0f * (float) d / (float) D);
        const float c = cosf(ang), s = sinf(ang);
        const float a = v[i], b = v[i + DL / 2];
        v[i] = a * c - b * s;
        v[i + DL / 2] = b * c + a * s;
    }
#pragma unroll
    for (int i = 0; i < DL; ++i) xr[lane + i * WS] = v[i];
}

// y [rows] = w [rows, n] (BF16) . x [n]: a warp a row, its lanes over the row in a fixed order; nothing when the block
// already ended (stop[0])
__global__ void fold_mv_k(const uint16_t* w, const float* x, float* y, int64_t rows, int64_t n, const int32_t* stop) {
    if (stop && stop[0]) return;
    const int lane = threadIdx.x % WS;
    const int64_t r = (int64_t) blockIdx.x * 8 + threadIdx.x / WS;
    if (r >= rows) return;
    const uint16_t* wr = w + r * n;
    float acc = 0.0f;
    for (int64_t i = lane * 8; i < n; i += WS * 8) {
        const uint4 u = __ldg((const uint4*) (wr + i));
        const uint32_t uu[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            acc = fmaf(__uint_as_float(uu[k] << 16), x[i + 2 * k], acc);
            acc = fmaf(__uint_as_float(uu[k] & 0xFFFF0000u), x[i + 2 * k + 1], acc);
        }
    }
    acc = warp_sum(acc);
    if (lane == 0) y[r] = acc;
}

}  // namespace

extern "C" {

int ns_ink2s_feat_put(ns_gpu* g, const float* src, float* feats, int64_t T, int64_t C, int64_t nf, int64_t k) {
    NS_TRY
    if (T <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    feat_put_k<<<blocks(T * C), 256, 0, g->q>>>(src, feats, T, C, nf, k);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_ink2s_feat_norm(ns_gpu* g, const float* feats, int64_t nf, const float* alpha, const float* w, float eps, float* y, void* xq, int64_t T, int64_t C) {
    NS_TRY
    if (T <= 0) return 0;
    if (C % 32 || C > 4096 || nf > 8) return ns_fail("ns_ink2s_feat_norm: rows of whole blocks of 32, at most 4096; at most 8 features");
    NS_CU(cudaSetDevice(g->dev));
    feat_norm_k<<<(unsigned) T, 256, 0, g->q>>>(feats, nf, alpha, w, eps, y, xq, T, C);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_ink2s_swiglu_xq(ns_gpu* g, const float* gate, const float* up, void* xq, int64_t T, int64_t F) {
    NS_TRY
    if (T <= 0) return 0;
    if (F % 32) return ns_fail("ns_ink2s_swiglu_xq: rows of whole blocks of 32");
    NS_CU(cudaSetDevice(g->dev));
    const int64_t nb = T * (F / 32);
    swiglu_xq_k<<<(unsigned) ((nb + 3) / 4), 4 * WS, 0, g->q>>>(gate, up, xq, T, F);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_ink2s_to_bf16(ns_gpu* g, const float* x, uint16_t* y, int64_t n) {
    NS_TRY
    if (n <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    to_bf16_k<<<blocks(n), 256, 0, g->q>>>(x, y, n);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_ink2s_rnn_inputs(ns_gpu* g, const float* state, const int32_t* tok, const uint16_t* w1, int64_t vocab, int64_t rank, const float* hidden,
                        const float* f_rnn, int64_t C, float* out) {
    NS_TRY
    NS_CU(cudaSetDevice(g->dev));
    rnn_inputs_k<<<blocks(2 * rank + 2 * C), 256, 0, g->q>>>(state, tok, w1, vocab, rank, hidden, f_rnn, C, out);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_ink2s_rnn_gates(ns_gpu* g, const float* proj, const float* bias, float* state, float* latent, int64_t rank) {
    NS_TRY
    NS_CU(cudaSetDevice(g->dev));
    rnn_gates_k<<<blocks(rank), 256, 0, g->q>>>(proj, bias, state, latent, rank);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_ink2s_bias_argmax(ns_gpu* g, const float* base, const float* bias, const float* fold_b, int64_t n, float cap, float* logits, int32_t* out, float* prob,
                         int32_t* stop, float pmin) {
    NS_TRY
    NS_CU(cudaSetDevice(g->dev));
    bias_argmax_k<<<1, BW, 0, g->q>>>(base, bias, fold_b, n, cap, logits, out, prob, stop, pmin);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_ink2s_qk_norm_rope(ns_gpu* g, float* x, int64_t T, int64_t H, const float* w, float eps, float theta, int64_t p0) {
    NS_TRY
    if (T <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    qk_norm_rope_k<<<(unsigned) (T * H), WS, 0, g->q>>>(x, H, w, eps, theta, p0);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_ink2s_fold_mv(ns_gpu* g, const uint16_t* w, const float* x, float* y, int64_t rows, int64_t n, const int32_t* stop) {
    NS_TRY
    if (rows <= 0) return 0;
    if (n % 256) return ns_fail("ns_ink2s_fold_mv: rows of a multiple of 256");
    NS_CU(cudaSetDevice(g->dev));
    fold_mv_k<<<(unsigned) ((rows + 7) / 8), 8 * WS, 0, g->q>>>(w, x, y, rows, n, stop);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

}  // extern "C"
