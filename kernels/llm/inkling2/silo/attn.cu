// attn.cu: the inkling2 silo's decode attention (silo.h: ns_ink2s_attn_part, _part_block, _splits) - the keys split at
// absolute positions (64 a split: more blocks at a short context), without the merge (the merge, the sink, the output
// gate and the quantization are ns_ink2s_merge_gate_xq). nextsycl's attn.cpp written for CUDA: a block a (row, key
// head, split), a warp a query head of the three; the split's keys and values in shared memory; the logits a lane a key
// (two a lane), the split's softmax at once, then the values with the lanes over the dims. A row's splits are the same
// whatever rows share its pass.
//
// A prompt pass's attention is pattn.cu's.
#include "common.cuh"

using namespace ink2s;

namespace {

constexpr int D = 128, KB = 64, G = 3, DL = D / WS, PK = KB / WS;

void range(int64_t T, int64_t p0, int64_t win, int64_t& s0, int64_t& ns) {
    const int64_t lo = win > 0 ? (p0 + 1 - win > 0 ? p0 + 1 - win : 0) : 0, hi = p0 + T;
    s0 = lo / KB;
    ns = (hi - 1) / KB - s0 + 1;
}

__global__ void attn_part_k(const float* qp, const __half* kc, const __half* vc, const float* r, const float* relb, int64_t E, int64_t R, float* part,
                            int64_t T, int64_t Hq, int64_t Hkv, int64_t cap, int64_t p0, int64_t win, float scale, int64_t s0, int64_t ns, bool block,
                            int kvq) {
    __shared__ __align__(16) __half ks[KB * D];
    __shared__ __align__(16) __half vs[KB * D];
    __shared__ float qs[G][D + 32 + KB];
    const int lane = threadIdx.x % WS, wi = threadIdx.x / WS;
    int64_t gi = blockIdx.x;
    const int64_t sp = gi % ns;
    gi /= ns;
    const int64_t hk = gi % Hkv, rr = gi / Hkv, hq = hk * G + wi;
    // a block (the drafter's): every row sees the window from the block's start and the whole block
    const int64_t pos = p0 + rr, wpos = block ? p0 : pos;
    const int64_t first = win > 0 ? (wpos + 1 - win > 0 ? wpos + 1 - win : 0) : 0, end = block ? p0 + T : pos + 1;
    const int64_t lo = first > (s0 + sp) * KB ? first : (s0 + sp) * KB, hi = end < (s0 + sp + 1) * KB ? end : (s0 + sp + 1) * KB;
    const int64_t nk = hi > lo ? hi - lo : 0;
    float* qx = qs[wi];
    float* rx = qx + D;
    float* px = rx + 32;
#pragma unroll
    for (int i = 0; i < DL; ++i) qx[lane * DL + i] = qp[(rr * Hq + hq) * D + lane * DL + i] * scale;
    rx[lane] = lane < R && r ? r[(rr * Hq + hq) * R + lane] : 0.0f;
    // the split's keys and values: a warp a key at a time, its lanes 4 dims each (8-byte loads)
    {
        const int64_t s0k = lo % cap;
        for (int jj = wi; jj < (int) nk; jj += G) {
            int64_t slot = s0k + jj;
            if (slot >= cap) slot -= cap;
            if (kvq) {
                // a q8 row: 128 int8, then 4 half scales (a block of 32): this lane's 4 values of block lane / 8
                const uint8_t* kr = (const uint8_t*) kc + (hk * cap + slot) * 136;
                const uint8_t* vr = (const uint8_t*) vc + (hk * cap + slot) * 136;
                const float dk = __half2float(((const __half*) (kr + 128))[lane / 8]), dv = __half2float(((const __half*) (vr + 128))[lane / 8]);
                const char4 a = ((const char4*) kr)[lane], b = ((const char4*) vr)[lane];
                __half2* ko = (__half2*) (ks + jj * D + lane * 4);
                __half2* vo = (__half2*) (vs + jj * D + lane * 4);
                ko[0] = __floats2half2_rn(a.x * dk, a.y * dk);
                ko[1] = __floats2half2_rn(a.z * dk, a.w * dk);
                vo[0] = __floats2half2_rn(b.x * dv, b.y * dv);
                vo[1] = __floats2half2_rn(b.z * dv, b.w * dv);
            } else {
                ((uint2*) (ks + jj * D))[lane] = ((const uint2*) (kc + (hk * cap + slot) * D))[lane];
                ((uint2*) (vs + jj * D))[lane] = ((const uint2*) (vc + (hk * cap + slot) * D))[lane];
            }
        }
    }
    __syncthreads();
    float lg[PK];
    float m = -INFINITY;
#pragma unroll
    for (int u = 0; u < PK; ++u) {
        const int jj = lane + u * WS;
        float v = -INFINITY;
        if (jj < nk) {
            float dot = 0.0f;
            const __half2* kr = (const __half2*) (ks + jj * D);
#pragma unroll 8
            for (int c = 0; c < D / 2; ++c) {
                const int ci = (c + lane) & (D / 2 - 1);   // lanes on different banks
                const float2 kv = __half22float2(kr[ci]);
                dot = fmaf(qx[2 * ci], kv.x, dot);
                dot = fmaf(qx[2 * ci + 1], kv.y, dot);
            }
            const int64_t d = pos - (lo + jj);
            if (d >= 0 && d < E) {
                for (int i = 0; i < R; ++i) dot = fmaf(rx[i], relb[d * R + i], dot);
            }
            v = dot;
        }
        lg[u] = v;
        m = fmaxf(m, v);
    }
    m = warp_max(m);
    float s = 0.0f;
#pragma unroll
    for (int u = 0; u < PK; ++u) {
        const float p = lane + u * WS < nk ? expf(lg[u] - m) : 0.0f;
        px[lane + u * WS] = p;
        s += p;
    }
    s = warp_sum(s);
    __syncwarp();
    float acc[DL];
#pragma unroll
    for (int i = 0; i < DL; ++i) acc[i] = 0.0f;
    for (int64_t jj = 0; jj < nk; ++jj) {
        const float p = px[jj];
        const uint2 raw = *(const uint2*) (vs + jj * D + lane * DL);
        const float2 a = __half22float2(*(const __half2*) &raw.x), b = __half22float2(*(const __half2*) &raw.y);
        acc[0] = fmaf(p, a.x, acc[0]);
        acc[1] = fmaf(p, a.y, acc[1]);
        acc[2] = fmaf(p, b.x, acc[2]);
        acc[3] = fmaf(p, b.y, acc[3]);
    }
    float* pp = part + ((rr * Hq + hq) * ns + sp) * (D + 2);
#pragma unroll
    for (int i = 0; i < DL; ++i) pp[lane * DL + i] = acc[i];
    if (lane == 0) {
        pp[D] = m;
        pp[D + 1] = s;
    }
}

int attn_part(ns_gpu* g, const float* qp, const void* kcv, const void* vcv, const float* r, const float* relb, int64_t E, int64_t R, float* part, int64_t T,
              int64_t Hq, int64_t Hkv, int64_t cap, int64_t p0, int64_t win, float scale, bool block, int kvq) {
    if (Hq != G * Hkv || R > WS || T < 1 || T > 16) return ns_fail("ns_ink2s_attn_part: 3 query heads a key head, a bias of at most 32, 1..16 rows");
    NS_CU(cudaSetDevice(g->dev));
    int64_t s0, ns;
    range(T, p0, win, s0, ns);
    attn_part_k<<<(unsigned) (T * Hkv * ns), G * WS, 0, g->q>>>(qp, (const __half*) kcv, (const __half*) vcv, r, relb, E, R, part, T, Hq, Hkv, cap, p0, win, scale, s0,
                                                            ns, block, kvq);
    NS_LAUNCH_CHECK();
    return 0;
}

}  // namespace

extern "C" {

int64_t ns_ink2s_attn_splits(int64_t T, int64_t p0, int64_t win) {
    int64_t s0, ns;
    range(T, p0, win, s0, ns);
    return ns;
}

int ns_ink2s_attn_part(ns_gpu* g, const float* qp, const void* kcv, const void* vcv, const float* r, const float* relb, int64_t E, int64_t R, float* part,
                       int64_t T, int64_t Hq, int64_t Hkv, int64_t cap, int64_t p0, int64_t win, float scale, int64_t kvq) {
    NS_TRY
    return attn_part(g, qp, kcv, vcv, r, relb, E, R, part, T, Hq, Hkv, cap, p0, win, scale, false, (int) kvq);
    NS_CATCH
}

int ns_ink2s_attn_part_block(ns_gpu* g, const float* q, const void* kc, const void* vc, float* part, int64_t T, int64_t Hq, int64_t Hkv, int64_t cap, int64_t p0,
                             int64_t win, float scale) {
    NS_TRY
    return attn_part(g, q, kc, vc, nullptr, nullptr, 0, 0, part, T, Hq, Hkv, cap, p0, win, scale, true, 0);
    NS_CATCH
}

}  // extern "C"
