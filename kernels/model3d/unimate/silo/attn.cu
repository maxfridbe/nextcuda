// attn.cu: UniMate's attention in the silo (silo.h). The engine's own kernel (unimate.cu) is nextsycl's: a thread a
// (group, head, query) with its 128 scores and 128 outputs in local memory, every thread reading its keys and values from
// global memory on its own - 38% of a flow on the 5070 (4.4 of 11.8 s), the spatial attention (61 frames x 8 heads, 71
// joints over 71) and the temporal (71 joints x 8 heads, 61 frames over 61) of every block of every evaluation. Here a block
// takes a (group, head): its keys and values, at most 128 rows of D, go to shared memory once, and a thread a query walks
// them in tiles of 32 (flash attention's online softmax, the scores in registers) - every thread reads the same key word,
// a broadcast.
#include "ns.h"
#include "ns_internal.hpp"
#include "silo.h"

#include <cmath>
#include <cstdint>

namespace {
constexpr int MAXS = 128, TK = 32;

template <int D>
__global__ void __launch_bounds__(MAXS) attn_k(const float* __restrict__ q, int64_t q_row, const float* __restrict__ k, int64_t k_row,
                                               const float* __restrict__ v, int64_t v_row, int64_t Sq, int64_t Skv, int64_t H, int64_t q_gs,
                                               int64_t q_ss, int64_t kv_gs, int64_t kv_ss, int64_t nv, float scale, const float* __restrict__ bias,
                                               float* __restrict__ out, int64_t out_row) {
    // dynamic shared memory: nv rows of keys, then nv of values
    extern __shared__ float sm[];
    float (*ks)[D] = reinterpret_cast<float (*)[D]>(sm);
    float (*vs)[D] = reinterpret_cast<float (*)[D]>(sm + nv * D);
    const int64_t gi = blockIdx.x / H, h = blockIdx.x % H;
    for (int e = threadIdx.x; e < nv * D; e += blockDim.x) {
        const int j = e / D, c = e % D;
        const int64_t r = gi * kv_gs + j * kv_ss;
        ks[j][c] = k[r * k_row + h * D + c];
        vs[j][c] = v[r * v_row + h * D + c];
    }
    __syncthreads();
    const int64_t i = threadIdx.x;
    if (i >= Sq) return;
    float qv[D], acc[D];
    const float* qp = q + (gi * q_gs + i * q_ss) * q_row + h * D;
#pragma unroll
    for (int c = 0; c < D; ++c) { qv[c] = qp[c]; acc[c] = 0.0f; }
    const float* br = bias ? bias + (h * Sq + i) * Skv : nullptr;
    float m = -INFINITY, l = 0.0f;
    for (int j0 = 0; j0 < nv; j0 += TK) {
        float s[TK];
        float mt = m;
#pragma unroll
        for (int j = 0; j < TK; ++j) {
            float d = 0.0f;
            if (j0 + j < nv) {
#pragma unroll
                for (int c = 0; c < D; ++c) d += qv[c] * ks[j0 + j][c];
                d *= scale;
                if (br) d += br[j0 + j];
            }
            s[j] = j0 + j < nv ? d : -INFINITY;
            mt = fmaxf(mt, s[j]);
        }
        const float corr = expf(m - mt);  // m = -inf the first time: 0, nothing to rescale
        float ps = 0.0f;
#pragma unroll
        for (int j = 0; j < TK; ++j) {
            s[j] = expf(s[j] - mt);
            ps += s[j];
        }
        l = l * corr + ps;
        m = mt;
#pragma unroll
        for (int c = 0; c < D; ++c) acc[c] *= corr;
#pragma unroll
        for (int j = 0; j < TK; ++j) {
            if (j0 + j >= nv) break;
#pragma unroll
            for (int c = 0; c < D; ++c) acc[c] += s[j] * vs[j0 + j][c];
        }
    }
    const float inv = 1.0f / l;
    float* op = out + (gi * q_gs + i * q_ss) * out_row + h * D;
#pragma unroll
    for (int c = 0; c < D; ++c) op[c] = acc[c] * inv;
}

template <int D>
int launch(ns_gpu* g, const float* q, int64_t q_row, const float* k, int64_t k_row, const float* v, int64_t v_row, int64_t G, int64_t Sq, int64_t Skv,
           int64_t H, int64_t q_gs, int64_t q_ss, int64_t kv_gs, int64_t kv_ss, int64_t nv, float scale, const float* bias, float* out, int64_t out_row) {
    // past 48 KB of shared memory needs asking for, once a kernel and card (the most: 128 keys and values of D)
    static bool asked[64] = {};
    if (g->dev < 0 || g->dev >= 64 || !asked[g->dev]) {
        NS_CU(cudaFuncSetAttribute(attn_k<D>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) (2 * MAXS * D * 4)));
        if (g->dev >= 0 && g->dev < 64) asked[g->dev] = true;
    }
    const int threads = (int) ((Sq + 31) / 32 * 32);
    attn_k<D><<<(unsigned) (G * H), threads, (size_t) (2 * nv * D * 4), g->q>>>(q, q_row, k, k_row, v, v_row, Sq, Skv, H, q_gs, q_ss, kv_gs, kv_ss, nv, scale, bias, out, out_row);
    NS_LAUNCH_CHECK();
    return 0;
}
}  // namespace

extern "C" {

int ns_model3d_unimate_silo_attention_supported(int64_t D, int64_t Sq, int64_t Skv) {
    return (D == 32 || D == 64) && Sq > 0 && Sq <= MAXS && Skv > 0 && Skv <= MAXS ? 1 : 0;
}

int ns_model3d_unimate_silo_attention(ns_gpu* g, const float* q, int64_t q_row, const float* k, int64_t k_row, const float* v, int64_t v_row,
                                      int64_t G, int64_t Sq, int64_t Skv, int64_t H, int64_t D, int64_t q_gs, int64_t q_ss, int64_t kv_gs,
                                      int64_t kv_ss, int64_t nvalid, float scale, const float* bias, float* out, int64_t out_row) {
    if (!ns_model3d_unimate_silo_attention_supported(D, Sq, Skv)) return ns_fail("ns_model3d_unimate_silo_attention: shapes it does not take");
    if (G <= 0 || H <= 0) return 0;
    const int64_t nv = nvalid < Skv ? nvalid : Skv;
    if (nv <= 0) return ns_fail("ns_model3d_unimate_silo_attention: no valid key");
    NS_CU(cudaSetDevice(g->dev));
    return D == 32 ? launch<32>(g, q, q_row, k, k_row, v, v_row, G, Sq, Skv, H, q_gs, q_ss, kv_gs, kv_ss, nv, scale, bias, out, out_row)
                   : launch<64>(g, q, q_row, k, k_row, v, v_row, G, Sq, Skv, H, q_gs, q_ss, kv_gs, kv_ss, nv, scale, bias, out, out_row);
}

}  // extern "C"
