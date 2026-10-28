// attn.cu: SkinTokens' attention in float32 (silo.h), one pass. The engine's own kernel (skintokens.cu) gives a block to
// every query row, writes its scores to a scratch table and reads every key again for the values - slow on the skin VAE's
// decoder (54,000 samples, 12 heads of 64 over 388 keys, once a joint) and on the encoders' cross-attention (512 latents over
// the 54,000 samples). Here a thread owns a query row (its q, its running output, flash attention's online softmax in
// registers) and the block walks the keys a tile at a time through shared memory: every thread reads the same key word
// (a broadcast, no bank conflict), so a tile of TK keys is D FMAs a key for the scores and D more for the values, and the
// scores never leave the registers. nextsycl's silo kernel (sub-group broadcasts on the B70) in CUDA terms.
#include "ns.h"
#include "ns_internal.hpp"
#include "silo.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace flash {
// the most keys one pass takes: its running sums over very many drift (nextsycl measured the 54,000 keys of the encoders at
// +1e-6 relative RMS over the engine's own kernel)
constexpr int64_t MAX_KEYS = 65536;
constexpr int TK = 32;  // keys a tile

template <int D, int ROWS>
__global__ void __launch_bounds__(ROWS) attn_k(const float* __restrict__ Q, int64_t q_row, int64_t q_head, const float* __restrict__ K, int64_t k_row,
                                                int64_t k_head, const float* __restrict__ V, int64_t v_row, int64_t v_head, int64_t Sq, int64_t Skv,
                                                int64_t H, int64_t group, float scale, int causal, float* __restrict__ out, int64_t out_row) {
    __shared__ __align__(16) float ks[TK][D];
    __shared__ __align__(16) float vs[TK][D];
    const int64_t bh = blockIdx.y, b = bh / H, hh = bh % H, hk = hh / group;
    const int64_t r0 = (int64_t) blockIdx.x * ROWS;
    const int64_t row = r0 + threadIdx.x;
    const int64_t rr = row < Sq ? row : Sq - 1;
    float qv[D], acc[D];
    {
        const float* qp = Q + (b * Sq + rr) * q_row + hh * q_head;
#pragma unroll
        for (int d = 0; d < D; d += 4) {
            const float4 v = *reinterpret_cast<const float4*>(qp + d);
            qv[d] = v.x * scale; qv[d + 1] = v.y * scale; qv[d + 2] = v.z * scale; qv[d + 3] = v.w * scale;
        }
    }
#pragma unroll
    for (int d = 0; d < D; ++d) acc[d] = 0.0f;
    float m = -INFINITY, l = 0.0f;
    const int64_t last = r0 + ROWS - 1 < Sq ? r0 + ROWS - 1 : Sq - 1;
    const int64_t nk = causal ? last + 1 : Skv;
    const float* kb = K + b * Skv * k_row + hk * k_head;
    const float* vb = V + b * Skv * v_row + hk * v_head;
    for (int64_t j0 = 0; j0 < nk; j0 += TK) {
        __syncthreads();
        // the tile: TK rows of D, a float4 a thread at a time
        for (int e = threadIdx.x; e < TK * D / 4; e += ROWS) {
            const int j = e / (D / 4), d = (e % (D / 4)) * 4;
            const int64_t jj = j0 + j < Skv ? j0 + j : Skv - 1;
            *reinterpret_cast<float4*>(&ks[j][d]) = *reinterpret_cast<const float4*>(kb + jj * k_row + d);
            *reinterpret_cast<float4*>(&vs[j][d]) = *reinterpret_cast<const float4*>(vb + jj * v_row + d);
        }
        __syncthreads();
        float s[TK];
        float mt = m;
#pragma unroll
        for (int j = 0; j < TK; ++j) {
            float t = 0.0f;
#pragma unroll
            for (int d = 0; d < D; ++d) t = fmaf(qv[d], ks[j][d], t);
            const bool ok = j0 + j < Skv && (!causal || j0 + j <= row);
            s[j] = ok ? t : -INFINITY;
            mt = fmaxf(mt, s[j]);
        }
        // nothing seen yet (all masked so far): no rescale, no weights
        const float corr = mt == -INFINITY ? 1.0f : expf(m - mt);
        float ps = 0.0f;
#pragma unroll
        for (int j = 0; j < TK; ++j) {
            s[j] = mt == -INFINITY ? 0.0f : expf(s[j] - mt);
            ps += s[j];
        }
        l = l * corr + ps;
        m = mt;
#pragma unroll
        for (int d = 0; d < D; ++d) acc[d] *= corr;
#pragma unroll
        for (int j = 0; j < TK; ++j)
#pragma unroll
            for (int d = 0; d < D; ++d) acc[d] = fmaf(s[j], vs[j][d], acc[d]);
    }
    if (row >= Sq) return;
    const float inv = 1.0f / l;
    float* op = out + (b * Sq + row) * out_row + hh * D;
#pragma unroll
    for (int d = 0; d < D; d += 4) *reinterpret_cast<float4*>(op + d) = make_float4(acc[d] * inv, acc[d + 1] * inv, acc[d + 2] * inv, acc[d + 3] * inv);
}

// whether attn_k takes these: q / k / v / out rows and heads float4-aligned
inline bool takes(const void* q, int64_t q_row, int64_t q_head, const void* k, int64_t k_row, int64_t k_head, const void* v, int64_t v_row, int64_t v_head,
                  const void* out, int64_t out_row) {
    auto al = [](const void* p) { return ((uintptr_t) p & 15) == 0; };
    return al(q) && al(k) && al(v) && al(out) && q_row % 4 == 0 && q_head % 4 == 0 && k_row % 4 == 0 && k_head % 4 == 0 && v_row % 4 == 0 &&
           v_head % 4 == 0 && out_row % 4 == 0;
}

template <int D>
int launch(cudaStream_t st, const float* Q, int64_t q_row, int64_t q_head, const float* K, int64_t k_row, int64_t k_head, const float* V, int64_t v_row,
           int64_t v_head, int64_t B, int64_t Sq, int64_t Skv, int64_t H, int64_t Hkv, float scale, int causal, float* out, int64_t out_row) {
    // few query rows (the encoders' 512 latents): smaller blocks, more of them on the card
    constexpr int ROWS = 64;
    const dim3 grid((unsigned) ((Sq + ROWS - 1) / ROWS), (unsigned) (B * H));
    attn_k<D, ROWS><<<grid, ROWS, 0, st>>>(Q, q_row, q_head, K, k_row, k_head, V, v_row, v_head, Sq, Skv, H, H / Hkv, scale, causal, out, out_row);
    NS_LAUNCH_CHECK();
    return 0;
}
}  // namespace flash

extern "C" {

int ns_model3d_skintokens_silo_attention_supported(int64_t D, int64_t Dv, int64_t H, int64_t Hkv, int64_t Skv) {
    return D == Dv && (D == 16 || D == 32 || D == 48 || D == 64) && Hkv > 0 && H % Hkv == 0 && Skv > 0 && Skv <= flash::MAX_KEYS ? 1 : 0;
}

int ns_model3d_skintokens_silo_attention(ns_gpu* g, const float* q, int64_t q_row, int64_t q_head, const float* k, int64_t k_row, int64_t k_head,
                                         const float* v, int64_t v_row, int64_t v_head, int64_t B, int64_t Sq, int64_t Skv, int64_t H, int64_t Hkv,
                                         int64_t D, int64_t Dv, float scale, int causal, float* out, int64_t out_row) {
    if (B <= 0 || Sq <= 0 || Skv <= 0 || H <= 0) return 0;
    if (!ns_model3d_skintokens_silo_attention_supported(D, Dv, H, Hkv, Skv)) return ns_fail("ns_model3d_skintokens_silo_attention: shapes it does not take");
    if (causal && Sq != Skv) return ns_fail("ns_model3d_skintokens_silo_attention: a causal mask needs Sq = Skv");
    if (!flash::takes(q, q_row, q_head, k, k_row, k_head, v, v_row, v_head, out, out_row))
        return ns_fail("ns_model3d_skintokens_silo_attention: rows or heads not float4-aligned");
    NS_CU(cudaSetDevice(g->dev));
    switch (D) {
        case 16: return flash::launch<16>(g->q, q, q_row, q_head, k, k_row, k_head, v, v_row, v_head, B, Sq, Skv, H, Hkv, scale, causal, out, out_row);
        case 32: return flash::launch<32>(g->q, q, q_row, q_head, k, k_row, k_head, v, v_row, v_head, B, Sq, Skv, H, Hkv, scale, causal, out, out_row);
        case 48: return flash::launch<48>(g->q, q, q_row, q_head, k, k_row, k_head, v, v_row, v_head, B, Sq, Skv, H, Hkv, scale, causal, out, out_row);
        default: return flash::launch<64>(g->q, q, q_row, q_head, k, k_row, k_head, v, v_row, v_head, B, Sq, Skv, H, Hkv, scale, causal, out, out_row);
    }
}

}  // extern "C"
