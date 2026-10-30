// decode.cu: SkinTokens' decode step's attention over the beams' slot pool in the silo (silo.h). The engine's own kernel
// (skintokens.cu) gives a block to every (beam, query head) and a thread to every key: neighbouring threads read rows 4 KB
// apart, the grouped heads read the same keys twice, and the weighted values run on half the threads. Here a block takes a
// (beam, key / value head) with the query heads that share it; a warp reads a key row whole (a lane four floats of its 128:
// one contiguous 512 bytes) and reduces the dot products by shuffles; the weighted values run a thread a dimension over rows
// that are contiguous across the threads. nextsycl's silo kernel (sub-groups of 16 on the B70) with warps of 32.
#include "ns.h"
#include "ns_internal.hpp"
#include "silo.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace {
constexpr int NW = 4, WG = 32 * NW, D = 128, MAXL = 4096;
static_assert(WG == D, "the weighted values run a thread a dimension");

__device__ float block_max(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = red[0];
    for (int i = 1; i < NW; ++i) t = fmaxf(t, red[i]);
    __syncthreads();
    return t;
}

__device__ float block_sum(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = 0.0f;
    for (int i = 0; i < NW; ++i) t += red[i];
    __syncthreads();
    return t;
}

// dynamic shared memory: G * len scores, then len slots
template <int G>
__global__ void __launch_bounds__(WG) slots_k(const float* __restrict__ qv, int64_t q_row, const float* __restrict__ kpool, const float* __restrict__ vpool,
                                              const int32_t* __restrict__ slots, int64_t stride, int64_t len, int64_t Hkv, float scale, float* __restrict__ out) {
    extern __shared__ __align__(16) float smem[];
    __shared__ float red[NW];
    __shared__ float inv[G];
    float* sc = smem;
    int32_t* sl = reinterpret_cast<int32_t*>(smem + G * len);
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, l = threadIdx.x;
    const int64_t gi = blockIdx.x, b = gi / Hkv, hk = gi % Hkv;
    const int64_t H = Hkv * G, row = Hkv * D;
    const int32_t* st = slots + b * stride;
    for (int64_t t = l; t < len; t += WG) sl[t] = st[t];
    __syncthreads();
    // the scores: a warp a key, its lanes four values each
    float qr[G][4];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        const float4 a = *reinterpret_cast<const float4*>(qv + b * q_row + (hk * G + g) * D + lane * 4);
        qr[g][0] = a.x; qr[g][1] = a.y; qr[g][2] = a.z; qr[g][3] = a.w;
    }
    // four keys at a time: their loads are independent, so the memory latency overlaps
    constexpr int U = 4;
    for (int64_t t0 = w; t0 < len; t0 += (int64_t) NW * U) {
        float4 kk[U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int64_t t = t0 + (int64_t) u * NW;
            kk[u] = *reinterpret_cast<const float4*>(kpool + (int64_t) sl[t < len ? t : 0] * row + hk * D + lane * 4);
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int64_t t = t0 + (int64_t) u * NW;
#pragma unroll
            for (int g = 0; g < G; ++g) {
                float d = qr[g][0] * kk[u].x;
                d = fmaf(qr[g][1], kk[u].y, d);
                d = fmaf(qr[g][2], kk[u].z, d);
                d = fmaf(qr[g][3], kk[u].w, d);
                for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
                if (lane == 0 && t < len) sc[g * len + t] = d * scale;
            }
        }
    }
    __syncthreads();
    // the softmax of each head's scores
#pragma unroll
    for (int g = 0; g < G; ++g) {
        float mx = -INFINITY;
        for (int64_t t = l; t < len; t += WG) mx = fmaxf(mx, sc[g * len + t]);
        mx = block_max(mx, red);
        float sum = 0.0f;
        for (int64_t t = l; t < len; t += WG) {
            const float e = expf(sc[g * len + t] - mx);
            sc[g * len + t] = e;
            sum += e;
        }
        sum = block_sum(sum, red);
        if (l == 0) inv[g] = 1.0f / sum;
    }
    __syncthreads();
    // the weighted values: a thread a dimension (WG = 128 = D), the rows contiguous across the threads
    float acc[G];
#pragma unroll
    for (int g = 0; g < G; ++g) acc[g] = 0.0f;
    const float* vb = vpool + hk * D + l;
    int64_t t = 0;
    for (; t + 8 <= len; t += 8) {
        float v[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) v[u] = vb[(int64_t) sl[t + u] * row];
#pragma unroll
        for (int u = 0; u < 8; ++u)
#pragma unroll
            for (int g = 0; g < G; ++g) acc[g] = fmaf(sc[g * len + t + u], v[u], acc[g]);
    }
    for (; t < len; ++t) {
        const float v = vb[(int64_t) sl[t] * row];
#pragma unroll
        for (int g = 0; g < G; ++g) acc[g] = fmaf(sc[g * len + t], v, acc[g]);
    }
#pragma unroll
    for (int g = 0; g < G; ++g) out[(b * H + hk * G + g) * D + l] = acc[g] * inv[g];
}

template <int G>
int run(ns_gpu* g, const float* q, int64_t q_row, const float* kpool, const float* vpool, const int32_t* slots, int64_t stride, int64_t len, int64_t B,
        int64_t Hkv, float scale, float* out) {
    // the shared memory past 48 KB needs asking for, once a kernel and card (the most it can take: MAXL positions)
    static bool asked[64] = {};
    if (g->dev < 0 || g->dev >= 64 || !asked[g->dev]) {
        NS_CU(cudaFuncSetAttribute(slots_k<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) ((G + 1) * MAXL * 4)));
        if (g->dev >= 0 && g->dev < 64) asked[g->dev] = true;
    }
    const size_t sh = (size_t) (G + 1) * (size_t) len * 4;
    slots_k<G><<<(unsigned) (B * Hkv), WG, sh, g->q>>>(q, q_row, kpool, vpool, slots, stride, len, Hkv, scale, out);
    NS_LAUNCH_CHECK();
    return 0;
}
}  // namespace

extern "C" {

int ns_model3d_skintokens_silo_attn_slots_supported(int64_t D_, int64_t len, int64_t H, int64_t Hkv) {
    return D_ == D && len > 0 && len <= MAXL && Hkv > 0 && (H == Hkv || H == 2 * Hkv) ? 1 : 0;
}

int ns_model3d_skintokens_silo_attn_slots(ns_gpu* g, const float* q, int64_t q_row, const float* kpool, const float* vpool, const int32_t* slots, int64_t stride,
                                          int64_t len, int64_t B, int64_t H, int64_t Hkv, int64_t D_, float scale, float* out) {
    if (B <= 0 || len <= 0) return 0;
    if (!ns_model3d_skintokens_silo_attn_slots_supported(D_, len, H, Hkv)) return ns_fail("ns_model3d_skintokens_silo_attn_slots: shapes it does not take");
    if (((uintptr_t) q & 15) || q_row % 4 || ((uintptr_t) kpool & 15) || ((uintptr_t) vpool & 15))
        return ns_fail("ns_model3d_skintokens_silo_attn_slots: float4-aligned rows needed");
    NS_CU(cudaSetDevice(g->dev));
    return H == Hkv ? run<1>(g, q, q_row, kpool, vpool, slots, stride, len, B, Hkv, scale, out)
                    : run<2>(g, q, q_row, kpool, vpool, slots, stride, len, B, Hkv, scale, out);
}

}  // extern "C"
