// attn.cu: YuE2's flow attention (head size 128, half in and out, 16 query heads over 8 key / value heads), built once
// a card. MOSS-Image's silo kernel (kernels/image/mossimage/silo/attn.cu: FlashAttention-2's scheme on mma.sync, the key
// and value tiles double-buffered by cp.async, their fragments by ldmatrix, the scores accumulated in half where a
// GeForce card runs that product at twice the float32 accumulation's rate) copied and given grouped-query heads: query
// head h reads key / value head h / G. The flow's rows (a song's frames, thousands) attend to every AR key of the chunk
// and to each other - most of a flow pass once its matrices are int8. YuE2's q and k are RMS-normed per head, so a
// score stays well inside half's range; the stage checks say what accumulating it in half costs.
#include "ns.h"
#include "ns_internal.hpp"
#include "silo.h"

#include <cuda_fp16.h>
#include <cstdint>

#ifndef NS_SILO_SM
#define NS_SILO_SM 0
#endif

namespace {

constexpr int D = 128;     // head size
constexpr int BK = 64;     // keys a tile
constexpr int LDK = 136;   // smem row stride in halves: 272 bytes, so ldmatrix's 8 rows land on distinct banks

__device__ __forceinline__ void mma16816(float c[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// the same product accumulated in half (GeForce cards run it at twice the rate of a float32 accumulation)
__device__ __forceinline__ void mma16816h(uint32_t c[2], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(c[0]), "+r"(c[1])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pack(float a, float b) {
    __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ void cp16(void* smem, const void* g, bool pred) {
    const unsigned s = (unsigned) __cvta_generic_to_shared(smem);
    const int n = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(g), "r"(n));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldsm4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* smem) {
    const unsigned s = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(s));
}
__device__ __forceinline__ void ldsm4t(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* smem) {
    const unsigned s = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(s));
}

// the online softmax of one tile's scores s (already in log2 units, masked), into o, m, l
__device__ __forceinline__ void online(float s[8][4], float o[16][4], float& m0, float& m1, float& l0, float& l1) {
    float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
    for (int nb = 0; nb < 8; ++nb) {
        mx0 = fmaxf(mx0, fmaxf(s[nb][0], s[nb][1]));
        mx1 = fmaxf(mx1, fmaxf(s[nb][2], s[nb][3]));
    }
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
    const float n0 = fmaxf(m0, mx0), n1 = fmaxf(m1, mx1);
    const float u0 = n0 == -INFINITY ? 0.0f : n0, u1 = n1 == -INFINITY ? 0.0f : n1;
    const float a0 = exp2f(m0 - u0), a1 = exp2f(m1 - u1);
    m0 = n0;
    m1 = n1;
    l0 *= a0;
    l1 *= a1;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        o[i][0] *= a0; o[i][1] *= a0;
        o[i][2] *= a1; o[i][3] *= a1;
    }
#pragma unroll
    for (int nb = 0; nb < 8; ++nb) {
        s[nb][0] = exp2f(s[nb][0] - u0);
        s[nb][1] = exp2f(s[nb][1] - u0);
        s[nb][2] = exp2f(s[nb][2] - u1);
        s[nb][3] = exp2f(s[nb][3] - u1);
        l0 += s[nb][0] + s[nb][1];
        l1 += s[nb][2] + s[nb][3];
    }
}

// O += P V for one tile, the values half in smem (Vs[key][dim]), P from the score accumulators
__device__ __forceinline__ void pv_half(float s[8][4], float o[16][4], const __half* Vs, int lane) {
    const int jj = lane >> 3, rr = lane & 7;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint32_t pa0 = pack(s[2 * j][0], s[2 * j][1]), pa1 = pack(s[2 * j][2], s[2 * j][3]);
        const uint32_t pa2 = pack(s[2 * j + 1][0], s[2 * j + 1][1]), pa3 = pack(s[2 * j + 1][2], s[2 * j + 1][3]);
#pragma unroll
        for (int nd2 = 0; nd2 < 8; ++nd2) {
            uint32_t r0, r1, r2, r3;
            ldsm4t(r0, r1, r2, r3, &Vs[(j * 16 + (jj & 1) * 8 + rr) * LDK + nd2 * 16 + (jj >> 1) * 8]);
            mma16816(o[2 * nd2], pa0, pa1, pa2, pa3, r0, r1);
            mma16816(o[2 * nd2 + 1], pa0, pa1, pa2, pa3, r2, r3);
        }
    }
}

// the normalized output of a warp's 16 rows
__device__ __forceinline__ void store(float o[16][4], float l0, float l1, __half* out, int r0, int r1, int Sq, int H, int h, int t) {
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    const float i0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, i1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
#pragma unroll
    for (int nd = 0; nd < 16; ++nd) {
        const int d = nd * 8 + 2 * t;
        if (r0 < Sq) *reinterpret_cast<uint32_t*>(out + ((long) r0 * H + h) * D + d) = pack(o[nd][0] * i0, o[nd][1] * i0);
        if (r1 < Sq) *reinterpret_cast<uint32_t*>(out + ((long) r1 * H + h) * D + d) = pack(o[nd][2] * i1, o[nd][3] * i1);
    }
}

template <int WARPS, bool QK16>
__global__ void __launch_bounds__(WARPS * 32) fa_half_k(const __half* __restrict__ q, const __half* __restrict__ k, const __half* __restrict__ v,
                                                        __half* __restrict__ out, int Sq, int Skv, int H, int G, long qs, long kvs, float scale_log2) {
    constexpr int BQ = WARPS * 16, NT = WARPS * 32;
    extern __shared__ __half smem[];
    __half* Ks = smem;                     // [2][BK][LDK]
    __half* Vs = smem + 2 * BK * LDK;      // [2][BK][LDK]
    const int h = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int qrow0 = blockIdx.x * BQ + warp * 16;
    const int r0 = qrow0 + g, r1 = r0 + 8;

    auto load = [&](int stage, int kt0) {
        for (int i = threadIdx.x; i < BK * D / 8; i += NT) {
            const int row = i / (D / 8), c = (i % (D / 8)) * 8;
            const int key = kt0 + row;
            const bool in = key < Skv;
            const long at = (long) (in ? key : 0) * kvs + (long) (h / G) * D + c;
            cp16(&Ks[(stage * BK + row) * LDK + c], k + at, in);
            cp16(&Vs[(stage * BK + row) * LDK + c], v + at, in);
        }
        cp_commit();
    };
    load(0, 0);

    uint32_t qa[8][4];
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
        const int d = kk * 16 + 2 * t;
        qa[kk][0] = r0 < Sq ? *reinterpret_cast<const uint32_t*>(q + r0 * qs + (long) h * D + d) : 0u;
        qa[kk][1] = r1 < Sq ? *reinterpret_cast<const uint32_t*>(q + r1 * qs + (long) h * D + d) : 0u;
        qa[kk][2] = r0 < Sq ? *reinterpret_cast<const uint32_t*>(q + r0 * qs + (long) h * D + d + 8) : 0u;
        qa[kk][3] = r1 < Sq ? *reinterpret_cast<const uint32_t*>(q + r1 * qs + (long) h * D + d + 8) : 0u;
    }
    float o[16][4];
#pragma unroll
    for (int i = 0; i < 16; ++i) { o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f; }
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;

    const int tiles = (Skv + BK - 1) / BK;
    const int jj = lane >> 3, rr = lane & 7;
    for (int it = 0; it < tiles; ++it) {
        if (it + 1 < tiles) {
            load((it + 1) & 1, (it + 1) * BK);
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();
        const __half* K = Ks + (it & 1) * BK * LDK;
        const __half* V = Vs + (it & 1) * BK * LDK;
        float s[8][4];
        if (QK16) {
            // the scores accumulated in half (q and k are normed: a score stays well inside half's range)
            uint32_t sh[8][2];
#pragma unroll
            for (int nb = 0; nb < 8; ++nb) { sh[nb][0] = sh[nb][1] = 0u; }
#pragma unroll
            for (int kk = 0; kk < 8; ++kk) {
#pragma unroll
                for (int nb2 = 0; nb2 < 4; ++nb2) {
                    uint32_t b0, b1, b2, b3;
                    ldsm4(b0, b1, b2, b3, &K[((nb2 * 2 + (jj >> 1)) * 8 + rr) * LDK + kk * 16 + (jj & 1) * 8]);
                    mma16816h(sh[2 * nb2], qa[kk][0], qa[kk][1], qa[kk][2], qa[kk][3], b0, b1);
                    mma16816h(sh[2 * nb2 + 1], qa[kk][0], qa[kk][1], qa[kk][2], qa[kk][3], b2, b3);
                }
            }
#pragma unroll
            for (int nb = 0; nb < 8; ++nb) {
                const float2 lo = __half22float2(*reinterpret_cast<const __half2*>(&sh[nb][0]));
                const float2 hi = __half22float2(*reinterpret_cast<const __half2*>(&sh[nb][1]));
                s[nb][0] = lo.x; s[nb][1] = lo.y; s[nb][2] = hi.x; s[nb][3] = hi.y;
            }
        } else {
#pragma unroll
            for (int nb = 0; nb < 8; ++nb) { s[nb][0] = s[nb][1] = s[nb][2] = s[nb][3] = 0.0f; }
#pragma unroll
            for (int kk = 0; kk < 8; ++kk) {
#pragma unroll
                for (int nb2 = 0; nb2 < 4; ++nb2) {
                    uint32_t b0, b1, b2, b3;
                    ldsm4(b0, b1, b2, b3, &K[((nb2 * 2 + (jj >> 1)) * 8 + rr) * LDK + kk * 16 + (jj & 1) * 8]);
                    mma16816(s[2 * nb2], qa[kk][0], qa[kk][1], qa[kk][2], qa[kk][3], b0, b1);
                    mma16816(s[2 * nb2 + 1], qa[kk][0], qa[kk][1], qa[kk][2], qa[kk][3], b2, b3);
                }
            }
        }
        const int kt0 = it * BK;
#pragma unroll
        for (int nb = 0; nb < 8; ++nb) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int col = kt0 + nb * 8 + 2 * t + (e & 1);
                s[nb][e] = col < Skv ? s[nb][e] * scale_log2 : -INFINITY;
            }
        }
        online(s, o, m0, m1, l0, l1);
        pv_half(s, o, V, lane);
        __syncthreads();
    }
    store(o, l0, l1, out, r0, r1, Sq, H, h, t);
}

}  // namespace

extern "C" {

int ns_audio_yue2_silo_info(int* sm) {
    *sm = NS_SILO_SM;
    return 0;
}

int ns_audio_yue2_silo_attention(ns_gpu* gp, const void* q, int64_t S, int64_t qs, const void* k, const void* v, int64_t Skv, int64_t kvs,
                                 int64_t H, int64_t Hkv, void* out, int warps, int half_scores) {
    if (qs % 8 || kvs % 8) return ns_fail("yue2 attention: rows of whole 16-byte groups (strides multiples of 8)");
    if (Hkv <= 0 || H % Hkv) return ns_fail("yue2 attention: the query heads a whole multiple of the key heads");
    NS_CU(cudaSetDevice(gp->dev));
    if (warps == 0) warps = 8;
    if (warps != 4 && warps != 8) return ns_fail("yue2 attention: 4 or 8 warps a block");
    const float scale_log2 = (1.0f / sqrtf((float) D)) * 1.4426950408889634f;
    const dim3 grid((unsigned) ((S + warps * 16 - 1) / (warps * 16)), (unsigned) H);
    const size_t smem = (size_t) 4 * BK * LDK * sizeof(__half);
    auto launch = [&](auto kern, int threads) -> int {
        NS_CU(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem));
        kern<<<grid, threads, smem, gp->q>>>((const __half*) q, (const __half*) k, (const __half*) v, (__half*) out, (int) S, (int) Skv, (int) H,
                                             (int) (H / Hkv), (long) qs, (long) kvs, scale_log2);
        NS_LAUNCH_CHECK();
        return 0;
    };
    if (warps == 4) return half_scores ? launch(fa_half_k<4, true>, 128) : launch(fa_half_k<4, false>, 128);
    return half_scores ? launch(fa_half_k<8, true>, 256) : launch(fa_half_k<8, false>, 256);
}

}  // extern "C"
