// attn.cu: Qwen-Image 2.1's attention in its silo - fused kernels (FlashAttention-2's scheme on mma.sync m16n8k16: the
// scores of a block of queries against 64 keys never leave registers, the softmax is online, the output is scaled
// once at the end) for head size 128, half in and out. The shared kernels (nsd_attention_*) go through cuBLAS and a
// float32 score table; this is for the shapes of this model only.
//
//   flash_k   64 query rows a block, K and V copied through registers, float32 sums (the first kernel, mode 0)
//   flash2_k  128 query rows a block (8 warps), K and V tiles by cp.async a tile ahead into swizzled shared memory,
//             fragments by ldmatrix (V's by ldmatrix.trans); P V's sums in half a 32-key slice at a time (PV16), Q K^T's
//             a 32-feature slice at a time (QK16), each slice added to float32. On the model's own q, k, v (1024 x 1024,
//             blocks 0-31) against float64: float32 sums 2.2e-4 relative rms, PV16 2.5e-4, QK16 5e-4 - 1.3e-3; on the
//             5070 at 4,123 keys 4.69 / 3.90 / 3.25 ms (float32 / PV16 / both), 5.34 the first kernel.
//
// Grouped heads (Hq over Hkv), q rows of stride qs, k / v rows of stride kvs (elements), out [Sq, Hq * 128].
// `causal` = 1: query row r may read keys 0..row0 + r; 0: all Skv keys.
#include "ns.h"
#include "ns_internal.hpp"

#include <cuda_fp16.h>
#include <cfloat>
#include <cstdint>

namespace {

constexpr int D = 128;     // head size
constexpr int BQ = 64;     // query rows a block (16 a warp, 4 warps)
constexpr int BK = 64;     // keys a tile
constexpr int LDK = 136;   // smem row stride in halves: 272 bytes, so a quad's 8 rows land on distinct banks

__device__ __forceinline__ void mma16816(float c[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pack(float a, float b) {
    __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<uint32_t*>(&h);
}

__global__ void __launch_bounds__(128) flash_k(const __half* __restrict__ q, const __half* __restrict__ k, const __half* __restrict__ v,
                                               __half* __restrict__ out, int Sq, int Skv, int Hq, int Hkv, long qs, long kvs, int causal,
                                               long row0, float scale_log2) {
    extern __shared__ __half smem[];
    __half* Ks = smem;
    __half* Vs = smem + BK * LDK;
    const int h = blockIdx.y, kvh = h / (Hq / Hkv);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int qrow0 = blockIdx.x * BQ + warp * 16;
    const int r0 = qrow0 + g, r1 = r0 + 8;

    // this warp's 16 query rows as A fragments (k = the 128 features, 8 steps of 16), read once
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

    int kv_end = Skv;
    if (causal) {
        const long last = row0 + (long) blockIdx.x * BQ + BQ;   // one past the block's last row's limit
        kv_end = (int) min((long) Skv, last);
    }
    for (int kt0 = 0; kt0 < kv_end; kt0 += BK) {
        __syncthreads();
        // the tile's keys and values to shared memory, 16 bytes a thread a time
        for (int i = threadIdx.x; i < BK * D / 8; i += 128) {
            const int row = i / (D / 8), c = (i % (D / 8)) * 8;
            const int key = kt0 + row;
            uint4 kk4 = make_uint4(0, 0, 0, 0), vv4 = make_uint4(0, 0, 0, 0);
            if (key < Skv) {
                kk4 = *reinterpret_cast<const uint4*>(k + key * kvs + (long) kvh * D + c);
                vv4 = *reinterpret_cast<const uint4*>(v + key * kvs + (long) kvh * D + c);
            }
            *reinterpret_cast<uint4*>(&Ks[row * LDK + c]) = kk4;
            *reinterpret_cast<uint4*>(&Vs[row * LDK + c]) = vv4;
        }
        __syncthreads();

        // S = Q K^T for 64 keys: 8 blocks of 8 keys, k over the features
        float s[8][4];
#pragma unroll
        for (int nb = 0; nb < 8; ++nb) { s[nb][0] = s[nb][1] = s[nb][2] = s[nb][3] = 0.0f; }
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
#pragma unroll
            for (int nb = 0; nb < 8; ++nb) {
                const __half* kr = &Ks[(nb * 8 + g) * LDK + kk * 16 + 2 * t];
                mma16816(s[nb], qa[kk][0], qa[kk][1], qa[kk][2], qa[kk][3], *reinterpret_cast<const uint32_t*>(kr),
                         *reinterpret_cast<const uint32_t*>(kr + 8));
            }
        }
        // scale (into log2 units), mask, the block's row maxima
        float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
        for (int nb = 0; nb < 8; ++nb) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int col = kt0 + nb * 8 + 2 * t + (e & 1);
                const long row = (e >> 1) ? r1 : r0;
                float x = s[nb][e] * scale_log2;
                if (col >= Skv || (causal && col > row0 + row)) x = -INFINITY;
                s[nb][e] = x;
                if (e < 2) mx0 = fmaxf(mx0, x); else mx1 = fmaxf(mx1, x);
            }
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
        // O += P V: P's accumulator layout is the A layout of the next product, 16 keys a step
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t pa0 = pack(s[2 * j][0], s[2 * j][1]), pa1 = pack(s[2 * j][2], s[2 * j][3]);
            const uint32_t pa2 = pack(s[2 * j + 1][0], s[2 * j + 1][1]), pa3 = pack(s[2 * j + 1][2], s[2 * j + 1][3]);
#pragma unroll
            for (int nd = 0; nd < 16; ++nd) {
                const int d = nd * 8 + g;
                const __half* v0 = &Vs[(16 * j + 2 * t) * LDK + d];
                __half2 hb0 = __halves2half2(v0[0], v0[LDK]);
                __half2 hb1 = __halves2half2(v0[8 * LDK], v0[9 * LDK]);
                mma16816(o[nd], pa0, pa1, pa2, pa3, *reinterpret_cast<uint32_t*>(&hb0), *reinterpret_cast<uint32_t*>(&hb1));
            }
        }
    }
    // the row sums across the quad, then the output
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    const float i0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, i1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
#pragma unroll
    for (int nd = 0; nd < 16; ++nd) {
        const int d = nd * 8 + 2 * t;
        if (r0 < Sq) *reinterpret_cast<uint32_t*>(out + ((long) r0 * Hq + h) * D + d) = pack(o[nd][0] * i0, o[nd][1] * i0);
        if (r1 < Sq) *reinterpret_cast<uint32_t*>(out + ((long) r1 * Hq + h) * D + d) = pack(o[nd][2] * i1, o[nd][3] * i1);
    }
}


// ---- the second kernel: 128 query rows a block (8 warps), K and V tiles by cp.async two tiles ahead, ldmatrix fragments
// (V's transposed by ldmatrix.trans), swizzled shared memory; half sums optional for each product (QK16: the scores a
// 32-feature slice at a time; PV16: the output a 32-key slice at a time), each slice's sum added to float32 - the
// tensor cores' half rate is twice their float32 rate on a GeForce.

constexpr int BQ2 = 128, THREADS2 = 256;
constexpr int TILE2 = BK * D;                // halves of one K or V tile (64 keys x 128)
constexpr int SMEM2 = 4 * TILE2 * 2;         // K and V, two stages: 64 KiB

__device__ __forceinline__ uint32_t s_u32(const void* p) { return (uint32_t) __cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp16(uint32_t dst, const void* src, bool ok) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void ldsm4(uint32_t r[4], uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm4t(uint32_t r[4], uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void mma16816h(uint32_t c[2], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(c[0]), "+r"(c[1]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// (row, 16-byte chunk) of a 64 x 128 tile: rows of 256 bytes, the chunk's low 3 bits XORed with the row's
__device__ __forceinline__ uint32_t sw(int row, int chunk) { return (uint32_t) (row * 256 + ((chunk ^ (row & 7)) << 4)); }

template <bool QK16, bool PV16>
__global__ void __launch_bounds__(THREADS2) flash2_k(const __half* __restrict__ q, const __half* __restrict__ k, const __half* __restrict__ v,
                                                     __half* __restrict__ out, int Sq, int Skv, int Hq, int Hkv, long qs, long kvs, int causal,
                                                     long row0, float scale_log2) {
    extern __shared__ __align__(128) unsigned char sm2[];
    const uint32_t sb = s_u32(sm2);   // K stage 0, K stage 1, V stage 0, V stage 1
    const int h = blockIdx.y, kvh = h / (Hq / Hkv);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int r0 = blockIdx.x * BQ2 + warp * 16 + g, r1 = r0 + 8;

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
    for (int i = 0; i < 16; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;

    int kv_end = Skv;
    if (causal) kv_end = (int) min((long) Skv, row0 + (long) blockIdx.x * BQ2 + BQ2);
    const int tiles = (kv_end + BK - 1) / BK;

    // a tile's K and V rows into stage `st`: 64 rows x 16 chunks each, 4 + 4 chunks a thread
    auto load = [&](int st, int kt0) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int idx = tid + i * THREADS2, row = idx >> 4, c = idx & 15, key = kt0 + row;
            const bool ok = key < Skv;
            const long off = (long) (ok ? key : 0) * kvs + (long) kvh * D + c * 8;
            cp16(sb + st * TILE2 * 2 + sw(row, c), k + off, ok);
            cp16(sb + (2 + st) * TILE2 * 2 + sw(row, c), v + off, ok);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };
    if (tiles > 0) load(0, 0);
    for (int j = 0; j < tiles; ++j) {
        const int kt0 = j * BK, st = j & 1;
        if (j + 1 < tiles) {
            load(st ^ 1, kt0 + BK);
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();
        const uint32_t ks = sb + st * TILE2 * 2, vs = sb + (2 + st) * TILE2 * 2;

        // S = Q K^T: 8 blocks of 8 keys
        float s[8][4];
#pragma unroll
        for (int nb = 0; nb < 8; ++nb) s[nb][0] = s[nb][1] = s[nb][2] = s[nb][3] = 0.0f;
#pragma unroll
        for (int kp = 0; kp < 4; ++kp) {           // 32 features at a time
            uint32_t sh[8][2];
            if (QK16) {
#pragma unroll
                for (int nb = 0; nb < 8; ++nb) sh[nb][0] = sh[nb][1] = 0u;
            }
#pragma unroll
            for (int kh = 0; kh < 2; ++kh) {
                const int kk = kp * 2 + kh;
#pragma unroll
                for (int nb = 0; nb < 8; nb += 2) {
                    uint32_t b[4];
                    ldsm4(b, ks + sw(nb * 8 + (lane & 7) + ((lane >> 4) << 3), kk * 2 + ((lane >> 3) & 1)));
                    if (QK16) {
                        mma16816h(sh[nb], qa[kk][0], qa[kk][1], qa[kk][2], qa[kk][3], b[0], b[1]);
                        mma16816h(sh[nb + 1], qa[kk][0], qa[kk][1], qa[kk][2], qa[kk][3], b[2], b[3]);
                    } else {
                        mma16816(s[nb], qa[kk][0], qa[kk][1], qa[kk][2], qa[kk][3], b[0], b[1]);
                        mma16816(s[nb + 1], qa[kk][0], qa[kk][1], qa[kk][2], qa[kk][3], b[2], b[3]);
                    }
                }
            }
            if (QK16) {
#pragma unroll
                for (int nb = 0; nb < 8; ++nb) {
                    const float2 lo = __half22float2(*reinterpret_cast<__half2*>(&sh[nb][0]));
                    const float2 hi = __half22float2(*reinterpret_cast<__half2*>(&sh[nb][1]));
                    s[nb][0] += lo.x; s[nb][1] += lo.y; s[nb][2] += hi.x; s[nb][3] += hi.y;
                }
            }
        }
        // the tile's row maxima of the raw scores (the scale is positive: it commutes with max), masking only the
        // tiles that reach past the keys or the causal limit
        const bool edge = kt0 + BK > Skv || (causal && kt0 + BK - 1 > row0 + (long) blockIdx.x * BQ2);
        if (edge) {
#pragma unroll
            for (int nb = 0; nb < 8; ++nb)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int col = kt0 + nb * 8 + 2 * t + (e & 1);
                    const long row = (e >> 1) ? r1 : r0;
                    if (col >= Skv || (causal && col > row0 + row)) s[nb][e] = -INFINITY;
                }
        }
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
        // m: the running maxima in log2 units
        const float n0 = fmaxf(m0, mx0 * scale_log2), n1 = fmaxf(m1, mx1 * scale_log2);
        const float u0 = n0 == -INFINITY ? 0.0f : n0, u1 = n1 == -INFINITY ? 0.0f : n1;
        const float a0 = exp2f(m0 - u0), a1 = exp2f(m1 - u1);
        m0 = n0;
        m1 = n1;
        l0 *= a0;
        l1 *= a1;
        // the output rescaled only when a row's maximum moved (after the first tiles it rarely does)
        if (!__all_sync(0xffffffffu, a0 == 1.0f && a1 == 1.0f)) {
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                o[i][0] *= a0; o[i][1] *= a0;
                o[i][2] *= a1; o[i][3] *= a1;
            }
        }
#pragma unroll
        for (int nb = 0; nb < 8; ++nb) {
            s[nb][0] = exp2f(fmaf(s[nb][0], scale_log2, -u0));
            s[nb][1] = exp2f(fmaf(s[nb][1], scale_log2, -u0));
            s[nb][2] = exp2f(fmaf(s[nb][2], scale_log2, -u1));
            s[nb][3] = exp2f(fmaf(s[nb][3], scale_log2, -u1));
            l0 += s[nb][0] + s[nb][1];
            l1 += s[nb][2] + s[nb][3];
        }
        // O += P V, 16 keys a step (P's accumulator layout is the A layout), 32 keys a half slice
#pragma unroll
        for (int jp = 0; jp < 2; ++jp) {
            uint32_t oh[16][2];
            if (PV16) {
#pragma unroll
                for (int nd = 0; nd < 16; ++nd) oh[nd][0] = oh[nd][1] = 0u;
            }
#pragma unroll
            for (int jh = 0; jh < 2; ++jh) {
                const int jj = jp * 2 + jh;
                const uint32_t pa0 = pack(s[2 * jj][0], s[2 * jj][1]), pa1 = pack(s[2 * jj][2], s[2 * jj][3]);
                const uint32_t pa2 = pack(s[2 * jj + 1][0], s[2 * jj + 1][1]), pa3 = pack(s[2 * jj + 1][2], s[2 * jj + 1][3]);
#pragma unroll
                for (int nd = 0; nd < 16; nd += 2) {
                    uint32_t b[4];
                    ldsm4t(b, vs + sw(16 * jj + (lane & 7) + (((lane >> 3) & 1) << 3), nd + (lane >> 4)));
                    if (PV16) {
                        mma16816h(oh[nd], pa0, pa1, pa2, pa3, b[0], b[1]);
                        mma16816h(oh[nd + 1], pa0, pa1, pa2, pa3, b[2], b[3]);
                    } else {
                        mma16816(o[nd], pa0, pa1, pa2, pa3, b[0], b[1]);
                        mma16816(o[nd + 1], pa0, pa1, pa2, pa3, b[2], b[3]);
                    }
                }
            }
            if (PV16) {
#pragma unroll
                for (int nd = 0; nd < 16; ++nd) {
                    const float2 lo = __half22float2(*reinterpret_cast<__half2*>(&oh[nd][0]));
                    const float2 hi = __half22float2(*reinterpret_cast<__half2*>(&oh[nd][1]));
                    o[nd][0] += lo.x; o[nd][1] += lo.y; o[nd][2] += hi.x; o[nd][3] += hi.y;
                }
            }
        }
        // every warp done with this stage before the next iteration's copy writes over it
        __syncthreads();
    }
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    const float i0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, i1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
#pragma unroll
    for (int nd = 0; nd < 16; ++nd) {
        const int d = nd * 8 + 2 * t;
        if (r0 < Sq) *reinterpret_cast<uint32_t*>(out + ((long) r0 * Hq + h) * D + d) = pack(o[nd][0] * i0, o[nd][1] * i0);
        if (r1 < Sq) *reinterpret_cast<uint32_t*>(out + ((long) r1 * Hq + h) * D + d) = pack(o[nd][2] * i1, o[nd][3] * i1);
    }
}

template <bool QK16, bool PV16>
int flash2(ns_gpu* gpu, const void* q, const void* k, const void* v, int64_t Sq, int64_t Skv, int64_t Hq, int64_t Hkv, int64_t qs, int64_t kvs,
           int causal, int64_t row0, void* out) {
    static int set_on[16] = {};
    if (gpu->dev < 16 && !set_on[gpu->dev]) {
        NS_CU(cudaFuncSetAttribute(flash2_k<QK16, PV16>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM2));
        set_on[gpu->dev] = 1;
    }
    dim3 grid((unsigned) ((Sq + BQ2 - 1) / BQ2), (unsigned) Hq);
    const float scale_log2 = (1.0f / sqrtf((float) D)) * 1.4426950408889634f;
    flash2_k<QK16, PV16><<<grid, THREADS2, SMEM2, gpu->q>>>((const __half*) q, (const __half*) k, (const __half*) v, (__half*) out, (int) Sq, (int) Skv,
                                                           (int) Hq, (int) Hkv, (long) qs, (long) kvs, causal, (long) row0, scale_log2);
    NS_LAUNCH_CHECK();
    return 0;
}

}  // namespace

extern "C" {

int ns_image_qi21_silo_attention_ex(ns_gpu* gpu, const void* q, const void* k, const void* v, int64_t Sq, int64_t Skv, int64_t Hq, int64_t Hkv,
                                    int64_t qs, int64_t kvs, int causal, int64_t row0, void* out, int mode);

// 1 when this silo takes the shape (head size 128, whole 16-byte rows)
int ns_image_qi21_silo_attention_supported(int64_t D_, int64_t qs, int64_t kvs) {
    return D_ == D && qs % 8 == 0 && kvs % 8 == 0;
}

int ns_image_qi21_silo_attention(ns_gpu* gpu, const void* q, const void* k, const void* v, int64_t Sq, int64_t Skv, int64_t Hq, int64_t Hkv,
                                 int64_t qs, int64_t kvs, int causal, int64_t row0, void* out) {
    return ns_image_qi21_silo_attention_ex(gpu, q, k, v, Sq, Skv, Hq, Hkv, qs, kvs, causal, row0, out, 0);
}

// `mode`: 0 the first kernel (float32 sums, 64 rows a block); 1 + bit 1 (QK16) + bit 2 (PV16): the second kernel
int ns_image_qi21_silo_attention_ex(ns_gpu* gpu, const void* q, const void* k, const void* v, int64_t Sq, int64_t Skv, int64_t Hq, int64_t Hkv,
                                    int64_t qs, int64_t kvs, int causal, int64_t row0, void* out, int mode) {
    if (Hq % Hkv) return ns_fail("silo attention: query heads are not a multiple of the key/value heads");
    NS_CU(cudaSetDevice(gpu->dev));
    if (mode & 1) {
        switch (mode & 6) {
        case 0: return flash2<false, false>(gpu, q, k, v, Sq, Skv, Hq, Hkv, qs, kvs, causal, row0, out);
        case 2: return flash2<true, false>(gpu, q, k, v, Sq, Skv, Hq, Hkv, qs, kvs, causal, row0, out);
        case 4: return flash2<false, true>(gpu, q, k, v, Sq, Skv, Hq, Hkv, qs, kvs, causal, row0, out);
        default: return flash2<true, true>(gpu, q, k, v, Sq, Skv, Hq, Hkv, qs, kvs, causal, row0, out);
        }
    }
    const size_t smem = (size_t) 2 * BK * LDK * sizeof(__half);
    static bool set = false;
    if (!set) {
        NS_CU(cudaFuncSetAttribute(flash_k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem));
        set = true;
    }
    dim3 grid((unsigned) ((Sq + BQ - 1) / BQ), (unsigned) Hq);
    const float scale_log2 = (1.0f / sqrtf((float) D)) * 1.4426950408889634f;
    flash_k<<<grid, 128, smem, gpu->q>>>((const __half*) q, (const __half*) k, (const __half*) v, (__half*) out, (int) Sq, (int) Skv, (int) Hq, (int) Hkv,
                                         (long) qs, (long) kvs, causal, (long) row0, scale_log2);
    NS_LAUNCH_CHECK();
    return 0;
}

}  // extern "C"
