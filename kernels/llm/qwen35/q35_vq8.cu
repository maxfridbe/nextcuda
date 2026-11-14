// q35_vq8.cu: the verify passes' products - 2..8 rows (a verify pass's: the token and its drafts) against the stored
// blocks, each weight block decoded to int8 once and multiplied with all the rows on the int8 tensor cores (mma
// m16n8k32: 16 matrix rows x the 8 input rows x a block of 32). ggml-cuda's MMVQ decodes a block again for every input
// row (the 3060 runs 8 of them at ~124 GB/s of its 360) and its MMQ tiles 128 rows for 8 (the same on the 3060). The
// sums are each block of 32's integer dot product, the type's scale on it, then (block scale x the row's Q8_1 scale)
// x that. The blocks are read by nextcuda's own code (nh_blocks.cuh's tables and sign masks; ggml-cuda's vecdotq.cuh
// helpers until 2026-11-13). The types Qwen3.8-27B GSQ-RCO's layers are mostly made of: IQ3_S,
// IQ4_XS, IQ3_XXS (82% of their bytes; the others go to MMQ - the 2-bit i-quants and Q4_K were written this way too,
// measured no faster than MMQ, and dropped). After nextsycl's silo/xmm.cpp (its XMX verify products), for CUDA.
// Measured (the 5070 + 3060, layers 0-32 on the 3060): an 8-row verify pass 61.0 ms with these and MMQ for the rest,
// 62.6 with MMQ alone, 66.5 with MMVQ; at 2..5 rows MMVQ stays faster (the engine's NS_Q35_VQ8 threshold)
#include "ns.h"
#include "ns_internal.hpp"
#include "q35.h"

#include "nh_blocks.cuh"

#include <cstdlib>
#include <string>

namespace {

using nsq::Q8_1;
constexpr int QK = 256;

// a block's bytes as a word at a 2-byte-aligned offset, a 4-byte-aligned one
__device__ __forceinline__ int i32_b2(const uint8_t* p) { return (int) nsq::u32_at(p); }
__device__ __forceinline__ int i32_b4(const uint8_t* p) { return *(const int*) p; }

// A type: the 4 values of group g (0..7) of piece ib (0..7) of a 256-block as int8 x 4 (`grid`: its lookup table,
// copied to shared memory: GRID words), a matrix row's scales for a piece of 32 (`rs`, once a row and piece), and the
// piece's share of one output from its integer dot product
struct Rs {
    float d;
    int a;   // the piece's integer scale: the product's integer times it, one conversion
};
// IQ3_S (110 bytes): half d, 64 bytes of grid indices' low parts, 8 of their ninth bits, 32 of signs, 4 of two scales
struct DecIq3s {
    static constexpr int BS = 110, GRID = 512;
    static __device__ __forceinline__ const uint32_t* table() { return nsq_iq3s_grid; }
    static __device__ __forceinline__ int vals(const uint8_t* b, int ib, int g, const uint32_t* grid) {
        const int idx = b[2 + ib * 8 + g] | (((b[66 + ib] >> g) & 1) << 8);
        return nsq::signed4(grid[idx], b[74 + ib * 4 + g / 2] >> (4 * (g & 1)));
    }
    static __device__ __forceinline__ Rs rs(const uint8_t* b, int ib) {
        return {nsq::half_at(b), 1 + 2 * ((b[106 + ib / 2] >> (4 * (ib & 1))) & 0x0F)};
    }
    using Ctx = const uint32_t*;
    static __device__ __forceinline__ Ctx ctx(const uint32_t* grid) { return grid; }
    static __device__ __forceinline__ void vals2(const uint8_t* b, int ib, int g, Ctx grid, int& lo, int& hi) {
        lo = vals(b, ib, g, grid);
        hi = vals(b, ib, g + 4, grid);
    }
};

// IQ3_XXS (98 bytes): half d, 64 grid indices, then a word a piece: 4 x 7 sign bits and a 4-bit scale
struct DecIq3xxs {
    static constexpr int BS = 98, GRID = 256;
    static __device__ __forceinline__ const uint32_t* table() { return nsq_iq3xxs_grid; }
    static __device__ __forceinline__ int vals(const uint8_t* b, int ib, int g, const uint32_t* grid) {
        const uint32_t signs = nsq::signs7(nsq::u32_at(b + 66 + 4 * ib) >> (7 * (g / 2)));
        return nsq::signed4(grid[b[2 + ib * 8 + g]], signs >> (4 * (g & 1)));
    }
    // (0.5 + s) / 2 = (2 s + 1) / 4: the quarter in d, exact
    static __device__ __forceinline__ Rs rs(const uint8_t* b, int ib) { return {0.25f * nsq::half_at(b), 1 + 2 * (int) (nsq::u32_at(b + 66 + 4 * ib) >> 28)}; }
    using Ctx = const uint32_t*;
    static __device__ __forceinline__ Ctx ctx(const uint32_t* grid) { return grid; }
    static __device__ __forceinline__ void vals2(const uint8_t* b, int ib, int g, Ctx grid, int& lo, int& hi) {
        lo = vals(b, ib, g, grid);
        hi = vals(b, ib, g + 4, grid);
    }
};

// IQ4_XS (136 bytes): half d, 16 bits of scales' high parts, 4 bytes of their low nibbles, 128 of level indices
struct DecIq4xs {
    static constexpr int BS = 136, GRID = 0;
    static __device__ __forceinline__ const uint32_t* table() { return nullptr; }
    // groups g and g + 4 at once: the low and high nibbles of the same 4 bytes
    struct Ctx {
        uint32_t t[4];
    };
    static __device__ __forceinline__ Ctx ctx(const uint32_t*) {
        Ctx c;
        nsq::iq4nl_table(c.t);
        return c;
    }
    static __device__ __forceinline__ void vals2(const uint8_t* b, int ib, int g, const Ctx& c, int& lo, int& hi) {
        const int2 v = nsq::levels8((uint32_t) i32_b4(b + 8 + 16 * ib + 4 * g), c.t);
        lo = v.x;
        hi = v.y;
    }
    static __device__ __forceinline__ Rs rs(const uint8_t* b, int ib) {
        const int ls = ((b[4 + ib / 2] >> (4 * (ib & 1))) & 0x0F) | (int) (((nsq::u16_at(b + 2) >> (2 * ib)) & 0x03) << 4);
        return {nsq::half_at(b), ls - 32};
    }
};

__device__ __forceinline__ float part(const Rs& r, int lo, float dy) { return (r.d * dy) * (float) (lo * r.a); }

__device__ __forceinline__ void mma_s8(int (&c)[4], int a0, int a1, int a2, int a3, int b0, int b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    (void) c; (void) a0; (void) a1; (void) a2; (void) a3; (void) b0; (void) b1;
#endif
}

template <class D, int NW>
struct Shape {
    static constexpr int BS = D::BS;
    static constexpr int STRIDE_W = ((NW * BS / 4) | 1);                    // a row's words in the stage, odd
    static constexpr int STAGE_W = 16 * STRIDE_W;
    static constexpr int GRID_W = D::GRID > 0 ? D::GRID : 1;
    static constexpr int RED_W = NW * 128;                                   // the warps' sums at the end
    static constexpr size_t SMEM = (size_t) (GRID_W + (STAGE_W > RED_W ? STAGE_W : RED_W)) * 4;
};

// A block of NW warps takes 16 matrix rows. Each step its threads copy the rows' next NW 256-blocks into the stage in
// shared memory (whole words: each row's run is contiguous in the matrix; the other blocks on the SM hide the wait - a
// cp.async double buffer and a register prefetch were both slower), and warp w multiplies 256-block w of them: lane l
// decodes matrix rows l / 4 and l / 4 + 8 (the
// A fragment's), groups l % 4 and l % 4 + 4 of each block of 32, loads input row l / 4 (the B fragment's) and keeps
// outputs (rows l / 4, + 8) x (input rows 2 (l % 4), + 1) (the C fragment's). The warps' sums are added at the end.
// Stage rows are an odd number of words apart (the 8 rows a decode reads: 8 banks)
template <class D, int NW>
__global__ void __launch_bounds__(32 * NW) vq8_k(const void* __restrict__ w, const Q8_1* __restrict__ y, float* __restrict__ out, int K, int n_out,
                                                 int ncols) {
    using S = Shape<D, NW>;
    extern __shared__ uint32_t sm[];
    uint32_t* grid = sm;
    uint32_t* stage = sm + S::GRID_W;
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, tid = threadIdx.x;
    const int nsb = K / QK, nkb = K / 32;
    const int row0 = blockIdx.x * 16;
    for (int i = tid; i < D::GRID; i += 32 * NW) grid[i] = D::table()[i];
    const int ra = lane / 4, rb = ra + 8;
    __syncthreads();
    const typename D::Ctx dc = D::ctx(grid);
    const int g = lane % 4, col = lane / 4, c0 = 2 * (lane % 4);
    const Q8_1* yc = y + (size_t) col * nkb;
    const Q8_1* y0 = y + (size_t) c0 * nkb;
    const Q8_1* y1 = y + (size_t) (c0 + 1) * nkb;
    const bool has_b = col < ncols, has0 = c0 < ncols, has1 = c0 + 1 < ncols;
    const size_t row_bytes = (size_t) nsb * S::BS;
    const char* base = (const char*) w + (size_t) row0 * row_bytes;
    float acc[4] = {0.f, 0.f, 0.f};
    for (int step = 0; step * NW < nsb; ++step) {
        {
            const int sb0 = step * NW, nb = min(NW, nsb - sb0), words = nb * S::BS / 4;
            __syncthreads();
            for (int i = tid; i < 16 * words; i += 32 * NW) {
                const int r = i / words, k = i % words;
                stage[r * S::STRIDE_W + k] = reinterpret_cast<const uint32_t*>(base + (size_t) r * row_bytes + (size_t) sb0 * S::BS)[k];
            }
            __syncthreads();
        }
        const int sb0 = step * NW, nb = min(NW, nsb - sb0);
        if (wid < nb) {
            const uint8_t* ba = reinterpret_cast<const uint8_t*>(stage + ra * S::STRIDE_W) + wid * S::BS;
            const uint8_t* bb = reinterpret_cast<const uint8_t*>(stage + rb * S::STRIDE_W) + wid * S::BS;
            const int sbk = (sb0 + wid) * (QK / 32);
#pragma unroll 4
            for (int ib = 0; ib < QK / 32; ++ib) {
                const int kb = sbk + ib;
                int a0, a1, a2, a3;
                D::vals2(ba, ib, g, dc, a0, a2);
                D::vals2(bb, ib, g, dc, a1, a3);
                const int b0 = has_b ? i32_b4((const uint8_t*) yc[kb].qs + 4 * g) : 0;
                const int b1 = has_b ? i32_b4((const uint8_t*) yc[kb].qs + 4 * (g + 4)) : 0;
                int c[4] = {0, 0, 0, 0};
                mma_s8(c, a0, a1, a2, a3, b0, b1);
                const Rs sa = D::rs(ba, ib), sbs = D::rs(bb, ib);
                if (has0) {
                    const float dy = __low2float(y0[kb].ds);
                    acc[0] += part(sa, c[0], dy);
                    acc[2] += part(sbs, c[2], dy);
                }
                if (has1) {
                    const float dy = __low2float(y1[kb].ds);
                    acc[1] += part(sa, c[1], dy);
                    acc[3] += part(sbs, c[3], dy);
                }
            }
        }
    }
    // the warps' sums (the stage reused)
    __syncthreads();
    float* red = reinterpret_cast<float*>(stage);
#pragma unroll
    for (int i = 0; i < 4; ++i) red[(wid * 4 + i) * 32 + lane] = acc[i];
    __syncthreads();
    if (wid != 0) return;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float s = 0.f;
        for (int k = 0; k < NW; ++k) s += red[(k * 4 + i) * 32 + lane];
        acc[i] = s;
    }
    if (has0) {
        out[(size_t) c0 * n_out + row0 + ra] = acc[0];
        out[(size_t) c0 * n_out + row0 + rb] = acc[2];
    }
    if (has1) {
        out[(size_t) (c0 + 1) * n_out + row0 + ra] = acc[1];
        out[(size_t) (c0 + 1) * n_out + row0 + rb] = acc[3];
    }
}

template <class D, int NW>
bool launch_nw(const void* w, const Q8_1* y, float* out, int64_t K, int64_t n_out, int64_t ncols, cudaStream_t q) {
    constexpr int BS = D::BS;
    const int64_t nsb = K / QK;
    // whole words: a row's bytes, a step's (and the last, shorter step's)
    if ((nsb * BS) % 4 != 0 || (NW * BS) % 4 != 0 || ((nsb % NW) * BS) % 4 != 0) return false;
    constexpr size_t smem = Shape<D, NW>::SMEM;
    static_assert(smem <= 48 * 1024, "a block's shared memory past the default 48 KiB");
    vq8_k<D, NW><<<(unsigned) (n_out / 16), 32 * NW, smem, q>>>(w, y, out, (int) K, (int) n_out, (int) ncols);
    return true;
}

template <class D>
bool launch(const void* w, const Q8_1* y, float* out, int64_t K, int64_t n_out, int64_t ncols, int nw, cudaStream_t q) {
    switch (nw) {
        case 2: return launch_nw<D, 2>(w, y, out, K, n_out, ncols, q);
        case 8: return launch_nw<D, 8>(w, y, out, K, n_out, ncols, q);
        default: return launch_nw<D, 4>(w, y, out, K, n_out, ncols, q);
    }
}

}  // namespace

extern "C" {

int ns_q35_vq8_supported(int type) {
    switch (type) {
        case nsq::IQ3_S: case nsq::IQ3_XXS: case nsq::IQ4_XS:
            return 1;
        default: return 0;
    }
}

int ns_q35_vq8(ns_gpu* g, int type, const void* w, const void* xq, float* y, int64_t n_in, int64_t n_out, int64_t ncols) {
    NS_TRY
    NS_CU(cudaSetDevice(g->dev));
    if (!ns_q35_vq8_supported(type) || n_in % QK != 0 || n_out % 16 != 0 || ncols < 1 || ncols > 8)
        return ns_fail("ns_q35_vq8: IQ3_S IQ3_XXS IQ4_XS, rows a multiple of 16, inputs a multiple of 256, 1..8 input rows");
    // warps a block: enough blocks for the card - fewer rows, more warps to share each one's blocks
    static const int nw_env = [] { const char* v = std::getenv("NS_Q35_VQ8_WARPS"); return v ? std::atoi(v) : 0; }();
    const int nw = nw_env > 0 ? nw_env : (n_out <= 2048 ? 8 : 4);
    const Q8_1* yq = static_cast<const Q8_1*>(xq);
    bool ok = false;
    switch (type) {
        case nsq::IQ3_S: ok = launch<DecIq3s>(w, yq, y, n_in, n_out, ncols, nw, g->q); break;
        case nsq::IQ3_XXS: ok = launch<DecIq3xxs>(w, yq, y, n_in, n_out, ncols, nw, g->q); break;
        default: ok = launch<DecIq4xs>(w, yq, y, n_in, n_out, ncols, nw, g->q); break;
    }
    // (a row's blocks not whole words: MMVQ's products, the same sums)
    if (!ok) return ns_q35_mmvq(g, type, w, xq, y, n_in, n_out, ncols);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

}  // extern "C"
