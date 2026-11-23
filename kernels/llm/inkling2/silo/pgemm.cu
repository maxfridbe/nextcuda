// pgemm.cu: the inkling2 silo's prompt products of the routed experts (silo.h: ns_ink2s_moe_prompt) - every expert's rows
// of a prompt chunk in one launch, its weights decoded from the repacked stack straight into shared memory as half (no
// expansion of each expert to a buffer, no product a launch an expert): a block a (64 output columns, 64 rows of one
// expert's run), four warps of 32 x 32, mma.sync m16n8k16 (half in, float32 sums), 32 values of the inner dimension at
// a time (one repacked line: 16 bytes of NVFP4 codes and two scales, or 32 Q8_0 quants and a scale).
#include "common.cuh"

using namespace ink2s;

namespace {

constexpr int BM = 64, BN = 64, BK = 32, PAD = BK + 8;

__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float ue4m3h(uint32_t x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    const uint32_t e = (x >> 3) & 0xF, m = x & 7;
    return e == 0 ? (float) m * (1.0f / 1024.0f) : __uint_as_float(((e + 119) << 23) | (m << 20));
}

__device__ __forceinline__ int qpos_inv(int p) {
    // the value at stored place p of a 16 (the 0 8 1 9 ... order)
    return (p & 1) ? 8 + (p >> 1) : (p >> 1);
}

// tiles: [expert, first row of x / y, rows] a tile
template <int TY>
__global__ void __launch_bounds__(128) moe_prompt_k(const uint8_t* w, int64_t rows_total, int64_t N, int64_t K, const __half* x, float* y, const int32_t* tiles) {
    __shared__ __align__(16) __half Xs[BM * PAD];
    __shared__ __align__(16) __half Ws[BN * PAD];
    const int tid = threadIdx.x, lane = tid % WS, wi = tid / WS;
    const int g = lane >> 2, tq = lane & 3;
    const int ex = tiles[blockIdx.y * 3], r0 = tiles[blockIdx.y * 3 + 1], nr = tiles[blockIdx.y * 3 + 2];
    const int64_t n0 = (int64_t) blockIdx.x * BN;
    const int wm = (wi / 2) * 32, wn = (wi % 2) * 32;
    const uint8_t* sc = w + align256(TY == 40 ? (size_t) (rows_total * K / 2) : (size_t) (rows_total * K));
    float acc[2][4][4];
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
        for (int b = 0; b < 4; ++b) acc[a][b][0] = acc[a][b][1] = acc[a][b][2] = acc[a][b][3] = 0.0f;
    const int64_t L = K / 32;
    for (int64_t l = 0; l < L; ++l) {
        __syncthreads();
        // the activations' tile: 64 rows x 32 halves, two 16-byte pieces a thread
        for (int i = tid; i < BM * 4; i += 128) {
            const int rr = i / 4, c8 = (i % 4) * 8;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (rr < nr) v = *(const uint4*) (x + (int64_t) (r0 + rr) * K + l * BK + c8);
            *(uint4*) (Xs + rr * PAD + c8) = v;
        }
        // the weights' tile: 64 rows of the expert x one line, two threads a row (a sub-block of 16 each)
        {
            const int rr = tid / 2, half_ = tid % 2;
            const int64_t row = (int64_t) ex * N + n0 + rr;
            __half* o = Ws + rr * PAD + half_ * 16;
            if constexpr (TY == 40) {
                const uint8_t* q = w + row * (K / 2) + l * 16 + half_ * 8;
                const float d = ue4m3h(sc[row * (K / 16) + l * 2 + half_]);
                const uint2 c = *(const uint2*) q;
                const uint8_t* cb = (const uint8_t*) &c;
                const float m8[8] = {0, 1, 2, 3, 4, 6, 8, 12};
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int lo = cb[j] & 15, hi = cb[j] >> 4;
                    o[j] = __float2half_rn((lo & 8 ? -m8[lo & 7] : m8[lo & 7]) * d);
                    o[j + 8] = __float2half_rn((hi & 8 ? -m8[hi & 7] : m8[hi & 7]) * d);
                }
            } else {
                const int8_t* q = (const int8_t*) w + row * K + l * 32 + half_ * 16;
                const float d = __half2float(((const __half*) sc)[row * L + l]);
#pragma unroll
                for (int p = 0; p < 16; ++p) o[qpos_inv(p)] = __float2half_rn(d * (float) q[p]);
            }
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            uint32_t a[2][4];
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                const __half* xr = Xs + (wm + mt * 16 + g) * PAD + kk * 16 + 2 * tq;
                a[mt][0] = *(const uint32_t*) xr;
                a[mt][1] = *(const uint32_t*) (xr + 8 * PAD);
                a[mt][2] = *(const uint32_t*) (xr + 8);
                a[mt][3] = *(const uint32_t*) (xr + 8 * PAD + 8);
            }
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const __half* wr = Ws + (wn + nt * 8 + g) * PAD + kk * 16 + 2 * tq;
                const uint32_t b0 = *(const uint32_t*) wr, b1 = *(const uint32_t*) (wr + 8);
#pragma unroll
                for (int mt = 0; mt < 2; ++mt) mma16816(acc[mt][nt], a[mt][0], a[mt][1], a[mt][2], a[mt][3], b0, b1);
            }
        }
    }
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const int rr = wm + mt * 16 + g;
            const int64_t c = n0 + wn + nt * 8 + 2 * tq;
            if (rr < nr) *(float2*) (y + (int64_t) (r0 + rr) * N + c) = make_float2(acc[mt][nt][0], acc[mt][nt][1]);
            if (rr + 8 < nr) *(float2*) (y + (int64_t) (r0 + rr + 8) * N + c) = make_float2(acc[mt][nt][2], acc[mt][nt][3]);
        }
}

}  // namespace

extern "C" {

int ns_ink2s_moe_prompt_supported(int type, int64_t N, int64_t K) {
    return (type == 40 || type == 8) && N % BN == 0 && K % 64 == 0 ? 1 : 0;
}

int ns_ink2s_moe_prompt(ns_gpu* g, int type, const void* w, int64_t rows_total, int64_t N, int64_t K, const uint16_t* x, float* y, const int32_t* tiles,
                        int64_t ntiles) {
    NS_TRY
    if (!ns_ink2s_moe_prompt_supported(type, N, K)) return ns_fail("ns_ink2s_moe_prompt: Q8_0 / NVFP4, outputs of a multiple of 64, inputs of 64");
    if (ntiles <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    const dim3 grid((unsigned) (N / BN), (unsigned) ntiles);
    if (type == 40) moe_prompt_k<40><<<grid, 128, 0, g->q>>>((const uint8_t*) w, rows_total, N, K, (const __half*) x, y, tiles);
    else moe_prompt_k<8><<<grid, 128, 0, g->q>>>((const uint8_t*) w, rows_total, N, K, (const __half*) x, y, tiles);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

}  // extern "C"
