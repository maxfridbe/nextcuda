// gemv.cu: SkinTokens' decode step's linear layers in the silo (silo.h): out [M, N] = x [M, K] . W^T for M at most 16 rows
// (the beams) with W held as bfloat16. TokenRig's weights are stored in bfloat16, so the bf16 copy is the checkpoint's own
// values and the product (float32 activations, float32 accumulation) differs from the float32 cuBLAS product only in the
// order of the sums. At M = 10 the step is bound by reading the weights: half the bytes, one pass.
//
// A block of NW warps takes R output rows; each warp a quarter of K, a lane 8 values (16 bytes of a row) a time for all M
// activation rows; the lanes' sums reduce by shuffles, the warps' through shared memory. nextsycl's silo kernel (sub-groups
// of 16 on the B70) with warps of 32.
#include "ns.h"
#include "ns_internal.hpp"
#include "silo.h"

#include <algorithm>
#include <cstdint>

namespace {
constexpr int NW = 4, VEC = 8;

__device__ inline float bf(uint32_t half_bits) {
    return __uint_as_float(half_bits << 16);
}

// MT: the activation rows padded up (M of them real), R: output rows a block
template <int MT, int R>
__global__ void __launch_bounds__(NW * 32) gemv_k(const float* __restrict__ x, const uint16_t* __restrict__ w, float* __restrict__ out, int64_t M, int64_t N,
                                                  int64_t K) {
    __shared__ float red[NW * R * MT];
    const int lane = threadIdx.x & 31, s = threadIdx.x >> 5;
    const int64_t n0 = (int64_t) blockIdx.x * R;
    const int64_t nv = K / VEC;
    const int64_t per = (nv + NW - 1) / NW;
    const int64_t v0 = s * per, v1 = nv < v0 + per ? nv : v0 + per;
    float acc[R][MT];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int m = 0; m < MT; ++m) acc[r][m] = 0.0f;
    for (int64_t v = v0 + lane; v < v1; v += 32) {
        float wv[R][VEC];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int64_t row = n0 + r < N ? n0 + r : N - 1;
            const uint4 u = *reinterpret_cast<const uint4*>(w + row * K + v * VEC);
            const uint32_t p[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                wv[r][2 * i] = bf(p[i] & 0xffffu);
                wv[r][2 * i + 1] = bf(p[i] >> 16);
            }
        }
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            float xv[VEC];
            if (m < M) {
                const float4 a = *reinterpret_cast<const float4*>(x + m * K + v * VEC);
                const float4 b = *reinterpret_cast<const float4*>(x + m * K + v * VEC + 4);
                xv[0] = a.x; xv[1] = a.y; xv[2] = a.z; xv[3] = a.w;
                xv[4] = b.x; xv[5] = b.y; xv[6] = b.z; xv[7] = b.w;
            } else {
#pragma unroll
                for (int i = 0; i < VEC; ++i) xv[i] = 0.0f;
            }
#pragma unroll
            for (int r = 0; r < R; ++r)
#pragma unroll
                for (int i = 0; i < VEC; ++i) acc[r][m] = fmaf(wv[r][i], xv[i], acc[r][m]);
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            float t = acc[r][m];
            for (int o = 16; o > 0; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
            if (lane == 0) red[(s * R + r) * MT + m] = t;
        }
    __syncthreads();
    if (s == 0) {
        for (int i = lane; i < R * MT; i += 32) {
            const int r = i / MT, m = i % MT;
            float t = 0.0f;
#pragma unroll
            for (int j = 0; j < NW; ++j) t += red[j * R * MT + i];
            if (m < M && n0 + r < N) out[m * N + n0 + r] = t;
        }
    }
}

template <int MT, int R>
int run(cudaStream_t st, const float* x, const uint16_t* w, float* out, int64_t M, int64_t N, int64_t K) {
    gemv_k<MT, R><<<(unsigned) ((N + R - 1) / R), NW * 32, 0, st>>>(x, w, out, M, N, K);
    NS_LAUNCH_CHECK();
    return 0;
}

template <int R>
int dispatch(cudaStream_t st, const float* x, const uint16_t* w, float* out, int64_t M, int64_t N, int64_t K) {
    if (M <= 1) return run<1, R>(st, x, w, out, M, N, K);
    if (M <= 2) return run<2, R>(st, x, w, out, M, N, K);
    if (M <= 4) return run<4, R>(st, x, w, out, M, N, K);
    if (M <= 8) return run<8, R>(st, x, w, out, M, N, K);
    if (M <= 12) return run<12, R>(st, x, w, out, M, N, K);
    return run<16, R>(st, x, w, out, M, N, K);
}
}  // namespace

extern "C" {

int ns_model3d_skintokens_silo_gemv_supported(int64_t M, int64_t N, int64_t K) {
    return M >= 1 && M <= 16 && N > 0 && K > 0 && K % VEC == 0 ? 1 : 0;
}

int ns_model3d_skintokens_silo_gemv(ns_gpu* g, const float* x, const uint16_t* w, int64_t M, int64_t N, int64_t K, float* out) {
    if (!ns_model3d_skintokens_silo_gemv_supported(M, N, K)) return ns_fail("ns_model3d_skintokens_silo_gemv: shapes it does not take");
    if (((uintptr_t) x & 15) || ((uintptr_t) w & 15)) return ns_fail("ns_model3d_skintokens_silo_gemv: x and w must be 16-byte aligned");
    NS_CU(cudaSetDevice(g->dev));
    return N >= 4096 ? dispatch<2>(g->q, x, w, out, M, N, K) : dispatch<1>(g->q, x, w, out, M, N, K);
}

}  // extern "C"
