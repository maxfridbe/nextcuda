// q35_q8f.cu: decode rows' products from Q8_0 matrices with the activations kept in float32 (nextcuda's own). The
// decode products elsewhere (ns_nh_mmvq, q35_vq8.cu) quantize each activation row to Q8_1 (int8 a block of 32) first:
// fine for text, but Haru-4B's silent / speak decision moved with it (its inputs sum a token's row and two audio
// projections). Here each output is d * sum(q_j * x_j) over a block in float32: the weights' rounding only.
// One warp a matrix row, lanes over its blocks (a block's 34 bytes read as 17 halves: d, then 32 int8), up to 8 rows
// of x at once (read through the cache: x is small and every warp reads it).
#include "ns.h"
#include "ns_internal.hpp"
#include "q35.h"

#include <cuda_fp16.h>

namespace {

template <int T>
__global__ void q8f_k(const uint8_t* __restrict__ w, const float* __restrict__ x, float* __restrict__ y, int64_t n_in, int64_t n_out) {
    const int lane = threadIdx.x & 31;
    const int64_t row = (int64_t) blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= n_out) return;
    const int64_t nb = n_in / 32;
    const uint8_t* wr = w + row * nb * 34;
    float acc[T];
#pragma unroll
    for (int t = 0; t < T; ++t) acc[t] = 0.0f;
    for (int64_t b = lane; b < nb; b += 32) {
        const uint8_t* blk = wr + b * 34;
        const __half_raw hr = {*(const unsigned short*) blk};
        const float d = __half2float(__half(hr));
        // the 32 int8 as 16 two-byte loads (a block is 2-byte aligned)
        float q[32];
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const unsigned short v = *(const unsigned short*) (blk + 2 + 2 * j);
            q[2 * j] = (float) (int8_t) (v & 0xff);
            q[2 * j + 1] = (float) (int8_t) (v >> 8);
        }
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float4* xr = (const float4*) (x + t * n_in + b * 32);
            float s = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float4 v = __ldg(xr + j);
                s += q[4 * j] * v.x + q[4 * j + 1] * v.y + q[4 * j + 2] * v.z + q[4 * j + 3] * v.w;
            }
            acc[t] += d * s;
        }
    }
#pragma unroll
    for (int t = 0; t < T; ++t) {
        float s = acc[t];
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (lane == 0) y[t * n_out + row] = s;
    }
}

template <int T>
void launch(cudaStream_t q, const void* w, const float* x, float* y, int64_t n_in, int64_t n_out) {
    const int warps = 8;
    q8f_k<T><<<(unsigned) ((n_out + warps - 1) / warps), 32 * warps, 0, q>>>((const uint8_t*) w, x, y, n_in, n_out);
}

}  // namespace

extern "C" {

int ns_q35_gemv_q8f(ns_gpu* g, const void* w, const float* x, float* y, int64_t n_in, int64_t n_out, int64_t rows) {
    NS_TRY
    NS_CU(cudaSetDevice(g->dev));
    if (rows < 1 || rows > 8) return ns_fail("ns_q35_gemv_q8f: 1..8 rows");
    if (n_in % 32 != 0) return ns_fail("ns_q35_gemv_q8f: the inputs a multiple of 32");
    switch (rows) {
        case 1: launch<1>(g->q, w, x, y, n_in, n_out); break;
        case 2: launch<2>(g->q, w, x, y, n_in, n_out); break;
        case 3: launch<3>(g->q, w, x, y, n_in, n_out); break;
        case 4: launch<4>(g->q, w, x, y, n_in, n_out); break;
        case 5: launch<5>(g->q, w, x, y, n_in, n_out); break;
        case 6: launch<6>(g->q, w, x, y, n_in, n_out); break;
        case 7: launch<7>(g->q, w, x, y, n_in, n_out); break;
        default: launch<8>(g->q, w, x, y, n_in, n_out); break;
    }
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

}  // extern "C"
