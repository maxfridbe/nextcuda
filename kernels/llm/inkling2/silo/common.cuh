// common.cuh: what the inkling2 silo's sources share - warp sums, the activations' Q8_0 layout (quants in the
// 0 8 1 9 ... order within each 16, then the float scales from a 256-byte boundary) and its rounding, BF16.
#pragma once
#include "ns.h"
#include "ns_internal.hpp"
#include "silo.h"

#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>
#include <string>

namespace ink2s {

constexpr int WS = 32;

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// a block's sum (every thread gets it); sh 32 floats of shared memory; blockDim a multiple of 32
__device__ __forceinline__ float block_sum(float v, float* sh) {
    const int lane = threadIdx.x % WS, w = threadIdx.x / WS, nw = blockDim.x / WS;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    float r = 0.0f;
    for (int i = 0; i < nw; ++i) r += sh[i];
    return r;
}

__host__ __device__ inline size_t align256(size_t n) { return (n + 255) / 256 * 256; }

// a value's place in its 32 (the 0 8 1 9 ... order within each 16)
__device__ __forceinline__ int qpos(int v) {
    const int w = v & 15;
    return (v & 16) + (w < 8 ? 2 * w : 2 * (w - 8) + 1);
}

// activations' quants and scales (rows T of n)
__device__ __forceinline__ int8_t* xq_q(void* xq) { return (int8_t*) xq; }
__device__ __forceinline__ float* xq_d(void* xq, int64_t T, int64_t n) { return (float*) ((uint8_t*) xq + align256((size_t) (T * n))); }

// ggml's Q8_0 rounding of a block of 32, a lane a value: d = amax / 127 (kept as half), q = round(x / d)
__device__ __forceinline__ void quant32(float v, int lane, int8_t* q, float* d) {
    const float amax = warp_max(fabsf(v));
    const float dd = amax / 127.0f;
    const float id = dd != 0.0f ? 1.0f / dd : 0.0f;
    q[qpos(lane)] = (int8_t) roundf(v * id);
    if (lane == 0) *d = __half2float(__float2half_rn(dd));
}

// float32 -> BF16 bits, round to nearest even (torch's .to(bfloat16))
__device__ __forceinline__ uint16_t to_bf16(float x) {
    uint32_t u = __float_as_uint(x);
    if ((u & 0x7F800000u) == 0x7F800000u) return (uint16_t) (u >> 16);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
__device__ __forceinline__ float bf16f(uint16_t b) { return __uint_as_float((uint32_t) b << 16); }
__device__ __forceinline__ float rbf(float x) { return bf16f(to_bf16(x)); }

inline unsigned blocks(int64_t n, int t = 256) { return (unsigned) ((n + t - 1) / t); }

}  // namespace ink2s
