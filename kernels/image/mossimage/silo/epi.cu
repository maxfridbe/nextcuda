// epi.cu: MOSS-Image's int8 epilogues - the int8 product's int32 results scaled and taken straight to what the next
// step reads (q k v as half, the MLP's activation, the gated residual), instead of a float32 [M, N] written and read
// again (nsd_int8_linear's epilogue, then to_half_rows / situ / gate_add: about a tenth of a step, 2026-10-26).
#include "ns.h"
#include "ns_internal.hpp"
#include "silo.h"

#include <cuda_fp16.h>

namespace {

constexpr int WG = 256;

unsigned blocks(int64_t n) {
    return (unsigned) ((n + WG - 1) / WG);
}

// four columns a thread (N a multiple of 4)
__global__ void epi_half_k(const int32_t* acc, int64_t M, int64_t N, const float* xs, const float* ws, float alpha, __half* out, int64_t ldo) {
    const int64_t i = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i >= M * N) return;
    const int64_t r = i / N, c = i % N;
    const int4 a = *reinterpret_cast<const int4*>(acc + i);
    const float4 w = *reinterpret_cast<const float4*>(ws + c);
    const float s = xs[r] * alpha;
    __half2* o = reinterpret_cast<__half2*>(out + r * ldo + c);
    o[0] = __floats2half2_rn((float) a.x * s * w.x, (float) a.y * s * w.y);
    o[1] = __floats2half2_rn((float) a.z * s * w.z, (float) a.w * s * w.w);
}

__device__ __forceinline__ float situ(float h, float g) {
    return 4.0f * tanhf(g * 0.25f) / (1.0f + expf(-g)) * 25.0f * tanhf(h * 0.04f);
}

__global__ void epi_situ_k(const int32_t* acc, int64_t M, int64_t F, const float* xs, const float* ws, __half* out) {
    const int64_t i = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i >= M * F) return;
    const int64_t r = i / F, c = i % F;
    const int32_t* row = acc + r * 2 * F;
    const int4 h = *reinterpret_cast<const int4*>(row + c);
    const int4 g = *reinterpret_cast<const int4*>(row + F + c);
    const float4 wh = *reinterpret_cast<const float4*>(ws + c);
    const float4 wg = *reinterpret_cast<const float4*>(ws + F + c);
    const float s = xs[r];
    __half2* o = reinterpret_cast<__half2*>(out + i);
    o[0] = __floats2half2_rn(situ((float) h.x * s * wh.x, (float) g.x * s * wg.x), situ((float) h.y * s * wh.y, (float) g.y * s * wg.y));
    o[1] = __floats2half2_rn(situ((float) h.z * s * wh.z, (float) g.z * s * wg.z), situ((float) h.w * s * wh.w, (float) g.w * s * wg.w));
}

__global__ void epi_gate_add_k(const int32_t* acc, int64_t M, int64_t N, const float* xs, const float* ws, float alpha, const float* gate, float* x) {
    const int64_t i = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i >= M * N) return;
    const int64_t r = i / N, c = i % N;
    const int4 a = *reinterpret_cast<const int4*>(acc + i);
    const float4 w = *reinterpret_cast<const float4*>(ws + c);
    const float4 gt = *reinterpret_cast<const float4*>(gate + c);
    const float s = xs[r] * alpha;
    float4 v = *reinterpret_cast<float4*>(x + i);
    v.x += gt.x * (float) a.x * s * w.x;
    v.y += gt.y * (float) a.y * s * w.y;
    v.z += gt.z * (float) a.z * s * w.z;
    v.w += gt.w * (float) a.w * s * w.w;
    *reinterpret_cast<float4*>(x + i) = v;
}

}  // namespace

extern "C" {

int ns_image_moss_silo_epi_half(ns_gpu* g, const int32_t* acc, int64_t M, int64_t N, const float* xs, const float* ws, float alpha, void* out,
                                int64_t ldo) {
    if (N % 4 || ldo % 4) return ns_fail("moss epilogue: columns a multiple of 4");
    NS_CU(cudaSetDevice(g->dev));
    epi_half_k<<<blocks(M * N / 4), WG, 0, g->q>>>(acc, M, N, xs, ws, alpha, (__half*) out, ldo);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_moss_silo_epi_situ(ns_gpu* g, const int32_t* acc, int64_t M, int64_t F, const float* xs, const float* ws, void* out) {
    if (F % 4) return ns_fail("moss epilogue: columns a multiple of 4");
    NS_CU(cudaSetDevice(g->dev));
    epi_situ_k<<<blocks(M * F / 4), WG, 0, g->q>>>(acc, M, F, xs, ws, (__half*) out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_moss_silo_epi_gate_add(ns_gpu* g, const int32_t* acc, int64_t M, int64_t N, const float* xs, const float* ws, float alpha,
                                    const float* gate, float* x) {
    if (N % 4) return ns_fail("moss epilogue: columns a multiple of 4");
    NS_CU(cudaSetDevice(g->dev));
    epi_gate_add_k<<<blocks(M * N / 4), WG, 0, g->q>>>(acc, M, N, xs, ws, alpha, gate, x);
    NS_LAUNCH_CHECK();
    return 0;
}

}  // extern "C"
