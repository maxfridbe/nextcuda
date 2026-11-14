// nh_iq.cu: the products from the stored blocks, and what else reads them - the activations made Q8_1, the decode
// products (one to eight columns of a matrix; more eight at a time), a layer's experts in one launch (a weight pointer
// an entry), the expansions to half and float and the embedding rows. Every GGUF type nh_blocks.cuh reads, through its
// one form (a 32-value piece: int8 and the halves' scales and offsets), so a type is one decoder, not a kernel each.
// nextcuda's own; the qwen35 engine (and the engines that borrow its products) call these too.
#include "ns.h"
#include "ns_internal.hpp"
#include "nh.h"
#include "nh_blocks.cuh"

#include <cuda_fp16.h>

namespace {

using nsq::Fmt;
using nsq::Q8_1;
using nsq::Sub;

constexpr int BF16 = 30;

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// x [n] -> Q8_1 blocks: a warp a block of 32, the scale the largest magnitude / 127, the half sum of the values
__global__ void q8_1_k(const float* __restrict__ x, Q8_1* __restrict__ y, int64_t nblk) {
    const int64_t b = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (b >= nblk) return;
    const float v = x[b * 32 + lane];
    float amax = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float sum = warp_sum(v);
    const float d = amax / 127.f;
    y[b].qs[lane] = (int8_t) (amax == 0.f ? 0 : (int) roundf(v / d));
    if (lane == 0) y[b].ds = __floats2half2_rn(d, sum);
}

// y [c][n_out] = row . x column c, a warp a row (4 a block of 128 threads), a lane a piece of 32 at a time
template<int T, int NC>
__global__ void __launch_bounds__(128) mmvq_k(const uint8_t* __restrict__ w, size_t rb, const Q8_1* __restrict__ x, float* __restrict__ y,
                                              int n_in, int n_out, int ncols) {
    const int row = blockIdx.x * 4 + threadIdx.y;
    if (row >= n_out) return;
    constexpr int spb = Fmt<T>::qk / 32;
    const int nsub = n_in / 32;
    const uint8_t* wr = w + (size_t) row * rb;
    float acc[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) acc[c] = 0.f;
    for (int j = threadIdx.x; j < nsub; j += 32) {
        Sub s;
        nsq::sub<T>(wr + (size_t) (j / spb) * Fmt<T>::bytes, j % spb, s);
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (c < ncols) acc[c] += nsq::dot<T>(s, x[(size_t) c * nsub + j]);
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        const float v = warp_sum(acc[c]);
        if (threadIdx.x == 0 && c < ncols) y[(size_t) c * n_out + row] = v;
    }
}

// the experts of a layer in one launch: entry j's weights at wp[j] (rows of rb bytes) . Q8_1 row xrow[j] of x
template<int T>
__global__ void __launch_bounds__(128) mmvq_ptrs_k(const unsigned long long* __restrict__ wp, size_t rb, const int32_t* __restrict__ xrow,
                                                   const Q8_1* __restrict__ x, float* __restrict__ y, int n_in, int n_out) {
    const int row = blockIdx.x * 4 + threadIdx.y;
    if (row >= n_out) return;
    const int e = blockIdx.y;
    constexpr int spb = Fmt<T>::qk / 32;
    const int nsub = n_in / 32;
    const uint8_t* wr = (const uint8_t*) wp[e] + (size_t) row * rb;
    const Q8_1* xr = x + (size_t) xrow[e] * nsub;
    float acc = 0.f;
    for (int j = threadIdx.x; j < nsub; j += 32) {
        Sub s;
        nsq::sub<T>(wr + (size_t) (j / spb) * Fmt<T>::bytes, j % spb, s);
        acc += nsq::dot<T>(s, xr[j]);
    }
    acc = warp_sum(acc);
    if (threadIdx.x == 0) y[(size_t) e * n_out + row] = acc;
}

template<typename D> __device__ __forceinline__ D to(float v);
template<> __device__ __forceinline__ float to<float>(float v) { return v; }
template<> __device__ __forceinline__ __half to<__half>(float v) { return __float2half_rn(v); }

// a piece's 32 values, scaled
template<int T, typename D> __device__ __forceinline__ void expand(const uint8_t* blk, int s, D* out) {
    Sub p;
    nsq::sub<T>(blk, s, p);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float sc = i < 4 ? p.s0 : p.s1, of = i < 4 ? p.m0 : p.m1;
#pragma unroll
        for (int k = 0; k < 4; ++k) out[4 * i + k] = to<D>(sc * (float) (int8_t) (p.v[i] >> (8 * k)) + of);
    }
}

// n values to D: a thread a piece of 32
template<int T, typename D>
__global__ void expand_k(const uint8_t* __restrict__ src, int64_t nsub, D* __restrict__ dst) {
    const int64_t j = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= nsub) return;
    constexpr int spb = Fmt<T>::qk / 32;
    expand<T, D>(src + (j / spb) * Fmt<T>::bytes, (int) (j % spb), dst + j * 32);
}

// rows of an embedding table: token t's row (row_bytes apart) to float32, a thread a piece of 32
template<int T>
__global__ void embed_k(const uint8_t* __restrict__ table, size_t row_bytes, const int32_t* __restrict__ tokens, int64_t n_embd, float* __restrict__ out) {
    const int64_t j = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (j >= n_embd / 32) return;
    constexpr int spb = Fmt<T>::qk / 32;
    expand<T, float>(table + (size_t) tokens[t] * row_bytes + (j / spb) * Fmt<T>::bytes, (int) (j % spb), out + (size_t) t * n_embd + j * 32);
}

__global__ void embed_bf16_k(const uint16_t* __restrict__ table, size_t row_elems, const int32_t* __restrict__ tokens, int64_t n_embd, float* __restrict__ out) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (i < n_embd) out[(size_t) t * n_embd + i] = __uint_as_float(((uint32_t) table[(size_t) tokens[t] * row_elems + i]) << 16);
}

// F32 / F16 / BF16 to float32, one thread a value
__global__ void to_f32_k(int type, const void* src, int64_t n, float* dst) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (type == 0) dst[i] = ((const float*) src)[i];
    else if (type == 1) dst[i] = __half2float(((const __half*) src)[i]);
    else dst[i] = __uint_as_float(((uint32_t) ((const uint16_t*) src)[i]) << 16);   // BF16
}

__global__ void to_half_k(const float* x, __half* y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half_rn(x[i]);
}

bool known(int t) { return nsq::fmt_of(t).x > 0; }

unsigned blocks_for(int64_t n, int per) { return (unsigned) ((n + per - 1) / per); }

template<int T>
void mmvq_launch(const uint8_t* w, const Q8_1* x, float* y, int n_in, int n_out, int ncols, cudaStream_t q) {
    const size_t rb = (size_t) (n_in / Fmt<T>::qk) * Fmt<T>::bytes;
    const dim3 grid(blocks_for(n_out, 4)), block(32, 4);
    const int nsub = n_in / 32;
    for (int c0 = 0; c0 < ncols; c0 += 8) {
        const int nc = ncols - c0 < 8 ? ncols - c0 : 8;
        const Q8_1* xc = x + (size_t) c0 * nsub;
        float* yc = y + (size_t) c0 * n_out;
        if (nc == 1) mmvq_k<T, 1><<<grid, block, 0, q>>>(w, rb, xc, yc, n_in, n_out, nc);
        else if (nc == 2) mmvq_k<T, 2><<<grid, block, 0, q>>>(w, rb, xc, yc, n_in, n_out, nc);
        else if (nc <= 4) mmvq_k<T, 4><<<grid, block, 0, q>>>(w, rb, xc, yc, n_in, n_out, nc);
        else mmvq_k<T, 8><<<grid, block, 0, q>>>(w, rb, xc, yc, n_in, n_out, nc);
    }
}

}  // namespace

extern "C" {

int ns_nh_to_half(ns_gpu* g, const float* x, void* y, int64_t n) {
    NS_TRY
    NS_CU(cudaSetDevice(g->dev));
    if (n > 0) to_half_k<<<blocks_for(n, 256), 256, 0, g->q>>>(x, (__half*) y, n);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

size_t ns_nh_row_bytes(int type, int64_t n) {
    if (type == BF16) return (size_t) n * 2;
    const int2 f = nsq::fmt_of(type);
    return f.x ? (size_t) (n / f.x) * f.y : 0;
}

int ns_nh_mmvq_supported(int type) { return known(type) ? 1 : 0; }

int ns_nh_dequant_f32(ns_gpu* g, int type, const void* src, int64_t n, float* dst) {
    NS_TRY
    NS_CU(cudaSetDevice(g->dev));
    if (n <= 0) return 0;
    if (type == 0 || type == 1 || type == BF16) {
        to_f32_k<<<blocks_for(n, 256), 256, 0, g->q>>>(type == BF16 ? 2 : type, src, n, dst);
        NS_LAUNCH_CHECK();
        return 0;
    }
    if (!known(type) || n % nsq::fmt_of(type).x != 0) return ns_fail("ns_nh_dequant_f32: type " + std::to_string(type) + " is not one read here, or n not whole blocks");
    const int64_t nsub = n / 32;
    switch (type) {
#define NH_EXP(T) case T: expand_k<T, float><<<blocks_for(nsub, 128), 128, 0, g->q>>>((const uint8_t*) src, nsub, dst); break;
        NSQ_TYPES(NH_EXP)
#undef NH_EXP
    }
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_nh_dequant_f16(ns_gpu* g, int type, const void* src, int64_t n, void* dst) {
    NS_TRY
    NS_CU(cudaSetDevice(g->dev));
    if (n <= 0) return 0;
    if (!known(type) || n % nsq::fmt_of(type).x != 0) return ns_fail("ns_nh_dequant_f16: type " + std::to_string(type) + " is not one read here, or n not whole blocks");
    const int64_t nsub = n / 32;
    switch (type) {
#define NH_EXP(T) case T: expand_k<T, __half><<<blocks_for(nsub, 128), 128, 0, g->q>>>((const uint8_t*) src, nsub, (__half*) dst); break;
        NSQ_TYPES(NH_EXP)
#undef NH_EXP
    }
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

size_t ns_nh_q8_1_bytes(int64_t C, int64_t T) {
    return (size_t) (C / 32) * sizeof(Q8_1) * (size_t) T;
}

int ns_nh_quantize(ns_gpu* g, const float* x, int64_t T, int64_t C, void* out) {
    NS_TRY
    NS_CU(cudaSetDevice(g->dev));
    if (C % 32 != 0) return ns_fail("ns_nh_quantize: the columns a multiple of 32");
    const int64_t nblk = T * C / 32;
    if (nblk > 0) q8_1_k<<<blocks_for(nblk * 32, 256), 256, 0, g->q>>>(x, (Q8_1*) out, nblk);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_nh_mmvq(ns_gpu* g, int type, const void* w, const void* xq, float* y, int64_t n_in, int64_t n_out, int64_t ncols) {
    NS_TRY
    NS_CU(cudaSetDevice(g->dev));
    if (!known(type)) return ns_fail("ns_nh_mmvq: type " + std::to_string(type) + " is not one the products take");
    if (n_in % nsq::fmt_of(type).x != 0) return ns_fail("ns_nh_mmvq: n_in not whole blocks");
    if (ncols <= 0 || n_out <= 0) return 0;
    const auto* W = (const uint8_t*) w;
    const auto* X = (const Q8_1*) xq;
    switch (type) {
#define NH_MV(T) case T: mmvq_launch<T>(W, X, y, (int) n_in, (int) n_out, (int) ncols, g->q); break;
        NSQ_TYPES(NH_MV)
#undef NH_MV
    }
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_nh_mmvq_ptrs(ns_gpu* g, int type, const void* wptrs, const int32_t* xrow, const void* xq, float* y, int64_t n_in, int64_t n_out, int64_t n_entries) {
    NS_TRY
    NS_CU(cudaSetDevice(g->dev));
    if (n_entries <= 0) return 0;
    if (!known(type)) return ns_fail("ns_nh_mmvq_ptrs: type " + std::to_string(type) + " is not one the products take");
    const size_t rb = ns_nh_row_bytes(type, n_in);
    const dim3 grid(blocks_for(n_out, 4), (unsigned) n_entries), block(32, 4);
    const auto* wp = (const unsigned long long*) wptrs;
    switch (type) {
#define NH_PT(T) case T: mmvq_ptrs_k<T><<<grid, block, 0, g->q>>>(wp, rb, xrow, (const Q8_1*) xq, y, (int) n_in, (int) n_out); break;
        NSQ_TYPES(NH_PT)
#undef NH_PT
    }
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

int ns_nh_embed(ns_gpu* g, int type, const void* table, int64_t row_bytes, const int32_t* tokens, int64_t n_tok, int64_t n_embd, float* out) {
    NS_TRY
    NS_CU(cudaSetDevice(g->dev));
    if (n_tok <= 0) return 0;
    if (type == BF16) {
        embed_bf16_k<<<dim3(blocks_for(n_embd, 256), (unsigned) n_tok), 256, 0, g->q>>>((const uint16_t*) table, (size_t) row_bytes / 2, tokens, n_embd, out);
        NS_LAUNCH_CHECK();
        return 0;
    }
    if (!known(type) || n_embd % nsq::fmt_of(type).x != 0) return ns_fail("ns_nh_embed: type " + std::to_string(type) + " is not one the embedding takes");
    const dim3 grid(blocks_for(n_embd / 32, 128), (unsigned) n_tok);
    switch (type) {
#define NH_EM(T) case T: embed_k<T><<<grid, 128, 0, g->q>>>((const uint8_t*) table, (size_t) row_bytes, tokens, n_embd, out); break;
        NSQ_TYPES(NH_EM)
#undef NH_EM
    }
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

}  // extern "C"
