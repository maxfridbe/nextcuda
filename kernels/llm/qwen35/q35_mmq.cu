// q35_mmq.cu: the prompt passes' and decode rows' products under q35.h's C ABI (ns_q35_mmq_*, ns_q35_mmvq) - nextcuda's
// own: the prompt products are q35_tiles.cu's (x quantized a scale a 128 values into its own layout, the matrices'
// blocks multiplied on the int8 tensor cores), the decode rows' nemotronh's ns_nh_mmvq. These entry points were
// ggml-cuda's MMQ and MMVQ (vendored from llama.cpp, MIT) until 2026-11-14; the engine and the other models that
// borrow the products call them by the same names. The types Qwen3.8-27B GSQ-RCO's file holds (IQ1_M, one tensor,
// is not one: the engine expands it to half) and Q8_0 (the DSpine draft's layers, made Q8_0 at load)
#include "ns.h"
#include "ns_internal.hpp"
#include "q35.h"
#include "nh.h"

#include <string>

extern "C" {

// x's layout for a type's prompt products (-1: none): matrices of one layout share it
int ns_q35_mmq_layout(int type) { return ns_q35_tiles_layout(type); }

int ns_q35_mmq_supported(int type) {
    int dev = 0, major = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) return 0;
    // the int8 tensor-core products (mma m16n8k32): Ampere and later (both cards here: sm_86, sm_120)
    return ns_q35_tiles_layout(type) > 0 && major >= 8 ? 1 : 0;
}

size_t ns_q35_mmq_q8_bytes(int64_t K, int64_t T) { return ns_q35_tiles_q8_bytes(K, T); }

int ns_q35_mmq_quantize(ns_gpu* g, int type, const float* x, int64_t T, int64_t K, void* xq) { return ns_q35_tiles_quantize(g, type, x, T, K, xq); }

int ns_q35_mmq(ns_gpu* g, int type, const void* w, int64_t n_out, int64_t K, const void* xq, int64_t T, float* y) {
    return ns_q35_tiles_mmq(g, type, w, n_out, K, xq, T, y);
}

// the products' room reserved now (inside a CUDA graph's capture an allocation fails)
int ns_q35_mmq_reserve(ns_gpu* g) { return ns_q35_tiles_reserve(g); }

int ns_q35_mmvq_supported(int type) { return ns_nh_mmvq_supported(type); }

// y [ncols][n_out] = W [n_out][n_in] . xq [ncols] (xq: ns_nh_quantize's Q8_1 rows), 1..8 rows
int ns_q35_mmvq(ns_gpu* g, int type, const void* w, const void* xq, float* y, int64_t n_in, int64_t n_out, int64_t ncols) {
    if (ncols < 1 || ncols > 8) return ns_fail("ns_q35_mmvq: 1..8 rows");
    return ns_nh_mmvq(g, type, w, xq, y, n_in, n_out, ncols);
}

}  // extern "C"
