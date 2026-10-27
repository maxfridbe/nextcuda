// silo.h: MOSS-Image's silo - its tuned kernels, built once a card (per-card), opened by the engine on each card
// (image/mossimage/src/ffi.rs). On the GPU's stream.
#pragma once
#include <stdint.h>
#include "ns.h"

#ifdef __cplusplus
extern "C" {
#endif

/* this build: the card's compute capability (86, 120) and whether FP8 attention is in it */
int ns_image_moss_silo_info(int* sm, int* fp8);
/* attention, head size 128, half in and out: S query rows of stride qs over Skv key / value rows of stride kvs, H heads,
 * out [S, H * 128]; warps 0 = the default (8: 128 queries a block), or 4; fp8 -1 = the scores accumulated in half,
 * 0 = in float32, 1 = from FP8 copies of q and k (each row scaled to e4m3), 2 = the values too (FP8: sm_89 and up) */
int ns_image_moss_silo_attention(ns_gpu* g, const void* q, int64_t S, int64_t qs, const void* k, const void* v, int64_t Skv, int64_t kvs,
                                 int64_t H, void* out, int warps, int fp8);
/* the int8 products' epilogues (nsd_int8_product's acc [M, N], xs [M]; wscale [N]):
 * half:     out[r, c] (row stride ldo) = acc * xs[r] * ws[c] * alpha, half
 * situ:     out[r, c] [M, F] half = 4 tanh(g / 4) sigmoid(g) * 25 tanh(h / 25), h = column c, g = column F + c
 * gate_add: x[r, c] += gate[c] * acc * xs[r] * ws[c] * alpha, x float32 [M, N] */
int ns_image_moss_silo_epi_half(ns_gpu* g, const int32_t* acc, int64_t M, int64_t N, const float* xs, const float* ws, float alpha, void* out,
                                int64_t ldo);
int ns_image_moss_silo_epi_situ(ns_gpu* g, const int32_t* acc, int64_t M, int64_t F, const float* xs, const float* ws, void* out);
int ns_image_moss_silo_epi_gate_add(ns_gpu* g, const int32_t* acc, int64_t M, int64_t N, const float* xs, const float* ws, float alpha,
                                    const float* gate, float* x);

#ifdef __cplusplus
}
#endif
