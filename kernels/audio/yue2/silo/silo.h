// silo.h: YuE2's silo - its tuned kernels, built once a card (per-card), opened by the engine on its card
// (audio/yue2/src/ffi.rs). On the GPU's stream.
#pragma once
#include <stdint.h>
#include "ns.h"

#ifdef __cplusplus
extern "C" {
#endif

/* this build: the card's compute capability (86, 120) */
int ns_audio_yue2_silo_info(int* sm);
/* attention, head size 128, half in and out: S query rows of stride qs over Skv key / value rows of stride kvs, H query heads
 * over Hkv key / value heads (query head h reads key head h / (H / Hkv)), out [S, H * 128]; warps 0 = the default (8: 128
 * queries a block) or 4; half_scores 1: the scores accumulated in half (twice the rate on GeForce), 0: in float32 */
int ns_audio_yue2_silo_attention(ns_gpu* g, const void* q, int64_t S, int64_t qs, const void* k, const void* v, int64_t Skv, int64_t kvs,
                                 int64_t H, int64_t Hkv, void* out, int warps, int half_scores);

#ifdef __cplusplus
}
#endif
