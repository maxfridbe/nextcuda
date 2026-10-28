/* silo.h: UniMate's silo (dist/silo/libnextcuda-unimate.so): its kernels tuned for its shapes, each with a _supported
 * check; the engine (model3d/unimate/src/ffi.rs) calls them for what they cover and the engine's own kernels (unimate.h)
 * otherwise (NS_SILO=0: never). */
#ifndef NS_MODEL3D_UNIMATE_SILO_H
#define NS_MODEL3D_UNIMATE_SILO_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* Whether ns_model3d_unimate_silo_attention takes these shapes (1) or the engine's own kernel must (0): heads of 32 or 64,
 * at most 128 queries and 128 keys a group */
int ns_model3d_unimate_silo_attention_supported(int64_t D, int64_t Sq, int64_t Skv);
/* ns_model3d_unimate_attention (unimate.h), the same arguments and result: a block a (group, head), its keys and values in
 * shared memory, a thread a query (its scores by flash attention's online softmax, in registers) */
int ns_model3d_unimate_silo_attention(ns_gpu* g, const float* q, int64_t q_row, const float* k, int64_t k_row, const float* v, int64_t v_row,
                                      int64_t G, int64_t Sq, int64_t Skv, int64_t H, int64_t D, int64_t q_gs, int64_t q_ss, int64_t kv_gs,
                                      int64_t kv_ss, int64_t nvalid, float scale, const float* bias, float* out, int64_t out_row);

#ifdef __cplusplus
}
#endif
#endif
