/* lx.h: the LTX-2.5 engine's own kernels (kernels/video/ltx2), beside the shared diffusion kernels (nsd.h). Bound by its
 * crate by name (video/ltx2/src/ffi.rs). 16-bit buffers are IEEE half (__half); every call is queued on the GPU's queue
 * (the caller syncs) and returns 0, or -1 with the reason in ns_last_error(). */
#ifndef NS_VIDEO_LX_H
#define NS_VIDEO_LX_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* out[r * ldo + c] half = x[r * ldx + c] * s, x float32 [M, C] rows ldx apart */
int ns_video_lx_to_half_rows(ns_gpu* g, const float* x, int64_t M, int64_t C, int64_t ldx, float s, void* out, int64_t ldo);
/* out float32 = x half * s (n values) */
int ns_video_lx_to_float(ns_gpu* g, const void* x, float* out, int64_t n, float s);
/* out [M, C] half = gelu_tanh(a[r, c]) * b[r, c] * s - a, b float32 [M, C] rows lda / ldb apart (b NULL: no product) */
int ns_video_lx_geglu(ns_gpu* g, const float* a, int64_t lda, const float* b, int64_t ldb, int64_t M, int64_t C, float s, void* out);
/* x float32 += a * y float32 (n values) */
int ns_video_lx_axpy(ns_gpu* g, float* x, const float* y, int64_t n, float a);
/* x [M, H * D] half, in place: head h of row r times 2 sigmoid(logits[r * H + h]) (logits float32 [M, H]) */
int ns_video_lx_head_gate(ns_gpu* g, void* x, int64_t M, int64_t H, int64_t D, const float* logits);
/* out [M, C] half = (x[r, c] * (1 + scale[c]) + shift[c]) * s, x float32 [M, C] (scale / shift float32 [C], NULL: 0) */
int ns_video_lx_mod_rows(ns_gpu* g, const float* x, int64_t M, int64_t C, const float* scale, const float* shift, float s, void* out);
/* out [M, C] float32 = layernorm(x [M, C] float32, no affine, eps) * (1 + scale[rows[r]]) + shift[rows[r]]: the DiT's
 * output norm (scale, shift float32 [R, C]; rows int32 [M], NULL: all row 0). One block a row (C at most 8,192). */
int ns_video_lx_ln_mod(ns_gpu* g, const float* x, int64_t M, int64_t C, float eps, const int32_t* rows, const float* scale,
                       const float* shift, float* out);

/* The video VAE's: out [M, C] half = silu?(x / sqrt(mean(x^2) + eps)) per row (PerChannelRMSNorm, then SiLU when silu) -
 * x half [M, C]; one block a row */
int ns_video_lx_rms_silu(ns_gpu* g, const void* x, int64_t M, int64_t C, float eps, int silu, void* out);
/* depth to space, channels-last: x half [T, H, W, C * s0 * s1 * s2] (channel c * s0 s1 s2 + i s1 s2 + j s2 + k) ->
 * out [T s0 - drop, H s1, W s2, C] (out[t s0 + i - drop, h s1 + j, w s2 + k, c]; the first `drop` frames dropped) */
int ns_video_lx_d2s(ns_gpu* g, const void* x, int64_t T, int64_t H, int64_t W, int64_t C, int s0, int s1, int s2, int drop, void* out);

#ifdef __cplusplus
}
#endif
#endif
