/* zimage.h: what Z-Image's single-stream DiT needs beside the shared diffusion kernels (nsd.h) - its modulated norms,
 * its tanh-gated residuals, its activations scaled into half's range, the sampler's step. The DiT is the foundation's
 * (crates/zimage) because two image engines run it: Ming-Image (image/mingimage) and Z-Image-Turbo (image/zimage);
 * linked, like the other diffusion kernels, into every kind's library and bound by name by that crate. 16-bit buffers
 * are IEEE half; every call is queued on g's queue (the caller syncs) and returns 0, or -1 with ns_last_error(). */
#ifndef NS_ZIMAGE_H
#define NS_ZIMAGE_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* out [M, C] half = layernorm(x [M, C] float32, no affine, eps) * (1 + scale[c]); scale float32 [C] or NULL: the DiT's
 * final layer. One block a row (C at most 8,192). */
int ns_zimage_ln_mod(ns_gpu* g, const float* x, int64_t M, int64_t C, float eps, const float* scale, void* out);
/* out [M, C] half = x [M, C] float32 * (1 + scale[c]) after an rms norm with weight w (float32 [C]), eps: the DiT's
 * modulated input norm, and with scale NULL a plain rms norm to half */
int ns_zimage_rms_mod(ns_gpu* g, const float* x, int64_t M, int64_t C, const float* w, float eps, const float* scale, void* out);
/* x [M, C] float32 += other [M, C] half * gate[c] (gate float32 [C]; NULL: 1) - the DiT's tanh-gated residuals */
int ns_zimage_gate_add(ns_gpu* g, float* x, int64_t M, int64_t C, const void* other, const float* gate);
/* The gated activation, scaled down: out[r, i] = silu(x[r, i]) * x[r, C + i] * s (x [M, 2C], out [M, C] half; float32
 * inside). The product reaches past half's range in the DiT's MLPs (65,504; a layer makes infinities): scaled by 1/256
 * here, the next linear's float32 output is multiplied back (nsd_scale) */
int ns_zimage_swiglu_scaled(ns_gpu* g, const void* x, int64_t M, int64_t C, float s, void* out);
/* out [M, C] half, rows `ld` halves apart = x [M, C] float32 * s: a projection's output put in its place of a wider
 * buffer (the DiT's q | k | v), scaled into half's range */
int ns_zimage_to_half_rows(ns_gpu* g, const float* x, int64_t M, int64_t C, float s, void* out, int64_t ld);
/* out float32 = x half (n values) */
int ns_zimage_to_float(ns_gpu* g, const void* x, float* out, int64_t n);
/* x float32 += a * y float32 (n values): the sampler's step */
int ns_zimage_axpy(ns_gpu* g, float* x, const float* y, int64_t n, float a);

#ifdef __cplusplus
}
#endif
#endif
