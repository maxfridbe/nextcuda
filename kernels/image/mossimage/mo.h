/* mo.h: the MOSS-Image engine's own kernels (kernels/image/mossimage), beside the shared diffusion kernels (nsd.h)
 * and Ming-Image's small ones it shares (mi.h). Bound by its crate by name (image/mossimage/src/ffi.rs). 16-bit
 * buffers are IEEE half; every call is queued on the GPU's queue (the caller syncs) and returns 0, or -1 with the
 * reason in ns_last_error(). */
#ifndef NS_IMAGE_MO_H
#define NS_IMAGE_MO_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* MOSS's clipped-gate activation, the DiT's and its text blocks' MLP: out [M, F] half =
 *   (4 tanh(g / 4) sigmoid(g)) * (25 tanh(h / 25)),  h = x[r, c], g = x[r, F + c]   (x [M, 2F] float32, rows `ld` apart)
 * bounded by 100 whatever x is: half holds it */
int ns_image_mo_situ(ns_gpu* g, const float* x, int64_t M, int64_t F, int64_t ld, void* out);
/* A group norm with affine weight / bias (float32 [C]), no activation: out [P, C] half of x [P, C] half (channels
 * last), G groups of C / G channels, statistics over every pixel of a group in float32 - the VAE's attention norm */
int ns_image_mo_group_norm(ns_gpu* g, const void* x, int64_t P, int64_t C, int64_t G, const float* w, const float* b, float eps, void* out);

#ifdef __cplusplus
}
#endif
#endif
