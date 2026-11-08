/* zi.h: the Z-Image-Turbo engine's own kernels (kernels/image/zimage) - what its VAE decoder (FLUX's 16-channel
 * AutoencoderKL) needs beside the shared diffusion kernels (nsd.h); the DiT's are the foundation's (zimage.h). Bound by
 * its crate by name (image/zimage/src/ffi.rs). 16-bit buffers are IEEE half, pictures channels-last; every call is
 * queued on the GPU's queue (the caller syncs) and returns 0, or -1 with the reason in ns_last_error(). */
#ifndef NS_IMAGE_ZI_H
#define NS_IMAGE_ZI_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* out [P, C] half = group norm of x [P, C] half (G groups over all P pixels, statistics in double), * w + b (float32
 * [C], or NULL): the middle attention's input norm (the resnets' norms are nsd_group_norm_silu) */
int ns_image_zi_group_norm(ns_gpu* g, const void* x, int64_t P, int64_t C, int64_t G, const float* w, const float* b, float eps, void* out);
/* out [2H, 2W, C] half = nearest 2x of x [H, W, C] half: the up blocks' upsampling before their convolution */
int ns_image_zi_up2(ns_gpu* g, const void* x, int64_t H, int64_t W, int64_t C, void* out);

#ifdef __cplusplus
}
#endif
#endif
