/* ph3.h: P-H3-Image's own kernels (kernels/image/ph3image): the deblock filter run on a decoded picture. Bound by name
 * by image/ph3image/src/deblock.rs, which holds the host reference and the steps between them. A plane is one colour
 * channel, float32 [H, W]; every call is queued on g's queue (the caller syncs) and returns 0, or -1 with the reason
 * in ns_last_error(). */
#ifndef NS_IMAGE_PH3_H
#define NS_IMAGE_PH3_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct ns_gpu ns_gpu;
/* hp = x - the mean of x over the (2r + 1) x (2r + 1) square around each pixel, edges replicated; tmp [H, W] is the
 * rows' pass */
int ns_image_ph3_highpass(ns_gpu* g, const float* x, int64_t H, int64_t W, int64_t r, float* tmp, float* hp);
/* per tile t (row-major, ceil(H/16) x ceil(W/16), a last partial tile over its pixels): energy[t] = the sum of hp^2;
 * dots[t] = the sum of hp * pat[position] (0 when pat is NULL) */
int ns_image_ph3_tile_stats(ns_gpu* g, const float* hp, int64_t H, int64_t W, const float* pat, float* energy, float* dots);
/* over the whole 16 x 16 tiles of hp, tile t weighing wts[t] (indexed as tile_stats'), at each of the 256 positions in
 * a tile: sums[2 p] the sum of w hp, sums[2 p + 1] of w hp^2 (p = row * 16 + column) */
int ns_image_ph3_tile_sums(ns_gpu* g, const float* hp, int64_t H, int64_t W, const float* wts, float* sums);
/* x -= a[tile] * pat[position], every pixel */
int ns_image_ph3_subtract(ns_gpu* g, float* x, int64_t H, int64_t W, const float* a, const float* pat);
/* the boundary pass over the 3 planes of x [3, H, W]: at each 16-pixel tile boundary (vertical ones when vertical,
 * else horizontal), on every line across it, the step left beyond the two sides' slopes spread as a ramp over n
 * pixels each side, by strength * (1 - activity / beta) * (1 - step / alpha) measured on the luma (0 at either) */
int ns_image_ph3_edges(ns_gpu* g, float* x, int64_t H, int64_t W, int vertical, int n, float alpha, float beta, float strength);
#ifdef __cplusplus
}
#endif
#endif
