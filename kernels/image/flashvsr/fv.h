/* fv.h: the FlashVSR engine's own kernels (kernels/image/flashvsr), beside the shared diffusion kernels (nsd.h). Bound
 * by its crate by name (image/flashvsr/src/ffi.rs). 16-bit buffers are IEEE half (__half); every call is queued on the
 * GPU's stream (the caller syncs) and returns 0, or -1 with the reason in ns_last_error().
 *
 * The DiT's tokens are kept in window order: a chunk's latent frames in pairs (a "group"), each group's 2 x h x w
 * patch grid cut into blocks of 2 x 8 x 8 = 128 tokens - row-major over the blocks, then (frame, y, x) inside one -
 * which is the order FlashVSR's window partition and block-sparse attention read them in. */
#ifndef NS_IMAGE_FV_H
#define NS_IMAGE_FV_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* patches: lat float32 [16, f, h8, w8] (f even, h8 and w8 multiples of 16) -> out half [L, 64] in window order, a
 * token's features (c, ky, kx) as the patch embedding's Conv3d weight [1536, 16, 1, 2, 2] flattens them */
int ns_image_fv_patchify(ns_gpu* g, const float* lat, int64_t f, int64_t h8, int64_t w8, void* out);
/* the head's output back to latents: out [16, f, h8, w8] = noise - unpatchify(pred [L, 64] float32, window order; a
 * token's features (ky, kx, c), as Wan's unpatchify reads them) */
int ns_image_fv_unpatchify_sub(ns_gpu* g, const float* pred, const float* noise, int64_t f, int64_t h8, int64_t w8, float* out);
/* x [L, C] float32 (window order) += lq [f * hp * wp, C] float32 (natural (frame, y, x) order) */
int ns_image_fv_add_lq(ns_gpu* g, float* x, const float* lq, int64_t f, int64_t hp, int64_t wp, int64_t C);
/* out [M, C] half = (x - mean) / sqrt(var + eps) * a[c] + b[c] per row of x float32 (a, b float32 [C]; NULL: 1, 0) */
int ns_image_fv_ln(ns_gpu* g, const float* x, int64_t M, int64_t C, float eps, const float* a, const float* b, void* out);
/* x [M, C] float32 += y [M, C] float32 * gate[c] (gate NULL: 1) */
int ns_image_fv_gate_add(ns_gpu* g, float* x, const float* y, int64_t M, int64_t C, const float* gate);
/* out [n, C] float32: the mean of each block of 128 rows of x half (rows `stride` halves apart) */
int ns_image_fv_block_mean(ns_gpu* g, const void* x, int64_t n, int64_t stride, int64_t C, float* out);
/* The draft attention between blocks: probs [H, nq, nk] = softmax over j of (qm[i] . km[j] per head of 128) / sqrt(128),
 * key blocks outside the query block's local window (lr x lr blocks of the nbh x nbw grid, block index mod nb) at
 * -inf. qm [nq, H * 128], km [nk, H * 128] float32. */
int ns_image_fv_block_probs(ns_gpu* g, const float* qm, const float* km, int64_t nq, int64_t nk, int64_t H, int64_t nbh, int64_t nbw, int64_t lr,
                            float* probs);
/* thr[s] = the rank-th largest of the n values of segment s of p float32 [S, n] (values >= 0); rank from 1 */
int ns_image_fv_topk_thr(ns_gpu* g, const float* p, int64_t S, int64_t n, int64_t rank, float* thr);
/* The selected key blocks: for each head h and query block i, the j with probs[h, i, j] > thr[h * G + i / nb], in
 * order: lists int32 [H, nq, nk], counts int32 [H, nq] (G = nq / nb query groups) */
int ns_image_fv_block_lists(ns_gpu* g, const float* probs, const float* thr, int64_t H, int64_t nq, int64_t nk, int64_t nb, int32_t* lists,
                            int32_t* counts);
/* Block-sparse attention, head size 128, half in and out: query rows q [Sq, H * 128] (query block = row / 128) over the
 * 128-key blocks its list names - block j's keys at kc / vc + j * 128 rows for j < split, else kt / vt + (j - split) * 128
 * rows (rows of H * 128 halves); a row with no block gives 0. out [Sq, H * 128]. */
int ns_image_fv_sparse_attn(ns_gpu* g, const void* q, int64_t Sq, int64_t H, const void* kc, const void* vc, const void* kt, const void* vt,
                            int64_t split, const int32_t* lists, const int32_t* counts, int64_t nk, void* out);
/* frames half [T, H, W, 3] -> out half [T / ff, H / r, W / r, 3 * ff * r * r], channel ((c * ff + t) * r + y) * r + x */
int ns_image_fv_unshuffle(ns_gpu* g, const void* x, int64_t T, int64_t H, int64_t W, int64_t ff, int64_t r, void* out);
/* im2col of a causal 3-D convolution, kernel kt x 3 x 3, stride (st, 1, 1), spatial padding 1 replicated: x half [T, H, W, C]
 * (the temporal padding already in front) -> col half [n, kt * 9 * C] for output pixels [p0, p0 + n) of [To, H, W],
 * columns (dt, dy, dx, c) */
int ns_image_fv_im2col3(ns_gpu* g, const void* x, int64_t T, int64_t H, int64_t W, int64_t C, int64_t kt, int64_t st, int64_t p0, int64_t n,
                        void* col);
/* im2col of a 3 x 3 convolution with zero padding 1 over N frames at H x W (the output's size): columns (dy, dx, c) for
 * c < Ca from a [N, H/u, W/u, lda] (u = 2 when `up`: the source read through a nearest 2x upsample), then for c < Cb
 * from b: frame i's is a's frame i - 1, frame 0's `mem` ([H/u, W/u, Cb], rows of lda; NULL: zeros) - a MemBlock's past.
 * Cb 0: no second source. col half [n, 9 * (Ca + Cb)] for pixels [p0, p0 + n) of [N, H, W]. */
int ns_image_fv_im2col2(ns_gpu* g, const void* a, int64_t lda, int64_t Ca, const void* mem, int64_t Cb, int64_t N, int64_t H, int64_t W, int up,
                        int64_t p0, int64_t n, void* col);
/* x half [M, C] -> out half: x / max(|x|, 1e-12) * sqrt(C) * gamma[c], then SiLU (gamma float32 [C]) */
int ns_image_fv_rms_silu(ns_gpu* g, const void* x, int64_t M, int64_t C, const float* gamma, void* out);
/* x half, n values: max(x, 0) in place; add_relu: x = max(x + y, 0) */
int ns_image_fv_relu(ns_gpu* g, void* x, int64_t n);
int ns_image_fv_add_relu(ns_gpu* g, void* x, const void* y, int64_t n);
/* the tiny decoder's input for one latent frame: out half [h8, w8, 784] = 3 tanh(v / 3) of (the 4 cond frames
 * [4, 8 h8, 8 w8, 3] half unshuffled to 768 channels, then the latent's 16 channels lat[c, t, y, x] of float32
 * [16, f, h8, w8]) */
int ns_image_fv_tcd_input(ns_gpu* g, const void* cond, const float* lat, int64_t f, int64_t t, int64_t h8, int64_t w8, void* out);
/* bytes [n] -> half x / 255 * 2 - 1 */
int ns_image_fv_u8_to_half(ns_gpu* g, const uint8_t* x, int64_t n, void* out);
/* float32 <-> half, n values */
int ns_image_fv_to_half(ns_gpu* g, const float* x, int64_t n, void* out);
int ns_image_fv_to_float(ns_gpu* g, const void* x, int64_t n, float* out);
/* Colour statistics: sums double [6] (per channel the sum and the sum of squares) of v over the box (x0, y0, w, h) of
 * an [H, W, 3] picture, v = x * a + b (x half); accumulated into sums (zero it first) */
int ns_image_fv_stats(ns_gpu* g, const void* x, int64_t W, int64_t x0, int64_t y0, int64_t w, int64_t h, float a, float b, double* sums);
/* The colour fix and the bytes: for the box of dec half [H, W, 3] (the decoder's [0, 1]-ish output), v = dec * 2 - 1,
 * y = clamp((v - mean_d) / std_d * std_l + mean_l, -1, 1) per channel (ms float32 [12]: mean_d, std_d, mean_l, std_l
 * for each channel; NULL: y = clamp(v)), out bytes [h, w, 3] = trunc(clamp((y + 1) * 127.5, 0, 255)) */
int ns_image_fv_to_u8(ns_gpu* g, const void* dec, int64_t W, int64_t x0, int64_t y0, int64_t w, int64_t h, const float* ms, uint8_t* out);
/* out float32 [n]: standard normal numbers from (seed, the index + offset) - a counter-based generator, so a chunk's
 * noise is the same whatever came before */
int ns_image_fv_randn(ns_gpu* g, float* out, int64_t n, uint64_t seed, uint64_t offset);

#ifdef __cplusplus
}
#endif
#endif
