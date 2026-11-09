/* svg.h: the vector rasterizer's kernels (kernels/image/svgrender) - DiffVG's prefiltered rendering of closed and open
 * cubic Bezier paths (fill by winding number, stroke by distance, coverage smoothed over a pixel by the signed distance
 * to the outline), its gradients to every control point, colour and stroke width, and what LIVE's optimiser runs
 * beside it (its unsigned-distance loss weight, the self-intersection loss, Adam). Bound by name from
 * image/svgrender/src/ffi.rs. Every call is queued on the GPU's queue (the caller syncs) and returns 0, or -1 with the
 * reason in ns_last_error().
 *
 * A scene: N paths; path i's points are pts[2 * off_i ..] (canvas pixels, x then y), nseg_i cubic segments of 3 points
 * each (an open path has one more, its end), its fill fill[4 i ..] and stroke stroke[4 i ..] (RGBA, straight alpha)
 * and stroke half-width hw[i] (DiffVG's stroke_width: the SVG's stroke-width is twice it). Paths are drawn in order,
 * each fill then its stroke (as SVG draws them). */
#ifndef NS_IMAGE_SVG_H
#define NS_IMAGE_SVG_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* a path's shape: where its points start, its segments, and flags */
typedef struct {
    int32_t off;
    int32_t nseg;
    int32_t flags;   /* NS_SVG_* */
    int32_t pad;
} ns_svg_path;

#define NS_SVG_CLOSED 1
#define NS_SVG_FILL 2
#define NS_SVG_STROKE 4
#define NS_SVG_EVENODD 8
/* cull the path at its box as DiffVG does (no pixel of filter margin outside it): its prefiltered edges, for the check */
#define NS_SVG_DIFFVG_BOX 32

/* box [4 N]: each path's control points' bounds (x0, y0, x1, y1), widened by its stroke's half-width when it has one */
int ns_image_svg_bounds(ns_gpu* g, int32_t N, const ns_svg_path* paths, const float* pts, const float* hw, float* box);

/* out [H, W, 4] float: the scene's pixels. samples 0: prefiltered (DiffVG's use_prefiltering: one sample at the
 * pixel's centre, coverage smoothstep(signed distance)); s >= 1: s x s hard samples a pixel on a regular grid (a box
 * filter). bg (RGB) or NULL: with it the pixels are composited over it (alpha 1), else straight RGBA as DiffVG returns.
 * first: paths before it are not drawn; mono 1: every path drawn opaque white (LIVE's coverage of the newest layer) */
int ns_image_svg_render(ns_gpu* g, int32_t W, int32_t H, int32_t N, const ns_svg_path* paths, const float* pts, const float* fill,
                        const float* stroke, const float* hw, const float* box, const float* bg, int32_t samples, int32_t first,
                        int32_t mono, float* out);

/* One optimiser step's rendering and backward pass, prefiltered (the gradients DiffVG's prefiltered mode gives).
 * mode 0 (LIVE): the pixels composited over bg (RGB), loss = mean over pixels of weight[p] * sum_c (pixel - target)^2
 *   (target [H, W, 3]; weight [H, W] or NULL: 1) added to *loss; the gradients of that.
 * mode 1 (a check against DiffVG): straight RGBA pixels (no background), d_img [H, W, 4] the loss's gradient to them.
 * The gradients are added to d_pts [2 P], d_fill [4 N], d_stroke [4 N], d_hw [N]. out [H, W, 4] or NULL: the pixels. */
int ns_image_svg_grad(ns_gpu* g, int32_t W, int32_t H, int32_t N, const ns_svg_path* paths, const float* pts, const float* fill,
                      const float* stroke, const float* hw, const float* box, const float* bg, int32_t mode, const float* target,
                      const float* weight, const float* d_img, float* d_pts, float* d_fill, float* d_stroke, float* d_hw, float* loss,
                      float* out);

/* LIVE's unsigned-distance weight (its calc_distance_weight / get_sdf): cov [H, W, 4] the newest layer's coverage
 * (ns_image_svg_render's mono render: its alpha, 0..1); phi = 2 cov - 1; the distance of each pixel to phi's zero level (to the nearest pixel of the other sign,
 * less half a pixel), at most 10; weight = clip((max - d) / max + keep, 0, 1) (keep [H, W] or NULL). A coverage all
 * of one sign gives keep (or 0). scratch: 4 ints */
int ns_image_svg_udf(ns_gpu* g, int32_t W, int32_t H, const float* cov, const float* keep, float* weight, float* dist, int32_t* scratch);

/* LIVE's xing loss (xing_loss_fn) over paths [first, N) - each closed path's segments' first and third control legs
 * turning against the first and second's turn - times scale / (N - first), added to *loss; its gradient to d_pts */
int ns_image_svg_xing(ns_gpu* g, int32_t N, int32_t first, const ns_svg_path* paths, const float* pts, float scale, float* d_pts,
                      float* loss);

/* Adam over n parameters p with gradients d (zeroed after), moments m and v. Each parameter's group grp[i] (0 points,
 * 1 fill, 2 stroke colour, 3 stroke half-width; 255: frozen) and owner path own[i]: its learning rate is lr[grp] times
 * LIVE's LinearDecayLR(decay_every, ratio) at the owner's own step (step - birth[own] - each layer's optimiser counts
 * from when the layer was added), and Adam's bias correction counts from there too. Then each group's clamp lo..hi.
 * lr4, lo4, hi4: host arrays of 4 */
int ns_image_svg_adam(ns_gpu* g, int64_t n, float* p, float* d, float* m, float* v, const uint8_t* grp, const int32_t* own,
                      const int32_t* birth, int32_t step, const float* lr4, int32_t decay_every, float ratio, float b1, float b2,
                      float eps, const float* lo4, const float* hi4);

/* out u8 [H, W, 3]: float pixels [H, W, 4] (alpha ignored; composited) to bytes */
int ns_image_svg_to_u8(ns_gpu* g, int32_t W, int32_t H, const float* px, uint8_t* out);

#ifdef __cplusplus
}
#endif
#endif
