// svg.cu: the vector rasterizer (svg.h) - DiffVG's prefiltered rendering and its backward pass, one thread a pixel,
// and LIVE's loss weight, xing loss and Adam. The geometry follows DiffVG (diffvg.cpp, winding_number.h,
// compute_distance.h): a cubic's crossings of the ray to +x from the cubic formula, its closest point by sampling then
// Newton (DiffVG isolates the quintic's roots; both find the same point), coverage smoothstep over [-1, 1] pixels of
// the signed distance, fragments composited back to front. Float32 throughout; the cubic's roots are polished by
// Newton steps so a scanline does not flip on a badly conditioned solve.
#include "ns.h"
#include "ns_internal.hpp"
#include "svg.h"

#include <cfloat>
#include <cstdint>

namespace {

struct V2 {
    float x, y;
};
__device__ inline V2 v2(float x, float y) { return V2{x, y}; }
__device__ inline V2 operator+(V2 a, V2 b) { return v2(a.x + b.x, a.y + b.y); }
__device__ inline V2 operator-(V2 a, V2 b) { return v2(a.x - b.x, a.y - b.y); }
__device__ inline V2 operator*(float s, V2 a) { return v2(s * a.x, s * a.y); }
__device__ inline float dot(V2 a, V2 b) { return a.x * b.x + a.y * b.y; }

__device__ inline V2 bez(const V2* p, float t) {
    const float u = 1.0f - t;
    return (u * u * u) * p[0] + (3.0f * u * u * t) * p[1] + (3.0f * u * t * t) * p[2] + (t * t * t) * p[3];
}
__device__ inline V2 bez_d(const V2* p, float t) {
    const float u = 1.0f - t;
    return (3.0f * u * u) * (p[1] - p[0]) + (6.0f * u * t) * (p[2] - p[1]) + (3.0f * t * t) * (p[3] - p[2]);
}
__device__ inline V2 bez_dd(const V2* p, float t) {
    const float u = 1.0f - t;
    return (6.0f * u) * (p[2] - 2.0f * p[1] + p[0]) + (6.0f * t) * (p[3] - 2.0f * p[2] + p[1]);
}

__device__ inline int npoints(const ns_svg_path& P) { return (P.flags & NS_SVG_CLOSED) ? 3 * P.nseg : 3 * P.nseg + 1; }

// segment j's four control points (a closed path's last segment ends at its first point)
__device__ inline void seg_points(const ns_svg_path& P, const float* pts, int j, V2* p, int* idx) {
    const int n = npoints(P);
    for (int k = 0; k < 4; k++) {
        int i = 3 * j + k;
        if (i >= n) i -= n;
        idx[k] = P.off + i;
        p[k] = v2(pts[2 * (P.off + i)], pts[2 * (P.off + i) + 1]);
    }
}

// the closest point of a cubic to q: (t, squared distance). Sixteen samples, then Newton on (B(t) - q) . B'(t) from
// every sampled local minimum, kept inside its neighbours' interval
__device__ inline float closest_t(const V2* p, V2 q, float* best_d2) {
    constexpr int S = 16;
    auto d2 = [&](float t) {
        const V2 e = bez(p, t) - q;
        return dot(e, e);
    };
    float tb = 0.0f, db = FLT_MAX;
    float a = FLT_MAX, b = d2(0.0f);
    for (int i = 0; i <= S; i++) {
        const float c = i < S ? d2((float) (i + 1) / S) : FLT_MAX;
        if (b <= a && b <= c) {
            const float lo = fmaxf(0.0f, (float) (i - 1) / S), hi = fminf(1.0f, (float) (i + 1) / S);
            float t = (float) i / S;
            float dt = b;
            for (int it = 0; it < 6; it++) {
                const V2 e = bez(p, t) - q, d1 = bez_d(p, t), dd = bez_dd(p, t);
                const float f = dot(e, d1), fp = dot(d1, d1) + dot(e, dd);
                if (!(fp > 0.0f)) break;
                const float tn = fminf(hi, fmaxf(lo, t - f / fp));
                if (tn == t) break;
                t = tn;
            }
            const float dn = d2(t);
            if (dn < dt) dt = dn; else t = (float) i / S;
            if (dt < db) {
                db = dt;
                tb = t;
            }
        }
        a = b;
        b = c;
    }
    *best_d2 = db;
    return tb;
}

// the real roots of a t^3 + b t^2 + c t + d (DiffVG's solve_cubic, with a linear case and Newton polish)
__device__ inline int solve_cubic(float a, float b, float c, float d, float* t) {
    int n = 0;
    if (fabsf(a) < 1e-6f) {
        if (fabsf(b) < 1e-6f) {
            if (fabsf(c) < 1e-12f) return 0;
            t[0] = -d / c;
            return 1;
        }
        const float disc = c * c - 4.0f * b * d;
        if (disc < 0.0f) return 0;
        const float r = sqrtf(disc);
        const float qq = c < 0.0f ? -0.5f * (c - r) : -0.5f * (c + r);
        t[0] = qq / b;
        t[1] = qq != 0.0f ? d / qq : t[0];
        n = 2;
    } else {
        const float B = b / a, C = c / a, D = d / a;
        const float Q = (B * B - 3.0f * C) / 9.0f;
        const float R = (2.0f * B * B * B - 9.0f * B * C + 27.0f * D) / 54.0f;
        if (R * R < Q * Q * Q) {
            const float th = acosf(fminf(1.0f, fmaxf(-1.0f, R / sqrtf(Q * Q * Q))));
            const float s = -2.0f * sqrtf(Q);
            t[0] = s * cosf(th / 3.0f) - B / 3.0f;
            t[1] = s * cosf((th + 2.0f * 3.14159265358979f) / 3.0f) - B / 3.0f;
            t[2] = s * cosf((th - 2.0f * 3.14159265358979f) / 3.0f) - B / 3.0f;
            n = 3;
        } else {
            const float A = R > 0.0f ? -cbrtf(R + sqrtf(R * R - Q * Q * Q)) : cbrtf(-R + sqrtf(R * R - Q * Q * Q));
            const float Bq = fabsf(A) > 1e-6f ? Q / A : 0.0f;
            t[0] = (A + Bq) - B / 3.0f;
            n = 1;
        }
    }
    for (int i = 0; i < n; i++) {
        float x = t[i];
        for (int it = 0; it < 2; it++) {
            const float f = ((a * x + b) * x + c) * x + d, fp = (3.0f * a * x + 2.0f * b) * x + c;
            if (fabsf(fp) > 1e-12f) x -= f / fp;
        }
        t[i] = x;
    }
    return n;
}

// the segment's crossings of the ray from q to +x, signed by the curve's direction (winding_number.h). The ends of
// [0, 1) are half open so a crossing through a joint counts once
__device__ inline int seg_winding(const V2* p, V2 q) {
    const float ymin = fminf(fminf(p[0].y, p[1].y), fminf(p[2].y, p[3].y)), ymax = fmaxf(fmaxf(p[0].y, p[1].y), fmaxf(p[2].y, p[3].y));
    if (q.y < ymin || q.y > ymax) return 0;
    if (fmaxf(fmaxf(p[0].x, p[1].x), fmaxf(p[2].x, p[3].x)) <= q.x) return 0;
    const float a = -p[0].y + 3.0f * p[1].y - 3.0f * p[2].y + p[3].y;
    const float b = 3.0f * p[0].y - 6.0f * p[1].y + 3.0f * p[2].y;
    const float c = -3.0f * p[0].y + 3.0f * p[1].y;
    const float d = p[0].y - q.y;
    float t[3];
    const int n = solve_cubic(a, b, c, d, t);
    int w = 0;
    for (int j = 0; j < n; j++) {
        if (t[j] >= 0.0f && t[j] < 1.0f) {
            if (bez(p, t[j]).x - q.x > 0.0f) {
                w += ((3.0f * a * t[j] + 2.0f * b) * t[j] + c) > 0.0f ? 1 : -1;
            }
        }
    }
    return w;
}

__device__ inline bool inside_path(const ns_svg_path& P, const float* pts, V2 q) {
    int w = 0;
    V2 p[4];
    int idx[4];
    for (int j = 0; j < P.nseg; j++) {
        seg_points(P, pts, j, p, idx);
        w += seg_winding(p, q);
    }
    return (P.flags & NS_SVG_EVENODD) ? (w & 1) != 0 : w != 0;
}

// the path's nearest segment within `radius` of q: (segment, t, distance); false when none is that near
__device__ inline bool path_distance(const ns_svg_path& P, const float* pts, V2 q, float radius, int* seg, float* t, float* dist) {
    float best = radius * radius;
    bool found = false;
    V2 p[4];
    int idx[4];
    for (int j = 0; j < P.nseg; j++) {
        seg_points(P, pts, j, p, idx);
        // the control points' box holds the segment: skip it when the box is farther than the best so far
        const float x0 = fminf(fminf(p[0].x, p[1].x), fminf(p[2].x, p[3].x)), x1 = fmaxf(fmaxf(p[0].x, p[1].x), fmaxf(p[2].x, p[3].x));
        const float y0 = fminf(fminf(p[0].y, p[1].y), fminf(p[2].y, p[3].y)), y1 = fmaxf(fmaxf(p[0].y, p[1].y), fmaxf(p[2].y, p[3].y));
        const float dx = fmaxf(0.0f, fmaxf(x0 - q.x, q.x - x1)), dy = fmaxf(0.0f, fmaxf(y0 - q.y, q.y - y1));
        if (dx * dx + dy * dy > best) continue;
        float d2;
        const float tt = closest_t(p, q, &d2);
        if (d2 <= best) {
            best = d2;
            *seg = j;
            *t = tt;
            found = true;
        }
    }
    *dist = sqrtf(best);
    return found;
}

__device__ inline float smoothstep1(float d) {
    const float t = fminf(1.0f, fmaxf(0.0f, (d + 1.0f) * 0.5f));
    return t * t * (3.0f - 2.0f * t);
}
__device__ inline float d_smoothstep1(float d, float g) {
    if (d < -1.0f || d > 1.0f) return 0.0f;
    const float t = (d + 1.0f) * 0.5f;
    return g * (6.0f * t - 6.0f * t * t) * 0.5f;
}

// a fragment kept for the backward pass: its path (2 i + stroke), alpha after coverage, colour, what lay under it
// (premultiplied), its distance (signed for a fill: + inside), the nearest segment and t (seg -1: none near)
struct Frag {
    int id;
    int seg;
    float a, r, g, b;
    float pr, pg, pb, pa;
    float d, t;
};
constexpr int CAP = 48;

// the gradient of distance |q - B(t)| to the segment's control points, times g, added to d_pts
__device__ inline void d_distance(const ns_svg_path& P, const float* pts, int seg, float t, V2 q, float g, float* d_pts) {
    V2 p[4];
    int idx[4];
    seg_points(P, pts, seg, p, idx);
    const V2 c = bez(p, t);
    const V2 e = c - q;
    const float dd = dot(e, e);
    if (dd < 1e-10f) return;  // DiffVG: no derivative at distance 0
    const V2 n = (g / sqrtf(dd)) * e;
    const float u = 1.0f - t;
    const float w[4] = {u * u * u, 3.0f * u * u * t, 3.0f * u * t * t, t * t * t};
    for (int k = 0; k < 4; k++) {
        atomicAdd(&d_pts[2 * idx[k]], n.x * w[k]);
        atomicAdd(&d_pts[2 * idx[k] + 1], n.y * w[k]);
    }
}

// one pixel's prefiltered fragments composited (and, with BACK, its gradients): DiffVG's sample_color_prefiltered
template <bool BACK>
__device__ void pixel(int W, int H, int N, const ns_svg_path* paths, const float* pts, const float* fill, const float* stroke,
                      const float* hw, const float* box, const float* bg, int mode, const float* target, const float* weight,
                      const float* d_img, float* d_pts, float* d_fill, float* d_stroke, float* d_hw, float* out, int x, int y,
                      int first, int mono, float* loss_px) {
    const V2 q = v2(x + 0.5f, y + 0.5f);
    Frag fr[BACK ? CAP : 1];
    int start = 0, count = 0;
    float ar = 0.0f, ag = 0.0f, ab = 0.0f, aa = 0.0f;  // premultiplied
    auto push = [&](int id, int seg, float a, float r, float g, float b, float d, float t) {
        if (BACK) {
            if (a >= 1.0f) {
                // an opaque fragment hides what is under it: nothing below it gets a gradient (DiffVG's d_prev * 0)
                start = 0;
                count = 0;
            }
            int slot;
            if (count < CAP) {
                slot = (start + count) % CAP;
                count++;
            } else {
                // full: the bottom one gives way (its gradient is through CAP layers of other paths)
                slot = start;
                start = (start + 1) % CAP;
            }
            Frag& f = fr[slot];
            f.id = id;
            f.seg = seg;
            f.a = a;
            f.r = r;
            f.g = g;
            f.b = b;
            f.pr = ar;
            f.pg = ag;
            f.pb = ab;
            f.pa = aa;
            f.d = d;
            f.t = t;
        }
        ar = ar * (1.0f - a) + a * r;
        ag = ag * (1.0f - a) + a * g;
        ab = ab * (1.0f - a) + a * b;
        aa = aa * (1.0f - a) + a;
    };
    for (int i = first; i < N; i++) {
        const float4 bx = reinterpret_cast<const float4*>(box)[i];
        const ns_svg_path P = paths[i];
        // a pixel within one pixel of the box can be partly covered; DiffVG's own cull (its BVH) stops at the box
        const float mg = (P.flags & NS_SVG_DIFFVG_BOX) ? 0.0f : 1.0f;
        if (q.x < bx.x - mg || q.x > bx.z + mg || q.y < bx.y - mg || q.y > bx.w + mg) continue;
        if (P.flags & NS_SVG_FILL) {
            int seg = -1;
            float t = 0.0f, dist = FLT_MAX;
            const bool found = path_distance(P, pts, q, 1.0f, &seg, &t, &dist);
            const bool in = inside_path(P, pts, q);
            if (found || in) {
                const float d = found ? (in ? dist : -dist) : FLT_MAX;
                const float w = smoothstep1(d);
                if (w > 0.0f) {
                    const float* c = fill + 4 * i;
                    if (mono) push(2 * i, found ? seg : -1, w, 1.0f, 1.0f, 1.0f, d, t);
                    else push(2 * i, found ? seg : -1, c[3] * w, c[0], c[1], c[2], d, t);
                }
            }
        }
        if ((P.flags & NS_SVG_STROKE) && hw[i] > 0.0f) {
            int seg = -1;
            float t = 0.0f, dist = FLT_MAX;
            if (path_distance(P, pts, q, hw[i] + 1.0f, &seg, &t, &dist)) {
                const float w = smoothstep1(dist + hw[i]) - smoothstep1(dist - hw[i]);
                if (w > 0.0f) {
                    const float* c = stroke + 4 * i;
                    if (mono) push(2 * i + 1, seg, w, 1.0f, 1.0f, 1.0f, dist, t);
                    else push(2 * i + 1, seg, c[3] * w, c[0], c[1], c[2], dist, t);
                }
            }
        }
    }
    const int pix = y * W + x;
    // the pixel, and the loss's gradient to the premultiplied colour and alpha
    float o[4];
    float dr = 0.0f, dg = 0.0f, db = 0.0f, da = 0.0f;
    if (mode == 0) {
        const float br = bg ? bg[0] : 1.0f, bgg = bg ? bg[1] : 1.0f, bb = bg ? bg[2] : 1.0f;
        o[0] = ar + br * (1.0f - aa);
        o[1] = ag + bgg * (1.0f - aa);
        o[2] = ab + bb * (1.0f - aa);
        o[3] = 1.0f;
        if (BACK) {
            const float wt = weight ? weight[pix] : 1.0f;
            const float* tg = target + 3 * pix;
            const float e0 = o[0] - tg[0], e1 = o[1] - tg[1], e2 = o[2] - tg[2];
            const float s = 2.0f * wt / ((float) W * (float) H);
            *loss_px = wt * (e0 * e0 + e1 * e1 + e2 * e2) / ((float) W * (float) H);
            dr = s * e0;
            dg = s * e1;
            db = s * e2;
            da = -(br * dr + bgg * dg + bb * db);
        }
    } else {
        float fr_ = ar, fg = ag, fb = ab;
        if (aa > 1e-6f) {
            fr_ /= aa;
            fg /= aa;
            fb /= aa;
        }
        o[0] = fr_;
        o[1] = fg;
        o[2] = fb;
        o[3] = aa;
        if (BACK && count > 0) {
            const float* gi = d_img + 4 * pix;
            dr = gi[0];
            dg = gi[1];
            db = gi[2];
            da = gi[3];
            if (aa > 1e-6f) {
                dr /= aa;
                dg /= aa;
                db /= aa;
                da -= (gi[0] * fr_ + gi[1] * fg + gi[2] * fb) / aa;
            }
        }
    }
    if (out) {
        for (int c = 0; c < 4; c++) out[4 * pix + c] = o[c];
    }
    // (a forward pass keeps no fragments: count is 0 and the loop is empty)
    for (int k = BACK ? count - 1 : -1; k >= 0; k--) {
        const Frag& f = fr[(start + k) % CAP];
        const float d_prev_a = da * (1.0f - f.a);
        float d_alpha = da * (1.0f - f.pa) + dr * (f.r - f.pr) + dg * (f.g - f.pg) + db * (f.b - f.pb);
        const float dcr = dr * f.a, dcg = dg * f.a, dcb = db * f.a;
        const int i = f.id >> 1;
        const bool is_stroke = f.id & 1;
        const ns_svg_path P = paths[i];
        if (!is_stroke) {
            const float w = smoothstep1(f.d);
            const float ca = fill[4 * i + 3];
            const float d_w = (w > 0.0f) ? ca * d_alpha : 0.0f;
            if (!mono) {
                atomicAdd(&d_fill[4 * i + 0], dcr);
                atomicAdd(&d_fill[4 * i + 1], dcg);
                atomicAdd(&d_fill[4 * i + 2], dcb);
                atomicAdd(&d_fill[4 * i + 3], d_alpha * w);
            }
            float d_d = d_smoothstep1(f.d, d_w);
            if (f.d < 0.0f) d_d = -d_d;
            if (f.seg >= 0 && fabsf(d_d) > 1e-10f) d_distance(P, pts, f.seg, f.t, q, d_d, d_pts);
        } else {
            const float h = hw[i];
            const float w = smoothstep1(f.d + h) - smoothstep1(f.d - h);
            const float ca = stroke[4 * i + 3];
            const float d_w = (w > 0.0f) ? ca * d_alpha : 0.0f;
            if (!mono) {
                atomicAdd(&d_stroke[4 * i + 0], dcr);
                atomicAdd(&d_stroke[4 * i + 1], dcg);
                atomicAdd(&d_stroke[4 * i + 2], dcb);
                atomicAdd(&d_stroke[4 * i + 3], d_alpha * w);
            }
            const float dp = d_smoothstep1(f.d + h, d_w), dm = -d_smoothstep1(f.d - h, d_w);
            const float d_d = dp + dm;
            atomicAdd(&d_hw[i], dp - dm);
            if (fabsf(d_d) > 1e-10f) d_distance(P, pts, f.seg, f.t, q, d_d, d_pts);
        }
        dr *= (1.0f - f.a);
        dg *= (1.0f - f.a);
        db *= (1.0f - f.a);
        da = d_prev_a;
    }
}

__global__ void bounds_k(int N, const ns_svg_path* paths, const float* pts, const float* hw, float* box) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const ns_svg_path P = paths[i];
    const int n = npoints(P);
    float x0 = FLT_MAX, y0 = FLT_MAX, x1 = -FLT_MAX, y1 = -FLT_MAX;
    for (int k = 0; k < n; k++) {
        const float x = pts[2 * (P.off + k)], y = pts[2 * (P.off + k) + 1];
        x0 = fminf(x0, x);
        y0 = fminf(y0, y);
        x1 = fmaxf(x1, x);
        y1 = fmaxf(y1, y);
    }
    const float h = (P.flags & NS_SVG_STROKE) ? fmaxf(0.0f, hw[i]) : 0.0f;
    box[4 * i + 0] = x0 - h;
    box[4 * i + 1] = y0 - h;
    box[4 * i + 2] = x1 + h;
    box[4 * i + 3] = y1 + h;
}

__global__ void grad_k(int W, int H, int N, const ns_svg_path* paths, const float* pts, const float* fill, const float* stroke,
                       const float* hw, const float* box, const float* bg, int mode, const float* target, const float* weight,
                       const float* d_img, float* d_pts, float* d_fill, float* d_stroke, float* d_hw, float* loss, float* out) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    float l = 0.0f;
    if (x < W && y < H) {
        pixel<true>(W, H, N, paths, pts, fill, stroke, hw, box, bg, mode, target, weight, d_img, d_pts, d_fill, d_stroke, d_hw, out, x, y, 0, 0, &l);
    }
    if (loss) {
        for (int o = 16; o > 0; o >>= 1) l += __shfl_down_sync(0xffffffffu, l, o);
        if ((threadIdx.x + threadIdx.y * blockDim.x) % 32 == 0 && l != 0.0f) atomicAdd(loss, l);
    }
}

// hard samples (samples x samples a pixel): each sample's fragments composited without coverage smoothing - inside a
// fill, within a stroke's half-width - then averaged (straight colour and alpha averaged apart, as DiffVG's box
// filter does them; over a background, the composited colours)
__device__ void pixel_hard(int W, int H, int N, const ns_svg_path* paths, const float* pts, const float* fill, const float* stroke,
                           const float* hw, const float* box, const float* bg, int s, int first, int mono, float* out, int x, int y) {
    float acc[4] = {0, 0, 0, 0};
    for (int sy = 0; sy < s; sy++) {
        for (int sx = 0; sx < s; sx++) {
            const V2 q = v2(x + (sx + 0.5f) / s, y + (sy + 0.5f) / s);
            float ar = 0.0f, ag = 0.0f, ab = 0.0f, aa = 0.0f;
            for (int i = first; i < N; i++) {
                const float4 bx = reinterpret_cast<const float4*>(box)[i];
                if (q.x < bx.x || q.x > bx.z || q.y < bx.y || q.y > bx.w) continue;
                const ns_svg_path P = paths[i];
                for (int part = 0; part < 2; part++) {
                    bool hit = false;
                    const float* c = part == 0 ? fill + 4 * i : stroke + 4 * i;
                    if (part == 0 && (P.flags & NS_SVG_FILL)) hit = inside_path(P, pts, q);
                    if (part == 1 && (P.flags & NS_SVG_STROKE) && hw[i] > 0.0f) {
                        int seg;
                        float t, dist;
                        hit = path_distance(P, pts, q, hw[i], &seg, &t, &dist);
                    }
                    if (!hit) continue;
                    const float a = mono ? 1.0f : c[3];
                    const float r = mono ? 1.0f : c[0], g = mono ? 1.0f : c[1], b = mono ? 1.0f : c[2];
                    ar = ar * (1.0f - a) + a * r;
                    ag = ag * (1.0f - a) + a * g;
                    ab = ab * (1.0f - a) + a * b;
                    aa = aa * (1.0f - a) + a;
                }
            }
            if (bg) {
                acc[0] += ar + bg[0] * (1.0f - aa);
                acc[1] += ag + bg[1] * (1.0f - aa);
                acc[2] += ab + bg[2] * (1.0f - aa);
                acc[3] += 1.0f;
            } else {
                const float k = aa > 1e-6f ? 1.0f / aa : 1.0f;
                acc[0] += ar * k;
                acc[1] += ag * k;
                acc[2] += ab * k;
                acc[3] += aa;
            }
        }
    }
    const float inv = 1.0f / (float) (s * s);
    for (int c = 0; c < 4; c++) out[4 * (y * W + x) + c] = acc[c] * inv;
}

__global__ void render_k(int W, int H, int N, const ns_svg_path* paths, const float* pts, const float* fill, const float* stroke,
                         const float* hw, const float* box, const float* bg, int samples, int first, int mono, float* out) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    if (samples <= 0) {
        float l;
        pixel<false>(W, H, N, paths, pts, fill, stroke, hw, box, bg, bg ? 0 : 1, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                     nullptr, out, x, y, first, mono, &l);
    } else {
        pixel_hard(W, H, N, paths, pts, fill, stroke, hw, box, bg, samples, first, mono, out, x, y);
    }
}

// LIVE's get_sdf: each pixel's distance to the other sign of phi = 2 cov - 1 (cov: a render's alpha) (a pixel at 0 is on the level), less half a
// pixel, truncated at 10; the counts of each sign and the largest distance
constexpr int UDF_R = 11;
constexpr float UDF_TRUNC = 10.0f;
__global__ void udf_dist_k(int W, int H, const float* cov, float* dist, int* scratch) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    const float phi = 2.0f * cov[4 * (y * W + x) + 3] - 1.0f;
    float best = UDF_TRUNC;
    if (phi == 0.0f) {
        best = 0.0f;
    } else {
        for (int dy = -UDF_R; dy <= UDF_R; dy++) {
            const int yy = y + dy;
            if (yy < 0 || yy >= H) continue;
            for (int dx = -UDF_R; dx <= UDF_R; dx++) {
                const int xx = x + dx;
                if (xx < 0 || xx >= W) continue;
                const float o = 2.0f * cov[4 * (yy * W + xx) + 3] - 1.0f;
                if ((phi > 0.0f && o <= 0.0f) || (phi < 0.0f && o >= 0.0f)) {
                    best = fminf(best, fmaxf(0.0f, sqrtf((float) (dx * dx + dy * dy)) - 0.5f));
                }
            }
        }
    }
    dist[y * W + x] = best;
    if (phi > 0.0f) atomicAdd(&scratch[0], 1);
    if (phi < 0.0f) atomicAdd(&scratch[1], 1);
    atomicMax(&scratch[2], __float_as_int(best));  // non-negative floats order as their bits
}

__global__ void udf_weight_k(int W, int H, const float* dist, const float* keep, const int* scratch, float* weight) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= W * H) return;
    float w = 0.0f;
    const float mx = __int_as_float(scratch[2]);
    if (scratch[0] > 0 && scratch[1] > 0 && mx > 0.0f) w = (mx - dist[i]) / mx;
    if (keep) w += keep[i];
    weight[i] = fminf(1.0f, fmaxf(0.0f, w));
}

__device__ inline float sine2(V2 a, V2 b) { return (a.x * b.y - a.y * b.x) / (sqrtf(dot(a, a)) * sqrtf(dot(b, b))); }

__global__ void xing_k(int N, int first, const ns_svg_path* paths, const float* pts, float scale, float* d_pts, float* loss) {
    const int i = first + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const ns_svg_path P = paths[i];
    if (!(P.flags & NS_SVG_CLOSED) || P.nseg <= 0) return;
    const int n = 3 * P.nseg;
    auto at = [&](int k) { k %= n; return v2(pts[2 * (P.off + k)], pts[2 * (P.off + k) + 1]); };
    float sum = 0.0f;
    const float per = scale / (float) P.nseg;
    for (int s = 0; s < P.nseg; s++) {
        const int k0 = 3 * s;
        const V2 x0 = at(k0), x1 = at(k0 + 1), x2 = at(k0 + 2), x3 = at(k0 + 3);
        const V2 v1 = x1 - x0, v2_ = x2 - x1, v3 = x3 - x2;
        const bool direct = sine2(v1, v2_) >= 0.0f;
        const float sa = sine2(v1, v3);
        float g;  // d loss / d sine(v1, v3)
        if (direct) {
            sum += fmaxf(0.0f, -sa);
            g = sa < 0.0f ? -1.0f : 0.0f;
        } else {
            sum += fmaxf(0.0f, sa);
            g = sa > 0.0f ? 1.0f : 0.0f;
        }
        if (g == 0.0f) continue;
        g *= per;
        const float n1 = sqrtf(dot(v1, v1)), n3 = sqrtf(dot(v3, v3));
        const float c = v1.x * v3.y - v1.y * v3.x;
        const V2 dv1 = (1.0f / (n1 * n3)) * v2(v3.y, -v3.x) - (c / (n1 * n1 * n1 * n3)) * v1;
        const V2 dv3 = (1.0f / (n1 * n3)) * v2(-v1.y, v1.x) - (c / (n1 * n3 * n3 * n3)) * v3;
        const int i0 = P.off + k0 % n, i1 = P.off + (k0 + 1) % n, i2 = P.off + (k0 + 2) % n, i3 = P.off + (k0 + 3) % n;
        atomicAdd(&d_pts[2 * i1], g * dv1.x);
        atomicAdd(&d_pts[2 * i1 + 1], g * dv1.y);
        atomicAdd(&d_pts[2 * i0], -g * dv1.x);
        atomicAdd(&d_pts[2 * i0 + 1], -g * dv1.y);
        atomicAdd(&d_pts[2 * i3], g * dv3.x);
        atomicAdd(&d_pts[2 * i3 + 1], g * dv3.y);
        atomicAdd(&d_pts[2 * i2], -g * dv3.x);
        atomicAdd(&d_pts[2 * i2 + 1], -g * dv3.y);
    }
    if (loss) atomicAdd(loss, sum * per);
}

__device__ inline float decay(int n, int every, float ratio) {
    if (every <= 0) return 1.0f;
    const int k = n / every;
    const float r = (float) (n % every) / (float) every;
    const float s = powf(ratio, (float) k), e = powf(ratio, (float) (k + 1));
    return s * (1.0f - r) + e * r;
}

// the groups' learning rates and clamps, passed by value
struct Groups {
    float lr[4], lo[4], hi[4];
};

__global__ void adam_k(int64_t n, float* p, float* d, float* m, float* v, const uint8_t* grp, const int32_t* own, const int32_t* birth,
                       int step, Groups G, int every, float ratio, float b1, float b2, float eps) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int gidx = grp[i];
    const float g = d[i];
    d[i] = 0.0f;
    if (gidx > 3) return;
    const int k = step - birth[own[i]];
    if (k < 0) return;
    const float lr = G.lr[gidx] * decay(k, every, ratio);
    const float mi = b1 * m[i] + (1.0f - b1) * g;
    const float vi = b2 * v[i] + (1.0f - b2) * g * g;
    m[i] = mi;
    v[i] = vi;
    const float t = (float) (k + 1);
    const float bc1 = 1.0f - powf(b1, t), bc2 = 1.0f - powf(b2, t);
    float x = p[i] - (lr / bc1) * mi / (sqrtf(vi) / sqrtf(bc2) + eps);
    x = fminf(G.hi[gidx], fmaxf(G.lo[gidx], x));
    p[i] = x;
}

__global__ void to_u8_k(int n, const float* px, uint8_t* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    for (int c = 0; c < 3; c++) {
        const float v = floorf(fminf(1.0f, fmaxf(0.0f, px[4 * i + c])) * 255.0f + 0.5f);
        out[3 * i + c] = (uint8_t) v;
    }
}

inline dim3 grid2(int W, int H) { return dim3((unsigned) ((W + 15) / 16), (unsigned) ((H + 15) / 16)); }

}  // namespace

extern "C" {

int ns_image_svg_bounds(ns_gpu* g, int32_t N, const ns_svg_path* paths, const float* pts, const float* hw, float* box) {
    if (N <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    bounds_k<<<(unsigned) ((N + 127) / 128), 128, 0, g->q>>>(N, paths, pts, hw, box);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_svg_render(ns_gpu* g, int32_t W, int32_t H, int32_t N, const ns_svg_path* paths, const float* pts, const float* fill,
                        const float* stroke, const float* hw, const float* box, const float* bg, int32_t samples, int32_t first,
                        int32_t mono, float* out) {
    if (W <= 0 || H <= 0) return ns_fail("svg render: an empty canvas");
    NS_CU(cudaSetDevice(g->dev));
    render_k<<<grid2(W, H), dim3(16, 16), 0, g->q>>>(W, H, N, paths, pts, fill, stroke, hw, box, bg, samples, first, mono, out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_svg_grad(ns_gpu* g, int32_t W, int32_t H, int32_t N, const ns_svg_path* paths, const float* pts, const float* fill,
                      const float* stroke, const float* hw, const float* box, const float* bg, int32_t mode, const float* target,
                      const float* weight, const float* d_img, float* d_pts, float* d_fill, float* d_stroke, float* d_hw, float* loss,
                      float* out) {
    if (W <= 0 || H <= 0) return ns_fail("svg grad: an empty canvas");
    if (mode == 0 && !target) return ns_fail("svg grad: mode 0 needs a target");
    if (mode == 1 && !d_img) return ns_fail("svg grad: mode 1 needs d_img");
    NS_CU(cudaSetDevice(g->dev));
    grad_k<<<grid2(W, H), dim3(16, 16), 0, g->q>>>(W, H, N, paths, pts, fill, stroke, hw, box, bg, mode, target, weight, d_img, d_pts, d_fill,
                                                   d_stroke, d_hw, loss, out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_svg_udf(ns_gpu* g, int32_t W, int32_t H, const float* cov, const float* keep, float* weight, float* dist, int32_t* scratch) {
    NS_CU(cudaSetDevice(g->dev));
    NS_CU(cudaMemsetAsync(scratch, 0, 4 * sizeof(int32_t), g->q));
    udf_dist_k<<<grid2(W, H), dim3(16, 16), 0, g->q>>>(W, H, cov, dist, scratch);
    NS_LAUNCH_CHECK();
    udf_weight_k<<<(unsigned) ((W * H + 255) / 256), 256, 0, g->q>>>(W, H, dist, keep, scratch, weight);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_svg_xing(ns_gpu* g, int32_t N, int32_t first, const ns_svg_path* paths, const float* pts, float scale, float* d_pts,
                      float* loss) {
    if (N - first <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    xing_k<<<(unsigned) ((N - first + 127) / 128), 128, 0, g->q>>>(N, first, paths, pts, scale / (float) (N - first), d_pts, loss);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_svg_adam(ns_gpu* g, int64_t n, float* p, float* d, float* m, float* v, const uint8_t* grp, const int32_t* own,
                      const int32_t* birth, int32_t step, const float* lr4, int32_t decay_every, float ratio, float b1, float b2,
                      float eps, const float* lo4, const float* hi4) {
    if (n <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    Groups G;
    for (int k = 0; k < 4; k++) {
        G.lr[k] = lr4[k];
        G.lo[k] = lo4[k];
        G.hi[k] = hi4[k];
    }
    adam_k<<<(unsigned) ((n + 255) / 256), 256, 0, g->q>>>(n, p, d, m, v, grp, own, birth, step, G, decay_every, ratio, b1, b2, eps);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_svg_to_u8(ns_gpu* g, int32_t W, int32_t H, const float* px, uint8_t* out) {
    NS_CU(cudaSetDevice(g->dev));
    to_u8_k<<<(unsigned) ((W * H + 255) / 256), 256, 0, g->q>>>(W * H, px, out);
    NS_LAUNCH_CHECK();
    return 0;
}

}  // extern "C"
