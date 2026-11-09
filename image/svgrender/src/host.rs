//! The rasterizer on the host, in float64: the same arithmetic as the kernels (svg.cu), for the unit tests (its
//! gradients against finite differences) and `selftest` (the kernels against it). Slow; a reference, not a path.

use crate::scene::{npoints, Scene, CLOSED, EVENODD, FILL, STROKE};

type V = [f64; 2];

fn sub(a: V, b: V) -> V {
    [a[0] - b[0], a[1] - b[1]]
}
fn dot(a: V, b: V) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}
fn bez(p: &[V; 4], t: f64) -> V {
    let u = 1.0 - t;
    let w = [u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t];
    [(0..4).map(|k| w[k] * p[k][0]).sum(), (0..4).map(|k| w[k] * p[k][1]).sum()]
}
fn bez_d(p: &[V; 4], t: f64) -> V {
    let u = 1.0 - t;
    let a = sub(p[1], p[0]);
    let b = sub(p[2], p[1]);
    let c = sub(p[3], p[2]);
    [3.0 * u * u * a[0] + 6.0 * u * t * b[0] + 3.0 * t * t * c[0], 3.0 * u * u * a[1] + 6.0 * u * t * b[1] + 3.0 * t * t * c[1]]
}
fn bez_dd(p: &[V; 4], t: f64) -> V {
    let u = 1.0 - t;
    [6.0 * u * (p[2][0] - 2.0 * p[1][0] + p[0][0]) + 6.0 * t * (p[3][0] - 2.0 * p[2][0] + p[1][0]),
     6.0 * u * (p[2][1] - 2.0 * p[1][1] + p[0][1]) + 6.0 * t * (p[3][1] - 2.0 * p[2][1] + p[1][1])]
}

/// Segment j of path i: its control points and their indices in the point array
pub fn seg(s: &Scene, i: usize, j: usize) -> ([V; 4], [usize; 4]) {
    let p = s.paths[i];
    let n = npoints(p.nseg, p.flags & CLOSED != 0);
    let mut pts = [[0.0; 2]; 4];
    let mut idx = [0usize; 4];
    for k in 0..4 {
        let mut q = 3 * j + k;
        if q >= n {
            q -= n;
        }
        idx[k] = p.off as usize + q;
        pts[k] = [s.points[2 * idx[k]] as f64, s.points[2 * idx[k] + 1] as f64];
    }
    (pts, idx)
}

/// The cubic's closest point to q: (t, squared distance) - sampling then Newton, as the kernel
pub fn closest_t(p: &[V; 4], q: V) -> (f64, f64) {
    const S: usize = 16;
    let d2 = |t: f64| {
        let e = sub(bez(p, t), q);
        dot(e, e)
    };
    let (mut tb, mut db) = (0.0, f64::MAX);
    let mut a = f64::MAX;
    let mut b = d2(0.0);
    for i in 0..=S {
        let c = if i < S { d2((i + 1) as f64 / S as f64) } else { f64::MAX };
        if b <= a && b <= c {
            let lo = ((i as f64 - 1.0) / S as f64).max(0.0);
            let hi = ((i as f64 + 1.0) / S as f64).min(1.0);
            let mut t = i as f64 / S as f64;
            for _ in 0..12 {
                let e = sub(bez(p, t), q);
                let d1 = bez_d(p, t);
                let f = dot(e, d1);
                let fp = dot(d1, d1) + dot(e, bez_dd(p, t));
                if fp <= 0.0 {
                    break;
                }
                let tn = (t - f / fp).clamp(lo, hi);
                if tn == t {
                    break;
                }
                t = tn;
            }
            let (mut tt, mut dt) = (t, d2(t));
            if b < dt {
                tt = i as f64 / S as f64;
                dt = b;
            }
            if dt < db {
                db = dt;
                tb = tt;
            }
        }
        a = b;
        b = c;
    }
    (tb, db)
}

/// The segment's signed crossings of the ray from q to +x (the kernel's, its roots found by bisection on each
/// monotonic piece here: the same crossings without the cubic formula's conditioning)
fn seg_winding(p: &[V; 4], q: V) -> i32 {
    let ys = [p[0][1], p[1][1], p[2][1], p[3][1]];
    let (ymin, ymax) = (ys.iter().cloned().fold(f64::MAX, f64::min), ys.iter().cloned().fold(f64::MIN, f64::max));
    if q[1] < ymin || q[1] > ymax {
        return 0;
    }
    if [p[0][0], p[1][0], p[2][0], p[3][0]].iter().cloned().fold(f64::MIN, f64::max) <= q[0] {
        return 0;
    }
    // y(t)'s turning points split [0, 1] into monotonic pieces
    let a = -p[0][1] + 3.0 * p[1][1] - 3.0 * p[2][1] + p[3][1];
    let b = 3.0 * p[0][1] - 6.0 * p[1][1] + 3.0 * p[2][1];
    let c = -3.0 * p[0][1] + 3.0 * p[1][1];
    let mut cuts = vec![0.0, 1.0];
    let (qa, qb, qc) = (3.0 * a, 2.0 * b, c);
    if qa.abs() > 1e-12 {
        let disc = qb * qb - 4.0 * qa * qc;
        if disc >= 0.0 {
            for r in [(-qb - disc.sqrt()) / (2.0 * qa), (-qb + disc.sqrt()) / (2.0 * qa)] {
                if r > 0.0 && r < 1.0 {
                    cuts.push(r);
                }
            }
        }
    } else if qb.abs() > 1e-12 {
        let r = -qc / qb;
        if r > 0.0 && r < 1.0 {
            cuts.push(r);
        }
    }
    cuts.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let y = |t: f64| ((a * t + b) * t + c) * t + p[0][1] - q[1];
    let mut w = 0;
    for k in 0..cuts.len() - 1 {
        let (mut lo, mut hi) = (cuts[k], cuts[k + 1]);
        let (ylo, yhi) = (y(lo), y(hi));
        // a root in [lo, hi) (half open, as the kernel: a crossing through a joint counts once)
        if ylo == 0.0 {
            hi = lo;
        } else if ylo * yhi > 0.0 || yhi == 0.0 {
            continue;
        } else {
            for _ in 0..80 {
                let m = 0.5 * (lo + hi);
                if (y(m) > 0.0) == (ylo > 0.0) {
                    lo = m;
                } else {
                    hi = m;
                }
            }
        }
        let t = 0.5 * (lo + hi);
        if !(0.0..1.0).contains(&t) {
            continue;
        }
        if bez(p, t)[0] - q[0] > 0.0 {
            w += if (3.0 * a * t + 2.0 * b) * t + c > 0.0 { 1 } else { -1 };
        }
    }
    w
}

pub fn inside(s: &Scene, i: usize, q: V) -> bool {
    let p = s.paths[i];
    let w: i32 = (0..p.nseg as usize).map(|j| seg_winding(&seg(s, i, j).0, q)).sum();
    if p.flags & EVENODD != 0 { w & 1 != 0 } else { w != 0 }
}

/// The path's nearest segment within `radius`: (segment, t, distance)
pub fn distance(s: &Scene, i: usize, q: V, radius: f64) -> Option<(usize, f64, f64)> {
    let mut best = radius * radius;
    let mut out = None;
    for j in 0..s.paths[i].nseg as usize {
        let (p, _) = seg(s, i, j);
        let (t, d2) = closest_t(&p, q);
        if d2 <= best {
            best = d2;
            out = Some((j, t, d2.sqrt()));
        }
    }
    out
}

fn smoothstep(d: f64) -> f64 {
    let t = ((d + 1.0) * 0.5).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
fn d_smoothstep(d: f64, g: f64) -> f64 {
    if !(-1.0..=1.0).contains(&d) {
        return 0.0;
    }
    let t = (d + 1.0) * 0.5;
    g * (6.0 * t - 6.0 * t * t) * 0.5
}

/// Gradients by parameter, the scene's layout
#[derive(Clone, Debug)]
pub struct Grads {
    pub pts: Vec<f64>,
    pub fill: Vec<[f64; 4]>,
    pub stroke: Vec<[f64; 4]>,
    pub hw: Vec<f64>,
}

impl Grads {
    pub fn zeros(s: &Scene) -> Grads {
        Grads { pts: vec![0.0; s.points.len()], fill: vec![[0.0; 4]; s.len()], stroke: vec![[0.0; 4]; s.len()], hw: vec![0.0; s.len()] }
    }
}

/// What the loss is: LIVE's weighted MSE over a background, or a given gradient of DiffVG's straight RGBA pixels
pub enum Loss<'a> {
    Mse { target: &'a [f32], weight: Option<&'a [f32]>, bg: [f64; 3] },
    Image { d_img: &'a [f32] },
}

struct Frag {
    i: usize,
    stroke: bool,
    a: f64,
    c: [f64; 3],
    prev: [f64; 4],
    d: f64,
    near: Option<(usize, f64)>,
}

/// The prefiltered pixels (straight RGBA, or over `bg` when given) and with a loss its value and gradients
pub fn run(s: &Scene, loss: Option<&Loss>, bg: Option<[f64; 3]>) -> (Vec<[f64; 4]>, f64, Grads) {
    let (w, h) = (s.width as usize, s.height as usize);
    let mut out = vec![[0.0; 4]; w * h];
    let mut g = Grads::zeros(s);
    let mut total = 0.0;
    let bg = match loss {
        Some(Loss::Mse { bg, .. }) => Some(*bg),
        Some(Loss::Image { .. }) => None,
        None => bg,
    };
    for y in 0..h {
        for x in 0..w {
            let q = [x as f64 + 0.5, y as f64 + 0.5];
            let mut frags: Vec<Frag> = Vec::new();
            let mut acc = [0.0f64; 4];
            let push = |f: Frag, acc: &mut [f64; 4], frags: &mut Vec<Frag>| {
                let a = f.a;
                let c = f.c;
                frags.push(f);
                for k in 0..3 {
                    acc[k] = acc[k] * (1.0 - a) + a * c[k];
                }
                acc[3] = acc[3] * (1.0 - a) + a;
            };
            for i in 0..s.len() {
                let p = s.paths[i];
                if p.flags & FILL != 0 {
                    let near = distance(s, i, q, 1.0);
                    let inn = inside(s, i, q);
                    if near.is_some() || inn {
                        let d = match near {
                            Some((_, _, dd)) => if inn { dd } else { -dd },
                            None => f64::MAX,
                        };
                        let wv = smoothstep(d);
                        if wv > 0.0 {
                            let c = s.fill[i];
                            let f = Frag { i, stroke: false, a: c[3] as f64 * wv, c: [c[0] as f64, c[1] as f64, c[2] as f64], prev: acc, d,
                                           near: near.map(|(j, t, _)| (j, t)) };
                            push(f, &mut acc, &mut frags);
                        }
                    }
                }
                if p.flags & STROKE != 0 && s.hw[i] > 0.0 {
                    let hw = s.hw[i] as f64;
                    if let Some((j, t, d)) = distance(s, i, q, hw + 1.0) {
                        let wv = smoothstep(d + hw) - smoothstep(d - hw);
                        if wv > 0.0 {
                            let c = s.stroke[i];
                            let f = Frag { i, stroke: true, a: c[3] as f64 * wv, c: [c[0] as f64, c[1] as f64, c[2] as f64], prev: acc, d, near: Some((j, t)) };
                            push(f, &mut acc, &mut frags);
                        }
                    }
                }
            }
            let pix = y * w + x;
            let (mut dc, mut da) = ([0.0f64; 3], 0.0f64);
            match bg {
                Some(b) => {
                    let o = [acc[0] + b[0] * (1.0 - acc[3]), acc[1] + b[1] * (1.0 - acc[3]), acc[2] + b[2] * (1.0 - acc[3]), 1.0];
                    out[pix] = o;
                    if let Some(Loss::Mse { target, weight, .. }) = loss {
                        let wt = weight.map(|v| v[pix] as f64).unwrap_or(1.0);
                        let n = (w * h) as f64;
                        for k in 0..3 {
                            let e = o[k] - target[3 * pix + k] as f64;
                            total += wt * e * e / n;
                            dc[k] = 2.0 * wt * e / n;
                        }
                        da = -(b[0] * dc[0] + b[1] * dc[1] + b[2] * dc[2]);
                    }
                }
                None => {
                    let mut f = [acc[0], acc[1], acc[2]];
                    if acc[3] > 1e-6 {
                        for v in f.iter_mut() {
                            *v /= acc[3];
                        }
                    }
                    out[pix] = [f[0], f[1], f[2], acc[3]];
                    if let (Some(Loss::Image { d_img }), false) = (loss, frags.is_empty()) {
                        let gi = &d_img[4 * pix..4 * pix + 4];
                        dc = [gi[0] as f64, gi[1] as f64, gi[2] as f64];
                        da = gi[3] as f64;
                        if acc[3] > 1e-6 {
                            for v in dc.iter_mut() {
                                *v /= acc[3];
                            }
                            da -= (gi[0] as f64 * f[0] + gi[1] as f64 * f[1] + gi[2] as f64 * f[2]) / acc[3];
                        }
                    }
                }
            }
            if loss.is_none() {
                continue;
            }
            for f in frags.iter().rev() {
                let d_prev_a = da * (1.0 - f.a);
                let d_alpha = da * (1.0 - f.prev[3]) + (0..3).map(|k| dc[k] * (f.c[k] - f.prev[k])).sum::<f64>();
                let dcol = [dc[0] * f.a, dc[1] * f.a, dc[2] * f.a];
                let i = f.i;
                let d_d;
                if !f.stroke {
                    let wv = smoothstep(f.d);
                    let d_w = if wv > 0.0 { s.fill[i][3] as f64 * d_alpha } else { 0.0 };
                    for k in 0..3 {
                        g.fill[i][k] += dcol[k];
                    }
                    g.fill[i][3] += d_alpha * wv;
                    d_d = if f.d < 0.0 { -d_smoothstep(f.d, d_w) } else { d_smoothstep(f.d, d_w) };
                } else {
                    let hw = s.hw[i] as f64;
                    let wv = smoothstep(f.d + hw) - smoothstep(f.d - hw);
                    let d_w = if wv > 0.0 { s.stroke[i][3] as f64 * d_alpha } else { 0.0 };
                    for k in 0..3 {
                        g.stroke[i][k] += dcol[k];
                    }
                    g.stroke[i][3] += d_alpha * wv;
                    let dp = d_smoothstep(f.d + hw, d_w);
                    let dm = -d_smoothstep(f.d - hw, d_w);
                    d_d = dp + dm;
                    g.hw[i] += dp - dm;
                }
                if let Some((j, t)) = f.near {
                    if d_d.abs() > 1e-10 {
                        let (p, idx) = seg(s, i, j);
                        let e = sub(bez(&p, t), q);
                        let dd = dot(e, e);
                        if dd >= 1e-10 {
                            let n = [d_d * e[0] / dd.sqrt(), d_d * e[1] / dd.sqrt()];
                            let u = 1.0 - t;
                            let wk = [u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t];
                            for k in 0..4 {
                                g.pts[2 * idx[k]] += n[0] * wk[k];
                                g.pts[2 * idx[k] + 1] += n[1] * wk[k];
                            }
                        }
                    }
                }
                for v in dc.iter_mut() {
                    *v *= 1.0 - f.a;
                }
                da = d_prev_a;
            }
        }
    }
    (out, total, g)
}

/// LIVE's xing loss over paths [first, len) times `scale` / their count, and its gradient to the points
pub fn xing(s: &Scene, first: usize, scale: f64) -> (f64, Vec<f64>) {
    let mut g = vec![0.0; s.points.len()];
    let n_paths = s.len().saturating_sub(first);
    if n_paths == 0 {
        return (0.0, g);
    }
    let sine = |a: V, b: V| (a[0] * b[1] - a[1] * b[0]) / (dot(a, a).sqrt() * dot(b, b).sqrt());
    let mut total = 0.0;
    for i in first..s.len() {
        let p = s.paths[i];
        if p.flags & CLOSED == 0 || p.nseg == 0 {
            continue;
        }
        let n = 3 * p.nseg as usize;
        let idx = |k: usize| p.off as usize + k % n;
        let at = |k: usize| [s.points[2 * idx(k)] as f64, s.points[2 * idx(k) + 1] as f64];
        let per = scale / n_paths as f64 / p.nseg as f64;
        for sgi in 0..p.nseg as usize {
            let k0 = 3 * sgi;
            let (x0, x1, x2, x3) = (at(k0), at(k0 + 1), at(k0 + 2), at(k0 + 3));
            let (v1, v2, v3) = (sub(x1, x0), sub(x2, x1), sub(x3, x2));
            let direct = sine(v1, v2) >= 0.0;
            let sa = sine(v1, v3);
            let gs = if direct {
                total += (-sa).max(0.0) * per;
                if sa < 0.0 { -1.0 } else { 0.0 }
            } else {
                total += sa.max(0.0) * per;
                if sa > 0.0 { 1.0 } else { 0.0 }
            };
            if gs == 0.0 {
                continue;
            }
            let gg = gs * per;
            let (n1, n3) = (dot(v1, v1).sqrt(), dot(v3, v3).sqrt());
            let c = v1[0] * v3[1] - v1[1] * v3[0];
            let dv1 = [v3[1] / (n1 * n3) - c * v1[0] / (n1 * n1 * n1 * n3), -v3[0] / (n1 * n3) - c * v1[1] / (n1 * n1 * n1 * n3)];
            let dv3 = [-v1[1] / (n1 * n3) - c * v3[0] / (n1 * n3 * n3 * n3), v1[0] / (n1 * n3) - c * v3[1] / (n1 * n3 * n3 * n3)];
            for (k, sign, dv) in [(k0 + 1, 1.0, dv1), (k0, -1.0, dv1), (k0 + 3, 1.0, dv3), (k0 + 2, -1.0, dv3)] {
                g[2 * idx(k)] += sign * gg * dv[0];
                g[2 * idx(k) + 1] += sign * gg * dv[1];
            }
        }
    }
    (total, g)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A blob (4 segments), a triangle-ish closed path over it, and an open stroke: the shapes the checks use
    pub fn scene() -> Scene {
        let mut s = Scene::new(24, 20);
        let c = [10.0f32, 9.0];
        let mut blob = Vec::new();
        for k in 0..12 {
            let a = k as f32 * std::f32::consts::PI * 2.0 / 12.0;
            let r = if k % 3 == 0 { 6.0 } else { 7.5 };
            blob.push([c[0] + r * a.cos(), c[1] + r * a.sin()]);
        }
        s.add(&blob, true, Some([0.9, 0.2, 0.1, 1.0]), None, EVENODD);
        s.add(&[[4.0, 15.0], [9.0, 6.0], [14.0, 4.0], [19.0, 3.0], [21.0, 10.0], [22.0, 14.0], [20.0, 17.0], [12.0, 18.0], [6.0, 18.0]], true,
              Some([0.1, 0.4, 0.8, 0.6]), None, 0);
        s.add(&[[2.0, 3.0], [8.0, 1.0], [14.0, 12.0], [21.0, 18.0]], false, None, Some(([0.2, 0.7, 0.2, 0.8], 1.2)), 0);
        s
    }

    fn target(s: &Scene) -> Vec<f32> {
        (0..(s.width * s.height) as usize * 3).map(|i| ((i * 37 % 101) as f32) / 100.0).collect()
    }

    /// The loss's gradient to every point, colour channel and half-width against central differences
    #[test]
    fn gradients_match_finite_differences() {
        let s = scene();
        let tg = target(&s);
        let wt: Vec<f32> = (0..(s.width * s.height) as usize).map(|i| 0.5 + (i % 7) as f32 / 14.0).collect();
        let d_img: Vec<f32> = (0..(s.width * s.height) as usize * 4).map(|i| ((i * 53 % 97) as f32) / 97.0 - 0.5).collect();
        for case in 0..2 {
            let loss = |sc: &Scene| -> f64 {
                if case == 0 {
                    run(sc, Some(&Loss::Mse { target: &tg, weight: Some(&wt), bg: [1.0, 1.0, 1.0] }), None).1
                } else {
                    let (px, _, _) = run(sc, None, None);
                    px.iter().enumerate().map(|(p, v)| (0..4).map(|k| v[k] * d_img[4 * p + k] as f64).sum::<f64>()).sum()
                }
            };
            let l = if case == 0 { Loss::Mse { target: &tg, weight: Some(&wt), bg: [1.0, 1.0, 1.0] } } else { Loss::Image { d_img: &d_img } };
            let (_, _, g) = run(&s, Some(&l), None);
            let fd = |set: &dyn Fn(&mut Scene, f32), e: f32| {
                let mut a = s.clone();
                set(&mut a, e);
                let mut b = s.clone();
                set(&mut b, -e);
                (loss(&a) - loss(&b)) / (2.0 * e as f64)
            };
            let mut worst = 0.0f64;
            let mut scale = 0.0f64;
            for k in 0..s.points.len() {
                let n = fd(&|sc, e| sc.points[k] += e, 0.01);
                worst = worst.max((n - g.pts[k]).abs());
                scale = scale.max(n.abs());
            }
            for i in 0..s.len() {
                for c in 0..4 {
                    if s.paths[i].flags & FILL != 0 {
                        let n = fd(&|sc, e| sc.fill[i][c] += e, 1e-3);
                        worst = worst.max((n - g.fill[i][c]).abs());
                        scale = scale.max(n.abs());
                    }
                    if s.paths[i].flags & STROKE != 0 {
                        let n = fd(&|sc, e| sc.stroke[i][c] += e, 1e-3);
                        worst = worst.max((n - g.stroke[i][c]).abs());
                        scale = scale.max(n.abs());
                    }
                }
                if s.paths[i].flags & STROKE != 0 {
                    let n = fd(&|sc, e| sc.hw[i] += e, 0.01);
                    worst = worst.max((n - g.hw[i]).abs());
                    scale = scale.max(n.abs());
                }
            }
            // the coverage is piecewise smooth (a pixel's nearest segment can change): a few percent of the largest
            assert!(worst <= 0.03 * scale, "case {case}: worst {worst:.3e} of {scale:.3e}");
        }
    }

    #[test]
    fn a_square_is_inside_and_its_edge_half_covered() {
        let mut s = Scene::new(10, 10);
        let l = |a: [f32; 2], b: [f32; 2]| [a, [a[0] + (b[0] - a[0]) / 3.0, a[1] + (b[1] - a[1]) / 3.0], [a[0] + 2.0 * (b[0] - a[0]) / 3.0, a[1] + 2.0 * (b[1] - a[1]) / 3.0]];
        let (a, b, c, d) = ([2.0, 2.0], [8.5, 2.0], [8.5, 8.0], [2.0, 8.0]);
        let pts: Vec<[f32; 2]> = [l(a, b), l(b, c), l(c, d), l(d, a)].concat();
        s.add(&pts, true, Some([0.0, 0.0, 0.0, 1.0]), None, 0);
        let (px, _, _) = run(&s, None, None);
        assert_eq!(px[5 * 10 + 5][3], 1.0);
        assert_eq!(px[0][3], 0.0);
        // the right edge at x 8.5 runs through pixel 8's centre: half covered
        assert!((px[5 * 10 + 8][3] - 0.5).abs() < 1e-9, "{}", px[5 * 10 + 8][3]);
    }

    #[test]
    fn xing_gradient_matches_finite_differences() {
        let mut s = Scene::new(10, 10);
        // a bow tie: the legs turn against each other
        s.add(&[[0.0, 0.0], [4.0, 4.0], [0.0, 4.0], [4.0, 0.0], [5.0, 2.0], [3.0, -1.0]], true, Some([1.0; 4]), None, 0);
        let (l, g) = xing(&s, 0, 1.0);
        assert!(l > 0.0);
        for k in 0..s.points.len() {
            let mut a = s.clone();
            a.points[k] += 1e-3;
            let mut b = s.clone();
            b.points[k] -= 1e-3;
            let n = (xing(&a, 0, 1.0).0 - xing(&b, 0, 1.0).0) / 2e-3;
            assert!((n - g[k]).abs() < 1e-3, "point {k}: {n} vs {}", g[k]);
        }
    }
}
