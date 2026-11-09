//! LIVE (Layer-wise Image Vectorization, Ma et al. 2022) as PyTorch-SVGRender runs it (pipelines/LIVE_pipeline.py,
//! painter/live): paths added a layer at a time where the picture is worst fitted (its SparseCoordInit: the error
//! quantised, the largest connected region of the most common level), each a circle of cubic segments in the target's
//! colour there; every layer optimised `iters` steps by Adam with LIVE's per-layer learning-rate decay, against the
//! MSE weighted by the unsigned distance to the newest paths (UDF) plus the xing loss. VectorFusion's styles
//! (painter/vectorfusion) change the primitive: pixel-art squares, black sketch strokes, coloured painting strokes.
//!
//! The differences from upstream, each for a reason: the rasterizer's gradients are DiffVG's prefiltered ones (the
//! analytic coverage over a pixel) where LIVE samples 2 x 2 jittered points and DiffVG's edge sampling - deterministic,
//! and a pixel's whole boundary term rather than a sample of it; the canvas keeps the picture's aspect (upstream
//! squashes it to a square); the strokes start where the error is (VectorFusion places them at random).

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use nextcuda_core::{DevBuf, Error, Gpu, Result};
use nextcuda_image::{Picture, SvgFitOutput, SvgFitRequest, SvgProgress};

use crate::dev::DevScene;
use crate::scene::{Scene, EVENODD, POLYGON};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Style {
    Iconography,
    PixelArt,
    Sketch,
    Painting,
}

impl Style {
    pub const ALL: [&'static str; 4] = ["iconography", "pixelart", "sketch", "painting"];

    pub fn parse(s: &str) -> Result<Style> {
        match s.trim().to_ascii_lowercase().replace(['-', '_', ' '], "").as_str() {
            "" | "iconography" | "icon" | "live" => Ok(Style::Iconography),
            "pixelart" | "pixel" => Ok(Style::PixelArt),
            "sketch" | "ink" | "lineart" => Ok(Style::Sketch),
            "painting" | "paint" => Ok(Style::Painting),
            o => Err(Error(format!("style {o:?}: one of {}", Style::ALL.join(", ")))),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Style::Iconography => "iconography",
            Style::PixelArt => "pixelart",
            Style::Sketch => "sketch",
            Style::Painting => "painting",
        }
    }
}

/// A fit's settings with every default filled in
#[derive(Clone, Debug)]
pub struct Plan {
    pub style: Style,
    pub width: u32,
    pub height: u32,
    pub schedule: Vec<u32>,
    pub iters: u32,
    pub segments: u32,
    /// a new circle's radius, canvas pixels (LIVE: 5 at 240)
    pub radius: f32,
    /// stroke half-width (DiffVG's stroke_width)
    pub hw: f32,
    pub udf: bool,
    pub xing_weight: f32,
    pub bg: [f32; 3],
    pub write_bg: bool,
    /// learning rates: points, fill, stroke colour, stroke half-width (LIVE's lr_base: 1, 0.01, 0.01, 0.1)
    pub lr: [f32; 4],
    pub preview_every: u32,
    pub seed: u64,
}

pub const DEFAULT_SIZE: u32 = 384;
pub const MAX_SIZE: u32 = 1024;
pub const MAX_PATHS: u32 = 1024;
pub const MAX_STEPS: u64 = 40_000;

impl Plan {
    pub fn new(req: &SvgFitRequest) -> Result<Plan> {
        let style = Style::parse(&req.style)?;
        let (pw, ph) = (req.picture.width.max(1), req.picture.height.max(1));
        let size = if req.size == 0 { DEFAULT_SIZE } else { req.size.clamp(64, MAX_SIZE) };
        let (width, height) = if pw >= ph {
            (size, ((size as f64 * ph as f64 / pw as f64).round() as u32).max(16))
        } else {
            (((size as f64 * pw as f64 / ph as f64).round() as u32).max(16), size)
        };
        let side = width.max(height) as f32;
        let (paths, iters, segments) = match style {
            Style::PixelArt => (if req.paths == 0 { 32 } else { req.paths.clamp(2, 128) }, if req.iters == 0 { 200 } else { req.iters }, 4),
            Style::Iconography => (if req.paths == 0 { 128 } else { req.paths }, if req.iters == 0 { 250 } else { req.iters },
                                   if req.segments == 0 { 4 } else { req.segments.clamp(1, 16) }),
            Style::Sketch | Style::Painting => (if req.paths == 0 { 96 } else { req.paths }, if req.iters == 0 { 250 } else { req.iters },
                                                if req.segments == 0 { if style == Style::Sketch { 2 } else { 1 } } else { req.segments.clamp(1, 8) }),
        };
        let paths = paths.min(MAX_PATHS);
        let schedule = if style == Style::PixelArt {
            vec![paths * paths]
        } else if !req.schedule.is_empty() {
            let s: Vec<u32> = req.schedule.iter().copied().filter(|&n| n > 0).collect();
            if s.iter().map(|&n| n as u64).sum::<u64>() > MAX_PATHS as u64 {
                return Err(Error(format!("schedule: at most {MAX_PATHS} paths in all")));
            }
            s
        } else {
            // VectorFusion's 'repeat' schedule (128 paths, 16 a layer): eight layers, the paths spread over them
            let layers = paths.clamp(1, 8);
            let each = paths / layers;
            let mut v = vec![each; layers as usize];
            for x in v.iter_mut().take((paths - each * layers) as usize) {
                *x += 1;
            }
            v
        };
        if schedule.is_empty() {
            return Err(Error("schedule: no paths".into()));
        }
        let iters = iters.clamp(1, 5000);
        let steps = schedule.len() as u64 * iters as u64;
        if steps > MAX_STEPS {
            return Err(Error(format!("{} layers x {iters} steps: at most {MAX_STEPS} steps a fit", schedule.len())));
        }
        let sw = if req.stroke_width > 0.0 { req.stroke_width } else if style == Style::Sketch { side / 160.0 } else { side / 60.0 };
        Ok(Plan {
            style,
            width,
            height,
            schedule,
            iters,
            segments,
            radius: 5.0 * side / 240.0,
            hw: (0.5 * sw).max(0.25),
            udf: req.udf.unwrap_or(style != Style::PixelArt),
            xing_weight: req.xing_weight.unwrap_or(0.01),
            bg: req.background.unwrap_or([1.0, 1.0, 1.0]),
            write_bg: req.background.is_some(),
            lr: [1.0, 0.01, 0.01, 0.1],
            preview_every: req.preview_every,
            seed: req.seed,
        })
    }

    pub fn steps(&self) -> u32 {
        self.schedule.len() as u32 * self.iters
    }
    pub fn paths(&self) -> u32 {
        self.schedule.iter().sum()
    }
    /// which parameter groups move: points, fill, stroke colour, stroke width (VectorFusion's style table)
    fn optimise(&self) -> [bool; 4] {
        match self.style {
            Style::Iconography => [true, true, false, false],
            Style::PixelArt => [false, true, false, false],
            Style::Sketch => [true, false, true, false],
            Style::Painting => [true, false, true, true],
        }
    }
    fn points_per_path(&self) -> usize {
        match self.style {
            Style::Iconography => 3 * self.segments as usize,
            Style::PixelArt => 12,
            _ => 3 * self.segments as usize + 1,
        }
    }
}

/// xorshift64*: the fit's random draws from its seed
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    pub fn uniform(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        ((self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32) / (1u64 << 24) as f32
    }
}

/// LIVE's SparseCoordInit: where the next paths go
pub struct SparseInit {
    w: usize,
    h: usize,
    map: Vec<u8>,
    count: BTreeMap<u8, i64>,
}

/// numpy's quantile (linear interpolation) of sorted values at q
fn quantile(sorted: &[f32], q: f64) -> f32 {
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let f = (pos - lo as f64) as f32;
    sorted[lo] + (sorted[hi] - sorted[lo]) * f
}

impl SparseInit {
    /// pred and gt: [H, W, 3] in 0..1
    pub fn new(pred: &[f32], gt: &[f32], w: usize, h: usize) -> SparseInit {
        let mut err: Vec<f32> = (0..w * h).map(|p| (0..3).map(|c| (pred[3 * p + c] - gt[3 * p + c]).powi(2)).sum::<f32>()).collect();
        for e in err.iter_mut() {
            if *e < 0.1 {
                *e = 0.0;
            }
        }
        let mut sorted = err.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut qs: Vec<f32> = (0..200).map(|i| quantile(&sorted, i as f64 / 199.0)).collect();
        qs.dedup();
        let bins: Vec<f32> = if qs.len() > 2 { qs[1..qs.len() - 1].to_vec() } else { Vec::new() };
        // np.digitize(x, bins, right=False): how many bins are <= x
        let map: Vec<u8> = err.iter().map(|&e| bins.partition_point(|&b| b <= e).min(255) as u8).collect();
        let mut count = BTreeMap::new();
        for &m in &map {
            *count.entry(m).or_insert(0i64) += 1;
        }
        if let Some(&k) = count.keys().next() {
            count.remove(&k);
        }
        SparseInit { w, h, map, count }
    }

    /// The next place: (x, y) in pixels
    pub fn next(&mut self, rng: &mut Rng) -> (f32, f32) {
        let Some(target) = self.count.iter().fold(None, |best: Option<(u8, i64)>, (&k, &v)| match best {
            Some((_, bv)) if bv >= v => best,
            _ => Some((k, v)),
        }).map(|(k, _)| k) else {
            return (rng.uniform() * self.w as f32, rng.uniform() * self.h as f32);
        };
        // the largest 4-connected region of that level (the first in raster order on a tie, as OpenCV labels)
        let (w, h) = (self.w, self.h);
        let mut label = vec![0u32; w * h];
        let mut best: Vec<usize> = Vec::new();
        let mut next = 1u32;
        let mut stack = Vec::new();
        for start in 0..w * h {
            if self.map[start] != target || label[start] != 0 {
                continue;
            }
            let mut comp = Vec::new();
            label[start] = next;
            stack.push(start);
            while let Some(p) = stack.pop() {
                comp.push(p);
                let (x, y) = (p % w, p / w);
                let mut visit = |q: usize| {
                    if self.map[q] == target && label[q] == 0 {
                        label[q] = next;
                        stack.push(q);
                    }
                };
                if x > 0 {
                    visit(p - 1);
                }
                if x + 1 < w {
                    visit(p + 1);
                }
                if y > 0 {
                    visit(p - w);
                }
                if y + 1 < h {
                    visit(p + w);
                }
            }
            next += 1;
            if comp.len() > best.len() {
                best = comp;
            }
        }
        if best.is_empty() {
            self.count.remove(&target);
            return self.next(rng);
        }
        best.sort_unstable();
        let n = best.len() as f64;
        let cx = best.iter().map(|&p| (p % w) as f64).sum::<f64>() / n;
        let cy = best.iter().map(|&p| (p / w) as f64).sum::<f64>() / n;
        let mut pick = best[0];
        let mut pd = f64::MAX;
        for &p in &best {
            let d = ((p % w) as f64 - cx).powi(2) + ((p / w) as f64 - cy).powi(2);
            if d < pd {
                pd = d;
                pick = p;
            }
        }
        let left = self.count.get(&target).copied().unwrap_or(0) - best.len() as i64;
        if left <= 0 {
            self.count.remove(&target);
        } else {
            self.count.insert(target, left);
        }
        for &p in &best {
            self.map[p] = 0;
        }
        ((pick % w) as f32, (pick / w) as f32)
    }
}

/// The target colour at a place (LIVE's color_ref, its coordinates clamped into the canvas)
fn colour_at(gt: &[f32], w: usize, h: usize, x: f32, y: f32) -> [f32; 3] {
    let xi = (x as i64).clamp(0, w as i64 - 1) as usize;
    let yi = (y as i64).clamp(0, h as i64 - 1) as usize;
    let p = yi * w + xi;
    [gt[3 * p], gt[3 * p + 1], gt[3 * p + 2]]
}

/// A path of the plan's style at (x, y), added to `s`
fn new_path(plan: &Plan, s: &mut Scene, gt: &[f32], x: f32, y: f32, rng: &mut Rng) {
    let (w, h) = (plan.width as usize, plan.height as usize);
    let c = colour_at(gt, w, h, x, y);
    match plan.style {
        Style::Iconography => {
            // LIVE's 'circle' init: 3 points a segment evenly round a circle
            let n = 3 * plan.segments as usize;
            let pts: Vec<[f32; 2]> = (0..n).map(|i| {
                let a = (i as f64 * 360.0 / n as f64).to_radians();
                [(a.cos() as f32) * plan.radius + x, (a.sin() as f32) * plan.radius + y]
            }).collect();
            s.add(&pts, true, Some([c[0], c[1], c[2], 1.0]), None, EVENODD);
        }
        Style::Sketch | Style::Painting => {
            // VectorFusion's stroke: a random walk of control points (0.1 of the canvas a step) from its start
            let side = plan.width.max(plan.height) as f32;
            let r = 0.1 * side;
            let mut p0 = [x, y];
            let mut pts = vec![p0];
            for _ in 0..plan.segments {
                for _ in 0..3 {
                    let q = [p0[0] + r * (rng.uniform() - 0.5), p0[1] + r * (rng.uniform() - 0.5)];
                    pts.push(q);
                    p0 = q;
                }
            }
            let colour = if plan.style == Style::Sketch { [0.0, 0.0, 0.0, 0.5 + 0.5 * rng.uniform()] } else { [c[0], c[1], c[2], 1.0] };
            s.add(&pts, false, None, Some((colour, plan.hw)), 0);
        }
        Style::PixelArt => unreachable!("pixel art is a grid"),
    }
}

/// The squares of a pixel-art grid, each its cell's mean colour
fn pixel_grid(plan: &Plan, s: &mut Scene, gt: &[f32]) {
    let n = plan.paths_side();
    let (w, h) = (plan.width as usize, plan.height as usize);
    let (cw, ch) = (plan.width as f32 / n as f32, plan.height as f32 / n as f32);
    for j in 0..n {
        for i in 0..n {
            let (x0, y0) = (i as f32 * cw, j as f32 * ch);
            let (x1, y1) = (x0 + cw, y0 + ch);
            let mut acc = [0.0f64; 3];
            let mut k = 0usize;
            for y in (y0 as usize)..(y1.ceil() as usize).min(h) {
                for x in (x0 as usize)..(x1.ceil() as usize).min(w) {
                    for c in 0..3 {
                        acc[c] += gt[3 * (y * w + x) + c] as f64;
                    }
                    k += 1;
                }
            }
            let m = |c: usize| (acc[c] / k.max(1) as f64) as f32;
            let line = |a: [f32; 2], b: [f32; 2]| [a, [a[0] + (b[0] - a[0]) / 3.0, a[1] + (b[1] - a[1]) / 3.0], [a[0] + 2.0 * (b[0] - a[0]) / 3.0, a[1] + 2.0 * (b[1] - a[1]) / 3.0]];
            let pts: Vec<[f32; 2]> = [line([x0, y0], [x1, y0]), line([x1, y0], [x1, y1]), line([x1, y1], [x0, y1]), line([x0, y1], [x0, y0])].concat();
            s.add(&pts, true, Some([m(0), m(1), m(2), 1.0]), None, POLYGON);
        }
    }
}

impl Plan {
    fn paths_side(&self) -> u32 {
        (self.schedule[0] as f64).sqrt().round() as u32
    }
}

/// The target [H, W, 3] in 0..1: the picture at the canvas's size, flattened on the background
pub fn target_of(pic: &Picture, plan: &Plan) -> Vec<f32> {
    let p = if pic.width == plan.width && pic.height == plan.height { pic.clone() } else { pic.resize_bilinear(plan.width, plan.height) };
    let ch = p.channels as usize;
    let n = (plan.width * plan.height) as usize;
    let mut out = vec![0f32; 3 * n];
    for i in 0..n {
        let a = if ch == 4 { p.data[4 * i + 3] as f32 / 255.0 } else { 1.0 };
        for c in 0..3 {
            let v = if ch >= 3 { p.data[ch * i + c] } else { p.data[ch * i] } as f32 / 255.0;
            out[3 * i + c] = v * a + plan.bg[c] * (1.0 - a);
        }
    }
    out
}

/// The fit (svg.h's kernels on `gpu`), progress each step; an error from `progress` stops it
pub fn fit(gpu: &Arc<Gpu>, req: &SvgFitRequest, progress: &mut dyn FnMut(&SvgProgress) -> Result<()>) -> Result<SvgFitOutput> {
    let t0 = Instant::now();
    let plan = Plan::new(req)?;
    let target = target_of(&req.picture, &plan);
    fit_target(gpu, &plan, &target, req.picture.width, req.picture.height, t0, progress).map(|(o, _)| o)
}

/// The fit of a target already at the canvas's size (`check` gives LIVE's own): what it made, and its scene
pub fn fit_target(gpu: &Arc<Gpu>, plan: &Plan, target: &[f32], out_w: u32, out_h: u32, t0: Instant,
                  progress: &mut dyn FnMut(&SvgProgress) -> Result<()>) -> Result<(SvgFitOutput, Scene)> {
    let (w, h) = (plan.width as usize, plan.height as usize);
    let px = w * h;
    let total = plan.paths() as usize;
    let mut sc = DevScene::new(gpu, plan.width, plan.height, total, total * plan.points_per_path())?;
    sc.set_bg(plan.bg)?;
    let f32buf = |n: usize| -> Result<DevBuf> {
        let b = DevBuf::f32(gpu, n)?;
        b.fill(0)?;
        Ok(b)
    };
    let tgt = DevBuf::from_f32(gpu, target)?;
    let cov = f32buf(4 * px)?;
    let weight = f32buf(px)?;
    let keep = f32buf(px)?;
    let dist = f32buf(px)?;
    let scratch = f32buf(4)?;
    let pixels = f32buf(4 * px)?;
    let mut host = Scene::new(plan.width, plan.height);
    let mut rng = Rng::new(plan.seed);
    let opt = plan.optimise();
    let mut pred: Vec<f32> = (0..px).flat_map(|_| plan.bg).collect();
    let mut step = 0u32;
    let of = plan.steps();
    let layers = plan.schedule.len() as u32;
    let mut last_loss = 0.0f32;
    let lo = [f32::NEG_INFINITY, 0.0, 0.0, 0.25];
    let hi = [f32::INFINITY, 1.0, 1.0, 0.25 * plan.width.max(plan.height) as f32];
    for (layer, &count) in plan.schedule.iter().enumerate() {
        let first = host.len();
        if plan.style == Style::PixelArt {
            pixel_grid(plan, &mut host, target);
        } else {
            let mut init = SparseInit::new(&pred, target, w, h);
            for _ in 0..count {
                let (x, y) = init.next(&mut rng);
                new_path(plan, &mut host, target, x, y, &mut rng);
            }
        }
        sc.add(&host, first, step as i32, opt)?;
        if plan.style == Style::Sketch {
            sc.freeze_stroke_rgb(first)?;
        }
        for it in 0..plan.iters {
            sc.bounds()?;
            sc.loss.fill(0)?;
            if plan.udf {
                sc.render(&cov, 2, false, first, true)?;
                sc.udf(&cov, (layer > 0).then_some(&keep), &weight, &dist, &scratch)?;
            }
            let last = it + 1 == plan.iters;
            sc.grad(0, Some(&tgt), plan.udf.then_some(&weight), None, last.then_some(&pixels))?;
            if plan.style == Style::Iconography && plan.xing_weight > 0.0 {
                sc.xing(first, plan.xing_weight)?;
            }
            sc.adam(step as i32, plan.lr, plan.iters as i32, 0.4, (0.9, 0.999), 1e-8, lo, hi)?;
            let l = sc.loss.to_f32()?;
            last_loss = l[0];
            let last_xing = l[1];
            step += 1;
            let want = last || (plan.preview_every > 0 && step % plan.preview_every == 0);
            let svg = if want { Some(sc.read()?.to_svg(out_w, out_h, plan.write_bg.then_some(plan.bg))) } else { None };
            progress(&SvgProgress { at: step, of, layer: layer as u32 + 1, layers, paths: sc.n as u32, loss: last_loss, xing: last_xing, svg,
                                    seconds: t0.elapsed().as_secs_f64() })?;
        }
        if plan.udf {
            keep.copy_within(0, &weight, 0, 4 * px)?;
        }
        // the next layer starts where this one's picture is worst (rendered after its last step)
        sc.bounds()?;
        sc.render(&pixels, 0, true, 0, false)?;
        let all = pixels.to_f32()?;
        pred = (0..px).flat_map(|p| [all[4 * p], all[4 * p + 1], all[4 * p + 2]]).collect();
    }
    let scene = sc.read()?;
    let mse = pred.iter().zip(target).map(|(a, b)| ((a - b) as f64).powi(2)).sum::<f64>() / pred.len() as f64;
    let psnr = if mse > 0.0 { 10.0 * (1.0 / mse).log10() } else { 99.0 };
    let rgb: Vec<u8> = pred.iter().map(|&v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8).collect();
    let mut render = Picture::new(plan.width, plan.height, 3);
    render.data = rgb;
    Ok((SvgFitOutput {
        svg: scene.to_svg(out_w, out_h, plan.write_bg.then_some(plan.bg)),
        paths: scene.len() as u32,
        steps: step,
        layers: plan.schedule.len() as u32,
        width: plan.width,
        height: plan.height,
        loss: last_loss,
        mse: mse as f32,
        psnr: psnr as f32,
        render: Some(render),
        seconds: t0.elapsed().as_secs_f64(),
    }, scene))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plans_fill_in_their_defaults_and_refuse_too_much() {
        let mut r = SvgFitRequest { picture: Picture::new(800, 400, 3), ..Default::default() };
        let p = Plan::new(&r).unwrap();
        assert_eq!((p.width, p.height), (384, 192));
        assert_eq!(p.paths(), 128);
        assert_eq!(p.schedule, vec![16; 8]);
        r.paths = 20;
        assert_eq!(Plan::new(&r).unwrap().schedule, vec![3, 3, 3, 3, 2, 2, 2, 2]);
        r.style = "pixel-art".into();
        r.paths = 0;
        assert_eq!(Plan::new(&r).unwrap().schedule, vec![32 * 32]);
        r.style = "sketch".into();
        r.schedule = vec![1, 2, 4];
        let p = Plan::new(&r).unwrap();
        assert_eq!((p.schedule.clone(), p.segments, p.points_per_path()), (vec![1, 2, 4], 2, 7));
        r.schedule = vec![600, 600];
        assert!(Plan::new(&r).is_err());
        r.style = "oil".into();
        assert!(Plan::new(&r).is_err());
    }

    #[test]
    fn sparse_init_goes_to_the_largest_wrong_region_first() {
        let (w, h) = (20, 10);
        let pred = vec![1.0f32; w * h * 3];
        let mut gt = pred.clone();
        // a 3 x 3 region and a 5 x 4 region, a little and very wrong: both above the quantiles' one cut (a map of two
        // values has none, and upstream places the paths at random then)
        for (x0, y0, x1, y1, v) in [(1, 1, 4, 4, 0.4), (10, 2, 15, 6, 0.0)] {
            for y in y0..y1 {
                for x in x0..x1 {
                    for c in 0..3 {
                        gt[3 * (y * w + x) + c] = v;
                    }
                }
            }
        }
        let mut s = SparseInit::new(&pred, &gt, w, h);
        let mut rng = Rng::new(1);
        // the larger region's pixel nearest its centroid (12, 3.5): (12, 3), the first of the tie in raster order
        assert_eq!(s.next(&mut rng), (12.0, 3.0));
        assert_eq!(s.next(&mut rng), (2.0, 2.0));
        // nothing left wrong: a random place inside the canvas
        let (x, y) = s.next(&mut rng);
        assert!((0.0..w as f32).contains(&x) && (0.0..h as f32).contains(&y));
    }

    #[test]
    fn the_rng_is_uniform_enough() {
        let mut r = Rng::new(7);
        let v: Vec<f32> = (0..10000).map(|_| r.uniform()).collect();
        let mean = v.iter().sum::<f32>() / v.len() as f32;
        assert!(v.iter().all(|x| (0.0..1.0).contains(x)));
        assert!((mean - 0.5).abs() < 0.02, "{mean}");
    }
}
