//! svgrender: PyTorch-SVGRender's technique in Rust and CUDA - a differentiable vector rasterizer (DiffVG's
//! prefiltered rendering of cubic Bezier paths, fill and stroke, and its gradients to every control point, colour and
//! stroke width: `kernels/image/svgrender`) and the optimiser over SVG parameters that runs on it (LIVE's layer-wise
//! image-to-SVG, VectorFusion's styles: `live`). No weights: the engine is the rasterizer and the optimiser, and a
//! picture to fit. Text to SVG is a picture model's picture fitted by this (the front door's image.svg_generate).

pub mod check;
pub mod dev;
pub mod ffi;
pub mod host;
pub mod live;
pub mod scene;

use std::sync::Arc;
use std::time::Instant;

use nextcuda_core::DevBuf;
use nextcuda_image::{Defaults, Error, Gpu, ImageEngine, ImageKind, ImageRequest, LoadOptions, ModelFiles, Picture, Result, Sampler, Schedule, Step,
                     SvgFitOutput, SvgFitRequest, SvgProgress};

pub const ARCH: &str = "svgrender";

pub fn kind() -> ImageKind {
    ImageKind { archs: &[ARCH], name: "svgrender: pictures fitted with SVG paths (DiffVG's rasterizer and LIVE's optimiser, Rust + CUDA)", roles: &[], load,
                options: &[] }
}

fn load(_files: &ModelFiles, gpus: &[Arc<Gpu>], _o: &LoadOptions, _log: &mut dyn FnMut(String)) -> Result<Box<dyn ImageEngine>> {
    let gpu = gpus.first().ok_or_else(|| Error("no GPU".into()))?;
    Ok(Box::new(SvgRender::load(gpu)?))
}

pub struct SvgRender {
    pub gpu: Arc<Gpu>,
    load_seconds: f64,
}

impl SvgRender {
    pub fn load(gpu: &Arc<Gpu>) -> Result<SvgRender> {
        let t0 = Instant::now();
        // the kernels are there (a library without them fails here, not at the first request)
        ffi::api()?;
        Ok(SvgRender { gpu: gpu.clone(), load_seconds: t0.elapsed().as_secs_f64() })
    }
}

impl ImageEngine for SvgRender {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn defaults(&self) -> Defaults {
        Defaults { width: live::DEFAULT_SIZE, height: live::DEFAULT_SIZE, steps: 250, cfg: 1.0, sampler: Sampler::Euler, schedule: Schedule::Simple, shift: 1.0 }
    }
    fn guidance(&self) -> bool {
        false
    }
    fn shift(&self) -> bool {
        false
    }
    fn takes_loras(&self) -> bool {
        false
    }
    fn load_seconds(&self) -> f64 {
        self.load_seconds
    }
    fn generate(&self, _req: &ImageRequest, _progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<Vec<Picture>> {
        Err(Error("svgrender fits pictures with paths (image.svg_fit); it does not draw from a prompt (image.svg_generate draws first)".into()))
    }
    fn report(&self) -> Vec<String> {
        vec![format!("{}: the vector rasterizer and LIVE's optimiser, no weights", self.gpu.name)]
    }
    fn fits_svg(&self) -> bool {
        true
    }
    fn svg_fit(&self, req: &SvgFitRequest, progress: &mut dyn FnMut(&SvgProgress) -> Result<()>) -> Result<SvgFitOutput> {
        live::fit(&self.gpu, req, progress).map_err(|e| Error(nextcuda_image::tag_device_oom(&e.0, None)))
    }
}

/// The kernels against the host's float64 arithmetic on a few scenes (fills, an even-odd blob, an open stroke):
/// pixels and every gradient, both modes; the worst relative errors (max |gpu - host| / max |host|), by what
pub fn selftest(gpu: &Arc<Gpu>, log: &mut dyn FnMut(String)) -> Result<Vec<(String, f64)>> {
    let mut out = Vec::new();
    for (name, s) in check::scenes() {
        let (w, h) = (s.width as usize, s.height as usize);
        let px = w * h;
        let mut dv = dev::DevScene::new(gpu, s.width, s.height, s.len(), s.points.len() / 2)?;
        dv.add(&s, 0, 0, [true; 4])?;
        dv.bounds()?;
        let tgt: Vec<f32> = (0..3 * px).map(|i| ((i * 37 % 101) as f32) / 100.0).collect();
        let d_img: Vec<f32> = (0..4 * px).map(|i| ((i * 53 % 97) as f32) / 97.0 - 0.5).collect();
        let tgt_d = DevBuf::from_f32(gpu, &tgt)?;
        let dimg_d = DevBuf::from_f32(gpu, &d_img)?;
        let pix = DevBuf::f32(gpu, 4 * px)?;
        for mode in 0..2 {
            dv.zero_grads()?;
            dv.loss.fill(0)?;
            if mode == 0 {
                dv.grad(0, Some(&tgt_d), None, None, Some(&pix))?;
            } else {
                dv.grad(1, None, None, Some(&dimg_d), Some(&pix))?;
            }
            let gp = pix.to_f32()?;
            let (gpts, gfill, gstroke, ghw) = dv.read_grads()?;
            let loss = if mode == 0 { host::Loss::Mse { target: &tgt, weight: None, bg: [1.0, 1.0, 1.0] } } else { host::Loss::Image { d_img: &d_img } };
            let (hp, hl, hg) = host::run(&s, Some(&loss), None);
            let rel = |a: &[f32], b: &[f64]| {
                let m = b.iter().fold(0.0f64, |x, v| x.max(v.abs())).max(1e-12);
                a.iter().zip(b).fold(0.0f64, |x, (p, q)| x.max((*p as f64 - q).abs())) / m
            };
            let hpx: Vec<f64> = hp.iter().flatten().copied().collect();
            let mut rows = vec![("pixels".to_string(), rel(&gp, &hpx)), ("points".to_string(), rel(&gpts, &hg.pts))];
            let flat4 = |v: &[[f32; 4]]| v.iter().flatten().copied().collect::<Vec<f32>>();
            let flat4h = |v: &[[f64; 4]]| v.iter().flatten().copied().collect::<Vec<f64>>();
            if s.paths.iter().any(|p| p.flags & scene::FILL != 0) {
                rows.push(("fill".into(), rel(&flat4(&gfill), &flat4h(&hg.fill))));
            }
            if s.paths.iter().any(|p| p.flags & scene::STROKE != 0) {
                rows.push(("stroke".into(), rel(&flat4(&gstroke), &flat4h(&hg.stroke))));
                rows.push(("stroke width".into(), rel(&ghw, &hg.hw)));
            }
            if mode == 0 {
                let l = dv.loss.to_f32()?[0] as f64;
                rows.push(("loss".into(), (l - hl).abs() / hl.abs().max(1e-12)));
            }
            for (what, e) in rows {
                let label = format!("{name}, {}: {what}", if mode == 0 { "MSE" } else { "d_img" });
                log(format!("{label:<44} {e:.2e}"));
                out.push((label, e));
            }
        }
    }
    Ok(out)
}

/// A LIVE step's time on `size` x `size` with `paths` paths of 4 segments (blobs of LIVE's size range, in random
/// places and colours, half of them translucent): (ms a whole step, ms the backward pass alone, ms a prefiltered render)
pub fn bench(gpu: &Arc<Gpu>, size: u32, paths: u32) -> Result<(f64, f64, f64)> {
    let mut rng = live::Rng::new(42);
    let mut s = scene::Scene::new(size, size);
    for _ in 0..paths {
        let (cx, cy) = (rng.uniform() * size as f32, rng.uniform() * size as f32);
        let r = size as f32 * (0.02 + 0.15 * rng.uniform());
        let pts: Vec<[f32; 2]> = (0..12).map(|k| {
            let a = k as f32 * std::f32::consts::PI / 6.0;
            let rr = r * (0.7 + 0.6 * rng.uniform());
            [cx + rr * a.cos(), cy + rr * a.sin()]
        }).collect();
        let alpha = if rng.uniform() < 0.5 { 1.0 } else { 0.6 };
        s.add(&pts, true, Some([rng.uniform(), rng.uniform(), rng.uniform(), alpha]), None, scene::EVENODD);
    }
    let px = (size * size) as usize;
    let mut dv = dev::DevScene::new(gpu, size, size, s.len(), s.points.len() / 2)?;
    dv.add(&s, 0, 0, [true, true, false, false])?;
    let tgt: Vec<f32> = (0..3 * px).map(|_| rng.uniform()).collect();
    let tgt = DevBuf::from_f32(gpu, &tgt)?;
    let (cov, weight, keep, dist, scratch) = (DevBuf::f32(gpu, 4 * px)?, DevBuf::f32(gpu, px)?, DevBuf::f32(gpu, px)?, DevBuf::f32(gpu, px)?, DevBuf::f32(gpu, 4)?);
    keep.fill(0)?;
    let first = s.len() * 7 / 8;
    let step = |dv: &dev::DevScene, k: i32| -> Result<()> {
        dv.bounds()?;
        dv.loss.fill(0)?;
        dv.render(&cov, 2, false, first, true)?;
        dv.udf(&cov, Some(&keep), &weight, &dist, &scratch)?;
        dv.grad(0, Some(&tgt), Some(&weight), None, None)?;
        dv.xing(first, 0.01)?;
        dv.adam(k, [1.0, 0.01, 0.01, 0.1], 500, 0.4, (0.9, 0.999), 1e-8, [f32::NEG_INFINITY, 0.0, 0.0, 0.25], [f32::INFINITY, 1.0, 1.0, 100.0])?;
        dv.loss.to_f32().map(|_| ())
    };
    for k in 0..5 {
        step(&dv, k)?;
    }
    let n = 40;
    let t = Instant::now();
    for k in 0..n {
        step(&dv, 5 + k)?;
    }
    let whole = t.elapsed().as_secs_f64() * 1e3 / n as f64;
    dv.bounds()?;
    gpu.sync()?;
    let t = Instant::now();
    for _ in 0..n {
        dv.grad(0, Some(&tgt), Some(&weight), None, None)?;
    }
    gpu.sync()?;
    let back = t.elapsed().as_secs_f64() * 1e3 / n as f64;
    let t = Instant::now();
    for _ in 0..n {
        dv.render(&cov, 0, true, 0, false)?;
    }
    gpu.sync()?;
    let fwd = t.elapsed().as_secs_f64() * 1e3 / n as f64;
    Ok((whole, back, fwd))
}
