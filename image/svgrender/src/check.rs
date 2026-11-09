//! The kernels and the optimiser against DiffVG and LIVE (`reference/svgrender/ref.py`'s dumps): the prefiltered
//! pixels and every gradient on a few scenes, the hard-sampled render against DiffVG's 16 x 16 jittered one, LIVE's
//! distance weight, its xing loss and gradient, its SparseCoordInit, and a whole LIVE fit's quality and time.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use nextcuda_core::{DevBuf, Error, Gpu, Result};
use nextcuda_image::{Picture, SvgFitRequest};
use serde_json::Value;

use crate::dev::DevScene;
use crate::live::{self, Plan, Rng, SparseInit};
use crate::scene::{Scene, EVENODD};

/// A .npy file of float32 (or uint8): (shape, values)
pub fn read_npy(path: &Path) -> Result<(Vec<usize>, Vec<f32>)> {
    let b = std::fs::read(path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
    if b.len() < 10 || &b[..6] != b"\x93NUMPY" {
        return Err(Error(format!("{}: not a .npy", path.display())));
    }
    let (hl, at) = if b[6] == 1 { (u16::from_le_bytes([b[8], b[9]]) as usize, 10) } else { (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12) };
    let head = String::from_utf8_lossy(&b[at..at + hl]).to_string();
    let shape: Vec<usize> = head.split("'shape': (").nth(1).and_then(|s| s.split(')').next())
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect()).unwrap_or_default();
    let d = &b[at + hl..];
    let v = if head.contains("'<f4'") {
        d.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    } else if head.contains("'|u1'") {
        d.iter().map(|x| *x as f32).collect()
    } else {
        return Err(Error(format!("{}: {head}", path.display())));
    };
    Ok((shape, v))
}

/// A float32 .npy of `shape` (the check's own output beside the reference's, with NS_SVG_DUMP=1)
pub fn write_npy(path: &Path, shape: &[usize], v: &[f32]) -> Result<()> {
    let dims = shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ");
    let mut head = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({dims}{}), }}", if shape.len() == 1 { "," } else { "" });
    while (10 + head.len() + 1) % 64 != 0 {
        head.push(' ');
    }
    head.push('\n');
    let mut b = b"\x93NUMPY\x01\x00".to_vec();
    b.extend_from_slice(&(head.len() as u16).to_le_bytes());
    b.extend_from_slice(head.as_bytes());
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    std::fs::write(path, b).map_err(|e| Error(format!("{}: {e}", path.display())))
}

/// (relative L2 error, max |difference|)
pub fn compare(got: &[f32], want: &[f32]) -> (f64, f64) {
    let (mut e2, mut w2, mut mx) = (0f64, 0f64, 0f64);
    for (g, w) in got.iter().zip(want) {
        let (g, w) = (*g as f64, *w as f64);
        e2 += (g - w) * (g - w);
        w2 += w * w;
        mx = mx.max((g - w).abs());
    }
    ((e2 / w2.max(1e-30)).sqrt(), mx)
}

/// A scene as ref.py writes it
pub fn scene_of(v: &Value) -> Result<Scene> {
    let n = |k: &str| v[k].as_u64().ok_or_else(|| Error(format!("scene: no {k}")));
    let mut s = Scene::new(n("width")? as u32, n("height")? as u32);
    for p in v["paths"].as_array().ok_or_else(|| Error("scene: no paths".into()))? {
        let pts: Vec<[f32; 2]> = p["points"].as_array().into_iter().flatten()
            .map(|q| [q[0].as_f64().unwrap_or(0.0) as f32, q[1].as_f64().unwrap_or(0.0) as f32]).collect();
        let c4 = |k: &str| p[k].as_array().map(|a| {
            let f = |i: usize| a.get(i).and_then(Value::as_f64).unwrap_or(0.0) as f32;
            [f(0), f(1), f(2), f(3)]
        });
        let stroke = c4("stroke").map(|c| (c, p["hw"].as_f64().unwrap_or(0.0) as f32));
        let flags = if p["evenodd"].as_bool().unwrap_or(true) { EVENODD } else { 0 };
        s.add(&pts, p["closed"].as_bool().unwrap_or(true), c4("fill"), stroke, flags);
    }
    Ok(s)
}

/// selftest's scenes: what ref.py's raster cases are made of, smaller
pub fn scenes() -> Vec<(&'static str, Scene)> {
    let blob = |cx: f32, cy: f32, r0: f32, r1: f32, ph: f32| -> Vec<[f32; 2]> {
        (0..12).map(|k| {
            let a = ph + k as f32 * std::f32::consts::PI * 2.0 / 12.0;
            let r = if k % 3 == 0 { r0 } else { r1 };
            [cx + r * a.cos(), cy + r * a.sin()]
        }).collect()
    };
    let mut fills = Scene::new(40, 32);
    fills.add(&blob(16.0, 15.0, 9.0, 12.0, 0.0), true, Some([0.9, 0.2, 0.1, 1.0]), None, EVENODD);
    fills.add(&blob(26.0, 17.0, 7.0, 10.0, 0.4), true, Some([0.1, 0.4, 0.8, 0.6]), None, EVENODD);
    fills.add(&[[4.0, 27.0], [13.0, 20.0], [20.0, 31.0], [33.0, 29.0], [39.0, 27.0], [39.5, 7.0], [33.0, 4.0], [20.0, 3.0], [7.0, 9.0]], true,
              Some([0.2, 0.8, 0.3, 0.45]), None, 0);
    let mut strokes = Scene::new(40, 28);
    strokes.add(&[[3.0, 4.0], [14.0, 1.0], [21.0, 24.0], [37.0, 21.0]], false, None, Some(([0.1, 0.1, 0.1, 1.0], 1.5)), 0);
    strokes.add(&[[4.0, 24.0], [11.0, 7.0], [20.0, 7.0], [24.0, 14.0], [28.0, 21.0], [35.0, 8.0], [35.0, 3.0]], false, None,
                Some(([0.8, 0.2, 0.5, 0.7], 2.5)), 0);
    strokes.add(&blob(21.0, 14.0, 6.0, 9.0, 0.0), true, None, Some(([0.1, 0.6, 0.6, 0.8], 0.8)), 0);
    vec![("fills", fills), ("strokes", strokes)]
}

/// Everything ref.py dumped under `dir` that is here; the worst relative error of the rasterizer's exact parts
pub fn run(gpu: &Arc<Gpu>, dir: &Path, only: &[&str], log: &mut dyn FnMut(String)) -> Result<f64> {
    let mut worst = 0.0f64;
    let want = |w: &str| only.is_empty() || only.contains(&w);
    if want("raster") {
        worst = worst.max(raster(gpu, &dir.join("raster"), log)?);
    }
    if want("udf") {
        udf(gpu, &dir.join("udf"), log)?;
    }
    if want("xing") {
        worst = worst.max(xing(gpu, &dir.join("xing"), log)?);
    }
    if want("sparse") {
        sparse(&dir.join("sparse"), log)?;
    }
    if want("live") {
        live_fit(gpu, &dir.join("live"), log)?;
    }
    Ok(worst)
}

fn cases(dir: &Path) -> Result<Vec<(String, std::path::PathBuf)>> {
    let mut v: Vec<_> = std::fs::read_dir(dir).map_err(|e| Error(format!("{}: {e}", dir.display())))?.flatten()
        .filter(|e| e.path().is_dir()).map(|e| (e.file_name().to_string_lossy().to_string(), e.path())).collect();
    v.sort();
    Ok(v)
}

fn raster(gpu: &Arc<Gpu>, dir: &Path, log: &mut dyn FnMut(String)) -> Result<f64> {
    let mut worst = 0.0f64;
    for (name, d) in cases(dir)? {
        let v: Value = serde_json::from_slice(&std::fs::read(d.join("scene.json")).map_err(|e| Error(e.to_string()))?).map_err(|e| Error(e.to_string()))?;
        let mut s = scene_of(&v)?;
        // DiffVG's own cull, so its prefiltered edges are what is compared (ours keep the pixel outside the box)
        for p in s.paths.iter_mut() {
            p.flags |= crate::scene::DIFFVG_BOX;
        }
        let px = (s.width * s.height) as usize;
        let mut dv = DevScene::new(gpu, s.width, s.height, s.len(), s.points.len() / 2)?;
        dv.add(&s, 0, 0, [true; 4])?;
        dv.bounds()?;
        let d_img = DevBuf::from_f32(gpu, &read_npy(&d.join("d_img.npy"))?.1)?;
        let pix = DevBuf::f32(gpu, 4 * px)?;
        dv.zero_grads()?;
        let t = Instant::now();
        dv.grad(1, None, None, Some(&d_img), Some(&pix))?;
        gpu.sync()?;
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let (gp, gf, gs, gh) = dv.read_grads()?;
        let flat = |v: &[[f32; 4]]| v.iter().flatten().copied().collect::<Vec<f32>>();
        if std::env::var("NS_SVG_DUMP").is_ok_and(|v| v == "1") {
            write_npy(&d.join("nextcuda_pre.npy"), &[s.height as usize, s.width as usize, 4], &pix.to_f32()?)?;
            write_npy(&d.join("nextcuda_g_points.npy"), &[gp.len() / 2, 2], &gp)?;
        }
        let mut rows = vec![("pixels (prefiltered)", compare(&pix.to_f32()?, &read_npy(&d.join("pre.npy"))?.1)),
                            ("d points", compare(&gp, &read_npy(&d.join("g_points.npy"))?.1))];
        if s.paths.iter().any(|p| p.flags & crate::scene::FILL != 0) {
            rows.push(("d fill", compare(&flat(&gf), &read_npy(&d.join("g_fill.npy"))?.1)));
        }
        if s.paths.iter().any(|p| p.flags & crate::scene::STROKE != 0) {
            rows.push(("d stroke", compare(&flat(&gs), &read_npy(&d.join("g_stroke.npy"))?.1)));
            rows.push(("d stroke width", compare(&gh, &read_npy(&d.join("g_hw.npy"))?.1)));
        }
        for (what, (rel, mx)) in rows {
            log(format!("raster {name:<9} {what:<22} rel L2 {rel:.2e}  max |diff| {mx:.2e}"));
            worst = worst.max(rel);
        }
        // the hard samples (a regular 16 x 16 grid) against DiffVG's jittered 16 x 16: the same coverage up to sampling
        dv.render(&pix, 16, false, 0, false)?;
        let (rel, mx) = compare(&pix.to_f32()?, &read_npy(&d.join("samp16.npy"))?.1);
        log(format!("raster {name:<9} {:<22} rel L2 {rel:.2e}  max |diff| {mx:.2e}  (sampling noise)  [backward {ms:.2} ms]", "pixels (16x16 samples)"));
    }
    Ok(worst)
}

fn udf(gpu: &Arc<Gpu>, dir: &Path, log: &mut dyn FnMut(String)) -> Result<()> {
    for (name, d) in cases(dir)? {
        let (shape, cov) = read_npy(&d.join("cov.npy"))?;
        let (h, w) = (shape[0], shape[1]);
        let want = read_npy(&d.join("weight.npy"))?.1;
        let rgba: Vec<f32> = cov.iter().flat_map(|&a| [0.0, 0.0, 0.0, a]).collect();
        let dv = DevScene::new(gpu, w as u32, h as u32, 1, 1)?;
        let c = DevBuf::from_f32(gpu, &rgba)?;
        let (wt, dist, scratch) = (DevBuf::f32(gpu, w * h)?, DevBuf::f32(gpu, w * h)?, DevBuf::f32(gpu, 4)?);
        dv.udf(&c, None, &wt, &dist, &scratch)?;
        let got = wt.to_f32()?;
        let (rel, mx) = compare(&got, &want);
        let mean = got.iter().zip(&want).map(|(a, b)| (a - b).abs() as f64).sum::<f64>() / got.len() as f64;
        log(format!("udf    {name:<9} weight                 rel L2 {rel:.2e}  max |diff| {mx:.2e}  mean |diff| {mean:.2e}  (skfmm's level set vs nearest pixels)"));
    }
    Ok(())
}

fn xing(gpu: &Arc<Gpu>, dir: &Path, log: &mut dyn FnMut(String)) -> Result<f64> {
    let v: Value = serde_json::from_slice(&std::fs::read(dir.join("case.json")).map_err(|e| Error(e.to_string()))?).map_err(|e| Error(e.to_string()))?;
    let mut s = Scene::new(64, 64);
    for p in v["points"].as_array().into_iter().flatten() {
        let pts: Vec<[f32; 2]> = p.as_array().into_iter().flatten().map(|q| [q[0].as_f64().unwrap() as f32, q[1].as_f64().unwrap() as f32]).collect();
        s.add(&pts, true, Some([1.0; 4]), None, 0);
    }
    let mut dv = DevScene::new(gpu, 64, 64, s.len(), s.points.len() / 2)?;
    dv.add(&s, 0, 0, [true; 4])?;
    dv.zero_grads()?;
    dv.loss.fill(0)?;
    dv.xing(0, v["weight"].as_f64().unwrap_or(0.01) as f32)?;
    let l = dv.loss.to_f32()?[1] as f64;
    let want = v["loss"].as_f64().unwrap_or(0.0);
    let gw: Vec<f32> = v["grads"].as_array().into_iter().flatten().flat_map(|p| p.as_array().cloned().unwrap_or_default())
        .flat_map(|q| [q[0].as_f64().unwrap() as f32, q[1].as_f64().unwrap() as f32]).collect();
    let (gp, _, _, _) = dv.read_grads()?;
    let (rel, mx) = compare(&gp, &gw);
    let lrel = (l - want).abs() / want.abs().max(1e-30);
    log(format!("xing   loss {l:.6e} (LIVE {want:.6e}, rel {lrel:.1e}); d points rel L2 {rel:.2e} max |diff| {mx:.2e}"));
    Ok(rel.max(lrel))
}

fn sparse(dir: &Path, log: &mut dyn FnMut(String)) -> Result<()> {
    for (name, d) in cases(dir)? {
        let (shape, gt) = read_npy(&d.join("gt.npy"))?;
        let pred = read_npy(&d.join("pred.npy"))?.1;
        let want: Vec<Vec<f64>> = serde_json::from_slice(&std::fs::read(d.join("coords.json")).map_err(|e| Error(e.to_string()))?).map_err(|e| Error(e.to_string()))?;
        let mut s = SparseInit::new(&pred, &gt, shape[1], shape[0]);
        let mut rng = Rng::new(0);
        let got: Vec<(f32, f32)> = (0..want.len()).map(|_| s.next(&mut rng)).collect();
        let same = got.iter().zip(&want).filter(|(g, w)| g.0 as f64 == w[0] && g.1 as f64 == w[1]).count();
        log(format!("sparse {name:<9} {same} of {} coordinates LIVE's (first {:?}, LIVE {:?})", want.len(), got[0], want[0]));
    }
    Ok(())
}

/// Our LIVE on LIVE's target with its settings, beside upstream's result: the PSNR of each over white at 4 x 4 samples
fn live_fit(gpu: &Arc<Gpu>, dir: &Path, log: &mut dyn FnMut(String)) -> Result<()> {
    for (name, d) in cases(dir)? {
        let Ok(b) = std::fs::read(d.join("result.json")) else { continue };
        let r: Value = serde_json::from_slice(&b).map_err(|e| Error(e.to_string()))?;
        if r["prefilter"] == true {
            log(format!("live   {name:<20} upstream (DiffVG prefiltered) PSNR {:.2} dB in {:.0} s on the CPU", r["psnr"].as_f64().unwrap_or(0.0),
                        r["seconds"].as_f64().unwrap_or(0.0)));
            continue;
        }
        let pic = r["picture"].as_str().unwrap_or("");
        let (shape, target) = read_npy(&dir.join(format!("{pic}-target.npy")))?;
        let size = r["size"].as_u64().unwrap_or(240) as u32;
        let (paths, layers, iters) = (r["paths"].as_u64().unwrap_or(32) as u32, r["layers"].as_u64().unwrap_or(8) as u32, r["iters"].as_u64().unwrap_or(200) as u32);
        let req = SvgFitRequest { picture: Picture::new(shape[1] as u32, shape[0] as u32, 3), size, schedule: vec![paths / layers; layers as usize], iters,
                                  ..Default::default() };
        let plan = Plan::new(&req)?;
        let t0 = Instant::now();
        let (out, scene) = live::fit_target(gpu, &plan, &target, size, size, t0, &mut |_| Ok(()))?;
        let mut dv = DevScene::new(gpu, scene.width, scene.height, scene.len(), scene.points.len() / 2)?;
        dv.add(&scene, 0, 0, [false; 4])?;
        dv.bounds()?;
        let pix = DevBuf::f32(gpu, 4 * (scene.width * scene.height) as usize)?;
        dv.render(&pix, 4, true, 0, false)?;
        let px = pix.to_f32()?;
        let n = (scene.width * scene.height) as usize;
        let mse = (0..n).flat_map(|p| (0..3).map(move |c| (p, c))).map(|(p, c)| ((px[4 * p + c] - target[3 * p + c]) as f64).powi(2)).sum::<f64>() / (3 * n) as f64;
        log(format!("live   {name:<20} ours PSNR {:.2} dB (prefiltered {:.2}) in {:.2} s on {} | upstream (DiffVG 2x2 samples) PSNR {:.2} dB in {:.0} s on the CPU",
                    10.0 * (1.0 / mse).log10(), out.psnr, out.seconds, gpu.name, r["psnr"].as_f64().unwrap_or(0.0), r["seconds"].as_f64().unwrap_or(0.0)));
        let _ = std::fs::write(d.join("nextcuda.svg"), &out.svg);
    }
    Ok(())
}
