//! IS-Net against reference/rembg/ref.py's dumps (rembg 2.0.67 on onnxruntime's CPU, float32): each picture's stages,
//! each fed the reference's own input where it has one, so a stage's error is its own.
//!
//! ```text
//!   input  the picture -> rembg's normalization (Lanczos to 1024, / max, - 0.5)  against input.npy
//!   net    input.npy -> the prediction (the sigmoid)                            against pred.npy
//!   mask   the whole way, the picture -> the 8-bit mask at its size                 against mask.png (equal bytes)
//!   pp     the mask -> post_process_mask                                            against mask_pp.png
//! ```

use std::path::Path;
use std::time::Instant;

use nextcuda_image::{Error, Picture, Result};

use crate::{isnet::IsNet, post_process_mask, IsNetEngine};

fn npy(path: &Path) -> Result<Vec<f32>> {
    let b = std::fs::read(path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
    let (hl, at) = if b[6] == 1 { (u16::from_le_bytes([b[8], b[9]]) as usize, 10) } else { (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12) };
    if !String::from_utf8_lossy(&b[at..at + hl]).contains("'<f4'") {
        return Err(Error(format!("{}: not float32", path.display())));
    }
    Ok(b[at + hl..].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

fn grey(path: &Path) -> Result<Picture> {
    let p = Picture::decode(&std::fs::read(path).map_err(|e| Error(format!("{}: {e}", path.display())))?).map_err(Error)?;
    // a grey PNG decodes as RGB: its first channel
    let c = p.channels as usize;
    Ok(Picture { width: p.width, height: p.height, channels: 1, data: p.data.chunks_exact(c).map(|v| v[0]).collect() })
}

/// (max |difference|, relative L2 error)
fn diff(a: &[f32], b: &[f32]) -> (f64, f64) {
    let (mut mx, mut e2, mut w2) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let d = (*x - *y) as f64;
        mx = mx.max(d.abs());
        e2 += d * d;
        w2 += (*y as f64) * (*y as f64);
    }
    (mx, (e2 / w2.max(1e-30)).sqrt())
}

/// (share of equal bytes, the largest difference)
fn bytes(a: &[u8], b: &[u8]) -> (f64, i32) {
    let eq = a.iter().zip(b).filter(|(x, y)| x == y).count();
    (eq as f64 / b.len().max(1) as f64, a.iter().zip(b).map(|(x, y)| (*x as i32 - *y as i32).abs()).max().unwrap_or(0))
}

/// Every picture folder in `dir`: the stages logged; the worst relative error of the prediction
pub fn run(net: &IsNet, dir: &Path, log: &mut dyn FnMut(String)) -> Result<f64> {
    let mut worst = 0f64;
    let mut subs: Vec<_> = std::fs::read_dir(dir).map_err(|e| Error(format!("{}: {e}", dir.display())))?.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
    subs.sort();
    for d in subs {
        let name = d.file_name().unwrap().to_string_lossy().into_owned();
        let pic = Picture::read_png(&d.join("picture.png")).map_err(Error)?;
        let (mi, ri) = diff(&IsNetEngine::input(&pic), &npy(&d.join("input.npy"))?);
        let t0 = Instant::now();
        let pred = net.forward(&npy(&d.join("input.npy"))?)?;
        let secs = t0.elapsed().as_secs_f64();
        let (mp, rp) = diff(&pred, &npy(&d.join("pred.npy"))?);
        worst = worst.max(rp);
        let t1 = Instant::now();
        let mine = IsNetEngine::mask_of(&net.forward(&IsNetEngine::input(&pic))?, pic.width, pic.height);
        let whole = t1.elapsed().as_secs_f64();
        let (eqm, mxm) = bytes(&mine.data, &grey(&d.join("mask.png"))?.data);
        let (eqp, mxp) = bytes(&post_process_mask(&mine).data, &grey(&d.join("mask_pp.png"))?.data);
        log(format!("{name:<10} {}x{}  input max {mi:.2e} rel {ri:.1e} | net max {mp:.2e} rel {rp:.1e} ({secs:.3} s) | mask {:.3}% equal, max {mxm} | post-processed {:.3}% equal | whole {whole:.3} s",
                    pic.width, pic.height, eqm * 100.0, eqp * 100.0));
        let _ = mxp;
    }
    Ok(worst)
}
