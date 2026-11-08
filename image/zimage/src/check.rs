//! The engine against nextsycl's reference/z-image-turbo/ref.py dumps (diffusers 84532a05's classes on the CPU in
//! float32, on the engine's own files: the Q8_0 DiT and Qwen3-4B GGUFs, each matrix expanded for its own product): each
//! stage fed the reference's own input, so a stage's error is its own.
//!
//! ```text
//!   noise   seed.txt                                  -> lat_0 (torch.randn on a CPU generator)
//!   tokens  prompt.txt                                -> ids (exactly)
//!   te      ids                                       -> te_h1, te_h18, te_h35 (the caption)
//!   dit     te_h35 (the reference's)                  -> cr_out; lat_i at every step -> vel_i (step 0: nr_out, l0_out)
//!   steps   lat_0 through every step                  -> lat_final
//!   vae     lat_final (the reference's)               -> rgb [3, H, W]; bytes as ref.png's
//!   full    the prompt and seed alone, all ours       -> rgb: the picture a request makes against the reference's
//! ```

use std::path::Path;

use nextcuda_core::DevBuf;
use nextcuda_image::{Error, Picture, Result};
use nextcuda_zimage::dit;

use crate::{he, ZImage};

/// A .npy file: its shape and values (float32 or int32)
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
    let v: Vec<f32> = if head.contains("'<f4'") {
        d.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    } else if head.contains("'<i4'") {
        d.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32).collect()
    } else {
        return Err(Error(format!("{}: {head}", path.display())));
    };
    Ok((shape, v))
}

/// (relative L2 error, max |difference|, cosine)
pub fn compare(got: &[f32], want: &[f32]) -> (f64, f64, f64) {
    let (mut e2, mut w2, mut g2, mut dot, mut mx) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (g, w) in got.iter().zip(want) {
        let (g, w) = (*g as f64, *w as f64);
        e2 += (g - w) * (g - w);
        w2 += w * w;
        g2 += g * g;
        dot += g * w;
        mx = mx.max((g - w).abs());
    }
    ((e2 / w2.max(1e-30)).sqrt(), mx, dot / (w2.sqrt() * g2.sqrt()).max(1e-30))
}

/// A stage's comparison logged; its relative error back
fn note(name: &str, got: &[f32], want: &[f32], log: &mut dyn FnMut(String)) -> f64 {
    let (rel, mx, cos) = compare(got, want);
    let mismatch = if got.len() != want.len() { format!("  LENGTHS {} vs {}", got.len(), want.len()) } else { String::new() };
    log(format!("{name:<26} rel {rel:.2e}  max {mx:.3e}  cos {cos:.6}  ({} values){mismatch}", want.len().min(got.len())));
    if got.len() != want.len() { f64::INFINITY } else { rel }
}

/// Picture bytes [H, W, 3] against the reference's rgb [3, H, W] made bytes as ref.png is (clamped, (x + 1) 127.5,
/// rounded): the share equal and within 1, PSNR
fn bytes_vs(name: &str, ours: &[u8], rgb: &[f32], pw: usize, ph: usize, log: &mut dyn FnMut(String)) {
    let n = pw * ph;
    let (mut eq, mut near, mut se) = (0usize, 0usize, 0f64);
    for p in 0..n {
        for c in 0..3 {
            let r = ((rgb[c * n + p].clamp(-1.0, 1.0) + 1.0) * 127.5).round() as i32;
            let d = ours[p * 3 + c] as i32 - r;
            eq += (d == 0) as usize;
            near += (d.abs() <= 1) as usize;
            se += (d * d) as f64;
        }
    }
    let psnr = 10.0 * (255.0f64 * 255.0 / (se / (3 * n) as f64).max(1e-12)).log10();
    log(format!("{name:<26} {:.1} % of bytes equal, {:.1} % within 1, PSNR {psnr:.1} dB", 100.0 * eq as f64 / (3 * n) as f64, 100.0 * near as f64 / (3 * n) as f64));
}

/// [H, W, 3] -> [3, H, W]
fn planes(v: &[f32], pw: usize, ph: usize, f: impl Fn(f32) -> f32) -> Vec<f32> {
    (0..3).flat_map(|c| (0..pw * ph).map(move |p| (c, p))).map(|(c, p)| f(v[p * 3 + c])).collect()
}

/// Runs the stages in the dump directory; the worst relative error of the stages that ran
pub fn run(m: &ZImage, dir: &Path, stages: &[&str], log: &mut dyn FnMut(String)) -> Result<f64> {
    let get = |name: &str| read_npy(&dir.join(format!("{name}.npy")));
    let mut worst = 0f64;
    let prompt = std::fs::read_to_string(dir.join("prompt.txt")).map_err(|e| Error(format!("prompt.txt: {e}")))?;
    let seed: u64 = std::fs::read_to_string(dir.join("seed.txt")).map_err(|e| Error(format!("seed.txt: {e}")))?.trim().parse()
        .map_err(|e| Error(format!("seed.txt: {e}")))?;
    let ids: Vec<u32> = get("ids")?.1.iter().map(|x| *x as u32).collect();
    let sig = get("sigmas")?.1;
    let (lshape, lat0) = get("lat_0")?;
    let (lh, lw) = (lshape[1], lshape[2]);
    let hw = (lh / 2, lw / 2);
    let n = hw.0 * hw.1;
    let n_pad = n.div_ceil(32) * 32;
    let steps = sig.len() - 1;
    let ours_sig = crate::sigmas(steps, crate::SHIFT);
    worst = worst.max(note("sigmas", &ours_sig, &sig, log));
    if stages.contains(&"noise") {
        let ours = nextcuda_diffusion::torch_rng::Mt19937::new(seed).randn(dit::LATENT * lh * lw);
        worst = worst.max(note(&format!("noise (seed {seed})"), &ours, &lat0, log));
    }
    if stages.contains(&"tokens") {
        let ours = m.tokens(&prompt)?;
        let same = ours == ids;
        log(format!("tokens                     {} ({} vs {} ids{})", if same { "identical" } else { "DIFFER" }, ours.len(), ids.len(),
                    ours.iter().zip(&ids).position(|(a, b)| a != b).map_or(String::new(), |p| format!(", first at {p}"))));
        if !same {
            worst = f64::INFINITY;
        }
    }
    if stages.contains(&"te") {
        let taps = [1usize, 18, crate::TE_TAP];
        let t0 = std::time::Instant::now();
        let got = m.te.encode_resident_taps(&m.resident, &ids, &[], &[], &taps, None, &mut |_| Ok(())).map_err(he)?;
        log(format!("te: {} tokens in {:.2} s", ids.len(), t0.elapsed().as_secs_f64()));
        for (k, t) in taps.iter().zip(&got) {
            worst = worst.max(note(&format!("te hidden state {k}"), t, &get(&format!("te_h{k}"))?.1, log));
        }
    }
    let cap = get("te_h35")?.1;
    if stages.contains(&"dit") {
        let mut g = dit::Grabs::default();
        let pre = m.dit.prefix(&m.nsd, &cap, &[], 0, Some(&mut g))?;
        let c = pre.rows;
        worst = worst.max(note("context refiner", &g.cr_out, &get("cr_out")?.1, log));
        let mods = m.dit.mods(&m.nsd, &sig)?;
        for (i, md) in mods.iter().enumerate() {
            let (_, li) = get(&format!("lat_{i}"))?;
            let lat = DevBuf::from_f32(&m.gpu, &dit::patchify(&li, lh, lw))?;
            let mut gi = dit::Grabs::default();
            let v = m.dit.velocity(&m.nsd, &pre, &lat, hw, md, if i == 0 { Some(&mut gi) } else { None })?.to_f32()?;
            if i == 0 {
                worst = worst.max(note("noise refiner", &gi.nr_out, &get("nr_out")?.1, log));
                let want = get("l0_out")?.1;
                worst = worst.max(note("layer 0 (image rows)", &gi.l0_out[..n * dit::DIM], &want[..n * dit::DIM], log));
                worst = worst.max(note("layer 0 (caption rows)", &gi.l0_out[n_pad * dit::DIM..(n_pad + c) * dit::DIM], &want[n_pad * dit::DIM..(n_pad + c) * dit::DIM], log));
            }
            let want = dit::patchify(&get(&format!("vel_{i}"))?.1, lh, lw);
            worst = worst.max(note(&format!("velocity, step {i}"), &v, &want, log));
        }
    }
    if stages.contains(&"steps") {
        let pre = m.prefix(&cap)?;
        let t0 = std::time::Instant::now();
        let lat = nextcuda_zimage::sample(&m.dit, &m.nsd, &m.gpu, &pre, None, 1.0, &lat0, lh, lw, &sig, &mut |_| Ok(()))?;
        log(format!("{steps} steps in {:.1} s", t0.elapsed().as_secs_f64()));
        worst = worst.max(note("latents after the steps", &lat, &get("lat_final")?.1, log));
    }
    let rgb = if stages.contains(&"vae") || stages.contains(&"full") { Some(get("rgb")?.1) } else { None };
    if let (true, Some(rgb)) = (stages.contains(&"vae"), rgb.as_ref()) {
        let (lf_shape, lf) = get("lat_final")?;
        let (v, pw, ph) = m.vae.decode_f32(&m.nsd, &lf, lf_shape[1], lf_shape[2])?;
        worst = worst.max(note("vae (rgb)", &planes(&v, pw, ph, |x| x), rgb, log));
        let (b, _, _) = m.decode(&lf, lh, lw)?;
        bytes_vs("vae bytes vs ref.png", &b, rgb, pw, ph, log);
    }
    if let (true, Some(rgb)) = (stages.contains(&"full"), rgb.as_ref()) {
        // what a request makes: our tokens, text encoder, noise from the seed, steps and decoder
        let t0 = std::time::Instant::now();
        let c = m.condition(&prompt)?;
        let pre = m.prefix(&c)?;
        let lat = m.latents(&pre, seed, lh, lw, &ours_sig, &mut |_| Ok(()))?;
        log(format!("full: {steps} steps in {:.1} s", t0.elapsed().as_secs_f64()));
        worst = worst.max(note("latents (all ours)", &lat, &get("lat_final")?.1, log));
        let (b, pw, ph) = m.decode(&lat, lh, lw)?;
        let pic = Picture { width: pw as u32, height: ph as u32, channels: 3, data: b };
        pic.write_png(&dir.join("ours_full.png"), &[]).map_err(Error)?;
        let clamped: Vec<f32> = rgb.iter().map(|x| ((x.clamp(-1.0, 1.0) + 1.0) * 127.5).round() / 127.5 - 1.0).collect();
        let ours: Vec<f32> = (0..3).flat_map(|ch| (0..pw * ph).map(move |p| (ch, p))).map(|(ch, p)| pic.data[p * 3 + ch] as f32 / 127.5 - 1.0).collect();
        note("picture (all ours) vs ref", &ours, &clamped, log);
        bytes_vs("picture bytes vs ref.png", &pic.data, rgb, pw, ph, log);
        log(format!("wrote {}", dir.join("ours_full.png").display()));
    }
    Ok(worst)
}
