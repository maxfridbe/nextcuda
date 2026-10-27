//! The engine against reference/mossimage/ref.py's dumps (OpenMOSS's transformer and scheduler as they are, on the CPU
//! in float32, the DiT on the same Q8_0 file; transformers' Qwen3.5 on the bf16 weights): each stage fed the
//! reference's own input, so a stage's error is its own.
//!
//! ```text
//!   tokens  the prompt's template, the empty negative's     -> ids.npy, neg_ids.npy (exactly)
//!   te      ids.npy, neg_ids.npy                            -> te_cond [L, 4096], te_neg
//!   dit     te_cond (the reference's)                       -> mod, head (the modulations), txt (the text embedder);
//!           lat_0 at step 0                                 -> x0, x21, x41, vel_c0; te_neg, block 22 left out -> vel_u0
//!   sched   the steps asked                                 -> sigmas
//!   steps   lat_0 through every step, guidance 3            -> lat_final
//!   vae     lat_final                                       -> z (unpacked), rgb [3, H, W]; our picture written as ours.png
//!   full    our text encoder's states, every step, the VAE  -> ours_full.png beside ref.png
//! ```

use std::path::Path;

use nextcuda_image::{Error, Picture, Result};

use crate::{dit, sigmas, timestep, MossImage};

pub fn read_npy(path: &Path) -> Result<(Vec<usize>, Vec<f64>)> {
    let b = std::fs::read(path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
    if b.len() < 10 || &b[..6] != b"\x93NUMPY" {
        return Err(Error(format!("{}: not a .npy", path.display())));
    }
    let (hl, at) = if b[6] == 1 { (u16::from_le_bytes([b[8], b[9]]) as usize, 10) } else { (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12) };
    let head = String::from_utf8_lossy(&b[at..at + hl]).to_string();
    if head.contains("'fortran_order': True") {
        return Err(Error(format!("{}: Fortran order", path.display())));
    }
    let shape: Vec<usize> = head.split("'shape': (").nth(1).and_then(|s| s.split(')').next())
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect()).unwrap_or_default();
    let d = &b[at + hl..];
    let v: Vec<f64> = if head.contains("'<f4'") {
        d.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64).collect()
    } else if head.contains("'<i4'") {
        d.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64).collect()
    } else {
        return Err(Error(format!("{}: {head}", path.display())));
    };
    Ok((shape, v))
}

fn f32s(v: Vec<f64>) -> Vec<f32> {
    v.into_iter().map(|x| x as f32).collect()
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

/// A stage's comparison logged; its cosine back (the worst is the check's result)
fn note(name: &str, got: &[f32], want: &[f32], log: &mut dyn FnMut(String)) -> f64 {
    let (rel, mx, cos) = compare(got, want);
    let mismatch = if got.len() != want.len() { format!("  LENGTHS {} vs {}", got.len(), want.len()) } else { String::new() };
    log(format!("{name:<26} rel {rel:.2e}  max {mx:.3e}  cos {cos:.6}  ({} values){mismatch}", want.len().min(got.len())));
    if got.len() != want.len() { 0.0 } else { cos }
}

/// Runs the stages; the lowest cosine seen
pub fn run(m: &MossImage, dir: &Path, stages: &[&str], log: &mut dyn FnMut(String)) -> Result<f64> {
    let get = |name: &str| -> Result<(Vec<usize>, Vec<f32>)> { read_npy(&dir.join(format!("{name}.npy"))).map(|(s, v)| (s, f32s(v))) };
    let ids_of = |name: &str| -> Result<Vec<u32>> { Ok(read_npy(&dir.join(format!("{name}.npy")))?.1.iter().map(|x| *x as u32).collect()) };
    let mut worst = 1f64;
    let ids = ids_of("ids")?;
    let nids = ids_of("neg_ids")?;
    if stages.contains(&"tokens") {
        let prompt = std::fs::read_to_string(dir.join("prompt.txt")).map_err(|e| Error(format!("prompt.txt: {e}")))?;
        for (what, ours, want) in [("prompt", m.te.ids(&prompt), &ids), ("negative", m.te.ids(""), &nids)] {
            let same = ours == *want;
            log(format!("tokens ({what:<8})        {} ({} vs {} ids{})", if same { "identical" } else { "DIFFER" }, ours.len(), want.len(),
                        ours.iter().zip(want.iter()).position(|(a, b)| a != b).map_or(String::new(), |p| format!(", first at {p}"))));
            if !same {
                worst = 0.0;
            }
        }
    }
    if stages.contains(&"te") {
        let t0 = std::time::Instant::now();
        let c = m.te.encode_ids(&ids)?;
        log(format!("te: {} tokens in {:.2} s", ids.len(), t0.elapsed().as_secs_f64()));
        worst = worst.min(note("te (prompt)", &c, &get("te_cond")?.1, log));
        worst = worst.min(note("te (negative)", &m.te.encode_ids(&nids)?, &get("te_neg")?.1, log));
    }
    let cond = get("te_cond")?.1;
    let neg = get("te_neg")?.1;
    let sig = get("sigmas")?.1;
    let (lshape, lat0) = get("lat_0")?;
    let (h, w) = (lshape[1], lshape[2]);
    let n = h * w;
    if stages.contains(&"sched") {
        let ours = sigmas(sig.len() - 1);
        let same = ours == sig;
        log(format!("sigmas                     {} ({} levels){}", if same { "identical" } else { "DIFFER" }, sig.len(),
                    if same { String::new() } else { format!(": ours {ours:?}, the reference's {sig:?}") }));
        if !same {
            worst = 0.0;
        }
    }
    if stages.contains(&"dit") {
        let (md, hd) = m.dit.modulation(timestep(sig[0]));
        worst = worst.min(note("modulation (shared)", &md, &get("mod")?.1, log));
        worst = worst.min(note("modulation (head)", &hd, &get("head")?.1, log));
        let pre = m.dit.prefix(&m.nsd, &cond)?;
        worst = worst.min(note("text embedder", &pre.x.to_f32()?, &get("txt")?.1, log));
        let npre = m.dit.prefix(&m.nsd, &neg)?;
        let sm = m.dit.step_mod(timestep(sig[0]))?;
        let lat = nextcuda_core::DevBuf::from_f32(&m.nsd.gpu, &dit::tokens(&lat0, n))?;
        let mut g = dit::Grabs { at: vec![0, 21, 41], outs: Vec::new() };
        let t0 = std::time::Instant::now();
        let v = m.dit.velocity(&m.nsd, &pre, &lat, (h, w), &sm, None, Some(&mut g))?.to_f32()?;
        log(format!("a forward of {} tokens in {:.2} s", n + pre.rows, t0.elapsed().as_secs_f64()));
        for (i, out) in g.at.iter().zip(&g.outs) {
            let want = get(&format!("x{i}"))?.1;
            worst = worst.min(note(&format!("block {i} (image rows)"), &out[..n * dit::DIM], &want[..n * dit::DIM], log));
            worst = worst.min(note(&format!("block {i} (caption rows)"), &out[n * dit::DIM..], &want[n * dit::DIM..], log));
        }
        worst = worst.min(note("velocity (cond), step 0", &dit::planes(&v, n), &get("vel_c0")?.1, log));
        let u = m.dit.velocity(&m.nsd, &npre, &lat, (h, w), &sm, Some(crate::UNCOND_SKIP), None)?.to_f32()?;
        worst = worst.min(note("velocity (uncond), step 0", &dit::planes(&u, n), &get("vel_u0")?.1, log));
    }
    let mut ours_lat = None;
    if stages.contains(&"steps") {
        let pre = m.dit.prefix(&m.nsd, &cond)?;
        let npre = m.dit.prefix(&m.nsd, &neg)?;
        let t0 = std::time::Instant::now();
        let lat = m.sample(&pre, Some(&npre), 3.0, &lat0, h, w, &sig, &mut |_| Ok(()))?;
        log(format!("{} steps in {:.1} s", sig.len() - 1, t0.elapsed().as_secs_f64()));
        worst = worst.min(note("latents after the steps", &lat, &get("lat_final")?.1, log));
        ours_lat = Some(lat);
    }
    if stages.contains(&"vae") {
        let (_, lf) = get("lat_final")?;
        worst = worst.min(note("vae input (unpacked)", &m.vae.unpack(&lf, h, w), &get("z")?.1, log));
        let (v, pw, ph) = m.vae.decode_f32(&m.nsd, &lf, h, w)?;
        let planes: Vec<f32> = (0..3).flat_map(|c| (0..pw * ph).map(move |p| (c, p))).map(|(c, p)| v[p * 3 + c].clamp(-1.0, 1.0)).collect();
        worst = worst.min(note("vae (rgb)", &planes, &get("rgb")?.1, log));
        let lat = ours_lat.unwrap_or(lf);
        let (rgb, pw, ph) = m.vae.decode(&m.nsd, &lat, h, w)?;
        Picture { width: pw as u32, height: ph as u32, channels: 3, data: rgb }.write_png(&dir.join("ours.png"), &[]).map_err(Error)?;
        log(format!("ours.png: {pw}x{ph}"));
    }
    if stages.contains(&"full") {
        // the whole pipeline on our side - our text encoder's states too - from the reference's starting noise
        let pre = m.dit.prefix(&m.nsd, &m.te.encode_ids(&ids)?)?;
        let npre = m.dit.prefix(&m.nsd, &m.te.encode_ids(&nids)?)?;
        let t0 = std::time::Instant::now();
        let lat = m.sample(&pre, Some(&npre), 3.0, &lat0, h, w, &sig, &mut |_| Ok(()))?;
        log(format!("full: {} steps in {:.1} s", sig.len() - 1, t0.elapsed().as_secs_f64()));
        worst = worst.min(note("latents (all ours)", &lat, &get("lat_final")?.1, log));
        let (rgb, pw, ph) = m.vae.decode(&m.nsd, &lat, h, w)?;
        let want = get("rgb")?.1;
        let ours: Vec<f32> = (0..3).flat_map(|ch| (0..pw * ph).map(move |p| (ch, p))).map(|(ch, p)| rgb[p * 3 + ch] as f32 / 127.5 - 1.0).collect();
        note("picture (all ours)", &ours, &want, log);
        Picture { width: pw as u32, height: ph as u32, channels: 3, data: rgb }.write_png(&dir.join("ours_full.png"), &[]).map_err(Error)?;
    }
    Ok(worst)
}
