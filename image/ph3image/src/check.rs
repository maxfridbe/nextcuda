//! The engine against nextsycl's reference/p-h3-image/ref.py dumps (diffusers 84532a05 on the CPU in float32, Pruna's
//! full bf16 model): each stage fed the reference's own input, so a stage's error is its own.
//!
//! ```text
//!   tokens  prompt.txt (and an edit's pictures)          -> ids, tags (exactly)
//!   vision  vis_img_I (the processor's picture)          -> vis_emb_I, vis_deep_I_J (the tower's merged tokens, deepstack)
//!   te      ids (an edit: our vision tokens)             -> hs_13 .. hs_36, cond; the reference's hs through our connector
//!   venc    ref_img_I                                    -> ref_lat_I (the posterior sampled at seed 42)
//!   dit     cond (the reference's)                       -> txt (the refiner), pos; lat_I / aud_I at every step -> vel_I
//!   steps   noise_v, noise_a through the 8 steps         -> lat_final
//!   vae     lat_final (the reference's)                  -> rgb [3, H, W]; bytes equal / within 1; ours.png
//! ```

use std::path::Path;

use nextcuda_h3::denoiser::{self, Conditions, Denoiser, RefIn, Schedule, Shape, TextRefiner};
use nextcuda_h3::device::Tensor;
use nextcuda_h3::dtype::DType;
use nextcuda_h3::vision::{Frames, Piece};
use nextcuda_image::{Error, Picture, Result};

use crate::connector::TAPS;
use crate::{ie, PH3Image, AUDIO_T, NOISE_AUG};

/// A .npy file: its shape and values (float32, float64 or int32)
pub fn read_npy(path: &Path) -> Result<(Vec<usize>, Vec<f64>)> {
    let b = std::fs::read(path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
    if b.len() < 10 || &b[..6] != b"\x93NUMPY" {
        return Err(Error(format!("{}: not a .npy", path.display())));
    }
    let (hl, at) = if b[6] == 1 { (u16::from_le_bytes([b[8], b[9]]) as usize, 10) } else { (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12) };
    let head = String::from_utf8_lossy(&b[at..at + hl]).to_string();
    let shape: Vec<usize> = head.split("'shape': (").nth(1).and_then(|s| s.split(')').next())
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect()).unwrap_or_default();
    let d = &b[at + hl..];
    let v: Vec<f64> = if head.contains("'<f4'") {
        d.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64).collect()
    } else if head.contains("'<f8'") {
        d.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().expect("8 bytes"))).collect()
    } else if head.contains("'<i4'") {
        d.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64).collect()
    } else if head.contains("'|u1'") {
        d.iter().map(|x| *x as f64).collect()
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

/// A stage's comparison logged; its relative error back (the worst is the check's result)
fn note(name: &str, got: &[f32], want: &[f32], log: &mut dyn FnMut(String)) -> f64 {
    let (rel, mx, cos) = compare(got, want);
    let mismatch = if got.len() != want.len() { format!("  LENGTHS {} vs {}", got.len(), want.len()) } else { String::new() };
    log(format!("{name:<24} rel {rel:.2e}  max {mx:.3e}  cos {cos:.6}  ({} values){mismatch}", want.len().min(got.len())));
    if got.len() != want.len() { f64::INFINITY } else { rel }
}

fn he(e: nextcuda_h3::Error) -> Error {
    ie(e)
}

/// Runs the stages in the dump directory; the worst relative error of the stages that ran
pub fn run(m: &PH3Image, dir: &Path, stages: &[&str], log: &mut dyn FnMut(String)) -> Result<f64> {
    let has = |name: &str| dir.join(format!("{name}.npy")).is_file();
    let get = |name: &str| -> Result<(Vec<usize>, Vec<f32>)> { read_npy(&dir.join(format!("{name}.npy"))).map(|(s, v)| (s, v.into_iter().map(|x| x as f32).collect())) };
    let ints = |name: &str| -> Result<Vec<i64>> { Ok(read_npy(&dir.join(format!("{name}.npy")))?.1.iter().map(|x| *x as i64).collect()) };
    let mut worst = 0f64;
    let prompt = std::fs::read_to_string(dir.join("prompt.txt")).map_err(|e| Error(format!("prompt.txt: {e}")))?;
    // an edit's pictures as the text encoder's processor got them
    let mut vis = Vec::new();
    while has(&format!("vis_img_{}", vis.len())) {
        let (s, v) = read_npy(&dir.join(format!("vis_img_{}.npy", vis.len())))?;
        vis.push(Picture { width: s[1] as u32, height: s[0] as u32, channels: 3, data: v.iter().map(|x| *x as u8).collect() });
    }
    let mut pieces = Vec::new();
    for (i, p) in vis.iter().enumerate() {
        pieces.push(Piece::Text(format!("<Picture {}>: ", i + 1)));
        pieces.push(Piece::Vision(Frames { frames: vec![p.data.iter().map(|b| *b as f32 / 255.0).collect()], w: p.width as usize, h: p.height as usize }));
    }
    pieces.push(Piece::Text(prompt.clone()));
    let ids = ints("ids")?;
    let tags = ints("tags")?;
    let presented = if stages.iter().any(|s| matches!(*s, "tokens" | "vision" | "te")) {
        Some(nextcuda_h3::vision::present_with(&pieces, &m.tok, Some((&m.tower, "model.visual.")), None, m.e.dev.index, log).map_err(he)?)
    } else {
        None
    };
    if let (true, Some(p)) = (stages.contains(&"tokens"), presented.as_ref()) {
        let ours: Vec<i64> = p.ids.iter().map(|x| *x as i64).collect();
        let ot: Vec<i64> = p.tags.iter().map(|x| *x as i64).collect();
        let same = ours == ids && ot == tags;
        log(format!("tokens                   {} ({} of {} ids equal, tags {})", if same { "identical" } else { "DIFFER" },
                    ours.iter().zip(&ids).filter(|(a, b)| a == b).count(), ids.len(), if ot == tags { "equal" } else { "differ" }));
        if !same {
            worst = f64::INFINITY;
        }
    }
    if let (true, Some(p)) = (stages.contains(&"vision"), presented.as_ref()) {
        for (i, s) in p.seen.iter().enumerate() {
            worst = worst.max(note(&format!("vision {i} merged"), &s.tokens, &get(&format!("vis_emb_{i}"))?.1, log));
            for (j, d) in s.deep.iter().enumerate() {
                worst = worst.max(note(&format!("vision {i} deepstack {j}"), d, &get(&format!("vis_deep_{i}_{j}"))?.1, log));
            }
        }
    }
    if let (true, Some(p)) = (stages.contains(&"te"), presented.as_ref()) {
        let t0 = std::time::Instant::now();
        let taps = m.te.encode_resident_taps(&m.resident, &p.ids, &p.seen, &p.embeds, &TAPS, Some(&m.final_norm), &mut |_| Ok(())).map_err(he)?;
        log(format!("te: {} tokens in {:.2} s", p.ids.len(), t0.elapsed().as_secs_f64()));
        for (k, t) in TAPS.iter().zip(&taps) {
            worst = worst.max(note(&format!("hidden state {k}"), t, &get(&format!("hs_{k}"))?.1, log));
        }
        worst = worst.max(note("connector output", &m.connector.forward(&taps).map_err(he)?, &get("cond")?.1, log));
        let refs: Vec<Vec<f32>> = TAPS.iter().map(|k| get(&format!("hs_{k}")).map(|x| x.1)).collect::<Result<_>>()?;
        worst = worst.max(note("connector (ref. states)", &m.connector.forward(&refs).map_err(he)?, &get("cond")?.1, log));
    }
    // the edit's conditioning latents
    let mut ref_lats = Vec::new();
    while has(&format!("ref_lat_{}", ref_lats.len())) {
        ref_lats.push(get(&format!("ref_lat_{}", ref_lats.len()))?);
    }
    if stages.contains(&"venc") && !ref_lats.is_empty() {
        let venc = nextcuda_h3::venc::VideoEncoder::load(&m.te_dev, &nextcuda_h3::safetensors::Checkpoint::open(&m.vae_path).map_err(he)?).map_err(he)?;
        for (i, (_, want)) in ref_lats.iter().enumerate() {
            let (s, img) = read_npy(&dir.join(format!("ref_img_{i}.npy")))?;
            let px: Vec<f32> = img.iter().map(|x| *x as f32 / 255.0).collect();
            let z = venc.encode_sampled(&px, s[0], s[1], crate::ENCODE_SEED, &mut || Ok(())).map_err(he)?;
            worst = worst.max(note(&format!("vae encoder {i}"), &z, want, log));
        }
    }
    if stages.contains(&"dit") || stages.contains(&"steps") {
        worst = worst.max(dit_stages(&m.e, m.connector.out, dir, stages, log)?);
    }
    if !stages.contains(&"vae") {
        return Ok(worst);
    }
    let (fshape, lat_final) = get("lat_final")?;
    let (h, w) = (fshape[1], fshape[2]);
    if stages.contains(&"vae") {
        let (px, _) = m.vae.decode(&lat_final, 1, h, w, &mut || Ok(())).map_err(he)?;
        let want = get("rgb")?.1;
        worst = worst.max(note("vae decoder", &px, &want, log));
        let b = |v: f32| (v * 255.0).round().clamp(0.0, 255.0) as i32;
        let (eq, near) = px.iter().zip(&want).fold((0usize, 0usize), |(e, n), (g, w)| {
            let d = (b(*g) - b(*w)).abs();
            (e + (d == 0) as usize, n + (d <= 1) as usize)
        });
        log(format!("vae decoder bytes        {:.1}% equal, {:.2}% within 1", eq as f64 * 100.0 / want.len() as f64, near as f64 * 100.0 / want.len() as f64));
        let (hh, ww) = (h * 16, w * 16);
        let plane = hh * ww;
        let data = (0..plane * 3).map(|i| b(px[(i % 3) * plane + i / 3]) as u8).collect();
        Picture { width: ww as u32, height: hh as u32, channels: 3, data }.write_png(&dir.join("ours.png"), &[]).map_err(Error)?;
    }
    Ok(worst)
}

/// The denoiser's stages alone: the refiner (or, with stage `reftext`, the reference's refined text) and the blocks at
/// every step from the reference's rows (`dit`), the 8 steps from its noise (`steps`)
pub fn dit_stages(e: &nextcuda_h3::jobs::Engine, out: usize, dir: &Path, stages: &[&str], log: &mut dyn FnMut(String)) -> Result<f64> {
    let has = |name: &str| dir.join(format!("{name}.npy")).is_file();
    let get = |name: &str| -> Result<(Vec<usize>, Vec<f32>)> { read_npy(&dir.join(format!("{name}.npy"))).map(|(s, v)| (s, v.into_iter().map(|x| x as f32).collect())) };
    let tags: Vec<i64> = read_npy(&dir.join("tags.npy"))?.1.iter().map(|x| *x as i64).collect();
    let mut worst = 0f64;
    let mut ref_lats = Vec::new();
    while has(&format!("ref_lat_{}", ref_lats.len())) {
        ref_lats.push(get(&format!("ref_lat_{}", ref_lats.len()))?);
    }
    let (fshape, lat_final) = get("lat_final")?;
    let (h, w) = (fshape[1], fshape[2]);
    let cond = get("cond")?.1;
    let model = e.blocks().map_err(he)?;
    let l = cond.len() / out;
    let ref_text = stages.contains(&"reftext");
    let text = |_: ()| -> Result<Tensor> {
        if ref_text {
            let t = get("txt")?.1;
            return Tensor::from_bytes(&e.dev, DType::BF16, &[t.len() / e.cfg.hidden, e.cfg.hidden], &denoiser::bf16_bytes(&t)).map_err(he);
        }
        let refiner = TextRefiner::load(&e.dev, &e.ck, e.threads).map_err(he)?;
        let t = Tensor::from_bytes(&e.dev, DType::BF16, &[l, out], &denoiser::bf16_bytes(&cond)).map_err(he)?;
        let r = refiner.run(&e.cfg, &t, None).map_err(he)?;
        e.dev.wait().map_err(he)?;
        Ok(r)
    };
    let sig = get("sigmas")?.1;
    let shape = Shape { t: 1, h, w, audio_t: AUDIO_T };
    let schedule = Schedule { shift_video: 1.0, shift_audio: 1.0, audio_scale: 1.0 };
    let vt = tags.iter().any(|t| *t == 0).then(|| tags.iter().map(|t| *t as i32).collect::<Vec<i32>>());
    // the conditioning rows: the leading rows of every step's video rows (the reference's, noised once)
    let crow: usize = ref_lats.iter().map(|(s, _)| (s[1] / 2) * (s[2] / 2)).sum();
    let audio_c = e.outer.audio_c;
    let audio_in = |rows: &[f32]| -> Vec<f32> {
        let mut a = vec![0f32; rows.len()];
        for r in 0..2 * AUDIO_T {
            for ch in 0..audio_c {
                a[ch * 2 * AUDIO_T + r] = rows[r * audio_c + ch];
            }
        }
        a
    };
    let make = |lat0: &[f32]| -> Result<Denoiser> {
        let mut cnd = Conditions { visual_aug: NOISE_AUG, mixed: true, ..Conditions::default() };
        let mut at = 0;
        for (s, _) in &ref_lats {
            let (rh, rw) = (s[1], s[2]);
            let n = (rh / 2) * (rw / 2) * 96;
            cnd.refs.push(RefIn::Image { latent: denoiser::unpatchify(&lat0[at..at + n], 24, 1, rh, rw), h: rh, w: rw });
            at += n;
        }
        Denoiser::new(&model, &e.outer, text(())?, vt.clone(), shape, schedule, &cnd).map_err(he)
    };
    if stages.contains(&"dit") {
        let lat0 = get("lat_0")?.1;
        let mut d = make(&lat0)?;
        let mut pos: Vec<f32> = d.layout.positions.iter().map(|v| *v as f32).collect();
        let want_pos = get("pos")?.1;
        pos.truncate(want_pos.len());
        worst = worst.max(note("positions", &pos, &want_pos, log));
        if has("txt") && !ref_text {
            worst = worst.max(note("refined text", &text(())?.to_f32().map_err(he)?, &get("txt")?.1, log));
        }
        for i in 0..sig.len() - 1 {
            if !has(&format!("lat_{i}")) {
                break;
            }
            let lat = get(&format!("lat_{i}"))?.1;
            let aud = get(&format!("aud_{i}"))?.1;
            let v = denoiser::unpatchify(&lat[crow * 96..], 24, 1, h, w);
            let (ov, _) = d.net(&v, &audio_in(&aud), sig[i], &mut |_| Ok(())).map_err(he)?;
            // ours is the negated head (H3's sign); the reference's the head
            let got: Vec<f32> = denoiser::patchify(&ov, 24, 1, h, w).iter().map(|x| -x).collect();
            let want = get(&format!("vel_{i}"))?.1;
            worst = worst.max(note(&format!("velocity, step {i}"), &got, &want[crow * 96..], log));
        }
    }
    if stages.contains(&"steps") {
        let nv = get("noise_v")?.1;
        let na = get("noise_a")?.1;
        let mut d = make(&get("lat_0")?.1)?;
        let noise_v = denoiser::unpatchify(&nv[crow * 96..], 24, 1, h, w);
        let (v, _) = denoiser::sample_masked(&mut d, &noise_v, &audio_in(&na), &sig, None, &mut |_, _, _| Ok(()), &mut |_, _| Ok(())).map_err(he)?;
        worst = worst.max(note("final latents (8 steps)", &v, &lat_final, log));
    }
    Ok(worst)
}

/// The denoiser alone over `gpus` (no text encoder, no decoder: room for a larger file than the engine's, as a check
/// of how much of the error is the quantization), `reserve` bytes kept on each card
pub fn run_denoiser(denoiser: &Path, gpus: &[usize], reserve: u64, dir: &Path, stages: &[&str], log: &mut dyn FnMut(String)) -> Result<f64> {
    nextcuda_core::use_kind("image");
    let devs = gpus.iter().map(|i| nextcuda_h3::device::Device::open_index(*i)).collect::<nextcuda_h3::Result<Vec<_>>>().map_err(he)?;
    let mut e = nextcuda_h3::jobs::Engine::load_cards(devs, denoiser, None, 8, log).map_err(he)?;
    e.reserve = reserve;
    let t0 = std::time::Instant::now();
    let b = e.blocks().map_err(he)?;
    log(format!("blocks : {} ({:.2} GiB) in {:.1} s", b.blocks.len(), b.load_bytes as f64 / (1u64 << 30) as f64, t0.elapsed().as_secs_f64()));
    let out = e.ck.get("condition_proj.weight").map_err(he)?.shape[1];
    dit_stages(&e, out, dir, stages, log)
}
