//! The engine against reference/ltx2's dumps (diffusers' and transformers' own classes on the CPU, bfloat16, the same
//! checkpoint): each stage fed the reference's own input, so a stage's error is its own.
//!
//! ```text
//!   tokens  prompt.txt                         -> ids.npy (exactly)
//!   te      ids.npy                            -> hs_all_f16 [T, 49, 3840] (each state's error), hs_<i>
//!   conn    hs_all_f16 (the reference's)       -> video_emb [1024, 4096], audio_emb [1024, 2048]; the empty negative
//!                                                 -> neg_video_emb, neg_audio_emb
//! ```

use std::path::Path;

use nextcuda_core::{Error, Result};

use crate::te::{States, HIDDEN, STATES};
use crate::Ltx2;

/// A .npy as (shape, float32 values): float32, float16 or int32
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
    } else if head.contains("'<f2'") {
        d.chunks_exact(2).map(|c| crate::quant::f16_val(u16::from_le_bytes([c[0], c[1]]))).collect()
    } else if head.contains("'<i4'") {
        d.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32).collect()
    } else {
        return Err(Error(format!("{}: {head}", path.display())));
    };
    if head.contains("'fortran_order': True") && shape.len() > 1 {
        // column-major: the first index fastest - into row-major
        let n = shape.len();
        let mut out = vec![0f32; v.len()];
        let mut idx = vec![0usize; n];
        for (i, val) in v.iter().enumerate() {
            let mut r = i;
            for d in 0..n {
                idx[d] = r % shape[d];
                r /= shape[d];
            }
            let mut c = 0;
            for d in 0..n {
                c = c * shape[d] + idx[d];
            }
            out[c] = *val;
        }
        return Ok((shape, out));
    }
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

pub fn note(name: &str, got: &[f32], want: &[f32], log: &mut dyn FnMut(String)) -> f64 {
    let (rel, mx, cos) = compare(got, want);
    let len = if got.len() == want.len() { String::new() } else { format!("  LENGTHS {} vs {}", got.len(), want.len()) };
    log(format!("{name:<24} rel {rel:.2e}  max {mx:.3e}  cos {cos:.6}  ({} values){len}", want.len().min(got.len())));
    rel
}

/// One state's rows of [T, 49, 3840]
fn state(v: &[f32], t: usize, s: usize) -> Vec<f32> {
    (0..t).flat_map(|r| v[(r * STATES + s) * HIDDEN..(r * STATES + s + 1) * HIDDEN].iter().copied()).collect()
}

pub fn run(m: &Ltx2, dir: &Path, stages: &[&str], log: &mut dyn FnMut(String)) -> Result<f64> {
    let get = |name: &str| read_npy(&dir.join(format!("{name}.npy")));
    let mut worst = 0f64;
    if stages.contains(&"quant") {
        // the host's Q4_K / Q8_0 blocks as the kernels read them
        let gpu = &m.gpus[0];
        let nsd = nextcuda_diffusion::kernels::Nsd::new(gpu)?;
        let x: Vec<f32> = (0..4096).map(|i| ((i as f32 * 0.731).sin() + (i as f32 * 0.0137).cos() * 0.5) * 0.03).collect();
        for (name, ty) in [("q4_k", nextcuda_gguf::GType::Q4K), ("q8_0", nextcuda_gguf::GType::Q8_0)] {
            let mut b = Vec::new();
            match ty {
                nextcuda_gguf::GType::Q4K => crate::quant::q4_k(&x, &mut b),
                _ => crate::quant::q8_0(&x, &mut b),
            }
            let d = nextcuda_core::DevBuf::new(gpu, b.len())?;
            d.write(0, &b)?;
            let o = nextcuda_core::DevBuf::f32(gpu, x.len())?;
            nsd.dequant(d.ptr(), ty.code(), x.len(), o.ptr(), nextcuda_diffusion::kernels::Dt::F32)?;
            nsd.wait()?;
            let g = o.to_f32()?;
            if ty == nextcuda_gguf::GType::Q4K {
                note("quant q4_k host / kernel", &crate::quant::dq4_k(&b), &g, log);
            }
            worst = worst.max(note(&format!("quant {name} vs float"), &g, &x, log).min(0.0));
        }
    }
    let ids: Vec<u32> = if dir.join("ids.npy").exists() { get("ids")?.1.iter().map(|x| *x as u32).collect() } else { Vec::new() };
    let t = ids.len();
    if stages.contains(&"tokens") {
        let prompt = std::fs::read_to_string(dir.join("prompt.txt")).map_err(|e| Error(format!("prompt.txt: {e}")))?;
        let ours = m.tokens(&prompt);
        let same = ours == ids;
        log(format!("tokens                   {} ({} vs {} ids{})", if same { "identical" } else { "DIFFER" }, ours.len(), ids.len(),
                    ours.iter().zip(&ids).position(|(a, b)| a != b).map_or(String::new(), |p| format!(", first at {p}"))));
        if !same {
            worst = worst.max(1.0);
        }
    }
    if stages.contains(&"te") {
        let te = m.text_encoder(log)?;
        let t0 = std::time::Instant::now();
        let s = te.encode(&ids)?;
        log(format!("te: {t} tokens in {:.2} s", t0.elapsed().as_secs_f64()));
        drop(te);
        let (_, want) = get("hs_all_f16")?;
        for i in [0usize, 1, 2, 6, 12, 24, 36, 46, 47, 48] {
            let r = note(&format!("te state {i}"), &state(&s.v, t, i), &state(&want, t, i), log);
            if i > 0 {
                worst = worst.max(r);
            }
        }
        note("te all states", &s.v, &want, log);
        if stages.contains(&"full") {
            // the text stage end to end: our states through our connectors
            let (video, audio) = m.connectors(&s, log)?;
            worst = worst.max(note("te+conn video", &video, &get("video_emb")?.1, log));
            worst = worst.max(note("te+conn audio", &audio, &get("audio_emb")?.1, log));
        }
    }
    if stages.contains(&"conn") {
        let (_, hs) = get("hs_all_f16")?;
        let (video, audio) = m.connectors(&States { tokens: t, v: hs }, log)?;
        worst = worst.max(note("conn video", &video, &get("video_emb")?.1, log));
        worst = worst.max(note("conn audio", &audio, &get("audio_emb")?.1, log));
        let (nv, na) = m.connectors(&States { tokens: 0, v: Vec::new() }, log)?;
        worst = worst.max(note("conn video (negative)", &nv, &get("neg_video_emb")?.1, log));
        worst = worst.max(note("conn audio (negative)", &na, &get("neg_audio_emb")?.1, log));
    }
    if stages.contains(&"dit") {
        // step 0 of the reference's schedule (sigma 1): each pass's velocities and the cond pass's blocks
        let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("meta.json")).map_err(|e| Error(format!("meta.json: {e}")))?)
            .map_err(|e| Error(format!("meta.json: {e}")))?;
        let lt: Vec<usize> = meta["latent"].as_array().ok_or("meta.json: latent")?.iter().filter_map(|v| v.as_u64().map(|x| x as usize)).collect();
        let la = meta["audio_frames"].as_u64().ok_or("meta.json: audio_frames")? as usize;
        let sig0 = meta["sigmas"][0].as_f64().unwrap_or(1.0) as f32;
        let text_dir = std::path::PathBuf::from(spec_text_dir(dir));
        let tget = |n: &str| read_npy(&text_dir.join(format!("{n}.npy"))).map(|x| x.1);
        let (lat, aud) = (get("lat_0")?.1, get("aud_0")?.1);
        m.with_dit(false, &mut |l: String| eprintln!("{l}"), |d| {
            let cond = d.text(&tget("video_emb")?, &tget("audio_emb")?)?;
            let neg = d.text(&tget("neg_video_emb")?, &tget("neg_audio_emb")?)?;
            let prep = d.prepare(crate::dit::Shape { f: lt[0], h: lt[1], w: lt[2], la, fps: 24.0 }, false)?;
            d.set_step(&prep, sig0 * 1000.0)?;
            for (name, text, stg, iso) in [("cond", &cond, false, false), ("uncond", &neg, false, false), ("stg", &cond, true, false), ("mod", &cond, false, true)] {
                let t0 = std::time::Instant::now();
                let caps: &[usize] = if name == "cond" { &[0, 1, 27, 28, 47] } else { &[] };
                let (v, a, c) = d.forward(&prep, &lat, &aud, &crate::dit::Pass { text, stg, isolate: iso }, caps)?;
                log(format!("dit {name} pass in {:.2} s", t0.elapsed().as_secs_f64()));
                for (bi, cv, ca) in &c {
                    note(&format!("dit blk{bi} video"), cv, &get(&format!("s0_blk{bi}_v"))?.1, log);
                    note(&format!("dit blk{bi} audio"), ca, &get(&format!("s0_blk{bi}_a"))?.1, log);
                }
                worst = worst.max(note(&format!("dit {name} video"), &v, &get(&format!("s0_{name}_v"))?.1, log));
                worst = worst.max(note(&format!("dit {name} audio"), &a, &get(&format!("s0_{name}_a"))?.1, log));
            }
            if stages.contains(&"steps") {
                // the whole short schedule from the reference's noise, its latents after each step
                let sig: Vec<f32> = meta["sigmas"].as_array().ok_or("meta.json: sigmas")?.iter().filter_map(|v| v.as_f64().map(|x| x as f32)).collect();
                let (mut l, mut a) = (lat.clone(), aud.clone());
                let mut got: Vec<(usize, Vec<f32>, Vec<f32>)> = Vec::new();
                let mut each = |k: usize, l: &[f32], a: &[f32]| got.push((k, l.to_vec(), a.to_vec()));
                let t0 = std::time::Instant::now();
                crate::sample::denoise(d, &prep, &cond, &neg, &mut l, &mut a, &sig, 0,
                                       &mut crate::sample::Hooks { progress: &mut |_, _| {}, cancel: &|| false, each: Some(&mut each) })?;
                log(format!("steps: {} in {:.1} s", sig.len() - 1, t0.elapsed().as_secs_f64()));
                for (k, gl, ga) in &got {
                    worst = worst.max(note(&format!("step {k} video latents"), gl, &get(&format!("lat_{k}"))?.1, log));
                    worst = worst.max(note(&format!("step {k} audio latents"), ga, &get(&format!("aud_{k}"))?.1, log));
                }
                std::fs::write(dir.join("ours_lat.f32"), crate::te::as_bytes(&l)).ok();
                std::fs::write(dir.join("ours_aud.f32"), crate::te::as_bytes(&a)).ok();
            }
            Ok(())
        })?;
    }
    if stages.contains(&"vae") {
        // the reference's final latents through our decoder, against its frames
        let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("meta.json")).map_err(|e| Error(format!("meta.json: {e}")))?)
            .map_err(|e| Error(format!("meta.json: {e}")))?;
        let lt: Vec<usize> = meta["latent"].as_array().ok_or("meta.json: latent")?.iter().filter_map(|v| v.as_u64().map(|x| x as usize)).collect();
        let k = meta["sigmas"].as_array().map_or(1, |a| a.len() - 1);
        let lat = get(&format!("lat_{k}"))?.1;
        // packed [T, 128] (tokens f, h, w) is already channels-last [F, H, W, 128]
        let t0 = std::time::Instant::now();
        let dec = crate::vae::Decoder::load(m.file("vae")?, &m.gpus[0])?;
        log(format!("vae loaded in {:.1} s", t0.elapsed().as_secs_f64()));
        let t0 = std::time::Instant::now();
        let px = dec.decode(&lat, lt[0], lt[1], lt[2], &mut |_, _| {})?;
        log(format!("vae decode in {:.2} s", t0.elapsed().as_secs_f64()));
        worst = worst.max(note("vae frames", &px, &get("video")?.1, log));
    }
    if stages.contains(&"audio") {
        // the reference's final audio latents through our audio VAE (its mel), its mel through our vocoder (its wave)
        let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("meta.json")).map_err(|e| Error(format!("meta.json: {e}")))?)
            .map_err(|e| Error(format!("meta.json: {e}")))?;
        let la = meta["audio_frames"].as_u64().ok_or("meta.json: audio_frames")? as usize;
        let k = meta["sigmas"].as_array().map_or(1, |a| a.len() - 1);
        let dec = crate::audio::AudioDecoder::load(m.file("audio-vae")?, &m.gpus[0])?;
        let t0 = std::time::Instant::now();
        let mel = dec.decode(&get(&format!("aud_{k}"))?.1, la)?;
        log(format!("audio vae decode in {:.2} s", t0.elapsed().as_secs_f64()));
        let (ms, want_mel) = get("mel")?;
        worst = worst.max(note("audio mel", &mel, &want_mel, log));
        let voc = crate::audio::Audio::load(m.file("vocoder")?, &m.gpus[0])?;
        let t0 = std::time::Instant::now();
        let wave = voc.vocode(&want_mel, ms[1])?;
        log(format!("vocoder in {:.2} s", t0.elapsed().as_secs_f64()));
        let flat: Vec<f32> = wave.concat();
        worst = worst.max(note("audio wave (48 kHz)", &flat, &get("wave")?.1, log));
    }
    Ok(worst)
}

/// The text reference a DiT reference was made from: `text` in its meta.json, else ../t1
fn spec_text_dir(dir: &Path) -> String {
    std::fs::read(dir.join("meta.json")).ok().and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .and_then(|v| v["text"].as_str().map(str::to_string))
        .unwrap_or_else(|| dir.parent().map(|p| p.join("t1").display().to_string()).unwrap_or_default())
}
