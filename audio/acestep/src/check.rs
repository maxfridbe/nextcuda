//! Stage by stage against the reference (reference/acestep/ref.py: ACE-Step's own pipeline, float32 on the CPU, as the
//! platform's music worker calls it), each from the reference's own input.
//!
//! ```text
//!   text      request.json                         -> text.json's ids (the caption prompt's, the lyrics')
//!   te        text.json's caption ids              -> te_out.npy [L, 1024]; lyric_emb.npy (the lyrics' embeddings)
//!   enc       te_out, lyric_emb, refer.npy         -> enc.npy [lyrics + 1 + caption, 2048] (the packed conditions)
//!   velocity  step_<i>_x, ctx.npy, steps.json, enc -> step_<i>_v.npy (every step, each from the reference's x)
//!   sample    noise.npy, enc.npy                   -> latents.npy (the whole flow from the reference's noise)
//!   vae       vae_in.npy                           -> wav_raw.npy [2, S] (the first window)
//!   vaefull   latents.npy                          -> audio.npy (every window, the peak)
//!   song      request.json, noise.npy              -> audio.npy (all of it: the texts, the flow, the decode, the peak)
//! ```

use std::path::Path;
use std::time::Instant;

use nextcuda_audio::{AudioRequest, Error, Result};
use serde_json::Value;

use crate::nn::Scratch;
use crate::sample::LATENT;
use crate::{AceStep, Trace};

/// A little-endian .npy of float32, int32 or int64 (C order): (shape, values as f64)
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
    } else if head.contains("'<i8'") {
        d.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().expect("8 bytes")) as f64).collect()
    } else if head.contains("'|b1'") {
        d.iter().map(|x| *x as f64).collect()
    } else {
        return Err(Error(format!("{}: {head}", path.display())));
    };
    Ok((shape, v))
}

fn f32s(dir: &Path, name: &str) -> Result<Vec<f32>> {
    Ok(read_npy(&dir.join(name))?.1.into_iter().map(|x| x as f32).collect())
}

/// (relative error |got - want| / |want|, cosine, max |got - want|)
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
    ((e2 / w2.max(1e-30)).sqrt(), dot / (w2.sqrt() * g2.sqrt()).max(1e-30), mx)
}

fn json(dir: &Path, name: &str) -> Result<Value> {
    std::fs::read(dir.join(name)).ok().and_then(|b| serde_json::from_slice(&b).ok()).ok_or_else(|| Error(format!("{}: no {name}", dir.display())))
}

/// The reference's request as this engine's
pub fn request(dir: &Path) -> Result<AudioRequest> {
    let r = json(dir, "request.json")?;
    let a = &r["args"];
    let lyrics = match a["lyrics_file"].as_str().filter(|f| !f.is_empty()) {
        Some(_) => r["params"]["lyrics"].as_str().map(str::to_string),
        None => a["lyrics"].as_str().filter(|l| !l.is_empty()).map(str::to_string),
    };
    let mut extra = nextcuda_audio::Given::new();
    if let Some(b) = a["bpm"].as_u64().filter(|b| *b > 0) {
        extra.insert("bpm".into(), b.to_string());
    }
    for (k, n) in [("key", "key"), ("timesig", "time-signature"), ("language", "language")] {
        if let Some(v) = a[k].as_str().filter(|v| !v.is_empty() && *v != "unknown") {
            extra.insert(n.into(), v.into());
        }
    }
    extra.insert("shift".into(), a["shift"].as_f64().unwrap_or(3.0).to_string());
    Ok(AudioRequest { prompt: a["caption"].as_str().unwrap_or("").into(), lyrics, seconds: a["duration"].as_f64().map(|d| d as f32),
                      steps: a["steps"].as_u64().map(|s| s as u32), seed: a["seed"].as_u64().unwrap_or(0), extra, ..Default::default() })
}

fn line(log: &mut dyn FnMut(String), what: &str, got: &[f32], want: &[f32], worst: &mut f64) {
    let (rel, cos, mx) = compare(got, want);
    *worst = worst.max(rel);
    log(format!("{what:<24} relative error {rel:.2e}  cos {cos:.6}  max |d| {mx:.3e}  ({} values{})", want.len(),
                if got.len() != want.len() { format!(", got {}", got.len()) } else { String::new() }));
}

/// Runs the stages named against `dir`'s dumps; the worst relative error
pub fn run(e: &AceStep, dir: &Path, stages: &[&str], log: &mut dyn FnMut(String)) -> Result<f64> {
    let mut worst = 0f64;
    let mut s = Scratch::default();
    let req = request(dir)?;
    let song = e.song(&req)?;
    let texts = json(dir, "text.json")?;
    let ids = |i: usize| -> Vec<u32> { texts[i]["ids"].as_array().map(|a| a.iter().filter_map(|x| x.as_u64().map(|y| y as u32)).collect()).unwrap_or_default() };
    let want_cap = ids(0);
    let want_lyr = ids(1);
    let all = stages.contains(&"all");
    let on = |n: &str| all || stages.contains(&n);
    if on("text") {
        let mut t = Trace::default();
        let _ = e.conditions(&mut s, &song, &mut t)?;
        let same = |a: &[u32], b: &[u32]| a == b;
        log(format!("text: caption ids {} ({} tokens{}), lyric ids {} ({} tokens{})", if same(&t.caption_ids, &want_cap) { "identical" } else { "DIFFER" },
                    want_cap.len(), if same(&t.caption_ids, &want_cap) { String::new() } else { format!(", got {}", t.caption_ids.len()) },
                    if same(&t.lyric_ids, &want_lyr) { "identical" } else { "DIFFER" }, want_lyr.len(),
                    if same(&t.lyric_ids, &want_lyr) { String::new() } else { format!(", got {}", t.lyric_ids.len()) }));
        if t.caption_ids != want_cap {
            let n = t.caption_ids.iter().zip(&want_cap).take_while(|(a, b)| a == b).count();
            log(format!("  caption ids part at {n}: ours {:?} theirs {:?}", &t.caption_ids[n..(n + 8).min(t.caption_ids.len())],
                        &want_cap[n..(n + 8).min(want_cap.len())]));
        }
        if t.lyric_ids != want_lyr {
            let n = t.lyric_ids.iter().zip(&want_lyr).take_while(|(a, b)| a == b).count();
            log(format!("  lyric ids part at {n}: ours {:?} theirs {:?}", &t.lyric_ids[n..(n + 8).min(t.lyric_ids.len())], &want_lyr[n..(n + 8).min(want_lyr.len())]));
        }
    }
    if on("te") {
        let t0 = Instant::now();
        let got = e.te.encode(&e.ops, &mut s, &want_cap)?;
        line(log, &format!("te ({:.2} s)", t0.elapsed().as_secs_f64()), &got, &f32s(dir, "te_out.npy")?, &mut worst);
        line(log, "lyric embeddings", &e.te.embed(&want_lyr)?, &f32s(dir, "lyric_emb.npy")?, &mut worst);
    }
    let enc_ref = f32s(dir, "enc.npy")?;
    if on("enc") {
        let got = e.dit.conditions(&e.ops, &mut s, &f32s(dir, "te_out.npy")?, &f32s(dir, "lyric_emb.npy")?, &f32s(dir, "refer.npy")?)?;
        line(log, "enc (conditions)", &got, &enc_ref, &mut worst);
    }
    let steps = json(dir, "steps.json")?;
    if on("velocity") {
        let cond = e.dit.cond(&e.ops, &mut s, &enc_ref)?;
        let ctx = f32s(dir, "ctx.npy")?;
        for (i, st) in steps.as_array().into_iter().flatten().enumerate() {
            let x = dir.join(format!("step_{i}_x.npy"));
            if !x.is_file() {
                break;
            }
            let t0 = Instant::now();
            let got = e.dit.velocity(&e.ops, &mut s, &f32s(dir, &format!("step_{i}_x.npy"))?, &ctx, st["t"].as_f64().unwrap_or(0.0) as f32, &cond)?;
            line(log, &format!("velocity {i} (t {:.3}, {:.2} s)", st["t"].as_f64().unwrap_or(0.0), t0.elapsed().as_secs_f64()), &got,
                 &f32s(dir, &format!("step_{i}_v.npy"))?, &mut worst);
        }
    }
    if on("sample") {
        let t0 = Instant::now();
        let mut t = Trace::default();
        let lat = e.sample(&mut s, &song, &enc_ref, f32s(dir, "noise.npy")?, &mut t, &mut |_, _| Ok(()))?;
        line(log, &format!("sample ({:.2} s)", t0.elapsed().as_secs_f64()), &lat, &f32s(dir, "latents.npy")?, &mut worst);
    }
    if on("vae") {
        let (shape, z) = read_npy(&dir.join("vae_in.npy"))?;
        let (c, n) = (shape[0], shape[1]);
        let mut zf = vec![0f32; c * n];
        for ch in 0..c {
            for f in 0..n {
                zf[f * LATENT + ch] = z[ch * n + f] as f32;
            }
        }
        let t0 = Instant::now();
        let wav = e.vae.decode(&e.ops, &mut s, &zf, &mut |_, _| Ok(()))?;
        line(log, &format!("vae ({:.2} s)", t0.elapsed().as_secs_f64()), &wav, &f32s(dir, "wav_raw.npy")?, &mut worst);
    }
    if on("vaefull") {
        // the reference's latents through the tiled decode and the peak: its audio
        let t0 = Instant::now();
        let a = e.sound(&mut s, &f32s(dir, "latents.npy")?, true, &mut |_, _| Ok(()))?;
        let n = a.samples.len() / 2;
        let mut cm = vec![0f32; a.samples.len()];
        for i in 0..n {
            cm[i] = a.samples[2 * i];
            cm[n + i] = a.samples[2 * i + 1];
        }
        line(log, &format!("vae, tiled ({:.2} s)", t0.elapsed().as_secs_f64()), &cm, &f32s(dir, "audio.npy")?, &mut worst);
    }
    if on("song") {
        let t0 = Instant::now();
        let mut t = Trace::default();
        let enc = e.conditions(&mut s, &song, &mut t)?;
        let lat = e.sample(&mut s, &song, &enc, f32s(dir, "noise.npy")?, &mut t, &mut |_, _| Ok(()))?;
        let a = e.sound(&mut s, &lat, true, &mut |_, _| Ok(()))?;
        let n = a.samples.len() / 2;
        let mut cm = vec![0f32; a.samples.len()];
        for i in 0..n {
            cm[i] = a.samples[2 * i];
            cm[n + i] = a.samples[2 * i + 1];
        }
        line(log, &format!("song ({:.2} s)", t0.elapsed().as_secs_f64()), &cm, &f32s(dir, "audio.npy")?, &mut worst);
    }
    Ok(worst)
}
