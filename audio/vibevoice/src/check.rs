//! Stage by stage against the reference (reference/vibevoice/ref.py: yvoice's own code in float32 on the CPU, the
//! same GGUF), each stage from the reference's own inputs:
//!
//! ```text
//!   prompt    meta.json's text and recordings  -> tokens.npy, speech_mask.npy, ref_wav_<i>.npy (normalised)
//!   voice     the recordings                   -> enc_mean_<i>.npy; voice_lat_<i>.npy through the connector -> embeds.npy
//!   prefill   embeds.npy                       -> the first call's hidden state (calls.npy)
//!   head      cond / neg / noise.npy, a frame each -> latent.npy
//!   decode    latent.npy streamed (reset at each speech end) -> chunk.npy
//!   semantic  chunk.npy streamed               -> sem.npy
//!   frames    the whole loop with the reference's draws and tokens forced (teacher forcing) -> cond, neg, next.npy,
//!             the take (audio.npy); the model's own choice of token at each step against the reference's
//! ```

use std::path::Path;
use std::time::Instant;

use nextcuda_audio::{Error, Result};
use nextcuda_core::DevBuf;

use crate::head::Solver;
use crate::{Forced, Trace, VibeVoice};

/// A little-endian .npy of float32, int8, int32 or int64 (C order): (shape, values as f64)
pub fn read_npy(path: &Path) -> Result<(Vec<usize>, Vec<f64>)> {
    let b = std::fs::read(path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
    if b.len() < 10 || &b[..6] != b"\x93NUMPY" {
        return Err(Error(format!("{}: not a .npy", path.display())));
    }
    let (hl, at) = if b[6] == 1 { (u16::from_le_bytes([b[8], b[9]]) as usize, 10) } else { (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12) };
    let head = String::from_utf8_lossy(&b[at..at + hl]).to_string();
    let fortran = head.contains("'fortran_order': True");
    let shape: Vec<usize> = head.split("'shape': (").nth(1).and_then(|s| s.split(')').next())
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect()).unwrap_or_default();
    let d = &b[at + hl..];
    let v: Vec<f64> = if head.contains("'<f4'") {
        d.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64).collect()
    } else if head.contains("'|i1'") || head.contains("'|b1'") || head.contains("'|u1'") {
        d.iter().map(|x| *x as i8 as f64).collect()
    } else if head.contains("'<i4'") {
        d.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64).collect()
    } else if head.contains("'<i8'") {
        d.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().expect("8 bytes")) as f64).collect()
    } else {
        return Err(Error(format!("{}: {head}", path.display())));
    };
    // column-major (a transposed tensor saved as it was): to row-major, 2-D
    if fortran && shape.len() == 2 {
        let (r, c) = (shape[0], shape[1]);
        let t = (0..r * c).map(|i| v[(i % c) * r + i / c]).collect();
        return Ok((shape, t));
    }
    if fortran && shape.len() > 2 {
        return Err(Error(format!("{}: Fortran order of {} dimensions", path.display(), shape.len())));
    }
    Ok((shape, v))
}

fn f32s(v: Vec<f64>) -> Vec<f32> {
    v.into_iter().map(|x| x as f32).collect()
}

/// (relative error, largest difference, cosine)
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

fn report(log: &mut dyn FnMut(String), worst: &mut f64, what: &str, got: &[f32], want: &[f32]) {
    if got.len() != want.len() {
        log(format!("{what:<28} sizes differ: {} here, {} in the reference", got.len(), want.len()));
        *worst = f64::INFINITY;
        return;
    }
    let (rel, mx, cos) = compare(got, want);
    log(format!("{what:<28} rel {rel:.2e}  max {mx:.2e}  cos {cos:.6}"));
    *worst = worst.max(rel);
}

/// Rows of a [n, w] array
fn rows(v: &[f32], w: usize) -> Vec<Vec<f32>> {
    v.chunks_exact(w).map(<[f32]>::to_vec).collect()
}

pub fn run(e: &VibeVoice, dir: &Path, stages: &[&str], log: &mut dyn FnMut(String)) -> Result<f64> {
    let mut worst = 0f64;
    let want = |n: &str| -> Result<Vec<f32>> { Ok(f32s(read_npy(&dir.join(n))?.1)) };
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json")).map_err(|e| Error(e.to_string()))?)
        .map_err(|e| Error(e.to_string()))?;
    let text = meta["text"].as_str().unwrap_or("").to_string();
    let steps = meta["steps"].as_u64().unwrap_or(10) as usize;
    let cfg = meta["cfg"].as_f64().unwrap_or(1.3) as f32;
    let (ops, nsd) = (&e.ops, &e.nsd);
    let h = e.lm.hidden;
    // the recordings, as the reference read them (24 kHz)
    let mut wavs = Vec::new();
    for r in meta["refs"].as_array().into_iter().flatten() {
        let p = r.as_str().unwrap_or("");
        let b = std::fs::read(p).map_err(|er| Error(format!("{p}: {er}")))?;
        let (s, rate) = nextcuda_audio::decode_audio(&b).map_err(Error)?;
        wavs.push(crate::resample(&s, rate, crate::RATE));
    }
    let p = e.prompt(&text, &wavs)?;
    let toks: Vec<u32> = read_npy(&dir.join("tokens.npy"))?.1.iter().map(|x| *x as u32).collect();
    let mask: Vec<bool> = read_npy(&dir.join("speech_mask.npy"))?.1.iter().map(|x| *x != 0.0).collect();
    let seq: Vec<u32> = read_npy(&dir.join("sequence.npy"))?.1.iter().map(|x| *x as u32).collect();
    let ids = e.ids;

    if stages.contains(&"prompt") {
        let same = p.tokens == toks && p.speech == mask;
        log(format!("{:<28} {} rows here, {} in the reference: {}", "prompt tokens", p.tokens.len(), toks.len(), if same { "identical" } else { "DIFFERENT" }));
        if !same {
            let first = p.tokens.iter().zip(&toks).position(|(a, b)| a != b).unwrap_or(p.tokens.len().min(toks.len()));
            log(format!("  first difference at row {first}: here {:?}, reference {:?}", &p.tokens[first..(first + 8).min(p.tokens.len())],
                        &toks[first..(first + 8).min(toks.len())]));
            worst = f64::INFINITY;
        }
        for (i, w) in p.voices.iter().enumerate() {
            if let Ok(r) = want(&format!("ref_wav_{i}.npy")) {
                let n = w.len().min(r.len());
                report(log, &mut worst, &format!("speaker {i}'s recording"), &w[..n], &r[..n]);
            }
        }
    }

    if stages.contains(&"voice") {
        let embeds = want("embeds.npy")?;
        let mut speech_rows = Vec::new();
        let longest = p.voices.iter().map(Vec::len).max().unwrap_or(0);
        for (i, w) in p.voices.iter().enumerate() {
            let lat = want(&format!("voice_lat_{i}.npy"))?;
            let mut rng = crate::Rng::new(0);
            let (rws, mean, _) = e.voice_rows(w, longest, &mut rng, Some(&lat))?;
            report(log, &mut worst, &format!("speaker {i}'s encoder means"), &mean, &want(&format!("enc_mean_{i}.npy"))?);
            speech_rows.extend(rws);
        }
        // the rows the language model reads: the connector's at the speech rows, the embedding elsewhere
        let (mut got_s, mut want_s, mut got_t, mut want_t) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut vi = 0;
        for (i, &t) in toks.iter().enumerate() {
            let w = &embeds[i * h..(i + 1) * h];
            if mask[i] {
                got_s.extend_from_slice(&speech_rows[vi * h..(vi + 1) * h]);
                want_s.extend_from_slice(w);
                vi += 1;
            } else {
                got_t.extend(e.row(t)?);
                want_t.extend_from_slice(w);
            }
        }
        report(log, &mut worst, "voice rows (connector)", &got_s, &want_s);
        report(log, &mut worst, "text rows (embedding)", &got_t, &want_t);
    }

    if stages.contains(&"prefill") {
        let embeds = want("embeds.npy")?;
        let l = embeds.len() / h;
        let c = e.lm.cache(ops, 512.min(l), l + 4)?;
        let last = e.lm.feed(ops, nsd, &c, &embeds, 0)?;
        let got = c.out.to_f32()?[last..last + h].to_vec();
        let calls = want("calls.npy")?;
        report(log, &mut worst, "prefill: last hidden", &got, &calls[h..2 * h]);
        let (best, logits) = e.head_lm.choose(ops, c.out.fp().wrapping_add(last))?;
        log(format!("{:<28} {:?} -> picks {} (the reference: {})", "prefill: four logits", logits.iter().map(|x| (x * 100.0).round() / 100.0).collect::<Vec<_>>(),
                    e.head_lm.ids[best], seq.first().copied().unwrap_or(0)));
    }

    if stages.contains(&"head") {
        let (cond, neg, noise, lat) = (want("cond.npy")?, want("neg.npy")?, want("noise.npy")?, want("latent.npy")?);
        let frames = lat.len() / 64;
        let solver = Solver::new(steps);
        let tembs: Vec<DevBuf> = solver.timesteps.iter().map(|&t| -> Result<DevBuf> { Ok(DevBuf::from_f32(&ops.gpu, &e.head.t_embed(ops, nsd, t as f32)?)?) })
            .collect::<Result<_>>()?;
        let two = DevBuf::f32(&ops.gpu, 2 * h)?;
        let mut got = Vec::new();
        for f in 0..frames {
            let mut c2 = cond[f * h..(f + 1) * h].to_vec();
            c2.extend_from_slice(&neg[f * h..(f + 1) * h]);
            two.write(0, &crate::bytes(&c2))?;
            let cnd = e.head.condition(ops, nsd, two.fp())?;
            let mut z = noise[f * 128..f * 128 + 64].to_vec();
            solver.run(&mut z, &mut |i, x| {
                let v = e.head.velocity(ops, nsd, &cnd, x, &tembs[i])?;
                Ok((0..64).map(|j| v[64 + j] + cfg * (v[j] - v[64 + j])).collect())
            })?;
            got.extend(z);
        }
        report(log, &mut worst, &format!("head: {frames} latents"), &got, &lat);
        report(log, &mut worst, "head: the first latent", &got[..64], &lat[..64]);
    }

    if stages.contains(&"decode") || stages.contains(&"semantic") {
        let (lat, chunk) = (want("latent.npy")?, want("chunk.npy")?);
        let frames = lat.len() / 64;
        let (ad, se) = (&e.acoustic_dec, &e.semantic_enc);
        let (dst, sst) = (ad.state(ops)?, se.state(ops)?);
        let (dw, sw) = (ad.work(ops, 1, false)?, se.work(ops, ad.hop, false)?);
        let (mut got_c, mut got_s) = (Vec::new(), Vec::new());
        let mut f = 0;
        for t in &seq {
            if *t == ids.end {
                dst.reset()?;
                sst.reset()?;
            }
            if *t != ids.frame || f >= frames {
                continue;
            }
            if stages.contains(&"decode") {
                let scaled: Vec<f32> = lat[f * 64..(f + 1) * 64].iter().map(|v| v / e.scaling - e.bias).collect();
                got_c.extend(ad.run(ops, nsd, &dst, &dw, &scaled, false)?);
            }
            if stages.contains(&"semantic") {
                let s = se.run(ops, nsd, &sst, &sw, &chunk[f * ad.hop..(f + 1) * ad.hop], false)?;
                got_s.extend_from_slice(&s[s.len() - 128..]);
            }
            f += 1;
        }
        if stages.contains(&"decode") {
            report(log, &mut worst, &format!("decoder: {frames} frames"), &got_c, &chunk[..got_c.len()]);
            report(log, &mut worst, "decoder: the first frame", &got_c[..ad.hop], &chunk[..ad.hop]);
        }
        if stages.contains(&"semantic") {
            let ws = want("sem.npy")?;
            report(log, &mut worst, &format!("semantic: {frames} frames"), &got_s, &ws);
            let per: Vec<String> = (0..got_s.len() / 128).map(|f| format!("{:.0e}", compare(&got_s[f * 128..(f + 1) * 128], &ws[f * 128..(f + 1) * 128]).0)).collect();
            log(format!("{:<28} {}", "semantic: by frame", per.join(" ")));
        }
    }

    if stages.contains(&"frames") {
        let vl: Vec<Vec<f32>> = (0..p.voices.len()).map(|i| want(&format!("voice_lat_{i}.npy"))).collect::<Result<_>>()?;
        let noise: Vec<Vec<f32>> = rows(&want("noise.npy")?, 128).into_iter().map(|r| r[..64].to_vec()).collect();
        let forced = Forced { voice_lat: Some(vl), noise: Some(noise), latent: Some(rows(&want("latent.npy")?, 64)), sem: Some(rows(&want("sem.npy")?, 128)),
                              tokens: Some(seq.clone()) };
        let mut tr = Trace::default();
        let t0 = Instant::now();
        let audio = e.generate_speech(&p, seq.len() + 1, steps, cfg, 0, &forced, Some(&mut tr), t0, &mut |_| Ok(()))?;
        let secs = t0.elapsed().as_secs_f64();
        let flat = |v: &[Vec<f32>]| v.concat();
        report(log, &mut worst, "frames: positive conditions", &flat(&tr.cond), &want("cond.npy")?);
        report(log, &mut worst, "frames: negative conditions", &flat(&tr.neg), &want("neg.npy")?);
        report(log, &mut worst, "frames: latents (free head)", &flat(&tr.latent), &want("latent.npy")?);
        report(log, &mut worst, "frames: next rows", &flat(&tr.next), &want("next.npy")?);
        let ra = want("audio.npy")?;
        let n = audio.len().min(ra.len());
        report(log, &mut worst, "frames: the take", &audio[..n], &ra[..n]);
        log(format!("{:<28} {} frames, {:.2} s of speech in {secs:.1} s", "frames: speed", tr.latent.len(), audio.len() as f64 / crate::RATE as f64));
    }
    Ok(worst)
}
