//! Stage by stage against the reference (reference/voiceisolator/ref.py: Inworld's own package in float32 on the CPU),
//! each from the reference's own input.
//!
//! ```text
//!   spec    input48.npy (its first chunk)          -> spec.npy [bins, frames, 2]
//!   net     spec.npy                               -> tokens.npy, block<i>_time.npy, block<i>.npy [bands, frames, width],
//!                                                     raw.npy [bins, frames, 4]
//!   chunk   input48.npy (its first chunk)          -> chunk.npy (the network's output)
//!   clips   clip<k>_in.npy, clip<k>.json (rate)    -> clip<k>.npy (VoiceIsolator.isolate); clip<k>_bf16.npy beside it when
//!                                                     the reference also ran under bfloat16, as it serves on a CUDA card
//! ```

use std::path::Path;
use std::time::Instant;

use nextcuda_audio::{Error, IsolateRequest, Result};

use crate::net::Taps;
use crate::VoiceIsolator;

/// A little-endian float32 .npy (C order): (shape, values)
pub fn read_npy(path: &Path) -> Result<(Vec<usize>, Vec<f32>)> {
    let b = std::fs::read(path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
    if b.len() < 10 || &b[..6] != b"\x93NUMPY" {
        return Err(Error(format!("{}: not a .npy", path.display())));
    }
    let (hl, at) = if b[6] == 1 { (u16::from_le_bytes([b[8], b[9]]) as usize, 10) } else { (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12) };
    let head = String::from_utf8_lossy(&b[at..at + hl]).to_string();
    if !head.contains("'<f4'") || head.contains("'fortran_order': True") {
        return Err(Error(format!("{}: {head} (float32, C order expected)", path.display())));
    }
    let shape: Vec<usize> = head.split("'shape': (").nth(1).and_then(|s| s.split(')').next())
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect()).unwrap_or_default();
    Ok((shape, b[at + hl..].as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()))
}

/// (relative error ||g - w|| / ||w||, the largest absolute difference, SNR in dB of the reference against the
/// difference, cosine)
pub fn compare(got: &[f32], want: &[f32]) -> (f64, f64, f64, f64) {
    let (mut e2, mut w2, mut g2, mut dot, mut mx) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (g, w) in got.iter().zip(want) {
        let (g, w) = (*g as f64, *w as f64);
        e2 += (g - w) * (g - w);
        w2 += w * w;
        g2 += g * g;
        dot += g * w;
        mx = mx.max((g - w).abs());
    }
    let rel = (e2 / w2.max(1e-30)).sqrt();
    (rel, mx, 10.0 * (w2.max(1e-30) / e2.max(1e-30)).log10(), dot / (w2.sqrt() * g2.sqrt()).max(1e-30))
}

fn line(log: &mut dyn FnMut(String), name: &str, got: &[f32], want: &[f32]) -> f64 {
    if got.len() != want.len() {
        log(format!("{name:>14}: {} values here, {} in the reference", got.len(), want.len()));
        return f64::INFINITY;
    }
    let (rel, mx, snr, cos) = compare(got, want);
    log(format!("{name:>14}: rel {rel:.2e}  max abs {mx:.2e}  SNR {snr:6.1} dB  cos {cos:.7}"));
    rel
}

/// Runs the stages asked for; the worst relative error
pub fn run(e: &VoiceIsolator, dir: &Path, stages: &[&str], log: &mut dyn FnMut(String)) -> Result<f64> {
    let mut worst = 0f64;
    let rate = e.net.cfg.rate as f64;
    let chunk_len = (e.net.cfg.chunk_seconds * rate) as usize;
    let bins = e.net.cfg.n_fft / 2 + 1;
    let input = || -> Result<Vec<f32>> {
        let (_, x) = read_npy(&dir.join("input48.npy"))?;
        Ok(x[..x.len().min(chunk_len)].to_vec())
    };
    if stages.contains(&"spec") {
        let x = input()?;
        let s = e.stft.forward(&x);
        let (shape, want) = read_npy(&dir.join("spec.npy"))?;
        // ours [frames, bins, 2] -> the reference's [bins, frames, 2]
        let mut got = vec![0f32; want.len()];
        if shape != [bins, s.frames, 2] {
            return Err(Error(format!("spec.npy {shape:?}, here [{bins}, {}, 2]", s.frames)));
        }
        for t in 0..s.frames {
            for k in 0..bins {
                got[(k * s.frames + t) * 2] = s.v[t * bins + k][0];
                got[(k * s.frames + t) * 2 + 1] = s.v[t * bins + k][1];
            }
        }
        worst = worst.max(line(log, "spec", &got, &want));
    }
    if stages.contains(&"net") {
        let (shape, sv) = read_npy(&dir.join("spec.npy"))?;
        let frames = shape[1];
        let mut spec = vec![[0f32; 2]; frames * bins];
        for k in 0..bins {
            for t in 0..frames {
                spec[t * bins + k] = [sv[(k * frames + t) * 2], sv[(k * frames + t) * 2 + 1]];
            }
        }
        let mut names: Vec<String> = vec!["tokens".into()];
        for i in 0..e.net.cfg.blocks.len() {
            if dir.join(format!("block{i}_time.npy")).is_file() {
                names.push(format!("block{i}_time"));
            }
            names.push(format!("block{i}"));
        }
        let want_names: Vec<&str> = names.iter().map(String::as_str).collect();
        let mut taps = Taps { want: &want_names, got: Vec::new() };
        let t0 = Instant::now();
        let raw = e.net.run(&spec, frames, Some(&mut taps), &mut |_, _| Ok(()))?;
        log(format!("the network over {frames} frames in {:.2} s (with the taps read back)", t0.elapsed().as_secs_f64()));
        for (name, got) in &taps.got {
            let (_, want) = read_npy(&dir.join(format!("{name}.npy")))?;
            worst = worst.max(line(log, name, got, &want));
        }
        // ours band-major [frames, 4 w_b] a band -> the reference's [bins, frames, 4]
        let (_, want) = read_npy(&dir.join("raw.npy"))?;
        let mut got = vec![0f32; want.len()];
        for &(s, en) in &e.net.edges {
            let w = en - s;
            for t in 0..frames {
                for k in 0..w {
                    for c in 0..4 {
                        got[((s + k) * frames + t) * 4 + c] = raw[4 * frames * s + t * 4 * w + k * 4 + c];
                    }
                }
            }
        }
        worst = worst.max(line(log, "raw", &got, &want));
    }
    if stages.contains(&"chunk") {
        let x = input()?;
        let t0 = Instant::now();
        let y = e.forward(&x, None, &mut |_, _| Ok(()))?;
        log(format!("a chunk of {:.2} s in {:.3} s", x.len() as f64 / rate, t0.elapsed().as_secs_f64()));
        let (_, want) = read_npy(&dir.join("chunk.npy"))?;
        worst = worst.max(line(log, "chunk", &y, &want));
    }
    if stages.contains(&"clips") {
        for k in 0.. {
            let meta = dir.join(format!("clip{k}.json"));
            if !meta.is_file() {
                break;
            }
            let m: serde_json::Value = serde_json::from_slice(&std::fs::read(&meta).map_err(|e| Error(e.to_string()))?).map_err(|e| Error(e.to_string()))?;
            let r = m["rate"].as_u64().unwrap_or(48000) as u32;
            let (_, x) = read_npy(&dir.join(format!("clip{k}_in.npy")))?;
            let t0 = Instant::now();
            let out = e.run(&IsolateRequest { samples: x.clone(), rate: r, out_rate: None, extra: Default::default() }, &mut |_| Ok(()))?;
            let took = t0.elapsed().as_secs_f64();
            let secs = x.len() as f64 / r as f64;
            log(format!("clip{k} {} ({r} Hz, {secs:.2} s) in {took:.3} s (real-time factor {:.4})", m["name"].as_str().unwrap_or("?"), took / secs));
            let (_, want) = read_npy(&dir.join(format!("clip{k}.npy")))?;
            worst = worst.max(line(log, "voice", &out.voice.samples, &want));
            let wb: Vec<f32> = x.iter().zip(&want).map(|(a, b)| a - b).collect();
            line(log, "background", &out.background.samples, &wb);
            let bf = dir.join(format!("clip{k}_bf16.npy"));
            if bf.is_file() {
                let (_, b) = read_npy(&bf)?;
                line(log, "bf16 vs fp32", &b, &want);
                line(log, "here vs bf16", &out.voice.samples, &b);
            }
            let rms = |v: &[f32]| (v.iter().map(|s| (*s as f64).powi(2)).sum::<f64>() / v.len().max(1) as f64).sqrt();
            log(format!("{:>14}: input {:.1} dBFS, voice {:.1} dBFS, removed {:.1} dBFS", "levels", 20.0 * rms(&x).log10(),
                        20.0 * rms(&out.voice.samples).log10(), 20.0 * rms(&out.background.samples).log10()));
        }
    }
    Ok(worst)
}
