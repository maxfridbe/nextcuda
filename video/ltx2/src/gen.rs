//! A clip, as the movie maker's LTX-2.5 engine makes one (engines/ltx2.py's generate_segment): the prompt through the
//! text encoder and the connectors (each loaded, run and released in turn), the DiT's guided denoise over noise (and
//! the first frame's latents for image to video), the video VAE's tiled decode, the audio VAE and the vocoder, an mp4.
//!
//! The frame count: what is asked (the studio's 17k + 5 grid, or any) made as the next 8k + 1 and cut back to it; the
//! canvas a multiple of 32. The noise is this engine's own (a seeded normal generator, not torch's), so a seed repeats
//! here, not across engines.

use std::path::PathBuf;
use std::time::Instant;

use nextcuda_core::{Error, Result};
use serde_json::{json, Value};

use crate::dit::{Shape, CH};
use crate::{conn, sample, Ltx2};

/// splitmix64, then Box-Muller: standard normals
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed ^ 0x9e37_79b9_7f4a_7c15)
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f64 {
        ((self.next() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    pub fn normals(&mut self, n: usize) -> Vec<f32> {
        let mut v = Vec::with_capacity(n + 1);
        while v.len() < n {
            let (u1, u2) = (self.uniform(), self.uniform());
            let r = (-2.0 * u1.ln()).sqrt();
            let a = 2.0 * std::f64::consts::PI * u2;
            v.push((r * a.cos()) as f32);
            v.push((r * a.sin()) as f32);
        }
        v.truncate(n);
        v
    }
}

/// The next 8k + 1 at or above n (9 at least)
pub fn align_frames(n: usize) -> usize {
    let n = n.max(9);
    n + (8 - (n - 1) % 8) % 8
}

pub struct Ctl<'a> {
    pub log: &'a mut dyn FnMut(String),
    pub progress: &'a mut dyn FnMut(&'static str, usize, usize),
    pub cancel: &'a dyn Fn() -> bool,
}

/// The generate job: {prompt, negative, width, height, frames | seconds, steps, seed, out, first_frame, fast}
pub fn generate(m: &Ltx2, spec: &Value, c: &mut Ctl) -> Result<Value> {
    let t_all = Instant::now();
    let prompt = spec["prompt"].as_str().unwrap_or("").trim().to_string();
    if prompt.is_empty() {
        return Err(Error("generate: a prompt".into()));
    }
    for k in ["guide_clip", "first_audio", "last_frame", "ref_audio", "source", "first_latent"] {
        if !spec[k].is_null() {
            return Err(Error(format!("{k}: LTX-2.5 here makes a clip from a prompt and (optionally) a first frame only")));
        }
    }
    let negative = spec["negative"].as_str().unwrap_or("").to_string();
    let num = |k: &str| spec[k].as_f64().or_else(|| spec[k].as_str().and_then(|s| s.parse().ok()));
    let width = num("width").unwrap_or(768.0) as usize;
    let height = num("height").unwrap_or(512.0) as usize;
    if width % 32 != 0 || height % 32 != 0 || width < 64 || height < 64 {
        return Err(Error(format!("{width} x {height}: the canvas is a multiple of 32 each way")));
    }
    let fps = num("fps").unwrap_or(24.0) as f32;
    let want = match num("frames") {
        Some(f) => f as usize,
        // as H3 and the studio count them: round(seconds x fps)
        None => (num("seconds").unwrap_or(5.0) * fps as f64).round() as usize,
    }
    .max(9);
    let frames = align_frames(want);
    let fast = spec["fast"].as_bool().unwrap_or(false) || num("fast").is_some_and(|v| v != 0.0) || m.fast_default;
    let steps = num("steps").map(|s| s as usize).filter(|s| *s > 0);
    // the distilled DiT: its own sigmas unless steps are asked for
    let sig = match (fast, steps) {
        (true, None) => sample::sigmas(0, true),
        (_, s) => sample::sigmas(s.unwrap_or(sample::DEV_STEPS), false),
    };
    let nsteps = sig.len() - 1;
    let seed = num("seed").map_or(0, |s| s as i64) as u64;
    let out = PathBuf::from(spec["out"].as_str().ok_or("generate: out (an .mp4 path)")?);
    let shape = Shape { f: (frames - 1) / 8 + 1, h: height / 32, w: width / 32, la: (frames as f64 / fps as f64 * 25.0).round() as usize, fps };
    (c.log)(format!("ltx2: {width} x {height}, {want} frames (made as {frames}), {nsteps} steps{}, seed {seed}", if fast { ", the distilled DiT" } else { "" }));

    // 1. the prompt (and the negative): the text encoder, then the connectors, each released after
    (c.progress)("encode", 0, nsteps);
    let t0 = Instant::now();
    m.with_dit(fast, &mut *c.log, |_| Ok(()))?;
    let (pid, nid) = (m.tokens(&prompt), m.tokens(&negative));
    let (ps, ns) = {
        let te = m.text_encoder(&mut *c.log)?;
        let r = (te.encode(&pid)?, te.encode(&nid)?);
        drop(te);
        r
    };
    let ((pv, pa), (nv, na)) = (m.connectors(&ps, &mut *c.log)?, m.connectors(&ns, &mut *c.log)?);
    let encode_s = t0.elapsed().as_secs_f64();
    (c.log)(format!("ltx2: prompt ({} tokens) encoded in {encode_s:.1} s", pid.len()));
    debug_assert_eq!(pv.len(), conn::SEQ * 4096);

    // 2. the first frame's latents (image to video)
    let first = match spec["first_frame"].as_str().filter(|s| !s.is_empty()) {
        Some(p) => {
            let t0 = Instant::now();
            let px = crate::media::read_picture(std::path::Path::new(p), width, height)?;
            let z = m.encode_first(&px, height, width)?;
            (c.log)(format!("ltx2: conditioned on {p} in {:.1} s", t0.elapsed().as_secs_f64()));
            Some(z)
        }
        None => None,
    };

    // 3. the denoise
    let mut rng = Rng::new(seed);
    let tv = shape.tv();
    let mut lat = rng.normals(tv * CH);
    let mut aud = rng.normals(shape.la * CH);
    let clean = if let Some(z) = &first {
        let n = shape.h * shape.w * CH;
        lat[..n].copy_from_slice(&z[..n]);
        shape.h * shape.w
    } else {
        0
    };
    let t0 = Instant::now();
    let (lat, aud, la, peak) = m.with_dit(fast, &mut *c.log, |d| {
        let cond = d.text(&pv, &pa)?;
        let neg = d.text(&nv, &na)?;
        let la = shape.la;
        let prep = d.prepare(Shape { ..shape }, first.is_some())?;
        let mut prog = |i: usize, n: usize| (c.progress)("denoise", i, n);
        sample::denoise(d, &prep, &cond, &neg, &mut lat, &mut aud, &sig, clean, &mut sample::Hooks { progress: &mut prog, cancel: c.cancel, each: None })?;
        let peak: Vec<f64> = d.gpus().iter().map(|g| g.memory().map_or(0.0, |(t, f)| (t - f.unwrap_or(t)) as f64 / (1u64 << 30) as f64)).collect();
        Ok((lat, aud, la, peak))
    })?;
    let denoise_s = t0.elapsed().as_secs_f64();
    (c.log)(format!("ltx2: {nsteps} steps in {denoise_s:.1} s ({:.2} s a step); cards in use {peak:.2?} GiB", denoise_s / nsteps as f64));

    // the final latents, when asked (a check of the decoders; the upscaler's input): raw float32 [Tv, 128], [La, 128]
    if let Some(p) = spec["save_latents"].as_str() {
        std::fs::write(format!("{p}.video.f32"), crate::te::as_bytes(&lat)).map_err(|e| Error(format!("{p}: {e}")))?;
        std::fs::write(format!("{p}.audio.f32"), crate::te::as_bytes(&aud)).map_err(|e| Error(format!("{p}: {e}")))?;
    }
    // 4. pictures and sound
    let t0 = Instant::now();
    let (f, h, w) = ((frames - 1) / 8 + 1, height / 32, width / 32);
    let px = {
        let dec = crate::vae::Decoder::load(m.file("vae")?, m.freest())?;
        let mut p = |i: usize, n: usize| (c.progress)("decode", i, n);
        dec.decode(&lat, f, h, w, &mut p)?
    };
    let sound = {
        let ad = crate::audio::AudioDecoder::load(m.file("audio-vae")?, m.freest())?;
        let mel = ad.decode(&aud, la)?;
        drop(ad);
        let voc = crate::audio::Audio::load(m.file("vocoder")?, m.freest())?;
        voc.vocode(&mel, 4 * la - 3)?
    };
    let decode_s = t0.elapsed().as_secs_f64();
    crate::media::write_mp4(&out, &px, want.min(frames), height, width, fps, Some((&sound, crate::audio::SAMPLE_RATE)))?;
    let secs = t_all.elapsed().as_secs_f64();
    (c.log)(format!("ltx2: decoded in {decode_s:.1} s; {} in {secs:.1} s", out.display()));
    Ok(json!({
        "out": out, "engine": "ltx2", "fast": fast, "steps": nsteps, "width": width, "height": height, "frames": want.min(frames), "gen_frames": frames,
        "fps": fps, "seed": seed, "sample_rate": crate::audio::SAMPLE_RATE, "encode_secs": round1(encode_s), "denoise_secs": round1(denoise_s),
        "decode_secs": round1(decode_s), "secs": round1(secs), "gpu_gib": peak,
    }))
}

fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}
