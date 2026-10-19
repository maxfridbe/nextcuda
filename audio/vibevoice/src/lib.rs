//! VibeVoice 1.5B (Microsoft): long-form, multi-speaker speech, each speaker's voice cloned from a recording. A
//! Qwen2.5-1.5B reads a script of "Speaker N:" lines after the speakers' recordings (their acoustic latents as rows);
//! each speech frame it calls for (7.5 a second) is drawn by a diffusion head from its hidden state, decoded to
//! 3,200 samples of 24 kHz, and fed back as the next row (the frame's acoustic latent and the semantic encoder's
//! reading of its samples).
//!
//! ```text
//!   prompt (lib.rs, as yvoice's processor): system text, " Voice input:", per speaker " Speaker i:" <start>
//!     latents... <end> "\n", " Text input:", the lines, " Speech output:" <start>
//!   loop (as yvoice's generate, one sample): the model picks one of <start> <end> <frame> <eos> (greedy);
//!     <frame>: the head's latent under guidance against a negative pass (the same rows since the turn's <start>,
//!     nothing else), decoded (codec.rs) and re-encoded; <end>: the tokenizers' streams start over; <start>: so
//!     does the negative pass
//! ```
//!
//! The weights are the GGUF the platform uses (`gguf-org/vibevoice-gguf`, the model's own tensor names): bf16 (the
//! language model half or `--opt-int8`; the tokenizers' products half, their convolutions float32), or the q8_0 file
//! (its matrices dequantized). config.json and tokenizer.json beside it. `check` compares the stages with the
//! reference's dumps (reference/vibevoice/ref.py: yvoice's own code on the same file).

pub mod check;
pub mod codec;
mod ffi;
pub mod head;
pub mod lm;
pub mod ops;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use nextcuda_audio::{At, Audio, AudioEngine, AudioKind, AudioRequest, Defaults, EngineOption, Error, LoadOptions, ModelFiles, Reference, Result, Speech, Step};
use nextcuda_core::{DevBuf, Gpu};
use nextcuda_diffusion::kernels::{none, Dt, Nsd};
use nextcuda_tok::Tokenizer;

use crate::codec::Stack;
use crate::head::{Head, Solver};
use crate::lm::Lm;
use crate::ops::{stage, Mat, Ops, Stage, Weights};

pub const ARCH: &str = "vibevoice";
pub const ROLES: &[&str] = &["model", "config", "tokenizer"];
pub const RATE: u32 = 24000;
/// speech frames a second (3,200 samples each)
pub const FPS: f32 = 7.5;
const MAX_FRAMES: usize = 64 * 60 * 75 / 10;
const SYSTEM: &str = " Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.\n";
/// the reference's audio normaliser: -25 dB FS
const TARGET_DB: f32 = -25.0;

pub const OPTIONS: &[EngineOption] = &[
    EngineOption { name: "int8", env: "NS_VV_INT8", value: "0|1", help: "the language model's matrices in int8 (less VRAM)", at: At::Load },
];

pub fn kind() -> AudioKind {
    AudioKind { archs: &[ARCH], name: "VibeVoice 1.5B (long-form multi-speaker speech: Speaker N: lines, each speaker cloned from a recording)", roles: ROLES,
                load, options: OPTIONS }
}

fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn AudioEngine>> {
    let gpu = gpus.first().ok_or_else(|| Error("no GPU".into()))?;
    Ok(Box::new(VibeVoice::load(files, gpu, o, log)?))
}

/// Floats as their bytes
pub fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// A connector (acoustic 64 -> 1536 / semantic 128 -> 1536): fc1, RMS norm, fc2
struct Connector {
    fc1: Mat,
    b1: DevBuf,
    norm: DevBuf,
    fc2: Mat,
    b2: DevBuf,
}

impl Connector {
    fn load(ops: &Ops, w: &Weights, pre: &str) -> Result<Connector> {
        Ok(Connector {
            fc1: w.mat1(ops, &format!("{pre}.fc1.weight"), false)?,
            b1: w.dev_f32(ops, &format!("{pre}.fc1.bias"))?,
            norm: w.dev_f32(ops, &format!("{pre}.norm.weight"))?,
            fc2: w.mat1(ops, &format!("{pre}.fc2.weight"), false)?,
            b2: w.dev_f32(ops, &format!("{pre}.fc2.bias"))?,
        })
    }

    /// rows x [m, in] (device) -> out [m, hidden] (device; `tmp` m x hidden)
    fn run(&self, ops: &Ops, nsd: &Nsd, st: &Stage, x: *const f32, m: usize, tmp: &DevBuf, out: *mut f32) -> Result<()> {
        self.fc1.apply(ops, nsd, x, m, st, self.b1.ptr(), tmp.fp())?;
        nsd.rms_norm_mod(tmp.ptr(), Dt::F32, m, self.fc1.n, self.norm.ptr(), 1e-6, none(), none(), none(), tmp.ptr(), Dt::F32)?;
        self.fc2.apply(ops, nsd, tmp.fp(), m, st, self.b2.ptr(), out)
    }
}

pub struct VibeVoice {
    pub ops: Ops,
    pub nsd: Nsd,
    pub tok: Tokenizer,
    pub weights: Weights,
    pub lm: Lm,
    /// the four tokens the generation picks from: start, end, frame, eos
    pub head_lm: lm::Head,
    pub head: Head,
    pub acoustic_enc: Stack,
    pub acoustic_dec: Stack,
    pub semantic_enc: Stack,
    acoustic_conn: Connector,
    semantic_conn: Connector,
    pub scaling: f32,
    pub bias: f32,
    pub ids: Ids,
    int8: bool,
    loaded: Instant,
    load_s: f64,
}

/// The special tokens
#[derive(Clone, Copy, Debug)]
pub struct Ids {
    pub start: u32,
    pub end: u32,
    pub frame: u32,
    pub eos: u32,
}

fn role(files: &ModelFiles, r: &str) -> Result<PathBuf> {
    files.get(r).cloned().ok_or_else(|| Error(format!("{ARCH}: no {r} file")))
}

/// A recording to 24 kHz mono (a windowed-sinc resampler: Hann, 32 zero crossings, cut at the lower Nyquist)
pub fn resample(x: &[f32], rate: u32, to: u32) -> Vec<f32> {
    if rate == to || x.is_empty() {
        return x.to_vec();
    }
    let ratio = to as f64 / rate as f64;
    let cut = ratio.min(1.0);
    let zc = 32.0;
    let half = (zc / cut).ceil() as isize;
    let n = (x.len() as f64 * ratio).round() as usize;
    (0..n).map(|i| {
        let center = i as f64 / ratio;
        let c0 = center.floor() as isize;
        let mut s = 0.0;
        for j in c0 - half..=c0 + half {
            if j < 0 || j as usize >= x.len() {
                continue;
            }
            let d = (j as f64 - center) * cut;
            if d.abs() >= zc {
                continue;
            }
            let sinc = if d == 0.0 { 1.0 } else { (std::f64::consts::PI * d).sin() / (std::f64::consts::PI * d) };
            s += x[j as usize] as f64 * sinc * (0.5 + 0.5 * (std::f64::consts::PI * d / zc).cos()) * cut;
        }
        s as f32
    }).collect()
}

/// The reference's AudioNormalizer: to -25 dB FS, then scaled down if it would clip
pub fn normalize(w: &[f32]) -> Vec<f32> {
    let rms = (w.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>() / w.len().max(1) as f64).sqrt() as f32;
    let s = 10f32.powf(TARGET_DB / 20.0) / (rms + 1e-6);
    let mut v: Vec<f32> = w.iter().map(|x| x * s).collect();
    let max = v.iter().fold(0f32, |m, x| m.max(x.abs()));
    if max > 1.0 {
        let d = max + 1e-6;
        v.iter_mut().for_each(|x| *x /= d);
    }
    v
}

/// A script's lines as (speaker, text): "Speaker N: words" lines (case-insensitive), others dropped; the speakers
/// counted from 0 when none is 0 (the processor's _parse_script). A text with no such line is one speaker's.
pub fn parse_script(text: &str) -> Vec<(usize, String)> {
    let mut out = Vec::new();
    for line in text.lines() {
        let l = line.trim();
        if l.is_empty() {
            continue;
        }
        let low = l.to_ascii_lowercase();
        let Some(rest) = low.strip_prefix("speaker") else { continue };
        let rest_orig = &l[l.len() - rest.len()..];
        let r = rest_orig.trim_start();
        let digits: String = r.chars().take_while(|c| c.is_ascii_digit()).collect();
        if digits.is_empty() || r.len() == rest_orig.len() && !rest_orig.starts_with(char::is_whitespace) {
            continue;
        }
        let after = r[digits.len()..].trim_start();
        let Some(words) = after.strip_prefix(':') else { continue };
        out.push((digits.parse().unwrap_or(0), format!(" {}", words.trim())));
    }
    if out.is_empty() {
        // plain text: one speaker, a line a paragraph (as the platform's to_script)
        out = text.lines().map(str::trim).filter(|l| !l.is_empty()).map(|l| (1, format!(" {l}"))).collect();
    }
    if let Some(min) = out.iter().map(|(s, _)| *s).min() {
        if min > 0 {
            out.iter_mut().for_each(|(s, _)| *s -= 1);
        }
    }
    out
}

/// splitmix64 and Box-Muller: the noise from a seed
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed ^ 0x7669_6265_766f_6963)
    }

    fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn normal(&mut self) -> f32 {
        let (u, v) = (self.uniform().max(1e-300), self.uniform());
        ((-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()) as f32
    }
}

/// The prompt as the model takes it
pub struct Prompt {
    pub tokens: Vec<u32>,
    /// which rows are a recording's latents (in order: speaker 0's frames, then speaker 1's ...)
    pub speech: Vec<bool>,
    /// the speakers' recordings (24 kHz, normalised) in the order their rows come
    pub voices: Vec<Vec<f32>>,
    /// the script's words, tokens: the budget's measure
    pub chars: usize,
}

/// What one generation's stages left (a check's comparison)
#[derive(Default)]
pub struct Trace {
    pub voice_means: Vec<Vec<f32>>,
    pub voice_lat: Vec<Vec<f32>>,
    pub embeds: Vec<f32>,
    pub prefill_h: Vec<f32>,
    pub cond: Vec<Vec<f32>>,
    pub neg: Vec<Vec<f32>>,
    pub latent: Vec<Vec<f32>>,
    pub chunk: Vec<Vec<f32>>,
    pub sem: Vec<Vec<f32>>,
    pub next: Vec<Vec<f32>>,
    pub tokens: Vec<u32>,
}

/// A check's inputs in place of the engine's own draws
#[derive(Default)]
pub struct Forced {
    /// each speaker's prompt latents (after the bias and scale)
    pub voice_lat: Option<Vec<Vec<f32>>>,
    /// each frame's starting noise [64]
    pub noise: Option<Vec<Vec<f32>>>,
    /// each frame's latent fed back in place of the made one (teacher forcing), with its semantic features
    pub latent: Option<Vec<Vec<f32>>>,
    pub sem: Option<Vec<Vec<f32>>>,
    /// the tokens taken in place of the model's choices
    pub tokens: Option<Vec<u32>>,
}

impl VibeVoice {
    pub fn load(files: &ModelFiles, gpu: &Arc<Gpu>, o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<VibeVoice> {
        let t0 = Instant::now();
        nextcuda_core::use_kind("audio");
        let int8 = o.setting("NS_VV_INT8").is_some_and(|v| v == "1");
        let (mp, cp, tp) = (role(files, "model")?, role(files, "config")?, role(files, "tokenizer")?);
        let conf: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&cp).map_err(|e| Error(format!("{}: {e}", cp.display())))?)
            .map_err(|e| Error(format!("{}: {e}", cp.display())))?;
        if conf["model_type"] != "vibevoice" {
            return Err(Error(format!("{}: model_type {}, not vibevoice", cp.display(), conf["model_type"])));
        }
        let weights = Weights::open(&mp)?;
        // half on the card: the file less its token embedding (left on disk), plus 1.5 GiB of work and caches
        let file = std::fs::metadata(&mp).map(|m| m.len()).unwrap_or(0);
        let need = file.saturating_sub(151_936 * 1536 * 2) / if int8 { 2 } else { 1 } + (3u64 << 29);
        let (total, free) = gpu.memory()?;
        let free = free.unwrap_or(total);
        if need > free {
            return Err(Error(format!("{ARCH} needs about {:.1} GiB on {}; {:.1} GiB are free", need as f64 / (1u64 << 30) as f64, gpu.name,
                                     free as f64 / (1u64 << 30) as f64)));
        }
        let ops = Ops::new(gpu)?;
        let nsd = Nsd::new(gpu)?;
        let tok = Tokenizer::from_hf_json(&tp).map_err(Error)?;
        let id = |t: &str| tok.id(t).ok_or_else(|| Error(format!("{}: no {t} (not Qwen2.5's tokenizer)", tp.display())));
        let ids = Ids { start: id("<|vision_start|>")?, end: id("<|vision_end|>")?, frame: id("<|vision_pad|>")?, eos: id("<|endoftext|>")? };
        let lm = Lm::load(&ops, &weights, int8, log)?;
        let head_lm = lm::Head::load(&ops, &weights, &[ids.start, ids.end, ids.frame, ids.eos])?;
        let head = Head::load(&ops, &weights)?;
        log("tokenizers".into());
        let acoustic_enc = Stack::load(&ops, &weights, "model.acoustic_tokenizer.encoder", false)?;
        let acoustic_dec = Stack::load(&ops, &weights, "model.acoustic_tokenizer.decoder", true)?;
        let semantic_enc = Stack::load(&ops, &weights, "model.semantic_tokenizer.encoder", false)?;
        let acoustic_conn = Connector::load(&ops, &weights, "model.acoustic_connector")?;
        let semantic_conn = Connector::load(&ops, &weights, "model.semantic_connector")?;
        let scaling = weights.scalar("model.speech_scaling_factor")?;
        let bias = weights.scalar("model.speech_bias_factor")?;
        if !scaling.is_finite() || !bias.is_finite() {
            return Err(Error(format!("{}: no speech scaling / bias factors (not a trained VibeVoice)", mp.display())));
        }
        gpu.sync()?;
        let load_s = t0.elapsed().as_secs_f64();
        log(format!("{ARCH} on {} in {load_s:.0} s ({:.1} GiB of matrices)", gpu.name,
                    (lm.bytes + head.bytes + acoustic_enc.bytes + acoustic_dec.bytes + semantic_enc.bytes) as f64 / (1u64 << 30) as f64));
        Ok(VibeVoice { ops, nsd, tok, weights, lm, head_lm, head, acoustic_enc, acoustic_dec, semantic_enc, acoustic_conn, semantic_conn, scaling, bias, ids,
                       int8, loaded: Instant::now(), load_s })
    }

    /// The prompt for a script and its speakers' recordings (24 kHz), as the processor makes it: a voice for each of
    /// the script's distinct speakers, in order (the recordings beyond those unused)
    pub fn prompt(&self, script: &str, voices: &[Vec<f32>]) -> Result<Prompt> {
        let lines = parse_script(script);
        if lines.is_empty() {
            return Err(Error("nothing to say".into()));
        }
        let mut speakers: Vec<usize> = lines.iter().map(|(s, _)| *s).collect();
        speakers.sort_unstable();
        speakers.dedup();
        let used = &voices[..voices.len().min(speakers.len())];
        let mut p = Prompt { tokens: Vec::new(), speech: Vec::new(), voices: Vec::new(), chars: lines.iter().map(|(_, t)| t.chars().count()).sum() };
        let text = |p: &mut Prompt, s: &str| {
            for t in self.tok.encode(s) {
                p.tokens.push(t);
                p.speech.push(false);
            }
        };
        text(&mut p, SYSTEM);
        if !used.is_empty() {
            text(&mut p, " Voice input:\n");
            for (i, w) in used.iter().enumerate() {
                let w = normalize(w);
                text(&mut p, &format!(" Speaker {i}:"));
                p.tokens.push(self.ids.start);
                p.speech.push(false);
                for _ in 0..w.len().div_ceil(3200) {
                    p.tokens.push(self.ids.frame);
                    p.speech.push(true);
                }
                p.tokens.push(self.ids.end);
                p.speech.push(false);
                text(&mut p, "\n");
                p.voices.push(w);
            }
        }
        text(&mut p, " Text input:\n");
        for (s, t) in &lines {
            text(&mut p, &format!(" Speaker {s}:{t}\n"));
        }
        text(&mut p, " Speech output:\n");
        p.tokens.push(self.ids.start);
        p.speech.push(false);
        Ok(p)
    }

    pub fn row(&self, t: u32) -> Result<Vec<f32>> {
        self.weights.rows_f32(&format!("{}embed_tokens.weight", lm::PRE), t as usize, t as usize + 1)
    }

    /// A speaker's recording -> its prompt rows [frames, hidden] (host), the encoder's means and the drawn latents
    pub fn voice_rows(&self, w: &[f32], rng: &mut Rng, forced: Option<&Vec<f32>>) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let (ops, nsd) = (&self.ops, &self.nsd);
        let st = self.acoustic_enc.state(ops)?;
        let work = self.acoustic_enc.work(ops, w.len(), true)?;
        let mean = self.acoustic_enc.run(ops, nsd, &st, &work, w, true)?;
        drop(work);
        let d = self.acoustic_enc.out_c;
        let frames = w.len().div_ceil(3200);
        let mean = mean[..frames * d].to_vec();
        // the reference's gaussian draw: one std a speaker (N(0, 1) * fix_std / 0.8), then the noise per value
        let lat: Vec<f32> = match forced {
            Some(f) => f.clone(),
            None => {
                let std = rng.normal() * 0.5 / 0.8;
                mean.iter().map(|m| (m + std * rng.normal() + self.bias) * self.scaling).collect()
            }
        };
        let rows = self.connect(&self.acoustic_conn, &lat, d)?;
        Ok((rows, mean, lat))
    }

    /// Rows [n, d] (host) through a connector -> [n, hidden] (host)
    fn connect(&self, c: &Connector, x: &[f32], d: usize) -> Result<Vec<f32>> {
        let n = x.len() / d;
        let h = self.lm.hidden;
        let g = &self.ops.gpu;
        let xd = DevBuf::from_f32(g, x)?;
        let tmp = DevBuf::f32(g, n * h)?;
        let out = DevBuf::f32(g, n * h)?;
        let st = stage(g, n * h, None)?;
        c.run(&self.ops, &self.nsd, &st, xd.fp(), n, &tmp, out.fp())?;
        Ok(out.to_f32()?)
    }

    /// One generation: the samples, and what its stages left when `trace` (a check). `max` bounds the steps (tokens).
    #[allow(clippy::too_many_arguments)]
    pub fn generate_speech(&self, p: &Prompt, max: usize, steps: usize, cfg: f32, seed: u64, forced: &Forced, mut trace: Option<&mut Trace>, t0: Instant,
                           progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<Vec<f32>> {
        let (ops, nsd, lm) = (&self.ops, &self.nsd, &self.lm);
        let h = lm.hidden;
        let mut rng = Rng::new(seed);
        // the prompt's rows: token embeddings, the speakers' latents through the acoustic connector
        let mut rows = vec![0f32; p.tokens.len() * h];
        let mut voice_rows = Vec::new();
        for (i, w) in p.voices.iter().enumerate() {
            let (r, mean, lat) = self.voice_rows(w, &mut rng, forced.voice_lat.as_ref().and_then(|v| v.get(i)))?;
            if let Some(t) = trace.as_deref_mut() {
                t.voice_means.push(mean);
                t.voice_lat.push(lat);
            }
            voice_rows.extend(r);
        }
        let mut vi = 0;
        for (i, &t) in p.tokens.iter().enumerate() {
            if p.speech[i] {
                rows[i * h..(i + 1) * h].copy_from_slice(&voice_rows[vi * h..(vi + 1) * h]);
                vi += 1;
            } else {
                rows[i * h..(i + 1) * h].copy_from_slice(&self.row(t)?);
            }
        }
        if let Some(t) = trace.as_deref_mut() {
            t.embeds = rows.clone();
        }
        let l = p.tokens.len();
        let pos = lm.cache(ops, 512.min(l), l + max + 2)?;
        let neg = lm.cache(ops, 1, max + 2)?;
        let last = lm.feed(ops, nsd, &pos, &rows, 0)?;
        progress(Step { phase: "prompt", at: 1, of: 1, seconds: t0.elapsed().as_secs_f64() })?;
        // the current positive hidden state (device copy) and position
        let hpos = DevBuf::f32(&ops.gpu, h)?;
        hpos.copy_within(0, &pos.out, last * 4, h * 4)?;
        if let Some(t) = trace.as_deref_mut() {
            t.prefill_h = hpos.to_f32()?;
        }
        let mut p_at = l;
        let mut n_at = 0;
        let ad = &self.acoustic_dec;
        let se = &self.semantic_enc;
        let (dec_st, sem_st) = (ad.state(ops)?, se.state(ops)?);
        let (dec_w, sem_w) = (ad.work(ops, 1, false)?, se.work(ops, ad.hop, false)?);
        let solver = Solver::new(steps);
        let tembs: Vec<DevBuf> = solver.timesteps.iter().map(|&t| -> Result<DevBuf> {
            Ok(DevBuf::from_f32(&ops.gpu, &self.head.t_embed(ops, nsd, t as f32)?)?)
        }).collect::<Result<_>>()?;
        let two = DevBuf::f32(&ops.gpu, 2 * h)?;
        let lat_dev = DevBuf::f32(&ops.gpu, 64)?;
        let sem_dev = DevBuf::f32(&ops.gpu, 128)?;
        let conn_tmp = DevBuf::f32(&ops.gpu, h)?;
        let conn_a = DevBuf::f32(&ops.gpu, h)?;
        let conn_s = DevBuf::f32(&ops.gpu, h)?;
        let conn_st = stage(&ops.gpu, h, None)?;
        // the row this step feeds both passes (the prompt's last, <start>, at the first)
        let mut input: Vec<f32> = rows[(l - 1) * h..l * h].to_vec();
        let mut samples: Vec<f32> = Vec::new();
        let mut frames = 0usize;
        let (ids, choices) = (self.ids, [self.ids.start, self.ids.end, self.ids.frame, self.ids.eos]);
        for step in 0..max {
            let (best, _) = self.head_lm.choose(ops, hpos.fp())?;
            let tok = match forced.tokens.as_ref() {
                Some(f) => match f.get(step) {
                    Some(t) => *t,
                    None => break,
                },
                None => choices[best],
            };
            if let Some(t) = trace.as_deref_mut() {
                t.tokens.push(tok);
            }
            if tok == ids.eos {
                break;
            }
            if tok == ids.end {
                dec_st.reset()?;
                sem_st.reset()?;
            }
            if tok == ids.start {
                n_at = 0;
            }
            let next: Vec<f32> = if tok == ids.frame {
                // the negative pass: this step's row after the turn's rows
                neg.x.write(0, &bytes(&input))?;
                lm.pass(ops, nsd, &neg, 1, n_at)?;
                n_at += 1;
                two.copy_within(0, &hpos, 0, h * 4)?;
                two.copy_within(h * 4, &neg.out, 0, h * 4)?;
                if let Some(t) = trace.as_deref_mut() {
                    t.cond.push(hpos.to_f32()?);
                    t.neg.push(neg.out.to_f32()?[..h].to_vec());
                }
                let cond = self.head.condition(ops, nsd, two.fp())?;
                let mut z: Vec<f32> = match forced.noise.as_ref().and_then(|n| n.get(frames)) {
                    Some(n) => n.clone(),
                    None => (0..64).map(|_| rng.normal()).collect(),
                };
                let head = &self.head;
                solver.run(&mut z, &mut |i, x| {
                    let v = head.velocity(ops, nsd, &cond, x, &tembs[i])?;
                    Ok((0..64).map(|j| v[64 + j] + cfg * (v[j] - v[64 + j])).collect())
                })?;
                let lat = match forced.latent.as_ref().and_then(|f| f.get(frames)) {
                    Some(f) => f.clone(),
                    None => z.clone(),
                };
                // decoded: latent / scale - bias, one frame through the decoder's stream
                let scaled: Vec<f32> = lat.iter().map(|v| v / self.scaling - self.bias).collect();
                let chunk = ad.run(ops, nsd, &dec_st, &dec_w, &scaled, false)?;
                let sem = match forced.sem.as_ref().and_then(|f| f.get(frames)) {
                    Some(f) => f.clone(),
                    None => {
                        let s = se.run(ops, nsd, &sem_st, &sem_w, &chunk, false)?;
                        s[s.len() - 128..].to_vec()
                    }
                };
                lat_dev.write(0, &bytes(&lat))?;
                sem_dev.write(0, &bytes(&sem))?;
                self.acoustic_conn.run(ops, nsd, &conn_st, lat_dev.fp(), 1, &conn_tmp, conn_a.fp())?;
                self.semantic_conn.run(ops, nsd, &conn_st, sem_dev.fp(), 1, &conn_tmp, conn_s.fp())?;
                ops.add(conn_a.fp(), conn_s.fp(), h)?;
                let row = conn_a.to_f32()?;
                if let Some(t) = trace.as_deref_mut() {
                    t.latent.push(z.clone());
                    t.chunk.push(chunk.clone());
                    t.sem.push(sem.clone());
                    t.next.push(row.clone());
                }
                samples.extend_from_slice(&chunk);
                frames += 1;
                progress(Step { phase: "frames", at: frames as u32, of: max as u32, seconds: t0.elapsed().as_secs_f64() })?;
                row
            } else {
                let i = choices.iter().position(|c| *c == tok).unwrap_or(0);
                self.head_lm.embeds[i].clone()
            };
            // the positive pass over the new row
            pos.x.write(0, &bytes(&next))?;
            lm.pass(ops, nsd, &pos, 1, p_at)?;
            p_at += 1;
            hpos.copy_within(0, &pos.out, 0, h * 4)?;
            input = next;
        }
        ops.gpu.sync()?;
        Ok(samples)
    }

    /// The speakers' recordings a request names: `speakers` in order, else the one reference
    pub fn voices(req: &AudioRequest) -> Vec<Vec<f32>> {
        let refs: Vec<&Reference> = if !req.speakers.is_empty() { req.speakers.iter().collect() } else { req.reference.iter().collect() };
        refs.iter().map(|r| resample(&r.samples, r.rate, RATE)).collect()
    }
}

/// The steps a script may take: about 14 characters of speech a second, half again on top, as the platform's
/// engine sized them (its token_budget), within `seconds` when a request names it
pub fn budget(chars: usize, seconds: Option<f32>) -> usize {
    let need = (chars as f32 / 14.0 * FPS * 1.5) as usize + 40;
    let cap = seconds.map(|s| (s * FPS).ceil() as usize + 8).unwrap_or(MAX_FRAMES);
    need.min(cap).clamp(16, MAX_FRAMES)
}

impl AudioEngine for VibeVoice {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn defaults(&self) -> Defaults {
        Defaults { seconds: 600.0, max_seconds: MAX_FRAMES as f32 / FPS, steps: 10, cfg: 1.3, rate: RATE }
    }
    fn speech(&self) -> Option<Speech> {
        Some(Speech { voices: Vec::new(), languages: vec!["en".into(), "zh".into()], instructions: false, design: false, clone: true, clone_needs_text: false,
                      clone_takes_text: false, direction: false, events: Vec::new(), requires: Some("a recording to clone (one a speaker for Speaker N: lines)".into()) })
    }
    fn options(&self) -> &'static [EngineOption] {
        OPTIONS
    }
    fn load_seconds(&self) -> f64 {
        self.load_s
    }
    fn generate(&self, req: &AudioRequest, progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<Audio> {
        let t0 = Instant::now();
        let d = self.defaults();
        if req.voice.as_deref().is_some_and(|v| !v.is_empty()) {
            return Err(Error("voice: VibeVoice has no built-in voices (send a recording, or a saved voice, a speaker)".into()));
        }
        let voices = VibeVoice::voices(req);
        if voices.is_empty() {
            return Err(Error("VibeVoice clones every voice: a recording (ref_audio, or a saved voice) for each speaker".into()));
        }
        if req.prompt.trim().is_empty() {
            return Err(Error("the text is empty".into()));
        }
        progress(Step { phase: "prompt", at: 0, of: 1, seconds: 0.0 })?;
        let p = self.prompt(&req.prompt, &voices)?;
        let max = budget(p.chars, req.seconds);
        let steps = req.steps.unwrap_or(d.steps).clamp(1, 50) as usize;
        let cfg = req.cfg.unwrap_or(d.cfg);
        let samples = self.generate_speech(&p, max, steps, cfg, req.seed, &Forced::default(), None, t0, progress)?;
        if samples.is_empty() {
            return Err(Error("the model said nothing (it ended before its first frame)".into()));
        }
        Ok(Audio { rate: RATE, channels: 1, samples })
    }
    fn report(&self) -> Vec<String> {
        vec![format!("{ARCH} on {} ({}, loaded {:.0} s ago)", self.ops.gpu.name, if self.int8 { "int8" } else { "half" }, self.loaded.elapsed().as_secs_f64())]
    }
}

/// A checkpoint's directory: the GGUF (the bf16 one when both are there), config.json, tokenizer.json (or
/// tokenizer/tokenizer.json)
pub fn files_in(dir: &Path) -> Result<ModelFiles> {
    let mut f = ModelFiles::new();
    let mut ggufs: Vec<PathBuf> = std::fs::read_dir(dir).map_err(|e| Error(format!("{}: {e}", dir.display())))?
        .flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "gguf")).collect();
    ggufs.sort_by_key(|p| !p.to_string_lossy().contains("bf16"));
    f.insert("model".into(), ggufs.into_iter().next().ok_or_else(|| Error(format!("{}: no .gguf", dir.display())))?);
    f.insert("config".into(), dir.join("config.json"));
    let tk = [dir.join("tokenizer.json"), dir.join("tokenizer/tokenizer.json")].into_iter().find(|p| p.exists())
        .ok_or_else(|| Error(format!("{}: no tokenizer.json", dir.display())))?;
    f.insert("tokenizer".into(), tk);
    Ok(f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_script_is_read_as_the_processor_reads_it() {
        let s = parse_script("Speaker 1: Hello there.\nnot a line\n  speaker 2 :  Hi!  \n");
        assert_eq!(s, vec![(0, " Hello there.".to_string()), (1, " Hi!".to_string())]);
        // a script that has a Speaker 0 keeps its numbers
        assert_eq!(parse_script("Speaker 0: a\nSpeaker 1: b")[1].0, 1);
        // plain text: one speaker, a line a paragraph
        assert_eq!(parse_script("One.\n\nTwo."), vec![(0, " One.".to_string()), (0, " Two.".to_string())]);
        // "Speakers 1:" is not a speaker line
        assert_eq!(parse_script("Speakers 1: x").len(), 1);
    }

    #[test]
    fn the_normaliser_meets_minus_25_db_and_never_clips() {
        let w: Vec<f32> = (0..24000).map(|i| (i as f32 * 0.01).sin() * 0.01).collect();
        let n = normalize(&w);
        let rms = (n.iter().map(|x| x * x).sum::<f32>() / n.len() as f32).sqrt();
        assert!((20.0 * rms.log10() - TARGET_DB).abs() < 0.01);
        let loud: Vec<f32> = (0..100).map(|i| if i == 50 { 1.0 } else { 0.0 }).collect();
        assert!(normalize(&loud).iter().all(|x| x.abs() <= 1.0));
    }

    #[test]
    fn the_solver_ends_on_its_data_prediction_and_its_timesteps_are_diffusers() {
        let s = Solver::new(10);
        assert_eq!(s.timesteps, vec![999, 899, 799, 699, 599, 500, 400, 300, 200, 100]);
        // a model that always predicts v for x0 = 0.5: the last step lands on 0.5 exactly
        let mut x = vec![1.0f32; 4];
        s.run(&mut x, &mut |i, x| {
            let sig = s.sigmas_at(i);
            let a = 1.0 / (sig * sig + 1.0).sqrt();
            let sg = sig * a;
            // x0 = a x - sg v  =>  v = (a x - 0.5) / sg
            Ok(x.iter().map(|xi| (a * xi - 0.5) / sg).collect())
        }).unwrap();
        assert!(x.iter().all(|v| (v - 0.5).abs() < 1e-4), "{x:?}");
    }

    #[test]
    fn the_budget_follows_the_text_and_the_asked_length() {
        assert_eq!(budget(140, None), (140.0 / 14.0 * 7.5 * 1.5) as usize + 40);
        assert_eq!(budget(14_000, Some(10.0)), 75 + 8);
        assert_eq!(budget(1, None), 40);
    }
}
