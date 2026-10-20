//! ACE-Step 1.5 on CUDA: songs from a description and lyrics in 50 languages, a 2B flow DiT (turbo: 8 steps, CFG
//! distilled) over the latents of the Oobleck VAE (48 kHz stereo, 25 latent frames a second), conditioned on the
//! caption's Qwen3-Embedding states, the lyrics' embeddings through their own encoder and a timbre reference.
//!
//! ```text
//!   caption, metas, lyrics -> text.rs: the handler's prompts -> qwen3.rs: the caption's last hidden state, the lyrics'
//!      token embeddings -> dit.rs: the condition encoder (lyrics, timbre, caption, packed), each layer's cross keys
//!   noise -> sample.rs's schedule (8 steps, shift 3): dit.rs's velocity, Euler, DCW -> the latents
//!   -> vae.rs: 48 kHz stereo (in the reference's windows), its peak to -1 dB
//! ```
//!
//! The weights are the checkpoint's own files (ACE-Step/Ace-Step1.5: acestep-v15-turbo, Qwen3-Embedding-0.6B, vae):
//! the DiT and the text encoder in bfloat16 on the GPU, the VAE in float32 (~6 GB in all). `check` compares the stages
//! with the reference's dumps (reference/acestep/ref.py: ACE-Step's own pipeline, as the platform runs it).

pub mod check;
pub mod dit;
mod ffi;
pub mod nn;
pub mod pth;
pub mod qwen3;
pub mod sample;
pub mod text;
pub mod vae;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use nextcuda_audio::{At, Audio, AudioEngine, AudioKind, AudioRequest, Defaults, EngineOption, Error, LoadOptions, ModelFiles, Result, Step};
use nextcuda_core::Gpu;
use nextcuda_tok::Tokenizer;

use crate::dit::{Config as DitConfig, Dit};
use crate::nn::{Ops, Scratch};
use crate::qwen3::{Config as TeConfig, Qwen3};
use crate::sample::{dcw_double, dcw_strengths, denoised, euler, schedule, Rng, LATENT};
use crate::text::{caption_prompt, instruction, lyrics_text, Metas, CAPTION_TOKENS, LYRIC_TOKENS};
use crate::vae::{Vae, RATE};

pub const ARCH: &str = "acestep";
pub const ROLES: &[&str] = &["dit", "dit_config", "silence", "text_encoder", "text_config", "tokenizer", "vae", "vae_config"];
/// latent frames a second
pub const FPS: f64 = 25.0;
/// the longest song (the silence latent's 15000 frames: 600 s)
pub const MAX_SECONDS: f32 = 600.0;

pub const OPTIONS: &[EngineOption] = &[
    EngineOption { name: "bpm", env: "", value: "120", help: "the tempo (beats a minute; the prompt's metas: N/A when not given)", at: At::Request },
    EngineOption { name: "key", env: "", value: "\"C major\"", help: "the key and scale (\"A minor\", \"F# major\")", at: At::Request },
    EngineOption { name: "time-signature", env: "", value: "4", help: "beats a bar (2, 3, 4, 6)", at: At::Request },
    EngineOption { name: "language", env: "", value: "en", help: "the lyrics' language (a code: en, zh, ja ...; unknown: none said)", at: At::Request },
    EngineOption { name: "shift", env: "NS_ACE_SHIFT", value: "3.0", help: "the schedule's shift (t' = s t / (1 + (s - 1) t); 3: the platform's)", at: At::Both },
    EngineOption { name: "dcw", env: "NS_ACE_DCW", value: "0|1", help: "the wavelet-domain correction of each step (DCW; on, as the reference does for turbo)", at: At::Both },
    EngineOption { name: "normalize", env: "NS_ACE_NORMALIZE", value: "0|1", help: "the song's peak to -1 dB (on, as the reference)", at: At::Both },
];

pub fn kind() -> AudioKind {
    AudioKind { archs: &[ARCH], name: "ACE-Step 1.5 (songs in 50 languages from a description and lyrics: a 2B turbo flow DiT, 48 kHz stereo)", roles: ROLES, load,
                options: OPTIONS }
}

fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn AudioEngine>> {
    let gpu = gpus.first().ok_or_else(|| Error("no GPU".into()))?;
    Ok(Box::new(AceStep::load(files, gpu, o, log)?))
}

fn role(files: &ModelFiles, r: &str) -> Result<PathBuf> {
    files.get(r).cloned().ok_or_else(|| Error(format!("{ARCH}: no {r} file")))
}

/// The checkpoint's directory as downloaded (ACE-Step/Ace-Step1.5): its files by role
pub fn files_in(dir: &Path) -> Result<ModelFiles> {
    let mut f = ModelFiles::new();
    for (r, name) in [("dit", "acestep-v15-turbo/model.safetensors"), ("dit_config", "acestep-v15-turbo/config.json"),
                      ("silence", "acestep-v15-turbo/silence_latent.pt"), ("text_encoder", "Qwen3-Embedding-0.6B/model.safetensors"),
                      ("text_config", "Qwen3-Embedding-0.6B/config.json"), ("tokenizer", "Qwen3-Embedding-0.6B/tokenizer.json"),
                      ("vae", "vae/diffusion_pytorch_model.safetensors"), ("vae_config", "vae/config.json")] {
        let p = dir.join(name);
        if !p.is_file() {
            return Err(Error(format!("{}: no {name} (an ACE-Step 1.5 checkpoint's directory)", dir.display())));
        }
        f.insert(r.into(), p);
    }
    Ok(f)
}

/// What a song is made from, once its texts are read: the request as the engine takes it
#[derive(Clone, Debug)]
pub struct Song {
    pub caption: String,
    pub lyrics: String,
    pub language: String,
    pub metas: Metas,
    pub frames: usize,
    pub steps: usize,
    pub shift: f32,
    pub dcw: bool,
    pub seed: u64,
}

/// What the stages made (the check reads them)
#[derive(Default)]
pub struct Trace {
    pub caption_ids: Vec<u32>,
    pub lyric_ids: Vec<u32>,
    pub te_out: Vec<f32>,
    pub lyric_emb: Vec<f32>,
    pub enc: Vec<f32>,
    pub velocities: Vec<Vec<f32>>,
    pub latents: Vec<f32>,
}

pub struct AceStep {
    pub ops: Ops,
    pub te: Qwen3,
    pub dit: Dit,
    pub vae: Vae,
    pub tok: Tokenizer,
    settings: LoadOptions,
    work: Mutex<Scratch>,
    load_s: f64,
}

fn vae_strides(path: &Path) -> Result<Vec<usize>> {
    let v: serde_json::Value = std::fs::read(path).ok().and_then(|b| serde_json::from_slice(&b).ok())
        .ok_or_else(|| Error(format!("{}: not JSON", path.display())))?;
    let s: Vec<usize> = v["downsampling_ratios"].as_array().map(|a| a.iter().filter_map(|x| x.as_u64().map(|y| y as usize)).collect()).unwrap_or_default();
    if s.iter().product::<usize>() != vae::HOP || v["sampling_rate"].as_u64() != Some(RATE as u64) {
        return Err(Error(format!("{}: strides {s:?} at {} Hz (an Oobleck of 1920 samples a frame at 48 kHz expected)", path.display(), v["sampling_rate"])));
    }
    Ok(s)
}

impl AceStep {
    pub fn load(files: &ModelFiles, gpu: &Arc<Gpu>, o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<AceStep> {
        let t0 = Instant::now();
        let ops = Ops::new(gpu)?;
        let tcfg = TeConfig::read(&role(files, "text_config")?)?;
        let te = Qwen3::load(&role(files, "text_encoder")?, tcfg, "", gpu)?;
        let dcfg = DitConfig::read(&role(files, "dit_config")?)?;
        let sil = pth::load_tensor(&role(files, "silence")?)?;
        // [1, 64, frames] -> frames-major
        let n = *sil.shape.last().unwrap_or(&0);
        if sil.shape.len() != 3 || sil.shape[1] != LATENT {
            return Err(Error(format!("silence_latent: shape {:?} ([1, 64, frames] expected)", sil.shape)));
        }
        let mut silence = vec![0f32; n * LATENT];
        for c in 0..LATENT {
            for f in 0..n {
                silence[f * LATENT + c] = sil.data[c * n + f];
            }
        }
        let dit = Dit::load(&role(files, "dit")?, dcfg, silence, gpu)?;
        let vae = Vae::load(&role(files, "vae")?, &vae_strides(&role(files, "vae_config")?)?, gpu, true)?;
        let tok = Tokenizer::from_hf_json(&role(files, "tokenizer")?).map_err(Error)?;
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        log(format!("acestep: the DiT ({} layers, {}) {:.2} GiB, the text encoder {:.2} GiB, the VAE {:.2} GiB on {} - loaded in {:.1} s",
                    dit.cfg.layers, if dit.cfg.turbo { "turbo" } else { "base" }, gib(dit.bytes), gib(te.bytes), gib(vae.bytes), gpu.name,
                    t0.elapsed().as_secs_f64()));
        Ok(AceStep { ops, te, dit, vae, tok, settings: o.clone(), work: Mutex::new(Scratch::default()), load_s: t0.elapsed().as_secs_f64() })
    }

    /// An option: the request's, else the load's (its settings, the environment)
    fn opt(&self, req: &AudioRequest, name: &str, env: &str) -> Option<String> {
        req.extra.get(name).cloned().or_else(|| if env.is_empty() { None } else { self.settings.setting(env) }).filter(|v| !v.trim().is_empty())
    }

    /// The request as a song: the platform's defaults where it says nothing (30 s, 8 steps, shift 3, DCW on)
    pub fn song(&self, req: &AudioRequest) -> Result<Song> {
        let seconds = req.seconds.unwrap_or(30.0);
        if !(1.0..=MAX_SECONDS).contains(&seconds) {
            return Err(Error(format!("{seconds} s: from 1 to {MAX_SECONDS} s")));
        }
        let num = |n: &str, env: &str| -> Result<Option<f64>> {
            self.opt(req, n, env).map(|v| v.trim().parse::<f64>().map_err(|_| Error(format!("{n}: {v:?} is not a number")))).transpose()
        };
        let flag = |n: &str, env: &str, d: bool| self.opt(req, n, env).map_or(d, |v| !matches!(v.trim(), "0" | "false" | "off" | "no"));
        let bpm = num("bpm", "")?.filter(|b| *b > 0.0).map(|b| b as u32);
        let lyrics = req.lyrics.clone().map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).unwrap_or_else(|| "[Instrumental]".into());
        Ok(Song {
            caption: req.prompt.chars().take(512).collect(),
            lyrics: lyrics.chars().take(4096).collect(),
            language: self.opt(req, "language", "").unwrap_or_else(|| "unknown".into()),
            metas: Metas { bpm, timesignature: self.opt(req, "time-signature", ""), keyscale: self.opt(req, "key", ""), duration: Some(seconds as f64) },
            frames: (seconds as f64 * FPS).round() as usize,
            steps: req.steps.unwrap_or(8) as usize,
            shift: num("shift", "NS_ACE_SHIFT")?.unwrap_or(3.0) as f32,
            dcw: flag("dcw", "NS_ACE_DCW", self.dit.cfg.turbo),
            seed: req.seed,
        })
    }

    /// tokens of `text` as the handler's tokenizer call makes them: cut at `max` with the post-processor's
    /// <|endoftext|> after (Qwen3-Embedding's TemplateProcessing "$A <|endoftext|>", counted in the limit)
    fn ids(&self, text: &str, max: usize) -> Vec<u32> {
        let mut v = self.tok.encode(text);
        let eos = self.tok.id("<|endoftext|>");
        v.truncate(max - eos.is_some() as usize);
        v.extend(eos);
        v
    }

    /// The song's conditions: (caption ids, lyric ids, the caption's encoding, the lyrics' embeddings, the packed
    /// conditions)
    pub fn conditions(&self, s: &mut Scratch, song: &Song, trace: &mut Trace) -> Result<Vec<f32>> {
        let cap = caption_prompt(&instruction("text2music", None), &song.caption, &song.metas.text());
        trace.caption_ids = self.ids(&cap, CAPTION_TOKENS);
        trace.lyric_ids = self.ids(&lyrics_text(&song.lyrics, &song.language), LYRIC_TOKENS);
        trace.te_out = self.te.encode(&self.ops, s, &trace.caption_ids)?;
        trace.lyric_emb = self.te.embed(&trace.lyric_ids)?;
        let refer = self.dit.silence[..750 * LATENT].to_vec();
        self.dit.conditions(&self.ops, s, &trace.te_out, &trace.lyric_emb, &refer)
    }

    /// The flow from `noise` to the latents [frames, 64] under the packed conditions (text2music: the context is the
    /// silence latent beside a chunk mask of ones)
    pub fn sample(&self, s: &mut Scratch, song: &Song, enc: &[f32], noise: Vec<f32>, trace: &mut Trace,
                  progress: &mut dyn FnMut(u32, u32) -> Result<()>) -> Result<Vec<f32>> {
        let frames = song.frames;
        if frames > self.dit.silence.len() / LATENT {
            return Err(Error(format!("{frames} frames: past the silence latent's {}", self.dit.silence.len() / LATENT)));
        }
        let mut ctx = vec![1f32; frames * 2 * LATENT];
        for f in 0..frames {
            ctx[f * 2 * LATENT..f * 2 * LATENT + LATENT].copy_from_slice(&self.dit.silence[f * LATENT..(f + 1) * LATENT]);
        }
        let cond = self.dit.cond(&self.ops, s, enc)?;
        let ts = schedule(song.steps, song.shift);
        let mut x = noise;
        for (i, t) in ts.iter().enumerate() {
            progress(i as u32, ts.len() as u32)?;
            let v = self.dit.velocity(&self.ops, s, &x, &ctx, *t, &cond)?;
            if i + 1 == ts.len() {
                // the last step: the clean estimate itself (DCW then has nothing to push)
                x = denoised(&x, &v, *t);
                trace.velocities.push(v);
                break;
            }
            let y = denoised(&x, &v, *t);
            euler(&mut x, &v, t - ts[i + 1]);
            if song.dcw {
                let (lo, hi) = dcw_strengths(*t, 0.05, 0.02);
                dcw_double(&mut x, &y, lo, hi);
            }
            trace.velocities.push(v);
        }
        progress(ts.len() as u32, ts.len() as u32)?;
        Ok(x)
    }

    /// A song's latents to its sound: the VAE, then the peak to -1 dB (the reference's normalization)
    pub fn sound(&self, s: &mut Scratch, latents: &[f32], normalize: bool, progress: &mut dyn FnMut(u32, u32) -> Result<()>) -> Result<Audio> {
        let wav = self.vae.decode(&self.ops, s, latents, &mut |a, b| progress(a as u32, b as u32))?;
        let n = wav.len() / 2;
        let mut samples = Vec::with_capacity(wav.len());
        for i in 0..n {
            samples.push(wav[i]);
            samples.push(wav[n + i]);
        }
        if normalize {
            let peak = samples.iter().fold(0f32, |m, x| m.max(x.abs()));
            if peak > 0.0 {
                let g = 10f32.powf(-1.0 / 20.0) / peak;
                samples.iter_mut().for_each(|x| *x *= g);
            }
        }
        Ok(Audio { rate: RATE, channels: 2, samples })
    }
}

impl AudioEngine for AceStep {
    fn arch(&self) -> &'static str {
        ARCH
    }

    fn defaults(&self) -> Defaults {
        Defaults { seconds: 30.0, max_seconds: MAX_SECONDS, steps: 8, cfg: 1.0, rate: RATE }
    }

    fn lyrics(&self) -> bool {
        true
    }

    fn options(&self) -> &'static [EngineOption] {
        OPTIONS
    }

    fn load_seconds(&self) -> f64 {
        self.load_s
    }

    fn generate(&self, req: &AudioRequest, progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<Audio> {
        let t0 = Instant::now();
        let song = self.song(req)?;
        let normalize = self.opt(req, "normalize", "NS_ACE_NORMALIZE").is_none_or(|v| !matches!(v.trim(), "0" | "false" | "off"));
        let mut s = self.work.lock().unwrap_or_else(|p| p.into_inner());
        let mut trace = Trace::default();
        progress(Step { phase: "prompt", at: 0, of: 1, seconds: 0.0 })?;
        let enc = self.conditions(&mut s, &song, &mut trace)?;
        progress(Step { phase: "prompt", at: 1, of: 1, seconds: t0.elapsed().as_secs_f64() })?;
        let noise = Rng::new(song.seed).normals(song.frames * LATENT);
        let lat = self.sample(&mut s, &song, &enc, noise, &mut trace, &mut |a, b| progress(Step { phase: "flow", at: a, of: b, seconds: t0.elapsed().as_secs_f64() }))?;
        let out = self.sound(&mut s, &lat, normalize, &mut |a, b| progress(Step { phase: "decode", at: a, of: b, seconds: t0.elapsed().as_secs_f64() }));
        s.clear();
        out
    }

    fn report(&self) -> Vec<String> {
        vec![format!("{ARCH} on {} ({}, loaded {:.0} s)", self.ops.gpu.name, if self.dit.cfg.turbo { "turbo" } else { "base" }, self.load_s)]
    }
}
