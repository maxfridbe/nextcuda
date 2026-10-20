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
pub mod codes;
pub mod dit;
mod ffi;
pub mod nn;
pub mod pth;
pub mod qwen3;
pub mod sample;
pub mod source;
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
use crate::sample::{dcw_double, dcw_strengths, denoised, euler, repaint_blend, repaint_inject, schedule, Rng, LATENT};
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
    EngineOption { name: "task", env: "", value: "text2music|cover|repaint", help: "what the song is made from: the words alone, or the request's recording performed anew (cover) or a span of it made again (repaint; an end past it extends it)", at: At::Request },
    EngineOption { name: "cover-strength", env: "", value: "0.5", help: "cover / repaint: the share of the steps on the recording's context (audio_cover_strength; the rest on the plain one)", at: At::Request },
    EngineOption { name: "cover-noise", env: "", value: "0", help: "cover: the flow started that close to the recording (cover_noise_strength; 0: from noise)", at: At::Request },
    EngineOption { name: "repaint-start", env: "", value: "0", help: "repaint: the span's start, seconds into the recording (before 0: silence put before it)", at: At::Request },
    EngineOption { name: "repaint-end", env: "", value: "-1", help: "repaint: the span's end, seconds (-1: the recording's end; past it: the song extended to it)", at: At::Request },
    EngineOption { name: "repaint-mode", env: "", value: "balanced|conservative|aggressive", help: "repaint: how the recording is kept around the span (its injection over the first steps, the edges' crossfades)", at: At::Request },
    EngineOption { name: "repaint-strength", env: "", value: "0.5", help: "repaint, balanced: 0 keeps the most of the recording, 1 the least", at: At::Request },
    EngineOption { name: "vae-sample", env: "NS_ACE_VAE_SAMPLE", value: "0|1", help: "the recording's VAE posterior sampled, as the reference (off: its mean)", at: At::Both },
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

/// What a song is made from (generate_music's task_type)
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Task {
    /// from the words alone
    Text2Music,
    /// the source performed anew: its codes (the tokenizer's hints) as the context for a share of the steps
    Cover,
    /// a span of the source made again (an end past the source: the song extended to it)
    Repaint,
}

impl Task {
    pub fn parse(s: &str) -> Result<Task> {
        Ok(match s.trim() {
            "" | "text2music" => Task::Text2Music,
            "cover" => Task::Cover,
            "repaint" | "extend" => Task::Repaint,
            "edit" | "lego" => return Err(Error(format!("task {s}: not in this build yet (text2music, cover, repaint)"))),
            t => return Err(Error(format!("task {t}: text2music, cover, repaint (extend), edit or lego"))),
        })
    }
}

/// A repaint's span and how the source is kept around it (`_resolve_repaint_config`)
#[derive(Clone, Debug)]
pub struct RepaintSpan {
    /// seconds in the padded source
    pub start: f64,
    pub end: f64,
    /// conservative | balanced | aggressive
    pub mode: String,
    pub strength: f64,
}

impl RepaintSpan {
    /// (injection ratio, latent crossfade frames, waveform crossfade seconds)
    pub fn config(&self) -> (f64, usize, f64) {
        let st = self.strength.clamp(0.0, 1.0);
        match self.mode.as_str() {
            "aggressive" => (0.0, 0, 0.0),
            "conservative" => (1.0, 25, 0.05),
            _ => {
                let inv = 1.0 - st;
                (inv, round_even(25.0 * inv) as usize, 0.05 * inv)
            }
        }
    }
}

/// Python's round(): halves to the even neighbour
pub fn round_even(x: f64) -> i64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && (r as i64) % 2 != 0 { (r - x.signum()) as i64 } else { r as i64 }
}

/// What a song is made from, once its texts are read: the request as the engine takes it
#[derive(Clone, Debug)]
pub struct Song {
    pub task: Task,
    pub caption: String,
    pub lyrics: String,
    pub language: String,
    pub metas: Metas,
    /// latent frames asked for (text2music; a recording's song is as long as the recording, or the repaint's end)
    pub frames: usize,
    pub steps: usize,
    pub shift: f32,
    pub dcw: bool,
    pub seed: u64,
    /// the DiT's instruction (the platform's engine leaves the handler's default for every task but lego)
    pub instruction: String,
    /// the source recording, stereo 48 kHz [2, n] channel-major, padded for a repaint past its ends
    pub source: Option<Vec<f32>>,
    /// audio_cover_strength: the share of the steps on the source's context (1 for text2music)
    pub cover_strength: f64,
    /// cover_noise_strength: the flow started that close to the source (0: from noise)
    pub cover_noise: f64,
    pub repaint: Option<RepaintSpan>,
    /// the source's VAE posterior sampled (the reference's); off: its mean
    pub vae_sample: bool,
}

/// What the stages made (the check reads them)
#[derive(Default)]
pub struct Trace {
    pub caption_ids: Vec<u32>,
    pub lyric_ids: Vec<u32>,
    pub te_out: Vec<f32>,
    pub lyric_emb: Vec<f32>,
    pub enc: Vec<f32>,
    pub target: Vec<f32>,
    pub ctx: Vec<f32>,
    pub velocities: Vec<Vec<f32>>,
    pub latents: Vec<f32>,
}

/// How the flow runs for a task: its context, the switch to the plain one, a repaint's span
pub struct Plan {
    /// latent frames of the flow
    pub frames: usize,
    /// latent frames of the sound returned
    pub out_frames: usize,
    /// context_latents [frames, 128]: the source latents (or a cover's hints) beside the chunk mask
    pub ctx: Vec<f32>,
    /// the non-cover branch (audio_cover_strength < 1): its conditions and context, from step int(n strength) on
    pub nc: Option<(Vec<f32>, Vec<f32>)>,
    pub strength: f64,
    /// a cover's start: the source latents, renoised to the schedule's t nearest 1 - cover_noise
    pub start: Option<(Vec<f32>, f64)>,
    /// a repaint: the frames made again (true), the source latents kept elsewhere, the injection ratio and the
    /// boundary's crossfade frames
    pub repaint: Option<(Vec<bool>, Vec<f32>, f64, usize)>,
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

/// the reference's shortest flow (max_latent_length = max(128, ...)): 5.12 s
pub const MIN_FRAMES: usize = 128;

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

    /// The request as a song: the platform's defaults where it says nothing (30 s, 8 steps, shift 3, DCW on; a task
    /// on a recording: cover strength 0.5, a balanced repaint at 0.5)
    pub fn song(&self, req: &AudioRequest) -> Result<Song> {
        let num = |n: &str, env: &str| -> Result<Option<f64>> {
            self.opt(req, n, env).map(|v| v.trim().parse::<f64>().map_err(|_| Error(format!("{n}: {v:?} is not a number")))).transpose()
        };
        let flag = |n: &str, env: &str, d: bool| self.opt(req, n, env).map_or(d, |v| !matches!(v.trim(), "0" | "false" | "off" | "no"));
        let task = Task::parse(&self.opt(req, "task", "").unwrap_or_default())?;
        let bpm = num("bpm", "")?.filter(|b| *b > 0.0).map(|b| b as u32);
        let lyrics = req.lyrics.clone().map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).unwrap_or_else(|| "[Instrumental]".into());
        let mut caption: String = req.prompt.trim().chars().take(512).collect();
        let (source, seconds, repaint) = match task {
            Task::Text2Music => {
                if req.source.is_some() {
                    return Err(Error("audio: a recording is for a task on it (task cover, repaint)".into()));
                }
                let seconds = req.seconds.unwrap_or(30.0);
                if !(1.0..=MAX_SECONDS).contains(&seconds) {
                    return Err(Error(format!("{seconds} s: from 1 to {MAX_SECONDS} s")));
                }
                (None, seconds as f64, None)
            }
            _ => {
                let a = req.source.as_ref().ok_or_else(|| Error(format!("task {task:?}: the recording to make it from (audio)")))?;
                let mut wav = source::stereo48(a)?;
                let n = wav.len() / 2;
                let dur = n as f64 / RATE as f64;
                if dur > MAX_SECONDS as f64 {
                    return Err(Error(format!("audio: {dur:.0} s, past the longest song ({MAX_SECONDS} s)")));
                }
                // the platform's engine: no description, the source's sound kept
                if caption.is_empty() {
                    caption = if task == Task::Cover {
                        "a faithful cover of the source recording, keeping its style, instrumentation, tempo and mood".into()
                    } else {
                        "continuing the source recording seamlessly, with the same style, instrumentation, tempo and mood".into()
                    };
                }
                let mut repaint = None;
                if task == Task::Repaint {
                    // prepare_padding_info: silence before a negative start, after an end past the source
                    let start = num("repaint-start", "")?.unwrap_or(0.0);
                    let end = num("repaint-end", "")?.filter(|e| *e >= 0.0);
                    let left = (-start).max(0.0);
                    let actual_end = end.unwrap_or(dur);
                    if actual_end > MAX_SECONDS as f64 {
                        return Err(Error(format!("repaint-end {actual_end} s: past the longest song ({MAX_SECONDS} s)")));
                    }
                    if actual_end <= start {
                        return Err(Error(format!("repaint {start} to {actual_end} s: an empty span")));
                    }
                    let right = (actual_end - dur).max(0.0);
                    let (lf, rf) = ((left * RATE as f64) as usize, (right * RATE as f64) as usize);
                    if lf > 0 || rf > 0 {
                        let mut p = Vec::with_capacity(2 * (n + lf + rf));
                        for ch in 0..2 {
                            p.extend(std::iter::repeat_n(0.0, lf));
                            p.extend_from_slice(&wav[ch * n..(ch + 1) * n]);
                            p.extend(std::iter::repeat_n(0.0, rf));
                        }
                        wav = p;
                    }
                    repaint = Some(RepaintSpan { start: start + left, end: end.unwrap_or(dur) + left,
                                                 mode: self.opt(req, "repaint-mode", "").unwrap_or_else(|| "balanced".into()),
                                                 strength: num("repaint-strength", "")?.unwrap_or(0.5) });
                }
                (Some(wav), dur, repaint)
            }
        };
        let strength = if task == Task::Text2Music { 1.0 } else { num("cover-strength", "")?.unwrap_or(0.5).clamp(0.0, 1.0) };
        let cover_noise = if task == Task::Text2Music { 0.0 } else { num("cover-noise", "")?.unwrap_or(0.0).clamp(0.0, 1.0) };
        Ok(Song {
            task,
            caption,
            lyrics: lyrics.chars().take(4096).collect(),
            language: self.opt(req, "language", "").or_else(|| req.language.clone().filter(|l| !l.trim().is_empty())).unwrap_or_else(|| "unknown".into()),
            metas: Metas { bpm, timesignature: self.opt(req, "time-signature", ""), keyscale: self.opt(req, "key", ""), duration: Some(seconds) },
            // create_target_wavs: round(seconds, 1) of samples, a frame each 1920
            frames: ((seconds * 10.0).round() / 10.0 * RATE as f64) as usize / vae::HOP,
            steps: req.steps.unwrap_or(8) as usize,
            shift: num("shift", "NS_ACE_SHIFT")?.unwrap_or(3.0) as f32,
            dcw: flag("dcw", "NS_ACE_DCW", self.dit.cfg.turbo),
            seed: req.seed,
            instruction: instruction("text2music", None),
            source,
            cover_strength: strength,
            cover_noise,
            repaint,
            vae_sample: flag("vae-sample", "NS_ACE_VAE_SAMPLE", true),
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

    /// The packed conditions of the song under `instr` (the caption prompt's encoding, the lyrics', the timbre of the
    /// silence latent: no reference) - trace holds the ids and the encodings
    pub fn conditions_as(&self, s: &mut Scratch, song: &Song, instr: &str, trace: &mut Trace) -> Result<Vec<f32>> {
        let cap = caption_prompt(instr, &song.caption, &song.metas.text());
        trace.caption_ids = self.ids(&cap, CAPTION_TOKENS);
        trace.lyric_ids = self.ids(&lyrics_text(&song.lyrics, &song.language), LYRIC_TOKENS);
        trace.te_out = self.te.encode(&self.ops, s, &trace.caption_ids)?;
        trace.lyric_emb = self.te.embed(&trace.lyric_ids)?;
        let refer = self.dit.silence[..750 * LATENT].to_vec();
        self.dit.conditions(&self.ops, s, &trace.te_out, &trace.lyric_emb, &refer)
    }

    /// The song's conditions under its own instruction
    pub fn conditions(&self, s: &mut Scratch, song: &Song, trace: &mut Trace) -> Result<Vec<f32>> {
        self.conditions_as(s, song, &song.instruction, trace)
    }

    /// The source's latents [frames, 64]: its VAE encoding (the posterior sampled from the seed's stream unless
    /// `vae_sample` is off), the silence latent for a silent recording
    pub fn target(&self, s: &mut Scratch, song: &Song, wav: &[f32], progress: &mut dyn FnMut(u32, u32) -> Result<()>) -> Result<Vec<f32>> {
        if wav.iter().all(|x| x.abs() < 1e-6) {
            return Ok(self.dit.silence(wav.len() / 2 / vae::HOP));
        }
        let mut rng = Rng::new(song.seed ^ 0x5EED_0F_5A_E1);
        let mut draw = || rng.normal();
        let noise: Option<&mut dyn FnMut() -> f32> = if song.vae_sample { Some(&mut draw) } else { None };
        self.vae.encode(&self.ops, s, wav, noise, &mut |a, b| progress(a as u32, b as u32))
    }

    /// The flow's plan for the song from its source's latents (`target`, None for text2music) and the non-cover
    /// conditions (`nc_enc`, when the cover strength is under 1): _prepare_batch's masks and source latents,
    /// prepare_condition's context
    pub fn plan(&self, s: &mut Scratch, song: &Song, target: Option<Vec<f32>>, nc_enc: Option<Vec<f32>>, trace: &mut Trace) -> Result<Plan> {
        let have = target.as_ref().map_or(song.frames, |t| t.len() / LATENT);
        let t = have.max(MIN_FRAMES);
        let out_frames = if song.task == Task::Text2Music { song.frames } else { have };
        let tiled = self.dit.silence(t);
        // the source padded to the flow's length with the silence latent's head
        let target = target.map(|mut x| {
            x.extend_from_slice(&self.dit.silence(t - have));
            x
        });
        let (mut chunk, mut src, mut repaint) = (vec![1f32; t], tiled.clone(), None);
        match song.task {
            Task::Text2Music => {}
            Task::Cover => src = target.clone().unwrap_or_default(),
            Task::Repaint => {
                let r = song.repaint.as_ref().ok_or_else(|| Error("repaint: no span".into()))?;
                let fr = |sec: f64| ((sec * RATE as f64) / vae::HOP as f64).floor() as i64;
                let s0 = fr(r.start).clamp(0, t as i64 - 1) as usize;
                let s1 = (fr(r.end).min(t as i64)).max(s0 as i64 + 1) as usize;
                // chunk_mask_mode "explicit": the span alone; the source there is the silence latent's
                chunk = (0..t).map(|i| if (s0..s1).contains(&i) { 1.0 } else { 0.0 }).collect();
                let clean = target.clone().unwrap_or_default();
                src = clean.clone();
                src[s0 * LATENT..s1 * LATENT].copy_from_slice(&tiled[s0 * LATENT..s1 * LATENT]);
                let (ratio, cf, _) = r.config();
                repaint = Some(((0..t).map(|i| (s0..s1).contains(&i)).collect(), clean, ratio, cf));
            }
        }
        let with = |a: &[f32]| -> Vec<f32> {
            let mut c = vec![0f32; t * 2 * LATENT];
            for f in 0..t {
                c[f * 2 * LATENT..f * 2 * LATENT + LATENT].copy_from_slice(&a[f * LATENT..(f + 1) * LATENT]);
                c[f * 2 * LATENT + LATENT..(f + 1) * 2 * LATENT].fill(chunk[f]);
            }
            c
        };
        // a cover: the source through the audio codes and back (the LM hints), cropped to the song
        let ctx = if song.task == Task::Cover {
            let (_, q) = self.dit.codes.tokenize(&self.ops, s, &src, &self.dit.silence)?;
            let mut hints = self.dit.codes.detokenize(&self.ops, s, &q)?;
            hints.truncate(t * LATENT);
            with(&hints)
        } else {
            with(&src)
        };
        trace.target = target.clone().unwrap_or_default();
        trace.ctx = ctx.clone();
        let nc = match nc_enc {
            Some(e) if song.cover_strength < 1.0 => Some((e, with(&tiled))),
            _ => None,
        };
        let start = if song.cover_noise > 0.0 && song.task == Task::Cover { Some((src, song.cover_noise)) } else { None };
        Ok(Plan { frames: t, out_frames, ctx, nc, strength: song.cover_strength, start, repaint })
    }

    /// The flow from `noise` to the latents [frames, 64] under the packed conditions and the plan: generate_audio's
    /// loop (Euler, DCW, the switch to the non-cover branch, a repaint's injection and boundary blend)
    pub fn sample(&self, s: &mut Scratch, song: &Song, plan: &Plan, enc: &[f32], noise: Vec<f32>, trace: &mut Trace,
                  progress: &mut dyn FnMut(u32, u32) -> Result<()>) -> Result<Vec<f32>> {
        let frames = plan.frames;
        if noise.len() != frames * LATENT {
            return Err(Error(format!("noise of {} values for {frames} frames", noise.len())));
        }
        let mut ts = schedule(song.steps, song.shift);
        let mut x = noise.clone();
        if let Some((src, cn)) = &plan.start {
            // the schedule's t nearest 1 - cover_noise: the source renoised to it, the steps from there
            let level = 1.0 - cn;
            let (i, t) = ts.iter().enumerate().fold((0, ts[0]), |b, (i, t)| if ((*t as f64) - level).abs() < ((b.1 as f64) - level).abs() { (i, *t) } else { b });
            x = noise.iter().zip(src).map(|(n, c)| t * n + (1.0 - t) * c).collect();
            ts = ts[i..].to_vec();
        }
        let n = ts.len();
        let cover_steps = (n as f64 * plan.strength) as usize;
        let inject = plan.repaint.as_ref().map_or(0, |r| round_even(r.2 * n as f64).max(0) as usize);
        let mut cond = self.dit.cond(&self.ops, s, enc)?;
        let mut ctx = &plan.ctx;
        let mut switched = false;
        for (i, t) in ts.iter().enumerate() {
            progress(i as u32, n as u32)?;
            if i >= cover_steps && !switched {
                switched = true;
                if let Some((e, c)) = &plan.nc {
                    cond = self.dit.cond(&self.ops, s, e)?;
                    ctx = c;
                }
            }
            let v = self.dit.velocity(&self.ops, s, &x, ctx, *t, &cond)?;
            if i + 1 == n {
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
            if let Some((mask, clean, _, _)) = &plan.repaint {
                if i < inject {
                    repaint_inject(&mut x, clean, mask, ts[i + 1], &noise);
                }
            }
            trace.velocities.push(v);
        }
        if let Some((mask, clean, _, cf)) = &plan.repaint {
            if *cf > 0 {
                repaint_blend(&mut x, clean, mask, *cf);
            }
        }
        progress(n as u32, n as u32)?;
        Ok(x)
    }

    /// A song's latents to its sound: the VAE (a peak over 1 brought to 1), a repaint's source spliced back outside its
    /// span, then the peak to -1 dB (the reference's normalization); `frames` of it kept
    pub fn sound(&self, s: &mut Scratch, song: &Song, latents: &[f32], frames: usize, normalize: bool,
                 progress: &mut dyn FnMut(u32, u32) -> Result<()>) -> Result<Audio> {
        let mut wav = self.vae.decode(&self.ops, s, latents, &mut |a, b| progress(a as u32, b as u32))?;
        let n = wav.len() / 2;
        let peak = wav.iter().fold(0f32, |m, x| m.max(x.abs()));
        if peak > 1.0 {
            wav.iter_mut().for_each(|x| *x /= peak);
        }
        if let (Some(r), Some(src)) = (&song.repaint, &song.source) {
            let (_, _, wav_cf) = r.config();
            if r.mode != "aggressive" {
                splice(&mut wav, src, r.start, r.end, (wav_cf * RATE as f64) as usize);
            }
        }
        let keep = (frames * vae::HOP).min(n);
        let mut samples = Vec::with_capacity(2 * keep);
        for i in 0..keep {
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

/// `apply_repaint_waveform_splice`: outside [start, end) seconds the source's own samples, a linear crossfade of `cf`
/// samples at each edge (pred and src [2, n] channel-major; the shorter's length is spliced)
pub fn splice(pred: &mut [f32], src: &[f32], start: f64, end: f64, cf: usize) {
    let (np, ns) = (pred.len() / 2, src.len() / 2);
    let m = np.min(ns);
    let s0 = ((start * RATE as f64) as usize).min(m);
    let s1 = ((end * RATE as f64) as usize).min(m).max(s0);
    if s0 == 0 && s1 >= m {
        return;
    }
    let mut mask = vec![0f32; m];
    mask[s0..s1].fill(1.0);
    if cf > 0 {
        let fs = s0.saturating_sub(cf);
        let n = s0 - fs;
        for (j, v) in mask[fs..s0].iter_mut().enumerate() {
            *v = (j + 1) as f32 / (n + 1) as f32;
        }
        let fe = (s1 + cf).min(m);
        let n = fe - s1;
        for (j, v) in mask[s1..fe].iter_mut().enumerate() {
            *v = 1.0 - (j + 1) as f32 / (n + 1) as f32;
        }
    }
    for ch in 0..2 {
        for (i, w) in mask.iter().enumerate() {
            let p = &mut pred[ch * np + i];
            *p = w * *p + (1.0 - w) * src[ch * ns + i];
        }
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

    fn sources(&self) -> bool {
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
        let el = || t0.elapsed().as_secs_f64();
        let target = match &song.source {
            Some(wav) => Some(self.target(&mut s, &song, wav, &mut |a, b| progress(Step { phase: "encode", at: a, of: b, seconds: el() }))?),
            None => None,
        };
        progress(Step { phase: "prompt", at: 0, of: 1, seconds: el() })?;
        let enc = self.conditions(&mut s, &song, &mut trace)?;
        // the non-cover branch's conditions: the default instruction's (the same as the song's unless it has its own)
        let nc = if song.cover_strength < 1.0 {
            let d = instruction("text2music", None);
            Some(if d == song.instruction { enc.clone() } else { self.conditions_as(&mut s, &song, &d, &mut Trace::default())? })
        } else {
            None
        };
        progress(Step { phase: "prompt", at: 1, of: 1, seconds: el() })?;
        let plan = self.plan(&mut s, &song, target, nc, &mut trace)?;
        let noise = Rng::new(song.seed).normals(plan.frames * LATENT);
        let lat = self.sample(&mut s, &song, &plan, &enc, noise, &mut trace, &mut |a, b| progress(Step { phase: "flow", at: a, of: b, seconds: el() }))?;
        let out = self.sound(&mut s, &song, &lat, plan.out_frames, normalize, &mut |a, b| progress(Step { phase: "decode", at: a, of: b, seconds: el() }));
        s.clear();
        out
    }

    fn report(&self) -> Vec<String> {
        vec![format!("{ARCH} on {} ({}, loaded {:.0} s)", self.ops.gpu.name, if self.dit.cfg.turbo { "turbo" } else { "base" }, self.load_s)]
    }
}
