//! LTX-2.5 (Lightricks): text and pictures to video with its own sound, in one pass - a 22B audio-video DiT (48
//! blocks: video 4096 wide, audio 2048, each attending to the other), Gemma-4 12B as its text encoder through two
//! connectors, a video VAE (x32 spatial, x8 temporal), an audio VAE and a vocoder to 48 kHz, and a latent upsampler
//! (x2) for the shot upscaler. Rust and CUDA: the shared diffusion kernels (`nsd`) and this engine's few
//! (`kernels/video/ltx2`), the DiT's blocks split over both cards and resident, the text stage loaded for a clip and
//! released, nothing streamed.
//!
//! The contract (`nextcuda_video::VideoEngine`): `generate` for a clip from a request, and jobs (`run_job`):
//! `generate`, and `check` (the stages against reference/ltx2's dumps).

pub mod audio;
pub mod cache;
pub mod check;
pub mod conn;
pub mod dit;
pub mod ffi;
pub mod gen;
pub mod media;
pub mod quant;
pub mod sample;
pub mod te;
pub mod vae;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use nextcuda_core::{Error, Result};
use nextcuda_tok::Tokenizer;
use nextcuda_video::{At, Clip, Defaults, EngineOption, Gpu, JobCtl, LoadOptions, ModelFiles, Progress, Sampler, Schedule, VideoEngine, VideoKind,
                     VideoRequest};
use serde_json::{json, Value};

pub const ARCH: &str = "ltx2";

/// The files it takes, by role: the dev DiT (GGUF), the distilled one (`fast`), the checkpoint's text encoder (any of
/// its shards; its directory is read), connectors, tokenizer.json, the VAEs, vocoder and latent upsampler
pub const ROLES: &[&str] = &["denoiser", "denoiser-fast", "text-encoder", "connectors", "tokenizer", "vae", "audio-vae", "vocoder", "upscaler"];

pub const OPTIONS: &[EngineOption] = &[
    EngineOption { name: "threads", env: "LTX2_THREADS", value: "N", help: "threads quantizing the text encoder the first time (default 8)", at: At::Load },
    EngineOption { name: "fast", env: "", value: "0|1", help: "the distilled DiT: 8 steps, no guidance (default 0: the dev DiT, 30 steps, guided)", at: At::Request },
    EngineOption { name: "first_frame", env: "", value: "PNG", help: "image to video: the clip starts on this picture", at: At::Request },
];

pub fn kind() -> VideoKind {
    VideoKind { archs: &[ARCH], name: "LTX-2.5 (22B audio-video DiT, Gemma-4 12B text encoder, video + audio VAEs, latent upsampler)", roles: ROLES, load,
                options: OPTIONS }
}

fn ve(e: Error) -> nextcuda_video::Error {
    nextcuda_video::Error(e.0)
}

fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> nextcuda_video::Result<Box<dyn VideoEngine>> {
    Ok(Box::new(Ltx2::load(files, gpus, o, log).map_err(ve)?))
}

pub struct Ltx2 {
    files: ModelFiles,
    pub gpus: Vec<Arc<Gpu>>,
    /// the checkpoint's directory (text_encoder/, connectors/ ... under it)
    components: PathBuf,
    text_cache: PathBuf,
    tok: Tokenizer,
    load_s: f64,
    lock: Mutex<()>,
    /// the distilled DiT unless a request says (a `-fast` registry entry: LTX2_FAST=1)
    pub fast_default: bool,
    /// the resident denoiser (dev or distilled), loaded on first use
    pub dit: Mutex<Option<(bool, dit::Dit)>>,
}

fn role<'a>(files: &'a ModelFiles, r: &str) -> Result<&'a PathBuf> {
    files.get(r).ok_or_else(|| Error(format!("{ARCH}: no {r} file")))
}

impl Ltx2 {
    pub fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Ltx2> {
        let t0 = std::time::Instant::now();
        nextcuda_core::use_kind("video");
        if gpus.is_empty() {
            return Err(Error(format!("{ARCH}: no GPU")));
        }
        let te_dir = role(files, "text-encoder")?.parent().ok_or("the text encoder's directory")?;
        let components = te_dir.parent().ok_or("the checkpoint's directory")?.to_path_buf();
        let threads = o.setting("LTX2_THREADS").and_then(|v| v.parse().ok()).unwrap_or(8);
        // LTX2_TE_Q8=1: the text encoder in Q8_0 (a check's form; it does not fit beside the DiT)
        let q8 = o.setting("LTX2_TE_Q8").is_some_and(|v| v == "1");
        let text_cache = cache::ensure(&components, threads, q8, log)?;
        let tok_path = files.get("tokenizer").cloned().unwrap_or_else(|| components.join("tokenizer/tokenizer.json"));
        let tok = Tokenizer::from_hf_json(&tok_path).map_err(Error)?;
        Ok(Ltx2 { files: files.clone(), gpus: gpus.to_vec(), components, text_cache, tok, load_s: t0.elapsed().as_secs_f64(), lock: Mutex::new(()), dit: Mutex::new(None),
                  fast_default: o.setting("LTX2_FAST").is_some_and(|v| v == "1") })
    }

    /// The card with the most free memory now
    pub fn freest(&self) -> &Arc<Gpu> {
        self.gpus.iter().max_by_key(|g| g.memory().map_or(0, |(t, f)| f.unwrap_or(t))).unwrap()
    }

    /// A role's file
    pub fn file(&self, r: &str) -> Result<&PathBuf> {
        role(&self.files, r)
    }

    /// The prompt's tokens: Gemma's tokenizer, no BOS (the pipeline's call adds none), at most 1,024
    pub fn tokens(&self, prompt: &str) -> Vec<u32> {
        let mut ids = self.tok.encode(prompt.trim());
        ids.truncate(conn::SEQ);
        ids
    }

    pub fn text_encoder(&self, log: &mut dyn FnMut(String)) -> Result<te::TextEncoder> {
        te::TextEncoder::load(&self.text_cache, &self.gpus, log)
    }

    /// The video and audio conditions of a prompt's states (the connectors loaded for it: video on the first card,
    /// audio on the last)
    pub fn connectors(&self, s: &te::States, log: &mut dyn FnMut(String)) -> Result<(Vec<f32>, Vec<f32>)> {
        let t0 = std::time::Instant::now();
        let f = nextcuda_gguf::Gguf::open(&self.text_cache).map_err(|e| Error(e.0))?;
        // each on the card with the most free memory then (the resident DiT holds more of the faster one)
        let v = conn::Connector::load(&f, "cv", self.freest())?.run(s)?;
        let a = conn::Connector::load(&f, "ca", self.freest())?.run(s)?;
        log(format!("connectors in {:.1} s", t0.elapsed().as_secs_f64()));
        Ok((v, a))
    }

    /// The denoiser resident on the cards (`fast`: the distilled one), loaded if it is not
    pub fn with_dit<T>(&self, fast: bool, log: &mut dyn FnMut(String), f: impl FnOnce(&dit::Dit) -> Result<T>) -> Result<T> {
        let mut g = self.dit.lock().unwrap();
        if g.as_ref().is_some_and(|(k, _)| *k != fast) {
            *g = None;
        }
        if g.is_none() {
            let path = role(&self.files, if fast { "denoiser-fast" } else { "denoiser" })?;
            let split: Option<Vec<usize>> = std::env::var("LTX2_SPLIT").ok().map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect());
            *g = Some((fast, dit::Dit::load(path, &self.gpus, split.as_deref(), log)?));
        }
        f(&g.as_ref().unwrap().1)
    }

    /// A picture [h, w, 3] in [0, 1] through the video VAE's encoder: its latent frame, normalized and packed [h/32 w/32, 128]
    pub fn encode_first(&self, px: &[f32], h: usize, w: usize) -> Result<Vec<f32>> {
        // re-compressed at H.264 CRF 18 as the pipeline does for LTX-2.5 (ltx-pipelines' image conditioning)
        let px = media::crf_roundtrip(px, w, h, 18)?;
        let enc = vae::Encoder::load(self.file("vae")?, self.freest())?;
        let x: Vec<f32> = px.iter().map(|v| v * 2.0 - 1.0).collect();
        let mut z = enc.encode(&x, 1, h, w)?;
        enc.normalize(&mut z);
        Ok(z)
    }

    fn job(&self, spec: &Value, ctl: &mut JobCtl) -> Result<Value> {
        let _g = self.lock.lock().unwrap();
        match spec["kind"].as_str().unwrap_or("") {
            "check" => {
                let dir = PathBuf::from(spec["dir"].as_str().ok_or("check: a reference directory (--dir)")?);
                let st = spec["stages"].as_str().unwrap_or("tokens,te,conn").to_string();
                let stages: Vec<&str> = st.split(',').map(str::trim).collect();
                let worst = check::run(self, &dir, &stages, &mut *ctl.log)?;
                Ok(json!({"worst_rel": worst}))
            }
            "generate" => {
                let cancel = || ctl.cancel.load(std::sync::atomic::Ordering::Relaxed);
                let mut prog = ctl.progress.take();
                let mut p = |stage: &'static str, i: usize, n: usize| {
                    if stage == "denoise" {
                        if let Some(f) = prog.as_mut() {
                            f(i, n);
                        }
                    }
                };
                gen::generate(self, spec, &mut gen::Ctl { log: &mut *ctl.log, progress: &mut p, cancel: &cancel })
            }
            k => Err(Error(format!("{ARCH}: no job {k}"))),
        }
    }
}

impl VideoEngine for Ltx2 {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn defaults(&self) -> Defaults {
        Defaults { width: 768, height: 512, seconds: 5.0, fps: 24.0, steps: 30, cfg: 3.0, sampler: Sampler::Euler, schedule: Schedule::Shift, shift: 1.0,
                   upscale: 1.0 }
    }
    fn load_seconds(&self) -> f64 {
        self.load_s
    }
    fn options(&self) -> &'static [EngineOption] {
        OPTIONS
    }
    fn jobs(&self) -> &'static [&'static str] {
        &["generate", "check"]
    }
    fn run_job(&self, spec: &Value, ctl: &mut JobCtl) -> nextcuda_video::Result<Value> {
        self.job(spec, ctl).map_err(ve)
    }
    fn generate(&self, req: &VideoRequest, progress: &mut dyn FnMut(Progress), cancel: &dyn Fn() -> bool) -> nextcuda_video::Result<Clip> {
        let d = self.defaults();
        let mut spec = json!({
            "kind": "generate", "prompt": req.prompt, "negative": req.negative.clone().unwrap_or_default(), "out": req.out,
            "width": req.width.unwrap_or(d.width), "height": req.height.unwrap_or(d.height),
            "seconds": req.seconds.unwrap_or(d.seconds), "seed": req.seed,
        });
        if let Some(s) = req.steps {
            spec["steps"] = json!(s);
        }
        for (k, v) in &req.extra {
            spec[k] = v.parse::<f64>().map_or_else(|_| json!(v), |n| json!(n));
        }
        let t0 = std::time::Instant::now();
        let mut log = |l: String| eprintln!("{l}");
        let flag = std::sync::atomic::AtomicBool::new(false);
        let mut steps = |done: usize, total: usize| {
            if cancel() {
                flag.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            progress(Progress { stage: "denoise", at: done as u32, of: total as u32, seconds: t0.elapsed().as_secs_f64() });
        };
        let v = self.run_job(&spec, &mut JobCtl { log: &mut log, cancel: &flag, progress: Some(&mut steps) })?;
        let n = |k: &str| v[k].as_u64().unwrap_or(0) as u32;
        let _ = Path::new(&self.components);
        Ok(Clip { path: req.out.clone(), width: n("width"), height: n("height"), frames: n("frames"), fps: 24.0, audio: true })
    }
}
