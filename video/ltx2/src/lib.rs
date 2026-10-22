//! LTX-2.5 (Lightricks): text and pictures to video with its own sound, in one pass - a 22B audio-video DiT (48
//! blocks: video 4096 wide, audio 2048, each attending to the other), Gemma-4 12B as its text encoder through two
//! connectors, a video VAE (x32 spatial, x8 temporal), an audio VAE and a vocoder to 48 kHz, and a latent upsampler
//! (x2) for the shot upscaler. Rust and CUDA: the shared diffusion kernels (`nsd`) and this engine's few
//! (`kernels/video/ltx2`), the DiT's blocks split over both cards and resident, the text stage loaded for a clip and
//! released, nothing streamed.
//!
//! The contract (`nextcuda_video::VideoEngine`): `generate` for a clip from a request, and jobs (`run_job`):
//! `generate`, and `check` (the stages against reference/ltx2's dumps).

pub mod cache;
pub mod check;
pub mod conn;
pub mod ffi;
pub mod quant;
pub mod te;

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
        let text_cache = cache::ensure(&components, threads, log)?;
        let tok_path = files.get("tokenizer").cloned().unwrap_or_else(|| components.join("tokenizer/tokenizer.json"));
        let tok = Tokenizer::from_hf_json(&tok_path).map_err(Error)?;
        Ok(Ltx2 { files: files.clone(), gpus: gpus.to_vec(), components, text_cache, tok, load_s: t0.elapsed().as_secs_f64(), lock: Mutex::new(()) })
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
        let v = conn::Connector::load(&f, "cv", &self.gpus[0])?.run(s)?;
        let a = conn::Connector::load(&f, "ca", self.gpus.last().unwrap())?.run(s)?;
        log(format!("connectors in {:.1} s", t0.elapsed().as_secs_f64()));
        Ok((v, a))
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
        let _ = (req, progress, cancel, Path::new(&self.components), &self.files);
        Err(nextcuda_video::Error(format!("{ARCH}: generate is not built yet")))
    }
}
