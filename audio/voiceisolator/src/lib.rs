//! Inworld's Voice Isolator (voice-isolator-v1, Apache-2.0) on CUDA: a recording's foreground voice kept and everything
//! else taken away, at 48 kHz - the offline checkpoint (64M parameters), a band-split dual-path network over the STFT.
//!
//! ```text
//!   samples (any rate) -> dsp.rs: 48 kHz (torchaudio's kaiser_best), windows of 8 s overlapping by 2 s
//!   -> each window: its STFT (n_fft 2048, hop 480, Hann, centred) -> 46 bands, a token a band and frame (width 512)
//!   -> net.rs: 7 blocks (time attention + band attention, bidirectional Mamba along time + across bands, alternating)
//!      -> per-band heads: a bounded complex mask and a bounded residual a bin
//!   -> the window + the iSTFT of the change; the windows crossfaded; back to the recording's rate
//! ```
//!
//! The voice and the rest (the recording less the voice) together give the recording back. The weights are the
//! checkpoint's own safetensors (`nextcuda models pull voice-isolator`), float32 on the card. `check` compares the
//! stages with the reference's dumps (reference/voiceisolator/ref.py).

pub mod check;
pub mod dsp;
mod ffi;
pub mod net;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use nextcuda_audio::{Audio, AudioEngine, AudioKind, AudioRequest, Defaults, EngineOption, Error, IsolateRequest, Isolated, LoadOptions, ModelFiles, Result,
                     Step};
use nextcuda_core::Gpu;
use nextcuda_gguf::safetensors::SafeTensors;

use crate::dsp::Stft;
use crate::net::{Config, Net, Taps};

pub const ARCH: &str = "voiceisolator";
pub const ROLES: &[&str] = &["model"];
pub const OPTIONS: &[EngineOption] = &[];

pub fn kind() -> AudioKind {
    AudioKind { archs: &[ARCH], name: "Voice Isolator (a recording's voice kept, the rest taken away; 48 kHz, Inworld)", roles: ROLES, load, options: OPTIONS }
}

fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn AudioEngine>> {
    let gpu = gpus.first().ok_or_else(|| Error("no GPU".into()))?;
    Ok(Box::new(VoiceIsolator::load(files, gpu, o, log)?))
}

/// A checkpoint file (or the directory holding denoiser.safetensors): its files by role
pub fn files_in(dir: &Path) -> Result<ModelFiles> {
    let p = if dir.is_file() { dir.to_path_buf() } else { dir.join("denoiser.safetensors") };
    if !p.is_file() {
        return Err(Error(format!("{}: no denoiser.safetensors (the voice isolator's checkpoint)", dir.display())));
    }
    let mut f = ModelFiles::new();
    f.insert("model".into(), p);
    Ok(f)
}

pub struct VoiceIsolator {
    pub net: Net,
    pub stft: Stft,
    load_s: f64,
}

impl VoiceIsolator {
    pub fn load(files: &ModelFiles, gpu: &Arc<Gpu>, _o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<VoiceIsolator> {
        let t0 = Instant::now();
        let path: PathBuf = files.get("model").cloned().ok_or_else(|| Error(format!("{ARCH}: no model file")))?;
        let st = SafeTensors::open(&path).map_err(|e| Error(e.0))?;
        let raw = st.metadata.get("config").ok_or_else(|| Error(format!("{}: no config in its metadata (the release's safetensors carry one)", path.display())))?;
        let v: serde_json::Value = serde_json::from_str(raw).map_err(|e| Error(format!("{}: its config: {e}", path.display())))?;
        let cfg = Config::from_json(&v)?;
        let stft = Stft::new(cfg.n_fft, cfg.hop);
        let net = Net::load(&st, cfg, gpu, log)?;
        let load_s = t0.elapsed().as_secs_f64();
        log(format!("voiceisolator: loaded in {load_s:.1} s"));
        Ok(VoiceIsolator { net, stft, load_s })
    }

    /// The network over one window of 48 kHz audio (model.py forward): the window plus the iSTFT of the heads' change
    /// of its spectrum
    pub fn forward(&self, x: &[f32], taps: Option<&mut Taps>, step: &mut dyn FnMut(usize, usize) -> Result<()>) -> Result<Vec<f32>> {
        let spec = self.stft.forward(x);
        let raw = self.net.run(&spec.v, spec.frames, taps, step)?;
        let corr = self.correct(&spec, &raw);
        let d = self.stft.inverse(&corr, x.len());
        Ok(x.iter().zip(&d).map(|(a, b)| a + b).collect())
    }

    /// The change of a spectrum from the heads' output (band-major, as Net::run gives it)
    pub fn correct(&self, spec: &dsp::Spectrum, raw: &[f32]) -> dsp::Spectrum {
        let frames = spec.frames;
        let mut band_of = vec![(0usize, 0usize); spec.bins];
        for &(s, e) in &self.net.edges {
            for b in &mut band_of[s..e] {
                *b = (s, e - s);
            }
        }
        dsp::correction(spec, self.net.cfg.bound, |t, k| {
            let (s, w) = band_of[k];
            let at = 4 * frames * s + t * 4 * w + (k - s) * 4;
            [raw[at], raw[at + 1], raw[at + 2], raw[at + 3]]
        })
    }

    /// 48 kHz audio of any length isolated (model.py enhance): windows of the checkpoint's chunk crossfaded over its
    /// overlap, the last ending at the clip's end
    pub fn enhance(&self, a: &[f32], progress: &mut dyn FnMut(usize, usize) -> Result<()>) -> Result<Vec<f32>> {
        let rate = self.net.cfg.rate as f64;
        let chunk = (self.net.cfg.chunk_seconds * rate) as usize;
        let overlap = ((self.net.cfg.overlap_seconds * rate) as usize).min(chunk / 2);
        let ws = dsp::windows(a.len(), chunk, overlap);
        let n = ws.len();
        if n == 1 && ws[0].1.is_empty() {
            progress(0, 1)?;
            let y = self.forward(a, None, &mut |_, _| Ok(()))?;
            progress(1, 1)?;
            return Ok(y);
        }
        let mut out = vec![0f32; a.len()];
        let mut weight = vec![0f32; a.len()];
        for (i, (s, w)) in ws.iter().enumerate() {
            progress(i, n)?;
            let y = self.forward(&a[*s..s + chunk], None, &mut |_, _| Ok(()))?;
            for (j, (yv, wv)) in y.iter().zip(w).enumerate() {
                out[s + j] += yv * wv;
                weight[s + j] += wv;
            }
        }
        progress(n, n)?;
        Ok(out.iter().zip(&weight).map(|(o, w)| o / w.max(1e-6)).collect())
    }

    /// A recording's voice (isolator.py isolate): mono at its rate -> 48 kHz -> isolated -> the answer's rate; the rest
    /// is the recording (at that rate) less the voice
    pub fn run(&self, req: &IsolateRequest, progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<Isolated> {
        let t0 = Instant::now();
        if req.samples.is_empty() || req.rate == 0 {
            return Err(Error("the audio is empty: there is nothing to isolate".into()));
        }
        if req.samples.iter().any(|v| !v.is_finite()) {
            return Err(Error("the audio holds NaN or infinite samples".into()));
        }
        let model_rate = self.net.cfg.rate;
        let out_rate = req.out_rate.unwrap_or(req.rate);
        let a = dsp::resample(&req.samples, req.rate, model_rate);
        let y = self.enhance(&a, &mut |at, of| progress(Step { phase: "isolate", at: at as u32, of: of as u32, seconds: t0.elapsed().as_secs_f64() }))?;
        let mut voice = dsp::resample(&y, model_rate, out_rate);
        let mut input = if out_rate == req.rate { req.samples.clone() } else { dsp::resample(&req.samples, req.rate, out_rate) };
        if out_rate == req.rate {
            // fit_length: the recording's own length
            voice.resize(req.samples.len(), 0.0);
        }
        input.resize(voice.len(), 0.0);
        let background: Vec<f32> = input.iter().zip(&voice).map(|(x, v)| x - v).collect();
        Ok(Isolated { voice: Audio { rate: out_rate, channels: 1, samples: voice }, background: Audio { rate: out_rate, channels: 1, samples: background } })
    }
}

impl AudioEngine for VoiceIsolator {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn defaults(&self) -> Defaults {
        Defaults { seconds: 0.0, max_seconds: 0.0, steps: 0, cfg: 1.0, rate: self.net.cfg.rate }
    }
    fn guidance(&self) -> bool {
        false
    }
    fn options(&self) -> &'static [EngineOption] {
        OPTIONS
    }
    fn load_seconds(&self) -> f64 {
        self.load_s
    }
    fn generate(&self, _req: &AudioRequest, _progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<Audio> {
        Err(Error("the voice isolator makes no sound of its own: audio.isolate takes a recording".into()))
    }
    fn isolates(&self) -> bool {
        true
    }
    fn isolate(&self, req: &IsolateRequest, progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<Isolated> {
        self.run(req, progress)
    }
    fn report(&self) -> Vec<String> {
        vec![format!("voice isolator on {}: {:.0} MiB of weights", self.net.gpu.name, self.net.weight_bytes as f64 / (1u64 << 20) as f64)]
    }
}
