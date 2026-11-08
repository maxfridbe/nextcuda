//! Z-Image-Turbo (Tongyi-MAI, Apache-2.0) in Rust and CUDA: few-step text to image. Tongyi-MAI's pipeline
//! (github.com/Tongyi-MAI/Z-Image src/zimage/pipeline.py; diffusers' ZImagePipeline is the same) over the foundation's
//! Z-Image DiT (crates/zimage, which Ming-Image runs too); no Python, no ggml.
//!
//! ```text
//!   prompt -> Qwen3's chat template ("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n", thinking on:
//!     no empty think block), at most 512 tokens -> Qwen3-4B -> the hidden state after layer 35 of 36 (hidden_states[-2],
//!     no final norm) [L, 2560]
//!   noise [16, H/8, W/8] (torch.randn on a CPU generator seeded `seed`) -> 8 Euler steps on FlowMatchEuler's levels
//!     (linspace(1, 1/8, 8), shift 3, then 0), the DiT's velocity each step, no guidance (the turbo is distilled from it)
//!   latents / 0.3611 + 0.1159 -> FLUX's 16-channel VAE decoder -> RGB
//! ```
//!
//! Two 12 GB cards: the DiT (Q8_0, the refiners bfloat16) and the VAE on the first, the text encoder (Qwen3-4B Q8_0, its
//! first 35 layers) on the second - every weight resident in its GGUF's type, nothing streamed. Sizes are multiples of
//! 16 (the VAE's 8 times the DiT's 2 x 2 patches); the official demo offers 1024, 1280 and 1536 pictures.

pub mod ffi;
pub mod vae;

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use nextcuda_core::Gpu;
use nextcuda_diffusion::kernels::Nsd;
use nextcuda_gguf::Gguf;
use nextcuda_h3::device::Device;
use nextcuda_h3::te::{Resident, TextEncoder};
use nextcuda_h3::tokenizer::Tokenizer;
use nextcuda_image::{Defaults, Error, ImageEngine, ImageKind, ImageRequest, LoadOptions, ModelFiles, Picture, Result, Sampler, Schedule, SizePresets, Step};
use nextcuda_zimage::dit;

pub const ARCH: &str = "z-image";
/// the model's name in what a request is told
const NAME: &str = "Z-Image-Turbo";
/// the transformer GGUF, the text encoder GGUF (Qwen3-4B), the VAE (diffusers' safetensors), the tokenizer
/// (`vocab.json`; `merges.txt` and `tokenizer_config.json` beside it: the repository's tokenizer/)
pub const ROLES: &[&str] = &["transformer", "text-encoder", "vae", "tokenizer"];
/// scheduler_config.json: FlowMatchEulerDiscreteScheduler's static shift (use_dynamic_shifting false: the pipeline's mu
/// is computed and not used)
pub const SHIFT: f32 = 3.0;
/// the turbo's steps (DMD-distilled for 8; the official demo fixes them)
pub const STEPS: u32 = 8;
/// a request may ask for other step counts within these (the model is made for 8)
const MAX_STEPS: u32 = 50;
/// the pipeline's max_sequence_length: the prompt's tokens (the template's included) are cut there
const MAX_TOKENS: usize = 512;
/// the hidden state taken: after this many of Qwen3-4B's 36 layers (hidden_states[-2]); the 36th is not loaded
const TE_TAP: usize = 35;
/// Qwen3-4B's rotary base (Qwen3-VL's encoders use 5e6)
const QWEN3_THETA: f32 = 1_000_000.0;
/// a side is a multiple of this: the VAE's 8 times the 2 x 2 patch
pub const GRID: u32 = 16;
/// the official demo's sizes: 1024, 1280 and 1536 at 1:1, 9:7, 4:3, 3:2, 16:9, 21:9 and their turns
const RATIOS: &[&str] = &["9:21", "9:16", "2:3", "3:4", "7:9", "1:1", "9:7", "4:3", "3:2", "16:9", "21:9"];
const PIXELS: &[u32] = &[1_048_576, 1_638_400, 2_359_296];
/// The largest picture: the demo's biggest, 1536 x 1536 or 2048 x 1152 (11.0 GB of the 5070 at the peak); at 2048 x 2048
/// the VAE's last stages want 2 GiB more than the card has beside the resident DiT
pub const MAX_PIXELS: u32 = 2_359_296;

pub fn kind() -> ImageKind {
    ImageKind {
        archs: &[ARCH],
        name: "Z-Image-Turbo (Z-Image's 6B single-stream DiT, Qwen3-4B text encoder, FLUX's VAE; 8 steps, no guidance; Rust + CUDA)",
        roles: ROLES,
        load,
        options: &[],
    }
}

fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], _o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn ImageEngine>> {
    Ok(Box::new(ZImage::load(files, gpus, log)?))
}

fn role<'f>(files: &'f ModelFiles, r: &str) -> Result<&'f Path> {
    files.get(r).map(|p| p.as_path()).ok_or_else(|| Error(format!("{ARCH}: no {r} file")))
}

fn he(e: nextcuda_h3::Error) -> Error {
    Error(e.0)
}

fn gib(b: u64) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

/// The `steps + 1` noise levels: linspace(1, 1/steps, steps) shifted (shift s / (1 + (shift - 1) s)), then 0 (the
/// official pipeline's: timesteps linspace(1000, 0, steps + 1) without the last, sigma_min 0)
pub fn sigmas(steps: usize, shift: f32) -> Vec<f32> {
    let n = steps.max(1);
    let sh = shift as f64;
    let mut out: Vec<f32> = (0..n).map(|i| {
        let s = 1.0 - i as f64 / n as f64;
        (sh * s / (1.0 + (sh - 1.0) * s)) as f32
    }).collect();
    out.push(0.0);
    out
}

/// Python's round(): halves to the even neighbour
fn py_round(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - x.signum()
    } else {
        r
    }
}

/// A size asked, each side rounded to the nearest multiple of 16 (the pipeline refuses others)
pub fn round_size(w: u32, h: u32) -> (u32, u32) {
    let r = |v: u32| ((py_round(v as f64 / GRID as f64) as u32) * GRID).max(GRID);
    (r(w), r(h))
}

/// The prompt as Qwen3 reads it: its chat template for one user turn and the assistant's start (apply_chat_template
/// with add_generation_prompt and enable_thinking: no empty think block)
pub fn template(prompt: &str) -> String {
    format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
}

pub struct ZImage {
    te: TextEncoder,
    resident: Resident,
    tok: Tokenizer,
    te_dev: Arc<Device>,
    nsd: Nsd,
    dit: dit::Dit,
    vae: vae::Vae,
    gpu: Arc<Gpu>,
    loaded: Instant,
    load_s: f64,
    busy: Mutex<()>,
    /// the last prompts' text features [L, 2560]
    conds: Mutex<Vec<(String, Arc<Vec<f32>>)>>,
}

impl ZImage {
    /// The DiT and the VAE on `gpus[0]`, the text encoder on `gpus[1]` (or beside them on one card large enough)
    pub fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], log: &mut dyn FnMut(String)) -> Result<ZImage> {
        let t0 = Instant::now();
        nextcuda_core::use_kind("image");
        let gpu = gpus.first().ok_or_else(|| Error("no GPU".into()))?.clone();
        let te_index = gpus.get(1).map_or(gpu.index, |g| g.index);
        let f = Gguf::open(role(files, "transformer")?).map_err(|e| Error(e.0))?;
        let dit_b: u64 = f.tensors.iter().map(|t| t.bytes).sum();
        let (total, free) = gpu.memory()?;
        // the transformer as stored, its float32 modulation linears, the VAE and ~2.5 GiB of activations at 1024 x 1024
        let need = dit_b + (1u64 << 29) + (1u64 << 28) + (5u64 << 29);
        let free = free.unwrap_or(total);
        if need > free {
            return Err(Error(format!("{ARCH}: the DiT needs about {:.1} GiB on {}; {:.1} GiB are free", gib(need), gpu.name, gib(free))));
        }
        // the text encoder first on its card: on one card, a full one says so before the DiT is read
        let tok_file = role(files, "tokenizer")?;
        let tok = Tokenizer::load_qwen(if tok_file.is_file() { tok_file.parent().unwrap_or(tok_file) } else { tok_file }).map_err(he)?;
        let te_path = role(files, "text-encoder")?;
        let mut te = TextEncoder::open(te_path).map_err(he)?;
        if te.layers != 36 || te.hidden != 2560 {
            return Err(Error(format!("{}: {} layers of {}, not Qwen3-4B's 36 of 2560", te_path.display(), te.layers, te.hidden)));
        }
        te.layers = TE_TAP;
        te.theta = QWEN3_THETA;
        let te_dev = Device::open_index(te_index).map_err(he)?;
        let t1 = Instant::now();
        let resident = te.load_resident(std::slice::from_ref(&te_dev), 8).map_err(he)?;
        let te_bytes: u64 = resident.placement().iter().map(|p| p.2).sum();
        log(format!("z-image te: Qwen3-4B, {TE_TAP} of 36 layers ({:.2} GiB) on {} in {:.1} s", gib(te_bytes), te_dev.name(), t1.elapsed().as_secs_f64()));
        let nsd = Nsd::new(&gpu)?;
        let dit = dit::Dit::load(&f, &nsd, dit::Pad::Tokens, log)?;
        let vae = vae::Vae::load(role(files, "vae")?, &nsd, log)?;
        Ok(ZImage { te, resident, tok, te_dev, nsd, dit, vae, gpu, loaded: Instant::now(), load_s: t0.elapsed().as_secs_f64(), busy: Mutex::new(()),
                    conds: Mutex::new(Vec::new()) })
    }

    /// The prompt's tokens as the text encoder reads them: the template's, cut at 512
    pub fn tokens(&self, prompt: &str) -> Result<Vec<u32>> {
        let mut ids = self.tok.encode(&template(prompt)).map_err(he)?;
        ids.truncate(MAX_TOKENS);
        Ok(ids)
    }

    /// The prompt's text features [L, 2560], the last few kept
    pub fn condition(&self, prompt: &str) -> Result<Arc<Vec<f32>>> {
        if let Some((_, c)) = self.conds.lock().unwrap_or_else(|p| p.into_inner()).iter().find(|(p, _)| p == prompt) {
            return Ok(c.clone());
        }
        let ids = self.tokens(prompt)?;
        let mut taps = self.te.encode_resident_taps(&self.resident, &ids, &[], &[], &[TE_TAP], None, &mut |_| Ok(())).map_err(he)?;
        let c = Arc::new(taps.pop().ok_or_else(|| Error("the text encoder gave no hidden state".into()))?);
        let mut list = self.conds.lock().unwrap_or_else(|p| p.into_inner());
        list.push((prompt.to_string(), c.clone()));
        if list.len() > 8 {
            list.remove(0);
        }
        Ok(c)
    }

    /// One picture's latents [16, lh, lw] after the caption `pre`, from noise seeded `seed`, over the levels `sig`
    pub fn latents(&self, pre: &dit::Prefix, seed: u64, lh: usize, lw: usize, sig: &[f32], step: &mut dyn FnMut(usize) -> Result<()>) -> Result<Vec<f32>> {
        let start = nextcuda_diffusion::torch_rng::Mt19937::new(seed).randn(dit::LATENT * lh * lw);
        nextcuda_zimage::sample(&self.dit, &self.nsd, &self.gpu, pre, None, 1.0, &start, lh, lw, sig, step)
    }

    /// The caption's rows for the DiT (once a prompt)
    pub fn prefix(&self, cap: &[f32]) -> Result<dit::Prefix> {
        self.dit.prefix(&self.nsd, cap, &[], 0, None)
    }

    /// RGB bytes [H, W, 3] of latents [16, lh, lw]
    pub fn decode(&self, lat: &[f32], lh: usize, lw: usize) -> Result<(Vec<u8>, usize, usize)> {
        self.vae.decode(&self.nsd, lat, lh, lw)
    }
}

impl ImageEngine for ZImage {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn defaults(&self) -> Defaults {
        Defaults { width: 1024, height: 1024, steps: STEPS, cfg: 1.0, sampler: Sampler::Euler, schedule: Schedule::Shift, shift: SHIFT }
    }
    fn samplers(&self) -> Vec<Sampler> {
        vec![Sampler::Euler]
    }
    fn schedules(&self) -> Vec<Schedule> {
        vec![Schedule::Shift]
    }
    /// distilled from guidance: one pass a step, no negative prompt (the pipeline's guidance_scale 0)
    fn guidance(&self) -> bool {
        false
    }
    /// the scheduler's shift (3; the official demo's "time shift", 1 to 10)
    fn shift(&self) -> bool {
        true
    }
    fn takes_loras(&self) -> bool {
        false
    }
    fn grid(&self) -> u32 {
        GRID
    }
    fn size_presets(&self) -> Option<SizePresets> {
        Some(SizePresets { ratios: RATIOS, pixels: PIXELS })
    }
    fn sides(&self) -> (u32, u32) {
        (256, 2048)
    }
    fn max_pixels(&self) -> Option<u32> {
        Some(MAX_PIXELS)
    }
    fn load_seconds(&self) -> f64 {
        self.load_s
    }

    fn generate(&self, req: &ImageRequest, progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<Vec<Picture>> {
        let _one = self.busy.lock().unwrap_or_else(|p| p.into_inner());
        let t0 = Instant::now();
        let d = self.defaults();
        if req.edit.is_some() {
            return Err(Error(format!("{NAME} makes pictures from text only: it takes no picture to edit")));
        }
        if req.negative.as_deref().is_some_and(|n| !n.trim().is_empty()) {
            return Err(Error(format!("{NAME} takes no negative prompt: it is distilled from guidance (8 steps, one pass a step, no CFG)")));
        }
        if req.cfg.is_some_and(|c| c != 1.0 && c != 0.0) {
            return Err(Error(format!("{NAME} takes no cfg (guidance scale): it is distilled from guidance - leave cfg out, or 1")));
        }
        if !req.loras.is_empty() {
            return Err(Error(format!("{NAME} takes no LoRAs")));
        }
        if req.sampler.is_some_and(|s| s != Sampler::Euler) {
            return Err(Error(format!("{NAME} samples with euler only")));
        }
        if req.schedule.is_some_and(|s| s != Schedule::Shift) {
            return Err(Error(format!("{NAME} runs its scheduler's own levels (schedule shift)")));
        }
        if req.rgba {
            return Err(Error(format!("{NAME} makes opaque pictures (RGB): no rgba")));
        }
        let steps = req.steps.unwrap_or(STEPS);
        if !(1..=MAX_STEPS).contains(&steps) {
            return Err(Error(format!("steps {steps}: 1 to {MAX_STEPS} ({NAME} is made for {STEPS})")));
        }
        let shift = req.shift.unwrap_or(SHIFT);
        if !(shift > 0.0 && shift <= 20.0) {
            return Err(Error(format!("shift {shift}: above 0, at most 20 (the model's is {SHIFT})")));
        }
        let (w, h) = match (req.width, req.height) {
            (Some(w), Some(h)) => round_size(w, h),
            (None, None) => (d.width, d.height),
            _ => return Err(Error("width and height go together, or neither".into())),
        };
        let (lo, hi) = self.sides();
        if w < lo || h < lo || w > hi || h > hi || w * h > MAX_PIXELS {
            return Err(Error(format!("{w}x{h}: sides from {lo} to {hi} (rounded to multiples of {GRID}), at most {MAX_PIXELS} pixels (1536 x 1536; the official demo's: \
                                      1024x1024, 1152x896, 1152x864, 1248x832, 1280x720, 1344x576; 1280x1280, 1536x1024, 1536x864; 1536x1536, \
                                      2048x1152 ... and their turns)")));
        }
        let prompt = req.prompt.trim();
        if prompt.is_empty() {
            return Err(Error("prompt: what to draw".into()));
        }
        let sig = sigmas(steps as usize, shift);
        let of = steps;
        progress(Step { picture: 0, at: 0, of, seconds: t0.elapsed().as_secs_f64() })?;
        let cap = self.condition(prompt)?;
        let pre = self.prefix(&cap)?;
        let (lh, lw) = (h as usize / 8, w as usize / 8);
        eprintln!("{ARCH}: {w}x{h}, {} text tokens, {steps} steps, shift {shift}, conditioned in {:.2} s", cap.len() / 2560, t0.elapsed().as_secs_f64());
        let mut out = Vec::new();
        for p in 0..req.n.max(1) {
            let tp = Instant::now();
            let mut times = Vec::new();
            let mut last = Instant::now();
            let mut step = |i: usize| {
                times.push(last.elapsed().as_secs_f64());
                last = Instant::now();
                progress(Step { picture: p, at: i as u32 + 1, of, seconds: t0.elapsed().as_secs_f64() })
            };
            let tag = |e: Error| Error(nextcuda_image::tag_device_oom(&e.0, Some(self.gpu.index)));
            let lat = self.latents(&pre, req.seed + p as u64, lh, lw, &sig, &mut step).map_err(tag)?;
            let tv = Instant::now();
            let (rgb, pw, ph) = self.decode(&lat, lh, lw).map_err(tag)?;
            eprintln!("{ARCH}: picture {} in {:.2} s (steps {}; vae {:.2} s)", p + 1, tp.elapsed().as_secs_f64(),
                      times.iter().map(|s| format!("{s:.2}")).collect::<Vec<_>>().join(" "), tv.elapsed().as_secs_f64());
            out.push(Picture { width: pw as u32, height: ph as u32, channels: 3, data: rgb });
        }
        Ok(out)
    }

    fn report(&self) -> Vec<String> {
        vec![format!("{ARCH} in Rust and CUDA: the DiT and VAE on {}, the text encoder (Qwen3-4B, {TE_TAP} layers) on {} (loaded {:.0} s ago)", self.gpu.name,
                     self.te_dev.name(), self.loaded.elapsed().as_secs_f64())]
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_levels_are_the_pipelines() {
        // Tongyi-MAI's pipeline (and diffusers' get_default_z_image_sigmas) at 8 steps, shift 3: linspace(1, 1/8, 8)
        // shifted, then 0
        let want = [1.0, 0.954545, 0.9, 0.833333, 0.75, 0.642857, 0.5, 0.3, 0.0];
        let s = super::sigmas(8, 3.0);
        assert_eq!(s.len(), 9);
        for (a, b) in s.iter().zip(want) {
            assert!((a - b).abs() < 1e-5, "{s:?}");
        }
    }

    #[test]
    fn sizes_round_to_sixteen_and_the_template_is_qwens() {
        assert_eq!(super::round_size(1024, 1024), (1024, 1024));
        assert_eq!(super::round_size(1000, 1290), (992, 1296));
        // a half rounds to the even multiple, as Python's round
        assert_eq!(super::round_size(1032, 1048), (1024, 1056));
        assert_eq!(super::template("a cat"), "<|im_start|>user\na cat<|im_end|>\n<|im_start|>assistant\n");
    }
}
