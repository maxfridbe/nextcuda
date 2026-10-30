//! MOSS-Image-SFT (OpenMOSS, Apache-2.0) in Rust and CUDA: a 5.1B text-to-image DiT over Qwen3.5-9B's text states and
//! FLUX.2's small VAE. OpenMOSS's pipeline (pipeline_moss_image.py) over the shared diffusion kernels, nextcuda's own
//! Qwen3.5 engine and this engine's kernels (kernels/image/mossimage); no Python, no ggml.
//!
//! ```text
//!   prompt -> the chat template -> Qwen3.5-9B -> its last hidden state [L, 4096] -> the DiT's text embedder
//!   noise [128, H/16, W/16] (seeded) -> Euler over the V2 schedule (shift 5, snapped to the 1,000-step grid), each step
//!     uncond + 3 (cond - uncond): the conditional branch, and the empty prompt's with block 22 left out
//!   latents -> FLUX.2's VAE (batch norm undone, 2 x 2 patches unpacked) -> RGB
//! ```
//!
//! Two cards: the DiT and the VAE on the first (5.5 GB and their activations), the text encoder on the second (9.5 GB),
//! every weight resident in its file's type - nothing streamed.
//!
//! LoRAs (the blocks' q k v o ffn_up_gate ffn_down) are merged into the matrices at load, before they are made int8. A
//! `turbo-lora` file is a few-step distillation (akhaliq's MOSS-Image-SFT-4-steps: DMD onto the scheduler's own 4-step
//! levels, t = 1000 / 910 / 717 / 24): with it the defaults are 4 steps and no guidance, or `NS_MOSS_STEPS` (an
//! 8-step distillation's entry sets 8).

pub mod check;
pub mod dit;
pub mod ffi;
pub mod te;
pub mod vae;

use std::sync::{Arc, Mutex};
use std::time::Instant;

use nextcuda_core::{DevBuf, Gpu};
use nextcuda_diffusion::kernels::Nsd;
use nextcuda_gguf::Gguf;
use nextcuda_diffusion::lora;
use nextcuda_image::{Defaults, Error, ImageEngine, ImageKind, ImageRequest, LoadOptions, LoraUse, ModelFiles, Picture, Result, Sampler, Schedule, SizePresets, Step};

pub const ARCH: &str = "moss-image";
/// the transformer GGUF, the text encoder GGUF (Qwen3.5-9B), the VAE (diffusers' safetensors)
pub const ROLES: &[&str] = &["transformer", "text-encoder", "vae"];
/// a turbo entry's few-step LoRA (merged at load; its defaults 4 steps, cfg 1)
pub const TURBO: &str = "turbo-lora";
/// MOSSImageScheduler's shift, and its training grid's steps
const SHIFT: f32 = 5.0;
const TRAIN_STEPS: usize = 1000;
/// the block the unconditional branch leaves out (uncond_skip_layers=(22,))
pub const UNCOND_SKIP: usize = 22;

/// The aspect ratios and pixel budgets of OpenMOSS's Gradio app (gradio_presets.py): 0.25 M (512 x 512), 1 M, 1.5 M,
/// 2 M and 4 M pixels, from 1:8 to 8:1; a size is `fit_size` of one at the other. This engine draws 64 to 4096 a side
/// (as nextsycl's) and at most `MAX_PIXELS`, so every combination is offered
const RATIOS: &[&str] = &["1:8", "1:4", "1:3", "1:2", "9:16", "2:3", "3:4", "1:1", "4:3", "3:2", "16:9", "2:1", "3:1", "4:1", "8:1"];
const PIXELS: &[u32] = &[262_144, 1_048_576, 1_572_864, 2_097_152, 4_194_304];
/// The largest picture: the largest of OpenMOSS's 4 M presets inside 4096 a side, 2512 x 1680 (each side rounded to 16
/// takes 3:2 a little past 4 M; 2048 x 2048 and these fit a 12 GB card beside the DiT, four times that would not)
pub const MAX_PIXELS: u32 = 4_220_160;

/// OpenMOSS's `fit_size`: the budget's root at the ratio (W / H), each side rounded to 16 (halves up), at least 16
pub fn fit_size(pixels: u32, ratio: f64) -> (u32, u32) {
    let side = |e: f64| (((e / 16.0 + 0.5).floor() as u32) * 16).max(16);
    (side((pixels as f64 * ratio).sqrt()), side((pixels as f64 / ratio).sqrt()))
}

pub fn kind() -> ImageKind {
    ImageKind { archs: &[ARCH], name: "MOSS-Image SFT (5.1B single-stream DiT, Qwen3.5-9B text encoder, FLUX.2 VAE; Rust + CUDA)", roles: ROLES, load, options: &[] }
}

fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn ImageEngine>> {
    let mut e = MossImage::load(files, gpus, &o.merge_loras, log)?;
    if let Some(v) = o.setting("NS_MOSS_STEPS") {
        e.steps = Some(v.trim().parse().ok().filter(|n| (1..=200).contains(n)).ok_or_else(|| Error(format!("NS_MOSS_STEPS={v}: a step count, 1 to 200")))?);
    }
    Ok(Box::new(e))
}

/// torch.linspace(start, end, n) in float32, as torch makes it: the first half from the start, the rest from the end,
/// each a fused multiply-add (its vectorized kernel's: plain products and sums differ in the last bit at 239 of a
/// thousand levels, and the schedule snaps to them)
fn linspace(start: f32, end: f32, n: usize) -> Vec<f32> {
    if n == 1 {
        return vec![start];
    }
    let step = (end - start) / (n - 1) as f32;
    let half = n / 2;
    (0..n).map(|i| if i < half { step.mul_add(i as f32, start) } else { (-step).mul_add((n - 1 - i) as f32, end) }).collect()
}

/// The schedule's grid: linspace(1, shift / (1000 + shift - 1), n), shifted (shift v / (1 + (shift - 1) v))
fn grid(n: usize, shift: f32) -> Vec<f32> {
    let lower = shift / (TRAIN_STEPS as f32 + shift - 1.0);
    linspace(1.0, lower, n).into_iter().map(|v| shift * v / (1.0 + (shift - 1.0) * v)).collect()
}

/// MOSSImageScheduler's levels for `steps` (its V2 nearest-grid schedule): the shifted grid snapped to the training
/// grid's nearest level (the lower on a tie), repeats dropped, then 0. May be fewer than steps + 1.
pub fn sigmas(steps: usize) -> Vec<f32> {
    sigmas_shift(steps, SHIFT)
}

/// `sigmas` at another shift (a request's)
pub fn sigmas_shift(steps: usize, shift: f32) -> Vec<f32> {
    let want = grid(steps.max(1), shift);
    let mut train = grid(TRAIN_STEPS, 1.0);
    train.reverse();
    let mut out: Vec<f32> = Vec::with_capacity(want.len() + 1);
    for s in want {
        // searchsorted (left): the first level not below s
        let upper = train.partition_point(|t| *t < s).min(TRAIN_STEPS - 1);
        let lower = upper.saturating_sub(1);
        let pick = if (s - train[lower]).abs() <= (train[upper] - s).abs() { lower } else { upper };
        let v = train[pick];
        if out.last() != Some(&v) {
            out.push(v);
        }
    }
    out.push(0.0);
    out
}

/// A level's timestep for the DiT: sigma times 1000, in float32 as the scheduler holds it
pub fn timestep(sigma: f32) -> f64 {
    (sigma * TRAIN_STEPS as f32) as f64
}

/// A seeded standard-normal draw (splitmix64, Box-Muller): the starting noise
pub fn noise(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed;
    let mut next = || {
        s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut out = Vec::with_capacity(n + 1);
    while out.len() < n {
        let (u, v) = (next().max(1e-300), next());
        let r = (-2.0 * u.ln()).sqrt();
        out.push((r * (std::f64::consts::TAU * v).cos()) as f32);
        out.push((r * (std::f64::consts::TAU * v).sin()) as f32);
    }
    out.truncate(n);
    out
}

pub struct MossImage {
    pub(crate) te: te::TextEncoder,
    pub(crate) nsd: Nsd,
    pub(crate) dit: dit::Dit,
    pub(crate) vae: vae::Vae,
    /// a turbo LoRA merged: 4 steps and no guidance by default
    turbo: bool,
    /// the default step count when set (NS_MOSS_STEPS); else 4 with a turbo LoRA, 50 without
    pub steps: Option<u32>,
    gpu: Arc<Gpu>,
    loaded: Instant,
    load_s: f64,
    busy: Mutex<()>,
    conds: Mutex<Vec<(String, Arc<Vec<f32>>)>>,
}

fn role<'f>(files: &'f ModelFiles, r: &str) -> Result<&'f std::path::Path> {
    files.get(r).map(|p| p.as_path()).ok_or_else(|| Error(format!("{ARCH}: no {r} file")))
}

impl MossImage {
    /// The DiT and the VAE on `gpus[0]`, the text encoder on `gpus[1]` (or beside them on one card large enough); the
    /// `turbo-lora` file and `loras` merged into the DiT
    pub fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], loras: &[LoraUse], log: &mut dyn FnMut(String)) -> Result<MossImage> {
        let t0 = Instant::now();
        nextcuda_core::use_kind("image");
        let gpu = gpus.first().ok_or_else(|| Error("no GPU".into()))?.clone();
        let te_gpu = gpus.get(1).cloned().unwrap_or_else(|| gpu.clone());
        let f = Gguf::open(role(files, "transformer")?).map_err(|e| Error(e.0))?;
        let dit_b: u64 = f.tensors.iter().map(|t| t.bytes).sum();
        let (total, free) = gpu.memory()?;
        // the transformer as stored, the VAE and ~2 GiB of activations at 1024 x 1024 (the VAE's last stage too)
        let need = dit_b + (1u64 << 29) + (2u64 << 30);
        let free = free.unwrap_or(total);
        if need > free {
            return Err(Error(format!("{ARCH}: the DiT needs about {:.1} GiB on {}; {:.1} GiB are free", need as f64 / (1u64 << 30) as f64, gpu.name,
                                      free as f64 / (1u64 << 30) as f64)));
        }
        // the text encoder first on its card: on one card, a full one says so before the DiT is read
        let te = te::TextEncoder::load(role(files, "text-encoder")?, &te_gpu, log)?;
        let nsd = Nsd::new(&gpu)?;
        // the int8 tensor cores unless NS_MOSS_INT8=0 (the matrices as stored: slower, a hair closer to the reference)
        let int8 = std::env::var("NS_MOSS_INT8").map_or(true, |v| v != "0");
        let turbo = files.get(TURBO);
        let mut deltas = Vec::new();
        for (name, path, scale) in turbo.map(|p| ("turbo", p.as_path(), 1.0)).into_iter().chain(loras.iter().map(|l| (l.name.as_str(), l.path.as_path(), l.scale))) {
            let d = lora::read(path, scale).map_err(Error)?;
            log(format!("LoRA {name} x {scale}: {} matrices, rank {}", d.len(), d.first().map_or(0, |x| x.r)));
            deltas.extend(d);
        }
        let dit = dit::Dit::load(&f, &nsd, int8, deltas, log)?;
        let vae = vae::Vae::load(role(files, "vae")?, &nsd, log)?;
        Ok(MossImage { te, nsd, dit, vae, turbo: turbo.is_some(), steps: None, gpu, loaded: Instant::now(), load_s: t0.elapsed().as_secs_f64(), busy: Mutex::new(()), conds: Mutex::new(Vec::new()) })
    }

    /// The text encoder's last hidden state for a prompt, the last few kept
    fn hidden(&self, prompt: &str) -> Result<Arc<Vec<f32>>> {
        if let Some((_, c)) = self.conds.lock().unwrap().iter().find(|(p, _)| p == prompt) {
            return Ok(c.clone());
        }
        let c = Arc::new(self.te.encode(prompt)?);
        let mut list = self.conds.lock().unwrap();
        list.push((prompt.to_string(), c.clone()));
        if list.len() > 8 {
            list.remove(0);
        }
        Ok(c)
    }

    /// Latents float32 [128, h, w] from the starting noise `start` (same shape): `sig`'s Euler steps over the DiT
    /// after `pre`, and with guidance (cfg not 1) uncond + cfg (cond - uncond), the unconditional branch over `npre`
    /// with block 22 left out
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sample(&self, pre: &dit::Prefix, npre: Option<&dit::Prefix>, cfg: f32, start: &[f32], h: usize, w: usize, sig: &[f32],
                         step: &mut dyn FnMut(usize) -> Result<()>) -> Result<Vec<f32>> {
        let k = ffi::api()?;
        let gpu = &self.gpu;
        let n = h * w * dit::LATENT;
        let lat = DevBuf::from_f32(gpu, &dit::tokens(start, h * w))?;
        for i in 0..sig.len() - 1 {
            let (s, sn) = (sig[i], sig[i + 1]);
            let md = self.dit.step_mod(timestep(s))?;
            let v = self.dit.velocity(&self.nsd, pre, &lat, (h, w), &md, None, None)?;
            let v = match npre {
                Some(np) => {
                    let u = self.dit.velocity(&self.nsd, np, &lat, (h, w), &md, Some(UNCOND_SKIP), None)?;
                    // SAFETY: n floats each: v = v - u, then u += cfg v (u + cfg (cond - uncond))
                    ffi::check(unsafe { (k.axpy)(gpu.raw(), v.fp(), u.fp(), n as i64, -1.0) }, "guidance")?;
                    ffi::check(unsafe { (k.axpy)(gpu.raw(), u.fp(), v.fp(), n as i64, cfg) }, "guidance")?;
                    u
                }
                None => v,
            };
            // the scheduler's step: x + (sigma_next - sigma) v
            // SAFETY: n floats each.
            ffi::check(unsafe { (k.axpy)(gpu.raw(), lat.fp(), v.fp(), n as i64, sn - s) }, "euler step")?;
            gpu.sync()?;
            step(i)?;
        }
        Ok(dit::planes(&lat.to_f32()?, h * w))
    }
}

impl ImageEngine for MossImage {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn defaults(&self) -> Defaults {
        let (steps, cfg) = if self.turbo { (4, 1.0) } else { (50, 3.0) };
        let steps = self.steps.unwrap_or(steps);
        Defaults { width: 1024, height: 1024, steps, cfg, sampler: Sampler::Euler, schedule: Schedule::Shift, shift: SHIFT }
    }
    fn samplers(&self) -> Vec<Sampler> {
        vec![Sampler::Euler]
    }
    fn size_presets(&self) -> Option<SizePresets> {
        Some(SizePresets { ratios: RATIOS, pixels: PIXELS })
    }
    fn sides(&self) -> (u32, u32) {
        (64, 4096)
    }
    fn max_pixels(&self) -> Option<u32> {
        Some(MAX_PIXELS)
    }
    /// a turbo entry's LoRA distilled the scheduler's own levels: another shift would leave them
    fn shift(&self) -> bool {
        !self.turbo
    }
    fn schedules(&self) -> Vec<Schedule> {
        vec![Schedule::Shift]
    }
    /// cfg not 1 (a turbo entry runs at 1): the pipeline's guidance against the negative prompt (empty by default), its branch leaving out block 22
    fn guidance(&self) -> bool {
        !self.turbo
    }
    fn load_seconds(&self) -> f64 {
        self.load_s
    }

    fn generate(&self, req: &ImageRequest, progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<Vec<Picture>> {
        let _one = self.busy.lock().unwrap();
        let d = self.defaults();
        let t0 = Instant::now();
        if req.edit.is_some() {
            return Err(Error(format!("{ARCH}: text to image only")));
        }
        let (rw, rh) = (req.width.unwrap_or(d.width) as usize, req.height.unwrap_or(d.height) as usize);
        if rw < 64 || rh < 64 || rw > 4096 || rh > 4096 || rw % 16 != 0 || rh % 16 != 0 || rw * rh > MAX_PIXELS as usize {
            return Err(Error(format!("{rw}x{rh}: multiples of 16 from 64 to 4096 a side, at most {MAX_PIXELS} pixels (2512 x 1680, 2048 x 2048, \
                                      4096 x 1024; 1024 x 1024 recommended)")));
        }
        let steps = req.steps.unwrap_or(d.steps).max(1) as usize;
        let cfg = req.cfg.filter(|c| *c >= 0.0).unwrap_or(d.cfg);
        let sig = sigmas_shift(steps, req.shift.filter(|s| s.is_finite() && *s > 0.0).unwrap_or(SHIFT));
        let of = (sig.len() - 1) as u32;
        progress(Step { picture: 0, at: 0, of, seconds: t0.elapsed().as_secs_f64() })?;
        let pre = self.dit.prefix(&self.nsd, &self.hidden(req.prompt.trim())?)?;
        let npre = if cfg != 1.0 {
            let neg = req.negative.as_deref().unwrap_or("").trim().to_string();
            Some(self.dit.prefix(&self.nsd, &self.hidden(&neg)?)?)
        } else {
            None
        };
        let (h, w) = (rh / 16, rw / 16);
        let mut out = Vec::new();
        for p in 0..req.n.max(1) {
            let start = noise(req.seed + p as u64, dit::LATENT * h * w);
            let mut step = |i: usize| progress(Step { picture: p, at: i as u32 + 1, of, seconds: t0.elapsed().as_secs_f64() });
            let lat = self.sample(&pre, npre.as_ref(), cfg, &start, h, w, &sig, &mut step)?;
            let (rgb, pw, ph) = self.vae.decode(&self.nsd, &lat, h, w)?;
            out.push(Picture { width: pw as u32, height: ph as u32, channels: 3, data: rgb });
        }
        Ok(out)
    }

    fn report(&self) -> Vec<String> {
        vec![format!("{ARCH} in Rust and CUDA: the DiT and VAE on {}, the text encoder on {} (loaded {:.0} s ago)", self.gpu.name, self.te.gpu.name,
                     self.loaded.elapsed().as_secs_f64())]
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn sizes_from_a_budget_and_a_ratio_are_openmoss_s() {
        // values from gradio_presets.fit_size
        assert_eq!(super::fit_size(262_144, 1.0), (512, 512));
        assert_eq!(super::fit_size(1_048_576, 16.0 / 9.0), (1360, 768));
        assert_eq!(super::fit_size(1_572_864, 2.0 / 3.0), (1024, 1536));
        assert_eq!(super::fit_size(2_097_152, 1.0), (1456, 1456));
        assert_eq!(super::fit_size(4_194_304, 1.0), (2048, 2048));
        assert!(super::RATIOS.contains(&"16:9") && super::PIXELS.contains(&1_048_576));
        // every 4 M preset whose sides fit 4096 is inside the engine's pixel limit
        for r in super::RATIOS {
            let (a, b) = r.split_once(':').unwrap();
            let (w, h) = super::fit_size(4_194_304, a.parse::<f64>().unwrap() / b.parse::<f64>().unwrap());
            assert!(w > 4096 || h > 4096 || w * h <= super::MAX_PIXELS, "{r}: {w}x{h}");
        }
    }

    #[test]
    fn a_shift_moves_the_levels_and_the_default_is_the_models() {
        assert_eq!(super::sigmas_shift(8, super::SHIFT), super::sigmas(8));
        let (a, b) = (super::sigmas_shift(8, 3.0), super::sigmas(8));
        assert_ne!(a, b);
        assert!(a.windows(2).all(|w| w[0] > w[1]) && a[0] == 1.0 && *a.last().unwrap() == 0.0);
    }

    #[test]
    fn the_levels_are_the_schedulers() {
        // MOSSImageScheduler().set_timesteps(8).sigmas, from the reference (pipeline_moss_image.py), bit for bit
        let s = super::sigmas(8);
        assert_eq!(s, [1.0, 0.968, 0.926, 0.871, 0.791, 0.671, 0.46300003, 0.024, 0.0]);
        assert_eq!(s[0], 1.0);
        assert_eq!(*s.last().unwrap(), 0.0);
        assert!(s.windows(2).all(|w| w[0] > w[1]), "{s:?}");
        // each level is on the training grid: a multiple of 0.999 / 999 above 0.001
        for v in &s[..s.len() - 1] {
            let k = (*v as f64 - 0.001) / (0.999 / 999.0);
            assert!((k - k.round()).abs() < 1e-3, "{v} is not a grid level");
        }
    }
}
