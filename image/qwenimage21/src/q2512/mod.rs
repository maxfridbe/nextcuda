//! Qwen-Image 2512 in Rust and CUDA: the pipeline is host code over the shared diffusion kernels and this crate's own
//! (the 2.1 engine's), the 20B MMDiT in the type its GGUF stores (the blocks that fit on the card, the rest streamed
//! from pinned host memory each step), the Qwen2.5-VL 7B text encoder streamed a layer at a time from its GGUF, the
//! Wan 2.1 VAE in half. No ggml.
//!
//! ```text
//!   prompt -> Qwen-Image's chat template -> Qwen2.5-VL 7B (GGUF) -> last hidden state after the final norm, the
//!     template's 34 leading tokens dropped
//!   noise [h/16 * w/16, 64] (seeded) -> Euler over the flow-matching sigmas, the MMDiT's velocity each step (a guided
//!     step: the prompt's and the negative's passes block by block)
//!   tokens -> latents [h/8, w/8, 16] -> VAE decoder -> RGB
//! ```
//!
//! The effort ladder is the platform's (and the engine's this one replaced): with a `turbo-lora`
//! file (lightx2v's Lightning), a request under 20 steps runs the LoRA (4 steps, guidance 1, the sigmas shifted by 3);
//! 20 or more is a full guided run (30 steps, guidance 4, the size's dynamic shift).

pub mod dit;
pub mod vae;

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use nextcuda_core::{DevBuf, Gpu};
use nextcuda_diffusion::kernels::Nsd;
use nextcuda_diffusion::lora::{self, Delta};
use nextcuda_gguf::Gguf;
use nextcuda_image::{At, Defaults, EngineOption, Error, ImageEngine, ImageRequest, LoadOptions, ModelFiles, Picture, Result, Sampler, Schedule, Step};
use nextcuda_qwen3vl::GgufTextEncoder;

use crate::ffi;
use crate::rust::{noise, sched};

pub const ARCH: &str = "qwen-image-2512";

/// The chat template's system turn and the user turn's opening: their tokens are dropped from the encoding
const PREFIX: &str = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the \
                      objects and background:<|im_end|>\n<|im_start|>user\n";

/// A full run's steps and guidance; the LoRA's; asking for at least `LADDER` steps bypasses the LoRA
const STEPS: u32 = 30;
const CFG: f32 = 4.0;
const LORA_STEPS: u32 = 4;
const LORA_CFG: f32 = 1.0;
const LADDER: u32 = 20;
/// Lightning's sigmas: shifted by 3 (lightx2v's scheduler: base and max shift ln 3), no terminal stretch
const LORA_SHIFT: f64 = 3.0;

pub const OPTIONS: &[EngineOption] = &[EngineOption {
    name: "reserve",
    env: "NC_Q2512_RESERVE_GIB",
    value: "GiB",
    help: "card memory kept for the activations and the VAE when the transformer's blocks are placed (the rest of the blocks stream from host memory; default 3.5)",
    at: At::Load,
}];

pub struct Q2512 {
    nsd: Nsd,
    te: GgufTextEncoder,
    dit: dit::Dit,
    vae: vae::Vae,
    gpu_name: String,
    loaded: Instant,
    load_s: f64,
    busy: Mutex<()>,
    embeds: Mutex<Vec<(String, Arc<dit::Text>)>>,
    /// the distilled LoRA (`turbo-lora`), read once; which LoRAs the blocks carry now
    turbo: Option<Vec<Delta>>,
    base_loras: Vec<Delta>,
    on: Mutex<Option<Vec<String>>>,
}

fn role<'f>(files: &'f ModelFiles, r: &str) -> Result<&'f Path> {
    files.get(r).map(|p| p.as_path()).ok_or_else(|| Error(format!("{ARCH}: no {r} file")))
}

/// `steps` levels from 1 shifted by `shift` (e / (e + 1/t - 1)), then 0
fn shifted(steps: usize, shift: f64) -> Vec<f32> {
    let n = steps.max(1);
    let mut v: Vec<f32> = (0..n).map(|i| 1.0 - i as f64 * (1.0 - 1.0 / n as f64) / (n - 1).max(1) as f64).map(|t| (shift / (shift + (1.0 / t - 1.0))) as f32).collect();
    v.push(0.0);
    v
}

impl Q2512 {
    pub fn load(files: &ModelFiles, gpu: &Arc<Gpu>, o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Q2512> {
        let t0 = Instant::now();
        nextcuda_core::use_kind("image");
        let nsd = Nsd::new(gpu)?;
        let te = GgufTextEncoder::load(role(files, "text-encoder")?, gpu, log)?;
        let vae = vae::Vae::load(role(files, "vae")?, &nsd, log)?;
        let turbo = match files.get("turbo-lora") {
            Some(p) => Some(lora::read(p, 1.0).map_err(Error)?),
            None => None,
        };
        let mut base_loras = Vec::new();
        for l in &o.merge_loras {
            base_loras.extend(lora::read(&l.path, l.scale).map_err(Error)?);
        }
        let lora_bytes: u64 = turbo.iter().flatten().chain(base_loras.iter()).map(|d| (d.a.len() * 2 + d.b.len() * 4) as u64).sum();
        let reserve = o.setting("NC_Q2512_RESERVE_GIB").and_then(|v| v.parse::<f64>().ok()).unwrap_or(3.5);
        let (total, free) = gpu.memory()?;
        let free = free.unwrap_or(total);
        let keep = (reserve * (1u64 << 30) as f64) as u64 + lora_bytes;
        // the staging buffer of a streamed block comes out of the same room (about 0.25 GiB)
        let budget = free.saturating_sub(keep + (1u64 << 28));
        if budget < (2u64 << 30) {
            return Err(nextcuda_image::no_room_error(gpu.index, ((2u64 << 30) - budget) >> 20,
                                                     &format!("{ARCH} needs at least {:.1} GiB free on {}; {:.1} GiB are", (keep + (9u64 << 28)) as f64 / (1u64 << 30) as f64,
                                                              gpu.name, free as f64 / (1u64 << 30) as f64)));
        }
        let f = Gguf::open(role(files, "transformer")?).map_err(|e| Error(e.0))?;
        let dit = dit::Dit::load(&f, &nsd, budget, log)?;
        let e = Q2512 {
            nsd,
            te,
            dit,
            vae,
            gpu_name: gpu.name.clone(),
            loaded: Instant::now(),
            load_s: 0.0,
            busy: Mutex::new(()),
            embeds: Mutex::new(Vec::new()),
            turbo,
            base_loras,
            on: Mutex::new(None),
        };
        Ok(Q2512 { load_s: t0.elapsed().as_secs_f64(), ..e })
    }

    /// The prompt's encoder states after the template's opening: float32 [T, 3584] on the card
    fn encode(&self, prompt: &str) -> Result<Arc<dit::Text>> {
        if let Some((_, e)) = self.embeds.lock().unwrap().iter().find(|(p, _)| p == prompt) {
            return Ok(e.clone());
        }
        let ids = self.te.tokenize(&format!("{PREFIX}{prompt}<|im_end|>\n<|im_start|>assistant\n"));
        let drop = self.te.tokenize(PREFIX).len();
        let all = self.te.encode_normed(&self.nsd, &ids)?;
        let t = ids.len() - drop;
        let d = self.te.hidden;
        let emb = DevBuf::f32(&self.nsd.gpu, t * d)?;
        emb.copy_within(0, &all, drop * d * 4, t * d * 4)?;
        self.nsd.gpu.sync()?;
        let e = Arc::new(dit::Text { emb, tokens: t });
        let mut c = self.embeds.lock().unwrap();
        c.push((prompt.to_string(), e.clone()));
        if c.len() > 8 {
            c.remove(0);
        }
        Ok(e)
    }

    /// The blocks carry `want` LoRAs (by name): set when they differ from what they carry
    fn use_loras(&self, want: Vec<String>, deltas: Vec<Delta>) -> Result<()> {
        let mut on = self.on.lock().unwrap();
        if on.as_ref() == Some(&want) {
            return Ok(());
        }
        *on = None;
        self.dit.set_loras(&deltas)?;
        *on = Some(want);
        Ok(())
    }

    fn round(v: u32) -> usize {
        ((v as f64 / 16.0).round() as usize).max(16) * 16
    }
}

impl ImageEngine for Q2512 {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn defaults(&self) -> Defaults {
        let (steps, cfg) = if self.turbo.is_some() { (LORA_STEPS, LORA_CFG) } else { (STEPS, CFG) };
        Defaults { width: 1024, height: 1024, steps, cfg, sampler: Sampler::Euler, schedule: Schedule::Shift, shift: 0.0 }
    }
    fn load_seconds(&self) -> f64 {
        self.load_s
    }
    fn samplers(&self) -> Vec<Sampler> {
        vec![Sampler::Euler]
    }
    fn schedules(&self) -> Vec<Schedule> {
        vec![Schedule::Shift]
    }
    /// text to image and img2img (a strength); reference edits are another checkpoint's
    fn edits(&self) -> bool {
        false
    }
    fn guidance(&self) -> bool {
        true
    }
    fn transparent(&self) -> bool {
        false
    }
    fn options(&self) -> &'static [EngineOption] {
        OPTIONS
    }
    fn steps_for(&self, req: &ImageRequest) -> u32 {
        let ask = req.steps.unwrap_or(0);
        match (self.turbo.is_some() && ask < LADDER, ask) {
            (true, 0) => LORA_STEPS,
            (false, 0) => STEPS,
            (_, s) => s,
        }
    }

    fn generate(&self, req: &ImageRequest, progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<Vec<Picture>> {
        let _one = self.busy.lock().unwrap();
        let t0 = Instant::now();
        if req.rgba {
            return Err(Error(format!("{ARCH}: its VAE makes no alpha (a transparent picture is Qwen-Image 2.1's)")));
        }
        let img2img = match &req.edit {
            Some(e) if e.condition => return Err(Error(format!("{ARCH}: reference edits are not part of this model (img2img, with a strength, is)"))),
            Some(e) => Some(e),
            None => None,
        };
        let d = self.defaults();
        let (w, h) = (Self::round(req.width.unwrap_or(d.width)), Self::round(req.height.unwrap_or(d.height)));
        let hw = (h / 16, w / 16);
        let ask = req.steps.unwrap_or(0);
        // the effort ladder: the LoRA below LADDER steps, a full guided run at or above it
        let distilled = self.turbo.is_some() && ask < LADDER;
        let steps = self.steps_for(req) as usize;
        let cfg = req.cfg.filter(|c| *c > 0.0).unwrap_or(if distilled { LORA_CFG } else { CFG });
        let levels = |n: usize| -> Vec<f32> { if distilled { shifted(n, LORA_SHIFT) } else { sched::sigmas(n, hw.0 * hw.1) } };
        let strength = img2img.map_or(1.0, |e| e.strength);
        if !(0.05..=1.0).contains(&strength) {
            return Err(Error(format!("strength {strength}: from 0.05 to 1")));
        }
        let sig = if strength < 1.0 { sched::tail(&levels(((steps as f32 / strength).round() as usize).max(steps)), steps) } else { levels(steps) };
        // the LoRAs of this run: the distilled one below the ladder, the entry's own, the request's
        let mut names = Vec::new();
        let mut deltas: Vec<Delta> = Vec::new();
        if distilled {
            names.push("turbo-lora".to_string());
            deltas.extend(self.turbo.iter().flatten().cloned());
        }
        if !self.base_loras.is_empty() {
            names.push("entry".to_string());
            deltas.extend(self.base_loras.iter().cloned());
        }
        for l in &req.loras {
            names.push(format!("{}:{}", l.path.display(), l.scale));
            deltas.extend(lora::read(&l.path, l.scale).map_err(Error)?);
        }
        self.use_loras(names, deltas)?;
        let k = ffi::api()?;
        let gpu = &self.nsd.gpu;
        let of = (sig.len() - 1) as u32;
        progress(Step { picture: 0, at: 0, of, seconds: t0.elapsed().as_secs_f64() })?;
        let start = match img2img {
            Some(e) if strength < 1.0 => {
                let r = e.image.rgb().resize_lanczos(w as u32, h as u32);
                let px: Vec<f32> = r.data.iter().map(|v| *v as f32 / 255.0 * 2.0 - 1.0).collect();
                Some(vae::pack(&self.vae.encode(&self.nsd, &px, h, w)?, h / 8, w / 8))
            }
            _ => None,
        };
        let t_text = Instant::now();
        let prompt = if req.prompt.trim().is_empty() { " " } else { req.prompt.trim() };
        let pos = self.encode(prompt)?;
        // true guidance above 1: against the negative prompt, " " when none was given
        let neg = crate::guidance_negative(cfg, req.negative.as_deref()).map(|n| self.encode(n)).transpose()?;
        self.nsd.trim()?;
        crate::stages::record(format!("get_learned_condition completed, taking {:.2}s", t_text.elapsed().as_secs_f64()));
        let texts: Vec<&dit::Text> = std::iter::once(pos.as_ref()).chain(neg.as_deref()).collect();
        let n = hw.0 * hw.1 * dit::LATENT;
        let mut out = Vec::new();
        for p in 0..req.n.max(1) {
            let mut nz = noise(req.seed + p as u64, n);
            if let Some(l0) = &start {
                let s0 = sig[0];
                for (x, l) in nz.iter_mut().zip(l0) {
                    *x = s0 * *x + (1.0 - s0) * l;
                }
            }
            let t_sample = Instant::now();
            let lat = DevBuf::from_f32(gpu, &nz)?;
            for i in 0..sig.len() - 1 {
                let v = self.dit.velocity(&self.nsd, &texts, &lat, hw, sig[i])?;
                let v0 = &v[0];
                if v.len() > 1 {
                    // true guidance: v_neg + cfg (v - v_neg), each token's norm put back to v's (the pipeline's)
                    // SAFETY: n floats each, rows of 64 values.
                    ffi::check(unsafe { (k.cfg_rescale)(gpu.raw(), v0.fp(), v[1].fp(), (n / dit::LATENT) as i64, dit::LATENT as i64, cfg, v0.fp()) }, "guidance")?;
                }
                // SAFETY: n floats each.
                ffi::check(unsafe { (k.axpy)(gpu.raw(), lat.fp(), v0.fp(), n as i64, sig[i + 1] - sig[i]) }, "euler step")?;
                gpu.sync()?;
                progress(Step { picture: p, at: i as u32 + 1, of, seconds: t0.elapsed().as_secs_f64() })?;
            }
            crate::stages::record(format!("generating 1 latent images completed, taking {:.2}s", t_sample.elapsed().as_secs_f64()));
            self.nsd.trim()?;
            let t_vae = Instant::now();
            let lv = vae::unpack(&lat.to_f32()?, h / 8, w / 8);
            let (rgb, pw, ph) = self.vae.decode(&self.nsd, &lv, h / 8, w / 8)?;
            self.nsd.trim()?;
            crate::stages::record(format!("decode_first_stage completed, taking {:.2}s", t_vae.elapsed().as_secs_f64()));
            out.push(Picture { width: pw as u32, height: ph as u32, channels: 3, data: rgb });
        }
        Ok(out)
    }

    fn report(&self) -> Vec<String> {
        vec![format!("{ARCH} in Rust and CUDA on {} ({} of {} blocks on the card; loaded {:.0} s ago{})", self.gpu_name, self.dit.resident, self.dit.blocks(),
                     self.loaded.elapsed().as_secs_f64(), if self.turbo.is_some() { ", Lightning LoRA below 20 steps" } else { "" })]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lightnings_sigmas_are_shifted_by_three() {
        let s = shifted(4, 3.0);
        assert_eq!(s.len(), 5);
        assert!((s[0] - 1.0).abs() < 1e-6);
        // t = 0.25: 3 / (3 + 3) = 0.5
        assert!((s[3] - 0.5).abs() < 1e-6);
        assert_eq!(s[4], 0.0);
    }

    #[test]
    fn the_canvas_rounds_to_sixteen() {
        assert_eq!(Q2512::round(1328), 1328);
        assert_eq!(Q2512::round(1000), 1008);
        assert_eq!(Q2512::round(100), 256);
    }
}
