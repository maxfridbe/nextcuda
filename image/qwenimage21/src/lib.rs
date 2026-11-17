//! Qwen-Image on NVIDIA GPUs, in Rust over the shared CUDA kernels and this crate's own: Qwen-Image 2.1 (`rust/`, the
//! 7B single-stream DiT kept on the card in the type its GGUF stores) and Qwen-Image 2512 (`q2512/`, the 20B MMDiT,
//! its blocks on the card while they fit and streamed from host memory after).
//!
//! The files are the GGUF transformer, the Qwen-VL text encoder as a GGUF (+ its vision projector for 2.1's edits), the
//! VAE; 2512's Lightning LoRA as `turbo-lora`.

pub mod describe;
pub mod ffi;
pub mod q2512;
pub mod rust;
pub mod selftest;
pub mod stages;

use std::sync::Arc;

use nextcuda_image::{Error, Gpu, ImageEngine, ImageKind, LoadOptions, ModelFiles, Result};

pub const ARCH: &str = "qwen-image-2.1";

/// The prompt true guidance steers away from: none at a scale of 1 or less (one pass a step), else the request's
/// negative prompt, or an empty one (" ", the model card's own empty negative) when none is given or it is blank -
/// a scale above 1 never needs a negative prompt (the published cfg 1-30 holds without one, as nextsycl's)
pub fn guidance_negative(cfg: f32, negative: Option<&str>) -> Option<&str> {
    (cfg > 1.0).then(|| negative.filter(|n| !n.trim().is_empty()).unwrap_or(" "))
}

/// The engine's kernels, each against host arithmetic on `gpu` (`nextcuda image selftest`): the worst error of each
pub fn selftest(gpu: &Arc<Gpu>) -> Result<Vec<f32>> {
    let mut v = selftest::run(gpu)?;
    // the shared diffusion kernels under it, when the library has them
    v.extend(nextcuda_diffusion::kernels_check::selftest(gpu)?);
    Ok(v)
}
/// The files it needs; `llm-vision` (the projector an edit's references go through) is read when there is one
pub const ROLES: &[&str] = &["transformer", "text-encoder", "vae"];

/// The Rust engine's options
pub const OPTIONS: &[nextcuda_image::EngineOption] = rust::OPTIONS;

pub fn kind() -> ImageKind {
    ImageKind { archs: &[ARCH], name: "Qwen-Image 2.1 (7B single-stream DiT, Qwen3-VL text encoder, RGBA VAE; Rust + CUDA)", roles: ROLES, load, options: OPTIONS }
}

/// Qwen-Image 2512 (the 20B MMDiT with Qwen2.5-VL): its own engine on the same kernels
pub const ARCH_2512: &str = q2512::ARCH;

pub fn kind_2512() -> ImageKind {
    ImageKind { archs: &[ARCH_2512], name: "Qwen-Image 2512 (20B MMDiT, Qwen2.5-VL text encoder; Rust + CUDA)", roles: ROLES, load: load_2512, options: q2512::OPTIONS }
}

fn load_2512(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn ImageEngine>> {
    let gpu = gpus.first().ok_or_else(|| Error("no GPU".into()))?;
    Ok(Box::new(q2512::Q2512::load(files, gpu, o, log)?))
}

/// Qwen-Image 2.1 on the Rust engine, on the first of `gpus`
fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn ImageEngine>> {
    let gpu = gpus.first().ok_or_else(|| Error("no GPU".into()))?;
    Ok(Box::new(rust::RustQwenImage21::load(files, gpu, o, log)?))
}

#[cfg(test)]
mod tests {
    use super::guidance_negative;

    #[test]
    fn guidance_above_1_runs_against_an_empty_negative_when_none_is_given() {
        // cfg 4 with no negative, an empty or a blank one: true guidance against the empty prompt, never a refusal
        for none in [None, Some(""), Some("   "), Some("\n")] {
            assert_eq!(guidance_negative(4.0, none), Some(" "), "{none:?}");
        }
        // a negative given is the one guided against
        assert_eq!(guidance_negative(4.0, Some("blurry, low quality")), Some("blurry, low quality"));
        // at 1 or below: one pass a step, the negative unread
        assert_eq!(guidance_negative(1.0, Some("blurry")), None);
        assert_eq!(guidance_negative(0.5, None), None);
        assert_eq!(guidance_negative(1.01, None), Some(" "));
    }
}
