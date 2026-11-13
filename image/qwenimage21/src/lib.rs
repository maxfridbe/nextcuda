//! Qwen-Image on NVIDIA GPUs. Qwen-Image 2.1 is the Rust engine (`rust/`): the pipeline in Rust over the shared CUDA
//! kernels, the transformer kept on the card in the type its GGUF stores. Qwen-Image 2512 runs through
//! stable-diffusion.cpp in process (`native.rs`, `sd.rs`) until its own Rust engine replaces it.
//!
//! The files are the GGUF transformer, the Qwen-VL text encoder as a GGUF (+ its vision projector for edits), the VAE.

pub mod describe;
pub mod ffi;
pub mod gguffix;
pub mod native;
pub mod rust;
pub mod sd;
pub mod selftest;
pub mod stages;

use std::sync::Arc;

use nextcuda_image::{Error, Gpu, ImageEngine, ImageKind, LoadOptions, ModelFiles, Result};

pub const ARCH: &str = "qwen-image-2.1";

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

/// Qwen-Image 2512 (the 20B MMDiT with Qwen2.5-VL): the same library, its own architecture and ladder
pub const ARCH_2512: &str = "qwen-image-2512";

pub fn kind_2512() -> ImageKind {
    ImageKind { archs: &[ARCH_2512], name: "Qwen-Image 2512 (20B MMDiT, Qwen2.5-VL text encoder; stable-diffusion.cpp)", roles: ROLES, load: load_2512, options: native::OPTIONS }
}

fn load_2512(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn ImageEngine>> {
    if !sd::available() {
        return Err(Error(format!("{ARCH_2512}: the image kernel library has no stable-diffusion.cpp bridge (kernels/build.sh with SDCPP and SDLIB)")));
    }
    Ok(Box::new(native::load(native::QWEN_IMAGE_2512, files, gpus, o, log)?))
}

/// Qwen-Image 2.1 on the Rust engine, on the first of `gpus`
fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn ImageEngine>> {
    let gpu = gpus.first().ok_or_else(|| Error("no GPU".into()))?;
    Ok(Box::new(rust::RustQwenImage21::load(files, gpu, o, log)?))
}
