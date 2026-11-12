//! Depth Anything V2 Small as an image engine: a picture's relative inverse depth (near = large), the depth map
//! `video.control` makes a depth control video of (H3's Fun ControlNet).
//!
//! - `prep.rs`: what transformers' DPTImageProcessor does to a picture (the size nearest 518 on the 14 grid that keeps
//!   its aspect, Pillow's bicubic, ImageNet's normalization), the position table resized to the patches, the map back
//!   at the picture's size (bilinear);
//! - `model.rs`: DINOv2 ViT-S/14 and the DPT head in float32 on the GPU - the shared diffusion kernels' products,
//!   norms, GELU and attention, the vision kernels' im2col and resizes; no kernels of its own.
//!
//! Its file: depth-anything/Depth-Anything-V2-Small-hf's model.safetensors (Apache-2.0; the Base and Large are
//! CC-BY-NC-4.0 and not served).

pub mod ffi;
pub mod model;
pub mod prep;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use nextcuda_core::Gpu;
use nextcuda_gguf::safetensors::SafeTensors;
use nextcuda_image::{Defaults, Error, ImageEngine, ImageKind, ImageRequest, LoadOptions, ModelFiles, Picture, Result, Sampler, Schedule, Step};

pub const ARCH: &str = "depth-anything-v2";

pub fn kind() -> ImageKind {
    ImageKind { archs: &[ARCH], name: "depth estimation: Depth Anything V2 Small (Rust + CUDA)", roles: &["model"], load, options: &[] }
}

/// The model's file (role `model`, else the only one)
fn file(files: &ModelFiles) -> Result<PathBuf> {
    files.get("model").or_else(|| files.values().next()).cloned().ok_or_else(|| Error("depth-anything-v2: no model file (the card names it)".into()))
}

fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], _o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn ImageEngine>> {
    let gpu = gpus.first().cloned().ok_or_else(|| Error("a depth model needs a GPU".into()))?;
    let e = DepthAnything::load(&file(files)?, &gpu)?;
    log(format!("depth-anything-v2: loaded in {:.1} s on {} ({:.0} MB of weights)", e.load_secs, gpu.name, e.net.lock().unwrap().bytes as f64 / 1e6));
    Ok(Box::new(e))
}

/// What one estimate's time went to, seconds
#[derive(Clone, Copy, Debug, Default)]
pub struct Times {
    /// the picture resized and normalized (host)
    pub prepare: f64,
    /// the network (upload, GPU, the map back)
    pub network: f64,
    /// the map resized to the picture (host)
    pub resize: f64,
}

pub struct DepthAnything {
    pub net: Mutex<model::Net>,
    pub load_secs: f64,
}

impl DepthAnything {
    pub fn load(path: &std::path::Path, gpu: &Arc<Gpu>) -> Result<DepthAnything> {
        let t0 = Instant::now();
        let st = SafeTensors::open(path).map_err(|e| Error(format!("{}: {}", path.display(), e.0)))?;
        let net = model::Net::load(&st, gpu)?;
        Ok(DepthAnything { net: Mutex::new(net), load_secs: t0.elapsed().as_secs_f64() })
    }

    /// The network's own output for a picture: the depth at the input's size (width, height: the 14 grid), and the
    /// times
    pub fn raw(&self, pic: &Picture) -> Result<(Vec<f32>, usize, usize, Times)> {
        let t0 = Instant::now();
        let (px, w, h) = prep::prepare(pic);
        let t1 = Instant::now();
        let d = self.net.lock().unwrap_or_else(|p| p.into_inner()).forward(&px, w, h)?;
        let t = Times { prepare: (t1 - t0).as_secs_f64(), network: t1.elapsed().as_secs_f64(), resize: 0.0 };
        Ok((d, w, h, t))
    }

    /// The depth at the picture's size, and the times
    pub fn timed(&self, pic: &Picture) -> Result<(Vec<f32>, Times)> {
        let (d, w, h, mut t) = self.raw(pic)?;
        let t0 = Instant::now();
        let out = prep::resize_bilinear(&d, w, h, pic.width as usize, pic.height as usize);
        t.resize = t0.elapsed().as_secs_f64();
        Ok((out, t))
    }
}

impl ImageEngine for DepthAnything {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn defaults(&self) -> Defaults {
        Defaults { width: 0, height: 0, steps: 1, cfg: 1.0, sampler: Sampler::Euler, schedule: Schedule::Simple, shift: 1.0 }
    }
    fn load_seconds(&self) -> f64 {
        self.load_secs
    }
    fn generate(&self, _req: &ImageRequest, _p: &mut dyn FnMut(Step) -> Result<()>) -> Result<Vec<Picture>> {
        Err(Error("depth-anything-v2 estimates depth (video.control with control depth); it does not draw".into()))
    }
    fn report(&self) -> Vec<String> {
        let n = self.net.lock().unwrap_or_else(|p| p.into_inner());
        vec![format!("weights {:.0} MB, working memory {:.0} MB", n.bytes as f64 / 1e6, n.arena_bytes() as f64 / 1e6)]
    }
    fn estimates_depth(&self) -> bool {
        true
    }
    fn depth(&self, pic: &Picture) -> Result<Vec<f32>> {
        self.timed(pic).map(|(d, _)| d)
    }
}
