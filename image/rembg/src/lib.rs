//! Background removal as an image engine (`image.remove_background`): a picture in, the picture as RGBA (its own RGB,
//! straight alpha) and its mask out, both at the picture's size.
//!
//! - `birefnet`: the foundation's remover (`nextcuda-remover`: ZhengPeng7/BiRefNet, Swin-L, float32 on the GPU), the
//!   one the 3D engines cut their pictures out with.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use nextcuda_core::Gpu;
use nextcuda_image::{Defaults, Error, ImageEngine, ImageKind, ImageRequest, LoadOptions, ModelFiles, Picture, Result, Sampler, Schedule, Step};
use nextcuda_remover::Remover;

pub const ARCH_BIREFNET: &str = "birefnet";

pub fn kind() -> ImageKind {
    ImageKind { archs: &[ARCH_BIREFNET], name: "background removal: BiRefNet (Rust + CUDA)", roles: &["remover"], load, options: &[] }
}

fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], _o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn ImageEngine>> {
    let gpu = gpus.first().cloned().ok_or_else(|| Error("a background remover needs a GPU".into()))?;
    let path: PathBuf = files.get("remover").or_else(|| files.values().next()).cloned()
        .ok_or_else(|| Error("birefnet: no remover file (the card names it)".into()))?;
    let r = Remover::load(&path, &gpu)?;
    log(format!("birefnet: loaded in {:.1} s on {}", r.load_seconds, gpu.name));
    Ok(Box::new(BiRefNet { load_secs: r.load_seconds, r: Mutex::new(r) }))
}

pub struct BiRefNet {
    r: Mutex<Remover>,
    load_secs: f64,
}

/// The picture's RGB with the mask as its alpha
pub fn with_alpha(pic: &Picture, mask: &Picture) -> Picture {
    let rgb = pic.rgb();
    Picture { width: pic.width, height: pic.height, channels: 4,
              data: rgb.data.chunks_exact(3).zip(&mask.data).flat_map(|(p, a)| [p[0], p[1], p[2], *a]).collect() }
}

impl ImageEngine for BiRefNet {
    fn arch(&self) -> &'static str {
        ARCH_BIREFNET
    }
    fn defaults(&self) -> Defaults {
        Defaults { width: 0, height: 0, steps: 1, cfg: 1.0, sampler: Sampler::Euler, schedule: Schedule::Simple, shift: 1.0 }
    }
    fn load_seconds(&self) -> f64 {
        self.load_secs
    }
    fn generate(&self, _req: &ImageRequest, _p: &mut dyn FnMut(Step) -> Result<()>) -> Result<Vec<Picture>> {
        Err(Error("birefnet removes backgrounds (image.remove_background); it does not draw".into()))
    }
    fn removes_background(&self) -> bool {
        true
    }
    fn remove_background(&self, pic: &Picture) -> Result<(Picture, Picture)> {
        let t0 = Instant::now();
        let r = self.r.lock().unwrap_or_else(|p| p.into_inner());
        let mask = r.mask(pic)?;
        let _ = t0;
        Ok((with_alpha(pic, &mask), mask))
    }
}
