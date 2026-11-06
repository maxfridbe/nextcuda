//! Background removal as an image engine (`image.remove_background`): a picture in, the picture as RGBA (its own RGB,
//! straight alpha) and its mask out, both at the picture's size.
//!
//! - `birefnet`: the foundation's remover (`nextcuda-remover`: ZhengPeng7/BiRefNet, Swin-L, float32 on the GPU), the
//!   one the 3D engines cut their pictures out with.
//! - `isnet`: IS-Net general-use as rembg runs it (`isnet.rs`), its preparation and mask rembg's own: Pillow's Lanczos
//!   to 1024, divided by the picture's largest value, less 0.5; the prediction scaled to its own [min, max], 8 bits,
//!   Lanczos back to the picture's size.
//!
//! `post_process` is rembg's post_process_mask for either: an opening (OpenCV's 3 x 3 ellipse, a cross), OpenCV's
//! 5 x 5 Gaussian (sigma 2, borders reflected 101), a threshold at 127 - the platform asked rembg for it.

pub mod check;
pub mod ffi;
pub mod isnet;
pub mod onnx;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use nextcuda_core::Gpu;
use nextcuda_image::{Defaults, Error, ImageEngine, ImageKind, ImageRequest, LoadOptions, ModelFiles, Picture, Result, Sampler, Schedule, Step};
use nextcuda_remover::Remover;

pub const ARCH_BIREFNET: &str = "birefnet";
pub const ARCH_ISNET: &str = "isnet";

pub fn kind() -> ImageKind {
    ImageKind { archs: &[ARCH_BIREFNET, ARCH_ISNET], name: "background removal: BiRefNet, IS-Net general-use (Rust + CUDA)", roles: &["remover"], load, options: &[] }
}

fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], _o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn ImageEngine>> {
    let gpu = gpus.first().cloned().ok_or_else(|| Error("a background remover needs a GPU".into()))?;
    if let Some(p) = files.get("isnet") {
        let t0 = Instant::now();
        let m = onnx::Model::open(p)?;
        let net = isnet::IsNet::load(&m, &gpu)?;
        let secs = t0.elapsed().as_secs_f64();
        log(format!("isnet: loaded in {secs:.1} s on {} ({:.0} MB)", gpu.name, net.bytes as f64 / 1e6));
        return Ok(Box::new(IsNetEngine { net: Mutex::new(net), load_secs: secs }));
    }
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
    fn remove_background(&self, pic: &Picture, post_process: bool) -> Result<(Picture, Picture)> {
        let r = self.r.lock().unwrap_or_else(|p| p.into_inner());
        let mut mask = r.mask(pic)?;
        if post_process {
            mask = post_process_mask(&mask);
        }
        Ok((with_alpha(pic, &mask), mask))
    }
}

pub struct IsNetEngine {
    net: Mutex<isnet::IsNet>,
    load_secs: f64,
}

impl IsNetEngine {
    /// rembg's input: Pillow's Lanczos to 1024 x 1024, divided by its largest value, less 0.5 (std 1), channels-last
    pub fn input(pic: &Picture) -> Vec<f32> {
        let r = pic.rgb().resize_lanczos(isnet::SIDE as u32, isnet::SIDE as u32);
        let mx = (*r.data.iter().max().unwrap_or(&0) as f32 / 255.0).max(1e-6);
        // rembg: im / 255 is not done; its array of bytes is divided by its own max (so 255 -> 1)
        let mxb = (mx * 255.0).max(1e-6);
        r.data.iter().map(|&v| v as f32 / mxb - 0.5).collect()
    }

    /// rembg's mask from the prediction: scaled to its [min, max], times 255 truncated, Lanczos to w x h
    pub fn mask_of(pred: &[f32], w: u32, h: u32) -> Picture {
        let (mi, ma) = pred.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
        let d = ma - mi;
        let m = Picture { width: isnet::SIDE as u32, height: isnet::SIDE as u32, channels: 1,
                          data: pred.iter().map(|&v| (((v - mi) / d) * 255.0) as u8).collect() };
        m.resize_lanczos(w, h)
    }
}

impl ImageEngine for IsNetEngine {
    fn arch(&self) -> &'static str {
        ARCH_ISNET
    }
    fn defaults(&self) -> Defaults {
        Defaults { width: 0, height: 0, steps: 1, cfg: 1.0, sampler: Sampler::Euler, schedule: Schedule::Simple, shift: 1.0 }
    }
    fn load_seconds(&self) -> f64 {
        self.load_secs
    }
    fn generate(&self, _req: &ImageRequest, _p: &mut dyn FnMut(Step) -> Result<()>) -> Result<Vec<Picture>> {
        Err(Error("isnet removes backgrounds (image.remove_background); it does not draw".into()))
    }
    fn removes_background(&self) -> bool {
        true
    }
    fn remove_background(&self, pic: &Picture, post_process: bool) -> Result<(Picture, Picture)> {
        let net = self.net.lock().unwrap_or_else(|p| p.into_inner());
        let pred = net.forward(&Self::input(pic))?;
        let mut mask = Self::mask_of(&pred, pic.width, pic.height);
        if post_process {
            mask = post_process_mask(&mask);
        }
        Ok((with_alpha(pic, &mask), mask))
    }
}

/// rembg's post_process: cv2.morphologyEx(MORPH_OPEN, the 3 x 3 ellipse - a cross), cv2.GaussianBlur 5 x 5 sigma 2
/// (BORDER_REFLECT_101, OpenCV's 8-bit fixed point), then 0 below 127, else 255
pub fn post_process_mask(m: &Picture) -> Picture {
    let (w, h) = (m.width as usize, m.height as usize);
    let at = |d: &[u8], x: i64, y: i64| -> Option<u8> { (x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h).then(|| d[y as usize * w + x as usize]) };
    let cross = [(0i64, 0i64), (1, 0), (-1, 0), (0, 1), (0, -1)];
    // erode then dilate; the border takes no part (OpenCV's morphology border value)
    let morph = |d: &[u8], min: bool| -> Vec<u8> {
        let mut o = vec![0u8; w * h];
        for y in 0..h as i64 {
            for x in 0..w as i64 {
                let vals = cross.iter().filter_map(|(dx, dy)| at(d, x + dx, y + dy));
                o[y as usize * w + x as usize] = if min { vals.min().unwrap() } else { vals.max().unwrap() };
            }
        }
        o
    };
    let opened = morph(&morph(&m.data, true), false);
    // OpenCV's 8-bit Gaussian: the 1-D kernel in 16-bit fixed point (its sum 256), rows then columns, each rounded
    let k: Vec<f64> = (-2i64..=2).map(|i| (-((i * i) as f64) / (2.0 * 4.0)).exp()).collect();
    let ks: f64 = k.iter().sum();
    let kf: Vec<i64> = k.iter().map(|v| (v / ks * 256.0).round() as i64).collect();
    let refl = |i: i64, n: i64| -> usize {
        let mut i = i;
        if i < 0 {
            i = -i;
        }
        if i >= n {
            i = 2 * n - 2 - i;
        }
        i.clamp(0, n - 1) as usize
    };
    let mut tmp = vec![0i64; w * h];
    for y in 0..h {
        for x in 0..w as i64 {
            tmp[y * w + x as usize] = (0..5).map(|t| kf[t] * opened[y * w + refl(x + t as i64 - 2, w as i64)] as i64).sum();
        }
    }
    let mut out = vec![0u8; w * h];
    for y in 0..h as i64 {
        for x in 0..w {
            let s: i64 = (0..5).map(|t| kf[t] * tmp[refl(y + t as i64 - 2, h as i64) * w + x]).sum();
            let v = ((s + (1 << 15)) >> 16).clamp(0, 255);
            out[y as usize * w + x] = if v < 127 { 0 } else { 255 };
        }
    }
    Picture { width: m.width, height: m.height, channels: 1, data: out }
}
