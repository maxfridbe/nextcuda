//! The shot upscaler (the movie maker's ltx2_upscale: diffusers' LTX2LatentUpsamplePipeline): a finished clip back
//! through the video VAE's encoder, its latents doubled in space by the latent upsampler - which adds the detail the
//! model knows - and decoded at twice the size; the sound copied. A long clip goes in windows of 121 frames (each
//! 8k + 1; a tail under 9 frames is dropped, as there).
//!
//! ```text
//!   upsampler (on the latents as the encoder gives them, not normalized):
//!     conv3d 128 -> 1024, GroupNorm(32), silu; 4 resnets; per frame conv2d 1024 -> 4096 and a 2 x 2 pixel shuffle;
//!     4 resnets; conv3d 1024 -> 128
//!   resnet: silu(GN(conv(silu(GN(conv(x))))) + x)       (zero padding in every axis; GroupNorm over the whole clip)
//! ```
//! The encoder's latents are the distribution's means (the pipeline draws a sample; its spread is negligible), the
//! tiles diffusers' defaults there (512 pixels, a stride of 448) for the encode and the decode.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use nextcuda_core::{DevBuf, Error, Gpu, Result};
use nextcuda_diffusion::kernels::{Dt, Nsd};
use nextcuda_gguf::safetensors::SafeTensors;
use serde_json::{json, Value};

use crate::ffi;
use crate::vae::{Conv, Ctx, Vol};

fn ge(e: nextcuda_gguf::Error) -> Error {
    Error(e.0)
}

struct Gn {
    w: DevBuf,
    b: DevBuf,
}

impl Gn {
    fn load(st: &SafeTensors, p: &str, gpu: &Arc<Gpu>) -> Result<Gn> {
        Ok(Gn { w: DevBuf::from_f32(gpu, &st.f32(&format!("{p}.weight")).map_err(ge)?)?, b: DevBuf::from_f32(gpu, &st.f32(&format!("{p}.bias")).map_err(ge)?)? })
    }
}

struct Block {
    c1: Conv,
    n1: Gn,
    c2: Conv,
    n2: Gn,
}

pub struct Upsampler {
    gpu: Arc<Gpu>,
    nsd: Nsd,
    conv_in: Conv,
    norm_in: Gn,
    pre: Vec<Block>,
    up: Conv,
    post: Vec<Block>,
    conv_out: Conv,
}

/// a 3^3 conv zero-padded in every axis (torch's Conv3d padding=1)
fn conv3(c: &Ctx, cv: &Conv, x: &Vol) -> Result<Vol> {
    let o = c.vol(x.t, x.h, x.w, cv.co)?;
    ffi::conv3d(c.nsd.raw(), x.d.ptr(), (x.t, x.h, x.w), x.c, cv.w_ptr(), cv.co, 3, cv.b_ptr(), o.d.ptr())?;
    Ok(o)
}

fn gn(c: &Ctx, n: &Gn, x: &Vol, res: Option<&Vol>, silu: bool) -> Result<Vol> {
    let o = c.vol(x.t, x.h, x.w, x.c)?;
    ffi::group_norm(c.gpu, x.d.ptr(), x.t * x.h * x.w, x.c, 32, n.w.fp(), n.b.fp(), 1e-5, res.map_or(std::ptr::null(), |r| r.d.ptr().cast_const()), silu, o.d.ptr())?;
    Ok(o)
}

impl Upsampler {
    pub fn load(path: &Path, gpu: &Arc<Gpu>) -> Result<Upsampler> {
        let st = SafeTensors::open(path).map_err(ge)?;
        let blk = |p: &str| -> Result<Block> {
            Ok(Block { c1: Conv::load(&st, &format!("{p}.conv1"), gpu)?, n1: Gn::load(&st, &format!("{p}.norm1"), gpu)?, c2: Conv::load(&st, &format!("{p}.conv2"), gpu)?,
                       n2: Gn::load(&st, &format!("{p}.norm2"), gpu)? })
        };
        Ok(Upsampler {
            conv_in: Conv::load(&st, "initial_conv", gpu)?,
            norm_in: Gn::load(&st, "initial_norm", gpu)?,
            pre: (0..4).map(|i| blk(&format!("res_blocks.{i}"))).collect::<Result<_>>()?,
            up: Conv::load(&st, "upsampler.0", gpu)?,
            post: (0..4).map(|i| blk(&format!("post_upsample_res_blocks.{i}"))).collect::<Result<_>>()?,
            conv_out: Conv::load(&st, "final_conv", gpu)?,
            nsd: Nsd::new(gpu)?,
            gpu: gpu.clone(),
        })
    }

    fn block(&self, c: &Ctx, b: &Block, x: Vol) -> Result<Vol> {
        let h = gn(c, &b.n1, &conv3(c, &b.c1, &x)?, None, true)?;
        gn(c, &b.n2, &conv3(c, &b.c2, &h)?, Some(&x), true)
    }

    /// Latents [f, h, w, 128] (not normalized) -> [f, 2h, 2w, 128]
    pub fn run(&self, z: &[f32], f: usize, h: usize, w: usize) -> Result<Vec<f32>> {
        let c = Ctx { gpu: &self.gpu, nsd: &self.nsd };
        let x = c.vol(f, h, w, 128)?;
        x.d.write(0, &z.iter().flat_map(|v| crate::quant::f16_bits(*v).to_le_bytes()).collect::<Vec<u8>>())?;
        let mut x = gn(&c, &self.norm_in, &conv3(&c, &self.conv_in, &x)?, None, true)?;
        for b in &self.pre {
            x = self.block(&c, b, x)?;
        }
        // per frame: conv2d to 4 x 1024, the 2 x 2 shuffle (channel c * 4 + i * 2 + j -> (2h + i, 2w + j))
        let y = c.vol(x.t, x.h, x.w, self.up.co)?;
        self.nsd.conv2d(x.d.ptr(), Dt::F16, x.t, x.h, x.w, x.c, self.up.w_ptr(), self.up.co, 3, self.up.b_ptr().cast(), y.d.ptr())?;
        let mut x = c.vol(x.t, 2 * x.h, 2 * x.w, self.up.co / 4)?;
        ffi::d2s(&self.gpu, y.d.ptr(), y.t, y.h, y.w, self.up.co / 4, (1, 2, 2), 0, x.d.ptr())?;
        for b in &self.post {
            x = self.block(&c, b, x)?;
        }
        let o = conv3(&c, &self.conv_out, &x)?;
        let n = o.t * o.h * o.w * o.c;
        let tmp = DevBuf::f32(&self.gpu, n)?;
        ffi::to_float(&self.gpu, o.d.ptr(), tmp.fp(), n, 1.0)?;
        self.nsd.wait()?;
        tmp.to_f32()
    }
}

/// The diffusers VAE's default tiles (enable_tiling() with no sizes): 512 pixels, a stride of 448
pub const TILE_PX: (usize, usize) = (512, 448);
/// frames a pass
pub const WINDOW: usize = crate::rules::WINDOW;

/// The upscale job: {source: an mp4, out: an mp4, scale: 2}
pub fn upscale(m: &crate::Ltx2, spec: &Value, log: &mut dyn FnMut(String), progress: &mut dyn FnMut(usize, usize)) -> Result<Value> {
    let t_all = Instant::now();
    let src = std::path::PathBuf::from(spec["source"].as_str().ok_or("upscale: source (an .mp4)")?);
    let out = std::path::PathBuf::from(spec["out"].as_str().ok_or("upscale: out (an .mp4)")?);
    if spec["scale"].as_f64().is_some_and(|s| s != 2.0) {
        log("ltx2: the latent upsampler only doubles; x2 is what this is".into());
    }
    let (px, total, w, h, fps, has_audio) = crate::media::read_video(&src)?;
    crate::rules::check_source(&src.display().to_string(), total, w, h)?;
    let gpu = m.freest().clone();
    let enc = crate::vae::Encoder::load(m.file("vae")?, &gpu)?;
    let ups = Upsampler::load(m.file("upscaler")?, &gpu)?;
    let dec = crate::vae::Decoder::load(m.file("vae")?, &gpu)?;
    let frame = w * h * 3;
    let (bw, bh) = (2 * w, 2 * h);
    let mut big: Vec<f32> = Vec::new();
    let mut done = 0;
    for (start, legal) in crate::rules::windows(total, WINDOW) {
        let x: Vec<f32> = px[start * frame..(start + legal) * frame].iter().map(|v| v * 2.0 - 1.0).collect();
        let z = enc.encode_tiled(&x, legal, h, w, TILE_PX)?;
        let lf = (legal - 1) / 8 + 1;
        let z2 = ups.run(&z, lf, h / 32, w / 32)?;
        let mut p = |_: usize, _: usize| {};
        let v = dec.decode_raw(&z2, lf, 2 * h / 32, 2 * w / 32, (TILE_PX.0 / 32, TILE_PX.1 / 32), &mut p)?;
        big.extend_from_slice(&v[..legal * bw * bh * 3]);
        done += legal;
        progress(done, total);
    }
    let secs = t_all.elapsed().as_secs_f64();
    crate::media::replace_picture(&src, &out, &big, done, bh, bw, fps, has_audio)?;
    log(format!("ltx2: {done} frames {w}x{h} -> {bw}x{bh} in {secs:.0} s"));
    Ok(json!({"out": out, "engine": "ltx2_upscale", "scale": 2, "frames": done, "width": bw, "height": bh, "upscale_secs": (secs * 10.0).round() / 10.0,
              "dropped_frames": total - done}))
}
