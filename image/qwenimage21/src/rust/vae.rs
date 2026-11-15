//! The decoder: Qwen-Image 2.1's VAE (diffusers' AutoencoderKLQwenImage21, a Wan 2.2-style decoder with 2-D kernels
//! and 64 latent channels, 16x down) from latents to RGBA. Everything is channels-last half on the GPU; one picture is
//! one frame, so the causal video parts reduce to their first chunk (the time convolutions are skipped, each
//! duplicate-upsample shortcut keeps its last time copy).
//!
//! ```text
//!   z [h, w, 64] (latents * std + mean) -> 1x1 conv -> 3x3 conv (1152)
//!   middle: resnet, attention (one head over all pixels), resnet
//!   5 up blocks (1152, 1152, 576, 288, 144): 3 resnets, then (blocks 0-3) nearest 2x + 3x3 conv, plus the block
//!     input's duplicate-upsample shortcut
//!   rms norm, silu, 3x3 conv -> 4 channels (RGBA) in [-1, 1]
//! resnet: x + conv(silu(norm(conv(silu(norm(x)))))), a 1x1 conv on the shortcut when the width changes
//! ```
//!
//! The file names its tensors as Wan / ComfyUI do (decoder.middle.0.residual.2, decoder.upsamples.3.upsamples.0, ...).

use std::path::Path;
use std::sync::Arc;

use nextcuda_core::{DevBuf, Error, Gpu, Result};
use nextcuda_diffusion::kernels::{none, Dt, Nsd};
use nextcuda_gguf::safetensors::SafeTensors;

use super::prof;
use crate::ffi;

/// The largest side, in latents, the decoder's up path runs in one piece (a 1024 x 1024 picture); a larger picture
/// goes in tiles of this side
const TILE: usize = 64;
/// Above TILE x TILE latents the first SPLIT up blocks run over the whole picture (at most 4x the latents: 512 x 512
/// at 2048 x 2048, 1152 and 576 channels), the rest a tile at a time
const SPLIT: usize = 2;
/// The reach of the up blocks after SPLIT, in pixels of the scale they start at: each output pixel reads pixels at
/// most this far away (3 x 3 convolutions: 6 at that scale, 1 + 6 at the next, 1 + 6 + 1 at the last - 11.5 in all,
/// the nearest upsamples' rounding within the 13th), so a tile's margin of this many gives its core exactly the pixels
/// of the whole picture
const SPLIT_HALO: usize = 13;

/// F.normalize's floor, as an epsilon on the mean square: never reached by real activations
const EPS: f32 = 1e-24;

/// A convolution: half weights [Co, Ci, k, k] (or [Co, k, k, Ci] when `taps_last`: the silo's 3 x 3), float32 bias
struct Conv {
    w: DevBuf,
    b: DevBuf,
    ci: usize,
    co: usize,
    k: usize,
    taps_last: bool,
}

struct Resnet {
    norm1: DevBuf,
    conv1: Conv,
    norm2: DevBuf,
    conv2: Conv,
    shortcut: Option<Conv>,
}

struct Up {
    resnets: Vec<Resnet>,
    /// the 2x upsample's conv (blocks 0-3)
    resample: Option<Conv>,
    /// the shortcut's time copies (2: a block that also upsamples in time; 1: space only), when it upsamples
    ft: Option<i32>,
    ci: usize,
}

/// The middle's attention block: one head of every channel
struct Att {
    norm: DevBuf,
    qkv: DevBuf,
    qkv_b: DevBuf,
    proj: DevBuf,
    proj_b: DevBuf,
}

/// An encoder down block: resnets, then (blocks 0-3) the zero-padded stride-2 conv; its input's average shortcut
/// (2x2 in space, and (blocks 1-3) a zero frame in time) added
struct Down {
    resnets: Vec<Resnet>,
    resample: Option<Conv>,
    ft: i32,
    ci: usize,
    co: usize,
}

/// The encoder (edits: the condition pictures to latents): RGBA [H, W, 4] in [-1, 1] -> conv_in (96) -> 5 down blocks
/// (96, 192, 384, 768, 768; 16x down) -> middle (resnet, attention, resnet) -> rms norm, silu, 3x3 conv (128) ->
/// 1x1 conv; the mean is the first 64 channels
struct Encoder {
    conv_in: Conv,
    downs: Vec<Down>,
    mid: [Resnet; 2],
    att: Att,
    out_norm: DevBuf,
    head: Conv,
    quant: Conv,
}

pub struct Vae {
    gpu: Arc<Gpu>,
    post_quant: Conv,
    conv_in: Conv,
    mid: [Resnet; 2],
    att: Att,
    enc: Option<Encoder>,
    ups: Vec<Up>,
    out_norm: DevBuf,
    conv_out: Conv,
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
    pub z: usize,
}

fn ge(e: nextcuda_gguf::Error) -> Error {
    Error(e.0)
}

/// Wan 2.2's latent statistics for the 64-channel VAE (diffusers' vae/config.json: latents_mean, latents_std), read
/// from the model's config when it is beside the file, else these
fn stats(file: &Path) -> Option<(Vec<f32>, Vec<f32>)> {
    let dir = file.parent()?;
    for name in ["config.json", "vae_config.json"] {
        if let Ok(s) = std::fs::read_to_string(dir.join(name)) {
            let v: serde_json::Value = serde_json::from_str(&s).ok()?;
            let f = |k: &str| v[k].as_array().map(|a| a.iter().filter_map(|x| x.as_f64()).map(|x| x as f32).collect::<Vec<f32>>());
            if let (Some(m), Some(s)) = (f("latents_mean"), f("latents_std")) {
                return Some((m, s));
            }
        }
    }
    None
}

impl Vae {
    /// The decoder from its file; `stats`: the latents' per-channel (mean, std)
    pub fn load(file: &Path, nsd: &Nsd, latent_stats: (Vec<f32>, Vec<f32>), log: &mut dyn FnMut(String)) -> Result<Vae> {
        let t0 = std::time::Instant::now();
        let gpu = nsd.gpu.clone();
        let st = SafeTensors::open(file).map_err(ge)?;
        // the silo's 3 x 3 convolutions read the weights [Co, 3, 3, Ci] (an implicit GEMM: a tap's channels together)
        let silo_conv = ffi::silo().is_some_and(|s| s.conv3x3.is_some());
        let half_as = |name: &str, taps_last: bool| -> Result<(DevBuf, Vec<u64>)> {
            let t = st.need(name).map_err(ge)?;
            let mut raw = st.read(t).map_err(ge)?;
            if taps_last {
                let (co, ci, kk) = (t.shape[0] as usize, t.shape[1] as usize, t.shape[2..].iter().product::<u64>() as usize);
                let mut p = vec![0u8; raw.len()];
                for o in 0..co {
                    for c in 0..ci {
                        for tap in 0..kk {
                            let (src, dst) = (((o * ci + c) * kk + tap) * 2, ((o * kk + tap) * ci + c) * 2);
                            p[dst..dst + 2].copy_from_slice(&raw[src..src + 2]);
                        }
                    }
                }
                raw = p;
            }
            let n = t.elements() as usize;
            let src = DevBuf::new(&gpu, raw.len())?;
            src.write(0, &raw)?;
            let out = DevBuf::new(&gpu, n * 2)?;
            let code = match t.dtype.as_str() {
                "BF16" => 30,
                other => return Err(Error(format!("{name}: {other} weights are not read yet"))),
            };
            nsd.dequant(src.ptr(), code, n, out.ptr(), Dt::F16)?;
            nsd.wait()?;
            Ok((out, t.shape.clone()))
        };
        let half = |name: &str| half_as(name, false);
        let vec = |name: &str| -> Result<DevBuf> { DevBuf::from_f32(&gpu, &st.f32(name).map_err(ge)?) };
        // `plain`: the stride-2 convolutions of the encoder, which the shared kernel runs
        let conv_as = |p: &str, plain: bool| -> Result<Conv> {
            let t = st.need(&format!("{p}.weight")).map_err(ge)?;
            // [Co, Ci, 3, 3], or the causal video conv's [Co, Ci, 1, 3, 3]
            let sh = &t.shape;
            let s3 = sh.len() >= 4 && sh[2..].iter().product::<u64>() == 9 && sh[sh.len() - 1] == 3 && sh[1] % 8 == 0 && sh[0] % 2 == 0;
            let taps_last = silo_conv && !plain && s3;
            let (w, shape) = half_as(&format!("{p}.weight"), taps_last)?;
            let (co, ci, k) = (shape[0] as usize, shape[1] as usize, *shape.last().unwrap_or(&1) as usize);
            Ok(Conv { w, b: vec(&format!("{p}.bias"))?, ci, co, k, taps_last })
        };
        let conv = |p: &str| conv_as(p, false);
        let resnet = |p: &str| -> Result<Resnet> {
            Ok(Resnet {
                norm1: vec(&format!("{p}.residual.0.gamma"))?,
                conv1: conv(&format!("{p}.residual.2"))?,
                norm2: vec(&format!("{p}.residual.3.gamma"))?,
                conv2: conv(&format!("{p}.residual.6"))?,
                shortcut: if st.tensor(&format!("{p}.shortcut.weight")).is_some() { Some(conv(&format!("{p}.shortcut"))?) } else { None },
            })
        };
        let mut ups = Vec::new();
        for i in 0.. {
            let p = format!("decoder.upsamples.{i}.upsamples");
            if st.tensor(&format!("{p}.0.residual.0.gamma")).is_none() {
                break;
            }
            let resnets = (0..3).map(|j| resnet(&format!("{p}.{j}"))).collect::<Result<Vec<_>>>()?;
            let ci = resnets[0].conv1.ci;
            let up = if st.tensor(&format!("{p}.3.resample.1.weight")).is_some() {
                let time = st.tensor(&format!("{p}.3.time_conv.weight")).is_some();
                Up { resnets, resample: Some(conv(&format!("{p}.3.resample.1"))?), ft: Some(if time { 2 } else { 1 }), ci }
            } else {
                Up { resnets, resample: None, ft: None, ci }
            };
            ups.push(up);
        }
        let att = |p: &str| -> Result<Att> {
            Ok(Att { norm: vec(&format!("{p}.norm.gamma"))?, qkv: half(&format!("{p}.to_qkv.weight"))?.0, qkv_b: vec(&format!("{p}.to_qkv.bias"))?,
                     proj: half(&format!("{p}.proj.weight"))?.0, proj_b: vec(&format!("{p}.proj.bias"))? })
        };
        // the encoder, when the file has it
        let enc = if st.tensor("encoder.conv1.weight").is_some() {
            let mut downs = Vec::new();
            for i in 0.. {
                let p = format!("encoder.downsamples.{i}.downsamples");
                if st.tensor(&format!("{p}.0.residual.0.gamma")).is_none() {
                    break;
                }
                let resnets = (0..2).map(|j| resnet(&format!("{p}.{j}"))).collect::<Result<Vec<_>>>()?;
                let (ci, co) = (resnets[0].conv1.ci, resnets[1].conv2.co);
                let resample = if st.tensor(&format!("{p}.2.resample.1.weight")).is_some() { Some(conv_as(&format!("{p}.2.resample.1"), true)?) } else { None };
                let ft = if st.tensor(&format!("{p}.2.time_conv.weight")).is_some() { 2 } else { 1 };
                downs.push(Down { resnets, resample, ft, ci, co });
            }
            Some(Encoder {
                conv_in: conv("encoder.conv1")?,
                downs,
                mid: [resnet("encoder.middle.0")?, resnet("encoder.middle.2")?],
                att: att("encoder.middle.1")?,
                out_norm: vec("encoder.head.0.gamma")?,
                head: conv("encoder.head.2")?,
                quant: conv("conv1")?,
            })
        } else {
            None
        };
        let post_quant = conv("conv2")?;
        let z = post_quant.ci;
        let (mean, std) = stats(file).unwrap_or(latent_stats);
        if mean.len() != z || std.len() != z {
            return Err(Error(format!("the VAE takes {z} latent channels; the statistics have {} / {}", mean.len(), std.len())));
        }
        let vae = Vae {
            gpu: gpu.clone(),
            post_quant,
            conv_in: conv("decoder.conv1")?,
            mid: [resnet("decoder.middle.0")?, resnet("decoder.middle.2")?],
            att: att("decoder.middle.1")?,
            enc,
            ups,
            out_norm: vec("decoder.head.0.gamma")?,
            conv_out: conv("decoder.head.2")?,
            mean,
            std,
            z,
        };
        log(format!("qwen-image 2.1 vae: {} up blocks on {} in {:.1} s", vae.ups.len(), gpu.name, t0.elapsed().as_secs_f64()));
        Ok(vae)
    }

    fn conv(&self, nsd: &Nsd, c: &Conv, x: &DevBuf, h: usize, w: usize) -> Result<DevBuf> {
        self.conv_up(nsd, c, x, h, w, 1)
    }

    /// The convolution of x [h, w, ci]'s nearest `up` x upsample (1 or 2): [h * up, w * up, co]. The silo's implicit
    /// GEMM reads the upsample from x as it gathers (no 4x buffer); the shared kernel needs it made first.
    fn conv_up(&self, nsd: &Nsd, c: &Conv, x: &DevBuf, h: usize, w: usize, up: usize) -> Result<DevBuf> {
        let (ho, wo) = (h * up, w * up);
        let out = DevBuf::new(&self.gpu, ho * wo * c.co * 2)?;
        if c.taps_last {
            let f = ffi::silo().and_then(|s| s.conv3x3).ok_or_else(|| Error("the silo's convolution went away".into()))?;
            // SAFETY: x holds h x w x ci halves, out (h up) x (w up) x co; the weights [co, 3, 3, ci].
            ffi::check(unsafe { f(self.gpu.raw(), x.ptr(), h as i64, w as i64, c.ci as i64, up as i32, c.w.ptr(), c.co as i64, c.b.fp(), out.ptr(),
                                  ffi::f16acc() as i32) }, "silo conv3x3")?;
        } else if up == 1 {
            nsd.conv2d(x.ptr(), Dt::F16, 1, h, w, c.ci, c.w.ptr(), c.co, c.k, c.b.ptr(), out.ptr())?;
        } else {
            let k = ffi::api()?;
            let u = DevBuf::new(&self.gpu, ho * wo * c.ci * 2)?;
            // SAFETY: x holds h x w x ci halves, u four times that.
            ffi::check(unsafe { (k.up2)(self.gpu.raw(), x.ptr(), h as i64, w as i64, c.ci as i64, u.ptr()) }, "upsample")?;
            nsd.conv2d(u.ptr(), Dt::F16, 1, ho, wo, c.ci, c.w.ptr(), c.co, c.k, c.b.ptr(), out.ptr())?;
            // `u` is freed on return: the work on it first
            nsd.wait()?;
        }
        prof::mark(nsd, if c.k == 1 { "vae conv 1x1" } else { "vae conv 3x3" })?;
        Ok(out)
    }

    /// silu(rms norm(x) * gamma), a new buffer
    fn norm_silu(&self, nsd: &Nsd, x: &DevBuf, px: usize, c: usize, gamma: &DevBuf) -> Result<DevBuf> {
        let k = ffi::api()?;
        let out = DevBuf::new(&self.gpu, px * c * 2)?;
        nsd.rms_norm_mod(x.ptr(), Dt::F16, px, c, gamma.ptr(), EPS, none(), none(), none(), out.ptr(), Dt::F16)?;
        // SAFETY: px x c halves.
        ffi::check(unsafe { (k.silu)(self.gpu.raw(), out.ptr(), (px * c) as i64) }, "silu")?;
        prof::mark(nsd, "vae norm + silu")?;
        Ok(out)
    }

    fn resnet(&self, nsd: &Nsd, r: &Resnet, x: DevBuf, h: usize, w: usize) -> Result<DevBuf> {
        let px = h * w;
        let a = self.norm_silu(nsd, &x, px, r.conv1.ci, &r.norm1)?;
        let a = self.conv(nsd, &r.conv1, &a, h, w)?;
        let a = self.norm_silu(nsd, &a, px, r.conv1.co, &r.norm2)?;
        let out = self.conv(nsd, &r.conv2, &a, h, w)?;
        let short = match &r.shortcut {
            Some(c) => self.conv(nsd, c, &x, h, w)?,
            None => x,
        };
        nsd.gate_add(out.ptr(), Dt::F16, px, r.conv2.co, short.ptr(), Dt::F16, none(), none())?;
        // the temporaries are freed on return: the work on them first (a free does not wait for the queue)
        nsd.wait()?;
        Ok(out)
    }

    /// RGBA [H, W, 4] bytes from latents float32 [h * w, z] (as the denoiser leaves them, before the statistics).
    /// The middle (its attention sees every latent) runs over the whole picture; the up path is local, so above `TILE`
    /// x `TILE` latents its first `SPLIT` blocks run whole and the rest a tile at a time, each with a margin of their
    /// reach (`SPLIT_HALO`): the same pixels, in the memory of a 1024 x 1024 picture (2048 x 2048 beside the denoiser on
    /// 12 GB).
    pub fn decode(&self, nsd: &Nsd, lat: &[f32], h: usize, w: usize) -> Result<(Vec<u8>, usize, usize)> {
        let gpu = &self.gpu;
        let k = ffi::api()?;
        let z = self.z;
        prof::reset_clock(nsd)?;
        let zl: Vec<f32> = lat.iter().enumerate().map(|(i, v)| v * self.std[i % z] + self.mean[i % z]).collect();
        let zf = DevBuf::from_f32(gpu, &zl)?;
        let x = DevBuf::new(gpu, zl.len() * 2)?;
        // SAFETY: h x w x z values each way.
        ffi::check(unsafe { (k.to_half)(gpu.raw(), zf.fp(), x.ptr(), zl.len() as i64) }, "to half")?;
        drop(zf);
        let x = self.conv(nsd, &self.post_quant, &x, h, w)?;
        let x = self.conv(nsd, &self.conv_in, &x, h, w)?;
        let x = self.resnet(nsd, &self.mid[0], x, h, w)?;
        let x = self.attention(nsd, &self.att, x, h, w, self.conv_in.co)?;
        let x = self.resnet(nsd, &self.mid[1], x, h, w)?;
        let s = self.scale();
        let (ph, pw) = (h * s, w * s);
        if h <= TILE && w <= TILE {
            let (x, hh, ww) = self.up_blocks(nsd, x, h, w, 0..self.ups.len())?;
            let out = self.head(nsd, x, hh, ww)?;
            return Ok((out, pw, ph));
        }
        // a larger picture: the first up blocks (few pixels, many channels) over the whole of it, then the rest - whose
        // reach is SPLIT_HALO pixels of that scale - a tile at a time with that margin, each tile at most TILE latents'
        // worth a side there (a 1024 x 1024 picture's memory): far less margin than tiling at the latents' scale
        let split = SPLIT.min(self.ups.len());
        let (x, h, w) = self.up_blocks(nsd, x, h, w, 0..split)?;
        let c = self.ups.get(split).map_or(self.conv_out.ci, |u| u.ci);
        let s = ph / h;
        let grown = 1usize << self.ups[..split].iter().filter(|u| u.resample.is_some()).count();
        let side = grown * TILE - 2 * SPLIT_HALO;
        let (ny, nx) = (h.div_ceil(side), w.div_ceil(side));
        let (cy_n, cx_n) = (h.div_ceil(ny), w.div_ceil(nx));
        let mut out = vec![0u8; ph * pw * 4];
        for y0 in (0..h).step_by(cy_n) {
            for x0 in (0..w).step_by(cx_n) {
                let (y1, x1) = ((y0 + cy_n).min(h), (x0 + cx_n).min(w));
                // the tile with its margin, cut from the whole picture at this scale
                let (ya, yb, xa, xb) = (y0.saturating_sub(SPLIT_HALO), (y1 + SPLIT_HALO).min(h), x0.saturating_sub(SPLIT_HALO), (x1 + SPLIT_HALO).min(w));
                let (th, tw) = (yb - ya, xb - xa);
                let t = DevBuf::new(gpu, th * tw * c * 2)?;
                t.copy_2d(0, tw * c * 2, &x, (ya * w + xa) * c * 2, w * c * 2, tw * c * 2, th)?;
                let (t, oh, ow) = self.up_blocks(nsd, t, th, tw, split..self.ups.len())?;
                let px = self.head(nsd, t, oh, ow)?;
                // its core into the picture
                let (cy, cx, ch, cw) = ((y0 - ya) * s, (x0 - xa) * s, (y1 - y0) * s, (x1 - x0) * s);
                for r in 0..ch {
                    let src = ((cy + r) * tw * s + cx) * 4;
                    let dst = ((y0 * s + r) * pw + x0 * s) * 4;
                    out[dst..dst + cw * 4].copy_from_slice(&px[src..src + cw * 4]);
                }
            }
        }
        Ok((out, pw, ph))
    }

    /// Pixels per latent: 2 for each up block that upsamples
    fn scale(&self) -> usize {
        1 << self.ups.iter().filter(|u| u.resample.is_some()).count()
    }

    /// Up blocks `blocks` from x [h, w, the first's channels] half: (their output, its size)
    fn up_blocks(&self, nsd: &Nsd, mut x: DevBuf, h: usize, w: usize, blocks: std::ops::Range<usize>) -> Result<(DevBuf, usize, usize)> {
        let gpu = &self.gpu;
        let k = ffi::api()?;
        let (mut h, mut w) = (h, w);
        for up in &self.ups[blocks] {
            // the block's input, for its shortcut
            let keep = match up.ft {
                Some(_) => {
                    let b = DevBuf::new(gpu, h * w * up.ci * 2)?;
                    b.copy_within(0, &x, 0, h * w * up.ci * 2)?;
                    Some(b)
                }
                None => None,
            };
            for r in &up.resnets {
                x = self.resnet(nsd, r, x, h, w)?;
            }
            if let (Some(c), Some(ft), Some(inp)) = (&up.resample, up.ft, keep) {
                let co = c.co;
                let y = self.conv_up(nsd, c, &x, h, w, 2)?;
                drop(x);
                x = y;
                prof::mark(nsd, "vae up 2x")?;
                // SAFETY: inp holds h x w x ci halves, x 2h x 2w x co.
                ffi::check(unsafe { (k.dupup_add)(gpu.raw(), inp.ptr(), h as i64, w as i64, up.ci as i64, co as i64, ft, x.ptr()) }, "shortcut")?;
                nsd.wait()?;
                h *= 2;
                w *= 2;
            }
        }
        Ok((x, h, w))
    }

    /// The output norm and conv on the up path's output x [h, w, c]: RGBA bytes
    fn head(&self, nsd: &Nsd, x: DevBuf, h: usize, w: usize) -> Result<Vec<u8>> {
        let gpu = &self.gpu;
        let k = ffi::api()?;
        let c = self.conv_out.ci;
        let a = self.norm_silu(nsd, &x, h * w, c, &self.out_norm)?;
        drop(x);
        let img = self.conv(nsd, &self.conv_out, &a, h, w)?;
        let rgba = DevBuf::new(gpu, h * w * 4)?;
        // SAFETY: h x w x 4 halves in, as many bytes out.
        ffi::check(unsafe { (k.to_rgba8)(gpu.raw(), img.ptr(), h as i64, w as i64, rgba.ptr().cast()) }, "to rgba")?;
        nsd.wait()?;
        gpu.sync()?;
        let mut out = vec![0u8; h * w * 4];
        rgba.read(0, &mut out)?;
        Ok(out)
    }

    /// The middle's attention: x + proj(attention(qkv(norm(x)))), one head of every channel over every pixel - as two
    /// GEMMs and a softmax (one head of 1,152 features is past the fused kernel), a block of query rows at a time.
    /// v is made transposed (W_v . norm(x)^T): the second GEMM reads it as its weights; its bias is added after
    /// (each row of weights sums to 1).
    fn attention(&self, nsd: &Nsd, a: &Att, x: DevBuf, h: usize, w: usize, c: usize) -> Result<DevBuf> {
        let gpu = &self.gpu;
        let k = ffi::api()?;
        let px = h * w;
        let n = DevBuf::new(gpu, px * c * 2)?;
        nsd.rms_norm_mod(x.ptr(), Dt::F16, px, c, a.norm.ptr(), EPS, none(), none(), none(), n.ptr(), Dt::F16)?;
        let wpart = |i: usize| -> *const std::ffi::c_void { a.qkv.ptr().cast::<u16>().wrapping_add(i * c * c).cast() };
        let bpart = |i: usize| -> *const std::ffi::c_void { a.qkv_b.fp().wrapping_add(i * c).cast() };
        let (q, kk, vt) = (DevBuf::new(gpu, px * c * 2)?, DevBuf::new(gpu, px * c * 2)?, DevBuf::new(gpu, c * px * 2)?);
        nsd.linear(n.ptr(), Dt::F16, px, c, wpart(0), c, bpart(0), q.ptr(), Dt::F16)?;
        nsd.linear(n.ptr(), Dt::F16, px, c, wpart(1), c, bpart(1), kk.ptr(), Dt::F16)?;
        nsd.linear(wpart(2), Dt::F16, c, c, n.ptr(), px, none(), vt.ptr(), Dt::F16)?;
        prof::mark(nsd, "vae att qkv")?;
        // query rows a block at a time: the block's scores float32 [rows, px] under ~256 MiB
        let rows_max = ((256usize << 20) / (px * 4)).clamp(1, px);
        let scores = DevBuf::f32(gpu, rows_max * px)?;
        let probs = DevBuf::new(gpu, rows_max * px * 2)?;
        let o = DevBuf::new(gpu, px * c * 2)?;
        let scale = 1.0 / (c as f32).sqrt();
        let mut r0 = 0;
        while r0 < px {
            let rows = rows_max.min(px - r0);
            let qa: *const std::ffi::c_void = q.ptr().cast::<u16>().wrapping_add(r0 * c).cast();
            nsd.linear(qa, Dt::F16, rows, c, kk.ptr(), px, none(), scores.ptr(), Dt::F32)?;
            // SAFETY: scores holds rows x px floats, probs as many halves.
            ffi::check(unsafe { (k.softmax_rows)(gpu.raw(), scores.fp(), rows as i64, px as i64, scale, probs.ptr()) }, "softmax")?;
            let oa: *mut std::ffi::c_void = o.ptr().cast::<u16>().wrapping_add(r0 * c).cast();
            nsd.linear(probs.ptr(), Dt::F16, rows, px, vt.ptr(), c, bpart(2), oa, Dt::F16)?;
            r0 += rows;
        }
        prof::mark(nsd, "vae attention")?;
        nsd.linear(o.ptr(), Dt::F16, px, c, a.proj.ptr(), c, a.proj_b.ptr(), n.ptr(), Dt::F16)?;
        nsd.gate_add(x.ptr(), Dt::F16, px, c, n.ptr(), Dt::F16, none(), none())?;
        nsd.wait()?;
        Ok(x)
    }

    /// Normalized latents float32 [h * w, z] (h = H / 16) of a picture: RGBA float32 [H, W, 4] in [-1, 1]
    pub fn encode(&self, nsd: &Nsd, rgba: &[f32], hh: usize, ww: usize) -> Result<Vec<f32>> {
        let e = self.enc.as_ref().ok_or_else(|| Error("this VAE file has no encoder (edits need it)".into()))?;
        let gpu = &self.gpu;
        let k = ffi::api()?;
        let xf = DevBuf::from_f32(gpu, rgba)?;
        let x = DevBuf::new(gpu, rgba.len() * 2)?;
        // SAFETY: H x W x 4 values each way.
        ffi::check(unsafe { (k.to_half)(gpu.raw(), xf.fp(), x.ptr(), rgba.len() as i64) }, "to half")?;
        let mut x = self.conv(nsd, &e.conv_in, &x, hh, ww)?;
        let (mut h, mut w) = (hh, ww);
        for d in &e.downs {
            let keep = DevBuf::new(gpu, h * w * d.ci * 2)?;
            keep.copy_within(0, &x, 0, h * w * d.ci * 2)?;
            for r in &d.resnets {
                x = self.resnet(nsd, r, x, h, w)?;
            }
            match &d.resample {
                Some(c) => {
                    let p = DevBuf::new(gpu, (h + 1) * (w + 1) * c.ci * 2)?;
                    // SAFETY: x holds h x w x ci halves, p (h + 1) x (w + 1) x ci.
                    ffi::check(unsafe { (k.pad_br)(gpu.raw(), x.ptr(), h as i64, w as i64, c.ci as i64, p.ptr()) }, "pad")?;
                    let out = DevBuf::new(gpu, (h / 2) * (w / 2) * c.co * 2)?;
                    nsd.conv3d_ex(p.ptr(), Dt::F16, (1, h + 1, w + 1), c.ci, c.w.ptr(), c.co, (1, 3, 3), (1, 2, 2), c.b.ptr(), out.ptr())?;
                    // SAFETY: keep holds h x w x ci halves, out (h / 2) x (w / 2) x co.
                    ffi::check(unsafe { (k.avg_down_add)(gpu.raw(), keep.ptr(), h as i64, w as i64, d.ci as i64, d.co as i64, d.ft, out.ptr()) }, "shortcut")?;
                    nsd.wait()?;
                    x = out;
                    h /= 2;
                    w /= 2;
                }
                None => {
                    // the shortcut at one-to-one: the block's input
                    nsd.gate_add(x.ptr(), Dt::F16, h * w, d.co, keep.ptr(), Dt::F16, none(), none())?;
                    nsd.wait()?;
                }
            }
        }
        let x = self.resnet(nsd, &e.mid[0], x, h, w)?;
        let x = self.attention(nsd, &e.att, x, h, w, e.mid[0].conv2.co)?;
        let x = self.resnet(nsd, &e.mid[1], x, h, w)?;
        let a = self.norm_silu(nsd, &x, h * w, e.head.ci, &e.out_norm)?;
        let x = self.conv(nsd, &e.head, &a, h, w)?;
        let x = self.conv(nsd, &e.quant, &x, h, w)?;
        let zc = e.quant.co;
        let f = DevBuf::f32(gpu, h * w * zc)?;
        // SAFETY: h x w x zc values each way.
        ffi::check(unsafe { (k.to_float)(gpu.raw(), x.ptr(), f.fp(), (h * w * zc) as i64) }, "to float")?;
        gpu.sync()?;
        let all = f.to_f32()?;
        let z = self.z;
        Ok((0..h * w).flat_map(|p| (0..z).map(move |c| (p, c))).map(|(p, c)| (all[p * zc + c] - self.mean[c]) / self.std[c]).collect())
    }
}
