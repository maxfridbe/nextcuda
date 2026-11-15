//! Qwen-Image's VAE (diffusers' AutoencoderKLQwenImage: Wan 2.1's, 16 latent channels, 8x down), RGB pictures to
//! latents and back. Everything is channels-last half on the GPU; one picture is one frame, so the causal 3-D
//! convolutions are their last time slice (the frames before are zeros) and the time convolutions are skipped.
//!
//! ```text
//!   decode: z [h, w, 16] = latents * std + mean -> 1x1 conv -> 3x3 conv (384) -> middle (resnet, attention, resnet)
//!     -> 3 resnets (384), up 2x + conv (192); 3 resnets (384), up; 3 resnets (192), up (96); 3 resnets (96)
//!     -> rms norm, silu, 3x3 conv -> RGB in [-1, 1]
//!   encode: RGB -> 3x3 conv (96) -> 2 resnets, down (stride-2 conv); 2 resnets (192), down; 2 resnets (384), down;
//!     2 resnets -> middle -> rms norm, silu, 3x3 conv (32) -> 1x1 conv -> the first 16 (the mean) -> (z - mean) / std
//! resnet: x + conv(silu(norm(conv(silu(norm(x)))))), a 1x1 conv on the shortcut when the width changes
//! ```

use std::path::Path;
use std::sync::Arc;

use nextcuda_core::{DevBuf, Error, Gpu, Result};
use nextcuda_diffusion::kernels::{none, Dt, Nsd};
use nextcuda_gguf::safetensors::SafeTensors;

use crate::ffi;
use crate::rust::vae::{tiles, Tile};

/// F.normalize's floor, as an epsilon on the mean square: never reached by real activations
const EPS: f32 = 1e-24;
pub const Z: usize = 16;
/// The latents' per-channel statistics (diffusers' vae/config.json latents_mean / latents_std; Wan 2.1's)
const MEAN: [f32; Z] = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921];
const STD: [f32; Z] = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160];
/// The largest side, in latents, the decoder's up path runs in one piece; a larger picture goes in tiles of this side
const TILE: usize = 128;
/// The up path's reach in latents (3 x 3 convolutions: 6 at the latent scale, 0.5 + 3 at 2x, ...: 12.3 in all): a
/// tile's margin of this many gives its core exactly the whole picture's pixels
const HALO: usize = 14;

struct Conv {
    w: DevBuf,
    b: DevBuf,
    ci: usize,
    co: usize,
    k: usize,
}

struct Resnet {
    norm1: DevBuf,
    conv1: Conv,
    norm2: DevBuf,
    conv2: Conv,
    shortcut: Option<Conv>,
}

enum Stage {
    Res(Resnet),
    /// up: nearest 2x, then the conv; down: zero pad right and bottom, then the conv at stride 2
    Resample(Conv),
}

struct Att {
    norm: DevBuf,
    qkv: DevBuf,
    qkv_b: DevBuf,
    proj: DevBuf,
    proj_b: DevBuf,
}

struct Encoder {
    conv_in: Conv,
    downs: Vec<Stage>,
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
    ups: Vec<Stage>,
    out_norm: DevBuf,
    conv_out: Conv,
    enc: Option<Encoder>,
}

fn ge(e: nextcuda_gguf::Error) -> Error {
    Error(e.0)
}

impl Vae {
    pub fn load(file: &Path, nsd: &Nsd, log: &mut dyn FnMut(String)) -> Result<Vae> {
        let t0 = std::time::Instant::now();
        let gpu = nsd.gpu.clone();
        let st = SafeTensors::open(file).map_err(ge)?;
        // a weight as half on the card; a 3-D kernel [co, ci, kt, kh, kw] as its last time slice [co, ci, kh, kw]
        let half = |name: &str| -> Result<(DevBuf, Vec<u64>)> {
            let t = st.need(name).map_err(ge)?;
            if t.dtype != "BF16" {
                return Err(Error(format!("{name}: {} weights are not read yet", t.dtype)));
            }
            let mut raw = st.read(t).map_err(ge)?;
            let mut shape = t.shape.clone();
            if shape.len() == 5 {
                let (co, ci, kt, kh, kw) = (shape[0] as usize, shape[1] as usize, shape[2] as usize, shape[3] as usize, shape[4] as usize);
                let plane = kh * kw * 2;
                let mut out = Vec::with_capacity(co * ci * plane);
                for i in 0..co * ci {
                    let at = (i * kt + kt - 1) * plane;
                    out.extend_from_slice(&raw[at..at + plane]);
                }
                raw = out;
                shape = vec![shape[0], shape[1], shape[3], shape[4]];
            }
            let n = raw.len() / 2;
            let src = DevBuf::new(&gpu, raw.len())?;
            src.write(0, &raw)?;
            let out = DevBuf::new(&gpu, n * 2)?;
            nsd.dequant(src.ptr(), 30, n, out.ptr(), Dt::F16)?;
            nsd.wait()?;
            Ok((out, shape))
        };
        let vec = |name: &str| -> Result<DevBuf> { DevBuf::from_f32(&gpu, &st.f32(name).map_err(ge)?) };
        let conv = |p: &str| -> Result<Conv> {
            let (w, shape) = half(&format!("{p}.weight"))?;
            let (co, ci, k) = (shape[0] as usize, shape[1] as usize, *shape.last().unwrap_or(&1) as usize);
            Ok(Conv { w, b: vec(&format!("{p}.bias"))?, ci, co, k })
        };
        let resnet = |p: &str| -> Result<Resnet> {
            Ok(Resnet {
                norm1: vec(&format!("{p}.residual.0.gamma"))?,
                conv1: conv(&format!("{p}.residual.2"))?,
                norm2: vec(&format!("{p}.residual.3.gamma"))?,
                conv2: conv(&format!("{p}.residual.6"))?,
                shortcut: if st.tensor(&format!("{p}.shortcut.weight")).is_some() { Some(conv(&format!("{p}.shortcut"))?) } else { None },
            })
        };
        let stages = |side: &str| -> Result<Vec<Stage>> {
            let mut v = Vec::new();
            for i in 0.. {
                let p = format!("{side}.{i}");
                if st.tensor(&format!("{p}.residual.0.gamma")).is_some() {
                    v.push(Stage::Res(resnet(&p)?));
                } else if st.tensor(&format!("{p}.resample.1.weight")).is_some() {
                    v.push(Stage::Resample(conv(&format!("{p}.resample.1"))?));
                } else {
                    break;
                }
            }
            Ok(v)
        };
        let att = |p: &str| -> Result<Att> {
            Ok(Att { norm: vec(&format!("{p}.norm.gamma"))?, qkv: half(&format!("{p}.to_qkv.weight"))?.0, qkv_b: vec(&format!("{p}.to_qkv.bias"))?,
                     proj: half(&format!("{p}.proj.weight"))?.0, proj_b: vec(&format!("{p}.proj.bias"))? })
        };
        let ups = stages("decoder.upsamples")?;
        if ups.is_empty() {
            return Err(Error(format!("{}: not a Wan 2.1 decoder (no decoder.upsamples.0)", file.display())));
        }
        let enc = if st.tensor("encoder.conv1.weight").is_some() {
            Some(Encoder {
                conv_in: conv("encoder.conv1")?,
                downs: stages("encoder.downsamples")?,
                mid: [resnet("encoder.middle.0")?, resnet("encoder.middle.2")?],
                att: att("encoder.middle.1")?,
                out_norm: vec("encoder.head.0.gamma")?,
                head: conv("encoder.head.2")?,
                quant: conv("conv1")?,
            })
        } else {
            None
        };
        let vae = Vae {
            gpu: gpu.clone(),
            post_quant: conv("conv2")?,
            conv_in: conv("decoder.conv1")?,
            mid: [resnet("decoder.middle.0")?, resnet("decoder.middle.2")?],
            att: att("decoder.middle.1")?,
            ups,
            out_norm: vec("decoder.head.0.gamma")?,
            conv_out: conv("decoder.head.2")?,
            enc,
        };
        if vae.post_quant.ci != Z || vae.conv_out.co != 3 {
            return Err(Error(format!("{}: {} latent channels and {} out: Qwen-Image's VAE takes 16 and makes RGB", file.display(), vae.post_quant.ci,
                                     vae.conv_out.co)));
        }
        log(format!("qwen-image vae: {} up stages on {} in {:.1} s", vae.ups.len(), gpu.name, t0.elapsed().as_secs_f64()));
        Ok(vae)
    }

    fn conv(&self, nsd: &Nsd, c: &Conv, x: &DevBuf, h: usize, w: usize) -> Result<DevBuf> {
        let out = DevBuf::new(&self.gpu, h * w * c.co * 2)?;
        nsd.conv2d(x.ptr(), Dt::F16, 1, h, w, c.ci, c.w.ptr(), c.co, c.k, c.b.fp().cast(), out.ptr())?;
        Ok(out)
    }

    /// silu(rms norm(x) * gamma), a new buffer
    fn norm_silu(&self, nsd: &Nsd, x: &DevBuf, px: usize, c: usize, gamma: &DevBuf) -> Result<DevBuf> {
        let k = ffi::api()?;
        let out = DevBuf::new(&self.gpu, px * c * 2)?;
        nsd.rms_norm_mod(x.ptr(), Dt::F16, px, c, gamma.fp().cast(), EPS, none(), none(), none(), out.ptr(), Dt::F16)?;
        // SAFETY: px x c halves.
        ffi::check(unsafe { (k.silu)(self.gpu.raw(), out.ptr(), (px * c) as i64) }, "silu")?;
        Ok(out)
    }

    /// Each intermediate let go as soon as the next is made (after the work on it: a free does not wait for the queue)
    fn resnet(&self, nsd: &Nsd, r: &Resnet, x: DevBuf, h: usize, w: usize) -> Result<DevBuf> {
        let px = h * w;
        let a1 = self.norm_silu(nsd, &x, px, r.conv1.ci, &r.norm1)?;
        let a2 = self.conv(nsd, &r.conv1, &a1, h, w)?;
        nsd.wait()?;
        drop(a1);
        let a3 = self.norm_silu(nsd, &a2, px, r.conv1.co, &r.norm2)?;
        nsd.wait()?;
        drop(a2);
        let out = self.conv(nsd, &r.conv2, &a3, h, w)?;
        nsd.wait()?;
        drop(a3);
        let short = match &r.shortcut {
            Some(c) => {
                let s = self.conv(nsd, c, &x, h, w)?;
                nsd.wait()?;
                drop(x);
                s
            }
            None => x,
        };
        nsd.gate_add(out.ptr(), Dt::F16, px, r.conv2.co, short.ptr(), Dt::F16, none(), none())?;
        nsd.wait()?;
        Ok(out)
    }

    /// The middle's attention: x + proj(attention(qkv(norm(x)))), one head of every channel over every pixel
    fn attention(&self, nsd: &Nsd, a: &Att, x: DevBuf, h: usize, w: usize, c: usize) -> Result<DevBuf> {
        let gpu = &self.gpu;
        let k = ffi::api()?;
        let px = h * w;
        let n = DevBuf::new(gpu, px * c * 2)?;
        nsd.rms_norm_mod(x.ptr(), Dt::F16, px, c, a.norm.fp().cast(), EPS, none(), none(), none(), n.ptr(), Dt::F16)?;
        let wpart = |i: usize| -> *const std::ffi::c_void { a.qkv.ptr().cast::<u16>().wrapping_add(i * c * c).cast() };
        let bpart = |i: usize| -> *const std::ffi::c_void { a.qkv_b.fp().wrapping_add(i * c).cast() };
        let (q, kk, vt) = (DevBuf::new(gpu, px * c * 2)?, DevBuf::new(gpu, px * c * 2)?, DevBuf::new(gpu, c * px * 2)?);
        nsd.linear(n.ptr(), Dt::F16, px, c, wpart(0), c, bpart(0), q.ptr(), Dt::F16)?;
        nsd.linear(n.ptr(), Dt::F16, px, c, wpart(1), c, bpart(1), kk.ptr(), Dt::F16)?;
        nsd.linear(wpart(2), Dt::F16, c, c, n.ptr(), px, none(), vt.ptr(), Dt::F16)?;
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
        nsd.linear(o.ptr(), Dt::F16, px, c, a.proj.ptr(), c, a.proj_b.fp().cast(), n.ptr(), Dt::F16)?;
        nsd.gate_add(x.ptr(), Dt::F16, px, c, n.ptr(), Dt::F16, none(), none())?;
        nsd.wait()?;
        Ok(x)
    }

    /// RGB [H, W, 3] bytes from normalized latents float32 [h, w, 16] (channels last, as the denoiser's tokens unpack).
    /// The middle (its attention sees every latent) runs whole; the up path, local, a tile at a time above `TILE`.
    pub fn decode(&self, nsd: &Nsd, lat: &[f32], h: usize, w: usize) -> Result<(Vec<u8>, usize, usize)> {
        let gpu = &self.gpu;
        let k = ffi::api()?;
        let zl: Vec<f32> = lat.iter().enumerate().map(|(i, v)| v * STD[i % Z] + MEAN[i % Z]).collect();
        let zf = DevBuf::from_f32(gpu, &zl)?;
        let x = DevBuf::new(gpu, zl.len() * 2)?;
        // SAFETY: h x w x 16 values each way.
        ffi::check(unsafe { (k.to_half)(gpu.raw(), zf.fp(), x.ptr(), zl.len() as i64) }, "to half")?;
        drop(zf);
        let x = self.conv(nsd, &self.post_quant, &x, h, w)?;
        let x = self.conv(nsd, &self.conv_in, &x, h, w)?;
        let x = self.resnet(nsd, &self.mid[0], x, h, w)?;
        let x = self.attention(nsd, &self.att, x, h, w, self.conv_in.co)?;
        let x = self.resnet(nsd, &self.mid[1], x, h, w)?;
        let s = 1usize << self.ups.iter().filter(|u| matches!(u, Stage::Resample(_))).count();
        let (ph, pw) = (h * s, w * s);
        if h <= TILE && w <= TILE {
            return Ok((self.up_path(nsd, x, h, w)?, pw, ph));
        }
        let c = self.conv_in.co;
        let core = TILE - 2 * HALO;
        let mut out = vec![0u8; ph * pw * 3];
        for Tile { y0, y1, x0, x1, ya, yb, xa, xb } in tiles(h, w, core, core, HALO) {
            let (th, tw) = (yb - ya, xb - xa);
            let t = DevBuf::new(gpu, th * tw * c * 2)?;
            t.copy_2d(0, tw * c * 2, &x, (ya * w + xa) * c * 2, w * c * 2, tw * c * 2, th)?;
            let px = self.up_path(nsd, t, th, tw)?;
            let (cy, cx, ch, cw) = ((y0 - ya) * s, (x0 - xa) * s, (y1 - y0) * s, (x1 - x0) * s);
            for r in 0..ch {
                let src = ((cy + r) * tw * s + cx) * 3;
                let dst = ((y0 * s + r) * pw + x0 * s) * 3;
                out[dst..dst + cw * 3].copy_from_slice(&px[src..src + cw * 3]);
            }
        }
        Ok((out, pw, ph))
    }

    /// The up stages, the output norm and conv from the middle's output x [h, w, 384]: RGB bytes
    fn up_path(&self, nsd: &Nsd, mut x: DevBuf, h: usize, w: usize) -> Result<Vec<u8>> {
        let gpu = &self.gpu;
        let k = ffi::api()?;
        let (mut h, mut w) = (h, w);
        for up in &self.ups {
            match up {
                Stage::Res(r) => x = self.resnet(nsd, r, x, h, w)?,
                Stage::Resample(c) => {
                    let u = DevBuf::new(gpu, 4 * h * w * c.ci * 2)?;
                    // SAFETY: x holds h x w x ci halves, u four times that.
                    ffi::check(unsafe { (k.up2)(gpu.raw(), x.ptr(), h as i64, w as i64, c.ci as i64, u.ptr()) }, "upsample")?;
                    nsd.wait()?;
                    drop(x);
                    h *= 2;
                    w *= 2;
                    x = self.conv(nsd, c, &u, h, w)?;
                    nsd.wait()?;
                }
            }
        }
        let a = self.norm_silu(nsd, &x, h * w, self.conv_out.ci, &self.out_norm)?;
        nsd.wait()?;
        drop(x);
        let img = self.conv(nsd, &self.conv_out, &a, h, w)?;
        let f = DevBuf::f32(gpu, h * w * 3)?;
        // SAFETY: h x w x 3 values each way.
        ffi::check(unsafe { (k.to_float)(gpu.raw(), img.ptr(), f.fp(), (h * w * 3) as i64) }, "to float")?;
        gpu.sync()?;
        Ok(f.to_f32()?.iter().map(|v| ((v.clamp(-1.0, 1.0) + 1.0) * 127.5).round() as u8).collect())
    }

    /// Normalized latents float32 [h, w, 16] (h = H / 8) of an RGB picture float32 [H, W, 3] in [-1, 1]
    pub fn encode(&self, nsd: &Nsd, rgb: &[f32], hh: usize, ww: usize) -> Result<Vec<f32>> {
        let e = self.enc.as_ref().ok_or_else(|| Error("this VAE file has no encoder (img2img needs it)".into()))?;
        let gpu = &self.gpu;
        let k = ffi::api()?;
        let xf = DevBuf::from_f32(gpu, rgb)?;
        let x = DevBuf::new(gpu, rgb.len() * 2)?;
        // SAFETY: H x W x 3 values each way.
        ffi::check(unsafe { (k.to_half)(gpu.raw(), xf.fp(), x.ptr(), rgb.len() as i64) }, "to half")?;
        drop(xf);
        let mut x = self.conv(nsd, &e.conv_in, &x, hh, ww)?;
        let (mut h, mut w) = (hh, ww);
        for d in &e.downs {
            match d {
                Stage::Res(r) => x = self.resnet(nsd, r, x, h, w)?,
                Stage::Resample(c) => {
                    let p = DevBuf::new(gpu, (h + 1) * (w + 1) * c.ci * 2)?;
                    // SAFETY: x holds h x w x ci halves, p (h + 1) x (w + 1) x ci.
                    ffi::check(unsafe { (k.pad_br)(gpu.raw(), x.ptr(), h as i64, w as i64, c.ci as i64, p.ptr()) }, "pad")?;
                    let out = DevBuf::new(gpu, (h / 2) * (w / 2) * c.co * 2)?;
                    nsd.conv3d_ex(p.ptr(), Dt::F16, (1, h + 1, w + 1), c.ci, c.w.ptr(), c.co, (1, 3, 3), (1, 2, 2), c.b.ptr(), out.ptr())?;
                    nsd.wait()?;
                    x = out;
                    h /= 2;
                    w /= 2;
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
        Ok((0..h * w).flat_map(|p| (0..Z).map(move |c| (p, c))).map(|(p, c)| (all[p * zc + c] - MEAN[c]) / STD[c]).collect())
    }
}

/// Latents [h, w, 16] (channels last) as the denoiser's tokens: [h/2 * w/2, 64], a token's values c * 4 + dy * 2 + dx
pub fn pack(lat: &[f32], h: usize, w: usize) -> Vec<f32> {
    let mut out = vec![0f32; h * w * Z];
    for y in 0..h {
        for x in 0..w {
            let tok = (y / 2) * (w / 2) + x / 2;
            for c in 0..Z {
                out[tok * 64 + c * 4 + (y % 2) * 2 + x % 2] = lat[(y * w + x) * Z + c];
            }
        }
    }
    out
}

/// The inverse of `pack`
pub fn unpack(tok: &[f32], h: usize, w: usize) -> Vec<f32> {
    let mut out = vec![0f32; h * w * Z];
    for y in 0..h {
        for x in 0..w {
            let t = (y / 2) * (w / 2) + x / 2;
            for c in 0..Z {
                out[(y * w + x) * Z + c] = tok[t * 64 + c * 4 + (y % 2) * 2 + x % 2];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packing_round_trips_and_follows_diffusers() {
        let (h, w) = (4, 6);
        let lat: Vec<f32> = (0..h * w * Z).map(|i| i as f32).collect();
        let p = pack(&lat, h, w);
        assert_eq!(unpack(&p, h, w), lat);
        // token 0's values: channel 0 at (0,0), (0,1), (1,0), (1,1)
        assert_eq!(&p[..4], &[lat[0], lat[Z], lat[w * Z], lat[(w + 1) * Z]]);
    }
}
