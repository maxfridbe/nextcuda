//! The decoder: FLUX.2's small VAE (diffusers' AutoencoderKLFlux2, the decoder of 96 / 192 / 384 / 384 channels),
//! from the DiT's latents to the picture. Everything is channels-last half on the GPU.
//!
//! ```text
//!   latents [128, h, w] -> * sqrt(var + 1e-4) + mean (the VAE's batch norm) -> 2 x 2 patches unpacked [2h, 2w, 32]
//!   1x1 conv (post_quant_conv) -> 3x3 conv (384)
//!   middle: resnet, attention (one head over all pixels), resnet
//!   up: 3 resnets (384), up 2x + conv; 3 resnets (384), up 2x + conv; 3 resnets (192), up 2x + conv; 3 resnets (96)
//!   group norm, silu, 3x3 conv -> RGB in [-1, 1]
//! resnet: x + conv(silu(gn(conv(silu(gn(x)))))), a 1x1 conv on the shortcut when the width changes
//! ```

use std::path::Path;
use std::sync::Arc;

use nextcuda_core::{DevBuf, Error, Gpu, Result};
use nextcuda_diffusion::kernels::{none, Dt, Nsd};
use nextcuda_gguf::safetensors::SafeTensors;

use crate::ffi;

const EPS: f32 = 1e-6;
const GROUPS: usize = 32;
/// the batch norm's epsilon (vae/config.json batch_norm_eps)
const BN_EPS: f64 = 1e-4;
/// the VAE's own latent channels (a DiT token holds 2 x 2 of them)
pub const CH: usize = 32;

struct Conv {
    w: DevBuf,
    b: DevBuf,
    ci: usize,
    co: usize,
    k: usize,
}

struct Gn {
    w: DevBuf,
    b: DevBuf,
}

struct Resnet {
    norm1: Gn,
    conv1: Conv,
    norm2: Gn,
    conv2: Conv,
    shortcut: Option<Conv>,
}

struct Att {
    norm: Gn,
    q: (DevBuf, DevBuf),
    k: (DevBuf, DevBuf),
    v: (DevBuf, DevBuf),
    o: (DevBuf, DevBuf),
}

struct Up {
    resnets: Vec<Resnet>,
    up: Option<Conv>,
}

pub struct Vae {
    gpu: Arc<Gpu>,
    mean: Vec<f32>,
    var: Vec<f32>,
    post_quant: Conv,
    conv_in: Conv,
    mid: [Resnet; 2],
    att: Att,
    ups: Vec<Up>,
    out_norm: Gn,
    conv_out: Conv,
}

fn ge(e: nextcuda_gguf::Error) -> Error {
    Error(e.0)
}

impl Vae {
    pub fn load(file: &Path, nsd: &Nsd, log: &mut dyn FnMut(String)) -> Result<Vae> {
        let t0 = std::time::Instant::now();
        let gpu = nsd.gpu.clone();
        let st = SafeTensors::open(file).map_err(ge)?;
        // a weight as half on the card (the file's float32 or bfloat16)
        let half = |name: &str| -> Result<(DevBuf, Vec<u64>)> {
            let t = st.need(name).map_err(ge)?;
            let v = st.f32(name).map_err(ge)?;
            let f = DevBuf::from_f32(&gpu, &v)?;
            let out = DevBuf::new(&gpu, v.len() * 2)?;
            nsd.dequant(f.ptr(), 0, v.len(), out.ptr(), Dt::F16)?;
            nsd.wait()?;
            Ok((out, t.shape.clone()))
        };
        let vec = |name: &str| -> Result<DevBuf> { DevBuf::from_f32(&gpu, &st.f32(name).map_err(ge)?) };
        let conv = |p: &str| -> Result<Conv> {
            let (w, shape) = half(&format!("{p}.weight"))?;
            let (co, ci, k) = (shape[0] as usize, shape[1] as usize, *shape.last().unwrap_or(&1) as usize);
            Ok(Conv { w, b: vec(&format!("{p}.bias"))?, ci, co, k })
        };
        let gn = |p: &str| -> Result<Gn> { Ok(Gn { w: vec(&format!("{p}.weight"))?, b: vec(&format!("{p}.bias"))? }) };
        let resnet = |p: &str| -> Result<Resnet> {
            Ok(Resnet {
                norm1: gn(&format!("{p}.norm1"))?,
                conv1: conv(&format!("{p}.conv1"))?,
                norm2: gn(&format!("{p}.norm2"))?,
                conv2: conv(&format!("{p}.conv2"))?,
                shortcut: if st.tensor(&format!("{p}.conv_shortcut.weight")).is_some() { Some(conv(&format!("{p}.conv_shortcut"))?) } else { None },
            })
        };
        let lin = |p: &str| -> Result<(DevBuf, DevBuf)> { Ok((half(&format!("{p}.weight"))?.0, vec(&format!("{p}.bias"))?)) };
        let mut ups = Vec::new();
        for i in 0.. {
            let p = format!("decoder.up_blocks.{i}");
            if st.tensor(&format!("{p}.resnets.0.conv1.weight")).is_none() {
                break;
            }
            let n = (0..).take_while(|j| st.tensor(&format!("{p}.resnets.{j}.conv1.weight")).is_some()).count();
            ups.push(Up {
                resnets: (0..n).map(|j| resnet(&format!("{p}.resnets.{j}"))).collect::<Result<_>>()?,
                up: if st.tensor(&format!("{p}.upsamplers.0.conv.weight")).is_some() { Some(conv(&format!("{p}.upsamplers.0.conv"))?) } else { None },
            });
        }
        if ups.is_empty() {
            return Err(Error(format!("{}: not FLUX.2's VAE (no decoder.up_blocks.0)", file.display())));
        }
        let a = "decoder.mid_block.attentions.0";
        let vae = Vae {
            gpu: gpu.clone(),
            mean: st.f32("bn.running_mean").map_err(ge)?,
            var: st.f32("bn.running_var").map_err(ge)?,
            post_quant: conv("post_quant_conv")?,
            conv_in: conv("decoder.conv_in")?,
            mid: [resnet("decoder.mid_block.resnets.0")?, resnet("decoder.mid_block.resnets.1")?],
            att: Att { norm: gn(&format!("{a}.group_norm"))?, q: lin(&format!("{a}.to_q"))?, k: lin(&format!("{a}.to_k"))?, v: lin(&format!("{a}.to_v"))?,
                       o: lin(&format!("{a}.to_out.0"))? },
            ups,
            out_norm: gn("decoder.conv_norm_out")?,
            conv_out: conv("decoder.conv_out")?,
        };
        if vae.post_quant.ci != CH || vae.mean.len() != 4 * CH || vae.conv_out.co != 3 {
            return Err(Error(format!("{}: {} latent channels, {} normalized, {} out: FLUX.2's VAE takes 32 (128 packed) and makes RGB", file.display(),
                                     vae.post_quant.ci, vae.mean.len(), vae.conv_out.co)));
        }
        log(format!("moss-image vae: {} up stages on {} in {:.1} s", vae.ups.len(), gpu.name, t0.elapsed().as_secs_f64()));
        Ok(vae)
    }

    /// The DiT's latents float32 [128, h, w] as the decoder's input: denormalized by the batch norm's statistics and
    /// unpacked (channel c * 4 + dy * 2 + dx -> pixel (2y + dy, 2x + dx) of channel c), float32 [32, 2h, 2w]
    pub fn unpack(&self, lat: &[f32], h: usize, w: usize) -> Vec<f32> {
        let (hh, ww) = (2 * h, 2 * w);
        let mut z = vec![0f32; CH * hh * ww];
        for c4 in 0..4 * CH {
            let (s, m) = (((self.var[c4] as f64) + BN_EPS).sqrt(), self.mean[c4] as f64);
            let (c, dy, dx) = (c4 / 4, (c4 / 2) % 2, c4 % 2);
            for y in 0..h {
                for x in 0..w {
                    z[c * hh * ww + (2 * y + dy) * ww + 2 * x + dx] = (lat[c4 * h * w + y * w + x] as f64 * s + m) as f32;
                }
            }
        }
        z
    }

    fn conv(&self, nsd: &Nsd, c: &Conv, x: &DevBuf, h: usize, w: usize) -> Result<DevBuf> {
        let out = DevBuf::new(&self.gpu, h * w * c.co * 2)?;
        nsd.conv2d(x.ptr(), Dt::F16, 1, h, w, c.ci, c.w.ptr(), c.co, c.k, c.b.fp().cast(), out.ptr())?;
        Ok(out)
    }

    /// silu(group norm(x)), a new buffer
    fn norm_silu(&self, nsd: &Nsd, x: &DevBuf, px: usize, c: usize, g: &Gn) -> Result<DevBuf> {
        let out = DevBuf::new(&self.gpu, px * c * 2)?;
        nsd.group_norm_silu(x.ptr(), Dt::F16, 1, px, c, GROUPS, g.w.ptr(), g.b.ptr(), EPS, out.ptr())?;
        Ok(out)
    }

    /// Each intermediate let go as soon as the next is made (after the work on it): at 1024 x 1024 a buffer of the
    /// last stage is 192 MiB, and the DiT's weights stay beside the VAE
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

    /// The middle's attention: x + o(attention(q, k, v of gn(x))), one head of every channel over every pixel - two
    /// GEMMs and a softmax, a block of query rows at a time; v made transposed (its bias added after: each row of
    /// weights sums to 1)
    fn attention(&self, nsd: &Nsd, x: DevBuf, h: usize, w: usize, c: usize) -> Result<DevBuf> {
        let gpu = &self.gpu;
        let k = ffi::api()?;
        let a = &self.att;
        let px = h * w;
        let n = DevBuf::new(gpu, px * c * 2)?;
        // SAFETY: px x c halves each way, c floats of weight and bias.
        ffi::check(unsafe { (k.group_norm)(gpu.raw(), x.ptr(), px as i64, c as i64, GROUPS as i64, a.norm.w.fp(), a.norm.b.fp(), EPS, n.ptr()) }, "group norm")?;
        let (q, kk, vt) = (DevBuf::new(gpu, px * c * 2)?, DevBuf::new(gpu, px * c * 2)?, DevBuf::new(gpu, c * px * 2)?);
        nsd.linear(n.ptr(), Dt::F16, px, c, a.q.0.ptr(), c, a.q.1.fp().cast(), q.ptr(), Dt::F16)?;
        nsd.linear(n.ptr(), Dt::F16, px, c, a.k.0.ptr(), c, a.k.1.fp().cast(), kk.ptr(), Dt::F16)?;
        nsd.linear(a.v.0.ptr(), Dt::F16, c, c, n.ptr(), px, none(), vt.ptr(), Dt::F16)?;
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
            nsd.linear(probs.ptr(), Dt::F16, rows, px, vt.ptr(), c, a.v.1.fp().cast(), oa, Dt::F16)?;
            r0 += rows;
        }
        nsd.linear(o.ptr(), Dt::F16, px, c, a.o.0.ptr(), c, a.o.1.fp().cast(), n.ptr(), Dt::F16)?;
        nsd.gate_add(x.ptr(), Dt::F16, px, c, n.ptr(), Dt::F16, none(), none())?;
        nsd.wait()?;
        Ok(x)
    }

    /// The picture from the DiT's latents float32 [128, h, w]: its RGB [H, W, 3] in [-1, 1] as float32 (unclamped),
    /// width, height
    pub fn decode_f32(&self, nsd: &Nsd, lat: &[f32], h: usize, w: usize) -> Result<(Vec<f32>, usize, usize)> {
        let z = self.unpack(lat, h, w);
        self.decode_z(nsd, &z, 2 * h, 2 * w)
    }

    /// From the decoder's input float32 [32, h, w] (unpacked)
    pub fn decode_z(&self, nsd: &Nsd, z: &[f32], h: usize, w: usize) -> Result<(Vec<f32>, usize, usize)> {
        let gpu = &self.gpu;
        let k = ffi::api()?;
        let mut zl = vec![0f32; h * w * CH];
        for c in 0..CH {
            for p in 0..h * w {
                zl[p * CH + c] = z[c * h * w + p];
            }
        }
        let zf = DevBuf::from_f32(gpu, &zl)?;
        let x = DevBuf::new(gpu, zl.len() * 2)?;
        // SAFETY: h x w x 32 values each way.
        ffi::check(unsafe { (k.to_half)(gpu.raw(), zf.fp(), x.ptr(), zl.len() as i64) }, "to half")?;
        let x = self.conv(nsd, &self.post_quant, &x, h, w)?;
        let x = self.conv(nsd, &self.conv_in, &x, h, w)?;
        let x = self.resnet(nsd, &self.mid[0], x, h, w)?;
        let x = self.attention(nsd, x, h, w, self.conv_in.co)?;
        let mut x = self.resnet(nsd, &self.mid[1], x, h, w)?;
        let (mut h, mut w) = (h, w);
        for up in &self.ups {
            for r in &up.resnets {
                x = self.resnet(nsd, r, x, h, w)?;
            }
            if let Some(c) = &up.up {
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
        let a = self.norm_silu(nsd, &x, h * w, self.conv_out.ci, &self.out_norm)?;
        nsd.wait()?;
        drop(x);
        let img = self.conv(nsd, &self.conv_out, &a, h, w)?;
        let f = DevBuf::f32(gpu, h * w * 3)?;
        // SAFETY: h x w x 3 halves in, as many floats out.
        ffi::check(unsafe { (k.to_float)(gpu.raw(), img.ptr(), f.fp(), (h * w * 3) as i64) }, "to float")?;
        nsd.wait()?;
        Ok((f.to_f32()?, w, h))
    }

    /// RGB [H, W, 3] bytes from the DiT's latents
    pub fn decode(&self, nsd: &Nsd, lat: &[f32], h: usize, w: usize) -> Result<(Vec<u8>, usize, usize)> {
        let (v, pw, ph) = self.decode_f32(nsd, lat, h, w)?;
        Ok((v.iter().map(|x| ((x.clamp(-1.0, 1.0) + 1.0) * 127.5).round() as u8).collect(), pw, ph))
    }
}
