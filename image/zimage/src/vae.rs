//! The decoder: Z-Image's VAE, FLUX.1's 16-channel autoencoder (diffusers' AutoencoderKL: 128 / 256 / 512 / 512
//! channels, no quant convolutions), from the DiT's latents to the picture. Everything is channels-last half on the GPU.
//!
//! ```text
//!   z [h, w, 16] = latents / 0.3611 + 0.1159 (vae/config.json scaling_factor, shift_factor) -> 3x3 conv (512)
//!   middle: resnet, attention (one head over all pixels), resnet
//!   up: 3 resnets (512), up 2x + conv; 3 resnets (512), up 2x + conv; 3 resnets (256), up 2x + conv; 3 resnets (128)
//!   group norm, silu, 3x3 conv -> RGB in [-1, 1]
//! resnet: x + conv(silu(gn(conv(silu(gn(x)))))), a 1x1 conv on the shortcut when the width changes
//! ```
//!
//! MOSS-Image's decoder (FLUX.2's) is the same network at other widths, behind a batch norm and 2 x 2 packing.

use std::path::Path;
use std::sync::Arc;

use nextcuda_core::{DevBuf, Error, Gpu, Result};
use nextcuda_diffusion::kernels::{none, Dt, Nsd};
use nextcuda_gguf::safetensors::SafeTensors;

use crate::ffi;

const EPS: f32 = 1e-6;
const GROUPS: usize = 32;
/// vae/config.json: latents / scaling_factor + shift_factor is what the decoder takes
pub const SCALE: f32 = 0.3611;
pub const SHIFT: f32 = 0.1159;
/// the latent channels
pub const CH: usize = 16;

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
    /// q, k and v stacked [3c, c] half, their biases float32 [3c]
    qkv: (DevBuf, DevBuf),
    o: (DevBuf, DevBuf),
}

struct Up {
    resnets: Vec<Resnet>,
    up: Option<Conv>,
}

pub struct Vae {
    gpu: Arc<Gpu>,
    conv_in: Conv,
    mid: [Resnet; 2],
    att: Att,
    ups: Vec<Up>,
    out_norm: Gn,
    conv_out: Conv,
    pub bytes: u64,
}

fn ge(e: nextcuda_gguf::Error) -> Error {
    Error(e.0)
}

impl Vae {
    pub fn load(file: &Path, nsd: &Nsd, log: &mut dyn FnMut(String)) -> Result<Vae> {
        let t0 = std::time::Instant::now();
        let gpu = nsd.gpu.clone();
        let st = SafeTensors::open(file).map_err(ge)?;
        let bytes = std::cell::Cell::new(0u64);
        // values as half on the card (the file's bfloat16 or float32)
        let half_of = |v: &[f32]| -> Result<DevBuf> {
            let f = DevBuf::from_f32(&gpu, v)?;
            let out = DevBuf::new(&gpu, v.len() * 2)?;
            nsd.dequant(f.ptr(), 0, v.len(), out.ptr(), Dt::F16)?;
            nsd.wait()?;
            bytes.set(bytes.get() + v.len() as u64 * 2);
            Ok(out)
        };
        let shape = |name: &str| -> Result<Vec<u64>> { Ok(st.need(name).map_err(ge)?.shape.clone()) };
        let floats = |name: &str| -> Result<Vec<f32>> { st.f32(name).map_err(ge) };
        let vec = |name: &str| -> Result<DevBuf> {
            let v = floats(name)?;
            bytes.set(bytes.get() + v.len() as u64 * 4);
            DevBuf::from_f32(&gpu, &v)
        };
        let conv = |p: &str| -> Result<Conv> {
            let s = shape(&format!("{p}.weight"))?;
            let (co, ci, k) = (s[0] as usize, s[1] as usize, *s.last().unwrap_or(&1) as usize);
            Ok(Conv { w: half_of(&floats(&format!("{p}.weight"))?)?, b: vec(&format!("{p}.bias"))?, ci, co, k })
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
        if ups.is_empty() || st.tensor("decoder.conv_in.weight").is_none() {
            return Err(Error(format!("{}: not an AutoencoderKL decoder (no decoder.up_blocks.0 / decoder.conv_in)", file.display())));
        }
        let a = "decoder.mid_block.attentions.0";
        let mut qkv_w = floats(&format!("{a}.to_q.weight"))?;
        qkv_w.extend(floats(&format!("{a}.to_k.weight"))?);
        qkv_w.extend(floats(&format!("{a}.to_v.weight"))?);
        let mut qkv_b = floats(&format!("{a}.to_q.bias"))?;
        qkv_b.extend(floats(&format!("{a}.to_k.bias"))?);
        qkv_b.extend(floats(&format!("{a}.to_v.bias"))?);
        let att = Att {
            norm: gn(&format!("{a}.group_norm"))?,
            qkv: (half_of(&qkv_w)?, DevBuf::from_f32(&gpu, &qkv_b)?),
            o: (half_of(&floats(&format!("{a}.to_out.0.weight"))?)?, vec(&format!("{a}.to_out.0.bias"))?),
        };
        let vae = Vae {
            gpu: gpu.clone(),
            conv_in: conv("decoder.conv_in")?,
            mid: [resnet("decoder.mid_block.resnets.0")?, resnet("decoder.mid_block.resnets.1")?],
            att,
            ups,
            out_norm: gn("decoder.conv_norm_out")?,
            conv_out: conv("decoder.conv_out")?,
            bytes: 0,
        };
        if vae.conv_in.ci != CH || vae.conv_out.co != 3 {
            return Err(Error(format!("{}: {} latent channels, {} out: FLUX's VAE takes {CH} and makes RGB", file.display(), vae.conv_in.ci, vae.conv_out.co)));
        }
        let vae = Vae { bytes: bytes.get(), ..vae };
        log(format!("z-image vae: {} up stages ({:.0} MB) on {} in {:.1} s", vae.ups.len(), vae.bytes as f64 / 1e6, gpu.name, t0.elapsed().as_secs_f64()));
        Ok(vae)
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
    /// last stage is 256 MiB, and the DiT's weights stay beside the VAE
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

    /// The middle's attention: x + o(attention(q, k, v of gn(x))), one head of every channel over every pixel
    fn attention(&self, nsd: &Nsd, x: DevBuf, h: usize, w: usize, c: usize) -> Result<DevBuf> {
        let gpu = &self.gpu;
        let k = ffi::api()?;
        let a = &self.att;
        let px = h * w;
        let n = DevBuf::new(gpu, px * c * 2)?;
        // SAFETY: px x c halves each way, c floats of weight and bias.
        ffi::check(unsafe { (k.group_norm)(gpu.raw(), x.ptr(), px as i64, c as i64, GROUPS as i64, a.norm.w.fp(), a.norm.b.fp(), EPS, n.ptr()) }, "group norm")?;
        let qkv = DevBuf::new(gpu, px * 3 * c * 2)?;
        nsd.linear(n.ptr(), Dt::F16, px, c, a.qkv.0.ptr(), 3 * c, a.qkv.1.fp().cast(), qkv.ptr(), Dt::F16)?;
        let at = |i: usize| -> *const std::ffi::c_void { qkv.ptr().cast::<u16>().wrapping_add(i * c).cast() };
        nsd.attention_batch(at(0), at(1), at(2), Dt::F16, 1, px, 1, c, 3 * c, n.ptr(), Dt::F16)?;
        let o = DevBuf::new(gpu, px * c * 2)?;
        nsd.linear(n.ptr(), Dt::F16, px, c, a.o.0.ptr(), c, a.o.1.fp().cast(), o.ptr(), Dt::F16)?;
        nsd.gate_add(x.ptr(), Dt::F16, px, c, o.ptr(), Dt::F16, none(), none())?;
        nsd.wait()?;
        Ok(x)
    }

    /// The picture from the DiT's latents float32 [16, h, w]: RGB [H, W, 3] in [-1, 1] as float32 (unclamped), width,
    /// height
    pub fn decode_f32(&self, nsd: &Nsd, lat: &[f32], h: usize, w: usize) -> Result<(Vec<f32>, usize, usize)> {
        let gpu = &self.gpu;
        let k = nextcuda_zimage::ffi::api()?;
        let ku = ffi::api()?;
        let mut zl = vec![0f32; h * w * CH];
        for c in 0..CH {
            for p in 0..h * w {
                zl[p * CH + c] = lat[c * h * w + p] / SCALE + SHIFT;
            }
        }
        let zf = DevBuf::from_f32(gpu, &zl)?;
        let x = DevBuf::new(gpu, zl.len() * 2)?;
        nsd.dequant(zf.ptr(), 0, zl.len(), x.ptr(), Dt::F16)?;
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
                ffi::check(unsafe { (ku.up2)(gpu.raw(), x.ptr(), h as i64, w as i64, c.ci as i64, u.ptr()) }, "upsample")?;
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

    /// RGB [H, W, 3] bytes from the DiT's latents, as the pipeline makes them ((x / 2 + 0.5) clamped, * 255, rounded)
    pub fn decode(&self, nsd: &Nsd, lat: &[f32], h: usize, w: usize) -> Result<(Vec<u8>, usize, usize)> {
        let (v, pw, ph) = self.decode_f32(nsd, lat, h, w)?;
        Ok((v.iter().map(|x| ((x / 2.0 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8).collect(), pw, ph))
    }
}
