//! The video VAE's decoder (diffusers' AutoencoderKLLTX2Video, LTX-2.5's config): latents [F, H, W, 128] to frames
//! [8 (F - 1) + 1, 32 H, 32 W, 3], channels-last in half on one card, decoded in the tiles the movie maker sets
//! (256 pixels, a stride of 192: spatial tiles blended over 64 pixels, every frame at once).
//!
//! ```text
//!   conv_in 128 -> 1024;  mid: 2 resnets
//!   up 0: x2 x2 x2 (1024 -> 4096, depth to space, /2) -> 512, 2 resnets
//!   up 1: x2 x2 x2 (512 -> 4096 -> 512), 4 resnets       up 2: x2 in time (512 -> 512 -> 256), 6 resnets
//!   up 3: x2 x2 in space (256 -> 512 -> 128), 4 resnets
//!   rms -> silu -> conv_out 128 -> 48 -> unpatchify 4 x 4 -> 3 channels
//!   resnet: x + conv2(silu(rms(conv1(silu(rms(x))))))    rms: per pixel over the channels (eps 1e-8, no weight)
//!   conv: 3 x 3 x 3, the time axis padded with the edge frames (non-causal), space with zeros
//! ```
//! Each up-sampler drops its first (stride_t - 1) frames: the time axis grows F -> 2F - 1 -> 4F - 3 -> 8F - 7.

use std::sync::Arc;

use nextcuda_core::{DevBuf, Error, Gpu, Result};
use nextcuda_diffusion::kernels::{none, Dt, Nsd};
use nextcuda_gguf::safetensors::SafeTensors;

use crate::ffi;

fn ge(e: nextcuda_gguf::Error) -> Error {
    Error(e.0)
}

const EPS: f32 = 1e-8;

pub struct Conv {
    w: DevBuf,
    b: DevBuf,
    pub ci: usize,
    pub co: usize,
    k: usize,
}

impl Conv {
    pub fn w_ptr(&self) -> *const std::ffi::c_void {
        self.w.ptr().cast_const()
    }
    pub fn b_ptr(&self) -> *const f32 {
        self.b.fp().cast_const()
    }
    pub fn load(st: &SafeTensors, p: &str, gpu: &Arc<Gpu>) -> Result<Conv> {
        let t = st.need(&format!("{p}.weight")).map_err(ge)?;
        let (co, ci, k) = (t.shape[0] as usize, t.shape[1] as usize, t.shape[2] as usize);
        let w: Vec<u8> = st.f32(&format!("{p}.weight")).map_err(ge)?.iter().flat_map(|v| crate::quant::f16_bits(*v).to_le_bytes()).collect();
        let wd = DevBuf::new(gpu, w.len())?;
        wd.write(0, &w)?;
        Ok(Conv { w: wd, b: DevBuf::from_f32(gpu, &st.f32(&format!("{p}.bias")).map_err(ge)?)?, ci, co, k })
    }
}

/// A channels-last half volume on a card
pub struct Vol {
    pub d: DevBuf,
    pub t: usize,
    pub h: usize,
    pub w: usize,
    pub c: usize,
}

pub struct Ctx<'a> {
    pub gpu: &'a Arc<Gpu>,
    pub nsd: &'a Nsd,
}

impl Ctx<'_> {
    pub fn vol(&self, t: usize, h: usize, w: usize, c: usize) -> Result<Vol> {
        Ok(Vol { d: DevBuf::new(self.gpu, t * h * w * c * 2)?, t, h, w, c })
    }

    /// conv (3^3: time padded with the edge frames when `causal` is false, else with the first frame twice in front)
    pub fn conv(&self, cv: &Conv, x: &Vol, causal: bool) -> Result<Vol> {
        self.conv_b(cv, x, causal, true)
    }

    /// conv, its bias added or not (a scaled input's conv adds it after the scale is undone)
    pub fn conv_b(&self, cv: &Conv, x: &Vol, causal: bool, bias: bool) -> Result<Vol> {
        let bp = if bias { cv.b.fp().cast_const() } else { std::ptr::null() };
        let frame = x.h * x.w * x.c * 2;
        if cv.k == 1 {
            let o = self.vol(x.t, x.h, x.w, cv.co)?;
            ffi::conv3d(self.nsd.raw(), x.d.ptr(), (x.t, x.h, x.w), x.c, cv.w.ptr(), cv.co, 1, bp, o.d.ptr())?;
            return Ok(o);
        }
        // the volume k - 1 frames longer - the edge frames repeated (non-causal: (k-1)/2 each side; causal: all in
        // front) - through the zero-padded conv, of whose output the frames whose window lies inside are kept
        let pad = (cv.k - 1) / 2;
        let te = x.t + cv.k - 1;
        let ext = DevBuf::new(self.gpu, te * frame)?;
        let front = if causal { cv.k - 1 } else { pad };
        for i in 0..front {
            ext.copy_within(i * frame, &x.d, 0, frame)?;
        }
        ext.copy_within(front * frame, &x.d, 0, x.t * frame)?;
        for i in front + x.t..te {
            ext.copy_within(i * frame, &x.d, (x.t - 1) * frame, frame)?;
        }
        let o = DevBuf::new(self.gpu, te * x.h * x.w * cv.co * 2)?;
        ffi::conv3d(self.nsd.raw(), ext.ptr(), (te, x.h, x.w), x.c, cv.w.ptr(), cv.co, cv.k, bp, o.ptr())?;
        let of = x.h * x.w * cv.co * 2;
        let out = self.vol(x.t, x.h, x.w, cv.co)?;
        out.d.copy_within(0, &o, pad * of, x.t * of)?;
        Ok(out)
    }

    pub fn rms_silu(&self, x: &Vol, silu: bool) -> Result<Vol> {
        let o = self.vol(x.t, x.h, x.w, x.c)?;
        ffi::rms_silu(self.gpu, x.d.ptr(), x.t * x.h * x.w, x.c, EPS, silu, o.d.ptr())?;
        Ok(o)
    }

    /// x += y (same shape)
    pub fn add(&self, x: &Vol, y: &Vol) -> Result<()> {
        self.nsd.gate_add(x.d.ptr(), Dt::F16, x.t * x.h * x.w, x.c, y.d.ptr(), Dt::F16, none(), none())
    }
}

/// A float32 channels-last volume (the encoder's residual stream: past half's range in its deep stages)
pub struct VolF {
    pub d: DevBuf,
    pub t: usize,
    pub h: usize,
    pub w: usize,
    pub c: usize,
}

impl Ctx<'_> {
    pub fn volf(&self, t: usize, h: usize, w: usize, c: usize) -> Result<VolF> {
        Ok(VolF { d: DevBuf::f32(self.gpu, t * h * w * c)?, t, h, w, c })
    }
    /// conv of a float32 volume: through half at a scale (2^-8), the scale undone in float32, then the bias
    pub fn conv_f(&self, cv: &Conv, x: &VolF, causal: bool) -> Result<VolF> {
        let s = 1.0 / 256.0;
        let n = x.t * x.h * x.w;
        let xh = self.vol(x.t, x.h, x.w, x.c)?;
        ffi::to_half_rows(self.gpu, x.d.fp(), n, x.c, x.c, s, xh.d.ptr(), x.c)?;
        let y = self.conv_b(cv, &xh, causal, false)?;
        let o = self.volf(y.t, y.h, y.w, y.c)?;
        ffi::to_float(self.gpu, y.d.ptr(), o.d.fp(), n * y.c, 1.0 / s)?;
        ffi::add_bias(self.gpu, o.d.fp(), n, y.c, cv.b.fp())?;
        Ok(o)
    }
    pub fn rms_silu_f(&self, x: &VolF) -> Result<Vol> {
        let o = self.vol(x.t, x.h, x.w, x.c)?;
        ffi::rms_silu_f32(self.gpu, x.d.fp(), x.t * x.h * x.w, x.c, EPS, true, o.d.ptr())?;
        Ok(o)
    }
}

pub struct Res {
    c1: Conv,
    c2: Conv,
}

impl Res {
    fn load(st: &SafeTensors, p: &str, gpu: &Arc<Gpu>) -> Result<Res> {
        Ok(Res { c1: Conv::load(st, &format!("{p}.conv1.conv"), gpu)?, c2: Conv::load(st, &format!("{p}.conv2.conv"), gpu)? })
    }
    /// over a float32 residual stream
    fn run_f(&self, c: &Ctx, x: VolF, causal: bool) -> Result<VolF> {
        let h = c.conv(&self.c1, &c.rms_silu_f(&x)?, causal)?;
        let h = c.conv(&self.c2, &c.rms_silu(&h, true)?, causal)?;
        c.nsd.gate_add(x.d.ptr(), Dt::F32, x.t * x.h * x.w, x.c, h.d.ptr(), Dt::F16, none(), none())?;
        Ok(x)
    }
    fn run(&self, c: &Ctx, x: Vol, causal: bool) -> Result<Vol> {
        let h = c.conv(&self.c1, &c.rms_silu(&x, true)?, causal)?;
        let h = c.conv(&self.c2, &c.rms_silu(&h, true)?, causal)?;
        c.add(&x, &h)?;
        Ok(x)
    }
}

struct Up {
    conv: Conv,
    stride: (usize, usize, usize),
    res: Vec<Res>,
}

pub struct Decoder {
    gpu: Arc<Gpu>,
    nsd: Nsd,
    conv_in: Conv,
    mid: Vec<Res>,
    ups: Vec<Up>,
    conv_out: Conv,
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

/// LTX-2.5's decoder: (resnets, up-sampler stride) per stage, after the mid block's 2
const UPS: [(usize, (usize, usize, usize)); 4] = [(2, (2, 2, 2)), (4, (2, 2, 2)), (6, (2, 1, 1)), (4, (1, 2, 2))];
/// the movie maker's tiles: 256 pixels (8 latents), a stride of 192 (6), blended over 64 pixels
pub const TILE: usize = 8;
pub const TILE_STRIDE: usize = 6;

impl Decoder {
    pub fn load(path: &std::path::Path, gpu: &Arc<Gpu>) -> Result<Decoder> {
        let st = SafeTensors::open(path).map_err(ge)?;
        let mut mid = Vec::new();
        for i in 0..2 {
            mid.push(Res::load(&st, &format!("decoder.mid_block.resnets.{i}"), gpu)?);
        }
        let mut ups = Vec::new();
        for (i, (n, stride)) in UPS.iter().enumerate() {
            let mut res = Vec::new();
            for j in 0..*n {
                res.push(Res::load(&st, &format!("decoder.up_blocks.{i}.resnets.{j}"), gpu)?);
            }
            ups.push(Up { conv: Conv::load(&st, &format!("decoder.up_blocks.{i}.upsamplers.0.conv.conv"), gpu)?, stride: *stride, res });
        }
        Ok(Decoder {
            conv_in: Conv::load(&st, "decoder.conv_in.conv", gpu)?,
            conv_out: Conv::load(&st, "decoder.conv_out.conv", gpu)?,
            mean: st.f32("latents_mean").map_err(ge)?,
            std: st.f32("latents_std").map_err(ge)?,
            mid,
            ups,
            nsd: Nsd::new(gpu)?,
            gpu: gpu.clone(),
        })
    }

    /// One tile: denormalized latents [f, h, w, 128] float32 -> pixels [8f - 7, 32h, 32w, 3] in [-1, 1]
    pub fn decode_tile(&self, z: &[f32], f: usize, h: usize, w: usize) -> Result<Vec<f32>> {
        let c = Ctx { gpu: &self.gpu, nsd: &self.nsd };
        let zh: Vec<u8> = z.iter().flat_map(|v| crate::quant::f16_bits(*v).to_le_bytes()).collect();
        let x = c.vol(f, h, w, 128)?;
        x.d.write(0, &zh)?;
        let mut x = c.conv(&self.conv_in, &x, false)?;
        for r in &self.mid {
            x = r.run(&c, x, false)?;
        }
        for u in &self.ups {
            let y = c.conv(&u.conv, &x, false)?;
            let (s0, s1, s2) = u.stride;
            let cout = y.c / (s0 * s1 * s2);
            let o = c.vol(x.t * s0 - (s0 - 1), x.h * s1, x.w * s2, cout)?;
            ffi::d2s(&self.gpu, y.d.ptr(), y.t, y.h, y.w, cout, u.stride, s0 - 1, o.d.ptr())?;
            x = o;
            for r in &u.res {
                x = r.run(&c, x, false)?;
            }
        }
        let y = c.conv(&self.conv_out, &c.rms_silu(&x, true)?, false)?;
        self.nsd.wait()?;
        let n = y.t * y.h * y.w * y.c;
        let tmp = DevBuf::f32(&self.gpu, n)?;
        ffi::to_float(&self.gpu, y.d.ptr(), tmp.fp(), n, 1.0)?;
        self.nsd.wait()?;
        let v = tmp.to_f32()?;
        // unpatchify: channel c * 16 + a * 4 + b is pixel (y 4 + b, x 4 + a) of colour c
        let (ho, wo) = (y.h * 4, y.w * 4);
        let mut out = vec![0f32; y.t * ho * wo * 3];
        for t in 0..y.t {
            for yy in 0..y.h {
                for xx in 0..y.w {
                    let src = &v[((t * y.h + yy) * y.w + xx) * 48..][..48];
                    for col in 0..3 {
                        for a in 0..4 {
                            for b in 0..4 {
                                out[((t * ho + yy * 4 + b) * wo + xx * 4 + a) * 3 + col] = src[col * 16 + a * 4 + b];
                            }
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    /// The pipeline's decode: latents [F, H, W, 128] normalized (the DiT's) -> frames [8F - 7, 32H, 32W, 3] in [0, 1],
    /// tiled as diffusers' tiled_decode (each tile blended into the ones above and left of it, in place)
    pub fn decode(&self, lat: &[f32], f: usize, h: usize, w: usize, progress: &mut dyn FnMut(usize, usize)) -> Result<Vec<f32>> {
        let z: Vec<f32> = lat.iter().enumerate().map(|(i, v)| v * self.std[i % 128] + self.mean[i % 128]).collect();
        self.decode_raw(&z, f, h, w, (TILE, TILE_STRIDE), progress)
    }

    /// Denormalized latents -> frames in [0, 1], in spatial tiles of `tile` = (size, stride) latents (diffusers'
    /// tiled_decode; the tiles blended over (size - stride) x 32 pixels)
    pub fn decode_raw(&self, z: &[f32], f: usize, h: usize, w: usize, tile: (usize, usize), progress: &mut dyn FnMut(usize, usize)) -> Result<Vec<f32>> {
        let (tile_n, tile_s) = tile;
        let tiled = h > tile_n || w > tile_n;
        let (tl, ts) = if tiled { (tile_n, tile_s) } else { (h.max(w), h.max(w)) };
        let fo = 8 * (f - 1) + 1;
        let (ho, wo) = (32 * h, 32 * w);
        let rows: Vec<usize> = (0..h).step_by(ts).collect();
        let cols: Vec<usize> = (0..w).step_by(ts).collect();
        let total = rows.len() * cols.len();
        // the decoded tiles: (pixels, height, width)
        let mut tiles: Vec<Vec<(Vec<f32>, usize, usize)>> = Vec::new();
        for (ri, &i) in rows.iter().enumerate() {
            let mut row = Vec::new();
            for (ci, &j) in cols.iter().enumerate() {
                let (th, tw) = (tl.min(h - i), tl.min(w - j));
                let mut sub = vec![0f32; f * th * tw * 128];
                for t in 0..f {
                    for y in 0..th {
                        let s = ((t * h + i + y) * w + j) * 128;
                        let d = (t * th + y) * tw * 128;
                        sub[d..d + tw * 128].copy_from_slice(&z[s..s + tw * 128]);
                    }
                }
                row.push((self.decode_tile(&sub, f, th, tw)?, th * 32, tw * 32));
                progress(ri * cols.len() + ci + 1, total);
            }
            tiles.push(row);
        }
        if !tiled {
            return Ok(tiles.remove(0).remove(0).0.iter().map(|v| (v / 2.0 + 0.5).clamp(0.0, 1.0)).collect());
        }
        let blend = 32 * (tile_n - tile_s);
        let stride = 32 * tile_s;
        for ri in 0..tiles.len() {
            for ci in 0..tiles[ri].len() {
                if ri > 0 {
                    let (above, cur) = { let (a, b) = tiles.split_at_mut(ri); (&a[ri - 1][ci], &mut b[0][ci]) };
                    let e = blend.min(above.1).min(cur.1);
                    let (ah, aw) = (above.1, above.2);
                    let cw = cur.2;
                    for t in 0..fo {
                        for y in 0..e {
                            let k = y as f32 / e as f32;
                            for x in 0..cw.min(aw) {
                                for col in 0..3 {
                                    let av = above.0[((t * ah + ah - e + y) * aw + x) * 3 + col];
                                    let bi = ((t * cur.1 + y) * cw + x) * 3 + col;
                                    cur.0[bi] = av * (1.0 - k) + cur.0[bi] * k;
                                }
                            }
                        }
                    }
                }
                if ci > 0 {
                    let (left, cur) = { let (a, b) = tiles[ri].split_at_mut(ci); (&a[ci - 1], &mut b[0]) };
                    let e = blend.min(left.2).min(cur.2);
                    let (lh, lw) = (left.1, left.2);
                    for t in 0..fo {
                        for y in 0..cur.1.min(lh) {
                            for x in 0..e {
                                let k = x as f32 / e as f32;
                                for col in 0..3 {
                                    let av = left.0[((t * lh + y) * lw + lw - e + x) * 3 + col];
                                    let bi = ((t * cur.1 + y) * cur.2 + x) * 3 + col;
                                    cur.0[bi] = av * (1.0 - k) + cur.0[bi] * k;
                                }
                            }
                        }
                    }
                }
            }
        }
        // each tile's first stride x stride pixels, side by side
        let mut out = vec![0f32; fo * ho * wo * 3];
        for (ri, row) in tiles.iter().enumerate() {
            for (ci, (px, th, tw)) in row.iter().enumerate() {
                let (y0, x0) = (ri * stride, ci * stride);
                for t in 0..fo {
                    for y in 0..stride.min(*th).min(ho.saturating_sub(y0)) {
                        for x in 0..stride.min(*tw).min(wo.saturating_sub(x0)) {
                            for col in 0..3 {
                                out[((t * ho + y0 + y) * wo + x0 + x) * 3 + col] = (px[((t * th + y) * tw + x) * 3 + col] / 2.0 + 0.5).clamp(0.0, 1.0);
                            }
                        }
                    }
                }
            }
        }
        Ok(out)
    }
}

// ---- the encoder ---------------------------------------------------------------------------------------------------

struct Down {
    res: Vec<Res>,
    conv: Conv,
    stride: (usize, usize, usize),
    /// the channels after it
    out: usize,
}

/// The video VAE's encoder (causal): frames [F, 32H, 32W, 3] in [-1, 1] -> the latent means [(F - 1) / 8 + 1, H, W, 128]
///
/// ```text
///   patchify 4 x 4 (channel c * 16 + x * 4 + y) -> conv_in 48 -> 128
///   down 0: 4 resnets 128, space / 2 -> 256      down 1: 6 resnets 256, time / 2 -> 512
///   down 2: 4 resnets 512, both / 2 -> 1024      down 3: 2 resnets 1024, both / 2 -> 1024;  mid: 2 resnets
///   rms, silu, conv_out -> 129 (the means, a log-variance)
///   down-sampler: the first (s_t - 1) frames repeated in front; conv(x) to out / (s_t s_h s_w), space to depth, plus
///                 x space-to-depth averaged in groups to the out channels
/// ```
pub struct Encoder {
    gpu: Arc<Gpu>,
    nsd: Nsd,
    conv_in: Conv,
    downs: Vec<Down>,
    mid: Vec<Res>,
    conv_out: Conv,
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

const DOWNS: [(usize, (usize, usize, usize), usize); 4] = [(4, (1, 2, 2), 256), (6, (2, 1, 1), 512), (4, (2, 2, 2), 1024), (2, (2, 2, 2), 1024)];

impl Encoder {
    pub fn load(path: &std::path::Path, gpu: &Arc<Gpu>) -> Result<Encoder> {
        let st = SafeTensors::open(path).map_err(ge)?;
        let mut downs = Vec::new();
        for (i, (n, stride, out)) in DOWNS.iter().enumerate() {
            let res = (0..*n).map(|j| Res::load(&st, &format!("encoder.down_blocks.{i}.resnets.{j}"), gpu)).collect::<Result<Vec<_>>>()?;
            downs.push(Down { res, conv: Conv::load(&st, &format!("encoder.down_blocks.{i}.downsamplers.0.conv.conv"), gpu)?, stride: *stride, out: *out });
        }
        Ok(Encoder {
            conv_in: Conv::load(&st, "encoder.conv_in.conv", gpu)?,
            mid: (0..2).map(|j| Res::load(&st, &format!("encoder.mid_block.resnets.{j}"), gpu)).collect::<Result<Vec<_>>>()?,
            conv_out: Conv::load(&st, "encoder.conv_out.conv", gpu)?,
            mean: st.f32("latents_mean").map_err(ge)?,
            std: st.f32("latents_std").map_err(ge)?,
            downs,
            nsd: Nsd::new(gpu)?,
            gpu: gpu.clone(),
        })
    }

    /// Frames [f, h, w, 3] in [-1, 1] (f = 8k + 1, h and w multiples of 32) -> latent means [(f-1)/8 + 1, h/32, w/32, 128]
    pub fn encode(&self, px: &[f32], f: usize, h: usize, w: usize) -> Result<Vec<f32>> {
        let c = Ctx { gpu: &self.gpu, nsd: &self.nsd };
        let (hp, wp) = (h / 4, w / 4);
        // patchify: channel c * 16 + x * 4 + y
        let mut v = vec![0u16; f * hp * wp * 48];
        for t in 0..f {
            for yy in 0..hp {
                for xx in 0..wp {
                    for col in 0..3 {
                        for a in 0..4 {
                            for b in 0..4 {
                                let p = px[((t * h + yy * 4 + b) * w + xx * 4 + a) * 3 + col];
                                v[((t * hp + yy) * wp + xx) * 48 + col * 16 + a * 4 + b] = crate::quant::f16_bits(p);
                            }
                        }
                    }
                }
            }
        }
        let x = c.vol(f, hp, wp, 48)?;
        x.d.write(0, &v.iter().flat_map(|q| q.to_le_bytes()).collect::<Vec<u8>>())?;
        let x0 = c.conv(&self.conv_in, &x, true)?;
        // the residual stream in float32 from here (down 3's outputs pass 2e5)
        let mut x = c.volf(x0.t, x0.h, x0.w, x0.c)?;
        ffi::to_float(&self.gpu, x0.d.ptr(), x.d.fp(), x0.t * x0.h * x0.w * x0.c, 1.0)?;
        drop(x0);
        // LTX2_DEBUG_ENC=DIR: each stage's output as raw float32 (a stage check against ref_enc.py's)
        let dump = std::env::var("LTX2_DEBUG_ENC").ok();
        let save = |v: &VolF, name: &str| -> Result<()> {
            if let Some(d) = &dump {
                self.nsd.wait()?;
                std::fs::write(std::path::Path::new(d).join(format!("{name}.f32")), crate::te::as_bytes(&v.d.to_f32()?)).ok();
            }
            Ok(())
        };
        save(&x, "enc_conv_in")?;
        for (di, d) in self.downs.iter().enumerate() {
            for r in &d.res {
                x = r.run_f(&c, x, true)?;
            }
            let (s0, s1, s2) = d.stride;
            // the first s0 - 1 frames repeated in front
            let x2 = if s0 > 1 {
                let frame = x.h * x.w * x.c * 4;
                let e = c.volf(x.t + s0 - 1, x.h, x.w, x.c)?;
                e.d.copy_within(0, &x.d, 0, (s0 - 1) * frame)?;
                e.d.copy_within((s0 - 1) * frame, &x.d, 0, x.t * frame)?;
                e
            } else {
                x
            };
            let k = s0 * s1 * s2;
            let groups = x2.c * k / d.out;
            let res = c.volf(x2.t / s0, x2.h / s1, x2.w / s2, d.out)?;
            ffi::s2d_f32(&self.gpu, x2.d.fp(), x2.t, x2.h, x2.w, x2.c, d.stride, groups, res.d.fp())?;
            let y = c.conv_f(&d.conv, &x2, true)?;
            let o = c.volf(y.t / s0, y.h / s1, y.w / s2, d.out)?;
            ffi::s2d_f32(&self.gpu, y.d.fp(), y.t, y.h, y.w, y.c, d.stride, 1, o.d.fp())?;
            self.nsd.gate_add(o.d.ptr(), Dt::F32, o.t * o.h * o.w, o.c, res.d.ptr(), Dt::F32, none(), none())?;
            x = o;
            save(&x, &format!("enc_down{di}"))?;
        }
        for r in &self.mid {
            x = r.run_f(&c, x, true)?;
        }
        save(&x, "enc_mid")?;
        let y = c.conv(&self.conv_out, &c.rms_silu_f(&x)?, true)?;
        if let Some(d) = &dump {
            let n = y.t * y.h * y.w * y.c;
            let tmp = DevBuf::f32(&self.gpu, n)?;
            ffi::to_float(&self.gpu, y.d.ptr(), tmp.fp(), n, 1.0)?;
            self.nsd.wait()?;
            std::fs::write(std::path::Path::new(d).join("enc_out.f32"), crate::te::as_bytes(&tmp.to_f32()?)).ok();
        }
        let n = y.t * y.h * y.w * y.c;
        let tmp = DevBuf::f32(&self.gpu, n)?;
        ffi::to_float(&self.gpu, y.d.ptr(), tmp.fp(), n, 1.0)?;
        self.nsd.wait()?;
        let all = tmp.to_f32()?;
        // the means: the first 128 of 129 channels
        Ok(all.chunks_exact(y.c).flat_map(|p| p[..128].iter().copied()).collect())
    }

    /// Frames as above, in spatial tiles of `tile` = (size, stride) pixels (diffusers' tiled_encode: the latent tiles
    /// blended over (size - stride) / 32 latents, in place, each cut to stride / 32)
    pub fn encode_tiled(&self, px: &[f32], f: usize, h: usize, w: usize, tile: (usize, usize)) -> Result<Vec<f32>> {
        let (tn, ts) = tile;
        if h <= tn && w <= tn {
            return self.encode(px, f, h, w);
        }
        let (lf, lh, lw) = ((f - 1) / 8 + 1, h / 32, w / 32);
        let (ltn, lts) = (tn / 32, ts / 32);
        let blend = ltn - lts;
        let rows: Vec<usize> = (0..h).step_by(ts).collect();
        let cols: Vec<usize> = (0..w).step_by(ts).collect();
        // (latents [lf, th, tw, 128], th, tw)
        let mut tiles: Vec<Vec<(Vec<f32>, usize, usize)>> = Vec::new();
        for &i in &rows {
            let mut row = Vec::new();
            for &j in &cols {
                let (ph, pw) = (tn.min(h - i), tn.min(w - j));
                let mut sub = vec![0f32; f * ph * pw * 3];
                for t in 0..f {
                    for y in 0..ph {
                        let s = ((t * h + i + y) * w + j) * 3;
                        let d = (t * ph + y) * pw * 3;
                        sub[d..d + pw * 3].copy_from_slice(&px[s..s + pw * 3]);
                    }
                }
                row.push((self.encode(&sub, f, ph, pw)?, ph / 32, pw / 32));
            }
            tiles.push(row);
        }
        for ri in 0..tiles.len() {
            for ci in 0..tiles[ri].len() {
                if ri > 0 {
                    let (a, b) = tiles.split_at_mut(ri);
                    let (above, cur) = (&a[ri - 1][ci], &mut b[0][ci]);
                    let e = blend.min(above.1).min(cur.1);
                    for t in 0..lf {
                        for y in 0..e {
                            let k = y as f32 / e as f32;
                            for x in 0..cur.2.min(above.2) {
                                for c in 0..128 {
                                    let av = above.0[((t * above.1 + above.1 - e + y) * above.2 + x) * 128 + c];
                                    let bi = ((t * cur.1 + y) * cur.2 + x) * 128 + c;
                                    cur.0[bi] = av * (1.0 - k) + cur.0[bi] * k;
                                }
                            }
                        }
                    }
                }
                if ci > 0 {
                    let (a, b) = tiles[ri].split_at_mut(ci);
                    let (left, cur) = (&a[ci - 1], &mut b[0]);
                    let e = blend.min(left.2).min(cur.2);
                    for t in 0..lf {
                        for y in 0..cur.1.min(left.1) {
                            for x in 0..e {
                                let k = x as f32 / e as f32;
                                for c in 0..128 {
                                    let av = left.0[((t * left.1 + y) * left.2 + left.2 - e + x) * 128 + c];
                                    let bi = ((t * cur.1 + y) * cur.2 + x) * 128 + c;
                                    cur.0[bi] = av * (1.0 - k) + cur.0[bi] * k;
                                }
                            }
                        }
                    }
                }
            }
        }
        let mut out = vec![0f32; lf * lh * lw * 128];
        for (ri, row) in tiles.iter().enumerate() {
            for (ci, (z, th, tw)) in row.iter().enumerate() {
                let (y0, x0) = (ri * lts, ci * lts);
                for t in 0..lf {
                    for y in 0..lts.min(*th).min(lh.saturating_sub(y0)) {
                        for x in 0..lts.min(*tw).min(lw.saturating_sub(x0)) {
                            let s = ((t * th + y) * tw + x) * 128;
                            let d = ((t * lh + y0 + y) * lw + x0 + x) * 128;
                            out[d..d + 128].copy_from_slice(&z[s..s + 128]);
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    /// The latents normalized as the pipeline's DiT reads them ((z - mean) / std)
    pub fn normalize(&self, z: &mut [f32]) {
        for (i, v) in z.iter_mut().enumerate() {
            *v = (*v - self.mean[i % 128]) / self.std[i % 128];
        }
    }
}
