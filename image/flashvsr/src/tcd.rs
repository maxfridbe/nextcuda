//! The tiny conditional decoder (FlashVSR's `TCDecoder`: TAEHV's decoder, widened to [512, 256, 128, 128] and
//! conditioned on the low-quality frames): latents to frames, 4 a latent frame, with no VAE.
//!
//! ```text
//!   input  [h8, w8, 784]: the 4 cond frames 8 x 8-unshuffled (768) and the latent (16), then 3 tanh(x / 3)
//!   1 conv 784->512, ReLU, 3 conv 512->512, ReLU, 5-7 MemBlock 512, 9 TGrow x1 (1x1 512->512)
//!   10 up 2x, conv 512->256, 11-13 MemBlock 256, 15 TGrow x2 (1x1 256->512: two frames)
//!   16 up 2x, conv 256->128, 17-19 MemBlock 128, 21 TGrow x2 (1x1 128->256: two frames)
//!   22 up 2x, conv 128->128, ReLU, 24 conv 128->128, ReLU, 26 conv 128->3   -> [H, W, 3] in about [0, 1]
//!   MemBlock(x, past) = ReLU(conv(ReLU(conv(ReLU(conv([x, past])))) + x)), past = the block's input one frame before
//! ```
//!
//! Frames go through one at a time, in order (the reference's sequential mode), each block's past kept. A TGrow's 1x1
//! convolution is done before the nearest upsample in front of the next convolution (they commute), and the upsample is
//! read in place by the unfolding. The stream's first 3 frames are dropped (the reference trims them).

use std::sync::Arc;

use nextcuda_core::{DevBuf, Error, Gpu, Result};
use nextcuda_diffusion::kernels::{Dt, Nsd};
use nextcuda_gguf::safetensors::SafeTensors;

use crate::dit::Lin;
use crate::ffi::{self, check, g};

const COL_BYTES: usize = 192 << 20;

/// A 3 x 3 conv's weights [co, ci, 3, 3] reordered to [co, 3, 3, ci]; a 1x1's as they are
fn conv_lin(gpu: &Arc<Gpu>, st: &SafeTensors, name: &str, bias: bool) -> Result<Lin> {
    let t = st.need(&format!("{name}.weight")).map_err(|e| Error(e.0))?;
    let (co, ci, kh) = (t.shape[0] as usize, t.shape[1] as usize, t.shape[2] as usize);
    let w = st.f32(&format!("{name}.weight")).map_err(|e| Error(e.0))?;
    let taps = kh * kh;
    let mut r = vec![0f32; w.len()];
    for o in 0..co {
        for c in 0..ci {
            for k in 0..taps {
                r[(o * taps + k) * ci + c] = w[(o * ci + c) * taps + k];
            }
        }
    }
    let b = if bias { Some(st.f32(&format!("{name}.bias")).map_err(|e| Error(e.0))?) } else { None };
    Lin::from(gpu, &r, b.as_deref(), co, taps * ci)
}

struct Mem {
    c1: Lin,
    c2: Lin,
    c3: Lin,
}

pub struct Tcd {
    gpu: Arc<Gpu>,
    c1: Lin,
    c3: Lin,
    mb0: [Mem; 3],
    tg9: Lin,
    c10: Lin,
    mb1: [Mem; 3],
    tg15: Lin,
    c16: Lin,
    mb2: [Mem; 3],
    tg21: Lin,
    c22: Lin,
    c24: Lin,
    c26: Lin,
    pub bytes: usize,
    state: Option<State>,
}

struct State {
    h8: usize,
    w8: usize,
    /// each MemBlock's past (its input one frame before), and whether it has one yet
    past: Vec<DevBuf>,
    have: Vec<bool>,
    trim: usize,
    inp: DevBuf,
    a: DevBuf,
    b: DevBuf,
    t1: DevBuf,
    t2: DevBuf,
    tg: DevBuf,
    tg2: DevBuf,
    f0: DevBuf,
    f1: DevBuf,
    col: DevBuf,
    pub out: DevBuf,
}

impl Tcd {
    pub fn load(st: &SafeTensors, gpu: &Arc<Gpu>) -> Result<Tcd> {
        let c = |n: &str, b: bool| conv_lin(gpu, st, &format!("decoder.{n}"), b);
        let mem = |i: usize| -> Result<Mem> { Ok(Mem { c1: c(&format!("{i}.conv.0"), true)?, c2: c(&format!("{i}.conv.2"), true)?, c3: c(&format!("{i}.conv.4"), true)? }) };
        let t = Tcd {
            gpu: gpu.clone(),
            c1: c("1", true)?,
            c3: c("3", false)?,
            mb0: [mem(5)?, mem(6)?, mem(7)?],
            tg9: c("9.conv", false)?,
            c10: c("10", false)?,
            mb1: [mem(11)?, mem(12)?, mem(13)?],
            tg15: c("15.conv", false)?,
            c16: c("16", false)?,
            mb2: [mem(17)?, mem(18)?, mem(19)?],
            tg21: c("21.conv", false)?,
            c22: c("22", false)?,
            c24: c("24", false)?,
            c26: c("26", true)?,
            bytes: 0,
            state: None,
        };
        let mut bytes = [&t.c1, &t.c3, &t.tg9, &t.c10, &t.tg15, &t.c16, &t.tg21, &t.c22, &t.c24, &t.c26].iter().map(|l| l.bytes()).sum::<usize>();
        for m in t.mb0.iter().chain(&t.mb1).chain(&t.mb2) {
            bytes += m.c1.bytes() + m.c2.bytes() + m.c3.bytes();
        }
        Ok(Tcd { bytes, ..t })
    }

    pub fn need_bytes(h8: usize, w8: usize) -> usize {
        let p8 = h8 * w8;
        // pasts: 3 x (512 at 1/8, 256 at 1/4, 128 at 1/2); buffers at full size: 3 x 128 channels; the rest smaller
        let past = 3 * (p8 * 512 + 4 * p8 * 256 + 16 * p8 * 128) * 2;
        let full = 64 * p8 * 128 * 2;
        past + 3 * full + 16 * p8 * 256 * 2 * 2 + p8 * 784 * 2 + COL_BYTES + 64 * p8 * 3 * 2
    }

    /// A new stream for latents of [h8, w8] (the pasts emptied, the first 3 frames to drop)
    pub fn reset(&mut self, h8: usize, w8: usize) -> Result<()> {
        if !self.state.as_ref().is_some_and(|s| s.h8 == h8 && s.w8 == w8) {
            self.state = None;
            let gpu = &self.gpu;
            let p8 = h8 * w8;
            let mut past = Vec::new();
            for (c, px) in [(512, p8), (256, 4 * p8), (128, 16 * p8)] {
                for _ in 0..3 {
                    past.push(DevBuf::new(gpu, px * c * 2)?);
                }
            }
            let full = 64 * p8 * 128 * 2;
            self.state = Some(State {
                h8,
                w8,
                past,
                have: vec![false; 9],
                trim: 3,
                inp: DevBuf::new(gpu, p8 * 784 * 2)?,
                a: DevBuf::new(gpu, full)?,
                b: DevBuf::new(gpu, full)?,
                t1: DevBuf::new(gpu, full)?,
                t2: DevBuf::new(gpu, 16 * p8 * 128 * 2)?,
                tg: DevBuf::new(gpu, 4 * p8 * 512 * 2)?,
                tg2: DevBuf::new(gpu, 16 * p8 * 256 * 2)?,
                f0: DevBuf::new(gpu, 16 * p8 * 128 * 2)?,
                f1: DevBuf::new(gpu, 16 * p8 * 128 * 2)?,
                col: DevBuf::new(gpu, COL_BYTES)?,
                out: DevBuf::new(gpu, 64 * p8 * 3 * 2)?,
            });
        }
        let s = self.state.as_mut().unwrap();
        s.have.iter_mut().for_each(|h| *h = false);
        s.trim = 3;
        Ok(())
    }

    /// A 3 x 3 conv at h x w over a (lda halves a pixel, ca channels; `up`: a at half the size, upsampled), with b's
    /// channels after them (a MemBlock's past; cb 0: none) -> out [h w, co]
    #[allow(clippy::too_many_arguments)]
    fn conv(&self, nsd: &Nsd, lin: &Lin, a: *const std::ffi::c_void, lda: usize, ca: usize, mem: Option<*const std::ffi::c_void>, cb: usize,
            h: usize, w: usize, up: bool, col: &DevBuf, out: &DevBuf) -> Result<()> {
        let k = ffi::api()?;
        let px = h * w;
        let kk = 9 * (ca + cb);
        if kk != lin.k {
            return Err(Error(format!("flashvsr decoder: a conv of {} inputs fed {kk}", lin.k)));
        }
        let tile = (COL_BYTES / (kk * 2)).max(1);
        let mut p0 = 0;
        while p0 < px {
            let n = tile.min(px - p0);
            let m = mem.unwrap_or(std::ptr::null());
            // SAFETY: a holds (h/u) x (w/u) x lda halves, col n x kk, out px x co.
            check(unsafe { (k.im2col2)(g(&self.gpu), a, lda as i64, ca as i64, m, cb as i64, 1, h as i64, w as i64, up as i32, p0 as i64, n as i64,
                                       col.ptr()) }, "decoder unfold")?;
            let o = unsafe { (out.ptr() as *mut u8).add(p0 * lin.n * 2) };
            lin.apply(nsd, col.ptr(), n, o as *mut _, Dt::F16)?;
            p0 += n;
        }
        Ok(())
    }

    fn relu(&self, x: &DevBuf, n: usize) -> Result<()> {
        let k = ffi::api()?;
        // SAFETY: x holds n halves.
        check(unsafe { (k.relu)(g(&self.gpu), x.ptr(), n as i64) }, "relu")
    }

    /// MemBlock `i` over x [h w, c] in place (t1, t2 its scratch)
    #[allow(clippy::too_many_arguments)]
    fn memblock(&self, nsd: &Nsd, m: &Mem, i: usize, s: &mut State, x: &DevBuf, c: usize, h: usize, w: usize, t1: &DevBuf, t2: &DevBuf) -> Result<()> {
        let k = ffi::api()?;
        let px = h * w;
        let past = if s.have[i] { Some(s.past[i].ptr() as *const _) } else { None };
        // a missing past is zeros: the unfolding reads none
        self.conv(nsd, &m.c1, x.ptr(), c, c, past, c, h, w, false, &s.col, t1)?;
        self.relu(t1, px * c)?;
        self.conv(nsd, &m.c2, t1.ptr(), c, c, None, 0, h, w, false, &s.col, t2)?;
        self.relu(t2, px * c)?;
        self.conv(nsd, &m.c3, t2.ptr(), c, c, None, 0, h, w, false, &s.col, t1)?;
        s.past[i].copy_within(0, x, 0, px * c * 2)?;
        s.have[i] = true;
        // SAFETY: x and t1 hold px x c halves.
        check(unsafe { (k.add_relu)(g(&self.gpu), x.ptr(), t1.ptr(), (px * c) as i64) }, "memblock")
    }

    /// One latent frame t of lat (float32 [16, f, h8, w8] on this card) with its 4 cond frames (half [4, H, W, 3]):
    /// each frame made (half [H, W, 3] in `out`) given to `emit` - none for the stream's first 3
    pub fn frame(&mut self, nsd: &Nsd, lat: &DevBuf, f: usize, t: usize, cond: *const std::ffi::c_void,
                 emit: &mut dyn FnMut(&DevBuf) -> Result<()>) -> Result<()> {
        let k = ffi::api()?;
        let gp = g(&self.gpu);
        let mut s = self.state.take().ok_or_else(|| Error("flashvsr: the decoder was not reset".into()))?;
        let r = self.frame_in(nsd, &mut s, lat, f, t, cond, emit, k, gp);
        self.state = Some(s);
        r
    }

    #[allow(clippy::too_many_arguments)]
    fn frame_in(&self, nsd: &Nsd, s: &mut State, lat: &DevBuf, f: usize, t: usize, cond: *const std::ffi::c_void,
                emit: &mut dyn FnMut(&DevBuf) -> Result<()>, k: &ffi::Api, gp: *mut std::ffi::c_void) -> Result<()> {
        let (h8, w8) = (s.h8, s.w8);
        // SAFETY: cond holds 4 frames of 8 h8 x 8 w8 x 3 halves, lat 16 x f x h8 x w8 floats, inp h8 x w8 x 784 halves.
        check(unsafe { (k.tcd_input)(gp, cond, lat.fp(), f as i64, t as i64, h8 as i64, w8 as i64, s.inp.ptr()) }, "decoder input")?;
        let p8 = h8 * w8;
        let (a, b, t1) = (s.a.view(0, p8 * 512 * 2)?, s.b.view(0, p8 * 512 * 2)?, s.t1.view(0, p8 * 512 * 2)?);
        self.conv(nsd, &self.c1, s.inp.ptr(), 784, 784, None, 0, h8, w8, false, &s.col, &a)?;
        self.relu(&a, p8 * 512)?;
        self.conv(nsd, &self.c3, a.ptr(), 512, 512, None, 0, h8, w8, false, &s.col, &b)?;
        self.relu(&b, p8 * 512)?;
        for (i, m) in self.mb0.iter().enumerate() {
            self.memblock(nsd, m, i, s, &b, 512, h8, w8, &t1, &a)?;
        }
        self.tg9.apply(nsd, b.ptr(), p8, s.tg.ptr(), Dt::F16)?;
        let (h4, w4) = (2 * h8, 2 * w8);
        let p4 = h4 * w4;
        let (a4, b4) = (s.a.view(0, p4 * 256 * 2)?, s.b.view(0, p4 * 256 * 2)?);
        let t14 = s.t1.view(0, p4 * 256 * 2)?;
        self.conv(nsd, &self.c10, s.tg.ptr(), 512, 512, None, 0, h4, w4, true, &s.col, &b4)?;
        for (i, m) in self.mb1.iter().enumerate() {
            self.memblock(nsd, m, 3 + i, s, &b4, 256, h4, w4, &t14, &a4)?;
        }
        // two frames: channels [0, 256) and [256, 512) of the 1x1's output
        self.tg15.apply(nsd, b4.ptr(), p4, s.tg2.ptr(), Dt::F16)?;
        let (h2, w2) = (2 * h4, 2 * w4);
        let p2 = h2 * w2;
        for sub in 0..2 {
            let src = unsafe { (s.tg2.ptr() as *mut u8).add(sub * 256 * 2) };
            let f0 = s.f0.view(0, p2 * 128 * 2)?;
            let f1 = s.f1.view(0, p2 * 128 * 2)?;
            let t22 = s.t2.view(0, p2 * 128 * 2)?;
            self.conv(nsd, &self.c16, src as *const _, 512, 256, None, 0, h2, w2, true, &s.col, &f0)?;
            for (i, m) in self.mb2.iter().enumerate() {
                self.memblock(nsd, m, 6 + i, s, &f0, 128, h2, w2, &f1, &t22)?;
            }
            // two frames again: channels [0, 128) and [128, 256)
            let tg3 = s.a.view(0, p2 * 256 * 2)?;
            self.tg21.apply(nsd, f0.ptr(), p2, tg3.ptr(), Dt::F16)?;
            let (h1, w1) = (2 * h2, 2 * w2);
            let p1 = h1 * w1;
            for sub2 in 0..2 {
                let src = unsafe { (tg3.ptr() as *mut u8).add(sub2 * 128 * 2) };
                let x = s.b.view(0, p1 * 128 * 2)?;
                let y = s.t1.view(0, p1 * 128 * 2)?;
                self.conv(nsd, &self.c22, src as *const _, 256, 128, None, 0, h1, w1, true, &s.col, &x)?;
                self.relu(&x, p1 * 128)?;
                self.conv(nsd, &self.c24, x.ptr(), 128, 128, None, 0, h1, w1, false, &s.col, &y)?;
                self.relu(&y, p1 * 128)?;
                self.conv(nsd, &self.c26, y.ptr(), 128, 128, None, 0, h1, w1, false, &s.col, &s.out)?;
                if s.trim > 0 {
                    s.trim -= 1;
                } else {
                    emit(&s.out)?;
                }
            }
        }
        Ok(())
    }
}
