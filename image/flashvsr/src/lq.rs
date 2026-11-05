//! The LQ projection (FlashVSR v1.1's `Causal_LQ4x_Proj`): the upscaled low-quality frames into the DiT, one latent
//! frame's tokens for each 4 frames, added to the tokens before block 0.
//!
//! ```text
//!   4 frames [4, H, W, 3] -> 16 x 16 pixel unshuffle [4, H/16, W/16, 768]
//!   -> causal conv3d 768 -> 2048, kernel (4, 3, 3), stride (2, 1, 1), the two frames before in front -> [2, ...]
//!   -> rms over the channels * sqrt(C) * gamma, SiLU
//!   -> causal conv3d 2048 -> 3072, the same over the two frames before and these two -> [1, ...] -> rms, SiLU
//!   -> linear 3072 -> 1536: [H/16 * W/16, 1536]
//! ```
//!
//! "Causal": each convolution sees the last two frames of its input from the call before (the stream's memory); the
//! first call has none - its one frame is taken 4 times, the convolutions' front padding repeats it, and it returns
//! nothing (it only fills the memory). Spatial padding repeats the border (the reference pads with 'replicate').
//! The convolutions are unfolded in tiles of pixels and multiplied by cuBLAS, their weights reordered at load to
//! (dt, dy, dx, c) to match.

use std::sync::Arc;

use nextcuda_core::{DevBuf, Error, Gpu, Result};
use nextcuda_diffusion::kernels::{Dt, Nsd};
use nextcuda_gguf::safetensors::SafeTensors;

use crate::dit::Lin;
use crate::ffi::{self, check, g};

const C0: usize = 768;
const C1: usize = 2048;
const C2: usize = 3072;
/// bytes of unfolded columns at a time
const COL_BYTES: usize = 192 << 20;

/// A conv3d's weights [co, ci, kt, 3, 3] reordered to [co, kt, 3, 3, ci] and its bias
fn conv_lin(gpu: &Arc<Gpu>, w: &[f32], b: &[f32], co: usize, ci: usize, kt: usize) -> Result<Lin> {
    let taps = kt * 9;
    let mut r = vec![0f32; w.len()];
    for o in 0..co {
        for c in 0..ci {
            for t in 0..taps {
                r[(o * taps + t) * ci + c] = w[(o * ci + c) * taps + t];
            }
        }
    }
    Lin::from(gpu, &r, Some(b), co, taps * ci)
}

pub struct LqProj {
    gpu: Arc<Gpu>,
    conv1: Lin,
    conv2: Lin,
    g1: DevBuf,
    g2: DevBuf,
    lin: Lin,
    pub bytes: usize,
    state: Option<State>,
}

/// The stream's memory and buffers, for one frame size
struct State {
    hp: usize,
    wp: usize,
    calls: usize,
    /// conv1's input: the two frames before, then the call's four [6, hp, wp, 768]
    in1: DevBuf,
    /// conv2's input: the two frames before, then conv1's two [4, hp, wp, 2048]
    in2: DevBuf,
    out1: DevBuf,
    out2: DevBuf,
    col: DevBuf,
    frames: DevBuf,
}

impl LqProj {
    pub fn load(st: &SafeTensors, gpu: &Arc<Gpu>) -> Result<LqProj> {
        let f = |n: &str| st.f32(n).map_err(|e| Error(e.0));
        let conv1 = conv_lin(gpu, &f("conv1.weight")?, &f("conv1.bias")?, C1, C0, 4)?;
        let conv2 = conv_lin(gpu, &f("conv2.weight")?, &f("conv2.bias")?, C2, C1, 4)?;
        let lin = Lin::from(gpu, &f("linear_layers.0.weight")?, Some(&f("linear_layers.0.bias")?), 1536, C2)?;
        let bytes = conv1.bytes() + conv2.bytes() + lin.bytes();
        Ok(LqProj { gpu: gpu.clone(), conv1, conv2, g1: DevBuf::from_f32(gpu, &f("norm1.gamma")?)?, g2: DevBuf::from_f32(gpu, &f("norm2.gamma")?)?, lin,
                    bytes, state: None })
    }

    pub fn need_bytes(hp: usize, wp: usize) -> usize {
        let px = hp * wp;
        px * (6 * C0 + 4 * C1 + 2 * C1 + C2) * 2 + COL_BYTES + 4 * 16 * hp * 16 * wp * 3 * 2
    }

    /// A new stream for frames of [16 hp, 16 wp] (the memory emptied)
    pub fn reset(&mut self, hp: usize, wp: usize) -> Result<()> {
        if !self.state.as_ref().is_some_and(|s| s.hp == hp && s.wp == wp) {
            self.state = None;
            let px = hp * wp;
            let gpu = &self.gpu;
            self.state = Some(State {
                hp,
                wp,
                calls: 0,
                in1: DevBuf::new(gpu, 6 * px * C0 * 2)?,
                in2: DevBuf::new(gpu, 4 * px * C1 * 2)?,
                out1: DevBuf::new(gpu, 2 * px * C1 * 2)?,
                out2: DevBuf::new(gpu, px * C2 * 2)?,
                col: DevBuf::new(gpu, COL_BYTES)?,
                frames: DevBuf::new(gpu, 4 * 16 * hp * 16 * wp * 3 * 2)?,
            });
        }
        self.state.as_mut().unwrap().calls = 0;
        Ok(())
    }

    /// One conv3d (kernel 4 x 3 x 3, stride 2 in time) over `inp` [t, hp, wp, ci] into out [to, hp, wp, co], biased
    #[allow(clippy::too_many_arguments)]
    fn conv(nsd: &Nsd, gpu: &Gpu, lin: &Lin, inp: &DevBuf, t: usize, hp: usize, wp: usize, ci: usize, col: &DevBuf, out: &DevBuf) -> Result<()> {
        let k = ffi::api()?;
        let to = (t - 4) / 2 + 1;
        let px = to * hp * wp;
        let kk = 36 * ci;
        let tile = (COL_BYTES / (kk * 2)).max(1);
        let mut p0 = 0;
        while p0 < px {
            let n = tile.min(px - p0);
            // SAFETY: inp holds t x hp x wp x ci halves, col n x kk, out px x co.
            check(unsafe { (k.im2col3)(g(gpu), inp.ptr(), t as i64, hp as i64, wp as i64, ci as i64, 4, 2, p0 as i64, n as i64, col.ptr()) },
                  "lq unfold")?;
            let o = unsafe { (out.ptr() as *mut u8).add(p0 * lin.n * 2) };
            lin.apply(nsd, col.ptr(), n, o as *mut _, Dt::F16)?;
            p0 += n;
        }
        Ok(())
    }

    /// The frames (half [n, 16 hp, 16 wp, 3] in [-1, 1] on this card, n = 1 for the first call, else 4): the latent
    /// frame's tokens into `out` (float32 [hp wp, 1536], natural order), or false for the first call
    pub fn stream(&mut self, nsd: &Nsd, frames: &DevBuf, n: usize, out: &DevBuf) -> Result<bool> {
        let k = ffi::api()?;
        let gp = g(&self.gpu);
        let s = self.state.as_mut().ok_or_else(|| Error("flashvsr: the LQ projection was not reset".into()))?;
        let (hp, wp) = (s.hp, s.wp);
        let px = hp * wp;
        let fr = px * C0 * 2;
        let first = s.calls == 0;
        if first != (n == 1) {
            return Err(Error(format!("flashvsr: the LQ projection's call {} took {n} frames", s.calls)));
        }
        // the call's 4 frames unshuffled into in1's last 4 places
        let (h, w) = (16 * hp, 16 * wp);
        let one = h * w * 3 * 2;
        if first {
            for i in 0..4 {
                s.frames.copy_within(i * one, frames, 0, one)?;
            }
        } else {
            s.frames.copy_within(0, frames, 0, 4 * one)?;
        }
        // SAFETY: frames holds 4 frames, in1 6 x px x 768 halves.
        let at2 = unsafe { (s.in1.ptr() as *mut u8).add(2 * fr) };
        check(unsafe { (k.unshuffle)(gp, s.frames.ptr(), 4, h as i64, w as i64, 1, 16, at2 as *mut _) }, "lq unshuffle")?;
        if first {
            // the front padding repeats the first frame
            s.in1.copy_within(0, &s.in1, 2 * fr, fr)?;
            s.in1.copy_within(fr, &s.in1, 2 * fr, fr)?;
        }
        Self::conv(nsd, &self.gpu, &self.conv1, &s.in1, 6, hp, wp, C0, &s.col, &s.out1)?;
        check(unsafe { (k.rms_silu)(gp, s.out1.ptr(), (2 * px) as i64, C1 as i64, self.g1.fp(), s.out1.ptr()) }, "lq norm1")?;
        // the memory for the next call: conv1's last two input frames, conv2's input's last two (conv1's output)
        s.in1.copy_within(0, &s.in1, 4 * fr, 2 * fr)?;
        let f2 = px * C1 * 2;
        s.calls += 1;
        if first {
            s.in2.copy_within(0, &s.out1, 0, 2 * f2)?;
            return Ok(false);
        }
        s.in2.copy_within(2 * f2, &s.out1, 0, 2 * f2)?;
        Self::conv(nsd, &self.gpu, &self.conv2, &s.in2, 4, hp, wp, C1, &s.col, &s.out2)?;
        check(unsafe { (k.rms_silu)(gp, s.out2.ptr(), px as i64, C2 as i64, self.g2.fp(), s.out2.ptr()) }, "lq norm2")?;
        s.in2.copy_within(0, &s.in2, 2 * f2, 2 * f2)?;
        self.lin.apply(nsd, s.out2.ptr(), px, out.ptr(), Dt::F32)?;
        Ok(true)
    }
}
