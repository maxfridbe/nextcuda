//! The engine's kernels (ffi.rs) as checked calls on device pointers and its matrices (half, or int8 with a scale a
//! row) - the VibeVoice engine's ops (audio/vibevoice/src/ops.rs) as they are: Haru's codec is VibeVoice's acoustic
//! tokenizer and its MeanFlow head the same adaptive-norm MLP shape as VibeVoice's diffusion head, so they run on the
//! same calls. The checkpoint's tensors are read in weights.rs.
//!
//! The pointers are device memory the host never dereferences: the kernels read them on the GPU, so the calls are not
//! `unsafe` fns; what they rest on is the sizes the caller names.
#![allow(clippy::not_unsafe_ptr_arg_deref, clippy::too_many_arguments)]

use std::ffi::c_void;
use std::sync::Arc;

use nextcuda_audio::{Error, Result};
use nextcuda_core::{DevBuf, Gpu};
use nextcuda_diffusion::kernels::{none, Dt, Nsd};

use crate::ffi;

type P = *const c_void;
type M = *mut c_void;

/// A float pointer `at` floats into a buffer
pub fn fp(b: &DevBuf, at: usize) -> *mut f32 {
    b.fp().wrapping_add(at)
}

/// A null pointer (an absent bias)
pub fn nul() -> P {
    none()
}

/// The engine's kernels on one GPU
pub struct Ops {
    pub gpu: Arc<Gpu>,
    k: &'static ffi::Api,
}

impl Ops {
    pub fn new(gpu: &Arc<Gpu>) -> Result<Ops> {
        Ok(Ops { gpu: gpu.clone(), k: ffi::api()? })
    }

    fn g(&self) -> M {
        self.gpu.raw()
    }

    /// out[m] (+)= x[m] . W^T (* scale + bias), m <= 8, K a multiple of 256, out rows `ldo` apart
    pub fn gemv(&self, x: *const f32, m: usize, w: &Mat, bias: P, out: *mut f32, ldo: usize, acc: bool) -> Result<()> {
        let (wt, sc) = match &w.scale {
            None => (0, std::ptr::null()),
            Some(s) => (1, s.fp() as *const f32),
        };
        // SAFETY: the caller's buffers hold the sizes named.
        ffi::check(unsafe { (self.k.gemv)(self.g(), x, m as i64, w.k as i64, w.w.ptr(), wt, sc, w.n as i64, bias.cast(), out, ldo as i64, acc as i32) }, "gemv")
    }

    pub fn to_half(&self, x: *const f32, out: M, n: usize) -> Result<()> {
        // SAFETY: n floats in, n halfs out.
        ffi::check(unsafe { (self.k.to_half)(self.g(), x, out, n as i64) }, "to half")
    }

    pub fn quant_rows(&self, w: *const f32, n: usize, k: usize, q: M, scale: *mut f32) -> Result<()> {
        // SAFETY: w [n, k] floats, q [n, k] bytes, scale n floats.
        ffi::check(unsafe { (self.k.quant_rows)(self.g(), w, n as i64, k as i64, q.cast(), scale) }, "int8 rows")
    }

    pub fn dequant_rows(&self, q: P, scale: *const f32, n: usize, k: usize, out: M) -> Result<()> {
        // SAFETY: q [n, k] bytes, scale n floats, out [n, k] halfs.
        ffi::check(unsafe { (self.k.dequant_rows)(self.g(), q.cast(), scale, n as i64, k as i64, out) }, "int8 rows to half")
    }

    pub fn add(&self, x: *mut f32, y: *const f32, n: usize) -> Result<()> {
        // SAFETY: n floats each.
        ffi::check(unsafe { (self.k.add)(self.g(), x, y, n as i64) }, "add")
    }

    /// out [c, r] = x [r, c]
    pub fn transpose(&self, x: *const f32, r: usize, c: usize, out: *mut f32) -> Result<()> {
        // SAFETY: r x c floats each.
        ffi::check(unsafe { (self.k.transpose)(self.g(), x, r as i64, c as i64, out) }, "transpose")
    }

    /// A 1-D convolution of x [ci, l] (no padding): out [co, lo], lo = (l - k) / stride + 1; w [co, ci, k]
    pub fn conv(&self, x: *const f32, ci: usize, l: usize, w: &DevBuf, co: usize, k: usize, bias: *const f32, stride: usize, out: *mut f32) -> Result<usize> {
        let lo = (l - k) / stride + 1;
        // SAFETY: the caller's buffers hold the sizes named.
        ffi::check(unsafe { (self.k.conv1d_lr)(self.g(), 0, x, 1, ci as i64, l as i64, w.fp(), co as i64, k as i64, bias, stride as i64, 1, 0, 0, out,
                                               lo as i64) }, "conv1d")?;
        Ok(lo)
    }

    /// A transposed one over x [ci, l], w [ci, co, k]: its full output with `cut_l` samples cut ahead and `cut_r`
    /// behind, out [co, lo] (lo = (l - 1) stride + k - cut_l - cut_r)
    pub fn conv_t(&self, x: *const f32, ci: usize, l: usize, w: &DevBuf, co: usize, k: usize, bias: *const f32, stride: usize, cut_l: usize, cut_r: usize,
                  out: *mut f32) -> Result<usize> {
        let lo = (l - 1) * stride + k - cut_l - cut_r;
        // SAFETY: as conv.
        ffi::check(unsafe { (self.k.conv1d_lr)(self.g(), 1, x, 1, ci as i64, l as i64, w.fp(), co as i64, k as i64, bias, stride as i64, 1, cut_l as i64,
                                               cut_r as i64, out, lo as i64) }, "conv_transpose1d")?;
        Ok(lo)
    }

    /// The causal depthwise convolution of x [c, l] (k - 1 zeros ahead): w [c, k], bias [c]
    pub fn dwconv(&self, x: *const f32, c: usize, l: usize, w: &DevBuf, bias: *const f32, k: usize, out: *mut f32) -> Result<()> {
        // SAFETY: x and out [c, l], w [c, k], bias c floats or null.
        ffi::check(unsafe { (self.k.dwconv)(self.g(), x, c as i64, l as i64, w.fp(), bias, k as i64, out) }, "depthwise conv")
    }

    /// Rotate-half rotary positions on l rows of h heads x d (xs floats apart), row r at position p0 + r
    pub fn rope(&self, x: *mut f32, xs: usize, l: usize, h: usize, d: usize, theta: f32, p0: usize) -> Result<()> {
        // SAFETY: l rows of h x d floats xs apart.
        ffi::check(unsafe { (self.k.rope)(self.g(), x, xs as i64, l as i64, h as i64, d as i64, theta, p0 as i64) }, "rope")
    }

    /// x [r, c] += scale[c] * y [r, c]
    pub fn scale_add(&self, x: *mut f32, y: *const f32, scale: &DevBuf, r: usize, c: usize) -> Result<()> {
        // SAFETY: x and y r x c floats, scale c.
        ffi::check(unsafe { (self.k.scale_add)(self.g(), x, y, scale.fp(), r as i64, c as i64) }, "scaled add")
    }

    pub fn silu(&self, x: *mut f32, n: usize) -> Result<()> {
        // SAFETY: n floats.
        ffi::check(unsafe { (self.k.silu)(self.g(), x, n as i64) }, "silu")
    }
}

/// A matrix [n, k] on the GPU: half, or int8 with a scale per row
pub struct Mat {
    pub w: DevBuf,
    pub scale: Option<DevBuf>,
    pub n: usize,
    pub k: usize,
}

/// Half staging for wide products: x as half (and an int8 matrix expanded), sized for the widest a model needs
pub struct Stage {
    pub xh: DevBuf,
    pub wh: Option<DevBuf>,
}

impl Mat {
    /// out [m, n] f32 = x [m, k] f32 . W^T + bias: a few rows (K a multiple of 256) by the engine's own product
    /// (bound by reading W once), the rest by the shared half GEMM through `st`
    pub fn apply(&self, ops: &Ops, nsd: &Nsd, x: *const f32, m: usize, st: &Stage, bias: P, out: *mut f32) -> Result<()> {
        if m <= 8 && self.k % 256 == 0 {
            return ops.gemv(x, m, self, bias, out, self.n, false);
        }
        if st.xh.len < m * self.k * 2 {
            return Err(Error(format!("a product of {m} x {}: the half staging holds {} values", self.k, st.xh.len / 2)));
        }
        ops.to_half(x, st.xh.ptr(), m * self.k)?;
        let w = match &self.scale {
            None => self.w.ptr(),
            Some(s) => {
                let wh = st.wh.as_ref().ok_or_else(|| Error("a half matrix buffer for a wide int8 product".into()))?;
                ops.dequant_rows(self.w.ptr(), s.fp(), self.n, self.k, wh.ptr())?;
                wh.ptr()
            }
        };
        nsd.linear(st.xh.ptr(), Dt::F16, m, self.k, w, self.n, bias, out.cast(), Dt::F32)
    }

    pub fn bytes(&self) -> usize {
        self.w.len + self.scale.as_ref().map_or(0, |s| s.len)
    }
}

/// A half staging buffer for products of up to `values` x-values, with the int8 matrices' expansion when any is
pub fn stage(gpu: &Arc<Gpu>, values: usize, int8_biggest: Option<usize>) -> Result<Stage> {
    Ok(Stage {
        xh: DevBuf::new(gpu, values.max(1) * 2).map_err(|e| Error(e.0))?,
        wh: int8_biggest.map(|n| DevBuf::new(gpu, n * 2)).transpose().map_err(|e| Error(e.0))?,
    })
}
