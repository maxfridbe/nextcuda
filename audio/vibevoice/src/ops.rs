//! The engine's kernels (ffi.rs) as checked calls on device pointers, its matrices (half, or int8 with a scale a row)
//! and the checkpoint's tensors: the GGUF's (`gguf-org/vibevoice-gguf`: the model's own state-dict names, BF16 and
//! F32, the language model's matrices Q8_0 in the smaller file), read on the host and converted on the GPU.
//!
//! The pointers are device memory the host never dereferences: the kernels read them on the GPU, so the calls are not
//! `unsafe` fns; what they rest on is the sizes the caller names.
#![allow(clippy::not_unsafe_ptr_arg_deref, clippy::too_many_arguments)]

use std::ffi::c_void;
use std::path::Path;
use std::sync::Arc;

use nextcuda_audio::{Error, Result};
use nextcuda_core::{DevBuf, Gpu};
use nextcuda_diffusion::kernels::{none, Dt, Nsd};
use nextcuda_gguf::{GType, Gguf, Tensor};

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

/// The checkpoint: one GGUF of the model's own tensor names
pub struct Weights {
    pub g: Gguf,
}

/// IEEE half bits to float
pub fn f16_to_f32(h: u16) -> f32 {
    let (s, e, m) = ((h >> 15) as u32, ((h >> 10) & 0x1f) as u32, (h & 0x3ff) as u32);
    let bits = if e == 0 {
        if m == 0 {
            s << 31
        } else {
            // subnormal: normalise
            let mut e = 127 - 15 + 1;
            let mut m = m;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            (s << 31) | (e << 23) | ((m & 0x3ff) << 13)
        }
    } else if e == 31 {
        (s << 31) | (0xff << 23) | (m << 13)
    } else {
        (s << 31) | ((e + 127 - 15) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

/// A tensor's bytes (a slice of its elements: [e0, e1)) as float32, for the types this checkpoint has
fn to_f32(t: &Tensor, b: &[u8]) -> Result<Vec<f32>> {
    Ok(match t.ty {
        GType::F32 => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        GType::BF16 => b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
        GType::F16 => b.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
        // blocks of 32: an f16 scale, 32 int8
        GType::Q8_0 => b.chunks_exact(34).flat_map(|blk| {
            let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
            blk[2..].iter().map(move |q| *q as i8 as f32 * d)
        }).collect(),
        ty => return Err(Error(format!("{}: type {}, not one this engine reads (F32, BF16, F16, Q8_0)", t.name, ty.name()))),
    })
}

impl Weights {
    pub fn open(path: &Path) -> Result<Weights> {
        Ok(Weights { g: Gguf::open(path).map_err(|e| Error(e.0))? })
    }

    pub fn find(&self, name: &str) -> Result<&Tensor> {
        self.g.tensor(name).ok_or_else(|| Error(format!("{}: no tensor {name}", self.g.paths[0].display())))
    }

    pub fn has(&self, name: &str) -> bool {
        self.g.tensor(name).is_some()
    }

    pub fn shape(&self, name: &str) -> Result<Vec<usize>> {
        Ok(self.find(name)?.shape.iter().map(|d| *d as usize).collect())
    }

    /// A whole tensor as float32 on the host
    pub fn f32(&self, name: &str) -> Result<Vec<f32>> {
        let t = self.find(name)?;
        to_f32(t, &self.g.read(t).map_err(|e| Error(e.0))?)
    }

    /// Rows [r0, r1) of a tensor [R, ...] as float32 on the host (a quantized tensor's rows are whole blocks)
    pub fn rows_f32(&self, name: &str, r0: usize, r1: usize) -> Result<Vec<f32>> {
        let t = self.find(name)?;
        let row: u64 = t.shape.iter().skip(1).product();
        let rb = t.ty.bytes(row).ok_or_else(|| Error(format!("{name}: a row of {row} does not divide into its blocks")))?;
        let mut b = vec![0u8; ((r1 - r0) as u64 * rb) as usize];
        self.g.read_into(t, r0 as u64 * rb, &mut b).map_err(|e| Error(e.0))?;
        to_f32(t, &b)
    }

    /// A tensor as float32 on the GPU
    pub fn dev_f32(&self, ops: &Ops, name: &str) -> Result<DevBuf> {
        DevBuf::from_f32(&ops.gpu, &self.f32(name)?).map_err(|e| Error(e.0))
    }

    /// The scalar a 0-d tensor holds
    pub fn scalar(&self, name: &str) -> Result<f32> {
        self.f32(name)?.first().copied().ok_or_else(|| Error(format!("{name}: empty")))
    }

    /// A matrix from row ranges of tensors [N, K], stacked: [(name, r0, r1)]; half, or int8 rows
    pub fn mat(&self, ops: &Ops, parts: &[(&str, usize, usize)], int8: bool) -> Result<Mat> {
        let (mut k, mut n) = (0, 0);
        for (name, r0, r1) in parts {
            let s = self.shape(name)?;
            let kk: usize = s.iter().skip(1).product();
            if k != 0 && kk != k {
                return Err(Error(format!("{name}: {kk} wide, the matrix {k}")));
            }
            if *r1 > s[0] || r0 >= r1 {
                return Err(Error(format!("{name}: rows {r0}..{r1} of {}", s[0])));
            }
            k = kk;
            n += r1 - r0;
        }
        let gpu = &ops.gpu;
        let w = DevBuf::new(gpu, n * k * if int8 { 1 } else { 2 }).map_err(|e| Error(e.0))?;
        let scale = if int8 { Some(DevBuf::f32(gpu, n).map_err(|e| Error(e.0))?) } else { None };
        let mut row = 0;
        for (name, r0, r1) in parts {
            let v = self.rows_f32(name, *r0, *r1)?;
            let f = DevBuf::from_f32(gpu, &v).map_err(|e| Error(e.0))?;
            match &scale {
                None => ops.to_half(f.fp(), w.ptr().wrapping_byte_add(row * k * 2), v.len())?,
                Some(s) => ops.quant_rows(f.fp(), r1 - r0, k, w.ptr().wrapping_byte_add(row * k), fp(s, row))?,
            }
            gpu.sync().map_err(|e| Error(e.0))?;
            row += r1 - r0;
        }
        Ok(Mat { w, scale, n, k })
    }

    /// A whole tensor as a matrix
    pub fn mat1(&self, ops: &Ops, name: &str, int8: bool) -> Result<Mat> {
        let r = self.shape(name)?[0];
        self.mat(ops, &[(name, 0, r)], int8)
    }

    /// Floats laid out side by side from several tensors (biases of a fused matrix)
    pub fn cat_f32(&self, ops: &Ops, names: &[&str]) -> Result<DevBuf> {
        let mut v = Vec::new();
        for n in names {
            v.extend(self.f32(n)?);
        }
        DevBuf::from_f32(&ops.gpu, &v).map_err(|e| Error(e.0))
    }
}

/// A half staging buffer for products of up to `values` x-values, with the int8 matrices' expansion when any is
pub fn stage(gpu: &Arc<Gpu>, values: usize, int8_biggest: Option<usize>) -> Result<Stage> {
    Ok(Stage {
        xh: DevBuf::new(gpu, values.max(1) * 2).map_err(|e| Error(e.0))?,
        wh: int8_biggest.map(|n| DevBuf::new(gpu, n * 2)).transpose().map_err(|e| Error(e.0))?,
    })
}
