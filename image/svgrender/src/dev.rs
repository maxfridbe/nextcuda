//! A scene on the card: its shapes, its parameters in one buffer (points, fills, strokes, half-widths - Adam steps
//! them as one array), their gradients and Adam's moments, sized for the most paths a run adds so a layer added is a
//! write, never a reallocation.

use std::ffi::c_void;
use std::sync::Arc;

use nextcuda_core::{DevBuf, Error, Gpu, Result};

use crate::ffi::{self, check};
use crate::scene::{npoints, PathShape, Scene, CLOSED};

/// Parameter groups (svg.h's ns_image_svg_adam)
pub const G_POINT: u8 = 0;
pub const G_FILL: u8 = 1;
pub const G_STROKE: u8 = 2;
pub const G_WIDTH: u8 = 3;
pub const G_FROZEN: u8 = 255;

pub struct DevScene {
    pub gpu: Arc<Gpu>,
    pub width: u32,
    pub height: u32,
    pub cap_paths: usize,
    pub cap_points: usize,
    /// the paths drawn now (the scene's first n)
    pub n: usize,
    /// points used so far
    pub n_points: usize,
    pub shapes: Vec<PathShape>,
    paths: DevBuf,
    pub params: DevBuf,
    pub grads: DevBuf,
    m: DevBuf,
    v: DevBuf,
    grp: DevBuf,
    own: DevBuf,
    birth: DevBuf,
    box_: DevBuf,
    bg: DevBuf,
    /// [0] the image loss, [1] the xing loss (summed by the kernels; zeroed each step)
    pub loss: DevBuf,
    grp_host: Vec<u8>,
    own_host: Vec<i32>,
    birth_host: Vec<i32>,
}

/// A float buffer's address `at` floats in
fn fat(b: &DevBuf, at: usize) -> *mut f32 {
    // SAFETY: callers stay inside the buffer (offsets computed from the capacities it was made with).
    unsafe { b.fp().add(at) }
}

fn bytes<T>(v: &[T]) -> &[u8] {
    // SAFETY: plain old data viewed as its bytes.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

impl DevScene {
    pub fn new(gpu: &Arc<Gpu>, width: u32, height: u32, cap_paths: usize, cap_points: usize) -> Result<DevScene> {
        let np = Self::param_len(cap_paths, cap_points);
        let z = |n: usize| -> Result<DevBuf> {
            let b = DevBuf::new(gpu, n.max(4))?;
            b.fill(0)?;
            Ok(b)
        };
        let s = DevScene {
            gpu: gpu.clone(),
            width,
            height,
            cap_paths,
            cap_points,
            n: 0,
            n_points: 0,
            shapes: Vec::new(),
            paths: z(16 * cap_paths)?,
            params: z(4 * np)?,
            grads: z(4 * np)?,
            m: z(4 * np)?,
            v: z(4 * np)?,
            grp: z(np)?,
            own: z(4 * np)?,
            birth: z(4 * cap_paths)?,
            box_: z(16 * cap_paths)?,
            bg: z(12)?,
            loss: z(8)?,
            grp_host: vec![G_FROZEN; np],
            own_host: vec![0; np],
            birth_host: vec![i32::MAX; cap_paths],
        };
        s.set_bg([1.0, 1.0, 1.0])?;
        Ok(s)
    }

    fn param_len(cap_paths: usize, cap_points: usize) -> usize {
        2 * cap_points + 9 * cap_paths
    }
    fn off_fill(&self) -> usize {
        2 * self.cap_points
    }
    fn off_stroke(&self) -> usize {
        2 * self.cap_points + 4 * self.cap_paths
    }
    fn off_hw(&self) -> usize {
        2 * self.cap_points + 8 * self.cap_paths
    }
    pub fn n_params(&self) -> usize {
        Self::param_len(self.cap_paths, self.cap_points)
    }

    pub fn set_bg(&self, c: [f32; 3]) -> Result<()> {
        self.bg.write(0, bytes(&c))
    }

    /// Adds `s`'s paths [from, len) after the ones here, each to be stepped from `step` on with the groups `opt`
    /// says (points, fill, stroke colour, half-width)
    pub fn add(&mut self, s: &Scene, from: usize, step: i32, opt: [bool; 4]) -> Result<()> {
        let fo = self.off_fill();
        let so = self.off_stroke();
        let ho = self.off_hw();
        for i in from..s.len() {
            let mut p = s.paths[i];
            let n = npoints(p.nseg, p.flags & CLOSED != 0);
            if self.n >= self.cap_paths || self.n_points + n > self.cap_points {
                return Err(Error(format!("svg scene: room for {} paths and {} points", self.cap_paths, self.cap_points)));
            }
            let k = self.n;
            let pts = s.path_points(i);
            p.off = self.n_points as i32;
            self.params.write(4 * 2 * self.n_points, bytes(pts))?;
            self.params.write(4 * (fo + 4 * k), bytes(&s.fill[i]))?;
            self.params.write(4 * (so + 4 * k), bytes(&s.stroke[i]))?;
            self.params.write(4 * (ho + k), bytes(&[s.hw[i]]))?;
            self.paths.write(16 * k, bytes(&[p]))?;
            let mut mark = |at: usize, len: usize, g: u8, on: bool| {
                for j in at..at + len {
                    self.grp_host[j] = if on { g } else { G_FROZEN };
                    self.own_host[j] = k as i32;
                }
            };
            mark(2 * self.n_points, 2 * n, G_POINT, opt[0]);
            mark(fo + 4 * k, 4, G_FILL, opt[1]);
            mark(so + 4 * k, 4, G_STROKE, opt[2]);
            mark(ho + k, 1, G_WIDTH, opt[3]);
            self.birth_host[k] = step;
            self.shapes.push(p);
            self.n += 1;
            self.n_points += n;
        }
        self.grp.write(0, &self.grp_host)?;
        self.own.write(0, bytes(&self.own_host))?;
        self.birth.write(0, bytes(&self.birth_host))?;
        Ok(())
    }

    /// Paths [from, n)'s stroke colours keep their RGB (a sketch's black), only their alpha moves
    pub fn freeze_stroke_rgb(&mut self, from: usize) -> Result<()> {
        let so = self.off_stroke();
        for k in from..self.n {
            for c in 0..3 {
                self.grp_host[so + 4 * k + c] = G_FROZEN;
            }
        }
        self.grp.write(0, &self.grp_host)
    }

    /// The scene as it stands (the parameters read back)
    pub fn read(&self) -> Result<Scene> {
        let all = self.params.to_f32()?;
        let mut s = Scene::new(self.width, self.height);
        s.paths = self.shapes.clone();
        s.points = all[..2 * self.n_points].to_vec();
        let (fo, so, ho) = (self.off_fill(), self.off_stroke(), self.off_hw());
        for k in 0..self.n {
            s.fill.push([all[fo + 4 * k], all[fo + 4 * k + 1], all[fo + 4 * k + 2], all[fo + 4 * k + 3]]);
            s.stroke.push([all[so + 4 * k], all[so + 4 * k + 1], all[so + 4 * k + 2], all[so + 4 * k + 3]]);
            s.hw.push(all[ho + k]);
        }
        Ok(s)
    }

    /// The gradients as a scene's layout: (points, fill, stroke, half-widths)
    pub fn read_grads(&self) -> Result<(Vec<f32>, Vec<[f32; 4]>, Vec<[f32; 4]>, Vec<f32>)> {
        let all = self.grads.to_f32()?;
        let (fo, so, ho) = (self.off_fill(), self.off_stroke(), self.off_hw());
        let q = |o: usize| (0..self.n).map(|k| [all[o + 4 * k], all[o + 4 * k + 1], all[o + 4 * k + 2], all[o + 4 * k + 3]]).collect::<Vec<_>>();
        Ok((all[..2 * self.n_points].to_vec(), q(fo), q(so), all[ho..ho + self.n].to_vec()))
    }

    pub fn zero_grads(&self) -> Result<()> {
        self.grads.fill(0)
    }

    fn ptrs(&self) -> (*const c_void, *const f32, *const f32, *const f32, *const f32) {
        (self.paths.ptr().cast_const(), fat(&self.params, 0), fat(&self.params, self.off_fill()), fat(&self.params, self.off_stroke()),
         fat(&self.params, self.off_hw()))
    }

    pub fn bounds(&self) -> Result<()> {
        let a = ffi::api()?;
        let (paths, pts, _, _, hw) = self.ptrs();
        // SAFETY: the buffers hold n paths' records, points, half-widths and boxes.
        check(unsafe { (a.bounds)(ffi::g(&self.gpu), self.n as i32, paths, pts, hw, self.box_.fp()) }, "svg bounds")
    }

    /// The pixels [H, W, 4] into `out` (svg.h's ns_image_svg_render); bounds() first
    pub fn render(&self, out: &DevBuf, samples: i32, bg: bool, first: usize, mono: bool) -> Result<()> {
        let a = ffi::api()?;
        let (paths, pts, fill, stroke, hw) = self.ptrs();
        out.bounds(0, 16 * (self.width * self.height) as usize)?;
        let bgp = if bg { self.bg.fp().cast_const() } else { std::ptr::null() };
        // SAFETY: as bounds(); out holds W H 4 floats (checked).
        check(unsafe {
            (a.render)(ffi::g(&self.gpu), self.width as i32, self.height as i32, self.n as i32, paths, pts, fill, stroke, hw, self.box_.fp(), bgp, samples,
                       first as i32, mono as i32, out.fp())
        }, "svg render")
    }

    /// One backward pass (svg.h's ns_image_svg_grad): mode 0 against `target` [H, W, 3] with `weight` [H, W], mode 1
    /// with `d_img` [H, W, 4]; the gradients added to `grads`, the loss to loss[0]; the pixels into `out`
    pub fn grad(&self, mode: i32, target: Option<&DevBuf>, weight: Option<&DevBuf>, d_img: Option<&DevBuf>, out: Option<&DevBuf>) -> Result<()> {
        let a = ffi::api()?;
        let (paths, pts, fill, stroke, hw) = self.ptrs();
        let px = (self.width * self.height) as usize;
        let p = |b: Option<&DevBuf>, n: usize| -> Result<*const f32> {
            match b {
                Some(b) => {
                    b.bounds(0, 4 * n)?;
                    Ok(b.fp().cast_const())
                }
                None => Ok(std::ptr::null()),
            }
        };
        let (t, w, d) = (p(target, 3 * px)?, p(weight, px)?, p(d_img, 4 * px)?);
        let o = p(out, 4 * px)?.cast_mut();
        let (fo, so, ho) = (self.off_fill(), self.off_stroke(), self.off_hw());
        // SAFETY: as bounds(); the images' sizes checked; the gradients share the parameters' layout.
        check(unsafe {
            (a.grad)(ffi::g(&self.gpu), self.width as i32, self.height as i32, self.n as i32, paths, pts, fill, stroke, hw, self.box_.fp(),
                     self.bg.fp(), mode, t, w, d, fat(&self.grads, 0), fat(&self.grads, fo), fat(&self.grads, so), fat(&self.grads, ho),
                     self.loss.fp(), o)
        }, "svg grad")
    }

    /// LIVE's xing loss over paths [first, n), its weight times 1e-3 (xing_loss_fn's scale), into loss[1]
    pub fn xing(&self, first: usize, weight: f32) -> Result<()> {
        let a = ffi::api()?;
        let (paths, pts, _, _, _) = self.ptrs();
        // SAFETY: as bounds().
        check(unsafe { (a.xing)(ffi::g(&self.gpu), self.n as i32, first as i32, paths, pts, 1e-3 * weight, self.grads.fp(), fat(&self.loss, 1)) },
              "svg xing")
    }

    /// One Adam step of every live parameter (LIVE's per-layer optimisers: each path's step counted from its birth)
    #[allow(clippy::too_many_arguments)]
    pub fn adam(&self, step: i32, lr: [f32; 4], decay_every: i32, ratio: f32, betas: (f32, f32), eps: f32, lo: [f32; 4], hi: [f32; 4]) -> Result<()> {
        let a = ffi::api()?;
        // SAFETY: every buffer has n_params() entries; lr, lo, hi are host arrays of 4 (svg.h).
        check(unsafe {
            (a.adam)(ffi::g(&self.gpu), self.n_params() as i64, self.params.fp(), self.grads.fp(), self.m.fp(), self.v.fp(), self.grp.ptr().cast_const().cast(),
                     self.own.ptr().cast_const().cast(), self.birth.ptr().cast_const().cast(), step, lr.as_ptr(), decay_every, ratio, betas.0, betas.1, eps,
                     lo.as_ptr(), hi.as_ptr())
        }, "svg adam")
    }

    /// LIVE's distance weight from `cov` [H, W, 4] (a mono render: its alpha) and `keep`, into `weight`; `dist` [H, W]
    /// and `scratch` (16 bytes) its work
    pub fn udf(&self, cov: &DevBuf, keep: Option<&DevBuf>, weight: &DevBuf, dist: &DevBuf, scratch: &DevBuf) -> Result<()> {
        let a = ffi::api()?;
        let px = 4 * (self.width * self.height) as usize;
        cov.bounds(0, 4 * px)?;
        for b in [weight, dist] {
            b.bounds(0, px)?;
        }
        let k = keep.map(|k| k.fp().cast_const()).unwrap_or(std::ptr::null());
        // SAFETY: each image holds W H floats (checked), scratch 4 ints.
        check(unsafe { (a.udf)(ffi::g(&self.gpu), self.width as i32, self.height as i32, cov.fp(), k, weight.fp(), dist.fp(), scratch.ptr().cast()) },
              "svg udf")
    }

    /// The pixels [H, W, 4] as RGB bytes (composited: alpha ignored)
    pub fn to_rgb8(&self, px: &DevBuf, staging: &DevBuf) -> Result<Vec<u8>> {
        let a = ffi::api()?;
        let n = (self.width * self.height) as usize;
        staging.bounds(0, 3 * n)?;
        // SAFETY: px holds W H 4 floats, staging 3 W H bytes (checked).
        check(unsafe { (a.to_u8)(ffi::g(&self.gpu), self.width as i32, self.height as i32, px.fp(), staging.ptr().cast()) }, "svg to_u8")?;
        let mut v = vec![0u8; 3 * n];
        staging.read(0, &mut v)?;
        Ok(v)
    }
}
