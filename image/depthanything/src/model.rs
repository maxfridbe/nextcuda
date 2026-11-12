//! Depth Anything V2 (transformers' DepthAnythingForDepthEstimation) in float32 on the GPU: DINOv2 with registers-free
//! ViT-S/14 as its backbone (384 wide, 12 layers of 6 heads; the hidden states after layers 3, 6, 9 and 12, each through
//! the backbone's final layer norm) and the DPT head.
//!
//! ```text
//!   x = [cls | patches 14 x 14 . W + b] + positions (the 37 x 37 table resized bicubically to the patches)
//!   per layer:  x += ls1 * attention(ln1(x)) Wo;  x += ls2 * fc2(gelu(fc1(ln2(x))))
//!   features i = ln(x after layer 3 i + 3), the class token dropped: [gh, gw, 384]
//!   reassemble: 1x1 to 48 / 96 / 192 / 384, then x4 / x2 (transposed convolutions, stride = kernel), as is, a 3x3
//!     stride-2 convolution; each a 3x3 convolution (no bias) to 64
//!   fusion, deepest first: f = RCU2(f3) -> up to f2's size -> 1x1;  f = RCU2(f + RCU1(f2)) -> up -> 1x1;  ... f0,
//!     the last up x 2   (RCU(x) = conv(relu(conv(relu(x)))) + x; up: bilinear, align_corners=True)
//!   head: 3x3 64 -> 32, up to the input's size (bilinear, align corners), 3x3 32 -> 32, ReLU, 1x1 -> 1, ReLU
//! ```
//!
//! Every map is channels last; the products are the diffusion kernels' float32 GEMMs (cuBLAS, no TF32), the
//! convolutions an im2col (the vision kernels') in slices then that GEMM.

use std::sync::Arc;

use std::sync::atomic::{AtomicUsize, Ordering};

use nextcuda_core::{Arena, DevBuf, Error, Gpu, Result};
use nextcuda_diffusion::kernels::{Dt, Nsd};
use nextcuda_gguf::safetensors::SafeTensors;

use crate::ffi::{self, check};
use crate::prep::{self, PATCH};

const EPS: f32 = 1e-6;
/// The im2col's slice, in floats (64 MiB)
const COLS: usize = 16 << 20;

type P = *const std::ffi::c_void;
/// The position table made for one patch grid (gh, gw)
type Positions = Option<((usize, usize), DevBuf)>;

/// A linear layer (a 1 x 1 convolution, a transposed one as a product): float32 [n, k], its bias [n] or none
pub struct Lin {
    w: DevBuf,
    b: Option<DevBuf>,
    pub n: usize,
    pub k: usize,
}

impl Lin {
    fn bias(&self) -> *const f32 {
        self.b.as_ref().map_or(std::ptr::null(), |b| b.fp())
    }
}

/// A k x k convolution: its weight as im2col rows take it ([co, k, k, ci]), stride, zero padding
pub struct Conv {
    lin: Lin,
    ci: usize,
    k: usize,
    s: usize,
    p: usize,
}

struct Layer {
    n1w: DevBuf,
    n1b: DevBuf,
    /// q | k | v rows
    qkv: Lin,
    o: Lin,
    ls1: DevBuf,
    n2w: DevBuf,
    n2b: DevBuf,
    fc1: Lin,
    fc2: Lin,
    ls2: DevBuf,
}

/// A pre-activation residual unit
struct Rcu {
    c1: Conv,
    c2: Conv,
}

struct Fusion {
    rcu1: Rcu,
    rcu2: Rcu,
    proj: Lin,
}

/// The reassembly's resize of one level: up by a transposed convolution (its product, its bias, the factor), none, or
/// down by a stride-2 convolution
enum Resize {
    Up(Lin, DevBuf, usize),
    Same,
    Down(Conv),
}

/// A feature map on the card: float32 [h, w, c]
pub struct Map {
    pub buf: DevBuf,
    pub h: usize,
    pub w: usize,
    pub c: usize,
}

pub struct Net {
    gpu: Arc<Gpu>,
    nsd: Nsd,
    k: &'static ffi::Api,
    patch: Lin,
    cls: DevBuf,
    /// the stored table [1 + 37 x 37, width], host side: resized per input size
    table: Vec<f32>,
    layers: Vec<Layer>,
    lnw: DevBuf,
    lnb: DevBuf,
    /// the layers whose outputs the head reads (0-based)
    taps: [usize; 4],
    width: usize,
    heads: usize,
    proj: Vec<Lin>,
    resize: Vec<Resize>,
    convs: Vec<Conv>,
    fusion: Vec<Fusion>,
    head1: Conv,
    head2: Conv,
    head3: Lin,
    /// zeros [rows]: gate_add's row index (one gate row, the layer scale)
    zeros: std::sync::Mutex<Option<DevBuf>>,
    /// the im2col's slice
    cols: DevBuf,
    /// per input size (gh, gw): the position table on the card
    pos: std::sync::Mutex<Positions>,
    /// a forward's temporaries, one allocation (grown to what a pass took when one spilled past it), and what this
    /// pass has asked of it
    arena: Arena,
    asked: AtomicUsize,
    pub bytes: usize,
}

struct Loader<'a> {
    st: &'a SafeTensors,
    gpu: &'a Arc<Gpu>,
    bytes: usize,
}

impl Loader<'_> {
    fn f32(&self, n: &str) -> Result<Vec<f32>> {
        self.st.f32(n).map_err(|e| Error(e.0))
    }
    fn shape(&self, n: &str) -> Result<Vec<usize>> {
        Ok(self.st.need(n).map_err(|e| Error(e.0))?.shape.iter().map(|&d| d as usize).collect())
    }
    fn up(&mut self, v: &[f32]) -> Result<DevBuf> {
        self.bytes += v.len() * 4;
        DevBuf::from_f32(self.gpu, v)
    }
    fn vec(&mut self, n: &str) -> Result<DevBuf> {
        let v = self.f32(n)?;
        self.up(&v)
    }
    fn lin_of(&mut self, w: &[f32], b: Option<Vec<f32>>, n: usize, k: usize) -> Result<Lin> {
        if w.len() != n * k {
            return Err(Error(format!("a weight of {} for {n} x {k}", w.len())));
        }
        let b = match b {
            Some(b) => Some(self.up(&b)?),
            None => None,
        };
        Ok(Lin { w: self.up(w)?, b, n, k })
    }
    /// `<p>.weight` [n, k, ...] (any trailing 1 x 1) and `<p>.bias` when there is one
    fn lin(&mut self, p: &str) -> Result<Lin> {
        let sh = self.shape(&format!("{p}.weight"))?;
        let (n, k) = (sh[0], sh[1..].iter().product::<usize>());
        let w = self.f32(&format!("{p}.weight"))?;
        let b = if self.st.tensor(&format!("{p}.bias")).is_some() { Some(self.f32(&format!("{p}.bias"))?) } else { None };
        self.lin_of(&w, b, n, k)
    }
    fn conv(&mut self, p: &str, s: usize, pad: usize) -> Result<Conv> {
        let sh = self.shape(&format!("{p}.weight"))?;
        let (co, ci, k) = (sh[0], sh[1], sh[2]);
        let w = prep::conv_weight(&self.f32(&format!("{p}.weight"))?, co, ci, k);
        let b = if self.st.tensor(&format!("{p}.bias")).is_some() { Some(self.f32(&format!("{p}.bias"))?) } else { None };
        Ok(Conv { lin: self.lin_of(&w, b, co, k * k * ci)?, ci, k, s, p: pad })
    }
    fn rcu(&mut self, p: &str) -> Result<Rcu> {
        Ok(Rcu { c1: self.conv(&format!("{p}.convolution1"), 1, 1)?, c2: self.conv(&format!("{p}.convolution2"), 1, 1)? })
    }
}

impl Net {
    /// The model from transformers' safetensors (Depth-Anything-V2-Small-hf: `backbone.`, `neck.`, `head.`)
    pub fn load(st: &SafeTensors, gpu: &Arc<Gpu>) -> Result<Net> {
        let mut l = Loader { st, gpu, bytes: 0 };
        let b = "backbone.";
        let psh = l.shape(&format!("{b}embeddings.patch_embeddings.projection.weight"))?;
        let width = psh[0];
        if psh[1..] != [3, PATCH, PATCH] {
            return Err(Error(format!("a patch embedding of {psh:?}: DINOv2's is [width, 3, 14, 14]")));
        }
        let pw = prep::conv_weight(&l.f32(&format!("{b}embeddings.patch_embeddings.projection.weight"))?, width, 3, PATCH);
        let pb = l.f32(&format!("{b}embeddings.patch_embeddings.projection.bias"))?;
        let patch = l.lin_of(&pw, Some(pb), width, 3 * PATCH * PATCH)?;
        let cls = l.vec(&format!("{b}embeddings.cls_token"))?;
        let table = l.f32(&format!("{b}embeddings.position_embeddings"))?;
        let mut layers = Vec::new();
        while st.tensor(&format!("{b}encoder.layer.{}.norm1.weight", layers.len())).is_some() {
            let p = format!("{b}encoder.layer.{}.", layers.len());
            let a = format!("{p}attention.attention.");
            let mut w = l.f32(&format!("{a}query.weight"))?;
            w.extend(l.f32(&format!("{a}key.weight"))?);
            w.extend(l.f32(&format!("{a}value.weight"))?);
            let mut bias = l.f32(&format!("{a}query.bias"))?;
            bias.extend(l.f32(&format!("{a}key.bias"))?);
            bias.extend(l.f32(&format!("{a}value.bias"))?);
            layers.push(Layer {
                n1w: l.vec(&format!("{p}norm1.weight"))?,
                n1b: l.vec(&format!("{p}norm1.bias"))?,
                qkv: l.lin_of(&w, Some(bias), 3 * width, width)?,
                o: l.lin(&format!("{p}attention.output.dense"))?,
                ls1: l.vec(&format!("{p}layer_scale1.lambda1"))?,
                n2w: l.vec(&format!("{p}norm2.weight"))?,
                n2b: l.vec(&format!("{p}norm2.bias"))?,
                fc1: l.lin(&format!("{p}mlp.fc1"))?,
                fc2: l.lin(&format!("{p}mlp.fc2"))?,
                ls2: l.vec(&format!("{p}layer_scale2.lambda1"))?,
            });
        }
        // the backbone's out_indices: Depth Anything V2's [3, 6, 9, 12] for the Small (12 layers); the larger ones
        // (Base / Large, not Apache-2.0) tap other layers and are not served here
        if layers.len() != 12 || width != 384 {
            return Err(Error(format!("{} layers of {width}: this engine runs Depth Anything V2 Small (12 of 384)", layers.len())));
        }
        let taps = [2, 5, 8, 11];
        let n = "neck.";
        let mut proj = Vec::new();
        let mut resize = Vec::new();
        let mut convs = Vec::new();
        for i in 0..4 {
            let r = format!("{n}reassemble_stage.layers.{i}.");
            proj.push(l.lin(&format!("{r}projection"))?);
            let rw = format!("{r}resize.weight");
            resize.push(match st.tensor(&rw).map(|t| t.shape.clone()) {
                None => Resize::Same,
                // ConvTranspose2d [ci, co, f, f] with stride f; a Conv2d [co, ci, 3, 3] with stride 2
                Some(sh) if i < 2 => {
                    let (ci, co, f) = (sh[0] as usize, sh[1] as usize, sh[2] as usize);
                    let w = prep::deconv_weight(&l.f32(&rw)?, ci, co, f);
                    let lin = l.lin_of(&w, None, co * f * f, ci)?;
                    let bias = l.vec(&format!("{r}resize.bias"))?;
                    Resize::Up(lin, bias, f)
                }
                Some(_) => Resize::Down(l.conv(&format!("{r}resize"), 2, 1)?),
            });
            convs.push(l.conv(&format!("{n}convs.{i}"), 1, 1)?);
        }
        let mut fusion = Vec::new();
        for i in 0..4 {
            let f = format!("{n}fusion_stage.layers.{i}.");
            fusion.push(Fusion { rcu1: l.rcu(&format!("{f}residual_layer1"))?, rcu2: l.rcu(&format!("{f}residual_layer2"))?, proj: l.lin(&format!("{f}projection"))? });
        }
        let head1 = l.conv("head.conv1", 1, 1)?;
        let head2 = l.conv("head.conv2", 1, 1)?;
        let head3 = l.lin("head.conv3")?;
        let lnw = l.vec(&format!("{b}layernorm.weight"))?;
        let lnb = l.vec(&format!("{b}layernorm.bias"))?;
        let cols = DevBuf::f32(gpu, COLS)?;
        let bytes = l.bytes + COLS * 4;
        Ok(Net {
            gpu: gpu.clone(),
            nsd: Nsd::new(gpu)?,
            k: ffi::api()?,
            patch,
            cls,
            table,
            layers,
            lnw,
            lnb,
            taps,
            width,
            heads: width / 64,
            proj,
            resize,
            convs,
            fusion,
            head1,
            head2,
            head3,
            zeros: std::sync::Mutex::new(None),
            cols,
            pos: std::sync::Mutex::new(None),
            arena: Arena::new(gpu, 256)?,
            asked: AtomicUsize::new(0),
            bytes,
        })
    }

    fn g(&self) -> *mut std::ffi::c_void {
        self.gpu.raw()
    }

    /// `n` floats of the pass's arena
    fn buf(&self, n: usize) -> Result<DevBuf> {
        self.asked.fetch_add((n * 4).next_multiple_of(256), Ordering::Relaxed);
        self.arena.f32(n)
    }

    fn map(&self, h: usize, w: usize, c: usize) -> Result<Map> {
        Ok(Map { buf: self.buf(h * w * c)?, h, w, c })
    }

    /// The arena's size now (bytes on the card beside the weights)
    pub fn arena_bytes(&self) -> usize {
        self.arena.len()
    }

    /// out [m, n] = x [m, k] . W^T + b, float32
    fn lin(&self, l: &Lin, x: P, m: usize, out: *mut std::ffi::c_void) -> Result<()> {
        self.nsd.linear(x, Dt::F32, m, l.k, l.w.ptr(), l.n, l.bias().cast(), out, Dt::F32)
    }

    fn lin_map(&self, l: &Lin, x: &Map) -> Result<Map> {
        let o = self.map(x.h, x.w, l.n)?;
        self.lin(l, x.buf.ptr(), x.h * x.w, o.buf.ptr())?;
        Ok(o)
    }

    /// A convolution of x: an im2col of the output rows a slice at a time, each slice's product
    fn conv(&self, c: &Conv, x: &Map) -> Result<Map> {
        if x.c != c.ci {
            return Err(Error(format!("a convolution of {} channels given {}", c.ci, x.c)));
        }
        let ho = (x.h + 2 * c.p - c.k) / c.s + 1;
        let wo = (x.w + 2 * c.p - c.k) / c.s + 1;
        let o = self.map(ho, wo, c.lin.n)?;
        let kk = c.lin.k;
        let per = (COLS / kk).max(1);
        let px = ho * wo;
        let mut r0 = 0;
        while r0 < px {
            let r = per.min(px - r0);
            // SAFETY: x holds [h, w, ci]; the slice r rows of kk floats (r * kk <= COLS).
            check(unsafe { (self.k.im2col)(self.g(), x.buf.fp(), x.h as i64, x.w as i64, x.c as i64, c.k as i64, c.k as i64, c.s as i64, c.p as i64, 1,
                                           ho as i64, wo as i64, r0 as i64, r as i64, self.cols.fp()) }, "im2col")?;
            self.lin(&c.lin, self.cols.ptr(), r, o.buf.fp().wrapping_add(r0 * c.lin.n).cast())?;
            r0 += r;
        }
        Ok(o)
    }

    fn relu(&self, x: &Map) -> Result<()> {
        // SAFETY: the whole map.
        check(unsafe { (self.k.relu)(self.g(), x.buf.fp(), (x.h * x.w * x.c) as i64) }, "relu")
    }

    /// x += y, same shapes
    fn add(&self, x: &Map, y: &Map) -> Result<()> {
        if (x.h, x.w, x.c) != (y.h, y.w, y.c) {
            return Err(Error(format!("adding [{}, {}, {}] to [{}, {}, {}]", y.h, y.w, y.c, x.h, x.w, x.c)));
        }
        // SAFETY: both maps the same size.
        check(unsafe { (self.k.add)(self.g(), x.buf.fp(), y.buf.fp(), (x.h * x.w * x.c) as i64) }, "add")
    }

    /// Bilinear with align_corners=True (DPT's fusion and head)
    fn up(&self, x: &Map, h: usize, w: usize) -> Result<Map> {
        let o = self.map(h, w, x.c)?;
        // SAFETY: x [h, w, c] in, out [ho, wo, c] (rows c apart: a whole map).
        check(unsafe { (self.k.resize_ac)(self.g(), x.buf.fp(), x.h as i64, x.w as i64, x.c as i64, h as i64, w as i64, o.buf.fp(), x.c as i64) }, "resize")?;
        Ok(o)
    }

    /// RCU(x) = conv2(relu(conv1(relu(x)))) + x
    fn rcu(&self, u: &Rcu, x: &Map) -> Result<Map> {
        let t = self.map(x.h, x.w, x.c)?;
        t.buf.copy_within(0, &x.buf, 0, x.buf.len)?;
        self.relu(&t)?;
        let a = self.conv(&u.c1, &t)?;
        self.relu(&a)?;
        let b = self.conv(&u.c2, &a)?;
        self.add(&b, x)?;
        Ok(b)
    }

    /// The position table for gh x gw patches on the card (made once a size)
    fn positions(&self, gh: usize, gw: usize) -> Result<std::sync::MutexGuard<'_, Positions>> {
        let mut g = self.pos.lock().unwrap_or_else(|p| p.into_inner());
        if g.as_ref().is_none_or(|(s, _)| *s != (gh, gw)) {
            let p = prep::positions(&self.table, self.width, gh, gw).map_err(Error)?;
            *g = Some(((gh, gw), DevBuf::from_f32(&self.gpu, &p)?));
        }
        Ok(g)
    }

    /// The relative inverse depth of `px` (float32 [h, w, 3], normalized; sides multiples of 14) at [h, w], float32 on
    /// the host
    pub fn forward(&self, px: &[f32], w: usize, h: usize) -> Result<Vec<f32>> {
        self.arena.reset();
        self.asked.store(0, Ordering::Relaxed);
        let spilled = self.arena.spills.swap(0, Ordering::Relaxed);
        debug_assert_eq!(spilled, 0);
        let v = self.pass(px, w, h)?;
        // a pass that did not fit in the arena: the next one will (the spilled buffers went with the pass)
        if self.arena.spills.swap(0, Ordering::Relaxed) > 0 {
            self.arena.set(DevBuf::new(&self.gpu, self.asked.load(Ordering::Relaxed))?);
        }
        Ok(v)
    }

    fn pass(&self, px: &[f32], w: usize, h: usize) -> Result<Vec<f32>> {
        if !w.is_multiple_of(PATCH) || !h.is_multiple_of(PATCH) || px.len() != w * h * 3 {
            return Err(Error(format!("depth: {} values for a {w} x {h} input (sides multiples of 14)", px.len())));
        }
        let (gh, gw) = (h / PATCH, w / PATCH);
        let (np, c) = (gh * gw, self.width);
        let s = np + 1;
        let gpu = &self.gpu;
        let input = self.buf(px.len())?;
        // SAFETY: plain floats viewed as their bytes.
        input.write(0, unsafe { std::slice::from_raw_parts(px.as_ptr().cast::<u8>(), px.len() * 4) })?;
        // the patches: an im2col (stride 14, no padding) and the projection, into rows 1.. behind the class token
        let x = self.buf(s * c)?;
        let pk = self.patch.k;
        let pcols = self.buf(np * pk)?;
        // SAFETY: input [h, w, 3]; pcols np rows of 3 * 14 * 14.
        check(unsafe { (self.k.im2col)(self.g(), input.fp(), h as i64, w as i64, 3, PATCH as i64, PATCH as i64, PATCH as i64, 0, 1, gh as i64, gw as i64, 0,
                                       np as i64, pcols.fp()) }, "patches")?;
        self.lin(&self.patch, pcols.ptr(), np, x.fp().wrapping_add(c).cast())?;
        x.copy_within(0, &self.cls, 0, c * 4)?;
        {
            let pos = self.positions(gh, gw)?;
            let (_, p) = pos.as_ref().expect("made");
            // SAFETY: both [s, c].
            check(unsafe { (self.k.add)(self.g(), x.fp(), p.fp(), (s * c) as i64) }, "positions")?;
        }
        let zeros = {
            let mut z = self.zeros.lock().unwrap_or_else(|p| p.into_inner());
            if z.as_ref().is_none_or(|b| b.len < s * 4) {
                let b = DevBuf::new(gpu, s * 4)?;
                b.fill(0)?;
                *z = Some(b);
            }
            z.as_ref().expect("made").ptr() as P
        };
        let ffn = self.layers[0].fc1.n;
        let hb = self.buf(s * c)?;
        let qkv = self.buf(s * 3 * c)?;
        let att = self.buf(s * c)?;
        let tmp = self.buf(s * c)?;
        let mid = self.buf(s * ffn)?;
        let mut feats = Vec::new();
        let d = c / self.heads;
        for (i, l) in self.layers.iter().enumerate() {
            self.nsd.layer_norm(x.ptr(), Dt::F32, s, c, l.n1w.ptr(), l.n1b.ptr(), EPS, hb.ptr(), Dt::F32)?;
            self.lin(&l.qkv, hb.ptr(), s, qkv.ptr())?;
            let q = qkv.fp();
            self.nsd.attention(q.cast(), q.wrapping_add(c).cast(), q.wrapping_add(2 * c).cast(), Dt::F32, s, self.heads, d, 3 * c, att.ptr(), Dt::F32)?;
            self.lin(&l.o, att.ptr(), s, tmp.ptr())?;
            self.nsd.gate_add(x.ptr(), Dt::F32, s, c, tmp.ptr(), Dt::F32, zeros, l.ls1.ptr())?;
            self.nsd.layer_norm(x.ptr(), Dt::F32, s, c, l.n2w.ptr(), l.n2b.ptr(), EPS, hb.ptr(), Dt::F32)?;
            self.lin(&l.fc1, hb.ptr(), s, mid.ptr())?;
            self.nsd.gelu(mid.ptr(), Dt::F32, s * ffn, false)?;
            self.lin(&l.fc2, mid.ptr(), s, tmp.ptr())?;
            self.nsd.gate_add(x.ptr(), Dt::F32, s, c, tmp.ptr(), Dt::F32, zeros, l.ls2.ptr())?;
            if self.taps.contains(&i) {
                let f = self.buf(s * c)?;
                self.nsd.layer_norm(x.ptr(), Dt::F32, s, c, self.lnw.ptr(), self.lnb.ptr(), EPS, f.ptr(), Dt::F32)?;
                feats.push(f);
            }
        }
        // the neck: each level's patches (the class token's row dropped) as a map, reassembled, convolved to 64
        let mut levels = Vec::new();
        for (i, f) in feats.iter().enumerate() {
            let map = Map { buf: f.view(c * 4, np * c * 4)?, h: gh, w: gw, c };
            let p = self.lin_map(&self.proj[i], &map)?;
            let r = match &self.resize[i] {
                Resize::Same => p,
                Resize::Down(cv) => self.conv(cv, &p)?,
                Resize::Up(lin, bias, f) => {
                    let ch = p.c;
                    let t = self.lin_map(lin, &p)?;
                    let base = self.buf(np * ch)?;
                    // SAFETY: bias [ch] to every one of np rows of base [np, ch].
                    check(unsafe { (self.k.broadcast_row)(self.g(), bias.fp(), np as i64, ch as i64, base.fp(), ch as i64) }, "bias rows")?;
                    let o = self.map(gh * f, gw * f, ch)?;
                    // SAFETY: t [gh, gw, ch f f], base [gh, gw, ch], o [gh f, gw f, ch].
                    let rc = unsafe { (self.k.pixel_shuffle_add)(self.nsd.raw(), t.buf.ptr(), Dt::F32 as i32, 1, gh as i64, gw as i64, ch as i64, *f as i64,
                                                                 base.ptr(), o.buf.ptr()) };
                    check(rc, "pixel shuffle")?;
                    o
                }
            };
            levels.push(self.conv(&self.convs[i], &r)?);
        }
        // fusion, deepest level first: each fused map up to the next level's size (the last x 2), then its 1 x 1
        let mut fused: Option<Map> = None;
        for (j, fu) in self.fusion.iter().enumerate() {
            let lvl = &levels[3 - j];
            let x = match fused.take() {
                None => self.rcu(&fu.rcu2, lvl)?,
                Some(f) => {
                    let r = self.rcu(&fu.rcu1, lvl)?;
                    self.add(&f, &r)?;
                    self.rcu(&fu.rcu2, &f)?
                }
            };
            let (th, tw) = if j < 3 { (levels[2 - j].h, levels[2 - j].w) } else { (x.h * 2, x.w * 2) };
            let u = self.up(&x, th, tw)?;
            fused = Some(self.lin_map(&fu.proj, &u)?);
        }
        let f = fused.expect("four levels");
        let a = self.conv(&self.head1, &f)?;
        let a = self.up(&a, gh * PATCH, gw * PATCH)?;
        let b = self.conv(&self.head2, &a)?;
        self.relu(&b)?;
        let out = self.lin_map(&self.head3, &b)?;
        self.relu(&out)?;
        let v = out.buf.to_f32()?;
        // (to_f32 waited for the stream: the temporaries are free to go)
        Ok(v)
    }
}
