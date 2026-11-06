//! The denoiser: FlashVSR's Wan 2.1 T2V-1.3B (diffsynth's `WanModel` as FlashVSR changes it), one step from noise
//! at timestep 1000, 30 blocks of 1,536 features (12 heads x 128), its weights in half on the card.
//!
//! ```text
//!   tokens: the noise's 1 x 2 x 2 patches [L, 64] -> patch embedding -> + the LQ projection (before block 0)
//!   block:  x += g1 * o(sparse_attention(rope(rms(q h)), rope(rms(k h)), v h)),  h = LN(x) * (1 + s1) + b1
//!           x += o(attention(rms(q LN3(x)), the prompt's cached k, v))
//!           x += g2 * W2 gelu_tanh(W1 h),  h = LN(x) * (1 + s2) + b2
//!   head:   LN(x) * (1 + scale) + shift -> linear -> [L, 64] -> latents = noise - output
//! ```
//!
//! The modulations are constant (one timestep, 1000) and the prompt is FlashVSR's fixed one (posi_prompt.pth), so both
//! are made once at load. The self-attention is FlashVSR's locality-constrained block-sparse attention over windows of
//! 2 x 8 x 8 tokens: per head, the means of each block's queries and keys score every key block (outside an
//! lr x lr window of blocks: none), the scores' softmax keeps its top k across each query group, and each query block
//! attends to the 128-key blocks it kept. Keys and values of the last three latent pairs are kept (the KV cache, the
//! stream's memory); a chunk's own pair is put in a spare and moved into the oldest pair's place after.
//!
//! The tokens are in window order throughout (see fv.h), so blocks are runs of 128 rows. The RoPE pairs features
//! (2i, 2i + 1); the shared kernel rotates (i, 64 + i), so the self-attention's q and k rows (and their norms' weights)
//! are permuted at load. The layers can be split over cards (`Stage`): each holds its blocks, their KV cache and its
//! working buffers.

use std::sync::Arc;

use nextcuda_core::{DevBuf, Error, Gpu, Result};
use nextcuda_diffusion::kernels::{none, Dt, Nsd};
use nextcuda_gguf::safetensors::SafeTensors;

use crate::ffi::{self, check, g, half_bytes};

pub const DIM: usize = 1536;
pub const HEADS: usize = 12;
pub const HEAD: usize = 128;
pub const FFN: usize = 8960;
pub const LAYERS: usize = 30;
pub const TEXT: usize = 4096;
const EPS: f32 = 1e-6;
/// rows of the MLP made at a time (its 8,960-wide hidden layer)
const FFN_ROWS: usize = 4096;

/// A linear layer on the card: weight half [n, k], bias float32 [n]
pub struct Lin {
    pub w: DevBuf,
    pub b: Option<DevBuf>,
    pub n: usize,
    pub k: usize,
}

impl Lin {
    pub fn from(gpu: &Arc<Gpu>, w: &[f32], b: Option<&[f32]>, n: usize, k: usize) -> Result<Lin> {
        if w.len() != n * k {
            return Err(Error(format!("a linear layer of {} values, {n} x {k} expected", w.len())));
        }
        let wd = DevBuf::new(gpu, n * k * 2)?;
        wd.write(0, &half_bytes(w))?;
        let b = b.map(|b| DevBuf::from_f32(gpu, b)).transpose()?;
        Ok(Lin { w: wd, b, n, k })
    }

    pub fn apply(&self, nsd: &Nsd, x: *const std::ffi::c_void, m: usize, out: *mut std::ffi::c_void, out_dt: Dt) -> Result<()> {
        let b = self.b.as_ref().map_or(none(), |b| b.ptr() as *const _);
        nsd.linear(x, Dt::F16, m, self.k, self.w.ptr(), self.n, b, out, out_dt)
    }

    pub fn bytes(&self) -> usize {
        self.n * self.k * 2 + self.b.as_ref().map_or(0, |_| self.n * 4)
    }
}

/// q / k row order within a head: pair (2i, 2i + 1) -> (i, 64 + i)
fn rope_src(j: usize) -> usize {
    if j < HEAD / 2 {
        2 * j
    } else {
        2 * (j - HEAD / 2) + 1
    }
}

fn permute_rows(w: &[f32], rows: usize) -> Vec<f32> {
    let k = w.len() / rows;
    let mut out = Vec::with_capacity(w.len());
    for r in 0..rows {
        let src = (r / HEAD) * HEAD + rope_src(r % HEAD);
        out.extend_from_slice(&w[src * k..(src + 1) * k]);
    }
    out
}

/// One block: its matrices, norms and constant modulation, the prompt's keys and values, its KV cache
pub struct Layer {
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    cq: Lin,
    co: Lin,
    w1: Lin,
    w2: Lin,
    norm_q: DevBuf,
    norm_k: DevBuf,
    cnorm_q: DevBuf,
    n3_w: DevBuf,
    n3_b: DevBuf,
    /// shift and 1 + scale before the attention and the MLP, and their gates: float32 [1536] each
    sh1: DevBuf,
    sc1: DevBuf,
    g1: DevBuf,
    sh2: DevBuf,
    sc2: DevBuf,
    g2: DevBuf,
    /// the prompt's keys (normed) and values for the cross-attention: half [512, 1536]
    ck: DevBuf,
    cv: DevBuf,
    ctx_rows: usize,
    /// the KV cache: three latent pairs' keys and values, half [3 nb 128, 1536], and the block means of the keys
    /// (float32 [4 nb, 1536]: the three pairs', then the chunk's own)
    kc: Option<DevBuf>,
    vc: Option<DevBuf>,
    km: Option<DevBuf>,
}

/// The shape of a chunk: its query groups (latent pairs), the block grid, its place in the stream
#[derive(Clone, Copy, Debug)]
pub struct Chunk {
    /// latent pairs in the chunk (3 for the first, then 1)
    pub groups: usize,
    pub nbh: usize,
    pub nbw: usize,
    /// the first chunk (its pairs fill the cache) or a later one (its pair goes to the spare, then over `evict`)
    pub first: bool,
    pub evict: usize,
    /// top-k blocks a query group keeps (FlashVSR's topk: int(nb^2 * ratio) - 1)
    pub topk: i64,
    /// the local window, in blocks
    pub local: usize,
}

impl Chunk {
    pub fn nb(&self) -> usize {
        self.nbh * self.nbw
    }
    pub fn rows(&self) -> usize {
        self.groups * self.nb() * 128
    }
    /// key blocks a query sees: the cached pairs and the chunk's own
    pub fn nk(&self) -> usize {
        if self.first {
            3 * self.nb()
        } else {
            4 * self.nb()
        }
    }
}

/// The working buffers of a stage, sized for the largest chunk
struct Work {
    rows: usize,
    nb: usize,
    h: DevBuf,
    q: DevBuf,
    attn: DevBuf,
    y: DevBuf,
    hid: DevBuf,
    kt: DevBuf,
    vt: DevBuf,
    qm: DevBuf,
    probs: DevBuf,
    thr: DevBuf,
    lists: DevBuf,
    counts: DevBuf,
}

/// Layers [from, to) of the DiT on one card
pub struct Stage {
    /// seconds by operation (NC_FLASHVSR_PROFILE=1: the card waited for after each)
    pub ops: Vec<(&'static str, f64)>,
    pub gpu: Arc<Gpu>,
    pub nsd: Nsd,
    pub from: usize,
    pub to: usize,
    layers: Vec<Layer>,
    work: Option<Work>,
    pub weight_bytes: usize,
}

/// The parts before and after the blocks: the patch embedding (on the first stage's card), the head (on the last's)
pub struct Ends {
    pub patch: Lin,
    pub head: Lin,
    pub head_sc: DevBuf,
    pub head_sh: DevBuf,
}

fn need(st: &SafeTensors, name: &str) -> Result<Vec<f32>> {
    st.f32(name).map_err(|e| Error(e.0))
}

fn silu(v: &mut [f32]) {
    v.iter_mut().for_each(|x| *x /= 1.0 + (-*x).exp());
}

fn host_linear(w: &[f32], b: &[f32], x: &[f32]) -> Vec<f32> {
    let k = x.len();
    w.chunks_exact(k).zip(b).map(|(row, bi)| row.iter().zip(x).map(|(a, c)| (*a as f64) * (*c as f64)).sum::<f64>() as f32 + bi).collect()
}

/// The timestep's embedding (1000, the one step) and the blocks' shared modulation: (t [1536], t_mod [6 x 1536])
pub fn time_embedding(st: &SafeTensors) -> Result<(Vec<f32>, Vec<f32>)> {
    let half = 128;
    let mut x = vec![0f32; 256];
    for i in 0..half {
        let f = 10000f64.powf(-(i as f64) / half as f64);
        let a = 1000.0 * f;
        x[i] = a.cos() as f32;
        x[half + i] = a.sin() as f32;
    }
    let mut h = host_linear(&need(st, "time_embedding.0.weight")?, &need(st, "time_embedding.0.bias")?, &x);
    silu(&mut h);
    let t = host_linear(&need(st, "time_embedding.2.weight")?, &need(st, "time_embedding.2.bias")?, &h);
    let mut ts = t.clone();
    silu(&mut ts);
    let tm = host_linear(&need(st, "time_projection.1.weight")?, &need(st, "time_projection.1.bias")?, &ts);
    Ok((t, tm))
}

impl Ends {
    pub fn load(st: &SafeTensors, first: &Arc<Gpu>, last: &Arc<Gpu>, t: &[f32]) -> Result<Ends> {
        let patch = Lin::from(first, &need(st, "patch_embedding.weight")?, Some(&need(st, "patch_embedding.bias")?), DIM, 64)?;
        let head = Lin::from(last, &need(st, "head.head.weight")?, Some(&need(st, "head.head.bias")?), 64, DIM)?;
        let m = need(st, "head.modulation")?;
        let sh: Vec<f32> = (0..DIM).map(|i| m[i] + t[i]).collect();
        let sc: Vec<f32> = (0..DIM).map(|i| 1.0 + m[DIM + i] + t[i]).collect();
        Ok(Ends { patch, head, head_sc: DevBuf::from_f32(last, &sc)?, head_sh: DevBuf::from_f32(last, &sh)? })
    }
}

impl Stage {
    /// Layers [from, to) onto `gpu`; `ctx` the prompt after the text embedding (float32 [rows, 1536], on the host)
    #[allow(clippy::too_many_arguments)]
    pub fn load(st: &SafeTensors, gpu: &Arc<Gpu>, from: usize, to: usize, tmod: &[f32], ctx: &[f32], log: &mut dyn FnMut(String)) -> Result<Stage> {
        let nsd = Nsd::new(gpu)?;
        let rows = ctx.len() / DIM;
        let ctx_d = DevBuf::new(gpu, rows * DIM * 2)?;
        ctx_d.write(0, &half_bytes(ctx))?;
        let mut layers = Vec::new();
        let mut bytes = 0;
        for l in from..to {
            let p = |s: &str| format!("blocks.{l}.{s}");
            let lin = |name: &str, perm: bool| -> Result<Lin> {
                let w = need(st, &p(&format!("{name}.weight")))?;
                let b = need(st, &p(&format!("{name}.bias")))?;
                let (n, k) = (b.len(), w.len() / b.len());
                if perm {
                    Lin::from(gpu, &permute_rows(&w, n), Some(&permute_rows(&b, n)), n, k)
                } else {
                    Lin::from(gpu, &w, Some(&b), n, k)
                }
            };
            let vecd = |name: &str, perm: bool| -> Result<DevBuf> {
                let v = need(st, &p(name))?;
                DevBuf::from_f32(gpu, &if perm { permute_rows(&v, v.len()) } else { v })
            };
            let m = need(st, &p("modulation"))?;
            let md = |i: usize, one: f32| DevBuf::from_f32(gpu, &(0..DIM).map(|c| one + m[i * DIM + c] + tmod[i * DIM + c]).collect::<Vec<f32>>());
            // the prompt's keys and values, once: k normed, v as is
            let ck_lin = lin("cross_attn.k", false)?;
            let cv_lin = lin("cross_attn.v", false)?;
            let ck = DevBuf::new(gpu, rows * DIM * 2)?;
            let cv = DevBuf::new(gpu, rows * DIM * 2)?;
            ck_lin.apply(&nsd, ctx_d.ptr(), rows, ck.ptr(), Dt::F16)?;
            cv_lin.apply(&nsd, ctx_d.ptr(), rows, cv.ptr(), Dt::F16)?;
            let nk = vecd("cross_attn.norm_k.weight", false)?;
            nsd.rms_norm_mod(ck.ptr(), Dt::F16, rows, DIM, nk.ptr(), EPS, none(), none(), none(), ck.ptr(), Dt::F16)?;
            nsd.wait()?;
            let layer = Layer {
                q: lin("self_attn.q", true)?,
                k: lin("self_attn.k", true)?,
                v: lin("self_attn.v", false)?,
                o: lin("self_attn.o", false)?,
                cq: lin("cross_attn.q", false)?,
                co: lin("cross_attn.o", false)?,
                w1: lin("ffn.0", false)?,
                w2: lin("ffn.2", false)?,
                norm_q: vecd("self_attn.norm_q.weight", true)?,
                norm_k: vecd("self_attn.norm_k.weight", true)?,
                cnorm_q: vecd("cross_attn.norm_q.weight", false)?,
                n3_w: vecd("norm3.weight", false)?,
                n3_b: vecd("norm3.bias", false)?,
                sh1: md(0, 0.0)?,
                sc1: md(1, 1.0)?,
                g1: md(2, 0.0)?,
                sh2: md(3, 0.0)?,
                sc2: md(4, 1.0)?,
                g2: md(5, 0.0)?,
                ck,
                cv,
                ctx_rows: rows,
                kc: None,
                vc: None,
                km: None,
            };
            bytes += [&layer.q, &layer.k, &layer.v, &layer.o, &layer.cq, &layer.co, &layer.w1, &layer.w2].iter().map(|l| l.bytes()).sum::<usize>()
                + rows * DIM * 4;
            layers.push(layer);
        }
        log(format!("flashvsr: blocks {from}-{} on {} ({:.2} GiB)", to - 1, gpu.name, bytes as f64 / (1u64 << 30) as f64));
        Ok(Stage { ops: Vec::new(), gpu: gpu.clone(), nsd, from, to, layers, work: None, weight_bytes: bytes })
    }

    /// What the KV cache and the working buffers take for a grid of nb blocks a pair (bytes)
    pub fn need_bytes(&self, nb: usize) -> usize {
        let per = 2 * 3 * nb * 128 * DIM * 2 + 4 * nb * DIM * 4;
        let rows = 3 * nb * 128;
        let work = rows * DIM * (2 + 2) + nb * 128 * DIM * (2 + 4) + FFN_ROWS * FFN * 2 + 2 * nb * 128 * DIM * 2 + HEADS * 3 * nb * 4 * nb * (4 + 4) + (1 << 26);
        per * self.layers.len() + work
    }

    /// Frees the KV cache and the working buffers (the weights stay); `prepare` makes them again
    pub fn release(&mut self) {
        self.work = None;
        for l in &mut self.layers {
            l.kc = None;
            l.vc = None;
            l.km = None;
        }
    }

    /// Whether the cache and the buffers are made for grids of nb blocks
    pub fn is_prepared(&self, nb: usize) -> bool {
        self.work.as_ref().is_some_and(|w| w.nb == nb)
    }

    /// The cache and the buffers for grids of nb blocks (made again when the grid changes; the cache emptied)
    pub fn prepare(&mut self, nb: usize) -> Result<()> {
        if self.work.as_ref().is_some_and(|w| w.nb == nb) {
            return Ok(());
        }
        self.work = None;
        for l in &mut self.layers {
            l.kc = None;
            l.vc = None;
            l.km = None;
        }
        let gpu = &self.gpu;
        for l in &mut self.layers {
            l.kc = Some(DevBuf::new(gpu, 3 * nb * 128 * DIM * 2)?);
            l.vc = Some(DevBuf::new(gpu, 3 * nb * 128 * DIM * 2)?);
            l.km = Some(DevBuf::new(gpu, 4 * nb * DIM * 4)?);
        }
        let rows = 3 * nb * 128;
        let nq = 3 * nb;
        let nk = 4 * nb;
        self.work = Some(Work {
            rows,
            nb,
            // the row-wise work goes a latent pair at a time: h and y for one pair
            h: DevBuf::new(gpu, nb * 128 * DIM * 2)?,
            q: DevBuf::new(gpu, rows * DIM * 2)?,
            attn: DevBuf::new(gpu, rows * DIM * 2)?,
            y: DevBuf::new(gpu, nb * 128 * DIM * 4)?,
            hid: DevBuf::new(gpu, FFN_ROWS * FFN * 2)?,
            kt: DevBuf::new(gpu, nb * 128 * DIM * 2)?,
            vt: DevBuf::new(gpu, nb * 128 * DIM * 2)?,
            qm: DevBuf::new(gpu, nq * DIM * 4)?,
            probs: DevBuf::new(gpu, HEADS * nq * nk * 4)?,
            thr: DevBuf::new(gpu, HEADS * 3 * 4)?,
            lists: DevBuf::new(gpu, HEADS * nq * nk * 4)?,
            counts: DevBuf::new(gpu, HEADS * nq * 4)?,
        });
        Ok(())
    }

    /// The stage's blocks over x (float32 [rows, 1536] on this card), the RoPE table cs (float32 [rows, 64, 2])
    pub fn run(&mut self, x: &DevBuf, cs: &DevBuf, c: &Chunk, grab: &mut dyn FnMut(usize, &DevBuf) -> Result<()>) -> Result<()> {
        let k = ffi::api()?;
        let gp = g(&self.gpu);
        let nb = c.nb();
        let rows = c.rows();
        let w = self.work.as_ref().ok_or_else(|| Error("flashvsr: the stage was not prepared".into()))?;
        if w.nb != nb || rows > w.rows {
            return Err(Error("flashvsr: the stage's buffers are for another grid".into()));
        }
        let nsd = &self.nsd;
        let xp = x.ptr() as *mut f32;
        let (nq, nk) = (c.groups * nb, c.nk());
        let prof = std::env::var("NC_FLASHVSR_PROFILE").is_ok_and(|v| v == "1");
        let mut t = std::time::Instant::now();
        let gpu = self.gpu.clone();
        let mut ops: Vec<(&'static str, f64)> = std::mem::take(&mut self.ops);
        let mut mark = |name: &'static str| {
            if !prof {
                return;
            }
            let _ = gpu.sync();
            let s = t.elapsed().as_secs_f64();
            t = std::time::Instant::now();
            match ops.iter_mut().find(|(n, _)| *n == name) {
                Some(e) => e.1 += s,
                None => ops.push((name, s)),
            }
        };
        // a latent pair's rows at a time for the row-wise work (h and y hold one pair)
        let grp = nb * 128;
        let slices: Vec<(usize, usize)> = (0..rows).step_by(grp).map(|r0| (r0, grp.min(rows - r0))).collect();
        // a row's place in a buffer of DIM-wide rows of `el` bytes
        let at = |p: *mut std::ffi::c_void, r0: usize, el: usize| -> *mut std::ffi::c_void {
            // SAFETY: r0 is a row of the buffer (the slices stay inside the chunk's rows)
            unsafe { (p as *mut u8).add(r0 * DIM * el) as *mut std::ffi::c_void }
        };
        for (li, l) in self.layers.iter().enumerate() {
            let (kc, vc, km) = (l.kc.as_ref().unwrap(), l.vc.as_ref().unwrap(), l.km.as_ref().unwrap());
            // self-attention
            // SAFETY (every call below): the buffers hold what the call reads and writes, sized by prepare.
            let (kd, vd) = if c.first { (kc, vc) } else { (&w.kt, &w.vt) };
            for &(r0, r) in &slices {
                check(unsafe { (k.ln)(gp, xp.add(r0 * DIM), r as i64, DIM as i64, EPS, l.sc1.fp(), l.sh1.fp(), w.h.ptr()) }, "ln1")?;
                l.q.apply(nsd, w.h.ptr(), r, at(w.q.ptr(), r0, 2), Dt::F16)?;
                l.k.apply(nsd, w.h.ptr(), r, at(kd.ptr(), r0, 2), Dt::F16)?;
                l.v.apply(nsd, w.h.ptr(), r, at(vd.ptr(), r0, 2), Dt::F16)?;
            }
            nsd.rms_norm_mod(w.q.ptr(), Dt::F16, rows, DIM, l.norm_q.ptr(), EPS, none(), none(), none(), w.q.ptr(), Dt::F16)?;
            nsd.rms_norm_mod(kd.ptr(), Dt::F16, rows, DIM, l.norm_k.ptr(), EPS, none(), none(), none(), kd.ptr(), Dt::F16)?;
            nsd.rope(w.q.ptr(), Dt::F16, rows, HEADS, HEAD, DIM, cs.ptr(), HEAD)?;
            nsd.rope(kd.ptr(), Dt::F16, rows, HEADS, HEAD, DIM, cs.ptr(), HEAD)?;
            mark("ln + q k v + norms + rope");
            // the draft: block means, their scores, the top k, the lists
            let km_at = if c.first { 0 } else { 3 * nb * DIM * 4 };
            check(unsafe { (k.block_mean)(gp, w.q.ptr(), nq as i64, DIM as i64, DIM as i64, w.qm.fp()) }, "query means")?;
            check(unsafe { (k.block_mean)(gp, kd.ptr(), (c.groups * nb) as i64, DIM as i64, DIM as i64, (km.ptr() as *mut u8).add(km_at) as *mut f32) },
                  "key means")?;
            check(unsafe { (k.block_probs)(gp, w.qm.fp(), km.fp(), nq as i64, nk as i64, HEADS as i64, c.nbh as i64, c.nbw as i64, c.local as i64,
                                           w.probs.fp()) }, "block scores")?;
            let n = nb * nk;
            let rank = (c.topk.min(n as i64 - 1)).max(0) + 1;
            check(unsafe { (k.topk_thr)(gp, w.probs.fp(), (HEADS * c.groups) as i64, n as i64, rank, w.thr.fp()) }, "top k")?;
            check(unsafe { (k.block_lists)(gp, w.probs.fp(), w.thr.fp(), HEADS as i64, nq as i64, nk as i64, nb as i64, w.lists.ptr() as *mut i32,
                                           w.counts.ptr() as *mut i32) }, "block lists")?;
            mark("draft: block scores, top k, lists");
            check(unsafe { (k.sparse_attn)(gp, w.q.ptr(), rows as i64, HEADS as i64, kc.ptr(), vc.ptr(), w.kt.ptr(), w.vt.ptr(), (3 * nb) as i64,
                                           w.lists.ptr() as *const i32, w.counts.ptr() as *const i32, nk as i64, w.attn.ptr()) }, "sparse attention")?;
            mark("sparse attention");
            if !c.first {
                // the chunk's pair over the oldest
                let pair = nb * 128 * DIM * 2;
                kc.copy_within(c.evict * pair, &w.kt, 0, pair)?;
                vc.copy_within(c.evict * pair, &w.vt, 0, pair)?;
                km.copy_within(c.evict * nb * DIM * 4, km, 3 * nb * DIM * 4, nb * DIM * 4)?;
            }
            for &(r0, r) in &slices {
                l.o.apply(nsd, at(w.attn.ptr(), r0, 2), r, w.y.ptr(), Dt::F32)?;
                check(unsafe { (k.gate_add)(gp, xp.add(r0 * DIM), w.y.fp(), r as i64, DIM as i64, l.g1.fp()) }, "attention residual")?;
            }
            mark("kv cache + o + residual");
            // cross-attention to the prompt
            for &(r0, r) in &slices {
                check(unsafe { (k.ln)(gp, xp.add(r0 * DIM), r as i64, DIM as i64, EPS, l.n3_w.fp(), l.n3_b.fp(), w.h.ptr()) }, "ln3")?;
                l.cq.apply(nsd, w.h.ptr(), r, at(w.q.ptr(), r0, 2), Dt::F16)?;
            }
            nsd.rms_norm_mod(w.q.ptr(), Dt::F16, rows, DIM, l.cnorm_q.ptr(), EPS, none(), none(), none(), w.q.ptr(), Dt::F16)?;
            nsd.flash_attention(w.q.ptr(), rows, DIM, l.ck.ptr(), l.cv.ptr(), l.ctx_rows, DIM, HEADS, HEADS, w.attn.ptr())?;
            for &(r0, r) in &slices {
                l.co.apply(nsd, at(w.attn.ptr(), r0, 2), r, w.y.ptr(), Dt::F32)?;
                check(unsafe { (k.gate_add)(gp, xp.add(r0 * DIM), w.y.fp(), r as i64, DIM as i64, std::ptr::null()) }, "cross residual")?;
            }
            mark("cross-attention");
            // the MLP, a pair's rows at a time, its hidden layer FFN_ROWS rows at a time
            for &(g0, gr) in &slices {
                check(unsafe { (k.ln)(gp, xp.add(g0 * DIM), gr as i64, DIM as i64, EPS, l.sc2.fp(), l.sh2.fp(), w.h.ptr()) }, "ln2")?;
                let mut r0 = 0;
                while r0 < gr {
                    let r = FFN_ROWS.min(gr - r0);
                    l.w1.apply(nsd, at(w.h.ptr(), r0, 2), r, w.hid.ptr(), Dt::F16)?;
                    nsd.gelu(w.hid.ptr(), Dt::F16, r * FFN, true)?;
                    l.w2.apply(nsd, w.hid.ptr(), r, w.y.ptr(), Dt::F32)?;
                    check(unsafe { (k.gate_add)(gp, xp.add((g0 + r0) * DIM), w.y.fp(), r as i64, DIM as i64, l.g2.fp()) }, "mlp residual")?;
                    r0 += r;
                }
            }
            mark("mlp");
            grab(self.from + li, x)?;
        }
        self.ops = ops;
        Ok(())
    }
}

/// The prompt through the text embedding (on `gpu`), float32 [rows, 1536] on the host
pub fn text_embedding(st: &SafeTensors, gpu: &Arc<Gpu>, ctx: &[f32]) -> Result<Vec<f32>> {
    let nsd = Nsd::new(gpu)?;
    let rows = ctx.len() / TEXT;
    let l0 = Lin::from(gpu, &need(st, "text_embedding.0.weight")?, Some(&need(st, "text_embedding.0.bias")?), DIM, TEXT)?;
    let l2 = Lin::from(gpu, &need(st, "text_embedding.2.weight")?, Some(&need(st, "text_embedding.2.bias")?), DIM, DIM)?;
    let x = DevBuf::new(gpu, rows * TEXT * 2)?;
    x.write(0, &half_bytes(ctx))?;
    let h = DevBuf::new(gpu, rows * DIM * 2)?;
    let o = DevBuf::new(gpu, rows * DIM * 4)?;
    l0.apply(&nsd, x.ptr(), rows, h.ptr(), Dt::F16)?;
    nsd.gelu(h.ptr(), Dt::F16, rows * DIM, true)?;
    l2.apply(&nsd, h.ptr(), rows, o.ptr(), Dt::F32)?;
    nsd.wait()?;
    o.to_f32()
}

/// The RoPE table of a chunk's tokens (window order): (cos, sin) for each of the 64 pairs, frames from `t0`
pub fn rope_table(groups: usize, hp: usize, wp: usize, t0: usize) -> Vec<f32> {
    // Wan's split of a head's 64 pairs: 22 for the frame, 21 for the row, 21 for the column, each its own frequencies
    let axis = |dim: usize| -> Vec<f64> { (0..dim / 2).map(|i| 1.0 / 10000f64.powf((2 * i) as f64 / dim as f64)).collect() };
    let (ft, fh, fw) = (axis(44), axis(42), axis(42));
    let f = 2 * groups;
    let rows = f * hp * wp;
    let mut cs = vec![0f32; rows * 128];
    let nbw = wp / 8;
    let nb = (hp / 8) * nbw;
    let put = |out: &mut [f32], pair: usize, a: f64| {
        let a = a.rem_euclid(std::f64::consts::TAU);
        out[2 * pair] = a.cos() as f32;
        out[2 * pair + 1] = a.sin() as f32;
    };
    for t in 0..f {
        for y in 0..hp {
            for x in 0..wp {
                let g = t / 2;
                let b = (y / 8) * nbw + x / 8;
                let idx = ((g * nb + b) << 7) + (t & 1) * 64 + (y & 7) * 8 + (x & 7);
                let out = &mut cs[idx * 128..(idx + 1) * 128];
                let mut pair = 0;
                for fr in &ft {
                    put(out, pair, (t0 + t) as f64 * fr);
                    pair += 1;
                }
                for fr in &fh {
                    put(out, pair, y as f64 * fr);
                    pair += 1;
                }
                for fr in &fw {
                    put(out, pair, x as f64 * fr);
                    pair += 1;
                }
            }
        }
    }
    cs
}
