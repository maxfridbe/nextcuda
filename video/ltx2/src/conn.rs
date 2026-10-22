//! The text connectors (diffusers' LTX2TextConnectors with per-modality projections, LTX-2.3's and 2.5's): Gemma's 49
//! hidden states into the DiT's two conditions, video [1024, 4096] and audio [1024, 2048].
//!
//! ```text
//!   s   = rms_norm(state) per token and state, the 49 side by side          [T, 49 x 3840]   (no weight)
//!   p   = proj(s * sqrt(dim / 3840)) + b                                      [T, dim]         (the scale folded in)
//!   x   = [p ; registers[T % 128 ...]]                                        [1024, dim]      the padding's rows are
//!                                                                               the learned registers, by position
//!   8 x x += attn(rms(x)); x += ff(rms(x))        attention over all 1,024 rows: q, k RMS-normed across the heads
//!                                                 (weights), split RoPE (1-D, base 4096, theta 1e4), per-head gates
//!                                                 2 sigmoid(to_gate_logits(rms(x))); ff gelu_tanh, 4 x dim
//!   out = rms_norm(x)                                                         (no weight)
//! ```

use std::sync::Arc;

use nextcuda_core::{DevBuf, Error, Gpu, Result};
use nextcuda_diffusion::kernels::{none, Dt, Nsd};
use nextcuda_gguf::Gguf;

use crate::ffi;
use crate::te::{as_bytes, host_f32, vec_on, Mat, HIDDEN, STATES};

pub const LAYERS: usize = 8;
pub const HEADS: usize = 32;
pub const REGISTERS: usize = 128;
/// the padded length the pipeline runs the connectors over
pub const SEQ: usize = 1024;
const EPS: f32 = 1e-6;
const THETA: f64 = 10_000.0;
const BASE_SEQ: f32 = 4096.0;

struct Block {
    q: Mat,
    k: Mat,
    v: Mat,
    o: Mat,
    g: Mat,
    qb: DevBuf,
    kb: DevBuf,
    vb: DevBuf,
    ob: DevBuf,
    gb: DevBuf,
    qn: DevBuf,
    kn: DevBuf,
    ff1: Mat,
    ff1b: DevBuf,
    ff2: Mat,
    ff2b: DevBuf,
}

/// One modality's connector on one card
pub struct Connector {
    pub dim: usize,
    gpu: Arc<Gpu>,
    nsd: Nsd,
    proj: Mat,
    proj_b: DevBuf,
    reg: Vec<f32>,
    blocks: Vec<Block>,
}

impl Connector {
    /// `short`: "cv" (video, 4096) or "ca" (audio, 2048)
    pub fn load(f: &Gguf, short: &str, gpu: &Arc<Gpu>) -> Result<Connector> {
        let m = |s: &str| Mat::load(f, &format!("{short}.{s}"), gpu);
        let v = |s: &str| vec_on(f, &format!("{short}.{s}"), gpu);
        let proj = m("proj")?;
        let dim = proj.rows;
        let mut blocks = Vec::new();
        for b in 0..LAYERS {
            let m = |s: &str| Mat::load(f, &format!("{short}.{b}.{s}"), gpu);
            let v = |s: &str| vec_on(f, &format!("{short}.{b}.{s}"), gpu);
            blocks.push(Block {
                q: m("q")?, k: m("k")?, v: m("v")?, o: m("o")?, g: m("g")?,
                qb: v("q_b")?, kb: v("k_b")?, vb: v("v_b")?, ob: v("o_b")?, gb: v("g_b")?,
                qn: v("qn")?, kn: v("kn")?,
                ff1: m("ff1")?, ff1b: v("ff1_b")?, ff2: m("ff2")?, ff2b: v("ff2_b")?,
            });
        }
        let reg = host_f32(f, &format!("{short}.reg"))?;
        if reg.len() != REGISTERS * dim {
            return Err(Error(format!("{short}.reg: {} values for {REGISTERS} x {dim}", reg.len())));
        }
        Ok(Connector { dim, nsd: Nsd::new(gpu)?, gpu: gpu.clone(), proj_b: v("proj_b")?, proj, reg, blocks })
    }

    /// (cos, sin) per (row, head) and pair: the 1-D split RoPE over the padded length
    fn rope(&self) -> Vec<f32> {
        let d = self.dim / HEADS;
        let nf = self.dim / 2;
        let freqs: Vec<f32> = (0..nf).map(|k| {
            let e = if nf > 1 { k as f64 / (nf - 1) as f64 } else { 0.0 };
            (THETA.powf(e) * std::f64::consts::PI / 2.0) as f32
        }).collect();
        let mut cs = vec![0f32; SEQ * self.dim];
        for p in 0..SEQ {
            let g = p as f32 / BASE_SEQ;
            for h in 0..HEADS {
                for i in 0..d / 2 {
                    let a = (g * 2.0 - 1.0) * freqs[h * (d / 2) + i];
                    let at = ((p * HEADS + h) * (d / 2) + i) * 2;
                    cs[at] = (a as f64).cos() as f32;
                    cs[at + 1] = (a as f64).sin() as f32;
                }
            }
        }
        cs
    }

    /// The condition [1024, dim] float32 from a prompt's states [T, 49, 3840] (T at most 1,024)
    pub fn run(&self, states: &crate::te::States) -> Result<Vec<f32>> {
        let (gpu, nsd) = (&self.gpu, &self.nsd);
        let t = states.tokens.min(SEQ);
        let dim = self.dim;
        let d = dim / HEADS;
        let n = SEQ;
        // the registers fill the rows past the prompt, by position
        let mut x0 = vec![0f32; n * dim];
        for r in t..n {
            x0[r * dim..(r + 1) * dim].copy_from_slice(&self.reg[(r % REGISTERS) * dim..(r % REGISTERS + 1) * dim]);
        }
        let x = DevBuf::from_f32(gpu, &x0)?;
        if t > 0 {
            let wide = STATES * HIDDEN;
            let s = DevBuf::f32(gpu, t * wide)?;
            s.write(0, as_bytes(&states.v[..t * wide]))?;
            let sh = DevBuf::new(gpu, t * wide * 2)?;
            nsd.rms_norm_mod(s.ptr(), Dt::F32, t * STATES, HIDDEN, none(), EPS, none(), none(), none(), sh.ptr(), Dt::F16)?;
            self.proj.apply(nsd, sh.ptr(), t, x.ptr(), Dt::F32, 0, self.proj_b.fp())?;
            nsd.wait()?;
        }
        let cs = DevBuf::from_f32(gpu, &self.rope())?;
        let h = DevBuf::new(gpu, n * 4 * dim * 2)?;
        let qf = DevBuf::f32(gpu, n * dim)?;
        let q = DevBuf::new(gpu, n * dim * 2)?;
        let k = DevBuf::new(gpu, n * dim * 2)?;
        let v = DevBuf::new(gpu, n * dim * 2)?;
        let att = DevBuf::new(gpu, n * dim * 2)?;
        let gl = DevBuf::f32(gpu, n * HEADS)?;
        let o = DevBuf::f32(gpu, n * 4 * dim)?;
        for b in &self.blocks {
            nsd.rms_norm_mod(x.ptr(), Dt::F32, n, dim, none(), EPS, none(), none(), none(), h.ptr(), Dt::F16)?;
            b.g.apply(nsd, h.ptr(), n, gl.ptr(), Dt::F32, 0, b.gb.fp())?;
            b.q.apply(nsd, h.ptr(), n, qf.ptr(), Dt::F32, 0, b.qb.fp())?;
            nsd.rms_norm_mod(qf.ptr(), Dt::F32, n, dim, b.qn.ptr(), EPS, none(), none(), none(), q.ptr(), Dt::F16)?;
            b.k.apply(nsd, h.ptr(), n, qf.ptr(), Dt::F32, 0, b.kb.fp())?;
            nsd.rms_norm_mod(qf.ptr(), Dt::F32, n, dim, b.kn.ptr(), EPS, none(), none(), none(), k.ptr(), Dt::F16)?;
            b.v.apply(nsd, h.ptr(), n, v.ptr(), Dt::F16, 0, b.vb.fp())?;
            nsd.rope(q.ptr(), Dt::F16, n * HEADS, 1, d, d, cs.ptr(), d)?;
            nsd.rope(k.ptr(), Dt::F16, n * HEADS, 1, d, d, cs.ptr(), d)?;
            nsd.attention(q.ptr(), k.ptr(), v.ptr(), Dt::F16, n, HEADS, d, dim, att.ptr(), Dt::F16)?;
            ffi::head_gate(gpu, att.ptr(), n, HEADS, d, gl.fp())?;
            b.o.apply(nsd, att.ptr(), n, o.ptr(), Dt::F32, 0, b.ob.fp())?;
            nsd.gate_add(x.ptr(), Dt::F32, n, dim, o.ptr(), Dt::F32, none(), none())?;
            nsd.rms_norm_mod(x.ptr(), Dt::F32, n, dim, none(), EPS, none(), none(), none(), h.ptr(), Dt::F16)?;
            b.ff1.apply(nsd, h.ptr(), n, o.ptr(), Dt::F32, 0, b.ff1b.fp())?;
            ffi::geglu(gpu, o.fp(), 4 * dim, std::ptr::null(), 0, n, 4 * dim, 1.0, h.ptr())?;
            b.ff2.apply(nsd, h.ptr(), n, o.ptr(), Dt::F32, 0, b.ff2b.fp())?;
            nsd.gate_add(x.ptr(), Dt::F32, n, dim, o.ptr(), Dt::F32, none(), none())?;
        }
        let out = DevBuf::f32(gpu, n * dim)?;
        nsd.rms_norm_mod(x.ptr(), Dt::F32, n, dim, none(), EPS, none(), none(), none(), out.ptr(), Dt::F32)?;
        nsd.wait()?;
        out.to_f32()
    }
}
