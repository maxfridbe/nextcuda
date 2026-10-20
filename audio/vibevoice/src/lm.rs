//! The language model: Qwen2.5-1.5B (transformers' Qwen2Model under `model.language_model`) - 28 layers of 1,536,
//! 12 query / 2 key-value heads of 128 with q / k / v biases, rotary positions (θ 1e6, rotate-half), an 8,960 SwiGLU,
//! RMS norms (1e-6), the last one's output the hidden state the diffusion head is conditioned on. Six query heads a
//! key head is not a shape the decode kernels of this library take, so the cache is float32 rows [t, 2 x 128] and the
//! attention the shared kernels' causal rows (any grouping). The token embedding
//! stays on disk (rows are read for the prompt's text); the output head is tied to it, and the generation only ever
//! chooses between four tokens, so it is those four rows (`Head`).

use nextcuda_audio::{Error, Result};
use nextcuda_core::DevBuf;
use nextcuda_diffusion::kernels::{none, Dt, Nsd};

use crate::ops::{fp, nul, stage, Mat, Ops, Stage, Weights};

pub const HEAD: usize = 128;
const EPS: f32 = 1e-6;
const THETA: f32 = 1_000_000.0;
pub const PRE: &str = "model.language_model.";

struct Layer {
    in_norm: DevBuf,
    post_norm: DevBuf,
    qkv: Mat,
    qkv_b: DevBuf,
    o: Mat,
    gu: Mat,
    down: Mat,
}

pub struct Lm {
    layers: Vec<Layer>,
    norm: DevBuf,
    pub hidden: usize,
    heads: usize,
    kv: usize,
    ffn: usize,
    pub bytes: usize,
}

/// One sequence's cache and work buffers for up to `rows` rows a pass and `t` positions
pub struct Cache {
    pub t: usize,
    rows: usize,
    k: Vec<DevBuf>,
    v: Vec<DevBuf>,
    pub x: DevBuf,
    hb: DevBuf,
    qkv: DevBuf,
    att: DevBuf,
    gu: DevBuf,
    act: DevBuf,
    tmp: DevBuf,
    st: Stage,
    pub out: DevBuf,
}

impl Lm {
    pub fn load(ops: &Ops, w: &Weights, int8: bool, log: &mut dyn FnMut(String)) -> Result<Lm> {
        let mut layers = Vec::new();
        let mut bytes = 0;
        for l in 0.. {
            let pre = format!("{PRE}layers.{l}.");
            if !w.has(&format!("{pre}input_layernorm.weight")) {
                break;
            }
            let n = |s: &str| format!("{pre}{s}.weight");
            let b = |s: &str| format!("{pre}{s}.bias");
            let (q, k) = (w.shape(&n("self_attn.q_proj"))?[0], w.shape(&n("self_attn.k_proj"))?[0]);
            let ffn = w.shape(&n("mlp.gate_proj"))?[0];
            let layer = Layer {
                in_norm: w.dev_f32(ops, &n("input_layernorm"))?,
                post_norm: w.dev_f32(ops, &n("post_attention_layernorm"))?,
                qkv: w.mat(ops, &[(&n("self_attn.q_proj"), 0, q), (&n("self_attn.k_proj"), 0, k), (&n("self_attn.v_proj"), 0, k)], int8)?,
                qkv_b: w.cat_f32(ops, &[&b("self_attn.q_proj"), &b("self_attn.k_proj"), &b("self_attn.v_proj")])?,
                o: w.mat1(ops, &n("self_attn.o_proj"), int8)?,
                gu: w.mat(ops, &[(&n("mlp.gate_proj"), 0, ffn), (&n("mlp.up_proj"), 0, ffn)], int8)?,
                down: w.mat1(ops, &n("mlp.down_proj"), int8)?,
            };
            bytes += layer.qkv.bytes() + layer.o.bytes() + layer.gu.bytes() + layer.down.bytes();
            layers.push(layer);
            if l % 7 == 6 {
                log(format!("language model: {} layers", l + 1));
            }
        }
        let first = layers.first().ok_or_else(|| Error(format!("no {PRE}layers in the checkpoint")))?;
        let hidden = first.o.n;
        let (heads, kv, ffn) = (first.o.k / HEAD, (first.qkv.n - first.o.k) / 2 / HEAD, first.gu.n / 2);
        Ok(Lm { norm: w.dev_f32(ops, &format!("{PRE}norm.weight"))?, layers, hidden, heads, kv, ffn, bytes })
    }

    fn qkv_width(&self) -> usize {
        (self.heads + 2 * self.kv) * HEAD
    }

    /// A cache for `t` positions, passes of up to `rows` rows
    pub fn cache(&self, ops: &Ops, rows: usize, t: usize) -> Result<Cache> {
        let g = &ops.gpu;
        let kv = || -> Result<Vec<DevBuf>> { (0..self.layers.len()).map(|_| DevBuf::f32(g, t * self.kv * HEAD)).collect() };
        let (r, w) = (rows.max(1), self.qkv_width());
        let widest = self.hidden.max(self.heads * HEAD).max(self.ffn);
        let int8 = self.layers.first().is_some_and(|l| l.qkv.scale.is_some());
        let biggest = self.layers.iter().flat_map(|l| [&l.qkv, &l.o, &l.gu, &l.down]).map(|m| m.n * m.k).max().unwrap_or(0);
        Ok(Cache {
            t,
            rows: r,
            k: kv()?,
            v: kv()?,
            x: DevBuf::f32(g, r * self.hidden)?,
            hb: DevBuf::f32(g, r * self.hidden)?,
            qkv: DevBuf::f32(g, r * w)?,
            att: DevBuf::f32(g, r * self.heads * HEAD)?,
            gu: DevBuf::f32(g, r * 2 * self.ffn)?,
            act: DevBuf::f32(g, r * self.ffn)?,
            tmp: DevBuf::f32(g, r * self.hidden)?,
            st: stage(g, r * widest, int8.then_some(biggest))?,
            out: DevBuf::f32(g, r * self.hidden)?,
        })
    }

    /// x [r, k] . W^T added into c.x
    fn residual(&self, ops: &Ops, nsd: &Nsd, c: &Cache, m: &Mat, x: *const f32, r: usize) -> Result<()> {
        if r <= 8 && m.k % 256 == 0 {
            return ops.gemv(x, r, m, nul(), c.x.fp(), m.n, true);
        }
        m.apply(ops, nsd, x, r, &c.st, nul(), c.tmp.fp())?;
        ops.add(c.x.fp(), c.tmp.fp(), r * m.n)
    }

    /// `n` new positions (rows of c.x, from position p0) through every layer; the normed output in c.out
    pub fn pass(&self, ops: &Ops, nsd: &Nsd, c: &Cache, n: usize, p0: usize) -> Result<()> {
        if p0 + n > c.t || n > c.rows {
            return Err(Error(format!("a pass of {n} rows at {p0}: the cache holds {} positions, {} rows a pass", c.t, c.rows)));
        }
        let (h, w, q) = (self.hidden, self.qkv_width(), self.heads * HEAD);
        for (li, l) in self.layers.iter().enumerate() {
            nsd.rms_norm_mod(c.x.ptr(), Dt::F32, n, h, l.in_norm.ptr(), EPS, none(), none(), none(), c.hb.ptr(), Dt::F32)?;
            l.qkv.apply(ops, nsd, c.hb.fp(), n, &c.st, l.qkv_b.ptr(), c.qkv.fp())?;
            ops.rope(c.qkv.fp(), w, n, self.heads, HEAD, THETA, p0)?;
            ops.rope(fp(&c.qkv, q), w, n, self.kv, HEAD, THETA, p0)?;
            // the new rows' keys and values into the cache (rows [p0, p0 + n))
            let kvw = self.kv * HEAD;
            c.k[li].copy_2d(p0 * kvw * 4, kvw * 4, &c.qkv, q * 4, w * 4, kvw * 4, n)?;
            c.v[li].copy_2d(p0 * kvw * 4, kvw * 4, &c.qkv, (q + kvw) * 4, w * 4, kvw * 4, n)?;
            // query rows [p0, p0 + n) over keys [0, p0 + r]: q is addressed from row 0
            let q0 = c.qkv.fp().wrapping_sub(p0 * w) as *const std::ffi::c_void;
            nsd.attention_causal_rows(q0, c.k[li].ptr(), c.v[li].ptr(), Dt::F32, p0, n, self.heads, self.kv, HEAD, w, kvw, c.att.ptr())?;
            self.residual(ops, nsd, c, &l.o, c.att.fp(), n)?;
            nsd.rms_norm_mod(c.x.ptr(), Dt::F32, n, h, l.post_norm.ptr(), EPS, none(), none(), none(), c.hb.ptr(), Dt::F32)?;
            l.gu.apply(ops, nsd, c.hb.fp(), n, &c.st, nul(), c.gu.fp())?;
            nsd.swiglu(c.gu.ptr(), Dt::F32, n, self.ffn, c.act.ptr(), Dt::F32)?;
            self.residual(ops, nsd, c, &l.down, c.act.fp(), n)?;
        }
        nsd.rms_norm_mod(c.x.ptr(), Dt::F32, n, h, self.norm.ptr(), EPS, none(), none(), none(), c.out.ptr(), Dt::F32)
    }

    /// Rows [n, hidden] (host) through the model from position p0, `chunk` at a time; the last row's hidden state
    /// left at c.out's row (n - 1) % chunk ... returned as its offset (floats) into c.out
    pub fn feed(&self, ops: &Ops, nsd: &Nsd, c: &Cache, rows: &[f32], p0: usize) -> Result<usize> {
        let h = self.hidden;
        let n = rows.len() / h;
        let mut done = 0;
        let mut last = 0;
        while done < n {
            let take = (n - done).min(c.rows);
            c.x.write(0, &crate::bytes(&rows[done * h..(done + take) * h]))?;
            self.pass(ops, nsd, c, take, p0 + done)?;
            last = (take - 1) * h;
            done += take;
        }
        Ok(last)
    }
}

/// The output head's rows for the tokens the generation chooses between (tied to the embedding), and those
/// tokens' embeddings
pub struct Head {
    pub ids: Vec<u32>,
    mat: Mat,
    pub embeds: Vec<Vec<f32>>,
    logits: DevBuf,
}

impl Head {
    pub fn load(ops: &Ops, w: &Weights, ids: &[u32]) -> Result<Head> {
        let name = format!("{PRE}embed_tokens.weight");
        let embeds: Vec<Vec<f32>> = ids.iter().map(|&i| w.rows_f32(&name, i as usize, i as usize + 1)).collect::<Result<_>>()?;
        let hidden = embeds[0].len();
        // a matrix [8, hidden] (the rows padded with zeros: the product takes up to 8 output rows at no cost)
        let mut m = vec![0f32; 8 * hidden];
        for (r, e) in embeds.iter().enumerate() {
            m[r * hidden..(r + 1) * hidden].copy_from_slice(e);
        }
        let f = DevBuf::from_f32(&ops.gpu, &m)?;
        let wbuf = DevBuf::new(&ops.gpu, 8 * hidden * 2)?;
        ops.to_half(f.fp(), wbuf.ptr(), 8 * hidden)?;
        ops.gpu.sync()?;
        Ok(Head { ids: ids.to_vec(), mat: Mat { w: wbuf, scale: None, n: 8, k: hidden }, embeds, logits: DevBuf::f32(&ops.gpu, 8)? })
    }

    /// The hidden state at `h`'s logits for the tokens, and the most likely one's index into `ids`
    pub fn choose(&self, ops: &Ops, h: *const f32) -> Result<(usize, Vec<f32>)> {
        ops.gemv(h, 1, &self.mat, nul(), self.logits.fp(), 8, false)?;
        let l = self.logits.to_f32()?;
        let l = l[..self.ids.len()].to_vec();
        let best = (0..l.len()).fold(0, |b, i| if l[i] > l[b] { i } else { b });
        Ok((best, l))
    }
}
