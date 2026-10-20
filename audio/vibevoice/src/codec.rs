//! The speech tokenizers (yvoice's modular_vibevoice_tokenizer): causal 1-D ConvNeXt stacks at 7.5 frames a second
//! (3,200 samples of 24 kHz a frame).
//!
//! ```text
//!   encoder (acoustic: 64-d means; semantic: 128-d): stem conv k7 1 -> 32, then per stage i (32 * 2^i channels;
//!     3, 3, 3, 3, 3, 3, 8 blocks) the blocks and a strided conv (k 2r, stride r for r = 2, 2, 4, 5, 5, 8), a head
//!     conv k7 2048 -> d
//!   decoder (acoustic): stem conv k7 64 -> 2048, 8 blocks, then per stage a transposed conv (k 2r, stride r for
//!     r = 8, 5, 5, 4, 2, 2) and 3 blocks, a head conv k7 32 -> 1
//!   block: x += gamma * dwconv7(rmsnorm(x));  x += ffn_gamma * fc2(gelu(fc1(rmsnorm(x))))
//! ```
//!
//! Every convolution is causal with zeros ahead. The generation runs them a frame at a time ("streaming"): each layer
//! keeps the input rows its next window needs - k - stride of them for a convolution (6 for k7; r for a stride-r
//! one), and for a transposed one (k = 2 stride) the one input before - which is what the reference's cache does
//! (it keeps k - 1 inputs of a transposed one and recomputes them; one is all an output there depends on). A voice
//! recording is encoded whole, as the reference's non-streaming path: each strided layer's input padded with zeros
//! behind to whole windows.
//!
//! Layout: the residual stream is rows [L, C] float32 (the norms and products work on rows); the convolutions take
//! [C, L], so a layer transposes around them. The products are half (the checkpoint's bf16), the rest float32.
#![allow(clippy::too_many_arguments)]

use nextcuda_audio::{Error, Result};
use nextcuda_core::DevBuf;
use nextcuda_diffusion::kernels::{none, Dt, Nsd};

use crate::ops::{fp, stage, Mat, Ops, Stage, Weights};

const EPS: f32 = 1e-5;
const K: usize = 7;

/// A convolution (or a transposed one) with what it needs of the rows before
struct Conv {
    w: DevBuf,
    b: DevBuf,
    ci: usize,
    co: usize,
    k: usize,
    stride: usize,
    transposed: bool,
}

impl Conv {
    fn load(ops: &Ops, w: &Weights, pre: &str, transposed: bool, stride: usize) -> Result<Conv> {
        let (wn, bn) = if transposed { (format!("{pre}.convtr.convtr.weight"), format!("{pre}.convtr.convtr.bias")) }
                       else { (format!("{pre}.conv.conv.weight"), format!("{pre}.conv.conv.bias")) };
        let s = w.shape(&wn)?;
        let (ci, co) = if transposed { (s[0], s[1]) } else { (s[1], s[0]) };
        Ok(Conv { w: w.dev_f32(ops, &wn)?, b: w.dev_f32(ops, &bn)?, ci, co, k: s[2], stride, transposed })
    }

    /// Input rows kept for the next window
    fn ctx(&self) -> usize {
        if self.transposed { 1 } else { self.k - self.stride }
    }
}

struct Block {
    c: usize,
    norm: DevBuf,
    dw_w: DevBuf,
    dw_b: DevBuf,
    gamma: DevBuf,
    ffn_norm: DevBuf,
    fc1: Mat,
    b1: DevBuf,
    fc2: Mat,
    b2: DevBuf,
    ffn_gamma: DevBuf,
}

impl Block {
    fn load(ops: &Ops, w: &Weights, pre: &str) -> Result<Block> {
        let n = |s: &str| format!("{pre}.{s}");
        let c = w.shape(&n("norm.weight"))?[0];
        Ok(Block {
            c,
            norm: w.dev_f32(ops, &n("norm.weight"))?,
            dw_w: w.dev_f32(ops, &n("mixer.conv.conv.conv.weight"))?,
            dw_b: w.dev_f32(ops, &n("mixer.conv.conv.conv.bias"))?,
            gamma: w.dev_f32(ops, &n("gamma"))?,
            ffn_norm: w.dev_f32(ops, &n("ffn_norm.weight"))?,
            fc1: w.mat1(ops, &n("ffn.linear1.weight"), false)?,
            b1: w.dev_f32(ops, &n("ffn.linear1.bias"))?,
            fc2: w.mat1(ops, &n("ffn.linear2.weight"), false)?,
            b2: w.dev_f32(ops, &n("ffn.linear2.bias"))?,
            ffn_gamma: w.dev_f32(ops, &n("ffn_gamma"))?,
        })
    }

    fn bytes(&self) -> usize {
        self.fc1.bytes() + self.fc2.bytes() + self.dw_w.len
    }
}

/// One stage: its conv ahead of its blocks
struct Stage_ {
    conv: Conv,
    blocks: Vec<Block>,
}

/// An encoder or the decoder
pub struct Stack {
    stages: Vec<Stage_>,
    head: Conv,
    pub bytes: usize,
    /// samples a frame (3,200)
    pub hop: usize,
    /// the output's channels (the latent's width, or 1 for the decoder)
    pub out_c: usize,
}

/// A stream's state: each layer's kept rows, in order (convs and blocks), zeros at the start
pub struct State {
    rows: Vec<DevBuf>,
}

/// Work buffers for passes of up to the input length they were made for
pub struct Work {
    a: DevBuf,
    b: DevBuf,
    cat: DevBuf,
    t1: DevBuf,
    t2: DevBuf,
    f: DevBuf,
    st: Stage,
    input: usize,
}

impl Stack {
    /// `pre`: model.acoustic_tokenizer.encoder etc.; `decoder`: the transposed upsampling stack
    pub fn load(ops: &Ops, w: &Weights, pre: &str, decoder: bool) -> Result<Stack> {
        let (layers, sub) = if decoder { ("upsample_layers", "") } else { ("downsample_layers", "") };
        let _ = sub;
        let mut stages = Vec::new();
        let mut bytes = 0;
        for i in 0.. {
            let cp = format!("{pre}.{layers}.{i}.0");
            let transposed = decoder && i > 0;
            let wn = if transposed { format!("{cp}.convtr.convtr.weight") } else { format!("{cp}.conv.conv.weight") };
            if !w.has(&wn) {
                break;
            }
            let s = w.shape(&wn)?;
            // a strided conv's kernel is twice its stride (the stem's 7 has stride 1)
            let k = s[2];
            let stride = if i == 0 { 1 } else { k / 2 };
            let conv = Conv::load(ops, w, &cp, transposed, stride)?;
            bytes += conv.w.len;
            let mut blocks = Vec::new();
            for j in 0.. {
                let bp = format!("{pre}.stages.{i}.{j}");
                if !w.has(&format!("{bp}.norm.weight")) {
                    break;
                }
                let b = Block::load(ops, w, &bp)?;
                bytes += b.bytes();
                blocks.push(b);
            }
            stages.push(Stage_ { conv, blocks });
        }
        if stages.is_empty() {
            return Err(Error(format!("{pre}: no {layers} in the checkpoint")));
        }
        let head = Conv::load(ops, w, &format!("{pre}.head"), false, 1)?;
        bytes += head.w.len;
        let hop = stages.iter().map(|s| s.conv.stride).product();
        Ok(Stack { out_c: head.co, stages, head, bytes, hop })
    }

    /// Each layer's (rows in, channels in, rows out, channels out) for an input of `l` rows (`whole`: a
    /// non-streaming pass, strided layers padded to whole windows)
    fn shapes(&self, l: usize, whole: bool) -> Vec<(usize, usize, usize, usize, usize)> {
        // (rows in, ci, extra zeros behind, rows out, co) per conv in order: stages' convs then the head
        let mut out = Vec::new();
        let mut l = l;
        let convs = self.stages.iter().map(|s| &s.conv).chain(std::iter::once(&self.head));
        for c in convs {
            let (lo, extra) = if c.transposed {
                (l * c.stride, 0)
            } else {
                let ctx = c.k - c.stride;
                let extra = if whole && c.stride > 1 {
                    // the reference's get_extra_padding_for_conv1d with its padding_total (= ctx)
                    let frames = (l + ctx).saturating_sub(c.k) as f64 / c.stride as f64 + 1.0;
                    let ideal = (frames.ceil() as usize - 1) * c.stride + (c.k - ctx);
                    ideal.saturating_sub(l)
                } else {
                    0
                };
                ((ctx + l + extra - c.k) / c.stride + 1, extra)
            };
            out.push((l, c.ci, extra, lo, c.co));
            l = lo;
        }
        out
    }

    /// The rows out for an input of `l`
    pub fn out_rows(&self, l: usize, whole: bool) -> usize {
        self.shapes(l, whole).last().map(|s| s.3).unwrap_or(0)
    }

    /// A fresh stream (zeros kept everywhere)
    pub fn state(&self, ops: &Ops) -> Result<State> {
        let mut rows = Vec::new();
        for s in &self.stages {
            rows.push(DevBuf::f32(&ops.gpu, s.conv.ctx() * s.conv.ci)?);
            for b in &s.blocks {
                rows.push(DevBuf::f32(&ops.gpu, (K - 1) * b.c)?);
            }
        }
        rows.push(DevBuf::f32(&ops.gpu, self.head.ctx() * self.head.ci)?);
        let st = State { rows };
        st.reset()?;
        Ok(st)
    }

    /// Buffers for passes of up to `l` input rows
    pub fn work(&self, ops: &Ops, l: usize, whole: bool) -> Result<Work> {
        let sh = self.shapes(l, whole);
        let (mut rows, mut cat, mut ffn) = (0usize, 0usize, 0usize);
        let mut si = 0;
        for (i, s) in self.stages.iter().enumerate() {
            let (li, ci, extra, lo, co) = sh[i];
            cat = cat.max((s.conv.ctx() + li + extra) * ci).max((s.conv.ctx() + li) * ci);
            rows = rows.max(li * ci).max(lo * co);
            for b in &s.blocks {
                cat = cat.max((K - 1 + lo) * b.c);
                ffn = ffn.max(lo * 4 * b.c);
            }
            // a transposed conv's full output before its trim
            if s.conv.transposed {
                rows = rows.max((li + 1) * s.conv.stride * co + s.conv.k * co);
            }
            si = i;
        }
        let (li, ci, _, lo, co) = sh[si + 1];
        cat = cat.max((self.head.ctx() + li) * ci);
        rows = rows.max(lo * co).max(li * ci);
        let g = &ops.gpu;
        let n = rows.max(cat);
        Ok(Work {
            a: DevBuf::f32(g, n)?,
            b: DevBuf::f32(g, n)?,
            cat: DevBuf::f32(g, cat)?,
            t1: DevBuf::f32(g, n.max(cat))?,
            t2: DevBuf::f32(g, n.max(cat))?,
            f: DevBuf::f32(g, ffn.max(1))?,
            st: stage(g, ffn.max(rows).max(1), None)?,
            input: l,
        })
    }

    /// One conv over rows x [l, ci] (w.a) after its kept rows (and `extra` zeros behind); out rows into w.a;
    /// the kept rows updated
    fn conv(&self, ops: &Ops, c: &Conv, kept: &DevBuf, w: &Work, l: usize, extra: usize) -> Result<usize> {
        let ctx = c.ctx();
        let ci = c.ci;
        // [kept | x | zeros] as rows, then [ci, ctx + l + extra]
        w.cat.copy_within(0, kept, 0, ctx * ci * 4)?;
        w.cat.copy_within(ctx * ci * 4, &w.a, 0, l * ci * 4)?;
        if extra > 0 {
            w.cat.write((ctx + l) * ci * 4, &vec![0u8; extra * ci * 4])?;
        }
        let total = ctx + l + extra;
        ops.transpose(w.cat.fp(), total, ci, w.t1.fp())?;
        let lo = if c.transposed {
            // the full output over [prev, x], its first `stride` (the prev input's own) and last k - stride cut
            ops.conv_t(w.t1.fp(), ci, total, &c.w, c.co, c.k, c.b.fp(), c.stride, c.stride, c.k - c.stride, w.t2.fp())?
        } else {
            ops.conv(w.t1.fp(), ci, total, &c.w, c.co, c.k, c.b.fp(), c.stride, w.t2.fp())?
        };
        // the rows kept: the last ctx of [kept | x] (before the zeros)
        kept.copy_within(0, &w.cat, l * ci * 4, ctx * ci * 4)?;
        ops.transpose(w.t2.fp(), c.co, lo, w.a.fp())?;
        Ok(lo)
    }

    /// One block over rows w.a [l, c]
    fn block(&self, ops: &Ops, nsd: &Nsd, b: &Block, kept: &DevBuf, w: &Work, l: usize) -> Result<()> {
        let c = b.c;
        // mixer: rows normed after the kept ones, [c, 6 + l], the causal depthwise conv, its last l columns
        w.cat.copy_within(0, kept, 0, (K - 1) * c * 4)?;
        nsd.rms_norm_mod(w.a.ptr(), Dt::F32, l, c, b.norm.ptr(), EPS, none(), none(), none(), fp(&w.cat, (K - 1) * c).cast(), Dt::F32)?;
        kept.copy_within(0, &w.cat, l * c * 4, (K - 1) * c * 4)?;
        let total = K - 1 + l;
        ops.transpose(w.cat.fp(), total, c, w.t1.fp())?;
        ops.dwconv(w.t1.fp(), c, total, &b.dw_w, b.dw_b.fp(), K, w.t2.fp())?;
        ops.transpose(w.t2.fp(), c, total, w.t1.fp())?;
        ops.scale_add(w.a.fp(), fp(&w.t1, (K - 1) * c), &b.gamma, l, c)?;
        // ffn
        nsd.rms_norm_mod(w.a.ptr(), Dt::F32, l, c, b.ffn_norm.ptr(), EPS, none(), none(), none(), w.b.ptr(), Dt::F32)?;
        b.fc1.apply(ops, nsd, w.b.fp(), l, &w.st, b.b1.ptr(), w.f.fp())?;
        nsd.gelu(w.f.ptr(), Dt::F32, l * 4 * c, false)?;
        b.fc2.apply(ops, nsd, w.f.fp(), l, &w.st, b.b2.ptr(), w.b.fp())?;
        ops.scale_add(w.a.fp(), w.b.fp(), &b.ffn_gamma, l, c)
    }

    /// Rows x [l, in channels] (host) through the stack, the stream's state carried; out rows [lo, out_c]
    pub fn run(&self, ops: &Ops, nsd: &Nsd, st: &State, w: &Work, x: &[f32], whole: bool) -> Result<Vec<f32>> {
        let ci0 = self.stages[0].conv.ci;
        let l = x.len() / ci0;
        if l > w.input {
            return Err(Error(format!("{l} rows: the work buffers were made for {}", w.input)));
        }
        let sh = self.shapes(l, whole);
        w.a.write(0, &crate::bytes(x))?;
        let mut cur = l;
        let mut k = 0;
        for (i, s) in self.stages.iter().enumerate() {
            cur = self.conv(ops, &s.conv, &st.rows[k], w, cur, sh[i].2)?;
            k += 1;
            for b in &s.blocks {
                self.block(ops, nsd, b, &st.rows[k], w, cur)?;
                k += 1;
            }
        }
        let lo = self.conv(ops, &self.head, &st.rows[k], w, cur, 0)?;
        let mut out = vec![0f32; lo * self.out_c];
        let bytes = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), out.len() * 4) };
        w.a.read(0, bytes)?;
        Ok(out)
    }
}

impl State {
    /// Back to zeros (the reference's set_to_zero at a speech end)
    pub fn reset(&self) -> Result<()> {
        for r in &self.rows {
            r.write(0, &vec![0u8; r.len])?;
        }
        Ok(())
    }
}
