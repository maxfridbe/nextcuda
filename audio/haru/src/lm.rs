//! The language model: Qwen3.5-4B's text layers (24 Gated DeltaNet + 8 gated attention layers) on nextcuda's own Qwen3.5
//! core (crates/qwen35, the GGUF convert.rs makes: Q8_0 matrices), one position a frame. A position's input is the
//! text token's row plus the two audio streams' projections (`Projector`: fc1, RMS norm, fc2 - here, on the VibeVoice
//! ops), added to the token rows by the core (`forward_rows_add`); the normed hidden state comes back through its
//! `result_norm` tap, the text logits from its tied output.
//!
//! Two rows matter more than the rest: PAD (silent) and EOP (about to speak) are on the text stream most frames, and
//! the release computes their logits in float32 ("the PAD / EOP decision is often closer than bfloat16's resolution",
//! haru/models/haru.py's logits_fp32). Here both rows are kept exact (the checkpoint's bfloat16 values in float32): their
//! logits are computed from them on the host, and when one is the input token its Q8_0 row's error is added back with
//! the audio streams, so the two tokens the model reads and writes most are never quantized.

use std::sync::Arc;

use nextcuda_audio::{Error, Result};
use nextcuda_core::{DevBuf, Gpu};
use nextcuda_diffusion::kernels::{Dt, Nsd};
use nextcuda_gguf::{GType, Gguf};
use nextcuda_qwen35::engine::{Qwen35, Session};

use crate::ops::{nul, stage, Mat, Ops, Stage};
use crate::weights::{bytes, Config, Weights};

/// Rows a projector pass takes at most
const PROJ_ROWS: usize = 64;

/// An audio stream's projection into the model: fc1 (+ bias), RMS norm (eps 1e-6), fc2 (+ bias)
pub struct Projector {
    fc1: Mat,
    b1: DevBuf,
    norm: DevBuf,
    fc2: Mat,
    b2: DevBuf,
}

impl Projector {
    fn load(ops: &Ops, w: &Weights, p: &str) -> Result<Projector> {
        Ok(Projector {
            fc1: w.mat1(ops, &format!("{p}.fc1.weight"), false)?,
            b1: w.dev_f32(ops, &format!("{p}.fc1.bias"))?,
            norm: w.dev_f32(ops, &format!("{p}.norm.weight"))?,
            fc2: w.mat1(ops, &format!("{p}.fc2.weight"), false)?,
            b2: w.dev_f32(ops, &format!("{p}.fc2.bias"))?,
        })
    }

    fn bytes(&self) -> usize {
        self.fc1.bytes() + self.fc2.bytes()
    }
}

pub struct Lm {
    pub cfg: Config,
    pub engine: Qwen35<'static>,
    pub file: &'static Gguf,
    pub model_proj: Projector,
    pub user_proj: Projector,
    /// PAD's and EOP's rows of the token table, exact (float32), and what each one's quantized row lacks
    exact: [(u32, Vec<f32>, Vec<f32>); 2],
    /// the last position's normed hidden state (on the GPU, for the head)
    pub hidden: DevBuf,
    // the projectors' work
    lat: DevBuf,
    t1: DevBuf,
    t2: DevBuf,
    out: DevBuf,
    st: Stage,
    pub lm_bytes: u64,
    pub ty: String,
}

/// A row of a GGUF tensor [rows, k] as float32 (Q8_0, F16, F32)
fn gguf_row(g: &Gguf, name: &str, row: usize, k: usize) -> Result<Vec<f32>> {
    let t = g.tensor(name).ok_or_else(|| Error(format!("no {name}")))?;
    let rb = t.ty.bytes(k as u64).ok_or_else(|| Error(format!("{name}: a row of {k}")))? as usize;
    let mut b = vec![0u8; rb];
    g.read_into(t, (row * rb) as u64, &mut b).map_err(|e| Error(e.0))?;
    Ok(match t.ty {
        GType::F32 => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        GType::F16 => b.chunks_exact(2).map(|c| nextcuda_gguf::safetensors::f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
        GType::Q8_0 => b.chunks_exact(34).flat_map(|blk| {
            let d = nextcuda_gguf::safetensors::f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
            blk[2..].iter().map(move |q| *q as i8 as f32 * d)
        }).collect(),
        t => return Err(Error(format!("{name}: {} rows not read", t.name()))),
    })
}

impl Lm {
    /// The core from its GGUF (`lm_file`) on `gpu`, the projectors and the exact rows from the checkpoint; `ctx`: the
    /// positions a conversation holds (the attention caches are made for it)
    pub fn load(ops: &Ops, w: &Weights, cfg: &Config, lm_file: &std::path::Path, gpu: &Arc<Gpu>, ctx: usize, log: &mut dyn FnMut(String)) -> Result<Lm> {
        let f: &'static Gguf = Box::leak(Box::new(Gguf::open(lm_file).map_err(|e| Error(e.0))?));
        if f.architecture() != "qwen35" {
            return Err(Error(format!("{}: a {} file, not the converted language model", lm_file.display(), f.architecture())));
        }
        let ty = f.tensor("blk.0.ffn_up.weight").map(|t| t.ty.name().to_lowercase()).unwrap_or_default();
        let lm_bytes = f.tensors.iter().map(|t| t.bytes).sum();
        // the attention cache in half unless NS_Q35_KV says otherwise (measured against the reference: q8_0's doubled
        // the worst hidden-state error over 12 frames, 5.4e-2 -> 1.25e-1; half costs 0.4 GiB more an hour)
        nextcuda_qwen35::engine::set_kv_default("f16");
        // and the activations unquantized: the decode products' Q8_1 rounding of each row moved the silent / speak
        // decision (NS_HARU_EXACT_X=0: the faster int8 products)
        if std::env::var("NS_HARU_EXACT_X").map_or(true, |v| v != "0") {
            nextcuda_qwen35::engine::set_exact_activations();
        }
        let engine = Qwen35::load(f, std::slice::from_ref(gpu), false, (ctx, 1), log).map_err(|e| Error(e.0))?;
        let h = cfg.hidden;
        let mut exact = Vec::new();
        for id in [cfg.pad, cfg.eop] {
            let ex = w.rows_f32("lm.embed_tokens.weight", id as usize, id as usize + 1)?;
            let q = gguf_row(f, "token_embd.weight", id as usize, h)?;
            let fix: Vec<f32> = ex.iter().zip(&q).map(|(a, b)| a - b).collect();
            exact.push((id, ex, fix));
        }
        let exact: [(u32, Vec<f32>, Vec<f32>); 2] = exact.try_into().map_err(|_| Error("the exact rows".into()))?;
        let g = &ops.gpu;
        Ok(Lm {
            cfg: cfg.clone(),
            engine,
            file: f,
            model_proj: Projector::load(ops, w, "model_audio_proj")?,
            user_proj: Projector::load(ops, w, "user_audio_proj")?,
            exact,
            hidden: DevBuf::f32(g, h)?,
            lat: DevBuf::f32(g, PROJ_ROWS * cfg.latent)?,
            t1: DevBuf::f32(g, PROJ_ROWS * h)?,
            t2: DevBuf::f32(g, PROJ_ROWS * h)?,
            out: DevBuf::f32(g, PROJ_ROWS * h)?,
            st: stage(g, PROJ_ROWS * h, None)?,
            lm_bytes,
            ty,
        })
    }

    /// The memory a conversation of `ctx` positions holds (bytes): each attention layer's half key / value cache, each
    /// DeltaNet layer's recurrence and convolution rows
    pub fn state_bytes(cfg: &Config, ctx: usize) -> usize {
        let delta = cfg.full.iter().filter(|f| !**f).count();
        let full = cfg.layers - delta;
        delta * ((cfg.conv_k - 1) * cfg.conv_dim() * 4 + cfg.lin_v_heads * cfg.lin_k_dim * cfg.lin_v_dim * 4) + full * 2 * cfg.kv_heads * ctx * cfg.head_dim * 2
    }

    pub fn proj_bytes(&self) -> usize {
        self.model_proj.bytes() + self.user_proj.bytes()
    }

    /// A conversation's state of `ctx` positions
    pub fn session(&self, ctx: usize) -> Result<Session> {
        self.engine.session(ctx).map_err(|e| Error(e.0))
    }

    pub fn reset(&self, s: &mut Session) -> Result<()> {
        self.engine.reset_session(s).map_err(|e| Error(e.0))
    }

    /// `p`'s projection of scaled latents [n, latent] added into `rows` [n, hidden] (host)
    fn project(&self, ops: &Ops, nsd: &Nsd, p: &Projector, lat: &[f32], rows: &mut [f32]) -> Result<()> {
        let (l, h) = (self.cfg.latent, self.cfg.hidden);
        for (lc, rc) in lat.chunks(PROJ_ROWS * l).zip(rows.chunks_mut(PROJ_ROWS * h)) {
            let n = lc.len() / l;
            self.lat.write(0, &bytes(lc))?;
            p.fc1.apply(ops, nsd, self.lat.fp(), n, &self.st, p.b1.ptr(), self.t1.fp())?;
            nsd.rms_norm_mod(self.t1.ptr(), Dt::F32, n, h, p.norm.ptr(), 1e-6, nul(), nul(), nul(), self.t2.ptr(), Dt::F32)?;
            p.fc2.apply(ops, nsd, self.t2.fp(), n, &self.st, p.b2.ptr(), self.out.fp())?;
            let v = self.out.to_f32()?;
            for (r, x) in rc.iter_mut().zip(&v[..n * h]) {
                *r += x;
            }
        }
        Ok(())
    }

    /// The rows added to `tokens`' rows: each audio stream's projection where it has one (`model` / `user`: (first
    /// row, scaled latents [n, latent])), and PAD's / EOP's quantization error where they are the token
    pub fn extra_rows(&self, ops: &Ops, nsd: &Nsd, tokens: &[u32], model: Option<(usize, &[f32])>, user: Option<(usize, &[f32])>) -> Result<Vec<f32>> {
        let (l, h) = (self.cfg.latent, self.cfg.hidden);
        let mut rows = vec![0f32; tokens.len() * h];
        for (p, s) in [(&self.model_proj, model), (&self.user_proj, user)] {
            let Some((r0, lat)) = s else { continue };
            let n = lat.len() / l;
            if n == 0 {
                continue;
            }
            if r0 + n > tokens.len() {
                return Err(Error(format!("audio rows {r0}..{} of {}", r0 + n, tokens.len())));
            }
            self.project(ops, nsd, p, lat, &mut rows[r0 * h..(r0 + n) * h])?;
        }
        for (i, t) in tokens.iter().enumerate() {
            if let Some((_, _, fix)) = self.exact.iter().find(|e| e.0 == *t) {
                for (r, x) in rows[i * h..(i + 1) * h].iter_mut().zip(fix) {
                    *r += x;
                }
            }
        }
        Ok(rows)
    }

    /// Rows through the model at the session's position (`extra`: `extra_rows`); with `out`, the last row's normed
    /// hidden state into `hidden`, and (its text logits - PAD's and EOP's exact -, the hidden state) returned
    pub fn forward(&self, s: &mut Session, tokens: &[u32], extra: &[f32], out: bool) -> Result<Option<(Vec<f32>, Vec<f32>)>> {
        let h = self.cfg.hidden;
        let mut got: Option<Vec<f32>> = None;
        let hidden = &self.hidden;
        let mut tap = |name: &str, b: &DevBuf| -> nextcuda_core::Result<()> {
            if name == "result_norm" {
                let n = b.len / 4 / h;
                hidden.copy_within(0, b, (n - 1) * h * 4, h * 4)?;
                got = Some(hidden.to_f32()?);
            }
            Ok(())
        };
        let mut logits = self.engine.forward_rows_add(s, tokens, extra, if out { 1 } else { 0 }, &mut tap).map_err(|e| Error(e.0))?;
        if !out {
            return Ok(None);
        }
        let hv = got.ok_or_else(|| Error("the language model gave no hidden state".into()))?;
        let mut lg = logits.pop().ok_or_else(|| Error("no logits".into()))?;
        for (id, row, _) in &self.exact {
            lg[*id as usize] = row.iter().zip(&hv).map(|(a, b)| a * b).sum();
        }
        Ok(Some((lg, hv)))
    }

    /// PAD's or EOP's exact row
    pub fn exact_row(&self, id: u32) -> Option<&[f32]> {
        self.exact.iter().find(|e| e.0 == id).map(|e| e.1.as_slice())
    }

    /// A token's row as the core reads it (its quantized row, float32) - for checks
    pub fn token_row(&self, id: u32) -> Result<Vec<f32>> {
        gguf_row(self.file, "token_embd.weight", id as usize, self.cfg.hidden)
    }

    /// The rows one pass takes at most
    pub fn chunk(&self) -> usize {
        self.engine.prefill_chunk().max(1)
    }
}
