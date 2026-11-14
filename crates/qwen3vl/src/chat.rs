//! Qwen3-VL's language model writing: the text encoder of `gguf.rs` with its weights kept on the card, a key/value
//! cache, the final norm and the untied head (`output.weight`, Q6_K in Qwen's Q4_K_M file; the embedding when a file
//! ties them), so it answers as the chat model it is - what image.describe's native describer is (Qwen-Image 2.1's
//! own text encoder reading the picture).
//!
//! ```text
//!   prompt   <|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|> x N<|vision_end|>
//!            {user}<|im_end|>\n<|im_start|>assistant\n       (the chat template's, the picture first as the platform sends it)
//!   prefill  the encoder's layers over every row at once (half, the quantized product through half), the picture's
//!            vision tokens at its pads, its deepstack features after the first layers, interleaved M-RoPE (t / h / w);
//!            each layer's keys and values written into the cache as they are made
//!   decode   one row: Q8_1 x the stored blocks (the shared decode products, lent by crates/qwen35), float32 residual stream,
//!            keys and values half in the cache, one query row over it (the flash kernel), the head's logits read back
//!   stop     <|im_end|> or <|endoftext|>, or max_tokens
//! ```
//!
//! A text row after the picture sits at one position on all three axes (past the picture's larger side), so a
//! written token's rotary angles are the plain ones at the next position.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use nextcuda_core::{DevBuf, Error, Gpu, Result};
use nextcuda_diffusion::kernels::{none, Dt, Nsd};
use nextcuda_gguf::{GType, Gguf};
use nextcuda_qwen35::products;
use nextcuda_tok::Tokenizer;

use crate::gguf::GgufTextEncoder;
use crate::vision::Seen;

const HEAD: usize = 128;

fn ge(e: nextcuda_gguf::Error) -> Error {
    Error(e.0)
}

/// A matrix on the card in its stored blocks
struct W {
    buf: DevBuf,
    ty: u32,
    rows: usize,
    cols: usize,
}

struct Layer {
    attn_norm: DevBuf,
    ffn_norm: DevBuf,
    q_norm: DevBuf,
    k_norm: DevBuf,
    q: W,
    k: W,
    v: W,
    o: W,
    gate: W,
    up: W,
    down: W,
}

/// The language model on one card, ready to write
pub struct Chat {
    file: Gguf,
    /// the embedding's tensor name (llama.cpp's token_embd.weight, or transformers' in a file of its names)
    embd: String,
    pub tok: Tokenizer,
    gpu: Arc<Gpu>,
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub ffn: usize,
    pub vocab: usize,
    theta: f64,
    eps: f32,
    layers: Vec<Layer>,
    out_norm: DevBuf,
    head: W,
    /// the weights' bytes on the card
    pub weight_bytes: u64,
    pub load_seconds: f64,
    /// <|im_end|>, <|endoftext|>
    pub stops: Vec<u32>,
}

/// A conversation's keys and values: half [cap][kv_heads x 128] a layer
pub struct Seq {
    k: Vec<DevBuf>,
    v: Vec<DevBuf>,
    pub len: usize,
    pub cap: usize,
    /// the next row's position (text: the same on t, h and w)
    pub next_pos: i64,
}

/// What writing made: the tokens, the time reading the prompt and writing
pub struct Written {
    pub tokens: Vec<u32>,
    pub text: String,
    pub prompt_tokens: usize,
    pub prompt_seconds: f64,
    pub generate_seconds: f64,
    /// a stop token ended it (else max_tokens)
    pub stopped: bool,
}

impl Chat {
    /// The file's language model onto `gpu`: every matrix as stored (4.7 GiB for Qwen3-VL 8B Q4_K_M). A llama.cpp
    /// file (its names, its tokenizer in the metadata) or a file of transformers' names (`model.language_model.*`,
    /// P-H3-Image's text encoder: no metadata, so the geometry from the shapes and the tokenizer given)
    pub fn load(path: &Path, tok: Option<Tokenizer>, gpu: &Arc<Gpu>, log: &mut dyn FnMut(String)) -> Result<Chat> {
        let t0 = Instant::now();
        let file = Gguf::open(path).map_err(ge)?;
        let hf = file.tensor("model.language_model.embed_tokens.weight").is_some();
        let arch = file.architecture().to_string();
        if !hf && arch != "qwen3vl" && arch != "qwen3" {
            return Err(Error(format!("{}: a {arch} model, not Qwen3-VL", path.display())));
        }
        // a layer tensor's name by llama.cpp's role
        let lname = |l: usize, role: &str| -> String {
            if !hf {
                return format!("blk.{l}.{role}.weight");
            }
            let t = match role {
                "attn_norm" => "input_layernorm",
                "ffn_norm" => "post_attention_layernorm",
                "attn_q_norm" => "self_attn.q_norm",
                "attn_k_norm" => "self_attn.k_norm",
                "attn_q" => "self_attn.q_proj",
                "attn_k" => "self_attn.k_proj",
                "attn_v" => "self_attn.v_proj",
                "attn_output" => "self_attn.o_proj",
                "ffn_gate" => "mlp.gate_proj",
                "ffn_up" => "mlp.up_proj",
                _ => "mlp.down_proj",
            };
            format!("model.language_model.layers.{l}.{t}.weight")
        };
        let (embd, out_norm_name, head_name) = if hf {
            ("model.language_model.embed_tokens.weight", "model.language_model.norm.weight", "lm_head.weight")
        } else {
            ("token_embd.weight", "output_norm.weight", "output.weight")
        };
        let shape = |n: &str| file.tensor(n).map(|t| t.shape.clone()).ok_or_else(|| Error(format!("{}: no tensor {n}", path.display())));
        let geti = |k: &str| -> Result<usize> {
            file.arch_meta(k).and_then(|v| v.as_u64()).map(|v| v as usize).ok_or_else(|| Error(format!("{}: no {arch}.{k}", path.display())))
        };
        let (n_layers, hidden, heads, kv_heads, ffn) = if hf {
            let e = shape(embd)?;
            (
                (0..).take_while(|l| file.tensor(&lname(*l, "attn_norm")).is_some()).count(),
                e[1] as usize,
                shape(&lname(0, "attn_q"))?[0] as usize / HEAD,
                shape(&lname(0, "attn_k"))?[0] as usize / HEAD,
                shape(&lname(0, "ffn_gate"))?[0] as usize,
            )
        } else {
            (geti("block_count")?, geti("embedding_length")?, geti("attention.head_count")?, geti("attention.head_count_kv")?, geti("feed_forward_length")?)
        };
        // Qwen3-VL's: theta 5e6, eps 1e-6 (a file of transformers' names carries no metadata)
        let theta = file.arch_meta("rope.freq_base").and_then(|v| v.as_f64()).unwrap_or(5_000_000.0);
        let eps = file.arch_meta("attention.layer_norm_rms_epsilon").and_then(|v| v.as_f64()).unwrap_or(1e-6) as f32;
        let tok = match tok {
            Some(t) => t,
            None => Tokenizer::from_gguf(&file).map_err(|e| Error(format!("{}: its tokenizer: {e}", path.display())))?,
        };
        let vocab = shape(embd)?[0] as usize;
        let mut bytes = 0u64;
        let matrix = |name: &str, bytes: &mut u64| -> Result<W> {
            let t = file.tensor(name).ok_or_else(|| Error(format!("{}: no tensor {name}", path.display())))?;
            let ty = t.ty.code();
            if !products::supported(ty) {
                return Err(Error(format!("{name}: {} weights have no decode product", t.ty.name())));
            }
            let raw = file.read(t).map_err(ge)?;
            let buf = DevBuf::new(gpu, raw.len())?;
            buf.write(0, &raw)?;
            *bytes += raw.len() as u64;
            // the crate's shape is (rows, columns)
            Ok(W { buf, ty, rows: t.shape[0] as usize, cols: t.shape[1] as usize })
        };
        let vector = |name: &str| -> Result<DevBuf> {
            let t = file.tensor(name).ok_or_else(|| Error(format!("{}: no tensor {name}", path.display())))?;
            let b = file.read(t).map_err(ge)?;
            let v: Vec<f32> = match t.ty {
                GType::F32 => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
                GType::F16 => b.chunks_exact(2).map(|c| nextcuda_gguf::safetensors::f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
                GType::BF16 => b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
                other => return Err(Error(format!("{name}: a {} vector", other.name()))),
            };
            DevBuf::from_f32(gpu, &v)
        };
        let mut layers = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let p = |r: &str| lname(l, r);
            layers.push(Layer {
                attn_norm: vector(&p("attn_norm"))?,
                ffn_norm: vector(&p("ffn_norm"))?,
                q_norm: vector(&p("attn_q_norm"))?,
                k_norm: vector(&p("attn_k_norm"))?,
                q: matrix(&p("attn_q"), &mut bytes)?,
                k: matrix(&p("attn_k"), &mut bytes)?,
                v: matrix(&p("attn_v"), &mut bytes)?,
                o: matrix(&p("attn_output"), &mut bytes)?,
                gate: matrix(&p("ffn_gate"), &mut bytes)?,
                up: matrix(&p("ffn_up"), &mut bytes)?,
                down: matrix(&p("ffn_down"), &mut bytes)?,
            });
        }
        // the head: its own matrix, or the embedding's when the file ties them (Qwen3-VL 4B does)
        let head = matrix(if file.tensor(head_name).is_some() { head_name } else { embd }, &mut bytes)?;
        let out_norm = vector(out_norm_name)?;
        let stops = ["<|im_end|>", "<|endoftext|>"].iter().filter_map(|t| tok.id(t)).collect();
        let load_seconds = t0.elapsed().as_secs_f64();
        log(format!("qwen3vl chat: {n_layers} layers, {hidden} wide, {heads} / {kv_heads} heads, {vocab} tokens; {:.2} GiB of weights on {} in {load_seconds:.1} s",
                    bytes as f64 / (1u64 << 30) as f64, gpu.name));
        Ok(Chat { file, embd: embd.to_string(), tok, gpu: gpu.clone(), hidden, heads, kv_heads, ffn, vocab, theta, eps, layers, out_norm, head, weight_bytes: bytes,
                  load_seconds, stops })
    }

    /// An empty conversation of at most `cap` rows
    pub fn seq(&self, cap: usize) -> Result<Seq> {
        let kvw = self.kv_heads * HEAD;
        let mut k = Vec::with_capacity(self.layers.len());
        let mut v = Vec::with_capacity(self.layers.len());
        for _ in 0..self.layers.len() {
            k.push(DevBuf::new(&self.gpu, cap * kvw * 2)?);
            v.push(DevBuf::new(&self.gpu, cap * kvw * 2)?);
        }
        Ok(Seq { k, v, len: 0, cap, next_pos: 0 })
    }

    /// The bytes a conversation of `cap` rows holds on the card
    pub fn seq_bytes(&self, cap: usize) -> u64 {
        (self.layers.len() * 2 * cap * self.kv_heads * HEAD * 2) as u64
    }

    /// The chat template's prompt: the system turn, then the user's picture (its `n` vision tokens) and words, then the
    /// assistant's turn opened
    pub fn prompt_ids(&self, system: &str, user: &str, n: usize) -> Result<Vec<u32>> {
        let text = format!("<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{user}<|im_end|>\n<|im_start|>assistant\n");
        let pad = self.tok.id("<|image_pad|>").ok_or_else(|| Error("the tokenizer has no <|image_pad|>".into()))?;
        let mut ids = Vec::new();
        for id in self.tok.encode(&text) {
            if id == pad {
                ids.extend(std::iter::repeat_n(pad, n));
            } else {
                ids.push(id);
            }
        }
        Ok(ids)
    }

    /// The (cos, sin) table of rows at positions `pos`: pair i turns with t, h or w as Qwen3-VL interleaves them (24 /
    /// 20 / 20 pairs: h and w on every third pair from the second, t on the rest)
    fn rope(&self, pos: &[[i64; 3]]) -> Vec<f32> {
        let mut cs = vec![0f32; pos.len() * HEAD];
        for (t, p) in pos.iter().enumerate() {
            for i in 0..HEAD / 2 {
                let axis = if i < 60 && i % 3 == 1 { 1 } else if i < 60 && i % 3 == 2 { 2 } else { 0 };
                let a = p[axis] as f64 / self.theta.powf(2.0 * i as f64 / HEAD as f64);
                cs[t * HEAD + 2 * i] = a.cos() as f32;
                cs[t * HEAD + 2 * i + 1] = a.sin() as f32;
            }
        }
        cs
    }

    /// Embedding rows of `ids` (read from the file, made float32 on the card) into `x` [ids][hidden]
    fn embed_into(&self, nsd: &Nsd, ids: &[u32], x: &DevBuf) -> Result<()> {
        let t = self.file.tensor(&self.embd).ok_or_else(|| Error(format!("no {}", self.embd)))?;
        let d = self.hidden;
        let rb = t.ty.bytes(d as u64).ok_or_else(|| Error(format!("{}: a type with no row size", self.embd)))? as usize;
        let mut raw = vec![0u8; ids.len() * rb];
        for (i, id) in ids.iter().enumerate() {
            if *id as usize >= self.vocab {
                return Err(Error(format!("token {id} is outside the {} of the embedding", self.vocab)));
            }
            self.file.read_into(t, *id as u64 * rb as u64, &mut raw[i * rb..(i + 1) * rb]).map_err(ge)?;
        }
        let src = DevBuf::new(&self.gpu, raw.len())?;
        src.write(0, &raw)?;
        nsd.dequant(src.ptr(), t.ty.code(), ids.len() * d, x.ptr(), Dt::F32)?;
        nsd.wait()
    }

    /// The prompt's pass (the conversation empty): `ids` with the pictures `seen` at their pads; the keys and values
    /// into `seq`; the last row's logits
    pub fn prefill(&self, nsd: &Nsd, seq: &mut Seq, ids: &[u32], seen: &[&Seen], stop: &mut dyn FnMut() -> bool) -> Result<Vec<f32>> {
        let s = ids.len();
        if seq.len != 0 {
            return Err(Error("prefill: the conversation is not empty".into()));
        }
        if s == 0 || s >= seq.cap {
            return Err(Error(format!("a prompt of {s} rows in a conversation of {}", seq.cap)));
        }
        let (d, h, kv, f) = (self.hidden, self.heads, self.kv_heads, self.ffn);
        let (qw, kvw) = (h * HEAD, kv * HEAD);
        let gpu = &self.gpu;
        let x = DevBuf::f32(gpu, s * d)?;
        self.embed_into(nsd, ids, &x)?;
        let pad = self.tok.id("<|image_pad|>").unwrap_or(u32::MAX);
        let mut runs = Vec::new();
        let mut i = 0;
        while i < s {
            if ids[i] == pad {
                let st = i;
                while i < s && ids[i] == pad {
                    i += 1;
                }
                runs.push((st, i - st));
            } else {
                i += 1;
            }
        }
        if runs.len() != seen.len() || runs.iter().zip(seen).any(|((_, len), v)| *len != v.n) {
            return Err(Error(format!("the prompt holds {} picture runs; {} pictures were seen", runs.len(), seen.len())));
        }
        for ((st, len), v) in runs.iter().zip(seen) {
            x.copy_within(st * d * 4, &v.tokens, 0, len * d * 4)?;
        }
        let pos: Vec<[i64; 3]> = if seen.is_empty() {
            (0..s as i64).map(|p| [p, p, p]).collect()
        } else {
            GgufTextEncoder::positions(ids, pad, &seen.iter().map(|v| v.grid).collect::<Vec<_>>())?
        };
        let cs = DevBuf::from_f32(gpu, &self.rope(&pos))?;
        let hb = DevBuf::new(gpu, s * d * 2)?;
        let q = DevBuf::new(gpu, s * qw * 2)?;
        let att = DevBuf::new(gpu, s * qw * 2)?;
        let o = DevBuf::f32(gpu, s * d)?;
        let gu = DevBuf::new(gpu, s * 2 * f * 2)?;
        let act = DevBuf::new(gpu, s * f * 2)?;
        let half_at = |b: &DevBuf, at: usize| -> *mut std::ffi::c_void { b.ptr().cast::<u16>().wrapping_add(at).cast() };
        for (l, ly) in self.layers.iter().enumerate() {
            if stop() {
                return Err(Error("cancelled".into()));
            }
            let (kc, vc) = (&seq.k[l], &seq.v[l]);
            nsd.rms_norm_mod(x.ptr(), Dt::F32, s, d, ly.attn_norm.ptr(), self.eps, none(), none(), none(), hb.ptr(), Dt::F16)?;
            nsd.linear_q(hb.ptr(), Dt::F16, s, d, ly.q.buf.ptr(), ly.q.ty, qw, none(), q.ptr(), Dt::F16, qw)?;
            // keys and values straight into the cache's first rows
            nsd.linear_q(hb.ptr(), Dt::F16, s, d, ly.k.buf.ptr(), ly.k.ty, kvw, none(), kc.ptr(), Dt::F16, kvw)?;
            nsd.linear_q(hb.ptr(), Dt::F16, s, d, ly.v.buf.ptr(), ly.v.ty, kvw, none(), vc.ptr(), Dt::F16, kvw)?;
            nsd.rms_rope(q.ptr(), Dt::F16, s, h, HEAD, qw, ly.q_norm.ptr(), self.eps, cs.ptr(), HEAD)?;
            nsd.rms_rope(kc.ptr(), Dt::F16, s, kv, HEAD, kvw, ly.k_norm.ptr(), self.eps, cs.ptr(), HEAD)?;
            nsd.attention_causal(q.ptr(), kc.ptr(), vc.ptr(), Dt::F16, s, h, kv, HEAD, qw, kvw, att.ptr())?;
            nsd.linear_q(att.ptr(), Dt::F16, s, qw, ly.o.buf.ptr(), ly.o.ty, d, none(), o.ptr(), Dt::F32, 0)?;
            nsd.gate_add(x.ptr(), Dt::F32, s, d, o.ptr(), Dt::F32, none(), none())?;
            nsd.rms_norm_mod(x.ptr(), Dt::F32, s, d, ly.ffn_norm.ptr(), self.eps, none(), none(), none(), hb.ptr(), Dt::F16)?;
            nsd.linear_q(hb.ptr(), Dt::F16, s, d, ly.gate.buf.ptr(), ly.gate.ty, f, none(), gu.ptr(), Dt::F16, 2 * f)?;
            nsd.linear_q(hb.ptr(), Dt::F16, s, d, ly.up.buf.ptr(), ly.up.ty, f, none(), half_at(&gu, f), Dt::F16, 2 * f)?;
            nsd.swiglu(gu.ptr(), Dt::F16, s, f, act.ptr(), Dt::F16)?;
            nsd.linear_q(act.ptr(), Dt::F16, s, f, ly.down.buf.ptr(), ly.down.ty, d, none(), o.ptr(), Dt::F32, 0)?;
            nsd.gate_add(x.ptr(), Dt::F32, s, d, o.ptr(), Dt::F32, none(), none())?;
            for ((st, len), v) in runs.iter().zip(seen) {
                if let Some(ds) = v.deep.get(l) {
                    nsd.gate_add(x.fp().wrapping_add(st * d).cast(), Dt::F32, *len, d, ds.ptr(), Dt::F32, none(), none())?;
                }
            }
            // a layer at a time: the GPU's queue stays short, so a cancel is seen within a layer
            nsd.wait()?;
        }
        seq.len = s;
        seq.next_pos = pos.iter().map(|p| p[0].max(p[1]).max(p[2])).max().unwrap_or(-1) + 1;
        let last = x.view((s - 1) * d * 4, d * 4)?;
        self.logits(nsd, &last)
    }

    /// The head over one residual row (float32 [hidden]): the final norm, then the logits, read back
    fn logits(&self, nsd: &Nsd, x: &DevBuf) -> Result<Vec<f32>> {
        let d = self.hidden;
        let h = DevBuf::f32(&self.gpu, d)?;
        nsd.rms_norm_mod(x.ptr(), Dt::F32, 1, d, self.out_norm.ptr(), self.eps, none(), none(), none(), h.ptr(), Dt::F32)?;
        let xq = DevBuf::new(&self.gpu, products::q8_1_bytes(d, 1)?)?;
        products::quantize(&self.gpu, h.fp(), 1, d, xq.ptr())?;
        let logits = DevBuf::f32(&self.gpu, self.head.rows)?;
        products::mmvq(&self.gpu, self.head.ty, self.head.buf.ptr(), xq.ptr(), logits.fp(), self.head.cols, self.head.rows, 1)?;
        nsd.wait()?;
        logits.to_f32()
    }

    /// `y` [w.rows] = x [w.cols] W^T for one row, x float32 (made Q8_1 into `xq` first)
    fn mv(&self, w: &W, x: &DevBuf, xq: &DevBuf, y: *mut f32) -> Result<()> {
        products::quantize(&self.gpu, x.fp(), 1, w.cols, xq.ptr())?;
        products::mmvq(&self.gpu, w.ty, w.buf.ptr(), xq.ptr(), y, w.cols, w.rows, 1)
    }

    /// One written token's pass: its row appended to `seq`; its logits
    pub fn step(&self, nsd: &Nsd, seq: &mut Seq, token: u32, work: &StepWork) -> Result<Vec<f32>> {
        if seq.len + 1 > seq.cap {
            return Err(Error(format!("the conversation holds {} rows", seq.cap)));
        }
        let (d, h, kv, f) = (self.hidden, self.heads, self.kv_heads, self.ffn);
        let (qw, kvw) = (h * HEAD, kv * HEAD);
        let StepWork { x, hn, qkv, qh, atth, attf, o, gu, act, xq, cs } = work;
        self.embed_into(nsd, &[token], x)?;
        let p = seq.next_pos;
        cs.write(0, &self.rope(&[[p, p, p]]).iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
        let at = |b: &DevBuf, i: usize| -> *mut f32 { b.fp().wrapping_add(i) };
        let half_row = |b: &DevBuf, row: usize| -> *mut std::ffi::c_void { b.ptr().cast::<u16>().wrapping_add(row * kvw).cast() };
        let n = seq.len + 1;
        for (l, ly) in self.layers.iter().enumerate() {
            nsd.rms_norm_mod(x.ptr(), Dt::F32, 1, d, ly.attn_norm.ptr(), self.eps, none(), none(), none(), hn.ptr(), Dt::F32)?;
            products::quantize(&self.gpu, hn.fp(), 1, d, xq.ptr())?;
            products::mmvq(&self.gpu, ly.q.ty, ly.q.buf.ptr(), xq.ptr(), at(qkv, 0), d, qw, 1)?;
            products::mmvq(&self.gpu, ly.k.ty, ly.k.buf.ptr(), xq.ptr(), at(qkv, qw), d, kvw, 1)?;
            products::mmvq(&self.gpu, ly.v.ty, ly.v.buf.ptr(), xq.ptr(), at(qkv, qw + kvw), d, kvw, 1)?;
            nsd.rms_rope(qkv.ptr(), Dt::F32, 1, h, HEAD, qw, ly.q_norm.ptr(), self.eps, cs.ptr(), HEAD)?;
            nsd.rms_rope(at(qkv, qw).cast(), Dt::F32, 1, kv, HEAD, kvw, ly.k_norm.ptr(), self.eps, cs.ptr(), HEAD)?;
            // q to half; k and v to half into the cache's next row
            nsd.dequant(qkv.ptr(), 0, qw, qh.ptr(), Dt::F16)?;
            nsd.dequant(at(qkv, qw).cast(), 0, kvw, half_row(&seq.k[l], seq.len), Dt::F16)?;
            nsd.dequant(at(qkv, qw + kvw).cast(), 0, kvw, half_row(&seq.v[l], seq.len), Dt::F16)?;
            nsd.flash_attention(qh.ptr(), 1, qw, seq.k[l].ptr(), seq.v[l].ptr(), n, kvw, h, kv, atth.ptr())?;
            nsd.dequant(atth.ptr(), 1, qw, attf.ptr(), Dt::F32)?;
            self.mv(&ly.o, attf, xq, o.fp())?;
            nsd.gate_add(x.ptr(), Dt::F32, 1, d, o.ptr(), Dt::F32, none(), none())?;
            nsd.rms_norm_mod(x.ptr(), Dt::F32, 1, d, ly.ffn_norm.ptr(), self.eps, none(), none(), none(), hn.ptr(), Dt::F32)?;
            products::quantize(&self.gpu, hn.fp(), 1, d, xq.ptr())?;
            products::mmvq(&self.gpu, ly.gate.ty, ly.gate.buf.ptr(), xq.ptr(), at(gu, 0), d, f, 1)?;
            products::mmvq(&self.gpu, ly.up.ty, ly.up.buf.ptr(), xq.ptr(), at(gu, f), d, f, 1)?;
            nsd.swiglu(gu.ptr(), Dt::F32, 1, f, act.ptr(), Dt::F32)?;
            self.mv(&ly.down, act, xq, o.fp())?;
            nsd.gate_add(x.ptr(), Dt::F32, 1, d, o.ptr(), Dt::F32, none(), none())?;
        }
        seq.len += 1;
        seq.next_pos += 1;
        self.logits(nsd, x)
    }

    /// A written token's buffers, made once a conversation
    pub fn step_work(&self) -> Result<StepWork> {
        let g = &self.gpu;
        let (d, qw, kvw, f) = (self.hidden, self.heads * HEAD, self.kv_heads * HEAD, self.ffn);
        Ok(StepWork {
            x: DevBuf::f32(g, d)?,
            hn: DevBuf::f32(g, d)?,
            qkv: DevBuf::f32(g, qw + 2 * kvw)?,
            qh: DevBuf::new(g, qw * 2)?,
            atth: DevBuf::new(g, qw * 2)?,
            attf: DevBuf::f32(g, qw)?,
            o: DevBuf::f32(g, d)?,
            gu: DevBuf::f32(g, 2 * f)?,
            act: DevBuf::f32(g, f)?,
            xq: DevBuf::new(g, products::q8_1_bytes(d.max(f).max(qw), 1)?)?,
            cs: DevBuf::f32(g, HEAD)?,
        })
    }

    /// The prompt read, then tokens written until a stop token or `max_tokens`: `pick` chooses each from its logits,
    /// `each(tokens so far)` is told after each and ends the writing early with true (a cancel)
    #[allow(clippy::too_many_arguments)]
    pub fn write(&self, nsd: &Nsd, ids: &[u32], seen: &[&Seen], max_tokens: usize, pick: &mut dyn FnMut(&[f32]) -> u32,
                 each: &mut dyn FnMut(usize) -> bool) -> Result<Written> {
        let t0 = Instant::now();
        let mut seq = self.seq(ids.len() + max_tokens + 1)?;
        let mut logits = self.prefill(nsd, &mut seq, ids, seen, &mut || each(0))?;
        let prompt_seconds = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let work = self.step_work()?;
        let mut out = Vec::new();
        let mut stopped = false;
        while out.len() < max_tokens {
            let t = pick(&logits);
            if self.stops.contains(&t) {
                stopped = true;
                break;
            }
            out.push(t);
            if each(out.len()) {
                return Err(Error("cancelled".into()));
            }
            if out.len() == max_tokens {
                break;
            }
            logits = self.step(nsd, &mut seq, t, &work)?;
        }
        Ok(Written { text: self.tok.decode(&out), tokens: out, prompt_tokens: ids.len(), prompt_seconds, generate_seconds: t1.elapsed().as_secs_f64(), stopped })
    }
}

/// One written token's buffers (Chat::step_work)
pub struct StepWork {
    x: DevBuf,
    hn: DevBuf,
    qkv: DevBuf,
    qh: DevBuf,
    atth: DevBuf,
    attf: DevBuf,
    o: DevBuf,
    gu: DevBuf,
    act: DevBuf,
    xq: DevBuf,
    cs: DevBuf,
}

/// A sampler: greedy at temperature 0, else softmax at the temperature drawn with a seeded generator (splitmix64)
pub struct Sampler {
    pub temperature: f32,
    state: u64,
}

impl Sampler {
    pub fn new(temperature: f32, seed: u64) -> Sampler {
        Sampler { temperature, state: seed ^ 0x9E37_79B9_7F4A_7C15 }
    }

    fn next(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn pick(&mut self, logits: &[f32]) -> u32 {
        let best = logits.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |b, (i, v)| if *v > b.1 { (i, *v) } else { b });
        if self.temperature <= 0.0 {
            return best.0 as u32;
        }
        let t = self.temperature as f64;
        let w: Vec<f64> = logits.iter().map(|v| ((*v - best.1) as f64 / t).exp()).collect();
        let total: f64 = w.iter().sum();
        let mut r = self.next() * total;
        for (i, x) in w.iter().enumerate() {
            r -= x;
            if r <= 0.0 {
                return i as u32;
            }
        }
        best.0 as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_takes_the_largest_and_a_seed_repeats_a_draw() {
        let l = [0.1f32, 2.0, -1.0, 1.9];
        assert_eq!(Sampler::new(0.0, 1).pick(&l), 1);
        let draws = |seed| (0..20).map(|_| 0).scan(Sampler::new(1.0, seed), |s, _| Some(s.pick(&l))).collect::<Vec<_>>();
        assert_eq!(draws(7), draws(7));
        assert!(draws(7).iter().all(|t| *t < 4));
    }
}
