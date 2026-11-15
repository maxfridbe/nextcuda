//! The same text encoder from a llama.cpp GGUF (Qwen3-VL's language model as `Qwen3VL-8B-Instruct-Q4_K_M.gguf`
//! holds it): Q4_K / Q6_K matrices, the tokenizer in the file's metadata. A layer's matrices go to the card as they
//! are stored and are made half a slice of rows at a time by the quantized linear, so an 8B encoder takes the
//! card's memory for two layers (about 260 MiB) and not for five gigabytes - the 12 GB cards hold the denoiser beside
//! it. The layers are kept in pinned host memory after their first read, and layer l + 1 is copied on the card's copy
//! lane while layer l runs (one copy a layer): the PCIe link is the only wait (0.27 s for a prompt on the 5070's x16,
//! 2.7 s on the 3060's x1, the link's 1.7 GB/s).
//!
//! One layer, as in `lib.rs` (the reference's): activations float32 between the layers, half inside them.
//!
//! ```text
//!   h = rmsnorm(x)                  q k v = h [Wq|Wk|Wv]   (32 query heads, 8 key/value heads, 128 features each)
//!   q, k: per-head rmsnorm + rotary positions (theta 5e6, pairs (i, 64 + i))
//!   x += attention(q, k, v) Wo      causal; 4 query heads share each key/value head
//!   h = rmsnorm(x)
//!   x += (silu(h Wgate) * (h Wup)) Wdown
//! ```
//!
//! The result is the last layer's hidden state for every token, before the final norm (as the image pipelines take it).

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use nextcuda_core::{DevBuf, Error, Gpu, HostBuf, Result};
use nextcuda_diffusion::kernels::{none, Dt, Nsd};
use nextcuda_gguf::{GType, Gguf};
use nextcuda_tok::Tokenizer;

use crate::vision::{self, Seen};

const HEAD: usize = 128;

fn ge(e: nextcuda_gguf::Error) -> Error {
    Error(e.0)
}

pub struct GgufTextEncoder {
    file: Gguf,
    pub tok: Tokenizer,
    gpu: Arc<Gpu>,
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub ffn: usize,
    layers: usize,
    theta: f64,
    eps: f32,
    /// Qwen3's per-head norms of q and k (Qwen2.5 has none, and biases on q k v instead)
    qk_norm: bool,
    qkv_bias: bool,
    /// each layer's matrices once read, in pinned host memory (`NC_TE_HOST_CACHE=0`: read from the file each time): the
    /// encoder is streamed to the card for every prompt, a layer copied while the one before it runs
    cache: Option<Mutex<HashMap<usize, Arc<HostLayer>>>>,
    /// the norms' and biases' float32 vectors, kept on the card (a few MiB)
    vecs: Mutex<HashMap<String, Arc<DevBuf>>>,
}

/// A layer's matrices as stored, one after another in one block of host memory (pinned when the driver allows):
/// one copy a layer to the card
struct HostLayer {
    buf: HostBuf,
    /// (offset, GGUF type, rows) of attn_q, attn_k, attn_v, attn_output, ffn_gate, ffn_up, ffn_down
    mats: Vec<(usize, u32, usize)>,
}

const MATS: [&str; 7] = ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"];

impl GgufTextEncoder {
    /// Open the file: the tokenizer and the geometry from its metadata; the weights are read per layer at encode time
    pub fn load(path: &Path, gpu: &Arc<Gpu>, log: &mut dyn FnMut(String)) -> Result<GgufTextEncoder> {
        let t0 = std::time::Instant::now();
        let file = Gguf::open(path).map_err(ge)?;
        let arch = file.architecture().to_string();
        // Qwen2.5-VL's language model too (Qwen-Image 2512's encoder): biases on q k v, no per-head norms
        if arch != "qwen3vl" && arch != "qwen3" && arch != "qwen2vl" {
            return Err(Error(format!("{}: a {arch} model, not Qwen3-VL or Qwen2.5-VL", path.display())));
        }
        let geti = |k: &str| -> Result<usize> {
            file.arch_meta(k).and_then(|v| v.as_u64()).map(|v| v as usize).ok_or_else(|| Error(format!("{}: no {arch}.{k}", path.display())))
        };
        let getf = |k: &str, d: f64| file.arch_meta(k).and_then(|v| v.as_f64()).unwrap_or(d);
        let layers = geti("block_count")?;
        let hidden = geti("embedding_length")?;
        let heads = geti("attention.head_count")?;
        let kv_heads = geti("attention.head_count_kv")?;
        let ffn = geti("feed_forward_length")?;
        let theta = getf("rope.freq_base", 5_000_000.0);
        let eps = getf("attention.layer_norm_rms_epsilon", 1e-6) as f32;
        let tok = Tokenizer::from_gguf(&file).map_err(Error)?;
        let qk_norm = file.tensor("blk.0.attn_q_norm.weight").is_some();
        let qkv_bias = file.tensor("blk.0.attn_q.bias").is_some();
        log(format!("{arch} text encoder (GGUF): {layers} layers, {hidden} wide, {heads} / {kv_heads} heads, {ffn} ffn; {:.1} s", t0.elapsed().as_secs_f64()));
        Ok(GgufTextEncoder {
            file,
            tok,
            gpu: gpu.clone(),
            hidden,
            heads,
            kv_heads,
            ffn,
            layers,
            theta,
            eps,
            qk_norm,
            qkv_bias,
            cache: (std::env::var("NC_TE_HOST_CACHE").map_or(true, |v| v != "0")).then(|| Mutex::new(HashMap::new())),
            vecs: Mutex::new(HashMap::new()),
        })
    }

    /// `text` tokenized as it is (the caller builds the chat template)
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        self.tok.encode(text)
    }

    fn tensor(&self, name: &str) -> Result<&nextcuda_gguf::Tensor> {
        self.file.tensor(name).ok_or_else(|| Error(format!("{}: no tensor {name}", self.file.paths[0].display())))
    }

    /// Layer `l`'s matrices in host memory: from the cache after the first read
    fn host_layer(&self, l: usize) -> Result<Arc<HostLayer>> {
        if let Some(c) = &self.cache {
            if let Some(h) = c.lock().unwrap().get(&l) {
                return Ok(h.clone());
            }
        }
        let mut layout = Vec::new();
        let mut len = 0usize;
        for m in MATS {
            let name = format!("blk.{l}.{m}.weight");
            let t = self.tensor(&name)?;
            let code = t.ty.code();
            if ![0, 1, 8, 12, 13, 14, 30].contains(&code) {
                return Err(Error(format!("{name}: {} weights are not read yet", t.ty.name())));
            }
            let n = t.ty.bytes(t.elements()).ok_or_else(|| Error(format!("{name}: a type with no size")))? as usize;
            layout.push((t, len, n, code, t.shape[0] as usize));
            len += n.div_ceil(256) * 256;
        }
        // pinned memory, so the copies run beside the encoder's work; pageable when the driver will not pin it
        let mut buf = HostBuf::new(&self.gpu, len).unwrap_or_else(|_| HostBuf::pageable(&self.gpu, len));
        let mut mats = Vec::new();
        for (t, off, n, code, rows) in layout {
            self.file.read_into(t, 0, &mut buf.as_mut_slice()[off..off + n]).map_err(ge)?;
            mats.push((off, code, rows));
        }
        let h = Arc::new(HostLayer { buf, mats });
        if let Some(c) = &self.cache {
            c.lock().unwrap().insert(l, h.clone());
        }
        Ok(h)
    }

    /// A float32 vector of the file (a norm's weight, a bias) on the card, kept there after the first use
    fn vector(&self, name: &str) -> Result<Arc<DevBuf>> {
        if let Some(v) = self.vecs.lock().unwrap().get(name) {
            return Ok(v.clone());
        }
        let t = self.tensor(name)?;
        let b = self.file.read(t).map_err(ge)?;
        let v: Vec<f32> = match t.ty {
            GType::F32 => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            GType::F16 => b.chunks_exact(2).map(|c| nextcuda_gguf::safetensors::f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
            GType::BF16 => b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
            other => return Err(Error(format!("{name}: a {} vector", other.name()))),
        };
        let d = Arc::new(DevBuf::from_f32(&self.gpu, &v)?);
        self.vecs.lock().unwrap().insert(name.to_string(), d.clone());
        Ok(d)
    }

    /// The embedding rows of `ids`, float32 [S, hidden] on the card: the rows' bytes read from the file, made float32
    /// by the dequant kernel
    fn embed(&self, nsd: &Nsd, ids: &[u32]) -> Result<DevBuf> {
        let t = self.tensor("token_embd.weight")?;
        let d = self.hidden;
        let rb = t.ty.bytes(d as u64).ok_or_else(|| Error("token_embd.weight: a type with no row size".into()))? as usize;
        let vocab = t.shape[0] as usize;
        let mut raw = vec![0u8; ids.len() * rb];
        for (i, id) in ids.iter().enumerate() {
            if *id as usize >= vocab {
                return Err(Error(format!("token {id} is outside the {vocab} of the embedding")));
            }
            self.file.read_into(t, *id as u64 * rb as u64, &mut raw[i * rb..(i + 1) * rb]).map_err(ge)?;
        }
        let src = DevBuf::new(&self.gpu, raw.len())?;
        src.write(0, &raw)?;
        let out = DevBuf::f32(&self.gpu, ids.len() * d)?;
        nsd.dequant(src.ptr(), t.ty.code(), ids.len() * d, out.ptr(), Dt::F32)?;
        nsd.wait()?;
        Ok(out)
    }

    /// The hidden state after the last layer for every token of `ids`: float32 [S, hidden] on the card
    pub fn encode_ids(&self, nsd: &Nsd, ids: &[u32]) -> Result<DevBuf> {
        self.encode_layers(nsd, ids, self.layers)
    }

    /// The hidden state after the last layer and the final norm (`output_norm`: what transformers' hidden_states[-1]
    /// is, and Qwen-Image's pipeline takes): float32 [S, hidden] on the card
    pub fn encode_normed(&self, nsd: &Nsd, ids: &[u32]) -> Result<DevBuf> {
        let x = self.encode_ids(nsd, ids)?;
        let w = self.vector("output_norm.weight")?;
        nsd.rms_norm_mod(x.ptr(), Dt::F32, ids.len(), self.hidden, w.ptr(), self.eps, none(), none(), none(), x.ptr(), Dt::F32)?;
        nsd.wait()?;
        Ok(x)
    }

    /// The hidden state after the first `n` layers
    pub fn encode_layers(&self, nsd: &Nsd, ids: &[u32], n: usize) -> Result<DevBuf> {
        self.encode_seen(nsd, ids, &[], n)
    }

    /// Each token's (time, height, width) position (transformers' get_rope_index): text tokens advance one on all
    /// three axes; a picture's run of `<|image_pad|>` tokens (its merged grid, row by row) sits at the run's start
    /// plus its row and column, and the next text starts past the larger side
    pub fn positions(ids: &[u32], pad: u32, grids: &[(usize, usize)]) -> Result<Vec<[i64; 3]>> {
        let mut out = Vec::with_capacity(ids.len());
        let (mut p, mut i, mut g) = (0i64, 0usize, 0usize);
        while i < ids.len() {
            if ids[i] != pad {
                out.push([p, p, p]);
                p += 1;
                i += 1;
                continue;
            }
            let (gh, gw) = *grids.get(g).ok_or_else(|| Error("more pictures in the prompt than were given".into()))?;
            let (lh, lw) = (gh / vision::MERGE, gw / vision::MERGE);
            for r in 0..lh {
                for c in 0..lw {
                    if ids.get(i) != Some(&pad) {
                        return Err(Error(format!("picture {g}: its run of image tokens is shorter than its {lh} x {lw} grid")));
                    }
                    out.push([p, p + r as i64, p + c as i64]);
                    i += 1;
                }
            }
            p += lh.max(lw) as i64;
            g += 1;
        }
        Ok(out)
    }

    /// The hidden state after `n` layers for `ids` with pictures in it: `seen` (the vision tower's, in the order of
    /// their `<|image_pad|>` runs) at their tokens, Qwen3-VL's interleaved M-RoPE (24 / 20 / 20 pairs: t, then h and
    /// w on every third pair from the second), the deepstack features added after the first layers
    pub fn encode_seen(&self, nsd: &Nsd, ids: &[u32], seen: &[&Seen], n: usize) -> Result<DevBuf> {
        let s = ids.len();
        let (d, h, kv, f) = (self.hidden, self.heads, self.kv_heads, self.ffn);
        let qkv_w = (h + 2 * kv) * HEAD;
        let gpu = &self.gpu;
        let x = self.embed(nsd, ids)?;
        let pad = self.tok.id("<|image_pad|>").unwrap_or(u32::MAX);
        // each picture's rows: the start of its run of image tokens
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
            return Err(Error(format!("the prompt holds {} picture runs {:?}; {} pictures were seen", runs.len(), runs.iter().map(|r| r.1).collect::<Vec<_>>(),
                                     seen.len())));
        }
        for ((st, len), v) in runs.iter().zip(seen) {
            x.copy_within(st * d * 4, &v.tokens, 0, len * d * 4)?;
        }
        if !seen.is_empty() && !self.qk_norm {
            return Err(Error("pictures in the prompt are read by Qwen3-VL's encoder only (this is Qwen2.5-VL's)".into()));
        }
        let pos: Vec<[i64; 3]> = if seen.is_empty() {
            (0..s as i64).map(|p| [p, p, p]).collect()
        } else {
            Self::positions(ids, pad, &seen.iter().map(|v| v.grid).collect::<Vec<_>>())?
        };
        let hb = DevBuf::new(gpu, s * d * 2)?;
        let qkv = DevBuf::new(gpu, s * qkv_w * 2)?;
        let att = DevBuf::new(gpu, s * h * HEAD * 2)?;
        let o = DevBuf::f32(gpu, s * d)?;
        let gu = DevBuf::new(gpu, s * 2 * f * 2)?;
        let act = DevBuf::new(gpu, s * f * 2)?;
        // (cos, sin) per token and pair (i, 64 + i); pair i's axis: interleaved M-RoPE (the three agree on text)
        let mut cs = vec![0f32; s * HEAD];
        for (t, p) in pos.iter().enumerate() {
            for i in 0..HEAD / 2 {
                let axis = if i < 60 && i % 3 == 1 { 1 } else if i < 60 && i % 3 == 2 { 2 } else { 0 };
                let a = p[axis] as f64 / self.theta.powf(2.0 * i as f64 / HEAD as f64);
                cs[t * HEAD + 2 * i] = a.cos() as f32;
                cs[t * HEAD + 2 * i + 1] = a.sin() as f32;
            }
        }
        let cs = DevBuf::from_f32(gpu, &cs)?;
        let half_at = |b: &DevBuf, at: usize| -> *mut std::ffi::c_void { b.ptr().cast::<u16>().wrapping_add(at).cast() };
        // the layers' matrices: two staging buffers on the card, layer l + 1 copied (the copy lane) while layer l runs
        let nl = n.min(self.layers);
        let host: Vec<Arc<HostLayer>> = (0..nl).map(|l| self.host_layer(l)).collect::<Result<_>>()?;
        let most = host.iter().map(|h| h.buf.len).max().unwrap_or(1);
        let stage = [DevBuf::new(gpu, most)?, DevBuf::new(gpu, most)?];
        let mut copied: Vec<i64> = vec![0; nl];
        let mut done: Vec<i64> = vec![0; nl];
        if nl > 0 {
            copied[0] = stage[0].copy_from_host_async(0, &host[0].buf, 0, host[0].buf.len, &[])?;
        }
        for l in 0..nl {
            if l + 1 < nl {
                // into the buffer layer l - 1 read: after its work
                let after: Vec<i64> = if l >= 1 { vec![done[l - 1]] } else { Vec::new() };
                copied[l + 1] = stage[(l + 1) % 2].copy_from_host_async(0, &host[l + 1].buf, 0, host[l + 1].buf.len, &after)?;
            }
            if host[l].buf.pinned() {
                gpu.await_ticket(copied[l])?;
            }
            let st = &stage[l % 2];
            let (wq, wk, wv, wo, wg, wu, wd) = {
                let m = |i: usize| -> (*const std::ffi::c_void, u32, usize) {
                    let (off, code, rows) = host[l].mats[i];
                    (st.ptr().cast::<u8>().wrapping_add(off).cast(), code, rows)
                };
                (m(0), m(1), m(2), m(3), m(4), m(5), m(6))
            };
            let p = format!("blk.{l}.");
            let (in_norm, post_norm) = (self.vector(&format!("{p}attn_norm.weight"))?, self.vector(&format!("{p}ffn_norm.weight"))?);
            nsd.rms_norm_mod(x.ptr(), Dt::F32, s, d, in_norm.ptr(), self.eps, none(), none(), none(), hb.ptr(), Dt::F16)?;
            let mut off = 0;
            for (name, w) in [("attn_q", wq), ("attn_k", wk), ("attn_v", wv)] {
                let bias = if self.qkv_bias { Some(self.vector(&format!("{p}{name}.bias"))?) } else { None };
                nsd.linear_q(hb.ptr(), Dt::F16, s, d, w.0, w.1, w.2, bias.as_ref().map_or(none(), |b| b.ptr()), half_at(&qkv, off), Dt::F16, qkv_w)?;
                off += w.2;
            }
            if self.qk_norm {
                let (q_norm, k_norm) = (self.vector(&format!("{p}attn_q_norm.weight"))?, self.vector(&format!("{p}attn_k_norm.weight"))?);
                nsd.rms_rope(qkv.ptr(), Dt::F16, s, h, HEAD, qkv_w, q_norm.ptr(), self.eps, cs.ptr(), HEAD)?;
                nsd.rms_rope(half_at(&qkv, h * HEAD), Dt::F16, s, kv, HEAD, qkv_w, k_norm.ptr(), self.eps, cs.ptr(), HEAD)?;
            } else {
                nsd.rope(qkv.ptr(), Dt::F16, s, h, HEAD, qkv_w, cs.ptr(), HEAD)?;
                nsd.rope(half_at(&qkv, h * HEAD), Dt::F16, s, kv, HEAD, qkv_w, cs.ptr(), HEAD)?;
            }
            nsd.attention_causal(qkv.ptr(), half_at(&qkv, h * HEAD), half_at(&qkv, (h + kv) * HEAD), Dt::F16, s, h, kv, HEAD, qkv_w, qkv_w, att.ptr())?;
            nsd.linear_q(att.ptr(), Dt::F16, s, h * HEAD, wo.0, wo.1, wo.2, none(), o.ptr(), Dt::F32, 0)?;
            nsd.gate_add(x.ptr(), Dt::F32, s, d, o.ptr(), Dt::F32, none(), none())?;
            nsd.rms_norm_mod(x.ptr(), Dt::F32, s, d, post_norm.ptr(), self.eps, none(), none(), none(), hb.ptr(), Dt::F16)?;
            nsd.linear_q(hb.ptr(), Dt::F16, s, d, wg.0, wg.1, wg.2, none(), gu.ptr(), Dt::F16, 2 * f)?;
            nsd.linear_q(hb.ptr(), Dt::F16, s, d, wu.0, wu.1, wu.2, none(), half_at(&gu, wg.2), Dt::F16, 2 * f)?;
            nsd.swiglu(gu.ptr(), Dt::F16, s, f, act.ptr(), Dt::F16)?;
            nsd.linear_q(act.ptr(), Dt::F16, s, f, wd.0, wd.1, wd.2, none(), o.ptr(), Dt::F32, 0)?;
            nsd.gate_add(x.ptr(), Dt::F32, s, d, o.ptr(), Dt::F32, none(), none())?;
            // deepstack: the pictures' features of this depth added at their tokens
            for ((st, len), v) in runs.iter().zip(seen) {
                if let Some(ds) = v.deep.get(l) {
                    nsd.gate_add(x.fp().wrapping_add(st * d).cast(), Dt::F32, *len, d, ds.ptr(), Dt::F32, none(), none())?;
                }
            }
            done[l] = gpu.ticket()?.unwrap_or(0);
        }
        // the staging buffers are freed on return: the work on them first
        nsd.wait()?;
        Ok(x)
    }
}
