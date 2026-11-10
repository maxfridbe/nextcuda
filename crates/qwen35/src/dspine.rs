//! The DSpine draft for Qwen3.8-27B (Tencent's `Qwen3.8-27B-DSpine-b16`: a folder of config.json and model.safetensors,
//! `architectures: DFlashDraftModel`), after nextsycl's port (llm/qwen35/src/dspine.rs there). A parallel block drafter:
//! one pass proposes a whole block of tokens, every layer handing each position's predicted feature to the next position
//! (the "code bridge"), so the block's tokens depend on each other without being drafted one at a time:
//!
//! ```text
//!   context   ctx_row = hidden_norm(fc(concat of the target's residual stream after layers 1, 16, 31, 46, 61))
//!             a row per committed position; each draft layer keeps k / v of ctx_row (its own k_proj / v_proj, k_norm,
//!             rotary at the row's position) in its cache - written behind every target pass (prompt chunks, verify rows).
//!             fc is linear: each target layer's share fc_j . aux_j is made on the card that runs layer j, the partial
//!             sum crosses to the next card with the rows (over two cards, the 3060's layers 1 and 16 send one row
//!             set, not two)
//!   block     rows [anchor, mask, mask, ...] (the anchor: the last committed token, not yet fed to the target; mask:
//!             token 248070), at the positions after the context; their input is the *target's* embedding of the token
//!   layer l   x -> input_layernorm -> q / k / v (32 / 8 heads of 128) -> q_norm, k_norm -> rotary (theta 1e7)
//!             -> attention over the context rows (window 8,192) and the block's rows causally -> o -> + x
//!             -> post_attention_layernorm -> SwiGLU of 17,408 -> + (raw = h + mlp)
//!   bridge    after every layer, for the block's rows:
//!               rms = |raw| / sqrt(H);  n = raw / (rms + eps);  code = sqrt(C) . unit(read_l n)   (C = 1024)
//!               row 0's code is sqrt(C) . codebook[anchor];  packed = [G; Gp; A] code  (q | s | v, rank 512 each)
//!               message[i] = sigmoid(q[i] + s[i-1] + bias) * v[i-1]  (row 0: none);  out = raw + rms . W_l message
//!   head      the final norm of the last layer's out, the target's head -> each row's 16 best tokens (candidates), and
//!             the refinement: row i's choice among its candidates is conditioned on row i - 1's choice (the anchor for
//!             row 0):
//!               tg[cand] = (head_row[cand] . norm_w) W_last        tsv[sender] = sqrt(C) codebook[sender] [Gp; A]^T
//!               m = sigmoid(q + tsv[sender][:512] + bias) * tsv[sender][512:]
//!               score[k] = raw . norm_w . head_row[cand k] + rms . m . tg[cand k];  the best k, for each sender k'
//!             the drafts are then the chain: row 0's best for the anchor, row i's best for row i - 1's choice. (nextsycl
//!             keeps tg and tsv as two vocabulary tables, 0.76 GiB; here each draft makes the rows it needs - its
//!             candidates' tg, its senders' tsv from the checkpoint's codebook rows - so the 12 GB card keeps that room)
//!   verify    the target runs [anchor, drafts] in one pass; the longest prefix whose tokens the target's own sampler draws
//!             is kept (the engine's, as with MTP); the block is causal, so drafting fewer rows than 16 gives those rows'
//!             values exactly
//! ```
//!
//! The weights come as bfloat16: the layers kept as Q8_0 (made on the host at load), the bridge's matrices as float32.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use nextcuda_core::{DevBuf, Error, Gpu, Result};
use nextcuda_gguf::safetensors::SafeTensors;
use nextcuda_gguf::GType;

use crate::engine::Mat;

fn e(x: impl std::fmt::Display) -> Error {
    Error(x.to_string())
}

fn st_err(x: nextcuda_gguf::Error) -> Error {
    Error(x.0)
}

/// What the checkpoint's config.json says the draft is, as far as the engine needs it (validated: this code runs the
/// serial-table, independent-write, terminal-free DSpine of Qwen3.8-27B and nothing else)
#[derive(Debug, Clone, PartialEq)]
pub struct Cfg {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub n_ff: usize,
    pub theta: f32,
    pub eps: f32,
    /// keys a query sees (itself the last)
    pub window: usize,
    pub mask_id: u32,
    /// the target layers whose outputs it reads (outputs: the residual stream after the layer)
    pub target_layers: Vec<usize>,
    pub code: usize,
    pub rank: usize,
    pub bridge_eps: f32,
    /// the candidates a row keeps
    pub topk: usize,
    /// the block of the checkpoint
    pub block: usize,
    pub vocab: usize,
}

impl Cfg {
    pub fn parse(text: &[u8]) -> Result<Cfg> {
        let c: serde_json::Value = serde_json::from_slice(text).map_err(|x| e(format!("dspine config.json: {x}")))?;
        let num = |v: &serde_json::Value, k: &str| v[k].as_u64().map(|x| x as usize).ok_or_else(|| e(format!("dspine config: no {k}")));
        if !c["architectures"].as_array().is_some_and(|a| a.iter().any(|x| x.as_str() == Some("DFlashDraftModel") || x.as_str() == Some("DSpineDraftModel"))) {
            return Err(e("dspine: not a DFlashDraftModel / DSpineDraftModel"));
        }
        let d = &c["dflash_config"];
        if d.is_null() {
            return Err(e("dspine config: no dflash_config"));
        }
        let want = |ok: bool, what: &str| if ok { Ok(()) } else { Err(e(format!("dspine: unsupported checkpoint ({what})"))) };
        want(d["selected_prefix_decode_mode"].as_str() == Some("serial"), "selected_prefix_decode_mode is not serial")?;
        want(d["selected_prefix_feature_writes"].as_bool() == Some(true), "no feature writes")?;
        want(d["selected_prefix_feature_write_mode"].as_str() == Some("independent"), "writes are not independent")?;
        want(d["selected_prefix_terminal"].as_bool() == Some(false), "a terminal write")?;
        want(d["draft_block_causal"].as_bool() == Some(true), "a block that is not causal")?;
        want(c["code_bridge_second_pass"]["mode"].as_str() == Some("serial_table"), "second pass is not serial_table")?;
        let layers = num(&c, "num_hidden_layers")?;
        let writes: Vec<usize> = d["selected_prefix_feature_layers"].as_array().into_iter().flatten().filter_map(|v| v.as_u64().map(|x| x as usize)).collect();
        want(writes == (1..=layers).collect::<Vec<_>>(), "a bridge write after other than every layer")?;
        want(c["attention_bias"].as_bool() != Some(true), "attention bias")?;
        want(c["hidden_act"].as_str().is_none_or(|a| a == "silu"), "an activation other than silu")?;
        want(c["rope_scaling"].is_null(), "scaled rotary positions")?;
        let window = match (c["use_sliding_window"].as_bool(), c["layer_types"].as_array()) {
            (Some(true), Some(t)) if t.len() == layers && t.iter().all(|x| x.as_str() == Some("sliding_attention")) => num(&c, "sliding_window")?,
            (Some(true), _) => return Err(e("dspine: sliding window layers of mixed kinds")),
            _ => 0,
        };
        let bridge = &d["selected_prefix_code_bridge"];
        let target_layers: Vec<usize> = d["target_layer_ids"].as_array().into_iter().flatten().filter_map(|v| v.as_u64().map(|x| x as usize)).collect();
        let topk = num(&c["code_bridge_second_pass"], "head_topk")?;
        want(d["selected_prefix_last_write_refine"]["head_topk"].as_u64().is_none_or(|k| k as usize == topk), "refinement widths disagree")?;
        want(topk.is_power_of_two() && (2..=32).contains(&topk), "head_topk out of range")?;
        let theta = c["rope_parameters"]["rope_theta"].as_f64().or_else(|| c["rope_theta"].as_f64()).ok_or_else(|| e("dspine config: no rope_theta"))? as f32;
        let cfg = Cfg {
            hidden: num(&c, "hidden_size")?,
            layers,
            heads: num(&c, "num_attention_heads")?,
            kv_heads: num(&c, "num_key_value_heads")?,
            head_dim: num(&c, "head_dim")?,
            n_ff: num(&c, "intermediate_size")?,
            theta,
            eps: c["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
            window,
            mask_id: num(d, "mask_token_id")? as u32,
            target_layers,
            code: num(bridge, "code_size")?,
            rank: num(bridge, "rank")?,
            bridge_eps: bridge["eps"].as_f64().ok_or_else(|| e("dspine config: no bridge eps"))? as f32,
            topk,
            block: num(&c, "block_size")?,
            vocab: num(&c, "vocab_size")?,
        };
        // the shapes this engine's kernels cover: heads of 128 (ns_q35_dsp_attn: at most 32 query rows a key head, so four
        // query heads a key head x 8 rows), the bridge's widths
        want(cfg.head_dim == 128 && cfg.heads == 4 * cfg.kv_heads, "heads other than 128 wide with four query heads a key head")?;
        want(cfg.target_layers.len() == 5 && cfg.target_layers.windows(2).all(|w| w[0] < w[1]), "target layers (five, ascending)")?;
        want(cfg.hidden.is_multiple_of(256) && cfg.n_ff.is_multiple_of(32) && cfg.rank == 512 && cfg.code == 1024, "bridge widths (code 1024, rank 512)")?;
        want(cfg.layers >= 1 && cfg.block >= 2, "layers / block")?;
        Ok(cfg)
    }
}

/// NS_Q35_DSPINE: the draft's folder (config.json, model.safetensors), or `auto` for a dspine/ folder beside the model's
/// file; unset: none (MTP). Opt-in
pub fn path(model: &Path) -> Option<PathBuf> {
    match std::env::var("NS_Q35_DSPINE").as_deref() {
        Ok("auto") => {
            let d = model.parent()?.join("dspine");
            d.join("model.safetensors").is_file().then_some(d)
        }
        Ok(v) if !v.is_empty() && v != "0" => Some(v.into()),
        _ => None,
    }
}

/// The most rows a verify pass (and so a draft block) holds: the ring's slack
pub const RING_SLACK: usize = 8;

/// A span of rows for the ring cache: `n` rows from row `row` of the source go to slots `slot`.. of every head's cache
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub row: usize,
    pub slot: usize,
    pub n: usize,
}

/// The draft's context k / v cache as a ring of the attention window (nextsycl's). Position p lives in slot `p % r`,
/// r = window + 8 (the slack a verify pass's rows overwrite and a rollback needs: rows p0.. overwrite the positions
/// p0 - r.., all before the first key any rolled-back position still reads). The attention kernel reads a contiguous run
/// of keys, so the ring is mirrored: slot s is also stored at s + r, and the `slots` = 2 r allocated per head let every
/// window (r - 1 + 8 keys at most, from a slot below r) read straight through the wrap. Rotary is applied before the
/// store, at the absolute position. A session shorter than the window (+ slack) never wraps: one flat cache
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ring {
    pub r: usize,
    pub slots: usize,
    pub mirrored: bool,
}

impl Ring {
    pub fn new(window: usize, max_ctx: usize) -> Ring {
        if window == 0 || max_ctx <= window + RING_SLACK {
            Ring { r: max_ctx.max(1), slots: max_ctx.max(1), mirrored: false }
        } else {
            let r = window + RING_SLACK;
            Ring { r, slots: 2 * r, mirrored: true }
        }
    }

    /// Where `n` rows of positions p.. are stored: runs that do not cross the wrap, each also at its mirror
    pub fn spans(&self, p: usize, n: usize) -> Vec<Span> {
        let mut out = Vec::new();
        let mut row = 0;
        while row < n {
            let slot = (p + row) % self.r;
            let take = (self.r - slot).min(n - row);
            out.push(Span { row, slot, n: take });
            if self.mirrored {
                out.push(Span { row, slot: slot + self.r, n: take });
            }
            row += take;
        }
        out
    }

    /// The slot of the first key a block at p0 reads (`window_start`): the run from there is `p0 - start + rows` keys long
    pub fn start_slot(&self, start: usize) -> usize {
        start % self.r
    }

    /// The slots a session at position `pos` holds that a copy or a checkpoint keeps: [0, n) of the first r (a mirror
    /// is written again from them)
    pub fn held(&self, pos: usize) -> usize {
        pos.min(self.r)
    }
}

/// The first key a block of queries at positions p0.. reads from the cache (their rows share it): the first row's whole
/// window - the later rows see a key or two more than their window
pub fn window_start(p0: usize, window: usize) -> usize {
    match window {
        0 => 0,
        w => (p0 + 1).saturating_sub(w),
    }
}

/// The drafts of a block from the refinement's choices: `idx[i][k]` is the candidate row i picks when the sender (row i - 1's
/// choice, the anchor for row 0) was its candidate k; `cand[i][k]` its token. Row 0's pick is the anchor's (every k of
/// row 0 is the same sender)
pub fn chain(idx: &[Vec<i32>], cand: &[Vec<i32>]) -> Vec<u32> {
    let mut out = Vec::with_capacity(idx.len());
    let mut j = 0usize;
    for (i, (ix, cd)) in idx.iter().zip(cand).enumerate() {
        let pick = if i == 0 { ix[0] } else { ix[j] } as usize;
        out.push(cd[pick] as u32);
        j = pick;
    }
    out
}

/// The block's tokens: the anchor, then `drafts` masks
pub fn block_tokens(anchor: u32, drafts: usize, mask: u32) -> Vec<u32> {
    let mut v = Vec::with_capacity(drafts + 1);
    v.push(anchor);
    v.extend(std::iter::repeat_n(mask, drafts));
    v
}

/// The senders of each (row, candidate) pair: the anchor for row 0, else row r - 1's candidate k
pub fn senders(anchor: u32, cand: &[i32], n: usize, k: usize) -> Vec<u32> {
    (0..n).flat_map(|r| (0..k).map(move |j| (r, j))).map(|(r, j)| if r == 0 { anchor } else { cand[(r - 1) * k + j] as u32 }).collect()
}

pub fn bf16(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect()
}

/// float32 to IEEE half (round to nearest even; overflow to infinity)
pub fn f32_to_f16(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32;
    let man = x & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let mut v = m >> shift;
        let rest = m & ((1 << shift) - 1);
        if rest > half || (rest == half && (v & 1) == 1) {
            v += 1;
        }
        return sign | v as u16;
    }
    let mut v = ((e as u32) << 10) | (man >> 13);
    let rest = man & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && (v & 1) == 1) {
        v += 1;
    }
    sign | v as u16
}

/// A row of values as Q8_0 blocks (32 values: a half scale, 32 int8) into `out`
fn quant_q8_0(x: &[f32], out: &mut [u8]) {
    for (ob, xb) in out.chunks_exact_mut(34).zip(x.chunks_exact(32)) {
        let amax = xb.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        ob[..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        for (j, v) in xb.iter().enumerate() {
            ob[2 + j] = (v * id).round().clamp(-127.0, 127.0) as i8 as u8;
        }
    }
}

/// Columns [c0, c0 + n) of a bfloat16 matrix [rows, cols] as a Q8_0 Mat [rows, n] on `gpu`
fn q8_columns(st: &SafeTensors, name: &str, c0: usize, n: usize, gpu: &Arc<Gpu>) -> Result<Mat> {
    let t = st.need(name).map_err(st_err)?;
    if t.dtype != "BF16" || t.shape.len() != 2 {
        return Err(e(format!("{name}: {} {:?}, not a bfloat16 matrix", t.dtype, t.shape)));
    }
    let (rows, cols) = (t.shape[0] as usize, t.shape[1] as usize);
    if !n.is_multiple_of(32) || c0 + n > cols {
        return Err(e(format!("{name}: columns {c0}..{} of a matrix of {cols}", c0 + n)));
    }
    let raw = st.read(t).map_err(st_err)?;
    let rb = n / 32 * 34;
    let mut out = vec![0u8; rows * rb];
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(16);
    let per = rows.div_ceil(threads);
    std::thread::scope(|s| {
        for (oc, ic) in out.chunks_mut(per * rb).zip(raw.chunks(per * cols * 2)) {
            s.spawn(move || {
                let mut row = vec![0f32; n];
                for (o, i) in oc.chunks_exact_mut(rb).zip(ic.chunks_exact(cols * 2)) {
                    for (j, v) in row.iter_mut().enumerate() {
                        let c = c0 + j;
                        *v = f32::from_bits((u16::from_le_bytes([i[2 * c], i[2 * c + 1]]) as u32) << 16);
                    }
                    quant_q8_0(&row, o);
                }
            });
        }
    });
    let buf = DevBuf::new(gpu, out.len())?;
    buf.write(0, &out)?;
    Ok(Mat { buf, ty: GType::Q8_0, rows, cols: n })
}

/// One decoder layer's weights
pub struct DLayer {
    pub in_norm: DevBuf,
    pub post_norm: DevBuf,
    pub q_norm: DevBuf,
    pub k_norm: DevBuf,
    pub q: Mat,
    pub k: Mat,
    pub v: Mat,
    pub o: Mat,
    pub gate: Mat,
    pub up: Mat,
    pub down: Mat,
}

pub struct Dspine {
    pub cfg: Cfg,
    pub st: SafeTensors,
    /// fc's five column groups (one per target layer's output), each on the GPU of its target layer
    pub fc: Vec<Mat>,
    pub hidden_norm: DevBuf,
    pub norm: DevBuf,
    pub layers: Vec<DLayer>,
    /// the bridge: per layer the read matrix [code, H], the packed (gate | pred gate | prev) [3 rank, code], the gate
    /// bias [rank], per layer the write matrix [H, rank] (all float32)
    pub read: Vec<Mat>,
    pub packed: Mat,
    pub gate_bias: DevBuf,
    pub write: Vec<Mat>,
    /// the refinement's two matrices, half: (norm_w x the last write matrix)^T [rank][H] (a candidate's tg is its head
    /// row through it) and [Gp; A] [2 rank][code] (a sender's tsv is its scaled codebook row through it)
    pub wt: DevBuf,
    pub pv: DevBuf,
    pub bytes: u64,
}

/// Its bytes on the GPUs, from the file's header and the config: (on the head's GPU, fc's share of a target layer - on
/// that layer's GPU)
pub fn weight_bytes(dir: &Path) -> Result<(u64, u64)> {
    let cfg = Cfg::parse(&std::fs::read(dir.join("config.json")).map_err(|x| e(format!("{}: {x}", dir.display())))?)?;
    let st = SafeTensors::open(&dir.join("model.safetensors")).map_err(st_err)?;
    let q8: u64 = st.tensors.iter().filter(|t| !t.name.starts_with("selected_prefix_bridge") && t.shape.len() == 2).map(|t| t.elements() / 32 * 34).sum();
    let vecs: u64 = st.tensors.iter().filter(|t| !t.name.starts_with("selected_prefix_bridge") && t.shape.len() == 1).map(|t| t.elements() * 4).sum();
    let (h, c, r, l) = (cfg.hidden as u64, cfg.code as u64, cfg.rank as u64, cfg.layers as u64);
    let bridge = 4 * (l * c * h + 3 * r * c + r + l * h * r);
    let fc_each = h * h / 32 * 34;
    let small = 2 * (r * h + 2 * r * c) + (16 << 20);
    Ok((q8 - 5 * fc_each + vecs + bridge + small, fc_each))
}

impl Dspine {
    /// Loads the draft: its layers and bridge onto `gpu` (the head's), fc's column group j onto `fc_gpu(layer)` (the GPU
    /// of target layer j). `head`: the target's head (rows [vocab, hidden]), checked against the config
    pub fn load(dir: &Path, gpu: &Arc<Gpu>, fc_gpu: &dyn Fn(usize) -> Arc<Gpu>, n_embd: usize, layers: usize, head: &Mat, log: &mut dyn FnMut(String)) -> Result<Dspine> {
        let cfg = Cfg::parse(&std::fs::read(dir.join("config.json")).map_err(|x| e(format!("{}: {x}", dir.display())))?)?;
        if cfg.hidden != n_embd || cfg.vocab != head.rows || head.cols != n_embd || cfg.target_layers.iter().any(|l| *l >= layers) {
            return Err(e(format!("dspine: hidden {} / vocabulary {} / target layers {:?} over a target of {n_embd} / {} / {layers} layers", cfg.hidden, cfg.vocab,
                                 cfg.target_layers, head.rows)));
        }
        let st = SafeTensors::open(&dir.join("model.safetensors")).map_err(st_err)?;
        let mut bytes = 0u64;
        let h = cfg.hidden;
        let vecf = |name: &str, bytes: &mut u64| -> Result<DevBuf> {
            let v = st.f32(name).map_err(st_err)?;
            let b = DevBuf::from_f32(gpu, &v)?;
            *bytes += b.len as u64;
            Ok(b)
        };
        let q8 = |name: &str, n: usize, on: &Arc<Gpu>, c0: usize, bytes: &mut u64| -> Result<Mat> {
            let m = q8_columns(&st, name, c0, n, on)?;
            *bytes += m.buf.len as u64;
            Ok(m)
        };
        // fc: one Mat a target layer's columns, on that layer's GPU
        let mut fc = Vec::with_capacity(5);
        for j in 0..5 {
            fc.push(q8("fc.weight", h, &fc_gpu(cfg.target_layers[j]), j * h, &mut bytes)?);
        }
        let hidden_norm = vecf("hidden_norm.weight", &mut bytes)?;
        let norm = vecf("norm.weight", &mut bytes)?;
        let mut ls = Vec::with_capacity(cfg.layers);
        for l in 0..cfg.layers {
            let p = format!("layers.{l}.");
            let w = |s: &str| format!("{p}{s}.weight");
            ls.push(DLayer {
                in_norm: vecf(&w("input_layernorm"), &mut bytes)?,
                post_norm: vecf(&w("post_attention_layernorm"), &mut bytes)?,
                q_norm: vecf(&w("self_attn.q_norm"), &mut bytes)?,
                k_norm: vecf(&w("self_attn.k_norm"), &mut bytes)?,
                q: q8(&w("self_attn.q_proj"), h, gpu, 0, &mut bytes)?,
                k: q8(&w("self_attn.k_proj"), h, gpu, 0, &mut bytes)?,
                v: q8(&w("self_attn.v_proj"), h, gpu, 0, &mut bytes)?,
                o: q8(&w("self_attn.o_proj"), cfg.heads * cfg.head_dim, gpu, 0, &mut bytes)?,
                gate: q8(&w("mlp.gate_proj"), h, gpu, 0, &mut bytes)?,
                up: q8(&w("mlp.up_proj"), h, gpu, 0, &mut bytes)?,
                down: q8(&w("mlp.down_proj"), cfg.n_ff, gpu, 0, &mut bytes)?,
            });
            log(format!("dspine: layer {} of {} ({:.2} GiB)", l + 1, cfg.layers, bytes as f64 / (1u64 << 30) as f64));
        }
        // the bridge, float32
        let f32mat = |v: &[f32], rows: usize, cols: usize, bytes: &mut u64| -> Result<Mat> {
            let buf = DevBuf::from_f32(gpu, v)?;
            *bytes += buf.len as u64;
            Ok(Mat { buf, ty: GType::F32, rows, cols })
        };
        let b = "selected_prefix_bridge.";
        for (name, want) in [("independent_write_version", 1), ("terminal_disabled_version", 1)] {
            let t = st.need(&format!("{b}{name}")).map_err(st_err)?;
            let raw = st.read(t).map_err(st_err)?;
            if raw.len() != 8 || i64::from_le_bytes(raw[..8].try_into().unwrap_or_default()) != want {
                return Err(e(format!("dspine: {name} is not {want}")));
            }
        }
        // a terminal-free checkpoint's `out` projection is zero
        if st.f32(&format!("{b}out.weight")).map_err(st_err)?.iter().any(|v| *v != 0.0) {
            return Err(e("dspine: a nonzero terminal projection"));
        }
        let (c, r) = (cfg.code, cfg.rank);
        let cb = st.need(&format!("{b}codebook")).map_err(st_err)?;
        if cb.shape != [cfg.vocab as u64, c as u64] || cb.dtype != "BF16" {
            return Err(e(format!("dspine: a codebook of {:?} {}", cb.shape, cb.dtype)));
        }
        let early = st.f32(&format!("{b}early_read_weights")).map_err(st_err)?;
        if early.len() != (cfg.layers - 1) * c * h {
            return Err(e("dspine: early_read_weights of another shape"));
        }
        let mut read = Vec::with_capacity(cfg.layers);
        for l in 0..cfg.layers - 1 {
            read.push(f32mat(&early[l * c * h..(l + 1) * c * h], c, h, &mut bytes)?);
        }
        drop(early);
        read.push(f32mat(&st.f32(&format!("{b}read_weight")).map_err(st_err)?, c, h, &mut bytes)?);
        let mut packed = st.f32(&format!("{b}gate.weight")).map_err(st_err)?;
        let pred = st.f32(&format!("{b}predecessor_gate.weight")).map_err(st_err)?;
        let prev = st.f32(&format!("{b}prev.weight")).map_err(st_err)?;
        if packed.len() != r * c || pred.len() != r * c || prev.len() != r * c {
            return Err(e("dspine: bridge matrices of other shapes"));
        }
        packed.extend_from_slice(&pred);
        packed.extend_from_slice(&prev);
        let packed_mat = f32mat(&packed, 3 * r, c, &mut bytes)?;
        let gate_bias = vecf(&format!("{b}gate.bias"), &mut bytes)?;
        let wall = st.f32(&format!("{b}feature_write_weights")).map_err(st_err)?;
        if wall.len() != cfg.layers * h * r {
            return Err(e("dspine: feature_write_weights of another shape"));
        }
        let mut write = Vec::with_capacity(cfg.layers);
        for l in 0..cfg.layers {
            write.push(f32mat(&wall[l * h * r..(l + 1) * h * r], h, r, &mut bytes)?);
        }
        // the refinement's matrices in half: wt [rank][H] = (norm[h] W_last[h][r])^T, pv = [Gp; A] [2 rank][code]
        let norm_w = st.f32("norm.weight").map_err(st_err)?;
        let last = &wall[(cfg.layers - 1) * h * r..];
        let mut wt = vec![0u16; r * h];
        for hh in 0..h {
            for rr in 0..r {
                wt[rr * h + hh] = f32_to_f16(norm_w[hh] * last[hh * r + rr]);
            }
        }
        let half = |v: &[u16]| -> Result<DevBuf> {
            let d = DevBuf::new(gpu, v.len() * 2)?;
            d.write(0, &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
            Ok(d)
        };
        let wt_dev = half(&wt)?;
        let pv: Vec<u16> = packed[r * c..].iter().map(|v| f32_to_f16(*v)).collect();
        let pv_dev = half(&pv)?;
        bytes += (wt_dev.len + pv_dev.len) as u64;
        drop(wall);
        gpu.sync()?;
        Ok(Dspine { st, fc, hidden_norm, norm, layers: ls, read, packed: packed_mat, gate_bias, write, wt: wt_dev, pv: pv_dev, bytes, cfg })
    }

    /// sqrt(C) x the codebook rows of `tokens`, as half (exact: a bfloat16 times 32 fits half's mantissa), [n][code]
    pub fn code_rows_f16(&self, tokens: &[u32], out: &mut [u8]) -> Result<()> {
        let t = self.st.need("selected_prefix_bridge.codebook").map_err(st_err)?;
        let c = self.cfg.code;
        let s = (c as f32).sqrt();
        let mut raw = vec![0u8; c * 2];
        for (i, tok) in tokens.iter().enumerate() {
            if *tok as usize >= self.cfg.vocab {
                return Err(e(format!("dspine: token {tok} past the vocabulary")));
            }
            self.st.read_into(t, (*tok as usize * c * 2) as u64, &mut raw).map_err(st_err)?;
            for (j, v) in bf16(&raw).into_iter().enumerate() {
                out[(i * c + j) * 2..(i * c + j) * 2 + 2].copy_from_slice(&f32_to_f16(v * s).to_le_bytes());
            }
        }
        Ok(())
    }

    /// sqrt(C) x the codebook row of `token` (the anchor's code), float32 bytes
    pub fn anchor_row(&self, token: u32, out: &mut [u8]) -> Result<()> {
        let t = self.st.need("selected_prefix_bridge.codebook").map_err(st_err)?;
        let c = self.cfg.code;
        if token as usize >= self.cfg.vocab {
            return Err(e(format!("dspine: token {token} past the vocabulary")));
        }
        let mut raw = vec![0u8; c * 2];
        self.st.read_into(t, (token as usize * c * 2) as u64, &mut raw).map_err(st_err)?;
        let s = (c as f32).sqrt();
        for (j, v) in bf16(&raw).into_iter().enumerate() {
            out[j * 4..j * 4 + 4].copy_from_slice(&(v * s).to_le_bytes());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CFG: &str = r#"{
      "architectures": ["DFlashDraftModel"], "attention_bias": false, "block_size": 16,
      "code_bridge_second_pass": {"first_exact": 4, "head_topk": 16, "mode": "serial_table"},
      "dflash_config": {"draft_block_causal": true, "mask_token_id": 248070,
        "selected_prefix_code_bridge": {"code_size": 1024, "eps": 1e-06, "kappa": 10.0, "rank": 512},
        "selected_prefix_decode_mode": "serial", "selected_prefix_feature_layers": [1,2,3,4,5],
        "selected_prefix_feature_write_mode": "independent", "selected_prefix_feature_writes": true,
        "selected_prefix_last_write_refine": {"head_topk": 16, "kind": "exact_sender", "version": 1, "weight": 1.0},
        "selected_prefix_terminal": false, "target_layer_ids": [1,16,31,46,61]},
      "head_dim": 128, "hidden_act": "silu", "hidden_size": 5120, "intermediate_size": 17408,
      "layer_types": ["sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention"],
      "num_attention_heads": 32, "num_hidden_layers": 5, "num_key_value_heads": 8, "rms_norm_eps": 1e-06,
      "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"}, "rope_scaling": null,
      "sliding_window": 8192, "use_sliding_window": true, "vocab_size": 248320}"#;

    #[test]
    fn config_of_the_checkpoint() {
        let c = Cfg::parse(CFG.as_bytes()).unwrap();
        assert_eq!((c.hidden, c.layers, c.heads, c.kv_heads, c.head_dim, c.n_ff), (5120, 5, 32, 8, 128, 17408));
        assert_eq!((c.window, c.mask_id, c.topk, c.code, c.rank, c.block), (8192, 248070, 16, 1024, 512, 16));
        assert_eq!(c.target_layers, vec![1, 16, 31, 46, 61]);
        assert_eq!(c.theta, 1e7);
    }

    #[test]
    fn config_refuses_what_it_does_not_run() {
        for (from, to) in [("\"serial\"", "\"parallel\""), ("\"serial_table\"", "\"table\""), ("[1,2,3,4,5]", "[5]"), ("\"selected_prefix_terminal\": false", "\"selected_prefix_terminal\": true"),
                           ("\"num_attention_heads\": 32", "\"num_attention_heads\": 16"), ("\"DFlashDraftModel\"", "\"LlamaForCausalLM\"")] {
            assert!(Cfg::parse(CFG.replace(from, to).as_bytes()).is_err(), "{from} -> {to}");
        }
        // no sliding window: the whole cache
        let c = Cfg::parse(CFG.replace("\"use_sliding_window\": true", "\"use_sliding_window\": false").as_bytes()).unwrap();
        assert_eq!(c.window, 0);
    }

    #[test]
    fn window_start_keeps_the_first_rows_window() {
        assert_eq!(window_start(100, 8192), 0);
        assert_eq!(window_start(8191, 8192), 0);
        assert_eq!(window_start(8192, 8192), 1);
        assert_eq!(window_start(20000, 8192), 20000 + 1 - 8192);
        assert_eq!(window_start(20000, 0), 0);
    }

    /// A model of the cache: slot -> the position stored there; reads the run a block at p0 reads
    fn store(c: &mut [Option<usize>], ring: &Ring, p: usize, n: usize) {
        for sp in ring.spans(p, n) {
            for i in 0..sp.n {
                c[sp.slot + i] = Some(p + sp.row + i);
            }
        }
    }

    fn run(c: &[Option<usize>], ring: &Ring, p0: usize, window: usize, t: usize) -> Vec<Option<usize>> {
        let start = window_start(p0, window);
        let s0 = ring.start_slot(start);
        c[s0..s0 + (p0 - start) + t].to_vec()
    }

    #[test]
    fn ring_sizes() {
        let r = Ring::new(8192, 131072);
        assert_eq!((r.r, r.slots, r.mirrored), (8200, 16400, true));
        let f = Ring::new(8192, 4096);
        assert_eq!((f.r, f.slots, f.mirrored), (4096, 4096, false));
        assert_eq!(Ring::new(8192, 8200).slots, 8200);
        assert_eq!(Ring::new(8192, 8201).slots, 2 * 8200);
        assert_eq!(Ring::new(0, 100000).slots, 100000);
        assert_eq!((r.held(100), r.held(9000)), (100, 8200));
    }

    #[test]
    fn ring_spans_split_at_the_wrap_and_mirror() {
        let r = Ring { r: 10, slots: 20, mirrored: true };
        assert_eq!(r.spans(3, 4), vec![Span { row: 0, slot: 3, n: 4 }, Span { row: 0, slot: 13, n: 4 }]);
        assert_eq!(r.spans(8, 5), vec![Span { row: 0, slot: 8, n: 2 }, Span { row: 0, slot: 18, n: 2 }, Span { row: 2, slot: 0, n: 3 }, Span { row: 2, slot: 10, n: 3 }]);
        assert_eq!(r.spans(30, 10), vec![Span { row: 0, slot: 0, n: 10 }, Span { row: 0, slot: 10, n: 10 }]);
        let f = Ring { r: 10, slots: 10, mirrored: false };
        assert_eq!(f.spans(2, 3), vec![Span { row: 0, slot: 2, n: 3 }]);
    }

    #[test]
    fn ring_reads_the_window_after_wraps_and_rollbacks() {
        // a small window: passes of 8 rows then rejected drafts (keep 1..8), a block read after each, checked against the
        // positions [window_start, p0 + t) a flat cache would hold
        for window in [16usize, 20, 33] {
            let ring = Ring::new(window, 1000);
            assert!(ring.mirrored);
            let mut c = vec![None; ring.slots];
            let mut pos = 0usize;
            let mut seed = 12345u64;
            while pos < 400 {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let t = if pos < 40 { 1 + (seed >> 40) as usize % 30 } else { 8 };
                store(&mut c, &ring, pos, t); // a pass of t rows at pos
                let keep = if t == 8 { 1 + (seed >> 33) as usize % 8 } else { t }; // the verify keeps some rows
                pos += keep;
                // the draft block at pos: its rows are stored, then the run is read
                let tb = 1 + (seed >> 20) as usize % 8;
                store(&mut c, &ring, pos, tb);
                let start = window_start(pos, window);
                let want: Vec<Option<usize>> = (start..pos + tb).map(Some).collect();
                assert_eq!(run(&c, &ring, pos, window, tb), want, "window {window} pos {pos} t {tb}");
            }
        }
    }

    #[test]
    fn chain_follows_the_senders() {
        // row 0: every sender picks candidate 2; row 1: sender 2 picks 0, sender 1 picks 3; row 2: sender 0 picks 1
        let cand = vec![vec![10, 11, 12, 13], vec![20, 21, 22, 23], vec![30, 31, 32, 33]];
        let idx = vec![vec![2, 2, 2, 2], vec![3, 3, 0, 3], vec![1, 0, 0, 0]];
        assert_eq!(chain(&idx, &cand), vec![12, 20, 31]);
        let idx = vec![vec![1, 1, 1, 1], vec![3, 3, 0, 3], vec![1, 0, 0, 0]];
        assert_eq!(chain(&idx, &cand), vec![11, 23, 30]);
        assert!(chain(&[], &[]).is_empty());
    }

    #[test]
    fn senders_are_the_anchor_then_the_previous_rows_candidates() {
        let cand = [10, 11, 20, 21, 30, 31];
        assert_eq!(senders(7, &cand, 3, 2), vec![7, 7, 10, 11, 20, 21]);
    }

    #[test]
    fn block_is_the_anchor_then_masks() {
        assert_eq!(block_tokens(7, 3, 99), vec![7, 99, 99, 99]);
        assert_eq!(block_tokens(7, 0, 99), vec![7]);
    }

    #[test]
    fn q8_0_blocks_round_trip() {
        let x: Vec<f32> = (0..64).map(|i| (i as f32 - 31.5) / 10.0).collect();
        let mut out = vec![0u8; 68];
        quant_q8_0(&x, &mut out);
        for (b, xb) in out.chunks_exact(34).zip(x.chunks_exact(32)) {
            let d = nextcuda_gguf::safetensors::f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
            for (q, v) in b[2..].iter().zip(xb) {
                assert!((*q as i8 as f32 * d - v).abs() <= d * 0.6, "{v}");
            }
        }
    }

    #[test]
    fn half_rounds_to_nearest() {
        for v in [0.0f32, 1.0, -2.0, 0.1, 65504.0, 1e-5, 6.1e-5, -3.3e-7, 12345.678] {
            let h = f32_to_f16(v);
            let back = nextcuda_gguf::safetensors::f16_to_f32(h);
            let tol = (v.abs() * 1e-3).max(6e-8);
            assert!((back - v).abs() <= tol, "{v} -> {back}");
        }
        assert_eq!(f32_to_f16(1e6), 0x7c00);
        // a bfloat16 times 32 is exact in half
        let b = f32::from_bits(0x3d4c_0000);
        assert_eq!(nextcuda_gguf::safetensors::f16_to_f32(f32_to_f16(b * 32.0)), b * 32.0);
    }

    #[test]
    fn bf16_is_the_top_half_of_a_float() {
        assert_eq!(bf16(&[0x80, 0x3f, 0x00, 0xc0]), vec![1.0, -2.0]);
    }
}
