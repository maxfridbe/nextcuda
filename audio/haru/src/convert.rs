//! The language model's GGUF, made once from the checkpoint and kept beside it: Haru-4B's `lm.*` (transformers'
//! Qwen3_5TextModel) as llama.cpp's `qwen35` layout, which nextcuda's Qwen3.5 core (crates/qwen35) runs - its decode
//! products read the matrices in their blocks (Q8_0 by default: 4.5 GB where bfloat16 is 8.5).
//!
//! What llama.cpp's convert_hf_to_gguf.py does for Qwen3.5 (MIT; the transformation is reproduced, not its code):
//! - the zero-centred RMS norms (attn_norm, post_attention_norm, the attention's q / k norms, output_norm) stored as
//!   1 + w; the DeltaNet's gated norm (ssm_norm) as it is;
//! - A_log stored as ssm_a = -exp(A_log); dt_bias as ssm_dt.bias; the causal conv [channels, 1, 4] as [channels, 4];
//! - the value heads reordered from grouped (value head h reads key head h / r, transformers' repeat_interleave) to
//!   tiled (value head j reads key head j mod Hk, the engine's `expand`): new head j = old head (j mod Hk) r + j / Hk,
//!   in every tensor laid out by value head - the v rows of in_proj_qkv and its conv channels, in_proj_z's rows,
//!   in_proj_a / in_proj_b's rows, A_log, dt_bias, and out_proj's columns;
//! - the output tied to the token embedding (no output.weight: the engine takes token_embd's).
//!
//! The 2-D projections are quantized (Q8_0: a half scale and 32 int8 a block, ggml's format, rounded half away from
//! zero as its reference quantizer does) or kept as half (`f16`, for checks); in_proj_a / in_proj_b (the decay's and the
//! write strength's 32 rows) and every vector stay float32.

use std::path::{Path, PathBuf};

use nextcuda_audio::{Error, Result};
use nextcuda_gguf::writer::Writer;
use nextcuda_gguf::{GType, Gguf, Value};

use crate::weights::{to_f32, Config, Weights};

/// Bumped when the file's layout changes: an older file is made again
pub const VERSION: u64 = 1;

/// The cache's file for a weight type ("q8_0", "f16") beside the checkpoint
pub fn path_for(model: &Path, ty: &str) -> PathBuf {
    model.with_file_name(format!("nextcuda-haru-lm-{ty}.gguf"))
}

/// The weight type by name
pub fn gtype(ty: &str) -> Result<GType> {
    match ty {
        "q8_0" => Ok(GType::Q8_0),
        "f16" => Ok(GType::F16),
        _ => Err(Error(format!("NS_HARU_LM {ty:?}: q8_0 or f16"))),
    }
}

/// IEEE half bits of a float, rounded to nearest even
pub fn f32_to_f16(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let e = ((x >> 23) & 0xff) as i32 - 127 + 15;
    let man = x & 0x7f_ffff;
    if ((x >> 23) & 0xff) == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    if e >= 31 {
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

/// Values as Q8_0 blocks (32 values: a half scale d = max |x| / 127, then round(x / d) as int8)
pub fn q8_0(x: &[f32], out: &mut Vec<u8>) {
    for b in x.chunks_exact(32) {
        let amax = b.iter().fold(0f32, |a, v| a.max(v.abs()));
        let d = amax / 127.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        out.extend(f32_to_f16(d).to_le_bytes());
        out.extend(b.iter().map(|v| (v * id).round().clamp(-127.0, 127.0) as i8 as u8));
    }
}

/// The new order of `heads` value heads (`per` consecutive rows / values each) read from the old: new head j is old
/// head (j mod hk) r + j / hk
pub fn head_order(hv: usize, hk: usize) -> Vec<usize> {
    let r = hv / hk;
    (0..hv).map(|j| (j % hk) * r + j / hk).collect()
}

/// Rows [n, k] (row-major) with rows [from, from + hv * per) reordered by value head
fn reorder_rows(v: &mut [f32], k: usize, from: usize, per: usize, order: &[usize]) {
    let block = per * k;
    let old: Vec<f32> = v[from * k..(from + order.len() * per) * k].to_vec();
    for (j, &h) in order.iter().enumerate() {
        v[(from + j * per) * k..(from + (j + 1) * per) * k].copy_from_slice(&old[h * block..(h + 1) * block]);
    }
}

/// A row-major matrix [n, k]'s columns reordered by value head (per columns a head)
fn reorder_cols(v: &mut [f32], k: usize, per: usize, order: &[usize]) {
    let mut row = vec![0f32; k];
    for r in v.chunks_exact_mut(k) {
        row.copy_from_slice(r);
        for (j, &h) in order.iter().enumerate() {
            r[j * per..(j + 1) * per].copy_from_slice(&row[h * per..(h + 1) * per]);
        }
    }
}

/// A tensor of the file: its name there, its source, its shape (outermost first), its type and what is done to it
struct Item {
    name: String,
    src: String,
    shape: Vec<u64>,
    ty: GType,
    fix: Fix,
}

#[derive(Clone, Copy)]
enum Fix {
    None,
    /// a zero-centred norm: 1 + w
    PlusOne,
    /// A_log -> -exp(A_log), reordered
    NegExp,
    /// rows by value head: (first row, rows a head)
    Rows(usize, usize),
    /// columns by value head (columns a head)
    Cols(usize),
}

fn plan(w: &Weights, c: &Config, ty: GType) -> Result<Vec<Item>> {
    let mut v = Vec::new();
    let shape = |n: &str| -> Result<Vec<u64>> { Ok(w.shape(n)?.iter().map(|d| *d as u64).collect()) };
    let mut push = |name: String, src: String, shape: Vec<u64>, ty: GType, fix: Fix| v.push(Item { name, src, shape, ty, fix });
    push("token_embd.weight".into(), "lm.embed_tokens.weight".into(), shape("lm.embed_tokens.weight")?, ty, Fix::None);
    push("output_norm.weight".into(), "lm.norm.weight".into(), shape("lm.norm.weight")?, GType::F32, Fix::PlusOne);
    let qk = c.lin_k_heads * c.lin_k_dim;
    for l in 0..c.layers {
        let s = |x: &str| format!("lm.layers.{l}.{x}");
        let d = |x: &str| format!("blk.{l}.{x}");
        push(d("attn_norm.weight"), s("input_layernorm.weight"), shape(&s("input_layernorm.weight"))?, GType::F32, Fix::PlusOne);
        push(d("post_attention_norm.weight"), s("post_attention_layernorm.weight"), shape(&s("post_attention_layernorm.weight"))?, GType::F32, Fix::PlusOne);
        if c.full[l] {
            for (to, from) in [("attn_q", "self_attn.q_proj"), ("attn_k", "self_attn.k_proj"), ("attn_v", "self_attn.v_proj"), ("attn_output", "self_attn.o_proj")] {
                let src = s(&format!("{from}.weight"));
                push(d(&format!("{to}.weight")), src.clone(), shape(&src)?, ty, Fix::None);
            }
            push(d("attn_q_norm.weight"), s("self_attn.q_norm.weight"), shape(&s("self_attn.q_norm.weight"))?, GType::F32, Fix::PlusOne);
            push(d("attn_k_norm.weight"), s("self_attn.k_norm.weight"), shape(&s("self_attn.k_norm.weight"))?, GType::F32, Fix::PlusOne);
        } else {
            let a = |x: &str| s(&format!("linear_attn.{x}"));
            push(d("attn_qkv.weight"), a("in_proj_qkv.weight"), shape(&a("in_proj_qkv.weight"))?, ty, Fix::Rows(2 * qk, c.lin_v_dim));
            push(d("attn_gate.weight"), a("in_proj_z.weight"), shape(&a("in_proj_z.weight"))?, ty, Fix::Rows(0, c.lin_v_dim));
            push(d("ssm_beta.weight"), a("in_proj_b.weight"), shape(&a("in_proj_b.weight"))?, GType::F32, Fix::Rows(0, 1));
            push(d("ssm_alpha.weight"), a("in_proj_a.weight"), shape(&a("in_proj_a.weight"))?, GType::F32, Fix::Rows(0, 1));
            let cs = shape(&a("conv1d.weight"))?;
            push(d("ssm_conv1d.weight"), a("conv1d.weight"), vec![cs[0], cs[2]], GType::F32, Fix::Rows(2 * qk, c.lin_v_dim));
            push(d("ssm_a"), a("A_log"), shape(&a("A_log"))?, GType::F32, Fix::NegExp);
            push(d("ssm_dt.bias"), a("dt_bias"), shape(&a("dt_bias"))?, GType::F32, Fix::Rows(0, 1));
            push(d("ssm_norm.weight"), a("norm.weight"), shape(&a("norm.weight"))?, GType::F32, Fix::None);
            push(d("ssm_out.weight"), a("out_proj.weight"), shape(&a("out_proj.weight"))?, ty, Fix::Cols(c.lin_v_dim));
        }
        push(d("ffn_gate.weight"), s("mlp.gate_proj.weight"), shape(&s("mlp.gate_proj.weight"))?, ty, Fix::None);
        push(d("ffn_up.weight"), s("mlp.up_proj.weight"), shape(&s("mlp.up_proj.weight"))?, ty, Fix::None);
        push(d("ffn_down.weight"), s("mlp.down_proj.weight"), shape(&s("mlp.down_proj.weight"))?, ty, Fix::None);
    }
    Ok(v)
}

/// The values of one tensor as the file holds them
fn convert(w: &Weights, c: &Config, it: &Item) -> Result<Vec<u8>> {
    let t = w.find(&it.src)?;
    let mut v = to_f32(t, &w.st.read(t).map_err(|e| Error(e.0))?)?;
    let order = head_order(c.lin_v_heads, c.lin_k_heads);
    let k = if it.shape.len() >= 2 { it.shape[1..].iter().product::<u64>() as usize } else { 1 };
    match it.fix {
        Fix::None => {}
        Fix::PlusOne => v.iter_mut().for_each(|x| *x += 1.0),
        Fix::NegExp => {
            v.iter_mut().for_each(|x| *x = -x.exp());
            reorder_rows(&mut v, 1, 0, 1, &order);
        }
        Fix::Rows(from, per) => reorder_rows(&mut v, k, from, per, &order),
        Fix::Cols(per) => reorder_cols(&mut v, k, per, &order),
    }
    Ok(match it.ty {
        GType::F32 => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        GType::F16 => v.iter().flat_map(|x| f32_to_f16(*x).to_le_bytes()).collect(),
        GType::Q8_0 => {
            if !k.is_multiple_of(32) {
                return Err(Error(format!("{}: rows of {k}, not whole Q8_0 blocks", it.src)));
            }
            let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).clamp(1, 16);
            let rows = v.len() / k;
            let per = rows.div_ceil(threads);
            let parts: Vec<Vec<u8>> = std::thread::scope(|s| {
                let hs: Vec<_> = v.chunks(per * k).map(|ch| s.spawn(move || {
                    let mut o = Vec::with_capacity(ch.len() / 32 * 34);
                    q8_0(ch, &mut o);
                    o
                })).collect();
                hs.into_iter().map(|h| h.join().unwrap_or_default()).collect()
            });
            parts.concat()
        }
        t => return Err(Error(format!("{}: type {} not written", it.src, t.name()))),
    })
}

/// The language model's GGUF beside the checkpoint, made when missing or of an older layout; its path
pub fn ensure(model: &Path, w: &Weights, c: &Config, ty: &str, log: &mut dyn FnMut(String)) -> Result<PathBuf> {
    let out = path_for(model, ty);
    if let Ok(g) = Gguf::open(&out) {
        if g.meta("nextcuda.haru.lm_version").and_then(|v| v.as_u64()) == Some(VERSION) {
            return Ok(out);
        }
    }
    let gt = gtype(ty)?;
    let t0 = std::time::Instant::now();
    log(format!("haru: making the language model's {ty} GGUF from the checkpoint (once): {}", out.display()));
    let items = plan(w, c, gt)?;
    let a = |k: &str| format!("qwen35.{k}");
    let meta: Vec<(String, Value)> = vec![
        ("general.architecture".into(), Value::Str("qwen35".into())),
        ("general.name".into(), Value::Str("Haru-4B language model (nextcuda)".into())),
        ("nextcuda.haru.lm_version".into(), Value::U(VERSION)),
        (a("block_count"), Value::U(c.layers as u64)),
        (a("context_length"), Value::U(262_144)),
        (a("embedding_length"), Value::U(c.hidden as u64)),
        (a("feed_forward_length"), Value::U(c.ffn as u64)),
        (a("attention.head_count"), Value::U(c.heads as u64)),
        (a("attention.head_count_kv"), Value::U(c.kv_heads as u64)),
        (a("attention.key_length"), Value::U(c.head_dim as u64)),
        (a("attention.value_length"), Value::U(c.head_dim as u64)),
        (a("rope.dimension_count"), Value::U(c.rot as u64)),
        (a("rope.freq_base"), Value::F(c.theta)),
        (a("attention.layer_norm_rms_epsilon"), Value::F(c.eps as f64)),
        (a("ssm.conv_kernel"), Value::U(c.conv_k as u64)),
        (a("ssm.state_size"), Value::U(c.lin_k_dim as u64)),
        (a("ssm.group_count"), Value::U(c.lin_k_heads as u64)),
        (a("ssm.time_step_rank"), Value::U(c.lin_v_heads as u64)),
        (a("ssm.inner_size"), Value::U((c.lin_v_heads * c.lin_v_dim) as u64)),
        (a("full_attention_interval"), Value::U(c.full.iter().position(|f| *f).map_or(4, |p| p + 1) as u64)),
    ];
    let meta_ref: Vec<(&str, Value)> = meta.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
    let dir: Vec<(String, GType, Vec<u64>)> = items.iter().map(|i| (i.name.clone(), i.ty, i.shape.clone())).collect();
    let tmp = out.with_extension("gguf.part");
    let mut wr = Writer::create(&tmp, &meta_ref, &dir).map_err(|e| Error(e.0))?;
    for (n, it) in items.iter().enumerate() {
        wr.data(&convert(w, c, it)?).map_err(|e| Error(e.0))?;
        if n % 40 == 39 {
            log(format!("haru: {} of {} tensors ({:.0} s)", n + 1, items.len(), t0.elapsed().as_secs_f64()));
        }
    }
    wr.finish().map_err(|e| Error(e.0))?;
    std::fs::rename(&tmp, &out).map_err(|e| Error(format!("{}: {e}", out.display())))?;
    log(format!("haru: the language model's GGUF made in {:.0} s", t0.elapsed().as_secs_f64()));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_heads_go_from_grouped_to_tiled() {
        // 4 value heads over 2 key heads: grouped 0,1 -> key 0, 2,3 -> key 1; tiled 0,2 -> key 0, 1,3 -> key 1
        let o = head_order(4, 2);
        assert_eq!(o, vec![0, 2, 1, 3]);
        for (j, h) in o.iter().enumerate() {
            assert_eq!(h / 2, j % 2, "new head {j} (old {h}) reads the same key head");
        }
        let mut rows: Vec<f32> = (0..8).map(|x| x as f32).collect(); // 4 heads of 2 rows, 1 wide
        reorder_rows(&mut rows, 1, 0, 2, &o);
        assert_eq!(rows, vec![0., 1., 4., 5., 2., 3., 6., 7.]);
        let mut cols: Vec<f32> = (0..4).map(|x| x as f32).collect(); // one row, 4 heads of 1 column
        reorder_cols(&mut cols, 4, 1, &o);
        assert_eq!(cols, vec![0., 2., 1., 3.]);
    }

    #[test]
    fn q8_0_blocks() {
        let x: Vec<f32> = (0..32).map(|i| i as f32 - 16.0).collect();
        let mut o = Vec::new();
        q8_0(&x, &mut o);
        assert_eq!(o.len(), 34);
        let d = 16.0f32 / 127.0;
        assert_eq!(u16::from_le_bytes([o[0], o[1]]), f32_to_f16(d));
        assert_eq!(o[2] as i8, -127);
        assert_eq!(o[2 + 16] as i8, 0);
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert_eq!(f32_to_f16(-2.0), 0xc000);
    }
}
