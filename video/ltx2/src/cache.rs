//! The text stage's weights, quantized once into a GGUF beside the checkpoint (`nextcuda-text.gguf` in the components'
//! directory) and read from there after: Gemma-4's 48 layers in Q4_K (5.6 GiB - what fits beside the resident DiT;
//! the platform's own engine runs it in 4 bits too), its embedding table in bfloat16 (looked up on the host), and the
//! connectors (the two 188,160-wide projections and two 8-layer transformers) in Q8_0. Built from the bfloat16
//! safetensors on the first load (a few minutes on the host); rebuilt when the file is missing or of another version.
//!
//! Names: `te.embed`, `te.norm`, `te.N.{q,k,v,o,gate,up,down}` and their norms; `cv.*` / `ca.*` the video / audio
//! connector (`proj`, `proj_b`, `reg`, `N.{q,k,v,o,g,ff1,ff2}` with `_b` biases, `N.{qn,kn}`). Two folds on the way:
//! Gemma's q_norm weight times sqrt(head_dim) (its attention has no 1/sqrt(d); the kernel's has), and the connectors'
//! projection columns from the checkpoint's (feature, layer) order to (layer, feature), times their modality scale.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use nextcuda_core::{Error, Result};
use nextcuda_gguf::safetensors::SafeTensors;
use nextcuda_gguf::writer::Writer;
use nextcuda_gguf::{GType, Gguf, Value};

use crate::quant;
use crate::te::{FFN, HIDDEN, LAYERS, STATES};

pub const VERSION: u64 = 1;
pub const FILE: &str = "nextcuda-text.gguf";
pub const FILE_Q8: &str = "nextcuda-text-q8.gguf";

fn ge(e: nextcuda_gguf::Error) -> Error {
    Error(e.0)
}

/// A checkpoint over one or more safetensors files (an index's weight map, or one file)
pub struct Shards {
    files: Vec<SafeTensors>,
    map: BTreeMap<String, usize>,
}

impl Shards {
    /// `dir`'s `*.safetensors.index.json` and its files, else every `.safetensors` file in it
    pub fn open(dir: &Path) -> Result<Shards> {
        let entries: Vec<PathBuf> = std::fs::read_dir(dir).map_err(|e| Error(format!("{}: {e}", dir.display())))?
            .filter_map(|e| e.ok().map(|e| e.path())).collect();
        let index = entries.iter().find(|p| p.to_string_lossy().ends_with(".safetensors.index.json"));
        let names: Vec<PathBuf> = match index {
            Some(ix) => {
                let j: serde_json::Value = serde_json::from_slice(&std::fs::read(ix).map_err(|e| Error(format!("{}: {e}", ix.display())))?).map_err(|e| Error(format!("{}: {e}", ix.display())))?;
                let mut v: Vec<String> = j["weight_map"].as_object().ok_or("an index without a weight_map")?.values()
                    .filter_map(|x| x.as_str().map(str::to_string)).collect();
                v.sort();
                v.dedup();
                v.into_iter().map(|f| dir.join(f)).collect()
            }
            None => {
                let mut v: Vec<PathBuf> = entries.into_iter().filter(|p| p.extension().is_some_and(|x| x == "safetensors")).collect();
                v.sort();
                v
            }
        };
        if names.is_empty() {
            return Err(Error(format!("{}: no safetensors", dir.display())));
        }
        let files = names.iter().map(|p| SafeTensors::open(p).map_err(ge)).collect::<Result<Vec<_>>>()?;
        let mut map = BTreeMap::new();
        for (i, f) in files.iter().enumerate() {
            for t in &f.tensors {
                map.insert(t.name.clone(), i);
            }
        }
        Ok(Shards { files, map })
    }

    pub fn has(&self, name: &str) -> bool {
        self.map.contains_key(name)
    }

    /// (shape, bfloat16 bytes)
    pub fn bf16(&self, name: &str) -> Result<(Vec<u64>, Vec<u8>)> {
        let f = &self.files[*self.map.get(name).ok_or_else(|| Error(format!("no tensor {name}")))?];
        let t = f.need(name).map_err(ge)?;
        if t.dtype != "BF16" {
            return Err(Error(format!("{name}: {}, bfloat16 expected", t.dtype)));
        }
        Ok((t.shape.clone(), f.read(t).map_err(ge)?))
    }

    pub fn f32(&self, name: &str) -> Result<Vec<f32>> {
        let f = &self.files[*self.map.get(name).ok_or_else(|| Error(format!("no tensor {name}")))?];
        f.f32(name).map_err(ge)
    }
}

/// One tensor of the cache: its name, type, shape and bytes (made when written)
struct Out {
    name: String,
    ty: GType,
    shape: Vec<u64>,
    make: Box<dyn FnOnce() -> Result<Vec<u8>> + 'static>,
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// The cache for the checkpoint under `components` (text_encoder/, connectors/), made if it is not there
/// `q8`: Gemma's matrices in Q8_0 instead (`nextcuda-text-q8.gguf`, 10.7 GiB: the checks' exact form - it does not fit
/// beside the DiT)
pub fn ensure(components: &Path, threads: usize, q8: bool, log: &mut dyn FnMut(String)) -> Result<PathBuf> {
    let path = components.join(if q8 { FILE_Q8 } else { FILE });
    if path.exists() {
        if let Ok(g) = Gguf::open(&path) {
            if g.meta("nextcuda.ltx2.text_version").and_then(Value::as_u64) == Some(VERSION) {
                return Ok(path);
            }
        }
        log(format!("{}: another version, rebuilt", path.display()));
    }
    build(components, &path, threads, q8, log)?;
    Ok(path)
}

fn build(components: &Path, path: &Path, threads: usize, q8: bool, log: &mut dyn FnMut(String)) -> Result<()> {
    let t0 = std::time::Instant::now();
    log(format!("quantizing the text encoder and connectors into {} (once)", path.display()));
    let te = std::sync::Arc::new(Shards::open(&components.join("text_encoder"))?);
    let cn = std::sync::Arc::new(Shards::open(&components.join("connectors"))?);
    let mut outs: Vec<Out> = Vec::new();
    // a matrix quantized (with a per-row transform), a vector as float32, a tensor's bytes as they are
    fn mat(outs: &mut Vec<Out>, src: &std::sync::Arc<Shards>, from: &str, to: &str, ty: GType, threads: usize, shape: [u64; 2],
           f: Option<std::sync::Arc<dyn Fn(usize, &mut Vec<f32>) + Send + Sync>>) {
        let (s, from) = (src.clone(), from.to_string());
        outs.push(Out { name: to.into(), ty, shape: shape.to_vec(), make: Box::new(move || {
            let (sh, b) = s.bf16(&from)?;
            if sh != shape {
                return Err(Error(format!("{from}: {sh:?}, {shape:?} expected")));
            }
            let id: std::sync::Arc<dyn Fn(usize, &mut Vec<f32>) + Send + Sync> = f.unwrap_or_else(|| std::sync::Arc::new(|_, _| {}));
            Ok(quant::quantize_rows(&b, shape[0] as usize, shape[1] as usize, ty, threads, &*id))
        }) });
    }
    fn vecf(outs: &mut Vec<Out>, src: &std::sync::Arc<Shards>, from: &str, to: &str, n: u64, scale: f32) {
        let (s, from) = (src.clone(), from.to_string());
        outs.push(Out { name: to.into(), ty: GType::F32, shape: vec![n], make: Box::new(move || {
            let v = s.f32(&from)?;
            if v.len() as u64 != n {
                return Err(Error(format!("{from}: {} values, {n} expected", v.len())));
            }
            Ok(f32_bytes(&v.iter().map(|x| x * scale).collect::<Vec<_>>()))
        }) });
    }
    let q4 = if q8 { GType::Q8_0 } else { GType::Q4K };
    let p = "model.language_model.";
    {
        let s = te.clone();
        outs.push(Out { name: "te.embed".into(), ty: GType::BF16, shape: vec![crate::te::VOCAB as u64, HIDDEN as u64], make: Box::new(move || {
            Ok(s.bf16("model.language_model.embed_tokens.weight")?.1)
        }) });
    }
    vecf(&mut outs, &te, &format!("{p}norm.weight"), "te.norm", HIDDEN as u64, 1.0);
    for i in 0..LAYERS {
        let l = format!("{p}layers.{i}.");
        let g = crate::te::is_global(i);
        let (hd, kv) = if g { (crate::te::GLOBAL_HEAD, 1) } else { (crate::te::HEAD, crate::te::KV_HEADS) };
        let (h, hd64, kv64) = (HIDDEN as u64, hd as u64, kv as u64);
        let heads = crate::te::HEADS as u64;
        for (from, to) in [("input_layernorm", "in_norm"), ("post_attention_layernorm", "post_attn_norm"), ("pre_feedforward_layernorm", "pre_ff_norm"),
                           ("post_feedforward_layernorm", "post_ff_norm")] {
            vecf(&mut outs, &te, &format!("{l}{from}.weight"), &format!("te.{i}.{to}"), h, 1.0);
        }
        vecf(&mut outs, &te, &format!("{l}self_attn.q_norm.weight"), &format!("te.{i}.q_norm"), hd64, (hd as f32).sqrt());
        vecf(&mut outs, &te, &format!("{l}self_attn.k_norm.weight"), &format!("te.{i}.k_norm"), hd64, 1.0);
        vecf(&mut outs, &te, &format!("{l}layer_scalar"), &format!("te.{i}.scalar"), 1, 1.0);
        mat(&mut outs, &te, &format!("{l}self_attn.q_proj.weight"), &format!("te.{i}.q"), q4, threads, [heads * hd64, h], None);
        mat(&mut outs, &te, &format!("{l}self_attn.k_proj.weight"), &format!("te.{i}.k"), q4, threads, [kv64 * hd64, h], None);
        if !g {
            mat(&mut outs, &te, &format!("{l}self_attn.v_proj.weight"), &format!("te.{i}.v"), q4, threads, [kv64 * hd64, h], None);
        }
        mat(&mut outs, &te, &format!("{l}self_attn.o_proj.weight"), &format!("te.{i}.o"), q4, threads, [h, heads * hd64], None);
        mat(&mut outs, &te, &format!("{l}mlp.gate_proj.weight"), &format!("te.{i}.gate"), q4, threads, [FFN as u64, h], None);
        mat(&mut outs, &te, &format!("{l}mlp.up_proj.weight"), &format!("te.{i}.up"), q4, threads, [FFN as u64, h], None);
        mat(&mut outs, &te, &format!("{l}mlp.down_proj.weight"), &format!("te.{i}.down"), q4, threads, [h, FFN as u64], None);
    }
    let q8 = GType::Q8_0;
    let wide = (HIDDEN * STATES) as u64;
    for (m, short, dim) in [("video", "cv", 4096u64), ("audio", "ca", 2048u64)] {
        // the checkpoint's columns are (feature c, state l) at c * 49 + l; ours (l, c) at l * 3840 + c, times the scale
        let scale = (dim as f32 / HIDDEN as f32).sqrt();
        let perm: std::sync::Arc<dyn Fn(usize, &mut Vec<f32>) + Send + Sync> = std::sync::Arc::new(move |_, row: &mut Vec<f32>| {
            let src = row.clone();
            for l in 0..STATES {
                for c in 0..HIDDEN {
                    row[l * HIDDEN + c] = src[c * STATES + l] * scale;
                }
            }
        });
        mat(&mut outs, &cn, &format!("{m}_text_proj_in.weight"), &format!("{short}.proj"), q8, threads, [dim, wide], Some(perm));
        vecf(&mut outs, &cn, &format!("{m}_text_proj_in.bias"), &format!("{short}.proj_b"), dim, 1.0);
        {
            let s = cn.clone();
            let name = format!("{m}_connector.learnable_registers");
            outs.push(Out { name: format!("{short}.reg"), ty: GType::F32, shape: vec![crate::conn::REGISTERS as u64, dim], make: Box::new(move || {
                Ok(f32_bytes(&s.f32(&name)?))
            }) });
        }
        for b in 0..crate::conn::LAYERS {
            let c = format!("{m}_connector.transformer_blocks.{b}.");
            let o = format!("{short}.{b}.");
            for (from, to) in [("attn1.to_q", "q"), ("attn1.to_k", "k"), ("attn1.to_v", "v"), ("attn1.to_out.0", "o")] {
                mat(&mut outs, &cn, &format!("{c}{from}.weight"), &format!("{o}{to}"), q8, threads, [dim, dim], None);
                vecf(&mut outs, &cn, &format!("{c}{from}.bias"), &format!("{o}{to}_b"), dim, 1.0);
            }
            mat(&mut outs, &cn, &format!("{c}attn1.to_gate_logits.weight"), &format!("{o}g"), q8, threads, [crate::conn::HEADS as u64, dim], None);
            vecf(&mut outs, &cn, &format!("{c}attn1.to_gate_logits.bias"), &format!("{o}g_b"), crate::conn::HEADS as u64, 1.0);
            vecf(&mut outs, &cn, &format!("{c}attn1.norm_q.weight"), &format!("{o}qn"), dim, 1.0);
            vecf(&mut outs, &cn, &format!("{c}attn1.norm_k.weight"), &format!("{o}kn"), dim, 1.0);
            mat(&mut outs, &cn, &format!("{c}ff.net.0.proj.weight"), &format!("{o}ff1"), q8, threads, [4 * dim, dim], None);
            vecf(&mut outs, &cn, &format!("{c}ff.net.0.proj.bias"), &format!("{o}ff1_b"), 4 * dim, 1.0);
            mat(&mut outs, &cn, &format!("{c}ff.net.2.weight"), &format!("{o}ff2"), q8, threads, [dim, 4 * dim], None);
            vecf(&mut outs, &cn, &format!("{c}ff.net.2.bias"), &format!("{o}ff2_b"), dim, 1.0);
        }
    }
    let tmp = path.with_extension("gguf.part");
    let dir: Vec<(String, GType, Vec<u64>)> = outs.iter().map(|o| (o.name.clone(), o.ty, o.shape.clone())).collect();
    let mut w = Writer::create(&tmp, &[("general.architecture", Value::Str("ltx2-text".into())), ("nextcuda.ltx2.text_version", Value::U(VERSION))], &dir)
        .map_err(ge)?;
    let n = outs.len();
    for (i, o) in outs.into_iter().enumerate() {
        let name = o.name.clone();
        let b = (o.make)().map_err(|e| Error(format!("{name}: {e}")))?;
        w.data(&b).map_err(ge)?;
        if i % 40 == 0 {
            log(format!("  {i}/{n} tensors ({name}), {:.0} s", t0.elapsed().as_secs_f64()));
        }
    }
    w.finish().map_err(ge)?;
    std::fs::rename(&tmp, path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
    log(format!("text cache written in {:.0} s", t0.elapsed().as_secs_f64()));
    Ok(())
}
