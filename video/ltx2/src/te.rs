//! The text encoder: Gemma-4 12B's language model (Gemma4UnifiedForConditionalGeneration's text side, as LTX-2.5
//! ships it), run over the prompt's tokens for its 49 hidden states - the embeddings, each layer's output, the final
//! norm of the last - which the connectors read. From the text cache (`cache`: Q4_K matrices), its 48 layers split
//! over the cards (half each), loaded for a clip and released after it, beside the resident DiT.
//!
//! ```text
//!   x = embed[ids] * bf16(sqrt(3840))                                  (the table on the host, bfloat16 rows)
//!   layer: x += post_attn_norm(attention(input_norm(x)))
//!          x += post_ff_norm(down(gelu_tanh(gate(h)) * up(h))),  h = pre_ff_norm(x);   x *= layer_scalar
//!   attention: q, k per-head RMS norms (weights), v an RMS norm without one; scale 1 (the kernel's 1/sqrt(d) folded
//!     into q's norm); causal. Sliding layers: 16 heads of 256 over 8 kv heads, RoPE theta 1e4 on all 256. Global
//!     layers (5, 11, ... 47): 16 heads of 512 over one kv head that is also the value (k_eq_v: no v_proj, v from k's
//!     projection before its norm), proportional RoPE: theta 1e6 on 64 of the 256 angle pairs, the rest unrotated.
//! ```
//!
//! The pipeline pads the prompt on the left to 1,024 tokens and masks the padding out; the real tokens' states are
//! the same when only they are run (the rotary angles are relative), so only they are.

use std::path::Path;
use std::sync::Arc;

use nextcuda_core::{DevBuf, Error, Gpu, Result};
use nextcuda_diffusion::kernels::{none, Dt, Nsd};
use nextcuda_gguf::{GType, Gguf};

use crate::ffi;

pub const HIDDEN: usize = 3840;
pub const HEADS: usize = 16;
pub const KV_HEADS: usize = 8;
pub const HEAD: usize = 256;
pub const GLOBAL_HEAD: usize = 512;
pub const FFN: usize = 15360;
pub const LAYERS: usize = 48;
pub const VOCAB: usize = 262144;
/// the hidden states the connectors read: the embeddings and 48 layers' outputs (the last one normed)
pub const STATES: usize = LAYERS + 1;
const EPS: f32 = 1e-6;
const THETA_LOCAL: f64 = 10_000.0;
const THETA_GLOBAL: f64 = 1_000_000.0;
/// proportional RoPE: int(0.25 * 512 / 2) rotated angle pairs
const GLOBAL_ANGLES: usize = 64;

/// layer i is a global (full) attention layer: every sixth
pub fn is_global(i: usize) -> bool {
    i % 6 == 5
}

fn ge(e: nextcuda_gguf::Error) -> Error {
    Error(e.0)
}

/// A matrix on a card as stored, [rows, k]
pub struct Mat {
    pub w: DevBuf,
    pub qtype: u32,
    pub rows: usize,
    pub k: usize,
}

impl Mat {
    pub fn load(f: &Gguf, name: &str, gpu: &Arc<Gpu>) -> Result<Mat> {
        let t = f.tensor(name).ok_or_else(|| Error(format!("{}: no tensor {name}", f.paths[0].display())))?;
        if t.shape.len() != 2 {
            return Err(Error(format!("{name}: {:?}, a matrix", t.shape)));
        }
        let w = DevBuf::new(gpu, t.bytes as usize)?;
        w.write(0, &f.read(t).map_err(ge)?)?;
        Ok(Mat { w, qtype: t.ty.code(), rows: t.shape[0] as usize, k: t.shape[1] as usize })
    }
    /// out [m, rows] (`ldo` apart) = x [m, k] half . W^T (+ bias)
    pub fn apply(&self, nsd: &Nsd, x: *const std::ffi::c_void, m: usize, out: *mut std::ffi::c_void, out_dt: Dt, ldo: usize, bias: *const f32) -> Result<()> {
        nsd.linear_q(x, Dt::F16, m, self.k, self.w.ptr(), self.qtype, self.rows, bias.cast(), out, out_dt, ldo)
    }
}

/// A float32 vector of the cache, on a card
pub fn vec_on(f: &Gguf, name: &str, gpu: &Arc<Gpu>) -> Result<DevBuf> {
    DevBuf::from_f32(gpu, &host_f32(f, name)?)
}

pub fn host_f32(f: &Gguf, name: &str) -> Result<Vec<f32>> {
    let t = f.tensor(name).ok_or_else(|| Error(format!("{}: no tensor {name}", f.paths[0].display())))?;
    if t.ty != GType::F32 {
        return Err(Error(format!("{name}: {:?}, float32 expected", t.ty)));
    }
    Ok(f.read(t).map_err(ge)?.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

struct Layer {
    global: bool,
    in_norm: DevBuf,
    post_attn_norm: DevBuf,
    pre_ff_norm: DevBuf,
    post_ff_norm: DevBuf,
    q_norm: DevBuf,
    k_norm: DevBuf,
    scalar: f32,
    q: Mat,
    k: Mat,
    v: Option<Mat>,
    o: Mat,
    gate: Mat,
    up: Mat,
    down: Mat,
}

/// One card's share of the layers
struct Part {
    gpu: Arc<Gpu>,
    nsd: Nsd,
    layers: Vec<Layer>,
    /// the first layer's index
    first: usize,
}

pub struct TextEncoder {
    parts: Vec<Part>,
    norm: DevBuf,
    /// the embedding table's bytes in the cache (bfloat16 rows), read a row at a time
    cache: Gguf,
    pub bytes: u64,
}

/// The hidden states of a prompt: [tokens, 49, 3840] float32 on the host
pub struct States {
    pub tokens: usize,
    pub v: Vec<f32>,
}

impl TextEncoder {
    /// From the text cache, the layers split evenly over `gpus` (the last part on the last card)
    pub fn load(cache: &Path, gpus: &[Arc<Gpu>], log: &mut dyn FnMut(String)) -> Result<TextEncoder> {
        let t0 = std::time::Instant::now();
        let f = Gguf::open(cache).map_err(ge)?;
        // the layers in proportion to each card's free memory (the DiT holds more of the faster one)
        let free: Vec<f64> = gpus.iter().map(|g| g.memory().map_or(1.0, |(t, f)| f.unwrap_or(t) as f64)).collect();
        let tot: f64 = free.iter().sum();
        let mut bounds = vec![0usize];
        let mut acc = 0.0;
        for (i, fr) in free.iter().enumerate() {
            acc += fr;
            bounds.push(if i + 1 == gpus.len() { LAYERS } else { ((LAYERS as f64 * acc / tot).round() as usize).min(LAYERS) });
        }
        let mut parts = Vec::new();
        let mut bytes = 0u64;
        for (pi, gpu) in gpus.iter().enumerate() {
            let (first, last) = (bounds[pi], bounds[pi + 1]);
            if first == last {
                continue;
            }
            let need: u64 = f.tensors.iter().filter(|t| {
                t.name.strip_prefix("te.").and_then(|s| s.split('.').next()).and_then(|s| s.parse::<usize>().ok()).is_some_and(|i| (first..last).contains(&i))
            }).map(|t| t.bytes).sum();
            if let (_, Some(free)) = gpu.memory()? {
                if need + (600 << 20) > free {
                    return Err(Error(format!("the text encoder's layers {first}..{last} need {:.2} GiB on {}; {:.2} GiB are free",
                                             (need + (600 << 20)) as f64 / (1u64 << 30) as f64, gpu.name, free as f64 / (1u64 << 30) as f64)));
                }
            }
            bytes += need;
            let mut layers = Vec::new();
            for i in first..last {
                let p = format!("te.{i}.");
                let m = |s: &str| Mat::load(&f, &format!("{p}{s}"), gpu);
                let v = |s: &str| vec_on(&f, &format!("{p}{s}"), gpu);
                let global = is_global(i);
                layers.push(Layer {
                    global,
                    in_norm: v("in_norm")?,
                    post_attn_norm: v("post_attn_norm")?,
                    pre_ff_norm: v("pre_ff_norm")?,
                    post_ff_norm: v("post_ff_norm")?,
                    q_norm: v("q_norm")?,
                    k_norm: v("k_norm")?,
                    scalar: host_f32(&f, &format!("{p}scalar"))?[0],
                    q: m("q")?,
                    k: m("k")?,
                    v: if global { None } else { Some(m("v")?) },
                    o: m("o")?,
                    gate: m("gate")?,
                    up: m("up")?,
                    down: m("down")?,
                });
            }
            parts.push(Part { nsd: Nsd::new(gpu)?, gpu: gpu.clone(), layers, first });
        }
        let last = &parts.last().ok_or("no GPU for the text encoder")?.gpu;
        let norm = vec_on(&f, "te.norm", last)?;
        log(format!("text encoder: Gemma-4 {LAYERS} layers, {:.2} GiB over {} card(s) in {:.1} s", bytes as f64 / (1u64 << 30) as f64, parts.len(),
                    t0.elapsed().as_secs_f64()));
        Ok(TextEncoder { parts, norm, cache: f, bytes })
    }

    /// (cos, sin) per token and angle pair for a layer kind
    fn rope(t: usize, global: bool) -> Vec<f32> {
        let (d, theta, rot) = if global { (GLOBAL_HEAD, THETA_GLOBAL, GLOBAL_ANGLES) } else { (HEAD, THETA_LOCAL, HEAD / 2) };
        let half = d / 2;
        let mut cs = vec![0f32; t * half * 2];
        for p in 0..t {
            for i in 0..half {
                let a = if i < rot { p as f64 / theta.powf(2.0 * i as f64 / d as f64) } else { 0.0 };
                // the reference's cos / sin are float32 then bfloat16 (the activations' type)
                cs[(p * half + i) * 2] = a.cos() as f32;
                cs[(p * half + i) * 2 + 1] = a.sin() as f32;
            }
        }
        cs
    }

    /// The prompt's tokens' 49 hidden states
    pub fn encode(&self, ids: &[u32]) -> Result<States> {
        let t = ids.len();
        if t == 0 {
            return Ok(States { tokens: 0, v: Vec::new() });
        }
        let mut states = vec![0f32; t * STATES * HIDDEN];
        let put = |states: &mut Vec<f32>, s: usize, x: &[f32]| {
            for r in 0..t {
                states[(r * STATES + s) * HIDDEN..(r * STATES + s + 1) * HIDDEN].copy_from_slice(&x[r * HIDDEN..(r + 1) * HIDDEN]);
            }
        };
        // the embeddings, scaled by bf16(sqrt(3840)) = 62 and rounded to bfloat16 as the reference's product is
        let et = self.cache.tensor("te.embed").ok_or("the text cache has no te.embed")?;
        let scale = quant_bf16(HIDDEN as f32).sqrt();
        let scale = quant_bf16(scale);
        let mut x0 = vec![0f32; t * HIDDEN];
        let mut row = vec![0u8; HIDDEN * 2];
        for (i, id) in ids.iter().enumerate() {
            let id = (*id as usize).min(VOCAB - 1);
            self.cache.read_into(et, (id * HIDDEN * 2) as u64, &mut row).map_err(ge)?;
            for c in 0..HIDDEN {
                x0[i * HIDDEN + c] = quant_bf16(crate::quant::bf16_val(u16::from_le_bytes([row[2 * c], row[2 * c + 1]])) * scale);
            }
        }
        put(&mut states, 0, &x0);
        let cs_l = Self::rope(t, false);
        let cs_g = Self::rope(t, true);
        let mut host = x0;
        let mut staging = Vec::new();
        let mut x_prev: Option<DevBuf> = None;
        for part in &self.parts {
            let gpu = &part.gpu;
            let nsd = &part.nsd;
            let x = DevBuf::f32(gpu, t * HIDDEN)?;
            match &x_prev {
                None => x.write(0, as_bytes(&host))?,
                Some(p) => {
                    staging.resize(p.len, 0u8);
                    x.copy_from_peer(p, &mut staging)?;
                }
            }
            let csl = DevBuf::from_f32(gpu, &cs_l)?;
            let csg = DevBuf::from_f32(gpu, &cs_g)?;
            let h = DevBuf::new(gpu, t * FFN * 2)?;
            let qw_max = HEADS * GLOBAL_HEAD + 2 * KV_HEADS * HEAD;
            let qkv = DevBuf::new(gpu, t * qw_max * 2)?;
            let att = DevBuf::new(gpu, t * HEADS * GLOBAL_HEAD * 2)?;
            let o = DevBuf::f32(gpu, t * HIDDEN)?;
            let g = DevBuf::f32(gpu, t * FFN)?;
            let u = DevBuf::f32(gpu, t * FFN)?;
            let half_at = |b: &DevBuf, n: usize| -> *mut std::ffi::c_void { b.ptr().cast::<u16>().wrapping_add(n).cast() };
            for (j, l) in part.layers.iter().enumerate() {
                let li = part.first + j;
                let (d, kv) = if l.global { (GLOBAL_HEAD, 1) } else { (HEAD, KV_HEADS) };
                let qw = HEADS * d + 2 * kv * d;
                let (ko, vo) = (HEADS * d, HEADS * d + kv * d);
                nsd.rms_norm_mod(x.ptr(), Dt::F32, t, HIDDEN, l.in_norm.ptr(), EPS, none(), none(), none(), h.ptr(), Dt::F16)?;
                l.q.apply(nsd, h.ptr(), t, qkv.ptr(), Dt::F16, qw, std::ptr::null())?;
                l.k.apply(nsd, h.ptr(), t, half_at(&qkv, ko), Dt::F16, qw, std::ptr::null())?;
                match &l.v {
                    Some(v) => v.apply(nsd, h.ptr(), t, half_at(&qkv, vo), Dt::F16, qw, std::ptr::null())?,
                    // k_eq_v: the value is the key's projection, before the key's norm and rotation
                    None => {
                        let (vb, kb) = (qkv.view(vo * 2, (t - 1) * qw * 2 + kv * d * 2)?, qkv.view(ko * 2, (t - 1) * qw * 2 + kv * d * 2)?);
                        vb.copy_2d(0, qw * 2, &kb, 0, qw * 2, kv * d * 2, t)?;
                    }
                }
                let cs = if l.global { &csg } else { &csl };
                nsd.rms_rope(qkv.ptr(), Dt::F16, t, HEADS, d, qw, l.q_norm.ptr(), EPS, cs.ptr(), d)?;
                nsd.rms_rope(half_at(&qkv, ko), Dt::F16, t, kv, d, qw, l.k_norm.ptr(), EPS, cs.ptr(), d)?;
                nsd.rms_rope(half_at(&qkv, vo), Dt::F16, t, kv, d, qw, none(), EPS, none(), 0)?;
                nsd.attention_causal(qkv.ptr(), half_at(&qkv, ko), half_at(&qkv, vo), Dt::F16, t, HEADS, kv, d, qw, qw, att.ptr())?;
                l.o.apply(nsd, att.ptr(), t, o.ptr(), Dt::F32, 0, std::ptr::null())?;
                nsd.rms_norm_mod(o.ptr(), Dt::F32, t, HIDDEN, l.post_attn_norm.ptr(), EPS, none(), none(), none(), o.ptr(), Dt::F32)?;
                nsd.gate_add(x.ptr(), Dt::F32, t, HIDDEN, o.ptr(), Dt::F32, none(), none())?;
                nsd.rms_norm_mod(x.ptr(), Dt::F32, t, HIDDEN, l.pre_ff_norm.ptr(), EPS, none(), none(), none(), h.ptr(), Dt::F16)?;
                l.gate.apply(nsd, h.ptr(), t, g.ptr(), Dt::F32, 0, std::ptr::null())?;
                l.up.apply(nsd, h.ptr(), t, u.ptr(), Dt::F32, 0, std::ptr::null())?;
                ffi::geglu(gpu, g.fp(), FFN, u.fp(), FFN, t, FFN, ffi::ACT_SCALE, h.ptr())?;
                l.down.apply(nsd, h.ptr(), t, o.ptr(), Dt::F32, 0, std::ptr::null())?;
                nsd.scale(o.ptr(), t * HIDDEN, 1.0 / ffi::ACT_SCALE)?;
                nsd.rms_norm_mod(o.ptr(), Dt::F32, t, HIDDEN, l.post_ff_norm.ptr(), EPS, none(), none(), none(), o.ptr(), Dt::F32)?;
                nsd.gate_add(x.ptr(), Dt::F32, t, HIDDEN, o.ptr(), Dt::F32, none(), none())?;
                if l.scalar != 1.0 {
                    nsd.scale(x.ptr(), t * HIDDEN, l.scalar)?;
                }
                if li + 1 < LAYERS {
                    nsd.wait()?;
                    host = x.to_f32()?;
                    put(&mut states, li + 1, &host);
                }
            }
            nsd.wait()?;
            x_prev = Some(x);
        }
        // the last state: the final norm of the last layer's output
        let x = x_prev.ok_or("no layers ran")?;
        let part = self.parts.last().unwrap();
        let fin = DevBuf::f32(&part.gpu, t * HIDDEN)?;
        part.nsd.rms_norm_mod(x.ptr(), Dt::F32, t, HIDDEN, self.norm.ptr(), EPS, none(), none(), none(), fin.ptr(), Dt::F32)?;
        part.nsd.wait()?;
        put(&mut states, LAYERS, &fin.to_f32()?);
        Ok(States { tokens: t, v: states })
    }
}

/// x rounded to bfloat16 (nearest even), as float32
pub fn quant_bf16(x: f32) -> f32 {
    let b = x.to_bits();
    let r = b.wrapping_add(0x7fff + ((b >> 16) & 1));
    f32::from_bits(r & 0xffff_0000)
}

pub fn as_bytes(v: &[f32]) -> &[u8] {
    // SAFETY: plain floats viewed as their bytes.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), v.len() * 4) }
}
