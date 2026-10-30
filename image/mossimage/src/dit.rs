//! The denoiser: MOSS-Image's DiT (pipeline_moss_image.py `MOSSImageTransformer`): 42 single-stream blocks of 3,072
//! features (24 heads x 128) over the image's tokens and the caption's, one modulation shared by every block, the
//! matrices kept on the card in the GGUF's types (Q8_0: 5.5 GB).
//!
//! ```text
//!   image tokens: each latent position's 128 channels -> video_embedder [N, 3072]
//!   caption: Qwen3.5's last hidden state [L, 4096] -> linear -> 2 text blocks (no timestep: computed once a prompt)
//!   [image | caption] -> 42 blocks -> the image's rows: rmsnorm * (1 + scale) + shift, linear -> [N, 128]
//!   block: x += gate_a * o(attention(qk-normed, rotated q k v of h)),  h = rmsnorm(x) * (1 + scale_a) + shift_a
//!          x += gate_f * down(situ(up_gate h')),                       h' = rmsnorm(x) * (1 + scale_f) + shift_f
//!   situ(h, g) = (4 tanh(g / 4) sigmoid(g)) (25 tanh(h / 25))
//! ```
//!
//! The six modulations are one linear of the timestep's embedding (no SiLU before it), the same for every block and
//! token. Positions are four axes of 16 bands each (r, f, h, w): a caption token j at (0, j, 0, 0), an image token at
//! (0, L, row - (H - 1) / 2, column - (W - 1) / 2); the rotation pairs features (i, 64 + i) - the shared kernel's own
//! order, so nothing is permuted.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::Arc;

use nextcuda_core::{DevBuf, Error, Gpu, Result};
use nextcuda_diffusion::kernels::{none, Dt, Nsd};
use nextcuda_diffusion::lora::{self, Delta};
use nextcuda_gguf::{GType, Gguf};

use crate::ffi;

pub const DIM: usize = 3072;
pub const HEADS: usize = 24;
pub const HEAD: usize = 128;
pub const FFN: usize = 8192;
/// a token's latent channels (the VAE's 32 in 2 x 2 patches)
pub const LATENT: usize = 128;
pub const TEXT_DIM: usize = 4096;
const FREQ: usize = 256;
const EPS: f32 = 1e-6;
const BANDS: usize = HEAD / 8;

fn ge(e: nextcuda_gguf::Error) -> Error {
    Error(e.0)
}

/// The ConvRot group: int8 matrices are rotated by the normalized 256 x 256 Hadamard matrix along their inputs
const GROUP: usize = 256;

/// A matrix on the card: as stored (dequantized a slice at a time for each product), or as int8 ConvRot rows with a
/// scale each for the card's int8 tensor cores (`NS_MOSS_INT8`, the default: a 1024 x 1024 step in about half the
/// time, 2026-10-26; 0: as stored)
struct Part {
    w: Option<DevBuf>,
    qtype: u32,
    rows: usize,
    k: usize,
    int8: Option<(DevBuf, DevBuf)>,
}

impl Part {
    fn apply(&self, nsd: &Nsd, x: *const std::ffi::c_void, m: usize, bias: *const std::ffi::c_void, out: *mut std::ffi::c_void, ldo: usize) -> Result<()> {
        match (&self.int8, &self.w) {
            (Some((w, sc)), _) => nsd.int8_linear(x, Dt::F16, m, self.k, w.ptr(), self.rows, sc.ptr(), bias, out, Dt::F32, GROUP, ldo),
            (None, Some(w)) => nsd.linear_q(x, Dt::F16, m, self.k, w.ptr(), self.qtype, self.rows, bias, out, Dt::F32, ldo),
            (None, None) => Err(Error("a matrix with no weights".into())),
        }
    }

    /// Parts stacked along their outputs as one int8 ConvRot matrix (one quantization of the activations serves
    /// every row; one product), the stored copies let go
    fn int8_of(nsd: &Nsd, parts: Vec<Part>) -> Result<Part> {
        let k = parts[0].k;
        if parts.iter().any(|p| p.k != k) || k % GROUP != 0 {
            return Err(Error(format!("int8 parts of {k} inputs: the same width, a multiple of {GROUP}")));
        }
        let n: usize = parts.iter().map(|p| p.rows).sum();
        let w = DevBuf::new(&nsd.gpu, n * k)?;
        let sc = DevBuf::f32(&nsd.gpu, n)?;
        let mut off = 0usize;
        for p in &parts {
            let src = p.w.as_ref().ok_or_else(|| Error("int8 from a matrix already int8".into()))?;
            nsd.int8_prepare(src.ptr(), p.qtype, 0, p.rows, k, GROUP, w.ptr().cast::<u8>().wrapping_add(off * k).cast(), sc.fp().wrapping_add(off).cast())?;
            off += p.rows;
        }
        nsd.wait()?;
        Ok(Part { w: None, qtype: 0, rows: n, k, int8: Some((w, sc)) })
    }

    /// The matrix with LoRA updates merged in, float32 on the card (W + B . A each; `int8_of` then makes its rows):
    /// the products are int8, so the updates go into the weights before they are quantized rather than beside them
    fn merged(nsd: &Nsd, p: Part, ds: &[Delta]) -> Result<Part> {
        let src = p.w.as_ref().ok_or_else(|| Error("a LoRA into a matrix already int8".into()))?;
        let w = DevBuf::f32(&nsd.gpu, p.rows * p.k)?;
        nsd.dequant(src.ptr(), p.qtype, p.rows * p.k, w.ptr(), Dt::F32)?;
        for d in ds {
            if d.n != p.rows || d.k != p.k {
                return Err(Error(format!("LoRA {}: [{}, {}], the matrix is [{}, {}]", d.target, d.n, d.k, p.rows, p.k)));
            }
            // dW [n, k] = B [n, r] . (A^T [k, r])^T
            let at: Vec<f32> = (0..d.k * d.r).map(|i| d.a[(i % d.r) * d.k + i / d.r]).collect();
            let (a, b) = (DevBuf::from_f32(&nsd.gpu, &at)?, DevBuf::from_f32(&nsd.gpu, &d.b)?);
            nsd.linear_acc(b.ptr(), Dt::F32, d.n, d.r, a.ptr(), d.k, w.ptr(), 0)?;
            nsd.wait()?;
        }
        Ok(Part { w: Some(w), qtype: 0, rows: p.rows, k: p.k, int8: None })
    }
}

struct Block {
    /// q | k | v: three parts as stored, or (int8) one matrix of their rows
    qkv: Vec<Part>,
    o: Part,
    up_gate: Part,
    down: Part,
    norm1: DevBuf,
    norm2: DevBuf,
    q_norm: DevBuf,
    k_norm: DevBuf,
}

/// A text block: no modulation, biases on every linear
struct TextBlock {
    qkv: Part,
    qkv_b: DevBuf,
    proj: Part,
    /// the output linear's bias times ACT_SCALE (its input, the attention, comes scaled)
    proj_bs: DevBuf,
    up: Part,
    up_b: DevBuf,
    gate: Part,
    gate_b: DevBuf,
    down: Part,
    down_b: DevBuf,
    norm1: DevBuf,
    norm2: DevBuf,
    q_norm: DevBuf,
    k_norm: DevBuf,
}

pub struct Dit {
    gpu: Arc<Gpu>,
    /// this card's silo (its attention, the int8 products' fused epilogues), when built; NS_SILO=0: the shared kernels
    silo: Option<&'static ffi::Silo>,
    /// the silo's attention: warps a block (0: its default; NS_MOSS_ATTN_WARPS) and how (NS_MOSS_ATTN: half - the
    /// scores accumulated in half, the default -, float, fp8 - the scores from FP8 copies of q and k -, fp8v - the
    /// values too; FP8 on a card with FP8 tensor cores, sm_89 and up), as the silo's code -1 / 0 / 1 / 2
    warps: i32,
    fp8: i32,
    /// the int8 products' fused epilogues (NS_MOSS_FUSED=0: nsd's epilogue, then the step's own kernels)
    fused: bool,
    video_w: DevBuf,
    video_b: DevBuf,
    text_proj: Part,
    text_proj_b: DevBuf,
    text_blocks: Vec<TextBlock>,
    layers: Vec<Block>,
    /// the timestep's MLP, the shared and the head modulation's linears: float32 on the host (a few products a step)
    t1: (Vec<f32>, Vec<f32>),
    t2: (Vec<f32>, Vec<f32>),
    shared: (Vec<f32>, Vec<f32>),
    head_mod: (Vec<f32>, Vec<f32>),
    bands: Vec<f32>,
    final_norm: DevBuf,
    final_w: DevBuf,
    pub bytes: u64,
}

/// A step's modulation on the card: shift_a, scale_a, gate_a, shift_f, scale_f, gate_f, then the head's shift and
/// scale, float32 [3072] each, one after the other
pub struct StepMod {
    m: DevBuf,
}

impl StepMod {
    fn at(&self, i: usize) -> *const std::ffi::c_void {
        self.m.fp().wrapping_add(i * DIM).cast_const().cast()
    }
}

/// The caption after the text embedder (float32 [L, 3072]) and its rotation table, once a prompt
pub struct Prefix {
    pub rows: usize,
    pub x: DevBuf,
    cs: Vec<f32>,
}

/// Intermediates kept for a check: the blocks' outputs asked for, image rows first
#[derive(Default)]
pub struct Grabs {
    pub at: Vec<usize>,
    pub outs: Vec<Vec<f32>>,
}

fn host_vec(f: &Gguf, name: &str) -> Result<Vec<f32>> {
    let t = f.tensor(name).ok_or_else(|| Error(format!("{}: no tensor {name}", f.paths[0].display())))?;
    let b = f.read(t).map_err(ge)?;
    Ok(match t.ty {
        GType::BF16 => b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
        GType::F32 => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        GType::F16 => b.chunks_exact(2).map(|c| nextcuda_gguf::safetensors::f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
        other => return Err(Error(format!("{name}: a {} tensor where float was expected", other.name()))),
    })
}

/// y = W x + b on the host (W [n, k] row-major), in double
fn host_linear(w: &[f32], b: &[f32], x: &[f64]) -> Vec<f64> {
    let k = x.len();
    w.chunks_exact(k).zip(b).map(|(row, bi)| row.iter().zip(x).map(|(a, c)| (*a as f64) * c).sum::<f64>() + *bi as f64).collect()
}

/// (cos, sin) per token and pair for positions (r, f, h, w): pair i turns by coordinate i / 16 times band i % 16
pub fn rope_table(bands: &[f32], pos: &[[f64; 4]]) -> Vec<f32> {
    let mut cs = vec![0f32; pos.len() * HEAD];
    for (t, p) in pos.iter().enumerate() {
        for i in 0..HEAD / 2 {
            // the reference's phase: a float32 coordinate times a float32 band
            let a = (p[i / BANDS] as f32 * bands[i % BANDS]) as f64;
            cs[t * HEAD + 2 * i] = a.cos() as f32;
            cs[t * HEAD + 2 * i + 1] = a.sin() as f32;
        }
    }
    cs
}

/// The image's positions: row y, column x of an h x w grid after a caption of `l` tokens
pub fn image_positions(h: usize, w: usize, l: usize) -> Vec<[f64; 4]> {
    let (ch, cw) = ((h as f64 - 1.0) / 2.0, (w as f64 - 1.0) / 2.0);
    (0..h * w).map(|i| [0.0, l as f64, (i / w) as f64 - ch, (i % w) as f64 - cw]).collect()
}

impl Dit {
    /// The weights from `f`; `int8`: the blocks' matrices as int8 ConvRot rows (q, k and v as one); `loras`: updates
    /// merged into the blocks' matrices first (int8 only: as stored they would have to be kept in float)
    pub fn load(f: &Gguf, nsd: &Nsd, int8: bool, loras: Vec<Delta>, log: &mut dyn FnMut(String)) -> Result<Dit> {
        let gpu = nsd.gpu.clone();
        let t0 = std::time::Instant::now();
        if !int8 && !loras.is_empty() {
            return Err(Error("a LoRA needs the int8 matrices (NS_MOSS_INT8=0 keeps them as stored, with nothing to merge into)".into()));
        }
        let merging = !loras.is_empty();
        // the updates by their module's key, each taken when its matrix is read
        let mut by_key: BTreeMap<String, Vec<Delta>> = BTreeMap::new();
        for d in loras {
            by_key.entry(lora::key(&d.target)).or_default().push(d);
        }
        let pending = RefCell::new(by_key);
        let stored = |name: &str| -> Result<Part> {
            let t = f.tensor(name).ok_or_else(|| Error(format!("{}: no tensor {name}", f.paths[0].display())))?;
            let code = t.ty.code();
            if ![0, 1, 8, 12, 13, 14, 30].contains(&code) {
                return Err(Error(format!("{name}: {} weights are not read yet (Q8_0, Q4_K, Q5_K, Q6_K, BF16, F16, F32)", t.ty.name())));
            }
            let (rows, k) = (t.shape[0] as usize, t.shape[1] as usize);
            let bytes = f.read(t).map_err(ge)?;
            let w = DevBuf::new(&gpu, bytes.len())?;
            w.write(0, &bytes)?;
            Ok(Part { w: Some(w), qtype: code, rows, k, int8: None })
        };
        let part = |name: &str| -> Result<Part> {
            let p = stored(name)?;
            match pending.borrow_mut().remove(&lora::key(name.strip_suffix(".weight").unwrap_or(name))) {
                Some(ds) => Part::merged(nsd, p, &ds),
                None => Ok(p),
            }
        };
        let vecd = |name: &str| -> Result<DevBuf> { DevBuf::from_f32(&gpu, &host_vec(f, name)?) };
        let scaled = |name: &str, s: f32| -> Result<DevBuf> { DevBuf::from_f32(&gpu, &host_vec(f, name)?.iter().map(|v| v * s).collect::<Vec<_>>()) };
        let hv = |name: &str| host_vec(f, name);
        let mat = |name: &str| -> Result<Part> {
            let p = part(name)?;
            if int8 { Part::int8_of(nsd, vec![p]) } else { Ok(p) }
        };
        let block = |p: &str| -> Result<Block> {
            let qkv = vec![part(&format!("{p}proj.q.weight"))?, part(&format!("{p}proj.k.weight"))?, part(&format!("{p}proj.v.weight"))?];
            Ok(Block {
                qkv: if int8 { vec![Part::int8_of(nsd, qkv)?] } else { qkv },
                o: mat(&format!("{p}proj.o.weight"))?,
                up_gate: mat(&format!("{p}proj.ffn_up_gate.weight"))?,
                down: mat(&format!("{p}proj.ffn_down.weight"))?,
                norm1: vecd(&format!("{p}norm1.weight"))?,
                norm2: vecd(&format!("{p}norm2.weight"))?,
                q_norm: vecd(&format!("{p}q_norm.weight"))?,
                k_norm: vecd(&format!("{p}k_norm.weight"))?,
            })
        };
        let text_block = |p: &str| -> Result<TextBlock> {
            Ok(TextBlock {
                qkv: stored(&format!("{p}qkv.weight"))?,
                qkv_b: vecd(&format!("{p}qkv.bias"))?,
                proj: stored(&format!("{p}proj.weight"))?,
                proj_bs: scaled(&format!("{p}proj.bias"), ffi::ACT_SCALE)?,
                up: stored(&format!("{p}ffn_up.weight"))?,
                up_b: vecd(&format!("{p}ffn_up.bias"))?,
                gate: stored(&format!("{p}ffn_gate.weight"))?,
                gate_b: vecd(&format!("{p}ffn_gate.bias"))?,
                down: stored(&format!("{p}ffn_down.weight"))?,
                down_b: vecd(&format!("{p}ffn_down.bias"))?,
                norm1: vecd(&format!("{p}norm1.weight"))?,
                norm2: vecd(&format!("{p}norm2.weight"))?,
                q_norm: vecd(&format!("{p}q_norm.weight"))?,
                k_norm: vecd(&format!("{p}k_norm.weight"))?,
            })
        };
        let count = |pre: &str, probe: &str| (0..).take_while(|i| f.tensor(&format!("{pre}{i}.{probe}")).is_some()).count();
        let (nl, nt) = (count("layers.", "proj.q.weight"), count("text_embedder.blocks.", "qkv.weight"));
        if nl == 0 || f.tensor("shared_block_modulation.weight").is_none() {
            return Err(Error(format!("{}: not MOSS-Image's DiT (no layers / shared_block_modulation)", f.paths[0].display())));
        }
        let silo = ffi::silo(gpu.cc);
        let env = |k: &str| std::env::var(k).ok().and_then(|v| v.trim().parse::<i32>().ok());
        let how = std::env::var("NS_MOSS_ATTN").unwrap_or_default();
        let fp8 = match how.trim() {
            "" | "half" => -1,
            "float" => 0,
            "fp8" => 1,
            "fp8v" => 2,
            other => return Err(Error(format!("NS_MOSS_ATTN={other}: half (the default), float, fp8 or fp8v"))),
        };
        if fp8 > 0 && !silo.is_some_and(|s| s.fp8) {
            return Err(Error(format!("NS_MOSS_ATTN={how}: {} has no FP8 attention ({}; FP8 needs sm_89 and up)", gpu.name,
                                     silo.map_or("no MOSS-Image silo built".to_string(), |s| format!("its silo is for sm_{}", s.sm)))));
        }
        let dit = Dit {
            gpu: gpu.clone(),
            silo,
            warps: env("NS_MOSS_ATTN_WARPS").unwrap_or(0),
            fp8,
            fused: env("NS_MOSS_FUSED").unwrap_or(1) != 0,
            video_w: vecd("video_embedder.weight")?,
            video_b: vecd("video_embedder.bias")?,
            text_proj: stored("text_embedder.proj.weight")?,
            text_proj_b: vecd("text_embedder.proj.bias")?,
            text_blocks: (0..nt).map(|i| text_block(&format!("text_embedder.blocks.{i}."))).collect::<Result<_>>()?,
            layers: (0..nl).map(|i| block(&format!("layers.{i}."))).collect::<Result<_>>()?,
            t1: (hv("time_embedding.0.weight")?, hv("time_embedding.0.bias")?),
            t2: (hv("time_embedding.2.weight")?, hv("time_embedding.2.bias")?),
            shared: (hv("shared_block_modulation.weight")?, hv("shared_block_modulation.bias")?),
            head_mod: (hv("video_head_modulation.weight")?, hv("video_head_modulation.bias")?),
            bands: hv("rope.bands")?,
            final_norm: vecd("final_norm_video.weight")?,
            final_w: vecd("final_linear_video.weight")?,
            bytes: f.tensors.iter().map(|t| t.bytes).sum(),
        };
        if let Some(k) = pending.borrow().keys().next() {
            return Err(Error(format!("a LoRA names {k} ({} modules left) that is not one of MOSS-Image's blocks' matrices (q k v o ffn_up_gate ffn_down: the ones merged into)", pending.borrow().len())));
        }
        if dit.bands.len() != BANDS || dit.text_proj.k != TEXT_DIM {
            return Err(Error(format!("{}: {} rope bands and text width {}: MOSS-Image's DiT has 16 and 4096", f.paths[0].display(), dit.bands.len(),
                                     dit.text_proj.k)));
        }
        let silo_words = match dit.silo {
            Some(s) => format!("its silo for sm_{}, attention {}", s.sm, ["half", "float", "fp8", "fp8v"][(dit.fp8 + 1) as usize]),
            None => "the shared kernels".into(),
        };
        log(format!("moss-image dit: {nl} blocks, {nt} text blocks on {} ({}{}, {silo_words}) in {:.1} s", gpu.name,
                    if int8 { "int8 ConvRot" } else { "as stored" }, if merging { ", LoRAs merged" } else { "" }, t0.elapsed().as_secs_f64()));
        Ok(dit)
    }

    pub fn layers(&self) -> usize {
        self.layers.len()
    }

    /// The modulations for timestep `t` (the scheduler's, sigma times 1000): the sinusoid [cos | sin] of 256 in
    /// double, the time MLP (linear, SiLU, linear), then the shared linear [6, 3072] and the head's [2, 3072]
    pub fn modulation(&self, t: f64) -> (Vec<f32>, Vec<f32>) {
        let half = FREQ / 2;
        let mut e = vec![0f64; FREQ];
        for i in 0..half {
            let a = t * 10000f64.powf(-(i as f64) / half as f64);
            // the reference makes it float32 before its MLP
            e[i] = a.cos() as f32 as f64;
            e[half + i] = a.sin() as f32 as f64;
        }
        let mut h = host_linear(&self.t1.0, &self.t1.1, &e);
        h.iter_mut().for_each(|x| *x /= 1.0 + (-*x).exp());
        let time = host_linear(&self.t2.0, &self.t2.1, &h);
        let m = host_linear(&self.shared.0, &self.shared.1, &time);
        let hd = host_linear(&self.head_mod.0, &self.head_mod.1, &time);
        (m.iter().map(|v| *v as f32).collect(), hd.iter().map(|v| *v as f32).collect())
    }

    /// A step's modulation on the card
    pub fn step_mod(&self, t: f64) -> Result<StepMod> {
        let (m, h) = self.modulation(t);
        let mut all = m;
        all.extend(h);
        Ok(StepMod { m: DevBuf::from_f32(&self.gpu, &all)? })
    }

    /// q, k, v (or the attention's input) from h [m, 3072] half into t.qkv (half, scaled by ACT_SCALE), normed and
    /// rotated; the attention over the m rows into t.att
    #[allow(clippy::too_many_arguments)]
    fn attend(&self, nsd: &Nsd, parts: [(&Part, *const std::ffi::c_void); 3], qn: &DevBuf, kn: &DevBuf, m: usize, cs: &DevBuf, t: &Tmp, fused: bool) -> Result<()> {
        let gpu = &self.gpu;
        let k = ffi::api()?;
        let qw = 3 * DIM;
        let half_at = |buf: &DevBuf, n: usize| -> *mut std::ffi::c_void { buf.ptr().cast::<u16>().wrapping_add(n).cast() };
        if fused {
            // one [3 * 3072, 3072] matrix and its bias: q | k | v side by side
            parts[0].0.apply(nsd, t.h.ptr(), m, parts[0].1, t.big.ptr(), 0)?;
            // SAFETY: big holds m x 9216 floats; qkv m x 9216 halves.
            ffi::check(unsafe { (k.to_half_rows)(gpu.raw(), t.big.fp(), m as i64, qw as i64, ffi::ACT_SCALE, t.qkv.ptr(), qw as i64) }, "q k v")?;
        } else {
            for (i, (p, b)) in parts.iter().enumerate() {
                p.apply(nsd, t.h.ptr(), m, *b, t.of.ptr(), 0)?;
                // SAFETY: of holds m x 3072 floats; qkv m rows of 9216 halves.
                ffi::check(unsafe { (k.to_half_rows)(gpu.raw(), t.of.fp(), m as i64, DIM as i64, ffi::ACT_SCALE, half_at(&t.qkv, i * DIM), qw as i64) }, "q k v")?;
            }
        }
        self.rope_attend(nsd, qn, kn, m, cs, t)
    }

    /// t.qkv's q and k normed and rotated, the attention over the m rows into t.att: the card's silo (its tile, FP8
    /// when asked), else the shared fused kernel
    fn rope_attend(&self, nsd: &Nsd, qn: &DevBuf, kn: &DevBuf, m: usize, cs: &DevBuf, t: &Tmp) -> Result<()> {
        let qw = 3 * DIM;
        let half_at = |buf: &DevBuf, n: usize| -> *mut std::ffi::c_void { buf.ptr().cast::<u16>().wrapping_add(n).cast() };
        // (the norm of q s with eps s^2 is the norm of q with eps: the scale is exact)
        let eps = EPS * ffi::ACT_SCALE * ffi::ACT_SCALE;
        nsd.rms_rope(t.qkv.ptr(), Dt::F16, m, HEADS, HEAD, qw, qn.ptr(), eps, cs.ptr(), HEAD)?;
        nsd.rms_rope(half_at(&t.qkv, DIM), Dt::F16, m, HEADS, HEAD, qw, kn.ptr(), eps, cs.ptr(), HEAD)?;
        let (q, k, v) = (t.qkv.ptr().cast_const(), half_at(&t.qkv, DIM).cast_const(), half_at(&t.qkv, 2 * DIM).cast_const());
        if let Some(si) = self.silo {
            // SAFETY: qkv holds m rows of 9216 halves, att m rows of 3072.
            return ffi::check(unsafe { (si.attention)(self.gpu.raw(), q, m as i64, qw as i64, k, v, m as i64, qw as i64, HEADS as i64, t.att.ptr(),
                                                      self.warps, self.fp8) }, "attention (silo)");
        }
        nsd.flash_attention(q, m, qw, k, v, m, qw, HEADS, HEADS, t.att.ptr())
    }

    /// One block with its matrices int8 and the card's silo: each product's int32 results taken by a fused epilogue
    /// straight to what comes next (q k v in half, the MLP's activation, the gated residual into x)
    #[allow(clippy::too_many_arguments)]
    fn block_fused(&self, nsd: &Nsd, si: &ffi::Silo, b: &Block, qkv: &Part, x: &DevBuf, m: usize, cs: &DevBuf, md: &StepMod, t: &Tmp) -> Result<()> {
        let g = self.gpu.raw();
        // the int8 product's rows a multiple of 4: t's buffers have the rows past m, whose results nothing reads
        let m4 = (m + 3) & !3;
        let acc = t.big.fp().cast::<i32>();
        let xs = t.xs.fp();
        let int8 = |p: &Part| p.int8.as_ref().map(|(w, s)| (w.ptr().cast_const(), s.fp().cast_const())).ok_or_else(|| Error("a matrix not int8".into()));
        let gate = |i: usize| md.at(i).cast::<f32>();
        let rows = t.rows.ptr();
        nsd.rms_norm_mod(x.ptr(), Dt::F32, m, DIM, b.norm1.ptr(), EPS, rows, md.at(1), md.at(0), t.h.ptr(), Dt::F16)?;
        let (w, ws) = int8(qkv)?;
        nsd.int8_product(t.h.ptr(), Dt::F16, m4, DIM, w, 3 * DIM, GROUP, acc.cast(), xs.cast())?;
        // SAFETY (each epilogue): acc holds m4 x N int32, xs m4 floats, ws N, the outputs m rows.
        ffi::check(unsafe { (si.epi_half)(g, acc, m as i64, (3 * DIM) as i64, xs, ws, ffi::ACT_SCALE, t.qkv.ptr(), (3 * DIM) as i64) }, "q k v")?;
        self.rope_attend(nsd, &b.q_norm, &b.k_norm, m, cs, t)?;
        let (w, ws) = int8(&b.o)?;
        nsd.int8_product(t.att.ptr(), Dt::F16, m4, DIM, w, DIM, GROUP, acc.cast(), xs.cast())?;
        // (the attention comes scaled by ACT_SCALE: its output linear's result is multiplied back)
        ffi::check(unsafe { (si.epi_gate_add)(g, acc, m as i64, DIM as i64, xs, ws, 1.0 / ffi::ACT_SCALE, gate(2), x.fp()) }, "attention out")?;
        nsd.rms_norm_mod(x.ptr(), Dt::F32, m, DIM, b.norm2.ptr(), EPS, rows, md.at(4), md.at(3), t.h.ptr(), Dt::F16)?;
        let (w, ws) = int8(&b.up_gate)?;
        nsd.int8_product(t.h.ptr(), Dt::F16, m4, DIM, w, 2 * FFN, GROUP, acc.cast(), xs.cast())?;
        ffi::check(unsafe { (si.epi_situ)(g, acc, m as i64, FFN as i64, xs, ws, t.act.ptr()) }, "situ")?;
        let (w, ws) = int8(&b.down)?;
        nsd.int8_product(t.act.ptr(), Dt::F16, m4, FFN, w, DIM, GROUP, acc.cast(), xs.cast())?;
        ffi::check(unsafe { (si.epi_gate_add)(g, acc, m as i64, DIM as i64, xs, ws, 1.0, gate(5), x.fp()) }, "mlp out")
    }

    /// One block over x float32 [m, 3072] in place, with step modulation `md`
    fn block(&self, nsd: &Nsd, b: &Block, x: &DevBuf, m: usize, cs: &DevBuf, md: &StepMod, t: &Tmp) -> Result<()> {
        let gpu = &self.gpu;
        let k = ffi::api()?;
        if let (Some(si), [qkv]) = (self.silo, b.qkv.as_slice()) {
            if qkv.int8.is_some() && self.fused {
                return self.block_fused(nsd, si, b, qkv, x, m, cs, md, t);
            }
        }
        let rows = t.rows.ptr();
        nsd.rms_norm_mod(x.ptr(), Dt::F32, m, DIM, b.norm1.ptr(), EPS, rows, md.at(1), md.at(0), t.h.ptr(), Dt::F16)?;
        match b.qkv.as_slice() {
            [qkv] => self.attend(nsd, [(qkv, none()), (qkv, none()), (qkv, none())], &b.q_norm, &b.k_norm, m, cs, t, true)?,
            [q, k, v] => self.attend(nsd, [(q, none()), (k, none()), (v, none())], &b.q_norm, &b.k_norm, m, cs, t, false)?,
            _ => return Err(Error("a block's q k v".into())),
        }
        b.o.apply(nsd, t.att.ptr(), m, none(), t.of.ptr(), 0)?;
        nsd.scale(t.of.ptr(), m * DIM, 1.0 / ffi::ACT_SCALE)?;
        nsd.gate_add(x.ptr(), Dt::F32, m, DIM, t.of.ptr(), Dt::F32, rows, md.at(2))?;
        nsd.rms_norm_mod(x.ptr(), Dt::F32, m, DIM, b.norm2.ptr(), EPS, rows, md.at(4), md.at(3), t.h.ptr(), Dt::F16)?;
        b.up_gate.apply(nsd, t.h.ptr(), m, none(), t.big.ptr(), 0)?;
        // SAFETY: big holds m x 2 * 8192 floats (hidden | gate), act m x 8192 halves.
        ffi::check(unsafe { (k.situ)(gpu.raw(), t.big.fp(), m as i64, FFN as i64, (2 * FFN) as i64, t.act.ptr()) }, "situ")?;
        b.down.apply(nsd, t.act.ptr(), m, none(), t.of.ptr(), 0)?;
        nsd.gate_add(x.ptr(), Dt::F32, m, DIM, t.of.ptr(), Dt::F32, rows, md.at(5))
    }

    /// The caption's rows: Qwen3.5's last hidden state (float32 [L, 4096]) through the text embedder's linear and its
    /// two blocks
    pub fn prefix(&self, nsd: &Nsd, hidden: &[f32]) -> Result<Prefix> {
        let gpu = &self.gpu;
        let k = ffi::api()?;
        let l = hidden.len() / TEXT_DIM;
        if l == 0 {
            return Err(Error("an empty caption".into()));
        }
        let hf = DevBuf::from_f32(gpu, hidden)?;
        let hh = DevBuf::new(gpu, hidden.len() * 2)?;
        // SAFETY: L x 4096 values each way.
        ffi::check(unsafe { (k.to_half)(gpu.raw(), hf.fp(), hh.ptr(), hidden.len() as i64) }, "to half")?;
        let x = DevBuf::f32(gpu, l * DIM)?;
        self.text_proj.apply(nsd, hh.ptr(), l, self.text_proj_b.fp().cast(), x.ptr(), 0)?;
        let pos: Vec<[f64; 4]> = (0..l).map(|j| [0.0, j as f64, 0.0, 0.0]).collect();
        let cs = rope_table(&self.bands, &pos);
        let csd = DevBuf::from_f32(gpu, &cs)?;
        let t = Tmp::new(gpu, l)?;
        for b in &self.text_blocks {
            nsd.rms_norm_mod(x.ptr(), Dt::F32, l, DIM, b.norm1.ptr(), EPS, none(), none(), none(), t.h.ptr(), Dt::F16)?;
            self.attend(nsd, [(&b.qkv, b.qkv_b.fp().cast()), (&b.qkv, none()), (&b.qkv, none())], &b.q_norm, &b.k_norm, l, &csd, &t, true)?;
            b.proj.apply(nsd, t.att.ptr(), l, b.proj_bs.fp().cast(), t.of.ptr(), 0)?;
            nsd.scale(t.of.ptr(), l * DIM, 1.0 / ffi::ACT_SCALE)?;
            nsd.gate_add(x.ptr(), Dt::F32, l, DIM, t.of.ptr(), Dt::F32, none(), none())?;
            nsd.rms_norm_mod(x.ptr(), Dt::F32, l, DIM, b.norm2.ptr(), EPS, none(), none(), none(), t.h.ptr(), Dt::F16)?;
            b.up.apply(nsd, t.h.ptr(), l, b.up_b.fp().cast(), t.big.ptr(), 2 * FFN)?;
            let gate_at: *mut std::ffi::c_void = t.big.fp().wrapping_add(FFN).cast();
            b.gate.apply(nsd, t.h.ptr(), l, b.gate_b.fp().cast(), gate_at, 2 * FFN)?;
            // SAFETY: as in a block.
            ffi::check(unsafe { (k.situ)(gpu.raw(), t.big.fp(), l as i64, FFN as i64, (2 * FFN) as i64, t.act.ptr()) }, "situ")?;
            b.down.apply(nsd, t.act.ptr(), l, b.down_b.fp().cast(), t.of.ptr(), 0)?;
            nsd.gate_add(x.ptr(), Dt::F32, l, DIM, t.of.ptr(), Dt::F32, none(), none())?;
        }
        nsd.wait()?;
        Ok(Prefix { rows: l, x, cs })
    }

    /// The model's velocity for latent tokens `lat` (float32 [h * w, 128]: a token a position, row-major) at a step:
    /// float32 [h * w, 128]. `skip`: a block left out (the unconditional branch's 22)
    #[allow(clippy::too_many_arguments)]
    pub fn velocity(&self, nsd: &Nsd, pre: &Prefix, lat: &DevBuf, hw: (usize, usize), md: &StepMod, skip: Option<usize>, mut grabs: Option<&mut Grabs>)
                    -> Result<DevBuf> {
        let gpu = &self.gpu;
        let n = hw.0 * hw.1;
        let c = pre.rows;
        let s = n + c;
        let x = DevBuf::f32(gpu, s * DIM)?;
        nsd.linear(lat.ptr(), Dt::F32, n, LATENT, self.video_w.ptr(), DIM, self.video_b.fp().cast(), x.ptr(), Dt::F32)?;
        // [image | caption]
        x.copy_within(n * DIM * 4, &pre.x, 0, c * DIM * 4)?;
        let mut cs = rope_table(&self.bands, &image_positions(hw.0, hw.1, c));
        cs.extend_from_slice(&pre.cs);
        let cs = DevBuf::from_f32(gpu, &cs)?;
        let t = Tmp::new(gpu, s)?;
        for (i, b) in self.layers.iter().enumerate() {
            if skip != Some(i) {
                self.block(nsd, b, &x, s, &cs, md, &t)?;
            }
            if let Some(g) = grabs.as_deref_mut() {
                if g.at.contains(&i) {
                    nsd.wait()?;
                    g.outs.push(x.to_f32()?);
                }
            }
        }
        // the head on the image's rows
        let hf = DevBuf::f32(gpu, n * DIM)?;
        nsd.rms_norm_mod(x.ptr(), Dt::F32, n, DIM, self.final_norm.ptr(), EPS, t.rows.ptr(), md.at(7), md.at(6), hf.ptr(), Dt::F32)?;
        let v = DevBuf::f32(gpu, n * LATENT)?;
        nsd.linear(hf.ptr(), Dt::F32, n, DIM, self.final_w.ptr(), LATENT, none(), v.ptr(), Dt::F32)?;
        nsd.wait()?;
        Ok(v)
    }
}

/// A pass's buffers, for up to `rows` tokens
struct Tmp {
    h: DevBuf,
    qkv: DevBuf,
    att: DevBuf,
    of: DevBuf,
    big: DevBuf,
    act: DevBuf,
    /// the int8 product's row scales
    xs: DevBuf,
    /// zeros: every row takes the modulation's one row
    rows: DevBuf,
}

impl Tmp {
    fn new(gpu: &Arc<Gpu>, rows: usize) -> Result<Tmp> {
        let r = DevBuf::new(gpu, rows * 4)?;
        r.write(0, &vec![0u8; rows * 4])?;
        // rows a multiple of 4: the int8 product's (its rows past the m asked for are read, never used)
        let rows = (rows + 3) & !3;
        Ok(Tmp {
            h: DevBuf::new(gpu, rows * DIM * 2)?,
            qkv: DevBuf::new(gpu, rows * 3 * DIM * 2)?,
            att: DevBuf::new(gpu, rows * DIM * 2)?,
            of: DevBuf::f32(gpu, rows * DIM)?,
            big: DevBuf::f32(gpu, rows * 2 * FFN)?,
            act: DevBuf::new(gpu, rows * FFN * 2)?,
            xs: DevBuf::f32(gpu, rows)?,
            rows: r,
        })
    }
}

/// Latents [128, h, w] as tokens [h * w, 128]
pub fn tokens(lat: &[f32], n: usize) -> Vec<f32> {
    let mut out = vec![0f32; lat.len()];
    for c in 0..LATENT {
        for p in 0..n {
            out[p * LATENT + c] = lat[c * n + p];
        }
    }
    out
}

/// The inverse: tokens [h * w, 128] to latents [128, h, w]
pub fn planes(tok: &[f32], n: usize) -> Vec<f32> {
    let mut out = vec![0f32; tok.len()];
    for c in 0..LATENT {
        for p in 0..n {
            out[c * n + p] = tok[p * LATENT + c];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    #[test]
    fn tokens_go_and_come_back() {
        let n = 6;
        let lat: Vec<f32> = (0..128 * n).map(|i| i as f32).collect();
        let t = super::tokens(&lat, n);
        // token 1's channel 2 is plane 2's position 1
        assert_eq!(t[128 + 2], (2 * n + 1) as f32);
        assert_eq!(super::planes(&t, n), lat);
    }

    #[test]
    fn rope_pairs_follow_the_axes() {
        let bands: Vec<f32> = (0..16).map(|b| 1.0 / 10000f32.powf(b as f32 / 16.0)).collect();
        let cs = super::rope_table(&bands, &[[0.0, 3.0, 0.0, 0.0], [0.0, 0.0, -1.5, 2.0]]);
        // token 0: the r axis (pairs 0..16) does not turn; the f axis's first pair (16) by 3
        assert!((cs[0] - 1.0).abs() < 1e-6 && cs[1].abs() < 1e-6);
        assert!((cs[2 * 16] - 3f32.cos()).abs() < 1e-6 && (cs[2 * 16 + 1] - 3f32.sin()).abs() < 1e-6);
        // token 1: h's first pair (32) by -1.5, w's (48) by 2
        assert!((cs[128 + 2 * 32 + 1] - (-1.5f32).sin()).abs() < 1e-6);
        assert!((cs[128 + 2 * 48] - 2f32.cos()).abs() < 1e-6);
        let p = super::image_positions(2, 3, 7);
        assert_eq!(p[0], [0.0, 7.0, -0.5, -1.0]);
        assert_eq!(p[5], [0.0, 7.0, 0.5, 1.0]);
    }
}
