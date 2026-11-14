//! The denoiser: Qwen-Image's dual-stream MMDiT (diffusers' QwenImageTransformer2DModel; Qwen-Image 2512 is its
//! checkpoint), 60 blocks of 3,072 features (24 heads x 128), its matrices kept in the type the GGUF stores them.
//!
//! The text and the picture each have their own weights in every block and meet in one joint attention:
//!
//! ```text
//!   shift1 scale1 gate1 shift2 scale2 gate2 = (img|txt)_mod(silu(temb))      per stream, from the timestep alone
//!   h = layernorm(x) * (1 + scale1) + shift1            q k v = h W + b; q, k: per-head rmsnorm + 3-axis RoPE
//!   [text | picture] attention over both streams' rows; out: to_add_out (text), to_out.0 (picture)
//!   x += gate1 * out
//!   x += gate2 * fc2(gelu_tanh(fc1(layernorm(x) * (1 + scale2) + shift2)))
//! ```
//!
//! The picture sits at frame 0 with its rows and columns centred on zero; the text's tokens follow from
//! max(h, w) / 2 on all three axes. The RoPE pairs features (2i, 2i + 1); the shared kernel rotates (i, 64 + i), so
//! q's and k's output rows (their biases and norms) are permuted at load - scores do not change, v is untouched.
//!
//! 20B parameters are 13.2 GB at Q4_K_M: more than a 12 GB card. The blocks that fit (the activations' room kept)
//! stay on the card; the others are kept in pinned host memory and copied into a staging buffer when their turn
//! comes - once a step for every pass (a guided step's two passes run each block back to back).

use std::sync::{Arc, Mutex};

use nextcuda_core::{DevBuf, Error, Gpu, HostBuf, Result};
use nextcuda_diffusion::kernels::{none, Dt, Nsd};
use nextcuda_diffusion::lora::{self, Delta};
use nextcuda_gguf::{GType, Gguf};

use crate::ffi;

pub const DIM: usize = 3072;
pub const HEADS: usize = 24;
pub const HEAD: usize = 128;
pub const FFN: usize = 12288;
/// the latents a token carries: 16 channels x 2 x 2
pub const LATENT: usize = 64;
/// the text encoder's width (Qwen2.5-VL 7B)
pub const CONTEXT: usize = 3584;
const EPS: f32 = 1e-6;
const AXES: [usize; 3] = [16, 56, 56];
/// v's scale into the half attention (its values pass 65,504 in the late blocks); the output linears undo it
const VS: f32 = 64.0;
/// rows the feed-forward runs at once
const FFN_ROWS: usize = 4096;

/// q / k row order: pair (2i, 2i + 1) -> (i, 64 + i)
fn rope_perm(j: usize) -> usize {
    if j < HEAD / 2 {
        2 * j
    } else {
        2 * (j - HEAD / 2) + 1
    }
}

fn ge(e: nextcuda_gguf::Error) -> Error {
    Error(e.0)
}

/// A LoRA's update of a matrix on the card: A [r, k] in the matrix's input type (bfloat16 for fc2, whose input is the
/// feed-forward's; else half), B [n, r] float32 (the outputs are float32; B carries the scale)
struct LoraPair {
    a: DevBuf,
    b: DevBuf,
    r: usize,
    dt: Dt,
}

/// One matrix of a block: where its stored bytes sit in the block's bytes, its type and shape, its float32 bias
struct Lin {
    key: String,
    off: usize,
    qtype: u32,
    n: usize,
    k: usize,
    bias: DevBuf,
    perm: bool,
    /// the bias divided by VS: to_out.0 and to_add_out, whose input is the attention's scaled output
    bias_v: Option<DevBuf>,
    /// its input in bfloat16 (fc2: the feed-forward's activations pass half's range), else half
    bf16: bool,
    loras: Mutex<Vec<LoraPair>>,
}

/// Where a block's bytes are
enum Home {
    Card(DevBuf),
    Host(HostBuf),
}

struct Block {
    home: Home,
    bytes: usize,
    img_mod: Lin,
    txt_mod: Lin,
    /// q k v of the picture, then of the text
    qkv: [Lin; 3],
    add_qkv: [Lin; 3],
    out: Lin,
    add_out: Lin,
    img_fc1: Lin,
    img_fc2: Lin,
    txt_fc1: Lin,
    txt_fc2: Lin,
    norm_q: DevBuf,
    norm_k: DevBuf,
    norm_aq: DevBuf,
    norm_ak: DevBuf,
}

/// One pass's text: its encoder states float32 [T, 3584] on the card
pub struct Text {
    pub emb: DevBuf,
    pub tokens: usize,
}

pub struct Dit {
    gpu: Arc<Gpu>,
    blocks: Vec<Block>,
    /// the staging buffer a streamed block is copied into (the largest block's bytes), when any block is streamed
    staging: Option<DevBuf>,
    // the small linears, float32
    img_in: (DevBuf, DevBuf),
    txt_norm: DevBuf,
    txt_in: (DevBuf, DevBuf),
    t1: (DevBuf, DevBuf),
    t2: (DevBuf, DevBuf),
    norm_out: (DevBuf, DevBuf),
    proj_out: (DevBuf, DevBuf),
    pub resident: usize,
}

/// A block's matrices and vectors by name (diffusers' names, as the GGUF keeps them)
const MATS: [&str; 14] = [
    "img_mod.1", "txt_mod.1", "attn.to_q", "attn.to_k", "attn.to_v", "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "attn.to_out.0",
    "attn.to_add_out", "img_mlp.net.0.proj", "img_mlp.net.2", "txt_mlp.net.0.proj", "txt_mlp.net.2",
];

fn half_at(buf: &DevBuf, at: usize) -> *mut std::ffi::c_void {
    buf.ptr().cast::<u16>().wrapping_add(at).cast()
}

impl Dit {
    /// The weights of `f`: the first blocks on the card while `budget` bytes last, the others in host memory
    pub fn load(f: &Gguf, nsd: &Nsd, budget: u64, log: &mut dyn FnMut(String)) -> Result<Dit> {
        let gpu = nsd.gpu.clone();
        let t0 = std::time::Instant::now();
        let tensor = |name: &str| f.tensor(name).ok_or_else(|| Error(format!("{}: no tensor {name}", f.paths[0].display())));
        let host_vec = |name: &str| -> Result<Vec<f32>> {
            let t = tensor(name)?;
            let b = f.read(t).map_err(ge)?;
            Ok(match t.ty {
                GType::BF16 => b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
                GType::F32 => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
                GType::F16 => b.chunks_exact(2).map(|c| nextcuda_gguf::safetensors::f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
                other => return Err(Error(format!("{name}: a {} tensor where a float one was expected", other.name()))),
            })
        };
        let permuted = |v: Vec<f32>| -> Vec<f32> { (0..v.len()).map(|j| v[(j / HEAD) * HEAD + rope_perm(j % HEAD)]).collect() };
        let small = |name: &str| -> Result<(DevBuf, DevBuf)> {
            Ok((DevBuf::from_f32(&gpu, &host_vec(&format!("{name}.weight"))?)?, DevBuf::from_f32(&gpu, &host_vec(&format!("{name}.bias"))?)?))
        };
        let n_blocks = (0..).take_while(|i| f.tensor(&format!("transformer_blocks.{i}.attn.to_q.weight")).is_some()).count();
        if n_blocks == 0 {
            return Err(Error(format!("{}: no transformer_blocks (not a Qwen-Image transformer)", f.paths[0].display())));
        }
        let mut blocks = Vec::with_capacity(n_blocks);
        let mut used = 0u64;
        let mut biggest = 0usize;
        let mut resident = 0;
        for i in 0..n_blocks {
            let p = format!("transformer_blocks.{i}.");
            // the block's matrices as stored, one after another (q's and k's rows permuted)
            let mut raw = Vec::new();
            let mut lins = Vec::new();
            for m in MATS {
                let name = format!("{p}{m}.weight");
                let t = tensor(&name)?;
                let code = t.ty.code();
                if ![0, 1, 8, 12, 13, 14, 30].contains(&code) {
                    return Err(Error(format!("{name}: {} weights are not read yet", t.ty.name())));
                }
                let (n, k) = (t.shape[0] as usize, t.shape[1] as usize);
                let perm = matches!(m, "attn.to_q" | "attn.to_k" | "attn.add_q_proj" | "attn.add_k_proj");
                let b = f.read(t).map_err(ge)?;
                let off = raw.len();
                if perm {
                    let rb = b.len() / n;
                    for r in 0..n {
                        let src = (r / HEAD) * HEAD + rope_perm(r % HEAD);
                        raw.extend_from_slice(&b[src * rb..(src + 1) * rb]);
                    }
                } else {
                    raw.extend_from_slice(&b);
                }
                // 16-byte alignment for the next matrix
                raw.resize(raw.len().div_ceil(256) * 256, 0);
                let bias = host_vec(&format!("{p}{m}.bias"))?;
                let bias_v = matches!(m, "attn.to_out.0" | "attn.to_add_out").then(|| DevBuf::from_f32(&gpu, &bias.iter().map(|x| x / VS).collect::<Vec<f32>>()))
                    .transpose()?;
                let bias = DevBuf::from_f32(&gpu, &if perm { permuted(bias) } else { bias })?;
                lins.push(Lin { key: lora::key(&format!("{p}{m}")), off, qtype: code, n, k, bias, perm, bias_v, bf16: m.ends_with("net.2"), loras: Mutex::new(Vec::new()) });
            }
            biggest = biggest.max(raw.len());
            let home = if used + raw.len() as u64 <= budget {
                used += raw.len() as u64;
                resident += 1;
                let d = DevBuf::new(&gpu, raw.len())?;
                d.write(0, &raw)?;
                Home::Card(d)
            } else {
                let mut h = HostBuf::new(&gpu, raw.len())?;
                h.as_mut_slice().copy_from_slice(&raw);
                Home::Host(h)
            };
            let mut it = lins.into_iter();
            let mut next = || it.next().expect("a matrix a name");
            let (img_mod, txt_mod) = (next(), next());
            let qkv = [next(), next(), next()];
            let add_qkv = [next(), next(), next()];
            let (out, add_out, img_fc1, img_fc2, txt_fc1, txt_fc2) = (next(), next(), next(), next(), next(), next());
            let vec = |name: &str| -> Result<DevBuf> { DevBuf::from_f32(&gpu, &permuted(host_vec(&format!("{p}attn.{name}.weight"))?)) };
            blocks.push(Block {
                home,
                bytes: raw.len(),
                img_mod,
                txt_mod,
                qkv,
                add_qkv,
                out,
                add_out,
                img_fc1,
                img_fc2,
                txt_fc1,
                txt_fc2,
                norm_q: vec("norm_q")?,
                norm_k: vec("norm_k")?,
                norm_aq: vec("norm_added_q")?,
                norm_ak: vec("norm_added_k")?,
            });
        }
        let staging = if resident < n_blocks { Some(DevBuf::new(&gpu, biggest)?) } else { None };
        let dit = Dit {
            gpu: gpu.clone(),
            blocks,
            staging,
            img_in: small("img_in")?,
            txt_norm: DevBuf::from_f32(&gpu, &host_vec("txt_norm.weight")?)?,
            txt_in: small("txt_in")?,
            t1: small("time_text_embed.timestep_embedder.linear_1")?,
            t2: small("time_text_embed.timestep_embedder.linear_2")?,
            norm_out: small("norm_out.linear")?,
            proj_out: small("proj_out")?,
            resident,
        };
        log(format!("qwen-image 2512: {n_blocks} blocks, {resident} on {} ({:.1} GiB), {} streamed from host memory; {:.1} s", gpu.name,
                    used as f64 / (1u64 << 30) as f64, n_blocks - resident, t0.elapsed().as_secs_f64()));
        Ok(dit)
    }

    /// The LoRA deltas as side paths of the blocks' matrices (replacing any before); an empty list removes them
    pub fn set_loras(&self, deltas: &[Delta]) -> Result<()> {
        let gpu = &self.gpu;
        let half = |v: &[f32], bf: bool| -> Result<DevBuf> {
            let b = DevBuf::new(gpu, v.len().max(1) * 2)?;
            let h: Vec<u8> = v.iter().flat_map(|x| if bf { bf16_bits(*x) } else { half_bits(*x) }.to_le_bytes()).collect();
            b.write(0, &h)?;
            Ok(b)
        };
        let mut by_key: std::collections::HashMap<String, Vec<&Delta>> = std::collections::HashMap::new();
        for d in deltas {
            by_key.entry(lora::key(&d.target)).or_default().push(d);
        }
        let mut used = std::collections::BTreeSet::new();
        for b in &self.blocks {
            for l in b.lins() {
                let mut v = l.loras.lock().unwrap();
                v.clear();
                for d in by_key.get(&l.key).into_iter().flatten() {
                    if d.n != l.n || d.k != l.k {
                        return Err(Error(format!("LoRA {}: {}x{}, the matrix is {}x{}", d.target, d.n, d.k, l.n, l.k)));
                    }
                    let bv: Vec<f32> = if l.perm {
                        (0..d.n).flat_map(|row| {
                            let src = (row / HEAD) * HEAD + rope_perm(row % HEAD);
                            d.b[src * d.r..(src + 1) * d.r].iter().copied()
                        }).collect()
                    } else {
                        d.b.clone()
                    };
                    v.push(LoraPair { a: half(&d.a, l.bf16)?, b: DevBuf::from_f32(gpu, &bv)?, r: d.r, dt: if l.bf16 { Dt::BF16 } else { Dt::F16 } });
                    used.insert(l.key.clone());
                }
            }
        }
        let missed: Vec<&str> = deltas.iter().filter(|d| !used.contains(&lora::key(&d.target))).map(|d| d.target.as_str()).collect();
        if !missed.is_empty() {
            return Err(Error(format!("LoRA modules this engine does not take ({} of them): {}", missed.len(), missed[..missed.len().min(4)].join(", "))));
        }
        Ok(())
    }

    /// The timestep's embedding, silu'd (what every modulation and the output norm read): half [3072] and float32
    fn temb(&self, nsd: &Nsd, t: f32) -> Result<(DevBuf, DevBuf)> {
        let gpu = &self.gpu;
        let mut emb = vec![0f32; 256];
        for i in 0..128 {
            let a = 1000.0 * t as f64 * (-(10000f64).ln() * i as f64 / 128.0).exp();
            emb[i] = a.cos() as f32;
            emb[128 + i] = a.sin() as f32;
        }
        let silu = |v: &mut [f32]| v.iter_mut().for_each(|x| *x /= 1.0 + (-*x).exp());
        let e = DevBuf::from_f32(gpu, &emb)?;
        let h = DevBuf::f32(gpu, DIM)?;
        nsd.linear(e.ptr(), Dt::F32, 1, 256, self.t1.0.ptr(), DIM, self.t1.1.ptr(), h.ptr(), Dt::F32)?;
        let mut hv = h.to_f32()?;
        silu(&mut hv);
        let h = DevBuf::from_f32(gpu, &hv)?;
        let te = DevBuf::f32(gpu, DIM)?;
        nsd.linear(h.ptr(), Dt::F32, 1, DIM, self.t2.0.ptr(), DIM, self.t2.1.ptr(), te.ptr(), Dt::F32)?;
        let mut tv = te.to_f32()?;
        silu(&mut tv);
        let hb = DevBuf::new(gpu, DIM * 2)?;
        hb.write(0, &tv.iter().flat_map(|x| half_bits(*x).to_le_bytes()).collect::<Vec<u8>>())?;
        Ok((hb, DevBuf::from_f32(gpu, &tv)?))
    }

    /// (cos, sin) per token and pair for positions `idx` (frame, height, width per token), in the permuted order
    fn rope_table(&self, idx: &[[i64; 3]]) -> Result<DevBuf> {
        let mut cs = vec![0f32; idx.len() * HEAD];
        for (t, p) in idx.iter().enumerate() {
            let mut pair = 0;
            for (axis, dim) in AXES.iter().enumerate() {
                for i in 0..dim / 2 {
                    let inv = 1.0 / 10000f64.powf(2.0 * i as f64 / *dim as f64);
                    let a = p[axis] as f64 * inv;
                    cs[t * HEAD + 2 * pair] = a.cos() as f32;
                    cs[t * HEAD + 2 * pair + 1] = a.sin() as f32;
                    pair += 1;
                }
            }
        }
        DevBuf::from_f32(&self.gpu, &cs)
    }

    /// The velocity of latents `lat` (float32 [hw.0 * hw.1, 64]) at sigma `t` for each text of `texts` (a guided
    /// step's prompt and negative): float32 [N, 64] each. Each block's weights come to the card once for all the passes.
    pub fn velocity(&self, nsd: &Nsd, texts: &[&Text], lat: &DevBuf, hw: (usize, usize), t: f32) -> Result<Vec<DevBuf>> {
        let gpu = &self.gpu;
        let k = ffi::api()?;
        let n = hw.0 * hw.1;
        let (temb_h, temb_f) = self.temb(nsd, t)?;
        // the picture's positions: frame 0, rows and columns centred; each text's after max(h, w) / 2 on every axis
        let (hh, ww) = (hw.0 as i64, hw.1 as i64);
        let start = hh.max(ww) / 2;
        let tmax = texts.iter().map(|x| x.tokens).max().unwrap_or(0);
        let mut idx: Vec<[i64; 3]> = (0..tmax as i64).map(|j| [start + j, start + j, start + j]).collect();
        idx.extend((0..n as i64).map(|i| [0, i / ww - (hh - hh / 2), i % ww - (ww - ww / 2)]));
        let cs_all = self.rope_table(&idx)?;
        // each pass's two streams, float32
        let mut xs = Vec::with_capacity(texts.len());
        for tx in texts {
            let xi = DevBuf::f32(gpu, n * DIM)?;
            nsd.linear(lat.ptr(), Dt::F32, n, LATENT, self.img_in.0.ptr(), DIM, self.img_in.1.ptr(), xi.ptr(), Dt::F32)?;
            let tn = DevBuf::f32(gpu, tx.tokens * CONTEXT)?;
            nsd.rms_norm_mod(tx.emb.ptr(), Dt::F32, tx.tokens, CONTEXT, self.txt_norm.ptr(), EPS, none(), none(), none(), tn.ptr(), Dt::F32)?;
            let xt = DevBuf::f32(gpu, tx.tokens * DIM)?;
            nsd.linear(tn.ptr(), Dt::F32, tx.tokens, CONTEXT, self.txt_in.0.ptr(), DIM, self.txt_in.1.ptr(), xt.ptr(), Dt::F32)?;
            nsd.wait()?;
            xs.push((xi, xt));
        }
        // the blocks' scratch, shared by the passes
        let rows = tmax + n;
        let h = DevBuf::new(gpu, rows * DIM * 2)?;
        // q|k|v float32 as the linears make them (v passes half's range in the late blocks), then half for the
        // attention with v scaled by 1/VS (q and k are normed, so the scale only reaches v: the attention's output
        // and the output linears' bias are scaled with it, and the product scaled back)
        let qkv = DevBuf::f32(gpu, rows * 3 * DIM)?;
        let qkvh = DevBuf::new(gpu, rows * 3 * DIM * 2)?;
        let att = DevBuf::new(gpu, rows * DIM * 2)?;
        // the linears' outputs before their gated adds, float32: Qwen-Image's activations pass half's range (the text
        // stream's reach millions; diffusers clips them in float16)
        let o = DevBuf::f32(gpu, rows * DIM)?;
        let fc = DevBuf::f32(gpu, FFN_ROWS.min(rows) * FFN)?;
        let fcb = DevBuf::new(gpu, FFN_ROWS.min(rows) * FFN * 2)?;
        let zeros = DevBuf::new(gpu, rows * 4)?;
        zeros.fill(0)?;
        let mods = DevBuf::f32(gpu, 2 * 6 * DIM)?;
        for b in &self.blocks {
            let w = match &b.home {
                Home::Card(d) => d,
                Home::Host(hb) => {
                    let st = self.staging.as_ref().expect("a staging buffer for streamed blocks");
                    st.write(0, &hb.as_slice()[..b.bytes])?;
                    st
                }
            };
            // the modulations: [shift1 scale1 gate1 shift2 scale2 gate2] of the picture, then of the text
            self.apply(nsd, w, &b.img_mod, temb_h.ptr(), 1, mods.ptr(), Dt::F32, 0)?;
            self.apply(nsd, w, &b.txt_mod, temb_h.ptr(), 1, mods.fp().wrapping_add(6 * DIM).cast(), Dt::F32, 0)?;
            let md = |stream: usize, i: usize| -> *const f32 { mods.fp().wrapping_add(stream * 6 * DIM + i * DIM) };
            for (pi, tx) in texts.iter().enumerate() {
                let (xi, xt) = &xs[pi];
                let tt = tx.tokens;
                let r = tt + n;
                // [text | picture] rows of h, then q|k|v of each
                // SAFETY: x holds rows x 3072 floats, h as many halves; the shift / scale rows 3072 floats each.
                ffi::check(unsafe { (k.ln_mod_shift)(gpu.raw(), xt.fp(), tt as i64, DIM as i64, EPS, md(1, 1), md(1, 0), h.ptr()) }, "norm + modulate")?;
                ffi::check(unsafe { (k.ln_mod_shift)(gpu.raw(), xi.fp(), n as i64, DIM as i64, EPS, md(0, 1), md(0, 0), half_at(&h, tt * DIM)) },
                           "norm + modulate")?;
                for (j, l) in b.add_qkv.iter().enumerate() {
                    self.apply(nsd, w, l, h.ptr(), tt, qkv.fp().wrapping_add(j * DIM).cast(), Dt::F32, 3 * DIM)?;
                }
                for (j, l) in b.qkv.iter().enumerate() {
                    self.apply(nsd, w, l, half_at(&h, tt * DIM), n, qkv.fp().wrapping_add(tt * 3 * DIM + j * DIM).cast(), Dt::F32, 3 * DIM)?;
                }
                // the table's first rows are the text positions (a shorter text reads the first of them), then the picture's
                let cs_txt = cs_all.fp().cast_const();
                let cs_img = cs_all.fp().wrapping_add(tmax * HEAD).cast_const();
                let f_at = |at: usize| -> *mut std::ffi::c_void { qkv.fp().wrapping_add(at).cast() };
                nsd.scale(qkv.ptr(), r * 3 * DIM, 1.0 / VS)?;
                nsd.rms_rope(f_at(0), Dt::F32, tt, HEADS, HEAD, 3 * DIM, b.norm_aq.ptr(), EPS, cs_txt.cast(), HEAD)?;
                nsd.rms_rope(f_at(DIM), Dt::F32, tt, HEADS, HEAD, 3 * DIM, b.norm_ak.ptr(), EPS, cs_txt.cast(), HEAD)?;
                nsd.rms_rope(f_at(tt * 3 * DIM), Dt::F32, n, HEADS, HEAD, 3 * DIM, b.norm_q.ptr(), EPS, cs_img.cast(), HEAD)?;
                nsd.rms_rope(f_at(tt * 3 * DIM + DIM), Dt::F32, n, HEADS, HEAD, 3 * DIM, b.norm_k.ptr(), EPS, cs_img.cast(), HEAD)?;
                // SAFETY: r x 9216 floats in, as many halves out.
                ffi::check(unsafe { (k.to_half)(gpu.raw(), qkv.fp(), qkvh.ptr(), (r * 3 * DIM) as i64) }, "to half")?;
                self.attend(nsd, qkvh.ptr(), half_at(&qkvh, DIM), half_at(&qkvh, 2 * DIM), r, att.ptr())?;
                let o_img: *mut std::ffi::c_void = o.fp().wrapping_add(tt * DIM).cast();
                self.apply_scaled(nsd, w, &b.add_out, att.ptr(), tt, o.ptr())?;
                self.apply_scaled(nsd, w, &b.out, half_at(&att, tt * DIM), n, o_img)?;
                nsd.scale(o.ptr(), r * DIM, VS)?;
                nsd.gate_add(xt.ptr(), Dt::F32, tt, DIM, o.ptr(), Dt::F32, zeros.ptr().cast(), md(1, 2).cast())?;
                nsd.gate_add(xi.ptr(), Dt::F32, n, DIM, o_img, Dt::F32, zeros.ptr().cast(), md(0, 2).cast())?;
                // the feed-forwards
                ffi::check(unsafe { (k.ln_mod_shift)(gpu.raw(), xt.fp(), tt as i64, DIM as i64, EPS, md(1, 4), md(1, 3), h.ptr()) }, "norm + modulate")?;
                ffi::check(unsafe { (k.ln_mod_shift)(gpu.raw(), xi.fp(), n as i64, DIM as i64, EPS, md(0, 4), md(0, 3), half_at(&h, tt * DIM)) },
                           "norm + modulate")?;
                self.ffn(nsd, w, &b.txt_fc1, &b.txt_fc2, &h, 0, tt, &fc, &fcb, &o)?;
                self.ffn(nsd, w, &b.img_fc1, &b.img_fc2, &h, tt, n, &fc, &fcb, &o)?;
                nsd.gate_add(xt.ptr(), Dt::F32, tt, DIM, o.ptr(), Dt::F32, zeros.ptr().cast(), md(1, 5).cast())?;
                nsd.gate_add(xi.ptr(), Dt::F32, n, DIM, o_img, Dt::F32, zeros.ptr().cast(), md(0, 5).cast())?;
            }
            // a streamed block's staging is written again by the next: its work first
            nsd.wait()?;
        }
        // the output norm (scale first, then shift) and the projection to 64 latent values a token
        let no = DevBuf::f32(gpu, 2 * DIM)?;
        nsd.linear(temb_f.ptr(), Dt::F32, 1, DIM, self.norm_out.0.ptr(), 2 * DIM, self.norm_out.1.ptr(), no.ptr(), Dt::F32)?;
        let mut out = Vec::with_capacity(texts.len());
        for (xi, _) in &xs {
            // SAFETY: n x 3072 floats in, as many halves out; scale and shift 3072 floats each.
            ffi::check(unsafe { (k.ln_mod_shift)(gpu.raw(), xi.fp(), n as i64, DIM as i64, EPS, no.fp(), no.fp().wrapping_add(DIM), h.ptr()) }, "norm + modulate")?;
            let xf = DevBuf::f32(gpu, n * DIM)?;
            ffi::check(unsafe { (k.to_float)(gpu.raw(), h.ptr(), xf.fp(), (n * DIM) as i64) }, "to float")?;
            let v = DevBuf::f32(gpu, n * LATENT)?;
            nsd.linear(xf.ptr(), Dt::F32, n, DIM, self.proj_out.0.ptr(), LATENT, self.proj_out.1.ptr(), v.ptr(), Dt::F32)?;
            nsd.wait()?;
            out.push(v);
        }
        Ok(out)
    }

    /// out [m, n] = x [m, k] (half, or bfloat16 for `l.bf16`) . W^T + b, W from the block's bytes `w`; the LoRAs'
    /// products added
    #[allow(clippy::too_many_arguments)]
    fn apply(&self, nsd: &Nsd, w: &DevBuf, l: &Lin, x: *const std::ffi::c_void, m: usize, out: *mut std::ffi::c_void, out_dt: Dt, ldo: usize) -> Result<()> {
        if m == 0 {
            return Ok(());
        }
        let x_dt = if l.bf16 { Dt::BF16 } else { Dt::F16 };
        let wp: *const std::ffi::c_void = w.ptr().cast::<u8>().wrapping_add(l.off).cast();
        nsd.linear_q(x, x_dt, m, l.k, wp, l.qtype, l.n, l.bias.ptr(), out, out_dt, ldo)?;
        let ls = l.loras.lock().unwrap();
        if !ls.is_empty() && out_dt != Dt::F32 {
            return Err(Error("the LoRA side paths add into float32 outputs".into()));
        }
        for lp in ls.iter() {
            // x A^T in the input's type, made float32, then B accumulated into the output's rows (`ldo` apart)
            let tmp = DevBuf::new(&self.gpu, m * lp.r * 2)?;
            nsd.linear(x, lp.dt, m, l.k, lp.a.ptr(), lp.r, none(), tmp.ptr(), lp.dt)?;
            let tf = DevBuf::f32(&self.gpu, m * lp.r)?;
            nsd.dequant(tmp.ptr(), if lp.dt == Dt::BF16 { 30 } else { 1 }, m * lp.r, tf.ptr(), Dt::F32)?;
            nsd.linear_acc(tf.ptr(), Dt::F32, m, lp.r, lp.b.ptr(), l.n, out, ldo)?;
            nsd.wait()?;
        }
        Ok(())
    }

    /// An output linear on the attention's output scaled by 1/VS: its bias scaled the same (the caller scales back)
    #[allow(clippy::too_many_arguments)]
    fn apply_scaled(&self, nsd: &Nsd, w: &DevBuf, l: &Lin, x: *const std::ffi::c_void, m: usize, out: *mut std::ffi::c_void) -> Result<()> {
        let small = l.bias_v.as_ref().ok_or_else(|| Error("an output linear without its scaled bias".into()))?;
        if m == 0 {
            return Ok(());
        }
        let wp: *const std::ffi::c_void = w.ptr().cast::<u8>().wrapping_add(l.off).cast();
        nsd.linear_q(x, Dt::F16, m, l.k, wp, l.qtype, l.n, small.ptr(), out, Dt::F32, 0)?;
        let ls = l.loras.lock().unwrap();
        for lp in ls.iter() {
            let tmp = DevBuf::new(&self.gpu, m * lp.r * 2)?;
            nsd.linear(x, lp.dt, m, l.k, lp.a.ptr(), lp.r, none(), tmp.ptr(), lp.dt)?;
            let tf = DevBuf::f32(&self.gpu, m * lp.r)?;
            nsd.dequant(tmp.ptr(), 1, m * lp.r, tf.ptr(), Dt::F32)?;
            nsd.linear_acc(tf.ptr(), Dt::F32, m, lp.r, lp.b.ptr(), l.n, out, 0)?;
            nsd.wait()?;
        }
        Ok(())
    }

    /// rows [r0, r0 + m) of h through fc1, gelu (tanh), fc2 into the same rows of o (float32), a slice of rows at a
    /// time; fc1's output float32, made bfloat16 for fc2 (it passes half's range)
    #[allow(clippy::too_many_arguments)]
    fn ffn(&self, nsd: &Nsd, w: &DevBuf, fc1: &Lin, fc2: &Lin, h: &DevBuf, r0: usize, m: usize, fc: &DevBuf, fcb: &DevBuf, o: &DevBuf) -> Result<()> {
        let cap = (fc.len / (FFN * 4)).max(1);
        let mut a = 0;
        while a < m {
            let r = cap.min(m - a);
            self.apply(nsd, w, fc1, half_at(h, (r0 + a) * DIM), r, fc.ptr(), Dt::F32, 0)?;
            nsd.gelu(fc.ptr(), Dt::F32, r * FFN, true)?;
            nsd.dequant(fc.ptr(), 0, r * FFN, fcb.ptr(), Dt::BF16)?;
            self.apply(nsd, w, fc2, fcb.ptr(), r, o.fp().wrapping_add((r0 + a) * DIM).cast(), Dt::F32, 0)?;
            a += r;
        }
        Ok(())
    }

    /// Full attention over `rows` rows of q|k|v (stride 3 x 3072), out [rows, 3072]: the silo's fused kernel when it
    /// takes the shape, else the shared one
    fn attend(&self, nsd: &Nsd, q: *const std::ffi::c_void, k: *const std::ffi::c_void, v: *const std::ffi::c_void, rows: usize, out: *mut std::ffi::c_void)
              -> Result<()> {
        let s = 3 * DIM;
        if let Some(si) = ffi::silo() {
            // SAFETY: a pure check of the shape.
            if unsafe { (si.attention_supported)(HEAD as i64, s as i64, s as i64) } == 1 {
                // SAFETY: q, k, v, out are the sizes named, on this GPU.
                let rc = unsafe { (si.attention)(self.gpu.raw(), q, k, v, rows as i64, rows as i64, HEADS as i64, HEADS as i64, s as i64, s as i64, 0, 0, out) };
                return ffi::check(rc, "attention (silo)");
            }
        }
        nsd.attention_qk(q, rows, s, k, v, rows, s, Dt::F16, HEADS, HEAD, out, Dt::F16)
    }
}

impl Dit {
    pub fn blocks(&self) -> usize {
        self.blocks.len()
    }
}

impl Block {
    fn lins(&self) -> Vec<&Lin> {
        let mut v = vec![&self.img_mod, &self.txt_mod, &self.out, &self.add_out, &self.img_fc1, &self.img_fc2, &self.txt_fc1, &self.txt_fc2];
        v.extend(self.qkv.iter());
        v.extend(self.add_qkv.iter());
        v
    }
}

/// A float32 as bfloat16 bits (round to nearest even)
pub fn bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() {
        return 0x7fc0;
    }
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

/// A float32 as IEEE half bits (round to nearest even)
pub fn half_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let mant = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = mant | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rest = m & ((1u32 << shift) - 1);
        let mut v = m >> shift;
        if rest > half || (rest == half && v & 1 == 1) {
            v += 1;
        }
        return sign | v as u16;
    }
    let mut v = ((e as u32) << 10) | (mant >> 13);
    let rest = mant & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && v & 1 == 1) {
        v += 1;
    }
    sign | v as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_bits_rounds_as_ieee() {
        assert_eq!(half_bits(1.0), 0x3c00);
        assert_eq!(half_bits(-2.0), 0xc000);
        assert_eq!(half_bits(65504.0), 0x7bff);
        assert_eq!(half_bits(1e6), 0x7c00);
        assert_eq!(half_bits(5.960_464_5e-8), 0x0001);
        assert_eq!(half_bits(0.0), 0);
    }

    #[test]
    fn the_rope_permutation_is_a_permutation_of_each_head() {
        let mut seen: Vec<usize> = (0..HEAD).map(rope_perm).collect();
        seen.sort();
        assert_eq!(seen, (0..HEAD).collect::<Vec<_>>());
    }
}
