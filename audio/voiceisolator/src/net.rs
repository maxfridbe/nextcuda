//! The isolator's network on one GPU, float32 throughout (the checkpoint's own weights; the products on cuBLAS without
//! TF32): the band encoders, seven blocks - time attention (rotary positions, exclusive self attention) then attention
//! across the bands, or bidirectional Mamba along time then across the bands - and the per-band heads.
//!
//! The tokens are [bands, frames, width] (a band's frames one after another) for the time paths and [frames, bands,
//! width] for the band paths, transposed between them. Origin: model.py of Inworld's voice isolator (Apache-2.0).
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::c_void;
use std::sync::Arc;

use nextcuda_audio::{Error, Result};
use nextcuda_core::{DevBuf, Gpu};
use nextcuda_diffusion::kernels::{none, Dt, Nsd};
use nextcuda_gguf::safetensors::SafeTensors;
use serde_json::Value;

use crate::dsp::band_edges;
use crate::ffi::{self, check};

type P = *const c_void;

const EPS: f32 = 1e-5;

/// The architecture, from the checkpoint's metadata ("config": its JSON)
#[derive(Clone, Debug)]
pub struct Config {
    pub width: usize,
    pub heads: usize,
    pub blocks: Vec<String>,
    pub n_fft: usize,
    pub hop: usize,
    pub rate: u32,
    pub plan: Vec<(f64, f64)>,
    pub bound: f32,
    pub chunk_seconds: f64,
    pub overlap_seconds: f64,
    pub exclusive_time: bool,
    pub exclusive_band: bool,
    pub state: usize,
    pub conv: usize,
    pub expand: usize,
}

impl Config {
    pub fn from_json(v: &Value) -> Result<Config> {
        let n = |k: &str| v[k].as_u64().map(|x| x as usize).ok_or_else(|| Error(format!("the checkpoint's config has no {k}")));
        let f = |k: &str| v[k].as_f64().ok_or_else(|| Error(format!("the checkpoint's config has no {k}")));
        // what this port runs: the two-sided (offline) checkpoint, one pass, bands, a complex mask with a residual
        for (k, ok) in [("causal", v["causal"] != true), ("dual_mode", v["dual_mode"] != true), ("mode_film", v["mode_film"] != true),
                        ("offline_passes", v["offline_passes"].as_u64().unwrap_or(1) == 1),
                        ("lookahead_frames_choices", v["lookahead_frames_choices"].as_array().is_none_or(|a| a.is_empty())),
                        ("representation", v["representation"].as_str().is_none_or(|x| x == "bands")),
                        ("decoder_type", v["decoder_type"].as_str().is_none_or(|x| x == "complex_mask_residual"))] {
            if !ok {
                return Err(Error(format!("this port runs the offline checkpoint (voice-isolator-v1); {k} = {} is not ported", v[k])));
            }
        }
        let axes = v["exclusive_axes"].as_str().unwrap_or("time");
        let excl = v["exclusive_self_attention"] != false;
        let plan = v["band_plan"].as_array().ok_or_else(|| Error("the checkpoint's config has no band_plan".into()))?
            .iter().map(|p| Some((p[0].as_f64()?, p[1].as_f64()?))).collect::<Option<Vec<_>>>()
            .ok_or_else(|| Error("band_plan: pairs of numbers".into()))?;
        let c = Config {
            width: n("width")?,
            heads: n("num_heads")?,
            blocks: v["block_types"].as_array().ok_or_else(|| Error("no block_types".into()))?.iter().map(|b| b.as_str().unwrap_or("").to_string()).collect(),
            n_fft: n("n_fft")?,
            hop: n("hop_length")?,
            rate: n("sample_rate")? as u32,
            plan,
            bound: f("correction_bound")? as f32,
            chunk_seconds: f("chunk_seconds")?,
            overlap_seconds: f("overlap_seconds")?,
            exclusive_time: excl && (axes == "both" || axes == "time"),
            exclusive_band: excl && (axes == "both" || axes == "band"),
            state: n("mamba_state")?,
            conv: n("mamba_conv")?,
            expand: n("mamba_expand")?,
        };
        if c.blocks.iter().any(|b| b != "transformer" && b != "mamba") {
            return Err(Error(format!("block types {:?}: transformer or mamba", c.blocks)));
        }
        if !c.width.is_multiple_of(c.heads) || !(c.width / c.heads).is_multiple_of(2) || c.state > 16 || !c.n_fft.is_power_of_two() {
            return Err(Error(format!("width {} over {} heads, {} states, n_fft {}: not what this port runs", c.width, c.heads, c.state, c.n_fft)));
        }
        Ok(c)
    }
}

struct Lin {
    w: DevBuf,
    b: Option<DevBuf>,
    n: usize,
    k: usize,
}

struct Ln {
    w: DevBuf,
    b: DevBuf,
    c: usize,
}

struct Mamba {
    in_proj: Lin,
    conv_w: DevBuf,
    conv_b: DevBuf,
    /// x_proj's rows apart: the step-size rank's (dt) and the states' B and C
    x_dt: Lin,
    x_bc: Lin,
    dt_proj: Lin,
    a_log: DevBuf,
    d: DevBuf,
    out_proj: Lin,
}

struct BiMamba {
    fwd: Mamba,
    bwd: Mamba,
    out: Lin,
    norm: Ln,
}

// seven of them, held for the model's life: their sizes do not matter
#[allow(clippy::large_enum_variant)]
enum Block {
    Attn {
        // along time
        t_norm1: Ln,
        t_qkv: Lin,
        t_proj: Lin,
        t_norm2: Ln,
        t_ff1: Lin,
        t_ff2: Lin,
        // across bands
        b_norm1: Ln,
        b_qkv: Lin,
        b_proj: Lin,
        b_norm2: Ln,
        b_ff1: Lin,
        b_ff2: Lin,
    },
    Mamba { time: BiMamba, freq: BiMamba },
}

struct Head {
    norm: Ln,
    l1: Lin,
    l2: Lin,
}

struct Loader<'a> {
    st: &'a SafeTensors,
    gpu: &'a Arc<Gpu>,
}

impl Loader<'_> {
    fn f(&self, name: &str) -> Result<DevBuf> {
        DevBuf::from_f32(self.gpu, &self.st.f32(name).map_err(|e| Error(e.0))?)
    }
    fn host(&self, name: &str) -> Result<Vec<f32>> {
        self.st.f32(name).map_err(|e| Error(e.0))
    }
    fn shape(&self, name: &str) -> Result<Vec<u64>> {
        Ok(self.st.need(name).map_err(|e| Error(e.0))?.shape.clone())
    }
    fn lin(&self, p: &str, bias: bool) -> Result<Lin> {
        let s = self.shape(&format!("{p}.weight"))?;
        Ok(Lin { w: self.f(&format!("{p}.weight"))?, b: if bias { Some(self.f(&format!("{p}.bias"))?) } else { None }, n: s[0] as usize, k: s[1] as usize })
    }
    /// a fused in-projection named apart (torch's MultiheadAttention: in_proj_weight / in_proj_bias)
    fn lin_named(&self, w: &str, b: &str) -> Result<Lin> {
        let s = self.shape(w)?;
        Ok(Lin { w: self.f(w)?, b: Some(self.f(b)?), n: s[0] as usize, k: s[1] as usize })
    }
    fn ln(&self, p: &str) -> Result<Ln> {
        let s = self.shape(&format!("{p}.weight"))?;
        Ok(Ln { w: self.f(&format!("{p}.weight"))?, b: self.f(&format!("{p}.bias"))?, c: s[0] as usize })
    }
    fn mamba(&self, p: &str) -> Result<Mamba> {
        let rank = self.shape(&format!("{p}.dt_proj.weight"))?[1] as usize;
        let xs = self.shape(&format!("{p}.x_proj.weight"))?;
        let (n, k) = (xs[0] as usize, xs[1] as usize);
        let xw = self.host(&format!("{p}.x_proj.weight"))?;
        Ok(Mamba {
            in_proj: self.lin(&format!("{p}.in_proj"), false)?,
            conv_w: self.f(&format!("{p}.conv1d.weight"))?,
            conv_b: self.f(&format!("{p}.conv1d.bias"))?,
            x_dt: Lin { w: DevBuf::from_f32(self.gpu, &xw[..rank * k])?, b: None, n: rank, k },
            x_bc: Lin { w: DevBuf::from_f32(self.gpu, &xw[rank * k..])?, b: None, n: n - rank, k },
            dt_proj: self.lin(&format!("{p}.dt_proj"), true)?,
            a_log: self.f(&format!("{p}.A_log"))?,
            d: self.f(&format!("{p}.D"))?,
            out_proj: self.lin(&format!("{p}.out_proj"), false)?,
        })
    }
    fn bimamba(&self, p: &str) -> Result<BiMamba> {
        Ok(BiMamba {
            fwd: self.mamba(&format!("{p}.forward_blocks"))?,
            bwd: self.mamba(&format!("{p}.backward_blocks"))?,
            out: self.lin(&format!("{p}.output_proj"), true)?,
            norm: self.ln(&format!("{p}.norm"))?,
        })
    }
}

/// The work buffers for chunks of up to `frames` frames
struct Work {
    frames: usize,
    x: DevBuf,
    xb: DevBuf,
    h: DevBuf,
    att: DevBuf,
    tmp: DevBuf,
    qkv: DevBuf,
    ff: DevBuf,
    cat: DevBuf,
    u: DevBuf,
    dl: DevBuf,
    y: DevBuf,
    xdbl: DevBuf,
    band_in: DevBuf,
    raw: DevBuf,
    rope: DevBuf,
}

pub struct Net {
    pub cfg: Config,
    pub edges: Vec<(usize, usize)>,
    pub gpu: Arc<Gpu>,
    nsd: Nsd,
    k: &'static ffi::Api,
    enc_norm: Vec<Ln>,
    enc_lin: Vec<Lin>,
    band_emb: DevBuf,
    blocks: Vec<Block>,
    film_scale: DevBuf,
    film_shift: DevBuf,
    heads: Vec<Head>,
    work: Work,
    /// what the weights take on the card
    pub weight_bytes: u64,
}

/// A stage's tokens, read back for the check
pub struct Taps<'a> {
    pub want: &'a [&'a str],
    pub got: Vec<(String, Vec<f32>)>,
}

impl Net {
    /// The checkpoint's network on `gpu`, its work buffers sized for the checkpoint's chunk
    pub fn load(st: &SafeTensors, cfg: Config, gpu: &Arc<Gpu>, log: &mut dyn FnMut(String)) -> Result<Net> {
        let edges = band_edges(cfg.n_fft, cfg.rate, &cfg.plan);
        let frames = (cfg.chunk_seconds * cfg.rate as f64) as usize / cfg.hop + 1;
        let (w, nb) = (cfg.width, edges.len());
        let rows = nb * frames;
        let inner = cfg.expand * w;
        let ffw = 4 * w;
        let bins = cfg.n_fft / 2 + 1;
        // the plan, against what the card has free
        let weight_bytes: u64 = st.tensors.iter().map(|t| t.elements() * 4).sum();
        let work_floats = rows * (5 * w + 3 * w + ffw.max(2 * inner) + 2 * w + 3 * inner + 64) + frames * bins * 6 + frames * w;
        // the time attention's score table for one band (nsd's, float32 and its probabilities)
        let scores = 2 * cfg.heads * frames * frames * 4;
        let need = weight_bytes + work_floats as u64 * 4 + scores as u64 + (256u64 << 20);
        let free = gpu.free_or_guess().map_err(|e| Error(e.0))?;
        if need > free {
            return Err(Error(format!("the voice isolator needs {:.2} GiB on {} and {:.2} GiB is free", need as f64 / (1u64 << 30) as f64, gpu.name,
                                     free as f64 / (1u64 << 30) as f64)));
        }
        log(format!("voiceisolator: {nb} bands of width {w}, {} blocks, {:.0} MiB of weights; chunks of {frames} frames ({:.2} GiB planned) on {}",
                    cfg.blocks.len(), weight_bytes as f64 / (1u64 << 20) as f64, need as f64 / (1u64 << 30) as f64, gpu.name));
        let l = Loader { st, gpu };
        let mut enc_norm = Vec::new();
        let mut enc_lin = Vec::new();
        for i in 0..nb {
            enc_norm.push(l.ln(&format!("encoders.{i}.0"))?);
            enc_lin.push(l.lin(&format!("encoders.{i}.1"), true)?);
        }
        let band_emb = l.f("band_embedding")?;
        let mut blocks = Vec::new();
        for (i, kind) in cfg.blocks.iter().enumerate() {
            let p = format!("blocks.{i}");
            blocks.push(if kind == "transformer" {
                let t = format!("{p}.time_transformer");
                let b = format!("{p}.band_transformer");
                Block::Attn {
                    t_norm1: l.ln(&format!("{t}.norm1"))?,
                    t_qkv: l.lin(&format!("{t}.attn.qkv"), true)?,
                    t_proj: l.lin(&format!("{t}.attn.proj"), true)?,
                    t_norm2: l.ln(&format!("{t}.norm2"))?,
                    t_ff1: l.lin(&format!("{t}.ffn.0"), true)?,
                    t_ff2: l.lin(&format!("{t}.ffn.3"), true)?,
                    b_norm1: l.ln(&format!("{b}.norm1"))?,
                    b_qkv: l.lin_named(&format!("{b}.self_attn.in_proj_weight"), &format!("{b}.self_attn.in_proj_bias"))?,
                    b_proj: l.lin(&format!("{b}.self_attn.out_proj"), true)?,
                    b_norm2: l.ln(&format!("{b}.norm2"))?,
                    b_ff1: l.lin(&format!("{b}.linear1"), true)?,
                    b_ff2: l.lin(&format!("{b}.linear2"), true)?,
                }
            } else {
                Block::Mamba { time: l.bimamba(&format!("{p}.time_mamba"))?, freq: l.bimamba(&format!("{p}.freq_mamba"))? }
            });
        }
        // the decoder's conditioning at its served controls (1, 0): scale | shift = W[:, 0] + b
        let cw = l.host("denoise_decoder.condition.weight")?;
        let cb = l.host("denoise_decoder.condition.bias")?;
        let cond: Vec<f32> = (0..2 * w).map(|j| cw[j * 2] * 1.0 + cw[j * 2 + 1] * 0.0 + cb[j]).collect();
        let film_scale = DevBuf::from_f32(gpu, &cond[..w])?;
        let film_shift = DevBuf::from_f32(gpu, &cond[w..])?;
        let mut heads = Vec::new();
        for i in 0..nb {
            let p = format!("denoise_decoder.heads.{i}");
            heads.push(Head { norm: l.ln(&format!("{p}.0"))?, l1: l.lin(&format!("{p}.1"), true)?, l2: l.lin(&format!("{p}.3"), true)? });
        }
        for (i, h) in heads.iter().enumerate() {
            if h.l2.n != 4 * (edges[i].1 - edges[i].0) || enc_lin[i].k != 2 * (edges[i].1 - edges[i].0) {
                return Err(Error(format!("band {i}: the checkpoint's sizes do not match the band plan's {:?}", edges[i])));
            }
        }
        // the rotary tables: cos | sin [frames, head / 2], as the reference makes them (float32)
        let hd = w / cfg.heads;
        let half = hd / 2;
        let inv: Vec<f32> = (0..half).map(|i| 10000f32.powf(-(i as f32) / half as f32)).collect();
        let mut cs = vec![0f32; 2 * frames * half];
        for t in 0..frames {
            for (i, f) in inv.iter().enumerate() {
                let a = t as f32 * f;
                cs[t * half + i] = a.cos();
                cs[frames * half + t * half + i] = a.sin();
            }
        }
        let fb = |n: usize| DevBuf::f32(gpu, n);
        let work = Work {
            frames,
            x: fb(rows * w)?,
            xb: fb(rows * w)?,
            h: fb(rows * w)?,
            att: fb(rows * w)?,
            tmp: fb(rows * w)?,
            qkv: fb(rows * 3 * w)?,
            ff: fb(rows * ffw.max(2 * inner))?,
            cat: fb(rows * 2 * w)?,
            u: fb(rows * inner)?,
            dl: fb(rows * inner)?,
            y: fb(rows * inner)?,
            xdbl: fb(rows * 64.max(2 * cfg.state + w.div_ceil(16)))?,
            band_in: fb(frames * bins * 2)?,
            raw: fb(frames * bins * 4)?,
            rope: DevBuf::from_f32(gpu, &cs)?,
        };
        Ok(Net { nsd: Nsd::new(gpu)?, k: ffi::api()?, gpu: gpu.clone(), edges, enc_norm, enc_lin, band_emb, blocks, film_scale, film_shift, heads, work, weight_bytes, cfg })
    }

    /// The longest chunk the work buffers hold, in frames
    pub fn max_frames(&self) -> usize {
        self.work.frames
    }

    fn g(&self) -> *mut c_void {
        self.gpu.raw()
    }

    fn lin(&self, x: P, m: usize, l: &Lin, out: *mut c_void) -> Result<()> {
        self.nsd.linear(x, Dt::F32, m, l.k, l.w.ptr(), l.n, l.b.as_ref().map_or(none(), |b| b.ptr() as P), out, Dt::F32)
    }

    fn ln(&self, x: P, m: usize, n: &Ln, out: *mut c_void) -> Result<()> {
        self.nsd.layer_norm(x, Dt::F32, m, n.c, n.w.ptr(), n.b.ptr(), EPS, out, Dt::F32)
    }

    /// x += y over [m, w]
    fn add(&self, x: *mut c_void, y: P, m: usize) -> Result<()> {
        self.nsd.gate_add(x, Dt::F32, m, self.cfg.width, y, Dt::F32, none(), none())
    }

    fn swap(&self, x: &DevBuf, a: usize, b: usize, out: &DevBuf) -> Result<()> {
        // SAFETY: x and out hold a x b x width floats.
        check(unsafe { (self.k.swap01)(self.g(), x.fp(), a as i64, b as i64, self.cfg.width as i64, out.fp()) }, "swap01")
    }

    /// A pre-norm transformer layer over x [seqs, len, width] (in place): attention (rotary and exclusive along time;
    /// within a frame's bands without either), then the GELU MLP
    #[allow(clippy::too_many_arguments)]
    fn attn_layer(&self, x: &DevBuf, seqs: usize, len: usize, along_time: bool, n1: &Ln, qkv: &Lin, proj: &Lin, n2: &Ln, f1: &Lin, f2: &Lin)
                  -> Result<()> {
        let w = &self.work;
        let (c, h) = (self.cfg.width, self.cfg.heads);
        let hd = c / h;
        let m = seqs * len;
        self.ln(x.ptr(), m, n1, w.h.ptr())?;
        self.lin(w.h.ptr(), m, qkv, w.qkv.ptr())?;
        if along_time {
            // SAFETY: qkv holds m rows of 3c, the table len rows.
            let sin = w.rope.fp().wrapping_add(w.frames * hd / 2);
            check(unsafe { (self.k.rope)(self.g(), w.qkv.fp(), m as i64, len as i64, h as i64, hd as i64, w.rope.fp(), sin) }, "rope")?;
            let q = w.qkv.ptr() as P;
            let kk = w.qkv.fp().wrapping_add(c) as P;
            let v = w.qkv.fp().wrapping_add(2 * c) as P;
            self.nsd.attention_batch(q, kk, v, Dt::F32, seqs, len, h, hd, 3 * c, w.att.ptr(), Dt::F32)?;
        } else {
            // SAFETY: qkv holds m rows of 3c, att m rows of c.
            check(unsafe { (self.k.short_attention)(self.g(), w.qkv.fp(), seqs as i64, len as i64, h as i64, hd as i64, w.att.fp()) }, "band attention")?;
        }
        if (along_time && self.cfg.exclusive_time) || (!along_time && self.cfg.exclusive_band) {
            // SAFETY: att m rows of c; the values at column 2c of qkv's rows.
            check(unsafe { (self.k.exclusive)(self.g(), w.att.fp(), w.qkv.fp().wrapping_add(2 * c), 3 * c as i64, m as i64, h as i64, hd as i64) },
                  "exclusive attention")?;
        }
        self.lin(w.att.ptr(), m, proj, w.tmp.ptr())?;
        self.add(x.ptr(), w.tmp.ptr(), m)?;
        self.ln(x.ptr(), m, n2, w.h.ptr())?;
        self.lin(w.h.ptr(), m, f1, w.ff.ptr())?;
        self.nsd.gelu(w.ff.ptr(), Dt::F32, m * f1.n, false)?;
        self.lin(w.ff.ptr(), m, f2, w.tmp.ptr())?;
        self.add(x.ptr(), w.tmp.ptr(), m)
    }

    /// One Mamba direction over x [seqs, len, width], its output added into cat's half (`half` 0: forward, 1: backward;
    /// cat already holds x in both)
    fn mamba(&self, x: &DevBuf, seqs: usize, len: usize, mb: &Mamba, reverse: bool, half: usize) -> Result<()> {
        let w = &self.work;
        let c = self.cfg.width;
        let m = seqs * len;
        let inner = mb.in_proj.n / 2;
        let n = self.cfg.state;
        self.lin(x.ptr(), m, &mb.in_proj, w.ff.ptr())?;
        // SAFETY: ff holds m rows of 2 inner; u, dl, y m rows of inner; xdbl m rows of the dt rank and m of 2 n.
        unsafe {
            check((self.k.conv_silu)(self.g(), w.ff.fp(), mb.in_proj.n as i64, seqs as i64, len as i64, inner as i64, mb.conv_w.fp(), self.cfg.conv as i64,
                                     mb.conv_b.fp(), reverse as i32, w.u.fp()), "mamba conv")?;
        }
        // xdbl: the step sizes' low-rank rows [m, rank], then B | C [m, 2 n]
        let bc = w.xdbl.fp().wrapping_add(m * mb.x_dt.n);
        self.lin(w.u.ptr(), m, &mb.x_dt, w.xdbl.ptr())?;
        self.lin(w.u.ptr(), m, &mb.x_bc, bc.cast())?;
        // delta = softplus(dt . dt_proj^T + bias)
        self.lin(w.xdbl.ptr(), m, &mb.dt_proj, w.dl.ptr())?;
        // SAFETY: as above.
        unsafe {
            check((self.k.softplus)(self.g(), w.dl.fp(), (m * inner) as i64), "mamba delta")?;
            check((self.k.scan)(self.g(), w.u.fp(), w.dl.fp(), mb.a_log.fp(), mb.d.fp(), bc, mb.x_bc.n as i64, 0, w.ff.fp(), mb.in_proj.n as i64,
                                inner as i64, seqs as i64, len as i64, inner as i64, n as i64, reverse as i32, w.y.fp()), "mamba scan")?;
        }
        // cat[:, half] += y . out_proj^T
        self.nsd.linear_acc(w.y.ptr(), Dt::F32, m, inner, mb.out_proj.w.ptr(), c, w.cat.fp().wrapping_add(half * c).cast(), 2 * c)
    }

    /// x += norm(output_proj([fwd(x) + x, bwd(x) + x])) over x [seqs, len, width]
    fn bimamba(&self, x: &DevBuf, seqs: usize, len: usize, b: &BiMamba) -> Result<()> {
        let w = &self.work;
        let m = seqs * len;
        // SAFETY: x m rows of width, cat m rows of 2 width.
        check(unsafe { (self.k.dup)(self.g(), x.fp(), m as i64, self.cfg.width as i64, w.cat.fp()) }, "dup")?;
        self.mamba(x, seqs, len, &b.fwd, false, 0)?;
        self.mamba(x, seqs, len, &b.bwd, true, 1)?;
        self.lin(w.cat.ptr(), m, &b.out, w.tmp.ptr())?;
        self.ln(w.tmp.ptr(), m, &b.norm, w.h.ptr())?;
        self.add(x.ptr(), w.h.ptr(), m)
    }

    /// The network over one chunk's spectrum ([frames, bins] of (re, im)): the heads' output, per band [frames, 4 bins]
    /// one after another (band b from 4 frames start_b); `taps` reads stages back
    pub fn run(&self, spec: &[[f32; 2]], frames: usize, mut taps: Option<&mut Taps>, step: &mut dyn FnMut(usize, usize) -> Result<()>) -> Result<Vec<f32>> {
        let w = &self.work;
        let bins = self.cfg.n_fft / 2 + 1;
        if frames == 0 || frames > w.frames || spec.len() != frames * bins {
            return Err(Error(format!("a chunk of {frames} frames (at most {})", w.frames)));
        }
        let (c, nb) = (self.cfg.width, self.edges.len());
        let m = nb * frames;
        // the bands' inputs: band b's [frames, 2 width_b] (re, im of each bin) from 2 frames start_b
        let mut band_in = vec![0f32; frames * bins * 2];
        for &(s, e) in &self.edges {
            let bw = e - s;
            for t in 0..frames {
                for k in 0..bw {
                    let v = spec[t * bins + s + k];
                    band_in[2 * frames * s + t * 2 * bw + 2 * k] = v[0];
                    band_in[2 * frames * s + t * 2 * bw + 2 * k + 1] = v[1];
                }
            }
        }
        // SAFETY: plain floats as bytes.
        w.band_in.write(0, unsafe { std::slice::from_raw_parts(band_in.as_ptr().cast::<u8>(), band_in.len() * 4) })?;
        for (i, &(s, _)) in self.edges.iter().enumerate() {
            let xin = w.band_in.fp().wrapping_add(2 * frames * s);
            let hn = w.h.fp();
            self.ln(xin as P, frames, &self.enc_norm[i], hn.cast())?;
            self.lin(hn as P, frames, &self.enc_lin[i], w.x.fp().wrapping_add(i * frames * c).cast())?;
        }
        // SAFETY: x [bands, frames, width], the embedding [bands, width].
        check(unsafe { (self.k.add_seq)(self.g(), w.x.fp(), nb as i64, frames as i64, c as i64, self.band_emb.fp()) }, "band embedding")?;
        let mut tap = |name: &str, b: &DevBuf, n: usize| -> Result<()> {
            if let Some(t) = taps.as_deref_mut() {
                if t.want.contains(&name) {
                    self.nsd.wait()?;
                    let mut v = b.to_f32()?;
                    v.truncate(n);
                    t.got.push((name.to_string(), v));
                }
            }
            Ok(())
        };
        tap("tokens", &w.x, m * c)?;
        let nblocks = self.blocks.len();
        for (bi, b) in self.blocks.iter().enumerate() {
            match b {
                Block::Attn { t_norm1, t_qkv, t_proj, t_norm2, t_ff1, t_ff2, b_norm1, b_qkv, b_proj, b_norm2, b_ff1, b_ff2 } => {
                    self.attn_layer(&w.x, nb, frames, true, t_norm1, t_qkv, t_proj, t_norm2, t_ff1, t_ff2)?;
                    tap(&format!("block{bi}_time"), &w.x, m * c)?;
                    self.swap(&w.x, nb, frames, &w.xb)?;
                    self.attn_layer(&w.xb, frames, nb, false, b_norm1, b_qkv, b_proj, b_norm2, b_ff1, b_ff2)?;
                    self.swap(&w.xb, frames, nb, &w.x)?;
                }
                Block::Mamba { time, freq } => {
                    self.bimamba(&w.x, nb, frames, time)?;
                    tap(&format!("block{bi}_time"), &w.x, m * c)?;
                    self.swap(&w.x, nb, frames, &w.xb)?;
                    self.bimamba(&w.xb, frames, nb, freq)?;
                    self.swap(&w.xb, frames, nb, &w.x)?;
                }
            }
            tap(&format!("block{bi}"), &w.x, m * c)?;
            step(bi + 1, nblocks + 1)?;
        }
        // the heads, after the conditioning
        // SAFETY: x [m, width], scale and shift [width].
        check(unsafe { (self.k.film)(self.g(), w.x.fp(), m as i64, c as i64, self.film_scale.fp(), self.film_shift.fp()) }, "film")?;
        for (i, &(s, _)) in self.edges.iter().enumerate() {
            let h = &self.heads[i];
            let xi = w.x.fp().wrapping_add(i * frames * c);
            self.ln(xi as P, frames, &h.norm, w.h.ptr())?;
            self.lin(w.h.ptr(), frames, &h.l1, w.tmp.ptr())?;
            self.nsd.gelu(w.tmp.ptr(), Dt::F32, frames * h.l1.n, false)?;
            self.lin(w.tmp.ptr(), frames, &h.l2, w.raw.fp().wrapping_add(4 * frames * s).cast())?;
        }
        self.nsd.wait()?;
        let mut raw = w.raw.to_f32()?;
        raw.truncate(4 * frames * bins);
        step(nblocks + 1, nblocks + 1)?;
        Ok(raw)
    }
}
