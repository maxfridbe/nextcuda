//! The denoiser: LTX-2.5's audio-video DiT (diffusers' LTX2VideoTransformer3DModel, the 2.5 flags), from the GGUF as
//! stored - its 48 blocks split over the cards and resident (Q4_K_M: 6.2 GiB a card), the timestep MLPs on the host.
//!
//! ```text
//!   x = patchify(latents) [Tv, 4096]          y = audio_patchify(audio latents) [Ta, 2048]
//!   block (video, audio side by side; mod_k = table[k] + adaLN(t)[k], per token row: a conditioned frame has t = 0):
//!     x += g_msa  * attn1(rms(x) (1 + sc_msa) + sh_msa)           self, split RoPE (t, h, w), gated heads
//!     y += ag_msa * audio_attn1(...)                               self, RoPE (t)
//!     x += g_tq   * attn2(rms(x) (1 + sc_tq) + sh_tq,  text_v (1 + sc_kv) + sh_kv)    the prompt (kv: adaLN(sigma))
//!     y += ag_tq  * audio_attn2(...)
//!     x += g_a2v  * a2v(rms(x) mod_a2v, rms(y) mod_a2v)            q video, kv audio, 32 x 64, temporal RoPE
//!     y += g_v2a  * v2a(rms(y) mod_v2a, rms(x) mod_v2a)            (both from the norms before either add)
//!     x += g_mlp  * ff(rms(x) (1 + sc_mlp) + sh_mlp)               gelu_tanh, 4 x
//!   out = proj_out(layernorm(x) (1 + sc) + sh),  (sh, sc) = scale_shift_table + embedded_timestep
//! ```
//!
//! Attention q / k are RMS-normed across all heads (weights), then rotated per head (the split RoPE: pairs (i, d/2 + i),
//! each head its own slice of the frequencies). STG (the guidance's perturbed pass) makes block 28's self-attentions
//! return their value projection; modality isolation drops a2v / v2a.

use std::sync::Arc;

use nextcuda_core::{DevBuf, Error, Gpu, Result};
use nextcuda_diffusion::kernels::{none, Dt, Nsd};
use nextcuda_gguf::{GType, Gguf};

use crate::ffi;
use crate::te::Mat;

pub const BLOCKS: usize = 48;
pub const VD: usize = 4096;
pub const AD: usize = 2048;
pub const HEADS: usize = 32;
pub const CH: usize = 128;
pub const TEXT: usize = 1024;
const EPS: f32 = 1e-6;
/// the block whose self-attentions STG perturbs
pub const STG_BLOCK: usize = 28;
/// feed-forward rows a pass (the 16,384-wide hidden in pieces)
const FF_ROWS: usize = 2048;

fn ge(e: nextcuda_gguf::Error) -> Error {
    Error(e.0)
}

/// A tensor of the GGUF as float32 on the host (F32, F16, BF16)
pub fn host(f: &Gguf, name: &str) -> Result<Vec<f32>> {
    let t = f.tensor(name).ok_or_else(|| Error(format!("{}: no tensor {name}", f.paths[0].display())))?;
    let b = f.read(t).map_err(ge)?;
    Ok(match t.ty {
        GType::F32 => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        GType::F16 => b.chunks_exact(2).map(|c| crate::quant::f16_val(u16::from_le_bytes([c[0], c[1]]))).collect(),
        GType::BF16 => b.chunks_exact(2).map(|c| crate::quant::bf16_val(u16::from_le_bytes([c[0], c[1]]))).collect(),
        o => return Err(Error(format!("{name}: {o:?}, a float type expected"))),
    })
}

struct Attn {
    q: Mat,
    k: Mat,
    v: Mat,
    o: Mat,
    g: Mat,
    qb: DevBuf,
    kb: DevBuf,
    vb: DevBuf,
    ob: DevBuf,
    gb: DevBuf,
    qn: DevBuf,
    kn: DevBuf,
}

impl Attn {
    fn load(f: &Gguf, p: &str, gpu: &Arc<Gpu>) -> Result<Attn> {
        let m = |s: &str| Mat::load(f, &format!("{p}.{s}.weight"), gpu);
        let v = |s: &str| -> Result<DevBuf> { DevBuf::from_f32(gpu, &host(f, &format!("{p}.{s}"))?) };
        Ok(Attn {
            q: m("to_q")?, k: m("to_k")?, v: m("to_v")?, o: m("to_out.0")?, g: m("to_gate_logits")?,
            qb: v("to_q.bias")?, kb: v("to_k.bias")?, vb: v("to_v.bias")?, ob: v("to_out.0.bias")?, gb: v("to_gate_logits.bias")?,
            qn: v("q_norm.weight")?, kn: v("k_norm.weight")?,
        })
    }
    fn inner(&self) -> usize {
        self.q.rows
    }
}

struct Ff {
    up: Mat,
    down: Mat,
    up_b: Option<DevBuf>,
    down_b: Option<DevBuf>,
}

struct Block {
    attn1: Attn,
    attn2: Attn,
    audio_attn1: Attn,
    audio_attn2: Attn,
    a2v: Attn,
    v2a: Attn,
    ff: Ff,
    audio_ff: Ff,
    /// the per-block tables (host): [9, 4096], [9, 2048], [2, 4096], [2, 2048], [5, 4096], [5, 2048]
    sst: Vec<f32>,
    asst: Vec<f32>,
    pst: Vec<f32>,
    apst: Vec<f32>,
    cav: Vec<f32>,
    caa: Vec<f32>,
}

/// One block's modulation for a step, its offsets into the card's table buffer (floats)
#[derive(Clone, Copy, Default)]
struct ModAt {
    /// [9][R][4096]
    v: usize,
    /// [9][1][2048]
    a: usize,
    /// prompt kv: [2][4096], [2][2048] (shift, scale)
    pv: usize,
    pa: usize,
    /// a2v / v2a: video [5][4096] (a2v scale, a2v shift, v2a scale, v2a shift, a2v gate), audio [5][2048]
    cv: usize,
    ca: usize,
}

struct Part {
    gpu: Arc<Gpu>,
    nsd: Nsd,
    blocks: Vec<Block>,
    first: usize,
    /// the step's tables (written by `set_step`) and each block's place in them
    mods: std::sync::Mutex<Option<(DevBuf, Vec<ModAt>, usize)>>,
}

/// An adaLN-single MLP on the host: sinusoid(256) -> linear_1 -> silu -> linear_2 = emb; linear(silu(emb))
struct AdaLn {
    l1: Vec<f32>,
    b1: Vec<f32>,
    l2: Vec<f32>,
    b2: Vec<f32>,
    lin: Vec<f32>,
    linb: Vec<f32>,
    dim: usize,
    out: usize,
}

/// out[n] = sum_k w[n, k] x[k] + b[n], on the host's threads
fn gemv(w: &[f32], b: &[f32], x: &[f32], n: usize) -> Vec<f32> {
    let k = x.len();
    let mut out = vec![0f32; n];
    let threads = std::thread::available_parallelism().map_or(8, |v| v.get()).min(16);
    let chunk = n.div_ceil(threads);
    std::thread::scope(|s| {
        for (ci, o) in out.chunks_mut(chunk).enumerate() {
            s.spawn(move || {
                for (j, v) in o.iter_mut().enumerate() {
                    let r = ci * chunk + j;
                    let row = &w[r * k..(r + 1) * k];
                    let mut acc = 0f64;
                    for (a, c) in row.chunks(8).zip(x.chunks(8)) {
                        let mut s8 = 0f32;
                        for i in 0..a.len() {
                            s8 += a[i] * c[i];
                        }
                        acc += s8 as f64;
                    }
                    *v = acc as f32 + b[r];
                }
            });
        }
    });
    out
}

fn silu(v: &[f32]) -> Vec<f32> {
    v.iter().map(|x| x / (1.0 + (-x).exp())).collect()
}

/// diffusers' get_timestep_embedding(t, 256, flip_sin_to_cos=True, downscale_freq_shift=0): [cos, sin]
pub fn sinusoid(t: f32) -> Vec<f32> {
    let half = 128;
    let mut out = vec![0f32; 256];
    for i in 0..half {
        let e = (-(10000f32.ln()) * i as f32 / half as f32).exp();
        let a = t * e;
        out[i] = a.cos();
        out[half + i] = a.sin();
    }
    out
}

impl AdaLn {
    fn load(f: &Gguf, p: &str) -> Result<AdaLn> {
        let h = |s: &str| host(f, &format!("{p}.{s}"));
        let linb = h("linear.bias")?;
        let b1 = h("emb.timestep_embedder.linear_1.bias")?;
        Ok(AdaLn { dim: b1.len(), out: linb.len(), l1: h("emb.timestep_embedder.linear_1.weight")?, b1, l2: h("emb.timestep_embedder.linear_2.weight")?,
                   b2: h("emb.timestep_embedder.linear_2.bias")?, lin: h("linear.weight")?, linb })
    }
    /// (linear(silu(emb)), emb) for timestep t (already x1000)
    fn run(&self, t: f32) -> (Vec<f32>, Vec<f32>) {
        let s = sinusoid(t);
        let h = silu(&gemv(&self.l1, &self.b1, &s, self.dim));
        let emb = gemv(&self.l2, &self.b2, &h, self.dim);
        (gemv(&self.lin, &self.linb, &silu(&emb), self.out), emb)
    }
}

/// LTX2_PROFILE=1: each part of a block timed (the card waited for after it), summed per card and part, logged per pass
static PROFILE: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("LTX2_PROFILE").is_ok_and(|v| v == "1"));
static TIMES: std::sync::Mutex<Vec<(String, f64)>> = std::sync::Mutex::new(Vec::new());

fn tick(part: &Part, what: &str, t: &mut std::time::Instant) -> Result<()> {
    if *PROFILE {
        part.nsd.wait()?;
        let k = format!("{}:{what}", part.gpu.index);
        let dt = t.elapsed().as_secs_f64();
        let mut v = TIMES.lock().unwrap();
        match v.iter_mut().find(|e| e.0 == k) {
            Some(e) => e.1 += dt,
            None => v.push((k, dt)),
        }
        *t = std::time::Instant::now();
    }
    Ok(())
}

/// The profile so far, and cleared
pub fn take_profile() -> Vec<(String, f64)> {
    std::mem::take(&mut *TIMES.lock().unwrap())
}

/// Room each card keeps beside its blocks: the activations of a large clip (1344 x 768, 5 s: about 2.5 GiB)
const HEADROOM: u64 = 5 << 29;

/// The blocks a card: in proportion to its speed (compute units x clock) - the passes are pipelined over the cards,
/// so a card's share of the time is its blocks over its speed - each card at most what fits beside HEADROOM, the rest
/// to the others
fn auto_split(gpus: &[Arc<Gpu>], per_block: u64) -> Result<Vec<usize>> {
    let n = gpus.len();
    let speed: Vec<f64> = gpus.iter().map(|g| nextcuda_core::gpu_units(g.index).map_or(1.0, |(u, c)| if u == 0 { 1.0 } else { u as f64 * c.max(1) as f64 })).collect();
    let total: f64 = speed.iter().sum();
    let mut counts: Vec<usize> = speed.iter().map(|s| (BLOCKS as f64 * s / total).round() as usize).collect();
    let cap: Vec<usize> = gpus.iter().map(|g| {
        let (t, f) = g.memory().unwrap_or((0, None));
        let room = (f.unwrap_or(t) as f64 * 0.94) as u64;
        (room.saturating_sub(HEADROOM) / per_block.max(1)) as usize
    }).collect();
    for i in 0..n {
        counts[i] = counts[i].min(cap[i]);
    }
    // what is left (rounding, caps) to the cards with room, fastest first
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|a, b| speed[*b].total_cmp(&speed[*a]));
    let mut left = BLOCKS as isize - counts.iter().sum::<usize>() as isize;
    while left > 0 {
        let Some(&i) = order.iter().find(|i| counts[**i] < cap[**i]) else { break };
        counts[i] += 1;
        left -= 1;
    }
    while left < 0 {
        let i = *order.last().unwrap();
        let i = if counts[i] > 0 { i } else { order[0] };
        counts[i] -= 1;
        left += 1;
    }
    if counts.iter().sum::<usize>() != BLOCKS {
        return Err(Error(format!("the DiT's {BLOCKS} blocks do not fit these cards beside {:.1} GiB of activations each (at most {cap:?})",
                                 HEADROOM as f64 / (1u64 << 30) as f64)));
    }
    Ok(counts)
}

pub struct Dit {
    parts: Vec<Part>,
    patchify: (DevBuf, DevBuf),
    audio_patchify: (DevBuf, DevBuf),
    proj_out: (DevBuf, DevBuf),
    audio_proj_out: (DevBuf, DevBuf),
    sst_out: Vec<f32>,
    asst_out: Vec<f32>,
    time: AdaLn,
    audio_time: AdaLn,
    prompt: AdaLn,
    audio_prompt: AdaLn,
    ca_video: AdaLn,
    ca_audio: AdaLn,
    a2v_gate: AdaLn,
    v2a_gate: AdaLn,
    /// the output modulation per row of the step: [R][2][4096] and [2][2048]
    out_mod: std::sync::Mutex<(Vec<f32>, Vec<f32>)>,
    pub bytes: u64,
}

/// The positions of a clip: its latent grid and audio length, and the RoPE tables built from them
pub struct Shape {
    pub f: usize,
    pub h: usize,
    pub w: usize,
    pub la: usize,
    pub fps: f32,
}

impl Shape {
    pub fn tv(&self) -> usize {
        self.f * self.h * self.w
    }
}

/// A pass's text conditions on each card: video [1024, 4096], audio [1024, 2048] float32
pub struct Text {
    pub v: Vec<DevBuf>,
    pub a: Vec<DevBuf>,
}

/// What a pass changes: the text, STG, modality isolation
pub struct Pass<'a> {
    pub text: &'a Text,
    pub stg: bool,
    pub isolate: bool,
}

/// Per-card RoPE tables of a shape: (cos, sin) per (token, head) and pair
struct Rope {
    video: DevBuf,
    audio: DevBuf,
    ca_video: DevBuf,
    ca_audio: DevBuf,
}

/// The pass's tables on the cards and its per-token rows
pub struct Prepared {
    ropes: Vec<Rope>,
    rows_v: Vec<DevBuf>,
    rows_a: Vec<DevBuf>,
    /// R: the distinct video timesteps (1, or 2 with a conditioned first frame)
    pub r: usize,
    pub rows: Vec<i32>,
    pub shape: Shape,
}

/// The RoPE frequency table of `dim` / (2 dims): theta^linspace(0, 1, n) pi / 2, float64 then float32
fn freqs(n: usize) -> Vec<f32> {
    (0..n).map(|k| {
        let e = if n > 1 { k as f64 / (n - 1) as f64 } else { 0.0 };
        (10_000f64.powf(e) * std::f64::consts::PI / 2.0) as f32
    }).collect()
}

/// cos / sin per (token, head, pair) from each token's grid fractions (`dims` of them): diffusers'
/// LTX2AudioVideoRotaryPosEmbed (split): angle[k * dims + i] = (2 g_i - 1) f_k, padded in front to dim / 2
fn rope_table(grid: &[Vec<f32>], dim: usize) -> Vec<f32> {
    let dims = grid[0].len();
    let n = dim / (2 * dims);
    let f = freqs(n);
    let half = dim / 2;
    let pad = half - n * dims;
    let mut cs = vec![0f32; grid.len() * half * 2];
    for (t, g) in grid.iter().enumerate() {
        for j in 0..half {
            let (c, s) = if j < pad {
                (1.0, 0.0)
            } else {
                let q = j - pad;
                let (k, i) = (q / dims, q % dims);
                let a = (g[i] * 2.0 - 1.0) * f[k];
                (a.cos(), a.sin())
            };
            cs[(t * half + j) * 2] = c;
            cs[(t * half + j) * 2 + 1] = s;
        }
    }
    cs
}

impl Shape {
    /// prepare_video_coords: pixel bounds [start, end) a latent cell, time causal (first frame 1 pixel frame), / fps;
    /// the midpoints over max_pos (20, 2048, 2048)
    fn video_grid(&self) -> Vec<Vec<f32>> {
        let mut g = Vec::with_capacity(self.tv());
        for f in 0..self.f {
            let s = ((f as f32 * 8.0) + 1.0 - 8.0).max(0.0) / self.fps;
            let e = (((f + 1) as f32 * 8.0) + 1.0 - 8.0).max(0.0) / self.fps;
            let t = (s + e) / 2.0 / 20.0;
            for h in 0..self.h {
                let hm = ((h as f32 * 32.0) + ((h + 1) as f32 * 32.0)) / 2.0 / 2048.0;
                for w in 0..self.w {
                    let wm = ((w as f32 * 32.0) + ((w + 1) as f32 * 32.0)) / 2.0 / 2048.0;
                    g.push(vec![t, hm, wm]);
                }
            }
        }
        g
    }
    /// prepare_audio_coords: mel frames 4 a latent, causal, hop 160 at 16 kHz; the midpoint over 20 s
    fn audio_grid(&self) -> Vec<Vec<f32>> {
        (0..self.la).map(|f| {
            let s = ((f as f32 * 4.0) + 1.0 - 4.0).max(0.0) * 160.0 / 16000.0;
            let e = (((f + 1) as f32 * 4.0) + 1.0 - 4.0).max(0.0) * 160.0 / 16000.0;
            vec![(s + e) / 2.0 / 20.0]
        }).collect()
    }
}

/// Scratch of a pass on one card, sized for its shape
struct Bufs {
    h: DevBuf,
    qf: DevBuf,
    q: DevBuf,
    k: DevBuf,
    v: DevBuf,
    att: DevBuf,
    gl: DevBuf,
    o: DevBuf,
    ff: DevBuf,
    act: DevBuf,
    // the a2v / v2a inputs: video and audio, modulated for each direction
    hv2: DevBuf,
    ha2: DevBuf,
    tm: DevBuf,
}

impl Bufs {
    fn new(gpu: &Arc<Gpu>, tv: usize, ta: usize) -> Result<Bufs> {
        let rows = tv.max(ta).max(TEXT);
        let ffr = FF_ROWS.min(rows);
        Ok(Bufs {
            h: DevBuf::new(gpu, rows * VD * 2)?,
            qf: DevBuf::f32(gpu, rows * VD)?,
            q: DevBuf::new(gpu, rows * VD * 2)?,
            k: DevBuf::new(gpu, rows * VD * 2)?,
            v: DevBuf::new(gpu, rows * VD * 2)?,
            att: DevBuf::new(gpu, rows * VD * 2)?,
            gl: DevBuf::f32(gpu, rows * HEADS)?,
            o: DevBuf::f32(gpu, rows * VD)?,
            ff: DevBuf::f32(gpu, ffr * 4 * VD)?,
            act: DevBuf::new(gpu, ffr * 4 * VD * 2)?,
            hv2: DevBuf::new(gpu, tv * VD * 2)?,
            ha2: DevBuf::new(gpu, ta * AD * 2)?,
            tm: DevBuf::new(gpu, TEXT * VD * 2)?,
        })
    }
}

fn fptr(b: &DevBuf, at: usize) -> *const f32 {
    b.fp().wrapping_add(at).cast_const()
}

fn half_at(b: &DevBuf, n: usize) -> *mut std::ffi::c_void {
    b.ptr().cast::<u16>().wrapping_add(n).cast()
}

impl Dit {
    /// The GGUF's blocks split over `gpus` (`split[i]`: the blocks on card i; None: evenly)
    pub fn load(path: &std::path::Path, gpus: &[Arc<Gpu>], split: Option<&[usize]>, log: &mut dyn FnMut(String)) -> Result<Dit> {
        let t0 = std::time::Instant::now();
        let f = Gguf::open(path).map_err(ge)?;
        let n = gpus.len().max(1);
        let block_bytes = |i: usize| -> u64 {
            let p = format!("transformer_blocks.{i}.");
            f.tensors.iter().filter(|t| t.name.starts_with(&p)).map(|t| t.bytes).sum()
        };
        let counts: Vec<usize> = match split {
            Some(s) if s.len() == n && s.iter().sum::<usize>() == BLOCKS => s.to_vec(),
            _ => auto_split(gpus, (0..BLOCKS).map(block_bytes).sum::<u64>() / BLOCKS as u64)?,
        };
        let mut parts = Vec::new();
        let mut first = 0;
        let mut bytes = 0u64;
        for (pi, gpu) in gpus.iter().enumerate() {
            let range = first..first + counts[pi];
            let need: u64 = range.clone().map(block_bytes).sum();
            if let (_, Some(free)) = gpu.memory()? {
                if need + (1 << 30) > free {
                    return Err(Error(format!("the DiT's blocks {}..{} need {:.2} GiB on {}; {:.2} GiB are free", range.start, range.end,
                                             (need + (1 << 30)) as f64 / (1u64 << 30) as f64, gpu.name, free as f64 / (1u64 << 30) as f64)));
                }
            }
            bytes += need;
            let mut blocks = Vec::new();
            for i in range.clone() {
                let p = format!("transformer_blocks.{i}");
                let a = |s: &str| Attn::load(&f, &format!("{p}.{s}"), gpu);
                let ffm = |s: &str, bias: bool| -> Result<Ff> {
                    let bv = |n: &str| -> Result<Option<DevBuf>> {
                        if bias { Ok(Some(DevBuf::from_f32(gpu, &host(&f, &format!("{p}.{s}.{n}.bias"))?)?)) } else { Ok(None) }
                    };
                    Ok(Ff { up: Mat::load(&f, &format!("{p}.{s}.net.0.proj.weight"), gpu)?, down: Mat::load(&f, &format!("{p}.{s}.net.2.weight"), gpu)?,
                            up_b: bv("net.0.proj")?, down_b: bv("net.2")? })
                };
                let h = |s: &str| host(&f, &format!("{p}.{s}"));
                blocks.push(Block {
                    attn1: a("attn1")?, attn2: a("attn2")?, audio_attn1: a("audio_attn1")?, audio_attn2: a("audio_attn2")?,
                    a2v: a("audio_to_video_attn")?, v2a: a("video_to_audio_attn")?,
                    ff: ffm("ff", f.tensor(&format!("{p}.ff.net.0.proj.bias")).is_some())?, audio_ff: ffm("audio_ff", true)?,
                    sst: h("scale_shift_table")?, asst: h("audio_scale_shift_table")?, pst: h("prompt_scale_shift_table")?,
                    apst: h("audio_prompt_scale_shift_table")?, cav: h("scale_shift_table_a2v_ca_video")?, caa: h("scale_shift_table_a2v_ca_audio")?,
                });
            }
            log(format!("  DiT blocks {}..{} on {}: {:.2} GiB", range.start, range.end, gpu.name, need as f64 / (1u64 << 30) as f64));
            parts.push(Part { nsd: Nsd::new(gpu)?, gpu: gpu.clone(), blocks, first, mods: std::sync::Mutex::new(None) });
            first = range.end;
        }
        let (g0, gl) = (&gpus[0], gpus.last().unwrap());
        let lin = |name: &str, gpu: &Arc<Gpu>| -> Result<(DevBuf, DevBuf)> {
            Ok((DevBuf::from_f32(gpu, &host(&f, &format!("{name}.weight"))?)?, DevBuf::from_f32(gpu, &host(&f, &format!("{name}.bias"))?)?))
        };
        let dit = Dit {
            patchify: lin("patchify_proj", g0)?,
            audio_patchify: lin("audio_patchify_proj", g0)?,
            proj_out: lin("proj_out", gl)?,
            audio_proj_out: lin("audio_proj_out", gl)?,
            sst_out: host(&f, "scale_shift_table")?,
            asst_out: host(&f, "audio_scale_shift_table")?,
            time: AdaLn::load(&f, "adaln_single")?,
            audio_time: AdaLn::load(&f, "audio_adaln_single")?,
            prompt: AdaLn::load(&f, "prompt_adaln_single")?,
            audio_prompt: AdaLn::load(&f, "audio_prompt_adaln_single")?,
            ca_video: AdaLn::load(&f, "av_ca_video_scale_shift_adaln_single")?,
            ca_audio: AdaLn::load(&f, "av_ca_audio_scale_shift_adaln_single")?,
            a2v_gate: AdaLn::load(&f, "av_ca_a2v_gate_adaln_single")?,
            v2a_gate: AdaLn::load(&f, "av_ca_v2a_gate_adaln_single")?,
            out_mod: std::sync::Mutex::new((Vec::new(), Vec::new())),
            parts,
            bytes,
        };
        for p in &dit.parts {
            p.nsd.wait()?;
        }
        log(format!("DiT: {BLOCKS} blocks, {:.2} GiB over {} card(s) in {:.1} s", bytes as f64 / (1u64 << 30) as f64, dit.parts.len(), t0.elapsed().as_secs_f64()));
        Ok(dit)
    }

    pub fn gpus(&self) -> Vec<Arc<Gpu>> {
        self.parts.iter().map(|p| p.gpu.clone()).collect()
    }

    /// A text condition on every card
    pub fn text(&self, v: &[f32], a: &[f32]) -> Result<Text> {
        let mut t = Text { v: Vec::new(), a: Vec::new() };
        for p in &self.parts {
            t.v.push(DevBuf::from_f32(&p.gpu, v)?);
            t.a.push(DevBuf::from_f32(&p.gpu, a)?);
        }
        Ok(t)
    }

    /// The RoPE tables and row tables of a shape; `first_clean`: the first latent frame is a conditioning one (its
    /// tokens use row 1, timestep 0)
    pub fn prepare(&self, shape: Shape, first_clean: bool) -> Result<Prepared> {
        let tv = shape.tv();
        let vg = shape.video_grid();
        let ag = shape.audio_grid();
        let video = rope_table(&vg, VD);
        let audio = rope_table(&ag, AD);
        let ca_video = rope_table(&vg.iter().map(|g| vec![g[0]]).collect::<Vec<_>>(), AD);
        let ca_audio = rope_table(&ag, AD);
        let rows: Vec<i32> = (0..tv).map(|i| i32::from(first_clean && i < shape.h * shape.w)).collect();
        let i32b = |gpu: &Arc<Gpu>, v: &[i32]| -> Result<DevBuf> {
            let b = DevBuf::new(gpu, v.len().max(1) * 4)?;
            b.write(0, &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
            Ok(b)
        };
        let mut p = Prepared { ropes: Vec::new(), rows_v: Vec::new(), rows_a: Vec::new(), r: if first_clean { 2 } else { 1 }, rows: rows.clone(), shape };
        for part in &self.parts {
            let g = &part.gpu;
            p.ropes.push(Rope { video: DevBuf::from_f32(g, &video)?, audio: DevBuf::from_f32(g, &audio)?, ca_video: DevBuf::from_f32(g, &ca_video)?,
                                ca_audio: DevBuf::from_f32(g, &ca_audio)? });
            p.rows_v.push(i32b(g, &rows)?);
            p.rows_a.push(i32b(g, &vec![0; tv.max(p.shape.la).max(TEXT)])?);
        }
        Ok(p)
    }

    /// The step's modulation tables on the cards: timestep `t` (sigma x 1000) for the noisy tokens (and 0 for a
    /// conditioning frame's, row 1), `t` too for the audio, the prompt's and the cross-modal adaLNs (use_cross_timestep)
    pub fn set_step(&self, prep: &Prepared, t: f32) -> Result<()> {
        let r = prep.r;
        let vts: Vec<f32> = if r == 2 { vec![t, 0.0] } else { vec![t] };
        let vt: Vec<(Vec<f32>, Vec<f32>)> = vts.iter().map(|x| self.time.run(*x)).collect();
        let (at, aemb) = self.audio_time.run(t);
        let (pt, _) = self.prompt.run(t);
        let (apt, _) = self.audio_prompt.run(t);
        let (cvt, _) = self.ca_video.run(t);
        let (cat, _) = self.ca_audio.run(t);
        let (gvt, _) = self.a2v_gate.run(t);
        let (gat, _) = self.v2a_gate.run(t);
        // the output layer's (shift, scale) per row
        let mut om = vec![0f32; r * 2 * VD];
        for (ri, (_, emb)) in vt.iter().enumerate() {
            for k in 0..2 {
                for c in 0..VD {
                    om[(ri * 2 + k) * VD + c] = self.sst_out[k * VD + c] + emb[c];
                }
            }
        }
        let mut aom = vec![0f32; 2 * AD];
        for k in 0..2 {
            for c in 0..AD {
                aom[k * AD + c] = self.asst_out[k * AD + c] + aemb[c];
            }
        }
        *self.out_mod.lock().unwrap() = (om, aom);
        for part in &self.parts {
            let mut buf: Vec<f32> = Vec::new();
            let mut ats = Vec::new();
            for b in &part.blocks {
                let mut m = ModAt { v: buf.len(), ..Default::default() };
                for k in 0..9 {
                    for (temb, _) in &vt {
                        buf.extend((0..VD).map(|c| b.sst[k * VD + c] + temb[k * VD + c]));
                    }
                }
                m.a = buf.len();
                for k in 0..9 {
                    buf.extend((0..AD).map(|c| b.asst[k * AD + c] + at[k * AD + c]));
                }
                m.pv = buf.len();
                for k in 0..2 {
                    buf.extend((0..VD).map(|c| b.pst[k * VD + c] + pt[k * VD + c]));
                }
                m.pa = buf.len();
                for k in 0..2 {
                    buf.extend((0..AD).map(|c| b.apst[k * AD + c] + apt[k * AD + c]));
                }
                m.cv = buf.len();
                for k in 0..4 {
                    buf.extend((0..VD).map(|c| b.cav[k * VD + c] + cvt[k * VD + c]));
                }
                buf.extend((0..VD).map(|c| b.cav[4 * VD + c] + gvt[c]));
                m.ca = buf.len();
                for k in 0..4 {
                    buf.extend((0..AD).map(|c| b.caa[k * AD + c] + cat[k * AD + c]));
                }
                buf.extend((0..AD).map(|c| b.caa[4 * AD + c] + gat[c]));
                ats.push(m);
            }
            let d = DevBuf::from_f32(&part.gpu, &buf)?;
            *part.mods.lock().unwrap() = Some((d, ats, r));
        }
        Ok(())
    }

    /// The latents through the patchify projections, on the first card
    fn patchify(&self, prep: &Prepared, lat: &[f32], aud: &[f32]) -> Result<(DevBuf, DevBuf)> {
        let (tv, ta) = (prep.shape.tv(), prep.shape.la);
        let p0 = &self.parts[0];
        let x = DevBuf::f32(&p0.gpu, tv * VD)?;
        let y = DevBuf::f32(&p0.gpu, ta * AD)?;
        let l = DevBuf::from_f32(&p0.gpu, lat)?;
        let a = DevBuf::from_f32(&p0.gpu, aud)?;
        p0.nsd.linear(l.ptr(), Dt::F32, tv, CH, self.patchify.0.ptr(), VD, self.patchify.1.ptr(), x.ptr(), Dt::F32)?;
        p0.nsd.linear(a.ptr(), Dt::F32, ta, CH, self.audio_patchify.0.ptr(), AD, self.audio_patchify.1.ptr(), y.ptr(), Dt::F32)?;
        p0.nsd.wait()?;
        Ok((x, y))
    }

    /// One card's blocks over x, y (on that card), in place; `capture`: the states after those blocks
    #[allow(clippy::too_many_arguments)]
    fn run_part(&self, pi: usize, prep: &Prepared, pass: &Pass, x: &DevBuf, y: &DevBuf, b: &Bufs, capture: &[usize],
                caps: &mut Vec<(usize, Vec<f32>, Vec<f32>)>) -> Result<()> {
        let part = &self.parts[pi];
        let guard = part.mods.lock().unwrap();
        let (mods, ats, r) = guard.as_ref().ok_or("set_step first")?;
        for (j, blk) in part.blocks.iter().enumerate() {
            let bi = part.first + j;
            self.block(part, pi, prep, blk, &ats[j], mods, *r, b, x, y, pass, bi)?;
            if capture.contains(&bi) {
                part.nsd.wait()?;
                caps.push((bi, x.to_f32()?, y.to_f32()?));
            }
        }
        part.nsd.wait()
    }

    /// The output layers on the last card: velocities video [Tv, 128], audio [La, 128]
    fn output(&self, prep: &Prepared, x: &DevBuf, y: &DevBuf) -> Result<(Vec<f32>, Vec<f32>)> {
        let (tv, ta) = (prep.shape.tv(), prep.shape.la);
        let pl = self.parts.last().unwrap();
        let (g, nsd) = (&pl.gpu, &pl.nsd);
        let (om, aom) = self.out_mod.lock().unwrap().clone();
        let r = prep.r;
        // ln_mod's tables: shift [R, C] and scale [R, C] from om [R][2][C]
        let mut sh = vec![0f32; r * VD];
        let mut sc = vec![0f32; r * VD];
        for ri in 0..r {
            sh[ri * VD..(ri + 1) * VD].copy_from_slice(&om[(ri * 2) * VD..(ri * 2 + 1) * VD]);
            sc[ri * VD..(ri + 1) * VD].copy_from_slice(&om[(ri * 2 + 1) * VD..(ri * 2 + 2) * VD]);
        }
        let (shd, scd) = (DevBuf::from_f32(g, &sh)?, DevBuf::from_f32(g, &sc)?);
        let n = DevBuf::f32(g, tv * VD)?;
        let rows = &prep.rows_v[self.parts.len() - 1];
        ffi::ln_mod(g, x.fp(), tv, VD, 1e-6, rows.ptr().cast(), scd.fp(), shd.fp(), n.fp())?;
        let out = DevBuf::f32(g, tv * CH)?;
        nsd.linear(n.ptr(), Dt::F32, tv, VD, self.proj_out.0.ptr(), CH, self.proj_out.1.ptr(), out.ptr(), Dt::F32)?;
        let (ashd, ascd) = (DevBuf::from_f32(g, &aom[..AD])?, DevBuf::from_f32(g, &aom[AD..])?);
        let na = DevBuf::f32(g, ta * AD)?;
        ffi::ln_mod(g, y.fp(), ta, AD, 1e-6, std::ptr::null(), ascd.fp(), ashd.fp(), na.fp())?;
        let aout = DevBuf::f32(g, ta * CH)?;
        nsd.linear(na.ptr(), Dt::F32, ta, AD, self.audio_proj_out.0.ptr(), CH, self.audio_proj_out.1.ptr(), aout.ptr(), Dt::F32)?;
        nsd.wait()?;
        Ok((out.to_f32()?, aout.to_f32()?))
    }

    /// One forward: velocities video [Tv, 128], audio [La, 128]; `capture`: (block, video, audio) states after those
    /// blocks
    #[allow(clippy::type_complexity)]
    pub fn forward(&self, prep: &Prepared, lat: &[f32], aud: &[f32], pass: &Pass, capture: &[usize]) -> Result<(Vec<f32>, Vec<f32>, Vec<(usize, Vec<f32>, Vec<f32>)>)> {
        let (tv, ta) = (prep.shape.tv(), prep.shape.la);
        let mut caps = Vec::new();
        let (mut x, mut y) = self.patchify(prep, lat, aud)?;
        let mut staging = Vec::new();
        for (pi, part) in self.parts.iter().enumerate() {
            if pi > 0 {
                let (nx, ny) = (DevBuf::f32(&part.gpu, tv * VD)?, DevBuf::f32(&part.gpu, ta * AD)?);
                staging.resize(x.len.max(y.len), 0u8);
                nx.copy_from_peer(&x, &mut staging)?;
                ny.copy_from_peer(&y, &mut staging)?;
                x = nx;
                y = ny;
            }
            let b = Bufs::new(&part.gpu, tv, ta)?;
            self.run_part(pi, prep, pass, &x, &y, &b, capture, &mut caps)?;
        }
        let (v, a) = self.output(prep, &x, &y)?;
        Ok((v, a, caps))
    }

    /// Several passes over the same latents, pipelined over the cards: while a card runs its blocks of one pass, the
    /// next card runs its blocks of the pass before (each card a thread; the states handed on through two slots a
    /// card, copied between cards through the host). The velocities of each pass, in order.
    pub fn forward_passes(&self, prep: &Prepared, lat: &[f32], aud: &[f32], passes: &[Pass]) -> Result<Vec<(Vec<f32>, Vec<f32>)>> {
        let (tv, ta) = (prep.shape.tv(), prep.shape.la);
        let n = self.parts.len();
        if n == 1 || passes.len() == 1 {
            return passes.iter().map(|p| self.forward(prep, lat, aud, p, &[]).map(|(v, a, _)| (v, a))).collect();
        }
        use std::sync::mpsc::{channel, Receiver, Sender};
        // each card's two slots of (x, y)
        let slots: Vec<Vec<(DevBuf, DevBuf)>> = self.parts.iter().map(|p| {
            (0..2).map(|_| Ok((DevBuf::f32(&p.gpu, tv * VD)?, DevBuf::f32(&p.gpu, ta * AD)?))).collect::<Result<Vec<_>>>()
        }).collect::<Result<_>>()?;
        // card i -> card i + 1: "pass p is in slot s"; and back: "slot s is free"
        let mut fwd_tx: Vec<Option<Sender<(usize, usize)>>> = Vec::new();
        let mut fwd_rx: Vec<Option<Receiver<(usize, usize)>>> = vec![None];
        let mut free_tx: Vec<Option<Sender<usize>>> = vec![None];
        let mut free_rx: Vec<Option<Receiver<usize>>> = Vec::new();
        for _ in 0..n - 1 {
            let (a, b) = channel();
            fwd_tx.push(Some(a));
            fwd_rx.push(Some(b));
            let (c, d) = channel();
            free_tx.push(Some(c));
            free_rx.push(Some(d));
        }
        fwd_tx.push(None);
        free_rx.push(None);
        let results: std::sync::Mutex<Vec<Option<(Vec<f32>, Vec<f32>)>>> = std::sync::Mutex::new(vec![None; passes.len()]);
        // every card's error: the first that is not a neighbour stopping is the cause
        let first_err: std::sync::Mutex<Vec<Error>> = std::sync::Mutex::new(Vec::new());
        std::thread::scope(|sc| {
            for pi in 0..n {
                let (ftx, frx, rtx, rrx) = (fwd_tx[pi].take(), fwd_rx[pi].take(), free_tx[pi].take(), free_rx[pi].take());
                let (slots, results, first_err) = (&slots, &results, &first_err);
                sc.spawn(move || {
                    let run = || -> Result<()> {
                        let part = &self.parts[pi];
                        let b = Bufs::new(&part.gpu, tv, ta)?;
                        let mut staging = Vec::new();
                        let mut caps = Vec::new();
                        let mut free_slots: Vec<usize> = vec![0, 1];
                        for (p, pass) in passes.iter().enumerate() {
                            // a free slot of this card (the next card hands them back once it has copied)
                            let s = match free_slots.pop() {
                                Some(s) => s,
                                None => rrx.as_ref().ok_or("no slot")?.recv().map_err(|_| Error("the next card stopped".into()))?,
                            };
                            let (x, y) = &slots[pi][s];
                            if pi == 0 {
                                let (px, py) = self.patchify(prep, lat, aud)?;
                                x.copy_within(0, &px, 0, px.len)?;
                                y.copy_within(0, &py, 0, py.len)?;
                            } else {
                                let (q, ps) = frx.as_ref().unwrap().recv().map_err(|_| Error("the card before stopped".into()))?;
                                if q != p {
                                    return Err(Error(format!("pipelined passes out of order: {q} for {p}")));
                                }
                                let (sx, sy) = &slots[pi - 1][ps];
                                staging.resize(sx.len.max(sy.len), 0u8);
                                x.copy_from_peer(sx, &mut staging)?;
                                y.copy_from_peer(sy, &mut staging)?;
                                // (the card before may be done with every pass, and gone: nothing to tell it)
                                let _ = rtx.as_ref().unwrap().send(ps);
                            }
                            self.run_part(pi, prep, pass, x, y, &b, &[], &mut caps)?;
                            if pi + 1 < n {
                                ftx.as_ref().unwrap().send((p, s)).map_err(|_| Error("the next card stopped".into()))?;
                            } else {
                                let r = self.output(prep, x, y)?;
                                results.lock().unwrap()[p] = Some(r);
                                free_slots.push(s);
                            }
                        }
                        Ok(())
                    };
                    if let Err(e) = run() {
                        first_err.lock().unwrap().push(Error(format!("card {}: {}", self.parts[pi].gpu.index, e.0)));
                    }
                });
            }
        });
        let errs = first_err.into_inner().unwrap();
        if let Some(e) = errs.iter().find(|e| !e.0.contains(" stopped")).or(errs.first()) {
            return Err(Error(e.0.clone()));
        }
        results.into_inner().unwrap().into_iter().map(|r| r.ok_or_else(|| Error("a pass gave no result".into()))).collect()
    }

    /// An attention: q from `hq` [m, qdim] half, k / v from `hkv` [n, kvdim] half, into `out` float32 [m, out]
    /// (the to_out projection, its bias); `rope`: the q and k tables (None: none); `skip`: STG - the value alone
    #[allow(clippy::too_many_arguments)]
    fn attention(&self, part: &Part, a: &Attn, hq: *const std::ffi::c_void, m: usize, hkv: *const std::ffi::c_void, n: usize, rope: Option<(&DevBuf, &DevBuf)>,
                 skip: bool, b: &Bufs) -> Result<()> {
        let (g, nsd) = (&part.gpu, &part.nsd);
        let inner = a.inner();
        let d = inner / HEADS;
        a.g.apply(nsd, hq, m, b.gl.ptr(), Dt::F32, 0, a.gb.fp())?;
        if skip {
            a.v.apply(nsd, hkv, n, b.att.ptr(), Dt::F16, 0, a.vb.fp())?;
        } else {
            a.q.apply(nsd, hq, m, b.qf.ptr(), Dt::F32, 0, a.qb.fp())?;
            nsd.rms_norm_mod(b.qf.ptr(), Dt::F32, m, inner, a.qn.ptr(), EPS, none(), none(), none(), b.q.ptr(), Dt::F16)?;
            a.k.apply(nsd, hkv, n, b.qf.ptr(), Dt::F32, 0, a.kb.fp())?;
            nsd.rms_norm_mod(b.qf.ptr(), Dt::F32, n, inner, a.kn.ptr(), EPS, none(), none(), none(), b.k.ptr(), Dt::F16)?;
            a.v.apply(nsd, hkv, n, b.v.ptr(), Dt::F16, 0, a.vb.fp())?;
            if let Some((rq, rk)) = rope {
                nsd.rope(b.q.ptr(), Dt::F16, m * HEADS, 1, d, d, rq.ptr(), d)?;
                nsd.rope(b.k.ptr(), Dt::F16, n * HEADS, 1, d, d, rk.ptr(), d)?;
            }
            if m == n && std::ptr::eq(hq, hkv) {
                nsd.attention(b.q.ptr(), b.k.ptr(), b.v.ptr(), Dt::F16, m, HEADS, d, inner, b.att.ptr(), Dt::F16)?;
            } else {
                nsd.attention_qk(b.q.ptr(), m, inner, b.k.ptr(), b.v.ptr(), n, inner, Dt::F16, HEADS, d, b.att.ptr(), Dt::F16)?;
            }
        }
        ffi::head_gate(g, b.att.ptr(), m, HEADS, d, b.gl.fp())?;
        a.o.apply(nsd, b.att.ptr(), m, b.o.ptr(), Dt::F32, 0, a.ob.fp())
    }

    #[allow(clippy::too_many_arguments)]
    fn ff(&self, part: &Part, f: &Ff, x: &DevBuf, rows: usize, dim: usize, rows_t: *const i32, sc: *const f32, sh: *const f32, gate: *const f32, b: &Bufs) -> Result<()> {
        let (g, nsd) = (&part.gpu, &part.nsd);
        let hid = 4 * dim;
        nsd.rms_norm_mod(x.ptr(), Dt::F32, rows, dim, none(), EPS, rows_t.cast(), sc.cast(), sh.cast(), b.h.ptr(), Dt::F16)?;
        let mut r0 = 0;
        while r0 < rows {
            let n = FF_ROWS.min(rows - r0);
            f.up.apply(nsd, half_at(&b.h, r0 * dim), n, b.ff.ptr(), Dt::F32, 0, f.up_b.as_ref().map_or(std::ptr::null(), |v| v.fp().cast_const()))?;
            ffi::geglu(g, b.ff.fp(), hid, std::ptr::null(), 0, n, hid, 1.0, b.act.ptr())?;
            f.down.apply(nsd, b.act.ptr(), n, b.o.fp().wrapping_add(r0 * dim).cast(), Dt::F32, 0, f.down_b.as_ref().map_or(std::ptr::null(), |v| v.fp().cast_const()))?;
            r0 += n;
        }
        nsd.gate_add(x.ptr(), Dt::F32, rows, dim, b.o.ptr(), Dt::F32, rows_t.cast(), gate.cast())
    }

    #[allow(clippy::too_many_arguments)]
    fn block(&self, part: &Part, pi: usize, prep: &Prepared, blk: &Block, at: &ModAt, mods: &DevBuf, r: usize, b: &Bufs, x: &DevBuf, y: &DevBuf, pass: &Pass,
             bi: usize) -> Result<()> {
        let nsd = &part.nsd;
        let tv = prep.shape.tv();
        let ta = prep.shape.la;
        let rv = prep.rows_v[pi].ptr().cast_const().cast::<i32>();
        let ra = prep.rows_a[pi].ptr().cast_const().cast::<i32>();
        let rope = &prep.ropes[pi];
        let vk = |k: usize| fptr(mods, at.v + k * r * VD);
        let ak = |k: usize| fptr(mods, at.a + k * AD);
        let skip = pass.stg && bi == STG_BLOCK;
        let mut t = std::time::Instant::now();
        // 1. self-attention, video then audio
        nsd.rms_norm_mod(x.ptr(), Dt::F32, tv, VD, none(), EPS, rv.cast(), vk(1).cast(), vk(0).cast(), b.h.ptr(), Dt::F16)?;
        self.attention(part, &blk.attn1, b.h.ptr(), tv, b.h.ptr(), tv, Some((&rope.video, &rope.video)), skip, b)?;
        nsd.gate_add(x.ptr(), Dt::F32, tv, VD, b.o.ptr(), Dt::F32, rv.cast(), vk(2).cast())?;
        tick(part, "video self", &mut t)?;
        nsd.rms_norm_mod(y.ptr(), Dt::F32, ta, AD, none(), EPS, ra.cast(), ak(1).cast(), ak(0).cast(), b.h.ptr(), Dt::F16)?;
        self.attention(part, &blk.audio_attn1, b.h.ptr(), ta, b.h.ptr(), ta, Some((&rope.audio, &rope.audio)), skip, b)?;
        nsd.gate_add(y.ptr(), Dt::F32, ta, AD, b.o.ptr(), Dt::F32, ra.cast(), ak(2).cast())?;
        // 2. the prompt: q modulated by (shift, scale, gate)_text_q, the text by the prompt adaLN's (shift, scale)
        let g = &part.gpu;
        nsd.rms_norm_mod(x.ptr(), Dt::F32, tv, VD, none(), EPS, rv.cast(), vk(7).cast(), vk(6).cast(), b.h.ptr(), Dt::F16)?;
        ffi::mod_rows(g, pass.text.v[pi].fp(), TEXT, VD, fptr(mods, at.pv + VD), fptr(mods, at.pv), 1.0, b.tm.ptr())?;
        self.attention(part, &blk.attn2, b.h.ptr(), tv, b.tm.ptr(), TEXT, None, false, b)?;
        nsd.gate_add(x.ptr(), Dt::F32, tv, VD, b.o.ptr(), Dt::F32, rv.cast(), vk(8).cast())?;
        tick(part, "video text", &mut t)?;
        nsd.rms_norm_mod(y.ptr(), Dt::F32, ta, AD, none(), EPS, ra.cast(), ak(7).cast(), ak(6).cast(), b.h.ptr(), Dt::F16)?;
        ffi::mod_rows(g, pass.text.a[pi].fp(), TEXT, AD, fptr(mods, at.pa + AD), fptr(mods, at.pa), 1.0, b.tm.ptr())?;
        self.attention(part, &blk.audio_attn2, b.h.ptr(), ta, b.tm.ptr(), TEXT, None, false, b)?;
        nsd.gate_add(y.ptr(), Dt::F32, ta, AD, b.o.ptr(), Dt::F32, ra.cast(), ak(8).cast())?;
        tick(part, "audio self+text", &mut t)?;
        // 3. audio <-> video, both from the norms before either update
        if !pass.isolate {
            let cv = |k: usize| fptr(mods, at.cv + k * VD);
            let ca = |k: usize| fptr(mods, at.ca + k * AD);
            // a2v: video (a2v scale 0, shift 1) queries, audio (a2v 0, 1) keys
            nsd.rms_norm_mod(x.ptr(), Dt::F32, tv, VD, none(), EPS, ra.cast(), cv(0).cast(), cv(1).cast(), b.h.ptr(), Dt::F16)?;
            nsd.rms_norm_mod(y.ptr(), Dt::F32, ta, AD, none(), EPS, ra.cast(), ca(0).cast(), ca(1).cast(), b.ha2.ptr(), Dt::F16)?;
            // v2a: audio (v2a scale 2, shift 3) queries, video (v2a 2, 3) keys - made now, before x changes
            nsd.rms_norm_mod(x.ptr(), Dt::F32, tv, VD, none(), EPS, ra.cast(), cv(2).cast(), cv(3).cast(), b.hv2.ptr(), Dt::F16)?;
            self.attention(part, &blk.a2v, b.h.ptr(), tv, b.ha2.ptr(), ta, Some((&rope.ca_video, &rope.ca_audio)), false, b)?;
            nsd.gate_add(x.ptr(), Dt::F32, tv, VD, b.o.ptr(), Dt::F32, ra.cast(), cv(4).cast())?;
            nsd.rms_norm_mod(y.ptr(), Dt::F32, ta, AD, none(), EPS, ra.cast(), ca(2).cast(), ca(3).cast(), b.ha2.ptr(), Dt::F16)?;
            // (y is unchanged so far in this section: its norm is the one before the update)
            self.attention(part, &blk.v2a, b.ha2.ptr(), ta, b.hv2.ptr(), tv, Some((&rope.ca_audio, &rope.ca_video)), false, b)?;
            nsd.gate_add(y.ptr(), Dt::F32, ta, AD, b.o.ptr(), Dt::F32, ra.cast(), ca(4).cast())?;
        }
        tick(part, "a2v v2a", &mut t)?;
        // 4. feed-forwards
        self.ff(part, &blk.ff, x, tv, VD, rv, vk(4), vk(3), vk(5), b)?;
        tick(part, "video ff", &mut t)?;
        self.ff(part, &blk.audio_ff, y, ta, AD, ra, ak(4), ak(3), ak(5), b)?;
        tick(part, "audio ff", &mut t)
    }
}
