//! Inkling-2-Mini (`inkling2_loop`, llama.cpp's src/models/inkling2.cpp and inkling2-loop.cpp; Thinking Machines'
//! Inkling2LoopForConditionalGeneration): 24 decoder layers of 2,304, the middle twelve (6..18) run three times with
//! the same weights - 48 layer applications a token, each with its own caches. A layer:
//!
//! ```text
//!   inputs   = mix(blocks, x)               attention over the depth (block residuals, normed values); 3 queries
//!                                           (q / k / v) on the full layers, one on the windowed ones; layer 0 none
//!   passes 2-3: inputs[j] = (1 - tanh F u) inputs[j] + tanh(G u) msg[j]   u = D [rms(input) | rms(msg)]
//!   q, k, v  = W (rms(inputs) . norm)        k, v through causal short convolutions (4 taps; the full layers add
//!                                           input-dependent taps per head), q / k L2-clipped, v RMS-normed
//!   attn     = softmax(q k / sqrt 128 + r . relbias[distance] ; sink) v      r [16] a head, sink a logit of no value;
//!                                           windowed layers 1,024 back; full layers scale q, r by 1 + 0.1 ln(pos / 8192)
//!   out      = conv(Wo (attn * sigmoid(Wg x)))
//!   x        = out (a block's first layer: the old x joins the blocks) or x + out
//!   x       += conv(MoE(rms(mix(blocks, x)) . norm))   128 experts, 8 by sigmoid + bias, SiTU, NVFP4 with scales
//! ```
//!
//! Passes 2 and 3 of the loop start from the prelude's output again, their inputs mixed with messages from the
//! previous pass (its blocks after the first and its output). The head: mix, norm, the output matrix, soft cap 30, the
//! padded vocabulary dropped. The weights stay in their GGUF blocks (Q8_0, BF16, NVFP4, F32): decode reads them
//! through this engine's own products, a prompt chunk expands each matrix (an expert at a time) to half and multiplies.
//!
//! nextsycl's llm/inkling2 engine (the same steps, kernels of the same names), over one GPU or two: when the model does
//! not fit one card, the looped layers (36 of the 48 applications) and as many of their neighbours as it has room for go to
//! the faster card, the rest of the prelude and the coda and the head to the other - a token crosses between them twice, carrying the stream and the depth attention's blocks
//! (each a row of 2,304 floats). Each card has its own work memory and its own silo binding (a variant per card).

use std::ffi::c_int;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use nextcuda_core::{Arena, DevBuf, Gpu, HostBuf};
use nextcuda_gguf::{GType, Gguf, Tensor};
use nextcuda_llm::{Error, GpuInfo, Result, Sampler, Tap};

use crate::ffi;

mod dspark;

fn e(x: impl std::fmt::Display) -> Error {
    Error(x.to_string())
}

/// The model's shape, from the file's metadata
#[derive(Clone, Debug)]
pub struct Geometry {
    /// physical layers
    pub layers: usize,
    pub n_embd: usize,
    pub heads: usize,
    pub heads_kv: usize,
    pub head: usize,
    pub eps: f32,
    /// the sliding window (positions a windowed layer's row sees, itself included)
    pub window: usize,
    /// each physical layer: windowed or full
    pub swa: Vec<bool>,
    pub rel_len: usize,
    pub rel_extent: usize,
    pub qk_clip: f32,
    pub rel_clip: f32,
    pub log_floor: f32,
    pub log_alpha: f32,
    pub conv: usize,
    pub block: usize,
    pub experts: usize,
    pub used: usize,
    pub n_ff: usize,
    pub w_scale: f32,
    pub situ_beta: f32,
    pub situ_lb: f32,
    pub softcap: f32,
    /// the loop: [core_start, core_end) run `passes` times
    pub core: (usize, usize),
    pub passes: usize,
    pub rank: usize,
    /// the output matrix's rows, and the tokenizer's (the logits a row)
    pub vocab_pad: usize,
    pub vocab: usize,
    pub ctx: usize,
}

/// The most rows the decode path takes in a pass (a verify pass); prompt rows beyond
pub const MAXV: usize = 16;

impl Geometry {
    pub fn read(f: &Gguf) -> std::result::Result<Geometry, String> {
        if f.architecture() == "inkling2" {
            return Err("an inkling2 file in Unsloth's root layout (the loop unrolled into 48 slots): this engine reads Mainline_Support/*.gguf_file \
                        (inkling2_loop, the layout llama.cpp's graph names)".into());
        }
        let u = |k: &str| f.arch_meta(k).and_then(|v| v.as_u64()).map(|v| v as usize).ok_or_else(|| format!("no {}.{k}", f.architecture()));
        let fl = |k: &str| f.arch_meta(k).and_then(|v| v.as_f64()).map(|v| v as f32).ok_or_else(|| format!("no {}.{k}", f.architecture()));
        // a per-layer array's first value, or the value itself
        let first = |k: &str| match f.arch_meta(k) {
            Some(v) if v.as_array().is_some() => v.as_array().and_then(|a| a.first()).and_then(|x| x.as_u64()).map(|x| x as usize),
            Some(v) => v.as_u64().map(|x| x as usize),
            None => None,
        }.ok_or_else(|| format!("no {}.{k}", f.architecture()));
        let layers = u("block_count")?;
        let swa: Vec<bool> = f.arch_meta("attention.sliding_window_pattern").and_then(|v| v.as_array())
            .map(|a| a.iter().map(|x| x.as_u64().unwrap_or(0) != 0).collect()).ok_or("no attention.sliding_window_pattern")?;
        if swa.len() != layers {
            return Err(format!("attention.sliding_window_pattern has {} flags for {layers} layers", swa.len()));
        }
        let g = Geometry {
            layers,
            n_embd: u("embedding_length")?,
            heads: first("attention.head_count")?,
            heads_kv: first("attention.head_count_kv")?,
            head: u("attention.key_length")?,
            eps: fl("attention.layer_norm_rms_epsilon")?,
            window: u("attention.sliding_window")?,
            swa,
            rel_len: u("attention.relative_length")?,
            rel_extent: u("attention.relative_extent")?,
            qk_clip: fl("attention.qk_clip")?,
            rel_clip: fl("attention.relative_clip")?,
            log_floor: u("attention.log_scale_floor")? as f32,
            log_alpha: fl("attention.log_scale_alpha")?,
            conv: u("shortconv.l_cache")?,
            block: u("attn_res.block_size")?,
            experts: u("expert_count")?,
            used: u("expert_used_count")?,
            n_ff: first("expert_feed_forward_length")?,
            w_scale: fl("expert_weights_scale").unwrap_or(1.0),
            situ_beta: fl("activation.situ_beta")?,
            situ_lb: fl("activation.situ_linear_beta")?,
            softcap: fl("final_logit_softcapping").unwrap_or(0.0),
            core: (u("loop.start")?, u("loop.end")?),
            passes: u("loop.count")?,
            rank: u("loop.gate_rank")?,
            vocab_pad: u("vocab_size")?,
            vocab: u("vocab_size_unpadded")?,
            ctx: u("context_length").unwrap_or(1 << 20),
        };
        if g.head != 128 || g.heads != 3 * g.heads_kv || g.rel_len > 16 || g.conv != 4 {
            return Err(format!("heads of {} ({} / {}), a bias of {}, {} taps: this engine takes heads of 128, 3 query heads a key head, a bias of at most 16 \
                                and 4 taps", g.head, g.heads, g.heads_kv, g.rel_len, g.conv));
        }
        if f.arch_meta("loop.varying").and_then(|v| v.as_u64()).is_some_and(|v| v != 0) {
            return Err("a varying loop: not this engine's".into());
        }
        if g.core.0 == 0 || g.core.0 >= g.core.1 || g.core.1 > g.layers || g.passes < 2 || !g.core.0.is_multiple_of(g.block) || !g.core.1.is_multiple_of(g.block) {
            return Err(format!("a loop of [{}, {}) x {} over {} layers (blocks of {})", g.core.0, g.core.1, g.passes, g.layers, g.block));
        }
        if g.used > 16 || g.experts > 256 {
            return Err(format!("{} of {} experts: at most 16 of 256", g.used, g.experts));
        }
        Ok(g)
    }

    /// Cache slots (layer applications): the physical layers, then each later pass's core
    pub fn slots(&self) -> usize {
        self.layers + (self.passes - 1) * (self.core.1 - self.core.0)
    }

    /// The physical layer of slot `s`
    pub fn phys(&self, s: usize) -> usize {
        if s < self.layers { s } else { self.core.0 + (s - self.layers) % (self.core.1 - self.core.0) }
    }

    /// Whether physical layer `l` is one of the loop's
    pub fn looped(&self, l: usize) -> bool {
        l >= self.core.0 && l < self.core.1
    }

    /// The order a pass runs: (slot, physical layer, pass)
    pub fn order(&self) -> Vec<(usize, usize, usize)> {
        let (a, b) = self.core;
        let mut v: Vec<(usize, usize, usize)> = (0..a).map(|l| (l, l, 0)).collect();
        for p in 0..self.passes {
            for l in a..b {
                let s = if p == 0 { l } else { self.layers + (p - 1) * (b - a) + (l - a) };
                v.push((s, l, p));
            }
        }
        v.extend((b..self.layers).map(|l| (l, l, 0)));
        v
    }

    /// A slot's ring of keys for a context and passes of `chunk` rows: a windowed layer's holds its window and a pass
    pub fn cap(&self, slot: usize, max_ctx: usize, chunk: usize) -> usize {
        if self.swa[self.phys(slot)] { max_ctx.min(self.window + chunk - 1) } else { max_ctx }
    }

    /// The window a slot's rows see (0: all)
    pub fn win(&self, slot: usize) -> usize {
        if self.swa[self.phys(slot)] { self.window } else { 0 }
    }

    /// A slot's convolution state's channel groups: k, v, the attention's output, the FFN's
    pub fn conv_groups(&self) -> [usize; 4] {
        let kd = self.heads_kv * self.head;
        [kd, kd, self.n_embd, self.n_embd]
    }
}

/// A matrix [rows (outputs), cols (inputs)] in its stored type (F32, BF16, Q8_0, NVFP4)
pub struct Mat {
    pub buf: DevBuf,
    pub ty: GType,
    pub rows: usize,
    pub cols: usize,
    /// repacked by the silo (silo.h: the quants apart from the scales); only the silo's kernels read it
    pub packed: bool,
}

/// A layer's routed experts: [experts][rows][cols] stored blocks (an expert `stride` bytes), or the silo's repacking
/// of the stack as one matrix of experts x rows; scale: a float an expert
pub struct Experts {
    pub buf: DevBuf,
    pub ty: GType,
    pub rows: usize,
    pub cols: usize,
    pub stride: usize,
    pub packed: bool,
    pub scale: Option<DevBuf>,
    /// the scales on the host (the prompt path's per-row factors)
    pub scale_h: Vec<f32>,
}

pub struct Layer {
    /// the engine's GPU it lives on (`Inkling2::devs`)
    dev: usize,
    swa: bool,
    /// windowed: one norm for q, k and v; full: one each
    norm: [DevBuf; 3],
    q: Mat,
    k: Mat,
    v: Mat,
    o: Mat,
    gate: Mat,
    rel: Mat,
    sink: Mat,
    /// full layers: the k / v convolutions' input-dependent taps
    k_dyn: Option<Mat>,
    v_dyn: Option<Mat>,
    q_clip: DevBuf,
    rel_clip: DevBuf,
    rel_bias: DevBuf,
    /// [C, 4] each: k, v, the attention's output, the FFN's
    convs: [DevBuf; 4],
    /// the depth attention's queries before the attention (1 or 3; none on layer 0) and before the FFN
    attn_res: Option<(DevBuf, usize)>,
    ffn_res: DevBuf,
    ffn_norm: DevBuf,
    router: Mat,
    exp_bias: Option<DevBuf>,
    gu: Experts,
    down: Experts,
}

/// A looped layer's controllers: the message queries (q, k, v, FFN) and, a controller each, its down / gain / forget
pub struct Loop {
    msg_q: DevBuf,
    down: [Mat; 4],
    gain: [Mat; 4],
    forget: [Mat; 4],
}

/// A slot's state (on its layer's GPU): its key / value rings ([kv heads][cap][128] half, slot = position % cap), its
/// convolutions' last three inputs (the four groups' [3][C] one after another) and the last decode pass's inputs (a
/// rollback's)
pub struct SlotState {
    kc: DevBuf,
    vc: DevBuf,
    cap: usize,
    conv: DevBuf,
    snap: DevBuf,
}

/// A conversation
pub struct Session {
    pub pos: usize,
    pub max_ctx: usize,
    slots: Vec<SlotState>,
    /// rows of the last pass (a rollback keeps some of them; past MAXV it cannot)
    last_rows: usize,
    /// the DSpark drafter's caches and f_rnn (a drafter loaded), on the head's GPU
    ds: Option<dspark::DsState>,
}

/// A conversation's state in host memory: per slot its keys / values still seen, then its convolution state
pub struct Checkpoint {
    pub pos: usize,
    bufs: Vec<Vec<u8>>,
    pub bytes: usize,
}

const CK_MAGIC: &[u8; 6] = b"NSINK1";

impl Checkpoint {
    pub fn write_to(&self, w: &mut dyn std::io::Write) -> std::io::Result<()> {
        let u = |w: &mut dyn std::io::Write, x: usize| w.write_all(&(x as u64).to_le_bytes());
        w.write_all(CK_MAGIC)?;
        u(w, self.pos)?;
        u(w, self.bufs.len())?;
        for b in &self.bufs {
            u(w, b.len())?;
            w.write_all(b)?;
        }
        Ok(())
    }

    pub fn read_from(r: &mut dyn std::io::Read) -> std::io::Result<Checkpoint> {
        let u = |r: &mut dyn std::io::Read| -> std::io::Result<usize> {
            let mut b = [0u8; 8];
            r.read_exact(&mut b)?;
            Ok(u64::from_le_bytes(b) as usize)
        };
        let mut magic = [0u8; 6];
        r.read_exact(&mut magic)?;
        if &magic != CK_MAGIC {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "not an inkling2 checkpoint"));
        }
        let pos = u(r)?;
        let n = u(r)?;
        if n > 1 << 16 {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "an inkling2 checkpoint of too many parts"));
        }
        let mut bufs = Vec::with_capacity(n);
        let mut bytes = 0;
        for _ in 0..n {
            let len = u(r)?;
            let mut v = vec![0u8; len];
            r.read_exact(&mut v)?;
            bytes += len;
            bufs.push(v);
        }
        Ok(Checkpoint { pos, bufs, bytes })
    }
}

/// A pass's result: its rows' logits, or their tokens picked on the GPU
enum Out {
    Logits(Vec<Vec<f32>>),
    Ids(Vec<u32>),
}

/// Generation from a fed prompt: the logits to draw the next token from, or the committed token not fed yet
pub struct Decoder {
    logits: Option<Vec<f32>>,
    next: Option<u32>,
    pub drafted: u64,
    pub accepted: u64,
    /// drafts asked for; the drafter's tokens after `next`
    draft: bool,
    drafts: Vec<u32>,
}

impl Decoder {
    pub fn new(logits: Vec<f32>, draft: bool) -> Decoder {
        Decoder { logits: Some(logits), next: None, drafted: 0, accepted: 0, draft, drafts: Vec::new() }
    }
    pub fn after(next: u32, draft: bool) -> Decoder {
        Decoder { logits: None, next: Some(next), drafted: 0, accepted: 0, draft, drafts: Vec::new() }
    }
    pub fn pending(&mut self) -> Option<u32> {
        self.next.take()
    }
}

/// NS_INK2_EXPERT_UNION=0: decode passes of several rows read an expert shared by rows once a row (A/B; on by default)
fn union_on() -> bool {
    std::env::var("NS_INK2_EXPERT_UNION").map_or(true, |v| v != "0")
}

/// NS_INK2_PATTN=0 / NS_INK2_PGEMM=0: a prompt pass's attention / routed experts on the engine's own kernels, not the
/// silo's (A/B)
fn prompt_silo(var: &str) -> bool {
    std::env::var(var).map_or(true, |v| v != "0")
}

/// NS_INK2_UNION: by rows a pass, (distinct experts summed, layer applications, the most)
type UnionStats = std::collections::BTreeMap<usize, (u64, u64, u64)>;

/// One GPU of the engine: its work memory, its expansion buffer, its silo, the staging of the hops it starts
pub struct Dev {
    pub gpu: Arc<Gpu>,
    arena: Arena,
    /// the largest matrix (or expert) expanded to half (a prompt chunk's products)
    w16: DevBuf,
    /// the silo's decode kernels for this GPU (kernels/llm/inkling2/silo), None: the engine's own throughout
    silo: Option<&'static ffi::Silo>,
    /// pinned host memory a hop from this GPU goes through
    staging: Mutex<HostBuf>,
    /// a pass's rows' drafter features [chunk, 4, C] (a drafter loaded)
    feats: Option<DevBuf>,
}

impl Dev {
    fn raw(&self) -> *mut std::ffi::c_void {
        self.gpu.raw()
    }
}

/// The pass's streams kept between layer applications (`Inkling2::bank`): the normed embedding, the stream's two
/// buffers, the prelude's output, the blocks the loop pushes (layer 12's, layer 18's), the previous pass's bank and
/// output. The depth attention's blocks are a list of these.
const X0: usize = 0;
const PRE: usize = 1;
const S12: usize = 2;
const S18: usize = 3;
const S12P: usize = 4;
const POUT: usize = 5;
const SA: usize = 6;
const SB: usize = 7;
const BANK: usize = 8;

/// The staging a hop copies through, a piece at a time (pinned host memory per GPU)
const STAGING: usize = 16 << 20;

pub struct Inkling2<'g> {
    pub g: Geometry,
    file: &'g Gguf,
    /// the GPUs: one, or two (the looped layers on the last)
    devs: Vec<Dev>,
    /// the GPU of the embedding, the prelude, the coda and the head
    hd: usize,
    k: &'static ffi::Api,
    layers: Vec<Layer>,
    /// the looped layers' controllers (core_start..core_end)
    loops: Vec<Loop>,
    tok_norm: DevBuf,
    output_norm: DevBuf,
    output_res: DevBuf,
    output: Mat,
    /// the token embedding's rows stay in the file (read a row a token)
    embd: Tensor,
    chunk: usize,
    work: Mutex<()>,
    /// NS_INK2_PROFILE=1: each kernel's time and calls
    prof: Option<Mutex<std::collections::BTreeMap<&'static str, (f64, u64)>>>,
    /// NS_INK2_PROFILE=gpu (with NS_PROFILE=gpu: the stream keeps timestamps): device times, no syncs
    gpu_stamps: bool,
    stamps: Mutex<Vec<(&'static str, usize, i64, i64)>>,
    load_s: f64,
    /// decode rows through the silo's fused steps (NS_INK2_FUSED=0: its products only, the steps as for a prompt)
    fused_on: bool,
    load_bytes: u64,
    /// NS_INK2_UNION=1: a pass's rows' chosen experts read back each layer application (a sync each): by rows a pass,
    /// the distinct experts' sum, the applications, the largest
    union: Option<Mutex<UnionStats>>,
    /// the DSpark drafter (LoadOptions.draft and its GGUF beside the model; on the head's GPU), tokens drafted a step
    ds: Option<dspark::DSpark>,
    drafts: usize,
}

/// NS_INK2_CHUNK when set; otherwise chosen at load
fn chunk_setting() -> Option<usize> {
    std::env::var("NS_INK2_CHUNK").ok().and_then(|v| v.parse().ok()).filter(|v| *v > MAXV)
}

/// A prompt chunk's work memory for `chunk` rows on one GPU (`ab` refuses to go past it): the pass's persistent rows
/// (the bank's 8 rows of C, the messages' 4, the embedding's), then the larger of a layer's two stages - the
/// attention's (its mixed and controlled inputs, the norms, the projections, a controller's or a product's transient
/// work) and the FFN's (the mix, the controller, the norm, the experts' grouped rows) - and the head's
fn arena_bytes(g: &Geometry, chunk: usize) -> usize {
    let (c, qd, kd) = (g.n_embd, g.heads * g.head, g.heads_kv * g.head);
    let row = |floats: usize| floats * 4;
    let persistent = row(16 * c) + c * 2;
    let ctrl = row(2 * c + 2 * g.rank + 2 * c) + c * 4;
    let attn = row(9 * c + 3 * qd + 4 * kd + 2 * g.heads * g.rel_len + g.heads + 2 * 4 * g.heads_kv + 2 * c) + ctrl.max(qd * 2);
    let rows = g.used;
    let ffn = row(3 * c) + ctrl.max(row(g.experts + 2 * g.used) + rows * (c * 2 + row(2 * g.n_ff) + g.n_ff * 2 + row(c)) + rows * 16) + row(c);
    let per_row = persistent + attn.max(ffn) + row(c);
    // a quarter more for the arena's 256-byte alignment and what the count above leaves out
    chunk * per_row * 5 / 4 + MAXV * g.vocab_pad * 4 * 3 + (64 << 20)
}

/// The bytes of a session's caches for `max_ctx` and passes of `chunk` rows: the slots `on` takes (by physical layer)
fn session_bytes(g: &Geometry, max_ctx: usize, chunk: usize, on: &dyn Fn(usize) -> bool) -> usize {
    let conv: usize = g.conv_groups().iter().sum::<usize>() * 4;
    (0..g.slots()).filter(|s| on(g.phys(*s))).map(|s| 2 * g.heads_kv * g.cap(s, max_ctx, chunk) * g.head * 2 + conv * (3 + 3 + MAXV)).sum()
}

fn gib(b: u64) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

/// The physical layer a tensor belongs to (`blk.N.`), None for the rest
fn layer_of(name: &str) -> Option<usize> {
    name.strip_prefix("blk.")?.split('.').next()?.parse().ok()
}

impl<'g> Inkling2<'g> {
    /// Loads the model onto the GPUs: the last of `gpus` alone when it holds everything (the server orders them
    /// weakest first), else the looped layers and the neighbours it has room for on the last, the rest on the one before it. `kv`: the context the
    /// sessions reserve (tokens in all; 0: 32K). A load that does not fit says which card is short (no_room: a 409).
    pub fn load(f: &'g Gguf, gpus: &[Arc<Gpu>], kv: (usize, usize), draft: bool, log: &mut dyn FnMut(String)) -> Result<Inkling2<'g>> {
        let t0 = Instant::now();
        if gpus.is_empty() {
            return Err(e("no GPU"));
        }
        let g = Geometry::read(f).map_err(e)?;
        let k = ffi::api()?;
        let embd = f.tensor("token_embd.weight").ok_or_else(|| e("no token_embd.weight"))?.clone();
        let tied = f.tensor("output.weight").is_none();
        // the weights a layer, and the rest's (the head and its norms; the embedding's rows stay in the file unless it is
        // the head)
        let mut lw = vec![0u64; g.layers];
        let mut head_w = 0u64;
        let mut biggest = 0u64;
        for t in &f.tensors {
            if t.name == "token_embd.weight" && !tied {
                continue;
            }
            biggest = biggest.max(t.bytes);
            match layer_of(&t.name) {
                Some(l) if l < g.layers => lw[l] += t.bytes,
                _ => head_w += t.bytes,
            }
        }
        let ctx = if kv.0 > 0 { kv.0 } else { 32768 };
        // what the driver and cuBLAS take beside the plan (its workspace, the module's code), and a raw tensor beside
        // its repacking while it loads
        let margin: u64 = (384 << 20) + biggest;
        let w16_bytes = [g.heads * g.head * g.n_embd, 2 * g.n_ff * g.n_embd].into_iter().max().unwrap_or(0) * 2;
        let tries: Vec<usize> = chunk_setting().map_or_else(|| vec![2048, 1024, 512], |c| vec![c]);
        let silo_any = ffi::silo_on(gpus[gpus.len() - 1].index)?;
        // the drafter (its file beside the model), when asked for, there and the silo loaded
        let dfile = match (draft && dspark::draft_count() != Some(0), silo_any, dspark::draft_path(f)) {
            (true, Some(_), Some(p)) => match Gguf::open(&p) {
                Ok(d) if d.architecture() == "dflash" => Some(d),
                Ok(_) => None,
                Err(x) => {
                    log(format!("inkling2: the drafter {}: {} - no drafts", p.display(), x.0));
                    None
                }
            },
            _ => None,
        };
        // its weights, the fold through the head (vocab x 256 BF16) - on the head's GPU
        let ds_fixed = dfile.as_ref().map_or(0u64, |d| d.tensors.iter().map(|t| t.bytes).sum::<u64>() + (g.vocab_pad * (256 * 2 + 4)) as u64);
        let mut free = Vec::with_capacity(gpus.len());
        for gpu in gpus {
            free.push(gpu.free_or_guess()?);
        }
        // what the layers [a, b) need on a GPU for prompt chunks of c rows (their weights and slots, the work memory;
        // `head`: the head, the embedding and the drafter there too)
        let need = |c: usize, on: &dyn Fn(usize) -> bool, head: bool| -> u64 {
            let weights: u64 = (0..g.layers).filter(|l| on(*l)).map(|l| lw[l]).sum::<u64>() + if head { head_w } else { 0 };
            let extra = kv.1.max(1).saturating_sub(1) as u64 * session_bytes(&g, (g.window + c).min(ctx), c, on) as u64;
            let ds_c = if head {
                dfile.as_ref().map_or(0, |_| (c * dspark::FEATS * g.n_embd * 4) as u64 + kv.1.max(1) as u64 * dspark::DSpark::session_bytes(&g, ctx, c, MAXV) as u64)
                    + ds_fixed
            } else {
                dfile.as_ref().map_or(0, |_| (c * dspark::FEATS * g.n_embd * 4) as u64)
            };
            weights + w16_bytes as u64 + arena_bytes(&g, c) as u64 + session_bytes(&g, ctx, c, on) as u64 + extra + margin + ds_c + STAGING as u64
        };
        // NS_INK2_SPLIT=A,B: the faster GPU's layers [A, B) (they hold the loop); else from the loop out, a layer at a
        // time (the coda's first, then the prelude's last), while it has the room - the faster card does more of a token
        let forced: Option<(usize, usize)> = std::env::var("NS_INK2_SPLIT").ok().and_then(|v| {
            let x: Vec<usize> = v.split(',').filter_map(|p| p.trim().parse().ok()).collect();
            (x.len() == 2 && x[0] >= 1 && x[0] <= g.core.0 && x[1] >= g.core.1 && x[1] <= g.layers).then(|| (x[0], x[1]))
        });
        // (GPU of the rest and the head, GPU of [a, b), a, b, chunk)
        let last = gpus.len() - 1;
        let mut plan: Option<(usize, usize, usize, usize, usize)> = None;
        let mut short: Option<(usize, u64, String)> = None;
        let all = |_: usize| true;
        let total_w: u64 = lw.iter().sum::<u64>() + head_w;
        for &c in &tries {
            let one = need(c, &all, true);
            if one <= free[last] && forced.is_none() {
                plan = Some((last, last, 0, g.layers, c));
                break;
            }
            if gpus.len() > 1 {
                let fits = |a: usize, b: usize| need(c, &|l: usize| l >= a && l < b, false) <= free[last];
                let (mut a, mut b) = forced.unwrap_or(g.core);
                if forced.is_none() && fits(a, b) {
                    loop {
                        if b < g.layers && fits(a, b + 1) {
                            b += 1;
                        } else if a > 1 && fits(a - 1, b) {
                            a -= 1;
                        } else {
                            break;
                        }
                    }
                }
                let (nr, nc) = (need(c, &|l: usize| l < a || l >= b, true), need(c, &|l: usize| l >= a && l < b, false));
                if nr <= free[last - 1] && nc <= free[last] {
                    plan = Some((last - 1, last, a, b, c));
                    break;
                }
                if short.is_none() {
                    short = Some(if nr > free[last - 1] {
                        (last - 1, nr - free[last - 1], format!("layers 0-{} and {b}-{} with the head need {:.1} GiB on {}; {:.1} GiB are free", a - 1, g.layers - 1, gib(nr),
                                                               gpus[last - 1].name, gib(free[last - 1])))
                    } else {
                        (last, nc - free[last], format!("layers {a}-{} need {:.1} GiB on {}; {:.1} GiB are free", b - 1, gib(nc), gpus[last].name, gib(free[last])))
                    });
                }
            } else if short.is_none() {
                short = Some((last, one - free[last], format!("{:.1} GiB of weights, work memory and a {ctx}-token session need {:.1} GiB on {}; {:.1} GiB are free",
                                                              gib(total_w), gib(one), gpus[last].name, gib(free[last]))));
            }
        }
        let Some((rest_i, core_i, sa, sb, chunk)) = plan else {
            let (i, by, msg) = short.unwrap_or((last, 0, "no room".into()));
            return Err(Error(nextcuda_core::no_room_message(gpus[i].index, (by >> 20) + 1, &format!("inkling2: {msg}"))));
        };
        // the engine's GPUs: the rest's (index 0), the one of [sa, sb) (index 1 when it is another)
        let mut picked = vec![rest_i];
        if core_i != rest_i {
            picked.push(core_i);
        }
        let mut devs = Vec::with_capacity(picked.len());
        for &i in &picked {
            let gpu = gpus[i].clone();
            let silo = ffi::silo_on(gpu.index)?;
            log(match silo {
                Some(s) => format!("inkling2: {} - decode kernels from its silo {}", gpu.name, s.path.display()),
                None => format!("inkling2: {} - no silo, its own kernels throughout", gpu.name),
            });
            devs.push(Dev { arena: Arena::new(&gpu, arena_bytes(&g, chunk))?, w16: DevBuf::new(&gpu, w16_bytes)?, silo, staging: Mutex::new(HostBuf::new(&gpu, STAGING)?),
                            feats: None, gpu });
        }
        let hd = 0;
        let cd = devs.len() - 1;
        if devs.len() > 1 {
            let apps = g.order().iter().filter(|(_, l, _)| *l >= sa && *l < sb).count();
            log(format!("inkling2: layers {sa}-{} ({apps} of the {} applications a token) on {}, the rest and the head on {}", sb - 1, g.order().len(), devs[cd].gpu.name,
                        devs[hd].gpu.name));
        }
        let dev_of = |l: usize| if l >= sa && l < sb { cd } else { hd };
        let loaded = std::cell::Cell::new(0u64);
        let mut layers = Vec::with_capacity(g.layers);
        for l in 0..g.layers {
            let dv = dev_of(l);
            let (gpu, silo) = (&devs[dv].gpu, devs[dv].silo);
            let read = |name: &str| -> Result<Mat> { mat(f, gpu, silo, name, &loaded) };
            let n = |s: &str| format!("blk.{l}.{s}.weight");
            let swa = g.swa[l];
            let norm = if swa {
                let a = read(&n("attn_norm"))?.buf;
                [a.view(0, a.len)?, a.view(0, a.len)?, a]
            } else {
                [read(&n("attn_q_inp_norm"))?.buf, read(&n("attn_k_inp_norm"))?.buf, read(&n("attn_v_inp_norm"))?.buf]
            };
            let attn_res = match f.tensor(&n("attn_res_query")) {
                Some(t) => {
                    let nq = if t.shape.len() > 1 { t.shape[0] as usize } else { 1 };
                    Some((read(&n("attn_res_query"))?.buf, nq))
                }
                None if l == 0 => None,
                None => return Err(e(format!("no {}", n("attn_res_query")))),
            };
            let exp = |which: &str| -> Result<Experts> {
                let name = format!("blk.{l}.ffn_{which}_exps.weight");
                let m = mat(f, gpu, silo, &name, &loaded)?;
                let t = f.tensor(&name).ok_or_else(|| e(format!("no {name}")))?;
                if t.shape.len() != 3 || t.shape[0] as usize != g.experts {
                    return Err(e(format!("{name}: {:?}, not [{} experts, rows, cols]", t.shape, g.experts)));
                }
                let (rows, cols) = (t.shape[1] as usize, t.shape[2] as usize);
                let stride = m.buf.len / g.experts;
                let (scale, scale_h) = match f.tensor(&format!("blk.{l}.ffn_{which}_exps.scale")) {
                    Some(st) => {
                        let raw = f.read(st).map_err(|x| e(x.0))?;
                        let h: Vec<f32> = raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                        (Some(DevBuf::from_f32(gpu, &h)?), h)
                    }
                    None => (None, vec![1.0; g.experts]),
                };
                Ok(Experts { buf: m.buf, ty: m.ty, rows, cols, stride, packed: m.packed, scale, scale_h })
            };
            let gu = exp("gate_up")?;
            let down = exp("down")?;
            let ly = Layer {
                dev: dv,
                swa,
                norm,
                q: read(&n("attn_q"))?,
                k: read(&n("attn_k"))?,
                v: read(&n("attn_v"))?,
                o: read(&n("attn_output"))?,
                gate: read(&n("attn_gate"))?,
                rel: read(&n("attn_rel"))?,
                sink: read(&n("attn_sink_proj"))?,
                k_dyn: if swa { None } else { Some(read(&n("attn_k_conv_proj"))?) },
                v_dyn: if swa { None } else { Some(read(&n("attn_v_conv_proj"))?) },
                q_clip: read(&n("attn_q_clip"))?.buf,
                rel_clip: read(&n("attn_rel_clip"))?.buf,
                rel_bias: read(&n("attn_rel_bias"))?.buf,
                convs: [read(&n("attn_k_conv"))?.buf, read(&n("attn_v_conv"))?.buf, read(&n("attn_post_conv"))?.buf, read(&n("ffn_post_conv"))?.buf],
                attn_res,
                ffn_res: read(&n("ffn_res_query"))?.buf,
                ffn_norm: read(&n("ffn_norm"))?.buf,
                router: read(&n("ffn_gate_inp"))?,
                exp_bias: match f.tensor(&format!("blk.{l}.exp_probs_b.bias")) {
                    Some(_) => Some(read(&format!("blk.{l}.exp_probs_b.bias"))?.buf),
                    None => None,
                },
                gu,
                down,
            };
            if ly.gu.rows != 2 * g.n_ff || ly.gu.cols != g.n_embd || ly.down.rows != g.n_embd || ly.down.cols != g.n_ff {
                return Err(e(format!("layer {l}'s experts: gate|up {}x{}, down {}x{}", ly.gu.rows, ly.gu.cols, ly.down.rows, ly.down.cols)));
            }
            layers.push(ly);
            if l % 6 == 5 {
                log(format!("inkling2: layer {} of {} ({:.1} GiB, {:.0} s)", l + 1, g.layers, gib(loaded.get()), t0.elapsed().as_secs_f64()));
            }
        }
        let mut loops = Vec::new();
        for l in g.core.0..g.core.1 {
            let (gpu, silo) = (&devs[dev_of(l)].gpu, devs[dev_of(l)].silo);
            let read = |name: &str| -> Result<Mat> { mat(f, gpu, silo, name, &loaded) };
            // the controllers in float32 (UD-Q8_K_XL stores them in BF16; the fused decode steps read float32): exact
            let read32 = |name: &str| -> Result<Mat> {
                let m = read(name)?;
                match m.ty {
                    GType::F32 => return Ok(m),
                    GType::BF16 => {}
                    t => return Err(e(format!("{name}: {} (float32 or BF16 expected)", t.name()))),
                }
                let mut raw = vec![0u8; m.rows * m.cols * 2];
                m.buf.read(0, &mut raw)?;
                let wide: Vec<u8> = raw.chunks_exact(2).flat_map(|b| [0, 0, b[0], b[1]]).collect();
                let buf = DevBuf::new(gpu, wide.len())?;
                buf.write(0, &wide)?;
                Ok(Mat { buf, ty: GType::F32, ..m })
            };
            let n = |s: &str| format!("blk.{l}.loop_{s}.weight");
            let ctrl = ["q", "k", "v", "ffn"];
            let part = |what: &str| -> Result<[Mat; 4]> {
                let v: Vec<Mat> = ctrl.iter().map(|c| read32(&n(&format!("{c}_ctrl_{what}")))).collect::<Result<_>>()?;
                <[Mat; 4]>::try_from(v).map_err(|_| e("four controllers"))
            };
            let down = part("down")?;
            let gain = part("gain")?;
            let forget = part("forget")?;
            loops.push(Loop { msg_q: read(&n("msg_query"))?.buf, down, gain, forget });
        }
        let (hgpu, hsilo) = (&devs[hd].gpu, devs[hd].silo);
        let read = |name: &str| -> Result<Mat> { mat(f, hgpu, hsilo, name, &loaded) };
        let tok_norm = read("token_embd_norm.weight")?.buf;
        let output_norm = read("output_norm.weight")?.buf;
        let output_res = read("output_res_query.weight")?.buf;
        let output = read(if tied { "token_embd.weight" } else { "output.weight" })?;
        if output.rows != g.vocab_pad || g.vocab > g.vocab_pad {
            return Err(e(format!("the output matrix has {} rows for a vocabulary of {} (padded {})", output.rows, g.vocab, g.vocab_pad)));
        }
        for d in &devs {
            d.gpu.sync()?;
        }
        let ds = match (&dfile, devs[hd].silo) {
            // a drafter that does not load: the model without drafts
            (Some(dfl), Some(sl)) => match dspark::load(dfl, &g, &devs[hd].gpu, k, sl, &output, &devs[hd].arena) {
                Ok(x) => {
                    loaded.set(loaded.get() + x.bytes);
                    Some(x)
                }
                Err(x) => {
                    log(format!("inkling2: the drafter does not load ({}) - no drafts", x.0));
                    None
                }
            },
            _ => None,
        };
        let drafts = ds.as_ref().map_or(0, |d| dspark::draft_count().unwrap_or(5).min(d.block - 1));
        if let Some(d) = &ds {
            log(format!("inkling2: the DSpark drafter ({} MB, block of {}), {drafts} drafts a step", d.bytes >> 20, d.block));
            for dv in devs.iter_mut() {
                dv.feats = Some(DevBuf::new(&dv.gpu, chunk * dspark::FEATS * g.n_embd * 4)?);
            }
        }
        let load_s = t0.elapsed().as_secs_f64();
        let on: Vec<&str> = devs.iter().map(|d| d.gpu.name.as_str()).collect();
        log(format!("inkling2: {} layers ({} applications a token: [{}, {}) x {}), {:.1} GiB on {} in {load_s:.0} s; prompt chunks of {chunk}", g.layers,
                    g.order().len(), g.core.0, g.core.1, g.passes, gib(loaded.get()), on.join(" + ")));
        Ok(Inkling2 { g, file: f, devs, hd, k, layers, loops, tok_norm, output_norm, output_res, output, embd, chunk, work: Mutex::new(()),
                      prof: std::env::var("NS_INK2_PROFILE").is_ok_and(|v| v == "1" || v == "gpu").then(|| Mutex::new(Default::default())), load_s, load_bytes: loaded.get(),
                      gpu_stamps: std::env::var("NS_INK2_PROFILE").is_ok_and(|v| v == "gpu"), stamps: Mutex::new(Vec::new()),
                      fused_on: std::env::var("NS_INK2_FUSED").map_or(true, |v| v != "0"),
                      union: std::env::var("NS_INK2_UNION").is_ok_and(|v| v == "1").then(|| Mutex::new(Default::default())), ds, drafts })
    }

    /// The GPU of physical layer `l`
    fn dev_of(&self, l: usize) -> &Dev {
        &self.devs[self.layers[l].dev]
    }

    /// The head's GPU
    fn hdev(&self) -> &Dev {
        &self.devs[self.hd]
    }

    /// A kernel call on `d`, its return code checked (NS_INK2_PROFILE=1: timed, the stream drained before and after)
    fn op(&self, d: &Dev, name: &'static str, f: impl FnOnce() -> c_int) -> Result<()> {
        let Some(p) = &self.prof else { return ffi::check(f(), name) };
        if self.gpu_stamps {
            // device timestamps around the launch, read at the pass's end (no syncs: the GPU's own time)
            let a = nextcuda_core::api()?;
            let (mut t0, mut t1) = (0i64, 0i64);
            // SAFETY: out pointers to locals.
            unsafe { (a.stamp)(d.raw(), &mut t0) };
            ffi::check(f(), name)?;
            // SAFETY: as above.
            unsafe { (a.stamp)(d.raw(), &mut t1) };
            let di = self.devs.iter().position(|x| std::ptr::eq(x, d)).unwrap_or(0);
            self.stamps.lock().unwrap().push((name, di, t0, t1));
            return Ok(());
        }
        d.gpu.sync()?;
        let t0 = Instant::now();
        ffi::check(f(), name)?;
        d.gpu.sync()?;
        let mut m = p.lock().unwrap();
        let ent = m.entry(name).or_insert((0.0, 0));
        ent.0 += t0.elapsed().as_secs_f64();
        ent.1 += 1;
        Ok(())
    }

    /// The pass's device timestamps into the profile (NS_INK2_PROFILE=gpu)
    fn stamps_in(&self) -> Result<()> {
        let Some(p) = &self.prof else { return Ok(()) };
        let v: Vec<(&'static str, usize, i64, i64)> = std::mem::take(&mut *self.stamps.lock().unwrap());
        if v.is_empty() {
            return Ok(());
        }
        let a = nextcuda_core::api()?;
        let mut m = p.lock().unwrap();
        for (name, di, t0, t1) in v {
            let mut ns = 0f64;
            // SAFETY: two tickets of this GPU's stream; waits for them.
            unsafe { (a.elapsed)(self.devs[di].raw(), t0, t1, &mut ns) };
            let ent = m.entry(name).or_insert((0.0, 0));
            ent.0 += ns * 1e-9;
            ent.1 += 1;
        }
        Ok(())
    }

    /// The profile's lines (`report`): each kernel's time in all and a token
    pub fn profile_lines(&self, tokens: usize) -> Vec<String> {
        let mut out = Vec::new();
        if let Some(u) = &self.union {
            for (rows, (sum, n, max)) in u.lock().unwrap().iter() {
                out.push(format!("[experts of a {rows}-row pass: {:.2} distinct a layer application on average, {max} at most ({n} applications; {} a row)]",
                                 *sum as f64 / (*n).max(1) as f64, self.g.used));
            }
        }
        let Some(p) = &self.prof else { return out };
        let m = p.lock().unwrap();
        let mut v: Vec<_> = m.iter().collect();
        v.sort_by(|a, b| b.1 .0.total_cmp(&a.1 .0));
        out.extend(v.iter().map(|(n, (s, c))| format!("[profile {n:<16} {s:>8.3} s {c:>8} calls {:>8.3} ms/token]", s * 1000.0 / (tokens + 1) as f64)));
        out
    }

    pub fn load_seconds(&self) -> f64 {
        self.load_s
    }
    pub fn load_bytes(&self) -> u64 {
        self.load_bytes
    }
    pub fn prefill_chunk(&self) -> usize {
        self.chunk
    }
    pub fn has_draft(&self) -> bool {
        self.drafts > 0
    }
    pub fn max_verify(&self) -> usize {
        self.drafts + 1
    }

    fn f32(&self, d: &Dev, n: usize) -> Result<DevBuf> {
        self.ab(d, n * 4)
    }

    /// n bytes of d's arena - never an allocation of its own past its end: such a buffer would be freed when it goes
    /// out of scope while kernels queued on it have not run (a fault on the GPU)
    fn ab(&self, d: &Dev, n: usize) -> Result<DevBuf> {
        let at = d.arena.mark().next_multiple_of(256);
        if at + n > d.arena.len() {
            return Err(e(format!("inkling2: the work memory ({:.0} MiB) is short of {:.1} MiB more for this pass (NS_INK2_CHUNK smaller)",
                                 d.arena.len() as f64 / (1 << 20) as f64, (at + n - d.arena.len()) as f64 / (1 << 20) as f64)));
        }
        d.arena.bytes(n)
    }

    /// y [t, w.rows] = x [t, w.cols] . W^T: decode rows through the stored blocks, prompt rows through half
    fn matmul(&self, d: &Dev, w: &Mat, x: &DevBuf, t: usize, y: &DevBuf) -> Result<()> {
        let k = self.k;
        let (n, kk) = (w.rows as i64, w.cols as i64);
        let ty = w.ty.code() as c_int;
        if t <= MAXV {
            if let (true, Some(s)) = (w.packed, d.silo) {
                // a repacked matrix: the silo's product on x quantized (its work rewound after)
                let mark = d.arena.mark();
                let q = self.xq(d, x, w.cols, t)?.ok_or_else(|| e("decode rows without the silo's activations"))?;
                // SAFETY: the repacked matrix, x's quants, y [t, rows].
                self.op(d, "mv silo", || unsafe { (s.mv)(d.raw(), ty, w.buf.ptr(), q.ptr(), y.fp(), kk, n, t as i64) })?;
                d.arena.rewind(mark);
                return Ok(());
            }
            // SAFETY: arithmetic only.
            if let Some(s) = d.silo.filter(|s| unsafe { (s.fmv_supported)(ty, kk) } != 0) {
                // SAFETY: the float32 / BF16 matrix, x [t, cols], y [t, rows].
                return self.op(d, "fmv silo", || unsafe { (s.fmv)(d.raw(), ty, w.buf.ptr(), x.fp(), y.fp(), kk, n, t as i64) });
            }
            // SAFETY: the matrix's blocks, x [t, cols], y [t, rows].
            return self.op(d, "mv", || unsafe { (k.mv)(d.raw(), ty, w.buf.ptr(), x.fp(), y.fp(), kk, n, t as i64) });
        }
        if w.ty == GType::F32 {
            // SAFETY: x [t, cols], the matrix [rows, cols] float32, y [t, rows].
            return self.op(d, "gemm", || unsafe { (k.gemm)(d.raw(), t as i64, n, kk, x.fp(), kk, w.buf.fp(), y.fp(), n, 0) });
        }
        let elems = w.rows * w.cols;
        if elems * 2 > d.w16.len {
            return Err(e(format!("a {}x{} matrix past the half buffer", w.rows, w.cols)));
        }
        // SAFETY: the blocks (or the silo's repacking) -> rows x cols halfs in w16; x [t, cols] -> half; y [t, rows] = xh . w16^T.
        match (w.packed, d.silo) {
            (true, Some(s)) => self.op(d, "expand silo", || unsafe { (s.unpack_f16)(d.raw(), ty, w.buf.ptr(), n, 0, n, kk, d.w16.ptr().cast()) })?,
            _ => self.op(d, "expand", || unsafe { (k.dequant_f16)(d.raw(), ty, w.buf.ptr(), elems as i64, d.w16.ptr().cast()) })?,
        }
        // x in half only for this product (the arena's later work reuses it: the stream is in order)
        let mark = d.arena.mark();
        let xh = self.ab(d, t * w.cols * 2)?;
        self.op(d, "to half", || unsafe { (k.to_f16)(d.raw(), x.fp(), xh.ptr().cast(), (t * w.cols) as i64) })?;
        self.op(d, "gemm f16", || unsafe { (k.gemm_f16)(d.raw(), t as i64, n, kk, xh.ptr().cast(), kk, d.w16.ptr().cast(), y.fp(), n, 0) })?;
        d.arena.rewind(mark);
        Ok(())
    }

    /// x [t, w.cols] . W^T into a new buffer
    fn mm(&self, d: &Dev, w: &Mat, x: &DevBuf, t: usize) -> Result<DevBuf> {
        let y = self.f32(d, t * w.rows)?;
        self.matmul(d, w, x, t, &y)?;
        Ok(y)
    }

    fn rms(&self, d: &Dev, x: &DevBuf, w: &DevBuf, t: usize) -> Result<DevBuf> {
        let c = self.g.n_embd;
        let y = self.f32(d, t * c)?;
        // SAFETY: x and y [t, c], w c floats.
        self.op(d, "rms norm", || unsafe { (self.k.rms_norm)(d.raw(), x.fp(), w.fp(), y.fp(), t as i64, c as i64, self.g.eps) })?;
        Ok(y)
    }

    /// The silo when decode rows use it
    fn dec(&self, d: &Dev, t: usize) -> Option<&'static ffi::Silo> {
        d.silo.filter(|_| t <= MAXV)
    }

    /// x [t, n] quantized for the silo's decode products (None: prompt rows, or no silo)
    fn xq(&self, d: &Dev, x: &DevBuf, n: usize, t: usize) -> Result<Option<DevBuf>> {
        let Some(s) = self.dec(d, t) else { return Ok(None) };
        // SAFETY: arithmetic only; then x [t, n] -> its quants and scales.
        let q = self.ab(d, unsafe { (s.xq_bytes)(t as i64, n as i64) })?;
        self.op(d, "xq", || unsafe { (s.xq)(d.raw(), x.fp(), q.ptr(), t as i64, n as i64) })?;
        Ok(Some(q))
    }

    /// The RMS norm (weight w) of x [t, C], and for decode rows with the silo its quantized form too (one launch)
    fn rms_q(&self, d: &Dev, x: &DevBuf, w: &DevBuf, t: usize) -> Result<(DevBuf, Option<DevBuf>)> {
        let Some(s) = self.dec(d, t) else { return Ok((self.rms(d, x, w, t)?, None)) };
        let c = self.g.n_embd;
        let y = self.f32(d, t * c)?;
        // SAFETY: arithmetic only; then x [t, c] normed into y and its quants.
        let q = self.ab(d, unsafe { (s.xq_bytes)(t as i64, c as i64) })?;
        self.op(d, "rms xq", || unsafe { (s.rms_xq)(d.raw(), x.fp(), w.fp(), y.fp(), q.ptr(), t as i64, c as i64, self.g.eps) })?;
        Ok((y, Some(q)))
    }

    /// Several products of decode rows in one launch when the silo takes them all (each its matrix, input and quantized
    /// input), else each alone
    fn mm_many(&self, d: &Dev, jobs: &[(&Mat, &DevBuf, Option<&DevBuf>)], t: usize) -> Result<Vec<DevBuf>> {
        if let Some(s) = self.dec(d, t) {
            // SAFETY: arithmetic only.
            let all = jobs.len() <= 8 && jobs.iter().all(|(w, _, q)| q.is_some() && w.packed && unsafe { (s.supported)(w.ty.code() as c_int, w.cols as i64) } != 0);
            if all {
                let ys: Vec<DevBuf> = jobs.iter().map(|(w, _, _)| self.f32(d, t * w.rows)).collect::<Result<_>>()?;
                let ty: Vec<c_int> = jobs.iter().map(|(w, _, _)| w.ty.code() as c_int).collect();
                let wp: Vec<*const std::ffi::c_void> = jobs.iter().map(|(w, _, _)| w.buf.ptr() as *const _).collect();
                let qp: Vec<*const std::ffi::c_void> = jobs.iter().map(|(_, _, q)| q.map_or(std::ptr::null(), |q| q.ptr() as *const _)).collect();
                let yp: Vec<*mut f32> = ys.iter().map(|y| y.fp()).collect();
                let rows: Vec<i64> = jobs.iter().map(|(w, _, _)| w.rows as i64).collect();
                let cols: Vec<i64> = jobs.iter().map(|(w, _, _)| w.cols as i64).collect();
                // SAFETY: each job's matrix, quantized input [t, cols] and output [t, rows], as the arrays say.
                self.op(d, "mv many silo", || unsafe {
                    (s.mv_many)(d.raw(), jobs.len() as c_int, ty.as_ptr(), wp.as_ptr(), qp.as_ptr(), yp.as_ptr(), rows.as_ptr(), cols.as_ptr(), t as i64)
                })?;
                return Ok(ys);
            }
        }
        jobs.iter().map(|(w, x, q)| self.mmq(d, w, x, *q, t)).collect()
    }

    /// x [t, w.cols] . W^T: through the silo's products when x is quantized for them (xq) and they take the matrix
    fn mmq(&self, d: &Dev, w: &Mat, x: &DevBuf, xq: Option<&DevBuf>, t: usize) -> Result<DevBuf> {
        if let (Some(s), Some(q), true) = (self.dec(d, t), xq, w.packed) {
            let y = self.f32(d, t * w.rows)?;
            // SAFETY: the matrix's repacked blocks, x's quants, y [t, rows].
            self.op(d, "mv silo", || unsafe { (s.mv)(d.raw(), w.ty.code() as c_int, w.buf.ptr(), q.ptr(), y.fp(), w.cols as i64, w.rows as i64, t as i64) })?;
            return Ok(y);
        }
        self.mm(d, w, x, t)
    }

    /// The depth attention: `nq` mixes [nq, t, C] of the states (at most 8) by `query`
    fn mix(&self, d: &Dev, states: &[&DevBuf], query: &DevBuf, nq: usize, t: usize) -> Result<DevBuf> {
        let c = self.g.n_embd;
        let out = self.f32(d, nq * t * c)?;
        let ptrs: Vec<*const f32> = states.iter().map(|s| s.fp() as *const f32).collect();
        // SAFETY: each state [t, c] on this GPU, query [nq, c], out [nq, t, c].
        self.op(d, "mix", || unsafe { (self.k.mix)(d.raw(), ptrs.as_ptr(), ptrs.len() as i64, query.fp(), nq as i64, out.fp(), t as i64, c as i64, 1e-6) })?;
        Ok(out)
    }

    /// A loop controller on x [t, C] with its message
    fn control(&self, d: &Dev, lp: &Loop, which: usize, x: &DevBuf, msg: &DevBuf, t: usize) -> Result<DevBuf> {
        let c = self.g.n_embd;
        let out = self.f32(d, t * c)?;
        // its work rewound after (the stream is in order)
        let mark = d.arena.mark();
        let z = self.f32(d, t * 2 * c)?;
        // SAFETY: x, msg [t, c]; z [t, 2c].
        self.op(d, "ctrl z", || unsafe { (self.k.ctrl_z)(d.raw(), x.fp(), msg.fp(), z.fp(), t as i64, c as i64, 1e-6) })?;
        let u = self.mm(d, &lp.down[which], &z, t)?;
        let fo = self.mm(d, &lp.forget[which], &u, t)?;
        let gn = self.mm(d, &lp.gain[which], &u, t)?;
        // SAFETY: all [t, c].
        self.op(d, "ctrl apply", || unsafe { (self.k.ctrl_apply)(d.raw(), x.fp(), msg.fp(), fo.fp(), gn.fp(), out.fp(), t as i64, c as i64) })?;
        d.arena.rewind(mark);
        Ok(out)
    }

    /// A short convolution of x [t, ch] with slot `ss`'s state group `grp` (dyn: [t, heads, 4] taps)
    #[allow(clippy::too_many_arguments)]
    fn conv(&self, d: &Dev, ss: &SlotState, grp: usize, w: &DevBuf, x: &DevBuf, dynt: Option<&DevBuf>, t: usize) -> Result<DevBuf> {
        let ch = self.g.conv_groups()[grp];
        let (state, snap) = self.conv_bufs(ss, grp, t)?;
        let out = self.f32(d, t * ch)?;
        let taps = if dynt.is_some() { 2 * self.g.conv + 1 } else { self.g.conv + 1 };
        let inv = 1.0 / (taps as f32).sqrt();
        let heads = self.g.heads_kv as i64;
        // SAFETY: x / out [t, ch], the state [3, ch], w [ch, 4], dyn [t, heads, 4], snap [3 + t, ch].
        self.op(d, "conv", || unsafe {
            (self.k.conv)(d.raw(), x.fp(), state.fp(), w.fp(), dynt.map_or(std::ptr::null(), |x| x.fp()), out.fp(),
                          snap.as_ref().map_or(std::ptr::null_mut(), |s| s.fp()), t as i64, ch as i64, heads, inv)
        })?;
        Ok(out)
    }

    /// The tokens' embeddings [t, n_embd] on `d` (their rows read from the file, expanded on the GPU)
    fn embed(&self, d: &Dev, tokens: &[u32]) -> Result<DevBuf> {
        let g = &self.g;
        let row = self.embd.ty.bytes(g.n_embd as u64).ok_or_else(|| e("the embedding's row size"))? as usize;
        let mut raw = vec![0u8; tokens.len() * row];
        for (i, t) in tokens.iter().enumerate() {
            if *t as usize >= g.vocab {
                return Err(e(format!("token {t} past the vocabulary ({})", g.vocab)));
            }
            self.file.read_into(&self.embd, *t as u64 * row as u64, &mut raw[i * row..(i + 1) * row]).map_err(|x| e(x.0))?;
        }
        let src = self.ab(d, raw.len())?;
        src.write_async(0, &raw)?;
        let x = self.f32(d, tokens.len() * g.n_embd)?;
        // SAFETY: the rows' blocks -> float32.
        self.op(d, "dequant", || unsafe { (self.k.dequant)(d.raw(), self.embd.ty.code() as c_int, src.ptr(), (tokens.len() * g.n_embd) as i64, x.fp()) })?;
        Ok(x)
    }

    /// Expert x of a stack expanded to half into d's w16 (from its GGUF blocks, or the silo's repacked stack of `experts`)
    fn expand_expert(&self, d: &Dev, st: &Experts, x: usize, experts: usize) -> Result<()> {
        let ty = st.ty.code() as c_int;
        let (rows, cols) = (st.rows as i64, st.cols as i64);
        match (st.packed, d.silo) {
            // SAFETY: rows [x rows, (x + 1) rows) of the repacked stack -> rows x cols halfs in w16.
            (true, Some(s)) => self.op(d, "expand silo", || unsafe {
                (s.unpack_f16)(d.raw(), ty, st.buf.ptr(), experts as i64 * rows, x as i64 * rows, rows, cols, d.w16.ptr().cast())
            }),
            _ => {
                let w = st.buf.view(x * st.stride, st.stride)?;
                // SAFETY: the expert's blocks -> rows x cols halfs in w16.
                self.op(d, "expand", || unsafe { (self.k.dequant_f16)(d.raw(), ty, w.ptr(), rows * cols, d.w16.ptr().cast()) })
            }
        }
    }

    /// The routed experts on h [t, C] (normed): their weighted sum [t, C]
    fn moe(&self, d: &Dev, ly: &Layer, h: &DevBuf, hq: Option<&DevBuf>, t: usize) -> Result<DevBuf> {
        let (g, k) = (&self.g, self.k);
        let (c, kk, ff) = (g.n_embd, g.used, g.n_ff);
        let ids = self.ab(d, t * kk * 4)?;
        let wts = self.f32(d, t * kk)?;
        let out = self.f32(d, t * c)?;
        let (gu, dn) = (&ly.gu, &ly.down);
        let bias = ly.exp_bias.as_ref().map_or(std::ptr::null(), |b| b.fp() as *const f32);
        let logits = self.f32(d, t * g.experts)?;
        match self.fused(d, t).filter(|_| ly.router.ty == GType::F32 && g.experts.is_multiple_of(8) && g.experts <= 128) {
            // decode rows: the router's product and choice in one launch
            // SAFETY: h [t, c], the router [E, c] float32, the scratch logits; ids, weights [t, K].
            Some(sl) => self.op(d, "router silo", || unsafe {
                (sl.fused.router)(d.raw(), h.fp(), ly.router.buf.fp(), bias, logits.fp(), ids.ptr().cast(), wts.fp(), t as i64, g.experts as i64, c as i64, kk as i64,
                                  g.w_scale)
            })?,
            None => {
                self.matmul(d, &ly.router, h, t, &logits)?;
                // SAFETY: logits [t, E] -> ids, weights [t, K].
                self.op(d, "route", || unsafe { (k.route)(d.raw(), logits.fp(), bias, ids.ptr().cast(), wts.fp(), t as i64, g.experts as i64, kk as i64, g.w_scale) })?
            }
        }
        if let (Some(u), true) = (&self.union, t <= MAXV) {
            let mut b = vec![0u8; t * kk * 4];
            ids.read(0, &mut b)?;
            let mut set: Vec<u32> = b.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            set.sort_unstable();
            set.dedup();
            let mut m = u.lock().unwrap();
            let ent = m.entry(t).or_insert((0, 0, 0));
            ent.0 += set.len() as u64;
            ent.1 += 1;
            ent.2 = ent.2.max(set.len() as u64);
        }
        // the experts' work rewound after (the stream is in order)
        let mark = d.arena.mark();
        if let (Some(s), Some(q)) = (self.dec(d, t), hq) {
            if gu.packed && dn.packed {
                // SAFETY: arithmetic only; then gate | up and SiTU into the act's quants, the down products weighted into out.
                let aq = self.ab(d, unsafe { (s.xq_bytes)((t * kk) as i64, ff as i64) })?;
                let sc = |x: &Experts| x.scale.as_ref().map_or(std::ptr::null(), |s| s.fp() as *const f32);
                if t > 1 && union_on() {
                    // rows that share experts: each distinct expert's weights read once for all its pairs
                    let p = t * kk;
                    let uni = self.ab(d, (1 + 2 * p + 16 * p) * 4)?;
                    let part = self.f32(d, p * dn.rows)?;
                    // SAFETY: ids [p] -> uni; then the products as silo.h names them.
                    unsafe {
                        self.op(d, "experts union", || (s.moe_union)(d.raw(), ids.ptr().cast(), p as i64, uni.ptr().cast()))?;
                        self.op(d, "experts gu silo", || (s.moe_gu_union)(d.raw(), gu.ty.code() as c_int, gu.buf.ptr(), g.experts as i64, sc(gu), uni.ptr().cast(), q.ptr(),
                                                                          aq.ptr(), t as i64, kk as i64, gu.cols as i64, ff as i64, g.situ_beta, g.situ_lb))?;
                        self.op(d, "experts down silo", || (s.moe_down_union)(d.raw(), dn.ty.code() as c_int, dn.buf.ptr(), g.experts as i64, sc(dn), uni.ptr().cast(),
                                                                              wts.fp(), aq.ptr(), part.fp(), out.fp(), t as i64, kk as i64, ff as i64, dn.rows as i64))?;
                    }
                    d.arena.rewind(mark);
                    return Ok(out);
                }
                self.op(d, "experts gu silo", || unsafe {
                    (s.moe_gu)(d.raw(), gu.ty.code() as c_int, gu.buf.ptr(), g.experts as i64, sc(gu), ids.ptr().cast(), q.ptr(), aq.ptr(), t as i64, kk as i64,
                               gu.cols as i64, ff as i64, g.situ_beta, g.situ_lb)
                })?;
                self.op(d, "experts down silo", || unsafe {
                    (s.moe_down)(d.raw(), dn.ty.code() as c_int, dn.buf.ptr(), g.experts as i64, sc(dn), ids.ptr().cast(), wts.fp(), aq.ptr(), out.fp(), t as i64,
                                 kk as i64, ff as i64, dn.rows as i64)
                })?;
                d.arena.rewind(mark);
                return Ok(out);
            }
        }
        if t <= MAXV {
            if gu.packed || dn.packed {
                return Err(e("the silo's repacked experts without its quantized activations"));
            }
            let y = self.f32(d, t * kk * gu.rows)?;
            let act = self.f32(d, t * kk * ff)?;
            let dd = self.f32(d, t * kk * c)?;
            // SAFETY: each (row, expert) pair's product from its expert's blocks; SiTU; the down products; the sum.
            unsafe {
                self.op(d, "experts gu", || (k.moe_mv)(d.raw(), gu.ty.code() as c_int, gu.buf.ptr(), gu.stride, gu.scale.as_ref().map_or(std::ptr::null(), |s| s.fp()),
                                                       ids.ptr().cast(), h.fp(), 0, y.fp(), t as i64, kk as i64, gu.cols as i64, gu.rows as i64))?;
                self.op(d, "situ", || (k.situ)(d.raw(), y.fp(), std::ptr::null(), act.fp(), std::ptr::null_mut(), (t * kk) as i64, ff as i64, g.situ_beta, g.situ_lb))?;
                self.op(d, "experts down", || (k.moe_mv)(d.raw(), dn.ty.code() as c_int, dn.buf.ptr(), dn.stride, dn.scale.as_ref().map_or(std::ptr::null(), |s| s.fp()),
                                                         ids.ptr().cast(), act.fp(), 1, dd.fp(), t as i64, kk as i64, dn.cols as i64, dn.rows as i64))?;
                self.op(d, "combine", || (k.combine)(d.raw(), dd.fp(), wts.fp(), out.fp(), t as i64, kk as i64, c as i64))?;
            }
            d.arena.rewind(mark);
            return Ok(out);
        }
        // a prompt chunk: the rows grouped by expert on the host, each expert's rows one product
        let mut hid = vec![0u8; t * kk * 4];
        ids.read(0, &mut hid)?;
        let mut hw = vec![0u8; t * kk * 4];
        wts.read(0, &mut hw)?;
        let id: Vec<usize> = hid.chunks_exact(4).map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize).collect();
        let wv: Vec<f32> = hw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        let mut by: Vec<Vec<usize>> = vec![Vec::new(); g.experts];
        for (p, &x) in id.iter().enumerate() {
            if x >= g.experts {
                return Err(e(format!("the router chose expert {x}")));
            }
            by[x].push(p);
        }
        let m = t * kk;
        let mut tok = Vec::with_capacity(m);
        let mut rs = Vec::with_capacity(m);
        let mut at = vec![0usize; m];
        let mut runs = Vec::new();
        for (x, ps) in by.iter().enumerate() {
            if ps.is_empty() {
                continue;
            }
            runs.push((x, tok.len(), ps.len()));
            for &p in ps {
                at[p] = tok.len();
                tok.push((p / kk) as i32);
                rs.push(gu.scale_h[x]);
            }
        }
        // each token's entries: its K rows, weighted (the down product's scale folded in)
        let mut t_ptr = Vec::with_capacity(t + 1);
        let mut ent = Vec::with_capacity(m);
        let mut cw = Vec::with_capacity(m);
        for r in 0..t {
            t_ptr.push(ent.len() as i32);
            for j in 0..kk {
                let p = r * kk + j;
                ent.push(at[p] as i32);
                cw.push(wv[p] * dn.scale_h[id[p]]);
            }
        }
        t_ptr.push(ent.len() as i32);
        let up_i = |v: &[i32]| -> Result<DevBuf> {
            let b = self.ab(d, v.len() * 4)?;
            b.write_async(0, &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
            Ok(b)
        };
        let up_f = |v: &[f32]| -> Result<DevBuf> {
            let b = self.ab(d, v.len() * 4)?;
            b.write_async(0, &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
            Ok(b)
        };
        let (tok_d, rs_d, tp_d, ent_d, cw_d) = (up_i(&tok)?, up_f(&rs)?, up_i(&t_ptr)?, up_i(&ent)?, up_f(&cw)?);
        let xh = self.ab(d, m * c * 2)?;
        let y = self.f32(d, m * gu.rows)?;
        let act = self.ab(d, m * ff * 2)?;
        let dd = self.f32(d, m * c)?;
        // SAFETY: rows of h gathered to half, [m, C]
        self.op(d, "gather", || unsafe { (k.gather_f16)(d.raw(), h.fp(), tok_d.ptr().cast(), xh.ptr().cast(), m as i64, c as i64) })?;
        let (gn, gk) = (gu.rows as i64, gu.cols as i64);
        let (dn_n, dn_k) = (dn.rows as i64, dn.cols as i64);
        // the silo's grouped products: every expert's rows in one launch, its weights decoded as it goes
        // SAFETY: arithmetic only.
        let grouped = d.silo.filter(|s| prompt_silo("NS_INK2_PGEMM") && gu.packed && dn.packed && unsafe { (s.moe_prompt_supported)(gu.ty.code() as c_int, gn, gk) } != 0
                                     && unsafe { (s.moe_prompt_supported)(dn.ty.code() as c_int, dn_n, dn_k) } != 0);
        if let Some(s) = grouped {
            let mut tiles: Vec<i32> = Vec::new();
            for &(x, r0, n) in &runs {
                for off in (0..n).step_by(64) {
                    tiles.extend([x as i32, (r0 + off) as i32, (n - off).min(64) as i32]);
                }
            }
            let tl = up_i(&tiles)?;
            let nt = (tiles.len() / 3) as i64;
            let rows_gu = (g.experts * gu.rows) as i64;
            let rows_dn = (g.experts * dn.rows) as i64;
            // SAFETY: the repacked stacks, x / act half [m, K], y / dd [m, N], the tiles.
            unsafe {
                self.op(d, "experts prompt silo", || (s.moe_prompt)(d.raw(), gu.ty.code() as c_int, gu.buf.ptr(), rows_gu, gn, gk, xh.ptr().cast(), y.fp(), tl.ptr().cast(), nt))?;
                self.op(d, "situ", || (k.situ)(d.raw(), y.fp(), rs_d.fp(), std::ptr::null_mut(), act.ptr().cast(), m as i64, ff as i64, g.situ_beta, g.situ_lb))?;
                self.op(d, "experts prompt silo", || (s.moe_prompt)(d.raw(), dn.ty.code() as c_int, dn.buf.ptr(), rows_dn, dn_n, dn_k, act.ptr().cast(), dd.fp(), tl.ptr().cast(), nt))?;
            }
            out.fill(0)?;
            // SAFETY: each token's rows weighted into out [t, C]
            self.op(d, "combine", || unsafe { (k.combine_rows)(d.raw(), dd.fp(), tp_d.ptr().cast(), ent_d.ptr().cast(), cw_d.fp(), out.fp(), t as i64, c as i64) })?;
            d.arena.rewind(mark);
            return Ok(out);
        }
        for &(x, r0, n) in &runs {
            // SAFETY: expert x's gate|up -> half in w16 (from its blocks, or the silo's repacked stack); its rows' product.
            self.expand_expert(d, gu, x, g.experts)?;
            self.op(d, "gemm f16", || unsafe {
                (k.gemm_f16)(d.raw(), n as i64, gn, gk, xh.ptr().cast::<u16>().add(r0 * c), gk, d.w16.ptr().cast(), y.fp().add(r0 * gu.rows), gn, 0)
            })?;
        }
        // SAFETY: [m, 2F] scaled by its expert's factor -> SiTU in half [m, F]
        self.op(d, "situ", || unsafe { (k.situ)(d.raw(), y.fp(), rs_d.fp(), std::ptr::null_mut(), act.ptr().cast(), m as i64, ff as i64, g.situ_beta, g.situ_lb) })?;
        for &(x, r0, n) in &runs {
            // SAFETY: expert x's down -> half in w16; its rows' product.
            self.expand_expert(d, dn, x, g.experts)?;
            self.op(d, "gemm f16", || unsafe {
                (k.gemm_f16)(d.raw(), n as i64, dn_n, dn_k, act.ptr().cast::<u16>().add(r0 * ff), dn_k, d.w16.ptr().cast(), dd.fp().add(r0 * c), dn_n, 0)
            })?;
        }
        out.fill(0)?;
        // SAFETY: each token's rows weighted into out [t, C]
        self.op(d, "combine", || unsafe { (k.combine_rows)(d.raw(), dd.fp(), tp_d.ptr().cast(), ent_d.ptr().cast(), cw_d.fp(), out.fp(), t as i64, c as i64) })?;
        d.arena.rewind(mark);
        Ok(out)
    }

    /// One layer application (slot `slot` of physical layer `l`, at `ip` in its pass) on the stream `cur`, its result
    /// into `dst`; `blocks` the depth attention's earlier states (a block's first layer adds `cur`); `msgs` [4, t, C]
    /// the loop's messages (passes 2 and 3)
    #[allow(clippy::too_many_arguments)]
    fn layer(&self, ss: &SlotState, slot: usize, l: usize, cur: &DevBuf, dst: &DevBuf, blocks: &mut Vec<DevBuf>, msgs: Option<(&Loop, &DevBuf)>, t: usize, p0: usize,
             tap: Tap) -> Result<()> {
        let (g, k) = (&self.g, self.k);
        let ly = &self.layers[l];
        let d = &self.devs[ly.dev];
        let c = g.n_embd;
        let tc = t * c * 4;
        let (hq, hk, hd) = (g.heads, g.heads_kv, g.head);
        let mark = d.arena.mark();
        // the attention's inputs: mixed over the depth (q, k, v on a full layer), then the loop's messages
        let mut st: Vec<&DevBuf> = blocks.iter().collect();
        st.push(cur);
        let mut inputs: Vec<DevBuf> = match &ly.attn_res {
            Some((q, nq)) => {
                let mixed = self.mix(d, &st, q, *nq, t)?;
                (0..3).map(|j| mixed.view(if *nq > 1 { j * tc } else { 0 }, tc)).collect::<Result<_>>()?
            }
            None => (0..3).map(|_| cur.view(0, tc)).collect::<Result<_>>()?,
        };
        if let Some((lp, m)) = msgs {
            for (j, inp) in inputs.iter_mut().enumerate() {
                *inp = self.control(d, lp, j, inp, &m.view(j * tc, tc)?, t)?;
            }
        }
        // the normed inputs (and for decode rows with the silo their quantized forms)
        let (normed, nq): (Vec<DevBuf>, Vec<Option<DevBuf>>) = if ly.swa && msgs.is_none() {
            let (n0, q0) = self.rms_q(d, &inputs[0], &ly.norm[0], t)?;
            let qv = |q: &Option<DevBuf>| -> Result<Option<DevBuf>> { q.as_ref().map(|q| q.view(0, q.len)).transpose() };
            (vec![n0.view(0, tc)?, n0.view(0, tc)?, n0], vec![qv(&q0)?, qv(&q0)?, q0])
        } else {
            let v: Vec<(DevBuf, Option<DevBuf>)> = (0..3).map(|j| self.rms_q(d, &inputs[j], &ly.norm[j], t)).collect::<Result<_>>()?;
            v.into_iter().unzip()
        };
        let qd = hq * hd;
        let [q, kr, vr, r, gate]: [DevBuf; 5] = self.mm_many(d, &[(&ly.q, &normed[0], nq[0].as_ref()), (&ly.k, &normed[1], nq[1].as_ref()),
                                                                  (&ly.v, &normed[2], nq[2].as_ref()), (&ly.rel, &normed[0], nq[0].as_ref()),
                                                                  (&ly.gate, &normed[0], nq[0].as_ref())], t)?
            .try_into().map_err(|_| e("five products"))?;
        let sink = self.mm(d, &ly.sink, &normed[0], t)?;
        let kdyn = match &ly.k_dyn {
            Some(m) => Some(self.mm(d, m, &normed[1], t)?),
            None => None,
        };
        let vdyn = match &ly.v_dyn {
            Some(m) => Some(self.mm(d, m, &normed[2], t)?),
            None => None,
        };
        let kc = self.conv(d, ss, 0, &ly.convs[0], &kr, kdyn.as_ref(), t)?;
        let vc = self.conv(d, ss, 1, &ly.convs[1], &vr, vdyn.as_ref(), t)?;
        let alpha = if ly.swa { 0.0 } else { g.log_alpha };
        // SAFETY: the rows [t, heads, dims] in place.
        unsafe {
            self.op(d, "clip", || (k.clip)(d.raw(), q.fp(), ly.q_clip.fp(), t as i64, hq as i64, hd as i64, g.qk_clip, alpha, g.log_floor, p0 as i64))?;
            self.op(d, "clip", || (k.clip)(d.raw(), kc.fp(), std::ptr::null(), t as i64, hk as i64, hd as i64, g.qk_clip, 0.0, g.log_floor, p0 as i64))?;
            self.op(d, "v norm", || (k.rms_rows)(d.raw(), vc.fp(), vc.fp(), (t * hk) as i64, hd as i64, g.eps))?;
            self.op(d, "clip", || (k.clip)(d.raw(), r.fp(), ly.rel_clip.fp(), t as i64, hq as i64, g.rel_len as i64, g.rel_clip, alpha, g.log_floor, p0 as i64))?;
        }
        tap(&format!("Qcur-{slot}"), &q)?;
        tap(&format!("Kcur-{slot}"), &kc)?;
        tap(&format!("Vcur-{slot}"), &vc)?;
        tap(&format!("Rcur-{slot}"), &r)?;
        let att = self.f32(d, t * qd)?;
        let win = g.win(slot) as i64;
        let extent = if ly.swa { g.window } else { g.rel_extent } as i64;
        // SAFETY: the rows and caches as ink.h names them.
        unsafe {
            self.op(d, "kv store", || (k.kv_store)(d.raw(), kc.fp(), vc.fp(), t as i64, hk as i64, hd as i64, ss.cap as i64, p0 as i64, ss.kc.ptr(), ss.vc.ptr()))?;
            match d.silo.filter(|s| t > MAXV && prompt_silo("NS_INK2_PATTN") && (s.attn_prompt_supported)(hq as i64, hk as i64, hd as i64, g.rel_len as i64) != 0) {
                // a prompt pass's rows: the silo's flash attention
                Some(s) => self.op(d, "attention silo", || (s.attn_prompt)(d.raw(), q.fp(), ss.kc.ptr(), ss.vc.ptr(), r.fp(), ly.rel_bias.fp(), extent,
                                                                          g.rel_len as i64, sink.fp(), att.fp(), t as i64, hq as i64, hk as i64, ss.cap as i64,
                                                                          p0 as i64, win))?,
                None => {
                    let sc = (k.attn_scratch)(t as i64, hq as i64, hd as i64, p0 as i64, win) as usize;
                    let part = if sc > 0 { Some(self.f32(d, sc)?) } else { None };
                    self.op(d, "attention", || (k.attn)(d.raw(), q.fp(), ss.kc.ptr(), ss.vc.ptr(), r.fp(), ly.rel_bias.fp(), extent, g.rel_len as i64, sink.fp(),
                                                        att.fp(), t as i64, hq as i64, hk as i64, hd as i64, ss.cap as i64, p0 as i64, win,
                                                        1.0 / (hd as f32).sqrt(), part.as_ref().map_or(std::ptr::null_mut(), |p| p.fp())))?;
                }
            }
            self.op(d, "gate", || (k.gate)(d.raw(), att.fp(), gate.fp(), (t * qd) as i64))?;
        }
        let aq = self.xq(d, &att, qd, t)?;
        let o = self.mmq(d, &ly.o, &att, aq.as_ref(), t)?;
        let o = self.conv(d, ss, 2, &ly.convs[2], &o, None, t)?;
        tap(&format!("attn_out-{slot}"), &o)?;
        // the first layer of a block starts the stream again (the old one joins the blocks)
        if l.is_multiple_of(g.block) {
            blocks.push(cur.view(0, tc)?);
            dst.copy_within(0, &o, 0, tc)?;
        } else {
            // SAFETY: [t, C] each.
            self.op(d, "add", || unsafe { (k.add)(d.raw(), cur.fp(), o.fp(), dst.fp(), (t * c) as i64) })?;
        }
        tap(&format!("ffn_inp-{slot}"), dst)?;
        // the attention's work is done with (the stream is in order): the FFN's reuses it
        d.arena.rewind(mark);
        let mut st: Vec<&DevBuf> = blocks.iter().collect();
        st.push(dst);
        let mut m = self.mix(d, &st, &ly.ffn_res, 1, t)?;
        if let Some((lp, msg)) = msgs {
            m = self.control(d, lp, 3, &m, &msg.view(3 * tc, tc)?, t)?;
        }
        let (h, hq) = self.rms_q(d, &m, &ly.ffn_norm, t)?;
        let f = self.moe(d, ly, &h, hq.as_ref(), t)?;
        let f = self.conv(d, ss, 3, &ly.convs[3], &f, None, t)?;
        tap(&format!("ffn_out-{slot}"), &f)?;
        // SAFETY: [t, C] each.
        self.op(d, "add", || unsafe { (k.add)(d.raw(), dst.fp(), f.fp(), dst.fp(), (t * c) as i64) })?;
        tap(&format!("l_out-{slot}"), dst)?;
        d.arena.rewind(mark);
        Ok(())
    }

    /// The silo's fused decode steps for a pass of `t` rows on `d` (decode rows, the silo there, NS_INK2_FUSED not 0)
    fn fused(&self, d: &Dev, t: usize) -> Option<&'static ffi::Silo> {
        self.dec(d, t).filter(|_| self.fused_on)
    }

    /// A slot's convolution state group `grp` ([3, ch]) and, for decode rows, its snapshot ([3 + t, ch])
    fn conv_bufs(&self, ss: &SlotState, grp: usize, t: usize) -> Result<(DevBuf, Option<DevBuf>)> {
        let groups = self.g.conv_groups();
        let ch = groups[grp];
        let off: usize = groups[..grp].iter().sum();
        let snap = if t <= MAXV { Some(ss.snap.view((3 + MAXV) * off * 4, (3 + t) * ch * 4)?) } else { None };
        Ok((ss.conv.view(3 * off * 4, 3 * ch * 4)?, snap))
    }

    /// A loop controller at decode width, the silo's two launches
    #[allow(clippy::too_many_arguments)]
    fn control_dec(&self, d: &Dev, f: &ffi::Fused, lp: &Loop, which: usize, x: &DevBuf, msg: &DevBuf, t: usize) -> Result<DevBuf> {
        let (c, rank) = (self.g.n_embd, self.g.rank);
        let u = self.f32(d, t * rank)?;
        let out = self.f32(d, t * c)?;
        // SAFETY: x, msg, out [t, c]; down [rank, 2c], gain / forget [c, rank] float32; u [t, rank].
        unsafe {
            self.op(d, "ctrl down silo", || (f.ctrl_down)(d.raw(), x.fp(), msg.fp(), lp.down[which].buf.fp(), u.fp(), t as i64, c as i64, rank as i64, 1e-6))?;
            self.op(d, "ctrl apply silo", || (f.ctrl_apply)(d.raw(), x.fp(), msg.fp(), u.fp(), lp.gain[which].buf.fp(), lp.forget[which].buf.fp(), out.fp(), t as i64,
                                                            c as i64, rank as i64))?;
        }
        Ok(out)
    }

    /// Loop controllers `which` of `lp` on their inputs and messages, in one launch for the down products and one for
    /// the rest
    #[allow(clippy::too_many_arguments)]
    fn control_many(&self, d: &Dev, f: &ffi::Fused, lp: &Loop, which: &[usize], xs: &[&DevBuf], msgs: &[&DevBuf], t: usize) -> Result<Vec<DevBuf>> {
        let (c, rank) = (self.g.n_embd, self.g.rank);
        let n = which.len();
        let us: Vec<DevBuf> = (0..n).map(|_| self.f32(d, t * rank)).collect::<Result<_>>()?;
        let outs: Vec<DevBuf> = (0..n).map(|_| self.f32(d, t * c)).collect::<Result<_>>()?;
        let xp: Vec<*const f32> = xs.iter().map(|x| x.fp() as *const f32).collect();
        let mp: Vec<*const f32> = msgs.iter().map(|x| x.fp() as *const f32).collect();
        let dp: Vec<*const f32> = which.iter().map(|&j| lp.down[j].buf.fp() as *const f32).collect();
        let gp: Vec<*const f32> = which.iter().map(|&j| lp.gain[j].buf.fp() as *const f32).collect();
        let fp: Vec<*const f32> = which.iter().map(|&j| lp.forget[j].buf.fp() as *const f32).collect();
        let up: Vec<*mut f32> = us.iter().map(|u| u.fp()).collect();
        let ucp: Vec<*const f32> = us.iter().map(|u| u.fp() as *const f32).collect();
        let op: Vec<*mut f32> = outs.iter().map(|o| o.fp()).collect();
        // SAFETY: each controller's x, msg, out [t, c]; down [rank, 2c], gain / forget [c, rank] float32; u [t, rank].
        unsafe {
            self.op(d, "ctrl down silo", || (f.ctrl_down_many)(d.raw(), n as c_int, xp.as_ptr(), mp.as_ptr(), dp.as_ptr(), up.as_ptr(), t as i64, c as i64, rank as i64,
                                                                 1e-6))?;
            self.op(d, "ctrl apply silo", || (f.ctrl_apply_many)(d.raw(), n as c_int, xp.as_ptr(), mp.as_ptr(), ucp.as_ptr(), gp.as_ptr(), fp.as_ptr(), op.as_ptr(),
                                                                   t as i64, c as i64, rank as i64))?;
        }
        Ok(outs)
    }

    /// Rows normed (each input its weight) and quantized for the silo's products, in one launch
    fn rms_q_many(&self, d: &Dev, s: &ffi::Silo, xs: &[&DevBuf], ws: &[&DevBuf], t: usize) -> Result<(Vec<DevBuf>, Vec<DevBuf>)> {
        let c = self.g.n_embd;
        let ys: Vec<DevBuf> = xs.iter().map(|_| self.f32(d, t * c)).collect::<Result<_>>()?;
        // SAFETY: arithmetic only.
        let qs: Vec<DevBuf> = xs.iter().map(|_| self.ab(d, unsafe { (s.xq_bytes)(t as i64, c as i64) })).collect::<Result<_>>()?;
        let xp: Vec<*const f32> = xs.iter().map(|x| x.fp() as *const f32).collect();
        let wp: Vec<*const f32> = ws.iter().map(|w| w.fp() as *const f32).collect();
        let yp: Vec<*mut f32> = ys.iter().map(|y| y.fp()).collect();
        let qp: Vec<*mut std::ffi::c_void> = qs.iter().map(|q| q.ptr()).collect();
        // SAFETY: each x [t, c] normed into its y and quantized into its q.
        self.op(d, "rms xq", || unsafe { (s.rms_xq_many)(d.raw(), xs.len() as c_int, xp.as_ptr(), wp.as_ptr(), yp.as_ptr(), qp.as_ptr(), t as i64, c as i64, self.g.eps) })?;
        Ok((ys, qs))
    }

    /// The depth attention of `states` by `nq` queries, each normed by its weight (None: left unnormed) and quantized:
    /// [nq, t, C] and each query's quantized rows
    #[allow(clippy::too_many_arguments)]
    fn mix_norm(&self, d: &Dev, f: &ffi::Fused, states: &[&DevBuf], query: &DevBuf, nq: usize, norms: Option<&[&DevBuf]>, t: usize)
                -> Result<(DevBuf, Vec<Option<DevBuf>>)> {
        let c = self.g.n_embd;
        let out = self.f32(d, nq * t * c)?;
        let xqs: Vec<Option<DevBuf>> = (0..nq).map(|_| -> Result<Option<DevBuf>> {
            match (norms, d.silo) {
                // SAFETY: arithmetic only.
                (Some(_), Some(s)) => Ok(Some(self.ab(d, unsafe { (s.xq_bytes)(t as i64, c as i64) })?)),
                _ => Ok(None),
            }
        }).collect::<Result<_>>()?;
        let sp: Vec<*const f32> = states.iter().map(|s| s.fp() as *const f32).collect();
        let nw: Vec<*const f32> = match norms {
            Some(n) => n.iter().map(|w| w.fp() as *const f32).collect(),
            None => vec![std::ptr::null(); nq],
        };
        let qp: Vec<*mut std::ffi::c_void> = xqs.iter().map(|q| q.as_ref().map_or(std::ptr::null_mut(), |q| q.ptr())).collect();
        // SAFETY: the states [t, c], the queries [nq, c], the norm weights [c], out [nq, t, c], each xq for [t, c].
        self.op(d, "mix norm silo", || unsafe {
            (f.mix_norm)(d.raw(), sp.as_ptr(), sp.len() as i64, query.fp(), nq as i64, nw.as_ptr(), out.fp(), qp.as_ptr(), t as i64, c as i64, 1e-6, self.g.eps)
        })?;
        Ok((out, xqs))
    }

    /// `layer` for decode rows with the silo: the same steps in fused launches
    #[allow(clippy::too_many_arguments)]
    fn layer_dec(&self, sl: &ffi::Silo, ss: &SlotState, slot: usize, l: usize, cur: &DevBuf, dst: &DevBuf, blocks: &mut Vec<DevBuf>, msgs: Option<(&Loop, &DevBuf)>,
                 t: usize, p0: usize, tap: Tap) -> Result<()> {
        let (g, f) = (&self.g, &sl.fused);
        let ly = &self.layers[l];
        let d = &self.devs[ly.dev];
        let c = g.n_embd;
        let tc = t * c * 4;
        let (hq, hk, hd) = (g.heads, g.heads_kv, g.head);
        let (qd, kd) = (hq * hd, hk * hd);
        let mark = d.arena.mark();
        let mut st: Vec<&DevBuf> = blocks.iter().collect();
        st.push(cur);
        // the attention's normed (and quantized) inputs: q, k, v
        let mut normed: Vec<DevBuf> = Vec::with_capacity(3);
        let mut nq: Vec<Option<DevBuf>> = Vec::with_capacity(3);
        let norms: Vec<&DevBuf> = ly.norm.iter().collect();
        match (&ly.attn_res, msgs) {
            (Some((q, n)), None) => {
                let (out, xs) = self.mix_norm(d, f, &st, q, *n, Some(&norms[..*n]), t)?;
                for j in 0..3 {
                    let k = if *n > 1 { j } else { 0 };
                    normed.push(out.view(k * tc, tc)?);
                    nq.push(xs[k].as_ref().map(|q| q.view(0, q.len)).transpose()?);
                }
            }
            (Some((q, n)), Some((lp, m))) => {
                // the three controllers in one launch each, their outputs normed and quantized in one more
                let (out, _) = self.mix_norm(d, f, &st, q, *n, None, t)?;
                let inps: Vec<DevBuf> = (0..3).map(|j| out.view(if *n > 1 { j * tc } else { 0 }, tc)).collect::<Result<_>>()?;
                let msgs: Vec<DevBuf> = (0..3).map(|j| m.view(j * tc, tc)).collect::<Result<_>>()?;
                let xs = self.control_many(d, f, lp, &[0, 1, 2], &inps.iter().collect::<Vec<_>>(), &msgs.iter().collect::<Vec<_>>(), t)?;
                let (ys, qs) = self.rms_q_many(d, sl, &xs.iter().collect::<Vec<_>>(), &[&ly.norm[0], &ly.norm[1], &ly.norm[2]], t)?;
                normed = ys;
                nq = qs.into_iter().map(Some).collect();
            }
            (None, _) => {
                let (y, q) = self.rms_q(d, cur, &ly.norm[0], t)?;
                for _ in 0..3 {
                    normed.push(y.view(0, tc)?);
                    nq.push(q.as_ref().map(|q| q.view(0, q.len)).transpose()?);
                }
            }
        }
        let [q, kr, vr, r, gate]: [DevBuf; 5] = self.mm_many(d, &[(&ly.q, &normed[0], nq[0].as_ref()), (&ly.k, &normed[1], nq[1].as_ref()),
                                                                  (&ly.v, &normed[2], nq[2].as_ref()), (&ly.rel, &normed[0], nq[0].as_ref()),
                                                                  (&ly.gate, &normed[0], nq[0].as_ref())], t)?
            .try_into().map_err(|_| e("five products"))?;
        // the sink and (full layers) the k / v taps: BF16 projections made in the post-projection launch when they are
        // BF16, else products of their own
        let bf = |m: &Mat| m.ty == GType::BF16 && m.cols == g.n_embd;
        let inline = bf(&ly.sink) && ly.k_dyn.as_ref().is_none_or(bf) && ly.v_dyn.as_ref().is_none_or(bf);
        let sink = if inline { self.f32(d, t * hq)? } else { self.mm(d, &ly.sink, &normed[0], t)? };
        let kdyn = if inline { None } else { ly.k_dyn.as_ref().map(|m| self.mm(d, m, &normed[1], t)).transpose()? };
        let vdyn = if inline { None } else { ly.v_dyn.as_ref().map(|m| self.mm(d, m, &normed[2], t)).transpose()? };
        // k and v through their convolutions into the caches, q and r clipped
        let (kst, ksn) = self.conv_bufs(ss, 0, t)?;
        let (vst, vsn) = self.conv_bufs(ss, 1, t)?;
        let kout = self.f32(d, t * kd)?;
        let vout = self.f32(d, t * kd)?;
        let alpha = if ly.swa { 0.0 } else { g.log_alpha };
        let p = ffi::Qkv {
            t: t as i64, p0: p0 as i64, cap: ss.cap as i64, hq: hq as i64, hkv: hk as i64, d: hd as i64, r_len: g.rel_len as i64, kraw: kr.fp(), vraw: vr.fp(),
            kdyn: kdyn.as_ref().map_or(std::ptr::null(), |x| x.fp()), vdyn: vdyn.as_ref().map_or(std::ptr::null(), |x| x.fp()), kstate: kst.fp(), vstate: vst.fp(),
            ksnap: ksn.as_ref().map_or(std::ptr::null_mut(), |s| s.fp()), vsnap: vsn.as_ref().map_or(std::ptr::null_mut(), |s| s.fp()), kw: ly.convs[0].fp(),
            vw: ly.convs[1].fp(), inv: 1.0 / ((g.conv + 1) as f32).sqrt(), inv_dyn: 1.0 / ((2 * g.conv + 1) as f32).sqrt(), kout: kout.fp(), vout: vout.fp(),
            kc: ss.kc.ptr(), vc: ss.vc.ptr(), q: q.fp(), r: r.fp(), qscale: ly.q_clip.fp(), rscale: ly.rel_clip.fp(), thr: g.qk_clip, rthr: g.rel_clip, alpha,
            floor: g.log_floor, eps: g.eps, c: c as i64, n0: normed[0].fp(), n1: normed[1].fp(), n2: normed[2].fp(),
            sinkw: if inline { ly.sink.buf.ptr().cast() } else { std::ptr::null() }, sink: sink.fp(),
            kdw: if inline { ly.k_dyn.as_ref().map_or(std::ptr::null(), |m| m.buf.ptr().cast()) } else { std::ptr::null() },
            vdw: if inline { ly.v_dyn.as_ref().map_or(std::ptr::null(), |m| m.buf.ptr().cast()) } else { std::ptr::null() },
        };
        // SAFETY: every buffer as Ink2sQkv describes it.
        self.op(d, "post qkv silo", || unsafe { (f.post_qkv)(d.raw(), &p) })?;
        tap(&format!("Qcur-{slot}"), &q)?;
        tap(&format!("Kcur-{slot}"), &kout)?;
        tap(&format!("Vcur-{slot}"), &vout)?;
        tap(&format!("Rcur-{slot}"), &r)?;
        // the attention split by key positions, merged with the sink, gated and quantized for the output product
        let win = g.win(slot) as i64;
        let extent = if ly.swa { g.window } else { g.rel_extent } as i64;
        // SAFETY: arithmetic only.
        let ns = unsafe { (f.attn_splits)(t as i64, p0 as i64, win) } as usize;
        let part = self.f32(d, t * hq * ns * (hd + 2))?;
        let att = self.f32(d, t * qd)?;
        // SAFETY: arithmetic only.
        let aq = self.ab(d, unsafe { (sl.xq_bytes)(t as i64, qd as i64) })?;
        // SAFETY: the rows and caches as silo.h names them.
        unsafe {
            self.op(d, "attention silo", || (f.attn_part)(d.raw(), q.fp(), ss.kc.ptr(), ss.vc.ptr(), r.fp(), ly.rel_bias.fp(), extent, g.rel_len as i64, part.fp(),
                                                          t as i64, hq as i64, hk as i64, ss.cap as i64, p0 as i64, win, 1.0 / (hd as f32).sqrt()))?;
            self.op(d, "merge silo", || (f.merge_gate_xq)(d.raw(), part.fp(), sink.fp(), gate.fp(), att.fp(), aq.ptr(), t as i64, hq as i64, ns as i64))?;
        }
        let o = self.mmq(d, &ly.o, &att, Some(&aq), t)?;
        // its convolution and the residual (a block's first layer: the stream starts again, the old one joins the blocks)
        let (ast, asn) = self.conv_bufs(ss, 2, t)?;
        let o2 = self.f32(d, t * c)?;
        let start = l.is_multiple_of(g.block);
        let inv = 1.0 / ((g.conv + 1) as f32).sqrt();
        // SAFETY: o, o2, dst, cur [t, c]; the state, its weights, the snapshot.
        self.op(d, "conv res silo", || unsafe {
            (f.conv_res)(d.raw(), o.fp(), ast.fp(), ly.convs[2].fp(), asn.as_ref().map_or(std::ptr::null_mut(), |s| s.fp()),
                         if start { std::ptr::null() } else { cur.fp() }, dst.fp(), o2.fp(), t as i64, c as i64, inv)
        })?;
        if start {
            blocks.push(cur.view(0, tc)?);
        }
        tap(&format!("attn_out-{slot}"), &o2)?;
        tap(&format!("ffn_inp-{slot}"), dst)?;
        d.arena.rewind(mark);
        // the FFN's normed input
        let mut st: Vec<&DevBuf> = blocks.iter().collect();
        st.push(dst);
        let (h, hq_) = match msgs {
            None => {
                let (out, mut xs) = self.mix_norm(d, f, &st, &ly.ffn_res, 1, Some(&[&ly.ffn_norm]), t)?;
                (out, xs.pop().flatten())
            }
            Some((lp, msg)) => {
                let (m, _) = self.mix_norm(d, f, &st, &ly.ffn_res, 1, None, t)?;
                let x = self.control_dec(d, f, lp, 3, &m, &msg.view(3 * tc, tc)?, t)?;
                self.rms_q(d, &x, &ly.ffn_norm, t)?
            }
        };
        let fo = self.moe(d, ly, &h, hq_.as_ref(), t)?;
        let (fst, fsn) = self.conv_bufs(ss, 3, t)?;
        let f2 = self.f32(d, t * c)?;
        // SAFETY: fo, f2, dst [t, c] (dst += the convolved FFN output); the state, its weights, the snapshot.
        self.op(d, "conv res silo", || unsafe {
            (f.conv_res)(d.raw(), fo.fp(), fst.fp(), ly.convs[3].fp(), fsn.as_ref().map_or(std::ptr::null_mut(), |s| s.fp()), dst.fp(), dst.fp(), f2.fp(),
                         t as i64, c as i64, inv)
        })?;
        tap(&format!("ffn_out-{slot}"), &f2)?;
        tap(&format!("l_out-{slot}"), dst)?;
        d.arena.rewind(mark);
        Ok(())
    }

    fn pass(&self, s: &mut Session, tokens: &[u32], n_out: usize, tap: Tap) -> Result<Vec<Vec<f32>>> {
        let _w = self.work.lock().unwrap();
        match self.pass_out(s, tokens, n_out, None, tap)? {
            Out::Logits(l) => Ok(l),
            Out::Ids(_) => Err(e("a pass's logits")),
        }
    }

    /// The pass's rows from GPU `from` to GPU `to` before an application there: the bank's streams the rest of the pass
    /// reads (`live`; a decode pass's whole bank in one copy), and the drafter's features of its rows
    fn hop(&self, from: usize, to: usize, banks: &[DevBuf], live: &[usize], t: usize, feats: bool) -> Result<()> {
        let tc = t * self.g.n_embd * 4;
        let (src, dst) = (&self.devs[from], &self.devs[to]);
        let mut staging = src.staging.lock().unwrap();
        let mut copy = |dbuf: &DevBuf, sbuf: &DevBuf, at: usize, len: usize| -> Result<()> {
            let mut o = 0;
            while o < len {
                let n = (len - o).min(STAGING);
                dbuf.view(at + o, n)?.copy_from_peer_async(&sbuf.view(at + o, n)?, &mut staging)?;
                o += n;
            }
            Ok(())
        };
        if t <= MAXV {
            copy(&banks[to], &banks[from], 0, BANK * tc)?;
        } else {
            for &sl in live {
                copy(&banks[to], &banks[from], sl * tc, tc)?;
            }
        }
        if let (true, Some(fs), Some(fd)) = (feats, &src.feats, &dst.feats) {
            copy(fd, fs, 0, t * dspark::FEATS * self.g.n_embd * 4)?;
        }
        Ok(())
    }

    /// A pass's last `n_out` rows: their logits, or (`pick`: a decoder whose logits nothing reads) each one's token
    /// picked on the GPU - the argmax (greedy), or a draw (`Some(sampling)`)
    fn pass_out(&self, s: &mut Session, tokens: &[u32], n_out: usize, pick: Option<Option<nextcuda_llm::DeviceSampling>>, tap: Tap) -> Result<Out> {
        let g = &self.g;
        let t = tokens.len();
        if t == 0 {
            return Err(e("an empty pass"));
        }
        if t > self.chunk {
            return Err(e(format!("a pass of {t} rows; it takes {}", self.chunk)));
        }
        if s.pos + t > s.max_ctx {
            return Err(e(format!("the context is full ({} of {})", s.pos + t, s.max_ctx)));
        }
        for d in &self.devs {
            d.arena.reset();
        }
        let p0 = s.pos;
        let c = g.n_embd;
        let tc = t * c * 4;
        // each GPU's bank of the pass's streams (X0 .. SB) and its messages
        let banks: Vec<DevBuf> = self.devs.iter().map(|d| self.f32(d, BANK * t * c)).collect::<Result<_>>()?;
        let msgs: Vec<DevBuf> = self.devs.iter().map(|d| self.f32(d, 4 * t * c)).collect::<Result<_>>()?;
        let sv = |di: usize, sl: usize| banks[di].view(sl * tc, tc);
        let mut at = self.layers[0].dev;
        {
            let d = &self.devs[at];
            let emb = self.embed(d, tokens)?;
            tap("inp_embd", &emb)?;
            let x0 = sv(at, X0)?;
            // SAFETY: emb, x0 [t, c], the norm's weight on the head's GPU (the embedding's)
            let tn = &self.tok_norm;
            self.op(d, "rms norm", || unsafe { (self.k.rms_norm)(d.raw(), emb.fp(), tn.fp(), x0.fp(), t as i64, c as i64, g.eps) })?;
        }
        let (cs, ce) = g.core;
        // the depth attention's blocks and the stream, as bank slots
        let mut blocks: Vec<usize> = Vec::new();
        let mut cur = X0;
        let mut prelude_blocks = 0;
        let drafting = s.ds.is_some() && self.ds.is_some();
        let taps: &[usize] = self.ds.as_ref().map_or(&[], |d| &d.taps);
        for (app, (slot, l, p)) in g.order().into_iter().enumerate() {
            let dv = self.layers[l].dev;
            if dv != at {
                // the streams the rest of the pass reads: the blocks, the stream, the loop's prelude output and bank
                let mut live: Vec<usize> = blocks.clone();
                live.extend([cur, PRE, S12P, POUT]);
                live.sort_unstable();
                live.dedup();
                self.hop(at, dv, &banks, &live, t, drafting)?;
                at = dv;
            }
            let d = &self.devs[at];
            if l == cs {
                if p == 0 {
                    banks[at].copy_within(PRE * tc, &banks[at], cur * tc, tc)?;
                    prelude_blocks = blocks.len();
                } else {
                    // the bank: the previous pass's blocks after its first (its block at the loop's start is the prelude)
                    if blocks.len() != prelude_blocks + 2 {
                        return Err(e(format!("the loop's bank: {} blocks", blocks.len())));
                    }
                    banks[at].copy_within(S12P * tc, &banks[at], S12 * tc, tc)?;
                    banks[at].copy_within(POUT * tc, &banks[at], cur * tc, tc)?;
                    blocks.truncate(prelude_blocks);
                }
                cur = PRE;
            }
            if l > cs && l.is_multiple_of(g.block) {
                // a block's first layer: the stream into a buffer of its own (it joins the blocks)
                let keep = if l < ce { S12 } else { S18 };
                banks[at].copy_within(keep * tc, &banks[at], cur * tc, tc)?;
                cur = keep;
            }
            let lp = if p > 0 {
                let lp = &self.loops[l - cs];
                let mk = d.arena.mark();
                let (sp, po) = (sv(at, S12P)?, sv(at, POUT)?);
                let m = match self.fused(d, t) {
                    Some(sl) => self.mix_norm(d, &sl.fused, &[&sp, &po], &lp.msg_q, 4, None, t)?.0,
                    None => self.mix(d, &[&sp, &po], &lp.msg_q, 4, t)?,
                };
                msgs[at].copy_within(0, &m, 0, 4 * tc)?;
                d.arena.rewind(mk);
                Some((lp, &msgs[at]))
            } else {
                None
            };
            if let (true, Some(fb), Some(k)) = (drafting, d.feats.as_ref(), taps.iter().position(|&a| a == app)) {
                // the drafter's feature k: the stream going into this application
                let sl = d.silo.ok_or_else(|| e("drafts need the silo"))?;
                let cv = sv(at, cur)?;
                // SAFETY: cur [t, c] -> feature k of fb [t, 4, c].
                self.op(d, "draft feature", || unsafe { (sl.draft.feat_put)(d.raw(), cv.fp(), fb.fp(), t as i64, c as i64, dspark::FEATS as i64, k as i64) })?;
            }
            let dst = if cur == SA { SB } else { SA };
            let (cv, dv_) = (sv(at, cur)?, sv(at, dst)?);
            let mut bv: Vec<DevBuf> = blocks.iter().map(|b| sv(at, *b)).collect::<Result<_>>()?;
            match self.fused(d, t) {
                Some(sl) => self.layer_dec(sl, &s.slots[slot], slot, l, &cv, &dv_, &mut bv, lp, t, p0, &mut *tap)?,
                None => self.layer(&s.slots[slot], slot, l, &cv, &dv_, &mut bv, lp, t, p0, &mut *tap)?,
            }
            if l.is_multiple_of(g.block) {
                blocks.push(cur);
            }
            cur = dst;
        }
        if at != self.hd {
            let mut live: Vec<usize> = blocks.clone();
            live.push(cur);
            live.sort_unstable();
            live.dedup();
            self.hop(at, self.hd, &banks, &live, t, drafting)?;
        }
        s.pos += t;
        s.last_rows = t;
        let d = self.hdev();
        let hd = self.hd;
        let n = n_out.clamp(1, t);
        let bv: Vec<DevBuf> = blocks.iter().chain(std::iter::once(&cur)).map(|b| sv(hd, *b)).collect::<Result<_>>()?;
        let st: Vec<&DevBuf> = bv.iter().collect();
        let fin = self.mix(d, &st, &self.output_res, 1, t)?;
        let normed = match (drafting, d.feats.as_ref()) {
            (true, Some(fb)) => {
                // every row's normed head input: the drafter's last feature, then its context and f_rnn
                let all = self.rms(d, &fin, &self.output_norm, t)?;
                let sl = d.silo.ok_or_else(|| e("drafts need the silo"))?;
                // SAFETY: all [t, c] -> feature 3 of fb.
                self.op(d, "draft feature", || unsafe {
                    (sl.draft.feat_put)(d.raw(), all.fp(), fb.fp(), t as i64, c as i64, dspark::FEATS as i64, (dspark::FEATS - 1) as i64)
                })?;
                self.ds_inject(s, t, p0)?;
                all.view((t - n) * c * 4, n * c * 4)?
            }
            _ => {
                let last = fin.view((t - n) * c * 4, n * c * 4)?;
                self.rms(d, &last, &self.output_norm, n)?
            }
        };
        tap("result_norm", &normed)?;
        let vp = self.output.rows;
        let v = g.vocab;
        let raw = self.f32(d, n * vp)?;
        for r0 in (0..n).step_by(MAXV) {
            let rn = (n - r0).min(MAXV);
            let x = normed.view(r0 * c * 4, rn * c * 4)?;
            let xq = self.xq(d, &x, c, rn)?;
            let y = self.mmq(d, &self.output, &x, xq.as_ref(), rn)?;
            raw.copy_within(r0 * vp * 4, &y, 0, rn * vp * 4)?;
        }
        let logits = self.f32(d, n * v)?;
        // SAFETY: n rows of vp -> n rows of v, soft-capped
        self.op(d, "softcap", || unsafe { (self.k.softcap)(d.raw(), raw.fp(), logits.fp(), n as i64, v as i64, vp as i64, g.softcap) })?;
        tap("result_output", &logits)?;
        if let Some(how) = pick {
            let best = self.ab(d, n * 4)?;
            match how {
                // SAFETY: logits [n, vocab] -> n int32.
                None => self.op(d, "argmax", || unsafe { (self.k.argmax)(d.raw(), logits.fp(), best.ptr().cast(), n as i64, v as i64) })?,
                Some(ds) => {
                    let pos0 = (s.pos - n) as i64;
                    // SAFETY: each row's draw (row i at its position) into n int32.
                    self.op(d, "sample", || unsafe {
                        (self.k.sample)(d.raw(), logits.fp(), best.ptr().cast(), n as i64, v as i64, ds.temperature, ds.top_p, ds.seed, pos0)
                    })?;
                }
            }
            let mut bb = vec![0u8; n * 4];
            best.read(0, &mut bb)?;
            self.stamps_in()?;
            return Ok(Out::Ids(bb.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()));
        }
        let all = logits.to_f32()?;
        self.stamps_in()?;
        Ok(Out::Logits(all.chunks(v).map(|c| c.to_vec()).collect()))
    }

    /// The bytes a session of `max_ctx` takes on engine GPU `di`
    fn session_need(&self, di: usize, max_ctx: usize) -> usize {
        let on = |l: usize| self.layers[l].dev == di;
        session_bytes(&self.g, max_ctx, self.chunk, &on) + if di == self.hd { self.ds.as_ref().map_or(0, |_| dspark::DSpark::session_bytes(&self.g, max_ctx, self.chunk, MAXV)) } else { 0 }
    }

    pub fn session(&self, max_ctx: usize) -> Result<Session> {
        let g = &self.g;
        // checked against what is free first, each GPU its slots (a refusal names the card: no_room)
        for (di, d) in self.devs.iter().enumerate() {
            let need = self.session_need(di, max_ctx) as u64;
            let free = d.gpu.free_or_guess()?;
            if need + (256 << 20) > free {
                return Err(Error(nextcuda_core::no_room_message(d.gpu.index, ((need + (256 << 20) - free) >> 20) + 1,
                                                                &format!("inkling2: a session of {max_ctx} tokens needs {:.2} GiB on {}; {:.2} GiB are free",
                                                                         gib(need), d.gpu.name, gib(free)))));
            }
        }
        let conv: usize = g.conv_groups().iter().sum();
        let mut slots = Vec::with_capacity(g.slots());
        for sl in 0..g.slots() {
            let gpu = &self.dev_of(g.phys(sl)).gpu;
            let cap = g.cap(sl, max_ctx, self.chunk);
            let n = g.heads_kv * cap * g.head * 2;
            let st = DevBuf::new(gpu, 3 * conv * 4)?;
            st.fill(0)?;
            slots.push(SlotState { kc: DevBuf::new(gpu, n)?, vc: DevBuf::new(gpu, n)?, cap, conv: st, snap: DevBuf::new(gpu, (3 + MAXV) * conv * 4)? });
        }
        let ds = self.ds_state(max_ctx)?;
        for d in &self.devs {
            d.gpu.sync()?;
        }
        Ok(Session { pos: 0, max_ctx, slots, last_rows: 0, ds })
    }

    pub fn reset_session(&self, s: &mut Session) -> Result<()> {
        for sl in &s.slots {
            sl.conv.fill(0)?;
        }
        s.pos = 0;
        s.last_rows = 0;
        Ok(())
    }

    /// Positions [from, to) of one ring (`heads` of cap slots) into another (dcap slots), in runs that wrap neither
    #[allow(clippy::too_many_arguments)]
    fn copy_ring(&self, dst: &DevBuf, dcap: usize, src: &DevBuf, scap: usize, heads: usize, from: usize, to: usize) -> Result<()> {
        let row = self.g.head * 2;
        for h in 0..heads {
            let mut p = from;
            while p < to {
                let (si, di) = (p % scap, p % dcap);
                let run = (to - p).min(scap - si).min(dcap - di);
                dst.copy_within((h * dcap + di) * row, src, (h * scap + si) * row, run * row)?;
                p += run;
            }
        }
        Ok(())
    }

    /// The first position a ring with window `w` still reads at `pos`
    fn kept(w: usize, pos: usize) -> usize {
        if w > 0 { pos.saturating_sub(w - 1) } else { 0 }
    }

    pub fn copy_session(&self, dst: &mut Session, src: &Session) -> Result<()> {
        if src.pos > dst.max_ctx {
            return Err(e(format!("a session of {} tokens into one of {}", src.pos, dst.max_ctx)));
        }
        for (sl, (d, sc)) in dst.slots.iter().zip(&src.slots).enumerate() {
            let from = Self::kept(self.g.win(sl), src.pos);
            self.copy_ring(&d.kc, d.cap, &sc.kc, sc.cap, self.g.heads_kv, from, src.pos)?;
            self.copy_ring(&d.vc, d.cap, &sc.vc, sc.cap, self.g.heads_kv, from, src.pos)?;
            d.conv.copy_within(0, &sc.conv, 0, d.conv.len)?;
        }
        if let (Some(dd), Some(sd)) = (dst.ds.as_mut(), src.ds.as_ref()) {
            self.ds_copy(dd, sd, src.pos, src.last_rows)?;
        }
        dst.pos = src.pos;
        dst.last_rows = 0;
        Ok(())
    }

    pub fn feed_until(&self, s: &mut Session, tokens: &[u32], stop: &(dyn Fn(usize) -> bool + Sync), tap: Tap) -> Result<(usize, Vec<f32>)> {
        let mut logits = Vec::new();
        let mut done = 0;
        for c in tokens.chunks(self.chunk) {
            if done > 0 && stop(done) {
                break;
            }
            logits = self.pass(s, c, 1, &mut *tap)?.pop().unwrap_or_default();
            done += c.len();
        }
        Ok((done, logits))
    }

    pub fn forward_rows(&self, s: &mut Session, tokens: &[u32], n_out: usize, tap: Tap) -> Result<Vec<Vec<f32>>> {
        self.pass(s, tokens, n_out, tap)
    }

    /// Back to the first `keep` rows of the last pass: the rings' later slots are written again; the convolutions'
    /// states from the pass's inputs (a decode pass keeps them)
    pub fn rollback(&self, s: &mut Session, keep: usize) -> Result<()> {
        if keep > s.last_rows {
            return Err(e(format!("a rollback to {keep} rows of a pass of {}", s.last_rows)));
        }
        if keep < s.last_rows {
            if s.last_rows > MAXV {
                return Err(e(format!("a rollback inside a prompt pass of {} rows", s.last_rows)));
            }
            self.conv_keep(s, keep)?;
        }
        s.pos -= s.last_rows - keep;
        s.last_rows = keep;
        Ok(())
    }

    /// Every slot's convolution states as after the first `keep` rows of the last decode pass
    fn conv_keep(&self, s: &Session, keep: usize) -> Result<()> {
        let groups = self.g.conv_groups();
        for (i, sl) in s.slots.iter().enumerate() {
            let d = self.dev_of(self.g.phys(i));
            let mut off = 0;
            for &ch in &groups {
                let snap = sl.snap.view((3 + MAXV) * off * 4, (3 + MAXV) * ch * 4)?;
                let st = sl.conv.view(3 * off * 4, 3 * ch * 4)?;
                // SAFETY: snap [3 + rows, ch], the state [3, ch].
                self.op(d, "conv keep", || unsafe { (self.k.conv_keep)(d.raw(), snap.fp(), st.fp(), keep as i64, ch as i64) })?;
                off += ch;
            }
        }
        Ok(())
    }

    pub fn forward_batch(&self, sessions: &mut [&mut Session], tokens: &[u32], tap: Tap) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(tokens.len());
        for (s, t) in sessions.iter_mut().zip(tokens) {
            out.push(self.pass(s, &[*t], 1, &mut *tap)?.pop().unwrap_or_default());
        }
        Ok(out)
    }

    /// The next committed token
    pub fn step(&self, s: &mut Session, d: &mut Decoder, smp: &mut dyn Sampler, tap: Tap) -> Result<Vec<u32>> {
        let _w = self.work.lock().unwrap();
        let drafting = d.draft && self.drafts > 0 && s.ds.is_some();
        // the last pass's row whose features the drafts continue from
        let mut last_row = s.last_rows.saturating_sub(1);
        let out = match (d.logits.take(), d.next.take()) {
            (Some(l), _) => vec![smp.sample(&l)],
            (None, Some(x)) => {
                let mut rows = vec![x];
                if drafting {
                    rows.extend(d.drafts.iter().take(s.max_ctx.saturating_sub(s.pos + 1)));
                }
                d.drafts.clear();
                // the tokens from the GPU when nothing reads the logits (greedy: the argmaxes; sampled: the draws)
                let pick = if smp.greedy() { Some(None) } else { smp.device().map(Some) };
                let picks: Vec<u32> = match self.pass_out(s, &rows, rows.len(), pick, tap)? {
                    Out::Logits(l) => l.iter().map(|r| smp.sample(r)).collect(),
                    Out::Ids(v) => v,
                };
                let mut out = Vec::with_capacity(rows.len());
                for (i, &y) in picks.iter().enumerate() {
                    out.push(y);
                    if i + 1 < rows.len() {
                        d.drafted += 1;
                        if y == rows[i + 1] {
                            d.accepted += 1;
                            continue;
                        }
                    }
                    break;
                }
                // the rejected rows undone (their caches written again later, the convolutions' states as after the kept)
                if out.len() < rows.len() {
                    self.rollback(s, out.len())?;
                }
                last_row = out.len() - 1;
                out
            }
            (None, None) => return Err(e("a decoder with nothing to draw from")),
        };
        let tail = *out.last().unwrap_or(&0);
        d.next = Some(tail);
        if drafting {
            d.drafts = self.ds_draft(s, tail, last_row, self.drafts)?;
        }
        Ok(out)
    }

    pub fn save(&self, s: &Session) -> Result<Checkpoint> {
        let g = &self.g;
        let row = g.head * 2;
        let mut bufs = Vec::new();
        for (sl, st) in s.slots.iter().enumerate() {
            let from = Self::kept(g.win(sl), s.pos);
            let n = s.pos - from;
            for c in [&st.kc, &st.vc] {
                let mut b = vec![0u8; g.heads_kv * n * row];
                for h in 0..g.heads_kv {
                    let mut p = from;
                    while p < s.pos {
                        let si = p % st.cap;
                        let run = (s.pos - p).min(st.cap - si);
                        let at = (h * n + p - from) * row;
                        c.read((h * st.cap + si) * row, &mut b[at..at + run * row])?;
                        p += run;
                    }
                }
                bufs.push(b);
            }
            let mut cb = vec![0u8; st.conv.len];
            st.conv.read(0, &mut cb)?;
            bufs.push(cb);
        }
        if let Some(ds) = &s.ds {
            bufs.extend(self.ds_save(ds, s.pos, s.last_rows)?);
        }
        let bytes = bufs.iter().map(|b| b.len()).sum();
        Ok(Checkpoint { pos: s.pos, bufs, bytes })
    }

    pub fn restore(&self, s: &mut Session, ck: &Checkpoint) -> Result<()> {
        let g = &self.g;
        let row = g.head * 2;
        if ck.pos > s.max_ctx {
            return Err(e(format!("a checkpoint of {} tokens into a session of {}", ck.pos, s.max_ctx)));
        }
        let ds_bufs = self.ds.as_ref().map_or(0, |d| d.save_bufs());
        if ck.bufs.len() != 3 * s.slots.len() + if s.ds.is_some() { ds_bufs } else { 0 } {
            return Err(e("a checkpoint of another shape"));
        }
        let mut it = ck.bufs.iter();
        for (sl, st) in s.slots.iter().enumerate() {
            let from = Self::kept(g.win(sl), ck.pos);
            let n = ck.pos - from;
            for c in [&st.kc, &st.vc] {
                let b = it.next().ok_or_else(|| e("a short checkpoint"))?;
                if b.len() != g.heads_kv * n * row {
                    return Err(e("a checkpoint of another shape"));
                }
                for h in 0..g.heads_kv {
                    let mut p = from;
                    while p < ck.pos {
                        let si = p % st.cap;
                        let run = (ck.pos - p).min(st.cap - si);
                        let at = (h * n + p - from) * row;
                        c.write((h * st.cap + si) * row, &b[at..at + run * row])?;
                        p += run;
                    }
                }
            }
            let cb = it.next().ok_or_else(|| e("a short checkpoint"))?;
            if cb.len() != st.conv.len {
                return Err(e("a checkpoint of another shape"));
            }
            st.conv.write(0, cb)?;
        }
        if let Some(ds) = s.ds.as_mut() {
            let rest: Vec<&Vec<u8>> = it.collect();
            self.ds_restore(ds, ck.pos, &rest)?;
        }
        s.pos = ck.pos;
        s.last_rows = 0;
        Ok(())
    }

    pub fn gpu_info(&self) -> Vec<GpuInfo> {
        self.devs.iter().enumerate().map(|(di, d)| {
            let (total, free) = d.gpu.memory().unwrap_or((0, None));
            // its layers: the loop's on the loop's GPU, else all of them
            let mine: Vec<usize> = (0..self.g.layers).filter(|l| self.layers[*l].dev == di).collect();
            let layers = (*mine.first().unwrap_or(&0) as u64, mine.last().map_or(0, |l| *l as u64 + 1));
            GpuInfo { index: d.gpu.index, name: d.gpu.name.clone(), pci: d.gpu.pci.clone(), total, free, layers, expert_slots: 0, host_slots: 0 }
        }).collect()
    }
}

/// A tensor read onto the GPU in its stored type: [rows (outputs), cols (inputs)] (a vector one row; an expert stack
/// [experts x rows, cols]); a Q8_0 or NVFP4 matrix repacked by the silo when there is one
pub fn mat(f: &Gguf, gpu: &Arc<Gpu>, silo: Option<&ffi::Silo>, name: &str, loaded: &std::cell::Cell<u64>) -> Result<Mat> {
    let t = f.tensor(name).ok_or_else(|| e(format!("no {name}")))?;
    let (rows, cols) = match t.shape.len() {
        1 => (1, t.shape[0] as usize),
        _ => (t.shape[..t.shape.len() - 1].iter().product::<u64>() as usize, t.shape[t.shape.len() - 1] as usize),
    };
    let ok = matches!(t.ty, GType::F32 | GType::BF16 | GType::Q8_0 | GType::NVFP4);
    if !ok {
        return Err(e(format!("{name}: {} - this engine reads F32, BF16, Q8_0 and NVFP4", t.ty.name())));
    }
    if t.shape.len() == 1 && t.ty != GType::F32 {
        return Err(e(format!("{name}: a {} vector (float32 expected)", t.ty.name())));
    }
    let raw = f.read(t).map_err(|x| e(x.0))?;
    loaded.set(loaded.get() + raw.len() as u64);
    let buf = DevBuf::new(gpu, raw.len())?;
    buf.write(0, &raw)?;
    let code = t.ty.code() as c_int;
    // SAFETY: arithmetic only.
    if let Some(s) = silo.filter(|s| t.shape.len() > 1 && unsafe { (s.supported)(code, cols as i64) } != 0) {
        // SAFETY: arithmetic only; then the blocks repacked into their own buffer (the raw one freed after the sync).
        let packed = DevBuf::new(gpu, unsafe { (s.packed_bytes)(code, rows as i64, cols as i64) })?;
        ffi::check(unsafe { (s.pack)(gpu.raw(), code, buf.ptr(), rows as i64, cols as i64, packed.ptr()) }, name)?;
        gpu.sync()?;
        return Ok(Mat { buf: packed, ty: t.ty, rows, cols, packed: true });
    }
    Ok(Mat { buf, ty: t.ty, rows, cols, packed: false })
}
