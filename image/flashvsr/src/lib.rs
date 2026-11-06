//! FlashVSR v1.1 (OpenImagingLab; Apache-2.0) in Rust and CUDA: one-step diffusion video super-resolution, a whole
//! clip as one stream. FlashVSR's tiny pipeline (diffsynth/pipelines/flashvsr_tiny.py) without Python:
//!
//! ```text
//!   frames -> Pillow bicubic x scale -> mirrored out to multiples of 128 ([-1, 1])
//!   chunk 0: frames 0-24 -> LQ projection (6 latent frames) + noise -> DiT (one step, its KV cache filled) -> latents
//!            -> tiny decoder with frames 0-20 as its condition -> frames 0-20
//!   chunk k: frames 8k+17 .. 8k+24 -> 2 latent frames -> DiT (attending to the cached 3 pairs) -> frames 8k+13 .. 8k+20
//!   each frame: the colour fix (AdaIN per channel to the upscaled input frame, the reference's), cut back to the
//!   picture, bytes
//! ```
//!
//! The stream has no length limit: the KV cache rolls (the last 3 latent pairs), the convolutions keep their last
//! frames, and the rotary positions are relative. The clip's last frame is repeated past its end (4 to 11 times, as
//! the reference pads) so every frame comes out. The DiT's layers can be split over cards (`NC_FLASHVSR_SPLIT`): at
//! 1792 x 1024 its KV cache is 7.9 GB, more than a 12 GB card holds beside the weights.
//!
//! Source: `dit.rs` the DiT and its block-sparse attention, `lq.rs` the LQ projection, `tcd.rs` the decoder,
//! `resize.rs` the frames' preparation, `check.rs` the check against reference/flashvsr/ref.py.

pub mod check;
pub mod dit;
pub mod ffi;
pub mod lq;
pub mod resize;
pub mod tcd;

use std::collections::VecDeque;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use nextcuda_core::{At, DevBuf, EngineOption, Gpu};
use nextcuda_diffusion::kernels::Dt;
use nextcuda_gguf::safetensors::SafeTensors;
use nextcuda_image::{ClipUpscaleRequest, Defaults, Error, ImageEngine, ImageKind, ImageRequest, LoadOptions, ModelFiles, Picture, Result, Sampler, Schedule,
    Step, UpscaleInfo, UpscaleRequest, UpscaleVariant};

use crate::dit::{Chunk, Ends, Stage, DIM};
use crate::ffi::{check as ck, g};
use crate::lq::LqProj;
use crate::resize::Geometry;
use crate::tcd::Tcd;

pub const ARCH: &str = "flashvsr";

pub const OPTIONS: &[EngineOption] = &[
    EngineOption { name: "split", env: "NC_FLASHVSR_SPLIT", value: "N,M", help: "the DiT's 30 blocks over the model's cards, in order (default: 17,13 on two, 30 on one)", at: At::Load },
    EngineOption { name: "decoder", env: "NC_FLASHVSR_DECODER", value: "first|last", help: "the card the decoder runs on (default first: the decoder is the heaviest stage, the first card the fastest)", at: At::Load },
    EngineOption { name: "sparse_ratio", env: "", value: "R", help: "the attention's top-k as FlashVSR's sparse_ratio (default 1.5: 8% faster than 2.0, as steady and as sharp on the platform's films)", at: At::Request },
    EngineOption { name: "local_range", env: "", value: "N", help: "the attention's local window in 128-pixel blocks (default 11: steadier; 9 sharper)", at: At::Request },
    EngineOption { name: "color_fix", env: "", value: "0|1", help: "match each frame's colour to the upscaled input frame (AdaIN, default 1)", at: At::Request },
];

pub fn kind() -> ImageKind {
    ImageKind { archs: &[ARCH], name: "FlashVSR v1.1 video super-resolution (Wan 2.1 1.3B, one step, streaming; Rust + CUDA)", roles: &["dit", "lq-proj", "decoder", "prompt"], load,
                options: OPTIONS }
}

fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn ImageEngine>> {
    Ok(Box::new(FlashVsr::load(files, gpus, o.setting("NC_FLASHVSR_SPLIT").as_deref(), o.setting("NC_FLASHVSR_DECODER").as_deref(), log)?))
}

const VARIANTS: &[UpscaleVariant] = &[UpscaleVariant {
    id: "flashvsr-v1.1",
    about: "real footage, faces and skin: one-step diffusion over the whole clip (temporally steady), 2x to 4x",
    scales: &[2, 3, 4],
    noise: &[],
}];

/// What a run takes
#[derive(Clone, Copy, Debug)]
pub struct Settings {
    pub scale: f64,
    pub seed: u64,
    pub sparse_ratio: f64,
    pub local_range: usize,
    pub color_fix: bool,
    /// the latent frame the stream's first one is, for the noise (a tile's or a segment's stream inside a clip)
    pub lat_base: usize,
}

impl Default for Settings {
    fn default() -> Self {
        Settings { scale: 2.0, seed: 0, sparse_ratio: 1.5, local_range: 11, color_fix: true, lat_base: 0 }
    }
}

/// A check's view into a run (check.rs): the noise to use, the tensors to compare
pub trait Hooks {
    /// latent frame t's noise [16, h8, w8] (None: the engine's own)
    fn noise(&mut self, _t: usize, _h8: usize, _w8: usize) -> Option<Vec<f32>> {
        None
    }
    /// chunk `c`'s tensor `name` (lq, b0, b15, b29, lat), on its card
    fn tensor(&mut self, _c: usize, _name: &str, _v: &DevBuf, _rows: usize) -> Result<()> {
        Ok(())
    }
    /// frame i as the decoder made it (half [th, tw, 3] in about [0, 1], before the colour fix)
    fn decoded(&mut self, _i: usize, _v: &DevBuf) -> Result<()> {
        Ok(())
    }
}

/// A frame read: prepared, or being prepared on its own thread
enum Slot {
    Ready(Vec<u8>),
    Pending(std::thread::JoinHandle<Vec<u8>>),
}

impl Slot {
    fn ready(&mut self) -> Result<&Vec<u8>> {
        if let Slot::Pending(_) = self {
            let Slot::Pending(h) = std::mem::replace(self, Slot::Ready(Vec::new())) else { unreachable!() };
            *self = Slot::Ready(h.join().map_err(|_| Error("flashvsr: a frame's preparation failed".into()))?);
        }
        match self {
            Slot::Ready(v) => Ok(v),
            Slot::Pending(_) => unreachable!(),
        }
    }
}

struct NoHooks;
impl Hooks for NoHooks {}

/// Where a run's time goes (NC_FLASHVSR_PROFILE=1: the cards waited for at each mark, the totals logged at the end)
struct Prof {
    on: bool,
    t: Instant,
    acc: Vec<(&'static str, f64)>,
}

/// A chunk for the last cards' blocks and the head
struct Job {
    chunk: usize,
    f: usize,
    lat0: usize,
    rows: usize,
    c: Chunk,
    cs: Vec<f32>,
    noise: Option<Vec<Vec<f32>>>,
}

/// What the second card's thread tells the first: x has been taken, chunk n's latents are made
enum Msg {
    Copied,
    Lat(usize),
}

/// The cards after the first: their blocks, the head and the step
struct Back<'a> {
    ends: &'a Ends,
    stages: &'a mut [Stage],
    xs: &'a [DevBuf],
    css: &'a [DevBuf],
    x0: &'a DevBuf,
    noise: &'a DevBuf,
    hbuf: &'a DevBuf,
    pred: &'a DevBuf,
    lat: &'a DevBuf,
    /// with one card, the head runs there: its kernel context and the card
    one: (&'a nextcuda_diffusion::kernels::Nsd, Arc<Gpu>),
    seed: u64,
    lat_base: usize,
    h8: usize,
    w8: usize,
}

impl Back<'_> {
    /// x from the first card into the second's buffer
    fn copy_in(&self, job: &Job, staging: &mut [u8]) -> Result<()> {
        if let Some(x1) = self.xs.first() {
            let src = self.x0.view(0, job.rows * DIM * 4)?;
            x1.copy_from_peer(&src, staging)?;
        }
        Ok(())
    }

    /// the blocks after the first card's, the head and the step: latents = noise - output, in `lat`
    fn run(&mut self, job: &Job, grab: &mut dyn FnMut(usize, &str, &DevBuf, usize) -> Result<()>) -> Result<()> {
        let k = ffi::api()?;
        let (f, rows, h8, w8) = (job.f, job.rows, self.h8, self.w8);
        for (i, st) in self.stages.iter_mut().enumerate() {
            if i > 0 {
                let mut staging = vec![0u8; rows * DIM * 4];
                let src = self.xs[i - 1].view(0, rows * DIM * 4)?;
                self.xs[i].copy_from_peer(&src, &mut staging)?;
            }
            // SAFETY: plain floats as bytes
            self.css[i].write(0, unsafe { std::slice::from_raw_parts(job.cs.as_ptr().cast::<u8>(), job.cs.len() * 4) })?;
            st.run(&self.xs[i], &self.css[i], &job.c, &mut |layer, x| {
                if [0, 15, 29].contains(&layer) {
                    grab(job.chunk, &format!("b{layer}"), x, rows)?;
                }
                Ok(())
            })?;
        }
        let (x, nsd, gpu) = match self.stages.last() {
            Some(st) => (self.xs.last().unwrap(), &st.nsd, st.gpu.clone()),
            None => (self.x0, self.one.0, self.one.1.clone()),
        };
        write_noise(self.noise, &gpu, job.noise.as_deref(), f, job.lat0 + self.lat_base, h8, w8, self.seed)?;
        let gl = g(&gpu);
        // SAFETY: the buffers hold rows x 1536 / 64 values and 16 x f x h8 x w8 latents
        ck(unsafe { (k.ln)(gl, x.fp(), rows as i64, DIM as i64, 1e-6, self.ends.head_sc.fp(), self.ends.head_sh.fp(), self.hbuf.ptr()) }, "head norm")?;
        self.ends.head.apply(nsd, self.hbuf.ptr(), rows, self.pred.ptr(), Dt::F32)?;
        ck(unsafe { (k.unpatchify_sub)(gl, self.pred.fp(), self.noise.fp(), f as i64, h8 as i64, w8 as i64, self.lat.fp()) }, "step")?;
        grab(job.chunk, "lat", self.lat, 16 * f * h8 * w8)?;
        nsd.wait()
    }
}

/// A chunk's noise [16, f, h8, w8] into buf: given (a check's, per latent frame [16, h8, w8]) or the engine's own
#[allow(clippy::too_many_arguments)]
fn write_noise(buf: &DevBuf, gpu: &Arc<Gpu>, given: Option<&[Vec<f32>]>, f: usize, lat0: usize, h8: usize, w8: usize, seed: u64) -> Result<()> {
    let k = ffi::api()?;
    for t in 0..f {
        for c in 0..16 {
            match given {
                Some(v) => {
                    // SAFETY: plain floats as bytes
                    let b = unsafe { std::slice::from_raw_parts(v[t][c * h8 * w8..].as_ptr().cast::<u8>(), h8 * w8 * 4) };
                    buf.write(((c * f + t) * h8 * w8) * 4, b)?;
                }
                None => {
                    // SAFETY: plane (c, t) of buf
                    let p = unsafe { buf.fp().add((c * f + t) * h8 * w8) };
                    ck(unsafe { (k.randn)(g(gpu), p, (h8 * w8) as i64, seed, (((lat0 + t) * 16 + c) * h8 * w8) as u64) }, "noise")?;
                }
            }
        }
    }
    Ok(())
}

impl Prof {
    /// `secs` that a mark's span included, booked under `name` instead (taken from `from`'s next mark)
    fn carve(&mut self, name: &'static str, secs: f64) {
        if !self.on {
            return;
        }
        match self.acc.iter_mut().find(|(n, _)| *n == name) {
            Some(e) => e.1 += secs,
            None => self.acc.push((name, secs)),
        }
        self.t += std::time::Duration::from_secs_f64(secs);
    }

    fn mark(&mut self, name: &'static str, gpus: &[Arc<Gpu>]) {
        if !self.on {
            return;
        }
        for g in gpus {
            let _ = g.sync();
        }
        let s = self.t.elapsed().as_secs_f64();
        self.t = Instant::now();
        match self.acc.iter_mut().find(|(n, _)| *n == name) {
            Some(e) => e.1 += s,
            None => self.acc.push((name, s)),
        }
    }
}

struct Inner {
    stages: Vec<Stage>,
    lq: LqProj,
    tcd: Tcd,
    /// the decoder's card (an index into the stages' cards) and its kernel context
    dec: usize,
    dec_nsd: nextcuda_diffusion::kernels::Nsd,
}

pub struct FlashVsr {
    gpus: Vec<Arc<Gpu>>,
    ends: Ends,
    inner: Mutex<Inner>,
    load_secs: f64,
    pub weight_bytes: Vec<usize>,
}

/// The first card (index) whose free bytes are below what the run needs there, with its free bytes; unknown free memory
/// and already prepared stages are never short
fn short_of(need: &[usize], free: &[Option<u64>], prepared: &[bool]) -> Option<(usize, u64)> {
    need.iter().enumerate().find_map(|(i, n)| match (free.get(i).copied().flatten(), prepared.get(i).copied().unwrap_or(false)) {
        (Some(f), false) if (*n as u64) > f => Some((i, f)),
        _ => None,
    })
}

fn pth_table(path: &Path) -> Result<SafeTensors> {
    nextcuda_gguf::pthst::open(path).map_err(|e| Error(e.0))
}

impl FlashVsr {
    pub fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], split: Option<&str>, decoder: Option<&str>, log: &mut dyn FnMut(String)) -> Result<FlashVsr> {
        let t0 = Instant::now();
        if gpus.is_empty() {
            return Err(Error("flashvsr needs a GPU".into()));
        }
        ffi::api()?;
        let path = |role: &str| files.get(role).cloned().ok_or_else(|| Error(format!("flashvsr: no {role} file (the card names dit, lq-proj, decoder, prompt)")));
        let st = SafeTensors::open(&path("dit")?).map_err(|e| Error(e.0))?;
        let counts: Vec<usize> = match split {
            Some(s) if !s.trim().is_empty() => s.split(',').map(|v| v.trim().parse::<usize>().map_err(|_| Error(format!("NC_FLASHVSR_SPLIT {s}: block counts")))).collect::<Result<_>>()?,
            // the first card (the 5070) also holds the LQ projection and the decoder: fewer blocks there
            // measured best on the 5070 + 3060 at 1728 x 960 (0.260 s a frame; 16,14: 0.276; 12,18: 0.339)
            _ if gpus.len() >= 2 => vec![17, 13],
            _ => vec![dit::LAYERS],
        };
        if counts.iter().sum::<usize>() != dit::LAYERS || counts.len() > gpus.len() || counts.contains(&0) {
            return Err(Error(format!("NC_FLASHVSR_SPLIT: {counts:?} is not 30 blocks over at most {} cards", gpus.len())));
        }
        let gpus: Vec<Arc<Gpu>> = gpus[..counts.len()].to_vec();
        let (t, tmod) = dit::time_embedding(&st)?;
        let prompt = nextcuda_gguf::pth::Pth::open(&path("prompt")?).map_err(|e| Error(e.0))?;
        let name = prompt.tensors.keys().next().cloned().ok_or_else(|| Error("flashvsr: the prompt file holds no tensor".into()))?;
        let ctx = prompt.f32(&name).map_err(|e| Error(e.0))?;
        let ctx = dit::text_embedding(&st, &gpus[0], &ctx)?;
        let mut stages = Vec::new();
        let mut from = 0;
        for (i, n) in counts.iter().enumerate() {
            stages.push(Stage::load(&st, &gpus[i], from, from + n, &tmod, &ctx, log)?);
            from += n;
        }
        let last = gpus.last().unwrap().clone();
        let ends = Ends::load(&st, &gpus[0], &last, &t)?;
        let lq = LqProj::load(&pth_table(&path("lq-proj")?)?, &gpus[0])?;
        let dec = match decoder.unwrap_or("first") {
            "first" | "" => 0,
            "last" => gpus.len() - 1,
            d => return Err(Error(format!("NC_FLASHVSR_DECODER {d}: first or last"))),
        };
        let tcd = Tcd::load(&pth_table(&path("decoder")?)?, &gpus[dec])?;
        let dec_nsd = nextcuda_diffusion::kernels::Nsd::new(&gpus[dec])?;
        let mut weight_bytes: Vec<usize> = stages.iter().map(|s| s.weight_bytes).collect();
        weight_bytes[0] += lq.bytes;
        weight_bytes[dec] += tcd.bytes;
        log(format!("flashvsr: loaded in {:.1} s ({})", t0.elapsed().as_secs_f64(),
                    weight_bytes.iter().zip(&gpus).map(|(b, g)| format!("{:.2} GiB on {}", *b as f64 / (1u64 << 30) as f64, g.name)).collect::<Vec<_>>().join(", ")));
        let _ = last;
        Ok(FlashVsr { gpus, ends, inner: Mutex::new(Inner { stages, lq, tcd, dec, dec_nsd }), load_secs: t0.elapsed().as_secs_f64(), weight_bytes })
    }

    /// The first stage whose card has less free than `need` for a grid of nb blocks (index, free bytes); a stage
    /// already prepared for the grid holds its buffers and is not short
    fn first_short(inner: &Inner, need: &[usize], nb: usize) -> Option<(usize, u64)> {
        let free: Vec<Option<u64>> = inner.stages.iter().map(|s| s.gpu.memory().ok().and_then(|m| m.1)).collect();
        let prepared: Vec<bool> = inner.stages.iter().map(|s| s.is_prepared(nb)).collect();
        short_of(need, &free, &prepared)
    }

    /// Frees the per-size buffers on every card (KV caches, workspaces, the projection's and decoder's stream state); the
    /// weights stay. The next run makes the ones it needs.
    fn release_buffers(inner: &mut Inner) {
        for s in &mut inner.stages {
            s.release();
        }
        inner.lq.release();
        inner.tcd.release();
    }

    /// The settings of a request (its variant, scale and options)
    pub fn settings(variant: Option<&str>, scale: Option<u32>, seed: Option<u64>, extra: &nextcuda_image::Given) -> Result<Settings> {
        if let Some(v) = variant.filter(|v| !v.is_empty() && *v != "flashvsr" && *v != VARIANTS[0].id) {
            return Err(Error(format!("{v}: flashvsr has {}", VARIANTS[0].id)));
        }
        let scale = scale.unwrap_or(2);
        if !(2..=4).contains(&scale) {
            return Err(Error(format!("flashvsr makes 2x to 4x, not {scale}x")));
        }
        let mut s = Settings { scale: scale as f64, seed: seed.unwrap_or(0), ..Settings::default() };
        if let Some(v) = extra.get("sparse_ratio") {
            s.sparse_ratio = v.parse().ok().filter(|x: &f64| (0.25..=8.0).contains(x)).ok_or_else(|| Error(format!("sparse_ratio {v}: 0.25 to 8")))?;
        }
        if let Some(v) = extra.get("local_range") {
            s.local_range = v.parse().ok().filter(|x: &usize| (3..=31).contains(x) && x % 2 == 1).ok_or_else(|| Error(format!("local_range {v}: odd, 3 to 31")))?;
        }
        if let Some(v) = extra.get("color_fix") {
            s.color_fix = !matches!(v.as_str(), "0" | "false" | "no" | "off");
        }
        Ok(s)
    }

    /// The memory a frame size takes on each card beyond the weights (bytes)
    pub fn need_bytes(&self, geo: &Geometry) -> Vec<usize> {
        let inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        let (hp, wp) = (geo.th / 16, geo.tw / 16);
        let nb = (hp / 8) * (wp / 8);
        let mut v: Vec<usize> = inner.stages.iter().map(|s| s.need_bytes(nb)).collect();
        v[0] += LqProj::need_bytes(hp, wp);
        v[inner.dec] += Tcd::need_bytes(geo.th / 8, geo.tw / 8);
        v
    }

    /// A clip through the stream: frames from `next` (RGB [h, w, 3], or already prepared [th, tw, 3] when `prepared`),
    /// each made frame (the picture's part, [sh, sw, 3]) to `emit`
    #[allow(clippy::too_many_arguments)]
    pub fn run(&self, geo: Geometry, set: &Settings, prepared: bool, total_hint: Option<u32>, next: &mut dyn FnMut() -> Result<Option<Vec<u8>>>,
               emit: &mut dyn FnMut(Picture) -> Result<()>, progress: &mut dyn FnMut(Step) -> Result<()>, hooks: &mut dyn Hooks) -> Result<u32> {
        let k = ffi::api()?;
        let t0 = Instant::now();
        let mut guard = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        let inner = &mut *guard;
        let (th, tw) = (geo.th, geo.tw);
        let (h8, w8, hp, wp) = (th / 8, tw / 8, th / 16, tw / 16);
        let (nbh, nbw) = (hp / 8, wp / 8);
        let nb = nbh * nbw;
        // the memory first: refused here, before anything is made, when a card cannot hold this size
        let need = {
            let mut v: Vec<usize> = inner.stages.iter().map(|s| s.need_bytes(nb)).collect();
            v[0] += LqProj::need_bytes(hp, wp);
            v[inner.dec] += Tcd::need_bytes(h8, w8);
            v
        };
        // short: first drop this engine's own cached buffers (every size's cache, workspaces and stream state; the weights stay),
        // then measure again - only a run that still does not fit is refused
        if Self::first_short(inner, &need, nb).is_some() {
            Self::release_buffers(inner);
        }
        if let Some((i, free)) = Self::first_short(inner, &need, nb) {
            let (s, n) = (&inner.stages[i], need[i] as u64);
            let short_mb = ((n - free) >> 20) + 1;
            return Err(nextcuda_image::no_room_error(s.gpu.index, short_mb, &format!(
                "flashvsr: {} x {} needs {:.1} GiB more on {} and it has {:.1} free (a smaller size, or NC_FLASHVSR_SPLIT over more cards)",
                geo.sw, geo.sh, n as f64 / (1u64 << 30) as f64, s.gpu.name, free as f64 / (1u64 << 30) as f64)));
        }
        for s in inner.stages.iter_mut() {
            s.prepare(nb)?;
        }
        inner.lq.reset(hp, wp)?;
        inner.tcd.reset(h8, w8)?;
        let ratio = set.sparse_ratio * 768.0 * 1280.0 / (th * tw) as f64;
        let topk = ((nb * nb) as f64 * ratio) as i64 - 1;
        let first_gpu = self.gpus[0].clone();
        let last_gpu = self.gpus.last().unwrap().clone();
        let dec_gpu = self.gpus[inner.dec].clone();
        let fsize = th * tw * 3;
        // the frames on the host: [base, base + len) of the clip, each prepared on a thread of its own as it is read (a chunk
        // ahead, so the bicubic runs beside the cards' work); the clip's count once it has ended
        let mut frames: VecDeque<Slot> = VecDeque::new();
        let mut base = 0usize;
        let mut ended: Option<usize> = None;
        let mut last: Vec<u8> = Vec::new();
        let prep_secs = std::cell::Cell::new(0f64);
        let mut read = |want: usize, frames: &mut VecDeque<Slot>, base: usize, ended: &mut Option<usize>, _last: &mut Vec<u8>| -> Result<()> {
            while ended.is_none() && base + frames.len() < want {
                match next()? {
                    Some(f) if prepared => frames.push_back(Slot::Ready(f)),
                    Some(f) => frames.push_back(Slot::Pending(std::thread::spawn(move || resize::prepare(&f, &geo)))),
                    None => *ended = Some(base + frames.len()),
                }
            }
            Ok(())
        };
        // frame i prepared (waited for); past the end, the clip's last
        let get = |i: usize, frames: &mut VecDeque<Slot>, base: usize, last: &mut Vec<u8>| -> Result<Vec<u8>> {
            let at = if i >= base && i - base < frames.len() { i - base } else if !frames.is_empty() { frames.len() - 1 } else { return Ok(last.clone()) };
            let tp = Instant::now();
            let is_last = at == frames.len() - 1;
            let p = frames[at].ready()?;
            prep_secs.set(prep_secs.get() + tp.elapsed().as_secs_f64());
            if p.len() != fsize {
                return Err(Error(format!("flashvsr: a frame of {} bytes, {} expected", p.len(), fsize)));
            }
            if is_last {
                last.clone_from(p);
            }
            Ok(p.clone())
        };
        // buffers: up to 25 frames of bytes and halves, the LQ tokens, noise, x and the latents
        let frames_u8 = DevBuf::new(&first_gpu, 4 * fsize)?;
        let frames_h = DevBuf::new(&first_gpu, 4 * fsize * 2)?;
        let lqtok = DevBuf::new(&first_gpu, 6 * hp * wp * DIM * 4)?;
        let rows0 = 3 * nb * 128;
        let noise0 = DevBuf::new(&first_gpu, 16 * 6 * h8 * w8 * 4)?;
        let patches = DevBuf::new(&first_gpu, rows0 * 64 * 2)?;
        let mut xs: Vec<DevBuf> = Vec::new();
        for s in &inner.stages {
            xs.push(DevBuf::new(&s.gpu, rows0 * DIM * 4)?);
        }
        let mut css: Vec<DevBuf> = Vec::new();
        for s in &inner.stages {
            css.push(DevBuf::new(&s.gpu, rows0 * 128 * 4)?);
        }
        let noise_l = DevBuf::new(&last_gpu, 16 * 6 * h8 * w8 * 4)?;
        let hbuf = DevBuf::new(&last_gpu, rows0 * DIM * 2)?;
        let pred = DevBuf::new(&last_gpu, rows0 * 64 * 4)?;
        let lat = DevBuf::new(&last_gpu, 16 * 6 * h8 * w8 * 4)?;
        let lat_d = if inner.dec == self.gpus.len() - 1 { lat.view(0, lat.len)? } else { DevBuf::new(&dec_gpu, lat.len)? };
        let mut lat_staging = vec![0u8; lat.len];
        let cond_u8 = DevBuf::new(&dec_gpu, 24 * fsize)?;
        let cond = DevBuf::new(&dec_gpu, 24 * fsize * 2)?;
        let sums = DevBuf::new(&dec_gpu, 12 * 8)?;
        let ms = DevBuf::new(&dec_gpu, 12 * 4)?;
        let out_u8 = DevBuf::new(&dec_gpu, geo.sh * geo.sw * 3)?;
        let mut staging = vec![0u8; if inner.stages.len() > 1 { rows0 * DIM * 4 } else { 0 }];
        let ns = inner.stages.len();
        let dec = inner.dec;
        let mut prof = Prof { on: std::env::var("NC_FLASHVSR_PROFILE").is_ok_and(|v| v == "1"), t: Instant::now(), acc: Vec::new() };
        // the cards overlap when the decoder is not on the last card: the last card runs chunk k's blocks while the
        // first decodes chunk k - 1 and starts chunk k + 1 (not when profiling: the marks wait for both cards)
        let pipelined = ns > 1 && dec != ns - 1 && !prof.on && std::env::var("NC_FLASHVSR_PIPELINE").map_or(true, |v| v != "0");
        let g0 = g(&first_gpu);
        let gd = g(&dec_gpu);
        let (front_stage, back_stages) = inner.stages.split_at_mut(1);
        let stage0 = &mut front_stage[0];
        let (lq, tcd, dec_nsd) = (&mut inner.lq, &mut inner.tcd, &inner.dec_nsd);
        let (x0, xs_b) = xs.split_first().unwrap();
        let (cs0, css_b) = css.split_first().unwrap();
        let back = Back { ends: &self.ends, stages: back_stages, xs: xs_b, css: css_b, x0, noise: &noise_l, hbuf: &hbuf, pred: &pred, lat: &lat, one: (dec_nsd, first_gpu.clone()), seed: set.seed, lat_base: set.lat_base, h8, w8 };
        let mut made = 0usize;
        std::thread::scope(|sc| -> Result<()> {
            let (jtx, jrx) = std::sync::mpsc::channel::<Job>();
            let (rtx, rrx) = std::sync::mpsc::channel::<std::result::Result<Msg, String>>();
            let (atx, arx) = std::sync::mpsc::channel::<()>();
            let mut back_inline = None;
            if pipelined {
                let mut back = back;
                let mut staging = std::mem::take(&mut staging);
                sc.spawn(move || {
                    let mut owed = false;
                    while let Ok(job) = jrx.recv() {
                        let r = (|| -> Result<()> {
                            back.copy_in(&job, &mut staging)?;
                            rtx.send(Ok(Msg::Copied)).map_err(|_| Error("stopped".into()))?;
                            // the latents' buffer is free once the first card has taken the last ones
                            if owed {
                                arx.recv().map_err(|_| Error("stopped".into()))?;
                            }
                            back.run(&job, &mut |_, _, _, _| Ok(()))?;
                            owed = true;
                            rtx.send(Ok(Msg::Lat(job.chunk))).map_err(|_| Error("stopped".into()))
                        })();
                        if let Err(e) = r {
                            let _ = rtx.send(Err(e.0));
                            return;
                        }
                    }
                });
            } else {
                back_inline = Some(back);
            }
            // the second card's messages, kept until asked for (its "latents made" can come before "x taken" of the next)
            let mut early: Vec<Msg> = Vec::new();
            let mut wait = |want: &dyn Fn(&Msg) -> bool| -> Result<()> {
                if let Some(i) = early.iter().position(|m| want(m)) {
                    early.remove(i);
                    return Ok(());
                }
                loop {
                    match rrx.recv() {
                        Ok(Ok(m)) if want(&m) => return Ok(()),
                        Ok(Ok(m)) => early.push(m),
                        Ok(Err(e)) => return Err(Error(e)),
                        Err(_) => return Err(Error("flashvsr: the second card's thread stopped".into())),
                    }
                }
            };
            let mut pending: Option<(usize, usize)> = None;
            let mut chunk = 0usize;
            loop {
                let hi = if chunk == 0 { 25 } else { 8 * chunk + 25 };
                prof.mark("other", &self.gpus);
                // a chunk ahead: the next chunk's frames prepare while this one runs
                read(hi + 8, &mut frames, base, &mut ended, &mut last)?;
                prof.mark("frames: ffmpeg decode wait", &self.gpus);
                let done = match ended {
                    Some(0) => return Err(Error("flashvsr: the clip has no frames".into())),
                    Some(n) => chunk + 2 >= (n + 3).div_ceil(8).max(3),
                    None => false,
                };
                if !done {
                    let (f, lat0, calls): (usize, usize, Vec<(usize, usize)>) = if chunk == 0 {
                        (6, 0, (0..7).map(|i| (if i == 0 { 0 } else { 4 * i - 3 }, 4 * i + 1)).collect())
                    } else {
                        (2, 4 + 2 * chunk, vec![(8 * chunk + 17, 8 * chunk + 21), (8 * chunk + 21, 8 * chunk + 25)])
                    };
                    let rows = f / 2 * nb * 128;
                    // SAFETY (the kernel calls below): every buffer holds what the call reads or writes, sized above.
                    // the LQ projection
                    let mut got = 0;
                    for (a, b) in calls {
                        let n = b - a;
                        let mut bytes = Vec::with_capacity(n * fsize);
                        for i in a..b {
                            bytes.extend_from_slice(&get(i, &mut frames, base, &mut last)?);
                        }
                        frames_u8.write(0, &bytes)?;
                        ck(unsafe { (k.u8_to_half)(g0, frames_u8.ptr() as *const u8, (n * fsize) as i64, frames_h.ptr()) }, "frames")?;
                        let out = lqtok.view(got * hp * wp * DIM * 4, hp * wp * DIM * 4)?;
                        if lq.stream(&stage0.nsd, &frames_h, n, &out)? {
                            got += 1;
                        }
                    }
                    if got != f {
                        return Err(Error(format!("flashvsr: the LQ projection made {got} latent frames, {f} expected")));
                    }
                    hooks.tensor(chunk, "lq", &lqtok, f * hp * wp)?;
                    prof.mark("lq projection", &self.gpus);
                    // the noise (a check's from its hooks, else the engine's own on each card)
                    let given: Option<Vec<Vec<f32>>> = (0..f).map(|t| hooks.noise(lat0 + t, h8, w8)).collect();
                    write_noise(&noise0, &first_gpu, given.as_deref(), f, lat0 + set.lat_base, h8, w8, set.seed)?;
                    // the tokens and the first card's blocks
                    ck(unsafe { (k.patchify)(g0, noise0.fp(), f as i64, h8 as i64, w8 as i64, patches.ptr()) }, "patches")?;
                    self.ends.patch.apply(&stage0.nsd, patches.ptr(), rows, x0.ptr(), Dt::F32)?;
                    ck(unsafe { (k.add_lq)(g0, x0.fp(), lqtok.fp(), f as i64, hp as i64, wp as i64, DIM as i64) }, "lq residual")?;
                    let cs = dit::rope_table(f / 2, hp, wp, lat0);
                    let c = Chunk { groups: f / 2, nbh, nbw, first: chunk == 0, evict: if chunk == 0 { 0 } else { (chunk - 1) % 3 }, topk, local: set.local_range };
                    // SAFETY: plain floats as bytes
                    cs0.write(0, unsafe { std::slice::from_raw_parts(cs.as_ptr().cast::<u8>(), cs.len() * 4) })?;
                    stage0.run(x0, cs0, &c, &mut |layer, x| {
                        if [0, 15, 29].contains(&layer) {
                            hooks.tensor(chunk, &format!("b{layer}"), x, rows)?;
                        }
                        Ok(())
                    })?;
                    prof.mark("dit stage 1", &self.gpus);
                    let job = Job { chunk, f, lat0, rows, c, cs, noise: given };
                    match back_inline.as_mut() {
                        Some(b) => {
                            if ns > 1 {
                                b.copy_in(&job, &mut staging)?;
                                prof.mark("x to the next card", &self.gpus);
                            }
                            b.run(&job, &mut |c, name, v, n| hooks.tensor(c, name, v, n))?;
                            prof.mark("dit stage 2 + head", &self.gpus);
                        }
                        None => {
                            jtx.send(job).map_err(|_| Error("flashvsr: the second card's thread stopped".into()))?;
                            wait(&|m| matches!(m, Msg::Copied))?;
                        }
                    }
                }
                // the chunk before (pipelined), or this one: its latents to the decoder, its frames out
                let finish = if pipelined { pending.take() } else if done { None } else { Some((chunk, if chunk == 0 { 6 } else { 2 })) };
                if let Some((pc, f)) = finish {
                    if pipelined {
                        wait(&|m| matches!(m, Msg::Lat(c) if *c == pc))?;
                    }
                    if dec != ns - 1 {
                        let src = lat.view(0, 16 * f * h8 * w8 * 4)?;
                        lat_d.copy_from_peer(&src, &mut lat_staging)?;
                    }
                    if pipelined {
                        let _ = atx.send(());
                    }
                    // the decoder's condition: the frames it makes (chunk 0: 0-20 with frame 0 three times in front)
                    let (pre, cur) = if pc == 0 { (0, 21) } else { (8 * pc + 13, 8 * pc + 21) };
                    let front = if pc == 0 { 3 } else { 0 };
                    let mut bytes = Vec::with_capacity((cur - pre + front) * fsize);
                    for _ in 0..front {
                        bytes.extend_from_slice(&get(0, &mut frames, base, &mut last)?);
                    }
                    for i in pre..cur {
                        bytes.extend_from_slice(&get(i, &mut frames, base, &mut last)?);
                    }
                    let nc = cur - pre + front;
                    cond_u8.write(0, &bytes)?;
                    ck(unsafe { (k.u8_to_half)(gd, cond_u8.ptr() as *const u8, (nc * fsize) as i64, cond.ptr()) }, "cond")?;
                    let mut idx = pre;
                    let limit = ended;
                    let real = |i: usize| limit.is_none_or(|n| i < n);
                    let mut emitted = Vec::new();
                    let mut colour_secs = 0f64;
                    let pon = prof.on;
                    for t in 0..f {
                        let cp = unsafe { (cond.ptr() as *mut u8).add(4 * t * fsize * 2) };
                        tcd.frame(dec_nsd, &lat_d, f, t, cp as *const _, &mut |dec| {
                            let i = idx;
                            idx += 1;
                            if !real(i) {
                                return Ok(());
                            }
                            hooks.decoded(i, dec)?;
                            if pon {
                                dec_gpu.sync()?;
                            }
                            let tcol = Instant::now();
                            let ci = i - pre + front;
                            let lqf = unsafe { (cond.ptr() as *mut u8).add(ci * fsize * 2) };
                            let msp = if set.color_fix {
                                sums.fill(0)?;
                                let s8 = sums.ptr() as *mut f64;
                                ck(unsafe { (k.stats)(gd, dec.ptr(), tw as i64, geo.left as i64, geo.top as i64, geo.sw as i64, geo.sh as i64, 2.0, -1.0, s8) },
                                   "stats")?;
                                ck(unsafe { (k.stats)(gd, lqf as *const _, tw as i64, geo.left as i64, geo.top as i64, geo.sw as i64, geo.sh as i64, 1.0, 0.0,
                                                      s8.add(6)) }, "stats")?;
                                ck(unsafe { (k.adain)(gd, s8, (geo.sw * geo.sh) as f64, ms.fp()) }, "colour")?;
                                ms.fp() as *const f32
                            } else {
                                std::ptr::null()
                            };
                            ck(unsafe { (k.to_u8)(gd, dec.ptr(), tw as i64, geo.left as i64, geo.top as i64, geo.sw as i64, geo.sh as i64, msp,
                                                  out_u8.ptr() as *mut u8) }, "bytes")?;
                            let mut pic = Picture::new(geo.sw as u32, geo.sh as u32, 3);
                            out_u8.read(0, &mut pic.data)?;
                            colour_secs += tcol.elapsed().as_secs_f64();
                            emitted.push(pic);
                            Ok(())
                        })?;
                    }
                    prof.carve("colour fix + bytes + readback", colour_secs);
                    prof.mark("decoder", &self.gpus);
                    for p in emitted {
                        emit(p)?;
                        made += 1;
                    }
                    progress(Step { picture: 0, at: made as u32, of: ended.map(|n| n as u32).or(total_hint).unwrap_or(0), seconds: t0.elapsed().as_secs_f64() })?;
                    prof.mark("emit (encoder pipe)", &self.gpus);
                }
                if done {
                    break;
                }
                if pipelined {
                    pending = Some((chunk, if chunk == 0 { 6 } else { 2 }));
                }
                // the frames no later chunk (nor the chunk still to decode) reads
                let keep = pending.map_or(8 * (chunk + 1) + 13, |(pc, _)| if pc == 0 { 0 } else { 8 * pc + 13 });
                while base < keep && !frames.is_empty() {
                    frames.pop_front();
                    base += 1;
                }
                chunk += 1;
            }
            drop(jtx);
            Ok(())
        })?;
        if prof.on {
            let total: f64 = prof.acc.iter().map(|(_, s)| s).sum();
            for st in inner.stages.iter() {
                let tot: f64 = st.ops.iter().map(|(_, s)| s).sum();
                for (n, s) in &st.ops {
                    eprintln!("flashvsr profile: blocks {}-{} on {}: {n:<34} {s:7.1} s {:5.1}%  {:.4} s a frame a block", st.from, st.to - 1, st.gpu.name,
                              100.0 * s / tot.max(1e-9), s / made.max(1) as f64 / (st.to - st.from) as f64);
                }
            }
            eprintln!("flashvsr profile: {made} frames, {total:.1} s, {:.3} s a frame (the cards taking turns, each mark waited for)", total / made.max(1) as f64);
            for (n, s) in &prof.acc {
                eprintln!("flashvsr profile: {n:<30} {s:8.1} s  {:5.1}%  {:.3} s a frame", 100.0 * s / total.max(1e-9), s / made.max(1) as f64);
            }
        }
        Ok(made as u32)
    }
}

/// Bytes a card may hold for FlashVSR before a size is run in tiles (of its 12 GiB: room for the driver and others)
const CARD_BUDGET: usize = 10_900 << 20;
/// A tile's largest side (the size the two cards run whole: 1728 x 960 out, 1792 x 1024 inside) and the tiles' overlap
const TILE_W: usize = 1792;
const TILE_H: usize = 1024;
const TILE_OVERLAP: usize = 128;
/// A tiled run's segments: frames made a pass, and the frames each tile's stream runs before them to fill its memory
const SEG: usize = 192;
const WARM: usize = 32;

/// Tiles along one side: (tile size, offsets)
fn tiles_along(total: usize, max: usize) -> (usize, Vec<usize>) {
    if total <= max {
        return (total, vec![0]);
    }
    let n = (total - TILE_OVERLAP).div_ceil(max - TILE_OVERLAP);
    let size = (total + (n - 1) * TILE_OVERLAP).div_ceil(n).div_ceil(128) * 128;
    let size = size.min(max);
    let offs = (0..n).map(|i| if n == 1 { 0 } else { i * (total - size) / (n - 1) }).collect();
    (size, offs)
}

impl FlashVsr {
    /// Whether a size runs whole on the cards (else in tiles)
    pub fn fits(&self, geo: &Geometry) -> bool {
        let need = self.need_bytes(geo);
        need.iter().zip(&self.weight_bytes).all(|(n, w)| n + w <= CARD_BUDGET)
    }

    /// A clip too big for the cards whole: in tiles of the upscaled frame (each its own stream), over segments of SEG
    /// frames, each tile's stream starting WARM frames before its segment (their frames thrown away: the KV cache and
    /// the convolutions' memory filled as in one stream) and running 8 past it (the frames it looks ahead to). The
    /// noise is the clip's by latent frame, so the tiles agree; their overlaps are blended linearly.
    #[allow(clippy::too_many_arguments)]
    pub fn run_tiled(&self, geo: Geometry, set: &Settings, total_hint: Option<u32>, next: &mut dyn FnMut() -> Result<Option<Vec<u8>>>,
                     emit: &mut dyn FnMut(Picture) -> Result<()>, progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<u32> {
        let t0 = Instant::now();
        let (tw, th) = (geo.tw, geo.th);
        let (tile_w, xs) = tiles_along(tw, TILE_W);
        let (tile_h, ys) = tiles_along(th, TILE_H);
        let tgeo = Geometry { w: tile_w, h: tile_h, sw: tile_w, sh: tile_h, tw: tile_w, th: tile_h, left: 0, top: 0 };
        if !self.fits(&tgeo) {
            return Err(Error(format!("flashvsr: a {tile_w} x {tile_h} tile does not fit the cards")));
        }
        let fsize = tw * th * 3;
        let mut frames: VecDeque<Vec<u8>> = VecDeque::new();
        let mut base = 0usize;
        let mut ended: Option<usize> = None;
        let mut made = 0usize;
        let mut s = 0usize;
        // a finished segment's frames, handed out one a tile frame while the next segment runs
        let mut pending: VecDeque<Vec<u8>> = VecDeque::new();
        let crop = |f: &Vec<u8>| -> Picture {
            let mut pic = Picture::new(geo.sw as u32, geo.sh as u32, 3);
            for r in 0..geo.sh {
                let at = ((geo.top + r) * tw + geo.left) * 3;
                pic.data[r * geo.sw * 3..(r + 1) * geo.sw * 3].copy_from_slice(&f[at..at + geo.sw * 3]);
            }
            pic
        };
        loop {
            // the segment [s, e), its stream from a, its frames up to e + 8 on hand
            while ended.is_none() && base + frames.len() < s + SEG + 8 {
                match next()? {
                    Some(f) => frames.push_back(resize::prepare(&f, &geo)),
                    None => ended = Some(base + frames.len()),
                }
            }
            let have = base + frames.len();
            let e = (s + SEG).min(ended.unwrap_or(usize::MAX)).min(have);
            if e <= s {
                for f in pending.drain(..) {
                    emit(crop(&f))?;
                    made += 1;
                }
                break;
            }
            let a = s.saturating_sub(WARM);
            let stop = (e + 8).min(have);
            let mut seg: Vec<Vec<u8>> = vec![vec![0u8; fsize]; e - s];
            for (j, &y0) in ys.iter().enumerate() {
                for (i, &x0) in xs.iter().enumerate() {
                    let mut idx = a;
                    let mut out = a;
                    let mut st = *set;
                    st.lat_base = a / 4;
                    let fr = &frames;
                    self.run(tgeo, &st, true, None, &mut || {
                        if idx >= stop {
                            return Ok(None);
                        }
                        let src = &fr[idx - base];
                        let mut t = vec![0u8; tile_w * tile_h * 3];
                        for r in 0..tile_h {
                            let at = ((y0 + r) * tw + x0) * 3;
                            t[r * tile_w * 3..(r + 1) * tile_w * 3].copy_from_slice(&src[at..at + tile_w * 3]);
                        }
                        idx += 1;
                        Ok(Some(t))
                    }, &mut |p| {
                        let k = out;
                        out += 1;
                        if k < s || k >= e {
                            return Ok(());
                        }
                        let dst = &mut seg[k - s];
                        // the overlaps with the tiles before (left, above) blended linearly, the rest copied
                        let ovx = if i > 0 { xs[i - 1] + tile_w - x0 } else { 0 };
                        let ovy = if j > 0 { ys[j - 1] + tile_h - y0 } else { 0 };
                        for r in 0..tile_h {
                            let ay = if r < ovy { (r as f32 + 0.5) / ovy as f32 } else { 1.0 };
                            let o = ((y0 + r) * tw + x0) * 3;
                            let q = r * tile_w * 3;
                            let plain = if ay < 1.0 { 0 } else { ovx };
                            for c in 0..plain.min(tile_w) {
                                let wgt = ((c as f32 + 0.5) / ovx as f32).min(ay);
                                for ch in 0..3 {
                                    let (d, v) = (dst[o + c * 3 + ch] as f32, p.data[q + c * 3 + ch] as f32);
                                    dst[o + c * 3 + ch] = (d + (v - d) * wgt).round() as u8;
                                }
                            }
                            if ay < 1.0 {
                                for c in 0..tile_w {
                                    let wgt = if c < ovx { ((c as f32 + 0.5) / ovx as f32).min(ay) } else { ay };
                                    for ch in 0..3 {
                                        let (d, v) = (dst[o + c * 3 + ch] as f32, p.data[q + c * 3 + ch] as f32);
                                        dst[o + c * 3 + ch] = (d + (v - d) * wgt).round() as u8;
                                    }
                                }
                            } else {
                                dst[o + plain * 3..o + tile_w * 3].copy_from_slice(&p.data[q + plain * 3..q + tile_w * 3]);
                            }
                        }
                        // the segment before, a frame at a time beside this work (its frames are not held back for a burst)
                        if let Some(f) = pending.pop_front() {
                            emit(crop(&f))?;
                            made += 1;
                        }
                        Ok(())
                    }, &mut |_| Ok(()), &mut NoHooks)?;
                }
            }
            // what the last segment's tiles did not hand out goes now; this segment's go during the next one
            for f in pending.drain(..) {
                emit(crop(&f))?;
                made += 1;
            }
            pending.extend(seg);
            progress(Step { picture: 0, at: made as u32, of: ended.map(|n| n as u32).or(total_hint).unwrap_or(0), seconds: t0.elapsed().as_secs_f64() })?;
            s = e;
            // the next segment's warm-up onward stays
            let keep = s.saturating_sub(WARM);
            while base < keep && !frames.is_empty() {
                frames.pop_front();
                base += 1;
            }
        }
        Ok(made as u32)
    }
}

impl ImageEngine for FlashVsr {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn defaults(&self) -> Defaults {
        Defaults { width: 0, height: 0, steps: 1, cfg: 1.0, sampler: Sampler::Euler, schedule: Schedule::Simple, shift: 1.0 }
    }
    fn load_seconds(&self) -> f64 {
        self.load_secs
    }
    fn options(&self) -> &'static [EngineOption] {
        OPTIONS
    }
    fn generate(&self, _req: &ImageRequest, _p: &mut dyn FnMut(Step) -> Result<()>) -> Result<Vec<Picture>> {
        Err(Error("flashvsr upscales clips (video.upscale) and pictures (image.upscale); it does not draw".into()))
    }
    fn upscale_variants(&self) -> Vec<UpscaleVariant> {
        VARIANTS.to_vec()
    }
    /// A picture as a still clip (FlashVSR needs a clip: the frame repeated to its 25), its first frame back
    fn upscale(&self, req: &UpscaleRequest, progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<(Picture, UpscaleInfo)> {
        let t0 = Instant::now();
        let set = Self::settings(req.variant.as_deref(), req.scale, None, &Default::default())?;
        let rgb = req.picture.rgb();
        let geo = Geometry::new(rgb.width as usize, rgb.height as usize, set.scale);
        let mut sent = false;
        let mut out = None;
        self.run(geo, &set, false, Some(1), &mut || Ok(if sent { None } else { sent = true; Some(rgb.data.clone()) }), &mut |p| {
            if out.is_none() {
                out = Some(p);
            }
            Ok(())
        }, progress, &mut NoHooks)?;
        let p = out.ok_or_else(|| Error("flashvsr made no frame".into()))?;
        Ok((p, UpscaleInfo { variant: VARIANTS[0].id.into(), scale: set.scale as u32, noise: None, tile: 0, tiles: 1, seconds: t0.elapsed().as_secs_f64() }))
    }
    fn upscales_clips(&self) -> bool {
        true
    }
    fn upscale_clip(&self, req: &ClipUpscaleRequest, next: &mut dyn FnMut() -> Result<Option<Vec<u8>>>, emit: &mut dyn FnMut(Picture) -> Result<()>,
                    progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<UpscaleInfo> {
        let t0 = Instant::now();
        let set = Self::settings(req.variant.as_deref(), req.scale, req.seed, &req.extra)?;
        let geo = Geometry::new(req.width as usize, req.height as usize, set.scale);
        let tiled = match std::env::var("NC_FLASHVSR_TILE").as_deref() {
            Ok("1") => true,
            Ok("0") => false,
            _ => !self.fits(&geo),
        };
        let n = if tiled { self.run_tiled(geo, &set, req.frames, next, emit, progress)? } else { self.run(geo, &set, false, req.frames, next, emit, progress, &mut NoHooks)? };
        Ok(UpscaleInfo { variant: VARIANTS[0].id.into(), scale: set.scale as u32, noise: None, tile: 0, tiles: n, seconds: t0.elapsed().as_secs_f64() })
    }
}

#[cfg(test)]
mod room_tests {
    use super::short_of;

    #[test]
    fn a_card_is_short_only_when_its_free_memory_is_known_and_below_the_need_and_the_stage_is_not_prepared() {
        assert_eq!(short_of(&[10, 10], &[Some(20), Some(5)], &[false, false]), Some((1, 5)));
        assert_eq!(short_of(&[10, 10], &[Some(20), Some(10)], &[false, false]), None);
        assert_eq!(short_of(&[10], &[Some(1)], &[true]), None);
        assert_eq!(short_of(&[10], &[None], &[false]), None);
    }
}
