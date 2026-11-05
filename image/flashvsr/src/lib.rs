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
    EngineOption { name: "split", env: "NC_FLASHVSR_SPLIT", value: "N,M", help: "the DiT's 30 blocks over the model's cards, in order (default: 18,12 on two, 30 on one)", at: At::Load },
    EngineOption { name: "sparse_ratio", env: "", value: "R", help: "the attention's top-k as FlashVSR's sparse_ratio (default 2.0; 1.5 faster, 2.0 steadier)", at: At::Request },
    EngineOption { name: "local_range", env: "", value: "N", help: "the attention's local window in 128-pixel blocks (default 11: steadier; 9 sharper)", at: At::Request },
    EngineOption { name: "color_fix", env: "", value: "0|1", help: "match each frame's colour to the upscaled input frame (AdaIN, default 1)", at: At::Request },
];

pub fn kind() -> ImageKind {
    ImageKind { archs: &[ARCH], name: "FlashVSR v1.1 video super-resolution (Wan 2.1 1.3B, one step, streaming; Rust + CUDA)", roles: &["dit", "lq-proj", "decoder", "prompt"], load,
                options: OPTIONS }
}

fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn ImageEngine>> {
    Ok(Box::new(FlashVsr::load(files, gpus, o.setting("NC_FLASHVSR_SPLIT").as_deref(), log)?))
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
}

impl Default for Settings {
    fn default() -> Self {
        Settings { scale: 2.0, seed: 0, sparse_ratio: 2.0, local_range: 11, color_fix: true }
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

struct NoHooks;
impl Hooks for NoHooks {}

struct Inner {
    stages: Vec<Stage>,
    lq: LqProj,
    tcd: Tcd,
}

pub struct FlashVsr {
    gpus: Vec<Arc<Gpu>>,
    ends: Ends,
    inner: Mutex<Inner>,
    load_secs: f64,
    pub weight_bytes: Vec<usize>,
}

fn pth_table(path: &Path) -> Result<SafeTensors> {
    nextcuda_gguf::pthst::open(path).map_err(|e| Error(e.0))
}

impl FlashVsr {
    pub fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], split: Option<&str>, log: &mut dyn FnMut(String)) -> Result<FlashVsr> {
        let t0 = Instant::now();
        if gpus.is_empty() {
            return Err(Error("flashvsr needs a GPU".into()));
        }
        ffi::api()?;
        let path = |role: &str| files.get(role).cloned().ok_or_else(|| Error(format!("flashvsr: no {role} file (the card names dit, lq-proj, decoder, prompt)")));
        let st = SafeTensors::open(&path("dit")?).map_err(|e| Error(e.0))?;
        let counts: Vec<usize> = match split {
            Some(s) if !s.trim().is_empty() => s.split(',').map(|v| v.trim().parse::<usize>().map_err(|_| Error(format!("NC_FLASHVSR_SPLIT {s}: block counts")))).collect::<Result<_>>()?,
            _ if gpus.len() >= 2 => vec![18, 12],
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
        let tcd = Tcd::load(&pth_table(&path("decoder")?)?, &last)?;
        let mut weight_bytes: Vec<usize> = stages.iter().map(|s| s.weight_bytes).collect();
        weight_bytes[0] += lq.bytes;
        *weight_bytes.last_mut().unwrap() += tcd.bytes;
        log(format!("flashvsr: loaded in {:.1} s ({})", t0.elapsed().as_secs_f64(),
                    weight_bytes.iter().zip(&gpus).map(|(b, g)| format!("{:.2} GiB on {}", *b as f64 / (1u64 << 30) as f64, g.name)).collect::<Vec<_>>().join(", ")));
        Ok(FlashVsr { gpus, ends, inner: Mutex::new(Inner { stages, lq, tcd }), load_secs: t0.elapsed().as_secs_f64(), weight_bytes })
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
        *v.last_mut().unwrap() += Tcd::need_bytes(geo.th / 8, geo.tw / 8);
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
            *v.last_mut().unwrap() += Tcd::need_bytes(h8, w8);
            v
        };
        for (s, n) in inner.stages.iter_mut().zip(&need) {
            if let Ok((_, Some(free))) = s.gpu.memory() {
                if !s.is_prepared(nb) && (*n as u64) > free {
                    return Err(Error(format!("flashvsr: {} x {} needs {:.1} GiB more on {} and it has {:.1} free (a smaller size, or NC_FLASHVSR_SPLIT over more cards)",
                                             geo.sw, geo.sh, *n as f64 / (1u64 << 30) as f64, s.gpu.name, free as f64 / (1u64 << 30) as f64)));
                }
            }
            s.prepare(nb)?;
        }
        inner.lq.reset(hp, wp)?;
        inner.tcd.reset(h8, w8)?;
        let ratio = set.sparse_ratio * 768.0 * 1280.0 / (th * tw) as f64;
        let topk = ((nb * nb) as f64 * ratio) as i64 - 1;
        let first_gpu = self.gpus[0].clone();
        let last_gpu = self.gpus.last().unwrap().clone();
        let fsize = th * tw * 3;
        // the frames on the host, prepared: [base, base + len) of the clip; the clip's count once it has ended
        let mut frames: VecDeque<Vec<u8>> = VecDeque::new();
        let mut base = 0usize;
        let mut ended: Option<usize> = None;
        let mut last: Vec<u8> = Vec::new();
        let mut read = |want: usize, frames: &mut VecDeque<Vec<u8>>, base: usize, ended: &mut Option<usize>, last: &mut Vec<u8>| -> Result<()> {
            while ended.is_none() && base + frames.len() < want {
                match next()? {
                    Some(f) => {
                        let p = if prepared { f } else { resize::prepare(&f, &geo) };
                        if p.len() != fsize {
                            return Err(Error(format!("flashvsr: a frame of {} bytes, {} expected", p.len(), fsize)));
                        }
                        last.clone_from(&p);
                        frames.push_back(p);
                    }
                    None => *ended = Some(base + frames.len()),
                }
            }
            Ok(())
        };
        let get = |i: usize, frames: &VecDeque<Vec<u8>>, base: usize, last: &Vec<u8>| -> Vec<u8> {
            if i >= base && i - base < frames.len() { frames[i - base].clone() } else { last.clone() }
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
        let cond_u8 = DevBuf::new(&last_gpu, 24 * fsize)?;
        let cond = DevBuf::new(&last_gpu, 24 * fsize * 2)?;
        let sums = DevBuf::new(&last_gpu, 12 * 8)?;
        let ms = DevBuf::new(&last_gpu, 12 * 4)?;
        let out_u8 = DevBuf::new(&last_gpu, geo.sh * geo.sw * 3)?;
        let mut staging = vec![0u8; if inner.stages.len() > 1 { rows0 * DIM * 4 } else { 0 }];
        let nsd0 = &inner.stages[0].nsd as *const nextcuda_diffusion::kernels::Nsd;
        let mut made = 0usize;
        let mut chunk = 0usize;
        let g0 = g(&first_gpu);
        let gl = g(&last_gpu);
        loop {
            let hi = if chunk == 0 { 25 } else { 8 * chunk + 25 };
            read(hi, &mut frames, base, &mut ended, &mut last)?;
            if let Some(n) = ended {
                if n == 0 {
                    return Err(Error("flashvsr: the clip has no frames".into()));
                }
                let nn = (n + 3).div_ceil(8).max(3);
                if chunk >= nn - 2 {
                    break;
                }
            }
            let (f, lat0, calls): (usize, usize, Vec<(usize, usize)>) = if chunk == 0 {
                (6, 0, (0..7).map(|i| (if i == 0 { 0 } else { 4 * i - 3 }, 4 * i + 1)).collect())
            } else {
                (2, 4 + 2 * chunk, vec![(8 * chunk + 17, 8 * chunk + 21), (8 * chunk + 21, 8 * chunk + 25)])
            };
            let rows = f / 2 * nb * 128;
            // SAFETY (the kernel calls below): every buffer holds what the call reads or writes, sized above.
            // the LQ projection
            {
                let nsd = unsafe { &*nsd0 };
                let mut got = 0;
                for (a, b) in calls {
                    let n = b - a;
                    let mut bytes = Vec::with_capacity(n * fsize);
                    for i in a..b {
                        bytes.extend_from_slice(&get(i, &frames, base, &last));
                    }
                    frames_u8.write(0, &bytes)?;
                    ck(unsafe { (k.u8_to_half)(g0, frames_u8.ptr() as *const u8, (n * fsize) as i64, frames_h.ptr()) }, "frames")?;
                    let out = lqtok.view(got * hp * wp * DIM * 4, hp * wp * DIM * 4)?;
                    if inner.lq.stream(nsd, &frames_h, n, &out)? {
                        got += 1;
                    }
                }
                if got != f {
                    return Err(Error(format!("flashvsr: the LQ projection made {got} latent frames, {f} expected")));
                }
                hooks.tensor(chunk, "lq", &lqtok, f * hp * wp)?;
            }
            // the noise, on the first card (the patches) and the last (the step)
            for (buf, gpu) in [(&noise0, &first_gpu), (&noise_l, &last_gpu)] {
                for t in 0..f {
                    match hooks.noise(lat0 + t, h8, w8) {
                        Some(v) => {
                            for c in 0..16 {
                                // SAFETY: plain floats as bytes
                                let b = unsafe { std::slice::from_raw_parts(v[c * h8 * w8..].as_ptr().cast::<u8>(), h8 * w8 * 4) };
                                buf.write(((c * f + t) * h8 * w8) * 4, b)?;
                            }
                        }
                        None => {
                            for c in 0..16 {
                                let p = unsafe { buf.fp().add((c * f + t) * h8 * w8) };
                                ck(unsafe { (k.randn)(g(gpu), p, (h8 * w8) as i64, set.seed, (((lat0 + t) * 16 + c) * h8 * w8) as u64) }, "noise")?;
                            }
                        }
                    }
                }
            }
            // the tokens
            ck(unsafe { (k.patchify)(g0, noise0.fp(), f as i64, h8 as i64, w8 as i64, patches.ptr()) }, "patches")?;
            self.ends.patch.apply(unsafe { &*nsd0 }, patches.ptr(), rows, xs[0].ptr(), Dt::F32)?;
            ck(unsafe { (k.add_lq)(g0, xs[0].fp(), lqtok.fp(), f as i64, hp as i64, wp as i64, DIM as i64) }, "lq residual")?;
            let cs = dit::rope_table(f / 2, hp, wp, lat0);
            let c = Chunk { groups: f / 2, nbh, nbw, first: chunk == 0, evict: if chunk == 0 { 0 } else { (chunk - 1) % 3 }, topk, local: set.local_range };
            let ns = inner.stages.len();
            for si in 0..ns {
                if si > 0 {
                    let src = xs[si - 1].view(0, rows * DIM * 4)?;
                    xs[si].copy_from_peer(&src, &mut staging)?;
                }
                // SAFETY: plain floats as bytes
                let csb = unsafe { std::slice::from_raw_parts(cs.as_ptr().cast::<u8>(), cs.len() * 4) };
                css[si].write(0, csb)?;
                let stage = &mut inner.stages[si];
                let x = &xs[si];
                stage.run(x, &css[si], &c, &mut |layer, x| {
                    if [0, 15, 29].contains(&layer) {
                        hooks.tensor(chunk, &format!("b{layer}"), x, rows)?;
                    }
                    Ok(())
                })?;
            }
            // the head and the step: latents = noise - output
            let x = &xs[ns - 1];
            let nsdl = &inner.stages[ns - 1].nsd;
            ck(unsafe { (k.ln)(gl, x.fp(), rows as i64, DIM as i64, 1e-6, self.ends.head_sc.fp(), self.ends.head_sh.fp(), hbuf.ptr()) }, "head norm")?;
            self.ends.head.apply(nsdl, hbuf.ptr(), rows, pred.ptr(), Dt::F32)?;
            ck(unsafe { (k.unpatchify_sub)(gl, pred.fp(), noise_l.fp(), f as i64, h8 as i64, w8 as i64, lat.fp()) }, "step")?;
            hooks.tensor(chunk, "lat", &lat, 16 * f * h8 * w8)?;
            // the decoder's condition: the frames it makes (chunk 0: 0-20 with frame 0 three times in front)
            let (pre, cur) = if chunk == 0 { (0, 21) } else { (8 * chunk + 13, 8 * chunk + 21) };
            let front = if chunk == 0 { 3 } else { 0 };
            let mut bytes = Vec::with_capacity((cur - pre + front) * fsize);
            for _ in 0..front {
                bytes.extend_from_slice(&get(0, &frames, base, &last));
            }
            for i in pre..cur {
                bytes.extend_from_slice(&get(i, &frames, base, &last));
            }
            let nc = cur - pre + front;
            cond_u8.write(0, &bytes)?;
            ck(unsafe { (k.u8_to_half)(gl, cond_u8.ptr() as *const u8, (nc * fsize) as i64, cond.ptr()) }, "cond")?;
            let mut idx = pre;
            let limit = ended;
            let real = |i: usize| limit.is_none_or(|n| i < n);
            let mut emitted = Vec::new();
            for t in 0..f {
                let cp = unsafe { (cond.ptr() as *mut u8).add(4 * t * fsize * 2) };
                inner.tcd.frame(nsdl, &lat, f, t, cp as *const _, &mut |dec| {
                    let i = idx;
                    idx += 1;
                    if !real(i) {
                        return Ok(());
                    }
                    hooks.decoded(i, dec)?;
                    let ci = i - pre + front;
                    let lqf = unsafe { (cond.ptr() as *mut u8).add(ci * fsize * 2) };
                    let msp = if set.color_fix {
                        sums.fill(0)?;
                        let s8 = sums.ptr() as *mut f64;
                        ck(unsafe { (k.stats)(gl, dec.ptr(), tw as i64, geo.left as i64, geo.top as i64, geo.sw as i64, geo.sh as i64, 2.0, -1.0, s8) },
                           "stats")?;
                        ck(unsafe { (k.stats)(gl, lqf as *const _, tw as i64, geo.left as i64, geo.top as i64, geo.sw as i64, geo.sh as i64, 1.0, 0.0,
                                              s8.add(6)) }, "stats")?;
                        let mut raw = [0u8; 96];
                        sums.read(0, &mut raw)?;
                        let s: Vec<f64> = raw.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
                        let n = (geo.sw * geo.sh) as f64;
                        let mut m = [0f32; 12];
                        for ch in 0..3 {
                            let (md, ml) = (s[ch] / n, s[6 + ch] / n);
                            let (vd, vl) = ((s[3 + ch] / n - md * md).max(0.0), (s[9 + ch] / n - ml * ml).max(0.0));
                            m[ch * 4] = md as f32;
                            m[ch * 4 + 1] = (vd + 1e-5).sqrt() as f32;
                            m[ch * 4 + 2] = ml as f32;
                            m[ch * 4 + 3] = (vl + 1e-5).sqrt() as f32;
                        }
                        // SAFETY: plain floats as bytes
                        ms.write(0, unsafe { std::slice::from_raw_parts(m.as_ptr().cast::<u8>(), 48) })?;
                        ms.fp() as *const f32
                    } else {
                        std::ptr::null()
                    };
                    ck(unsafe { (k.to_u8)(gl, dec.ptr(), tw as i64, geo.left as i64, geo.top as i64, geo.sw as i64, geo.sh as i64, msp,
                                          out_u8.ptr() as *mut u8) }, "bytes")?;
                    let mut pic = Picture::new(geo.sw as u32, geo.sh as u32, 3);
                    out_u8.read(0, &mut pic.data)?;
                    emitted.push(pic);
                    Ok(())
                })?;
            }
            for p in emitted {
                emit(p)?;
                made += 1;
            }
            progress(Step { picture: 0, at: made as u32, of: ended.map(|n| n as u32).or(total_hint).unwrap_or(0), seconds: t0.elapsed().as_secs_f64() })?;
            // the frames no later chunk reads
            let keep = 8 * (chunk + 1) + 13;
            while base < keep && !frames.is_empty() {
                frames.pop_front();
                base += 1;
            }
            chunk += 1;
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
        let n = self.run(geo, &set, false, req.frames, next, emit, progress, &mut NoHooks)?;
        Ok(UpscaleInfo { variant: VARIANTS[0].id.into(), scale: set.scale as u32, noise: None, tile: 0, tiles: n, seconds: t0.elapsed().as_secs_f64() })
    }
}
