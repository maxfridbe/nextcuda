//! P-H3-Image (Pruna AI; MiniMax H3 Community License, its Qwen3-VL-4B parts Apache-2.0) in Rust and CUDA: text to
//! image, and edits from 1 to 8 pictures. Pruna's diffusers pipeline (modular_pipelines/p_h3_image) over the
//! foundation's MiniMax H3 core (crates/h3): the image is H3's video with one latent frame.
//!
//! ```text
//!   prompt -> Qwen3-VL-4B (no chat template, no special tokens) -> its hidden states after layers 13, 19, 25, 31 and
//!     the normed last -> the text connector -> [L, 5120], what H3's own 32B encoder would give its denoiser
//!   an edit: per picture "<Picture i>: " and its vision block (the encoder's vision tower, ~1 MP), then
//!     "Edit <Picture 1>. " and the instruction; the pictures' latents (H3's encoder, the posterior sampled at seed 42)
//!     packed ahead of the target as H3's ref2va references, held at t = 0.999
//!   [text | references | 2 audio latents | 1 video latent frame] -> H3's 50 blocks, 8 Euler steps on the
//!     checkpoint's own sigmas, no guidance; the audio rows ride along and are dropped
//!   latents -> the picture decoder (H3's ViT decoder for one frame, in 256-pixel tiles) -> RGB
//! ```
//!
//! On two 12 GB cards (an RTX 5070 and an RTX 3060): the denoiser (vantagewithai's Q4_K_M GGUF of Pruna's weights)
//! split over both, the text encoder (Q8_0) and the decoder (int8 ConvRot) resident on the second - every weight on a
//! card, nothing streamed. The prompt upsampler (Qwen3.5-9B) is not run: prompts are rendered as written.

pub mod connector;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use nextcuda_h3::denoiser::{self, Conditions, Denoiser, RefIn, Schedule, Shape, TextRefiner};
use nextcuda_h3::device::{Device, Tensor};
use nextcuda_h3::dtype::DType;
use nextcuda_h3::gguf::{GType, Gguf};
use nextcuda_h3::jobs;
use nextcuda_h3::safetensors::Checkpoint;
use nextcuda_h3::te::{Resident, TextEncoder};
use nextcuda_h3::vision::{Frames, Piece};
use nextcuda_image::{At, Defaults, EngineOption, Error, Gpu, ImageEngine, ImageKind, ImageRequest, LoadOptions, ModelFiles, Picture, Result, Sampler,
                     Schedule as Sched, SizePresets, Step};

use connector::{Connector, TAPS};

pub const ARCH: &str = "p-h3-image";
/// the denoiser (a GGUF of H3's transformer), the text encoder (Qwen3-VL-4B with its connector and vision tower, a
/// GGUF), the VAE (H3's encoder and the one-frame decoder), the tokenizer (`vocab.json`; `merges.txt` and
/// `tokenizer_config.json` beside it: Qwen's, the files H3 uses)
pub const ROLES: &[&str] = &["denoiser", "text-encoder", "vae", "tokenizer"];

/// The checkpoint's sampling grid (modular_model_index.json `sample_sigmas`), used as it is: the scheduler's config
/// sets shift_provided_sigmas false, so its shift of 12 does not touch it (ComfyUI's workflows: ModelSamplingMiniMaxH3
/// at 1.0 for both shifts, ManualSigmas)
pub const SIGMAS: [f32; 9] = [1.0, 0.984753, 0.967905, 0.951688, 0.925372, 0.881923, 0.810083, 0.528718, 0.0];
/// an edit takes at most this many pictures (the first is edited, the rest are references)
pub const MAX_PICTURES: usize = 8;
/// audio latents per stereo channel the layout denoises beside the picture, then drops
const AUDIO_T: usize = 2;
/// the seed every edit picture's posterior is sampled under, whatever the request's (keyframe_encode_seed)
const ENCODE_SEED: u64 = 42;
/// the timestep an edit's pictures are noised to and held at (keyframe_noise_aug)
const NOISE_AUG: f32 = 0.999;
/// a canvas side is a multiple of this (16 the VAE's compression times the 2 x 2 patch)
pub const GRID: u32 = 32;
/// the pictures of an edit for the vision tower: about a megapixel each (vision_pixels)
const VISION_PIXELS: f64 = 1_000_000.0;
/// an edit's pictures for the VAE keep their size between these areas (reference_min_pixels, reference_max_pixels);
/// with no canvas asked the first sets it, at least canvas_min_pixels
const REF_MIN: f64 = 1_048_576.0;
const REF_MAX: f64 = 8_388_608.0;
const CANVAS_MIN: f64 = 4_194_304.0;

/// The model card's resolutions: 8 MP (best for text to image) and 4 MP (edits) at seven aspect ratios, and 1 MP (as
/// nextsycl offers them); a page makes a size as the budget at the ratio, each side rounded to 32 (2912 x 2912,
/// 3360 x 2496 ... 2048 x 2048, 2368 x 1760 ... 1024 x 1024)
const RATIOS: &[&str] = &["9:16", "2:3", "3:4", "1:1", "4:3", "3:2", "16:9"];
const PIXELS: &[u32] = &[1_048_576, 4_194_304, 8_388_608];
/// The largest canvas, about 8.9 MP (nextsycl's limit: every 8 MP preset and a little over)
pub const MAX_PIXELS: u32 = 8_912_896;
/// The steps: the checkpoint's grid has 8; another grid comes as `options.sigmas`
pub const STEPS: u32 = 8;

/// A request's options: another sampling grid than the checkpoint's
pub const OPTIONS: &[EngineOption] = &[EngineOption {
    name: "sigmas",
    env: "",
    value: "S,S,...",
    help: "the sampling grid instead of the checkpoint's 8 steps: strictly decreasing from at most 1 down to 0 (one step between two)",
    at: At::Request,
}];

/// The text encoder's taps after these layers, the last through its final norm (connector::TAPS)
const TE_LAYERS: usize = 36;

pub fn kind() -> ImageKind {
    ImageKind {
        archs: &[ARCH],
        name: "P-H3-Image (MiniMax H3's 50-block DiT over one latent frame, Qwen3-VL-4B text encoder and connector, H3's VAE encoder and a one-frame ViT decoder; Rust + CUDA)",
        roles: ROLES,
        load,
        options: OPTIONS,
    }
}

fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn ImageEngine>> {
    let reserve = match o.setting("NC_PH3_RESERVE_GIB") {
        Some(v) => v.trim().parse::<f64>().ok().filter(|g| (0.5..=8.0).contains(g)).ok_or_else(|| Error(format!("NC_PH3_RESERVE_GIB={v}: GiB, 0.5 to 8")))?,
        None => 2.0,
    };
    Ok(Box::new(PH3Image::load(files, gpus, (reserve * (1u64 << 30) as f64) as u64, log).map_err(ie)?))
}

fn ie(e: nextcuda_h3::Error) -> Error {
    Error(e.0)
}

fn role<'f>(files: &'f ModelFiles, r: &str) -> nextcuda_h3::Result<&'f Path> {
    files.get(r).map(|p| p.as_path()).ok_or_else(|| nextcuda_h3::Error(format!("{ARCH}: no {r} file")))
}

fn gib(b: u64) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

/// Python's round(): halves to the even neighbour (what diffusers' size arithmetic uses)
fn py_round(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - x.signum()
    } else {
        r
    }
}

/// diffusers' `_size_at_area`: about `pixels` at the picture's own aspect ratio, sides multiples of 32; a picture
/// already on the grid and within 20% of the area keeps its size
fn size_at_area(w: u32, h: u32, pixels: f64) -> (u32, u32) {
    let m = GRID as f64;
    let area = (w as f64) * (h as f64);
    if w % GRID == 0 && h % GRID == 0 && 0.8 * pixels <= area && area <= 1.2 * pixels {
        return (w, h);
    }
    let ratio = w as f64 / h as f64;
    let side = |v: f64| (py_round(v / m) * m).max(m) as u32;
    (side((pixels * ratio).sqrt()), side((pixels / ratio).sqrt()))
}

/// diffusers' `_size_in_area_range`: the picture's size, scaled to `lo` or `hi` pixels when its area is outside them,
/// sides rounded to multiples of 32
fn size_in_area_range(w: u32, h: u32, lo: f64, hi: f64) -> (u32, u32) {
    let m = GRID as f64;
    let area = (w as f64) * (h as f64);
    let scale = (area.clamp(lo, hi) / area).sqrt();
    let side = |v: u32| (py_round(v as f64 * scale / m) * m).max(m) as u32;
    (side(w), side(h))
}

/// A picture's RGB float32 [h, w, 3] in [0, 1] at `w` x `h` (PIL's Lanczos when the size changes, as diffusers'
/// VaeImageProcessor.resize does; the bytes are the resized picture's)
fn pixels_at(p: &Picture, (w, h): (u32, u32)) -> Vec<f32> {
    let rgb = p.rgb();
    let r = if (rgb.width, rgb.height) == (w, h) { rgb } else { rgb.resize_lanczos(w, h) };
    r.data.iter().map(|b| *b as f32 / 255.0).collect()
}

/// A request's own grid (`options.sigmas`): strictly decreasing, at most 1, ending at 0, used as written (as the
/// scheduler takes given sigmas: no shift)
pub fn parse_sigmas(v: &str) -> std::result::Result<Vec<f32>, String> {
    let s: Vec<f32> = v.split(',').map(|x| x.trim().parse::<f32>()).collect::<std::result::Result<_, _>>()
        .map_err(|_| format!("options.sigmas {v:?}: numbers, comma separated"))?;
    if s.len() < 2 || s[0] > 1.0 || *s.last().unwrap_or(&1.0) != 0.0 || !s.windows(2).all(|w| w[0] > w[1]) {
        return Err(format!("options.sigmas {v:?}: at least two levels, strictly decreasing from at most 1 down to 0"));
    }
    Ok(s)
}

/// diffusers' `round_canvas`: each side to the nearest multiple of 32
pub fn round_canvas(w: u32, h: u32) -> (u32, u32) {
    let r = |v: u32| ((py_round(v as f64 / GRID as f64) as u32) * GRID).max(GRID);
    (r(w), r(h))
}

/// One reference of an edit, ready for every picture of the request: its normalized latents and latent size
struct Encoded {
    latent: Vec<f32>,
    h: usize,
    w: usize,
}

pub struct PH3Image {
    e: jobs::Engine,
    te: TextEncoder,
    resident: Resident,
    final_norm: Vec<f32>,
    connector: Connector,
    vae: nextcuda_h3::vae::VideoDecoder,
    vae_path: PathBuf,
    /// the vision tower's tensors in the text encoder's GGUF (`model.visual.*`), as a table its loader reads
    tower: nextcuda_gguf::safetensors::SafeTensors,
    tok: nextcuda_h3::tokenizer::Tokenizer,
    /// the card the text encoder, the connector and the decoder are on
    te_dev: Arc<Device>,
    loaded: Instant,
    load_s: f64,
    busy: Mutex<()>,
    /// the last prompts' conditioning (text to image): [L, 5120]
    conds: Mutex<Vec<(String, Arc<Vec<f32>>)>>,
}

/// The tensors of a GGUF under `prefix` that are stored as floats, as a safetensors-style table over the same file
fn float_table(path: &Path, prefix: &str) -> nextcuda_h3::Result<nextcuda_gguf::safetensors::SafeTensors> {
    let g = Gguf::open(path)?;
    let tensors = g.entries.values().filter(|e| e.name.starts_with(prefix)).filter_map(|e| {
        let dtype = match e.ty {
            GType::F32 => "F32",
            GType::F16 => "F16",
            GType::BF16 => "BF16",
            _ => return None,
        };
        Some(nextcuda_gguf::safetensors::StTensor { name: e.name.clone(), dtype: dtype.into(), shape: e.shape.iter().map(|d| *d as u64).collect(),
                                                    offset: e.offset, bytes: e.bytes as u64 })
    }).collect();
    nextcuda_gguf::safetensors::SafeTensors::from_tensors(path, tensors).map_err(|e| nextcuda_h3::Error(e.0))
}

impl PH3Image {
    /// The text encoder, its connector and the decoder on the last card, then the denoiser's blocks over all of them
    /// (`reserve` bytes kept on each beside its blocks for a picture's buffers)
    pub fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], reserve: u64, log: &mut dyn FnMut(String)) -> nextcuda_h3::Result<PH3Image> {
        let t0 = Instant::now();
        nextcuda_core::use_kind("image");
        let idx: Vec<usize> = if gpus.is_empty() { vec![0] } else { gpus.iter().map(|g| g.index).collect() };
        let devs = idx.iter().map(|i| Device::open_index(*i)).collect::<nextcuda_h3::Result<Vec<_>>>()?;
        let te_dev = devs.last().cloned().ok_or("no card")?;
        let te_path = role(files, "text-encoder")?.to_path_buf();
        let tok_file = role(files, "tokenizer")?;
        let tok = nextcuda_h3::tokenizer::Tokenizer::load(if tok_file.is_file() { tok_file.parent().unwrap_or(tok_file) } else { tok_file })?;
        // the text encoder resident first: its card's room for the blocks is what is left
        let te = TextEncoder::open(&te_path)?;
        if te.layers != TE_LAYERS {
            return Err(nextcuda_h3::Error(format!("{}: {} layers, not Qwen3-VL-4B's {TE_LAYERS}", te_path.display(), te.layers)));
        }
        let t1 = Instant::now();
        let resident = te.load_resident(std::slice::from_ref(&te_dev), 8)?;
        let te_bytes: u64 = resident.placement().iter().map(|p| p.2).sum();
        log(format!("te     : Qwen3-VL-4B, {} layers ({:.2} GiB) on {} in {:.1} s", te.layers, gib(te_bytes), te_dev.name(), t1.elapsed().as_secs_f64()));
        let final_norm = te.final_norm()?;
        let te_ck = Checkpoint::open(&te_path)?;
        let connector = Connector::load(&te_dev, &te_ck)?;
        if connector.text != te.hidden {
            return Err(nextcuda_h3::Error(format!("the text connector reads {} a token, the encoder gives {}", connector.text, te.hidden)));
        }
        let tower = float_table(&te_path, "model.visual.")?;
        let vae_path = role(files, "vae")?.to_path_buf();
        let t1 = Instant::now();
        let vae = nextcuda_h3::vae::VideoDecoder::load(&te_dev, &Checkpoint::open(&vae_path)?, 8)?;
        log(format!("vae    : the one-frame decoder ({:.2} GiB) on {} in {:.1} s", gib(vae.load_bytes), te_dev.name(), t1.elapsed().as_secs_f64()));
        let mut e = jobs::Engine::load_cards(devs.clone(), role(files, "denoiser")?, None, 8, log)?;
        if connector.out != e.ck.get("condition_proj.weight")?.shape[1] {
            return Err("the text connector's width is not the denoiser's text width".into());
        }
        e.reserve = reserve;
        let t1 = Instant::now();
        let b = e.blocks()?;
        let place: Vec<String> = devs.iter().enumerate().map(|(d, dev)| format!("{} on {}", b.dev_of.iter().filter(|x| **x == d).count(), dev.name())).collect();
        log(format!("blocks : {} ({:.2} GiB) in {:.1} s: {}", b.blocks.len(), gib(b.load_bytes), t1.elapsed().as_secs_f64(), place.join(", ")));
        for d in &devs {
            log(format!("memory : {} holds {:.2} GiB of its {:.2}", d.name(), gib(d.mem_used()), gib(d.mem_cap())));
        }
        Ok(PH3Image { e, te, resident, final_norm, connector, vae, vae_path, tower, tok, te_dev, loaded: Instant::now(), load_s: t0.elapsed().as_secs_f64(),
                      busy: Mutex::new(()), conds: Mutex::new(Vec::new()) })
    }

    /// The prompt's pieces through the text encoder and the connector: the conditioning [L, 5120] and, with pictures in
    /// it, each token's modality tag (a vision block's tokens run on the video tables)
    fn condition(&self, pieces: &[Piece], log: &mut dyn FnMut(String)) -> nextcuda_h3::Result<(Vec<f32>, Option<Vec<i32>>)> {
        let presented = nextcuda_h3::vision::present_with(pieces, &self.tok, Some((&self.tower, "model.visual.")), None, self.e.dev.index, log)?;
        let taps = self.te.encode_resident_taps(&self.resident, &presented.ids, &presented.seen, &presented.embeds, &TAPS, Some(&self.final_norm),
                                                &mut |_| Ok(()))?;
        let ctx = self.connector.forward(&taps)?;
        Ok((ctx, presented.has_vision().then(|| presented.tags.clone())))
    }

    /// A text-to-image prompt's conditioning, from the last few kept
    fn prompt_condition(&self, prompt: &str, log: &mut dyn FnMut(String)) -> nextcuda_h3::Result<Arc<Vec<f32>>> {
        if let Some((_, c)) = self.conds.lock().unwrap().iter().find(|(p, _)| p == prompt) {
            return Ok(c.clone());
        }
        let (c, _) = self.condition(&[Piece::Text(prompt.to_string())], log)?;
        let c = Arc::new(c);
        let mut list = self.conds.lock().unwrap();
        list.push((prompt.to_string(), c.clone()));
        if list.len() > 8 {
            list.remove(0);
        }
        Ok(c)
    }

    /// One picture: the denoiser over `refs` and the text, the starting noise and the references' augmentation from
    /// one torch generator seeded `seed` (as diffusers draws them: each reference's noise, then the picture's, then the
    /// audio's), Euler over `sig`, the decoder. RGB float32 [3, H, W] in [0, 1].
    #[allow(clippy::too_many_arguments)]
    fn picture(&self, ctx: &[f32], tags: Option<&Vec<i32>>, refs: &[Encoded], (w, h): (usize, usize), seed: u64, sig: &[f32],
               step: &mut dyn FnMut(usize) -> Result<()>) -> nextcuda_h3::Result<Vec<f32>> {
        let e = &self.e;
        let model = e.blocks()?;
        let l = ctx.len() / self.connector.out;
        let text = {
            let refiner = TextRefiner::load(&e.dev, &e.ck, e.threads)?;
            let t = Tensor::from_bytes(&e.dev, DType::BF16, &[l, self.connector.out], &denoiser::bf16_bytes(ctx))?;
            let r = refiner.run(&e.cfg, &t, None)?;
            e.dev.wait()?;
            r
        };
        let mut g = nextcuda_h3::noise::Mt19937::new(seed);
        let mut cond = Conditions { seed, visual_aug: NOISE_AUG, mixed: true, ..Conditions::default() };
        for r in refs {
            let n = g.randn(r.latent.len());
            let latent = r.latent.iter().zip(&n).map(|(z, n)| NOISE_AUG * z + (1.0 - NOISE_AUG) * n).collect();
            cond.refs.push(RefIn::Image { latent, h: r.h, w: r.w });
        }
        let shape = Shape { t: 1, h: h / 16, w: w / 16, audio_t: AUDIO_T };
        let noise_v = g.randn(24 * shape.h * shape.w);
        // diffusers draws the audio rows as [channel-major rows, 32 features]: here [32 features, 2 channels, frames]
        let rows = g.randn(2 * AUDIO_T * e.outer.audio_c);
        let mut noise_a = vec![0f32; rows.len()];
        for r in 0..2 * AUDIO_T {
            for c in 0..e.outer.audio_c {
                noise_a[c * 2 * AUDIO_T + r] = rows[r * e.outer.audio_c + c];
            }
        }
        // the audio steps down the same grid as the picture, unscaled (diffusers' audio_scheduler, also unshifted)
        let schedule = Schedule { shift_video: 1.0, shift_audio: 1.0, audio_scale: 1.0 };
        let mut d = Denoiser::new(&model, &e.outer, text, tags.cloned(), shape, schedule, &cond)?;
        let (v, _) = denoiser::sample_masked(&mut d, &noise_v, &noise_a, sig, None, &mut |i, _, _| step(i).map_err(|x| nextcuda_h3::Error(x.0)),
                                             &mut |_, _| Ok(()))?;
        drop(d);
        drop(model);
        let (px, _) = self.vae.decode(&v, 1, shape.h, shape.w, &mut || Ok(()))?;
        Ok(px)
    }

    /// An edit's pictures: their latents for the denoiser (at their own sizes, sampled as diffusers encodes them) and
    /// their prompt pieces for the text encoder; the canvas (`size`, else the first picture's at 4 to 8 MP)
    #[allow(clippy::type_complexity)]
    fn edit_inputs(&self, pics: &[&Picture], prompt: &str, size: Option<(u32, u32)>, log: &mut dyn FnMut(String))
                   -> nextcuda_h3::Result<(Vec<Encoded>, Vec<Piece>, (u32, u32))> {
        let mut ref_sizes: Vec<(u32, u32)> = pics.iter().map(|p| size_in_area_range(p.width, p.height, REF_MIN, REF_MAX)).collect();
        let canvas = match size {
            Some(s) => s,
            None => {
                ref_sizes[0] = size_in_area_range(pics[0].width, pics[0].height, CANVAS_MIN.max(REF_MIN), REF_MAX);
                ref_sizes[0]
            }
        };
        let venc = nextcuda_h3::venc::VideoEncoder::load(&self.te_dev, &Checkpoint::open(&self.vae_path)?)?;
        let mut refs = Vec::new();
        let mut pieces = Vec::new();
        for (i, (p, rs)) in pics.iter().zip(&ref_sizes).enumerate() {
            let px = pixels_at(p, *rs);
            let latent = venc.encode_sampled(&px, rs.1 as usize, rs.0 as usize, ENCODE_SEED, &mut || Ok(()))?;
            refs.push(Encoded { latent, h: rs.1 as usize / 16, w: rs.0 as usize / 16 });
            let vs = size_at_area(p.width, p.height, VISION_PIXELS);
            log(format!("picture: <Picture {}> {}x{} -> {}x{} for the denoiser, {}x{} for the vision tower", i + 1, p.width, p.height, rs.0, rs.1, vs.0, vs.1));
            pieces.push(Piece::Text(format!("<Picture {}>: ", i + 1)));
            pieces.push(Piece::Vision(Frames { frames: vec![pixels_at(p, vs)], w: vs.0 as usize, h: vs.1 as usize }));
        }
        let lead = prompt.trim_start();
        let text = if lead.starts_with("Edit <Picture 1>.") || lead.starts_with("subject_definitions:") { prompt.to_string() } else { format!("Edit <Picture 1>. {prompt}") };
        pieces.push(Piece::Text(text));
        Ok((refs, pieces, canvas))
    }
}

impl ImageEngine for PH3Image {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn defaults(&self) -> Defaults {
        Defaults { width: 2912, height: 2912, steps: STEPS, cfg: 1.0, sampler: Sampler::Euler, schedule: Sched::Shift, shift: 1.0 }
    }
    fn edits(&self) -> bool {
        true
    }
    fn samplers(&self) -> Vec<Sampler> {
        vec![Sampler::Euler]
    }
    fn schedules(&self) -> Vec<Sched> {
        vec![Sched::Shift]
    }
    /// guidance-distilled: one pass a step, no negative prompt
    fn guidance(&self) -> bool {
        false
    }
    fn shift(&self) -> bool {
        false
    }
    fn takes_loras(&self) -> bool {
        false
    }
    fn grid(&self) -> u32 {
        GRID
    }
    fn size_presets(&self) -> Option<SizePresets> {
        Some(SizePresets { ratios: RATIOS, pixels: PIXELS })
    }
    fn sides(&self) -> (u32, u32) {
        (256, 4096)
    }
    fn max_pixels(&self) -> Option<u32> {
        Some(MAX_PIXELS)
    }
    fn load_seconds(&self) -> f64 {
        self.load_s
    }
    fn options(&self) -> &'static [EngineOption] {
        OPTIONS
    }
    /// 8 steps (the checkpoint's grid); another count is refused, another grid comes as options.sigmas
    fn steps_free(&self) -> bool {
        false
    }
    fn steps_for(&self, req: &ImageRequest) -> u32 {
        req.extra.get("sigmas").and_then(|v| parse_sigmas(v).ok()).map_or(STEPS, |s| s.len() as u32 - 1)
    }

    fn generate(&self, req: &ImageRequest, progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<Vec<Picture>> {
        let _one = self.busy.lock().unwrap_or_else(|p| p.into_inner());
        let t0 = Instant::now();
        let d = self.defaults();
        if req.negative.is_some() {
            return Err(Error(format!("{ARCH} takes no negative prompt: it is guidance-distilled (8 steps, one pass a step, no CFG)")));
        }
        if req.cfg.is_some_and(|c| c != 1.0) {
            return Err(Error(format!("{ARCH} takes no cfg (guidance scale): it is guidance-distilled - leave cfg out, or 1")));
        }
        if !req.loras.is_empty() {
            return Err(Error(format!("{ARCH} takes no LoRAs")));
        }
        if req.steps.is_some_and(|s| s != STEPS) {
            return Err(Error(format!("{ARCH} runs {STEPS} steps (the checkpoint's own sigmas); another grid goes in options.sigmas")));
        }
        if req.sampler.is_some_and(|s| s != Sampler::Euler) {
            return Err(Error(format!("{ARCH} samples with euler only")));
        }
        let sig = match req.extra.get("sigmas") {
            Some(v) => parse_sigmas(v).map_err(Error)?,
            None => SIGMAS.to_vec(),
        };
        let asked = match (req.width, req.height) {
            (Some(w), Some(h)) => Some(round_canvas(w, h)),
            (None, None) => None,
            _ => return Err(Error("width and height go together, or neither".into())),
        };
        if let Some((w, h)) = asked {
            if w < 256 || h < 256 || w > 4096 || h > 4096 || w * h > MAX_PIXELS {
                return Err(Error(format!("{w}x{h}: sides from 256 to 4096 (rounded to multiples of 32), at most {MAX_PIXELS} pixels (8 MP: 2912x2912, \
                                          3360x2496, 3552x2368, 3872x2176 and their turns; 4 MP: 2048x2048, 2368x1760, 2496x1664, 2720x1536 ...)")));
            }
        }
        let of = (sig.len() - 1) as u32;
        let mut log = |l: String| eprintln!("{l}");
        let (ctx, tags, refs, canvas) = match &req.edit {
            None => {
                let c = self.prompt_condition(&req.prompt, &mut log).map_err(ie)?;
                ((*c).clone(), None, Vec::new(), asked.unwrap_or((d.width, d.height)))
            }
            Some(ed) => {
                if ed.mask.is_some() {
                    return Err(Error(format!("{ARCH} edits by instruction: it takes no mask")));
                }
                if ed.strength < 1.0 || !ed.condition {
                    return Err(Error(format!("{ARCH} edits by instruction from the pictures as references: no img2img (strength below 1, reference false)")));
                }
                let pics: Vec<&Picture> = std::iter::once(&ed.image).chain(ed.refs.iter()).collect();
                if pics.len() > MAX_PICTURES {
                    return Err(Error(format!("an edit takes at most {MAX_PICTURES} pictures, got {}", pics.len())));
                }
                let (refs, pieces, canvas) = self.edit_inputs(&pics, &req.prompt, asked, &mut log).map_err(ie)?;
                let (c, tags) = self.condition(&pieces, &mut log).map_err(ie)?;
                (c, tags, refs, canvas)
            }
        };
        let (w, h) = (canvas.0 as usize, canvas.1 as usize);
        log(format!("{ARCH}: {w}x{h}, {} text tokens, {} picture(s), {} steps, conditioned in {:.1} s", ctx.len() / self.connector.out, refs.len(), of,
                    t0.elapsed().as_secs_f64()));
        let mut out = Vec::new();
        for p in 0..req.n.max(1) {
            progress(Step { picture: p, at: 0, of, seconds: t0.elapsed().as_secs_f64() })?;
            let tp = Instant::now();
            let mut times = Vec::new();
            let mut last = Instant::now();
            let mut step = |i: usize| {
                times.push(last.elapsed().as_secs_f64());
                last = Instant::now();
                progress(Step { picture: p, at: i as u32 + 1, of, seconds: t0.elapsed().as_secs_f64() })
            };
            let px = self.picture(&ctx, tags.as_ref(), &refs, (w, h), req.seed + p as u64, &sig, &mut step).map_err(|e| Error(nextcuda_image::tag_device_oom(&e.0, None)))?;
            log(format!("{ARCH}: picture {} in {:.1} s (steps {})", p + 1, tp.elapsed().as_secs_f64(),
                        times.iter().map(|s| format!("{s:.2}")).collect::<Vec<_>>().join(" ")));
            let plane = w * h;
            let mut data = vec![0u8; plane * 3];
            for i in 0..plane {
                for c in 0..3 {
                    data[i * 3 + c] = (px[c * plane + i] * 255.0).round().clamp(0.0, 255.0) as u8;
                }
            }
            out.push(Picture { width: w as u32, height: h as u32, channels: 3, data });
        }
        Ok(out)
    }

    fn report(&self) -> Vec<String> {
        let mut v = vec![format!("{ARCH} in Rust and CUDA: the denoiser over {}, the text encoder and the decoder on {} (loaded {:.0} s ago)",
                                 self.e.devs.iter().map(|d| d.name()).collect::<Vec<_>>().join(" + "), self.te_dev.name(), self.loaded.elapsed().as_secs_f64())];
        for d in &self.e.devs {
            v.push(format!("{}: {:.2} GiB of {:.2} held", d.name(), gib(d.mem_used()), gib(d.mem_cap())));
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_follow_diffusers() {
        // _size_at_area: a 1000 x 1000 picture is within 20% of a megapixel but not on the grid
        assert_eq!(size_at_area(1000, 1000, VISION_PIXELS), (992, 992));
        assert_eq!(size_at_area(1024, 1024, VISION_PIXELS), (1024, 1024));
        assert_eq!(size_at_area(4000, 3000, VISION_PIXELS), (1152, 864));
        // _size_in_area_range: a 4 MP picture keeps its size between 1 and 8 MP; a small one grows to 4 MP as a canvas
        assert_eq!(size_in_area_range(2048, 2048, REF_MIN, REF_MAX), (2048, 2048));
        assert_eq!(size_in_area_range(512, 512, CANVAS_MIN, REF_MAX), (2048, 2048));
        assert_eq!(size_in_area_range(800, 600, REF_MIN, REF_MAX), (1184, 896));
        assert_eq!(py_round(2.5), 2.0);
        assert_eq!(py_round(3.5), 4.0);
        assert_eq!(py_round(-2.5), -2.0);
    }

    #[test]
    fn a_given_grid_is_checked_and_sizes_round_to_the_grid() {
        assert_eq!(parse_sigmas("1, 0.9, 0.5, 0").unwrap(), vec![1.0, 0.9, 0.5, 0.0]);
        assert!(parse_sigmas("1, 0.5").is_err());
        assert!(parse_sigmas("1, 0.5, 0.7, 0").is_err());
        assert_eq!(round_canvas(1000, 1000), (992, 992));
        // Python rounds a half to the even neighbour: 2896 / 32 = 90.5 -> 90
        assert_eq!(round_canvas(2912, 2896), (2912, 2880));
    }

    #[test]
    fn the_presets_give_the_cards_sizes() {
        // a page's size: the budget at the ratio, each side rounded to the grid
        let fit = |px: u32, r: f64| {
            let side = |e: f64| ((e / GRID as f64).round() as u32) * GRID;
            (side((px as f64 * r).sqrt()), side((px as f64 / r).sqrt()))
        };
        assert_eq!(fit(8_388_608, 1.0), (2912, 2912));
        assert_eq!(fit(8_388_608, 4.0 / 3.0), (3360, 2496));
        assert_eq!(fit(4_194_304, 1.0), (2048, 2048));
        assert!(2912 * 2912 <= MAX_PIXELS && 3872 * 2176 <= MAX_PIXELS && 3360 * 2496 <= MAX_PIXELS && 3552 * 2368 <= MAX_PIXELS);
    }
}
