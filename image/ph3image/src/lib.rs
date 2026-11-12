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
//!
//! A `turbo-lora` file (the card p-h3-image-pdmd2: pDMD's 2-NFE LoRA, rank 128 on every block's four linears and the
//! text refiner's) rides beside the Q4_K blocks as low-rank side paths - `out += B (A x)`, bfloat16, resident on each
//! block's card - rather than merged: a merge would expand each Q4_K matrix, add and quantize again, and 4-bit steps
//! are coarser than most of the delta. With it the grid is the flow's shift 12 over N + 1 even levels (2 steps:
//! 1, 0.923, 0), no guidance, about 1 MP. An edit's strength below 1 starts from the first picture, encoded and noised
//! to that level (`(1 - s0) z + s0 noise`; pDMD's edit pass keeps the picture at s0 0.4 to 0.55), as a plain img2img
//! (reference false) or with the pictures also the edit's references.

pub mod check;
pub mod connector;
pub mod deblock;
pub mod describe;
pub mod sampler;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use nextcuda_h3::denoiser::{self, Conditions, Denoiser, RefIn, Schedule, Shape, TextRefiner};
use nextcuda_h3::device::{Device, Tensor};
use nextcuda_h3::dtype::DType;
use nextcuda_h3::gguf::{GType, Gguf};
use nextcuda_h3::jobs;
use nextcuda_h3::lora::{self, LoraSet};
use nextcuda_h3::safetensors::Checkpoint;
use nextcuda_h3::te::{Resident, TextEncoder};
use nextcuda_h3::vision::{Frames, Piece, Presented};
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
/// a few-step LoRA's file role (pDMD 2-NFE), resident beside the blocks
pub const TURBO: &str = "turbo-lora";
/// with it: the flow's shift, the default steps and the most a request may ask for (on the same grid)
pub const TURBO_SHIFT: f32 = 12.0;
pub const TURBO_STEPS: u32 = 2;
pub const TURBO_MAX_STEPS: u32 = 8;
/// with it: the canvas an edit takes when none is asked, and its pictures' size for the denoiser (about 1 MP: a live
/// editor's frame)
const TURBO_PIXELS: f64 = 1_048_576.0;
/// conditionings and pictures kept for the next requests (a live editor sends the same picture with a prompt typed on)
const KEEP_CONDS: usize = 16;
/// with it: the room kept on each card beside the blocks (NC_PH3_RESERVE_GIB; 2 without it)
const TURBO_RESERVE_GIB: f64 = 1.5;
const KEEP_PICS: usize = 2;

/// with it: the sampler's eta when a request names none (sampler.rs: 1, all fresh noise at each step - ComfyUI's
/// SamplerLCM, the sampler distribution-matching distillation steps with; eta 0 is flow Euler, as before)
pub const TURBO_ETA: f32 = 1.0;

/// A request's options: another sampling grid than the checkpoint's, the sampler's eta and s_noise (also the
/// request's own fields eta and s_noise)
pub const OPTIONS: &[EngineOption] = &[
    EngineOption {
        name: "sigmas",
        env: "",
        value: "S,S,...",
        help: "the sampling grid instead of the checkpoint's 8 steps: strictly decreasing from at most 1 down to 0 (one step between two)",
        at: At::Request,
    },
    EngineOption {
        name: "eta",
        env: "",
        value: "0..1",
        help: "how much of each step's noise is fresh: 1 (with the pDMD LoRA the default; sampler lcm) x0 then new noise at the next level, 0 (sampler euler; the base model's default) deterministic flow Euler; clamped to 0..1",
        at: At::Request,
    },
    EngineOption {
        name: "s_noise",
        env: "",
        value: "0..2",
        help: "the fresh noise's scale (1 by default; ComfyUI's s_noise); clamped to 0..2",
        at: At::Request,
    },
    EngineOption {
        name: "deblock",
        env: "",
        value: "true|false",
        help: "the deblock filter after the decoder (deblock.rs; on by default): the decoder's 16-pixel tile pattern subtracted, then a flatness-gated pass over the tile boundaries",
        at: At::Request,
    },
    EngineOption {
        name: "deblock_strength",
        env: "",
        value: "0..1",
        help: "how much of the deblock filter's correction is applied (1 by default; 0 is off); clamped to 0..1",
        at: At::Request,
    },
];

/// The deblock filter's strength when a request names none and leaves it on (deblock.rs)
pub const DEBLOCK_STRENGTH: f32 = 1.0;

/// A request's deblock filter: its strength, or None when off. `deblock` true / false (also 1 / 0, on / off), and
/// `deblock_strength` 0..1 (clamped; 0 is off; given alone it turns the filter on); neither: `default_on`.
pub fn deblock_setting(get: &dyn Fn(&str) -> Option<String>, default_on: bool) -> std::result::Result<Option<f32>, String> {
    let on = match get("deblock").map(|v| v.trim().to_ascii_lowercase()) {
        None => None,
        Some(v) => Some(match v.as_str() {
            "true" | "1" | "on" | "yes" => true,
            "false" | "0" | "off" | "no" => false,
            _ => return Err(format!("deblock {v:?}: true or false")),
        }),
    };
    let strength = get("deblock_strength")
        .map(|v| v.trim().parse::<f32>().ok().filter(|x| x.is_finite()).map(|x| x.clamp(0.0, 1.0)).ok_or(format!("deblock_strength {v:?}: a number")))
        .transpose()?;
    let on = on.unwrap_or(strength.is_some() || default_on);
    Ok(Some(strength.unwrap_or(DEBLOCK_STRENGTH)).filter(|s| on && *s > 0.0))
}

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
        // a turbo entry runs about 1 MP: less kept for the buffers, room for the LoRA beside the blocks
        None if files.contains_key(TURBO) => TURBO_RESERVE_GIB,
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

/// An edit's instruction as the encoder reads it: after "Edit <Picture 1>. " unless it starts so already (or is an
/// upsampler's brief, "subject_definitions: ...")
pub fn edit_prompt(prompt: &str) -> String {
    let lead = prompt.trim_start();
    if lead.starts_with("Edit <Picture 1>.") || lead.starts_with("subject_definitions:") { prompt.to_string() } else { format!("Edit <Picture 1>. {prompt}") }
}

/// The few-step grid: `steps` + 1 even levels of t from `top` down to 0 through the flow's shift (diffusers'
/// set_shift(12), set_timesteps(steps + 1)); `top` is a level already shifted (1, or an img2img's start)
pub fn shift_grid(steps: u32, shift: f32, top: f32) -> Vec<f32> {
    let n = steps.max(1);
    // the t whose shifted level is `top`
    let t0 = top / (shift - (shift - 1.0) * top);
    let mut s: Vec<f32> = (0..n).map(|i| denoiser::shifted(t0 * (1.0 - i as f32 / n as f32), shift)).collect();
    s[0] = top;
    s.push(0.0);
    s
}

/// A grid started at `s0` (an img2img's strength): `s0`, then the grid's levels below it
pub fn start_at(grid: &[f32], s0: f32) -> Vec<f32> {
    std::iter::once(s0).chain(grid.iter().copied().filter(|s| *s < s0)).collect()
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
    /// the same card, for the deblock filter's kernels (deblock.rs)
    px_gpu: Arc<Gpu>,
    loaded: Instant,
    load_s: f64,
    busy: Mutex<()>,
    /// the turbo LoRA's side paths on the blocks and the text refiner, resident (the `turbo-lora` file)
    lora: Option<LoraSet>,
    /// the last conditionings, newest last: by prompt (text to image) or pictures and prompt (an edit)
    conds: Mutex<Vec<(String, Arc<Cond>)>>,
    /// the last edits' pictures, newest last
    pics: Mutex<Vec<Arc<PicSet>>>,
}

/// A prompt's conditioning: the connector's [L, 5120], the tokens' modality tags (with pictures in it) and, once a
/// picture has been made from it, the text refiner's tokens (bfloat16 bytes [L, hidden]: the refiner is loaded only
/// for a new prompt)
struct Cond {
    ctx: Vec<f32>,
    tags: Option<Vec<i32>>,
    refined: Mutex<Option<Arc<Vec<u8>>>>,
}

/// What an edit's pictures give every prompt about them: their latents for the denoiser, their presentation to the
/// text encoder up to the instruction (the vision tower's tokens) and the canvas; an img2img's start, encoded once
struct PicSet {
    key: u64,
    refs: Vec<Encoded>,
    prefix: Option<Presented>,
    canvas: (u32, u32),
    start: Mutex<Option<Arc<Vec<f32>>>>,
}

/// The pictures of a request and what is asked of them, as one number
fn pics_key(pics: &[&Picture], size: Option<(u32, u32)>, condition: bool) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    (size, condition, pics.len()).hash(&mut h);
    for p in pics {
        (p.width, p.height, p.channels).hash(&mut h);
        p.data.hash(&mut h);
    }
    h.finish()
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
        // the turbo LoRA's matrices are counted beside each block when the blocks are placed over the cards
        let turbo = files.get(TURBO).map(|p| Checkpoint::open(p)).transpose()?;
        if let Some(ck) = &turbo {
            e.side = (0..e.cfg.blocks).map(|i| lora::block_bytes(ck, "blocks", i)).collect();
        }
        let t1 = Instant::now();
        let b = e.blocks()?;
        let place: Vec<String> = devs.iter().enumerate().map(|(d, dev)| format!("{} on {}", b.dev_of.iter().filter(|x| **x == d).count(), dev.name())).collect();
        log(format!("blocks : {} ({:.2} GiB) in {:.1} s: {}", b.blocks.len(), gib(b.load_bytes), t1.elapsed().as_secs_f64(), place.join(", ")));
        let lora = match &turbo {
            Some(ck) => {
                let t1 = Instant::now();
                let l = LoraSet::load_placed(&b.devs, &|i| b.card_of(i), ck, 1.0, b.blocks.len(), 2)?;
                let want = 4 * (b.blocks.len() + 2);
                if l.layers != want {
                    return Err(nextcuda_h3::Error(format!("{}: {} of the {want} layers it should hold (every block's and the text refiner's four linears)",
                                                          ck.path.display(), l.layers)));
                }
                log(format!("lora   : {} ({} layers, rank {}, {:.2} GiB) in {:.1} s", ck.path.display(), l.layers, l.max_rank,
                            gib(e.side.iter().sum()), t1.elapsed().as_secs_f64()));
                Some(l)
            }
            None => None,
        };
        drop(b);
        for d in &devs {
            log(format!("memory : {} holds {:.2} GiB of its {:.2}", d.name(), gib(d.mem_used()), gib(d.mem_cap())));
        }
        let px_gpu = match gpus.last() {
            Some(g) => g.clone(),
            None => Gpu::open(te_dev.index).map_err(|e| nextcuda_h3::Error(e.0))?,
        };
        Ok(PH3Image { e, te, resident, final_norm, connector, vae, vae_path, tower, tok, te_dev, px_gpu, loaded: Instant::now(), load_s: t0.elapsed().as_secs_f64(),
                      busy: Mutex::new(()), lora, conds: Mutex::new(Vec::new()), pics: Mutex::new(Vec::new()) })
    }

    /// A request's sampler as (eta, s_noise), clamped: eta as given (the field or options.eta), else the sampler named
    /// (euler 0, lcm 1), else the model's (the pDMD LoRA's TURBO_ETA, the base model's 0)
    fn eta_for(&self, req: &ImageRequest) -> std::result::Result<(f32, f32), String> {
        let num = |k: &str| req.extra.get(k).map(|v| v.trim().parse::<f32>().ok().filter(|x| x.is_finite()).ok_or(format!("{k} {v:?}: a number"))).transpose();
        if let Some(s) = req.sampler.filter(|s| !matches!(s, Sampler::Euler | Sampler::Lcm)) {
            return Err(format!("{ARCH} samples with lcm (eta 1) or euler (eta 0), eta between them; not {}", s.name()));
        }
        let eta = match (num("eta")?, req.sampler) {
            (Some(e), _) => e,
            (None, Some(Sampler::Euler)) => 0.0,
            (None, Some(_)) => 1.0,
            (None, None) if self.turbo() => TURBO_ETA,
            (None, None) => 0.0,
        };
        Ok((sampler::clamp_eta(eta), sampler::clamp_s_noise(num("s_noise")?.unwrap_or(1.0))))
    }

    /// A request's deblock strength, or None when off (`deblock_setting`; on by default)
    fn deblock_for(&self, req: &ImageRequest) -> std::result::Result<Option<f32>, String> {
        deblock_setting(&|k| req.extra.get(k).cloned(), true)
    }

    fn turbo(&self) -> bool {
        self.lora.is_some()
    }

    /// A conditioning from the last few kept, else made by `make` and kept
    fn cond_cached(&self, key: String, make: &mut dyn FnMut() -> nextcuda_h3::Result<(Vec<f32>, Option<Vec<i32>>)>) -> nextcuda_h3::Result<Arc<Cond>> {
        if let Some((_, c)) = self.conds.lock().unwrap().iter().find(|(k, _)| *k == key) {
            return Ok(c.clone());
        }
        let (ctx, tags) = make()?;
        let c = Arc::new(Cond { ctx, tags, refined: Mutex::new(None) });
        let mut list = self.conds.lock().unwrap();
        list.push((key, c.clone()));
        if list.len() > KEEP_CONDS {
            list.remove(0);
        }
        Ok(c)
    }

    /// The prompt's pieces through the text encoder and the connector: the conditioning [L, 5120] and, with pictures in
    /// it, each token's modality tag (a vision block's tokens run on the video tables)
    fn condition(&self, pieces: &[Piece], log: &mut dyn FnMut(String)) -> nextcuda_h3::Result<(Vec<f32>, Option<Vec<i32>>)> {
        let presented = nextcuda_h3::vision::present_with(pieces, &self.tok, Some((&self.tower, "model.visual.")), None, self.e.dev.index, log)?;
        self.encode(&presented)
    }

    fn encode(&self, presented: &Presented) -> nextcuda_h3::Result<(Vec<f32>, Option<Vec<i32>>)> {
        let taps = self.te.encode_resident_taps(&self.resident, &presented.ids, &presented.seen, &presented.embeds, &TAPS, Some(&self.final_norm),
                                                &mut |_| Ok(()))?;
        let ctx = self.connector.forward(&taps)?;
        Ok((ctx, presented.has_vision().then(|| presented.tags.clone())))
    }

    /// A text-to-image prompt's conditioning, from the last few kept
    fn prompt_condition(&self, prompt: &str, log: &mut dyn FnMut(String)) -> nextcuda_h3::Result<Arc<Cond>> {
        self.cond_cached(format!("t2i\u{0}{prompt}"), &mut || self.condition(&[Piece::Text(prompt.to_string())], log))
    }

    /// An edit's conditioning: its pictures' presentation (kept) with the instruction's tokens after it - the encoder is
    /// causal and the instruction comes last, so this is the whole prompt's presentation, the tower not run again
    fn edit_condition(&self, set: &PicSet, prompt: &str) -> nextcuda_h3::Result<Arc<Cond>> {
        let prefix = set.prefix.as_ref().ok_or("an edit's pictures were kept without their presentation")?;
        self.cond_cached(format!("edit\u{0}{:x}\u{0}{prompt}", set.key), &mut || {
            let ids = self.tok.encode(&edit_prompt(prompt))?;
            let mut p = Presented { ids: prefix.ids.clone(), tags: prefix.tags.clone(), seen: Vec::new(), embeds: Vec::new() };
            p.tags.extend(std::iter::repeat_n(1, ids.len()));
            p.ids.extend(ids);
            // the tower's tokens borrowed for the encode, not copied
            let taps = self.te.encode_resident_taps(&self.resident, &p.ids, &prefix.seen, &prefix.embeds, &TAPS, Some(&self.final_norm), &mut |_| Ok(()))?;
            Ok((self.connector.forward(&taps)?, Some(p.tags)))
        })
    }

    /// An edit's pictures, from the last few kept, else encoded (and kept): with `condition` their latents as the
    /// edit's references and their presentation; the canvas (`size`, else the first picture's)
    fn pic_set(&self, pics: &[&Picture], size: Option<(u32, u32)>, condition: bool, log: &mut dyn FnMut(String)) -> nextcuda_h3::Result<Arc<PicSet>> {
        let key = pics_key(pics, size, condition);
        if let Some(s) = self.pics.lock().unwrap().iter().find(|s| s.key == key) {
            log(format!("{ARCH}: the pictures as kept from the last request"));
            return Ok(s.clone());
        }
        let set = Arc::new(match condition {
            true => {
                let (refs, mut pieces, canvas) = self.edit_inputs(pics, size, log)?;
                pieces.pop();
                let prefix = nextcuda_h3::vision::present_with(&pieces, &self.tok, Some((&self.tower, "model.visual.")), None, self.e.dev.index, log)?;
                PicSet { key, refs, prefix: Some(prefix), canvas, start: Mutex::new(None) }
            }
            false => {
                let canvas = size.unwrap_or_else(|| if self.turbo() { size_at_area(pics[0].width, pics[0].height, TURBO_PIXELS) }
                                                     else { size_in_area_range(pics[0].width, pics[0].height, CANVAS_MIN, REF_MAX) });
                PicSet { key, refs: Vec::new(), prefix: None, canvas, start: Mutex::new(None) }
            }
        });
        let mut list = self.pics.lock().unwrap();
        list.push(set.clone());
        if list.len() > KEEP_PICS {
            list.remove(0);
        }
        Ok(set)
    }

    /// An img2img's start: the first picture's latents at the canvas (the first reference's when it is that size)
    fn start_latent(&self, set: &PicSet, first: &Picture) -> nextcuda_h3::Result<Arc<Vec<f32>>> {
        let mut s = set.start.lock().unwrap();
        if let Some(z) = s.as_ref() {
            return Ok(z.clone());
        }
        let (w, h) = set.canvas;
        let z = match set.refs.first().filter(|r| (r.w * 16, r.h * 16) == (w as usize, h as usize)) {
            Some(r) => r.latent.clone(),
            None => {
                let venc = nextcuda_h3::venc::VideoEncoder::load(&self.te_dev, &Checkpoint::open(&self.vae_path)?)?;
                venc.encode_sampled(&pixels_at(first, (w, h)), h as usize, w as usize, ENCODE_SEED, &mut || Ok(()))?
            }
        };
        let z = Arc::new(z);
        *s = Some(z.clone());
        Ok(z)
    }

    /// One picture: the denoiser over `refs` and the text, the starting noise and the references' augmentation from
    /// one torch generator seeded `seed` (as diffusers draws them: each reference's noise, then the picture's, then the
    /// audio's), Euler over `sig`, the decoder. RGB float32 [3, H, W] in [0, 1]. `start`: an img2img's latents, noised
    /// to `sig[0]` with the picture's noise. `tick(Some(i))` after step i, `tick(None)` between blocks and stages (where
    /// a cancel stops it).
    /// `eta` above 0: the sampler's (sampler.rs) fresh noise at each step from the same generator, after the starting
    /// noise; 0: Euler, as before.
    #[allow(clippy::too_many_arguments)]
    fn picture(&self, cond: &Cond, refs: &[Encoded], (w, h): (usize, usize), seed: u64, sig: &[f32], start: Option<&[f32]>,
               (eta, s_noise): (f32, f32), tick: &mut dyn FnMut(Option<usize>) -> Result<()>) -> nextcuda_h3::Result<Vec<f32>> {
        let e = &self.e;
        let model = e.blocks()?;
        let l = cond.ctx.len() / self.connector.out;
        let refined = cond.refined.lock().unwrap().clone();
        let text = match refined {
            Some(b) => Tensor::from_bytes(&e.dev, DType::BF16, &[l, e.cfg.hidden], &b)?,
            None => {
                let refiner = TextRefiner::load(&e.dev, &e.ck, e.threads)?;
                let t = Tensor::from_bytes(&e.dev, DType::BF16, &[l, self.connector.out], &denoiser::bf16_bytes(&cond.ctx))?;
                let r = refiner.run(&e.cfg, &t, self.lora.as_ref().map(|x| x.refiner.as_slice()))?;
                e.dev.wait()?;
                *cond.refined.lock().unwrap() = Some(Arc::new(denoiser::bf16_bytes(&r.to_f32()?)));
                r
            }
        };
        let ie = |x: Error| nextcuda_h3::Error(x.0);
        tick(None).map_err(ie)?;
        let mut g = nextcuda_h3::noise::Mt19937::new(seed);
        let mut c = Conditions { seed, visual_aug: NOISE_AUG, mixed: true, ..Conditions::default() };
        for r in refs {
            let n = g.randn(r.latent.len());
            let latent = r.latent.iter().zip(&n).map(|(z, n)| NOISE_AUG * z + (1.0 - NOISE_AUG) * n).collect();
            c.refs.push(RefIn::Image { latent, h: r.h, w: r.w });
        }
        let shape = Shape { t: 1, h: h / 16, w: w / 16, audio_t: AUDIO_T };
        let noise_v = g.randn(24 * shape.h * shape.w);
        // diffusers draws the audio rows as [channel-major rows, 32 features]: here [32 features, 2 channels, frames]
        let rows = g.randn(2 * AUDIO_T * e.outer.audio_c);
        let mut noise_a = vec![0f32; rows.len()];
        for r in 0..2 * AUDIO_T {
            for ch in 0..e.outer.audio_c {
                noise_a[ch * 2 * AUDIO_T + r] = rows[r * e.outer.audio_c + ch];
            }
        }
        // the audio steps down the same grid as the picture, unscaled (diffusers' audio_scheduler, also unshifted)
        let schedule = Schedule { shift_video: 1.0, shift_audio: 1.0, audio_scale: 1.0 };
        let mut d = Denoiser::new(&model, &e.outer, text, cond.tags.clone(), shape, schedule, &c)?;
        d.lora = self.lora.as_ref();
        let s0 = sig[0];
        let xv: Vec<f32> = match start {
            Some(z) if z.len() == noise_v.len() => z.iter().zip(&noise_v).map(|(z, n)| (1.0 - s0) * z + s0 * n).collect(),
            Some(z) => return Err(nextcuda_h3::Error(format!("an img2img start of {} latents for a {w}x{h} canvas", z.len()))),
            None => noise_v.iter().map(|n| n * s0).collect(),
        };
        let xa: Vec<f32> = noise_a.iter().map(|n| n * s0).collect();
        let tick = std::cell::RefCell::new(tick);
        let v = if eta <= 0.0 {
            denoiser::sample_from(&mut d, xv, xa, &noise_v, sig, None, &mut |i, _, _| (tick.borrow_mut())(Some(i)).map_err(ie),
                                  &mut |_, b| if b % 10 == 9 { (tick.borrow_mut())(None).map_err(ie) } else { Ok(()) })?.0
        } else {
            let mut xs = vec![xv, xa];
            sampler::run(&mut xs, sig, eta, s_noise,
                         &mut |xs, s, _| {
                             let (dv, da) = d.denoised(&xs[0], &xs[1], s, &mut |b| if b % 10 == 9 { (tick.borrow_mut())(None).map_err(ie) } else { Ok(()) })?;
                             Ok(vec![dv, da])
                         },
                         &mut |n| g.randn(n),
                         &mut |i| (tick.borrow_mut())(Some(i)).map_err(ie))?;
            xs.swap_remove(0)
        };
        drop(d);
        drop(model);
        (tick.borrow_mut())(None).map_err(ie)?;
        let (px, _) = self.vae.decode(&v, 1, shape.h, shape.w, &mut || Ok(()))?;
        Ok(px)
    }

    /// An edit's pictures: their latents for the denoiser (at their own sizes, sampled as diffusers encodes them) and
    /// their prompt pieces for the text encoder up to the instruction; the canvas (`size`, else the first picture's at
    /// 4 to 8 MP - with the turbo LoRA about 1 MP, the pictures too)
    #[allow(clippy::type_complexity)]
    fn edit_inputs(&self, pics: &[&Picture], size: Option<(u32, u32)>, log: &mut dyn FnMut(String))
                   -> nextcuda_h3::Result<(Vec<Encoded>, Vec<Piece>, (u32, u32))> {
        let (lo, hi, canvas_lo) = if self.turbo() { (TURBO_PIXELS, TURBO_PIXELS, TURBO_PIXELS) } else { (REF_MIN, REF_MAX, CANVAS_MIN.max(REF_MIN)) };
        let mut ref_sizes: Vec<(u32, u32)> = pics.iter().map(|p| size_in_area_range(p.width, p.height, lo, hi)).collect();
        let canvas = match size {
            Some(s) => s,
            None => {
                ref_sizes[0] = size_in_area_range(pics[0].width, pics[0].height, canvas_lo, hi);
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
        Ok((refs, pieces, canvas))
    }
}

impl ImageEngine for PH3Image {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn defaults(&self) -> Defaults {
        match self.turbo() {
            true => Defaults { width: 1024, height: 1024, steps: TURBO_STEPS, cfg: 1.0, sampler: Sampler::Lcm, schedule: Sched::Shift, shift: TURBO_SHIFT },
            false => Defaults { width: 2912, height: 2912, steps: STEPS, cfg: 1.0, sampler: Sampler::Euler, schedule: Sched::Shift, shift: 1.0 },
        }
    }
    fn edits(&self) -> bool {
        true
    }
    /// lcm (eta 1) and euler (eta 0): the two ends of the one sampler (sampler.rs), `eta` between them
    fn samplers(&self) -> Vec<Sampler> {
        match self.turbo() {
            true => vec![Sampler::Lcm, Sampler::Euler],
            false => vec![Sampler::Euler, Sampler::Lcm],
        }
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
    /// 8 steps (the checkpoint's grid); another count is refused, another grid comes as options.sigmas. With the turbo
    /// LoRA 1 to 8 on its shifted grid (2 by default)
    fn steps_free(&self) -> bool {
        self.turbo()
    }
    fn rendered_prompt(&self, req: &ImageRequest) -> String {
        match &req.edit {
            Some(ed) if ed.condition => edit_prompt(&req.prompt),
            _ => req.prompt.clone(),
        }
    }
    fn steps_for(&self, req: &ImageRequest) -> u32 {
        let own = if self.turbo() { req.steps.unwrap_or(TURBO_STEPS) } else { STEPS };
        req.extra.get("sigmas").and_then(|v| parse_sigmas(v).ok()).map_or(own, |s| s.len() as u32 - 1)
    }
    fn run_info(&self, req: &ImageRequest) -> Vec<(&'static str, String)> {
        let mut v = match self.eta_for(req) {
            Ok((eta, s_noise)) => vec![("sampler", if eta > 0.0 { "lcm" } else { "euler" }.into()), ("eta", eta.to_string()), ("s_noise", s_noise.to_string())],
            Err(_) => Vec::new(),
        };
        match self.deblock_for(req) {
            Ok(Some(s)) => v.extend([("deblock", "on".to_string()), ("deblock_strength", s.to_string())]),
            Ok(None) => v.push(("deblock", "off".to_string())),
            Err(_) => {}
        }
        v
    }

    fn generate(&self, req: &ImageRequest, progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<Vec<Picture>> {
        let _one = self.busy.lock().unwrap_or_else(|p| p.into_inner());
        let t0 = Instant::now();
        let d = self.defaults();
        if req.negative.is_some() {
            return Err(Error(format!("{ARCH} takes no negative prompt: it is guidance-distilled (one pass a step, no CFG)")));
        }
        if req.cfg.is_some_and(|c| c != 1.0) {
            return Err(Error(format!("{ARCH} takes no cfg (guidance scale): it is guidance-distilled - leave cfg out, or 1")));
        }
        if !req.loras.is_empty() {
            return Err(Error(format!("{ARCH} takes no LoRAs")));
        }
        match (self.turbo(), req.steps) {
            (true, Some(s)) if !(1..=TURBO_MAX_STEPS).contains(&s) => {
                return Err(Error(format!("{ARCH} with the pDMD LoRA runs 1 to {TURBO_MAX_STEPS} steps on its shift-{TURBO_SHIFT} grid (2 by default)")));
            }
            (false, Some(s)) if s != STEPS => {
                return Err(Error(format!("{ARCH} runs {STEPS} steps (the checkpoint's own sigmas); another grid goes in options.sigmas")));
            }
            _ => {}
        }
        let noise = self.eta_for(req).map_err(Error)?;
        let deblock = self.deblock_for(req).map_err(Error)?;
        let given = req.extra.get("sigmas").map(|v| parse_sigmas(v)).transpose().map_err(Error)?;
        let steps = req.steps.unwrap_or(d.steps);
        // an img2img's start: the edit's strength below 1
        let s0 = req.edit.as_ref().map(|e| e.strength).filter(|s| *s < 1.0);
        let sig = match (&given, s0) {
            (Some(g), None) => g.clone(),
            (Some(g), Some(s)) => start_at(g, s),
            (None, s) if self.turbo() => shift_grid(steps, TURBO_SHIFT, s.unwrap_or(1.0)),
            (None, None) => SIGMAS.to_vec(),
            (None, Some(s)) => start_at(&SIGMAS, s),
        };
        if sig.len() < 2 || sig[0] <= 0.0 {
            return Err(Error(format!("strength {}: no step left on the grid below it", s0.unwrap_or(1.0))));
        }
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
        // a cancel is seen between the stages, not only at the steps
        progress(Step { picture: 0, at: 0, of, seconds: 0.0 })?;
        let (cond, set) = match &req.edit {
            None => (self.prompt_condition(&req.prompt, &mut log).map_err(ie)?, None),
            Some(ed) => {
                if ed.mask.is_some() {
                    return Err(Error(format!("{ARCH} edits by instruction: it takes no mask")));
                }
                let pics: Vec<&Picture> = std::iter::once(&ed.image).chain(ed.refs.iter()).collect();
                if pics.len() > MAX_PICTURES {
                    return Err(Error(format!("an edit takes at most {MAX_PICTURES} pictures, got {}", pics.len())));
                }
                if !ed.condition && pics.len() > 1 {
                    return Err(Error(format!("{ARCH}: a plain img2img (reference false) starts from one picture, got {}", pics.len())));
                }
                let set = self.pic_set(&pics, asked, ed.condition, &mut log).map_err(ie)?;
                progress(Step { picture: 0, at: 0, of, seconds: t0.elapsed().as_secs_f64() })?;
                let cond = match ed.condition {
                    true => self.edit_condition(&set, &req.prompt).map_err(ie)?,
                    false => self.prompt_condition(&req.prompt, &mut log).map_err(ie)?,
                };
                (cond, Some(set))
            }
        };
        let start = match (&set, s0) {
            (Some(set), Some(_)) => Some(self.start_latent(set, &req.edit.as_ref().ok_or("no edit")?.image).map_err(ie)?),
            _ => None,
        };
        let refs: &[Encoded] = set.as_ref().map_or(&[], |s| s.refs.as_slice());
        let canvas = set.as_ref().map(|s| s.canvas).or(asked).unwrap_or((d.width, d.height));
        let (w, h) = (canvas.0 as usize, canvas.1 as usize);
        log(format!("{ARCH}: {w}x{h}, {} text tokens, {} picture(s), {} steps on {:?}{}, eta {} s_noise {}, conditioned in {:.2} s",
                    cond.ctx.len() / self.connector.out, refs.len(), of, sig, if self.turbo() { " (pDMD LoRA)" } else { "" }, noise.0, noise.1,
                    t0.elapsed().as_secs_f64()));
        let mut out = Vec::new();
        for p in 0..req.n.max(1) {
            progress(Step { picture: p, at: 0, of, seconds: t0.elapsed().as_secs_f64() })?;
            let tp = Instant::now();
            let mut times = Vec::new();
            let mut last = Instant::now();
            let mut at = 0u32;
            let mut tick = |i: Option<usize>| {
                if let Some(i) = i {
                    times.push(last.elapsed().as_secs_f64());
                    last = Instant::now();
                    at = i as u32 + 1;
                }
                progress(Step { picture: p, at, of, seconds: t0.elapsed().as_secs_f64() })
            };
            let mut px = self.picture(&cond, refs, (w, h), req.seed + p as u64, &sig, start.as_deref().map(|v| v.as_slice()), noise, &mut tick)
                .map_err(|e| Error(nextcuda_image::tag_device_oom(&e.0, None)))?;
            if let Some(strength) = deblock {
                let td = Instant::now();
                let rep = deblock::gpu(&self.px_gpu, &mut px, w, h, strength).map_err(|e| Error(nextcuda_image::tag_device_oom(&e.0, None)))?;
                log(format!("{ARCH}: deblock {strength} in {:.1} ms (the tile pattern's RMS {:.2} {:.2} {:.2} levels)", td.elapsed().as_secs_f64() * 1e3,
                            rep.pattern_rms[0] * 255.0, rep.pattern_rms[1] * 255.0, rep.pattern_rms[2] * 255.0));
            }
            log(format!("{ARCH}: picture {} in {:.2} s (steps {})", p + 1, tp.elapsed().as_secs_f64(),
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
    fn deblock_is_on_by_default_and_off_skips_it() {
        let ask = |pairs: &[(&str, &str)]| {
            let m: std::collections::BTreeMap<String, String> = pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
            deblock_setting(&|k| m.get(k).cloned(), true)
        };
        assert_eq!(ask(&[]), Ok(Some(DEBLOCK_STRENGTH)));
        // off: None, so generate hands the decoder's pixels on untouched (bit-identical to before the filter)
        assert_eq!(ask(&[("deblock", "false")]), Ok(None));
        assert_eq!(ask(&[("deblock", "0")]), Ok(None));
        assert_eq!(ask(&[("deblock", "true"), ("deblock_strength", "0")]), Ok(None));
        assert_eq!(ask(&[("deblock_strength", "0.5")]), Ok(Some(0.5)));
        assert_eq!(ask(&[("deblock", "true"), ("deblock_strength", "3")]), Ok(Some(1.0)));
        assert_eq!(ask(&[("deblock", "false"), ("deblock_strength", "0.5")]), Ok(None));
        assert!(ask(&[("deblock", "maybe")]).is_err());
        assert!(ask(&[("deblock_strength", "x")]).is_err());
        assert_eq!(deblock_setting(&|_| None, false), Ok(None));
    }

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
    fn an_edit_is_rendered_after_its_prefix_once() {
        assert_eq!(edit_prompt("add a hat"), "Edit <Picture 1>. add a hat");
        assert_eq!(edit_prompt("  Edit <Picture 1>. add a hat"), "  Edit <Picture 1>. add a hat");
        assert_eq!(edit_prompt("subject_definitions: a fox"), "subject_definitions: a fox");
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
    fn the_turbo_grid_is_pdmds_and_an_img2img_starts_at_its_strength() {
        // diffusers' set_shift(12), set_timesteps(3): 1, 12 * 0.5 / (1 + 11 * 0.5), 0
        let g = shift_grid(2, TURBO_SHIFT, 1.0);
        assert_eq!(g.len(), 3);
        assert_eq!(g[0], 1.0);
        assert!((g[1] - 12.0 / 13.0).abs() < 1e-6);
        assert_eq!(g[2], 0.0);
        // from 0.55: the same spacing in t below the start, two steps
        let s = shift_grid(2, TURBO_SHIFT, 0.55);
        assert_eq!(s[0], 0.55);
        assert!(s[1] > 0.3 && s[1] < 0.4, "{s:?}");
        assert_eq!(s.len(), 3);
        // the checkpoint's grid from 0.55: 0.55, then its levels below
        assert_eq!(start_at(&SIGMAS, 0.55), vec![0.55, 0.528718, 0.0]);
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
