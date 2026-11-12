//! `image.describe`: a picture to the prompt that would draw it - nextsycl's contract (docs/api/image-describe.md
//! there), field for field, so a front end works against both boxes.
//!
//! - The describer is a chat model that reads pictures, already served: Qwen3.8-27B with its vision tower (the mmproj
//!   beside it, the qwen35 engine). Any chat model whose registry entry has an `mmproj` (or `vision`) file reads
//!   pictures; system.models lists `vision` and `describe` in its capabilities and `image.describe` in its methods.
//! - One instruction per image-model family (`target`), no thinking (`think_budget: 0`: the chat server closes the
//!   thinking before it starts), greedy by default; the answer cut at `PROMPT:` / `NEGATIVE:`.
//! - A job in the image queue (jobs.rs) like image.upscale: it takes an image slot, then asks the chat server through
//!   the switcher (`chat_target`: the room made first - idle models stopped, pinned and held ones never - then the
//!   describer loaded, or used as it is when loaded). The answer is streamed (/api/chat's lines), so a cancel closes
//!   the stream within a second and the chat server stops at its next token (while the picture is read: after it).
//!
//! - `describer`: "best" (the default: the chat model above), "native" (the target's own text encoder: an exact target
//!   model's, a family's enabled model with the smallest encoder), or a describer's id from image.describers - a chat
//!   model that reads pictures, or an image model's text encoder that does (`NATIVES`: Qwen3-VL-8B-Instruct from
//!   Qwen-Image 2.1's files, Qwen3-VL-4B-Instruct from P-H3-Image's), with any target. A text encoder describes in its
//!   image model's own server (POST /v1/images/describe: the pipeline dropped for it, loaded again by the next picture),
//!   reached as image.remove_background reaches its model: room planned, the lease, image.cancel through the server.
//!
//! What the answer carries: the prompt (and negative), the family and model it is for, the picture's size and the size
//! the describer saw, the size to draw it at on the target's grid (`suggested_size`), the tokens and the speeds, which
//! describer wrote it (`describer`).

use std::io::{Read, Write};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use super::life::s;
use super::registry;
use super::Home;
use crate::http::{Conn, Target};

type Fail = (u16, String);

/// The families a prompt is written for (`target`)
pub const FAMILIES: [&str; 8] = ["natural", "qwen-image", "z-image", "h3", "hunyuanimage", "moss-image", "ming-image", "tags"];

/// A whole describe may take this long (a swap included)
const MAX_DESCRIBE_S: u64 = 1800;

/// The families whose models read a negative prompt (true guidance)
pub fn reads_negative(family: &str) -> bool {
    matches!(family, "natural" | "qwen-image" | "hunyuanimage" | "moss-image" | "tags")
}

/// An image model's family by its id's prefix (a model not registered here: nextsycl's ids too)
pub fn family_by_prefix(id: &str) -> Option<&'static str> {
    let id = id.to_ascii_lowercase();
    let p = |x: &str| id.starts_with(x);
    Some(if p("qwen-image") {
        "qwen-image"
    } else if p("z-image") {
        "z-image"
    } else if p("p-h3-image") || p("minimax-h3-image") || p("h3-image") || p("sycl-h3-image") {
        "h3"
    } else if p("hunyuanimage") || p("hunyuan-image") {
        "hunyuanimage"
    } else if p("moss-image") {
        "moss-image"
    } else if p("ming-image") {
        "ming-image"
    } else if p("sdxl") || p("illustrious") || p("noobai") || p("pony") || p("animagine") {
        "tags"
    } else {
        return None;
    })
}

/// A registered model's family when it draws pictures from a prompt (by its architecture, else its id); None: it
/// does not draw (an OCR model, an upscaler, a vectorizer, a background remover, a chat or sound model)
pub fn family_of_model(m: &Value) -> Option<&'static str> {
    if registry::kind_of(m) != "image" {
        return None;
    }
    let arch = s(m, "arch");
    let draws_not = arch.contains("ocr") || matches!(arch.as_str(), "realesrgan" | "waifu2x" | "flashvsr" | "seedvr2" | "starvector" | "svgrender" | "birefnet" | "isnet" | "depth-anything-v2");
    if draws_not {
        return None;
    }
    Some(match arch.as_str() {
        a if a.starts_with("qwen-image") => "qwen-image",
        "z-image" => "z-image",
        a if a.contains("h3") => "h3",
        a if a.starts_with("hunyuanimage") => "hunyuanimage",
        "moss-image" => "moss-image",
        "ming-image" => "ming-image",
        _ => family_by_prefix(&s(m, "id")).unwrap_or("natural"),
    })
}

/// `target` read: (family, the model it named or None)
pub fn resolve_target(target: &str, all: &[Value]) -> Result<(&'static str, Option<String>), Fail> {
    let t = target.trim();
    if let Some(f) = FAMILIES.iter().find(|f| f.eq_ignore_ascii_case(t)) {
        return Ok((f, None));
    }
    if let Some(m) = all.iter().find(|m| m["id"] == t) {
        return match family_of_model(m) {
            Some(f) => Ok((f, Some(t.to_string()))),
            None => Err((400, format!("target {t}: it does not draw pictures from prompts"))),
        };
    }
    match family_by_prefix(t) {
        Some(f) => Ok((f, Some(t.to_string()))),
        None => Err((400, format!("target {t}: neither a family ({}) nor an image model that draws", FAMILIES.join(", ")))),
    }
}

/// The size to draw a `w` x `h` picture at for a family: the family's best area at the picture's aspect (clamped to
/// 1:4 .. 4:1), each side on its grid and within its range - 1 MP on multiples of 16 (256-2048) for most, 4 MP on
/// multiples of 32 for h3 (256-5024) and hunyuanimage (768-3072)
pub fn suggested_size(family: &str, w: u32, h: u32) -> (u32, u32) {
    let (area, grid, lo, hi) = match family {
        "h3" => (4_194_304.0, 32.0, 256.0, 5024.0),
        "hunyuanimage" => (4_194_304.0, 32.0, 768.0, 3072.0),
        _ => (1_048_576.0, 16.0, 256.0, 2048.0),
    };
    let r = (w.max(1) as f64 / h.max(1) as f64).clamp(0.25, 4.0);
    let (mut sw, mut sh) = ((area * r).sqrt(), (area / r).sqrt());
    // the long side within the range, then the short one (the aspect kept)
    let long = sw.max(sh);
    if long > hi {
        sw *= hi / long;
        sh *= hi / long;
    }
    let short = sw.min(sh);
    if short < lo {
        sw *= lo / short;
        sh *= lo / short;
    }
    let side = |x: f64| ((x / grid).round() * grid).clamp((lo / grid).ceil() * grid, (hi / grid).floor() * grid) as u32;
    (side(sw), side(sh))
}

/// What a request asks, checked: everything but the picture and the describer
#[derive(Debug, Clone, PartialEq)]
pub struct Ask {
    pub family: &'static str,
    pub target_model: Option<String>,
    pub detail: &'static str,
    /// about this many words (tags: tags)
    pub words: usize,
    /// a negative prompt asked for
    pub negative_asked: bool,
    /// ... and written (the family reads one)
    pub negative: bool,
    pub language: String,
    pub instructions: String,
    pub max_pixels: u64,
    pub max_tokens: usize,
    pub temperature: f64,
    pub seed: Option<u64>,
}

/// A request's values checked (the messages name the field and its range: "max_words 3: from 8 to 400")
pub fn ask_of(b: &Value, all: &[Value]) -> Result<Ask, Fail> {
    let bad = |m: String| Err((400, m));
    let int = |k: &str, lo: i64, hi: i64| -> Result<Option<i64>, Fail> {
        match &b[k] {
            Value::Null => Ok(None),
            v => match v.as_i64().or_else(|| v.as_f64().filter(|x| x.fract() == 0.0).map(|x| x as i64)) {
                Some(x) if (lo..=hi).contains(&x) => Ok(Some(x)),
                _ => Err((400, format!("{k} {v}: from {lo} to {hi}"))),
            },
        }
    };
    let target = b["target"].as_str().filter(|t| !t.trim().is_empty()).unwrap_or("natural");
    let (family, target_model) = resolve_target(target, all)?;
    let detail = match b["detail"].as_str().unwrap_or("medium") {
        "short" => "short",
        "medium" => "medium",
        "long" => "long",
        d => return bad(format!("detail {d}: short, medium or long")),
    };
    let by_detail = match (family == "tags", detail) {
        (true, "short") => 12,
        (true, "long") => 45,
        (true, _) => 25,
        (false, "short") => 35,
        (false, "long") => 170,
        (false, _) => 90,
    };
    let words = int("max_words", 8, 400)?.map_or(by_detail, |x| x as usize);
    let negative_asked = b["include_negative"] == true;
    let negative = negative_asked && reads_negative(family);
    let language = b["language"].as_str().map(str::trim).filter(|l| !l.is_empty()).unwrap_or("English").to_string();
    if language.chars().count() > 40 {
        return bad(format!("language: at most 40 characters ({})", language.chars().count()));
    }
    let instructions = b["instructions"].as_str().map(str::trim).unwrap_or("").to_string();
    if instructions.chars().count() > 2000 {
        return bad(format!("instructions: at most 2000 characters ({})", instructions.chars().count()));
    }
    let max_pixels = int("max_pixels", 65_536, 4_194_304)?.unwrap_or(1_048_576) as u64;
    let max_tokens = int("max_tokens", 16, 4096)?.map_or(2 * words + 40 + if negative { 80 } else { 0 }, |x| x as usize);
    let temperature = match &b["temperature"] {
        Value::Null => 0.0,
        v => match v.as_f64() {
            Some(t) if (0.0..=2.0).contains(&t) => t,
            _ => return bad(format!("temperature {v}: from 0 to 2")),
        },
    };
    let seed = match &b["seed"] {
        Value::Null => None,
        v => Some(v.as_u64().ok_or_else(|| (400, format!("seed {v}: a whole number, 0 or more")))?),
    };
    Ok(Ask { family, target_model, detail, words, negative_asked, negative, language, instructions, max_pixels, max_tokens, temperature, seed })
}

/// How each family's prompts are written (nextsycl's table)
fn style(family: &str) -> &'static str {
    match family {
        "qwen-image" => "Write it as you would tell a painter: the subject and its look, the setting, the light, the camera (lens, angle, distance), the style - one long paragraph. Any text in the picture goes in double quotes.",
        "z-image" => "Write sentences, never a list of tags. Name the kind of picture first; then the subject, what it wears or holds, the setting, the light. Each piece of text in the picture is quoted exactly, with where it is and its lettering. Only what is there.",
        "h3" => "Open with one sentence naming the real object the picture is (\"a photograph by an experienced street photographer\", \"a flat vector illustration\"). Then one scene: the viewpoint, the subject, a few supporting elements, the light with its source and direction. Quote every word of text in the picture with where it is, then say \"no other text\". No hype words (stunning, epic, cinematic, 8k, masterpiece).",
        "hunyuanimage" => "Write it as you would tell a painter: the subject and its look, the setting, the light, the camera (lens, angle, distance), the style - one long paragraph. Quote any text in the picture in double quotes, spelled exactly.",
        "moss-image" => "Write natural sentences.",
        "ming-image" => "It is a design: describe the layout first, then each block - its exact words in double quotes, its colours, its lettering, its position.",
        "tags" => "Write comma-separated tags, the most important first: the kind of picture, the subject and its count, its attributes, the colours, the setting, the style, the lighting, the framing.",
        _ => "Write one paragraph of plain, concrete sentences, the most important things first.",
    }
}

/// The describer's instruction (the system turn) for a request
pub fn instruction(a: &Ask) -> String {
    let what = match &a.target_model {
        Some(m) => format!("the text-to-image model {m}"),
        None if a.family == "natural" => "a text-to-image model".to_string(),
        None => format!("a {} text-to-image model", a.family),
    };
    let size = if a.family == "tags" { format!("about {} tags", a.words) } else { format!("about {} words", a.words) };
    let mut t = format!("You write prompts for {what}. Look at the picture and write the prompt that would make the model draw this picture again.\n\
Describe only what is visible: nothing outside the frame, nothing guessed. Say what kind of picture it is (a photograph, an illustration, a painting, a 3D render, a screenshot, a diagram ...). \
Name the subjects with their colours, shapes, materials, counts and places in the frame; then the background; the light; the viewpoint and framing; the palette. \
Copy any text in the picture exactly, in double quotes.\n{}\nWrite {size}, in {}.", style(a.family), a.language);
    if !a.instructions.is_empty() {
        t += &format!("\nAlso: {}", a.instructions);
    }
    t += "\nAnswer in this shape and nothing else:\nPROMPT: <the prompt>";
    if a.negative {
        t += "\nNEGATIVE: <what the picture must not have, as a short comma-separated list>";
    }
    t
}

/// Where `key` (any case, maybe in Markdown's ** or #) starts in `text`, and where what follows it starts
fn marker(text: &str, key: &str) -> Option<(usize, usize)> {
    let low = text.to_ascii_lowercase();
    let k = key.to_ascii_lowercase();
    let at = low.find(&k)?;
    // the marker's decoration before it on its line (**PROMPT:**, ## NEGATIVE:)
    let start = text[..at].trim_end_matches(['*', '#', '_', ' ']).len();
    let after = at + k.len();
    let rest = &text[after..];
    let skip = rest.len() - rest.trim_start_matches(['*', '_']).len();
    Some((start, after + skip))
}

/// An answer's text tidied: trimmed, its lines one paragraph, wrapping quotes or Markdown markers off
fn tidy(t: &str) -> String {
    let one = t.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut x = one.trim().trim_matches(['*', '_']).trim().to_string();
    // the whole of it in quotes (and only those quotes): unwrapped
    if x.len() >= 2 && x.starts_with('"') && x.ends_with('"') && !x[1..x.len() - 1].contains('"') {
        x = x[1..x.len() - 1].trim().to_string();
    }
    x
}

/// The describer's answer cut: the prompt (after `PROMPT:`, before `NEGATIVE:`; the whole answer when it has no
/// `PROMPT:`) and the negative prompt (after `NEGATIVE:`, when one was asked for and written)
pub fn cut(answer: &str, negative: bool) -> (String, Option<String>) {
    // thinking that came through anyway: dropped
    let a = match answer.rfind("</think>") {
        Some(i) => &answer[i + "</think>".len()..],
        None => answer,
    };
    let (p_end, n_from) = match marker(a, "NEGATIVE PROMPT:").or_else(|| marker(a, "NEGATIVE:")) {
        Some((s0, s1)) => (s0, Some(s1)),
        None => (a.len(), None),
    };
    // the prompt is what follows the first PROMPT: up to the next one - except when the describer wrote the prompt
    // first with no label and then started it again after one (the blob portrait, 2026-11-11: a whole prompt, then
    // "PROMPT: " and the same words until max_tokens): then the unlabelled one before it, which is whole
    let (p_from, p_end) = match marker(&a[..p_end], "PROMPT:") {
        Some((s0, _)) if tidy(&a[..s0]).split_whitespace().count() >= 8 => (0, s0),
        Some((_, s1)) => (s1, marker(&a[s1..p_end], "PROMPT:").map_or(p_end, |(s0, _)| s1 + s0)),
        None => (0, p_end),
    };
    let prompt = tidy(&a[p_from..p_end]);
    let neg = n_from.filter(|_| negative).map(|i| tidy(&a[i..])).filter(|n| !n.is_empty());
    (prompt, neg)
}

/// The picture read: its size, then flattened on white and made at most `max_pixels` (Lanczos, the aspect kept) -
/// (width, height, the size seen, the PNG the describer gets)
pub fn see(image: &str, max_pixels: u64) -> Result<(u32, u32, u32, u32, Vec<u8>), Fail> {
    let bytes = crate::http::unbase64(image.trim()).map_err(|e| (400, format!("image: not base64 or a data: URL ({e})")))?;
    let (w, h) = picture_size(&bytes)?;
    let mut pic = nextcuda_diffusion::picture::Picture::decode(&bytes).map_err(|e| (400, format!("image: {e}")))?;
    if pic.channels == 4 {
        // transparency on white
        let data = pic.data.chunks_exact(4).flat_map(|p| {
            let a = p[3] as u32;
            let on = |c: u8| ((c as u32 * a + 255 * (255 - a) + 127) / 255) as u8;
            [on(p[0]), on(p[1]), on(p[2])]
        }).collect();
        pic = nextcuda_diffusion::picture::Picture { width: pic.width, height: pic.height, channels: 3, data };
    }
    let pic = pic.rgb();
    let (sw, sh) = seen_size(w, h, max_pixels);
    let pic = if (sw, sh) == (w, h) { pic } else { pic.resize_lanczos(sw, sh) };
    let png = pic.png_bytes(&[]).map_err(|e| (500, format!("image: {e}")))?;
    Ok((w, h, sw, sh, png))
}

/// A picture's size from its header, refused over 40 MP or when it does not read
pub fn picture_size(bytes: &[u8]) -> Result<(u32, u32), Fail> {
    let (w, h) = image::ImageReader::new(std::io::Cursor::new(bytes)).with_guessed_format().ok().and_then(|r| r.into_dimensions().ok())
        .ok_or_else(|| (400, "image: not a picture this reads (PNG, JPEG, WebP)".to_string()))?;
    if w as u64 * h as u64 > crate::image::MAX_PICTURE_PIXELS {
        return Err((400, format!("image: {w} x {h} is over 40 MP")));
    }
    Ok((w, h))
}

/// The size the describer sees: the picture's, or made smaller to `max_pixels` (the aspect kept)
pub fn seen_size(w: u32, h: u32, max_pixels: u64) -> (u32, u32) {
    let px = w as u64 * h as u64;
    if px <= max_pixels {
        return (w, h);
    }
    let k = (max_pixels as f64 / px as f64).sqrt();
    (((w as f64 * k).floor() as u32).max(1), ((h as f64 * k).floor() as u32).max(1))
}

/// Whether a chat model's entry reads pictures: a vision tower beside it (mmproj), not switched off
pub fn reads_pictures(m: &Value) -> bool {
    registry::kind_of(m) == "llm" && ["mmproj", "vision"].iter().any(|r| m["files"][*r].as_str().is_some_and(|f| !f.is_empty()))
        && m["env"]["NS_Q35_VISION"] != "0"
}

/// An image model's text encoder that reads pictures and writes: a describer of its own (describer "native", or by id)
#[derive(Debug, PartialEq)]
pub struct NativeKind {
    /// its id among the describers (image.describers)
    pub id: &'static str,
    /// short, for a menu
    pub name: &'static str,
    /// the model it is
    pub base: &'static str,
    /// whose it is, for a menu ("Qwen-Image's own")
    pub own: &'static str,
    /// the image architecture whose files hold it
    pub arch: &'static str,
    /// the file roles it reads
    pub roles: &'static [&'static str],
}

/// The text encoders that describe: Qwen-Image 2.1's Qwen3-VL-8B-Instruct (the Q4_K_M GGUF and its projector) and
/// P-H3-Image's Qwen3-VL-4B-Instruct (its encoder GGUF holds the language model whole - 36 layers, the final norm, the
/// head tied to the embedding - and the vision tower; the tokenizer files beside it)
pub const NATIVES: [NativeKind; 2] = [
    NativeKind { id: "qwen3-vl-8b-instruct", name: "Qwen3-VL-8B", base: "Qwen3-VL-8B-Instruct", own: "Qwen-Image's own", arch: "qwen-image-2.1",
                 roles: &["text-encoder", "llm-vision"] },
    NativeKind { id: "qwen3-vl-4b-instruct", name: "Qwen3-VL-4B", base: "Qwen3-VL-4B-Instruct", own: "H3's own", arch: "p-h3-image",
                 roles: &["text-encoder", "tokenizer"] },
];

/// A registered image model's own describer: its architecture's, when the entry has the files it reads
pub fn native_of(m: &Value) -> Option<&'static NativeKind> {
    if registry::kind_of(m) != "image" {
        return None;
    }
    let arch = s(m, "arch");
    NATIVES.iter().find(|k| k.arch == arch && k.roles.iter().all(|r| m["files"][*r].as_str().is_some_and(|f| !f.is_empty())))
}

/// What a model's (or a family's) text encoder is, for a refusal: why it cannot describe
fn encoder_of(arch_or_family: &str) -> &'static str {
    match arch_or_family {
        "z-image" => "Qwen3-4B",
        "qwen-image-2512" => "Qwen2.5-VL-7B, its vision projector not downloaded",
        "qwen-image" => "Qwen2.5-VL-7B or Qwen3-VL-8B",
        "moss-image" => "Qwen3.5-9B, its vision projector not downloaded",
        "hunyuanimage" => "Qwen2.5-VL-7B and ByT5",
        "ming-image" => "Ming's own",
        "h3" | "minimax-h3" => "Qwen3-VL",
        _ => "",
    }
}

/// The bytes of an entry's files of `roles` (what a describer reads), from the drive
pub fn files_bytes(m: &Value, roles: &[&str]) -> u64 {
    roles.iter().filter_map(|r| m["files"][*r].as_str()).filter_map(|f| std::fs::metadata(f).ok()).map(|md| md.len()).sum()
}

/// A chat model's short name for a menu: "qwen3.8-27b-gsq-iq3s" -> "Qwen3.8-27B"
pub fn chat_name(id: &str) -> String {
    let parts: Vec<&str> = id.split('-').collect();
    let n = parts.iter().position(|p| p.len() > 1 && p.ends_with('b') && p[..p.len() - 1].chars().all(|c| c.is_ascii_digit() || c == '.')).map_or(1, |i| i + 1);
    let mut out = parts[..n.min(parts.len())].join("-");
    if let Some(r) = out.strip_prefix("qwen") {
        out = format!("Qwen{r}");
    }
    if out.ends_with('b') {
        out.pop();
        out.push('B');
    }
    out
}

/// A message whose field is `field` whatever its first word (api.rs reads the tag off: the error's `field`)
pub fn with_field(field: &str, m: &str) -> String {
    format!("{FIELD_TAG}{field}\u{1}{m}")
}

pub const FIELD_TAG: &str = "\u{1}field:";

/// A message's field tag: (the field, the message without it)
pub fn field_untag(m: &str) -> Option<(&str, &str)> {
    m.strip_prefix(FIELD_TAG)?.split_once('\u{1}')
}

/// Who describes: a chat model (through the switcher), or an image model's text encoder (in that model's server)
#[derive(Debug, Clone, PartialEq)]
pub enum Pick {
    Chat(String),
    Native { host: String, kind: &'static NativeKind },
}

/// The image model a native describer runs in: among `hosts` (its enabled models), a loaded one, else the one whose
/// card takes the least
fn host_of(hosts: &[&Value], loaded: &[String]) -> Option<String> {
    let gib = |m: &Value| m["vram_gib"].as_f64().unwrap_or(f64::MAX);
    hosts.iter().find(|m| loaded.iter().any(|l| m["id"] == l.as_str())).or_else(|| hosts.iter().min_by(|a, b| gib(a).total_cmp(&gib(b)))).map(|m| s(m, "id"))
}

/// describer "native": the target's own text encoder - the target model's when it names one, else its family's
/// enabled model with the smallest encoder (a loaded one first among equals). Refused (invalid_params, field
/// describer) when none reads pictures
pub fn native_for(ask: &Ask, all: &[Value], loaded: &[String]) -> Result<Pick, Fail> {
    let refuse = |what: &str, enc: &str| {
        let enc = if enc.is_empty() { String::new() } else { format!(" ({enc})") };
        Err((400, with_field("describer", &format!("target {what}: its text encoder{enc} does not read pictures; use describer best"))))
    };
    if let Some(t) = &ask.target_model {
        let Some(m) = all.iter().find(|m| m["id"] == t.as_str()) else {
            // a model not here (nextsycl's, or a card not downloaded: the caller checks that first)
            return refuse(t, encoder_of(ask.family));
        };
        return match native_of(m) {
            Some(k) if m["enabled"] != false => Ok(Pick::Native { host: t.clone(), kind: k }),
            Some(_) => Err((409, format!("model {t} is disabled (system.enable {t})"))),
            None => refuse(t, encoder_of(&s(m, "arch"))),
        };
    }
    let fam: Vec<&Value> = all.iter().filter(|m| family_of_model(m) == Some(ask.family) && native_of(m).is_some()).collect();
    let on: Vec<&Value> = fam.iter().copied().filter(|m| m["enabled"] != false).collect();
    if on.is_empty() {
        if let Some(m) = fam.first() {
            return Err((409, format!("no {} model with a text encoder that reads pictures is enabled (system.enable {})", ask.family, s(m, "id"))));
        }
        return refuse(ask.family, encoder_of(ask.family));
    }
    // the smallest encoder (its files), then a loaded model, then the least memory
    let size = |m: &Value| native_of(m).map_or(u64::MAX, |k| files_bytes(m, k.roles));
    let least = on.iter().map(|m| size(m)).min().unwrap_or(0);
    let smallest: Vec<&Value> = on.iter().copied().filter(|m| size(m) == least).collect();
    let host = host_of(&smallest, loaded).expect("one at least");
    let kind = native_of(smallest.iter().find(|m| m["id"] == host.as_str()).expect("chosen from them")).expect("filtered");
    Ok(Pick::Native { host, kind })
}

/// The words of a prompt
pub fn word_count(t: &str) -> usize {
    t.split_whitespace().count()
}

impl Home {
    /// The describer for a request: the one named (a chat model that reads pictures, enabled, downloaded), else the
    /// loaded chat model when it reads pictures (no swap), else the enabled one whose card takes the least memory.
    /// `block`: a model a card offers and not downloaded is fetched and waited for (NS_AUTO_PULL), else refused
    pub(crate) fn describer(&self, b: &Value, all: &[Value], block: bool) -> Result<String, Fail> {
        if let Some(id) = b["model"].as_str().map(str::trim).filter(|x| !x.is_empty()) {
            let Some(m) = all.iter().find(|m| m["id"] == id) else {
                if self.not_downloaded(id).is_some() {
                    self.need_downloaded(id, b["request_id"].as_str(), block)?;
                    return Ok(id.to_string());
                }
                return Err((404, format!("model {id}: no such model (system.models)")));
            };
            if !reads_pictures(m) {
                return Err((400, format!("model {id}: it does not read pictures (a chat model with a vision tower: system.models, capability describe)")));
            }
            if m["enabled"] == false {
                return Err((409, format!("model {id} is disabled (system.enable {id})")));
            }
            return Ok(id.to_string());
        }
        let on: Vec<&Value> = all.iter().filter(|m| m["enabled"] != false && reads_pictures(m)).collect();
        if let Some(l) = self.chat_loaded(all).filter(|l| on.iter().any(|m| m["id"] == l.as_str())) {
            return Ok(l);
        }
        let gib = |m: &Value| m["vram_gib"].as_f64().unwrap_or(f64::MAX);
        on.iter().min_by(|a, b| gib(a).total_cmp(&gib(b))).map(|m| s(m, "id"))
            .ok_or_else(|| (409, "no enabled chat model reads pictures (system.enable qwen3.8-27b-gsq-iq3s: a chat model with its vision tower)".to_string()))
    }

    /// Who describes a request (`describer`: best, native or an id from image.describers). `block`: a model not
    /// downloaded is fetched and waited for (NS_AUTO_PULL), else refused
    pub(crate) fn pick_describer(&self, b: &Value, all: &[Value], ask: &Ask, block: bool) -> Result<Pick, Fail> {
        let mode = b["describer"].as_str().map(str::trim).filter(|x| !x.is_empty()).unwrap_or("best");
        match mode {
            "best" => Ok(Pick::Chat(self.describer(b, all, block)?)),
            "native" => {
                // a target model a card offers and not downloaded: model_not_downloaded (fetched with NS_AUTO_PULL)
                if let Some(t) = ask.target_model.as_deref().filter(|t| !all.iter().any(|m| m["id"] == *t)) {
                    let kind = self.o.cfg.cards().iter().find(|c| c["id"] == t).and_then(|c| NATIVES.iter().find(|k| c["arch"] == k.arch));
                    if let Some(kind) = kind.filter(|_| self.not_downloaded(t).is_some()) {
                        self.need_downloaded(t, b["request_id"].as_str(), block)?;
                        if !block {
                            return Ok(Pick::Native { host: t.to_string(), kind });
                        }
                        return native_for(ask, &self.models(), &self.loaded_of("image"));
                    }
                }
                native_for(ask, all, &self.loaded_of("image"))
            }
            id => {
                if let Some(k) = NATIVES.iter().find(|k| k.id == id) {
                    let hosts: Vec<&Value> = all.iter().filter(|m| native_of(m) == Some(k) && m["enabled"] != false).collect();
                    return match host_of(&hosts, &self.loaded_of("image")) {
                        Some(host) => Ok(Pick::Native { host, kind: k }),
                        None => Err((409, format!("describer {id}: no enabled {} model here has its files (image.describers)", k.arch))),
                    };
                }
                if all.iter().any(|m| m["id"] == id && reads_pictures(m)) {
                    let mut b = b.clone();
                    b["model"] = json!(id);
                    return Ok(Pick::Chat(self.describer(&b, all, block)?));
                }
                Err((404, format!("describer {id}: no such describer here - best, native or an id from image.describers")))
            }
        }
    }

    /// image.describers: every describer this box has - the chat models that read pictures and the image models' text
    /// encoders that do - with what it is, its size and the image models that share it; `default` is best's pick now
    pub(crate) fn describers_view(&self, all: &[Value]) -> Value {
        let best = self.describer(&json!({}), all, false).ok();
        let gb = |b: u64| (b as f64 / 1e9 * 10.0).round() / 10.0;
        let mut list: Vec<Value> = all.iter().filter(|m| reads_pictures(m)).map(|m| {
            let id = s(m, "id");
            let roles: Vec<&str> = m["files"].as_object().map(|o| o.keys().map(String::as_str).collect()).unwrap_or_default();
            json!({"id": id, "name": chat_name(&id), "base": chat_name(&id), "kind": "chat", "own": "a chat model",
                   "size_gb": gb(files_bytes(m, &roles)), "models": [], "available": m["enabled"] != false,
                   "why": if m["enabled"] == false { json!(format!("{id} is disabled (system.enable {id})")) } else { Value::Null },
                   "default": best.as_deref() == Some(id.as_str())})
        }).collect();
        for k in &NATIVES {
            let hosts: Vec<&Value> = all.iter().filter(|m| native_of(m) == Some(k)).collect();
            let on: Vec<&Value> = hosts.iter().copied().filter(|m| m["enabled"] != false).collect();
            let why = if hosts.is_empty() {
                json!(format!("no {} model is downloaded here", k.arch))
            } else if on.is_empty() {
                json!(format!("its models are disabled (system.enable {})", s(hosts[0], "id")))
            } else {
                Value::Null
            };
            list.push(json!({"id": k.id, "name": k.name, "base": k.base, "kind": "native", "own": k.own,
                             "size_gb": hosts.first().map(|m| gb(files_bytes(m, k.roles))), "models": hosts.iter().map(|m| s(m, "id")).collect::<Vec<_>>(),
                             "available": !on.is_empty(), "why": why, "default": false}));
        }
        json!({"describers": list})
    }

    /// image.describe as submitted: what would be refused when it runs, refused now (wait: false too)
    pub(crate) fn precheck_describe(&self, b: &Value, all: &[Value]) -> Result<(), Fail> {
        let ask = ask_of(b, all)?;
        let bytes = crate::http::unbase64(s(b, "image").trim()).map_err(|e| (400, format!("image: not base64 or a data: URL ({e})")))?;
        picture_size(&bytes)?;
        self.pick_describer(b, all, &ask, false).map(|_| ())
    }

    /// image.describe's work (a job of the image queue: its request_id is the job's id)
    pub(crate) fn describe(&self, b: &Value) -> Result<Value, Fail> {
        let t0 = Instant::now();
        let all = self.models();
        let ask = ask_of(b, &all)?;
        let pick = self.pick_describer(b, &all, &ask, true)?;
        let (w, h, sw, sh, png) = see(&s(b, "image"), ask.max_pixels)?;
        let mode = b["describer"].as_str().map(str::trim).filter(|x| !x.is_empty()).unwrap_or("best").to_string();
        let seen = (sw, sh);
        let wrote = match &pick {
            Pick::Chat(id) => self.describe_chat(b, &ask, id, &png, t0)?,
            Pick::Native { host, kind } => self.describe_native(b, &ask, host, kind, &png, t0)?,
        };
        Ok(answer(&ask, wrote, &mode, (w, h), seen, t0))
    }

    /// A chat model writes it (through the switcher; the answer streamed)
    fn describe_chat(&self, b: &Value, ask: &Ask, id: &str, png: &[u8], t0: Instant) -> Result<Wrote, Fail> {
        let job = b["request_id"].as_str();
        let cancelled = || job.is_some_and(|j| self.job_cancelled(j));
        // the describer: room on its cards, then (through the switcher) loaded when another chat model is
        self.note(job, "loading", &format!("the describer {id}: room on its cards, then its load"));
        let t = self.chat_target(Some(id))?;
        super::rsvp::note_start(id, 0.0);
        if cancelled() {
            return Err((499, "cancelled".into()));
        }
        self.note(job, "running", &format!("{id} reading the picture"));
        let mut body = json!({
            "model": id, "stream": true, "think_budget": 0, "max_tokens": ask.max_tokens, "temperature": ask.temperature,
            "messages": [
                {"role": "system", "content": instruction(ask)},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{}", crate::http::base64(png))}},
                    {"type": "text", "text": "Write the prompt for this picture."}]}],
        });
        if let Some(seed) = ask.seed {
            body["seed"] = json!(seed);
        }
        let deadline = t0 + Duration::from_secs(MAX_DESCRIBE_S);
        let (text, done) = stream_lines(&t, "/api/chat", &body, deadline, &|n| {
            if n > 0 {
                self.note(job, "running", &format!("{id} writing: {n} tokens"));
            }
            cancelled()
        })?;
        // rsvp_s: the describer stays loaded that long from the end of the answer (as llm.chat)
        if let Some(g) = super::rsvp::granted(super::rsvp::asked(), self.max_rsvp()) {
            self.hold_after(id, g);
            if g > 0.0 {
                let until = self.held_s(id) + super::now();
                let _ = registry::update(&self.o.cfg, id, |e| e["hold_until"] = json!(until));
            }
        }
        let num = |k: &str| done[k].as_f64().unwrap_or(0.0);
        Ok(Wrote {
            text, model: id.to_string(), finish: done["finish"].as_str() == Some("length"),
            prompt: num("prompt_tokens"), reused: num("reused"), generated: num("generated"),
            read_s: num("read_seconds") + num("vision_seconds"), gen_s: num("generate_seconds"), energy: done["energy_wh"].as_f64(), held_mb: None,
        })
    }

    /// An image model's text encoder writes it, in that model's server (the pipeline dropped for it there)
    fn describe_native(&self, b: &Value, ask: &Ask, host: &str, kind: &NativeKind, png: &[u8], _t0: Instant) -> Result<Wrote, Fail> {
        let job = b["request_id"].as_str();
        self.note(job, "loading", &format!("{host}'s text encoder ({}): room on its card, then its load", kind.base));
        let mut body = json!({"model": host, "image": crate::http::base64(png), "system": instruction(ask), "user": "Write the prompt for this picture.",
                              "max_tokens": ask.max_tokens, "temperature": ask.temperature, "seed": ask.seed.unwrap_or(0)});
        if let Some(j) = job {
            body["request_id"] = json!(j);
        }
        // a server started for this loads the text encoder alone (its pipeline when a picture is drawn)
        *self.describe_first.lock().unwrap().entry(host.to_string()).or_insert(0) += 1;
        struct Done<'a>(&'a Home, &'a str);
        impl Drop for Done<'_> {
            fn drop(&mut self) {
                let mut d = self.0.describe_first.lock().unwrap();
                if let Some(n) = d.get_mut(self.1) {
                    *n = n.saturating_sub(1);
                    if *n == 0 {
                        d.remove(self.1);
                    }
                }
            }
        }
        let _first = Done(self, host);
        let out = self.relay_with_room(host, &|| {
            let t = self.ensure_for(host, job)?;
            self.sent(job);
            super::api::relay(&t, "POST", "/v1/images/describe", Some(&body), MAX_DESCRIBE_S)
        })?;
        let num = |k: &str| out[k].as_f64().unwrap_or(0.0);
        let enc = out["encoder"].as_str().filter(|e| !e.is_empty()).unwrap_or(kind.base);
        Ok(Wrote {
            text: s(&out, "text"), model: format!("{host} text encoder ({enc})"), finish: out["finish"] == "length",
            prompt: num("prompt_tokens"), reused: 0.0, generated: num("generated"), read_s: num("read_seconds"), gen_s: num("generate_seconds"),
            energy: out["energy_wh"].as_f64(), held_mb: out["held_mb"].as_u64(),
        })
    }
}

/// What a describer wrote, and its counts
pub struct Wrote {
    pub text: String,
    /// the answer's `model`
    pub model: String,
    /// max_tokens cut it
    pub finish: bool,
    pub prompt: f64,
    pub reused: f64,
    pub generated: f64,
    pub read_s: f64,
    pub gen_s: f64,
    pub energy: Option<f64>,
    /// a text encoder's memory on its card, MiB
    pub held_mb: Option<u64>,
}

/// The answer: the prompt cut out, the notes, the sizes, the counts
fn answer(ask: &Ask, wr: Wrote, describer: &str, (w, h): (u32, u32), (sw, sh): (u32, u32), t0: Instant) -> Value {
    let (prompt, negative) = cut(&wr.text, ask.negative);
    let mut notes: Vec<String> = Vec::new();
    if ask.negative_asked && !ask.negative {
        notes.push(format!("{} takes no negative prompt (no guidance): none written", ask.family));
    } else if ask.negative && negative.is_none() {
        notes.push("the describer wrote no negative prompt".into());
    }
    let finish = if wr.finish { "length" } else { "stop" };
    if wr.finish {
        notes.push(format!("max_tokens {} cut the answer short: raise max_tokens", ask.max_tokens));
    }
    if prompt.is_empty() {
        notes.push("the describer wrote no prompt".into());
    }
    let r3 = |x: f64| (x * 1000.0).round() / 1000.0;
    let r1 = |x: f64| (x * 10.0).round() / 10.0;
    let (gw, gh) = suggested_size(ask.family, w, h);
    let mut v = json!({
        "prompt": prompt, "negative_prompt": negative, "target": ask.family, "target_model": ask.target_model, "model": wr.model, "describer": describer,
        "detail": ask.detail, "language": ask.language, "words": word_count(&prompt),
        "width": w, "height": h, "seen": {"width": sw, "height": sh},
        "suggested_size": {"width": gw, "height": gh, "size": format!("{gw}x{gh}")},
        "tokens": {"prompt": wr.prompt as u64, "reused": wr.reused as u64, "generated": wr.generated as u64},
        "timings": {"pp_tok_s": r1((wr.prompt - wr.reused).max(0.0) / wr.read_s.max(1e-9)), "tg_tok_s": r1(wr.generated / wr.gen_s.max(1e-9)),
                    "prompt_seconds": r3(wr.read_s), "generate_seconds": r3(wr.gen_s), "seconds": r3(t0.elapsed().as_secs_f64())},
        "finish": finish, "energy_wh": wr.energy.map(|x| (x * 100.0).round() / 100.0), "notes": notes,
    });
    if let Some(mb) = wr.held_mb {
        v["memory_mb"] = json!(mb);
    }
    v
}

/// A chat request answered as JSON lines (/api/chat, stream): the content and the closing line. The read wakes every
/// 250 ms to ask `stop(tokens so far)`; true closes the connection (the chat server stops at its next token) and
/// answers 499. Past `deadline`: 504. The server's refusal keeps its status (the switcher's 409 / 503 too).
fn stream_lines(t: &Target, path: &str, body: &Value, deadline: Instant, stop: &dyn Fn(usize) -> bool) -> Result<(String, Value), Fail> {
    let gone = |e: String| (502, format!("the chat server does not answer: {e}"));
    let mut c = Conn::connect(t).map_err(|e| gone(e.to_string()))?;
    let text = serde_json::to_vec(body).unwrap_or_default();
    write!(c, "POST {path} HTTP/1.1\r\nHost: nextcuda\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", text.len())
        .and_then(|_| c.write_all(&text)).and_then(|_| c.flush()).map_err(|e| gone(e.to_string()))?;
    c.set_read_timeout(Some(Duration::from_millis(250))).map_err(|e| gone(e.to_string()))?;
    let mut buf: Vec<u8> = Vec::new();
    let mut chunk = [0u8; 16384];
    let mut head: Option<u16> = None;
    let (mut content, mut done, mut n) = (String::new(), Value::Null, 0usize);
    loop {
        match c.read(&mut chunk) {
            Ok(0) => break,
            Ok(k) => buf.extend_from_slice(&chunk[..k]),
            Err(e) if matches!(e.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut | std::io::ErrorKind::Interrupted) => {}
            Err(e) => return Err(gone(e.to_string())),
        }
        if head.is_none() {
            if let Some(end) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                let line = String::from_utf8_lossy(&buf[..end]).lines().next().unwrap_or("").to_string();
                head = Some(line.split_whitespace().nth(1).and_then(|x| x.parse().ok()).unwrap_or(502));
                buf.drain(..end + 4);
            }
        }
        // the answer's lines as they come (a refusal is read whole, at the end)
        if head == Some(200) {
            while let Some(i) = buf.iter().position(|b| *b == b'\n') {
                let line: Vec<u8> = buf.drain(..=i).collect();
                let Ok(v) = serde_json::from_slice::<Value>(&line) else { continue };
                if let Some(t) = v["content"].as_str() {
                    content += t;
                    n += 1;
                }
                if let Some(e) = v["error"].as_str().or(v["error"]["message"].as_str()) {
                    return Err((502, format!("the chat server: {e}")));
                }
                if v["done"] == true {
                    done = v;
                }
            }
            if !done.is_null() {
                return Ok((content, done));
            }
        }
        if stop(n) {
            return Err((499, "cancelled".into()));
        }
        if Instant::now() > deadline {
            return Err((504, format!("the describe ran over {} minutes", MAX_DESCRIBE_S / 60)));
        }
    }
    match head {
        Some(200) => Err((502, "the chat server's stream ended early".into())),
        Some(st) => {
            let v: Value = serde_json::from_slice(&buf).unwrap_or(Value::Null);
            let m = v["error"]["message"].as_str().or_else(|| v["error"].as_str()).map(str::to_string)
                .unwrap_or_else(|| String::from_utf8_lossy(&buf).trim().chars().take(300).collect());
            // the switcher's and the server's: a refusal of the request 400, a model in use 409, no room 503; else 502
            let st = match st {
                400 | 409 | 503 => st,
                413 => 400,
                _ => 502,
            };
            Err((st, m))
        }
        None => Err((502, "the chat server answered nothing".into())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn models() -> Vec<Value> {
        vec![
            json!({"id": "z-image-turbo", "kind": "image", "arch": "z-image"}),
            json!({"id": "p-h3-image", "kind": "image", "arch": "p-h3-image"}),
            json!({"id": "qwen-image-2.1-turbo-q8", "kind": "image", "arch": "qwen-image-2.1"}),
            json!({"id": "moss-image-sft-q8", "kind": "image", "arch": "moss-image"}),
            json!({"id": "ming-image-0.1-design", "kind": "image", "arch": "ming-image"}),
            json!({"id": "unlimited-ocr-q8", "kind": "image", "arch": "unlimited-ocr"}),
            json!({"id": "real-esrgan", "kind": "image", "arch": "realesrgan"}),
            json!({"id": "birefnet", "kind": "image", "arch": "birefnet"}),
            json!({"id": "qwen3.8-27b-gsq-iq3s", "kind": "llm", "arch": "qwen35", "files": {"model": "/m.gguf", "mmproj": "/mm.gguf"}}),
        ]
    }

    #[test]
    fn a_target_is_a_family_or_a_drawing_model() {
        let all = models();
        assert_eq!(resolve_target("natural", &all).unwrap(), ("natural", None));
        assert_eq!(resolve_target("H3", &all).unwrap(), ("h3", None));
        assert_eq!(resolve_target("tags", &all).unwrap(), ("tags", None));
        // registered: by architecture
        assert_eq!(resolve_target("z-image-turbo", &all).unwrap(), ("z-image", Some("z-image-turbo".into())));
        assert_eq!(resolve_target("p-h3-image", &all).unwrap().0, "h3");
        assert_eq!(resolve_target("qwen-image-2.1-turbo-q8", &all).unwrap().0, "qwen-image");
        assert_eq!(resolve_target("moss-image-sft-q8", &all).unwrap().0, "moss-image");
        assert_eq!(resolve_target("ming-image-0.1-design", &all).unwrap().0, "ming-image");
        // not registered here: by the id's prefix (nextsycl's models)
        assert_eq!(resolve_target("minimax-h3-image-q8", &all).unwrap(), ("h3", Some("minimax-h3-image-q8".into())));
        assert_eq!(resolve_target("p-h3-image-q8", &all).unwrap().0, "h3");
        assert_eq!(resolve_target("hunyuanimage-2.1-q8", &all).unwrap().0, "hunyuanimage");
        assert_eq!(resolve_target("z-image-turbo-int8", &all).unwrap().0, "z-image");
        // what does not draw from a prompt, and what is nothing at all
        let e = resolve_target("unlimited-ocr-q8", &all).unwrap_err();
        assert_eq!(e, (400, "target unlimited-ocr-q8: it does not draw pictures from prompts".to_string()));
        for t in ["real-esrgan", "birefnet", "qwen3.8-27b-gsq-iq3s"] {
            assert_eq!(resolve_target(t, &all).unwrap_err().0, 400, "{t}");
        }
        assert!(resolve_target("bogus", &all).unwrap_err().1.starts_with("target bogus: neither a family"));
    }

    #[test]
    fn the_request_is_checked_with_the_fields_named() {
        let all = models();
        let a = ask_of(&json!({"image": "x"}), &all).unwrap();
        assert_eq!((a.family, a.detail, a.words, a.max_tokens, a.negative, a.language.as_str(), a.max_pixels), ("natural", "medium", 90, 220, false, "English", 1_048_576));
        assert_eq!(a.temperature, 0.0);
        let a = ask_of(&json!({"target": "tags", "detail": "short", "include_negative": true}), &all).unwrap();
        assert_eq!((a.words, a.negative, a.max_tokens), (12, true, 2 * 12 + 40 + 80));
        let a = ask_of(&json!({"target": "z-image-turbo", "detail": "long", "include_negative": true}), &all).unwrap();
        assert_eq!((a.words, a.negative_asked, a.negative, a.max_tokens), (170, true, false, 380));
        let a = ask_of(&json!({"max_words": 50, "max_tokens": 300, "seed": 7, "temperature": 0.7}), &all).unwrap();
        assert_eq!((a.words, a.max_tokens, a.seed, a.temperature), (50, 300, Some(7), 0.7));
        let err = |b: Value| ask_of(&b, &all).unwrap_err();
        assert_eq!(err(json!({"max_words": 3})), (400, "max_words 3: from 8 to 400".to_string()));
        assert_eq!(err(json!({"max_words": 401})).1, "max_words 401: from 8 to 400");
        assert_eq!(err(json!({"max_pixels": 1000})).1, "max_pixels 1000: from 65536 to 4194304");
        assert_eq!(err(json!({"max_tokens": 5000})).1, "max_tokens 5000: from 16 to 4096");
        assert_eq!(err(json!({"temperature": 2.5})).1, "temperature 2.5: from 0 to 2");
        assert_eq!(err(json!({"seed": -1})).1, "seed -1: a whole number, 0 or more");
        assert_eq!(err(json!({"detail": "huge"})).1, "detail huge: short, medium or long");
        assert!(err(json!({"language": "x".repeat(41)})).1.starts_with("language: at most 40"));
        assert!(err(json!({"instructions": "x".repeat(2001)})).1.starts_with("instructions: at most 2000"));
        assert!(err(json!({"target": "real-esrgan"})).1.contains("does not draw"));
    }

    #[test]
    fn the_suggested_size_is_on_the_targets_grid() {
        // nextsycl's example: a 1024 x 768 picture for z-image
        assert_eq!(suggested_size("z-image", 1024, 768), (1184, 880));
        assert_eq!(suggested_size("natural", 1000, 1000), (1024, 1024));
        for f in ["natural", "qwen-image", "z-image", "moss-image", "ming-image", "tags"] {
            let (w, h) = suggested_size(f, 1920, 1080);
            assert!(w % 16 == 0 && h % 16 == 0 && (w * h) as f64 <= 1_048_576.0 * 1.05, "{f} {w}x{h}");
        }
        // 4 MP on multiples of 32
        let (w, h) = suggested_size("h3", 1024, 1024);
        assert_eq!((w, h), (2048, 2048));
        let (w, h) = suggested_size("h3", 1920, 1080);
        assert!(w % 32 == 0 && h % 32 == 0 && w > h, "{w}x{h}");
        // the aspect clamped to 4:1, the sides to the range
        let (w, h) = suggested_size("z-image", 10000, 100);
        assert_eq!((w, h), (2048, 512));
        let (w, h) = suggested_size("hunyuanimage", 10000, 100);
        assert_eq!((w, h), (3072, 768));
        let (w, h) = suggested_size("hunyuanimage", 100, 10000);
        assert_eq!((w, h), (768, 3072));
        let (w, h) = suggested_size("h3", 100, 10000);
        assert_eq!((w, h), (1024, 4096));
    }

    #[test]
    fn the_answer_is_cut_at_prompt_and_negative() {
        let (p, n) = cut("PROMPT: A red barn in a green field.\nNEGATIVE: blur, people", true);
        assert_eq!((p.as_str(), n.as_deref()), ("A red barn in a green field.", Some("blur, people")));
        // no negative asked: what follows NEGATIVE: is not the prompt and not kept
        let (p, n) = cut("PROMPT: A red barn.\nNEGATIVE: blur", false);
        assert_eq!((p.as_str(), n), ("A red barn.", None));
        // Markdown, quotes, lines, case
        let (p, n) = cut("**Prompt:** \"A red\nbarn.\"\n\n**Negative prompt:** text, watermark", true);
        assert_eq!((p.as_str(), n.as_deref()), ("A red barn.", Some("text, watermark")));
        // no marker: the whole answer; thinking that came through: dropped
        assert_eq!(cut("A red barn.", false).0, "A red barn.");
        assert_eq!(cut("<think>\nhm\n</think>\n\nPROMPT: A barn.", false).0, "A barn.");
        // text in the picture keeps its quotes
        assert_eq!(cut("PROMPT: A sign reading \"OPEN\" on a door.", false).0, "A sign reading \"OPEN\" on a door.");
        // asked, not written
        assert_eq!(cut("PROMPT: A barn.", true).1, None);
        // a prompt written first with no label, then begun again after one: the whole first one
        let twice = "A glossy green slime figure on a suburban lawn at dusk under a purple sky. PROMPT: A glossy green slime figure on a sub";
        assert_eq!(cut(twice, false).0, "A glossy green slime figure on a suburban lawn at dusk under a purple sky.");
        // a label written twice: only what the first one starts
        assert_eq!(cut("PROMPT: A red barn in a field.\nPROMPT: A red barn in", false).0, "A red barn in a field.");
    }

    #[test]
    fn the_instruction_says_the_family_words_language_and_shape() {
        let all = models();
        let a = ask_of(&json!({"target": "z-image-turbo", "language": "French", "instructions": "ignore the watermark"}), &all).unwrap();
        let t = instruction(&a);
        assert!(t.contains("z-image-turbo") && t.contains("about 90 words") && t.contains("in French") && t.contains("never a list of tags"));
        assert!(t.contains("Also: ignore the watermark") && t.contains("PROMPT:") && !t.contains("NEGATIVE:"));
        let a = ask_of(&json!({"target": "tags", "include_negative": true}), &all).unwrap();
        let t = instruction(&a);
        assert!(t.contains("about 25 tags") && t.contains("comma-separated tags") && t.contains("NEGATIVE:"));
        assert!(instruction(&ask_of(&json!({"target": "h3"}), &all).unwrap()).contains("no other text"));
    }

    #[test]
    fn the_picture_is_seen_at_most_max_pixels() {
        assert_eq!(seen_size(1024, 768, 1_048_576), (1024, 768));
        assert_eq!(seen_size(4000, 3000, 1_048_576), (1182, 886));
        let (w, h) = seen_size(8000, 1000, 65_536);
        assert!(w as u64 * h as u64 <= 65_536 && w / h == 8, "{w}x{h}");
        // a transparent picture on white, made smaller
        let mut p = nextcuda_diffusion::picture::Picture::new(600, 400, 4);
        for px in p.data.chunks_exact_mut(4) {
            px.copy_from_slice(&[255, 0, 0, 0]);
        }
        let png = p.png_bytes(&[]).unwrap();
        let (w, h, sw, sh, out) = see(&format!("data:image/png;base64,{}", crate::http::base64(&png)), 65_536).unwrap();
        assert_eq!((w, h), (600, 400));
        assert!(sw as u64 * sh as u64 <= 65_536);
        let seen = nextcuda_diffusion::picture::Picture::decode(&out).unwrap();
        assert_eq!((seen.channels, &seen.data[..3]), (3, &[255u8, 255, 255][..]));
        assert_eq!(see("not a picture", 1_048_576).unwrap_err().0, 400);
    }

    fn natives() -> Vec<Value> {
        vec![
            json!({"id": "qwen-image-2.1-q8", "kind": "image", "arch": "qwen-image-2.1", "vram_gib": 9.0, "files": {"text-encoder": "/te.gguf", "llm-vision": "/mm.gguf"}}),
            json!({"id": "qwen-image-2.1-rrq4", "kind": "image", "arch": "qwen-image-2.1", "vram_gib": 7.5, "files": {"text-encoder": "/te.gguf", "llm-vision": "/mm.gguf"}}),
            json!({"id": "qwen-image-2512-q4km", "kind": "image", "arch": "qwen-image-2512", "vram_gib": 9.0, "files": {"text-encoder": "/q25.gguf"}}),
            json!({"id": "p-h3-image", "kind": "image", "arch": "p-h3-image", "vram_gib": 10.4, "files": {"text-encoder": "/h3te.gguf", "tokenizer": "/vocab.json"}}),
            json!({"id": "z-image-turbo", "kind": "image", "arch": "z-image", "files": {"text-encoder": "/q3.gguf"}}),
            json!({"id": "qwen3.8-27b-gsq-iq3s", "kind": "llm", "arch": "qwen35", "files": {"model": "/m.gguf", "mmproj": "/mm.gguf"}}),
        ]
    }

    #[test]
    fn native_is_the_targets_own_encoder() {
        let all = natives();
        let ask = |b: Value| ask_of(&b, &all).unwrap();
        // an exact model: its own
        let p = native_for(&ask(json!({"target": "qwen-image-2.1-q8"})), &all, &[]).unwrap();
        assert_eq!(p, Pick::Native { host: "qwen-image-2.1-q8".into(), kind: &NATIVES[0] });
        assert_eq!(native_for(&ask(json!({"target": "p-h3-image"})), &all, &[]).unwrap(), Pick::Native { host: "p-h3-image".into(), kind: &NATIVES[1] });
        // a family: its enabled model with the smallest encoder - the same files here, so a loaded one, else the least memory
        assert_eq!(native_for(&ask(json!({"target": "qwen-image"})), &all, &[]).unwrap(), Pick::Native { host: "qwen-image-2.1-rrq4".into(), kind: &NATIVES[0] });
        assert_eq!(native_for(&ask(json!({"target": "qwen-image"})), &all, &["qwen-image-2.1-q8".into()]).unwrap(),
                   Pick::Native { host: "qwen-image-2.1-q8".into(), kind: &NATIVES[0] });
        // refused, field describer
        let e = native_for(&ask(json!({"target": "z-image"})), &all, &[]).unwrap_err();
        assert_eq!(e.0, 400);
        assert_eq!(field_untag(&e.1), Some(("describer", "target z-image: its text encoder (Qwen3-4B) does not read pictures; use describer best")));
        let e = native_for(&ask(json!({"target": "qwen-image-2512-q4km"})), &all, &[]).unwrap_err();
        assert!(field_untag(&e.1).unwrap().1.starts_with("target qwen-image-2512-q4km: its text encoder (Qwen2.5-VL-7B"), "{}", e.1);
        assert_eq!(native_for(&ask(json!({"target": "natural"})), &all, &[]).unwrap_err().0, 400);
        // disabled: a conflict
        let mut off = all.clone();
        off[0]["enabled"] = json!(false);
        assert_eq!(native_for(&ask(json!({"target": "qwen-image-2.1-q8"})), &off, &[]).unwrap_err().0, 409);
    }

    #[test]
    fn a_native_needs_its_files_and_names_read_short() {
        let all = natives();
        assert_eq!(native_of(&all[0]), Some(&NATIVES[0]));
        assert_eq!(native_of(&all[2]), None);
        assert_eq!(native_of(&all[3]), Some(&NATIVES[1]));
        assert_eq!(native_of(&json!({"id": "x", "kind": "image", "arch": "qwen-image-2.1", "files": {"text-encoder": "/te.gguf"}})), None);
        assert_eq!(chat_name("qwen3.8-27b-gsq-iq3s"), "Qwen3.8-27B");
        assert_eq!(chat_name("qwen3.8-27b-q4-describe"), "Qwen3.8-27B");
        let t = with_field("describer", "target x: y");
        assert_eq!(field_untag(&t), Some(("describer", "target x: y")));
        assert_eq!(field_untag("target x: y"), None);
    }

    #[test]
    fn a_describer_reads_pictures() {
        let all = models();
        assert!(reads_pictures(&all[8]));
        assert!(!reads_pictures(&json!({"id": "nemotron", "kind": "llm", "files": {"model": "/n.gguf"}})));
        assert!(!reads_pictures(&json!({"id": "q", "kind": "llm", "files": {"mmproj": "/mm.gguf"}, "env": {"NS_Q35_VISION": "0"}})));
        assert!(!reads_pictures(&all[0]));
    }
}
