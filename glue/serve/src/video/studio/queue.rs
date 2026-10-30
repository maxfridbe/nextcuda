//! The clip queue: what a request becomes (a queue item), holds (all, per project, per batch), projects, and the
//! files they persist to. The item shape is the legacy server's, so scene files and tools written for it work
//! unchanged (docs/LEGACY-API.md).

use std::collections::BTreeSet;
use std::path::Path;

use serde_json::{json, Map, Value};

use super::{RpcError, Studio};

pub const MAGIC: [&str; 4] = ["prev", "prev_cam", "first", "first_cam"];
pub const CHAIN_MODES: [(&str, &str); 4] = [
    ("video", "the previous clip's mp4: its last frame decoded and encoded again"),
    ("png", "the previous clip's lossless last frame (.last.png)"),
    ("latent", "the previous clip's last latent frame (no decode, no codec)"),
    ("none", "no chaining"),
];

/// The holds and the queue, persisted.
#[derive(Default)]
pub struct Queue {
    pub items: Vec<Value>,
    pub paused: bool,
    pub paused_projects: BTreeSet<String>,
    pub paused_batches: BTreeSet<String>,
}

pub fn s(v: &Value, k: &str) -> String {
    v.get(k).and_then(|x| x.as_str()).unwrap_or("").to_string()
}

/// "Name 03/12: ..." -> "Name": the series a clip belongs to (the text before the clip number).
pub fn label_prefix(label: &str) -> String {
    let bytes = label.as_bytes();
    for (i, c) in label.char_indices() {
        if c.is_ascii_digit() {
            // digits, then '/'
            let j = label[i..].find(|ch: char| !ch.is_ascii_digit()).map_or(label.len(), |k| i + k);
            if j < bytes.len() && bytes[j] == b'/' && (i == 0 || !label[..i].ends_with(|ch: char| ch.is_ascii_digit())) {
                return label[..i].trim().to_string();
            }
        }
    }
    String::new()
}

/// "Name 03/12: ..." -> 3: the edit order.
pub fn label_number(label: &str) -> Option<u64> {
    for (i, c) in label.char_indices() {
        if c.is_whitespace() {
            let rest = &label[i + c.len_utf8()..];
            let digits: String = rest.chars().take_while(|d| d.is_ascii_digit()).collect();
            if !digits.is_empty() && rest[digits.len()..].starts_with('/') && rest[digits.len() + 1..].starts_with(|d: char| d.is_ascii_digit()) {
                return digits.parse().ok();
            }
        }
    }
    None
}

/// A clip's place in the edit: the `clip` its request stated; for clips made before that key, the label's "NN/MM"
/// number, else the pass number the creative platform put after the `#` of its `reentrant_id`
/// ("<run>:shots:0:0#17" is the 17th; its labels, "<run>:shots:0:017", carry no "NN/MM"), else 0.
pub fn clip_number(item: &Value) -> u64 {
    item["clip"].as_u64().filter(|c| *c > 0).or_else(|| label_number(&s(item, "label"))).or_else(|| {
        let id = s(item, "reentrant_id");
        let tail = id.rsplit_once('#')?.1;
        tail.chars().all(|c| c.is_ascii_digit()).then(|| tail.parse().ok()).flatten()
    }).unwrap_or(0)
}

/// The project an item belongs to: its own, or its label's series.
pub fn project_of(item: &Value) -> String {
    let p = s(item, "project");
    if !p.is_empty() {
        return p;
    }
    let label = s(item, "label");
    // the reference requires "<prefix> NN/MM"
    match label_number(&label) {
        Some(_) => label_prefix(&label),
        None => String::new(),
    }
}

/// The spoken lines of a prompt ("<d>[English] ...</d>"), joined.
pub fn dialogue_of(prompt: &str) -> String {
    let mut out = Vec::new();
    let mut rest = prompt;
    while let Some(a) = rest.find("<d>[") {
        let after = &rest[a + 4..];
        let Some(close) = after.find(']') else { break };
        let body = after[close + 1..].trim_start();
        let Some(end) = body.find("</d>") else { break };
        out.push(body[..end].split_whitespace().collect::<Vec<_>>().join(" "));
        rest = &body[end + 4..];
    }
    out.join(" / ")
}

/// The legacy label filter: word characters and ` -.,:'()!?/`, 80 at most.
fn clean_label(l: &str) -> String {
    let kept: String = l.chars().filter(|c| c.is_alphanumeric() || *c == '_' || " -.,:'()!?/".contains(*c)).collect();
    kept.chars().take(80).collect::<String>().trim().to_string()
}

pub fn check_canvas(w: i64, h: i64) -> Result<(), RpcError> {
    if w % 32 != 0 || h % 32 != 0 {
        return Err(RpcError::param("width and height must be multiples of 32"));
    }
    if !(256..=1344).contains(&w) || !(256..=1344).contains(&h) {
        return Err(RpcError::param("width and height must be between 256 and 1344"));
    }
    if w * h > 768 * 1344 {
        return Err(RpcError::param(&format!("area {} exceeds the model's 1032192 pixel cap", w * h)));
    }
    Ok(())
}

/// Frame counts the model makes: 17k + 5, from 22 (0.92 s) to 362 (15.08 s) at 24 fps
pub const FRAMES: (u64, u64) = (22, 362);

/// `frames`: one the model makes, else the error naming the nearest ones
pub fn check_frames(f: u64) -> Result<u64, RpcError> {
    if f % 17 == 5 && (FRAMES.0..=FRAMES.1).contains(&f) {
        return Ok(f);
    }
    let down = (FRAMES.0..=f.min(FRAMES.1)).rev().find(|n| n % 17 == 5);
    let up = (f.max(FRAMES.0)..=FRAMES.1).find(|n| n % 17 == 5);
    let near: Vec<String> = down.into_iter().chain(up).collect::<BTreeSet<_>>().into_iter().map(|n| n.to_string()).collect();
    Err(RpcError::param(&format!("frames {f}: the model makes 17k + 5 frames ({}-{}); nearest: {}", FRAMES.0, FRAMES.1, near.join(" or "))))
}

/// The keys a chain object stands for. `chain: {from, context_frames, sound, still, mode, color_reanchor}`:
/// - context_frames N (17k + 5): the last N frames of `from` as one moving keyframe at frame 0 (guide_clip) - the
///   clip replays them, so a join trims N / 2 each side; with `sound` (default true) the sound's last N / 24 s too
///   (first_audio, first_audio_s)
/// - still (default: when context_frames is 0): `from`'s last frame as the first frame (first_frame, chain_mode `mode`)
/// - color_reanchor: an anchor or picture whose brightness the carried pictures are scaled to (exposure_ref)
///
/// Keys given flat as well win. pin_renorm and keys the box does not have are listed back as ignored.
pub fn unchain(p: &Value) -> Result<(Value, Vec<String>), RpcError> {
    let mut out = p.clone();
    let mut ignored = Vec::new();
    let Some(c) = p.get("chain").filter(|c| !c.is_null()) else { return Ok((out, ignored)) };
    let c = c.as_object().ok_or_else(|| RpcError::param("chain: an object {from, context_frames, sound, still, mode, color_reanchor}"))?;
    let from = c.get("from").and_then(|v| v.as_str()).unwrap_or("prev").to_string();
    let n = c.get("context_frames").map_or(Ok(0), |v| v.as_u64().ok_or_else(|| RpcError::param("chain.context_frames: a whole number")))?;
    if n > 0 && (n % 17 != 5 || n > FRAMES.1) {
        let near = check_frames(n).err().map(|e| e.message).unwrap_or_default();
        return Err(RpcError::param(&format!("chain.context_frames {n}: 5, 22, 39 ... (17k + 5) - {}", near.rsplit("; ").next().unwrap_or(""))));
    }
    let set = |out: &mut Value, k: &str, v: Value| {
        if out.get(k).is_none_or(|x| x.is_null()) {
            out[k] = v;
        }
    };
    // a clip made to a soundtrack has its sound already: the chain carries the picture only
    let soundtrack = p.get("soundtrack").is_some_and(|v| v.as_str().is_some_and(|s| !s.is_empty()));
    if n > 0 {
        set(&mut out, "guide_clip", json!(format!("{from}:{n}:0")));
        if !soundtrack && c.get("sound").and_then(|v| v.as_bool()) != Some(false) {
            set(&mut out, "first_audio", json!(from));
            set(&mut out, "first_audio_s", json!(n as f64 / 24.0));
        }
    }
    if c.get("still").and_then(|v| v.as_bool()).unwrap_or(n == 0) {
        set(&mut out, "first_frame", json!(from));
        if let Some(m) = c.get("mode") {
            set(&mut out, "chain_mode", m.clone());
        }
    }
    if let Some(r) = c.get("color_reanchor").filter(|v| !v.is_null()) {
        let r = match r {
            Value::Bool(true) => json!(from),
            Value::Bool(false) => Value::Null,
            v => v.clone(),
        };
        set(&mut out, "exposure_ref", r);
    }
    for k in c.keys().filter(|k| !["from", "context_frames", "sound", "still", "mode", "color_reanchor"].contains(&k.as_str())) {
        ignored.push(format!("chain.{k}"));
    }
    if let Some(o) = out.as_object_mut() {
        o.remove("chain");
    }
    Ok((out, ignored))
}

/// Request keys the box takes and does nothing with: H3 runs at cfg 1 (one pass, no unconditional branch), so
/// there is nothing for a negative prompt to steer
pub const IGNORED: [&str; 2] = ["negative_prompt", "cfg"];

/// What each denoiser takes (engines.list): the text/image-to-video ones and Ref2VA
/// Keys accepted and listed back in `ignored`: nothing on this box reads them (cfg is 1, a single pass a step; no
/// pin renormalisation)
fn ignored_keys() -> Vec<String> {
    IGNORED.iter().map(|k| k.to_string()).chain(["chain.pin_renorm".to_string()]).collect()
}

pub fn inputs_of(quant: &str) -> Value {
    let common = ["prompt", "seconds", "frames", "steps", "seed", "width", "height", "te", "first_frame", "last_frame", "exposure_ref", "guide_clip",
                  "first_audio", "first_audio_s", "soundtrack", "soundtrack_start", "ref_audios", "loras", "upscale", "upscaler", "shift_video", "shift_audio", "sampler", "schedule", "chain", "source", "regen", "regen_box"];
    if quant == "REF2VA" {
        json!({"takes": common.iter().chain(["ref_images", "ref_videos", "ref_image_size", "ref_video_sound", "te_pictures"].iter()).collect::<Vec<_>>(),
               "not": ["control_video", "control_mask", "control_source", "control_strength", "control_start", "control_end"],
               "chosen": "by itself when ref_images or ref_videos are given", "ignored": ignored_keys()})
    } else {
        json!({"takes": common.iter().chain(["control_video", "control_mask", "control_source", "control_strength", "control_start", "control_end"].iter()).collect::<Vec<_>>(),
               "not": ["ref_images", "ref_videos"], "chosen": if quant == "INT8" { "by default" } else { "by engine" }, "ignored": ignored_keys()})
    }
}

/// A number parameter: the default when absent, rejected (not clamped) outside [lo, hi].
pub fn num(p: &Value, key: &str, default: f64, lo: f64, hi: f64, int: bool) -> Result<Value, RpcError> {
    let v = match p.get(key) {
        None | Some(Value::Null) => default,
        Some(Value::Number(n)) => n.as_f64().unwrap_or(default),
        Some(Value::Bool(b)) => *b as i64 as f64,
        Some(Value::String(t)) => t.trim().parse::<f64>().map_err(|_| RpcError::param(&format!("{key} must be a number")))?,
        _ => return Err(RpcError::param(&format!("{key} must be a number"))),
    };
    let v = if int { v.trunc() } else { v };
    if !(lo..=hi).contains(&v) {
        return Err(RpcError::param(&format!("{key} must be between {lo} and {hi}")));
    }
    Ok(if int { json!(v as i64) } else { json!(v) })
}

/// A request's denoiser -> its name in the front end's terms: INT8 (the default; the legacy Q8_0 was its counterpart),
/// or a GGUF form the engine was given (Q6_K, Q4_K_M).
pub fn engine_name(e: Option<&str>) -> String {
    // LTX-2.5's ids are lower case and its registry names (ltx-2.5, ltx-2.5-fast) are read as the engine too
    match e.map(|e| e.trim().to_lowercase()).as_deref() {
        Some("ltx2" | "ltx-2.5" | "ltx-2.5-fast" | "ltx-2.5-dev") => return LTX.into(),
        Some("ltx2_upscale" | "ltx-2.5-upscale") => return LTX_UPSCALE.into(),
        _ => {}
    }
    match e.map(|e| e.trim().to_uppercase()).filter(|e| !e.is_empty()).as_deref() {
        // none named: INT8 when the daemon has it (or nothing is known yet), else the first it has (nextcuda: Q4_K_M)
        None => default_engine(),
        // asked for by name but not on this box (nextcuda: a front end defaults to INT8, the box has Q4_K_M only): the
        // denoiser it has, rather than a job the daemon fails with "no model INT8"
        Some("Q8_0") | Some("INT8") => default_engine(),
        Some(e) => e.to_string(),
    }
}

/// nextcuda: the front end's denoiser names (INT8, Q6_K, Q4_K_M) and the registry ids of the models the daemon has
/// (`--model`, `--engine`), set once by the studio's start
pub static ENGINES: std::sync::OnceLock<Vec<(String, String)>> = std::sync::OnceLock::new();

/// nextcuda: each video model's licence by its registry id, set once by the studio's start
pub static LICENSES: std::sync::OnceLock<Vec<(String, Value)>> = std::sync::OnceLock::new();

/// A video model's licence keys (`license`, `commercial`, `license_note`), none when the registry has no such model
pub fn licence_of(id: &str) -> Option<Value> {
    LICENSES.get()?.iter().find(|(i, _)| i == id).map(|(_, l)| l.clone()).filter(|l| l.is_object())
}

/// Models the daemon lists that the start did not name (named by their suffix: `quant_of`)
static LEARNED: std::sync::Mutex<Vec<(String, String)>> = std::sync::Mutex::new(Vec::new());

/// The daemon's model ids, as it lists them: any the start did not name join under their suffix's name
pub fn learn_engines(ids: &[String]) {
    let mut l = LEARNED.lock().unwrap_or_else(|e| e.into_inner());
    for id in ids {
        let named = ENGINES.get().is_some_and(|m| m.iter().any(|(_, i)| i == id)) || l.iter().any(|(_, i)| i == id);
        let q = quant_of(id).to_string();
        // LTX-2.5's models (dev and distilled) share the name ltx2: each is kept, the dev one is told by its id
        if !named && (q == LTX || !ENGINES.get().is_some_and(|m| m.iter().any(|(n, _)| *n == q))) {
            l.push((q, id.clone()));
        }
    }
}

/// Every name -> id pair: the start's, then the learned ones
fn engine_pairs() -> Vec<(String, String)> {
    let mut v = ENGINES.get().cloned().unwrap_or_default();
    v.extend(LEARNED.lock().unwrap_or_else(|e| e.into_inner()).iter().cloned());
    v
}

/// The front end's name for a model id: by its suffix (minimax-h3 -> INT8, -q6k -> Q6_K, -q4km -> Q4_K_M, -q8 -> Q8_0,
/// -ref2va -> REF2VA); LTX-2.5's models (ltx-2.5, ltx-2.5-fast) are the one engine `ltx2`
pub fn quant_of(id: &str) -> &'static str {
    if id.starts_with("ltx") {
        return LTX;
    }
    match id.rsplit('-').next().unwrap_or("") {
        "q6k" => "Q6_K",
        "q4km" => "Q4_K_M",
        "q8" => "Q8_0",
        "ref2va" => "REF2VA",
        _ => "INT8",
    }
}

/// The denoiser a request without one gets: INT8 when the daemon has it or names none, else its first
pub fn default_engine() -> String {
    let pairs = engine_pairs();
    if pairs.is_empty() || pairs.iter().any(|(n, _)| n == "INT8") { "INT8".into() } else { pairs[0].0.clone() }
}

/// The front-end names of the models the daemon has (INT8 unless it has others and not that one)
pub fn engine_names() -> Vec<String> {
    let mut v = vec![default_engine()];
    for (q, _) in engine_pairs() {
        if !v.contains(&q) {
            v.push(q);
        }
    }
    if v.iter().any(|n| n == LTX) {
        v.push(LTX_UPSCALE.to_string());
    }
    v
}

/// Whether the daemon has a model of that front-end name (the LTX upscaler is made by the LTX models)
pub fn engine_known(name: &str) -> bool {
    let name = if name == LTX_UPSCALE { LTX } else { name };
    engine_pairs().iter().any(|(q, _)| q == name)
}

/// A front-end name -> the daemon's model id (an unknown name passes as it is)
pub fn engine_id(name: &str) -> String {
    engine_pairs().into_iter().find(|(q, _)| q == name).map_or_else(|| name.to_string(), |(_, id)| id)
}

/// A daemon's model id -> the front end's name
pub fn engine_quant(id: &str) -> String {
    engine_pairs().into_iter().find(|(_, i)| i == id).map_or_else(|| id.to_string(), |(q, _)| q)
}

/// LTX-2.5's engine ids, as video.engines.list names them and video.generate's `engine` takes them: the shots, and
/// the latent upscaler that makes a finished clip twice the size (the creative platform addresses it on its own)
pub const LTX: &str = "ltx2";
pub const LTX_UPSCALE: &str = "ltx2_upscale";

/// Whether an engine name (as engine_name settled it) is one of LTX-2.5's
pub fn is_ltx(engine: &str) -> bool {
    engine == LTX || engine == LTX_UPSCALE
}

/// The model variants of LTX-2.5 a request names: `dev` (30 guided steps) or `distilled` (8, no guidance)
pub const LTX_VARIANTS: [&str; 2] = ["dev", "distilled"];
/// Steps by variant when a request names none
pub const LTX_STEPS: (i64, i64) = (30, 8);
/// Frames LTX makes: any count from nine (the engine makes the next 8k + 1 and cuts back to what was asked)
pub const LTX_FRAMES: (u64, u64) = (9, 361);
/// The canvas side: a multiple of 32
pub const LTX_SIDE: (i64, i64) = (256, 1344);
/// The tokens a clip may take (width/32 x height/32 x latent frames). The DiT's working memory barely grows with them
/// (flash attention: 9.7 GiB on the 5070 at 3,840 tokens, 9.8 at 9,216), the time does: 12.8 s a step at 640 x 384 x 121
/// (3,840), 22.9 at 768 x 512 (6,144), 39.4 at 1024 x 576 (9,216: the largest run on the 12 GB pair)
pub const LTX_MAX_TOKENS: i64 = 9216;

/// LTX's latent frames of a frame count: the count made as the next 8k + 1, (that - 1) / 8 + 1
pub fn ltx_latent_frames(frames: u64) -> i64 {
    (frames.max(9) as i64 - 1 + 7) / 8 + 1
}

/// The DiT tokens of a canvas and length
pub fn ltx_tokens(w: i64, h: i64, frames: u64) -> i64 {
    (w / 32) * (h / 32) * ltx_latent_frames(frames)
}

/// A canvas LTX makes: multiples of 32 within the side limits, and few enough tokens for the cards
pub fn check_ltx_canvas(w: i64, h: i64, frames: u64) -> Result<(), RpcError> {
    if w % 32 != 0 || h % 32 != 0 {
        return Err(RpcError::param("width and height must be multiples of 32"));
    }
    if !(LTX_SIDE.0..=LTX_SIDE.1).contains(&w) || !(LTX_SIDE.0..=LTX_SIDE.1).contains(&h) {
        return Err(RpcError::param(&format!("width and height must be between {} and {}", LTX_SIDE.0, LTX_SIDE.1)));
    }
    let t = ltx_tokens(w, h, frames);
    if t > LTX_MAX_TOKENS {
        return Err(RpcError::param(&format!("{w}x{h} for {frames} frames is {t} tokens; the engine takes at most {LTX_MAX_TOKENS} (640x384 for 121 frames is 3840, 1024x576 is 9216): a smaller canvas or fewer frames")));
    }
    Ok(())
}

/// `frames` for LTX: whole, from nine
pub fn check_ltx_frames(f: u64) -> Result<u64, RpcError> {
    if !(LTX_FRAMES.0..=LTX_FRAMES.1).contains(&f) {
        return Err(RpcError::param(&format!("frames {f}: LTX makes {} to {} frames (any count; the model works on 8k + 1 and the clip is cut to what was asked)", LTX_FRAMES.0, LTX_FRAMES.1)));
    }
    Ok(f)
}

/// The daemon's LTX model ids, as the start named them and the daemon listed them
pub fn ltx_ids() -> Vec<String> {
    engine_pairs().into_iter().filter(|(q, _)| q == LTX).map(|(_, id)| id).collect()
}

/// The model id for an LTX variant among the daemon's models (ids starting `ltx`; the dev one is the id without
/// `-fast`): the dev variant needs the dev model; the distilled can be made by either (the request says `fast`)
pub fn ltx_model(ids: &[String], variant: &str) -> Result<String, RpcError> {
    let ltx: Vec<&String> = ids.iter().filter(|i| i.starts_with("ltx")).collect();
    let dev = ltx.iter().find(|i| !i.ends_with("-fast")).map(|i| i.to_string());
    match (variant, dev) {
        ("dev", Some(d)) => Ok(d),
        ("dev", None) => Err(RpcError::param("variant dev: this box has only the distilled LTX-2.5 model (ltx-2.5-fast); pull ltx-2.5")),
        (_, Some(d)) => Ok(d),
        (_, None) => ltx.first().map(|i| i.to_string()).ok_or_else(|| RpcError::param("engine ltx2? (this box has no LTX-2.5 model: nextcuda models pull ltx-2.5)")),
    }
}

/// The keys LTX's engines do not read: refused with the key named, rather than accepted and ignored. (Its sound is
/// its own; a clip carries on from another only by a first frame; the latent upscaler is the engine ltx2_upscale.)
pub const LTX_NOT: [&str; 22] = ["guide_clip", "first_audio", "first_audio_s", "last_frame", "exposure_ref", "soundtrack", "soundtrack_start", "ref_audios",
                                 "ref_images", "ref_videos", "ref_image_size", "ref_video_sound", "te_pictures", "control_video", "control_mask",
                                 "control_source", "source", "regen", "regen_box", "shift_video", "shift_audio", "cond_noise_aug"];

/// What video.engines.list says LTX's shot engine takes: the keys, the choices, the ranges
pub fn ltx_inputs(variants: &[&str]) -> Value {
    json!({"takes": ["prompt", "negative_prompt", "frames", "seconds", "steps", "seed", "width", "height", "variant", "first_frame", "label", "reentrant_id",
                     "project", "batch", "clip", "camera", "priority", "queue"],
           "not": LTX_NOT.iter().chain(["loras", "sampler", "schedule", "upscale", "upscaler"].iter()).collect::<Vec<_>>(),
           "chosen": "engine: ltx2", "ignored": ["cfg", "te", "chain.pin_renorm"],
           "options": {
               "variant": {"values": variants, "default": variants.first(), "help": "dev: the take (30 guided steps, four passes a step); distilled: a preview (8 steps, unguided: one pass a step, about 80 s a clip; softer, its sound as much a draw)"},
               "steps": {"min": 1, "max": 40, "default": {"dev": LTX_STEPS.0, "distilled": LTX_STEPS.1},
                         "help": "distilled with no steps named runs its own eight sigmas; naming steps runs the ordinary schedule"},
               "frames": {"min": LTX_FRAMES.0, "max": LTX_FRAMES.1, "step": 1, "default": 121, "help": "any count; made as the next 8k + 1 and cut back; seconds x 24 when frames are not given"},
               "width": {"min": LTX_SIDE.0, "max": LTX_SIDE.1, "multiple_of": 32, "default": 640}, "height": {"min": LTX_SIDE.0, "max": LTX_SIDE.1, "multiple_of": 32, "default": 384},
               "tokens": {"max": LTX_MAX_TOKENS, "help": "width/32 x height/32 x ((frames - 1)/8 + 1) at most; 640x384 for 121 frames is 3840"},
               "seed": {"min": -2147483648i64, "max": 2147483648i64, "default": 0},
               "first_frame": {"help": "a picture (a file in out/, video.upload): the clip starts on it (image to video); an anchor word (prev, first ...) is a clip's mp4 and is not read here"}},
           "fps": 24, "audio": "the mp4 carries the model's own stereo sound at 48 kHz, made from the prompt"})
}

/// The rows video.engines.list gives LTX-2.5 from the daemon's LTX models ((id, path, gib), the licence of the first):
/// the shot engine `ltx2` and the upscaler `ltx2_upscale`, in the H3 rows' shape (quant is what video.generate's
/// `engine` takes; `engine` and `stage` say what it is; `options` the choices); none when the daemon has no LTX model
pub fn ltx_rows(models: &[(String, String, f64)], licence: &Option<Value>) -> Vec<Value> {
    let Some(first) = models.iter().find(|m| !m.0.ends_with("-fast")).or_else(|| models.first()) else { return Vec::new() };
    let has_dev = models.iter().any(|m| !m.0.ends_with("-fast"));
    let variants: Vec<&str> = if has_dev { vec!["dev", "distilled"] } else { vec!["distilled"] };
    let file = first.1.rsplit('/').next().unwrap_or(&first.1).to_string();
    let mut shots = json!({"quant": LTX, "engine": LTX, "stage": "shots", "file": file, "path": first.1, "gib": (first.2 * 100.0).round() / 100.0, "ready": true,
                           "default": false, "inputs": ltx_inputs(&variants), "models": models.iter().map(|m| m.0.clone()).collect::<Vec<_>>()});
    // the upscaler holds the VAE (1.4 GiB) and the latent upsampler (1.0 GiB) and no DiT
    let mut up = json!({"quant": LTX_UPSCALE, "engine": LTX_UPSCALE, "stage": "upscale", "file": "ltx-2.5 latent upsampler + video VAE", "path": first.1, "gib": 2.4,
                        "ready": true, "default": false, "inputs": ltx_upscale_inputs(), "models": models.iter().map(|m| m.0.clone()).collect::<Vec<_>>()});
    for row in [&mut shots, &mut up] {
        if let Some(l) = licence.as_ref().and_then(|l| l.as_object()) {
            row.as_object_mut().unwrap().extend(l.clone());
        }
    }
    vec![shots, up]
}

/// What video.engines.list says LTX's upscaler takes
pub fn ltx_upscale_inputs() -> Value {
    json!({"takes": ["source_clip", "label", "reentrant_id", "project", "batch", "clip", "camera", "priority", "queue"],
           "not": ["prompt", "frames", "seconds", "steps", "seed", "width", "height", "first_frame"],
           "chosen": "engine: ltx2_upscale", "ignored": [],
           "options": {"scale": {"values": [2], "default": 2, "help": "the latent upsampler doubles each side; the clip is windowed 121 frames at a time, a tail under nine frames is dropped"},
                       "source_clip": {"help": "a finished clip's file in out/ (name or name.mp4, 8-bit H.264, sides multiples of 32); the sound is copied"}},
           "stage": "upscale"})
}

impl Studio {
    /// A request -> a queue item, checked (the legacy `build_item`, plus the engine's new options).
    pub fn build_item(&self, p: &Value) -> Result<Value, RpcError> {
        let engine = engine_name(p.get("engine").and_then(|v| v.as_str()));
        if is_ltx(&engine) {
            return self.build_ltx_item(p, &engine);
        }
        let prompt = p.get("prompt").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
        if prompt.is_empty() {
            return Err(RpcError::param("prompt is required"));
        }
        let te = p.get("te").and_then(|v| v.as_str()).unwrap_or("teacher");
        if te != "teacher" && te != "student" {
            return Err(RpcError::param("te must be teacher or student"));
        }
        let width = num(p, "width", 768.0, 256.0, 1344.0, true)?;
        let height = num(p, "height", 576.0, 256.0, 1344.0, true)?;
        check_canvas(width.as_i64().unwrap_or(0), height.as_i64().unwrap_or(0))?;
        let chain_mode = p.get("chain_mode").and_then(|v| v.as_str()).unwrap_or("video");
        if !CHAIN_MODES.iter().any(|(m, _)| *m == chain_mode) {
            return Err(RpcError::param("chain_mode must be one of video, png, latent, none"));
        }
        // the sampler and schedule by ComfyUI's names (none: the engine's own Euler on its shifted schedule)
        let named = |k: &str, ok: &dyn Fn(&str) -> bool, names: &str| -> Result<Value, RpcError> {
            match p.get(k).and_then(|v| v.as_str()).map(str::trim).filter(|v| !v.is_empty()) {
                None => Ok(Value::Null),
                Some(v) if ok(v) => Ok(json!(v)),
                Some(v) => Err(RpcError::param(&format!("{k} {v}? ({names})"))),
            }
        };
        let sampler = named("sampler", &|v| nextcuda_diffusion::Sampler::parse(v).is_some(), &nextcuda_diffusion::samplers::NAMES.join(", "))?;
        let schedule = named("schedule", &|v| nextcuda_diffusion::Schedule::parse(v).is_some(),
                             &nextcuda_diffusion::Schedule::ALL.iter().map(|s| s.name()).collect::<Vec<_>>().join(", "))?;
        let mut it = Map::new();
        it.insert("prompt".into(), json!(prompt));
        it.insert("sampler".into(), sampler);
        it.insert("schedule".into(), schedule);
        // frames, when given, set the length exactly (seconds = frames / 24); else seconds snapped up to the grid
        match p.get("frames").filter(|v| !v.is_null()) {
            Some(f) => {
                let f = check_frames(f.as_u64().ok_or_else(|| RpcError::param("frames: a whole number"))?)?;
                it.insert("seconds".into(), json!(f as f64 / 24.0));
                it.insert("frames".into(), json!(f));
            }
            None => {
                let sec = num(p, "seconds", 10.0, 1.0, 15.1, false)?;
                it.insert("frames".into(), json!(super::runner::frames_of(sec.as_f64().unwrap_or(10.0))));
                it.insert("seconds".into(), sec);
            }
        }
        it.insert("reentrant_id".into(), reentrant(p).map_or(Value::Null, |r| json!(r)));
        // the clip's place in its project's edit (1 and up): the timeline's n, stated rather than read from the label
        match p.get("clip").filter(|v| !v.is_null()) {
            Some(c) => match c.as_u64().filter(|c| (1..=100_000).contains(c)) {
                Some(c) => it.insert("clip".into(), json!(c)),
                None => return Err(RpcError::param("clip: a whole number from 1 (the clip's place in its project's timeline)")),
            },
            None => it.insert("clip".into(), Value::Null),
        };
        it.insert("steps".into(), num(p, "steps", 10.0, 1.0, 40.0, true)?);
        it.insert("seed".into(), num(p, "seed", 0.0, -2147483648.0, 2147483648.0, true)?);
        it.insert("te".into(), json!(te));
        it.insert("label".into(), json!(clean_label(&p.get("label").map(|v| v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string())).unwrap_or_default())));
        let engine = engine_name(p.get("engine").and_then(|v| v.as_str()));
        if engine != "INT8" && !engine_known(&engine) {
            let _ = self.engines(); // the daemon's list (a model added since the studio started)
            if !engine_known(&engine) {
                return Err(RpcError::param(&format!("engine {engine}? (video.engines.list: {})", engine_names().join(", "))));
            }
        }
        it.insert("engine".into(), json!(engine));
        it.insert("priority".into(), num(p, "priority", 0.0, -9.0, 9.0, true)?);
        it.insert("width".into(), width);
        it.insert("height".into(), height);
        it.insert("chain_mode".into(), json!(chain_mode));
        for k in ["cond_noise_aug", "camera", "exposure_ref", "last_frame", "first_audio", "first_audio_s", "upscale", "upscaler", "guide_clip", "shift_video",
                  "shift_audio", "source", "regen", "regen_box"] {
            it.insert(k.into(), p.get(k).cloned().unwrap_or(Value::Null));
        }
        for k in ["project", "batch"] {
            let v = p.get(k).and_then(|v| v.as_str()).unwrap_or("");
            it.insert(k.into(), json!(v.chars().take(60).collect::<String>().trim()));
        }
        let loras: Vec<String> = p.get("loras").and_then(|v| v.as_array()).map(|a| a.iter().map(|x| x.as_str().map(str::to_string).unwrap_or_else(|| x.to_string()).chars().take(200).collect()).collect()).unwrap_or_default();
        it.insert("loras".into(), json!(loras));
        // a first frame that is not an anchor word is a file in out/
        let ff = p.get("first_frame").and_then(|v| v.as_str()).unwrap_or("");
        let ff = if ff.is_empty() {
            Value::Null
        } else if MAGIC.contains(&ff) {
            json!(ff)
        } else {
            let b = basename(ff);
            if !self.out.join(&b).exists() {
                return Err(RpcError::not_found(&format!("first_frame {b} not found")));
            }
            json!(b)
        };
        it.insert("first_frame".into(), ff);
        for (k, max, what) in [("ref_images", 9, "ref_image"), ("ref_videos", 3, "ref_video"), ("ref_audios", 3, "ref_audio")] {
            let list: Vec<String> = p.get(k).and_then(|v| v.as_array()).map(|a| a.iter().filter_map(|x| x.as_str()).map(basename).collect()).unwrap_or_default();
            if list.len() > max {
                return Err(RpcError::param(&format!("{k}: at most {max} ({} given)", list.len())));
            }
            for f in &list {
                if !self.out.join(f).exists() {
                    return Err(RpcError::not_found(&format!("{what} {f} not found in out/")));
                }
            }
            it.insert(k.into(), if list.is_empty() { Value::Null } else { json!(list) });
        }
        // pictures and clips as references: the Ref2VA denoiser, the one trained for them
        if it["ref_images"].is_array() || it["ref_videos"].is_array() {
            let _ = self.engines(); // the daemon's list (a model added since the studio started)
            if !engine_known("REF2VA") {
                return Err(RpcError::param("reference pictures and clips need the Ref2VA denoiser (nextcuda models pull minimax-h3-ref2va, then add it to NS_VIDEO_ENGINES)"));
            }
            it.insert("engine".into(), json!("REF2VA"));
        }
        for k in ["ref_video_sound", "te_pictures"] {
            it.insert(k.into(), p.get(k).filter(|v| v.is_boolean()).cloned().unwrap_or(Value::Null));
        }
        // a driving soundtrack: a sound file beside the clips, from a start second
        match p.get("soundtrack").and_then(|v| v.as_str()).map(basename).filter(|v| !v.is_empty()) {
            Some(f) => {
                if !self.out.join(&f).exists() {
                    return Err(RpcError::not_found(&format!("soundtrack {f} not found in out/ (video.upload it)")));
                }
                if p.get("first_audio").is_some_and(|v| !v.is_null()) {
                    return Err(RpcError::param("soundtrack is the clip's whole sound: not with first_audio"));
                }
                it.insert("soundtrack".into(), json!(f));
                it.insert("soundtrack_start".into(), num(p, "soundtrack_start", 0.0, 0.0, 86400.0, false)?);
            }
            None => {
                it.insert("soundtrack".into(), Value::Null);
                it.insert("soundtrack_start".into(), Value::Null);
            }
        }
        // the ControlNet: a control video, a mask over a source; files beside the clips
        for k in ["control_video", "control_mask", "control_source"] {
            let v = p.get(k).and_then(|v| v.as_str()).map(basename).filter(|v| !v.is_empty());
            if let Some(f) = &v {
                if !self.out.join(f).exists() {
                    return Err(RpcError::not_found(&format!("{k} {f} not found in out/")));
                }
            }
            it.insert(k.into(), v.map_or(Value::Null, |f| json!(f)));
        }
        if !it["control_video"].is_null() || !it["control_mask"].is_null() {
            if it["engine"] == "REF2VA" {
                return Err(RpcError::param("the ControlNet steers the text/image-to-video denoiser, not Ref2VA: references and control do not mix"));
            }
            it.insert("control_strength".into(), num(p, "control_strength", 1.0, 0.0, 10.0, false)?);
            it.insert("control_start".into(), num(p, "control_start", 0.0, 0.0, 1.0, false)?);
            it.insert("control_end".into(), num(p, "control_end", 1.0, 0.0, 1.0, false)?);
        }
        let ris = p.get("ref_image_size").and_then(|v| v.as_str()).filter(|v| *v == "match" || *v == "max");
        it.insert("ref_image_size".into(), ris.map_or(Value::Null, |v| json!(v)));
        Ok(Value::Object(it))
    }

    /// A request for LTX-2.5's engines -> a queue item, checked: `ltx2` makes a clip from a prompt (and a first frame),
    /// `ltx2_upscale` makes a finished clip twice the size. Anything H3 takes that they do not is refused by name.
    fn build_ltx_item(&self, p: &Value, engine: &str) -> Result<Value, RpcError> {
        let _ = self.engines(); // the daemon's list (a model added since the studio started)
        if !engine_known(engine) {
            return Err(RpcError::param(&format!("engine {engine}? (video.engines.list: {})", engine_names().join(", "))));
        }
        let given = |k: &str| p.get(k).is_some_and(|v| !v.is_null() && *v != json!("") && *v != json!(false) && *v != json!([]));
        let upscale = engine == LTX_UPSCALE;
        let mut it = Map::new();
        let mut not: Vec<&str> = LTX_NOT.to_vec();
        not.extend(["loras", "sampler", "schedule", "upscaler"]);
        if let Some(k) = not.iter().find(|k| given(k)) {
            return Err(RpcError::param(&format!("{k}: not a key {engine} takes (video.engines.list: its inputs)")));
        }
        if p.get("upscale").and_then(|v| v.as_f64()).is_some_and(|u| u > 1.0) {
            return Err(RpcError::param("upscale: not a key ltx2 takes - a finished clip is made twice the size by engine ltx2_upscale"));
        }
        if let Some(c) = p.get("chain").filter(|c| !c.is_null()) {
            return Err(RpcError::param(&format!("chain {c}: not a key {engine} takes - first_frame is the one way a clip carries on")));
        }
        let frames: u64;
        let (width, height);
        let mut variant = "dev";
        let mut steps_given = false;
        let prompt;
        if upscale {
            for k in ["prompt", "frames", "seconds", "steps", "seed", "width", "height", "first_frame", "variant", "negative_prompt"] {
                if given(k) {
                    return Err(RpcError::param(&format!("{k}: not a key ltx2_upscale takes - it takes source_clip (a finished clip) and makes it twice the size")));
                }
            }
            let src = p.get("source_clip").and_then(|v| v.as_str()).map(basename).filter(|v| !v.is_empty())
                .ok_or_else(|| RpcError::param("source_clip is required: the finished clip to upscale (a file in out/)"))?;
            let src = if src.contains('.') { src } else { format!("{src}.mp4") };
            if !src.ends_with(".mp4") || !self.out.join(&src).exists() {
                return Err(RpcError::not_found(&format!("source_clip {src} not found in out/ (an .mp4)")));
            }
            let side: Value = std::fs::read(self.out.join(src.trim_end_matches(".mp4")).with_extension("json")).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or(Value::Null);
            let (w, h) = (side["job"]["width"].as_i64().unwrap_or(0), side["job"]["height"].as_i64().unwrap_or(0));
            if w > 0 && (w % 32 != 0 || h % 32 != 0) {
                return Err(RpcError::param(&format!("source_clip {src} is {w}x{h}: the video VAE takes sides that are multiples of 32")));
            }
            frames = side["frames"].as_u64().or_else(|| side["job"]["frames"].as_u64()).unwrap_or(0);
            width = json!(w * 2);
            height = json!(h * 2);
            prompt = format!("upscale x2 of {src}");
            it.insert("source_clip".into(), json!(src));
            it.insert("seed".into(), json!(0));
            it.insert("steps".into(), json!(1));
            it.insert("first_frame".into(), Value::Null);
            it.insert("seconds".into(), json!(frames as f64 / 24.0));
            it.insert("frames".into(), json!(frames));
        } else {
            prompt = p.get("prompt").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
            if prompt.is_empty() {
                return Err(RpcError::param("prompt is required"));
            }
            variant = match p.get("variant").and_then(|v| v.as_str()).map(str::trim).filter(|v| !v.is_empty()) {
                None => "dev",
                Some(v) => *LTX_VARIANTS.iter().find(|x| **x == v).ok_or_else(|| RpcError::param(&format!("variant {v}? ({})", LTX_VARIANTS.join(", "))))?,
            };
            ltx_model(&ltx_ids(), variant)?;
            frames = match p.get("frames").filter(|v| !v.is_null()) {
                Some(f) => check_ltx_frames(f.as_u64().ok_or_else(|| RpcError::param("frames: a whole number"))?)?,
                None => {
                    let sec = num(p, "seconds", 5.0, 0.4, 15.1, false)?.as_f64().unwrap_or(5.0);
                    (super::pyround(sec * 24.0) as u64).max(LTX_FRAMES.0)
                }
            };
            let (w, h) = (num(p, "width", 640.0, 0.0, 1e6, true)?, num(p, "height", 384.0, 0.0, 1e6, true)?);
            check_ltx_canvas(w.as_i64().unwrap_or(0), h.as_i64().unwrap_or(0), frames)?;
            width = w;
            height = h;
            steps_given = given("steps");
            let dflt = if variant == "distilled" { LTX_STEPS.1 } else { LTX_STEPS.0 };
            let steps = num(p, "steps", dflt as f64, 1.0, 40.0, true)?;
            it.insert("seed".into(), num(p, "seed", 0.0, -2147483648.0, 2147483648.0, true)?);
            it.insert("seconds".into(), json!(frames as f64 / 24.0));
            it.insert("frames".into(), json!(frames));
            it.insert("steps".into(), steps.clone());
            it.insert("source_clip".into(), Value::Null);
            // a first frame is a picture in out/ (the platform video.uploads it); an anchor word is a clip's mp4, which
            // this engine does not read as a picture
            let ff = p.get("first_frame").and_then(|v| v.as_str()).map(basename).unwrap_or_default();
            let ff = if ff.is_empty() {
                Value::Null
            } else {
                if MAGIC.contains(&ff.as_str()) || ![".png", ".jpg", ".jpeg", ".webp"].iter().any(|e| ff.to_lowercase().ends_with(e)) {
                    return Err(RpcError::param(&format!("first_frame {ff}: ltx2 starts a clip on a picture (png, jpg, webp - video.upload it); not an anchor word or a clip")));
                }
                if !self.out.join(&ff).exists() {
                    return Err(RpcError::not_found(&format!("first_frame {ff} not found")));
                }
                json!(ff)
            };
            it.insert("first_frame".into(), ff);
        }
        it.insert("prompt".into(), json!(prompt));
        it.insert("negative".into(), json!(p.get("negative_prompt").and_then(|v| v.as_str()).unwrap_or("").trim()));
        it.insert("variant".into(), json!(variant));
        it.insert("steps_given".into(), json!(steps_given));
        it.insert("sampler".into(), Value::Null);
        it.insert("schedule".into(), Value::Null);
        it.insert("reentrant_id".into(), reentrant(p).map_or(Value::Null, |r| json!(r)));
        match p.get("clip").filter(|v| !v.is_null()) {
            Some(c) => match c.as_u64().filter(|c| (1..=100_000).contains(c)) {
                Some(c) => it.insert("clip".into(), json!(c)),
                None => return Err(RpcError::param("clip: a whole number from 1 (the clip's place in its project's timeline)")),
            },
            None => it.insert("clip".into(), Value::Null),
        };
        it.insert("te".into(), json!("teacher"));
        it.insert("label".into(), json!(clean_label(&p.get("label").map(|v| v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string())).unwrap_or_default())));
        it.insert("engine".into(), json!(engine));
        it.insert("priority".into(), num(p, "priority", 0.0, -9.0, 9.0, true)?);
        it.insert("width".into(), width);
        it.insert("height".into(), height);
        it.insert("chain_mode".into(), json!("none"));
        for k in ["camera", "last_frame", "exposure_ref", "first_audio", "first_audio_s", "upscale", "upscaler", "guide_clip", "cond_noise_aug", "shift_video",
                  "shift_audio", "source", "regen", "regen_box", "ref_images", "ref_videos", "ref_audios", "soundtrack", "soundtrack_start", "ref_image_size",
                  "ref_video_sound", "te_pictures"] {
            it.insert(k.into(), p.get(k).filter(|_| k == "camera").cloned().unwrap_or(Value::Null));
        }
        for k in ["project", "batch"] {
            let v = p.get(k).and_then(|v| v.as_str()).unwrap_or("");
            it.insert(k.into(), json!(v.chars().take(60).collect::<String>().trim()));
        }
        it.insert("loras".into(), json!([]));
        Ok(Value::Object(it))
    }
}

/// A request's reentrant_id (reentrantid is read too)
pub fn reentrant(p: &Value) -> Option<String> {
    ["reentrant_id", "reentrantid"].iter().find_map(|k| p.get(*k).and_then(|v| v.as_str())).map(str::trim).filter(|r| !r.is_empty()).map(|r| r.chars().take(200).collect())
}

pub fn basename(p: &str) -> String {
    Path::new(p).file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default()
}

impl Queue {
    /// Where an item of this priority goes: after every item of the same or a higher one
    pub fn slot(&self, priority: i64) -> usize {
        self.items.iter().position(|i| i["priority"].as_i64().unwrap_or(0) < priority).unwrap_or(self.items.len())
    }

    pub fn held(&self, item: &Value) -> bool {
        let p = project_of(item);
        let b = s(item, "batch");
        self.paused || (!p.is_empty() && self.paused_projects.contains(&p)) || (!b.is_empty() && self.paused_batches.contains(&b))
    }

    pub fn load(dir: &Path) -> Queue {
        let items = std::fs::read(dir.join("queue.json")).ok().and_then(|b| serde_json::from_slice::<Vec<Value>>(&b).ok()).unwrap_or_default();
        let p: Value = std::fs::read(dir.join("paused.json")).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or(Value::Null);
        let set = |k: &str| -> BTreeSet<String> { p.get(k).and_then(|v| v.as_array()).map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()).unwrap_or_default() };
        Queue { items, paused: p.get("all").and_then(|v| v.as_bool()).unwrap_or(false), paused_projects: set("projects"), paused_batches: set("batches") }
    }

    pub fn save(&self, dir: &Path) {
        let _ = super::write_atomic(&dir.join("queue.json"), &serde_json::to_vec(&self.items).unwrap_or_default());
    }

    pub fn save_paused(&self, dir: &Path) {
        let v = json!({"all": self.paused, "projects": self.paused_projects, "batches": self.paused_batches});
        let _ = super::write_atomic(&dir.join("paused.json"), &serde_json::to_vec(&v).unwrap_or_default());
    }

    /// The queued projects in queue order, with their batches.
    pub fn projects(&self) -> Vec<Value> {
        let mut order: Vec<String> = Vec::new();
        for it in &self.items {
            let p = project_of(it);
            if !order.contains(&p) {
                order.push(p);
            }
        }
        order
            .iter()
            .map(|p| {
                let mine: Vec<&Value> = self.items.iter().filter(|i| project_of(i) == *p).collect();
                let first = self.items.iter().position(|i| project_of(i) == *p).unwrap_or(0);
                let mut batches: Vec<String> = mine.iter().map(|i| s(i, "batch")).collect();
                batches.sort();
                batches.dedup();
                let secs = |v: &[&Value]| (v.iter().map(|i| i["seconds"].as_f64().unwrap_or(0.0)).sum::<f64>() * 100.0).round() / 100.0;
                json!({"project": p, "clips": mine.len(), "video_s": secs(&mine), "paused": self.paused_projects.contains(p), "first_index": first,
                       "batches": batches.iter().map(|b| {
                           let bi: Vec<&Value> = mine.iter().copied().filter(|i| s(i, "batch") == *b).collect();
                           json!({"batch": b, "clips": bi.len(), "video_s": secs(&bi), "paused": self.paused_batches.contains(b)})
                       }).collect::<Vec<_>>()})
            })
            .collect()
    }

    pub fn payload(&self) -> Map<String, Value> {
        let mut m = Map::new();
        m.insert("queue".into(), json!(self.items.iter().map(|i| { let l = s(i, "label"); if l.is_empty() { s(i, "prompt").chars().take(40).collect() } else { l } }).collect::<Vec<_>>()));
        m.insert("queue_items".into(), json!(self.items.iter().map(|i| json!({
            "label": s(i, "label"), "seconds": i["seconds"], "steps": i["steps"], "engine": i["engine"], "chain": !i["first_frame"].is_null(),
            "size": format!("{}x{}", i["width"].as_i64().unwrap_or(640), i["height"].as_i64().unwrap_or(480)),
            "project": project_of(i), "batch": s(i, "batch"), "held": self.held(i)})).collect::<Vec<_>>()));
        m.insert("paused".into(), json!(self.paused));
        m.insert("paused_projects".into(), json!(self.paused_projects));
        m.insert("paused_batches".into(), json!(self.paused_batches));
        m.insert("projects".into(), json!(self.projects()));
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels() {
        assert_eq!(label_prefix("Harbour Night 03/12: the lamp"), "Harbour Night");
        assert_eq!(label_number("Harbour Night 03/12: the lamp"), Some(3));
        assert_eq!(label_number("no number"), None);
        // the order of use: clip, the label's NN/MM, the reentrant_id's #N (all digits after the last #), 0
        let c = |label: &str, id: Value| json!({"label": label, "reentrant_id": id});
        assert_eq!(clip_number(&c("a2c6221e-bde2:shots:0:016", json!("a2c6221e-bde2:shots:0:0#16"))), 16);
        assert_eq!(clip_number(&c("Name 03/12: x", json!("r#9"))), 3);
        assert_eq!(clip_number(&c("cp-smoke:shots:0:0", Value::Null)), 0);
        assert_eq!(clip_number(&c("x", json!("r#x1"))), 0);
        assert_eq!(clip_number(&json!({"clip": 5, "label": "Name 03/12: x", "reentrant_id": "r#9"})), 5);
        assert_eq!(clip_number(&json!({"clip": null, "label": "Name 03/12: x"})), 3);
        assert_eq!(project_of(&json!({"label": "Speech 07/20: words"})), "Speech");
        assert_eq!(project_of(&json!({"label": "x", "project": "P"})), "P");
        assert_eq!(dialogue_of("says: <d>[English] First   batch.</d> then <d>[English] More.</d>"), "First batch. / More.");
        assert_eq!(clean_label("a<b>c 01/02: ok"), "abc 01/02: ok");
    }

    #[test]
    fn ltx_models_are_the_engine_ltx2() {
        assert_eq!(quant_of("ltx-2.5"), "ltx2");
        assert_eq!(quant_of("ltx-2.5-fast"), "ltx2");
        assert_eq!(quant_of("minimax-h3-q4km"), "Q4_K_M");
        // the engine names a client may send, in any case, and the registry's own
        for e in ["ltx2", "LTX2", " ltx-2.5 ", "ltx-2.5-fast"] {
            assert_eq!(engine_name(Some(e)), "ltx2", "{e}");
        }
        assert_eq!(engine_name(Some("ltx2_upscale")), "ltx2_upscale");
        assert_eq!(engine_name(Some("q4_k_m")), "Q4_K_M");
        assert!(is_ltx("ltx2") && is_ltx("ltx2_upscale") && !is_ltx("INT8"));
    }

    #[test]
    fn ltx_frames_canvas_and_tokens() {
        assert_eq!(ltx_latent_frames(9), 2);
        assert_eq!(ltx_latent_frames(121), 16);
        assert_eq!(ltx_latent_frames(120), 16, "120 frames are made as 121");
        assert_eq!(ltx_latent_frames(122), 17);
        assert_eq!(ltx_latent_frames(1), 2, "at least nine frames");
        assert_eq!(ltx_tokens(640, 384, 121), 3840, "the tested shape");
        assert_eq!(check_ltx_frames(9).unwrap(), 9);
        assert_eq!(check_ltx_frames(120).unwrap(), 120, "not on the 8k + 1 grid: allowed, the engine cuts back");
        assert!(check_ltx_frames(8).is_err() && check_ltx_frames(362).is_err());
        assert!(check_ltx_canvas(640, 384, 121).is_ok());
        assert!(check_ltx_canvas(650, 384, 121).unwrap_err().message.contains("multiples of 32"));
        assert!(check_ltx_canvas(224, 384, 121).unwrap_err().message.contains("between 256 and 1344"));
        assert!(check_ltx_canvas(1344, 768, 121).unwrap_err().message.contains("tokens"));
        // the limit is tokens, not area: a smaller canvas for a longer clip
        assert!(check_ltx_canvas(640, 384, 361).is_err() && check_ltx_canvas(480, 320, 241).is_ok() && check_ltx_canvas(1024, 576, 121).is_ok() && check_ltx_canvas(1024, 608, 121).is_err());
    }

    #[test]
    fn ltx_variants_pick_models() {
        let both = vec!["ltx-2.5-fast".to_string(), "ltx-2.5".to_string()];
        assert_eq!(ltx_model(&both, "dev").unwrap(), "ltx-2.5");
        assert_eq!(ltx_model(&both, "distilled").unwrap(), "ltx-2.5", "the dev model makes the distilled variant too");
        let fast = vec!["ltx-2.5-fast".to_string()];
        assert!(ltx_model(&fast, "dev").unwrap_err().message.contains("only the distilled"));
        assert_eq!(ltx_model(&fast, "distilled").unwrap(), "ltx-2.5-fast");
        assert!(ltx_model(&[], "dev").is_err());
    }

    #[test]
    fn engines_list_has_ltx2_and_its_upscaler() {
        let models = vec![("ltx-2.5".to_string(), "/m/ltx-2.5/dev.gguf".to_string(), 14.61), ("ltx-2.5-fast".to_string(), "/m/ltx-2.5/dist.gguf".to_string(), 13.2)];
        let lic = Some(json!({"license": "ltx-2-community", "commercial": true, "license_note": "x"}));
        let rows = ltx_rows(&models, &lic);
        assert_eq!(rows.len(), 2);
        let (shots, up) = (&rows[0], &rows[1]);
        assert_eq!((shots["quant"].as_str(), shots["engine"].as_str(), shots["stage"].as_str()), (Some("ltx2"), Some("ltx2"), Some("shots")));
        assert_eq!((up["quant"].as_str(), up["engine"].as_str(), up["stage"].as_str()), (Some("ltx2_upscale"), Some("ltx2_upscale"), Some("upscale")));
        for r in &rows {
            assert_eq!((r["ready"].clone(), r["license"].clone(), r["commercial"].clone()), (json!(true), json!("ltx-2-community"), json!(true)));
            assert!(r["inputs"]["takes"].is_array() && r["inputs"]["not"].is_array() && r["gib"].as_f64().unwrap() > 0.0);
        }
        assert_eq!(shots["file"], "dev.gguf");
        let o = &shots["inputs"]["options"];
        assert_eq!(o["variant"]["values"], json!(["dev", "distilled"]));
        assert_eq!(o["steps"]["default"], json!({"dev": 30, "distilled": 8}));
        assert_eq!((o["frames"]["min"].clone(), o["frames"]["max"].clone()), (json!(9), json!(361)));
        assert_eq!(o["width"]["multiple_of"], 32);
        assert_eq!(up["inputs"]["options"]["scale"]["values"], json!([2]));
        // the first frame is a key it takes; a sound-pinning key is not (the platform reads takes for that)
        let takes: Vec<&str> = shots["inputs"]["takes"].as_array().unwrap().iter().filter_map(|t| t.as_str()).collect();
        assert!(takes.contains(&"first_frame") && takes.contains(&"negative_prompt") && !takes.contains(&"soundtrack") && !takes.contains(&"first_audio"));
        // a box with only the distilled model offers only that variant; none without a model
        let only = ltx_rows(&models[1..], &None);
        assert_eq!(only[0]["inputs"]["options"]["variant"]["values"], json!(["distilled"]));
        assert!(ltx_rows(&[], &None).is_empty());
    }

    /// A studio over a scratch directory (no engine daemon: its socket does not exist)
    fn studio(tag: &str) -> Studio {
        let dir = std::env::temp_dir().join(format!("ncd-studio-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("out")).unwrap();
        let llm: &'static super::super::llm::Llm = Box::leak(Box::new(super::super::llm::Llm::load(None, &dir, dir.clone())));
        let _ = ENGINES.set(vec![("ltx2".into(), "ltx-2.5".into()), ("ltx2".into(), "ltx-2.5-fast".into())]);
        Studio { out: dir.join("out"), out_in: "/out".into(), dir: dir.clone(), socket: dir.join("none.sock"), ui: dir.clone(), templates: None,
                 gpustat: dir.join("gpu"), stall_s: 60, st: Default::default(), cache: Default::default(), llm, joins: Default::default() }
    }

    #[test]
    fn a_ltx_shot_is_checked_and_becomes_a_job() {
        let s = studio("shot");
        std::fs::write(s.out.join("ref_1_face.png"), b"png").unwrap();
        let ok = s.build_item(&json!({"engine": "ltx2", "prompt": "A fox.", "frames": 121, "width": 640, "height": 384, "seed": 7, "label": "L 01/02: fox",
                                      "reentrant_id": "r#1", "first_frame": "ref_1_face.png", "negative_prompt": "blurry"})).unwrap();
        assert_eq!((ok["engine"].as_str(), ok["variant"].as_str(), ok["steps"].as_i64(), ok["frames"].as_u64()), (Some("ltx2"), Some("dev"), Some(30), Some(121)));
        assert_eq!((ok["chain_mode"].as_str(), ok["first_frame"].as_str(), ok["negative"].as_str()), (Some("none"), Some("ref_1_face.png"), Some("blurry")));
        assert!((ok["seconds"].as_f64().unwrap() - 121.0 / 24.0).abs() < 1e-9);
        let (spec, _) = s.spec(&crate::video::studio::State::default(), &ok, "h3_20260101_000000");
        assert_eq!((spec["kind"].as_str(), spec["engine"].as_str(), spec["fast"].clone(), spec["steps"].clone()), (Some("generate"), Some("ltx-2.5"), json!(false), json!(30)));
        assert_eq!((spec["first_frame"].as_str(), spec["out"].as_str(), spec["frames"].clone()), (Some("/out/ref_1_face.png"), Some("/out/h3_20260101_000000.mp4"), json!(121)));
        // seconds are frames at 24 fps, as asked (5 s = 120), not snapped to H3's grid
        let by_s = s.build_item(&json!({"engine": "ltx2", "prompt": "x", "seconds": 5})).unwrap();
        assert_eq!((by_s["frames"].as_u64(), by_s["width"].as_i64(), by_s["height"].as_i64()), (Some(120), Some(640), Some(384)));
        // the distilled variant: 8 steps and its own sigmas unless steps are named
        let d = s.build_item(&json!({"engine": "ltx2", "prompt": "x", "variant": "distilled"})).unwrap();
        let (dspec, _) = s.spec(&crate::video::studio::State::default(), &d, "n");
        assert_eq!((d["steps"].as_i64(), dspec["fast"].clone(), dspec.get("steps").is_none()), (Some(8), json!(true), true));
        let d = s.build_item(&json!({"engine": "ltx2", "prompt": "x", "variant": "distilled", "steps": 12})).unwrap();
        assert_eq!(s.spec(&crate::video::studio::State::default(), &d, "n").0["steps"], json!(12));
        // refusals name the key
        let err = |v: Value| s.build_item(&v).unwrap_err().message;
        assert!(err(json!({"engine": "ltx2", "prompt": "x", "frames": 8})).contains("frames 8"));
        assert!(err(json!({"engine": "ltx2", "prompt": "x", "width": 650})).contains("multiples of 32"));
        assert!(err(json!({"engine": "ltx2", "prompt": "x", "width": 1280, "height": 704})).contains("tokens"));
        assert!(err(json!({"engine": "ltx2", "prompt": "x", "variant": "turbo"})).contains("variant turbo?"));
        assert!(err(json!({"engine": "ltx2", "prompt": "x", "guide_clip": "prev:22:0"})).starts_with("guide_clip:"));
        assert!(err(json!({"engine": "ltx2", "prompt": "x", "ref_images": ["a.png"]})).starts_with("ref_images:"));
        assert!(err(json!({"engine": "ltx2", "prompt": "x", "upscale": 1.5})).contains("ltx2_upscale"));
        assert!(err(json!({"engine": "ltx2", "prompt": "x", "first_frame": "prev"})).contains("not an anchor word"));
        assert!(err(json!({"engine": "ltx2", "prompt": "x", "first_frame": "h3_1.mp4"})).contains("picture"));
        assert!(err(json!({"engine": "ltx2", "prompt": "  "})).contains("prompt is required"));
        assert_eq!(s.build_item(&json!({"engine": "ltx2", "prompt": "x", "first_frame": "nope.png"})).unwrap_err().code, "not_found");
        let _ = std::fs::remove_dir_all(&s.dir);
    }

    #[test]
    fn a_ltx_upscale_takes_a_finished_clip() {
        let s = studio("up");
        std::fs::write(s.out.join("h3_20260101_000001.mp4"), b"mp4").unwrap();
        std::fs::write(s.out.join("h3_20260101_000001.json"), br#"{"job": {"width": 640, "height": 384, "frames": 121}, "frames": 121}"#).unwrap();
        let up = s.build_item(&json!({"engine": "ltx2_upscale", "prompt": "", "source_clip": "h3_20260101_000001", "label": "U 01/01", "reentrant_id": "u#1"})).unwrap();
        assert_eq!((up["engine"].as_str(), up["source_clip"].as_str(), up["width"].as_i64(), up["height"].as_i64(), up["frames"].as_u64()),
                   (Some("ltx2_upscale"), Some("h3_20260101_000001.mp4"), Some(1280), Some(768), Some(121)));
        let (spec, _) = s.spec(&crate::video::studio::State::default(), &up, "h3_20260101_000002");
        assert_eq!((spec["kind"].as_str(), spec["source"].as_str(), spec["out"].as_str(), spec["scale"].clone()),
                   (Some("upscale"), Some("/out/h3_20260101_000001.mp4"), Some("/out/h3_20260101_000002.mp4"), json!(2)));
        let err = |v: Value| s.build_item(&v).unwrap_err();
        assert!(err(json!({"engine": "ltx2_upscale", "prompt": "a fox", "source_clip": "h3_20260101_000001"})).message.contains("prompt: not a key"));
        assert!(err(json!({"engine": "ltx2_upscale", "prompt": ""})).message.contains("source_clip is required"));
        assert_eq!(err(json!({"engine": "ltx2_upscale", "prompt": "", "source_clip": "gone.mp4"})).code, "not_found");
        std::fs::write(s.out.join("h3_x.mp4"), b"mp4").unwrap();
        std::fs::write(s.out.join("h3_x.json"), br#"{"job": {"width": 650, "height": 384}}"#).unwrap();
        assert!(err(json!({"engine": "ltx2_upscale", "prompt": "", "source_clip": "h3_x"})).message.contains("multiples of 32"));
        let _ = std::fs::remove_dir_all(&s.dir);
    }

    #[test]
    fn ltx_progress_is_read_from_the_engine_log() {
        use super::super::runner::ltx_stage;
        let l = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let none = json!(null);
        assert_eq!(ltx_stage(false, &[], &none, 30).1, "starting");
        let log = l(&["ltx2: 640 x 384, 121 frames (made as 121), 30 steps, seed 0"]);
        assert_eq!(ltx_stage(false, &log, &none, 30).1, "text encoder");
        let log = l(&["ltx2: 640 x 384, 121 frames (made as 121), 30 steps, seed 0", "ltx2: prompt (74 tokens) encoded in 40.1 s"]);
        assert_eq!(ltx_stage(false, &log, &none, 30), (10.0, "conditioning done".into(), 0));
        let (pct, stage, done) = ltx_stage(false, &log, &json!({"done": 15, "total": 30}), 30);
        assert_eq!((pct, stage.as_str(), done), (46.0, "sampling 16/30", 15));
        let mut log = log;
        log.push("ltx2: 30 steps in 646.0 s (21.55 s a step); cards in use [7.4, 7.5] GiB".into());
        assert_eq!(ltx_stage(false, &log, &json!({"done": 30, "total": 30}), 30), (80.0, "sampling done, decoding".into(), 30));
        log.push("ltx2: decoded in 20.1 s; /out/x.mp4 in 700.0 s".into());
        assert_eq!(ltx_stage(false, &log, &none, 30).0, 95.0);
        let (pct, stage, _) = ltx_stage(true, &[], &json!({"done": 60, "total": 120}), 1);
        assert_eq!((pct, stage.as_str()), (50.0, "upscaling 60/120 frames"));
        assert_eq!(ltx_stage(true, &l(&["ltx2: 120 frames 640x384 -> 1280x768 in 75 s"]), &none, 1).1, "muxing");
    }

    #[test]
    fn frames_and_chains() {
        assert_eq!(check_frames(107).unwrap(), 107);
        let e = check_frames(97).unwrap_err().message;
        assert!(e.contains("90 or 107"), "{e}");
        assert!(check_frames(400).unwrap_err().message.contains("nearest: 362"));
        assert!(check_frames(10).unwrap_err().message.contains("nearest: 22"));
        let (p, ig) = unchain(&json!({"prompt": "x", "chain": {"from": "prev", "context_frames": 22, "color_reanchor": "first", "pin_renorm": true}})).unwrap();
        assert_eq!(p["guide_clip"], "prev:22:0");
        assert_eq!(p["first_audio"], "prev");
        assert!((p["first_audio_s"].as_f64().unwrap() - 22.0 / 24.0).abs() < 1e-9);
        assert!(p["first_frame"].is_null() && p.get("chain").is_none());
        assert_eq!(p["exposure_ref"], "first");
        assert_eq!(ig, vec!["chain.pin_renorm".to_string()]);
        let (p, _) = unchain(&json!({"chain": {"mode": "png"}, "first_frame": "a.png"})).unwrap();
        assert_eq!(p["first_frame"], "a.png"); // the flat key wins
        assert_eq!(p["chain_mode"], "png");
        assert!(unchain(&json!({"chain": {"context_frames": 30}})).is_err());
        // a clip made to a soundtrack carries the picture alone: its sound is the song's own stretch
        let (p, _) = unchain(&json!({"soundtrack": "song.flac", "soundtrack_start": 9.21, "chain": {"from": "prev", "context_frames": 22}})).unwrap();
        assert_eq!(p["guide_clip"], "prev:22:0");
        assert!(p.get("first_audio").is_none() && p.get("first_audio_s").is_none());
        assert!(inputs_of("Q4_K_M")["takes"].as_array().unwrap().contains(&json!("soundtrack")));
    }

    #[test]
    fn priority_places() {
        let q = Queue { items: vec![json!({"priority": 2}), json!({"priority": 0}), json!({}), json!({"priority": -1})], ..Queue::default() };
        assert_eq!(q.slot(0), 3); // after its equals, ahead of lower ones
        assert_eq!(q.slot(1), 1);
        assert_eq!(q.slot(5), 0);
        assert_eq!(q.slot(-5), 4);
    }
}
