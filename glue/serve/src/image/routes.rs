//! The image server's routes (rpcgen `#[route]`): what it serves, what wfe/image and wfe/ocr call
//! (`src/gen/routes.ts`) and what docs/api/image.openapi.yml lists, all from this table. The OpenAI-compatible
//! ones take the bodies the front door's methods relay (home/methods.rs: image.generate's body is
//! ImageGenerateBody, the job's options aside); the answers are typed here and each is held to its type as it is
//! sent (`Checked`).

use std::sync::atomic::Ordering;

use rpcgen::{route, routes, Checked, Rpc, Sent};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::{edit_body, err, init_picture, ImageServer};
use crate::home::methods::{ImageEditBody, ImageGenerateBody, ImageOcrReq, ImageUpscaleBody, ImageVectorizeBody, VideoUpscaleBody};
use crate::page::{CancelReq, Cancelled, Cx, Fail, FileName, Gpu, Health, ModelList};

/// The engine's own defaults (a request's absent fields)
#[derive(Serialize, Deserialize, Rpc)]
pub struct Defaults {
    pub width: i64,
    pub height: i64,
    pub steps: i64,
    pub cfg: f64,
    pub sampler: String,
    pub schedule: String,
    pub shift: f64,
}

/// An instruction an OCR model is trained on
#[derive(Serialize, Deserialize, Rpc)]
pub struct OcrPromptInfo {
    pub id: String,
    pub prompt: String,
    /// what it reads
    pub title: String,
}

/// An option of the engine's own (a request sends it in `options`)
#[derive(Serialize, Deserialize, Rpc)]
pub struct EngineOptionInfo {
    pub name: String,
    /// what its value looks like ("" a flag)
    pub value: String,
    pub help: String,
}

/// A LoRA registered for the model
#[derive(Serialize, Deserialize, Rpc)]
pub struct LoraInfo {
    pub id: String,
    pub title: String,
    /// the scale it is merged at now; null: not loaded
    pub loaded: Option<f64>,
}

/// A variant an upscaler offers
#[derive(Serialize, Deserialize, Rpc)]
pub struct UpscaleInfo {
    pub variant: String,
    pub about: String,
    pub scales: Vec<i64>,
    /// the denoise levels it takes (empty: none)
    pub noise: Vec<i64>,
}

/// The model and what it takes; while it loads, the model and `loading` only
#[derive(Serialize, Deserialize, Rpc)]
pub struct Info {
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loading: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arch: Option<String>,
    /// it takes pictures (edits)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edits: Option<bool>,
    /// it reads pictures (an OCR model: the OCR page)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reads: Option<bool>,
    /// it takes guidance: cfg above 1 and a negative prompt
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub guidance: Option<bool>,
    /// the most pictures an edit takes
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_pictures: Option<i64>,
    /// the most pages a reading takes
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_pages: Option<i64>,
    /// an OCR model's instructions
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompts: Option<Vec<OcrPromptInfo>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub defaults: Option<Defaults>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub samplers: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schedules: Option<Vec<String>>,
    /// the engine's own per-request options (sent as `options`)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub options: Option<Vec<EngineOptionInfo>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loras: Option<Vec<LoraInfo>>,
    /// it keeps what it makes (an output directory: links, the history)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub saves: Option<bool>,
    /// it serves its page
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wfe: Option<bool>,
    /// it writes pictures as SVG (a vectorizer)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vectorize: Option<bool>,
    /// an upscaler's variants (empty: it draws)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upscale: Option<Vec<UpscaleInfo>>,
}

/// What the server is doing: busy false, or what it is making and how far it is
#[derive(Serialize, Deserialize, Rpc)]
pub struct Progress {
    pub busy: bool,
    /// LoRAs being merged (the engine reloaded)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loading: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    /// the picture being made (1-based) of `pictures`
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub picture: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pictures: Option<i64>,
    /// the step (or tile, or token) done of `of`
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub at: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub of: Option<i64>,
    /// what is counted: tiles (an upscale), frames (a clip's), tokens (a vectorize)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,
    /// a reading (OCR): its pages, the text so far and its tokens
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reading: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pages: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens: Option<i64>,
    /// seconds since it started
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seconds: Option<f64>,
    /// requests waiting for the engine
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub waiting: Option<i64>,
    /// the front door's job running, if it named one
    #[serde(default)]
    pub request_id: Option<String>,
}

/// What a history entry was: a picture made (none), or another kind of work
#[derive(Serialize, Deserialize, Rpc)]
pub enum HistoryKind {
    #[serde(rename = "ocr")]
    Ocr,
    #[serde(rename = "vectorize")]
    Vectorize,
    #[serde(rename = "upscale")]
    Upscale,
    #[serde(rename = "upscale-video")]
    UpscaleVideo,
}

/// Something the server made, newest first: a picture (with its settings), or a reading, an SVG, an upscale
#[derive(Serialize, Deserialize, Rpc)]
pub struct HistoryEntry {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<HistoryKind>,
    /// the file (GET it here): a picture, an upscale, a clip
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// the prompt, or what the work was ("realesrgan-x4plus 4x")
    pub prompt: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steps: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loras: Option<Vec<String>>,
    /// the pictures it was made from (an edit; 0: none)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pictures: Option<i64>,
    /// a reading's pages, text and tokens
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pages: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tok_s: Option<f64>,
    pub seconds: f64,
    /// the card's energy, watt-hours
    #[serde(default)]
    pub wh: Option<f64>,
    /// unix seconds
    pub created: i64,
}

/// What the server made, newest first (files deleted since drop out)
#[derive(Serialize, Deserialize, Rpc)]
pub struct History {
    pub data: Vec<HistoryEntry>,
}

/// A picture made: its link (response_format url) or its PNG (b64_json)
#[derive(Serialize, Deserialize, Rpc)]
pub struct PictureMade {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub b64_json: Option<String>,
    pub revised_prompt: String,
    pub seed: i64,
    pub width: i64,
    pub height: i64,
}

/// How pictures were made
#[derive(Serialize, Deserialize, Rpc)]
pub struct PicturesRun {
    pub seconds: f64,
    pub steps: i64,
    /// the first picture's; the n-th is seed + n
    pub seed: i64,
    /// name:scale, as merged
    pub loras: Vec<String>,
    /// the card's energy, watt-hours
    pub wh: Option<f64>,
}

/// OpenAI's images answer, with how they were made (`nextsycl`: the key nextsycl's contract names it; the front
/// door's methods call it run)
#[derive(Serialize, Deserialize, Rpc)]
pub struct Pictures {
    pub created: i64,
    pub data: Vec<PictureMade>,
    pub nextsycl: PicturesRun,
}

/// An SVG written
#[derive(Serialize, Deserialize, Rpc)]
pub struct Svg {
    /// the SVG as the model wrote it (from <svg)
    pub svg: String,
    /// </svg> ended it (false: max_length stopped it, an element maybe half written)
    pub closed: bool,
    pub tokens: i64,
}

/// How an SVG was written
#[derive(Serialize, Deserialize, Rpc)]
pub struct VectorizeRun {
    pub seconds: f64,
    pub prompt_seconds: f64,
    pub tokens: i64,
    pub tok_s: f64,
    pub wh: Option<f64>,
}

/// A picture written as SVG
#[derive(Serialize, Deserialize, Rpc)]
pub struct Vectorized {
    pub created: i64,
    pub data: Vec<Svg>,
    pub nextsycl: VectorizeRun,
}

/// A picture or a clip upscaled: its link or its bytes (b64_json)
#[derive(Serialize, Deserialize, Rpc)]
pub struct UpscaledItem {
    pub width: i64,
    pub height: i64,
    pub variant: String,
    /// the factor made (waifu2x 4x and up: its 2x model again each time)
    pub scale: i64,
    pub noise: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub b64_json: Option<String>,
    /// a clip: its frames, frame rate (as ffprobe gives it, 24/1) and file name
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frames: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fps: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// How an upscale went
#[derive(Serialize, Deserialize, Rpc)]
pub struct UpscaleRun {
    pub seconds: f64,
    /// a picture: its tiles and their side
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tiles: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tile: Option<i64>,
    /// a clip: its frames and frames a second made
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frames: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fps_done: Option<f64>,
    pub wh: Option<f64>,
}

/// A picture or a clip upscaled
#[derive(Serialize, Deserialize, Rpc)]
pub struct Upscaled {
    pub created: i64,
    pub data: Vec<UpscaledItem>,
    pub nextsycl: UpscaleRun,
}

/// A reading (OCR): the text and how it went
#[derive(Serialize, Deserialize, Rpc)]
pub struct Reading {
    pub created: i64,
    pub model: String,
    /// as the model wrote it (its layout markup kept; several pages: a <PAGE> mark before each)
    pub text: String,
    pub prompt: String,
    pub pages: i64,
    pub tokens: i64,
    pub prompt_tokens: i64,
    pub prompt_seconds: f64,
    pub read_tok_s: f64,
    pub write_seconds: f64,
    pub tok_s: f64,
    pub seconds: f64,
    pub wh: Option<f64>,
}

/// The card: its name, memory, power and temperatures
#[route(get = "/api/gpu", tags = "page")]
fn gpu(s: &ImageServer, _cx: &mut Cx) -> Result<Gpu, Fail> {
    Ok(s.gpu_json())
}

/// Whether the server answers: ok, or loading (HTTP 503) while its engine loads
#[route(get = "/health", tags = "page")]
fn health(s: &ImageServer, cx: &mut Cx) -> Result<Sent<Health>, Fail> {
    let loaded = s.engine.try_lock().map(|e| e.is_some()).unwrap_or(true);
    cx.json(if loaded { 200 } else { 503 }, &json!(Health { status: if loaded { "ok" } else { "loading" }.into() }));
    Ok(Sent::new())
}

/// The model served, as OpenAI lists models
#[route(get = "/v1/models", alias = "/models", tags = "openai")]
fn models(s: &ImageServer, _cx: &mut Cx) -> Result<ModelList, Fail> {
    Ok(ModelList::of(&s.model))
}

/// The model and what it takes: its defaults, samplers, LoRAs, options; an OCR model's instructions, an upscaler's variants
#[route(get = "/api/info", tags = "page")]
fn info(s: &ImageServer, _cx: &mut Cx) -> Result<Checked<Info>, Fail> {
    Ok(Checked::new(s.info()))
}

/// What the server is doing now (the page polls it)
#[route(get = "/api/progress", tags = "page")]
fn progress(s: &ImageServer, _cx: &mut Cx) -> Result<Checked<Progress>, Fail> {
    let mut p = s.progress.lock().unwrap().clone();
    p["waiting"] = json!(s.waiting.load(Ordering::Relaxed));
    p["request_id"] = json!(s.cancel.current());
    Ok(Checked::new(p))
}

/// What the server made, newest first
#[route(get = "/api/history", tags = "page")]
fn history(s: &ImageServer, _cx: &mut Cx) -> Result<Checked<History>, Fail> {
    // files deleted since (system.files.delete) drop out
    let mut h = s.history.lock().unwrap();
    if let Some(d) = &s.out_dir {
        h.retain(|e| e["url"].as_str().and_then(|u| u.rsplit('/').next()).is_none_or(|f| d.join(f).exists()));
    }
    Ok(Checked::new(json!({"data": h.iter().collect::<Vec<_>>()})))
}

/// Stops a request: the one named (running or waiting), else the one running
#[route(post = "/api/cancel", tags = "page")]
fn cancel(s: &ImageServer, _cx: &mut Cx, req: CancelReq) -> Result<Cancelled, Fail> {
    let busy = s.progress.lock().unwrap()["busy"] == true;
    let running = s.cancel.request(req.id.as_deref(), busy);
    Ok(Cancelled { cancelled: running, id: req.id })
}

/// A picture (or an upscaled clip) an answer links
#[route(get = "/v1/images/files/{name}", tags = "files", produces = "image/png, video/mp4")]
fn file(s: &ImageServer, cx: &mut Cx, req: FileName) -> Result<Sent, Fail> {
    s.file(cx, &req.name);
    Ok(Sent::new())
}

/// Pictures from a prompt (OpenAI's images API, and ours: steps, seed, sampler, schedule, cfg, negative_prompt, loras); an init picture (image) makes it an img2img
#[route(post = "/v1/images/generations", alias = "/images/generations", tags = "openai")]
fn generations(s: &ImageServer, cx: &mut Cx, req: ImageGenerateBody) -> Result<Checked<Pictures>, Fail> {
    let mut body = serde_json::to_value(&req).map_err(|e| (500, e.to_string()))?;
    // an init picture: a plain img2img (the denoise starts from it; it is no edit's reference)
    let mut pics = Vec::new();
    if let Some(i) = ["image", "init_image"].iter().find_map(|k| body[*k].as_str().filter(|v| !v.is_empty())).map(str::to_string) {
        pics.push(init_picture(&i).map_err(|m| (400, m))?);
        body["reference"] = json!(false);
    }
    s.generate(&body, &cx.base(), pics).map(Checked::new)
}

/// An edit: pictures (up to 8) and the instructions, as OpenAI's multipart form (image, image[] files) or JSON with images (base64 / data URLs)
#[route(post = "/v1/images/edits", alias = "/images/edits", tags = "openai", body = "json, multipart", form = "ImageEditBody")]
fn edits(s: &ImageServer, cx: &mut Cx) -> Result<Checked<Pictures>, Fail> {
    let (body, pics) = edit_body(&cx.req).map_err(|m| (400, m))?;
    s.generate(&body, &cx.base(), pics).map(Checked::new)
}

/// A picture written as SVG by a vectorizer (StarVector)
#[route(post = "/v1/images/vectorize", tags = "openai")]
fn vectorize(s: &ImageServer, _cx: &mut Cx, req: ImageVectorizeBody) -> Result<Checked<Vectorized>, Fail> {
    s.vectorize(&serde_json::to_value(&req).map_err(|e| (500, e.to_string()))?).map(Checked::new)
}

/// A picture upscaled (Real-ESRGAN, waifu2x): a link, or the PNG as base64
#[route(post = "/v1/images/upscale", tags = "openai")]
fn upscale(s: &ImageServer, cx: &mut Cx, req: ImageUpscaleBody) -> Result<Checked<Upscaled>, Fail> {
    s.upscale(&serde_json::to_value(&req).map_err(|e| (500, e.to_string()))?, &cx.base()).map(Checked::new)
}

/// A clip upscaled frame by frame (ffmpeg decodes and encodes, the audio copied): the MP4 linked, or as base64
#[route(post = "/v1/videos/upscale", tags = "openai")]
fn upscale_video(s: &ImageServer, cx: &mut Cx, req: VideoUpscaleBody) -> Result<Checked<Upscaled>, Fail> {
    s.upscale_video(&serde_json::to_value(&req).map_err(|e| (500, e.to_string()))?, &cx.base()).map(Checked::new)
}

/// Pages read (an OCR model): pictures or PDFs, the text as the model writes it
#[route(post = "/v1/ocr", tags = "openai")]
fn ocr(s: &ImageServer, _cx: &mut Cx, req: ImageOcrReq) -> Result<Checked<Reading>, Fail> {
    s.ocr(&serde_json::to_value(&req).map_err(|e| (500, e.to_string()))?).map(Checked::new)
}

routes![sync ImageServer, Cx, Fail; gpu, health, models, info, progress, history, cancel, file, generations, edits, vectorize, upscale, upscale_video, ocr];

/// The table, made once
pub fn table() -> &'static [rpcgen::SyncRoute<ImageServer, Cx, Fail>] {
    static T: std::sync::OnceLock<Vec<rpcgen::SyncRoute<ImageServer, Cx, Fail>>> = std::sync::OnceLock::new();
    T.get_or_init(routes)
}

/// The routes, described (the page's client, the document)
pub fn infos() -> Vec<rpcgen::RouteInfo> {
    routes().into_iter().map(|r| r.info).collect()
}

/// An error in OpenAI's shape (what the routes answer)
pub(super) fn fail(m: String) -> Value {
    err(m)
}
