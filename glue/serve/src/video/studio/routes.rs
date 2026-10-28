//! The studio's own routes (rpcgen `#[route]`): what it serves besides POST /rpc/<method> and the daemon's /engine
//! passed through - the page's polling (/api/status, /api/summary ...), the legacy actions (/api/queue/pause,
//! /api/join, /api/gen ...), uploads, stills and clips. The table is what the studio dispatches by, wfe/video's client
//! (src/gen/routes.ts) and docs/api/video.openapi.yml are made from it (`nextcuda gen-api`), and each answer the
//! studio builds as JSON is held to its type as it is sent (`Checked`).
//!
//! The records a clip carries (a queue item, the running job, a finished clip's record) are the legacy server's open
//! objects (every generate key, plus what starting it settled): they are typed by the keys everything reads and keep
//! the rest as they are. An error's body is each route's own, as it always was (`{"error"}`, or `{"ok": false,
//! "error"}`), so a route fails with its status and that body.

use std::collections::BTreeMap;
use std::net::TcpStream;

use rpcgen::{route, routes, BadPayload, Checked, Done, RouteError, RouteReply, Rpc, RpcType, Sent, Shape, SyncRoute};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use super::queue::{self, s};
use super::{http, library, Studio, TYPES_TS};

/// A route's failure: the status and the body to answer with.
pub type Fail = (u16, Value);

/// What a route's handler gets besides the studio: the connection (for a file it writes itself) and the request.
pub struct Cx {
    pub s: TcpStream,
    pub req: http::Request,
}

/// Runs the route of the table that serves the request and writes its answer (JSON as the studio always wrote it:
/// pretty, no-cache); false when no route does. Any method but POST is served as a GET, as the studio always did.
pub fn serve(studio: &Studio, cx: &mut Cx) -> bool {
    let method = if cx.req.method == "POST" { "POST" } else { "GET" };
    let Some((r, params)) = rpcgen::find_route(table(), method, &cx.req.path) else { return false };
    let query = cx.req.target.split_once('?').map(|q| q.1.to_string()).unwrap_or_default();
    // a JSON body is the request; an upload's bytes the handler reads from cx
    let body = if r.info.body == rpcgen::Body::Json { std::mem::take(&mut cx.req.body) } else { Vec::new() };
    let _ = match r.call(studio, cx, params, &query, &body) {
        Ok(RouteReply::Json(v)) => http::respond(&cx.s, 200, &v),
        Ok(RouteReply::Sent) => Ok(()),
        Err(RouteError::Payload(BadPayload(m))) => http::respond(&cx.s, 400, &json!({"error": m})),
        Err(RouteError::Answer(m)) => http::respond(&cx.s, 500, &json!({"error": m})),
        Err(RouteError::Handler((code, body))) => http::respond(&cx.s, code, &body),
    };
    true
}

/// A number as the studio reads one: a JSON number, or text or a boolean that reads as one (its range is the
/// studio's to check)
#[derive(Debug, Clone)]
pub struct Num(pub Value);

impl<'de> Deserialize<'de> for Num {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let v = Value::deserialize(d)?;
        match &v {
            Value::Number(_) | Value::Bool(_) => Ok(Num(v)),
            Value::String(t) if t.trim().parse::<f64>().is_ok() => Ok(Num(v)),
            _ => Err(serde::de::Error::custom(format!("{v} is not a number"))),
        }
    }
}

impl Serialize for Num {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(s)
    }
}

impl RpcType for Num {
    fn shape(_: &mut rpcgen::Defs) -> Shape {
        Shape::Number
    }
}

/// Chaining a clip onto another: what `chain` sets (guide_clip, first_audio(_s), first_frame, chain_mode,
/// exposure_ref; a flat key of the request wins)
#[derive(Serialize, Deserialize, Rpc)]
pub struct Chain {
    /// the clip it follows: an anchor (prev, prev_cam, first, first_cam) or a clip's name (default prev)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub from: Option<String>,
    /// 0 or 17k + 5: the last N frames as a moving keyframe at frame 0 (trim N/2 each side when joining)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_frames: Option<Num>,
    /// with context_frames: the last N/24 s of sound too (default true)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sound: Option<bool>,
    /// from's last frame as first_frame (default: when context_frames is 0)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub still: Option<bool>,
    /// video | png | latent
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    /// exposure_ref: the carried pictures scaled to its brightness (an anchor, or true for from)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color_reanchor: Option<Value>,
    /// other keys
    #[serde(flatten)]
    pub rest: Map<String, Value>,
}

/// A clip asked for (video.generate's body, and the legacy /api/gen's): the studio checks each value and its range
#[derive(Serialize, Deserialize, Rpc)]
pub struct ClipRequest {
    pub prompt: String,
    /// 1-15.1, snapped up to 17k + 5 frames; the model is trained on 124-362 frames at 24 fps
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seconds: Option<Num>,
    /// instead of seconds: exactly 17k + 5 (22-362), else 400 naming the nearest
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frames: Option<Num>,
    /// idempotent submit: a clip of this id queued, running or made is answered instead
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reentrant_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chain: Option<Chain>,
    /// 1-40; below 8 only with a turbo LoRA
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steps: Option<Num>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<Num>,
    /// teacher | student (the text encoder)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub te: Option<Value>,
    /// "Name NN/MM: ..." - the NN drives edit order
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<Value>,
    /// 256-1344, multiples of 32, width * height <= 1032192
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<Num>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<Num>,
    /// the denoiser (engines' quant): INT8 (default), Q6_K, Q4_K_M; REF2VA is chosen by itself for ref_images / ref_videos
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine: Option<String>,
    /// png | video | latent | none
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chain_mode: Option<String>,
    /// an anchor (prev, prev_cam, first, first_cam: the series' previous clip or hub, *_cam the same camera only) or a still in the clips directory
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_frame: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_frame: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exposure_ref: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_audio: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_audio_s: Option<Value>,
    /// a driving soundtrack (a song: a sound file in the clips directory, video.upload): the clip's length of it from
    /// soundtrack_start is pinned under the whole clip as an audio keyframe, and the picture is made to it (a singer's
    /// mouth to the singing); the clip carries the file's own sound. Not with first_audio (a chain's sound is off)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub soundtrack: Option<Value>,
    /// seconds into the soundtrack at the clip's first frame (default 0); a chain's next clip starts where its first
    /// carried frame is in the song
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub soundtrack_start: Option<Num>,
    /// harmful here: decodes as visible confetti
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cond_noise_aug: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub camera: Option<Value>,
    /// latent upscale at decode, 1.5 = free resolution
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upscale: Option<Value>,
    /// esrgan-general (default) | esrgan-anime | latent
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upscaler: Option<Value>,
    /// "path:strength"
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loras: Option<Vec<Value>>,
    /// at most 3; voices to speak in: <Audio 1>, <Audio 2> ... (files in the clips directory)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ref_audios: Option<Vec<Value>>,
    /// at most 9 (Ref2VA), named in the prompt as <Picture i>
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ref_images: Option<Vec<Value>>,
    /// at most 3 (Ref2VA), named as <Video k>
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ref_videos: Option<Vec<Value>>,
    /// match | max
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ref_image_size: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ref_video_sound: Option<Value>,
    /// the keyframe pictures shown to the text encoder too (default on)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub te_pictures: Option<Value>,
    /// motion guide: "<anchor or file>[:frames[:at]]" - the last frames of a clip as one moving keyframe
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub guide_clip: Option<Value>,
    /// flow shifts (12, 3)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shift_video: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shift_audio: Option<Value>,
    /// masked run: a clip whose latents are kept outside the regenerated part
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<Value>,
    /// "from-to" seconds to regenerate; a source shorter than the clip is extended
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub regen: Option<Value>,
    /// "x0,y0,x1,y1" pixels: regenerate only inside
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub regen_box: Option<Value>,
    /// the Fun ControlNet: a control video (canny / depth / HED / MLSD / pose frames), a mask (white: regenerate) over a source video; files in the clips directory
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub control_video: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub control_mask: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub control_source: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub control_strength: Option<Num>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub control_start: Option<Num>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub control_end: Option<Num>,
    /// ComfyUI's sampler and schedule names (none: Euler on the shifted schedule)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampler: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schedule: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch: Option<String>,
    /// busy without it: 409 busy
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub queue: Option<bool>,
    /// -9..9: a queued clip goes ahead of lower ones
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub priority: Option<Num>,
    /// other keys (negative_prompt and cfg are taken and listed back as ignored: cfg 1, nothing to steer)
    #[serde(flatten)]
    pub rest: Map<String, Value>,
}

/// A clip's record: a queue item, the running job, or what /api/gen and /api/clip answer - every generate key as the
/// studio settled it, and what starting it added (the engine's id, the clip's name, when it started, its energy)
#[derive(Serialize, Deserialize, Rpc)]
pub struct Clip {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// the clip: h3_YYYYMMDD_HHMMSS (once it starts)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// the engine's job id
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// unix seconds
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub started: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seconds: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub energy_wh: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reentrant_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frames: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steps: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    /// teacher | student
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub te: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<i64>,
    /// an anchor or a still's file
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_frame: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub camera: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upscale: Option<f64>,
    /// "path:strength"
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loras: Option<Vec<String>>,
    /// /api/clip: the words spoken, and queued, held or running
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dialogue: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<ClipState>,
    /// every other key of the record
    #[serde(flatten)]
    pub rest: Map<String, Value>,
}

/// Where a clip is
#[derive(Serialize, Deserialize, Rpc)]
pub enum ClipState {
    #[serde(rename = "done")]
    Done,
    #[serde(rename = "running")]
    Running,
    #[serde(rename = "queued")]
    Queued,
    #[serde(rename = "held")]
    Held,
}

/// Which language model owns the card when no render does
#[derive(Serialize, Deserialize, Rpc)]
pub struct LlmStatus {
    /// a key of choices, or none
    pub mode: String,
    pub up: bool,
    pub running: Vec<String>,
    pub starting: Option<String>,
    /// mode -> its title
    pub choices: BTreeMap<String, String>,
    /// mode -> its URL
    pub urls: BTreeMap<String, String>,
    pub url: Option<String>,
}

/// A queued clip, as the queue panel shows it
#[derive(Serialize, Deserialize, Rpc)]
pub struct QueueItem {
    pub label: String,
    pub seconds: f64,
    pub steps: Option<i64>,
    pub engine: Option<String>,
    /// WxH
    pub size: String,
    /// it starts on a first_frame
    pub chain: bool,
    pub project: String,
    pub batch: String,
    /// paused (queue, project or batch): not run until resumed
    pub held: bool,
}

/// A batch of a project's queued clips
#[derive(Serialize, Deserialize, Rpc)]
pub struct Batch {
    pub batch: String,
    pub clips: i64,
    pub video_s: f64,
    pub paused: bool,
}

/// A project of the queue
#[derive(Serialize, Deserialize, Rpc)]
pub struct Project {
    pub project: String,
    pub clips: i64,
    pub video_s: f64,
    pub paused: bool,
    /// its first clip's place in the queue (0-based)
    pub first_index: i64,
    pub batches: Vec<Batch>,
}

/// What kind of failure the last clip had
#[derive(Serialize, Deserialize, Rpc)]
pub enum FailureCode {
    #[serde(rename = "out_of_memory")]
    OutOfMemory,
    #[serde(rename = "engine_lost")]
    EngineLost,
    #[serde(rename = "invalid_input")]
    InvalidInput,
    #[serde(rename = "engine_error")]
    EngineError,
    #[serde(rename = "cancelled")]
    Cancelled,
}

/// A failed or cancelled clip's error, as data
#[derive(Serialize, Deserialize, Rpc)]
pub struct Failure {
    pub code: FailureCode,
    /// true only for engine_lost: the same request may work
    pub retryable: bool,
    pub message: String,
    pub hint: String,
}

/// The studio now: the clip being made (or the last one) and how far it is, the queue and its holds, the projects,
/// the language model; `version` changes when any of it does
#[derive(Serialize, Deserialize, Rpc)]
pub struct Status {
    /// nothing running
    pub idle: bool,
    /// the running or last clip; null: nothing made since the studio started
    pub job: Option<Clip>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pct: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stage: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub done: Option<bool>,
    /// failed or cancelled: what happened, and the engine's last lines
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub failure: Option<Failure>,
    /// the card the clip runs on
    #[serde(default)]
    pub gpu: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exited: Option<bool>,
    /// 0 done, 1 failed (once it ended)
    #[serde(default)]
    pub rc: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub elapsed: Option<i64>,
    /// seconds left, estimated
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eta: Option<i64>,
    #[serde(default)]
    pub rms: Option<f64>,
    /// the stages' seconds, by the legacy record's labels (TOTAL, video decoded ...)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub times: Option<BTreeMap<String, f64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frames: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steps_done: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steps_total: Option<i64>,
    #[serde(default)]
    pub per_step: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metrics: Option<Vec<Value>>,
    /// done: the clip (/out/<name>.mp4)
    #[serde(default)]
    pub file: Option<String>,
    /// while it samples: a picture of the shot so far (PNG URL, new each step)
    #[serde(default)]
    pub preview: Option<String>,
    /// the engine's last 12 lines
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine_log: Option<Vec<String>>,
    /// the queue's labels, in order
    pub queue: Vec<String>,
    pub queue_items: Vec<QueueItem>,
    /// the whole queue held
    pub paused: bool,
    pub paused_projects: Vec<String>,
    pub paused_batches: Vec<String>,
    pub projects: Vec<Project>,
    /// the queue's projects, then the finished clips', newest first
    pub all_projects: Vec<String>,
    pub llm: LlmStatus,
    /// changes when the status does (video.wait waits for that)
    pub version: String,
    pub queue_len: i64,
    pub stale: bool,
}

/// The clips directory's disk
#[derive(Serialize, Deserialize, Rpc)]
pub struct Disk {
    pub path: String,
    pub total_gb: f64,
    pub used_gb: f64,
    pub free_gb: f64,
    pub pct: f64,
}

/// The running clip's project, so far
#[derive(Serialize, Deserialize, Rpc)]
pub struct ProjectStats {
    pub project: String,
    pub clips: i64,
    pub video_s: f64,
    pub gpu_s: i64,
    /// GPU seconds a second of video
    pub ratio: f64,
    pub wh: i64,
}

/// The rates of the last clips made and what the queue will take
#[derive(Serialize, Deserialize, Rpc)]
pub struct Summary {
    /// wall seconds a second of video, over the last 10 clips
    pub rate_wall_per_video_s: f64,
    pub rate_wh_per_video_s: f64,
    /// how many clips the rates are of
    pub sample_clips: i64,
    pub running: bool,
    pub current_eta_s: f64,
    pub current_video_s: f64,
    pub queued: i64,
    pub queued_video_s: f64,
    pub remaining_wall_s: f64,
    /// unix seconds when the queue is done (null: nothing to do)
    pub eta_ts: Option<f64>,
    pub remaining_kwh: f64,
    pub disk: Option<Disk>,
    pub project_stats: Option<ProjectStats>,
    pub done_total: i64,
    pub done_video_s: f64,
    pub done_wall_s: f64,
    pub done_wh: f64,
}

/// A PCIe link's generation and width
#[derive(Serialize, Deserialize, Rpc)]
pub struct PcieLink {
    pub gen: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gts: Option<String>,
    pub width: i64,
}

/// A card's three links: now, the card's best, the slot's best
#[derive(Serialize, Deserialize, Rpc)]
pub struct Pcie {
    pub cur: PcieLink,
    pub card_max: PcieLink,
    pub slot_max: PcieLink,
}

/// The card's telemetry (the gpustat file, or the first card sampled here, as `nextcuda gpustat` writes it), with
/// how old it is; no telemetry: `error` alone
#[derive(Serialize, Deserialize, Rpc)]
pub struct Gpu {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pci: Option<String>,
    /// unix seconds sampled, and seconds since
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ts: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub age: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vram_used_mb: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vram_total_mb: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub busy_pct: Option<f64>,
    /// °C
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temp_pkg: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temp_vram: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temp_vram_max: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temp_pcie: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temp_mctrl: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub power_w: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pkg_power_w: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub power_cap_w: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pcie: Option<Pcie>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub host_load1: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub host_cpus: Option<i64>,
    /// the card's energy counter, joules (a clip's energy is its rise)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub energy_j: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fan_pct: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fan_rpm: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub freq_mhz: Option<f64>,
    /// no telemetry to read
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// a sampler's other readings (the file another sampler wrote)
    #[serde(flatten)]
    pub rest: Map<String, Value>,
}

/// A denoiser the engine has
#[derive(Serialize, Deserialize, Rpc)]
pub struct EngineInfo {
    /// INT8, Q6_K, Q4_K_M, REF2VA
    pub quant: String,
    pub file: String,
    pub path: String,
    pub gib: f64,
    pub ready: bool,
    pub default: Option<bool>,
    /// what it takes (video.generate's keys): takes, not, chosen, ignored
    pub inputs: Value,
    /// the licence of its weights, whether it allows commercial use (false: the picker tags it non-commercial), the conditions it puts on it
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub commercial: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license_note: Option<String>,
}

/// The denoisers
#[derive(Serialize, Deserialize, Rpc)]
pub struct Engines {
    pub engines: Vec<EngineInfo>,
}

/// A canvas the table offers, with its price against 640x480
#[derive(Serialize, Deserialize, Rpc)]
pub struct Canvas {
    pub width: i64,
    pub height: i64,
    pub aspect: String,
    pub note: String,
    pub mpx: f64,
    pub cost: f64,
    pub default: bool,
}

/// The canvases
#[derive(Serialize, Deserialize, Rpc)]
pub struct Canvases {
    pub canvases: Vec<Canvas>,
}

/// The defaults a new clip starts from
#[derive(Serialize, Deserialize, Rpc)]
pub struct PlanDefaults {
    pub seconds: i64,
    pub steps: i64,
    pub width: i64,
    pub height: i64,
    pub engine: String,
}

/// A canvas table cell measured on one GPU: a clip step's time and peak VRAM, or `over` when the engine refused the size
#[derive(Serialize, Deserialize, Rpc)]
pub struct PlanCell {
    pub tokens: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub s_per_step: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gib: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub over: Option<bool>,
}

/// One GPU's measured step cost (`nextcuda video plan measure`, plan.json as it wrote it): the 50 blocks at each
/// token count, and the factor a real clip's step is of them
#[derive(Serialize, Deserialize, Rpc)]
pub struct GpuPlan {
    pub gpu: i64,
    /// the denoiser measured (INT8 when absent: a plan from before engines)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine: Option<String>,
    pub name: String,
    pub shared: bool,
    pub step_scale: f64,
    pub clip_tokens: Option<f64>,
    /// [tokens, seconds a step] measured, each a pair of numbers (rpcgen describes no array of arrays in parity yet)
    pub points: Vec<Value>,
    pub step_a: f64,
    pub step_b: f64,
    /// the canvas table's cells measured on this GPU, keyed "WxH|seconds"
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cells: Option<BTreeMap<String, PlanCell>>,
}

/// A cell this box has run (the legacy table's)
#[derive(Serialize, Deserialize, Rpc)]
pub struct MeasuredCell {
    pub peak: f64,
    pub resv: Option<f64>,
    pub s_per_step: Option<f64>,
    pub steps: Option<i64>,
    pub tokens: f64,
    pub name: String,
}

/// Everything the canvas grid needs to price and disqualify a combination
#[derive(Serialize, Deserialize, Rpc)]
pub struct Plan {
    pub engines: Vec<EngineInfo>,
    pub canvases: Vec<Canvas>,
    pub defaults: PlanDefaults,
    /// what the engine may hold of a card, GiB
    pub cap_gib: f64,
    pub gib_per_token: f64,
    pub margin_gib: f64,
    /// the fastest GPU's step time: a * tokens^2 + b * tokens
    pub step_a: f64,
    pub step_b: f64,
    /// measured per GPU; empty until `nextcuda video plan measure` has run on the box
    pub gpus: Vec<GpuPlan>,
    /// chain_mode -> what it does
    pub chain_modes: BTreeMap<String, String>,
    /// cells this box has run, keyed "<engine>|<W>x<H>|<seconds>" (none yet)
    pub measured: BTreeMap<String, MeasuredCell>,
    /// the sampler and schedule names the engine takes
    pub samplers: Vec<String>,
    pub schedules: Vec<String>,
}

/// A finished clip in the clips directory, newest first, with its record's summary when it has one
#[derive(Serialize, Deserialize, Rpc)]
pub struct ListRow {
    pub file: String,
    /// bytes; WxH when the clip has a record (its bytes in size2): the legacy page's mixed key
    pub size: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size2: Option<i64>,
    pub mtime: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steps: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub te: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seconds: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub energy_wh: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speech_pct: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub camera: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project: Option<String>,
}

/// A prompt template (the templates directory's .txt files)
#[derive(Serialize, Deserialize, Rpc)]
pub struct Template {
    pub name: String,
    pub text: String,
}

/// A joined film (an mp4 that is not a single clip)
#[derive(Serialize, Deserialize, Rpc)]
pub struct Film {
    pub file: String,
    pub mb: f64,
    pub seconds: Option<f64>,
    pub mtime: f64,
}

/// Where a join is
#[derive(Serialize, Deserialize, Rpc)]
pub enum JoinState {
    #[serde(rename = "running")]
    Running,
    #[serde(rename = "done")]
    Done,
    #[serde(rename = "failed")]
    Failed,
}

/// A film being joined, or joined since the studio started
#[derive(Serialize, Deserialize, Rpc)]
pub struct Join {
    pub project: String,
    pub clips: i64,
    pub file: String,
    pub state: JoinState,
    pub started: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finished: Option<f64>,
    /// the join's last line
    pub message: String,
}

/// The joins
#[derive(Serialize, Deserialize, Rpc)]
pub struct Joins {
    pub joins: Vec<Join>,
}

/// A join started
#[derive(Serialize, Deserialize, Rpc)]
pub struct JoinStarted {
    pub ok: bool,
    pub join: Join,
}

/// A clip of a project's timeline, in edit order
#[derive(Serialize, Deserialize, Rpc)]
pub struct TimelineClip {
    /// the label's clip number (the edit order)
    pub n: i64,
    /// the clip (null: not made yet)
    pub name: Option<String>,
    pub state: ClipState,
    pub seconds: Option<f64>,
    pub camera: Option<String>,
    pub label: String,
    /// spoken words only, for alt text
    pub dialogue: String,
    /// /thumb/<name>.webp
    pub thumb: Option<String>,
}

/// A project's timeline
#[derive(Serialize, Deserialize, Rpc)]
pub struct Timeline {
    pub clips: Vec<TimelineClip>,
}

/// A portable scene document: prompts are final, a runner merges the defaults into each clip and posts it
#[derive(Serialize, Deserialize, Rpc)]
pub struct Scene {
    /// 1
    pub version: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    pub project: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exported: Option<i64>,
    /// the settings every clip shares
    pub defaults: Map<String, Value>,
    /// each clip: n (the edit order), prompt, and its own settings
    pub clips: Vec<Map<String, Value>>,
}

/// A finished clip's record (`<clip>.json`): the job as made, its stages' times, frames, wall time, energy
#[derive(Serialize, Deserialize, Rpc)]
pub struct ClipRecord {
    pub job: Clip,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub times: Option<BTreeMap<String, f64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frames: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wall: Option<f64>,
    #[serde(default)]
    pub energy_wh: Option<f64>,
    /// the record's other keys (rms ...)
    #[serde(flatten)]
    pub rest: Map<String, Value>,
}

/// A queue cleared
#[derive(Serialize, Deserialize, Rpc)]
pub struct Cleared {
    pub cleared: i64,
}

/// A scene imported
#[derive(Serialize, Deserialize, Rpc)]
pub struct Imported {
    /// clips queued from it
    pub imported: i64,
    pub project: String,
    /// the queue's length now, and how many of it are held
    pub queued: i64,
    pub held: i64,
}

/// A file kept
#[derive(Serialize, Deserialize, Rpc)]
pub struct Uploaded {
    pub ok: bool,
    /// its name in the clips directory (ref_<seconds>_<name>): what a request names it by
    pub name: String,
}

/// A project's timeline or scene
#[derive(Serialize, Deserialize, Rpc)]
pub struct ProjectRef {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project: Option<String>,
}

/// A queued clip by its label
#[derive(Serialize, Deserialize, Rpc)]
pub struct LabelRef {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

/// A file a link names
#[derive(Serialize, Deserialize, Rpc)]
pub struct FileName {
    /// the file's name as the link has it
    pub name: String,
}

/// An upload's name
#[derive(Serialize, Deserialize, Rpc)]
pub struct UploadName {
    /// the file's own name (its extension says what it is: png, jpg, jpeg, webp, mp4, mov, webm, mkv, wav, mp3, flac, m4a, ogg)
    #[serde(default)]
    pub name: String,
}

/// A hold on or off
#[derive(Serialize, Deserialize, Rpc)]
pub struct HoldReq {
    /// true: hold; false: release (releasing the whole queue releases every project and batch too)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub on: Option<bool>,
    /// this project only
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project: Option<String>,
    /// this batch only
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch: Option<String>,
}

/// A scene to queue: the document as `scene`, or the body itself
#[derive(Serialize, Deserialize, Rpc)]
pub struct SceneImport {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scene: Option<Map<String, Value>>,
    /// the project to queue it as (default: the scene's)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project: Option<String>,
    /// queue it held (its project paused)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub paused: Option<bool>,
    /// the scene's own keys, when the body is the document
    #[serde(flatten)]
    pub rest: Map<String, Value>,
}

fn encode<T: Serialize>(v: &T) -> Result<Value, Fail> {
    serde_json::to_value(v).map_err(|e| (500, json!({"error": e.to_string()})))
}

/// The card's telemetry
#[route(get = "/api/gpu", tags = "page")]
fn gpu(s: &Studio, _cx: &mut Cx) -> Result<Checked<Gpu>, Fail> {
    Ok(Checked::new(s.gpu()))
}

/// The rates of the last clips made, what the queue will take, the disk, the running project's totals
#[route(get = "/api/summary", tags = "page")]
fn summary(s: &Studio, _cx: &mut Cx) -> Result<Checked<Summary>, Fail> {
    Ok(Checked::new(s.summary(&mut s.st.lock().unwrap())))
}

/// The denoisers the engine has
#[route(get = "/api/engines", tags = "page")]
fn engines(s: &Studio, _cx: &mut Cx) -> Result<Checked<Engines>, Fail> {
    Ok(Checked::new(json!({"engines": s.engines()})))
}

/// The canvases offered
#[route(get = "/api/canvases", tags = "page")]
fn canvases(s: &Studio, _cx: &mut Cx) -> Result<Checked<Canvases>, Fail> {
    Ok(Checked::new(json!({"canvases": s.canvases()})))
}

/// The canvas table's model: the denoisers, canvases, defaults, what a card holds and each GPU's measured step cost
#[route(get = "/api/plan", tags = "page")]
fn plan(s: &Studio, _cx: &mut Cx) -> Result<Checked<Plan>, Fail> {
    Ok(Checked::new(s.plan()))
}

/// The studio now (as POST /rpc/status)
#[route(get = "/api/status", tags = "page")]
fn status(s: &Studio, _cx: &mut Cx) -> Result<Checked<Status>, Fail> {
    Ok(Checked::new(s.versioned_status()))
}

/// Every finished clip, newest first
#[route(get = "/api/list", tags = "page")]
fn list(s: &Studio, _cx: &mut Cx) -> Result<Checked<Vec<ListRow>>, Fail> {
    Ok(Checked::new(json!(s.listing())))
}

/// The prompt templates
#[route(get = "/api/templates", tags = "page")]
fn templates(s: &Studio, _cx: &mut Cx) -> Result<Checked<Vec<Template>>, Fail> {
    Ok(Checked::new(json!(s.templates())))
}

/// The joined films, newest first
#[route(get = "/api/films", tags = "page")]
fn films(s: &Studio, _cx: &mut Cx) -> Result<Checked<Vec<Film>>, Fail> {
    Ok(Checked::new(json!(s.films())))
}

/// The films being joined, and joined since the studio started
#[route(get = "/api/join", tags = "page")]
fn joins(s: &Studio, _cx: &mut Cx) -> Result<Checked<Joins>, Fail> {
    Ok(Checked::new(json!({"joins": *s.joins.lock().unwrap()})))
}

/// A project's clips in edit order: finished (the newest retry of each number), running, held, queued
#[route(get = "/api/timeline", tags = "page")]
fn timeline(s: &Studio, _cx: &mut Cx, req: ProjectRef) -> Result<Checked<Timeline>, Fail> {
    Ok(Checked::new(json!({"clips": s.timeline(&s.st.lock().unwrap(), req.project.as_deref().unwrap_or(""))})))
}

/// The RPC layer's TypeScript (its types and an rpc() helper)
#[route(get = "/api/types.ts", tags = "page", produces = "application/typescript")]
fn types_ts(_s: &Studio, cx: &mut Cx) -> Result<Sent, Fail> {
    let _ = http::respond_bytes(&cx.s, 200, "application/typescript; charset=utf-8", TYPES_TS.as_bytes());
    Ok(Sent::new())
}

/// A project as a portable scene file (pretty JSON)
#[route(get = "/api/scene", tags = "page")]
fn scene(s: &Studio, cx: &mut Cx, req: ProjectRef) -> Result<Sent<Scene>, Fail> {
    let doc = Checked::<Scene>::new(s.scene_export(&s.st.lock().unwrap(), req.project.as_deref().unwrap_or(""))).0;
    let _ = http::respond_bytes(&cx.s, 200, "application/json; charset=utf-8", serde_json::to_string_pretty(&doc).unwrap_or_default().as_bytes());
    Ok(Sent::new())
}

/// A clip not made yet, by its label: running, held or queued, with the words it speaks
#[route(get = "/api/clip", tags = "page")]
fn clip(s: &Studio, _cx: &mut Cx, req: LabelRef) -> Result<Checked<Clip>, Fail> {
    let label = req.label.unwrap_or_default();
    let st = s.st.lock().unwrap();
    let running = st.job.as_ref().filter(|j| !j.finished).map(|j| (j.rec.clone(), "running"));
    let found = running.into_iter().chain(st.q.items.iter().map(|i| (i.clone(), if st.q.held(i) { "held" } else { "queued" }))).find(|(i, _)| s_of(i, "label") == label);
    match found {
        Some((mut i, state)) => {
            i["dialogue"] = json!(queue::dialogue_of(&s_of(&i, "prompt")));
            i["state"] = json!(state);
            Ok(Checked::new(i))
        }
        None => Err((404, json!({"error": "no queued clip with that label"}))),
    }
}

fn s_of(v: &Value, k: &str) -> String {
    s(v, k)
}

/// A finished clip's record
#[route(get = "/api/job/{name}", tags = "page")]
fn job(s: &Studio, _cx: &mut Cx, req: FileName) -> Result<Checked<ClipRecord>, Fail> {
    let n = queue::basename(&req.name);
    match s.record(&n) {
        Some(v) => Ok(Checked::new(v)),
        None => Err((404, json!({"error": format!("no record for {n}")}))),
    }
}

/// A finished clip's 240-pixel still, 5 s in (made once, then kept)
#[route(get = "/thumb/{name}", tags = "files", produces = "image/webp")]
fn thumb(s: &Studio, cx: &mut Cx, req: FileName) -> Result<Sent, Fail> {
    let n = queue::basename(&req.name);
    let n = n.strip_suffix(".webp").unwrap_or(&n);
    match library::is_clip_name(n).then(|| s.thumb(n)).flatten() {
        Some(b) => {
            let _ = http::respond_bytes(&cx.s, 200, "image/webp", &b);
            Ok(Sent::new())
        }
        None => Err((404, json!({"error": "no thumbnail"}))),
    }
}

/// A clip (mp4, in byte ranges: the player seeks), or its preview while it samples (<clip>.preview.png) and its last frame (<clip>.last.png)
#[route(get = "/out/{name}", tags = "files", produces = "video/mp4, image/png", ranges = true)]
fn out(s: &Studio, cx: &mut Cx, req: FileName) -> Result<Sent, Fail> {
    s.file(&cx.s, &cx.req, &queue::basename(&req.name)).map_err(|e| (500, json!({"error": e.0})))?;
    Ok(Sent::new())
}

/// A reference picture, clip or sound, kept beside the clips (references are named by file there): the body is the file
#[route(post = "/api/upload", tags = "page", body = "bytes")]
fn upload(s: &Studio, cx: &mut Cx, req: UploadName) -> Result<Uploaded, Fail> {
    match s.upload(&req.name, &cx.req.body) {
        Ok(name) => Ok(Uploaded { ok: true, name }),
        Err(e) => Err((400, json!({"ok": false, "error": e}))),
    }
}

/// Holds the queue, a project or a batch (on), or releases it
#[route(post = "/api/queue/pause", tags = "page")]
fn queue_pause(s: &Studio, _cx: &mut Cx, req: HoldReq) -> Result<Done, Fail> {
    s.hold(req.on == Some(true), req.project.as_deref().unwrap_or("").trim(), req.batch.as_deref().unwrap_or("").trim());
    Ok(Done::new())
}

/// Joins a project's finished clips, in timeline order, into one film (in the background: GET /api/join follows it)
#[route(post = "/api/join", tags = "page")]
fn join(s: &Studio, _cx: &mut Cx, req: ProjectRef) -> Result<Checked<JoinStarted>, Fail> {
    s.join(req.project.as_deref().unwrap_or("").trim()).map(Checked::new).map_err(|e| (400, json!({"ok": false, "error": e})))
}

/// Empties the queue
#[route(post = "/api/queue/clear", tags = "page")]
fn queue_clear(s: &Studio, _cx: &mut Cx) -> Result<Cleared, Fail> {
    let mut st = s.st.lock().unwrap();
    let n = st.q.items.len();
    st.q.items.clear();
    st.q.save(&s.dir);
    Ok(Cleared { cleared: n as i64 })
}

/// Queues a scene file's clips (all or nothing), as a project, held or not
#[route(post = "/api/scene/import", tags = "page")]
fn scene_import(s: &Studio, _cx: &mut Cx, req: SceneImport) -> Result<Checked<Imported>, Fail> {
    let body = encode(&req)?;
    let doc = if body["scene"].is_object() { &body["scene"] } else { &body };
    s.scene_import(&mut s.st.lock().unwrap(), doc, body["project"].as_str(), body["paused"] == true)
        .map(Checked::new)
        .map_err(|e| (e.http, json!({"error": e.message})))
}

/// The oldest route: a clip made or queued, its seconds and steps clamped (the RPC refuses them instead) and an unknown te read as teacher; answers the queue position and label, or the job started
#[route(post = "/api/gen", tags = "page")]
fn gen(s: &Studio, _cx: &mut Cx, req: ClipRequest) -> Result<Checked<Clip>, Fail> {
    let body = encode(&req)?;
    let mut b = body.clone();
    let f = |v: &Value, d: f64| v.as_f64().or_else(|| v.as_str().and_then(|x| x.parse().ok())).unwrap_or(d);
    b["seconds"] = json!(f(&body["seconds"], 10.0).clamp(1.0, 15.1));
    b["steps"] = json!((f(&body["steps"], 10.0) as i64).clamp(1, 40));
    if !matches!(body["te"].as_str(), Some("teacher" | "student")) {
        b["te"] = json!("teacher");
    }
    match s.generate(&b) {
        Ok(v) if v["queued"] == true => Ok(Checked::new(json!({"queued": v["position"], "label": v["label"]}))),
        Ok(v) => Ok(Checked::new(v["job"].clone())),
        Err(e) => Err((e.http, json!({"error": e.message}))),
    }
}

routes![sync Studio, Cx, Fail; gpu, summary, engines, canvases, plan, status, list, templates, films, joins, timeline, types_ts, scene, clip, job,
        thumb, out, upload, queue_pause, join, queue_clear, scene_import, gen];

/// The table, made once
pub fn table() -> &'static [SyncRoute<Studio, Cx, Fail>] {
    static T: std::sync::OnceLock<Vec<SyncRoute<Studio, Cx, Fail>>> = std::sync::OnceLock::new();
    T.get_or_init(routes)
}

/// The routes, described (the page's client, the document)
pub fn infos() -> Vec<rpcgen::RouteInfo> {
    routes().into_iter().map(|r| r.info).collect()
}
