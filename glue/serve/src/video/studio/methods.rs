//! The studio's RPC methods (`POST /rpc/<method>`, H3's names: `status`, `queue.pause` ...), each its #[rpc]
//! handler (rpcgen): the table the studio dispatches by and the TypeScript it serves at /api/types.ts and wfe/video
//! imports (src/gen/studio.ts), made by `nextcuda gen-api`. The work is still `Studio::rpc`'s; a handler checks its
//! request against its type and holds the answer to its own (`Checked`). The envelope - `{"ok": true, "result"}` or
//! `{"ok": false, "error": {"code", "message"}}` - and the error codes are the studio's as they were.

use rpcgen::{rpc, BadPayload, Checked, Rpc};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::home::methods::VideoGenerateReq;
use super::routes::{Canvases, Clip, ClipRecord, Engines, FileName, Film, Gpu, LlmStatus, Plan, Project, ProjectRef, Scene,
                    SceneImport, Status, Summary, Template, Timeline, Imported, Cleared, Num};
use super::{RpcError, Studio};

impl From<BadPayload> for RpcError {
    fn from(b: BadPayload) -> Self {
        RpcError::param(&b.0)
    }
}

type Answer<T> = Result<Checked<T>, RpcError>;

/// The work as `Studio::rpc` does it, the answer held to T
fn legacy<T: serde::de::DeserializeOwned + Serialize>(s: &Studio, name: &str, req: impl Serialize) -> Answer<T> {
    let b = serde_json::to_value(req).map_err(|e| RpcError::param(&e.to_string()))?;
    let b = if b.is_null() { Value::Object(Map::new()) } else { b };
    s.rpc(name, &b).map(Checked::new)
}

/// What `wait` takes
#[derive(Serialize, Deserialize, Rpc)]
pub struct WaitReq {
    /// the status version the caller has: it answers when the status is another (or at timeout_s)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    /// seconds at most
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(min = 0, max = 60, default = 30)]
    pub timeout_s: Option<Num>,
}

/// The status, and whether it changed from the version given
#[derive(Serialize, Deserialize, Rpc)]
pub struct Waited {
    #[serde(flatten)]
    pub status: Status,
    pub changed: bool,
}

/// The templates (prompt starts the page offers)
#[derive(Serialize, Deserialize, Rpc)]
pub struct Templates {
    pub templates: Vec<Template>,
}

/// What generate answered: started (job), queued (position ...), or a reentrant_id's clip as it is (state)
#[derive(Serialize, Deserialize, Rpc)]
pub struct Generated {
    pub queued: bool,
    /// queued: its place (1 first)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub position: Option<i64>,
    /// started or running: the job's record
    #[serde(default)]
    pub job: Option<Clip>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub held: Option<bool>,
    /// the clip as made: 17k + 5 frames
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frames: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seconds: Option<Value>,
    /// what the request had that the box does nothing with
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ignored: Option<Vec<String>>,
    /// a clip of this reentrant_id was there already: this is it
    pub reentered: bool,
    /// reentered: queued, held, running or done
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<String>,
    /// done: the clip and its file
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
}

/// The queue: every clip waiting, its request as checked
#[derive(Serialize, Deserialize, Rpc)]
pub struct QueueList {
    pub queue: Vec<Clip>,
}

/// A project or a batch (none: the whole queue)
#[derive(Serialize, Deserialize, Rpc)]
pub struct Scope {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch: Option<String>,
}

/// What holding or releasing did
#[derive(Serialize, Deserialize, Rpc)]
pub struct HoldResult {
    /// what was held ("queue", "project x", "batch y")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub paused: Option<String>,
    /// what was released
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resumed: Option<String>,
    pub queued: i64,
    pub held: i64,
    pub paused_all: bool,
    pub paused_projects: Vec<String>,
    pub paused_batches: Vec<String>,
}

/// A clip by its label (none: the one running)
#[derive(Serialize, Deserialize, Rpc)]
pub struct LabelReq {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

/// What a cancel did
#[derive(Serialize, Deserialize, Rpc)]
pub struct Cancelled {
    /// the running clip stopped
    pub stopped: bool,
    /// queued clips dropped
    pub removed: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

/// A queued clip to drop: by its label, or by its place (the label, when given too, has to match)
#[derive(Serialize, Deserialize, Rpc)]
pub struct RemoveReq {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// 0 first
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<u64>,
}

/// What dropping did
#[derive(Serialize, Deserialize, Rpc)]
pub struct Removed {
    pub removed: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<i64>,
    pub label: String,
}

/// A queued clip moved (and its priority set)
#[derive(Serialize, Deserialize, Rpc)]
pub struct MoveReq {
    pub label: String,
    /// 1 first
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(min = 1, default = 1)]
    pub position: Option<Num>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(min = -9, max = 9)]
    pub priority: Option<Num>,
}

/// Where it went, and the queue's labels now
#[derive(Serialize, Deserialize, Rpc)]
pub struct Moved {
    pub label: String,
    pub position: i64,
    pub queue: Vec<String>,
}

/// The chat model the studio's prompt writer runs on (none: stopped)
#[derive(Serialize, Deserialize, Rpc)]
pub struct ModeReq {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
}

/// Every film joined
#[derive(Serialize, Deserialize, Rpc)]
pub struct Films {
    pub films: Vec<Film>,
}

/// The projects in the queue, and whether the whole queue is held
#[derive(Serialize, Deserialize, Rpc)]
pub struct Projects {
    pub projects: Vec<Project>,
    pub paused_all: bool,
}

/// How many of the clips made to list
#[derive(Serialize, Deserialize, Rpc)]
pub struct JobsReq {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(min = 1, max = 500, default = 50)]
    pub limit: Option<Num>,
}

/// The clips made, newest first: one type a key (size in bytes, canvas WxH)
#[derive(Serialize, Deserialize, Rpc)]
pub struct Jobs {
    pub jobs: Vec<Map<String, Value>>,
}

/// A file in the clips directory
#[derive(Serialize, Deserialize, Rpc)]
pub struct FileStat {
    pub name: String,
    pub bytes: i64,
    pub sha256: String,
    /// unix seconds
    pub mtime: f64,
}

/// The files deleted (a clip and every file of it)
#[derive(Serialize, Deserialize, Rpc)]
pub struct Deleted {
    pub deleted: Vec<String>,
}

/// What the studio is doing: the clip being made (stage, progress, its preview), the queue, the projects
#[rpc(name = "status")]
fn status(s: &Studio) -> Answer<Status> {
    legacy(s, "status", Map::new())
}

/// The status once it changes from `version` (or at timeout_s): a long poll
#[rpc(name = "wait")]
fn wait(s: &Studio, req: WaitReq) -> Answer<Waited> {
    legacy(s, "wait", req)
}

/// The card the studio runs on: its memory, power, temperatures, link
#[rpc(name = "gpu")]
fn gpu(s: &Studio) -> Answer<Gpu> {
    legacy(s, "gpu", Map::new())
}

/// The clips made, their time and energy, by project
#[rpc(name = "summary")]
fn summary(s: &Studio) -> Answer<Summary> {
    legacy(s, "summary", Map::new())
}

/// The engines (quantizations) the daemon offers
#[rpc(name = "engines.list")]
fn engines_list(s: &Studio) -> Answer<Engines> {
    legacy(s, "engines.list", Map::new())
}

/// The canvases (sizes) that fit, by engine
#[rpc(name = "canvases.list")]
fn canvases_list(s: &Studio) -> Answer<Canvases> {
    legacy(s, "canvases.list", Map::new())
}

/// What fits on the cards: length and size by engine, measured and estimated
#[rpc(name = "plan")]
fn plan(s: &Studio) -> Answer<Plan> {
    legacy(s, "plan", Map::new())
}

/// The prompt templates
#[rpc(name = "templates.list")]
fn templates_list(s: &Studio) -> Answer<Templates> {
    legacy(s, "templates.list", Map::new())
}

/// A clip: started at once, or queued (queue: true) when one is running; a reentrant_id's clip is answered as it is.
/// It takes what the front door's video.generate takes (which passes its body on here)
#[rpc(name = "generate")]
fn generate(s: &Studio, req: VideoGenerateReq) -> Answer<Generated> {
    legacy(s, "generate", req)
}

/// The clips waiting
#[rpc(name = "queue.list")]
fn queue_list(s: &Studio) -> Answer<QueueList> {
    legacy(s, "queue.list", Map::new())
}

/// Holds the queue, a project or a batch (their clips wait until resumed)
#[rpc(name = "queue.pause")]
fn queue_pause(s: &Studio, req: Scope) -> Answer<HoldResult> {
    legacy(s, "queue.pause", req)
}

/// Releases the queue (every project and batch too), a project or a batch
#[rpc(name = "queue.resume")]
fn queue_resume(s: &Studio, req: Scope) -> Answer<HoldResult> {
    legacy(s, "queue.resume", req)
}

/// Drops the clips waiting: all, a project's or a batch's
#[rpc(name = "queue.clear")]
fn queue_clear(s: &Studio, req: Scope) -> Answer<Cleared> {
    legacy(s, "queue.clear", req)
}

/// Stops the clip running, or one by its label (running: stopped; queued: dropped)
#[rpc(name = "cancel")]
fn cancel(s: &Studio, req: LabelReq) -> Answer<Cancelled> {
    legacy(s, "cancel", req)
}

/// Drops a queued clip
#[rpc(name = "queue.remove")]
fn queue_remove(s: &Studio, req: RemoveReq) -> Answer<Removed> {
    legacy(s, "queue.remove", req)
}

/// Moves a queued clip (and sets its priority)
#[rpc(name = "queue.move")]
fn queue_move(s: &Studio, req: MoveReq) -> Answer<Moved> {
    legacy(s, "queue.move", req)
}

/// The chat model the prompt writer uses: set (and started), or none
#[rpc(name = "llm.mode")]
fn llm_mode(s: &Studio, req: ModeReq) -> Answer<LlmStatus> {
    legacy(s, "llm.mode", req)
}

/// A project as a scene document (its clips' final prompts and settings)
#[rpc(name = "scene.export")]
fn scene_export(s: &Studio, req: ProjectRef) -> Answer<Scene> {
    legacy(s, "scene.export", req)
}

/// A scene document queued (as a project; held with paused)
#[rpc(name = "scene.import")]
fn scene_import(s: &Studio, req: SceneImport) -> Answer<Imported> {
    legacy(s, "scene.import", req)
}

/// The films joined
#[rpc(name = "films.list")]
fn films_list(s: &Studio) -> Answer<Films> {
    legacy(s, "films.list", Map::new())
}

/// A project's clips in order: made, running, queued
#[rpc(name = "timeline")]
fn timeline(s: &Studio, req: ProjectRef) -> Answer<Timeline> {
    legacy(s, "timeline", req)
}

/// The projects in the queue
#[rpc(name = "projects.list")]
fn projects_list(s: &Studio) -> Answer<Projects> {
    legacy(s, "projects.list", Map::new())
}

/// The clips made, newest first
#[rpc(name = "jobs.list")]
fn jobs_list(s: &Studio, req: JobsReq) -> Answer<Jobs> {
    legacy(s, "jobs.list", req)
}

/// A clip's record (its request, its times, its energy)
#[rpc(name = "jobs.get")]
fn jobs_get(s: &Studio, req: FileName) -> Answer<ClipRecord> {
    legacy(s, "jobs.get", req)
}

/// A file in the clips directory: its size, hash and time
#[rpc(name = "files.stat")]
fn files_stat(s: &Studio, req: FileName) -> Answer<FileStat> {
    legacy(s, "files.stat", req)
}

/// A finished clip and every file of it, or an upload (never the clip being made)
#[rpc(name = "files.delete")]
fn files_delete(s: &Studio, req: FileName) -> Answer<Deleted> {
    legacy(s, "files.delete", req)
}

rpcgen::rpc_methods![sync Studio, RpcError; status, wait, gpu, summary, engines_list, canvases_list, plan, templates_list, generate, queue_list,
                     queue_pause, queue_resume, queue_clear, cancel, queue_remove, queue_move, llm_mode, scene_export, scene_import, films_list,
                     timeline, projects_list, jobs_list, jobs_get, files_stat, files_delete];

/// The table, made once
pub fn table() -> &'static [rpcgen::SyncMethod<Studio, RpcError>] {
    static T: std::sync::OnceLock<Vec<rpcgen::SyncMethod<Studio, RpcError>>> = std::sync::OnceLock::new();
    T.get_or_init(rpc_methods)
}

/// The methods, described
pub fn infos() -> Vec<rpcgen::MethodInfo> {
    rpc_methods().into_iter().map(|m| m.info).collect()
}

/// The TypeScript the studio serves at /api/types.ts and wfe/video imports: the methods' types and client, and the
/// envelope a call answers in
pub fn typescript() -> String {
    let mut ts = rpcgen::typescript("nextcuda video studio (POST /rpc/<method>; the answer in {ok, result | error})", &infos())
        .expect("the studio's methods describe consistent types");
    ts.push_str(ENVELOPE_TS);
    ts
}

const ENVELOPE_TS: &str = r#"
/** What a call answers: its result, or the error as data (code: invalid_request, invalid_param, not_found, busy, not_running, unknown_method, internal). */
export type Envelope<T> = { ok: true; result: T } | { ok: false; error: { code: string; message: string } };

/** A call of the studio's at `base` (its own port, or under the front door's /open/<id>/): the result, or an Error of the envelope's words. */
export function studio(base: string) {
  return client(async (method: string, body: unknown) => {
    const r = await fetch(`${base}/rpc/${method}`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body ?? {}) });
    const e = (await r.json()) as Envelope<unknown>;
    if (!e.ok) throw new Error(`${e.error.code}: ${e.error.message}`);
    return e.result;
  });
}
"#;
