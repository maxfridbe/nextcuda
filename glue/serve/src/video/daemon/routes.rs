//! The daemon's routes (`/engine/...` on its Unix socket; the studio passes them on): its table (rpcgen `#[route]`),
//! what it dispatches by and what docs/api/video-engine.openapi.yml lists. A job's spec and an engine's own report
//! are the engine's open objects: typed by the keys the studio and the front door read, the rest kept as it is.

use std::os::unix::net::UnixStream;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use rpcgen::{route, routes, BadPayload, Checked, RouteError, RouteReply, Rpc, SyncRoute};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use super::{gpu_busy, now, rss_gib, Daemon, JobRec};
use crate::http;

/// A route's failure: the status and the body (`{"error"}`).
pub type Fail = (u16, Value);

/// What a route's handler gets besides the daemon
pub struct Cx {
    pub s: UnixStream,
    pub req: http::Request,
}

fn fail(status: u16, m: impl Into<String>) -> Fail {
    (status, json!({"error": m.into()}))
}

/// Runs the route that serves the request and writes its answer; false when none does
pub fn serve(d: &Daemon, cx: &mut Cx) -> bool {
    let Some((r, params)) = rpcgen::find_route(table(), &cx.req.method, &cx.req.path) else { return false };
    let query = cx.req.target.split_once('?').map(|q| q.1.to_string()).unwrap_or_default();
    let body = if r.info.body == rpcgen::Body::Json { std::mem::take(&mut cx.req.body) } else { Vec::new() };
    let (status, v) = match r.call(d, cx, params, &query, &body) {
        Ok(RouteReply::Json(v)) => (200, v),
        Ok(RouteReply::Sent) => return true,
        Err(RouteError::Payload(BadPayload(m))) => (400, json!({"error": m})),
        Err(RouteError::Answer(m)) => (500, json!({"error": m})),
        Err(RouteError::Handler(f)) => f,
    };
    http::respond(&cx.s, status, &v);
    true
}

/// A denoiser the daemon has
#[derive(Serialize, Deserialize, Rpc)]
pub struct EngineFile {
    /// INT8, or a GGUF form's name
    pub name: String,
    pub path: String,
    pub gib: f64,
    pub default: bool,
}

/// A slot's worker process
#[derive(Serialize, Deserialize, Rpc)]
pub struct Worker {
    pub pid: i64,
    pub rss_gib: Option<f64>,
}

/// Where a job is
#[derive(Serialize, Deserialize, Rpc)]
pub enum JobState {
    #[serde(rename = "queued")]
    Queued,
    #[serde(rename = "running")]
    Running,
    #[serde(rename = "done")]
    Done,
    #[serde(rename = "failed")]
    Failed,
    #[serde(rename = "cancelled")]
    Cancelled,
}

/// Steps done of the job's total, as it last reported
#[derive(Serialize, Deserialize, Rpc)]
pub struct JobProgress {
    pub done: i64,
    pub total: i64,
}

/// A job, in short
#[derive(Serialize, Deserialize, Rpc)]
pub struct JobSummary {
    pub id: i64,
    /// the spec's kind: bench-blocks, check-block, denoise, decode, encode, generate
    pub kind: Option<String>,
    pub state: JobState,
    /// unix seconds
    pub created: f64,
    pub started: Option<f64>,
    pub finished: Option<f64>,
    pub error: Option<String>,
    pub progress: Option<JobProgress>,
    /// the GPU it runs or ran on (or the one its spec asked for)
    pub gpu: Option<i64>,
    /// its log's last line
    pub last: Option<String>,
    /// running: seconds since it started
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub elapsed_seconds: Option<f64>,
}

/// A job in full: its summary, the spec it was given, its log and its result
#[derive(Serialize, Deserialize, Rpc)]
pub struct Job {
    #[serde(flatten)]
    pub summary: JobSummary,
    pub spec: Map<String, Value>,
    pub log: Vec<String>,
    /// what the engine made (the kind's own report; null until done)
    pub result: Value,
}

/// A GPU the daemon serves, and the engine on it
#[derive(Serialize, Deserialize, Rpc)]
pub struct Slot {
    pub gpu: i64,
    /// the cards a spanning slot's worker takes (empty: gpu alone)
    pub span: Vec<i64>,
    pub name: String,
    pub pci: String,
    pub mem_gib: f64,
    /// shared with the language models
    pub shared: bool,
    /// unloaded, starting..., waiting for the GPU..., loading, loaded, unloading
    pub engine: String,
    /// the engine's own report of what it loaded (model ...)
    pub info: Value,
    /// the denoiser loaded, by name
    pub denoiser: Option<String>,
    pub worker: Option<Worker>,
    pub busy_pct: Option<f64>,
    /// the engine's GPU memory, its cap and the card's free memory (its last report)
    pub engine_gib: Value,
    pub cap_gib: Value,
    pub card_free_gib: Value,
    pub running: Option<JobSummary>,
    pub idle_seconds: i64,
    /// a loaded engine with nothing to do: seconds until it unloads
    pub unload_in_seconds: Option<i64>,
}

/// The daemon: its denoisers, its GPUs and what runs on them, the queue
#[derive(Serialize, Deserialize, Rpc)]
pub struct Status {
    pub version: String,
    /// the default denoiser's checkpoint
    pub model: String,
    pub engines: Vec<EngineFile>,
    pub gpus: Vec<Slot>,
    /// the waiting jobs' ids, in order
    pub queued: Vec<i64>,
    pub unload_after_idle_seconds: Option<i64>,
}

/// A GPU found on the machine at start, and whether the daemon serves it
#[derive(Serialize, Deserialize, Rpc)]
pub struct FoundGpu {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<i64>,
    pub served: bool,
    /// what the listing said of it (name, pci, memory ...)
    #[serde(flatten)]
    pub rest: Map<String, Value>,
}

/// A job by its id
#[derive(Serialize, Deserialize, Rpc)]
pub struct JobRef {
    pub id: String,
}

/// A job to run: its kind and the kind's own keys
#[derive(Serialize, Deserialize, Rpc)]
pub struct JobSpec {
    /// bench-blocks, check-block, denoise, decode, encode, generate
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<Value>,
    /// a denoiser the daemon has (default: its model)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine: Option<Value>,
    /// a GPU it serves (default: the first free)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu: Option<Value>,
    /// the kind's own keys
    #[serde(flatten)]
    pub rest: Map<String, Value>,
}

/// A job queued
#[derive(Serialize, Deserialize, Rpc)]
pub struct Queued {
    pub id: i64,
}

/// A job told to stop
#[derive(Serialize, Deserialize, Rpc)]
pub struct Cancelling {
    pub id: i64,
    /// cancelled, "cancelling: ..." (it stops at the next block boundary), or the state it had already
    pub state: String,
}

/// A job forgotten
#[derive(Serialize, Deserialize, Rpc)]
pub struct Removed {
    pub id: i64,
    pub removed: bool,
}

/// Which GPU's engine to unload
#[derive(Serialize, Deserialize, Rpc)]
pub struct UnloadReq {
    /// none: every one
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu: Option<Value>,
}

/// An unload asked of one GPU
#[derive(Serialize, Deserialize, Rpc)]
pub struct Unloading {
    pub gpu: i64,
    /// now, or after the running job
    pub unload: String,
}

/// The unloads asked
#[derive(Serialize, Deserialize, Rpc)]
pub struct Unloads {
    pub unload: Vec<Unloading>,
}

/// The daemon stopping
#[derive(Serialize, Deserialize, Rpc)]
pub struct ShuttingDown {
    pub shutdown: String,
}

/// The daemon: its denoisers, GPUs, engines and queue
#[route(get = "/engine/status", tags = "engine")]
fn status(d: &Daemon, _cx: &mut Cx) -> Result<Checked<Status>, Fail> {
    let mut s = d.s.lock().unwrap();
    let idle_limit = d.opts.idle;
    let mut gpus = Vec::new();
    for k in 0..s.slots.len() {
        let running = s.slots[k].running.and_then(|r| s.jobs.get(&r)).map(|j| {
            let mut v = j.summary();
            v["elapsed_seconds"] = json!(j.started.map(|t| now() - t));
            v
        });
        let sl = &mut s.slots[k];
        let idle = sl.last_active.elapsed().as_secs();
        let busy = gpu_busy(&sl.pci, &mut sl.busy_sample);
        let unload_in = match (idle_limit, sl.worker_pid, sl.running) {
            (Some(limit), Some(_), None) => Some(limit.as_secs().saturating_sub(idle)),
            _ => None,
        };
        gpus.push(json!({
            "gpu": sl.gpu, "span": sl.span, "name": sl.name, "pci": sl.pci, "mem_gib": sl.mem_gib, "shared": sl.shared,
            "engine": sl.engine, "info": sl.info, "denoiser": d.opts.engine_of_path(sl.info.get("model")),
            "worker": sl.worker_pid.map(|pid| json!({"pid": pid, "rss_gib": rss_gib(pid)})),
            "busy_pct": busy, "engine_gib": sl.stats["gib_in_use"], "cap_gib": sl.stats["gib_cap"],
            "card_free_gib": sl.stats["gib_free_card"], "running": running,
            "idle_seconds": if sl.running.is_some() { 0 } else { idle }, "unload_in_seconds": unload_in}));
    }
    Ok(Checked::new(json!({"version": crate::VERSION, "model": d.opts.model, "engines": d.opts.engine_names().iter().filter_map(|n| d.opts.engine_path(Some(n))).map(|(n, p)| json!({
                     "name": n, "path": p, "gib": std::fs::metadata(&p).map_or(0.0, |m| m.len() as f64 / (1u64 << 30) as f64),
                     "default": n == d.opts.model_name})).collect::<Vec<_>>(), "gpus": gpus,
                 "queued": s.queue.iter().collect::<Vec<_>>(),
                 "unload_after_idle_seconds": d.opts.idle.map(|d| d.as_secs())})))
}

/// The GPUs found at start, each marked served or not
#[route(get = "/engine/gpus", tags = "engine")]
fn gpus(d: &Daemon, _cx: &mut Cx) -> Result<Checked<Vec<FoundGpu>>, Fail> {
    let s = d.s.lock().unwrap();
    let served: Vec<usize> = s.slots.iter().map(|sl| sl.gpu).collect();
    let list: Vec<Value> = d
        .found
        .iter()
        .map(|g| {
            let mut g = g.clone();
            g["served"] = json!(g["index"].as_u64().is_some_and(|i| served.contains(&(i as usize))));
            g
        })
        .collect();
    Ok(Checked::new(json!(list)))
}

/// Every job, newest first
#[route(get = "/engine/jobs", tags = "engine")]
fn jobs(d: &Daemon, _cx: &mut Cx) -> Result<Checked<Vec<JobSummary>>, Fail> {
    let s = d.s.lock().unwrap();
    Ok(Checked::new(json!(s.jobs.values().rev().map(JobRec::summary).collect::<Vec<_>>())))
}

/// A job in full
#[route(get = "/engine/jobs/{id}", tags = "engine")]
fn job(d: &Daemon, _cx: &mut Cx, req: JobRef) -> Result<Checked<Job>, Fail> {
    let s = d.s.lock().unwrap();
    match req.id.parse::<u64>().ok().and_then(|i| s.jobs.get(&i)) {
        Some(j) => Ok(Checked::new(j.full())),
        None => Err(fail(404, format!("no job {}", req.id))),
    }
}

/// Queues a job: a kind (bench-blocks, check-block, denoise, decode, encode, generate) and its keys, on an engine and a GPU the daemon has
#[route(post = "/engine/jobs", tags = "engine")]
fn submit(d: &Daemon, _cx: &mut Cx, req: JobSpec) -> Result<Queued, Fail> {
    let body = serde_json::to_value(&req).map_err(|e| fail(500, e.to_string()))?;
    let mut s = d.s.lock().unwrap();
    if s.shutdown {
        return Err(fail(409, "the daemon is shutting down"));
    }
    if body.get("kind").and_then(|k| k.as_str()).is_none() {
        return Err(fail(400, "a job needs a \"kind\" (bench-blocks, check-block, denoise, decode, encode, generate)"));
    }
    if let Some(e) = body.get("engine") {
        if !e.as_str().is_some_and(|e| d.opts.engine_path(Some(e)).is_some()) {
            return Err(fail(400, format!("\"engine\": {e} is not one this daemon has ({:?})", d.opts.engine_names())));
        }
    }
    if let Some(g) = body.get("gpu") {
        let served: Vec<usize> = s.slots.iter().map(|sl| sl.gpu).collect();
        if !g.as_u64().is_some_and(|g| served.contains(&(g as usize))) {
            return Err(fail(400, format!("\"gpu\": {g} is not one this daemon serves ({served:?})")));
        }
    }
    let id = s.next;
    s.next += 1;
    s.jobs.insert(id, JobRec { id, spec: body, state: "queued", log: Vec::new(), result: None, error: None,
                               cancel: Arc::new(AtomicBool::new(false)), created: now(), started: None,
                               finished: None, progress: None, gpu: None });
    s.queue.push_back(id);
    d.cv.notify_all();
    Ok(Queued { id: id as i64 })
}

/// Stops a job: a queued one is dropped, a running one stops at its next block boundary
#[route(post = "/engine/jobs/{id}/cancel", tags = "engine", body = "none")]
fn cancel(d: &Daemon, _cx: &mut Cx, req: JobRef) -> Result<Cancelling, Fail> {
    let Some(i) = req.id.parse::<u64>().ok() else { return Err(fail(404, format!("no job {}", req.id))) };
    let mut s = d.s.lock().unwrap();
    s.queue.retain(|q| *q != i);
    let id = i as i64;
    match s.jobs.get_mut(&i) {
        Some(j) if j.state == "queued" => {
            j.state = "cancelled";
            j.finished = Some(now());
            Ok(Cancelling { id, state: "cancelled".into() })
        }
        Some(j) if j.state == "running" => {
            j.cancel.store(true, Ordering::Relaxed);
            Ok(Cancelling { id, state: "cancelling: it stops at the next block boundary".into() })
        }
        Some(j) => Ok(Cancelling { id, state: j.state.into() }),
        None => Err(fail(404, format!("no job {i}"))),
    }
}

/// Forgets a job that is not running
#[route(post = "/engine/jobs/{id}/remove", tags = "engine", body = "none")]
fn remove(d: &Daemon, _cx: &mut Cx, req: JobRef) -> Result<Removed, Fail> {
    let Some(i) = req.id.parse::<u64>().ok() else { return Err(fail(404, format!("no job {}", req.id))) };
    let mut s = d.s.lock().unwrap();
    match s.jobs.get(&i).map(|j| j.state) {
        None => Err(fail(404, format!("no job {i}"))),
        Some("running") => Err(fail(409, format!("job {i} is running: stop it first"))),
        Some(_) => {
            s.queue.retain(|q| *q != i);
            s.jobs.remove(&i);
            Ok(Removed { id: i as i64, removed: true })
        }
    }
}

/// How long to keep the workers loaded
#[derive(Serialize, Deserialize, Rpc)]
pub struct KeepReq {
    /// seconds from now: no worker is unloaded for idleness before then (a later, shorter ask never shortens it)
    pub seconds: f64,
}

/// The time workers are kept until
#[derive(Serialize, Deserialize, Rpc)]
pub struct Kept {
    /// unix seconds
    pub keep_until: f64,
}

/// Keeps the loaded workers from unloading for idleness for a while (the front door asks this of a pinned or held model, and stops asking when the pin or hold ends)
#[route(post = "/engine/keep", tags = "engine")]
fn keep(d: &Daemon, _cx: &mut Cx, req: KeepReq) -> Result<Kept, Fail> {
    let mut k = d.keep_until.lock().unwrap();
    *k = k.max(now() + req.seconds.clamp(0.0, 86400.0));
    Ok(Kept { keep_until: *k })
}

/// Unloads a GPU's engine (every GPU's without one named): now, or after its running job
#[route(post = "/engine/unload", tags = "engine")]
fn unload(d: &Daemon, _cx: &mut Cx, req: UnloadReq) -> Result<Unloads, Fail> {
    let want = req.gpu.as_ref().and_then(|g| g.as_u64()).map(|g| g as usize);
    let mut s = d.s.lock().unwrap();
    let mut hit = Vec::new();
    for sl in s.slots.iter_mut().filter(|sl| want.is_none_or(|g| g == sl.gpu)) {
        sl.unload_requested = true;
        hit.push(Unloading { gpu: sl.gpu as i64, unload: if sl.running.is_some() { "after the running job" } else { "now" }.into() });
    }
    d.cv.notify_all();
    if hit.is_empty() {
        Err(fail(400, "no such GPU served"))
    } else {
        Ok(Unloads { unload: hit })
    }
}

/// Stops the daemon: the queue cancelled, running jobs stop at their next block boundary, then the engines unload
#[route(post = "/engine/shutdown", tags = "engine")]
fn shutdown(d: &Daemon, _cx: &mut Cx) -> Result<ShuttingDown, Fail> {
    let mut s = d.s.lock().unwrap();
    s.shutdown = true;
    for id in std::mem::take(&mut s.queue) {
        if let Some(j) = s.jobs.get_mut(&id) {
            j.state = "cancelled";
        }
    }
    let running: Vec<u64> = s.slots.iter().filter_map(|sl| sl.running).collect();
    for r in running {
        s.jobs[&r].cancel.store(true, Ordering::Relaxed);
    }
    d.cv.notify_all();
    Ok(ShuttingDown { shutdown: "running jobs stop at their next block boundary, then the engines unload".into() })
}

routes![sync Daemon, Cx, Fail; status, gpus, jobs, job, submit, cancel, remove, keep, unload, shutdown];

/// The table, made once
pub fn table() -> &'static [SyncRoute<Daemon, Cx, Fail>] {
    static T: std::sync::OnceLock<Vec<SyncRoute<Daemon, Cx, Fail>>> = std::sync::OnceLock::new();
    T.get_or_init(routes)
}

/// The routes, described (the document)
pub fn infos() -> Vec<rpcgen::RouteInfo> {
    routes().into_iter().map(|r| r.info).collect()
}

