//! The chat server's routes (rpcgen `#[route]`): what it serves and what docs/api/llm.openapi.yml lists, from this
//! table. `/server/...` are its control routes, answered on its socket only (`handle` refuses them on the port);
//! the answers are typed here and each is held to its type as it is sent (`Checked`).

use std::sync::atomic::Ordering;

use rpcgen::{route, routes, Checked, Rpc, Sent};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use super::{now, Api, Server, STOP};
use crate::page::{Cx, Fail, Health};

/// The disk tier of the prompt cache
#[derive(Serialize, Deserialize, Rpc)]
pub struct DiskCache {
    pub entries: i64,
    pub bytes: i64,
    pub budget: i64,
}

/// The prompt cache: prompts read before, kept on the card (and on disk) to be reused
#[derive(Serialize, Deserialize, Rpc)]
pub struct CacheFigures {
    pub entries: i64,
    pub bytes: i64,
    pub budget: i64,
    pub evictions: i64,
    /// the tokens the sessions running hold
    pub live_tokens: i64,
    /// each running session's tokens
    pub sessions: Vec<i64>,
    pub disk: Option<DiskCache>,
}

/// Whether it is busy, and its prompt cache (none while a prompt is being read with it: a status never waits)
#[derive(Serialize, Deserialize, Rpc)]
pub struct Status {
    pub busy: bool,
    pub prompt_cache: Option<CacheFigures>,
}

/// A card the model runs on
#[derive(Serialize, Deserialize, Rpc)]
pub struct LlmGpu {
    pub index: i64,
    pub name: String,
    pub pci: Option<String>,
    /// bytes
    pub total: i64,
    pub free: Option<i64>,
    /// the layers it runs [first, end)
    pub layers: Vec<i64>,
    pub expert_slots: i64,
    pub host_slots: i64,
    pub watts: Option<f64>,
    pub temp_c: Option<f64>,
    pub vram_temp_c: Option<f64>,
}

/// A request's sampling as asked
#[derive(Serialize, Deserialize, Rpc)]
pub struct RequestSettings {
    pub effort: String,
    pub temperature: f64,
    pub top_p: f64,
    pub stream: bool,
    pub max_tokens: i64,
}

/// What an answer (or its thinking) came to: its length and its start
#[derive(Serialize, Deserialize, Rpc)]
pub struct Said {
    pub chars: i64,
    pub preview: String,
}

/// Where a request is
#[derive(Serialize, Deserialize, Rpc)]
#[serde(rename_all = "lowercase")]
pub enum RequestState {
    Waiting,
    Reading,
    Generating,
    Done,
    Failed,
}

/// A request, running or done: what was asked, how far it is, what it came to
#[derive(Serialize, Deserialize, Rpc)]
pub struct RequestRow {
    pub id: i64,
    /// web (the page), socket or tcp
    pub via: String,
    pub state: RequestState,
    /// unix seconds
    pub started: i64,
    pub prompt_tokens: i64,
    pub max_tokens: i64,
    pub generated: i64,
    pub model: String,
    /// the route it came by
    pub api: String,
    pub settings: RequestSettings,
    pub messages: i64,
    pub prompt_chars: i64,
    /// the last user message's start
    pub last_user: String,
    /// reading: the tokens read so far, at what rate
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub read: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub read_tok_s: Option<f64>,
    /// the prompt's tokens found in the cache, and where (the card, the disk)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reused: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub read_seconds: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tok_s: Option<f64>,
    /// done: why it stopped (stop, length, tool_calls)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generate_seconds: Option<f64>,
    /// the draft model's tokens [accepted, drafted]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub drafts: Option<Vec<i64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoints_saved: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ended: Option<i64>,
    /// the cards' energy over it, joules (other requests at the same time draw on the same counters)
    #[serde(default)]
    pub energy_j: Option<f64>,
    #[serde(default)]
    pub avg_watts: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub answer: Option<Said>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<Said>,
    /// failed: why
    #[serde(default)]
    pub error: Option<String>,
}

/// The server in full: the model, its cards, the requests running, the cache
#[derive(Serialize, Deserialize, Rpc)]
pub struct ServerStatus {
    pub model: String,
    pub version: String,
    pub uptime_seconds: f64,
    /// the longest context a request may have
    pub context: i64,
    /// each slot's context
    pub contexts: Vec<i64>,
    /// a draft model (multi-token prediction) is loaded
    pub mtp: bool,
    /// requests made at once
    pub parallel: i64,
    pub gpus: Vec<LlmGpu>,
    pub busy: bool,
    /// the first running (as `active[0]`)
    pub running: Option<RequestRow>,
    pub active: Vec<RequestRow>,
    pub waiting: i64,
    /// requests served since it started
    pub served: i64,
    pub prompt_cache: Option<CacheFigures>,
    /// shutting down once the requests running end
    pub stopping: bool,
}

/// The requests running and the last ones done
#[derive(Serialize, Deserialize, Rpc)]
pub struct Requests {
    pub running: Option<RequestRow>,
    pub active: Vec<RequestRow>,
    /// newest first
    pub done: Vec<RequestRow>,
    /// how many done it keeps
    pub keep: i64,
}

/// A request by its id
#[derive(Serialize, Deserialize, Rpc)]
pub struct RequestId {
    /// its id (a leading # is allowed)
    pub id: String,
}

/// A prompt kept in the cache
#[derive(Serialize, Deserialize, Rpc)]
pub struct CacheEntry {
    pub tokens: i64,
    pub bytes: i64,
    /// kept on disk (not on the card)
    pub disk: bool,
}

/// The prompt cache's entries
#[derive(Serialize, Deserialize, Rpc)]
pub struct CacheList {
    pub entries: Vec<CacheEntry>,
    pub bytes: i64,
    pub budget: i64,
    pub evictions: i64,
    pub disk: Option<DiskCache>,
}

/// The cache emptied
#[derive(Serialize, Deserialize, Rpc)]
pub struct Cleared {
    /// entries dropped
    pub dropped: i64,
}

/// A shutdown asked for
#[derive(Serialize, Deserialize, Rpc)]
pub struct Stopping {
    pub stopping: bool,
    /// a request is running: the server ends after it
    pub running: bool,
}

/// The model's state in OpenAI's model list (llama.cpp's)
#[derive(Serialize, Deserialize, Rpc)]
pub struct ModelState {
    /// loaded
    pub value: String,
}

/// The model served, in OpenAI's list
#[derive(Serialize, Deserialize, Rpc)]
pub struct LlmModel {
    pub id: String,
    pub object: String,
    pub owned_by: String,
    pub created: i64,
    pub status: ModelState,
    /// its context
    pub context: i64,
}

/// OpenAI's list of models
#[derive(Serialize, Deserialize, Rpc)]
pub struct LlmModels {
    pub object: String,
    pub data: Vec<LlmModel>,
}

/// A chat: OpenAI's chat completions body (and /api/chat's: effort, thinking)
#[derive(Serialize, Deserialize, Rpc)]
pub struct ChatBody {
    /// [{role, content}]: content a string or OpenAI's parts (text, image_url); an assistant's reasoning_content (/api/chat: thinking) and tool_calls
    pub messages: Vec<Map<String, Value>>,
    /// the model's id (any: the one served)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// as server-sent events (default false; /api/chat: true, as lines)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(default = 1.0)]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(default = 0.95)]
    pub top_p: Option<f64>,
    /// OpenAI tool definitions
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Value>>,
    /// none: no tools
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    /// none, low, medium, high or max (/api/chat: effort)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    /// llama.cpp's: enable_thinking (false answers without thinking), reasoning_effort
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_template_kwargs: Option<Map<String, Value>>,
    /// what else the caller sent (seed, logprobs, logprob_chain ...): read as it came
    #[serde(flatten)]
    pub rest: Map<String, Value>,
}

fn checked<T: serde::de::DeserializeOwned + Serialize>(v: Value) -> Result<Checked<T>, Fail> {
    Ok(Checked::new(v))
}

/// Whether the server answers
#[route(get = "/health", tags = "server")]
fn health(_s: &Server, _cx: &mut Cx) -> Result<Health, Fail> {
    Ok(Health { status: "ok".into() })
}

/// Whether it is busy, and its prompt cache's figures
#[route(get = "/status", tags = "server")]
fn status(s: &Server, _cx: &mut Cx) -> Result<Checked<Status>, Fail> {
    let st = s.status_json();
    checked(json!({"busy": st["busy"], "prompt_cache": st["prompt_cache"]}))
}

/// The server in full: the model, its cards, the requests running, the cache (its socket only)
#[route(get = "/server/status", tags = "control")]
fn server_status(s: &Server, _cx: &mut Cx) -> Result<Checked<ServerStatus>, Fail> {
    checked(s.status_json())
}

/// The requests running and the last ones done (its socket only)
#[route(get = "/server/requests", tags = "control")]
fn requests(s: &Server, _cx: &mut Cx) -> Result<Checked<Requests>, Fail> {
    let done: Vec<Value> = s.done.lock().unwrap().iter().rev().cloned().collect();
    let running: Vec<Value> = s.running.lock().unwrap().values().cloned().collect();
    checked(json!({"running": running.first().cloned(), "active": running, "done": done, "keep": s.keep}))
}

/// A request, running or among the last done (its socket only)
#[route(get = "/server/requests/{id}", tags = "control")]
fn request(s: &Server, _cx: &mut Cx, req: RequestId) -> Result<Checked<RequestRow>, Fail> {
    let id: Option<u64> = req.id.trim_start_matches('#').parse().ok();
    let running = id.and_then(|i| s.running.lock().unwrap().get(&i).cloned());
    match running.or_else(|| s.done.lock().unwrap().iter().find(|r| r["id"].as_u64() == id).cloned()) {
        Some(r) => checked(r),
        None => Err((404, format!("no request {} (the server keeps the last {})", req.id, s.keep))),
    }
}

/// The prompt cache's entries (its socket only); 409 while a prompt is being read with it
#[route(get = "/server/cache", tags = "control")]
fn cache(s: &Server, _cx: &mut Cx) -> Result<CacheList, Fail> {
    let c = s.cache.try_lock().map_err(|_| (409, "a prompt is being read; ask again when it is done".to_string()))?;
    let entries = c.list().iter().map(|(t, b, _, disk)| CacheEntry { tokens: *t as i64, bytes: *b as i64, disk: *disk }).collect();
    Ok(CacheList { entries, bytes: c.bytes() as i64, budget: c.budget() as i64, evictions: c.evictions as i64,
                   disk: c.disk().map(|(n, b, g)| DiskCache { entries: n as i64, bytes: b as i64, budget: g as i64 }) })
}

/// Empties the prompt cache (its socket only); 409 while a prompt is being read with it
#[route(post = "/server/cache/clear", tags = "control")]
fn cache_clear(s: &Server, _cx: &mut Cx) -> Result<Cleared, Fail> {
    let mut c = s.cache.try_lock().map_err(|_| (409, "a prompt is being read; ask again when it is done".to_string()))?;
    Ok(Cleared { dropped: c.clear() as i64 })
}

/// Ends the server once the requests running end (its socket only)
#[route(post = "/server/shutdown", tags = "control")]
fn shutdown(s: &Server, _cx: &mut Cx) -> Result<Stopping, Fail> {
    STOP.store(true, Ordering::SeqCst);
    Ok(Stopping { stopping: true, running: s.inflight.load(Ordering::SeqCst) > 0 })
}

/// The model served, as OpenAI lists models
#[route(get = "/v1/models", alias = "/models", tags = "openai")]
fn models(s: &Server, _cx: &mut Cx) -> Result<LlmModels, Fail> {
    Ok(LlmModels { object: "list".into(), data: vec![LlmModel { id: s.name.clone(), object: "model".into(), owned_by: "nextcuda".into(),
        created: now() as i64, status: ModelState { value: "loaded".into() }, context: s.max_ctx as i64 }] })
}

/// A chat (OpenAI's chat completions: stream for server-sent events; tools); the answer is OpenAI's chat.completion, or its chunks
#[route(post = "/v1/chat/completions", alias = "/chat/completions", tags = "openai", produces = "application/json, text/event-stream")]
fn chat(s: &Server, cx: &mut Cx, req: ChatBody) -> Result<Sent<Value>, Fail> {
    s.chat_route(cx, req, Api::OpenAi)
}

/// A chat for the page: the answer streamed as lines of JSON (effort, thinking as /api/chat names them)
#[route(post = "/api/chat", tags = "page", produces = "application/x-ndjson")]
fn chat_lines(s: &Server, cx: &mut Cx, req: ChatBody) -> Result<Sent<Value>, Fail> {
    s.chat_route(cx, req, Api::Lines)
}

routes![sync Server, Cx, Fail; health, status, server_status, requests, request, cache, cache_clear, shutdown, models, chat, chat_lines];

/// The table, made once
pub fn table() -> &'static [rpcgen::SyncRoute<Server, Cx, Fail>] {
    static T: std::sync::OnceLock<Vec<rpcgen::SyncRoute<Server, Cx, Fail>>> = std::sync::OnceLock::new();
    T.get_or_init(routes)
}

/// The routes, described (the document)
pub fn infos() -> Vec<rpcgen::RouteInfo> {
    routes().into_iter().map(|r| r.info).collect()
}

/// An error as the chat server words one
pub(super) fn fail(m: String) -> Value {
    json!({"error": {"message": m}})
}
