//! The front door's own routes (rpcgen `#[route]`): its page's API (`/api/...`), its health and its documents - what
//! it dispatches by, what wfe/home calls (src/gen/routes.ts) and what docs/api/home.openapi.yml lists, from this
//! table. (`/rpc/...` are the methods, methods.rs; `/v1/...` the OpenAI-compatible routes, behind the token.) The
//! answers are typed here and each is held to its type as it is sent (`Checked`).

use std::collections::BTreeMap;
use std::sync::Arc;

use rpcgen::{route, routes, Checked, Rpc, Sent};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::{err, yaml, Home};
use crate::page::{Cx, Fail, Health};

/// The machine: memory, load, CPUs
#[derive(Serialize, Deserialize, Rpc)]
pub struct HostStat {
    pub mem_total_mb: i64,
    pub mem_avail_mb: i64,
    pub load1: Option<f64>,
    pub cpus: i64,
}

/// A PCIe link: its generation and lanes
#[derive(Serialize, Deserialize, Rpc)]
pub struct Link {
    pub gen: Option<i64>,
    pub width: i64,
}

/// A card's PCIe link now, the most the card does and the most its slot does
#[derive(Serialize, Deserialize, Rpc)]
pub struct Pcie {
    pub cur: Link,
    pub card_max: Link,
    pub slot_max: Link,
}

/// A process on a card
#[derive(Serialize, Deserialize, Rpc)]
pub struct Proc {
    pub pid: i64,
    pub vram_mb: i64,
    /// nextcuda, or another program (python: its script)
    pub program: String,
    /// another program's command line (its start)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub args: Option<String>,
    /// nextcuda's: the kind and command it runs (llm, image serve ...)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    /// the model's file, or its registry id where it is a registered one (model_id, title)
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default)]
    pub port: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpus: Option<Vec<String>>,
}

/// An enabled model's memory vouched for on a card (chat and video models: one of the group, `of` them)
#[derive(Serialize, Deserialize, Rpc)]
pub struct Vouched {
    pub model: String,
    pub kind: String,
    pub mb: i64,
    pub headroom_mb: i64,
    pub headroom_given: bool,
    /// the memory is what it took when it ran (else an estimate)
    pub measured: bool,
    pub state: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub of: Option<i64>,
}

/// A card: its readings, its processes, the memory vouched for
#[derive(Serialize, Deserialize, Rpc)]
pub struct CardState {
    pub index: i64,
    pub name: String,
    pub pci: String,
    /// when it was read (unix seconds)
    pub ts: f64,
    pub vram_used_mb: i64,
    pub vram_total_mb: i64,
    pub busy_pct: f64,
    pub temp_pkg: Option<f64>,
    pub temp_vram: Option<f64>,
    pub temp_vram_max: Option<f64>,
    pub temp_pcie: Option<f64>,
    pub temp_mctrl: Option<f64>,
    pub power_w: Option<f64>,
    pub pkg_power_w: Option<f64>,
    pub power_cap_w: Option<i64>,
    pub pcie: Pcie,
    pub host_load1: f64,
    pub host_cpus: i64,
    /// the energy counter since the front door started, joules
    pub energy_j: f64,
    pub fan_pct: Option<f64>,
    pub fan_rpm: Option<f64>,
    pub freq_mhz: Option<i64>,
    pub procs: Vec<Proc>,
    pub vouched: Vec<Vouched>,
    pub vouched_mb: i64,
    pub vouched_headroom_mb: i64,
    /// the models vouched for pass the card
    pub overbooked: bool,
}

/// A model's state
#[derive(Serialize, Deserialize, Rpc)]
#[serde(rename_all = "lowercase")]
pub enum ModelState {
    Disabled,
    Unloaded,
    Loading,
    Loaded,
    Busy,
    /// a video model: the daemon has it (its engine), not loaded now
    Ready,
}

/// A registered model: kind, enabled, its cards, its state, the memory vouched for
#[derive(Serialize, Deserialize, Rpc)]
pub struct ModelView {
    pub id: String,
    pub kind: String,
    pub title: Option<String>,
    pub arch: Option<String>,
    pub enabled: bool,
    pub gpus: Vec<i64>,
    pub state: ModelState,
    /// a chat model that spans every card
    pub spans: bool,
    pub capabilities: Vec<String>,
    /// the methods it serves (image.generate ...)
    pub methods: Vec<String>,
    /// card index -> MiB
    pub vouch_mb: BTreeMap<String, i64>,
    pub measured: bool,
    pub headroom_mb: BTreeMap<String, i64>,
    pub headroom_given: bool,
    /// the port its page is on
    pub page: i64,
}

/// A front end or helper beside the models
#[derive(Serialize, Deserialize, Rpc)]
pub struct Service {
    pub id: String,
    pub title: String,
    pub what: String,
    pub port: i64,
    pub up: bool,
    #[serde(default)]
    pub detail: Option<String>,
    /// its page, on its own port
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub link: Option<String>,
    /// its page through this door
    #[serde(default)]
    pub href: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api: Option<String>,
}

/// Something the page ran (this program with arguments): its output's tail
#[derive(Serialize, Deserialize, Rpc)]
pub struct RunView {
    pub id: i64,
    pub label: String,
    pub started: f64,
    /// its exit code; null: running
    pub rc: Option<i64>,
    pub out: String,
}

/// The ports the page links
#[derive(Serialize, Deserialize, Rpc)]
pub struct Ports {
    pub video: i64,
    pub chat_ui: i64,
}

/// Maintenance mode's notice, as callers see it
#[derive(Serialize, Deserialize, Rpc)]
pub struct Maintenance {
    pub on: bool,
    pub message: String,
    /// unix seconds
    pub since: f64,
    /// about when it ends; null: unknown
    pub until: Option<f64>,
    pub retry_after_s: f64,
    /// the words a refused call gets
    pub notice: String,
    /// the key that lets a call through (the page's own copy)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bypass: Option<String>,
}

/// A drive the models are (or may be) on
#[derive(Serialize, Deserialize, Rpc)]
pub struct DriveView {
    pub id: String,
    pub disk: String,
    pub model: String,
    pub link: Option<String>,
    pub link_gb_s: Option<f64>,
    pub device: String,
    pub fstype: String,
    pub mount: String,
    pub folder: String,
    pub size: i64,
    pub free: i64,
    /// what is kept free
    pub margin: i64,
    pub models_bytes: i64,
    pub models: Vec<String>,
    /// its read speed as measured (system.drive.measure); null: not measured
    pub measured: Option<Value>,
    pub measuring: bool,
    pub rate_gb_s: Option<f64>,
}

/// Whether a model's files may move to a drive: what they need there, or why not
#[derive(Serialize, Deserialize, Rpc)]
pub struct MoveCheck {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub need: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub why: Option<String>,
}

/// A model's files on the drives
#[derive(Serialize, Deserialize, Rpc)]
pub struct ModelFiles {
    pub bytes: i64,
    pub files: i64,
    pub missing: i64,
    /// drive -> bytes on it
    pub drives: BTreeMap<String, i64>,
    /// it streams weights from its drive while it runs
    pub streams: bool,
    pub moving: bool,
    /// drive -> whether it may move there
    #[serde(rename = "move")]
    pub moves: BTreeMap<String, MoveCheck>,
}

/// The drives, the models' files on them, a move under way
#[derive(Serialize, Deserialize, Rpc)]
pub struct DrivesView {
    pub drives: Vec<DriveView>,
    pub models: BTreeMap<String, ModelFiles>,
    #[serde(rename = "move")]
    pub moving: Option<Value>,
    /// the drive being measured
    pub measuring: Option<String>,
    pub folders: Vec<String>,
}

/// A model download's state
#[derive(Serialize, Deserialize, Rpc)]
#[serde(rename_all = "lowercase")]
pub enum PullState {
    Queued,
    Downloading,
    Verifying,
    Done,
    Failed,
    Cancelled,
}

/// A model's download
#[derive(Serialize, Deserialize, Rpc)]
pub struct PullView {
    pub model: String,
    pub kind: Option<String>,
    pub title: Option<String>,
    pub drive: String,
    pub folder: String,
    pub state: PullState,
    pub error: Option<String>,
    pub bytes_total: i64,
    pub bytes_done: i64,
    pub pct: f64,
    /// bytes a second
    pub speed: f64,
    pub eta_s: Option<f64>,
    pub started: f64,
    pub ended: Option<f64>,
    /// the run its output is in
    pub run: i64,
}

/// A model a card offers that is not here yet
#[derive(Serialize, Deserialize, Rpc)]
pub struct Available {
    pub id: String,
    pub kind: String,
    pub title: Option<String>,
    pub about: Option<String>,
    /// the cards it is made for, as its model card says
    pub gpus: Option<String>,
    pub bytes: i64,
    pub downloaded: bool,
    /// where it would go
    pub drive: String,
    pub folder: String,
    pub fits: bool,
    pub free: i64,
    pub pull: Option<PullView>,
    pub runs_here: bool,
    pub why_not: Option<String>,
}

/// The box at a glance (the page polls it)
#[derive(Serialize, Deserialize, Rpc)]
pub struct State {
    pub ts: f64,
    pub host: HostStat,
    pub cards: Vec<CardState>,
    pub models: Vec<ModelView>,
    pub services: Vec<Service>,
    /// the last runs, newest first
    pub runs: Vec<RunView>,
    pub idle_minutes: i64,
    pub ports: Ports,
    pub maintenance: Option<Maintenance>,
    pub drives: DrivesView,
    pub available: Vec<Available>,
    pub pulls: Vec<PullView>,
    /// a request for a model not downloaded fetches it (NS_AUTO_PULL)
    pub auto_pull: bool,
}

/// A setting this program reads
#[derive(Serialize, Deserialize, Rpc)]
pub struct SettingRow {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<String>,
    pub what: String,
    /// the programs that read it
    pub readers: Vec<String>,
    /// its value in the file; null: not there
    pub file: Option<String>,
    /// its value in this service's environment (which wins over the file)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub env: Option<String>,
}

/// A family of settings (a prefix)
#[derive(Serialize, Deserialize, Rpc)]
pub struct SettingPrefix {
    pub prefix: String,
    pub what: String,
    pub readers: Vec<String>,
}

/// The settings file and what each setting is
#[derive(Serialize, Deserialize, Rpc)]
pub struct Config {
    pub path: String,
    pub text: String,
    pub settings: Vec<SettingRow>,
    /// the file's lines that are no setting listed
    pub other: Vec<SettingRow>,
    pub prefixes: Vec<SettingPrefix>,
}

/// The API tab: the token and the document with it filled in
#[derive(Serialize, Deserialize, Rpc)]
pub struct Spec {
    /// the bearer token (this page's own copy)
    pub token: String,
    /// the OpenAPI document of what the enabled kinds offer
    pub spec: Value,
}

/// A run by its id
#[derive(Serialize, Deserialize, Rpc)]
pub struct RunId {
    pub id: i64,
}

/// What a page button does
#[derive(Serialize, Deserialize, Rpc)]
#[serde(rename_all = "snake_case")]
pub enum ActionDo {
    Enable,
    Disable,
    Gpus,
    Load,
    Unload,
    Config,
    Restart,
    Maintenance,
    Measure,
    Move,
    MoveCancel,
    Pull,
    PullCancel,
    AutoPull,
}

/// A page button: what it does and to what
#[derive(Serialize, Deserialize, Rpc)]
pub struct Action {
    #[serde(rename = "do")]
    pub act: ActionDo,
    /// the model (enable, disable, gpus, load, unload, move, pull, pull_cancel)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// enable, gpus: the cards ([] or none: any)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpus: Option<Value>,
    /// disable: even while it runs
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub force: Option<bool>,
    /// restart: serve, switch, studio, video, or a kind
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service: Option<String>,
    /// measure, move: the drive
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub drive: Option<String>,
    /// auto_pull, maintenance: on or off
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub on: Option<bool>,
    /// config: the whole file (text) or settings to set ({NAME: value | null}); maintenance: message, until ...
    #[serde(flatten)]
    pub rest: Map<String, Value>,
}

/// Whether the front door answers
#[route(get = "/health", tags = "page")]
fn health(_h: &Arc<Home>, _cx: &mut Cx) -> Result<Health, Fail> {
    Ok(Health { status: "ok".into() })
}

/// The OpenAPI document of what the enabled kinds offer (no token needed; its examples say <token>)
#[route(get = "/openapi.yml", alias = "/openapi.yaml", tags = "documents", produces = "application/yaml")]
fn openapi_yml(h: &Arc<Home>, cx: &mut Cx) -> Result<Sent, Fail> {
    let spec = h.openapi(&cx.req, false);
    cx.bytes(200, "application/yaml; charset=utf-8", yaml::to_yaml(&spec).as_bytes(), "Cache-Control: no-cache\r\n");
    Ok(Sent::new())
}

/// The same document as JSON
#[route(get = "/openapi.json", tags = "documents")]
fn openapi_json(h: &Arc<Home>, cx: &mut Cx) -> Result<Value, Fail> {
    Ok(h.openapi(&cx.req, false))
}

/// The box at a glance: the cards, the models, the services, the drives, the downloads (the page polls it)
#[route(get = "/api/state", tags = "page")]
fn state(h: &Arc<Home>, _cx: &mut Cx) -> Result<Checked<State>, Fail> {
    Ok(Checked::new(h.state()))
}

/// The settings file and what each setting is
#[route(get = "/api/config", tags = "page")]
fn config(h: &Arc<Home>, _cx: &mut Cx) -> Result<Checked<Config>, Fail> {
    Ok(Checked::new(h.config()))
}

/// The API tab: the token and the document with it in the examples
#[route(get = "/api/spec", tags = "page")]
fn spec(h: &Arc<Home>, cx: &mut Cx) -> Result<Spec, Fail> {
    Ok(Spec { token: h.token.clone(), spec: h.openapi(&cx.req, true) })
}

/// A run's output (its last 64 KiB)
#[route(get = "/api/runs/{id}", tags = "page")]
fn run(h: &Arc<Home>, _cx: &mut Cx, req: RunId) -> Result<Checked<RunView>, Fail> {
    h.run_json(req.id.max(0) as u64, 64 * 1024).map(Checked::new).ok_or_else(|| (404, format!("no run {}", req.id)))
}

/// A page button: enable, disable, gpus, load, unload, config, restart, maintenance, measure, move, move_cancel, pull, pull_cancel, auto_pull; a load, a measurement and a move run on (the state shows them), the others answer when done
#[route(post = "/api/action", tags = "page")]
fn action(h: &Arc<Home>, _cx: &mut Cx, req: Action) -> Result<Map<String, Value>, Fail> {
    let b = serde_json::to_value(&req).map_err(|e| (500, e.to_string()))?;
    match h.action(&b)? {
        Value::Object(o) => Ok(o),
        other => Ok(Map::from_iter([("result".to_string(), other)])),
    }
}

routes![sync Arc<Home>, Cx, Fail; health, openapi_yml, openapi_json, state, config, spec, run, action];

/// The table, made once
pub fn table() -> &'static [rpcgen::SyncRoute<Arc<Home>, Cx, Fail>] {
    static T: std::sync::OnceLock<Vec<rpcgen::SyncRoute<Arc<Home>, Cx, Fail>>> = std::sync::OnceLock::new();
    T.get_or_init(routes)
}

/// The routes, described (the page's client, the document)
pub fn infos() -> Vec<rpcgen::RouteInfo> {
    routes().into_iter().map(|r| r.info).collect()
}

/// An error as the page's routes word one
pub(super) fn fail(m: String) -> Value {
    err(m)
}

#[cfg(test)]
mod tests {
    /// A live box's answer against its type: `NS_STATE_SAMPLE=state.json cargo test -p nextcuda-serve live_state -- --ignored`
    /// (a sample taken with curl http://127.0.0.1:8000/api/state on the box)
    #[test]
    #[ignore]
    fn live_state_sample_is_a_state() {
        let p = std::env::var("NS_STATE_SAMPLE").expect("NS_STATE_SAMPLE");
        let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
        if let Err(m) = rpcgen::conforms::<super::State>(&v) {
            panic!("{m}");
        }
    }
}
