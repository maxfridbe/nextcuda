//! The 3D server's routes (rpcgen `#[route]`): what it serves, what wfe/model3d calls (`src/gen/routes.ts`) and what
//! docs/api/model3d.openapi.yml lists, all from this table. The OpenAI-style ones take the bodies the front door's
//! methods relay (home/methods.rs: model3d.generate's Model3dGenerateBody, model3d.rig's Model3dRigBody, the job's
//! options aside); the answers are typed here and each is held to its type as it is sent (`Checked`).

use std::sync::atomic::Ordering;

use rpcgen::{route, routes, Checked, Rpc, Sent};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use super::{err, Model3dServer};
use crate::home::methods::{Model3dGenerateBody, Model3dRigBody};
use crate::page::{CancelReq, Cancelled, Cx, Fail, FileName, Gpu, Health, ModelList};

/// What the engine does
#[derive(Serialize, Deserialize, Rpc)]
pub struct Abilities {
    /// makes meshes from pictures
    pub generate: bool,
    /// paints them (a texture)
    pub texture: bool,
    /// the most views (pictures) a mesh is made from
    pub views: i64,
    /// rigs a mesh (a skeleton and skin weights)
    pub rig: bool,
    /// animates one (a motion in words)
    pub animate: bool,
}

/// The engine's own defaults (a request's absent fields)
#[derive(Serialize, Deserialize, Rpc)]
pub struct Defaults {
    pub steps: i64,
    pub cfg: f64,
    pub resolution: i64,
    pub faces: i64,
    pub texture_size: i64,
    pub frames: i64,
    pub fps: f64,
}

/// An option of the engine's own (a request sends it in `options`)
#[derive(Serialize, Deserialize, Rpc)]
pub struct EngineOptionInfo {
    pub name: String,
    /// what its value looks like ("" a flag)
    pub value: String,
    pub help: String,
}

/// The model and what it does; while a request runs, the model and `busy` only
#[derive(Serialize, Deserialize, Rpc)]
pub struct Info {
    pub model: String,
    /// the licence of its weights, whether it allows commercial use (false: the page tags it non-commercial), the conditions it puts on it
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub commercial: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license_note: Option<String>,
    /// the engine is running a request: what changes is not reported
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub busy: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arch: Option<String>,
    /// it serves its page
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wfe: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub abilities: Option<Abilities>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub defaults: Option<Defaults>,
    /// the engine's own per-request options (sent as `options`)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub options: Option<Vec<EngineOptionInfo>>,
    /// what the engine says of itself (its files, its precision ...)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub report: Option<Vec<String>>,
    /// it keeps what it makes (an output directory: links, the history)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub saves: Option<bool>,
}

/// What the server is doing: busy false, or what it is making and how far it is
#[derive(Serialize, Deserialize, Rpc)]
pub struct Progress {
    pub busy: bool,
    /// the pictures it is made from, or the motion
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    /// the engine's
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub at: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub of: Option<i64>,
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

/// A mesh made here: its links, and its settings (its sidecar <name>.json)
#[derive(Serialize, Deserialize, Rpc)]
pub struct Mesh {
    /// the GLB (glTF binary): GET it here
    pub url: String,
    /// the picture it was made from (PNG)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub picture: Option<String>,
    pub bytes: i64,
    /// unix seconds
    pub created: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arch: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steps: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cfg: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolution: Option<i64>,
    /// the triangle budget asked
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub faces: Option<i64>,
    /// what it has
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub triangles: Option<i64>,
    /// a texture was asked for
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub texture: Option<bool>,
    /// it has one
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub textured: Option<bool>,
    /// rigged: its joints
    #[serde(default)]
    pub rigged: Option<i64>,
    /// animated: its frames
    #[serde(default)]
    pub animated: Option<i64>,
    /// a rig's motion
    #[serde(default)]
    pub prompt: Option<String>,
    /// rig: the frames asked
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frames: Option<i64>,
    /// each picture's side (null: not said)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub views: Option<Vec<Option<String>>>,
    /// seconds it took
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub took: Option<f64>,
    /// the card's energy, watt-hours
    #[serde(default)]
    pub wh: Option<f64>,
    /// what else its settings record (a file made by an older server, or by the front door's model3d.tool)
    #[serde(flatten)]
    pub rest: Map<String, Value>,
}

/// The meshes in the output directory, newest first (read from it each time)
#[derive(Serialize, Deserialize, Rpc)]
pub struct History {
    pub data: Vec<Mesh>,
}

/// How a mesh was made
#[derive(Serialize, Deserialize, Rpc)]
pub struct MeshRun {
    pub took: f64,
    pub seed: i64,
    pub wh: Option<f64>,
    pub triangles: i64,
}

/// A mesh made (response_format url): its links and settings, and how it was made (`nextcuda`: this server's key)
#[derive(Serialize, Deserialize, Rpc)]
pub struct Meshes {
    pub created: i64,
    pub data: Vec<Mesh>,
    pub nextcuda: MeshRun,
}

/// The card: its name, memory, power and temperatures
#[route(get = "/api/gpu", tags = "page")]
fn gpu(s: &Model3dServer, _cx: &mut Cx) -> Result<Gpu, Fail> {
    Ok(s.gpu_json())
}

/// Whether the server answers
#[route(get = "/health", tags = "page")]
fn health(_s: &Model3dServer, _cx: &mut Cx) -> Result<Health, Fail> {
    Ok(Health { status: "ok".into() })
}

/// The model served, as OpenAI lists models
#[route(get = "/v1/models", alias = "/models", tags = "openai")]
fn models(s: &Model3dServer, _cx: &mut Cx) -> Result<ModelList, Fail> {
    Ok(ModelList::of(&s.model))
}

/// The model and what it does: its abilities, defaults, options
#[route(get = "/api/info", tags = "page")]
fn info(s: &Model3dServer, _cx: &mut Cx) -> Result<Checked<Info>, Fail> {
    Ok(Checked::new(crate::page::with_license(s.info(), &s.license)))
}

/// What the server is doing now (the page polls it)
#[route(get = "/api/progress", tags = "page")]
fn progress(s: &Model3dServer, _cx: &mut Cx) -> Result<Checked<Progress>, Fail> {
    let mut p = s.progress.lock().unwrap().clone();
    p["waiting"] = json!(s.waiting.load(Ordering::Relaxed));
    p["request_id"] = json!(s.cancel.current());
    Ok(Checked::new(p))
}

/// The meshes made here, newest first
#[route(get = "/api/history", tags = "page")]
fn history(s: &Model3dServer, _cx: &mut Cx) -> Result<Checked<History>, Fail> {
    // read again: the front door's model3d.tool writes meshes here too, and deleted ones drop out
    let mut h = s.history.lock().unwrap();
    if let Some(d) = &s.out_dir {
        *h = super::earlier(d);
    }
    Ok(Checked::new(json!({"data": h.iter().collect::<Vec<_>>()})))
}

/// Stops a request: the one named (running or waiting), else the one running (between steps)
#[route(post = "/api/cancel", tags = "page")]
fn cancel(s: &Model3dServer, _cx: &mut Cx, req: CancelReq) -> Result<Cancelled, Fail> {
    let busy = s.progress.lock().unwrap()["busy"] == true;
    let running = s.cancel.request(req.id.as_deref(), busy);
    Ok(Cancelled { cancelled: running, id: req.id })
}

/// A file made here: the GLB, its picture (<f>.png) or its settings (<f>.json)
#[route(get = "/v1/model3d/files/{name}", tags = "files", produces = "model/gltf-binary, image/png, application/json")]
fn file(s: &Model3dServer, cx: &mut Cx, req: FileName) -> Result<Sent, Fail> {
    s.file(cx, &req.name);
    Ok(Sent::new())
}

/// A mesh from pictures (image, or images: several views [{image, side}]), textured or not; or a mesh given (mesh / file) painted. response_format glb (default: the file) or url (JSON with its link)
#[route(post = "/v1/model3d/generate", tags = "openai", produces = "application/json, model/gltf-binary")]
fn generate(s: &Model3dServer, cx: &mut Cx, req: Model3dGenerateBody) -> Result<Sent<Meshes>, Fail> {
    s.answer(cx, false, &serde_json::to_value(&req).map_err(|e| (500, e.to_string()))?);
    Ok(Sent::new())
}

/// A mesh (mesh: GLB or OBJ, base64 / a data: URL; or file: a GLB made here) rigged - a skeleton and skin weights - or animated (prompt: the motion). response_format glb (default) or url
#[route(post = "/v1/model3d/rig", tags = "openai", produces = "application/json, model/gltf-binary")]
fn rig(s: &Model3dServer, cx: &mut Cx, req: Model3dRigBody) -> Result<Sent<Meshes>, Fail> {
    s.answer(cx, true, &serde_json::to_value(&req).map_err(|e| (500, e.to_string()))?);
    Ok(Sent::new())
}

routes![sync Model3dServer, Cx, Fail; gpu, health, models, info, progress, history, cancel, file, generate, rig];

impl Model3dServer {
    /// A generation or a rig, its answer written: the JSON (url, held to Meshes) or the GLB
    fn answer(&self, cx: &mut Cx, rig: bool, b: &Value) {
        match self.run_request(rig, b, &cx.base()) {
            Ok((Some(v), _)) => cx.json(200, &Checked::<Meshes>::new(v).0),
            Ok((None, glb)) => cx.bytes(200, "model/gltf-binary", &glb, ""),
            Err((code, m)) => cx.json(code, &err(m)),
        }
    }
}

/// The table, made once
pub fn table() -> &'static [rpcgen::SyncRoute<Model3dServer, Cx, Fail>] {
    static T: std::sync::OnceLock<Vec<rpcgen::SyncRoute<Model3dServer, Cx, Fail>>> = std::sync::OnceLock::new();
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
