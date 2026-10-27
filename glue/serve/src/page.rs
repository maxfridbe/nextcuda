//! What the model servers (image, audio, 3D) share of their own routes: the context a route's handler gets (`Cx`:
//! the connection, the request, the CORS headers), the answers every one of them gives the same way (the card, the
//! model list, a cancel), and `serve`, which finds a request's route in a server's table (rpcgen `#[route]`), runs
//! it and writes its answer. A server's table is what it serves, what its page's client (`wfe/<page>/src/gen/`) calls
//! and what its document (`docs/api/<server>.openapi.yml`) lists: `nextcuda gen-api` writes both from it.

use rpcgen::{BadPayload, RouteError, RouteReply, Rpc, SyncRoute};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::http::{self, Conn, Request};

/// A route's failure: the status and the words.
pub type Fail = (u16, String);

/// The card's licence of a server's model (`license`, `commercial`, `license_note`: an object, or null for none) in
/// the answer of its `/api/info`, which the page names the model from
pub fn with_license(mut info: Value, licence: &Value) -> Value {
    if let (Some(o), Some(l)) = (info.as_object_mut(), licence.as_object()) {
        o.extend(l.clone());
    }
    info
}

/// The licence keys of a registry entry, for a server's `license` (null: the entry has none)
pub fn licence_of(entry: &Value) -> Value {
    let l: serde_json::Map<String, Value> = ["license", "commercial", "license_note"].iter().filter_map(|k| Some((k.to_string(), entry.get(*k)?.clone()))).collect();
    if l.is_empty() { Value::Null } else { Value::Object(l) }
}

/// What a route's handler gets besides its server: the connection (a handler that writes its own answer, a file),
/// the request as read (its head for a Range, its body for a multipart form) and the CORS headers to send.
pub struct Cx {
    pub s: Conn,
    pub req: Request,
    pub cors: String,
}

impl Cx {
    /// `http://<host>` as the request was addressed (links in an answer), or "" without a Host
    pub fn base(&self) -> String {
        self.req.host.as_deref().map(|h| format!("http://{h}")).unwrap_or_default()
    }

    /// Bytes with their type, the CORS headers and `headers` (each ending \r\n)
    pub fn bytes(&mut self, status: u16, kind: &str, body: &[u8], headers: &str) {
        let h = format!("{}{headers}", self.cors);
        http::respond_bytes(&mut self.s, status, kind, body, &h);
    }

    /// JSON with the CORS headers
    pub fn json(&mut self, status: u16, v: &Value) {
        http::respond_with(&mut self.s, status, v, &self.cors);
    }
}

/// Runs the route of `table` that serves the request, writing its answer (an error as `err` words it): false when
/// none does (the server's other paths: its page, 404).
pub fn serve<S>(table: &[SyncRoute<S, Cx, Fail>], state: &S, cx: &mut Cx, err: fn(String) -> Value) -> bool {
    let Some((r, params)) = rpcgen::find_route(table, &cx.req.method, &cx.req.path) else { return false };
    let query = cx.req.target.split_once('?').map(|q| q.1.to_string()).unwrap_or_default();
    // a JSON body is the request; any other the handler reads from cx
    let body = if r.info.body == rpcgen::Body::Json { std::mem::take(&mut cx.req.body) } else { Vec::new() };
    match r.call(state, cx, params, &query, &body) {
        Ok(RouteReply::Json(v)) => cx.json(200, &v),
        Ok(RouteReply::Sent) => {}
        Err(RouteError::Payload(BadPayload(m))) => cx.json(400, &err(m)),
        Err(RouteError::Answer(m)) => cx.json(500, &err(m)),
        Err(RouteError::Handler((code, m))) => cx.json(code, &err(m)),
    }
    true
}

/// The card a model server runs on, as its page shows it
#[derive(Serialize, Deserialize, Rpc)]
pub struct Gpu {
    pub name: String,
    pub vram_used_mb: u64,
    pub vram_total_mb: u64,
    /// watts now (none: the driver does not say)
    pub power_w: Option<f64>,
    /// °C
    pub temp_pkg: Option<f64>,
    /// the memory's °C
    pub temp_vram: Option<f64>,
}

/// Whether the server answers (and has its model loaded)
#[derive(Serialize, Deserialize, Rpc)]
pub struct Health {
    /// ok, or loading (HTTP 503 then)
    pub status: String,
}

/// OpenAI's list of models: the one this server serves
#[derive(Serialize, Deserialize, Rpc)]
pub struct ModelList {
    /// list
    pub object: String,
    pub data: Vec<ModelEntry>,
}

/// A model in OpenAI's list
#[derive(Serialize, Deserialize, Rpc)]
pub struct ModelEntry {
    pub id: String,
    /// model
    pub object: String,
    pub owned_by: String,
    pub created: i64,
}

impl ModelList {
    pub fn of(model: &str) -> ModelList {
        ModelList { object: "list".into(), data: vec![ModelEntry { id: model.into(), object: "model".into(), owned_by: "nextcuda".into(), created: 0 }] }
    }
}

/// What a cancel takes
#[derive(Serialize, Deserialize, Rpc)]
pub struct CancelReq {
    /// that request only (the front door's job), running or waiting; none: the one running
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
}

/// What a cancel did
#[derive(Serialize, Deserialize, Rpc)]
pub struct Cancelled {
    /// a request running was told to stop
    pub cancelled: bool,
    pub id: Option<String>,
}

/// A file a link names (`.../files/{name}`)
#[derive(Serialize, Deserialize, Rpc)]
pub struct FileName {
    /// the file's name as the answer's url has it
    pub name: String,
}
