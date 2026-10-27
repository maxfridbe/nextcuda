//! Routes: an HTTP route that is not an RPC method - a page's own API (`GET /api/info`), an OpenAI-compatible one
//! (`POST /v1/images/generations`), a file a link names (`GET /v1/images/files/{name}`) - described by its handler
//! as a method is: `#[route(get = "/api/info")]` and a doc comment on the handler, its request (path and query
//! parameters, or a JSON body) and its answer `Rpc` types. A table of them (`routes![sync State, Cx, Error; ...]`)
//! is what the server dispatches by, and the route's OpenAPI paths and a typed TypeScript client are made from the
//! same table, so a route the page calls, the server serves and the document lists is one declaration.
//!
//! ```
//! use rpcgen::{route, routes, Rpc, Sent};
//! pub struct Server;
//! pub struct Cx { pub sent: Vec<u8> }
//! #[derive(serde::Serialize, Rpc)]
//! pub struct Info { pub loaded: bool }
//! #[derive(serde::Deserialize, Rpc)]
//! pub struct FileRef {
//!     /// the file's name
//!     pub name: String,
//!     /// its width, when a smaller copy will do
//!     pub width: Option<i64>,
//! }
//! /// The model's state
//! #[route(get = "/api/info", tags = "page")]
//! fn info(_s: &Server, _cx: &mut Cx) -> Result<Info, String> { Ok(Info { loaded: true }) }
//! /// A picture an answer links
//! #[route(get = "/files/{name}", produces = "image/png")]
//! fn file(_s: &Server, cx: &mut Cx, req: FileRef) -> Result<Sent, String> {
//!     cx.sent = format!("{} {:?}", req.name, req.width).into_bytes();
//!     Ok(Sent::new())
//! }
//! routes![sync Server, Cx, String; info, file];
//! let t = routes();
//! let (r, params) = rpcgen::find_route(&t, "get", "/files/a.png").unwrap();
//! let mut cx = Cx { sent: Vec::new() };
//! r.call(&Server, &mut cx, params, "width=64", b"").unwrap();
//! assert_eq!(cx.sent, b"a.png Some(64)");
//! let infos: Vec<_> = t.iter().map(|r| r.info.clone()).collect();
//! rpcgen::check_routes(&infos).unwrap();
//! rpcgen::route_parity(&infos).unwrap();
//! let ts = rpcgen::routes_typescript("t", &infos).unwrap();
//! assert!(ts.contains("info: (): Promise<Info> =>") && ts.contains("file: (req: FileRef): string =>"), "{ts}");
//! ```
//!
//! Two routes of one method and path in a table do not compile:
//!
//! ```compile_fail,E0080
//! use rpcgen::{route, routes, Rpc};
//! pub struct Server;
//! #[derive(serde::Serialize, Rpc)]
//! pub struct Info { pub loaded: bool }
//! /// The model's state
//! #[route(get = "/api/info")]
//! fn info(_s: &Server, _cx: &mut ()) -> Result<Info, String> { Ok(Info { loaded: true }) }
//! /// Again
//! #[route(get = "/api/info")]
//! fn info2(_s: &Server, _cx: &mut ()) -> Result<Info, String> { Ok(Info { loaded: true }) }
//! routes![sync Server, (), String; info, info2];
//! ```

use super::*;
use std::marker::PhantomData;

/// What a route's request body is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Body {
    /// none: the request is the path and query parameters (a GET)
    None,
    /// a JSON object: the request type (a POST's default)
    Json,
    /// multipart/form-data: the handler reads the parts; `form` describes them
    Multipart,
    /// the raw bytes of a file: the handler reads them; the request is the path and query parameters
    Bytes,
    /// a JSON object or the same fields as multipart/form-data (OpenAI's edits): the handler reads either; `form`
    /// describes both
    JsonOrForm,
}

impl Body {
    fn word(self) -> &'static str {
        match self {
            Body::None => "none",
            Body::Json => "json",
            Body::Multipart => "multipart",
            Body::Bytes => "bytes",
            Body::JsonOrForm => "json, multipart",
        }
    }
}

/// What a route is: everything the document and the client say of it.
#[derive(Debug, Clone)]
pub struct RouteInfo {
    /// the client's name for it (the handler's, in camelCase)
    pub name: &'static str,
    /// GET or POST
    pub method: &'static str,
    /// `/v1/images/files/{name}`: a `{field}` is a path parameter, a field of the request
    pub path: &'static str,
    /// the same route at other paths (an older name kept)
    pub aliases: Vec<&'static str>,
    pub summary: &'static str,
    pub description: &'static str,
    pub tags: Vec<&'static str>,
    pub body: Body,
    /// the path and query parameters, or the JSON body (Body::Json)
    pub request: Shape,
    /// Body::Multipart: the parts
    pub form: Option<Shape>,
    pub response: Shape,
    pub defs: Defs,
    /// the answer's media types (none: application/json, or application/octet-stream for a file)
    pub produces: Vec<&'static str>,
    /// a file served in byte ranges (a Range header answered 206)
    pub ranges: bool,
    /// a body a caller sends (Body::Json), as JSON text
    pub example: Option<&'static str>,
}

impl RouteInfo {
    pub fn new<Req: RpcType, Resp: RpcType>(name: &'static str, method: &'static str, path: &'static str, summary: &'static str, description: &'static str, tags: &[&'static str]) -> Self {
        let mut defs = Defs::default();
        let request = Req::shape(&mut defs);
        let response = Resp::shape(&mut defs);
        let body = if method == "GET" { Body::None } else { Body::Json };
        RouteInfo {
            name, method, path, aliases: Vec::new(), summary, description, tags: tags.to_vec(), body, request, form: None, response, defs,
            produces: Vec::new(), ranges: false, example: None,
        }
    }

    pub fn with_body(mut self, body: Body) -> Self {
        self.body = body;
        self
    }

    pub fn with_form<F: RpcType>(mut self) -> Self {
        self.form = Some(F::shape(&mut self.defs));
        self
    }

    pub fn with_aliases(mut self, aliases: &[&'static str]) -> Self {
        self.aliases = aliases.to_vec();
        self
    }

    pub fn with_produces(mut self, produces: &[&'static str]) -> Self {
        self.produces = produces.to_vec();
        self
    }

    pub fn with_ranges(mut self, ranges: bool) -> Self {
        self.ranges = ranges;
        self
    }

    pub fn with_example(mut self, example: Option<&'static str>) -> Self {
        self.example = example;
        self
    }

    /// `GET /api/info`: how the document and the client's JSDoc name it
    pub fn key(&self) -> String {
        format!("{} {}", self.method, self.path)
    }

    /// The path's parameters (`{name}`), in order.
    pub fn path_params(&self) -> Vec<&'static str> {
        self.path.split('/').filter_map(|s| s.strip_prefix('{').and_then(|s| s.strip_suffix('}'))).collect()
    }

    /// The request's fields (an object type's; none for NoParams).
    pub fn request_fields(&self) -> Vec<Field> {
        object_fields(&self.defs, &self.request)
    }

    /// The query parameters: the request's fields that are not in the path (none when the body is the request).
    pub fn query_fields(&self) -> Vec<Field> {
        if self.body == Body::Json {
            return Vec::new();
        }
        let path = self.path_params();
        self.request_fields().into_iter().filter(|f| !path.contains(&f.name.as_str())).collect()
    }

    /// The answer is a file (bytes with a media type), not JSON.
    pub fn answers_bytes(&self) -> bool {
        self.response == Shape::Blob
    }

    /// The answer is JSON or, as the request asks, a file (`produces = "application/json, audio/wav"`).
    pub fn json_or_bytes(&self) -> bool {
        !self.answers_bytes() && self.produces.iter().any(|t| !t.contains("json"))
    }

    /// The answer's media types.
    pub fn media_types(&self) -> Vec<&'static str> {
        if !self.produces.is_empty() {
            self.produces.clone()
        } else if self.answers_bytes() {
            vec!["application/octet-stream"]
        } else {
            vec!["application/json"]
        }
    }

    /// The parameters of a request `path` that this route serves (its method aside): None when it is another's.
    /// The fixed parts match in any case, as method names do.
    pub fn match_path(&self, path: &str) -> Option<Vec<(String, String)>> {
        std::iter::once(self.path).chain(self.aliases.iter().copied()).find_map(|tpl| match_template(tpl, path))
    }
}

fn object_fields(defs: &Defs, s: &Shape) -> Vec<Field> {
    match s {
        Shape::Ref(n) => match defs.types.get(n) {
            Some(Named::Object { object, .. }) => object.fields.clone(),
            _ => Vec::new(),
        },
        _ => Vec::new(),
    }
}

fn match_template(tpl: &str, path: &str) -> Option<Vec<(String, String)>> {
    let (a, b): (Vec<&str>, Vec<&str>) = (tpl.split('/').collect(), path.split('/').collect());
    if a.len() != b.len() {
        return None;
    }
    let mut params = Vec::new();
    for (t, p) in a.iter().zip(&b) {
        match t.strip_prefix('{').and_then(|t| t.strip_suffix('}')) {
            Some(name) if !p.is_empty() => params.push((name.to_string(), percent_decode(p))),
            Some(_) => return None,
            None if t.eq_ignore_ascii_case(p) => {}
            None => return None,
        }
    }
    Some(params)
}

fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'%' if i + 2 < b.len() => {
                match u8::from_str_radix(std::str::from_utf8(&b[i + 1..i + 3]).unwrap_or(""), 16) {
                    Ok(v) => {
                        out.push(v);
                        i += 3;
                        continue;
                    }
                    Err(_) => out.push(b'%'),
                }
            }
            b'+' => out.push(b' '),
            c => out.push(c),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// A query string as its pairs, decoded.
pub fn query_pairs(query: &str) -> Vec<(String, String)> {
    query
        .split('&')
        .filter(|kv| !kv.is_empty())
        .map(|kv| match kv.split_once('=') {
            Some((k, v)) => (percent_decode(k), percent_decode(v)),
            None => (percent_decode(kv), String::new()),
        })
        .collect()
}

/// A parameter's text as the value its field's type reads: a number, a boolean, a string.
fn coerce(shape: &Shape, text: &str) -> Value {
    match shape {
        Shape::Nullable(inner) => coerce(inner, text),
        Shape::Integer => text.parse::<i64>().map(Value::from).unwrap_or_else(|_| json!(text)),
        Shape::Number => text.parse::<f64>().ok().and_then(|f| serde_json::Number::from_f64(f).map(Value::Number)).unwrap_or_else(|| json!(text)),
        Shape::Bool => match text {
            "true" | "1" | "" => json!(true),
            "false" | "0" => json!(false),
            _ => json!(text),
        },
        _ => json!(text),
    }
}

/// The request a route's handler decodes: its path parameters and query (each as its field's type reads it), or
/// for a JSON body the body with the path parameters.
pub fn route_input(info: &RouteInfo, params: &[(String, String)], query: &str, body: &[u8]) -> Result<Value, BadPayload> {
    let fields = info.request_fields();
    let shape_of = |k: &str| fields.iter().find(|f| f.name == k).map(|f| f.shape.clone()).unwrap_or(Shape::String);
    let mut o = Map::new();
    if info.body == Body::Json && !body.is_empty() {
        match serde_json::from_slice::<Value>(body) {
            Ok(Value::Object(m)) => o = m,
            Ok(_) => return Err(BadPayload("the body is not a JSON object".into())),
            Err(e) => return Err(BadPayload(format!("the body is not JSON: {e}"))),
        }
    }
    if info.body != Body::Json {
        for (k, v) in query_pairs(query) {
            let shape = shape_of(&k);
            let val = coerce(match &shape { Shape::Array(i) => i, s => s }, &v);
            match (&shape, o.get_mut(&k)) {
                (Shape::Array(_) | Shape::Nullable(_), Some(Value::Array(a))) => a.push(val),
                (Shape::Array(_), None) => {
                    o.insert(k, json!([val]));
                }
                _ => {
                    o.insert(k, val);
                }
            }
        }
    }
    for (k, v) in params {
        o.insert(k.clone(), coerce(&shape_of(k), v));
    }
    Ok(Value::Object(o))
}

/// What a route's handler did: an answer to send as JSON, or the answer written already (a file, a stream, a
/// request passed on).
#[derive(Debug, Clone, PartialEq)]
pub enum RouteReply {
    Json(Value),
    Sent,
}

/// A route's failure: the request did not fit, the answer did not encode, or the handler's own error.
#[derive(Debug)]
pub enum RouteError<E> {
    Payload(BadPayload),
    Answer(String),
    Handler(E),
}

/// An answer the handler wrote itself, described as `T` (a file: `Blob`, the default; or the JSON a request
/// passed on returns).
pub struct Sent<T = Blob>(PhantomData<fn() -> T>);

impl<T> Sent<T> {
    pub fn new() -> Self {
        Sent(PhantomData)
    }
}

impl<T> Default for Sent<T> {
    fn default() -> Self {
        Sent::new()
    }
}

impl<T: RpcType> RpcType for Sent<T> {
    fn shape(defs: &mut Defs) -> Shape {
        T::shape(defs)
    }
}

/// What a route's handler answers.
pub trait RouteAnswer {
    fn into_route_reply(self) -> Result<RouteReply, String>;
}

impl<T: Serialize + RpcType> RouteAnswer for T {
    fn into_route_reply(self) -> Result<RouteReply, String> {
        serde_json::to_value(self).map(RouteReply::Json).map_err(|e| format!("the answer did not encode: {e}"))
    }
}

impl<T> RouteAnswer for Sent<T> {
    fn into_route_reply(self) -> Result<RouteReply, String> {
        Ok(RouteReply::Sent)
    }
}

/// A route handler's `Result`: `#[route]` names the answer's type through this, so an alias of a Result is fine.
pub trait RouteResult {
    type Ok: RpcType;
    type Err;
    fn into_route(self) -> Result<RouteReply, RouteError<Self::Err>>;
}

impl<T: RouteAnswer + RpcType, E> RouteResult for Result<T, E> {
    type Ok = T;
    type Err = E;
    fn into_route(self) -> Result<RouteReply, RouteError<E>> {
        self.map_err(RouteError::Handler)?.into_route_reply().map_err(RouteError::Answer)
    }
}

/// A route handler's answer for a handler that answers `R`.
pub type RouteReplyOf<R> = Result<RouteReply, RouteError<<R as RouteResult>::Err>>;

/// A route's request from its input (`route_input`).
pub fn decode_route<T: DeserializeOwned, E>(v: Value) -> Result<T, RouteError<E>> {
    serde_json::from_value(v).map_err(|e| RouteError::Payload(BadPayload(e.to_string())))
}

/// An answer a handler made as a JSON value, described as `T` and held to it (`conforms`) as it is made: the
/// sync counterpart of `Shaped::value`. A debug build answers a mismatch as an error naming where (code
/// `answer_shape`), so a test that calls the route sees it; a release build sends the answer and says on stderr
/// where it differs (the place only: a value could be a secret).
pub struct Checked<T>(pub Value, PhantomData<fn() -> T>);

impl<T: DeserializeOwned + Serialize> Checked<T> {
    pub fn new(v: Value) -> Self {
        if let Err(m) = conforms::<T>(&v) {
            let ty = std::any::type_name::<T>().rsplit("::").next().unwrap_or("");
            if cfg!(debug_assertions) {
                return Checked(error_body(&format!("the answer is not a {ty}: {m}"), "answer_shape", false), PhantomData);
            }
            eprintln!("rpcgen: an answer is not the {ty} its route declares, at `{}`", if m.at.is_empty() { "." } else { &m.at });
        }
        Checked(v, PhantomData)
    }
}

impl<T> Serialize for Checked<T> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(s)
    }
}

impl<T: RpcType> RpcType for Checked<T> {
    fn shape(defs: &mut Defs) -> Shape {
        T::shape(defs)
    }
}

/// One route of a sync table: what it is, and its handler (the request decoded from `route_input`'s value).
pub struct SyncRoute<S, C, E> {
    pub info: RouteInfo,
    pub handler: for<'a, 'b> fn(&'a S, &'b mut C, Value) -> Result<RouteReply, RouteError<E>>,
}

impl<S, C, E> SyncRoute<S, C, E> {
    /// Runs the route for a request: its path parameters (`find_route`), its query string and its body.
    pub fn call(&self, s: &S, cx: &mut C, params: Vec<(String, String)>, query: &str, body: &[u8]) -> Result<RouteReply, RouteError<E>> {
        let input = route_input(&self.info, &params, query, body).map_err(RouteError::Payload)?;
        (self.handler)(s, cx, input)
    }
}

/// A request's path parameters: (name, value), decoded.
pub type PathParams = Vec<(String, String)>;

/// The route of a table that serves `method path` (the method in any case), and its path parameters.
pub fn find_route<'t, S, C, E>(table: &'t [SyncRoute<S, C, E>], method: &str, path: &str) -> Option<(&'t SyncRoute<S, C, E>, PathParams)> {
    table.iter().filter(|r| r.info.method.eq_ignore_ascii_case(method)).find_map(|r| r.info.match_path(path).map(|p| (r, p)))
}

/// What a table must be for its document and client to say what it serves: each path parameter a field the
/// request requires, a JSON body without path parameters (the body is the request), a GET without a body, a
/// form only on a multipart route, and no two routes at one method and path.
pub fn check_routes(routes: &[RouteInfo]) -> Result<(), String> {
    let mut seen = std::collections::BTreeSet::new();
    for r in routes {
        let k = r.key();
        for p in std::iter::once(r.path).chain(r.aliases.iter().copied()) {
            if !seen.insert(format!("{} {}", r.method, p.to_ascii_lowercase())) {
                return Err(format!("{k}: two routes at {} {p}", r.method));
            }
        }
        if r.method == "GET" && r.body != Body::None {
            return Err(format!("{k}: a GET has no body"));
        }
        if r.form.is_some() && !matches!(r.body, Body::Multipart | Body::JsonOrForm) {
            return Err(format!("{k}: a form is a multipart route's"));
        }
        if r.body == Body::JsonOrForm && r.form.is_none() {
            return Err(format!("{k}: a JSON or multipart body is described by its form"));
        }
        let fields = r.request_fields();
        for p in r.path_params() {
            if r.body == Body::Json {
                return Err(format!("{k}: a JSON body is the request; {{{p}}} would be outside it"));
            }
            match fields.iter().find(|f| f.name == p) {
                Some(f) if !f.optional && matches!(f.shape, Shape::String | Shape::Integer) => {}
                Some(_) => return Err(format!("{k}: {{{p}}} is a required string or integer field")),
                None => return Err(format!("{k}: {{{p}}} is not a field of the request")),
            }
        }
        if r.example.is_some() && r.body != Body::Json {
            return Err(format!("{k}: an example is a JSON body's"));
        }
    }
    Ok(())
}

fn route_defs(routes: &[RouteInfo]) -> Result<Defs, String> {
    let mut d = Defs::default();
    for r in routes {
        d.merge(&r.defs).map_err(|e| format!("{}: {e}", r.key()))?;
    }
    Ok(d)
}

/// The named types of these routes as JSON Schema (#/components/schemas), for a document written around them.
pub fn route_components(routes: &[RouteInfo]) -> Result<Map<String, Value>, String> {
    Ok(route_defs(routes)?.types.iter().map(|(k, v)| (k.clone(), named_schema(v))).collect())
}

fn route_examples(r: &RouteInfo) -> Option<Value> {
    r.example.and_then(|e| serde_json::from_str::<Value>(e).ok())
}

/// A route's OpenAPI operation.
fn route_op(r: &RouteInfo) -> Value {
    let mut op = json!({"operationId": r.name, "summary": r.summary, "tags": r.tags});
    if r.description.len() > r.summary.len() {
        op["description"] = json!(r.description);
    }
    let fields = r.request_fields();
    let mut params = Vec::new();
    for p in r.path_params() {
        let f = fields.iter().find(|f| f.name == p).expect("check_routes: a path parameter is a field");
        let mut q = json!({"name": p, "in": "path", "required": true, "schema": field_schema(f)});
        if !f.doc.is_empty() {
            q["description"] = json!(f.doc);
        }
        params.push(q);
    }
    for f in r.query_fields() {
        // an optional parameter may be left out: its schema is the value's, not "or null"
        let mut bare = f.clone();
        if let Shape::Nullable(inner) = &f.shape {
            bare.shape = (**inner).clone();
        }
        let mut q = json!({"name": f.name, "in": "query", "required": !f.optional, "schema": field_schema(&bare)});
        if !f.doc.is_empty() {
            q["description"] = json!(f.doc);
        }
        params.push(q);
    }
    if r.ranges {
        params.push(json!({"name": "Range", "in": "header", "required": false, "schema": {"type": "string"},
                           "description": "bytes=a-b, bytes=a- or bytes=-n (one span): answered 206 with Content-Range"}));
    }
    if !params.is_empty() {
        op["parameters"] = Value::Array(params);
    }
    match r.body {
        Body::None => {}
        Body::Json => {
            let mut media = json!({"schema": schema(&r.request)});
            if let Some(e) = route_examples(r) {
                media["example"] = e;
            }
            op["requestBody"] = json!({"required": true, "content": {"application/json": media}});
        }
        Body::Multipart => {
            let s = r.form.as_ref().map(schema).unwrap_or_else(|| json!({"type": "object"}));
            op["requestBody"] = json!({"required": true, "content": {"multipart/form-data": {"schema": s}}});
        }
        Body::Bytes => {
            op["requestBody"] = json!({"required": true, "content": {"application/octet-stream": {"schema": {"type": "string", "format": "binary"}}}});
        }
        Body::JsonOrForm => {
            let s = r.form.as_ref().map(schema).unwrap_or_else(|| json!({"type": "object"}));
            op["requestBody"] = json!({"required": true, "content": {"application/json": {"schema": s}, "multipart/form-data": {"schema": s}}});
        }
    }
    // a JSON answer that may be a file instead (response_format wav): the file's types are bytes
    let content: Map<String, Value> = r
        .media_types()
        .iter()
        .map(|t| {
            let s = if !r.answers_bytes() && !t.contains("json") { json!({"type": "string", "format": "binary"}) } else { schema(&r.response) };
            (t.to_string(), json!({"schema": s}))
        })
        .collect();
    let mut responses = json!({"200": {"description": if r.answers_bytes() { "the file" } else { "ok" }, "content": content},
                               "default": {"description": "an error: the body's `error` says what"}});
    if r.ranges {
        responses["206"] = json!({"description": "the span asked for", "content": content});
    }
    op["responses"] = responses;
    if !r.aliases.is_empty() {
        op["x-aliases"] = json!(r.aliases);
    }
    op
}

/// The routes as OpenAPI paths (`{"/api/info": {"get": {...}}}`), for a document written around them; their named
/// types are `route_components`'.
pub fn route_paths(routes: &[RouteInfo]) -> Result<Map<String, Value>, String> {
    check_routes(routes)?;
    let mut paths = Map::new();
    for r in routes {
        let entry = paths.entry(r.path.to_string()).or_insert_with(|| json!({}));
        entry[r.method.to_ascii_lowercase()] = route_op(r);
    }
    Ok(paths)
}

/// The OpenAPI document of a table of routes: `info` as given (the project's own prose), `servers`, a path per
/// route, the named types as components.
pub fn routes_openapi(info: Value, server: &str, routes: &[RouteInfo]) -> Result<String, String> {
    let paths = route_paths(routes)?;
    let schemas = route_components(routes)?;
    let y = |v: Value| serde_yaml::to_value(v).map_err(|e| e.to_string());
    let mut doc = serde_yaml::Mapping::new();
    doc.insert("openapi".into(), "3.1.0".into());
    doc.insert("info".into(), y(info)?);
    doc.insert("servers".into(), y(json!([{"url": server}]))?);
    doc.insert("paths".into(), y(Value::Object(paths))?);
    doc.insert("components".into(), y(json!({"schemas": schemas}))?);
    serde_yaml::to_string(&doc).map_err(|e| e.to_string())
}

/// A route's JSDoc: its text, then `@route`, `@alias`, `@tags`, `@body`, `@produces`, `@ranges`, `@example`.
fn route_doc(r: &RouteInfo) -> (String, Vec<(String, String)>) {
    let text = if r.description.len() > r.summary.len() { r.description } else { r.summary }.to_string();
    let mut tags = vec![("route".to_string(), r.key())];
    for a in &r.aliases {
        tags.push(("alias".to_string(), a.to_string()));
    }
    if !r.tags.is_empty() {
        tags.push(("tags".to_string(), r.tags.join(", ")));
    }
    if matches!(r.body, Body::Multipart | Body::Bytes | Body::JsonOrForm) {
        tags.push(("body".to_string(), r.body.word().to_string()));
    }
    if !r.produces.is_empty() {
        tags.push(("produces".to_string(), r.produces.join(", ")));
    }
    if r.ranges {
        tags.push(("ranges".to_string(), String::new()));
    }
    if let Some(e) = route_examples(r) {
        tags.push(("example".to_string(), e.to_string()));
    }
    (text, tags)
}

/// The typed client of a table of routes, as a TypeScript module: an interface or a union per named type, and
/// `routes(base, fetcher)` with one function per route. `base` goes before each path without its leading `/`, so
/// the default `""` makes every call relative to the page (a page served at its own port and under a proxy's
/// prefix alike). A JSON answer resolves to its type; a body whose `error` is set (and, where the type has an
/// `error` field of its own, `ok` false) rejects with a `RouteError` carrying its words - the body says, not the
/// status. A file a GET serves is its URL; one a POST answers, a Blob.
pub fn routes_typescript(title: &str, routes: &[RouteInfo]) -> Result<String, String> {
    check_routes(routes)?;
    let defs = route_defs(routes)?;
    let mut out = String::new();
    out.push_str(&format!(
        "// {title}: the routes' types and a typed client.\n// Generated by rpcgen from the #[route] handlers: do not edit; regenerate.\n/* eslint-disable */\n\n"
    ));
    for (name, n) in &defs.types {
        match n {
            Named::Object { doc, object } => {
                ts_doc(&mut out, "", doc);
                out.push_str(&format!("export interface {name} {{\n"));
                for f in &object.fields {
                    ts_jsdoc(&mut out, "  ", &f.doc, &field_tags(f));
                    out.push_str(&format!("  {}{}: {};\n", ts_key(&f.name), if f.optional { "?" } else { "" }, field_ts(f)));
                }
                if object.rest.is_some() {
                    out.push_str("  /** other keys */\n  [key: string]: unknown;\n");
                }
                out.push_str("}\n\n");
            }
            Named::Union { doc, values } => {
                ts_doc(&mut out, "", doc);
                let v: Vec<String> = values.iter().map(|v| format!("{v:?}")).collect();
                out.push_str(&format!("export type {name} = {};\n\n", if v.is_empty() { "never".into() } else { v.join(" | ") }));
            }
            Named::Pending => {}
        }
    }
    out.push_str(ROUTES_TS_RUNTIME);
    out.push_str("export function routes(base = \"\", fetcher: typeof fetch = (u, i) => fetch(u, i)) {\n");
    out.push_str("  const r = runtime(base, fetcher);\n  return {\n");
    for rt in routes {
        let (text, tags) = route_doc(rt);
        ts_jsdoc(&mut out, "    ", &text, &tags);
        let fields = rt.request_fields();
        let req_ty = ts(&rt.request);
        let has_req = !fields.is_empty();
        let all_optional = fields.iter().all(|f| f.optional);
        let req_param = if !has_req { String::new() } else if all_optional { format!("req: {req_ty} = {{}}") } else { format!("req: {req_ty}") };
        let req_arg = if has_req { "req" } else { "{}" };
        let path = format!("{:?}", rt.path);
        let query = if rt.body == Body::Json { "false" } else { "true" };
        let url = format!("r.url({path}, {req_arg}, {query})");
        // the answer's type has a field `error` of its own: only `ok: false` says it failed
        let own_error = matches!(&rt.response, Shape::Ref(n) if object_fields(&defs, &Shape::Ref(n.clone())).iter().any(|f| f.name == "error"));
        let resp_ts = if rt.json_or_bytes() { format!("{} | Blob", ts(&rt.response)) } else { ts(&rt.response) };
        let bytes = rt.answers_bytes() || rt.json_or_bytes();
        let (params, call) = match rt.body {
            Body::None if rt.answers_bytes() && rt.method == "GET" => (req_param.clone(), format!("string => {url}")),
            Body::None if bytes => (req_param.clone(), format!("Promise<{resp_ts}> => r.blob({:?}, {url})", rt.method)),
            Body::None => (req_param.clone(), format!("Promise<{resp_ts}> => r.json({:?}, {url}, undefined, undefined, {own_error})", rt.method)),
            Body::Json => {
                let p = if has_req { req_param.clone() } else { format!("req: {req_ty} = {{}}") };
                let send = if bytes { format!("Promise<{resp_ts}> => r.blob({:?}, {url}, JSON.stringify(req), \"application/json\")", rt.method) } else {
                    format!("Promise<{resp_ts}> => r.json({:?}, {url}, JSON.stringify(req), \"application/json\", {own_error})", rt.method)
                };
                (p, send)
            }
            Body::JsonOrForm => {
                let form = rt.form.as_ref().map(ts).unwrap_or_else(|| "Record<string, unknown>".into());
                let what = format!("body: {form} | FormData");
                let p = if has_req { format!("{what}, {req_param}") } else { what };
                (p, format!("Promise<{resp_ts}> => r.either({:?}, {url}, body, {own_error})", rt.method))
            }
            Body::Multipart | Body::Bytes => {
                let what = if rt.body == Body::Multipart { "form: FormData" } else { "data: Blob | ArrayBuffer | Uint8Array" };
                let arg = if rt.body == Body::Multipart { "form" } else { "data" };
                let p = if has_req { format!("{what}, {req_param}") } else { what.to_string() };
                let send = if bytes { format!("Promise<{resp_ts}> => r.blob({:?}, {url}, {arg})", rt.method) } else {
                    format!("Promise<{resp_ts}> => r.json({:?}, {url}, {arg}, undefined, {own_error})", rt.method)
                };
                (p, send)
            }
        };
        out.push_str(&format!("    {}: ({params}): {call},\n", ts_key(rt.name)));
    }
    out.push_str("  };\n}\n");
    Ok(out)
}

const ROUTES_TS_RUNTIME: &str = r#"/** A route's failure: the words its body's `error` said, and the body. */
export class RouteError extends Error {
  constructor(message: string, readonly body: unknown) {
    super(message);
  }
}

function runtime(base: string, fetcher: typeof fetch) {
  const words = (e: any): string => (typeof e === "string" ? e : (e?.message ?? JSON.stringify(e)));
  const read = async (res: Response, u: string, ownError: boolean): Promise<any> => {
    const text = await res.text();
    let v: any = null;
    try {
      v = text ? JSON.parse(text) : null;
    } catch {
      throw new RouteError(`${u}: the answer is not JSON (HTTP ${res.status})`, text);
    }
    const failed = v !== null && typeof v === "object" && !Array.isArray(v) && v.error != null && (!ownError || v.ok === false);
    if (failed) throw new RouteError(words(v.error), v);
    return v;
  };
  return {
    url(path: string, req: object, query: boolean): string {
      const fields = req as Record<string, unknown>;
      const used = new Set<string>();
      let p = path.replace(/\{([^}]+)\}/g, (_, k: string) => {
        used.add(k);
        return encodeURIComponent(String(fields[k] ?? ""));
      });
      if (query) {
        const q = new URLSearchParams();
        for (const [k, v] of Object.entries(fields)) {
          if (used.has(k) || v === undefined || v === null) continue;
          for (const x of Array.isArray(v) ? v : [v]) q.append(k, String(x));
        }
        const s = q.toString();
        if (s) p += `?${s}`;
      }
      return base + p.slice(1);
    },
    async json(method: string, u: string, body?: BodyInit, type?: string, ownError = false): Promise<any> {
      const res = await fetcher(u, { method, body, headers: type ? { "content-type": type } : undefined });
      return read(res, u, ownError);
    },
    async either(method: string, u: string, body: unknown, ownError = false): Promise<any> {
      const form = body instanceof FormData;
      const res = await fetcher(u, { method, body: form ? body : JSON.stringify(body), headers: form ? undefined : { "content-type": "application/json" } });
      return read(res, u, ownError);
    },
    async blob(method: string, u: string, body?: BodyInit, type?: string): Promise<any> {
      const res = await fetcher(u, { method, body, headers: type ? { "content-type": type } : undefined });
      if ((res.headers.get("content-type") ?? "").startsWith("application/json")) return read(res, u, false);
      return res.blob();
    },
  };
}

"#;

fn facts_of_route_openapi(doc: &str) -> Result<Facts, String> {
    let v: Value = serde_yaml::from_str::<Value>(doc).map_err(|e| format!("the OpenAPI document does not parse: {e}"))?;
    let mut f = Facts::new();
    for (path, ops) in v["paths"].as_object().into_iter().flatten() {
        for (method, op) in ops.as_object().into_iter().flatten() {
            let name = format!("{} {path}", method.to_ascii_uppercase());
            let text = op["description"].as_str().or_else(|| op["summary"].as_str()).unwrap_or("");
            f.insert(format!("{name}: text: {text}"));
            for t in op["tags"].as_array().into_iter().flatten() {
                f.insert(format!("{name}: tag: {}", t.as_str().unwrap_or("")));
            }
            for a in op["x-aliases"].as_array().into_iter().flatten() {
                f.insert(format!("{name}: alias: {}", a.as_str().unwrap_or("")));
            }
            for ct in op["responses"]["200"]["content"].as_object().into_iter().flatten().map(|(k, _)| k) {
                if ct != "application/json" && ct != "application/octet-stream" {
                    f.insert(format!("{name}: produces: {ct}"));
                }
            }
            if op["responses"].get("206").is_some() {
                f.insert(format!("{name}: ranges"));
            }
            let content = &op["requestBody"]["content"];
            match (content.get("multipart/form-data").is_some(), content.get("application/json").is_some()) {
                (true, true) => {
                    f.insert(format!("{name}: body: json, multipart"));
                }
                (true, false) => {
                    f.insert(format!("{name}: body: multipart"));
                }
                _ => {}
            }
            if content.get("application/octet-stream").is_some() {
                f.insert(format!("{name}: body: bytes"));
            }
            if let Some(e) = content["application/json"].get("example") {
                f.insert(format!("{name}: example: {e}"));
            }
        }
    }
    schema_facts(&v, &mut f);
    Ok(f)
}

/// Whether the OpenAPI document and the TypeScript made from these routes say the same: every description, tag,
/// alias, body kind, media type, example and named type's fact in one is in the other.
pub fn route_parity(routes: &[RouteInfo]) -> Result<(), String> {
    let doc = routes_openapi(json!({"title": "parity", "version": "0"}), "/", routes)?;
    let ts = routes_typescript("parity", routes)?;
    let a = facts_of_route_openapi(&doc)?;
    let b = facts_of_typescript(&ts);
    let only_doc: Vec<&String> = a.difference(&b).collect();
    let only_ts: Vec<&String> = b.difference(&a).collect();
    if only_doc.is_empty() && only_ts.is_empty() {
        return Ok(());
    }
    let mut msg = String::from("the routes' OpenAPI document and TypeScript differ:\n");
    for x in only_doc.iter().take(40) {
        msg.push_str(&format!("  only in the document: {x}\n"));
    }
    for x in only_ts.iter().take(40) {
        msg.push_str(&format!("  only in the TypeScript: {x}\n"));
    }
    Err(msg)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(path: &'static str, body: Body) -> RouteInfo {
        let mut defs = Defs::default();
        let mut o = Object::default();
        o.field::<String>(&mut defs, "name", "the file", false);
        o.field::<Option<i64>>(&mut defs, "width", "", false);
        o.field::<Vec<String>>(&mut defs, "tag", "", true);
        defs.named("FileRef", "", o);
        RouteInfo {
            name: "file", method: if body == Body::Json { "POST" } else { "GET" }, path, aliases: vec!["/old/{name}"], summary: "A file", description: "A file",
            tags: vec![], body, request: Shape::Ref("FileRef".into()), form: None, response: Shape::Blob, defs, produces: vec!["image/png"], ranges: true, example: None,
        }
    }

    #[test]
    fn a_path_matches_its_template_in_any_case_and_decodes_its_parameters() {
        let r = info("/v1/Files/{name}", Body::None);
        assert_eq!(r.match_path("/v1/files/a%20b.png"), Some(vec![("name".into(), "a b.png".into())]));
        assert_eq!(r.match_path("/old/x"), Some(vec![("name".into(), "x".into())]));
        assert_eq!(r.match_path("/v1/files/"), None);
        assert_eq!(r.match_path("/v1/files/a/b"), None);
    }

    #[test]
    fn query_parameters_are_read_as_their_fields_types() {
        let r = info("/f/{name}", Body::None);
        let v = route_input(&r, &[("name".into(), "a".into())], "width=64&tag=x&tag=y+z&other=1", b"").unwrap();
        assert_eq!(v, json!({"name": "a", "width": 64, "tag": ["x", "y z"], "other": "1"}));
        let v = route_input(&r, &[("name".into(), "a".into())], "width=wide", b"").unwrap();
        assert_eq!(v["width"], "wide", "a value its type does not read is left for the decoder to refuse");
    }

    #[test]
    fn a_table_that_could_not_be_described_is_refused() {
        assert!(check_routes(&[info("/f/{name}", Body::None)]).is_ok());
        assert!(check_routes(&[info("/f/{nope}", Body::None)]).unwrap_err().contains("not a field"));
        assert!(check_routes(&[info("/f/{width}", Body::None)]).unwrap_err().contains("required"));
        assert!(check_routes(&[info("/f/{name}", Body::Json)]).unwrap_err().contains("JSON body"));
        assert!(check_routes(&[info("/f/{name}", Body::None), info("/F/{name}", Body::None)]).unwrap_err().contains("two routes"));
    }

    #[test]
    fn the_document_and_the_client_agree_and_parity_sees_a_drop() {
        let r = vec![info("/f/{name}", Body::None)];
        route_parity(&r).unwrap();
        let doc = routes_openapi(json!({"title": "t", "version": "0"}), "/", &r).unwrap();
        let ts = routes_typescript("t", &r).unwrap();
        assert!(ts.contains("file: (req: FileRef): string => r.url(\"/f/{name}\", req, true)"), "{ts}");
        let a = facts_of_route_openapi(&doc).unwrap();
        for (drop, what) in [("     * @ranges\n", "ranges"), ("     * @alias /old/{name}\n", "alias"), ("     * @produces image/png\n", "produces")] {
            assert!(ts.contains(drop), "{drop:?} is in {ts}");
            assert!(a.difference(&facts_of_typescript(&ts.replace(drop, ""))).any(|x| x.contains(what)), "dropping {what} is seen");
        }
    }

    #[test]
    fn a_checked_answer_names_where_it_differs() {
        #[derive(serde::Deserialize, Serialize)]
        struct Info {
            loaded: bool,
        }
        assert_eq!(Checked::<Info>::new(json!({"loaded": true})).0, json!({"loaded": true}));
        if cfg!(debug_assertions) {
            let v = Checked::<Info>::new(json!({"loaded": true, "busy": 1})).0;
            assert_eq!(v["code"], "answer_shape");
            assert!(v["error"].as_str().unwrap().contains("busy"), "{v}");
        }
    }
}
