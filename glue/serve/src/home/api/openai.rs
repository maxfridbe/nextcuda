//! The front door's OpenAI-compatible routes (`/v1/...`, behind the token, rpcgen `#[route]`): each to the switcher
//! or the kind's server for the model the request names, a file to where it is kept. The table is what `api` serves
//! after the token and maintenance checks, and what the OpenAPI document (`openapi`) lists of the enabled kinds. The
//! bodies are the kinds' servers' own (a route here passes the request on as it came, so its types are theirs) and
//! the answers theirs too.

use std::sync::Arc;

use rpcgen::{route, routes, RouteError, RouteReply, Rpc, Sent};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::super::{registry, Home};
use super::Fail;
use crate::audio::routes::{AudioSpeechBody, Sound, Transcript};
use crate::home::methods::{AudioTranscribeBody, ImageEditBody, ImageGenerateBody, Model3dGenerateBody, Model3dRigBody};
use crate::http::{self, Request};
use crate::image::routes::Pictures;
use crate::model3d::routes::Meshes;
use crate::page::{Cx, FileName};
use crate::serve::routes::ChatBody;

/// An enabled model, as OpenAI lists them (with its kind)
#[derive(Serialize, Deserialize, Rpc)]
pub struct DoorModel {
    pub id: String,
    pub object: String,
    pub owned_by: String,
    pub created: i64,
    /// llm, image or audio
    pub kind: String,
    /// its title
    pub name: Option<String>,
}

/// OpenAI's list: the enabled chat, image and audio models
#[derive(Serialize, Deserialize, Rpc)]
pub struct DoorModels {
    pub object: String,
    pub data: Vec<DoorModel>,
}

/// Whether a kind has a model enabled (409 otherwise)
fn need(h: &Arc<Home>, k: &str) -> Result<(), Fail> {
    if h.kinds_on(&h.models()).contains(&k) { Ok(()) } else { Err((409, format!("no {k} model is enabled"))) }
}

fn pass(cx: &mut Cx, t: &http::Target) -> Result<Sent<()>, Fail> {
    http::forward(&mut cx.s, t, &cx.req).map_err(|e| (502, e))?;
    Ok(Sent::new())
}

/// The enabled chat, image and audio models
#[route(get = "/v1/models", tags = "openai")]
fn models(h: &Arc<Home>, _cx: &mut Cx) -> Result<DoorModels, Fail> {
    let data = h.models().iter().filter(|m| m["enabled"] != false && matches!(registry::kind_of(m), "llm" | "image" | "audio")).map(|m| DoorModel {
        id: m["id"].as_str().unwrap_or("").into(), object: "model".into(), owned_by: "nextcuda".into(), created: 0,
        kind: registry::kind_of(m).into(), name: m["title"].as_str().map(str::to_string),
    }).collect();
    Ok(DoorModels { object: "list".into(), data })
}

/// Chat (stream: true for server-sent events; tools), to the chat model named (its X-Keep-Up-S header keeps it loaded that long after)
#[route(post = "/v1/chat/completions", tags = "openai", produces = "application/json, text/event-stream",
        example = r#"{"messages": [{"role": "user", "content": "Hello"}], "stream": false}"#)]
fn chat(h: &Arc<Home>, cx: &mut Cx, req: ChatBody) -> Result<Sent<Value>, Fail> {
    let model = req.model;
    // a card's model not downloaded is said as such (chat_target refuses or fetches it), not "none enabled"
    if !model.as_deref().is_some_and(|m| h.not_downloaded(m).is_some()) {
        need(h, "llm")?;
    }
    let t = h.chat_target(model.as_deref())?;
    let r = pass(cx, &t);
    if let Some(secs) = crate::switch::keep_up_s(&cx.req.head) {
        if let Some(id) = h.chat_loaded(&h.models()) {
            h.keep_chat(&id, secs);
        }
    }
    r.map(|_| Sent::new())
}

/// A picture, an edit or a sound's request to the kind's server for the model it names (a JSON field, or a multipart
/// part): a model not downloaded refused (409 model_not_downloaded) or, with NS_AUTO_PULL, fetched first; OpenAI's own
/// names (dall-e-3, tts-1 ...) mean "the kind's model"; speech with a voice or a recording and no model of ours named:
/// the speech model for it
fn to_kind(h: &Arc<Home>, cx: &mut Cx, kind: &str) -> Result<(), Fail> {
    let req: &Request = &cx.req;
    let ctype = http::header(&req.head, "content-type").unwrap_or_default();
    let named = if let Some(b) = ctype.split("boundary=").nth(1) {
        http::multipart(&req.body, b.trim_matches('"')).ok()
            .and_then(|ps| ps.into_iter().find(|p| p.name == "model").map(|p| String::from_utf8_lossy(&p.data).trim().to_string()))
    } else {
        serde_json::from_slice::<Value>(&req.body).ok().and_then(|v| v["model"].as_str().map(str::to_string))
    };
    let fetched = named.as_deref().filter(|n| h.not_downloaded(n).is_some()).map(str::to_string);
    if let Some(n) = &fetched {
        h.need_downloaded(n, None, true)?;
    }
    let all = h.models();
    if fetched.is_none() {
        need(h, kind)?;
    }
    let named = named.filter(|n| all.iter().any(|m| m["id"] == n.as_str()));
    let body: Value = serde_json::from_slice(&req.body).unwrap_or(Value::Null);
    let has = |k: &str| body[k].as_str().is_some_and(|v| !v.is_empty());
    let many = body["speakers"].as_array().is_some_and(|a| !a.is_empty());
    let id = if kind == "audio" && named.is_none() && (has("voice") || has("ref_audio") || many) {
        h.speech_model(&body, &all)?
    } else {
        h.pick(kind, &json!({"model": named}), &all)?
    };
    let t = h.ensure(&id)?;
    pass(cx, &t).map(|_| ())
}

/// Pictures (and ours: steps, seed, sampler, schedule, cfg, negative_prompt, loras), to the image model named
#[route(post = "/v1/images/generations", tags = "openai", example = r#"{"prompt": "a lighthouse at dusk", "size": "1024x1024"}"#)]
fn generations(h: &Arc<Home>, cx: &mut Cx, _req: ImageGenerateBody) -> Result<Sent<Pictures>, Fail> {
    to_kind(h, cx, "image").map(|_| Sent::new())
}

/// Edits: multipart (image, image[] ...) or JSON images, to the image model named
#[route(post = "/v1/images/edits", tags = "openai", body = "json, multipart", form = "ImageEditBody",
        example = r#"{"prompt": "make it night", "images": ["data:image/png;base64,..."]}"#)]
fn edits(h: &Arc<Home>, cx: &mut Cx) -> Result<Sent<Pictures>, Fail> {
    to_kind(h, cx, "image").map(|_| Sent::new())
}

/// A song (input: the lyrics, instructions: the music) or speech (input: the text; voice, language, instructions, ref_audio, ref_text); response_format wav or url
#[route(post = "/v1/audio/speech", tags = "openai", produces = "application/json, audio/wav",
        example = r#"{"instructions": "warm folk", "input": "[verse]\nRiver runs slow", "response_format": "url"}"#)]
fn speech(h: &Arc<Home>, cx: &mut Cx, _req: AudioSpeechBody) -> Result<Sent<Sound>, Fail> {
    to_kind(h, cx, "audio").map(|_| Sent::new())
}

/// A recording written down, to the recogniser (OpenAI's whisper-1, or no name: the one here)
#[route(post = "/v1/audio/transcriptions", tags = "openai")]
fn transcriptions(h: &Arc<Home>, cx: &mut Cx, req: AudioTranscribeBody) -> Result<Sent<Transcript>, Fail> {
    need(h, "audio")?;
    let all = h.models();
    let named = req.model.filter(|n| all.iter().any(|m| m["id"] == n.as_str()));
    let id = h.recogniser(&json!({"model": named}), &all)?;
    let t = h.ensure(&id)?;
    pass(cx, &t).map(|_| Sent::new())
}

/// A picture an answer links (data[].url of image.generate / image.edit); no byte ranges (always the whole file)
#[route(get = "/v1/images/files/{name}", tags = "files", produces = "image/png, video/mp4")]
fn image_file(h: &Arc<Home>, cx: &mut Cx, _req: FileName) -> Result<Sent, Fail> {
    pass(cx, &h.file_target("image")).map(|_| Sent::new())
}

/// A song or speech an answer links (data[].url of audio.generate / audio.speak); byte ranges (Range) supported
#[route(get = "/v1/audio/files/{name}", tags = "files", produces = "audio/wav", ranges = true)]
fn audio_file(h: &Arc<Home>, cx: &mut Cx, _req: FileName) -> Result<Sent, Fail> {
    pass(cx, &h.file_target("audio")).map(|_| Sent::new())
}

/// A mesh an answer links (data[].url of model3d.generate / model3d.rig), or its picture (.png) or its record (.json); from the directory (no 3D server need run)
#[route(get = "/v1/model3d/files/{name}", tags = "files", produces = "model/gltf-binary, image/png, application/json")]
fn model3d_file(h: &Arc<Home>, cx: &mut Cx, req: FileName) -> Result<Sent, Fail> {
    let name = req.name.as_str();
    let kind = match name.rsplit('.').next() {
        Some("glb") => "model/gltf-binary",
        Some("png") => "image/png",
        Some("json") => "application/json",
        _ => return Err((404, format!("no file {name}"))),
    };
    let plain = crate::files::plain(name).ok_or_else(|| (404, format!("no file {name}")))?;
    let b = std::fs::read(h.out_dir("model3d").join(plain)).map_err(|_| (404, format!("no file {name}")))?;
    cx.bytes(200, kind, &b, "Cache-Control: max-age=31536000, immutable\r\n");
    Ok(Sent::new())
}

/// The 3D model for a request: the one named (a card's not downloaded: refused, or with NS_AUTO_PULL fetched first),
/// else the first enabled that does it
fn to_model3d(h: &Arc<Home>, cx: &mut Cx, model: Option<String>, want: &str) -> Result<(), Fail> {
    let fetched = model.as_deref().filter(|n| h.not_downloaded(n).is_some()).map(str::to_string);
    if let Some(n) = &fetched {
        h.need_downloaded(n, None, true)?;
    } else {
        need(h, "model3d")?;
    }
    let all = h.models();
    let named = model.filter(|n| all.iter().any(|m| m["id"] == n.as_str()));
    let id = match named {
        Some(n) => h.pick("model3d", &json!({"model": n}), &all)?,
        None => all.iter().filter(|m| registry::kind_of(m) == "model3d" && m["enabled"] != false)
            .find(|m| m["does"].as_array().is_none_or(|a| a.iter().any(|x| x == want || (want == "rig" && x == "animate"))))
            .map(|m| m["id"].as_str().unwrap_or("").to_string()).ok_or_else(|| (409u16, format!("no enabled 3D model does {want}")))?,
    };
    let t = h.ensure(&id)?;
    pass(cx, &t).map(|_| ())
}

/// A mesh from a picture (image) or several views (images: [{image, side}]); texture, seed, steps, faces; response_format glb or url
#[route(post = "/v1/model3d/generate", tags = "openai", produces = "application/json, model/gltf-binary",
        example = r#"{"image": "data:image/png;base64,iVBORw0...", "texture": true, "response_format": "url"}"#)]
fn model3d_generate(h: &Arc<Home>, cx: &mut Cx, req: Model3dGenerateBody) -> Result<Sent<Meshes>, Fail> {
    to_model3d(h, cx, req.model, "generate").map(|_| Sent::new())
}

/// A mesh rigged (skeleton and skin weights) or animated (prompt: the motion); mesh as base64 GLB / OBJ, or file: a GLB made here
#[route(post = "/v1/model3d/rig", tags = "openai", produces = "application/json, model/gltf-binary",
        example = r#"{"file": "1792132541-trellis2-42.glb", "response_format": "url"}"#)]
fn model3d_rig(h: &Arc<Home>, cx: &mut Cx, req: Model3dRigBody) -> Result<Sent<Meshes>, Fail> {
    to_model3d(h, cx, req.model, "rig").map(|_| Sent::new())
}

/// A finished clip: <clip>.mp4 (video.jobs.list's file, video.status's and video.timeline's file URLs); also <clip>.preview.png (video.status's preview while it samples) and <clip>.last.png (a clip's last frame); byte ranges (Range) supported
#[route(get = "/v1/video/files/{name}", tags = "files", produces = "video/mp4, image/png", ranges = true)]
fn video_file(h: &Arc<Home>, cx: &mut Cx, req: FileName) -> Result<Sent, Fail> {
    // a clip in the studio's output (its /out/ route; a Range header passed on for seeking)
    need(h, "video")?;
    let name = req.name.as_str();
    let still = name.ends_with(".preview.png") || name.ends_with(".last.png");
    if name.is_empty() || name.contains('/') || !(name.ends_with(".mp4") || still) {
        return Err((404, format!("no clip {name}")));
    }
    let head = String::from_utf8_lossy(&cx.req.head).into_owned();
    let range = head.split("\r\n").find(|l| l.split(':').next().is_some_and(|k| k.trim().eq_ignore_ascii_case("range")));
    let up = format!("GET /out/{name} HTTP/1.1\r\nHost: studio\r\n{}Connection: close\r\n\r\n", range.map(|r| format!("{r}\r\n")).unwrap_or_default());
    let req = Request { method: "GET".into(), path: format!("/out/{name}"), target: format!("/out/{name}"), head: up.into_bytes(),
                        origin: None, host: None, body: Vec::new() };
    http::forward(&mut cx.s, &h.target("video"), &req).map_err(|e| (502, e))?;
    Ok(Sent::new())
}

/// A finished clip's 240-pixel still, 5 s in (video.timeline's thumb)
#[route(get = "/v1/video/thumbs/{name}", tags = "files", produces = "image/webp")]
fn video_thumb(h: &Arc<Home>, cx: &mut Cx, req: FileName) -> Result<Sent, Fail> {
    need(h, "video")?;
    let name = req.name.as_str();
    if name.is_empty() || name.contains('/') || !name.ends_with(".webp") {
        return Err((404, format!("no still {name}")));
    }
    let req = Request { method: "GET".into(), path: format!("/thumb/{name}"), target: format!("/thumb/{name}"),
                        head: format!("GET /thumb/{name} HTTP/1.1\r\nHost: studio\r\nConnection: close\r\n\r\n").into_bytes(),
                        origin: None, host: None, body: Vec::new() };
    http::forward(&mut cx.s, &h.target("video"), &req).map_err(|e| (502, e))?;
    Ok(Sent::new())
}

/// A saved voice's sample (audio.voices.list's sample)
#[route(get = "/v1/audio/voices/{name}", tags = "files", produces = "audio/wav")]
fn voice_sample(h: &Arc<Home>, cx: &mut Cx, req: FileName) -> Result<Sent, Fail> {
    let name = req.name.trim_end_matches(".wav");
    let b = h.voices().sample_path(name).and_then(|f| std::fs::read(f).ok()).ok_or_else(|| (404, format!("no saved voice {name}")))?;
    cx.bytes(200, "audio/wav", &b, "Cache-Control: no-cache\r\n");
    Ok(Sent::new())
}

routes![sync Arc<Home>, Cx, Fail; models, chat, generations, edits, speech, transcriptions, image_file, audio_file, model3d_file,
        model3d_generate, model3d_rig, video_file, video_thumb, voice_sample];

/// The table, made once
pub(crate) fn table() -> &'static [rpcgen::SyncRoute<Arc<Home>, Cx, Fail>] {
    static T: std::sync::OnceLock<Vec<rpcgen::SyncRoute<Arc<Home>, Cx, Fail>>> = std::sync::OnceLock::new();
    T.get_or_init(routes)
}

/// The routes, described
pub(crate) fn infos() -> Vec<rpcgen::RouteInfo> {
    routes().into_iter().map(|r| r.info).collect()
}

/// The kind a route is for (the document lists a kind's routes while a model of it is enabled); none: always listed
pub(crate) fn kind_of(path: &str) -> Option<&'static str> {
    [("/v1/chat/", "llm"), ("/v1/images/", "image"), ("/v1/audio/", "audio"), ("/v1/model3d/", "model3d"), ("/v1/video/", "video")]
        .iter().find(|(p, _)| path.starts_with(p)).map(|(_, k)| *k)
}

/// Serves an OpenAI-compatible request: Err with the status and words when it fails (the caller words it as OpenAI
/// does), a route not here among them. A JSON body is checked against its route's type and passed on as it came.
pub(crate) fn serve(h: &Arc<Home>, cx: &mut Cx) -> Result<(), Fail> {
    let Some((r, params)) = rpcgen::find_route(table(), &cx.req.method, &cx.req.path) else {
        return Err((404, format!("no route {} {} (GET /openapi.yml)", cx.req.method, cx.req.path)));
    };
    let query = cx.req.target.split_once('?').map(|q| q.1.to_string()).unwrap_or_default();
    let body = if r.info.body == rpcgen::Body::Json { cx.req.body.clone() } else { Vec::new() };
    match r.call(h, cx, params, &query, &body) {
        Ok(RouteReply::Json(v)) => {
            http::respond(&mut cx.s, 200, &v);
            Ok(())
        }
        Ok(RouteReply::Sent) => Ok(()),
        Err(RouteError::Payload(b)) => Err((400, b.0)),
        Err(RouteError::Answer(m)) => Err((500, m)),
        Err(RouteError::Handler(f)) => Err(f),
    }
}
