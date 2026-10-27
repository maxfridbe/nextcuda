//! The audio server's routes (rpcgen `#[route]`): what it serves, what wfe/audio calls (`src/gen/routes.ts`) and
//! what docs/api/audio.openapi.yml lists, all from this table. The OpenAI-compatible ones take the bodies the front
//! door's methods relay (home/methods.rs: audio.generate's AudioGenerateBody, audio.speak's AudioSpeakBody, the job's
//! options aside); /v1/audio/speech makes a song or speech, as the model is, so it takes both (AudioSpeechBody). The
//! answers are typed here and each is held to its type as it is sent (`Checked`).

use std::sync::atomic::Ordering;

use rpcgen::{route, routes, Checked, Rpc, Sent};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use super::{err, AudioServer};
use crate::home::methods::{AudioTranscribeBody, SongFields, SpeechFields};
use crate::page::{CancelReq, Cancelled, Cx, Fail, FileName, Gpu, Health, ModelList};

/// The engine's own defaults (a request's absent fields)
#[derive(Serialize, Deserialize, Rpc)]
pub struct Defaults {
    pub seconds: f64,
    /// the longest it makes
    pub max_seconds: f64,
    pub steps: i64,
    pub cfg: f64,
    /// samples a second of what it returns
    pub rate: i64,
}

/// An option of the engine's own (a request sends it in `options`)
#[derive(Serialize, Deserialize, Rpc)]
pub struct EngineOptionInfo {
    pub name: String,
    /// what its value looks like ("" a flag)
    pub value: String,
    pub help: String,
}

/// What a recogniser writes down
#[derive(Serialize, Deserialize, Rpc)]
pub struct RecogniserInfo {
    /// the languages it writes down (codes)
    pub languages: Vec<String>,
    /// it times the words
    pub words: bool,
    /// it translates into English
    pub translate: bool,
}

/// What a speech engine takes
#[derive(Serialize, Deserialize, Rpc)]
pub struct SpeechInfo {
    /// its built-in voices
    pub voices: Vec<String>,
    /// the languages it speaks (besides auto)
    pub languages: Vec<String>,
    /// it styles the speech by instructions
    pub instructions: bool,
    /// it makes a voice from instructions alone
    pub design: bool,
    /// it clones a recording
    pub clone: bool,
    pub clone_needs_text: bool,
    /// a transcript of the recording makes a closer clone
    pub clone_takes_text: bool,
    /// it styles a cloned voice by instructions (voice direction)
    pub direction: bool,
    /// the vocal events it reads inline ("(laugh)", "[笑]" ...)
    pub events: Vec<String>,
    /// what a request must carry besides the text: voice, instructions or reference; none: nothing
    pub requires: Option<String>,
    /// the speakers a script may have, each its own saved voice ("Speaker 1: ...", "Speaker 2: ...": VibeVoice); 0: one voice
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speakers: Option<i64>,
}

/// The model and what it takes; while a request runs, the model and `busy` only
#[derive(Serialize, Deserialize, Rpc)]
pub struct Info {
    pub model: String,
    /// the engine is running a request: what changes is not reported
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub busy: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arch: Option<String>,
    /// it sings words
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lyrics: Option<bool>,
    /// `cfg` changes the sound (false: a distilled checkpoint runs unguided)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub guidance: Option<bool>,
    /// it serves its page
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wfe: Option<bool>,
    /// a recogniser: what it writes down (none: it makes sound)
    #[serde(default)]
    pub recogniser: Option<RecogniserInfo>,
    /// a speech engine: its voices and what it takes (none: songs)
    #[serde(default)]
    pub speech: Option<SpeechInfo>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub defaults: Option<Defaults>,
    /// the engine's own per-request options (sent as `options`)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub options: Option<Vec<EngineOptionInfo>>,
    /// what the engine says of itself (its files, its precision ...)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub report: Option<Vec<String>>,
    /// it keeps what it makes (an output directory: links, the history, saved voices)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub saves: Option<bool>,
}

/// What the server is doing: busy false, or what it is making and how far it is
#[derive(Serialize, Deserialize, Rpc)]
pub struct Progress {
    pub busy: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    /// the engine's: prompt, tokens, flow, decode (a song); listen ... (a transcript)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,
    /// the units done (frames, steps, windows) of `of`
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub at: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub of: Option<i64>,
    /// seconds since it started
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seconds: Option<f64>,
    /// the length asked for (seconds of sound)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub want: Option<f64>,
    /// requests waiting for the engine
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub waiting: Option<i64>,
    /// the front door's job running, if it named one
    #[serde(default)]
    pub request_id: Option<String>,
}

/// A song or speech made here: its link, what it was made from (the WAV's own tags) and how
#[derive(Serialize, Deserialize, Rpc)]
pub struct SoundItem {
    /// the WAV (GET it here; byte ranges)
    pub url: String,
    /// a song: its description; speech: the text said
    pub prompt: String,
    /// a song's lyrics ("" none)
    pub lyrics: String,
    /// seconds of sound
    pub seconds: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steps: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cfg: Option<f64>,
    /// seconds it took to make
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub took: Option<f64>,
    /// the card's energy, watt-hours
    #[serde(default)]
    pub wh: Option<f64>,
    /// unix seconds
    pub created: i64,
    /// speech: the voice (built-in or saved)
    #[serde(default)]
    pub voice: Option<String>,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub instructions: Option<String>,
    /// speech: spoken in a recording's voice
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cloned: Option<bool>,
    /// speech: the file it carried on from (continue_from)
    #[serde(default)]
    pub continued: Option<String>,
    /// what else the file's settings record (a file made by an older server)
    #[serde(flatten)]
    pub rest: Map<String, Value>,
}

/// The songs and speech in the output directory, newest first
#[derive(Serialize, Deserialize, Rpc)]
pub struct History {
    pub data: Vec<SoundItem>,
}

/// How a sound was made
#[derive(Serialize, Deserialize, Rpc)]
pub struct SoundRun {
    pub took: f64,
    /// seconds of sound
    pub seconds: f64,
    pub seed: i64,
    pub steps: i64,
    pub cfg: f64,
    pub wh: Option<f64>,
}

/// A sound made (response_format url): its link and settings, and how it was made (`nextsycl`: the key nextsycl's
/// contract names it; the front door's methods call it run)
#[derive(Serialize, Deserialize, Rpc)]
pub struct Sound {
    pub created: i64,
    pub data: Vec<SoundItem>,
    pub nextsycl: SoundRun,
}

/// A stretch or a word of a transcript, timed
#[derive(Serialize, Deserialize, Rpc)]
pub struct Timed {
    /// seconds from the recording's start
    pub start: f64,
    pub end: f64,
    pub text: String,
    /// a word: the model's mean probability of its tokens
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub probability: Option<f64>,
}

/// How a transcript was made
#[derive(Serialize, Deserialize, Rpc)]
pub struct TranscriptRun {
    pub took: f64,
    pub wh: Option<f64>,
    /// seconds of recording
    pub seconds: f64,
}

/// A recording written down
#[derive(Serialize, Deserialize, Rpc)]
pub struct Transcript {
    pub model: String,
    /// the whole transcript
    pub text: String,
    /// the language written down (a code: en), detected or as asked
    pub language: String,
    /// seconds of recording
    pub duration: f64,
    /// stretches timed by the model's timestamp tokens
    pub segments: Vec<Timed>,
    /// each word timed (empty with words: false)
    pub words: Vec<Timed>,
    pub created: i64,
    pub nextsycl: TranscriptRun,
}

/// A saved voice (the voices library beside the output directory)
#[derive(Serialize, Deserialize, Rpc)]
pub struct SavedVoice {
    pub name: String,
    /// what the sample says (none: an x-vector clone)
    #[serde(default)]
    pub text: Option<String>,
    /// a designed voice's description
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub language: Option<String>,
    /// clone or design
    pub kind: String,
    /// the sample's length
    pub seconds: f64,
    /// its samples a second
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rate: Option<i64>,
    pub created: i64,
    /// where it came from: a recording, made here (and by which model)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub from: Option<String>,
    /// what else its record says (its owner ...)
    #[serde(flatten)]
    pub rest: Map<String, Value>,
}

/// The saved voices
#[derive(Serialize, Deserialize, Rpc)]
pub struct Voices {
    pub data: Vec<SavedVoice>,
}

/// A voice to save: a recording sent, or a file made here
#[derive(Serialize, Deserialize, Rpc)]
pub struct VoiceAdd {
    /// lowercase letters, digits, - and _ (a letter first, at most 40)
    pub name: String,
    /// the recording: any format, base64 or a data: URL (1 s at least; 3-15 s is best; past 60 s it is cut)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample: Option<String>,
    /// or a WAV made here (its name or url): what it says, its voice's description and language come with it
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    /// what the recording says (else a file's own)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// over a voice of that name
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replace: Option<bool>,
}

/// A voice to remove
#[derive(Serialize, Deserialize, Rpc)]
pub struct VoiceRemove {
    pub name: String,
}

/// A voice removed
#[derive(Serialize, Deserialize, Rpc)]
pub struct VoiceRemoved {
    pub removed: String,
}

/// A saved voice's sample (`.../voices/{name}`, with or without .wav)
#[derive(Serialize, Deserialize, Rpc)]
pub struct VoiceName {
    pub name: String,
}

/// What /v1/audio/speech takes: a song (a song model: input the lyrics, instructions the music) or speech (a speech
/// model: input the text; voice, language, instructions, ref_audio, ref_text ...), as the model this server runs is
#[derive(Serialize, Deserialize, Rpc)]
pub struct AudioSpeechBody {
    /// a song: the lyrics ([verse] / [chorus] tags on lines of their own; empty: instrumental); speech: the text to say (required; also as text)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<String>,
    /// a song: the music - genre, mood, tempo, instruments, voice (required unless a recording is given; also as prompt or description); speech: how to say it, the voice to make up, or a cloned voice's direction
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    /// guidance (also as guidance_scale)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cfg: Option<f64>,
    /// a song: the lyrics' language as a code; speech: the language spoken (auto by default)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// the model's id (this server's; OpenAI's and the reference servers' names are taken for it)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// the engine's own options, {NAME: value} (/api/info lists them)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub options: Option<Map<String, Value>>,
    /// a song: its length; speech: the most to make (also as duration, or max_new_tokens: frames)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seconds: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    pub seed: Option<i64>,
    /// also as num_inference_steps
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "rpcgen::de::opt_int")]
    pub steps: Option<i64>,
    /// wav (default: the file) or url (JSON with its link; needs an output directory)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,
    /// speech: false says the text as written (default: symbols as words, no markdown)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normalize: Option<bool>,
    #[serde(flatten)]
    pub song: SongFields,
    #[serde(flatten)]
    pub speech: SpeechFields,
    /// what else the caller sent: the other names a field is taken by, request_id
    #[serde(flatten)]
    pub rest: Map<String, Value>,
}

/// The card: its name, memory, power and temperatures
#[route(get = "/api/gpu", tags = "page")]
fn gpu(s: &AudioServer, _cx: &mut Cx) -> Result<Gpu, Fail> {
    Ok(s.gpu_json())
}

/// Whether the server answers
#[route(get = "/health", tags = "page")]
fn health(_s: &AudioServer, _cx: &mut Cx) -> Result<Health, Fail> {
    Ok(Health { status: "ok".into() })
}

/// The model served, as OpenAI lists models
#[route(get = "/v1/models", alias = "/models", tags = "openai")]
fn models(s: &AudioServer, _cx: &mut Cx) -> Result<ModelList, Fail> {
    Ok(ModelList::of(&s.model))
}

/// The model and what it takes: its defaults, options, a speech engine's voices, a recogniser's languages
#[route(get = "/api/info", tags = "page")]
fn info(s: &AudioServer, _cx: &mut Cx) -> Result<Checked<Info>, Fail> {
    Ok(Checked::new(s.info()))
}

/// What the server is doing now (the page polls it)
#[route(get = "/api/progress", tags = "page")]
fn progress(s: &AudioServer, _cx: &mut Cx) -> Result<Checked<Progress>, Fail> {
    let mut p = s.progress.lock().unwrap().clone();
    p["waiting"] = json!(s.waiting.load(Ordering::Relaxed));
    p["request_id"] = json!(s.cancel.current());
    Ok(Checked::new(p))
}

/// The songs and speech made here, newest first (their settings from the WAVs)
#[route(get = "/api/history", tags = "page")]
fn history(s: &AudioServer, _cx: &mut Cx) -> Result<Checked<History>, Fail> {
    // files deleted since (system.files.delete) drop out
    let mut h = s.history.lock().unwrap();
    if let Some(d) = &s.out_dir {
        h.retain(|e| e["url"].as_str().and_then(|u| u.rsplit('/').next()).is_none_or(|f| d.join(f).exists()));
    }
    Ok(Checked::new(json!({"data": h.iter().collect::<Vec<_>>()})))
}

/// Stops a request: the one named (running or waiting), else the one running (between frames or steps)
#[route(post = "/api/cancel", tags = "page")]
fn cancel(s: &AudioServer, _cx: &mut Cx, req: CancelReq) -> Result<Cancelled, Fail> {
    let busy = s.progress.lock().unwrap()["busy"] == true;
    let running = s.cancel.request(req.id.as_deref(), busy);
    Ok(Cancelled { cancelled: running, id: req.id })
}

/// A song or speech an answer links, whole or the byte range asked for (the player seeks)
#[route(get = "/v1/audio/files/{name}", tags = "files", produces = "audio/wav", ranges = true)]
fn file(s: &AudioServer, cx: &mut Cx, req: FileName) -> Result<Sent, Fail> {
    s.file(cx, &req.name);
    Ok(Sent::new())
}

/// The saved voices (none without an output directory)
#[route(get = "/api/voices", tags = "voices")]
fn voices(s: &AudioServer, _cx: &mut Cx) -> Result<Checked<Voices>, Fail> {
    Ok(Checked::new(json!({"data": s.voices().map(|l| l.list()).unwrap_or_default()})))
}

/// A saved voice's sample
#[route(get = "/v1/audio/voices/{name}", tags = "voices", produces = "audio/wav")]
fn voice_sample(s: &AudioServer, cx: &mut Cx, req: VoiceName) -> Result<Sent, Fail> {
    let name = req.name.trim_end_matches(".wav");
    match s.voices().and_then(|l| l.sample_path(name)).and_then(|p| std::fs::read(p).ok()) {
        Some(b) => cx.bytes(200, "audio/wav", &b, "Cache-Control: no-cache\r\n"),
        None => cx.json(404, &err(format!("no saved voice {name}"))),
    }
    Ok(Sent::new())
}

/// Saves a voice: a recording sent, or a file made here (its text, description and language with it)
#[route(post = "/api/voices/add", tags = "voices")]
fn voice_add(s: &AudioServer, _cx: &mut Cx, req: VoiceAdd) -> Result<Checked<SavedVoice>, Fail> {
    let b = serde_json::to_value(&req).map_err(|e| (500, e.to_string()))?;
    s.voice_op("/api/voices/add", &b).map(Checked::new).map_err(|m| (400, m))
}

/// Removes a saved voice
#[route(post = "/api/voices/remove", tags = "voices")]
fn voice_remove(s: &AudioServer, _cx: &mut Cx, req: VoiceRemove) -> Result<Checked<VoiceRemoved>, Fail> {
    let b = serde_json::to_value(&req).map_err(|e| (500, e.to_string()))?;
    s.voice_op("/api/voices/remove", &b).map(Checked::new).map_err(|m| (400, m))
}

/// A recording written down (a recogniser: whisper): its text, language, timed stretches and words (JSON only, no multipart form)
#[route(post = "/v1/audio/transcriptions", alias = "/audio/transcriptions", tags = "openai")]
fn transcriptions(s: &AudioServer, cx: &mut Cx, req: AudioTranscribeBody) -> Result<Sent<Transcript>, Fail> {
    let b = serde_json::to_value(&req).map_err(|e| (500, e.to_string()))?;
    // the answer first: retiring ends the process
    match s.transcribe(&b) {
        Ok(v) => cx.json(200, &Checked::<Transcript>::new(v).0),
        Err((code, m)) => cx.json(code, &err(m)),
    }
    s.maybe_retire();
    Ok(Sent::new())
}

/// A song or speech, as the model is (the reference server's speech endpoint): the WAV, or with response_format url JSON with its link
#[route(post = "/v1/audio/speech", alias = "/audio/speech", tags = "openai", produces = "application/json, audio/wav")]
fn speech(s: &AudioServer, cx: &mut Cx, req: AudioSpeechBody) -> Result<Sent<Sound>, Fail> {
    let b = serde_json::to_value(&req).map_err(|e| (500, e.to_string()))?;
    let base = cx.base();
    match s.generate(&b, &base) {
        Ok((Some(v), _)) => cx.json(200, &Checked::<Sound>::new(v).0),
        Ok((None, wav)) => cx.bytes(200, "audio/wav", &wav, ""),
        Err((code, m)) => cx.json(code, &err(m)),
    }
    s.maybe_retire();
    Ok(Sent::new())
}

routes![sync AudioServer, Cx, Fail; gpu, health, models, info, progress, history, cancel, file, voices, voice_sample, voice_add, voice_remove,
        transcriptions, speech];

/// The table, made once
pub fn table() -> &'static [rpcgen::SyncRoute<AudioServer, Cx, Fail>] {
    static T: std::sync::OnceLock<Vec<rpcgen::SyncRoute<AudioServer, Cx, Fail>>> = std::sync::OnceLock::new();
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
