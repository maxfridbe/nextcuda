//! What the enabled models do: their state, the memory they are vouched for, enabling and disabling, loading on
//! demand and unloading.
//!
//! - chat (llm): the video studio's llm.mode loads one at a time (it knows when a render holds the card); the switcher
//!   loads on Open WebUI's request and unloads after the idle time; enabled = listed by the switcher. A box with
//!   neither (the front door alone): this page starts the chat server itself (`nextcuda llm start ID`) for the model
//!   a request names, stopping the one before it.
//! - image, audio, model3d: a server each (`nextcuda image|audio|model3d start ID --gpu N --idle-exit S`), started on the first
//!   request for an enabled model on the first of its cards with room, swapped when a request names another (after the
//!   one running), ending itself when idle. A second (`--slot 1`, NS_IMAGE_PORT2 / NS_AUDIO_PORT2) holds another model,
//!   or a second copy of a model enabled on two cards: started when a request finds its copy with work and its other
//!   card with room; requests go to the copy without work, else the one with fewer. A start waits for the card to have the model's memory free (1.5 GiB to spare: past that the xe driver
//!   spills into host memory and the box can hang), and answers 503 no_room when it does not come free.
//! - video: the daemon (`nextcuda video start`) with the enabled video models as its engines, on their GPUs; its
//!   workers load per clip and let the card go when idle. No video model enabled: the daemon is stopped. A video
//!   model enabled and the daemon not running (or running without its engine): the next video request through the
//!   front door, `video.engines.list` included, applies the daemon first and waits for it (`video_ensure`), and a
//!   watcher brings back one that crashed (`keeper`).
//! - concurrency profile (the model card's `concurrency` / `coexistence`, crates/core profile.rs): never more copies of a
//!   model on a card than `max_copies_per_card`, extra copies (when requests wait) only up to `recommended_copies`;
//!   an `exclusive` model has its card to itself - idle residents are stopped for it, nothing else is placed there
//!   while it is loaded, and a busy, pinned or held resident makes it wait / answer no_room naming it; `shared_with`
//!   limits what may be a resident beside a model. Unmeasured models: one copy, shared with anything that fits.
//! - pinned (the page's pin / unpin; `pinned` in the registry entry beside `enabled`): a model loaded now and kept
//!   there - its server starts without `--idle-exit`, the keeper loads it again after a restart and keeps a running
//!   server touched, the video daemon runs with no worker idle unload, the switcher does not unload it, and no room
//!   plan stops it for another model (that model then answers 503 no_room, naming the pin). Unpinning only clears
//!   the flag: the model stays loaded until the idle rules take it. Disabling a model unpins it. There is no start
//!   button: an enabled model is loaded by its first request; `load` / `unload` stay as RPC actions for tools.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use super::{now, Home};
use nextcuda_core::profile::{self, Profile};
use crate::http::{call_for, Target};
use super::registry;

/// Memory to keep free on a card beyond a model's (the xe driver has no out-of-memory: past the card it spills)
const MARGIN_MB: u64 = 1536;

/// How long a request waits for a card to have room before 503 no_room (a job waits longer: JOB_ROOM_WAIT_S, its
/// status saying what for)
const ROOM_WAIT_S: u64 = 90;
const JOB_ROOM_WAIT_S: u64 = 1800;


/// Idle work that holds memory on a card and can go for a model that needs the room
struct Holder {
    what: Free,
    /// its memory there
    mb: u64,
    model: String,
    /// which go first: picture and sound servers (0), the video worker (1), a chat model (2)
    order: u8,
}

enum Free {
    /// an image or audio server: (kind, slot)
    Server(String, usize),
    /// the video daemon's worker on this card
    Video(usize),
    /// the chat model
    Chat(String),
}

/// The video daemon's engines (model ids), and (engine, GPU) per loaded worker
type VideoDaemon = (Vec<String>, Vec<(String, usize)>);

/// How long a request waits to start: the caller's `max_willing_to_wait_s` when it gave one (it replaces the default,
/// for a job's 30 minutes too, and enforces itself: wait_too_long), else the default
fn wait_limit(default: Duration, max_wait: Option<f64>) -> Duration {
    max_wait.map_or(default, |_| Duration::MAX)
}

/// How long a video request waits for the daemon to come up before 503
const VIDEO_WAIT_S: u64 = 120;

/// The models kept: pinned, or held (a hold that has not run out at `now`)
fn kept_from(all: &[Value], holds: &BTreeMap<String, super::rsvp::Hold>, now: f64) -> BTreeSet<String> {
    let mut v = pinned_ids(all);
    v.extend(holds.iter().filter(|(_, h)| h.until > now).map(|(id, _)| id.clone()));
    v
}

/// The ids of the pinned models
fn pinned_ids(all: &[Value]) -> BTreeSet<String> {
    all.iter().filter(|m| registry::pinned(m)).map(|m| s(m, "id")).collect()
}

/// The holders a room plan may stop: not a pinned model's
fn evictable(holders: Vec<Holder>, pinned: &BTreeSet<String>) -> Vec<Holder> {
    holders.into_iter().filter(|h| !pinned.contains(&h.model)).collect()
}

/// The arguments that end a server after `idle_minutes` idle: none for a pinned model (or when idle exit is off)
fn idle_exit_args(idle_minutes: u64, pinned: bool) -> Vec<String> {
    if idle_minutes > 0 && !pinned {
        vec!["--idle-exit".into(), (idle_minutes * 60).to_string()]
    } else {
        Vec::new()
    }
}

/// The enabled video models' ids, the registry's `enabled` (missing: on)
fn video_enabled(all: &[Value]) -> Vec<String> {
    all.iter().filter(|m| registry::kind_of(m) == "video" && m["enabled"] != false).map(|m| s(m, "id")).collect()
}

/// The daemon's first model and the others (`--engine`), from the enabled ones: minimax-h3 first if enabled
fn video_engines(enabled: &[String]) -> (String, Vec<String>) {
    let first = enabled.iter().find(|i| *i == "minimax-h3").or(enabled.first()).cloned().unwrap_or_default();
    let rest = enabled.iter().filter(|i| **i != first).cloned().collect();
    (first, rest)
}

/// Whether a video request has to apply the daemon first: an enabled model's engine is not there (the daemon not
/// running: `running` none)
fn video_missing(enabled: &[String], running: Option<&[String]>) -> bool {
    !enabled.is_empty() && running.is_none_or(|r| enabled.iter().any(|e| !r.contains(e)))
}

/// Whether something listens on 127.0.0.1:`port`
fn listening(port: u16) -> bool {
    std::net::TcpStream::connect_timeout(&std::net::SocketAddr::from(([127, 0, 0, 1], port)), Duration::from_millis(300)).is_ok()
}

/// An image, audio or 3D model: it loads whole on one card (the first of its cards with room)
fn one_card(m: &Value) -> bool {
    matches!(registry::kind_of(m), "image" | "audio" | "model3d")
}

/// A chat model whose model card spreads one copy over every card (`gpus: all`; its title says "both GPUs"): it
/// does not fit on fewer, so its cards are not a choice (an image or audio model's `all` is where its copies may go)
pub(crate) fn spans_all(m: &Value, model_cards: &[Value]) -> bool {
    registry::kind_of(m) == "llm"
        && model_cards.iter().find(|c| [s(m, "catalog"), s(m, "id")].iter().any(|id| !id.is_empty() && c["id"] == id.as_str()))
            .is_some_and(|c| s(c, "gpus") == "all")
}

/// The cards processes of model `id` hold memory on, in the named cards
fn held_on(cards: &Value, id: &str) -> Vec<usize> {
    cards.as_array().into_iter().flatten().filter(|c| c["procs"].as_array().into_iter().flatten().any(|p| p["model_id"] == id))
        .filter_map(|c| c["index"].as_u64()).map(|g| g as usize).collect()
}

/// The card the server listening on `port` is on (its process's --port), in the named cards
fn card_of_port(cards: &Value, port: u16) -> Option<usize> {
    let port = port.to_string();
    cards.as_array()?.iter().find(|c| c["procs"].as_array().into_iter().flatten().any(|p| p["port"].as_str() == Some(port.as_str())))
        .and_then(|c| c["index"].as_u64()).map(|g| g as usize)
}

/// One of the kind's servers: its slot (`--slot`), port, model (when one answers), and whether it loads or has work
pub(crate) struct Slot {
    slot: usize,
    port: u16,
    model: Option<String>,
    loading: bool,
    busy: bool,
}

/// A request's server, chosen by `ensure`: counted as having work until dropped (after the answer)
pub(crate) struct Lease<'a> {
    target: Target,
    pub(crate) port: u16,
    home: &'a Home,
    /// the model served and the keep-alive granted the request (rsvp_s): its hold starts when this is dropped, at
    /// the end of the request's work - answered, failed or cancelled
    hold: Option<(String, f64)>,
}

impl std::ops::Deref for Lease<'_> {
    type Target = Target;
    fn deref(&self) -> &Target {
        &self.target
    }
}

impl Drop for Lease<'_> {
    fn drop(&mut self) {
        if let Some(n) = self.home.inflight.lock().unwrap().get_mut(&self.port) {
            *n = n.saturating_sub(1);
        }
        if let Some((id, secs)) = self.hold.take() {
            self.home.hold_after(&id, secs);
        }
    }
}

/// What a model can do, in neutral words (a client picks a model by them, on any box): its tags, and the /rpc
/// methods that take it
pub(crate) fn capabilities(m: &Value) -> (Vec<&'static str>, Vec<&'static str>) {
    let arch = s(m, "arch");
    let speech = |c: &str| m["speech"].as_array().is_some_and(|a| a.iter().any(|x| x == c));
    match registry::kind_of(m) {
        "llm" => {
            let mut c = vec!["chat", "stream"];
            if m["tools"] == true {
                c.push("tools");
            }
            (c, vec!["llm.chat"])
        }
        "image" if arch.contains("ocr") => (vec!["ocr", "read_pages", "pdf"], vec!["image.ocr"]),
        "image" if arch == "realesrgan" || arch == "waifu2x" => (vec!["upscale", "upscale_video", "tiles"], vec!["image.upscale", "video.upscale"]),
        "image" if arch == "starvector" => (vec!["vectorize", "svg"], vec!["image.vectorize"]),
        // text to image alone: no edits, references or starting picture (MOSS merges LoRAs at load; a turbo entry's
        // is a few-step distillation)
        "image" if arch == "ming-image" => (vec!["generate"], vec!["image.generate"]),
        "image" if arch == "moss-image" => {
            let mut c = vec!["generate", "loras"];
            if m["files"]["turbo-lora"].is_string() {
                c.push("few_step");
            }
            (c, vec!["image.generate"])
        }
        "image" => {
            let mut c = vec!["generate", "edit", "compose", "img2img", "loras"];
            if m["env"]["NS_QI_SIGMAS"].is_string() {
                c.push("few_step");
            }
            (c, vec!["image.generate", "image.edit"])
        }
        "audio" if arch == "whisper" => (vec!["transcribe", "speech_to_text", "word_timestamps", "language_detection", "translate_to_english"], vec!["audio.transcribe"]),
        "audio" if m["speech"].is_array() => {
            let mut c = vec!["speech"];
            if speech("voices") {
                c.push("builtin_voices");
            }
            if speech("clone") {
                c.extend(["clone", "saved_voices", "continue_from"]);
            }
            if speech("design") {
                c.push("voice_design");
            }
            // a cloned voice steered by an instruction (a recording and instructions together)
            if speech("direction") {
                c.push("voice_direction");
            }
            // several speakers, a recording each (a script of "Speaker N:" lines: VibeVoice)
            if speech("speakers") {
                c.push("multi_speaker");
            }
            // vocal events inline in the text ("(laugh)", "[笑]")
            if speech("events") {
                c.push("vocal_events");
            }
            let mut methods = vec!["audio.speak"];
            if speech("design") {
                methods.push("audio.voices.design");
            }
            (c, methods)
        }
        "audio" => (vec!["song", "lyrics", "instrumental"], vec!["audio.generate"]),
        "model3d" => {
            // what its card says it does (`does`): generate (meshes from pictures), texture, views (several), rig, animate
            let does = |c: &str| m["does"].as_array().is_some_and(|a| a.iter().any(|x| x == c));
            let mut c = Vec::new();
            let mut methods = Vec::new();
            if does("generate") {
                c.extend(["mesh", "image_to_3d", "glb"]);
                if does("texture") {
                    c.push("texture");
                }
                if does("views") {
                    c.push("multi_view");
                }
                methods.push("model3d.generate");
            } else if does("texture") {
                // a paint-only model: model3d.generate with a mesh (or file) paints it
                c.push("texture");
                methods.push("model3d.generate");
            }
            if does("rig") {
                c.extend(["rig", "skeleton", "skin_weights"]);
            }
            if does("animate") {
                c.extend(["animate", "text_to_motion"]);
            }
            if does("skeletal") {
                // it animates a rigged mesh's skeleton (UniMate): the mesh it is given must carry a skin
                c.extend(["skeletal_animation", "needs_rig"]);
            }
            if does("rig") || does("animate") {
                methods.push("model3d.rig");
            }
            (c, methods)
        }
        "video" => {
            let mut c = vec!["text_to_video", "image_to_video", "sound", "chain", "first_last_frame", "voice_reference"];
            if s(m, "id").contains("ref2va") {
                c.extend(["reference_images", "reference_videos"]);
            } else {
                c.extend(["controlnet", "inpaint"]);
            }
            (c, vec!["video.generate"])
        }
        _ => (Vec::new(), Vec::new()),
    }
}

pub(crate) fn s(m: &Value, k: &str) -> String {
    m.get(k).and_then(Value::as_str).unwrap_or("").to_string()
}

impl Home {
    pub(crate) fn ncards(&self) -> usize {
        self.cards.lock().unwrap().as_array().map_or(0, Vec::len).max(1)
    }

    /// An entry's cards: its `gpus` ("all", "1", "0,1")
    pub(crate) fn gpus_of(&self, m: &Value) -> Vec<usize> {
        let g = s(m, "gpus");
        if g.is_empty() || g == "all" {
            if registry::kind_of(m) == "llm" && g.is_empty() { return vec![0] }
            return (0..self.ncards()).collect();
        }
        let v: Vec<usize> = g.split([',', ' ']).filter_map(|x| x.trim().parse().ok()).collect();
        if v.is_empty() { vec![0] } else { v }
    }

    fn port_of(&self, kind: &str) -> u16 {
        match kind {
            "image" => self.o.ports.image,
            "audio" => self.o.ports.audio,
            "model3d" => self.o.ports.model3d,
            "video" => self.o.ports.video,
            _ => self.o.ports.switch,
        }
    }

    pub(crate) fn target(&self, kind: &str) -> Target {
        Target::Tcp(format!("127.0.0.1:{}", self.port_of(kind)))
    }

    /// Where a chat request goes: the switcher when one runs (it loads the model the request names), else the chat
    /// server, `model` (or the one loaded, or the first enabled) loaded first
    pub(crate) fn chat_target(&self, model: Option<&str>) -> Result<Target, (u16, String)> {
        // a card's chat model not downloaded: refused (409 model_not_downloaded), or with NS_AUTO_PULL fetched first
        if let Some(id) = model {
            self.need_downloaded(id, None, true)?;
        }
        if listening(self.o.ports.switch) {
            // the switcher loads the model named: the room for it first, as any load
            // (no model named: the one the switcher would load - the one loaded, else the first enabled)
            let all = self.models();
            let id = model.map(str::to_string).or_else(|| self.chat_loaded(&all))
                .or_else(|| all.iter().find(|m| registry::kind_of(m) == "llm" && m["enabled"] != false).map(|m| s(m, "id")));
            if let Some(id) = id {
                self.chat_room(&id)?;
            }
            return Ok(self.target("llm"));
        }
        let all = self.models();
        let chat = |m: &&Value| registry::kind_of(m) == "llm" && m["enabled"] != false;
        let id = model.filter(|id| all.iter().filter(chat).any(|m| m["id"] == *id)).map(str::to_string)
            .or_else(|| self.chat_loaded(&all))
            .or_else(|| all.iter().find(chat).map(|m| s(m, "id")))
            .ok_or((409, "no chat model is enabled (system.enable)".to_string()))?;
        self.load(&id)?;
        Ok(Target::Tcp(format!("127.0.0.1:{}", self.o.ports.llm)))
    }

    /// The chat server for `id` started by this page (no studio to ask): the one running stopped first
    fn chat_start(&self, id: &str) -> Result<(), (u16, String)> {
        let _g = self.kind_lock["llm"].lock().unwrap();
        if self.chat_loaded(&self.models()).as_deref() == Some(id) {
            return Ok(());
        }
        if listening(self.o.ports.llm) {
            self.exec("llm stop (another model)".into(), &["llm".into(), "stop".into()]).map_err(|e| (500, e))?;
        }
        let port = self.o.ports.llm.to_string();
        self.exec(format!("llm start {id}"), &["llm".into(), "start".into(), id.into(), "--port".into(), port]).map_err(|e| (500, e))?;
        Ok(())
    }

    /// The memory a model takes on each of its cards, MiB: as seen while it ran, else its model card's `vram_gib` (as
    /// measured where it runs), else from its files' sizes - never more than a card holds beside the spare. An image or
    /// audio model loads whole on one of its cards: what it takes there, for each.
    fn vouched_mb(&self, m: &Value, card_mb: u64) -> (BTreeMap<usize, u64>, bool) {
        let id = s(m, "id");
        let gpus = self.gpus_of(m);
        let single = one_card(m);
        // a model that ran fits the card: a peak seen above what a card may be asked for (an edit's, with its vision tower,
        // touching the card's whole memory) would otherwise refuse the model from then on, its margin never free
        let cap = card_mb.saturating_sub(MARGIN_MB + 512);
        if let Some(seen) = self.seen.lock().unwrap().get(&id) {
            if single && !seen.is_empty() {
                let mb = seen.values().copied().max().unwrap_or(0).min(cap);
                return (gpus.iter().map(|g| (*g, mb)).collect(), true);
            }
            if !single && gpus.iter().all(|i| seen.contains_key(i)) {
                return (gpus.iter().map(|g| (*g, seen[g].min(cap))).collect(), true);
            }
        }
        if let Some(g) = m["vram_gib"].as_f64().filter(|g| *g > 0.0) {
            return (gpus.iter().map(|i| (*i, ((g * 1024.0) as u64).min(cap))).collect(), false);
        }
        let files: u64 = registry::paths(m).iter().filter(|p| p.is_file()).filter_map(|p| std::fs::metadata(p).ok()).map(|x| x.len() >> 20).sum();
        let per = match registry::kind_of(m) {
            "video" => card_mb.saturating_sub(4096),
            "image" => files * 105 / 100 + 4096,
            "audio" => files * 110 / 100 + 2048,
            "model3d" => files * 110 / 100 + 6144,
            _ => files / gpus.len() as u64 + 2048,
        };
        (gpus.iter().map(|g| (*g, per.min(cap))).collect(), false)
    }

    /// Of what a model is vouched for on each card, MiB: the part that is not its weights - the working memory it
    /// needs to run without running out (KV for its context, arenas, an image's or a song's activations). Its entry's
    /// or its model card's `headroom_gib` (as measured: steady vs peak), else what is vouched less its files' share of
    /// the card (an llm's files split over its cards; an image, audio or video model's whole on each). None known (0):
    /// files at least what is vouched - weights partly off the card (host-mirrored or streamed experts).
    /// Whether it was given (else derived from the files) comes with it.
    fn headroom_mb(m: &Value, cards: &[Value], vouch: &BTreeMap<usize, u64>) -> (BTreeMap<usize, u64>, bool) {
        let card = cards.iter().find(|c| [s(m, "catalog"), s(m, "id")].iter().any(|id| !id.is_empty() && c["id"] == id.as_str()));
        let given = m["headroom_gib"].as_f64().or_else(|| card.and_then(|c| c["headroom_gib"].as_f64())).filter(|g| *g >= 0.0);
        if let Some(g) = given {
            return (vouch.iter().map(|(i, mb)| (*i, ((g * 1024.0) as u64).min(*mb))).collect(), true);
        }
        let paths: std::collections::BTreeSet<_> = registry::paths(m).into_iter().filter(|p| p.is_file()).collect();
        let files: u64 = paths.iter().filter_map(|p| std::fs::metadata(p).ok()).map(|x| x.len() >> 20).sum();
        let weights = match registry::kind_of(m) {
            "llm" => files / vouch.len().max(1) as u64,
            _ => files,
        };
        (vouch.iter().map(|(i, mb)| (*i, if files > 0 { mb.saturating_sub(weights) } else { 0 })).collect(), false)
    }

    /// What the studio says of the chat card: (selected mode, the mode starting)
    fn chat_mode(&self) -> (Option<String>, Option<String>) {
        let v = call_for(&self.target("video"), "POST", "/rpc/llm.mode", Some(&json!({})), 4).ok().map(|v| v.get("result").cloned().unwrap_or(v));
        let g = |k: &str| v.as_ref().and_then(|v| v[k].as_str()).map(str::to_string);
        (g("mode"), g("starting"))
    }

    /// Whether the studio runs the chat card and `id` is one of its modes (its llm modes file names it): then it
    /// switches models (it knows when a render holds the card); else this page starts the chat server itself
    fn studio_chat(&self, id: &str) -> bool {
        listening(self.o.ports.video)
            && call_for(&self.target("video"), "POST", "/rpc/llm.mode", Some(&json!({})), 4).ok()
                .map(|v| v.get("result").cloned().unwrap_or(v))
                .is_some_and(|v| v["choices"].as_object().is_some_and(|c| c.contains_key(id)))
    }

    /// The registered chat model the chat server answers as now
    pub(crate) fn chat_loaded(&self, all: &[Value]) -> Option<String> {
        let v = call_for(&Target::Tcp(format!("127.0.0.1:{}", self.o.ports.llm)), "GET", "/v1/models", None, 3).ok()?;
        let m = v["data"].get(0)?;
        if m["status"]["value"].as_str().unwrap_or("loaded") != "loaded" {
            return None;
        }
        let cur = m["id"].as_str()?;
        all.iter().filter(|x| registry::kind_of(x) == "llm" && cur.ends_with(&s(x, "id"))).max_by_key(|x| s(x, "id").len()).map(|x| s(x, "id"))
    }

    /// The kind's servers' ports, the first's first (image, audio and model3d have a second when its port is set)
    pub(crate) fn slots(&self, kind: &str) -> Vec<u16> {
        let p = &self.o.ports;
        match kind {
            "image" => [p.image, p.image2].into_iter().filter(|x| *x != 0).collect(),
            "audio" => [p.audio, p.audio2].into_iter().filter(|x| *x != 0).collect(),
            "model3d" => [p.model3d, p.model3d2].into_iter().filter(|x| *x != 0).collect(),
            _ => vec![self.port_of(kind)],
        }
    }

    /// Each of the kind's servers: what it holds (the ones that answer), and whether it has work - its own or requests
    /// sent on from here
    fn servers(&self, kind: &str) -> Vec<Slot> {
        self.slots(kind).into_iter().enumerate().map(|(slot, port)| {
            let sent = self.inflight.lock().unwrap().get(&port).copied().unwrap_or(0);
            // a server with a request from here is working on it and may not answer until it is done: an image
            // server mid-picture held system.state and system.busy for 6 s a server, past the platform's 8 s, and
            // the box went "offline" for the start of every picture (2026-10-22) - what it said last, busy
            let last = (sent > 0).then(|| self.answers.lock().unwrap().get(&port).cloned()).flatten();
            let st = last.map(|(m, l, _)| (m, l, true)).or_else(|| self.server_at(port));
            Slot { slot, port, model: st.as_ref().map(|x| x.0.clone()), loading: st.as_ref().is_some_and(|x| x.1),
                   busy: st.as_ref().is_some_and(|x| x.2) || sent > 0 }
        }).collect()
    }

    /// Every place work runs, and on which card: the image and audio servers (two each when the second port is set),
    /// the video daemon's GPUs, the chat server's cards
    pub(crate) fn work_slots(&self, all: &[Value], cards: &Value) -> Vec<Value> {
        let mut v = Vec::new();
        for kind in ["image", "audio", "model3d"] {
            if !all.iter().any(|m| registry::kind_of(m) == kind && m["enabled"] != false) {
                continue;
            }
            for x in self.servers(kind) {
                v.push(json!({"kind": kind, "slot": x.slot, "port": x.port, "gpu": card_of_port(cards, x.port),
                              "model": x.model, "loading": x.loading, "busy": x.busy && !x.loading, "up": x.model.is_some()}));
            }
        }
        if all.iter().any(|m| registry::kind_of(m) == "video" && m["enabled"] != false) {
            if let Ok(st) = call_for(&self.target("video"), "GET", "/engine/status", None, 4) {
                for g in st["gpus"].as_array().into_iter().flatten() {
                    v.push(json!({"kind": "video", "slot": g["gpu"], "gpu": g["gpu"], "model": g["denoiser"],
                                  "loading": false, "busy": !g["running"].is_null(), "up": g["engine"].as_str().is_some_and(|e| e != "unloaded")}));
                }
            }
        }
        if let Some(id) = all.iter().any(|m| registry::kind_of(m) == "llm").then(|| self.chat_loaded(all)).flatten() {
            let gpus = all.iter().find(|m| m["id"] == id.as_str()).map(|m| self.gpus_of(m)).unwrap_or_default();
            v.push(json!({"kind": "llm", "slot": 0, "gpu": gpus.first(), "gpus": gpus, "model": id, "loading": false, "busy": false, "up": true}));
        }
        v
    }

    /// The models the kind's servers hold
    pub(crate) fn loaded_of(&self, kind: &str) -> Vec<String> {
        self.slots(kind).into_iter().filter_map(|p| self.server_at(p).map(|x| x.0)).collect()
    }

    /// Where to fetch a file the kind's servers made (they share the output directory): a server that answers
    pub(crate) fn file_target(&self, kind: &str) -> Target {
        let p = self.slots(kind).into_iter().find(|p| listening(*p)).unwrap_or(self.port_of(kind));
        Target::Tcp(format!("127.0.0.1:{p}"))
    }

    /// The image or audio server on `port`: its model, whether it is loading, whether it is busy - when one answers
    fn server_at(&self, port: u16) -> Option<(String, bool, bool)> {
        let t = Target::Tcp(format!("127.0.0.1:{port}"));
        let info = call_for(&t, "GET", "/api/info", None, 3).ok()?;
        let p = call_for(&t, "GET", "/api/progress", None, 3).unwrap_or(json!({}));
        let st = (s(&info, "model"), info["loading"] == true, p["busy"] == true || p["waiting"].as_u64().unwrap_or(0) > 0);
        self.answers.lock().unwrap().insert(port, st.clone());
        Some(st)
    }

    /// The video daemon's engines (their model ids) and the ones loaded on a GPU, when it runs
    fn video_daemon(&self) -> Option<VideoDaemon> {
        let v = call_for(&self.target("video"), "GET", "/engine/status", None, 4).ok()?;
        let names = v["engines"].as_array()?.iter().filter_map(|e| e["name"].as_str().map(str::to_string)).collect();
        // a slot's engine is "unloaded" or holds a denoiser (by its model id)
        let loaded = v["gpus"].as_array().into_iter().flatten()
            .filter(|g| g["engine"].as_str().is_some_and(|e| e != "unloaded"))
            .filter_map(|g| Some((g["denoiser"].as_str()?.to_string(), g["gpu"].as_u64()? as usize)))
            .collect();
        Some((names, loaded))
    }

    /// Every registered model (LoRAs aside): kind, enabled, cards, state, memory vouched for
    pub(crate) fn model_states(&self, all: &[Value], cards: &Value) -> Vec<Value> {
        let card_mb = |g: usize| cards.get(g).and_then(|c| c["vram_total_mb"].as_u64()).unwrap_or(32768);
        let loading = self.loading.lock().unwrap().clone();
        let need = |k: &str| all.iter().any(|m| registry::kind_of(m) == k);
        let chat = if need("llm") { (self.chat_loaded(all), self.chat_mode()) } else { (None, (None, None)) };
        let image = if need("image") { self.servers("image") } else { Vec::new() };
        let audio = if need("audio") { self.servers("audio") } else { Vec::new() };
        let m3d = if need("model3d") { self.servers("model3d") } else { Vec::new() };
        let slots_of = |k: &str| -> &Vec<Slot> { match k { "image" => &image, "audio" => &audio, _ => &m3d } };
        let video = if need("video") { self.video_daemon() } else { None };
        // the model cards, read once: for the headroom of entries registered before cards carried it
        // and whether a chat model spans every card
        let mc = if all.iter().all(|m| m.get("headroom_gib").is_some() && m.get("license").is_some()) && !need("llm") { Vec::new() } else { self.o.cfg.cards() };
        all.iter().filter(|m| registry::kind_of(m) != "lora").map(|m| {
            let (id, kind) = (s(m, "id"), registry::kind_of(m).to_string());
            let enabled = m["enabled"] != false;
            let gpus = self.gpus_of(m);
            let (mut vouch, measured) = self.vouched_mb(m, gpus.first().map_or(32768, |g| card_mb(*g)));
            // an image or audio model allowed several cards counts on the ones its copies are on, else the first it tries
            if one_card(m) && vouch.len() > 1 {
                let on: Vec<usize> = held_on(cards, &id).into_iter().filter(|g| vouch.contains_key(g)).collect();
                let on = if on.is_empty() { vec![gpus[0]] } else { on };
                vouch.retain(|g, _| on.contains(g));
            }
            let (headroom, headroom_given) = Self::headroom_mb(m, &mc, &vouch);
            let mut state = match kind.as_str() {
                "llm" if chat.0.as_deref() == Some(id.as_str()) => "loaded",
                "llm" if chat.1 .1.as_deref() == Some(id.as_str()) => "loading",
                "image" | "audio" | "model3d" => {
                    let mine: Vec<&Slot> = slots_of(&kind).iter().filter(|x| x.model.as_deref() == Some(id.as_str())).collect();
                    if mine.iter().any(|x| x.busy && !x.loading) {
                        "busy"
                    } else if mine.iter().any(|x| !x.loading) {
                        "loaded"
                    } else if !mine.is_empty() {
                        "loading"
                    } else {
                        "unloaded"
                    }
                }
                "video" => match &video {
                    Some((_, loaded)) if loaded.iter().any(|(e, _)| *e == id) => "loaded",
                    Some((names, _)) if names.contains(&id) => "ready",
                    _ => "unloaded",
                },
                _ => "unloaded",
            };
            if loading.contains_key(&id) && state == "unloaded" {
                state = "loading";
            }
            if !enabled && state == "unloaded" {
                state = "disabled";
            }
            let (caps, methods) = capabilities(m);
            // its licence: the entry's own, else its card's (an entry registered before cards carried it)
            let card = mc.iter().find(|c| [s(m, "catalog"), s(m, "id")].iter().any(|i| !i.is_empty() && c["id"] == i.as_str()));
            let licence = registry::licence(if m.get("license").is_some() { m } else { card.unwrap_or(m) });
            let mut row = json!({"id": id, "kind": kind, "title": m["title"], "arch": m["arch"], "enabled": enabled, "pinned": enabled && registry::pinned(m), "held_s": self.held_s(&id).round(), "gpus": gpus, "state": state,
                   "spans": spans_all(m, &mc),
                   "capabilities": caps, "methods": methods,
                   "vouch_mb": vouch.iter().map(|(g, mb)| (g.to_string(), json!(mb))).collect::<serde_json::Map<String, Value>>(),
                   "measured": measured,
                   "headroom_mb": headroom.iter().map(|(g, mb)| (g.to_string(), json!(mb))).collect::<serde_json::Map<String, Value>>(),
                   "headroom_given": headroom_given,
                   "page": match kind.as_str() { "image" | "audio" | "model3d" => json!(slots_of(&kind).iter()
                                                      .find(|x| x.model.as_deref() == Some(id.as_str())).map_or(self.port_of(&kind), |x| x.port)),
                                                  "video" => json!(self.o.ports.video), _ => json!(self.o.ports.chat_ui) }});
            row.as_object_mut().unwrap().extend(licence);
            // its concurrency profile (the card's; "unmeasured" until a bench says otherwise) and the copies running now
            let prof = if m.get("concurrency").is_some() || m.get("coexistence").is_some() { Profile::of_entry(m) }
                       else { card.map_or_else(Profile::default, Profile::of_entry) };
            let (pc, px) = prof.to_json();
            let copies_now = match kind.as_str() {
                "image" | "audio" | "model3d" => slots_of(&kind).iter().filter(|x| x.model.as_deref() == Some(id.as_str())).count(),
                _ => usize::from(matches!(state, "loaded" | "busy")),
            };
            row["recommended_copies"] = pc["recommended_copies"].clone();
            row["max_copies_per_card"] = pc["max_copies_per_card"].clone();
            row["copies_now"] = json!(copies_now);
            row["concurrency"] = pc;
            row["coexistence"] = px;
            row
        }).collect()
    }

    /// Each card's vouched memory: the enabled models on it (chat models count once - the largest, or the loaded one:
    /// one runs at a time), and whether that passes the card
    pub(crate) fn vouch(&self, cards: &mut Value, models: &[Value]) {
        for c in cards.as_array_mut().into_iter().flatten() {
            let gi = c["index"].as_u64().unwrap_or(0).to_string();
            let total = c["vram_total_mb"].as_u64().unwrap_or(0);
            let on: Vec<&Value> = models.iter().filter(|m| m["enabled"] == true && m["vouch_mb"].get(&gi).is_some()).collect();
            let mb = |m: &Value| m["vouch_mb"][&gi].as_u64().unwrap_or(0);
            let hr = |m: &Value| m["headroom_mb"][&gi].as_u64().unwrap_or(0);
            // chat models and video engines count once a card: one of each runs there at a time (the loaded one, else
            // the largest)
            let mut list: Vec<Value> = on.iter().filter(|m| !matches!(m["kind"].as_str(), Some("llm" | "video")))
                .map(|m| json!({"model": m["id"], "kind": m["kind"], "mb": mb(m), "headroom_mb": hr(m), "headroom_given": m["headroom_given"],
                                "measured": m["measured"], "state": m["state"]})).collect();
            for kind in ["llm", "video"] {
                let group: Vec<&&Value> = on.iter().filter(|m| m["kind"] == kind).collect();
                if let Some(c) = group.iter().find(|m| m["state"] == "loaded").or_else(|| group.iter().max_by_key(|m| mb(m))) {
                    list.push(json!({"model": c["id"], "kind": kind, "mb": mb(c), "headroom_mb": hr(c), "headroom_given": c["headroom_given"],
                                     "measured": c["measured"], "state": c["state"], "of": group.len()}));
                }
            }
            let sum: u64 = list.iter().map(|v| v["mb"].as_u64().unwrap_or(0)).sum();
            c["vouched"] = json!(list);
            c["vouched_mb"] = json!(sum);
            // of it, the models' headroom (their working memory beyond the weights)
            c["vouched_headroom_mb"] = json!(list.iter().map(|v| v["headroom_mb"].as_u64().unwrap_or(0)).sum::<u64>());
            c["overbooked"] = json!(sum > total);
        }
    }

    fn entry(&self, id: &str) -> Result<Value, (u16, String)> {
        let m = registry::find(&self.o.cfg, id).map_err(|e| (500, e))?.ok_or_else(|| (404, format!("no model {id} (nextcuda models list)")))?;
        if registry::kind_of(&m) == "lora" {
            return Err((400, format!("{id} is a LoRA: it is enabled with the model that uses it")));
        }
        Ok(m)
    }

    /// Enable `id` on cards `gpus` (a list; none: as it was)
    pub(crate) fn enable(self: &Arc<Self>, id: &str, gpus: Option<&Value>) -> Result<String, (u16, String)> {
        let m = self.entry(id)?;
        if let Some(why) = self.moving_why(id) {
            return Err((409, why));
        }
        let n = self.ncards();
        let gpus: Option<Vec<usize>> = gpus.and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_u64).map(|g| g as usize).collect());
        if let Some(g) = &gpus {
            if g.is_empty() || g.iter().any(|x| *x >= n) {
                return Err((400, format!("gpus {g:?}: one or more of 0..{}", n - 1)));
            }
        }
        registry::update(&self.o.cfg, id, |e| {
            e["enabled"] = json!(true);
            if let Some(g) = &gpus {
                e["gpus"] = json!(g.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
            }
        }).map_err(|e| (500, e))?;
        if registry::kind_of(&m) == "video" {
            let me = self.clone();
            std::thread::spawn(move || {
                if let Err(e) = me.video_apply(false) {
                    let (rid, log) = me.new_run("video: apply the enabled models".into());
                    let _ = std::fs::write(log, e);
                    me.finish(rid, 1);
                }
            });
        }
        Ok(s(&m, "title"))
    }

    /// The cards an enabled (or disabled) model may use, changed in place - more than fit is allowed (the card is
    /// shown overbooked: the first model to load holds it). A loaded model whose card is taken away unloads (after
    /// the requests it is answering): its next request loads it on the new cards.
    pub(crate) fn set_gpus(self: &Arc<Self>, id: &str, gpus: Option<&Value>) -> Result<Value, (u16, String)> {
        let m = self.entry(id)?;
        let n = self.ncards();
        let g: Vec<usize> = gpus.and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_u64).map(|g| g as usize).collect()).unwrap_or_default();
        if g.is_empty() || g.iter().any(|x| *x >= n) {
            return Err((400, format!("gpus {g:?}: one or more of 0..{}", n - 1)));
        }
        let kind = registry::kind_of(&m).to_string();
        let mut g = g;
        g.sort();
        g.dedup();
        if g.len() < n && spans_all(&m, &self.o.cfg.cards()) {
            return Err((400, format!("{id} is spread over every card (its model card: gpus all): it does not run on fewer")));
        }
        let old = self.gpus_of(&m);
        if old == g {
            return Ok(json!({"model": id, "gpus": g, "changed": false}));
        }
        // a loaded model leaves the cards it was on (it reloads on the new ones at its next request)
        let state = self.model_states(std::slice::from_ref(&m), &json!([])).first().map(|x| s(x, "state")).unwrap_or_default();
        // (an image or audio model only when its card is the one taken away)
        let on = held_on(&self.named_cards(&self.models()), id);
        let kept = one_card(&m) && !on.is_empty() && on.iter().all(|c| g.contains(c));
        if m["enabled"] != false && matches!(state.as_str(), "loaded" | "busy" | "loading") && kind != "video" && !kept {
            self.unload(id)?;
        }
        registry::update(&self.o.cfg, id, |e| e["gpus"] = json!(g.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","))).map_err(|e| (500, e))?;
        if kind == "video" && m["enabled"] != false {
            let me = self.clone();
            std::thread::spawn(move || {
                if let Err(e) = me.video_apply(false) {
                    let (rid, log) = me.new_run("video: apply the enabled models' cards".into());
                    let _ = std::fs::write(log, e);
                    me.finish(rid, 1);
                }
            });
        }
        Ok(json!({"model": id, "gpus": g, "changed": true, "was": old}))
    }

    /// Disable `id`: unloaded first (after the requests it is answering); a video model while a clip renders only
    /// with `force` (the clip stops at its next step)
    pub(crate) fn disable(self: &Arc<Self>, id: &str, force: bool) -> Result<(), (u16, String)> {
        let m = self.entry(id)?;
        if registry::kind_of(&m) == "video" && !force && self.rendering() {
            return Err((409, "a clip is rendering: pause the studio's queue and wait, or force (the clip stops at its next step)".into()));
        }
        self.unload(id)?;
        registry::update(&self.o.cfg, id, |e| {
            e["enabled"] = json!(false);
            e["pinned"] = json!(false);
        }).map_err(|e| (500, e))?;
        if registry::kind_of(&m) == "video" {
            self.video_apply(force).map_err(|e| (500, e))?;
        }
        Ok(())
    }

    fn rendering(&self) -> bool {
        call_for(&self.target("video"), "POST", "/rpc/status", Some(&json!({})), 4).ok()
            .map(|v| v.get("result").cloned().unwrap_or(v)).is_some_and(|r| r["idle"] == false)
    }

    /// Pin `id`: kept in the registry beside `enabled`, then loaded now in the background (the page's pin button: it
    /// spins while the state says loading). A load that fails (no room) leaves the pin set: the keeper tries again and
    /// the failure is a run, like any load's
    pub(crate) fn pin(self: &Arc<Self>, id: &str) -> Result<(), (u16, String)> {
        let m = self.entry(id)?;
        if m["enabled"] == false {
            return Err((409, format!("{id} is disabled: enable it first")));
        }
        registry::update(&self.o.cfg, id, |e| e["pinned"] = json!(true)).map_err(|e| (500, e))?;
        self.load_async(id)
    }

    /// Unpin `id`: the flag only. The model stays loaded until the idle rules take it
    pub(crate) fn unpin(&self, id: &str) -> Result<(), (u16, String)> {
        self.entry(id)?;
        registry::update(&self.o.cfg, id, |e| e["pinned"] = json!(false)).map_err(|e| (500, e))?;
        Ok(())
    }

    /// Load `id` now, in the background (`load` for tools, and the pin): it spins while the state says loading)
    pub(crate) fn load_async(self: &Arc<Self>, id: &str) -> Result<(), (u16, String)> {
        let m = self.entry(id)?;
        if m["enabled"] == false {
            return Err((409, format!("{id} is disabled: enable it first")));
        }
        if let Some(why) = self.moving_why(id) {
            return Err((409, why));
        }
        if self.loading.lock().unwrap().insert(id.to_string(), now()).is_some() {
            return Ok(());
        }
        let (me, id) = (self.clone(), id.to_string());
        std::thread::spawn(move || {
            if let Err((_, e)) = me.load(&id) {
                let (rid, log) = me.new_run(format!("load {id}"));
                let _ = std::fs::write(log, &e);
                me.finish(rid, 1);
            }
            me.loading.lock().unwrap().remove(&id);
        });
        Ok(())
    }

    /// Load `id` and return when it answers
    pub(crate) fn load(&self, id: &str) -> Result<(), (u16, String)> {
        let m = self.entry(id)?;
        if let Some(why) = self.moving_why(id) {
            return Err((409, why));
        }
        match registry::kind_of(&m) {
            "image" | "audio" | "model3d" => self.ensure(id).map(|_| ()),
            "video" => self.video_apply(false).map_err(|e| (500, e)),
            _ => {
                let all = self.models();
                if self.chat_loaded(&all).as_deref() == Some(id) {
                    return Ok(());
                }
                self.chat_room(id)?;
                if !self.studio_chat(id) {
                    return self.chat_start(id);
                }
                call_for(&self.target("video"), "POST", "/rpc/llm.mode", Some(&json!({"mode": id})), 600).map_err(|e| (502, format!("the studio's llm.mode: {e}")))?;
                let t0 = Instant::now();
                while t0.elapsed() < Duration::from_secs(900) {
                    if self.chat_loaded(&all).as_deref() == Some(id) {
                        return Ok(());
                    }
                    std::thread::sleep(Duration::from_secs(3));
                }
                Err((504, format!("{id} did not come up in 15 minutes (a render may hold the card)")))
            }
        }
    }

    /// Unload `id` if it is loaded (after the requests it is answering)
    pub(crate) fn unload(&self, id: &str) -> Result<(), (u16, String)> {
        let m = self.entry(id)?;
        let kind = registry::kind_of(&m);
        match kind {
            "image" | "audio" | "model3d" => {
                let _g = self.kind_lock[kind].lock().unwrap();
                for x in self.servers(kind).iter().filter(|x| x.model.as_deref() == Some(id)) {
                    self.exec(format!("{kind} stop ({id})"), &[kind.into(), "stop".into(), "--slot".into(), x.slot.to_string()]).map_err(|e| (500, e))?;
                }
                Ok(())
            }
            "video" => self.exec("video: unload the engines".into(), &["video".into(), "unload".into()]).map(|_| ()).map_err(|e| (500, e)),
            _ => {
                if self.chat_loaded(&self.models()).as_deref() == Some(id) && !self.studio_chat(id) {
                    return self.exec(format!("llm stop ({id})"), &["llm".into(), "stop".into()]).map(|_| ()).map_err(|e| (500, e));
                }
                if self.chat_loaded(&self.models()).as_deref() == Some(id) {
                    call_for(&self.target("video"), "POST", "/rpc/llm.mode", Some(&json!({"mode": "none"})), 120)
                        .map_err(|e| (502, format!("the studio's llm.mode: {e}")))?;
                }
                Ok(())
            }
        }
    }

    /// The image, audio or 3D server holding `id`, started (or swapped to it) when it does not: where to send its requests
    pub(crate) fn ensure(&self, id: &str) -> Result<Lease<'_>, (u16, String)> {
        self.ensure_for(id, None)
    }

    /// `ensure` for a job (`job`: its id): what it waits for is noted for <kind>.status (stage, waiting_for), it waits
    /// for room up to JOB_ROOM_WAIT_S (a request without a job ROOM_WAIT_S), and a cancel ends the wait
    pub(crate) fn ensure_for(&self, id: &str, job: Option<&str>) -> Result<Lease<'_>, (u16, String)> {
        let m = self.entry(id)?;
        let kind = registry::kind_of(&m);
        if !matches!(kind, "image" | "audio" | "model3d") {
            return Err((400, format!("{id} is a {kind} model")));
        }
        if m["enabled"] == false {
            return Err((409, format!("{id} is disabled (enable it on :8000 or nextcuda models enable {id})")));
        }
        let default = Duration::from_secs(if job.is_some() { JOB_ROOM_WAIT_S } else { ROOM_WAIT_S });
        // the caller's max_willing_to_wait_s bounds the wait for the request to start (and never lengthens the default)
        let max_wait = super::rsvp::max_wait();
        let limit = wait_limit(default, max_wait);
        let t0 = Instant::now();
        loop {
            if job.is_some_and(|j| self.job_cancelled(j)) {
                return Err((499, format!("cancelled while it waited for {id}")));
            }
            // its files moving to another drive: it waits for the move like for room
            if let Some(why) = self.moving_why(id) {
                self.note(job, "waiting", &why);
                if t0.elapsed() > limit {
                    return Err((409, why));
                }
                std::thread::sleep(Duration::from_secs(3));
                continue;
            }
            let why = match self.place(&m, id, kind, job)? {
                Ok(lease) => {
                    super::rsvp::note_start(id, t0.elapsed().as_secs_f64());
                    return Ok(lease);
                }
                Err(why) => why,
            };
            self.note(job, "waiting", &why);
            if max_wait.is_some_and(|mw| t0.elapsed().as_secs_f64() > mw) {
                // not started in time (the ETA was unknown): withdrawn - nothing was started for it past this point
                return Err(self.wait_expired(&format!("{kind}.request"), &json!({"model": id}), max_wait.unwrap_or(0.0), t0.elapsed().as_secs_f64()));
            }
            if t0.elapsed() > limit {
                return Err((503, format!("{why}; {id} loads once one of them idles out or is stopped")));
            }
            std::thread::sleep(Duration::from_secs(3));
        }
    }

    /// One try at a server for `id` (the kind's lock held while it lasts): a lease, or why there is none yet (no card
    /// of its has the room, even with the idle models there stopped)
    pub(crate) fn place(&self, m: &Value, id: &str, kind: &str, job: Option<&str>) -> Result<Result<Lease<'_>, String>, (u16, String)> {
        let gpus = self.gpus_of(m);
        let _g = self.kind_lock[kind].lock().unwrap();
        let sv = self.servers(kind);
        let pinned = self.kept_ids(&self.models());
        let kept = |x: &Slot| x.model.as_ref().is_some_and(|m| pinned.contains(m));
        let inflight = |p: u16| self.inflight.lock().unwrap().get(&p).copied().unwrap_or(0);
        let mine: Vec<&Slot> = sv.iter().filter(|x| x.model.as_deref() == Some(id)).collect();
        // a server of it with nothing to do
        if let Some(x) = mine.iter().find(|x| !x.busy && inflight(x.port) == 0) {
            if x.loading {
                self.note(job, "loading", &format!("{id} loading (server {})", x.slot + 1));
                self.wait_up(x.port, id)?;
            }
            return Ok(Ok(self.lease(x.port)));
        }
        let card_mb = |c: &Value| c["vram_total_mb"].as_u64().unwrap_or(0);
        let (need, _) = self.vouched_mb(m, self.cards.lock().unwrap().get(gpus[0]).map_or(32768, card_mb));
        let asked = self.asked.lock().unwrap().get(id).copied().unwrap_or(0);
        let want = |g: usize| (need.get(&g).copied().unwrap_or(0) + MARGIN_MB).max(asked);
        let cards = self.named_cards(&self.models());
        // a server that holds another model and could go now
        let idle_other = |x: &&Slot| x.model.is_some() && x.model.as_deref() != Some(id) && !kept(x) && !x.busy && !x.loading && inflight(x.port) == 0;
        if !mine.is_empty() {
            // its servers all have work: another copy on a card of its without one that has the room now (or once idle
            // picture and sound servers there stop), in a slot no server holds or one an idle model holds - else the copy
            // with the fewest requests (its server queues it)
            let on: Vec<usize> = mine.iter().filter_map(|x| card_of_port(&cards, x.port)).collect();
            // copies per card: extra ones up to the profile's recommended count (the default 1: one a card), never past its max
            let prof = self.profile_of(m);
            let per_card = prof.concurrency.recommended_copies.min(prof.concurrency.max_copies_per_card) as usize;
            let spare = sv.iter().find(|x| x.model.is_none()).or_else(|| sv.iter().find(idle_other));
            let plan = pick_card(gpus.iter().copied().filter(|g| extra_copy_allowed(&on, *g, per_card))
                .map(|g| (g, card_free(&cards, g), self.room_plan_for(&cards, g, want(g), kind, true, Some(m)))));
            if let (Some(x), Some((gpu, plan))) = (spare, plan) {
                // (a plan another request's lock stands in the way of: no copy now, the requests share the first)
                let started = self.free(&plan, kind, id).and_then(|_| {
                    self.stop_slot(kind, x, id)?;
                    if !self.wait_room(gpu, want(gpu)) {
                        return Err((503, format!("GPU {gpu} did not free the room")));
                    }
                    self.note(job, "loading", &format!("a second copy of {id} loading on GPU {gpu} (server {})", x.slot + 1));
                    self.start_at(kind, id, x, gpu).inspect_err(|(_, e)| { self.learn_ask(id, e, gpu, want(gpu)); })
                });
                match started {
                    Ok(()) => return Ok(Ok(self.lease(x.port))),
                    Err((_, e)) => eprintln!("{id}: a second copy did not start ({e}); the requests share the first"),
                }
            }
            let x = mine.iter().min_by_key(|x| inflight(x.port)).unwrap();
            if x.loading {
                self.note(job, "loading", &format!("{id} loading (server {})", x.slot + 1));
                self.wait_up(x.port, id)?;
            }
            return Ok(Ok(self.lease(x.port)));
        }
        // not loaded: a card of its with the room now (the one with the most free), else the first that has it once idle
        // models there go (picture and sound servers, an idle video worker, a chat model not kept by X-Keep-Up-S); a
        // slot no server holds, else one an idle model holds, else one a busy model of the kind holds, after its requests
        let plan = pick_card(gpus.iter().map(|&g| (g, card_free(&cards, g), self.room_plan_for(&cards, g, want(g), kind, false, Some(m)))));
        let empty = sv.iter().find(|x| x.model.is_none());
        let idle = sv.iter().filter(idle_other).min_by_key(|x| card_of_port(&cards, x.port).is_none_or(|g| Some(g) != plan.as_ref().map(|p| p.0)));
        let on_mine = |x: &&Slot| x.model.is_some() && !kept(x) && card_of_port(&cards, x.port).is_some_and(|g| gpus.contains(&g));
        let Some((gpu, plan)) = plan else {
            // no room even then: a busy server of another model of the kind on a card of its goes after its requests
            // (it holds the room), then the next try loads it
            let held = |x: &Slot| card_of_port(&cards, x.port).map(|g| {
                let mb = cards[g]["procs"].as_array().into_iter().flatten().filter(|p| p["port"].as_str() == Some(x.port.to_string().as_str()))
                    .map(|p| p["vram_mb"].as_u64().unwrap_or(0)).sum::<u64>();
                (g, mb)
            });
            let frees = |x: &&Slot| held(x).is_some_and(|(g, mb)| self.room_plan_for(&cards, g, want(g).saturating_sub(mb), kind, false, Some(m)).is_some());
            if let Some(x) = sv.iter().filter(on_mine).filter(frees).min_by_key(|x| inflight(x.port)) {
                self.drain(x, kind, id, job)?;
                return Ok(Err(format!("server {} stopped for {id}", x.slot + 1)));
            }
            return Ok(Err(self.no_room(&cards, &gpus, id, want(gpus[0]))));
        };
        let x = match (empty, idle) {
            (Some(e), _) => e,
            (None, Some(i)) => i,
            // every server busy with another model: the one with the fewest requests, after them
            _ => {
                // (a pinned model's server is never the one: every server pinned, no room)
                let Some(x) = sv.iter().filter(|x| x.model.is_some() && !kept(x)).min_by_key(|x| (!on_mine(x) as u8, inflight(x.port))) else {
                    return Ok(Err(self.no_room(&cards, &gpus, id, want(gpus[0]))));
                };
                self.drain(x, kind, id, job)?;
                x
            }
        };
        self.stop_slot(kind, x, id)?;
        // an idle model whose kind is busy changing its own server stays: the next try plans again
        if let Err((_, e)) = self.free(&plan, kind, id) {
            return Ok(Err(e));
        }
        // what was stopped has let go of the room before the load (a holder that took work meanwhile is not stopped:
        // the next try plans again)
        if !self.wait_room(gpu, want(gpu)) {
            return Ok(Err(format!("GPU {gpu} has not freed {:.1} GiB for {id} yet", want(gpu) as f64 / 1024.0)));
        }
        self.note(job, "loading", &format!("{id} loading on GPU {gpu} (server {})", x.slot + 1));
        if let Err((code, e)) = self.start_at(kind, id, x, gpu) {
            // the server asked more than was planned for: the next try plans for that (stopping idle work for it)
            if self.learn_ask(id, &e, gpu, want(gpu)) {
                return Ok(Err(format!("{id} needed more room on GPU {gpu} than planned ({e}); planned again")));
            }
            return Err((code, e));
        }
        Ok(Ok(self.lease(x.port)))
    }

    /// Server `x` (another model's) after its requests: waited for, with the kind's lock held, so no new request is
    /// sent to it meanwhile; then stopped
    fn drain(&self, x: &Slot, kind: &str, id: &str, job: Option<&str>) -> Result<(), (u16, String)> {
        let Some(other) = &x.model else { return Ok(()) };
        let inflight = |p: u16| self.inflight.lock().unwrap().get(&p).copied().unwrap_or(0);
        let t0 = Instant::now();
        while self.server_at(x.port).is_some_and(|(_, _, busy)| busy) || inflight(x.port) > 0 {
            if job.is_some_and(|j| self.job_cancelled(j)) {
                return Err((499, format!("cancelled while it waited for {other}'s requests")));
            }
            if t0.elapsed() > Duration::from_secs(JOB_ROOM_WAIT_S) {
                return Err((503, format!("{other} has been busy for 30 minutes; {id} waits for it")));
            }
            self.note(job, "waiting", &format!("server {} finishing {other}'s requests, then {id} loads there", x.slot + 1));
            std::thread::sleep(Duration::from_secs(2));
        }
        self.stop_slot(kind, x, id)
    }

    /// Stop the server in slot `x` when it holds a model (one of the kind, its lock held), for `id`
    fn stop_slot(&self, kind: &str, x: &Slot, id: &str) -> Result<(), (u16, String)> {
        if let Some(other) = self.server_at(x.port).map(|s| s.0) {
            self.exec(format!("{kind} stop ({other}, for {id})"), &[kind.into(), "stop".into(), "--slot".into(), x.slot.to_string()]).map_err(|e| (500, e))?;
        }
        Ok(())
    }

    /// The message for no room: each card, what it has free and who holds the rest (busy or idle)
    fn no_room(&self, cards: &Value, gpus: &[usize], id: &str, want: u64) -> String {
        let free = |g: usize| cards.get(g).map_or(0, |c| c["vram_total_mb"].as_u64().unwrap_or(0).saturating_sub(c["vram_used_mb"].as_u64().unwrap_or(0)));
        let busy: Vec<String> = gpus.iter().map(|g| {
            let holders: Vec<String> = cards.get(*g).and_then(|c| c["procs"].as_array()).into_iter().flatten()
                .map(|p| format!("{} {:.1} GiB", p["model"].as_str().or(p["program"].as_str()).unwrap_or("?"), p["vram_mb"].as_u64().unwrap_or(0) as f64 / 1024.0))
                .collect();
            format!("GPU {g}: {:.1} GiB free ({})", free(*g) as f64 / 1024.0, holders.join(", "))
        }).collect();
        let all = self.models();
        let pinned = pinned_holders(cards, gpus, &pinned_ids(&all));
        let held = pinned_holders(cards, gpus, &self.kept_ids(&all).difference(&pinned_ids(&all)).cloned().collect());
        let excl: Vec<String> = pinned_holders(cards, gpus, &all.iter().filter(|o| self.profile_of(o).coexistence.mode == profile::Mode::Exclusive).map(|o| s(o, "id")).collect());
        let mut pin = if pinned.is_empty() { String::new() } else { format!("; pinned: {} (unpin to free the room)", pinned.join(", ")) };
        if !excl.is_empty() {
            pin += &format!("; exclusive: {} (runs alone on its card)", excl.join(", "));
        }
        if self.entry(id).is_ok_and(|m| self.profile_of(&m).coexistence.mode == profile::Mode::Exclusive) {
            pin += &format!("; {id} is exclusive: it needs its card to itself");
        }
        if !held.is_empty() {
            pin += &format!("; held by rsvp_s: {} (until its hold ends: {})", held.join(", "),
                            held.iter().map(|h| format!("{h} {:.0} s", self.held_s(h))).collect::<Vec<_>>().join(", "));
        }
        format!("{} held by work that is not idle - {}; {id} needs ~{:.1} GiB{pin}", if gpus.len() > 1 { "its cards are" } else { "its card is" },
                busy.join("; "), want as f64 / 1024.0)
    }

    /// What could go from card `g` so `want` MiB is free there: none needed (Some(empty)), the idle holders to stop
    /// (Some), or no way now (None). Idle: a picture or sound server without work, the video daemon's worker while the
    /// studio has nothing to make, a chat model answering nothing and not kept (chat_idle). `light`: picture and sound servers
    /// only (for a second copy, which is no reason to stop a video or chat model). The kind asking is never asked to
    /// free its own kind's chat or video.
    /// `room_plan` for a model `m` asking: the residents that may not share a card with it (an exclusive model's, or
    /// its own exclusivity, or `shared_with`) are stopped as well when idle, and make the card no choice when they are
    /// not (busy, pinned, held)
    fn room_plan_for(&self, cards: &Value, g: usize, want: u64, asking: &str, light: bool, m: Option<&Value>) -> Option<Vec<Holder>> {
        let c = cards.get(g)?;
        let free = card_free(cards, g);
        let (exclusive, blocks) = match m {
            Some(m) => self.sharing(m),
            None => (false, Box::new(|_: &str| false) as Box<dyn Fn(&str) -> bool + '_>),
        };
        // the nextcuda models on the card (a foreign process is not one)
        let residents: Vec<String> = c["procs"].as_array().into_iter().flatten().filter(|p| p["kind"].is_string())
            .map(|p| p["model_id"].as_str().or(p["model"].as_str()).unwrap_or("?").to_string()).collect();
        if free >= want && !residents.iter().any(|r| exclusive || blocks(r)) {
            return Some(Vec::new());
        }
        let inflight = |p: u16| self.inflight.lock().unwrap().get(&p).copied().unwrap_or(0);
        let mut idle: Vec<Holder> = Vec::new();
        let mut video_checked = None;
        for p in c["procs"].as_array().into_iter().flatten() {
            let mb = p["vram_mb"].as_u64().unwrap_or(0);
            let model = p["model_id"].as_str().or(p["model"].as_str()).unwrap_or("?").to_string();
            match p["kind"].as_str() {
                Some(k @ ("image" | "audio" | "model3d")) => {
                    let Some(port) = p["port"].as_str().and_then(|x| x.parse::<u16>().ok()) else { continue };
                    let Some(x) = self.servers(k).into_iter().find(|x| x.port == port) else { continue };
                    if !x.busy && !x.loading && inflight(port) == 0 {
                        idle.push(Holder { what: Free::Server(k.to_string(), x.slot), mb, model, order: 0 });
                    }
                }
                Some("video") if !light && asking != "video" => {
                    let ok = *video_checked.get_or_insert_with(|| self.video_idle());
                    if ok {
                        idle.push(Holder { what: Free::Video(g), mb, model, order: 1 });
                    }
                }
                Some("llm") if !light && asking != "llm" => {
                    if let Some(id) = self.chat_idle() {
                        idle.push(Holder { what: Free::Chat(id), mb, model, order: 2 });
                    }
                }
                _ => {}
            }
        }
        plan_with_blockers(free, want, evictable(idle, &self.kept_ids(&self.models())), &residents, exclusive, &*blocks)
    }

    /// A model's profile: its entry's, else its model card's (cards read at most once a minute), else the defaults
    pub(crate) fn profile_of(&self, m: &Value) -> Profile {
        if m.get("concurrency").is_some() || m.get("coexistence").is_some() {
            return Profile::of_entry(m);
        }
        let cards = self.card_cache();
        cards.iter().find(|c| [s(m, "catalog"), s(m, "id")].iter().any(|i| !i.is_empty() && c["id"] == i.as_str()))
            .map_or_else(Profile::default, Profile::of_entry)
    }

    fn card_cache(&self) -> Vec<Value> {
        let mut c = self.cards_cache.lock().unwrap();
        if c.0.is_none_or(|t| t.elapsed() > Duration::from_secs(60)) {
            *c = (Some(Instant::now()), self.o.cfg.cards());
        }
        c.1.clone()
    }

    /// Whether model `m` is exclusive, and a test for another model's id: may it NOT share a card with `m`
    pub(crate) fn sharing(&self, m: &Value) -> (bool, Box<dyn Fn(&str) -> bool + '_>) {
        let (id, kind) = (s(m, "id"), registry::kind_of(m).to_string());
        let me = self.profile_of(m);
        let others: BTreeMap<String, (String, Profile)> = self.models().iter().map(|o| (s(o, "id"), (registry::kind_of(o).to_string(), self.profile_of(o)))).collect();
        let exclusive = me.coexistence.mode == profile::Mode::Exclusive;
        (exclusive, Box::new(move |other: &str| {
            other != id && others.get(other).is_some_and(|(ok, op)| !me.may_share(&id, &kind, op, other, ok))
        }))
    }


    /// A server's refusal to load for want of memory (its "needs about X GiB ...; Y GiB are free", in `e`) on card
    /// `gpu`, planned for `want` MiB: what it asks, as the front door counts free memory, kept for `id`. Whether that
    /// is more than was planned (another try may then succeed).
    fn learn_ask(&self, id: &str, e: &str, gpu: usize, want: u64) -> bool {
        let Some((need, had)) = asked_mb(e) else { return false };
        let ask = ask_mb(need, had, card_free(&self.cards.lock().unwrap(), gpu));
        eprintln!("{id}: its server asked {need} MiB with {had} MiB free on GPU {gpu}; planned {want} MiB, from now {ask}");
        let mut a = self.asked.lock().unwrap();
        let more = ask > want && a.get(id).is_none_or(|x| ask > *x);
        if more {
            a.insert(id.to_string(), ask);
        }
        more
    }

    /// Stop the idle holders of a plan (for `id`; the kind `kind` asking holds its own lock; another kind's is tried, and
    /// a server whose lock is taken is left: the next try plans again)
    fn free(&self, plan: &[Holder], kind: &str, id: &str) -> Result<(), (u16, String)> {
        for h in plan {
            match &h.what {
                Free::Server(k, slot) => {
                    let guard = if k == kind { None } else {
                        match self.kind_lock[k.as_str()].try_lock() {
                            Ok(g) => Some(g),
                            Err(_) => return Err((503, format!("{} is changing its {k} server; {id} tries again", h.model))),
                        }
                    };
                    // still idle (it may have taken a request since)
                    let x = self.servers(k).into_iter().find(|x| x.slot == *slot);
                    if x.as_ref().is_some_and(|x| !x.busy && !x.loading && x.model.is_some()) {
                        self.exec(format!("{k} stop ({}, idle, for {id})", h.model), &[k.clone(), "stop".into(), "--slot".into(), slot.to_string()]).map_err(|e| (500, e))?;
                    }
                    drop(guard);
                }
                Free::Video(g) => {
                    if self.video_idle() {
                        call_for(&self.target("video"), "POST", "/engine/unload", Some(&json!({"gpu": g})), 30).map_err(|e| (502, format!("the video daemon: {e}")))?;
                        let (rid, log) = self.new_run(format!("video: GPU {g}'s idle worker let go, for {id}"));
                        let _ = std::fs::write(log, "");
                        self.finish(rid, 0);
                    }
                }
                Free::Chat(chat) => {
                    if self.chat_idle().as_deref() == Some(chat.as_str()) {
                        self.unload(chat)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Wait (30 s at most) until card `g` has `want` MiB free: a stopped process lets go of its memory in a moment.
    /// Whether it came.
    fn wait_room(&self, g: usize, want: u64) -> bool {
        let t0 = Instant::now();
        while t0.elapsed() < Duration::from_secs(30) {
            if card_free(&self.cards.lock().unwrap(), g) >= want {
                return true;
            }
            std::thread::sleep(Duration::from_secs(1));
        }
        false
    }

    /// Whether the studio has nothing to make: no clip running, none runnable in the queue
    fn video_idle(&self) -> bool {
        call_for(&self.target("video"), "POST", "/rpc/status", Some(&json!({})), 4).ok().map(|v| v.get("result").cloned().unwrap_or(v))
            .is_some_and(|st| st["idle"] == true && st["queue_items"].as_array().is_none_or(|a| a.iter().all(|i| i["held"] == true)))
    }

    /// The chat model loaded, when it may be preempted: answering nothing, not changing, and not kept - a request
    /// that sent X-Keep-Up-S (Open WebUI's connection does: 300 s) keeps it that long after it ended, here or through
    /// the switcher (its GET /keep); a request without the header keeps nothing
    pub(crate) fn chat_idle(&self) -> Option<String> {
        let all = self.models();
        let id = self.chat_loaded(&all)?;
        if self.kept_ids(&all).contains(&id) {
            return None;
        }
        let v = call_for(&Target::Unix(self.chat_socket()), "GET", "/server/requests", None, 3).ok()?;
        if v["active"].as_array().is_some_and(|a| !a.is_empty()) || self.chat_mode().1.is_some() {
            return None;
        }
        if self.chat_keep.lock().unwrap().get(&id).is_some_and(|until| *until > now()) {
            return None;
        }
        if listening(self.o.ports.switch) {
            // the switcher not answering: kept, to be safe
            let k = call_for(&Target::Tcp(format!("127.0.0.1:{}", self.o.ports.switch)), "GET", "/keep", None, 3).ok()?;
            if k["kept"][id.as_str()].as_f64().is_some_and(|s| s > 0.0) {
                return None;
            }
        }
        Some(id)
    }

    /// A chat request's keep (X-Keep-Up-S / keep_up_s): `id` kept that long from now
    pub(crate) fn keep_chat(&self, id: &str, secs: u64) {
        self.chat_keep.lock().unwrap().insert(id.to_string(), now() + secs as f64);
    }

    /// The chat server's control socket (its /server routes are not on its port)
    pub(crate) fn chat_socket(&self) -> std::path::PathBuf {
        self.o.cfg.get("NS_SOCKET_DIR").map(std::path::PathBuf::from).unwrap_or_else(|| {
            std::path::PathBuf::from(std::env::var("XDG_RUNTIME_DIR").unwrap_or_else(|_| "/tmp".into())).join("nextcuda")
        }).join("nextcuda.sock")
    }

    /// Room for chat model `id` before it loads (it is not loaded): the chat model loaded now counts as free (it goes
    /// when this one comes); idle picture, sound and video work on its cards stops for it; else, after ROOM_WAIT_S, 503
    /// no_room - the answer the other kinds give
    pub(crate) fn chat_room(&self, id: &str) -> Result<(), (u16, String)> {
        let all = self.models();
        let Some(m) = all.iter().find(|m| m["id"] == id && registry::kind_of(m) == "llm") else { return Ok(()) };
        let cur = self.chat_loaded(&all);
        if cur.as_deref() == Some(id) {
            return Ok(());
        }
        // the chat server holds one model at a time: a pinned or held one is not swapped out for this one
        if let Some(c) = cur.filter(|c| self.kept_ids(&all).contains(c)) {
            let why = if pinned_ids(&all).contains(&c) { "pinned: ".to_string() + &c + " (unpin to free the room)" }
                      else { format!("held by rsvp_s: {c} ({:.0} s more)", self.held_s(&c)) };
            return Err((503, format!("the chat server holds one model at a time and {why}; {id} loads once it is released")));
        }
        let gpus = self.gpus_of(m);
        let t0 = Instant::now();
        loop {
            let cards = self.named_cards(&all);
            let card_mb = |g: usize| cards.get(g).and_then(|c| c["vram_total_mb"].as_u64()).unwrap_or(32768);
            let (need, _) = self.vouched_mb(m, card_mb(gpus[0]));
            // what the chat model loaded now holds on a card (it leaves)
            // (a pinned chat model does not leave: it is not counted as free)
            let leaving = !self.chat_loaded(&all).is_some_and(|c| self.kept_ids(&all).contains(&c));
            let chat_on = |g: usize| cards.get(g).and_then(|c| c["procs"].as_array()).into_iter().flatten()
                .filter(|p| p["kind"] == "llm" && leaving).map(|p| p["vram_mb"].as_u64().unwrap_or(0)).sum::<u64>();
            let want = |g: usize| (need.get(&g).copied().unwrap_or(0) + MARGIN_MB).saturating_sub(chat_on(g));
            let free = |g: usize| cards.get(g).map_or(0, |c| card_mb(g).saturating_sub(c["vram_used_mb"].as_u64().unwrap_or(0)));
            let short: Vec<usize> = gpus.iter().copied().filter(|g| free(*g) < want(*g)).collect();
            if short.is_empty() {
                return Ok(());
            }
            let mut freed = false;
            for g in &short {
                if let Some(plan) = self.room_plan_for(&cards, *g, want(*g), "llm", false, Some(m)) {
                    if self.free(&plan, "llm", id).is_ok() {
                        self.wait_room(*g, want(*g));
                        freed = true;
                    }
                }
            }
            if t0.elapsed() > Duration::from_secs(ROOM_WAIT_S) {
                let lack = short.iter().map(|g| self.no_room(&cards, &[*g], id, want(*g))).collect::<Vec<_>>().join(" / ");
                return Err((503, format!("{lack}; {id} loads once one of them idles out or is stopped")));
            }
            if !freed {
                std::thread::sleep(Duration::from_secs(3));
            }
        }
    }

    /// `id` started in slot `x` on card `gpu`, answering
    fn start_at(&self, kind: &str, id: &str, x: &Slot, gpu: usize) -> Result<(), (u16, String)> {
        // never more copies of a model on one card than its profile allows (max_copies_per_card)
        if let Ok(m) = self.entry(id) {
            let max = self.profile_of(&m).concurrency.max_copies_per_card as usize;
            let cards = self.cards.lock().unwrap().clone();
            let there = self.servers(kind).iter().filter(|o| o.slot != x.slot && o.model.as_deref() == Some(id) && card_of_port(&cards, o.port) == Some(gpu)).count();
            if there >= max {
                return Err((503, format!("{id} already has {there} cop{} on GPU {gpu}: its card's max_copies_per_card is {max}", if there == 1 { "y" } else { "ies" })));
            }
        }
        // a model whose engine splits its stages over its cards (its card's split: true - MiniMax Music 3's language
        // model on one, its flow stage on the other) gets them all, `gpu` first
        let on = match self.entry(id) {
            Ok(m) if m["split"] == true => {
                let mut v = vec![gpu];
                v.extend(self.gpus_of(&m).into_iter().filter(|g| *g != gpu));
                v.iter().map(|g| g.to_string()).collect::<Vec<_>>().join(",")
            }
            _ => gpu.to_string(),
        };
        let mut args = vec![kind.to_string(), "start".into(), id.into(), "--gpu".into(), on, "--port".into(), x.port.to_string(),
                            "--host".into(), self.o.serve_host.clone(), "--wfe".into(), "--slot".into(), x.slot.to_string()];
        // (a pinned model's server does not end itself when idle)
        args.extend(idle_exit_args(self.o.idle_minutes, self.entry(id).is_ok_and(|m| registry::pinned(&m))));
        self.loading.lock().unwrap().insert(id.to_string(), now());
        let r = self.exec(format!("{kind} start {id} on GPU {gpu}{}", if x.slot > 0 { format!(" (server {})", x.slot + 1) } else { String::new() }), &args);
        self.loading.lock().unwrap().remove(id);
        r.map_err(|e| (502, e))?;
        self.wait_up(x.port, id)
    }

    /// The longest keep-alive this box grants: NS_MAX_RSVP_S (unset or 0: a day)
    pub(crate) fn max_rsvp(&self) -> f64 {
        super::rsvp::cap_of(self.o.cfg.get("NS_MAX_RSVP_S"))
    }

    /// A request for model `id` has ended (answered, failed or cancelled): its hold runs `granted` seconds from now
    /// (never shortened by a later, shorter one); 0 releases it. The clock starts here, not when the request came.
    pub(crate) fn hold_after(&self, id: &str, granted: f64) {
        let mut h = self.holds.lock().unwrap();
        let next = super::rsvp::extend(h.get(id).copied(), now(), granted);
        h.insert(id.to_string(), next);
    }

    /// Seconds model `id` is held for from now (0: not held)
    pub(crate) fn held_s(&self, id: &str) -> f64 {
        super::rsvp::left(self.holds.lock().unwrap().get(id), now())
    }

    /// The models kept out of room planning and idle unloading: the pinned and the held
    pub(crate) fn kept_ids(&self, all: &[Value]) -> BTreeSet<String> {
        kept_from(all, &self.holds.lock().unwrap(), now())
    }


    /// A request on its way to the server on `port` (counted until the lease is dropped)
    fn lease(&self, port: u16) -> Lease<'_> {
        *self.inflight.lock().unwrap().entry(port).or_insert(0) += 1;
        let model = self.answers.lock().unwrap().get(&port).map(|a| a.0.clone()).filter(|m| !m.is_empty());
        let hold = model.zip(super::rsvp::granted(super::rsvp::asked(), self.max_rsvp()));
        Lease { target: Target::Tcp(format!("127.0.0.1:{port}")), port, home: self, hold }
    }

    fn wait_up(&self, port: u16, id: &str) -> Result<(), (u16, String)> {
        let t0 = Instant::now();
        while t0.elapsed() < Duration::from_secs(600) {
            if self.server_at(port).is_some_and(|(m, l, _)| m == id && !l) {
                return Ok(());
            }
            std::thread::sleep(Duration::from_secs(1));
        }
        Err((504, format!("{id} did not answer in 10 minutes")))
    }

    /// The video daemon as the enabled video models say: their engines on their cards, or stopped when none is
    /// enabled; restarted only when that changes (and not while a clip renders, unless `force`). A pin or hold on a
    /// video model never restarts it: the keeper asks the daemon to keep its workers (`/engine/keep`) for as long as
    /// one lasts, and an unpin or the end of a hold simply stops the asking - the daemon's idle unload is back
    pub(crate) fn video_apply(&self, force: bool) -> Result<(), String> {
        let all = self.models();
        let enabled: Vec<&Value> = all.iter().filter(|m| registry::kind_of(m) == "video" && m["enabled"] != false).collect();
        if all.iter().all(|m| registry::kind_of(m) != "video") {
            return Ok(());
        }
        let _g = self.kind_lock["video"].lock().unwrap();
        let running = self.video_daemon();
        let marker = std::path::PathBuf::from(self.o.cfg.get("NS_SOCKET_DIR").unwrap_or_else(|| "/tmp".into())).join("video.disabled");
        if enabled.is_empty() {
            let _ = std::fs::write(&marker, "no video model is enabled (nextcuda serve)\n");
            if running.is_some() {
                if !force && self.rendering() {
                    return Err("a clip is rendering; the daemon stops when the queue is idle (or force)".into());
                }
                self.exec("video stop (no video model enabled)".into(), &["video".into(), "stop".into()])?;
            }
            return Ok(());
        }
        let _ = std::fs::remove_file(&marker);
        // the studio (the queue, the page, what video.* reaches): a plain process on this box, started with the daemon
        if !listening(self.o.ports.video) {
            self.exec("video serve (the studio)".into(), &["video".into(), "serve".into(), "--port".into(), self.o.ports.video.to_string()])?;
        }
        // (the first model is minimax-h3 when it is enabled, the others are `--engine`; tooling/deploy.sh brings the
        // daemon back with the same ones)
        let (first, others) = video_engines(&video_enabled(&all));
        let mut want: Vec<String> = vec![first.clone()];
        want.extend(others);
        let mut gpus: Vec<usize> = enabled.iter().flat_map(|m| self.gpus_of(m)).collect();
        gpus.sort();
        gpus.dedup();
        if let Some((names, loaded)) = &running {
            let (mut a, mut b) = (names.clone(), want.clone());
            a.sort();
            b.sort();
            let _ = &loaded;
            if a == b {
                return Ok(());
            }
            if !force && self.rendering() {
                return Err("a clip is rendering; the engines change when the queue is idle (or force)".into());
            }
            self.exec("video stop (the enabled models changed)".into(), &["video".into(), "stop".into()])?;
        }
        let mut args = vec!["video".to_string(), "start".into(), "--model".into(), first];
        for o in &want[1..] {
            args.extend(["--engine".into(), o.clone()]);
        }
        for g in gpus {
            args.extend(["--gpu".into(), g.to_string()]);
        }
        self.exec(format!("video start ({})", want.join(", ")), &args).map(|_| ())
    }

    /// A video request's first step: an enabled video model whose engine the daemon lacks (not running, crashed, or
    /// started before the model was enabled) has the daemon applied now (`video_apply`), and the request waits for
    /// it - VIDEO_WAIT_S at most, then 503. Nothing enabled: nothing to do (the request answers that no video model is
    /// enabled). A clip rendering keeps the daemon it has (the engines change when the queue is idle). A stop for a
    /// model change is video_apply's, under the same lock: this never starts what it is stopping.
    pub(crate) fn video_ensure(&self) -> Result<(), (u16, String)> {
        self.video_ensure_within(Duration::from_secs(VIDEO_WAIT_S))
    }

    pub(crate) fn video_ensure_within(&self, limit: Duration) -> Result<(), (u16, String)> {
        let enabled = video_enabled(&self.models());
        let running = self.video_daemon().map(|d| d.0);
        if !video_missing(&enabled, running.as_deref()) {
            return Ok(());
        }
        if let Err(e) = self.video_apply(false) {
            // a daemon that runs and cannot change now (a clip renders) answers with the engines it has
            return if running.is_some() { Ok(()) } else { Err((502, format!("the video engine is not running and did not start: {e}"))) };
        }
        let t0 = Instant::now();
        while t0.elapsed() < limit {
            if !video_missing(&enabled, self.video_daemon().map(|d| d.0).as_deref()) {
                return Ok(());
            }
            std::thread::sleep(Duration::from_secs(1));
        }
        Err((504, format!("the video engine did not come up in {} s (nextcuda video start; the runs on this page have its log)", limit.as_secs())))
    }

    /// A queued video clip watched to its end (it leaves the studio's queue when it finishes, fails or is cancelled): the
    /// hold of its rsvp_s starts then. `every`: seconds between looks. Runs on its own thread.
    pub(crate) fn watch_clip(&self, label: String, model: String, granted: f64, every: f64) {
        let Some(me) = self.me.get().and_then(std::sync::Weak::upgrade) else { return };
        std::thread::spawn(move || {
            let q: String = label.chars().map(|c| if c.is_ascii_alphanumeric() || "-_.".contains(c) { c.to_string() } else { format!("%{:02X}", c as u32 & 0xff) }).collect();
            let t0 = Instant::now();
            while t0.elapsed() < Duration::from_secs(7 * 86400) {
                // 200 while the clip is running, held or queued; 404 once it is not
                match crate::http::call_status(&me.target("video"), "GET", &format!("/api/clip?label={q}"), None, 5) {
                    Ok((200, _)) | Err(_) => {}
                    Ok(_) => break,
                }
                std::thread::sleep(Duration::from_secs_f64(every));
            }
            me.hold_after(&model, granted);
        });
    }

    /// The pinned, enabled models (not video: its daemon is video_apply's) loaded in the background: at serve start
    pub(crate) fn load_pinned(self: &Arc<Self>) {
        for m in self.models().iter().filter(|m| registry::pinned(m) && m["enabled"] != false && registry::kind_of(m) != "video") {
            let _ = self.load_async(m["id"].as_str().unwrap_or(""));
        }
    }

    /// Every 10 s the models that are kept (pinned, or held by an rsvp_s) are touched - a running server's `--idle-exit`
    /// clock is the server's own, and GET / is what a page load is, work to it - and a model whose last rsvp_s was 0
    /// goes when it is idle. Every 30 s: the video daemon back when it is gone (twice in a row: a deploy stops and
    /// starts it within seconds), and a pinned model loaded again when unloaded (every 5 min at most; a failed load
    /// is a run)
    pub(crate) fn keeper(self: &Arc<Self>) {
        let mut gone = 0;
        let mut tick = 0u32;
        let mut tried: BTreeMap<String, Instant> = BTreeMap::new();
        loop {
            std::thread::sleep(Duration::from_secs(10));
            self.keep_tick();
            tick += 1;
            if tick % 3 != 0 {
                continue;
            }
            let all = self.models();
            let enabled = video_enabled(&all);
            gone = if video_missing(&enabled, self.video_daemon().map(|d| d.0).as_deref()) { gone + 1 } else { 0 };
            if gone >= 2 {
                if let Err(e) = self.video_apply(false) {
                    eprintln!("video: {e}");
                }
                gone = 0;
            }
            for m in all.iter().filter(|m| registry::pinned(m) && m["enabled"] != false && registry::kind_of(m) != "video") {
                let id = s(m, "id");
                let state = self.model_states(std::slice::from_ref(m), &json!([])).first().map(|x| s(x, "state")).unwrap_or_default();
                if state == "unloaded" && tried.get(&id).is_none_or(|t| t.elapsed() > Duration::from_secs(300)) {
                    tried.insert(id.clone(), Instant::now());
                    let _ = self.load_async(&id);
                }
            }
        }
    }

    /// One pass of the keeper's 10 s work: touch what is kept, release what asked to go
    pub(crate) fn keep_tick(&self) {
        let all = self.models();
        let kept = self.kept_ids(&all);
        let releasing: Vec<String> = self.holds.lock().unwrap().iter().filter(|(id, h)| h.release && !kept.contains(*id)).map(|(id, _)| id.clone()).collect();
        for m in all.iter().filter(|m| matches!(registry::kind_of(m), "image" | "audio" | "model3d")) {
            let (id, kind) = (s(m, "id"), registry::kind_of(m));
            let Some(x) = self.servers(kind).into_iter().find(|x| x.model.as_deref() == Some(id.as_str())) else { continue };
            if kept.contains(&id) {
                let _ = call_for(&Target::Tcp(format!("127.0.0.1:{}", x.port)), "GET", "/", None, 3);
            } else if releasing.contains(&id) && !x.busy && !x.loading && self.inflight.lock().unwrap().get(&x.port).copied().unwrap_or(0) == 0 {
                // (rsvp_s 0: gone as soon as idle)
                self.holds.lock().unwrap().remove(&id);
                let _ = self.unload(&id);
            }
        }
        self.withdraw_waiting();
        // a pinned or held video model: the daemon keeps its workers loaded 60 s from this tick (asked again every tick)
        if all.iter().any(|m| registry::kind_of(m) == "video" && m["enabled"] != false && kept.contains(&s(m, "id"))) {
            let _ = call_for(&self.target("video"), "POST", "/engine/keep", Some(&json!({"seconds": 60})), 3);
        }
        for id in releasing {
            if self.chat_idle().as_deref() == Some(id.as_str()) {
                self.holds.lock().unwrap().remove(&id);
                let _ = self.unload(&id);
            }
        }
    }
}

/// The pinned models holding memory on the named cards
fn pinned_holders(cards: &Value, gpus: &[usize], pinned: &BTreeSet<String>) -> Vec<String> {
    let mut v: Vec<String> = gpus.iter().filter_map(|g| cards.get(*g)).flat_map(|c| c["procs"].as_array().into_iter().flatten())
        .filter_map(|p| p["model_id"].as_str().or(p["model"].as_str())).filter(|m| pinned.contains(*m)).map(str::to_string).collect();
    v.sort();
    v.dedup();
    v
}

/// Card `g`'s free memory as the sampler has it (its total less what is used), MiB
fn card_free(cards: &Value, g: usize) -> u64 {
    cards.get(g).map_or(0, |c| c["vram_total_mb"].as_u64().unwrap_or(0).saturating_sub(c["vram_used_mb"].as_u64().unwrap_or(0)))
}

/// Whether card `g` may take another copy of a model that has copies on the cards `on` (one entry a copy), up to `per_card` there
fn extra_copy_allowed(on: &[usize], g: usize, per_card: usize) -> bool {
    on.iter().filter(|c| **c == g).count() < per_card
}

/// `plan_room` with the residents that may not stay: each of `residents` that `blocks` (all of them when `exclusive`)
/// goes with the idle holders, and when one is not among the idle holders (busy, pinned, held) there is no plan
fn plan_with_blockers(free: u64, want: u64, idle: Vec<Holder>, residents: &[String], exclusive: bool, blocks: &dyn Fn(&str) -> bool) -> Option<Vec<Holder>> {
    let must = |r: &str| exclusive || blocks(r);
    let blockers: Vec<&String> = residents.iter().filter(|r| must(r)).collect();
    if blockers.is_empty() {
        return plan_room(free, want, idle);
    }
    let (mut forced, idle): (Vec<Holder>, Vec<Holder>) = idle.into_iter().partition(|h| must(&h.model));
    // every blocker is a holder that can go
    let mut gone: Vec<&str> = forced.iter().map(|h| h.model.as_str()).collect();
    for b in &blockers {
        match gone.iter().position(|g| g == &b.as_str()) {
            Some(i) => {
                gone.remove(i);
            }
            None => return None,
        }
    }
    let freed: u64 = forced.iter().map(|h| h.mb).sum();
    let mut plan = plan_room(free + freed, want, idle)?;
    forced.append(&mut plan);
    Some(forced)
}

/// What of a card's idle holders must go so `want` MiB is free (it has `free`): none (Some(empty)), the lightest to
/// bring back first - picture and sound servers, then the video worker, then a chat model - or no way (None)
fn plan_room(free: u64, want: u64, mut idle: Vec<Holder>) -> Option<Vec<Holder>> {
    idle.sort_by_key(|h| (h.order, std::cmp::Reverse(h.mb)));
    let mut got = free;
    let mut plan = Vec::new();
    for h in idle {
        if got >= want {
            break;
        }
        got += h.mb;
        plan.push(h);
    }
    (got >= want).then_some(plan)
}

/// The card to load on, of (card, its free MiB, its plan) in the model's order: one with the room now - the one with the
/// most free, so an estimate on the low side does not land the model beside another on a full card - before the first
/// whose room needs idle work stopped
fn pick_card(plans: impl Iterator<Item = (usize, u64, Option<Vec<Holder>>)>) -> Option<(usize, Vec<Holder>)> {
    let mut later = None;
    let mut now: Option<(usize, u64, Vec<Holder>)> = None;
    for (g, free, plan) in plans {
        match plan {
            Some(p) if p.is_empty() => {
                if now.as_ref().is_none_or(|n| free > n.1) {
                    now = Some((g, free, p));
                }
            }
            Some(p) => {
                later.get_or_insert((g, p));
            }
            None => {}
        }
    }
    now.map(|(g, _, p)| (g, p)).or(later)
}

/// A server's "needs about X GiB on <card>; Y GiB are free" (its refusal to load), as (X, Y) in MiB
fn asked_mb(e: &str) -> Option<(u64, u64)> {
    let gib = |t: &str| t.trim().parse::<f64>().ok().map(|g| (g * 1024.0) as u64);
    let need = e.split("needs about ").nth(1)?.split(" GiB").next()?;
    let free = e.split(" GiB are free").next()?.rsplit(' ').next()?;
    Some((gib(need)?, gib(free)?))
}

/// What a card must have free, as the front door counts it (`front_free`: total less used), for a server that asked
/// `need` MiB and found `had` free: the difference is its context and what the driver keeps (half a GiB at least: the
/// sample may still count the server that ended), with a quarter GiB for the asking's rounding
fn ask_mb(need: u64, had: u64, front_free: u64) -> u64 {
    need + front_free.saturating_sub(had).max(512) + 256
}

// ---- ETAs (history.rs), max_willing_to_wait_s ----

impl Home {
    /// The model a request is for: the one it names, else the kind's pick
    pub(crate) fn eta_model(&self, method: &str, b: &Value, all: &[Value]) -> String {
        let kind = super::jobs::kind_of_method(method);
        match b["model"].as_str().filter(|m| !m.is_empty()) {
            Some(m) => m.to_string(),
            None => self.pick(kind, b, all).unwrap_or_default(),
        }
    }

    /// A job's expected run in seconds: its shape's median, else its model's (None: no history)
    fn job_run_est(&self, j: &Value) -> Option<f64> {
        let body = &j["body"];
        let all = self.models();
        let method = s(j, "method");
        let model = self.eta_model(&method, body, &all);
        self.history.eta(&method, &model, body, true, Some(0.0))["run_s"].as_f64()
    }

    /// The ETA of a request of `method` with body `b`, now: `exclude` is its own job when it is already queued (only the
    /// jobs ahead of it count). The wait is 0 when its model is loaded with a server free; else the work ahead of it on
    /// the kind's servers (running jobs' remaining time and queued jobs ahead, each from history) over the servers, plus
    /// the model's spin-up from history; unknown when any part has no history. Nothing is started or changed.
    pub(crate) fn eta_for(&self, method: &str, b: &Value, exclude: Option<&str>) -> Value {
        let all = self.models();
        let model = self.eta_model(method, b, &all);
        let kind = super::jobs::kind_of_method(method);
        let (hot, ahead) = match kind {
            "image" | "audio" | "model3d" => {
                let sv = self.servers(kind);
                let inflight = |p: u16| self.inflight.lock().unwrap().get(&p).copied().unwrap_or(0);
                let hot = sv.iter().any(|x| x.model.as_deref() == Some(model.as_str()) && !x.loading && !x.busy && inflight(x.port) == 0);
                let mine_order = exclude.and_then(|id| self.jobs.get(id)).map(|j| j["order"].as_f64().or_else(|| j["created"].as_f64()).unwrap_or(0.0));
                let jobs: Vec<Value> = self.jobs.list.lock().unwrap().iter().filter(|j| j["kind"] == kind && Some(s(j, "id").as_str()) != exclude)
                    .filter(|j| j["state"] == "running" || (j["state"] == "queued" && mine_order.is_none_or(|o| j["order"].as_f64().or_else(|| j["created"].as_f64()).unwrap_or(0.0) < o))).cloned().collect();
                let mut total = Some(0.0);
                for j in &jobs {
                    let est = self.job_run_est(j);
                    let left = est.map(|e| if j["state"] == "running" { (e - (now() - j["started"].as_f64().unwrap_or_else(now))).max(0.0) } else { e });
                    total = total.zip(left).map(|(a, b)| a + b);
                }
                (hot, total.map(|t| t / sv.len().max(1) as f64))
            }
            "llm" => {
                let hot = self.chat_loaded(&all).as_deref() == Some(model.as_str());
                (hot, if hot { Some(0.0) } else { None })
            }
            // the studio's queue is its own
            _ => {
                let up = self.video_daemon().is_some_and(|d| d.0.contains(&model));
                let hot = up && self.video_idle();
                (hot, if hot { Some(0.0) } else { None })
            }
        };
        self.history.eta(method, &model, b, hot, ahead)
    }

    /// `max_willing_to_wait_s` against the ETA, before anything is queued, loaded or evicted: the refusal (503 with the
    /// wait_too_long fields) when the ETA's wait is known and longer
    pub(crate) fn check_wait(&self, method: &str, b: &Value, max: f64) -> Result<(), (u16, String)> {
        let eta = self.eta_for(method, b, None);
        match eta["wait_s"].as_f64() {
            Some(w) if w > max => Err((503, super::rsvp::wait_tag(json!({"eta": eta, "max_willing_to_wait_s": max}),
                                                                  &format!("it would start in about {w:.0} s, past the {max:.0} s you will wait; nothing was started for it")))),
            _ => Ok(()),
        }
    }

    /// The error for a request withdrawn after `waited` s without starting (the ETA was unknown)
    pub(crate) fn wait_expired(&self, method: &str, b: &Value, max: f64, waited: f64) -> (u16, String) {
        (503, super::rsvp::wait_tag(json!({"eta": self.eta_for(method, b, None), "max_willing_to_wait_s": max, "waited_s": waited.round()}),
                                    &format!("it had not started after {waited:.0} s, past the {max:.0} s you will wait; withdrawn")))
    }

    /// A finished work request into the history (a model is needed: none, nothing is recorded)
    pub(crate) fn record_work(&self, method: &str, b: &Value, model: &str, wait_s: f64, run_s: f64, outcome: &str) {
        if model.is_empty() {
            return;
        }
        self.history.record(super::history::Record { method: method.into(), model: model.into(), shape: super::history::shape_of(method, model, b),
                                                     wait_s, run_s, outcome: outcome.into() });
    }

    /// Queued jobs that will not start in time for their `max_willing_to_wait_s` are withdrawn: failed with
    /// wait_too_long - by the clock (waited past it), or by an updated ETA (the queue moved, a holder got busier)
    pub(crate) fn withdraw_waiting(&self) {
        let queued: Vec<Value> = self.jobs.list.lock().unwrap().iter().filter(|j| j["state"] == "queued" && j["body"]["max_willing_to_wait_s"].is_number()).cloned().collect();
        for j in queued {
            let (id, method, body) = (s(&j, "id"), s(&j, "method"), j["body"].clone());
            let max = body["max_willing_to_wait_s"].as_f64().unwrap_or(0.0);
            let waited = now() - j["created"].as_f64().unwrap_or_else(now);
            let eta = self.eta_for(&method, &body, Some(&id));
            let late = waited > max || eta["wait_s"].as_f64().is_some_and(|w| waited + w > max);
            if !late {
                continue;
            }
            let err = (503, super::rsvp::wait_tag(json!({"eta": eta, "max_willing_to_wait_s": max, "waited_s": waited.round()}),
                                                  &format!("it would not start within the {max:.0} s you will wait ({waited:.0} s gone); withdrawn")));
            self.jobs.fail_queued(&id, &method, err);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_recogniser_transcribes_and_speaks_nothing() {
        let (caps, methods) = capabilities(&json!({"id": "whisper-large-v3-turbo", "kind": "audio", "arch": "whisper"}));
        assert_eq!(methods, vec!["audio.transcribe"]);
        assert!(caps.contains(&"word_timestamps") && !caps.contains(&"speech"));
        let (_, methods) = capabilities(&json!({"id": "voxcpm2", "kind": "audio", "arch": "voxcpm2", "speech": ["clone"]}));
        assert!(methods.contains(&"audio.speak") && !methods.contains(&"audio.transcribe"));
    }

    #[test]
    fn a_clone_that_takes_a_style_is_voice_direction() {
        let (caps, _) = capabilities(&json!({"id": "voxcpm2", "kind": "audio", "arch": "voxcpm2", "speech": ["design", "clone", "direction"]}));
        assert!(caps.contains(&"voice_direction") && caps.contains(&"clone"));
        let (caps, _) = capabilities(&json!({"id": "qwen3-tts-base", "kind": "audio", "arch": "qwen3-tts", "speech": ["clone"]}));
        assert!(!caps.contains(&"voice_direction"));
    }

    #[test]
    fn a_chat_model_split_over_every_card_has_no_choice_of_cards() {
        let cards = vec![
            json!({"id": "glm-5.3-flash-uncensored", "kind": "llm", "gpus": "all"}),
            json!({"id": "qwen3.8-27b-q8", "kind": "llm", "gpus": "0"}),
            json!({"id": "breeze-tts-2", "kind": "audio", "gpus": "all"}),
        ];
        // the registry entry may say one card (set before): the model card decides
        assert!(spans_all(&json!({"id": "glm-5.3-flash-uncensored", "kind": "llm", "gpus": "0"}), &cards));
        // a variant registered under its own id, its card named by `catalog`
        assert!(spans_all(&json!({"id": "glm-128k", "catalog": "glm-5.3-flash-uncensored", "kind": "llm"}), &cards));
        assert!(!spans_all(&json!({"id": "qwen3.8-27b-q8", "kind": "llm"}), &cards), "a one-card chat model");
        assert!(!spans_all(&json!({"id": "breeze-tts-2", "kind": "audio"}), &cards), "an audio model's all is where its copies may go");
        assert!(!spans_all(&json!({"id": "my-own", "kind": "llm", "gpus": "all"}), &cards), "no model card: a choice");
    }

    // brainiac's two RTX 5070s: breeze-tts-2 idle in audio server 1 on GPU 0 (5.9 GiB), GPU 1 all but empty;
    // qwen3-tts-custom (seen at 3914 MiB) asked for
    fn two_cards() -> Value {
        json!([{"vram_total_mb": 12227, "vram_used_mb": 6134}, {"vram_total_mb": 12288, "vram_used_mb": 157}])
    }

    fn breeze() -> Vec<Holder> {
        vec![Holder { what: Free::Server("audio".into(), 0), mb: 6040, model: "breeze-tts-2".into(), order: 0 }]
    }

    fn plans(cards: &Value, gpus: &[usize], want: u64) -> Option<(usize, Vec<Holder>)> {
        pick_card(gpus.iter().map(|&g| (g, card_free(cards, g), plan_room(card_free(cards, g), want, if g == 0 { breeze() } else { vec![] }))))
    }

    #[test]
    fn a_model_goes_to_the_card_with_room_not_beside_another() {
        let cards = two_cards();
        let want = 3914 + MARGIN_MB;
        // GPU 0 has the room too, by the estimate (it had not: the server asked 5.6 GiB with 5.3 free): the emptier card
        let (g, plan) = plans(&cards, &[0, 1], want).unwrap();
        assert_eq!((g, plan.len()), (1, 0));
        // a card with room now before an earlier one whose room needs breeze stopped
        let (g, plan) = plans(&cards, &[0, 1], 7000).unwrap();
        assert_eq!((g, plan.len()), (1, 0));
    }

    #[test]
    fn a_refusal_for_memory_plans_the_idle_server_away() {
        let cards = two_cards();
        let e = "audio start qwen3-tts-custom on GPU 0 (server 2): nextcuda: nextcuda-audio-1 ended while loading (nextcuda: qwen3-tts needs \
                 about 5.6 GiB on NVIDIA GeForce RTX 5070; 5.3 GiB are free): nextcuda audio logs";
        let (need, had) = asked_mb(e).unwrap();
        assert_eq!((need, had), (5734, 5427));
        let ask = ask_mb(need, had, card_free(&cards, 0));
        assert!(ask > card_free(&cards, 0) && ask > 3914 + MARGIN_MB, "{ask}");
        // the model on GPU 0 only (its entry's gpus: "0"): breeze goes first
        let (g, plan) = plans(&cards, &[0], ask).unwrap();
        assert_eq!(g, 0);
        assert!(matches!(&plan[..], [Holder { what: Free::Server(k, 0), .. }] if k == "audio"));
        // on both cards: the empty one, nothing stopped
        let (g, plan) = plans(&cards, &[0, 1], ask).unwrap();
        assert_eq!((g, plan.len()), (1, 0));
        assert_eq!(asked_mb("qwen3.8 needs about 30.1 GiB on X (28.0 GiB of weights); 22.5 GiB are free").map(|x| x.0), Some(30822));
        assert_eq!(asked_mb("nextcuda-audio-1 ended while loading"), None);
    }

    fn holder(model: &str, order: u8) -> Holder {
        Holder { what: Free::Server("audio".into(), 0), mb: 6040, model: model.into(), order }
    }

    #[test]
    fn a_pinned_model_is_not_an_evictable_holder() {
        let all = vec![json!({"id": "breeze-tts-2", "kind": "audio", "pinned": true}), json!({"id": "other", "kind": "audio"})];
        let pinned = pinned_ids(&all);
        assert_eq!(pinned.iter().collect::<Vec<_>>(), vec!["breeze-tts-2"]);
        let left = evictable(vec![holder("breeze-tts-2", 0), holder("other", 0)], &pinned);
        assert_eq!(left.len(), 1);
        assert_eq!(left[0].model, "other");
        // the plan for the room: breeze alone held it, pinned - no plan; unpinned it is the one to stop
        let cards = two_cards();
        assert!(plan_room(card_free(&cards, 0), 7000, evictable(breeze(), &pinned)).is_none());
        assert_eq!(plan_room(card_free(&cards, 0), 7000, evictable(breeze(), &BTreeSet::new())).map(|p| p.len()), Some(1));
        // (a pin on another model changes nothing)
        assert_eq!(plan_room(card_free(&cards, 0), 7000, evictable(breeze(), &pinned_ids(&[json!({"id": "x", "pinned": true})]))).map(|p| p.len()), Some(1));
    }

    #[test]
    fn a_pinned_model_is_started_without_the_idle_exit() {
        assert_eq!(idle_exit_args(10, false), vec!["--idle-exit", "600"]);
        assert!(idle_exit_args(10, true).is_empty());
        assert!(idle_exit_args(0, false).is_empty());
    }

    #[test]
    fn the_pin_is_in_the_entry_and_out_of_the_lora_free_ids() {
        assert!(registry::pinned(&json!({"id": "a", "pinned": true})));
        assert!(!registry::pinned(&json!({"id": "a", "pinned": false})));
        assert!(!registry::pinned(&json!({"id": "a"})), "an entry from before pins is not pinned");
        assert!(!registry::pinned(&json!({"id": "a", "pinned": "yes"})));
    }

    #[test]
    fn the_pinned_models_on_a_card_are_named_for_no_room() {
        let cards = json!([{"procs": [{"model_id": "qwen-image", "vram_mb": 9000}, {"model": "breeze-tts-2", "vram_mb": 900}]}, {"procs": []}]);
        let pinned: BTreeSet<String> = ["qwen-image".to_string()].into();
        assert_eq!(pinned_holders(&cards, &[0, 1], &pinned), vec!["qwen-image"]);
        assert!(pinned_holders(&cards, &[1], &pinned).is_empty());
    }

    fn ids(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn the_daemon_is_applied_first_when_an_enabled_engine_is_missing() {
        let all = vec![json!({"id": "minimax-h3", "kind": "video"}), json!({"id": "ltx2", "kind": "video", "enabled": true}),
                       json!({"id": "old", "kind": "video", "enabled": false}), json!({"id": "x", "kind": "image"})];
        let want = video_enabled(&all);
        assert_eq!(want, ids(&["minimax-h3", "ltx2"]));
        // not running
        assert!(video_missing(&want, None));
        // running without ltx2 (it was enabled after the daemon started)
        assert!(video_missing(&want, Some(&ids(&["minimax-h3"]))));
        // running with them (an extra one is video_apply's business, not a request's)
        assert!(!video_missing(&want, Some(&ids(&["ltx2", "minimax-h3", "old"]))));
        // no video model enabled: nothing to apply, the request is refused as it was
        assert!(!video_missing(&[], None));
    }

    #[test]
    fn the_daemons_first_model_is_h3_when_enabled() {
        assert_eq!(video_engines(&ids(&["ltx2", "minimax-h3"])), ("minimax-h3".to_string(), ids(&["ltx2"])));
        assert_eq!(video_engines(&ids(&["ltx2"])), ("ltx2".to_string(), vec![]));
        assert_eq!(video_engines(&[]), (String::new(), vec![]));
    }

    fn h(model: &str, mb: u64) -> Holder {
        Holder { what: Free::Server("audio".into(), 0), mb, model: model.into(), order: 0 }
    }

    #[test]
    fn an_exclusive_model_clears_its_card_of_idle_residents_and_waits_for_the_rest() {
        let res = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        let none = |_: &str| false;
        // lots of room, but exclusive: the idle resident goes anyway
        let plan = plan_with_blockers(9000, 2000, vec![h("a", 3000)], &res(&["a"]), true, &none).unwrap();
        assert_eq!(plan.iter().map(|x| x.model.as_str()).collect::<Vec<_>>(), vec!["a"]);
        // a resident that cannot go (busy, pinned, held: not among the idle holders): no plan, the card is no choice
        assert!(plan_with_blockers(9000, 2000, vec![h("a", 3000)], &res(&["a", "busy-one"]), true, &none).is_none());
        // not exclusive and nothing blocks: the plain plan (nothing to stop with room)
        assert_eq!(plan_with_blockers(9000, 2000, vec![h("a", 3000)], &res(&["a"]), false, &none).unwrap().len(), 0);
        // a resident that does not share with the newcomer (shared_with) goes even with room; one that does stays
        let blocks = |m: &str| m == "a";
        let plan = plan_with_blockers(9000, 2000, vec![h("a", 3000), h("b", 3000)], &res(&["a", "b"]), false, &blocks).unwrap();
        assert_eq!(plan.iter().map(|x| x.model.as_str()).collect::<Vec<_>>(), vec!["a"]);
        // the freed memory counts toward the room wanted
        let plan = plan_with_blockers(1000, 5000, vec![h("a", 4500), h("b", 3000)], &res(&["a", "b"]), false, &blocks).unwrap();
        assert_eq!(plan.iter().map(|x| x.model.as_str()).collect::<Vec<_>>(), vec!["a"]);
    }

    #[test]
    fn extra_copies_stop_at_the_profiles_count_a_card() {
        // one a card (the default): a card with a copy takes none, another card does
        assert!(!extra_copy_allowed(&[0], 0, 1));
        assert!(extra_copy_allowed(&[0], 1, 1));
        // two worth running: a second on the same card, not a third
        assert!(extra_copy_allowed(&[0], 0, 2));
        assert!(!extra_copy_allowed(&[0, 0], 0, 2));
        assert!(extra_copy_allowed(&[0, 0], 1, 2));
    }

    #[test]
    fn a_job_waits_as_long_as_the_caller_will_not_the_default() {
        let d = Duration::from_secs(90);
        assert_eq!(wait_limit(d, None), d);
        // the caller's limit replaces the default whether it is shorter or longer, and enforces itself
        assert_eq!(wait_limit(d, Some(5.0)), Duration::MAX);
        assert_eq!(wait_limit(Duration::from_secs(1800), Some(7200.0)), Duration::MAX);
    }
}
