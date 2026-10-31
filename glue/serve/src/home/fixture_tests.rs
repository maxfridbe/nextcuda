//! The backplane's lifecycle against a fake box, no GPU: the program it runs (`exe`) is a shell script that logs its
//! arguments and "starts" and "stops" servers by writing and removing a file; the image/audio servers and the video
//! studio are threads on 127.0.0.1 that answer from those files; the card sampler is a thread that reads them too.
//! What is asserted is what the front door ran (the log) and what it answered.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::registry::Store;
use super::{now, Home, Options, Ports};
use serde_json::{json, Value};

struct Mem {
    models: Mutex<Vec<Value>>,
    settings: Mutex<std::collections::BTreeMap<String, String>>,
    file: PathBuf,
}

struct Shared(Arc<Mem>);

impl Store for Shared {
    fn models(&self) -> Result<Vec<Value>, String> {
        Ok(self.0.models.lock().unwrap().clone())
    }
    fn update(&self, id: &str, f: &mut dyn FnMut(&mut Value)) -> Result<Value, String> {
        let mut v = self.0.models.lock().unwrap();
        let m = v.iter_mut().find(|m| m["id"] == id).ok_or("no model")?;
        f(m);
        Ok(m.clone())
    }
    fn registry_file(&self) -> PathBuf {
        self.0.file.clone()
    }
    fn get(&self, name: &str) -> Option<String> {
        self.0.settings.lock().unwrap().get(name).cloned()
    }
    fn in_env(&self, _: &str) -> bool {
        false
    }
    fn file(&self) -> PathBuf {
        PathBuf::new()
    }
    fn guide(&self, _: &Value) -> Option<String> {
        None
    }
}

struct World {
    dir: PathBuf,
    home: Arc<Home>,
    mem: Arc<Mem>,
    stop: Arc<AtomicBool>,
    audio: u16,
}

impl Drop for World {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn read_head(s: &mut std::net::TcpStream) -> String {
    let mut buf = Vec::new();
    let mut b = [0u8; 1024];
    while !buf.windows(4).any(|w| w == b"\r\n\r\n") {
        match s.read(&mut b) {
            Ok(0) | Err(_) => break,
            Ok(n) => buf.extend_from_slice(&b[..n]),
        }
    }
    // (the body too, or closing on it resets the caller's connection)
    let head = String::from_utf8_lossy(&buf).into_owned();
    let len = head.lines().find_map(|l| l.to_ascii_lowercase().strip_prefix("content-length:").and_then(|v| v.trim().parse::<usize>().ok())).unwrap_or(0);
    let have = buf.len() - buf.windows(4).position(|w| w == b"\r\n\r\n").map_or(0, |i| i + 4);
    let mut rest = vec![0u8; len.saturating_sub(have)];
    let _ = s.read_exact(&mut rest);
    head
}

fn reply(s: &mut std::net::TcpStream, status: &str, body: &Value) {
    let b = body.to_string();
    let _ = write!(s, "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{b}", b.len());
}

/// A kind's server: answers /api/info and /api/progress from model-PORT and busy-PORT files; with no model file it
/// does not answer (nothing is loaded)
fn serve_kind(l: TcpListener, dir: PathBuf, port: u16) {
    std::thread::spawn(move || {
        for mut s in l.incoming().flatten() {
            let head = read_head(&mut s);
            let Ok(model) = std::fs::read_to_string(dir.join(format!("model-{port}"))) else { continue };
            if head.starts_with("GET /api/info") {
                reply(&mut s, "200 OK", &json!({"model": model.trim(), "loading": false}));
            } else {
                reply(&mut s, "200 OK", &json!({"busy": dir.join(format!("busy-{port}")).exists()}));
            }
        }
    });
}

/// The video studio: /engine/status from the `engines` file (no file: the daemon is down), /rpc/status idle unless
/// a `renders` file is there
fn serve_studio(l: TcpListener, dir: PathBuf) {
    std::thread::spawn(move || {
        for mut s in l.incoming().flatten() {
            let head = read_head(&mut s);
            if head.contains("/engine/status") {
                match std::fs::read_to_string(dir.join("engines")) {
                    Ok(e) => {
                        let engines: Vec<Value> = e.trim().split(',').filter(|x| !x.is_empty()).map(|n| json!({"name": n})).collect();
                        reply(&mut s, "200 OK", &json!({"engines": engines, "gpus": [], "unload_after_idle_seconds": 600}));
                    }
                    Err(_) => reply(&mut s, "502 Bad Gateway", &json!({"error": "the daemon does not answer"})),
                }
            } else if head.contains("/engine/keep") {
                let mut f = std::fs::OpenOptions::new().create(true).append(true).open(dir.join("keeps")).unwrap();
                let _ = writeln!(f, "keep");
                reply(&mut s, "200 OK", &json!({"keep_until": 0}));
            } else if let Some(i) = head.find("/api/clip?label=") {
                let label: String = head[i + 16..].chars().take_while(|c| c.is_ascii_alphanumeric() || "-_.".contains(*c)).collect();
                if dir.join(format!("clip-{label}")).exists() {
                    reply(&mut s, "200 OK", &json!({"label": label, "state": "running"}));
                } else {
                    reply(&mut s, "404 Not Found", &json!({"error": "no queued clip with that label"}));
                }
            } else if head.contains("/rpc/status") {
                reply(&mut s, "200 OK", &json!({"result": {"idle": !dir.join("renders").exists()}}));
            } else {
                reply(&mut s, "404 Not Found", &json!({"error": "no"}));
            }
        }
    });
}

const SCRIPT: &str = r#"#!/bin/sh
D=__DIR__
echo "$* |idle=$NS_VIDEO_IDLE" >> $D/log
kind=$1; cmd=$2; shift 2
case "$kind" in
 video)
  case "$cmd" in
   start) eng=""; while [ $# -gt 0 ]; do case "$1" in --model|--engine) eng="$eng,$2"; shift;; esac; shift; done; echo "${eng#,}" > $D/engines;;
   stop) rm -f $D/engines;;
  esac;;
 image|audio|model3d)
  case "$cmd" in
   start) id=$1; port=""; while [ $# -gt 0 ]; do [ "$1" = --port ] && port=$2; shift; done; echo "$id" > $D/model-$port;;
   stop) slot=0; while [ $# -gt 0 ]; do [ "$1" = --slot ] && slot=$2; shift; done; port=$(sed -n "$((slot+1))p" $D/ports-$kind); rm -f $D/model-$port;;
  esac;;
esac
exit 0
"#;

/// The chat server: /v1/models names the model in the `llm-model` file (no file: nothing answers)
fn serve_llm(l: TcpListener, dir: PathBuf) {
    std::thread::spawn(move || {
        for mut s in l.incoming().flatten() {
            let _ = read_head(&mut s);
            let Ok(m) = std::fs::read_to_string(dir.join("llm-model")) else { continue };
            reply(&mut s, "200 OK", &json!({"data": [{"id": m.trim()}]}));
        }
    });
}

/// A box with one audio server slot (one 12 GiB card) and a video studio; `idle_minutes` 10
fn world(name: &str, models: Vec<Value>) -> World {
    let dir = std::env::temp_dir().join(format!("nextcuda-fx-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let exe = dir.join("fake.sh");
    std::fs::write(&exe, SCRIPT.replace("__DIR__", &dir.to_string_lossy())).unwrap();
    std::process::Command::new("chmod").arg("755").arg(&exe).status().unwrap();
    let (la, ls) = (TcpListener::bind("127.0.0.1:0").unwrap(), TcpListener::bind("127.0.0.1:0").unwrap());
    let (audio, video) = (la.local_addr().unwrap().port(), ls.local_addr().unwrap().port());
    std::fs::write(dir.join("ports-audio"), format!("{audio}\n")).unwrap();
    serve_kind(la, dir.clone(), audio);
    serve_studio(ls, dir.clone());
    let ll = TcpListener::bind("127.0.0.1:0").unwrap();
    let llm = ll.local_addr().unwrap().port();
    serve_llm(ll, dir.clone());
    let mem = Arc::new(Mem { models: Mutex::new(models), settings: Mutex::new(Default::default()), file: dir.join("models.json") });
    let ports = Ports { llm, switch: 1, image: 0, audio, model3d: 0, image2: 0, audio2: 0, model3d2: 0, video, chat_ui: 0 };
    let o = Options { wfe: PathBuf::new(), exe, ports, serve_host: "127.0.0.1".into(), cfg: Box::new(Shared(mem.clone())), idle_minutes: 10 };
    let home = Home::build(o).unwrap();
    *home.cards.lock().unwrap() = cards_of(&dir, audio, &mem);
    // the sampler: the card as the files say it is
    let stop = Arc::new(AtomicBool::new(false));
    let (h, d, m, st) = (home.clone(), dir.clone(), mem.clone(), stop.clone());
    std::thread::spawn(move || {
        while !st.load(Ordering::Relaxed) {
            *h.cards.lock().unwrap() = cards_of(&d, audio, &m);
            std::thread::sleep(Duration::from_millis(40));
        }
    });
    World { dir, home, mem, stop, audio }
}

/// One 12 GiB card with the audio server's model on it when its file is there
fn cards_of(dir: &Path, audio: u16, mem: &Mem) -> Value {
    let mut procs = Vec::new();
    let mut used = 300;
    if let Ok(id) = std::fs::read_to_string(dir.join(format!("model-{audio}"))) {
        let id = id.trim().to_string();
        let mb = mem.models.lock().unwrap().iter().find(|m| m["id"] == id.as_str()).and_then(|m| m["vram_gib"].as_f64()).unwrap_or(1.0) * 1024.0;
        used += mb as u64;
        procs.push(json!({"kind": "audio", "port": audio.to_string(), "model_id": id, "model": id, "vram_mb": mb as u64}));
    }
    json!([{"index": 0, "vram_total_mb": 12288, "vram_used_mb": used, "procs": procs}])
}

fn model(id: &str, kind: &str, gib: f64) -> Value {
    json!({"id": id, "title": id, "kind": kind, "arch": "x", "enabled": true, "gpus": "0", "vram_gib": gib, "file": ""})
}

impl World {
    fn log(&self) -> String {
        std::fs::read_to_string(self.dir.join("log")).unwrap_or_default()
    }
    fn count(&self, needle: &str) -> usize {
        self.log().lines().filter(|l| l.contains(needle)).count()
    }
    fn wait_log(&self, needle: &str) {
        let t0 = Instant::now();
        while self.count(needle) == 0 {
            assert!(t0.elapsed() < Duration::from_secs(10), "no `{needle}` in the log:\n{}", self.log());
            std::thread::sleep(Duration::from_millis(50));
        }
    }
    fn entry(&self, id: &str) -> Value {
        self.mem.models.lock().unwrap().iter().find(|m| m["id"] == id).cloned().unwrap()
    }
    /// A model loaded as if a request had loaded it
    fn load_now(&self, id: &str) {
        std::fs::write(self.dir.join(format!("model-{}", self.audio)), id).unwrap();
        std::thread::sleep(Duration::from_millis(150));
    }
}

fn big(id: &str) -> Value {
    // 7 GiB on a 12 GiB card: two do not fit together
    model(id, "audio", 7.0)
}

#[test]
fn a_request_spins_the_model_up_and_it_ends_when_idle() {
    let w = world("spin", vec![big("a")]);
    let lease = w.home.ensure("a").unwrap();
    drop(lease);
    assert_eq!(w.count("audio start a"), 1);
    // the idle timer is the server's own --idle-exit, in minutes the setting says
    assert!(w.log().contains("--idle-exit 600"), "{}", w.log());
    // loaded now: the next request does not start it again
    drop(w.home.ensure("a").unwrap());
    assert_eq!(w.count("audio start a"), 1);
}

#[test]
fn a_disabled_model_never_loads() {
    let mut d = big("d");
    d["enabled"] = json!(false);
    d["pinned"] = json!(true);
    let w = world("disabled", vec![d]);
    assert_eq!(w.home.ensure("d").err().map(|e| e.0), Some(409));
    assert_eq!(w.home.load_async("d").err().map(|e| e.0), Some(409));
    assert_eq!(w.home.pin("d").err().map(|e| e.0), Some(409));
    // and a pin left on it is not loaded at serve start
    w.home.load_pinned();
    std::thread::sleep(Duration::from_millis(300));
    assert_eq!(w.log(), "");
}

#[test]
fn a_pin_loads_now_keeps_the_server_without_an_idle_exit_and_survives_a_restart() {
    let w = world("pin", vec![big("a")]);
    w.home.pin("a").unwrap();
    w.wait_log("audio start a");
    assert!(!w.log().contains("--idle-exit"), "a pinned model's server does not end itself: {}", w.log());
    assert_eq!(w.entry("a")["pinned"], true);
    // a restart of serve: the same registry, the model gone from the card -> loaded again from the pin
    std::fs::remove_file(w.dir.join(format!("model-{}", w.audio))).unwrap();
    let again = Home::build(Options {
        wfe: PathBuf::new(), exe: w.dir.join("fake.sh"), serve_host: "127.0.0.1".into(), cfg: Box::new(Shared(w.mem.clone())), idle_minutes: 10,
        ports: Ports { llm: 1, switch: 1, image: 0, audio: w.audio, model3d: 0, image2: 0, audio2: 0, model3d2: 0, video: 1, chat_ui: 0 },
    }).unwrap();
    *again.cards.lock().unwrap() = cards_of(&w.dir, w.audio, &w.mem);
    again.load_pinned();
    let t0 = Instant::now();
    while w.count("audio start a") < 2 {
        assert!(t0.elapsed() < Duration::from_secs(10), "not loaded again:\n{}", w.log());
        std::thread::sleep(Duration::from_millis(50));
    }
}

#[test]
fn unpin_clears_the_flag_and_unloads_nothing_and_disable_unpins() {
    let w = world("unpin", vec![big("a"), big("b")]);
    w.home.pin("a").unwrap();
    w.wait_log("audio start a");
    w.home.unpin("a").unwrap();
    assert_eq!(w.entry("a")["pinned"], false);
    assert_eq!(w.count("audio stop"), 0, "unpin does not unload: {}", w.log());
    w.home.pin("b").unwrap();
    assert_eq!(w.entry("b")["pinned"], true);
    w.home.disable("b", false).unwrap();
    assert_eq!(w.entry("b")["pinned"], false);
    assert_eq!(w.entry("b")["enabled"], false);
}

#[test]
fn a_pinned_model_is_not_evicted_and_the_no_room_answer_names_the_pin() {
    let w = world("pinned-room", vec![big("a"), big("b")]);
    w.home.pin("a").unwrap();
    w.wait_log("audio start a");
    std::thread::sleep(Duration::from_millis(200));
    let b = w.entry("b");
    let why = match w.home.place(&b, "b", "audio", None).unwrap() {
        Ok(_) => panic!("b took the pinned model's room"),
        Err(why) => why,
    };
    assert!(why.contains("pinned: a"), "{why}");
    assert!(why.contains("needs ~"), "{why}");
    assert_eq!(w.count("audio stop"), 0, "{}", w.log());
    // as the caller sees it: 503, code no_room
    let e = super::api::rpc_error("audio.speak", 503, None, &why);
    assert_eq!(e["code"], "no_room");
}

#[test]
fn an_idle_unpinned_holder_is_evicted_for_room_then_the_other_loads() {
    let w = world("evict", vec![big("a"), big("b")]);
    drop(w.home.ensure("a").unwrap());
    std::thread::sleep(Duration::from_millis(200));
    let b = w.entry("b");
    let lease = w.home.place(&b, "b", "audio", None).unwrap().ok().expect("room made by stopping the idle one");
    drop(lease);
    assert_eq!(w.count("audio stop"), 1);
    assert_eq!(w.count("audio start b"), 1);
    // after an unpin the pinned one is evictable too
    let w = world("evict2", vec![big("a"), big("b")]);
    w.home.pin("a").unwrap();
    w.wait_log("audio start a");
    std::thread::sleep(Duration::from_millis(200));
    w.home.unpin("a").unwrap();
    let b = w.entry("b");
    assert!(w.home.place(&b, "b", "audio", None).unwrap().is_ok());
    assert_eq!(w.count("audio stop"), 1);
}

#[test]
fn a_video_request_with_the_daemon_down_applies_it_first() {
    let w = world("video-down", vec![model("minimax-h3", "video", 20.0), model("ltx2", "video", 20.0)]);
    // no engines file: the daemon is down
    w.home.video_ensure_within(Duration::from_secs(10)).unwrap();
    assert_eq!(w.count("video start --model minimax-h3 --engine ltx2 --gpu 0"), 1, "{}", w.log());
    // up now: a second request starts nothing
    w.home.video_ensure_within(Duration::from_secs(10)).unwrap();
    assert_eq!(w.count("video start"), 1);
}

#[test]
fn a_video_daemon_without_an_enabled_engine_is_restarted_unless_a_clip_renders() {
    let w = world("video-engine", vec![model("minimax-h3", "video", 20.0), model("ltx2", "video", 20.0)]);
    std::fs::write(w.dir.join("engines"), "minimax-h3").unwrap();
    std::fs::write(w.dir.join("renders"), "").unwrap();
    // a clip renders: the daemon it has answers, nothing is stopped
    w.home.video_ensure_within(Duration::from_secs(5)).unwrap();
    assert_eq!(w.count("video stop"), 0, "{}", w.log());
    std::fs::remove_file(w.dir.join("renders")).unwrap();
    w.home.video_ensure_within(Duration::from_secs(10)).unwrap();
    assert_eq!(w.count("video stop"), 1);
    assert_eq!(w.count("video start --model minimax-h3 --engine ltx2"), 1);
}

#[test]
fn no_video_model_enabled_starts_no_daemon() {
    let mut off = model("ltx2", "video", 20.0);
    off["enabled"] = json!(false);
    let w = world("video-off", vec![off]);
    w.home.video_ensure_within(Duration::from_secs(2)).unwrap();
    assert_eq!(w.count("video start"), 0);
}

/// Serve one request for `id` under an rsvp_s of `ask` (as `work` does): the lease is held while `f` runs
fn request<T>(w: &World, id: &str, ask: Option<f64>, f: impl FnOnce() -> T) -> T {
    let body = ask.map_or(json!({}), |a| json!({"rsvp_s": a}));
    let _s = super::rsvp::scope(&body);
    let lease = w.home.ensure(id).unwrap();
    let r = f();
    drop(lease);
    r
}

fn refused(w: &World, id: &str) -> Result<String, ()> {
    let m = w.entry(id);
    match w.home.place(&m, id, "audio", None).unwrap() {
        Ok(_) => Err(()),
        Err(why) => Ok(why),
    }
}

#[test]
fn a_hold_keeps_the_model_past_its_idle_time_and_blocks_room_eviction() {
    let w = world("hold", vec![big("a"), big("b")]);
    request(&w, "a", Some(600.0), || ());
    assert!(w.home.held_s("a") > 590.0, "{}", w.home.held_s("a"));
    assert!(w.home.kept_ids(&w.home.models()).contains("a"));
    std::thread::sleep(Duration::from_millis(200));
    let why = refused(&w, "b").expect("b cannot take a held model's room");
    assert!(why.contains("held by rsvp_s: a"), "{why}");
    assert_eq!(w.count("audio stop"), 0);
    // the keeper's tick touches the held server (its --idle-exit clock) and does not stop it
    w.home.keep_tick();
    assert_eq!(w.count("audio stop"), 0);
}

#[test]
fn a_hold_blocks_eviction_until_it_expires_then_allows_it() {
    let w = world("hold-expire", vec![big("a"), big("b")]);
    drop(w.home.ensure("a").unwrap());
    w.home.hold_after("a", 1.0);
    std::thread::sleep(Duration::from_millis(200));
    assert!(refused(&w, "b").is_ok());
    std::thread::sleep(Duration::from_millis(1100));
    assert!(w.home.kept_ids(&w.home.models()).is_empty());
    assert!(refused(&w, "b").is_err(), "the hold ran out: a is evicted for b");
    assert_eq!(w.count("audio stop"), 1);
}

#[test]
fn the_cap_clamps_a_larger_rsvp_and_an_absent_one_holds_nothing() {
    let w = world("hold-cap", vec![big("a")]);
    w.mem.settings.lock().unwrap().insert("NS_MAX_RSVP_S".into(), "60".into());
    request(&w, "a", Some(9000.0), || ());
    let h = w.home.held_s("a");
    assert!(h > 55.0 && h <= 60.0, "{h}");
    let w = world("hold-none", vec![big("a")]);
    request(&w, "a", None, || ());
    assert_eq!(w.home.held_s("a"), 0.0);
    assert!(w.home.kept_ids(&w.home.models()).is_empty(), "the normal idle rules");
}

#[test]
fn a_later_shorter_rsvp_does_not_shorten_a_hold() {
    let w = world("hold-longer", vec![big("a")]);
    request(&w, "a", Some(600.0), || ());
    request(&w, "a", Some(5.0), || ());
    assert!(w.home.held_s("a") > 590.0, "{}", w.home.held_s("a"));
}

#[test]
fn the_hold_starts_when_the_request_ends_not_when_it_came() {
    let w = world("hold-clock", vec![big("a")]);
    // a request that runs longer than its rsvp: still held for the full rsvp after it ends
    request(&w, "a", Some(1.5), || std::thread::sleep(Duration::from_millis(1800)));
    assert!(w.home.held_s("a") > 1.2, "{}", w.home.held_s("a"));
    // while it runs the server has work (never idle, never evictable) whatever the rsvp
    let w = world("hold-busy", vec![big("a"), big("b")]);
    request(&w, "a", Some(0.0), || {
        let (h, m) = (w.home.clone(), w.entry("b"));
        std::thread::spawn(move || {
            let _ = h.place(&m, "b", "audio", None);
        });
        std::thread::sleep(Duration::from_millis(500));
        assert_eq!(w.count("audio stop"), 0, "a model serving a request is not stopped for b: {}", w.log());
    });
    // a request that fails starts its hold at that moment
    let w = world("hold-fail", vec![big("a")]);
    let r: Result<(), String> = (|| {
        let _s = super::rsvp::scope(&json!({"rsvp_s": 300.0}));
        let _lease = w.home.ensure("a").map_err(|e| e.1)?;
        Err("the engine failed".to_string())
    })();
    assert!(r.is_err());
    assert!(w.home.held_s("a") > 290.0);
    // an in-flight request with a shorter rsvp does not shorten a finished one's hold
    request(&w, "a", Some(600.0), || ());
    request(&w, "a", Some(10.0), || std::thread::sleep(Duration::from_millis(100)));
    assert!(w.home.held_s("a") > 590.0);
}

#[test]
fn rsvp_zero_releases_the_model_as_soon_as_it_is_idle_unless_pinned_or_held() {
    let w = world("release", vec![big("a")]);
    request(&w, "a", Some(0.0), || ());
    assert_eq!(w.count("audio stop"), 0, "the tick is what releases it");
    w.home.keep_tick();
    assert_eq!(w.count("audio stop"), 1, "{}", w.log());
    // a pin overrides: nothing releases it
    let w = world("release-pin", vec![big("a")]);
    w.home.pin("a").unwrap();
    w.wait_log("audio start a");
    std::thread::sleep(Duration::from_millis(300));
    request(&w, "a", Some(0.0), || ());
    w.home.keep_tick();
    assert_eq!(w.count("audio stop"), 0);
    // an earlier, longer hold stands against a later 0
    let w = world("release-held", vec![big("a")]);
    request(&w, "a", Some(600.0), || ());
    request(&w, "a", Some(0.0), || ());
    w.home.keep_tick();
    assert_eq!(w.count("audio stop"), 0);
}

fn seed(w: &World, model: &str, wait_s: f64, run_s: f64) {
    let shape = super::history::shape_of("audio.speak", model, &json!({"model": model}));
    w.home.history.record(super::history::Record { method: "audio.speak".into(), model: model.into(), shape, wait_s, run_s, outcome: "ok".into() });
}

fn code_of(e: &(u16, String)) -> Value {
    super::api::rpc_error("audio.speak", e.0, None, &e.1)
}

#[test]
fn a_request_past_max_willing_to_wait_is_declined_with_no_side_effects() {
    let w = world("wait-decline", vec![big("a"), big("b")]);
    seed(&w, "b", 60.0, 5.0);
    let body = json!({"model": "b", "max_willing_to_wait_s": 10.0});
    let e = w.home.check_wait("audio.speak", &body, 10.0).err().expect("b would take ~60 s to start");
    let err = code_of(&e);
    assert_eq!(err["code"], "wait_too_long");
    assert_eq!(err["retryable"], true);
    assert_eq!(err["max_willing_to_wait_s"], 10.0);
    assert_eq!(err["eta"]["basis"], "shape");
    assert_eq!(err["eta"]["wait_s"], 60.0);
    // nothing started, stopped or queued for it
    assert_eq!(w.log(), "");
    // below the limit it is accepted
    assert!(w.home.check_wait("audio.speak", &body, 120.0).is_ok());
}

#[test]
fn a_hot_model_is_never_declined_and_absent_keeps_the_default_waits() {
    let w = world("wait-hot", vec![big("a")]);
    seed(&w, "a", 60.0, 5.0);
    drop(w.home.ensure("a").unwrap());
    std::thread::sleep(Duration::from_millis(150));
    assert!(w.home.check_wait("audio.speak", &json!({"model": "a"}), 0.0).is_ok(), "hot: wait_s 0");
    assert_eq!(super::rsvp::max_wait(), None);
}

#[test]
fn system_eta_answers_without_doing_any_work() {
    let w = world("eta-only", vec![big("a"), big("b")]);
    seed(&w, "b", 60.0, 5.0);
    let r = w.home.work("system.eta", &json!({"method": "audio.speak", "model": "b", "max_willing_to_wait_s": 10.0})).unwrap();
    assert_eq!(r["would_accept"], false);
    assert_eq!(r["eta"]["wait_s"], 60.0);
    assert_eq!(r["eta"]["run_s"], 5.0);
    assert_eq!(r["max_willing_to_wait_s"], 10.0);
    let r = w.home.work("system.eta", &json!({"method": "audio.speak", "model": "b", "max_willing_to_wait_s": 100.0})).unwrap();
    assert_eq!(r["would_accept"], true);
    // a model with no history: the wait is unknown, so it would be accepted (and enforced by the clock)
    let r = w.home.work("system.eta", &json!({"method": "audio.speak", "model": "a", "max_willing_to_wait_s": 1.0})).unwrap();
    assert_eq!((r["eta"]["basis"].as_str(), r["would_accept"].as_bool()), (Some("none"), Some(true)));
    assert_eq!(w.log(), "", "nothing was loaded");
    assert!(w.home.work("system.eta", &json!({"method": "system.models"})).is_err());
}

#[test]
fn an_unknown_eta_is_enforced_by_the_clock_and_nothing_starts_for_it() {
    let w = world("wait-clock", vec![big("a"), big("c")]);
    w.home.pin("a").unwrap();
    w.wait_log("audio start a");
    std::thread::sleep(Duration::from_millis(200));
    let t0 = Instant::now();
    let e = {
        let _s = super::rsvp::scope(&json!({"max_willing_to_wait_s": 1.0}));
        w.home.ensure("c").err().expect("no room beside the pin")
    };
    assert!(t0.elapsed() < Duration::from_secs(6));
    let err = code_of(&e);
    assert_eq!(err["code"], "wait_too_long");
    assert!(err["waited_s"].as_f64().unwrap() >= 1.0, "{err}");
    assert_eq!(w.count("audio start c"), 0);
    assert_eq!(w.count("audio stop"), 0);
    // without the limit the same request is the room answer, naming the pin
    let why = refused(&w, "c").unwrap();
    assert!(why.contains("pinned: a"), "{why}");
}

#[test]
fn a_queued_job_that_cannot_start_in_time_is_withdrawn() {
    let w = world("wait-job", vec![big("a")]);
    let job = |id: &str, max: f64, age: f64| json!({"id": id, "kind": "audio", "method": "audio.speak", "label": id, "state": "queued", "created": now() - age,
                                                    "order": now(), "body": {"model": "a", "text": "hi", "max_willing_to_wait_s": max}});
    w.home.jobs.list.lock().unwrap().push(job("late", 2.0, 30.0));
    w.home.jobs.list.lock().unwrap().push(job("fine", 600.0, 30.0));
    w.home.withdraw_waiting();
    let l = w.home.jobs.list.lock().unwrap();
    let late = l.iter().find(|j| j["id"] == "late").unwrap();
    assert_eq!(late["state"], "failed");
    assert_eq!(late["error"]["code"], "wait_too_long");
    assert_eq!(l.iter().find(|j| j["id"] == "fine").unwrap()["state"], "queued");
}

#[test]
fn a_finished_request_goes_into_the_history_and_feeds_the_next_eta() {
    let w = world("history", vec![big("a")]);
    w.home.record_work("audio.speak", &json!({"model": "a", "text": "hi"}), "a", 12.0, 3.0, "ok");
    w.home.record_work("audio.speak", &json!({"model": "a", "text": "hi"}), "a", 40.0, 9.0, "failed");
    let eta = w.home.eta_for("audio.speak", &json!({"model": "a", "text": "hi"}), None);
    assert_eq!((eta["basis"].as_str(), eta["run_s"].as_f64()), (Some("shape"), Some(3.0)), "{eta}");
    w.home.history.sync();
    assert!(w.dir.join("history").read_dir().unwrap().count() >= 2);
}

#[test]
fn rsvp_s_is_in_the_methods_that_run_a_model() {
    let has = |n: &str| super::methods::find(n).unwrap().fields.iter().any(|f| f.name == "rsvp_s" && f.optional);
    assert!(has("audio.speak") && has("image.generate") && has("llm.chat") && has("video.generate"));
    assert!(!has("system.models"));
}

fn chat(id: &str) -> Value {
    json!({"id": id, "title": id, "kind": "llm", "arch": "x", "enabled": true, "gpus": "0", "vram_gib": 6.0, "file": ""})
}

#[test]
fn a_pinned_or_held_chat_model_is_not_swapped_out_for_another() {
    let w = world("chat", vec![chat("a"), chat("b")]);
    std::fs::write(w.dir.join("llm-model"), "a").unwrap();
    assert!(w.home.chat_loaded(&w.home.models()).as_deref() == Some("a"));
    assert!(!w.home.kept_ids(&w.home.models()).contains("a"));
    // pinned: b is answered no_room naming the pin
    w.mem.models.lock().unwrap()[0]["pinned"] = json!(true);
    let e = w.home.chat_room("b").err().expect("a is pinned");
    assert!(e.1.contains("pinned: a"), "{}", e.1);
    assert_eq!(code_of(&e)["code"], "no_room");
    // the pinned model itself is fine
    assert!(w.home.chat_room("a").is_ok());
    // held (an rsvp_s) instead of pinned
    w.mem.models.lock().unwrap()[0]["pinned"] = json!(false);
    w.home.hold_after("a", 300.0);
    let e = w.home.chat_room("b").err().expect("a is held");
    assert!(e.1.contains("held by rsvp_s: a"), "{}", e.1);
    assert!(w.home.chat_idle().is_none(), "a held chat model is not an idle holder");
}

#[test]
fn a_pinned_or_held_video_model_asks_the_daemon_to_keep_its_workers() {
    let w = world("video-keep", vec![model("ltx2", "video", 20.0), model("minimax-h3", "video", 20.0)]);
    w.home.keep_tick();
    assert!(!w.dir.join("keeps").exists(), "nothing pinned or held: the daemon's own idle rule");
    w.mem.models.lock().unwrap()[0]["pinned"] = json!(true);
    w.home.keep_tick();
    assert_eq!(std::fs::read_to_string(w.dir.join("keeps")).unwrap().lines().count(), 1);
    // unpinned: it stops asking (the daemon is back to its normal idle with no restart)
    w.mem.models.lock().unwrap()[0]["pinned"] = json!(false);
    w.home.keep_tick();
    assert_eq!(std::fs::read_to_string(w.dir.join("keeps")).unwrap().lines().count(), 1);
    assert_eq!(w.count("video stop"), 0);
    // a hold does the same
    w.home.hold_after("minimax-h3", 120.0);
    w.home.keep_tick();
    assert_eq!(std::fs::read_to_string(w.dir.join("keeps")).unwrap().lines().count(), 2);
}

#[test]
fn a_clips_hold_starts_when_the_clip_ends() {
    let w = world("clip", vec![model("ltx2", "video", 20.0)]);
    std::fs::write(w.dir.join("clip-x"), "").unwrap();
    w.home.watch_clip("x".into(), "ltx2".into(), 300.0, 0.1);
    std::thread::sleep(Duration::from_millis(500));
    assert_eq!(w.home.held_s("ltx2"), 0.0, "the clip is still running: no hold yet");
    std::fs::remove_file(w.dir.join("clip-x")).unwrap();
    let t0 = Instant::now();
    while w.home.held_s("ltx2") == 0.0 {
        assert!(t0.elapsed() < Duration::from_secs(5), "no hold after the clip ended");
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(w.home.held_s("ltx2") > 290.0);
}

#[test]
fn a_jobs_rsvp_starts_its_hold_when_the_job_ends() {
    let w = world("job-hold", vec![big("a")]);
    let job = w.home.submit("audio.speak", &json!({"model": "a", "input": "hello", "wait": false, "rsvp_s": 300.0})).unwrap();
    let id = job["id"].as_str().unwrap().to_string();
    let t0 = Instant::now();
    loop {
        if w.home.jobs.get(&id).is_some_and(|j| matches!(j["state"].as_str(), Some("done" | "failed" | "cancelled"))) {
            break;
        }
        assert!(t0.elapsed() < Duration::from_secs(15), "the job did not end");
        std::thread::sleep(Duration::from_millis(50));
    }
    // (the fake server cannot speak: it ended, ok or failed - the hold starts at its end all the same)
    std::thread::sleep(Duration::from_millis(100));
    assert!(w.home.held_s("a") > 290.0, "{} {:?}", w.home.held_s("a"), w.home.jobs.get(&id));
    w.home.history.sync();
    assert_eq!(w.home.history.of("model|a").len(), 1);
}

#[test]
fn the_wait_answers_keep_their_outcome_in_the_body() {
    // the /rpc envelope is HTTP 200; error.status says 503 for both, the contract is code + retryable + the fields
    let nr = super::api::rpc_error("audio.speak", 503, None, "its card is held by work that is not idle - GPU 0; x needs ~8.0 GiB; pinned: a (unpin to free the room)");
    assert_eq!((nr["code"].as_str(), nr["retryable"].as_bool(), nr["status"].as_u64()), (Some("no_room"), Some(true), Some(503)));
    let wt = super::api::rpc_error("audio.speak", 503, None, &super::rsvp::wait_tag(json!({"eta": {"wait_s": 60.0}, "max_willing_to_wait_s": 10.0}), "later"));
    assert_eq!((wt["code"].as_str(), wt["retryable"].as_bool(), wt["status"].as_u64()), (Some("wait_too_long"), Some(true), Some(503)));
    assert_eq!(wt["eta"]["wait_s"], 60.0);
    assert_eq!(wt["max_willing_to_wait_s"], 10.0);
}

fn with(mut m: Value, k: &str, v: Value) -> Value {
    m[k] = v;
    m
}

fn small(id: &str) -> Value {
    model(id, "audio", 2.0)
}

fn exclusive() -> Value {
    json!({"mode": "exclusive", "exclusive_reason": "its working memory fills the card"})
}

#[test]
fn an_exclusive_model_gets_its_card_to_itself() {
    let w = world("excl", vec![small("a"), with(small("x"), "coexistence", exclusive())]);
    drop(w.home.ensure("a").unwrap());
    std::thread::sleep(Duration::from_millis(200));
    // a is idle: stopped for x although x would fit beside it by memory
    let x = w.entry("x");
    let lease = w.home.place(&x, "x", "audio", None).unwrap().ok().expect("x gets the card");
    drop(lease);
    assert_eq!(w.count("audio stop"), 1);
    assert_eq!(w.count("audio start x"), 1);
}

#[test]
fn nothing_else_is_started_on_an_exclusive_models_card_and_the_answer_names_it() {
    let w = world("excl2", vec![small("a"), with(small("x"), "coexistence", exclusive())]);
    w.home.pin("x").unwrap();
    w.wait_log("audio start x");
    std::thread::sleep(Duration::from_millis(200));
    let why = refused(&w, "a").expect("a cannot share x's card");
    assert!(why.contains("exclusive: x"), "{why}");
    assert!(why.contains("pinned: x"), "{why}");
    assert_eq!(w.count("audio stop"), 0);
    // unpinned and idle, x is just an idle holder: it goes for a
    w.home.unpin("x").unwrap();
    assert!(refused(&w, "a").is_err());
    assert_eq!(w.count("audio stop"), 1);
}

#[test]
fn shared_with_limits_who_may_be_a_resident_and_a_pin_makes_it_a_refusal() {
    // a runs beside images only; b is audio
    let a = with(small("a"), "coexistence", json!({"shared_with": ["kind:image"]}));
    let w = world("shared-with", vec![a, small("b")]);
    w.home.pin("a").unwrap();
    w.wait_log("audio start a");
    std::thread::sleep(Duration::from_millis(200));
    assert!(refused(&w, "b").is_ok(), "b is not on a's list and a is pinned: no room");
    assert_eq!(w.count("audio stop"), 0);
    // unpinned, a is an idle resident b may not stay beside: it is stopped for b
    w.home.unpin("a").unwrap();
    assert!(refused(&w, "b").is_err());
    assert_eq!(w.count("audio stop"), 1);
}

#[test]
fn the_profile_is_in_the_models_state_and_unmeasured_by_default() {
    let measured = with(small("m"), "concurrency", json!({"max_copies_per_card": 2, "recommended_copies": 2, "scaling": {"1": 1.0, "2": 1.6},
                                                         "measured": {"date": "2026-10-31", "card": "RTX 5070", "method": "bench"}}));
    let w = world("profile", vec![small("u"), measured]);
    let all = w.home.models();
    let rows = w.home.model_states(&all, &w.home.named_cards(&all));
    let row = |id: &str| rows.iter().find(|r| r["id"] == id).unwrap().clone();
    let u = row("u");
    assert_eq!((u["concurrency"]["status"].as_str(), u["recommended_copies"].as_u64(), u["max_copies_per_card"].as_u64(), u["copies_now"].as_u64()), (Some("unmeasured"), Some(1), Some(1), Some(0)));
    assert_eq!(u["coexistence"]["mode"], "shared");
    let m = row("m");
    assert_eq!((m["concurrency"]["status"].as_str(), m["recommended_copies"].as_u64(), m["max_copies_per_card"].as_u64()), (Some("measured"), Some(2), Some(2)));
    assert_eq!(m["concurrency"]["scaling"]["2"], 1.6);
    // the dispatcher's list: system.busy's engines
    let busy = w.home.busy(&all);
    let e = busy["engines"].as_array().unwrap();
    assert!(e.iter().any(|x| x["id"] == "m" && x["recommended_copies"] == 2 && x["status"] == "measured"));
    assert!(e.iter().any(|x| x["id"] == "u" && x["status"] == "unmeasured" && x["copies_now"] == 0));
    drop(w.home.ensure("u").unwrap());
    std::thread::sleep(Duration::from_millis(200));
    let rows = w.home.model_states(&all, &w.home.named_cards(&all));
    assert_eq!(rows.iter().find(|r| r["id"] == "u").unwrap()["copies_now"], 1);
}
