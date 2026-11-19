//! How long each loaded model has been loaded, and how long since its last request - per copy (a server, the chat
//! server, a video worker on a card), as system.state, system.models, system.busy and the page list them.
//!
//! - `loaded_at` (UTC, Unix seconds): when the copy finished loading - the front door's own start of it answering
//!   (`wait_up`), else the first time it was seen loaded (a copy started before this front door: the chat server's
//!   uptime when it says it).
//! - `last_request_at` (UTC, Unix seconds): the last moment a request was on the copy - now while one runs, else when the
//!   last one ended (an RPC, an OpenAI call, its page or a live conversation through the front door; the chat server's
//!   own request list for chat; the video daemon's worker idle clock for video). null: no request since it loaded.
//! - `loaded_s_ago` / `idle_s`: the same as seconds before the answer (`idle_s` 0 while a request runs; null with
//!   `last_request_at` null) - what a page shows without trusting its own clock.
//!
//! Kept in ages.json beside serve.json, with the boot it belongs to, so a restart of the front door keeps the ages of
//! the servers that outlive it; a reboot (every server gone) drops them.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Mutex;

use serde_json::{json, Value};

/// One loaded copy's clock
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Age {
    pub model: String,
    pub loaded_at: f64,
    pub last_request_at: Option<f64>,
    /// requests on it now (through the front door)
    pub active: usize,
}

impl Age {
    /// Its fields as an answer gives them, at `now`; `busy`: a request runs on it (whoever counted it)
    pub(crate) fn view(&self, now: f64, busy: bool) -> Value {
        let last = if busy || self.active > 0 { Some(now) } else { self.last_request_at };
        json!({"loaded_at": round(self.loaded_at), "last_request_at": last.map(round),
               "loaded_s_ago": secs(now - self.loaded_at), "idle_s": last.map(|l| secs(now - l))})
    }

    /// Seconds idle as the idle unload counts them: since the last request, else since it loaded
    pub(crate) fn idle_for(&self, now: f64) -> f64 {
        (now - self.last_request_at.unwrap_or(self.loaded_at).max(self.loaded_at)).max(0.0)
    }
}

fn round(t: f64) -> f64 {
    (t * 1000.0).round() / 1000.0
}

fn secs(d: f64) -> u64 {
    d.max(0.0).round() as u64
}

/// The fields a copy that is not known has (every one null)
pub(crate) fn unknown() -> Value {
    json!({"loaded_at": null, "last_request_at": null, "loaded_s_ago": null, "idle_s": null})
}

/// The key of a copy: an image/audio/3D server by its port, the chat server, a video worker by its card
pub(crate) fn port_key(port: u16) -> String {
    format!("port:{port}")
}
pub(crate) const CHAT: &str = "llm";
pub(crate) fn video_key(gpu: usize) -> String {
    format!("video:{gpu}")
}

/// The clocks of the loaded copies, by key
pub(crate) struct Ages {
    map: Mutex<BTreeMap<String, Age>>,
    file: Option<PathBuf>,
    boot: String,
}

fn boot_id() -> String {
    std::fs::read_to_string("/proc/sys/kernel/random/boot_id").map(|s| s.trim().to_string()).unwrap_or_default()
}

impl Ages {
    /// The clocks kept in `file` (when it is of this boot)
    pub(crate) fn open(file: Option<PathBuf>) -> Ages {
        let boot = boot_id();
        let map = file.as_ref().and_then(|f| std::fs::read(f).ok()).and_then(|b| serde_json::from_slice::<Value>(&b).ok())
            .filter(|v| v["boot"] == boot.as_str())
            .and_then(|v| v["copies"].as_object().cloned())
            .map(|o| o.into_iter().filter_map(|(k, v)| Some((k, Age {
                model: v["model"].as_str()?.to_string(), loaded_at: v["loaded_at"].as_f64()?, last_request_at: v["last_request_at"].as_f64(), active: 0,
            }))).collect())
            .unwrap_or_default();
        Ages { map: Mutex::new(map), file, boot }
    }

    fn save(&self, map: &BTreeMap<String, Age>) {
        let Some(f) = &self.file else { return };
        let copies: serde_json::Map<String, Value> = map.iter().map(|(k, a)| (k.clone(), json!({"model": a.model, "loaded_at": a.loaded_at, "last_request_at": a.last_request_at}))).collect();
        let tmp = f.with_extension("json.tmp");
        if std::fs::write(&tmp, json!({"boot": self.boot, "copies": copies}).to_string()).is_ok() {
            let _ = std::fs::rename(&tmp, f);
        }
    }

    /// `model` is loaded at `key` as seen at `now`: its clock, started now (or at `since`, when the copy says when it
    /// started) when none was kept for this model there
    pub(crate) fn seen(&self, key: &str, model: &str, now: f64, since: impl FnOnce() -> Option<f64>) -> Age {
        let mut m = self.map.lock().unwrap();
        if let Some(a) = m.get(key).filter(|a| a.model == model) {
            return a.clone();
        }
        let a = Age { model: model.into(), loaded_at: since().filter(|t| *t <= now).unwrap_or(now), last_request_at: None, active: 0 };
        m.insert(key.into(), a.clone());
        self.save(&m);
        a
    }

    /// The clock at `key`, when one is kept
    pub(crate) fn get(&self, key: &str) -> Option<Age> {
        self.map.lock().unwrap().get(key).cloned()
    }

    /// Nothing is loaded at `key` any more
    pub(crate) fn gone(&self, key: &str) {
        let mut m = self.map.lock().unwrap();
        if m.remove(key).is_some() {
            self.save(&m);
        }
    }

    /// A request on `key` starts (its copy of `model` is seen loaded, if it was not kept)
    pub(crate) fn begin(&self, key: &str, model: &str, now: f64) {
        let mut m = self.map.lock().unwrap();
        let a = m.entry(key.into()).or_insert_with(|| Age { model: model.into(), loaded_at: now, last_request_at: None, active: 0 });
        a.active += 1;
        a.last_request_at = Some(now);
        self.save(&m);
    }

    /// A request on `key` ended (answered, failed or cancelled)
    pub(crate) fn end(&self, key: &str, now: f64) {
        let mut m = self.map.lock().unwrap();
        if let Some(a) = m.get_mut(key) {
            a.active = a.active.saturating_sub(1);
            a.last_request_at = Some(now);
            self.save(&m);
        }
    }

    /// The copy at `key` says when its last request was (`at`; the chat server's list, the video worker's idle clock)
    pub(crate) fn last_was(&self, key: &str, at: Option<f64>) {
        let mut m = self.map.lock().unwrap();
        if let Some(a) = m.get_mut(key) {
            // never before it loaded (a request on the copy before is not on this one)
            let at = at.filter(|t| *t >= a.loaded_at - 1.0);
            if a.last_request_at != at && at.is_some_and(|t| a.last_request_at.is_none_or(|l| t > l)) {
                a.last_request_at = at;
                self.save(&m);
            }
        }
    }
}

/// A model's fields from its copies: loaded since its longest-loaded copy, its last request the latest of any copy;
/// `unload_in_s` the soonest a copy's idle unload comes (null for every copy: none comes)
pub(crate) fn model_view(instances: &[Value], unload_in: Option<u64>) -> Value {
    let f = |k: &'static str| instances.iter().filter_map(move |i| i[k].as_f64());
    let loaded_at = f("loaded_at").reduce(f64::min);
    let last = f("last_request_at").reduce(f64::max);
    let loaded_ago = f("loaded_s_ago").reduce(f64::max);
    let idle = f("idle_s").reduce(f64::min);
    json!({"loaded_at": loaded_at, "last_request_at": last, "loaded_s_ago": loaded_ago.map(|x| x as u64), "idle_s": idle.map(|x| x as u64),
           "unload_in_s": unload_in, "instances": instances})
}

/// Seconds until an idle copy ends itself (`idle_minutes` after its last request, else after it loaded); none when it
/// does not: idle unload off, the model pinned or held, a request on it
pub(crate) fn unload_in(age: &Age, now: f64, idle_minutes: u64, kept: bool, busy: bool) -> Option<u64> {
    if idle_minutes == 0 || kept || busy || age.active > 0 {
        return None;
    }
    Some(secs(idle_minutes as f64 * 60.0 - age.idle_for(now)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("ns-ages-{}-{name}", std::process::id()));
        let _ = std::fs::create_dir_all(&d);
        d.join("ages.json")
    }

    #[test]
    fn a_copy_loaded_then_used_says_both_ages() {
        let a = Ages::open(None);
        a.seen("port:1", "m", 1000.0, || None);
        let v = a.get("port:1").unwrap().view(1100.0, false);
        assert_eq!((v["loaded_s_ago"].as_u64(), v["idle_s"].clone(), v["last_request_at"].clone()), (Some(100), Value::Null, Value::Null), "no request yet: {v}");
        a.begin("port:1", "m", 1200.0);
        let v = a.get("port:1").unwrap().view(1260.0, false);
        assert_eq!((v["idle_s"].as_u64(), v["last_request_at"].as_f64()), (Some(0), Some(1260.0)), "while it runs: now, idle 0: {v}");
        a.end("port:1", 1300.0);
        let v = a.get("port:1").unwrap().view(1330.0, false);
        assert_eq!((v["loaded_at"].as_f64(), v["last_request_at"].as_f64(), v["loaded_s_ago"].as_u64(), v["idle_s"].as_u64()), (Some(1000.0), Some(1300.0), Some(330), Some(30)));
    }

    #[test]
    fn another_model_at_the_same_place_starts_a_new_clock() {
        let a = Ages::open(None);
        a.seen("port:1", "m", 1000.0, || None);
        a.end("port:1", 1050.0);
        assert_eq!(a.seen("port:1", "m", 2000.0, || None).loaded_at, 1000.0, "the same model: its clock kept");
        let b = a.seen("port:1", "n", 2000.0, || Some(1990.0));
        assert_eq!((b.loaded_at, b.last_request_at), (1990.0, None), "another model: a new clock, from when it says it started");
        assert_eq!(a.seen("port:2", "x", 3000.0, || Some(4000.0)).loaded_at, 3000.0, "a start in the future is not believed");
        a.gone("port:1");
        assert!(a.get("port:1").is_none());
    }

    #[test]
    fn the_clocks_outlive_a_restart_of_the_same_boot_only() {
        let f = tmp("restart");
        let _ = std::fs::remove_file(&f);
        let a = Ages::open(Some(f.clone()));
        a.seen("port:1", "m", 1000.0, || None);
        a.begin("port:1", "m", 1100.0);
        a.end("port:1", 1110.0);
        let b = Ages::open(Some(f.clone()));
        assert_eq!(b.get("port:1"), Some(Age { model: "m".into(), loaded_at: 1000.0, last_request_at: Some(1110.0), active: 0 }));
        // another boot's file: every server it knew is gone
        std::fs::write(&f, json!({"boot": "another", "copies": {"port:1": {"model": "m", "loaded_at": 1.0}}}).to_string()).unwrap();
        assert!(Ages::open(Some(f)).get("port:1").is_none());
    }

    #[test]
    fn a_copys_own_last_request_never_goes_back_nor_before_it_loaded() {
        let a = Ages::open(None);
        a.seen(CHAT, "q", 1000.0, || None);
        a.last_was(CHAT, Some(900.0));
        assert_eq!(a.get(CHAT).unwrap().last_request_at, None, "a request before it loaded is another copy's");
        a.last_was(CHAT, Some(1500.0));
        a.last_was(CHAT, Some(1400.0));
        assert_eq!(a.get(CHAT).unwrap().last_request_at, Some(1500.0));
    }

    #[test]
    fn a_model_is_as_old_as_its_oldest_copy_and_as_idle_as_its_latest_used() {
        let i = [json!({"loaded_at": 100.0, "last_request_at": 500.0, "loaded_s_ago": 900, "idle_s": 500}),
                 json!({"loaded_at": 300.0, "last_request_at": 800.0, "loaded_s_ago": 700, "idle_s": 200})];
        let v = model_view(&i, Some(60));
        assert_eq!((v["loaded_at"].as_f64(), v["last_request_at"].as_f64(), v["loaded_s_ago"].as_u64(), v["idle_s"].as_u64(), v["unload_in_s"].as_u64()),
                   (Some(100.0), Some(800.0), Some(900), Some(200), Some(60)));
        let v = model_view(&[], None);
        assert!(v["loaded_at"].is_null() && v["idle_s"].is_null() && v["instances"].as_array().unwrap().is_empty());
    }

    #[test]
    fn the_unload_countdown_runs_from_the_last_request_and_stops_while_kept() {
        let age = Age { model: "m".into(), loaded_at: 1000.0, last_request_at: Some(1500.0), active: 0 };
        assert_eq!(unload_in(&age, 1560.0, 10, false, false), Some(540));
        assert_eq!(unload_in(&Age { last_request_at: None, ..age.clone() }, 1060.0, 10, false, false), Some(540), "no request: from when it loaded");
        assert_eq!(unload_in(&age, 1560.0, 10, true, false), None, "pinned or held");
        assert_eq!(unload_in(&age, 1560.0, 10, false, true), None, "a request on it");
        assert_eq!(unload_in(&age, 1560.0, 0, false, false), None, "idle unload off");
        assert_eq!(unload_in(&age, 9999.0, 10, false, false), Some(0));
    }
}
