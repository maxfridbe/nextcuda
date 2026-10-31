//! `rsvp_s`: how long, after this request, the model that serves it stays loaded - a keep-alive (Ollama's keep_alive).
//!
//! - the caller sends `rsvp_s` (seconds, optional) on any method that runs a model; the server caps it with
//!   `NS_MAX_RSVP_S` (unset or 0: DEFAULT_MAX_RSVP_S, 24 h). Larger is not an error: it is clamped, and the answer
//!   says what was granted (`rsvp_granted_s`, added to an object result);
//! - when the request ends the model is *held* until now + the granted seconds (never shortened by a later, shorter
//!   rsvp: the hold is the latest of them). A held model is as good as in use: the idle timer does not take it (the
//!   backplane keeps the server's idle clock fresh while it is held) and room planning does not evict it. A pin is
//!   a hold with no end. When the hold ends the normal idle rules begin;
//! - no `rsvp_s`: no hold, the model's normal idle time. `rsvp_s: 0`: release - the model goes as soon as it is idle
//!   (nothing pinned or held keeps it) instead of waiting out the idle time.
//!
//! The field is not in every request struct: `with_field` adds it to the method table's description of the methods
//! that run a model (the document and the wfe client are made from that table), and the request's `rsvp_s` is read
//! from its body by the front door (a thread-local for the length of the call) where the model is served.

use std::cell::Cell;

use rpcgen::{Field, MethodInfo, Named, Shape};
use serde_json::{json, Value};

/// The longest keep-alive when NS_MAX_RSVP_S says none: a day
pub const DEFAULT_MAX_RSVP_S: f64 = 86400.0;

thread_local! {
    static ASKED: Cell<Option<f64>> = const { Cell::new(None) };
    static MAX_WAIT: Cell<Option<f64>> = const { Cell::new(None) };
    static STARTED: std::cell::RefCell<Option<(String, f64)>> = const { std::cell::RefCell::new(None) };
}

/// The caller's `rsvp_s` and `max_willing_to_wait_s` while a call runs on this thread (restored when dropped)
pub struct Scope(Option<f64>, Option<f64>);

pub fn scope(body: &Value) -> Scope {
    let was = ASKED.with(|a| a.replace(body["rsvp_s"].as_f64().filter(|x| *x >= 0.0)));
    let wait = MAX_WAIT.with(|a| a.replace(body["max_willing_to_wait_s"].as_f64().filter(|x| *x >= 0.0)));
    Scope(was, wait)
}

impl Drop for Scope {
    fn drop(&mut self) {
        ASKED.with(|a| a.set(self.0));
        MAX_WAIT.with(|a| a.set(self.1));
    }
}

/// The longest the caller will wait for its request to start (`max_willing_to_wait_s`), for the call on this thread
pub fn max_wait() -> Option<f64> {
    MAX_WAIT.with(Cell::get)
}

/// The model that took the call on this thread and the seconds it waited for it (room, spin-up): noted when the
/// request starts, read when it ends (the history's wait_s)
pub fn note_start(model: &str, waited_s: f64) {
    STARTED.with(|s| *s.borrow_mut() = Some((model.to_string(), waited_s)));
}

pub fn take_start() -> Option<(String, f64)> {
    STARTED.with(|s| s.borrow_mut().take())
}

const WAIT_TAG: &str = "wait_too_long:";

/// A `wait_too_long` refusal as a Fail's message: the error's fields as JSON (eta, max_willing_to_wait_s, waited_s)
pub fn wait_tag(fields: Value, words: &str) -> String {
    format!("{WAIT_TAG}{}\n{words}", fields)
}

pub fn wait_untag(m: &str) -> Option<(Value, String)> {
    let rest = m.strip_prefix(WAIT_TAG)?;
    let (f, words) = rest.split_once('\n')?;
    Some((serde_json::from_str(f).ok()?, words.to_string()))
}

/// The caller's `rsvp_s` for the call running on this thread
pub fn asked() -> Option<f64> {
    ASKED.with(Cell::get)
}

/// The server's longest keep-alive from its setting: unset, 0 or not a number is the default
pub fn cap_of(raw: Option<String>) -> f64 {
    raw.and_then(|r| r.trim().parse::<f64>().ok()).filter(|c| *c > 0.0).unwrap_or(DEFAULT_MAX_RSVP_S)
}

/// The keep-alive granted: the caller's against the cap, 0 or more; none when the caller asked for none
pub fn granted(asked: Option<f64>, cap: f64) -> Option<f64> {
    asked.map(|a| a.clamp(0.0, cap))
}

/// A model's hold: kept until `until` (unix seconds); `release`: the last rsvp was 0 (go when idle)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hold {
    pub until: f64,
    pub release: bool,
}

/// The hold after a request that ended at `now` with `granted` seconds: the latest of the old and the new, never
/// shortened; 0 releases (unless something still holds it)
pub fn extend(cur: Option<Hold>, now: f64, granted: f64) -> Hold {
    let old = cur.unwrap_or(Hold { until: 0.0, release: false });
    if granted > 0.0 {
        Hold { until: old.until.max(now + granted), release: false }
    } else {
        Hold { until: old.until, release: old.until <= now }
    }
}

/// Seconds a hold has left at `now` (0: none)
pub fn left(h: Option<&Hold>, now: f64) -> f64 {
    h.map_or(0.0, |h| (h.until - now).max(0.0))
}

/// Whether a method runs a model (and so takes `rsvp_s`): the ones that start work, the engines' listing
pub fn takes(method: &str) -> bool {
    !method.starts_with("system.") && (super::maint::starts_work(method) || method == "video.engines.list")
}

/// The method's description with the optional `rsvp_s` field in its request
pub fn with_field(mut info: MethodInfo) -> MethodInfo {
    if !takes(info.name) {
        return info;
    }
    if let Shape::Ref(n) = &info.request {
        if let Some(Named::Object { object, .. }) = info.defs.types.get_mut(n) {
            if !object.fields.iter().any(|f| f.name == "max_willing_to_wait_s") {
                object.fields.push(Field {
                    name: "max_willing_to_wait_s".into(),
                    doc: "the longest, in seconds, you will wait for this request to START (queue, making room, loading the model; its run time is not counted). If the ETA's wait_s (system.eta) is known and longer, the answer is at once error.code wait_too_long (eta, max_willing_to_wait_s) and nothing is queued, loaded or evicted - send it to another machine. If the ETA is unknown the request goes on but is withdrawn, with the same code (waited_s), when it has not started in time. Absent: the server's own waits".into(),
                    shape: Shape::Number,
                    optional: true,
                    extra: json!({"minimum": 0}),
                    one_of: Vec::new(),
                });
            }
            if !object.fields.iter().any(|f| f.name == "rsvp_s") {
                object.fields.push(Field {
                    name: "rsvp_s".into(),
                    doc: "keep-alive: how many seconds after this request the model that serves it stays loaded (not unloaded when idle, not stopped for another model's room). Clamped to the server's NS_MAX_RSVP_S (default 24 h) - the answer's rsvp_granted_s says what was granted; a later, shorter value never shortens a hold. 0: release - the model goes as soon as it is idle. Absent: the model's normal idle time".into(),
                    shape: Shape::Number,
                    optional: true,
                    extra: json!({"minimum": 0}),
                    one_of: Vec::new(),
                });
            }
        }
    }
    info
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_server_cap_clamps_a_larger_ask() {
        assert_eq!(granted(Some(900.0), 600.0), Some(600.0));
        assert_eq!(granted(Some(30.0), 600.0), Some(30.0));
        assert_eq!(granted(Some(-5.0), 600.0), Some(0.0));
        // absent: nothing asked, the normal idle time
        assert_eq!(granted(None, 600.0), None);
    }

    #[test]
    fn the_cap_setting_defaults_to_a_day_when_unset_zero_or_junk() {
        assert_eq!(cap_of(None), DEFAULT_MAX_RSVP_S);
        assert_eq!(cap_of(Some("0".into())), DEFAULT_MAX_RSVP_S);
        assert_eq!(cap_of(Some("soon".into())), DEFAULT_MAX_RSVP_S);
        assert_eq!(cap_of(Some(" 120 ".into())), 120.0);
    }

    #[test]
    fn a_hold_extends_and_a_shorter_later_one_does_not_shorten_it() {
        let h = extend(None, 1000.0, 600.0);
        assert_eq!(h, Hold { until: 1600.0, release: false });
        // a later request asks for less: the longer hold stays
        let h2 = extend(Some(h), 1100.0, 60.0);
        assert_eq!(h2.until, 1600.0);
        // a later one asks for more
        assert_eq!(extend(Some(h2), 1200.0, 1000.0).until, 2200.0);
        assert_eq!(left(Some(&h2), 1100.0), 500.0);
        assert_eq!(left(Some(&h2), 1700.0), 0.0);
        assert_eq!(left(None, 1.0), 0.0);
    }

    #[test]
    fn zero_releases_unless_something_still_holds() {
        assert!(extend(None, 1000.0, 0.0).release);
        let held = extend(None, 1000.0, 600.0);
        assert!(!extend(Some(held), 1100.0, 0.0).release, "the earlier hold stands");
        // once it has run out, 0 releases
        assert!(extend(Some(held), 1700.0, 0.0).release);
        // a positive ask after a release holds again
        assert!(!extend(Some(Hold { until: 0.0, release: true }), 2000.0, 30.0).release);
    }

    #[test]
    fn the_scope_reads_the_callers_ask_and_restores() {
        assert_eq!(asked(), None);
        {
            let _s = scope(&json!({"rsvp_s": 12.5}));
            assert_eq!(asked(), Some(12.5));
            {
                let _t = scope(&json!({}));
                assert_eq!(asked(), None);
            }
            assert_eq!(asked(), Some(12.5));
        }
        assert_eq!(asked(), None);
    }

    #[test]
    fn the_methods_that_run_a_model_take_it() {
        assert!(takes("image.generate") && takes("llm.chat") && takes("video.generate") && takes("video.engines.list"));
        assert!(!takes("system.load") && !takes("image.status") && !takes("system.models"));
    }
}
