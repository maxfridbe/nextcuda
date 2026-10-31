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

use rpcgen::{Field, MethodInfo, Named, Shape};
use serde_json::json;

pub use backplane_scheduler::rsvp::*;

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
    fn the_methods_that_run_a_model_take_it() {
        assert!(takes("image.generate") && takes("llm.chat") && takes("video.generate") && takes("video.engines.list"));
        assert!(!takes("system.load") && !takes("image.status") && !takes("system.models"));
    }
}
