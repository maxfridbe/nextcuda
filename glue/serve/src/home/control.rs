//! video.control's controls that need a model: `depth`, an image-queue job like video.upscale - a depth estimator
//! (Depth Anything V2: capability "depth") reads every frame on its image server (POST /v1/videos/depth), the result
//! beside the clips as the studio's canny / gray are (video/studio/control.rs makes all of them alike). Canny and gray
//! stay the studio's, on the CPU.

use std::path::PathBuf;

use serde_json::{json, Value};

use super::api::relay;
use super::life::s;
use super::{now, registry, Home};
use crate::video::studio::control;

type Fail = (u16, String);

/// Whether a video.control request is for a control a model makes (depth): it then runs as an image job, not the
/// studio's call
pub fn by_model(name: &str, b: &Value) -> bool {
    name == "video.control" && control::MODEL_KINDS.contains(&b["control"].as_str().unwrap_or("").trim().to_lowercase().as_str())
}

/// A registry entry that estimates depth (video.control's depth)
pub(super) fn estimates_depth(m: &Value) -> bool {
    s(m, "arch") == "depth-anything-v2"
}

impl Home {
    /// The clips directory (the studio's: NS_VIDEO_OUT)
    pub(super) fn clips_dir(&self) -> PathBuf {
        let dir = self.o.cfg.get("NS_VIDEO_OUT").filter(|d| !d.is_empty())
            .unwrap_or_else(|| format!("{}/.local/share/nextcuda/video", std::env::var("HOME").unwrap_or_else(|_| "/tmp".into())));
        PathBuf::from(dir)
    }

    /// The depth model for a request: the one named (it must estimate depth), else the loaded or first enabled one
    pub(super) fn depth_model(&self, b: &Value, all: &[Value]) -> Result<String, Fail> {
        if b["model"].as_str().is_some_and(|x| !x.is_empty()) {
            let id = self.pick("image", b, all)?;
            if all.iter().any(|m| m["id"] == id.as_str() && !estimates_depth(m)) {
                return Err((400, format!("{id} does not estimate depth (control depth needs a depth model: depth-anything-v2-small)")));
            }
            return Ok(id);
        }
        let on: Vec<&Value> = all.iter().filter(|m| registry::kind_of(m) == "image" && m["enabled"] != false && estimates_depth(m)).collect();
        self.loaded_of("image").into_iter().find(|l| on.iter().any(|m| m["id"] == l.as_str())).or_else(|| on.first().map(|m| s(m, "id")))
            .ok_or_else(|| (409, "no depth model is enabled (system.enable depth-anything-v2-small)".to_string()))
    }

    /// The request read against its source: the source, its seconds and the settings - what a submit checks first
    fn control_request(&self, b: &Value) -> Result<(PathBuf, String, f64, control::Settings), Fail> {
        let (src, n) = control::source(&self.clips_dir(), b).map_err(|e| (e.http, e.message))?;
        let (w, h, seconds) = control::probe(&src).ok_or_else(|| (400, format!("{n}: not a clip ffprobe reads (no video stream)")))?;
        let st = control::settings(b, (w, h, seconds)).map_err(|e| (e.http, e.message))?;
        Ok((src, n, seconds, st))
    }

    /// What a depth control's submit would be refused for, refused now (wait false would answer, then fail)
    pub(super) fn precheck_control(&self, b: &Value, all: &[Value]) -> Result<(), Fail> {
        self.control_request(b)?;
        self.depth_model(b, all).map(|_| ())
    }

    /// video.control with a control a model makes (a job of the image queue: its request_id is the job's id): the depth
    /// model's server makes it beside the clips; the answer is the studio's for canny, with the model and the range
    pub(super) fn control_by_model(&self, b: &Value, all: &[Value]) -> Result<Value, Fail> {
        let (src, n, seconds, st) = self.control_request(b)?;
        let id = self.depth_model(b, all)?;
        let name = control::output_name(&n, &st.control, now() as u64);
        let job = b["request_id"].as_str();
        let body = json!({"model": id, "request_id": job, "path": src.to_string_lossy(), "name": name, "width": st.width, "height": st.height,
                          "frames": st.frames, "start_s": st.start_s});
        let t0 = std::time::Instant::now();
        let r = self.relay_with_room(&id, &|| {
            let t = self.ensure_for(&id, job)?;
            self.sent(job);
            relay(&t, "POST", "/v1/videos/depth", Some(&body), 86400)
        })?;
        let made = r["frames"].as_u64().unwrap_or(0) as u32;
        let mut a = control::answer(&name, &st, made, seconds, t0.elapsed().as_secs_f64());
        a["file"] = json!(format!("http://nextcuda/v1/video/files/{name}"));
        a["model"] = json!(id);
        a["range"] = r["range"].clone();
        a["frame_seconds"] = r["frame_seconds"].clone();
        Ok(a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depth_is_a_models_job_and_canny_the_studios() {
        assert!(by_model("video.control", &json!({"control": "depth"})));
        assert!(by_model("video.control", &json!({"control": " Depth "})));
        assert!(!by_model("video.control", &json!({"control": "canny"})));
        assert!(!by_model("video.control", &json!({})), "canny by default");
        assert!(!by_model("video.upscale", &json!({"control": "depth"})));
        assert!(estimates_depth(&json!({"arch": "depth-anything-v2"})));
        assert!(!estimates_depth(&json!({"arch": "birefnet"})));
    }
}
