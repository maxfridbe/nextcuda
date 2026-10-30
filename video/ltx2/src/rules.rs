//! The request rules, apart from any GPU: what a `generate` or `upscale` job may ask for, and the numbers derived
//! from it (the frame count the model makes, the latent grid, the audio length, the upscaler's windows). The
//! engine's code reads them from here, so a rule that drifts shows in a test, not in a clip.
//!
//! - frames: the model works on 8k + 1 (a latent frame is 8 pixel frames, the first alone); any count from nine is
//!   asked for, made as the next 8k + 1 and cut back to what was asked
//! - canvas: a multiple of 32 each way (the VAE's spatial factor)
//! - audio: 25 latent frames a second of the generated length (frames / fps)

use std::path::PathBuf;

use nextcuda_core::{Error, Result};
use serde_json::Value;

use crate::sample;

/// The frame grid: counts the model makes are 8k + 1
pub const FRAME_STEP: usize = 8;
/// The least the model makes (one latent frame past the first)
pub const MIN_FRAMES: usize = FRAME_STEP + 1;
/// The VAE's spatial factor: canvases are multiples of this
pub const CANVAS_MULTIPLE: usize = 32;
/// The smallest side the engine takes
pub const MIN_SIDE: usize = 64;
pub const DEFAULT_FPS: f32 = 24.0;
pub const DEFAULT_WIDTH: usize = 768;
pub const DEFAULT_HEIGHT: usize = 512;
pub const DEFAULT_SECONDS: f64 = 5.0;
/// Latent audio frames a second
pub const AUDIO_RATE: f64 = 25.0;
/// The upscaler's frames a pass (each an 8k + 1)
pub const WINDOW: usize = 121;
/// What the job asks for that this engine does not make: refused with the key named
pub const NOT_TAKEN: [&str; 6] = ["guide_clip", "first_audio", "last_frame", "ref_audio", "source", "first_latent"];

/// The next 8k + 1 at or above n (nine at least)
pub fn align_frames(n: usize) -> usize {
    let n = n.max(MIN_FRAMES);
    n + (FRAME_STEP - (n - 1) % FRAME_STEP) % FRAME_STEP
}

/// Latent frames of a pixel-frame count made as 8k + 1: (frames - 1) / 8 + 1
pub fn latent_frames(frames: usize) -> usize {
    (align_frames(frames) - 1) / FRAME_STEP + 1
}

/// Audio latent frames for `frames` made at `fps`
pub fn audio_latents(frames: usize, fps: f32) -> usize {
    (frames as f64 / fps as f64 * AUDIO_RATE).round() as usize
}

/// The DiT tokens of a canvas and length: (w / 32) (h / 32) latent frames
pub fn tokens(width: usize, height: usize, frames: usize) -> usize {
    (width / CANVAS_MULTIPLE) * (height / CANVAS_MULTIPLE) * latent_frames(frames)
}

/// A `generate` job, checked
#[derive(Debug, Clone, PartialEq)]
pub struct Generate {
    pub prompt: String,
    pub negative: String,
    pub width: usize,
    pub height: usize,
    pub fps: f32,
    /// the frames asked for (the clip is cut to this), and the 8k + 1 the model makes
    pub want: usize,
    pub frames: usize,
    pub fast: bool,
    /// four guided passes a step (the dev DiT's way); false: the prompt's pass alone (the distilled DiT's: it is trained
    /// without guidance, and guided it over-saturates the picture and clips the sound)
    pub guided: bool,
    /// the steps asked for; None with `fast`: the distilled DiT's own eight sigmas
    pub steps: Option<usize>,
    pub seed: u64,
    pub out: PathBuf,
    pub first_frame: Option<String>,
}

impl Generate {
    /// The sigmas this job runs (N steps and a trailing zero)
    pub fn sigmas(&self) -> Vec<f32> {
        match (self.fast, self.steps) {
            (true, None) => sample::sigmas(0, true),
            (_, s) => sample::sigmas(s.unwrap_or(sample::DEV_STEPS), false),
        }
    }
}

fn number(spec: &Value, k: &str) -> Option<f64> {
    spec[k].as_f64().or_else(|| spec[k].as_str().and_then(|s| s.parse().ok()))
}

/// The generate job's keys: {prompt, negative, width, height, frames | seconds, fps, steps, seed, out, first_frame, fast,
/// guided (default: not with `fast`)};
/// `fast_default`: the registry entry's (ltx-2.5-fast) when the job does not say
pub fn parse_generate(spec: &Value, fast_default: bool) -> Result<Generate> {
    let prompt = spec["prompt"].as_str().unwrap_or("").trim().to_string();
    if prompt.is_empty() {
        return Err(Error("generate: a prompt".into()));
    }
    for k in NOT_TAKEN {
        if !spec[k].is_null() {
            return Err(Error(format!("{k}: LTX-2.5 here makes a clip from a prompt and (optionally) a first frame only")));
        }
    }
    let width = number(spec, "width").unwrap_or(DEFAULT_WIDTH as f64) as usize;
    let height = number(spec, "height").unwrap_or(DEFAULT_HEIGHT as f64) as usize;
    if width % CANVAS_MULTIPLE != 0 || height % CANVAS_MULTIPLE != 0 || width < MIN_SIDE || height < MIN_SIDE {
        return Err(Error(format!("{width} x {height}: the canvas is a multiple of {CANVAS_MULTIPLE} each way")));
    }
    let fps = number(spec, "fps").unwrap_or(DEFAULT_FPS as f64) as f32;
    let want = match number(spec, "frames") {
        Some(f) => f as usize,
        // as H3 and the studio count them: round(seconds x fps)
        None => (number(spec, "seconds").unwrap_or(DEFAULT_SECONDS) * fps as f64).round() as usize,
    }
    .max(MIN_FRAMES);
    // an explicit `fast` (the studio sends it either way) decides; else the registry entry's default
    let fast = spec["fast"].as_bool().or_else(|| number(spec, "fast").map(|v| v != 0.0)).unwrap_or(fast_default);
    Ok(Generate {
        prompt,
        negative: spec["negative"].as_str().unwrap_or("").to_string(),
        width,
        height,
        fps,
        want,
        frames: align_frames(want),
        fast,
        guided: spec["guided"].as_bool().unwrap_or(!fast),
        steps: number(spec, "steps").map(|s| s as usize).filter(|s| *s > 0),
        seed: number(spec, "seed").map_or(0, |s| s as i64) as u64,
        out: PathBuf::from(spec["out"].as_str().ok_or("generate: out (an .mp4 path)")?),
        first_frame: spec["first_frame"].as_str().filter(|s| !s.is_empty()).map(str::to_string),
    })
}

/// An `upscale` job, checked against the source's size
#[derive(Debug, Clone, PartialEq)]
pub struct Upscale {
    pub source: PathBuf,
    pub out: PathBuf,
}

pub fn parse_upscale(spec: &Value) -> Result<Upscale> {
    Ok(Upscale {
        source: PathBuf::from(spec["source"].as_str().ok_or("upscale: source (an .mp4)")?),
        out: PathBuf::from(spec["out"].as_str().ok_or("upscale: out (an .mp4)")?),
    })
}

/// What the VAE and the upsampler take of a source: nine frames or more, sides multiples of 32
pub fn check_source(name: &str, total: usize, w: usize, h: usize) -> Result<()> {
    if total < MIN_FRAMES {
        return Err(Error(format!("{name}: {total} frames; the VAE needs at least nine")));
    }
    if w % CANVAS_MULTIPLE != 0 || h % CANVAS_MULTIPLE != 0 {
        return Err(Error(format!("{name}: {w} x {h}; the VAE takes multiples of {CANVAS_MULTIPLE}")));
    }
    Ok(())
}

/// The upscaler's passes over `total` frames: (start, frames) each an 8k + 1 of at most `window`; a tail shorter
/// than nine frames is dropped (so the clip may come back a few frames shorter: at most seven past the last window)
pub fn windows(total: usize, window: usize) -> Vec<(usize, usize)> {
    let mut v = Vec::new();
    let mut start = 0;
    while start < total {
        let n = (total - start).min(window);
        let legal = n - (n - 1) % FRAME_STEP;
        if legal < MIN_FRAMES {
            break;
        }
        v.push((start, legal));
        start += n;
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn frames_are_made_as_the_next_8k_plus_1() {
        for (asked, made) in [(1, 9), (8, 9), (9, 9), (10, 17), (16, 17), (17, 17), (120, 121), (121, 121), (122, 129), (240, 241), (361, 361)] {
            assert_eq!(align_frames(asked), made, "{asked}");
            assert_eq!((align_frames(asked) - 1) % 8, 0);
        }
        // latent frames: the first pixel frame alone, then eight each
        assert_eq!((latent_frames(9), latent_frames(121), latent_frames(120), latent_frames(122)), (2, 16, 16, 17));
        assert_eq!(tokens(640, 384, 121), 3840, "the tested shape");
        assert_eq!(tokens(640, 384, 120), 3840, "120 frames are made as 121");
    }

    #[test]
    fn the_audio_follows_the_frames_made() {
        // 25 latent frames a second of the 8k + 1 made (121 frames at 24 fps = 5.04 s = 126)
        assert_eq!(audio_latents(121, 24.0), 126);
        assert_eq!(audio_latents(9, 24.0), 9);
        assert_eq!(audio_latents(241, 24.0), 251);
        assert_eq!(audio_latents(121, 25.0), 121);
    }

    #[test]
    fn a_generate_job_is_read_with_the_defaults() {
        let g = parse_generate(&json!({"prompt": " A fox. ", "out": "/o/x.mp4"}), false).unwrap();
        assert_eq!((g.prompt.as_str(), g.width, g.height, g.fps), ("A fox.", 768, 512, 24.0));
        assert_eq!((g.want, g.frames, g.fast, g.steps, g.seed), (120, 121, false, None, 0), "5 s = 120 frames, made as 121");
        assert_eq!(g.sigmas().len(), 31, "dev: 30 steps and the trailing zero");
        // frames win over seconds; numbers may arrive as strings (the command line's)
        let g = parse_generate(&json!({"prompt": "x", "out": "o", "frames": "97", "seconds": 9, "width": "640", "height": 384, "steps": "12", "seed": "-1"}), false).unwrap();
        assert_eq!((g.want, g.frames, g.width, g.steps, g.seed), (97, 97, 640, Some(12), u64::MAX));
        assert_eq!(g.sigmas().len(), 13);
        // fewer than nine frames are nine
        assert_eq!(parse_generate(&json!({"prompt": "x", "out": "o", "frames": 3}), false).unwrap().want, 9);
        assert_eq!(parse_generate(&json!({"prompt": "x", "out": "o", "seconds": 0.1}), false).unwrap().frames, 9);
        // a first frame and a negative prompt ride along; an empty first frame is none
        let g = parse_generate(&json!({"prompt": "x", "out": "o", "first_frame": "/o/a.png", "negative": "blur"}), false).unwrap();
        assert_eq!((g.first_frame.as_deref(), g.negative.as_str()), (Some("/o/a.png"), "blur"));
        assert_eq!(parse_generate(&json!({"prompt": "x", "out": "o", "first_frame": ""}), false).unwrap().first_frame, None);
    }

    #[test]
    fn fast_is_the_requests_else_the_entrys() {
        let j = |v: Value| parse_generate(&v, true).unwrap();
        assert!(j(json!({"prompt": "x", "out": "o"})).fast, "the ltx-2.5-fast card's default");
        assert!(!j(json!({"prompt": "x", "out": "o", "fast": false})).fast, "an explicit false wins (the studio sends it)");
        // guidance follows the variant: the dev DiT is guided, the distilled one is not, unless the job says
        assert!(!j(json!({"prompt": "x", "out": "o"})).guided && j(json!({"prompt": "x", "out": "o", "fast": false})).guided);
        assert!(j(json!({"prompt": "x", "out": "o", "guided": true})).guided);
        assert!(!parse_generate(&json!({"prompt": "x", "out": "o", "guided": false}), false).unwrap().guided);
        assert!(j(json!({"prompt": "x", "out": "o", "fast": 1})).fast);
        assert!(!parse_generate(&json!({"prompt": "x", "out": "o", "fast": 0}), true).unwrap().fast);
        // the distilled DiT with no steps runs its own eight sigmas; with steps, the ordinary schedule
        let g = j(json!({"prompt": "x", "out": "o"}));
        assert_eq!(g.sigmas(), sample::sigmas(0, true));
        assert_eq!(g.sigmas().len(), 9);
        let g = j(json!({"prompt": "x", "out": "o", "steps": 8}));
        assert_eq!(g.sigmas(), sample::sigmas(8, false));
        assert_ne!(g.sigmas(), sample::sigmas(0, true));
    }

    #[test]
    fn a_bad_generate_job_says_what_is_wrong() {
        let err = |v: Value| parse_generate(&v, false).unwrap_err().0;
        assert!(err(json!({"out": "o"})).contains("a prompt"));
        assert!(err(json!({"prompt": "   ", "out": "o"})).contains("a prompt"));
        assert!(err(json!({"prompt": "x"})).contains("out"));
        assert!(err(json!({"prompt": "x", "out": "o", "width": 650})).contains("multiple of 32"));
        assert!(err(json!({"prompt": "x", "out": "o", "height": 31})).contains("31"));
        assert!(err(json!({"prompt": "x", "out": "o", "width": 0})).contains("multiple"));
        for k in NOT_TAKEN {
            assert!(err(json!({"prompt": "x", "out": "o", k: "a"})).starts_with(k), "{k}");
        }
    }

    #[test]
    fn the_upscaler_windows_cover_the_clip() {
        assert_eq!(windows(121, 121), vec![(0, 121)]);
        assert_eq!(windows(122, 121), vec![(0, 121)], "one frame left is dropped");
        assert_eq!(windows(125, 121), vec![(0, 121)], "a tail under nine is dropped");
        assert_eq!(windows(130, 121), vec![(0, 121), (121, 9)]);
        assert_eq!(windows(242, 121), vec![(0, 121), (121, 121)]);
        assert_eq!(windows(120, 121), vec![(0, 113)], "120 frames: the last seven are cut to the 8k + 1 below");
        assert_eq!(windows(9, 121), vec![(0, 9)]);
        assert!(windows(8, 121).is_empty());
        for total in 9..400 {
            let w = windows(total, WINDOW);
            assert!(w.iter().all(|(_, n)| (n - 1) % 8 == 0 && *n <= WINDOW && *n >= 9), "{total}");
            let done: usize = w.iter().map(|(_, n)| n).sum();
            assert!(done <= total && total - done < WINDOW, "{total}: {done}");
            // consecutive windows start where the last one's source frames ended
            assert!(w.windows(2).all(|p| p[1].0 == p[0].0 + WINDOW));
        }
    }

    #[test]
    fn a_source_must_suit_the_vae() {
        assert!(check_source("a.mp4", 121, 640, 384).is_ok());
        assert!(check_source("a.mp4", 8, 640, 384).unwrap_err().0.contains("at least nine"));
        assert!(check_source("a.mp4", 121, 650, 384).unwrap_err().0.contains("multiples of 32"));
        assert_eq!(parse_upscale(&json!({"source": "/o/a.mp4", "out": "/o/b.mp4"})).unwrap().source, PathBuf::from("/o/a.mp4"));
        assert!(parse_upscale(&json!({"out": "/o/b.mp4"})).is_err() && parse_upscale(&json!({"source": "a"})).is_err());
    }
}
