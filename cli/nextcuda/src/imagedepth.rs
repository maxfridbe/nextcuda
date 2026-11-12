//! `nextcuda image depth` and `depth-clip`: Depth Anything V2 in this process - pictures' depth (the network's own
//! output kept for a check against reference/depthanything/ref.py, a grey PNG to look at), clocked; a clip made a depth
//! control video as the image server makes it for video.control (video/studio/control.rs make_depth).
//!
//!     nextcuda image depth PICTURE... [--model ID | --file model.safetensors] [--out DIR] [--runs N] [--gpu N]
//!     nextcuda image depth-clip SRC.mp4 DST.mp4 [--width W --height H] [--frames N] [--start S] [--model ID | --file F] [--gpu N]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use nextcuda_image::{ImageEngine, Picture};
use nextcuda_image_depthanything::DepthAnything;
use nextcuda_models::{config::Config, registry as models};
use nextcuda_serve::video::studio::control;

use crate::image::opt;

/// The positional arguments (each --flag takes one value)
fn positional(args: &[String]) -> Vec<&String> {
    let mut v = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a.starts_with("--") {
            it.next();
        } else {
            v.push(a);
        }
    }
    v
}

/// The engine: --file (a safetensors file), else the registered model (--model, default depth-anything-v2-small)
fn engine(cfg: &Config, args: &[String]) -> Result<(DepthAnything, Arc<nextcuda_core::Gpu>), String> {
    nextcuda_core::use_kind("image");
    let path = match opt(args, "--file") {
        Some(f) => PathBuf::from(f),
        None => {
            let id = opt(args, "--model").unwrap_or("depth-anything-v2-small");
            let m = models::all(cfg)?.into_iter().find(|m| m["id"] == id).ok_or_else(|| format!("no model {id} (nextcuda models pull depth-anything-v2-small)"))?;
            PathBuf::from(m["files"]["model"].as_str().ok_or_else(|| format!("{id}: no model file"))?)
        }
    };
    let i: usize = opt(args, "--gpu").and_then(|v| v.parse().ok()).unwrap_or(0);
    let gpu = nextcuda_core::Gpu::open(i).map_err(|e| e.0)?;
    let free0 = gpu.memory().ok().and_then(|m| m.1);
    let e = DepthAnything::load(&path, &gpu).map_err(|e| e.0)?;
    let free1 = gpu.memory().ok().and_then(|m| m.1);
    eprintln!("loaded {} in {:.2} s on {} ({})", path.display(), e.load_secs, gpu.name, mem(free0, free1));
    Ok((e, gpu))
}

fn mem(before: Option<u64>, after: Option<u64>) -> String {
    match (before, after) {
        (Some(a), Some(b)) => format!("the card's free memory {:.0} MiB less", (a as f64 - b as f64) / 1048576.0),
        _ => "memory unknown".into(),
    }
}

/// `nextcuda image depth`
pub fn depth_cmd(cfg: &Config, args: &[String]) -> Result<(), String> {
    let pics = positional(args);
    if pics.is_empty() {
        return Err("nextcuda image depth PICTURE... [--model ID | --file model.safetensors] [--out DIR] [--runs N] [--gpu N]".into());
    }
    let (e, gpu) = engine(cfg, args)?;
    let out = PathBuf::from(opt(args, "--out").unwrap_or("."));
    std::fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
    let runs: usize = opt(args, "--runs").and_then(|v| v.parse().ok()).unwrap_or(1).max(1);
    let free0 = gpu.memory().ok().and_then(|m| m.1);
    for p in pics {
        let pic = Picture::decode(&std::fs::read(p).map_err(|e| format!("{p}: {e}"))?)?.rgb();
        let stem = Path::new(p).file_stem().and_then(|s| s.to_str()).unwrap_or("picture").to_string();
        // the network's input and its own output, at its input size: the reference's pixel_values and predicted_depth
        let (px, iw, ih) = nextcuda_image_depthanything::prep::prepare(&pic);
        let bytes: Vec<u8> = px.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(out.join(format!("{stem}.input.{iw}x{ih}.f32")), bytes).map_err(|e| e.to_string())?;
        let (raw, w, h, t) = e.raw(&pic).map_err(|e| e.0)?;
        let bytes: Vec<u8> = raw.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(out.join(format!("{stem}.{w}x{h}.f32")), bytes).map_err(|e| e.to_string())?;
        let (d, _) = e.timed(&pic).map_err(|e| e.0)?;
        let (lo, hi) = d.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
        let grey = Picture { width: pic.width, height: pic.height, channels: 1, data: d.iter().map(|&v| control::depth_byte(v, lo, hi)).collect() };
        grey.write_png(&out.join(format!("{stem}-depth.png")), &[("model", "depth-anything-v2-small".into())])?;
        println!("{stem}: {}x{} -> input {w}x{h}, depth {lo:.4}..{hi:.4}; first: prepare {:.1} ms, network {:.1} ms", pic.width, pic.height,
                 t.prepare * 1e3, t.network * 1e3);
        if runs > 1 {
            let mut sum = nextcuda_image_depthanything::Times::default();
            let t0 = Instant::now();
            for _ in 0..runs {
                let (_, t) = e.timed(&pic).map_err(|e| e.0)?;
                sum.prepare += t.prepare;
                sum.network += t.network;
                sum.resize += t.resize;
            }
            let n = runs as f64;
            println!("  {runs} runs: {:.1} ms a picture (prepare {:.1}, network {:.1}, resize {:.1})", t0.elapsed().as_secs_f64() / n * 1e3,
                     sum.prepare / n * 1e3, sum.network / n * 1e3, sum.resize / n * 1e3);
        }
    }
    println!("{}; {}", mem(free0, gpu.memory().ok().and_then(|m| m.1)), e.report().join(", "));
    Ok(())
}

/// `nextcuda image depth-clip`: what the image server's POST /v1/videos/depth does, in this process
pub fn depth_clip_cmd(cfg: &Config, args: &[String]) -> Result<(), String> {
    let pos = positional(args);
    let (src, dst) = match (pos.first(), pos.get(1)) {
        (Some(a), Some(b)) => (PathBuf::from(a), PathBuf::from(b)),
        _ => return Err("nextcuda image depth-clip SRC.mp4 DST.mp4 [--width W --height H] [--frames N] [--start S] [--model ID | --file F] [--gpu N]".into()),
    };
    let (w, h, secs) = control::probe(&src).ok_or_else(|| format!("{}: not a clip ffprobe reads", src.display()))?;
    let mut p = serde_json::json!({"control": "depth"});
    for (k, f) in [("width", "--width"), ("height", "--height"), ("frames", "--frames"), ("start_s", "--start")] {
        if let Some(v) = opt(args, f).and_then(|v| v.parse::<f64>().ok()) {
            p[k] = serde_json::json!(v);
        }
    }
    let s = control::settings(&p, (w, h, secs)).map_err(|e| e.message)?;
    let (e, _gpu) = engine(cfg, args)?;
    let t0 = Instant::now();
    let mut net = 0.0;
    let (made, range) = control::make_depth(&src, &dst, &s, &mut |rgb| {
        let t = Instant::now();
        let pic = Picture { width: s.width, height: s.height, channels: 3, data: rgb.to_vec() };
        let d = e.depth(&pic).map_err(|e| e.0);
        net += t.elapsed().as_secs_f64();
        d
    }, &mut |n| {
        if n % 24 == 0 {
            eprintln!("  {n} / {} frames, {:.1} s", s.frames, t0.elapsed().as_secs_f64());
        }
        Ok(())
    })?;
    println!("{}: {made} frames {}x{} in {:.1} s ({:.1} ms a frame of depth), range {:.4}..{:.4}", dst.display(), s.width, s.height, t0.elapsed().as_secs_f64(),
             net / made.max(1) as f64 * 1e3, range.0, range.1);
    Ok(())
}
