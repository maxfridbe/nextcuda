//! A clip in and out: frames (and sound) to .mp4 through ffmpeg (H.264 yuv420p + AAC, as H3's engine writes them), and
//! a picture read and fitted to the canvas (the first frame of an image-to-video clip).

use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

use nextcuda_core::{Error, Result};

/// Frames [count, h, w, 3] in [0, 1] (interleaved) and stereo sound -> `path`; the sound cut or padded to exactly the
/// frames' length
pub fn write_mp4(path: &Path, px: &[f32], count: usize, h: usize, w: usize, fps: f32, audio: Option<(&[Vec<f32>], u32)>) -> Result<()> {
    if px.len() < count * h * w * 3 {
        return Err(Error(format!("write_mp4: {} values for {count} x {h} x {w} x 3", px.len())));
    }
    let tmp = path.with_extension("pcm.tmp");
    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", &format!("{w}x{h}"), "-r", &format!("{fps}"), "-i", "-"]);
    if let Some((ch, rate)) = audio {
        let want = ((count as f64 / fps as f64) * rate as f64).round() as usize;
        let mut pcm = Vec::with_capacity(want * ch.len() * 2);
        for i in 0..want {
            for c in ch {
                let v = c.get(i).copied().unwrap_or(0.0);
                pcm.extend_from_slice(&((v.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes());
            }
        }
        std::fs::write(&tmp, pcm).map_err(|e| Error(format!("{}: {e}", tmp.display())))?;
        cmd.args(["-f", "s16le", "-ar", &rate.to_string(), "-ac", &ch.len().to_string(), "-i"]).arg(&tmp);
        cmd.args(["-c:a", "aac", "-b:a", "192k"]);
    }
    cmd.args(["-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p"]).arg(path);
    let mut child = cmd.stdin(Stdio::piped()).spawn().map_err(|e| Error(format!("starting ffmpeg: {e}")))?;
    {
        let mut stdin = child.stdin.take().ok_or("ffmpeg has no stdin")?;
        let frame = h * w * 3;
        let mut rgb = vec![0u8; frame];
        for f in 0..count {
            for (i, v) in px[f * frame..(f + 1) * frame].iter().enumerate() {
                rgb[i] = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
            }
            stdin.write_all(&rgb).map_err(|e| Error(format!("writing frames to ffmpeg: {e}")))?;
        }
    }
    let st = child.wait().map_err(|e| Error(format!("ffmpeg: {e}")))?;
    let _ = std::fs::remove_file(&tmp);
    if !st.success() {
        return Err(Error(format!("ffmpeg failed writing {} ({st})", path.display())));
    }
    Ok(())
}

/// A picture (or a video's first frame) resized to w x h (Lanczos, stretched as the movie maker's PIL resize does):
/// [h, w, 3] in [0, 1]
pub fn read_picture(path: &Path, w: usize, h: usize) -> Result<Vec<f32>> {
    let out = Command::new("ffmpeg")
        .args(["-loglevel", "error", "-i"])
        .arg(path)
        .args(["-frames:v", "1", "-vf", &format!("scale={w}:{h}:flags=lanczos"), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
        .output()
        .map_err(|e| Error(format!("starting ffmpeg: {e}")))?;
    if !out.status.success() || out.stdout.len() < w * h * 3 {
        return Err(Error(format!("{}: ffmpeg could not read a picture ({})", path.display(), String::from_utf8_lossy(&out.stderr).trim())));
    }
    Ok(out.stdout[..w * h * 3].iter().map(|b| *b as f32 / 255.0).collect())
}

/// Every frame of a video at its own size: ([frames, h, w, 3] in [0, 1], frames, w, h, fps, whether it has sound)
pub fn read_video(path: &Path) -> Result<(Vec<f32>, usize, usize, usize, f32, bool)> {
    let probe = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate", "-of", "csv=p=0:s=x"])
        .arg(path)
        .output()
        .map_err(|e| Error(format!("starting ffprobe: {e}")))?;
    let s = String::from_utf8_lossy(&probe.stdout).to_string();
    let f: Vec<&str> = s.trim().lines().next().unwrap_or("").split('x').collect();
    if f.len() < 3 {
        return Err(Error(format!("{}: ffprobe found no video ({})", path.display(), s.trim())));
    }
    let (w, h): (usize, usize) = (f[0].parse().unwrap_or(0), f[1].parse().unwrap_or(0));
    let fps = f[2].split_once('/').map_or(24.0, |(a, b)| a.parse::<f32>().unwrap_or(24.0) / b.parse::<f32>().unwrap_or(1.0).max(1.0));
    let audio = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0"])
        .arg(path)
        .output()
        .is_ok_and(|o| !o.stdout.is_empty());
    let out = Command::new("ffmpeg")
        .args(["-loglevel", "error", "-i"])
        .arg(path)
        .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
        .output()
        .map_err(|e| Error(format!("starting ffmpeg: {e}")))?;
    let frame = w * h * 3;
    if frame == 0 || out.stdout.len() < frame {
        return Err(Error(format!("{}: ffmpeg could not read frames", path.display())));
    }
    let n = out.stdout.len() / frame;
    Ok((out.stdout[..n * frame].iter().map(|b| *b as f32 / 255.0).collect(), n, w, h, fps, audio))
}

/// A clip with its picture replaced by `px` (at its new size) and its own sound copied
pub fn replace_picture(src: &Path, out: &Path, px: &[f32], count: usize, h: usize, w: usize, fps: f32, has_audio: bool) -> Result<()> {
    let tmp = out.with_extension("part.mp4");
    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", &format!("{w}x{h}"), "-r", &format!("{fps}"), "-i", "-"]);
    if has_audio {
        cmd.arg("-i").arg(src).args(["-map", "0:v", "-map", "1:a", "-c:a", "copy", "-shortest"]);
    }
    cmd.args(["-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p"]).arg(&tmp);
    let mut child = cmd.stdin(Stdio::piped()).spawn().map_err(|e| Error(format!("starting ffmpeg: {e}")))?;
    {
        let mut stdin = child.stdin.take().ok_or("ffmpeg has no stdin")?;
        let frame = h * w * 3;
        let mut rgb = vec![0u8; frame];
        for f in 0..count {
            for (i, v) in px[f * frame..(f + 1) * frame].iter().enumerate() {
                rgb[i] = (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            }
            stdin.write_all(&rgb).map_err(|e| Error(format!("writing frames to ffmpeg: {e}")))?;
        }
    }
    let st = child.wait().map_err(|e| Error(format!("ffmpeg: {e}")))?;
    if !st.success() {
        return Err(Error(format!("ffmpeg failed writing {} ({st})", out.display())));
    }
    std::fs::rename(&tmp, out).map_err(|e| Error(format!("{}: {e}", out.display())))
}

/// A picture [h, w, 3] in [0, 1] through H.264 at `crf` (libx264 veryfast, yuv420p) and back - the compression the
/// model's image conditioning was trained with
pub fn crf_roundtrip(px: &[f32], w: usize, h: usize, crf: u32) -> Result<Vec<f32>> {
    let dir = std::env::temp_dir().join(format!("ltx2-crf-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos())));
    std::fs::create_dir_all(&dir).map_err(|e| Error(format!("{}: {e}", dir.display())))?;
    let mp4 = dir.join("f.mp4");
    let mut child = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", &format!("{w}x{h}"), "-r", "1", "-i", "-", "-c:v", "libx264",
               "-preset", "veryfast", "-crf", &crf.to_string(), "-pix_fmt", "yuv420p"])
        .arg(&mp4)
        .stdin(Stdio::piped())
        .spawn()
        .map_err(|e| Error(format!("starting ffmpeg: {e}")))?;
    {
        let mut stdin = child.stdin.take().ok_or("ffmpeg has no stdin")?;
        let rgb: Vec<u8> = px.iter().map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8).collect();
        stdin.write_all(&rgb).map_err(|e| Error(format!("ffmpeg: {e}")))?;
    }
    let ok = child.wait().map_err(|e| Error(format!("ffmpeg: {e}")))?.success();
    let back = if ok { read_picture(&mp4, w, h) } else { Err(Error("ffmpeg could not re-compress the first frame".into())) };
    let _ = std::fs::remove_dir_all(&dir);
    back
}
