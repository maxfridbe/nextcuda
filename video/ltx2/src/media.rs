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

#[cfg(test)]
mod tests {
    use super::*;

    fn ffmpeg() -> bool {
        Command::new("ffmpeg").arg("-version").stdout(Stdio::null()).stderr(Stdio::null()).status().is_ok_and(|s| s.success())
    }

    fn tmp(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("ltx2-media-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// `count` frames of a w x h picture whose brightness rises with the frame
    fn clip(count: usize, h: usize, w: usize) -> Vec<f32> {
        (0..count).flat_map(|f| std::iter::repeat_n(0.2 + 0.6 * f as f32 / count as f32, h * w * 3)).collect()
    }

    fn probe(p: &Path, entry: &str, stream: &str) -> String {
        let o = Command::new("ffprobe").args(["-v", "error", "-select_streams", stream, "-show_entries", entry, "-of", "csv=p=0"]).arg(p).output().unwrap();
        String::from_utf8_lossy(&o.stdout).trim().to_string()
    }

    #[test]
    fn a_clip_is_written_read_and_its_sound_is_cut_to_the_frames() {
        if !ffmpeg() {
            eprintln!("no ffmpeg: skipped");
            return;
        }
        let d = tmp("rw");
        let (count, h, w) = (24, 64, 96);
        let px = clip(count, h, w);
        // 3 s of sound for a 1 s clip: cut to the frames' length; stereo at 48 kHz
        let tone: Vec<f32> = (0..3 * 48_000).map(|i| (i as f32 * 0.05).sin() * 0.5).collect();
        let p = d.join("a.mp4");
        write_mp4(&p, &px, count, h, w, 24.0, Some((&[tone.clone(), tone.clone()], 48_000))).unwrap();
        let dur: f64 = probe(&p, "format=duration", "a:0").parse().unwrap_or(0.0);
        assert!((dur - 1.0).abs() < 0.1, "the clip is one second, not three: {dur}");
        assert_eq!(probe(&p, "stream=channels", "a:0"), "2");
        assert_eq!(probe(&p, "stream=sample_rate", "a:0"), "48000");
        let (back, n, bw, bh, fps, audio) = read_video(&p).unwrap();
        assert_eq!((n, bw, bh, audio), (24, 96, 64, true));
        assert!((fps - 24.0).abs() < 1e-3);
        // brightness survives H.264 within a few levels and still rises
        let mean = |f: usize| back[f * h * w * 3..(f + 1) * h * w * 3].iter().sum::<f32>() / (h * w * 3) as f32;
        assert!((mean(0) - 0.2).abs() < 0.03 && (mean(23) - px[23 * h * w * 3]).abs() < 0.03, "{} {}", mean(0), mean(23));
        // a short sound is padded with silence to the frames' length
        let q = d.join("b.mp4");
        write_mp4(&q, &px, count, h, w, 24.0, Some((&[vec![0.1; 4800], vec![0.1; 4800]], 48_000))).unwrap();
        let dur: f64 = probe(&q, "format=duration", "a:0").parse().unwrap_or(0.0);
        assert!((dur - 1.0).abs() < 0.1, "{dur}");
        // no sound: no audio stream
        let r = d.join("c.mp4");
        write_mp4(&r, &px, count, h, w, 24.0, None).unwrap();
        assert!(!read_video(&r).unwrap().5);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn too_few_values_are_refused_before_ffmpeg_runs() {
        let e = write_mp4(Path::new("/nonexistent/x.mp4"), &[0.0; 10], 2, 4, 4, 24.0, None).unwrap_err().0;
        assert!(e.contains("10 values for 2 x 4 x 4 x 3"), "{e}");
    }

    #[test]
    fn a_picture_is_fitted_to_the_canvas_and_the_sound_survives_a_new_picture() {
        if !ffmpeg() {
            eprintln!("no ffmpeg: skipped");
            return;
        }
        let d = tmp("pic");
        let (count, h, w) = (9, 64, 64);
        let px = clip(count, h, w);
        let tone: Vec<f32> = (0..48_000).map(|i| (i as f32 * 0.05).sin() * 0.5).collect();
        let src = d.join("src.mp4");
        write_mp4(&src, &px, count, h, w, 24.0, Some((&[tone.clone(), tone], 48_000))).unwrap();
        // a video's first frame as a picture, stretched to 128 x 32
        let pic = read_picture(&src, 128, 32).unwrap();
        assert_eq!(pic.len(), 32 * 128 * 3);
        assert!((pic[0] - 0.2).abs() < 0.03);
        // a bigger picture in its place, its own sound copied
        let big = clip(count, 128, 128);
        let out = d.join("big.mp4");
        replace_picture(&src, &out, &big, count, 128, 128, 24.0, true).unwrap();
        let (_, n, bw, bh, _, audio) = read_video(&out).unwrap();
        assert_eq!((n, bw, bh, audio), (9, 128, 128, true));
        // a re-compressed still keeps its size and stays close to itself
        let still = vec![0.5f32; 64 * 64 * 3];
        let rt = crf_roundtrip(&still, 64, 64, 18).unwrap();
        assert_eq!(rt.len(), still.len());
        assert!(rt.iter().all(|v| (v - 0.5).abs() < 0.03));
        assert!(read_picture(&d.join("missing.png"), 8, 8).is_err());
        let _ = std::fs::remove_dir_all(&d);
    }
}
