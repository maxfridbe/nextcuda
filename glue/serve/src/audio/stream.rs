//! Speech streamed as OpenAI's /v1/audio/speech streams it: `Transfer-Encoding: chunked`, the audio of each stretch
//! of the text sent as soon as that stretch is spoken, in any of OpenAI's formats (mp3, opus, aac, flac, wav, pcm).
//!
//! The text is cut into stretches of whole sentences (`chunks`, the first one short so the first sound comes soon),
//! each spoken by the engine as a request of its own; their samples go, in order, into one ffmpeg that runs for the
//! whole answer, and what it writes is the body. One encoder for the whole answer and not one a stretch: an Ogg (opus)
//! or FLAC stream made of several encodes is not one valid stream (each would bring its own headers), speed
//! (`atempo`) keeps its state across the stretches instead of starting again at every one, and the resampling to
//! OpenAI's 24 kHz is done once. `pcm` is exactly OpenAI's: raw s16le, 24 kHz, mono, no header; `wav` is that behind a
//! header whose lengths are 0xFFFFFFFF (unknown while streaming), as OpenAI sends it; mp3 carries no Xing header (it
//! would say a length of 0), aac is ADTS (an MP4 needs its index, which a stream cannot give).
//!
//! The answer starts when the first stretch is spoken, so an error before then is an ordinary JSON error; after it an
//! error ends the connection without the closing chunk (the client sees the audio cut short) and is logged. A client
//! that goes away is seen when a write fails: `gone` is set, the stretch being spoken stops at its next step, and no
//! other is started.

use std::io::{Read, Write};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};

/// OpenAI's rate for pcm (and what every format is made at)
pub const RATE: u32 = 24_000;

/// A response format: its content type and ffmpeg's output for it
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Wire {
    pub name: &'static str,
    pub mime: &'static str,
    args: &'static [&'static str],
    /// a streamed WAV header goes first
    wav_header: bool,
}

const PCM: &[&str] = &["-c:a", "pcm_s16le", "-f", "s16le"];

/// The formats OpenAI streams (url and a song's whole wav are the server's own, not here)
pub fn wire(name: &str) -> Option<Wire> {
    Some(match name {
        "mp3" => Wire { name: "mp3", mime: "audio/mpeg", args: &["-c:a", "libmp3lame", "-b:a", "128k", "-write_xing", "0", "-id3v2_version", "0", "-f", "mp3"], wav_header: false },
        "aac" => Wire { name: "aac", mime: "audio/aac", args: &["-c:a", "aac", "-b:a", "128k", "-f", "adts"], wav_header: false },
        // short Ogg pages: a stretch's sound is not held back for a second
        "opus" => Wire { name: "opus", mime: "audio/ogg", args: &["-c:a", "libopus", "-b:a", "96k", "-page_duration", "200000", "-f", "ogg"], wav_header: false },
        "flac" => Wire { name: "flac", mime: "audio/flac", args: &["-c:a", "flac", "-f", "flac"], wav_header: false },
        "wav" => Wire { name: "wav", mime: "audio/wav", args: PCM, wav_header: true },
        "pcm" => Wire { name: "pcm", mime: "audio/L16", args: PCM, wav_header: false },
        _ => return None,
    })
}

/// A WAV header for 24 kHz mono s16le of a length not known yet
pub fn streaming_wav_header() -> [u8; 44] {
    let mut h = [0u8; 44];
    h[0..4].copy_from_slice(b"RIFF");
    h[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
    h[8..16].copy_from_slice(b"WAVEfmt ");
    h[16..20].copy_from_slice(&16u32.to_le_bytes());
    h[20..22].copy_from_slice(&1u16.to_le_bytes());
    h[22..24].copy_from_slice(&1u16.to_le_bytes());
    h[24..28].copy_from_slice(&RATE.to_le_bytes());
    h[28..32].copy_from_slice(&(RATE * 2).to_le_bytes());
    h[32..34].copy_from_slice(&2u16.to_le_bytes());
    h[34..36].copy_from_slice(&16u16.to_le_bytes());
    h[36..40].copy_from_slice(b"data");
    h[40..44].copy_from_slice(&u32::MAX.to_le_bytes());
    h
}

/// `atempo` takes 0.5-2 a filter; a chain covers OpenAI's 0.25-4 (beyond it: clamped). None: 1
pub fn atempo(speed: f64) -> Option<String> {
    if (speed - 1.0).abs() < 1e-3 {
        return None;
    }
    let mut left = speed.clamp(0.25, 4.0);
    let mut parts = Vec::new();
    while left > 2.0 {
        parts.push("atempo=2.0".to_string());
        left /= 2.0;
    }
    while left < 0.5 {
        parts.push("atempo=0.5".to_string());
        left /= 0.5;
    }
    parts.push(format!("atempo={left:.4}"));
    Some(parts.join(","))
}

/// OpenAI's model names (tts-1, tts-1-hd, gpt-4o-mini-tts ...): here they mean "the speech model serving"
pub fn openai_model(m: &str) -> bool {
    let m = m.trim().to_ascii_lowercase();
    m.starts_with("tts-") || m.starts_with("gpt-")
}

/// OpenAI's voices, each a built-in voice of qwen3-tts-custom: its English speakers (ryan, aiden) for OpenAI's male
/// voices, its clear female ones (vivian, serena) for the others. A saved voice of the same name wins
pub const OPENAI_VOICES: [(&str, &str); 11] = [
    ("alloy", "vivian"), ("ash", "ryan"), ("ballad", "aiden"), ("coral", "serena"), ("echo", "aiden"), ("fable", "ryan"),
    ("nova", "serena"), ("onyx", "ryan"), ("sage", "vivian"), ("shimmer", "serena"), ("verse", "aiden"),
];

/// The built-in voice an OpenAI voice name means on an engine with `builtin` voices: the table's, if the engine has
/// it, else its first; None: not an OpenAI name, or an engine without built-in voices (its own default voice speaks)
pub fn openai_voice(name: &str, builtin: &[String]) -> Option<Option<String>> {
    let lower = name.trim().to_ascii_lowercase();
    let (_, want) = OPENAI_VOICES.iter().find(|(n, _)| *n == lower)?;
    Some(builtin.iter().find(|v| v.eq_ignore_ascii_case(want)).or_else(|| builtin.first()).cloned())
}

/// The text in stretches of whole sentences, in order: the first up to `first` characters (the first sound soon), the
/// others up to `rest`; a sentence longer than that is cut at a comma or a space
pub fn chunks(text: &str, first: usize, rest: usize) -> Vec<String> {
    let mut sentences: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        cur.push(c);
        let ends = matches!(c, '.' | '!' | '?' | '。' | '！' | '？' | '\n') && chars.peek().is_none_or(|n| n.is_whitespace() || !n.is_alphanumeric());
        if ends {
            // closing quotes and brackets stay with their sentence
            while let Some(&n) = chars.peek() {
                if matches!(n, '"' | '\'' | ')' | ']' | '”' | '’' | '»') {
                    cur.push(n);
                    chars.next();
                } else {
                    break;
                }
            }
            if !cur.trim().is_empty() {
                sentences.push(cur.trim().to_string());
            }
            cur.clear();
        }
    }
    if !cur.trim().is_empty() {
        sentences.push(cur.trim().to_string());
    }
    let mut out: Vec<String> = Vec::new();
    let mut now = String::new();
    for s in sentences {
        let limit = if out.is_empty() { first } else { rest };
        if !now.is_empty() && now.chars().count() + 1 + s.chars().count() > limit {
            out.push(std::mem::take(&mut now));
        }
        let limit = if out.is_empty() { first } else { rest };
        let mut s = s;
        // a sentence too long alone: cut at the last comma (else space) before the limit
        while s.chars().count() > limit && now.is_empty() {
            let cut: String = s.chars().take(limit).collect();
            let at = cut.rfind([',', ';', ':']).map(|i| i + 1).or_else(|| cut.rfind(' ')).filter(|&i| i > 0).unwrap_or(cut.len());
            out.push(s[..at].trim().to_string());
            s = s[at..].trim().to_string();
        }
        if !now.is_empty() {
            now.push(' ');
        }
        now.push_str(&s);
    }
    if !now.trim().is_empty() {
        out.push(now);
    }
    out.retain(|c| !c.trim().is_empty());
    out
}

/// A stretch's sound: mono samples at their rate
pub struct Pcm {
    pub samples: Vec<f32>,
    pub rate: u32,
}

/// The HTTP answer's head for a streamed format (each header in `extra` ends \r\n)
pub fn head(w: &Wire, extra: &str) -> String {
    format!("HTTP/1.1 200 OK\r\nContent-Type: {}\r\nTransfer-Encoding: chunked\r\nX-Sample-Rate: {RATE}\r\nCache-Control: no-store\r\n{extra}Connection: close\r\n\r\n", w.mime)
}

/// One HTTP chunk
fn chunk(out: &mut impl Write, b: &[u8]) -> std::io::Result<()> {
    if b.is_empty() {
        return Ok(());
    }
    write!(out, "{:x}\r\n", b.len())?;
    out.write_all(b)?;
    out.write_all(b"\r\n")?;
    out.flush()
}

/// Streams the answer: the head, then the encoder's output as HTTP chunks, from `first` (spoken already) and each
/// stretch `next` gives (None: no more) until the last, then the closing chunk. Err: why it stopped early (the
/// connection is then to be dropped without the closing chunk); `gone` set when the client went away.
pub fn send(out: &mut (impl Write + Send), w: Wire, speed: f64, extra: &str, first: Pcm, next: &mut dyn FnMut() -> Option<Result<Pcm, String>>,
            gone: &AtomicBool) -> Result<(), String> {
    let rate = first.rate.max(1);
    let mut cmd = Command::new("ffmpeg");
    // no probing: the input's format is given, and ffmpeg otherwise reads seconds of it before it writes anything
    cmd.args(["-hide_banner", "-loglevel", "error", "-probesize", "32", "-analyzeduration", "0", "-f", "f32le", "-ar", &rate.to_string(), "-ac", "1",
              "-i", "pipe:0"]);
    let mut filter = vec![format!("aresample={RATE}")];
    filter.extend(atempo(speed));
    cmd.args(["-filter:a", &filter.join(","), "-ac", "1", "-ar", &RATE.to_string()]).args(w.args).args(["-flush_packets", "1", "pipe:1"]);
    let mut child: Child = cmd.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn()
        .map_err(|e| format!("ffmpeg: {e}"))?;
    let mut stdin = child.stdin.take().ok_or("ffmpeg's input")?;
    let stdout = child.stdout.take().ok_or("ffmpeg's output")?;
    let mut stderr = child.stderr.take().ok_or("ffmpeg's errors")?;
    let o = &mut *out;
    let res = std::thread::scope(|sc| {
        // the encoder's output to the client as it comes; it owns ffmpeg's output, so when it stops (the client gone)
        // the encoder's writes fail instead of filling the pipe and stalling the feed
        let writer = sc.spawn(move || -> Result<(), String> {
            let out = o;
            let mut stdout = stdout;
            let mut lead = head(&w, extra).into_bytes();
            let gone_now = |e: std::io::Error| {
                gone.store(true, Ordering::Relaxed);
                format!("the client went away ({e})")
            };
            out.write_all(&lead).map_err(gone_now)?;
            lead.clear();
            if w.wav_header {
                chunk(out, &streaming_wav_header()).map_err(gone_now)?;
            }
            let mut buf = vec![0u8; 16 * 1024];
            loop {
                let n = stdout.read(&mut buf).map_err(|e| format!("ffmpeg's output: {e}"))?;
                if n == 0 {
                    return Ok(());
                }
                chunk(out, &buf[..n]).map_err(gone_now)?;
            }
        });
        let feed = |p: &Pcm, stdin: &mut std::process::ChildStdin| -> Result<(), String> {
            if p.rate.max(1) != rate {
                return Err(format!("a stretch at {} Hz after one at {rate} Hz", p.rate));
            }
            let b: Vec<u8> = p.samples.iter().flat_map(|s| s.to_le_bytes()).collect();
            stdin.write_all(&b).map_err(|e| format!("ffmpeg's input: {e}"))
        };
        let mut fed = feed(&first, &mut stdin);
        while fed.is_ok() && !gone.load(Ordering::Relaxed) {
            match next() {
                None => break,
                Some(Ok(p)) => fed = feed(&p, &mut stdin),
                Some(Err(e)) => fed = Err(e),
            }
        }
        if fed.is_err() || gone.load(Ordering::Relaxed) {
            // stopped early: the encoder is ended, not drained
            let _ = child.kill();
        }
        // closing its input is what makes the encoder write its tail and end
        drop(stdin);
        let wrote = writer.join().unwrap_or_else(|_| Err("the writer panicked".into()));
        // the client gone is the cause (the encoder then fails too, its output closed)
        wrote.and(fed)
    });
    let status = child.wait().map_err(|e| format!("ffmpeg: {e}"))?;
    res?;
    if !status.success() {
        let mut e = String::new();
        let _ = stderr.read_to_string(&mut e);
        return Err(format!("ffmpeg {status}: {}", e.lines().last().unwrap_or("")));
    }
    out.write_all(b"0\r\n\r\n").and_then(|_| out.flush()).map_err(|e| {
        gone.store(true, Ordering::Relaxed);
        format!("the client went away ({e})")
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    /// A stretch of `secs` at `rate`: a constant (which tells the stretches apart) under a quiet tone
    pub(crate) fn tone(secs: f32, rate: u32, level: f32) -> Pcm {
        let n = (secs * rate as f32) as usize;
        Pcm { samples: (0..n).map(|i| level + 0.05 * (i as f32 / rate as f32 * 440.0 * std::f32::consts::TAU).sin()).collect(), rate }
    }

    /// The body of a chunked answer: (its head, the data, whether it closed properly)
    pub(crate) fn dechunk(raw: &[u8]) -> (String, Vec<u8>, bool) {
        let h = raw.windows(4).position(|w| w == b"\r\n\r\n").map(|p| p + 4).expect("a head");
        let head = String::from_utf8_lossy(&raw[..h]).to_string();
        let mut rest = &raw[h..];
        let mut body = Vec::new();
        while let Some(p) = rest.windows(2).position(|w| w == b"\r\n") {
            let n = usize::from_str_radix(std::str::from_utf8(&rest[..p]).unwrap().trim(), 16).unwrap();
            if n == 0 {
                return (head, body, true);
            }
            if rest.len() < p + 2 + n + 2 {
                break;
            }
            body.extend_from_slice(&rest[p + 2..p + 2 + n]);
            rest = &rest[p + 4 + n..];
        }
        (head, body, false)
    }

    /// The decoded length in seconds (ffmpeg, to 24 kHz mono s16)
    pub(crate) fn decoded_secs(b: &[u8], fmt: &str) -> f64 {
        let dir = std::env::temp_dir().join(format!("nc-speech-{}-{}", std::process::id(), Instant::now().elapsed().as_nanos() ^ b.len() as u128));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join(format!("out.{fmt}"));
        std::fs::write(&p, b).unwrap();
        let mut c = Command::new("ffmpeg");
        c.args(["-v", "error"]);
        if fmt == "pcm" {
            c.args(["-f", "s16le", "-ar", "24000", "-ac", "1"]);
        }
        let o = c.arg("-i").arg(&p).args(["-f", "s16le", "-ar", "24000", "-ac", "1", "-"]).output().unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        assert!(o.status.success(), "{fmt}: {}", String::from_utf8_lossy(&o.stderr));
        o.stdout.len() as f64 / 48_000.0
    }

    /// A writer that notes when each write came (from its start)
    struct Clocked {
        t0: Instant,
        b: Arc<Mutex<Vec<u8>>>,
        at: Arc<Mutex<Vec<Duration>>>,
        /// fail every write after this many bytes (a client that hangs up)
        cut: Option<usize>,
    }

    impl Write for Clocked {
        fn write(&mut self, d: &[u8]) -> std::io::Result<usize> {
            let mut b = self.b.lock().unwrap();
            if self.cut.is_some_and(|c| b.len() > c) {
                return Err(std::io::Error::new(std::io::ErrorKind::BrokenPipe, "hung up"));
            }
            b.extend_from_slice(d);
            self.at.lock().unwrap().push(self.t0.elapsed());
            Ok(d.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// What a run gave: its end, the raw answer, when each write came, when each stretch was made, stretches asked for
    type Ran = (Result<(), String>, Vec<u8>, Vec<Duration>, Vec<Duration>, usize);

    /// A fake engine: `n` stretches of 1 s at 16 kHz (resampled on the way), each taking `delay`, stretch `fail` failing
    fn run(fmt: &str, speed: f64, n: usize, delay: Duration, fail: Option<usize>, cut: Option<usize>) -> Ran {
        let t0 = Instant::now();
        let (b, at) = (Arc::new(Mutex::new(Vec::new())), Arc::new(Mutex::new(Vec::new())));
        let mut out = Clocked { t0, b: b.clone(), at: at.clone(), cut };
        let mut made = vec![];
        std::thread::sleep(delay);
        let first = tone(1.0, 16_000, 0.1);
        made.push(t0.elapsed());
        let mut i = 1;
        let mut asked = 1;
        let gone = AtomicBool::new(false);
        let mut next = || {
            if i >= n {
                return None;
            }
            asked += 1;
            std::thread::sleep(delay);
            let k = i;
            i += 1;
            if Some(k) == fail {
                return Some(Err(format!("stretch {k} failed")));
            }
            made.push(t0.elapsed());
            Some(Ok(tone(1.0, 16_000, 0.1 * (k as f32 + 1.0))))
        };
        let r = send(&mut out, wire(fmt).unwrap(), speed, "", first, &mut next, &gone);
        let raw = b.lock().unwrap().clone();
        let at = at.lock().unwrap().clone();
        (r, raw, at, made, asked)
    }

    #[test]
    fn each_format_is_one_decodable_stream_in_http_chunks() {
        for fmt in ["mp3", "aac", "opus", "flac", "wav", "pcm"] {
            let (r, raw, _, _, _) = run(fmt, 1.0, 3, Duration::from_millis(10), None, None);
            r.unwrap();
            let (head, body, closed) = dechunk(&raw);
            assert!(closed, "{fmt}: the closing chunk");
            assert!(head.contains("Transfer-Encoding: chunked") && head.contains(&format!("Content-Type: {}", wire(fmt).unwrap().mime)), "{head}");
            if fmt == "wav" {
                assert_eq!(&body[..4], b"RIFF");
                assert_eq!(&body[4..8], &[0xff; 4]);
            }
            let secs = decoded_secs(&body, fmt);
            assert!((secs - 3.0).abs() < 0.12, "{fmt}: {secs} s");
        }
    }

    #[test]
    fn pcm_is_openais_raw_24k_mono_s16le_in_order() {
        let (r, raw, _, _, _) = run("pcm", 1.0, 4, Duration::from_millis(10), None, None);
        r.unwrap();
        let (_, body, _) = dechunk(&raw);
        let s: Vec<i16> = body.as_chunks::<2>().0.iter().map(|c| i16::from_le_bytes(*c)).collect();
        assert!((s.len() as i64 - 96_000).abs() < 200, "{} samples", s.len());
        for k in 0..4 {
            let part = &s[k * 24_000 + 2_000..(k + 1) * 24_000 - 2_000];
            let mean = part.iter().map(|&v| v as f64).sum::<f64>() / part.len() as f64 / 32767.0;
            assert!((mean - 0.1 * (k as f64 + 1.0)).abs() < 0.01, "stretch {k}: {mean}");
        }
    }

    #[test]
    fn speed_runs_over_the_whole_stream() {
        for (fmt, speed, want) in [("mp3", 2.0, 2.0), ("pcm", 0.5, 8.0), ("opus", 4.0, 1.0), ("wav", 1.25, 3.2)] {
            let (r, raw, _, _, _) = run(fmt, speed, 4, Duration::from_millis(5), None, None);
            r.unwrap();
            let secs = decoded_secs(&dechunk(&raw).1, fmt);
            assert!((secs - want).abs() < 0.12, "{fmt} at {speed}: {secs} s, wanted {want}");
        }
    }

    #[test]
    fn the_first_sound_goes_out_before_the_last_stretch_is_spoken() {
        for fmt in ["pcm", "mp3", "opus", "wav", "aac", "flac"] {
            let (r, raw, at, made, _) = run(fmt, 1.0, 5, Duration::from_millis(300), None, None);
            r.unwrap();
            // the head, then the first data (the head is one write, the first chunk's size line the next)
            let first = at[1];
            assert!(first < made[1], "{fmt}: first bytes {first:?}, stretch 2 made at {:?}", made[1]);
            let total = *at.last().unwrap();
            eprintln!("{fmt}: first byte {first:?}, total {total:?} ({} bytes, 5 stretches of 0.3 s)", raw.len());
        }
    }

    #[test]
    fn an_error_mid_stream_ends_it_without_the_closing_chunk() {
        let (r, raw, _, _, _) = run("pcm", 1.0, 4, Duration::from_millis(10), Some(2), None);
        assert!(r.unwrap_err().contains("stretch 2 failed"));
        let (_, body, closed) = dechunk(&raw);
        assert!(!closed, "no closing chunk: the client knows it is cut short");
        assert!(body.len() >= 40_000, "the first stretch went out: {}", body.len());
    }

    #[test]
    fn a_client_that_hangs_up_stops_the_rest() {
        let (r, _, _, _, asked) = run("mp3", 1.0, 8, Duration::from_millis(200), None, Some(1000));
        assert!(r.unwrap_err().contains("went away"));
        assert!(asked <= 3, "{asked} of 8 stretches asked for after the hang-up");
    }

    #[test]
    fn text_is_cut_at_sentences_short_first() {
        let t = "Streaming raw audio chunks directly to standard output. This is the second sentence. And here is a third one, which is a little longer than the others are.";
        let c = chunks(t, 60, 120);
        assert_eq!(c[0], "Streaming raw audio chunks directly to standard output.");
        assert_eq!(c[1], "This is the second sentence. And here is a third one, which is a little longer than the others are.");
        assert_eq!(c.join(" "), t);
        let long = "word ".repeat(100);
        let c = chunks(&long, 50, 50);
        assert!(c.iter().all(|x| x.chars().count() <= 50), "{c:?}");
        assert_eq!(c.join(" ").split_whitespace().count(), 100);
        assert_eq!(chunks("Dr. Smith went home.", 100, 100), vec!["Dr. Smith went home."]);
        assert_eq!(chunks("  ", 10, 10), Vec::<String>::new());
    }

    #[test]
    fn openai_names() {
        let builtin: Vec<String> = ["serena", "vivian", "uncle_fu", "ryan", "aiden"].iter().map(|s| s.to_string()).collect();
        assert_eq!(openai_voice("alloy", &builtin), Some(Some("vivian".into())));
        assert_eq!(openai_voice("Onyx", &builtin), Some(Some("ryan".into())));
        assert_eq!(openai_voice("nova", &["eric".into()]), Some(Some("eric".into())), "else the engine's first");
        assert_eq!(openai_voice("verse", &[]), Some(None), "an engine with no built-in voices speaks its own");
        assert_eq!(openai_voice("ryan", &builtin), None, "not an OpenAI name");
        assert!(openai_model("tts-1") && openai_model("tts-1-hd") && openai_model("gpt-4o-mini-tts") && !openai_model("qwen3-tts-custom"));
        assert_eq!(atempo(4.0).unwrap(), "atempo=2.0,atempo=2.0000");
        assert_eq!(atempo(0.25).unwrap(), "atempo=0.5,atempo=0.5000");
        assert_eq!(atempo(1.0), None);
    }
}
