//! ZONOS3-Mini (Zyphra, Apache-2.0): multilingual speech at 44.1 kHz, in a preset voice or a voice cloned from a
//! recording. An 11-layer backbone of 1024 (gated DeltaNet mixers with a global-attention layer in four, Full AttnRes,
//! a top-2-of-64 latent MoE) reads SuperBPE text after a voice - a speaker row (a 192-d MECT-B2 embedding, projected)
//! or a reference prefix (a recording's codes) - and three condition rows (loudness, bandwidth, speaking rate), and
//! writes ten 4096-code books at 15 Hz in the delay pattern; the ZAC codec's MR-iSTFT decoder makes them 44.1 kHz.
//!
//! ```text
//!   prompt (prompt.rs): [speaker row | reference prefix] [3 condition rows] BOS text EOS [3 silence rows]
//!   loop (lm.rs + sample.rs): a row of ten codes a step, codebook c of frame f on row f + c; the first end-of-audio
//!     fixes the last frame, eleven rows more drain the delay (speaker CFG: a twin without the voice, in lockstep)
//!   decode (codec.rs): the rows undelayed, the silence frames decoded ahead and cut, -3 dB and the soft limiter
//! ```
//!
//! Voices: a preset's codes (.codes.npy: the reference prefix, the checkpoint's default speaker mode) or its embedding
//! (.embedding.npy: the speaker row); a recording by the speaker encoder (spk.rs: the speaker row). Steering (steer.rs)
//! doses named directions. The files are Zyphra's GGUFs (`Zyphra/ZONOS3-Mini`, gguf/): the backbone, a decoder
//! (zac-keep2: the default, 2 layers a stage; zac: the full one), the speaker encoder and the steering bundle.
//!
//! Origin: a port of Zyphra's reference runtime (github.com/Zyphra/ZONOS3-Mini: zonos3.reference, the ggml server
//! and the MLX server; MIT, the model Apache-2.0) to Rust and this project's CUDA kernels. Not ported: raw-reference
//! cloning from a recording (it needs the 4 GB acoustic encoder and the 1.5 GB SSL teacher; a preset's shipped codes
//! are used as they are), text normalization (the server's speech text pass runs before), LACI, speech editing,
//! accent adapters, control spans and streaming decode.

pub mod check;
pub mod codec;
mod ffi;
pub mod lm;
pub mod ops;
pub mod prompt;
pub mod sample;
pub mod spk;
pub mod steer;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use nextcuda_audio::{At, Audio, AudioEngine, AudioKind, AudioRequest, Defaults, EngineOption, Error, LoadOptions, ModelFiles, Result, Speech, Step};
use nextcuda_core::Gpu;
use nextcuda_tok::Tokenizer;

use crate::codec::Decoder;
use crate::lm::{Cond, Lm};
use crate::ops::{Ops, Weights};
use crate::prompt::Voice;
use crate::sample::{Rng, Sampling};
use crate::spk::SpeakerEncoder;
use crate::steer::Steer;

pub const ARCH: &str = lm::ARCH;
pub const ROLES: &[&str] = &["model", "decoder", "decoder-full", "speaker-encoder", "steer", "voice-*"];
pub const RATE: u32 = 44100;
/// code frames a second (2940 samples each)
pub const FPS: f32 = 15.0;
/// the GGUF servers' frame budget (the generation registry's max_frames)
pub const MAX_FRAMES: usize = 2000;
/// a text longer than this many characters is said in stretches of whole sentences
const MAX_CHARS: usize = 400;
/// the silence between stretches, seconds
const GAP: f32 = 0.25;
/// the soft limiter's knee (-1 dBFS)
const KNEE: f32 = 0.891_250_9;

pub const OPTIONS: &[EngineOption] = &[
    EngineOption { name: "speaker-mode", env: "", value: "reference|embedding", help: "how a preset voice comes in: its codes as a reference prefix (reference, the model's default) or its speaker embedding (embedding; steering's voice controls need it). A recording is always cloned by its embedding", at: At::Request },
    EngineOption { name: "controls", env: "", value: "ID=DOSE,...", help: "steering: doses of the bundle's directions, e.g. delivery.tempo=2 (faster), mood.valence=3, emotions.happy=4, delivery.whisper=3; identity.*, pitch.*, accents_*, texture.* and room.* move the speaker embedding (speaker-mode embedding or a recording)", at: At::Request },
    EngineOption { name: "speaking-rate", env: "", value: "BYTES/S", help: "the speaking-rate condition: UTF-8 bytes of text a second (about 14 is ordinary English; unset: the model's choice)", at: At::Request },
    EngineOption { name: "loudness", env: "", value: "LUFS", help: "the loudness condition (-30 to -15; unset: unknown)", at: At::Request },
    EngineOption { name: "fmax", env: "", value: "HZ", help: "the bandwidth condition (8000 to 20000; unset: unknown)", at: At::Request },
    EngineOption { name: "decoder", env: "", value: "keep2|full", help: "the codec decoder: keep2 (the default: two layers a stage) or full (the full-size head, when its file is there)", at: At::Request },
    EngineOption { name: "temperature", env: "", value: "0.75", help: "sampling temperature", at: At::Request },
    EngineOption { name: "top-k", env: "", value: "80", help: "sampling top-k", at: At::Request },
    EngineOption { name: "top-p", env: "", value: "0.69", help: "sampling top-p", at: At::Request },
    EngineOption { name: "repetition-penalty", env: "", value: "2.4", help: "the repetition penalty over the last 96 frames of the first six codebooks", at: At::Request },
    EngineOption { name: "silence", env: "", value: "0..30", help: "codec-silence frames after the text (3: 0.2 s, decoded ahead and cut)", at: At::Request },
];

pub fn kind() -> AudioKind {
    AudioKind { archs: &[ARCH], name: "ZONOS3-Mini (multilingual speech at 44.1 kHz: preset voices, voices cloned from a recording, steering; Apache-2.0)", roles: ROLES,
                load, options: OPTIONS }
}

fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn AudioEngine>> {
    let gpu = gpus.first().ok_or_else(|| Error("no GPU".into()))?;
    Ok(Box::new(Zonos3::load(files, gpu, o, log)?))
}

/// Zyphra's gguf/ folder as downloaded (backbone-*.gguf, zac-keep2.gguf, zac.gguf, spk-encoder.gguf, steer.gguf,
/// voices/<Language>/<Name>_*.codes.npy / .embedding.npy): its files by role
pub fn files_in(dir: &Path) -> Result<ModelFiles> {
    let mut f = ModelFiles::new();
    let model = ["backbone-ptq.gguf", "backbone-f16.gguf", "backbone-q8_0.gguf"].iter().map(|n| dir.join(n)).find(|p| p.is_file())
        .ok_or_else(|| Error(format!("{}: no backbone-ptq.gguf", dir.display())))?;
    f.insert("model".into(), model);
    for (role, name) in [("decoder", "zac-keep2.gguf"), ("decoder-full", "zac.gguf"), ("speaker-encoder", "spk-encoder.gguf"), ("steer", "steer.gguf")] {
        if dir.join(name).is_file() {
            f.insert(role.into(), dir.join(name));
        }
    }
    if !f.contains_key("decoder") {
        return Err(Error(format!("{}: no zac-keep2.gguf", dir.display())));
    }
    let mut stack = vec![dir.join("voices")];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
                continue;
            }
            let n = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            let (stem, kind) = match (n.strip_suffix(".codes.npy"), n.strip_suffix(".embedding.npy")) {
                (Some(s), _) => (s, ""),
                (_, Some(s)) => (s, "-emb"),
                _ => continue,
            };
            let name = stem.split('_').next().unwrap_or(stem).to_lowercase();
            f.insert(format!("voice-{name}{kind}"), p.clone());
        }
    }
    Ok(f)
}

/// A preset voice's files
#[derive(Clone, Debug, Default)]
pub struct VoiceFiles {
    pub codes: Option<PathBuf>,
    pub embedding: Option<PathBuf>,
}

pub struct Zonos3 {
    pub ops: Ops,
    pub lm: Lm,
    pub tok: Tokenizer,
    /// the default decoder first
    pub decoders: Vec<Decoder>,
    /// the full-size decoder's file (zac.gguf), loaded on first use: its matrices take 4 GiB on the card
    full_path: Option<PathBuf>,
    full: std::sync::OnceLock<Decoder>,
    pub spk: Option<SpeakerEncoder>,
    pub steer: Option<Steer>,
    pub voices: BTreeMap<String, VoiceFiles>,
    /// one request at a time on the card
    lock: Mutex<()>,
    load_s: f64,
}

/// A request's own option
fn given<'a>(req: &'a AudioRequest, name: &str) -> Option<&'a str> {
    req.extra.get(name).map(String::as_str).filter(|s| !s.trim().is_empty())
}

fn num<T: std::str::FromStr>(req: &AudioRequest, name: &str) -> Result<Option<T>> {
    given(req, name).map(|v| v.trim().parse::<T>().map_err(|_| Error(format!("option {name}: {v:?} is not a number")))).transpose()
}

/// The stretches a long text is said in: whole sentences, each at most `max` characters (a sentence longer than that
/// is cut at its last comma, else its last space, before the limit)
pub fn stretches(text: &str, max: usize) -> Vec<String> {
    let text = text.trim();
    if text.chars().count() <= max {
        return vec![text.to_string()];
    }
    let mut sentences = Vec::new();
    let mut cur = String::new();
    let chars: Vec<char> = text.chars().collect();
    for (i, c) in chars.iter().enumerate() {
        cur.push(*c);
        let end = matches!(c, '.' | '!' | '?' | '\n' | '。' | '！' | '？' | '…');
        if end && chars.get(i + 1).is_none_or(|n| n.is_whitespace() || !c.is_ascii()) {
            sentences.push(std::mem::take(&mut cur));
        }
    }
    if !cur.trim().is_empty() {
        sentences.push(cur);
    }
    let mut out: Vec<String> = Vec::new();
    for s in sentences {
        let mut s = s.trim().to_string();
        while s.chars().count() > max {
            let head: String = s.chars().take(max).collect();
            let cut = head.rfind([',', ';', ':']).or_else(|| head.rfind(' ')).filter(|i| *i > 0).map_or(head.len(), |i| i + 1);
            out.push(s[..cut].trim().to_string());
            s = s[cut..].trim().to_string();
        }
        if s.is_empty() {
            continue;
        }
        match out.last_mut() {
            Some(last) if last.chars().count() + 1 + s.chars().count() <= max => {
                last.push(' ');
                last.push_str(&s);
            }
            _ => out.push(s),
        }
    }
    out
}

/// The output stage: the decoder's headroom as gain, then the soft limiter above -1 dBFS
pub fn output(pcm: &mut [f32], headroom_db: f32) {
    let g = 10f32.powf(-headroom_db / 20.0);
    let w = 1.0 - KNEE;
    for x in pcm.iter_mut() {
        let v = *x * g;
        *x = if v.abs() > KNEE { v.signum() * (KNEE + w * ((v.abs() - KNEE) / w).tanh()) } else { v };
    }
}

/// One stretch's generation: the rows drawn (each ten codes) and the aligned end frame
pub struct Take {
    pub rows: Vec<[u32; 10]>,
    pub eos: Option<usize>,
}

impl Zonos3 {
    pub fn load(files: &ModelFiles, gpu: &Arc<Gpu>, _o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Zonos3> {
        let t0 = Instant::now();
        nextcuda_core::use_kind("audio");
        let role = |r: &str| files.get(r).cloned().ok_or_else(|| Error(format!("{ARCH}: no {r} file")));
        let size = |p: &Path| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
        let (model, dec) = (role("model")?, role("decoder")?);
        // float32 mixers and head (~4x their stored size), the decoder's tables and matrices (float32), a long take's
        // activations
        let need = size(&model) / 2 + 600 * (1 << 20) + size(&dec) * 2 + (1u64 << 30);
        let (total, free) = gpu.memory().map_err(ops::e)?;
        let free = free.unwrap_or(total);
        if need > free {
            return Err(Error(format!("{ARCH} needs about {:.1} GiB on {}; {:.1} GiB are free", need as f64 / (1u64 << 30) as f64, gpu.name,
                                     free as f64 / (1u64 << 30) as f64)));
        }
        let ops = Ops::new(gpu)?;
        let w = Weights::open(&model)?;
        let lm = Lm::load(&ops, &w, log)?;
        let tok = Tokenizer::from_hf_json_text_keep_gaps(&lm.c.tokenizer).map_err(Error)?;
        let decoders = vec![Decoder::load(&ops, &Weights::open(&dec)?, log)?];
        let spk = match files.get("speaker-encoder") {
            Some(p) => Some(SpeakerEncoder::load(&ops, &Weights::open(p)?, log)?),
            None => None,
        };
        let steer = match files.get("steer") {
            Some(p) => Some(Steer::load(&Weights::open(p)?)?),
            None => None,
        };
        let mut voices: BTreeMap<String, VoiceFiles> = BTreeMap::new();
        for (r, p) in files {
            let Some(name) = r.strip_prefix("voice-") else { continue };
            match name.strip_suffix("-emb") {
                Some(n) => voices.entry(n.to_string()).or_default().embedding = Some(p.clone()),
                None => voices.entry(name.to_string()).or_default().codes = Some(p.clone()),
            }
        }
        gpu.sync().map_err(ops::e)?;
        let load_s = t0.elapsed().as_secs_f64();
        log(format!("{ARCH} on {} in {load_s:.1} s: {} voices, decoder {}{}, speaker encoder {}, steering {}", gpu.name, voices.len(), decoders[0].variant,
                    if files.contains_key("decoder-full") { " (full_mr on request)" } else { "" },
                    if spk.is_some() { "yes" } else { "no" }, if steer.is_some() { "yes" } else { "no" }));
        Ok(Zonos3 { ops, lm, tok, decoders, full_path: files.get("decoder-full").cloned(), full: std::sync::OnceLock::new(), spk, steer, voices,
                    lock: Mutex::new(()), load_s })
    }

    pub fn decoder(&self, req: &AudioRequest) -> Result<&Decoder> {
        match given(req, "decoder") {
            None => Ok(&self.decoders[0]),
            Some(d) => {
                let want = match d.trim() {
                    "keep2" | "keep2_plain" => "keep2_plain",
                    "full" | "full_mr" => "full_mr",
                    x => return Err(Error(format!("decoder {x:?}: keep2 or full"))),
                };
                if let Some(d) = self.decoders.iter().find(|x| x.variant == want) {
                    return Ok(d);
                }
                if want == "full_mr" {
                    if let Some(d) = self.full.get() {
                        return Ok(d);
                    }
                    let p = self.full_path.as_ref().ok_or_else(|| Error("decoder full: this model has no zac.gguf".into()))?;
                    let d = Decoder::load(&self.ops, &Weights::open(p)?, &mut |m| eprintln!("{m}"))?;
                    let _ = self.full.set(d);
                    return self.full.get().ok_or_else(|| Error("decoder full: not loaded".into()));
                }
                Err(Error(format!("decoder {want} is not loaded")))
            }
        }
    }

    /// The request's voice: a recording (its embedding), a preset (its codes, or its embedding), or the default preset
    pub fn voice(&self, req: &AudioRequest, embedding_mode: bool) -> Result<Voice> {
        if let Some(r) = &req.reference {
            let spk = self.spk.as_ref().ok_or_else(|| Error(format!("{ARCH}: cloning a recording needs the speaker encoder (spk-encoder.gguf)")))?;
            if r.samples.len() < r.rate.max(1) as usize / 2 {
                return Err(Error("the recording is shorter than half a second".into()));
            }
            return Ok(Voice::Embedding(spk.embed(&self.ops, &r.samples, r.rate)?));
        }
        let name = match req.voice.as_deref().map(str::trim).filter(|v| !v.is_empty()) {
            Some(v) => v.to_lowercase(),
            None if self.voices.is_empty() => return Ok(Voice::None),
            None => ["savannah", "wyatt"].iter().find(|n| self.voices.contains_key(**n)).map(|s| s.to_string())
                .unwrap_or_else(|| self.voices.keys().next().cloned().unwrap_or_default()),
        };
        if name == "none" {
            return Ok(Voice::None);
        }
        let v = self.voices.get(&name).ok_or_else(|| Error(format!("no voice {name:?} (this model's: {})", self.voices.keys().cloned().collect::<Vec<_>>().join(", "))))?;
        let read = |p: &PathBuf| std::fs::read(p).map_err(|x| Error(format!("{}: {x}", p.display())));
        match (embedding_mode, &v.codes, &v.embedding) {
            (false, Some(c), _) => Ok(Voice::Reference(prompt::read_codes(&read(c)?)?)),
            (_, _, Some(e)) => Ok(Voice::Embedding(prompt::read_embedding(&read(e)?)?)),
            (true, Some(_), None) => Err(Error(format!("voice {name} has no embedding file: speaker-mode reference"))),
            _ => Err(Error(format!("voice {name} has no files"))),
        }
    }

    /// One stretch: the prompt's rows through the backbone, then a row of codes a step until the end frame drains
    #[allow(clippy::too_many_arguments)]
    pub fn take(&self, text: &str, voice: &Voice, buckets: &[Option<usize>], deltas: &[(usize, Vec<f32>)], cfg: f32, s: &Sampling, silence: usize,
                max_frames: usize, rng: &mut Rng, progress: &mut dyn FnMut(usize) -> Result<()>) -> Result<(Take, Vec<[u32; 10]>)> {
        let c = &self.lm.c;
        let ids = prompt::text_ids(&self.tok, c, text);
        let p = prompt::build(c, &ids, voice, buckets, silence)?;
        let cond = Cond {
            speaker: p.speaker.as_ref().map(|(r, v)| Ok::<_, Error>((*r, self.lm.project_speaker(v)?))).transpose()?,
            buckets: p.buckets.clone(),
            deltas: Vec::new(),
        };
        let cap = p.rows + max_frames + 2;
        let mut seq = self.lm.seq(&self.ops, cap, p.rows)?;
        self.lm.forward(&self.ops, &mut seq, &p.ids, &cond, false)?;
        // speaker CFG: the same text without the voice, every condition unknown
        let guided = cfg != 1.0 && !matches!(voice, Voice::None);
        let mut twin = None;
        if guided {
            let tp = prompt::build(c, &ids, &Voice::None, &[], silence)?;
            let tc = Cond { speaker: None, buckets: tp.buckets.clone(), deltas: Vec::new() };
            let mut ts = self.lm.seq(&self.ops, tp.rows + max_frames + 2, tp.rows)?;
            self.lm.forward(&self.ops, &mut ts, &tp.ids, &tc, false)?;
            twin = Some((ts, tc));
        }
        let step_cond = Cond { speaker: None, buckets: Vec::new(), deltas: deltas.to_vec() };
        let mut rows: Vec<[u32; 10]> = Vec::new();
        let (mut eos, mut countdown) = (None::<usize>, 0usize);
        let ncb = c.ncb;
        loop {
            let mut logits = self.lm.logits(&seq, 1)?;
            if let Some((ts, _)) = &twin {
                let u = self.lm.logits(ts, 1)?;
                for (l, u) in logits.iter_mut().zip(&u) {
                    *l = u + cfg * (*l - u);
                }
            }
            let total = rows.len();
            // the cold start: a codebook whose delayed stream has not begun is padding (the stream row counts the
            // silence rows; a prompt that ends in a recording's rows has begun)
            let stream_row = if p.audio_prefix && p.prefill.is_empty() { usize::MAX } else { p.prefill.len() + total };
            let r = sample::row(&logits, ncb, s, c.codebook, c.pad, c.eoa, &rows, stream_row, rng);
            rows.push(r);
            if eos.is_none() {
                if let Some(mc) = (0..ncb).rev().find(|cb| r[*cb] == c.eoa) {
                    eos = Some(total.saturating_sub(mc));
                    countdown = ncb + 1;
                }
            }
            let mut finished = rows.len() >= max_frames;
            if countdown > 0 {
                countdown -= 1;
                if countdown == 0 {
                    finished = true;
                }
            }
            if rows.len() % 15 == 0 || finished {
                progress(rows.len())?;
            }
            if finished {
                break;
            }
            let ids: Vec<u32> = r[..ncb].iter().copied().chain(std::iter::once(c.text_vocab)).collect();
            self.lm.forward(&self.ops, &mut seq, &ids, &step_cond, false)?;
            if let Some((ts, _)) = twin.as_mut() {
                self.lm.forward(&self.ops, ts, &ids, &step_cond, false)?;
            }
        }
        Ok((Take { rows, eos }, p.prefill))
    }

    /// A take's rows to PCM: undelayed (frame f's codebook c on row f + c), the complete frames before the end, the
    /// silence frames decoded ahead and their samples cut
    pub fn decode_take(&self, dec: &Decoder, take: &Take, prefill: &[[u32; 10]]) -> Result<Vec<f32>> {
        let c = &self.lm.c;
        let ncb = c.ncb;
        let buffer: Vec<[u32; 10]> = prefill.iter().chain(&take.rows).copied().collect();
        let mut target = buffer.len().saturating_sub(ncb - 1);
        if let Some(e) = take.eos {
            target = target.min(e + prefill.len());
        }
        for (i, r) in buffer.iter().enumerate() {
            if let Some(mc) = (0..ncb).rev().find(|cb| r[*cb] == c.eoa) {
                target = target.min(i.saturating_sub(mc));
            }
        }
        if target <= prefill.len() {
            return Ok(Vec::new());
        }
        let codes: Vec<[u32; 10]> = (0..target).map(|f| {
            let mut fr = [0u32; 10];
            for (cb, x) in fr.iter_mut().enumerate().take(ncb) {
                let v = buffer.get(f + cb).map_or(c.pad, |r| r[cb]);
                *x = if (v as usize) < c.codebook { v } else { 0 };
            }
            fr
        }).collect();
        let pcm = dec.decode(&self.ops, &codes)?;
        Ok(pcm[(prefill.len() * dec.hop).min(pcm.len())..].to_vec())
    }
}

impl AudioEngine for Zonos3 {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn defaults(&self) -> Defaults {
        Defaults { seconds: 60.0, max_seconds: 600.0, steps: 0, cfg: 1.0, rate: RATE }
    }
    fn guidance(&self) -> bool {
        true
    }
    fn speech(&self) -> Option<Speech> {
        Some(Speech {
            voices: self.voices.keys().cloned().collect(),
            languages: ["en", "fr", "es", "de", "it", "pt", "nl", "pl", "ru", "sv", "tr", "uk", "cs", "da", "fi", "el", "hu", "no", "ro", "sk", "bg", "zh",
                        "ja", "ko", "ar", "he", "vi", "id", "th", "tl", "ms", "ka", "hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "or", "ur"].iter().map(|s| s.to_string()).collect(),
            instructions: false,
            design: false,
            clone: self.spk.is_some(),
            clone_needs_text: false,
            clone_takes_text: false,
            direction: false,
            events: Vec::new(),
            requires: None,
            speakers: 0,
        })
    }
    fn options(&self) -> &'static [EngineOption] {
        OPTIONS
    }
    fn load_seconds(&self) -> f64 {
        self.load_s
    }

    fn generate(&self, req: &AudioRequest, progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<Audio> {
        let _one = self.lock.lock().unwrap_or_else(|p| p.into_inner());
        let t0 = Instant::now();
        let text = req.prompt.trim();
        if text.is_empty() {
            return Err(Error("no text to say".into()));
        }
        let c = &self.lm.c;
        let embedding_mode = match given(req, "speaker-mode") {
            None | Some("reference") => false,
            Some("embedding") => true,
            Some(m) => return Err(Error(format!("speaker-mode {m:?}: reference or embedding"))),
        };
        let controls = given(req, "controls").map(steer::parse_controls).transpose()?.unwrap_or_default();
        let raw_controls = controls.iter().any(|(id, d)| *d != 0.0 && self.steer.as_ref().is_some_and(|s| s.dirs.iter().any(|x| &x.id == id && x.source.is_none())));
        progress(Step { phase: "voice", at: 0, of: 1, seconds: 0.0 })?;
        let mut voice = self.voice(req, embedding_mode || raw_controls)?;
        let mut deltas = Vec::new();
        if !controls.is_empty() {
            let st = self.steer.as_ref().ok_or_else(|| Error(format!("{ARCH}: controls need the steering bundle (steer.gguf)")))?;
            let emb = match &voice {
                Voice::Embedding(v) => Some(v.as_slice()),
                _ => None,
            };
            let r = st.resolve(&controls, emb, c.dim)?;
            if let Some(e) = r.embedding {
                voice = Voice::Embedding(e);
            }
            deltas = r.deltas;
        }
        let s = Sampling {
            temperature: num(req, "temperature")?.unwrap_or(0.75),
            topk: num(req, "top-k")?.unwrap_or(80),
            top_p: num(req, "top-p")?.unwrap_or(0.69),
            repetition_penalty: num(req, "repetition-penalty")?.unwrap_or(2.4),
            ..Sampling::default()
        };
        let silence: usize = num(req, "silence")?.unwrap_or(3);
        if silence > 30 {
            return Err(Error("silence: 0 to 30 frames".into()));
        }
        let mut buckets: Vec<Option<usize>> = vec![None; c.cond_names.len()];
        for (i, name) in c.cond_names.iter().enumerate() {
            let opt = match name.as_str() {
                "speaking_rate" => "speaking-rate",
                n => n,
            };
            if let Some(v) = num::<f64>(req, opt)? {
                buckets[i] = Some(prompt::bucket(&c.cond_edges[i], v));
            }
        }
        let cfg = req.cfg.unwrap_or(1.0);
        if !cfg.is_finite() || cfg < 0.0 {
            return Err(Error(format!("cfg {cfg}: a scale of 0 or more (1: none)")));
        }
        let dec = self.decoder(req)?;
        // `seconds` caps the whole when given; without it a long text is said to its end (600 s at most)
        let seconds = req.seconds.unwrap_or(self.defaults().max_seconds).clamp(1.0, self.defaults().max_seconds);
        let parts = stretches(text, MAX_CHARS);
        let mut rng = Rng::new(req.seed);
        let mut out = Vec::new();
        let mut frames_done = 0usize;
        let budget = (seconds * FPS) as usize;
        for (i, part) in parts.iter().enumerate() {
            let left = budget.saturating_sub(frames_done);
            if left < 15 {
                break;
            }
            // a take that never ends (no end-of-audio drawn) stops at four rows a byte of its text and a few
            // seconds more: the slowest steered pace says about three
            let max_frames = left.min(MAX_FRAMES).min(4 * part.len() + 90);
            let (take, prefill) = self.take(part, &voice, &buckets, &deltas, cfg, &s, silence, max_frames, &mut rng, &mut |n| {
                progress(Step { phase: "tokens", at: (frames_done + n) as u32, of: budget as u32, seconds: t0.elapsed().as_secs_f64() })
            })?;
            frames_done += take.rows.len();
            let t_gen = t0.elapsed().as_secs_f64();
            progress(Step { phase: "decode", at: i as u32, of: parts.len() as u32, seconds: t_gen })?;
            let mut pcm = self.decode_take(dec, &take, &prefill)?;
            eprintln!("{ARCH}: stretch {}/{}: {} rows ({:.1} s of sound), generated by {:.2} s, decoded ({}) by {:.2} s", i + 1, parts.len(), take.rows.len(),
                      pcm.len() as f64 / RATE as f64, t_gen, dec.variant, t0.elapsed().as_secs_f64());
            output(&mut pcm, dec.headroom_db);
            if !out.is_empty() {
                out.extend(std::iter::repeat_n(0.0f32, (GAP * RATE as f32) as usize));
            }
            out.extend(pcm);
        }
        progress(Step { phase: "decode", at: parts.len() as u32, of: parts.len() as u32, seconds: t0.elapsed().as_secs_f64() })?;
        Ok(Audio { rate: dec.rate, channels: 1, samples: out })
    }

    fn report(&self) -> Vec<String> {
        vec![format!("{ARCH} on {}: backbone {:.0} MiB, decoders {}, {} voices", self.ops.gpu.name, self.lm.bytes as f64 / (1u64 << 20) as f64,
                     self.decoders.iter().map(|d| d.variant.clone()).collect::<Vec<_>>().join(", "), self.voices.len())]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_texts_are_said_in_sentences() {
        assert_eq!(stretches("  Short one. ", 400), vec!["Short one.".to_string()]);
        let t = "First sentence here. Second one is a bit longer! Third? Fourth.";
        let s = stretches(t, 30);
        assert!(s.iter().all(|x| x.chars().count() <= 30), "{s:?}");
        assert_eq!(s.join(" "), t);
        let long = "word ".repeat(50);
        let s = stretches(&long, 40);
        assert!(s.iter().all(|x| x.chars().count() <= 40 && !x.is_empty()));
    }

    #[test]
    fn the_limiter_bends_only_above_the_knee() {
        let mut v = vec![0.5, -0.5, 1.5, -3.0];
        output(&mut v, 0.0);
        assert_eq!(v[0], 0.5);
        assert_eq!(v[1], -0.5);
        assert!(v[2] > KNEE && v[2] < 1.0);
        assert!(v[3] < -KNEE && v[3] >= -1.0);
        let mut g = vec![1.0];
        output(&mut g, 3.0);
        assert!((g[0] - 0.70794576).abs() < 1e-5);
    }

    #[test]
    fn a_folder_of_ggufs_is_found() {
        let d = std::env::temp_dir().join(format!("z3files-{}", std::process::id()));
        std::fs::create_dir_all(d.join("voices/English")).unwrap();
        for f in ["backbone-ptq.gguf", "zac-keep2.gguf", "spk-encoder.gguf", "voices/English/Savannah_en-American_F_1.codes.npy",
                  "voices/English/Savannah_en-American_F_1.embedding.npy"] {
            std::fs::write(d.join(f), b"x").unwrap();
        }
        let f = files_in(&d).unwrap();
        std::fs::remove_dir_all(&d).ok();
        assert!(f.contains_key("model") && f.contains_key("decoder") && f.contains_key("speaker-encoder"));
        assert!(!f.contains_key("decoder-full"));
        assert!(f["voice-savannah"].ends_with("Savannah_en-American_F_1.codes.npy"));
        assert!(f["voice-savannah-emb"].ends_with("Savannah_en-American_F_1.embedding.npy"));
    }
}
