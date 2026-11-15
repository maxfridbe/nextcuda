//! Stage checks against the reference's dumps (`nextcuda audio check DIR --model <zonos3 model> --stages ...`): the
//! engine runs the stage on the reference's inputs in DIR and writes its outputs beside them as .npy, for a
//! comparison script to read (the reference: Zyphra's zonos3-cli / zac-cli / spk-encoder-cli on the same GGUFs).
//!
//! ```text
//!   lm        ids.npy [rows, 11] + ids.cond.json (zonos3-cli --build-prompt)   -> nc_logits.npy [rows, 10, 4098]
//!   forced    ids.npy + ids.cond.json + rows.npy [steps, 11] (generated rows)   -> nc_forced.npy [1 + steps, 10, 4098]:
//!             the prompt's last row, then each row fed back (teacher forcing)
//!   decoder   codes.npy [frames, 10]                                            -> nc_pcm.npy (44.1 kHz, before the gain)
//!   speaker   speaker.wav (any rate; the reference's input: 44.1 kHz)           -> nc_embedding.npy [192], nc_fbank.npy
//!   encoder   reference.wav (any rate)                                        -> nc_codes.npy [frames, 10] (raw-reference codes)
//!   stream    prompt.txt (+ options.json)                                      -> nc_whole.npy, nc_stream.npy (the same take whole and streamed)
//!   prompt    prompt.txt (+ voice.codes.npy | voice.embedding.npy)              -> nc_ids.npy [rows, 11]
//! ```

use std::path::Path;

use nextcuda_audio::{Error, Result};

use crate::lm::Cond;
use crate::prompt::{self, Voice};
use crate::Zonos3;

pub const STAGES: &[&str] = &["lm", "forced", "decoder", "speaker", "encoder", "stream", "prompt"];

/// A float32 .npy of `shape`
pub fn npy(path: &Path, shape: &[usize], v: &[f32]) -> Result<()> {
    let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
    let mut head = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({}{}), }}", dims.join(", "), if shape.len() == 1 { "," } else { "" });
    while (10 + head.len() + 1) % 64 != 0 {
        head.push(' ');
    }
    head.push('\n');
    let mut b = b"\x93NUMPY\x01\x00".to_vec();
    b.extend((head.len() as u16).to_le_bytes());
    b.extend(head.as_bytes());
    for x in v {
        b.extend(x.to_le_bytes());
    }
    std::fs::write(path, b).map_err(|e| Error(format!("{}: {e}", path.display())))
}

fn read(dir: &Path, name: &str) -> Result<Vec<u8>> {
    std::fs::read(dir.join(name)).map_err(|e| Error(format!("{}: {e}", dir.join(name).display())))
}

/// The reference's prompt (ids.npy, float32 or integer [rows, 11]) and its conditioning (ids.cond.json)
fn reference_prompt(z: &Zonos3, dir: &Path) -> Result<(Vec<u32>, usize, Cond)> {
    let (shape, v) = prompt::read_npy(&read(dir, "ids.npy")?)?;
    let w = z.lm.c.row_width();
    if shape.len() != 2 || shape[1] != w {
        return Err(Error(format!("ids.npy is {shape:?}; [rows, {w}] expected")));
    }
    let ids: Vec<u32> = v.iter().map(|x| *x as u32).collect();
    let j: serde_json::Value = serde_json::from_slice(&read(dir, "ids.cond.json")?).map_err(|e| Error(format!("ids.cond.json: {e}")))?;
    let mut cond = Cond::default();
    let srow = j["speaker_row"].as_i64().unwrap_or(-1);
    if srow >= 0 {
        let v: Vec<f32> = j["speaker_embedding"].as_array().ok_or_else(|| Error("ids.cond.json: a speaker row without its embedding".into()))?
            .iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect();
        cond.speaker = Some((srow as usize, z.lm.project_speaker(&v)?));
    }
    let crow = j["condition_row"].as_u64().unwrap_or(0) as usize;
    for (i, b) in j["condition_buckets"].as_array().into_iter().flatten().enumerate() {
        cond.buckets.push((crow + i, b.as_u64().unwrap_or(0) as usize));
    }
    Ok((ids, shape[0], cond))
}

pub fn run(z: &Zonos3, dir: &Path, stages: &[String], log: &mut dyn FnMut(String)) -> Result<()> {
    let c = &z.lm.c;
    for st in stages {
        match st.as_str() {
            "lm" => {
                let (ids, rows, cond) = reference_prompt(z, dir)?;
                let mut seq = z.lm.seq(&z.ops, rows + 1, rows)?;
                let m = z.lm.forward(&z.ops, &mut seq, &ids, &cond, true)?;
                let l = z.lm.logits(&seq, m)?;
                npy(&dir.join("nc_logits.npy"), &[m, c.ncb, c.audio_vocab], &l)?;
                log(format!("lm: {rows} rows -> nc_logits.npy"));
            }
            "forced" => {
                let (ids, rows, cond) = reference_prompt(z, dir)?;
                let (shape, gen) = prompt::read_npy(&read(dir, "rows.npy")?)?;
                let w = c.row_width();
                let steps = if shape.len() == 2 { shape[0] } else { 0 };
                let gen: Vec<u32> = gen.iter().map(|x| *x as u32).collect();
                let mut seq = z.lm.seq(&z.ops, rows + steps + 1, rows)?;
                // the alignment probe of every step when probe_heads.json is here (LACI's heads): nc_probe.npy
                let heads = std::fs::read_to_string(dir.join("probe_heads.json")).ok().map(|t| crate::laci::Heads::parse(&t, "probe_heads.json")).transpose()?;
                if let Some(h) = &heads {
                    let col: Vec<u32> = (0..rows).map(|r| ids[r * c.row_width() + c.ncb]).collect();
                    let st = col.iter().position(|t| *t == c.bos).ok_or_else(|| Error("no BOS row".into()))?;
                    let en = st + 1 + col[st + 1..].iter().position(|t| *t == c.eos).ok_or_else(|| Error("no EOS row".into()))?;
                    seq.probe = Some(crate::lm::Probe::new(st, en - st + 1, h.heads.clone()));
                }
                let mut probe_out: Vec<f32> = Vec::new();
                z.lm.forward(&z.ops, &mut seq, &ids, &cond, false)?;
                let mut out = z.lm.logits(&seq, 1)?;
                let step = Cond::default();
                for s in 0..steps {
                    let mut r: Vec<u32> = gen[s * shape[1]..s * shape[1] + c.ncb].to_vec();
                    r.push(c.text_vocab);
                    debug_assert_eq!(r.len(), w);
                    z.lm.forward(&z.ops, &mut seq, &r, &step, false)?;
                    out.extend(z.lm.logits(&seq, 1)?);
                    if let Some((p, m)) = seq.probe.as_ref().and_then(|pr| pr.readout) {
                        probe_out.extend([p as f32, m as f32]);
                    }
                }
                if heads.is_some() {
                    npy(&dir.join("nc_probe.npy"), &[probe_out.len() / 2, 2], &probe_out)?;
                }
                npy(&dir.join("nc_forced.npy"), &[1 + steps, c.ncb, c.audio_vocab], &out)?;
                log(format!("forced: {rows} prompt rows + {steps} rows -> nc_forced.npy"));
            }
            "decoder" => {
                let codes = prompt::read_codes(&read(dir, "codes.npy")?)?;
                let pcm = z.decoders[0].decode(&z.ops, &codes)?;
                npy(&dir.join("nc_pcm.npy"), &[pcm.len()], &pcm)?;
                log(format!("decoder ({}): {} frames -> nc_pcm.npy ({} samples)", z.decoders[0].variant, codes.len(), pcm.len()));
            }
            "speaker" => {
                let spk = z.spk.as_ref().ok_or_else(|| Error("no speaker encoder".into()))?;
                let (samples, rate) = nextcuda_audio::decode_audio(&read(dir, "speaker.wav")?).map_err(Error)?;
                let e = spk.embed(&z.ops, &samples, rate)?;
                npy(&dir.join("nc_embedding.npy"), &[e.len()], &e)?;
                log(format!("speaker: {} samples at {rate} Hz -> nc_embedding.npy (norm {:.3})", samples.len(), e.iter().map(|x| x * x).sum::<f32>().sqrt()));
            }
            "encoder" => {
                let (samples, rate) = nextcuda_audio::decode_audio(&read(dir, "reference.wav")?).map_err(Error)?;
                let codes = z.reference_codes(&samples, rate, log)?;
                let f: Vec<f32> = codes.iter().flat_map(|r| r.iter().map(|x| *x as f32)).collect();
                npy(&dir.join("nc_codes.npy"), &[codes.len(), 10], &f)?;
                log(format!("encoder: {} samples at {rate} Hz -> nc_codes.npy ({} frames)", samples.len(), codes.len()));
            }
            "stream" => {
                use nextcuda_audio::AudioEngine;
                let text = String::from_utf8_lossy(&read(dir, "prompt.txt")?).trim().to_string();
                let mut req = nextcuda_audio::AudioRequest { prompt: text, seed: 7, voice: Some("rupert".into()), ..Default::default() };
                if let Ok(b) = read(dir, "options.json") {
                    let v: serde_json::Value = serde_json::from_slice(&b).map_err(|x| Error(x.to_string()))?;
                    for (k, x) in v.as_object().into_iter().flatten() {
                        req.extra.insert(k.clone(), x.as_str().map_or_else(|| x.to_string(), str::to_string));
                    }
                }
                let t0 = std::time::Instant::now();
                let whole = z.generate(&req, &mut |_| Ok(()))?;
                let tw = t0.elapsed().as_secs_f64();
                let t1 = std::time::Instant::now();
                let mut first: Option<f64> = None;
                let mut pieces = 0usize;
                let streamed = z.generate_stream(&req, &mut |_| Ok(()), &mut |a| {
                    first.get_or_insert(t1.elapsed().as_secs_f64());
                    pieces += 1;
                    let _ = a;
                    Ok(())
                })?;
                npy(&dir.join("nc_whole.npy"), &[whole.samples.len()], &whole.samples)?;
                npy(&dir.join("nc_stream.npy"), &[streamed.samples.len()], &streamed.samples)?;
                log(format!("stream: whole {} samples in {tw:.2} s; streamed {} samples in {pieces} pieces, the first after {:.2} s, all by {:.2} s",
                            whole.samples.len(), streamed.samples.len(), first.unwrap_or(0.0), t1.elapsed().as_secs_f64()));
            }
            "prompt" => {
                let text = String::from_utf8_lossy(&read(dir, "prompt.txt")?).trim().to_string();
                let voice = if let Ok(b) = read(dir, "voice.codes.npy") {
                    Voice::Reference(prompt::read_codes(&b)?)
                } else if let Ok(b) = read(dir, "voice.embedding.npy") {
                    Voice::Embedding(prompt::read_embedding(&b)?)
                } else {
                    Voice::None
                };
                // the engine's text path: the markup, the normalizer when normalize.txt names a language, the markers
                let parsed = crate::markup::parse(&text)?;
                let lang = std::fs::read_to_string(dir.join("normalize.txt")).ok().map(|l| l.trim().to_string());
                let normed = crate::markup::normalize(&parsed.text, lang.as_deref().unwrap_or("none"));
                let (clean, _) = crate::markup::take_spans(&normed, &parsed.spans)?;
                std::fs::write(dir.join("nc_text.txt"), clean.as_bytes()).map_err(|e| Error(e.to_string()))?;
                let tids = crate::markup::text_ids(&z.tok, c, &crate::markup::Spec::read(&c.spec), &clean)?;
                let p = prompt::build(c, &tids, &voice, &[], 3)?;
                let f: Vec<f32> = p.ids.iter().map(|x| *x as f32).collect();
                npy(&dir.join("nc_ids.npy"), &[p.rows, c.row_width()], &f)?;
                log(format!("prompt: {} text ids, {} rows -> nc_ids.npy", tids.len(), p.rows));
            }
            s => return Err(Error(format!("no stage {s} (the stages: {})", STAGES.join(", ")))),
        }
    }
    Ok(())
}
