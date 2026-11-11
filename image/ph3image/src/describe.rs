//! image.describe's native describer for P-H3-Image: its own text encoder - Qwen3-VL-4B whole in the encoder GGUF
//! (`model.language_model.*`: 36 layers, the final norm, the embedding the head is tied to; `model.visual.*`: the
//! 24-block vision tower and its deepstack mergers, bf16), the tokenizer from the card's tokenizer files - reading the
//! picture and writing the prompt (crates/qwen3vl `Reader`). The image server loads it in place of the pipeline.
//!
//! Memory: 4.0 GiB of Q8_0 weights (the 0.4 GiB tied embedding counted once, as the head), 0.8 GiB of the vision
//! tower in half, the cache (36 layers x 8 key heads x 128 x 2 x half: 144 KiB a row).

use std::path::Path;
use std::sync::Arc;

use nextcuda_core::Gpu;
use nextcuda_image::{DescribeOutput, DescribeRequest, Describer, Error, ModelFiles, Result};
use nextcuda_qwen3vl::Reader;
use nextcuda_tok::Tokenizer;

use crate::ARCH;

pub struct ReaderDescriber(pub Reader);

/// The describer of a P-H3-Image card: its text encoder (with its vision tower) onto the first of its cards
pub fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], log: &mut dyn FnMut(String)) -> Result<Box<dyn Describer>> {
    let gpu = gpus.first().ok_or_else(|| Error("no GPU".into()))?;
    let te = files.get("text-encoder").ok_or_else(|| Error(format!("{ARCH}: no text-encoder file")))?;
    // vocab.json's folder holds merges.txt and tokenizer_config.json beside it (the card's tokenizer roles)
    let vocab = files.get("tokenizer").ok_or_else(|| Error(format!("{ARCH}: no tokenizer file")))?;
    let tok = Tokenizer::from_vocab_merges(vocab.parent().unwrap_or(Path::new("."))).map_err(Error)?;
    let r = Reader::load("Qwen3-VL-4B-Instruct".into(), te, None, Some(tok), gpu, log).map_err(|e| Error(e.0))?;
    Ok(Box::new(ReaderDescriber(r)))
}

impl Describer for ReaderDescriber {
    fn encoder(&self) -> String {
        self.0.name.clone()
    }

    fn load_seconds(&self) -> f64 {
        self.0.load_seconds
    }

    fn describe(&self, req: &DescribeRequest, progress: &mut dyn FnMut(usize) -> Result<()>) -> Result<DescribeOutput> {
        let p = req.picture.rgb();
        let a = self.0.answer(&p.data, p.width, p.height, &req.system, &req.user, req.max_tokens, req.temperature, req.seed, &mut |n| progress(n).is_err())
            .map_err(|e| Error(e.0))?;
        Ok(DescribeOutput {
            text: a.text,
            prompt_tokens: a.prompt_tokens,
            generated: a.generated,
            read_seconds: a.read_seconds,
            generate_seconds: a.generate_seconds,
            finish: if a.stopped { "stop".into() } else { "length".into() },
            seen: a.seen,
            held_mb: a.held_mb,
        })
    }
}
