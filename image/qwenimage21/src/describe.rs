//! image.describe's native describer for Qwen-Image 2.1: its own text encoder (Qwen3-VL-8B-Instruct, the card's
//! `text-encoder` GGUF and its `llm-vision` projector) reading the picture and writing the prompt (crates/qwen3vl
//! `Reader`: the vision tower of an edit's references and the language model as the chat model it is). The image
//! server loads it in place of the pipeline (the 12 GB card holds one or the other) and keeps it until the next
//! picture is drawn.
//!
//! Memory: 4.7 GiB of Q4_K / Q6_K weights, 1.1 GiB of the vision tower's half weights, the cache (36 layers x 8 key
//! heads x 128 x 2 x half: 144 KiB a row - 1,400 rows for a 1 MP picture and 300 written tokens is 200 MiB), the
//! vision tower's activations while the picture is read.

use std::path::Path;
use std::sync::Arc;

use nextcuda_core::Gpu;
use nextcuda_image::{DescribeOutput, DescribeRequest, Describer, Error, ModelFiles, Result};
use nextcuda_qwen3vl::Reader;

use crate::ARCH;

/// A Reader behind the image contract (P-H3-Image's encoder is the same thing: image/ph3image wraps its own)
pub struct ReaderDescriber(pub Reader);

/// The encoder's name from its file: "Qwen3VL-8B-Instruct-Q4_K_M.gguf" -> "Qwen3-VL-8B-Instruct"
pub fn encoder_name(path: &Path) -> String {
    let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let quant = |p: &str| {
        let u = p.to_ascii_uppercase();
        ["Q2", "Q3", "Q4", "Q5", "Q6", "Q8", "IQ", "F16", "BF16", "F32"].iter().any(|q| u.starts_with(q))
    };
    let parts: Vec<&str> = stem.split('-').collect();
    let keep = parts.iter().rposition(|p| !quant(p)).map_or(parts.len(), |i| i + 1);
    let name = parts[..keep].join("-");
    if let Some(rest) = name.strip_prefix("Qwen3VL") {
        return format!("Qwen3-VL{rest}");
    }
    if let Some(rest) = name.strip_prefix("Qwen2.5VL") {
        return format!("Qwen2.5-VL{rest}");
    }
    name
}

/// The describer of a Qwen-Image 2.1 card: its text encoder and projector onto the first of its cards
pub fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], log: &mut dyn FnMut(String)) -> Result<Box<dyn Describer>> {
    let gpu = gpus.first().ok_or_else(|| Error("no GPU".into()))?;
    let te = files.get("text-encoder").ok_or_else(|| Error(format!("{ARCH}: no text-encoder file")))?;
    let vis = files.get("llm-vision").ok_or_else(|| Error(format!("{ARCH}: no llm-vision file (the text encoder's projector): it cannot read pictures")))?;
    let r = Reader::load(encoder_name(te), te, Some(vis), None, gpu, log).map_err(|e| Error(e.0))?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_encoder_is_named_from_its_file() {
        assert_eq!(encoder_name(Path::new("/m/Qwen3VL-8B-Instruct-Q4_K_M.gguf")), "Qwen3-VL-8B-Instruct");
        assert_eq!(encoder_name(Path::new("/m/Qwen2.5VL-7B-Instruct-F16.gguf")), "Qwen2.5-VL-7B-Instruct");
        assert_eq!(encoder_name(Path::new("/m/model.gguf")), "model");
    }
}
