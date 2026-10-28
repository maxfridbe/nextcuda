//! The text encoder: Qwen3.5-9B (gated DeltaNet and gated full attention), its last hidden state after the final norm
//! for every token of the prompt in the chat template - the pipeline's `encode_prompt`. It is nextcuda's own Qwen3.5
//! model core (crates/qwen35) on its card, the rows read through its `result_norm` tap; the head's logits it makes beside
//! them are dropped.
//!
//! ```text
//!   <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n   (add_generation_prompt, enable_thinking)
//! ```

use std::path::Path;
use std::sync::{Arc, Mutex};

use nextcuda_core::{Error, Gpu, Result};
use nextcuda_gguf::Gguf;
use nextcuda_qwen35::engine::{Qwen35, Session};
use nextcuda_tok::Tokenizer;

/// the hidden state's width
pub const HIDDEN: usize = 4096;
/// the pipeline's max_sequence_length
pub const MAX_TOKENS: usize = 4096;

pub struct TextEncoder {
    engine: Qwen35<'static>,
    tok: Tokenizer,
    session: Mutex<Session>,
    pub gpu: Arc<Gpu>,
}

/// The chat template's text for a prompt
pub fn template(prompt: &str) -> String {
    format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n")
}

impl TextEncoder {
    /// The model on `gpu` alone (its file kept open for the process: the engine reads its token rows from it)
    pub fn load(file: &Path, gpu: &Arc<Gpu>, log: &mut dyn FnMut(String)) -> Result<TextEncoder> {
        let f: &'static Gguf = Box::leak(Box::new(Gguf::open(file).map_err(|e| Error(e.0))?));
        if f.architecture() != "qwen35" {
            return Err(Error(format!("{}: a {} file; MOSS-Image's text encoder is Qwen3.5 (qwen35)", file.display(), f.architecture())));
        }
        let tok = Tokenizer::from_gguf(f).map_err(Error)?;
        let engine = Qwen35::load(f, std::slice::from_ref(gpu), false, (MAX_TOKENS, 1), log).map_err(|e| Error(e.0))?;
        let session = engine.session(MAX_TOKENS).map_err(|e| Error(e.0))?;
        Ok(TextEncoder { engine, tok, session: Mutex::new(session), gpu: gpu.clone() })
    }

    /// The template's ids for a prompt (cut at 4096, as the pipeline's tokenizer call truncates)
    pub fn ids(&self, prompt: &str) -> Vec<u32> {
        let mut ids = self.tok.encode(&template(prompt));
        ids.truncate(MAX_TOKENS);
        ids
    }

    /// The last hidden state float32 [L, 4096] for the template's ids: a pass a chunk of the engine's rows, every row's
    /// normed state read from its tap
    pub fn encode_ids(&self, ids: &[u32]) -> Result<Vec<f32>> {
        let mut s = self.session.lock().unwrap();
        self.engine.reset_session(&mut s).map_err(|e| Error(e.0))?;
        let mut out = Vec::with_capacity(ids.len() * HIDDEN);
        for c in ids.chunks(self.engine.prefill_chunk().max(1)) {
            let mut rows: Option<Vec<f32>> = None;
            let mut tap = |name: &str, b: &nextcuda_core::DevBuf| -> nextcuda_core::Result<()> {
                if name == "result_norm" {
                    rows = Some(b.to_f32().map_err(|e| nextcuda_core::Error(e.0))?);
                }
                Ok(())
            };
            self.engine.forward_rows(&mut s, c, c.len(), &mut tap).map_err(|e| Error(e.0))?;
            let r = rows.ok_or_else(|| Error("the text encoder's pass gave no result_norm rows".into()))?;
            if r.len() != c.len() * HIDDEN {
                return Err(Error(format!("result_norm: {} values for {} rows", r.len(), c.len())));
            }
            out.extend(r);
        }
        Ok(out)
    }

    pub fn encode(&self, prompt: &str) -> Result<Vec<f32>> {
        self.encode_ids(&self.ids(prompt))
    }
}
