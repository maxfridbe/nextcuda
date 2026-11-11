//! A Qwen3-VL that reads a picture and answers: the vision tower and the language model (`chat`) together, one
//! request at a time - what an image model's own text encoder is as image.describe's native describer (Qwen-Image
//! 2.1's 8B from its llama.cpp files, P-H3-Image's 4B from its encoder GGUF of transformers' names). The image engines
//! wrap it in the image contract's Describer.

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use nextcuda_core::{Error, Gpu, Result};
use nextcuda_diffusion::kernels::Nsd;
use nextcuda_tok::Tokenizer;

use crate::{Chat, Sampler, Vision};

pub struct Reader {
    nsd: Nsd,
    gpu: Arc<Gpu>,
    pub chat: Chat,
    vision: Vision,
    /// what it is ("Qwen3-VL-8B-Instruct")
    pub name: String,
    pub load_seconds: f64,
    /// the card's free memory before it loaded (what its holding is measured against)
    free_before: u64,
    work: Mutex<()>,
}

/// What it read and wrote
pub struct Answer {
    pub text: String,
    pub prompt_tokens: usize,
    pub generated: usize,
    pub read_seconds: f64,
    pub generate_seconds: f64,
    pub stopped: bool,
    /// the size the vision tower read
    pub seen: (u32, u32),
    /// the card's memory it held, MiB (the most of its samples after each stage)
    pub held_mb: u64,
}

/// The side the vision tower reads: the nearest multiple of 32 (at least 32)
fn side32(x: u32) -> u32 {
    (((x as f64) / 32.0).round() as u32).max(1) * 32
}

/// The size the vision tower reads a `w` x `h` picture at: sides on its 32-pixel grid, the area at most the picture's
pub fn grid_size(w: u32, h: u32) -> (u32, u32) {
    let (mut gw, mut gh) = (side32(w), side32(h));
    while (gw as u64 * gh as u64) > (w as u64 * h as u64).max(1024) && (gw > 32 || gh > 32) {
        if gw >= gh { gw -= 32 } else { gh -= 32 }
    }
    (gw, gh)
}

impl Reader {
    /// The language model from `lm` (its tokenizer in it, else `tok`) and the vision tower: from the llama.cpp
    /// projector `mmproj`, or from `lm` itself under `model.visual.` when there is none
    pub fn load(name: String, lm: &Path, mmproj: Option<&Path>, tok: Option<Tokenizer>, gpu: &Arc<Gpu>, log: &mut dyn FnMut(String)) -> Result<Reader> {
        let t0 = Instant::now();
        let free_before = gpu.memory().ok().and_then(|(_, f)| f).unwrap_or(0);
        let nsd = Nsd::new(gpu)?;
        let vision = match mmproj {
            Some(p) => Vision::load(p, gpu, &mut |l| log(l))?,
            None => Vision::load_hf(lm, "model.visual.", gpu, &mut |l| log(l))?,
        };
        let chat = Chat::load(lm, tok, gpu, &mut |l| log(l))?;
        if vision.out != chat.hidden {
            return Err(Error(format!("the vision tower gives {} wide tokens; the language model is {} wide", vision.out, chat.hidden)));
        }
        let load_seconds = t0.elapsed().as_secs_f64();
        log(format!("{name}: the vision tower and the language model on {} in {load_seconds:.1} s", gpu.name));
        Ok(Reader { nsd, gpu: gpu.clone(), chat, vision, name, load_seconds, free_before, work: Mutex::new(()) })
    }

    fn held_mb(&self) -> u64 {
        let free = self.gpu.memory().ok().and_then(|(_, f)| f).unwrap_or(self.free_before);
        self.free_before.saturating_sub(free) >> 20
    }

    /// `rgb` (w x h, 8-bit RGB) read with the instruction `system` and the user's words `user`, the answer written:
    /// `each(tokens so far)` true stops it (a cancel)
    #[allow(clippy::too_many_arguments)]
    pub fn answer(&self, rgb: &[u8], w: u32, h: u32, system: &str, user: &str, max_tokens: usize, temperature: f32, seed: u64,
                  each: &mut dyn FnMut(usize) -> bool) -> Result<Answer> {
        let _one = self.work.lock().unwrap_or_else(|p| p.into_inner());
        let t0 = Instant::now();
        let mut held = self.held_mb();
        let (gw, gh) = grid_size(w, h);
        let px: Vec<f32> = if (gw, gh) == (w, h) {
            rgb.iter().map(|v| *v as f32).collect()
        } else {
            let p = nextcuda_diffusion::picture::Picture { width: w, height: h, channels: 3, data: rgb.to_vec() };
            p.resize_lanczos(gw, gh).data.iter().map(|v| *v as f32).collect()
        };
        let seen = self.vision.see(&self.nsd, &px, gh as usize, gw as usize)?;
        held = held.max(self.held_mb());
        let ids = self.chat.prompt_ids(system, user, seen.n)?;
        let mut sampler = Sampler::new(temperature, seed);
        let mut stopped_by_caller = false;
        let written = self.chat.write(&self.nsd, &ids, &[&seen], max_tokens, &mut |l| sampler.pick(l), &mut |n| {
            stopped_by_caller = each(n);
            stopped_by_caller
        });
        held = held.max(self.held_mb());
        let written = match written {
            Err(_) if stopped_by_caller => return Err(Error("cancelled".into())),
            x => x?,
        };
        Ok(Answer {
            text: written.text,
            prompt_tokens: written.prompt_tokens,
            generated: written.tokens.len(),
            read_seconds: t0.elapsed().as_secs_f64() - written.generate_seconds,
            generate_seconds: written.generate_seconds,
            stopped: written.stopped,
            seen: (gw, gh),
            held_mb: held,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_tower_reads_sides_of_32_at_most_the_area() {
        assert_eq!(grid_size(1024, 1024), (1024, 1024));
        let (w, h) = grid_size(1000, 700);
        assert!(w % 32 == 0 && h % 32 == 0 && w * h <= 1000 * 700, "{w}x{h}");
        assert_eq!(grid_size(5, 5), (32, 32));
    }
}
