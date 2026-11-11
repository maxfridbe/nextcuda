//! LoRAs: low-rank additions to the denoiser's linear layers, from a `.safetensors` in ComfyUI's layout for this model
//! (`diffusion_model.blocks.N.attn.qkv_proj.lora_A.weight` [r, K] / `lora_B.weight` [N, r], optionally `.alpha`).
//!
//! Applied beside the int8 weights at run time - `out += strength * alpha / r * B (A x)` - rather than merged into
//! them: merging would mean expanding every touched matrix, adding, and quantizing again, which costs time and
//! precision; the side path costs two thin matrix products per touched layer.

use std::sync::Arc;

use crate::device::{Device, Tensor};
use crate::dit::host;
use crate::dtype::DType;
use crate::ops::Lora;
use crate::safetensors::Checkpoint;
use crate::{Error, Result};

/// The four linears of a block, in the order the block runs them.
pub const LINEARS: [&str; 4] = ["attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"];

/// A block's LoRAs, per linear of `LINEARS` (several LoRAs stack: each adds its own side path).
pub type BlockLora = [Vec<Lora>; 4];

pub struct LoraSet {
    pub blocks: Vec<BlockLora>,
    pub refiner: Vec<BlockLora>,
    pub max_rank: usize,
    pub layers: usize,
}

/// A one-value tensor (a LoRA's alpha): a float, or an integer as some converters write it (pDMD's ComfyUI file: int64)
fn scalar(ck: &Checkpoint, name: &str) -> Result<f32> {
    let e = ck.get(name)?;
    let b = ck.read(name)?;
    let v = match e.dtype {
        DType::I64 if b.len() >= 8 => i64::from_le_bytes(b[..8].try_into().unwrap_or_default()) as f32,
        DType::I32 if b.len() >= 4 => i32::from_le_bytes(b[..4].try_into().unwrap_or_default()) as f32,
        _ => host(ck, name)?.first().copied().ok_or_else(|| Error(format!("{name}: an empty alpha")))?,
    };
    Ok(v)
}

/// The bytes block `i` of `prefix` ("blocks") takes on a card for a LoRA in `ck` (its A and B, bfloat16): what the
/// denoiser's placement keeps beside the block's own matrices
pub fn block_bytes(ck: &Checkpoint, prefix: &str, i: usize) -> u64 {
    LINEARS.iter().flat_map(|l| ["lora_A", "lora_B"].map(|ab| format!("diffusion_model.{prefix}.{i}.{l}.{ab}.weight")))
        .filter_map(|n| ck.get(&n).ok().map(|e| e.shape.iter().product::<usize>() as u64 * 2)).sum()
}

fn bf16(dev: &Arc<Device>, shape: &[usize], v: &[f32]) -> Result<Tensor> {
    Tensor::from_bytes(dev, DType::BF16, shape, &crate::denoiser::bf16_bytes(v))
}

impl LoraSet {
    /// `blocks` / `refiner`: how many of each the model has.
    pub fn load(dev: &Arc<Device>, ck: &Checkpoint, strength: f32, blocks: usize, refiner: usize) -> Result<LoraSet> {
        LoraSet::load_placed(std::slice::from_ref(dev), &|_| 0, ck, strength, blocks, refiner)
    }

    /// The same, for a denoiser split over cards: block `i`'s layers go on card `devs[card_of(i)]` (the text refiner's on
    /// the first card, where the text is).
    pub fn load_placed(devs: &[Arc<Device>], card_of: &dyn Fn(usize) -> usize, ck: &Checkpoint, strength: f32, blocks: usize, refiner: usize) -> Result<LoraSet> {
        let mut max_rank = 0;
        let mut layers = 0;
        let mut side = |prefix: &str, n: usize, placed: bool| -> Result<Vec<BlockLora>> {
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let dev = &devs[if placed { card_of(i).min(devs.len() - 1) } else { 0 }];
                let mut bl: BlockLora = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
                for (j, l) in LINEARS.iter().enumerate() {
                    let p = format!("diffusion_model.{prefix}.{i}.{l}");
                    let (ka, kb) = (format!("{p}.lora_A.weight"), format!("{p}.lora_B.weight"));
                    if !ck.entries.contains_key(&ka) {
                        continue;
                    }
                    let (a, mut b) = (host(ck, &ka)?, host(ck, &kb)?);
                    let (sa, sb) = (ck.get(&ka)?.shape.clone(), ck.get(&kb)?.shape.clone());
                    let r = sa[0];
                    if sb[1] != r {
                        return Err(Error(format!("{p}: lora_A rank {r}, lora_B {:?}", sb)));
                    }
                    let alpha = match ck.entries.contains_key(&format!("{p}.alpha")) {
                        true => scalar(ck, &format!("{p}.alpha"))? / r as f32,
                        false => 1.0,
                    };
                    let scale = strength * alpha;
                    b.iter_mut().for_each(|v| *v *= scale);
                    bl[j].push(Lora { a: bf16(dev, &sa, &a)?, b: bf16(dev, &sb, &b)? });
                    max_rank = max_rank.max(r);
                    layers += 1;
                }
                out.push(bl);
            }
            Ok(out)
        };
        let blocks = side("blocks", blocks, true)?;
        let refiner = side("token_refiner.blocks", refiner, false)?;
        if layers == 0 {
            return Err(Error(format!("{}: no LoRA layers this model has", ck.path.display())));
        }
        Ok(LoraSet { blocks, refiner, max_rank, layers })
    }

    /// Another LoRA's layers on top of these.
    pub fn stack(&mut self, other: LoraSet) {
        for (mine, theirs) in self.blocks.iter_mut().zip(other.blocks).chain(self.refiner.iter_mut().zip(other.refiner)) {
            for (m, t) in mine.iter_mut().zip(theirs) {
                m.extend(t);
            }
        }
        self.max_rank = self.max_rank.max(other.max_rank);
        self.layers += other.layers;
    }
}
