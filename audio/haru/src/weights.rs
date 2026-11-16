//! The checkpoint (higurashi-ai/Haru-4B: one model.safetensors in bfloat16 - the language model `lm.*`, the two audio
//! projectors, the MeanFlow head `head.*`, the codec `codec.*` and the latents' scale and bias - beside its
//! config.json and tokenizer.json). The language model runs from its converted GGUF (convert.rs); everything else is
//! read from the safetensors here by name and put on the GPU as the VibeVoice engine's ops take it: matrices as half
//! (bfloat16 -> half: exact for these weights' range), the rest float32. Config: nextsycl's audio/haru/src/weights.rs.

use std::path::Path;

use nextcuda_audio::{Error, Result};
use nextcuda_core::DevBuf;
use nextcuda_gguf::safetensors::{f16_to_f32, SafeTensors, StTensor};
use serde_json::Value;

use crate::ops::{fp, Mat, Ops};

pub struct Weights {
    pub st: SafeTensors,
}

/// Floats as their bytes
pub fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// A tensor's raw bytes as float32 (BF16, F16, F32)
pub fn to_f32(t: &StTensor, b: &[u8]) -> Result<Vec<f32>> {
    Ok(match t.dtype.as_str() {
        "F32" => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        "BF16" => b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
        "F16" => b.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
        d => return Err(Error(format!("{}: {d}, not a float type this engine reads", t.name))),
    })
}

impl Weights {
    pub fn open(path: &Path) -> Result<Weights> {
        let st = SafeTensors::open(path).map_err(|e| Error(e.0))?;
        let w = Weights { st };
        for name in ["lm.embed_tokens.weight", "head.cond_proj.weight", "codec.decoder.head.weight", "codec.encoder.stem.weight", "model_audio_proj.fc1.weight",
                     "user_audio_proj.fc1.weight", "latent_scale"] {
            if !w.has(name) {
                return Err(Error(format!("{}: no {name} - not a Haru checkpoint (higurashi-ai/Haru-4B's model.safetensors)", path.display())));
            }
        }
        Ok(w)
    }

    pub fn find(&self, name: &str) -> Result<&StTensor> {
        self.st.need(name).map_err(|e| Error(e.0))
    }

    pub fn has(&self, name: &str) -> bool {
        self.st.tensor(name).is_some()
    }

    /// A tensor's shape (outermost first)
    pub fn shape(&self, name: &str) -> Result<Vec<usize>> {
        Ok(self.find(name)?.shape.iter().map(|d| *d as usize).collect())
    }

    /// A whole tensor as float32 on the host
    pub fn f32(&self, name: &str) -> Result<Vec<f32>> {
        let t = self.find(name)?;
        to_f32(t, &self.st.read(t).map_err(|e| Error(e.0))?)
    }

    /// Rows [r0, r1) of a tensor [R, ...] as float32 on the host
    pub fn rows_f32(&self, name: &str, r0: usize, r1: usize) -> Result<Vec<f32>> {
        let t = self.find(name)?;
        let row: u64 = t.shape.iter().skip(1).product();
        let es = match t.dtype.as_str() {
            "F32" => 4,
            "BF16" | "F16" => 2,
            d => return Err(Error(format!("{name}: {d}, not a float type this engine reads"))),
        };
        let mut b = vec![0u8; (r1 - r0) * row as usize * es];
        self.st.read_into(t, r0 as u64 * row * es as u64, &mut b).map_err(|e| Error(e.0))?;
        to_f32(t, &b)
    }

    /// A tensor as float32 on the GPU
    pub fn dev_f32(&self, ops: &Ops, name: &str) -> Result<DevBuf> {
        DevBuf::from_f32(&ops.gpu, &self.f32(name)?).map_err(|e| Error(e.0))
    }

    /// The scalar a 0-d tensor holds
    pub fn scalar(&self, name: &str) -> Result<f32> {
        self.f32(name)?.first().copied().ok_or_else(|| Error(format!("{name}: empty")))
    }

    /// A matrix from row ranges of tensors [N, K], stacked: [(name, r0, r1)]; half, or int8 rows (the VibeVoice
    /// engine's Weights::mat)
    pub fn mat(&self, ops: &Ops, parts: &[(&str, usize, usize)], int8: bool) -> Result<Mat> {
        let (mut k, mut n) = (0, 0);
        for (name, r0, r1) in parts {
            let s = self.shape(name)?;
            let kk: usize = s.iter().skip(1).product();
            if k != 0 && kk != k {
                return Err(Error(format!("{name}: {kk} wide, the matrix {k}")));
            }
            if *r1 > s[0] || r0 >= r1 {
                return Err(Error(format!("{name}: rows {r0}..{r1} of {}", s[0])));
            }
            k = kk;
            n += r1 - r0;
        }
        let gpu = &ops.gpu;
        let w = DevBuf::new(gpu, n * k * if int8 { 1 } else { 2 }).map_err(|e| Error(e.0))?;
        let scale = if int8 { Some(DevBuf::f32(gpu, n).map_err(|e| Error(e.0))?) } else { None };
        let mut row = 0;
        for (name, r0, r1) in parts {
            // in pieces of about 64 MiB of float32
            let step = ((64usize << 20) / (4 * k)).max(1);
            let mut a = *r0;
            while a < *r1 {
                let b = (a + step).min(*r1);
                let v = self.rows_f32(name, a, b)?;
                let f = DevBuf::from_f32(gpu, &v).map_err(|e| Error(e.0))?;
                match &scale {
                    None => ops.to_half(f.fp(), w.ptr().wrapping_byte_add(row * k * 2), v.len())?,
                    Some(s) => ops.quant_rows(f.fp(), b - a, k, w.ptr().wrapping_byte_add(row * k), fp(s, row))?,
                }
                gpu.sync().map_err(|e| Error(e.0))?;
                row += b - a;
                a = b;
            }
        }
        Ok(Mat { w, scale, n, k })
    }

    /// A whole tensor as a matrix
    pub fn mat1(&self, ops: &Ops, name: &str, int8: bool) -> Result<Mat> {
        let r = self.shape(name)?[0];
        self.mat(ops, &[(name, 0, r)], int8)
    }
}

/// The checkpoint's config.json (`model_type: haru`): the frame, the special ids, the language model's, the codec's
/// and the head's shapes
#[derive(Clone, Debug)]
pub struct Config {
    pub rate: usize,
    pub frame: usize,
    pub latent: usize,
    pub pad: u32,
    pub eop: u32,
    pub eos: Option<u32>,
    pub hidden: usize,
    pub ffn: usize,
    pub layers: usize,
    /// "linear_attention" / "full_attention" per layer
    pub full: Vec<bool>,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rot: usize,
    pub theta: f64,
    pub eps: f32,
    pub lin_k_heads: usize,
    pub lin_v_heads: usize,
    pub lin_k_dim: usize,
    pub lin_v_dim: usize,
    pub conv_k: usize,
    pub vocab: usize,
    pub ratios: Vec<usize>,
    pub codec_eps: f32,
    pub latent_std: f32,
    pub head_layers: usize,
    pub head_hidden: usize,
    pub head_eps: f32,
    pub interval: f32,
    pub head_steps: usize,
    pub guidance: f32,
}

impl Config {
    pub fn read(path: &Path) -> Result<Config> {
        let b = std::fs::read(path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
        let c: Value = serde_json::from_slice(&b).map_err(|e| Error(format!("{}: {e}", path.display())))?;
        Config::parse(&c).map_err(|e| Error(format!("{}: {}", path.display(), e.0)))
    }

    pub fn parse(c: &Value) -> Result<Config> {
        if c["model_type"].as_str().unwrap_or("haru") != "haru" {
            return Err(Error(format!("model_type {}: not a Haru checkpoint", c["model_type"])));
        }
        let num = |v: &Value, k: &str| v[k].as_f64().ok_or_else(|| Error(format!("no {k}: not Haru's config.json")));
        let us = |v: &Value, k: &str| num(v, k).map(|x| x as usize);
        let list = |v: &Value, k: &str| -> Result<Vec<usize>> {
            v[k].as_array().map(|a| a.iter().filter_map(|x| x.as_u64().map(|x| x as usize)).collect()).ok_or_else(|| Error(format!("no {k}")))
        };
        let lm = &c["lm"];
        let codec = &c["codec"];
        let head = &c["head"];
        let layers = us(lm, "num_hidden_layers")?;
        let full: Vec<bool> = match lm["layer_types"].as_array() {
            Some(a) => a.iter().map(|x| x.as_str() == Some("full_attention")).collect(),
            None => {
                let every = us(lm, "full_attention_interval")?;
                (0..layers).map(|l| (l + 1) % every == 0).collect()
            }
        };
        if full.len() != layers {
            return Err(Error(format!("{} layer types for {layers} layers", full.len())));
        }
        let rp = &lm["rope_parameters"];
        let head_dim = us(lm, "head_dim")?;
        let partial = rp["partial_rotary_factor"].as_f64().or_else(|| lm["partial_rotary_factor"].as_f64()).unwrap_or(1.0);
        let theta = rp["rope_theta"].as_f64().or_else(|| lm["rope_theta"].as_f64()).ok_or_else(|| Error("no rope_theta".into()))?;
        Ok(Config {
            rate: us(c, "sample_rate")?,
            frame: us(c, "frame_samples")?,
            latent: us(c, "latent_dim")?,
            pad: us(c, "pad_token_id")? as u32,
            eop: us(c, "eop_token_id")? as u32,
            eos: lm["eos_token_id"].as_u64().map(|x| x as u32),
            hidden: us(lm, "hidden_size")?,
            ffn: us(lm, "intermediate_size")?,
            layers,
            full,
            heads: us(lm, "num_attention_heads")?,
            kv_heads: us(lm, "num_key_value_heads")?,
            head_dim,
            rot: (head_dim as f64 * partial) as usize,
            theta,
            eps: num(lm, "rms_norm_eps")? as f32,
            lin_k_heads: us(lm, "linear_num_key_heads")?,
            lin_v_heads: us(lm, "linear_num_value_heads")?,
            lin_k_dim: us(lm, "linear_key_head_dim")?,
            lin_v_dim: us(lm, "linear_value_head_dim")?,
            conv_k: us(lm, "linear_conv_kernel_dim")?,
            vocab: us(lm, "vocab_size")?,
            ratios: list(codec, "ratios")?,
            codec_eps: num(codec, "norm_eps").unwrap_or(1e-5) as f32,
            latent_std: num(codec, "latent_std").unwrap_or(0.5) as f32,
            head_layers: us(head, "num_layers")?,
            head_hidden: us(head, "hidden_size")?,
            head_eps: num(head, "norm_eps").unwrap_or(1e-5) as f32,
            interval: num(head, "interval_embed").unwrap_or(7.0) as f32,
            head_steps: us(head, "num_steps").unwrap_or(1).max(1),
            guidance: num(head, "guidance_scale").unwrap_or(1.5) as f32,
        })
    }

    /// The Gated DeltaNet's convolved channels: queries, keys and values
    pub fn conv_dim(&self) -> usize {
        2 * self.lin_k_heads * self.lin_k_dim + self.lin_v_heads * self.lin_v_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_release_config_is_read() {
        let c = serde_json::json!({
            "model_type": "haru", "sample_rate": 24000, "frame_samples": 3200, "latent_dim": 64, "pad_token_id": 248076, "eop_token_id": 248071,
            "lm": {"vocab_size": 248320, "hidden_size": 2560, "intermediate_size": 9216, "num_hidden_layers": 4, "num_attention_heads": 16,
                   "num_key_value_heads": 4, "head_dim": 256, "rms_norm_eps": 1e-6, "full_attention_interval": 4, "linear_conv_kernel_dim": 4,
                   "linear_key_head_dim": 128, "linear_value_head_dim": 128, "linear_num_key_heads": 16, "linear_num_value_heads": 32,
                   "rope_parameters": {"partial_rotary_factor": 0.25, "rope_theta": 10000000}, "eos_token_id": 248044},
            "codec": {"ratios": [8, 5, 5, 4, 2, 2], "norm_eps": 1e-5, "latent_std": 0.5},
            "head": {"hidden_size": 2560, "num_layers": 4, "norm_eps": 1e-5, "interval_embed": 7.0, "num_steps": 1, "guidance_scale": 1.5}
        });
        let c = Config::parse(&c).unwrap();
        assert_eq!((c.rot, c.conv_dim(), c.full.clone(), c.eos), (64, 8192, vec![false, false, false, true], Some(248044)));
        assert!(Config::parse(&serde_json::json!({"model_type": "other"})).is_err());
    }
}
