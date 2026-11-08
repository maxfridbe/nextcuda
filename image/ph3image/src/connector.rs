//! P-H3-Image's text connector (diffusers' `PH3ImageTextConnector`): from the Qwen3-VL-4B text encoder's hidden
//! states to the conditioning MiniMax H3's denoiser reads - the place where a 4B encoder stands in for H3's 32B one.
//!
//! ```text
//!   x = [h13 | h19 | h25 | h31 | h36]            the encoder's hidden states after those layers (h36: through its final norm)
//!   y = ((x - input_mean) / input_std) . W^T     W [5120, 12800], float32, no bias
//!   y = y * output_std + output_mean
//!   y[0] = sink_embedding                        the first token is an attention sink: the same row for every prompt
//! ```
//!
//! Everything float32, as the released checkpoint keeps it; the product on the card, the rest on the host.

use std::sync::Arc;

use nextcuda_h3::device::{Device, Tensor};
use nextcuda_h3::dtype::{bytes_to_f32, DType};
use nextcuda_h3::ops::Linear;
use nextcuda_h3::safetensors::Checkpoint;
use nextcuda_h3::{Error, Result};

/// The hidden states it reads, in transformers' `output_hidden_states` numbering (the config's hidden_state_indices)
pub const TAPS: [usize; 5] = [13, 19, 25, 31, 36];
const PREFIX: &str = "text_connector.";

pub struct Connector {
    input_mean: Vec<f32>,
    input_std: Vec<f32>,
    output_mean: Vec<f32>,
    output_std: Vec<f32>,
    sink: Vec<f32>,
    proj: Linear,
    /// the encoder's width (each tap's)
    pub text: usize,
    /// the conditioning's width (H3's text_dim, 5120)
    pub out: usize,
}

fn host(ck: &Checkpoint, name: &str) -> Result<Vec<f32>> {
    let name = format!("{PREFIX}{name}");
    bytes_to_f32(&ck.read(&name)?, ck.get(&name)?.dtype)
}

impl Connector {
    /// From the text encoder's file (P-H3-Image's GGUF carries the connector beside the encoder), the product on `dev`
    pub fn load(dev: &Arc<Device>, ck: &Checkpoint) -> Result<Connector> {
        let w = host(ck, "proj.weight")?;
        let shape = ck.get(&format!("{PREFIX}proj.weight"))?.shape.clone();
        let (out, inp) = (shape[0], shape[1]);
        if inp % TAPS.len() != 0 || w.len() != out * inp {
            return Err(Error(format!("the text connector's projection is {shape:?}: not [out, {} x hidden]", TAPS.len())));
        }
        let bytes: Vec<u8> = w.iter().flat_map(|f| f.to_le_bytes()).collect();
        let c = Connector {
            input_mean: host(ck, "input_mean")?,
            input_std: host(ck, "input_std")?,
            output_mean: host(ck, "output_mean")?,
            output_std: host(ck, "output_std")?,
            sink: host(ck, "sink_embedding")?,
            proj: Linear { weight: Tensor::from_bytes(dev, DType::F32, &[out, inp], &bytes)?, bias: None },
            text: inp / TAPS.len(),
            out,
        };
        if c.input_mean.len() != inp || c.input_std.len() != inp || c.output_mean.len() != out || c.output_std.len() != out || c.sink.len() != out {
            return Err(Error("the text connector's statistics do not fit its projection".into()));
        }
        Ok(c)
    }

    /// The taps (each [tokens, text], in `TAPS` order) -> the conditioning [tokens, out]
    pub fn forward(&self, taps: &[Vec<f32>]) -> Result<Vec<f32>> {
        if taps.len() != TAPS.len() || taps.iter().any(|t| t.len() != taps[0].len() || t.len() % self.text != 0) {
            return Err(Error(format!("the text connector takes {} hidden states of {} a token", TAPS.len(), self.text)));
        }
        let l = taps[0].len() / self.text;
        let inp = self.text * TAPS.len();
        let mut x = vec![0f32; l * inp];
        for r in 0..l {
            for (j, t) in taps.iter().enumerate() {
                for c in 0..self.text {
                    let k = j * self.text + c;
                    x[r * inp + k] = (t[r * self.text + c] - self.input_mean[k]) / self.input_std[k];
                }
            }
        }
        let dev = self.proj.weight.buf.device();
        let xt = Tensor::from_bytes(dev, DType::F32, &[l, inp], &x.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>())?;
        let yt = Tensor::new(dev, DType::F32, &[l, self.out])?;
        self.proj.forward(&xt, &yt)?;
        let mut y = yt.to_f32()?;
        for r in 0..l {
            let row = &mut y[r * self.out..(r + 1) * self.out];
            if r == 0 {
                row.copy_from_slice(&self.sink);
            } else {
                for (c, v) in row.iter_mut().enumerate() {
                    *v = *v * self.output_std[c] + self.output_mean[c];
                }
            }
        }
        Ok(y)
    }
}
