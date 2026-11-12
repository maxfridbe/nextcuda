//! MiniMax H3's Fun ControlNet-Union (ComfyUI `ldm/minimax/controlnet.py`, `MiniMaxH3FunControlPatch`): a second stream
//! of DiT blocks beside the denoiser's that steers a clip by a control video (canny, depth, HED, MLSD or pose frames,
//! made beforehand) and/or regenerates the masked part of a source video.
//!
//! ```text
//!   hint   = [control video's latent (24) | visibility (1) | the source's latent with the mask blacked out (24)]
//!            (what is not given is zeros), patchified like the target video: 49 x 4 values a token
//!   before block 0          stash the stream h
//!   after block 0           c = h with every video-like row replaced by proj_in(its hint row; zeros but for the
//!                           target video), then c = before_proj(c) + h
//!   after injection layer i c = control block k(c) (the denoiser's tables, positions and rows); the stream gets
//!                           strength * after_proj_k(c) added on every row but the audio ones
//! ```
//!
//! Version 2.0 (`control_blocks_places` in the file's metadata): ten blocks, after layers 0, 5, 10 ... 45; the blocks
//! are int8 ConvRot like the denoiser's and their tables come from the same timestep curve.
//!
//! On a denoiser split over two 12 GB cards (nextcuda's Q4_K_M: 10.6 GiB of blocks) the 4.2 GiB of control blocks do
//! not fit on one card beside its half: each control block goes beside the denoiser block it follows when that card
//! has the room ([`placement`]), and the control stream moves with them - once a step at most, through host memory, as
//! the denoiser's own stream does; a block's addition is moved to the card the denoiser's stream is on.

use std::sync::Arc;

use crate::device::{Device, Tensor};
use crate::dit::{Blocks, Config};
use crate::dtype::DType;
use crate::ops::Linear;
use crate::safetensors::Checkpoint;
use crate::{Error, Result};

pub struct Control {
    pub blocks: Blocks,
    /// each control block's card (an index into `blocks.devs`)
    pub dev_of: Vec<usize>,
    /// the denoiser layers after which each control block runs
    pub at: Vec<usize>,
    /// hint channels a latent pixel (49)
    pub in_dim: usize,
    proj_in: Linear,
    before: Linear,
    /// strength folded in
    after: Vec<Linear>,
    /// active from noise level `start` down to `end`
    pub sigmas: (f32, f32),
}

fn f32_tensor(dev: &Arc<Device>, v: &[f32], shape: &[usize]) -> Result<Tensor> {
    Tensor::from_bytes(dev, DType::F32, shape, &v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>())
}

/// A bfloat16 linear from the file (weight and float32 bias), its outputs scaled by `s`
fn bf16_linear(dev: &Arc<Device>, ck: &Checkpoint, name: &str, s: f32) -> Result<Linear> {
    let w = ck.get(&format!("{name}.weight"))?;
    let v = crate::dtype::bytes_to_f32(&ck.read(&format!("{name}.weight"))?, w.dtype)?;
    let bf: Vec<u8> = v.iter().flat_map(|x| crate::dtype::f32_to_bf16(x * s).to_le_bytes()).collect();
    let b: Vec<f32> = crate::dit::host(ck, &format!("{name}.bias"))?.iter().map(|x| x * s).collect();
    Ok(Linear { weight: Tensor::from_bytes(dev, DType::BF16, &w.shape, &bf)?, bias: Some(f32_tensor(dev, &b, &[b.len()])?) })
}

impl Control {
    /// The patch's control blocks and the denoiser layers they follow, read from the file (`main`: the denoiser's
    /// shape)
    pub fn places(ck: &Checkpoint, main: &Config) -> Result<Vec<usize>> {
        let n = (0..).take_while(|i| ck.get(&format!("control_blocks.{i}.after_proj.weight")).is_ok()).count();
        if n == 0 {
            return Err(Error(format!("{}: not a MiniMax H3 Fun ControlNet (no control_blocks)", ck.path.display())));
        }
        let at: Vec<usize> = match ck.metadata.get("control_blocks_places") {
            Some(p) => serde_json::from_str(p).map_err(|e| Error(format!("control_blocks_places: {e}")))?,
            None => (0..n).map(|i| i * 10).collect(),
        };
        if at.len() != n || at.first() != Some(&0) || at.windows(2).any(|w| w[0] >= w[1]) || at.iter().any(|l| *l >= main.blocks) {
            return Err(Error(format!("control blocks after layers {at:?}: {n} blocks, from layer 0, increasing, within the denoiser's {}", main.blocks)));
        }
        Ok(at)
    }

    /// The patch on one card: its blocks shaped as the denoiser's (`main`); `strength` (1.0), and where it acts
    /// (`sigmas`: from, to - the whole schedule is (1, 0))
    pub fn load(dev: &Arc<Device>, ck: &Checkpoint, main: Config, strength: f32, sigmas: (f32, f32), threads: usize) -> Result<Control> {
        let n = Self::places(ck, &main)?.len();
        Self::load_placed(std::slice::from_ref(dev), vec![0; n], ck, main, strength, sigmas, threads)
    }

    /// The patch over the denoiser's cards, control block `k` on `devs[dev_of[k]]` ([`placement`])
    pub fn load_placed(devs: &[Arc<Device>], dev_of: Vec<usize>, ck: &Checkpoint, main: Config, strength: f32, sigmas: (f32, f32), threads: usize) -> Result<Control> {
        let at = Self::places(ck, &main)?;
        let n = at.len();
        let td = ck.get("control_blocks.0.adaln_proj.linear.weight")?.shape[1];
        if td != main.t_dim {
            return Err(Error(format!("the ControlNet's timestep tables are {td} wide, the denoiser's {}: another adaln form (convert one to match)", main.t_dim)));
        }
        let blocks = Blocks::load_placed(devs, ck, "control_blocks", Config { blocks: n, ..main }, dev_of.clone(), threads)?;
        // the projection in and before_proj run with block 0 (`start`), each after_proj with its block
        let dev = &devs[dev_of[0]];
        let pw = ck.get("control_proj_in.weight")?.shape.clone(); // [hidden, in_dim * 4]
        let proj_in = Linear {
            weight: f32_tensor(dev, &crate::dit::host(ck, "control_proj_in.weight")?, &pw)?,
            bias: Some(crate::dit::small(dev, ck, "control_proj_in.bias")?),
        };
        Ok(Control {
            blocks,
            dev_of: dev_of.clone(),
            at,
            in_dim: pw[1] / 4,
            proj_in,
            before: bf16_linear(dev, ck, "control_blocks.0.before_proj", 1.0)?,
            after: (0..n).map(|i| bf16_linear(&devs[dev_of[i]], ck, &format!("control_blocks.{i}.after_proj"), strength)).collect::<Result<_>>()?,
            sigmas,
        })
    }

    /// Control block `k`'s card (an index into the denoiser's cards)
    pub fn card_of(&self, k: usize) -> usize {
        self.dev_of.get(k).copied().unwrap_or(0)
    }

    pub fn active(&self, sigma: f32) -> bool {
        self.sigmas.1 <= sigma && sigma <= self.sigmas.0
    }

    /// The control stream after block 0: `h` (the stream before block 0, overwritten with the result) with its
    /// video-like rows replaced by their hint rows' projection (`rows`: (first row, [n, in_dim * 4] float32) per
    /// video-like segment, in order), then before_proj of that added to `h`. `c` is scratch of the stream's size.
    pub fn start(&self, h: &Tensor, c: &Tensor, rows: &[(usize, &Tensor)]) -> Result<()> {
        let dev = h.buf.device().clone();
        c.copy_rows(0, h, 0, h.shape[0])?;
        for (at, r) in rows {
            let out = Tensor::new(&dev, DType::BF16, &[r.shape[0], h.shape[1]])?;
            self.proj_in.forward(r, &out)?;
            c.copy_rows(*at, &out, 0, r.shape[0])?;
        }
        self.before.forward_acc(c, h)
    }

    /// Control block `k` on the stream `c`, and its addition to the denoiser's stream `x` (`skip` scratch of the
    /// stream's size; `audio`: the audio rows, which get none: (first row, rows) and a zero tensor at least that long)
    #[allow(clippy::too_many_arguments)]
    pub fn step(&self, k: usize, c: &Tensor, x: &Tensor, skip: &Tensor, step: &crate::dit::Step, s: &crate::dit::Scratch,
                audio: (&[(usize, usize)], &Tensor)) -> Result<()> {
        self.addition(k, c, skip, step, s, audio)?;
        crate::ops::add(x, skip)
    }

    /// Control block `k` on the stream `c` (on its card) and what it adds to the denoiser's stream, in `skip` (on the
    /// same card); the caller adds it where the denoiser's stream is
    pub fn addition(&self, k: usize, c: &Tensor, skip: &Tensor, step: &crate::dit::Step, s: &crate::dit::Scratch, audio: (&[(usize, usize)], &Tensor)) -> Result<()> {
        self.blocks.block(k, c, step, s, None, None)?;
        self.after[k].forward(c, skip)?;
        for (at, n) in audio.0 {
            skip.copy_rows(*at, audio.1, 0, *n)?;
        }
        Ok(())
    }
}

/// Where each control block goes over the denoiser's cards: beside the denoiser block it follows (`home[k]`, that
/// block's card) while that card has `bytes[k]` to spare beyond `keep[card]` (a clip's buffers), else on the card with
/// the most room left; None when one has no card at all. `free[card]`: the bytes free on each card now.
pub fn placement(bytes: &[u64], home: &[usize], free: &[u64], keep: &[u64]) -> Option<Vec<usize>> {
    let mut room: Vec<i128> = free.iter().zip(keep).map(|(f, k)| *f as i128 - *k as i128).collect();
    let mut out = Vec::with_capacity(bytes.len());
    for (k, b) in bytes.iter().enumerate() {
        let b = *b as i128;
        let h = home.get(k).copied().unwrap_or(0).min(room.len().saturating_sub(1));
        let d = if room[h] >= b {
            h
        } else {
            let (d, r) = room.iter().enumerate().max_by_key(|(_, r)| **r)?;
            if *r < b {
                return None;
            }
            d
        };
        room[d] -= b;
        out.push(d);
    }
    Some(out)
}

/// Linear resampling of one axis with half-pixel centres (PyTorch's align_corners=False): `n_in` -> `n_out`
fn taps(n_out: usize, n_in: usize) -> Vec<(usize, usize, f32)> {
    let scale = n_in as f64 / n_out as f64;
    (0..n_out)
        .map(|i| {
            let src = ((i as f64 + 0.5) * scale - 0.5).max(0.0);
            let i0 = (src.floor() as usize).min(n_in - 1);
            let i1 = (i0 + 1).min(n_in - 1);
            (i0, i1, (src - i0 as f64) as f32)
        })
        .collect()
}

/// A [f, h, w] volume to [t, lh, lw] by trilinear interpolation (align_corners=False), as ComfyUI shrinks the
/// visibility mask to the latent grid
pub fn trilinear(v: &[f32], (f, h, w): (usize, usize, usize), (t, lh, lw): (usize, usize, usize)) -> Vec<f32> {
    let (tf, th, tw) = (taps(t, f), taps(lh, h), taps(lw, w));
    // along width, then height, then time
    let mut a = vec![0f32; f * h * lw];
    for r in 0..f * h {
        for (x, (i0, i1, l)) in tw.iter().enumerate() {
            a[r * lw + x] = v[r * w + i0] * (1.0 - l) + v[r * w + i1] * l;
        }
    }
    let mut b = vec![0f32; f * lh * lw];
    for fi in 0..f {
        for (y, (i0, i1, l)) in th.iter().enumerate() {
            for x in 0..lw {
                b[(fi * lh + y) * lw + x] = a[(fi * h + i0) * lw + x] * (1.0 - l) + a[(fi * h + i1) * lw + x] * l;
            }
        }
    }
    let n = lh * lw;
    let mut out = vec![0f32; t * n];
    for (ti, (i0, i1, l)) in tf.iter().enumerate() {
        for j in 0..n {
            out[ti * n + j] = b[i0 * n + j] * (1.0 - l) + b[i1 * n + j] * l;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn control_blocks_go_beside_the_layers_they_follow() {
        let gib = 1u64 << 30;
        // a split denoiser: layers 0-24 on the 5070, 25-49 on the 3060; ten 0.42 GiB control blocks after 0, 5 ... 45
        let home: Vec<usize> = (0..10).map(|k| usize::from(k * 5 >= 25)).collect();
        let bytes = vec![450 << 20; 10];
        let p = placement(&bytes, &home, &[5 * gib, 5 * gib], &[2 * gib, 2 * gib]).unwrap();
        assert_eq!(p, home, "each beside its layer");
        // the first card short of room: the rest of its blocks go to the other
        let p = placement(&bytes, &home, &[3 * gib, 6 * gib], &[2 * gib, 2 * gib]).unwrap();
        assert_eq!(p, vec![0, 0, 1, 1, 1, 1, 1, 1, 1, 1]);
        // no room anywhere: refused, not squeezed
        assert!(placement(&bytes, &home, &[2 * gib, 3 * gib], &[2 * gib, 2 * gib]).is_none());
        // one card: everything on it
        assert_eq!(placement(&bytes[..2], &[0, 0], &[8 * gib], &[2 * gib]).unwrap(), vec![0, 0]);
    }

    #[test]
    fn trilinear_matches_pytorch() {
        // torch.nn.functional.interpolate(x, size, mode="trilinear", align_corners=False) on these volumes
        let close = |o: &[f32], want: &[f32]| o.len() == want.len() && o.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-5);
        let v: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let o = trilinear(&v, (2, 2, 2), (1, 1, 3));
        assert!(close(&o, &[3.0, 3.5, 4.0]), "{o:?}");
        let v: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let o = trilinear(&v, (4, 2, 3), (2, 3, 2));
        assert!(close(&o, &[3.25, 4.75, 4.75, 6.25, 6.25, 7.75, 15.25, 16.75, 16.75, 18.25, 18.25, 19.75]), "{o:?}");
    }
}
