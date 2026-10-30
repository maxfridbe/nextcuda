//! The sound: the audio VAE's decoder (diffusers' AutoencoderKLLTX2Audio: latents [8, La, 16] to a stereo log-mel
//! [2, 4 La - 3, 64]) and the vocoder with bandwidth extension (LTX2VocoderWithBWE: the mel to 16 kHz, its own mel at
//! 200 frames a second, a second vocoder for the residual over a x3 resampling - 48 kHz).
//!
//! ```text
//!   decoder (2-D over time x mel, causal in time: padded 2 in front, none behind; pixel RMS norms, eps 1e-6):
//!     conv_in 8 -> 512; mid: 2 resnets; up 2: 3 resnets 512, x2 nearest + conv (its first row dropped);
//!     up 1: 3 resnets -> 256, x2; up 0: 3 resnets -> 128; rms, silu, conv_out -> 2; cut to 4 La - 3 frames
//!   vocoder (1-D, float32): the 2 x 64 mel rows as 128 channels; conv 7 -> hidden; per stage a transposed conv
//!     (x5 x2 x2 x2 x2 x2; hidden 1536 halving) and the mean of three residual stacks (kernels 3, 7, 11, dilations
//!     1, 3, 5, anti-aliased SnakeBeta); act, conv 7 -> 2
//!   BWE: the 16 kHz wave padded to whole hops of 80, its causal STFT (512) and mel (64), the BWE vocoder (hidden
//!     512, x6 x5 x2 x2 x2) + the wave resampled x3 (a Hann-windowed sinc, 43 taps), clamped to [-1, 1]
//! ```

use std::sync::Arc;

use nextcuda_core::{DevBuf, Error, Gpu, Result};
use nextcuda_diffusion::kernels::{none, Dt, Nsd};
use nextcuda_gguf::safetensors::SafeTensors;

use crate::ffi;
use crate::vae::{Ctx, Vol};

fn ge(e: nextcuda_gguf::Error) -> Error {
    Error(e.0)
}

pub const SAMPLE_RATE: u32 = 48_000;

/// A 2-D conv (k x k), its weight half [Co, Ci, k, k]
struct Conv2 {
    w: DevBuf,
    b: DevBuf,
    ci: usize,
    co: usize,
    k: usize,
}

impl Conv2 {
    fn load(st: &SafeTensors, p: &str, gpu: &Arc<Gpu>) -> Result<Conv2> {
        let t = st.need(&format!("{p}.weight")).map_err(ge)?;
        let (co, ci, k) = (t.shape[0] as usize, t.shape[1] as usize, t.shape[2] as usize);
        let w: Vec<u8> = st.f32(&format!("{p}.weight")).map_err(ge)?.iter().flat_map(|v| crate::quant::f16_bits(*v).to_le_bytes()).collect();
        let wd = DevBuf::new(gpu, w.len())?;
        wd.write(0, &w)?;
        Ok(Conv2 { w: wd, b: DevBuf::from_f32(gpu, &st.f32(&format!("{p}.bias")).map_err(ge)?)?, ci, co, k })
    }
}

struct Res2 {
    c1: Conv2,
    c2: Conv2,
    nin: Option<Conv2>,
}

pub struct AudioDecoder {
    gpu: Arc<Gpu>,
    nsd: Nsd,
    conv_in: Conv2,
    mid: Vec<Res2>,
    /// levels 2, 1, 0 in the order they run: (resnets, the up-sampler's conv)
    ups: Vec<(Vec<Res2>, Option<Conv2>)>,
    conv_out: Conv2,
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

/// A causal 2-D conv over a channels-last plane [t, m, c] (t the time axis): a zero row in front, the conv zero-padded
/// by k/2, the last row dropped - the window of row t covers rows t - 2 ..= t
fn conv2(c: &Ctx, cv: &Conv2, x: &Vol) -> Result<Vol> {
    let (t, m) = (x.h, x.w);
    if cv.k == 1 {
        let o = c.vol(1, t, m, cv.co)?;
        c.nsd.conv2d(x.d.ptr(), Dt::F16, 1, t, m, cv.ci, cv.w.ptr(), cv.co, 1, cv.b.ptr(), o.d.ptr())?;
        return Ok(o);
    }
    let pad = cv.k - 1 - cv.k / 2;
    let row = m * x.c * 2;
    let ext = DevBuf::new(c.gpu, (t + pad) * row)?;
    ext.view(0, pad * row)?.fill(0)?;
    ext.copy_within(pad * row, &x.d, 0, t * row)?;
    let o = DevBuf::new(c.gpu, (t + pad) * m * cv.co * 2)?;
    c.nsd.conv2d(ext.ptr(), Dt::F16, 1, t + pad, m, cv.ci, cv.w.ptr(), cv.co, cv.k, cv.b.ptr(), o.ptr())?;
    let out = c.vol(1, t, m, cv.co)?;
    out.d.copy_within(0, &o, 0, t * m * cv.co * 2)?;
    Ok(out)
}

impl Res2 {
    fn load(st: &SafeTensors, p: &str, gpu: &Arc<Gpu>) -> Result<Res2> {
        let nin = if st.tensor(&format!("{p}.nin_shortcut.conv.weight")).is_some() { Some(Conv2::load(st, &format!("{p}.nin_shortcut.conv"), gpu)?) } else { None };
        Ok(Res2 { c1: Conv2::load(st, &format!("{p}.conv1.conv"), gpu)?, c2: Conv2::load(st, &format!("{p}.conv2.conv"), gpu)?, nin })
    }
    fn run(&self, c: &Ctx, x: Vol) -> Result<Vol> {
        let h = conv2(c, &self.c1, &rms(c, &x)?)?;
        let h = conv2(c, &self.c2, &rms(c, &h)?)?;
        let x = match &self.nin {
            Some(n) => conv2(c, n, &x)?,
            None => x,
        };
        c.add(&x, &h)?;
        Ok(x)
    }
}

fn rms(c: &Ctx, x: &Vol) -> Result<Vol> {
    let o = c.vol(1, x.h, x.w, x.c)?;
    ffi::rms_silu(c.gpu, x.d.ptr(), x.h * x.w, x.c, 1e-6, true, o.d.ptr())?;
    Ok(o)
}

impl AudioDecoder {
    pub fn load(path: &std::path::Path, gpu: &Arc<Gpu>) -> Result<AudioDecoder> {
        let st = SafeTensors::open(path).map_err(ge)?;
        let mid = vec![Res2::load(&st, "decoder.mid.block_1", gpu)?, Res2::load(&st, "decoder.mid.block_2", gpu)?];
        let mut ups = Vec::new();
        for level in [2usize, 1, 0] {
            let res = (0..3).map(|i| Res2::load(&st, &format!("decoder.up.{level}.block.{i}"), gpu)).collect::<Result<Vec<_>>>()?;
            let up = if level != 0 { Some(Conv2::load(&st, &format!("decoder.up.{level}.upsample.conv.conv"), gpu)?) } else { None };
            ups.push((res, up));
        }
        Ok(AudioDecoder {
            conv_in: Conv2::load(&st, "decoder.conv_in.conv", gpu)?,
            conv_out: Conv2::load(&st, "decoder.conv_out.conv", gpu)?,
            mean: st.f32("latents_mean").map_err(ge)?,
            std: st.f32("latents_std").map_err(ge)?,
            mid,
            ups,
            nsd: Nsd::new(gpu)?,
            gpu: gpu.clone(),
        })
    }

    /// The DiT's audio latents [La, 128] (normalized, packed: channel c, mel bin m at c * 16 + m) -> the log-mel
    /// [2][4 La - 3][64]
    pub fn decode(&self, aud: &[f32], la: usize) -> Result<Vec<f32>> {
        let c = Ctx { gpu: &self.gpu, nsd: &self.nsd };
        // denormalized (per packed feature), into channels-last [La, 16, 8]
        let mut z = vec![0f32; la * 128];
        for t in 0..la {
            for ch in 0..8 {
                for m in 0..16 {
                    let f = ch * 16 + m;
                    z[(t * 16 + m) * 8 + ch] = aud[t * 128 + f] * self.std[f] + self.mean[f];
                }
            }
        }
        let x = c.vol(1, la, 16, 8)?;
        x.d.write(0, &z.iter().flat_map(|v| crate::quant::f16_bits(*v).to_le_bytes()).collect::<Vec<u8>>())?;
        let mut x = conv2(&c, &self.conv_in, &x)?;
        for r in &self.mid {
            x = r.run(&c, x)?;
        }
        for (res, up) in &self.ups {
            for r in res {
                x = r.run(&c, x)?;
            }
            if let Some(u) = up {
                let o = c.vol(1, 2 * x.h, 2 * x.w, x.c)?;
                ffi::up2(&self.gpu, x.d.ptr(), x.h, x.w, x.c, o.d.ptr())?;
                let y = conv2(&c, u, &o)?;
                // the first time row dropped
                let row = y.w * y.c * 2;
                let d = c.vol(1, y.h - 1, y.w, y.c)?;
                d.d.copy_within(0, &y.d, row, (y.h - 1) * row)?;
                x = d;
            }
        }
        let y = conv2(&c, &self.conv_out, &rms(&c, &x)?)?;
        let n = y.h * y.w * y.c;
        let tmp = DevBuf::f32(&self.gpu, n)?;
        ffi::to_float(&self.gpu, y.d.ptr(), tmp.fp(), n, 1.0)?;
        self.nsd.wait()?;
        let v = tmp.to_f32()?;
        let tt = 4 * la - 3;
        let mut mel = vec![0f32; 2 * tt * 64];
        for ch in 0..2 {
            for t in 0..tt.min(y.h) {
                for m in 0..64.min(y.w) {
                    mel[(ch * tt + t) * 64 + m] = v[(t * y.w + m) * y.c + ch];
                }
            }
        }
        Ok(mel)
    }
}

// ---- the vocoder ---------------------------------------------------------------------------------------------------

struct Conv1 {
    w: DevBuf,
    b: Option<DevBuf>,
    /// [Co, Ci, K] (a transposed one: [Ci, Co, K])
    ci: usize,
    co: usize,
    k: usize,
}

impl Conv1 {
    fn load(st: &SafeTensors, p: &str, gpu: &Arc<Gpu>, transposed: bool) -> Result<Conv1> {
        let t = st.need(&format!("{p}.weight")).map_err(ge)?;
        let (a, c, k) = (t.shape[0] as usize, t.shape[1] as usize, t.shape[2] as usize);
        let (co, ci) = if transposed { (c, a) } else { (a, c) };
        let b = if st.tensor(&format!("{p}.bias")).is_some() { Some(DevBuf::from_f32(gpu, &st.f32(&format!("{p}.bias")).map_err(ge)?)?) } else { None };
        Ok(Conv1 { w: DevBuf::from_f32(gpu, &st.f32(&format!("{p}.weight")).map_err(ge)?)?, b, ci, co, k })
    }
}

struct Act {
    alpha: DevBuf,
    beta: DevBuf,
    up: DevBuf,
    down: DevBuf,
}

impl Act {
    fn load(st: &SafeTensors, p: &str, gpu: &Arc<Gpu>) -> Result<Act> {
        let v = |s: &str| -> Result<DevBuf> { DevBuf::from_f32(gpu, &st.f32(&format!("{p}.{s}")).map_err(ge)?) };
        Ok(Act { alpha: v("act.alpha")?, beta: v("act.beta")?, up: v("upsample.filter")?, down: v("downsample.filter")? })
    }
}

struct Stack {
    k: usize,
    c1: Vec<Conv1>,
    c2: Vec<Conv1>,
    a1: Vec<Act>,
    a2: Vec<Act>,
}

/// One LTX2Vocoder (the stage-1 one, or the BWE generator)
struct Vocoder {
    conv_in: Conv1,
    ups: Vec<(Conv1, usize)>,
    stacks: Vec<Stack>,
    act_out: Act,
    conv_out: Conv1,
}

const DIL: [usize; 3] = [1, 3, 5];
const KERNELS: [usize; 3] = [3, 7, 11];

impl Vocoder {
    fn load(st: &SafeTensors, p: &str, factors: &[usize], gpu: &Arc<Gpu>) -> Result<Vocoder> {
        let mut ups = Vec::new();
        let mut stacks = Vec::new();
        for (i, f) in factors.iter().enumerate() {
            ups.push((Conv1::load(st, &format!("{p}.upsamplers.{i}"), gpu, true)?, *f));
            for (j, k) in KERNELS.iter().enumerate() {
                let r = format!("{p}.resnets.{}", i * 3 + j);
                stacks.push(Stack {
                    k: *k,
                    c1: (0..3).map(|d| Conv1::load(st, &format!("{r}.convs1.{d}"), gpu, false)).collect::<Result<_>>()?,
                    c2: (0..3).map(|d| Conv1::load(st, &format!("{r}.convs2.{d}"), gpu, false)).collect::<Result<_>>()?,
                    a1: (0..3).map(|d| Act::load(st, &format!("{r}.acts1.{d}"), gpu)).collect::<Result<_>>()?,
                    a2: (0..3).map(|d| Act::load(st, &format!("{r}.acts2.{d}"), gpu)).collect::<Result<_>>()?,
                });
            }
        }
        Ok(Vocoder {
            conv_in: Conv1::load(st, &format!("{p}.conv_in"), gpu, false)?,
            ups,
            stacks,
            act_out: Act::load(st, &format!("{p}.act_out"), gpu)?,
            conv_out: Conv1::load(st, &format!("{p}.conv_out"), gpu, false)?,
        })
    }
}

/// A float32 signal [c, l] on the card
struct Sig {
    d: DevBuf,
    c: usize,
    l: usize,
}

pub struct Audio {
    gpu: Arc<Gpu>,
    nsd: Nsd,
    voc: Vocoder,
    bwe: Vocoder,
    stft: DevBuf,
    mel_basis: Vec<f32>,
}

impl Audio {
    pub fn load(path: &std::path::Path, gpu: &Arc<Gpu>) -> Result<Audio> {
        let st = SafeTensors::open(path).map_err(ge)?;
        Ok(Audio {
            voc: Vocoder::load(&st, "vocoder", &[5, 2, 2, 2, 2, 2], gpu)?,
            bwe: Vocoder::load(&st, "bwe_generator", &[6, 5, 2, 2, 2], gpu)?,
            stft: DevBuf::from_f32(gpu, &st.f32("mel_stft.stft_fn.forward_basis").map_err(ge)?)?,
            mel_basis: st.f32("mel_stft.mel_basis").map_err(ge)?,
            nsd: Nsd::new(gpu)?,
            gpu: gpu.clone(),
        })
    }

    fn sig(&self, c: usize, l: usize) -> Result<Sig> {
        Ok(Sig { d: DevBuf::f32(&self.gpu, c * l)?, c, l })
    }

    fn conv(&self, x: &Sig, cv: &Conv1, dil: usize, pad: usize) -> Result<Sig> {
        let lo = x.l + 2 * pad - dil * (cv.k - 1);
        let y = self.sig(cv.co, lo)?;
        self.nsd.conv1d(x.d.ptr(), 1, x.c, x.l, cv.w.ptr(), cv.co, cv.k, cv.b.as_ref().map_or(none(), |b| b.ptr()), 1, dil, pad, y.d.ptr(), lo)?;
        Ok(y)
    }

    fn act(&self, x: &Sig, a: &Act) -> Result<Sig> {
        let y = self.sig(x.c, x.l)?;
        ffi::aa_snake(self.nsd.raw(), x.d.fp(), 1, x.c, x.l, a.alpha.fp(), a.beta.fp(), a.up.fp(), a.down.fp(), y.d.fp())?;
        Ok(y)
    }

    fn add(&self, x: &Sig, y: &Sig) -> Result<()> {
        self.nsd.gate_add(x.d.ptr(), Dt::F32, x.c, x.l, y.d.ptr(), Dt::F32, none(), none())
    }

    fn run(&self, v: &Vocoder, x: Sig) -> Result<Sig> {
        let mut x = self.conv(&x, &v.conv_in, 1, (v.conv_in.k - 1) / 2)?;
        for (i, (up, s)) in v.ups.iter().enumerate() {
            let pad = (up.k - s) / 2;
            let lo = (x.l - 1) * s + up.k - 2 * pad;
            let y = self.sig(up.co, lo)?;
            self.nsd.conv_transpose1d(x.d.ptr(), 1, x.c, x.l, up.w.ptr(), up.co, up.k, up.b.as_ref().map_or(none(), |b| b.ptr()), *s, pad, y.d.ptr(), lo)?;
            let acc = self.sig(y.c, y.l)?;
            acc.d.fill(0)?;
            for st in &v.stacks[i * 3..i * 3 + 3] {
                let mut cur = self.sig(y.c, y.l)?;
                cur.d.copy_within(0, &y.d, 0, y.d.len)?;
                for j in 0..3 {
                    let t = self.act(&cur, &st.a1[j])?;
                    let t = self.conv(&t, &st.c1[j], DIL[j], DIL[j] * (st.k - 1) / 2)?;
                    let t = self.act(&t, &st.a2[j])?;
                    let t = self.conv(&t, &st.c2[j], 1, (st.k - 1) / 2)?;
                    self.add(&t, &cur)?;
                    cur = t;
                }
                self.add(&acc, &cur)?;
            }
            self.nsd.scale(acc.d.ptr(), acc.c * acc.l, 1.0 / 3.0)?;
            x = acc;
        }
        let x = self.act(&x, &v.act_out)?;
        self.conv(&x, &v.conv_out, 1, (v.conv_out.k - 1) / 2)
    }

    /// The log-mel [2][T][64] -> stereo samples at 48 kHz ([2][480 T])
    pub fn vocode(&self, mel: &[f32], t: usize) -> Result<Vec<Vec<f32>>> {
        // the vocoder's input: channel c, bin m as row c * 64 + m
        let mut x = vec![0f32; 128 * t];
        for c in 0..2 {
            for ti in 0..t {
                for m in 0..64 {
                    x[(c * 64 + m) * t + ti] = mel[(c * t + ti) * 64 + m];
                }
            }
        }
        let s = Sig { d: DevBuf::from_f32(&self.gpu, &x)?, c: 128, l: t };
        let w = self.run(&self.voc, s)?;
        self.nsd.wait()?;
        let n = w.l;
        let wave = w.d.to_f32()?;
        // padded to whole hops, its causal STFT on the card (432 zeros in front), the mel on the host
        let np = n.div_ceil(80) * 80;
        let left = 512 - 80;
        let mut padded = vec![0f32; 2 * (left + np)];
        for c in 0..2 {
            padded[c * (left + np) + left..c * (left + np) + left + n].copy_from_slice(&wave[c * n..(c + 1) * n]);
        }
        let frames = (left + np - 512) / 80 + 1;
        let xs = DevBuf::from_f32(&self.gpu, &padded)?;
        let spec = DevBuf::f32(&self.gpu, 2 * 514 * frames)?;
        self.nsd.conv1d(xs.ptr(), 2, 1, left + np, self.stft.ptr(), 514, 512, none(), 80, 1, 0, spec.ptr(), frames)?;
        self.nsd.wait()?;
        let sp = spec.to_f32()?;
        let mut bmel = vec![0f32; 128 * frames];
        let mut mag = vec![0f32; 257];
        for c in 0..2 {
            for f in 0..frames {
                for k in 0..257 {
                    let re = sp[(c * 514 + k) * frames + f];
                    let im = sp[(c * 514 + 257 + k) * frames + f];
                    mag[k] = (re * re + im * im).sqrt();
                }
                for m in 0..64 {
                    let v: f32 = (0..257).map(|k| self.mel_basis[m * 257 + k] * mag[k]).sum();
                    bmel[(c * 64 + m) * frames + f] = v.max(1e-5).ln();
                }
            }
        }
        let r = self.run(&self.bwe, Sig { d: DevBuf::from_f32(&self.gpu, &bmel)?, c: 128, l: frames })?;
        self.nsd.wait()?;
        let res = r.d.to_f32()?;
        let skip = resample3(&padded, left, np);
        let out_n = n * 3;
        Ok((0..2).map(|c| (0..out_n).map(|i| (res[c * r.l + i] + skip[c][i]).clamp(-1.0, 1.0)).collect()).collect())
    }
}

/// UpSample1d(ratio 3, window "hann"): each channel replicate-padded by 7, a transposed conv (stride 3) with the
/// 43-tap Hann-windowed sinc, times 3, the first 42 and last 40 samples dropped
fn resample3(padded: &[f32], left: usize, np: usize) -> Vec<Vec<f32>> {
    let (ratio, width) = (3usize, 7usize);
    let k = 2 * width * ratio + 1;
    let filt: Vec<f32> = (0..k).map(|i| {
        let t = (i as f32 / ratio as f32 - width as f32) * 0.99;
        let tc = t.clamp(-6.0, 6.0);
        let w = (tc * std::f32::consts::PI / 6.0 / 2.0).cos().powi(2);
        let s = if t == 0.0 { 1.0 } else { (std::f32::consts::PI * t).sin() / (std::f32::consts::PI * t) };
        s * w * 0.99 / ratio as f32
    }).collect();
    (0..2).map(|c| {
        let x = &padded[c * (left + np) + left..c * (left + np) + left + np];
        let xp: Vec<f32> = (0..np + 2 * width).map(|i| x[(i as isize - width as isize).clamp(0, np as isize - 1) as usize]).collect();
        let full = (xp.len() - 1) * ratio + k;
        let mut y = vec![0f32; full];
        for (p, v) in xp.iter().enumerate() {
            for (j, f) in filt.iter().enumerate() {
                y[p * ratio + j] += v * f;
            }
        }
        y[2 * width * ratio..full - (k - ratio)].iter().map(|v| v * ratio as f32).collect()
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// two channels of `np` samples with `left` samples of padding before each, as `resample3` is given them
    fn padded(ch0: &[f32], ch1: &[f32], left: usize) -> Vec<f32> {
        let np = ch0.len();
        let mut v = vec![0f32; 2 * (left + np)];
        v[left..left + np].copy_from_slice(ch0);
        v[left + np + left..].copy_from_slice(ch1);
        v
    }

    #[test]
    fn the_three_times_resampler_gives_three_samples_for_each() {
        let x = vec![0.5f32; 40];
        let y = resample3(&padded(&x, &x, 7), 7, 40);
        assert_eq!(y.len(), 2);
        assert_eq!(y[0].len(), 120);
        // a constant (replicate-padded at the ends) stays near 0.99 of itself: the filter's gain is 0.99
        assert!(y[0][30..90].iter().all(|v| (v - 0.5 * 0.99).abs() < 0.02), "{:?}", &y[0][30..40]);
    }

    #[test]
    fn the_resampler_keeps_the_channels_apart_and_a_tone_in_place() {
        let n = 200;
        let tone: Vec<f32> = (0..n).map(|i| (i as f32 * 0.2).sin()).collect();
        let quiet = vec![0.0f32; n];
        let y = resample3(&padded(&tone, &quiet, 7), 7, n);
        assert!(y[1].iter().all(|v| v.abs() < 1e-6), "the second channel is silent");
        // the tone at three times the rate: sample 3 i of the output is the input's sample i (the sinc is 1 at its centre)
        for i in 30..170 {
            assert!((y[0][3 * i] - 0.99 * tone[i]).abs() < 0.03, "{i}: {} vs {}", y[0][3 * i], tone[i]);
        }
    }

    #[test]
    fn the_vocoders_output_is_48_khz() {
        assert_eq!(SAMPLE_RATE, 48_000);
        assert_eq!((DIL, KERNELS), ([1, 3, 5], [3, 7, 11]));
    }
}
