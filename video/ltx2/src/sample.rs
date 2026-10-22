//! The sampler the movie maker's pipeline runs (diffusers' modular LTX-2 denoise loop, LTX25AutoBlocks): flow-match
//! Euler over sigmas linspace(1, 1/N, N) then 0 (shift 1, no dynamic shift) - or the distilled DiT's eight - with four
//! passes a step, combined per modality in x0 space (LTX2Guidance's delta form, then rescale):
//!
//! ```text
//!   x0_p = latents - v_p sigma                     p in cond, uncond (the empty prompt), stg (block 28), mod (isolated)
//!   x0   = c + (cfg - 1)(c - u) + stg (c - s) + (mod - 1)(c - m)      video 3 / 1 / 3, audio 7 / 1 / 3
//!   x0   = 0.7 x0 std(c) / std(x0) + 0.3 x0                          (guidance_rescale 0.7)
//!   latents += (sigma_next - sigma) (latents - x0) / sigma
//! ```
//!
//! Image to video: the first latent frame is the picture's, at timestep 0, and is never stepped.

use nextcuda_core::Result;

use crate::dit::{Dit, Pass, Prepared, Text, CH};

pub const DEV_STEPS: usize = 30;
/// diffusers.pipelines.ltx2.utils.DISTILLED_SIGMA_VALUES
pub const DISTILLED: [f32; 8] = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875];

pub struct Guide {
    pub cfg: f32,
    pub stg: f32,
    pub modality: f32,
    pub rescale: f32,
}

pub const VIDEO: Guide = Guide { cfg: 3.0, stg: 1.0, modality: 3.0, rescale: 0.7 };
pub const AUDIO: Guide = Guide { cfg: 7.0, stg: 1.0, modality: 3.0, rescale: 0.7 };

/// The schedule: N sigmas (float32 as numpy's linspace then the scheduler's cast) and a trailing 0
pub fn sigmas(steps: usize, fast: bool) -> Vec<f32> {
    let mut s: Vec<f32> = if fast {
        DISTILLED.to_vec()
    } else {
        let n = steps.max(1);
        (0..n).map(|i| if n == 1 { 1.0 } else { (1.0 + (1.0 / n as f64 - 1.0) * i as f64 / (n - 1) as f64) as f32 }).collect()
    };
    s.push(0.0);
    s
}

fn std(v: &[f32]) -> f64 {
    let n = v.len() as f64;
    let m = v.iter().map(|x| *x as f64).sum::<f64>() / n;
    (v.iter().map(|x| (*x as f64 - m).powi(2)).sum::<f64>() / (n - 1.0).max(1.0)).sqrt()
}

/// The guided x0 from the four passes' x0 (each [n]): delta form, then the rescale
pub fn combine(g: &Guide, c: &[f32], u: &[f32], s: &[f32], m: &[f32]) -> Vec<f32> {
    let mut p: Vec<f32> = (0..c.len()).map(|i| c[i] + (g.cfg - 1.0) * (c[i] - u[i]) + g.stg * (c[i] - s[i]) + (g.modality - 1.0) * (c[i] - m[i])).collect();
    if g.rescale > 0.0 {
        let f = (std(c) / std(&p).max(1e-30)) as f32;
        for x in &mut p {
            *x = g.rescale * (*x * f) + (1.0 - g.rescale) * *x;
        }
    }
    p
}

/// What the loop reports and asks
pub struct Hooks<'a> {
    pub progress: &'a mut dyn FnMut(usize, usize),
    pub cancel: &'a dyn Fn() -> bool,
    /// the latents after each step (a check)
    pub each: Option<&'a mut dyn FnMut(usize, &[f32], &[f32])>,
}

/// Denoise `lat` [Tv, 128] / `aud` [La, 128] in place over `sig`; `clean`: the first frame's tokens are kept
#[allow(clippy::too_many_arguments)]
pub fn denoise(d: &Dit, prep: &Prepared, cond: &Text, neg: &Text, lat: &mut [f32], aud: &mut [f32], sig: &[f32], clean: usize, h: &mut Hooks) -> Result<()> {
    let steps = sig.len() - 1;
    let passes = [Pass { text: cond, stg: false, isolate: false }, Pass { text: neg, stg: false, isolate: false }, Pass { text: cond, stg: true, isolate: false },
                  Pass { text: cond, stg: false, isolate: true }];
    for i in 0..steps {
        if (h.cancel)() {
            return Err(nextcuda_core::Error("cancelled".into()));
        }
        let (s, sn) = (sig[i], sig[i + 1]);
        d.set_step(prep, s * 1000.0)?;
        let mut xv: Vec<Vec<f32>> = Vec::new();
        let mut xa: Vec<Vec<f32>> = Vec::new();
        for p in &passes {
            let (v, a, _) = d.forward(prep, lat, aud, p, &[])?;
            xv.push(lat.iter().zip(&v).map(|(x, v)| x - v * s).collect());
            xa.push(aud.iter().zip(&a).map(|(x, v)| x - v * s).collect());
        }
        let pv = combine(&crate::sample::VIDEO, &xv[0], &xv[1], &xv[2], &xv[3]);
        let pa = combine(&crate::sample::AUDIO, &xa[0], &xa[1], &xa[2], &xa[3]);
        for (j, x) in lat.iter_mut().enumerate() {
            if j / CH >= clean {
                let v = (*x - pv[j]) / s;
                *x += (sn - s) * v;
            }
        }
        for (j, x) in aud.iter_mut().enumerate() {
            let v = (*x - pa[j]) / s;
            *x += (sn - s) * v;
        }
        (h.progress)(i + 1, steps);
        if let Some(e) = h.each.as_deref_mut() {
            e(i + 1, lat, aud);
        }
    }
    Ok(())
}
