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
    /// four passes a step combined by the guidance (the dev DiT); false: the prompt's pass alone, its x0 as it is
    /// (the distilled DiT, which is trained to run without guidance)
    pub guided: bool,
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
        for (v, a) in d.forward_passes(prep, lat, aud, if h.guided { &passes } else { &passes[..1] })? {
            xv.push(lat.iter().zip(&v).map(|(x, v)| x - v * s).collect());
            xa.push(aud.iter().zip(&a).map(|(x, v)| x - v * s).collect());
        }
        let (pv, pa) = if h.guided {
            (combine(&crate::sample::VIDEO, &xv[0], &xv[1], &xv[2], &xv[3]), combine(&crate::sample::AUDIO, &xa[0], &xa[1], &xa[2], &xa[3]))
        } else {
            (xv.swap_remove(0), xa.swap_remove(0))
        };
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
        let prof = crate::dit::take_profile();
        if !prof.is_empty() {
            eprintln!("profile step {}: {}", i + 1, prof.iter().map(|(k, v)| format!("{k} {v:.2}s")).collect::<Vec<_>>().join(", "));
        }
        (h.progress)(i + 1, steps);
        if let Some(e) = h.each.as_deref_mut() {
            e(i + 1, lat, aud);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_dev_schedule_is_linear_from_one_to_one_over_n_then_zero() {
        let s = sigmas(30, false);
        assert_eq!(s.len(), 31);
        assert_eq!(s[0], 1.0);
        assert!((s[29] - 1.0 / 30.0).abs() < 1e-6, "the last step's sigma is 1/N");
        assert_eq!(s[30], 0.0);
        let d: Vec<f32> = s.windows(2).take(29).map(|w| w[0] - w[1]).collect();
        assert!(d.iter().all(|x| (x - d[0]).abs() < 1e-5), "even steps");
        assert!((d[0] - (1.0 - 1.0 / 30.0) / 29.0).abs() < 1e-6);
        // the other counts
        assert_eq!(sigmas(1, false), vec![1.0, 0.0]);
        assert_eq!(sigmas(0, false), vec![1.0, 0.0], "none asked: one step");
        let s12 = sigmas(12, false);
        assert_eq!((s12.len(), s12[0], s12[12]), (13, 1.0, 0.0));
        assert!((s12[11] - 1.0 / 12.0).abs() < 1e-6);
        assert!(sigmas(50, false).windows(2).all(|w| w[0] > w[1]), "strictly falling");
        assert_eq!(DEV_STEPS, 30);
    }

    #[test]
    fn the_distilled_schedule_is_its_own_eight_whatever_is_asked() {
        for asked in [0, 4, 8, 30] {
            let s = sigmas(asked, true);
            assert_eq!(s.len(), 9);
            assert_eq!(&s[..8], &DISTILLED);
            assert_eq!(s[8], 0.0);
        }
        // five of the eight sit in the top 2.5% of the noise range, as the Python engine's note says
        assert_eq!(DISTILLED.iter().filter(|x| **x >= 0.975).count(), 5);
        assert!(DISTILLED.windows(2).all(|w| w[0] > w[1]));
    }

    fn pass(n: usize, f: impl Fn(usize) -> f32) -> Vec<f32> {
        (0..n).map(f).collect()
    }

    #[test]
    fn guidance_with_nothing_to_steer_is_the_conditional_x0() {
        let c = pass(64, |i| (i as f32 * 0.37).sin() * 3.0);
        // all four passes agree: the delta terms vanish, the rescale is the identity
        let p = combine(&VIDEO, &c, &c, &c, &c);
        assert!(p.iter().zip(&c).all(|(a, b)| (a - b).abs() < 1e-5));
        // scales of 1 / 0 / 1: the same whatever the other passes say
        let g = Guide { cfg: 1.0, stg: 0.0, modality: 1.0, rescale: 0.0 };
        let u = pass(64, |i| i as f32);
        assert_eq!(combine(&g, &c, &u, &u, &u), c);
    }

    #[test]
    fn guidance_combines_the_passes_in_delta_form() {
        // no rescale: x0 = c + (cfg - 1)(c - u) + stg (c - s) + (mod - 1)(c - m), element by element
        let g = Guide { cfg: 3.0, stg: 1.0, modality: 3.0, rescale: 0.0 };
        let (c, u, s, m) = ([1.0f32, 2.0], [0.5f32, 1.0], [0.0f32, 4.0], [0.25f32, 2.5]);
        let p = combine(&g, &c, &u, &s, &m);
        for i in 0..2 {
            let want = c[i] + 2.0 * (c[i] - u[i]) + 1.0 * (c[i] - s[i]) + 2.0 * (c[i] - m[i]);
            assert!((p[i] - want).abs() < 1e-6, "{i}: {} vs {want}", p[i]);
        }
        // the audio guide's scales
        assert_eq!((AUDIO.cfg, AUDIO.stg, AUDIO.modality, AUDIO.rescale), (7.0, 1.0, 3.0, 0.7));
        assert_eq!((VIDEO.cfg, VIDEO.stg, VIDEO.modality, VIDEO.rescale), (3.0, 1.0, 3.0, 0.7));
    }

    #[test]
    fn the_rescale_pulls_the_spread_back_toward_the_conditional_one() {
        let c = pass(256, |i| (i as f32 * 0.21).sin());
        let u = pass(256, |i| (i as f32 * 0.21).sin() * 0.3);
        let s = u.clone();
        let m = u.clone();
        let raw = combine(&Guide { rescale: 0.0, ..VIDEO }, &c, &u, &s, &m);
        let full = combine(&Guide { rescale: 1.0, ..VIDEO }, &c, &u, &s, &m);
        let mix = combine(&VIDEO, &c, &u, &s, &m);
        let (sc, sr, sf, sm) = (std(&c), std(&raw), std(&full), std(&mix));
        assert!(sr > sc * 1.5, "guidance widens the spread: {sr} vs {sc}");
        assert!((sf - sc).abs() / sc < 1e-4, "a full rescale matches the conditional std: {sf} vs {sc}");
        // 0.7 of the rescaled and 0.3 of the raw: the std between them, both being the same shape
        assert!((sm - (0.7 * sf + 0.3 * sr)).abs() / sm < 1e-3, "{sm}");
    }

    #[test]
    fn the_std_is_the_sample_one() {
        assert!((std(&[1.0, 2.0, 3.0, 4.0]) - 1.2909944).abs() < 1e-6, "n - 1 in the divisor");
        assert_eq!(std(&[5.0]), 0.0);
    }
}
