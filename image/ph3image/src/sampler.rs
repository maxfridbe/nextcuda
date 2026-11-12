//! The few-step sampler with `eta`: at each step the model's x0 estimate, then the latent put back at the next level
//! as `(1 - next) x0 + next eps`, where `eps` is the step's own noise estimate (`(x - (1 - s) x0) / s`) mixed with
//! fresh noise - `sqrt(1 - eta^2) eps_hat + eta s_noise z`.
//!
//! - eta 0: no fresh noise; the step is flow Euler (`x += (x - x0) / s (next - s)`), the deterministic sampler, run
//!   with the same arithmetic as `nextcuda_h3::denoiser::sample_from`.
//! - eta 1: all fresh noise - ComfyUI's `sample_lcm` (k_diffusion: x0, then `noise_scaling(next, z, x0)`, for a
//!   CONST model `next z + (1 - next) x0`), the sampler distribution-matching distilled models (DMD, pDMD, LCM) are
//!   trained to be stepped with: each step lands an x0 and the next starts from it under new noise.
//! - between: DDIM-style, the two noises mixed so the variance stays one.
//!
//! `s_noise` scales the fresh noise (ComfyUI's ancestral samplers' s_noise; 1 by default). The fresh draws come from
//! the request's seed: the same seed gives the same picture.

/// eta's range: 0 (deterministic) to 1 (all fresh noise); outside it is clamped
pub const ETA_MAX: f32 = 1.0;
/// s_noise's range
pub const S_NOISE_MAX: f32 = 2.0;

/// eta as the sampler uses it: clamped to 0..1 (above 1 is 1; not a number is 0)
pub fn clamp_eta(v: f32) -> f32 {
    if v.is_nan() { 0.0 } else { v.clamp(0.0, ETA_MAX) }
}

/// s_noise as the sampler uses it: clamped to 0..2 (not a number is 1)
pub fn clamp_s_noise(v: f32) -> f32 {
    if v.is_nan() { 1.0 } else { v.clamp(0.0, S_NOISE_MAX) }
}

/// One step of `x` from level `s` to `next`, given the model's x0 estimate there and fresh noise `z` (read only when
/// eta > 0 and next > 0; may be empty otherwise)
pub fn step(x: &mut [f32], x0: &[f32], s: f32, next: f32, eta: f32, s_noise: f32, z: &[f32]) {
    if eta <= 0.0 {
        // flow Euler, as sample_from steps it
        for (x, dn) in x.iter_mut().zip(x0) {
            let slope = (*x - dn) / s;
            *x += slope * (next - s);
        }
        return;
    }
    if next <= 0.0 {
        x.copy_from_slice(x0);
        return;
    }
    let keep = (1.0 - eta * eta).max(0.0).sqrt();
    let fresh = eta * s_noise;
    for ((x, x0), z) in x.iter_mut().zip(x0).zip(z) {
        let eps_hat = (*x - (1.0 - s) * x0) / s;
        let eps = keep * eps_hat + fresh * z;
        *x = (1.0 - next) * x0 + next * eps;
    }
}

/// Steps the streams `xs` (each already at `sig[0]`) down `sig`. `model(xs, s, i)`: each stream's x0 estimate at
/// level s (step i); `noise(n)`: n standard normals, drawn per stream in order at each step that adds noise;
/// `done(i)` after step i.
pub fn run<E>(
    xs: &mut [Vec<f32>],
    sig: &[f32],
    eta: f32,
    s_noise: f32,
    model: &mut dyn FnMut(&[Vec<f32>], f32, usize) -> Result<Vec<Vec<f32>>, E>,
    noise: &mut dyn FnMut(usize) -> Vec<f32>,
    done: &mut dyn FnMut(usize) -> Result<(), E>,
) -> Result<(), E> {
    let (eta, s_noise) = (clamp_eta(eta), clamp_s_noise(s_noise));
    for i in 0..sig.len().saturating_sub(1) {
        let (s, next) = (sig[i], sig[i + 1]);
        let x0 = model(xs, s, i)?;
        for (x, x0) in xs.iter_mut().zip(&x0) {
            let z = if eta > 0.0 && next > 0.0 { noise(x.len()) } else { Vec::new() };
            step(x, x0, s, next, eta, s_noise, &z);
        }
        done(i)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nextcuda_h3::noise::Mt19937;

    /// a stand-in model: x0 a fixed function of x and s
    fn toy(xs: &[Vec<f32>], s: f32, _i: usize) -> Result<Vec<Vec<f32>>, ()> {
        Ok(xs.iter().map(|x| x.iter().enumerate().map(|(j, v)| 0.3 * v * (1.0 - s) + 0.1 * (j as f32).sin()).collect()).collect())
    }

    fn start(seed: u64) -> Vec<Vec<f32>> {
        let mut g = Mt19937::new(seed);
        vec![g.randn(64), g.randn(16)]
    }

    fn sampled(seed: u64, eta: f32, s_noise: f32, sig: &[f32]) -> Vec<Vec<f32>> {
        let mut g = Mt19937::new(seed);
        let mut xs = vec![g.randn(64), g.randn(16)];
        run(&mut xs, sig, eta, s_noise, &mut toy, &mut |n| g.randn(n), &mut |_| Ok(())).unwrap();
        xs
    }

    #[test]
    fn eta_zero_is_the_old_euler_exactly() {
        let sig = [1.0, 0.923_076_9, 0.8, 0.0];
        // sample_from's loop, as it was
        let mut old = start(3);
        for i in 0..sig.len() - 1 {
            let (s, next) = (sig[i], sig[i + 1]);
            let dn = toy(&old, s, i).unwrap();
            for (xv, dv) in old.iter_mut().zip(&dn) {
                for (x, dn) in xv.iter_mut().zip(dv) {
                    let slope = (*x - dn) / s;
                    *x += slope * (next - s);
                }
            }
        }
        let mut xs = start(3);
        let mut drew = 0;
        run(&mut xs, &sig, 0.0, 1.0, &mut toy, &mut |n| { drew += n; vec![0.0; n] }, &mut |_| Ok(())).unwrap();
        assert_eq!(drew, 0, "eta 0 draws no noise");
        for (a, b) in xs.iter().flatten().zip(old.iter().flatten()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        // and a negative eta is eta 0
        assert_eq!(sampled(3, -0.5, 1.0, &sig), sampled(3, 0.0, 1.0, &sig));
    }

    #[test]
    fn eta_is_clamped() {
        assert_eq!(clamp_eta(1.7), 1.0);
        assert_eq!(clamp_eta(-1.0), 0.0);
        assert_eq!(clamp_eta(f32::NAN), 0.0);
        assert_eq!(clamp_eta(0.4), 0.4);
        assert_eq!(clamp_s_noise(5.0), 2.0);
        assert_eq!(clamp_s_noise(f32::NAN), 1.0);
        let sig = [1.0, 0.923_076_9, 0.0];
        // above 1 samples as 1
        assert_eq!(sampled(5, 3.0, 1.0, &sig), sampled(5, 1.0, 1.0, &sig));
    }

    #[test]
    fn eta_one_is_lcm() {
        // x0, then the next level from x0 and fresh noise alone: (1 - next) x0 + next s_noise z
        let (s, next) = (0.6, 0.25);
        let x0 = vec![0.5, -1.0, 2.0];
        let z = vec![0.3, 1.2, -0.7];
        for x in [vec![0.0, 0.0, 0.0], vec![9.0, -3.0, 1.0]] {
            let mut x = x;
            step(&mut x, &x0, s, next, 1.0, 0.9, &z);
            for j in 0..3 {
                assert!((x[j] - (0.75 * x0[j] + 0.25 * 0.9 * z[j])).abs() < 1e-6, "{x:?}");
            }
        }
        // the last step lands x0
        let mut x = vec![4.0, 4.0, 4.0];
        step(&mut x, &x0, s, 0.0, 1.0, 1.0, &[]);
        assert_eq!(x, x0);
    }

    #[test]
    fn the_noise_comes_from_the_seed() {
        let sig = [1.0, 0.923_076_9, 0.6, 0.0];
        assert_eq!(sampled(11, 1.0, 1.0, &sig), sampled(11, 1.0, 1.0, &sig));
        assert_eq!(sampled(11, 0.5, 1.1, &sig), sampled(11, 0.5, 1.1, &sig));
        assert_ne!(sampled(11, 1.0, 1.0, &sig), sampled(12, 1.0, 1.0, &sig));
        // and it is fresh: eta 1 is not eta 0
        assert_ne!(sampled(11, 1.0, 1.0, &sig), sampled(11, 0.0, 1.0, &sig));
    }
}
