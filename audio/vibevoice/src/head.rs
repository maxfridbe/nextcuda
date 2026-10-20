//! The diffusion head (yvoice's VibeVoiceDiffusionHead) and its sampler: a frame's 64-d acoustic latent from the
//! language model's hidden state, by v-prediction over 1,000 cosine-scheduled steps sampled with DPM-Solver++
//! (second order, multistep; diffusers' DPMSolverMultistepScheduler as yvoice configures it: linspace timesteps,
//! lower order at the end, the last sigma zero).
//!
//! ```text
//!   x = noisy_proj(z)  t = mlp(sinusoid(t))  c = cond_proj(h) + t
//!   4 x: shift, scale, gate = adaLN(silu(c));  x += gate * swiglu_ffn(rmsnorm(x) * (1 + scale) + shift)
//!   final: shift, scale = adaLN(silu(c));  v = linear(rmsnorm(x) * (1 + scale) + shift)       (no norm weight)
//! ```
//!
//! Guidance as the reference's sample_speech_tokens: the positive and negative conditions are two rows of one pass
//! over the same z, v = v_neg + cfg (v_pos - v_neg).

use nextcuda_audio::{Error, Result};
use nextcuda_core::DevBuf;
use nextcuda_diffusion::kernels::{none, Dt, Nsd};

use crate::ops::{fp, nul, stage, Mat, Ops, Stage, Weights};

const PRE: &str = "model.prediction_head.";
const EPS: f32 = 1e-5;
const FREQ: usize = 256;
const TRAIN_STEPS: usize = 1000;

struct Layer {
    norm: DevBuf,
    shift: Mat,
    scale: Mat,
    gate: Mat,
    gu: Mat,
    down: Mat,
}

pub struct Head {
    proj: Mat,
    cond: Mat,
    t1: Mat,
    t2: Mat,
    layers: Vec<Layer>,
    f_shift: Mat,
    f_scale: Mat,
    out: Mat,
    pub hidden: usize,
    pub latent: usize,
    ffn: usize,
    pub bytes: usize,
    // two rows' work
    x: DevBuf,
    c: DevBuf,
    sc: DevBuf,
    tabs: DevBuf,
    h: DevBuf,
    gu: DevBuf,
    act: DevBuf,
    tmp: DevBuf,
    z: DevBuf,
    v: DevBuf,
    rows: DevBuf,
    temb: DevBuf,
    st: Stage,
}

/// One timestep's sinusoid (cos then sin, as TimestepEmbedder)
fn sinusoid(t: f32) -> Vec<f32> {
    let half = FREQ / 2;
    let f: Vec<f32> = (0..half).map(|i| (-(10000f32.ln()) * i as f32 / half as f32).exp()).collect();
    let mut e: Vec<f32> = f.iter().map(|w| (t * w).cos()).collect();
    e.extend(f.iter().map(|w| (t * w).sin()));
    e
}

impl Head {
    pub fn load(ops: &Ops, w: &Weights) -> Result<Head> {
        let n = |s: &str| format!("{PRE}{s}.weight");
        let hidden = w.shape(&n("cond_proj"))?[0];
        let latent = w.shape(&n("noisy_images_proj"))?[1];
        let split = |name: &str, i: usize| -> Result<Mat> { w.mat(ops, &[(name, i * hidden, (i + 1) * hidden)], false) };
        let mut layers = Vec::new();
        for l in 0.. {
            let p = format!("{PRE}layers.{l}.");
            if !w.has(&format!("{p}norm.weight")) {
                break;
            }
            let ada = format!("{p}adaLN_modulation.1.weight");
            let ffn = w.shape(&format!("{p}ffn.gate_proj.weight"))?[0];
            layers.push(Layer {
                norm: w.dev_f32(ops, &format!("{p}norm.weight"))?,
                shift: split(&ada, 0)?,
                scale: split(&ada, 1)?,
                gate: split(&ada, 2)?,
                gu: w.mat(ops, &[(&format!("{p}ffn.gate_proj.weight"), 0, ffn), (&format!("{p}ffn.up_proj.weight"), 0, ffn)], false)?,
                down: w.mat1(ops, &format!("{p}ffn.down_proj.weight"), false)?,
            });
        }
        let ffn = layers.first().map(|l| l.gu.n / 2).ok_or_else(|| Error(format!("no {PRE}layers in the checkpoint")))?;
        let fa = n("final_layer.adaLN_modulation.1");
        let g = &ops.gpu;
        let head = Head {
            proj: w.mat1(ops, &n("noisy_images_proj"), false)?,
            cond: w.mat1(ops, &n("cond_proj"), false)?,
            t1: w.mat1(ops, &n("t_embedder.mlp.0"), false)?,
            t2: w.mat1(ops, &n("t_embedder.mlp.2"), false)?,
            f_shift: split(&fa, 0)?,
            f_scale: split(&fa, 1)?,
            out: w.mat1(ops, &n("final_layer.linear"), false)?,
            layers,
            hidden,
            latent,
            ffn,
            bytes: 0,
            x: DevBuf::f32(g, 2 * hidden)?,
            c: DevBuf::f32(g, 2 * hidden)?,
            sc: DevBuf::f32(g, 2 * hidden)?,
            tabs: DevBuf::f32(g, 3 * 2 * hidden)?,
            h: DevBuf::f32(g, 2 * hidden)?,
            gu: DevBuf::f32(g, 2 * 2 * ffn)?,
            act: DevBuf::f32(g, 2 * ffn)?,
            tmp: DevBuf::f32(g, 2 * hidden)?,
            z: DevBuf::f32(g, 2 * latent)?,
            v: DevBuf::f32(g, 2 * latent)?,
            rows: DevBuf::new(g, 8)?,
            temb: DevBuf::f32(g, hidden)?,
            st: stage(g, 2 * hidden.max(ffn).max(FREQ), None)?,
        };
        head.rows.write(0, &[0, 0, 0, 0, 1, 0, 0, 0])?;
        let mut head = head;
        head.bytes = [&head.proj, &head.cond, &head.t1, &head.t2, &head.f_shift, &head.f_scale, &head.out].iter().map(|m| m.bytes()).sum::<usize>()
            + head.layers.iter().map(|l| l.shift.bytes() * 3 + l.gu.bytes() + l.down.bytes()).sum::<usize>();
        Ok(head)
    }

    /// The timestep embedding mlp(sinusoid(t)) [hidden], on the host
    pub fn t_embed(&self, ops: &Ops, nsd: &Nsd, t: f32) -> Result<Vec<f32>> {
        let s = DevBuf::from_f32(&ops.gpu, &sinusoid(t))?;
        self.t1.apply(ops, nsd, s.fp(), 1, &self.st, nul(), self.tmp.fp())?;
        ops.silu(self.tmp.fp(), self.hidden)?;
        self.t2.apply(ops, nsd, self.tmp.fp(), 1, &self.st, nul(), self.temb.fp())?;
        Ok(self.temb.to_f32()?)
    }

    /// The conditions' projection: cond_proj of [pos, neg] (device, 2 rows) into c, kept for the frame
    pub fn condition(&self, ops: &Ops, nsd: &Nsd, h2: *const f32) -> Result<DevBuf> {
        let out = DevBuf::f32(&ops.gpu, 2 * self.hidden)?;
        self.cond.apply(ops, nsd, h2, 2, &self.st, nul(), out.fp())?;
        Ok(out)
    }

    /// v [2, latent] for z (the same row twice) at a timestep (its embedding `temb` on the device)
    pub fn velocity(&self, ops: &Ops, nsd: &Nsd, cond: &DevBuf, z: &[f32], temb: &DevBuf) -> Result<Vec<f32>> {
        let (h, l) = (self.hidden, self.latent);
        let mut zz = z.to_vec();
        zz.extend_from_slice(z);
        self.z.write(0, &crate::bytes(&zz))?;
        self.proj.apply(ops, nsd, self.z.fp(), 2, &self.st, nul(), self.x.fp())?;
        // c = cond + t for both rows, then silu(c)
        self.c.copy_within(0, cond, 0, 2 * h * 4)?;
        ops.add(self.c.fp(), temb.fp(), h)?;
        ops.add(fp(&self.c, h), temb.fp(), h)?;
        self.sc.copy_within(0, &self.c, 0, 2 * h * 4)?;
        ops.silu(self.sc.fp(), 2 * h)?;
        let (shift, scale, gate) = (self.tabs.fp(), fp(&self.tabs, 2 * h), fp(&self.tabs, 4 * h));
        for ly in &self.layers {
            ly.shift.apply(ops, nsd, self.sc.fp(), 2, &self.st, nul(), shift)?;
            ly.scale.apply(ops, nsd, self.sc.fp(), 2, &self.st, nul(), scale)?;
            ly.gate.apply(ops, nsd, self.sc.fp(), 2, &self.st, nul(), gate)?;
            nsd.rms_norm_mod(self.x.ptr(), Dt::F32, 2, h, ly.norm.ptr(), EPS, self.rows.ptr(), scale.cast(), shift.cast(), self.h.ptr(), Dt::F32)?;
            ly.gu.apply(ops, nsd, self.h.fp(), 2, &self.st, nul(), self.gu.fp())?;
            nsd.swiglu(self.gu.ptr(), Dt::F32, 2, self.ffn, self.act.ptr(), Dt::F32)?;
            ly.down.apply(ops, nsd, self.act.fp(), 2, &self.st, nul(), self.tmp.fp())?;
            nsd.gate_add(self.x.ptr(), Dt::F32, 2, h, self.tmp.ptr(), Dt::F32, self.rows.ptr(), gate.cast())?;
        }
        self.f_shift.apply(ops, nsd, self.sc.fp(), 2, &self.st, nul(), shift)?;
        self.f_scale.apply(ops, nsd, self.sc.fp(), 2, &self.st, nul(), scale)?;
        nsd.rms_norm_mod(self.x.ptr(), Dt::F32, 2, h, none(), EPS, self.rows.ptr(), scale.cast(), shift.cast(), self.h.ptr(), Dt::F32)?;
        self.out.apply(ops, nsd, self.h.fp(), 2, &self.st, nul(), self.v.fp())?;
        let v = self.v.to_f32()?;
        Ok(v[..2 * l].to_vec())
    }
}

/// DPM-Solver++ (2M, midpoint, lower order final, final sigma 0) over the cosine schedule, v-prediction
pub struct Solver {
    pub timesteps: Vec<i64>,
    sigmas: Vec<f32>,
}

impl Solver {
    pub fn new(steps: usize) -> Solver {
        // betas_for_alpha_bar (cosine, max beta 0.999) in float64, as python's math does, then float32 as the tensor
        let ab = |t: f64| ((t + 0.008) / 1.008 * std::f64::consts::PI / 2.0).cos().powi(2);
        let betas: Vec<f32> = (0..TRAIN_STEPS).map(|i| {
            let (t1, t2) = (i as f64 / TRAIN_STEPS as f64, (i + 1) as f64 / TRAIN_STEPS as f64);
            (1.0 - ab(t2) / ab(t1)).min(0.999) as f32
        }).collect();
        let mut acp = Vec::with_capacity(TRAIN_STEPS);
        let mut p = 1f32;
        for b in &betas {
            p *= 1.0 - b;
            acp.push(p);
        }
        // linspace(0, 999, steps + 1).round()[::-1][:-1]
        let last = TRAIN_STEPS as f64 - 1.0;
        let ts: Vec<i64> = (0..=steps).map(|i| (last * i as f64 / steps as f64).round() as i64).rev().take(steps).collect();
        let sig: Vec<f32> = acp.iter().map(|a| ((1.0 - a) / a).sqrt()).collect();
        let mut sigmas: Vec<f32> = ts.iter().map(|&t| sig[t as usize]).collect();
        sigmas.push(0.0);
        Solver { timesteps: ts, sigmas }
    }

    /// Step i's sigma
    pub fn sigmas_at(&self, i: usize) -> f32 {
        self.sigmas[i]
    }

    fn alpha_sigma(s: f32) -> (f32, f32) {
        let a = 1.0 / (s * s + 1.0).sqrt();
        (a, s * a)
    }

    /// The sample from `x` (the noise) by `model(i, x)` -> v at step i
    pub fn run(&self, x: &mut [f32], model: &mut dyn FnMut(usize, &[f32]) -> Result<Vec<f32>>) -> Result<()> {
        let n = self.timesteps.len();
        let mut prev_x0: Option<Vec<f32>> = None;
        let mut lower = 0;
        for i in 0..n {
            let v = model(i, x)?;
            let (a_s, s_s) = Solver::alpha_sigma(self.sigmas[i]);
            let x0: Vec<f32> = x.iter().zip(&v).map(|(xi, vi)| a_s * xi - s_s * vi).collect();
            let (a_t, s_t) = Solver::alpha_sigma(self.sigmas[i + 1]);
            let lambda = |a: f32, s: f32| a.ln() - s.ln();
            let final_ = i == n - 1;
            let (l_t, l_s) = (lambda(a_t, s_t), lambda(a_s, s_s));
            let h = l_t - l_s;
            if lower < 1 || final_ || prev_x0.is_none() {
                // first order: x_t = (s_t / s_s) x - a_t (e^-h - 1) x0   (s_t = 0 at the end: x0 itself)
                let c1 = if s_t == 0.0 { 0.0 } else { s_t / s_s };
                let c2 = a_t * ((-h).exp() - 1.0);
                for (xi, x0i) in x.iter_mut().zip(&x0) {
                    *xi = c1 * *xi - c2 * x0i;
                }
                if final_ && s_t == 0.0 {
                    x.copy_from_slice(&x0);
                }
            } else {
                let m1 = prev_x0.as_ref().unwrap();
                let (a_s1, s_s1) = Solver::alpha_sigma(self.sigmas[i - 1]);
                let h0 = l_s - lambda(a_s1, s_s1);
                let r0 = h0 / h;
                let c1 = s_t / s_s;
                let c2 = a_t * ((-h).exp() - 1.0);
                for ((xi, d0), m1i) in x.iter_mut().zip(&x0).zip(m1) {
                    let d1 = (1.0 / r0) * (d0 - m1i);
                    *xi = c1 * *xi - c2 * d0 - 0.5 * c2 * d1;
                }
            }
            prev_x0 = Some(x0);
            lower = (lower + 1).min(2);
        }
        Ok(())
    }
}
