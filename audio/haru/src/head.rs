//! The MeanFlow head (haru/models/meanflow_head.py): the next frame's 64-value audio latent from the language model's
//! hidden state, in one step of a small MLP. It predicts the mean velocity u(z, t, h | c, w) over an interval h (noise
//! at t = 0 to data at t = 1); a step is z += h u. Its condition is cond_proj(hidden) plus four scalar embeddings (t,
//! the interval constant, h and the guidance's 1 - 1/w: sinusoidal features through a two-layer MLP), made on the host
//! once a guidance (after nextsycl's audio/haru/src/head.rs); then four adaptive-norm SwiGLU layers and a final
//! adaptive norm and projection - the shape of VibeVoice's diffusion head, run on the same calls as the VibeVoice
//! engine's head (audio/vibevoice/src/head.rs): the guidance is an input here, so one row, no negative pass.
//!
//! ```text
//!   c = silu(cond_proj(h) + t_embed(t) + dt_embed(7) + h_embed(1) + omega_embed(1 - 1/w));  x = latent_proj(z)
//!   4 x: shift, scale, gate = ada(c);  x += gate * swiglu_ffn(rmsnorm(x) * w * (1 + scale) + shift)
//!   final: shift, scale = ada(c);  u = proj(rmsnorm(x) * (1 + scale) + shift)
//! ```

use std::collections::HashMap;
use std::sync::Mutex;

use nextcuda_audio::{Error, Result};
use nextcuda_core::DevBuf;
use nextcuda_diffusion::kernels::{none, Dt, Nsd};

use crate::ops::{fp, nul, stage, Mat, Ops, Stage};
use crate::weights::{bytes, Config, Weights};

const FREQ: usize = 256;

/// A scalar's sinusoidal features through fc1, SiLU, fc2 (no biases), on the host
struct Scalar {
    fc1: Vec<f32>,
    fc2: Vec<f32>,
}

impl Scalar {
    fn embed(&self, v: f32, hidden: usize) -> Vec<f32> {
        let half = FREQ / 2;
        let feats: Vec<f32> = (0..FREQ).map(|i| {
            let f = (-(10000f32.ln()) * (i % half) as f32 / half as f32).exp();
            if i < half { (v * f).cos() } else { (v * f).sin() }
        }).collect();
        let mut a = vec![0f32; hidden];
        for (o, row) in a.iter_mut().zip(self.fc1.chunks_exact(FREQ)) {
            let s: f32 = row.iter().zip(&feats).map(|(w, x)| w * x).sum();
            *o = s / (1.0 + (-s).exp());
        }
        self.fc2.chunks_exact(hidden).map(|row| row.iter().zip(&a).map(|(w, x)| w * x).sum()).collect()
    }
}

struct Layer {
    norm: DevBuf,
    /// ada's rows: shift, scale, gate (one product: [3 hidden] out)
    ada: Mat,
    gu: Mat,
    down: Mat,
}

pub struct Head {
    pub hidden: usize,
    pub latent: usize,
    ffn: usize,
    eps: f32,
    interval: f32,
    pub steps: usize,
    cond: Mat,
    latent_proj: Mat,
    layers: Vec<Layer>,
    final_ada: Mat,
    proj: Mat,
    scalars: [Scalar; 4],
    /// the scalar embeddings' sum for each step, by guidance (its bits)
    consts: Mutex<HashMap<u32, Vec<DevBuf>>>,
    pub bytes: usize,
    // one row's work
    cv: DevBuf,
    c: DevBuf,
    z: DevBuf,
    x: DevBuf,
    ada: DevBuf,
    h: DevBuf,
    gu: DevBuf,
    act: DevBuf,
    tmp: DevBuf,
    u: DevBuf,
    rows: DevBuf,
    st: Stage,
}

impl Head {
    pub fn load(ops: &Ops, w: &Weights, cfg: &Config) -> Result<Head> {
        let h = cfg.head_hidden;
        if h != cfg.hidden {
            return Err(Error(format!("head.hidden_size {h}: the language model's {} expected", cfg.hidden)));
        }
        let mut layers = Vec::new();
        for l in 0..cfg.head_layers {
            let n = |s: &str| format!("head.layers.{l}.{s}");
            let ffn = w.shape(&n("ffn.gate_proj.weight"))?[0];
            layers.push(Layer {
                norm: w.dev_f32(ops, &n("norm.weight"))?,
                ada: w.mat1(ops, &n("ada.weight"), false)?,
                gu: w.mat(ops, &[(&n("ffn.gate_proj.weight"), 0, ffn), (&n("ffn.up_proj.weight"), 0, ffn)], false)?,
                down: w.mat1(ops, &n("ffn.down_proj.weight"), false)?,
            });
        }
        let ffn = layers.first().map(|l| l.gu.n / 2).ok_or_else(|| Error("no head.layers in the checkpoint".into()))?;
        let scalar = |p: &str| -> Result<Scalar> {
            let (a, b) = (format!("head.{p}.fc1.weight"), format!("head.{p}.fc2.weight"));
            if w.shape(&a)? != [h, FREQ] || w.shape(&b)? != [h, h] {
                return Err(Error(format!("head.{p}: [{h}, {FREQ}] and [{h}, {h}] expected")));
            }
            Ok(Scalar { fc1: w.f32(&a)?, fc2: w.f32(&b)? })
        };
        let g = &ops.gpu;
        let mut head = Head {
            hidden: h,
            latent: cfg.latent,
            ffn,
            eps: cfg.head_eps,
            interval: cfg.interval,
            steps: cfg.head_steps,
            cond: w.mat1(ops, "head.cond_proj.weight", false)?,
            latent_proj: w.mat1(ops, "head.latent_proj.weight", false)?,
            final_ada: w.mat1(ops, "head.final.ada.weight", false)?,
            proj: w.mat1(ops, "head.final.proj.weight", false)?,
            layers,
            scalars: [scalar("t_embed")?, scalar("dt_embed")?, scalar("h_embed")?, scalar("omega_embed")?],
            consts: Mutex::new(HashMap::new()),
            bytes: 0,
            cv: DevBuf::f32(g, h)?,
            c: DevBuf::f32(g, h)?,
            z: DevBuf::f32(g, cfg.latent.max(4))?,
            x: DevBuf::f32(g, h)?,
            ada: DevBuf::f32(g, 3 * h)?,
            h: DevBuf::f32(g, h)?,
            gu: DevBuf::f32(g, 2 * ffn)?,
            act: DevBuf::f32(g, ffn)?,
            tmp: DevBuf::f32(g, h)?,
            u: DevBuf::f32(g, cfg.latent.max(4))?,
            rows: DevBuf::new(g, 4)?,
            st: stage(g, h.max(ffn).max(cfg.latent), None)?,
        };
        head.rows.write(0, &[0, 0, 0, 0])?;
        head.bytes = [&head.cond, &head.latent_proj, &head.final_ada, &head.proj].iter().map(|m| m.bytes()).sum::<usize>()
            + head.layers.iter().map(|l| l.ada.bytes() + l.gu.bytes() + l.down.bytes()).sum::<usize>();
        Ok(head)
    }

    /// The scalar embeddings' sum for each step at guidance `w` (t = i / steps, the interval constant, h = 1 / steps,
    /// omega = 1 - 1 / w), on the device
    fn consts(&self, ops: &Ops, w: f32) -> Result<Vec<DevBuf>> {
        let mut m = self.consts.lock().map_err(|_| Error("head lock".into()))?;
        if let Some(v) = m.get(&w.to_bits()) {
            return v.iter().map(|b| b.view(0, b.len).map_err(|e| Error(e.0))).collect();
        }
        let hstep = 1.0 / self.steps as f32;
        let mut v = Vec::new();
        for i in 0..self.steps {
            let parts = [self.scalars[0].embed(i as f32 * hstep, self.hidden), self.scalars[1].embed(self.interval, self.hidden),
                         self.scalars[2].embed(hstep, self.hidden), self.scalars[3].embed(1.0 - 1.0 / w, self.hidden)];
            let sum: Vec<f32> = (0..self.hidden).map(|j| parts.iter().map(|p| p[j]).sum()).collect();
            v.push(DevBuf::from_f32(&ops.gpu, &sum)?);
        }
        let out = v.iter().map(|b| b.view(0, b.len).map_err(|e| Error(e.0))).collect::<Result<Vec<_>>>()?;
        m.insert(w.to_bits(), v);
        Ok(out)
    }

    /// The normalized latent [latent] from the hidden state `hidden` (device, `hidden` floats) and `noise` (z at
    /// t = 0), at guidance `w`
    pub fn sample(&self, ops: &Ops, nsd: &Nsd, hidden: *const f32, noise: &[f32], w: f32) -> Result<Vec<f32>> {
        let hd = self.hidden;
        let consts = self.consts(ops, w)?;
        self.cond.apply(ops, nsd, hidden, 1, &self.st, nul(), self.cv.fp())?;
        let mut z = noise.to_vec();
        let hstep = 1.0 / self.steps as f32;
        let (shift, scale, gate) = (self.ada.fp(), fp(&self.ada, hd), fp(&self.ada, 2 * hd));
        for cst in &consts {
            self.c.copy_within(0, &self.cv, 0, hd * 4)?;
            ops.add(self.c.fp(), cst.fp(), hd)?;
            ops.silu(self.c.fp(), hd)?;
            self.z.write(0, &bytes(&z))?;
            self.latent_proj.apply(ops, nsd, self.z.fp(), 1, &self.st, nul(), self.x.fp())?;
            for ly in &self.layers {
                ly.ada.apply(ops, nsd, self.c.fp(), 1, &self.st, nul(), self.ada.fp())?;
                nsd.rms_norm_mod(self.x.ptr(), Dt::F32, 1, hd, ly.norm.ptr(), self.eps, self.rows.ptr(), scale.cast(), shift.cast(), self.h.ptr(), Dt::F32)?;
                ly.gu.apply(ops, nsd, self.h.fp(), 1, &self.st, nul(), self.gu.fp())?;
                nsd.swiglu(self.gu.ptr(), Dt::F32, 1, self.ffn, self.act.ptr(), Dt::F32)?;
                ly.down.apply(ops, nsd, self.act.fp(), 1, &self.st, nul(), self.tmp.fp())?;
                nsd.gate_add(self.x.ptr(), Dt::F32, 1, hd, self.tmp.ptr(), Dt::F32, self.rows.ptr(), gate.cast())?;
            }
            self.final_ada.apply(ops, nsd, self.c.fp(), 1, &self.st, nul(), self.ada.fp())?;
            nsd.rms_norm_mod(self.x.ptr(), Dt::F32, 1, hd, none(), self.eps, self.rows.ptr(), scale.cast(), shift.cast(), self.h.ptr(), Dt::F32)?;
            self.proj.apply(ops, nsd, self.h.fp(), 1, &self.st, nul(), self.u.fp())?;
            let u = self.u.to_f32()?;
            for (a, b) in z.iter_mut().zip(&u) {
                *a += hstep * b;
            }
        }
        Ok(z)
    }
}
