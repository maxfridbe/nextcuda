//! Z-Image's single-stream DiT (Tongyi-MAI's 6B denoiser) and its Euler sampler. It sits in the foundation because two
//! image engines run it: Ming-Image (image/mingimage: inclusionAI's design model, the DiT conditioned by Ling-Mini-2.0)
//! and Z-Image-Turbo (image/zimage: Tongyi-MAI's own few-step model, conditioned by Qwen3-4B). Its kernels are the
//! shared diffusion ones and its own (kernels/diffusion/zimage.h).

pub mod dit;
pub mod ffi;

use nextcuda_core::{DevBuf, Gpu, Result};
use nextcuda_diffusion::kernels::Nsd;

/// Latents float32 [16, lh, lw] from the starting noise `start` (same shape): Euler over the levels `sig` (the steps
/// run between each two; a step from a level to itself is skipped), the DiT's velocity after the caption `pre` each
/// step and, with `npre`, guidance pos + cfg (pos - neg) against that second caption. `step(i)` after step i.
#[allow(clippy::too_many_arguments)]
pub fn sample(dit: &dit::Dit, nsd: &Nsd, gpu: &Gpu, pre: &dit::Prefix, npre: Option<&dit::Prefix>, cfg: f32, start: &[f32], lh: usize, lw: usize,
              sig: &[f32], step: &mut dyn FnMut(usize) -> Result<()>) -> Result<Vec<f32>> {
    let k = ffi::api()?;
    let hw = (lh / 2, lw / 2);
    let n = hw.0 * hw.1 * dit::PATCH_IN;
    let mods = dit.mods(nsd, sig)?;
    let lat = DevBuf::from_f32(&nsd.gpu, &dit::patchify(start, lh, lw))?;
    for i in 0..sig.len() - 1 {
        let (s, sn) = (sig[i], sig[i + 1]);
        if sn != s {
            let v = dit.velocity(nsd, pre, &lat, hw, &mods[i], None)?;
            if let Some(np) = npre {
                let vn = dit.velocity(nsd, np, &lat, hw, &mods[i], None)?;
                // SAFETY: n floats each: vn = vn - v, then v += -cfg vn (v + cfg (v - vn))
                ffi::check(unsafe { (k.axpy)(gpu.raw(), vn.fp(), v.fp(), n as i64, -1.0) }, "guidance")?;
                ffi::check(unsafe { (k.axpy)(gpu.raw(), v.fp(), vn.fp(), n as i64, -cfg) }, "guidance")?;
            }
            // the scheduler's step on the pipeline's minus the model's output
            // SAFETY: n floats each.
            ffi::check(unsafe { (k.axpy)(gpu.raw(), lat.fp(), v.fp(), n as i64, -(sn - s)) }, "euler step")?;
            gpu.sync()?;
        }
        step(i)?;
    }
    Ok(dit::unpatchify(&lat.to_f32()?, lh, lw))
}
