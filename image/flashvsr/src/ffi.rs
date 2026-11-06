//! This engine's own kernels (`kernels/image/flashvsr/fv.h`), bound from the image kernel library by name.

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

use nextcuda_core::{Error, Gpu, Result};

type G = *mut c_void;
type M = *mut c_void;
type P = *const c_void;

pub struct Api {
    pub patchify: unsafe extern "C" fn(G, *const f32, i64, i64, i64, M) -> c_int,
    pub unpatchify_sub: unsafe extern "C" fn(G, *const f32, *const f32, i64, i64, i64, *mut f32) -> c_int,
    pub add_lq: unsafe extern "C" fn(G, *mut f32, *const f32, i64, i64, i64, i64) -> c_int,
    pub ln: unsafe extern "C" fn(G, *const f32, i64, i64, f32, *const f32, *const f32, M) -> c_int,
    pub gate_add: unsafe extern "C" fn(G, *mut f32, *const f32, i64, i64, *const f32) -> c_int,
    pub block_mean: unsafe extern "C" fn(G, P, i64, i64, i64, *mut f32) -> c_int,
    pub block_probs: unsafe extern "C" fn(G, *const f32, *const f32, i64, i64, i64, i64, i64, i64, *mut f32) -> c_int,
    pub topk_thr: unsafe extern "C" fn(G, *const f32, i64, i64, i64, *mut f32) -> c_int,
    pub block_lists: unsafe extern "C" fn(G, *const f32, *const f32, i64, i64, i64, i64, *mut i32, *mut i32) -> c_int,
    pub sparse_attn: unsafe extern "C" fn(G, P, i64, i64, P, P, P, P, i64, *const i32, *const i32, i64, M) -> c_int,
    pub unshuffle: unsafe extern "C" fn(G, P, i64, i64, i64, i64, i64, M) -> c_int,
    pub im2col3: unsafe extern "C" fn(G, P, i64, i64, i64, i64, i64, i64, i64, i64, M) -> c_int,
    pub im2col2: unsafe extern "C" fn(G, P, i64, i64, P, i64, i64, i64, i64, c_int, i64, i64, M) -> c_int,
    pub rms_silu: unsafe extern "C" fn(G, P, i64, i64, *const f32, M) -> c_int,
    pub relu: unsafe extern "C" fn(G, M, i64) -> c_int,
    pub add_relu: unsafe extern "C" fn(G, M, P, i64) -> c_int,
    pub tcd_input: unsafe extern "C" fn(G, P, *const f32, i64, i64, i64, i64, M) -> c_int,
    pub u8_to_half: unsafe extern "C" fn(G, *const u8, i64, M) -> c_int,
    pub to_half: unsafe extern "C" fn(G, *const f32, i64, M) -> c_int,
    pub to_float: unsafe extern "C" fn(G, P, i64, *mut f32) -> c_int,
    pub stats: unsafe extern "C" fn(G, P, i64, i64, i64, i64, i64, f32, f32, *mut f64) -> c_int,
    pub to_u8: unsafe extern "C" fn(G, P, i64, i64, i64, i64, i64, *const f32, *mut u8) -> c_int,
    pub randn: unsafe extern "C" fn(G, *mut f32, i64, u64, u64) -> c_int,
    pub adain: unsafe extern "C" fn(G, *const f64, f64, *mut f32) -> c_int,
    pub conv3_small: unsafe extern "C" fn(G, P, i64, i64, i64, *const f32, i64, *const f32, M) -> c_int,
}

// SAFETY: function pointers into the library, which stays loaded for the process.
unsafe impl Send for Api {}
unsafe impl Sync for Api {}

pub fn api() -> Result<&'static Api> {
    static API: OnceLock<std::result::Result<Api, String>> = OnceLock::new();
    API.get_or_init(|| {
        let lib = nextcuda_core::api().map_err(|e| e.0)?;
        macro_rules! sym {
            ($name:literal) => {{
                let p = lib.symbol($name);
                if p.is_null() {
                    return Err(format!("{} is not in the kernel library (the image library, rebuilt: kernels/build.sh)", $name));
                }
                // SAFETY: the symbol is declared in fv.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Api {
            patchify: sym!("ns_image_fv_patchify"),
            unpatchify_sub: sym!("ns_image_fv_unpatchify_sub"),
            add_lq: sym!("ns_image_fv_add_lq"),
            ln: sym!("ns_image_fv_ln"),
            gate_add: sym!("ns_image_fv_gate_add"),
            block_mean: sym!("ns_image_fv_block_mean"),
            block_probs: sym!("ns_image_fv_block_probs"),
            topk_thr: sym!("ns_image_fv_topk_thr"),
            block_lists: sym!("ns_image_fv_block_lists"),
            sparse_attn: sym!("ns_image_fv_sparse_attn"),
            unshuffle: sym!("ns_image_fv_unshuffle"),
            im2col3: sym!("ns_image_fv_im2col3"),
            im2col2: sym!("ns_image_fv_im2col2"),
            rms_silu: sym!("ns_image_fv_rms_silu"),
            relu: sym!("ns_image_fv_relu"),
            add_relu: sym!("ns_image_fv_add_relu"),
            tcd_input: sym!("ns_image_fv_tcd_input"),
            u8_to_half: sym!("ns_image_fv_u8_to_half"),
            to_half: sym!("ns_image_fv_to_half"),
            to_float: sym!("ns_image_fv_to_float"),
            stats: sym!("ns_image_fv_stats"),
            to_u8: sym!("ns_image_fv_to_u8"),
            randn: sym!("ns_image_fv_randn"),
            adain: sym!("ns_image_fv_adain"),
            conv3_small: sym!("ns_image_fv_conv3_small"),
        })
    })
    .as_ref()
    .map_err(|e| Error(e.clone()))
}

/// A call's return code as a result (the reason from the library)
pub fn check(rc: c_int, what: &str) -> Result<()> {
    if rc == 0 {
        return Ok(());
    }
    let why = nextcuda_core::api().map(|a| a.error()).unwrap_or_default();
    Err(Error(format!("{what}: {why}")))
}

/// The GPU handle the kernels take
pub fn g(gpu: &Gpu) -> G {
    gpu.raw()
}

/// IEEE half from float32, rounded to nearest even
pub fn f16_bits(v: f32) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32;
    let man = x & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rest = m & ((1u32 << shift) - 1);
        let mut r = m >> shift;
        if rest > half || (rest == half && (r & 1) == 1) {
            r += 1;
        }
        return sign | r as u16;
    }
    let mut r = ((e as u32) << 10) | (man >> 13);
    let rest = man & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && (r & 1) == 1) {
        r += 1;
    }
    sign | r as u16
}

/// Floats to half bytes
pub fn half_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for x in v {
        out.extend_from_slice(&f16_bits(*x).to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halves_round_to_nearest_even() {
        for v in [0.0f32, 1.0, -2.5, 65504.0, 1e-5, 3.14159, -1e-8, 0.333333] {
            let h = f16_bits(v);
            let back = nextcuda_gguf::safetensors::f16_to_f32(h);
            assert!((back - v).abs() <= v.abs() * 1e-3 + 1e-7, "{v} -> {back}");
        }
        assert_eq!(f16_bits(1e6), 0x7c00);
    }
}
