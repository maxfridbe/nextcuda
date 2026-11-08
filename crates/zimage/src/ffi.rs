//! The DiT's own kernels (`kernels/diffusion/zimage.h`, in every kind's library), bound by name.

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

type G = *mut c_void;
type M = *mut c_void;
type P = *const c_void;

pub struct Api {
    pub ln_mod: unsafe extern "C" fn(G, *const f32, i64, i64, f32, *const f32, M) -> c_int,
    pub rms_mod: unsafe extern "C" fn(G, *const f32, i64, i64, *const f32, f32, *const f32, M) -> c_int,
    pub gate_add: unsafe extern "C" fn(G, *mut f32, i64, i64, P, *const f32) -> c_int,
    pub swiglu_scaled: unsafe extern "C" fn(G, P, i64, i64, f32, M) -> c_int,
    pub to_half_rows: unsafe extern "C" fn(G, *const f32, i64, i64, f32, M, i64) -> c_int,
    pub to_float: unsafe extern "C" fn(G, P, *mut f32, i64) -> c_int,
    pub axpy: unsafe extern "C" fn(G, *mut f32, *const f32, i64, f32) -> c_int,
}

// SAFETY: function pointers into the library, which stays loaded for the process.
unsafe impl Send for Api {}
unsafe impl Sync for Api {}

pub fn api() -> nextcuda_core::Result<&'static Api> {
    static API: OnceLock<std::result::Result<Api, String>> = OnceLock::new();
    API.get_or_init(|| {
        let lib = nextcuda_core::api().map_err(|e| e.0)?;
        macro_rules! sym {
            ($name:literal) => {{
                let p = lib.symbol($name);
                if p.is_null() {
                    return Err(format!("{} is not in the kernel library (rebuilt: kernels/build.sh)", $name));
                }
                // SAFETY: the symbol is declared in zimage.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Api {
            ln_mod: sym!("ns_zimage_ln_mod"),
            rms_mod: sym!("ns_zimage_rms_mod"),
            gate_add: sym!("ns_zimage_gate_add"),
            swiglu_scaled: sym!("ns_zimage_swiglu_scaled"),
            to_half_rows: sym!("ns_zimage_to_half_rows"),
            to_float: sym!("ns_zimage_to_float"),
            axpy: sym!("ns_zimage_axpy"),
        })
    })
    .as_ref()
    .map_err(|e| nextcuda_core::Error(e.clone()))
}

/// The scale the MLPs' gated products are made at in half (their next linear's float32 output multiplied back)
pub const ACT_SCALE: f32 = 1.0 / 256.0;

/// out [m, c] half = silu(x[:, :c]) * x[:, c:] * ACT_SCALE (x [m, 2c] half)
pub fn swiglu(gpu: &nextcuda_core::Gpu, x: P, m: usize, c: usize, out: M) -> nextcuda_core::Result<()> {
    let k = api()?;
    // SAFETY: x holds m x 2c halves, out m x c.
    check(unsafe { (k.swiglu_scaled)(gpu.raw(), x, m as i64, c as i64, ACT_SCALE, out) }, "swiglu")
}

/// A call's return code as a result (the reason from the library)
pub fn check(rc: c_int, what: &str) -> nextcuda_core::Result<()> {
    if rc == 0 {
        return Ok(());
    }
    let why = nextcuda_core::api().map(|a| a.error()).unwrap_or_default();
    Err(nextcuda_core::Error(format!("{what}: {why}")))
}
