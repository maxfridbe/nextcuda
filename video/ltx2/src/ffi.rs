//! This engine's own kernels (`kernels/video/ltx2/lx.h`), and the shared video kernels the diffusion crate does not
//! bind (`nsd_video.inc`: the VAE's convolutions and norms), bound from the video kernel library by name.

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

use nextcuda_core::{Error, Gpu, Result};

type G = *mut c_void;
type M = *mut c_void;
type P = *const c_void;

pub struct Api {
    pub to_half_rows: unsafe extern "C" fn(G, *const f32, i64, i64, i64, f32, M, i64) -> c_int,
    pub to_float: unsafe extern "C" fn(G, P, *mut f32, i64, f32) -> c_int,
    pub geglu: unsafe extern "C" fn(G, *const f32, i64, *const f32, i64, i64, i64, f32, M) -> c_int,
    pub axpy: unsafe extern "C" fn(G, *mut f32, *const f32, i64, f32) -> c_int,
    pub head_gate: unsafe extern "C" fn(G, M, i64, i64, i64, *const f32) -> c_int,
    pub mod_rows: unsafe extern "C" fn(G, *const f32, i64, i64, *const f32, *const f32, f32, M) -> c_int,
    pub ln_mod: unsafe extern "C" fn(G, *const f32, i64, i64, f32, *const i32, *const f32, *const f32, *mut f32) -> c_int,
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
                    return Err(format!("{} is not in the kernel library (the video library, rebuilt: kernels/build.sh)", $name));
                }
                // SAFETY: the symbol is declared in lx.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Api {
            to_half_rows: sym!("ns_video_lx_to_half_rows"),
            to_float: sym!("ns_video_lx_to_float"),
            geglu: sym!("ns_video_lx_geglu"),
            axpy: sym!("ns_video_lx_axpy"),
            head_gate: sym!("ns_video_lx_head_gate"),
            mod_rows: sym!("ns_video_lx_mod_rows"),
            ln_mod: sym!("ns_video_lx_ln_mod"),
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

/// The scale Gemma's MLP products are made at in half (the down projection's float32 output multiplied back)
pub const ACT_SCALE: f32 = 1.0 / 256.0;

/// out [m, c] half (rows `ldo` apart) = x [m, c] float32 (rows `ldx` apart) * s
pub fn to_half_rows(g: &Gpu, x: *const f32, m: usize, c: usize, ldx: usize, s: f32, out: M, ldo: usize) -> Result<()> {
    // SAFETY: the caller's buffers hold the rows named.
    check(unsafe { (api()?.to_half_rows)(g.raw(), x, m as i64, c as i64, ldx as i64, s, out, ldo as i64) }, "to half")
}

pub fn to_half(g: &Gpu, x: *const f32, n: usize, out: M) -> Result<()> {
    to_half_rows(g, x, 1, n, n, 1.0, out, n)
}

/// out float32 = x half * s
pub fn to_float(g: &Gpu, x: P, out: *mut f32, n: usize, s: f32) -> Result<()> {
    // SAFETY: n values each way.
    check(unsafe { (api()?.to_float)(g.raw(), x, out, n as i64, s) }, "to float")
}

/// out [m, c] half = gelu_tanh(a) * b * s (b null: gelu alone), a / b float32 rows `lda` / `ldb` apart
#[allow(clippy::too_many_arguments)]
pub fn geglu(g: &Gpu, a: *const f32, lda: usize, b: *const f32, ldb: usize, m: usize, c: usize, s: f32, out: M) -> Result<()> {
    // SAFETY: the caller's buffers hold the rows named.
    check(unsafe { (api()?.geglu)(g.raw(), a, lda as i64, b, ldb as i64, m as i64, c as i64, s, out) }, "geglu")
}

/// x += a y over n float32
pub fn axpy(g: &Gpu, x: *mut f32, y: *const f32, n: usize, a: f32) -> Result<()> {
    // SAFETY: n values each.
    check(unsafe { (api()?.axpy)(g.raw(), x, y, n as i64, a) }, "axpy")
}

/// x [m, h * d] half: head j of row r times 2 sigmoid(logits[r, j])
pub fn head_gate(g: &Gpu, x: M, m: usize, h: usize, d: usize, logits: *const f32) -> Result<()> {
    // SAFETY: m x h x d halves, m x h floats.
    check(unsafe { (api()?.head_gate)(g.raw(), x, m as i64, h as i64, d as i64, logits) }, "head gate")
}

/// out [m, c] half = (x * (1 + scale) + shift) * s, scale / shift float32 [c] (null: none)
#[allow(clippy::too_many_arguments)]
pub fn mod_rows(g: &Gpu, x: *const f32, m: usize, c: usize, scale: *const f32, shift: *const f32, s: f32, out: M) -> Result<()> {
    // SAFETY: m x c values each way, c each table.
    check(unsafe { (api()?.mod_rows)(g.raw(), x, m as i64, c as i64, scale, shift, s, out) }, "modulate")
}

/// out [m, c] float32 = layernorm(x) * (1 + scale[rows[r]]) + shift[rows[r]]
#[allow(clippy::too_many_arguments)]
pub fn ln_mod(g: &Gpu, x: *const f32, m: usize, c: usize, eps: f32, rows: *const i32, scale: *const f32, shift: *const f32, out: *mut f32) -> Result<()> {
    // SAFETY: the caller's buffers hold the rows named.
    check(unsafe { (api()?.ln_mod)(g.raw(), x, m as i64, c as i64, eps, rows, scale, shift, out) }, "layer norm (modulated)")
}
