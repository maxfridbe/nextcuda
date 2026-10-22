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
    pub rms_silu: unsafe extern "C" fn(G, P, i64, i64, f32, c_int, M) -> c_int,
    pub d2s: unsafe extern "C" fn(G, P, i64, i64, i64, i64, c_int, c_int, c_int, c_int, M) -> c_int,
    pub conv3d: unsafe extern "C" fn(M, P, c_int, i64, i64, i64, i64, P, i64, i64, *const f32, M) -> c_int,
    pub up2: unsafe extern "C" fn(G, P, i64, i64, i64, M) -> c_int,
    pub s2d: unsafe extern "C" fn(G, P, i64, i64, i64, i64, c_int, c_int, c_int, c_int, M) -> c_int,
    pub aa_snake: unsafe extern "C" fn(M, *const f32, i64, i64, i64, *const f32, *const f32, *const f32, *const f32, *mut f32) -> c_int,
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
            rms_silu: sym!("ns_video_lx_rms_silu"),
            d2s: sym!("ns_video_lx_d2s"),
            conv3d: sym!("nsd_conv3d"),
            up2: sym!("ns_video_lx_up2"),
            s2d: sym!("ns_video_lx_s2d"),
            aa_snake: sym!("nsd_aa_snake"),
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

/// out [m, c] half = silu?(x / rms(x)) per row (x half)
pub fn rms_silu(g: &Gpu, x: P, m: usize, c: usize, eps: f32, silu: bool, out: M) -> Result<()> {
    // SAFETY: m x c halves each way.
    check(unsafe { (api()?.rms_silu)(g.raw(), x, m as i64, c as i64, eps, c_int::from(silu), out) }, "rms (silu)")
}

/// 3-D depth to space of a channels-last volume (see lx.h)
#[allow(clippy::too_many_arguments)]
pub fn d2s(g: &Gpu, x: P, t: usize, h: usize, w: usize, c: usize, s: (usize, usize, usize), drop: usize, out: M) -> Result<()> {
    // SAFETY: x holds t h w (c s0 s1 s2) halves, out (t s0 - drop) (h s1) (w s2) c.
    check(unsafe { (api()?.d2s)(g.raw(), x, t as i64, h as i64, w as i64, c as i64, s.0 as c_int, s.1 as c_int, s.2 as c_int, drop as c_int, out) }, "depth to space")
}

/// A k^3 convolution of a channels-last volume [t, h, w, ci] half, zero padding k / 2 (nsd_conv3d), on `nsd`'s context
#[allow(clippy::too_many_arguments)]
pub fn conv3d(ctx: *mut c_void, x: P, thw: (usize, usize, usize), ci: usize, w: P, co: usize, k: usize, bias: *const f32, out: M) -> Result<()> {
    // SAFETY: the caller's buffers hold the sizes named.
    check(unsafe { (api()?.conv3d)(ctx, x, 1, thw.0 as i64, thw.1 as i64, thw.2 as i64, ci as i64, w, co as i64, k as i64, bias, out) }, "conv3d")
}

/// nearest x2 of a channels-last half plane [h, w, c]
pub fn up2(g: &Gpu, x: P, h: usize, w: usize, c: usize, out: M) -> Result<()> {
    // SAFETY: h w c halves in, 4 h w c out.
    check(unsafe { (api()?.up2)(g.raw(), x, h as i64, w as i64, c as i64, out) }, "nearest x2")
}

/// BigVGAN's anti-aliased SnakeBeta over float32 [b, c, l] (log alpha / beta [c], the 12-tap filters), on `nsd`'s
/// context
#[allow(clippy::too_many_arguments)]
pub fn aa_snake(ctx: *mut c_void, x: *const f32, b: usize, c: usize, l: usize, la: *const f32, lb: *const f32, up: *const f32, down: *const f32, out: *mut f32)
                -> Result<()> {
    // SAFETY: the caller's buffers hold the sizes named.
    check(unsafe { (api()?.aa_snake)(ctx, x, b as i64, c as i64, l as i64, la, lb, up, down, out) }, "anti-aliased snake")
}

/// 3-D space to depth with a group mean (see lx.h)
#[allow(clippy::too_many_arguments)]
pub fn s2d(g: &Gpu, x: P, t: usize, h: usize, w: usize, c: usize, s: (usize, usize, usize), groups: usize, out: M) -> Result<()> {
    // SAFETY: x holds t h w c halves, out t h w c / groups.
    check(unsafe { (api()?.s2d)(g.raw(), x, t as i64, h as i64, w as i64, c as i64, s.0 as c_int, s.1 as c_int, s.2 as c_int, groups as c_int, out) }, "space to depth")
}
