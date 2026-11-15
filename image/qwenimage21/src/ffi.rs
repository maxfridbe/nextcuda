//! This engine's own kernels (`kernels/image/qwenimage21/qi21.h`), bound from the image kernel library by name.

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

type G = *mut c_void;
type M = *mut c_void;
type P = *const c_void;

pub struct Api {
    pub quant_rows: unsafe extern "C" fn(G, *const f32, i64, i64, *mut i8, *mut f32) -> c_int,
    pub softmax_rows: unsafe extern "C" fn(G, *const f32, i64, i64, f32, M) -> c_int,
    pub ln_mod: unsafe extern "C" fn(G, *const f32, i64, i64, f32, *const f32, M) -> c_int,
    pub ln_mod_shift: unsafe extern "C" fn(G, *const f32, i64, i64, f32, *const f32, *const f32, M) -> c_int,
    pub gelu_tanh: unsafe extern "C" fn(G, M, i64) -> c_int,
    pub silu: unsafe extern "C" fn(G, M, i64) -> c_int,
    pub axpy: unsafe extern "C" fn(G, *mut f32, *const f32, i64, f32) -> c_int,
    pub to_half: unsafe extern "C" fn(G, *const f32, M, i64) -> c_int,
    pub to_float: unsafe extern "C" fn(G, P, *mut f32, i64) -> c_int,
    pub up2: unsafe extern "C" fn(G, P, i64, i64, i64, M) -> c_int,
    pub dupup_add: unsafe extern "C" fn(G, P, i64, i64, i64, i64, c_int, M) -> c_int,
    pub to_rgba8: unsafe extern "C" fn(G, P, i64, i64, *mut u8) -> c_int,
    pub pad_br: unsafe extern "C" fn(G, P, i64, i64, i64, M) -> c_int,
    pub avg_down_add: unsafe extern "C" fn(G, P, i64, i64, i64, i64, c_int, M) -> c_int,
    pub cfg_rescale: unsafe extern "C" fn(G, *const f32, *const f32, i64, i64, f32, *mut f32) -> c_int,
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
                    return Err(format!("{} is not in the kernel library (the image library, rebuilt: kernels/build.sh)", $name));
                }
                // SAFETY: the symbol is declared in qi21.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Api {
            quant_rows: sym!("ns_image_qi21_quant_rows"),
            softmax_rows: sym!("ns_image_qi21_softmax_rows"),
            ln_mod: sym!("ns_image_qi21_ln_mod"),
            ln_mod_shift: sym!("ns_image_qi21_ln_mod_shift"),
            gelu_tanh: sym!("ns_image_qi21_gelu_tanh"),
            silu: sym!("ns_image_qi21_silu"),
            axpy: sym!("ns_image_qi21_axpy"),
            to_half: sym!("ns_image_qi21_to_half"),
            to_float: sym!("ns_image_qi21_to_float"),
            up2: sym!("ns_image_qi21_up2"),
            dupup_add: sym!("ns_image_qi21_dupup_add"),
            to_rgba8: sym!("ns_image_qi21_to_rgba8"),
            pad_br: sym!("ns_image_qi21_pad_br"),
            avg_down_add: sym!("ns_image_qi21_avg_down_add"),
            cfg_rescale: sym!("ns_image_qi21_cfg_rescale"),
        })
    })
    .as_ref()
    .map_err(|e| nextcuda_core::Error(e.clone()))
}

/// A call's return code as a result (the reason from the library)
pub fn check(rc: c_int, what: &str) -> nextcuda_core::Result<()> {
    if rc == 0 {
        return Ok(());
    }
    let why = nextcuda_core::api().map(|a| a.error()).unwrap_or_default();
    Err(nextcuda_core::Error(format!("{what}: {why}")))
}

/// The model's silo (`dist/silo/libnextcuda-qwenimage21.so`: kernels/image/qwenimage21/silo), when it is built: the
/// ops it covers, each used where its `_supported` check says it takes the shape - the shared kernels otherwise
pub struct Silo {
    /// attention(gpu, q, k, v, Sq, Skv, Hq, Hkv, qs, kvs, causal, row0, out)
    #[allow(clippy::type_complexity)]
    pub attention: unsafe extern "C" fn(G, P, P, P, i64, i64, i64, i64, i64, i64, c_int, i64, M) -> c_int,
    pub attention_supported: unsafe extern "C" fn(i64, i64, i64) -> c_int,
    /// attention_ex(..., out, mode): mode 1 the 128-row kernel, + 2 half sums in Q K^T, + 4 half sums in P V
    #[allow(clippy::type_complexity)]
    pub attention_ex: Option<unsafe extern "C" fn(G, P, P, P, i64, i64, i64, i64, i64, i64, c_int, i64, M, c_int) -> c_int>,
    /// gemm(gpu, x, M, K, lda, w, N, bias, out, out_f32, ldo, f16acc): out [M, N] = x . w^T + bias, half operands
    #[allow(clippy::type_complexity)]
    pub gemm: Option<unsafe extern "C" fn(G, P, i64, i64, i64, P, i64, *const f32, M, c_int, i64, c_int) -> c_int>,
    /// q8_split(gpu, src, N, K, q, d): a Q8_0 matrix split for gemm_q8
    pub q8_split: Option<unsafe extern "C" fn(G, P, i64, i64, M, M) -> c_int>,
    /// gemm_q8(gpu, x, M, K, lda, q, d, N, gate, out, epi, ldo, f16acc): gemm with a split Q8_0 matrix; epi 0 half
    /// out, 1 float32, 2 float32 out += gate * the product
    #[allow(clippy::type_complexity)]
    pub gemm_q8: Option<unsafe extern "C" fn(G, P, i64, i64, i64, P, P, i64, *const f32, M, c_int, i64, c_int) -> c_int>,
    /// rms_silu(gpu, x, P, C, gamma, eps, out): silu(rms norm(x) * gamma), half
    pub rms_silu: Option<unsafe extern "C" fn(G, P, i64, i64, *const f32, f32, M) -> c_int>,
    /// conv3x3(gpu, x, H, W, Ci, up, w, Co, bias, out, f16acc): a 3 x 3 convolution of x's nearest `up` x upsample,
    /// channels last, weights [Co, 3, 3, Ci]
    #[allow(clippy::type_complexity)]
    pub conv3x3: Option<unsafe extern "C" fn(G, P, i64, i64, i64, c_int, P, i64, *const f32, M, c_int) -> c_int>,
}

// SAFETY: function pointers into a library that stays loaded for the process.
unsafe impl Send for Silo {}
unsafe impl Sync for Silo {}

pub fn silo() -> Option<&'static Silo> {
    static S: OnceLock<Option<Silo>> = OnceLock::new();
    S.get_or_init(|| {
        let lib = nextcuda_core::silo("qwenimage21").ok().flatten()?;
        let a = lib.symbol("ns_image_qi21_silo_attention");
        let s = lib.symbol("ns_image_qi21_silo_attention_supported");
        if a.is_null() || s.is_null() {
            return None;
        }
        let (g, c, x) = (lib.symbol("ns_image_qi21_silo_gemm"), lib.symbol("ns_image_qi21_silo_conv3x3"), lib.symbol("ns_image_qi21_silo_attention_ex"));
        // SAFETY: declared in the silo's attn.cu and gemm.cu with these signatures.
        #[allow(clippy::missing_transmute_annotations)]
        Some(unsafe {
            Silo {
                attention: std::mem::transmute(a),
                attention_supported: std::mem::transmute(s),
                attention_ex: (!x.is_null()).then(|| std::mem::transmute(x)),
                gemm: (!g.is_null()).then(|| std::mem::transmute(g)),
                conv3x3: (!c.is_null()).then(|| std::mem::transmute(c)),
                rms_silu: { let p = lib.symbol("ns_image_qi21_silo_rms_silu"); (!p.is_null()).then(|| std::mem::transmute(p)) },
                q8_split: { let p = lib.symbol("ns_image_qi21_silo_q8_split"); (!p.is_null()).then(|| std::mem::transmute(p)) },
                gemm_q8: { let p = lib.symbol("ns_image_qi21_silo_gemm_q8"); (!p.is_null()).then(|| std::mem::transmute(p)) },
            }
        })
    })
    .as_ref()
}

/// Half sums in the silo's products (`NS_QI_F16ACC`, default 1): each 32-deep slice of a product summed in half on
/// the tensor cores (twice their float32 rate on a GeForce), the slices' sums in float32
pub fn f16acc() -> bool {
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| std::env::var("NS_QI_F16ACC").map_or(true, |v| v != "0"))
}

/// The silo attention's mode: its 128-row kernel, with `f16acc` the P V product's sums in half (the same error as
/// float32's on the model's own q, k, v); the Q K^T product's too with `NS_QI_QK16=1` (4-6x the scores' rounding,
/// about 5 % a step at 2048 x 2048 - off: on the 3060 it moved a picture's fine texture, mean |difference| 0.31 of
/// 255 against float32's 0.14)
pub fn attention_mode() -> i32 {
    static F: OnceLock<i32> = OnceLock::new();
    *F.get_or_init(|| {
        let qk16 = f16acc() && std::env::var("NS_QI_QK16").is_ok_and(|v| v == "1");
        1 | if qk16 { 2 } else { 0 } | if f16acc() { 4 } else { 0 }
    })
}
