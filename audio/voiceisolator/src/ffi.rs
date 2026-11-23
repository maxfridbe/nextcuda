//! The kernels of the audio library this engine needs beyond the shared diffusion ones (nsd: the float32 products on
//! cuBLAS, the layer norm, GELU, the batched attention): its own (`kernels/audio/voiceisolator/vi.h`).

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

type G = *mut c_void;
type F = *const f32;
type W = *mut f32;

#[allow(clippy::type_complexity)]
pub struct Api {
    pub swap01: unsafe extern "C" fn(G, F, i64, i64, i64, W) -> c_int,
    pub add_seq: unsafe extern "C" fn(G, W, i64, i64, i64, F) -> c_int,
    pub film: unsafe extern "C" fn(G, W, i64, i64, F, F) -> c_int,
    pub rope: unsafe extern "C" fn(G, W, i64, i64, i64, i64, F, F) -> c_int,
    pub exclusive: unsafe extern "C" fn(G, W, F, i64, i64, i64, i64) -> c_int,
    pub short_attention: unsafe extern "C" fn(G, F, i64, i64, i64, i64, W) -> c_int,
    pub dup: unsafe extern "C" fn(G, F, i64, i64, W) -> c_int,
    pub conv_silu: unsafe extern "C" fn(G, F, i64, i64, i64, i64, F, i64, F, c_int, W) -> c_int,
    pub softplus: unsafe extern "C" fn(G, W, i64) -> c_int,
    pub scan: unsafe extern "C" fn(G, F, F, F, F, F, i64, i64, F, i64, i64, i64, i64, i64, i64, c_int, W) -> c_int,
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
                    return Err(format!("{} is not in the kernel library (the audio library, rebuilt: kernels/build.sh)", $name));
                }
                // SAFETY: the symbol is declared in vi.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Api {
            swap01: sym!("ns_audio_vi_swap01"),
            add_seq: sym!("ns_audio_vi_add_seq"),
            film: sym!("ns_audio_vi_film"),
            rope: sym!("ns_audio_vi_rope"),
            exclusive: sym!("ns_audio_vi_exclusive"),
            short_attention: sym!("ns_audio_vi_short_attention"),
            dup: sym!("ns_audio_vi_dup"),
            conv_silu: sym!("ns_audio_vi_conv_silu"),
            softplus: sym!("ns_audio_vi_softplus"),
            scan: sym!("ns_audio_vi_scan"),
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
