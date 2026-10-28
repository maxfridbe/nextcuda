//! The kernels of the audio library this engine needs beyond the shared diffusion ones (nsd): its own
//! (`kernels/audio/yue2/yue.h`: the Q8_0 decode product and embedding rows, the attention's inputs and the decode's
//! attention, the flow stage's element-wise steps) and ACE-Step's SnakeBeta (`kernels/audio/acestep/ace.h`: Oobleck's
//! activation, YuE2-Vae's too).

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

type G = *mut c_void;
type M = *mut c_void;
type P = *const c_void;

#[allow(clippy::type_complexity)]
pub struct Api {
    pub gemv_q8: unsafe extern "C" fn(G, *const f32, i64, i64, P, i64, *mut f32, i64, c_int) -> c_int,
    pub embed_q8: unsafe extern "C" fn(G, P, i64, *const i32, i64, *mut f32) -> c_int,
    pub qkv: unsafe extern "C" fn(G, *const f32, i64, *const f32, *const f32, f32, f32, i64, M, M, M, i64) -> c_int,
    pub attn_scratch: unsafe extern "C" fn(i64) -> i64,
    pub attn: unsafe extern "C" fn(G, P, P, P, i64, *mut f32, *mut f32) -> c_int,
    pub add_vec_rows: unsafe extern "C" fn(G, *mut f32, i64, i64, *const f32, *const f32) -> c_int,
    pub axpy: unsafe extern "C" fn(G, *mut f32, *const f32, *const f32, i64, f32) -> c_int,
    pub snake: unsafe extern "C" fn(G, *const f32, i64, i64, *const f32, *const f32, *mut f32) -> c_int,
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
                // SAFETY: the symbol is declared in yue.h / ace.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Api {
            gemv_q8: sym!("ns_audio_yue2_gemv_q8"),
            embed_q8: sym!("ns_audio_yue2_embed_q8"),
            qkv: sym!("ns_audio_yue2_qkv"),
            attn_scratch: sym!("ns_audio_yue2_attn_scratch"),
            attn: sym!("ns_audio_yue2_attn"),
            add_vec_rows: sym!("ns_audio_yue2_add_vec_rows"),
            axpy: sym!("ns_audio_yue2_axpy"),
            snake: sym!("ns_audio_ace_snake"),
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

/// The model's silo for one card (`kernels/audio/yue2/silo`, built once a card: dist/silo/libnextcuda-yue2.sm<NN>.so): the
/// flow's attention (MOSS-Image's tiles with grouped-query heads)
pub struct Silo {
    /// the card the build is for (86, 120)
    pub sm: i32,
    pub attention: unsafe extern "C" fn(G, P, i64, i64, P, P, i64, i64, i64, i64, M, c_int, c_int) -> c_int,
}

// SAFETY: function pointers into a library that stays loaded for the process.
unsafe impl Send for Silo {}
unsafe impl Sync for Silo {}

/// The silo for a card of compute capability `cc` (Gpu::cc), when it is built (NS_SILO=0: none - the shared kernels)
pub fn silo(cc: u32) -> Option<&'static Silo> {
    static SILOS: std::sync::Mutex<Vec<(u32, Option<&'static Silo>)>> = std::sync::Mutex::new(Vec::new());
    let mut m = SILOS.lock().unwrap();
    if let Some((_, s)) = m.iter().find(|(c, _)| *c == cc) {
        return *s;
    }
    let s = (|| {
        let lib = nextcuda_core::silo_for("yue2", cc).ok().flatten()?;
        let sym = |n: &str| Some(lib.symbol(n)).filter(|p| !p.is_null());
        let info = sym("ns_audio_yue2_silo_info")?;
        let mut sm = 0 as c_int;
        // SAFETY: the symbols are declared in the silo's silo.h with these signatures.
        unsafe {
            let info: unsafe extern "C" fn(*mut c_int) -> c_int = std::mem::transmute(info);
            info(&mut sm);
            Some(&*Box::leak(Box::new(Silo { sm, attention: std::mem::transmute::<*mut c_void, _>(sym("ns_audio_yue2_silo_attention")?) })))
        }
    })();
    m.push((cc, s));
    s
}
