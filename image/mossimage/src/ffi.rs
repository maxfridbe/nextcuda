//! This engine's own kernels (`kernels/image/mossimage/mo.h`) and the few of Ming-Image's it shares (`mi.h`: the
//! sampler's step, the conversions, the VAE's upsampling), bound from the image kernel library by name.

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

type G = *mut c_void;
type M = *mut c_void;
type P = *const c_void;

pub struct Api {
    pub situ: unsafe extern "C" fn(G, *const f32, i64, i64, i64, M) -> c_int,
    pub group_norm: unsafe extern "C" fn(G, P, i64, i64, i64, *const f32, *const f32, f32, M) -> c_int,
    pub axpy: unsafe extern "C" fn(G, *mut f32, *const f32, i64, f32) -> c_int,
    pub to_half: unsafe extern "C" fn(G, *const f32, M, i64) -> c_int,
    pub to_float: unsafe extern "C" fn(G, P, *mut f32, i64) -> c_int,
    pub to_half_rows: unsafe extern "C" fn(G, *const f32, i64, i64, f32, M, i64) -> c_int,
    pub up2: unsafe extern "C" fn(G, P, i64, i64, i64, M) -> c_int,
    pub softmax_rows: unsafe extern "C" fn(G, *const f32, i64, i64, f32, M) -> c_int,
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
                // SAFETY: the symbol is declared in mo.h / mi.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Api {
            situ: sym!("ns_image_mo_situ"),
            group_norm: sym!("ns_image_mo_group_norm"),
            axpy: sym!("ns_image_mi_axpy"),
            to_half: sym!("ns_image_mi_to_half"),
            to_float: sym!("ns_image_mi_to_float"),
            to_half_rows: sym!("ns_image_mi_to_half_rows"),
            up2: sym!("ns_image_mi_up2"),
            softmax_rows: sym!("ns_image_mi_softmax_rows"),
        })
    })
    .as_ref()
    .map_err(|e| nextcuda_core::Error(e.clone()))
}

/// The scale q, k and v are put in half at (the attention's output linear's float32 result multiplied back): v reaches
/// past half's range in a DiT this deep
pub const ACT_SCALE: f32 = 1.0 / 256.0;

/// A call's return code as a result (the reason from the library)
pub fn check(rc: c_int, what: &str) -> nextcuda_core::Result<()> {
    if rc == 0 {
        return Ok(());
    }
    let why = nextcuda_core::api().map(|a| a.error()).unwrap_or_default();
    Err(nextcuda_core::Error(format!("{what}: {why}")))
}

/// The model's silo for one card (`kernels/image/mossimage/silo`, built once a card: dist/silo/libnextcuda-mossimage.sm<NN>.so):
/// its attention (tiled for that card; FP8 where the card has it) and the int8 products' fused epilogues
pub struct Silo {
    /// the card the build is for (86, 120) and whether FP8 attention is in it
    pub sm: i32,
    pub fp8: bool,
    pub attention: unsafe extern "C" fn(G, P, i64, i64, P, P, i64, i64, i64, M, c_int, c_int) -> c_int,
    pub epi_half: unsafe extern "C" fn(G, *const i32, i64, i64, *const f32, *const f32, f32, M, i64) -> c_int,
    pub epi_situ: unsafe extern "C" fn(G, *const i32, i64, i64, *const f32, *const f32, M) -> c_int,
    pub epi_gate_add: unsafe extern "C" fn(G, *const i32, i64, i64, *const f32, *const f32, f32, *const f32, *mut f32) -> c_int,
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
        let lib = nextcuda_core::silo_for("mossimage", cc).ok().flatten()?;
        let sym = |n: &str| Some(lib.symbol(n)).filter(|p| !p.is_null());
        let info = sym("ns_image_moss_silo_info")?;
        let (mut sm, mut fp8) = (0 as c_int, 0 as c_int);
        // SAFETY: the symbols are declared in the silo's silo.h with these signatures.
        unsafe {
            let info: unsafe extern "C" fn(*mut c_int, *mut c_int) -> c_int = std::mem::transmute(info);
            info(&mut sm, &mut fp8);
            Some(&*Box::leak(Box::new(Silo {
                sm,
                fp8: fp8 != 0,
                attention: std::mem::transmute::<*mut c_void, _>(sym("ns_image_moss_silo_attention")?),
                epi_half: std::mem::transmute::<*mut c_void, _>(sym("ns_image_moss_silo_epi_half")?),
                epi_situ: std::mem::transmute::<*mut c_void, _>(sym("ns_image_moss_silo_epi_situ")?),
                epi_gate_add: std::mem::transmute::<*mut c_void, _>(sym("ns_image_moss_silo_epi_gate_add")?),
            })))
        }
    })();
    m.push((cc, s));
    s
}
