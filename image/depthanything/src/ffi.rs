//! The shared kernels the depth network calls beyond `nextcuda_diffusion::kernels::Nsd` (its products, norms, GELU,
//! attention): the vision ones (`kernels/diffusion/vision.h`: im2col, the align-corners resize, ReLU, sums, a row
//! broadcast) and the diffusion library's pixel shuffle (`nsd.h`: a transposed convolution's scatter), bound from the
//! image kernel library by name. The engine has no kernels of its own.

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

use nextcuda_core::{Error, Result};

type G = *mut c_void;

pub struct Api {
    pub im2col: unsafe extern "C" fn(G, *const f32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, *mut f32) -> c_int,
    pub resize_ac: unsafe extern "C" fn(G, *const f32, i64, i64, i64, i64, i64, *mut f32, i64) -> c_int,
    pub relu: unsafe extern "C" fn(G, *mut f32, i64) -> c_int,
    pub add: unsafe extern "C" fn(G, *mut f32, *const f32, i64) -> c_int,
    pub broadcast_row: unsafe extern "C" fn(G, *const f32, i64, i64, *mut f32, i64) -> c_int,
    /// on the diffusion context (`Nsd::raw`), not the GPU
    pub pixel_shuffle_add: unsafe extern "C" fn(G, *const c_void, c_int, i64, i64, i64, i64, i64, *const c_void, *mut c_void) -> c_int,
}

// SAFETY: function pointers into the library, which stays loaded for the process.
unsafe impl Send for Api {}
unsafe impl Sync for Api {}

macro_rules! sym {
    ($lib:expr, $name:literal) => {{
        let p = $lib.symbol($name);
        if p.is_null() {
            return Err(format!("{} is not in the kernel library (the image library, rebuilt: kernels/build.sh)", $name));
        }
        // SAFETY: the symbol is declared in vision.h / nsd.h with the field's signature.
        #[allow(clippy::missing_transmute_annotations)]
        unsafe {
            std::mem::transmute::<*mut c_void, _>(p)
        }
    }};
}

pub fn api() -> Result<&'static Api> {
    static API: OnceLock<std::result::Result<Api, String>> = OnceLock::new();
    API.get_or_init(|| {
        let lib = nextcuda_core::api().map_err(|e| e.0)?;
        Ok(Api {
            im2col: sym!(lib, "ns_vision_im2col"),
            resize_ac: sym!(lib, "ns_vision_resize_ac"),
            relu: sym!(lib, "ns_vision_relu"),
            add: sym!(lib, "ns_vision_add"),
            broadcast_row: sym!(lib, "ns_vision_broadcast_row"),
            pixel_shuffle_add: sym!(lib, "nsd_pixel_shuffle_add"),
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
