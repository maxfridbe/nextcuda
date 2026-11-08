//! This engine's own kernels (`kernels/image/zimage/zi.h`), bound from the image kernel library by name.

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

type G = *mut c_void;
type M = *mut c_void;
type P = *const c_void;

pub struct Api {
    pub group_norm: unsafe extern "C" fn(G, P, i64, i64, i64, *const f32, *const f32, f32, M) -> c_int,
    pub up2: unsafe extern "C" fn(G, P, i64, i64, i64, M) -> c_int,
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
                // SAFETY: the symbol is declared in zi.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Api { group_norm: sym!("ns_image_zi_group_norm"), up2: sym!("ns_image_zi_up2") })
    })
    .as_ref()
    .map_err(|e| nextcuda_core::Error(e.clone()))
}

pub use nextcuda_zimage::ffi::check;
