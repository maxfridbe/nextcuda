//! The rasterizer's kernels (`kernels/image/svgrender/svg.h`), bound from the image kernel library by name.

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

use nextcuda_core::{Error, Gpu, Result};

type G = *mut c_void;
type F = *const f32;
type FM = *mut f32;
type PS = *const c_void;

pub struct Api {
    pub bounds: unsafe extern "C" fn(G, i32, PS, F, F, FM) -> c_int,
    pub render: unsafe extern "C" fn(G, i32, i32, i32, PS, F, F, F, F, F, F, i32, i32, i32, FM) -> c_int,
    #[allow(clippy::type_complexity)]
    pub grad: unsafe extern "C" fn(G, i32, i32, i32, PS, F, F, F, F, F, F, i32, F, F, F, FM, FM, FM, FM, FM, FM) -> c_int,
    pub udf: unsafe extern "C" fn(G, i32, i32, F, F, FM, FM, *mut i32) -> c_int,
    pub xing: unsafe extern "C" fn(G, i32, i32, PS, F, f32, FM, FM) -> c_int,
    #[allow(clippy::type_complexity)]
    pub adam: unsafe extern "C" fn(G, i64, FM, FM, FM, FM, *const u8, *const i32, *const i32, i32, F, i32, f32, f32, f32, f32, F, F) -> c_int,
    pub to_u8: unsafe extern "C" fn(G, i32, i32, F, *mut u8) -> c_int,
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
                // SAFETY: the symbol is declared in svg.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Api {
            bounds: sym!("ns_image_svg_bounds"),
            render: sym!("ns_image_svg_render"),
            grad: sym!("ns_image_svg_grad"),
            udf: sym!("ns_image_svg_udf"),
            xing: sym!("ns_image_svg_xing"),
            adam: sym!("ns_image_svg_adam"),
            to_u8: sym!("ns_image_svg_to_u8"),
        })
    })
    .as_ref()
    .map_err(|e| Error(e.clone()))
}

/// The kernel's answer as a Result (the library's last error on -1)
pub fn check(rc: c_int, what: &str) -> Result<()> {
    if rc == 0 {
        return Ok(());
    }
    let why = nextcuda_core::api().map(|a| a.error()).unwrap_or_default();
    Err(Error(format!("{what}: {why}")))
}

/// `g`'s handle for the calls
pub fn g(gpu: &Gpu) -> G {
    gpu.raw()
}
