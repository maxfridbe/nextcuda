//! The decode products, lent to the foundation's other language models (qwen3vl's chat): one to eight rows times a
//! matrix kept in its GGUF blocks on the card - nextcuda's own (nemotronh's `ns_nh_mmvq`; ggml-cuda's MMVQ `ns_q35_mmvq` for a type only it has), the
//! rows first made Q8_1 (`quantize`) - and the logits' argmax. All on the GPU's own stream.

use std::ffi::c_void;

use nextcuda_core::{Error, Gpu, Result};

use crate::ffi;

/// Whether a GGUF type (its code: 12 Q4_K, 14 Q6_K, 8 Q8_0 ...) has a decode product
pub fn supported(ty: u32) -> bool {
    let Ok(k) = ffi::api() else { return false };
    // SAFETY: arithmetic only.
    unsafe { (k.q35_mmvq_supported)(ty as i32) != 0 || (k.mmvq_supported)(ty as i32) != 0 }
}

/// Bytes of `rows` rows of `cols` values made Q8_1 (cols a multiple of 32)
pub fn q8_1_bytes(cols: usize, rows: usize) -> Result<usize> {
    let k = ffi::api()?;
    // SAFETY: arithmetic only.
    Ok(unsafe { (k.q8_1_bytes)(cols as i64, rows as i64) })
}

/// x [rows][cols] float32 -> xq, its Q8_1 blocks (q8_1_bytes(cols, rows) bytes)
pub fn quantize(gpu: &Gpu, x: *const f32, rows: usize, cols: usize, xq: *mut c_void) -> Result<()> {
    if cols % 32 != 0 {
        return Err(Error(format!("quantize: rows of {cols} values are not whole blocks of 32")));
    }
    let k = ffi::api()?;
    // SAFETY: the caller's buffers hold rows x cols floats and their Q8_1 blocks.
    ffi::check(unsafe { (k.quantize)(gpu.raw(), x, rows as i64, cols as i64, xq) }, "quantize")
}

/// y [rows][n_out] = xq [rows] . W^T, W [n_out][n_in] in GGUF type `ty`'s blocks, xq as `quantize` made it (1 to 8 rows)
#[allow(clippy::too_many_arguments)]
pub fn mmvq(gpu: &Gpu, ty: u32, w: *const c_void, xq: *const c_void, y: *mut f32, n_in: usize, n_out: usize, rows: usize) -> Result<()> {
    let k = ffi::api()?;
    // SAFETY: arithmetic only.
    let f = if unsafe { (k.mmvq_supported)(ty as i32) } != 0 {
        k.mmvq
    } else if unsafe { (k.q35_mmvq_supported)(ty as i32) } != 0 {
        k.q35_mmvq
    } else {
        return Err(Error(format!("mmvq: GGUF type {ty} has no decode product")));
    };
    // SAFETY: the matrix in its blocks, xq's Q8_1 rows, y rows x n_out floats.
    ffi::check(unsafe { f(gpu.raw(), ty as i32, w, xq, y, n_in as i64, n_out as i64, rows as i64) }, "mmvq")
}

/// Each row's largest value's index: x [rows][n] float32 -> out [rows] int32
pub fn argmax(gpu: &Gpu, x: *const f32, rows: usize, n: usize, out: *mut i32) -> Result<()> {
    let k = ffi::api()?;
    // SAFETY: x rows x n floats, out rows ints.
    ffi::check(unsafe { (k.argmax)(gpu.raw(), x, rows as i64, n as i64, out) }, "argmax")
}
