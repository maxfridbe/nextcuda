//! The GGUF blocks the CPU's experts are stored in, the activation's quantized forms they are multiplied with, and the
//! plain (scalar) products: the reference the vector paths (x86.rs) are checked against, and the path on a CPU that
//! has neither AVX2 nor AVX-512. The arithmetic is the format's: integer sums a block (exact), then float32 scales, in
//! the order a reader of these files sums them, so the result is the same on every path to float rounding.
//!
//! Types: the up matrices' IQ1_S, IQ2_XXS, IQ2_S, IQ3_XXS (against the activation as Q8_K: 256 values a block, a
//! float32 scale and the sums of each 16) and the down matrices' Q2_0, IQ4_NL (against Q8_0: 32 values, a half
//! scale), and Q8_0 for either - the types Nemotron 3.5 Super UD-IQ1_M's experts are stored in.

use super::grids::{IQ1S_GRID, IQ2S_GRID, IQ2XXS_GRID, IQ3XXS_GRID, IQ4NL_VALUES};
use nextcuda_gguf::safetensors::f16_to_f32;

pub const Q8_0: i32 = 8;
pub const IQ2_XXS: i32 = 16;
pub const IQ3_XXS: i32 = 18;
pub const IQ1_S: i32 = 19;
pub const IQ4_NL: i32 = 20;
pub const IQ2_S: i32 = 22;
pub const Q2_0: i32 = 42;

/// IQ1_S's grid values are shifted by this, the sign of the shift a bit of each 32 (the format's IQ1S_DELTA)
const IQ1S_DELTA: f32 = 0.125;

/// The activation's quantized form a weight type is multiplied with
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Act {
    /// 256 values a block: int8, a float32 scale, the sum of each 16
    Q8K,
    /// 32 values a block: int8, a half scale (kept here as the float32 it stands for)
    Q80,
    /// Q8_0's values in Q2_0's order: per 64 (a Q2_0 block, two of Q8_0's) value 32h + 4b + i at 16i + 8h + b - the
    /// order a Q2_0 block's bits come out in when each byte is shifted by 2i - with each 32's sum
    Q20,
}

/// (values a block, bytes a block, the activation's form) of a type the CPU multiplies
pub fn format(ty: i32) -> Option<(usize, usize, Act)> {
    Some(match ty {
        IQ2_XXS => (256, 66, Act::Q8K),
        IQ3_XXS => (256, 98, Act::Q8K),
        IQ1_S => (256, 50, Act::Q8K),
        IQ2_S => (256, 82, Act::Q8K),
        IQ4_NL => (32, 18, Act::Q80),
        Q2_0 => (64, 18, Act::Q20),
        Q8_0 => (32, 34, Act::Q80),
        _ => return None,
    })
}

/// The type's name in the GGUF tables
pub fn name(ty: i32) -> &'static str {
    match ty {
        IQ2_XXS => "IQ2_XXS",
        IQ3_XXS => "IQ3_XXS",
        IQ1_S => "IQ1_S",
        IQ2_S => "IQ2_S",
        IQ4_NL => "IQ4_NL",
        Q2_0 => "Q2_0",
        Q8_0 => "Q8_0",
        _ => "?",
    }
}

/// The bytes of a row of n values of `ty` (n a multiple of its block)
pub fn row_bytes(ty: i32, n: usize) -> usize {
    let (qk, bytes, _) = format(ty).expect("a type the CPU multiplies");
    n / qk * bytes
}

/// A row of activations quantized: the values (int8), a scale a block, for Q8_K the sums of each 16 values, for
/// Q20 each 32's sum; and for Q20 per 64 the scales as the vector paths' int32 lanes take them (`pat`: the first
/// 32's, twice, the second's, twice, all that again) and the part the values' shift by one takes off (`c`:
/// sum over the two 32s of scale x sum)
#[derive(Clone, Debug, PartialEq)]
pub struct Quant {
    pub act: Act,
    pub qs: Vec<i8>,
    pub d: Vec<f32>,
    pub bsums: Vec<i16>,
    pub pat: Vec<f32>,
    pub c: Vec<f32>,
}

/// float32 -> half, rounded to nearest even (the scale a Q8_0 block keeps)
pub fn f32_to_f16(f: f32) -> u16 {
    let u = f.to_bits();
    let sign = ((u >> 16) & 0x8000) as u16;
    let exp = ((u >> 23) & 0xff) as i32;
    let man = u & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 31 {
        return sign | 0x7c00;
    }
    let (m, shift) = if e <= 0 {
        if e < -10 {
            return sign;
        }
        (man | 0x80_0000, (14 - e) as u32)
    } else {
        (man, 13u32)
    };
    let half = 1u32 << (shift - 1);
    let rest = m & ((1u32 << shift) - 1);
    let mut h = m >> shift;
    if rest > half || (rest == half && h & 1 == 1) {
        h += 1;
    }
    // a carry out of the mantissa moves into the exponent, as the bits are laid out
    if e <= 0 {
        sign | h as u16
    } else {
        sign | (((e as u32) << 10) + h) as u16
    }
}

/// x rounded to the nearest integer, ties to even (|x| < 2^22)
#[inline]
fn nearest(x: f32) -> i32 {
    ((x + 12_582_912.0f32).to_bits() & 0x007f_ffff) as i32 - 0x0040_0000
}

/// x as Q8_K: per 256 the value of the largest magnitude maps to -127 (its sign kept in the scale), the rest
/// rounded to nearest even
pub fn quantize_q8k(x: &[f32]) -> Quant {
    assert!(x.len().is_multiple_of(256), "Q8_K rows are whole blocks of 256");
    let nb = x.len() / 256;
    let mut q = Quant { act: Act::Q8K, qs: vec![0; x.len()], d: vec![0.0; nb], bsums: vec![0; nb * 16], pat: Vec::new(), c: Vec::new() };
    for b in 0..nb {
        let xs = &x[b * 256..(b + 1) * 256];
        let (mut amax, mut max) = (0f32, 0f32);
        for &v in xs {
            if v.abs() > amax {
                amax = v.abs();
                max = v;
            }
        }
        if amax == 0.0 {
            continue;
        }
        let iscale = -127.0f32 / max;
        let qs = &mut q.qs[b * 256..(b + 1) * 256];
        for (o, &v) in qs.iter_mut().zip(xs) {
            *o = nearest(iscale * v).min(127) as i8;
        }
        for j in 0..16 {
            q.bsums[b * 16 + j] = qs[j * 16..(j + 1) * 16].iter().map(|&v| v as i16).sum();
        }
        q.d[b] = 1.0 / iscale;
    }
    q
}

/// x as Q8_0: per 32 the largest magnitude maps to 127, the scale kept as half, the values rounded to nearest even
pub fn quantize_q80(x: &[f32]) -> Quant {
    assert!(x.len().is_multiple_of(32), "Q8_0 rows are whole blocks of 32");
    let nb = x.len() / 32;
    let mut q = Quant { act: Act::Q80, qs: vec![0; x.len()], d: vec![0.0; nb], bsums: Vec::new(), pat: Vec::new(), c: Vec::new() };
    for b in 0..nb {
        let xs = &x[b * 32..(b + 1) * 32];
        let amax = xs.iter().fold(0f32, |m, v| m.max(v.abs()));
        q.d[b] = f16_to_f32(f32_to_f16(amax / 127.0));
        let id = if amax != 0.0 { 127.0 / amax } else { 0.0 };
        for (o, &v) in q.qs[b * 32..(b + 1) * 32].iter_mut().zip(xs) {
            *o = nearest(v * id) as i8;
        }
    }
    q
}

/// x as Q8_0, in Q2_0's order (Act::Q20)
pub fn quantize_q20(x: &[f32]) -> Quant {
    assert!(x.len().is_multiple_of(64), "Q2_0 rows are whole blocks of 64");
    let mut q = quantize_q80(x);
    let src = std::mem::take(&mut q.qs);
    q.qs = vec![0; src.len()];
    for g in 0..src.len() / 64 {
        for h in 0..2 {
            for b in 0..8 {
                for i in 0..4 {
                    q.qs[64 * g + 16 * i + 8 * h + b] = src[64 * g + 32 * h + 4 * b + i];
                }
            }
        }
    }
    q.bsums = src.chunks(32).map(|c| c.iter().map(|&v| v as i16).sum()).collect();
    for g in 0..src.len() / 64 {
        let (d0, d1) = (q.d[2 * g], q.d[2 * g + 1]);
        q.pat.extend([d0, d0, d1, d1, d0, d0, d1, d1]);
        q.c.push(d0 * q.bsums[2 * g] as f32 + d1 * q.bsums[2 * g + 1] as f32);
    }
    q.act = Act::Q20;
    q
}

/// x quantized as `ty`'s products take it
pub fn quantize(ty: i32, x: &[f32]) -> Quant {
    match format(ty).expect("a type the CPU multiplies").2 {
        Act::Q8K => quantize_q8k(x),
        Act::Q80 => quantize_q80(x),
        Act::Q20 => quantize_q20(x),
    }
}

#[inline]
pub fn half(b: &[u8], at: usize) -> f32 {
    half_bits(u16::from_le_bytes([b[at], b[at + 1]]))
}

/// A half's value without branches: its exponent and mantissa placed as a float's, scaled by 2^112 to rebias the
/// exponent - exact for every finite half, subnormals and zero too (a scale is never infinite or NaN)
#[inline(always)]
pub fn half_bits(h: u16) -> f32 {
    let v = f32::from_bits(((h & 0x7fff) as u32) << 13) * f32::from_bits(0x7780_0000);
    f32::from_bits(v.to_bits() | ((h & 0x8000) as u32) << 16)
}

#[inline]
fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

/// The 8-bit sign mask a 7-bit code stands for: bit 7 makes the count of negative values even
#[inline]
pub const fn signs7(k: u32) -> u32 {
    let k = k & 127;
    k | ((k.count_ones() & 1) << 7)
}

#[inline]
fn grid_byte(g: u64, j: usize) -> i32 {
    ((g >> (8 * j)) & 0xff) as i32
}

/// IQ2_XXS (66 bytes / 256): a half scale, then per 32 values 4 grid indices (a byte each) and a word of 4 x 7 sign
/// bits and a 4-bit scale
fn dot_iq2xxs(w: &[u8], a: &Quant) -> f32 {
    let mut sumf = 0f32;
    for (i, blk) in w.as_chunks::<66>().0.iter().enumerate() {
        let d = half(blk, 0) * a.d[i];
        let q8 = &a.qs[i * 256..];
        let mut bsum = 0i32;
        for ib in 0..8 {
            let (idx, aux) = (&blk[2 + 8 * ib..2 + 8 * ib + 4], u32_at(blk, 2 + 8 * ib + 4));
            let ls = 2 * (aux >> 28) as i32 + 1;
            let mut sumi = 0i32;
            for l in 0..4 {
                let g = IQ2XXS_GRID[idx[l] as usize];
                let s = signs7(aux >> (7 * l));
                for j in 0..8 {
                    let v = grid_byte(g, j) * q8[ib * 32 + l * 8 + j] as i32;
                    sumi += if s >> j & 1 == 1 { -v } else { v };
                }
            }
            bsum += sumi * ls;
        }
        sumf += d * bsum as f32;
    }
    0.125 * sumf
}

/// IQ2_S (82 bytes / 256): a half scale, 32 low bytes of grid indices, 32 sign bytes, 8 bytes of the indices' high
/// bits, 8 bytes of two 4-bit scales
fn dot_iq2s(w: &[u8], a: &Quant) -> f32 {
    let mut sumf = 0f32;
    for (i, blk) in w.as_chunks::<82>().0.iter().enumerate() {
        let d = half(blk, 0) * a.d[i];
        let (qs, signs, qh, sc) = (&blk[2..34], &blk[34..66], &blk[66..74], &blk[74..82]);
        let q8 = &a.qs[i * 256..];
        let mut bsum = 0i32;
        for ib in 0..8 {
            let ls = [1 + 2 * (sc[ib] & 0xf) as i32, 1 + 2 * (sc[ib] >> 4) as i32];
            let mut sumi = [0i32; 2];
            for l in 0..4 {
                let g = IQ2S_GRID[(qs[ib * 4 + l] as usize) | ((qh[ib] as usize) << (8 - 2 * l) & 0x300)];
                let s = signs[ib * 4 + l];
                for j in 0..8 {
                    let v = grid_byte(g, j) * q8[ib * 32 + l * 8 + j] as i32;
                    sumi[l / 2] += if s >> j & 1 == 1 { -v } else { v };
                }
            }
            bsum += ls[0] * sumi[0] + ls[1] * sumi[1];
        }
        sumf += d * bsum as f32;
    }
    0.125 * sumf
}

/// IQ3_XXS (98 bytes / 256): a half scale, 64 grid indices (4 values each), then per 32 values a word of 4 x 7 sign
/// bits and a 4-bit scale
fn dot_iq3xxs(w: &[u8], a: &Quant) -> f32 {
    let mut sumf = 0f32;
    for (i, blk) in w.as_chunks::<98>().0.iter().enumerate() {
        let d = half(blk, 0) * a.d[i];
        let q3 = &blk[2..66];
        let q8 = &a.qs[i * 256..];
        let mut bsum = 0i32;
        for ib in 0..8 {
            let aux = u32_at(blk, 66 + 4 * ib);
            let ls = 2 * (aux >> 28) as i32 + 1;
            let mut sumi = 0i32;
            for l in 0..4 {
                let s = signs7(aux >> (7 * l));
                for (h, &gi) in q3[ib * 8 + 2 * l..ib * 8 + 2 * l + 2].iter().enumerate() {
                    let g = IQ3XXS_GRID[gi as usize] as u64;
                    for j in 0..4 {
                        let v = grid_byte(g, j) * q8[ib * 32 + l * 8 + h * 4 + j] as i32;
                        sumi += if s >> (h * 4 + j) & 1 == 1 { -v } else { v };
                    }
                }
            }
            bsum += sumi * ls;
        }
        sumf += d * bsum as f32;
    }
    0.25 * sumf
}

/// IQ1_S (50 bytes / 256): a half scale, 32 low bytes of grid indices (values -1 / 0 / 1), then per 32 values a
/// 16-bit word: the indices' high bits (4 x 3), a 3-bit scale and the shift's sign
fn dot_iq1s(w: &[u8], a: &Quant) -> f32 {
    let mut sumf = 0f32;
    for (i, blk) in w.as_chunks::<50>().0.iter().enumerate() {
        let qs = &blk[2..34];
        let q8 = &a.qs[i * 256..];
        let (mut sumi, mut sumi1) = (0i32, 0i32);
        for ib in 0..8 {
            let qh = u16::from_le_bytes([blk[34 + 2 * ib], blk[35 + 2 * ib]]) as usize;
            let ls = 2 * ((qh >> 12) & 7) as i32 + 1;
            let delta = if qh & 0x8000 != 0 { -1 } else { 1 };
            let mut lsum = 0i32;
            for l in 0..4 {
                let g = IQ1S_GRID[qs[ib * 4 + l] as usize | (((qh >> (3 * l)) & 7) << 8)];
                for j in 0..8 {
                    lsum += q8[ib * 32 + l * 8 + j] as i32 * ((g >> (8 * j)) as u8 as i8) as i32;
                }
            }
            sumi += ls * lsum;
            sumi1 += ls * delta * (a.bsums[i * 16 + 2 * ib] as i32 + a.bsums[i * 16 + 2 * ib + 1] as i32);
        }
        sumf += half(blk, 0) * a.d[i] * (sumi as f32 + IQ1S_DELTA * sumi1 as f32);
    }
    sumf
}

/// IQ4_NL (18 bytes / 32): a half scale, 16 bytes of two 4-bit indices into the sixteen levels (the low nibbles the
/// first 16 values)
fn dot_iq4nl(w: &[u8], a: &Quant) -> f32 {
    let mut sumf = 0f32;
    for (i, blk) in w.as_chunks::<18>().0.iter().enumerate() {
        let d = a.d[i] * half(blk, 0);
        let y = &a.qs[i * 32..];
        let (mut s1, mut s2) = (0i32, 0i32);
        for j in 0..16 {
            s1 += y[j] as i32 * IQ4NL_VALUES[(blk[2 + j] & 0xf) as usize] as i32;
            s2 += y[j + 16] as i32 * IQ4NL_VALUES[(blk[2 + j] >> 4) as usize] as i32;
        }
        sumf += d * (s1 + s2) as f32;
    }
    sumf
}

/// Q2_0 (18 bytes / 64): a half scale, 16 bytes of 2-bit values (four a byte, the lowest bits first) standing for
/// -1, 0, 1, 2; a block is two of the activation's 32s (in Act::Q20's order: value 32h + 4b + i at 16i + 8h + b)
fn dot_q20(w: &[u8], a: &Quant) -> f32 {
    let mut sumf = 0f32;
    for (g, blk) in w.as_chunks::<18>().0.iter().enumerate() {
        let d0 = half(blk, 0);
        let y = &a.qs[64 * g..];
        let mut sumi = 0f32;
        for h in 0..2 {
            // sum of (v - 1) y = sum of v y - the 32's sum
            let mut sb = -(a.bsums[2 * g + h] as i32);
            for b in 0..8 {
                let byte = blk[2 + 8 * h + b];
                for i in 0..4 {
                    sb += ((byte >> (2 * i)) & 3) as i32 * y[16 * i + 8 * h + b] as i32;
                }
            }
            sumi += a.d[2 * g + h] * sb as f32;
        }
        sumf += d0 * sumi;
    }
    sumf
}

/// Q8_0 (34 bytes / 32): a half scale and 32 int8
fn dot_q80(w: &[u8], a: &Quant) -> f32 {
    let mut sumf = 0f32;
    for (i, blk) in w.as_chunks::<34>().0.iter().enumerate() {
        let y = &a.qs[i * 32..];
        let sumi: i32 = (0..32).map(|j| blk[2 + j] as i8 as i32 * y[j] as i32).sum();
        sumf += sumi as f32 * (half(blk, 0) * a.d[i]);
    }
    sumf
}

/// The plain product of a row of `ty` (all its blocks) and the activation quantized for it
pub fn kernel(ty: i32) -> fn(&[u8], &Quant) -> f32 {
    match ty {
        IQ2_XXS => dot_iq2xxs,
        IQ2_S => dot_iq2s,
        IQ3_XXS => dot_iq3xxs,
        IQ1_S => dot_iq1s,
        IQ4_NL => dot_iq4nl,
        Q2_0 => dot_q20,
        Q8_0 => dot_q80,
        _ => panic!("type {ty} is not one the CPU multiplies"),
    }
}

/// A row of `ty` (all its blocks: `w` holds exactly the row) . the activation quantized for it, plainly
#[cfg(test)]
pub fn dot(ty: i32, w: &[u8], a: &Quant) -> f32 {
    kernel(ty)(w, a)
}

/// A row of `ty` as float32: what the blocks stand for (the tests' independent check of the products)
#[cfg(test)]
pub fn dequantize(ty: i32, w: &[u8]) -> Vec<f32> {
    let (qk, bytes, _) = format(ty).expect("a type the CPU multiplies");
    let mut out = Vec::with_capacity(w.len() / bytes * qk);
    for blk in w.chunks_exact(bytes) {
        let d = half(blk, 0);
        match ty {
            IQ2_XXS => {
                for ib in 0..8 {
                    let aux = u32_at(blk, 2 + 8 * ib + 4);
                    let db = d * (0.5 + (aux >> 28) as f32) * 0.25;
                    for l in 0..4 {
                        let g = IQ2XXS_GRID[blk[2 + 8 * ib + l] as usize];
                        let s = signs7(aux >> (7 * l));
                        out.extend((0..8).map(|j| db * grid_byte(g, j) as f32 * if s >> j & 1 == 1 { -1.0 } else { 1.0 }));
                    }
                }
            }
            IQ2_S => {
                for ib in 0..8 {
                    for l in 0..4 {
                        let sc = blk[74 + ib];
                        let db = d * (0.5 + (if l < 2 { sc & 0xf } else { sc >> 4 }) as f32) * 0.25;
                        let g = IQ2S_GRID[(blk[2 + ib * 4 + l] as usize) | ((blk[66 + ib] as usize) << (8 - 2 * l) & 0x300)];
                        let s = blk[34 + ib * 4 + l];
                        out.extend((0..8).map(|j| db * grid_byte(g, j) as f32 * if s >> j & 1 == 1 { -1.0 } else { 1.0 }));
                    }
                }
            }
            IQ3_XXS => {
                for ib in 0..8 {
                    let aux = u32_at(blk, 66 + 4 * ib);
                    let db = d * (0.5 + (aux >> 28) as f32) * 0.5;
                    for l in 0..4 {
                        let s = signs7(aux >> (7 * l));
                        for h in 0..2 {
                            let g = IQ3XXS_GRID[blk[2 + ib * 8 + 2 * l + h] as usize] as u64;
                            out.extend((0..4).map(|j| db * grid_byte(g, j) as f32 * if s >> (h * 4 + j) & 1 == 1 { -1.0 } else { 1.0 }));
                        }
                    }
                }
            }
            IQ1_S => {
                for ib in 0..8 {
                    let qh = u16::from_le_bytes([blk[34 + 2 * ib], blk[35 + 2 * ib]]) as usize;
                    let dl = d * (2 * ((qh >> 12) & 7) + 1) as f32;
                    let delta = if qh & 0x8000 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA };
                    for l in 0..4 {
                        let g = IQ1S_GRID[blk[2 + ib * 4 + l] as usize | (((qh >> (3 * l)) & 7) << 8)];
                        out.extend((0..8).map(|j| dl * (((g >> (8 * j)) as u8 as i8) as f32 + delta)));
                    }
                }
            }
            IQ4_NL => {
                out.extend((0..16).map(|j| d * IQ4NL_VALUES[(blk[2 + j] & 0xf) as usize] as f32));
                out.extend((0..16).map(|j| d * IQ4NL_VALUES[(blk[2 + j] >> 4) as usize] as f32));
            }
            Q2_0 => out.extend((0..64).map(|j| d * (((blk[2 + j / 4] >> (2 * (j % 4))) & 3) as i32 - 1) as f32)),
            Q8_0 => out.extend((0..32).map(|j| d * blk[2 + j] as i8 as f32)),
            _ => unreachable!(),
        }
    }
    out
}
