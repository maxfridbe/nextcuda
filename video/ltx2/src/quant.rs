//! The text side's weights made small once, on the host: Gemma-4's and the connectors' bfloat16 matrices into GGUF
//! blocks the shared kernels multiply in place (`nsd_linear_q`) - Q4_K (llama.cpp's `quantize_row_q4_K_ref`, its
//! weighted min / scale search, without an importance matrix) and Q8_0 (a scale a block of 32). The DiT's GGUF comes
//! quantized; these are what the text stage needs to sit beside it on two 12 GB cards (see `cache`).

/// float32 -> IEEE half bits, rounded to nearest even
pub fn f16_bits(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let e = ((x >> 23) & 0xff) as i32;
    let m = x & 0x7f_ffff;
    if e == 255 {
        return sign | 0x7c00 | if m != 0 { 0x200 } else { 0 };
    }
    let e2 = e - 127 + 15;
    if e2 >= 31 {
        return sign | 0x7c00;
    }
    if e2 <= 0 {
        if e2 < -10 {
            return sign;
        }
        let m = m | 0x80_0000;
        let shift = (14 - e2) as u32;
        let half = 1u32 << (shift - 1);
        let mut r = m >> shift;
        let rem = m & ((1 << shift) - 1);
        if rem > half || (rem == half && r & 1 == 1) {
            r += 1;
        }
        return sign | r as u16;
    }
    let mut r = ((e2 as u32) << 10) | (m >> 13);
    let rem = m & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && r & 1 == 1) {
        r += 1;
    }
    sign | r as u16
}

pub fn f16_val(h: u16) -> f32 {
    nextcuda_gguf::safetensors::f16_to_f32(h)
}

pub fn bf16_val(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn nearest(x: f32) -> i32 {
    x.round_ties_even() as i32
}

/// llama.cpp's make_qkx2_quants: levels 0..nmax for x with a scale and a min, weighted least squares over a few
/// candidate scales; returns (scale, -min)
#[allow(clippy::too_many_arguments)]
fn qkx2(x: &[f32], w: &[f32], nmax: i32, l: &mut [u8], aux: &mut [u8], rmin: f32, rdelta: f32, nstep: i32) -> (f32, f32) {
    let n = x.len();
    let (mut min, mut max) = (x[0], x[0]);
    let (mut sum_w, mut sum_x) = (w[0], w[0] * x[0]);
    for i in 1..n {
        min = min.min(x[i]);
        max = max.max(x[i]);
        sum_w += w[i];
        sum_x += w[i] * x[i];
    }
    if min > 0.0 {
        min = 0.0;
    }
    if max == min {
        l[..n].fill(0);
        return (0.0, -min);
    }
    let mut iscale = nmax as f32 / (max - min);
    let mut scale = 1.0 / iscale;
    let mut best = 0f32;
    for i in 0..n {
        let q = nearest(iscale * (x[i] - min)).clamp(0, nmax);
        l[i] = q as u8;
        let d = scale * q as f32 + min - x[i];
        best += w[i] * d * d;
    }
    for is in 0..=nstep {
        iscale = (rmin + rdelta * is as f32 + nmax as f32) / (max - min);
        let (mut sl, mut sl2, mut sxl) = (0f32, 0f32, 0f32);
        for i in 0..n {
            let q = nearest(iscale * (x[i] - min)).clamp(0, nmax);
            aux[i] = q as u8;
            let qf = q as f32;
            sl += w[i] * qf;
            sl2 += w[i] * qf * qf;
            sxl += w[i] * qf * x[i];
        }
        let d = sum_w * sl2 - sl * sl;
        if d > 0.0 {
            let mut this_scale = (sum_w * sxl - sum_x * sl) / d;
            let mut this_min = (sl2 * sum_x - sl * sxl) / d;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sxl / sl2;
            }
            let mut err = 0f32;
            for i in 0..n {
                let e = this_scale * aux[i] as f32 + this_min - x[i];
                err += w[i] * e * e;
            }
            if err < best {
                l[..n].copy_from_slice(&aux[..n]);
                best = err;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    (scale, -min)
}

/// Q4_K: 256 values in 144 bytes (half d, half dmin, 12 bytes of 6-bit scales and mins, 128 bytes of nibbles)
pub fn q4_k(x: &[f32], out: &mut Vec<u8>) {
    assert!(x.len().is_multiple_of(256));
    let mut l = [0u8; 256];
    let mut aux = [0u8; 32];
    let mut w = [0f32; 32];
    for b in x.chunks_exact(256) {
        let mut scales = [0f32; 8];
        let mut mins = [0f32; 8];
        let (mut max_scale, mut max_min) = (0f32, 0f32);
        for j in 0..8 {
            let s = &b[32 * j..32 * j + 32];
            let av = (s.iter().map(|v| v * v).sum::<f32>() / 32.0).sqrt();
            for i in 0..32 {
                w[i] = av + s[i].abs();
            }
            let (sc, mn) = qkx2(s, &w, 15, &mut l[32 * j..32 * j + 32], &mut aux, -1.0, 0.1, 20);
            scales[j] = sc;
            mins[j] = mn;
            max_scale = max_scale.max(sc);
            max_min = max_min.max(mn);
        }
        let inv_scale = if max_scale > 0.0 { 63.0 / max_scale } else { 0.0 };
        let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };
        let mut sc = [0u8; 12];
        for j in 0..8 {
            let ls = nearest(inv_scale * scales[j]).clamp(0, 63) as u8;
            let lm = nearest(inv_min * mins[j]).clamp(0, 63) as u8;
            if j < 4 {
                sc[j] = ls;
                sc[j + 4] = lm;
            } else {
                sc[j + 4] = (ls & 0xf) | ((lm & 0xf) << 4);
                sc[j - 4] |= (ls >> 4) << 6;
                sc[j] |= (lm >> 4) << 6;
            }
        }
        let d = f16_bits(max_scale / 63.0);
        let dmin = f16_bits(max_min / 63.0);
        for j in 0..8 {
            let (s, m) = scale_min_k4(j, &sc);
            let dd = f16_val(d) * s as f32;
            if dd == 0.0 {
                continue;
            }
            let dm = f16_val(dmin) * m as f32;
            for i in 0..32 {
                l[32 * j + i] = nearest((b[32 * j + i] + dm) / dd).clamp(0, 15) as u8;
            }
        }
        out.extend(d.to_le_bytes());
        out.extend(dmin.to_le_bytes());
        out.extend(sc);
        for j in (0..256).step_by(64) {
            for i in 0..32 {
                out.push(l[j + i] | (l[j + i + 32] << 4));
            }
        }
    }
}

fn scale_min_k4(j: usize, q: &[u8; 12]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0xf) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }
}

/// Q4_K blocks back to float32 (the check of the quantizer against the kernels' dequantization)
pub fn dq4_k(b: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(b.len() / 144 * 256);
    for blk in b.chunks_exact(144) {
        let d = f16_val(u16::from_le_bytes([blk[0], blk[1]]));
        let dmin = f16_val(u16::from_le_bytes([blk[2], blk[3]]));
        let sc: [u8; 12] = blk[4..16].try_into().unwrap();
        let q = &blk[16..];
        let mut v = [0f32; 256];
        for (c, j0) in (0..256).step_by(64).enumerate() {
            let (s1, m1) = scale_min_k4(2 * c, &sc);
            let (s2, m2) = scale_min_k4(2 * c + 1, &sc);
            for i in 0..32 {
                let byte = q[c * 32 + i];
                v[j0 + i] = d * s1 as f32 * (byte & 0xf) as f32 - dmin * m1 as f32;
                v[j0 + 32 + i] = d * s2 as f32 * (byte >> 4) as f32 - dmin * m2 as f32;
            }
        }
        out.extend(v);
    }
    out
}

/// Q8_0: 32 values in 34 bytes (half d, 32 int8)
pub fn q8_0(x: &[f32], out: &mut Vec<u8>) {
    assert!(x.len().is_multiple_of(32));
    for b in x.chunks_exact(32) {
        let amax = b.iter().fold(0f32, |a, v| a.max(v.abs()));
        let d = amax / 127.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        out.extend(f16_bits(d).to_le_bytes());
        out.extend(b.iter().map(|v| nearest(v * id).clamp(-127, 127) as i8 as u8));
    }
}

/// `rows` rows of `k` values (bfloat16 bytes, row-major) quantized row by row on `threads` threads, `f` applied to
/// each row's values first (a scale folded in, a permutation)
pub fn quantize_rows(src: &[u8], rows: usize, k: usize, ty: nextcuda_gguf::GType, threads: usize, f: &(dyn Fn(usize, &mut Vec<f32>) + Sync)) -> Vec<u8> {
    use nextcuda_gguf::GType;
    let per = ty.bytes(k as u64).expect("whole blocks") as usize;
    let mut out = vec![0u8; per * rows];
    let chunk = rows.div_ceil(threads.max(1));
    std::thread::scope(|s| {
        for (ci, o) in out.chunks_mut(chunk * per).enumerate() {
            s.spawn(move || {
                let mut row = Vec::with_capacity(k);
                let mut q = Vec::with_capacity(per);
                for (ri, dst) in o.chunks_mut(per).enumerate() {
                    let r = ci * chunk + ri;
                    row.clear();
                    row.extend(src[r * k * 2..(r + 1) * k * 2].chunks_exact(2).map(|c| bf16_val(u16::from_le_bytes([c[0], c[1]]))));
                    f(r, &mut row);
                    q.clear();
                    match ty {
                        GType::Q4K => q4_k(&row, &mut q),
                        GType::Q8_0 => q8_0(&row, &mut q),
                        _ => unreachable!("quantize_rows: {ty:?}"),
                    }
                    dst.copy_from_slice(&q);
                }
            });
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_round_trip() {
        for v in [0.0f32, 1.0, -2.5, 65504.0, 1e-5, 3.14159, -0.0001] {
            let h = f16_bits(v);
            assert!((f16_val(h) - v).abs() <= v.abs() * 1e-3 + 1e-7, "{v} -> {}", f16_val(h));
        }
        assert_eq!(f16_bits(1.0), 0x3c00);
        assert_eq!(f16_bits(f32::INFINITY), 0x7c00);
    }

    #[test]
    fn q4k_small_error() {
        // a gaussian-like row (a sum of uniforms): Q4_K's error on such weights is about 0.06 to 0.09 relative
        let mut s = 12345u64;
        let mut u = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5 };
        let x: Vec<f32> = (0..1024).map(|_| (0..4).map(|_| u()).sum::<f32>() * 0.02).collect();
        let mut b = Vec::new();
        q4_k(&x, &mut b);
        assert_eq!(b.len(), 4 * 144);
        let y = dq4_k(&b);
        let e: f32 = x.iter().zip(&y).map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt();
        let n: f32 = x.iter().map(|a| a * a).sum::<f32>().sqrt();
        assert!(e / n < 0.1, "relative error {}", e / n);
    }

    #[test]
    fn half_special_values() {
        assert_eq!(f16_bits(0.0), 0);
        assert_eq!(f16_bits(-0.0), 0x8000);
        assert_eq!(f16_bits(-1.0), 0xbc00);
        assert_eq!(f16_bits(65504.0), 0x7bff, "the largest half");
        assert_eq!(f16_bits(65520.0), 0x7c00, "rounds up to infinity");
        assert_eq!(f16_bits(1e9), 0x7c00);
        assert_eq!(f16_bits(f32::NEG_INFINITY), 0xfc00);
        assert_eq!(f16_bits(f32::NAN) & 0x7c00, 0x7c00);
        assert_ne!(f16_bits(f32::NAN) & 0x3ff, 0, "a NaN stays one");
        // subnormals: 2^-24 is the least, below half of it is zero
        assert_eq!(f16_bits(2f32.powi(-24)), 1);
        assert_eq!(f16_bits(2f32.powi(-25) * 0.99), 0);
        assert_eq!(f16_bits(2f32.powi(-14)), 0x0400, "the least normal");
        // ties to even: 1 + 2^-11 is half way between 1 and 1 + 2^-10
        assert_eq!(f16_bits(1.0 + 2f32.powi(-11)), 0x3c00);
        assert_eq!(f16_bits(1.0 + 3.0 * 2f32.powi(-11)), 0x3c02);
        for h in [0u16, 1, 0x3c00, 0x7bff, 0x8001, 0xc000] {
            assert_eq!(f16_bits(f16_val(h)), h, "{h:#x}");
        }
    }

    #[test]
    fn bf16_is_the_high_half_of_a_float() {
        assert_eq!(bf16_val(0x3f80), 1.0);
        assert_eq!(bf16_val(0xc000), -2.0);
        assert_eq!(bf16_val(0), 0.0);
    }

    #[test]
    fn q8_0_blocks_are_a_half_scale_and_32_int8() {
        let x: Vec<f32> = (0..64).map(|i| (i as f32 - 31.5) * 0.1).collect();
        let mut b = Vec::new();
        q8_0(&x, &mut b);
        assert_eq!(b.len(), 2 * 34);
        // the first block's scale is its largest magnitude over 127; that value is +-127
        let d = f16_val(u16::from_le_bytes([b[0], b[1]]));
        assert!((d - 3.15 / 127.0).abs() < 1e-4, "{d}");
        assert_eq!(b[2] as i8, -127);
        for (i, v) in x[..32].iter().enumerate() {
            let q = b[2 + i] as i8 as f32;
            assert!((q * d - v).abs() <= d * 0.51 + 1e-4, "{i}");
        }
        // an all-zero block has a zero scale and zeros
        let mut z = Vec::new();
        q8_0(&[0.0; 32], &mut z);
        assert_eq!(z, vec![0u8; 34]);
    }

    #[test]
    fn rows_are_quantized_with_the_row_function_applied_first() {
        // two rows of 32 bf16 values; the function scales row 1 by 2 and negates row 0
        let bf = |v: f32| ((v.to_bits() >> 16) as u16).to_le_bytes();
        let src: Vec<u8> = (0..64).flat_map(|i| bf(if i < 32 { 1.0 } else { 0.5 })).collect();
        let f = |r: usize, row: &mut Vec<f32>| row.iter_mut().for_each(|v| *v *= if r == 0 { -1.0 } else { 2.0 });
        for threads in [1, 2, 5] {
            let q = quantize_rows(&src, 2, 32, nextcuda_gguf::GType::Q8_0, threads, &f);
            assert_eq!(q.len(), 68);
            assert_eq!(q[2] as i8, -127, "row 0 negated, {threads} threads");
            assert_eq!(q[36] as i8, 127, "row 1 doubled to 1.0, {threads} threads");
        }
    }
}
