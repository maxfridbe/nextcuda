//! The CPU products checked: the plain ones against the blocks' values summed in float64 (an independent reading of
//! the format), the vector paths against the plain ones (to float rounding), on made-up blocks and - with `NS_TEST_NH_BLOCKS=DIR` - on rows of Nemotron's own file (`TYPE_role.bin`
//! and `.txt`: "rows cols tensor").

use super::blocks::{self, dequantize, format, quantize, Act, Quant, IQ1_S, IQ2_S, IQ2_XXS, IQ3_XXS, IQ4_NL, Q2_0, Q8_0};
use super::{Cpu, Isa};

use super::TYPES;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

/// rows of random blocks of `ty`, each block's half scale a small positive number
fn random_rows(ty: i32, rows: usize, cols: usize, r: &mut Rng) -> Vec<u8> {
    let (qk, bytes, _) = format(ty).unwrap();
    let mut w: Vec<u8> = (0..rows * cols / qk * bytes).map(|_| r.next() as u8).collect();
    for blk in w.chunks_exact_mut(bytes) {
        let d = blocks::f32_to_f16(0.002 + 0.05 * r.f().abs());
        blk[..2].copy_from_slice(&d.to_le_bytes());
    }
    w
}

/// the activation's int8 in the values' own order (Act::Q20 keeps them in Q2_0's)
fn natural(q: &Quant) -> Vec<i8> {
    if q.act != Act::Q20 {
        return q.qs.clone();
    }
    let mut o = vec![0i8; q.qs.len()];
    for g in 0..o.len() / 64 {
        for h in 0..2 {
            for b in 0..8 {
                for i in 0..4 {
                    o[64 * g + 32 * h + 4 * b + i] = q.qs[64 * g + 16 * i + 8 * h + b];
                }
            }
        }
    }
    o
}

fn act_values(q: &Quant) -> Vec<f64> {
    let per = if q.act == Act::Q8K { 256 } else { 32 };
    natural(q).iter().enumerate().map(|(i, &v)| q.d[i / per] as f64 * v as f64).collect()
}

/// (the products against the float64 reference, worst error as a part of the sum of |terms|; the vector paths'
/// worst difference from the plain ones, likewise)
fn check(ty: i32, w: &[u8], rows: usize, cols: usize, x: &[f32]) -> (f64, f64) {
    let q = quantize(ty, x);
    let ya = act_values(&q);
    let rb = blocks::row_bytes(ty, cols);
    let (mut worst_ref, mut worst_vec) = (0f64, 0f64);
    let paths: Vec<Cpu> = [Isa::Avx2, Isa::Avx512].into_iter().filter(|&i| i <= Isa::detect()).map(|isa| Cpu::new(isa, false)).collect();
    for r in 0..rows {
        let row = &w[r * rb..(r + 1) * rb];
        let s = blocks::dot(ty, row, &q);
        let wd = dequantize(ty, row);
        let (mut exact_sum, mut mag) = (0f64, 0f64);
        for (a, b) in wd.iter().zip(&ya) {
            exact_sum += *a as f64 * b;
            mag += (*a as f64 * b).abs();
        }
        worst_ref = worst_ref.max((s as f64 - exact_sum).abs() / mag.max(1e-30));
        for p in &paths {
            let v = p.dot(ty, row, &q);
            worst_vec = worst_vec.max((v as f64 - s as f64).abs() / mag.max(1e-30));
        }
    }
    (worst_ref, worst_vec)
}

#[test]
fn the_products_on_made_up_blocks() {
    let mut r = Rng(0x9e37_79b9_7f4a_7c15);
    for ty in TYPES {
        let cols = if format(ty).unwrap().0 == 256 { 1024 } else { 2688 };
        let w = random_rows(ty, 48, cols, &mut r);
        let x: Vec<f32> = (0..cols).map(|_| r.f() * 3.0).collect();
        let (er, ev) = check(ty, &w, 48, cols, &x);
        eprintln!("type {ty}: plain vs float64 {er:.2e}, vector vs plain {ev:.2e}");
        assert!(er < 2e-5, "type {ty}: the plain products are off the blocks' values by {er:e}");
        assert!(ev < 2e-6, "type {ty}: the vector products are off the plain ones by {ev:e}");
    }
}

#[test]
fn the_quantizers() {
    let mut r = Rng(12345);
    let x: Vec<f32> = (0..2048).map(|_| r.f() * 7.0).collect();
    let k = blocks::quantize_q8k(&x);
    for (i, &v) in x.iter().enumerate() {
        let d = k.d[i / 256];
        assert!((d * k.qs[i] as f32 - v).abs() <= d.abs() * 0.5001, "Q8_K value {i}");
    }
    for j in 0..2048 / 16 {
        assert_eq!(k.bsums[j], k.qs[j * 16..j * 16 + 16].iter().map(|&v| v as i16).sum::<i16>());
    }
    let z = blocks::quantize_q80(&x);
    for (i, &v) in x.iter().enumerate() {
        let d = z.d[i / 32];
        // the scale is kept as half: the values were rounded against the unrounded one
        assert!((d * z.qs[i] as f32 - v).abs() <= d * 0.51 + d * 127.0 * 1e-3, "Q8_0 value {i}");
    }
    let q2 = blocks::quantize_q20(&x);
    assert_eq!(natural(&q2), z.qs, "Q2_0's order holds Q8_0's values");
    for b in 0..2048 / 32 {
        assert_eq!(q2.bsums[b], z.qs[b * 32..b * 32 + 32].iter().map(|&v| v as i16).sum::<i16>());
    }
    assert!(blocks::quantize_q8k(&[0.0; 256]).qs.iter().all(|&v| v == 0));
    // every finite half, read without branches as the products read their scales
    for h in (0..0x7c00u16).chain(0x8000..0xfc00) {
        assert_eq!(blocks::half_bits(h).to_bits(), nextcuda_gguf::safetensors::f16_to_f32(h).to_bits(), "{h:#x}");
    }
    for f in [0.0f32, 1.0, -2.5, 65504.0, 1e-6, 6.1e-5, 0.333] {
        assert_eq!(nextcuda_gguf::safetensors::f16_to_f32(blocks::f32_to_f16(f)), half_ref(f), "{f}");
    }
}

/// float -> half -> float by search: the nearest half (ties to even mantissa)
fn half_ref(f: f32) -> f32 {
    let mut best = (f32::INFINITY, 0u16);
    for h in 0..=0x7bffu16 {
        let v = nextcuda_gguf::safetensors::f16_to_f32(h).copysign(f);
        let e = (v - f).abs();
        if e < best.0 || (e == best.0 && h & 1 == 0) {
            best = (e, h);
        }
    }
    nextcuda_gguf::safetensors::f16_to_f32(best.1).copysign(f)
}

/// rows of the file's own blocks (NS_TEST_NH_BLOCKS), when there
fn real_rows() -> Vec<(i32, String, usize, usize, Vec<u8>)> {
    let Ok(dir) = std::env::var("NS_TEST_NH_BLOCKS") else { return Vec::new() };
    let names = [(IQ1_S, "IQ1_S_up"), (IQ2_XXS, "IQ2_XXS_up"), (IQ2_S, "IQ2_S_up"), (IQ3_XXS, "IQ3_XXS_up"), (Q8_0, "Q8_0_up"), (Q2_0, "Q2_0_down"), (IQ4_NL, "IQ4_NL_down"), (Q8_0, "Q8_0_down")];
    names
        .iter()
        .filter_map(|&(ty, n)| {
            let txt = std::fs::read_to_string(format!("{dir}/{n}.txt")).ok()?;
            let mut it = txt.split_whitespace();
            let (rows, cols): (usize, usize) = (it.next()?.parse().ok()?, it.next()?.parse().ok()?);
            Some((ty, n.to_string(), rows, cols, std::fs::read(format!("{dir}/{n}.bin")).ok()?))
        })
        .collect()
}

/// an activation like the engine's: the latent row (up), relu^2 of an up product (down)
fn activation(cols: usize, r: &mut Rng, down: bool) -> Vec<f32> {
    (0..cols)
        .map(|_| {
            let v = r.f() * 2.0;
            if down {
                let v = v.max(0.0);
                v * v
            } else {
                v
            }
        })
        .collect()
}

#[test]
fn the_products_on_the_files_blocks() {
    let rows = real_rows();
    if rows.is_empty() {
        eprintln!("NS_TEST_NH_BLOCKS not set: skipped");
        return;
    }
    let mut r = Rng(777);
    for (ty, name, nrows, cols, w) in rows {
        let x = activation(cols, &mut r, name.ends_with("down"));
        let (er, ev) = check(ty, &w, nrows, cols, &x);
        eprintln!("{name}: {nrows} rows of {cols}: plain vs float64 {er:.2e}, vector vs plain {ev:.2e}");
        assert!(er < 2e-5 && ev < 2e-6, "{name}");
    }
}
