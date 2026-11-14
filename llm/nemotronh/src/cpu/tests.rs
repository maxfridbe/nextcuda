//! The CPU products checked: the plain ones against the blocks' values summed in float64 (an independent reading of
//! the format), the vector paths against the plain ones (to float rounding), on made-up blocks and - with `NS_TEST_NH_BLOCKS=DIR` - on rows of Nemotron's own file (`TYPE_role.bin`
//! and `.txt`: "rows cols tensor").

use super::blocks::{self, dequantize, format, quantize, Act, Quant, IQ1_S, IQ2_S, IQ2_XXS, IQ3_XXS, IQ4_NL, Q2_0, Q8_0};
use super::{Cpu, Isa, Out};

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

/// The products against ggml-cpu's (llama.cpp's CPU backend, which this replaced), on the file's blocks: the
/// activation quantized by both (the bytes compared), each product with its own. Run once on the box, with
/// NS_TEST_NH_BLOCKS and NS_TEST_NH_GGML_DIR (libggml-base.so, libggml-cpu.so): `cargo test -- --ignored ggml`.
#[test]
#[ignore]
fn the_products_against_ggml_cpu() {
    use std::ffi::{c_char, c_int, c_void, CString};
    extern "C" {
        fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }
    type FromFloat = unsafe extern "C" fn(*const f32, *mut c_void, i64);
    type VecDot = unsafe extern "C" fn(c_int, *mut f32, usize, *const c_void, usize, *const c_void, usize, c_int);
    #[repr(C)]
    struct Traits {
        from_float: Option<FromFloat>,
        vec_dot: Option<VecDot>,
        vec_dot_type: c_int,
        nrows: i64,
    }
    let dir = std::env::var("NS_TEST_NH_GGML_DIR").expect("NS_TEST_NH_GGML_DIR");
    let open = |n: &str| {
        let c = CString::new(format!("{dir}/{n}")).unwrap();
        // SAFETY: a path; the library stays loaded.
        let h = unsafe { dlopen(c.as_ptr(), 2 | 0x100) };
        assert!(!h.is_null(), "{n}");
        h
    };
    let (_base, cpu) = (open("libggml-base.so"), open("libggml-cpu.so"));
    let sym = |n: &str| {
        let c = CString::new(n).unwrap();
        // SAFETY: a symbol of a loaded library.
        let p = unsafe { dlsym(cpu, c.as_ptr()) };
        assert!(!p.is_null(), "{n}");
        p
    };
    // SAFETY: ggml-cpu.h's signatures.
    let (traits, init): (unsafe extern "C" fn(c_int) -> *const Traits, unsafe extern "C" fn()) = unsafe {
        (
            std::mem::transmute::<*mut c_void, unsafe extern "C" fn(c_int) -> *const Traits>(sym("ggml_get_type_traits_cpu")),
            std::mem::transmute::<*mut c_void, unsafe extern "C" fn()>(sym("ggml_cpu_init")),
        )
    };
    unsafe { init() };
    let rows = real_rows();
    assert!(!rows.is_empty(), "NS_TEST_NH_BLOCKS");
    let ours = Cpu::get().unwrap();
    eprintln!("paths: {}", ours.paths());
    let mut r = Rng(4242);
    for (ty, name, nrows, cols, w) in rows {
        let x = activation(cols, &mut r, name.ends_with("down"));
        // SAFETY: the traits of a type ggml has; buffers of a row of the vec_dot type.
        let (vt, dot) = unsafe { ((*traits(ty)).vec_dot_type, (*traits(ty)).vec_dot.unwrap()) };
        let q = quantize(ty, &x);
        let (gq, same_q) = unsafe {
            let ff = (*traits(vt)).from_float.unwrap();
            let bytes = if vt == 15 { cols / 256 * 292 } else { cols / 32 * 34 };
            let mut g = vec![0u8; bytes];
            ff(x.as_ptr(), g.as_mut_ptr().cast(), cols as i64);
            (g.clone(), quant_bytes(&q) == g)
        };
        let rb = blocks::row_bytes(ty, cols);
        let (mut worst, mut mag_max, mut diff_q) = (0f64, 0f64, 0usize);
        if !same_q {
            diff_q = quant_bytes(&q).iter().zip(&gq).filter(|(a, b)| a != b).count();
        }
        for row in 0..nrows {
            let wr = &w[row * rb..(row + 1) * rb];
            let mut s = 0f32;
            // SAFETY: a row of the type, its activation in ggml's layout.
            unsafe { dot(cols as c_int, &mut s, 0, wr.as_ptr().cast(), 0, gq.as_ptr().cast(), 0, 1) };
            let v = ours.dot(ty, wr, &q);
            let mag: f64 = dequantize(ty, wr).iter().zip(act_values(&q)).map(|(a, b)| (*a as f64 * b).abs()).sum();
            worst = worst.max((v as f64 - s as f64).abs() / mag.max(1e-30));
            mag_max = mag_max.max(mag);
        }
        // one thread's rate through the rows (weights' bytes a second, best of 5 runs of 20 passes): ggml a row a
        // call (as the engine called it), ours as the engine calls it now, the rows of a part at once
        let rate = |f: &dyn Fn()| {
            let mut best = 0f64;
            for _ in 0..5 {
                let t = std::time::Instant::now();
                for _ in 0..20 {
                    f();
                }
                best = best.max((20 * nrows * rb) as f64 / t.elapsed().as_secs_f64() / 1e9);
            }
            best
        };
        let g_rate = rate(&|| {
            for row in 0..nrows {
                let mut s = 0f32;
                // SAFETY: as above
                unsafe { dot(cols as c_int, &mut s, 0, w[row * rb..].as_ptr().cast(), 0, gq.as_ptr().cast(), 0, 1) };
                std::hint::black_box(s);
            }
        });
        let mut y = vec![0f32; nrows];
        let rates: Vec<String> = [Isa::Scalar, Isa::Avx2, Isa::Avx512]
            .into_iter()
            .filter(|&i| i <= Isa::detect())
            .map(|isa| {
                let c = Cpu::new(isa, false);
                let yp = Out(y.as_mut_ptr());
                // SAFETY: y holds nrows, one closure at a time writes it; w holds nrows rows.
                format!("{} {:.1}", isa.name(), rate(&|| unsafe { c.rows(ty, w.as_ptr(), cols, &q, yp.at(0, nrows)) }))
            })
            .collect();
        eprintln!("{name} ({nrows} rows of {cols}): activation bytes {}, worst |ours - ggml| / sum|terms| {worst:.2e}; one thread, GB/s: ggml {g_rate:.1}, ours {}", if same_q { "equal".to_string() } else { format!("{diff_q} differ") }, rates.join(", "));
        assert!(worst < 2e-5, "{name}");
    }
}

/// a Quant in ggml's block layout (block_q8_K: float d, 256 int8, 16 int16; block_q8_0: half d, 32 int8)
fn quant_bytes(q: &Quant) -> Vec<u8> {
    let mut o = Vec::new();
    match q.act {
        Act::Q8K => {
            for b in 0..q.d.len() {
                o.extend(q.d[b].to_le_bytes());
                o.extend(q.qs[b * 256..b * 256 + 256].iter().map(|&v| v as u8));
                o.extend(q.bsums[b * 16..b * 16 + 16].iter().flat_map(|v| v.to_le_bytes()));
            }
        }
        Act::Q80 | Act::Q20 => {
            let qs = natural(q);
            for b in 0..q.d.len() {
                o.extend(blocks::f32_to_f16(q.d[b]).to_le_bytes());
                o.extend(qs[b * 32..b * 32 + 32].iter().map(|&v| v as u8));
            }
        }
    }
    o
}

/// A layer's CPU share timed (`cargo test -- --ignored layer`): 11 experts (half of a token's 22) of Nemotron's shapes,
/// up [2688][1024] and down [1024][2688] from the file's rows (NS_TEST_NH_BLOCKS, repeated to fill them), on the
/// engine's pool of the machine's threads - these products against ggml-cpu's (NS_TEST_NH_GGML_DIR, a row a call as
/// the engine called them), the same parts and order
#[test]
#[ignore]
fn layer_against_ggml_cpu() {
    use super::{experts, Job, Pool};
    use std::ffi::{c_char, c_int, c_void, CString};
    use std::sync::Arc;
    extern "C" {
        fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }
    type FromFloat = unsafe extern "C" fn(*const f32, *mut c_void, i64);
    type VecDot = unsafe extern "C" fn(c_int, *mut f32, usize, *const c_void, usize, *const c_void, usize, c_int);
    #[repr(C)]
    struct Traits {
        from_float: Option<FromFloat>,
        vec_dot: Option<VecDot>,
        vec_dot_type: c_int,
        nrows: i64,
    }
    let dir = std::env::var("NS_TEST_NH_GGML_DIR").expect("NS_TEST_NH_GGML_DIR");
    let rows = real_rows();
    assert!(!rows.is_empty(), "NS_TEST_NH_BLOCKS");
    // SAFETY: paths; the libraries stay loaded; ggml-cpu.h's signatures.
    let (traits, _base): (unsafe extern "C" fn(c_int) -> *const Traits, *mut c_void) = unsafe {
        let base = dlopen(CString::new(format!("{dir}/libggml-base.so")).unwrap().as_ptr(), 2 | 0x100);
        let cpu = dlopen(CString::new(format!("{dir}/libggml-cpu.so")).unwrap().as_ptr(), 2 | 0x100);
        assert!(!base.is_null() && !cpu.is_null());
        let init: unsafe extern "C" fn() = std::mem::transmute::<*mut c_void, unsafe extern "C" fn()>(dlsym(cpu, c"ggml_cpu_init".as_ptr()));
        init();
        (std::mem::transmute::<*mut c_void, unsafe extern "C" fn(c_int) -> *const Traits>(dlsym(cpu, c"ggml_get_type_traits_cpu".as_ptr())), base)
    };
    let fill = |ty: i32, want_rows: usize, cols: usize| -> Vec<u8> {
        let (_, _, _, c, w) = rows.iter().find(|r| r.0 == ty && r.3 == cols).expect("rows of the type");
        let rb = blocks::row_bytes(ty, *c);
        (0..want_rows).flat_map(|r| w[(r % (w.len() / rb)) * rb..][..rb].to_vec()).collect()
    };
    let (lat, ff, n) = (1024usize, 2688usize, 11usize);
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let pool = Pool::new(threads);
    let ours = Cpu::get().unwrap();
    eprintln!("{threads} threads; paths: {}", ours.paths());
    for (up_ty, down_ty) in [(IQ2_S, Q2_0), (IQ3_XXS, Q2_0), (IQ1_S, IQ4_NL), (IQ2_XXS, Q2_0)] {
        let ups: Vec<Vec<u8>> = (0..n).map(|_| fill(up_ty, ff, lat)).collect();
        let downs: Vec<Vec<u8>> = (0..n).map(|_| fill(down_ty, lat, ff)).collect();
        let jobs: Vec<Job> = (0..n).map(|e| Job { up: ups[e].as_ptr() as usize, down: downs[e].as_ptr() as usize, w: 0.1 }).collect();
        let mut r = Rng(99);
        let x: Vec<f32> = (0..lat).map(|_| r.f() * 2.0).collect();
        // the same layer through ggml: a row a call, the parts of 64 rows over the pool as the engine had them
        let ggml_layer = || -> Vec<f32> {
            const ROWS: usize = 64;
            // SAFETY: the traits of types ggml has; buffers of rows of the vec_dot types.
            let q = |ty: i32, v: &[f32]| unsafe {
                let vt = (*traits(ty)).vec_dot_type;
                let bytes = if vt == 15 { v.len() / 256 * 292 } else { v.len() / 32 * 34 };
                let mut b = vec![0u8; bytes + 64];
                (*traits(vt)).from_float.unwrap()(v.as_ptr(), b.as_mut_ptr().cast(), v.len() as i64);
                b
            };
            let run = |ty: i32, mats: Arc<Vec<usize>>, xq: Arc<Vec<Vec<u8>>>, cols: usize, nrow: usize| -> Vec<f32> {
                let mut out = vec![0f32; n * nrow];
                let o = Out(out.as_mut_ptr());
                let chunks = nrow.div_ceil(ROWS);
                // SAFETY: as the engine's old path: rows of the matrices, each part its own range of out.
                let dot = unsafe { (*traits(ty)).vec_dot.unwrap() };
                let rb = blocks::row_bytes(ty, cols);
                pool.run(n * chunks, Arc::new(move |i| {
                    let (e, c) = (i / chunks, i % chunks);
                    let rr = c * ROWS..((c + 1) * ROWS).min(nrow);
                    let y = unsafe { o.at(e * nrow + rr.start, rr.len()) };
                    for (k, row) in rr.enumerate() {
                        unsafe { dot(cols as c_int, y.as_mut_ptr().add(k), 0, (mats[e] as *const u8).add(row * rb).cast(), 0, xq[e.min(xq.len() - 1)].as_ptr().cast(), 0, 1) };
                    }
                }));
                out
            };
            let xq = Arc::new(vec![q(up_ty, &x)]);
            let u = run(up_ty, Arc::new(jobs.iter().map(|j| j.up).collect()), xq, lat, ff);
            let hq: Arc<Vec<Vec<u8>>> = Arc::new(u.chunks(ff).map(|v| q(down_ty, &v.iter().map(|a| a.max(0.0) * a.max(0.0)).collect::<Vec<f32>>())).collect());
            let d = run(down_ty, Arc::new(jobs.iter().map(|j| j.down).collect()), hq, ff, lat);
            let mut out = vec![0f32; lat];
            for row in d.chunks(lat) {
                for (o, v) in out.iter_mut().zip(row) {
                    *o += 0.1 * v;
                }
            }
            out
        };
        let time = |f: &dyn Fn() -> Vec<f32>| {
            let mut t = Vec::new();
            let mut last = Vec::new();
            for _ in 0..25 {
                let s = std::time::Instant::now();
                last = f();
                t.push(s.elapsed().as_secs_f64() * 1e3);
            }
            t.sort_by(f64::total_cmp);
            (t[t.len() / 2], last)
        };
        let (tg, yg) = time(&ggml_layer);
        let (to, yo) = time(&|| experts(ours, &pool, &jobs, up_ty, down_ty, &x, ff));
        let mag = yg.iter().fold(0f32, |m, v| m.max(v.abs()));
        let err = yg.iter().zip(&yo).fold(0f32, |m, (a, b)| m.max((a - b).abs())) / mag;
        eprintln!("{} up + {} down, {n} experts: ggml {tg:.2} ms, ours {to:.2} ms ({:.2}x); worst |ours - ggml| / max {err:.1e}", blocks::name(up_ty), blocks::name(down_ty), tg / to);
    }
}
