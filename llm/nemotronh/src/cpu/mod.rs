//! The CPU's share of a decode layer's experts (after nextsycl's deepseek2 `cpu.rs`, and Strata's CPU tier): experts in
//! pinned host memory are multiplied where they are, by the host's cores, instead of being copied over PCIe - the
//! link is decode's bottleneck (about 1 GB of experts a token at ~24 GB/s), the host's memory is not, and the two run
//! at once. The products are this crate's own (blocks.rs: the formats and the plain products; x86.rs: AVX2 and
//! AVX-512), for the types Nemotron's experts are stored in; a layer of another type leaves every expert to the GPU.
//! Where AVX-512 is there both vector forms of each type's product are timed once and the faster kept: on Zen 4,
//! whose 512-bit units are two 256-bit halves, the AVX2 forms were the faster for every type (measured 2026-11-13).

mod blocks;
mod grids;
#[cfg(target_arch = "x86_64")]
mod x86;

#[cfg(test)]
mod tests;

pub use blocks::{format, row_bytes, Quant};
use std::sync::{Arc, Condvar, Mutex};

/// Which of the products' paths this CPU runs
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Isa {
    Scalar,
    Avx2,
    Avx512,
}

impl Isa {
    /// The widest this CPU has, no wider than `NS_NH_CPU_ISA` (scalar, avx2, avx512) asks
    pub fn detect() -> Isa {
        #[cfg(target_arch = "x86_64")]
        let have = if x86::has_avx512() {
            Isa::Avx512
        } else if x86::has_avx2() {
            Isa::Avx2
        } else {
            Isa::Scalar
        };
        #[cfg(not(target_arch = "x86_64"))]
        let have = Isa::Scalar;
        have.min(Isa::cap_of(std::env::var("NS_NH_CPU_ISA").ok().as_deref()))
    }

    /// The widest path NS_NH_CPU_ISA allows: "scalar", "avx2", else (unset, "avx512", anything else) no cap
    fn cap_of(v: Option<&str>) -> Isa {
        match v {
            Some("scalar") => Isa::Scalar,
            Some("avx2") => Isa::Avx2,
            _ => Isa::Avx512,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Isa::Scalar => "scalar",
            Isa::Avx2 => "AVX2",
            Isa::Avx512 => "AVX-512",
        }
    }
}

/// The CPU's products, on the widest vector path this CPU has
pub struct Cpu {
    pub isa: Isa,
    /// each type's vector product and the path it is (empty: the plain products)
    #[cfg(target_arch = "x86_64")]
    kernels: Vec<(i32, x86::Kernel, Isa)>,
}

/// The types the CPU multiplies
pub const TYPES: [i32; 7] = [blocks::IQ2_XXS, blocks::IQ2_S, blocks::IQ3_XXS, blocks::IQ1_S, blocks::IQ4_NL, blocks::Q2_0, blocks::Q8_0];

impl Cpu {
    /// The products, chosen once: None with NS_NH_CPU=0
    pub fn get() -> Option<&'static Cpu> {
        static G: std::sync::OnceLock<Option<Cpu>> = std::sync::OnceLock::new();
        G.get_or_init(|| if std::env::var("NS_NH_CPU").is_ok_and(|v| v == "0") { None } else { Some(Cpu::new(Isa::detect(), true)) }).as_ref()
    }

    /// The products on `isa`; with `timed`, AVX-512's types each on the faster of their two vector forms
    pub fn new(isa: Isa, timed: bool) -> Cpu {
        #[cfg(target_arch = "x86_64")]
        {
            let kernels = TYPES
                .iter()
                .filter_map(|&ty| {
                    let (k2, k5) = (x86::kernel_avx2(ty)?, x86::kernel_avx512(ty)?);
                    Some(match isa {
                        Isa::Scalar => return None,
                        Isa::Avx2 => (ty, k2, Isa::Avx2),
                        Isa::Avx512 if !timed => (ty, k5, Isa::Avx512),
                        Isa::Avx512 => {
                            if time(ty, k5) < time(ty, k2) {
                                (ty, k5, Isa::Avx512)
                            } else {
                                (ty, k2, Isa::Avx2)
                            }
                        }
                    })
                })
                .collect();
            Cpu { isa, kernels }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = timed;
            Cpu { isa }
        }
    }

    /// Each type's path, for the log: "IQ2_S AVX2, ..."
    pub fn paths(&self) -> String {
        #[cfg(target_arch = "x86_64")]
        if !self.kernels.is_empty() {
            return self.kernels.iter().map(|(ty, _, isa)| format!("{} {}", blocks::name(*ty), isa.name())).collect::<Vec<_>>().join(", ");
        }
        "plain".into()
    }

    /// Whether the CPU multiplies `ty` (a GGUF type code)
    pub fn supports(&self, ty: i32) -> bool {
        format(ty).is_some()
    }

    /// x quantized as `ty`'s products take it
    pub fn quantize(&self, ty: i32, x: &[f32]) -> Quant {
        blocks::quantize(ty, x)
    }

    /// y[i] = row i of the matrix at w (rows of `cols`, type `ty`) . xq, for as many rows as y holds
    ///
    /// # Safety
    /// w points at y.len() rows of `ty`'s blocks, each `row_bytes(ty, cols)` long.
    pub unsafe fn rows(&self, ty: i32, w: *const u8, cols: usize, xq: &Quant, y: &mut [f32]) {
        let (qk, bytes, act) = format(ty).expect("a type the CPU multiplies");
        let (nb, rb) = (cols / qk, cols / qk * bytes);
        assert!(cols.is_multiple_of(qk) && xq.act == act && xq.qs.len() == cols, "type {ty}: {cols} values against an activation of {}", xq.qs.len());
        #[cfg(target_arch = "x86_64")]
        {
            if let Some(&(_, k, _)) = self.kernels.iter().find(|k| k.0 == ty) {
                for (i, o) in y.iter_mut().enumerate() {
                    // SAFETY: detect() chose this path only where the CPU has it; the caller's matrix holds the rows,
                    // each nb blocks, and xq as many blocks' values (checked above).
                    *o = unsafe { k(w.add(i * rb), nb, xq) };
                }
                return;
            }
        }
        let k = blocks::kernel(ty);
        for (i, o) in y.iter_mut().enumerate() {
            // SAFETY: the caller's matrix holds the rows.
            *o = k(unsafe { std::slice::from_raw_parts(w.add(i * rb), rb) }, xq);
        }
    }

    /// A row of `ty` (`w` exactly its blocks) . xq
    pub fn dot(&self, ty: i32, w: &[u8], xq: &Quant) -> f32 {
        let (qk, bytes, _) = format(ty).expect("a type the CPU multiplies");
        assert_eq!(w.len() % bytes, 0);
        let mut y = [0f32];
        // SAFETY: w is one row's blocks.
        unsafe { self.rows(ty, w.as_ptr(), w.len() / bytes * qk, xq, &mut y) };
        y[0]
    }
}

/// Seconds a vector product takes over 64 rows of made-up blocks of `ty` (the best of 5 runs of 8 passes)
#[cfg(target_arch = "x86_64")]
fn time(ty: i32, k: x86::Kernel) -> f64 {
    let (qk, bytes, _) = format(ty).unwrap();
    let cols = if qk == 256 { 1024 } else { 2688 };
    let (nb, rows) = (cols / qk, 64);
    let mut r = 0x2545_f491_4f6c_dd1du64;
    let mut next = || {
        r ^= r << 13;
        r ^= r >> 7;
        r ^= r << 17;
        r
    };
    let mut w: Vec<u8> = (0..rows * nb * bytes).map(|_| next() as u8).collect();
    for b in w.chunks_exact_mut(bytes) {
        // a scale of about 0.01 (half 0x2000): the blocks' values stay finite
        b[0] = 0;
        b[1] = 0x20;
    }
    let x: Vec<f32> = (0..cols).map(|_| (next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5).collect();
    let q = blocks::quantize(ty, &x);
    let mut best = f64::MAX;
    for _ in 0..5 {
        let t = std::time::Instant::now();
        for _ in 0..8 {
            for i in 0..rows {
                // SAFETY: the caller chose k for a CPU that has its features; w holds the rows, q their values.
                std::hint::black_box(unsafe { k(w.as_ptr().add(i * nb * bytes), nb, &q) });
            }
        }
        best = best.min(t.elapsed().as_secs_f64());
    }
    best
}

type JobFn = Arc<dyn Fn(usize) + Send + Sync>;

/// Workers that take the parts of a job (nextsycl's pool): the caller works too, and `run` returns when all are done
pub struct Pool {
    shared: Arc<Shared>,
    pub threads: usize,
}

struct Shared {
    /// (the job, its generation, its parts, stop)
    m: Mutex<(Option<JobFn>, u64, usize, bool)>,
    work: Condvar,
    /// generation << 32 | the next part
    ctr: std::sync::atomic::AtomicU64,
    left: std::sync::atomic::AtomicUsize,
}

impl Pool {
    pub fn new(threads: usize) -> Pool {
        let shared = Arc::new(Shared { m: Mutex::new((None, 0, 0, false)), work: Condvar::new(), ctr: 0.into(), left: 0.into() });
        for _ in 0..threads.saturating_sub(1) {
            let sh = shared.clone();
            std::thread::spawn(move || {
                let mut seen = 0u64;
                loop {
                    let (job, gen, parts) = {
                        let mut g = sh.m.lock().unwrap();
                        loop {
                            if g.3 {
                                return;
                            }
                            if g.1 != seen {
                                seen = g.1;
                                if let Some(j) = g.0.clone() {
                                    break (j, g.1, g.2);
                                }
                            }
                            g = sh.work.wait(g).unwrap();
                        }
                    };
                    Self::work(&sh, &job, gen, parts);
                }
            });
        }
        Pool { shared, threads }
    }

    fn work(sh: &Shared, job: &JobFn, gen: u64, parts: usize) {
        use std::sync::atomic::Ordering::*;
        loop {
            let cur = sh.ctr.load(Acquire);
            if cur >> 32 != gen & 0xffff_ffff {
                return;
            }
            let i = (cur & 0xffff_ffff) as usize;
            if i >= parts {
                return;
            }
            if sh.ctr.compare_exchange_weak(cur, cur + 1, AcqRel, Acquire).is_err() {
                continue;
            }
            // a part that panics still counts as done (else `run` would wait forever); the panic is reported
            if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| job(i))).is_err() {
                eprintln!("[nemotronh cpu: a part of the experts' products panicked]");
            }
            sh.left.fetch_sub(1, AcqRel);
        }
    }

    /// f(0..parts) over the workers and this thread; returns when every part is done
    pub fn run(&self, parts: usize, f: JobFn) {
        use std::sync::atomic::Ordering::*;
        if parts == 0 {
            return;
        }
        let sh = &*self.shared;
        let gen = {
            let mut g = sh.m.lock().unwrap();
            g.1 += 1;
            g.0 = Some(f.clone());
            g.2 = parts;
            sh.left.store(parts, Release);
            sh.ctr.store((g.1 & 0xffff_ffff) << 32, Release);
            g.1
        };
        sh.work.notify_all();
        Self::work(sh, &f, gen, parts);
        while sh.left.load(Acquire) > 0 {
            std::hint::spin_loop();
        }
        sh.m.lock().unwrap().0 = None;
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.m.lock().unwrap().3 = true;
        self.shared.work.notify_all();
    }
}

/// A raw pointer the pool's parts write their own disjoint ranges through
#[derive(Clone, Copy)]
struct Out(*mut f32);
// SAFETY: every part writes a range no other part writes; the buffer outlives `run` (it waits for all parts).
unsafe impl Send for Out {}
unsafe impl Sync for Out {}
impl Out {
    /// SAFETY: the caller's range is inside the buffer and written by it alone
    #[allow(clippy::mut_from_ref)]
    unsafe fn at(&self, i: usize, n: usize) -> &mut [f32] {
        std::slice::from_raw_parts_mut(self.0.add(i), n)
    }
}

/// The quantized rows the pool's parts write, one each
#[derive(Clone, Copy)]
struct QuantOut(*mut Option<Quant>);
// SAFETY: every part writes its own entry; the vector outlives `run`.
unsafe impl Send for QuantOut {}
unsafe impl Sync for QuantOut {}
impl QuantOut {
    /// SAFETY: entry e is inside the vector and written by the caller alone
    unsafe fn set(&self, e: usize, q: Quant) {
        *self.0.add(e) = Some(q);
    }
}

/// One expert the CPU multiplies: its up [n_ff, latent] and down [latent, n_ff] matrices' bytes in pinned memory, and its weight
pub struct Job {
    pub up: usize,
    pub down: usize,
    pub w: f32,
}

/// sum over the jobs of w * down . relu^2(up . x): the latent's [latent] for the GPU to add
pub fn experts(g: &'static Cpu, pool: &Pool, jobs: &[Job], up_ty: i32, down_ty: i32, x: &[f32], n_ff: usize) -> Vec<f32> {
    const ROWS: usize = 64;
    let (n, lat) = (jobs.len(), x.len());
    if n == 0 {
        return vec![0.0; lat];
    }
    let addrs: Arc<Vec<(usize, usize)>> = Arc::new(jobs.iter().map(|j| (j.up, j.down)).collect());
    let xq = Arc::new(g.quantize(up_ty, x));
    let mut u = vec![0f32; n * n_ff];
    let chunks = n_ff.div_ceil(ROWS);
    {
        let (addrs, out, xq) = (addrs.clone(), Out(u.as_mut_ptr()), xq.clone());
        pool.run(n * chunks, Arc::new(move |i| {
            let (e, c) = (i / chunks, i % chunks);
            let r = c * ROWS..((c + 1) * ROWS).min(n_ff);
            // SAFETY: rows [e][r] of u are this part's alone
            let y = unsafe { out.at(e * n_ff + r.start, r.len()) };
            // SAFETY: the expert's up matrix, n_ff rows of lat, in pinned memory for the call
            unsafe { g.rows(up_ty, (addrs[e].0 as *const u8).add(r.start * row_bytes(up_ty, lat)), lat, &xq, y) };
        }));
    }
    // relu^2 of each expert's up product, quantized for its down matrix: an expert a part
    let u = Arc::new(u);
    let mut uq: Vec<Option<Quant>> = (0..n).map(|_| None).collect();
    {
        let (u, out) = (u.clone(), QuantOut(uq.as_mut_ptr()));
        pool.run(n, Arc::new(move |e| {
            let h: Vec<f32> = u[e * n_ff..(e + 1) * n_ff].iter().map(|a| {
                let r = a.max(0.0);
                r * r
            }).collect();
            // SAFETY: entry e is this part's alone; uq outlives the run
            unsafe { out.set(e, g.quantize(down_ty, &h)) };
        }));
    }
    let uq: Arc<Vec<Quant>> = Arc::new(uq.into_iter().map(|q| q.expect("every expert quantized")).collect());
    let mut d = vec![0f32; n * lat];
    let dchunks = lat.div_ceil(ROWS);
    {
        let (addrs, out, uq) = (addrs.clone(), Out(d.as_mut_ptr()), uq.clone());
        pool.run(n * dchunks, Arc::new(move |i| {
            let (e, c) = (i / dchunks, i % dchunks);
            let r = c * ROWS..((c + 1) * ROWS).min(lat);
            // SAFETY: as above
            let y = unsafe { out.at(e * lat + r.start, r.len()) };
            // SAFETY: the expert's down matrix, lat rows of n_ff
            unsafe { g.rows(down_ty, (addrs[e].1 as *const u8).add(r.start * row_bytes(down_ty, n_ff)), n_ff, &uq[e], y) };
        }));
    }
    let mut out = vec![0f32; lat];
    for (j, row) in jobs.iter().zip(d.chunks(lat)) {
        for (o, v) in out.iter_mut().zip(row) {
            *o += j.w * v;
        }
    }
    out
}
