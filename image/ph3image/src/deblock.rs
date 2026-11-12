//! The deblock filter: the H3-family decoder's 16-pixel grid taken out of a decoded picture.
//!
//! The decoder is a ViT whose last linear writes each latent token's 16 x 16 pixels with nothing blending neighbours
//! (docs/image.md, "The grid texture"). What it leaves is mostly one **fixed pattern a tile**: the same 16 x 16
//! residual at every tile (brighter top rows, darker bottom ones, a fine dot texture), its strength following the
//! local content - faint on flat sky, several levels in fur and after an img2img. Measured on 2026-11-11 pictures, the
//! tiles' mean high-pass pattern in one half of a picture correlates 0.87 - 1.0 with the other half's.
//!
//! Two stages, each colour plane on its own, then the planes together:
//!
//! 1. **The pattern.** `hp` = the picture less its 17 x 17 box mean. Its weighted mean over every whole tile at each
//!    of the 256 positions, less its own mean, is the pattern `P`. A tile weighs `1 / (1 + sqrt(E / E_med))` (`E` its
//!    high-pass energy, `E_med` the median tile's): a long edge or a busy patch, which could line up with the tiles
//!    by chance, hardly counts. Each position is shrunk towards 0 by how far its mean stands above what averaging
//!    non-periodic content would leave (a Wiener factor `1 - var / (N_eff P^2)`, the variance counted
//!    `NOISE_FACTOR` times for neighbouring tiles' correlation): a grid-free picture's `P` comes out near 0, so it
//!    is left alone. Each tile's amplitude is its projection on `P`, the weighted mean over the 5 x 5 tiles around
//!    it, held to 0 .. `A_MAX`; the tile loses `strength x amplitude x P`. A subtraction of the one pattern, no
//!    low-pass: edges and fur keep their pixels but for the grid in them.
//! 2. **The boundaries.** At each tile boundary, on every line across it, H.264's edge test on the luma: the step
//!    across (`|q0 - p0|`) against `ALPHA` and the activity on each side (the largest of `|p0 - p1|`, `|p1 - p2|`,
//!    `|q0 - q1|`, `|q1 - q2|`) against `BETA`. Where both are small (flat areas), the step left beyond the two sides'
//!    own slopes is spread as a linear ramp over `N_RAMP` pixels each side, weighted `(1 - act / BETA)(1 - step /
//!    ALPHA)`: 0 across a real edge or texture. This takes out a tile's leftover offset (DC) and slope mismatch
//!    with its neighbour; on the decoder's pictures the first stage does nearly all the work.
//!
//! [`host`] is the reference (the unit tests); [`gpu`] runs the same on the GPU's kernels (`kernels/image/ph3image/
//! ph3.h`), the 256-value steps between them on the host.

use std::ffi::{c_int, c_void};
use std::sync::{Arc, OnceLock};

use nextcuda_core::DevBuf;
use nextcuda_image::{Error, Gpu, Result};

/// the decoder's tile side, pixels
pub const TILE: usize = 16;
/// the high-pass's box radius (17 x 17: a whole tile, so the pattern is not part of the mean taken off)
pub const BOX_R: usize = 8;
/// the variance of a position's mean over the tiles counted this many times (neighbouring tiles' content correlates)
pub const NOISE_FACTOR: f64 = 4.0;
/// a tile's energy floor in the weights (0..1 units squared; a picture of flat tiles weighs them all alike)
pub const E_FLOOR: f64 = (0.04 / 255.0) * (0.04 / 255.0);
/// the most a tile's amplitude is held to (the pattern's mean is 1)
pub const A_MAX: f32 = 3.0;
/// the amplitude averaged over the (2 A_R + 1)^2 tiles around each
pub const A_R: usize = 2;
/// the boundary test, on the luma in 0..1: a step at least this is an edge
pub const ALPHA: f32 = 8.0 / 255.0;
/// ... and the activity on either side at least this is texture
pub const BETA: f32 = 3.0 / 255.0;
/// the ramp's pixels each side of a boundary
pub const N_RAMP: usize = 4;
/// the luma's weights (BT.601, as the edge test reads it)
const LUMA: [f32; 3] = [0.299, 0.587, 0.114];

/// What a run found: the pattern's RMS a channel (0..1 units) and how many tiles there were
#[derive(Clone, Debug, Default)]
pub struct Report {
    pub pattern_rms: [f32; 3],
    pub tiles: usize,
}

/// Each tile's weight from its high-pass energy (`energy`, the sum of squares over its pixels; th x tw row-major):
/// `1 / (1 + sqrt(E / (E_med + E_FLOOR)))`, E the mean square
pub fn weights(energy: &[f32], tile_pixels: &[usize]) -> Vec<f32> {
    let e: Vec<f64> = energy.iter().zip(tile_pixels).map(|(s, n)| *s as f64 / (*n).max(1) as f64).collect();
    let mut sorted = e.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let med = sorted.get(sorted.len() / 2).copied().unwrap_or(0.0) + E_FLOOR;
    e.iter().map(|v| (1.0 / (1.0 + (v / med).sqrt())) as f32).collect()
}

/// The pattern from the weighted sums over the whole tiles at each position (`sums[2p]`, `sums[2p + 1]`: the sum of
/// w hp, of w hp^2; `sw` and `sw2` the sums of the weights and their squares): the means less their mean, each shrunk
/// by its Wiener factor
pub fn pattern(sums: &[f64], sw: f64, sw2: f64) -> Vec<f32> {
    if sw <= 0.0 {
        return vec![0.0; TILE * TILE];
    }
    let n_eff = sw * sw / sw2.max(1e-30);
    let mean: Vec<f64> = (0..TILE * TILE).map(|p| sums[2 * p] / sw).collect();
    let avg = mean.iter().sum::<f64>() / mean.len() as f64;
    (0..TILE * TILE)
        .map(|p| {
            let m = mean[p] - avg;
            let var = (sums[2 * p + 1] / sw - mean[p] * mean[p]).max(0.0);
            let noise = var / n_eff * NOISE_FACTOR;
            (m * (1.0 - noise / (m * m).max(1e-30)).clamp(0.0, 1.0)) as f32
        })
        .collect()
}

/// Each tile's amplitude from its dot with the pattern (`dots`, th x tw row-major): over the pattern's own energy, the
/// weighted mean over the tiles around it, held to 0 .. A_MAX; times `strength`. All 0 when the pattern is (near)
/// nothing.
pub fn amplitudes(dots: &[f32], wts: &[f32], th: usize, tw: usize, pat: &[f32], strength: f32) -> Vec<f32> {
    let e: f64 = pat.iter().map(|p| (*p as f64) * (*p as f64)).sum();
    if e < 1e-14 {
        return vec![0.0; th * tw];
    }
    let mut out = vec![0f32; th * tw];
    for i in 0..th {
        for j in 0..tw {
            let (mut s, mut k) = (0.0, 0.0);
            for y in i.saturating_sub(A_R)..(i + A_R + 1).min(th) {
                for x in j.saturating_sub(A_R)..(j + A_R + 1).min(tw) {
                    s += wts[y * tw + x] as f64 * dots[y * tw + x] as f64 / e;
                    k += wts[y * tw + x] as f64;
                }
            }
            out[i * tw + j] = strength * ((s / k.max(1e-30)) as f32).clamp(0.0, A_MAX);
        }
    }
    out
}

/// The pixels in each tile (th x tw row-major; a last partial row or column of tiles has fewer)
fn tile_pixels(w: usize, h: usize) -> Vec<usize> {
    let (th, tw) = tiles(w, h);
    (0..th * tw).map(|t| (h - (t / tw) * TILE).min(TILE) * (w - (t % tw) * TILE).min(TILE)).collect()
}

/// The weighted sums' totals over the whole tiles: (sum of w, sum of w^2)
fn weight_totals(wts: &[f32], w: usize, h: usize) -> (f64, f64) {
    let tw = w.div_ceil(TILE);
    let (mut s, mut s2) = (0.0, 0.0);
    for ty in 0..h / TILE {
        for tx in 0..w / TILE {
            let v = wts[ty * tw + tx] as f64;
            s += v;
            s2 += v * v;
        }
    }
    (s, s2)
}

fn tiles(w: usize, h: usize) -> (usize, usize) {
    (h.div_ceil(TILE), w.div_ceil(TILE))
}

/// The filter on the host: `px` RGB float32 [3, h, w] (0..1), in place. `strength` 0..1 scales both stages.
pub fn host(px: &mut [f32], w: usize, h: usize, strength: f32) -> Report {
    let plane = w * h;
    let mut rep = Report { tiles: (w / TILE) * (h / TILE), ..Report::default() };
    if w < TILE || h < TILE || strength <= 0.0 {
        return rep;
    }
    let (th, tw) = tiles(w, h);
    for c in 0..3 {
        let x = &mut px[c * plane..][..plane];
        let hp = highpass(x, w, h, BOX_R);
        let tile = |y: usize, xx: usize| (y / TILE) * tw + xx / TILE;
        let pos = |y: usize, xx: usize| (y % TILE) * TILE + xx % TILE;
        let mut energy = vec![0f32; th * tw];
        for y in 0..h {
            for xx in 0..w {
                energy[tile(y, xx)] += hp[y * w + xx] * hp[y * w + xx];
            }
        }
        let wts = weights(&energy, &tile_pixels(w, h));
        let mut sums = vec![0f64; 2 * TILE * TILE];
        for y in 0..(h / TILE) * TILE {
            for xx in 0..(w / TILE) * TILE {
                let (v, wt) = (hp[y * w + xx] as f64, wts[tile(y, xx)] as f64);
                sums[2 * pos(y, xx)] += wt * v;
                sums[2 * pos(y, xx) + 1] += wt * v * v;
            }
        }
        let (sw, sw2) = weight_totals(&wts, w, h);
        let pat = pattern(&sums, sw, sw2);
        rep.pattern_rms[c] = (pat.iter().map(|p| p * p).sum::<f32>() / pat.len() as f32).sqrt();
        let mut dots = vec![0f32; th * tw];
        for y in 0..h {
            for xx in 0..w {
                dots[tile(y, xx)] += hp[y * w + xx] * pat[pos(y, xx)];
            }
        }
        let a = amplitudes(&dots, &wts, th, tw, &pat, strength);
        for y in 0..h {
            for xx in 0..w {
                x[y * w + xx] -= a[tile(y, xx)] * pat[pos(y, xx)];
            }
        }
    }
    edges_host(px, w, h, true, strength);
    edges_host(px, w, h, false, strength);
    rep
}

/// x less its (2r + 1)^2 box mean, edges replicated
fn highpass(x: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    let n = (2 * r + 1) as f32;
    let at = |v: isize, len: usize| v.clamp(0, len as isize - 1) as usize;
    let mut tmp = vec![0f32; w * h];
    for y in 0..h {
        for xx in 0..w {
            let mut s = 0f32;
            for k in -(r as isize)..=r as isize {
                s += x[y * w + at(xx as isize + k, w)];
            }
            tmp[y * w + xx] = s / n;
        }
    }
    let mut hp = vec![0f32; w * h];
    for y in 0..h {
        for xx in 0..w {
            let mut s = 0f32;
            for k in -(r as isize)..=r as isize {
                s += tmp[at(y as isize + k, h) * w + xx];
            }
            hp[y * w + xx] = x[y * w + xx] - s / n;
        }
    }
    hp
}

/// The boundary pass (as `ns_image_ph3_edges`): vertical boundaries when `vertical`, else horizontal
fn edges_host(px: &mut [f32], w: usize, h: usize, vertical: bool, strength: f32) {
    let plane = w * h;
    let (len, lines) = if vertical { (w, h) } else { (h, w) };
    let n = N_RAMP;
    let mut b = TILE;
    while b + n <= len {
        for line in 0..lines {
            let (at, step) = if vertical { (line * w + b, 1isize) } else { (b * w + line, w as isize) };
            let o = |k: isize| (at as isize + k * step) as usize;
            let luma = |px: &[f32], k: isize| (0..3).map(|c| LUMA[c] * px[c * plane + o(k)]).sum::<f32>();
            let (p0, p1, p2) = (luma(px, -1), luma(px, -2), luma(px, -3));
            let (q0, q1, q2) = (luma(px, 0), luma(px, 1), luma(px, 2));
            let act = (p0 - p1).abs().max((p1 - p2).abs()).max((q0 - q1).abs().max((q1 - q2).abs()));
            let g = strength * (1.0 - act / BETA).max(0.0) * (1.0 - (q0 - p0).abs() / ALPHA).max(0.0);
            if g <= 0.0 {
                continue;
            }
            for c in 0..3 {
                let v = &mut px[c * plane..][..plane];
                let (pp0, pp1, qq0, qq1) = (v[o(-1)], v[o(-2)], v[o(0)], v[o(1)]);
                let e = g * ((qq0 - pp0) - ((pp0 - pp1) + (qq1 - qq0)) * 0.5);
                for k in 0..n {
                    let wgt = ((n - k) as f32 - 0.5) / (2 * n) as f32;
                    v[o(-(k as isize) - 1)] += e * wgt;
                    v[o(k as isize)] -= e * wgt;
                }
            }
        }
        b += TILE;
    }
}

type G = *mut c_void;

struct Api {
    highpass: unsafe extern "C" fn(G, *const f32, i64, i64, i64, *mut f32, *mut f32) -> c_int,
    tile_stats: unsafe extern "C" fn(G, *const f32, i64, i64, *const f32, *mut f32, *mut f32) -> c_int,
    tile_sums: unsafe extern "C" fn(G, *const f32, i64, i64, *const f32, *mut f32) -> c_int,
    subtract: unsafe extern "C" fn(G, *mut f32, i64, i64, *const f32, *const f32) -> c_int,
    edges: unsafe extern "C" fn(G, *mut f32, i64, i64, c_int, c_int, f32, f32, f32) -> c_int,
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
        // SAFETY: the symbol is declared in ph3.h with the field's signature.
        #[allow(clippy::missing_transmute_annotations)]
        unsafe {
            std::mem::transmute::<*mut c_void, _>(p)
        }
    }};
}

fn api() -> Result<&'static Api> {
    static API: OnceLock<std::result::Result<Api, String>> = OnceLock::new();
    API.get_or_init(|| {
        let lib = nextcuda_core::api().map_err(|e| e.0)?;
        Ok(Api {
            highpass: sym!(lib, "ns_image_ph3_highpass"),
            tile_stats: sym!(lib, "ns_image_ph3_tile_stats"),
            tile_sums: sym!(lib, "ns_image_ph3_tile_sums"),
            subtract: sym!(lib, "ns_image_ph3_subtract"),
            edges: sym!(lib, "ns_image_ph3_edges"),
        })
    })
    .as_ref()
    .map_err(|e| Error(e.clone()))
}

fn check(rc: c_int, what: &str) -> Result<()> {
    if rc == 0 {
        return Ok(());
    }
    let why = nextcuda_core::api().map(|a| a.error()).unwrap_or_default();
    Err(Error(format!("deblock {what}: {why}")))
}

/// The filter on `gpu`'s kernels: `px` RGB float32 [3, h, w] (0..1), in place - uploaded, filtered, read back.
/// The same arithmetic as [`host`] but for the order of the GPU's sums.
pub fn gpu(gpu: &Arc<Gpu>, px: &mut [f32], w: usize, h: usize, strength: f32) -> Result<Report> {
    let plane = w * h;
    let mut rep = Report { tiles: (w / TILE) * (h / TILE), ..Report::default() };
    if w < TILE || h < TILE || strength <= 0.0 {
        return Ok(rep);
    }
    if px.len() != 3 * plane {
        return Err(Error(format!("deblock: {} values for a {w}x{h} RGB picture", px.len())));
    }
    let r = api()?;
    let g = gpu.raw();
    let (th, tw) = tiles(w, h);
    let (wi, hi) = (w as i64, h as i64);
    let xs = DevBuf::from_f32(gpu, px)?;
    let tmp = DevBuf::f32(gpu, plane)?;
    let hp = DevBuf::f32(gpu, plane)?;
    let sums = DevBuf::f32(gpu, 2 * TILE * TILE)?;
    let energy = DevBuf::f32(gpu, th * tw)?;
    let dots = DevBuf::f32(gpu, th * tw)?;
    let pixels = tile_pixels(w, h);
    for c in 0..3 {
        // SAFETY: plane c of xs, which holds 3 planes
        let x = unsafe { xs.fp().add(c * plane) };
        // SAFETY: every buffer is sized as the call reads and writes it (ph3.h); all on gpu's queue.
        check(unsafe { (r.highpass)(g, x, hi, wi, BOX_R as i64, tmp.fp(), hp.fp()) }, "high-pass")?;
        // SAFETY: the buffers are sized as ph3.h reads and writes them; the work queued on gpu's queue.
        check(unsafe { (r.tile_stats)(g, hp.fp(), hi, wi, std::ptr::null(), energy.fp(), dots.fp()) }, "tile energies")?;
        let wts = weights(&energy.to_f32()?, &pixels);
        let wts_d = DevBuf::from_f32(gpu, &wts)?;
        // SAFETY: the buffers are sized as ph3.h reads and writes them; the work queued on gpu's queue.
        check(unsafe { (r.tile_sums)(g, hp.fp(), hi, wi, wts_d.fp(), sums.fp()) }, "tile sums")?;
        let s: Vec<f64> = sums.to_f32()?.into_iter().map(f64::from).collect();
        let (sw, sw2) = weight_totals(&wts, w, h);
        let pat = pattern(&s, sw, sw2);
        rep.pattern_rms[c] = (pat.iter().map(|p| p * p).sum::<f32>() / pat.len() as f32).sqrt();
        let pat_d = DevBuf::from_f32(gpu, &pat)?;
        // SAFETY: the buffers are sized as ph3.h reads and writes them; the work queued on gpu's queue.
        check(unsafe { (r.tile_stats)(g, hp.fp(), hi, wi, pat_d.fp(), energy.fp(), dots.fp()) }, "tile dots")?;
        let a = amplitudes(&dots.to_f32()?, &wts, th, tw, &pat, strength);
        let a_d = DevBuf::from_f32(gpu, &a)?;
        // SAFETY: the buffers are sized as ph3.h reads and writes them; the work queued on gpu's queue.
        check(unsafe { (r.subtract)(g, x, hi, wi, a_d.fp(), pat_d.fp()) }, "subtract")?;
        // the weights, pattern and amplitudes stay alive until the subtraction has run
        gpu.sync()?;
    }
    for vertical in [1, 0] {
        // SAFETY: the buffers are sized as ph3.h reads and writes them; the work queued on gpu's queue.
        check(unsafe { (r.edges)(g, xs.fp(), hi, wi, vertical, N_RAMP as c_int, ALPHA, BETA, strength) }, "boundaries")?;
    }
    let out = xs.to_f32()?;
    px.copy_from_slice(&out);
    Ok(rep)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A smooth picture: a diagonal colour ramp with a soft bump, no 16-pixel structure of its own
    fn smooth(w: usize, h: usize) -> Vec<f32> {
        let mut px = vec![0f32; 3 * w * h];
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    let (u, v) = (x as f32 / w as f32, y as f32 / h as f32);
                    let bump = (-((u - 0.4).powi(2) + (v - 0.6).powi(2)) * 8.0).exp();
                    px[c * w * h + y * w + x] = 0.2 + 0.3 * u + 0.15 * v * (c as f32 + 1.0) / 3.0 + 0.2 * bump;
                }
            }
        }
        px
    }

    /// The energy at the 16-pixel grid: the RMS of the picture's tile-mean residual (each tile position's mean over
    /// all tiles of the high-pass), the luma
    fn grid_rms(px: &[f32], w: usize, h: usize) -> f32 {
        let plane = w * h;
        let y: Vec<f32> = (0..plane).map(|i| (0..3).map(|c| LUMA[c] * px[c * plane + i]).sum()).collect();
        let hp = highpass(&y, w, h, BOX_R);
        let mut m = vec![0f64; TILE * TILE];
        let n = (w / TILE) * (h / TILE);
        for yy in 0..h {
            for x in 0..w {
                m[(yy % TILE) * TILE + x % TILE] += hp[yy * w + x] as f64 / n as f64;
            }
        }
        let avg = m.iter().sum::<f64>() / m.len() as f64;
        (m.iter().map(|v| (v - avg) * (v - avg)).sum::<f64>() / m.len() as f64).sqrt() as f32
    }

    /// The picture `smooth` with `add(c, tile row, tile column, row in tile, column in tile)` (0..1 units) on it
    fn with(w: usize, h: usize, add: impl Fn(usize, usize, usize, usize, usize) -> f32) -> Vec<f32> {
        let mut px = smooth(w, h);
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    px[c * w * h + y * w + x] += add(c, y / TILE, x / TILE, y % TILE, x % TILE);
                }
            }
        }
        px
    }

    fn rms_off(p: &[f32], q: &[f32]) -> f32 {
        (p.iter().zip(q).map(|(a, b)| (a - b) * (a - b)).sum::<f32>() / p.len() as f32).sqrt() * 255.0
    }

    /// The mean step across the tile boundaries less the mean step inside the tiles (levels, luma, both directions)
    fn boundary_excess(px: &[f32], w: usize, h: usize) -> f32 {
        let plane = w * h;
        let y = |i: usize| (0..3).map(|c| LUMA[c] * px[c * plane + i]).sum::<f32>() * 255.0;
        let (mut on, mut non, mut off, mut noff) = (0.0, 0, 0.0, 0);
        for yy in 0..h {
            for x in 1..w {
                let d = (y(yy * w + x) - y(yy * w + x - 1)).abs();
                if x % TILE == 0 { on += d; non += 1 } else { off += d; noff += 1 }
            }
        }
        for yy in 1..h {
            for x in 0..w {
                let d = (y(yy * w + x) - y((yy - 1) * w + x)).abs();
                if yy % TILE == 0 { on += d; non += 1 } else { off += d; noff += 1 }
            }
        }
        on / non as f32 - off / noff as f32
    }

    #[test]
    fn a_tile_pattern_is_taken_out() {
        let (w, h) = (256, 192);
        let clean = smooth(w, h);
        // the decoder's kind of residual: a ramp down each tile (brighter top rows) and a fine dot pattern, the same
        // in every tile, its mean 0 (the filter leaves a picture's mean as it is)
        let mut px = with(w, h, |_, _, _, i, j| (7.5 - i as f32) / 7.5 * 2.0 / 255.0 + if (i + j) % 4 == 0 { 1.5 / 255.0 } else { -0.5 / 255.0 });
        let (g0, e0) = (grid_rms(&px, w, h), rms_off(&px, &clean));
        let rep = host(&mut px, w, h, 1.0);
        let (g1, e1) = (grid_rms(&px, w, h), rms_off(&px, &clean));
        assert!(rep.pattern_rms.iter().all(|r| *r > 0.5 / 255.0), "the pattern found: {:?}", rep.pattern_rms);
        assert!(g1 < g0 * 0.05, "grid {:.3} -> {:.3} levels", g0 * 255.0, g1 * 255.0);
        assert!(e1 < e0 * 0.2, "off the clean picture by {e0:.3} -> {e1:.3} levels");
    }

    #[test]
    fn tile_offsets_lose_their_steps() {
        let (w, h) = (256, 192);
        // a small offset a tile, +-1 level by a fixed pseudo-random sign: no pattern common to the tiles, steps at
        // their boundaries
        let mut px = with(w, h, |_, ty, tx, _, _| match (ty * 31 + tx * 17) % 3 {
            0 => 1.0 / 255.0,
            1 => -1.0 / 255.0,
            _ => 0.0,
        });
        let b0 = boundary_excess(&px, w, h);
        host(&mut px, w, h, 1.0);
        let b1 = boundary_excess(&px, w, h);
        assert!(b0 > 0.5 && b1 < b0 * 0.25, "the boundaries' extra step {b0:.3} -> {b1:.3} levels");
    }

    #[test]
    fn a_sharp_edge_on_a_boundary_is_kept() {
        let (w, h) = (128, 128);
        // a hard step at x = 64 (a tile boundary) on a flat field: dark left, light right; a fine texture below it
        let mut px = vec![0f32; 3 * w * h];
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    let v = if x < 64 { 0.2 } else { 0.7 };
                    let tex = if y >= 64 { 0.04 * (((x * 7 + y * 13) % 5) as f32 - 2.0) } else { 0.0 };
                    px[c * w * h + y * w + x] = v + tex;
                }
            }
        }
        let before = px.clone();
        host(&mut px, w, h, 1.0);
        let worst = px.iter().zip(&before).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(worst < 0.5 / 255.0, "moved by {:.3} levels", worst * 255.0);
        // the step across the boundary itself is untouched
        for y in [10, 70, 120] {
            let (p0, q0) = (px[y * w + 63], px[y * w + 64]);
            assert!((q0 - p0 - (before[y * w + 64] - before[y * w + 63])).abs() < 0.5 / 255.0);
        }
    }

    #[test]
    fn a_grid_free_picture_is_left_nearly_alone() {
        let (w, h) = (256, 256);
        let mut px = smooth(w, h);
        // a texture with no 16-pixel period (periods 7 and 11)
        for (i, v) in px.iter_mut().enumerate() {
            let (x, y) = (i % w, (i / w) % h);
            *v += 0.02 * ((x as f32 * 0.9).sin() + (y as f32 * 0.57).cos());
        }
        let before = px.clone();
        host(&mut px, w, h, 1.0);
        let rms = (px.iter().zip(&before).map(|(a, b)| (a - b) * (a - b)).sum::<f32>() / px.len() as f32).sqrt();
        assert!(rms * 255.0 < 0.15, "changed by {:.3} levels RMS", rms * 255.0);
    }

    #[test]
    fn strength_zero_is_no_change() {
        let (w, h) = (64, 48);
        let mut px = smooth(w, h);
        let before = px.clone();
        host(&mut px, w, h, 0.0);
        assert_eq!(px, before);
    }
}
