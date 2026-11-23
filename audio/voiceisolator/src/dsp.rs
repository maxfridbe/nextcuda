//! The host side of the isolator: the bands, the STFT and its inverse (torch.stft / torch.istft as the reference calls
//! them: n_fft 2048, hop 480, a periodic Hann window, centred with reflection), the heads' correction of the spectrum,
//! the 8 s windows and their crossfade, and torchaudio's kaiser_best resampler to and from 48 kHz.
//!
//! Origin: Inworld's voice isolator (github.com/inworld-ai/voice-isolator: model.py band_edges, stft, istft,
//! correction, bounded_correction, enhance; audio.py resample; Apache-2.0) and torchaudio's sinc resampler
//! (functional._get_sinc_resample_kernel, BSD-2), written anew. The FFT is a radix-2 one in float64.

use std::f64::consts::PI;

/// Bin ranges [start, end) of the bands: (width Hz, up to Hz) pairs of the plan, the rest one band
pub fn band_edges(n_fft: usize, rate: u32, plan: &[(f64, f64)]) -> Vec<(usize, usize)> {
    let bins = n_fft / 2 + 1;
    let hz = rate as f64 / n_fft as f64;
    let mut edges = Vec::new();
    let mut start = 0;
    for &(w, upto) in plan {
        let width = ((w / hz).round_ties_even() as usize).max(1);
        let upto = bins.min((upto / hz).round_ties_even() as usize);
        while start < upto {
            let end = (start + width).min(upto);
            edges.push((start, end));
            start = end;
        }
    }
    if start < bins {
        edges.push((start, bins));
    }
    edges
}

/// The FFT's twiddles for size n: e^{-2 pi i k / n}, k < n / 2
pub fn twiddles(n: usize) -> Vec<(f64, f64)> {
    (0..n / 2).map(|k| { let (s, c) = (-2.0 * PI * k as f64 / n as f64).sin_cos(); (c, s) }).collect()
}

/// In-place radix-2 FFT (n a power of two, `tw` its twiddles); `inverse`: e^{+i}, unscaled
pub fn fft(re: &mut [f64], im: &mut [f64], tw: &[(f64, f64)], inverse: bool) {
    let n = re.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let sign = if inverse { -1.0 } else { 1.0 };
    let mut len = 2;
    while len <= n {
        let step = n / len;
        for i in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let (c, s) = tw[k * step];
                let s = s * sign;
                let (a, b) = (i + k, i + k + len / 2);
                let tr = re[b] * c - im[b] * s;
                let ti = re[b] * s + im[b] * c;
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
            }
        }
        len <<= 1;
    }
}

/// `f(t, out)` for each of `frames` rows of `width` values, on a few threads
fn par_rows<T: Send + Copy + Default>(frames: usize, width: usize, f: impl Fn(usize, &mut [T]) + Sync) -> Vec<T> {
    let mut v = vec![T::default(); frames * width];
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).clamp(1, 8);
    let per = frames.div_ceil(threads).max(1);
    std::thread::scope(|sc| {
        for (ci, part) in v.chunks_mut(per * width).enumerate() {
            let f = &f;
            sc.spawn(move || {
                for (j, row) in part.chunks_mut(width).enumerate() {
                    f(ci * per + j, row);
                }
            });
        }
    });
    v
}

/// torch.hann_window(n) (periodic), in float32
pub fn hann(n: usize) -> Vec<f32> {
    (0..n).map(|i| (0.5 - 0.5 * (2.0 * PI * i as f64 / n as f64).cos()) as f32).collect()
}

/// The STFT and its inverse at one size
pub struct Stft {
    pub n_fft: usize,
    pub hop: usize,
    window: Vec<f32>,
    tw: Vec<(f64, f64)>,
}

/// A spectrum: [frames, bins] of (re, im), float32 as torch keeps it
pub struct Spectrum {
    pub frames: usize,
    pub bins: usize,
    pub v: Vec<[f32; 2]>,
}

impl Stft {
    pub fn new(n_fft: usize, hop: usize) -> Stft {
        Stft { n_fft, hop, window: hann(n_fft), tw: twiddles(n_fft) }
    }

    /// frames of a clip of n samples (centred)
    pub fn frames(&self, n: usize) -> usize {
        1 + n / self.hop
    }

    /// torch.stft(center=True, pad_mode reflect when the clip is longer than n_fft / 2, else zeros), onesided
    pub fn forward(&self, x: &[f32]) -> Spectrum {
        let (nf, half, n) = (self.n_fft, self.n_fft / 2, x.len());
        let reflect = n > half;
        let at = |i: isize| -> f32 {
            let j = i - half as isize;
            if j >= 0 && (j as usize) < n {
                return x[j as usize];
            }
            if !reflect {
                return 0.0;
            }
            let j = if j < 0 { -j } else { 2 * (n as isize - 1) - j };
            x[j.clamp(0, n as isize - 1) as usize]
        };
        let frames = self.frames(n);
        let bins = half + 1;
        let v = par_rows(frames, bins, |t, row: &mut [[f32; 2]]| {
            let (mut re, mut im) = (vec![0f64; nf], vec![0f64; nf]);
            for (i, (r, w)) in re.iter_mut().zip(&self.window).enumerate() {
                *r = at((t * self.hop + i) as isize) as f64 * *w as f64;
            }
            fft(&mut re, &mut im, &self.tw, false);
            for (k, r) in row.iter_mut().enumerate() {
                *r = [re[k] as f32, im[k] as f32];
            }
        });
        Spectrum { frames, bins, v }
    }

    /// torch.istft(center=True, length): each frame's inverse (the DC and Nyquist bins' imaginary parts dropped, as
    /// irfft does) windowed and overlap-added, over the window's squared envelope
    pub fn inverse(&self, s: &Spectrum, length: usize) -> Vec<f32> {
        let (nf, half) = (self.n_fft, self.n_fft / 2);
        let total = nf + self.hop * (s.frames - 1);
        // each frame's windowed inverse on the threads, then the overlap-add in order
        let frames = par_rows(s.frames, nf, |t, out: &mut [f64]| {
            let (mut re, mut im) = (vec![0f64; nf], vec![0f64; nf]);
            let row = &s.v[t * s.bins..(t + 1) * s.bins];
            for k in 0..=half {
                re[k] = row[k][0] as f64;
                im[k] = if k == 0 || k == half { 0.0 } else { row[k][1] as f64 };
            }
            for k in 1..half {
                re[nf - k] = re[k];
                im[nf - k] = -im[k];
            }
            fft(&mut re, &mut im, &self.tw, true);
            for i in 0..nf {
                out[i] = re[i] / nf as f64 * self.window[i] as f64;
            }
        });
        let mut y = vec![0f64; total];
        let mut env = vec![0f64; total];
        for t in 0..s.frames {
            for i in 0..nf {
                let w = self.window[i] as f64;
                y[t * self.hop + i] += frames[t * nf + i];
                env[t * self.hop + i] += w * w;
            }
        }
        (0..length).map(|i| {
            let j = i + half;
            if j < total && env[j] > 1e-11 { (y[j] / env[j]) as f32 } else { 0.0 }
        }).collect()
    }
}

/// The heads' output as the change of the spectrum: spectrum x mask + a bounded residual, where raw holds per bin
/// (mask re, mask im, residual re, residual im) - mask = 2 tanh(raw[0..2]); residual = tanh(raw[2..4]) over the larger
/// of its norm and 1, times `bound` |bin|. `raw(t, bin)` reads the heads' four values
pub fn correction(spec: &Spectrum, bound: f32, raw: impl Fn(usize, usize) -> [f32; 4]) -> Spectrum {
    let mut v = vec![[0f32; 2]; spec.v.len()];
    for t in 0..spec.frames {
        for k in 0..spec.bins {
            let [sr, si] = spec.v[t * spec.bins + k];
            let r = raw(t, k);
            let (mr, mi) = (2.0 * r[0].tanh(), 2.0 * r[1].tanh());
            let (c0, c1) = (r[2].tanh(), r[3].tanh());
            let norm = (c0 * c0 + c1 * c1).sqrt().max(1.0);
            let mag = sr.hypot(si);
            let (e0, e1) = (c0 / norm * (bound * mag), c1 / norm * (bound * mag));
            v[t * spec.bins + k] = [sr * mr - si * mi + e0, sr * mi + si * mr + e1];
        }
    }
    Spectrum { frames: spec.frames, bins: spec.bins, v }
}

/// torch.linspace(a, b, n) in float32 (the halves from either end)
fn linspace(a: f32, b: f32, n: usize) -> Vec<f32> {
    if n == 1 {
        return vec![a];
    }
    let step = (b - a) / (n - 1) as f32;
    (0..n).map(|i| if i < n / 2 { a + i as f32 * step } else { b - (n - 1 - i) as f32 * step }).collect()
}

/// The windows of an offline render (model.py enhance): (start, weights) a window of `chunk` samples, the last ending at
/// the clip's end; a clip no longer than one window is one pass (weights empty: taken whole)
pub fn windows(n: usize, chunk: usize, overlap: usize) -> Vec<(usize, Vec<f32>)> {
    if n <= chunk {
        return vec![(0, Vec::new())];
    }
    let mut starts: Vec<usize> = (0..n - chunk).step_by(chunk - overlap).collect();
    starts.push(n - chunk);
    let mut out = Vec::new();
    for (i, &s) in starts.iter().enumerate() {
        let mut w = vec![1f32; chunk];
        if i > 0 {
            let fade_in = starts[i - 1] + chunk - overlap - s;
            for x in w.iter_mut().take(fade_in) {
                *x = 0.0;
            }
            if overlap > 0 {
                for (j, v) in linspace(0.0, 1.0, overlap).into_iter().enumerate() {
                    w[fade_in + j] = v;
                }
            }
        }
        if overlap > 0 && s + chunk < n {
            for (j, v) in linspace(1.0, 0.0, overlap).into_iter().enumerate() {
                w[chunk - overlap + j] = v;
            }
        }
        out.push((s, w));
    }
    out
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// The modified Bessel function of the first kind, order 0 (its series)
fn i0(x: f64) -> f64 {
    let (mut sum, mut term, q) = (1.0, 1.0, x * x / 4.0);
    for k in 1..500 {
        term *= q / (k * k) as f64;
        sum += term;
        if term < sum * 1e-17 {
            break;
        }
    }
    sum
}

/// torchaudio.functional.resample with the reference's kaiser_best settings (sinc_interp_kaiser, 64 zero crossings,
/// rolloff 0.9475937167399596, beta 14.769656459379492): the kernel in float64 as torchaudio makes it (its phase
/// offsets in float32), then float32; out length ceil(new / orig * n)
pub fn resample(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || x.is_empty() {
        return x.to_vec();
    }
    const WIDTH: f64 = 64.0;
    const ROLLOFF: f64 = 0.9475937167399596;
    const BETA: f64 = 14.769656459379492;
    let g = gcd(from as u64, to as u64);
    let (orig, new) = ((from as u64 / g) as usize, (to as u64 / g) as usize);
    let base = orig.min(new) as f64 * ROLLOFF;
    let width = (WIDTH * orig as f64 / base).ceil() as usize;
    let taps = 2 * width + orig;
    let i0b = i0(BETA as f32 as f64);
    let scale = base / orig as f64;
    let mut kernel = vec![0f32; new * taps];
    for p in 0..new {
        let off = (-(p as f32) / new as f32) as f64;
        for k in 0..taps {
            let idx = (k as f64 - width as f64) / orig as f64;
            let t = ((off + idx) * base).clamp(-WIDTH, WIDTH);
            let r = 1.0 - (t / WIDTH) * (t / WIDTH);
            let window = i0(BETA as f32 as f64 * r.sqrt()) / i0b;
            let tp = t * PI;
            let s = if tp == 0.0 { 1.0 } else { tp.sin() / tp };
            kernel[p * taps + k] = (s * window * scale) as f32;
        }
    }
    let n = x.len();
    let out_len = ((new as f64 * n as f64) / orig as f64).ceil() as usize;
    let at = |i: isize| -> f32 { if i >= 0 && (i as usize) < n { x[i as usize] } else { 0.0 } };
    // a row a stride of `orig` input samples: its `new` outputs, one a phase (on the threads)
    let mut y = par_rows(out_len.div_ceil(new), new, |j, row: &mut [f32]| {
        let s0 = (j * orig) as isize - width as isize;
        for (p, out) in row.iter_mut().enumerate() {
            let kp = &kernel[p * taps..(p + 1) * taps];
            let mut acc = 0f32;
            for (k, w) in kp.iter().enumerate() {
                acc += w * at(s0 + k as isize);
            }
            *out = acc;
        }
    });
    y.truncate(out_len);
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_released_plan_makes_46_bands() {
        let plan = [(100.0, 1000.0), (250.0, 4000.0), (500.0, 8000.0), (1000.0, 16000.0), (2000.0, 24000.0)];
        let e = band_edges(2048, 48000, &plan);
        assert_eq!(e.len(), 46);
        assert_eq!(e[0], (0, 4));
        assert_eq!(e[10], (40, 43));
        assert_eq!(*e.last().unwrap(), (1024, 1025));
        assert!(e.windows(2).all(|w| w[0].1 == w[1].0));
    }

    #[test]
    fn stft_round_trips() {
        let s = Stft::new(2048, 480);
        let x: Vec<f32> = (0..20000).map(|i| ((i as f32) * 0.01).sin() * 0.5 + ((i * 7919) % 101) as f32 / 400.0).collect();
        let y = s.inverse(&s.forward(&x), x.len());
        let worst = x.iter().zip(&y).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(worst < 1e-5, "{worst}");
    }

    #[test]
    fn windows_cover_the_clip_with_unit_weight() {
        let (n, chunk, ov) = (1000, 300, 80);
        let ws = windows(n, chunk, ov);
        let mut sum = vec![0f32; n];
        for (s, w) in &ws {
            for (j, v) in w.iter().enumerate() {
                sum[s + j] += v;
            }
        }
        assert_eq!(ws.last().unwrap().0, n - chunk);
        assert!(sum.iter().all(|v| (v - 1.0).abs() < 1e-6), "{sum:?}");
    }

    #[test]
    fn resampling_keeps_a_tone() {
        let x: Vec<f32> = (0..16000).map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin()).collect();
        let y = resample(&x, 16000, 48000);
        assert_eq!(y.len(), 48000);
        let w: Vec<f32> = (0..48000).map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48000.0).sin()).collect();
        let worst = y[1000..47000].iter().zip(&w[1000..47000]).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(worst < 1e-3, "{worst}");
    }
}
