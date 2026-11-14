//! The products on x86's vector units, AVX2 and AVX-512 (F, BW, VL, VNNI), chosen at run time
//! (`is_x86_feature_detected`; a CPU with AVX-512 but not VNNI takes the AVX2 path).
//! Each decodes a block's values to int8 (grid lookups, signs, nibble levels), multiplies them with the activation's
//! int8 on `maddubs` (one side made unsigned, its sign moved to the other: exact, no pair reaches 2^15) and sums the
//! pairs with the block's scales (VNNI's `dpbusd` does both steps in one). A block's integer sums are exact; the
//! blocks are added in float32 lanes, so a product equals the plain one (blocks.rs) to float rounding. Blocks are
//! read through pointers: a row is exactly its blocks (the callers' slices), and the activation holds as many.

use super::blocks::{self, signs7, Quant, IQ1_S, IQ2_S, IQ2_XXS, IQ3_XXS, IQ4_NL, Q2_0, Q8_0};
use super::grids::{IQ1S_GRID, IQ2S_GRID, IQ2XXS_GRID, IQ3XXS_GRID, IQ4NL_VALUES};
use std::arch::x86_64::*;

/// An 8-bit sign mask as 8 bytes of 1 / -1 (`_mm256_sign_epi8`'s operand), for every mask
static SIGNS8: [u64; 256] = {
    let mut t = [0u64; 256];
    let mut m = 0;
    while m < 256 {
        let mut v = 0u64;
        let mut j = 0;
        while j < 8 {
            v |= (if m >> j & 1 == 1 { 0xffu64 } else { 0x01 }) << (8 * j);
            j += 1;
        }
        t[m] = v;
        m += 1;
    }
    t
};

/// The 7-bit sign codes (IQ2_XXS, IQ3_XXS) as 8 bytes of 1 / -1
static KSIGNS8: [u64; 128] = {
    let mut t = [0u64; 128];
    let mut k = 0;
    while k < 128 {
        t[k] = SIGNS8[signs7(k as u32) as usize];
        k += 1;
    }
    t
};

/// The 7-bit sign codes as their 8-bit masks
static KSIGNS: [u8; 128] = {
    let mut t = [0u8; 128];
    let mut k = 0;
    while k < 128 {
        t[k] = signs7(k as u32) as u8;
        k += 1;
    }
    t
};

/// Whether the vector paths may run here
pub fn has_avx2() -> bool {
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && is_x86_feature_detected!("f16c")
}

/// Whether BMI2's pdep is fast here: it is microcode on AMD before Zen 3 (family 0x19), hundreds of cycles. Asked
/// once (cpuid traps to the hypervisor in a VM)
pub fn fast_pdep() -> bool {
    static FAST: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FAST.get_or_init(|| {
        if !is_x86_feature_detected!("bmi2") {
            return false;
        }
        let (v, f) = (__cpuid(0), __cpuid(1));
        let amd = v.ebx == 0x6874_7541 && v.edx == 0x6974_6e65 && v.ecx == 0x444d_4163;
        let family = ((f.eax >> 8) & 0xf) + if (f.eax >> 8) & 0xf == 0xf { (f.eax >> 20) & 0xff } else { 0 };
        !amd || family >= 0x19
    })
}

pub fn has_avx512() -> bool {
    has_avx2()
        && is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512vl")
        && is_x86_feature_detected!("avx512vnni")
}

#[inline(always)]
unsafe fn u32_at(p: *const u8) -> u32 {
    std::ptr::read_unaligned(p.cast::<u32>())
}

#[inline(always)]
unsafe fn u16_at(p: *const u8) -> usize {
    std::ptr::read_unaligned(p.cast::<u16>()) as usize
}

/// the half at p, as float32
#[inline(always)]
unsafe fn h2f(p: *const u8) -> f32 {
    blocks::half_bits(std::ptr::read_unaligned(p.cast::<u16>()))
}

/// a 7-bit sign code (the low 7 bits of x >> s) as 8 bytes of 1 / -1
#[inline(always)]
unsafe fn ks8(x: u32, s: u32) -> i64 {
    *KSIGNS8.get_unchecked((x >> s & 127) as usize) as i64
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hsum_ps(v: __m256) -> f32 {
    let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_movehdup_ps(s));
    _mm_cvtss_f32(s)
}

/// sum over pairs of w . y as int32 lanes, w and y int8 (the sign of w moved onto y)
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn dot8(w: __m256i, y: __m256i) -> __m256i {
    _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(w), _mm256_sign_epi8(y, w)), _mm256_set1_epi16(1))
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn dot8_512(w: __m512i, y: __m512i) -> __m512i {
    let ys = _mm512_mask_sub_epi8(y, _mm512_movepi8_mask(w), _mm512_setzero_si512(), y);
    _mm512_dpbusd_epi32(_mm512_setzero_si512(), _mm512_abs_epi8(w), ys)
}

/// int16 lanes: the low 256 bits a, the high b
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn halves16(a: i16, b: i16) -> __m512i {
    _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_set1_epi16(a)), _mm256_set1_epi16(b), 1)
}

/// float32 lanes: the low 256 bits a, the high b
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn halves_ps(a: f32, b: f32) -> __m512 {
    _mm512_mask_blend_ps(0xff00, _mm512_set1_ps(a), _mm512_set1_ps(b))
}

// ---- IQ2_XXS (66 bytes: half d, then per 32 values 4 index bytes and a word of signs and scale)

/// one 32's int32 lanes times its scale
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn iq2xxs_32(p: *const u8, q8: *const i8) -> __m256i {
    let (idx, aux) = (u32_at(p), u32_at(p.add(4)));
    let g = |s: u32| *IQ2XXS_GRID.get_unchecked((idx >> s & 255) as usize) as i64;
    let gv = _mm256_set_epi64x(g(24), g(16), g(8), g(0));
    let sv = _mm256_set_epi64x(ks8(aux, 21), ks8(aux, 14), ks8(aux, 7), ks8(aux, 0));
    let p16 = _mm256_maddubs_epi16(gv, _mm256_sign_epi8(_mm256_loadu_si256(q8.cast()), sv));
    _mm256_madd_epi16(p16, _mm256_set1_epi16((2 * (aux >> 28) + 1) as i16))
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn iq2xxs_avx2(w: *const u8, nb: usize, a: &Quant) -> f32 {
    let mut sumf = _mm256_setzero_ps();
    for i in 0..nb {
        let blk = w.add(i * 66);
        let q8 = a.qs.as_ptr().add(i * 256);
        let (mut s0, mut s1) = (_mm256_setzero_si256(), _mm256_setzero_si256());
        for ib in (0..8).step_by(2) {
            s0 = _mm256_add_epi32(s0, iq2xxs_32(blk.add(2 + 8 * ib), q8.add(32 * ib)));
            s1 = _mm256_add_epi32(s1, iq2xxs_32(blk.add(10 + 8 * ib), q8.add(32 * ib + 32)));
        }
        sumf = _mm256_fmadd_ps(_mm256_set1_ps(h2f(blk) * *a.d.get_unchecked(i)), _mm256_cvtepi32_ps(_mm256_add_epi32(s0, s1)), sumf);
    }
    0.125 * hsum_ps(sumf)
}

/// the signs of two 32s of IQ2_XXS / IQ3_XXS (their sign words a and b) as a mask, a bit a value
#[inline(always)]
unsafe fn mask7(a: u32, b: u32) -> u64 {
    let k = |x: u32, s: u32| *KSIGNS.get_unchecked((x >> s & 127) as usize) as u64;
    k(a, 0) | k(a, 7) << 8 | k(a, 14) << 16 | k(a, 21) << 24 | k(b, 0) << 32 | k(b, 7) << 40 | k(b, 14) << 48 | k(b, 21) << 56
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,avx2,fma,f16c")]
unsafe fn iq2xxs_512(w: *const u8, nb: usize, a: &Quant) -> f32 {
    let mut sumf = _mm512_setzero_ps();
    let z = _mm512_setzero_si512();
    for i in 0..nb {
        let blk = w.add(i * 66);
        let q8 = a.qs.as_ptr().add(i * 256);
        let mut acc = _mm512_setzero_si512();
        for ib in (0..8).step_by(2) {
            let p = blk.add(2 + 8 * ib);
            let (i0, a0, i1, a1) = (u32_at(p), u32_at(p.add(4)), u32_at(p.add(8)), u32_at(p.add(12)));
            let g = |x: u32, s: u32| *IQ2XXS_GRID.get_unchecked((x >> s & 255) as usize) as i64;
            let gv = _mm512_set_epi64(g(i1, 24), g(i1, 16), g(i1, 8), g(i1, 0), g(i0, 24), g(i0, 16), g(i0, 8), g(i0, 0));
            let q = _mm512_loadu_si512(q8.add(ib * 32).cast());
            let p16 = _mm512_maddubs_epi16(gv, _mm512_mask_sub_epi8(q, mask7(a0, a1), z, q));
            acc = _mm512_add_epi32(acc, _mm512_madd_epi16(p16, halves16((2 * (a0 >> 28) + 1) as i16, (2 * (a1 >> 28) + 1) as i16)));
        }
        sumf = _mm512_fmadd_ps(_mm512_set1_ps(h2f(blk) * *a.d.get_unchecked(i)), _mm512_cvtepi32_ps(acc), sumf);
    }
    0.125 * _mm512_reduce_add_ps(sumf)
}

// ---- IQ2_S (82 bytes: half d, 32 index bytes, 32 sign bytes, 8 bytes of index high bits, 8 of two scales)

#[inline(always)]
unsafe fn iq2s_grid(qs: *const u8, qh: u8, l: usize) -> i64 {
    *IQ2S_GRID.get_unchecked((*qs.add(l) as usize) | ((qh as usize) << (8 - 2 * (l & 3)) & 0x300)) as i64
}

/// 4 sign bytes (32 values) as a mask of 0xff where a value is negative
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn signs32(p: *const u8) -> __m256i {
    let pick = _mm256_set_epi64x(0x0303_0303_0303_0303, 0x0202_0202_0202_0202, 0x0101_0101_0101_0101, 0);
    let bit = _mm256_set1_epi64x(0x8040_2010_0804_0201u64 as i64);
    let v = _mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32(u32_at(p) as i32), pick), bit);
    _mm256_cmpeq_epi8(v, bit)
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn iq2s_32(blk: *const u8, ib: usize, q8: *const i8, ls: __m256i) -> __m256i {
    let (qs, qh) = (blk.add(2 + 4 * ib), *blk.add(66 + ib));
    let gv = _mm256_set_epi64x(iq2s_grid(qs, qh, 3), iq2s_grid(qs, qh, 2), iq2s_grid(qs, qh, 1), iq2s_grid(qs, qh, 0));
    let m = signs32(blk.add(34 + 4 * ib));
    let q = _mm256_loadu_si256(q8.cast());
    let p16 = _mm256_maddubs_epi16(gv, _mm256_sub_epi8(_mm256_xor_si256(q, m), m));
    _mm256_madd_epi16(p16, ls)
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn iq2s_avx2(w: *const u8, nb: usize, a: &Quant) -> f32 {
    let mut sumf = _mm256_setzero_ps();
    let (m4, one) = (_mm_set1_epi8(0x0f), _mm_set1_epi8(1));
    for i in 0..nb {
        let blk = w.add(i * 82);
        let q8 = a.qs.as_ptr().add(i * 256);
        // the scales 2s + 1 as int16: the 8 low nibbles in the low 128 bits, the high ones in the high 128
        let sc = std::ptr::read_unaligned(blk.add(74).cast::<u64>()) as i64;
        let s8 = _mm_add_epi8(_mm_slli_epi16(_mm_and_si128(_mm_set_epi64x(sc >> 4, sc), m4), 1), one);
        let sc16 = _mm256_cvtepu8_epi16(s8);
        // a 32's int16 lanes: its low nibble's scale for the first 16 values, its high one's for the rest
        let ls = |ib: usize| _mm256_shuffle_epi8(sc16, _mm256_set1_epi16(((2 * ib) | ((2 * ib + 1) << 8)) as i16));
        let (mut s0, mut s1) = (_mm256_setzero_si256(), _mm256_setzero_si256());
        for ib in (0..8).step_by(2) {
            s0 = _mm256_add_epi32(s0, iq2s_32(blk, ib, q8.add(32 * ib), ls(ib)));
            s1 = _mm256_add_epi32(s1, iq2s_32(blk, ib + 1, q8.add(32 * ib + 32), ls(ib + 1)));
        }
        sumf = _mm256_fmadd_ps(_mm256_set1_ps(h2f(blk) * *a.d.get_unchecked(i)), _mm256_cvtepi32_ps(_mm256_add_epi32(s0, s1)), sumf);
    }
    0.125 * hsum_ps(sumf)
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,avx2,fma,f16c")]
unsafe fn iq2s_512(w: *const u8, nb: usize, a: &Quant) -> f32 {
    let mut sumf = _mm512_setzero_ps();
    let z = _mm512_setzero_si512();
    for i in 0..nb {
        let blk = w.add(i * 82);
        let q8 = a.qs.as_ptr().add(i * 256);
        let mut acc = _mm512_setzero_si512();
        for ib in (0..8).step_by(2) {
            let qs = blk.add(2 + 4 * ib);
            let (h0, h1) = (*blk.add(66 + ib), *blk.add(67 + ib));
            let gv = _mm512_set_epi64(
                iq2s_grid(qs, h1, 7),
                iq2s_grid(qs, h1, 6),
                iq2s_grid(qs, h1, 5),
                iq2s_grid(qs, h1, 4),
                iq2s_grid(qs, h0, 3),
                iq2s_grid(qs, h0, 2),
                iq2s_grid(qs, h0, 1),
                iq2s_grid(qs, h0, 0),
            );
            // the sign bytes of the two 32s are 8 bytes in a row: a bit a value
            let m = std::ptr::read_unaligned(blk.add(34 + 4 * ib).cast::<u64>());
            let q = _mm512_loadu_si512(q8.add(ib * 32).cast());
            let p16 = _mm512_maddubs_epi16(gv, _mm512_mask_sub_epi8(q, m, z, q));
            // int32 lanes 0-3 the first 32's low scale, 4-7 its high, 8-15 the second's
            let l = |sc: u8| {
                let (lo, hi) = (1 + 2 * (sc & 0xf) as i32, 1 + 2 * (sc >> 4) as i32);
                _mm256_set_m128i(_mm_set1_epi32(hi | hi << 16), _mm_set1_epi32(lo | lo << 16))
            };
            let ls = _mm512_inserti64x4(_mm512_castsi256_si512(l(*blk.add(74 + ib))), l(*blk.add(75 + ib)), 1);
            acc = _mm512_add_epi32(acc, _mm512_madd_epi16(p16, ls));
        }
        sumf = _mm512_fmadd_ps(_mm512_set1_ps(h2f(blk) * *a.d.get_unchecked(i)), _mm512_cvtepi32_ps(acc), sumf);
    }
    0.125 * _mm512_reduce_add_ps(sumf)
}

// ---- IQ3_XXS (98 bytes: half d, 64 index bytes (4 values each), then per 32 a word of signs and scale)

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn iq3xxs_32(blk: *const u8, ib: usize, q8: *const i8) -> __m256i {
    let q3 = blk.add(2 + 8 * ib);
    let aux = u32_at(blk.add(66 + 4 * ib));
    let g = |l: usize| *IQ3XXS_GRID.get_unchecked(*q3.add(l) as usize) as i32;
    let gv = _mm256_set_epi32(g(7), g(6), g(5), g(4), g(3), g(2), g(1), g(0));
    let sv = _mm256_set_epi64x(ks8(aux, 21), ks8(aux, 14), ks8(aux, 7), ks8(aux, 0));
    let p16 = _mm256_maddubs_epi16(gv, _mm256_sign_epi8(_mm256_loadu_si256(q8.cast()), sv));
    _mm256_madd_epi16(p16, _mm256_set1_epi16((2 * (aux >> 28) + 1) as i16))
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn iq3xxs_avx2(w: *const u8, nb: usize, a: &Quant) -> f32 {
    let mut sumf = _mm256_setzero_ps();
    for i in 0..nb {
        let blk = w.add(i * 98);
        let q8 = a.qs.as_ptr().add(i * 256);
        let (mut s0, mut s1) = (_mm256_setzero_si256(), _mm256_setzero_si256());
        for ib in (0..8).step_by(2) {
            s0 = _mm256_add_epi32(s0, iq3xxs_32(blk, ib, q8.add(32 * ib)));
            s1 = _mm256_add_epi32(s1, iq3xxs_32(blk, ib + 1, q8.add(32 * ib + 32)));
        }
        sumf = _mm256_fmadd_ps(_mm256_set1_ps(h2f(blk) * *a.d.get_unchecked(i)), _mm256_cvtepi32_ps(_mm256_add_epi32(s0, s1)), sumf);
    }
    0.25 * hsum_ps(sumf)
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,avx2,fma,f16c")]
unsafe fn iq3xxs_512(w: *const u8, nb: usize, a: &Quant) -> f32 {
    let mut sumf = _mm512_setzero_ps();
    let z = _mm512_setzero_si512();
    for i in 0..nb {
        let blk = w.add(i * 98);
        let q8 = a.qs.as_ptr().add(i * 256);
        let mut acc = _mm512_setzero_si512();
        for ib in (0..8).step_by(2) {
            let q3 = blk.add(2 + 8 * ib);
            let g = |l: usize| *IQ3XXS_GRID.get_unchecked(*q3.add(l) as usize) as i32;
            let gv = _mm512_inserti64x4(
                _mm512_castsi256_si512(_mm256_set_epi32(g(7), g(6), g(5), g(4), g(3), g(2), g(1), g(0))),
                _mm256_set_epi32(g(15), g(14), g(13), g(12), g(11), g(10), g(9), g(8)),
                1,
            );
            let (a0, a1) = (u32_at(blk.add(66 + 4 * ib)), u32_at(blk.add(70 + 4 * ib)));
            let q = _mm512_loadu_si512(q8.add(ib * 32).cast());
            let p16 = _mm512_maddubs_epi16(gv, _mm512_mask_sub_epi8(q, mask7(a0, a1), z, q));
            acc = _mm512_add_epi32(acc, _mm512_madd_epi16(p16, halves16((2 * (a0 >> 28) + 1) as i16, (2 * (a1 >> 28) + 1) as i16)));
        }
        sumf = _mm512_fmadd_ps(_mm512_set1_ps(h2f(blk) * *a.d.get_unchecked(i)), _mm512_cvtepi32_ps(acc), sumf);
    }
    0.25 * _mm512_reduce_add_ps(sumf)
}

// ---- IQ1_S (50 bytes: half d, 32 index bytes (values -1 / 0 / 1), per 32 a word of index high bits, scale, shift sign)

#[inline(always)]
unsafe fn iq1s_grid(qs: *const u8, qh: usize, l: usize) -> i64 {
    *IQ1S_GRID.get_unchecked(*qs.add(l) as usize | (((qh >> (3 * (l & 3))) & 7) << 8)) as i64
}

/// a 32's four grid entries: the low index bytes at qs, the high bits in qh's low 12, laid into 16-bit fields by pdep
#[inline]
#[target_feature(enable = "avx2,bmi2")]
unsafe fn iq1s_grids_pdep(qs: *const u8, qh: usize) -> __m256i {
    let idx = _pdep_u64(u32_at(qs) as u64, 0x00ff_00ff_00ff_00ff) | _pdep_u64(qh as u64, 0x0700_0700_0700_0700);
    let g = |l: u32| *IQ1S_GRID.get_unchecked((idx >> (16 * l) & 0xffff) as usize) as i64;
    _mm256_set_epi64x(g(3), g(2), g(1), g(0))
}

/// the 8 shift terms of an IQ1_S block: scale x the shift's sign (bit 15) x each 32's activation sum, as int32
/// lanes (the scales 2 x qh[12..15] + 1, from one load of the 8 words)
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn iq1s_shift(blk: *const u8, bs: *const i16) -> __m256i {
    let qh = _mm_loadu_si128(blk.add(34).cast());
    let ls = _mm_add_epi16(_mm_and_si128(_mm_srli_epi16(qh, 11), _mm_set1_epi16(0xe)), _mm_set1_epi16(1));
    let neg = _mm_srai_epi16(qh, 15);
    let lsd = _mm_sub_epi16(_mm_xor_si128(ls, neg), neg);
    // each 32's two sums of 16 against its scale, twice
    let l2 = _mm256_set_m128i(_mm_unpackhi_epi16(lsd, lsd), _mm_unpacklo_epi16(lsd, lsd));
    _mm256_madd_epi16(_mm256_loadu_si256(bs.cast()), l2)
}

/// the IQ1_S product on AVX2, a 32's grid entries from `grids` (four lookups, or pdep's)
#[inline(always)]
unsafe fn iq1s_body(w: *const u8, nb: usize, a: &Quant, grids: impl Fn(*const u8, usize) -> __m256i) -> f32 {
    let (mut sumf, mut sum1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
    for i in 0..nb {
        let blk = w.add(i * 50);
        let q8 = a.qs.as_ptr().add(i * 256);
        let (mut a0, mut a1) = (_mm256_setzero_si256(), _mm256_setzero_si256());
        for ib in (0..8).step_by(2) {
            let (h0, h1) = (u16_at(blk.add(34 + 2 * ib)), u16_at(blk.add(36 + 2 * ib)));
            let qs = blk.add(2 + 4 * ib);
            let (g0, g1) = (grids(qs, h0), grids(qs.add(4), h1));
            // the grid's values are -1 / 0 / 1: |g| . (q8 with g's sign)
            let p0 = _mm256_maddubs_epi16(_mm256_sign_epi8(g0, g0), _mm256_sign_epi8(_mm256_loadu_si256(q8.add(32 * ib).cast()), g0));
            let p1 = _mm256_maddubs_epi16(_mm256_sign_epi8(g1, g1), _mm256_sign_epi8(_mm256_loadu_si256(q8.add(32 * ib + 32).cast()), g1));
            a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(p0, _mm256_set1_epi16((2 * ((h0 >> 12) & 7) + 1) as i16)));
            a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(p1, _mm256_set1_epi16((2 * ((h1 >> 12) & 7) + 1) as i16)));
        }
        let d = _mm256_set1_ps(h2f(blk) * *a.d.get_unchecked(i));
        sumf = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(_mm256_add_epi32(a0, a1)), sumf);
        sum1 = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(iq1s_shift(blk, a.bsums.as_ptr().add(i * 16))), sum1);
    }
    hsum_ps(sumf) + 0.125 * hsum_ps(sum1)
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn iq1s_avx2(w: *const u8, nb: usize, a: &Quant) -> f32 {
    iq1s_body(w, nb, a, |qs, qh| _mm256_set_epi64x(iq1s_grid(qs, qh, 3), iq1s_grid(qs, qh, 2), iq1s_grid(qs, qh, 1), iq1s_grid(qs, qh, 0)))
}

/// as iq1s_avx2, a 32's four grid indices made at once by pdep
#[target_feature(enable = "avx2,fma,f16c,bmi2")]
unsafe fn iq1s_avx2_pdep(w: *const u8, nb: usize, a: &Quant) -> f32 {
    iq1s_body(w, nb, a, |qs, qh| iq1s_grids_pdep(qs, qh))
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,avx2,fma,f16c")]
unsafe fn iq1s_512(w: *const u8, nb: usize, a: &Quant) -> f32 {
    let (mut sumf, mut sum1) = (_mm512_setzero_ps(), 0f32);
    let z = _mm512_setzero_si512();
    for i in 0..nb {
        let blk = w.add(i * 50);
        let q8 = a.qs.as_ptr().add(i * 256);
        let bs = a.bsums.as_ptr().add(i * 16);
        let mut acc = _mm512_setzero_si512();
        let mut sumi1 = 0i32;
        for ib in (0..8).step_by(2) {
            let (h0, h1) = (u16_at(blk.add(34 + 2 * ib)), u16_at(blk.add(36 + 2 * ib)));
            let (l0, l1) = (2 * ((h0 >> 12) & 7) as i32 + 1, 2 * ((h1 >> 12) & 7) as i32 + 1);
            let qs = blk.add(2 + 4 * ib);
            let gv = _mm512_set_epi64(
                iq1s_grid(qs, h1, 7),
                iq1s_grid(qs, h1, 6),
                iq1s_grid(qs, h1, 5),
                iq1s_grid(qs, h1, 4),
                iq1s_grid(qs, h0, 3),
                iq1s_grid(qs, h0, 2),
                iq1s_grid(qs, h0, 1),
                iq1s_grid(qs, h0, 0),
            );
            let q = _mm512_loadu_si512(q8.add(ib * 32).cast());
            let p16 = _mm512_maddubs_epi16(_mm512_abs_epi8(gv), _mm512_mask_sub_epi8(q, _mm512_movepi8_mask(gv), z, q));
            acc = _mm512_add_epi32(acc, _mm512_madd_epi16(p16, halves16(l0 as i16, l1 as i16)));
            let sb = |k: usize| *bs.add(k) as i32;
            let (e0, e1) = (1 - 2 * (h0 >> 15) as i32, 1 - 2 * (h1 >> 15) as i32);
            sumi1 += l0 * e0 * (sb(2 * ib) + sb(2 * ib + 1)) + l1 * e1 * (sb(2 * ib + 2) + sb(2 * ib + 3));
        }
        let d = h2f(blk) * *a.d.get_unchecked(i);
        sumf = _mm512_fmadd_ps(_mm512_set1_ps(d), _mm512_cvtepi32_ps(acc), sumf);
        sum1 += d * sumi1 as f32;
    }
    _mm512_reduce_add_ps(sumf) + 0.125 * sum1
}

/// as iq1s_512, the grid indices made by pdep
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,avx2,fma,f16c,bmi2")]
unsafe fn iq1s_512_pdep(w: *const u8, nb: usize, a: &Quant) -> f32 {
    let (mut sumf, mut sum1) = (_mm512_setzero_ps(), 0f32);
    let z = _mm512_setzero_si512();
    for i in 0..nb {
        let blk = w.add(i * 50);
        let q8 = a.qs.as_ptr().add(i * 256);
        let bs = a.bsums.as_ptr().add(i * 16);
        let mut acc = _mm512_setzero_si512();
        let mut sumi1 = 0i32;
        for ib in (0..8).step_by(2) {
            let (h0, h1) = (u16_at(blk.add(34 + 2 * ib)), u16_at(blk.add(36 + 2 * ib)));
            let (l0, l1) = (2 * ((h0 >> 12) & 7) as i32 + 1, 2 * ((h1 >> 12) & 7) as i32 + 1);
            let qs = blk.add(2 + 4 * ib);
            let gv = _mm512_inserti64x4(_mm512_castsi256_si512(iq1s_grids_pdep(qs, h0)), iq1s_grids_pdep(qs.add(4), h1), 1);
            let q = _mm512_loadu_si512(q8.add(ib * 32).cast());
            let p16 = _mm512_maddubs_epi16(_mm512_abs_epi8(gv), _mm512_mask_sub_epi8(q, _mm512_movepi8_mask(gv), z, q));
            acc = _mm512_add_epi32(acc, _mm512_madd_epi16(p16, halves16(l0 as i16, l1 as i16)));
            let sb = |k: usize| *bs.add(k) as i32;
            let (e0, e1) = (1 - 2 * (h0 >> 15) as i32, 1 - 2 * (h1 >> 15) as i32);
            sumi1 += l0 * e0 * (sb(2 * ib) + sb(2 * ib + 1)) + l1 * e1 * (sb(2 * ib + 2) + sb(2 * ib + 3));
        }
        let d = h2f(blk) * *a.d.get_unchecked(i);
        sumf = _mm512_fmadd_ps(_mm512_set1_ps(d), _mm512_cvtepi32_ps(acc), sumf);
        sum1 += d * sumi1 as f32;
    }
    _mm512_reduce_add_ps(sumf) + 0.125 * sum1
}

// ---- IQ4_NL (18 bytes: half d, 16 bytes of two level indices; against Q8_0)

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn iq4nl_values(blk: *const u8, t: __m128i) -> __m256i {
    let q = _mm_loadu_si128(blk.add(2).cast());
    let m = _mm_set1_epi8(0x0f);
    _mm256_set_m128i(_mm_shuffle_epi8(t, _mm_and_si128(_mm_srli_epi16(q, 4), m)), _mm_shuffle_epi8(t, _mm_and_si128(q, m)))
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn iq4nl_avx2(w: *const u8, nb: usize, a: &Quant) -> f32 {
    let t = _mm_loadu_si128(IQ4NL_VALUES.as_ptr().cast());
    let (mut acc0, mut acc1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
    let (y, d) = (a.qs.as_ptr(), a.d.as_ptr());
    let mut i = 0;
    while i + 2 <= nb {
        let (b0, b1) = (w.add(i * 18), w.add(i * 18 + 18));
        let p0 = dot8(iq4nl_values(b0, t), _mm256_loadu_si256(y.add(i * 32).cast()));
        let p1 = dot8(iq4nl_values(b1, t), _mm256_loadu_si256(y.add(i * 32 + 32).cast()));
        acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p0), _mm256_set1_ps(*d.add(i) * h2f(b0)), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p1), _mm256_set1_ps(*d.add(i + 1) * h2f(b1)), acc1);
        i += 2;
    }
    if i < nb {
        let b0 = w.add(i * 18);
        let p0 = dot8(iq4nl_values(b0, t), _mm256_loadu_si256(y.add(i * 32).cast()));
        acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p0), _mm256_set1_ps(*d.add(i) * h2f(b0)), acc0);
    }
    hsum_ps(_mm256_add_ps(acc0, acc1))
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,avx2,fma,f16c")]
unsafe fn iq4nl_512(w: *const u8, nb: usize, a: &Quant) -> f32 {
    let t = _mm_loadu_si128(IQ4NL_VALUES.as_ptr().cast());
    let t4 = _mm512_broadcast_i32x4(t);
    let m = _mm512_set1_epi8(0x0f);
    let (y, d) = (a.qs.as_ptr(), a.d.as_ptr());
    let mut acc = _mm512_setzero_ps();
    let mut i = 0;
    while i + 2 <= nb {
        let (b0, b1) = (w.add(i * 18), w.add(i * 18 + 18));
        let (q0, q1) = (_mm_loadu_si128(b0.add(2).cast()), _mm_loadu_si128(b1.add(2).cast()));
        let lo = _mm256_set_m128i(_mm_srli_epi16(q0, 4), q0);
        let hi = _mm256_set_m128i(_mm_srli_epi16(q1, 4), q1);
        let wv = _mm512_shuffle_epi8(t4, _mm512_and_si512(_mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1), m));
        let p = dot8_512(wv, _mm512_loadu_si512(y.add(i * 32).cast()));
        acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(p), halves_ps(*d.add(i) * h2f(b0), *d.add(i + 1) * h2f(b1)), acc);
        i += 2;
    }
    let mut s = _mm512_reduce_add_ps(acc);
    if i < nb {
        let b0 = w.add(i * 18);
        let p0 = dot8(iq4nl_values(b0, t), _mm256_loadu_si256(y.add(i * 32).cast()));
        s += hsum_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(p0), _mm256_set1_ps(*d.add(i) * h2f(b0))));
    }
    s
}

// ---- Q2_0 (18 bytes: half d, 16 bytes of 2-bit values, four a byte from the lowest bits; against Act::Q20)

/// a block's values (0..3, before the shift by one): bytes shifted by 0 and 2 (the first vector), 4 and 6 - the
/// activation is in that order
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn q20_values(blk: *const u8) -> (__m256i, __m256i) {
    let q = _mm256_broadcastsi128_si256(_mm_loadu_si128(blk.add(2).cast()));
    let m3 = _mm256_set1_epi8(3);
    (_mm256_and_si256(_mm256_srlv_epi64(q, _mm256_set_epi64x(2, 2, 0, 0)), m3), _mm256_and_si256(_mm256_srlv_epi64(q, _mm256_set_epi64x(6, 6, 4, 4)), m3))
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn q20_avx2(w: *const u8, nb: usize, a: &Quant) -> f32 {
    let (mut acc0, mut acc1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
    let mut csum = 0f32;
    let (y, pat, c) = (a.qs.as_ptr(), a.pat.as_ptr(), a.c.as_ptr());
    let ones = _mm256_set1_epi16(1);
    let one = |g: usize| {
        let blk = w.add(g * 18);
        let (v0, v1) = q20_values(blk);
        // the values are unsigned: maddubs takes them as they are
        let p0 = _mm256_maddubs_epi16(v0, _mm256_loadu_si256(y.add(64 * g).cast()));
        let p1 = _mm256_maddubs_epi16(v1, _mm256_loadu_si256(y.add(64 * g + 32).cast()));
        let p = _mm256_madd_epi16(_mm256_add_epi16(p0, p1), ones);
        (_mm256_mul_ps(_mm256_cvtepi32_ps(p), _mm256_loadu_ps(pat.add(8 * g))), h2f(blk))
    };
    let mut g = 0;
    while g + 2 <= nb {
        let (p0, d0) = one(g);
        let (p1, d1) = one(g + 1);
        acc0 = _mm256_fmadd_ps(p0, _mm256_set1_ps(d0), acc0);
        acc1 = _mm256_fmadd_ps(p1, _mm256_set1_ps(d1), acc1);
        csum += d0 * *c.add(g) + d1 * *c.add(g + 1);
        g += 2;
    }
    if g < nb {
        let (p0, d0) = one(g);
        acc0 = _mm256_fmadd_ps(p0, _mm256_set1_ps(d0), acc0);
        csum += d0 * *c.add(g);
    }
    hsum_ps(_mm256_add_ps(acc0, acc1)) - csum
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,avx2,fma,f16c")]
unsafe fn q20_512(w: *const u8, nb: usize, a: &Quant) -> f32 {
    let (mut acc0, mut acc1) = (_mm512_setzero_ps(), _mm512_setzero_ps());
    let mut csum = 0f32;
    let (y, pat, c) = (a.qs.as_ptr(), a.pat.as_ptr(), a.c.as_ptr());
    let (m3, sh) = (_mm512_set1_epi8(3), _mm512_set_epi64(6, 6, 4, 4, 2, 2, 0, 0));
    let one = |g: usize| {
        let blk = w.add(g * 18);
        let v = _mm512_and_si512(_mm512_srlv_epi64(_mm512_broadcast_i32x4(_mm_loadu_si128(blk.add(2).cast())), sh), m3);
        let p = _mm512_dpbusd_epi32(_mm512_setzero_si512(), v, _mm512_loadu_si512(y.add(64 * g).cast()));
        let pv = _mm512_castpd_ps(_mm512_broadcast_f64x4(_mm256_castps_pd(_mm256_loadu_ps(pat.add(8 * g)))));
        (_mm512_mul_ps(_mm512_cvtepi32_ps(p), pv), h2f(blk))
    };
    let mut g = 0;
    while g + 2 <= nb {
        let (p0, d0) = one(g);
        let (p1, d1) = one(g + 1);
        acc0 = _mm512_fmadd_ps(p0, _mm512_set1_ps(d0), acc0);
        acc1 = _mm512_fmadd_ps(p1, _mm512_set1_ps(d1), acc1);
        csum += d0 * *c.add(g) + d1 * *c.add(g + 1);
        g += 2;
    }
    if g < nb {
        let (p0, d0) = one(g);
        acc0 = _mm512_fmadd_ps(p0, _mm512_set1_ps(d0), acc0);
        csum += d0 * *c.add(g);
    }
    _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1)) - csum
}

// ---- Q8_0 (34 bytes: half d, 32 int8)

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn q80_avx2(w: *const u8, nb: usize, a: &Quant) -> f32 {
    let (mut acc0, mut acc1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
    let (y, d) = (a.qs.as_ptr(), a.d.as_ptr());
    let one = |i: usize| {
        let blk = w.add(i * 34);
        let p = dot8(_mm256_loadu_si256(blk.add(2).cast()), _mm256_loadu_si256(y.add(i * 32).cast()));
        (_mm256_cvtepi32_ps(p), _mm256_set1_ps(h2f(blk) * *d.add(i)))
    };
    let mut i = 0;
    while i + 2 <= nb {
        let (p0, d0) = one(i);
        let (p1, d1) = one(i + 1);
        acc0 = _mm256_fmadd_ps(p0, d0, acc0);
        acc1 = _mm256_fmadd_ps(p1, d1, acc1);
        i += 2;
    }
    if i < nb {
        let (p0, d0) = one(i);
        acc0 = _mm256_fmadd_ps(p0, d0, acc0);
    }
    hsum_ps(_mm256_add_ps(acc0, acc1))
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,avx2,fma,f16c")]
unsafe fn q80_512(w: *const u8, nb: usize, a: &Quant) -> f32 {
    let (y, d) = (a.qs.as_ptr(), a.d.as_ptr());
    let mut acc = _mm512_setzero_ps();
    let mut i = 0;
    while i + 2 <= nb {
        let (b0, b1) = (w.add(i * 34), w.add(i * 34 + 34));
        let wv = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256(b0.add(2).cast())), _mm256_loadu_si256(b1.add(2).cast()), 1);
        let p = dot8_512(wv, _mm512_loadu_si512(y.add(i * 32).cast()));
        acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(p), halves_ps(h2f(b0) * *d.add(i), h2f(b1) * *d.add(i + 1)), acc);
        i += 2;
    }
    let mut s = _mm512_reduce_add_ps(acc);
    if i < nb {
        let b0 = w.add(i * 34);
        let p = dot8(_mm256_loadu_si256(b0.add(2).cast()), _mm256_loadu_si256(y.add(i * 32).cast()));
        s += hsum_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(h2f(b0) * *d.add(i))));
    }
    s
}

/// A product of a row (its blocks at a pointer, how many) and an activation that holds as many
pub type Kernel = unsafe fn(*const u8, usize, &Quant) -> f32;

/// `ty`'s product on AVX2 (the caller has checked has_avx2() before calling it)
pub fn kernel_avx2(ty: i32) -> Option<Kernel> {
    Some(match ty {
        IQ2_XXS => iq2xxs_avx2,
        IQ2_S => iq2s_avx2,
        IQ3_XXS => iq3xxs_avx2,
        IQ1_S if fast_pdep() => iq1s_avx2_pdep,
        IQ1_S => iq1s_avx2,
        IQ4_NL => iq4nl_avx2,
        Q2_0 => q20_avx2,
        Q8_0 => q80_avx2,
        _ => return None,
    })
}

/// `ty`'s product on AVX-512 (the caller has checked has_avx512() before calling it)
pub fn kernel_avx512(ty: i32) -> Option<Kernel> {
    Some(match ty {
        IQ2_XXS => iq2xxs_512,
        IQ2_S => iq2s_512,
        IQ3_XXS => iq3xxs_512,
        IQ1_S if fast_pdep() => iq1s_512_pdep,
        IQ1_S => iq1s_512,
        IQ4_NL => iq4nl_512,
        Q2_0 => q20_512,
        Q8_0 => q80_512,
        _ => return None,
    })
}
