// nh_blocks.cuh: the GGUF block formats the LLM engines keep their weights in, read on the GPU - written for nextcuda
// from the formats themselves (the layouts and codebooks are data of the file format, nh_grids.cuh). Every type is read
// the same way: a 32-value piece of a block (`sub`) becomes 32 int8 values (8 words, value order, little-endian) and
// per half of 16 a scale and an offset, value = scale x int8 + offset. That one form serves the products (the int8
// against the activation's Q8_1 int8 with dp4a or the tensor cores, exact integer sums, then the scales), the
// expansions to half / float and the embedding rows - nh_iq.cu (Nemotron-H) and qwen35's q35_mmq.cu / q35_vq8.cu.
//
// To fit int8: IQ1_M's values (grid + delta, delta +-1/8 an 8) are kept as 8 x grid +- 1 with the scale / 8; Q3_K
// and Q6_K's 16-value scales are the halves' scales; the K-quants' mins, Q5_1's min and IQ1_S's delta are the
// offsets. An offset meets the activation's sum the way the format's readers have always taken it (Fmt::mins): the
// stored sum of the values (Q5_1, IQ1_S), or the sum of the quantized ones (the K-quants) - a product then differs from
// any other reader's of the same file only in float rounding.
#pragma once
#include <cuda_fp16.h>
#include <stdint.h>

#include "nh_grids.cuh"

namespace nsq {

enum : int {
    Q4_0 = 2, Q4_1 = 3, Q5_0 = 6, Q5_1 = 7, Q8_0 = 8, Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14, IQ2_XXS = 16, IQ2_XS = 17,
    IQ3_XXS = 18, IQ1_S = 19, IQ4_NL = 20, IQ3_S = 21, IQ2_S = 22, IQ4_XS = 23, IQ1_M = 29, Q2_0 = 42
};

// every type this header reads, for dispatch: X(type)
#define NSQ_TYPES(X) X(2) X(3) X(6) X(7) X(8) X(10) X(11) X(12) X(13) X(14) X(16) X(17) X(18) X(19) X(20) X(21) X(22) X(23) X(29) X(42)

// 32 values: v[8] (4 int8 each), the halves' scales and offsets
struct Sub {
    int v[8];
    float s0, s1, m0, m1;
};

// a format: values a block, bytes a block, and its offsets - 0 none; 1 one for all 32, against the activation's stored
// sum; 2 one a half, against the quantized values' sums; 3 one for all 32, against the quantized values' sum
template<int T> struct Fmt;
template<> struct Fmt<Q4_0> { static constexpr int qk = 32, bytes = 18, mins = 0; };
template<> struct Fmt<Q4_1> { static constexpr int qk = 32, bytes = 20, mins = 1; };
template<> struct Fmt<Q5_0> { static constexpr int qk = 32, bytes = 22, mins = 0; };
template<> struct Fmt<Q5_1> { static constexpr int qk = 32, bytes = 24, mins = 1; };
template<> struct Fmt<Q8_0> { static constexpr int qk = 32, bytes = 34, mins = 0; };
template<> struct Fmt<Q2_K> { static constexpr int qk = 256, bytes = 84, mins = 2; };
template<> struct Fmt<Q3_K> { static constexpr int qk = 256, bytes = 110, mins = 0; };
template<> struct Fmt<Q4_K> { static constexpr int qk = 256, bytes = 144, mins = 3; };
template<> struct Fmt<Q5_K> { static constexpr int qk = 256, bytes = 176, mins = 3; };
template<> struct Fmt<Q6_K> { static constexpr int qk = 256, bytes = 210, mins = 0; };
template<> struct Fmt<IQ2_XXS> { static constexpr int qk = 256, bytes = 66, mins = 0; };
template<> struct Fmt<IQ2_XS> { static constexpr int qk = 256, bytes = 74, mins = 0; };
template<> struct Fmt<IQ3_XXS> { static constexpr int qk = 256, bytes = 98, mins = 0; };
template<> struct Fmt<IQ1_S> { static constexpr int qk = 256, bytes = 50, mins = 1; };
template<> struct Fmt<IQ4_NL> { static constexpr int qk = 32, bytes = 18, mins = 0; };
template<> struct Fmt<IQ3_S> { static constexpr int qk = 256, bytes = 110, mins = 0; };
template<> struct Fmt<IQ2_S> { static constexpr int qk = 256, bytes = 82, mins = 0; };
template<> struct Fmt<IQ4_XS> { static constexpr int qk = 256, bytes = 136, mins = 0; };
template<> struct Fmt<IQ1_M> { static constexpr int qk = 256, bytes = 56, mins = 0; };
template<> struct Fmt<Q2_0> { static constexpr int qk = 64, bytes = 18, mins = 0; };

// whether a type's two halves of 32 can have different scales (the tile products then scale each half's sum)
template<int T> struct Halves { static constexpr bool v = T == Q2_K || T == Q3_K || T == Q6_K || T == IQ2_XS || T == IQ2_S || T == IQ1_M; };

// (values a block, bytes a block) of a type, on the host or the GPU; (0, 0) for one not here
__host__ __device__ inline int2 fmt_of(int t) {
    switch (t) {
#define NSQ_FMT(T) case T: return make_int2(Fmt<T>::qk, Fmt<T>::bytes);
        NSQ_TYPES(NSQ_FMT)
#undef NSQ_FMT
        default: return make_int2(0, 0);
    }
}

// ---- reading a block: blocks are 2-byte aligned (every size is even), so words are read as two halves

__device__ __forceinline__ uint32_t u16_at(const uint8_t* p) { return *(const uint16_t*) p; }
__device__ __forceinline__ uint32_t u32_at(const uint8_t* p) { return u16_at(p) | (u16_at(p + 2) << 16); }
__device__ __forceinline__ float half_at(const uint8_t* p) { return __half2float(*(const __half*) p); }
// 32 bytes as 8 words: two 16-byte loads where p is 16-byte aligned (Q4_K / Q5_K blocks are 144 / 176 bytes, so
// every block of an aligned row is), else two-byte loads
__device__ __forceinline__ void words8(const uint8_t* p, uint32_t (&w)[8]) {
    if (((uintptr_t) p & 15) == 0) {
        const uint4 a = ((const uint4*) p)[0], c = ((const uint4*) p)[1];
        w[0] = a.x; w[1] = a.y; w[2] = a.z; w[3] = a.w; w[4] = c.x; w[5] = c.y; w[6] = c.z; w[7] = c.w;
    } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) w[i] = u32_at(p + 4 * i);
    }
}

// 4 sign bits (bit j: value j negative) as a byte mask a value; v with those bytes negated
__device__ __forceinline__ uint32_t smask4(uint32_t b) { return ((b | b << 7 | b << 14 | b << 21) & 0x01010101u) * 0xffu; }
__device__ __forceinline__ int signed4(uint32_t v, uint32_t sbits) {
    const uint32_t m = smask4(sbits & 15);
    return (int) __vsub4(v ^ m, m);
}
// 4 bits, one into bit 0 of each byte
__device__ __forceinline__ uint32_t spread4(uint32_t b) { b &= 15; return (b | b << 7 | b << 14 | b << 21) & 0x01010101u; }
// a 7-bit sign code as its 8 bits: bit 7 makes the count of negative values even
__device__ __forceinline__ uint32_t signs7(uint32_t k) { k &= 127; return k | ((__popc(k) & 1) << 7); }

// 4 level indices (a byte each, 0-15) through IQ4_NL's levels (t: the table as 4 words)
__device__ __forceinline__ int levels4(uint32_t x, const uint32_t (&t)[4]) {
    const uint32_t sel = (x & 7) | ((x >> 4) & 0x70) | ((x >> 8) & 0x700) | ((x >> 12) & 0x7000);
    const uint32_t lo = __byte_perm(t[0], t[1], sel), hi = __byte_perm(t[2], t[3], sel);
    const uint32_t m = ((x >> 3) & 0x01010101u) * 0xffu;
    return (int) ((hi & m) | (lo & ~m));
}
// 8 level indices - the low nibbles of q's 4 bytes (x) and their high nibbles (y) - through the sixteen levels: each
// 16-bit half of q's nibbles looked up in both halves of the table at once, the right half taken by each index's bit 3
// (byte_perm uses only a selector nibble's low 3 bits), then the bytes put back in order
__device__ __forceinline__ int2 levels8(uint32_t q, const uint32_t (&t)[4]) {
    const uint32_t pick = 0x32103210u | ((q & 0x88888888u) >> 1);
    uint32_t r[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const uint32_t sel = q >> (16 * i);
        r[i] = __byte_perm(__byte_perm(t[0], t[1], sel), __byte_perm(t[2], t[3], sel), pick >> (16 * i));
    }
    return make_int2((int) __byte_perm(r[0], r[1], 0x6420), (int) __byte_perm(r[0], r[1], 0x7531));
}
__device__ __forceinline__ void iq4nl_table(uint32_t (&t)[4]) {
    const uint32_t* w = (const uint32_t*) nsq_iq4nl_values;
    t[0] = w[0]; t[1] = w[1]; t[2] = w[2]; t[3] = w[3];
}

// the K-quants' 6-bit scale and min j of 8 (Q4_K, Q5_K), from the 12 scale bytes read as 3 words (q 4-byte aligned
// in the block: at 4)
__device__ __forceinline__ void scale_min_k4(const uint8_t* q, int j, int& sc, int& m) {
    const uint32_t w0 = u32_at(q), w1 = u32_at(q + 4), w2 = u32_at(q + 8);
    const int k = (j & 3) * 8;
    if (j < 4) {
        sc = (w0 >> k) & 63;
        m = (w1 >> k) & 63;
    } else {
        sc = ((w2 >> k) & 15) | (((w0 >> k) >> 6 & 3) << 4);
        m = ((w2 >> k) >> 4 & 15) | (((w1 >> k) >> 6 & 3) << 4);
    }
}

// piece s (0 .. qk/32 - 1) of the block at b
template<int T> __device__ __forceinline__ void sub(const uint8_t* b, int s, Sub& o);

template<> __device__ __forceinline__ void sub<Q8_0>(const uint8_t* b, int, Sub& o) {
    o.s0 = o.s1 = half_at(b);
    o.m0 = o.m1 = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) o.v[i] = (int) u32_at(b + 2 + 4 * i);
}

// Q4_0 / Q4_1: 16 bytes of two nibbles (the low ones the first 16 values); Q4_0's minus 8, Q4_1's with a min
template<bool SIGNED> __device__ __forceinline__ void q4_values(const uint8_t* qs, Sub& o) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t w = u32_at(qs + 4 * i);
        const uint32_t lo = w & 0x0f0f0f0fu, hi = (w >> 4) & 0x0f0f0f0fu;
        o.v[i] = (int) (SIGNED ? __vsub4(lo, 0x08080808u) : lo);
        o.v[i + 4] = (int) (SIGNED ? __vsub4(hi, 0x08080808u) : hi);
    }
}
template<> __device__ __forceinline__ void sub<Q4_0>(const uint8_t* b, int, Sub& o) {
    o.s0 = o.s1 = half_at(b);
    o.m0 = o.m1 = 0.f;
    q4_values<true>(b + 2, o);
}
template<> __device__ __forceinline__ void sub<Q4_1>(const uint8_t* b, int, Sub& o) {
    o.s0 = o.s1 = half_at(b);
    o.m0 = o.m1 = half_at(b + 2);
    q4_values<false>(b + 4, o);
}

// Q5_0 / Q5_1: 16 bytes of two nibbles (the low ones the first 16 values) and a word of fifth bits
template<bool SIGNED> __device__ __forceinline__ void q5_values(const uint8_t* qs, uint32_t qh, Sub& o) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t w = u32_at(qs + 4 * i);
        const uint32_t lo = (w & 0x0f0f0f0fu) | (spread4(qh >> (4 * i)) << 4);
        const uint32_t hi = ((w >> 4) & 0x0f0f0f0fu) | (spread4(qh >> (16 + 4 * i)) << 4);
        o.v[i] = (int) (SIGNED ? __vsub4(lo, 0x10101010u) : lo);
        o.v[i + 4] = (int) (SIGNED ? __vsub4(hi, 0x10101010u) : hi);
    }
}
template<> __device__ __forceinline__ void sub<Q5_0>(const uint8_t* b, int, Sub& o) {
    o.s0 = o.s1 = half_at(b);
    o.m0 = o.m1 = 0.f;
    q5_values<true>(b + 6, u32_at(b + 2), o);
}
template<> __device__ __forceinline__ void sub<Q5_1>(const uint8_t* b, int, Sub& o) {
    o.s0 = o.s1 = half_at(b);
    o.m0 = o.m1 = half_at(b + 2);
    q5_values<false>(b + 8, u32_at(b + 4), o);
}

// Q2_0: 64 values of 2 bits (four a byte, the lowest first) standing for -1 .. 2
template<> __device__ __forceinline__ void sub<Q2_0>(const uint8_t* b, int s, Sub& o) {
    o.s0 = o.s1 = half_at(b);
    o.m0 = o.m1 = 0.f;
    const uint32_t q[2] = {u32_at(b + 2 + 8 * s), u32_at(b + 6 + 8 * s)};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const uint32_t x = (q[i / 4] >> (8 * (i & 3))) & 255;
        o.v[i] = (int) __vsub4((x | x << 6 | x << 12 | x << 18) & 0x03030303u, 0x01010101u);
    }
}

template<> __device__ __forceinline__ void sub<IQ4_NL>(const uint8_t* b, int, Sub& o) {
    uint32_t t[4];
    iq4nl_table(t);
    o.s0 = o.s1 = half_at(b);
    o.m0 = o.m1 = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int2 v = levels8(u32_at(b + 2 + 4 * i), t);
        o.v[i] = v.x;
        o.v[i + 4] = v.y;
    }
}

// IQ4_XS: half d, 16 bits of scales' high parts, 4 bytes of their low nibbles, then 32 a piece as IQ4_NL's
template<> __device__ __forceinline__ void sub<IQ4_XS>(const uint8_t* b, int s, Sub& o) {
    uint32_t t[4];
    iq4nl_table(t);
    const int ls = ((b[4 + s / 2] >> (4 * (s & 1))) & 15) | (((u16_at(b + 2) >> (2 * s)) & 3) << 4);
    o.s0 = o.s1 = half_at(b) * (float) (ls - 32);
    o.m0 = o.m1 = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int2 v = levels8(u32_at(b + 8 + 16 * s + 4 * i), t);
        o.v[i] = v.x;
        o.v[i + 4] = v.y;
    }
}

// Q2_K: 16 bytes of scales (low nibble) and mins (high), 64 of 2-bit values, half d, half dmin
template<> __device__ __forceinline__ void sub<Q2_K>(const uint8_t* b, int s, Sub& o) {
    const int n = s / 4, j = s % 4;
    const float d = half_at(b + 80), dmin = half_at(b + 82);
    const int sc0 = b[8 * n + 2 * j], sc1 = b[8 * n + 2 * j + 1];
    o.s0 = d * (float) (sc0 & 15);
    o.s1 = d * (float) (sc1 & 15);
    o.m0 = -dmin * (float) (sc0 >> 4);
    o.m1 = -dmin * (float) (sc1 >> 4);
#pragma unroll
    for (int i = 0; i < 8; ++i) o.v[i] = (int) ((u32_at(b + 16 + 32 * n + 4 * i) >> (2 * j)) & 0x03030303u);
}

// Q3_K: 32 bytes of high bits, 64 of 2-bit values, 12 of 6-bit scales (minus 32), half d
template<> __device__ __forceinline__ void sub<Q3_K>(const uint8_t* b, int s, Sub& o) {
    const int n = s / 4, j = s % 4;
    const float d = half_at(b + 108);
    const uint8_t* sc = b + 96;
    auto scale = [&](int k) {
        const int lo = k < 8 ? (sc[k] & 15) : (sc[k - 8] >> 4);
        const int hi = (sc[8 + (k & 3)] >> (2 * (k >> 2))) & 3;
        return (float) ((lo | (hi << 4)) - 32);
    };
    o.s0 = d * scale(8 * n + 2 * j);
    o.s1 = d * scale(8 * n + 2 * j + 1);
    o.m0 = o.m1 = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const uint32_t q = (u32_at(b + 32 + 32 * n + 4 * i) >> (2 * j)) & 0x03030303u;
        const uint32_t h = (u32_at(b + 4 * i) >> (4 * n + j)) & 0x01010101u;
        o.v[i] = (int) __vsub4(q | (h << 2), 0x04040404u);
    }
}

// Q4_K / Q5_K: half d, half dmin, 12 bytes of 6-bit scales and mins, (Q5_K: 32 bytes of fifth bits,) 128 of nibbles
template<> __device__ __forceinline__ void sub<Q4_K>(const uint8_t* b, int s, Sub& o) {
    int sc, m;
    scale_min_k4(b + 4, s, sc, m);
    o.s0 = o.s1 = half_at(b) * (float) sc;
    o.m0 = o.m1 = -half_at(b + 2) * (float) m;
    uint32_t q[8];
    words8(b + 16 + 32 * (s / 2), q);
#pragma unroll
    for (int i = 0; i < 8; ++i) o.v[i] = (int) ((q[i] >> (4 * (s & 1))) & 0x0f0f0f0fu);
}
template<> __device__ __forceinline__ void sub<Q5_K>(const uint8_t* b, int s, Sub& o) {
    int sc, m;
    scale_min_k4(b + 4, s, sc, m);
    o.s0 = o.s1 = half_at(b) * (float) sc;
    o.m0 = o.m1 = -half_at(b + 2) * (float) m;
    uint32_t ql[8], qh[8];
    words8(b + 48 + 32 * (s / 2), ql);
    words8(b + 16, qh);
#pragma unroll
    for (int i = 0; i < 8; ++i) o.v[i] = (int) (((ql[i] >> (4 * (s & 1))) & 0x0f0f0f0fu) | (((qh[i] >> s) & 0x01010101u) << 4));
}

// Q6_K: 128 bytes of low nibbles, 64 of 2-bit high parts, 16 int8 scales (one a 16), half d
template<> __device__ __forceinline__ void sub<Q6_K>(const uint8_t* b, int s, Sub& o) {
    const int n = s / 4, k = s % 4;
    const float d = half_at(b + 208);
    o.s0 = d * (float) (int8_t) b[192 + 8 * n + 2 * k];
    o.s1 = d * (float) (int8_t) b[192 + 8 * n + 2 * k + 1];
    o.m0 = o.m1 = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const uint32_t lo = (u32_at(b + 64 * n + 32 * (k & 1) + 4 * i) >> (4 * (k >> 1))) & 0x0f0f0f0fu;
        const uint32_t hi = (u32_at(b + 128 + 32 * n + 4 * i) >> (2 * k)) & 0x03030303u;
        o.v[i] = (int) __vsub4(lo | (hi << 4), 0x20202020u);
    }
}

// 8 grid magnitudes (a u64 entry) with 8 sign bits, into two words
__device__ __forceinline__ void grid8(uint64_t g, uint32_t sbits, int& a, int& c) {
    a = signed4((uint32_t) g, sbits);
    c = signed4((uint32_t) (g >> 32), sbits >> 4);
}

// IQ2_XXS: half d, then a piece: 4 grid indices (bytes) and a word of 4 x 7 sign bits and a 4-bit scale
template<> __device__ __forceinline__ void sub<IQ2_XXS>(const uint8_t* b, int s, Sub& o) {
    const uint32_t idx = u32_at(b + 2 + 8 * s), aux = u32_at(b + 6 + 8 * s);
    o.s0 = o.s1 = half_at(b) * (0.5f + (float) (aux >> 28)) * 0.25f;
    o.m0 = o.m1 = 0.f;
#pragma unroll
    for (int l = 0; l < 4; ++l) grid8(nsq_iq2xxs_grid[(idx >> (8 * l)) & 255], signs7(aux >> (7 * l)), o.v[2 * l], o.v[2 * l + 1]);
}

// IQ2_XS: half d, 32 words (9-bit grid index, 7-bit sign code), 8 bytes of two 4-bit scales
template<> __device__ __forceinline__ void sub<IQ2_XS>(const uint8_t* b, int s, Sub& o) {
    const float d = half_at(b);
    const int sc = b[66 + s];
    o.s0 = d * (0.5f + (float) (sc & 15)) * 0.25f;
    o.s1 = d * (0.5f + (float) (sc >> 4)) * 0.25f;
    o.m0 = o.m1 = 0.f;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const uint32_t q = u16_at(b + 2 + 8 * s + 2 * l);
        grid8(nsq_iq2xs_grid[q & 511], signs7(q >> 9), o.v[2 * l], o.v[2 * l + 1]);
    }
}

// IQ2_S: half d, 32 bytes of grid indices' low parts, 32 of signs, 8 of their high bits, 8 of two 4-bit scales
template<> __device__ __forceinline__ void sub<IQ2_S>(const uint8_t* b, int s, Sub& o) {
    const float d = half_at(b);
    const int sc = b[74 + s], qh = b[66 + s];
    o.s0 = d * (0.5f + (float) (sc & 15)) * 0.25f;
    o.s1 = d * (0.5f + (float) (sc >> 4)) * 0.25f;
    o.m0 = o.m1 = 0.f;
    const uint32_t qs = u32_at(b + 2 + 4 * s), sg = u32_at(b + 34 + 4 * s);
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const int i = ((qs >> (8 * l)) & 255) | ((qh << (8 - 2 * l)) & 0x300);
        grid8(nsq_iq2s_grid[i], sg >> (8 * l), o.v[2 * l], o.v[2 * l + 1]);
    }
}

// IQ3_XXS: half d, 64 grid indices (4 values each), then a word a piece: 4 x 7 sign bits and a 4-bit scale
template<> __device__ __forceinline__ void sub<IQ3_XXS>(const uint8_t* b, int s, Sub& o) {
    const uint32_t aux = u32_at(b + 66 + 4 * s);
    o.s0 = o.s1 = half_at(b) * (0.5f + (float) (aux >> 28)) * 0.5f;
    o.m0 = o.m1 = 0.f;
    const uint32_t q[2] = {u32_at(b + 2 + 8 * s), u32_at(b + 6 + 8 * s)};
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const uint32_t sg = signs7(aux >> (7 * l));
        const uint32_t w = q[l / 2] >> (16 * (l & 1));
        o.v[2 * l] = signed4(nsq_iq3xxs_grid[w & 255], sg);
        o.v[2 * l + 1] = signed4(nsq_iq3xxs_grid[(w >> 8) & 255], sg >> 4);
    }
}

// IQ3_S: half d, 64 bytes of grid indices' low parts, 8 of their ninth bits, 32 of signs, 4 of two 4-bit scales
template<> __device__ __forceinline__ void sub<IQ3_S>(const uint8_t* b, int s, Sub& o) {
    const int qh = b[66 + s];
    o.s0 = o.s1 = half_at(b) * (float) (1 + 2 * ((b[106 + s / 2] >> (4 * (s & 1))) & 15));
    o.m0 = o.m1 = 0.f;
    const uint32_t q[2] = {u32_at(b + 2 + 8 * s), u32_at(b + 6 + 8 * s)}, signs = u32_at(b + 74 + 4 * s);
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const uint32_t sg = signs >> (8 * l);
        const uint32_t w = q[l / 2] >> (16 * (l & 1));
        const int i1 = (w & 255) | ((qh << (8 - 2 * l)) & 256);
        const int i2 = ((w >> 8) & 255) | ((qh << (7 - 2 * l)) & 256);
        o.v[2 * l] = signed4(nsq_iq3s_grid[i1], sg);
        o.v[2 * l + 1] = signed4(nsq_iq3s_grid[i2], sg >> 4);
    }
}

// 8 values of an IQ1 grid entry (nibbles of value + 1) as 8 x value + delta, delta +-1
__device__ __forceinline__ void iq1_grid(uint32_t e, bool neg, int& a, int& c) {
    const uint32_t k = neg ? 0x09090909u : 0x07070707u;   // 8 - delta
    a = (int) __vsub4((e & 0x0f0f0f0fu) << 3, k);
    c = (int) __vsub4(((e >> 4) & 0x0f0f0f0fu) << 3, k);
}

// IQ1_S: half d, 32 bytes of grid indices' low parts, then a 16-bit word a piece: 4 x 3 high bits, a 3-bit scale and
// the delta's sign; the values the grid's entries + 1 (0 / 1 / 2), the offset delta - 1
template<> __device__ __forceinline__ void sub<IQ1_S>(const uint8_t* b, int s, Sub& o) {
    const uint32_t qh = u16_at(b + 34 + 2 * s);
    o.s0 = o.s1 = half_at(b) * (float) (2 * ((qh >> 12) & 7) + 1);
    o.m0 = o.m1 = o.s0 * ((qh & 0x8000) ? -1.125f : -0.875f);
    const uint32_t qs = u32_at(b + 2 + 4 * s);
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const uint32_t e = nsq_iq1s_grid[((qs >> (8 * l)) & 255) | (((qh >> (3 * l)) & 7) << 8)];
        o.v[2 * l] = (int) (e & 0x0f0f0f0fu);
        o.v[2 * l + 1] = (int) ((e >> 4) & 0x0f0f0f0fu);
    }
}

// IQ1_M: 32 bytes of grid indices' low parts, 16 of their high bits and the deltas' signs (a nibble per 8 values), 8 of
// 3-bit scales (one a 16) whose top nibbles together are the block's half d
template<> __device__ __forceinline__ void sub<IQ1_M>(const uint8_t* b, int s, Sub& o) {
    const uint32_t sc0 = u16_at(b + 48), sc1 = u16_at(b + 50), sc2 = u16_at(b + 52), sc3 = u16_at(b + 54);
    const __half_raw hr = {(unsigned short) ((sc0 >> 12) | ((sc1 >> 8) & 0x00f0) | ((sc2 >> 4) & 0x0f00) | (sc3 & 0xf000))};
    const float d = __half2float(__half(hr));
    const uint32_t sc = u16_at(b + 48 + 2 * (s / 2));
    o.s0 = d * (float) (2 * ((sc >> (6 * (s & 1))) & 7) + 1) * 0.125f;
    o.s1 = d * (float) (2 * ((sc >> (6 * (s & 1) + 3)) & 7) + 1) * 0.125f;
    o.m0 = o.m1 = 0.f;
    const uint32_t qs = u32_at(b + 4 * s), qh = u16_at(b + 32 + 2 * s);
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const uint32_t h = (qh >> (8 * (l / 2))) & 255;
        const int i = ((qs >> (8 * l)) & 255) | ((h << (8 - 4 * (l & 1))) & 0x700);
        iq1_grid(nsq_iq1s_grid[i], h & ((l & 1) ? 0x80 : 0x08), o.v[2 * l], o.v[2 * l + 1]);
    }
}

// ---- the activation: Q8_1, 32 values as int8, a half scale and the half sum of the values it stands for
struct Q8_1 {
    __half2 ds;
    int8_t qs[32];
};
static_assert(sizeof(Q8_1) == 36, "Q8_1 is 36 bytes");

// piece p of a row of `t` against one Q8_1 block of the activation
template<int T> __device__ __forceinline__ float dot(const Sub& w, const Q8_1& y) {
    const int* yq = (const int*) y.qs;
    int i0 = 0, i1 = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        i0 = __dp4a(w.v[i], yq[i], i0);
        i1 = __dp4a(w.v[i + 4], yq[i + 4], i1);
    }
    const float2 ds = __half22float2(y.ds);
    float r = ds.x * (w.s0 * (float) i0 + w.s1 * (float) i1);
    if (Fmt<T>::mins == 1) r += w.m0 * ds.y;
    if (Fmt<T>::mins >= 2) {
        int a = 0, c = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            a = __dp4a(0x01010101, yq[i], a);
            c = __dp4a(0x01010101, yq[i + 4], c);
        }
        r += Fmt<T>::mins == 2 ? ds.x * (w.m0 * (float) a + w.m1 * (float) c) : ds.x * w.m0 * (float) (a + c);
    }
    return r;
}

}  // namespace nsq
