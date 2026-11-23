// pattn.cu: the inkling2 silo's prompt attention (silo.h: ns_ink2s_attn_prompt) - flash attention on the tensor cores
// (mma.sync m16n8k16, half in, float32 sums; sm_80 and later) with Inkling's own terms: the relative bias r . relb[distance]
// added to each logit within the extent, the window, and the row's sink as a first logit of no value. A block a (32 rows,
// key head): six warps, each 16 rows of one of the three query heads that share the key head; the keys 64 at a time in
// shared memory (the values transposed, as the second product's operand wants them); the online softmax in registers.
// After nextsycl's pattn.cpp (XMX flash attention with the bias and the sink), written for NVIDIA's tensor cores.
#include "common.cuh"

using namespace ink2s;

namespace {

constexpr int D = 128, KT = 64, KP = D + 8, VP = KT + 8, G = 3, RG = 2, NW = G * RG, R_MAX = 16;

__device__ __forceinline__ uint32_t pack2(float a, float b) {
    const __half2 h = __floats2half2_rn(a, b);
    return *(const uint32_t*) &h;
}

__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__global__ void __launch_bounds__(NW * WS) pattn_k(const float* q, const __half* kc, const __half* vc, const float* r, const float* relb, int64_t E, int64_t R,
                                                   const float* sink, float* out, int64_t T, int64_t Hq, int64_t Hkv, int64_t cap, int64_t p0, int64_t win,
                                                   float scale, int kvq) {
    __shared__ __align__(16) __half Ks[KT * KP];
    __shared__ __align__(16) __half Vt[D * VP];
    __shared__ float rs[NW][16][R_MAX];
    const int tid = threadIdx.x, lane = tid % WS, w = tid / WS;
    const int g = lane >> 2, tq = lane & 3;
    const int64_t hk = blockIdx.y, h = hk * G + w % G;
    const int64_t t0 = (int64_t) blockIdx.x * (16 * RG), row0 = t0 + (w / G) * 16;
    const int64_t ra = row0 + g, rb = row0 + g + 8;           // this lane's two rows
    const bool la = ra < T, lb = rb < T;
    const int64_t pa = p0 + ra, pb = p0 + rb;
    // the block's keys: from the first row's window to the last row
    const int64_t tl = (t0 + 16 * RG < T ? t0 + 16 * RG : T) - 1;
    const int64_t lo = win > 0 ? (p0 + t0 + 1 - win > 0 ? p0 + t0 + 1 - win : 0) : 0, hi = p0 + tl + 1;
    const int64_t fa = win > 0 ? (pa + 1 - win > 0 ? pa + 1 - win : 0) : 0, fb = win > 0 ? (pb + 1 - win > 0 ? pb + 1 - win : 0) : 0;
    // q as the first product's A operand (scaled, half): 8 steps of 16 dims, 4 registers each
    uint32_t qa[8][4];
    {
        const float* qra = q + (ra * Hq + h) * D;
        const float* qrb = q + (rb * Hq + h) * D;
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
            const int c = kk * 16 + 2 * tq;
            qa[kk][0] = la ? pack2(qra[c] * scale, qra[c + 1] * scale) : 0u;
            qa[kk][1] = lb ? pack2(qrb[c] * scale, qrb[c + 1] * scale) : 0u;
            qa[kk][2] = la ? pack2(qra[c + 8] * scale, qra[c + 9] * scale) : 0u;
            qa[kk][3] = lb ? pack2(qrb[c + 8] * scale, qrb[c + 9] * scale) : 0u;
        }
    }
    // the warp's rows' bias queries
    for (int i = lane; i < 16 * R_MAX; i += WS) {
        const int rr = i / R_MAX, k = i % R_MAX;
        const int64_t row = row0 + rr;
        rs[w][rr][k] = row < T && k < R ? r[(row * Hq + h) * R + k] : 0.0f;
    }
    // the sink: the first logit, of no value
    float ma = la ? sink[ra * Hq + h] : 0.0f, mb = lb ? sink[rb * Hq + h] : 0.0f;
    float sa = 1.0f, sb = 1.0f;
    float acc[16][4];
#pragma unroll
    for (int n = 0; n < 16; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.0f;
    for (int64_t j0 = lo; j0 < hi; j0 += KT) {
        __syncthreads();
        // the tile's keys (row-major) and values (transposed); keys past the block's range zero
        for (int i = tid; i < KT * (D / 8); i += NW * WS) {
            const int jj = i / (D / 8), c8 = (i % (D / 8)) * 8;
            const int64_t j = j0 + jj;
            uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
            if (j < hi) {
                const int64_t slot = j % cap;
                if (kvq) {
                    // a q8 row: 128 int8, then 4 half scales: these 8 values of block c8 / 32
                    const uint8_t* kr = (const uint8_t*) kc + (hk * cap + slot) * 136;
                    const uint8_t* vr = (const uint8_t*) vc + (hk * cap + slot) * 136;
                    const float dk = __half2float(((const __half*) (kr + 128))[c8 / 32]), dv = __half2float(((const __half*) (vr + 128))[c8 / 32]);
                    const int8_t* kq = (const int8_t*) kr + c8;
                    const int8_t* vq = (const int8_t*) vr + c8;
                    __half2* ko = (__half2*) &kv;
                    __half2* vo = (__half2*) &vv;
#pragma unroll
                    for (int k2 = 0; k2 < 4; ++k2) {
                        ko[k2] = __floats2half2_rn(kq[2 * k2] * dk, kq[2 * k2 + 1] * dk);
                        vo[k2] = __floats2half2_rn(vq[2 * k2] * dv, vq[2 * k2 + 1] * dv);
                    }
                } else {
                    kv = *(const uint4*) (kc + (hk * cap + slot) * D + c8);
                    vv = *(const uint4*) (vc + (hk * cap + slot) * D + c8);
                }
            }
            *(uint4*) (Ks + jj * KP + c8) = kv;
            const __half* vh = (const __half*) &vv;
#pragma unroll
            for (int k = 0; k < 8; ++k) Vt[(c8 + k) * VP + jj] = vh[k];
        }
        __syncthreads();
        // the logits: 16 rows x 64 keys
        float s[8][4];
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.0f;
#pragma unroll
            for (int kk = 0; kk < 8; ++kk) {
                const __half* kr = Ks + (nt * 8 + g) * KP + kk * 16 + 2 * tq;
                mma16816(s[nt], qa[kk][0], qa[kk][1], qa[kk][2], qa[kk][3], *(const uint32_t*) kr, *(const uint32_t*) (kr + 8));
            }
        }
        // the bias, the mask, the tile's maxima
        float mxa = ma, mxb = mb;
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const bool upper = e >= 2;
                const int64_t j = j0 + nt * 8 + 2 * tq + (e & 1);
                const int64_t pos = upper ? pb : pa;
                const bool ok = (upper ? lb : la) && j < hi && j <= pos && j >= (upper ? fb : fa);
                float v = -INFINITY;
                if (ok) {
                    v = s[nt][e];
                    const int64_t d = pos - j;
                    if (d < E) {
                        const float* rr = rs[w][upper ? g + 8 : g];
                        const float* rb2 = relb + d * R;
                        for (int k = 0; k < R; ++k) v = fmaf(rr[k], rb2[k], v);
                    }
                }
                s[nt][e] = v;
                if (upper) mxb = fmaxf(mxb, v);
                else mxa = fmaxf(mxa, v);
            }
        }
        mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 1));
        mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 2));
        mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 1));
        mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 2));
        const float ca = expf(ma - mxa), cb = expf(mb - mxb);
        ma = mxa;
        mb = mxb;
        float suma = 0.0f, sumb = 0.0f;
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            s[nt][0] = expf(s[nt][0] - ma);
            s[nt][1] = expf(s[nt][1] - ma);
            s[nt][2] = expf(s[nt][2] - mb);
            s[nt][3] = expf(s[nt][3] - mb);
            suma += s[nt][0] + s[nt][1];
            sumb += s[nt][2] + s[nt][3];
        }
        suma += __shfl_xor_sync(0xffffffffu, suma, 1);
        suma += __shfl_xor_sync(0xffffffffu, suma, 2);
        sumb += __shfl_xor_sync(0xffffffffu, sumb, 1);
        sumb += __shfl_xor_sync(0xffffffffu, sumb, 2);
        sa = sa * ca + suma;
        sb = sb * cb + sumb;
#pragma unroll
        for (int n = 0; n < 16; ++n) {
            acc[n][0] *= ca;
            acc[n][1] *= ca;
            acc[n][2] *= cb;
            acc[n][3] *= cb;
        }
        // the values: P (16 x 64, the logits' layout is the A operand's) . V
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            const uint32_t a0 = pack2(s[2 * kk][0], s[2 * kk][1]), a1 = pack2(s[2 * kk][2], s[2 * kk][3]);
            const uint32_t a2 = pack2(s[2 * kk + 1][0], s[2 * kk + 1][1]), a3 = pack2(s[2 * kk + 1][2], s[2 * kk + 1][3]);
#pragma unroll
            for (int n = 0; n < 16; ++n) {
                const __half* vr = Vt + (n * 8 + g) * VP + kk * 16 + 2 * tq;
                mma16816(acc[n], a0, a1, a2, a3, *(const uint32_t*) vr, *(const uint32_t*) (vr + 8));
            }
        }
    }
    const float ia = 1.0f / sa, ib = 1.0f / sb;
#pragma unroll
    for (int n = 0; n < 16; ++n) {
        const int c = n * 8 + 2 * tq;
        if (la) *(float2*) (out + (ra * Hq + h) * D + c) = make_float2(acc[n][0] * ia, acc[n][1] * ia);
        if (lb) *(float2*) (out + (rb * Hq + h) * D + c) = make_float2(acc[n][2] * ib, acc[n][3] * ib);
    }
}

}  // namespace

extern "C" {

int ns_ink2s_attn_prompt_supported(int64_t Hq, int64_t Hkv, int64_t D_, int64_t R) {
    return Hq == G * Hkv && D_ == D && R <= R_MAX ? 1 : 0;
}

int ns_ink2s_attn_prompt(ns_gpu* g, const float* q, const void* kc, const void* vc, const float* r, const float* relb, int64_t E, int64_t R, const float* sink,
                         float* out, int64_t T, int64_t Hq, int64_t Hkv, int64_t cap, int64_t p0, int64_t win, int64_t kvq) {
    NS_TRY
    if (!ns_ink2s_attn_prompt_supported(Hq, Hkv, D, R)) return ns_fail("ns_ink2s_attn_prompt: 3 query heads a key head of 128, a bias of at most 16");
    if (T <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    const dim3 grid((unsigned) ((T + 16 * RG - 1) / (16 * RG)), (unsigned) Hkv);
    pattn_k<<<grid, NW * WS, 0, g->q>>>(q, (const __half*) kc, (const __half*) vc, r, relb, E, R, sink, out, T, Hq, Hkv, cap, p0, win, 1.0f / sqrtf((float) D), (int) kvq);
    NS_LAUNCH_CHECK();
    return 0;
    NS_CATCH
}

}  // extern "C"
