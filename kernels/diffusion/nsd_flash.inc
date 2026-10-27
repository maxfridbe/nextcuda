// nsd_flash.inc: attention in one fused kernel for head size 128, half in and out - FlashAttention-2's scheme on
// mma.sync m16n8k16 (the scores of 64 queries against 64 keys never leave registers, the softmax is online), the
// kernel Qwen-Image 2.1's silo carries (kernels/image/qwenimage21/silo/attn.cu), here for every engine that wants it.
// nsd_attention_qk materializes a float32 score table through cuBLAS: at 4,000 tokens and 24 heads that was a third
// of a MOSS-Image step (2026-10-26). Part of nsd.cu's translation unit (included at its end).

namespace {

constexpr int FA_D = 128;     // head size
constexpr int FA_BQ = 64;     // query rows a block (16 a warp, 4 warps)
constexpr int FA_BK = 64;     // keys a tile
constexpr int FA_LDK = 136;   // smem row stride in halves: 272 bytes, so a quad's 8 rows land on distinct banks

__device__ __forceinline__ void fa_mma16816(float c[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t fa_pack(float a, float b) {
    __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<uint32_t*>(&h);
}

__global__ void __launch_bounds__(128) fa_flash_k(const __half* __restrict__ q, const __half* __restrict__ k, const __half* __restrict__ v,
                                               __half* __restrict__ out, int Sq, int Skv, int Hq, int Hkv, long qs, long kvs, int causal,
                                               long row0, float scale_log2) {
    extern __shared__ __half smem[];
    __half* Ks = smem;
    __half* Vs = smem + FA_BK * FA_LDK;
    const int h = blockIdx.y, kvh = h / (Hq / Hkv);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int qrow0 = blockIdx.x * FA_BQ + warp * 16;
    const int r0 = qrow0 + g, r1 = r0 + 8;

    // this warp's 16 query rows as A fragments (k = the 128 features, 8 steps of 16), read once
    uint32_t qa[8][4];
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
        const int d = kk * 16 + 2 * t;
        qa[kk][0] = r0 < Sq ? *reinterpret_cast<const uint32_t*>(q + r0 * qs + (long) h * FA_D + d) : 0u;
        qa[kk][1] = r1 < Sq ? *reinterpret_cast<const uint32_t*>(q + r1 * qs + (long) h * FA_D + d) : 0u;
        qa[kk][2] = r0 < Sq ? *reinterpret_cast<const uint32_t*>(q + r0 * qs + (long) h * FA_D + d + 8) : 0u;
        qa[kk][3] = r1 < Sq ? *reinterpret_cast<const uint32_t*>(q + r1 * qs + (long) h * FA_D + d + 8) : 0u;
    }

    float o[16][4];
#pragma unroll
    for (int i = 0; i < 16; ++i) { o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f; }
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;

    int kv_end = Skv;
    if (causal) {
        const long last = row0 + (long) blockIdx.x * FA_BQ + FA_BQ;   // one past the block's last row's limit
        kv_end = (int) min((long) Skv, last);
    }
    for (int kt0 = 0; kt0 < kv_end; kt0 += FA_BK) {
        __syncthreads();
        // the tile's keys and values to shared memory, 16 bytes a thread a time
        for (int i = threadIdx.x; i < FA_BK * FA_D / 8; i += 128) {
            const int row = i / (FA_D / 8), c = (i % (FA_D / 8)) * 8;
            const int key = kt0 + row;
            uint4 kk4 = make_uint4(0, 0, 0, 0), vv4 = make_uint4(0, 0, 0, 0);
            if (key < Skv) {
                kk4 = *reinterpret_cast<const uint4*>(k + key * kvs + (long) kvh * FA_D + c);
                vv4 = *reinterpret_cast<const uint4*>(v + key * kvs + (long) kvh * FA_D + c);
            }
            *reinterpret_cast<uint4*>(&Ks[row * FA_LDK + c]) = kk4;
            *reinterpret_cast<uint4*>(&Vs[row * FA_LDK + c]) = vv4;
        }
        __syncthreads();

        // S = Q K^T for 64 keys: 8 blocks of 8 keys, k over the features
        float s[8][4];
#pragma unroll
        for (int nb = 0; nb < 8; ++nb) { s[nb][0] = s[nb][1] = s[nb][2] = s[nb][3] = 0.0f; }
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
#pragma unroll
            for (int nb = 0; nb < 8; ++nb) {
                const __half* kr = &Ks[(nb * 8 + g) * FA_LDK + kk * 16 + 2 * t];
                fa_mma16816(s[nb], qa[kk][0], qa[kk][1], qa[kk][2], qa[kk][3], *reinterpret_cast<const uint32_t*>(kr),
                         *reinterpret_cast<const uint32_t*>(kr + 8));
            }
        }
        // scale (into log2 units), mask, the block's row maxima
        float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
        for (int nb = 0; nb < 8; ++nb) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int col = kt0 + nb * 8 + 2 * t + (e & 1);
                const long row = (e >> 1) ? r1 : r0;
                float x = s[nb][e] * scale_log2;
                if (col >= Skv || (causal && col > row0 + row)) x = -INFINITY;
                s[nb][e] = x;
                if (e < 2) mx0 = fmaxf(mx0, x); else mx1 = fmaxf(mx1, x);
            }
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const float n0 = fmaxf(m0, mx0), n1 = fmaxf(m1, mx1);
        const float u0 = n0 == -INFINITY ? 0.0f : n0, u1 = n1 == -INFINITY ? 0.0f : n1;
        const float a0 = exp2f(m0 - u0), a1 = exp2f(m1 - u1);
        m0 = n0;
        m1 = n1;
        l0 *= a0;
        l1 *= a1;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            o[i][0] *= a0; o[i][1] *= a0;
            o[i][2] *= a1; o[i][3] *= a1;
        }
#pragma unroll
        for (int nb = 0; nb < 8; ++nb) {
            s[nb][0] = exp2f(s[nb][0] - u0);
            s[nb][1] = exp2f(s[nb][1] - u0);
            s[nb][2] = exp2f(s[nb][2] - u1);
            s[nb][3] = exp2f(s[nb][3] - u1);
            l0 += s[nb][0] + s[nb][1];
            l1 += s[nb][2] + s[nb][3];
        }
        // O += P V: P's accumulator layout is the A layout of the next product, 16 keys a step
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t pa0 = fa_pack(s[2 * j][0], s[2 * j][1]), pa1 = fa_pack(s[2 * j][2], s[2 * j][3]);
            const uint32_t pa2 = fa_pack(s[2 * j + 1][0], s[2 * j + 1][1]), pa3 = fa_pack(s[2 * j + 1][2], s[2 * j + 1][3]);
#pragma unroll
            for (int nd = 0; nd < 16; ++nd) {
                const int d = nd * 8 + g;
                const __half* v0 = &Vs[(16 * j + 2 * t) * FA_LDK + d];
                __half2 hb0 = __halves2half2(v0[0], v0[FA_LDK]);
                __half2 hb1 = __halves2half2(v0[8 * FA_LDK], v0[9 * FA_LDK]);
                fa_mma16816(o[nd], pa0, pa1, pa2, pa3, *reinterpret_cast<uint32_t*>(&hb0), *reinterpret_cast<uint32_t*>(&hb1));
            }
        }
    }
    // the row sums across the quad, then the output
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    const float i0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, i1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
#pragma unroll
    for (int nd = 0; nd < 16; ++nd) {
        const int d = nd * 8 + 2 * t;
        if (r0 < Sq) *reinterpret_cast<uint32_t*>(out + ((long) r0 * Hq + h) * FA_D + d) = fa_pack(o[nd][0] * i0, o[nd][1] * i0);
        if (r1 < Sq) *reinterpret_cast<uint32_t*>(out + ((long) r1 * Hq + h) * FA_D + d) = fa_pack(o[nd][2] * i1, o[nd][3] * i1);
    }
}

}  // namespace

extern "C" int nsd_flash_attention(void* ctx, const void* q, int64_t Sq, int64_t qs, const void* k, const void* v, int64_t Skv, int64_t kvs,
                                   int64_t Hq, int64_t Hkv, void* out) {
    auto* c = (Ctx*) ctx;
    CU(cudaSetDevice(c->dev));
    if (Hkv <= 0 || Hq % Hkv) return fail("nsd_flash_attention: query heads are not a multiple of the key/value heads");
    if (qs % 8 || kvs % 8) return fail("nsd_flash_attention: rows of whole 16-byte groups (strides multiples of 8)");
    const size_t smem = (size_t) 2 * FA_BK * FA_LDK * sizeof(__half);
    static int set_on = -1;
    if (set_on != c->dev) {
        CU(cudaFuncSetAttribute(fa_flash_k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem));
        set_on = c->dev;
    }
    dim3 grid((unsigned) ((Sq + FA_BQ - 1) / FA_BQ), (unsigned) Hq);
    const float scale_log2 = (1.0f / sqrtf((float) FA_D)) * 1.4426950408889634f;
    fa_flash_k<<<grid, 128, smem, c->s>>>((const __half*) q, (const __half*) k, (const __half*) v, (__half*) out, (int) Sq, (int) Skv, (int) Hq,
                                          (int) Hkv, (long) qs, (long) kvs, 0, 0L, scale_log2);
    LAUNCHED("flash attention");
    return 0;
}
