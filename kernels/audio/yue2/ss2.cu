// ss2.cu: SheetSage2's encoder kernels (audio/yue2/src/sheet/mert.rs) - the three operations of MERT-v2 the shared
// diffusion kernels (nsd) do not have: the depthwise convolutions along time (the ConvNeXt layers' k = 7, the conformer
// convolution module's k = 31), ConvNeXt-V2's global response norm (a reduction over the whole sequence per channel,
// then over the channels), and the convolution module's GLU. Everything else of the encoder is nsd's (the products,
// the layer norms, GELU, RoPE, the attention).
//
// Rows are channels-last [T, C] float32, as the products write them: a warp's threads read consecutive channels.
// Precise arithmetic (kernels/build.sh builds audio/ without --use_fast_math): the norm's sums in float64 (30000 rows
// of up to 512 channels at the first stage; the reference reduces in float32 in its own order).
#include "ns.h"
#include "ns_internal.hpp"
#include "yue.h"

#include <cuda_fp16.h>

#include <cmath>

namespace {

unsigned blocks(int64_t n, int t = 256) { return (unsigned) ((n + t - 1) / t); }

__global__ void dwconv_k(const float* x, int64_t T, int64_t C, const float* w, int K, const float* bias, float* out) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * C) return;
    const int64_t t = i / C, c = i % C;
    const int h = K / 2;
    float a = bias ? bias[c] : 0.0f;
    for (int j = 0; j < K; ++j) {
        const int64_t s = t + j - h;
        if (s >= 0 && s < T) a += w[c * K + j] * x[s * C + c];
    }
    out[i] = a;
}

// the squares' sum of each channel over T: a block 32 channels x 8 row lanes, float64
__global__ void colsq_k(const float* x, int64_t T, int64_t C, float* mag) {
    __shared__ double part[8][33];
    const int cx = threadIdx.x, ty = threadIdx.y;
    const int64_t c = (int64_t) blockIdx.x * 32 + cx;
    double s = 0.0;
    if (c < C)
        for (int64_t t = ty; t < T; t += 8) {
            const double v = x[t * C + c];
            s += v * v;
        }
    part[ty][cx] = s;
    __syncthreads();
    if (ty == 0 && c < C) {
        double a = 0.0;
        for (int k = 0; k < 8; ++k) a += part[k][cx];
        mag[c] = (float) sqrt(a);
    }
}

// the channels' mean of the magnitudes, into part[C]
__global__ void colmean_k(float* part, int64_t C) {
    __shared__ double s[256];
    double a = 0.0;
    for (int64_t c = threadIdx.x; c < C; c += 256) a += part[c];
    s[threadIdx.x] = a;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) {
        if ((int) threadIdx.x < o) s[threadIdx.x] += s[threadIdx.x + o];
        __syncthreads();
    }
    if (threadIdx.x == 0) part[C] = (float) (s[0] / (double) C);
}

__global__ void grn_apply_k(const float* x, int64_t T, int64_t C, const float* gamma, const float* beta, const float* part, __half* out) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * C) return;
    const int64_t c = i % C;
    const float n = part[c] / (part[C] + 1e-6f);
    const float v = x[i];
    out[i] = __float2half_rn(gamma[c] * (v * n) + beta[c] + v);
}

__global__ void glu_k(const float* x, int64_t T, int64_t C, float* out) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * C) return;
    const int64_t t = i / C, c = i % C;
    const float a = x[t * 2 * C + c], b = x[t * 2 * C + C + c];
    out[i] = a * (1.0f / (1.0f + expf(-b)));
}

}  // namespace

extern "C" {

int ns_audio_yue2_dwconv(ns_gpu* g, const float* x, int64_t T, int64_t C, const float* w, int64_t K, const float* bias, float* out) {
    if (T * C <= 0) return 0;
    if (K % 2 == 0 || K > 63) return ns_fail("ns_audio_yue2_dwconv: an odd kernel up to 63");
    NS_CU(cudaSetDevice(g->dev));
    dwconv_k<<<blocks(T * C), 256, 0, g->q>>>(x, T, C, w, (int) K, bias, out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_yue2_grn(ns_gpu* g, const float* x, int64_t T, int64_t C, const float* gamma, const float* beta, void* out, float* part) {
    if (T * C <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    colsq_k<<<(unsigned) ((C + 31) / 32), dim3(32, 8), 0, g->q>>>(x, T, C, part);
    NS_LAUNCH_CHECK();
    colmean_k<<<1, 256, 0, g->q>>>(part, C);
    NS_LAUNCH_CHECK();
    grn_apply_k<<<blocks(T * C), 256, 0, g->q>>>(x, T, C, gamma, beta, part, (__half*) out);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_yue2_glu(ns_gpu* g, const float* x, int64_t T, int64_t C, float* out) {
    if (T * C <= 0) return 0;
    NS_CU(cudaSetDevice(g->dev));
    glu_k<<<blocks(T * C), 256, 0, g->q>>>(x, T, C, out);
    NS_LAUNCH_CHECK();
    return 0;
}

}  // extern "C"
