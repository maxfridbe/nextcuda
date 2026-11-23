/* vi.h: the Voice Isolator engine's own kernels (audio/voiceisolator), beside the shared diffusion kernels (nsd.h: the
 * float32 products on cuBLAS, the layer norm, GELU, the batched attention). Bound by its crate by name
 * (audio/voiceisolator/src/ffi.rs). Every buffer is float32; every call is queued on the GPU's stream (the caller
 * syncs) and returns 0, or -1 with the reason in ns_last_error().
 *
 * The network is a band-split dual-path one over tokens [seqs, L, W]: time blocks run along a band's frames, band
 * blocks along a frame's bands (the tokens transposed between them). The arithmetic follows Inworld's reference
 * (github.com/inworld-ai/voice-isolator, src/inworld/voice_isolation/model.py and scan.py; Apache-2.0), written anew
 * here. */
#ifndef NS_AUDIO_VOICEISOLATOR_H
#define NS_AUDIO_VOICEISOLATOR_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* x [A, B, W] -> out [B, A, W] */
int ns_audio_vi_swap01(ns_gpu* g, const float* x, int64_t A, int64_t B, int64_t W, float* out);

/* x[(s * L + t) * W + w] += v[s * W + w] (x [S, L, W], v [S, W]): the band embedding */
int ns_audio_vi_add_seq(ns_gpu* g, float* x, int64_t S, int64_t L, int64_t W, const float* v);

/* x[r, w] = x[r, w] * (1 + scale[w]) + shift[w] over rows [rows, W] (the decoder's conditioning) */
int ns_audio_vi_film(ns_gpu* g, float* x, int64_t rows, int64_t W, const float* scale, const float* shift);

/* Rotary positions on the queries and keys of the projection's rows (qkv [rows, 3 H D]: q at 0, k at H D), in place:
 * row r is at position r % L; pairs (i, D/2 + i) of each head rotated by the tables cs (cos) and sn (sin), [>= L, D/2]:
 * a' = a cos - b sin, b' = a sin + b cos */
int ns_audio_vi_rope(ns_gpu* g, float* qkv, int64_t rows, int64_t L, int64_t H, int64_t D, const float* cs, const float* sn);

/* Exclusive self attention: each head's output less its component along the token's own value,
 * o -= (o . v) / max(v . v, 1e-6) v. o [rows, H D]; v at the projection's rows (row stride vs) */
int ns_audio_vi_exclusive(ns_gpu* g, float* o, const float* v, int64_t vs, int64_t rows, int64_t H, int64_t D);

/* Attention within short sequences (a frame's bands): S tokens a sequence, `seqs` sequences, H heads of D; q, k, v at
 * 0, H D, 2 H D of the projection's rows qkv [seqs S, 3 H D]; softmax(q k^T / sqrt(D)) v, no mask; out [seqs S, H D] */
int ns_audio_vi_short_attention(ns_gpu* g, const float* qkv, int64_t seqs, int64_t S, int64_t H, int64_t D, float* out);

/* cat [rows, 2 W] = [x, x] (the bidirectional Mamba's residuals, its two scans added on after) */
int ns_audio_vi_dup(ns_gpu* g, const float* x, int64_t rows, int64_t W, float* cat);

/* The Mamba's depthwise convolution (K taps, a causal one: zero before the sequence) and SiLU over the first C columns
 * of the in-projection's rows xz [S L, P]: out[s, t, c] = silu(b[c] + sum_k w[c, k] u[t - (K - 1) + k]); reverse:
 * the sequence read backwards, out[t] = silu(b + sum_k w[k] u[t + (K - 1) - k]). out [S L, C] */
int ns_audio_vi_conv_silu(ns_gpu* g, const float* xz, int64_t P, int64_t S, int64_t L, int64_t C, const float* w, int64_t K, const float* b,
                          int reverse, float* out);

/* softplus in place over n values (the Mamba's step sizes): log1p(exp(x)), x above 20 as it is */
int ns_audio_vi_softplus(ns_gpu* g, float* x, int64_t n);

/* The selective scan, float32, a thread a (sequence, channel) with its N states (at most 16): h = exp(delta a) h +
 * delta u B_t, y = (C_t . h + D u) silu(z); a = -exp(a_log) [C, N]; B, C at columns bc and bc + N of xdbl (row stride
 * xs); z at column zc of the in-projection's rows (stride zs). u, delta, y [S L, C]; reverse: the sequence from its end */
int ns_audio_vi_scan(ns_gpu* g, const float* u, const float* delta, const float* a_log, const float* d, const float* xdbl, int64_t xs, int64_t bc,
                     const float* xz, int64_t zs, int64_t zc, int64_t S, int64_t L, int64_t C, int64_t N, int reverse, float* y);

#ifdef __cplusplus
}
#endif
#endif
