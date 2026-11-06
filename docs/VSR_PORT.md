# Video super-resolution: SeedVR2 and FlashVSR on nextcuda

The platform upscales finished films with `video.upscale`. Its only models so far are the per-frame Real-ESRGAN and
waifu2x. Their anime models make real faces waxy, and realesrgan-x4plus takes about 6 s a frame on brainiac-nvidia
(5,665 frames: more than nine hours). Two one-step diffusion VSR models restore whole clips, with temporal context.
This page covers what they need on two 12 GB cards (the RTX 5070 and the RTX 3060, 32 GB of RAM), and what the
port does. The rule holds: no Python, PyTorch or ONNX at run time, and no weights streamed from host memory or disk.

The target is x2 on the platform's 864 x 480 films (1728 x 960 out), 5,000-6,000 frames long.

## Feasibility

| | FlashVSR v1.1 (tiny decoder) | SeedVR2-3B | SeedVR2-7B |
|---|---|---|---|
| what it is | Wan 2.1 T2V-1.3B distilled to one step, streaming: 8 frames a step with a KV cache of the last 3 latent pairs; a causal LQ projection in place of the VAE encoder; the "tiny conditional decoder" (TAEHV-like, 2D convs with frame memory) in place of the VAE decoder | NaDiT (MM-DiT, 32 layers, 2,560 wide, 3D window attention), one step from noise + the LQ latent; the SeedVR causal 3D VAE (16 channels, x8 space, x4 time) both ways | the same, 7B; also a "sharp" checkpoint |
| weights | DiT 1.36B (5.7 GB fp32 -> 2.8 GB fp16); LQ projection 288M (0.58 GB bf16); decoder 47M (0.19 GB fp32) | DiT 3.39B: 13.6 GB fp32, 6.8 GB fp16, 3.4 GB fp8 (numz/SeedVR2_comfyUI); VAE 0.5 GB fp16 | DiT 8.24B: 16.5 GB fp16, 8.2 GB fp8 |
| licence | Apache-2.0 (JunhaoZhuang/FlashVSR-v1.1; code OpenImagingLab/FlashVSR Apache-2.0; base Wan 2.1 Apache-2.0): **commercial yes** | Apache-2.0 (ByteDance-Seed/SeedVR2-3B, code ByteDance-Seed/SeedVR): **commercial yes** | Apache-2.0: **commercial yes** |
| 1728 x 960 working size | padded to 1792 x 1024 (128-pixel windows); 14,336 tokens per latent pair | 108 x 60 patches a latent frame (6,480 tokens); windows of about 2,900 tokens | the same |
| memory at 1792 x 1024 | weights 2.8 GB + KV cache 7.9 GB (30 layers x 3 pairs x 14,336 x 1,536 x K, V in fp16) + about 1.5 GB working: **more than one 12 GB card; fits split over both** (layers 0-17 + the LQ projection on the 5070 about 8.5 GB, layers 18-29 + the decoder on the 3060 about 5.5 GB). A frame up to about 1280 x 768 fits on one card | weights 6.8 GB fp16 + VAE 0.5 GB + attention work about 1-2 GB in windowed attention; the VAE at full resolution needs spatial tiles (the ComfyUI node's 512-pixel tiles): **fits one card** | fp16 16.5 GB: **only split over both cards** (8.2 GB each, the VAE beside the smaller half); fp8 8.2 GB fits on the 5070 alone (FP8 MMA there; the 3060 has none). Tight |
| compute per frame (estimate) | DiT about 1e13 FLOP, decoder about 4e12, LQ projection about 1.2e12 | DiT about 1.1e13, VAE encode + decode at full resolution about 4e13 (3x3x3 convs at 128-512 channels) | DiT about 2.6e13 + the same VAE |
| speed on the pair | **measured: 0.43 s a frame** (a film of 5,665 frames in about 41 min; estimated 0.3-0.4) | about 1-1.5 s a frame (about 2 hours a film) | about 2-3 s a frame |
| temporal handling | native: a causal stream with a rolling KV cache and causal convolutions, so a film runs as one stream with no seams; RoPE positions are relative, so it has no length limit | clips of 4n + 1 frames processed separately; long films need overlapping chunks blended in the overlap (the ComfyUI node's `temporal_overlap`) | the same |
| porting work | the Wan DiT blocks (linear, norms, RoPE, cross-attention to a fixed prompt), block-sparse attention with its top-k block mask, the LQ projection (causal 3D convs), the tiny decoder (2D convs): **small** | NaDiT with MM layers and shifted 3D windows, plus the full causal 3D VAE both ways: **large** | the same as 3B |

## Decision

**FlashVSR goes first.** Both fit on the two cards without streaming weights. FlashVSR is the lighter of the two, by
a lot. Its weights are 2.8 GB against 6.8 GB, it costs about a third of SeedVR2-3B's compute per frame, and it needs
no VAE encoder or full VAE decoder. Its streaming form also answers the long-film requirement natively: there are no
chunk seams to blend. Its one cost is the KV cache, which at 1792 x 1024 is larger than its weights. The engine splits
the 30 layers over the two cards, pipelined, so neither card holds more than about 8.5 GB. Below about 1280 x 768 it
runs on one card.

SeedVR2-3B is second. It fits one card in fp16 with a tiled VAE. SeedVR2-7B fits only split over both cards, or as fp8
on the 5070 alone. Neither streams weights.

## The names (the same on nextsycl, hardac1)

- `"model": "flashvsr"`, `"variant": "flashvsr-v1.1"`, `"scale": 2` (2 to 4)
- `"model": "seedvr2"`, `"variant": "seedvr2-3b" | "seedvr2-7b"`, `"scale": 2`

The request and answer shapes are realesrgan's `video.upscale` (`file` or `video`, `crf`, `response_format`). See
the port's section below for the options it adds.

## The port: FlashVSR v1.1 (done)

`image/flashvsr` (Rust), `kernels/image/flashvsr/fv.cu` (its kernels), `reference/flashvsr/ref.py` (the check's
reference), `model-cards/image/flashvsr.yml`. It is an image-kind engine, like the upscalers. A new contract method,
`ImageEngine::upscale_clip`, lets `video.upscale` give it the whole clip, streamed frame by frame, instead of each
frame to `upscale`.

- **What runs:** the official tiny pipeline (`flashvsr_tiny.py` with v1.1's `Causal_LQ4x_Proj`). Each frame is upscaled
  by Pillow's bicubic (done arithmetic for arithmetic in Rust) and mirrored out to multiples of 128. The official
  script centre-crops to them instead, which would cut 32 pixels off every side of 1728 x 960. Then come the LQ
  projection (causal conv3d in im2col tiles + cuBLAS), noise, and the Wan DiT in half with fp32 residuals. Its
  attention is FlashVSR's own: block means score the key blocks, the softmax over an 11 x 11 local window keeps the
  top k across each query group (a radix select), and a FlashAttention-2 kernel walks each query block's list of 128-key
  blocks. Next are the KV cache of three latent pairs and one step from noise at t = 1000. Last, the tiny conditional
  decoder (im2col + cuBLAS; its nearest upsample read in place; a direct kernel for the 128 -> 3 output conv) and the
  AdaIN colour fix against the upscaled input.
- **Long films:** the stream is the temporal scheme. Each 8-frame chunk attends to the 24 frames before it, and the
  convolutions keep their last frames, so there are no chunk seams to blend. RoPE positions are relative, so the
  stream has no length limit; the reference itself indexes a 1,024-position table and stops after about 4,000
  frames. The clip's last frame is repeated 4-11 times past its end, as the reference pads, so every frame comes out.
- **Checked** against the reference (float32, CPU), 33 frames at 1024 x 768. Each stage's relative error: the LQ
  projection 3.8e-4, block 0 2e-4, block 15 2.3e-3, block 29 1.9e-3, the latents 1.8e-3 (cos 0.999998), the decoded
  frames 9e-4. The bytes are 64.1 dB PSNR against the reference's. The method: `nextcuda image check DIR --model flashvsr`.
- **Measured** on the platform film (864 x 480, 24 fps), x2, 240 frames: 0.43 s a frame, the two cards pipelined (the
  3060 runs chunk k's last 18 blocks while the 5070 decodes chunk k - 1 and starts chunk k + 1). Peak memory
  (nvidia-smi, the whole card): 10.5 GiB on the 5070 (9.4 ours), 7.9 GiB on the 3060. Before tuning it was 1.08 s a
  frame; the profile (`NC_FLASHVSR_PROFILE=1`) showed the decoder at 64%, on the slower card.
- **Quality:** see `docs/screenshots/vsr-face-{60,120,200}.jpg`, crops of the singer's face at 1:1, bicubic x2 against
  realesrgan-x4plus at x4 brought down to x2 (as the platform does) and FlashVSR x2. Real-ESRGAN paints the skin
  waxy and smears the lashes and the microphone's mesh. FlashVSR keeps skin texture, teeth, single hairs and the mesh,
  and stays steady over the clip.

### The API

```json
{"model": "flashvsr", "variant": "flashvsr-v1.1", "scale": 2, "video": "data:video/mp4;base64,...", "crf": 16}
{"model": "flashvsr", "variant": "flashvsr-v1.1", "scale": 2, "file": "h3_20261019_121939"}
```

`video.upscale`: the request and the answer are Real-ESRGAN's (`video` or `file`, `crf`, `response_format`; the answer's
`data[0]` has `url`, `frames`, `fps`, `width`, `height`, `variant`, `scale`). Its own keys, all optional, go at the top
level or in `options`:

| key | default | |
|---|---|---|
| `scale` | 2 | 2, 3 or 4 |
| `seed` | 0 | the noise of the one step (it changes fine detail only) |
| `sparse_ratio` | 2.0 | the attention's top k (FlashVSR's own knob: 1.5 faster, 2.0 steadier) |
| `local_range` | 11 | the local window in 128-pixel blocks (9 sharper, 11 steadier) |
| `color_fix` | 1 | each frame's colour matched to the upscaled input's (AdaIN) |

`image.upscale` takes the same `model` / `variant` / `scale` and runs the picture as a still clip (about 16 s a
picture: it is a video model).

With no `model`, a `flashvsr-...` variant picks the flashvsr card. The platform's client sends `model` from its
CP_REMOTE_MODELS map, so its catalogue entry should send `"model": "flashvsr"` with `"variant": "flashvsr-v1.1"`.

## SeedVR2: what is done, what a port needs

Done here: the feasibility above, the weights (`~/nextcuda/models/seedvr2-3b` on brainiac-nvidia: ByteDance-Seed/SeedVR2-3B
@ 37255ff8, `seedvr2_ema_3b.pth` 13.6 GB float32, `ema_vae.pth`, `pos_emb.pt`; Apache-2.0, commercial yes), and the
reference: `reference/seedvr2/ref.py` runs the official code (configs_3b) on the CPU in float32 (apex and flash-attn
stubbed with plain torch) and dumps every stage - the prepared frames, the VAE's latents (mean and sample), the
noise, the timestep embedding, the video tokens after blocks 0, 9, 10 and 31, the DiT's output, x0 and the decoded
frames. A run is at brainiac-nvidia:~/nextcuda/ref/sv/d5 (5 frames of the platform film's singer, 256 x 192 -> 512 x 384;
33 minutes on the CPU), the matching reference for FlashVSR at ~/nextcuda/ref/fv/d33. The engine itself is not written yet. What it is, read from the code (the facts a port needs):

- **Input:** torchvision bicubic (a = -0.75, float) to scale x the frame, clamped to [0, 1], cropped to multiples of 16,
  to [-1, 1]; the clip padded with its last frame to 4n + 1.
- **VAE** (video_vae_v3, SD3-style, 16 latent channels, x8 space, x4 time, scaling 0.9152, no shift): causal
  3-D convs (time padding = the first frame repeated, or the previous slice's last frames: the "memory"), GroupNorm
  per frame, a mid block with single-head spatial attention per frame (512 channels), the encoder's down blocks
  with temporal down in two of them, the decoder's up blocks with an upscale 1x1 conv + pixel shuffle (+ the first
  frame's duplicate removed on temporal up). Every part is per-frame or causal: run in slices of 4 frames with
  memory (as the official `slicing` does), it streams a film of any length exactly, in both directions.
- **DiT** (NaDiT 3B): 32 blocks of 2,560 (20 heads x 128), the first 10 with separate video / text weights (MM), the
  rest shared, the last updating only the video. In: 1x2x2 patches of [noise 16, LQ latent 16, mask 1] = 132 -> 2,560;
  text: pos_emb [L, 5120] -> linear. AdaSingle modulation (one shared time embedding 15,360 = 6 x 2,560, plus each
  layer's own shift / scale / gate). Attention: windows over the patch grid sized as for 720p (3 x 3 spatial windows
  at 1280 x 720, scaled to the frame's area; time windows of ceil(min(t, 30) / 4) latent frames), shifted by half a
  window on odd layers. Each window attends jointly with the whole text; text outputs are averaged over the windows.
  RoPE "mmrope3d": video (t + L, h, w) local to the window, text (i, i, i), 42 dims an axis, interleaved pairs; q / k
  RMS-normed per head. SwiGLU MLP (6,912 hidden). Out: RMS norm + modulation, linear 2,560 -> 64.
- **Sampling:** one step at t = 1000 (the shift transform leaves it there): x0 = noise - v. cfg 1 (the negative prompt
  unused).
- **On the two cards:** the DiT 6.8 GB in half on one card, the VAE beside it (0.5 GB) or on the other. At 1728 x 960 the
  VAE needs spatial tiles or streaming by rows (a 128-channel 3x3x3 conv's im2col of a full frame is 11 GB); 1.0-1.5 s a
  frame expected.
- **Long films:** the VAE streams exactly; the DiT sees a chunk of latent frames at once (the official code does the whole
  clip in one pass, memory growing with it). The plan: encode the film as a stream, run the DiT on overlapping chunks
  of latent frames (e.g. 8 latent frames = 29 frames, overlap 2), blend x0 linearly across each overlap, then decode
  the blended latents as a stream. No seams in the decoder; the overlap carries the DiT's context across chunks.

hardac1 (nextsycl) is porting SeedVR2 now with the same names. Its engine is Rust over the same nsd ABI (nextcuda's
nsd.h is nextsycl's H3 interface), so taking that engine over, checked against these dumps, is probably cheaper than a
second port from scratch.

## FlashVSR performance (2026-11-05)

Measured on the platform's film (864 x 480, 24 fps, from 55 s in), in process (`nextcuda image upscale-clip`, the same
engine path as `video.upscale`), under the GPU lock with nothing else on the cards. A run's utilisation is nvidia-smi
sampled every 0.5 s (`util_*.log` beside the outputs in brainiac-nvidia:~/nextcuda-vsr/out). The numerics stayed at
**64.1 dB** against the official pipeline after every change (`nextcuda image check`, sparse_ratio 2.0 as the
reference ran; worst relative error 2.3e-3).

### Where the time went (x2, before tuning; `NC_FLASHVSR_PROFILE=1`, each stage waited for)

| stage | card | s a frame | |
|---|---|---:|---|
| frames: ffmpeg decode wait | host | 0.002 | |
| frames: Pillow bicubic + mirror | host | 0.021 | on the first card's thread |
| LQ projection | 5070 | 0.027 | |
| DiT blocks 0-11 | 5070 | 0.088 | 0.0074 a block |
| x to the next card | both | 0.008 | |
| DiT blocks 12-29 + head | 3060 | 0.326 | 0.0183 a block |
| decoder (im2col + cuBLAS) | 5070 | **0.269** | the biggest stage |
| colour fix + bytes + readback | 5070 | 0.001 | |
| encoder pipe | host | 0.004-0.021 | |

Inside a DiT block (either card): the sparse attention 51%, the MLP 27%, q/k/v + norms + RoPE 9%, the cross-attention
8%, o + the KV cache 4%, the draft (block scores, top k, lists) 1%. The attention kernel runs at about 18 TFLOPS on
the 3060 and 44 on the 5070, and the MLP's cuBLAS at about 20 on the 3060: 70-77% of the cards' fp16 tensor rate with
float32 accumulation. The blocks are compute-bound; half-precision accumulation would double that rate on these
GeForce cards, but it moves the numerics, so it is not used.

### The changes, each measured (x2, pipelined, 240 frames unless said)

| change | s a frame | card 0 / card 1 busy |
|---|---:|---|
| deployed port (a873a37): decoder on the 5070, split 12 / 18, the cards pipelined | 0.432 | 82% / 71% |
| decoder's 3 x 3 convs as an implicit GEMM (tensor cores, taps gathered in place, upsample and MemBlock past read directly, bias / ReLU / residual fused): decoder 0.269 -> 0.082 | 0.339 | 46% / 89% |
| split 16 / 14 (balance the cards) | 0.282 | 74% / 80% |
| frames prepared on threads a chunk ahead; ffmpeg read and the encoder fed on threads of their own | 0.274 | 68% / 83% |
| the DiT's row-wise work a latent pair at a time, the decoder's scratch smaller: 17 blocks fit on the 5070, split 17 / 13 | 0.260 | 77% / 85% |
| sparse_ratio 1.5 (the new default; below) | 0.240 | 76% / 80% |
| the same over 480 frames (the first chunk's start amortised) | **0.225** | **85% / 85%** |

The 5070 peaks at 11.5 GiB of its 12 (the card, everything counted), the 3060 at 5.8. 18 / 12 does not fit beside the
decoder. Neither card idles: the 3060 runs chunk k's last 13 blocks while the 5070 decodes chunk k - 1 and runs chunk
k + 1's LQ projection and first 17 blocks.

### x4

At x4 (864 x 480 -> 3456 x 1920) the KV cache alone is 28.6 GiB in half, more than the two cards hold. The engine then
runs in tiles: 1792 x 1024 tiles of the upscaled frame overlapping by 128 pixels (2 x 2 for 3456 x 1920), each its own
stream, over segments of 192 frames. Each tile's stream starts 32 frames before its segment to refill its KV cache and
the convolutions' memory (the outputs are discarded), and runs 8 frames past it (its lookahead). The noise is the
clip's by latent frame, so the tiles agree; overlaps are blended linearly. `NC_FLASHVSR_TILE` forces or forbids it.
Measured: **1.161 s a frame** over 384 frames (two full segments), both cards 76-78% busy; the dips are each tile stream's start and, before the last change, the segment's frames going to the encoder in a burst. The seams don't show (`docs/screenshots/vsr-x4-seams-120.jpg`, a crop across both overlap bands). The frame-to-frame
change across a segment boundary (1.98) is inside the clip's normal range (median 2.42).

### Settings (x2, 240 frames)

| sparse_ratio | local_range | s a frame | frame-to-frame change / the source's | mean abs difference to the source (0-255) |
|---|---|---:|---:|---:|
| 2.0 | 11 | 0.259 | 1.133 | 2.67 |
| 2.0 | 9 | 0.260 | 1.140 | 2.69 |
| **1.5** | **11** | **0.240** | **1.131** | 2.68 |
| 1.5 | 9 | 0.240 | 1.139 | 2.71 |

The steadiness ratio is the mean change between consecutive frames, brought down to the source's size, over the
source's own (1.0: as steady as the source). The four look the same up close (`docs/screenshots/vsr-settings-120.jpg`).
**Recommended and now the defaults: sparse_ratio 1.5, local_range 11.** 1.5 is 8% faster and no less steady here.
local_range 9 costs the same and is slightly less steady.
