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
| speed estimate on the pair | about 0.3-0.4 s a frame (a film of 5,665 frames in about 35 min) | about 1-1.5 s a frame (about 2 hours a film) | about 2-3 s a frame |
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
