# Pictures, designs, upscaling and vectors

The image kind (Qwen-Image through stable-diffusion.cpp's ggml-cuda, or the Rust + CUDA engine) and the models that sit
beside it. Settings: [environment.md](environment.md); the models: [models.md](models.md).

## Qwen-Image

- **Qwen-Image 2.1:** text to image (RGBA with `rgba`), img2img with a `strength`, edits and compositions from up to
  8 reference pictures with an optional mask, the effort ladder below. `NC_IMAGE_BACKEND=rust` runs it on the Rust
  engine instead (text to image so far): the pipeline is Rust, the kernels CUDA (cuBLAS and our own), the
  transformer kept on the card in its stored GGUF type, no ggml.
- **Qwen-Image 2512:** text to image and img2img. Under 20 steps the Lightning LoRA runs (4 steps, guidance 1); at 20
  or more it is bypassed for a full guided run - the platform's effort ladder, so an engine behind it behaves as
  the one it replaces.
- **A 12 GB card is enough:** the weights stay in host memory and are staged to the card per graph
  (`NC_SD_OFFLOAD`, `NC_SD_MAX_VRAM`); the text encoder, the transformer and the VAE take turns.
- **Community GGUFs are accepted as they are:** a file that stores `img_in.weight` reshaped (realrebelai's does) gets
  a corrected copy made once beside it; your file is not changed.
- **Progress and cancel:** every step is reported as it is made and a cancel stops the step in progress.
- **Energy and timing in the answer:** `nextsycl.seconds`, `steps`, `seed` come back in each result (the key keeps
  nextsycl's name: the platform reads it).

### The clocks

`nextcuda models pull <id>`, then ask the front door (:8000) for a picture - or `nextcuda image gen "<prompt>"`.
Measured by `nextcuda image bench` (8 steps, true guidance: two passes a step; warm pictures, the first after a load in
brackets in the full tables of [`benchy/`](benchy/)), weights offloaded to host memory, in process:

**RTX 5070, 12 GB, PCIe gen4 x16**

| id | engine | picture | text encode | per step | VAE | total | peak GPU memory |
|---|---|---|---:|---:|---:|---:|---:|
| `qwen-image-2.1-q8` | stable-diffusion.cpp | 768x768 | 0.7 s | 0.95 s | 1.6 s | 13.4 s | 8.2 GB |
| `qwen-image-2.1-q8` | stable-diffusion.cpp | 1024x1024 | 0.3 s | 1.42 s | 3.5 s | 26.3 s | 8.6 GB |
| `qwen-image-2.1-q8` | **Rust + CUDA** | 768x768 | 2.7 s | 1.62 s | 0.5 s | 16.3 s | 10.6 GB |
| `qwen-image-2.1-q8` | **Rust + CUDA** | 1024x1024 | 2.8 s | 3.26 s | 0.9 s | 29.9 s | 11.6 GB |
| `qwen-image-2.1-rrq4` | stable-diffusion.cpp | 768x768 | 0.4 s | 1.03 s | 1.6 s | 13.8 s | 6.4 GB |
| `qwen-image-2.1-rrq4` | stable-diffusion.cpp | 1024x1024 | 0.3 s | 1.52 s | 3.5 s | 28.0 s | 6.8 GB |
| `qwen-image-2512-q4km` (Lightning, 8 steps) | stable-diffusion.cpp | 768x768 | 0.6 s | 0.42 s | 2.2 s | 16.6 s | 11.0 GB |
| `qwen-image-2512-q4km` (Lightning, 8 steps) | stable-diffusion.cpp | 1024x1024 | 0.3 s | 0.44 s | 4.4 s | 28.5 s | 11.0 GB |

**RTX 3060, 12 GB, PCIe gen4 x1** (the card sits on a x1 link: about 2 GB/s to the host, where the 5070 has about 25)

| id | engine | picture | text encode | per step | VAE | total | peak GPU memory |
|---|---|---|---:|---:|---:|---:|---:|
| `qwen-image-2.1-q8` | stable-diffusion.cpp | 768x768 | 3.0 s | 2.35 s | 4.2 s | 36.9 s | 8.1 GB |
| `qwen-image-2.1-q8` | stable-diffusion.cpp | 1024x1024 | 2.9 s | 3.18 s | 9.0 s | 64.7 s | 8.5 GB |
| `qwen-image-2.1-q8` | **Rust + CUDA** | 768x768 | 7.8 s | 3.81 s | 1.3 s | 39.8 s | 10.6 GB |
| `qwen-image-2.1-rrq4` | stable-diffusion.cpp | 768x768 | 3.0 s | 2.48 s | 4.2 s | 37.5 s | 6.5 GB |
| `qwen-image-2512-q4km` (Lightning, 4 steps) | stable-diffusion.cpp | 768x768 | 4.3 s | 0.75 s | 6.0 s | 37.6 s | 11.3 GB |

The first picture after a load is slower (the weights page in: the 2512 takes 70-85 s on either card). Energy and
average power are in the full tables: the 5070 draws about 220 W while sampling, the 3060 about 140 W.

What the clocks say: the step is compute, not PCIe (a resident transformer gave the same 1.27 s/step); the Rust
engine's VAE is 3x faster than ggml's and its steps are 1.7-2.3x slower, which is where the silo goes (attention at
4,000 tokens is cuBLAS and a score table today, a fused kernel in the silo); its text encoder streams the layers
over PCIe on every prompt, so it costs 2.7 s on the 5070 and 7.8 s on the 3060's x1 link (a host-resident cache of
the encoder is the fix). [`PHASES.md`](PHASES.md) has the history of the numbers.

### Engine options: `--opt-NAME`

| option | variable | |
|---|---|---|
| `--opt-offload 0\|1` | `NC_SD_OFFLOAD` | weights in host memory, staged per graph (default 1) |
| `--opt-max-vram GiB` | `NC_SD_MAX_VRAM` | the card memory the weights and buffers may take |
| `--opt-sd-args k=v;...` | `NC_SD_ARGS` | `threads=N`, `flash_attn=0\|1`, `vae_tiling=0\|1` |
| `--opt-backend native\|rust\|python` | `NC_IMAGE_BACKEND` | stable-diffusion.cpp in process (default), the Rust engine on the shared CUDA kernels, or the phase 1 sd-cli worker |

## P-H3-Image: text to image and edits on MiniMax H3

Pruna AI's image model (`image/ph3image`, card `p-h3-image`; the MiniMax H3 Community License, its Qwen3-VL-4B parts
Apache-2.0): MiniMax H3's own 50-block denoiser over one latent frame, conditioned by a Qwen3-VL-4B text encoder whose
hidden states (after layers 13, 19, 25, 31 and the normed last) a connector maps to H3's 5120-wide text space; H3's VAE
encoder and a one-frame ViT decoder. Pruna's diffusers pipeline over the foundation's H3 core (`crates/h3`), which the
video engine runs too.

- **Files:** vantagewithai's Q4_K_M GGUF of Pruna's transformer (12.6 GB; its non-quantized tensors are Pruna's bf16
  ones), SearchingMan's Q8_0 text encoder GGUF (5.4 GB, the vision tower and connector inside), Pruna's int8 ConvRot
  VAE (3.1 GB). Pruna's LoRA for an existing H3 is not used: it leaves out the fine-tuned AdaLN projections and
  explains 12-70% of the block matrices' change (FL2VA Q4_K_M + LoRA measured 0.13 off in velocity on nextsycl).
- **Sampling:** 8 Euler steps on the checkpoint's sigmas as written (no shift), no guidance; the 2 audio latents the
  layout carries are stepped on the same grid and dropped; the noise from one torch generator, as diffusers draws it.
- **Edits:** 1 to 8 pictures; each a `<Picture i>` vision block (about 1 MP) for the encoder and its latents (the
  posterior sampled at seed 42, at its own size within 1-8 MP) as an H3 ref2va reference held at t 0.999;
  `Edit <Picture 1>. ` goes before the instruction. Without a size the canvas is picture 1 at 4-8 MP.
- **Two 12 GB cards:** the text encoder (3.6 GiB) and the decoder (2.25 GiB) resident on the 3060, the blocks 39 on
  the 5070 and 11 on the 3060 (`NC_PH3_RESERVE_GIB`, 2, is the room each keeps); loads in ~15 s.

| picture | per step | total | peak (5070 / 3060) |
|---|---:|---:|---:|
| 1024 x 1024 | 1.33 s | 12.5 s | 10.1 / 9.5 GB |
| 2048 x 2048 (4 MP) | 6.1 s | 56 s | 10.1 / 9.9 GB |
| 2912 x 2912 (8 MP, the default) | 19.1 s | 166 s | 10.6 / 10.6 GB |
| edit, 1024 x 1024 from a 1 MP picture | 4.3 s | 44 s (7.5 s conditioning) | 10.9 / 10.0 GB |
| edit, 2048 x 2048 (the default canvas) from a 1 MP picture | 23.2 s | 218 s (24 s conditioning) | 10.9 / 11.0 GB |

**Checked** (2026-11-08) against nextsycl's `reference/p-h3-image/ref.py` (diffusers 84532a05 on the CPU in float32,
Pruna's full bf16 model; 576 x 384, seed 3, the model's own prompt; an edit of one 1024 x 1024 picture on that canvas):
`nextcuda image check DIR --model p-h3-image` (each stage fed the reference's input; `--denoiser FILE` runs another
denoiser alone over both cards, `reftext` feeds it the reference's refined text). Relative L2 errors:

| stage | text to image | edit | nextsycl (int8) t2i / edit |
|---|---|---|---|
| tokens | 33 of 33 equal | 1,071 of 1,071 equal | equal |
| vision tower (float32) | - | 7e-6, deepstack 3e-6 - 1.1e-5 | - / 3.2e-4 |
| hidden states 13 - 31 (Q8_0 encoder) | 5.6e-3 - 6.2e-3 | 3.4e-3 - 5.1e-3 | 5.9e-2 - 6.6e-2 / 2.5e-2 - 3.4e-2 |
| hidden state 36 (final norm) | 3.9e-2 | 1.8e-2 | 7.4e-2 / 4.4e-2 |
| connector output | 5.5e-4 (from the reference's states 4e-8) | 2.2e-3 | 1.6e-3 / 5.9e-3 |
| VAE encoder (sampled posterior) | - | 4.2e-3 | - / 2.4e-3 |
| positions | exact | exact | exact |
| refined text | 7.6e-3 | 9.5e-3 | 6.3e-3 / 9.5e-3 |
| velocities, Q4_K_M (the card's) | 6.1e-2 - 1.5e-1 | 9.2e-2 - 2.1e-1 | |
| velocities, AtomicChat Q8_0 (blocks only) | 9.8e-3 - 3.1e-2 | 2.8e-2 - 4.8e-2 | 1.5e-2 - 2.9e-2 / 2.3e-2 - 5.5e-2 |
| final latents, 8 steps: Q4_K_M / Q8_0 | 2.3e-1 / 2.9e-2 | 2.9e-1 / 5.9e-2 | 3.7e-2 / 3.4e-2 |
| VAE decoder (int8) | 2.7e-3; 80.0 % of bytes equal, 99.2 % within 1 | 3.1e-3; 80.7 %, 99.35 % | 5.2e-3 / 6.2e-3 |

Every stage but the denoiser is at or under nextsycl's error. The denoiser's is the Q4_K_M file's: the same engine on
the Q8_0 file lands where nextsycl's int8 and Q8_0 do (nextsycl measured H3 FL2VA Q4_K_M + LoRA at 1.3e-1 / 2.2e-1 too).
The Q8_0 blocks (19.1 GiB) fill both cards alone: beside the text encoder and decoder only a ~4-bit file fits two
12 GB cards. The 8-step run is chaotic (nextsycl: two runs of the same kernels at 8 MP are 39 dB apart): the Q4_K_M
pictures keep the reference's scene and composition, details differ.

An edit's references add their rows to the sequence: several pictures at a 4 MP canvas pass the cards' room and are
refused as no_room - a smaller canvas (`size`) or smaller pictures fit.

## Designs: Ming-Image-0.1-Design

inclusionAI's design model (`image/mingimage`): Z-Image's 6B single-stream DiT (the foundation's, `crates/zimage`)
conditioned by Ling-Mini-2.0 (a 16B MoE, 1.4B active, 256 experts a layer routed on the host) through a 1.5B
connector, and an RGBA VAE (Wan 2.1's decoder) - inclusionAI's pipeline on the shared kernels and its own
(`kernels/image/mingimage`). The DiT and the VAE on the 5070, the text encoder on the 3060, every weight resident in
its GGUF's type.

- **Checked** (2026-10-19) against inclusionAI's own DiT class and its text side line for line, in float32 on the
  CPU on the same files (`reference/mingimage/ref.py`): tokens identical; the direct condition 8e-3, the query
  features 5e-2 (the MoE's routing ties: 3 % of the query tokens' expert picks differ by the last layer); the DiT's
  velocity 1e-3 (cos 0.999999) at steps 0 and 5; the VAE 2.4e-4; the whole pipeline's picture from the same noise 1.5e-2
  (cos 0.99989) - the same picture by eye.
- **Its schedule is the official code's**, not the scheduler config's intent: FlowMatchEuler at a static shift of 6
  (the pipeline's dynamic-shift switch is set on a frozen config and never reaches the scheduler), 12 steps of which
  the last runs at sigma 0.
- **Half's range:** the MLPs' gated products and the last blocks' values reach past 65,504 (infinities from block 12, or
  28 on some prompts): both are made at 1/256 in half and multiplied back in float32 (the norms' epsilon scaled with
  them), so the tensor cores keep half.
- **Speed** (RTX 5070 + 3060): 1024 x 1024 in 17.7 s (1.49 s a step), 2048 x 2048 in 164 s; 9.5 GB / 10.0 GB at
  1024, 10.9 GB on the 5070 at 2048. Loads in 10 s.
- **Transparency:** RGBA is decoded, but with these GGUF files the model's alpha comes back opaque (as through
  ComfyUI on the same files); the official phrase is put before the prompt when RGBA is asked.

## Z-Image-Turbo: text to image in 8 steps

Tongyi-MAI's few-step model (`image/zimage`, card `z-image-turbo`; Apache-2.0): Z-Image's 6B single-stream DiT - the
same denoiser Ming-Image runs, now the foundation's (`crates/zimage`) - distilled to 8 steps without guidance,
conditioned by Qwen3-4B, decoded by FLUX's 16-channel VAE. Photoreal pictures and legible text in them. Tongyi-MAI's pipeline
(`src/zimage/pipeline.py`; diffusers' `ZImagePipeline` is the same) on the shared kernels, the DiT's and its own
(`kernels/image/zimage`: the VAE's group norm and upsampling).

- **Files:** unsloth's Q8_0 GGUF of the transformer (7.2 GB: the blocks Q8_0, the refiners, embedders and pad tokens
  bfloat16 / float32 - range reads against Tongyi-MAI's weights: the float tensors bit-equal, the Q8_0 rows 0.55 % off),
  Qwen's own Q8_0 GGUF of Qwen3-4B (Z-Image's text encoder is Qwen3-4B's weights; only the last layer's MLP differs,
  and that layer is never read), Tongyi-MAI's VAE and tokenizer.
- **Text:** Qwen3's chat template around the prompt (`<|im_start|>user ... <|im_end|> <|im_start|>assistant`, thinking
  on: no empty think block), at most 512 tokens; the hidden state after layer 35 of 36 (`hidden_states[-2]`, no final
  norm) is the condition - the encoder loads those 35 layers only (P-H3-Image's Qwen3 tower in `crates/h3`, here at
  Qwen3's rotary base 1e6 and reading llama.cpp's tensor names).
- **Sampling:** 8 Euler steps on FlowMatchEuler's levels as the official pipeline makes them - linspace(1, 1/8, 8)
  shifted by 3 (static: the scheduler config sets no dynamic shift), then 0 - one pass a step, no guidance (the
  pipeline's guidance_scale 0). The noise is torch.randn's on a CPU generator seeded `seed`. Unlike Ming-Image's
  reference, Z-Image pads the caption and the image to multiples of 32 with its learned pad tokens and lets every
  attention see them; the DiT does the same here.
- **The request:** `steps` 1-50 (default 8), `shift` (default 3, the demo's time shift), `size` sides rounded to 16, 256
  to 2048 a side, at most 2.4 MP (1536 x 1536, 2048 x 1152; at 2048 x 2048 the VAE's last stages pass the 5070's room).
  A negative prompt, a cfg other than 1 (or 0), another sampler or schedule, LoRAs, RGBA and an init picture are
  refused (invalid_params). Text to image only.
- **Two 12 GB cards:** the DiT and the VAE resident on the 5070, the text encoder (3.5 GiB of Q8_0) on the 3060; loads
  in 15 s with the files in the page cache (90 s from the drive).

| picture | per step | total | peak (5070 / 3060) |
|---|---:|---:|---:|
| 1024 x 1024 | 1.36 s | 11.7 s (VAE 0.75 s) | 9.4 / 3.9 GB |
| 1280 x 720 | 1.18 s | 10.2 s | 9.4 / 3.9 GB |
| 832 x 1248 | 1.35 s | 11.6 s | 9.4 / 3.9 GB |
| 1536 x 1536 | 4.26 s | 36.4 s (VAE 1.75 s) | 11.0 / 3.9 GB |

**Checked** (2026-11-08) against nextsycl's `reference/z-image-turbo/ref.py` (ddc7cc5: diffusers 84532a05's classes on
the CPU in float32, run on this card's own files - the Q8_0 DiT and Qwen3-4B GGUFs, each matrix expanded for its own
product): `nextcuda image check DIR --model z-image-turbo` (each stage fed the reference's input; `full` is a request's
whole path from the prompt and seed). The README's prompt, 8 steps; relative L2 errors:

| stage | 1024 x 1024, seed 42 | 1280 x 720, seed 7 (16 image pad rows) | nextsycl (BF16, B70) |
|---|---|---|---|
| noise (torch's CPU randn) | 3.2e-8 | 3.2e-8 | 3.2e-8 |
| tokens, sigmas | identical (101), exact | identical, exact | identical, exact |
| text encoder after layer 1 / 18 / 35 | 5.6e-4 / 1.7e-3 / 2.2e-3 | the same | 3.7e-4 / 6.1e-5 / 2.5e-4 |
| context refiner, noise refiner | 1.4e-4, 1.1e-4 | 1.4e-4, 1.1e-4 | 1.5e-4, 1.2e-4 |
| layer 0 (image / caption rows) | 1.3e-4 / 1.3e-4 | 1.3e-4 / 1.3e-4 | 1.4e-4 |
| velocity, each of the 8 steps | 0.7e-3 .. 3.2e-3, cos >= 0.999995 | 0.8e-3 .. 1.9e-3 | 0.8e-3 .. 4.3e-3 |
| latents after 8 steps | 1.3e-2, cos 0.99991 | 1.3e-2, cos 0.99991 | 1.6e-2 .. 4.9e-2 |
| VAE decoder | 4.0e-4; 97.4 % of bytes equal, 100 % within 1 | 3.9e-4; 97.5 % | 3.2e-4 |
| the whole request vs ref.png | cos 0.99977, PSNR 42.8 dB | cos 0.99974, 41.6 dB | cos 0.9996 / 0.9991 |

A seed draws the same noise as diffusers (and nextsycl), so on the same files it gives the reference's picture. Against
the full-precision model (ref.py on unsloth's BF16 GGUF - the official weights bit for bit - and the official text
encoder; nextsycl's default card) the picture is the Q8_0 file's, not the engine's: ours cos 0.9906 (PSNR 26.8 dB),
diffusers itself on the Q8_0 files 0.9908 - the same scene and composition, details drawn differently. The BF16 DiT
(12.3 GB) does not fit the 5070 beside its work. The
Q8_0 text encoder against the official bf16 one (ref.py on Tongyi-MAI's text_encoder shards): 6.3e-3 / 3.9e-3 / 7.0e-3
after layers 1 / 18 / 35 - Qwen's Qwen3-4B GGUF is Z-Image's encoder up to its quantization.

## Upscaling: Real-ESRGAN and waifu2x

`image.upscale` and `video.upscale` (every frame, the sound copied; an MP4 back): the graphs and weights
realesrgan-ncnn-vulkan and waifu2x-ncnn-vulkan ship, run in Rust and CUDA (`image/upscale`, `kernels/image/upscale`)
- an ncnn graph runner over the shared strided convolutions, a transposed convolution as the convolution of its
dilated input with the flipped kernel - and every choice the two programs make: Real-ESRGAN's tiles of 200 with 10
mirrored pixels around, waifu2x's of 400 with 7, 18 or 28 repeated ones (its far side rounded up for cunet), the
alpha bicubic, waifu2x's 2x model run again for 4x and up.

- **Checked** (2026-10-19) against the programs' own output on the same pictures (`nextcuda image upscale-check`):
  every variant and denoise level, 58-66 dB on a photograph (worst pixel 1-4 of 255), 63-69 dB on an RGBA logo.
- **Speed** (RTX 5070): a 720p frame doubled by animevideov3 in 0.31 s (a 5 s clip, 124 frames, in 10.4 s through
  video.upscale), waifu2x cunet 0.49 s; x4plus 512 -> 2048 in 3.4-3.8 s. **Memory**: 0.3 GB (animevideov3) to 0.9 GB
  (x4plus, cunet) at the default tiles.

## Video super-resolution: FlashVSR v1.1

`video.upscale` with `"model": "flashvsr", "variant": "flashvsr-v1.1"` (`image/flashvsr`, `kernels/image/flashvsr`): OpenImagingLab's
one-step diffusion VSR (Apache-2.0) - Wan 2.1 T2V-1.3B distilled to one step and made a stream: 8 frames a step, each
step's attention reaching the 24 frames before it (a KV cache of three latent pairs, locality-constrained block-sparse
attention), a causal LQ projection in and a tiny conditional decoder out, then FlashVSR's colour fix. The clip goes
through whole: no seams, no length limit, detail steady from frame to frame; faces and skin stay natural where
Real-ESRGAN's models turn them waxy. A picture (`image.upscale`) is a still clip of 25 frames.

- **Checked** (2026-11-05) against the official pipeline in float32 on the CPU (`reference/flashvsr/ref.py`, the
  block-sparse kernel replaced by its exact masked equivalent), 33 frames at 1024 x 768: the LQ projection 3.8e-4, the
  DiT's blocks 2e-3 (cos 0.999998), the latents 1.8e-3, the decoder's frames 9e-4, the bytes 64 dB.
- **Speed** (864 x 480 -> 1728 x 960, the platform's films): 0.43 s a frame on the 5070 + 3060 pipelined (a film of 5,665
  frames in about 41 min); realesrgan-x4plus takes 6.2 s a frame for the same x2 (made at x4, brought down).
- **Memory**: the frame is mirrored out to multiples of 128 (1792 x 1024 inside). Its KV cache is 7.9 GiB there, more than
  one card beside the weights: the 30 blocks are split, 12 on the 5070 with the LQ projection and the decoder (9.4 GiB),
  18 on the 3060 (7.0 GiB). Settings: `NC_FLASHVSR_SPLIT`, `NC_FLASHVSR_DECODER`, `NC_FLASHVSR_PIPELINE`.

<p align="center"><img src="screenshots/vsr-face-120.jpg" alt="bicubic, realesrgan-x4plus and FlashVSR at x2 on a platform film's frame" width="900"></p>

## Vectors: StarVector-1B

`image.vectorize` (a job in the image queue; `nextcuda image vectorize FILE`): a picture written as SVG code by
StarVector-1B im2svg - a CLIP ViT-L/14 encoder, an adapter and StarCoder-1B (`image/starvector`), decoded as the model's
own `generate_im2svg` does (2 beams sampled, top_p 0.9; `max_length` 5000 by default). It suits logos, icons and simple
illustrations; a drawing that does not fit the length limit comes back cut off (`closed: false`).

- **Checked** against the starvector package in float32 (`reference/starvector/ref.py`): the picture's pixels exact, the
  encoder 1.8e-2 / 7.5e-3, the adapter 9.8e-3 / 4.4e-3, the prefill logits 4.6e-4 / 8.0e-4, 48 teacher-forced rows all the
  reference's tokens, an icon's 300 greedy tokens the reference's text.
- **Speed** (RTX 5070): 196 tokens/s greedy, 125 with 2 beams; a complex picture ran to the limit, 4,741 tokens in 38.8 s,
  3.35 GiB at the peak - it fits any card.
