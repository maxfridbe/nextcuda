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

## Designs: Ming-Image-0.1-Design

inclusionAI's design model (`image/mingimage`): Z-Image's 6B single-stream DiT conditioned by Ling-Mini-2.0 (a 16B MoE,
1.4B active, 256 experts a layer routed on the host) through a 1.5B connector, and an RGBA VAE (Wan 2.1's decoder) -
inclusionAI's pipeline on the shared kernels and its own (`kernels/image/mingimage`). The DiT and the VAE on the 5070,
the text encoder on the 3060, every weight resident in its GGUF's type.

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
