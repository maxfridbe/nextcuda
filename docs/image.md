# Pictures, designs, upscaling and vectors

The image kind (Qwen-Image 2.1 on the Rust + CUDA engine, 2512 still through stable-diffusion.cpp's ggml-cuda) and the
models that sit beside it. Settings: [environment.md](environment.md); the models: [models.md](models.md).

## Qwen-Image

- **Qwen-Image 2.1:** text to image (RGBA with `rgba`), img2img with a `strength`, edits and compositions from up to
  8 reference pictures with an optional mask (it rides as the last picture, a sentence in the prompt saying what it
  is), every shared sampler and schedule, LoRAs. It runs on the Rust engine (`image/qwenimage21/src/rust/`): the
  pipeline is Rust, the kernels CUDA (cuBLAS and our own), the transformer kept on the card in its stored GGUF type.
  stable-diffusion.cpp and the phase 1 sd-cli worker are gone from it (2026-11-13).
- **2048 x 2048 on a 12 GB card:** the feed-forward runs 4,096 rows at a time, the kernels' scratch is let go between
  the stages, an edit's text and picture prefix spills to host memory when the card has not the room, and the VAE's
  up path runs in tiles of 64 latents with a 14-latent margin (its reach: the pixels are the whole picture's) above
  1024 x 1024. An edit of one picture at 2048 x 2048 peaks at 11.0 GiB on the 3060 (41 s a guided step there).
- **Out of card memory is `no_room`:** a picture the card has not the room for (a model loaded beside it, a size
  too big for what is free) answers 503 `no_room`, retryable - the front door frees idle work on the card and tries
  again - not a 400; a load that does not fit says so the same way.
- **Transparency (2.1, `background: "transparent"`):** the model card's phrase goes around the prompt, and the alpha
  is the VAE's fourth channel. (stable-diffusion.cpp needed `kernels/image/sd/sdcpp-vae-precision.patch` for it - its
  half VAE convolutions overflowed into opaque white blocks, sd.cpp#2024; the Rust engine's convolutions were never
  affected.) The models list `rgba`.
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
| `--opt-int8 0\|1` | `NS_QI_INT8` | 2.1: the DiT's block matrices as int8 ConvRot (the card's int8 tensor cores) |
| `--opt-sigmas 1,X,...` | `NS_QI_SIGMAS` | 2.1: a few-step model's own noise levels |

The `NC_SD_*` options are Qwen-Image 2512's (stable-diffusion.cpp).

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
- **The pDMD 2-step card (`p-h3-image-pdmd2`) and `eta`:** pDMD's 2-NFE LoRA beside the blocks, the shift-12 grid
  over N + 1 even levels (2 steps: 1, 0.923, 0), and a stochastic sampler (`image/ph3image/src/sampler.rs`): each
  step takes the model's x0, then puts the latent back at the next level as `(1 - next) x0 + next eps`, with
  `eps = sqrt(1 - eta^2) eps_hat + eta s_noise z` (`eps_hat` the step's own noise estimate, `z` fresh noise from the
  request's seed, drawn after the starting noise). eta 1 (this card's default; `sampler: lcm`) is ComfyUI's
  SamplerLCM, the sampler distribution-matching distillation steps with; eta 0 (`sampler: euler`; the base card's
  default) is flow Euler, the same arithmetic and the same pixels as before. Requests: `eta` (0 to 1, clamped: 3 is
  1) and `s_noise` (default 1, clamped to 0..2) as fields of image.generate / image.edit, or in `options`; the answer's
  `run` says `sampler`, `eta` and `s_noise`. An img2img (strength below 1) starts from the picture noised to its
  strength, then the same steps.
- **The grid texture (measured 2026-11-11, 1024 x 1024, seed 3):** every P-H3 picture here carries a 16-pixel
  periodic texture - the energy at the latent grid's harmonics 17 to 23 times their neighbours' (a camera photo:
  1.1). eta does not remove it: 2 steps 20.8 (eta 0) / 18.2 (eta 1), 4 steps 23.3 / 19.4, an instruction edit 18.4 /
  19.4, a 3-step polish from 0.5 46 / 54; the base card's 8 steps 17. A grid-free picture (1.0) sent through an
  img2img at strength 0.05, one step, comes back at 142 with either eta, and at strength 0.3, 2 steps, at 247 / 313:
  the texture comes with the latent path (the VAE encoder, the decoder, or the denoiser near t 0), not the sampler.
  `nextcuda image vae-roundtrip PICTURE --vae ph3=FILE --vae h3=FILE` (the denoiser skipped; each encoder through each
  decoder) narrows it: the encoders do not matter (P-H3's int8 and H3's fp16 encodes decode to the same score: the
  latent space is shared), and both decoders put a faint 16-pixel grid on a grid-free picture - a smooth gradient
  1.4 -> 11.7 (P-H3's int8 picture decoder) / 39 (H3's fp16 video decoder), a noisy one 1.0 -> 146 / 80, the 2-step
  fox 20.8 -> 23.7 / 32 - at about a quarter of a level RMS on flat areas, invisible by eye. The decoder is a ViT whose
  last linear writes each token's 16 x 16 pixels with nothing blending neighbours (ComfyUI's `ldm/minimax/vae.py`
  unpatchifies the same way: (C, T, pt, H, p, W, p)), and ours matched the reference's decode within one level on
  99.2 % of bytes (above). So the grid is the decoder architecture's, not a porting bug, and H3's full fp16 video
  decoder is no way out: blockier on flat areas and colour-shifted (pinker) on the fox. What makes it visible is what
  the denoiser leaves in the latents at 2 - 4 steps and after an img2img start (0.3: 247 - 313).
  Times are the same for both: 4.2 s a 2-step picture, 7.1 s 4 steps, 5.7 s a 3-step polish (pictures cached).
- **The deblock filter (`deblock`, on by default; `image/ph3image/src/deblock.rs`, kernels `kernels/image/ph3image`):**
  run on the decoded picture before it is quantized. What the grid is, measured: not a step at the tile boundaries
  (the pixel differences across them are no larger than inside) but one fixed 16 x 16 pattern every tile carries -
  brighter top rows, darker bottom ones, a fine dot texture - whose mean in one half of a picture correlates 0.87 -
  1.0 with the other half's, its strength following the content (faint on sky, several levels in fur). So the filter
  (1) estimates that pattern from the picture itself (the high-pass's mean at each of the 256 positions over all
  tiles, a busy tile weighing less, each position shrunk by a Wiener factor so a grid-free picture's comes out near
  0), and subtracts it at each tile's own amplitude (its projection, averaged over the 5 x 5 tiles around it, 0 to 3),
  then (2) runs H.264's edge test across each boundary (the luma's step against 8 levels, each side's activity
  against 3) and, where both are small, spreads the step left beyond the sides' slopes over 4 pixels each side. No
  low-pass anywhere: edges and fur keep their pixels. Requests: `deblock` (true / false) and `deblock_strength` (0 to
  1, default 1; 0 is off) as fields of image.generate / image.edit, or in `options`; the answer's `run` and the PNG's
  text say `deblock` (on / off) and `deblock_strength`. `deblock: false` hands the decoder's pixels on as before (the
  filter is not run). `nextcuda image deblock PICTURE OUT.png` runs it on a saved picture (the GPU's, timed, against the
  host reference).
  Measured 2026-11-11 (1024 x 1024, seed 3; the grid score as above, a camera photo 1.1), off -> on: pDMD 2 steps
  18.2 -> 1.1, 4 steps 19.4 -> 1.1, a portrait photo prompt 20.4 -> 0.7, an instruction edit 19.4 -> 2.0, the 3-step
  polish from 0.5 53.9 -> 1.6, an img2img at 0.3 313 -> 1.6; the base card's 8 steps 17.1 -> 2.8 and 20.6 -> 2.6
  (`deblock_strength` 0.5: 7.4). The VAE round trips (filtered from their 8-bit PNGs): the 2-step fox 23.8 -> 1.3
  (P-H3's decoder) / 32.4 -> 1.1 (H3's), the noisy synthetic 146 -> 9.7 / 80 -> 6.6, the smooth gradient 11.8 -> 7.1
  / 39 -> 8.9 (a pattern under half a level, partly the 8-bit rounding's). A grid-free picture is left alone (a
  rescaled render: its pattern comes out 0, the picture moves 0.08 levels RMS). The pixels it moves are the pattern's
  (1.4 - 3.5 levels RMS on renders); fur, wrinkles and edges stay as sharp (3x crops checked). With `deblock: false`
  the pictures are byte-identical to before. Time: 19 ms a 1 MP picture, 154 ms at 8.5 MP (18 ms a megapixel, the
  copies to and from the card included; the host reference 160 ms a megapixel); the GPU's pixels equal the host
  reference's.

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

## SVG paths: svgrender (DiffVG + LIVE)

`image.svg_fit` (a picture to SVG) and `image.svg_generate` (a prompt to SVG), card `svgrender-live` (`image/svgrender`,
`kernels/image/svgrender`): PyTorch-SVGRender's technique in Rust and CUDA - DiffVG's differentiable rasterizer
(prefiltered: cubic Bezier paths, fill by winding number, stroke by distance, each pixel's coverage smoothed over the
signed distance to the outline, and the gradient of every pixel to every control point, colour and stroke width) and
LIVE's layer-wise optimiser on it (paths added where the fit is worst, Adam with LIVE's per-layer decay, the
distance-weighted MSE and the xing loss), with VectorFusion's styles: `iconography`, `pixelart`, `sketch`,
`painting`. No weights; a job like the others (progress per step with the SVG so far, cancel at the next step).
`image.svg_generate` draws the prompt with Z-Image-Turbo first (with the style's words after it) and fits that.
The port, its checks and what it leaves out: [ports/SVGRENDER.md](ports/SVGRENDER.md).

- **Checked** (2026-11-09) against DiffVG 85802a7 (CPU, float32) and PyTorch-SVGRender dadfbba's LIVE
  (`reference/svgrender/ref.py`, `nextcuda image svg-check DIR`): prefiltered pixels 9e-7 - 1.7e-4 relative L2,
  colour and stroke-width gradients 6e-7 - 1e-5, control-point gradients 2.7e-4 - 2.2e-2; LIVE's xing loss 9e-8;
  SparseCoordInit's placements the same (23 of 24); a whole LIVE fit (32 paths, 8 layers x 200 steps, 240 px) the
  same quality: 17.73 / 22.93 dB against upstream's 17.79 / 22.72 dB.
- **Speed** (a LIVE step: the distance weight, the backward pass, xing, Adam): 0.31 ms at 240 px with 128 paths,
  1.1 ms at 512 px, 2.0 ms at 512 px with 512 paths (RTX 5070; the 3060 1.4-2.6x that). The default fit (128 paths,
  2000 steps, 384 px) takes ~1.2 s; upstream LIVE's 1600 steps took 245 s on the CPU. Memory: ~0.2 GB.
- **Text to SVG**: 12 s of drawing (1024 x 1024, 8 steps) and ~2 s of fitting for 96 paths.

## Describing a picture: image.describe

`image.describe` (nextsycl's contract, field for field): a picture to the prompt that would draw it, for an image
model's family - `target` is `natural`, `qwen-image`, `z-image`, `h3`, `hunyuanimage`, `moss-image`, `ming-image`,
`tags`, or a model's id (z-image-turbo, p-h3-image, nextsycl's minimax-h3-image ...: its family by architecture when
registered here, else by its id's prefix). The answer has the prompt (and with `include_negative` a negative prompt
for the families that read one), `suggested_size` - the target's best area at the picture's aspect on its grid: 1 MP
on multiples of 16, 4 MP on multiples of 32 for h3 and hunyuanimage - and the tokens and speeds. A job in the image
queue like image.upscale (label, reentrant_id, wait, image.status, image.cancel by label or reentrant_id).

- **The describer**: a chat model that reads pictures - Qwen3.8-27B GSQ IQ3_S with its vision tower (the mmproj beside
  it; [llm.md](llm.md)), the card already served for chat: no new engine, no other card. A chat model whose entry has
  an `mmproj` file lists `vision` and `describe` in system.models. Unnamed, the loaded chat model is used when it reads
  pictures (no swap), else the enabled one whose card takes the least memory. It is not split differently for this: a
  smaller-context card would save ~1 GB of cache, but the 27B over both cards cannot sit beside a picture model
  either way, and a second chat entry would swap with the chat model the platform uses.
- **How**: the picture flattened on white and shown at most `max_pixels` (1 MP, Lanczos); one instruction per family
  (nextsycl's table) in the system turn; `think_budget: 0` (no thinking), greedy at temperature 0; the answer streamed
  from `/api/chat` through the switcher and cut at `PROMPT:` / `NEGATIVE:`. A cancel closes the stream: the job ended
  0.47 s after image.cancel.
- **Measured** (2026-11-11, a 1024 x 768 flat illustration, for z-image-turbo, medium): cold - z-image-turbo stopped
  for it, the describer loaded in ~12 s - 18.3 s in all; the prompt 1,154 tokens read at 415 tok/s with the tower
  (2.8 s), 110 written at 55 tok/s (2.0 s). Again (the prompt cached): 2.2 s. For h3, short: 3.2 s (632 / 51 tok/s).
  The describer holds 11.1 GB of the 5070 and 6.0 GB of the 3060 with the picture read. Drawn back with z-image-turbo
  at its suggested 1184 x 880: 12.3 s, the barn, the sign's "FARM", the sun and the hill where they were.

### Who describes: `describer`

`describer` is `best` (the default: the chat model above), `native` (the target's own text encoder) or the id of one
of `image.describers` - with any target. image.describers lists every describer here: the chat models that read
pictures and the image models' text encoders that do, each with its name, base model, size, the image models that
share it, whether it is available (else why) and which one best picks now. Image models whose encoder reads pictures
say `describe_native` in system.models.

| describer | the files | runs in | on the box |
|---|---|---|---|
| `qwen3-vl-8b-instruct` (Qwen-Image's own) | Qwen-Image 2.1's `text-encoder` (Qwen3VL-8B-Instruct Q4_K_M) and `llm-vision` (its F16 mmproj): all four 2.1 cards share them | a 2.1 model's server: the loaded one, else the smallest | 4.35 GiB of weights + 1.1 GiB tower; 6.1 GiB held |
| `qwen3-vl-4b-instruct` (H3's own) | P-H3-Image's `text-encoder` GGUF: Qwen3-VL-4B whole in transformers' names (36 layers, the final norm, the head tied to the embedding, the 24-block vision tower in bf16), the card's `tokenizer` files | p-h3-image's server | 3.98 GiB of Q8_0 + 0.8 GiB tower; 5.2 GiB held |

- **native**: a target model's own encoder; a family's enabled model with the smallest encoder (a loaded one first,
  then the least memory). None that reads pictures: invalid_params with field `describer` ("target z-image: its
  text encoder (Qwen3-4B) does not read pictures; use describer best"); a target model a card offers and not
  downloaded: model_not_downloaded.
- **How it runs** (crates/qwen3vl `chat` and `reader`): the encoder's layers over the prompt (the picture's vision
  tokens at its pads, deepstack, interleaved M-RoPE), keys and values into a half cache, then one row a token on
  ggml-cuda's MMVQ from the stored blocks (crates/qwen35 `products`), the final norm and the head; the same
  instruction as the chat describer, no thinking (Qwen3-VL Instruct has none). The image server loads it in place of
  the pipeline (one or the other on a 12 GB card) and keeps it until the next picture is drawn, which loads the
  pipeline again; a server the front door starts for a describe (`--describe-first`) loads no pipeline at all.
  `nextcuda image describe PICTURE --model ID` runs one in a process.
- **Measured** (2026-11-11, the blob portrait 640 x 352, target qwen-image, medium, the 5070): Qwen3-VL-8B - 436
  rows read in 0.72 s (604/s with the tower), 115 tokens in 1.25 s (92 tok/s), loaded in 3.2 s; through the front
  door 7.7 s cold (rrq4's server started), 1.5 s again. Qwen3-VL-4B - 0.57 s (765/s), 147 tokens at 97.5 tok/s,
  loaded in 4.5 s. Best (Qwen3.8-27B over both cards) on the same picture: 595 rows at 293/s, 114 tokens at 50 tok/s,
  27.5 s cold (its load), 11.1 + 6.0 GB held.

The other encoders, not built:

- **qwen-image-2512** (Qwen2.5-VL-7B): its card has the language model only; the projector
  (mmproj-Qwen2.5-VL-7B-Instruct-f16, ~1.35 GB) is a download. Then: the Qwen2.5-VL vision tower moves from
  image/olmocr2 into the foundation (its windowed attention, 2-D rope, merger), and `chat` takes Qwen2's layer (q/k/v
  biases, no q/k norm, M-RoPE sections 16/24/24 side by side, theta 1e6, no deepstack); a NATIVES row with roles
  text-encoder + llm-vision.
- **moss-image** (Qwen3.5-9B on the qwen35 engine): the engine already writes and has a vision tower
  (crates/qwen35 `vision`); it needs the 9B's mmproj (a download) and a Describer over Qwen35's session, picture and
  step - no new decode loop.
