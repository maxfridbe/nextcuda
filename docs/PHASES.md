# nextcuda: phases

nextcuda is nextsycl's control plane and API (front door, jobs, queue, precheck, contracts, CLI, web front ends) with
NVIDIA behind it instead of SYCL. The platform talks to it with the same `/rpc/<kind>.<method>` API.

## Phase 1 (this tree): wired up

- **Foundation:** `crates/core` is NVIDIA-only (`nvidia-smi`; buffers are host memory: engines own their GPU memory).
  `glue/serve/src/gpustat.rs` samples cards with `nvidia-smi` for the front door's placement.
- **Engines:** `image/qwenimage21`: Qwen-Image 2.1 through stable-diffusion.cpp (`sd-cli`, CUDA), driven by the
  platform's own python engine (`image/qwenimage21/python/`, vendored) over JSON lines. Arch name, file roles and
  contract are the SYCL engine's, so the registry, model ids and API are the same.
- **Models:** `model-cards/image/qwen-image-2.1-{q8,q6k}.yml` are the platform's files (leejet GGUF, Qwen3-VL GGUF +
  projector, VAE).
- **Run:** servers are plain processes (`src/native.rs`), not containers; `NS_CONTAINER_ENGINE` brings containers
  back. Build: `./build.sh` (debian:trixie toolchain, glibc matches the GPU host) -> `dist/`.
- **Not here yet:** llm, video (H3), audio (speech, music), OCR, 3D; the turbo preset (a fused LoRA, a local part).
  `image check`, the stage checks and the kernels directory are gone with SYCL.

## Phase 2 progress

- **Done (verified on the 5070 and 3060):** `kernels/ns/ns.cu` (the ns.h ABI on CUDA: devices, memory, streams,
  copies, tickets as events), `kernels/image/qwenimage21/qi21.cu` (the engine's own kernels, ported), `crates/sys`
  (dlopen) and a device-backed `crates/core` (host-memory fallback without the library), `kernels/build.sh`
  (nvcc, sm_86 + sm_120 + PTX), `nextcuda image selftest` (every qi21 kernel against host arithmetic: ok).
- **Next, in order:** `kernels/diffusion/nsd.cu` (H3's shared kernels: int8 ConvRot linear, rms norm + modulation,
  rope, attention, dequant, conv; oneDNN's matmul and fused SDPA become cuBLASLt and a CUDA flash attention);
  `crates/qwen3vl` (the text encoder), the DiT (`dit.rs`), the VAE (`vae.rs`), the sampler; then `check` against
  sd.cpp's pictures (the python engine is the reference); then the silo (`kernels/image/qwenimage21/silo`).
- **Qwen-Image-2512** is a different architecture from 2.1 (a 20B dual-stream MMDiT with a Qwen2.5-VL encoder, not
  the 7B single-stream DiT with Qwen3-VL), so it is its own engine crate on the same foundation; it follows 2.1's
  nsd port, which it shares most kernels with.

## Phase 2 progress (the Rust engine on the shared CUDA kernels)

- **Done, checked on the 5070:** `kernels/diffusion/nsd.cu` (cuBLAS linears, a quantized linear that keeps GGUF
  weights in their stored type and makes a slice of rows half at a time, RMS norm + modulation, RoPE, SwiGLU, gated
  add, attention for grouped heads, im2col convolutions); `nextcuda image selftest` runs each against host
  arithmetic (dequant of Q8_0 / Q4_K / Q5_K / Q6_K exact; the rest within 1e-3).
- **Running, checked on the 5070 (2026-10-16):** nextsycl's engine as `image/qwenimage21/src/rust/` - the DiT with
  its blocks in the type the GGUF stores (mixed-quant files included), the VAE, every ComfyUI sampler and schedule,
  sigma presets, img2img with strength, edits with up to 8 references through the vision tower, LoRAs as side
  paths, RGBA, progress and cancel; a text encoder read from a llama.cpp GGUF a layer at a time with a host cache
  and an embedding cache (`crates/qwen3vl/src/gguf.rs`).
- **nextsycl's documented optimizations, done:** fused flash attention in the silo (18.3 -> 5.5 ms a call at 4k
  tokens), layer norm + modulation fused, ConvRot int8 W8A8 (`NS_QI_INT8=1`; cuBLASLt int8 tensor cores, about 2x a
  step here against 1.4x on the B70), AOT per-card builds, the kernels checked against host arithmetic.
- **The Rust engine is the only one for 2.1 (2026-11-13):** stable-diffusion.cpp's in-process path and the sd-cli
  worker are gone from it; masks ported (the last picture, a sentence in the prompt), the canvas of an edit from the
  last picture that is not the mask. 2048 x 2048 fits a 12 GB card: the feed-forward in 4,096-row slices, the kernels'
  scratch freed between stages (`nsd_trim`), the VAE's up path in 64-latent tiles with a 14-latent margin above
  1024 x 1024 (within one level of the whole picture's: the products' rounding). An edit of one picture to 2048 x 2048, guided, on the 3060: 41 s a step, peak
  11.0 GiB (it was refused before: a 256 MiB allocation short). Out of card memory anywhere in the image server -
  the engine's allocations, the kernels' scratch ("did not fit in memory"), cuBLAS's workspace, a load beside other
  work - is answered 503 `no_room` (retryable), never a 400.
- **Qwen-Image 2512 in Rust (2026-11-13):** `image/qwenimage21/src/q2512/` - the 60-block MMDiT in the GGUF's
  types (Q4_K_M: 33 blocks on a 12 GB card, the other 27 copied from pinned host memory each step), Qwen2.5-VL 7B from
  its GGUF (`crates/qwen3vl`'s encoder now reads Qwen2.5's biases and plain RoPE, and its final norm), the Wan 2.1
  VAE, Lightning below 20 steps. nextsycl has no 2512 engine to port: written from diffusers' QwenImage pipeline and
  stable-diffusion.cpp's qwen_image.hpp. Its activations pass half's range (the text stream reaches 1e9; v in the
  late blocks passes 65,504): float32 linear outputs, bfloat16 into fc2, v scaled by 1/64 into the half attention.
  Checked against stable-diffusion.cpp's pictures (same prompts and seed; the two draw their noise differently, so
  the scenes differ): the same subjects, composition quality and text ("OPEN" sign) - see the table below.
- **stable-diffusion.cpp removed (2026-11-13):** `src/sd.rs`, `native.rs`, `gguffix.rs`, `kernels/image/sd/`, the
  sd.cpp link in `kernels/build.sh` and `crates/sys`, `libstable-diffusion.so` from deploy. Its rows in the tables
  here and in image.md are history.
- **Not done:** `image check` stage dumps (no reference dumps exist for CUDA), int8 at the quality nextsycl documents (30.4 dB PSNR against half here, about 39 dB
  there; the picture looks right, the cause is not found), attention at nextsycl's ~110 TFLOPS (ldmatrix and
  cp.async not used yet, about 50 now).

### Rust engine clocks (RTX 5070, 8 steps guided, `nextcuda image bench`, 3 seeds, warm median)

| model | picture | stable-diffusion.cpp | Rust, half | Rust, int8 |
|---|---|---:|---:|---:|
| qwen-image-2.1 Q8_0 | 768x768 | 13.4 s | 11.2 s | 5.4 s |
| qwen-image-2.1 Q8_0 | 1024x1024 | 26.3 s | 20.2 s | 10.9 s |
| qwen-image-2.1 Q4 (rrq4) | 1024x1024 | - | - | 10.9 s |

Peak GPU memory with int8: 9.5 GB (768) and 10.6-11.1 GB (1024): close to the 12 GB limit. Edit with references:
45 s (sd.cpp 89 s); turbo LoRA at 4 steps 5.4 s. Matrices: `benchy-results/image-v1-5070-rust2` and `-rust-int8`.

### Qwen-Image 2512 Q4_K_M, Rust against stable-diffusion.cpp (RTX 3060 12 GB, PCIe x1, 1024 x 1024, seed 7, 2026-11-13)

| run | stable-diffusion.cpp | Rust | peak GPU memory (sd.cpp / Rust) |
|---|---:|---:|---:|
| Lightning, 4 steps (fox) | 177.9 s | 60.9 s (8.45 s a step) | 11.2 / 9.4 GB |
| Lightning, 4 steps (lighthouse with an OPEN sign) | 167.6 s | 59.5 s | 11.2 / 9.4 GB |
| full, 20 steps guided (fox) | 356.5 s | 265.7 s (11.95 s a step) | 11.4 / 8.0 GB |

Through the front door on the 5070 (x16): Lightning 4 steps at 1024 x 1024 31.1 s, peak 9.0 GiB; full 20 steps
guided at 1328 x 1328 172.8 s (8.6 s a step), peak 8.7 GiB; Qwen-Image 2.1 Q8_0 12 steps at 1024 x 1024 32.9 s.

The picture after the load; the load itself (13.2 GB from the HDD) is about 100 s for both. On the 3060's x1 link
most of a Lightning step is the 27 streamed blocks (5.9 GB); the 5070's x16 link moves them about 10x faster.

## Clocks (RTX 5070 12 GB, in process through stable-diffusion.cpp - removed since, 2026-10-15)

| model | picture | text encoder | sampler | VAE | total |
|---|---|---|---|---|---|
| qwen-image-2.1 Q8_0 | 768x768, 12 steps | 0.8-1.2 s | 1.27 s/step (16.5 s) | 2.4 s | 19.6 s warm, 30 s after a cold page-in |
| qwen-image-2.1, DiT resident on the card | same | 0.84 s | 1.27 s/step | 2.6 s | 22 s: the step is compute, not PCIe |
| qwen-image-2512 Q4_K_M, Lightning 4 steps | 1024x1024 | 0.8 s | 2.8 s/step | 4.7 s | 63 s (about 45 s of it the first graph streaming 13 GB in) |
| qwen-image-2512 Q4_K_M, full | 1024x1024, 24 steps | 0.8 s | 4.5 s/step | 4.7 s | 121 s |

The phase 1 worker made the 2.1 picture in 31.8 s; in process it is 19.6 s because the load is paid once. For
reference nextsycl's B70 makes Qwen-Image 2.1 at 1024x1024 in 0.6 s/step (0.41 with int8 weights), with about twice
the tokens of the 768 picture here: the 5070 has room for roughly 3x. Where it goes is the silo's first target
(the quantized matmuls and attention at 2,400-4,400 tokens).

## Phase 2: nextsycl's Rust-only runtime setup

Bring in how nextsycl runs engines without Python (the `engine` crates, `nextcuda-sys` over `dist/libnextcuda-<kind>.so`
built from CUDA kernels), one engine at a time behind the same contracts; the python worker for an engine goes when
its Rust one passes the same checks.

## nextsycl's contracts, declared ahead of their engines (2026-10-17)

Every API contract nextsycl has is in place here, so a client sees one API on both boxes. A method or engine whose
port has not landed answers with what it would take and does, and refuses the call with a reason:

- **Speech engines:** `audio/qwen3tts`, `audio/voxcpm2`, `audio/breezetts2` (`nextcuda audio engines`) carry nextsycl's
  registry entries, file roles, options and abilities (`speech()`), and their cards are in `model-cards/audio/`.
  Loading says "not ported to CUDA yet". The audio contract has nextsycl's `direction` and `events` (Breeze TTS 2's
  voice direction and vocal events), and `audio.speak` its Breeze fields and checks: `cfg`, a clone with instructions
  going to a model that directs, a transcript for a model that clones only with one.
- **3D:** `model3d/contract` and `crates/mesh` (GLB / OBJ and the CPU mesh tools) as nextsycl's. The `model3d.*`
  methods are in the spec with their fields, answers and errors, and answer 501 `not_implemented`.
- **Drives:** `system.drives`, `system.drive.measure`, `system.move` and `system.move.cancel` likewise: declared, 501.
- `system.version` lists the methods of the kinds that are on, so none of these show there until they work.

Then the same day: the three speech engines run on CUDA in Rust (no Python) - Breeze TTS 2, Qwen3-TTS and VoxCPM2,
nextsycl's code over CUDA ports of its audio kernels (`kernels/audio/`), each checked against its upstream package
(README "Speech"). Still declared only: model3d, the drives, music (MiniMax Music 3), OCR, video.

## Third-party C++ out (2026-11-13)

The goal: no third-party C or C++ in nextcuda, at run time or compiled in - nextcuda's own CUDA kernels and Rust.

- **Done: Nemotron's CPU experts in Rust** (`llm/nemotronh/src/cpu/`, replacing ggml-cpu's `libggml-base.so` /
  `libggml-cpu.so`, dlopened): the activation quantized byte for byte as ggml-cpu did, products within 1.4e-7 of the sum
  of |terms| of its; a layer's CPU share on the box's 8 threads 1.01-1.09x its speed. The libraries are gone from the
  box. (ggml-cpu's own path segfaulted in that decode - it was off by default.)
- **Done: Nemotron's products from the stored blocks** (`kernels/llm/nemotronh/nh_iq.cu` on `nh_blocks.cuh`,
  replacing Strata's `iq_kernels.cu` and ggml's `ggml-common.h`): expansions bit for bit, products within 3e-5 of the
  largest value; from DRAM on the 5070 the same or faster per type (Q6_K 13-41 %), several columns 1.5-3x.
- **Done: Qwen3.8's decode and verify products** (ggml-cuda's MMVQ before): the shared `nh_iq.cu` products with rows
  a warp group and K split over its warps, the shape per card and row count from timing; decode 60.9 tok/s, not 53.9,
  verify passes 8-20 % shorter, the same tokens. The opt-in verify products (`q35_vq8.cu`, `NS_Q35_VQ8`) read their
  blocks with `nh_blocks.cuh`, not `vecdotq.cuh`: the same speed or faster except IQ4_XS's 17408-long rows on the 5070
  (125 us, not 111).
- **Not done: Qwen3.8's prompt products** (ggml-cuda's MMQ, in `kernels/llm/qwen35/third_party/ggml-cuda`, which stays
  for them). nextcuda's own chunk products (`q35_tiles.cu`, `NS_Q35_TILES=1`, 2026-11-14) match or beat MMQ at 512
  input rows but not on short prompts. x is quantized a scale a 128 values so the sums stay integer within it (each
  mma's sums times the format's own integer scale; the K-quants' mins by dp2a against x's sums of 16), the weights
  decoded by producer warps while others multiply (named barriers; blocks staged by 16-byte cp.async, the grid
  codebooks in shared memory), ldmatrix fragments, 64-row tiles, K split where the tiles are too few. The sum of the
  model's matrices' times, own / MMQ: 512 rows 0.955 on the 5070, 1.009 on the 3060 (Q6_K 1.19 there); 128 rows 1.036 /
  0.984; 47 rows 1.42 / 1.33; 16 rows 1.16 / 1.35. Through the engine a 3119-token prompt 6 % faster (the same
  tokens), 16.6K 2.6 %; KL to MMQ's logits 0.00145, perplexity over 3000 tokens +0.25 % (a scale a 64: KL 0.00106, a
  32: 0.00076, each slower). Tried and dropped: a scale a 256 (faster, KL 0.00318, tokens changed), a 32's own power of
  two (slower), the int-to-float conversions as integer adds, deeper copy pipelines, 16 multiplying warps, MMQ's own
  loop shape (every warp decoding straight from global memory, 128-row tiles, 4 barriers a 256: no faster on short
  prompts, where the gap is: the decode of a 64-row tile is latency-bound when the step has few input rows).

