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
  there; the picture looks right, the cause is not found), attention at nextsycl's ~110 TFLOPS (about 88 with half
  sums since 2026-11-14, below).

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

## Qwen-Image 2.1 speed (2026-11-14)

The step was compute at ~80 % of what the cards do with float32 sums: the 5070's mma.sync peaks at 71.5 TFLOPS with
float32 accumulators and 143 with half ones (the 3060: 25.8 / 51.6), and cuBLAS already ran the linears at 66-70. So
the speed is in half sums, kept short: each 32-deep slice of a product summed in half on the tensor cores and the
slices added in float32 (`NS_QI_F16ACC`) - errors at the level of the output's own rounding to half.

- **Done:** `silo/gemm.cu` (our own: cp.async 3-4 stages, ldmatrix from swizzled shared memory, a 128 x 128 tile of 4
  warps on the 5070, 128 x 256 of 8 on the 3060) for the DiT's linears - 103-106 TFLOPS on the 5070 against cuBLAS's
  70, 34 on the 3060 against 27; Q8_0 read as int8 + scales by the tensor cores' fragments on the 5070 (no dequant
  pass); the gated residual adds in the out / down epilogues. The VAE's 3 x 3 convolutions as an implicit GEMM through
  it (the 2x upsample folded into the gather; 3.1 -> 1.3 s of conv at 2048 x 2048; a 5-D weight shape had kept most
  of them on im2col at first), and its tiles cut after the first two up blocks (2.7x -> 1.2x the area). Attention:
  `flash2_k`, 128 rows a block, cp.async K / V, ldmatrix(.trans), half sums in P V (Q K^T's too with `NS_QI_QK16=1`: off, it moved a 3060
  picture's fine texture - mean |difference| 0.31 of 255 against float32 sums' 0.14); on the
  model's own q, k, v against float64, relative rms 2.2e-4 (float32 sums), 2.5e-4 (P V half), 5e-4 - 1.3e-3 (both);
  82.6 -> ~61 ms a call at 2048 x 2048 (~50 with Q K^T in half too). The text encoder's layers pinned and copied a layer ahead (5070 0.48 -> 0.19 s;
  the 3060's x1 link 3.2 -> 2.7 s, the link's own speed).
- **Clocks** ([`image.md`](image.md)): the 2048 x 2048 turbo picture 49.7 -> 30.1 s on the 5070, 132.5 -> 85.0 s on
  the 3060; 1024 x 1024 8.4 -> 5.5 s and 23.1 -> 17.8 s; the full model at 2048 x 2048 115.7 -> 77.0 s and 308 -> 214 s.
  Not the half asked for: at the half rate a step is ~70 % of the tensor cores' peak already.
- **Tried, no gain (left out):** the half sums flushed every 64 or 128 deep (4 and 8 steps: no faster); tile orders of
  4 / 16 row tiles; `__launch_bounds__(.., 2)` (spills on sm_86: 10x slower); 128 x 256, 256 x 128 and BK 64 tiles on
  the 5070 (84-98 TFLOPS); Q8_0 fragments on the 3060 (31 against its 33.5 TFLOPS on dequantized slices); masking
  only edge tiles / max on raw scores / rescale skipped when no row's maximum moved in the attention (within noise:
  kept, it is not slower).
- **Not done:** the text encoder resident on the card (the 7.3 GB DiT and its 2048 x 2048 activations leave the 12 GB
  cards ~1.5 GB: 7 of 36 layers, ~0.5 s on the 3060, against a no_room risk at 2048 x 2048); the swiglu in the gate|up
  product's epilogue (gate and up rows would have to be interleaved, LoRA slots with them; ~2 %).

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
- **Done: Qwen3.8's prompt products** (2026-11-14; ggml-cuda's MMQ before). `q35_tiles.cu`: x a scale a 128 values,
  integer sums within it (the formats' own integer scales; the K-quants' mins by dp2a), ldmatrix fragments. Short
  prompts (64 rows or fewer) and the grid i-quants through MMQ's structure, ported (mq_k: 128-row tiles, every warp
  decoding straight from global memory, mmq_x as MMQ picks it, stream-k with fixup, load_tiles' word-a-lane reads for
  IQ4_XS / Q6_K), two blocks an SM; the rest through a producer / multiplier pipeline (kmm_k: producer warps stage and
  decode, named barriers). What closed the short prompts' gap: each thread's 4 pieces' loads in flight together (the
  blocks' alignment chosen once a chunk, not a branch a piece: 1.21 -> 1.01), x read a half ahead into registers, the
  fixup laid out along y's rows (its scattered writes had cost 24 us on one matrix). The model's matrices' summed
  time against MMQ's: 16 rows 0.984 / 0.953 (5070 / 3060), 47 0.970 / 0.920, 128 0.982 / 0.991, 512 0.951 / 0.989
  (Q6_K on the 3060 at 512 still 1.2). Engine: 181 tokens 461 tok/s against 382, 3119 1168 against 1028, 16.6K 1343
  against 1266; KL to MMQ's logits 0.0015, perplexity over 3000 tokens 7.582 against 7.575.
- **Done: third-party C++ out of nextcuda.** `kernels/llm/qwen35/third_party/ggml-cuda` removed (2026-11-14): the
  `ns_q35_mmq*` / `ns_q35_mmvq` entry points are q35_tiles.cu's and nh_iq.cu's; `NS_Q35_TILES` and `NS_Q35_MMVQ` are
  gone with it.

