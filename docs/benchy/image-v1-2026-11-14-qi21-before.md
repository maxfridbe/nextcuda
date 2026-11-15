# nextcuda image bench, 2026-11-14

GPU 0 (NVIDIA GeForce RTX 5070); prompt: "a lighthouse on a rocky coast at dusk, long exposure, dramatic clouds, photorealistic"; 6 steps; seeds 1..3; 3 pictures a size, the first (cold) in brackets beside the warm median.

baseline

GPU 0 (NVIDIA GeForce RTX 5070): PCIe gen1 x16 (max gen4 x16) at the start of the run (the link steps down when a card is idle; each row says what it ran at)

GPU 1 (NVIDIA GeForce RTX 3060): PCIe gen1 x1 (max gen4 x16) at the start of the run (the link steps down when a card is idle; each row says what it ran at)

Stages in seconds; peaks of GPU memory are the model's own (the idle reading before the load taken off); energy and watts are the card's `power.draw` over the picture.

## qwen-image-2.1-turbo-q8 (Qwen-Image 2.1 Turbo Q8_0 (Viggle v0.3: 6 steps, no guidance; Rust + CUDA))

load 6.9 s; GPU memory after the load 8655 MiB (idle 32); engine: qwen-image-2.1 in Rust and CUDA on NVIDIA GeForce RTX 5070 (loaded 1 s ago)

| size | steps | text encode | step 1 | step (mean of 2..n) | sampler | VAE | total | peak GPU MiB | PCIe | host RSS MiB | energy J | avg W |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 1024x1024 | 6 | 0.52 (4.06) | 1.2 (1.2) | 1.17 (1.16) | 7.0 (7.0) | 0.9 (0.9) | 8.4 (11.9) | 11147 (10283) | g4 x16 | 4863 | 2003 (1978) | 235 (168) |
| 2048x2048 | 6 | 0.52 (0.45) | 6.6 (6.6) | 6.58 (6.57) | 39.5 (39.4) | 9.7 (9.7) | 49.7 (49.5) | 11191 (11189) | g4 x16 | 4896 | 12302 (12229) | 247 (247) |

# nextcuda image bench, 2026-11-15

GPU 1 (NVIDIA GeForce RTX 3060); prompt: "a lighthouse on a rocky coast at dusk, long exposure, dramatic clouds, photorealistic"; 6 steps; seeds 1..3; 3 pictures a size, the first (cold) in brackets beside the warm median.

baseline

GPU 0 (NVIDIA GeForce RTX 5070): PCIe gen1 x16 (max gen4 x16) at the start of the run (the link steps down when a card is idle; each row says what it ran at)

GPU 1 (NVIDIA GeForce RTX 3060): PCIe gen1 x1 (max gen4 x16) at the start of the run (the link steps down when a card is idle; each row says what it ran at)

Stages in seconds; peaks of GPU memory are the model's own (the idle reading before the load taken off); energy and watts are the card's `power.draw` over the picture.

## qwen-image-2.1-turbo-q8 (Qwen-Image 2.1 Turbo Q8_0 (Viggle v0.3: 6 steps, no guidance; Rust + CUDA))

load 7.8 s; GPU memory after the load 8594 MiB (idle 31); engine: qwen-image-2.1 in Rust and CUDA on NVIDIA GeForce RTX 3060 (loaded 1 s ago)

| size | steps | text encode | step 1 | step (mean of 2..n) | sampler | VAE | total | peak GPU MiB | PCIe | host RSS MiB | energy J | avg W |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 1024x1024 | 6 | 3.21 (4.40) | 3.0 (3.0) | 2.96 (2.96) | 17.8 (17.8) | 2.2 (2.2) | 23.1 (24.4) | 11091 (11109) | g4 x1 | 4734 | 3343 (3249) | 144 (134) |
| 2048x2048 | 6 | 3.24 (3.21) | 17.5 (17.4) | 17.48 (17.45) | 104.9 (104.6) | 24.4 (24.4) | 132.6 (132.2) | 11135 (11133) | g4 x1 | 4769 | 20485 (20238) | 155 (153) |

# nextcuda image bench, 2026-11-15

GPU 0 (NVIDIA GeForce RTX 5070); prompt: "a lighthouse on a rocky coast at dusk, long exposure, dramatic clouds, photorealistic"; 8 steps; seeds 1..3; 3 pictures a size, the first (cold) in brackets beside the warm median.

base-full-0-1024

GPU 0 (NVIDIA GeForce RTX 5070): PCIe gen4 x16 (max gen4 x16) at the start of the run (the link steps down when a card is idle; each row says what it ran at)

GPU 1 (NVIDIA GeForce RTX 3060): PCIe gen4 x1 (max gen4 x16) at the start of the run (the link steps down when a card is idle; each row says what it ran at)

Stages in seconds; peaks of GPU memory are the model's own (the idle reading before the load taken off); energy and watts are the card's `power.draw` over the picture.

## qwen-image-2.1-q8 (Qwen-Image 2.1 Q8_0 (text to image, RGBA, edits; stable-diffusion.cpp on CUDA))

load 4.7 s; GPU memory after the load 8655 MiB (idle 32); engine: qwen-image-2.1 in Rust and CUDA on NVIDIA GeForce RTX 5070 (loaded 1 s ago)

| size | steps | text encode | step 1 | step (mean of 2..n) | sampler | VAE | total | peak GPU MiB | PCIe | host RSS MiB | energy J | avg W |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 1024x1024 | 8 | 0.53 (2.09) | 2.4 (2.4) | 2.37 (2.36) | 18.9 (18.9) | 0.9 (0.9) | 20.4 (21.9) | 10283 (11147) | g4 x16 | 4864 | 5067 (5052) | 247 (229) |

# nextcuda image bench, 2026-11-15

GPU 0 (NVIDIA GeForce RTX 5070); prompt: "a lighthouse on a rocky coast at dusk, long exposure, dramatic clouds, photorealistic"; 8 steps; seeds 1..3; 3 pictures a size, the first (cold) in brackets beside the warm median.

base-full-0-2048

GPU 0 (NVIDIA GeForce RTX 5070): PCIe gen4 x16 (max gen4 x16) at the start of the run (the link steps down when a card is idle; each row says what it ran at)

GPU 1 (NVIDIA GeForce RTX 3060): PCIe gen1 x1 (max gen4 x16) at the start of the run (the link steps down when a card is idle; each row says what it ran at)

Stages in seconds; peaks of GPU memory are the model's own (the idle reading before the load taken off); energy and watts are the card's `power.draw` over the picture.

## qwen-image-2.1-q8 (Qwen-Image 2.1 Q8_0 (text to image, RGBA, edits; stable-diffusion.cpp on CUDA))

load 1.9 s; GPU memory after the load 8655 MiB (idle 32); engine: qwen-image-2.1 in Rust and CUDA on NVIDIA GeForce RTX 5070 (loaded 1 s ago)

| size | steps | text encode | step 1 | step (mean of 2..n) | sampler | VAE | total | peak GPU MiB | PCIe | host RSS MiB | energy J | avg W |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 2048x2048 | 8 | 0.53 (1.95) | 13.2 (13.2) | 13.18 (13.18) | 105.4 (105.4) | 9.8 (9.8) | 115.8 (117.2) | 11189 (11189) | g4 x16 | 4903 | 29090 (29046) | 251 (248) |

# nextcuda image bench, 2026-11-15

GPU 1 (NVIDIA GeForce RTX 3060); prompt: "a lighthouse on a rocky coast at dusk, long exposure, dramatic clouds, photorealistic"; 8 steps; seeds 1..3; 3 pictures a size, the first (cold) in brackets beside the warm median.

base-full-1-1024

GPU 0 (NVIDIA GeForce RTX 5070): PCIe gen4 x16 (max gen4 x16) at the start of the run (the link steps down when a card is idle; each row says what it ran at)

GPU 1 (NVIDIA GeForce RTX 3060): PCIe gen1 x1 (max gen4 x16) at the start of the run (the link steps down when a card is idle; each row says what it ran at)

Stages in seconds; peaks of GPU memory are the model's own (the idle reading before the load taken off); energy and watts are the card's `power.draw` over the picture.

## qwen-image-2.1-q8 (Qwen-Image 2.1 Q8_0 (text to image, RGBA, edits; stable-diffusion.cpp on CUDA))

load 7.0 s; GPU memory after the load 8594 MiB (idle 31); engine: qwen-image-2.1 in Rust and CUDA on NVIDIA GeForce RTX 3060 (loaded 1 s ago)

| size | steps | text encode | step 1 | step (mean of 2..n) | sampler | VAE | total | peak GPU MiB | PCIe | host RSS MiB | energy J | avg W |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 1024x1024 | 8 | 3.39 (7.16) | 5.9 (5.9) | 5.96 (5.89) | 47.6 (47.1) | 2.2 (2.2) | 53.2 (56.5) | 11091 (11111) | g4 x1 | 4738 | 8291 (8146) | 156 (144) |

# nextcuda image bench, 2026-11-15

GPU 1 (NVIDIA GeForce RTX 3060); prompt: "a lighthouse on a rocky coast at dusk, long exposure, dramatic clouds, photorealistic"; 8 steps; seeds 1..3; 3 pictures a size, the first (cold) in brackets beside the warm median.

base-full-1-2048

GPU 0 (NVIDIA GeForce RTX 5070): PCIe gen4 x16 (max gen4 x16) at the start of the run (the link steps down when a card is idle; each row says what it ran at)

GPU 1 (NVIDIA GeForce RTX 3060): PCIe gen1 x1 (max gen4 x16) at the start of the run (the link steps down when a card is idle; each row says what it ran at)

Stages in seconds; peaks of GPU memory are the model's own (the idle reading before the load taken off); energy and watts are the card's `power.draw` over the picture.

## qwen-image-2.1-q8 (Qwen-Image 2.1 Q8_0 (text to image, RGBA, edits; stable-diffusion.cpp on CUDA))

load 6.9 s; GPU memory after the load 8594 MiB (idle 31); engine: qwen-image-2.1 in Rust and CUDA on NVIDIA GeForce RTX 3060 (loaded 1 s ago)

| size | steps | text encode | step 1 | step (mean of 2..n) | sampler | VAE | total | peak GPU MiB | PCIe | host RSS MiB | energy J | avg W |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 2048x2048 | 8 | 3.48 (7.17) | 34.9 (34.7) | 35.07 (34.94) | 280.4 (279.3) | 24.5 (24.5) | 308.4 (311.0) | 11133 (11153) | g4 x1 | 4810 | 48675 (48445) | 158 (156) |

