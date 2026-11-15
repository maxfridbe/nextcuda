# nextcuda image bench, 2026-11-15

GPU 0 (NVIDIA GeForce RTX 5070); prompt: "a lighthouse on a rocky coast at dusk, long exposure, dramatic clouds, photorealistic"; 6 steps; seeds 1..3; 3 pictures a size, the first (cold) in brackets beside the warm median.

dev-turbo-0-1024

GPU 0 (NVIDIA GeForce RTX 5070): PCIe gen1 x16 (max gen4 x16) at the start of the run (the link steps down when a card is idle; each row says what it ran at)

GPU 1 (NVIDIA GeForce RTX 3060): PCIe gen4 x1 (max gen4 x16) at the start of the run (the link steps down when a card is idle; each row says what it ran at)

Stages in seconds; peaks of GPU memory are the model's own (the idle reading before the load taken off); energy and watts are the card's `power.draw` over the picture.

## qwen-image-2.1-turbo-q8 (Qwen-Image 2.1 Turbo Q8_0 (Viggle v0.3: 6 steps, no guidance; Rust + CUDA))

load 2.4 s; GPU memory after the load 8527 MiB (idle 32); engine: qwen-image-2.1 in Rust and CUDA on NVIDIA GeForce RTX 5070 (loaded 1 s ago)

| size | steps | text encode | step 1 | step (mean of 2..n) | sampler | VAE | total | peak GPU MiB | PCIe | host RSS MiB | energy J | avg W |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 1024x1024 | 6 | 0.20 (2.30) | 0.8 (0.8) | 0.81 (0.81) | 4.9 (4.8) | 0.3 (0.4) | 5.4 (7.5) | 10829 (9387) | g4 x16 | 4878 | 1332 (1312) | 242 (175) |

# nextcuda image bench, 2026-11-15

GPU 0 (NVIDIA GeForce RTX 5070); prompt: "a lighthouse on a rocky coast at dusk, long exposure, dramatic clouds, photorealistic"; 6 steps; seeds 1..3; 3 pictures a size, the first (cold) in brackets beside the warm median.

dev-turbo-0-2048

GPU 0 (NVIDIA GeForce RTX 5070): PCIe gen1 x16 (max gen4 x16) at the start of the run (the link steps down when a card is idle; each row says what it ran at)

GPU 1 (NVIDIA GeForce RTX 3060): PCIe gen4 x1 (max gen4 x16) at the start of the run (the link steps down when a card is idle; each row says what it ran at)

Stages in seconds; peaks of GPU memory are the model's own (the idle reading before the load taken off); energy and watts are the card's `power.draw` over the picture.

## qwen-image-2.1-turbo-q8 (Qwen-Image 2.1 Turbo Q8_0 (Viggle v0.3: 6 steps, no guidance; Rust + CUDA))

load 2.4 s; GPU memory after the load 8527 MiB (idle 32); engine: qwen-image-2.1 in Rust and CUDA on NVIDIA GeForce RTX 5070 (loaded 1 s ago)

| size | steps | text encode | step 1 | step (mean of 2..n) | sampler | VAE | total | peak GPU MiB | PCIe | host RSS MiB | energy J | avg W |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 2048x2048 | 6 | 0.19 (2.27) | 4.4 (4.4) | 4.43 (4.41) | 26.6 (26.4) | 1.7 (1.7) | 28.5 (30.4) | 10383 (10289) | g4 x16 | 4906 | 7187 (7158) | 253 (234) |

# nextcuda image bench, 2026-11-15

GPU 1 (NVIDIA GeForce RTX 3060); prompt: "a lighthouse on a rocky coast at dusk, long exposure, dramatic clouds, photorealistic"; 6 steps; seeds 1..3; 3 pictures a size, the first (cold) in brackets beside the warm median.

dev-turbo-1-1024

GPU 0 (NVIDIA GeForce RTX 5070): PCIe gen4 x16 (max gen4 x16) at the start of the run (the link steps down when a card is idle; each row says what it ran at)

GPU 1 (NVIDIA GeForce RTX 3060): PCIe gen1 x1 (max gen4 x16) at the start of the run (the link steps down when a card is idle; each row says what it ran at)

Stages in seconds; peaks of GPU memory are the model's own (the idle reading before the load taken off); energy and watts are the card's `power.draw` over the picture.

## qwen-image-2.1-turbo-q8 (Qwen-Image 2.1 Turbo Q8_0 (Viggle v0.3: 6 steps, no guidance; Rust + CUDA))

load 7.3 s; GPU memory after the load 8594 MiB (idle 31); engine: qwen-image-2.1 in Rust and CUDA on NVIDIA GeForce RTX 3060 (loaded 1 s ago)

| size | steps | text encode | step 1 | step (mean of 2..n) | sampler | VAE | total | peak GPU MiB | PCIe | host RSS MiB | energy J | avg W |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 1024x1024 | 6 | 2.86 (4.75) | 2.3 (2.2) | 2.27 (2.27) | 13.6 (13.6) | 1.0 (1.1) | 17.5 (19.4) | 10965 (10099) | g4 x1 | 4739 | 2780 (2767) | 156 (143) |

# nextcuda image bench, 2026-11-15

GPU 1 (NVIDIA GeForce RTX 3060); prompt: "a lighthouse on a rocky coast at dusk, long exposure, dramatic clouds, photorealistic"; 6 steps; seeds 1..3; 3 pictures a size, the first (cold) in brackets beside the warm median.

dev-turbo-1-2048

GPU 0 (NVIDIA GeForce RTX 5070): PCIe gen1 x16 (max gen4 x16) at the start of the run (the link steps down when a card is idle; each row says what it ran at)

GPU 1 (NVIDIA GeForce RTX 3060): PCIe gen4 x1 (max gen4 x16) at the start of the run (the link steps down when a card is idle; each row says what it ran at)

Stages in seconds; peaks of GPU memory are the model's own (the idle reading before the load taken off); energy and watts are the card's `power.draw` over the picture.

## qwen-image-2.1-turbo-q8 (Qwen-Image 2.1 Turbo Q8_0 (Viggle v0.3: 6 steps, no guidance; Rust + CUDA))

load 7.3 s; GPU memory after the load 8594 MiB (idle 31); engine: qwen-image-2.1 in Rust and CUDA on NVIDIA GeForce RTX 3060 (loaded 1 s ago)

| size | steps | text encode | step 1 | step (mean of 2..n) | sampler | VAE | total | peak GPU MiB | PCIe | host RSS MiB | energy J | avg W |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 2048x2048 | 6 | 2.86 (4.73) | 12.0 (11.9) | 12.04 (11.99) | 72.2 (71.8) | 5.1 (5.0) | 80.2 (81.6) | 10429 (10615) | g4 x1 | 4770 | 13782 (13697) | 171 (168) |

# nextcuda image bench, 2026-11-15

GPU 0 (NVIDIA GeForce RTX 5070); prompt: "a lighthouse on a rocky coast at dusk, long exposure, dramatic clouds, photorealistic"; 8 steps; seeds 1..3; 3 pictures a size, the first (cold) in brackets beside the warm median.

new-full-0-1024

GPU 0 (NVIDIA GeForce RTX 5070): PCIe gen1 x16 (max gen4 x16) at the start of the run (the link steps down when a card is idle; each row says what it ran at)

GPU 1 (NVIDIA GeForce RTX 3060): PCIe gen4 x1 (max gen4 x16) at the start of the run (the link steps down when a card is idle; each row says what it ran at)

Stages in seconds; peaks of GPU memory are the model's own (the idle reading before the load taken off); energy and watts are the card's `power.draw` over the picture.

## qwen-image-2.1-q8 (Qwen-Image 2.1 Q8_0 (text to image, RGBA, edits; stable-diffusion.cpp on CUDA))

load 5.1 s; GPU memory after the load 8527 MiB (idle 32); engine: qwen-image-2.1 in Rust and CUDA on NVIDIA GeForce RTX 5070 (loaded 1 s ago)

| size | steps | text encode | step 1 | step (mean of 2..n) | sampler | VAE | total | peak GPU MiB | PCIe | host RSS MiB | energy J | avg W |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 1024x1024 | 8 | 0.24 (4.50) | 1.7 (1.6) | 1.67 (1.65) | 13.4 (13.2) | 0.4 (0.4) | 14.0 (18.1) | 10833 (9967) | g4 x16 | 4873 | 3505 (3588) | 250 (196) |

# nextcuda image bench, 2026-11-15

GPU 0 (NVIDIA GeForce RTX 5070); prompt: "a lighthouse on a rocky coast at dusk, long exposure, dramatic clouds, photorealistic"; 8 steps; seeds 1..3; 3 pictures a size, the first (cold) in brackets beside the warm median.

new-full-0-2048

GPU 0 (NVIDIA GeForce RTX 5070): PCIe gen1 x16 (max gen4 x16) at the start of the run (the link steps down when a card is idle; each row says what it ran at)

GPU 1 (NVIDIA GeForce RTX 3060): PCIe gen4 x1 (max gen4 x16) at the start of the run (the link steps down when a card is idle; each row says what it ran at)

Stages in seconds; peaks of GPU memory are the model's own (the idle reading before the load taken off); energy and watts are the card's `power.draw` over the picture.

## qwen-image-2.1-q8 (Qwen-Image 2.1 Q8_0 (text to image, RGBA, edits; stable-diffusion.cpp on CUDA))

load 2.3 s; GPU memory after the load 8527 MiB (idle 32); engine: qwen-image-2.1 in Rust and CUDA on NVIDIA GeForce RTX 5070 (loaded 1 s ago)

| size | steps | text encode | step 1 | step (mean of 2..n) | sampler | VAE | total | peak GPU MiB | PCIe | host RSS MiB | energy J | avg W |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 2048x2048 | 8 | 0.24 (2.52) | 9.0 (8.9) | 8.98 (8.96) | 71.8 (71.6) | 1.9 (1.9) | 74.0 (76.1) | 10387 (9965) | g4 x16 | 4904 | 18792 (18798) | 254 (247) |

# nextcuda image bench, 2026-11-15

GPU 1 (NVIDIA GeForce RTX 3060); prompt: "a lighthouse on a rocky coast at dusk, long exposure, dramatic clouds, photorealistic"; 8 steps; seeds 1..3; 3 pictures a size, the first (cold) in brackets beside the warm median.

new-full-1-1024

GPU 0 (NVIDIA GeForce RTX 5070): PCIe gen4 x16 (max gen4 x16) at the start of the run (the link steps down when a card is idle; each row says what it ran at)

GPU 1 (NVIDIA GeForce RTX 3060): PCIe gen1 x1 (max gen4 x16) at the start of the run (the link steps down when a card is idle; each row says what it ran at)

Stages in seconds; peaks of GPU memory are the model's own (the idle reading before the load taken off); energy and watts are the card's `power.draw` over the picture.

## qwen-image-2.1-q8 (Qwen-Image 2.1 Q8_0 (text to image, RGBA, edits; stable-diffusion.cpp on CUDA))

load 7.2 s; GPU memory after the load 8594 MiB (idle 31); engine: qwen-image-2.1 in Rust and CUDA on NVIDIA GeForce RTX 3060 (loaded 1 s ago)

| size | steps | text encode | step 1 | step (mean of 2..n) | sampler | VAE | total | peak GPU MiB | PCIe | host RSS MiB | energy J | avg W |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 1024x1024 | 8 | 3.08 (7.85) | 4.6 (4.5) | 4.57 (4.55) | 36.6 (36.4) | 1.1 (1.1) | 40.8 (45.4) | 10969 (10967) | g4 x1 | 4742 | 6872 (7059) | 168 (155) |

# nextcuda image bench, 2026-11-15

GPU 1 (NVIDIA GeForce RTX 3060); prompt: "a lighthouse on a rocky coast at dusk, long exposure, dramatic clouds, photorealistic"; 8 steps; seeds 1..3; 3 pictures a size, the first (cold) in brackets beside the warm median.

new-full-1-2048

GPU 0 (NVIDIA GeForce RTX 5070): PCIe gen4 x16 (max gen4 x16) at the start of the run (the link steps down when a card is idle; each row says what it ran at)

GPU 1 (NVIDIA GeForce RTX 3060): PCIe gen1 x1 (max gen4 x16) at the start of the run (the link steps down when a card is idle; each row says what it ran at)

Stages in seconds; peaks of GPU memory are the model's own (the idle reading before the load taken off); energy and watts are the card's `power.draw` over the picture.

## qwen-image-2.1-q8 (Qwen-Image 2.1 Q8_0 (text to image, RGBA, edits; stable-diffusion.cpp on CUDA))

load 7.2 s; GPU memory after the load 8594 MiB (idle 31); engine: qwen-image-2.1 in Rust and CUDA on NVIDIA GeForce RTX 3060 (loaded 1 s ago)

| size | steps | text encode | step 1 | step (mean of 2..n) | sampler | VAE | total | peak GPU MiB | PCIe | host RSS MiB | energy J | avg W |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 2048x2048 | 8 | 3.08 (7.52) | 24.1 (23.9) | 24.12 (24.06) | 192.9 (192.2) | 5.5 (5.5) | 201.5 (205.3) | 10619 (10045) | g4 x1 | 4773 | 35302 (35389) | 175 (172) |

