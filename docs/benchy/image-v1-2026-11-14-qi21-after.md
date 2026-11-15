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
| 1024x1024 | 6 | 0.19 (2.17) | 0.8 (0.8) | 0.83 (0.82) | 5.0 (4.9) | 0.4 (0.3) | 5.5 (7.4) | 10829 (9963) | g4 x16 | 4875 | 1322 (1309) | 240 (174) |

# nextcuda image bench, 2026-11-15

GPU 0 (NVIDIA GeForce RTX 5070); prompt: "a lighthouse on a rocky coast at dusk, long exposure, dramatic clouds, photorealistic"; 6 steps; seeds 1..3; 3 pictures a size, the first (cold) in brackets beside the warm median.

dev-turbo-0-2048

GPU 0 (NVIDIA GeForce RTX 5070): PCIe gen4 x16 (max gen4 x16) at the start of the run (the link steps down when a card is idle; each row says what it ran at)

GPU 1 (NVIDIA GeForce RTX 3060): PCIe gen1 x1 (max gen4 x16) at the start of the run (the link steps down when a card is idle; each row says what it ran at)

Stages in seconds; peaks of GPU memory are the model's own (the idle reading before the load taken off); energy and watts are the card's `power.draw` over the picture.

## qwen-image-2.1-turbo-q8 (Qwen-Image 2.1 Turbo Q8_0 (Viggle v0.3: 6 steps, no guidance; Rust + CUDA))

load 2.5 s; GPU memory after the load 8527 MiB (idle 32); engine: qwen-image-2.1 in Rust and CUDA on NVIDIA GeForce RTX 5070 (loaded 1 s ago)

| size | steps | text encode | step 1 | step (mean of 2..n) | sampler | VAE | total | peak GPU MiB | PCIe | host RSS MiB | energy J | avg W |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 2048x2048 | 6 | 0.19 (2.12) | 4.7 (4.7) | 4.70 (4.68) | 28.2 (28.1) | 1.7 (1.7) | 30.1 (31.9) | 10063 (9905) | g4 x16 | 4905 | 7622 (7538) | 252 (235) |

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
| 1024x1024 | 6 | 2.85 (4.73) | 2.3 (2.3) | 2.31 (2.30) | 13.9 (13.8) | 1.1 (1.0) | 17.8 (19.6) | 10965 (10099) | g4 x1 | 4746 | 2775 (2822) | 156 (143) |

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
| 2048x2048 | 6 | 2.86 (4.74) | 12.9 (12.7) | 12.86 (12.80) | 77.1 (76.7) | 5.1 (5.1) | 85.1 (86.5) | 10441 (10523) | g4 x1 | 4814 | 14432 (14333) | 169 (166) |

# nextcuda image bench, 2026-11-15

GPU 0 (NVIDIA GeForce RTX 5070); prompt: "a lighthouse on a rocky coast at dusk, long exposure, dramatic clouds, photorealistic"; 8 steps; seeds 1..3; 3 pictures a size, the first (cold) in brackets beside the warm median.

dev-full-0-1024

GPU 0 (NVIDIA GeForce RTX 5070): PCIe gen1 x16 (max gen4 x16) at the start of the run (the link steps down when a card is idle; each row says what it ran at)

GPU 1 (NVIDIA GeForce RTX 3060): PCIe gen4 x1 (max gen4 x16) at the start of the run (the link steps down when a card is idle; each row says what it ran at)

Stages in seconds; peaks of GPU memory are the model's own (the idle reading before the load taken off); energy and watts are the card's `power.draw` over the picture.

## qwen-image-2.1-q8 (Qwen-Image 2.1 Q8_0 (text to image, RGBA, edits; stable-diffusion.cpp on CUDA))

load 2.4 s; GPU memory after the load 8527 MiB (idle 32); engine: qwen-image-2.1 in Rust and CUDA on NVIDIA GeForce RTX 5070 (loaded 1 s ago)

| size | steps | text encode | step 1 | step (mean of 2..n) | sampler | VAE | total | peak GPU MiB | PCIe | host RSS MiB | energy J | avg W |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 1024x1024 | 8 | 0.23 (2.18) | 1.7 (1.6) | 1.67 (1.65) | 13.4 (13.2) | 0.3 (0.4) | 14.0 (15.7) | 10831 (9967) | g4 x16 | 4876 | 3494 (3484) | 249 (221) |

# nextcuda image bench, 2026-11-15

GPU 0 (NVIDIA GeForce RTX 5070); prompt: "a lighthouse on a rocky coast at dusk, long exposure, dramatic clouds, photorealistic"; 8 steps; seeds 1..3; 3 pictures a size, the first (cold) in brackets beside the warm median.

dev-full-0-2048

GPU 0 (NVIDIA GeForce RTX 5070): PCIe gen4 x16 (max gen4 x16) at the start of the run (the link steps down when a card is idle; each row says what it ran at)

GPU 1 (NVIDIA GeForce RTX 3060): PCIe gen1 x1 (max gen4 x16) at the start of the run (the link steps down when a card is idle; each row says what it ran at)

Stages in seconds; peaks of GPU memory are the model's own (the idle reading before the load taken off); energy and watts are the card's `power.draw` over the picture.

## qwen-image-2.1-q8 (Qwen-Image 2.1 Q8_0 (text to image, RGBA, edits; stable-diffusion.cpp on CUDA))

load 2.4 s; GPU memory after the load 8527 MiB (idle 32); engine: qwen-image-2.1 in Rust and CUDA on NVIDIA GeForce RTX 5070 (loaded 1 s ago)

| size | steps | text encode | step 1 | step (mean of 2..n) | sampler | VAE | total | peak GPU MiB | PCIe | host RSS MiB | energy J | avg W |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 2048x2048 | 8 | 0.24 (2.18) | 9.4 (9.4) | 9.39 (9.40) | 75.1 (75.2) | 1.7 (1.7) | 77.1 (79.1) | 10391 (10293) | g4 x16 | 4907 | 19519 (19532) | 253 (247) |

# nextcuda image bench, 2026-11-15

GPU 1 (NVIDIA GeForce RTX 3060); prompt: "a lighthouse on a rocky coast at dusk, long exposure, dramatic clouds, photorealistic"; 8 steps; seeds 1..3; 3 pictures a size, the first (cold) in brackets beside the warm median.

dev-full-1-1024

GPU 0 (NVIDIA GeForce RTX 5070): PCIe gen4 x16 (max gen4 x16) at the start of the run (the link steps down when a card is idle; each row says what it ran at)

GPU 1 (NVIDIA GeForce RTX 3060): PCIe gen1 x1 (max gen4 x16) at the start of the run (the link steps down when a card is idle; each row says what it ran at)

Stages in seconds; peaks of GPU memory are the model's own (the idle reading before the load taken off); energy and watts are the card's `power.draw` over the picture.

## qwen-image-2.1-q8 (Qwen-Image 2.1 Q8_0 (text to image, RGBA, edits; stable-diffusion.cpp on CUDA))

load 7.1 s; GPU memory after the load 8594 MiB (idle 31); engine: qwen-image-2.1 in Rust and CUDA on NVIDIA GeForce RTX 3060 (loaded 1 s ago)

| size | steps | text encode | step 1 | step (mean of 2..n) | sampler | VAE | total | peak GPU MiB | PCIe | host RSS MiB | energy J | avg W |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 1024x1024 | 8 | 3.08 (7.64) | 4.7 (4.6) | 4.67 (4.61) | 37.3 (36.9) | 1.1 (1.1) | 41.5 (45.6) | 10105 (10103) | g4 x1 | 4740 | 6943 (7084) | 168 (155) |

# nextcuda image bench, 2026-11-15

GPU 1 (NVIDIA GeForce RTX 3060); prompt: "a lighthouse on a rocky coast at dusk, long exposure, dramatic clouds, photorealistic"; 8 steps; seeds 1..3; 3 pictures a size, the first (cold) in brackets beside the warm median.

dev-full-1-2048

GPU 0 (NVIDIA GeForce RTX 5070): PCIe gen1 x16 (max gen4 x16) at the start of the run (the link steps down when a card is idle; each row says what it ran at)

GPU 1 (NVIDIA GeForce RTX 3060): PCIe gen4 x1 (max gen4 x16) at the start of the run (the link steps down when a card is idle; each row says what it ran at)

Stages in seconds; peaks of GPU memory are the model's own (the idle reading before the load taken off); energy and watts are the card's `power.draw` over the picture.

## qwen-image-2.1-q8 (Qwen-Image 2.1 Q8_0 (text to image, RGBA, edits; stable-diffusion.cpp on CUDA))

load 7.2 s; GPU memory after the load 8594 MiB (idle 31); engine: qwen-image-2.1 in Rust and CUDA on NVIDIA GeForce RTX 3060 (loaded 1 s ago)

| size | steps | text encode | step 1 | step (mean of 2..n) | sampler | VAE | total | peak GPU MiB | PCIe | host RSS MiB | energy J | avg W |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 2048x2048 | 8 | 3.08 (7.80) | 25.8 (25.6) | 25.77 (25.73) | 206.1 (205.6) | 5.1 (5.1) | 214.3 (218.5) | 10527 (10617) | g4 x1 | 4771 | 37068 (37146) | 173 (170) |

