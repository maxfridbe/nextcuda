# svgrender: PyTorch-SVGRender's technique in Rust and CUDA

Upstream: [PyTorch-SVGRender](https://github.com/ximinng/PyTorch-SVGRender) (MPL-2.0; checked at dadfbba) and
[DiffVG](https://github.com/BachiLi/diffvg) (Apache-2.0; 85802a7). Card `svgrender-live` (no weights: `files: {}`),
engine `image/svgrender`, kernels `kernels/image/svgrender/svg.cu`. Methods `image.svg_fit` and `image.svg_generate`
([image.md](../image.md#svg-paths-svgrender-diffvg--live)).

## What was ported

| upstream | here | |
|---|---|---|
| DiffVG `sample_color_prefiltered` + its backward | `svg.cu` `pixel<BACK>` | cubic Bezier paths, open and closed; fill (even-odd or nonzero winding), stroke (half-width), straight-alpha compositing back to front; one thread a pixel, fragments kept for the backward pass (an opaque fragment clears what is under it: its gradient is exactly zero) |
| DiffVG `compute_distance` / `closest_point` (cubic) | `closest_t` | 16 samples then Newton on (B(t) - q) . B'(t) from every sampled minimum, instead of isolating the quintic's roots: the same point |
| DiffVG `compute_winding_number` (cubic) | `seg_winding` | the cubic formula as DiffVG's, in float32, roots polished by Newton; [0, 1) half open so a crossing through a joint counts once |
| DiffVG `d_compute_distance` | `d_distance` | the envelope form (DiffVG's implicit-t term is ~0 at a stationary point) |
| DiffVG sampled render (forward) | `pixel_hard` | s x s regular samples a pixel (DiffVG jitters them with a seed): previews, LIVE's 2 x 2 coverage |
| LIVE `Painter`: circle init, `SparseCoordInit` | `live.rs` | the error quantised into 200 quantiles, the biggest 4-connected region of the most common level, the pixel nearest its centroid |
| LIVE `calc_distance_weight` / `get_sdf` | `udf_dist_k`, `udf_weight_k` | distance to the nearest pixel of the other sign less half a pixel (skfmm marches the interpolated level set) |
| LIVE `xing_loss_fn` | `xing_k` | value and analytic gradient |
| LIVE `PainterOptimizer` (Adam + `LinearDecayLR` per layer) | `adam_k` | every path's step and bias correction counted from the layer that added it, as the per-layer optimisers do |
| VectorFusion's styles (`painter/vectorfusion`) | `Style` | iconography, pixelart (a grid of squares, colours only), sketch (black strokes, alpha only), painting (coloured strokes, width trained) |
| VectorFusion's prompt words | `api.rs` `svg_style_words` | "minimal flat 2d vector icon. lineal color. on a white background. trending on artstation", ... |

`image.svg_generate` is VectorFusion's first two stages: the prompt drawn by a picture model this box runs
(Z-Image-Turbo by default), then fitted by LIVE.

## What was not, and why

- **Score distillation (VectorFusion's SDS, SVGDreamer's VPSD).** SDS needs the gradient of the diffusion loss to the
  rendered pixels: through the VAE *encoder* and a denoiser forward at a random level, with a large guidance scale
  (VectorFusion: 100). Here that means writing the backward pass of FLUX's VAE encoder in CUDA (none of the engines
  has an encoder backward) and fitting its activations beside the 7.2 GB Z-Image DiT on a 12 GB card; and
  Z-Image-Turbo is distilled to 8 steps without guidance, the kind of model whose SDS gradients are known to be poor.
  The raster-then-LIVE path gives clean SVGs in seconds; SDS would add minutes a picture and a port of its own.
- **DiffVG's edge sampling** (the boundary term LIVE's default mode samples). The prefiltered gradients cover the same
  boundary analytically and deterministically; upstream LIVE itself fits as well or better with them (below).
- **VectorFusion's path re-initialisation, CLIPasso, DiffSketcher, Word-As-Image, CLIP losses**: they need CLIP or
  more diffusion models; depth over breadth.
- **Shapes other than paths** (DiffVG's circles, ellipses, rects as primitives; gradients fills; transforms): LIVE and
  VectorFusion use paths with flat colours only.

## Differences, each for a reason

- A canvas keeps the picture's aspect (upstream squashes it to a square). Default 384 px.
- A path's fill is drawn, then its stroke (SVG's order); DiffVG draws a group's stroke under its fill. The paths the
  styles make have one or the other.
- Coverage is kept one pixel outside a path's box (DiffVG's BVH culls there and loses the outer half of the
  anti-aliasing). `NS_SVG_DIFFVG_BOX` (flag 32) culls as DiffVG for the check.
- New strokes start where the error is (VectorFusion places them at random).

## Accuracy (2026-11-09, `nextcuda image svg-check /mnt/hdd/comparisons/svgrender`)

Reference: `reference/svgrender/ref.py` in a venv on brainiac-nvidia (torch 2.15 CPU, pydiffvg built CPU-only), dumps
in `/mnt/hdd/comparisons/svgrender`. Loss for the gradients: sum(pixels x a fixed random gradient), DiffVG's straight
RGBA output. Relative L2:

| case | pixels | d points | d fill | d stroke | d width |
|---|---|---|---|---|---|
| fills (3 paths, opaque / translucent, both rules) | 3.4e-6 | 5.8e-3 | 1.2e-6 | | |
| crossing (a figure eight, a self-crossing loop) | 1.7e-4 | 2.2e-2 | 9.9e-4 | | |
| strokes (open and closed) | 8.9e-7 | 2.7e-4 | | 8.2e-7 | 3.5e-6 |
| mixed | 9.2e-7 | 7.2e-4 | 6.0e-7 | 2.3e-6 | 1.2e-5 |

The points' larger error sits where a pixel's nearest segment changes between two (the cusp of the figure eight).
The 16 x 16 sampled renders differ by 2.5e-3 - 5.6e-3 (jittered vs regular samples). The GPU against the host's float64
copy (`nextcuda image svg-selftest`): 3e-4 at worst. LIVE's parts: xing loss 8.9e-8 and its gradient 1.4e-6;
SparseCoordInit 12 of 12 and 11 of 12 placements identical; the distance weight 5e-2 (mean 0.02 of a 0..1 weight:
nearest pixels against skfmm's interpolated level set).

A whole LIVE fit, upstream's settings (32 paths, 8 layers of 200 steps, 240 px; PSNR over white at 4 x 4 samples):

| picture | ours (RTX 3060) | upstream, DiffVG 2 x 2 sampled (CPU) | upstream, DiffVG prefiltered (CPU) |
|---|---|---|---|
| astronaut (scikit-image) | 17.73 dB in 1.0 s | 17.79 dB in 245 s | 18.43 dB in 109 s |
| chelsea (scikit-image) | 22.93 dB in 1.1 s | 22.72 dB in 222 s | 23.51 dB in 101 s |

## Speed (`nextcuda image svg-bench`)

A whole LIVE step (bounds, the 2 x 2 coverage and distance weight, the backward pass, xing, Adam, the loss read back):

| canvas | paths | RTX 5070 | RTX 3060 |
|---|---|---|---|
| 240 | 32 / 128 / 512 | 0.21 / 0.31 / 0.74 ms | 0.42 / 0.72 / 1.80 ms |
| 512 | 32 / 128 / 512 | 0.75 / 1.11 / 1.99 ms | 1.35 / 2.13 / 5.11 ms |

The default fit (128 paths, 8 x 250 steps, 384 px) is 1.15 s on the 5070. Through the front door: a 96-path
text-to-SVG is 12.2 s of Z-Image-Turbo and 1.7 s of fit. No silo yet: the steps are a millisecond.

## How to run

```sh
nextcuda image svg-fit picture.png --paths 128 --out out.svg --png out.png      # in process
nextcuda image svg-selftest --gpu 0                                             # kernels vs host float64
nextcuda image svg-check /mnt/hdd/comparisons/svgrender                         # vs DiffVG / LIVE dumps
nextcuda image svg-bench --sizes 240,512 --paths 32,128,512
python -I reference/svgrender/ref.py --svgrender PyTorch-SVGRender --out DIR   # the dumps (dev only)
```

```
POST /rpc/image.svg_fit      {"image": "upload:...", "style": "iconography", "paths": 128}
POST /rpc/image.svg_generate {"prompt": "a red fox sitting in the snow", "paths": 96}
```
