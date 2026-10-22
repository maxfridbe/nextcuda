"""LTX-2.5's video decode as the movie maker runs it (the conv VAE, enable_tiling(256, 256, 25, 192, 192, 16) +
slicing) with the official code on the CPU in float32, from latents an engine saved (`save_latents`): the check of the
tiled decode at a real clip's size.

    ref_vae.py COMPONENTS LATENTS_PREFIX OUT_DIR --width W --height H --frames F

LATENTS_PREFIX.video.f32 holds the packed, normalized latents [T, 128] float32. Dumps video.npy [frames, H, W, 3] in
[0, 1] (float16) and meta.json.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("components")
    ap.add_argument("latents")
    ap.add_argument("out")
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--frames", type=int, required=True)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    from diffusers import AutoencoderKLLTX2Video
    from diffusers.modular_pipelines.ltx2.decoders import _denormalize_latents, _unpack_latents

    F, H, W = (a.frames - 1) // 8 + 1, a.height // 32, a.width // 32
    lat = torch.from_numpy(np.fromfile(a.latents + ".video.f32", dtype=np.float32).reshape(1, F * H * W, 128))
    vae = AutoencoderKLLTX2Video.from_pretrained(a.components, subfolder="vae", torch_dtype=torch.float32).eval()
    vae.enable_tiling(tile_sample_min_height=256, tile_sample_min_width=256, tile_sample_min_num_frames=25, tile_sample_stride_height=192,
                      tile_sample_stride_width=192, tile_sample_stride_num_frames=16)
    vae.enable_slicing()
    with torch.no_grad():
        x = _denormalize_latents(_unpack_latents(lat, F, H, W, 1, 1), vae.latents_mean, vae.latents_std, vae.config.scaling_factor)
        video = vae.decode(x, None, return_dict=False)[0]
        video = ((video[0].permute(1, 2, 3, 0) + 1) / 2).clamp(0, 1)
    np.save(os.path.join(a.out, "video.npy"), video.numpy().astype(np.float16))
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump({"width": a.width, "height": a.height, "frames": a.frames, "latents": a.latents}, f)
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
