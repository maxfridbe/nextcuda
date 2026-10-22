"""LTX-2.5's image-to-video conditioning with the official code on the CPU in float32: the picture re-compressed at
CRF 18, preprocessed, through the video VAE's encoder (argmax), normalized - the first latent frame the DiT is given.

    ref_enc.py COMPONENTS PICTURE OUT_DIR --width W --height H

Dumps: pic.npy [H, W, 3] in [0, 1] (the picture after the re-compression, what the encoder sees before [-1, 1]),
latents.npy [H/32 W/32, 128] (normalized, packed), meta.json.
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
    ap.add_argument("picture")
    ap.add_argument("out")
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from PIL import Image
    from diffusers import AutoencoderKLLTX2Video
    from diffusers.pipelines.ltx2.utils import apply_image_conditioning_crf, LTX2_5_IMAGE_CRF
    from diffusers.video_processor import VideoProcessor
    from diffusers.modular_pipelines.ltx2.encoders import _normalize_latents

    img = Image.open(a.picture).convert("RGB")
    if img.size != (a.width, a.height):
        img = img.resize((a.width, a.height), Image.LANCZOS)
    img = Image.fromarray(apply_image_conditioning_crf(np.array(img), LTX2_5_IMAGE_CRF))
    np.save(os.path.join(a.out, "pic.npy"), np.asarray(img).astype(np.float32) / 255.0)
    vp = VideoProcessor(vae_scale_factor=32)
    x = vp.preprocess(img, height=a.height, width=a.width).to(torch.float32)
    vae = AutoencoderKLLTX2Video.from_pretrained(a.components, subfolder="vae", torch_dtype=torch.float32).eval()
    with torch.no_grad():
        z = vae.encode(x.unsqueeze(2)).latent_dist.mode()
        z = _normalize_latents(z, vae.latents_mean, vae.latents_std)
    # [1, 128, 1, h, w] -> packed [h w, 128]
    np.save(os.path.join(a.out, "latents.npy"), z[0, :, 0].permute(1, 2, 0).reshape(-1, 128).contiguous().numpy())
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump({"width": a.width, "height": a.height, "picture": a.picture}, f)
    print("done")


if __name__ == "__main__":
    sys.exit(main())
