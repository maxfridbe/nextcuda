"""Depth Anything V2 Small as transformers runs it (DepthAnythingForDepthEstimation + DPTImageProcessor, the slow PIL
one; float32 on the CPU) - the reference for nextcuda's image/depthanything. The model is built from its config (below,
as the -hf repository's config.json) and its safetensors file; nothing is downloaded and no repository code runs.

    python -I ref.py --model model.safetensors --out DIR PICTURE...

Dumps per picture STEM in DIR: STEM.input.<w>x<h>.f32 (pixel_values as [h, w, 3], channels last) and
STEM.ref.<w>x<h>.f32 (predicted_depth [h, w]), float32 little-endian - what `nextcuda image depth` writes as
STEM.input.<w>x<h>.f32 and STEM.<w>x<h>.f32. compare.py reads both.
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from transformers import DepthAnythingConfig, DepthAnythingForDepthEstimation, DPTImageProcessor

CONFIG = {
    "backbone_config": {"model_type": "dinov2", "hidden_size": 384, "image_size": 518, "num_attention_heads": 6,
                        "out_features": ["stage3", "stage6", "stage9", "stage12"], "out_indices": [3, 6, 9, 12],
                        "patch_size": 14, "reshape_hidden_states": False},
    "fusion_hidden_size": 64, "head_hidden_size": 32, "head_in_index": -1, "neck_hidden_sizes": [48, 96, 192, 384],
    "patch_size": 14, "reassemble_factors": [4, 2, 1, 0.5], "reassemble_hidden_size": 384, "use_pretrained_backbone": False,
}
PROCESSOR = dict(do_normalize=True, do_pad=False, do_rescale=True, do_resize=True, ensure_multiple_of=14, image_mean=[0.485, 0.456, 0.406],
                 image_std=[0.229, 0.224, 0.225], keep_aspect_ratio=True, resample=3, rescale_factor=1 / 255, size={"height": 518, "width": 518})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("pictures", nargs="+")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = DepthAnythingForDepthEstimation(DepthAnythingConfig(**CONFIG)).eval()
    missing, unexpected = model.load_state_dict(load_file(a.model), strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    proc = DPTImageProcessor(**PROCESSOR)
    os.makedirs(a.out, exist_ok=True)
    for p in a.pictures:
        stem = os.path.splitext(os.path.basename(p))[0]
        img = Image.open(p).convert("RGB")
        px = proc(images=img, return_tensors="pt")["pixel_values"]
        h, w = px.shape[2], px.shape[3]
        px[0].permute(1, 2, 0).contiguous().numpy().astype("<f4").tofile(os.path.join(a.out, f"{stem}.input.{w}x{h}.f32"))
        with torch.no_grad():
            d = model(pixel_values=px).predicted_depth[0]
        d.numpy().astype("<f4").tofile(os.path.join(a.out, f"{stem}.ref.{w}x{h}.f32"))
        print(stem, img.size, "->", (w, h), "depth", float(d.min()), float(d.max()), flush=True)


if __name__ == "__main__":
    main()
