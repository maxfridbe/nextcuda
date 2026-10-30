"""LTX-2.5's shot upscaler (the movie maker's ltx2_upscale: LTX2LatentUpsamplePipeline, vae.enable_tiling()) with the
official code on the CPU in float32, the encoder's latents taken as the means (the pipeline samples; argmax here so
the check is exact): the first FRAMES frames of a clip doubled.

    ref_ups.py COMPONENTS CLIP OUT_DIR [--frames 17]

Dumps: src.npy [F, H, W, 3] (the frames read), enc.npy [f, h, w, 128] (the encoder's means), ups.npy [f, 2h, 2w, 128]
(the upsampler's latents), video.npy [F, 2H, 2W, 3] in [0, 1] (float16), meta.json.
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("components")
    ap.add_argument("clip")
    ap.add_argument("out")
    ap.add_argument("--frames", type=int, default=17)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    from diffusers import AutoencoderKLLTX2Video
    from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel

    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", a.clip],
                           capture_output=True, text=True, check=True).stdout.strip().split("x")
    w, h = int(probe[0]), int(probe[1])
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", a.clip, "-frames:v", str(a.frames), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)[: a.frames].astype(np.float32) / 255.0
    np.save(os.path.join(a.out, "src.npy"), frames)
    vae = AutoencoderKLLTX2Video.from_pretrained(a.components, subfolder="vae", torch_dtype=torch.float32).eval()
    vae.enable_tiling()
    ups = LTX2LatentUpsamplerModel.from_pretrained(a.components, subfolder="latent_upsampler", torch_dtype=torch.float32).eval()
    x = torch.from_numpy(frames * 2 - 1).permute(3, 0, 1, 2)[None]
    with torch.no_grad():
        z = vae.encode(x).latent_dist.mode()
        np.save(os.path.join(a.out, "enc.npy"), z[0].permute(1, 2, 3, 0).contiguous().numpy())
        z2 = ups(z)
        np.save(os.path.join(a.out, "ups.npy"), z2[0].permute(1, 2, 3, 0).contiguous().numpy())
        v = vae.decode(z2, None, return_dict=False)[0]
        v = ((v[0].permute(1, 2, 3, 0) + 1) / 2).clamp(0, 1)
    np.save(os.path.join(a.out, "video.npy"), v.numpy().astype(np.float16))
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump({"width": w, "height": h, "frames": a.frames, "clip": a.clip}, f)
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
