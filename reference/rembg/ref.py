"""rembg (2.0.67) with isnet-general-use on the CPU (onnxruntime, float32) - the reference for nextcuda's IS-Net
(image/rembg): for each picture in a folder, what rembg makes of it, every stage dumped.

    python -I ref.py --model DIR_WITH_isnet-general-use.onnx --pictures DIR --out DIR

Dumps per picture NAME (DIR/NAME/): input.npy [1024, 1024, 3] float32 (rembg's normalize: Lanczos to 1024, / its max,
- 0.5), pred.npy [1024, 1024] (the network's output_image, the sigmoid), mask.png (rembg's mask: scaled to its range,
8 bits, Lanczos back), mask_pp.png (with post_process_mask), cutout.png (rembg.remove(..., post_process_mask=True),
the platform's call), picture.png (the picture as read).
"""
import argparse
import glob
import os

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pictures", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ["U2NET_HOME"] = a.model
    from rembg import new_session, remove
    from rembg.bg import post_process
    s = new_session("isnet-general-use", providers=["CPUExecutionProvider"])
    for p in sorted(glob.glob(os.path.join(a.pictures, "*"))):
        name = os.path.splitext(os.path.basename(p))[0]
        o = os.path.join(a.out, name)
        os.makedirs(o, exist_ok=True)
        img = Image.open(p)
        img.convert("RGB").save(os.path.join(o, "picture.png"))
        inp = s.normalize(img, (0.5, 0.5, 0.5), (1.0, 1.0, 1.0), (1024, 1024))
        x = list(inp.values())[0]
        np.save(os.path.join(o, "input.npy"), x[0].transpose(1, 2, 0).astype(np.float32))
        pred = s.inner_session.run(None, inp)[0][0, 0]
        np.save(os.path.join(o, "pred.npy"), pred.astype(np.float32))
        mask = s.predict(img)[0]
        mask.save(os.path.join(o, "mask.png"))
        Image.fromarray(post_process(np.array(mask))).save(os.path.join(o, "mask_pp.png"))
        remove(img, session=s, post_process_mask=True).save(os.path.join(o, "cutout.png"))
        print(name, img.size, flush=True)


if __name__ == "__main__":
    main()
