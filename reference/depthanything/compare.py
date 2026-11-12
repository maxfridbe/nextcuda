"""nextcuda's depth against the reference's: for each STEM.ref.<w>x<h>.f32 in DIR (ref.py), the engine's
STEM.<w>x<h>.f32 (and the inputs, STEM.input...) - the largest difference, relative to the reference's range, and the
mean one.

    python -I compare.py REF_DIR OURS_DIR
"""
import glob
import os
import sys

import numpy as np


def main():
    ref, ours = sys.argv[1], sys.argv[2]
    worst = 0.0
    for r in sorted(glob.glob(os.path.join(ref, "*.ref.*.f32"))):
        stem, size = os.path.basename(r)[:-4].split(".ref.")
        for kind, a, b in [("input", os.path.join(ref, f"{stem}.input.{size}.f32"), os.path.join(ours, f"{stem}.input.{size}.f32")),
                           ("depth", r, os.path.join(ours, f"{stem}.{size}.f32"))]:
            if not os.path.exists(b):
                print(f"{stem} {kind}: no {b}")
                continue
            x, y = np.fromfile(a, "<f4"), np.fromfile(b, "<f4")
            if x.shape != y.shape:
                print(f"{stem} {kind}: {x.shape} vs {y.shape}")
                continue
            d = np.abs(x - y)
            span = float(x.max() - x.min()) or 1.0
            rel = float(d.max()) / span
            if kind == "depth":
                worst = max(worst, rel)
            print(f"{stem} {kind} {size}: max abs {d.max():.3e}, mean abs {d.mean():.3e}, max / range {rel:.3e} (range {x.min():.4f}..{x.max():.4f}),"
                  f" pixels off by > 1/255 of the range: {(d > span / 255).mean() * 100:.3f}%")
    print(f"worst depth difference / range: {worst:.3e}")


if __name__ == "__main__":
    main()
