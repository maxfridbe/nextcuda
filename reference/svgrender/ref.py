"""DiffVG and LIVE (PyTorch-SVGRender) on the CPU in float32 - the reference for nextcuda's svgrender
(image/svgrender, kernels/image/svgrender). Dev-only: run in a venv on the GPU host, never at runtime.

    python -I ref.py --svgrender PATH_TO_PyTorch-SVGRender --out DIR [--pictures DIR] [--live-paths 32] [--live-layers 8]
                     [--live-iters 200] [--live-size 240]

The venv: torch (CPU), pydiffvg built from BachiLi/diffvg (DIFFVG_CUDA=0), numpy, pillow, scikit-fmm, opencv-python-headless,
matplotlib, omegaconf, scikit-image. Dumps (DIR/...):

  raster/<case>/scene.json      the scene: {width, height, paths: [{points, closed, fill, stroke, hw, evenodd}]}
  raster/<case>/pre.npy         DiffVG's prefiltered render (use_prefiltering, one sample at each pixel's centre) [H, W, 4]
  raster/<case>/d_img.npy       a fixed gradient of the loss to those pixels [H, W, 4] (loss = sum(pre * d_img))
  raster/<case>/g_<name>.npy    the gradients DiffVG's backward gives: points (all paths' concatenated), fill, stroke, hw
  raster/<case>/samp16.npy      DiffVG's sampled render, 16 x 16 jittered samples a pixel (seed 0) [H, W, 4]
  udf/<case>/cov.npy, weight.npy  LIVE's calc_distance_weight: a 2 x 2 coverage and get_sdf's weight from it
  xing/case.json                xing_loss_fn on a few paths: {points: [...], loss, grads: [...]}
  sparse/<pic>/{pred,gt}.npy, coords.json  SparseCoordInit's first coordinates for a target and a white prediction
  live/<pic>/target.npy         LIVE's target (Resize to size x size, ToTensor) [H, W, 3]
  live/<pic>/result.json, final.svg, final.npy  upstream LIVE (its pipeline's loop; Painter, PainterOptimizer, xing, UDF)
"""
import argparse
import importlib.util
import json
import os
import sys
import time
import types

import numpy as np
import torch


def load_upstream(root):
    """painter/live of PyTorch-SVGRender without its package __init__ (which pulls in diffusers)"""
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
        return m
    pkg = types.ModuleType("pytorch_svgrender")
    pkg.__path__ = []
    sys.modules["pytorch_svgrender"] = pkg
    mh = types.ModuleType("pytorch_svgrender.model_helper")
    sys.modules["pytorch_svgrender.model_helper"] = mh
    dh = load("pytorch_svgrender.model_helper.diffvg_helper", os.path.join(root, "pytorch_svgrender/model_helper/diffvg_helper.py"))
    mh.DiffVGState = dh.DiffVGState
    pp = load("pytorch_svgrender.painter.live.painter_params", os.path.join(root, "pytorch_svgrender/painter/live/painter_params.py"))
    xl = load("pytorch_svgrender.painter.live.xing_loss", os.path.join(root, "pytorch_svgrender/painter/live/xing_loss.py"))
    return pp, xl


def blob(cx, cy, r0, r1, n=12, phase=0.0):
    out = []
    for k in range(n):
        a = phase + k * 2 * np.pi / n
        r = r0 if k % 3 == 0 else r1
        out.append([cx + r * np.cos(a), cy + r * np.sin(a)])
    return out


def scenes():
    """The rasterizer's cases: fills (opaque, translucent, overlapping), an even-odd and a nonzero self-crossing path,
    open and closed strokes. A path has a fill or a stroke, not both (DiffVG draws a group's stroke under its fill,
    SVG over it: the two agree on either alone)"""
    s = {}
    s["fills"] = {"width": 64, "height": 48, "paths": [
        {"points": blob(24, 22, 14, 18), "closed": True, "fill": [0.9, 0.2, 0.1, 1.0], "evenodd": True},
        {"points": blob(40, 26, 10, 15, phase=0.4), "closed": True, "fill": [0.1, 0.4, 0.8, 0.6], "evenodd": True},
        {"points": [[6, 40], [20, 30], [30, 46], [50, 44], [60, 40], [62, 10], [50, 6], [30, 4], [10, 14]], "closed": True,
         "fill": [0.2, 0.8, 0.3, 0.45], "evenodd": False},
    ]}
    s["crossing"] = {"width": 48, "height": 48, "paths": [
        # a figure eight: its lobes wind opposite ways (inside under both rules) - and a star-ish loop that crosses itself
        {"points": [[6, 24], [6, 6], [24, 6], [24, 24], [24, 42], [42, 42], [42, 24], [42, 6], [24, 6], [24, 24], [24, 42], [6, 42]],
         "closed": True, "fill": [0.8, 0.6, 0.1, 0.9], "evenodd": True},
        {"points": [[10, 30], [40, 10], [44, 40], [14, 12], [8, 44], [40, 44], [30, 26], [34, 20], [12, 20]], "closed": True,
         "fill": [0.3, 0.1, 0.7, 0.5], "evenodd": False},
    ]}
    s["strokes"] = {"width": 56, "height": 40, "paths": [
        {"points": [[4, 6], [20, 2], [30, 34], [52, 30]], "closed": False, "stroke": [0.1, 0.1, 0.1, 1.0], "hw": 1.5},
        {"points": [[6, 34], [16, 10], [28, 10], [34, 20], [40, 30], [50, 12], [50, 4]], "closed": False, "stroke": [0.8, 0.2, 0.5, 0.7], "hw": 2.5},
        {"points": blob(30, 20, 9, 13), "closed": True, "stroke": [0.1, 0.6, 0.6, 0.8], "hw": 0.8},
    ]}
    s["mixed"] = {"width": 80, "height": 60, "paths": [
        {"points": blob(30, 30, 20, 26), "closed": True, "fill": [0.95, 0.85, 0.3, 1.0], "evenodd": True},
        {"points": blob(52, 24, 8, 12, phase=1.0), "closed": True, "fill": [0.2, 0.2, 0.9, 0.8], "evenodd": True},
        {"points": [[8, 50], [30, 40], [50, 58], [74, 44]], "closed": False, "stroke": [0.0, 0.0, 0.0, 0.9], "hw": 1.2},
        {"points": blob(18, 18, 5, 7, phase=0.3), "closed": True, "fill": [0.9, 0.1, 0.1, 0.7], "evenodd": True},
    ]}
    return s


def build(pydiffvg, sc, grads=False):
    shapes, groups = [], []
    for i, p in enumerate(sc["paths"]):
        pts = torch.tensor(p["points"], dtype=torch.float32, requires_grad=grads)
        nseg = len(p["points"]) // 3 if p["closed"] else (len(p["points"]) - 1) // 3
        hw = torch.tensor(float(p.get("hw", 0.0)), requires_grad=grads and "stroke" in p)
        shapes.append(pydiffvg.Path(num_control_points=torch.LongTensor([2] * nseg), points=pts, stroke_width=hw, is_closed=p["closed"]))
        fill = torch.tensor(p["fill"], dtype=torch.float32, requires_grad=grads) if "fill" in p else None
        stroke = torch.tensor(p["stroke"], dtype=torch.float32, requires_grad=grads) if "stroke" in p else None
        groups.append(pydiffvg.ShapeGroup(shape_ids=torch.tensor([i]), fill_color=fill, stroke_color=stroke,
                                          use_even_odd_rule=p.get("evenodd", True)))
    return shapes, groups


def render(pydiffvg, sc, shapes, groups, samples, seed=0, prefilter=False):
    args = pydiffvg.RenderFunction.serialize_scene(sc["width"], sc["height"], shapes, groups, use_prefiltering=prefilter)
    return pydiffvg.RenderFunction.apply(sc["width"], sc["height"], samples, samples, seed, None, *args)


def raster(pydiffvg, out):
    for name, sc in scenes().items():
        d = os.path.join(out, "raster", name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "scene.json"), "w") as f:
            json.dump(sc, f)
        shapes, groups = build(pydiffvg, sc, grads=True)
        img = render(pydiffvg, sc, shapes, groups, 1, prefilter=True)
        rng = np.random.default_rng(1234)
        d_img = rng.uniform(-0.5, 0.5, size=img.shape).astype(np.float32)
        (img * torch.from_numpy(d_img)).sum().backward()
        np.save(os.path.join(d, "pre.npy"), img.detach().numpy().astype(np.float32))
        np.save(os.path.join(d, "d_img.npy"), d_img)
        np.save(os.path.join(d, "g_points.npy"), np.concatenate([s.points.grad.numpy() for s in shapes]).astype(np.float32))
        np.save(os.path.join(d, "g_fill.npy"), np.stack([g.fill_color.grad.numpy() if g.fill_color is not None else np.zeros(4, np.float32)
                                                          for g in groups]).astype(np.float32))
        np.save(os.path.join(d, "g_stroke.npy"), np.stack([g.stroke_color.grad.numpy() if g.stroke_color is not None else np.zeros(4, np.float32)
                                                            for g in groups]).astype(np.float32))
        np.save(os.path.join(d, "g_hw.npy"), np.array([float(s.stroke_width.grad) if s.stroke_width.grad is not None else 0.0 for s in shapes],
                                                      np.float32))
        shapes, groups = build(pydiffvg, sc)
        with torch.no_grad():
            samp = render(pydiffvg, sc, shapes, groups, 16, seed=0)
        np.save(os.path.join(d, "samp16.npy"), samp.numpy().astype(np.float32))
        print("raster", name, img.shape, flush=True)


def udf(pydiffvg, pp, out):
    """LIVE's weight from the 2 x 2 coverage of a scene's paths (calc_distance_weight, its get_sdf)"""
    for name in ["fills", "mixed"]:
        sc = scenes()[name]
        d = os.path.join(out, "udf", name)
        os.makedirs(d, exist_ok=True)
        shapes, groups = build(pydiffvg, sc)
        for i, g in enumerate(groups):
            g.fill_color = torch.FloatTensor([1, 1, 1, 1])
            g.stroke_color = None
        for s in shapes:
            s.stroke_width = torch.FloatTensor([0])
        with torch.no_grad():
            im = render(pydiffvg, sc, shapes, groups, 2, seed=0)
        cov = im[:, :, 3].numpy()
        w = np.clip(pp.get_sdf(cov, normalize="to1"), 0, 1)
        np.save(os.path.join(d, "cov.npy"), cov.astype(np.float32))
        np.save(os.path.join(d, "weight.npy"), w.astype(np.float32))
        print("udf", name, flush=True)


def xing(xl, out):
    os.makedirs(os.path.join(out, "xing"), exist_ok=True)
    cases = [blob(20, 20, 8, 12), [[0, 0], [4, 4], [0, 4], [4, 0], [5, 2], [3, -1]],
             [[10, 10], [30, 0], [0, 0], [20, 10], [25, 30], [5, 30], [0, 20], [8, 25], [30, 25]]]
    pts = [torch.tensor(c, dtype=torch.float32, requires_grad=True) for c in cases]
    loss = xl.xing_loss_fn(pts) * 0.01
    loss.backward()
    with open(os.path.join(out, "xing", "case.json"), "w") as f:
        json.dump({"points": cases, "weight": 0.01, "loss": float(loss), "grads": [p.grad.numpy().tolist() for p in pts]}, f)
    print("xing", float(loss), flush=True)


def pictures(src):
    """(name, PIL picture): the folder's, else scikit-image's public-domain samples"""
    from PIL import Image
    if src:
        for p in sorted(os.listdir(src)):
            yield os.path.splitext(p)[0], Image.open(os.path.join(src, p)).convert("RGB")
        return
    from skimage import data
    yield "astronaut", Image.fromarray(data.astronaut())
    yield "chelsea", Image.fromarray(data.chelsea())


def sparse(pp, out, src, size):
    from torchvision import transforms
    for name, pic in pictures(src):
        d = os.path.join(out, "sparse", name)
        os.makedirs(d, exist_ok=True)
        gt = transforms.Compose([transforms.Resize(size=(size, size)), transforms.ToTensor()])(pic).unsqueeze(0)
        pred = torch.ones_like(gt)
        np.save(os.path.join(d, "gt.npy"), gt[0].permute(1, 2, 0).numpy().astype(np.float32))
        np.save(os.path.join(d, "pred.npy"), pred[0].permute(1, 2, 0).numpy().astype(np.float32))
        init = pp.SparseCoordInit(pred, gt)
        coords = [[float(v) for v in init()] for _ in range(12)]
        with open(os.path.join(d, "coords.json"), "w") as f:
            json.dump(coords, f)
        print("sparse", name, coords[:3], flush=True)


def live(pydiffvg, pp, xl, out, src, size, paths, layers, iters, prefilter):
    """LIVE_pipeline.painterly_rendering's loop on upstream's Painter (DiffVG's 2 x 2 sampling, or its prefiltering)"""
    from torchvision import transforms
    import omegaconf
    for name, pic in pictures(src):
        tag = name + ("-prefilter" if prefilter else "")
        d = os.path.join(out, "live", tag)
        os.makedirs(d, exist_ok=True)
        target = transforms.Compose([transforms.Resize(size=(size, size)), transforms.ToTensor()])(pic).unsqueeze(0)
        np.save(os.path.join(out, "live", f"{name}-target.npy"), target[0].permute(1, 2, 0).numpy().astype(np.float32))
        torch.manual_seed(0)
        np.random.seed(0)
        import random
        random.seed(0)
        cfg = omegaconf.OmegaConf.create({"print_timing": False})
        r = pp.Painter(target, cfg, 4, "circle", 5, canvas_size=size, device=torch.device("cpu"))
        if prefilter:
            def warp(seed=0, r=r):
                r.clip_curve_shape()
                a = pydiffvg.RenderFunction.serialize_scene(r.canvas_width, r.canvas_height, r.shapes, r.shape_groups, use_prefiltering=True)
                return pydiffvg.RenderFunction.apply(r.canvas_width, r.canvas_height, 1, 1, seed, None, *a)
            r.render_warp = warp
        r.component_wise_path_init(pred=None, init_type="sparse")
        lr = omegaconf.OmegaConf.create({"point": 1, "color": 0.01, "bg": 0.01, "stroke_width": 0.1, "stroke_color": 0.01})
        schedule = [paths // layers] * layers
        opts = [pp.PainterOptimizer(r, iters, lr, False, False) for _ in schedule]
        keep = 0
        t0 = time.time()
        losses = []
        for li, n in enumerate(schedule):
            r.init_image(num_paths=n)
            opts[li].init_optimizers()
            for t in range(iters):
                img = r.get_image(step=t)
                w = r.calc_distance_weight(keep)
                l_rec = (((img - target) ** 2).sum(1) * w).mean()
                l_x = xl.xing_loss_fn(r.get_point_parameters()) * 0.01
                loss = l_rec + l_x
                for i in range(li + 1):
                    opts[i].zero_grad_()
                loss.backward()
                for i in range(li + 1):
                    opts[i].step_()
                r.clip_curve_shape()
                for i in range(li + 1):
                    opts[i].update_lr()
            losses.append(float(l_rec))
            keep = w.detach().cpu().numpy() * 1
            r.component_wise_path_init(pred=img, init_type="sparse")
            print("live", tag, "layer", li, float(l_rec), f"{time.time() - t0:.1f}s", flush=True)
        secs = time.time() - t0
        with torch.no_grad():
            # the result as it is seen: rendered over white with many samples
            a = pydiffvg.RenderFunction.serialize_scene(size, size, r.shapes, r.shape_groups)
            fin = pydiffvg.RenderFunction.apply(size, size, 4, 4, 0, None, *a)
            fin = fin[:, :, 3:4] * fin[:, :, :3] + (1 - fin[:, :, 3:4])
        mse = float(((fin - target[0].permute(1, 2, 0)) ** 2).mean())
        np.save(os.path.join(d, "final.npy"), fin.numpy().astype(np.float32))
        r.save_svg(os.path.join(d, "final.svg"))
        res = {"picture": name, "size": size, "paths": paths, "layers": layers, "iters": iters, "prefilter": prefilter, "seconds": secs,
               "layer_losses": losses, "mse": mse, "psnr": 10 * np.log10(1.0 / mse)}
        with open(os.path.join(d, "result.json"), "w") as f:
            json.dump(res, f)
        print("live", tag, res, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--svgrender", required=True, help="a PyTorch-SVGRender checkout")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pictures", default=None)
    ap.add_argument("--live-size", type=int, default=240)
    ap.add_argument("--live-paths", type=int, default=32)
    ap.add_argument("--live-layers", type=int, default=8)
    ap.add_argument("--live-iters", type=int, default=200)
    ap.add_argument("--only", default="raster,udf,xing,sparse,live,live-prefilter")
    a = ap.parse_args()
    import pydiffvg
    pydiffvg.set_use_gpu(False)
    pp, xl = load_upstream(a.svgrender)
    only = a.only.split(",")
    if "raster" in only:
        raster(pydiffvg, a.out)
    if "udf" in only:
        udf(pydiffvg, pp, a.out)
    if "xing" in only:
        xing(xl, a.out)
    if "sparse" in only:
        sparse(pp, a.out, a.pictures, a.live_size)
    if "live" in only:
        live(pydiffvg, pp, xl, a.out, a.pictures, a.live_size, a.live_paths, a.live_layers, a.live_iters, False)
    if "live-prefilter" in only:
        live(pydiffvg, pp, xl, a.out, a.pictures, a.live_size, a.live_paths, a.live_layers, a.live_iters, True)


if __name__ == "__main__":
    main()
