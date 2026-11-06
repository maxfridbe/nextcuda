"""SeedVR2 (ByteDance-Seed; Apache-2.0) on the CPU in float32 - the reference for a port: the official code
(github.com/ByteDance-Seed/SeedVR @ e4de8c24: configs_3b/main.yaml, models/dit_v2 NaDiT, models/video_vae_v3) run
as it is, from the source tree, on a few frames, every stage dumped. What is not the official code's:

- apex's FusedRMSNorm / FusedLayerNorm are plain torch norms (the same arithmetic in float32);
- flash_attn_varlen_func is scaled_dot_product_attention per sequence, and the attention's .bfloat16() casts are
  kept in float32 (the official run rounds q, k, v to bfloat16 there);
- the VAE's latent is also dumped as the posterior's mean (the official run samples it).

    CUDA_VISIBLE_DEVICES= python -I ref.py --src SEEDVR_SRC --model MODEL_DIR --frames PNG_DIR --out DIR
        [--scale 2] [--count 5] [--seed 666]

MODEL_DIR: SeedVR2-3B's seedvr2_ema_3b.pth, ema_vae.pth, pos_emb.pt. The frames are upscaled as the official script
does (torchvision bicubic on [0, 1] floats to scale x the size, clamped, cropped to multiples of 16), the clip padded
to 4n + 1 frames with its last. Dumps (DIR, .npy float32): lq [3, T, H, W] (the prepared frames in [-1, 1]);
lat_mean, lat [t, h, w, 16] (the posterior's mean and the sample used, scaled by 0.9152); noise [t, h, w, 16];
txt [L, 5120]; emb [15360] (the timestep's embedding); blk0, blk9, blk10, blk31 [t h w / 4, 2560] (the video tokens
after those blocks); v [t, h, w, 16] (the DiT's output); x0 [t, h, w, 16]; out [3, T, H, W] (the decoded frames).
"""
import argparse
import glob
import os
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

torch.set_grad_enabled(False)


class FusedRMSNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(self.shape)) if elementwise_affine else None

    def forward(self, x):
        dims = tuple(range(-len(self.shape), 0))
        y = x.float() * torch.rsqrt(x.float().pow(2).mean(dims, keepdim=True) + self.eps)
        if self.weight is not None:
            y = y * self.weight
        return y.to(x.dtype)


class FusedLayerNorm(torch.nn.LayerNorm):
    def __init__(self, normalized_shape, elementwise_affine=True, eps=1e-5):
        super().__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None, max_seqlen_k=None, **kw):
    out = torch.empty(q.shape, dtype=torch.float32)
    cq, ckk = cu_seqlens_q.tolist(), cu_seqlens_k.tolist()
    for i in range(len(cq) - 1):
        qs = q[cq[i]:cq[i + 1]].float().transpose(0, 1)
        ks = k[ckk[i]:ckk[i + 1]].float().transpose(0, 1)
        vs = v[ckk[i]:ckk[i + 1]].float().transpose(0, 1)
        out[cq[i]:cq[i + 1]] = F.scaled_dot_product_attention(qs, ks, vs).transpose(0, 1)
    return out.to(q.dtype)


def stubs():
    import importlib.machinery
    # diffusers looks for these packages when it is imported: it is imported before the stubs exist
    import diffusers  # noqa: F401
    import diffusers.models.normalization  # noqa: F401
    import diffusers.models.embeddings  # noqa: F401
    apex = types.ModuleType("apex")
    norm = types.ModuleType("apex.normalization")
    norm.FusedRMSNorm, norm.FusedLayerNorm = FusedRMSNorm, FusedLayerNorm
    apex.normalization = norm
    sys.modules["apex"], sys.modules["apex.normalization"] = apex, norm
    fa = types.ModuleType("flash_attn")
    fa.flash_attn_varlen_func = flash_attn_varlen_func
    # diffusers' attention dispatch imports this name when the (half-installed) package is found; never called here
    fa.flash_attn_func = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("flash_attn_func is not stubbed"))
    for m in (apex, norm, fa):
        m.__spec__ = importlib.machinery.ModuleSpec(m.__name__, None)
    sys.modules["flash_attn"] = fa


def save(out, name, t):
    a = t.detach().float().contiguous().numpy() if torch.is_tensor(t) else np.asarray(t)
    np.save(os.path.join(out, name + ".npy"), a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--seed", type=int, default=666)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(os.cpu_count())
    stubs()
    sys.path.insert(0, a.src)
    os.chdir(a.src)  # the configs name each other by paths from the tree's root
    from torchvision.transforms import functional as TVF
    from torchvision.transforms import InterpolationMode
    from common.config import load_config, create_object
    from einops import rearrange

    # the VAE's slicing asks the (one-process) group for its neighbours' ranks
    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group("gloo", init_method="tcp://127.0.0.1:29517", rank=0, world_size=1)
    config = load_config(os.path.join(a.src, "configs_3b/main.yaml"))
    dit = create_object(config.dit.model)
    print(dit.load_state_dict(torch.load(os.path.join(a.model, "seedvr2_ema_3b.pth"), map_location="cpu", weights_only=True, mmap=True), strict=True))
    dit = dit.float().eval()
    vae = create_object(config.vae.model)
    print(vae.load_state_dict(torch.load(os.path.join(a.model, "ema_vae.pth"), map_location="cpu", weights_only=True)))
    vae = vae.float().eval()
    vae.set_causal_slicing(split_size=4, memory_device="same")
    vae.set_memory_limit(conv_max_mem=0.5, norm_max_mem=0.5)

    # the frames, as the official script prepares them
    paths = sorted(glob.glob(os.path.join(a.frames, "*.png")))[:a.count]
    x = torch.stack([torch.from_numpy(np.asarray(Image.open(p).convert("RGB"))).permute(2, 0, 1) for p in paths]).float() / 255.0
    h, w = x.shape[-2:]
    hh, ww = round(h * a.scale), round(w * a.scale)
    x = TVF.resize(x, size=(hh, ww), interpolation=InterpolationMode.BICUBIC).clamp(0, 1)
    x = x[..., : hh // 16 * 16, : ww // 16 * 16]
    x = (x - 0.5) / 0.5
    x = rearrange(x, "t c h w -> c t h w")
    t = x.size(1)
    if (t - 1) % 4:
        x = torch.cat([x, x[:, -1:].repeat(1, 4 - (t - 1) % 4, 1, 1)], dim=1)
    save(a.out, "lq", x)
    print("frames", tuple(x.shape))

    # the VAE: the posterior's mean and a sample, scaled
    scale = config.vae.scaling_factor
    post = vae.encode(x.unsqueeze(0)).posterior
    torch.manual_seed(a.seed)
    z = post.sample().squeeze(2) if hasattr(post, "sample") else post.mode()
    lat_mean = rearrange(post.mode(), "b c ... -> b ... c")[0] * scale
    lat = rearrange(z if z.ndim == 5 else z.unsqueeze(2), "b c ... -> b ... c")[0] * scale
    save(a.out, "lat_mean", lat_mean)
    save(a.out, "lat", lat)
    print("latents", tuple(lat.shape))

    # the DiT: one step from noise at t = 1000, the condition the latent and a mask of ones
    torch.manual_seed(a.seed + 1)
    noise = torch.randn_like(lat)
    save(a.out, "noise", noise)
    cond = torch.cat([lat, torch.ones_like(lat[..., :1])], dim=-1)
    txt = torch.load(os.path.join(a.model, "pos_emb.pt"), map_location="cpu", weights_only=True).float()
    save(a.out, "txt", txt)
    grabs = {}
    for i in (0, 9, 10, 31):
        dit.blocks[i].register_forward_hook(lambda m, inp, out, i=i: grabs.__setitem__(i, out[0].clone()))
    orig_bf16 = torch.Tensor.bfloat16
    torch.Tensor.bfloat16 = lambda self, *args, **kw: self
    try:
        tt, th, tw, _ = lat.shape
        vid = torch.cat([noise, cond], dim=-1).reshape(-1, 33)
        emb = dit.emb_in(torch.tensor([1000.0]), device=torch.device("cpu"), dtype=torch.float32)
        save(a.out, "emb", emb[0])
        v = dit(vid=vid, txt=txt, vid_shape=torch.tensor([[tt, th, tw]]), txt_shape=torch.tensor([[txt.shape[0]]]),
                timestep=torch.tensor([1000.0])).vid_sample
    finally:
        torch.Tensor.bfloat16 = orig_bf16
    for i, g in grabs.items():
        save(a.out, f"blk{i}", g)
    v = v.reshape(lat.shape)
    save(a.out, "v", v)
    x0 = noise - v
    save(a.out, "x0", x0)

    # the decoder
    out = vae.decode(rearrange(x0 / scale, "t h w c -> c t h w").unsqueeze(0)).sample[0]
    save(a.out, "out", out)
    o = ((out.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 3, 0).numpy()
    for i in range(min(t, o.shape[0])):
        Image.fromarray(o[i]).save(os.path.join(a.out, f"out_{i:03d}.png"))
    print("done", tuple(out.shape))


if __name__ == "__main__":
    main()
