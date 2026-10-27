"""MOSS-Image-SFT on the CPU in float32 - the reference for nextcuda's engine (image/mossimage): the stages' tensors
dumped for `nextcuda image check`. The arithmetic is OpenMOSS's own (pipeline_moss_image.py, beside the model: its
MOSSImageTransformer and MOSSImageScheduler run as they are); the DiT's weights are nextcuda's file (the Q8_0 GGUF
dequantized), so both sides run the same numbers. The text encoder is transformers' Qwen3_5Model on the bf16 weights,
each layer made float32 while it runs (the 9B model does not fit the host in float32 at once); the VAE is diffusers'
AutoencoderKLFlux2 in float32.

    CUDA_VISIBLE_DEVICES= python -I ref.py --model MOSS_DIR --out DIR [--prompt TEXT] [--size 256] [--steps 8] [--seed 7]
        [--stages te,dit,vae]

MOSS_DIR: MOSS-Image-SFT-Q8_0.gguf, pipeline_moss_image.py, tokenizer/, vae/, ref/text_encoder/ (bf16). One stage
a run is fine (te then dit then vae: each reads the one before's dumps).

Dumps (DIR, .npy float32 unless said): prompt.txt; ids.npy, neg_ids.npy (int32: the chat template's tokens, the
prompt's and the empty negative's); te_cond [L, 4096], te_neg [Ln, 4096] (the text encoder's last hidden state,
normed); te_x0, te_x3 [L, 4096] (layers 0 and 3's outputs, for a mismatch's place); sigmas [n + 1] (the scheduler's
levels, its 0 last), timesteps [n]; lat_0 [128, h, w] (the starting noise); at step 0: mod [6, 3072], head [2, 3072]
(the shared and the head modulation), txt [L, 3072] (the text embedder's output), x0, x21, x41 [N + L, 3072] (those
blocks' outputs, image rows first), vel_c0, vel_u0 [128, h, w] (the conditional velocity, the unconditional one with
block 22 skipped); lat_final [128, h, w] (after every step, guidance 3); z [32, 2h, 2w] (the latents denormalized and
unpacked, the VAE's input); rgb [3, H, W] (the VAE's output, clamped to [-1, 1]) and ref.png.
"""
import argparse
import os
import sys

import numpy as np
import torch

torch.set_grad_enabled(False)


def save(out, name, t):
    a = t.detach().float().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)
    if a.dtype != np.int32:
        a = a.astype(np.float32)
    np.save(os.path.join(out, name + ".npy"), np.ascontiguousarray(a))


def te_stage(a, out):
    from transformers import AutoTokenizer, Qwen3_5Model
    tok = AutoTokenizer.from_pretrained(os.path.join(a.model, "tokenizer"))

    def ids_of(text):
        f = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
        return f, tok(f, return_tensors="pt", truncation=True, max_length=4096)["input_ids"]

    f, ids = ids_of(a.prompt)
    _, nids = ids_of("")
    with open(os.path.join(out, "prompt.txt"), "w") as fh:
        fh.write(a.prompt)
    with open(os.path.join(out, "template.txt"), "w") as fh:
        fh.write(f)
    save(out, "ids", ids[0].to(torch.int32).numpy())
    save(out, "neg_ids", nids[0].to(torch.int32).numpy())
    m = Qwen3_5Model.from_pretrained(os.path.join(a.model, "ref", "text_encoder"), dtype=torch.bfloat16).eval()
    lm = m.language_model
    # each layer (and the final norm) float32 while it runs, bfloat16 again after: the host holds the model in bf16
    def up(mod, args, kwargs):
        mod.float()
        return tuple(x.float() if torch.is_tensor(x) and x.is_floating_point() else x for x in args), kwargs
    def down(mod, args, kwargs, out):
        mod.to(torch.bfloat16)
        return out
    for mod in list(lm.layers) + [lm.norm]:
        mod.register_forward_pre_hook(up, with_kwargs=True)
        mod.register_forward_hook(down, with_kwargs=True)
    grab = {}
    for i in (0, 3):
        lm.layers[i].register_forward_hook(lambda mod, args, out, i=i: grab.__setitem__(i, (out[0] if isinstance(out, tuple) else out)[0].clone()))
    for name, x in (("te_cond", ids), ("te_neg", nids)):
        emb = lm.embed_tokens(x).float()
        h = lm(inputs_embeds=emb, attention_mask=torch.ones_like(x), use_cache=False).last_hidden_state[0]
        save(out, name, h)
        if name == "te_cond":
            save(out, "te_x0", grab[0])
            save(out, "te_x3", grab[3])
        print(name, tuple(h.shape), float(h.abs().mean()), flush=True)


def load_dit(a):
    sys.path.insert(0, a.model)
    import pipeline_moss_image as P
    from gguf import GGUFReader
    from gguf.quants import dequantize
    r = GGUFReader(os.path.join(a.model, "MOSS-Image-SFT-Q8_0.gguf"))
    sd = {}
    for t in r.tensors:
        v = dequantize(t.data, t.tensor_type)
        shape = [int(x) for x in reversed(list(t.shape))]
        sd[t.name] = torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32).reshape(shape))
    m = P.MOSSImageTransformer().float().eval()
    m.load_state_dict(sd, strict=True)
    return P, m


def dit_stage(a, out):
    P, m = load_dit(a)
    cond = torch.from_numpy(np.load(os.path.join(out, "te_cond.npy")))[None]
    neg = torch.from_numpy(np.load(os.path.join(out, "te_neg.npy")))[None]
    sch = P.MOSSImageScheduler()
    sch.set_timesteps(a.steps)
    save(out, "sigmas", sch.sigmas)
    save(out, "timesteps", sch.timesteps)
    h = w = a.size // 16
    g = torch.Generator().manual_seed(a.seed)
    lat = torch.randn((1, 128, h, w), generator=g, dtype=torch.float32)
    save(out, "lat_0", lat[0])
    # step 0's insides: the modulations, the text embedder, blocks 0, 21 and 41
    t0 = sch.timesteps[0]
    half = 128
    freq = 10000 ** (-torch.arange(half, dtype=torch.float64) / half)
    ph = torch.tensor([float(t0)], dtype=torch.float64)[:, None] * freq
    e = torch.cat((ph.cos(), ph.sin()), -1).float()
    tm = m.time_embedding(e)
    save(out, "mod", m.shared_block_modulation(tm).view(6, -1))
    save(out, "head", m.video_head_modulation(tm).view(2, -1))
    grab = {}
    hooks = [m.text_embedder.register_forward_hook(lambda mod, args, o: grab.__setitem__("txt", o[0].clone()))]
    for i in (0, 21, 41):
        hooks.append(m.layers[i].register_forward_hook(lambda mod, args, o, i=i: grab.__setitem__(i, o[0].clone())))
    vc = m(lat, t0, cond).sample.float()
    for hk in hooks:
        hk.remove()
    save(out, "txt", grab["txt"])
    for i in (0, 21, 41):
        save(out, f"x{i}", grab[i])
    vu = m(lat, t0, neg, skip_layers=[22]).sample.float()
    save(out, "vel_c0", vc[0])
    save(out, "vel_u0", vu[0])
    print("step 0", float(vc.abs().mean()), float(vu.abs().mean()), flush=True)
    for i, t in enumerate(sch.timesteps):
        if i == 0:
            c, u = vc, vu
        else:
            c = m(lat, t, cond).sample.float()
            u = m(lat, t, neg, skip_layers=[22]).sample.float()
        v = u + a.cfg * (c - u)
        lat = sch.step(v, t, lat, return_dict=False)[0]
        print("step", i, float(t), flush=True)
    save(out, "lat_final", lat[0])


def vae_stage(a, out):
    from diffusers import AutoencoderKLFlux2
    from PIL import Image
    vae = AutoencoderKLFlux2.from_pretrained(os.path.join(a.model, "vae"), torch_dtype=torch.float32).eval()
    z = torch.from_numpy(np.load(os.path.join(out, "lat_final.npy")))[None]
    bn = vae.bn
    eps = vae.config.batch_norm_eps
    z = z * (bn.running_var + eps).sqrt().view(1, -1, 1, 1) + bn.running_mean.view(1, -1, 1, 1)
    b, c, h, w = z.shape
    z = z.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2 * h, 2 * w)
    save(out, "z", z[0])
    img = vae.decode(z, return_dict=False)[0].clamp(-1, 1)
    save(out, "rgb", img[0])
    px = ((img[0].permute(1, 2, 0).numpy() + 1) * 127.5).round().clip(0, 255).astype(np.uint8)
    Image.fromarray(px).save(os.path.join(out, "ref.png"))
    print("vae", tuple(img.shape), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--prompt", default="A red fox sitting in fresh snow at dawn, soft golden light, detailed fur, photograph")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--cfg", type=float, default=3.0)
    p.add_argument("--stages", default="te,dit,vae")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(os.cpu_count())
    st = a.stages.split(",")
    if "te" in st:
        te_stage(a, a.out)
    if "dit" in st:
        dit_stage(a, a.out)
    if "vae" in st:
        vae_stage(a, a.out)


if __name__ == "__main__":
    main()
