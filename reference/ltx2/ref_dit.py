"""LTX-2.5's denoiser, sampler and decoders with the official code on the CPU in float32: the reference for nextcuda's
DiT, guidance, VAE and audio stages. The same GGUF the engine runs (diffusers' GGUF linears dequantize it per call),
so the check measures our arithmetic, not the quantization.

    ref_dit.py COMPONENTS GGUF TEXT_REF_DIR OUT_DIR [--width 256] [--height 256] [--frames 17] [--steps 3] [--seed 0]

TEXT_REF_DIR is ref_text.py's output (video_emb, audio_emb, neg_video_emb, neg_audio_emb). What the movie maker's
engine runs (LTX25AutoBlocks with the conv VAE decode): sigmas linspace(1, 1/N, N) (shift 1, no dynamic shift), four
passes a step (cond, uncond, STG on block 28, modality isolation) combined per modality in x0 space by LTX2Guidance
(video 3 / 1 / 3, audio 7 / 1 / 3, rescale 0.7), Euler. Dumps (numpy float32):

    lat_0.npy, aud_0.npy                 the starting noise, packed [T, 128] / [La, 128]
    s0_<pass>_v.npy, s0_<pass>_a.npy     step 0's velocities of each pass (cond, uncond, stg, mod)
    s0_blk<i>_v.npy, s0_blk<i>_a.npy     step 0's cond pass after blocks 0, 1, 27, 28, 47
    lat_<k>.npy, aud_<k>.npy             the latents after step k (k = 1..N)
    video.npy                            the decoded frames [F, H, W, 3] in [0, 1]
    audio_z.npy                          the audio VAE's input [8, La, 16]
    mel.npy, wave.npy                    the audio VAE's mel [2, L, 64] and the vocoder's wave [2, S] at 48 kHz
    meta.json                            the shapes, sigmas, timings
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("components")
    ap.add_argument("gguf")
    ap.add_argument("text")
    ap.add_argument("out")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--frames", type=int, default=17)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--worker", default="/home/maxfridbe/movie-maker/python")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    sys.path.insert(0, a.worker)
    from accelerate import init_empty_weights
    from diffusers import LTX2VideoTransformer3DModel, AutoencoderKLLTX2Video, AutoencoderKLLTX2Audio, FlowMatchEulerDiscreteScheduler
    from diffusers.loaders.single_file_utils import convert_ltx2_transformer_to_diffusers
    from diffusers.models.model_loading_utils import load_gguf_checkpoint
    from diffusers.quantizers.gguf.utils import _replace_with_gguf_linear
    from diffusers.modular_pipelines.ltx2.guider import LTX2Guidance
    from diffusers.modular_pipelines.ltx2.denoise import _pack_latents, _unpack_latents
    from diffusers.modular_pipelines.ltx2.before_denoise import _pack_audio_latents
    from mv_worker.engines.h3_convert import assign_state
    from mv_worker.engines.ltx2_config import transformer_config

    meta = {"width": a.width, "height": a.height, "frames": a.frames, "steps": a.steps, "seed": a.seed}
    save = lambda n, t: np.save(os.path.join(a.out, n + ".npy"), t.detach().float().cpu().numpy())

    t0 = time.time()
    cfg = LTX2VideoTransformer3DModel.load_config(transformer_config(a.components, a.gguf))
    with init_empty_weights():
        model = LTX2VideoTransformer3DModel.from_config(cfg)
    want = {k for k, _ in model.named_parameters()} | {k for k, _ in model.named_buffers()}
    state = convert_ltx2_transformer_to_diffusers(load_gguf_checkpoint(a.gguf))
    for k in [k for k in state if k not in want]:
        alt = k.replace("_adaln_single.", "_adaln.")
        if alt in want:
            state[alt] = state.pop(k)
    _replace_with_gguf_linear(model, torch.float32, state)
    left = assign_state(model, state, compute_dtype=torch.float32)
    if left:
        raise RuntimeError(f"unassigned: {left[:5]}")
    model.eval()
    print(f"denoiser loaded in {time.time() - t0:.0f}s", flush=True)

    vae = AutoencoderKLLTX2Video.from_pretrained(a.components, subfolder="vae", torch_dtype=torch.float32).eval()
    audio_vae = AutoencoderKLLTX2Audio.from_pretrained(a.components, subfolder="audio_vae", torch_dtype=torch.float32).eval()
    from diffusers.pipelines.ltx2.vocoder import LTX2VocoderWithBWE
    vocoder = LTX2VocoderWithBWE.from_pretrained(a.components, subfolder="vocoder", torch_dtype=torch.float32).eval()

    # the shapes the pipeline derives
    F = (a.frames - 1) // 8 + 1
    H, W = a.height // 32, a.width // 32
    fps = 24.0
    la = round(a.frames / fps * 16000 / 160 / 4)
    meta.update(latent=[F, H, W], audio_frames=la)
    g = torch.Generator("cpu").manual_seed(a.seed)
    lat = _pack_latents(torch.randn((1, 128, F, H, W), generator=g, dtype=torch.float32))
    aud = _pack_audio_latents(torch.randn((1, 8, la, 16), generator=g, dtype=torch.float32))
    save("lat_0", lat[0])
    save("aud_0", aud[0])

    emb = {k: torch.from_numpy(np.load(os.path.join(a.text, k + ".npy")))[None] for k in ("video_emb", "audio_emb", "neg_video_emb", "neg_audio_emb")}
    mask = torch.ones((1, 1024), dtype=torch.float32)

    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(a.components, subfolder="scheduler")
    sigmas = np.linspace(1.0, 1 / a.steps, a.steps)
    sched.set_timesteps(sigmas=sigmas, mu=None)
    asched = FlowMatchEulerDiscreteScheduler.from_pretrained(a.components, subfolder="scheduler")
    asched.set_timesteps(sigmas=sigmas, mu=None)
    sched.set_begin_index(0)
    asched.set_begin_index(0)
    meta["sigmas"] = sched.sigmas.tolist()
    vg = LTX2Guidance(guidance_scale=3.0, stg_scale=1.0, modality_scale=3.0, guidance_rescale=0.7, spatio_temporal_guidance_blocks=[28])
    ag = LTX2Guidance(guidance_scale=7.0, stg_scale=1.0, modality_scale=3.0, guidance_rescale=0.7)

    video_coords = model.rope.prepare_video_coords(1, F, H, W, lat.device, fps=fps)
    audio_coords = model.audio_rope.prepare_audio_coords(1, la, lat.device)

    blocks = {}
    hooks = []
    capture = {"on": False}
    for bi in (0, 1, 27, 28, 47):
        def hook(_m, _a, out, bi=bi):
            if capture["on"]:
                blocks[bi] = (out[0][0].clone(), out[1][0].clone())
        hooks.append(model.transformer_blocks[bi].register_forward_hook(hook))

    passes = [("cond", "video_emb", "audio_emb", None, False), ("uncond", "neg_video_emb", "neg_audio_emb", None, False),
              ("stg", "video_emb", "audio_emb", [28], False), ("mod", "video_emb", "audio_emb", None, True)]
    timing = []
    for i, t in enumerate(sched.timesteps):
        vg.set_state(step=i, num_inference_steps=a.steps, timestep=t)
        ag.set_state(step=i, num_inference_steps=a.steps, timestep=t)
        ts = t.expand(1)
        x0v, x0a = {}, {}
        for name, ve, ae, stg, iso in passes:
            t1 = time.time()
            capture["on"] = i == 0 and name == "cond"
            with torch.no_grad():
                v, au = model(hidden_states=lat, audio_hidden_states=aud, encoder_hidden_states=emb[ve], audio_encoder_hidden_states=emb[ae],
                              timestep=ts, audio_timestep=ts, sigma=ts, encoder_attention_mask=mask, audio_encoder_attention_mask=mask,
                              num_frames=F, height=H, width=W, fps=fps, audio_num_frames=la, video_coords=video_coords, audio_coords=audio_coords,
                              isolate_modalities=iso, spatio_temporal_guidance_blocks=stg, use_cross_timestep=True, return_dict=False)
            timing.append(time.time() - t1)
            print(f"  step {i} {name} in {timing[-1]:.0f}s", flush=True)
            if i == 0:
                save(f"s0_{name}_v", v[0])
                save(f"s0_{name}_a", au[0])
            x0v[name] = lat - v.float() * sched.sigmas[i]
            x0a[name] = aud - au.float() * asched.sigmas[i]
        if i == 0:
            for bi, (hv, ha) in blocks.items():
                save(f"s0_blk{bi}_v", hv)
                save(f"s0_blk{bi}_a", ha)
        pv = vg.forward(x0v["cond"], x0v["uncond"], x0v["stg"], x0v["mod"]).pred
        pa = ag.forward(x0a["cond"], x0a["uncond"], x0a["stg"], x0a["mod"]).pred
        vv = (lat - pv) / sched.sigmas[i]
        va = (aud - pa) / asched.sigmas[i]
        lat = sched.step(vv, t, lat, return_dict=False)[0]
        aud = asched.step(va, t, aud, return_dict=False)[0]
        save(f"lat_{i + 1}", lat[0])
        save(f"aud_{i + 1}", aud[0])
    for h in hooks:
        h.remove()
    meta["pass_seconds"] = timing

    with torch.no_grad():
        t1 = time.time()
        from diffusers.modular_pipelines.ltx2.decoders import _denormalize_latents, _denormalize_audio_latents, _unpack_audio_latents
        x = _denormalize_latents(_unpack_latents(lat, F, H, W, 1, 1), vae.latents_mean, vae.latents_std, vae.config.scaling_factor)
        video = vae.decode(x, None, return_dict=False)[0]  # [1, 3, frames, h, w] in [-1, 1]
        video = ((video[0].permute(1, 2, 3, 0) + 1) / 2).clamp(0, 1)
        save("video", video)
        meta["vae_seconds"] = time.time() - t1
        t1 = time.time()
        z = _unpack_audio_latents(_denormalize_audio_latents(aud, audio_vae.latents_mean, audio_vae.latents_std), la, num_mel_bins=16)
        save("audio_z", z[0])
        mel = audio_vae.decode(z, return_dict=False)[0]
        save("mel", mel[0])
        wave = vocoder(mel)
        save("wave", wave[0])
        meta["audio_seconds"] = time.time() - t1
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
