#!/usr/bin/env python3
"""VibeVoice 1.5B's own code (Microsoft's, as the `yvoice` package carries it) on the same GGUF, in float32 on the
CPU: the dumps `nextcuda audio check <dir> --model vibevoice-1.5b` compares the engine's stages with.

    ref.py --gguf vibevoice-1.5b-bf16.gguf --model-dir <dir with config.json, preprocessor_config.json, tokenizer/>
           --ref speaker1.wav [--ref speaker2.wav] --text "Speaker 1: ...\\nSpeaker 2: ..." --out DIR
           [--frames 40] [--steps 10] [--cfg 1.3] [--seed 42]

The model is built as the platform's engine builds it (python/cp_worker/kinds/audio/vibevoice.py): the config beside
the GGUF, every tensor dequantized (float32 here), yvoice's VibeVoiceForConditionalGenerationInference and its
processor from their parts. The generation is yvoice's `generate`, untouched; what it does is recorded by wrapping
the pieces it calls:

    tokens.npy          the prompt's ids [L];  speech_mask.npy: where the voice rows go [L] (int8)
    ref_wav_<i>.npy     speaker i's recording as the processor hands it to the encoder (24 kHz, -25 dB FS)
    enc_mean_<i>.npy    the acoustic encoder's means for it [T, 64]
    voice_lat_<i>.npy   the latents the prompt used (mean + the gaussian draw) [T, 64] - after the bias and scale
    embeds.npy          the language model's prompt rows [L, 1536] (voice rows the connector's)
    calls.npy           every language-model call's last input row and last hidden state [n, 2, 1536]; calls.json
                        says which were the negative (guidance) passes and their sequence lengths
    cond.npy, neg.npy   each frame's positive / negative condition [F, 1536]; noise.npy its starting noise [F, 2, 64]
    latent.npy          each frame's latent from the diffusion head [F, 64]
    chunk.npy           each frame's decoded samples [F, 3200]; sem.npy its semantic features [F, 128]
    next.npy            each frame's next input row (acoustic + semantic connectors) [F, 1536]
    sequence.npy        the generated tokens; audio.npy the take (24 kHz); meta.json the settings and special ids
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--ref", action="append", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--cfg", type=float, default=1.3)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(os.cpu_count() or 4)

    from gguf_connector.quant import dequantize
    from gguf_connector.reader import GGUFReader
    from transformers.modeling_utils import no_init_weights
    from yvoice.modular.configuration_vibevoice import VibeVoiceConfig
    from yvoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
    from yvoice.modular.modular_vibevoice_text_tokenizer import VibeVoiceTextTokenizerFast
    from yvoice.processor.vibevoice_processor import VibeVoiceProcessor
    from yvoice.processor.vibevoice_tokenizer_processor import VibeVoiceTokenizerProcessor

    # the platform engine's helpers: the cache-preparation fix and the script form
    sys.path.insert(0, os.path.expanduser("~/creative-platform/python"))
    from cp_worker.kinds.audio.vibevoice import fit_cache_prep

    config = VibeVoiceConfig.from_pretrained(a.model_dir)
    with no_init_weights():
        model = VibeVoiceForConditionalGenerationInference(config)
    sd = {}
    for t in GGUFReader(a.gguf).tensors:
        w = np.ascontiguousarray(dequantize(t.data, t.tensor_type)).astype(np.float32)
        sd[t.name] = torch.from_numpy(w.copy())
    missing, _ = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if k != "lm_head.weight"]
    assert not missing, missing[:5]
    del sd
    model.tie_weights()
    model = model.float().eval()
    fit_cache_prep(model)
    model.set_ddpm_inference_steps(num_steps=a.steps)

    pc = json.load(open(os.path.join(a.model_dir, "preprocessor_config.json")))
    au = pc.get("audio_processor") or {}
    tokenizer = VibeVoiceTextTokenizerFast.from_pretrained(os.path.join(a.model_dir, "tokenizer"))
    processor = VibeVoiceProcessor(
        tokenizer=tokenizer,
        audio_processor=VibeVoiceTokenizerProcessor(sampling_rate=au.get("sampling_rate", 24000), normalize_audio=au.get("normalize_audio", True),
                                                    target_dB_FS=au.get("target_dB_FS", -25), eps=au.get("eps", 1e-6)),
        speech_tok_compress_ratio=pc.get("speech_tok_compress_ratio", 3200), db_normalize=pc.get("db_normalize", True))

    import soundfile as sf

    def load(p):
        w, sr = sf.read(p, dtype="float32", always_2d=False)
        if w.ndim > 1:
            w = w.mean(axis=1)
        if sr != 24000:
            import librosa
            w = librosa.resample(w, orig_sr=sr, target_sr=24000).astype(np.float32)
        return w

    wavs = [load(p) for p in a.ref]
    script = a.text.replace("\\n", "\n")
    inputs = processor(text=[script], voice_samples=[wavs], return_tensors="pt", padding=True, return_attention_mask=True)
    np.save(f"{a.out}/tokens.npy", inputs["input_ids"][0].numpy().astype(np.int64))
    np.save(f"{a.out}/speech_mask.npy", inputs["speech_input_mask"][0].numpy().astype(np.int8))
    st, sm = inputs["speech_tensors"], inputs["speech_masks"]
    for i in range(st.shape[0]):
        n = int(sm[i].sum())
        np.save(f"{a.out}/ref_wav_{i}.npy", st[i].numpy()[: min(st.shape[1], n * 3200)].astype(np.float32))
        with torch.no_grad():
            m = model.model.acoustic_tokenizer.encode(st[i : i + 1].unsqueeze(1)).mean[0]
        np.save(f"{a.out}/enc_mean_{i}.npy", m[:n].numpy().astype(np.float32))

    rec = {"calls": [], "neg": [], "cond": [], "negc": [], "noise": [], "latent": [], "chunk": [], "sem": [], "embeds": None, "voice": None}

    # the prompt's voice rows: what _process_speech_inputs makes (its gaussian draw included)
    orig_psi = model._process_speech_inputs

    def psi(speech_tensors, speech_masks, speech_type="audio"):
        feats, conn = orig_psi(speech_tensors, speech_masks, speech_type)
        rec["voice"] = (feats.detach().clone(), speech_masks.clone())
        return feats, conn

    model._process_speech_inputs = psi

    # every language-model call: its rows in, its last hidden state out
    lm = model.model.language_model

    def hook(mod, args, kwargs, out):
        e = kwargs.get("inputs_embeds")
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        if rec["embeds"] is None:
            rec["embeds"] = e[0].detach().clone()
        rec["calls"].append((e[0, -1].detach().clone(), h[0, -1].detach().clone(), int(kwargs["past_key_values"].get_seq_length())
                             if kwargs.get("past_key_values") is not None else 0, int(e.shape[1])))

    lm.register_forward_hook(hook, with_kwargs=True)

    # each frame's diffusion: the conditions, the noise, the latent (sample_speech_tokens as yvoice has it)
    def sample(condition, neg_condition, cfg_scale=3.0):
        model.model.noise_scheduler.set_timesteps(model.ddpm_inference_steps)
        rec["cond"].append(condition[0].detach().clone())
        rec["negc"].append(neg_condition[0].detach().clone())
        condition = torch.cat([condition, neg_condition], dim=0)
        speech = torch.randn(condition.shape[0], model.config.acoustic_vae_dim).to(condition)
        rec["noise"].append(speech.detach().clone())
        for t in model.model.noise_scheduler.timesteps:
            half = speech[: len(speech) // 2]
            combined = torch.cat([half, half], dim=0)
            eps = model.model.prediction_head(combined, t.repeat(combined.shape[0]).to(combined), condition=condition)
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            speech = model.model.noise_scheduler.step(eps, t, speech).prev_sample
        out = speech[: len(speech) // 2]
        rec["latent"].append(out[0].detach().clone())
        return out

    model.sample_speech_tokens = sample

    at = model.model.acoustic_tokenizer
    orig_dec = at.decode

    def dec(latents, **kw):
        y = orig_dec(latents, **kw)
        rec["chunk"].append(y[0, 0].detach().clone())
        return y

    at.decode = dec
    se = model.model.semantic_tokenizer
    orig_enc = se.encode

    def enc(audio, **kw):
        o = orig_enc(audio, **kw)
        rec["sem"].append(o.mean[0, -1].detach().clone())
        return o

    se.encode = enc

    torch.manual_seed(a.seed)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=a.frames, cfg_scale=a.cfg, tokenizer=processor.tokenizer,
                             generation_config={"do_sample": False}, show_progress_bar=False)

    f32 = lambda xs: np.stack([x.float().numpy() for x in xs]).astype(np.float32)
    np.save(f"{a.out}/embeds.npy", rec["embeds"].float().numpy().astype(np.float32))
    feats, masks = rec["voice"]
    for i in range(feats.shape[0]):
        n = int(masks[i].sum())
        np.save(f"{a.out}/voice_lat_{i}.npy", feats[i, :n].float().numpy().astype(np.float32))
    np.save(f"{a.out}/calls.npy", np.stack([np.stack([c[0].float().numpy(), c[1].float().numpy()]) for c in rec["calls"]]).astype(np.float32))
    if rec["cond"]:
        np.save(f"{a.out}/cond.npy", f32(rec["cond"]))
        np.save(f"{a.out}/neg.npy", f32(rec["negc"]))
        np.save(f"{a.out}/noise.npy", f32(rec["noise"]))
        np.save(f"{a.out}/latent.npy", f32(rec["latent"]))
        np.save(f"{a.out}/chunk.npy", f32(rec["chunk"]))
        np.save(f"{a.out}/sem.npy", f32(rec["sem"]))
        lat = torch.stack(rec["latent"])
        semf = torch.stack(rec["sem"])
        with torch.no_grad():
            nxt = model.model.acoustic_connector(lat) + model.model.semantic_connector(semf)
        np.save(f"{a.out}/next.npy", nxt.float().numpy().astype(np.float32))
    seq = out.sequences[0].numpy().astype(np.int64)
    np.save(f"{a.out}/sequence.npy", seq[inputs["input_ids"].shape[1]:])
    sp = out.speech_outputs[0]
    audio = sp.float().numpy().reshape(-1).astype(np.float32) if sp is not None else np.zeros(0, np.float32)
    np.save(f"{a.out}/audio.npy", audio)
    sf.write(f"{a.out}/audio.wav", audio, 24000)
    json.dump({
        "text": script, "refs": a.ref, "frames": a.frames, "steps": a.steps, "cfg": a.cfg, "seed": a.seed,
        "speech_start": tokenizer.speech_start_id, "speech_end": tokenizer.speech_end_id, "speech_diffusion": tokenizer.speech_diffusion_id,
        "eos": tokenizer.eos_token_id, "scaling": float(model.model.speech_scaling_factor), "bias": float(model.model.speech_bias_factor),
        "calls": [{"cache": c[2], "rows": c[3]} for c in rec["calls"]],
        "timesteps": [int(t) for t in model.model.noise_scheduler.timesteps],
    }, open(f"{a.out}/meta.json", "w"), indent=1)
    print(f"{len(rec['latent'])} frames, {len(audio) / 24000:.2f} s, {len(rec['calls'])} language-model calls -> {a.out}")


if __name__ == "__main__":
    main()
