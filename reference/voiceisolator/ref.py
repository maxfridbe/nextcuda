"""Inworld's Voice Isolator in its own package (inworld-voice-isolation, github.com/inworld-ai/voice-isolator), on the
CPU in float32 - the reference for nextcuda's audio/voiceisolator: the stages' tensors of one chunk and the whole
isolation of each recording, dumped for `nextcuda audio check`.

    CUDA_VISIBLE_DEVICES= PYTHONPATH=<voice-isolator>/src python -I ref.py CHECKPOINT --out DIR CLIP [CLIP ...] [--bf16]

Dev tooling only: nothing in nextcuda runs Python.

Dumps (DIR):
  input48.npy      the first clip, mono at 48 kHz (to_mono, kaiser_best resampling) - its first 8 s chunk is the stages'
  spec.npy         [bins, frames, 2] the chunk's STFT (real, imaginary)
  tokens.npy       [bands, frames, width] the band encoders' tokens with the band embedding
  block<i>.npy     [bands, frames, width] after block i (0..6); block0_time.npy (after its time transformer),
                   block1_time.npy (after its time Mamba)
  raw.npy          [bins, frames, 4] the heads' output (mask re, im, residual re, im)
  chunk.npy        [samples] the network's output for the chunk (forward)
  clip<k>.npy      clip k isolated (VoiceIsolator.isolate: offline, chunks of 8 s over 2 s, at the clip's own rate);
  clip<k>_in.npy   the clip as read (mono), clip<k>.json its rate and name; with --bf16 also clip<k>_bf16.npy
                   (precision bf16: the network under bfloat16 autocast, as the package serves it on a CUDA card)
"""
import argparse
import json
import os

import numpy as np
import soundfile as sf
import torch

from inworld.voice_isolation import audio as audio_lib
from inworld.voice_isolation import isolator as iso


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bf16", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    save = lambda name, t: np.save(os.path.join(a.out, name), np.ascontiguousarray(t.detach().float().cpu().numpy()))

    vi = iso.VoiceIsolator(a.checkpoint, device="cpu", precision="fp32")
    model = vi.model

    # the stages of the first clip's first chunk
    x, sr = audio_lib.load_audio(a.clips[0])
    x48 = audio_lib.resample(x, sr, 48000)
    save("input48.npy", x48)
    chunk = x48[: int(model.config.chunk_seconds * 48000)][None]
    with torch.inference_mode():
        spec = model.stft(chunk)
        save("spec.npy", torch.view_as_real(spec[0]))
        f = model.encode(spec)
        save("tokens.npy", f[0])
        for i, block in enumerate(model.blocks):
            if i == 0:
                b, n, t, w = f.shape
                y = block.time_transformer(f.reshape(b * n, t, w), False)
                save("block0_time.npy", y.reshape(n, t, w))
            if i == 1:
                b, n, t, w = f.shape
                y = f.reshape(b * n, t, w)
                y = y + block.time_mamba(y, False)
                save("block1_time.npy", y.reshape(n, t, w))
            f = block(f, False)
            save(f"block{i}.npy", f[0])
        raw = model.denoise_decoder(f)
        save("raw.npy", raw[0])
        out = chunk + model.istft(model.correction(raw, spec), chunk.shape[-1])
        save("chunk.npy", out[0])
        save("chunk_fwd.npy", model(chunk)[0])

    # every clip, whole, as the package isolates it
    for k, path in enumerate(a.clips):
        x, sr = audio_lib.load_audio(path)
        save(f"clip{k}_in.npy", x)
        y = vi.isolate(x, sr)
        save(f"clip{k}.npy", y)
        with open(os.path.join(a.out, f"clip{k}.json"), "w") as fh:
            json.dump({"name": os.path.basename(path), "rate": sr, "samples": int(x.shape[-1])}, fh)
        print(f"clip{k}: {os.path.basename(path)} {sr} Hz {x.shape[-1] / sr:.2f} s", flush=True)
    if a.bf16:
        vb = iso.VoiceIsolator(model, device="cpu", precision="bf16")
        for k, path in enumerate(a.clips):
            x, sr = audio_lib.load_audio(path)
            save(f"clip{k}_bf16.npy", vb.isolate(x, sr))


if __name__ == "__main__":
    main()
