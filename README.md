<p align="center"><img src="docs/logo-card.svg" alt="nextcuda - the swiss-army compute engine: video, chat, image, audio" width="520"></p>

We stand on the shoulders of giants.

# nextcuda

- **nextsycl's control plane, on NVIDIA** - the same front door on :8000, the same `/rpc/<kind>.<method>` API, the same
  jobs, queue and model registry; what runs the models under it is CUDA
- **No Python** - Rust on the control side, CUDA ([stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)'s
  ggml-cuda kernels) on the GPU side; the model stays loaded between pictures
- **Made for the cards people have** - two 12 GB cards (an RTX 5070 and an RTX 3060) run chat, pictures, OCR, speech,
  songs, video with sound and 3D: Qwen3.8-27B reads prompts at ~1,200 tok/s over both cards (llama.cpp: 925),
  MiniMax H3 makes video with sound across both, Nemotron 3.5 Super (120B) chats on one, its experts streamed
- **Every kind behind one door** - 43 model cards ([models.md](docs/models.md)), each started on the card with room when
  a request names it and let go when idle
- **A single API control plane** - one service, `nextcuda.service`, answers on :8000 and starts a model server when a
  request names its model
- **A Docker-style CLI** - `nextcuda <kind> start | stop | ps | logs | ...`, as a process, not a container
- **Silo architecture** - a model's tuned kernels live in its own library; the shared ones stay as they are
- **Test driven** - the architecture rules, the model cards and the kernels are checked by `cargo test` and
  `nextcuda image selftest`

It is the same program as [nextsycl](https://github.com/) with the SYCL runtime taken out and CUDA put in, so what
you know of one is what you know of the other. This page is the front door; the detail is in [the docs](#docs).

## The control plane

`http://<host>:8000/` - every card, what each holds and what the enabled models are vouched for there; every model
by kind with its cards (tick which may hold it), enable / disable, start / stop and its own page; the models a card
offers that are not downloaded yet; the front ends (Open WebUI on :3000, the video studio). The API and Config tabs
are the OpenAPI document and `nextcuda.conf`.

<p align="center"><img src="docs/screenshots/control-plane-gpus.png" alt="the GPUs: memory in use by process, and what each enabled model is vouched for" width="900"></p>

<p align="center"><img src="docs/screenshots/control-plane-models.png" alt="the models by kind: chat, images and OCR, audio, video - their cards, enable, start and their pages" width="900"></p>

<p align="center"><img src="docs/screenshots/control-plane-downloads.png" alt="the models not downloaded, a download each, and the front ends" width="900"></p>

Models download on demand, cards are shared rather than fought over, and a maintenance mode sets the box aside while
callers are told; `system.ping` is the cheap "is it there, does it take work" call. The API, the rules and the details
are in [control-plane.md](docs/control-plane.md).

## Quick start

```sh
nextcuda models pull qwen-image-2.1-q8 --from ~/models/qwen21   # link the files you already have
nextcuda image gen "a small red fox in fresh snow, soft morning light" --model qwen-image-2.1-q8 --size 768x768 \
        --steps 12 --seed 7 --out fox.png --local
nextcuda gpus                                   # each GPU: memory, units, copy rates
nextcuda image selftest --gpu 0                 # every kernel of the engine against host arithmetic
```

Through the front door:

```sh
TOKEN=$(python3 -c "import json;print(json.load(open('serve.json'))['token'])")
curl -s -X POST http://127.0.0.1:8000/rpc/image.generate -H "Authorization: Bearer $TOKEN" \
     -d '{"model": "qwen-image-2.1-q8", "prompt": "a lighthouse at dusk", "size": "768x768", "steps": 12,
          "wait": false, "label": "lighthouse", "reentrant_id": "lh-1"}'
curl -s -X POST http://127.0.0.1:8000/rpc/image.status -H "Authorization: Bearer $TOKEN" -d '{"reentrant_id": "lh-1"}'
```

`./build.sh` builds the program, `tooling/deploy.sh <host>` puts it on a GPU host as a systemd user service; settings are
`nextcuda.conf`, then the environment ([environment.md](docs/environment.md)). [build.md](docs/build.md) has the rest.

## What runs where

| kind | models | on the two cards | detail |
|---|---|---|---|
| chat | Qwen3.8-27B (DeltaNet + attention, MTP drafts, vision), Nemotron 3.5 Super | Qwen3.8 over both, pipelined; Nemotron on the 5070 | [llm.md](docs/llm.md) |
| image, OCR | Qwen-Image 2.1 and 2512, Ming-Image, Unlimited-OCR, PaddleOCR-VL, olmOCR-2 | one card each; Ming-Image the DiT on the 5070, its text encoder on the 3060 | [image.md](docs/image.md) |
| upscale | Real-ESRGAN, waifu2x (their ncnn graphs) | one card | [image.md](docs/image.md) |
| vectors | StarVector-1B | one card (3.35 GiB at the peak) | [image.md](docs/image.md) |
| speech | Breeze TTS 2, Qwen3-TTS, VoxCPM2, VibeVoice (several speakers) | one card each; VoxCPM2 int8 fits beside the chat model | [audio.md](docs/audio.md) |
| listening | Whisper large-v3-turbo | one card | [audio.md](docs/audio.md) |
| songs | MiniMax Music 3 (int8), ACE-Step 1.5 | MiniMax: the language model on the 5070, the flow stage and decoder on the 3060 | [audio.md](docs/audio.md) |
| video | MiniMax H3 (the native daemon, `--span`) and the video studio | the denoiser over both cards, nothing streamed: a clip in ~62 s from the SSD | [models.md](docs/models.md) |
| 3D | TRELLIS.2, Hunyuan3D-2mini and 2.1 (and its paint alone), UniRig, SkinTokens, RigNet, AnimateAnyMesh, UniMate | one card each | [3d.md](docs/3d.md) |

Not here yet: HunyuanImage 2.1; the bigger cards' forms (the models [marked too big](docs/models.md)). The answers'
`nextsycl` result key keeps its name (the platform reads it).

## Docs

- [models.md](docs/models.md) - every model card: what it is, its size, whether it runs here
- [control-plane.md](docs/control-plane.md) - the front door, downloads, shared cards, maintenance mode, `system.ping`
- [llm.md](docs/llm.md) - Qwen3.8-27B over both cards, Nemotron 3.5 Super: the numbers, the settings, the server
- [image.md](docs/image.md) - Qwen-Image and its clocks, Ming-Image designs, upscaling, StarVector
- [audio.md](docs/audio.md) - speech, conversation, listening, songs
- [3d.md](docs/3d.md) - meshes, rigs, Hunyuan3D 2.1 paint on a 12 GB card
- [build.md](docs/build.md) - how it is built, running it as a service, clocking and checking it
- [lessons.md](docs/lessons.md) - what the measurements taught
- [PHASES.md](docs/PHASES.md) - what is done, the clocks, and what comes next
- [architecture.md](docs/architecture.md) - the layers, the contracts, the silos (nextsycl's, which holds)
- [environment.md](docs/environment.md) - every variable
- [model-cards/README.md](model-cards/README.md) - the model card format
- [CONTRIBUTING.md](CONTRIBUTING.md) - the rules and how a model is ported

## Standing on

- [nextsycl](https://github.com/) - the control plane, the contracts, the web front end and the rules.
- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) and [ggml](https://github.com/ggml-org/ggml)
  - the CUDA kernels the engines run on.
- Qwen-Image 2.1 and 2512 (Qwen), the GGUFs of leejet, unsloth and realrebelai, and Viggle's and lightx2v's LoRAs.
