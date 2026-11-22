<p align="center"><img src="docs/logo-card.svg" alt="nextcuda - the swiss-army compute engine: video, chat, image, audio" width="520"></p>

We stand on the shoulders of giants.

# nextcuda

- **nextsycl's control plane, on NVIDIA** - the same front door on :8000, the same `/rpc/<kind>.<method>` API, the same
  jobs, queue and model registry; what runs the models under it is CUDA
- **No Python** - Rust on the control side, CUDA kernels (cuBLAS and our own) on the GPU side; the model stays loaded
  between pictures
- **Made for the cards people have** - two 12 GB cards (an RTX 5070 and an RTX 3060) run chat, pictures, OCR, speech,
  songs, video with sound and 3D: Qwen3.8-27B reads prompts at ~1,200 tok/s over both cards (llama.cpp: 925),
  MiniMax H3 and LTX-2.5 make video with sound across both, Nemotron 3.5 Super (120B) chats on one, its experts streamed
- **Every kind behind one door** - 52 model cards ([models.md](docs/models.md)), each started on the card with room when
  a request names it and let go when idle
- **A single API control plane** - one service, `nextcuda.service`, answers on :8000 and starts a model server when a
  request names its model
- **A Docker-style CLI** - `nextcuda <kind> start | stop | ps | logs | ...`, as a process, not a container
- **Silos and variants** - a model's tuned kernels live in its own library, one per architecture; a class of card
  that wants other ops gets a variant of it, never a fork ([below](#silos-and-variants-how-tuning-stays-out-of-each-others-way))
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
| image, OCR | Qwen-Image 2.1 and 2512, Ming-Image, Z-Image-Turbo, P-H3-Image, Unlimited-OCR, PaddleOCR-VL, olmOCR-2 | one card each; Ming-Image and Z-Image-Turbo the DiT on the 5070, the text encoder on the 3060; P-H3-Image's H3 denoiser over both | [image.md](docs/image.md) |
| upscale | Real-ESRGAN, waifu2x (their ncnn graphs); FlashVSR v1.1 (video super-resolution, a clip as one stream) | one card; FlashVSR over both, pipelined: 0.43 s a frame for 864 x 480 -> 1728 x 960 | [image.md](docs/image.md), [VSR_PORT.md](docs/VSR_PORT.md) |
| vectors | StarVector-1B | one card (3.35 GiB at the peak) | [image.md](docs/image.md) |
| speech | Breeze TTS 2, Qwen3-TTS, VoxCPM2, VibeVoice (several speakers) | one card each; VoxCPM2 int8 fits beside the chat model | [audio.md](docs/audio.md) |
| listening | Whisper large-v3-turbo | one card | [audio.md](docs/audio.md) |
| songs | MiniMax Music 3 (int8), ACE-Step 1.5 | MiniMax: the language model on the 5070, the flow stage and decoder on the 3060 | [audio.md](docs/audio.md) |
| video | MiniMax H3 (the native daemon, `--span`), LTX-2.5 (`ltx2`, `ltx2_upscale`) and the video studio | the denoiser over both cards, nothing streamed: an H3 clip in ~62 s from the SSD; LTX-2.5's 22B DiT 32 blocks on the 5070 and 16 on the 3060, 12.8 s a step at 640 x 384 for 5 s | [models.md](docs/models.md), [ports/ltx2.md](docs/ports/ltx2.md) |
| 3D | TRELLIS.2, Hunyuan3D-2mini and 2.1 (and its paint alone), UniRig, SkinTokens, RigNet, AnimateAnyMesh, UniMate | one card each | [3d.md](docs/3d.md) |

Not here yet: HunyuanImage 2.1; the bigger cards' forms (the models [marked too big](docs/models.md)). The answers'
`nextsycl` result key keeps its name (the platform reads it).

## Silos and variants: how tuning stays out of each other's way

Every kernel an engine runs comes from one of three layers. Each GPU picks an op from the most specific layer that has
it when the engine loads: **variant → silo → shared kernel** (nextsycl's layout, in CUDA terms).

| | shared kernel | silo | variant |
|---|---|---|---|
| **Scope** | every engine of a kind | one model architecture (never one card) | one model on one class of hardware |
| **Where** | `kernels/ns`, `kernels/diffusion`, an engine's kernels another engine calls | `kernels/<kind>/<arch>/silo/` | `kernels/<kind>/<arch>/silo/variants/<name>/` + `variant.yml` |
| **Builds to** | the kind's library | `dist/silo/libnextcuda-<arch>.so` (every card's code in it) | `dist/silo/libnextcuda-<arch>@<name>.so` and its `.yml` |
| **Holds** | the plain, correct op | the model's ops tuned for its shapes | only the ops that differ from the silo (a delta, under the silo's C names) |
| **Settings** | - | - | optional `settings:` in `variant.yml`: engine options (the names `--opt-NAME` takes) as the engine's defaults on its cards - a path the engine already has (an option with enumerated values), never a memory size |
| **Chosen by** | always there (the fallback) | the engine's arch | its `match` against what the GPU reports: `vram_mb_min/max`, `units_min/max` (SMs), `mhz_min/max`, `cc_min/max` (compute capability, 86 for 8.6); a `name` substring as a last resort. Two variants of one silo never both match a card |
| **Override** | `NS_SILO=0` (silo off) | `NS_SILO=0` | `NS_SILO_VARIANT=<name>\|none`, or a model card's `silo.variant` |
| **Who may change it** | a bug fix or new type, with every engine that calls it re-checked | that model's porter | whoever tunes that hardware |
| **Must prove** | every caller's check and bench | the same output as the shared kernel; before and after numbers | the same output as the silo (`NS_SILO_VARIANT=none` against it); before and after numbers on its hardware; a gate baseline recorded on a card it matches, and the gate's table from that card |

**What it protects against**
- **Pull requests fighting.** Someone tuning for an 8 GB card, or a card with fewer SMs, adds a variant. The default
  silo the RTX 5070 and RTX 3060 run doesn't change, so two tunings for different hardware never edit the same file.
- **One model's speed-up slowing another.** Tuning goes in that model's silo, never in a shared kernel, so another
  engine's speed and output can't move.
- **Silent output changes.** Each layer must give the same output as the layer below it, and the merge gate
  (`tooling/gate.sh`) compares each engine's output exactly for a fixed seed (an LLM's greedy tokens and their
  log-probabilities, a picture's, a sound's or a mesh's bytes); its throughput must stay within 3% on each card a
  change touches.
- **Hard-coded memory.** A variant never changes a memory plan. Memory is planned from what the device reports
  (`Gpu::free_or_guess`), so a smaller card gets a smaller plan without a fork. A variant's `settings` may not set an
  option that sizes memory (`TOKENS`, `GIB`, `MIB`, `BYTES`).
- **Forking an engine for one card.** A card that wants another host-side path - a chunking, a fusion, an expert
  placement - gets it as an engine option with enumerated values (`value: "spread|pack"`), and its variant picks one
  in `settings`. A new path is an engine change under the gate; the variant only chooses. Precedence, highest first:
  a request's options, `--opt` / the environment, the model card's settings, the variant's, the engine's default.
- **Two tunings claiming one card.** The architecture test fails when two variants of one silo can match the same
  card (their numeric ranges meet; a name never separates them). The fix is a tighter `match`. At run time the first
  by name is taken, with a warning.
- **Hardware we don't own.** The gate runs on a contributor's own box (`NS_GATE_HOST=local`), and its baselines are
  per card (`rtx-3060-28u-12g`: the name, SMs and memory the card reports). A variant's pull request records its
  card's baseline (`NS_GATE_RECORD=1`), commits it, and pastes the gate's table from that card; the architecture test
  fails on a variant with no baseline on a card it matches.
- **Mixed boxes.** The layer is chosen per GPU (`nextcuda_core::silo_for(arch, gpu)`). The 5070 beside the 3060 each
  get their own ops. An engine that binds once per process (`nextcuda_core::silo(arch)`) takes a variant only when
  every GPU matches it, and a variant's settings (process-wide) apply only when every GPU the engine loads on matches
  it. `nextcuda silo [ARCH]` shows what each GPU takes and the settings that come with it.

The rules are enforced by tests in `cli/nextcuda/tests/architecture.rs` and by the gate, not only by review
(CONTRIBUTING.md rule 5). nextcuda has no variant yet: the 5070 and the 3060 run the default silos (a variant is added
when a measurement on one class of card justifies it); nextsycl's example is its `kernels/llm/qwen35/silo/variants/b65/`.

## Docs

- [models.md](docs/models.md) - every model card: what it is, its size, whether it runs here
- [control-plane.md](docs/control-plane.md) - the front door, downloads, shared cards, maintenance mode, `system.ping`
- [llm.md](docs/llm.md) - Qwen3.8-27B over both cards, Nemotron 3.5 Super: the numbers, the settings, the server
- [image.md](docs/image.md) - Qwen-Image and its clocks, Ming-Image designs, Z-Image-Turbo, upscaling, StarVector
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
- [ggml](https://github.com/ggml-org/ggml) - the quantized formats, and the designs nextcuda's int8 products learned
  from: MMQ's tile loop, load_tiles' reads and stream-k (ported, MIT; ggml-cuda's MMQ / MMVQ themselves were vendored
  for Qwen3.8 until 2026-11-14). Nemotron's products (Strata's kernels before) and its CPU experts (ggml-cpu before)
  are nextcuda's own since 2026-11-13, and stable-diffusion.cpp ran Qwen-Image here until then; their numbers stay in
  the benchmark docs.
- Qwen-Image 2.1 and 2512 (Qwen), the GGUFs of leejet, unsloth and realrebelai, and Viggle's and lightx2v's LoRAs.
