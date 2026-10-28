# Model cards

One YAML file a model, `<kind>/<id>.yml` (kind: `llm`, `image`, `video`, `audio`, `lora`): everything nextsycl knows
about a model it offers, read when it is needed - not built into the program. `nextsycl models search` lists the
cards, `nextsycl models pull <id>` downloads a card's files, checks them and registers the model with the card's
settings; the front door's `describePromptGuideMD` answers with the card's guide. A release carries this folder
beside the program; `NS_MODEL_CARDS` names another one. A new model is a new card: no code changes unless its
architecture is new (an engine: CONTRIBUTING.md).

```yaml
# Qwen3.8-27B UD-Q4_K_M (4-bit; the B70, 128K + 32K)
id: qwen3.8-27b-q4              # the file's name; what the registry, the API and the CLI call it
kind: llm                       # the file's folder
arch: qwen35                    # the engine that runs it (GGUF general.architecture, or the engine's ARCH)
title: Qwen3.8-27B UD-Q4_K_M (4-bit; the B70, 128K + 32K)
license: apache-2.0             # the licence of the weights it downloads (an SPDX id, else a short name)
commercial: true                # false only when that licence forbids commercial use: the pages tag the model "non-commercial"
license_note: 'A sentence.'     # optional: the conditions a commercial licence puts on use (territories, user or revenue limits, a notice)
about: Qwen's dense Qwen3.8 27B ...
gpus: '0'                       # the GPUs it loads on: '0', '1', '0,1', or 'all' (weakest first)
vram_gib: 29.0                  # its memory on each card when it runs, as measured (the front door keeps the card
                                #   free for it before it has measured the model itself; else from the files' sizes)
headroom_gib: 3.3               # optional: of vram_gib, the working memory beyond the weights (KV for its context, arenas,
                                #   a song's or an image's activations) - the front door shows it apart (else: vram_gib less
                                #   the files' share of the card; none when the files do not all sit on the card)
ctx: 131072,32768               # llm: each session's context
tools: true                     # llm: tool calls offered to it (false: dropped)
tasks: true                     # llm: a front end's background tasks (titles, tags) accepted
env: {}                         # settings its engine reads; '{role}' is that file's path once pulled
silo:                           # its tuned kernels (CONTRIBUTING rule 5): kernels/<kind>/<name>/silo, built as
  name: qwen35                  #   dist/silo/libnextsycl-<name>.so (default: the architecture's own, when built)
  required: true                # no load without it (default false: the shared kernels for what it lacks)
files:                          # by role: a file defined here, or another card's file by its key
  model:
    key: q27-q4                 # the file's id among all cards (a file is downloaded once, defined once)
    url: https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/<revision>/Qwen3.8-27B-UD-Q4_K_M.gguf
    size: 16464440224           # bytes
    sha256: 322e194f...         # checked after the download
    path: qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_M.gguf   # under NS_MODELS
  mtp: q27-mtp                  # defined in qwen3.8-27b-q8.yml
guide: guides/chat.md           # how to prompt it (Markdown, relative to this folder)
```

`silo` also takes a name alone (`silo: qwen35`) or `none` (the shared kernels only); without it a model runs on its
architecture's silo when one is built. `nextsycl models search` shows each model's (`-`: the shared kernels). The
registry entry carries it as the loader's settings (`NS_SILO_NAME`, `NS_SILO_REQUIRED`, `NS_SILO=0`).

Other keys: `local` (role: what it is - a part made on the machine rather than downloaded, found in nextsycl.conf
or another registry entry), `base` (a lora: the model it rides on), `defaults` (image / video / audio: the
request's defaults - steps, cfg, size, seconds ...), `speech` (audio: `voices`, `design`, `clone`).

### Licences

Every card has `license` and `commercial` (and `license_note` where the licence puts conditions on commercial use):
the licence of the weights its files download - for a quantization the base model's too, and the stricter applies.
`commercial: false` means the licence forbids commercial use (CC BY-NC, research-only, a custom "non-commercial"
licence); a licence that allows it with conditions (a territory, a user or revenue limit, a notice to show, a
copyleft) is `commercial: true` and the conditions go in `license_note`. The pull copies the three keys into the
registry entry; an entry registered before (its `catalog` names a card) gets them from its card when the registry is
read - the registry's file is not rewritten. `system.models` / `system.state` rows, `/api/info` of the model servers,
the video studio's denoiser list and `nextcuda models search` carry them, and every page shows a "non-commercial" tag
(the licence and its note as its tooltip) beside a model whose `commercial` is false. The licences as read from each
repository's `cardData.license` and its licence text (2026-10-27):

| cards | licence (`license`) | commercial | source |
| --- | --- | --- | --- |
| acestep-1.5, acestep-1.5-base | mit | yes | https://huggingface.co/ACE-Step/Ace-Step1.5 (Qwen3-Embedding inside: Apache-2.0) |
| breeze-tts-2, breeze-tts-2-q4 | breezeblue-research-and-non-commercial-1.1 | **no** | https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE ("does not grant any commercial rights"; the GGUF repo https://huggingface.co/cstr/breeze-tts-2-GGUF follows it) |
| minimax-music3, minimax-music3-int8 | minimax-music3-community | yes, with a note | https://huggingface.co/MiniMaxAI/MiniMax-Music3/blob/main/LICENSE (show the name on a commercial product's interface; written authorization above US$20 million yearly revenue) |
| qwen3-tts-base, -custom, -design | apache-2.0 | yes | https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base (and -CustomVoice, -VoiceDesign) |
| vibevoice-1.5b, vibevoice-1.5b-int8 | mit | yes | https://huggingface.co/microsoft/VibeVoice-1.5B, https://huggingface.co/gguf-org/vibevoice-gguf (tokenizer: Qwen2.5-1.5B, Apache-2.0) |
| voxcpm2, voxcpm2-int8 | apache-2.0 | yes | https://huggingface.co/openbmb/VoxCPM2 |
| whisper-large-v3-turbo | mit | yes | https://huggingface.co/openai/whisper-large-v3-turbo |
| ming-image-0.1-design | mit | yes | https://huggingface.co/inclusionAI/Ming-Image-0.1-Design, https://huggingface.co/Comfy-Org/Ming-Image |
| moss-image-sft-q8, -turbo4-q8, -turbo8-q8, -dmd2-q8 | apache-2.0 | yes | https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT (the text encoder: Qwen3.5-9B, Apache-2.0) |
| olmocr-2-7b-q8 | apache-2.0 | yes | https://huggingface.co/allenai/olmOCR-2-7B-1025 |
| paddleocr-vl-1.6 | apache-2.0 | yes | https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6 |
| qwen-image-2.1-q8, -q6k, -rrq4 | qwen-research | **no** | https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE (grant "FOR NON-COMMERCIAL PURPOSES ONLY"; the GGUFs https://huggingface.co/leejet/Qwen-Image-2.1-GGUF say the licence follows the original) |
| qwen-image-2512-q4km | apache-2.0 | yes | https://huggingface.co/Qwen/Qwen-Image-2512 (Lightning LoRA and Qwen2.5-VL: Apache-2.0) |
| real-esrgan | bsd-3-clause | yes | https://github.com/xinntao/Real-ESRGAN |
| waifu2x | mit | yes | https://github.com/nagadomi/waifu2x |
| starvector-1b-im2svg | apache-2.0 | yes | https://huggingface.co/starvector/starvector-1b-im2svg |
| unlimited-ocr-q8, -q4 | mit | yes | https://huggingface.co/baidu/Unlimited-OCR |
| nemotron-3.5-super-iq1m | openmdw-1.1 | yes | https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16 (OpenMDW 1.1, permissive) |
| qwen3.8-27b-gsq-iq3s, -iq3s-long | apache-2.0 | yes | https://huggingface.co/Qwen/Qwen3.8-27B |
| animateanymesh | apache-2.0 | yes, with a note | https://huggingface.co/JarrentWu/AnimateAnyMesh (tokenizer files from Stable Diffusion 1.5: CreativeML OpenRAIL-M) |
| hunyuan3d-2.1, hunyuan3d-2.1-paint | tencent-hunyuan-community | yes, with a note | https://huggingface.co/tencent/Hunyuan3D-2.1/blob/main/LICENSE (not licensed in the EU, UK, South Korea; above 1 million monthly active users ask Tencent) |
| hunyuan3d-2mini-turbo | tencent-hunyuan-community | yes, with a note | https://huggingface.co/tencent/Hunyuan3D-2mini/blob/main/LICENSE (the same terms, for 2.0) |
| pixal3d | mit | yes, with a note | https://huggingface.co/TencentARC/Pixal3D (MIT); DINOv3 features under Meta's DINOv3 licence |
| trellis2, trellis2-512 | mit | yes, with a note | https://huggingface.co/microsoft/TRELLIS.2-4B (MIT); DINOv3 as above; BiRefNet (MIT) |
| rignet | gpl-3.0 | yes, with a note | https://github.com/zhan-xu/RigNet (GPLv3; UMass offers a commercial licence) |
| unirig | mit | yes | https://huggingface.co/VAST-AI/UniRig |
| skintokens | mit | yes | https://huggingface.co/VAST-AI/SkinTokens (code https://github.com/VAST-AI-Research/SkinTokens: MIT) |
| unimate | cc-by-nc-4.0 | **no** | https://huggingface.co/Linzhan/UniMate (the checkpoints CC BY-NC 4.0; the code https://github.com/Friedrich-M/UniMate MIT; flan-T5-base Apache-2.0) |
| minimax-h3, -q4km, -q6k, -q8, -ref2va | minimax-h3-community | yes, with a note | https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE (not licensed in the EU, UK, South Korea or the US; US$20 million yearly revenue needs written authorization; show "MiniMax H3"; outputs may not train another model; encoder Qwen3-VL-32B, Apache-2.0) |

The rules, checked when the cards load and by `cargo test` (`the_model_cards_are_whole`): the id is the file's
name and the kind its folder; a file key is defined once (defining it again the same way is allowed) and every key
a card names is defined; links are https and pinned, with a size and a SHA-256; a model's guide exists; every
`{role}` in `env` is one of its files or local parts; it has a `license` and `commercial: true` or `false` (and a
`license_note`, when present, that is a sentence). An entry registered without a card (`models add`) gets the
guide of a card of its architecture, else of its kind.
