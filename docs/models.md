# Supported models

52 model cards in `model-cards/` (one YAML a model, every file a pinned link with its size and SHA-256);
`nextcuda models pull <id> [--from DIR]` downloads what is missing or hard-links a file already on the machine,
checks it and registers the model. **Served** - running on this box today; **download** - its card is here and it
runs once pulled; **too big** - more than these 12 GB cards hold: the page shows it "won't run here" (why, in GiB)
with its Download off (`system.models.pull` still fetches it, for bigger cards).

| kind | id | model | on disk | here |
|---|---|---|---|---|
| chat | `qwen3.8-27b-gsq-iq3s` | Qwen3.8-27B, ISTA-DASLab's GSQ-RCO IQ3_S with its MTP head and the BF16 vision projector: text and pictures, thinking switch, 80K context (192K with q4_0) | 12.2 GiB | served, both cards whole |
| chat | `qwen3.8-27b-gsq-iq3s-long` | the same files, fast / long: the cache q4_0 rotated, 160K of context (a speech model still fits beside it) | (the above's) | served |
| chat | `qwen3.8-27b-gsq-iq3s-dspine` | the same files with Tencent's DSpine-b16 draft (Apache-2.0, 4.1 GB) in MTP's place, 7 drafts a step, 48K: for code (75 tok/s against the MTP card's 57; slower on prose and long documents) ([ports/dspine.md](ports/dspine.md)) | (the above's) + 4.1 GB | served, opt-in |
| chat | `nemotron-3.5-super-iq1m` | NVIDIA Nemotron 3.5 Super 120B-A12B, unsloth's UD-IQ1_M (text) | 43.0 GiB | served (one card, experts streamed; off by default) |
| image | `qwen-image-2.1-q8` | Qwen-Image 2.1 Q8_0 with Qwen3-VL 8B: text to image, RGBA, edits from up to 8 pictures | 13.6 GiB | served |
| image | `qwen-image-2.1-rrq4` | Qwen-Image 2.1, [realrebelai's Q4](https://huggingface.co/realrebelai/Qwen-Image-2.1_GGUFs) | 5.5 GiB | served |
| image | `qwen-image-2.1-turbo-q8` | Qwen-Image 2.1 with Viggle's 6-step turbo merged (cfg 1, its own sigmas; the Rust engine, the transformer on the card) | 7.2 GiB more | served |
| image | `qwen-image-2512-q4km` | Qwen-Image 2512 Q4_K_M with Qwen2.5-VL 7B and the Lightning 4-step LoRA | 17.7 GiB | served |
| image | `qwen-image-2.1-q6k` | Qwen-Image 2.1 with a Q6_K transformer | 12.0 GiB | download |
| image | `ming-image-0.1-design` | Ming-Image-0.1-Design: UI, posters, infographics - text spelled right; over both cards | 16.3 GiB | served |
| image | `z-image-turbo` | Z-Image-Turbo (Tongyi-MAI): text to image in 8 steps, photoreal, legible text in the picture; the DiT on the 5070, Qwen3-4B on the 3060 | 10.9 GiB | served |
| vectors | `starvector-1b-im2svg` | StarVector-1B im2svg: a picture written as SVG code - logos, icons, simple illustrations (`image.vectorize`) | 4.8 GiB | served |
| upscale | `real-esrgan` | Real-ESRGAN: x4plus, x4plus-anime, animevideov3 2x / 3x / 4x (image.upscale, video.upscale) | 45 MB | served |
| upscale | `waifu2x` | waifu2x: cunet, upconv_7 anime / photo, denoise -1 to 3, 1x to 32x | 34 MB | served |
| upscale | `flashvsr` | FlashVSR v1.1: one-step diffusion video super-resolution, 2x-4x, a clip as one stream (video.upscale; image.upscale as a still clip) | 6.0 GB | served |
| OCR | `unlimited-ocr-q8` | Unlimited-OCR Q8_0: pages to Markdown | 3.7 GiB | served |
| OCR | `unlimited-ocr-q4` | Unlimited-OCR Q4_K_M | 2.6 GiB | served |
| OCR | `paddleocr-vl-1.6` | PaddleOCR-VL 1.6: pages, tables, formulas, charts, seals | 1.7 GiB | served |
| OCR | `olmocr-2-7b-q8` | olmOCR-2 7B Q8_0: pages to Markdown | 8.8 GiB | served |
| speech | `breeze-tts-2` | Breeze TTS 2 Q8_0 with Qwen3-TTS's 12 Hz codec: English and Chinese, voices designed, cloned or directed, vocal events (non-commercial) | 4.0 GiB | served |
| speech | `qwen3-tts-custom` | Qwen3-TTS 12 Hz 1.7B CustomVoice: nine built-in voices styled by an instruction, ten languages | 4.2 GiB | served |
| speech | `voxcpm2` | VoxCPM2 2B: 30 languages at 48 kHz, voices designed or cloned | 4.6 GiB | served |
| speech | `voxcpm2-int8` | VoxCPM2, its language models in int8: 3.3 GiB on a card, beside the chat model | (voxcpm2's) | served (Open WebUI's voice) |
| speech | `breeze-tts-2-q4` | Breeze TTS 2 Q4_K, a smaller download | 2.7 GiB | download |
| speech | `qwen3-tts-base` | Qwen3-TTS Base: a voice cloned from a recording | 4.2 GiB | download |
| speech | `qwen3-tts-design` | Qwen3-TTS VoiceDesign: a voice made up from a description | 4.2 GiB | download |
| speech | `vibevoice-1.5b` | VibeVoice 1.5B: a script of up to four speakers, each cloned from a recording | 5.1 GiB | served |
| speech | `vibevoice-1.5b-int8` | VibeVoice, its language model in int8 | (the above's) | served |
| speech | `zonos3-mini` | ZONOS3-Mini (Zyphra, Apache-2.0): 43 languages at 44.1 kHz, 27 preset voices, a voice cloned from a recording, steering of pace and emotion; ~15x real time | 2.5 GiB | served |
| listening | `whisper-large-v3-turbo` | Whisper large-v3-turbo: speech to text in 99 languages, words timed, the language detected (MIT) | 1.5 GiB | served |
| songs | `minimax-music3-int8` | MiniMax Music 3, its language model in int8: songs from lyrics and a description, up to 6 minutes, 44.1 kHz stereo | (music3's) | served, both cards |
| songs | `minimax-music3` | MiniMax Music 3 in half | 26.6 GiB | too big (23 GB a card) |
| songs | `acestep-1.5` | ACE-Step 1.5 turbo: songs in 50 languages from a description and lyrics, covers, repaints, lyric edits; 48 kHz stereo | 9.4 GiB | served |
| songs | `yue2-3b-q8` | YuE2-3B Q8_0 (non-commercial, CC BY-NC 4.0): songs with vocals from a style and lyrics, English and Mandarin, a melody / chords score first (sung again with new words over the same score), up to 6 minutes, 48 kHz stereo | 4.2 GiB | served, both cards |
| songs | `acestep-1.5-base` | ACE-Step 1.5 base: a vocal track made in a recording's context (lego) | 9.4 GiB | served |
| video | `minimax-h3-q4km` | MiniMax H3 Q4_K_M: text, pictures and a voice to video with sound; the denoiser in 10.6 GiB | 10.6 GiB | served, both cards |
| video | `minimax-h3-q6k` | MiniMax H3 Q6_K: the denoiser in 15.4 GiB | 15.4 GiB | download |
| video | `minimax-h3-q8` | MiniMax H3 Q8_0, a reference form | 20.0 GiB | too big |
| video | `minimax-h3-ref2va` | MiniMax H3 Ref2VA (int8 ConvRot): reference pictures, clips and sounds to video | 19.5 GiB | too big |
| video | `minimax-h3` | MiniMax H3, int8 ConvRot denoiser: the fastest | 47.9 GiB | too big |
| video | `ltx-2.5` | LTX-2.5 (Lightricks), the dev DiT Q4_K_M: text and a first frame to video with its own 48 kHz sound, 30 guided steps; engines `ltx2` (variant dev or distilled) and `ltx2_upscale` (a clip twice the size). LTX-2.x Community License: free below US$10M annual revenue, a paid license above ([ports/ltx2.md](ports/ltx2.md)) | 58.7 GiB of files (both DiTs, the bf16 text encoder, connectors, VAEs, vocoder, upsampler) + a 10.7 GiB text cache made beside them | served, both cards |
| video | `ltx-2.5-fast` | the same with the distilled DiT (8 steps, no guidance: a preview, thinner sound); its files are `ltx-2.5`'s (the distilled DiT, 13.2 GiB, is the one it runs) | (the above's) | served, both cards |
| 3D | `trellis2-512` | TRELLIS.2 4B Q8_0 at 512: picture to textured mesh, the light path (MIT) | 1.3 GiB | served |
| 3D | `hunyuan3d-2mini-turbo` | Hunyuan3D-2mini-turbo: a mesh from a picture in seconds (shape) | 3.9 GiB | served |
| 3D | `unirig` | UniRig: a skeleton and skin weights for any mesh (MIT) | 5.4 GiB | served |
| 3D | `skintokens` | SkinTokens / TokenRig: a skeleton and skin weights as one token sequence (MIT) | 1.5 GiB | served |
| 3D | `rignet` | RigNet: a skeleton and skin weights for a character mesh (GPLv3) | 0.2 GiB | served |
| 3D | `animateanymesh` | AnimateAnyMesh: text-driven mesh animation, 16 frames as morph targets (Apache-2.0) | 0.9 GiB | served |
| 3D | `unimate` | UniMate: a rigged mesh's skeleton animated from a sentence, 60 frames of glTF skeletal animation (weights CC BY-NC 4.0: non-commercial) | 2.1 GiB | served |
| 3D | `trellis2` | TRELLIS.2 4B Q8_0, the 1024 cascade | 8.8 GiB | download |
| 3D | `pixal3d` | Pixal3D Q8_0: pixel-aligned TRELLIS.2, the 1024 cascade (MIT) | 8.9 GiB | download |
| 3D | `hunyuan3d-2.1` | Hunyuan3D 2.1: a detailed mesh with PBR textures | 15.5 GiB | too big |
| 3D | `hunyuan3d-2.1-paint` | Hunyuan3D 2.1's paint model alone: PBR textures for a mesh you give it, from a picture (`model3d.generate` with `mesh`) | 8.2 GiB | served (on a 12 GB card it peaks at 11.4 of 12.2 GiB: nothing beside it) |

Each model's guide (how to ask it, what it takes, what it answers) is in `model-cards/guides/`.

## Image controls (schema 1)

Every picture-making image card (Qwen-Image 2.1 / 2512, Z-Image, P-H3, Ming, MOSS) carries `controls:` - the request
rules a client may build on - and `system.models` (and `system.state`) puts them on the model's row. The creative
platform's image control offers only these values and its request builder refuses anything outside them, so a new
model needs no platform change. The values were read off each engine, not guessed; where an engine accepts more than
it makes well (a turbo's guidance, an unbounded step count, a value it clamps or ignores) the rules offer only what it
makes well. `glue/models/src/catalog.rs` `every_picture_making_card_states_its_request_rules` checks every card.

```yaml
controls:
  version: 1                     # integer; a reader refuses (treats as unknown) a version above its own; fields added
                                 # within a version are optional
  steps: {min: 6, max: 6, default: 6}          # min = max: fixed (the box's own count: a client leaves steps out)
  cfg: {min: 1, max: 30, default: 4}           # null: no guidance (cfg 1 only)
  negative: true                 # a negative prompt is read
  size:
    min_side: 256                # each side
    max_side: 4096
    multiple: 16                 # each side a multiple of this (refused otherwise - a client does not rely on rounding)
    max_pixels: 4194304          # width x height at most (optional)
    default: 1024x1024
    sizes: []                    # optional: only these WxH
    presets: {ratios: ['16:9'], pixels: [1048576]}   # optional: what the model was trained around, for a picker
  seed: true                     # the seed is honoured (picture i: seed + i)
  n: 8                           # pictures one request makes at most
  loras: true                    # per-request LoRAs
  edit:                          # null: text to image only (no pictures, no img2img)
    images: 8                    # pictures one edit takes (the first is the one changed)
    mask: false                  # a mask picture (no engine receives one today)
    strength: true               # img2img strength below 1
    reference: true              # the pictures condition the model (the instruction edit)
    strength_with_reference: false   # strength below 1 also applies with reference true (else it needs reference false)
  transparent: true              # background "transparent" gives an RGBA picture
  sampler: {values: [euler, lcm], default: euler}    # optional: the samplers a client may name
  shift: {min: 0.1, max: 20, default: 3}             # optional: the schedule's shift a client may set; absent: its own
  options:                       # optional: the engine's own per-request options (`options: {name: value}`)
    - {name: eta, type: number, min: 0, max: 1, default: 0, help: '...'}   # type number | bool | text
  note: '...'                    # optional: one sentence for a picker (the int8 fast mode's seed caveat)
```

| model | steps | cfg | negative | sides (multiple) | max pixels | edit | alpha | LoRAs | other |
|---|---|---|---|---|---|---|---|---|---|
| `qwen-image-2.1-q8`, `-q6k`, `-rrq4` | 20-100, 40 | 1-30, 4 | yes | 256-4096 (16) | 4,194,304 | 8 pictures, strength with reference false | yes | yes | |
| `qwen-image-2.1-q8-fast` | 20-100, 40 | 1-30, 4 | yes | 256-4096 (16) | 4,194,304 | as above | yes | yes | int8: a seed's picture differs slightly |
| `qwen-image-2.1-turbo-q8` | 6 fixed | none | no | 256-4096 (16) | 4,194,304 | as above | yes | yes | |
| `qwen-image-2.1-turbo-q8-fast` | 6 fixed | none | no | 256-4096 (16) | 4,194,304 | as above | yes | yes | int8 |
| `qwen-image-2512-q4km` | 1-100, 4 (under 20 Lightning) | 1-30, 1 | yes | 256-4096 (16) | 4,194,304 | none | no | yes | |
| `z-image-turbo` | 1-50, 8 | none | no | 256-2048 (16) | 2,359,296 | none | no | no | euler; shift 0.1-20, 3; presets |
| `p-h3-image` | 8 fixed | none | no | 256-4096 (32), default 2912x2912 | 8,912,896 | 8 pictures, strength with or without reference (reference false: 1) | no | no | euler / lcm; eta, s_noise, deblock, deblock_strength |
| `p-h3-image-pdmd2` | 1-8, 2 | none | no | 256-4096 (32) | 8,912,896 | as above | no | no | lcm default, eta 1 |
| `ming-image-0.1-design` | 1-50, 12 | 1-10, 1 | no | 512-4096 (32) | 4,194,304 | none | yes | no | made at its nearest trained size, resized |
| `moss-image-sft-q8` | 20-100, 50 | 1-30, 3 | yes | 64-4096 (16) | 4,220,160 | none | no | yes | shift 0.1-20, 5; presets |
| `moss-image-sft-turbo4-q8` | 1-8, 4 | none | no | 64-4096 (16) | 4,220,160 | none | no | yes | presets |
| `moss-image-sft-turbo8-q8`, `-dmd2-q8` | 1-16, 8 | none | no | 64-4096 (16) | 4,220,160 | none | no | yes | presets |

Every row: seed honoured, up to 8 pictures a request. What the engines accept beyond these (found reading them,
2026-11-15): a turbo 2.1 runs guided with cfg above 1 and any step count on a schedule other than `shift`; 2.1 ignores
`shift` on its `shift` schedule and a reference edit's strength; 2512 ignores sampler, schedule and shift; Z-Image, P-H3
and Ming drop per-request LoRAs at their reload; P-H3 clamps eta, s_noise and deblock_strength; Ming reads no negative
prompt and makes any size at its nearest trained one; MOSS's turbo cards honour cfg and shift though the engine reports
neither; the full 2.1 and MOSS models take any step count, where under 20 a picture stops half made (image job
c989bf51: a turbo's 6 steps sent to the full 2.1). The rules leave all of that out.

## What runs where, by engine

| kind | engines | on the two cards |
|---|---|---|
| chat | `qwen35` (Qwen3.8: DeltaNet + attention, MTP drafts, vision), `nemotron_h_moe` | Qwen3.8 over both, pipelined; Nemotron on the 5070 |
| image, OCR | our Rust + CUDA engines (Qwen-Image 2.1 and 2512, Ming-Image, Z-Image-Turbo, Unlimited-OCR, PaddleOCR-VL, olmOCR-2) | one card each; olmOCR-2 the decoder on one, the vision tower on the other; Ming-Image the DiT on the 5070, its text encoder on the 3060 |
| speech | `breeze-tts-2`, `qwen3-tts`, `voxcpm2`, `vibevoice` (several speakers), `zonos3` | one card each; VoxCPM2 int8 fits beside the chat model |
| songs | `minimax-music3` (int8), `acestep-1.5` | MiniMax: the language model on the 5070, the flow stage and decoder on the 3060 |
| video | `minimax-h3` (the native daemon, `--span`), `ltx2` (LTX-2.5: Gemma-4 text stage, a 22B audio-video DiT, VAEs, vocoder, latent upsampler) and the video studio (scenes, plans, a film's clips joined) | the denoiser over both cards, nothing streamed: an H3 clip in ~62 s from the SSD; LTX-2.5's DiT resident, 32 blocks on the 5070 and 16 on the 3060, its text stage loaded for a clip and released |
| 3D | TRELLIS.2, Hunyuan3D-2mini and 2.1 (its paint alone too), UniRig, SkinTokens, RigNet, AnimateAnyMesh, UniMate | one card each |
| listening | `whisper` (large-v3-turbo) | one card |
| upscale | `realesrgan`, `waifu2x` (their ncnn graphs) | one card |
| upscale | `flashvsr` (Rust + CUDA) | both cards, pipelined (one card up to about 1280 x 768 out) |
| vectors | `starvector` (StarVector-1B im2svg) | one card |
