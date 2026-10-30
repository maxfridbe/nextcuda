# Supported models

43 model cards in `model-cards/` (one YAML a model, every file a pinned link with its size and SHA-256);
`nextcuda models pull <id> [--from DIR]` downloads what is missing or hard-links a file already on the machine,
checks it and registers the model. **Served** - running on this box today; **download** - its card is here and it
runs once pulled; **too big** - more than these 12 GB cards hold: the page shows it "won't run here" (why, in GiB)
with its Download off (`system.models.pull` still fetches it, for bigger cards).

| kind | id | model | on disk | here |
|---|---|---|---|---|
| chat | `qwen3.8-27b-gsq-iq3s` | Qwen3.8-27B, ISTA-DASLab's GSQ-RCO IQ3_S with its MTP head and the BF16 vision projector: text and pictures, thinking switch, 80K context (192K with q4_0) | 12.2 GiB | served, both cards whole |
| chat | `qwen3.8-27b-gsq-iq3s-long` | the same files, fast / long: the cache q4_0 rotated, 160K of context (a speech model still fits beside it) | (the above's) | served |
| chat | `nemotron-3.5-super-iq1m` | NVIDIA Nemotron 3.5 Super 120B-A12B, unsloth's UD-IQ1_M (text) | 43.0 GiB | served (one card, experts streamed; off by default) |
| image | `qwen-image-2.1-q8` | Qwen-Image 2.1 Q8_0 with Qwen3-VL 8B: text to image, RGBA, edits from up to 8 pictures | 13.6 GiB | served |
| image | `qwen-image-2.1-rrq4` | Qwen-Image 2.1, [realrebelai's Q4](https://huggingface.co/realrebelai/Qwen-Image-2.1_GGUFs) | 5.5 GiB | served |
| image | `qwen-image-2512-q4km` | Qwen-Image 2512 Q4_K_M with Qwen2.5-VL 7B and the Lightning 4-step LoRA | 17.7 GiB | served |
| image | `qwen-image-2.1-q6k` | Qwen-Image 2.1 with a Q6_K transformer | 12.0 GiB | download |
| image | `ming-image-0.1-design` | Ming-Image-0.1-Design: UI, posters, infographics - text spelled right; over both cards | 16.3 GiB | served |
| vectors | `starvector-1b-im2svg` | StarVector-1B im2svg: a picture written as SVG code - logos, icons, simple illustrations (`image.vectorize`) | 4.8 GiB | served |
| upscale | `real-esrgan` | Real-ESRGAN: x4plus, x4plus-anime, animevideov3 2x / 3x / 4x (image.upscale, video.upscale) | 45 MB | served |
| upscale | `waifu2x` | waifu2x: cunet, upconv_7 anime / photo, denoise -1 to 3, 1x to 32x | 34 MB | served |
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

## What runs where, by engine

| kind | engines | on the two cards |
|---|---|---|
| chat | `qwen35` (Qwen3.8: DeltaNet + attention, MTP drafts, vision), `nemotron_h_moe` | Qwen3.8 over both, pipelined; Nemotron on the 5070 |
| image, OCR | stable-diffusion.cpp's ggml-cuda (Qwen-Image), our Rust + CUDA engines (Qwen-Image 2.1 `rust`, Ming-Image, Unlimited-OCR, PaddleOCR-VL, olmOCR-2) | one card each; olmOCR-2 the decoder on one, the vision tower on the other; Ming-Image the DiT on the 5070, its text encoder on the 3060 |
| speech | `breeze-tts-2`, `qwen3-tts`, `voxcpm2`, `vibevoice` (several speakers) | one card each; VoxCPM2 int8 fits beside the chat model |
| songs | `minimax-music3` (int8), `acestep-1.5` | MiniMax: the language model on the 5070, the flow stage and decoder on the 3060 |
| video | `minimax-h3` (the native daemon, `--span`) and the video studio (scenes, plans, a film's clips joined) | the denoiser over both cards, nothing streamed: a clip in ~62 s from the SSD |
| 3D | TRELLIS.2, Hunyuan3D-2mini and 2.1 (its paint alone too), UniRig, SkinTokens, RigNet, AnimateAnyMesh, UniMate | one card each |
| listening | `whisper` (large-v3-turbo) | one card |
| upscale | `realesrgan`, `waifu2x` (their ncnn graphs) | one card |
| vectors | `starvector` (StarVector-1B im2svg) | one card |
