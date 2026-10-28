# Speech, conversation, songs and listening

The audio kind: every engine runs on the shared audio kernels (`kernels/audio/*`), in Rust and CUDA, checked against the
model's own package in float32. Settings: [environment.md](environment.md); the models: [models.md](models.md).

## Speech: Breeze TTS 2, Qwen3-TTS, VoxCPM2

nextsycl's engine (`audio/breezetts2`) and the 12 Hz codec (`crates/codec12hz`), unchanged, over the audio kernel
library: the music engine's decode kernels (`kernels/audio/minimaxmusic3/mm3.cu`: the small-batch product, the cached
attention; the codec's convolutions as an unfolding and one cuBLAS GEMM a tile, float32), the codec's
(`kernels/audio/qwen3tts/q3t.cu`) and Breeze's own (`kernels/audio/breezetts2/bz2.cu`) - CUDA ports with nextsycl's
names and arithmetic. No Python.

```sh
nextcuda models pull breeze-tts-2
nextcuda audio gen "(sigh) Welcome aboard." --model breeze-tts-2 --instructions "a calm adult male narrator" --out a.wav
nextcuda audio check <reference dump dir> --model breeze-tts-2        # reference/breezetts2/ref.py makes the dumps
```

- **Checked** (2026-10-17) against upstream's own code in float32 on the CPU (`reference/breezetts2/ref.py`, a designed
  voice, guidance 4): token ids identical; with the Q8_0 file the text encoder 6.6e-3, the prompt rows 9.5e-3, the
  backbone's logits 4.7e-3 and the depth decoder's 9.1e-3 (teacher forced: 1 of 60 first codes picks otherwise), the
  codec 1.4e-6 - nextsycl's numbers for the same file. Takes made here transcribe word for word (Whisper small).
- **Qwen3-TTS and VoxCPM2**: nextsycl's engines too (`audio/qwen3tts`, `audio/voxcpm2`; VoxCPM2's kernels
  `kernels/audio/voxcpm2/vcp.cu`). Checked against their own packages in float32 (`reference/qwen3tts/ref.py`,
  `reference/voxcpm2/ref.py`): Qwen3-TTS's prompt rows 1.9e-5, first logits 5.4e-4, frames forced: no first code
  differs (the other codes 4 of 585, near ties), the codec 1.2e-6; VoxCPM2's prompt tokens identical, both language
  models 3e-4, each patch from the reference's history and noise 5e-3 (cos 0.99999), the VAE 1.1e-6 - run free it
  takes another valid path after a few patches, as nextsycl documents. On the 5070, 5.6 s of Qwen3-TTS speech in 1.0 s,
  4.8 s of VoxCPM2's in 1.0 s; both transcribe word for word.
- **Speed** (RTX 5070): 5.5 GiB loaded in 4 s; 26 frames a second with guidance (2.1x real time; the B70: 22); through
  the front door, 12.7 s of designed speech in 6.5 s.
- **Saved voices** (`audio.voices.add`: a recording, its transcript, a description): spoken by name by any model that
  clones (`voice: "skippy"` on `/v1/audio/speech`). VoxCPM2 takes a saved voice's timbre alone - continuing its
  recording (`NS_VCP_CONTINUE=1`) led the recording's last words into each sentence (Whisper heard them ahead of 4 of
  5 lines; the timbre alone said all the plain ones word for word).
- **Text as it is said:** math and code symbols become words before any speech model sees them (`Q·K^T / √d_k`:
  "Q dot K to the T, divided by square root of d sub k"), markdown's marks go, dashes and brackets are pauses
  (`NS_SPEECH_NORMALIZE=0` or `normalize: false` for the text as sent) - one such line had a voice loop for 18 s.

## Conversation: VibeVoice 1.5B

`audio.speak` with `model: vibevoice-1.5b` (or `speakers` and no model): a script of `Speaker N: words` lines in up to
four voices, each cloned from a recording (`speakers`: saved voices, recordings or uploads; one `ref_audio` / `voice`
for one speaker). Microsoft's model on the shared kernels (`audio/vibevoice`; no kernel of its own): a Qwen2.5-1.5B,
a diffusion head (DPM-Solver++, 10 steps, guidance 1.3 against a pass that sees only the turn), the 7.5 Hz acoustic and
semantic tokenizers streamed a frame at a time - the platform's GGUF.

- **Checked** (2026-10-19) against yvoice's own code in float32 on the same file (`reference/vibevoice/ref.py`): the
  prompt's tokens identical, the prefill 6.3e-4, the diffusion head 1.7e-4, the decoder 1.3e-4, the whole loop teacher
  forced 1.3e-4 (cos 1.000000); takes transcribe word for word (the Whisper port).
- **Speed** (RTX 5070): 12 s of a three-turn dialogue in 2.2 s; 5.4 GiB (`vibevoice-1.5b-int8`: 4.3 GiB).

## Listening: Whisper large-v3-turbo

`audio.transcribe` (and `/v1/audio/transcriptions`, `nextcuda audio transcribe FILE`): openai's large-v3-turbo on
the shared kernels (`audio/whisper`; no kernel of its own) - the log-mel, the 32-layer encoder and 4-layer decoder,
openai's greedy decoding rules and long-form 30 s loop, words timed by the alignment heads, the language detected.

- **Checked** (2026-10-19) against openai-whisper itself on the same weights (`reference/whisper/ref.py`): on an
  18.6 s clip, a 76 s voice sample and a 48 s song, the segments are openai's word for word and every word is in step
  with its times off by 0.000 s (the stages within 2e-2, f16).
- **Speed** (RTX 5070): a 148 s song heard in 1.8 s, 2.5 GB at the peak; the platform's songs listen here now (its
  local word pass took 28.5 s for the same take).

## Songs: ACE-Step 1.5

`audio.generate` with `model: acestep-1.5` (the turbo) or `acestep-1.5-base`: songs from a description and lyrics, in 50
languages at 48 kHz stereo, and the tasks `text2music`, `cover`, `repaint` / extend, a lyric `edit` (flow-edit) and
`lego` (a vocal track over an instrumental, on the base checkpoint). The 5 Hz planner LM (the platform's `use_lm`) writes
the metas, the caption and the audio codes first; `NS_ACE_PLANNER=0` at load leaves it out. Settings:
[environment.md](environment.md) (`NS_ACE_*`); the guide: `model-cards/guides/acestep.md`.

- **Through the platform** (2026-10-21): a 30 s song with lyrics rendered on the box in 27.2 s and was listened back by
  its own Whisper.

## Songs: YuE2-3B

`audio.generate` with `model: yue2-3b-q8`: M-A-P's YuE2 (non-commercial, CC BY-NC 4.0) in Rust and CUDA
(`audio/yue2`, `kernels/audio/yue2`; the plan and the port's record: [ports/yue2.md](ports/yue2.md)). One AR-NAR
Mixture-of-Transformers writes a score (ABC: melody and chords for `cot: full`, the default; a melody for `melody`; none
for `off`) and the semantic codes, then the acoustic latents by flow matching (32 midpoint steps); YuE2-Vae decodes 48 kHz
stereo. scragnog's GGUFs: the language model Q8_0 (every matrix in its blocks on the card), the VAE float32. A song's
score comes back with it (`score`, and the WAV's comment): sung again with the same seed and style and new lyrics
(`abc`), the melody stays. The guide: `model-cards/guides/yue2.md`.

- **Checked** (2026-10-27) against upstream's yue2_infer 0.1.5 on the CPU in float32 with the same Q8_0 file
  (`reference/yue2/ref.py`): the prompts identical in every cot mode; the whole sequence teacher-forced (prompt, a
  160-token score, 225 codes), the logits cosine 1.000000 (rel 3e-4 to 2e-3) and the argmax the same on every row checked;
  the twin's prefill 1.000000; torch.randn's noise to 5e-7; the first velocity 0.999996, every midpoint step from the
  reference's state 1.000000, 8 steps from the noise 0.99998; the VAE 1.000000 (rel 1.6e-6).
- **Sung** (Whisper large-v3-turbo on the take): a verse and two choruses, word for word.
