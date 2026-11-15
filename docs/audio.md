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
- **OpenAI's `/v1/audio/speech`, streamed:** OpenAI's request as it is (`model` tts-1 / tts-1-hd / gpt-4o-mini-tts:
  the speech model serving; `voice` alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer, verse:
  qwen3-tts-custom's vivian, ryan, aiden, serena - a saved voice of the name wins, a model without built-in voices
  speaks its own; `response_format`, `speed` 0.25-4, `instructions`) answers `Transfer-Encoding: chunked`: the text in
  stretches of whole sentences (the first short), each sent as soon as it is spoken, through one ffmpeg for the whole
  answer (`glue/serve/src/audio/stream.rs`): mp3 (the default), opus (Ogg), aac (ADTS), flac, wav (a header of
  unknown length) or pcm - raw s16le, 24 kHz, mono, OpenAI's, so
  `curl ... -d '{"model":"tts-1","input":"...","voice":"alloy","response_format":"pcm"}' --output - | ffplay -f s16le -ar 24k -ac 1 -nodisp -`
  plays as it comes. An error before the first sound is a JSON error; after it the answer ends without its closing
  chunk; a client that hangs up stops the stretch being spoken at its next step and the rest are not spoken.
  `stream_format: "sse"` is refused. `response_format: url` (and a song's wav, unless `stream: true`) is the whole file, as before.
  Two more, from nextsycl: `sample_rate` (8000-96000, 24000 by default; `X-Sample-Rate` and a streamed wav's header
  say it; opus takes 8000, 12000, 16000, 24000 or 48000, mp3 up to 48000), and `input` as a list of texts
  (`["One.", "Two."]`), said in turn as one stream - the same voice and seed, `gap` seconds (0.3 by default, 0-5) of
  silence between - through the same single ffmpeg. A list is a stream, so `response_format: url` takes one text only
  (a 400 says so); a script of `speakers` takes no list. A part that fails ends the stream uncleanly, as above.
- **Text as it is said:** math and code symbols become words before any speech model sees them (`Q·K^T / √d_k`:
  "Q dot K to the T, divided by square root of d sub k"), markdown's marks go, dashes and brackets are pauses
  (`NS_SPEECH_NORMALIZE=0` or `normalize: false` for the text as sent) - one such line had a voice loop for 18 s.

## Speech: ZONOS3-Mini

`audio.speak` with `model: zonos3-mini`: Zyphra's ZONOS3-Mini (Apache-2.0) in Rust and CUDA (`audio/zonos3`,
`kernels/audio/zonos3/z3.cu`; the port's record and its API for other runtimes: [ports/zonos3.md](ports/zonos3.md)).
43 languages at 44.1 kHz: 27 preset voices (`voice`), a voice cloned from a recording (`ref_audio`, saved voices: the
MECT-B2 speaker encoder's embedding), steering (`options.controls`: pace, mood, emotions, the voice's identity), speaker
guidance (`cfg`). An 11-layer backbone of 1024 (gated DeltaNet with global attention a layer in four, Full AttnRes, a
top-2-of-64 latent MoE kept in its Q4_0 blocks) writes ten 4096-code books at 15 Hz; the ZAC codec's MR-iSTFT decoder
makes the sound. Zyphra's own GGUFs, every file a role of the card (`zonos3-mini`). The guide:
`model-cards/guides/zonos3.md`.

```sh
nextcuda models pull zonos3-mini
nextcuda audio gen "Good morning." --model zonos3-mini --voice rupert --out a.wav
nextcuda audio gen "Good morning." --model zonos3-mini --ref-audio me.wav --opt-controls delivery.tempo=1,emotions.happy=3 --out b.wav
nextcuda audio check <dump dir> --model zonos3-mini --stages prompt,lm,forced,decoder,speaker   # reference/zonos3/ makes the dumps
```

- **Checked** (2026-11-15) against Zyphra's ggml runtime (`zonos3-cli`, `zac-cli`, `spk-encoder-cli` built on the
  CPU, the same GGUFs; `reference/zonos3`): the prompt's ids identical (SuperBPE, the speaker / condition / silence
  rows); the backbone's logits over a 26-row speaker-embedding prompt cosine 0.99969 mean (0.9959 at the worst row;
  rel 2.1e-2; argmax the same in 96.9 %), over a 232-row reference-prefix prompt 0.99983 (94.2 %); teacher-forced over
  a 68-step take of the reference's own 0.99984 mean (0.9984 worst), argmax 92.9 % - the reference's CPU products
  quantize the activations to 8 bits, ours are float32 over the same blocks; the decoder's PCM cosine 1.000000 (rel
  3.3e-5); the speaker embedding cosine 1.000000 (rel 1.2e-6).
- **Heard** (Whisper large-v3-turbo on the box): preset voices in both speaker modes, a clone, steered takes (tempo
  +2: 6.9 s, -2: 13.9 s against 7.7 s; happy: Whisper writes the lines with exclamation marks), speaker CFG 1.5, the
  full decoder and German - word for word; 53 s of long-form English in three stretches with one slip. A clone of
  Rupert's recording measures 0.884 against it by the speaker encoder (0.13 against Savannah's).
- **Speed** (RTX 5070): 7.7 s of speech in 0.55 s, 53 s in 3.2 s (~240 frames a second, about 16x real time); a
  clone's speaker encoding of 13 s of recording 0.1 s. 2.1 GiB at the peak with the keep2 decoder (the backbone 0.8
  GiB, the decoder 0.7); the full decoder adds 4 GiB on the card when a request first names it (2.5 s to load), then
  decodes a 9 s take in 0.1 s as the default one does.
- **Loudness** (2026-11-15): the model drifts quieter through a long take with some voices, and a preset keeps its
  recording's level. Zyphra's own runtime does the same on the same inputs (a cloned voice, seeds 1-3, gated speech level
  per 5 s: -24 -> -32, -29 -> -45, -24 -> -36 dB; Savannah's codes -26 to -32 dB), and serves it 3 dB down. The engine now
  levels each take (`level`, on by default): the speech around each 50 ms frame (a 3 s window, a gate 35 dB under the
  loud frames of the 10 s about it) is brought to -19 dBFS active RMS by a gain smoothed over 0.5 s (-12 to +50 dB), then
  the soft limiter. (Until 2026-11-15 evening the gate was the whole take's loud frames and the gain at most +40 dB: a
  saved clone whose take faded 40 dB and more kept its tail 5 dB low - platform job f77a24fb, -24 dB active at 35 s.) Measured on the same takes, the gated level per 5 s spreads 0.9-2.0 dB (it spread 2.9-21 dB): a 70 s Rupert
  take -16.5 to -17.4, Savannah -16.9 to -18.0, the full decoder -16.7 to -17.6, the user's clone (the take that faded
  21 dB) -16.8 to -18.8. `level: off` gives the decoder's level less 3 dB, as Zyphra serves it.
- **Text** (2026-11-15): Zyphra's normalizer (`zyphra-textnorm`, pure Rust; option `normalize`) and the edit build's
  markup - `[pause]` / `[pause 0.25|0.5|1|2]`, `[start]`, `[ipa ...]`, and control spans `[steer ID=DOSE,...]words[/steer]`
  (steering over those words alone, ramped in and out) - token ids identical to Zyphra's `zonos3-cli --request`.
- **Raw-reference cloning** (2026-11-15; `options.speaker-mode: reference` with a recording): the recording's first
  60 s through Zyphra's reference encoder (`zac-encoder.gguf`: Snake residual convs, five MOSS stages, a semantic
  encoder and RVQ, with the 16-layer Wav2Vec2-BERT teacher `ssl-teacher.gguf`) to the codec's own codes, read as the
  prefix as a preset's are - in the reference's production BF16 recipe (operands and tensor boundaries rounded to
  bfloat16; float32 differs from it by 24 % at the teacher's output). Codes against Zyphra's `zac-encoder-cli` on
  Rupert's 13 s: codebook 0 the same in 97.5 %, 1: 95.5 %, 2-9: 76-89 % (Zyphra's own CPU encoder against the codes
  Zyphra ships for that recording: 98.0, 95.5, 75-89 %); the teacher in float32 matches the reference's float32
  teacher to 1e-5. The encoders load for the encode (5.6 GiB, 1.4-2.1 s) and are freed after it; the codes are kept
  for the last 64 recordings. Encode 1.0 s for 13 s; the card peaked at 8.5 GiB. A take cloned this way measures 0.839
  against the recording by the speaker encoder (the shipped preset codes 0.845, the embedding 0.884), Whisper word
  for word.
- **LACI, short texts, streaming** (2026-11-15): `options.laci on` checks the alignment heads' reading of the text at
  every step and rolls a skipped, stalled or truncated stretch back (Zyphra's detector, ACI mask, re-prefill); a text of
  a word or two (`short-text`, auto) is said with a pause and a filler sentence and cut where the filler starts - 18 of
  18 short texts said exactly against 10 of 18 without (Zyphra's server: 18 and 9); the probe matches Zyphra's at 97 %
  of the steps. `/v1/audio/speech` with `stream` gets the sound as it is decoded (identical to the whole take, rel
  1.2e-6; first sound after 0.08 s of a 0.78 s take).
- **Speech editing** (2026-11-15, `audio.edit`): a recording of up to 30 s, its transcript and the words it should say;
  the changed words are said anew in its voice by the edit LM (the edit backbone with its LoRA), the rest kept frame for
  frame. Word timings by Zyphra's FuseAlign in Rust and CUDA (within 1 ms of Zyphra's on every boundary measured); the
  edit LM matches Zyphra's runtime with the adapter at cosine 0.99987. The transcript is the caller's: the box's Whisper
  (`audio.transcribe`) makes one - Zyphra's own pipeline takes the transcript as given, its Qwen3-ASR is optional.
- Accent adapters: Zyphra ships none (`edit/accents.gguf` is the speech-edit adapter); accents are the steering bundle's
  `accents_*` directions.

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
- **Against upstream as it runs** (the bf16 checkpoint in bf16 on the 5070, `ref.py --device cuda:0 --dtype bf16
  --weights bf16`): the teacher-forced logits cosine 0.99998 to 0.999998 (rel 2.5e-3 to 9e-3), the argmax the same on
  every row; the twin's prefill 0.99996; the first velocity 0.99982 (int8 flow; Q8_0 0.99991), 8 steps 0.99979 (0.99987):
  the Q8_0 file's own distance (HOT-Step measures its NAR rel-L2 0.019 against bf16's 0.013).
- **Sung** (Whisper large-v3-turbo on the takes): English folk in each cot mode (full 77.6 s, melody 103 s, off 103 s)
  word for word; upstream's Mandarin demo (今晚不眠, seed 12300) 222.6 s, its choruses and bridge as written; the
  English song again over its own score with new lyrics (same seed and style): every new line sung, each line starting
  within 0.2 s of the first take's - the melody kept. The same request and seed make the same WAV, byte for byte.
- **Speed** (RTX 5070 the language model, RTX 3060 the VAE): 223.8 s of song (the Mandarin demo) in 72.5 s - the score
  11.7 s, 5594 codes at 190 a second, 32 flow steps in 26.8 s, the VAE 4.2 s (as ported, the flow on the Q8_0 matrices
  and nsd's attention: 88.1 s, its steps 41.6 s; int8 matrices 33.5 s; the silo's attention, scores in float32, 30.7 s); a 30 s song in 12 s (224 codes a second). 21 s to load from the HDD. Peak 7.2 GiB on the
  5070, 1.4 GiB on the 3060. The flow's (NAR path's) matrices are int8 ConvRot rows by default (`NS_YUE2_INT8=0`: Q8_0 as
  stored): the first velocity cosine 0.99991 against the reference (Q8_0: 0.999996), 8 steps from the noise 0.99985
  (0.99998) - bf16's own distance from float32 by HOT-Step's measure; the silo's attention with its scores in half (the
  default; `NS_YUE2_ATTN=float`) 0.99989 and 0.99978; the takes transcribe as before.
- **Covers** (2026-10-27): a recording given is transcribed by SheetSage2 on the 3060 (`audio/yue2/src/sheet`, checked
  against upstream: [ports/yue2.md](ports/yue2.md)) and its score sung in the new style. The test recording is the USAF
  Band of Mid-America's Jingle Bells (2020, public domain, an instrumental: `/mnt/hdd/nextcuda-models/yue2/test` on the
  box). Its lead line is in SheetSage2's `Ins` voice; the engine moves it to the `Vocal` voice YuE2 sings, because with
  the score as transcribed all three styles came out instrumental and Whisper heard no words. With lyrics tagged to the
  score's sections (`[Intro]`, a long `[Verse]`, `[Outro]`), all three styles sing the traditional words over the
  transcribed tune (Whisper hears the verse and the choruses): jazz 135.7 s in 39.4 s, metal 117.1 s in 34.5 s,
  synthwave 138.4 s in 40.4 s. The whole song with new words ("Rolling on, rolling on ...") is sung line for line,
  163.2 s in 47.1 s. Each run spends 7.2 s on the transcription; peaks are 7.0 GiB on the 5070 and 3.1 GiB on the 3060.
- **Instrumental covers** (2026-11-11, `instrumental: true`): the lead stays in the `Ins` voice (passage by passage: a
  passage SheetSage2 heard as sung moves there; with a line in both the Vocal notes become rests), the Vocal voice is
  all rests and the lyrics are the score's sections with no words. The test recording is Paganini's Caprice No. 24
  (viola and piano, Elias Goldstein, 2009, public domain by its author: `/mnt/hdd/nextcuda-models/yue2/test`, 310.7 s).
  Whisper large-v3-turbo writes no lyrics on the takes - only its stock no-speech phrases ("Thank you", "Субтитры
  сделал DimaTorzok") over whole 30 s windows, as it does on the recording itself; timed segments (words it heard),
  techno seed 11 17 s ("so"), synthwave seed 11 0 s, techno seed 12 1 s ("Bye."), the same with cfg 1.3 0 s. Before
  the passage-by-passage move (the tune rested where SheetSage2 had it in Vocal, 45 of its 1183 notes lost) techno seed 12
  mumbled for 93 s - Whisper timed 17 segments of nonsense words over 60-121 s and 236-297 s. A 300 s take in 142 s;
  synthwave seed 11 ended itself at 80.8 s.
  Through the platform (48c6c18, the recording fetched from its Commons link as a FLAC; 3 takes each, unguided):
  Whisper timed words in 2 of 5 takes over their last third - techno take 2 38 s ("The moment I go to all my
  appalling..."), synthwave take 2 95 s of nonsense words; the others 1-6 s of single words over sustained notes. The
  same synthwave take made again with cfg 1.3 (the same seed and FLAC; at cfg 1 it is the platform's take byte for
  byte): one 20 s segment, one word. So an instrumental cover is guided by 1.3 unless the request names a cfg.
- **Text-only instrumentals are refused** (no recording: `instrumental` needs one). Tried as plain songs: `[Instrumental]`
  as the lyrics (synthwave seed 11, techno seed 12) and no lyrics (synthwave seed 11), 120 s each. Whisper timed no
  words in any, but that cannot tell humming from a synth line, and the plan YuE2 draws for them names verses and
  choruses: in one of the three it wrote a sung phrase (16 notes) into the Vocal voice, the voice it sings. Not shipped
  until something can hear a wordless voice.
