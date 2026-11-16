# Haru-4B on nextcuda

Haru-4B (higurashi.ai, Apache-2.0) is a full-duplex spoken dialogue model in lockstep: every 133 ms (7.5 Hz) one frame
of the user's audio in, one frame of its own audio and text out, silence included. nextcuda runs it as the `audio/haru`
engine (arch `haru`, card `haru-4b`) behind `/v1/realtime` with nextsycl's lockstep extension
([../api/realtime.md](../api/realtime.md)): the same API as hardac1's nextsycl, so the creative platform's Talk page
uses either box. This is a port of nextsycl's `audio/haru` (the user's own project: its docs/ports/haru-4b.md has the
model's anatomy and the release's sources); what follows is what differs here and what was measured.

## What was ported, from where

| part | here | origin |
|---|---|---|
| session, lockstep conversation (`duplex.rs`), character card (`card.rs`, `cards.json`), sampling (DRY, top-k, the detokenizer), resampling, stage checks | Rust as nextsycl has it | nextsycl `audio/haru` |
| language model (Qwen3.5-4B text layers) | nextcuda's Qwen3.5 core (`crates/qwen35`) on a Q8_0 GGUF made once from the checkpoint (`convert.rs`); input rows = token row + audio projections (`Qwen35::forward_rows_add`, new) | llama.cpp's qwen35 tensor layout (MIT; the transformation reproduced, not its code) |
| codec (causal VibeVoice acoustic tokenizer, a frame at a time) | the VibeVoice engine's `codec.rs` on Haru's tensor names | nextcuda `audio/vibevoice` |
| MeanFlow head (one step) | on the VibeVoice ops (adaptive-norm SwiGLU, as VibeVoice's diffusion head); scalar embeddings on the host | nextsycl `head.rs` + nextcuda `audio/vibevoice/src/head.rs` |
| kernels | one new: `ns_q35_gemv_q8f` (q35_q8f.cu: Q8_0 matrices x float32 activations, decode rows); the rest shared: qwen35 / nemotronh (llm library, now linked into the audio library too), mm3 / q3t (audio), nsd | nextcuda |
| server: `/v1/realtime`, WebSocket, the splice over http and https, `system.models` row | `glue/serve/src/realtime.rs`, `http.rs`, `tls.rs`, `home/life.rs` | nextsycl `glue/serve` |

Skipped: nothing of the model. The offline `audio.generate` path (a recording answered) is nextsycl's and is ported but
was not exercised here. No silo, no tuning: it runs 3.3-6.9x real time.

## Weights and quantisation

The card's files are the release's (`model.safetensors` 10.5 GB bf16, `config.json`, `tokenizer.json`; sha256 checked).
At the first load the engine writes `nextcuda-haru-lm-q8_0.gguf` beside them (4.5 GB, 12 s on brainiac-nvidia):

- every 2-D projection of the language model and the token embedding (also the tied output) Q8_0;
- `in_proj_a` / `in_proj_b` (the DeltaNet's decay and write strength), every norm, A_log, dt_bias, the conv kernels F32;
- PAD's and EOP's embedding rows kept exact: their logits are computed in float32 from the checkpoint's rows (the
  release's `logits_fp32`: the silent / speak decision is close), and their Q8_0 rows' error is added back whenever they
  are the input token;
- the head, the projectors and the codec are read from the safetensors as half (bf16 -> f16, exact in range);
- the attention cache in half (`NS_Q35_KV` overrides; q8_0 doubled the worst hidden-state error);
- the activations unquantized (`NS_HARU_EXACT_X`, on): the shared decode products round each activation row to Q8_1
  first, and that moved the silent / speak decision - live calls fell silent after a few words ("Hello, my", "I'm a
  Haru, I") and the speak-first dump chose PAD where the reference says "."; `ns_q35_gemv_q8f` multiplies the Q8_0
  blocks by float32 rows, the prompt takes the half product instead of MMQ's int8. The weights' rounding is all that
  is left.

bf16 does not fit: the half GGUF (`NS_HARU_LM=f16`, 8.4 GB) needs 12.2 GiB on a card with 11.5 free, so Q8_0 is the
only format; its cost is measured against the release's float32 reference below.

## Checks against the reference

`reference/haru/ref.py` (nextsycl's) runs the release's own PyTorch code (github.com/higurashi-ai/Haru @ 7fe6b76) in
float32 on the CPU over the same safetensors; `nextcuda audio check DUMP --dir .../haru-4b --stages codec,prompt,frames`
compares each stage from the reference's own input, the frames teacher-forced (the reference's argmax token, its
head's latent from a fixed noise). Voice: a ZONOS3 preset recording (Imogen, 12.7 s); the 5070.

| stage | dump 1 (listening; user: Cillian, 9.7 s) | dump 2 (speak-first; user: a TTS question) |
|---|---|---|
| codec encoder, a frame at a time | rel 1.5e-3, cos 0.999999 | rel 2.2e-4, cos 1.000000 |
| codec decoder | rel 1.4e-4 | rel 1.4e-4 |
| card ids (268) | identical | identical |
| prompt embeddings (363 / 365 positions) | rel 5.4e-4 | rel 1.7e-4 |
| prompt hidden / logits | rel 8.0e-3 / 3.6e-3 (cos 0.999994) | rel 8.5e-3 / 2.9e-3 (cos 0.999996) |
| PAD / EOP logits at the prompt | 39.151 / 36.609 (ref 39.188 / 36.609) | -73.71 / -56.44 (ref -73.69 / -56.49) |
| teacher-forced frames: argmax | 13 of 13 | 17 of 17 |
| PAD-vs-EOP decision | 13 of 13 | 17 of 17 |
| worst hidden / logits rel | 5.6e-2 / 4.0e-2 | 6.4e-2 / 5.4e-2 |
| the head alone (the reference's hidden state and noise) | worst rel 4.5e-4 | worst rel 3.0e-3 |

With the shared int8-activation products (`NS_HARU_EXACT_X=0`) the same dumps gave argmax 13 of 13 and 16 of 17, prompt
logits rel 9.1e-3 / 4.6e-2 and PAD / EOP at the speak-first prompt -61.5 / -46.2: the activations' rounding, not the
weights', was most of the difference. A half language model (`NS_HARU_LM=f16`) does not fit beside the rest, so Q8_0
weights against bf16 are compared through the float32 reference only.

The head's latent from our own hidden state differs by 1e-2 - 1e-1 (8e-1 on the first spoken frame): the head amplifies
the language model's Q8_0 difference most on speech frames, as nextsycl saw with bf16 at a smaller scale (its logits
rel 1e-4 - 3e-3; Q8_0's are 3e-3 - 5e-2). The decisions that shape a call - silent or speak, which word - agree on all
30 frames.

## Speed and memory (brainiac-nvidia)

`nextcuda audio check DIR --stages speed` (120 frames of a live session, each from its samples in to its samples out):

| card | a frame (median) | worst | real time | (int8 activations) |
|---|---|---|---|---|
| RTX 5070 | 19.4 ms | 20.2 ms | 6.9x | 17.0 ms |
| RTX 3060 | 40.4 ms | 41.5 ms | 3.3x | 29.5 ms |

Memory: 7.2 GiB in use after the load (language model 4.2, head 0.6, codec 0.7 + 0.7), a conversation of 28,672
positions (an hour) 0.93 GiB in the default half cache: 8.3 GiB on the card with a conversation (`vram_gib: 9.0`). The
prompt (the card alone, 268 positions) takes 0.2 s on the 5070, 0.4 s on the 3060.

## Calls

Over the front door with `reference/haru/client.py`, at real time (TTS questions - "Hi Haru! How are you doing today?",
"So what do you like to do on the weekends?", "That sounds fun. Have you ever been to Japan?" - with pauses; voice: a
saved ZONOS3 preset recording, Imogen); the model's side written down by Whisper large-v3-turbo through the platform's
/api/audio/api/transcribe. Every frame is answered (246 sent, 247 back: frame 0 is the prompt's).

| call | first audio after connect | turnaround median / p95 | Whisper |
|---|---|---|---|
| 1 (int8 activations, before TCP_NODELAY) | 0.65 s | 60.5 / 65.3 ms | "Hey, I'm Haru, 22 years old. A senior in college. I love to go running every morning to clear my head. I'm really into everything tech and startups and business. Yes, I went to Tokyo about two years ago. It was amazing." |
| 3 (int8 activations) | 0.17 s | 17.6 / 17.9 ms | "I'm a Haru, I" on its text stream, then silent through two questions |
| 4 | 0.32 s | 20.0 / 20.4 ms | "Okay, so, um, I'm Haru. Nice to meet you. I'm good. I'm doing well. I'm 22 years old, a senior in college, and I've been thinking about everything. Uh, new tech, new startups, economics, history, you name it. Japan? Yeah, I have, but it's been like 20 years ago." |
| 5 | 0.27 s | 20.0 / 20.7 ms | "Hey, I'm Haru. I go running most mornings. I like to clear my head. Nope. Haven't been yet. I'd like to say it, though." (its text: "I'd love to though") |
| 6 | 0.27 s | 21.5 / 23.1 ms | "Hi, I'm Haru. I'm 22, a senior in college. I go running most mornings just to clear my head. I'm actually Japanese, but I only lived there for about six years. I like to think of myself as a global..." |

Whisper hears its text stream word for word, give or take a word (and a "Thank you." over the silence at the end).
Through the creative platform's Talk socket (`/api/talk/live`, box remote-nextcuda, the voice "bubbly british girl"
saved on the box by the platform): ready in 0.5-0.7 s, first audio 0.37-0.48 s after ready, the platform's turnaround
18-21 ms; Whisper: "Welcome to our lives. I'm truly excited to meet you. My name is Haru." / "Hi, I'm Haru. I'm 22."

## Differences from nextsycl (API)

None in the protocol: the `nextsycl` JSON key is kept (`nextcuda` accepted as an alias), the row is
`capabilities ["realtime","lockstep"], methods [], lockstep: true`. nextcuda sets TCP_NODELAY on a spliced upgrade's
sockets and on the realtime socket (nextsycl does not: Nagle against delayed ACKs added ~40 ms a frame here).
