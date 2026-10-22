# Running the LLMs

Chat on the two 12 GB cards: Qwen3.8-27B whole across both, Nemotron 3.5 Super (120B) on one with its experts streamed. The
settings are in [environment.md](environment.md); the model cards in [models.md](models.md).

## Qwen3.8-27B over both cards

**Qwen3.8-27B GSQ-RCO IQ3_S** (`llm/qwen35`; 48 gated DeltaNet + 16 gated attention layers, dense SwiGLU, its MTP
draft head, the vision tower) whole in the two cards' VRAM, nothing streamed: the 5070 does the work (40 of the 64
layers, the head and the draft block), the 3060 supports it (the other 24 layers, the vision tower). 80K context with
pictures (`NS_Q35_VISION=0`: 96K text only), the cache as q8_0. The same file brainiac-llm's llama.cpp serves.

```sh
curl -s http://127.0.0.1:8000/v1/chat/completions -H "authorization: Bearer $TOKEN" -H 'content-type: application/json' \
     -d '{"model": "qwen3.8-27b-gsq-iq3s", "messages": [{"role": "user", "content": "Hello"}],
          "chat_template_kwargs": {"enable_thinking": false}}'
```

Measured on the box against llama.cpp (b11112, its `qwen3.8-vision` / `-text` presets: q8_0 cache, flash attention,
MTP drafts) on the same prompts, greedy:

| | nextcuda | llama.cpp |
|---|---|---|
| a 16.6K-token prompt read | **1,200 tok/s** (CLI), ~1,160 through :8000 | 925 |
| a 3.1K-token prompt read | 1,010 tok/s | - |
| a 256-token story written | **44 tok/s** | 35.4 |
| written after the 16.6K prompt | 44-53 tok/s | 43.6 |

What it took (2026-10-18, from 691 tok/s and 31 tok/s that morning):

- **Both cards at once.** A prompt is pipelined like llama.cpp's ubatches: the 3060's layers of chunk k+1 are queued
  before the 5070's of chunk k, the rows cross by `ns_copy_peer_async` (pinned staging, ordered by events on the
  devices - the host never waits), chunks before the last read nothing back, token rows upload through pinned slots.
- **One CUDA graph a card a chunk.** A chunk is ~1,000 kernels a card; launched one by one they filled the stream's
  queue and the single host thread waited, leaving the 3060 idle ~80 ms a chunk. Captured and launched as one graph
  each, the 3060 is busy 13.2 of 14.3 s.
- **The work on the faster card.** The planner picks the chunk (512 rows) that leaves the 5070 the most layers; the
  cache as q8_0 (272 bytes a head's row, not 512) made room for them.
- **Prompt attention:** the queries in registers, key / value tiles in two `cp.async` stages, the hardware exponent:
  25% faster on the 3060.
- **ggml-cuda's MMQ and MMVQ** for the products (vendored from llama.cpp's tree, `kernels/llm/qwen35/third_party`): the
  shared matrix-vector kernels read the 3060's layers at ~97 GB/s of its 360. With MMVQ two drafts a step is best.

Settings ([`environment.md`](environment.md)): `NS_Q35_PIPE`, `NS_Q35_GRAPHS`, `NS_Q35_MMVQ`, `NS_Q35_KV`, `NS_Q35_KV_ROT`, `NS_Q35_DRAFTS` (2),
`NS_Q35_CHUNK`, `NS_Q35_HEAD`, `NS_Q35_VISION`; `NS_PROFILE=gpu NS_Q35_PROFILE=1` times every kernel by card and phase.
`nextcuda llm generate MODEL.gguf --gpu 1,0 --ctx 81920 --prompt-file F --effort none` is the engine without the server.
**The context's format** (`NS_Q35_KV`, `NS_Q35_KV_ROT`): per card, chosen by the plan. Measured after the 16.6K prompt,
256 tokens teacher-forced (`llm generate --score TEXT --ref OUT`) against the half cache, where only changing the chunk
size gives KL 0.00093:

| `NS_Q35_KV` | 3060 / 5070 | KL mean (max) | top-1 same | 80K: 5070 layers, prompt | 192K |
|---|---|---|---|---|---|
| `auto` (default) | half / q8_0 | 0.00054 (0.003) | 97.3% | 40, 1,230 tok/s | q8 / q8: 873 tok/s |
| `q8` | q8_0 / q8_0 | 0.00080 (0.005) | 97.7% | | |
| `q4` | q4_0 / q4_0 | 0.0082 (0.24) | 93.4% | | |
| `q4` + `NS_Q35_KV_ROT=1` | q4_0 rotated | 0.0055 (0.21) | 97.3% | | |
| `auto-q4` + `NS_Q35_KV_ROT=1` | half / q4_0 rotated | 0.0032 (0.017) | 96.5% | 45, 1,493 tok/s | 1,145 tok/s |

q8_0 is within the noise; q4_0 is ~3-9x it, the rotation (a 32-block Hadamard transform of keys, values and queries,
QuaRot's) taking a third off. q4_0 on the 5070 is the fast and long-context choice: five more layers there.

Also done: prompts with pictures are pipelined too (their positions and rows through pinned slots); a layer's
tensors are one allocation (one each had lost ~260 MB on the 5070 to 2 MiB rounding); a shared input is quantized
once for the products that read it; four drafts a step load at 80K.

## Nemotron 3.5 Super

**Nemotron 3.5 Super 120B-A12B UD-IQ1_M** (`llm/nemotronh`; Mamba-2 + latent MoE + attention) on one
12 GB card. The dense weights (5.8 GiB) and the experts wanted most stay in VRAM, a cache of experts in pinned host
memory, the rest is read from the file as decode asks for it; the 3060 on its x1 link is not used.

```sh
nextcuda models pull nemotron-3.5-super-iq1m          # 46 GB, or --from DIR to link files you have
nextcuda llm start nemotron-3.5-super-iq1m            # a process (pid, log in the socket directory), up in ~25 s
nextcuda llm chat "What is the capital of France?"    # one request through the control socket, streamed
nextcuda llm status                                   # the model, the card, the prompt cache
nextcuda llm logs | nextcuda llm stop
```

The server speaks OpenAI's chat API on `127.0.0.1:8085` (`/v1/chat/completions`, `/v1/models`; `/api/chat` for a page);
through the front door, `llm.chat` (or `/v1/chat/completions` on :8000) starts it when a request names the model:

```sh
curl -s http://127.0.0.1:8085/v1/chat/completions -H 'content-type: application/json' \
     -d '{"model": "nemotron-3.5-super-iq1m", "max_tokens": 200, "messages": [{"role": "user", "content": "Hello"}]}'
```

- **Prompt cache:** the server checkpoints each prompt (every attention layer's keys and values, every Mamba layer's
  state: ~175 MB + 8 KB a token) at the start of its first and last user turn and at its end, in host memory
  (`NS_PROMPT_CACHE_MIB`, the card's 2048) and then on disk (`NS_CACHE_DIR`, default `~/.cache/nextcuda/prompts`).
  A request that shares a system prompt or continues a chat reads only what is new: measured, a 1,314-token prompt
  read in 18.1 s the first time took 2.1 s the second.
- **The expert profile** (`expert-profile.txt` beside the model): which experts decode asked for. The next start fills
  VRAM and the pinned cache by it, so the first runs of a new topic are slower than later ones.
- **Memory:** `NS_NH_PINNED_GIB` is the biggest knob (the card sets 22; 25 is faster but leaves the 31 GB box little
  else, 18 slower). The prompt cache comes out of the same RAM.
- **Speed on the 5070** (`nh --bench`, [`benchy/nemotron-3.5-super-iq1m.md`](benchy/nemotron-3.5-super-iq1m.md)): decode 8-10.5 tok/s with 25 GiB
  pinned, 6-8 with the card's 22, 4-6 with 18; prompts 170-190 tok/s (a prompt longer than 128 rows borrows 2.6 GiB of
  the VRAM's hot experts for passes of 2,048 rows, read back after it).
- **Engine options** (`nextcuda llm engines`): `--opt-pinned-gib`, `--opt-chunk` (prompt rows a pass, 2048),
  `--opt-reserve-mib` (VRAM left free, 700). `NS_NH_CPU_FRAC` gives part of each layer's pinned experts to the CPU
  (ggml-cpu, `libggml-base.so` / `libggml-cpu.so` beside the program) - measured slower here than the copies over
  gen4 x16, so it is off.
- **Open WebUI** (`http://<host>:3000`, the page's Chat link): `tooling/open-webui.sh` (re)makes its container (the
  `open-webui` volume kept) with the front door's `/v1` and token as its OpenAI connection, and
  `tooling/open-webui.service` starts it with the box. The chat model loads on the first message and unloads after
  `NS_IDLE_MINUTES`; set the connection's header `{"X-Keep-Up-S": "300"}` to keep it up between messages. Its
  read-aloud speaks through `/v1/audio/speech` (Admin > Settings > Audio: OpenAI, `http://host.containers.internal:8000/v1`,
  model `voxcpm2-int8`, a saved voice's name) - sentence by sentence, faster than it plays, beside the chat model.
  A follow-up on a long pasted document reuses its checkpoint (the prompt cache keeps the start of each reply).
- **`nh`** (`llm/nemotronh/src/bin/nh.rs`): the engine without the server - `nh MODEL --prompt TEXT --n N`, or
  `--bench 20,2000,8000 --new 128` for the numbers.
