# Inkling-2-Mini on nextcuda

Thinking Machines' Inkling-2-Mini (2026-11-20, Apache-2.0) is a looped MoE of 26.2B (3.4B active): 24 layers of 2,304,
the middle twelve run three times with the same weights (48 layer applications a token), 128 NVFP4 experts (8 a token,
SiTU), attention with a relative-position bias and a per-row sink (20 layers over a 1,024 window, 4 full), short
convolutions, attention over the depth, and a DSpark drafter. nextcuda runs Unsloth's UD-Q4_K_XL GGUF (its
`Mainline_Support` layout, arch `inkling2_loop`) as the `llm/inkling2` engine, card `inkling-2-mini-q4kxl`, over the
RTX 5070 and the RTX 3060 of brainiac-nvidia. This is a port of nextsycl's `llm/inkling2` (the user's own project: its
docs/ports/inkling2.md has the model's anatomy, the GGUF layouts and the sources); what follows is what differs here and
what was measured.

## What was ported, from where

| part | here | origin |
|---|---|---|
| the forward pass (the loop's slot numbering, the depth mixes, the controllers, the router, the coda, the head), sessions, checkpoints, the decoder | `llm/inkling2/src/engine.rs`, restructured for two GPUs (below) | nextsycl `llm/inkling2/src/engine.rs` |
| the DSpark drafter (four kept features, the non-causal block pass, the Markov RNN, the head fold) | `engine/dspark.rs`, on the head's GPU | nextsycl `engine/dspark.rs` |
| the float32 kernels (ink.h: every op's semantics) | `kernels/llm/inkling2/ink.cu`: the same names and arithmetic, a warp of 32 for a sub-group of 16; the RMS norm, half and the cuBLAS products its shared library had | nextsycl `kernels/llm/inkling2/ink.cpp` |
| the decode silo (silo.h's API) | `kernels/llm/inkling2/silo/*.cu`, written for CUDA (below) | nextsycl's silo API; the kernels are new |
| the prompt silo: flash attention with the bias and the sink, the experts' grouped products | `silo/pattn.cu`, `silo/pgemm.cu` (tensor cores, mma.sync) | new (nextsycl's pattn.cpp is XMX) |
| NVFP4 (ggml 40), `inkling_chat` / `inkling_channels`, the `inkling2` pre-tokenizer, `<|message_user|>` in the user-turn markers, the `ink` channel split in the server | `crates/gguf`, `crates/tok`, `glue/serve/src/serve.rs` | nextsycl, as is |
| the card's files, pinned revision, drafter key; the guide | `model-cards/llm/inkling-2-mini-q4kxl.yml`, `guides/inkling2.md` | nextsycl's cards |
| deq-ref, ink-dump (the parity tools, run outside nextcuda against llama.cpp 690b87d built for the CPU) | `reference/inkling2/` | nextsycl |

Not ported: UD-Q8_K_XL (22.9 GB: not on two 12 GB cards with a context), the vision tower (nextsycl has none either).
No third-party code: the NVFP4 decode, the products, the attention and the grouped expert GEMM are this port's own.

## Two cards

UD-Q4_K_XL is 17.6 GB (15.9 GiB on the GPUs: the embedding's rows stay in the file). It does not fit a 12 GB card, so the
engine splits it - by what the cards report (`free_or_guess`), never by name or size: the faster card (the last of
`--gpu`, the 5070) takes the loop's layers 6-17 and, a layer at a time from the loop out (the coda's first), as many
neighbours as it has room for with the session's cache and the work memory; the rest of the prelude and the coda, the
head, the embedding and the drafter go to the other. Whatever the split, a token crosses twice: after the prelude and
before the head (`hop`: the pass's bank of streams the rest of the pass reads - the stream, the depth attention's blocks,
the loop's prelude output and bank - and the drafter's features; a decode pass's whole bank, 73 KB, in one copy through
pinned memory, ordered on the GPUs only). At 32K the plan was layers 6-20 on the 5070 (39 of the 48 applications);
`NS_INK2_SPLIT=A,B` pins it. A load or a session that does not fit refuses with `no_room` naming the card (a 409 at the
front door).

## The 5070 alone

`NS_INK2_PLACE=one` (card `inkling-2-mini-q4kxl-5070`) keeps the 3060 free: the loop's layers, every attention and
the head stay on the 5070; the routed experts of layers outside the loop (the largest stacks first, until the rest fits
- all twelve, 8.3 GiB, at 32K) are repacked on the host into the silo's layout and kept in pinned memory
(`engine/stream.rs`). A pool of slots a matrix shape (gate | up, down; NVFP4, Q8_0) takes what the plan leaves (64 - 150
experts each). At such a layer application the pass's ids are read back, the experts not in the pool copied in on the
copy lane (LRU over the slots; 28 GB/s over the 5070's PCIe gen4 x16), the stream ordered after them, and the same
decode / verify / grouped prompt kernels run on the pool with the ids remapped to slots - so the text is the same:
greedy 48 / 48, 48 / 48, 32 / 32 against llama.cpp. 25-45% of the experts a token asks for are found in the pools; a
token copies ~60-100 experts (~0.3 GB). Measured (no drafter; its verify rows would pull more experts over PCIe):

| | pp 512 | pp 4K | pp 32K | tg after 512 | tg after 4K | tg after 32K |
|---|---|---|---|---|---|---|
| the 5070 alone | 593 | 1,547 | 1,112 | 65.8 | 43.2 | 36.9 |
| two cards | 838 | 1,449 | 1,289 | 80.2 | 75.4 | 60.2 |

Prompts read as fast or faster on the 5070 alone: on this box the 3060 sits on a PCIe gen4 x1 link (1.6 GB/s), which
two-card prompt passes cross twice. A smaller quantization was the other way to one card: Unsloth's UD-Q2_K_XL (9.1
GB) would fit whole, but its experts are IQ1_S / IQ2_XXS / IQ2_XS / IQ2_S (gate | up) with eight other K / IQ types the
engine does not have, at a quality cost nextsycl's port measured as not worth it for UD-Q4_K_M already; streaming keeps
UD-Q4_K_XL's exact weights.

## The silo, for CUDA

- **Repacked matrices.** Q8_0 and NVFP4 rows are repacked at load (losslessly): the quants or the E2M1 codes of all rows,
  then the scales. Within each 16 values the quants are stored in the order 0 8 1 9 2 10 ... 7 15 - the order an NVFP4
  byte's two nibbles hold them - and the activations are quantized in the same order, so the dot is unchanged.
- **Decode products** (1-16 rows): 8 lanes a row, a 32-value line at a time (16 bytes of NVFP4 codes, or 32 of Q8_0
  quants), decoded once and dotted with every row of Q8_0 activations by `dp4a`. An NVFP4 word's four codes become four
  int8 values (twice their E2M1 values) by three byte permutes: magnitudes from one table, negations from another, the
  byte picked by the code's sign bit; the UE4M3 scales by their bits. The lanes' sums are added in a fixed order and the
  per-row arithmetic is written with explicit FMAs, so a row's result does not depend on what shares its pass.
- **Experts**: gate | up with SiTU and the activations' quantization in one launch (a block 32 activations: 64 rows of
  the expert); down weighted and summed in order; the union forms of a verify pass (each distinct expert's lines read
  once for all its rows) give the same bits.
- **Fused steps, decode attention, the drafter's steps**: dec.cpp / attn.cpp / draft.cpp's, for CUDA.
- **Prompt rows**: flash attention on the tensor cores (mma.sync m16n8k16, half in, float32 sums) with the relative bias
  added to each logit within the extent, the window and the sink as a first logit; and the routed experts as grouped
  products (every expert's rows one launch a matrix, its weights decoded from the repacked stack into shared memory as
  the product runs).

## Checks

Reference: llama.cpp 690b87d built for the CPU (`llama-ink-dump`, `deq-ref`), UD-Q4_K_XL. Engine: brainiac-nvidia,
3060 + 5070 (`--gpu 1,0`).

- **Weights**: every expansion (Q8_0, NVFP4, BF16) bit for bit ggml's `dequantize_row_*` (`llm kernels` with
  `NS_INK2_DEQ_REF`: 0 differences in 4,096 rows of each type); the silo's products within the Q8_0 activations'
  rounding (rel err 4.4e-4 Q8_0, 6.3e-4 NVFP4 - nextsycl's figures).
- **Layers** (`llm check`, `l_out` and the logits by cosine; the next token):

| prompt | tokens | layer outputs | logits | next token | greedy run against llama.cpp's |
|---|---|---|---|---|---|
| short | 23 | 0.9991 - 0.99998 | 0.99999 | same | 48 of 48 the same |
| medium | 511 | at least 0.980 | 0.9995 | same | 48 of 48 (the engine's own prompt kernels); 28 of 48 with the silo's |
| long | 3,305 (the last 233 rows compared) | at least 0.992 | 0.9999 | same | 32 of 32 the same |

  The medium prompt's token 28 moves with either of the silo's prompt kernels alone (`NS_INK2_PATTN=0`,
  `NS_INK2_PGEMM=0`: both off gives 48 of 48); both paths are as far from llama.cpp at every layer (0.982 / 0.980), and
  the step's log-probabilities differ by ~0.2 between them - the sensitivity nextsycl documents (48 applications of the
  same weights amplify a rounding difference), not a fault.
- **spec-check**: a 5-row verify pass against one-row decode: 0 difference, the top token the same in 100 of 100 rows.

## Speed (brainiac-nvidia, 3060 + 5070)

`llm generate --ids`, greedy, the prompt then 128 generated tokens (`pp` the prompt's tokens a second, `tg` the
generated tokens a second after it), no drafter; the long reference prompt's tokens repeated to the size:

| | pp 512 | pp 4K | pp 32K | tg after 512 | tg after 4K | tg after 32K |
|---|---|---|---|---|---|---|
| nextcuda, 3060 + 5070 | 838 | 1,449 | 1,289 | 80.2 | 75.4 | 60.2 |
| (the engine's own prompt kernels, `NS_INK2_PATTN=0 NS_INK2_PGEMM=0`) | 791 | 1,386 | 1,091 | 77.4 | 72.8 | 59.3 |
| nextsycl, B70 (32 GB; its doc, round 4) | 1,134 | 2,095 | 1,650 | ~70 | - | - |
| nextsycl, B65 | 831 | 1,458 | 1,084 | ~62 | - | - |

The engine's own prompt path (expansion of each expert to half, a cuBLAS product each, float32 attention) read 3.3K
tokens at ~420 tok/s; the silo's flash attention and grouped expert products made it 1,330.

**Where a decode token goes** (`NS_INK2_PROFILE=gpu NS_PROFILE=gpu`, 256 tokens, both cards' device time a token -
they run one after the other, so it adds to the token's ~13 ms): the experts 4.7 ms (gate | up 2.8, down 2.0; ~390 GB/s
over the 1.85 GB they read), the output projection and the head 2.4 (the head alone ~1.5 on the 3060), the five
attention projections in one launch 2.1, the depth mixes 1.0 (~120 single-block launches), the router 0.5, the
attention 0.5, the controllers 0.8, the post-projection step, the norms and the convolutions ~1.1. The bound: a token
reads ~4 GB, 2.5 on the 5070 (~4.2 ms at 600 GB/s) and 1.5 on the 3060 (~4.3 ms at 340): ~8.5 ms, ~118 tok/s; tg 80
is ~68% of it.

**The drafter** (400 tokens of the chat template, effort low; `drafts` 5, `p_min` 0.8):

| text | temperature | plain | drafted, rested after a fruitless step (`NS_INK2_DRAFT_REST`) | drafted, every step |
|---|---|---|---|---|
| prose | 0 | 79.9 | 75.9 | 67.9 |
| code | 0 | 79.9 | 100.4 | 101.1 |
| prose | 1.0 | 77.0 | 74.9 | 68.4 |
| code | 1.0 | 76.0 | 92.6 | 92.2 |

A step's drafter pass and the head's product over its rows cost ~2.5 ms on the 3060 whether or not its drafts are
taken: on prose (few drafts sure enough, a third of those rejected) that was a loss of 15%. The rest - a step without
the drafter after one whose drafts bought nothing, then 2, 4 .. 8 while that goes on, back to drafting every step once
one is accepted - keeps code's gain and most of prose's speed. A sweep of `drafts` / `p_min` (2..7, 0.8 / 0.9) moved
code by a few percent and never made prose pay.

## Memory and context

A token of context takes 32 KiB of attention cache in half (8 full-attention slots: the loop's 6 on the 5070, layers
5 and 23 on the 3060; the windowed slots are fixed rings of a window and a chunk), 17 KiB in q8 (`NS_INK2_KV=q8`: a
head's row of 128 int8 and 4 half scales, read and written by the silo's kernels). `llm serve --ctx max` (`ctx: max`
on a card) plans the load for the most tokens that fit (prompt chunks of 1,024 or 512), then measures what the cards
have free after the weights and serves one session of the most a session's own check passes; a load that cannot hold
1K is refused (no_room: a 409), so an answer never runs out of room mid-way.

| card | placement | cache | context | memory (5070 / 3060) | pp / tg at that context |
|---|---|---|---|---|---|
| inkling-2-mini-q4kxl | two cards, drafter | half | 32K (the card) | 10,851 / 10,595 MiB | 1,289 / 60.2 (32K, no drafter) |
| (`--ctx max`) | two cards, drafter | half | 125K (128,000) | 11,559 / 11,465 MiB | - |
| (`--ctx max`, q8) | two cards, drafter | q8 | 210K (215,040) | 11,123 / 11,665 MiB | - |
| inkling-2-mini-q4kxl-long | two cards, no drafter | q8 | 256K (`max:262144`; they hold 296,960) | ~11,300 / ~11,300 MiB | 566 / 21.6 (a 250K prompt) |
| inkling-2-mini-q4kxl-5070 | the 5070, experts streamed | half | 32K (the card) | 11,115 MiB | 1,112 / 36.9 (32K) |
| (`--ctx max`) | the 5070 | half | 55K (56,320) | 11,593 MiB | - |
| inkling-2-mini-q4kxl-5070-long | the 5070 | q8 | 110K (112,640) | 11,561 MiB | 660 / 30.7 (a 90K prompt) |

`--ctx max` never goes past the model's own context (its GGUF's `context_length`: 1,048,576 for this file), and a card
may cap it lower (`max:N`): the long card serves 262,144, the context nextsycl's cards and the release quote. Two cards
reach it with the q8 cache and no drafter (with the drafter they hold 215,040, so it stays off there) (the drafter's own cache - half, a full-attention layer of its three - and its
weights take ~1.3 GiB of the 3060). The q8 cache against the half one (both on the silo): logits cosine against
llama.cpp 0.99984 / 0.99988 on the medium / long prompts (half: 0.99939 / 0.9999), every layer within the same range,
the next token the same; greedy runs the same as the half cache's on the medium and long prompts, the short one's token
13 moved (the reference's own top-1 / top-2 gap there 0.11: a near tie). Of the ways to more context it measured the
cheapest: streaming more experts frees memory only on the 5070-alone placement (it already streams all twelve layers
outside the loop at its maximum), and the loop's experts are read three times a token - streaming them would cost
far more than the cache saves; a cache on the host would put every token's attention over PCIe.

## Not done

- **UD-Q8_K_XL** (22.9 GB): the weights alone are more than the two cards less their contexts; nextsycl's card for it
  is not ported.
- **The head on the 5070**: the 3060 reads the head (492 MB, ~1.5 ms of a ~12.5 ms token) and the drafter's head
  product; the 5070's room goes to layers instead. Putting the head there would cost a third crossing a token.
- **The drafter's own cost on prose**: a step's drafter pass and its head product (~2.5 ms) do not pay where few drafts
  are accepted; `NS_INK2_DRAFT_REST` rests it after a fruitless step, which narrows the loss but does not remove it.
- **Small decode launches**: ~3 ms of a token are latency-bound one-block kernels (the depth mixes, the router's choice,
  the controllers, the post-projection step); fusing them further is nextsycl's round 4 territory.
- **The drafter against a reference**: as nextsycl - no runnable reference produces its tokens; its acceptance is the
  check, and the verify keeps the text the model's own (spec-check 0).
- **Vision**: not in either port.

## Lessons

- **Two sessions building one tree.** podman's `:Z` relabels the mounted tree for the container that mounts it: two
  `./build.sh` at once (this port's and the voice isolator's) failed each other with "Permission denied" mid-build. This
  port built from a git worktree of its own after that.
- **A shared array read as float4 must be declared aligned** (`__align__(16)`): the depth mix's row buffer was not,
  and the first silo decode step stopped with a misaligned address.
- **512 threads of a template with 16 accumulators** asked for more registers than a block has (the union form of the
  experts' gate | up): `__launch_bounds__(512)` keeps it to what fits.
- **The medium prompt's sensitivity** is real: two prompt paths, each as close to llama.cpp at every layer, give
  log-probabilities ~0.2 apart at its 28th generated token. Judge a prompt path by the layers and the next token, a
  greedy run only as far as the reference's own gaps allow (nextsycl's lesson, seen again).
