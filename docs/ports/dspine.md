# DSpine drafter for Qwen3.8-27B (crates/qwen35), over the two cards

Tencent's `Qwen3.8-27B-DSpine-b16` (huggingface.co/tencent/Qwen3.8-27B-DSpine-b16, Apache-2.0, 2.03 B parameters in
bfloat16, `DFlashDraftModel`): a speculative-decoding *draft* for the dense Qwen3.8-27B, run by the qwen35 engine in MTP's
place. Ported from nextsycl's port (its `docs/ports/dspine.md`, `llm/qwen35/src/dspine.rs`, `kernels/llm/qwen35/silo/dspine.cpp`,
`tooling/dspine-ref.py`: the working answer, checked there against Tencent's `dspine_mlx`). Opt-in: the card
`qwen3.8-27b-gsq-iq3s-dspine` (`NS_Q35_DSPINE=auto`), exact (the target's own sampler verifies every draft). It wins on
code (75 tok/s against MTP's 57, greedy, at 48K of context) and loses on prose and long documents, so the MTP card stays the
default.

Files: `crates/qwen35/src/dspine.rs` (config, weights, the ring, pure logic with tests), `crates/qwen35/src/engine.rs`
(the fc partial sums in `run_layers`, `dspine_rows`, `dspine_draft`), `kernels/llm/qwen35/dspine.cu` (the bridge, the
refinement, top-k, an attention for heads of 128), `kernels/llm/qwen35/q35_vq8.cu` (the verify products), the reference
`tooling/dspine-ref.py` (dev-only numpy, never at runtime). The checkpoint is the original safetensors, read directly.

## What it is (nextsycl's port, unchanged in substance)

- **Context**: for every committed position, `ctx = hidden_norm(fc(concat of the target's residual stream after layers 1,
  16, 31, 46, 61))` (fc 25,600 -> 5,120); each of the five draft layers keeps k / v of `ctx` (k_norm, rotary at the
  position) in its cache, written behind every target pass (prompt chunks and verify rows).
- **Block**: rows `[anchor, mask x k]` (the last committed token, then mask token 248070) embedded with the *target's*
  embedding; five layers (32 query / 8 key heads of 128, q / k norm, NeoX rotary at theta 1e7, SwiGLU of 17,408), attention
  over the context (window 8,192) and the block causally. After every layer the **code bridge**: `rms = |raw| / sqrt(H)`,
  `code = sqrt(1024) unit(read_l (raw / (rms + eps)))` (row 0: `sqrt(1024) codebook[anchor]`), `packed = [G; Gp; A] code`,
  `message[i] = sigmoid(q[i] + s[i-1] + bias) v[i-1]`, `out = raw + rms W_l message`.
- **Head and refinement**: the final norm, the target's head -> each row's 16 best tokens; row i's pick among them is
  conditioned on row i - 1's pick (the anchor for row 0): `score[k] = raw . norm_w . head_row[cand k] + rms . m . tg[cand k]`,
  `m = sigmoid(q + tsv[sender][:512] + bias) tsv[sender][512:]`. The drafts are the chain of picks.
- **The ring**: the draft attends only the last 8,192 positions, so its cache is a mirrored ring of 8,200 slots a head
  (nextsycl's `Ring`: slot `p % 8200`, also written at `+ 8200`, so any window reads one contiguous run; 8 slots of slack for
  a verify pass and its rollback): 0.31 GiB a session at any context, copied and checkpointed by position.

## What changed for nextcuda

- **Two cards.** The target's layers are split 3060 / 5070 (the 5070 last: the head, and now the draft). Target layers 1
  and 16 run on the 3060, 31 / 46 / 61 on the 5070. fc is linear, so each card makes its own share: fc's column group j
  lives on the card of target layer j, `run_layers` adds `fc_j . x` into a partial sum after layer j, and the partial sum
  crosses to the 5070 beside the rows (one extra row set over PCIe, not two of aux states; in a pipelined prompt it crosses
  into its own pair of buffers, as the rows do). `finish` then norms it and writes the draft's k / v into the ring
  (`dspine_rows`). The draft itself runs on the 5070 next to the head it shares.
- **No vocabulary tables.** nextsycl builds `TG` (V x 512) and `TSV` (V x 1,024) in half at load: 0.76 GiB. Here each draft
  makes only the rows it needs: its 7 x 16 candidates' head rows are gathered from the head's blocks (one gather, one
  expansion to half), `tg = rows . (norm_w W_last)^T` and `tsv = sqrt(C) codebook[senders] . [Gp; A]^T` are two small cuBLAS
  products (the senders' codebook rows are read from the checkpoint file - page cache - after the candidates come back).
  That is the 0.76 GiB the 5070 does not have, for ~0.3 ms a draft. (It also matches the reference better: tg is float32.)
- **Attention for heads of 128**: the engine's attention is for the target's 256; `ns_q35_dsp_attn` is a small split-K
  kernel (the four query heads of a key head x up to 8 rows = 32 queries a block, 64 keys a run, the runs merged), so no
  four-call trick.
- **Q8_0 in MMQ**: the draft's layers are Q8_0 (made on the host at load, as nextsycl); `q35_mmq.cu` now has ggml's Q8_0
  MMQ case, so prompt chunks' fc and k / v products run on the tensor cores (the target has no Q8_0 tensor: nothing else
  changes).
- **Memory** (1.87 GiB of draft weights on the 5070, 0.05 on the 3060; its ring 0.31 GiB a session; a verify pass's DeltaNet
  snapshots for 7 drafts, 0.5 GiB more than MTP's 2): the plan moves target layers from the 5070 to the 3060 to make room -
  layers 0-22 on the 3060 with MTP at 80K, 0-32 with DSpine at 80K, 0-29 at 48K. More layers on the slower card make every
  pass slower (one row 37.4 ms against 33.8) and prompts slower (15K tokens at 902-927 tok/s against 1,245).

## The verify products (nextsycl's "XMX verify products", for CUDA) - measured first

nextsycl's gain needed fast 5-8-column products for the 8-row verify pass. Measured here first, the whole pass (MTP's split,
greedy rows, `--rows-bench`):

| rows a pass | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MMVQ (ggml-cuda, the default) | 33.8 | 36.4 | 42.0 | 45.4 | 48.2 | 54.4 | 60.8 | 66.5 ms |
| MMQ (`NS_Q35_VERIFY_MMQ=2`) | 33.9 | 52.2 | 53.5 | 52.9 | 53.5 | 53.9 | 54.6 | 54.8 ms |

MMVQ decodes each weight block again for every input row, so from 6 rows MMQ's tensor cores win (8 rows: -18%); below
that MMVQ stays. Per card (DSpine's split): MMQ instead of MMVQ at 8 rows saves 12 ms on the 3060 and 3.4 on the 5070
(the 3060 runs MMQ's 8 columns at ~124 GB/s of its 360). `NS_Q35_VERIFY_MMQ=6` (a threshold a card).

Then a dedicated kernel, `q35_vq8.cu` (`NS_Q35_VQ8`): each block of 16 matrix rows staged in shared memory a step
(coalesced words), each 32-value block decoded to int8 once and multiplied with all 8 input rows by one `mma.m16n8k32`
(MMVQ's integer sums block by block: checked against MMVQ with `NS_Q35_VQ8_CHECK=1`, 1e-7 relative on every matrix shape),
for IQ3_S / IQ4_XS / IQ3_XXS (82% of the layers' bytes; the 2-bit i-quants and Q4_K were written too, no faster than MMQ,
dropped). A cp.async double buffer and a register prefetch were both slower than the plain staged loop. Result: an 8-row
pass 61.0 ms (DSpine's split) against 62.6 with MMQ alone - 2.5%, 150 GB/s on the 3060 against MMQ's 124, 396 on the 5070
against 343. The 3060's int8 decode of these i-quants stays the limit (MMVQ reads them at ~240 GB/s one row at a time). On
code at 80K: 65.3 tok/s with MMQ, 68.3 with vq8 (+5%, 7 drafts).

## Checked

- **Against the numpy reference** (`tooling/dspine-ref.py`, nextsycl's, reading this engine's dump: `NS_Q35_DSPINE_DUMP`,
  the residual stream after each target layer of every pass from the card that ran it, the fc partial sums, each draft):
  the fc partial sums made across the two cards against fc of the aux rows: worst cosine 0.99996; final rows cosine
  0.9997-0.99994; the last layer's raw 0.99987-0.99989, packed 0.99999; the candidates are the logits' top 16 in every row;
  the refinement's picks match in 112 of 112 (row, sender) cases; the drafts are the same tokens (3 of 3 drafts checked, 196
  context positions). Past the ring's wrap (the 14,930-token document, drafts at 14,930 and 14,938: the window read from the
  mirrored ring): fc partial sums 0.99994, final rows 0.99982-0.99992, 112 of 112 picks, the same drafts.
- **Exactness** (the same layer split for every run, `NS_SPLIT=30`: which card runs a layer changes its float sums, so runs
  with different splits differ anyway): with 1 draft (2-row verify passes) DSpine's output is byte for byte plain greedy on
  code and prose, and MTP's with 1 draft the same; at 3 drafts (4 rows) code is still byte-identical. Where the engine's
  multi-row passes sum in another order than one row (the attention from 3 rows on the tensor cores; MMQ / vq8 from 6
  rows; over a 15K context even at 2 rows) greedy text departs at a near-tie, for MTP as for DSpine: MTP 2 drafts at byte 479
  of code and 458 of prose, DSpine 7 at 315 and 458; on the document MTP 1 draft at 165, DSpine 1-7 at 639. The text is
  as valid, not the same - as nextsycl's 5-7 drafts with its XMX products.
- **Through the server** (`nextcuda llm serve`, the card's settings, one session): a chat, the same chat again (177 of its
  184 prompt tokens restored from a checkpoint - the ring saved and restored by position) and a follow-up (from the live
  session): the second answer the first's byte for byte, drafts 38 of 39, 97 of 105 accepted; 81-106 tok/s on short code
  answers.

## Speed (brainiac-nvidia: RTX 5070 + RTX 3060, 2026-11-09, greedy 256 tokens with --ignore-eos)

tg tok/s (tokens a verify round = 256 / (256 - drafts accepted) in brackets); every run one load. The prompts: a code task
(196 tokens with the chat template), a short story (191), and a 14,930-token document to summarize section by section.
MTP is the live card (`qwen3.8-27b-gsq-iq3s`: 2 drafts, 80K); DSpine with `NS_Q35_VERIFY_MMQ=6 NS_Q35_VQ8=6`.

| prompt | plain | MTP 2 (80K) | DSpine 3 / 5 / 7 at 80K | DSpine 3 / 5 / 7 at 48K (the card) |
|---|---:|---:|---:|---:|
| code | 31.1 | 57.2 (2.6) | 57.0 (3.3) / 60.9 (4.3) / 71.3 (4.9) | 61.0 (3.4) / 67.5 (4.6) / **75.0** (4.9) |
| prose | 30.8 | **44.4** (2.1) | 35.1 (2.1) / 32.7 (2.2) / 32.5 (2.2) | 37.6 (2.1) / 31.7 (2.1) / 32.1 (2.1) |
| 15K document | 29.0 | **52.1** (2.7) | 45.0 (3.1) / 44.8 (3.4) / 47.6 (3.9) | 49.3 (3.1) / 48.8 (3.5) / 49.6 (3.8) |
| 15K document, prompt tok/s | 1,453 | 1,245 | 922 / 927 / 902 | 1,048 / 1,015 / 993 |

- **Code**: DSpine accepts 4.9 tokens a round at 7 drafts against MTP's 2.6 at 2, and wins by 25% at the same 80K context,
  31% at 48K (75.0 against 57.2) - and 39% on the same split (layers 0-29 on the 3060 for both: MTP 54.0, DSpine 74.9).
  nextsycl measured +21% on its B70 (71 against 59).
- **Prose**: the drafts past the second are rarely right on this story (2.1-2.2 tokens a round whatever the drafts), and a
  longer round costs more: MTP wins. (nextsycl's prose prompt drafted better: 3.2 a round.)
- **The long document**: DSpine accepts more a round (3.9 against 2.7) but a round costs more; MTP wins, as on nextsycl.
- **Prompts** read 15-30% slower: the layers the draft pushes onto the 3060.

### The context trade

What the draft's 2.7 GiB on the 5070 cost, by context (7 drafts, code; the 5070's layers; peak memory a card):

| ctx | split (3060 / 5070) | code tg | 5070 / 3060 peak | room on the 3060 |
|---|---|---:|---|---|
| MTP card, 80K | 0-22 / 23-63 | 57.2 | 11.1 / 6.0 GB | 6.2 GB: a speech model (voxcpm2-int8 4.2 GB) fits beside it |
| DSpine 80K | 0-32 / 33-63 | 71.3 | 11.0 / 8.6 GB | 3.6 GB: voxcpm2-int8 does not |
| DSpine 48K (the card) | 0-29 / 30-63 | 75.0 | 11.1 / 7.4 GB | 4.8 GB: it does |
| DSpine 32K | 0-28 / 29-63 | 75.1 | | |

So the card gives up 32K of context (80K -> 48K) for +5% over DSpine at 80K and, more to the point, to keep the speech model
beside it as the MTP card does; below 48K nothing more is won.

## Not done

15 drafts / a 16-row verify (the engine's verify pass is at most 8 rows: the products and the snapshots), a confidence cut of
the drafts (prose would want 2-3, code 7), a faster 3060 decode of the i-quants at 8 columns, and DSpine on one card.

## Env

`NS_Q35_DSPINE=DIR|auto`, `NS_Q35_DSPINE_DUMP=DIR` (debug dump for the reference), `NS_Q35_DRAFTS` (at most 7),
`NS_Q35_VERIFY_MMQ`, `NS_Q35_VQ8` (thresholds, one a card allowed), `NS_Q35_VQ8_CHECK`, `NS_Q35_VQ8_TYPES`, `NS_Q35_VQ8_WARPS`
(docs/environment.md). `nextcuda llm generate ... --rows-bench` times passes of 1..8 rows.
