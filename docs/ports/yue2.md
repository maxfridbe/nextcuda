# Porting YuE2 (M-A-P YuE2-3B) to nextcuda

The plan of the port, written before the code and kept as the record of what was decided and why. YuE2 turns a style
prompt and lyrics into a whole song with vocals, 48 kHz stereo; non-commercial (CC BY-NC 4.0).

## What the model is

The reference is upstream's inference wheel, `yue2_infer-0.1.5` (m-a-p/YuE2-3B rev 8f24312f, unzipped and read whole):
`protocol.py` (the prompts, the token ids, the defaults), `sampling.py` (the draws), `cuda_graph.py` (the decode loop),
`nar.py` (the flow stage), `modeling_yue2.py`, `modeling_vae.py`. HOT-Step CPP's port (scragnog/HOT-Step-CPP, engine/src/yue2)
and its GGUF converter (engine/tools/convert-yue2.py) give the GGUF's tensor names; its README the quantization numbers
(Q8_0: argmax 100 %, NAR rel-L2 0.019 against bf16's own 0.013).

One backbone, two sets of weights a layer (a Mixture of Transformers):

```text
  28 layers of 2048 (16 query heads over 8 key / value heads of 128, a 6144 SwiGLU, RMS norm 1e-6, q / k RMS-normed per
  head, rotate-half RoPE theta 1e6), a vocabulary of 184704: Qwen's 151643 BPE tokens, 208 specials, MUSIC_START 151851,
  MUSIC_END 151852, 32768 semantic codes from 151853, LATENT_START / END / PAD; untied head
  each layer: AR attention + AR MLP (input_layernorm, self_attn, post_attention_layernorm, mlp)
              NAR attention + NAR MLP (nar_input_layernorm, nar_self_attn, nar_pre_mlp_layernorm, nar_mlp)
  flow heads: vae2llm (64 -> 2048), llm2vae (2048 -> 64), the timestep MLP (256 sinusoid -> 2048 -> SiLU -> 2048),
              latent_pos_embed (a fixed sinusoid table [24576, 2048], stored in bf16)
```

The song, as `YuE2Pipeline.__call__` makes it:

1. **The prompt** - `[EOD] + encode("{instruction}\n[Tags]\n{style}\n[Lyrics]\n{lyrics}\n")`, the instruction the cot
   mode's (`off`, `melody`, `full`); tiktoken's `encode_ordinary` (no special token parsed out of the text) after NFC.
   - `off`: `+ [ABC_START, ABC_END, MUSIC_START]`, no score.
   - `melody` / `full`: `+ [ABC_START]`, then the AR path draws the score (ABC text tokens, < 151643, until ABC_END;
     temperature 0.7, top-p 0.9, top-k 30, repetition penalty 1.005 over the last 100, 32 to 4096 tokens), and the
     semantic phase's prompt is `... + [ABC_START] + abc + [ABC_END, MUSIC_START]`. A score given (`abc`) skips the draw:
     a cover's melody (SheetSage2's transcription), or an earlier song's score with new words.
2. **The semantic codes** - the AR path again, a code a step (32768 codes + MUSIC_END; temperature 1, top-p 0.95,
   top-k 100, repetition penalty 1.2 over the last 50, 200 to 9000 codes - 25 a second, six minutes), with
   classifier-free guidance when its scale is not 1 (default 1.0 for full / melody, **1.01 for off**: two branches by
   default there), the twin's prompt `[EOD] + encode(instruction) + ...` (the same score when there is one). Each phase
   seeds its own generator with the request's seed. `off`'s draw does its CFG arithmetic in bf16 and keeps three
   candidates from top-p (its `legacy_off`).
3. **The acoustic latents** - flow matching on the NAR path, one latent frame (64 channels, 25 Hz) a code. The chunk's
   AR tokens are `semantic prompt + codes + MUSIC_END` (one chunk up to (24576 - prompt - 3) / 2 frames: a whole song in
   practice); their K / V (the AR path's, causal - exactly the generation's cache) stay fixed for the whole solve. The
   NAR rows are `[LATENT_START, frames..., LATENT_END]` as `vae2llm(x) + time(t) + pos[local]` (x zero at the two ends),
   RoPE positions after the AR tokens, attending to every AR key and every NAR key (bidirectional), the NAR weights;
   `llm2vae(norm(h))` of the frames' rows is the velocity. 32 midpoint steps from t = 1 to 0: raw = logit(t) clamped to
   +-20, the time embedder fed sigmoid(raw) (timestep_shift 1), `mid = x - v(x, t) dt / 2`, `x -= v(mid, t - dt / 2) dt`.
   The noise is one `torch.randn((frames, 64))` on the CPU generator seeded with the seed (mt19937 + Box-Muller).
4. **The VAE** - YuE2-Vae's Oobleck decoder (SnakeBeta, weight norm folded in the GGUF; channels 64 x [1, 2, 4, 8, 16,
   32], strides 6, 5, 4, 4, 2, 2 = 1920 samples a frame; float32), tiled: cores of 512 frames with 16 either side
   (exact: the decoder's support is 12), clamped to [-1, 1].

## The files (model-cards/audio/yue2-3b-q8.yml)

scragnog/YuE2-GGUF rev eb7de09: `yue2-lm-q8_0.gguf` (3.92 GB: every matrix Q8_0, the flow heads bf16, norms f32, the
tokenizer in its metadata - no `tokenizer.ggml.pre`: the engine names Qwen2's split itself), `yue2-vae-standard-f32.gguf`
(530 MB, F32 by design). Both converted from the same weights as m-a-p/YuE2-3B rev 8f24312 (model.safetensors sha256
1d55c42c... - the GGUF's pin) and YuE2-Vae 152733a (807ce9d5...). The other files of the repo: the other quants;
`yue2-vae-legacy-f32.gguf` (the paper's benchmark decoder); `yue2-tok-f16.gguf` (a community real-audio semantic
tokenizer - MERT-v2 features through an 8-layer head to the 32768 codes - for HOT-Step's training data, not
upstream's pipeline); `mms-fa-f32.gguf` (MMS forced alignment: lyric timestamps and section markers after a render);
`sheetsage2-f16.gguf` (SheetSage2: a recording to its lead sheet as ABC - the cover's score); `minted/*` (codes minted
for HOT-Step's adapter training); `checkpoints/*` (an int8 ConvRot training checkpoint, Comfy-Org's).

## The engine (audio/yue2)

```text
  audio/yue2/src/lib.rs      the engine: the request, the phases, AudioEngine; load and the memory plan
  audio/yue2/src/lm.rs       the backbone on the card: the Q8_0 matrices as stored, the cache, prefill, decode, the NAR pass
  audio/yue2/src/prompt.rs   the prompts and the token ids (protocol.py)
  audio/yue2/src/sample.rs   the draws (sampling.py): penalty, top-k, top-p, the generator
  audio/yue2/src/vae.rs      the decoder, tiled
  audio/yue2/src/check.rs    the stages against reference/yue2/ref.py's dumps
  audio/yue2/src/ffi.rs      its kernels
  kernels/audio/yue2/yue.cu  its kernels (ns_audio_yue2_*)
```

**Kernels.** The shared diffusion kernels (nsd) do most of it: `linear_q` (a Q8_0 matrix dequantized to half or bf16 a
slice at a time, one cuBLAS GEMM: the prompt's prefill and the NAR pass, hundreds to thousands of rows), `rms_norm_mod`,
`swiglu`, `gate_add`, `attention_causal` (the prefill), `flash_attention` (the NAR pass: head 128, half, no mask - the
NAR rows over the whole cache), `conv1d` / `conv_transpose1d` (the VAE). This engine's own (`kernels/audio/yue2`): the
decode's product straight from the Q8_0 blocks for one or two rows (the shared one dequantizes the whole matrix to half
first - five bytes of traffic a weight instead of one), the embedding rows from the Q8_0 table, the per-head q / k norm
and RoPE writing q in half and k / v into the cache, the decode's attention against the half cache, the NAR rows' input
(`vae2llm(x) + bias + time + pos`, 64 inputs), the midpoint's update, SnakeBeta (`x + sin^2(x e^a) / (e^b + 1e-9)`:
ACE-Step's kernel takes the same precomputed alpha / inverse beta - used from its library).

**Precision.** Residual stream float32; the products' inputs bf16 (the reference's own type: no range risk in the MLP's
down projection), the products accumulate in float32; q, k, v and the cache in half (normed q / k; v small). The ODE
state float32 (the reference keeps it in bf16: ours is the closer to the exact solve).

**Placement.** Everything resident (no streaming). On one 12 GB card: the Q8_0 backbone 3.0 GB + embedding and head 0.8
GB, the cache (half, 112 KiB a token: a six-minute song's prompt + score + 9000 codes + 9002 NAR rows, ~21000 tokens, 2.4
GB; the twin's, when guided, prompt + 9000: ~1 GB), the NAR pass's activations (~1 GB at 9000 rows, its MLP in row
blocks), the VAE (0.5 GB + 0.8 GB for a 512-frame tile). With two cards the VAE and its tiles go on the second (as
MiniMax Music 3's flow stage does), so the first holds the language model with room for the longest song. The load
checks the plan against the free memory and refuses with the numbers.

**Speed plan.** Decode: two branches a step, 7 products a layer read from the Q8_0 blocks once (2 x 1.7 GB of AR weights a
token read once for both rows), CUDA-graph-free first; then the step's launches cut (fused norms, one graph a step) if
the launches dominate. The NAR pass: 64 passes over ~9000 rows; int8 ConvRot matrices (nsd_int8_linear, as MOSS-Image)
and a fused attention are the second pass, each measured against the Q8_0 path's stage checks before it is the default.

## The check (`nextcuda audio check DIR --model yue2-3b-q8`)

reference/yue2/ref.py runs upstream's wheel on the CPU in float32, the weights the Q8_0 file dequantized (or the bf16
checkpoint), a short song (a few hundred codes, 8 ODE steps): the stages are the same at any length.

```text
  tokens   the prompt's text and ids, every cot mode               exactly
  ar       the whole sequence teacher-forced: logits at rows of the prefill, the score, the codes   cosine, argmax
  neg      the twin's prefill logits                                 cosine
  noise    the chunk's torch.randn                                   exactly (to 1e-6)
  vel      the first velocity from the reference's noise             cosine
  steps    each midpoint step from the reference's state             cosine; the whole solve from the noise
  vae      the reference's latents to audio                          cosine
```

## Covers and new words (planned in from the start)

A cover is upstream's own path: a score given (`abc`), `cot = melody` (recommended), the new style; the recording is
never fed to the audio stage. So the engine takes `abc` (and returns the score it drew, so a song can be made again with
new words over the same melody: same seed, same score, same style, new lyrics). From a recording the score comes from
SheetSage2 (`sheetsage2-f16.gguf`: a recording to a melody lead sheet as ABC), the words from Whisper (the box's own
engine) when the caller gives none. SheetSage2 is its own model (HOT-Step's port: an encoder over audio features, a
decoder of note events, the events written as ABC) - ported after text-to-song is checked, as `audio/yue2/src/sheet*`
on the same kernels. MMS-FA (lyric timestamps) and the community audio tokenizer are not needed for either.

## SheetSage2, as ported (audio/yue2/src/sheet)

```text
  sheet/wave.rs      FFmpeg's mono 24 kHz: sqrt(1/2) (L + R) (libswresample's unnormalized float matrix), its Kaiser sinc
  sheet/mel.rs       torchaudio's log-mel on the host (float64 FFT)
  sheet/mert.rs      MERT-v2-FullSong + the layer mix + the projection on the card (kernels/audio/yue2/ss2.cu: dwconv, grn, glu)
  sheet/decoder.rs   BART's decoder: the self-attention cache, the cross keys / values once a window
  sheet/tokens.rs    the v1 vocabulary, the greedy decode's grammar, sequences <-> events
  sheet/stitch.rs    the 300 s windows, overlap prefixes, stitching
  sheet/notation.rs  exports + notation: beats, measures, Vocal / Ins voices, keys, chords -> ABC (spell.rs: key-aware spelling)
  sheet/check.rs     the stages against reference/yue2/sheetsage_ref.py (wave, mel, encoder, decoder, generate, notation, transcribe)
```

Two things of upstream's run decide its result and are reproduced: the rotary table MERT caches at its first call, which
transcribe makes under bf16 autocast (bf16 angles of bf16 positions: exact float32 angles give a memory at cosine 0.934
of upstream's), and the GGUF's values (converted from a bf16 merge: its norms, biases and layer weights are bf16-rounded;
upstream's own transcription changes with them, so the reference to compare with is `sheetsage_ref.py --bf16-params`).
Measured on Jingle Bells (USAF Band 2020, 132.8 s) and the same three times over (398 s, two windows): the wave to
2.2e-6, the mel to 3e-5, the memory 0.999995 against upstream on the file's numbers in float32; the notation from
upstream's tokens byte for byte (one and two windows, the overlap prefix identical); end to end the notes 264 of 264
(and one more) on the short one, precision 0.984 / recall 0.988 on the long one; 6.6 s on the 3060 for 132.8 s.
