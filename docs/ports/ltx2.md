# Porting LTX-2.5 (Lightricks) to nextcuda

LTX-2.5 makes a clip with its own sound in one pass: a 22B audio-video DiT, Gemma-4 12B as the text encoder, a video
VAE, an audio VAE and a 48 kHz vocoder, and a latent upsampler for a second pass that doubles a finished clip. This is
the record of the port: what it is, how it sits on the two 12 GB cards, what was measured against the Python engine of
the creative platform (`cp_worker/kinds/movie/engines/ltx2.py`, the reference), what differs from it, and what it
cannot do. Rust and CUDA, no Python at run time, nothing streamed (a stage is loaded whole, used, and freed).

The license is Lightricks' LTX-2.x Community License: free below US$10 million of annual revenue, a paid license above
it. Read [the license](#license) before offering this to anyone.

## The pieces

```text
  text        Gemma-4 12B (48 layers, 3840 wide, local / global attention every sixth layer), Q4_K, 49 hidden states
              -> two connectors (8-layer transformers, 188,160-wide projections), Q8_0: video 4096 x 1024, audio 2048 x 1024
  denoiser    the AV DiT: 48 blocks, video stream 4096 wide, audio stream 2048, each attending to the other
              (a2v / v2a), gated attention, prompt modulation; Q4_K_M GGUF (dev 15.7 GB, distilled 14.2 GB)
  sampler     flow-match Euler, sigmas linspace(1, 1/N, N) then 0 (dev, 30 steps) or the distilled DiT's own eight;
              four passes a step (the prompt, the empty prompt, STG on block 28, modality isolation) combined per
              modality in x0 space (video 3 / 1 / 3, audio 7 / 1 / 3, rescale 0.7)
  video VAE   decoder (x32 spatial, x8 temporal), tiled; the encoder for a first frame (image to video) and for the upscaler
  audio       audio VAE decoder -> mel; BWE vocoder -> 48 kHz stereo
  upscaler    the latent upsampler (x2): a finished clip back through the encoder (tiles of 512 px, stride 448), its
              latents doubled in space, decoded at twice the size, the sound copied
```

Source: `video/ltx2/src` (`gen.rs` a clip, `ups.rs` the upscaler, `dit.rs` / `sample.rs` the denoiser and its loop,
`te.rs` / `conn.rs` / `cache.rs` the text stage, `vae.rs` / `tiles.rs` / `audio.rs` the decoders, `rules.rs` what a job
may ask for, `config.rs` the GGUF's own config), `kernels/video/ltx2` (its few kernels beside the shared diffusion ones),
`reference/ltx2` (the float32 CPU reference scripts the checks compare against), `model-cards/video/ltx-2.5*.yml`.

## How it sits on the two cards

A clip needs more than a card holds, so a stage is loaded, used and let go - never streamed:

1. **The DiT is resident** from the first clip: its 48 blocks split over both cards in proportion to their speed
   (compute units x clock), each card at most what fits beside 2.5 GiB of working memory: **32 blocks (7.79 GiB) on
   the RTX 5070, 16 (3.97 GiB) on the RTX 3060**. The four passes of a step are pipelined over the two cards (the 5070
   starts the next pass while the 3060 finishes one). `LTX2_SPLIT=a,b` overrides the split.
2. **The text stage** (Gemma-4 in Q4_K, 5.7 GiB over both cards, beside the resident DiT) encodes the prompt and the
   negative prompt and is freed; each connector (video, audio) is then loaded on the freest card, run and freed. About
   30 s for a prompt.
3. **The decoders** (video VAE 1.4 GiB, audio VAE, vocoder) load on the freest card after the denoise, one at a time.
4. The first load quantizes Gemma-4 and the connectors from the checkpoint's bfloat16 into `nextcuda-text.gguf` (11.5 GB,
   Q4_K / Q8_0) beside the components, once, on the host (a few minutes); the engine reads it after that.

Peak memory (`nvidia-smi`, the whole process): 10.7 GiB on the 5070 and 9.2 on the 3060 for 640 x 384 x 121 frames;
11.0 and 9.2 for 1024 x 576 x 121. The denoiser's working memory barely grows with the clip (flash attention), the
time does. A different clip shape reuses the resident DiT; the dev and distilled DiTs are swapped (a reload) when a
request changes variant.

## What it makes

| key | |
|---|---|
| canvas | width, height: multiples of 32 |
| length | any frame count from 9, at 24 fps; the model makes the next 8k + 1 and the clip is cut back to what was asked |
| size limit | the DiT's tokens, width/32 x height/32 x ((frames - 1)/8 + 1), at most 9,216 (1024 x 576 for 121 frames, the largest run on the pair) |
| first frame | one picture (the clip starts on it): the picture is re-compressed at H.264 CRF 18 as the pipeline does, encoded by the video VAE, and held at timestep 0 |
| negative prompt | read: it is the empty-prompt branch's text |
| sound | the model's own, 48 kHz stereo AAC in the mp4, from the prompt |
| not here | guide clips, continuing from a sound, end frames, reference pictures or sounds, a soundtrack, LoRAs, other samplers |

### The request: what a client sends

The platform addresses LTX through the same methods as H3: `video.engines.list` to see what a box has,
`video.generate` to submit (by `label` / `reentrant_id`), `video.status` / `video.wait` to follow, `video.cancel`,
the clip at `/out/<name>.mp4` (`GET /v1/video/files/<name>` on the front door), `video.upload` for pictures. All
outcomes are in the body (`ok`, `error.code`, `error.retryable`), never the HTTP status.

**`video.engines.list`** answers one row per engine; the LTX rows (the H3 rows keep their shape and gain `engine:
"minimax_h3"` and `stage: "shots"`):

```json
{"quant": "ltx2", "engine": "ltx2", "stage": "shots", "file": "ltx-2.5-22b-dev-transformer-Q4_K_M.gguf", "path": "...", "gib": 14.61,
 "ready": true, "default": false, "models": ["ltx-2.5", "ltx-2.5-fast"], "license": "ltx-2-community", "commercial": true, "license_note": "...",
 "inputs": {"takes": ["prompt", "negative_prompt", "frames", "seconds", "steps", "seed", "width", "height", "variant", "first_frame", "label",
                      "reentrant_id", "project", "batch", "clip", "camera", "priority", "queue"],
            "not": ["guide_clip", "first_audio", "first_audio_s", "last_frame", "exposure_ref", "soundtrack", "soundtrack_start", "ref_audios",
                    "ref_images", "ref_videos", "ref_image_size", "ref_video_sound", "te_pictures", "control_video", "control_mask", "control_source",
                    "source", "regen", "regen_box", "shift_video", "shift_audio", "cond_noise_aug", "loras", "sampler", "schedule", "upscale", "upscaler"],
            "chosen": "engine: ltx2", "ignored": ["cfg", "te", "chain.pin_renorm"],
            "options": {"variant": {"values": ["dev", "distilled"], "default": "dev"},
                        "steps": {"min": 1, "max": 40, "default": {"dev": 30, "distilled": 8}},
                        "frames": {"min": 9, "max": 361, "step": 1, "default": 121},
                        "width": {"min": 256, "max": 1344, "multiple_of": 32, "default": 640}, "height": {"min": 256, "max": 1344, "multiple_of": 32, "default": 384},
                        "tokens": {"max": 9216}, "seed": {"min": -2147483648, "max": 2147483648, "default": 0}, "first_frame": {"help": "..."}},
            "fps": 24, "audio": "the mp4 carries the model's own stereo sound at 48 kHz, made from the prompt"}}
{"quant": "ltx2_upscale", "engine": "ltx2_upscale", "stage": "upscale", "file": "ltx-2.5 latent upsampler + video VAE", "gib": 2.4, "ready": true,
 "inputs": {"takes": ["source_clip", "label", "reentrant_id", "project", "batch", "clip", "camera", "priority", "queue"], "not": ["prompt", "frames", ...],
            "options": {"scale": {"values": [2]}, "source_clip": {"help": "..."}}, "stage": "upscale"}}
```

A box with only the `ltx-2.5-fast` model lists `"variant": {"values": ["distilled"]}`; a box with no LTX model lists no
LTX rows. `engine` in `video.generate` takes the `quant` (`ltx2`, `ltx2_upscale`; `ltx-2.5` and `ltx-2.5-fast` read as
`ltx2`, the latter does not choose the variant).

**A shot** - `video.generate`:

```json
{"engine": "ltx2", "prompt": "<one paragraph of prose: what is seen and heard>", "frames": 121, "width": 640, "height": 384, "seed": 7,
 "variant": "dev", "steps": 30, "label": "<unit>", "reentrant_id": "<unit>", "queue": false}
```

| key | |
|---|---|
| `engine` | `ltx2` (required for LTX: no engine means H3's default) |
| `prompt` | required, one paragraph (see `describePromptGuideMD`: ltx-2.5) |
| `frames` | 9 to 361, any count, exactly this many come back; or `seconds` (x 24, rounded; default 5 = 120 frames) |
| `width`, `height` | multiples of 32, 256 to 1344, default 640 x 384; width/32 x height/32 x ((frames - 1)/8 + 1) at most 9,216 |
| `variant` | `dev` (default, 30 guided steps) or `distilled` (8 steps, its own sigmas) |
| `steps` | 1 to 40; default 30 (dev) or 8 (distilled: with no `steps` it runs the distilled DiT's own eight sigmas; naming `steps` runs the ordinary schedule) |
| `seed` | integer, default 0; the noise is this engine's own, so a seed repeats here, not across engines |
| `negative_prompt` | optional text for the guidance's empty-prompt branch |
| `label`, `reentrant_id`, `project`, `batch`, `clip`, `priority`, `queue` | as for H3 |

**A first-frame shot**: the same, with `"first_frame": "<name>"` - a png, jpg or webp in the clips directory (the
platform sends it with `video.upload`, which answers the stored name). It is stretched to the canvas. An anchor word
(`prev`, `first`) or an mp4 is refused, and so is a name that is not there (`not_found`).

**An upscale** - `video.generate`:

```json
{"engine": "ltx2_upscale", "prompt": "", "source_clip": "<a clip's name or name.mp4 in the clips directory>", "label": "<unit>", "reentrant_id": "<unit>"}
```

`prompt` is the empty string (the field is required by the method's shape); `frames`, `seconds`, `steps`, `seed`,
`width`, `height`, `first_frame` and `variant` are refused. The clip must be an mp4 whose sides are multiples of 32;
the result is a **new clip** (its own name, an h3_YYYYmmdd_HHMMSS like any other), twice the size, with the source's
sound copied. A source runs in windows of 121 frames; a tail of fewer than nine frames is dropped, and every window is
cut to an 8k + 1 count, so **a 120-frame clip comes back as 113 frames** (the answer's `frames` says; a clip of 8k + 1
frames - 121, 241 - loses none).

What a request that H3 would take and LTX does not is refused with `invalid_param` naming the key (`guide_clip:
not a key ltx2 takes`), not accepted and ignored. `cfg`, `te` and `chain.pin_renorm` are accepted and listed in the
answer's `ignored`.

**Following a clip**: the answer to `video.generate` is `{queued, position | job, frames, seconds, ignored,
reentered}`; `video.status` reports `stage` ("text encoder", "conditioning done", "sampling 14/30", "sampling done,
decoding", "audio decoded, muxing"; upscale: "upscaling 60/120 frames"), `pct`, `steps_done` / `steps_total`, `eta`,
`failure {code, retryable, message, hint}` and, when done, `file: /out/<name>.mp4`. `video.cancel {label}` stops a
running clip at its next step or drops a queued one. A repeat of a `reentrant_id` is answered (`reentered: true`,
`state` queued / running / done, the clip's `name` and `file`) instead of making a second clip; a failed clip is made
again. Each finished clip has its record (`video.jobs.get`) with the stage times (`text encoder + connectors`,
`sampled N steps`, `video + audio decoded`, `TOTAL`).

## Checked against the reference

The reference is the official diffusers code on the CPU in float32 (`reference/ltx2/ref_*.py`, run by `nextcuda video job
check --dir ...` stage by stage); the engine runs the same GGUF, so the checks measure our arithmetic, not the
quantization. Relative L2 error unless noted:

| stage | against the reference |
|---|---|
| tokenizer | identical tokens |
| Gemma-4 states (49) | Q8_0 <= 1.9 %, Q4_K 2.9 % (the quantization's own) |
| connectors | 0.4 %; the text stage together (Q4_K) 2.7 % |
| DiT velocities, 4 passes of a step | <= 6.4e-4 |
| three guided steps' latents | 1.4 % |
| video VAE decode | 5.8e-4; tiled at 640 x 384 x 121: 7.7e-4 |
| audio VAE (mel) / vocoder (48 kHz wave) | 3.6e-4 / 2.3e-5 |
| VAE encoder (first frame latents) | 2.2e-3 (float32 residual path) |
| upscaler (tiled encode, latent upsampler, tiled decode) | <= 2.5e-3 |

The unit tests (`cargo test -p nextcuda-video-ltx2`, 57, no GPU) hold what can drift without a GPU: the sigma
schedules and the guidance's delta form and rescale, the frame / canvas / token rules, the config flags read from the
GGUF's metadata (below), the RoPE tables, the block split over the cards, the VAE tile stitching, the upscaler's
windows, the quantizers, the `.npy` reader the checks use, the safetensors shards, ffmpeg's mp4 round trip.

### The config, from the GGUF

diffusers' LTX-2 code is LTX-2.0's; 2.5 differs in seven config flags and one key prefix. The GGUFs carry Lightricks'
own config in their metadata; `config.rs` ports the Python engine's `ltx2_config.py` and checks it when the DiT loads, so
an LTX-2.0 file fails first with the flag named instead of deep in the load: `apply_gated_attention` -> `gated_attn` /
`audio_gated_attn`, `cross_attention_adaln` -> `cross_attn_mod` / `audio_cross_attn_mod`, `use_keyframes_abs_pos_embedding`
(all true), `qk_norm` `rms_norm` -> `rms_norm_across_heads`, `ff_bias` false, `use_prompt_embeddings` false,
`use_prompt_adaln_single` true; the prefix `prompt_adaln_single.` is diffusers' `prompt_adaln.`. The derived config equals
Lightricks' official `LTX-2.5-Diffusers` transformer config on every key both have (a test holds both).

## Speed and memory

RTX 5070 (blocks 0..32) + RTX 3060 (blocks 32..48), 640 x 384, 121 frames, 24 fps (5.04 s), seed 7:

| run | denoise | per step | whole clip |
|---|---|---|---|
| dev, 12 steps | 153 s | 12.8 s | 204 s |
| dev, 30 steps (the take) | 384 s | 12.8 s | 435 s |
| distilled, 8 steps, unguided (the default) | 37 s | 4.6 s | 83 s |
| distilled, 8 steps, guided (as before 2026-10-29) | 93 s | 11.6 s | 146 s |
| dev, 768 x 512, 121 frames (6,144 tokens), 1 step | 22.9 s | 22.9 s | 89 s (1 step) |
| dev, 1024 x 576, 121 frames (9,216 tokens), 1 step | 39.4 s | 39.4 s | 122 s (1 step) |
| upscale of the 5 s clip (121 frames) to 1280 x 768 | | | 34 s |

The whole clip adds about 30 s for the text stage, 10 to 50 s for the decoders (more with the canvas), and nothing
for the DiT after the first clip (resident); the first clip of a process also loads the DiT (13 s from the page cache,
minutes from a cold disk) and the first ever builds the text cache.

**Against the Python engine** (its docstring: 12 steps, about 15 s each, 5 s at 640 x 384, on the 5070 alone, the
GGUF streamed from host memory): the same 12 steps take **12.8 s each here, 17 % less**, with the DiT resident over both
cards, and a 30-step take 385 s against its 465 s. Where the time goes (`LTX2_PROFILE=1`, a step = 4 passes): the 5070
spends 7.9 s of a step on its 32 blocks, the 3060 10.2 s on its 16 (0.64 s a block against 0.25), self-attention 40 %,
the feed-forward 31 %, text cross-attention 18 %, the audio-video cross-attention 9 %; the pipeline's fill adds the rest (a step is 4 x the slower card's
pass plus one of the faster's). A split of 35 / 13 would balance them (10.7 s a step, -16 %) but the 5070 has no room
for 35 blocks beside the 2.5 GiB the split keeps for activations at the larger canvases (11.0 GiB used of 12.2 at
1024 x 576); a shape-aware split (a reload when the shape changes) is the known next step.

## What it looks and sounds like

Clips made on the box (640 x 384, 121 frames, 24 fps; frames from the mp4s, the sound measured: RMS and peak in dBFS,
the spectrum's centroid, the frequency under which 95 % of the energy lies, the share of the energy below 1 kHz and between
1 and 4 kHz; the sound as measured, not listened to):

| clip | what | picture | sound |
|---|---|---|---|
| k30 | dev, 30 steps, a woman cooking who turns and says a line | clean, natural, film-like, consistent (a camera on a tripod appears in the frame: the prompt asked for a "locked-off tripod") | speech at 4.0 to 4.8 s, RMS -26 / peak -6 dBFS, 59 % below 1 kHz, 95 % under 2.8 kHz, no clipping; the rest digital silence (-80 to -90 dBFS): the requested pan simmer is missing |
| d8u | distilled, 8 steps, unguided, the same prompt | natural, warm, a little softer than the dev take; no artifacts | speech at 2.0 to 3.0 s, RMS -22 / peak -2 dBFS, 73 % below 1 kHz, no clipping; silence between (-72 dBFS) |
| d8 | the same, guided (the first port's way) | over-saturated, over-sharp, plastic skin, smeared steam | clipped (209 samples at 0 dBFS), harsh; silence between |
| s12 | dev, 12 steps | hazy, low contrast, amber cast: too few steps | one 0.8 s line at -34 dBFS, RMS -46, the rest -69 dBFS |
| g2 | dev, 30 steps, a fox in snow (the sheet `~/ltx-dev/out/g2_sheet.png`) | good: a coherent fox across 5 s, kicked-up snow, backlit dawn | RMS -41 / peak -24 dBFS: quiet, 94 % of the energy between 1 and 4 kHz, 95 % under 3.2 kHz: thin |
| i1 | dev, a first frame (image to video) | follows the picture | RMS -66 dBFS, peak -50: effectively silent |

The picture is good: the dev take at 30 steps is the quality bar and the unguided distilled one is close; below about 20
steps the dev picture goes hazy. The sound is the weak part and it is a draw: of four dev takes one had a full speech
level, one was thin and quiet, one nearly silent, one (12 steps) quiet; none had the ambience the prompt asked for
(the simmer, wind). Speech, when it comes, is in range for the model's band (95 % of the energy under 3 kHz) and
unclipped. The dev checkpoint was chosen to fix tinny sound; on these samples it does not make full, steady sound
every time: expect to normalize (peak -24 dBFS is common) and to retry a seed when it is thin.

## What differs from the Python engine

- **Both cards and the dev DiT resident**; the Python engine keeps one card (it found a second one unreliable) and
  streams the blocks that do not fit.
- **The noise is this engine's own** (a seeded splitmix64 + Box-Muller, not torch's generator): a seed repeats here, not
  across engines. The stages agree with the reference given the same noise.
- **The negative prompt is used** (the guidance's empty-prompt branch takes it); the Python engine says it is not wired
  through and ignores it.
- **Frame counts**: any count from 9 (made as 8k + 1, cut back); the platform's 17k + 5 counts are fine.
- **The upscaler** is the same model and tiles; it holds the whole clip's pixels on the host, windows of 121 frames,
  and writes through ffmpeg at CRF 16 (as Python does). The first port, like the Python engine, ended its mp4 with
  `-shortest`, which stopped the encoder before it had flushed: a 121-frame clip came back as 118; the sound is now
  cut to the pictures' length instead (`-t`) and every frame is kept (a test holds it).
- **The distilled variant runs unguided** (one pass a step, no combination: the DiT is trained without guidance). The
  port first ran it with the dev DiT's four guided passes, as the Python engine seems to (its distilled step costs as much
  as a dev one): the picture came out over-saturated and over-sharp and the sound clipped (peak 0 dBFS, 209 clipped
  samples); unguided it is natural, unclipped and 2.5 times faster. `guided` (0|1) on the job sets it either way.
- **Audio** is not rebalanced or normalized (see the limits).
- **Text quantization**: Q4_K by llama.cpp's search (the Python engine uses bitsandbytes NF4); the states differ from
  the bf16 reference by 2.9 % (Q4_K) as above.
- **Error shapes**: refusals are `invalid_param` in the body, naming the key; out-of-memory is `out_of_memory`,
  `retryable: false`.

## Known limits

- **No preview picture** while it samples (`preview` is null): H3 decodes the latents' middle frame each step; the
  video VAE is not kept loaded beside the DiT here.
- **No chaining**: no guide clip, no sound continuation, no end frame. A longer clip is one clip (361 frames at most; the
  token limit allows 297 frames, 12.4 s, at 640 x 384 - only 121 frames were run), not a chain.
- **The sound is the model's draw**: it ranges from a full, clean voice to a thin hiss or silence between lines for the same
  prompt (below), and the dev DiT's output is often quiet (peaks of -24 dBFS). It is not normalized, rebalanced or retried
  here; the Python engine measured the low-frequency share and flagged "thin" takes (`audio_balance`), this does not.
- **The 3060 is the bottleneck** of a step (above).
- **The first clip of a process is slow** (DiT load; text cache on the very first).
- **One clip at a time**: the engine takes a lock per job; the studio queues the rest.
- **ffmpeg is a run-time dependency** (mp4 in and out); there is no Python.

## License

The weights are under the **LTX-2.x Community License Agreement** (Lightricks Ltd., license date August 11, 2026). Its
whole text is embedded in the metadata of both DiT GGUFs (the first 200 KB of the file; no LICENSE file ships beside
the weights) and was read in full for the cards (`license: ltx-2-community`, `commercial: true`, the conditions in
`license_note`). What it says, with its numbers:

- **Free for any purpose, commercial included, below US$10,000,000 of annual revenue** (section 2.1): "Entities with
  annual revenues of at least $10,000,000 (the 'Commercial Entities') are required to obtain a paid license for any use
  (excluding use solely for a Non-Commercial Purpose ...) of LTX-2.x and Derivatives of LTX-2.x". Revenue is counted for
  the **Entity with all its subsidiaries, affiliates and companies under common control, together** (section 1.6). The
  paid license is a "Commercial Use Agreement" from ltxv-licensing@lightricks.com; use without it is a material breach
  and the entity owes the fees for the period, within 30 days of demand.
- **A Commercial Entity's free use** is only a Non-Commercial Purpose (2.2): an individual's personal research, study,
  hobby or entertainment not connected to any business or employer, or testing, evaluation and non-commercial R&D "in a
  non-production or development environment" - and only while it receives no direct or indirect payment from the use. Use
  "for revenue-generating activity in any manner", "in direct interactions with or that has impact on end users", or to
  train, fine-tune or distill any model for commercial use needs the paid license.
- **Hosting for others is allowed** (section 3: "remote access purposes (e.g. software-as-a-service)") on conditions: the
  use restrictions (section 4, Attachment A) must be an enforceable term of whatever governs your users' use, with notice
  to them; recipients get a copy of the agreement; modified files carry a notice; Derivatives (fine-tunes, LoRAs, models
  trained on outputs) are distributed only under this agreement, and a Commercial Entity receiving one needs its own paid
  license.
- **Outputs** (section 5): Lightricks claims no rights in them; you are accountable for inputs, outputs and their uses.
- **Transparency and updates** (section 6): the licensor may restrict or update the model remotely; you must not remove,
  disable or circumvent any "safety or security measures, disclosures, metadata, watermarking, content provenance, latent
  disclosure" in the model or on an output, must keep them in any service through which outputs are made available, and
  are responsible for the EU AI Act and California AI Transparency Act duties (labelling content as machine-made); the
  licence is revocable at once if the model is changed so it can no longer carry such a disclosure. Section 6 also
  carries indemnities (AI-regulation breaches, export controls).
- **Attachment A** (use restrictions, plus the Acceptable Use Policy at static.lightricks.com/legal/ltx-acceptable-use-policy.pdf,
  which Lightricks may update and which governs as of the time of use): no unlawful use, no exploiting or harming minors,
  no knowingly false content meant to deceive, no personal identifiable information to harm, no machine-made content
  placed in a context without saying it is machine generated, no impersonation (deepfakes) without consent, no fully
  automated decisions affecting legal rights, no discrimination, no medical advice, no law-enforcement or immigration
  profiling, no malware, no military / weapons / nuclear use, **"for commercial use only: to train, improve, or fine-tune
  any other machine learning model ... or competing model"**, no circumventing the safety / watermark features, and
  **no product, service or application "that directly competes with Licensor's commercial products or services, or is
  designed to replace or substitute Licensor's offerings in the market" without a separate commercial license** (item 20).
- Termination (section 13) on any breach, automatically on unauthorized commercial use or an AI-regulation breach, and
  on suing Lightricks over IP in the model or its outputs; then the weights and derivatives must be deleted. New York
  law, ICC arbitration in New York, a jury and class-action waiver for non-consumers (sections 12 to 15).

**For this box and the platform**: a person or a company under US$10 million of annual revenue (counting affiliates) may
use it commercially, subject to Attachment A and the transparency duties; above that, a paid license is needed for
anything but private or non-production use. **A human has to confirm** (the reading is mine, not legal advice): (a)
whether the operator of the platform counts as one Entity with its customers (the agreement speaks of "you" and of
hosting for third parties: if third parties use the model through the platform, their own revenue and the "use
restrictions pass-through" apply to them); (b) whether the item 20 competition clause reaches a creative platform that
sells video generation; (c) what "latent disclosure ... included or embedded within LTX-2.x" means for this engine, which
runs only its own port of the sampler and decoders and **adds no watermark or provenance metadata** to the mp4 (the
diffusers pipeline it was ported from may add some; this was not checked); (d) the revenue threshold's measuring period
("annual revenues" is not defined further). `commercial` is `true` in the cards because the license allows commercial use
below the threshold; the `license_note` says the rest.
