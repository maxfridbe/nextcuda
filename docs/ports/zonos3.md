# ZONOS3-Mini on nextcuda: the port's record and its API

Zyphra's ZONOS3-Mini (weights Apache-2.0, `Zyphra/ZONOS3-Mini` on Hugging Face; the reference runtime
github.com/Zyphra/ZONOS3-Mini is MIT) as the audio engine `zonos3` (`audio/zonos3`, `kernels/audio/zonos3/z3.cu`), on
Zyphra's own GGUFs. Ported from the reference's three runtimes read side by side - `zonos3.reference` (PyTorch, the
numerical oracle), the ggml server (`servers/ggml/src`: zonos3-graph.cpp, zac.cpp, spk-encoder.cpp, the prompt and
sampler) and the MLX server (the most compact statement of the same math over the same GGUFs). No third-party code is
in the port; the source files carry origin notes.

## What runs

| stage | file (role) | where | how |
|---|---|---|---|
| prompt | the backbone's embedded tokenizer.json | host | SuperBPE-8192 (a byte-level BPE whose Split keeps the text between its matches: `nextcuda-tok`'s `from_hf_json_text_keep_gaps`), ids + 4 between BOS 2 and EOS 3; rows `[speaker row \| reference prefix] [loudness] [fmax] [speaking rate] text [3 silence rows]`, ten audio columns (pad 4097) and a text column (pad 8196) |
| backbone | `backbone-ptq.gguf` (model) | GPU | 11 layers of 1024, GDN layers 0-2, 4-6, 8-10 (4 heads of 128: in_proj 2056 = qkv \| z \| a \| b, causal conv 4 + SiLU, L2-normed q/k, decay `exp(-exp(A_log) softplus(a + dt_bias))`, beta `sigmoid(b)`, the delta rule in float32, per-head RMS norm x SiLU(z)); attention layers 3 and 7 (16 q / 4 kv heads of 32, q/k RMS norm, q x temp, interleaved RoPE 1e4, causal, a sigmoid gate a head); Full AttnRes: 23 sources (the normed embedding and every sublayer output), each sublayer's input the softmax-over-depth mix scored by its query against the sources' weightless RMS norms; layer 0 a SwiGLU of 2048, layers 1-10 top-2-of-64 latent MoE (router after an RMS norm, `topk(bias - logits)`, softmax of the chosen raw logits, latent 256, experts 1536 in Q4_0); 10 x 4098 heads, `15 tanh(x / 15)` |
| sampling | - | host | the registry's defaults (temperature 0.75, top-k 80, top-p 0.69, repetition 2.4 over the last 96 rows of codebooks 0-5); padding and every special id but EOA masked; one exponential-race draw a codebook (PCG32 seeded by the request's seed); the delay pattern's leading triangle padded (codebook c > the stream row, the silence rows counted: the ggml server's rule); the first EOA fixes the end frame (row - its highest codebook), 11 rows more drain |
| decoder | `zac-keep2.gguf` (decoder; `zac.gguf`, decoder-full, on request) | GPU + one product for the iDFT | semantic table + 4 causal ConvNeXt blocks + 9 residual tables, 5 MOSS stages (windowed causal attention 150/300/480, layer scales, patch x2 four times: 15 -> 240 Hz), 8 ConvNeXt blocks, two branches (n_fft 736 / 2208, hop 184: log-magnitude clamped at ln 100, phase; the inverse real DFT as one float32 product with its basis, Hann overlap-add over the squared envelope), the 127-tap FIR mixer, 44160 -> 44100 polyphase; the 3 silence frames decoded ahead and cut; -3 dB and the soft limiter above -1 dBFS |
| speaker encoder | `spk-encoder.gguf` (speaker-encoder) | host frontend, GPU network | 16 kHz by the file's polyphase kernel, Kaldi fbank (x32768, 25/10 ms, mean removed, pre-emphasis 0.97, Hamming, 512-point, 80 mels, log floor 1.19e-7, CMN) in float64; MECT-B2 (grouped 3x3 residual CNNs, flatten, projection + LN, positional depthwise conv 113, 4-head attention, a 4-expert dense-softmax FFN), the four stages fused, ASTP, BN, 192-d (not L2-normed) |
| steering | `steer.gguf` (steer) | host | raw sets move the speaker embedding (summed, rescaled to its norm); attnres_source sets add to bank source 19 (L9's attention output; mandarin_hidden 15) on every generated row |
| voices | `voice-<name>` (.codes.npy), `voice-<name>-emb` (.embedding.npy) | - | 27 presets; codes = the reference prefix (the checkpoint's default mode), embedding = the speaker row |

Not ported: raw-reference cloning of a new recording (`zac-encoder.gguf` 4 GB + `ssl-teacher.gguf` 1.5 GB: the
acoustic encoder and a Wav2Vec2-BERT teacher - a recording is cloned by its embedding instead, which the reference also
serves), text normalization (zyphra-textnorm; the server's speech text pass runs first), LACI and short-text handling,
speech editing, accent adapters, control spans, steer-then-hold (the shipped bundle has no release block), streaming.

## Checked

Against Zyphra's ggml runtime built for the CPU on the same GGUFs (`reference/zonos3/ref.sh` makes the dumps,
`reference/zonos3/compare.py` compares; `nextcuda audio check DIR --model zonos3-mini --stages prompt,lm,forced,decoder,speaker`):

| stage | result |
|---|---|
| prompt ids (speaker row, conditions, SuperBPE text, silence) | identical |
| backbone logits, 26-row embedding prompt | cosine 0.99969 mean, 0.9959 worst row, rel 2.1e-2, argmax 96.9 % |
| backbone logits, 232-row reference-prefix prompt | cosine 0.99983 mean, 0.9964 worst, rel 2.0e-2, argmax 94.2 % |
| teacher-forced, a 68-step take of the reference's | cosine 0.99984 mean, 0.9984 worst, rel 1.8e-2, argmax 92.9 %; the take's drawn codes in our top-80 99.8 % |
| decoder PCM (60 frames) | cosine 1.000000, rel 3.3e-5 |
| speaker embedding (13 s) | cosine 1.000000, rel 1.2e-6 |

The backbone's distance is the reference's own: its CPU products over Q8_0 / Q4_0 blocks quantize the activations to
8 bits (ours are float32 over the same blocks); Zyphra's notes give cosine 0.99995 and argmax 96-97 % between their
QAT backbone and its Python golden. Takes heard by Whisper large-v3-turbo: word for word in both preset modes, a clone,
steered takes, CFG 1.5, the full decoder, German; one slip in 53 s of long-form. A clone of a recording measures 0.884
against it by the speaker encoder.

## Speed (RTX 5070)

~240 rows a second (15 rows = 1 s of sound): 7.7 s of speech in 0.55 s, 53 s in 3.2 s (real-time factor 0.06-0.07).
2.1 GiB at the peak with the keep2 decoder; the full decoder 4 GiB more on first use (float32 matrices).

## The API (for another runtime: nextsycl)

The model card `zonos3-mini` (`model-cards/audio/zonos3-mini.yml`): kind audio, arch `zonos3`, speech `[voices,
clone]`, roles `model` (backbone-ptq.gguf), `decoder` (zac-keep2.gguf), `decoder-full` (zac.gguf), `speaker-encoder`
(spk-encoder.gguf), `steer` (steer.gguf), and per preset `voice-<name>` / `voice-<name>-emb` - 59 files pinned at
Zyphra's revision d2d142a8 with their SHA-256s; `nextcuda models pull zonos3-mini`.

Requests are the audio kind's (`audio.speak`, `/v1/audio/speech`); `AudioRequest` maps as:

| field | meaning |
|---|---|
| `input` / prompt | the text (any of 43 languages; no language tag: `language` is accepted and not used) |
| `voice` | a preset (savannah - the default -, wyatt, rupert, imogen, cillian, jett, morgana, sakura, thaddeus, willow, gaspard, liesel, wolfgang, ananya, arjun, ginevra, lorenzo, minjun, seoyeon, haoran, meilin, leonor, thiago, rodrigo, ximena, chaitanya, sravya; `none`: unconditioned), or a saved voice (cloned) |
| `ref_audio` (reference) | a recording to clone (any rate; its first 60 s), by its speaker embedding; no transcript |
| `cfg` | speaker guidance `uncond + cfg (cond - uncond)` against a twin without the voice and with every condition unknown (1: off) |
| `seconds` | the most to make (60 by default, 600 at most; 2000 rows a stretch) |
| `seed` | the draws |
| `options.speaker-mode` | `reference` (a preset's codes as the prefix; the default) or `embedding` (its speaker row) |
| `options.controls` | `ID=DOSE,...` steering: `delivery.tempo`, `delivery.whisper`, `mood.valence`, `emotions.happy`, `identity.gender`, `accents_en.british` ... (raw ones need an embedding) |
| `options.speaking-rate`, `loudness`, `fmax` | the condition rows' values (bucketed by the backbone's edges, bisect right; unset: unknown) |
| `options.decoder` | `keep2` (default) or `full` |
| `options.temperature`, `top-k`, `top-p`, `repetition-penalty`, `silence` | sampling and the silence prefill (0-30) |

A text over ~400 characters is said a stretch of whole sentences at a time, the same voice and draws, 0.25 s apart.
The answer is 44.1 kHz mono.
