# Voice Isolator on nextcuda: the port's record and its API

Inworld's Voice Isolator v1 (weights and the reference package `inworld-voice-isolation` Apache-2.0;
`inworld/voice-isolator` on Hugging Face, github.com/inworld-ai/voice-isolator) as the audio engine `voiceisolator`
(`audio/voiceisolator`, `kernels/audio/voiceisolator/vi.cu`), on Inworld's own safetensors. Ported from the package's
`model.py` (the network, `band_edges`, `stft` / `istft`, `correction`, `enhance`), `scan.py` (the selective scan),
`isolator.py` (`isolate`: mono, resampling, the offline passes) and `audio.py` (the kaiser_best resampler), with
torchaudio's sinc resampler read for the kernel it makes. No third-party code is in the port; the sources carry origin
notes. The offline checkpoint only (`denoiser.safetensors`, 64.0M parameters); the causal one
(`denoiser_realtime.safetensors`, streaming with a look-ahead) is not ported.

## What runs

| stage | where | how |
|---|---|---|
| input | host | mono; resampled to 48 kHz by torchaudio's `sinc_interp_kaiser` with the package's kaiser_best settings (64 zero crossings, rolloff 0.9475937, beta 14.7697): the polyphase kernel in float64 as torchaudio makes it (its phase offsets in float32), applied in float32 |
| windows | host | `enhance`: a clip up to 8 s is one pass; a longer one in windows of 8 s every 6 s, the last ending at the clip's end, linear crossfades of 2 s (`torch.linspace`'s float32 halves), the sum over the summed weights |
| STFT | host | n_fft 2048, hop 480, periodic Hann, centred (reflection; zeros under 1024 samples), 1025 bins; a radix-2 FFT in float64, frames on threads |
| encoders | GPU | 46 bands (`band_edges` of the plan (100, 1000) (250, 4000) (500, 8000) (1000, 16000) (2000, 24000) Hz, Python's round half to even): each band's (re, im) pairs through a LayerNorm and a linear to 512, plus the band embedding; tokens [bands, frames, 512] |
| blocks 0, 2, 4, 6 | GPU | along time (a band's frames): pre-norm attention, 8 heads of 64, rotary positions (pairs i, 32 + i; the tables in float32 as the reference makes them), exclusive self attention (each head's output less its component along the token's own value, `max(v.v, 1e-6)`), the GELU MLP of 2048; then across a frame's 46 bands: MultiheadAttention's in-projection, plain attention (vi.cu `short_attention`: the table in shared memory), the MLP |
| blocks 1, 3, 5 | GPU | bidirectional Mamba along time, then across the bands: in-projection to 2 x 1024 (u, z), causal depthwise conv of 4 + SiLU (the backward scan reads the sequence from its end), x_proj's rank-32 rows and B, C (16 states each), delta = softplus(dt_proj), the selective scan in float32 (a thread a sequence and channel, its 16 states in registers: decay, drive, read out, skip, SiLU(z) gate), out-projection added onto the residual copies, output_proj of [fwd + x, bwd + x], LayerNorm, added |
| heads | GPU | the decoder's conditioning at its served controls (1, 0): x (1 + scale) + shift; per band LayerNorm, linear 512, GELU, linear to 4 a bin (mask re, im, residual re, im) |
| correction | host | mask = 2 tanh(raw[0..2]); residual = tanh(raw[2..4]) over max(its norm, 1) x 0.5 |bin|; the change = spectrum x mask + residual |
| output | host | the window + the iSTFT of the change (irfft dropping DC and Nyquist imaginaries, Hann overlap-add over the squared envelope, `length`); the windows crossfaded; back to the recording's rate (kaiser_best) and its length; the rest = the recording less the voice |

Everything on the card is float32 (the checkpoint's own weights; cuBLAS products without TF32, nsd's layer norm, GELU and
batched time attention); the package itself serves bfloat16 autocast on a CUDA card.

## Checked

Against the package on the workstation's CPU in float32 (`reference/voiceisolator/ref.py` makes the dumps; `nextcuda
audio check DIR --model voice-isolator --stages spec,net,chunk,clips`), each stage from the reference's own input. The
clips: the package's own demo samples (noisy speech: sample_4 7.5 s, sample_1 14.0 s, sample_2 24.7 s, sample_7 4.0 s,
sample_14 resampled to 44.1 kHz) and a clean LibriSpeech utterance at 16 kHz (2035-147960-0000, from the same demo).

| stage | rel | max abs | SNR |
|---|---|---|---|
| STFT (7.5 s) | 1.6e-7 | 1.6e-5 | 136 dB |
| band tokens | 1.2e-7 | 9.5e-7 | 138 dB |
| blocks 0-6 (each) | 5.7e-7 - 9.5e-7 | 4e-5 (block 6: 1.3e-2 on values ~1e4) | 120-125 dB |
| heads' output | 3.1e-7 | 4.8e-6 | 130 dB |
| a window's audio | 9.5e-6 | 1.3e-5 | 100 dB |
| sample_4, 7.5 s, 48 kHz (noisy) | 9.5e-6 | 1.3e-5 | 100 dB |
| sample_1, 14.0 s, 48 kHz (noisy; 2 windows) | 1.2e-6 | 7.4e-7 | 118 dB |
| sample_2, 24.7 s, 48 kHz (noisy; 4 windows) | 1.7e-7 | 1.8e-7 | 135 dB |
| sample_7, 4.0 s, 48 kHz (noisy) | 3.7e-7 | 1.6e-7 | 129 dB |
| sample_14 at 44.1 kHz, 10.8 s (noisy; resampled both ways) | 2.4e-5 | 2.9e-5 | 92 dB |
| LibriSpeech at 16 kHz, 9.0 s (clean; resampled both ways) | 1.3e-4 | 1.8e-4 | 77 dB |

The same on the RTX 5070 and the RTX 3060. The resampled clips' distance is the resampler's float32 sums (137 to 283
taps a phase, summed in another order than torch's conv1d).

The package as it serves on a CUDA card (`precision` auto: the network under bfloat16 autocast; run here under the
CPU's autocast) is further from its own float32 than this port is: SNR 51-62 dB against float32 on the same six clips
(rel 7.6e-4 to 2.8e-3), so this port and the served package differ by the same 51-62 dB.

## Speed and memory

Measured with `audio check --stages clips` (the whole request: resampling, STFT, network, iSTFT, crossfade), the box
otherwise idle:

| clip | RTX 5070 | RTX 3060 |
|---|---|---|
| 4.0 s (one window) | 0.22 s (RTF 0.054) | 0.52 s (0.128) |
| 7.5 s (one window) | 0.39-0.46 s (0.052-0.061) | 0.95-0.99 s (0.13) |
| 14.0 s (two windows) | 1.27 s (0.091) | 3.1 s (0.22) |
| 24.7 s (four windows) | 1.72 s (0.070) | 4.2 s (0.17) |
| 10.8 s at 44.1 kHz | 0.89 s (0.082) | 2.1 s (0.20) |
| 9.0 s at 16 kHz | 0.87 s (0.097) | 2.1 s (0.23) |

An 8 s window's network is ~0.33 s on the 5070 (nsys: the float32 products about half of it, cuBLAS's sgemm; the
time attention, the scans, the band attention and the norms the rest). Memory: 0.24 GiB of weights, 1.96 GiB at the
card's peak (the CUDA context included) for any length - the work buffers are sized for one 8 s window at load.
What would make it faster, not done: TF32 or bfloat16 products (the package serves bfloat16; a declared option), the
STFT on the card.

## The API (for another runtime: nextsycl)

The model card `voice-isolator` (`model-cards/audio/voice-isolator.yml`): kind audio, arch `voiceisolator`, one role
`model` (denoiser.safetensors at Inworld's revision 9b0285b3, 256260128 bytes, SHA-256 6473...ae53; its architecture is
the file's `config` metadata); `nextcuda models pull voice-isolator`. Capability `voice_isolation`, method `audio.isolate`.

`audio.isolate` (a job like audio.speak; `/v1/audio/isolate` on the audio server):

| field | meaning |
|---|---|
| `audio` | the recording: WAV, MP3, FLAC, Ogg, M4A as base64, a data URL, or `upload:<name>`; any rate, mixed to mono |
| `model` | a voice isolator's id; none: the loaded or first enabled one |
| `rate` | the answer's samples a second (8000-192000); none: the recording's |
| `response_format` | `url` (default) or `wav` (the voice's file alone) |

The answer: `{created, data: [the voice], background: the rest, rate, run: {took, seconds}}`, each a sound item with its
`url`; float32 WAVs (format 3), as long as the recording at the answer's rate. The audio contract gains
`AudioEngine::isolates` / `isolate(IsolateRequest) -> Isolated {voice, background}`.
