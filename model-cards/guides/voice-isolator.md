# Voice Isolator: a recording's voice, and what was taken from it

`audio.isolate` keeps a recording's foreground voice and takes away everything else - noise, room tone, music, other
sounds. The answer is two files: the **voice**, and the **rest** (the recording less the voice: exactly what was taken
away; the two added together give the recording back).

```json
{"audio": "upload:up_1792400000_take2.wav"}
```

- **audio**: WAV, MP3, FLAC, Ogg or M4A, at any rate (mixed to mono) - base64, a data URL, or `upload:<name>` after
  `audio.upload`.
- **rate**: the answer's samples a second; leave it out for the recording's own. The model works at 48 kHz: other rates
  are resampled to it and back (torchaudio's kaiser_best, as Inworld's package does).
- **response_format**: `url` (the default: JSON, the voice in `data`, the rest in `background`) or `wav` (the voice's
  file alone).

What to expect:

- Both files are 32-bit float WAVs as long as the recording, at its level: nothing is normalized, and a voice that peaks
  over full scale (it happens on loud inputs) is kept as it is.
- Long recordings go in 8 s windows crossfaded over 2 s; each window sees all of itself (the offline checkpoint).
- It keeps the *foreground* voice. Two people talking at once are both voice; a singer over a band is voice, the band is
  the rest. Speech far in the background may be taken away with the noise.
- Use it before cloning a voice from a noisy recording: the speech models copy what they hear, room and hiss included.
