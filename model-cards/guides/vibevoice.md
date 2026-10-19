# Prompting VibeVoice 1.5B (multi-speaker speech, 24 kHz)

Long-form conversation in up to four voices, each cloned from a recording - a podcast, an interview, a scene read by
its cast. No built-in voices, no written direction: the voice is the recording's, the delivery the model's reading of
the words. English and Chinese.

| what you send | what you get |
|---|---|
| `input` + `ref_audio` (or a saved `voice`) | the text in that recording's voice (one speaker) |
| `input` as `Speaker N: words` lines + `speakers` | a conversation: Speaker 1 in `speakers[0]`'s voice, Speaker 2 in `speakers[1]`'s ... |

`speakers` takes saved voices' names, recordings (base64, a data: URL) or `upload:<name>` (audio.upload). A line that is
not a `Speaker N:` line belongs to nobody and is dropped; plain text with no such line is read by Speaker 1.

```json
{"model": "vibevoice-1.5b", "speakers": ["skippy", "sttng"],
 "input": "Speaker 1: Did you hear that?\nSpeaker 2: Hear what? The engines are quiet.\nSpeaker 1: Exactly."}
```

## The recordings
3-15 seconds of one person speaking, clean and close; every second of each is read again on every request (7.5
frames a second of prompt), so longer is slower and rarely better. The recording's level is normalised (-25 dB FS).

## Settings
`steps` (10; the diffusion head's per frame: more is finer and slower), `cfg` (1.3; much higher and the voices shout),
`seed` (the head's noise and the clones' draw), `seconds` (a bound on the take; by default it follows the text).
