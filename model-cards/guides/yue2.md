# Prompting YuE2 (songs with vocals)

YuE2-3B is M-A-P's song model, **non-commercial (CC BY-NC 4.0)**: attribution, no commercial use (a commercial
licence comes from the YuE2 team). Two texts make a song - a **style** (`instructions`) and the **lyrics** (`input`) -
and it writes a score of the melody (and chords) before the music, so the words land on a tune.

## The style
A comma-separated list of tags: genre, mood, tempo feel, the voice, the instruments, the production. Upstream's own:

`City Pop, upbeat, danceable, groovy bass, electric guitar, synth, energetic, joyful, neon city night`
`Indie folk, warm male vocal, acoustic guitar, soft drums, melancholic, heartfelt`
`Jazz, expressive lead vocal, piano, tenor saxophone, upright bass, brushed drums, no guitar, spacious modern harmony`

Name the voice (male / female vocal, its character) and leave out what should not be there ("no guitar").

## The lyrics
- Section tags on lines of their own: `[Intro]`, `[Verse]`, `[Pre-Chorus]`, `[Chorus]`, `[Bridge]`, `[Interlude]`,
  `[Outro]`; a blank line between sections. An empty `[Intro]` or `[Interlude]` is an instrumental passage.
- One sung line per line; repeat a chorus in full where it comes back (the model sings what is written).
- English and Mandarin are what it was made for (upstream's demos: English rock, metal, folk; Mandarin pop, funk); write
  Mandarin in characters, phrases separated by spaces.

```
[Verse]
I walked along the river when the city lights were low
Every window held a story that I used to know

[Chorus]
Carry me home, carry me home
Over the water, I am not alone
```

## The score (cot)
- `cot: full` (default): a chord-annotated ABC score first, then the music.
- `cot: melody`: a melody-only score - upstream's recommendation for covers.
- `cot: off`: no score; the codes from the words alone (guided by 1.01 by default).
- `abc`: a score to sing over instead of one drawn (with `cot` melody or full). Every song made with a score says it
  back (`score` in the answer): give it again with the **same seed and style and new lyrics** to keep the melody and
  sing new words; edit its chords for a reharmonization. A cover's score is a recording's melody as ABC (no chord
  symbols for `cot: melody`), its sections matching the lyrics'.

## Length, guidance, steps
`seconds` is the most it makes (up to 360; the model ends the song when the lyrics are sung - a verse and two choruses
run about a minute and a half). `cfg` guides the codes against the instruction alone (1: none; try 1.2 for a closer
reading of the style). `steps` are the flow's (32, upstream's). Each song is one seed: the same request and seed make the
same song on this engine.
