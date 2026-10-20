# ACE-Step 1.5

A song from a **caption** (`instructions`: genre, mood, instruments, the voice, production) and **lyrics** (`input`),
`seconds` long (1 to 600; 30 when not said). It sings the lyrics in the time it is given - the length is the request's,
not the lyrics' - so give a long song enough words, and a short one few.

## The caption

Comma-separated tags and short phrases, the way ACE-Step was trained: `upbeat indie pop, bright electric guitars,
punchy drums, warm bass, female lead vocal, summery`. Name the voice (`warm melodic alto female lead vocal`), the
instruments, the mood and the production; tempo and key are better given as options than in words.

## The lyrics

Structure tags on lines of their own - `[Verse]`, `[Pre-Chorus]`, `[Chorus]`, `[Bridge]`, `[Outro]`,
`[Instrumental]` - and the words under them, a line a phrase. No lyrics (or `[Instrumental]`): an instrumental.

## Options

| option | what |
|---|---|
| `bpm` | the tempo (beats a minute) |
| `key` | key and scale: `C major`, `A minor`, `F# major` |
| `time-signature` | beats a bar: `4`, `3`, `6` |
| `language` | the lyrics' language as a code (`en`, `zh`, `ja`, `es` ...); `unknown` when not given |
| `shift` | the schedule's shift (3) |
| `dcw` | the per-step wavelet correction (on) |
| `normalize` | the peak to -1 dB (on) |

`steps` (8) can go up to 20; turbo is guidance-distilled, so `cfg` is not used.
