# Prompting LTX-2.5 (video with its own sound)

LTX-2.5 makes a clip **with its own sound** from one paragraph of ordinary prose that describes what is seen *and
heard*. It reads the whole paragraph as a description of the clip. H3's structured format (sections, `<d>[English]`
tags, `<Subject N>` tokens) does not belong here.

## Rules
- **One paragraph, present tense, no labels**: no bullet points, no "Audio:" / "Visual:" headings.
- **Never write stage directions about speaking**: LTX says them out loud ("her lips moving with the words" was
  spoken as a line). Describe the line, not the act of saying it: `she says with a clear, present voice, "Dinner is
  ready."`
- **Do not ask for film stock unless you want it**: "35mm grain" is obeyed. For a clean picture say so: "clean modern
  digital cinematography, pin-sharp focus, a locked-off tripod, no film grain and no camera shake".
- **Describe the sound explicitly**: the audio is guided harder than the picture (7 against 3) and follows the
  caption closely. For dialogue: "the dialogue is clean, centered, close-mic, natural, and distortion-free". Give a
  voice its character ("a very deep, resonating, mechanically processed voice").
- **Say when you want no music** ("with no background music, only the quiet simmer of the pan beneath her voice").
- **Length**: any frame count from 9 (the model works on 8k + 1 frames at 24 fps and the clip is cut to what was
  asked: 120 frames = 5 s). The canvas a multiple of 32; what limits a clip is its tokens, width/32 x height/32 x
  ((frames - 1)/8 + 1), at most 9,216 on a 12 GB pair: 640 x 384 for 5 s is 3,840 (the movie maker's size), 1024 x 576
  for 5 s is 9,216 (the largest tried).
- **A first frame**: `first_frame` (a picture file in the clips directory - `video.upload` it) makes it image to video:
  the picture is the clip's first frame. An anchor word (`prev`, `first`) or a clip is not read as a picture.
- **A negative prompt** is read (the guidance has an empty-prompt branch to steer); H3 ignores it.
- **The sound is the model's draw**: five seeds on one prompt range from a thin hiss to a full mix, and it comes out
  quiet (peaks around -24 dBFS: normalize it downstream). Say the sound in the prompt; try another seed when it is thin.

## Engines
`video.engines.list` gives two LTX rows; `video.generate`'s `engine` takes their `quant`:
- `ltx2` (the shots): `variant` `dev` (the default; the `ltx-2.5` model: 30 guided steps, four passes a step - the
  prompt, the empty prompt, STG and modality isolation: the take) or `distilled` (the `ltx-2.5-fast` model's DiT on its
  own eight sigmas, unguided: about 80 s a clip against the dev take's 7 minutes; a good preview, softer than the dev
  take, its sound as much a draw).
- `ltx2_upscale`: a finished clip (`source_clip`, a file in the clips directory) made twice the size through LTX's
  latent upsampler, its sound copied; no prompt (send ""). It makes a new clip.

## Example
> A woman in her forties, wearing a simple, light-colored top, stands at a stainless steel stove in a warm, modern
> kitchen bathed in soft evening light, with visible steam rising from a metal pan she is stirring; a static medium
> shot frames her from a front-facing angle on a locked-off tripod, maintaining pin-sharp focus with clean modern
> digital cinematography and no film grain or camera shake; as she stirs, she turns toward the camera and clearly
> speaks with a clear, present voice, "Dinner is ready," followed by a slight pause, and then she continues, "come
> and sit down"; the dialogue is clean, centered, close-mic, natural, and distortion-free, with no background music,
> only the quiet simmer of the pan beneath her voice.
