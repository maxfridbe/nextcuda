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
- **Length**: any length up to 15 s in one clip (made as 8k + 1 frames at 24 fps and cut to what was asked); the canvas a
  multiple of 32 (640 x 384 is the movie maker's size; up to 1344 x 768).
- **A first frame**: `first_frame` (a picture) makes it image to video - the picture is the clip's first frame.

## Engines
- `ltx-2.5` (LTX-2.5): the dev DiT, 30 guided steps (four passes a step: the prompt, the empty prompt, STG and
  modality isolation). The take.
- `ltx-2.5-fast` (LTX-2.5-FAST): the distilled DiT on its own eight sigmas - a preview: visibly worse in picture and
  sound (thin audio).

## Example
> A woman in her forties, wearing a simple, light-colored top, stands at a stainless steel stove in a warm, modern
> kitchen bathed in soft evening light, with visible steam rising from a metal pan she is stirring; a static medium
> shot frames her from a front-facing angle on a locked-off tripod, maintaining pin-sharp focus with clean modern
> digital cinematography and no film grain or camera shake; as she stirs, she turns toward the camera and clearly
> speaks with a clear, present voice, "Dinner is ready," followed by a slight pause, and then she continues, "come
> and sit down"; the dialogue is clean, centered, close-mic, natural, and distortion-free, with no background music,
> only the quiet simmer of the pan beneath her voice.
