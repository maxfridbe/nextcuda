# P-H3-Image

Pruna AI's image model on MiniMax H3: text to image, and edits from 1 to 8 pictures (the first is changed, the others
are references the instruction can name).

- **Write full, specific prompts.** The model is sensitive to prompting and renders a detailed description much better
  than a short request; Pruna runs a prompt upsampler (Qwen3.5-9B) in front of it upstream, which this engine does not:
  the prompt is rendered as written. Hype words and very long prompts make pictures worse. Quote short text to render
  ("a sign that reads "OPEN"").
- **Edits:** `image.edit` with `images` (the picture to change first, up to 7 references after it) and the instruction
  as the prompt; it is sent as `Edit <Picture 1>. <instruction>` with each picture shown as `<Picture i>`, so name
  them that way ("the hat from <Picture 2>"). Without a size the edit is made at the first picture's size, scaled to 4
  to 8 megapixels.
- **Sizes:** sides multiples of 32. The model is best at 8 MP (2912 x 2912, 3360 x 2496, 3552 x 2368, 3872 x 2176 and
  their turns) and fine at 4 MP (2048 x 2048, 2368 x 1760, 2496 x 1664, 2720 x 1536 and their turns), which edits
  prefer.
- **Steps 8, no guidance:** the checkpoint is guidance-distilled; 8 steps run its own sigmas (another grid:
  `options.sigmas`, e.g. "1,0.95,0.8,0.5,0"). A negative prompt, a cfg other than 1, another step count or sampler
  is refused.
- **Time on the two 12 GB cards:** 1 MP 12.5 s, 4 MP 56 s, 8 MP (the default 2912 x 2912) 166 s; an edit at 1 MP about
  45 s, at its default 4 MP canvas about 3.5 minutes. Sizes are rounded to multiples of 32.
- **Seed:** the same seed, prompt and settings make the same picture.
- **Weak spots (Pruna's):** hands, small people in the background, dense or small text.
