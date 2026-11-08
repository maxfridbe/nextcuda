# Z-Image-Turbo

Tongyi-MAI's few-step text-to-image model: Z-Image's 6B single-stream DiT distilled to 8 steps, read by Qwen3-4B.
Photoreal pictures, and text in the picture (English and Chinese) that comes out legible.

- **Write a full description.** It follows long, specific prompts well: the subject, the setting, the light, the
  camera or the style. Prompts in English or Chinese both work.
- **Text in the picture:** quote it and say where it goes and how it looks ("a poster with the title "NIGHT MARKET" in
  large white serif letters at the top"). Short lines render best; long paragraphs can come out garbled.
- **Sizes:** sides multiples of 16 (others are rounded). It is made for about a megapixel and up: the official demo
  offers 1024 x 1024, 1152 x 896, 1152 x 864, 1248 x 832, 1280 x 720, 1344 x 576, at 1280 (1280 x 1280, 1536 x 1024,
  1536 x 864 ...) and at 1536 (1536 x 1536, 2048 x 1152 ...), and their turns. At most 2048 a side and 2.4 MP.
- **Steps 8, no guidance:** the turbo is distilled from guidance, so it runs one pass a step and takes no negative
  prompt and no cfg (both are refused). Other step counts run but it is made for 8. `shift` (default 3) is the
  scheduler's time shift, as the official demo's slider: higher spends more of the steps on the composition.
- **Seed:** the same seed, prompt and settings make the same picture.
- **Text to image only:** it does not edit pictures (Z-Image-Edit is a separate model).
