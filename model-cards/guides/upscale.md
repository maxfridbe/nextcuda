# Upscaling: Real-ESRGAN, waifu2x and FlashVSR

`image.upscale` makes a picture bigger; `video.upscale` does it to every frame of a clip (the sound copied as it was)
and answers an MP4. Both are jobs (wait, reentrant_id, `image.status`, `image.cancel`).

| model | variant | scales | for |
|---|---|---|---|
| real-esrgan | `realesrgan-x4plus` | 4 | photographs, general pictures (the default at 4x) |
| real-esrgan | `realesrgan-x4plus-anime` | 4 | illustration, anime |
| real-esrgan | `realesr-animevideov3` | 2, 3, 4 | video frames, fast (the default at 2x and 3x) |
| waifu2x | `models-cunet` | 1, 2, 4 ... 32 | illustration; `noise` -1 (none) to 3 |
| waifu2x | `models-upconv_7_anime_style_art_rgb` | 2, 4 ... 32 | anime-style art |
| waifu2x | `models-upconv_7_photo` | 2, 4 ... 32 | photographs |
| flashvsr | `flashvsr-v1.1` | 2, 3, 4 | real footage, faces and skin: one-step diffusion over the whole clip (video.upscale; a picture is a still clip) |

```json
{"image": "upload:up_1792400000_still.png", "variant": "realesrgan-x4plus", "scale": 4}
{"file": "h3_20261019_121939", "variant": "realesr-animevideov3", "scale": 2}
{"image": "data:image/png;base64,...", "model": "waifu2x", "variant": "models-cunet", "noise": 1, "scale": 2}
{"file": "h3_20261019_121939", "model": "flashvsr", "variant": "flashvsr-v1.1", "scale": 2}
```

- Real-ESRGAN sharpens and invents texture; waifu2x denoises and stays closer to the source. 1x with waifu2x is a
  denoise alone (`noise` 0-3).
- A picture goes through in tiles (`tile`: Real-ESRGAN 200, waifu2x 400) with context around each, exactly as the
  ncnn programs do, so the output is theirs (58-66 dB against them); a larger tile is a little faster and takes more
  memory.
- Transparent pictures keep their alpha (upscaled bicubically, as the programs do).
- With no `model`, a waifu2x variant (`models-...`) picks waifu2x, `flashvsr-...` FlashVSR, and any other Real-ESRGAN.
- FlashVSR sees the clip as a whole: its frames go through as one stream (8 at a time, each attending to the 24 before
  it), so detail stays steady from frame to frame and a film of any length has no seams. It keeps faces and skin
  natural where Real-ESRGAN's models turn them waxy or plastic. Its options (in `options`, or beside the others):
  `sparse_ratio` (2.0; 1.5 is faster), `local_range` (11, steadier; 9 sharper), `color_fix` (1: each frame's colour
  matched to the input's), `seed`. Inside, the frame is mirrored out to multiples of 128 and the result cut back, so
  any size works; the memory grows with the frame (1728 x 960 out takes both cards).
