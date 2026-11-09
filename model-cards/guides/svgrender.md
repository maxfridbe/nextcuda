# svgrender LIVE: pictures as SVG paths

`image.svg_fit {image}` fits a picture with paths; `image.svg_generate {prompt}` draws the prompt first (Z-Image-Turbo,
with the style's words after it) and fits that. Both answer a complete SVG; `image.status {label}` shows the step, the
loss and, every `preview_every` steps, the SVG so far; `image.cancel {label}` stops it at the next step.

- **style**: `iconography` (filled closed paths, LIVE - logos, icons, flat illustrations), `pixelart` (a grid of
  squares, `paths` a side), `sketch` (black open strokes - line art), `painting` (coloured strokes).
- **paths**: more paths, more detail (and a larger file). 32-64 for an icon, 128-256 for an illustration.
- **iters** a layer (250) and the **schedule** (paths added per layer, default eight layers): more steps fit closer.
- **size**: the canvas fitted on (384). The SVG keeps the picture's own size through its viewBox.
- Pictures with flat colours and clear shapes fit best; photographs come out posterised.
- For text to SVG, describe the subject only: the style's words (VectorFusion's "minimal flat 2d vector icon. lineal
  color. on a white background") are added unless `prompt_suffix` says otherwise.
