# MOSS-Image SFT

A general text-to-image model: photographs, illustration, product shots, scenes with legible text in them.

- **Describe the picture as a caption would:** the subject, what it does, where, the light, the medium ("photograph",
  "watercolour"). The prompt reaches the model through Qwen3.5-9B's chat template as it is: no quality tags are
  added, and none are needed.
- **Text in the picture:** quote the words ("a shop sign that reads "OPEN"").
- **Size:** 1024 x 1024 is the default and the recommended size; any multiple of 16 from 256 to 2048 a side works
  (larger is slower: the attention runs over every position).
- **Steps 50, cfg 3** (the pipeline's own). Guidance runs a second pass a step against the negative prompt (empty by
  default) with one block left out; cfg 1 turns it off and halves the time, at a cost in detail and prompt-following.
  Fewer steps (20-30) are faster and usually close.
- **Seed:** the same seed, prompt and settings make the same picture.
