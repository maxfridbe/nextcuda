# Lessons learned

- **CUDA's GPU numbers are not nvidia-smi's.** CUDA puts the fastest card first; `CUDA_DEVICE_ORDER=PCI_BUS_ID` makes
  card 0 the one nvidia-smi calls 0. The program sets it before any CUDA call.
- **The step is not the PCIe link.** Keeping the 7.3 GB transformer resident on the 5070 gave the same 1.27 s/step as
  streaming it: the time is in the matmuls and attention, so that is where the silo goes.
- **A community GGUF is a converter's choice.** One stores `img_in.weight` as [1024, 256] for [4096, 64]; the loader
  takes the model's hidden size from it and refuses the file. A shape fix in a copy is the right size of answer.
- **The first picture pays the page-in.** The first graph after a load streams the weights from disk to host memory
  (30-45 s for a 13 GB model); the second picture does not.

- **Two cards are a pipeline, not a relay.** Splitting the layers only adds the cards' times; the speed came from the
  first card starting the next chunk while the second finishes this one - and from nothing on the host waiting.
- **A launch queue fills.** One host thread feeding two cards a thousand kernels each blocks in `cudaLaunchKernel`
  (nsys showed 0.17 ms a launch); graphs, one a stage, fixed it.
- **Measure the other runtime, kernel by kernel.** nsys on llama.cpp showed its per-card kernels were not much faster -
  the cards overlapped - and, for decode, that its matrix-vector products were.

- **A read-aloud that repeats is a queue that ran dry.** Open WebUI ends playback when its next sentence is late; the
  speech model was being swapped out for the chat model's title and tag requests. Fit them side by side.
- **Score a voice with ears you trust.** Whisper against the text, the port against its reference package on the
  same lines: the same error rate - the trouble was the mode (continuing a recording) and the symbols, not the port.
- **A cache needs a floor, not just a budget.** Checkpoints of forms the engine no longer makes filled the drive; the
  cache now drops them at start and keeps 8 GiB of its drive free.
