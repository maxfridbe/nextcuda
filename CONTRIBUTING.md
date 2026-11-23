# Contributing

nextcuda welcomes new models - image first, then language, video and audio - and work on the ones it has. Each model
architecture is its own **engine**: a complete runtime for that model, free to be as specialised as it needs to be
fast. This guide is the few rules every change follows; [skills/cuda-porting/SKILL.md](skills/cuda-porting/SKILL.md)
is the method (how to check, clock and tune an engine) and `docs/benchy/README.md` the format of a result.

## The rules

1. **CUDA is the only GPU runtime.** Kernels are CUDA (nvcc), on the GPU's own stream; cuBLAS is fine, and no
   third-party C / C++ kernels are vendored (nextcuda's own since 2026-11-14; a design learned from one carries an
   origin note). No SYCL, Vulkan, OpenCL, HIP, Metal, wgpu, or a framework that brings another (candle, PyTorch, ONNX
   Runtime, burn...). Host code is Rust. Python is a fallback for an engine whose Rust one is not there yet, never the
   first choice.
2. **Third-party crates are fine** - anything that does not bring another GPU runtime (rule 1). Prefer a few
   well-known ones; say why in the pull request.
3. **Keep the architecture** (`docs/architecture.md`):
   - an engine is its own crate under its kind (`image/<arch>`, ...), with its kernels in `kernels/<kind>/<arch>/`;
   - it depends only on the foundation (`crates/`) and its own kind's contract (`<kind>/contract`) - never on another
     kind, the glue (`glue/`) or the program (`cli/`);
   - what two kinds both need moves down into the foundation, not sideways;
   - its kernels' C functions are named `ns_<kind>_<arch>_*`, declared in its own header, and bound by its own crate
     by name - never added to the shared `kernels/ns/ns.h`.
4. **Exact before fast.** An engine is checked against a reference before it is tuned, and every speed claim comes
   with the measurement that made it: `nextcuda image bench` before and after, both quoted.
5. **Tune in the model's silo, never in the shared kernels; tune for other cards in a variant of it.** A kernel tuned
   for one model goes in that model's silo, `kernels/<kind>/<arch>/silo/` - its own library,
   `dist/silo/libnextcuda-<arch>.so`, which the engine opens at load and calls for what it covers, the shared kernel
   for everything else (`NS_SILO=0` turns it off). Start a silo kernel as a copy of the shared one, then change the
   copy. **A silo is per model architecture, never per card**: one library holds every card's code (what a card has
   or lacks - FP8 tensor cores - is decided in it by what the card reports, `cudaGetDeviceProperties`). A model tuned
   for another class of card - an 8 GB card, one of fewer SMs, another compute capability - goes in a **variant** of
   its silo, `kernels/<kind>/<arch>/silo/variants/<name>/`: a manifest (`variant.yml`: a one-line `description`, who
   `measured` it on what, a `match` rule on what the device reports - `vram_mb_min` / `_max`, `units_min` / `_max`
   (SMs), `mhz_min` / `_max`, `cc_min` / `_max` (compute capability, 86 for 8.6), a `name` substring as the last
   resort) and only the ops that differ, under the silo's C names, built as `dist/silo/libnextcuda-<arch>@<name>.so`
   (`kernels/ns/ns_variant.hpp` hands the default silo's op the shapes a variant's op does not cover). Each GPU takes
   its ops at load from the variant it matches, else the silo's, else the shared kernel (`nextcuda_core::silo_for`). A
   variant gives the same outputs as its silo (`NS_SILO_VARIANT=none` against it), its pull request quotes before /
   after numbers on its hardware, and it never changes a memory plan (memory is planned from what the device
   reports). **A variant may also set its engine's options for its cards**: an optional `settings:` block in
   `variant.yml`, option name -> value, the names the engine declares (`EngineOption`, the ones `--opt-NAME` takes).
   Where the variant applies to every GPU the engine loads on, each becomes the engine's default - below a request's
   options, `--opt` / the environment and the model card's settings, above the engine's own default (`nextcuda silo`
   and the load's log say what applied). **An engine's alternative host-side paths** (another chunking, a fusion, an
   expert placement) are declared options with enumerated values (`value: "spread|pack"`): the variant picks one by
   name, and a new path is itself an engine change under the gate, never a variant's own code. A setting never sizes
   memory (an option whose value is `TOKENS`, `GIB`, `MIB` or `BYTES` is refused). **Two variants of one silo never
   match the same card**: their numeric `match` ranges must not meet (a name never tells two rules apart) - give one a
   tighter `match`. And a variant has a **gate baseline on a card it matches**, recorded on that card and committed in
   its pull request (the merge gate below).

6. **A front-door method is its handler.** `glue/serve/src/home/methods.rs`: `#[rpc(name = "image.generate", tags
   = "image", example = r#"{...}"#)]` on a fn with a doc comment (its summary), and a request struct with
   `#[derive(Rpc)]` (each field's doc comment its description, `#[schema(min, max, default, one_of ...)]` its
   limits). The table, the OpenAPI document's operations, system.version's list and the wfe client
   (`nextcuda gen-api` -> `wfe/home/src/gen/nextcuda.ts`) are made from that; never describe a method a second
   time. The generator, rpcgen, is vibe-creative-platform's, vendored in `vendor/rpcgen`: change it there, then
   `tooling/sync_rpcgen.sh` (a test fails on a copy edited here, and on a stale client). What happened is in the
   answer's envelope (`error.code`, `retryable`), never only in the HTTP status.

Rules 1, 3 and 5 are checked by `cargo test` (`cli/nextcuda/tests/architecture.rs`: no per-card silo, valid variant
manifests that hold only their silo's ops, settings their engine takes, no two variants of a silo matching one card, a
gate baseline for each variant, no card name or VRAM size in the engines' plans, every variable documented) and by the
merge gate; rule 6 by the compiler, `tsc` and
`cargo test` (`home::api_files_tests`).

## Building

```sh
./build.sh                        # the program and the web front end, in a debian:trixie container: dist/
kernels/build.sh                  # the CUDA libraries, with nvcc, on a host that has it (OUT=... for where)
cargo test --workspace            # the architecture rules, the model cards, the unit tests
nextcuda image selftest --gpu 0   # the engine's kernels against host arithmetic, on the GPU
```

## Adding a model

1. **A model of an architecture already served needs only its card**: `model-cards/<kind>/<id>.yml` (every file a
   pinned link with its size and SHA-256; `model-cards/README.md` has the format), then `nextcuda models pull <id>`.
2. **A new architecture is an engine crate** registered in the program's list of that kind
   (`cli/nextcuda/src/image.rs`: `engines()`), with a row in `docs/environment.md` for every setting it reads (the
   architecture test fails on one that has none).
3. **Then make it fast**, as its own change: clock it with the kind's bench, find where the time is, put the faster
   kernels in its silo, check the results are the same and clock it again.

## Pull requests

- One change a pull request, with what it does, why, and how it was checked (the commands and their output).
- Speed changes: the numbers before and after, on which GPU, with which command.
- **The merge gate**: `tooling/gate.sh <base> <head>` works out from the paths changed which engines they touch (an
  engine's own code, kernels or silo: that engine; a silo variant: that engine on the GPUs that take it; a shared part:
  the engines `tooling/gate-baselines.json` maps it to), builds `<head>` in a build copy (`NS_GATE_HOST`, the project's
  box brainiac-nvidia; never the service's files), runs each engine's quick run on each of its GPUs twice under the GPU
  lock and only while the box is idle, and compares with the baselines: the output exactly for a fixed seed (an LLM's
  greedy tokens and their log-probabilities, a picture's, a sound's or a mesh's bytes, a clip's decoded frames), the
  throughput within 3%. Paste its table into the pull request; a change that moves a number on purpose records new
  baselines (`NS_GATE_RECORD=1`) and says why. An engine with no baseline is listed as such - add its run to the file
  when you port one.
- **Gating hardware the project does not own.** The baselines are per card, keyed by the id `nextcuda silo` prints
  (`rtx-3060-28u-12g`: the name, SMs and memory it reports, with the `device` recorded beside it; a run over several
  GPUs: their ids joined by `+`), so the gate runs on your own box: `NS_GATE_HOST=local` (no ssh), `NS_GATE_GPUS` your
  GPUs, `NS_GATE_PATHS=FROM=TO` where the baselines' model files are on your disk. Record your card's baseline on the
  base commit (`NS_GATE_RECORD=1 tooling/gate.sh <base> <base> <kind>/<arch>`), commit it in the same pull request,
  then run the gate on the head and **paste its table from your hardware**. A variant's pull request always does
  this: the architecture test fails on a variant with no baseline on a card its `match` accepts.

## Code style

- Write like the code around you: the same comment density, naming and idioms. Comments say what something is and
  why, in plain words.
- `cargo clippy --all-targets -- -D warnings` is clean; `unsafe` blocks carry a `// SAFETY:` line.
- CUDA kernels: every exported function is `extern "C"`, returns 0 or -1 with `ns_fail(reason)`, and queues its work
  on the GPU's stream; a launch is checked with `NS_LAUNCH_CHECK()`.
