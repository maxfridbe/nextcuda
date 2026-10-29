# Building, running as a service, clocking and checking

## How it is built

```
kernels/ns            the C ABI (GPUs, memory, streams, copies) on CUDA - nextsycl's ns.h, the same functions
kernels/image/qwenimage21   the engine's own kernels (qi21.cu)
kernels/image/sd      a flat C ABI over stable-diffusion.cpp's ggml-cuda: one loaded model, progress, cancel
crates/core           the safe layer over libnextcuda-<kind>.so (host-memory fallback without it)
image/qwenimage21     the engines: Qwen-Image 2.1 and 2512 behind nextcuda-image's contract
kernels/llm/nemotronh, llm/nemotronh   Nemotron-H: Mamba-2, attention and Strata's i-quant products (CUDA), the
                      engine behind nextcuda-llm's contract (llm/contract), its expert cache and reader threads
glue/serve, glue/models, cli/nextcuda    the control plane, the registry and the program, as nextsycl's
```

The rules of nextsycl's [`architecture.md`](architecture.md) hold - dependencies point down, one library a kind, tuning in silos -
and `cli/nextcuda/tests/architecture.rs` checks them on every `cargo test`, with CUDA as the one GPU runtime.

`./build.sh` builds the web front ends and then the program in a debian:trixie toolchain image (its glibc is the GPU
host's); `kernels/build.sh` builds the CUDA libraries with `nvcc` for the 3060 (sm_86) and the 5070 (sm_120).

The front ends are inside the binary (`glue/serve/build.rs` embeds `dist/wfe`; `NS_WFE_FROM_DISK=1` reads the
directory instead). The kernel libraries are inside it when they were there to embed at build time
(`crates/sys/build.rs`: `NS_LIB_DIST`, else `dist/`): `nvcc` is on the GPU host, so only `EMBED=1 tooling/deploy.sh`
brings them back first; such a binary opens them from anonymous memory files (`NS_LIB_FROM_DISK=1`: the files). The
silos stay files (`dist/silo/`, beside the executable).

## Run it as a service

`tooling/nextcuda.service` is a systemd user unit (`:8000`, restarted on failure); `tooling/deploy.sh [host]` builds
here, builds the CUDA libraries where `nvcc` is, copies it all over and restarts the unit:

```sh
./tooling/deploy.sh gpu-host            # an ssh host with CUDA; SKIP_BUILD=1, SKIP_KERNELS=1, RESTART=0 to do less
EMBED=1 ./tooling/deploy.sh gpu-host    # the kernel libraries inside the binary too (built there first, brought back)
systemctl --user status nextcuda.service
journalctl --user -u nextcuda.service -f
```

Settings are `nextcuda.conf` beside `dist/` (or `~/.config/nextcuda.conf`), then the environment; every variable is
in [`environment.md`](environment.md).

## Clocking it

`nextcuda image bench` is the image kind's benchy: fixed prompt and seeds, cold and warm pictures kept apart, every
stage timed, peak GPU memory, power and energy sampled beside each picture, `matrix.md` and `matrix.jsonl` written.
The method (what to capture, how a result is kept and compared, how an engine is tuned in its silo) is
[`benchy/README.md`](benchy/README.md) and `skills/cuda-porting/SKILL.md`.

```sh
nextcuda image bench --models qwen-image-2.1-q8,qwen-image-2512-q4km --sizes 768x768,1024x1024 --steps 8 --runs 3 \
        --note "what is different about this run"
```

## Checking it

```sh
cargo test --workspace            # the architecture rules, the model cards, the samplers' math
nextcuda image selftest --gpu 0   # the CUDA kernels against host arithmetic
```
