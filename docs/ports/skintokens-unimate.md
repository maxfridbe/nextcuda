<!-- Queued port (2026-10-27). The brief below is as written for the nextsycl ports; for nextcuda:
     - nextcuda's own rules win where they differ (CONTRIBUTING, model-cards/README.md): cards carry `license`,
       `commercial` and `license_note` (UniMate's checkpoints are CC BY-NC 4.0: `commercial: false`, tagged
       non-commercial in the pages); CUDA kernels replace the SYCL ones (shared kernels under kernels/, the model's own
       in a per-card silo); commit messages end with the session's Claude-Session line, which this repo uses, in place
       of the brief's "no attribution trailers" (that rule was nextsycl's).
     - Then the creative platform's Model Maker: SkinTokens as a rig engine beside UniRig / RigNet, UniMate as an
       animate engine beside AnimateAnyMesh (box-made: `model3d.rig` / its animate method), with their own options,
       wording and licence tags.
     - Source: the two nextsycl branches named below were not on GitHub (maxfridbe/nextsycl) when this was queued:
       they need pushing (or a path to the checkout) before the port starts. -->

# Porting SkinTokens and UniMate to nextcuda

nextsycl (Rust + SYCL, Intel B70/B65) already runs both models. Port from that code: it is the working answer, checked
against the reference, so you do not have to rediscover the model. Read this file, then read the two crates.

## Where the finished code is (nextsycl repo, unmerged branches)

| model | branch | commit | crate |
|---|---|---|---|
| SkinTokens / TokenRig (rigging) | `worktree-agent-a6b68d6bb2c9d7e26` | 6ff7ffa | `model3d/skintokens` |
| UniMate (skeleton animation) | `worktree-agent-a79953646f163abb0` | 78cff3a | `model3d/unimate` |

Per model, also read: `model-cards/model3d/<id>.yml` (weights, sizes, sha256), `model-cards/guides/<id>.md`,
`reference/<id>/ref.py` (the reference script), `docs/ports/<id>.md` (the brief), the `docs/3d.md` section, and
`kernels/model3d/<id>/` (the SYCL kernels you replace with CUDA). If nextcuda has its own CONTRIBUTING / model-card
conventions, they win over anything here.

## Rules that carried over

- Exact before fast. Get every stage within float32 noise of the reference, then tune.
- Tune only in the model's own kernel directory (a "silo"), never in shared kernels.
- Models are described by model cards (YAML, read at run time), not by code tables.
- Never execute code or load pickles from downloaded repos/checkpoints. Read the reference, write your own `ref.py`
  that transcribes the modules, and read checkpoints with a safe pickle reader (tensor data only).
- Keep at least 1.5 GiB of VRAM free on a shared GPU; never kill another process's run; delete test outputs by exact name.
- Commit as you go; no attribution trailers (Co-Authored-By, session links, "Generated with").

## Target hardware: a 12 GB GPU

Both models fit with room to spare. Budget (measured on the B70, so treat as a guide and re-measure on CUDA):

| model | weights resident | peak seen | card figure |
|---|---|---|---|
| SkinTokens | about 1.1 GB checkpoint (bf16 values kept as is) | about 3.7 GiB | 4.5 GiB |
| UniMate | about 0.4 GB EMA tensors in float32, plus T5 encoder (about 0.5 GB) | about 1 GiB above idle | 2.0 GiB |

- Usable budget is 12 GB minus the 1.5 GiB that stays free, so about 10.5 GiB. Run one model at a time; both together
  would still fit (about 6.5 GiB), but do not rely on it on a card that also drives a display.
- SkinTokens: the beams' key/value cache grows with skeleton length. Size it from the longest allowed skeleton and refuse
  or lower the beam count (`--opt-beams`) instead of running out of memory. Greedy (1 beam) is the safe fallback.
- SkinTokens: the encoders attend over 54,000 keys. Never materialise the score matrix (54,000 x 54,000); use a fused or
  chunked attention, as the silo's one-pass kernel does.
- UniMate: always padded to 71 joints x 60 frames, so memory does not depend on the rig. Keep the weights in float32
  while matching the reference; only try fp16/bf16 afterwards, and re-run the checks (a flow-matching ODE can amplify
  small errors).
- No offloading, quantisation or tiling is needed at this size. If a stage does not fit, the stage is wrong, not the card.

## Method (what worked, in order)

1. Pull weights, write the reference script `ref.py` (CPU torch, float32) that dumps each stage's output for a fixed
   seed and input mesh.
2. Port stage by stage in Rust with the per-stage check (`model3d check <dump dir> --model ID`), reporting relative RMS.
   Fix a stage before starting the next one.
3. Only when all stages pass: end-to-end run, then a silo for the hot kernels (attention first), re-running the checks
   after every kernel change.
4. Card + guide + docs, tests (`cargo test`, `clippy -D warnings`).

Port the sampling and grammar logic (CPU) exactly; the numbers below only hold if it is exact.

## SkinTokens / TokenRig (rig a mesh: skeleton + skin weights)

- Upstream: https://github.com/VAST-AI-Research/SkinTokens @ 273b691. Code and weights MIT.
- Weights, Hugging Face `VAST-AI/SkinTokens` @ 79736cad0fd84de384d5eede659b4ebd24effe33:
  `experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt` (1,131,603,979 B) is the only file needed at
  inference, because it carries a bf16 copy of the skin VAE. `experiments/skin_vae_2_10_32768/last.ckpt` (487,311,745 B)
  is unused. The Qwen3-0.6B config is only a shape; the decoder is built from scratch.
- Pipeline as ported: mesh normalised, 54,000 sampled points; Michelangelo shape encoder (cross-attention then
  self-attention, farthest points on the embeddings); a Qwen3-0.6B-shaped decoder (RoPE, GQA, RMSNorm, SwiGLU) over the
  shape prefix writing the skeleton tokens then the skin tokens as one sequence, grammar-constrained, beam search with
  sampling (10 beams, top_k 5, top_p 0.95, temperature 1.0, repetition penalty 2.0); the FSQ-CVAE decoder turns skin tokens
  into weights per sample point; each vertex takes its 8 nearest sample points, top 4 bones, normalised.
- Quirk to keep: the upstream demo forces the end token one token early, so the last joint's fourth skin token reads as
  code 0. The default reproduces this; `--opt-skin-end fix` lets the model say it.
- Not ported: the demo's `--use_postprocess` and `--use_skeleton`.
- Targets the SYCL port hit (relative RMS vs `ref.py`): encoder latents 4.5e-5, teacher-forced logits (130 positions)
  1.3e-5 with argmax equal at 130/130, greedy decoding the same 129 tokens, joints and parents identical, skin over the
  54,000 samples 9.6e-7, weights on vertices 8.1e-8. The default beam-sampling path has no reference (the reference
  decodes greedily only): check greedy and the beam cache's slot-pool history, and keep the sampling logic identical to
  UniRig's/transformers'.
- Speed to match or beat (B70): 3.0 s for a 3.6K-vertex humanoid with 10 beams (143 decode steps 1.5 s), 2.5 s greedy,
  about 3.7 GiB VRAM. The three silo kernels were: one-pass grouped-head attention, decode-step attention over the beams'
  slot pool, bf16 decode products.

## UniMate (animate a rigged asset from a sentence)

- Upstream: https://github.com/Friedrich-M/UniMate @ b78c780. Code MIT; **checkpoints CC BY-NC 4.0** (non-commercial;
  say so in the card and guide).
- Weights, Hugging Face `Linzhan/UniMate` @ 174fc755ec248d0c98f27759b7a95648e7f05905: run directory
  `unimate_uniml3d_f60_v3` (`config.json`, `dataset_stats.npy`, `checkpoints/checkpoint_step_150000.pt`, 1,185,836,040 B,
  the `graph_adaln` variant; read only the EMA tensors). Text encoder `google/flan-t5-base` (encoder only, pin a revision).
- Pipeline as ported: read the rig (joints, parents, rest pose, joint names); rule-based joint-name cleaning and facing
  (upstream uses an LLM plus a review step, which an engine cannot call); T5 embeddings of the caption and of each joint
  name; the denoiser (spatial attention per frame and temporal attention per joint, graph-distance biases, adaLN
  conditioning) sampled as a flow-matching ODE with classifier-free guidance (cfg 3, an adaptive solver); 60 frames,
  de-normalised with `dataset_stats.npy`; decoded to joint rotations plus root motion; written as a glTF skeletal animation.
- Limits: at most 71 joints, a single root, plain ASCII prompts (flan-T5's Unicode normalisation table is missing),
  `--steps` ignored (adaptive solver). The rig is padded to 71 joints always.
- Names matter more than joint count. With real names (a quadruped's "Right Front Hip") the motion is stable across seeds.
  With generic names (`bone_0` from UniRig/SkinTokens) it is weak and unstable; structure-only mode is out of
  distribution. Say this in the guide and suggest trying several seeds or renaming joints.
- Targets (relative RMS vs `ref.py`, same noise): T5 7.4e-7, one denoiser pass 1.5e-7 to 4.3e-7, guided velocity 1.3e-6,
  full ODE 1.2e-6 (the same step and evaluation counts), decoded motion 3.2e-7, rest pose 1.8e-7, graph relations identical.
- Speed (B70): 13 to 14.5 s per rig (104 to 116 evaluations), about 2 GiB; no silo yet, so there is room.
- The mesh crate gained skeletal-animation output and a fix keeping one skin over several primitives; port those too.

## Chain

`model3d rig mesh.glb --model skintokens` then `model3d rig out.glb --model unimate --prompt "An object walks forward."`.
Verify both by posing the result in a viewer, not only by numbers; neither port has had that look yet.
