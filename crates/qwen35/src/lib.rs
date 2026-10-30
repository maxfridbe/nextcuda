//! nextcuda-qwen35: the Qwen3.5 / Qwen3.8 model core - load a llama.cpp GGUF onto one or more cards, run the prompt and
//! decode passes (`engine`: gated DeltaNet and gated full attention, the dense SwiGLU, sessions, checkpoints, MTP drafts,
//! the vision tower `vision`), with a tap on any named tensor (`result_norm` is the hidden state MOSS-Image conditions on).
//!
//! It is the foundation's because two kinds run this one model: the llm engine (`llm/qwen35`) serves it behind the llm
//! contract, and the image engine MOSS-Image uses it as its text encoder - neither may depend on the other (CONTRIBUTING
//! rule 3). It knows no contract: errors are `nextcuda_core::Error`, a sampler is a closure. Its kernels are
//! `kernels/llm/qwen35/` (and the nemotronh engine's shared products), bound from the process's kernel library by name.

pub mod engine;
mod ffi;
pub mod vision;
