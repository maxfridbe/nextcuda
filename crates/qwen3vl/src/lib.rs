//! nextcuda-qwen3vl: Qwen3-VL's language model as a text encoder - what Qwen-Image 2.1 conditions on. Read from a
//! llama.cpp GGUF (`gguf`), its matrices in their stored types on the card, made half a slice at a time.
//!
//! nextsycl's version of this crate read ComfyUI's int8 ConvRot file; this one reads the llama.cpp GGUFs (the text
//! tower, and the vision tower from the projector file `vision`: what an edit's reference pictures go through). `chat`
//! is the same language model writing (its weights kept on the card, a key/value cache, the head): image.describe's
//! native describer.

pub mod chat;
pub mod gguf;
pub mod reader;
pub mod vision;

pub use chat::{Chat, Sampler, Written};
pub use gguf::GgufTextEncoder;
pub use reader::Reader;
pub use vision::{Seen, Vision};
