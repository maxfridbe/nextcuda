//! MiniMax H3's model core: H3's engine (the sycl-h3 studio's `h3-core`, its jobs and its mp4 writer) moved in whole,
//! so its numbers are H3's. The arithmetic runs in the diffusion kernels (`kernels/diffusion`, H3's under the `nsd_`
//! prefix), which every kind's library holds; this crate owns everything around it: the device and its memory,
//! reading checkpoints, the model graph, the sampler, the encoders and decoders, the jobs.
//!
//! It sits in the foundation because two kinds run it: the video engine (`video/h3`: clips with sound) and
//! P-H3-Image (`image/ph3image`: Pruna's image model, H3's denoiser over one latent frame).

pub mod qwen3vl_vision;
pub mod audio;
pub mod control;
pub mod denoiser;
pub mod device;
pub mod dit;
pub mod dtype;
pub mod esrgan;
pub mod jobs;
pub mod media;
pub mod gguf;
pub mod layout;
pub mod noise;
pub mod load;
pub mod lora;
pub mod ops;
pub mod reference;
pub mod rng;
pub mod safetensors;
pub mod sys;
pub mod te;
pub mod tokenizer;
pub mod upscale;
pub mod venc;
pub mod vae;
pub mod vision;

pub type Result<T> = std::result::Result<T, Error>;

/// One error type for the engine: a message, with the context added on the way up.
#[derive(Debug)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error(e.to_string())
    }
}
impl From<String> for Error {
    fn from(e: String) -> Self {
        Error(e)
    }
}
impl From<&str> for Error {
    fn from(e: &str) -> Self {
        Error(e.to_string())
    }
}

/// Adds what was being done to an error: `read(..).ctx("reading the header")?`.
pub trait Ctx<T> {
    fn ctx(self, what: impl std::fmt::Display) -> Result<T>;
}
impl<T, E: std::fmt::Display> Ctx<T> for std::result::Result<T, E> {
    fn ctx(self, what: impl std::fmt::Display) -> Result<T> {
        self.map_err(|e| Error(format!("{what}: {e}")))
    }
}
