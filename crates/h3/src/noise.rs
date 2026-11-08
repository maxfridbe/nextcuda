//! The sampler's starting noise, value for value what the reference pipeline draws: `torch.manual_seed(seed)` and
//! `torch.randn(shape, dtype=float32, device="cpu")` for each stream in turn (video, then audio), from one generator
//! (the generator itself: nextcuda_diffusion::torch_rng).

pub use nextcuda_diffusion::torch_rng::Mt19937;

/// The starting noise of a clip: video [C, T, H, W] values, then audio [C, 2, T], drawn in that order.
pub fn clip_noise(seed: u64, video: usize, audio: usize) -> (Vec<f32>, Vec<f32>) {
    let mut g = Mt19937::new(seed);
    let v = g.randn(video);
    let a = g.randn(audio);
    (v, a)
}
