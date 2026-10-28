//! Dense Qwen3.5 / Qwen3.8 (`qwen35`: Qwen3.8-27B and its siblings) behind the runtime's `Engine` contract
//! (`nextcuda-llm`). The model core - its geometry from the file's metadata, gated DeltaNet and gated full attention,
//! the dense SwiGLU, the sessions and checkpoints - is the foundation's `nextcuda-qwen35` (MOSS-Image encodes with the
//! same core); this crate is what the contract adds: the opaque handles, the sampler, the registry entry `kind()`. Its own kernels are in
//! `kernels/llm/qwen35/` (with glm5next's generic conv, L2 norm and delta-rule scan); the products from the stored
//! blocks, the dequantizers and the norms are the nemotronh engine's, in the same library. After nextsycl's qwen35.

use std::any::Any;
use std::sync::Arc;

use nextcuda_gguf::Gguf;
use nextcuda_qwen35::engine;
use nextcuda_llm::{At, EngineOption};
use nextcuda_llm::{Checkpoint, CheckpointState, Decoder, DecoderState, Engine, EngineKind, GpuInfo, LoadOptions, Result, Sampler, Session, SessionState,
                   Tap};

// the core keeps its own copies of the picture token's layout (it knows no contract); they must be the contract's
const _: () = assert!(engine::PICTURE == nextcuda_llm::PICTURE_TOKEN && engine::PICTURE_INDEX_BITS == nextcuda_llm::PICTURE_INDEX_BITS);

pub const ARCH: &str = "qwen35";

pub const OPTIONS: &[EngineOption] = &[
    EngineOption { name: "chunk", env: "NS_Q35_CHUNK", value: "TOKENS", help: "a prompt pass's rows (default 2048 when it fits beside the weights and a session, else 1024 or 512)", at: At::Load },
    EngineOption { name: "mtp", env: "NS_Q35_MTP", value: "FILE", help: "the MTP draft block's GGUF (default: the one in an MTP/ folder beside the model)", at: At::Load },
    EngineOption { name: "drafts", env: "NS_Q35_DRAFTS", value: "N", help: "tokens drafted a step with MTP (default 3, at most 7; 0: none)", at: At::Load },
];

pub fn kind() -> EngineKind {
    EngineKind { archs: &[ARCH], name: "Qwen3.5 / Qwen3.8 dense (gated DeltaNet + gated full attention, SwiGLU: Qwen3.8-27B ...)", load, info,
                 kernels: None, chat: nextcuda_tok::qwen38_chat, options: OPTIONS, prepare: None }
}

fn info(f: &Gguf) -> std::result::Result<String, String> {
    let g = engine::Geometry::read(f)?;
    let full = (0..g.layers).filter(|l| g.full(*l)).count();
    Ok(format!("{ARCH}: {} layers of {} ({full} gated full attention: {} / {} heads of {}, rotary {} at {}; {} gated DeltaNet: {} / {} heads of {}), \
                a dense SwiGLU of {}, vocabulary {}, context {}", g.layers, g.n_embd, g.heads, g.heads_kv, g.head, g.n_rot, g.theta,
               g.layers - full, g.k_heads, g.v_heads, g.state, g.n_ff, g.vocab, g.ctx))
}

fn load<'g>(f: &'g Gguf, gpus: &[Arc<nextcuda_core::Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn Engine + 'g>> {
    Ok(Box::new(Qwen35(engine::Qwen35::load(f, gpus, o.draft, o.kv, log)?)))
}

// The contract's traits are implemented on these newtypes: the core's types and the traits are both foreign here.
struct Qwen35<'g>(engine::Qwen35<'g>);
struct Sess(engine::Session);
struct Dec(engine::Decoder);
struct Ck(engine::Checkpoint);

impl SessionState for Sess {
    fn pos(&self) -> usize {
        self.0.pos
    }
    fn max_ctx(&self) -> usize {
        self.0.max_ctx
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl DecoderState for Dec {
    fn drafts(&self) -> (u64, u64) {
        (self.0.drafted, self.0.accepted)
    }
    fn pending(&mut self) -> Option<u32> {
        self.0.pending()
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl CheckpointState for Ck {
    fn pos(&self) -> usize {
        self.0.pos
    }
    fn bytes(&self) -> usize {
        self.0.bytes
    }
    fn write_to(&self, w: &mut dyn std::io::Write) -> std::io::Result<()> {
        self.0.write_to(w)
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn sess(s: &Session) -> Result<&engine::Session> {
    s.get::<Sess>().map(|s| &s.0).ok_or_else(|| nextcuda_core::Error("a session of another engine".into()))
}
fn sess_mut(s: &mut Session) -> Result<&mut engine::Session> {
    s.get_mut::<Sess>().map(|s| &mut s.0).ok_or_else(|| nextcuda_core::Error("a session of another engine".into()))
}

impl Engine for Qwen35<'_> {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn load_seconds(&self) -> f64 {
        self.0.load_seconds()
    }
    fn load_bytes(&self) -> u64 {
        self.0.load_bytes()
    }
    fn has_draft(&self) -> bool {
        self.0.has_draft()
    }
    fn prefill_chunk(&self) -> usize {
        self.0.prefill_chunk()
    }
    fn max_verify(&self) -> usize {
        self.0.max_verify()
    }
    fn cache_fingerprint(&self) -> String {
        // checkpoints hold the draft block's cache too when it is loaded
        // nsq35-2: a checkpoint carries the session's rotary shift (pictures); -3: the DeltaNet state value-column first
        // (each GPU's cache type: f16-kv, q8-kv, f16,q8-kv ... - NS_Q35_KV)
        let kv = format!("{}-kv", self.0.kv_types());
        if self.has_draft() { format!("qwen35 {kv} nsq35-3 mtp") } else { format!("qwen35 {kv} nsq35-3") }
    }
    fn session(&self, max_ctx: usize) -> Result<Session> {
        Ok(Session::new(Sess(self.0.session(max_ctx)?)))
    }
    fn reset_session(&self, s: &mut Session) -> Result<()> {
        self.0.reset_session(sess_mut(s)?)
    }
    fn copy_session(&self, dst: &mut Session, src: &Session) -> Result<()> {
        let src = sess(src)?;
        self.0.copy_session(sess_mut(dst)?, src)
    }
    fn feed(&self, s: &mut Session, tokens: &[u32], tap: Tap) -> Result<Vec<f32>> {
        Ok(self.0.feed_until(sess_mut(s)?, tokens, &|_| false, tap)?.1)
    }
    fn feed_until(&self, s: &mut Session, tokens: &[u32], stop: &(dyn Fn(usize) -> bool + Sync), tap: Tap) -> Result<(usize, Vec<f32>)> {
        self.0.feed_until(sess_mut(s)?, tokens, stop, tap)
    }
    fn forward(&self, s: &mut Session, tokens: &[u32], tap: Tap) -> Result<Vec<f32>> {
        Ok(self.0.forward_rows(sess_mut(s)?, tokens, 1, tap)?.pop().unwrap_or_default())
    }
    fn forward_rows(&self, s: &mut Session, tokens: &[u32], n_out: usize, tap: Tap) -> Result<Vec<Vec<f32>>> {
        self.0.forward_rows(sess_mut(s)?, tokens, n_out, tap)
    }
    fn rollback(&self, s: &mut Session, keep: usize) -> Result<()> {
        self.0.rollback(sess_mut(s)?, keep)
    }
    fn forward_batch(&self, sessions: &mut [&mut Session], tokens: &[u32], tap: Tap) -> Result<Vec<Vec<f32>>> {
        let mut own: Vec<&mut engine::Session> = Vec::with_capacity(sessions.len());
        for s in sessions.iter_mut() {
            own.push(&mut s.get_mut::<Sess>().ok_or_else(|| nextcuda_core::Error("a session of another engine".into()))?.0);
        }
        self.0.forward_batch(&mut own, tokens, tap)
    }
    fn decoder(&self, logits: Vec<f32>, draft: bool) -> Decoder {
        Decoder::new(Dec(engine::Decoder::new(logits, draft)))
    }
    fn decoder_after(&self, next: u32, draft: bool) -> Decoder {
        Decoder::new(Dec(engine::Decoder::after(next, draft)))
    }
    fn step(&self, s: &mut Session, d: &mut Decoder, smp: &mut dyn Sampler, tap: Tap) -> Result<Vec<u32>> {
        let d = &mut d.get_mut::<Dec>().ok_or_else(|| nextcuda_core::Error("a decoder of another engine".into()))?.0;
        self.0.step(sess_mut(s)?, d, &mut |l| smp.sample(l), tap)
    }
    fn save(&self, s: &Session) -> Result<Checkpoint> {
        Ok(Checkpoint::new(Ck(self.0.save(sess(s)?)?)))
    }
    fn restore(&self, s: &mut Session, ck: &Checkpoint) -> Result<()> {
        let ck = &ck.get::<Ck>().ok_or_else(|| nextcuda_core::Error("a checkpoint of another engine".into()))?.0;
        self.0.restore(sess_mut(s)?, ck)
    }
    fn read_checkpoint(&self, r: &mut dyn std::io::Read) -> std::io::Result<Checkpoint> {
        Ok(Checkpoint::new(Ck(engine::Checkpoint::read_from(r)?)))
    }
    fn gpu_info(&self) -> Vec<GpuInfo> {
        self.0.gpu_info().into_iter()
            .map(|c| GpuInfo { index: c.index, name: c.name, pci: c.pci, total: c.total, free: c.free, layers: c.layers, expert_slots: 0, host_slots: 0 })
            .collect()
    }
    fn sees(&self) -> bool {
        self.0.sees()
    }
    fn add_picture(&self, key: u32, rgb: &[u8], w: usize, h: usize) -> Result<(usize, usize)> {
        self.0.add_picture_rgb(key, rgb, w, h)
    }
    fn report(&self, _tokens: usize) -> Vec<String> {
        self.0.profile_lines()
    }
}
