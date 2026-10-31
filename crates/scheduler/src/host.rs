//! What the host (the box's front door) provides. The scheduler decides; it never touches a process, a socket or a card.
//!
//! | trait method | what it is on the box |
//! | --- | --- |
//! | `now`, `sleep` | the clock (a fake one in tests: sleeping advances it) |
//! | `snapshot` | the card sampler + the servers' states + the registry + pins and holds, one consistent view |
//! | `server` | one server's state read fresh (busy / holding), not from a snapshot |
//! | `start`, `stop` | start a model on a card in a server slot and wait until it answers; stop a slot's server |
//! | `free_video`, `free_chat` | let go of the video daemon's idle worker on a card; unload the idle chat model |
//! | `try_lock_kind` | the per-kind lock; None when another request holds it |
//! | `wait_loaded` | wait until a loading server answers |
//! | `note`, `cancelled` | a job's stage and a cancel flag |
//! | `learn_ask` | a server refused for want of memory: remember what it asked, say whether that is more than planned |

use crate::room::{Idleness, Server, Snapshot};
use crate::Fail;

/// Whatever holds a lock until it is dropped (a mutex guard)
pub trait Guard {}
impl<T> Guard for T {}

pub trait Host: Idleness {
    /// Seconds on the host's clock
    fn now(&self) -> f64;
    /// Wait; a fake clock advances by it
    fn sleep(&self, secs: f64);
    /// The box now
    fn snapshot(&self) -> Snapshot;
    /// One server of a kind, read fresh
    fn server(&self, kind: &str, port: u16) -> Option<Server>;
    /// `id` started in server `slot` (listening on `port`) on card `gpu`, answering (or the error: a refusal for memory
    /// says "needs about X GiB on <card>; Y GiB are free")
    fn start(&self, kind: &str, id: &str, slot: usize, port: u16, gpu: usize) -> Result<(), Fail>;
    /// Stop the server in `slot`; `label` is what to call it in a log
    fn stop(&self, kind: &str, slot: usize, port: u16, label: &str) -> Result<(), Fail>;
    /// Let the video daemon's idle worker on `gpu` go, for model `for_id`
    fn free_video(&self, gpu: usize, for_id: &str) -> Result<(), Fail>;
    /// Unload the idle chat model `chat`
    fn free_chat(&self, chat: &str) -> Result<(), Fail>;
    /// The kind's lock, when it is free (the guard is dropped to release it)
    fn try_lock_kind<'a>(&'a self, kind: &str) -> Option<Box<dyn Guard + 'a>>;
    /// Wait until the loading server on `port` answers as `id`
    fn wait_loaded(&self, port: u16, id: &str) -> Result<(), Fail>;
    /// A job's stage and what it waits for
    fn note(&self, job: Option<&str>, stage: &str, why: &str);
    /// Whether a cancel was asked for the job
    fn cancelled(&self, job: &str) -> bool;
    /// A server refused to load for want of memory on `gpu`, planned for `want` MiB: remembered; whether it is more than planned
    fn learn_ask(&self, id: &str, err: &str, gpu: usize, want: u64) -> bool;
}
