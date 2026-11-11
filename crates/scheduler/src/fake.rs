//! A fake host on a fake clock: cards with memory, servers that start and stop by writing a field, requests that finish at
//! a scripted time. The scenario suite (tests/scenarios.rs) runs on it, and so can a port's own tests.
//!
//! The clock moves only when the scheduler sleeps (or the script calls `advance`); `start` takes `load_s` seconds on it.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;

use crate::host::Host;
use crate::profile::Profile;
use crate::room::{Card, Idleness, ModelInfo, Proc, Server, Snapshot};
use crate::rsvp::{self, Hold};
use crate::Fail;

/// MiB a card always shows used (the driver's own context)
const BASE_USED_MB: u64 = 300;

struct FakeServer {
    kind: String,
    slot: usize,
    port: u16,
    gpu: usize,
    model: Option<String>,
    /// busy until this time on the fake clock
    busy_until: f64,
    inflight: usize,
    /// started while `unseen` was on: no process and no memory on its card yet (weights still in host memory)
    unseen: bool,
}

struct State {
    now: f64,
    cards: Vec<u64>,
    servers: Vec<FakeServer>,
    models: BTreeMap<String, ModelInfo>,
    vram_mb: BTreeMap<String, u64>,
    /// a split model's memory on each of its cards but the first
    rest_mb: BTreeMap<String, u64>,
    /// memory taken on a card that no snapshot shows, until a time on the clock: (card, MiB, until) - another program
    /// that grew between the sample and a start
    hidden: Vec<(usize, u64, f64)>,
    pinned: BTreeSet<String>,
    holds: BTreeMap<String, Hold>,
    cancelled: BTreeSet<String>,
    video_idle: bool,
    /// the chat model, if loaded: (id, mb, gpu)
    chat: Option<(String, u64, usize)>,
    chat_idle: bool,
    locked: BTreeSet<String>,
    load_s: f64,
    /// servers started from now on do not show on their card (see FakeServer::unseen)
    unseen: bool,
    /// what the host did, in order: "start audio a slot 0 gpu 0", "stop audio slot 0 (...)"
    log: Vec<String>,
    notes: Vec<(String, String)>,
}

pub struct FakeHost {
    st: Mutex<State>,
}

impl FakeHost {
    /// Cards of the given sizes (MiB) and servers: `(kind, count)` each on its own port, all on card 0 until a model starts
    /// on another
    pub fn new(cards_mb: &[u64], servers: &[(&str, usize)]) -> FakeHost {
        let mut v = Vec::new();
        let mut port = 9000;
        for (kind, n) in servers {
            for slot in 0..*n {
                v.push(FakeServer { kind: kind.to_string(), slot, port, gpu: 0, model: None, busy_until: 0.0, inflight: 0, unseen: false });
                port += 1;
            }
        }
        FakeHost { st: Mutex::new(State {
            now: 1000.0, cards: cards_mb.to_vec(), servers: v, models: BTreeMap::new(), vram_mb: BTreeMap::new(), rest_mb: BTreeMap::new(), hidden: Vec::new(), pinned: BTreeSet::new(), holds: BTreeMap::new(),
            cancelled: BTreeSet::new(), video_idle: true, chat: None, chat_idle: true, locked: BTreeSet::new(), load_s: 10.0, unseen: false, log: Vec::new(), notes: Vec::new(),
        }) }
    }

    /// A model of `kind` that takes `vram_mb` on a card, allowed on `gpus`, with a profile
    pub fn add(&self, id: &str, kind: &str, vram_mb: u64, gpus: &[usize], profile: Profile) -> ModelInfo {
        let m = ModelInfo { id: id.into(), kind: kind.into(), enabled: true, gpus: gpus.to_vec(), profile, split: false };
        let mut st = self.st.lock().unwrap();
        st.models.insert(id.into(), m.clone());
        st.vram_mb.insert(id.into(), vram_mb);
        m
    }

    /// A split model: one server takes all of `gpus`, `first_mb` on the card it is started on and `rest_mb` on each other
    pub fn add_split(&self, id: &str, kind: &str, first_mb: u64, rest_mb: u64, gpus: &[usize]) -> ModelInfo {
        let mut m = self.add(id, kind, first_mb, gpus, Profile::default());
        m.split = true;
        let mut st = self.st.lock().unwrap();
        st.models.insert(id.into(), m.clone());
        st.rest_mb.insert(id.into(), rest_mb);
        m
    }

    /// `mb` taken on card `g` for `secs` that no snapshot shows (a start there finds it)
    pub fn hide(&self, g: usize, mb: u64, secs: f64) {
        let mut st = self.st.lock().unwrap();
        let until = st.now + secs;
        st.hidden.push((g, mb, until));
    }

    pub fn set_profile(&self, id: &str, p: Profile) {
        self.st.lock().unwrap().models.get_mut(id).unwrap().profile = p;
    }

    pub fn disable(&self, id: &str) {
        self.st.lock().unwrap().models.get_mut(id).unwrap().enabled = false;
    }

    pub fn pin(&self, id: &str, on: bool) {
        let mut st = self.st.lock().unwrap();
        if on { st.pinned.insert(id.into()); } else { st.pinned.remove(id); }
    }

    /// A request ended for `id`: its hold per rsvp.rs
    pub fn hold_after(&self, id: &str, granted: f64) {
        let mut st = self.st.lock().unwrap();
        let now = st.now;
        let next = rsvp::extend(st.holds.get(id).copied(), now, granted);
        st.holds.insert(id.into(), next);
    }

    /// The server holding `id` is busy for `secs` more (its own work), and `inflight` requests have been sent to it
    pub fn busy(&self, id: &str, secs: f64, inflight: usize) {
        let mut st = self.st.lock().unwrap();
        let until = st.now + secs;
        for s in st.servers.iter_mut().filter(|s| s.model.as_deref() == Some(id)) {
            s.busy_until = until;
            s.inflight = inflight;
        }
    }

    pub fn cancel(&self, job: &str) {
        self.st.lock().unwrap().cancelled.insert(job.into());
    }

    /// Servers started from now on show no process and no memory on their card, as a real one in its first seconds
    pub fn start_unseen(&self, on: bool) {
        self.st.lock().unwrap().unseen = on;
    }

    pub fn set_load_s(&self, s: f64) {
        self.st.lock().unwrap().load_s = s;
    }

    pub fn lock_kind(&self, kind: &str) {
        self.st.lock().unwrap().locked.insert(kind.into());
    }

    /// Put the chat model on a card (loaded), idle or not
    pub fn chat(&self, id: &str, mb: u64, gpu: usize, idle: bool) {
        let mut st = self.st.lock().unwrap();
        st.chat = Some((id.into(), mb, gpu));
        st.chat_idle = idle;
    }

    pub fn advance(&self, secs: f64) {
        self.st.lock().unwrap().now += secs;
    }

    pub fn log(&self) -> Vec<String> {
        self.st.lock().unwrap().log.clone()
    }

    pub fn count(&self, needle: &str) -> usize {
        self.st.lock().unwrap().log.iter().filter(|l| l.contains(needle)).count()
    }

    pub fn notes(&self) -> Vec<(String, String)> {
        self.st.lock().unwrap().notes.clone()
    }

    /// The model each server of a kind holds, by slot
    pub fn loaded(&self, kind: &str) -> Vec<Option<String>> {
        self.st.lock().unwrap().servers.iter().filter(|s| s.kind == kind).map(|s| s.model.clone()).collect()
    }

    pub fn held_s(&self, id: &str) -> f64 {
        let st = self.st.lock().unwrap();
        rsvp::left(st.holds.get(id), st.now)
    }

    /// What server `s` holds on card `g`: its model on the card it was started on, a split model's share on its others
    fn held_on(st: &State, s: &FakeServer, g: usize) -> u64 {
        let Some(m) = s.model.as_ref().filter(|_| !s.unseen) else { return 0 };
        if s.gpu == g {
            return st.vram_mb.get(m).copied().unwrap_or(0);
        }
        match (st.rest_mb.get(m), st.models.get(m)) {
            (Some(mb), Some(info)) if info.gpus.contains(&g) => *mb,
            _ => 0,
        }
    }

    /// What a start on card `g` finds taken: what a snapshot shows and the hidden memory not run out
    fn used_really(st: &State, g: usize) -> u64 {
        FakeHost::used(st, g) + st.hidden.iter().filter(|h| h.0 == g && h.2 > st.now).map(|h| h.1).sum::<u64>()
    }

    fn used(st: &State, g: usize) -> u64 {
        BASE_USED_MB
            + st.servers.iter().map(|s| FakeHost::held_on(st, s, g)).sum::<u64>()
            + st.chat.as_ref().filter(|c| c.2 == g).map_or(0, |c| c.1)
    }

    fn server_of(st: &State, s: &FakeServer) -> Server {
        Server { slot: s.slot, port: s.port, model: s.model.clone(), loading: false, busy: s.busy_until > st.now || s.inflight > 0, inflight: s.inflight, gpu: s.model.is_some().then_some(s.gpu) }
    }
}

impl Idleness for FakeHost {
    fn video_idle(&self) -> bool {
        self.st.lock().unwrap().video_idle
    }
    fn chat_idle(&self) -> Option<String> {
        let st = self.st.lock().unwrap();
        let (id, _, _) = st.chat.clone()?;
        let kept = rsvp::kept_from(&st.pinned, &st.holds, st.now);
        (st.chat_idle && !kept.contains(&id)).then_some(id)
    }
}

impl Host for FakeHost {
    fn now(&self) -> f64 {
        self.st.lock().unwrap().now
    }

    fn sleep(&self, secs: f64) {
        self.st.lock().unwrap().now += secs;
    }

    fn snapshot(&self) -> Snapshot {
        let st = self.st.lock().unwrap();
        let mut snap = Snapshot::default();
        for (g, total) in st.cards.iter().enumerate() {
            let mut procs = Vec::new();
            for s in &st.servers {
                let mb = FakeHost::held_on(&st, s, g);
                if let (Some(m), true) = (&s.model, mb > 0 || (s.gpu == g && !s.unseen)) {
                    procs.push(Proc { kind: Some(s.kind.clone()), model: m.clone(), port: Some(s.port), mb });
                }
            }
            if let Some((id, mb, cg)) = &st.chat {
                if *cg == g {
                    procs.push(Proc { kind: Some("llm".into()), model: id.clone(), port: None, mb: *mb });
                }
            }
            snap.cards.push(Card { index: g, name: format!("card {g}"), total_mb: *total, used_mb: FakeHost::used(&st, g), procs });
        }
        for s in &st.servers {
            snap.servers.entry(s.kind.clone()).or_default().push(FakeHost::server_of(&st, s));
        }
        snap.models = st.models.clone();
        snap.pinned = st.pinned.clone();
        snap.kept = rsvp::kept_from(&st.pinned, &st.holds, st.now);
        snap.held_left = st.holds.iter().map(|(k, h)| (k.clone(), rsvp::left(Some(h), st.now))).filter(|(_, l)| *l > 0.0).collect();
        snap
    }

    fn server(&self, kind: &str, port: u16) -> Option<Server> {
        let st = self.st.lock().unwrap();
        st.servers.iter().find(|s| s.kind == kind && s.port == port).map(|s| FakeHost::server_of(&st, s))
    }

    fn start(&self, kind: &str, id: &str, slot: usize, _port: u16, gpu: usize) -> Result<(), Fail> {
        let mut st = self.st.lock().unwrap();
        // each card it takes (a split model's: all of its own, `gpu` first) must have its share free, as a real engine checks
        let taken = st.models.get(id).map_or(vec![gpu], |m| m.cards_taken(gpu));
        for (i, g) in taken.into_iter().enumerate() {
            let need = if i == 0 { st.vram_mb.get(id) } else { st.rest_mb.get(id) }.copied().unwrap_or(0);
            let free = st.cards[g].saturating_sub(FakeHost::used_really(&st, g));
            if free < need {
                st.log.push(format!("refused {kind} {id} gpu {gpu}: short on card {g}"));
                return Err((502, format!("{id} needs about {:.1} GiB on card {g}; {:.1} GiB are free", need as f64 / 1024.0, free as f64 / 1024.0)));
            }
        }
        let load = st.load_s;
        let unseen = st.unseen;
        let line = format!("start {kind} {id} slot {slot} gpu {gpu}");
        let s = st.servers.iter_mut().find(|s| s.kind == kind && s.slot == slot).ok_or((500, "no such slot".to_string()))?;
        s.model = Some(id.into());
        s.gpu = gpu;
        s.unseen = unseen;
        s.busy_until = 0.0;
        s.inflight = 0;
        st.now += load;
        st.log.push(line);
        Ok(())
    }

    fn stop(&self, kind: &str, slot: usize, _port: u16, label: &str) -> Result<(), Fail> {
        let mut st = self.st.lock().unwrap();
        if let Some(s) = st.servers.iter_mut().find(|s| s.kind == kind && s.slot == slot) {
            s.model = None;
            s.busy_until = 0.0;
            s.inflight = 0;
        }
        st.log.push(format!("stop {kind} slot {slot} ({label})"));
        Ok(())
    }

    fn free_video(&self, gpu: usize, for_id: &str) -> Result<(), Fail> {
        self.st.lock().unwrap().log.push(format!("free video gpu {gpu} for {for_id}"));
        Ok(())
    }

    fn free_chat(&self, chat: &str) -> Result<(), Fail> {
        let mut st = self.st.lock().unwrap();
        st.chat = None;
        st.log.push(format!("unload chat {chat}"));
        Ok(())
    }

    fn try_lock_kind<'a>(&'a self, kind: &str) -> Option<Box<dyn crate::host::Guard + 'a>> {
        if self.st.lock().unwrap().locked.contains(kind) { None } else { Some(Box::new(())) }
    }

    fn wait_loaded(&self, _port: u16, _id: &str) -> Result<(), Fail> {
        Ok(())
    }

    fn note(&self, job: Option<&str>, stage: &str, why: &str) {
        if let Some(j) = job {
            self.st.lock().unwrap().notes.push((format!("{j}:{stage}"), why.to_string()));
        }
    }

    fn cancelled(&self, job: &str) -> bool {
        self.st.lock().unwrap().cancelled.contains(job)
    }

    fn learn_ask(&self, id: &str, _err: &str, gpu: usize, _want: u64, rest: bool) -> bool {
        self.st.lock().unwrap().log.push(format!("learn {id} gpu {gpu}{}", if rest { " (rest)" } else { "" }));
        false
    }
}
