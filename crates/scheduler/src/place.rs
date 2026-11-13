//! `place`: one try at a server for a model, over a [`Host`]. The caller holds the kind's lock while it lasts and loops
//! (sleeping, noting the wait) until it has a server or its wait runs out.

use std::collections::BTreeMap;

use crate::host::Host;
use crate::room::{card_free, idle_servers, card_of, extra_copy_allowed, no_room, pick_card, room_plan, Card, Free, Holder, ModelInfo, Server, Snapshot, JOB_ROOM_WAIT_S, MARGIN_MB, TIE_MB};
use crate::Fail;

/// What to place
pub struct PlaceReq<'a> {
    pub model: &'a ModelInfo,
    /// the job asking (its stage is noted, a cancel ends a wait)
    pub job: Option<&'a str>,
    /// the memory the model is vouched for on each card, MiB (a split model's: on the card its server is started on)
    pub need_mb: &'a BTreeMap<usize, u64>,
    /// a split model's memory on each of its other cards, MiB (a card without an entry: its `need_mb`)
    pub rest_mb: &'a BTreeMap<usize, u64>,
    /// MiB a server once refused for (learned): at least this is planned for
    pub asked_mb: u64,
    /// MiB a split model's server once refused for on one of its other cards (learned)
    pub asked_rest_mb: u64,
}

/// The answer of one try
#[derive(Debug, PartialEq)]
pub enum Outcome {
    /// a server of the model to send the request to (the host leases its port)
    Placed { slot: usize, port: u16 },
    /// not yet: why (no card of its has the room, even with the idle models there stopped; a server stopped for it ...)
    NotYet(String),
}

/// Wait (30 s at most) until card `g` has `want` MiB free: a stopped process lets go of its memory in a moment. Whether
/// it came.
pub fn wait_room(host: &dyn Host, g: usize, want: u64) -> bool {
    let t0 = host.now();
    while host.now() - t0 < 30.0 {
        if card_free(&host.snapshot().cards, g) >= want {
            return true;
        }
        host.sleep(1.0);
    }
    false
}

/// Stop the idle holders of a plan (for `id`; the kind `kind` asking holds its own lock; another kind's is tried, and a
/// server whose lock is taken is left: the next try plans again)
pub fn free_plan(host: &dyn Host, plan: &[Holder], kind: &str, id: &str) -> Result<(), Fail> {
    for h in plan {
        match &h.what {
            Free::Server(k, slot) => {
                let guard = if k == kind { None } else {
                    match host.try_lock_kind(k) {
                        Some(g) => Some(g),
                        None => return Err((503, format!("{} is changing its {k} server; {id} tries again", h.model))),
                    }
                };
                // still idle (it may have taken a request since)
                let snap = host.snapshot();
                let x = snap.servers_of(k).iter().find(|x| x.slot == *slot).cloned();
                if let Some(x) = x.filter(|x| !x.busy && !x.loading && x.model.is_some()) {
                    host.stop(k, *slot, x.port, &format!("{k} stop ({}, idle, for {id})", h.model))?;
                }
                drop(guard);
            }
            Free::Video(g) => {
                if host.video_idle() {
                    host.free_video(*g, id)?;
                }
            }
            Free::Chat(chat) => {
                if host.chat_idle().as_deref() == Some(chat.as_str()) {
                    host.free_chat(chat)?;
                }
            }
        }
    }
    Ok(())
}

/// Stop the server `x` when it holds a model (one of the kind, its lock held), for `id`
fn stop_slot(host: &dyn Host, kind: &str, x: &Server, id: &str) -> Result<(), Fail> {
    if let Some(other) = host.server(kind, x.port).and_then(|s| s.model) {
        host.stop(kind, x.slot, x.port, &format!("{kind} stop ({other}, for {id})"))?;
    }
    Ok(())
}

/// Server `x` (another model's) after its requests: waited for, with the kind's lock held, so no new request is sent to
/// it meanwhile; then stopped
fn drain(host: &dyn Host, x: &Server, kind: &str, id: &str, job: Option<&str>) -> Result<(), Fail> {
    let Some(other) = &x.model else { return Ok(()) };
    let t0 = host.now();
    while host.server(kind, x.port).is_some_and(|s| s.busy || s.inflight > 0) {
        if job.is_some_and(|j| host.cancelled(j)) {
            return Err((499, format!("cancelled while it waited for {other}'s requests")));
        }
        if host.now() - t0 > JOB_ROOM_WAIT_S as f64 {
            return Err((503, format!("{other} has been busy for 30 minutes; {id} waits for it")));
        }
        host.note(job, "waiting", &format!("server {} finishing {other}'s requests, then {id} loads there", x.slot + 1));
        host.sleep(2.0);
    }
    stop_slot(host, kind, x, id)
}

/// One try at a server for the model (the kind's lock held while it lasts): a server, or why there is none yet (no card
/// of its has the room, even with the idle models there stopped)
pub fn place(host: &dyn Host, req: &PlaceReq) -> Result<Outcome, Fail> {
    let m = req.model;
    let (id, kind, job) = (m.id.as_str(), m.kind.as_str(), req.job);
    if !m.enabled {
        return Err((409, format!("{id} is disabled (enable it)")));
    }
    let gpus = &m.gpus;
    let snap = host.snapshot();
    let sv: Vec<Server> = snap.servers_of(kind).to_vec();
    let kept = |x: &Server| x.model.as_ref().is_some_and(|mm| snap.kept.contains(mm));
    let mine: Vec<&Server> = sv.iter().filter(|x| x.model.as_deref() == Some(id)).collect();
    let want = |g: usize| (req.need_mb.get(&g).copied().unwrap_or(0) + MARGIN_MB).max(req.asked_mb);
    let want_rest = |g: usize| (req.rest_mb.get(&g).or(req.need_mb.get(&g)).copied().unwrap_or(0) + MARGIN_MB).max(req.asked_rest_mb);
    // what each card a server started on `g` takes must have free: `g` its share, a split model's other cards theirs
    // (2026-11-11: MiniMax Music 3 was planned on the 5070 alone - its idle TTS stopped for it - and started there with
    // its flow stage on the 3060, which ACE-Step's song held: 36 starts died "needs about 6.8 GiB on RTX 3060")
    let wants = |g: usize| -> Vec<(usize, u64)> {
        m.cards_taken(g).into_iter().enumerate().map(|(i, c)| (c, if i == 0 { want(c) } else { want_rest(c) })).collect()
    };
    // the idle holders to stop so that every card a start on `g` takes has its room (each less what `less` gives back
    // there), or no way now
    let plan_on = |g: usize, light: bool, less: &dyn Fn(usize) -> u64| -> Option<Vec<Holder>> {
        let mut all: Vec<Holder> = Vec::new();
        for (c, w) in wants(g) {
            for h in room_plan(&snap, host, c, w.saturating_sub(less(c)), kind, light, Some(m))? {
                if !all.iter().any(|a| a.what == h.what) {
                    all.push(h);
                }
            }
        }
        Some(all)
    };
    let nothing = |_: usize| 0;
    let cards = &snap.cards;
    // a server that holds another model and could go now
    let idle_other = |x: &&Server| x.model.is_some() && x.model.as_deref() != Some(id) && !kept(x) && x.idle();
    // another copy of it on one of the cards `allowed` lets (of its own, where it may have one more) that has the room
    // now, or once idle picture and sound servers there stop; in a slot no server holds, or one an idle model holds.
    // None: no such card or slot, or the start failed (the requests use a copy it has)
    let start_copy = |allowed: &dyn Fn(usize) -> bool| -> Option<Outcome> {
        let on: Vec<usize> = mine.iter().filter_map(|x| card_of(cards, x)).collect();
        // copies per card: extra ones up to the profile's recommended count (the default 1: one a card), never past its max
        let per_card = m.profile.concurrency.recommended_copies.min(m.profile.concurrency.max_copies_per_card) as usize;
        let x = sv.iter().find(|x| x.model.is_none()).or_else(|| sv.iter().find(idle_other))?;
        let (gpu, plan) = pick_card(gpus.iter().copied().filter(|g| allowed(*g) && extra_copy_allowed(&on, *g, per_card))
            .map(|g| (g, card_free(cards, g), plan_on(g, true, &nothing))))?;
        // (a plan another request's lock stands in the way of: no copy now, the requests share the first)
        let started = free_plan(host, &plan, kind, id).and_then(|_| {
            stop_slot(host, kind, x, id)?;
            if !wants(gpu).into_iter().all(|(c, w)| wait_room(host, c, w)) {
                return Err((503, format!("GPU {gpu} did not free the room")));
            }
            host.note(job, "loading", &format!("a second copy of {id} loading on {} (server {})", on_cards(&m.cards_taken(gpu)), x.slot + 1));
            host.start(kind, id, x.slot, x.port, gpu).inspect_err(|(_, e)| {
                short_of_room(host, cards, m, gpu, e, &wants(gpu));
            })
        });
        match started {
            Ok(()) => Some(Outcome::Placed { slot: x.slot, port: x.port }),
            Err((_, e)) => {
                eprintln!("{id}: a second copy did not start ({e}); the requests share the first");
                None
            }
        }
    };
    // a server of it with nothing to do - one on a card no other work saturates first; when every such server's card is
    // taken by other work (a busy server of another model there, the video worker), a copy on a card of its that is not,
    // where one has the room now or once idle picture and sound servers there stop: two heavy jobs do not share one
    // card's compute while another card of the model sits idle (hardac1, 2026-11-13: YuE2 stayed loaded on the card a
    // FlashVSR upscale took and made no progress for 15 minutes beside it, the other card idle; nextsycl cd4da5c)
    let contended = |x: &Server| !x.loading && card_of(cards, x).is_some_and(|g| card_contended(&snap, host, g, x.port));
    let free_now: Vec<&&Server> = mine.iter().filter(|x| !x.busy && x.inflight == 0).collect();
    if let Some(x) = free_now.iter().find(|x| !contended(x)).or_else(|| free_now.first()).copied() {
        if contended(x) {
            let taken: Vec<usize> = mine.iter().filter_map(|x| card_of(cards, x)).collect();
            // every card a copy started on `g` takes (a split model's too) quiet
            let quiet = |g: usize| !taken.contains(&g) && m.cards_taken(g).into_iter().all(|c| !card_contended(&snap, host, c, 0));
            if let Some(o) = start_copy(&quiet) {
                return Ok(o);
            }
        }
        if x.loading {
            host.note(job, "loading", &format!("{id} loading (server {})", x.slot + 1));
            host.wait_loaded(x.port, id)?;
        }
        return Ok(Outcome::Placed { slot: x.slot, port: x.port });
    }
    if !mine.is_empty() {
        // its servers all have work: another copy on a card of its that has the room now (or once idle picture and sound
        // servers there stop) - else the copy with the fewest requests (its server queues it)
        if let Some(o) = start_copy(&|_| true) {
            return Ok(o);
        }
        let x = mine.iter().min_by_key(|x| x.inflight).unwrap();
        if x.loading {
            host.note(job, "loading", &format!("{id} loading (server {})", x.slot + 1));
            host.wait_loaded(x.port, id)?;
        }
        return Ok(Outcome::Placed { slot: x.slot, port: x.port });
    }
    // not loaded: a card of its with the room now (the one with the most free), else the first that has it once idle
    // models there go (picture and sound servers, an idle video worker, a chat model not kept); a slot no server holds,
    // else one an idle model holds, else one a busy model of the kind holds, after its requests. A split model's other
    // cards must have their room too, or the card is no choice.
    let plan = pick_card(gpus.iter().map(|&g| (g, card_free(cards, g), plan_on(g, false, &nothing))));
    let empty = sv.iter().find(|x| x.model.is_none());
    let idle = sv.iter().filter(idle_other).min_by_key(|x| card_of(cards, x).is_none_or(|g| Some(g) != plan.as_ref().map(|p| p.0)));
    let on_mine = |x: &&Server| x.model.is_some() && !kept(x) && card_of(cards, x).is_some_and(|g| gpus.contains(&g));
    let no_room_now = || {
        let mut why = no_room(&snap, gpus, m, want(gpus[0]));
        if m.split {
            why += &format!(" on the first of its cards and ~{:.1} GiB on each other (its stages split over them)", want_rest(gpus[gpus.len() - 1]) as f64 / 1024.0);
        }
        why
    };
    let Some((gpu, plan)) = plan else {
        // no room even then: a busy server of another model of the kind on a card of its goes after its requests (it
        // holds the room), then the next try loads it
        let held = |x: &Server, c: usize| cards.iter().find(|k| k.index == c).into_iter().flat_map(|k| k.procs.iter())
            .filter(|p| p.port == Some(x.port)).map(|p| p.mb).sum::<u64>();
        let frees = |x: &&Server| card_of(cards, x).is_some_and(|g| plan_on(g, false, &|c| held(x, c)).is_some());
        if let Some(x) = sv.iter().filter(on_mine).filter(frees).min_by_key(|x| x.inflight) {
            drain(host, x, kind, id, job)?;
            return Ok(Outcome::NotYet(format!("server {} stopped for {id}", x.slot + 1)));
        }
        return Ok(Outcome::NotYet(no_room_now()));
    };
    let x = match (empty, idle) {
        (Some(e), _) => e,
        (None, Some(i)) => i,
        // every server busy with another model: the one with the fewest requests, after them
        _ => {
            // (a kept model's server is never the one: every server kept, no room)
            let Some(x) = sv.iter().filter(|x| x.model.is_some() && !kept(x)).min_by_key(|x| (!on_mine(x) as u8, x.inflight)) else {
                return Ok(Outcome::NotYet(no_room_now()));
            };
            drain(host, x, kind, id, job)?;
            x
        }
    };
    stop_slot(host, kind, x, id)?;
    // an idle model whose kind is busy changing its own server stays: the next try plans again
    if let Err((_, e)) = free_plan(host, &plan, kind, id) {
        return Ok(Outcome::NotYet(e));
    }
    // what was stopped has let go of the room before the load (a holder that took work meanwhile is not stopped: the next
    // try plans again)
    for (c, w) in wants(gpu) {
        if !wait_room(host, c, w) {
            return Ok(Outcome::NotYet(format!("GPU {c} has not freed {:.1} GiB for {id} yet", w as f64 / 1024.0)));
        }
    }
    host.note(job, "loading", &format!("{id} loading on {} (server {})", on_cards(&m.cards_taken(gpu)), x.slot + 1));
    if let Err((code, e)) = host.start(kind, id, x.slot, x.port, gpu) {
        // a refusal for memory is a wait, not a failure: what it asked is learned and the next try plans for it
        if let Some(why) = short_of_room(host, cards, m, gpu, &e, &wants(gpu)) {
            return Ok(Outcome::NotYet(why));
        }
        return Err((code, e));
    }
    Ok(Outcome::Placed { slot: x.slot, port: x.port })
}

/// Whether card `g`'s compute is taken by other work: a busy server of another model there (any kind's: a FlashVSR
/// upscale, a picture), or the video worker making something while it holds memory there. A server of the model
/// asking (`port`) does not count. Memory says nothing of this: an idle copy on a saturated card answers slowly (a
/// song beside a film upscale made no progress for 15 minutes on hardac1, 2026-11-13, while the other card sat idle)
pub fn card_contended(snap: &Snapshot, host: &dyn Host, g: usize, port: u16) -> bool {
    let Some(c) = snap.cards.iter().find(|c| c.index == g) else { return false };
    c.procs.iter().any(|p| match (p.kind.as_deref(), p.port) {
        (Some("video"), _) => !host.video_idle(),
        (Some(k), Some(pp)) if pp != port => snap.servers_of(k).iter().any(|x| x.port == pp && (x.busy || x.inflight > 0)),
        _ => false,
    })
}

/// "GPU 0", or a split model's "GPUs 0 and 1"
fn on_cards(cs: &[usize]) -> String {
    match cs {
        [g] => format!("GPU {g}"),
        _ => format!("GPUs {}", cs.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(" and ")),
    }
}

/// The card a refusal for memory names ("needs about X GiB on <card name>; ..."), of the cards `among` (the ones the
/// start took): by its name, the one with the least free when two share it. None: no card named, or none of them.
pub fn refused_card(cards: &[Card], e: &str, among: &[usize]) -> Option<usize> {
    let name = e.split(" GiB on ").nth(1)?.split(';').next()?.trim();
    cards.iter().filter(|c| among.contains(&c.index) && c.name == name).min_by_key(|c| c.free()).map(|c| c.index)
}

/// A start of `m` on `gpu` that failed (`e`), when it was a refusal for memory on one of the cards it took (`wants`: each
/// with what was planned there): the ask learned for that card (the host's `learn_ask`; `rest`: a split model's other
/// card) and why the request waits - the next try plans for the ask, so it waits for busy work there or stops idle work.
/// None: another failure, or a need past the card's whole memory (it would never load there).
pub fn short_of_room(host: &dyn Host, cards: &[Card], m: &ModelInfo, gpu: usize, e: &str, wants: &[(usize, u64)]) -> Option<String> {
    let (need, _) = asked_mb(e)?;
    let taken: Vec<usize> = wants.iter().map(|w| w.0).collect();
    let card = refused_card(cards, e, &taken).unwrap_or(gpu);
    let planned = wants.iter().find(|w| w.0 == card).map_or(0, |w| w.1);
    let more = host.learn_ask(&m.id, e, card, planned, card != gpu);
    let total = cards.iter().find(|c| c.index == card).map_or(u64::MAX, |c| c.total_mb);
    if need > total {
        return None;
    }
    Some(if more {
        format!("{} needed more room on GPU {card} than planned ({e}); planned again", m.id)
    } else {
        format!("GPU {card} was short of the room {} asked ({e}); it waits for the room there", m.id)
    })
}

/// The loop around [`place`]: tries again every 3 s until a server is found, a cancel (499), the wait runs out (503 no_room:
/// `default_wait`, or - when the caller gave `max_wait` - that, answered `wait_too_long` with the `eta` it was given) or
/// a failure. Returns the (slot, port) to send the request to.
pub fn ensure(host: &dyn Host, req: &PlaceReq, default_wait: std::time::Duration, max_wait: Option<f64>, eta: &serde_json::Value) -> Result<(usize, u16), Fail> {
    let limit = crate::rsvp::wait_limit(default_wait, max_wait);
    let t0 = host.now();
    loop {
        if req.job.is_some_and(|j| host.cancelled(j)) {
            return Err((499, format!("cancelled while it waited for {}", req.model.id)));
        }
        let why = match place(host, req)? {
            Outcome::Placed { slot, port } => return Ok((slot, port)),
            Outcome::NotYet(why) => why,
        };
        host.note(req.job, "waiting", &why);
        let waited = host.now() - t0;
        if let Some(mw) = max_wait.filter(|mw| waited > *mw) {
            return Err(crate::eta::wait_expired(eta, mw, waited));
        }
        if waited > limit.as_secs_f64() {
            return Err((503, format!("{why}; {} loads once one of them idles out or is stopped", req.model.id)));
        }
        host.sleep(3.0);
    }
}

/// A server's "needs about X GiB on <card>; Y GiB are free" (its refusal to load), as (X, Y) in MiB
pub fn asked_mb(e: &str) -> Option<(u64, u64)> {
    let gib = |t: &str| t.trim().parse::<f64>().ok().map(|g| (g * 1024.0) as u64);
    let need = e.split("needs about ").nth(1)?.split(" GiB").next()?;
    let free = e.split(" GiB are free").next()?.rsplit(' ').next()?;
    Some((gib(need)?, gib(free)?))
}

/// What a card must have free, as the front door counts it (`front_free`: total less used), for a server that asked
/// `need` MiB and found `had` free: the difference is its context and what the driver keeps (half a GiB at least: the
/// sample may still count the server that ended), with a quarter GiB for the asking's rounding
///
/// A sample taken before the card's other work ended can make that difference most of a card: then it says nothing
/// about the server, and the ask is the server's need with the half GiB. An ask past the card's whole memory can never
/// be met and would keep the model off the card for good (YuE2: 3.9 GiB asked with 1 GiB free while the sample said
/// 11 GiB, planned 15 GiB on a 12 GiB card from then on, 2026-11-10)
pub fn ask_mb(need: u64, had: u64, front_free: u64, total: u64) -> u64 {
    let ask = need + front_free.saturating_sub(had).max(512) + 256;
    if ask > total { need + 512 + 256 } else { ask }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_refusal_for_memory_is_read_and_planned_for() {
        let e = "audio start qwen3-tts-custom on GPU 0 (server 2): nextcuda: nextcuda-audio-1 ended while loading (nextcuda: qwen3-tts needs \
                 about 5.6 GiB on NVIDIA GeForce RTX 5070; 5.3 GiB are free): nextcuda audio logs";
        let (need, had) = asked_mb(e).unwrap();
        assert_eq!((need, had), (5734, 5427));
        let ask = ask_mb(need, had, 6093, 12227);
        assert!(ask > 6093, "{ask}");
        // a stale sample (the card looked nearly empty while something else still held it) never makes an ask the
        // card cannot hold
        let (need, had) = asked_mb("yue2 needs about 3.8 GiB on NVIDIA GeForce RTX 5070; 1.0 GiB are free").unwrap();
        let ask = ask_mb(need, had, 11282, 12227);
        assert!(ask <= 12227 && ask >= need, "{ask}");
        assert_eq!(asked_mb("qwen3.8 needs about 30.1 GiB on X (28.0 GiB of weights); 22.5 GiB are free").map(|x| x.0), Some(30822));
        assert_eq!(asked_mb("nextcuda-audio-1 ended while loading"), None);
    }

    #[test]
    fn a_refusal_is_put_on_the_card_it_names_not_the_one_started_on() {
        let card = |index: usize, name: &str, used_mb: u64| Card { index, name: name.into(), total_mb: 12227, used_mb, procs: vec![] };
        let cards = vec![card(0, "NVIDIA GeForce RTX 5070", 300), card(1, "NVIDIA GeForce RTX 3060", 10900)];
        // 2026-11-11: MiniMax Music 3 started on GPU 0, its flow stage refused on the 3060 (GPU 1)
        let e = "audio start minimax-music3-int8 on GPU 0 (server 2): nextcuda: nextcuda-audio-1 ended while loading (nextcuda: \
                 minimax-music3 needs about 6.8 GiB on NVIDIA GeForce RTX 3060; 1.2 GiB are free): nextcuda audio logs";
        assert_eq!(refused_card(&cards, e, &[0, 1]), Some(1));
        assert_eq!(refused_card(&cards, e, &[0]), None, "only the cards the start took");
        assert_eq!(refused_card(&cards, "the server ended while loading", &[0, 1]), None);
        // two cards of one name: the fuller one
        let twins = vec![card(0, "B70", 9000), card(1, "B70", 300)];
        assert_eq!(refused_card(&twins, "x needs about 6.0 GiB on B70; 2.0 GiB are free", &[0, 1]), Some(0));
    }

    #[test]
    fn a_split_model_takes_its_other_cards_after_the_one_it_starts_on() {
        let m = |split| ModelInfo { id: "m".into(), kind: "audio".into(), enabled: true, gpus: vec![0, 1], profile: Default::default(), split };
        assert_eq!(m(true).cards_taken(0), vec![0, 1]);
        assert_eq!(m(true).cards_taken(1), vec![1, 0]);
        assert_eq!(m(false).cards_taken(1), vec![1]);
        assert_eq!(on_cards(&[0, 1]), "GPUs 0 and 1");
        assert_eq!(on_cards(&[1]), "GPU 1");
    }
}

/// A video clip about to be sent to the video daemon
pub struct VideoReq<'a> {
    /// the video model (its cards are the ones the clip may run on)
    pub model: &'a ModelInfo,
    /// the card the request names, when it names one
    pub gpu: Option<usize>,
    /// the memory the clip is estimated to take on each card, MiB (a card without an entry: only the margin). The host
    /// gives what it knows of the clip; what it does not know it leaves out, and nothing is then refused for size
    pub need_mb: &'a BTreeMap<usize, u64>,
}

/// Room for a video clip, before it is sent on (2026-11-04: a MiniMax H3 film clip, 768x576, started on GPU 0 of a 12 GB
/// box with two idle Qwen-Image servers still resident there, and failed 45 s in with "nsd_alloc: 927 MiB would pass the
/// device memory cap (10420 of 11195 MiB in use)"; the clip had been accepted because the card had room for the model
/// on paper, and the video engine takes what is free as it goes, so every MiB a picture server holds is one the clip
/// can need).
///
/// A clip gets its card to itself as far as the lightest-to-bring-back work goes: on the card where the idle picture,
/// sound and 3D servers (not pinned or held, not busy) free the most, they are stopped now. The card is the named one,
/// else the model's card with the most free once its idle servers are gone (a near tie goes to the first listed). The
/// video daemon picks its own slot, so this is the best card to clear, not a promise it runs there.
///
/// And when the clip's estimate does not fit on ANY of its cards even with every server there gone (the ones that could
/// go once they are idle, busy ones included - `free + all those servers`, not the foreign processes and not a pinned
/// model), the answer is 409 at once with nothing stopped: the same clip will not fit on this box later either. Busy
/// servers that would make it fit are no reason to refuse: the clip goes on and the daemon starts it when the card is
/// free. Returns the card cleared (None: no idle server to stop).
pub fn room_for_video(host: &dyn Host, req: &VideoReq) -> Result<Option<usize>, Fail> {
    let m = req.model;
    let snap = host.snapshot();
    let gpus: Vec<usize> = match req.gpu {
        Some(g) if m.gpus.contains(&g) => vec![g],
        _ => m.gpus.clone(),
    };
    let want = |g: usize| req.need_mb.get(&g).copied().unwrap_or(0) + MARGIN_MB;
    let ever = |g: usize| {
        let servers: u64 = snap.cards.iter().find(|c| c.index == g).into_iter().flat_map(|c| c.procs.iter())
            .filter(|p| matches!(p.kind.as_deref(), Some("image" | "audio" | "model3d")) && !snap.kept.contains(&p.model)).map(|p| p.mb).sum();
        card_free(&snap.cards, g) + servers
    };
    if !gpus.is_empty() && gpus.iter().all(|g| ever(*g) < want(*g)) {
        let cards: Vec<String> = gpus.iter().map(|g| format!("GPU {g} {:.1} GiB", ever(*g) as f64 / 1024.0)).collect();
        return Err((409, format!("this clip will not fit: it needs ~{:.1} GiB free on a card and, with every picture and sound server stopped, {} {} - \
                                  fewer frames, a smaller canvas or a smaller engine (q6k, q4km) would",
                                  want(gpus[0]) as f64 / 1024.0, if gpus.len() > 1 { "the cards have" } else { "the card has" }, cards.join(", "))));
    }
    let after = |g: usize| card_free(&snap.cards, g) + idle_servers(&snap, g).iter().map(|h| h.mb).sum::<u64>();
    let mut best: Option<usize> = None;
    for &g in &gpus {
        if best.is_none_or(|b| after(g) > after(b) + TIE_MB) {
            best = Some(g);
        }
    }
    let Some(g) = best else { return Ok(None) };
    let plan = idle_servers(&snap, g);
    if plan.is_empty() {
        return Ok(None);
    }
    let (before, freed) = (card_free(&snap.cards, g), plan.iter().map(|h| h.mb).sum::<u64>());
    free_plan(host, &plan, "video", &m.id)?;
    wait_room(host, g, before + freed * 9 / 10);
    Ok(Some(g))
}
