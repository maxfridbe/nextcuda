# Changelog

Bump `VERSION` in `src/lib.rs` and `version` in `Cargo.toml` with each entry.

## 0.2.0 - 2026-11-11

Split models are planned on every card they take, and a refusal for memory is a wait, not a failure.

- `ModelInfo.split` (`cards_taken`): one server takes every card of the model, the one it is started on first. `place`
  plans the room on each (`PlaceReq.rest_mb`, `asked_rest_mb`: its share on the others) and stops idle work only when
  all of them can have it; else it waits or answers no_room. (MiniMax Music 3 int8 on brainiac-nvidia, 2026-11-11: planned
  on the 5070 alone, its idle TTS stopped, then 36 starts died on the 3060 its flow stage needed, which ACE-Step's song
  held.)
- A start refused for memory (`short_of_room`) is learned for the card its refusal names (`refused_card`) and the try
  answers `NotYet`; only a need past that card's whole memory is still a failure. `Host::learn_ask` takes `rest`.
- `fake`: `add_split`, `hide` (memory a snapshot does not show); scenarios 26-30.

## 0.1.0 - 2026-10-31

First extraction from nextcuda's backplane (`glue/serve/src/home`): nextcuda depends on it, behaviour unchanged.

- `profile`: model-card `concurrency` / `coexistence` schema, defaults, validation; the bench's decisions.
- `rsvp`: `rsvp_s` keep-alives and holds, the server cap, `max_willing_to_wait_s`, the `wait_too_long` error tag.
- `room`, `place`: snapshots and pure room planning; `place` / `ensure` over the `Host` trait (idle eviction, busy-holder
  drain, second copies up to the recommended count, exclusive and `shared_with` residents, pins and holds honoured).
- `shape`, `history`, `eta`: request shapes per kind (a registry), the rolling history (last 100, storage behind a trait),
  ETAs, `would_accept`, withdrawal of late queued jobs.
- `fake`: a fake host on a fake clock; `tests/scenarios.rs`: 22 scenarios.
