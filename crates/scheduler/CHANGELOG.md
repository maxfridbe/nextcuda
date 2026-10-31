# Changelog

Bump `VERSION` in `src/lib.rs` and `version` in `Cargo.toml` with each entry.

## 0.1.0 - 2026-10-31

First extraction from nextcuda's backplane (`glue/serve/src/home`): nextcuda depends on it, behaviour unchanged.

- `profile`: model-card `concurrency` / `coexistence` schema, defaults, validation; the bench's decisions.
- `rsvp`: `rsvp_s` keep-alives and holds, the server cap, `max_willing_to_wait_s`, the `wait_too_long` error tag.
- `room`, `place`: snapshots and pure room planning; `place` / `ensure` over the `Host` trait (idle eviction, busy-holder
  drain, second copies up to the recommended count, exclusive and `shared_with` residents, pins and holds honoured).
- `shape`, `history`, `eta`: request shapes per kind (a registry), the rolling history (last 100, storage behind a trait),
  ETAs, `would_accept`, withdrawal of late queued jobs.
- `fake`: a fake host on a fake clock; `tests/scenarios.rs`: 22 scenarios.
