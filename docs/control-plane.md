# The control plane

`http://<host>:8000/` - every card, what each holds and what the enabled models are vouched for there; every model
by kind with its cards (tick which may hold it), enable / disable, pin / unpin and its own page; the models a card
offers that are not downloaded yet; the front ends (Open WebUI on :3000, the video studio). The API and Config tabs
are the OpenAPI document and `nextcuda.conf`.

- **Downloads on demand:** a request for a model whose files are not here answers `model_not_downloaded` (its size,
  whether it fits, how to pull it); the page's Download (with progress and cancel) or `system.models.pull` fetches it;
  the "download automatically" box (`NS_AUTO_PULL`) lets a request wait for it instead.
- **Cards shared, not fought over:** a model starts on the first of its cards with room; idle picture, sound, video
  and chat servers are stopped for it (lightest first) only when it does not fit beside them. Qwen3.8 keeps ~5.9 GB of
  the 3060 free, so a voice (VoxCPM2 int8, 3.3 GiB) reads its replies aloud without the chat model being swapped out.
- **Maintenance mode** sets the box aside (callers told, queues held) - below.

## Models on demand: one lifecycle for every kind

Every enabled model, of every kind (chat, image, audio, 3D, video), is **on demand**, and there are no start buttons:

- **Enabled, unloaded -> loaded:** a request for it spins it up on a card that can hold it, stopping idle, unpinned,
  unheld holders for the room. Chat goes through the switcher or `nextcuda llm start`, picture/sound/3D through a
  server a model (`--idle-exit`), video through the daemon (`nextcuda video start`, the enabled video models as its
  engines). A video request with the daemon down (or running without an enabled model's engine) has the backplane
  apply the daemon first and wait (`video.*`, `video.engines.list` included; 120 s at most, then `unavailable` /
  `timeout`); a watcher brings back one that crashed (twice missing in a row, 30 s apart) and a restart of the daemon
  only happens when no clip renders. `tooling/deploy.sh` brings the daemon back with the `--model`/`--engine` it had; the
  backplane's own rule (first model `minimax-h3` if enabled, `--engine` for the others, `--gpu` the models' cards) fixes
  any difference at the next video request.
- **Loaded -> unloaded:** when idle (`NS_IDLE_MINUTES`, `NS_VIDEO_IDLE` for a video worker, the switcher's idle time),
  or sooner when another model needs its room. A model that cannot fit is answered `no_room` (503) with who holds the
  card - naming a pin or a hold.
- **Pin (the page's `pin` / `unpin` buttons, `system.pin` / `system.unpin`, `{"do": "pin"}`):** the only manual
  control. A pinned model is **hot**: loaded now (in the background), never unloaded for idleness (its server starts
  without `--idle-exit`, the backplane keeps a running one's idle clock fresh, the video daemon is asked every 10 s
  to keep its workers (`POST /engine/keep`, which its idle unload respects), the switcher leaves it), never chosen as an evictable idle holder. Kept in the registry entry
  (`pinned`, beside `enabled`): after a restart of `nextcuda serve` the pinned, enabled models are loaded again, the
  video daemon first. Unpin only clears the flag; the model stays loaded until the idle rules take it. Disabling a model
  unpins it. A pin that cannot load (no room) stays set; the failure is a run on the page and the backplane tries again.
  `system.load` / `system.unload` stay as RPC actions for tools; the page no longer has buttons for them.
- **Keep-alive (`rsvp_s`):** see below. A pin is a hold with no end. The chat server holds one model at a time: a pinned or held chat model is never swapped out - a request for another chat model gets `no_room` naming the pin or hold (the switcher answers the same).

### `rsvp_s`: keep the model loaded after this request

Every JSON method that runs a model (not `/v1`) takes an optional `rsvp_s` (seconds): keep the model that served this
request loaded for that long **after the request is fulfilled** (answered, failed or cancelled; for a job when the job
ends - the clock never runs while the request does). Clamped to `NS_MAX_RSVP_S` (default 86400); a larger value is not
an error, the object result says `rsvp_granted_s`. The hold is the latest of the holds (a later, shorter `rsvp_s`
never shortens it). A held model is not idle for the idle timers and room planning leaves it (the backplane keeps its
server's idle clock fresh every 10 s), then the normal idle rules begin. `rsvp_s: 0` releases: the model goes as soon
as it is idle. No `rsvp_s`: the normal idle time. The page shows "held for N more s". Video: the hold keeps the model
out of room planning, and the front door asks the daemon (`/engine/keep`, every 10 s) not to unload its workers for
idleness while a pin or hold lasts; a video request's hold starts when its clip leaves the studio's queue (finished,
failed or cancelled; the front door watches the clip). Unpin or the end of a hold just stops the asking: the normal
idle unload is back, no restart. Chat holds are written to the registry (`hold_until`) for the switcher.

### ETAs and `max_willing_to_wait_s`

The backplane keeps a rolling history, the last 100 finished work requests per key (`history/*.jsonl` beside
`serve.json`; a line each: method, model, shape, `wait_s`, `run_s`, ok/failed/cancelled; written off the request path,
read at start; a file is cut back to 100 when it holds 200). Each request is stored under its **shape** -
`method|model|` the size-driving parameters bucketed (image: width, height to 64, steps to 4; video: width, height,
frames to 8, steps to 2, engine; speech: text length in powers of two, voice-independent; song: duration to 10 s and
lyrics length; chat: prompt and `max_tokens` in powers of two; 3D: steps, resolution, faces; anything else
`method|model`) - and under its model. An **ETA** is `{wait_s, run_s, total_s, samples, basis, spread}`: `run_s` the
median of the shape's ok runs (`basis: "shape"`), else the model's (`"model"`), else none (`"none"`, no number is
invented; failed and cancelled runs feed only waits); `wait_s` is 0 for a loaded model with a free server, else the
work ahead on the kind's servers (running jobs' remaining time and the queue ahead, each from history) over its servers
plus the model's spin-up from history, and null when a part has no history. `system.eta {method, params, model?,
max_willing_to_wait_s?}` answers it (`would_accept`) and does no work; queued and running jobs carry `eta` and
`eta_remaining_s`.

Every such method also takes `max_willing_to_wait_s`: the longest you will wait for the request to **start** (not run).
A known `eta.wait_s` over it is answered at once, `{code: "wait_too_long", retryable: true, eta, max_willing_to_wait_s}`,
with nothing queued, loaded or evicted; with an unknown ETA the request goes on but is withdrawn with the same code
(`waited_s`) if it has not started in time, a queued job likewise (its status says `wait_too_long`); a loaded model with
a free server always starts. Absent: the waits are the server's (90 s a request, 30 min a job); when given, the
limit replaces them for requests and jobs alike. All of this is independent of `rsvp_s`.

The outcome is in the JSON body, never the HTTP status: a `/rpc` answer is HTTP 200 with `{ok: false, error: {code,
message, retryable, status, ...}}`; `no_room` and `wait_too_long` both carry `error.status` 503 and `retryable: true`
(`wait_too_long` adds `eta`, `max_willing_to_wait_s`, and `waited_s` when it was withdrawn after waiting; `no_room`'s
message names a pin or hold that blocks). The `/v1` routes (OpenAI's shape) answer the HTTP status itself.

### Concurrency profile: how many copies, and who may share a card

Each model's card (`model-cards/README.md`: `concurrency`, `coexistence`) says how many copies run on one card and are
worth running, and whether other models may run beside it. The backplane enforces it for pictures, sound and 3D
(the servers it starts) and for the residents it finds on a card:
- never more copies of a model on a card than `max_copies_per_card` (`start_at` refuses the extra one); a further copy
  starts, when requests are waiting, only up to `recommended_copies` a card (the default: one a card);
- an **exclusive** model gets its card to itself: idle residents are stopped for it even where the memory would fit, and a
  resident that cannot go (busy, pinned, held) makes it wait and then answer `no_room`, naming it; while an exclusive
  model is loaded nothing else is started on its card - the answer to a request that would is `no_room` with
  `exclusive: <id> (runs alone on its card)`;
- `shared_with` limits who may be a resident beside a model (`kind:<kind>`, `model:<id>`, either side's list counts);
- pins and holds are honoured throughout: a pinned or held resident is never stopped for these either;
- the video daemon and chat: the rules apply to what shares their cards (a resident exclusive video or chat model blocks
  others the same way); the daemon's own worker count a card is one.

Where clients see it: every model row (`system.models`, `system.state`) has `concurrency` (with `status`:
`measured` or `unmeasured`), `coexistence`, `copies_now`, `recommended_copies` and `max_copies_per_card`; `system.busy`
has `engines`, one entry a model: `{id, kind, copies_now, recommended_copies, max_copies_per_card, mode, status}`, for
a dispatcher. The home page shows "unmeasured" or "n/rec copies".

**Measuring:** `nextcuda models bench <id> [--copies 1,2,3] [--beside <id>...] [--requests N] [--gpu N] [--write]`
(under `flock ~/.nextcuda-gpu.lock`) starts n copies on one card, sends the kind's standard small request set to all at
once, and prints each run (units a minute, each request's seconds, the card's peak VRAM, failures) and the proposed
block; `--beside` runs it next to each named model, alone and together. It never edits a card; `--write` puts the block in
the card's file. Pictures (generation), speech and songs, and chat have a standard request; the video daemon, 3D and the
models that take an input file (OCR, upscalers, recognisers) are measured by hand until they do.

Table to fill in (one row a model; "unmeasured" until a bench is run and its block is in the card):

| model | kind | status | max copies | recommended | mode | shared with | measured on |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `nemotron-3.5-super-iq1m` | llm | unmeasured | 1 | 1 | shared | | |
| `qwen3.8-27b-gsq-iq3s-long` | llm | unmeasured | 1 | 1 | shared | | |
| `qwen3.8-27b-gsq-iq3s` | llm | unmeasured | 1 | 1 | shared | | |
| `ming-image-0.1-design` | image | unmeasured | 1 | 1 | shared | | |
| `moss-image-sft-dmd2-q8` | image | unmeasured | 1 | 1 | shared | | |
| `moss-image-sft-q8` | image | unmeasured | 1 | 1 | shared | | |
| `moss-image-sft-turbo4-q8` | image | unmeasured | 1 | 1 | shared | | |
| `moss-image-sft-turbo8-q8` | image | unmeasured | 1 | 1 | shared | | |
| `olmocr-2-7b-q8` | image | unmeasured | 1 | 1 | shared | | |
| `paddleocr-vl-1.6` | image | unmeasured | 1 | 1 | shared | | |
| `qwen-image-2.1-q6k` | image | unmeasured | 1 | 1 | shared | | |
| `qwen-image-2.1-q8` | image | unmeasured | 1 | 1 | shared | | |
| `qwen-image-2.1-rrq4` | image | unmeasured | 1 | 1 | shared | | |
| `qwen-image-2512-q4km` | image | unmeasured | 1 | 1 | shared | | |
| `real-esrgan` | image | unmeasured | 1 | 1 | shared | | |
| `starvector-1b-im2svg` | image | unmeasured | 1 | 1 | shared | | |
| `unlimited-ocr-q4` | image | unmeasured | 1 | 1 | shared | | |
| `unlimited-ocr-q8` | image | unmeasured | 1 | 1 | shared | | |
| `waifu2x` | image | unmeasured | 1 | 1 | shared | | |
| `acestep-1.5-base` | audio | unmeasured | 1 | 1 | shared | | |
| `acestep-1.5` | audio | unmeasured | 1 | 1 | shared | | |
| `breeze-tts-2-q4` | audio | unmeasured | 1 | 1 | shared | | |
| `breeze-tts-2` | audio | unmeasured | 1 | 1 | shared | | |
| `minimax-music3-int8` | audio | unmeasured | 1 | 1 | shared | | |
| `minimax-music3` | audio | unmeasured | 1 | 1 | shared | | |
| `qwen3-tts-base` | audio | unmeasured | 1 | 1 | shared | | |
| `qwen3-tts-custom` | audio | unmeasured | 1 | 1 | shared | | |
| `qwen3-tts-design` | audio | unmeasured | 1 | 1 | shared | | |
| `vibevoice-1.5b-int8` | audio | unmeasured | 1 | 1 | shared | | |
| `vibevoice-1.5b` | audio | unmeasured | 1 | 1 | shared | | |
| `voxcpm2-int8` | audio | unmeasured | 1 | 1 | shared | | |
| `voxcpm2` | audio | unmeasured | 1 | 1 | shared | | |
| `whisper-large-v3-turbo` | audio | unmeasured | 1 | 1 | shared | | |
| `yue2-3b-q8` | audio | unmeasured | 1 | 1 | shared | | |
| `ltx-2.5-fast` | video | unmeasured | 1 | 1 | shared | | |
| `ltx-2.5` | video | unmeasured | 1 | 1 | shared | | |
| `minimax-h3-q4km` | video | unmeasured | 1 | 1 | shared | | |
| `minimax-h3-q6k` | video | unmeasured | 1 | 1 | shared | | |
| `minimax-h3-q8` | video | unmeasured | 1 | 1 | shared | | |
| `minimax-h3-ref2va` | video | unmeasured | 1 | 1 | shared | | |
| `minimax-h3` | video | unmeasured | 1 | 1 | shared | | |
| `animateanymesh` | model3d | unmeasured | 1 | 1 | shared | | |
| `hunyuan3d-2.1-paint` | model3d | unmeasured | 1 | 1 | shared | | |
| `hunyuan3d-2.1` | model3d | unmeasured | 1 | 1 | shared | | |
| `hunyuan3d-2mini-turbo` | model3d | unmeasured | 1 | 1 | shared | | |
| `pixal3d` | model3d | unmeasured | 1 | 1 | shared | | |
| `rignet` | model3d | unmeasured | 1 | 1 | shared | | |
| `skintokens` | model3d | unmeasured | 1 | 1 | shared | | |
| `trellis2-512` | model3d | unmeasured | 1 | 1 | shared | | |
| `trellis2` | model3d | unmeasured | 1 | 1 | shared | | |
| `unimate` | model3d | unmeasured | 1 | 1 | shared | | |
| `unirig` | model3d | unmeasured | 1 | 1 | shared | | |

## The front door and the model servers

- **The front door on :8000:** `/rpc/<kind>.<method>` with a bearer token (`serve.json`, or `NS_API_TOKEN`); every
  answer is the envelope `{ok, result}` or `{ok: false, error: {code, message, retryable, ...}}`. `image.generate`
  and `image.edit` run as jobs (`wait: false`, `label`, `reentrant_id`), asked for with `image.status` /
  `image.jobs`, stopped with `image.cancel`; `image.upload`, `image.queue.*`, `system.busy`, `system.cards`,
  `system.models` and `system.version` (which lists its features) are there as they are on nextsycl.
- **Maintenance mode** (the page's Maintenance button, `system.maintenance {on, message, eta_s, unload}`): the box set
  aside for development while callers are told. What would start work - chat, generation, edits, speech,
  `system.load` / `enable`, the OpenAI POST routes - is refused with error code `maintenance` (503, retryable,
  `retry_after_s`; `/v1` answers HTTP 503 with `Retry-After`); status, listings and cancels answer as always. Queued
  image and audio jobs wait, the video queue pauses, running work finishes; the switcher answers Open WebUI's chats
  with the notice (`maintenance.json` beside the registry). Requests with `X-Maintenance-Bypass: <key>` (the key turning
  it on answers) go through.
- **A model server a model, on demand:** the first request for a registered model starts `nextcuda image serve` for
  it on the first card with room, swaps it when a request names another, and lets it go after the idle time.
- **Servers as processes:** `nextcuda image start` is a process with a pid and a log in the socket directory;
  `NS_CONTAINER_ENGINE` brings containers back.

## Is the control plane there: `system.ping`

`POST /rpc/system.ping` answers at once and loads or starts nothing: `{available, maintenance, reason, retry_after_s,
version, api, at}` - `reason` is the maintenance notice's words when the box is set aside, null otherwise. It is what a
client asks on the way to sending work and after an error (the platform's worker does, and once a minute when work is
queued); `system.version`'s `features` lists it. `system.busy` is the fuller answer (what is working, per card).
