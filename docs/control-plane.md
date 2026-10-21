# The control plane

`http://<host>:8000/` - every card, what each holds and what the enabled models are vouched for there; every model
by kind with its cards (tick which may hold it), enable / disable, start / stop and its own page; the models a card
offers that are not downloaded yet; the front ends (Open WebUI on :3000, the video studio). The API and Config tabs
are the OpenAPI document and `nextcuda.conf`.

- **Downloads on demand:** a request for a model whose files are not here answers `model_not_downloaded` (its size,
  whether it fits, how to pull it); the page's Download (with progress and cancel) or `system.models.pull` fetches it;
  the "download automatically" box (`NS_AUTO_PULL`) lets a request wait for it instead.
- **Cards shared, not fought over:** a model starts on the first of its cards with room; idle picture, sound, video
  and chat servers are stopped for it (lightest first) only when it does not fit beside them. Qwen3.8 keeps ~5.9 GB of
  the 3060 free, so a voice (VoxCPM2 int8, 3.3 GiB) reads its replies aloud without the chat model being swapped out.
- **Maintenance mode** sets the box aside (callers told, queues held) - below.

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
