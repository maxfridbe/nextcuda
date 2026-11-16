# Live conversations: `/v1/realtime`

A WebSocket for talking with a full-duplex speech model: the microphone streams in, the model listens and answers
with speech (and its text) as it makes it; you can cut in. It speaks OpenAI Realtime's events, so an OpenAI Realtime
client works by changing the URL and key. The API is nextsycl's (its docs/api/realtime.md), ported to nextcuda as it
is - the same events, the same extensions under the same JSON key `nextsycl` - so a client of one box works against
the other unchanged (the creative platform's Talk page). On nextcuda the full-duplex model is Haru-4B
(docs/ports/haru-4b.md: lockstep frames, the extension below).

```text
ws://<box>:8000/v1/realtime?model=<id>          the front door: Authorization: Bearer <token>, or the browser's
                                                sub-protocol "openai-insecure-api-key.<token>"; no model (or one not
                                                here, as OpenAI's gpt-realtime): the loaded or first enabled model
                                                whose card's speech list says realtime
wss://<box>:8443/v1/realtime?model=<id>         the same over https (the front door's TLS port, NS_TLS_PORT)
ws://<box>:8000/open/<id>/v1/realtime           the model's own page's (no token: as every /open/ page); wss on :8443 too
ws://127.0.0.1:<port>/v1/realtime               the model's audio server itself (nextcuda audio serve; /realtime too)
```

The front door loads the model as for any request (rsvp, room, a card), passes the upgrade on to its audio server and
copies both ways while the socket lasts; the request's lease is held meanwhile and the server counts as busy (its
`/api/progress` says `realtime_sessions` and `phase: conversation`), so it is neither stopped for room nor idled out.
Maintenance mode refuses new conversations. Not an upgrade: 400; no full-duplex model enabled: 409 (OpenAI's error
JSON, `{"error": {"message", "type"}}`).

## Audio

| format | in | out |
|---|---|---|
| `pcm16` (default) | 16-bit little-endian mono, 24 kHz unless `nextsycl.input_sample_rate` (or the GA shape's `rate`) says another | 24 kHz unless `nextsycl.output_sample_rate` |
| `g711_ulaw`, `g711_alaw` | 8 kHz | 8 kHz |
| `pcm_f32le` | float32 mono, at `sample_rate_hz` | float32 |

What comes in is resampled for the model (Haru hears and speaks at 24 kHz). `input_audio_buffer.append` carries base64;
a binary WebSocket message is taken as raw audio in the session's input format (an extension).

## Client events

| event | what it does |
|---|---|
| `session.update {session}` | the settings, beta or GA shape: `instructions` (the system prompt; empty: the model's own), `voice` (a built-in one, or a saved voice's name: its recording - the voice library's, `audio.voices.add` - cloned), `input_audio_format` / `output_audio_format` (GA: `audio.input.format`, `audio.output.format`, `audio.output.voice`), `turn_detection` (`{"type": "server_vad"}` - the model decides when you have finished; `silence_duration_ms`: the longest pause before it answers anyway; `null`: only your commit ends a turn), `tools` (functions it may call), `tool_choice: "none"`, `temperature`, `max_response_output_tokens` (a number, or `"inf"`: as much as the context holds), `modalities` (`["text"]`: no audio out). Extensions: `extra_body` (`language`, `turn_fallback_s`, `history`, `sample_rate_hz`) and `nextsycl` (`language`, `input_sample_rate`, `output_sample_rate`, `voice`, `history`, `seed`, `options`: the engine's own, `card`: a lockstep model's). The key is nextsycl's, kept for compatibility with nextsycl's API; `nextcuda` is taken as the same |
| `input_audio_buffer.append {audio}` | your audio (base64) |
| `input_audio_buffer.commit` | your turn is over now (with or without turn detection) |
| `input_audio_buffer.clear` | your audio not yet taken is dropped |
| `response.create` | answer now (a text turn, a function's result) |
| `response.cancel` | stop the answer being made |
| `conversation.item.create {item}` | `{"type": "message", "role": "user" or "assistant", "content": [{"type": "input_text", "text": ...}]}` - a turn as text; `{"type": "function_call_output", "call_id", "output"}` - a function's result |
| `conversation.item.truncate {item_id, audio_end_ms}` | you heard only this much of an answer (the client stopped playing it): its history keeps that share |
| `session.close` | ends it (closing the socket does the same) |

## Server events

| event | when |
|---|---|
| `session.created`, `session.updated` | at the start, after each update; `session.nextsycl` tells the model's own: `input_rate`, `output_rate`, `frame_ms` and `lockstep`, `voices`, `clone`, `languages`, `tools`, and the session's `input_sample_rate`, `output_sample_rate`, `language`, `card` (the key is `nextsycl` on nextcuda too) |
| `input_audio_buffer.speech_started {audio_start_ms, item_id}` | you started speaking |
| `input_audio_buffer.speech_stopped {audio_end_ms}` | your speech stopped |
| `input_audio_buffer.committed`, `conversation.item.created` | your turn taken (the model hears it directly: `transcript` is null) |
| `response.created`, `response.output_item.added`, `conversation.item.created`, `response.content_part.added` | an answer begins (its item `msg_<response id>`) |
| `response.audio_transcript.delta {delta}` | its text as written |
| `response.audio.delta {delta}` | its sound as spoken (base64, the session's output format) |
| `response.output_item.added` / `response.function_call_arguments.delta` / `.done {call_id, name, arguments}` / `response.output_item.done` | a function call it makes (an item of type function_call) |
| `response.audio.done`, `response.audio_transcript.done {transcript}`, `response.content_part.done`, `response.output_item.done`, `response.done {response: {id, status, output, nextsycl: {audio_ms}}}` | the answer ended: `completed`, `cancelled` (cut in, cancelled) or `failed` |
| `error {error: {type, code, message}}` | a bad event or a failure (the session goes on) |

A client using the GA shape (`session.type: "realtime"`, or `?flavor=ga` in the URL) gets GA's names:
`response.output_audio.delta`, `response.output_audio_transcript.delta`, `response.output_text.delta`,
`conversation.item.added`.

## Lockstep models (Haru-4B)

OpenAI's events assume a model that answers a turn; Haru-4B is full duplex in lockstep: every 133 ms (7.5 Hz) one
3,200-sample 24 kHz frame of yours in, one of its own out (silence included), its text alongside. It hears and
speaks at once: when to answer, back-channel ("mm-hm" while you talk) or stop because you cut in is the model's own
doing, not a turn rule's. Its card's speech list is `[realtime, lockstep]`: its `system.models` row has
`capabilities: ["realtime", "lockstep"]`, `methods: []` (it reads nothing aloud) and `lockstep: true`. On this socket:

| | |
|---|---|
| `session.created` | `nextsycl.frame_ms: 133, lockstep: true` (a turn-taking model: 0, false), `voices: []` (its voice is a recording: `clone: true`), `languages`, `tools: false` |
| the card | `session.update {session: {nextsycl: {card: {...}}}}` - `persona`, `name`, `gender` (`female` / `male`), `backchannel` (`low` / `mid` / `high`), `other_name`, `relationship`, `situation`; fields left out keep the model's own card, `null` goes back to it; a string is a whole card's text. `nextsycl.language` picks the card's wording; `session.nextsycl.card` echoes it |
| the voice | `voice`: a saved voice by name - its recording is the voice reference |
| options | `nextsycl.options`: the engine's own (docs/ports/haru-4b.md); `temperature`; `nextsycl.seed` |
| your audio | `input_audio_buffer.append` (or a binary message) of any size as it is recorded; the model cuts it into frames (resampled to 24 kHz first) |
| its audio | one long-running response (`response.created` once): each frame of yours is answered by exactly one `response.audio.delta` of one frame (133 ms) in the session's output format, in order, and its new text (`response.audio_transcript.delta`). Sent as fast as made |
| turn events | `input_audio_buffer.committed` / `response.done` do not come while it runs |
| `response.cancel` | ends the response (`response.done` cancelled) |
| not taken | `conversation.item.create` (a turn as text), function calls |

What the engine does with the card, the voice, the options and the context's end is Haru-4B's, in
docs/ports/haru-4b.md. A client plays the deltas back to back behind a short jitter buffer and records with echo
cancellation (or headphones): the model hears whatever the microphone picks up, its own voice included.

## How nextcuda differs from nextsycl

- **The models**: nextcuda has Haru-4B only; nextsycl's turn-taking Sophon-Realtime is not ported. The socket and its
  events are the same for both kinds of model.
- **The JSON key**: `nextsycl` everywhere, as on nextsycl (compatibility); nextcuda also takes `nextcuda` in
  `session.update` (`nextsycl` wins when both are sent). Answers say `nextsycl` only.
- **The audio server's engine**: held only to open a conversation (`AudioEngine::realtime`, `converse`), not for its
  length (nextsycl holds a read lock); the conversation runs on the engine's own threads, and the server is busy, not
  retired after a GPU error, while one is open.
- **https**: nextsycl pumps every TLS connection through a socket pair; nextcuda reads and writes a TLS session from
  the request's own thread and pumps it only when a WebSocket is spliced (`TlsConn::pump`). Same behaviour on the wire.
- **The document**: `GET /openapi.yml` lists `/v1/realtime` while an audio model is enabled (nextsycl: while a
  full-duplex one is).

## Example

A lockstep conversation with Haru-4B (a saved voice, a card of its own; `--say` / `--gap` the user's script):

```sh
python3 -I reference/haru/client.py --url "ws://localhost:8000/v1/realtime?model=haru-4b" --token $TOKEN --voice MYVOICE \
    --card card.json --say hello.wav --gap 5 --say question.wav --gap 8 --out DIR
```

(it writes what each side said and the frames' turnaround; docs/ports/haru-4b.md has its options).
