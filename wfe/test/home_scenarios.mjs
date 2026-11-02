// The front door's Status page from a made-up state: every kind of model it serves is listed (3D too), the cards are named
// as the box names them. Run after ./wfe/build.sh: node wfe/test/home_scenarios.mjs
import { Scenarios, mod, stubFetch } from "./lib.mjs";

const sc = new Scenarios("home");
// the page polls the front door and sets timers when it loads: neither is wanted here
stubFetch(() => ({}));
const realTimeout = globalThis.setTimeout;
globalThis.setTimeout = () => 0;
const { App } = await mod("home", "components/App.js");
const { ui } = await mod("home", "main.js");
globalThis.setTimeout = realTimeout;

const find = (v, pred, out = []) => { if (v && typeof v === "object") { if (pred(v)) out.push(v); for (const c of v.children || []) find(c, pred, out); } return out; };
const link = { gen: 4, width: 16 };
const card = { index: 0, name: "NVIDIA GeForce RTX 3060", pci: "0000:01:00.0", ts: 0, vram_used_mb: 2048, vram_total_mb: 12288, busy_pct: 0, pcie: { cur: link, card_max: link, slot_max: link },
               host_load1: 0, host_cpus: 8, energy_j: 0, procs: [], vouched: [], vouched_mb: 0, vouched_headroom_mb: 0, overbooked: false };
const model = (id, kind, o = {}) => ({ id, kind, title: id, enabled: true, gpus: [0], state: "unloaded", spans: false, capabilities: [], methods: [], vouch_mb: { 0: 6000 },
                                       measured: false, headroom_mb: {}, headroom_given: false, page: 8000, pinned: false, held_s: 0, copies_now: 0, recommended_copies: 1, max_copies_per_card: 1,
                                       concurrency: { max_copies_per_card: 1, recommended_copies: 1, scaling: null, measured: null, status: "unmeasured" },
                                       coexistence: { mode: "shared", shared_with: [], exclusive_reason: "", peak_extra_gib: null }, ...o });
ui.error = null;
ui.tab = "status";
ui.state = {
  ts: 0, host: { mem_total_mb: 65536, mem_avail_mb: 32768, load1: 0.5, cpus: 8 }, cards: [card],
  models: [model("qwen-image-2.1-q8", "image"), model("trellis2-q8", "model3d"), model("vibevoice-1.5b", "audio"), model("qwen3-8b", "llm")],
  services: [], runs: [], idle_minutes: 10, ports: { video: 8100, chat_ui: 3000 }, maintenance: null, drives: { drives: [] }, available: [], pulls: [], auto_pull: false,
};
sc.show("status", App())
  .has(/RTX 3060/).lacks(/NVIDIA GeForce/).has(/Images/).has(/Audio/).has(/Chat/).has(/\b3D\b/).has(/trellis2-q8/).has(/1 of 1 enabled/);

// a card lists every video engine and chat model it may run: the one that counts, and the alternatives dim and uncounted
{
  const v = (name, kind, mb, counted) => ({ model: name, kind, mb, headroom_mb: 0, headroom_given: false, measured: false, state: "ready", of: 2, counted });
  const keep = ui.state.cards;
  ui.state.cards = [{ ...card, vouched: [v("minimax-h3-q4km", "video", 10179, true), v("ltx-2.5-dev", "video", 9960, false)], vouched_mb: 10179 }];
  sc.show("card-video-alternative", App())
    .has(/minimax-h3-q4km/).has(/ltx-2\.5-dev/).has(/one of 2 video engines: one runs at a time/)
    .has(/an alternative: one of 2 video engines runs at a time, so it is not counted/);
  ui.state.cards = keep;
}

// a card's allocations are one list, a line a model: the lines with a job on top (next to their model), then by category,
// the larger first; each line has the bar of its chunk of the card
{
  const proc = (pid, kind, id, mb, o = {}) => ({ pid, vram_mb: mb, program: "nextcuda", kind, command: "serve", model: id, model_id: id, port: null, ...o });
  const v = (model, kind, mb, state = "unloaded", o = {}) => ({ model, kind, mb, headroom_mb: 0, headroom_given: false, measured: true, state, counted: true, ...o });
  const keep = ui.state.cards;
  ui.state.cards = [{ ...card, vram_total_mb: 12288, vram_used_mb: 9000,
    procs: [proc(1, "audio", "voxcpm2", 5000), proc(2, "image", "qwen-image-2.1-q8", 4000, { job_title: "Cover art for Blackbird", job_label: "img-77" })],
    vouched: [v("voxcpm2", "audio", 5058, "loaded"), v("qwen-image-2.1-q8", "image", 10179, "busy"), v("qwen3-8b", "llm", 6000), v("acestep-1.5", "audio", 1756), v("breeze-tts-2", "audio", 6466)],
    vouched_mb: 29459, overbooked: true }];
  const t = sc.show("card-allocations", App());
  // the allocation lines: between the card's numbers and the Models panel
  const from = t.lines.findIndex((l) => /^in use /.test(l)) + 1, to = t.lines.findIndex((l) => /^Models /.test(l));
  const rows = t.lines.slice(from, to);
  const at = (re) => rows.findIndex((l) => re.test(l));
  const order = ["qwen-image-2.1-q8", "qwen3-8b", "breeze-tts-2", "voxcpm2", "acestep-1.5"].map((n) => at(new RegExp(n + "\\b")));
  sc.check("card-allocations", order.every((i, k) => i >= 0 && (k === 0 || i > order[k - 1])),
    `the job's model first, then chat, then audio larger first (${order.join(",")})`);
  // the job has a line of its own under the model that makes it, not squeezed beside the name
  const at0 = rows.findIndex((l) => /qwen-image-2\.1-q8/.test(l));
  sc.check("card-allocations", /Cover art for Blackbird/.test(rows[at0 + 1] ?? "") && /img-77/.test(rows[at0 + 1] ?? ""), `the job's full line follows its model (${rows[at0 + 1]})`);
  sc.check("card-allocations", !/Cover art/.test(rows[at0] ?? ""), "the model's own line does not carry the job");
  sc.check("card-allocations", find(App(), (n) => n.data?.class?.abar).length === 5, "a bar for each line");
  // the eye: hide the reservations and only what is loaded stays, each with its job
  const list = (t) => { const a = t.lines.findIndex((l) => /^in use /.test(l)) + 1, b = t.lines.findIndex((l) => /^Models /.test(l)); return t.lines.slice(a, b).join("\n"); };
  ui.hideRsvp = true;
  const h = sc.show("card-reservations-hidden", App());
  const hl = list(h);
  sc.check("card-reservations-hidden", /qwen-image-2\.1-q8/.test(hl) && /voxcpm2/.test(hl) && /Cover art for Blackbird/.test(hl), "what is loaded stays, with its job");
  sc.check("card-reservations-hidden", !/qwen3-8b|breeze-tts-2|acestep-1\.5/.test(hl), `the reservations are gone from the card's list (${hl.split("\n").length} lines)`);
  h.has(/reservations hidden/);
  ui.state.cards = [{ ...card, procs: [], vouched: [v("qwen3-8b", "llm", 6000), v("acestep-1.5", "audio", 1756)], vouched_mb: 7756 }];
  const idle = sc.show("card-reservations-hidden-idle", App());
  sc.check("card-reservations-hidden-idle", !/qwen3-8b/.test(list(idle)) && /nothing loaded · 2 reserved, hidden/.test(list(idle)), "an idle card says what is hidden");
  ui.hideRsvp = false;
  const shown = sc.show("card-reservations-shown", App());
  sc.check("card-reservations-shown", /qwen3-8b/.test(list(shown)) && /acestep-1\.5/.test(list(shown)) && !/reservations hidden/.test(shown.text), "shown again: every reserved model is back");
  ui.state.cards = keep;
}

// the model sections fold shut, one by one
{
  ui.state.models = [model("qwen-image-2.1-q8", "image"), model("voxcpm2", "audio")];
  ui.collapsed = {};
  sc.show("models-open", App()).has(/qwen-image-2\.1-q8/).has(/voxcpm2/);
  ui.collapsed = { image: true };
  sc.show("models-image-folded", App()).has(/Images/).lacks(/qwen-image-2\.1-q8/).has(/voxcpm2/).has(/0 of 1 enabled|1 of 1 enabled/);
  ui.collapsed = {};
}

// the concurrency profile: unmeasured until a bench says otherwise, then copies and exclusivity
{
  ui.state.models = [model("qwen-image-2.1-q8", "image"),
                     model("voxcpm2", "audio", { copies_now: 1, recommended_copies: 2, max_copies_per_card: 3, state: "loaded",
                       concurrency: { max_copies_per_card: 3, recommended_copies: 2, scaling: { 1: 1, 2: 1.6 }, measured: { date: "2026-10-31", card: "RTX 5070", method: "bench" }, status: "measured" } }),
                     model("trellis2-q8", "model3d", { coexistence: { mode: "exclusive", shared_with: [], exclusive_reason: "fills the card", peak_extra_gib: null },
                       concurrency: { max_copies_per_card: 1, recommended_copies: 1, scaling: null, measured: { date: "d", card: "c", method: "m" }, status: "measured" } })];
  sc.show("status-profile", App()).has(/unmeasured/).has(/1\/2 copies/).has(/exclusive/);
}

// no start / stop buttons: an enabled model is on demand, and the only manual controls are pin / unpin
{
  const pinned = { pinned: true, state: "loaded" };
  ui.state.models = [model("qwen-image-2.1-q8", "image", { state: "loaded" }), model("trellis2-q8", "model3d", pinned),
                     model("vibevoice-1.5b", "audio", { held_s: 90 }), model("qwen3-8b", "llm", { enabled: false, state: "disabled" })];
  const v = sc.show("status-pin", App());
  v.has(/\bpin\b/).has(/\bunpin\b/).has(/pinned/).has(/held for 90 more s/).lacks(/\bstart\b/).lacks(/\bstop\b/).lacks(/starting/);
}

// a model whose licence forbids commercial use wears a "non-commercial" tag (the licence in its tooltip); one that allows it
// does not, nor does one conditioned (a note) or one with no licence known
{
  const qwen = { license: "qwen-research", commercial: false, license_note: "non-commercial purposes only" };
  ui.state.models = [model("qwen-image-2.1-q8", "image", qwen), model("trellis2-q8", "model3d", { license: "mit", commercial: true }),
                     model("minimax-h3", "video", { license: "minimax-h3-community", commercial: true, license_note: "not licensed in the EU" }), model("qwen3-8b", "llm")];
  const v = sc.show("status-licences", App());
  const tagged = v.lines.filter((l) => /non-commercial\s*$/.test(l));
  sc.check("status-licences", tagged.length === 1 && /^qwen-image-2\.1-q8/.test(tagged[0]) && /qwen-research: no commercial use - non-commercial purposes only/.test(tagged[0]),
    `only the Qwen-Image row is tagged, its licence in the tooltip (${JSON.stringify(tagged)})`);
  v.has(/trellis2-q8/).has(/minimax-h3/);
  // a model a card offers and is not downloaded: tagged before it is fetched
  const avail = (id, o) => ({ id, kind: "audio", title: id, about: "", gpus: "0", bytes: 3e9, downloaded: false, drive: "/m", folder: "/m", fits: true, free: 1e11, pull: null, runs_here: true, why_not: null, ...o });
  ui.state.available = [avail("breeze-tts-2", { license: "breezeblue-research-and-non-commercial-1.1", commercial: false }), avail("voxcpm2", { license: "apache-2.0", commercial: true })];
  const w = sc.show("status-licences-available", App());
  const nc = w.lines.filter((l) => /non-commercial\s*$/.test(l) && /^(breeze-tts-2|voxcpm2)/.test(l));
  sc.check("status-licences-available", nc.length === 1 && /^breeze-tts-2/.test(nc[0]), `only the Breeze card is tagged (${JSON.stringify(nc)})`);
}
// a card's process that makes a job names it: the job's title, the label small beside it; without a job nothing is added
{
  ui.state.models = [model("qwen-image-2.1-q8", "image", { state: "busy" })];
  const proc = (o) => ({ pid: 11, vram_mb: 6000, program: "nextcuda", kind: "image", command: "serve", model: "qwen-image-2.1-q8", model_id: "qwen-image-2.1-q8", port: "8101", ...o });
  ui.state.cards = [{ ...card, procs: [proc({ job_title: "Blackbird - pass 12 of 24", job_label: "dad173a2:shots:0:019" })] }];
  sc.show("status-job-title", App()).has(/Blackbird - pass 12 of 24/).has(/dad173a2:shots:0:019/);
  ui.state.cards = [{ ...card, procs: [proc({})] }];
  sc.show("status-no-job", App()).lacks(/Blackbird/).lacks(/shots:0/);
  ui.state.cards = [card];
}

sc.done();
process.exit(process.exitCode ?? 0);
