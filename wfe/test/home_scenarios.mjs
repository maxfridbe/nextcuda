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
sc.done();
process.exit(process.exitCode ?? 0);
