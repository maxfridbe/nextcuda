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
                                       measured: false, headroom_mb: {}, headroom_given: false, page: 8000, ...o });
ui.error = null;
ui.tab = "status";
ui.state = {
  ts: 0, host: { mem_total_mb: 65536, mem_avail_mb: 32768, load1: 0.5, cpus: 8 }, cards: [card],
  models: [model("qwen-image-2.1-q8", "image"), model("trellis2-q8", "model3d"), model("vibevoice-1.5b", "audio"), model("qwen3-8b", "llm")],
  services: [], runs: [], idle_minutes: 10, ports: { video: 8100, chat_ui: 3000 }, maintenance: null, drives: { drives: [] }, available: [], pulls: [], auto_pull: false,
};
sc.show("status", App())
  .has(/RTX 3060/).lacks(/NVIDIA GeForce/).has(/Images/).has(/Audio/).has(/Chat/).has(/\b3D\b/).has(/trellis2-q8/).has(/1 of 1 enabled/);
sc.done();
process.exit(process.exitCode ?? 0);
