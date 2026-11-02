// The video studio's new-clip form: references and control do not mix (the server refuses the pair), so the form shows the one
// in use and sends only it; the upscaler is asked only when there is an upscale. Run after ./wfe/build.sh:
//   node wfe/test/video_scenarios.mjs
import { Scenarios, mod, press, sent, stubFetch } from "./lib.mjs";

const sc = new Scenarios("video");
const { Creation, form } = await mod("video", "components/Creation.js");

function open(patch) {
  Object.assign(form, { refImages: [], refVideos: [], refAudios: [], controlVideo: "", controlMask: "", controlSource: "", upscale: 1.5, chain: false }, patch);
  stubFetch(() => ({ ok: true, result: { queued: true, position: 1 } }));
}
const rpc = () => sent.find((s) => /rpc\/generate/.test(s.url))?.body ?? {};

open({});
sc.show("plain", Creation()).has(/References/).has(/Control/).has(/upscaler/).has(/up to 15 s per clip/).lacks(/4–15 s/).lacks(/B65/);
await press(Creation(), /Generate/);
sc.check("plain", !("ref_images" in rpc()) && !("control_video" in rpc()) && rpc().upscaler === "esrgan-general", `a plain request (${JSON.stringify(Object.keys(rpc()))})`);

open({ upscale: 1 });
sc.show("no-upscale", Creation()).lacks(/upscaler/, "an upscaler choice with nothing to upscale");

open({ refImages: ["ref_1_a.png"] });
sc.show("with-references", Creation()).has(/References/).lacks(/a pose \/ depth \/ edge video/, "the Control panel beside references");
open({ refImages: ["ref_1_a.png"], controlVideo: "ctl.mp4" });
await press(Creation(), /Generate/);
sc.check("with-references", rpc().ref_images?.length === 1 && !("control_video" in rpc()) && !("control_strength" in rpc()), "references win: control is not sent with them");

open({ controlVideo: "ctl.mp4" });
sc.show("with-control", Creation()).has(/Control/).lacks(/pictures, clips, sounds the prompt names/, "the References panel beside a control");
await press(Creation(), /Generate/);
sc.check("with-control", rpc().control_video === "ctl.mp4" && rpc().control_strength === 1, "control sent with its strength");

// the pace: standard asks for the steps and the sampler; Pika Flex is six steps in one word, so the steps, sampler and
// schedule are not asked and not sent, and the other choices (upscale, references, control, soundtrack) stay
open({ speed: "standard", steps: 8, sampler: "euler" });
sc.show("pace-standard", Creation()).has(/pace/).has(/Standard/).has(/Pika Flex · 6 steps/).has(/steps/).has(/ComfyUI.s samplers/).has(/The sigma schedule/);
await press(Creation(), /Generate/);
sc.check("pace-standard", !("speed" in rpc()) && rpc().steps === 8 && rpc().sampler === "euler", `standard sends the steps and the sampler (${JSON.stringify(rpc())})`);
open({ speed: "pika-flex", sampler: "euler" });
sc.show("pace-pika", Creation()).has(/6 \(set by Pika Flex\)/).has(/upscaler/).has(/References/).lacks(/ComfyUI.s samplers/).lacks(/The sigma schedule/).has(/6 steps/);
await press(Creation(), /Generate/);
sc.check("pace-pika", rpc().speed === "pika-flex" && !("sampler" in rpc()) && !("schedule" in rpc()), `Pika Flex sends the word alone (${JSON.stringify(rpc())})`);
open({ speed: "pika-flex", refImages: ["ref_1_a.png"] });
await press(Creation(), /Generate/);
sc.check("pace-pika-refs", rpc().speed === "pika-flex" && rpc().ref_images?.length === 1, "Pika Flex with references");
open({ speed: "pika-flex", controlVideo: "ctl.mp4" });
await press(Creation(), /Generate/);
sc.check("pace-pika-control", rpc().speed === "pika-flex" && rpc().control_video === "ctl.mp4", "Pika Flex with a control video");
open({ speed: "pika-flex", soundtrack: "song.mp3", soundtrackStart: 3 });
await press(Creation(), /Generate/);
sc.check("pace-pika-sound", rpc().speed === "pika-flex" && rpc().soundtrack === "song.mp3", "Pika Flex with a soundtrack");
open({ speed: "standard" });

// a clip's detail names its pace: a Pika Flex job (the server stores its expansion: lcm sampler on the knots) says so
{
  const { Detail } = await mod("video", "components/Detail.js");
  const { state } = await mod("video", "state.js");
  state.detail = { kind: "queued", data: { label: "a", steps: 6, sampler: "lcm", knots: "1000,750,500,250,100,40,0" } };
  sc.show("detail-pika", Detail()).has(/pace/).has(/Pika Flex/);
  state.detail = { kind: "queued", data: { label: "b", steps: 8 } };
  sc.show("detail-standard", Detail()).has(/pace/).has(/standard/).lacks(/Pika Flex/);
  state.detail = null;
}

// the licence: the denoiser picker tags a denoiser whose card forbids commercial use (the licence in its tooltip); the ones
// that allow it, with conditions or not, are not tagged
{
  const { CanvasPicker } = await mod("video", "components/CanvasPicker.js");
  const { state } = await mod("video", "state.js");
  const eng = (quant, o = {}) => ({ quant, file: `${quant}.gguf`, path: "/m", gib: 10, ready: true, default: false, inputs: {}, ...o });
  const pick = (name, engines) => {
    state.plan = { engines, canvases: [{ width: 640, height: 480, aspect: "4:3", note: "", mpx: 0.31, cost: 1, default: true }], defaults: { seconds: 5, steps: 8, width: 640, height: 480, engine: "INT8" },
                   cap_gib: 24, gib_per_token: 0.001, margin_gib: 1, step_a: 0, step_b: 0.001, gpus: [], measured: {} };
    return sc.show(name, CanvasPicker({ width: 640, height: 480, engine: "INT8", seconds: 5, steps: 8, onPick: () => {} }));
  };
  pick("licence-picker-tagged", [eng("INT8", { default: true, license: "minimax-h3-community", commercial: true, license_note: "not licensed in the EU" }), eng("Q6_K", { license: "made-up-research", commercial: false })])
    .has(/\[button Q6_K 10G .*non-commercial/).has(/made-up-research: no commercial use/).lacks(/\[button INT8 10G .*non-commercial/);
  pick("licence-picker-none", [eng("INT8", { default: true, license: "minimax-h3-community", commercial: true }), eng("Q6_K")]).lacks(/non-commercial/);
}

sc.done();
