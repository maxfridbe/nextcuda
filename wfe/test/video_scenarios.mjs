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

sc.done();
