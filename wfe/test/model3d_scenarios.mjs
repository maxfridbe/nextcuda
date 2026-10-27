// The 3D page for each engine: TRELLIS.2 (textured, one view), Hunyuan3D without its paint stage, UniRig / RigNet (rig only: no
// steps, no guidance), AnimateAnyMesh (animates), from the info answer each gives, and what a click would send.
// Run after ./wfe/build.sh: node wfe/test/model3d_scenarios.mjs
import { Scenarios, mod, sent, stubFetch } from "./lib.mjs";

const sc = new Scenarios("model3d");
const { Create } = await mod("model3d", "components/Create.js");
const { state } = await mod("model3d", "state.js");
const { go } = await mod("model3d", "actions.js");

const zero = { steps: 0, cfg: 0, resolution: 0, faces: 0, texture_size: 0, frames: 0, fps: 0 };
const eng = (model, arch, abilities, defaults) => ({ model, arch, wfe: true, saves: true, options: [], report: [], abilities: { generate: false, texture: false, views: 0, rig: false, animate: false, ...abilities }, defaults: { ...zero, ...defaults } });
const trellis = eng("trellis2-q8", "trellis2", { generate: true, texture: true, views: 1 }, { steps: 12, cfg: 7.5, resolution: 1024, faces: 300000, texture_size: 2048 });
const hunyuan = eng("hunyuan3d-2.1-shape", "hunyuan3d", { generate: true, views: 1 }, { steps: 50, cfg: 5, resolution: 384, faces: 40000, texture_size: 2048 });
const unirig = eng("unirig", "unirig", { rig: true }, {});
const rignet = eng("rignet", "rignet", { rig: true }, { faces: 3000 });
const animate = eng("animateanymesh", "animateanymesh", { animate: true }, { steps: 63, cfg: 3, frames: 16, fps: 10 });

const pic = { name: "a.png", data: "data:image/png;base64,AAAA", side: "front" };
function open(info, mode, form = {}) {
  state.info = JSON.parse(JSON.stringify(info));
  state.mode = mode;
  // a form kept from another model: a seed, steps and guidance, a texture size
  state.form = { pictures: [pic], texture: true, seed: 5, steps: 30, cfg: 4, resolution: null, faces: null, textureSize: 512, removeBackground: true,
                 mesh: "data:model/gltf-binary;base64,AAAA", meshName: "m.glb", file: "", prompt: "", frames: null, options: {}, ...form };
  state.busy = false;
  state.error = null;
  state.status = "idle";
  stubFetch(() => ({ data: [{ url: "/v1/model3d/files/a.glb", bytes: 1, created: 1 }] }));
}
const body = () => sent.find((s) => /generate|rig|generations/.test(s.url))?.body ?? {};

open(trellis, "make");
sc.show("trellis2-make", Create()).has(/a picture of the object/).lacks(/up to \d+ views/).has(/textured/).has(/\bsteps\b/).has(/guidance/).has(/resolution/).has(/texture/).has(/triangles/)
  .has(/cut out the background/).has(/\[button Make\]/).lacks(/frames/);
await go();
sc.check("trellis2-make", body().seed === 5 && body().steps === 30 && body().cfg === 4 && body().texture === true && body().texture_size === 512, `trellis sends steps / cfg / texture (${JSON.stringify(Object.keys(body()))})`);

open(hunyuan, "make");
sc.show("hunyuan3d-no-paint", Create()).lacks(/textured/).lacks(/placeholder="2048"/, "a texture size for an engine that does not paint").has(/triangles/).has(/\[button Make\]/);
await go();
sc.check("hunyuan3d-no-paint", body().texture === false && !("texture_size" in body()) || body().texture === false, "no texture asked of an engine that does not paint");

for (const [name, info] of [["unirig", unirig], ["rignet", rignet]]) {
  open(info, "rig");
  sc.show(`${name}-rig`, Create()).has(/the mesh/).has(/\[button Rig\]/).lacks(/\bsteps\b/).lacks(/guidance/).lacks(/resolution/).lacks(/textured/).lacks(/cut out the background/).lacks(/frames/).lacks(/the motion/)
    .has(/seed/);
  await go();
  sc.check(`${name}-rig`, /rig/.test(sent[0]?.url ?? "") && body().seed === 5 && !("steps" in body()) && !("cfg" in body()), `a rigger is sent its seed only (${JSON.stringify(Object.keys(body()))})`);
}

open(animate, "rig", { prompt: "a dragon flapping its wings" });
sc.show("animateanymesh", Create()).has(/the motion/).has(/frames/).has(/\bsteps\b/).has(/guidance/).has(/\[button Animate\]/).lacks(/resolution/).lacks(/textured/);
await go();
sc.check("animateanymesh", body().steps === 30 && body().cfg === 4 && body().prompt === "a dragon flapping its wings", "an animator takes steps, cfg and the motion");

sc.done();
