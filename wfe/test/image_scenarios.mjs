// The image page, rendered for each model family from the info answer its engine gives (image.rs info()), and what a
// click would send (fetch stubbed): qwen-image normal / turbo, generate / edit, MOSS full / turbo, Ming, an upscaler,
// StarVector. Run after ./wfe/build.sh: node wfe/test/image_scenarios.mjs
import { Scenarios, mod, sent, stubFetch } from "./lib.mjs";

const sc = new Scenarios("image");
const { App } = await mod("image", "components/App.js");
const { state } = await mod("image", "state.js");
const { adoptDefaults, generate, setBudget } = await mod("image", "actions.js");

const base = {
  model: "m", loading: false, reads: false, edits: false, guidance: true, negative: true, steps_free: true, takes_loras: true,
  grid: 16, transparent: false, shift: true, sizes: null, min_side: 256, max_side: 4096, max_pictures: 8, max_pages: 24, saves: true, wfe: true,
  samplers: ["euler"], schedules: ["shift"], options: [], vectorize: false, upscale: [], prompts: [],
  loras: [{ id: "detail", title: "Detail", loaded: null }],
  defaults: { width: 1024, height: 1024, steps: 40, cfg: 4, sampler: "euler", schedule: "shift", shift: 1 },
};
const qwen = { ...base, model: "qwen-image-2.1-q8", arch: "qwen-image-2.1", edits: true, transparent: true, shift: false };
const qwenTurbo = { ...qwen, model: "qwen-image-2.1-turbo", guidance: false, negative: false, steps_free: false,
  defaults: { ...base.defaults, steps: 6, cfg: 1 } };
const mossSizes = {
  ratios: ["1:8", "1:4", "1:3", "1:2", "9:16", "2:3", "3:4", "1:1", "4:3", "3:2", "16:9", "2:1", "3:1", "4:1", "8:1"],
  pixels: [262144, 1048576, 1572864, 2097152, 4194304],
};
const moss = { ...base, model: "moss-image-sft-q8", arch: "moss-image", sizes: mossSizes, min_side: 64, max_side: 4096, max_pixels: 4220160,
  defaults: { ...base.defaults, steps: 50, cfg: 3, shift: 5 } };
const mossTurbo = { ...moss, model: "moss-image-sft-turbo4-q8", guidance: false, negative: false, shift: false,
  defaults: { ...moss.defaults, steps: 4, cfg: 1 } };
const ming = { ...base, model: "ming-image-0.1-design", arch: "ming-image", negative: false, transparent: true, shift: false, takes_loras: false,
  min_side: 64, defaults: { ...base.defaults, steps: 12, cfg: 1, shift: 6 } };
const upscaler = { ...base, model: "real-esrgan", arch: "realesrgan", guidance: false, negative: false, takes_loras: false, loras: [],
  upscale: [{ variant: "x4", about: "general", scales: [2, 4], noise: [] }], defaults: { width: 0, height: 0, steps: 0, cfg: 1, sampler: "euler", schedule: "shift", shift: 0 } };
const starvector = { ...base, model: "starvector-1b-im2svg", arch: "starvector", guidance: false, negative: false, takes_loras: false, loras: [],
  vectorize: true, defaults: { width: 224, height: 224, steps: 0, cfg: 1, sampler: "euler", schedule: "shift", shift: 0 } };

const answer = (url) => url.includes("generations") || url.includes("edits")
  ? { created: 1, data: [{ url: "/v1/images/files/a.png", seed: 7, revised_prompt: "p", width: 1024, height: 1024 }], nextsycl: { steps: 4, loras: [], seconds: 1, wh: null } }
  : url.includes("history") ? { data: [] } : {};

/** a fresh page for a model; `form` is what an earlier visit to another model left in the saved form */
function open(info, form = {}, pictures = []) {
  state.info = JSON.parse(JSON.stringify(info));
  state.form = { prompt: "a fox", negative: "", width: 1024, height: 1024, steps: null, n: 1, seed: null, sampler: "", schedule: "", cfg: null,
                 shift: null, rgba: false, loras: {}, options: {}, ...form };
  state.pictures = pictures;
  state.sizeFromPictures = true;
  state.panels = { sampling: true, loras: true, pictures: true, create: true };
  state.results = [];
  state.history = [];
  state.error = null;
  state.busy = false;
  state.status = "idle";
  adoptDefaults();
  stubFetch(answer);
}
const body = () => sent.find((s) => /generations|edits/.test(s.url))?.body;
const go = async () => { await generate(); return { url: sent.find((s) => /generations|edits/.test(s.url))?.url ?? "", b: body() ?? {} }; };
const stale = { negative: "blurry", steps: 20, cfg: 7, shift: 3, rgba: true, loras: { detail: 1 } };
const pic = { url: "data:image/png;base64,AAAA", name: "fox.png", w: 640, h: 480 };

// qwen-image, normal: everything the page offers
open(qwen);
sc.show("qwen-normal-generate", App())
  .has(/negative prompt/).has(/guidance \(cfg\)/).has(/\bsteps\b/).has(/transparent \(RGBA\)/).has(/LoRAs \(0 on\)/)
  .has(/Pictures to edit \(none: a new picture\)/).has(/a red fox in fresh snow/).lacks(/shift\b.*placeholder/)
  .has(/\[button Generate\]/).lacks(/\[button Edit/);
{
  const { b } = await go();
  sc.check("qwen-normal-generate", b.size === "1024x1024" && b.prompt === "a fox" && !("negative_prompt" in b), "plain request: size and prompt only");
}

// qwen-image, an edit: pictures first, the prompt becomes instructions, the size follows the picture
open(qwen, {}, [pic]);
{
  const v = sc.show("qwen-normal-edit", App());
  v.has(/Pictures to edit \(1: an edit\)/).has(/what to change: put the hat from picture 2/).has(/\[button Edit\]/)
    .has(/follows the pictures/).lacks(/\[button Generate/).lacks(/width/);
  const iPic = v.lines.findIndex((l) => /Pictures to edit/.test(l)), iPrompt = v.lines.findIndex((l) => /\[textarea/.test(l));
  sc.check("qwen-normal-edit", iPic >= 0 && iPic < iPrompt, "the pictures are asked before the instructions");
  const { url, b } = await go();
  sc.check("qwen-normal-edit", /edits/.test(url) && b.images?.length === 1 && !("size" in b), "an edit goes to the edits route with its picture and no size");
}

// qwen-image turbo: a few-step entry at cfg 1: steps, cfg and the negative prompt are not the page's to ask, nor sent
open(qwenTurbo, stale);
sc.show("qwen-turbo-generate", App())
  .lacks(/negative prompt/).has(/steps 6 \(this entry's own\)/).has(/guidance \(cfg\) 1 \(this entry runs unguided\)/).has(/transparent/);
{
  const { b } = await go();
  sc.check("qwen-turbo-generate", !("steps" in b) && !("cfg" in b) && !("negative_prompt" in b), `turbo sends no steps / cfg / negative (${JSON.stringify(b)})`);
}

// MOSS-Image full: sizes as OpenMOSS's presets inside 64..4096 and its pixel limit, no alpha, a shift, a negative prompt
open(moss, stale);
sc.show("moss-full", App())
  .has(/negative prompt/).has(/placeholder="3"/, "cfg default 3").has(/shift/).lacks(/transparent/).lacks(/Pictures to edit/).has(/LoRAs/)
  .has(/\[button 1:1\]/).has(/\[button 16:9\]/).has(/\[button 1M /).has(/\[button 2M /).has(/\[button 512² /)
  .lacks(/\[button 21:9\]/).has(/\[button 1:8\]/, "1:8 fits 4096 a side at 1M").has(/up to 4 megapixels a picture/)
  .has(/width .*step|\[input number/);
{
  const { b } = await go();
  sc.check("moss-full", !("background" in b), "no transparent background asked of an engine without alpha");
  sc.check("moss-full", b.shift === 3 && b.negative_prompt === "blurry" && b.cfg === 7, "MOSS takes shift, negative and cfg as given");
  setBudget(2097152, null);
  sc.check("moss-full", state.form.width === 1456 && state.form.height === 1456, `2M at 1:1 is 1456x1456 (${state.form.width}x${state.form.height})`);
  setBudget(4194304, 16 / 9);
  sc.check("moss-full", state.form.width === 2736 && state.form.height === 1536, `4M at 16:9 is 2736x1536 (${state.form.width}x${state.form.height})`);
  setBudget(4194304, 8);
  sc.check("moss-full", state.form.width === 2736, "4M at 8:1 would pass 4096 a side: not applied");
}
// a size left by another model is brought inside this one's sides and pixel limit
open(moss, { width: 3072, height: 2048 });
sc.check("moss-full", state.form.width * state.form.height <= 4220160, `a 3072x2048 form comes inside 4.2 M pixels (${state.form.width}x${state.form.height})`);
open(moss, { width: 8192, height: 512 });
sc.check("moss-full", state.form.width === 4096, `an 8192 wide form comes in to 4096 (${state.form.width})`);

// MOSS-Image turbo (a LoRA merged at load): unguided, the scheduler's own levels
open(mossTurbo, stale);
sc.show("moss-turbo", App())
  .lacks(/negative prompt/).has(/guidance \(cfg\) 1 \(this entry runs unguided\)/).has(/\[button 1:1\]/);
{
  const { b } = await go();
  sc.check("moss-turbo", !("cfg" in b) && !("negative_prompt" in b) && !("shift" in b), "turbo sends no cfg / negative / shift");
}

// Ming-Image: text to image only, a zero-condition guidance (no negative), fixed shift, no LoRAs
open(ming, { ...stale });
sc.show("ming", App())
  .lacks(/negative prompt/).has(/guidance \(cfg\)/).has(/transparent \(RGBA\)/).lacks(/LoRAs/).lacks(/Pictures to edit/).lacks(/placeholder="model's"/)
  .lacks(/LoRAs merged/);
{
  const { b } = await go();
  sc.check("ming", !("negative_prompt" in b) && !("shift" in b) && !("loras" in b) && b.background === "transparent",
    `Ming sends no negative / shift / loras; transparency yes (${JSON.stringify(b)})`);
}

// an upscaler and a vectorizer served on this page: it says what they are and offers no drawing form
open(upscaler);
sc.show("upscaler", App()).has(/enlarges pictures/).has(/does not draw from a prompt/).lacks(/negative prompt/).lacks(/\[button Generate/).lacks(/steps/);
open(starvector);
sc.show("starvector", App()).has(/writes pictures as SVG/).has(/does not draw from a prompt/).lacks(/guidance/).lacks(/LoRAs/);

sc.done();
