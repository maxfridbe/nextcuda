// The OCR page for each reader (Unlimited-OCR, PaddleOCR-VL, olmOCR 2), from the info answer each gives (ocr_takes: what of
// a reading it reads by), and what a click would send. Run after ./wfe/build.sh: node wfe/test/ocr_scenarios.mjs
import { Scenarios, mod, sent, stubFetch } from "./lib.mjs";

const sc = new Scenarios("ocr");
const { App } = await mod("ocr", "components/App.js");
const { state } = await mod("ocr", "state.js");
const { read } = await mod("ocr", "actions.js");

const base = { loading: false, reads: true, max_pages: 24, max_pictures: 8, saves: true, wfe: true, guidance: false, grid: 16, steps_free: true, takes_loras: false,
               defaults: { width: 0, height: 0, steps: 0, cfg: 1, sampler: "euler", schedule: "shift", shift: 0 }, loras: [], options: [], samplers: [], schedules: [] };
const unlimited = { ...base, model: "unlimited-ocr-q8", arch: "unlimited-ocr", ocr_takes: ["tiles", "figures", "columns", "guard", "each_page", "boxes"],
  prompts: [{ id: "parse", prompt: "document parsing.", title: "Document → Markdown with boxes (the model's own)" }, { id: "free", prompt: "Free OCR.", title: "Plain text" }] };
const paddle = { ...base, model: "paddleocr-vl-1.6", arch: "paddleocr-vl", ocr_takes: ["columns", "guard"],
  prompts: [{ id: "ocr", prompt: "OCR:", title: "Text (the model's own)" }, { id: "table", prompt: "Table Recognition:", title: "A table" }] };
const olm = { ...base, model: "olmocr-2-7b-q8", arch: "olmocr2", ocr_takes: ["columns"],
  prompts: [{ id: "page", prompt: "page", title: "Page → Markdown (the model's own)" }] };
const drawing = { ...base, model: "qwen-image-2.1-q8", arch: "qwen-image-2.1", reads: false, ocr_takes: [] };

const page = (n) => ({ name: `p${n}.png`, url: `data:image/png;base64,P${n}`, width: 800, height: 1000 });
function open(info, pages = [page(1)], form = {}) {
  state.info = JSON.parse(JSON.stringify(info));
  state.pages = pages;
  // a form kept from another model: tiles off, no guard, one document, figures off
  state.form = { preset: "parse", prompt: "", tiles: false, maxTokens: 4096, temperature: 0, guard: false, figures: false, perPage: false, columns: true, ...form };
  state.result = null;
  state.busy = false;
  state.error = null;
  stubFetch(() => ({ text: "Hello", pages: 1, tokens: 3, prompt_tokens: 10, seconds: 1 }));
}
const reqs = () => sent.filter((s) => /ocr/.test(s.url)).map((s) => s.body);

open(unlimited);
sc.show("unlimited-one-page", App()).has(/tiles/).has(/split columns/).has(/read inside pictures/).has(/repetition guard/).has(/max tokens/)
  .has(/Document → Markdown with boxes/).has(/\[button Read the page\]/).lacks(/each page on its own/);
await read();
sc.check("unlimited-one-page", reqs().length === 1 && reqs()[0].tiles === false && reqs()[0].figures === false && reqs()[0].ngram === 0 && "columns" in reqs()[0],
  `unlimited sends tiles, figures, columns, ngram (${JSON.stringify(reqs()[0] && Object.keys(reqs()[0]))})`);
sc.show("unlimited-reading-with-boxes-tab-absent", App()).lacks(/\[button Boxes/, "the Boxes tab for a reading without boxes").has(/\[button Document\]/);

open(unlimited, [page(1), page(2)], { perPage: false });
sc.show("unlimited-two-pages-as-one", App()).has(/each page on its own/).has(/as one document/).has(/the model's own multi-page reading/).has(/\[button Read 2 pages\]/);
await read();
sc.check("unlimited-two-pages-as-one", reqs().length === 1 && reqs()[0].each_page === false && reqs()[0].images.length === 2, "one request, both pages, each_page false");

open(paddle, [page(1), page(2)], { perPage: false });
sc.show("paddle-two-pages", App()).has(/split columns/).has(/repetition guard/).lacks(/tiles/).lacks(/read inside pictures/).lacks(/as one document/).lacks(/each page on its own/)
  .has(/Text \(the model's own\)/).has(/A table/);
await read();
sc.check("paddle-two-pages", reqs().length === 2 && reqs().every((b) => !("tiles" in b) && !("figures" in b) && !("each_page" in b) && b.ngram === 0 && "columns" in b),
  `paddle reads page by page, without tiles / figures / each_page (${JSON.stringify(reqs().map((b) => Object.keys(b)))})`);

open(olm, [page(1)], { preset: "custom" });
sc.show("olmocr-custom-instruction", App()).has(/split columns/).lacks(/repetition guard/).lacks(/tiles/).lacks(/read inside pictures/)
  .has(/placeholder="Your instruction"\]/).lacks(/grounding/);
await read();
sc.check("olmocr-custom-instruction", reqs().length === 1 && !("ngram" in reqs()[0]) && !("tiles" in reqs()[0]) && !("figures" in reqs()[0]), "olmOCR is not sent tiles / figures / ngram");

// a reading that holds boxes shows the Boxes tab
open(unlimited);
state.result = { text: "<|ref|>title<|/ref|><|det|>title [10, 10, 500, 60]<|/det|>\nHello", pages: 1, tokens: 5, seconds: 1 };
state.resultPages = [page(1)];
state.busy = false;
sc.show("unlimited-reading-with-boxes", App()).has(/\[button Boxes\]/);
// an image model on this page
open(drawing);
sc.show("image-model-here", App()).has(/makes pictures; this page reads them/).has(/\[button Read the page disabled\]/);

// the licence: a reader whose card forbids commercial use wears the tag in the header; one that allows it does not
open({ ...unlimited, license: "mit", commercial: true });
sc.show("licence-commercial", App()).has(/unlimited-ocr-q8/).lacks(/non-commercial/).lacks(/no commercial use/);
open({ ...paddle, license: "made-up-research", commercial: false });
sc.show("licence-non-commercial", App()).has(/paddleocr-vl-1\.6 \(made-up-research: no commercial use\) non-commercial/);

sc.done();
