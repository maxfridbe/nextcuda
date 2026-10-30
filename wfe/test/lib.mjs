// The scenario tests' shared part: the built pages (dist/wfe, after ./wfe/build.sh) loaded as ES modules, a page rendered
// headlessly from a made-up state (App() builds a vnode tree; no DOM, no server), its visible text collected, and fetch
// stubbed so what a click would send can be read. No server is contacted: every network call goes to the stub.
import { cpSync, existsSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const dist = join(here, "..", "..", "dist", "wfe");
/** where each scenario's full text is written (read them: anything that reads wrong for its state is a bug) */
export const outDir = process.env.NS_SCEN_OUT || join(here, "..", "..", "dist", "scenarios");

let root = null;
/** The built pages as a module root ("type": "module" beside them, as check.mjs does) */
export function prepare() {
  if (root) return root;
  if (!existsSync(join(dist, "image"))) throw new Error(`${dist}: not built (./wfe/build.sh first)`);
  root = join(tmpdir(), `nswfe-scenarios-${process.pid}`);
  rmSync(root, { recursive: true, force: true });
  mkdirSync(root, { recursive: true });
  cpSync(dist, root, { recursive: true });
  writeFileSync(join(root, "package.json"), '{"type":"module"}');
  process.on("exit", () => rmSync(root, { recursive: true, force: true }));
  globalThis.localStorage = { getItem: () => null, setItem: () => {} };
  globalThis.document = { getElementById: () => null, baseURI: "http://box/" };
  globalThis.window = { prompt: () => null, confirm: () => true };
  globalThis.location = { protocol: "http:", hostname: "box", href: "http://box/" };
  globalThis.requestAnimationFrame = () => 0;
  return root;
}

/** import a built module: lib("image", "components/App.js") */
export async function mod(app, path) {
  return import(join(prepare(), app, "src", path));
}

/** Every request the page makes, answered by `answer(url, init)` (a JSON-able value) */
export const sent = [];
export function stubFetch(answer = () => ({})) {
  sent.length = 0;
  globalThis.fetch = async (url, init = {}) => {
    let body;
    try { body = typeof init.body === "string" ? JSON.parse(init.body) : init.body; } catch { body = init.body; }
    sent.push({ url: String(url), method: init.method ?? "GET", body });
    return new Response(JSON.stringify(answer(String(url), body) ?? {}), { status: 200, headers: { "content-type": "application/json" } });
  };
}

const BLOCK = new Set(["div", "section", "label", "button", "option", "textarea", "h1", "h2", "h3", "h4", "li", "pre", "figure", "figcaption", "a", "select", "input", "audio", "img", "p", "span.hint"]);

/** The page as the reader sees it, a line a block: text, `[input ...]`, `[button ...]`, `[select a | b]`, with the
 *  title / placeholder / alt attributes a reader gets on hover or in an empty field */
export function dump(vnode) {
  const lines = [];
  let cur = "";
  const flush = () => { const t = cur.replace(/\s+/g, " ").trim(); if (t) lines.push(t); cur = ""; };
  const attr = (n, k) => n.data?.attrs?.[k] ?? n.data?.props?.[k];
  const walk = (n, depth) => {
    if (n == null || n === false) return;
    if (typeof n === "string" || typeof n === "number") { cur += ` ${n}`; return; }
    if (n.sel === undefined) { if (n.text != null) cur += ` ${n.text}`; return; }
    const tag = n.sel.split(/[.#]/)[0];
    const hidden = n.data?.style?.display === "none";
    if (hidden) return;
    if (tag === "input") {
      flush();
      const type = attr(n, "type") ?? "text";
      const bits = [type];
      if (attr(n, "placeholder")) bits.push(`placeholder="${attr(n, "placeholder")}"`);
      if (attr(n, "title")) bits.push(`title="${attr(n, "title")}"`);
      if (type === "checkbox") bits.push(n.data?.props?.checked ? "checked" : "unchecked");
      else if (type !== "file" && n.data?.props?.value) bits.push(`value=${n.data.props.value}`);
      if (attr(n, "disabled")) bits.push("disabled");
      if (type === "file") bits.push(`accept=${attr(n, "accept") ?? ""}`);
      lines.push(`[input ${bits.join(" ")}]`);
      return;
    }
    if (tag === "textarea") {
      flush();
      lines.push(`[textarea placeholder="${attr(n, "placeholder") ?? ""}"${n.data?.props?.value ? ` value="${n.data.props.value}"` : ""}]`);
      return;
    }
    if (tag === "select") {
      flush();
      const opts = (n.children ?? []).map((o) => (o.text ?? (o.children ?? []).map((c) => c.text ?? "").join("")));
      lines.push(`[select ${opts.join(" | ")}]`);
      return;
    }
    if (tag === "button") {
      flush();
      const save = lines.length;
      cur = "";
      for (const c of n.children ?? []) walk(c, depth + 1);
      const label = cur.replace(/\s+/g, " ").trim();
      cur = "";
      lines.push(`[button ${label}${attr(n, "title") ? ` title="${attr(n, "title")}"` : ""}${attr(n, "disabled") ? " disabled" : ""}]`);
      void save;
      return;
    }
    if (tag === "img" || tag === "audio") {
      flush();
      lines.push(`[${tag}${attr(n, "alt") ? ` alt="${attr(n, "alt")}"` : ""}]`);
      return;
    }
    const isBlock = BLOCK.has(tag);
    if (isBlock) flush();
    if (attr(n, "title") && tag !== "a") cur += ` (${attr(n, "title")})`;
    if (n.text != null) cur += ` ${n.text}`;
    for (const c of n.children ?? []) walk(c, depth + 1);
    if (isBlock) flush();
  };
  walk(vnode, 0);
  flush();
  return lines;
}

/** The click handler of the first button whose label matches, called (what a press does) */
export async function press(vnode, re) {
  let found = null;
  const walk = (n) => {
    if (found || !n || typeof n !== "object") return;
    if (n.sel && n.sel.split(/[.#]/)[0] === "button" && n.data?.on?.click) {
      const label = (n.children ?? []).map((c) => c.text ?? (c.children ?? []).map((x) => x.text ?? "").join("")).join("").trim();
      if (re.test(label)) found = n;
    }
    for (const c of n.children ?? []) walk(c);
  };
  walk(vnode);
  if (!found) throw new Error(`no button ${re}`);
  await found.data.on.click(new Event("click"));
}

/** One scenario: the text of what was rendered, written to outDir/<app>/<name>.txt, with checks */
export class Scenarios {
  constructor(app) {
    this.app = app;
    this.failed = 0;
    this.ran = 0;
    mkdirSync(join(outDir, app), { recursive: true });
  }
  /** render `App()` (the caller has set the state), save and return its lines */
  show(name, vnode) {
    const lines = dump(vnode);
    writeFileSync(join(outDir, this.app, `${name}.txt`), lines.join("\n") + "\n");
    this.ran++;
    return new View(this, name, lines);
  }
  check(name, ok, what) {
    if (!ok) { this.failed++; console.log(`FAIL ${this.app}/${name}: ${what}`); }
  }
  done() {
    console.log(`${this.app}: ${this.ran} scenarios, ${this.failed} failed (text in ${join(outDir, this.app)})`);
    if (this.failed) process.exitCode = 1;
  }
}

class View {
  constructor(sc, name, lines) { this.sc = sc; this.name = name; this.lines = lines; this.text = lines.join("\n"); }
  has(re, what = String(re)) { this.sc.check(this.name, re.test(this.text), `should show ${what}`); return this; }
  lacks(re, what = String(re)) { this.sc.check(this.name, !re.test(this.text), `should not show ${what}`); return this; }
}
