/** What the buttons do: build the request from the form, send it, keep the result. */
import { api, song } from "./api.js";
import { render, saveForm, state } from "./state.js";
import type { AudioSpeechBody, VoiceAdd } from "./gen/routes.js";
import type { Song } from "./types.js";

export function randomSeed(): void {
  state.form.seed = Math.floor(Math.random() * 1_000_000);
  saveForm();
  render();
}

/** Use a song's description, lyrics and seed again (speech: its text, voice, language, instructions). */
export function reuse(s: Song): void {
  if (state.info?.speech) {
    state.form.text = s.prompt;
    if (s.voice) state.form.voice = s.voice;
    state.form.language = s.language ?? "";
    state.form.instructions = s.instructions ?? "";
  } else {
    state.form.prompt = s.prompt;
    state.form.lyrics = s.lyrics;
  }
  state.form.seed = s.seed;
  saveForm();
  render();
}

/** A structure tag on a line of its own, at the end of the lyrics. */
export function addTag(tag: string): void {
  const l = state.form.lyrics.replace(/\s+$/, "");
  state.form.lyrics = `${l}${l ? "\n" : ""}[${tag}]\n`;
  saveForm();
  render();
}

export async function refreshVoices(): Promise<void> {
  try {
    state.voices = await api.voices();
    render();
  } catch {
    /* a server without an output directory */
  }
}

/** Ask for a name and save a voice: the recording picked (with its transcript), or a file made here. */
export async function saveVoice(from: { sample?: string; text?: string; file?: string; description?: string | null }): Promise<void> {
  const name = (window.prompt("Save this voice as (lowercase letters, digits, - and _):") ?? "").trim();
  if (!name) return;
  try {
    const body: VoiceAdd = { name };
    if (from.sample) body.sample = from.sample;
    if (from.file) body.file = from.file;
    if (from.text) body.text = from.text;
    if (from.description) body.description = from.description;
    const v = await api.addVoice(body);
    state.status = `saved voice ${v.name} (${v.seconds.toFixed(1)} s)`;
    state.error = null;
    await refreshVoices();
  } catch (e) {
    state.error = e instanceof Error ? e.message : String(e);
    render();
  }
}

export async function deleteVoice(name: string): Promise<void> {
  if (!window.confirm(`Delete the saved voice ${name}?`)) return;
  try {
    await api.removeVoice(name);
    if (state.form.voice === name) state.form.voice = "";
    await refreshVoices();
  } catch (e) {
    state.error = e instanceof Error ? e.message : String(e);
    render();
  }
}

export async function refreshHistory(): Promise<void> {
  try {
    state.history = await api.history();
    render();
  } catch {
    /* transient */
  }
}

/** A recording chosen for cloning, kept as a data: URL. */
export function pickReference(file: File | null): void {
  if (!file) {
    state.form.refAudio = "";
    state.form.refName = "";
    render();
    return;
  }
  const r = new FileReader();
  r.onload = () => {
    state.form.refAudio = String(r.result ?? "");
    state.form.refName = file.name;
    render();
  };
  r.readAsDataURL(file);
}

/** A multi-speaker script's voices: the saved voices picked for the first speakers, in order, up to the first gap */
export function cast(sp: { speakers?: number | null }): string[] {
  if ((sp.speakers ?? 0) < 2) return [];
  const known = new Set(state.voices.map((v) => v.name));
  const out: string[] = [];
  for (const n of (state.form.voices ?? []).slice(0, sp.speakers ?? 0)) {
    if (!n || !known.has(n)) break;
    out.push(n);
  }
  return out.length >= 2 ? out : [];
}

/** Whether the voice picked is one of the model's own (a form kept from another model may hold any name) */
const builtin = (v: string): boolean => (state.info?.speech?.voices ?? []).some((x) => x.toLowerCase() === v.toLowerCase());

/** Whether the form can go: songs need a description, speech its text (and what the model needs besides). */
export function canGo(): boolean {
  const f = state.form;
  const sp = state.info?.speech;
  if (!sp) return !!f.prompt.trim();
  if (!f.text.trim()) return false;
  if (sp.requires === "voice" && !builtin(f.voice)) return false;
  if (sp.requires === "instructions" && !f.instructions.trim()) return false;
  if (sp.requires === "reference" && !f.refAudio && !state.voices.some((v) => v.name === f.voice) && cast(sp).length < 2) return false;
  // a model that needs the recording's transcript is not sent one without it
  if (sp.clone_needs_text && f.refAudio && !state.voices.some((v) => v.name === f.voice) && !f.refText.trim()) return false;
  return true;
}

function body(): AudioSpeechBody {
  const f = state.form;
  const d = state.info?.defaults;
  const sp = state.info?.speech;
  const b: AudioSpeechBody = sp
    ? { model: state.info?.model, input: f.text.trim(), seconds: f.seconds, response_format: "url" }
    : { model: state.info?.model, instructions: f.prompt.trim(), input: f.lyrics.trim(), seconds: f.seconds, response_format: "url" };
  if (sp) {
    // only what the model takes: a voice or language left in the form from another model is not sent
    if (builtin(f.voice)) b.voice = f.voice;
    if (f.language && sp.languages.includes(f.language)) b.language = f.language;
    if ((sp.instructions || sp.design) && f.instructions.trim()) b.instructions = f.instructions.trim();
    const script = cast(sp);
    if (script.length) b.speakers = script;
    else if (sp.clone && state.voices.some((v) => v.name === f.voice)) b.voice = f.voice;
    else if (sp.clone && f.refAudio) {
      b.ref_audio = f.refAudio;
      if (f.refText.trim() && (sp.clone_needs_text || sp.clone_takes_text)) b.ref_text = f.refText.trim();
    }
  }
  if (f.seed !== null) b.seed = f.seed;
  // a song's steps and guidance: speech has neither, and a distilled model runs unguided
  if (!sp && f.steps !== null && f.steps !== d?.steps) b.steps = f.steps;
  if (!sp && state.info?.guidance !== false && f.cfg !== null && f.cfg !== d?.cfg) b.cfg = f.cfg;
  const opts = Object.fromEntries(Object.entries(f.options).filter(([k, v]) =>
    v.trim() !== "" && (state.info?.options ?? []).some((o) => o.name === k)));
  if (Object.keys(opts).length) b.options = opts;
  return b;
}

export async function generate(): Promise<void> {
  if (state.busy || !state.info || !canGo()) return;
  saveForm();
  state.busy = true;
  state.error = null;
  state.status = "starting";
  render();
  try {
    const j = await api.generate(body());
    state.result = j.data[0] ? song(j.data[0]) : null;
    const n = j.nextsycl;
    const wh = n.wh != null ? ` · ${n.wh.toFixed(2)} Wh` : "";
    state.status = `done: ${n.seconds.toFixed(1)} s of sound in ${n.took.toFixed(1)} s (${(n.seconds / n.took).toFixed(2)}× real time)${wh}`;
    await refreshHistory();
  } catch (e) {
    const m = e instanceof Error ? e.message : String(e);
    state.error = m === "cancelled" ? null : m;
    state.status = m === "cancelled" ? "cancelled" : "failed";
  }
  state.busy = false;
  render();
}

export async function cancel(): Promise<void> {
  state.status = "cancelling";
  render();
  await api.cancel();
}
