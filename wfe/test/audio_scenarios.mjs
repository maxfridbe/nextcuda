// The audio page for each model family: songs (ACE-Step base / turbo, MiniMax Music 3) and speech (Qwen3-TTS custom voice /
// voice design / base, VoxCPM2, VibeVoice's cast, Breeze TTS 2), from the info answer each engine gives, and what a
// click would send. Run after ./wfe/build.sh: node wfe/test/audio_scenarios.mjs
import { Scenarios, mod, sent, stubFetch } from "./lib.mjs";

const sc = new Scenarios("audio");
const { App } = await mod("audio", "components/App.js");
const { state } = await mod("audio", "state.js");
const { canGo, generate } = await mod("audio", "actions.js");

const song = (o) => ({ model: "m", wfe: true, lyrics: true, guidance: true, saves: true, options: [], report: [], speech: null,
                       defaults: { seconds: 30, max_seconds: 360, steps: 50, cfg: 7, rate: 48000 }, ...o });
const sp = (o) => ({ voices: [], languages: [], instructions: false, design: false, clone: false, clone_needs_text: false, clone_takes_text: false,
                     direction: false, events: [], requires: null, speakers: 0, ...o });
const speech = (arch, s, o = {}) => ({ model: arch, arch, wfe: true, lyrics: false, saves: true, options: [], report: [],
                                       defaults: { seconds: 60, max_seconds: 300, steps: 0, cfg: 1, rate: 24000 }, speech: sp(s), ...o });

const ace = song({ model: "acestep-1.5-base", arch: "acestep" });
const aceTurbo = song({ model: "acestep-1.5-turbo", arch: "acestep", guidance: false, defaults: { seconds: 30, max_seconds: 360, steps: 8, cfg: 1, rate: 48000 } });
const mm3 = song({ model: "minimax-music3-int8", arch: "minimax-music3", defaults: { seconds: 60, max_seconds: 360, steps: 30, cfg: 1.7, rate: 48000 } });
const custom = speech("qwen3-tts", { voices: ["vivian", "ryan"], languages: ["english", "chinese"], instructions: true, requires: "voice" });
const design = speech("qwen3-tts", { languages: ["english", "chinese"], instructions: true, design: true, requires: "instructions" });
const base = speech("qwen3-tts", { languages: ["english", "chinese"], clone: true, clone_takes_text: true, requires: "reference" });
const vox = speech("voxcpm2", { instructions: true, design: true, clone: true, clone_takes_text: true, direction: true });
const vibe = speech("vibevoice", { languages: ["en", "zh"], clone: true, requires: "reference", speakers: 4 });
const breeze = speech("breeze-tts-2", { languages: ["chinese", "english"], instructions: true, design: true, clone: true, clone_needs_text: true,
                                        clone_takes_text: true, direction: true, events: ["(laugh)", "(sigh)"] });

const saved = [{ name: "anna", kind: "clone", text: "hello" }, { name: "boris", kind: "clone", text: "hi" }, { name: "cleo", kind: "design" }];
function open(info, form = {}, voices = saved) {
  state.info = JSON.parse(JSON.stringify(info));
  // a form kept from another model: a voice and a language this one does not have, steps and cfg of a song, a transcript
  state.form = { prompt: "folk", lyrics: "[verse]\nla", seconds: 60, steps: 20, cfg: 5, seed: null, options: {}, text: "Good morning.", voice: "zed",
                 language: "klingon", instructions: "", refAudio: "", refName: "", refText: "stale transcript", ...form };
  state.voices = voices;
  state.result = null;
  state.history = [];
  state.busy = false;
  state.error = null;
  state.status = "idle";
  state.panels = { advanced: true, create: true };
  stubFetch((url) => url.includes("speech") ? { created: 1, data: [{ url: "/v1/audio/files/a.wav", prompt: "t", lyrics: "", seconds: 1 }], nextsycl: { seconds: 1, took: 1, wh: null } } : {});
}
const body = () => sent.find((s) => /speech/.test(s.url))?.body;
const go = async () => { await generate(); return body() ?? {}; };

// songs
open(ace);
sc.show("acestep-base", App()).has(/lyrics/).has(/steps/).has(/guidance \(cfg\)/).lacks(/windows of 8 s/, "MiniMax's flow windows").lacks(/each window/).has(/Advanced \(20 steps · guidance 5\)/);
{ const b = await go(); sc.check("acestep-base", b.steps === 20 && b.cfg === 5, "a base checkpoint takes steps and cfg"); }
open(aceTurbo);
sc.show("acestep-turbo", App()).has(/Advanced \(20 steps · unguided\)/).has(/off \(this checkpoint is distilled\)/).lacks(/each window/);
{ const b = await go(); sc.check("acestep-turbo", b.steps === 20 && !("cfg" in b), `turbo sends steps but no cfg (${JSON.stringify(b)})`); }
open(mm3);
sc.show("minimax-music3", App()).has(/steps \(each window\)/).has(/windows of 8 s, 4 s apart/).has(/guidance \(cfg\)/);

// speech
open(custom, { voice: "zed" });
sc.show("qwen3-custom-voice-stale-voice", App()).has(/\[button Vivian\]/).has(/how to say it/).lacks(/voice to clone/).lacks(/saved voice/).lacks(/Advanced.*talker/)
  .has(/language/).has(/\[button Speak disabled\]/, "Speak is held until one of its voices is picked");
sc.check("qwen3-custom-voice-stale-voice", canGo() === false, "a voice from another model does not satisfy requires=voice");
open(custom, { voice: "ryan", language: "english" });
{
  const b = await go();
  sc.check("qwen3-custom-voice", b.voice === "ryan" && b.language === "english" && !("steps" in b) && !("cfg" in b) && !("ref_text" in b),
    `custom voice: voice and language, no song fields (${JSON.stringify(b)})`);
  const v = sc.show("qwen3-custom-voice", App());
  v.has(/The talker draws a frame 12.5 times a second/, "Qwen3-TTS's own Advanced note");
}
open(design, { instructions: "A calm old woman" });
sc.show("qwen3-voice-design", App()).has(/the voice/).has(/who speaks: age, gender/).lacks(/\[button Vivian/).lacks(/voice to clone/);
{ const b = await go(); sc.check("qwen3-voice-design", b.instructions === "A calm old woman" && !("voice" in b) && !("language" in b), `design sends instructions only (${JSON.stringify(b)})`); }
open(base, {}, saved);
sc.show("qwen3-base-clone", App()).has(/a saved voice/).has(/the voice to clone/).has(/or a recording below/).lacks(/how to say it/).lacks(/several speakers/)
  .has(/\[button Speak disabled\]/);
open(base, { voice: "anna" });
{ const b = await go(); sc.check("qwen3-base-clone", b.voice === "anna" && !("language" in b), `a saved voice is sent as voice (${JSON.stringify(b)})`); }
open(vox, { voice: "anna", instructions: "warm" });
{
  const v = sc.show("voxcpm2", App());
  v.has(/the voice, or the style/).has(/or with a cloned voice its style/, "VoxCPM2 styles a cloned voice").lacks(/talker/).has(/This model has no options of its own/);
  const b = await go();
  sc.check("voxcpm2", b.voice === "anna" && b.instructions === "warm" && !("language" in b), `VoxCPM2: saved voice and its instructions (${JSON.stringify(b)})`);
}
open(vibe, {}, saved);
sc.show("vibevoice-no-cast", App()).has(/a script of several speakers/).has(/speaker 1/).has(/speaker 4/).lacks(/what the recording says/).lacks(/how to say it/).has(/\[button Speak disabled\]/);
open(vibe, { voices: ["anna", "boris"], text: "Speaker 1: hi\nSpeaker 2: hello" });
{
  const v = sc.show("vibevoice-cast", App());
  v.has(/2 speakers cast/).lacks(/\[button Speak disabled\]/);
  const b = await go();
  sc.check("vibevoice-cast", JSON.stringify(b.speakers) === '["anna","boris"]' && !("ref_text" in b) && !("voice" in b), `the cast goes as speakers (${JSON.stringify(b)})`);
}
open(vibe, {}, []);
sc.show("vibevoice-no-saved-voices", App()).has(/save two or more voices first/);
open(breeze, { voice: "anna", instructions: "calm" });
{
  const v = sc.show("breeze", App());
  v.has(/vocal events inline: \(laugh\) \(sigh\)/).has(/or with a cloned voice its style/).lacks(/talker/).lacks(/what the recording says/, "a transcript, the saved voice has its own");
  const b = await go();
  sc.check("breeze", b.voice === "anna" && b.instructions === "calm", `Breeze: a saved voice directed by instructions (${JSON.stringify(b)})`);
}
open(breeze, { voice: "", refAudio: "data:audio/wav;base64,AAAA", refName: "me.wav", refText: "" });
sc.show("breeze-recording", App()).has(/what the recording says/).has(/required/).has(/save as voice/);

sc.done();
