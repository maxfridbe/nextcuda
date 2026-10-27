/** The only module that talks to the server: its routes' typed client (src/gen/routes.ts). */
import { routes } from "./gen/routes.js";
import type { AudioSpeechBody, SoundItem, VoiceAdd } from "./gen/routes.js";
import type { GenAnswer, SavedVoice, Song } from "./types.js";

/** A server path made relative to the page, so the page works on its server's own port and through the front door
 *  (`/open/<id>/`) alike; a link the server made absolute (from the Host it was asked on) keeps only its path. */
export function here(u: string): string {
  if (/^(data|blob):/.test(u)) return u;
  const p = /^https?:\/\//.test(u) ? new URL(u).pathname : u;
  return p.startsWith("/") ? p.slice(1) : p;
}

const r = routes();

/** A sound the server made, as the page keeps it */
export const song = (s: SoundItem): Song => ({
  url: here(s.url), prompt: s.prompt, lyrics: s.lyrics, seed: s.seed ?? 0, steps: s.steps ?? undefined, cfg: s.cfg ?? undefined, seconds: s.seconds,
  took: s.took ?? undefined, wh: s.wh, model: s.model ?? undefined, created: s.created, voice: s.voice, language: s.language,
  instructions: s.instructions, cloned: s.cloned ?? undefined,
});

export const api = {
  info: r.info,
  progress: r.progress,
  gpu: r.gpu,
  history: async (): Promise<Song[]> => (await r.history()).data.map(song),
  cancel: async (): Promise<void> => { await r.cancel(); },
  voices: async (): Promise<SavedVoice[]> => (await r.voices()).data,

  /** Save a voice: a recording (sample, a data: URL) or a file made here; an error carries the server's message. */
  addVoice: (body: VoiceAdd): Promise<SavedVoice> => r.voiceAdd(body),
  async removeVoice(name: string): Promise<void> {
    await r.voiceRemove({ name });
  },

  /** The speech endpoint (the reference server's), its answer as JSON with the song's link; an error carries the
   *  server's message. */
  async generate(body: AudioSpeechBody): Promise<GenAnswer> {
    const j = await r.speech({ ...body, response_format: "url" });
    if (j instanceof Blob) throw new Error("the server answered the WAV itself, not its link");
    for (const d of j.data ?? []) if (d.url) d.url = here(d.url);
    return j;
  },
};
