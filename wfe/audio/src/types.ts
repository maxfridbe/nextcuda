/** What the audio server answers: its routes' types (src/gen/routes.ts, made from glue/serve/src/audio/routes.rs), and what the page keeps. */

export type { Defaults, Gpu, Info, Progress, SavedVoice, SpeechInfo as Speech } from "./gen/routes.js";
import type { Sound } from "./gen/routes.js";

/** A song or speech made, as the page keeps it (from a history entry or an answer: api.ts) */
export interface Song {
  url: string;
  prompt: string;
  lyrics: string;
  seed: number;
  steps?: number;
  cfg?: number;
  /** seconds of sound */
  seconds: number;
  /** seconds it took to make */
  took?: number;
  wh?: number | null;
  model?: string;
  created: number;
  /** speech: the voice, language, instructions, whether it was cloned */
  voice?: string | null;
  language?: string | null;
  instructions?: string | null;
  cloned?: boolean;
}

/** What /v1/audio/speech answers with response_format url */
export type GenAnswer = Sound;

/** The form: every option a request can carry. */
export interface Form {
  prompt: string;
  lyrics: string;
  seconds: number;
  steps: number | null;
  cfg: number | null;
  seed: number | null;
  /** the engine's own options, by name (empty: not sent) */
  options: Record<string, string>;
  /** speech: the text to say, the voice, the language ("" auto), how to say it / the voice to make up */
  text: string;
  voice: string;
  language: string;
  instructions: string;
  /** speech: the recording to clone (not kept across reloads) and its transcript */
  refAudio: string;
  refName: string;
  refText: string;
}

