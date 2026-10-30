/** What the image server answers: its routes' types (src/gen/routes.ts, made from glue/serve/src/image/routes.rs), and what the page keeps. */

export type { Defaults, Gpu, Info, LoraInfo, Progress } from "./gen/routes.js";
import type { Pictures } from "./gen/routes.js";

/** A picture made, as the page keeps it (from a history entry or an answer) */
export interface Picture {
  url: string;
  prompt: string;
  seed: number;
  steps: number;
  width: number;
  height: number;
  loras: string[];
  seconds: number;
  wh?: number | null;
  /** the pictures it was made from (an edit) */
  pictures?: number;
  created: number;
}

/** What a generation or an edit answers */
export type GenAnswer = Pictures;

/** The form: every option a request can carry. */
export interface Form {
  prompt: string;
  negative: string;
  width: number;
  height: number;
  steps: number | null;
  n: number;
  seed: number | null;
  sampler: string;
  schedule: string;
  cfg: number | null;
  shift: number | null;
  rgba: boolean;
  /** id -> scale, for the LoRAs ticked */
  loras: Record<string, number>;
  /** the engine's own options, by name (empty: not sent) */
  options: Record<string, string>;
}

/** A picture given to an edit: a data: URL and its size */
export interface Given {
  url: string;
  name: string;
  w: number;
  h: number;
}
