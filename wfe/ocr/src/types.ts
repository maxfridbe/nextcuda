/** What the server sends: its routes' types (src/gen/routes.ts, made from glue/serve/src/image/routes.rs), and what the page keeps. */

export type { Gpu, Info, Progress } from "./gen/routes.js";

/** A reading, as the page keeps it: from /v1/ocr's answer or a history entry (api.ts) */
export interface Reading {
  text: string;
  prompt?: string;
  pages: number;
  tokens: number;
  prompt_tokens?: number;
  seconds: number;
  tok_s?: number;
  read_tok_s?: number;
  wh?: number | null;
  created?: number;
}

/** A page given: its name and data URL (kept for the boxes view) */
export interface Page {
  name: string;
  url: string;
  width: number;
  height: number;
}

export interface Form {
  preset: string;
  prompt: string;
  tiles: boolean;
  maxTokens: number;
  temperature: number;
  guard: boolean;
  /** read the text inside the pictures the model marks */
  figures: boolean;
  /** several pages: each read on its own (tiles, finer print) rather than as one document (the model's multi-page
   *  reading, each page seen at 1024) */
  perPage: boolean;
  /** find blocks set side by side and read each on its own */
  columns: boolean;
}
