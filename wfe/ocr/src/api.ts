/** The only module that talks to the server: its routes' typed client (src/gen/routes.ts). */
import { routes } from "./gen/routes.js";
import type { ImageOcrReq } from "./gen/routes.js";
import type { Reading } from "./types.js";

/** A server path made relative to the page, so the page works on its server's own port and through the front door
 *  (`/open/<id>/`) alike; a link the server made absolute (from the Host it was asked on) keeps only its path. */
export function here(u: string): string {
  if (/^(data|blob):/.test(u)) return u;
  const p = /^https?:\/\//.test(u) ? new URL(u).pathname : u;
  return p.startsWith("/") ? p.slice(1) : p;
}

const r = routes();

export const api = {
  info: r.info,
  progress: r.progress,
  gpu: r.gpu,
  history: async (): Promise<Reading[]> =>
    (await r.history()).data.filter((x) => x.kind === "ocr").map((x) => ({
      text: x.text ?? "", prompt: x.prompt, pages: x.pages ?? 0, tokens: x.tokens ?? 0, prompt_tokens: x.prompt_tokens ?? undefined,
      seconds: x.seconds, tok_s: x.tok_s ?? undefined, wh: x.wh, created: x.created,
    })),

  /** A reading: the pages as data URLs; an error carries the server's message. */
  ocr: (body: ImageOcrReq): Promise<Reading> => r.ocr(body),
};
