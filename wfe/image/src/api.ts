/** The only module that talks to the server: its routes' typed client (src/gen/routes.ts). */
import { routes } from "./gen/routes.js";
import type { ImageEditBody, ImageGenerateBody } from "./gen/routes.js";
import type { GenAnswer, Picture } from "./types.js";

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
  /** the pictures made (the other work an image server does - upscales, SVGs - has pages of its own) */
  history: async (): Promise<Picture[]> =>
    (await r.history()).data.filter((p) => !p.kind && p.url).map((p) => ({
      url: here(p.url ?? ""), prompt: p.prompt, seed: p.seed ?? 0, steps: p.steps ?? 0, width: p.width ?? 0, height: p.height ?? 0,
      loras: p.loras ?? [], seconds: p.seconds, wh: p.wh, pictures: p.pictures ?? undefined, created: p.created,
    })),

  /** OpenAI's images endpoint (an edit: the edits one, its pictures as data URLs), with our fields; an error carries
   *  the server's message. */
  async generate(body: ImageGenerateBody & Partial<ImageEditBody>, edit = false): Promise<GenAnswer> {
    const j = edit ? await r.edits({ ...body, images: body.images ?? [] }) : await r.generations(body);
    for (const d of j.data ?? []) if (d.url) d.url = here(d.url);
    return j;
  },
};
