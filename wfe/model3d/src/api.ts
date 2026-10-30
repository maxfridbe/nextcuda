/** The only module that talks to the server: its routes' typed client (src/gen/routes.ts). */
import { routes } from "./gen/routes.js";
import type { Model3dGenerateBody, Model3dRigBody } from "./gen/routes.js";
import type { Answer, Made } from "./types.js";

/** A server path made relative to the page, so the page works on its server's own port and through the front door
 *  (`/open/<id>/`) alike; a link the server made absolute (from the Host it was asked on) keeps only its path. */
export function here(u: string): string {
  if (/^(data|blob):/.test(u)) return u;
  const p = /^https?:\/\//.test(u) ? new URL(u).pathname : u;
  return p.startsWith("/") ? p.slice(1) : p;
}

const linked = (m: Made): Made => ({ ...m, url: here(m.url), picture: m.picture ? here(m.picture) : m.picture });

const r = routes();

/** The JSON answer (response_format url), its links made relative */
function answer(j: Answer | Blob): Answer {
  if (j instanceof Blob) throw new Error("the server answered the GLB itself, not its link");
  if (Array.isArray(j.data)) j.data = j.data.map(linked);
  return j;
}

export const api = {
  info: r.info,
  progress: r.progress,
  gpu: r.gpu,
  history: async (): Promise<Made[]> => (await r.history()).data.map(linked),
  cancel: async (): Promise<void> => { await r.cancel(); },
  /** a mesh from pictures; its answer links the GLB (an error carries the server's message) */
  generate: async (body: Model3dGenerateBody): Promise<Answer> => answer(await r.generate({ ...body, response_format: "url" })),
  /** a mesh rigged or animated */
  rig: async (body: Model3dRigBody): Promise<Answer> => answer(await r.rig({ ...body, response_format: "url" })),
};
