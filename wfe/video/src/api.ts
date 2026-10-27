/** The only module that talks to the server: the routes' typed client (src/gen/routes.ts), and rpc() for the RPC
 *  layer, typed by Methods. */
import { routes } from "./gen/routes.js";
import type { Envelope, Methods, Scene } from "./types.js";

// relative: the page is served at the studio's root or under the front door's /open/<id>/
const base = ".";
const r = routes();

export async function rpc<M extends keyof Methods>(
  method: M,
  params: Methods[M][0],
): Promise<Methods[M][1]> {
  const res = await fetch(`${base}/rpc/${String(method)}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(params ?? {}),
  });
  const e = (await res.json()) as Envelope<Methods[M][1]>;
  if (!e.ok) throw new Error(`${e.error.code}: ${e.error.message}`);
  return e.result;
}

export const api = {
  status: r.status,
  gpu: r.gpu,
  summary: r.summary,
  list: r.list,
  films: r.films,
  joins: r.joins,
  timeline: (project: string) => r.timeline({ project }),
  job: (name: string) => r.job({ name }),
  clip: (label: string) => r.clip({ label }),
  plan: r.plan,
  /** A file to the studio, kept by the name it answers with */
  upload: (f: File) => r.upload(f, { name: f.name }),

  hold: (scope: { project?: string; batch?: string; on: boolean }) => r.queuePause(scope),
  clearQueue: (scope: { project?: string; batch?: string } = {}) =>
    rpc("queue.clear", scope),
  join: (project: string) => r.join({ project }),
  importScene: (scene: Scene, project: string, paused: boolean) =>
    r.sceneImport({ scene: scene as unknown as Record<string, unknown>, project, paused }),
};
