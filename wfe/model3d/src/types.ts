/** What the 3D server answers: its routes' types (src/gen/routes.ts, made from glue/serve/src/model3d/routes.rs), and what the page keeps. */

export type { Abilities, Gpu, Info, Progress } from "./gen/routes.js";
import type { Mesh, Meshes } from "./gen/routes.js";

/** A mesh made here (its settings from the sidecar JSON) */
export type Made = Mesh;

/** What a generation or a rig answers with response_format url */
export type Answer = Meshes;

/** A picture chosen for the request */
export interface Pic { name: string; data: string; side: string }

export interface Form {
  pictures: Pic[];
  texture: boolean;
  seed: number | null;
  steps: number | null;
  cfg: number | null;
  resolution: number | null;
  faces: number | null;
  textureSize: number | null;
  removeBackground: boolean;
  /** rig: a mesh file sent (a data: URL), or one made here */
  mesh: string;
  meshName: string;
  file: string;
  prompt: string;
  frames: number | null;
  options: Record<string, string>;
}
