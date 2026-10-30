/** What the front door answers: its routes' types (src/gen/routes.ts, made from glue/serve/src/home/routes.rs), and
 *  the page's own views of them. */

export type { Available, Config, Maintenance, ModelState, Proc, Service, SettingRow, State } from "./gen/routes.js";
import type { CardState, ModelView, Spec as SpecAnswer } from "./gen/routes.js";

export type Card = CardState;
export type Model = ModelView;

/** An operation of the OpenAPI document, as the API tab reads it */
export interface Operation {
  tags: string[];
  operationId?: string;
  summary: string;
  requestBody?: { content: { "application/json": { example?: unknown; schema?: { properties?: Record<string, { type: string; description?: string; enum?: string[] }>; required?: string[] } } } };
  parameters?: unknown[];
  "x-curl"?: string;
}

/** The API tab's answer, its document read as the tab reads it */
export type Spec = Omit<SpecAnswer, "spec"> & {
  spec: { info: { description: string; version: string }; servers: { url: string }[]; paths: Record<string, Record<string, Operation>> };
};

/** What saving the settings answers (an action's answer: its own fields) */
export interface ConfigSaved { changed: string[]; restart: string[]; note?: string | null }
