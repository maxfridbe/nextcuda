/** What the studio answers: its routes' types (src/gen/routes.ts, made from glue/serve/src/video/studio/routes.rs) and
 *  its RPC methods' (src/gen/studio.ts, from methods.rs); and what the page sends. */

export type {
  Batch, Canvas, ClipState, Disk, EngineInfo, Film, Gpu, GpuPlan, Join, ListRow, LlmStatus, Plan, Project, ProjectStats,
  QueueItem, Scene, Status, Summary, TimelineClip,
} from "./gen/routes.js";
import type { Clip, ClipRecord } from "./gen/routes.js";

/** A clip's record: a queue item, the running job, /api/clip's answer */
export type Job = Clip;

/** A finished clip's record (/api/job/<name>) */
export type JobDetail = ClipRecord;

/** "prev" = previous clip of this series, "first" = the series hub, "*_cam" = same camera only. */
export type Anchor = "prev" | "prev_cam" | "first" | "first_cam" | (string & {});
export type ChainMode = "png" | "video" | "latent" | "none";
/** a denoiser the engine has: INT8 (the default), or a GGUF k-quant it was given (Q6_K, Q4_K_M) */
export type Engine = string;

/** What the page sends to /rpc/generate: the studio's request (the front door's video.generate's) */
export type { VideoGenerateReq as GenerateRequest } from "./gen/studio.js";

export type { HoldResult, Scope } from "./gen/studio.js";
