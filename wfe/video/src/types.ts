/** What the studio answers: its routes' types (src/gen/routes.ts, made from glue/serve/src/video/studio/routes.rs);
 *  and the RPC layer's (POST /rpc/<method>, its envelope and requests), which is not generated yet. */

export type {
  Batch, Canvas, ClipState, Disk, EngineInfo, Film, Gpu, GpuPlan, Join, ListRow, LlmStatus, Plan, Project, ProjectStats,
  QueueItem, Scene, Status, Summary, TimelineClip,
} from "./gen/routes.js";
import type { Clip, ClipRecord, Film, LlmStatus, Project, Scene, Status, Summary, Gpu, TimelineClip } from "./gen/routes.js";

/** A clip's record: a queue item, the running job, /api/clip's answer */
export type Job = Clip;

/** A finished clip's record (/api/job/<name>) */
export type JobDetail = ClipRecord;

export type Envelope<T> =
  | { ok: true; result: T }
  | { ok: false; error: { code: ErrorCode; message: string } };

export type ErrorCode =
  | "invalid_request" | "invalid_param" | "not_found"
  | "busy" | "not_running" | "unknown_method" | "internal";

/** "prev" = previous clip of this series, "first" = the series hub, "*_cam" = same camera only. */
export type Anchor = "prev" | "prev_cam" | "first" | "first_cam" | (string & {});
export type ChainMode = "png" | "video" | "latent" | "none";
/** a denoiser the engine has: INT8 (the default), or a GGUF k-quant it was given (Q6_K, Q4_K_M) */
export type Engine = string;

/** What the page sends to /rpc/generate (the fields it sets; the studio takes every key of ClipRequest). */
export interface GenerateRequest {
  prompt: string;
  seconds?: number;
  steps?: number;
  seed?: number;
  te?: "teacher" | "student";
  label?: string;
  width?: number;
  height?: number;
  engine?: Engine;
  chain_mode?: ChainMode;
  first_frame?: Anchor | null;
  last_frame?: Anchor | null;
  exposure_ref?: Anchor | null;
  first_audio?: Anchor | null;
  first_audio_s?: number;
  cond_noise_aug?: number | null;
  camera?: string | null;
  upscale?: number;
  upscaler?: string;
  /** ComfyUI's sampler and schedule names (none: Euler on the shifted schedule) */
  sampler?: string | null;
  schedule?: string | null;
  loras?: string[];
  /** references (Ref2VA): files in the studio's output directory, named in the prompt as <Picture i>, <Video k>,
   *  <Audio j> (a clip's own sound takes the next <Audio j> before its <Video k>) */
  ref_images?: string[] | null;
  ref_videos?: string[] | null;
  ref_audios?: string[] | null;
  ref_video_sound?: boolean | null;
  ref_image_size?: "match" | "max" | null;
  /** the keyframe pictures shown to the text encoder too (default on) */
  te_pictures?: boolean | null;
  /** the Fun ControlNet: a control video (canny / depth / HED / MLSD / pose frames), a mask (white: regenerate) over
   *  a source video; files in the studio's output directory */
  control_video?: string | null;
  control_mask?: string | null;
  control_source?: string | null;
  control_strength?: number | null;
  project?: string;
  batch?: string;
  queue?: boolean;
}

export interface Scope { project?: string; batch?: string }

export interface HoldResult {
  queued: number;
  held: number;
  paused_all: boolean;
  paused_projects: string[];
  paused_batches: string[];
}

/** RPC method -> [request, response]. Keeps rpc() honest at the call site. */
export interface Methods {
  status: [Record<string, never>, Status];
  wait: [{ version?: string; timeout_s?: number }, Status];
  generate: [GenerateRequest, { queued: boolean; position?: number; job?: Job }];
  cancel: [{ label?: string }, { stopped: boolean; removed: number; label?: string }];
  "queue.remove": [{ index?: number; label?: string }, { removed: number; index?: number; label: string }];
  "queue.list": [Record<string, never>, { queue: Job[] }];
  "queue.clear": [Scope, { cleared: number }];
  "queue.pause": [Scope, HoldResult];
  "queue.resume": [Scope, HoldResult];
  "projects.list": [Record<string, never>, { projects: Project[]; paused_all: boolean }];
  timeline: [{ project: string }, { clips: TimelineClip[] }];
  "scene.export": [{ project: string }, Scene];
  "scene.import": [
    { scene: Scene; project?: string; paused?: boolean },
    { imported: number; project: string; queued: number; held: number },
  ];
  "films.list": [Record<string, never>, { films: Film[] }];
  summary: [Record<string, never>, Summary];
  gpu: [Record<string, never>, Gpu];
  "llm.mode": [{ mode?: string }, LlmStatus];
}
