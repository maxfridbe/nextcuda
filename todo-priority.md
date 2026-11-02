# todo: high priority

- [x] **A name on every job (2026-11-02, the user's request).** Take `title` (free text, 120 characters, never matched
      on) on every job-making request and carry it on the job views (`*.status`, `*.jobs`, `system.busy`, the cards'
      processes); the front end lists it in place of the label wherever jobs and queues are shown, the label small
      beside it. A request without one shows the label as today. The platform sends a title for every unit and shows
      it in its GPUs page; nextsycl gets the same (`docs/NEXTSYCL_API_CHANGES.md` in vibe-creative-platform, "High
      priority"). Studio (video) jobs too.
      Done (2026-11-02): `title` on image/audio/model3d jobs (`JobOpts`) and on `video.generate` (shot and upscale
      engines; `video.engines.list` lists it as taken); kept on the job, not in `body`, trimmed and cut to 120
      characters, blank means none, never matched on (label / reentrant_id find jobs as before). Returned on
      `*.status`, `*.jobs`, `system.busy` (`jobs`, and the `video` block next to `label`), `video.status` (running job and
      queue items), the video timeline, finished-clip list rows and a reentered queued clip. A card's process in
      `system.cards` / `system.state` and `system.busy`'s per-card `work` carry `job_title` and `job_label` when the job is
      known (image/audio/3D by the server's port, video by the card the studio's clip runs on). Front ends: the video
      studio's queue, running clip, progress, finished list (title filterable) and detail, and the home page's card
      processes show the title with the label small beside it. Scenarios added in wfe/test.
      Left: the image, audio, model3d and ocr front ends do not list jobs by label, so were not changed; the video
      timeline strip still shows labels (the field is in its data). Not deployed anywhere.
