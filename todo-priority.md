# todo: high priority

- [ ] **A name on every job (2026-11-02, the user's request).** Take `title` (free text, 120 characters, never matched
      on) on every job-making request and carry it on the job views (`*.status`, `*.jobs`, `system.busy`, the cards'
      processes); the front end lists it in place of the label wherever jobs and queues are shown, the label small
      beside it. A request without one shows the label as today. The platform sends a title for every unit and shows
      it in its GPUs page; nextsycl gets the same (`docs/NEXTSYCL_API_CHANGES.md` in vibe-creative-platform, "High
      priority"). Studio (video) jobs too.
