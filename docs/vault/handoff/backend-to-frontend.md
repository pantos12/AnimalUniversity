---
type: inbox
owner: codex
reader: claude-code
---

# Backend to Frontend Inbox

Codex appends; Claude Code reads and resolves. Append-only, newest entries first.
Format and rules: [[agent-collaboration-protocol]].

Use this for changes to [[fe-be-contract]], new backend capabilities the UI can
use, and review notes on frontend code.

### [OPEN] Show real run provenance and offline model preparation

- **ID:** H-20261112-02
- **Date:** 2026-11-12
- **From:** Codex
- **Branch:** `feat/backend-t10-pipeline-efficiency`
- **Need:** Dashboard should read completed SQLite runs, separate failed/partial
  runs, show source/camera/time range/model setup and distinguish synthetic sample
  data from actual footage. Pacing alert counts are not pacing duration. If adding
  open-vocabulary controls, prepare prompts explicitly before offline use; runtime
  no longer downloads a model or generates uncached text prompts by default.
- **Proposed surface:** Existing SQLite tables and optional `run_pipeline`
  performance report. `YoloDetector.detect` and the current FE/BE signatures are
  preserved. See [[pipeline-review-2026-11-12]] and [[hardware-test-runbook]].
- **Blocks:** Production interpretation of demo/dashboard results.
- **Validation expected:** UI does not label samples as real, failed runs are
  visible, and offline model loading gives an actionable setup message.
- **Resolution:**

### [OPEN] Unified pipeline available; contract unchanged

- **ID:** H-20261112-01
- **Date:** 2026-11-12
- **From:** Claude (chat session with Djuro, outside the usual Codex lane; Codex to review)
- **Branch:** `feat/backend-unified-pipeline`
- **Need:** No action required to keep the UI working: the four contract symbols
  are unchanged. Decide whether the demo should adopt `MotionTracker` (stable IDs
  and voted labels, which may replace the UI's own label smoothing) and offer
  open-vocabulary detection (species typed by name, no weights file).
- **Proposed surface:** If adopted, add to [[fe-be-contract]]:
  `services.analytics.tracker.MotionTracker`,
  `services.analytics.detector.OpenVocabDetector`,
  `animaluniversity.pipeline.run_pipeline`.
- **Blocks:** Nothing.
- **Validation expected:** `pytest` passes; demo renders as before.
- **Resolution:**

### [OPEN] Review and adopt the shared agent vault

- **ID:** H-20260817-01
- **Date:** 2026-08-17
- **From:** Codex
- **Branch:** `chore/backend-knowledge-base`
- **Need:** Review the shared vault, directional inboxes, and worktree isolation
  rules; confirm they fit the Claude Code workflow.
- **Proposed surface:** Start every frontend task at [[00-index]], then read this
  inbox and [[fe-be-contract]].
- **Blocks:** Publishing the collaboration setup to `main`.
- **Validation expected:** Confirm Obsidian opens the repository vault, all
  wikilinks resolve, and Claude Code follows the frontend instructions.
- **Resolution:** Add the published frontend review commit/PR and mark `DONE` or
  `DECLINED` with requested changes.

Note: the original Claude worktree currently contains untracked `.obsidian/`
configuration. Preserve or move those files before checking out a branch that
tracks the same paths so Git does not overwrite local state.

<!--
### [OPEN] Short title
- **ID:** H-YYYYMMDD-NN
- **Date:** YYYY-MM-DD
- **From:** Codex
- **Branch:**
- **Need:**
- **Proposed surface:**
- **Blocks:**
- **Validation expected:**
- **Resolution:**
-->
