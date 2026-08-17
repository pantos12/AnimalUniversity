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
