---
type: adr
owner: shared
status: accepted
date: 2026-08-17
---

# ADR-001: Split Frontend and Backend Ownership

## Status

accepted

## Context

Claude Code and Codex work on the same repository. Without an explicit seam they
can create conflicting edits and duplicated logic. The application already has a
natural boundary between the Streamlit UI and Python pipeline services.

## Decision

- Claude Code owns `apps/ui_streamlit/` and `.streamlit/`.
- Codex owns `services/`, `animaluniversity/`, `scripts/`, `main.py`, `infra/`, and
  `models/`.
- Cross-boundary work uses the handoff inboxes and [[fe-be-contract]].
- Each agent uses its own worktree and role-prefixed branch.
- Shared files receive minimal, coordinated changes.

## Consequences

- Merge conflicts become visible contract/review discussions.
- Each agent may read but not directly fix the other domain.
- New frontend dependencies on backend behavior require an explicit contract
  update and handoff.
- `main.py` is backend despite being at the repository root.

## Alternatives considered

- **One shared working tree.** Rejected because concurrent agents can overwrite or
  invalidate each other's uncommitted work.
- **Split ownership by feature.** Rejected because most features cross UI and
  pipeline files, making the boundary unstable.
- **Keep the knowledge vault outside Git.** Rejected because unversioned notes are
  not reviewable and drift away from the code.
