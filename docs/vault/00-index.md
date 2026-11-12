---
type: moc
owner: shared
---

# AnimalUniversity Agent Vault

Shared, version-controlled knowledge base for Claude Code and Codex. Both agents
read this note at the start of every work session.

## Ownership

| Agent | Domain | Instruction file |
| --- | --- | --- |
| Claude Code | Frontend: `apps/ui_streamlit/`, `.streamlit/` | [[CLAUDE]] |
| Codex | Backend: `services/`, `animaluniversity/`, `scripts/`, `main.py`, `infra/`, `models/` | [[AGENTS]] |

Full rationale: [[adr-001-agent-ownership-split]].

## Start here

- [[system-overview]] - what the system does and how the parts fit
- [[work-queue]] - the one list of backend tasks; take work from here
- [[codex-operating-guide]] - the routine every backend session follows

- [[agent-collaboration-protocol]] - branches, worktrees, handoffs, and reviews
- [[fe-be-contract]] - the sanctioned interface between frontend and backend
- [[milestone-recording-ingest]] - current backend priority and constraints

## Handoff inboxes

Each agent writes one inbox and reads the other:

- [[frontend-to-backend]] - Claude Code writes; Codex reads and resolves
- [[backend-to-frontend]] - Codex writes; Claude Code reads and resolves

## Decisions

- [[adr-001-agent-ownership-split]]
- [[adr-002-offline-milestone-ingest-first]]
- [[adr-003-unified-pipeline-and-open-vocab-bootstrap]] (proposed)
- [[adr-004-offline-models-and-recording-splits]] (proposed)
- Template: [[_adr-template]]

## Knowledge

- [[system-overview]]
- [[vision-models]] - which models, why, and how each claim was checked
- [[pipeline-review-2026-11-12]] - fixes, synthetic benchmarks and remaining risks
- [[hardware-test-runbook]] - Mac M5 and NVIDIA test commands
- [[verification-record]] - what has actually been run, and what has not

## Session logs

Newest first. Every working session adds one under `docs/vault/log/`.

- [[2026-11-12-codex-efficiency-review]]

- [[2026-11-12-claude-review-and-unified-pipeline]]

## Existing design notes

- [[architecture]]
- [[milestone_xprotect_plan]]

## Vault rules

1. Never implement changes in the other agent's owned code.
2. Cross-boundary work goes through the appropriate handoff inbox.
3. Facts live in one canonical note; other notes link to it.
4. Decisions that outlive a task become ADRs.
5. Keep secrets, sensitive media, and private scratch notes out of Git.
6. The vault is the repository: knowledge ships in the same PR as the code.
7. Say what was verified and what was not. Never present unexecuted code as working.
