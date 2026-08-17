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
- Template: [[_adr-template]]

## Existing design notes

- [[architecture]]
- [[milestone_xprotect_plan]]

## Vault rules

1. Never implement changes in the other agent's owned code.
2. Cross-boundary work goes through the appropriate handoff inbox.
3. Facts live in one canonical note; other notes link to it.
4. Decisions that outlive a task become ADRs.
5. Keep secrets, sensitive media, and private scratch notes out of Git.
