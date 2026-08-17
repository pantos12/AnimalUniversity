---
type: convention
owner: shared
---

# Agent Collaboration Protocol

How [[CLAUDE]] and [[AGENTS]] coordinate without conflicting edits or duplicated
logic.

## Session start

1. Read [[00-index]].
2. Read the inbox addressed to your role and resolve relevant `OPEN` entries.
3. Read [[fe-be-contract]] before boundary work.
4. Fetch current Git state and use a role-specific worktree and branch.

## Worktree and branch isolation

- Codex uses its own worktree and `feat/backend-*` or `fix/backend-*` branches.
- Claude Code uses its own worktree and `feat/frontend-*` or `fix/frontend-*`
  branches.
- Never run both agents in the same working tree.
- Never commit directly to `main`.
- Keep each PR limited to one logical change and one ownership domain.

## Crossing the boundary

Do not edit the other agent's implementation. Instead:

1. Append a request to the inbox read by the owning agent.
2. Mark it `OPEN`; include the need, proposed surface, and blocking branch/task.
3. Let the owner implement it, record validation, and update the entry to `DONE`.
4. Rebase or merge only after the owning change is published.

## Inbox entry format

```markdown
### [OPEN] Short title
- **ID:** H-YYYYMMDD-NN
- **Date:** YYYY-MM-DD
- **From:** Claude Code | Codex
- **Branch:** role/type-slug
- **Need:** What is needed and why the requesting layer cannot own it
- **Proposed surface:** Suggested interface, if known
- **Blocks:** Branch, PR, or task
- **Validation expected:** Observable acceptance criteria
- **Resolution:** Filled by the owning agent with commit/PR and result
```

Status values: `OPEN` -> `DONE`, or `DECLINED` with a reason.

## Improving the collaboration

- Reading and reviewing the other domain is allowed; editing it is not.
- Push back on requests that put logic in the wrong layer.
- Promote repeated corrections into this protocol or an ADR.
- Record surprising constraints in the relevant owned note.
- Use commit IDs, PR links, commands, and file references rather than vague status.

## Decisions

Anything that outlives a task becomes an ADR created from [[_adr-template]] under
`docs/vault/decisions/`.

## Shared files

`README.md`, `requirements.txt`, `docs/architecture.md`, `.gitignore`, and
`.env.example` belong to neither agent. Keep diffs minimal and disclose them in the
PR body so the other agent can rebase safely.
