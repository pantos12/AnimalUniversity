---
type: convention
owner: shared
updated: 2026-11-12
---

# Codex Operating Guide

How backend agents work in this repository so that several can run at once
without colliding or leaving the knowledge base behind. Applies to every Codex
session. Ownership rules stay as in [[AGENTS]] and
[[agent-collaboration-protocol]]; this note adds the working routine.

## One knowledge base, two views

The vault *is* the repository. Obsidian opens the repo folder; GitHub hosts the
same files. There is nothing to sync by hand: a note is published when its
commit is pushed and merged. So:

- Knowledge goes in `docs/vault/` in the same PR as the code it describes.
- Never keep project knowledge only in a chat, a PR comment or a local note.
- Never edit notes on `main` directly. Notes travel with branches like code.

## Start of every session

1. `git fetch && git status`. Work in your own worktree, never a shared one.
2. Read [[00-index]], then [[work-queue]], then open entries in
   [[frontend-to-backend]].
3. Take the highest-priority `TODO` task that is not blocked. One task per
   session and per branch.
4. Claim it: change its status to `CLAIMED <agent> <branch> <date>`, commit only
   that line, push. If the push is rejected because someone else claimed it,
   pick another task.

## While working

- Branch: `feat/backend-<task-id>-<slug>` from current `origin/main`.
- Write the test first when fixing a bug. A bug fix without a test that failed
  before the fix is not done.
- Tests must run without a GPU, model weights, network or real footage. Use the
  helpers in `tests/conftest.py` (synthetic clips, `BrightBlobDetector`).
  Model-dependent checks belong in [[verification-record]], run on real
  hardware, not in the test suite.
- Keep the [[fe-be-contract]] symbols stable. A change to them is a breaking
  change: flag it in the PR title and post in [[backend-to-frontend]].
- Stay inside the task. Found something else? Add a task to [[work-queue]]
  instead of fixing it in the same PR.

## Say what was proven

Every PR description and every session log has two lists:

- **Verified:** what was run, and how (command, environment).
- **Not verified:** what was written but not executed, and why.

Do not describe unexecuted code as working. Do not write version numbers, API
names or behaviors of external tools from memory: check the installed package
or its documentation and say which you checked.

## Before opening the PR

- [ ] `pytest` passes locally.
- [ ] New behavior has tests, including near-miss negative cases.
- [ ] If detection, tracking or behavior changed: before/after table from
      `scripts/au_eval.py` (once T3 exists).
- [ ] Vault updated: the task's status in [[work-queue]]; the relevant note
      under `docs/vault/backend/` or `docs/vault/knowledge/`; an ADR under
      `docs/vault/decisions/` if a lasting decision was made.
- [ ] Session log added: `docs/vault/log/YYYY-MM-DD-<agent>-<slug>.md` with
      what changed, verified / not verified, and what the next agent should
      know. Copy the frontmatter from an existing log.
- [ ] No secrets, camera URLs, server addresses, footage, frames, weights or
      databases in the diff.
- [ ] Rebased on `origin/main`; frontend files untouched.

## After merge

Set the task to `DONE <PR> <date>` (in the same PR is fine). If the work
unblocks another task, change that task's `BLOCKED` line.

## Running several agents at once

- Tasks in the same priority group that touch different files can run in
  parallel: for example T3 (new script), T5 (eventbridge) and T8 (infra).
- Tasks that touch `services/analytics/` or `animaluniversity/pipeline/` should
  not run in parallel with each other. If they must, the second agent rebases
  after the first merges.
- Shared files (`README.md`, `requirements.txt`, `docs/vault/00-index.md`,
  `work-queue.md`) cause most conflicts. Make the smallest possible edit and
  commit it separately so a rebase is trivial.

## When blocked

Set the task to `BLOCKED <reason>`, write what is needed and from whom (for
footage, credentials or server access that is Djuro), then take the next task.
Do not invent data, credentials or results to get past a block.
