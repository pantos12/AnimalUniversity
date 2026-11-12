---
type: work-queue
owner: codex
status: active
updated: 2026-11-12
---

# Backend Work Queue

The single list of backend work. Agents take tasks from here and nowhere else.
How to work a task: [[codex-operating-guide]]. Background:
[[2026-11-12-claude-review-and-unified-pipeline]], [[system-overview]].

**Status values:** `TODO`, `CLAIMED <agent> <branch> <date>`, `IN REVIEW <PR>`,
`DONE <PR> <date>`, `BLOCKED <reason>`. Change the status line in its own small
commit *before* starting, so two agents never take the same task.

Tasks marked **needs Djuro** cannot be finished by an agent alone.

---

## Review requested by Djuro

### T10. Review unified pipeline reliability, efficiency and model choices
- **Status:** CLAIMED Codex feat/backend-t10-pipeline-efficiency 2026-11-12
- **Do:** Review PR #3, reproduce correctness/resource issues, improve offline
  model preparation and archived inference efficiency, and document model
  candidates using official sources and measured results where available.
- **Done when:** Regression tests pass, CPU benchmark evidence is recorded,
  model claims distinguish execution from recommendation, and a review PR lists
  what still needs representative animal footage. Keep frontend implementation
  untouched and preserve existing public call signatures.

---

## P0: prove it on real hardware and real footage

### T1. First run on a real machine
- **Status:** TODO
- **Why:** Model-dependent code has never been executed (see the session log).
- **Do:** Fresh virtual environment, `pip install -r requirements.txt`, `pytest`.
  Then `python scripts/au_run.py --video <any animal clip> --classes "<species>"
  --db data/au.db --report-json data/runs/t1.json`. Run `main.py`, the UI and
  `scripts/au_train.py` for two epochs on a tiny dataset to check the
  `ultralytics` upgrade broke nothing.
- **Done when:** All commands exit 0, or each failure is fixed with a test.
  Record versions (Python, torch, ultralytics, GPU), frames per second, and
  what was wrong in [[verification-record]]. Move confirmed items from "not
  verified" to "verified" there.

### T2. Real footage intake (needs Djuro: footage)
- **Status:** BLOCKED waiting for first XProtect exports
- **Why:** The manifest contract assumes one MP4 per recording. Unknown whether
  real exports match.
- **Do:** Document what an export from this site actually is (container, codec,
  one file or many, how times are named) in [[milestone-recording-ingest]].
  Add `scripts/au_make_manifest.py` that writes a manifest from a video plus
  `--camera-id --camera-name --enclosure-id --start-time --timezone`, reading
  duration from the file. Adjust the contract if exports span several files.
- **Done when:** A real export runs through `au_run.py --milestone-manifest`
  with correct real-world times in SQLite, and a test covers the manifest tool.
- **Never commit footage.** `data/` is git-ignored; keep it that way.

### T3. Evaluation harness
- **Status:** TODO (build with synthetic clips now; real labels need Djuro)
- **Why:** Without ground truth nobody can say whether a change made things
  better. This is what makes the rest of the queue safe to do in parallel.
- **Do:** Ground-truth format `data/eval/<clip>.pacing.csv` with
  `start_s,end_s,animal` for each pacing bout. `scripts/au_eval.py` runs the
  pipeline over a folder of clips and reports, per clip and overall: pacing
  precision and recall (an event inside a labeled bout is a hit), number of
  track IDs per labeled animal, and frames per second. Output JSON plus a
  Markdown table.
- **Done when:** Tests cover the scoring; the synthetic clips score as
  expected; results table is appended to [[verification-record]]. Every later
  PR that touches detection, tracking or behavior must paste its before/after
  table.

---

## P1: make it accurate and remove the old paths

### T4. Site-specific model
- **Status:** TODO (after T1, T2)
- **Do:** `au_autolabel.py` over the archived clips, review `review/`, train
  with `au_train.py` from `yolo26s.pt`, compare open-vocabulary vs fine-tuned
  on the T3 set.
- **Done when:** The comparison table is in [[verification-record]] and the
  chosen checkpoint's path convention is documented. Weights stay out of Git.

### T5. XProtect event round trip (needs Djuro: Event Server access)
- **Status:** TODO
- **Do:** Follow the setup in `services/eventbridge/event_sender.py`. Send one
  test event with `python -m services.eventbridge.event_sender`. Fix the XML if
  the server rejects it. Write `docs/vault/backend/xprotect-events-runbook.md`.
- **Done when:** The event appears in the Smart Client and the runbook says
  exactly how to repeat it. Never write server addresses or credentials in Git.

### T6. Retire the old pipelines
- **Status:** TODO (after T1)
- **Do:** Make `scripts/au_live.py` a thin wrapper over `run_pipeline`. Replace
  `main.py` with a wrapper or remove it, migrating anything that reads
  `zoo_university.db`. Drop the `supervision` dependency. Update `README.md`.
- **Done when:** One pipeline remains, tests pass, and the UI still runs.
  Post a notice in [[backend-to-frontend]] because `main.py` behavior changes.

### T7. More behaviors
- **Status:** TODO (after T3)
- **Do:** Introduce a `BehaviorAnalyzer` interface
  (`update(tracks, timestamp_s, zone) -> List[BehaviorEvent]`) and let
  `run_pipeline` take a list of analyzers. Add, in this order: inactivity
  (no movement beyond half a body length for N minutes), zone occupancy (time
  per named zone), activity budget (moving vs still per hour).
- **Done when:** Each has tests in the style of `tests/test_pacing.py`
  (positive cases and near-miss negatives) and a T3 score where labels exist.

---

## P2: run it as a service

### T8. Multi-camera service
- **Status:** TODO
- **Do:** One YAML file listing cameras (name, enclosure, source, zone,
  classes or weights). A runner that starts one pipeline per camera, restarts
  on failure and logs health. Replace the placeholder in
  `infra/docker-compose.yml` with a real image.
- **Done when:** Two sources run side by side from one config for an hour
  without drift, and the runbook explains start, stop and logs.

### T9. Event review hook
- **Status:** TODO (needs Djuro: what Jev accepts)
- **Do:** A `CallbackSink` handler that saves a short clip and key frames for
  each event under `data/events/`. Optional second handler that sends key
  frames to an external reviewer and stores its verdict next to the event.
- **Done when:** Works offline with the reviewer disabled; the reviewer is
  configured only through environment variables.

---

## Frontend (Claude Code), tracked here for visibility

- Adopt `MotionTracker` in the demo and remove the UI's own label smoothing if
  the tracker's voted labels make it redundant. See H-20261112-01 in
  [[backend-to-frontend]].
- A results page that reads `SqliteSink` output: clips, pacing by hour, paths.
