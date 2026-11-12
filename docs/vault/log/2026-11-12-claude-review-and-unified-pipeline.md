---
type: session-log
owner: shared
date: 2026-11-12
author: Claude (chat session with Djuro)
branch: feat/backend-unified-pipeline
---

# 2026-11-12: Project review and unified pipeline

Djuro asked for a whole-project review, improvements using current vision
tools, and a clear path to finishing. This note records what was found, what
was changed and what is still unproven. Decision record:
[[adr-003-unified-pipeline-and-open-vocab-bootstrap]]. Next work: [[work-queue]].

## The question that started it: live feed or cold storage?

Cold storage first, live second, same code for both. Archived footage is the
only way to (a) get training data from our own cameras, (b) measure whether a
behavior rule is right, and (c) rerun the same clip after every change. Live is
the deployment target, not the development environment. So the pipeline now
takes a *frame source*, and file, Milestone export and RTSP are interchangeable.

## What was wrong

| Area | Finding | How it was confirmed |
| --- | --- | --- |
| Structure | Three pipelines (`main.py`, `scripts/au_live.py`, Streamlit demo) shared little code; pacing ran only on live | Read the code |
| Install | `main.py` uses `sv.ByteTrack`; gone from `supervision` 0.31.7 (still in 0.30.0); requirement allowed `<1.0.0` | Inspected the published wheels |
| Live ingest | Reader slept between reads, so frames queue and analysis falls behind real time | Read the code |
| Pacing | Looked only at horizontal movement; any jitter across the centerline counted as a crossing | Read the code; new tests cover both |
| Events | XProtect XML unescaped, not in Milestone's `urn:milestone-systems` layout | Read the code |
| PR #2 | Based on the first commit, far behind `main`; no tests; mixed naive/aware times raised `TypeError`; any string accepted as timezone; errors did not name the bad field; strict mode not adjustable | Ran the validator |
| Labeling | Needed a hand-drawn prompt box per clip and an uninitialized SAM2 submodule | Read the code |

## What was changed

- `animaluniversity/pipeline/`: `FrameSource` (`VideoFileSource`,
  `milestone_export_source`, `RtspSource`), `run_pipeline`, and sinks
  (`SqliteSink`, `XProtectEventSink`, `CallbackSink`).
- `services/analytics/detector.py`: `OpenVocabDetector` (Ultralytics YOLOE),
  detects species by name with no training.
- `services/analytics/tracker.py`: `MotionTracker`. `IoUTracker` untouched.
- `services/analytics/behavior.py`: pacing along any direction, dead band in
  mid-route, route length relative to body size, straightness check, history
  released for vanished tracks.
- `services/eventbridge/event_sender.py`: escaped, namespaced AnalyticsEvent.
- `animaluniversity/ingest/milestone.py`: the PR #2 fixes above, plus
  `frame_time_utc`.
- `animaluniversity/labeling/` and `scripts/au_autolabel.py`: archived footage
  to YOLO dataset with a `review/` folder of drawn boxes.
- `scripts/au_run.py`: one command for any source.
- `requirements.txt`: `ultralytics>=8.4,<9`; `supervision<0.31`.
- Tests: 4 to 39. CI workflow added (`.github/workflows/tests.yml`).

The frontend was not touched. The four [[fe-be-contract]] symbols are unchanged.

## Verified and not verified

Verified by automated tests in a CPU-only sandbox:

- Ingest validation and frame enrichment.
- Tracker: one ID kept for a fast animal through dropouts and label flicker,
  where `IoUTracker` produced more than five; no swap when two animals cross.
- Pacing: horizontal, vertical and diagonal detected; standing, one walk
  across, circling and small shuffles not detected.
- Whole pipeline on synthetic clips, including SQLite contents, real-world
  times from a manifest, failure status, and repeatability.
- Live source hands over the newest frame when analysis is slow (fake camera).
- Event XML parses and escapes.

**Not verified. Treat as unproven until someone runs it:**

- `OpenVocabDetector` and anything else that loads a model. The sandbox had no
  torch or GPU. What was checked: `ultralytics` 8.4.211 ships `YOLOE`,
  `yoloe-26*-seg.pt`, YOLO26 and SAM 3, and `set_classes(names)` is its API.
- The upgraded `ultralytics` against `main.py`, the UI and training scripts.
- The AnalyticsEvent XML against a real XProtect Event Server. The layout was
  written from memory of Milestone's documentation.
- Everything on real zoo footage. All test clips are synthetic.
- The CI workflow has not run on GitHub yet.

## Where Jev could fit

Not in the real-time loop. Two optional uses through `CallbackSink`: a second
opinion on flagged events (needs image input), and plain-language keeper
summaries from the SQLite events (text only). Open question for Djuro: does
Jev accept images?

## Process note

This work was done outside the normal lanes (backend belongs to Codex), at
Djuro's request. ADR-003 is therefore `proposed`, and Codex should review the
branch before it merges.
