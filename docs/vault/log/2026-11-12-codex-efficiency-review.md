---
type: session-log
owner: codex
updated: 2026-11-12
---

# T10: Pipeline efficiency and model review

Djuro requested a review of the unified pipeline and improvements where justified.
[Review PR #4](https://github.com/pantos12/AnimalUniversity/pull/4).
Worked in an isolated worktree based on PR #3 (`45fef2d`); claimed T10 in its own
published commit before editing. The review branch is stacked on PR #3 because
the reviewed implementation is not on main yet. No frontend implementation edits.

Implemented offline model preparation, optional boxes-only YOLOE setup, archived
batching, profiling, guarded cleanup, transactional SQLite flushes, stricter
input validation, recording-group dataset splits and append integrity checks.
Added the hardware diagnostic after Djuro identified the M5 development machine
and RTX 3080 deployment plan. Updated shared README/requirements and vault notes
to describe the new setup and changed dataset behavior; added a frontend handoff.

## Verified

- 87 CPU-side tests pass with both pytest entry points, including the 39 original
  tests. Bug regressions were reproduced before implementing their fixes.
- Actual Ultralytics 8.4.211 / torch 2.15.0 model preparation and CPU inference.
  Python socket connections were denied during inference, with zero attempted
  calls to socket.connect.
- Small boxes-only/batch 4: 34.046 median FPS vs original small segmentation/
  batch 1: 22.921 FPS on short synthetic clips; nano boxes-only/batch 4: 66.866 FPS.
  Detailed methodology and limitations: [[pipeline-review-2026-11-12]].
- CLI with a prepared model, SQLite and profiled JSON output completed.
- Same 200-frame synthetic horizontal/vertical/circle summaries before and after:
  one track each; pacing alert counts 3 / 3 / 0. Batch equivalence also tested.
- Hardware diagnostic: CPU tensor operation passed; MPS built but unavailable to
  this process; CUDA unavailable here. Compilation and diff whitespace checks pass.

## Not verified

- Real animal detection, real pacing precision/recall, identity under occlusion.
- NVIDIA or Apple GPU execution, sustained multi-camera throughput or memory.
- Complete pinned requirements compatibility, training, legacy pipelines and UI.
- Live XProtect export intake and Event Server acceptance.

T1 remains open. Added T11 for decoder/capture timing, live-source lifecycle,
bounded long-run report memory and durable asynchronous event delivery. The
review PR records CI status. Next steps: [[hardware-test-runbook]], independent
real recordings, T3 evaluation and human-reviewed labels before choosing a
production model or interpreting dashboard alerts.
