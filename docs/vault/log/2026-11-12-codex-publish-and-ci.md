---
type: session-log
owner: codex
updated: 2026-11-12
---

# Publish unified pipeline and repair CI invocation

Published the supplied unified-pipeline bundle as PR #3 and closed superseded
PR #2 after verifying its original commit is an ancestor. The GitHub API
recreated the merge and two later commits because direct Git push had no
credentials. The published snapshot at bcec720 matches both supplied artifacts
exactly (tree 582fd602a34591b5d47b3a5cd9384d88d042b0a6).

GitHub Actions run 43255370748 failed before collection: the `pytest` console
launcher could not import `services`. Reproduced locally. Updated only the CI
invocation to `python -m pytest -q`, which puts the repository root on the import
path. This follow-up commit also records the verification here.

## Verified

- Patch SHA-256: 1ee68295d97bb79ccbdb040bc206b84d0d0cbb70fb597f991ff98f3cca504e7f.
- Bundle prerequisites and final tree equality against the patch and remote.
- Original PR #2 commit 97d25b5372771ac56a7c9759a17041b052181c59 preserved.
- Python 3.12.14, macOS arm64, isolated environment: `python -m pytest -q`,
  39 passed. Packages: numpy 2.5.4, opencv-python-headless 5.0.0.93,
  pydantic 2.14.3, pytest 9.2.0.
- `pytest -q` reproduced the CI import failure locally before the invocation fix.

## Not verified

- Model loading, inference, training, GPU execution, UI and real animal footage.
- Live RTSP or an actual XProtect Event Server round trip.
- T1 remains TODO; this publication does not complete it.

Next agent: consult [[work-queue]] and [[verification-record]] for model and
real-footage validation. The PR records the result of the follow-up CI run.
