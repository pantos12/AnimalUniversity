---
type: knowledge
owner: shared
updated: 2026-11-12
---

# Verification Record

What has actually been run, where, and with what result. Append; do not
rewrite history. Format per entry: date, who, environment, command, result.

## 2026-11-12, Claude, CPU-only sandbox (Python 3.12, no torch)

- `pytest`: 39 passed.
- Pipeline on three synthetic two-minute clips with one frame in ten withheld
  from the detector: pacing reported on the two pacing clips (11 and 7
  events), none on the circling clip; one track ID per clip.
- Not run: any model, any real footage, XProtect, GitHub Actions.

## 2026-11-12, Codex, M5/macOS 27.0.1 arm64

- Python 3.12.14 CPU test environment: 87 tests pass (including the 39 originals),
  using both `pytest -q` and `python -m pytest -q`. Model packages are not needed
  by these tests. Regressions reproduced before fixes include partial SQLite
  batch loss, sink-startup cleanup, class-order overwrite and invalid sampling.
- Separate model environment: torch 2.15.0, Ultralytics 8.4.211, numpy 2.5.4,
  OpenCV 5.0.0.93, pydantic 2.14.3, CLIP 1.0 from its official repository.
  This is not a complete `requirements.txt` compatibility check.
- Prepared small/nano YOLOE-26 boxes-only checkpoints with cached lion prompts;
  actual model loading and CPU prediction executed. A process with Python socket.connect denied
  completed six warmed configurations with zero attempted calls. Detailed
  timings and synthetic-only limits: [[pipeline-review-2026-11-12]].
- Full CLI smoke: `au_run.py` with a prepared small checkpoint, CPU, batch 4,
  profile, SQLite and JSON output completed on 32 synthetic frames, zero animals.
- Hardware diagnostic: MPS built=True, available=False; CUDA=False; CPU tensor
  probe passed. Neither Apple GPU nor RTX 3080 execution has been tested.
- Before/after summaries matched on horizontal/vertical/circling 40-second
  synthetic clips with 10% missed detections: one track per clip, pacing alert
  counts 3 / 3 / 0, with 200 analyzed frames each. Not accuracy ground truth.
- PR #3's [GitHub CI run 2](https://github.com/pantos12/AnimalUniversity/actions/runs/43255778030)
  passed. Review-branch CI status is recorded in its PR.

## Still unverified (remove a line when an entry above proves it)

- `OpenVocabDetector` loads and detects on real footage.
- `ultralytics>=8.4` works with `main.py`, the UI and `scripts/au_train.py`.
- AnalyticsEvent XML is accepted by an XProtect Event Server.
- Real XProtect exports fit the manifest contract.
- Pacing precision and recall on real animals.
