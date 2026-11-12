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

## Still unverified (remove a line when an entry above proves it)

- `OpenVocabDetector` loads and detects on real footage.
- `ultralytics>=8.4` works with `main.py`, the UI and `scripts/au_train.py`.
- AnalyticsEvent XML is accepted by an XProtect Event Server.
- Real XProtect exports fit the manifest contract.
- Pacing precision and recall on real animals.
- The CI workflow passes on GitHub.
