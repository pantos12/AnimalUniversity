# CLAUDE.md — Claude Code (Frontend)

Claude Code owns the **frontend**. Codex owns the backend (see `AGENTS.md`).

## Owned by Claude Code — edit freely
- `apps/ui_streamlit/` — Streamlit app, layout, demo page, overlay rendering
- `.streamlit/config.toml` — theme and server config

## Owned by Codex — do NOT edit
- `services/ingest/`, `services/analytics/`, `services/eventbridge/`
- `animaluniversity/` — `core`, `yolo`, `sam2`, `tracking`, `metrics`, `utils`
- `scripts/` — training, labeling, live-run, env setup
- `main.py` — batch video processing pipeline (backend, despite the name)
- `infra/`, `models/`

## Shared — coordinate before editing
`README.md`, `requirements.txt`, `docs/`, `.gitignore`, `.env.example`

Touch these only when the change is unavoidable, keep the diff minimal, and say so
in the PR description so the other agent can rebase.

## FE/BE contract
`apps/ui_streamlit/app.py` depends on exactly these backend symbols:

```python
from animaluniversity.core.video import get_video_metadata
from animaluniversity.utils.paths import get_data_dir, get_models_dir
from services.analytics.detector import YoloDetector
from services.analytics.tracker import Detection, IoUTracker, Track
```

Treat this list as the API. If the UI needs something new from the backend,
**do not reach into other backend modules** — request the addition and let Codex
implement it behind this surface.

## Git workflow
- Branch from `main` as `feat/frontend-<slug>` (or `fix/frontend-<slug>`).
- Never commit directly to `main`.
- Rebase on `origin/main` before opening a PR.
- One PR per logical change; keep frontend PRs free of backend files.

## Running the UI
```powershell
.\scripts\run_ui.ps1
```
