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

## Shared knowledge base (required)
The repository is an Obsidian vault. Shared, versioned knowledge starts at
`docs/vault/00-index.md`; Git is the source of truth.

Before frontend work:
1. Read `docs/vault/00-index.md` and the collaboration protocol.
2. Resolve relevant `OPEN` entries in the backend-to-frontend inbox.
3. Read the FE/BE contract before boundary work.
4. Read relevant frontend notes and ADRs.

Before finishing frontend work:
- Update the relevant frontend-owned note when status, UX assumptions, commands,
  or contracts changed.
- Append to the frontend-to-backend inbox when Codex must act or know about an API
  or behavior expectation.
- Record durable decisions as ADRs under `docs/vault/decisions/`.
- Follow vault ownership rules. Never commit secrets, credentials, production
  camera URLs, sensitive media, absolute private vault paths, or scratch notes.

## Git workflow
- Branch from `main` as `feat/frontend-<slug>` (or `fix/frontend-<slug>`).
- Never commit directly to `main`.
- Rebase on `origin/main` before opening a PR.
- One PR per logical change; keep frontend PRs free of backend files.
- Use a separate worktree from Codex. Never let both agents modify the same working
  tree concurrently.

## Running the UI
```powershell
.\scripts\run_ui.ps1
```
