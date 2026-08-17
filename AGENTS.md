# AGENTS.md — Codex (Backend)

Codex owns the **backend**. Claude Code owns the frontend (see `CLAUDE.md`).

## Owned by Codex — edit freely
- `services/ingest/` — RTSP ingest + frame pipeline
- `services/analytics/` — detection, tracking, behavior, reporting
- `services/eventbridge/` — XProtect analytics event sender
- `animaluniversity/` — `core`, `yolo`, `sam2`, `tracking`, `metrics`, `utils`
- `scripts/` — training, labeling, live-run, env setup
- `main.py` — batch video processing pipeline
- `infra/`, `models/`

## Owned by Claude Code — do NOT edit
- `apps/ui_streamlit/`
- `.streamlit/config.toml`

## Shared — coordinate before editing
`README.md`, `requirements.txt`, `docs/`, `.gitignore`, `.env.example`

Touch these only when the change is unavoidable, keep the diff minimal, and say so
in the PR description so the other agent can rebase.

## FE/BE contract
The Streamlit UI imports exactly these symbols. Changing their signatures or
behavior is a **breaking change** — flag it in the PR title:

```python
from animaluniversity.core.video import get_video_metadata
from animaluniversity.utils.paths import get_data_dir, get_models_dir
from services.analytics.detector import YoloDetector
from services.analytics.tracker import Detection, IoUTracker, Track
```

New capability the UI needs should be exposed through this surface rather than by
having the UI import deeper backend modules.

## Shared knowledge base (required)
The repository is an Obsidian vault. Shared, versioned knowledge starts at
`docs/vault/00-index.md`; Git is the source of truth.

Before backend work:
1. Read `docs/vault/00-index.md` and the collaboration protocol.
2. Resolve relevant `OPEN` entries in the frontend-to-backend inbox.
3. Read the FE/BE contract before boundary work.
4. Read relevant backend notes and ADRs.

Before finishing backend work:
- Update the relevant backend-owned note when status, risks, commands, or contracts
  changed.
- Append to the backend-to-frontend inbox when Claude Code must act or know about
  an API or behavior change.
- Record durable decisions as ADRs under `docs/vault/decisions/`.
- Follow vault ownership rules. Never commit secrets, credentials, production
  camera URLs, sensitive media, absolute private vault paths, or scratch notes.

## Git workflow
- Branch from `main` as `feat/backend-<slug>` (or `fix/backend-<slug>`).
- Never commit directly to `main`.
- Rebase on `origin/main` before opening a PR.
- One PR per logical change; keep backend PRs free of frontend files.
- Use a separate worktree from Claude Code. Never let both agents modify the same
  working tree concurrently.

## Notes
- Submodule `animaluniversity/sam2/third_party/segment-anything-2` is **not
  initialized** in this clone. Run
  `git submodule update --init --recursive` before SAM2 work.
