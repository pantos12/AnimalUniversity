# AnimalUniversity

Animal behavior analytics from Milestone XProtect live feeds.

## Structure
- services/ingest        RTSP ingest + frame pipeline
- services/analytics     detection, tracking, behavior logic
- services/eventbridge   XProtect analytics events/metadata sender
- infra/                 deployment and container configs
- docs/                  design notes and API references
- data/                  local dev data (ignored by git)

## Quick start (dev)
1) Set `RTSP_URL` and run ingest
2) Plug in detector + tracker
3) Send AnalyticsEvent XML via EventBridge

## Local Quickstart
1) Create and activate a virtual environment, then install deps:
   - PowerShell: `.\scripts\setup_env.ps1`
   - Bash: `./scripts/setup_env.sh`
2) Extract frames from a local video:
   - `python scripts/au_frames.py --video "data/raw_videos/example.mp4" --fps 1 --out "data/frames/example"`
3) Run the Streamlit UI:
   - `.\scripts\run_ui.ps1`
