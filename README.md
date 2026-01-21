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
