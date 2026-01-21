# AnimalUniversity

Animal behavior analytics from Milestone XProtect live feeds.

## Structure
- services/ingest        RTSP ingest + frame pipeline
- services/analytics     detection, tracking, behavior logic
- services/eventbridge   XProtect analytics events/metadata sender
- infra/                 deployment and container configs
- docs/                  design notes and API references
- data/                  local dev data (ignored by git)

## Next
- Configure XProtect source (AI Bridge RTSP)
- Implement detector + tracker
- Emit analytics events back to XProtect
