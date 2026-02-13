# Analytics Service

Runs detection, tracking, and behavior logic over frames.

## Current runtime
- Detector: Ultralytics YOLO (`services/analytics/detector.py`)
- Tracker: lightweight IoU tracker (`services/analytics/tracker.py`)
- Behavior: pacing detector (`services/analytics/behavior.py`)

## Live entrypoint
Use `scripts/au_live.py` to run RTSP ingest + analytics + optional EventBridge send.
