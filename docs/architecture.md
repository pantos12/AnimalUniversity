# Architecture

## Overview
Ingest RTSP from Milestone AI Bridge -> detection/tracking -> behavior rules -> events/metadata back to XProtect.

## Components
- Ingest: pulls RTSP via OpenCV or GStreamer
- Analytics: detector + tracker + behavior rules
- EventBridge: sends AnalyticsEvent XML to XProtect Event Server

## Model strategy
- Runtime model is YOLO-based for low-latency live feeds.
- `ena24` and `custom` selections run directly on provided YOLO checkpoints.
- `sam2` and `sam3` selections are train-first workflows:
  1) label/pseudo-label frames with SAM2
  2) convert to YOLO dataset
  3) train YOLO
  4) deploy trained YOLO checkpoint to live pipeline
- Current implementation maps `sam3` to the SAM2-assisted workflow until a validated SAM3 runtime path is added.

## MVP
1) RTSP ingest works for 1 camera
2) Detection+tracking on live stream
3) One behavior rule (pacing)
4) Send analytics event to XProtect
5) Metadata overlays (optional)
