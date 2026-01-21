# Architecture

## Overview
Ingest RTSP from Milestone AI Bridge -> detection/tracking -> behavior rules -> events/metadata back to XProtect.

## Components
- Ingest: pulls RTSP via OpenCV or GStreamer
- Analytics: detector + tracker + behavior rules
- EventBridge: sends AnalyticsEvent XML to XProtect Event Server

## MVP
1) RTSP ingest works for 1 camera
2) Detection+tracking on live stream
3) One behavior rule (pacing)
4) Send analytics event to XProtect
5) Metadata overlays (optional)
