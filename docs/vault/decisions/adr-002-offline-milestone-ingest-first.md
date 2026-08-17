---
type: adr
owner: shared
status: accepted
date: 2026-08-17
---

# ADR-002: Build Offline Milestone Recording Ingest First

## Status

accepted

## Context

Production cameras use Milestone XProtect, but the first integration phase must be
repeatable without a live stream or production system. A single exported MP4 and a
JSON manifest can model the recording boundary deterministically.

## Decision

Build and validate the Milestone recording contract against a mock MP4 export and
JSON manifest, fully offline, before implementing live-stream integration.

This phase includes input validation, frame extraction, metadata enrichment, and
existing pipeline entry points. It excludes SAM2 behavior, YOLO training, and new
tracking logic.

## Consequences

- Tests do not require a live XProtect connection.
- Camera, enclosure, and timezone-aware timestamps are first-class data.
- Test fixtures stay small and deterministic.
- Frontend work waits for a stable backend handoff before using this flow.

Related status note: [[milestone-recording-ingest]].

## Alternatives considered

- **Start with live RTSP.** Rejected because it makes tests depend on camera and
  production-like infrastructure.
- **Start with model/tracking changes.** Rejected because the ingest contract must
  be stable before downstream behavior can be tested reliably.
