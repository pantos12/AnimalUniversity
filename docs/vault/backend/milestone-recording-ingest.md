---
type: backend-status
owner: codex
status: active
updated: 2026-08-17
---

# Milestone Recording Ingest

Current backend priority. Durable scope decision:
[[adr-002-offline-milestone-ingest-first]].

## Goal

Accept one mock Milestone XProtect export as an MP4 plus a JSON manifest, validate
the pair, extract frames, attach camera/enclosure/timestamp metadata, and pass the
result through stable existing pipeline entry points.

## Manifest contract

- `camera_id`, `camera_name`, `enclosure_id`
- optional `species`
- timezone-aware `start_time` and `end_time`
- `timezone`
- optional `fps` and `resolution`
- `video_filename`
- optional `notes`

## Constraints

- Work fully offline after dependencies are installed.
- Do not add SAM2 behavior, YOLO training, or new tracking logic in this phase.
- Reject missing or mismatched video/manifest pairs with actionable errors.
- Preserve recording, camera, enclosure, and timestamp context on frame metadata.
- Keep videos, extracted frames, weights, and runtime databases out of Git.

## Intended backend location

- `animaluniversity/ingest/milestone.py` - contract, loader, and validation
- `animaluniversity/ingest/__init__.py` - package exports
- `docs/` - contract documentation and examples
- `tests/` - automated coverage

The public UI-facing surface is not defined yet. Once stable, Codex updates
[[fe-be-contract]] and posts a notice in [[backend-to-frontend]].

## Open questions

- Should the first UI select a fixed drop folder, a directory, or two files?
- Should frame timestamps expose only timezone-aware ISO 8601 values, or also UTC
  plus the original IANA timezone?
