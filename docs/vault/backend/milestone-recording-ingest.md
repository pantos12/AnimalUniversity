---
type: backend-status
owner: codex
status: active
updated: 2026-11-12
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

## Status (2026-11-12)

- Contract, loader and validation implemented in `animaluniversity/ingest/`,
  with tests in `tests/test_milestone_ingest.py`.
- Validation fixes: mixed naive/aware times, unknown IANA timezones and
  malformed JSON now give clear errors that name the field; `strict` and
  `duration_tolerance_s` are caller-controlled.
- Exports can be analysed directly, without extracting frames, through
  `animaluniversity.pipeline.milestone_export_source`
  (see [[adr-003-unified-pipeline-and-open-vocab-bootstrap]]).

## Open questions

- Should the first UI select a fixed drop folder, a directory, or two files?
- What do real XProtect exports from this site look like (container, codec,
  whether one export spans several files)? The contract assumes one MP4.

Resolved: frame timestamps expose the offset-aware ISO 8601 value, a UTC value
(`frame_time_utc`) and the original IANA timezone.
