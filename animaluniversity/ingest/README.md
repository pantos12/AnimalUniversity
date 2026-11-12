# Milestone Recording Contract

This document defines the **offline Milestone recording contract** used for local
ingest tests. A single exported MP4 video and a JSON manifest are dropped into a
known folder, then validated and ingested with frame extraction.

## Drop folder layout

```
data/milestone_drop/
  mock_recording.mp4
  mock_recording.json
```

You can choose any base directory, but the `video_filename` in the manifest must
match a file in that folder. The ingest helper expects the manifest path and
base directory, so this is fully offline and local.

## Manifest schema

The JSON manifest must conform to this schema:

```json
{
  "camera_id": "C01",
  "camera_name": "Lion Yard 1",
  "enclosure_id": "E-LION-1",
  "species": "Panthera leo",
  "start_time": "2024-07-01T10:00:00-05:00",
  "end_time": "2024-07-01T10:10:00-05:00",
  "timezone": "America/Chicago",
  "fps": 10,
  "resolution": "1920x1080",
  "video_filename": "mock_recording.mp4",
  "notes": "Mock export for offline ingest tests."
}
```

### Field details

- `camera_id` / `camera_name`: identifiers for the source camera.
- `enclosure_id`: enclosure identifier for downstream analytics.
- `species`: optional, used for metadata enrichment.
- `start_time` / `end_time`: ISO 8601 timestamps with timezones.
- `timezone`: IANA timezone string for reference.
- `fps`: optional target frame rate of the recording.
- `resolution`: optional `WIDTHxHEIGHT` string.
- `video_filename`: file name only, no path segments.
- `notes`: optional extra context.

## Ingest helper

Use `animaluniversity.ingest.milestone.ingest_milestone_recording` to:

1. Validate the manifest + video file.
2. Extract frames into `<output_dir>/<video_stem>/frame_*.jpg`.
3. Write a `milestone_manifest.jsonl` enriched with camera, enclosure, and
   timestamp metadata.
