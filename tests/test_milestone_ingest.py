from __future__ import annotations

import json
from pathlib import Path

import pytest

from animaluniversity.ingest.milestone import (
    MilestoneRecordingManifest,
    ingest_milestone_recording,
    load_milestone_manifest,
)
from tests.conftest import make_pacing_video

BASE = {
    "camera_id": "C01",
    "camera_name": "Lion Yard 1",
    "enclosure_id": "E-LION-1",
    "species": "Panthera leo",
    "start_time": "2024-07-01T10:00:00-04:00",
    "end_time": "2024-07-01T10:00:10-04:00",
    "timezone": "America/New_York",
    "fps": 10,
    "resolution": "640x360",
    "video_filename": "rec.mp4",
}


def _write(tmp_path: Path, **overrides) -> Path:
    path = tmp_path / "rec.json"
    path.write_text(json.dumps({**BASE, **overrides}), encoding="utf-8")
    return path


def test_valid_manifest_loads(tmp_path: Path) -> None:
    manifest = load_milestone_manifest(_write(tmp_path))
    assert manifest.camera_id == "C01"


def test_mixed_naive_and_aware_times_is_a_clean_error(tmp_path: Path) -> None:
    # Used to escape as a raw TypeError from the datetime comparison.
    with pytest.raises(ValueError, match="timezone-aware"):
        load_milestone_manifest(_write(tmp_path, start_time="2024-07-01T10:00:00"))


def test_unknown_timezone_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="IANA"):
        load_milestone_manifest(_write(tmp_path, timezone="Narnia/Lion"))


def test_error_names_the_bad_fields(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"camera_id": "C01", "fps": -1}), encoding="utf-8")
    with pytest.raises(ValueError) as info:
        load_milestone_manifest(path)
    message = str(info.value)
    assert "camera_name" in message and "fps" in message


def test_not_json_is_a_clean_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_milestone_manifest(path)


def test_path_in_video_filename_rejected() -> None:
    with pytest.raises(ValueError):
        MilestoneRecordingManifest(**{**BASE, "video_filename": "../rec.mp4"})


def test_ingest_end_to_end(tmp_path: Path) -> None:
    make_pacing_video(tmp_path / "rec.mp4", seconds=10.0)
    result = ingest_milestone_recording(
        _write(tmp_path), base_dir=tmp_path, output_dir=tmp_path / "frames", target_fps=1
    )
    rows = [json.loads(line) for line in result.enriched_manifest_path.read_text().splitlines()]
    assert len(rows) >= 9
    assert rows[0]["enclosure_id"] == "E-LION-1"
    assert rows[0]["frame_time"].startswith("2024-07-01T10:00:0")
    assert rows[0]["frame_time_utc"].startswith("2024-07-01T14:00:0")


def test_duration_mismatch_strict_vs_lenient(tmp_path: Path) -> None:
    make_pacing_video(tmp_path / "rec.mp4", seconds=10.0)
    manifest = _write(tmp_path, end_time="2024-07-01T10:01:00-04:00")  # claims 60 s
    with pytest.raises(ValueError, match="duration"):
        ingest_milestone_recording(manifest, base_dir=tmp_path, output_dir=tmp_path / "f1", target_fps=1)
    result = ingest_milestone_recording(
        manifest, base_dir=tmp_path, output_dir=tmp_path / "f2", target_fps=1, strict=False
    )
    assert result.frames_manifest
