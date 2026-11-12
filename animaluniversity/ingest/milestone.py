from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from animaluniversity.core.video import extract_frames, get_video_metadata

logger = logging.getLogger(__name__)


class MilestoneRecordingManifest(BaseModel):
    camera_id: str
    camera_name: str
    enclosure_id: str
    species: Optional[str] = None
    start_time: datetime
    end_time: datetime
    timezone: str
    fps: Optional[int] = None
    resolution: Optional[str] = None
    video_filename: str
    notes: Optional[str] = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("camera_id", "camera_name", "enclosure_id")
    @classmethod
    def _validate_identifier(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("identifier must be non-empty")
        return value

    @field_validator("fps")
    @classmethod
    def _validate_fps(cls, value: Optional[int]) -> Optional[int]:
        if value is None:
            return value
        if value <= 0:
            raise ValueError("fps must be > 0 when provided")
        return value

    @field_validator("timezone")
    @classmethod
    def _validate_timezone(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("timezone must be a non-empty string")
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(
                f"timezone must be a valid IANA name such as 'America/New_York', got {value!r}"
            ) from exc
        return value

    @field_validator("video_filename")
    @classmethod
    def _validate_video_filename(cls, value: str) -> str:
        if (not value or value in {".", ".."} or "\x00" in value
                or Path(value).name != value or PureWindowsPath(value).name != value):
            raise ValueError("video_filename must be a file name, not a path")
        return value

    @model_validator(mode="after")
    def _validate_time_range(self) -> "MilestoneRecordingManifest":
        if self.start_time.tzinfo is None or self.end_time.tzinfo is None:
            raise ValueError("start_time and end_time must be timezone-aware ISO 8601 values")
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be after start_time")
        return self


@dataclass(frozen=True)
class MilestoneIngestResult:
    manifest: MilestoneRecordingManifest
    video_path: Path
    video_metadata: Dict[str, float]
    frames_dir: Path
    frames_manifest: List[Dict[str, float | str]]
    enriched_manifest_path: Path


def load_milestone_manifest(manifest_path: str | Path) -> MilestoneRecordingManifest:
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Milestone manifest is not valid JSON: {manifest_path} ({exc})") from exc
    try:
        return MilestoneRecordingManifest.model_validate(payload)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in err['loc']) or 'manifest'}: {err['msg']}"
            for err in exc.errors()
        )
        raise ValueError(f"Invalid Milestone manifest {manifest_path}: {problems}") from exc


def load_milestone_recording_manifest(
    manifest_path: str | Path,
) -> MilestoneRecordingManifest:
    return load_milestone_manifest(manifest_path)


def _parse_resolution(resolution: str) -> Optional[Tuple[int, int]]:
    if "x" not in resolution.lower():
        return None
    parts = resolution.lower().split("x")
    if len(parts) != 2:
        return None
    try:
        width = int(parts[0].strip())
        height = int(parts[1].strip())
    except ValueError:
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


def validate_milestone_assets(
    manifest: MilestoneRecordingManifest,
    base_dir: str | Path,
    *,
    strict: bool = True,
    duration_tolerance_s: float = 2.0,
) -> Tuple[Path, Dict[str, float]]:
    """Check the video against its manifest.

    With ``strict=False`` mismatches are logged instead of raised. Real XProtect
    exports often differ from the requested time range by a few seconds (exports
    snap to keyframes), so callers handling real footage usually want
    ``strict=False`` or a larger ``duration_tolerance_s``.
    """
    base_dir = Path(base_dir)
    video_path = base_dir / manifest.video_filename
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if video_path.suffix.lower() != ".mp4":
        message = f"Expected .mp4 video, got {video_path.suffix}"
        if strict:
            raise ValueError(message)
        logger.warning(message)

    metadata = get_video_metadata(video_path)
    duration = metadata.get("duration_sec", 0.0)
    expected_duration = (manifest.end_time - manifest.start_time).total_seconds()
    if duration and abs(duration - expected_duration) > duration_tolerance_s:
        message = (
            "Video duration does not match manifest time range "
            f"(video={duration:.2f}s, manifest={expected_duration:.2f}s)."
        )
        if strict:
            raise ValueError(message)
        logger.warning(message)

    if manifest.fps:
        if abs(metadata.get("fps", 0.0) - float(manifest.fps)) > 1.0:
            message = (
                "Manifest fps does not match video metadata "
                f"(video={metadata.get('fps')}, manifest={manifest.fps})."
            )
            if strict:
                raise ValueError(message)
            logger.warning(message)

    if manifest.resolution:
        parsed = _parse_resolution(manifest.resolution)
        if parsed:
            width, height = parsed
            if int(metadata.get("width", 0.0)) != width or int(metadata.get("height", 0.0)) != height:
                message = (
                    "Manifest resolution does not match video metadata "
                    f"(video={int(metadata.get('width', 0.0))}x{int(metadata.get('height', 0.0))}, "
                    f"manifest={manifest.resolution})."
                )
                if strict:
                    raise ValueError(message)
                logger.warning(message)
        else:
            message = f"Resolution value could not be parsed: {manifest.resolution}"
            if strict:
                raise ValueError(message)
            logger.warning(message)

    return video_path, metadata


def _write_enriched_manifest(
    out_path: Path,
    rows: Iterable[Dict[str, Any]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _enrich_frames(
    frames_manifest: List[Dict[str, float | str]],
    recording: MilestoneRecordingManifest,
) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    for row in frames_manifest:
        timestamp_sec = float(row["timestamp_sec"])
        absolute_time = recording.start_time + timedelta(seconds=timestamp_sec)
        enriched.append(
            {
                **row,
                "camera_id": recording.camera_id,
                "camera_name": recording.camera_name,
                "enclosure_id": recording.enclosure_id,
                "species": recording.species,
                "timezone": recording.timezone,
                "recording_start_time": recording.start_time.isoformat(),
                "recording_end_time": recording.end_time.isoformat(),
                "frame_time": absolute_time.isoformat(),
                "frame_time_utc": absolute_time.astimezone(timezone.utc).isoformat(),
                "video_filename": recording.video_filename,
            }
        )
    return enriched


def ingest_milestone_recording(
    manifest_path: str | Path,
    *,
    base_dir: str | Path,
    output_dir: str | Path,
    target_fps: Optional[int] = None,
    overwrite: bool = False,
    resume: bool = False,
    strict: bool = True,
    duration_tolerance_s: float = 2.0,
) -> MilestoneIngestResult:
    """
    Validate and ingest an offline Milestone recording.

    This validates the JSON manifest and video, extracts frames, and writes
    an enriched frame manifest with camera + enclosure metadata.
    """
    recording = load_milestone_manifest(manifest_path)
    video_path, video_metadata = validate_milestone_assets(
        recording,
        base_dir,
        strict=strict,
        duration_tolerance_s=duration_tolerance_s,
    )

    fps = target_fps or recording.fps or max(int(video_metadata.get("fps", 1.0)), 1)
    frames_dir = Path(output_dir) / Path(recording.video_filename).stem
    frames_manifest = extract_frames(
        video_path=video_path,
        out_dir=frames_dir,
        fps=int(fps),
        overwrite=overwrite,
        resume=resume,
    )

    enriched = _enrich_frames(frames_manifest, recording)
    enriched_manifest_path = frames_dir / "milestone_manifest.jsonl"
    _write_enriched_manifest(enriched_manifest_path, enriched)

    return MilestoneIngestResult(
        manifest=recording,
        video_path=video_path,
        video_metadata=video_metadata,
        frames_dir=frames_dir,
        frames_manifest=frames_manifest,
        enriched_manifest_path=enriched_manifest_path,
    )
