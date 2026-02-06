from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import cv2

logger = logging.getLogger(__name__)


def get_video_metadata(video_path: str | Path) -> Dict[str, float]:
    """Return basic video metadata."""
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration_sec = (frame_count / fps) if fps else 0.0
    cap.release()

    return {
        "duration_sec": float(duration_sec),
        "fps": float(fps),
        "width": float(width),
        "height": float(height),
        "frame_count": float(frame_count),
    }


def _write_manifest(manifest_path: Path, rows: List[Dict[str, float | str]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _read_manifest_tail(manifest_path: Path) -> Optional[Dict[str, float | str]]:
    if not manifest_path.exists():
        return None
    last = None
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                last = json.loads(line)
    return last


def extract_frames(
    video_path: str | Path,
    out_dir: str | Path,
    fps: int = 1,
    start_sec: float = 0,
    end_sec: Optional[float] = None,
    overwrite: bool = False,
    use_ffmpeg: bool = True,
    max_frames: Optional[int] = None,
    resume: bool = False,
) -> List[Dict[str, float | str]]:
    """
    Extract frames from a video.

    Returns a manifest list with keys: frame_index, timestamp_sec, frame_path.
    Also writes out_dir/manifest.jsonl.
    """
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if fps <= 0:
        raise ValueError("fps must be > 0")

    manifest: List[Dict[str, float | str]] = []
    output_pattern = str(out_dir / "frame_%06d.jpg")
    manifest_path = out_dir / "manifest.jsonl"

    if resume and overwrite:
        raise ValueError("resume=True is incompatible with overwrite=True")

    if manifest_path.exists() and not overwrite and not resume:
        logger.info("Manifest already exists and overwrite=False: %s", manifest_path)
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                manifest.append(json.loads(line))
        return manifest

    # Try ffmpeg-python first for shorter clips, unless disabled
    ffmpeg_ok = False
    if resume and use_ffmpeg:
        logger.warning("resume=True is only supported with OpenCV. Disabling ffmpeg.")
        use_ffmpeg = False

    if use_ffmpeg:
        try:
            import ffmpeg  # type: ignore

            kwargs = {}
            if end_sec is not None:
                duration = max(0.0, end_sec - start_sec)
                kwargs["t"] = duration
            stream = ffmpeg.input(str(video_path), ss=start_sec, **kwargs)
            stream = stream.filter("fps", fps=fps)
            runner = stream.output(output_pattern, start_number=1)
            if overwrite:
                runner = runner.overwrite_output()
            runner.run(quiet=True)
            ffmpeg_ok = True
            logger.info("Extracted frames using ffmpeg.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("ffmpeg extraction failed; falling back to OpenCV. Reason: %s", exc)

    if not ffmpeg_ok:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Unable to open video: {video_path}")

        native_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if native_fps <= 0:
            native_fps = 30.0
        resume_state = _read_manifest_tail(manifest_path) if resume else None
        if resume_state:
            start_sec = float(resume_state["timestamp_sec"]) + (1.0 / float(fps))
            start_index = int(resume_state["frame_index"])
        else:
            start_index = 0

        start_msec = max(0.0, start_sec) * 1000.0
        end_msec = (end_sec * 1000.0) if end_sec is not None else None
        cap.set(cv2.CAP_PROP_POS_MSEC, start_msec)

        interval_msec = 1000.0 / float(fps)
        next_msec = start_msec
        frame_idx = start_index

        mode = "a" if resume else "w"
        with manifest_path.open(mode, encoding="utf-8") as mf:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                current_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                if end_msec is not None and current_msec > end_msec:
                    break
                if current_msec + 1e-3 >= next_msec:
                    frame_idx += 1
                    frame_path = out_dir / f"frame_{frame_idx:06d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    record = {
                        "frame_index": frame_idx,
                        "timestamp_sec": float(current_msec / 1000.0),
                        "frame_path": str(frame_path),
                    }
                    mf.write(json.dumps(record) + "\n")
                    manifest.append(record)
                    if frame_idx % 200 == 0:
                        logger.info("Extracted %d frames...", frame_idx)
                    if max_frames is not None and frame_idx >= max_frames:
                        logger.info("Reached max_frames=%d, stopping early.", max_frames)
                        break
                    next_msec += interval_msec
        cap.release()
        logger.info("Extracted frames using OpenCV.")

    if ffmpeg_ok:
        # Build manifest from files on disk
        frame_files = sorted(out_dir.glob("frame_*.jpg"))
        for i, frame_path in enumerate(frame_files, start=1):
            timestamp_sec = start_sec + (i - 1) / float(fps)
            manifest.append(
                {
                    "frame_index": i,
                    "timestamp_sec": float(timestamp_sec),
                    "frame_path": str(frame_path),
                }
            )
        _write_manifest(manifest_path, manifest)

    logger.info("Wrote manifest: %s (%d frames)", manifest_path, len(manifest))
    return manifest


def sample_frames_uniform(frame_dir: str | Path, n: int = 200) -> List[str]:
    """Return up to n frame paths uniformly sampled from a directory."""
    frame_dir = Path(frame_dir)
    frames = sorted(frame_dir.glob("frame_*.jpg"))
    if not frames:
        return []
    if n >= len(frames):
        return [str(p) for p in frames]

    step = max(1, len(frames) // n)
    sampled = frames[::step][:n]
    return [str(p) for p in sampled]
