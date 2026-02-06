from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from animaluniversity.core.video import extract_frames


def _make_tiny_video(path: Path, fps: int = 10, frames: int = 10) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (64, 48))
    if not writer.isOpened():
        pytest.skip("VideoWriter could not be opened (codec unavailable).")
    for _ in range(frames):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_extract_frames_manifest(tmp_path: Path) -> None:
    video_path = tmp_path / "tiny.mp4"
    _make_tiny_video(video_path)

    out_dir = tmp_path / "frames"
    manifest = extract_frames(video_path=video_path, out_dir=out_dir, fps=2, overwrite=True)

    manifest_path = out_dir / "manifest.jsonl"
    assert manifest_path.exists()
    assert len(manifest) > 0
