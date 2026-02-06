from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from animaluniversity.core.video import get_video_metadata


def _make_tiny_video(path: Path, fps: int = 10, frames: int = 10) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (64, 48))
    if not writer.isOpened():
        pytest.skip("VideoWriter could not be opened (codec unavailable).")
    for _ in range(frames):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_get_video_metadata(tmp_path: Path) -> None:
    video_path = tmp_path / "tiny.mp4"
    _make_tiny_video(video_path)

    meta = get_video_metadata(video_path)
    assert meta["frame_count"] > 0
    assert meta["fps"] > 0
    assert meta["width"] == 64
    assert meta["height"] == 48
