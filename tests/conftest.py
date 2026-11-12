from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np
import pytest

from services.analytics.tracker import Detection


def make_pacing_video(
    path: Path,
    seconds: float = 40.0,
    fps: int = 10,
    size=(640, 360),
    box: int = 60,
    period_s: float = 8.0,
    vertical: bool = False,
) -> None:
    """Write a clip of a bright square walking back and forth (a stand-in animal)."""
    width, height = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        pytest.skip("VideoWriter could not be opened (codec unavailable).")
    for i in range(int(seconds * fps)):
        t = i / fps
        phase = (t % period_s) / period_s
        tri = 2 * phase if phase < 0.5 else 2 * (1 - phase)  # 0 -> 1 -> 0
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        if vertical:
            x, y = width // 2 - box // 2, int(20 + tri * (height - box - 40))
        else:
            x, y = int(40 + tri * (width - box - 80)), height // 2 - box // 2
        frame[y : y + box, x : x + box] = 255
        writer.write(frame)
    writer.release()


class BrightBlobDetector:
    """Test double with a real ``detect``: finds bright blobs in the image."""

    def __init__(self, label: str = "lion", confidence: float = 0.9) -> None:
        self.label = label
        self.confidence = confidence

    def detect(self, frame_bgr) -> List[Detection]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h < 100:
                continue
            out.append(
                Detection(bbox_xyxy=(float(x), float(y), float(x + w), float(y + h)),
                          confidence=self.confidence, label=self.label)
            )
        return out


@pytest.fixture
def pacing_video(tmp_path: Path) -> Path:
    path = tmp_path / "pacing.mp4"
    make_pacing_video(path)
    return path
