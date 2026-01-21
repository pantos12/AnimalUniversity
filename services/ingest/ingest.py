"""RTSP ingest skeleton.

This is a minimal, dependency-light scaffold. If OpenCV is installed, it can open
an RTSP stream and yield frames. Otherwise it will explain what is missing.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Generator, Optional


@dataclass
class IngestConfig:
    rtsp_url: str
    target_fps: float = 5.0
    reconnect_seconds: float = 5.0


def _require_cv2():
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "OpenCV (cv2) is required for RTSP ingest. Install opencv-python."
        ) from exc
    return cv2


def frame_stream(cfg: IngestConfig) -> Generator[tuple[float, "object"], None, None]:
    cv2 = _require_cv2()
    cap = None
    interval = 1.0 / max(cfg.target_fps, 0.1)

    while True:
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(cfg.rtsp_url)
            if not cap.isOpened():
                time.sleep(cfg.reconnect_seconds)
                continue

        ok, frame = cap.read()
        if not ok:
            cap.release()
            cap = None
            time.sleep(cfg.reconnect_seconds)
            continue

        ts = time.time()
        yield ts, frame
        time.sleep(interval)


def main() -> int:
    rtsp_url = os.environ.get("RTSP_URL", "")
    if not rtsp_url:
        print("Set RTSP_URL to your Milestone AI Bridge stream.")
        return 2

    cfg = IngestConfig(rtsp_url=rtsp_url)
    for ts, _frame in frame_stream(cfg):
        print(f"frame @ {ts:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
