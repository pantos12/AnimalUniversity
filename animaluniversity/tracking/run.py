from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional


def track_video(
    video_path: str | Path,
    weights_path: str | Path,
    output_dir: str | Path,
    tracker: Literal["bytetrack", "botsort"] = "botsort",
    device: Optional[str] = None,
) -> Path:
    """
    Run YOLO inference + tracking on a video.

    Inputs:
    - video_path: input video file
    - weights_path: model weights
    - output_dir: directory for outputs
    - tracker: tracking algorithm
    - device: device string or None for auto

    Outputs:
    - path to output directory with tracks.jsonl and annotated video
    """
    raise NotImplementedError(
        "Tracking pipeline not implemented yet. "
        "Implement in animaluniversity/tracking/run.py."
    )
