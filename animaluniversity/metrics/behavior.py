from __future__ import annotations

from pathlib import Path
from typing import Optional


def compute_metrics(
    tracks_path: str | Path,
    roi_path: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
) -> Path:
    """
    Compute behavior metrics from tracking output.

    Inputs:
    - tracks_path: JSONL with per-frame tracks
    - roi_path: optional ROI polygons for time-in-zone
    - output_dir: directory for report output

    Outputs:
    - path to behavior_report.json
    """
    raise NotImplementedError(
        "Behavior metrics not implemented yet. "
        "Implement in animaluniversity/metrics/behavior.py."
    )
