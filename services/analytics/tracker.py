"""Tracker scaffold.

Define interfaces for detections and tracks. Actual models are plugged in later.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple


@dataclass
class Detection:
    bbox_xyxy: Tuple[float, float, float, float]
    confidence: float
    label: str


@dataclass
class Track:
    track_id: int
    bbox_xyxy: Tuple[float, float, float, float]
    label: str
    score: float


class Tracker:
    def update(self, detections: Iterable[Detection]) -> List[Track]:
        """Update the tracker with current detections."""
        raise NotImplementedError


class NoopTracker(Tracker):
    """Pass-through tracker for early integration tests."""

    def __init__(self) -> None:
        self._next_id = 1

    def update(self, detections: Iterable[Detection]) -> List[Track]:
        tracks: List[Track] = []
        for det in detections:
            tracks.append(
                Track(
                    track_id=self._next_id,
                    bbox_xyxy=det.bbox_xyxy,
                    label=det.label,
                    score=det.confidence,
                )
            )
            self._next_id += 1
        return tracks
