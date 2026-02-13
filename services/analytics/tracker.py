"""Tracking utilities for live analytics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


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


def _bbox_iou(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    if denom <= 0:
        return 0.0
    return inter / denom


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


@dataclass
class _TrackState:
    bbox_xyxy: Tuple[float, float, float, float]
    label: str
    score: float
    last_seen_frame: int


class IoUTracker(Tracker):
    """Greedy IoU tracker with per-label matching.

    This keeps IDs stable enough for basic behavior analytics without external
    dependencies.
    """

    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 8) -> None:
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self._next_id = 1
        self._frame_idx = 0
        self._states: Dict[int, _TrackState] = {}

    def update(self, detections: Iterable[Detection]) -> List[Track]:
        self._frame_idx += 1
        dets = list(detections)
        active_ids = [
            tid
            for tid, st in self._states.items()
            if (self._frame_idx - st.last_seen_frame) <= self.max_missed
        ]

        # Greedy assignment on IoU, keeping labels consistent.
        candidates: List[Tuple[float, int, int]] = []
        for det_idx, det in enumerate(dets):
            for track_id in active_ids:
                state = self._states[track_id]
                if state.label != det.label:
                    continue
                iou = _bbox_iou(det.bbox_xyxy, state.bbox_xyxy)
                if iou >= self.iou_threshold:
                    candidates.append((iou, det_idx, track_id))
        candidates.sort(reverse=True)

        assigned_det: Dict[int, int] = {}
        used_tracks: set[int] = set()
        for _iou, det_idx, track_id in candidates:
            if det_idx in assigned_det or track_id in used_tracks:
                continue
            assigned_det[det_idx] = track_id
            used_tracks.add(track_id)

        tracks: List[Track] = []
        for det_idx, det in enumerate(dets):
            track_id = assigned_det.get(det_idx)
            if track_id is None:
                track_id = self._next_id
                self._next_id += 1

            self._states[track_id] = _TrackState(
                bbox_xyxy=det.bbox_xyxy,
                label=det.label,
                score=det.confidence,
                last_seen_frame=self._frame_idx,
            )
            tracks.append(
                Track(
                    track_id=track_id,
                    bbox_xyxy=det.bbox_xyxy,
                    label=det.label,
                    score=det.confidence,
                )
            )

        stale_ids = [
            tid
            for tid, st in self._states.items()
            if (self._frame_idx - st.last_seen_frame) > self.max_missed
        ]
        for track_id in stale_ids:
            del self._states[track_id]

        return tracks
