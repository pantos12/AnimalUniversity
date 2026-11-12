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

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_missed: int = 8,
        allow_label_switch: bool = False,
        switch_iou_threshold: float = 0.55,
        switch_conf_max: float = 0.65,
        switch_iou_penalty: float = 0.12,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.allow_label_switch = allow_label_switch
        self.switch_iou_threshold = switch_iou_threshold
        self.switch_conf_max = switch_conf_max
        self.switch_iou_penalty = switch_iou_penalty
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
                iou = _bbox_iou(det.bbox_xyxy, state.bbox_xyxy)
                if state.label == det.label:
                    if iou >= self.iou_threshold:
                        candidates.append((iou, det_idx, track_id))
                    continue

                if not self.allow_label_switch:
                    continue
                if det.confidence > self.switch_conf_max:
                    continue
                if iou < self.switch_iou_threshold:
                    continue
                candidates.append((iou - self.switch_iou_penalty, det_idx, track_id))
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

            prev_state = self._states.get(track_id)
            assigned_label = det.label
            # Preserve stable label on low-confidence cross-label switches.
            if prev_state is not None and prev_state.label != det.label and self.allow_label_switch:
                assigned_label = prev_state.label

            self._states[track_id] = _TrackState(
                bbox_xyxy=det.bbox_xyxy,
                label=assigned_label,
                score=det.confidence,
                last_seen_frame=self._frame_idx,
            )
            tracks.append(
                Track(
                    track_id=track_id,
                    bbox_xyxy=det.bbox_xyxy,
                    label=assigned_label,
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


@dataclass
class _MotionState:
    bbox_xyxy: Tuple[float, float, float, float]
    velocity: Tuple[float, float]
    label_votes: Dict[str, float]
    score: float
    hits: int
    last_seen_frame: int

    @property
    def label(self) -> str:
        return max(self.label_votes.items(), key=lambda kv: kv[1])[0]


def _shift(
    bbox: Tuple[float, float, float, float], dx: float, dy: float
) -> Tuple[float, float, float, float]:
    return (bbox[0] + dx, bbox[1] + dy, bbox[2] + dx, bbox[3] + dy)


def _center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _diag(bbox: Tuple[float, float, float, float]) -> float:
    return ((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2) ** 0.5


def _area(bbox: Tuple[float, float, float, float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


class MotionTracker(Tracker):
    """ByteTrack-style tracker with motion prediction and label voting.

    Built for low frame rates (about 5 fps) where an animal can move further than
    its own body between frames, which defeats plain IoU matching:

    - Each track carries a velocity estimate, and matching is done against the
      *predicted* box, so a track survives missed detections.
    - When predicted and detected boxes do not overlap at all, a match is still
      allowed if the centers are close relative to the box size and the sizes are
      similar.
    - Matching ignores the class label. The label reported for a track is a
      confidence-weighted vote over its history, so a detector that flickers
      between "bear" and "dog" neither splits the track nor changes its label.
    - Two-stage association: confident detections are matched first, then
      low-confidence ones may only extend existing tracks, never start new ones.

    No dependencies beyond the standard library.
    """

    def __init__(
        self,
        iou_threshold: float = 0.2,
        max_missed: int = 15,
        high_conf: float = 0.4,
        center_gate: float = 1.5,
        velocity_smoothing: float = 0.6,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.high_conf = high_conf
        self.center_gate = center_gate
        self.velocity_smoothing = velocity_smoothing
        self._next_id = 1
        self._frame_idx = 0
        self._states: Dict[int, _MotionState] = {}

    def _predicted(self, state: _MotionState) -> Tuple[float, float, float, float]:
        gap = self._frame_idx - state.last_seen_frame
        return _shift(state.bbox_xyxy, state.velocity[0] * gap, state.velocity[1] * gap)

    def _match_score(self, det: Detection, state: _MotionState) -> float:
        predicted = self._predicted(state)
        iou = max(_bbox_iou(det.bbox_xyxy, predicted), _bbox_iou(det.bbox_xyxy, state.bbox_xyxy))
        if iou >= self.iou_threshold:
            return 1.0 + iou

        # No usable overlap: fall back to center distance, gated by box size.
        area_det, area_trk = _area(det.bbox_xyxy), _area(predicted)
        if area_det <= 0 or area_trk <= 0:
            return 0.0
        if not 0.4 <= area_det / area_trk <= 2.5:
            return 0.0
        scale = (_diag(det.bbox_xyxy) + _diag(predicted)) / 2.0
        if scale <= 0:
            return 0.0
        (dx, dy), (px, py) = _center(det.bbox_xyxy), _center(predicted)
        dist = ((dx - px) ** 2 + (dy - py) ** 2) ** 0.5 / scale
        gap = max(1, self._frame_idx - state.last_seen_frame)
        gate = self.center_gate * min(gap, 3)
        if dist > gate:
            return 0.0
        return 1.0 - dist / gate

    def _associate(
        self, dets: List[Detection], det_indices: List[int], track_ids: List[int]
    ) -> Dict[int, int]:
        candidates: List[Tuple[float, int, int]] = []
        for det_idx in det_indices:
            for track_id in track_ids:
                score = self._match_score(dets[det_idx], self._states[track_id])
                if score > 0:
                    candidates.append((score, det_idx, track_id))
        candidates.sort(reverse=True)
        matches: Dict[int, int] = {}
        used: set[int] = set()
        for _score, det_idx, track_id in candidates:
            if det_idx in matches or track_id in used:
                continue
            matches[det_idx] = track_id
            used.add(track_id)
        return matches

    def update(self, detections: Iterable[Detection]) -> List[Track]:
        self._frame_idx += 1
        dets = list(detections)
        high = [i for i, d in enumerate(dets) if d.confidence >= self.high_conf]
        low = [i for i, d in enumerate(dets) if d.confidence < self.high_conf]

        track_ids = list(self._states.keys())
        matches = self._associate(dets, high, track_ids)
        remaining = [tid for tid in track_ids if tid not in set(matches.values())]
        matches.update(self._associate(dets, low, remaining))

        tracks: List[Track] = []
        for det_idx, det in enumerate(dets):
            track_id = matches.get(det_idx)
            if track_id is None:
                if det.confidence < self.high_conf:
                    continue  # weak detections never start a track
                track_id = self._next_id
                self._next_id += 1
                self._states[track_id] = _MotionState(
                    bbox_xyxy=det.bbox_xyxy,
                    velocity=(0.0, 0.0),
                    label_votes={det.label: det.confidence},
                    score=det.confidence,
                    hits=1,
                    last_seen_frame=self._frame_idx,
                )
            else:
                state = self._states[track_id]
                gap = max(1, self._frame_idx - state.last_seen_frame)
                (cx, cy), (px, py) = _center(det.bbox_xyxy), _center(state.bbox_xyxy)
                observed = ((cx - px) / gap, (cy - py) / gap)
                if state.hits == 1:
                    velocity = observed
                else:
                    a = self.velocity_smoothing
                    velocity = (
                        a * observed[0] + (1 - a) * state.velocity[0],
                        a * observed[1] + (1 - a) * state.velocity[1],
                    )
                state.bbox_xyxy = det.bbox_xyxy
                state.velocity = velocity
                state.label_votes[det.label] = state.label_votes.get(det.label, 0.0) + det.confidence
                state.score = det.confidence
                state.hits += 1
                state.last_seen_frame = self._frame_idx

            state = self._states[track_id]
            tracks.append(
                Track(
                    track_id=track_id,
                    bbox_xyxy=det.bbox_xyxy,
                    label=state.label,
                    score=det.confidence,
                )
            )

        stale = [
            tid
            for tid, st in self._states.items()
            if (self._frame_idx - st.last_seen_frame) > self.max_missed
        ]
        for track_id in stale:
            del self._states[track_id]
        return tracks
