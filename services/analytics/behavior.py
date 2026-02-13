"""Behavior rules scaffold."""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Tuple


@dataclass
class Zone:
    name: str
    polygon: List[Tuple[float, float]]


@dataclass
class BehaviorEvent:
    name: str
    track_id: int
    score: float
    details: Dict[str, str]


def detect_pacing(tracks: Iterable[object], zone: Zone, window_s: float) -> List[BehaviorEvent]:
    """Placeholder for pacing detection.

    Expected to analyze recent track paths and detect oscillation within a zone.
    """
    _ = (tracks, zone, window_s)
    return []


def _point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test."""
    if len(polygon) < 3:
        return True
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = (yi > y) != (yj > y)
        if intersects:
            denom = (yj - yi) if (yj - yi) else 1e-6
            cross_x = (xj - xi) * (y - yi) / denom + xi
            if x < cross_x:
                inside = not inside
        j = i
    return inside


def _bbox_center(bbox_xyxy: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox_xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class PacingAnalyzer:
    """Online pacing detector using centerline crossings over a time window."""

    def __init__(
        self,
        window_s: float = 20.0,
        min_crossings: int = 4,
        min_span_px: float = 80.0,
        cooldown_s: float = 10.0,
    ) -> None:
        self.window_s = max(window_s, 1.0)
        self.min_crossings = max(min_crossings, 1)
        self.min_span_px = max(min_span_px, 1.0)
        self.cooldown_s = max(cooldown_s, 0.0)
        self._history: Dict[int, Deque[Tuple[float, float]]] = defaultdict(deque)
        self._last_emit: Dict[int, float] = {}

    def update(
        self,
        tracks: Iterable[object],
        timestamp_s: float,
        zone: Zone | None = None,
    ) -> List[BehaviorEvent]:
        events: List[BehaviorEvent] = []

        for track in tracks:
            if not hasattr(track, "track_id") or not hasattr(track, "bbox_xyxy"):
                continue
            track_id = int(track.track_id)
            center = _bbox_center(track.bbox_xyxy)
            if zone is not None and not _point_in_polygon(center, zone.polygon):
                continue

            hist = self._history[track_id]
            hist.append((timestamp_s, center[0]))
            min_ts = timestamp_s - self.window_s
            while hist and hist[0][0] < min_ts:
                hist.popleft()

            if len(hist) < 6:
                continue

            xs = [x for _t, x in hist]
            x_span = max(xs) - min(xs)
            if x_span < self.min_span_px:
                continue

            centerline = (max(xs) + min(xs)) / 2.0
            crossing_count = 0
            for i in range(1, len(xs)):
                prev = xs[i - 1] - centerline
                curr = xs[i] - centerline
                if prev == 0:
                    continue
                if (prev > 0 and curr < 0) or (prev < 0 and curr > 0):
                    crossing_count += 1

            if crossing_count < self.min_crossings:
                continue

            last_emit = self._last_emit.get(track_id)
            if last_emit is not None and (timestamp_s - last_emit) < self.cooldown_s:
                continue

            self._last_emit[track_id] = timestamp_s
            events.append(
                BehaviorEvent(
                    name="pacing",
                    track_id=track_id,
                    score=min(1.0, crossing_count / float(self.min_crossings * 2)),
                    details={
                        "crossings": str(crossing_count),
                        "span_px": f"{x_span:.1f}",
                        "window_s": f"{self.window_s:.1f}",
                        "zone": zone.name if zone else "global",
                    },
                )
            )

        return events
