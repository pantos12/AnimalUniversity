"""Behavior rules scaffold."""
from __future__ import annotations

import math
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
    """Online pacing detector: repeated back-and-forth travel along one route.

    For each track the recent positions are projected onto their principal axis
    of movement, so pacing is detected whichever way the route runs in the image
    (left-right, toward/away from the camera, or diagonal). A reversal is counted
    each time the animal travels from one end zone of the route to the other; the
    middle of the route is a dead band, so an animal standing or jittering
    mid-route does not accumulate crossings.

    ``min_span_px`` is an absolute minimum route length in pixels.
    ``min_span_body_lengths`` additionally requires the route to be that many
    times the animal's own size, which keeps the rule independent of camera
    resolution and of how far away the animal is.
    """

    def __init__(
        self,
        window_s: float = 20.0,
        min_crossings: int = 4,
        min_span_px: float = 80.0,
        cooldown_s: float = 10.0,
        min_span_body_lengths: float = 1.5,
        min_straightness: float = 0.75,
    ) -> None:
        self.window_s = max(window_s, 1.0)
        self.min_crossings = max(min_crossings, 1)
        self.min_span_px = max(min_span_px, 1.0)
        self.cooldown_s = max(cooldown_s, 0.0)
        self.min_span_body_lengths = max(min_span_body_lengths, 0.0)
        self.min_straightness = min(max(min_straightness, 0.0), 1.0)
        # (timestamp, center_x, center_y, body_size)
        self._history: Dict[int, Deque[Tuple[float, float, float, float]]] = defaultdict(deque)
        self._last_emit: Dict[int, float] = {}

    def _prune(self, timestamp_s: float) -> None:
        """Forget tracks that have not been seen for a full window."""
        cutoff = timestamp_s - self.window_s
        for track_id in [tid for tid, h in self._history.items() if not h or h[-1][0] < cutoff]:
            del self._history[track_id]
        emit_cutoff = timestamp_s - max(self.cooldown_s, self.window_s)
        for track_id in [tid for tid, t in self._last_emit.items() if t < emit_cutoff]:
            if track_id not in self._history:
                del self._last_emit[track_id]

    @staticmethod
    def _principal_axis(
        xs: List[float], ys: List[float]
    ) -> Tuple[Tuple[float, float], float]:
        """Return the unit vector of greatest variance and its share of total variance."""
        n = float(len(xs))
        mx, my = sum(xs) / n, sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs) / n
        syy = sum((y - my) ** 2 for y in ys) / n
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
        total = sxx + syy
        if total <= 1e-9:
            return (1.0, 0.0), 0.0
        half_trace = total / 2.0
        root = math.sqrt(max(0.0, ((sxx - syy) / 2.0) ** 2 + sxy**2))
        lam = half_trace + root
        if abs(sxy) > 1e-9:
            vx, vy = lam - syy, sxy
        elif sxx >= syy:
            vx, vy = 1.0, 0.0
        else:
            vx, vy = 0.0, 1.0
        norm = math.hypot(vx, vy) or 1.0
        return (vx / norm, vy / norm), lam / total

    def update(
        self,
        tracks: Iterable[object],
        timestamp_s: float,
        zone: Zone | None = None,
    ) -> List[BehaviorEvent]:
        events: List[BehaviorEvent] = []
        self._prune(timestamp_s)

        for track in tracks:
            if not hasattr(track, "track_id") or not hasattr(track, "bbox_xyxy"):
                continue
            track_id = int(track.track_id)
            center = _bbox_center(track.bbox_xyxy)
            if zone is not None and not _point_in_polygon(center, zone.polygon):
                continue
            x1, y1, x2, y2 = track.bbox_xyxy
            body = max(x2 - x1, y2 - y1)

            hist = self._history[track_id]
            hist.append((timestamp_s, center[0], center[1], body))
            min_ts = timestamp_s - self.window_s
            while hist and hist[0][0] < min_ts:
                hist.popleft()

            if len(hist) < 6:
                continue

            xs = [h[1] for h in hist]
            ys = [h[2] for h in hist]
            axis, straightness = self._principal_axis(xs, ys)
            if straightness < self.min_straightness:
                continue  # wandering, not a repeated route

            proj = [x * axis[0] + y * axis[1] for x, y in zip(xs, ys)]
            lo, hi = min(proj), max(proj)
            span = hi - lo
            bodies = sorted(h[3] for h in hist)
            median_body = bodies[len(bodies) // 2]
            if span < self.min_span_px or span < self.min_span_body_lengths * median_body:
                continue

            # Count end-to-end traversals; the middle 40% of the route is a dead band.
            low_edge, high_edge = lo + 0.3 * span, hi - 0.3 * span
            side = 0
            crossing_count = 0
            for value in proj:
                current = -1 if value <= low_edge else (1 if value >= high_edge else 0)
                if current == 0:
                    continue
                if side != 0 and current != side:
                    crossing_count += 1
                side = current

            if crossing_count < self.min_crossings:
                continue

            last_emit = self._last_emit.get(track_id)
            if last_emit is not None and (timestamp_s - last_emit) < self.cooldown_s:
                continue

            self._last_emit[track_id] = timestamp_s
            angle = math.degrees(math.atan2(axis[1], axis[0])) % 180.0
            events.append(
                BehaviorEvent(
                    name="pacing",
                    track_id=track_id,
                    score=min(1.0, crossing_count / float(self.min_crossings * 2)),
                    details={
                        "crossings": str(crossing_count),
                        "span_px": f"{span:.1f}",
                        "span_body_lengths": f"{(span / median_body) if median_body else 0.0:.1f}",
                        "axis_deg": f"{angle:.0f}",
                        "straightness": f"{straightness:.2f}",
                        "window_s": f"{self.window_s:.1f}",
                        "zone": zone.name if zone else "global",
                    },
                )
            )

        return events
