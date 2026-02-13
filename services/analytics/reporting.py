"""Runtime analytics aggregation and reporting."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from math import hypot
from typing import Dict, Iterable, List, Tuple

from services.analytics.behavior import BehaviorEvent
from services.analytics.tracker import Detection, Track


def _bbox_center(bbox_xyxy: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox_xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass
class TrackAggregate:
    track_id: int
    label: str
    first_ts: float
    last_ts: float
    frames_seen: int = 0
    conf_sum: float = 0.0
    conf_max: float = 0.0
    distance_px: float = 0.0
    last_center: Tuple[float, float] | None = None
    event_counts: Counter[str] = field(default_factory=Counter)

    def update(self, ts: float, score: float, center: Tuple[float, float]) -> None:
        self.last_ts = ts
        self.frames_seen += 1
        self.conf_sum += float(score)
        self.conf_max = max(self.conf_max, float(score))
        if self.last_center is not None:
            self.distance_px += hypot(center[0] - self.last_center[0], center[1] - self.last_center[1])
        self.last_center = center

    def to_dict(self) -> Dict[str, object]:
        duration_s = max(0.0, self.last_ts - self.first_ts)
        avg_conf = (self.conf_sum / float(self.frames_seen)) if self.frames_seen else 0.0
        avg_speed = (self.distance_px / duration_s) if duration_s > 1e-6 else 0.0
        return {
            "track_id": self.track_id,
            "label": self.label,
            "duration_s": round(duration_s, 3),
            "frames_seen": self.frames_seen,
            "avg_confidence": round(avg_conf, 4),
            "max_confidence": round(self.conf_max, 4),
            "distance_px": round(self.distance_px, 2),
            "avg_speed_px_s": round(avg_speed, 2),
            "events": dict(self.event_counts),
        }


class AnalyticsAccumulator:
    """Accumulates per-frame analytics into a compact summary report."""

    def __init__(self) -> None:
        self.frames_processed = 0
        self.frames_with_detections = 0
        self.total_detections = 0
        self.max_concurrent_tracks = 0
        self.detections_by_label: Counter[str] = Counter()
        self.behavior_events_by_name: Counter[str] = Counter()
        self.track_stats: Dict[int, TrackAggregate] = {}

    def update(
        self,
        ts: float,
        detections: Iterable[Detection],
        tracks: Iterable[Track],
        behavior_events: Iterable[BehaviorEvent],
    ) -> None:
        self.frames_processed += 1

        detections_list = list(detections)
        tracks_list = list(tracks)
        events_list = list(behavior_events)

        if detections_list:
            self.frames_with_detections += 1
        self.total_detections += len(detections_list)
        for det in detections_list:
            self.detections_by_label[str(det.label)] += 1

        self.max_concurrent_tracks = max(self.max_concurrent_tracks, len(tracks_list))
        for tr in tracks_list:
            agg = self.track_stats.get(tr.track_id)
            if agg is None:
                agg = TrackAggregate(
                    track_id=tr.track_id,
                    label=str(tr.label),
                    first_ts=ts,
                    last_ts=ts,
                )
                self.track_stats[tr.track_id] = agg
            agg.update(ts=ts, score=float(tr.score), center=_bbox_center(tr.bbox_xyxy))

        for ev in events_list:
            name = str(ev.name)
            self.behavior_events_by_name[name] += 1
            agg = self.track_stats.get(int(ev.track_id))
            if agg is not None:
                agg.event_counts[name] += 1

    def summary(self, top_tracks: int = 20) -> Dict[str, object]:
        frame_ratio = (
            float(self.frames_with_detections) / float(self.frames_processed)
            if self.frames_processed
            else 0.0
        )
        tracks = [agg.to_dict() for agg in self.track_stats.values()]
        tracks.sort(key=lambda row: float(row["duration_s"]), reverse=True)
        return {
            "frames_processed": self.frames_processed,
            "frames_with_detections": self.frames_with_detections,
            "frames_with_detections_ratio": round(frame_ratio, 4),
            "detections_total": self.total_detections,
            "detections_by_label": dict(self.detections_by_label),
            "unique_tracks": len(self.track_stats),
            "max_concurrent_tracks": self.max_concurrent_tracks,
            "behavior_events_total": int(sum(self.behavior_events_by_name.values())),
            "behavior_events_by_name": dict(self.behavior_events_by_name),
            "top_tracks": tracks[: max(0, top_tracks)],
        }
