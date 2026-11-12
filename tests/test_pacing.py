from __future__ import annotations

import math

from services.analytics.behavior import PacingAnalyzer, Zone
from services.analytics.tracker import Track


def _track(cx: float, cy: float, size: float = 60.0, track_id: int = 1) -> Track:
    return Track(track_id, (cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2), "lion", 0.9)


def _triangle(t: float, period: float) -> float:
    phase = (t % period) / period
    return 2 * phase if phase < 0.5 else 2 * (1 - phase)


def _run(position, seconds: float = 40.0, fps: float = 5.0, analyzer=None, zone=None):
    analyzer = analyzer or PacingAnalyzer()
    events = []
    for i in range(int(seconds * fps)):
        t = i / fps
        events += analyzer.update([_track(*position(t))], timestamp_s=t, zone=zone)
    return events


def test_horizontal_pacing_detected() -> None:
    events = _run(lambda t: (100 + 400 * _triangle(t, 8.0), 300))
    assert events and events[0].name == "pacing"
    assert abs(float(events[0].details["axis_deg"])) < 10 or float(events[0].details["axis_deg"]) > 170


def test_vertical_pacing_detected() -> None:
    # The original analyzer only looked at x, so this was invisible to it.
    events = _run(lambda t: (300, 100 + 400 * _triangle(t, 8.0)))
    assert events
    assert 80 <= float(events[0].details["axis_deg"]) <= 100


def test_diagonal_pacing_detected() -> None:
    events = _run(lambda t: (100 + 300 * _triangle(t, 8.0), 100 + 300 * _triangle(t, 8.0)))
    assert events
    assert 35 <= float(events[0].details["axis_deg"]) <= 55


def test_standing_with_jitter_is_not_pacing() -> None:
    events = _run(lambda t: (300 + 6 * math.sin(t * 9), 300 + 6 * math.cos(t * 7)))
    assert events == []


def test_one_walk_across_then_resting_mid_route_is_not_pacing() -> None:
    # Old logic counted every jitter across the centerline as a crossing.
    def position(t: float):
        if t < 5:
            return (100 + 80 * t, 300)
        return (300 + 4 * math.sin(t * 11), 300)

    assert _run(position) == []


def test_wandering_in_a_circle_is_not_pacing() -> None:
    events = _run(lambda t: (300 + 200 * math.cos(t), 300 + 200 * math.sin(t)))
    assert events == []


def test_small_shuffle_relative_to_body_is_not_pacing() -> None:
    # 100 px route but the animal is 300 px long: weight shifting, not pacing.
    analyzer = PacingAnalyzer()
    events = []
    for i in range(200):
        t = i / 5.0
        events += analyzer.update([_track(300 + 100 * _triangle(t, 4.0), 300, size=300)], timestamp_s=t)
    assert events == []


def test_cooldown_limits_repeat_events() -> None:
    events = _run(lambda t: (100 + 400 * _triangle(t, 8.0), 300), seconds=60.0)
    # 60 s of continuous pacing with a 10 s cooldown: a handful of events, not one per frame
    assert 2 <= len(events) <= 6


def test_zone_filters_tracks() -> None:
    far_zone = Zone("elsewhere", [(2000, 2000), (2100, 2000), (2100, 2100), (2000, 2100)])
    assert _run(lambda t: (100 + 400 * _triangle(t, 8.0), 300), zone=far_zone) == []


def test_history_of_vanished_tracks_is_released() -> None:
    analyzer = PacingAnalyzer(window_s=20.0)
    for track_id in range(500):
        analyzer.update([_track(100, 100, track_id=track_id)], timestamp_s=float(track_id))
    assert len(analyzer._history) <= 25
