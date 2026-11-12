from __future__ import annotations

from services.analytics.tracker import Detection, IoUTracker, MotionTracker


def _box(cx: float, cy: float, size: float = 100.0):
    return (cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2)


def _fast_animal_frames():
    """An animal moving 1.5 body lengths per frame, with dropouts and label flicker."""
    frames = []
    for i in range(30):
        if i in (8, 9, 17):  # detector misses it
            frames.append([])
            continue
        label = "dog" if i in (5, 12, 20) else "bear"
        conf = 0.5 if label == "dog" else 0.85
        frames.append([Detection(_box(100 + 150 * i, 300), conf, label)])
    return frames


def test_motion_tracker_keeps_one_id_for_fast_animal() -> None:
    tracker = MotionTracker()
    ids, labels = set(), []
    for dets in _fast_animal_frames():
        for tr in tracker.update(dets):
            ids.add(tr.track_id)
            labels.append(tr.label)
    assert ids == {1}
    # label is voted over history, so brief misclassifications do not stick
    assert labels[-1] == "bear" and labels.count("dog") == 0


def test_iou_tracker_fragments_on_the_same_input() -> None:
    # Documents why MotionTracker exists: same input, plain IoU matching.
    tracker = IoUTracker(iou_threshold=0.3, max_missed=8)
    ids = set()
    for dets in _fast_animal_frames():
        ids.update(tr.track_id for tr in tracker.update(dets))
    assert len(ids) > 5


def test_two_animals_side_by_side_keep_their_ids() -> None:
    tracker = MotionTracker()
    seen = {}
    for i in range(25):
        dets = [
            Detection(_box(100 + 40 * i, 200), 0.9, "zebra"),
            Detection(_box(100 + 40 * i, 420), 0.9, "zebra"),
        ]
        for tr in tracker.update(dets):
            lane = "top" if tr.bbox_xyxy[1] < 300 else "bottom"
            seen.setdefault(lane, set()).add(tr.track_id)
    assert len(seen["top"]) == 1 and len(seen["bottom"]) == 1
    assert seen["top"] != seen["bottom"]


def test_animals_crossing_paths_do_not_swap() -> None:
    # Each id must keep travelling in one direction through the crossing.
    tracker = MotionTracker()
    xs = {}
    for i in range(21):
        dets = [
            Detection(_box(100 + 50 * i, 300), 0.9, "zebra"),
            Detection(_box(1100 - 50 * i, 330), 0.9, "zebra"),
        ]
        for tr in tracker.update(dets):
            xs.setdefault(tr.track_id, []).append((tr.bbox_xyxy[0] + tr.bbox_xyxy[2]) / 2)
    assert len(xs) == 2
    for series in xs.values():
        steps = [b - a for a, b in zip(series, series[1:])]
        assert all(s > 0 for s in steps) or all(s < 0 for s in steps)


def test_weak_detection_does_not_start_a_track() -> None:
    tracker = MotionTracker(high_conf=0.4)
    assert tracker.update([Detection(_box(100, 100), 0.2, "bear")]) == []
    assert len(tracker.update([Detection(_box(100, 100), 0.8, "bear")])) == 1
    # ...but a weak detection does extend an existing track
    assert tracker.update([Detection(_box(110, 100), 0.2, "bear")])[0].track_id == 1


def test_stale_tracks_are_dropped() -> None:
    tracker = MotionTracker(max_missed=3)
    tracker.update([Detection(_box(100, 100), 0.9, "bear")])
    for _ in range(5):
        tracker.update([])
    assert tracker.update([Detection(_box(100, 100), 0.9, "bear")])[0].track_id == 2
