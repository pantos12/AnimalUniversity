from __future__ import annotations

import json
import sqlite3
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from animaluniversity.labeling import autolabel_video
from animaluniversity.pipeline import (
    CallbackSink,
    RtspSource,
    SqliteSink,
    VideoFileSource,
    milestone_export_source,
    run_pipeline,
)
from services.analytics.detector import detections_from_result
from services.eventbridge.event_sender import MILESTONE_NS, build_analytics_event_xml
from tests.conftest import BrightBlobDetector, make_pacing_video


def test_video_source_resamples_with_video_time(pacing_video: Path) -> None:
    frames = list(VideoFileSource(pacing_video, target_fps=5))
    assert 195 <= len(frames) <= 201  # 40 s at 5 fps
    gaps = [b.timestamp_s - a.timestamp_s for a, b in zip(frames, frames[1:])]
    assert all(abs(g - 0.2) < 1e-6 for g in gaps)
    assert frames[0].wall_time is None


def test_same_clip_gives_identical_results_twice(pacing_video: Path) -> None:
    def run():
        return run_pipeline(VideoFileSource(pacing_video, target_fps=5), BrightBlobDetector())

    assert run() == run()


def test_archived_clip_end_to_end_with_sqlite(pacing_video: Path, tmp_path: Path) -> None:
    db = tmp_path / "au.db"
    seen = []
    summary = run_pipeline(
        VideoFileSource(pacing_video, target_fps=5),
        BrightBlobDetector(),
        sinks=[SqliteSink(db), CallbackSink(lambda frame, ev, info: seen.append(ev.name))],
    )
    assert summary["status"] == "completed"
    assert summary["unique_tracks"] == 1
    assert summary["behavior_events_by_name"]["pacing"] >= 1
    assert seen and set(seen) == {"pacing"}

    conn = sqlite3.connect(db)
    assert conn.execute("SELECT status, source_kind FROM runs").fetchall() == [("completed", "file")]
    assert conn.execute("SELECT COUNT(*) FROM track_points").fetchone()[0] == summary["detections_total"]
    assert conn.execute("SELECT COUNT(*) FROM behavior_events WHERE name='pacing'").fetchone()[0] >= 1


def test_vertical_pacing_clip_is_detected(tmp_path: Path) -> None:
    clip = tmp_path / "vertical.mp4"
    make_pacing_video(clip, vertical=True)
    summary = run_pipeline(VideoFileSource(clip, target_fps=5), BrightBlobDetector())
    assert summary["behavior_events_by_name"].get("pacing", 0) >= 1


def test_milestone_export_carries_real_time_and_camera(tmp_path: Path) -> None:
    make_pacing_video(tmp_path / "rec.mp4", seconds=40.0)
    manifest = tmp_path / "rec.json"
    manifest.write_text(
        json.dumps(
            {
                "camera_id": "C01",
                "camera_name": "Lion Yard 1",
                "enclosure_id": "E-LION-1",
                "start_time": "2024-07-01T10:00:00-04:00",
                "end_time": "2024-07-01T10:00:40-04:00",
                "timezone": "America/New_York",
                "video_filename": "rec.mp4",
            }
        )
    )
    db = tmp_path / "au.db"
    summary = run_pipeline(
        milestone_export_source(manifest, target_fps=5), BrightBlobDetector(), sinks=[SqliteSink(db)]
    )
    assert summary["source"]["enclosure_id"] == "E-LION-1"
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT camera_id, enclosure_id FROM runs").fetchone() == ("C01", "E-LION-1")
    first, last = conn.execute("SELECT MIN(wall_time), MAX(wall_time) FROM track_points").fetchone()
    assert first.startswith("2024-07-01T10:00:00") and last.startswith("2024-07-01T10:00:39")
    event_time = conn.execute("SELECT wall_time FROM behavior_events LIMIT 1").fetchone()[0]
    assert event_time.startswith("2024-07-01T10:00:")


def test_failed_run_is_marked_failed(pacing_video: Path, tmp_path: Path) -> None:
    class Exploding:
        def detect(self, image):
            raise RuntimeError("boom")

    db = tmp_path / "au.db"
    with pytest.raises(RuntimeError):
        run_pipeline(VideoFileSource(pacing_video, target_fps=5), Exploding(), sinks=[SqliteSink(db)])
    assert sqlite3.connect(db).execute("SELECT status FROM runs").fetchone() == ("failed",)


class _FakeCamera:
    """Produces numbered frames at 100 fps, like a camera that never waits."""

    def __init__(self, url: str) -> None:
        self.n = 0

    def isOpened(self) -> bool:
        return True

    def read(self):
        time.sleep(0.01)
        self.n += 1
        return True, np.full((4, 4, 3), self.n % 256, dtype=np.uint8)

    def release(self) -> None:
        pass


def test_live_source_delivers_newest_frame_not_a_backlog() -> None:
    source = RtspSource("rtsp://fake", target_fps=10, max_frames=5, capture_factory=_FakeCamera)
    frames = []
    for frame in source:
        frames.append(frame)
        time.sleep(0.15)  # analysis slower than the camera
    assert len(frames) == 5
    seqs = [f.source_frame for f in frames]
    # a slow consumer must skip ahead instead of working through every frame
    assert seqs[-1] - seqs[0] > 20
    assert all(f.wall_time.tzinfo is not None for f in frames)
    assert "fake" not in source.info.name  # URL (may hold credentials) is never stored


def test_event_xml_is_escaped_and_namespaced() -> None:
    xml = build_analytics_event_xml(
        "Lion <Yard> & 1",
        "AnimalBehavior.Pacing",
        {"zone": "a<b", "score": "0.9"},
        timestamp=datetime(2024, 7, 1, 14, 0, tzinfo=timezone.utc),
    )
    root = ET.fromstring(xml)
    ns = {"m": MILESTONE_NS}
    assert root.tag == f"{{{MILESTONE_NS}}}AnalyticsEvent"
    assert root.find("m:EventHeader/m:Source/m:Name", ns).text == "Lion <Yard> & 1"
    assert root.find("m:EventHeader/m:Message", ns).text == "AnimalBehavior.Pacing"
    assert root.find("m:EventHeader/m:Timestamp", ns).text == "2024-07-01T14:00:00+00:00"
    assert "zone=a<b" in root.find("m:EventHeader/m:CustomTag", ns).text
    with pytest.raises(ValueError):
        build_analytics_event_xml("c", "t", {}, timestamp=datetime(2024, 7, 1))


def test_detections_from_result_accepts_arrays() -> None:
    class Boxes:
        xyxy = np.array([[1.0, 2.0, 30.0, 40.0], [5.0, 5.0, 9.0, 9.0]])
        conf = np.array([0.9, 0.3])
        cls = np.array([1.0, 0.0])

        def __len__(self):
            return 2

    class Result:
        boxes = Boxes()
        names = {0: "tiger", 1: "red panda"}

    dets = detections_from_result(Result())
    assert [d.label for d in dets] == ["red panda", "tiger"]
    assert dets[0].bbox_xyxy == (1.0, 2.0, 30.0, 40.0)


def test_autolabel_builds_trainable_dataset(pacing_video: Path, tmp_path: Path) -> None:
    out = tmp_path / "dataset"
    result = autolabel_video(
        VideoFileSource(pacing_video, target_fps=1), BrightBlobDetector(label="lion"), ["lion"], out,
        split="train",
    )
    assert result["frames_kept"] >= 35 and result["boxes_per_class"]["lion"] >= 35
    images = sorted((out / "images").rglob("*.jpg"))
    labels = sorted((out / "labels").rglob("*.txt"))
    assert len(images) == len(labels) == result["frames_kept"]
    assert (out / "images" / "val").exists() and not any((out / "images" / "val").iterdir())
    cls, xc, yc, w, h = labels[0].read_text().split()
    assert cls == "0" and all(0.0 <= float(v) <= 1.0 for v in (xc, yc, w, h))
    assert abs(float(w) - 60 / 640) < 0.01 and abs(float(h) - 60 / 360) < 0.01
    assert 'names:\n  - "lion"' in (out / "dataset.yaml").read_text()
    assert any((out / "review").iterdir())


def test_autolabel_ignores_unknown_labels_and_low_confidence(pacing_video: Path, tmp_path: Path) -> None:
    result = autolabel_video(
        VideoFileSource(pacing_video, target_fps=1, max_frames=10),
        BrightBlobDetector(label="visitor"),
        ["lion"],
        tmp_path / "d",
        keep_empty_fraction=0.0,
    )
    assert result["boxes"] == 0 and result["frames_kept"] == 0
