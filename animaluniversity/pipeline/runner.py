"""One analytics loop for archived footage and live streams."""
from __future__ import annotations

import logging
from typing import Callable, Dict, Optional, Sequence

from animaluniversity.pipeline.sinks import Sink
from animaluniversity.pipeline.sources import Frame, FrameSource
from services.analytics.behavior import PacingAnalyzer, Zone
from services.analytics.reporting import AnalyticsAccumulator
from services.analytics.tracker import MotionTracker, Tracker

logger = logging.getLogger(__name__)


def run_pipeline(
    source: FrameSource,
    detector,
    tracker: Optional[Tracker] = None,
    pacing: Optional[PacingAnalyzer] = None,
    sinks: Sequence[Sink] = (),
    zone: Optional[Zone] = None,
    max_frames: Optional[int] = None,
    log_every: int = 0,
    on_frame: Optional[Callable[[Frame, list, list], None]] = None,
    top_tracks: int = 20,
) -> Dict[str, object]:
    """Run detect -> track -> behavior over every frame of ``source``.

    ``detector`` is anything with ``detect(image) -> List[Detection]``
    (``YoloDetector``, ``OpenVocabDetector``, or a test double).
    Returns the summary report. Sinks are always closed, with status
    ``completed``, ``stopped`` (Ctrl+C) or ``failed``.
    """
    tracker = tracker or MotionTracker()
    pacing = pacing or PacingAnalyzer()
    report = AnalyticsAccumulator()
    info = source.info

    for sink in sinks:
        sink.open(info)

    status = "completed"
    frames = 0
    try:
        for frame in source:
            detections = detector.detect(frame.image)
            tracks = tracker.update(detections)
            events = pacing.update(tracks=tracks, timestamp_s=frame.timestamp_s, zone=zone)
            report.update(
                ts=frame.timestamp_s, detections=detections, tracks=tracks, behavior_events=events
            )
            for sink in sinks:
                sink.write(frame, tracks, events)
            if on_frame is not None:
                on_frame(frame, tracks, events)
            frames += 1
            if log_every > 0 and frames % log_every == 0:
                logger.info(
                    "frames=%d t=%.1fs detections=%d tracks=%d events=%d",
                    frames, frame.timestamp_s, len(detections), len(tracks), len(events),
                )
            if max_frames is not None and frames >= max_frames:
                break
    except KeyboardInterrupt:
        status = "stopped"
        logger.info("Stopped by user after %d frames.", frames)
    except Exception:
        status = "failed"
        raise
    finally:
        source.close()
        for sink in sinks:
            try:
                sink.close(status)
            except Exception as exc:  # noqa: BLE001
                logger.error("Sink close failed: %s", exc)

    summary = report.summary(top_tracks=top_tracks)
    summary["status"] = status
    summary["source"] = {
        "kind": info.kind,
        "name": info.name,
        "camera_id": info.camera_id,
        "camera_name": info.camera_name,
        "enclosure_id": info.enclosure_id,
    }
    return summary
