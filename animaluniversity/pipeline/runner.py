"""One analytics loop for archived footage and live streams."""
from __future__ import annotations

import logging
import sys
from itertools import islice
from time import perf_counter
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
    batch_size: int = 1,
    profile: bool = False,
) -> Dict[str, object]:
    """Run detect -> track -> behavior over every frame of ``source``.

    ``detector`` is anything with ``detect(image) -> List[Detection]``
    (``YoloDetector``, ``OpenVocabDetector``, or a test double).
    Returns the summary report. Sinks are always closed, with status
    ``completed``, ``stopped`` (Ctrl+C) or ``failed``.
    """
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if max_frames is not None and (not isinstance(max_frames, int) or max_frames < 1):
        raise ValueError("max_frames must be a positive integer")
    if source.info.kind == "rtsp" and batch_size != 1:
        raise ValueError("Use batch_size=1 for live sources to avoid buffering latency")
    tracker = tracker if tracker is not None else MotionTracker()
    pacing = pacing if pacing is not None else PacingAnalyzer()
    report = AnalyticsAccumulator()
    info = source.info

    status = "completed"
    frames = 0
    opened = []
    iterator = None
    started = perf_counter()
    timings = {"source_s": 0.0, "inference_s": 0.0, "analytics_s": 0.0, "sink_s": 0.0}
    inference_calls = 0
    try:
        for sink in sinks:
            opened.append(sink)  # open may allocate resources before raising
            sink.open(info)
        iterator = iter(source)
        while max_frames is None or frames < max_frames:
            tick = perf_counter()
            size = batch_size if max_frames is None else min(batch_size, max_frames - frames)
            batch = list(islice(iterator, size))
            timings["source_s"] += perf_counter() - tick
            if not batch:
                break
            tick = perf_counter()
            if batch_size > 1 and callable(getattr(detector, "detect_batch", None)):
                results = list(detector.detect_batch([frame.image for frame in batch]))
                inference_calls += 1
            else:
                results = [detector.detect(frame.image) for frame in batch]
                inference_calls += len(batch)
            timings["inference_s"] += perf_counter() - tick
            if len(results) != len(batch):
                raise ValueError("Detector result count must match the number of frames")
            for frame, detections in zip(batch, results):
                tick = perf_counter()
                tracks = tracker.update(detections)
                events = pacing.update(tracks=tracks, timestamp_s=frame.timestamp_s, zone=zone)
                report.update(
                    ts=frame.timestamp_s, detections=detections, tracks=tracks, behavior_events=events
                )
                timings["analytics_s"] += perf_counter() - tick
                tick = perf_counter()
                for sink in sinks:
                    sink.write(frame, tracks, events)
                if on_frame is not None:
                    on_frame(frame, tracks, events)
                timings["sink_s"] += perf_counter() - tick
                frames += 1
                if log_every > 0 and frames % log_every == 0:
                    logger.info("frames=%d t=%.1fs detections=%d tracks=%d events=%d",
                                frames, frame.timestamp_s, len(detections), len(tracks), len(events))
    except KeyboardInterrupt:
        status = "stopped"
        logger.info("Stopped by user after %d frames.", frames)
    except Exception:
        status = "failed"
        raise
    finally:
        cleanup_errors = []
        closers = [getattr(iterator, "close", None), source.close]
        for close in closers:
            if close is None:
                continue
            try:
                close()
            except Exception as exc:  # noqa: BLE001
                cleanup_errors.append(exc)
                logger.error("Source close failed: %s", exc)
        if cleanup_errors and status == "completed":
            status = "failed"
        for sink in reversed(opened):
            try:
                sink.close(status)
            except Exception as exc:  # noqa: BLE001
                cleanup_errors.append(exc)
                logger.error("Sink close failed: %s", exc)
                if status == "completed":
                    status = "failed"
        # Cleanup must not hide an inference/open error, or report success if
        # buffered results could not be saved.
        if cleanup_errors and status in {"completed", "failed"}:
            if sys.exc_info()[0] is None:
                raise RuntimeError("Pipeline cleanup failed") from cleanup_errors[0]

    summary = report.summary(top_tracks=top_tracks)
    summary["status"] = status
    if profile:
        elapsed = perf_counter() - started
        summary["performance"] = {
            **{key: round(value, 6) for key, value in timings.items()},
            "elapsed_s": round(elapsed, 6),
            "frames_per_second": round(frames / elapsed, 3) if elapsed else 0.0,
            "inference_calls": inference_calls,
            "batch_size": batch_size,
        }
    summary["source"] = {
        "kind": info.kind,
        "name": info.name,
        "camera_id": info.camera_id,
        "camera_name": info.camera_name,
        "enclosure_id": info.enclosure_id,
    }
    return summary
