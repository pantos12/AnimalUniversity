"""Run animal analytics on archived footage or a live stream, same pipeline.

Archived clip:
    python scripts/au_run.py --video data/raw_videos/lion.mp4 --classes "lion" --db data/au.db

Milestone export (MP4 + JSON manifest; stores real-world times and camera):
    python scripts/au_run.py --milestone-manifest data/milestone_drop/rec.json --classes "lion"

Live:
    python scripts/au_run.py --rtsp-url "rtsp://..." --weights models/yolo/best.pt --emit-events
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger("au_run")


def _parse_zone(raw: Optional[str]) -> Optional[List[Tuple[float, float]]]:
    if not raw:
        return None
    points = []
    for token in raw.split(";"):
        token = token.strip()
        if token:
            x_str, y_str = token.split(",")
            points.append((float(x_str), float(y_str)))
    if len(points) < 3:
        raise ValueError("--zone must define at least 3 points: x1,y1;x2,y2;x3,y3")
    return points


def _parse_classes(raw: Optional[str]) -> List[str]:
    return [c.strip() for c in (raw or "").split(",") if c.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_argument_group("source (choose one)")
    src.add_argument("--video", help="Archived video file.")
    src.add_argument("--milestone-manifest", help="JSON manifest of a Milestone export.")
    src.add_argument("--base-dir", help="Folder holding the exported video (default: manifest folder).")
    src.add_argument("--rtsp-url", help="Live stream URL (or set env RTSP_URL).")
    src.add_argument("--target-fps", type=float, default=5.0, help="Frames analysed per second.")
    src.add_argument("--max-frames", type=int, default=None)

    det = p.add_argument_group("detector")
    det.add_argument("--weights", help="Trained YOLO checkpoint. Omit to use open-vocabulary detection.")
    det.add_argument("--classes", help='Species to find by name when no --weights is given, e.g. "tiger,red panda".')
    det.add_argument("--open-vocab-weights", default=None, help="YOLOE checkpoint (default: yoloe-26s-seg.pt).")
    det.add_argument("--conf", type=float, default=0.25)
    det.add_argument("--imgsz", type=int, default=640)
    det.add_argument("--device", default=None)

    beh = p.add_argument_group("tracking and behavior")
    beh.add_argument("--tracker", choices=["motion", "iou"], default="motion")
    beh.add_argument("--zone", help="Enclosure polygon: x1,y1;x2,y2;...")
    beh.add_argument("--pacing-window-s", type=float, default=20.0)
    beh.add_argument("--pacing-min-crossings", type=int, default=4)

    out = p.add_argument_group("output")
    out.add_argument("--db", help="SQLite database for tracks and events (appends runs).")
    out.add_argument("--report-json", help="Write the run summary here.")
    out.add_argument("--emit-events", action="store_true", help="Send behavior events to XProtect.")
    out.add_argument("--source-id", help="Camera name/address as XProtect knows it.")
    out.add_argument("--log-every", type=int, default=50)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = build_parser().parse_args(argv)

    from animaluniversity.pipeline import (
        RtspSource,
        SqliteSink,
        VideoFileSource,
        XProtectEventSink,
        milestone_export_source,
        run_pipeline,
    )
    from services.analytics.behavior import PacingAnalyzer, Zone
    from services.analytics.tracker import IoUTracker, MotionTracker

    rtsp_url = args.rtsp_url or os.environ.get("RTSP_URL", "")
    chosen = [bool(args.video), bool(args.milestone_manifest), bool(rtsp_url)]
    if sum(chosen) != 1:
        logger.error("Choose exactly one source: --video, --milestone-manifest or --rtsp-url.")
        return 2

    if args.video:
        source = VideoFileSource(args.video, target_fps=args.target_fps, max_frames=args.max_frames)
    elif args.milestone_manifest:
        source = milestone_export_source(
            args.milestone_manifest,
            base_dir=args.base_dir,
            target_fps=args.target_fps,
            max_frames=args.max_frames,
        )
    else:
        source = RtspSource(rtsp_url, target_fps=args.target_fps, max_frames=args.max_frames)

    if args.weights:
        from services.analytics.detector import YoloDetector

        detector = YoloDetector(
            weights_path=args.weights, conf_threshold=args.conf, device=args.device, imgsz=args.imgsz
        )
        logger.info("Detector: trained YOLO checkpoint %s", args.weights)
    else:
        classes = _parse_classes(args.classes)
        if not classes:
            logger.error('Give --weights (trained model) or --classes "tiger,lion" (open-vocabulary).')
            return 2
        from services.analytics.detector import OpenVocabDetector

        detector = OpenVocabDetector(
            classes=classes,
            weights_path=args.open_vocab_weights,
            conf_threshold=args.conf,
            device=args.device,
            imgsz=args.imgsz,
        )
        logger.info("Detector: open-vocabulary, classes=%s", classes)

    tracker = MotionTracker() if args.tracker == "motion" else IoUTracker(iou_threshold=0.3, max_missed=8)
    pacing = PacingAnalyzer(window_s=args.pacing_window_s, min_crossings=args.pacing_min_crossings)
    zone_points = _parse_zone(args.zone)
    zone = Zone(name="zone_1", polygon=zone_points) if zone_points else None

    sinks = []
    if args.db:
        sinks.append(SqliteSink(args.db))
    if args.emit_events:
        sinks.append(XProtectEventSink(source_id=args.source_id))

    summary = run_pipeline(
        source=source,
        detector=detector,
        tracker=tracker,
        pacing=pacing,
        sinks=sinks,
        zone=zone,
        log_every=args.log_every,
    )

    logger.info(
        "Done (%s): frames=%d, detections=%d, unique_tracks=%d, events=%s",
        summary["status"],
        int(summary["frames_processed"]),
        int(summary["detections_total"]),
        int(summary["unique_tracks"]),
        summary["behavior_events_by_name"],
    )
    if args.report_json:
        path = Path(args.report_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info("Wrote summary to %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
