from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)


def _parse_class_ids(raw: Optional[str]) -> Optional[List[int]]:
    if not raw:
        return None
    ids: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        ids.append(int(token))
    return ids if ids else None


def _parse_zone(raw: Optional[str]) -> Optional[List[Tuple[float, float]]]:
    if not raw:
        return None
    points = []
    for token in raw.split(";"):
        token = token.strip()
        if not token:
            continue
        x_str, y_str = token.split(",")
        points.append((float(x_str), float(y_str)))
    if len(points) < 3:
        raise ValueError("--zone must define at least 3 points: x1,y1;x2,y2;x3,y3")
    return points


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run live RTSP analytics pipeline.")
    parser.add_argument("--rtsp-url", default=None, help="RTSP URL (default: env RTSP_URL).")
    parser.add_argument(
        "--model-choice",
        default="ena24",
        choices=["ena24", "sam2", "sam3", "custom"],
        help="Model strategy for live runtime.",
    )
    parser.add_argument("--weights", default=None, help="YOLO checkpoint path.")
    parser.add_argument("--device", default=None, help="Device string (cuda/cpu/mps/auto).")
    parser.add_argument("--target-fps", type=float, default=5.0, help="Ingest target FPS.")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--class-ids", default=None, help="Comma-separated class IDs filter.")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames.")
    parser.add_argument(
        "--zone",
        default=None,
        help="Optional behavior zone polygon: x1,y1;x2,y2;...",
    )
    parser.add_argument("--emit-events", action="store_true", help="Send pacing events to XProtect.")
    parser.add_argument("--source-id", default="CAMERA_1", help="SourceID for EventBridge payload.")
    parser.add_argument(
        "--event-type",
        default="AnimalBehavior.Pacing",
        help="EventType for pacing payloads.",
    )
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = build_parser()
    args = parser.parse_args()

    try:
        from services.analytics.behavior import PacingAnalyzer, Zone
        from services.analytics.detector import YoloDetector, resolve_runtime_model
        from services.analytics.tracker import IoUTracker
        from services.eventbridge.event_sender import build_analytics_event_xml, send_event
        from services.ingest.ingest import IngestConfig, frame_stream
    except Exception as exc:  # noqa: BLE001
        logger.error("Import error: %s", exc)
        logger.error("Install dependencies with: bash scripts/setup_env.sh")
        return 2

    rtsp_url = args.rtsp_url or os.environ.get("RTSP_URL", "")
    if not rtsp_url:
        logger.error("Missing RTSP URL. Set --rtsp-url or RTSP_URL env var.")
        return 2

    zone_points = _parse_zone(args.zone)
    zone = Zone(name="zone_1", polygon=zone_points) if zone_points else None
    runtime_model = resolve_runtime_model(
        model_choice=args.model_choice,
        weights=args.weights,
        models_dir=ROOT / "models",
    )
    logger.info("Model choice: %s", runtime_model.model_choice)
    logger.info("Using weights: %s", runtime_model.weights_path)
    logger.info("Plan note: %s", runtime_model.plan_note)

    detector = YoloDetector(
        weights_path=runtime_model.weights_path,
        conf_threshold=args.conf,
        device=args.device,
        class_ids=_parse_class_ids(args.class_ids),
        imgsz=args.imgsz,
    )
    tracker = IoUTracker(iou_threshold=0.3, max_missed=8)
    pacing = PacingAnalyzer(window_s=20.0, min_crossings=4, min_span_px=80.0, cooldown_s=10.0)

    ingest_cfg = IngestConfig(rtsp_url=rtsp_url, target_fps=args.target_fps)
    frame_count = 0
    event_count = 0

    for ts, frame in frame_stream(ingest_cfg):
        frame_count += 1
        detections = detector.detect(frame)
        tracks = tracker.update(detections)
        behavior_events = pacing.update(tracks=tracks, timestamp_s=ts, zone=zone)

        if frame_count % 20 == 0:
            logger.info(
                "frames=%d detections=%d tracks=%d pacing_events=%d",
                frame_count,
                len(detections),
                len(tracks),
                len(behavior_events),
            )

        if args.emit_events:
            for event in behavior_events:
                xml_payload = build_analytics_event_xml(
                    source_id=args.source_id,
                    event_type=args.event_type,
                    properties={
                        "track_id": str(event.track_id),
                        "score": f"{event.score:.3f}",
                        **event.details,
                    },
                )
                try:
                    send_event(xml_payload)
                    event_count += 1
                except Exception as exc:  # noqa: BLE001
                    logger.error("Event send failed: %s", exc)

        if args.max_frames is not None and frame_count >= args.max_frames:
            break

    logger.info("Done. Processed frames=%d, emitted_events=%d", frame_count, event_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
