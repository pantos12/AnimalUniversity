from __future__ import annotations

import hashlib
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from animaluniversity.core.video import get_video_metadata
from animaluniversity.utils.paths import get_data_dir, get_models_dir
from services.analytics.detector import YoloDetector
from services.analytics.tracker import IoUTracker


def _list_videos(raw_dir: Path) -> List[Path]:
    exts = {".mp4", ".avi", ".mkv", ".mov"}
    return sorted([p for p in raw_dir.glob("*") if p.suffix.lower() in exts])


def _list_weights() -> List[Path]:
    candidates = []
    roots = [
        get_models_dir(),
        get_data_dir() / "runs",
    ]
    for root in roots:
        if not root.exists():
            continue
        for pt in root.rglob("*.pt"):
            candidates.append(pt)
    # Prefer trained "best.pt" first.
    candidates.sort(key=lambda p: ("best.pt" not in p.name, str(p)))
    return candidates


def _color_for_key(key: str) -> Tuple[int, int, int]:
    digest = hashlib.md5(key.encode("utf-8")).digest()
    return (int(digest[0]), int(digest[1]), int(digest[2]))


def _draw_overlay(
    frame: np.ndarray,
    tracks: List[object],
    progress_ratio: float,
    frame_time_s: float,
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    # Top translucent title bar.
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, 88), (18, 20, 26), -1)
    out = cv2.addWeighted(overlay, 0.45, out, 0.55, 0)

    for tr in tracks:
        x1, y1, x2, y2 = [int(v) for v in tr.bbox_xyxy]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        rx = max(12, int((x2 - x1) / 2))
        ry = max(8, int((y2 - y1) / 2))
        color = _color_for_key(f"{tr.track_id}:{tr.label}")
        cv2.ellipse(out, (cx, cy), (rx, ry), 0, 0, 360, color, 2)

        text = f"{tr.label} {float(tr.score):.2f}  id:{tr.track_id}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        tx = max(0, min(w - tw - 8, x1))
        ty = max(th + 6, y1 - 6)
        cv2.rectangle(out, (tx - 4, ty - th - 4), (tx + tw + 4, ty + baseline + 2), color, -1)
        cv2.putText(
            out,
            text,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

    # Circular progress indicator.
    center = (w // 2, h // 2)
    radius = max(24, min(w, h) // 20)
    cv2.circle(out, center, radius, (255, 255, 255), 3)
    arc_end = int(360 * max(0.0, min(1.0, progress_ratio)))
    cv2.ellipse(out, center, (radius, radius), -90, 0, arc_end, (70, 220, 255), 5)

    # Timestamp badge.
    stamp = f"{frame_time_s:.1f}s"
    (tw, th), baseline = cv2.getTextSize(stamp, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
    pad = 10
    x1 = 14
    y1 = h - 16 - th - baseline - pad
    x2 = x1 + tw + pad * 2
    y2 = y1 + th + baseline + pad * 2
    cv2.rectangle(out, (x1, y1), (x2, y2), (22, 22, 22), -1)
    cv2.putText(
        out,
        stamp,
        (x1 + pad, y2 - pad - baseline),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )

    return out


def _render_demo_clip(
    video_path: Path,
    weights_path: Path,
    out_path: Path,
    infer_fps: int,
    max_frames: int,
    conf: float,
) -> Dict[str, object]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, max(1.0, float(infer_fps)), (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {out_path}")

    detector = YoloDetector(weights_path=weights_path, conf_threshold=conf, device="cpu", imgsz=640)
    tracker = IoUTracker(iou_threshold=0.3, max_missed=8)

    frame_step = max(1, int(round(native_fps / max(1, infer_fps))))
    frame_idx = -1
    processed = 0
    detections_total = 0
    conf_sum = 0.0
    detections_by_label: Counter[str] = Counter()
    tracks_by_label: Dict[str, set[int]] = defaultdict(set)

    while processed < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % frame_step != 0:
            continue

        processed += 1
        dets = detector.detect(frame)
        tracks = tracker.update(dets)

        detections_total += len(dets)
        for det in dets:
            label = str(det.label)
            detections_by_label[label] += 1
            conf_sum += float(det.confidence)
        for tr in tracks:
            tracks_by_label[str(tr.label)].add(int(tr.track_id))

        ts = frame_idx / native_fps
        rendered = _draw_overlay(
            frame=frame,
            tracks=tracks,
            progress_ratio=(processed / float(max_frames)),
            frame_time_s=ts,
        )
        writer.write(rendered)

    cap.release()
    writer.release()

    avg_conf = (conf_sum / detections_total) if detections_total else 0.0
    summary = {
        "video_path": str(video_path),
        "weights_path": str(weights_path),
        "frames_processed": processed,
        "detections_total": detections_total,
        "avg_confidence": round(avg_conf, 4),
        "detections_by_label": dict(detections_by_label),
        "unique_tracks_by_label": {k: len(v) for k, v in tracks_by_label.items()},
        "output_video_path": str(out_path),
    }
    return summary


def _render_header() -> None:
    st.markdown(
        """
        <style>
        .au-wrap {max-width: 1040px; margin: 0 auto; padding: 8px 12px 20px 12px;}
        .au-title {font-size: 3rem; line-height: 1.12; font-weight: 700; color: #161a22; margin: 0 0 10px 0;}
        .au-subtitle {font-size: 1.8rem; line-height: 1.25; font-weight: 500; color: #121821; margin: 0 0 16px 0;}
        .au-bullets {font-size: 1.15rem; line-height: 1.55; color: #1f2733; margin: 0 0 12px 0;}
        .au-video-card {border-radius: 24px; border: 1px solid #e4e8ef; overflow: hidden; box-shadow: 0 12px 32px rgba(0,0,0,.06);}
        </style>
        <div class="au-wrap">
          <div class="au-title">taking animal detection to the next level</div>
          <div class="au-subtitle">real-time; species-level classes; minimal manual annotations</div>
          <div class="au-bullets">- auto-annotated dataset bootstrap with SAM3/SAM2-ready workflow</div>
          <div class="au-bullets">- fine-tuned YOLO species model</div>
          <div class="au-bullets">- track and count animals with stable track IDs</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    st.set_page_config(page_title="AnimalUniversity - Realtime Species Demo", layout="wide")
    _render_header()

    data_dir = get_data_dir()
    raw_dir = data_dir / "raw_videos"
    run_dir = data_dir / "runs" / "demo"
    raw_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    with st.sidebar:
        st.header("Demo Controls")
        upload = st.file_uploader("Upload video", type=["mp4", "avi", "mkv", "mov"])
        existing = _list_videos(raw_dir)
        selected_video = st.selectbox("Or select existing", [""] + [p.name for p in existing])

        weights = _list_weights()
        weight_options = [str(p) for p in weights]
        if not weight_options:
            st.error("No .pt model found under models/ or data/runs/.")
            return
        default_idx = 0
        for i, path in enumerate(weight_options):
            if "species" in path and "best.pt" in path:
                default_idx = i
                break
        selected_weight = st.selectbox("Model weights", weight_options, index=default_idx)
        infer_fps = st.slider("Inference FPS", min_value=1, max_value=10, value=3, step=1)
        max_frames = st.slider("Max frames", min_value=60, max_value=1200, value=240, step=30)
        conf = st.slider("Confidence", min_value=0.05, max_value=0.9, value=0.25, step=0.05)
        run_btn = st.button("Generate Realtime Demo", type="primary")

    video_path: Path | None = None
    if upload is not None:
        video_path = raw_dir / upload.name
        if not video_path.exists():
            video_path.write_bytes(upload.read())
        st.success(f"Uploaded: {video_path}")
    elif selected_video:
        video_path = raw_dir / selected_video

    if video_path and video_path.exists():
        st.subheader("Video Metadata")
        st.json(get_video_metadata(video_path))

    if run_btn:
        if not video_path or not video_path.exists():
            st.error("Select or upload a video first.")
            return
        weights_path = Path(selected_weight)
        if not weights_path.exists():
            st.error(f"Weights not found: {weights_path}")
            return

        out_path = run_dir / f"{video_path.stem}_{weights_path.stem}_demo.mp4"
        with st.spinner("Running species detection + tracking + visual overlay..."):
            summary = _render_demo_clip(
                video_path=video_path,
                weights_path=weights_path,
                out_path=out_path,
                infer_fps=int(infer_fps),
                max_frames=int(max_frames),
                conf=float(conf),
            )

        st.success("Demo clip generated.")
        st.markdown('<div class="au-video-card">', unsafe_allow_html=True)
        st.video(str(out_path))
        st.markdown("</div>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Frames Processed", int(summary["frames_processed"]))
        c2.metric("Detections Total", int(summary["detections_total"]))
        c3.metric("Avg Confidence", float(summary["avg_confidence"]))

        st.subheader("Species Counts")
        st.json(summary["detections_by_label"])
        st.subheader("Unique Tracks By Species")
        st.json(summary["unique_tracks_by_label"])
        st.subheader("Run Summary")
        st.code(json.dumps(summary, indent=2), language="json")


if __name__ == "__main__":
    main()
