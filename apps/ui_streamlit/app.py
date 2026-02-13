from __future__ import annotations

import hashlib
import json
import logging
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from animaluniversity.core.video import get_video_metadata
from animaluniversity.utils.paths import get_data_dir, get_models_dir
from services.analytics.detector import YoloDetector
from services.analytics.tracker import Detection, IoUTracker, Track


@dataclass(frozen=True)
class DemoOptions:
    profile_name: str
    allow_labels: Optional[Set[str]]
    relabel_map: Dict[str, str]
    min_area_ratio: float
    single_target_mode: bool
    lock_persistence_frames: int
    tracker_iou: float
    tracker_max_missed: int
    smoothing_alpha: float
    imgsz: int


@dataclass
class RenderTrack:
    track_id: int
    bbox_xyxy: Tuple[float, float, float, float]
    label: str
    score: float
    predicted: bool = False


@dataclass
class TargetLockState:
    track_id: int
    bbox_xyxy: Tuple[float, float, float, float]
    label: str
    score: float
    velocity_xy: Tuple[float, float] = (0.0, 0.0)
    missed_frames: int = 0


@dataclass
class SessionStats:
    frames_processed: int = 0
    frames_with_detections: int = 0
    frames_with_target_lock: int = 0
    detections_total_raw: int = 0
    detections_total_kept: int = 0
    confidence_sum: float = 0.0
    suppressed_small_area: int = 0
    suppressed_focus_filter: int = 0
    detections_by_label: Counter[str] = field(default_factory=Counter)
    unique_tracks_by_label: Dict[str, Set[int]] = field(default_factory=lambda: defaultdict(set))


def _list_videos(raw_dir: Path) -> List[Path]:
    exts = {".mp4", ".avi", ".mkv", ".mov", ".mpeg"}
    return sorted([p for p in raw_dir.glob("*") if p.suffix.lower() in exts])


def _list_weights() -> List[Path]:
    candidates: List[Path] = []
    for root in [get_models_dir(), get_data_dir() / "runs"]:
        if not root.exists():
            continue
        candidates.extend(root.rglob("*.pt"))
    candidates.sort(key=lambda p: ("best.pt" not in p.name, str(p)))
    return candidates


def _save_upload(upload, raw_dir: Path) -> Optional[Path]:
    if upload is None:
        return None
    dst = raw_dir / upload.name
    sig = f"{upload.name}:{upload.size}"
    if st.session_state.get("last_upload_sig") == sig and dst.exists():
        return dst

    upload.seek(0)
    with dst.open("wb") as f:
        while True:
            chunk = upload.read(16 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    st.session_state["last_upload_sig"] = sig
    upload.seek(0)
    return dst


def _color_for_key(key: str) -> Tuple[int, int, int]:
    digest = hashlib.md5(key.encode("utf-8")).digest()
    return (int(digest[0]), int(digest[1]), int(digest[2]))


def _clip_bbox(bbox: Tuple[float, float, float, float], width: int, height: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    x1 = float(max(0, min(width - 1, int(round(x1)))))
    y1 = float(max(0, min(height - 1, int(round(y1)))))
    x2 = float(max(0, min(width - 1, int(round(x2)))))
    y2 = float(max(0, min(height - 1, int(round(y2)))))
    if x2 <= x1:
        x2 = min(width - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(height - 1, y1 + 1)
    return (x1, y1, x2, y2)


def _bbox_iou(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    if denom <= 0:
        return 0.0
    return inter / denom


def _center_xy(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _predict_bbox(
    bbox: Tuple[float, float, float, float],
    velocity: Tuple[float, float],
    width: int,
    height: int,
) -> Tuple[float, float, float, float]:
    dx, dy = velocity
    x1, y1, x2, y2 = bbox
    return _clip_bbox((x1 + dx, y1 + dy, x2 + dx, y2 + dy), width, height)


def _smooth_bbox(
    track_id: int,
    bbox: Tuple[float, float, float, float],
    cache: Dict[int, Tuple[float, float, float, float]],
    alpha: float,
) -> Tuple[float, float, float, float]:
    previous = cache.get(track_id)
    if previous is None:
        cache[track_id] = bbox
        return bbox
    smoothed = tuple((alpha * n) + ((1.0 - alpha) * p) for n, p in zip(bbox, previous))
    cache[track_id] = smoothed
    return smoothed


def _postprocess_detections(
    detections: List[Detection],
    frame_width: int,
    frame_height: int,
    options: DemoOptions,
    stats: SessionStats,
) -> List[Detection]:
    processed: List[Detection] = []
    frame_area = float(frame_width * frame_height)
    for det in detections:
        stats.detections_total_raw += 1
        src_label = str(det.label).lower().strip()
        x1, y1, x2, y2 = det.bbox_xyxy
        area = max(0.0, (x2 - x1) * (y2 - y1))
        area_ratio = area / max(1.0, frame_area)
        if area_ratio < options.min_area_ratio:
            stats.suppressed_small_area += 1
            continue
        if options.allow_labels is not None and src_label not in options.allow_labels:
            stats.suppressed_focus_filter += 1
            continue

        mapped_label = options.relabel_map.get(src_label, src_label)
        clipped = _clip_bbox(det.bbox_xyxy, frame_width, frame_height)
        processed.append(
            Detection(
                bbox_xyxy=clipped,
                confidence=float(det.confidence),
                label=mapped_label,
            )
        )
        stats.detections_total_kept += 1
        stats.confidence_sum += float(det.confidence)
        stats.detections_by_label[mapped_label] += 1
    return processed


def _to_render_track(track: Track, label_override: Optional[str] = None) -> RenderTrack:
    return RenderTrack(
        track_id=int(track.track_id),
        bbox_xyxy=track.bbox_xyxy,
        label=label_override or str(track.label),
        score=float(track.score),
        predicted=False,
    )


def _pick_target_track(
    tracks: List[Track],
    state: Optional[TargetLockState],
    frame_width: int,
    frame_height: int,
) -> Optional[Track]:
    if not tracks:
        return None
    if state is None:
        return max(tracks, key=lambda t: t.score)

    predicted_bbox = _predict_bbox(state.bbox_xyxy, state.velocity_xy, frame_width, frame_height)
    scored: List[Tuple[float, Track]] = []
    for tr in tracks:
        iou = _bbox_iou(tr.bbox_xyxy, predicted_bbox)
        score = (0.62 * iou) + (0.38 * float(tr.score))
        if tr.track_id == state.track_id:
            score += 0.18
        scored.append((score, tr))
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_track = scored[0]
    if best_score < 0.12:
        return None
    return best_track


def _update_target_lock(
    tracks: List[Track],
    state: Optional[TargetLockState],
    label_votes: Dict[int, Counter[str]],
    options: DemoOptions,
    frame_width: int,
    frame_height: int,
) -> Tuple[Optional[TargetLockState], List[RenderTrack], bool]:
    selected = _pick_target_track(
        tracks=tracks,
        state=state,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    if selected is not None:
        tid = int(selected.track_id)
        label_votes[tid][str(selected.label)] += 1
        stable_label = label_votes[tid].most_common(1)[0][0]
        if state is None:
            updated_state = TargetLockState(
                track_id=tid,
                bbox_xyxy=selected.bbox_xyxy,
                label=stable_label,
                score=float(selected.score),
                velocity_xy=(0.0, 0.0),
                missed_frames=0,
            )
        else:
            prev_cx, prev_cy = _center_xy(state.bbox_xyxy)
            cur_cx, cur_cy = _center_xy(selected.bbox_xyxy)
            inst_vx = cur_cx - prev_cx
            inst_vy = cur_cy - prev_cy
            smooth_vx = (0.75 * state.velocity_xy[0]) + (0.25 * inst_vx)
            smooth_vy = (0.75 * state.velocity_xy[1]) + (0.25 * inst_vy)
            updated_state = TargetLockState(
                track_id=tid,
                bbox_xyxy=selected.bbox_xyxy,
                label=stable_label,
                score=float(selected.score),
                velocity_xy=(smooth_vx, smooth_vy),
                missed_frames=0,
            )
        return updated_state, [_to_render_track(selected, label_override=updated_state.label)], True

    if state is None:
        return None, [], False
    if state.missed_frames >= options.lock_persistence_frames:
        return None, [], False

    predicted_bbox = _predict_bbox(state.bbox_xyxy, state.velocity_xy, frame_width, frame_height)
    predicted_state = TargetLockState(
        track_id=state.track_id,
        bbox_xyxy=predicted_bbox,
        label=state.label,
        score=max(0.05, state.score * 0.97),
        velocity_xy=state.velocity_xy,
        missed_frames=state.missed_frames + 1,
    )
    predicted_track = RenderTrack(
        track_id=predicted_state.track_id,
        bbox_xyxy=predicted_state.bbox_xyxy,
        label=predicted_state.label,
        score=predicted_state.score,
        predicted=True,
    )
    return predicted_state, [predicted_track], True


def _draw_overlay(
    frame: np.ndarray,
    tracks: List[RenderTrack],
    track_history: Dict[int, Deque[Tuple[int, int]]],
    progress_ratio: float,
    frame_time_s: float,
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    # Glass header strip.
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, 92), (22, 25, 31), -1)
    out = cv2.addWeighted(overlay, 0.38, out, 0.62, 0)

    for tr in tracks:
        x1, y1, x2, y2 = [int(v) for v in tr.bbox_xyxy]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        rx = max(14, int((x2 - x1) / 2))
        ry = max(10, int((y2 - y1) / 2))
        color = _color_for_key(f"{tr.track_id}:{tr.label}")

        history = track_history.setdefault(tr.track_id, deque(maxlen=40))
        history.append((cx, cy))
        points = list(history)
        if len(points) >= 2:
            for i in range(1, len(points)):
                alpha = i / max(1, len(points) - 1)
                trail_color = tuple(int(alpha * c) for c in color)
                cv2.line(out, points[i - 1], points[i], trail_color, 2, cv2.LINE_AA)

        line_thickness = 2 if tr.predicted else 3
        cv2.ellipse(out, (cx, cy), (rx, ry), 0, 0, 360, color, line_thickness, cv2.LINE_AA)

        state_tag = "pred" if tr.predicted else "live"
        text = f"{tr.label}  {float(tr.score):.2f}  id:{tr.track_id}  {state_tag}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.54, 1)
        tx = max(8, min(w - tw - 12, x1))
        ty = max(th + 8, y1 - 8)

        pill = out.copy()
        cv2.rectangle(pill, (tx - 6, ty - th - 8), (tx + tw + 8, ty + baseline + 4), color, -1)
        out = cv2.addWeighted(pill, 0.68, out, 0.32, 0)
        cv2.putText(
            out,
            text,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (18, 18, 20),
            1,
            cv2.LINE_AA,
        )

    # Progress ring.
    center = (w // 2, h // 2)
    radius = max(28, min(w, h) // 18)
    cv2.circle(out, center, radius, (240, 240, 240), 2, cv2.LINE_AA)
    arc_end = int(360 * max(0.0, min(1.0, progress_ratio)))
    cv2.ellipse(out, center, (radius, radius), -90, 0, arc_end, (54, 168, 255), 6, cv2.LINE_AA)

    # Time badge.
    stamp = f"{frame_time_s:.1f}s"
    (tw, th), baseline = cv2.getTextSize(stamp, cv2.FONT_HERSHEY_SIMPLEX, 0.64, 2)
    pad = 10
    bx1 = 14
    by1 = h - 14 - th - baseline - (pad * 2)
    bx2 = bx1 + tw + (pad * 2)
    by2 = by1 + th + baseline + (pad * 2)
    badge = out.copy()
    cv2.rectangle(badge, (bx1, by1), (bx2, by2), (16, 17, 20), -1)
    out = cv2.addWeighted(badge, 0.78, out, 0.22, 0)
    cv2.putText(
        out,
        stamp,
        (bx1 + pad, by2 - pad - baseline),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.64,
        (246, 246, 248),
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
    options: DemoOptions,
) -> Dict[str, object]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    frame_step = max(1, int(round(native_fps / max(1, infer_fps))))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, max(1.0, float(infer_fps)), (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {out_path}")

    detector = YoloDetector(
        weights_path=weights_path,
        conf_threshold=conf,
        device="cpu",
        imgsz=options.imgsz,
    )
    tracker = IoUTracker(iou_threshold=options.tracker_iou, max_missed=options.tracker_max_missed)
    bbox_smoothing_cache: Dict[int, Tuple[float, float, float, float]] = {}
    track_history: Dict[int, Deque[Tuple[int, int]]] = {}
    label_votes: Dict[int, Counter[str]] = defaultdict(Counter)
    target_lock_state: Optional[TargetLockState] = None
    stats = SessionStats()

    frame_idx = -1
    while stats.frames_processed < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % frame_step != 0:
            continue
        stats.frames_processed += 1

        raw_dets = detector.detect(frame)
        dets = _postprocess_detections(
            detections=raw_dets,
            frame_width=width,
            frame_height=height,
            options=options,
            stats=stats,
        )
        if dets:
            stats.frames_with_detections += 1

        tracks = tracker.update(dets)
        render_tracks: List[RenderTrack]
        if options.single_target_mode:
            target_lock_state, render_tracks, has_lock = _update_target_lock(
                tracks=tracks,
                state=target_lock_state,
                label_votes=label_votes,
                options=options,
                frame_width=width,
                frame_height=height,
            )
            if has_lock:
                stats.frames_with_target_lock += 1
        else:
            render_tracks = []
            for tr in tracks:
                tid = int(tr.track_id)
                label_votes[tid][str(tr.label)] += 1
                stable = label_votes[tid].most_common(1)[0][0]
                render_tracks.append(_to_render_track(tr, label_override=stable))

        for tr in render_tracks:
            smoothed = _smooth_bbox(
                track_id=tr.track_id,
                bbox=tr.bbox_xyxy,
                cache=bbox_smoothing_cache,
                alpha=options.smoothing_alpha,
            )
            tr.bbox_xyxy = smoothed
            stats.unique_tracks_by_label[tr.label].add(tr.track_id)

        ts = frame_idx / native_fps
        rendered = _draw_overlay(
            frame=frame,
            tracks=render_tracks,
            track_history=track_history,
            progress_ratio=(stats.frames_processed / float(max_frames)),
            frame_time_s=ts,
        )
        writer.write(rendered)

    cap.release()
    writer.release()

    avg_conf = stats.confidence_sum / stats.detections_total_kept if stats.detections_total_kept else 0.0
    lock_ratio = (
        stats.frames_with_target_lock / stats.frames_processed if stats.frames_processed else 0.0
    )
    summary = {
        "video_path": str(video_path),
        "weights_path": str(weights_path),
        "profile": options.profile_name,
        "frames_processed": stats.frames_processed,
        "frames_with_detections": stats.frames_with_detections,
        "frames_with_target_lock": stats.frames_with_target_lock,
        "target_lock_ratio": round(lock_ratio, 4),
        "detections_total_raw": stats.detections_total_raw,
        "detections_total_kept": stats.detections_total_kept,
        "avg_confidence": round(avg_conf, 4),
        "detections_by_label": dict(stats.detections_by_label),
        "unique_tracks_by_label": {k: len(v) for k, v in stats.unique_tracks_by_label.items()},
        "suppressed": {
            "small_area": stats.suppressed_small_area,
            "focus_filter": stats.suppressed_focus_filter,
        },
        "output_video_path": str(out_path),
    }
    return summary


def _render_header() -> None:
    st.markdown(
        """
        <style>
        .stApp {
          font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Helvetica Neue", Helvetica, sans-serif;
          background:
            radial-gradient(1200px 700px at 90% -10%, #eff6ff 0%, transparent 55%),
            radial-gradient(900px 500px at -10% 20%, #f5f3ff 0%, transparent 45%),
            linear-gradient(180deg, #f7f8fb 0%, #f4f5f8 100%);
        }
        section[data-testid="stSidebar"] {
          background: linear-gradient(180deg, rgba(251,252,255,.94) 0%, rgba(246,248,252,.9) 100%);
          border-right: 1px solid rgba(38,54,84,.08);
        }
        .au-wrap {max-width: 1160px; margin: 0 auto; padding: 6px 8px 20px 8px;}
        .au-eyebrow {font-size: .88rem; letter-spacing: .12em; text-transform: uppercase; color: #56739e; font-weight: 700;}
        .au-title {font-size: 3.2rem; line-height: 1.05; font-weight: 700; color: #0f1728; margin: 8px 0 12px 0;}
        .au-subtitle {font-size: 1.25rem; line-height: 1.45; max-width: 860px; color: #2c3648; margin: 0;}
        .au-chip {display: inline-block; margin-top: 14px; padding: 8px 12px; border-radius: 999px; font-size: .88rem; font-weight: 600; color: #173b6c; background: rgba(123, 196, 255, .22); border: 1px solid rgba(91, 149, 231, .35);}
        .au-video-card {border-radius: 22px; border: 1px solid rgba(20,32,58,.12); overflow: hidden; box-shadow: 0 18px 40px rgba(24,32,58,.1);}
        .au-section {font-size: 1.35rem; font-weight: 700; color: #111827; margin-top: 1.1rem;}
        .au-note {font-size: .92rem; color: #5f6f86;}
        </style>
        <div class="au-wrap">
          <div class="au-eyebrow">AnimalUniversity Vision Engine</div>
          <div class="au-title">Precision Animal Tracking, Built for Operators</div>
          <p class="au-subtitle">
            Detect, lock, and follow the correct animal over long footage with cleaner overlays,
            stronger tracking persistence, and production-ready summary analytics.
          </p>
          <div class="au-chip">Panda Focus profile now includes label stabilization and target lock</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    st.set_page_config(page_title="AnimalUniversity - Vision Demo", layout="wide")
    _render_header()

    data_dir = get_data_dir()
    raw_dir = data_dir / "raw_videos"
    run_dir = data_dir / "runs" / "demo"
    raw_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    with st.sidebar:
        st.header("Demo Controls")
        upload = st.file_uploader("Upload video", type=["mp4", "avi", "mkv", "mov", "mpeg"])
        existing = _list_videos(raw_dir)
        selected_video = st.selectbox("Or select existing", [""] + [p.name for p in existing])

        weights = _list_weights()
        if not weights:
            st.error("No .pt model found under models/ or data/runs/.")
            return
        weight_options = [str(p) for p in weights]
        default_idx = 0
        for i, path in enumerate(weight_options):
            if "species" in path and "best.pt" in path:
                default_idx = i
                break
        selected_weight = st.selectbox("Model weights", weight_options, index=default_idx)

        profile_name = st.selectbox(
            "Detection profile",
            ["Panda Focus", "General Species"],
            index=0,
            help=(
                "Panda Focus keeps a single locked subject and remaps supported source labels "
                "to a panda output label."
            ),
        )
        bird_fallback = st.checkbox(
            "Use bird fallback for panda mapping",
            value=True,
            help="Useful when the model confuses panda as bird in some frames.",
        )
        infer_fps = st.slider("Inference FPS", min_value=1, max_value=12, value=4, step=1)
        max_frames = st.slider("Max frames", min_value=60, max_value=3600, value=480, step=60)
        conf_default = 0.1 if profile_name == "Panda Focus" else 0.25
        conf = st.slider("Confidence", min_value=0.05, max_value=0.9, value=conf_default, step=0.05)
        min_area_percent = st.slider(
            "Minimum object area (%)",
            min_value=0.01,
            max_value=5.0,
            value=0.20 if profile_name == "Panda Focus" else 0.05,
            step=0.01,
            help="Filters tiny noisy detections.",
        )
        lock_persistence = st.slider(
            "Target lock persistence (frames)",
            min_value=0,
            max_value=240,
            value=36,
            step=6,
        )
        imgsz = st.select_slider("Model image size", options=[640, 736, 832, 960, 1024], value=960)
        run_btn = st.button("Generate Premium Tracking Demo", type="primary")

    video_path: Optional[Path] = _save_upload(upload, raw_dir)
    if video_path is not None:
        st.success(f"Uploaded: {video_path.name}")
    elif selected_video:
        video_path = raw_dir / selected_video

    if video_path and video_path.exists():
        with st.expander("Video Metadata", expanded=False):
            st.json(get_video_metadata(video_path))

    if run_btn:
        if not video_path or not video_path.exists():
            st.error("Select or upload a video first.")
            return
        weights_path = Path(selected_weight)
        if not weights_path.exists():
            st.error(f"Weights not found: {weights_path}")
            return

        profile = profile_name.lower().replace(" ", "_")
        allow_labels: Optional[Set[str]] = None
        relabel_map: Dict[str, str] = {}
        single_target_mode = False
        tracker_max_missed = 28
        tracker_iou = 0.25

        if profile == "panda_focus":
            allow_labels = {"bear"}
            relabel_map = {"bear": "panda"}
            if bird_fallback:
                allow_labels = {"bear", "bird"}
                relabel_map["bird"] = "panda"
            single_target_mode = True
            tracker_max_missed = max(40, lock_persistence + 8)
            tracker_iou = 0.18

        options = DemoOptions(
            profile_name=profile_name,
            allow_labels=allow_labels,
            relabel_map=relabel_map,
            min_area_ratio=float(min_area_percent) / 100.0,
            single_target_mode=single_target_mode,
            lock_persistence_frames=int(lock_persistence),
            tracker_iou=float(tracker_iou),
            tracker_max_missed=int(tracker_max_missed),
            smoothing_alpha=0.56,
            imgsz=int(imgsz),
        )

        out_path = run_dir / f"{video_path.stem}_{weights_path.stem}_{profile}.mp4"
        with st.spinner("Running detection, stabilization, and visual polish..."):
            summary = _render_demo_clip(
                video_path=video_path,
                weights_path=weights_path,
                out_path=out_path,
                infer_fps=int(infer_fps),
                max_frames=int(max_frames),
                conf=float(conf),
                options=options,
            )

        st.success("Demo ready.")
        st.markdown('<div class="au-video-card">', unsafe_allow_html=True)
        st.video(str(out_path))
        st.markdown("</div>", unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Frames Processed", int(summary["frames_processed"]))
        c2.metric("Detections Kept", int(summary["detections_total_kept"]))
        c3.metric("Avg Confidence", float(summary["avg_confidence"]))
        c4.metric("Target Lock Ratio", f'{100.0 * float(summary["target_lock_ratio"]):.1f}%')

        st.markdown('<div class="au-section">Species Breakdown</div>', unsafe_allow_html=True)
        rows: List[Dict[str, object]] = []
        for label, count in sorted(summary["detections_by_label"].items()):
            rows.append(
                {
                    "Species": label,
                    "Detections": int(count),
                    "Unique Tracks": int(summary["unique_tracks_by_label"].get(label, 0)),
                }
            )
        if rows:
            st.dataframe(rows, hide_index=True, use_container_width=True)
        else:
            st.warning("No detections survived the current filters. Lower confidence or min object area.")

        sup = summary["suppressed"]
        st.markdown(
            f'<div class="au-note">Suppressed detections: '
            f'{int(sup["small_area"])} small-area, {int(sup["focus_filter"])} outside focus profile.</div>',
            unsafe_allow_html=True,
        )
        with st.expander("Technical Run Summary", expanded=False):
            st.code(json.dumps(summary, indent=2), language="json")


if __name__ == "__main__":
    main()
