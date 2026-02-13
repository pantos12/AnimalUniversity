from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
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

SMALL_SPECIES = {"bird"}


@dataclass(frozen=True)
class DemoOptions:
    analysis_mode: str
    allow_labels: Optional[Set[str]]
    min_area_ratio: float
    track_primary_subject: bool
    stabilize_species_labels: bool
    use_scene_consensus: bool
    scene_consensus_min_ratio: float
    scene_consensus_max_conf: float
    scene_history_window: int
    detector_augment: bool
    tiled_recall: bool
    tile_overlap: float
    startup_dense_seconds: float
    small_species_area_guard: float
    small_species_guard_max_conf: float
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
    suppressed_label_filter: int = 0
    label_corrections: int = 0
    detections_by_label: Counter[str] = field(default_factory=Counter)
    rendered_by_label: Counter[str] = field(default_factory=Counter)
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


def _open_video_writer(
    preferred_path: Path,
    fps: float,
    size: Tuple[int, int],
) -> Tuple[cv2.VideoWriter, Path, str]:
    candidates = [
        ("avc1", ".mp4"),
        ("mp4v", ".mp4"),
        ("MJPG", ".avi"),
    ]
    for codec, suffix in candidates:
        out_path = preferred_path.with_suffix(suffix)
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer, out_path, codec
        writer.release()
    raise RuntimeError("Could not create output video writer with avc1/mp4v/MJPG codecs.")


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
            stats.suppressed_label_filter += 1
            continue

        clipped = _clip_bbox(det.bbox_xyxy, frame_width, frame_height)
        processed.append(
            Detection(
                bbox_xyxy=clipped,
                confidence=float(det.confidence),
                label=src_label,
            )
        )
        stats.detections_total_kept += 1
        stats.confidence_sum += float(det.confidence)
        stats.detections_by_label[src_label] += 1
    return processed


def _nms_detections(
    detections: List[Detection],
    iou_threshold: float = 0.82,
    cross_label_iou_threshold: float = 0.72,
) -> List[Detection]:
    ranked = sorted(detections, key=lambda d: float(d.confidence), reverse=True)
    kept: List[Detection] = []
    for det in ranked:
        suppressed = False
        for keep in kept:
            ov = _bbox_iou(det.bbox_xyxy, keep.bbox_xyxy)
            if det.label == keep.label and ov >= iou_threshold:
                suppressed = True
                break
            if det.label != keep.label and ov >= cross_label_iou_threshold and det.confidence <= keep.confidence:
                suppressed = True
                break
        if not suppressed:
            kept.append(det)
    return kept


def _detect_with_tiles(detector: YoloDetector, frame: np.ndarray, overlap: float = 0.18) -> List[Detection]:
    h, w = frame.shape[:2]
    ox = max(8, int(w * max(0.0, min(overlap, 0.4))))
    oy = max(8, int(h * max(0.0, min(overlap, 0.4))))
    mx = w // 2
    my = h // 2
    tiles = [
        (0, 0, min(w, mx + ox), min(h, my + oy)),
        (max(0, mx - ox), 0, w, min(h, my + oy)),
        (0, max(0, my - oy), min(w, mx + ox), h),
        (max(0, mx - ox), max(0, my - oy), w, h),
    ]

    detections: List[Detection] = []
    for x1, y1, x2, y2 in tiles:
        if x2 - x1 < 16 or y2 - y1 < 16:
            continue
        crop = frame[y1:y2, x1:x2]
        for det in detector.detect(crop):
            bx1, by1, bx2, by2 = det.bbox_xyxy
            detections.append(
                Detection(
                    bbox_xyxy=(
                        float(bx1 + x1),
                        float(by1 + y1),
                        float(bx2 + x1),
                        float(by2 + y1),
                    ),
                    confidence=float(det.confidence),
                    label=str(det.label),
                )
            )
    return detections


def _stabilize_track_label(
    track: RenderTrack,
    label_votes: Dict[int, Counter[str]],
    scene_labels: Deque[str],
    scene_counter: Counter[str],
    frame_area: float,
    options: DemoOptions,
    stats: SessionStats,
) -> str:
    current = str(track.label).lower().strip()
    votes = label_votes[int(track.track_id)]
    votes[current] += 1

    chosen = current
    if options.stabilize_species_labels and votes:
        chosen = votes.most_common(1)[0][0]

    if options.use_scene_consensus and scene_counter:
        dominant_label, dominant_count = scene_counter.most_common(1)[0]
        total = max(1, sum(scene_counter.values()))
        dominant_ratio = dominant_count / float(total)
        chosen_ratio = scene_counter.get(chosen, 0) / float(total)
        if (
            dominant_label != chosen
            and dominant_ratio >= options.scene_consensus_min_ratio
            and chosen_ratio <= 0.20
            and float(track.score) <= options.scene_consensus_max_conf
        ):
            chosen = dominant_label

    # Generic guard: very large low-confidence "small species" labels are likely mismatches.
    if chosen in SMALL_SPECIES:
        x1, y1, x2, y2 = track.bbox_xyxy
        area_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / max(1.0, frame_area)
        if (
            area_ratio >= options.small_species_area_guard
            and float(track.score) <= options.small_species_guard_max_conf
        ):
            non_small = [(lbl, cnt) for lbl, cnt in scene_counter.items() if lbl not in SMALL_SPECIES]
            if non_small:
                non_small.sort(key=lambda t: t[1], reverse=True)
                chosen = non_small[0][0]

    if chosen != current:
        stats.label_corrections += 1

    scene_labels.append(chosen)
    scene_counter[chosen] += 1
    while len(scene_labels) > options.scene_history_window:
        old = scene_labels.popleft()
        scene_counter[old] -= 1
        if scene_counter[old] <= 0:
            del scene_counter[old]
    return chosen


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

    # Progress ring moved away from center to avoid obscuring subjects.
    radius = max(22, min(w, h) // 28)
    center = (max(radius + 20, w - radius - 26), radius + 24)
    ring_bg = out.copy()
    cv2.circle(ring_bg, center, radius + 10, (18, 19, 24), -1, cv2.LINE_AA)
    out = cv2.addWeighted(ring_bg, 0.55, out, 0.45, 0)
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

    out_width = width if width % 2 == 0 else max(2, width - 1)
    out_height = height if height % 2 == 0 else max(2, height - 1)
    writer, out_path, codec = _open_video_writer(
        preferred_path=out_path,
        fps=max(1.0, float(infer_fps)),
        size=(out_width, out_height),
    )

    detector = YoloDetector(
        weights_path=weights_path,
        conf_threshold=conf,
        nms_iou=0.9 if not options.track_primary_subject else 0.75,
        device="cpu",
        imgsz=options.imgsz,
        augment=options.detector_augment,
    )
    tracker = IoUTracker(
        iou_threshold=options.tracker_iou,
        max_missed=options.tracker_max_missed,
        allow_label_switch=True,
        switch_iou_threshold=max(0.5, options.tracker_iou + 0.2),
        switch_conf_max=0.72,
        switch_iou_penalty=0.10,
    )
    bbox_smoothing_cache: Dict[int, Tuple[float, float, float, float]] = {}
    track_history: Dict[int, Deque[Tuple[int, int]]] = {}
    label_votes: Dict[int, Counter[str]] = defaultdict(Counter)
    scene_labels: Deque[str] = deque()
    scene_counter: Counter[str] = Counter()
    target_lock_state: Optional[TargetLockState] = None
    stats = SessionStats()

    frame_idx = -1
    while stats.frames_processed < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        t_s = frame_idx / native_fps if native_fps > 0 else 0.0
        active_step = 1 if t_s <= options.startup_dense_seconds else frame_step
        if frame_idx % active_step != 0:
            continue
        stats.frames_processed += 1

        raw_dets = detector.detect(frame)
        if options.tiled_recall and t_s <= options.startup_dense_seconds:
            raw_dets.extend(_detect_with_tiles(detector=detector, frame=frame, overlap=options.tile_overlap))
        raw_dets = _nms_detections(raw_dets)
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
        if options.track_primary_subject:
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
                render_tracks.append(_to_render_track(tr))

        for tr in render_tracks:
            if not tr.predicted:
                tr.label = _stabilize_track_label(
                    track=tr,
                    label_votes=label_votes,
                    scene_labels=scene_labels,
                    scene_counter=scene_counter,
                    frame_area=float(width * height),
                    options=options,
                    stats=stats,
                )
            else:
                scene_labels.append(tr.label)
                scene_counter[tr.label] += 1
                while len(scene_labels) > options.scene_history_window:
                    old = scene_labels.popleft()
                    scene_counter[old] -= 1
                    if scene_counter[old] <= 0:
                        del scene_counter[old]

        for tr in render_tracks:
            smoothed = _smooth_bbox(
                track_id=tr.track_id,
                bbox=tr.bbox_xyxy,
                cache=bbox_smoothing_cache,
                alpha=options.smoothing_alpha,
            )
            tr.bbox_xyxy = smoothed
            stats.rendered_by_label[tr.label] += 1
            stats.unique_tracks_by_label[tr.label].add(tr.track_id)

        ts = frame_idx / native_fps
        rendered = _draw_overlay(
            frame=frame,
            tracks=render_tracks,
            track_history=track_history,
            progress_ratio=(stats.frames_processed / float(max_frames)),
            frame_time_s=ts,
        )
        if rendered.shape[1] != out_width or rendered.shape[0] != out_height:
            rendered = cv2.resize(rendered, (out_width, out_height), interpolation=cv2.INTER_AREA)
        writer.write(np.ascontiguousarray(rendered, dtype=np.uint8))

    cap.release()
    writer.release()

    avg_conf = stats.confidence_sum / stats.detections_total_kept if stats.detections_total_kept else 0.0
    lock_ratio = (
        stats.frames_with_target_lock / stats.frames_processed if stats.frames_processed else 0.0
    )
    summary = {
        "video_path": str(video_path),
        "weights_path": str(weights_path),
        "mode": options.analysis_mode,
        "writer_codec": codec,
        "detector_augment": options.detector_augment,
        "tiled_recall": options.tiled_recall,
        "startup_dense_seconds": options.startup_dense_seconds,
        "frames_processed": stats.frames_processed,
        "frames_with_detections": stats.frames_with_detections,
        "frames_with_target_lock": stats.frames_with_target_lock,
        "target_lock_ratio": round(lock_ratio, 4),
        "detections_total_raw": stats.detections_total_raw,
        "detections_total_kept": stats.detections_total_kept,
        "avg_confidence": round(avg_conf, 4),
        "detections_by_label": dict(stats.detections_by_label),
        "rendered_by_label": dict(stats.rendered_by_label),
        "unique_tracks_by_label": {k: len(v) for k, v in stats.unique_tracks_by_label.items()},
        "suppressed": {
            "small_area": stats.suppressed_small_area,
            "label_filter": stats.suppressed_label_filter,
        },
        "label_corrections": stats.label_corrections,
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
          <div class="au-chip">Automatic animal detection with stable target tracking</div>
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

        track_primary = st.checkbox(
            "Track primary animal only",
            value=False,
            help="Off: track all detected animals. On: lock to a single target.",
        )
        stabilize_species = st.checkbox(
            "Stabilize species labels",
            value=True,
            help="Uses short track history to reduce label flips in multi-animal scenes.",
        )
        scene_consensus = st.checkbox(
            "Use scene consensus for low-confidence labels",
            value=True,
            help="If one species dominates the scene, low-confidence outliers are corrected.",
        )
        high_accuracy = st.checkbox(
            "High accuracy mode (TTA)",
            value=True,
            help="Runs augmented inference for hard scenes (slower, better recall).",
        )
        tiled_recall = st.checkbox(
            "Enable tiled recall assist",
            value=True,
            help="Runs extra tiled detections at startup to catch partially occluded animals.",
        )
        startup_dense_seconds = st.slider(
            "Startup dense scan (seconds)",
            min_value=0.0,
            max_value=20.0,
            value=8.0,
            step=0.5,
            help="Scans every frame at the start to catch multiple animals early.",
        )
        infer_fps = st.slider("Inference FPS", min_value=1, max_value=12, value=4, step=1)
        max_frames = st.slider("Max frames", min_value=60, max_value=3600, value=480, step=60)
        conf = st.slider("Confidence", min_value=0.05, max_value=0.9, value=0.05, step=0.01)
        min_area_percent = st.slider(
            "Minimum object area (%)",
            min_value=0.01,
            max_value=5.0,
            value=0.02,
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
        imgsz = st.select_slider("Model image size", options=[640, 736, 832, 960, 1024], value=1024)
        st.caption("For multiple animals in-frame, confidence 0.05-0.12 usually catches more subjects.")
        run_btn = st.button("Run Analysis", type="primary")

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

        allow_labels: Optional[Set[str]] = None
        tracker_iou = 0.18 if track_primary else 0.25
        tracker_max_missed = max(40, lock_persistence + 8) if track_primary else 28

        options = DemoOptions(
            analysis_mode="automatic",
            allow_labels=allow_labels,
            min_area_ratio=float(min_area_percent) / 100.0,
            track_primary_subject=bool(track_primary),
            stabilize_species_labels=bool(stabilize_species),
            use_scene_consensus=bool(scene_consensus),
            scene_consensus_min_ratio=0.65,
            scene_consensus_max_conf=0.55,
            scene_history_window=180,
            detector_augment=bool(high_accuracy),
            tiled_recall=bool(tiled_recall),
            tile_overlap=0.18,
            startup_dense_seconds=float(startup_dense_seconds),
            small_species_area_guard=0.018,
            small_species_guard_max_conf=0.72,
            lock_persistence_frames=int(lock_persistence),
            tracker_iou=float(tracker_iou),
            tracker_max_missed=int(tracker_max_missed),
            smoothing_alpha=0.56,
            imgsz=int(imgsz),
        )

        stamp = int(time.time())
        out_path = run_dir / f"{video_path.stem}_{weights_path.stem}_analysis_{stamp}.mp4"
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
        for label, count in sorted(summary["rendered_by_label"].items()):
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
            f'{int(sup["small_area"])} small-area, {int(sup["label_filter"])} outside label filter. '
            f'Label corrections applied: {int(summary["label_corrections"])}.</div>',
            unsafe_allow_html=True,
        )
        with st.expander("Technical Run Summary", expanded=False):
            st.code(json.dumps(summary, indent=2), language="json")


if __name__ == "__main__":
    main()
