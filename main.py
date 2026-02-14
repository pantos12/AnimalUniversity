from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

try:
    import supervision as sv
except Exception as exc:  # noqa: BLE001
    raise ImportError(
        "Missing dependency: supervision. Install it with `pip install supervision`."
    ) from exc


DEFAULT_ANIMAL_CLASSES: tuple[str, ...] = (
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
)


@dataclass(frozen=True)
class TrackRecord:
    session_id: int
    frame_number: int
    timestamp_sec: float
    animal_id: int
    class_name: str
    bbox_json: str


class DatabaseManager:
    """SQLite persistence layer for AnimalUniversity sessions + tracks."""

    def __init__(self, db_path: str | Path = "zoo_university.db") -> None:
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode = WAL;")
        self.conn.execute("PRAGMA synchronous = NORMAL;")
        self.conn.execute("PRAGMA temp_store = MEMORY;")
        self._create_tables()

    def _create_tables(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                start_time TEXT NOT NULL,
                status TEXT NOT NULL
            );
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tracks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                frame_number INTEGER NOT NULL,
                timestamp_sec REAL NOT NULL,
                animal_id INTEGER NOT NULL,
                class_name TEXT NOT NULL,
                bbox_json TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            );
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_tracks_session_frame
            ON tracks (session_id, frame_number);
            """
        )
        self.conn.commit()

    def create_session(self, filename: str, status: str = "running") -> int:
        started_at = datetime.now(tz=timezone.utc).isoformat()
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO sessions (filename, start_time, status) VALUES (?, ?, ?);",
            (filename, started_at, status),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def update_session_status(self, session_id: int, status: str) -> None:
        self.conn.execute("UPDATE sessions SET status = ? WHERE id = ?;", (status, session_id))
        self.conn.commit()

    def bulk_insert_tracks(self, rows: Sequence[TrackRecord]) -> None:
        if not rows:
            return
        payload = [
            (
                r.session_id,
                r.frame_number,
                r.timestamp_sec,
                r.animal_id,
                r.class_name,
                r.bbox_json,
            )
            for r in rows
        ]
        self.conn.executemany(
            """
            INSERT INTO tracks (
                session_id,
                frame_number,
                timestamp_sec,
                animal_id,
                class_name,
                bbox_json
            ) VALUES (?, ?, ?, ?, ?, ?);
            """,
            payload,
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


class VideoProcessor:
    """Streaming video analyzer with low-memory settings for Apple Silicon."""

    def __init__(
        self,
        db: DatabaseManager,
        model_name: str = "yolo11n.pt",
        frame_stride: int = 5,
        max_width: int = 1280,
        flush_every_frames: int = 500,
        max_frames: Optional[int] = None,
        conf_threshold: float = 0.15,
        imgsz: int = 960,
        animal_classes: Optional[Iterable[str]] = None,
    ) -> None:
        self.db = db
        self.frame_stride = max(1, int(frame_stride))
        self.max_width = max(320, int(max_width))
        self.flush_every_frames = max(1, int(flush_every_frames))
        self.max_frames = int(max_frames) if max_frames is not None else None
        self.conf_threshold = float(conf_threshold)
        self.imgsz = int(imgsz)
        self.animal_classes = {
            c.strip().lower()
            for c in (
                animal_classes
                if animal_classes
                else DEFAULT_ANIMAL_CLASSES
            )
            if c.strip()
        }

        # Mac optimization: route inference to MPS when available.
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        try:
            self.model = YOLO(model_name)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Failed to load YOLO model. If running offline, pass a local checkpoint path via "
                "`--model /absolute/path/to/model.pt` (e.g., a file under models/ or data/runs/)."
            ) from exc
        self.tracker = sv.ByteTrack()

        self.model_names = self._read_model_names()
        self.allowed_class_ids = self._resolve_allowed_class_ids()
        self.class_id_to_name = {i: n for i, n in self.model_names.items() if i in self.allowed_class_ids}
        # Stabilize class label per tracker_id to reduce frame-to-frame class flicker.
        self._track_class_votes: dict[int, dict[int, int]] = {}

        if not self.allowed_class_ids:
            raise ValueError(
                f"No target animal classes found in model labels. Requested: {sorted(self.animal_classes)}"
            )

    def _read_model_names(self) -> dict[int, str]:
        raw = self.model.names
        if isinstance(raw, dict):
            return {int(k): str(v).lower() for k, v in raw.items()}
        return {i: str(name).lower() for i, name in enumerate(raw)}

    def _resolve_allowed_class_ids(self) -> List[int]:
        allowed: List[int] = []
        for class_id, class_name in self.model_names.items():
            if class_name in self.animal_classes:
                allowed.append(class_id)
        return sorted(allowed)

    def _iter_frames(
        self,
        cap: cv2.VideoCapture,
        fps: float,
    ) -> Iterator[Tuple[int, float, np.ndarray]]:
        """Generator-based frame streaming (never stores all frames)."""
        frame_number = -1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_number += 1
            frame = self._resize_if_needed(frame)
            ts = frame_number / fps if fps > 0 else 0.0
            yield frame_number, ts, frame

    def _resize_if_needed(self, frame: np.ndarray) -> np.ndarray:
        # Mac optimization: cap width to reduce memory and MPS workload.
        h, w = frame.shape[:2]
        if w <= self.max_width:
            return frame
        scale = self.max_width / float(w)
        new_w = self.max_width
        new_h = max(2, int(h * scale))
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _to_sv_detections(result) -> sv.Detections:
        if hasattr(sv.Detections, "from_ultralytics"):
            return sv.Detections.from_ultralytics(result)

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return sv.Detections.empty()
        xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        conf = boxes.conf.detach().cpu().numpy().astype(np.float32)
        cls = boxes.cls.detach().cpu().numpy().astype(np.int32)
        return sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls)

    @staticmethod
    def _update_tracker(tracker: sv.ByteTrack, detections: sv.Detections) -> sv.Detections:
        if hasattr(tracker, "update_with_detections"):
            return tracker.update_with_detections(detections)
        if hasattr(tracker, "update"):
            return tracker.update(detections)
        raise AttributeError("ByteTrack API not supported by installed supervision version.")

    def process_video(self, video_path: str | Path) -> int:
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        session_id = self.db.create_session(filename=video_path.name, status="running")
        buffer: List[TrackRecord] = []
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            self.db.update_session_status(session_id, "failed_open")
            raise RuntimeError(f"Unable to open video: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if self.max_frames is not None and self.max_frames > 0:
            total_frames = min(total_frames, self.max_frames) if total_frames > 0 else self.max_frames
        pbar = tqdm(total=total_frames if total_frames > 0 else None, unit="frame")

        try:
            for frame_number, timestamp_sec, frame in self._iter_frames(cap=cap, fps=fps):
                found_animals = 0
                if frame_number % self.frame_stride == 0:
                    results = self.model.predict(
                        source=frame,
                        device=self.device,
                        classes=self.allowed_class_ids,
                        conf=self.conf_threshold,
                        imgsz=self.imgsz,
                        verbose=False,
                        stream=False,
                    )
                    result = results[0]
                    detections = self._to_sv_detections(result)
                    tracked = self._update_tracker(self.tracker, detections)
                    found_animals = self._append_track_records(
                        session_id=session_id,
                        frame_number=frame_number,
                        timestamp_sec=timestamp_sec,
                        tracked=tracked,
                        out_buffer=buffer,
                    )

                if frame_number > 0 and (frame_number % self.flush_every_frames == 0):
                    self.db.bulk_insert_tracks(buffer)
                    buffer.clear()
                    if self.device == "mps" and hasattr(torch, "mps"):
                        # Mac optimization: release cached MPS memory during long runs.
                        torch.mps.empty_cache()

                pbar.update(1)
                denom = total_frames if total_frames > 0 else "?"
                pbar.set_description(f"Processing Frame {frame_number}/{denom}")
                pbar.set_postfix_str(f"found {found_animals} animals")

                if self.max_frames is not None and (frame_number + 1) >= self.max_frames:
                    break

            self.db.bulk_insert_tracks(buffer)
            buffer.clear()
            self.db.update_session_status(session_id, "completed")
            return session_id
        except Exception:
            self.db.bulk_insert_tracks(buffer)
            self.db.update_session_status(session_id, "failed")
            raise
        finally:
            pbar.close()
            cap.release()

    def _append_track_records(
        self,
        session_id: int,
        frame_number: int,
        timestamp_sec: float,
        tracked: sv.Detections,
        out_buffer: List[TrackRecord],
    ) -> int:
        if len(tracked) == 0:
            return 0

        tracker_ids = getattr(tracked, "tracker_id", None)
        class_ids = getattr(tracked, "class_id", None)
        boxes = tracked.xyxy
        if tracker_ids is None or class_ids is None:
            return 0

        inserted = 0
        for idx in range(len(tracked)):
            track_id = tracker_ids[idx]
            class_id = class_ids[idx]
            if track_id is None or class_id is None:
                continue
            if isinstance(track_id, float) and np.isnan(track_id):
                continue
            resolved_class_id = self._resolve_track_class_id(track_id=int(track_id), class_id=int(class_id))
            class_name = self.class_id_to_name.get(resolved_class_id, str(resolved_class_id))
            x1, y1, x2, y2 = boxes[idx].tolist()
            bbox_json = json.dumps(
                [round(float(x1), 3), round(float(y1), 3), round(float(x2), 3), round(float(y2), 3)]
            )
            out_buffer.append(
                TrackRecord(
                    session_id=session_id,
                    frame_number=frame_number,
                    timestamp_sec=round(float(timestamp_sec), 3),
                    animal_id=int(track_id),
                    class_name=class_name,
                    bbox_json=bbox_json,
                )
            )
            inserted += 1
        return inserted

    def _resolve_track_class_id(self, track_id: int, class_id: int) -> int:
        votes = self._track_class_votes.setdefault(track_id, {})
        votes[class_id] = votes.get(class_id, 0) + 1
        # Keep the most frequently seen class for this tracker_id.
        return max(votes, key=votes.get)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AnimalUniversity v1.1 video post-processing.")
    parser.add_argument("video_path", help="Path to the input video file.")
    parser.add_argument("--db-path", default="zoo_university.db", help="SQLite database path.")
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics model name/path.")
    parser.add_argument("--frame-stride", type=int, default=5, help="Run inference every Nth frame.")
    parser.add_argument("--max-width", type=int, default=1280, help="Max frame width before inference.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on total source frames to process (for bounded runs).",
    )
    parser.add_argument(
        "--flush-every-frames",
        type=int,
        default=500,
        help="Bulk insert + commit interval in source-frame units.",
    )
    parser.add_argument("--conf", type=float, default=0.15, help="Detection confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=960, help="Inference image size.")
    parser.add_argument(
        "--animal-classes",
        nargs="+",
        default=list(DEFAULT_ANIMAL_CLASSES),
        help="Class labels to keep from model output.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    db = DatabaseManager(db_path=args.db_path)
    try:
        processor = VideoProcessor(
            db=db,
            model_name=args.model,
            frame_stride=args.frame_stride,
            max_width=args.max_width,
            flush_every_frames=args.flush_every_frames,
            max_frames=args.max_frames,
            conf_threshold=args.conf,
            imgsz=args.imgsz,
            animal_classes=args.animal_classes,
        )
        session_id = processor.process_video(args.video_path)
        print(
            f"Completed session {session_id}. "
            f"Results saved to {Path(args.db_path).resolve().as_posix()} on device={processor.device}."
        )
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
