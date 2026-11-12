"""Frame sources: one interface for archived footage and live streams.

Everything downstream of a source (detector, tracker, behavior rules, sinks)
sees the same ``Frame`` objects, so recorded footage and a live camera run
through identical code. Recorded footage gives repeatable results for training
and evaluation; live uses the same logic once it is trusted.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceInfo:
    """Where frames come from. Carried onto every stored track and event."""

    kind: str  # "file" | "milestone_export" | "rtsp"
    name: str
    camera_id: Optional[str] = None
    camera_name: Optional[str] = None
    enclosure_id: Optional[str] = None
    species: Optional[str] = None


@dataclass
class Frame:
    index: int  # running count of frames yielded by the source
    image: Any  # BGR numpy array
    timestamp_s: float  # seconds since the start of the source; use for behavior timing
    wall_time: Optional[datetime] = None  # timezone-aware real-world time, when known
    source_frame: Optional[int] = None  # frame number inside the original video


class FrameSource:
    info: SourceInfo

    def __iter__(self) -> Iterator[Frame]:
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - default no-op
        return None


class VideoFileSource(FrameSource):
    """Archived footage. Timestamps come from the video itself, so runs repeat exactly.

    ``start_time`` is the real-world time of the first frame (timezone-aware).
    When given, each frame also carries ``wall_time``.
    """

    def __init__(
        self,
        video_path: str | Path,
        target_fps: Optional[float] = None,
        start_time: Optional[datetime] = None,
        info: Optional[SourceInfo] = None,
        max_frames: Optional[int] = None,
    ) -> None:
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")
        if start_time is not None and start_time.tzinfo is None:
            raise ValueError("start_time must be timezone-aware")
        self.target_fps = target_fps
        self.start_time = start_time
        self.max_frames = max_frames
        self.info = info or SourceInfo(kind="file", name=self.video_path.name)

    def __iter__(self) -> Iterator[Frame]:
        import cv2

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {self.video_path}")
        try:
            native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if native_fps <= 0:
                logger.warning("Video reports no frame rate; assuming 25 fps: %s", self.video_path)
                native_fps = 25.0
            step_s = 1.0 / self.target_fps if self.target_fps and self.target_fps > 0 else 0.0
            next_due_s = 0.0
            source_frame = 0
            yielded = 0
            while True:
                # grab() skips the costly decode for frames we are not going to use
                if not cap.grab():
                    break
                timestamp_s = source_frame / native_fps
                current = source_frame
                source_frame += 1
                if timestamp_s + 1e-9 < next_due_s:
                    continue
                ok, image = cap.retrieve()
                if not ok:
                    continue
                next_due_s = (next_due_s + step_s) if step_s else 0.0
                wall = self.start_time + timedelta(seconds=timestamp_s) if self.start_time else None
                yield Frame(
                    index=yielded,
                    image=image,
                    timestamp_s=timestamp_s,
                    wall_time=wall,
                    source_frame=current,
                )
                yielded += 1
                if self.max_frames is not None and yielded >= self.max_frames:
                    break
        finally:
            cap.release()


def milestone_export_source(
    manifest_path: str | Path,
    base_dir: Optional[str | Path] = None,
    target_fps: Optional[float] = None,
    strict: bool = False,
    max_frames: Optional[int] = None,
) -> VideoFileSource:
    """Open a Milestone export (MP4 + JSON manifest) as a frame source.

    Camera, enclosure and the recording's real start time come from the manifest,
    so every track and event is stored with real-world time. No frames are
    written to disk. ``strict=False`` by default because real exports rarely
    match the requested time range to the second.
    """
    from animaluniversity.ingest.milestone import load_milestone_manifest, validate_milestone_assets

    manifest_path = Path(manifest_path)
    recording = load_milestone_manifest(manifest_path)
    video_path, _meta = validate_milestone_assets(
        recording, base_dir or manifest_path.parent, strict=strict
    )
    info = SourceInfo(
        kind="milestone_export",
        name=recording.video_filename,
        camera_id=recording.camera_id,
        camera_name=recording.camera_name,
        enclosure_id=recording.enclosure_id,
        species=recording.species,
    )
    return VideoFileSource(
        video_path,
        target_fps=target_fps or recording.fps,
        start_time=recording.start_time,
        info=info,
        max_frames=max_frames,
    )


@dataclass
class _Latest:
    image: Any = None
    seq: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


class RtspSource(FrameSource):
    """Live stream that always hands over the *newest* frame.

    A background thread reads the stream continuously and keeps only the latest
    frame. Reading at the camera's pace and discarding what analysis cannot keep
    up with prevents the decoder buffer from filling, which otherwise makes
    "live" analytics drift seconds or minutes behind reality.
    """

    def __init__(
        self,
        url: str,
        target_fps: float = 5.0,
        reconnect_seconds: float = 5.0,
        info: Optional[SourceInfo] = None,
        max_frames: Optional[int] = None,
        capture_factory: Optional[Callable[[str], Any]] = None,
    ) -> None:
        if not url:
            raise ValueError("RTSP url is required")
        self.url = url
        self.target_fps = max(target_fps, 0.1)
        self.reconnect_seconds = reconnect_seconds
        self.max_frames = max_frames
        # Never put the URL in SourceInfo: it can contain camera credentials.
        self.info = info or SourceInfo(kind="rtsp", name="live")
        self._capture_factory = capture_factory
        self._latest = _Latest()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _open(self):
        if self._capture_factory is not None:
            return self._capture_factory(self.url)
        import cv2

        return cv2.VideoCapture(self.url)

    def _reader(self) -> None:
        cap = None
        while not self._stop.is_set():
            if cap is None or not cap.isOpened():
                cap = self._open()
                if not cap.isOpened():
                    logger.warning("Stream not available; retrying in %.0fs", self.reconnect_seconds)
                    self._stop.wait(self.reconnect_seconds)
                    continue
            ok, image = cap.read()
            if not ok:
                logger.warning("Stream read failed; reconnecting in %.0fs", self.reconnect_seconds)
                cap.release()
                cap = None
                self._stop.wait(self.reconnect_seconds)
                continue
            with self._latest.lock:
                self._latest.image = image
                self._latest.seq += 1
        if cap is not None:
            cap.release()

    def __iter__(self) -> Iterator[Frame]:
        self._stop.clear()
        self._thread = threading.Thread(target=self._reader, name="rtsp-reader", daemon=True)
        self._thread.start()
        interval = 1.0 / self.target_fps
        started = time.monotonic()
        last_seq = 0
        yielded = 0
        try:
            while not self._stop.is_set():
                tick = time.monotonic()
                with self._latest.lock:
                    image, seq = self._latest.image, self._latest.seq
                if image is None or seq == last_seq:
                    time.sleep(min(interval, 0.01))
                    continue
                last_seq = seq
                yield Frame(
                    index=yielded,
                    image=image,
                    timestamp_s=tick - started,
                    wall_time=datetime.now(tz=timezone.utc),
                    source_frame=seq,
                )
                yielded += 1
                if self.max_frames is not None and yielded >= self.max_frames:
                    break
                remaining = interval - (time.monotonic() - tick)
                if remaining > 0:
                    time.sleep(remaining)
        finally:
            self.close()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)
