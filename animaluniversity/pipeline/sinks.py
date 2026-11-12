"""Where pipeline results go: SQLite, XProtect, or any callback."""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, List, Optional

from animaluniversity.pipeline.sources import Frame, SourceInfo

logger = logging.getLogger(__name__)


class Sink:
    def open(self, info: SourceInfo) -> None:
        return None

    def write(self, frame: Frame, tracks: Iterable[object], events: Iterable[object]) -> None:
        return None

    def close(self, status: str = "completed") -> None:
        return None


class SqliteSink(Sink):
    """Stores every track position and behavior event with camera and real time.

    One database can hold many runs (archived clips and live sessions alike), so
    questions such as "how much did the lion pace last week, by hour" become a
    single SQL query.
    """

    def __init__(self, db_path: str | Path, flush_every: int = 200) -> None:
        self.db_path = Path(db_path)
        self.flush_every = max(1, flush_every)
        self._conn: Optional[sqlite3.Connection] = None
        self._run_id: Optional[int] = None
        self._points: List[tuple] = []
        self._events: List[tuple] = []

    @property
    def run_id(self) -> Optional[int]:
        return self._run_id

    def open(self, info: SourceInfo) -> None:
        if self._conn is not None:
            raise RuntimeError("SQLite sink is already open")
        if self._points or self._events:
            raise RuntimeError("SQLite sink has uncommitted results from a failed run; use a new sink")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute("PRAGMA journal_mode = WAL;")
        self._conn.execute("PRAGMA foreign_keys = ON;")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_kind TEXT NOT NULL,
                source_name TEXT NOT NULL,
                camera_id TEXT,
                camera_name TEXT,
                enclosure_id TEXT,
                species TEXT,
                started_at TEXT NOT NULL,
                status TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS track_points (
                run_id INTEGER NOT NULL REFERENCES runs(id),
                frame_index INTEGER NOT NULL,
                timestamp_s REAL NOT NULL,
                wall_time TEXT,
                track_id INTEGER NOT NULL,
                label TEXT NOT NULL,
                score REAL NOT NULL,
                x1 REAL NOT NULL, y1 REAL NOT NULL, x2 REAL NOT NULL, y2 REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_track_points_run ON track_points (run_id, track_id, frame_index);
            CREATE TABLE IF NOT EXISTS behavior_events (
                run_id INTEGER NOT NULL REFERENCES runs(id),
                frame_index INTEGER NOT NULL,
                timestamp_s REAL NOT NULL,
                wall_time TEXT,
                name TEXT NOT NULL,
                track_id INTEGER NOT NULL,
                score REAL NOT NULL,
                details_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_behavior_events_run ON behavior_events (run_id, name);
            """
        )
        cur = self._conn.execute(
            "INSERT INTO runs (source_kind, source_name, camera_id, camera_name, enclosure_id, "
            "species, started_at, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
            (
                info.kind,
                info.name,
                info.camera_id,
                info.camera_name,
                info.enclosure_id,
                info.species,
                datetime.now(tz=timezone.utc).isoformat(),
                "running",
            ),
        )
        self._run_id = int(cur.lastrowid)
        self._conn.commit()

    def write(self, frame: Frame, tracks: Iterable[object], events: Iterable[object]) -> None:
        if self._conn is None:
            raise RuntimeError("SQLite sink must be opened before writing")
        wall = frame.wall_time.isoformat() if frame.wall_time else None
        for tr in tracks:
            x1, y1, x2, y2 = tr.bbox_xyxy
            self._points.append(
                (self._run_id, frame.index, frame.timestamp_s, wall, int(tr.track_id),
                 str(tr.label), float(tr.score), float(x1), float(y1), float(x2), float(y2))
            )
        for ev in events:
            self._events.append(
                (self._run_id, frame.index, frame.timestamp_s, wall, str(ev.name),
                 int(ev.track_id), float(ev.score), json.dumps(ev.details))
            )
        if self._events or len(self._points) >= self.flush_every:
            self._flush()

    def _flush(self) -> None:
        if self._conn is None:
            return
        # Commit tracks and their events together. Keep both buffers intact on
        # failure so a retry cannot lose points or duplicate half a batch.
        with self._conn:
            if self._points:
                self._conn.executemany(
                    "INSERT INTO track_points VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);", self._points
                )
            if self._events:
                self._conn.executemany(
                    "INSERT INTO behavior_events VALUES (?, ?, ?, ?, ?, ?, ?, ?);", self._events
                )
        self._points.clear()
        self._events.clear()

    def close(self, status: str = "completed") -> None:
        if self._conn is None:
            return
        try:
            self._flush()
            with self._conn:
                self._conn.execute("UPDATE runs SET status = ? WHERE id = ?;", (status, self._run_id))
        except Exception:
            try:
                with self._conn:
                    self._conn.execute("UPDATE runs SET status = 'failed' WHERE id = ?;", (self._run_id,))
            except Exception as exc:
                logger.error("Could not mark SQLite run failed: %s", exc)
            raise
        finally:
            self._conn.close()
            self._conn = None


class XProtectEventSink(Sink):
    """Sends each behavior event to the XProtect Event Server.

    A failed send is logged and counted, never raised: losing one alert must not
    stop analysis of the stream.
    """

    def __init__(
        self,
        source_id: Optional[str] = None,
        event_type_prefix: str = "AnimalBehavior.",
        endpoint: Optional[str] = None,
    ) -> None:
        self.source_id = source_id
        self.event_type_prefix = event_type_prefix
        self.endpoint = endpoint
        self.sent = 0
        self.failed = 0
        self._info: Optional[SourceInfo] = None

    def open(self, info: SourceInfo) -> None:
        self._info = info

    def write(self, frame: Frame, tracks: Iterable[object], events: Iterable[object]) -> None:
        from services.eventbridge.event_sender import build_analytics_event_xml, send_event

        info = self._info
        source = self.source_id or (info.camera_name if info else None) or "CAMERA_1"
        for ev in events:
            props = {"track_id": str(ev.track_id), "score": f"{float(ev.score):.3f}", **ev.details}
            if info and info.enclosure_id:
                props["enclosure_id"] = info.enclosure_id
            xml = build_analytics_event_xml(
                source_id=source,
                event_type=f"{self.event_type_prefix}{str(ev.name).capitalize()}",
                properties=props,
                timestamp=frame.wall_time,
            )
            try:
                send_event(xml, endpoint=self.endpoint)
                self.sent += 1
            except Exception as exc:  # noqa: BLE001
                self.failed += 1
                logger.error("XProtect event send failed: %s", exc)


class CallbackSink(Sink):
    """Calls ``on_event(frame, event, info)`` for each behavior event.

    Extension point for anything that should react to an event without being
    part of the real-time loop: saving a clip, paging a keeper, or handing the
    frame to an external model for a second opinion.
    """

    def __init__(self, on_event: Callable[[Frame, object, Optional[SourceInfo]], None]) -> None:
        self._on_event = on_event
        self._info: Optional[SourceInfo] = None

    def open(self, info: SourceInfo) -> None:
        self._info = info

    def write(self, frame: Frame, tracks: Iterable[object], events: Iterable[object]) -> None:
        for ev in events:
            try:
                self._on_event(frame, ev, self._info)
            except Exception as exc:  # noqa: BLE001
                logger.error("Event callback failed: %s", exc)
