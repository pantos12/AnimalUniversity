import sqlite3

import pytest

from animaluniversity.pipeline.sinks import SqliteSink
from animaluniversity.pipeline.sources import Frame, SourceInfo
from services.analytics.behavior import BehaviorEvent
from services.analytics.tracker import Track


def test_failed_flush_rolls_back_both_tables_and_retains_buffers(tmp_path):
    sink = SqliteSink(tmp_path / "au.db", flush_every=100)
    sink.open(SourceInfo("file", "test"))
    sink._conn.execute("CREATE TRIGGER reject_events BEFORE INSERT ON behavior_events "
                       "BEGIN SELECT RAISE(ABORT, 'disk simulation'); END")
    track = Track(1, (0, 0, 10, 10), "lion", 0.9)
    event = BehaviorEvent("pacing", 1, 0.9, {})
    with pytest.raises(sqlite3.IntegrityError):
        sink.write(Frame(0, None, 0.0), [track], [event])
    assert len(sink._points) == len(sink._events) == 1
    assert sink._conn.execute("SELECT COUNT(*) FROM track_points").fetchone()[0] == 0
    sink._conn.execute("DROP TRIGGER reject_events")
    sink.close("failed")
    with sqlite3.connect(sink.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM track_points").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM behavior_events").fetchone()[0] == 1
        assert conn.execute("SELECT status FROM runs").fetchone()[0] == "failed"


def test_failed_close_always_releases_database_connection(tmp_path, monkeypatch):
    sink = SqliteSink(tmp_path / "au.db")
    sink.open(SourceInfo("file", "test"))

    def fail():
        raise OSError("flush failed")

    monkeypatch.setattr(sink, "_flush", fail)
    with pytest.raises(OSError):
        sink.close()
    assert sink._conn is None


def test_opening_active_sink_twice_is_rejected(tmp_path):
    sink = SqliteSink(tmp_path / "au.db")
    sink.open(SourceInfo("file", "test"))
    try:
        with pytest.raises(RuntimeError, match="already"):
            sink.open(SourceInfo("file", "other"))
    finally:
        sink.close()
