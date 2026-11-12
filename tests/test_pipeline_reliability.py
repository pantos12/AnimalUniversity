from __future__ import annotations

import numpy as np
import pytest

from animaluniversity.pipeline import run_pipeline
from animaluniversity.pipeline.sources import Frame, FrameSource, SourceInfo
from animaluniversity.pipeline.sinks import Sink
from animaluniversity.pipeline import VideoFileSource
from tests.conftest import BrightBlobDetector


class MemorySource(FrameSource):
    info = SourceInfo(kind="file", name="memory")

    def __init__(self, count=7):
        self.count = count
        self.released = False
        self.closed = False
        self.read = 0

    def __iter__(self):
        try:
            for i in range(self.count):
                self.read += 1
                yield Frame(i, np.full((4, 4, 3), i, dtype=np.uint8), i / 5)
        finally:
            self.released = True

    def close(self):
        self.closed = True


class BatchDetector:
    def __init__(self):
        self.batches = []

    def detect(self, image):
        return []

    def detect_batch(self, images):
        self.batches.append([int(im[0, 0, 0]) for im in images])
        return [[] for _ in images]


class RecordingSink(Sink):
    def __init__(self, fail_open=False, fail_close=False):
        self.fail_open, self.fail_close = fail_open, fail_close
        self.status = None

    def open(self, info):
        if self.fail_open:
            raise RuntimeError("open failed")

    def close(self, status="completed"):
        self.status = status
        if self.fail_close:
            raise RuntimeError("close failed")


def test_batches_preserve_order_and_do_not_read_past_limit():
    source, detector = MemorySource(), BatchDetector()
    seen = []
    summary = run_pipeline(source, detector, batch_size=3, max_frames=5,
                           on_frame=lambda f, *_: seen.append(f.index))
    assert detector.batches == [[0, 1, 2], [3, 4]]
    assert seen == list(range(5)) and source.read == 5
    assert summary["frames_processed"] == 5
    assert source.released and source.closed


def test_batch_and_single_frame_summaries_match():
    assert run_pipeline(MemorySource(), BatchDetector()) == run_pipeline(
        MemorySource(), BatchDetector(), batch_size=4)


def test_pacing_events_and_track_ids_match_across_batch_sizes(pacing_video):
    class BatchedBlob(BrightBlobDetector):
        def detect_batch(self, images):
            return [self.detect(image) for image in images]

    def run(batch_size):
        seen = []
        summary = run_pipeline(VideoFileSource(pacing_video, target_fps=5), BatchedBlob(),
                               batch_size=batch_size,
                               on_frame=lambda f, tracks, events: seen.append(
                                   (f.timestamp_s, [t.track_id for t in tracks], events)))
        return summary, seen

    single, batched = run(1), run(4)
    assert single == batched
    assert single[0]["behavior_events_by_name"]["pacing"] > 0


def test_early_stop_closes_the_source_iterator():
    source = MemorySource()
    run_pipeline(source, BatchDetector(), max_frames=1)
    assert source.released and source.closed


def test_sink_open_failure_closes_partial_resources():
    source = MemorySource()
    first, failing = RecordingSink(), RecordingSink(fail_open=True)
    with pytest.raises(RuntimeError, match="open failed"):
        run_pipeline(source, BatchDetector(), sinks=[first, failing])
    assert first.status == failing.status == "failed"
    assert source.closed


def test_cleanup_failure_does_not_hide_original_error():
    class BrokenSource(MemorySource):
        def close(self):
            raise RuntimeError("source close failed")

    class BrokenDetector(BatchDetector):
        def detect(self, image):
            raise ValueError("inference failed")

    sink = RecordingSink()
    with pytest.raises(ValueError, match="inference failed"):
        run_pipeline(BrokenSource(), BrokenDetector(), sinks=[sink])
    assert sink.status == "failed"


def test_unsaved_results_are_not_reported_as_completed():
    with pytest.raises(RuntimeError, match="cleanup"):
        run_pipeline(MemorySource(), BatchDetector(), sinks=[RecordingSink(fail_close=True)])


def test_batch_cardinality_failure_is_not_silently_truncated():
    class BadBatch(BatchDetector):
        def detect_batch(self, images):
            return [[]]

    sink = RecordingSink()
    with pytest.raises(ValueError, match="result.*frame"):
        run_pipeline(MemorySource(), BadBatch(), batch_size=3, sinks=[sink])
    assert sink.status == "failed"


def test_live_sources_reject_batches_that_add_latency():
    source = MemorySource()
    source.info = SourceInfo(kind="rtsp", name="live")
    with pytest.raises(ValueError, match="live"):
        run_pipeline(source, BatchDetector(), batch_size=3)


def test_profiling_is_opt_in_and_reports_actual_work():
    summary = run_pipeline(MemorySource(), BatchDetector(), profile=True, batch_size=3)
    perf = summary["performance"]
    assert perf["inference_calls"] == 3
    assert perf["frames_per_second"] > 0
    assert perf["elapsed_s"] >= perf["inference_s"] >= 0


@pytest.mark.parametrize("kwargs", [{"batch_size": 0}, {"batch_size": -1}, {"max_frames": 0}])
def test_invalid_limits_fail_before_processing(kwargs):
    source = MemorySource()
    with pytest.raises(ValueError):
        run_pipeline(source, BatchDetector(), **kwargs)
    assert source.read == 0
