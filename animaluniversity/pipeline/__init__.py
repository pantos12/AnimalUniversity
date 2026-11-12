"""Unified analytics pipeline: any frame source -> detect -> track -> behavior -> sinks."""

from animaluniversity.pipeline.runner import run_pipeline
from animaluniversity.pipeline.sinks import CallbackSink, Sink, SqliteSink, XProtectEventSink
from animaluniversity.pipeline.sources import (
    Frame,
    FrameSource,
    RtspSource,
    SourceInfo,
    VideoFileSource,
    milestone_export_source,
)

__all__ = [
    "CallbackSink",
    "Frame",
    "FrameSource",
    "RtspSource",
    "Sink",
    "SourceInfo",
    "SqliteSink",
    "VideoFileSource",
    "XProtectEventSink",
    "milestone_export_source",
    "run_pipeline",
]
