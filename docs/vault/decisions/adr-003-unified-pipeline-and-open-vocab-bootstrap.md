---
type: adr
owner: shared
status: proposed
date: 2026-11-12
---

# ADR-003: One Pipeline for Archived and Live Footage; Open-Vocabulary Bootstrap

## Status

proposed (awaiting review by Djuro, Codex and Claude Code)

## Context

The repository had three pipelines that shared little code: `main.py` (batch,
ByteTrack, SQLite, no behavior rules), `scripts/au_live.py` (live, IoU tracker,
pacing) and the Streamlit demo (own tiling, NMS and label smoothing). Behavior
analysis ran only on live streams, so archived footage could not be used to
check whether pacing detection was right. The offline Milestone ingest from
[[adr-002-offline-milestone-ingest-first]] produced frames but fed nothing.

Problems found while reviewing:

- `main.py` calls `sv.ByteTrack`, which no longer exists in current
  `supervision` (present through 0.30.0, absent in 0.31.7). The requirement
  `supervision<1.0.0` therefore broke fresh installs.
- The live reader slept between reads without draining the stream, so analysis
  falls progressively behind real time.
- Pacing looked only at horizontal movement and counted any jitter across the
  centerline as a crossing.
- The XProtect event XML was unescaped and did not use Milestone's
  `urn:milestone-systems` AnalyticsEvent layout.
- The labeling path needed a hand-drawn prompt box per clip and an
  uninitialized SAM2 submodule.

## Decision

1. **One loop, interchangeable sources.** `animaluniversity.pipeline` defines
   `FrameSource` with `VideoFileSource`, `milestone_export_source` and
   `RtspSource`. `run_pipeline` runs detect -> track -> behavior -> sinks on any
   of them. Archived footage is the development and evaluation path; live is
   the same code with a different source.
2. **Timestamps.** Behavior timing uses `Frame.timestamp_s` (video time for
   files, monotonic time for live). Real-world time is `Frame.wall_time`,
   timezone-aware; stored values are ISO 8601 with offset, and the frame
   manifest also carries `frame_time_utc`. This closes the timestamp open
   question in [[milestone-recording-ingest]].
3. **Open-vocabulary bootstrap.** `OpenVocabDetector` (Ultralytics YOLOE)
   detects species by name with no training. It is used for first results on a
   new enclosure and by `scripts/au_autolabel.py` to label archived footage.
   The production detector remains a fine-tuned YOLO checkpoint (`YoloDetector`).
4. **Tracking.** `MotionTracker` (motion prediction, class-agnostic matching,
   voted labels) is the pipeline default. `IoUTracker` is unchanged because the
   UI depends on it.
5. **Storage.** `SqliteSink` stores track points and behavior events per run
   with camera, enclosure and real time.

## Consequences

- `scripts/au_run.py` supersedes `main.py` and `scripts/au_live.py`. Both still
  work and are not removed here; retire them once `au_run.py` has run on real
  footage.
- `ultralytics` moves to `>=8.4` (YOLO26, YOLOE, SAM 3). `supervision` is
  capped below 0.31 until `main.py` is retired.
- The [[fe-be-contract]] symbols are unchanged. The UI may later adopt
  `run_pipeline`/`MotionTracker`; that is a frontend decision.
- Model-dependent paths (`OpenVocabDetector`, the upgraded `ultralytics`) were
  not executed in the authoring environment (no GPU/torch). Everything else is
  covered by tests. First run on real hardware must confirm them.

## Alternatives considered

- **SAM 3 text prompts for labeling.** Higher mask quality, but heavier and
  gated weights. YOLOE is enough for boxes; SAM 3 stays an option if
  segmentation is needed.
- **`trackers` / BoT-SORT with ReID.** Stronger identity over long gaps, but
  needs `numpy>=2` and more dependencies. Revisit if ID switches remain a
  problem on real footage.
- **Vision-language model in the loop for behavior.** Too slow and costly per
  frame. Suitable only as a second opinion on already-flagged events, through
  `CallbackSink`.
