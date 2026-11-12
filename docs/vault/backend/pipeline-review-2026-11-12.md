---
type: review
owner: codex
updated: 2026-11-12
---

# Unified pipeline review

Review base: PR #3, commit `45fef2d`. Djuro requested a review of its changes,
efficiency improvements and a current model comparison. The frontend
implementation was inspected at its backend boundary and left to its owner.

## Judgment

The common source/runner/sink interface is useful. The immediate gains are
reliable data, offline startup and removing unnecessary inference work. A larger
vision model cannot repair mislabeled training data or establish pacing accuracy.
Use prepared YOLOE-26 small as the first box-detection candidate; compare it with
a site-tuned detector on independent camera/day recordings before production.

## Implemented from the review

| Finding | Change |
| --- | --- |
| Raw YOLOE startup can fetch weights, CLIP and a text encoder | Explicit preparation command; runtime requires local weights and matching cached prompts; CLI defaults disable network checks and dependency installation |
| Box-only pipeline runs a segmentation model | Optional preparation from the matching detection architecture; no unused masks at inference |
| Archived inference dispatches each frame separately | Bounded opt-in batches, retaining sequential tracker/behavior updates and original timestamps; live batching rejected |
| No measured throughput in reports | Opt-in source/inference/analytics/sink timing and FPS |
| Random adjacent-frame validation inflates confidence | Whole-recording/group splits with stable IDs and explicit train/val assignment |
| Dataset append silently changes class IDs or overwrites files | Locked class order, recording identity, collision checks, incomplete-run markers and checked image writes |
| Empty detection can become a false negative training label | Skip candidate negatives by default; require explicit opt-in and review |
| Half a failed SQLite batch can disappear from memory | Atomic transaction, rollback and retained buffers; always release connections and propagate failed finalization |
| Sink startup/cleanup errors leak resources or hide inference failures | Guard partial startup, explicitly close iterators, close every opened sink, preserve original errors |
| Invalid rates/portable paths pass validation | Finite positive sampling rates, positive limits, blank-ID and Windows/POSIX filename validation |
| Console `pytest` cannot find repository packages | Repository pytest configuration; both test entry points work |

## Measured inference throughput

CPU on the M5/macOS 27.0.1 environment, Python 3.12.14, torch 2.15.0,
Ultralytics 8.4.211, NumPy 2.5.4, OpenCV 5.0.0.93; 2 torch threads, image size
640, one prompted class. Median of three warmed runs of 32 synthetic 640x360
frames, sampled at 5 FPS. Includes source, analytics and inference; excludes
model loading and prompt preparation. Python socket.connect calls were denied
and counted: **zero attempts**. MPS is unavailable to this process.

| Model configuration | Batch | Median FPS |
| --- | ---: | ---: |
| YOLOE-26 small, segmentation (original architecture) | 1 | 22.921 |
| YOLOE-26 small, segmentation | 4 | 25.093 |
| YOLOE-26 small, boxes only | 1 | 28.632 |
| YOLOE-26 small, boxes only | 4 | 34.046 |
| YOLOE-26 nano, boxes only | 1 | 53.944 |
| YOLOE-26 nano, boxes only | 4 | 66.866 |

Small boxes-only/batch 4 processed about **49% more frames per second** than
the original segmentation/batch 1 configuration on this workload. This is a
short synthetic throughput measurement: all model runs detected zero animals.
It establishes no recall, precision, real-time camera capacity, GPU performance
or equivalence of the converted model's predictions on real footage. Nano is a
speed candidate, not an accuracy upgrade. Choose batch size on the target device.

The preparation command records model checksums and class order. The model
package and real inference were executed here, improving on the prior
source-inspection-only verification; T1 remains incomplete.

## Behavior regression comparison

Same 40-second synthetic clips, 10 FPS originals sampled at 5 FPS, with every
tenth detector output withheld. Compared base and review pipeline summaries.

| Clip | Before: tracks / pacing alerts | After: tracks / pacing alerts |
| --- | --- | --- |
| Horizontal pacing | 1 / 3 | 1 / 3 |
| Vertical pacing | 1 / 3 | 1 / 3 |
| Circling | 1 / 0 | 1 / 0 |

All 200 analyzed frames per clip and complete deterministic summaries matched.
An additional regression test compares frame timestamps, track IDs and events
between batch sizes 1 and 4. This is a regression check using a bright-square
detector; it is not the T3 real-footage evaluation harness.

## Model decisions and next work

- Keep YOLOE-26 as the initial open-vocabulary candidate. The official
  [YOLOE documentation](https://docs.ultralytics.com/models/yoloe/) supports
  cached prompts and a matching detection-only architecture. The checked
  package implements both; the preparation and local inference paths ran here.
- Compare a fine-tuned YOLO26 small/medium model on the RTX machine after T2/T3.
  [YOLO26](https://docs.ultralytics.com/models/yolo26/) is a candidate, with no
  project-specific accuracy claim yet.
- Include [MegaDetector V6](https://github.com/microsoft/MegaDetector) as an
  animal-detection baseline. It identifies animals/people/vehicles, not species;
  known single-species enclosures may use manifest species metadata. It has not
  been integrated or benchmarked here.
- [SpeciesNet](https://github.com/google/cameratrapai) is a camera-trap image
  classification candidate; evaluate its domain fit before adding a second
  per-frame model. [SAM 3](https://github.com/facebookresearch/sam3) remains a
  candidate for mask labeling if boxes are insufficient, not an automatic
  replacement for this box pipeline. Neither ran here.

Prioritize representative day/night/occlusion clips, human-reviewed independent
labels and T3 evaluation next. Then address VFR/decoder timestamps, RTSP
reconnect/thread failures, bounded long-run report memory and a durable event
outbox so a slow Event Server cannot stall inference (T11). Tie the dashboard
to SQLite with run status, model configuration, time range and explicit data
provenance; alert counts are not pacing durations or clinical conclusions.
See the frontend handoff, [[work-queue]] and [[hardware-test-runbook]].
