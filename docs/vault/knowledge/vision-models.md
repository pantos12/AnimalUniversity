---
type: knowledge
owner: shared
updated: 2026-11-12
---

# Vision Model Choices

What we use, what we looked at, and how each claim was checked. Update this
note when a model is tried on real footage; put measured results in
[[verification-record]].

## In use

| Role | Choice | Why |
| --- | --- | --- |
| Bootstrap detection and auto-labeling | YOLOE (`yoloe-26s-seg.pt`) through `OpenVocabDetector` | Detects named species with no training; same library as the runtime model |
| Production detection | Fine-tuned YOLO checkpoint through `YoloDetector` (start from `yolo26s.pt`) | Fast, and accurate on our own cameras once trained on them |
| Tracking | `MotionTracker` (in repo, no dependencies) | Holds identity at about 5 fps and through missed detections |

Checked on 2026-11-12 by inspecting the published `ultralytics` 8.4.211
package: it contains `YOLOE`, the `yoloe-26` and `yolo26` model families, SAM 3
(`models/sam/sam3`), and tracker configs for BoT-SORT, ByteTrack, OC-SORT and
others. **None of these were executed here.**

## Considered, not adopted yet

- **SAM 3 text prompts** for labeling: masks rather than boxes; heavier.
  Revisit if segmentation or pose becomes necessary.
- **`trackers` package / BoT-SORT with ReID:** stronger identity over long
  gaps; needs `numpy>=2`. Revisit if ID switches show up in evaluation.
- **Wildlife-specific models** (`PytorchWildlife`/MegaDetector, `speciesnet`):
  built for camera-trap stills; worth a comparison in T4 for unusual species.
- **Vision-language model for behavior:** only as a reviewer of flagged events,
  never per frame.

## Rules for this note

Name the version you checked and how. Do not list a model as "in use" until it
has run on our hardware.
