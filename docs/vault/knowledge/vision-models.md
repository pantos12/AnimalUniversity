---
type: knowledge
owner: shared
updated: 2026-11-12
---

# Vision Model Choices

## Status and evidence

No model is validated for production animal accuracy. The previous note's
"in use" heading described implementation choices, not executed inference.
Codex has now executed YOLOE-26 small segmentation and small/nano boxes-only
variants on synthetic frames with cached lion prompts, CPU only. See
[[pipeline-review-2026-11-12]] for the six measured configurations and limits.

| Role | Candidate | Evidence / next decision |
| --- | --- | --- |
| Initial box detection | Prepared YOLOE-26 small, boxes only | Local preparation and inference ran; 34.046 median FPS at batch 4 on a short synthetic clip; real-animal accuracy unmeasured |
| Lower compute | Prepared YOLOE-26 nano, boxes only | 66.866 median FPS at batch 4 on the same workload; evaluate recall before selecting |
| Site-tuned detection | YOLO26 small/medium | Candidate for RTX 3080 evaluation after independent labels; not trained or run here |
| Tracking | Existing MotionTracker | Synthetic regression unchanged; identity across real occlusion unverified |

Inspected and executed Ultralytics 8.4.211. Runtime takes local weights and
matching cached prompts, or a prepared checkpoint. Setup is explicit through
`au_prepare_model.py`; `au_doctor.py` checks the actual hardware. The full
repository requirements, UI and training upgrade are still T1 work.

## Alternatives to evaluate

- [YOLOE documentation](https://docs.ultralytics.com/models/yoloe/) documents
  cached prompts and a matching boxes-only architecture; both paths were
  checked against the installed package.
- [YOLO26](https://docs.ultralytics.com/models/yolo26/) is the fine-tuning
  candidate. Site-tuning is a hypothesis to test, not guaranteed accuracy.
- [MegaDetector V6](https://github.com/microsoft/MegaDetector) detects animals,
  people and vehicles, not species. Compare it for known-species enclosures.
- [SpeciesNet](https://github.com/google/cameratrapai) classifies camera-trap
  images; validate transfer to zoo video before adding it to the pipeline.
- [SAM 3](https://github.com/facebookresearch/sam3) is a segmentation/labeling
  candidate if masks are needed. It has not been integrated or run here.
- ReID and video-language reviewers remain evaluation candidates for identity
  gaps and flagged events, respectively; they are not measured improvements.

Do not choose models from synthetic FPS alone or claim animal behavior accuracy
from a bright-square clip. Use independent camera/day validation groups, T3
metrics, day/night examples and observed deployment-device memory use.
