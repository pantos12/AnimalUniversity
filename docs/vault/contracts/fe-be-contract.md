---
type: contract
owner: shared
status: active
---

# FE/BE Contract

This is the single source of truth for backend symbols used by the frontend. If
this note and the code disagree, verify the code and correct this note in the same
change.

## Current surface

`apps/ui_streamlit/app.py` may import these backend symbols:

```python
from animaluniversity.core.video import get_video_metadata
from animaluniversity.utils.paths import get_data_dir, get_models_dir
from services.analytics.detector import YoloDetector
from services.analytics.tracker import Detection, IoUTracker, Track
```

| Symbol | Module | Frontend use |
| --- | --- | --- |
| `get_video_metadata` | `animaluniversity.core.video` | Duration, FPS, resolution |
| `get_data_dir` | `animaluniversity.utils.paths` | Upload, frame, and run paths |
| `get_models_dir` | `animaluniversity.utils.paths` | Model-weight discovery |
| `YoloDetector` | `services.analytics.detector` | Per-frame detection |
| `Detection`, `Track`, `IoUTracker` | `services.analytics.tracker` | Boxes, IDs, labels |

## Rules

### Frontend

- Request a new backend capability in [[frontend-to-backend]].
- Do not import deeper backend modules or duplicate backend logic in the UI.
- Presentation-only derivations remain in the UI.

### Backend

- Treat changed signatures or behavior above as breaking changes.
- Record breaking and additive interface changes in [[backend-to-frontend]].
- Add a newly approved symbol here in the same PR that exposes it.

## Change log

| Date | Change | By | PR |
| --- | --- | --- | --- |
| 2026-08-17 | Initial contract extracted from current UI imports | Both | Not published |
