# AnimalUniversity

Animal behavior analytics for Milestone XProtect cameras, on archived footage and live feeds.

## Start here: one pipeline, archived or live

`scripts/au_run.py` runs detect -> track -> behavior on any source. Develop and
check results on archived footage, then point the same command at a live stream.

```bash
# 1. Archived clip, no trained model yet: name the species (open-vocabulary detection)
python scripts/au_run.py --video data/raw_videos/lion_yard.mp4 --classes "lion" \
  --db data/au.db --report-json data/runs/lion_yard.json

# 2. Milestone export (MP4 + JSON manifest): stores real-world time, camera, enclosure
python scripts/au_run.py --milestone-manifest data/milestone_drop/rec.json --classes "lion" --db data/au.db

# 3. Build a training set from archived footage, review data/datasets/site_v1/review, then train
python scripts/au_autolabel.py --video data/raw_videos/lion_yard.mp4 --classes "lion" --out data/datasets/site_v1
python scripts/au_train.py --dataset data/datasets/site_v1 --weights yolo26s.pt --epochs 50 --out data/runs/train

# 4. Live, with the model you trained, sending pacing events to XProtect
python scripts/au_run.py --rtsp-url "rtsp://<stream>" --weights models/yolo/best.pt --emit-events \
  --source-id "<camera name in XProtect>" --db data/au.db
```

Results land in SQLite (`runs`, `track_points`, `behavior_events`), for example:

```sql
SELECT substr(wall_time, 1, 13) AS hour, COUNT(*) AS pacing_events
FROM behavior_events WHERE name = 'pacing' GROUP BY hour ORDER BY hour;
```

Design and rationale: `docs/vault/decisions/adr-003-unified-pipeline-and-open-vocab-bootstrap.md`.
Run the tests with `pytest`.

The sections below describe the earlier, separate scripts. They still work.

## Structure
- `services/ingest` RTSP ingest + frame pipeline
- `services/analytics` detection, tracking, behavior logic
- `services/eventbridge` XProtect analytics event sender
- `infra` deployment and container configs
- `docs` design notes and runbooks
- `data` local dev data (ignored by git)

## Recommended model plan
For production on live XProtect feeds:

1. Start with `ENA24` (or your best current YOLO checkpoint) for immediate live detection.
2. Collect site-specific frames from your cameras.
3. Use `SAM2` for fast pseudo-labeling, then convert to YOLO dataset and fine-tune.
4. Deploy the fine-tuned YOLO checkpoint for runtime.
5. `SAM3` is currently treated as the same train-first workflow (SAM2 fallback path in code).

This gives fast startup and better long-term accuracy.

## Local quickstart
1. Create and activate a virtual environment, then install deps.
   - PowerShell: `.\scripts\setup_env.ps1`
   - Bash: `./scripts/setup_env.sh`
2. Extract frames from a local video.
   - `python scripts/au_frames.py --video "data/raw_videos/example.mp4" --fps 1 --out "data/frames/example"`
3. (Optional) Run the Streamlit UI.
   - `.\scripts\run_ui.ps1`

## Live RTSP analytics (XProtect)
Run detection + tracking + pacing analysis on a live stream:

```bash
python scripts/au_live.py \
  --rtsp-url "rtsp://<xprotect-ai-bridge-stream>" \
  --model-choice ena24 \
  --weights "models/yolo/ena24.pt" \
  --target-fps 5 \
  --conf 0.25 \
  --emit-events
```

`--model-choice` options:
- `ena24`: direct YOLO runtime using ENA24/base weights.
- `custom`: direct YOLO runtime with `--weights`.
- `sam2`: train-first path; run live with the YOLO model produced from SAM2 labels.
- `sam3`: currently mapped to SAM2 workflow, then YOLO runtime.

To analyze pacing only inside an enclosure zone:

```bash
python scripts/au_live.py \
  --rtsp-url "rtsp://<stream>" \
  --model-choice custom \
  --weights "models/yolo/best.pt" \
  --zone "120,80;980,80;980,620;120,620"
```

## Training pipeline A: ENA24/COCO -> YOLO
```bash
python scripts/au_build_dataset.py \
  --coco-json "data/annotations/ena24.json" \
  --images-dir "data/images/ena24" \
  --out "data/datasets/ena24_yolo" \
  --classes tiger lion elephant

python scripts/au_train.py \
  --dataset "data/datasets/ena24_yolo" \
  --weights "models/yolo/yolov8n.pt" \
  --epochs 100 \
  --imgsz 960 \
  --batch 8 \
  --out "data/runs/train"
```

## Training pipeline B: SAM2 masks -> YOLO
1. Auto-label frames with SAM2:
```bash
python scripts/au_sam2_label.py \
  --checkpoint "models/sam2/sam2.1_hiera_small.pt" \
  --frames-dir "data/frames/cam1" \
  --out "data/masks/cam1" \
  --prompt-box 300 180 900 700
```

2. Convert mask manifest to YOLO dataset:
```bash
python scripts/au_build_dataset.py \
  --masks-manifest "data/masks/cam1/masks_manifest.jsonl" \
  --out "data/datasets/cam1_sam2_yolo" \
  --mode detect \
  --class-name animal
```

3. Train YOLO on the generated dataset:
```bash
python scripts/au_train.py \
  --dataset "data/datasets/cam1_sam2_yolo" \
  --weights "models/yolo/yolov8n.pt" \
  --epochs 100 \
  --imgsz 960 \
  --batch 8 \
  --out "data/runs/train"
```

## Species-level training from local video frames
Build a multi-class species dataset (bird/cat/dog/horse/sheep/cow/elephant/bear/zebra/giraffe)
from extracted frames using pseudo-labeling:

```bash
python scripts/au_bootstrap_species.py \
  --frame-dirs data/frames/videoplayback_s1 data/frames/videoplayback_s2 data/frames/videoplayback_s3 data/frames/videoplayback_s4 \
  --out data/datasets/videoplayback_species_bootstrap \
  --weights models/yolo/yolov8n.pt \
  --device cpu
```

Then train species model:

```bash
python scripts/au_train.py \
  --dataset data/datasets/videoplayback_species_bootstrap \
  --weights models/yolo/yolov8n.pt \
  --epochs 20 \
  --imgsz 640 \
  --batch 8 \
  --out data/runs/train_species
```

## Realtime visual demo (screenshot-style)
The Streamlit app now renders a styled demo page with:
- bold headline + bullet summary
- colorful per-species overlay labels
- ellipse outlines and track IDs
- summary cards and JSON metrics

Run:

```powershell
.\scripts\run_ui.ps1
```
