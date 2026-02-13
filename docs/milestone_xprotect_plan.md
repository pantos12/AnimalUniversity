# Milestone XProtect Execution Plan

## Goal
Run live analytics on Milestone XProtect streams, detect animals reliably, and emit behavior events back to XProtect.

## Chosen approach
1. Use YOLO at runtime for speed and stability.
2. Start with ENA24/base weights for immediate deployment.
3. Improve with SAM2-assisted labeling + fine-tuning on your own camera data.
4. Keep SAM3 as a future upgrade path; current code maps it to SAM2 workflow.

## Executable phases
1. **Connect and sync**
   - Ensure local `main` tracks `origin/main`.
   - Keep one source of truth in `https://github.com/pantos12/AnimalUniversity`.

2. **Live analytics**
   - Run: `python scripts/au_live.py --rtsp-url ... --model-choice ena24 --weights models/yolo/ena24.pt --emit-events`
   - Output: live detections, tracks, pacing events, optional EventBridge pushes.

3. **Data improvement loop**
   - Extract frames from camera clips/live captures.
   - Run SAM2 auto-labeling.
   - Convert masks to YOLO dataset.
   - Train YOLO and redeploy best checkpoint.

4. **Behavior expansion**
   - Add additional rules after pacing baseline (loitering, aggression, zone intrusion).
   - Map each rule to distinct `EventType` in XProtect.

## Commands
See top-level `README.md` for exact commands for:
- ENA24/COCO training path
- SAM2 mask-manifest training path
- live RTSP analytics execution
