"""Build a YOLO training set from archived footage, no manual boxes.

    python scripts/au_autolabel.py \
        --video data/raw_videos/lion_yard.mp4 \
        --classes "lion" \
        --out data/datasets/site_v1 --fps 1

Run it once per clip with the same --out to grow one dataset. Review the images
in <out>/review before training, then:

    python scripts/au_train.py --dataset data/datasets/site_v1 --weights yolo26s.pt
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", help="Archived video file.")
    p.add_argument("--milestone-manifest", help="JSON manifest of a Milestone export.")
    p.add_argument("--classes", required=True, help='Comma-separated species names, e.g. "tiger,red panda".')
    p.add_argument("--out", required=True, help="Dataset folder to create or extend.")
    p.add_argument("--fps", type=float, default=1.0, help="Frames sampled per second of video.")
    p.add_argument("--min-conf", type=float, default=0.35)
    p.add_argument("--keep-empty-fraction", type=float, default=0.0,
                   help="Opt into candidate negatives; review them before training (default: skip).")
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--split", choices=["train", "val"], help="Assign this recording group to one split.")
    p.add_argument("--recording-id", help="Stable unique recording ID, including across machines.")
    p.add_argument("--split-group", help="Group related clips (e.g. camera/day) into the same split.")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--weights", default=None, help="YOLOE checkpoint (default: yoloe-26s-seg.pt).")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--device", default=None)
    p.add_argument("--prompt-embeddings", help="Cached YOLOE .npz prompts for offline startup.")
    args = p.parse_args(argv)
    os.environ.setdefault("YOLO_OFFLINE", "true")
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")

    if bool(args.video) == bool(args.milestone_manifest):
        logging.error("Choose exactly one of --video or --milestone-manifest.")
        return 2

    from animaluniversity.labeling import autolabel_video
    from animaluniversity.pipeline import VideoFileSource, milestone_export_source
    from services.analytics.detector import OpenVocabDetector

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    if args.video:
        source = VideoFileSource(args.video, target_fps=args.fps, max_frames=args.max_frames)
    else:
        source = milestone_export_source(args.milestone_manifest, target_fps=args.fps, max_frames=args.max_frames)

    detector = OpenVocabDetector(
        classes=classes,
        weights_path=args.weights,
        conf_threshold=min(args.min_conf, 0.25),
        device=args.device,
        imgsz=args.imgsz,
        prompt_embeddings=args.prompt_embeddings,
    )
    result = autolabel_video(
        source=source,
        detector=detector,
        class_names=classes,
        out_dir=args.out,
        min_conf=args.min_conf,
        keep_empty_fraction=args.keep_empty_fraction,
        val_fraction=args.val_fraction,
        split=args.split,
        recording_id=args.recording_id,
        split_group=args.split_group,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
