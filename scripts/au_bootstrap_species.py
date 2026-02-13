from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from animaluniversity.yolo.bootstrap import (
    COCO_ANIMAL_CLASS_ID_TO_NAME,
    build_species_bootstrap_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a species-level YOLO dataset from video frames using pseudo-labels."
    )
    parser.add_argument(
        "--frame-dirs",
        nargs="+",
        required=True,
        help="One or more directories containing frame_*.jpg files.",
    )
    parser.add_argument("--out", required=True, help="Output dataset directory.")
    parser.add_argument("--weights", required=True, help="Detector weights for pseudo-labeling.")
    parser.add_argument(
        "--class-ids",
        nargs="*",
        type=int,
        default=None,
        help=(
            "Optional subset of COCO animal class IDs. "
            f"Supported: {sorted(COCO_ANIMAL_CLASS_ID_TO_NAME.keys())}"
        ),
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Model predict confidence threshold.")
    parser.add_argument("--min-conf", type=float, default=0.35, help="Minimum confidence kept in labels.")
    parser.add_argument(
        "--neg-keep-prob",
        type=float,
        default=0.2,
        help="Probability of keeping a frame with no kept detections.",
    )
    parser.add_argument("--train-split", type=float, default=0.7)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--test-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu", help="Ultralytics device (cpu, mps, 0, ...).")
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = build_parser()
    args = parser.parse_args()

    summary = build_species_bootstrap_dataset(
        frame_dirs=args.frame_dirs,
        output_dir=args.out,
        weights=args.weights,
        class_ids=args.class_ids,
        conf=args.conf,
        min_conf=args.min_conf,
        neg_keep_prob=args.neg_keep_prob,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
