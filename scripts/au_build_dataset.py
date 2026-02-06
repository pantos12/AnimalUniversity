from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from animaluniversity.yolo.dataset import convert_coco_to_yolo, list_coco_categories


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a YOLO dataset from COCO JSON.")
    parser.add_argument("--coco-json", required=True, help="Path to COCO JSON (e.g., ena24.json).")
    parser.add_argument("--images-dir", required=True, help="Directory containing COCO images.")
    parser.add_argument("--out", required=True, help="Output dataset directory.")
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Optional list of class names to include (case-insensitive).",
    )
    parser.add_argument("--train-split", type=float, default=0.7)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--test-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--list-classes", action="store_true", help="List classes and exit.")
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = build_parser()
    args = parser.parse_args()

    if args.list_classes:
        classes = list_coco_categories(args.coco_json)
        for name in classes:
            print(name)
        return 0

    dataset_yaml = convert_coco_to_yolo(
        coco_json=args.coco_json,
        images_dir=args.images_dir,
        output_dir=args.out,
        class_names=args.classes,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
    )
    logging.info("Dataset ready: %s", dataset_yaml)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
