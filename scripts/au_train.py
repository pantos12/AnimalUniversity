from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from animaluniversity.yolo.train import train_yolo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train YOLO on a dataset.")
    parser.add_argument("--dataset", required=True, help="Path to dataset directory.")
    parser.add_argument("--mode", default="detect", choices=["detect", "seg"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="Device string or auto.")
    parser.add_argument("--weights", required=True, help="Path to YOLO weights (.pt).")
    parser.add_argument("--out", default=None, help="Output directory for runs.")
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = build_parser()
    args = parser.parse_args()

    run_dir = train_yolo(
        dataset_dir=args.dataset,
        mode=args.mode,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        weights=args.weights,
        output_dir=args.out,
    )
    logging.info("Training run saved to: %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
