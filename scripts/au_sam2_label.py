from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from animaluniversity.sam2.sam2_runner import SAM2Runner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SAM2 auto-labeling on frames.")
    parser.add_argument("--checkpoint", required=True, help="Path to SAM2 checkpoint (.pt).")
    parser.add_argument("--frames-dir", required=True, help="Directory of extracted frames.")
    parser.add_argument("--out", required=True, help="Output directory for masks.")
    parser.add_argument("--prompt-box", nargs=4, type=int, default=None, help="x1 y1 x2 y2")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu). Default auto.")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames processed.")
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = build_parser()
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_paths:
        logging.error("No frames found in %s", frames_dir)
        return 1

    prompts = {}
    if args.prompt_box:
        prompts["box"] = args.prompt_box
    else:
        import cv2

        img = cv2.imread(str(frame_paths[0]))
        if img is None:
            logging.error("Failed to read first frame for default prompt.")
            return 1
        h, w = img.shape[:2]
        x1, y1 = int(w * 0.25), int(h * 0.25)
        x2, y2 = int(w * 0.75), int(h * 0.75)
        prompts["box"] = [x1, y1, x2, y2]
        logging.info("No prompt box provided. Using default box: %s", prompts["box"])

    runner = SAM2Runner(model_path=args.checkpoint, device=args.device)
    runner.segment_frames(
        frame_paths=frame_paths,
        prompts=prompts,
        output_dir=args.out,
        max_frames=args.max_frames,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
