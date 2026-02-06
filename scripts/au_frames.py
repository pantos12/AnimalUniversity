from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from animaluniversity.core.video import extract_frames, get_video_metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract frames from a video.")
    parser.add_argument("--video", required=False, help="Path to input video.")
    parser.add_argument("--fps", type=int, default=1, help="Frames per second to sample.")
    parser.add_argument("--out", required=False, help="Output directory for frames.")
    parser.add_argument("--start-sec", type=float, default=0, help="Start time in seconds.")
    parser.add_argument("--end-sec", type=float, default=None, help="End time in seconds.")
    parser.add_argument(
        "--no-ffmpeg",
        action="store_true",
        help="Disable ffmpeg and use OpenCV extractor (streaming-friendly).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after N frames (useful for quick tests).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing manifest and continue extracting frames.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing frames and manifest.",
    )
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = build_parser()
    args = parser.parse_args()

    if not args.video or not args.out:
        parser.print_help()
        logging.error("Both --video and --out are required.")
        return 2

    video_path = Path(args.video)
    out_dir = Path(args.out)

    if not video_path.exists():
        logging.error("Video not found: %s", video_path)
        return 1

    metadata = get_video_metadata(video_path)
    logging.info("Video metadata: %s", metadata)

    manifest = extract_frames(
        video_path=video_path,
        out_dir=out_dir,
        fps=args.fps,
        start_sec=args.start_sec,
        end_sec=args.end_sec,
        overwrite=args.overwrite,
        use_ffmpeg=not args.no_ffmpeg,
        max_frames=args.max_frames,
        resume=args.resume,
    )

    manifest_path = out_dir / "manifest.jsonl"
    logging.info("Extracted %d frames. Manifest: %s", len(manifest), manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
