"""Prepare a locally runnable YOLOE checkpoint with cached species prompts.

This is the explicit connected setup step. Runtime inference does not need the
text encoder afterwards. Use --boxes-only when the pipeline only consumes boxes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Local YOLOE text-prompt checkpoint.")
    parser.add_argument("--classes", required=True, help="Comma-separated species names.")
    parser.add_argument("--out", required=True, help="Prepared .pt checkpoint; refuses overwrites.")
    parser.add_argument("--allow-download", action="store_true",
                        help="Allow weight/text-encoder downloads during this setup step.")
    parser.add_argument("--boxes-only", action="store_true", help="Remove unused segmentation branch.")
    args = parser.parse_args(argv)
    weights, out = Path(args.weights), Path(args.out)
    if weights.suffix != ".pt":
        parser.error("--weights must be a YOLOE .pt checkpoint")
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    if not classes or len(set(classes)) != len(classes):
        parser.error("--classes must contain distinct, non-empty names")
    if out.suffix != ".pt" or out.exists() or out.with_suffix(".json").exists():
        parser.error("--out must be a new .pt path")
    # Generating new prompts may fetch a text encoder even when .pt is local.
    if not args.allow_download:
        parser.error("Prompt preparation may download a text encoder; explicitly pass --allow-download")
    architecture = None
    if args.boxes_only:
        match = re.fullmatch(r"(yoloe-(?:26|11)[nslmx])-seg", weights.stem)
        if match is None:
            parser.error("--boxes-only requires a standard yoloe-26*/11*-seg.pt checkpoint")
        architecture = match.group(1) + ".yaml"

    import ultralytics
    from ultralytics import YOLOE

    model = YOLOE(architecture).load(str(weights)) if architecture else YOLOE(str(weights))
    model.set_classes(classes)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    metadata = {
        "ultralytics": ultralytics.__version__, "classes": classes,
        "source_checkpoint": weights.name, "boxes_only": args.boxes_only,
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "accuracy_validated": False,
    }
    out.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
