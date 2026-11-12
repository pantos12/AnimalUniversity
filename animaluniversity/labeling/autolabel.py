"""Turn archived footage into a YOLO training set with no manual boxes.

A detector (normally the open-vocabulary ``OpenVocabDetector``, prompted with
species names) labels sampled frames; the result is written in the same dataset
layout that ``scripts/au_train.py`` already trains on.

These are *pseudo-labels*: good enough to bootstrap, wrong often enough that a
person should review a sample before training on them. ``review/`` receives a
contact-sheet-friendly copy of each kept frame with its boxes drawn.
"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from animaluniversity.pipeline.sources import FrameSource

logger = logging.getLogger(__name__)


def autolabel_video(
    source: FrameSource,
    detector,
    class_names: Sequence[str],
    out_dir: str | Path,
    min_conf: float = 0.35,
    val_fraction: float = 0.15,
    keep_empty_fraction: float = 0.1,
    prefix: Optional[str] = None,
    seed: int = 13,
    draw_review: bool = True,
) -> Dict[str, object]:
    """Label frames from ``source`` and write/extend a YOLO detection dataset.

    Frames with no detection are mostly skipped; ``keep_empty_fraction`` of them
    are kept as background examples, which reduces false alarms after training.
    Calling this again with another clip and the same ``out_dir`` adds to the
    dataset (use a different ``prefix`` per clip, default is the source name).
    """
    import cv2

    names: List[str] = [c.strip() for c in class_names if c and c.strip()]
    if not names:
        raise ValueError("class_names must not be empty")
    index = {name: i for i, name in enumerate(names)}
    out_dir = Path(out_dir)
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    review_dir = out_dir / "review"
    if draw_review:
        review_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    stem_prefix = prefix or Path(source.info.name).stem or "clip"
    stats = {"frames_seen": 0, "frames_kept": 0, "empty_kept": 0, "boxes": 0}
    per_class = {name: 0 for name in names}

    try:
        for frame in source:
            stats["frames_seen"] += 1
            height, width = frame.image.shape[:2]
            lines: List[str] = []
            kept = []
            for det in detector.detect(frame.image):
                if det.confidence < min_conf or det.label not in index:
                    continue
                x1, y1, x2, y2 = det.bbox_xyxy
                x1, x2 = max(0.0, min(x1, width)), max(0.0, min(x2, width))
                y1, y2 = max(0.0, min(y1, height)), max(0.0, min(y2, height))
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                lines.append(
                    f"{index[det.label]} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
                    f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"
                )
                kept.append((det, (int(x1), int(y1), int(x2), int(y2))))
                per_class[det.label] += 1

            if not lines:
                if rng.random() >= keep_empty_fraction:
                    continue
                stats["empty_kept"] += 1

            split = "val" if rng.random() < val_fraction else "train"
            stem = f"{stem_prefix}_{frame.source_frame if frame.source_frame is not None else frame.index:07d}"
            cv2.imwrite(str(out_dir / "images" / split / f"{stem}.jpg"), frame.image)
            (out_dir / "labels" / split / f"{stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
            stats["frames_kept"] += 1
            stats["boxes"] += len(lines)

            if draw_review and kept:
                canvas = frame.image.copy()
                for det, (x1, y1, x2, y2) in kept:
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 255), 2)
                    cv2.putText(
                        canvas, f"{det.label} {det.confidence:.2f}", (x1, max(14, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA,
                    )
                cv2.imwrite(str(review_dir / f"{stem}.jpg"), canvas)
    finally:
        source.close()

    with (out_dir / "dataset.yaml").open("w", encoding="utf-8") as f:
        f.write(f"path: {out_dir.resolve().as_posix()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(names)}\n")
        f.write("names:\n")
        for name in names:
            f.write(f"  - {name}\n")

    result: Dict[str, object] = {**stats, "boxes_per_class": per_class, "dataset": str(out_dir)}
    logger.info("Auto-label result: %s", result)
    return result
