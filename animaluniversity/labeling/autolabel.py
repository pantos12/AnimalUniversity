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
import hashlib
import json
import math
import random
import re
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
    keep_empty_fraction: float = 0.0,
    prefix: Optional[str] = None,
    seed: int = 13,
    draw_review: bool = True,
    split: Optional[str] = None,
    recording_id: Optional[str] = None,
    split_group: Optional[str] = None,
) -> Dict[str, object]:
    """Label frames from ``source`` and write/extend a YOLO detection dataset.

    Frames with no detection are skipped by default: an open-vocabulary miss is
    not a verified background image. Opt into ``keep_empty_fraction`` only when
    those candidate negatives will be reviewed.
    Splits are assigned to entire recording groups, never neighboring frames.
    Use ``split_group`` to keep clips from the same camera/day together, or
    explicit ``split='train'/'val'`` on independent groups. A single recording
    cannot provide independent training and validation data.

    Dataset class order is immutable; repeated recordings and legacy datasets
    without a dataset_manifest.json are rejected rather than overwritten.
    Use a stable ``recording_id`` when moving footage between machines.
    """
    import cv2

    names: List[str] = [c.strip() for c in class_names if c and c.strip()]
    if not names:
        raise ValueError("class_names must not be empty")
    if len(set(names)) != len(names):
        raise ValueError("class_names must be unique")
    for key, value in [("min_conf", min_conf), ("val_fraction", val_fraction),
                       ("keep_empty_fraction", keep_empty_fraction)]:
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{key} must be between 0 and 1")
    if split not in {None, "train", "val"}:
        raise ValueError("split must be train or val")
    index = {name: i for i, name in enumerate(names)}
    out_dir = Path(out_dir)
    manifest_path = out_dir / "dataset_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("version") != 1 or manifest.get("class_names") != names:
            raise ValueError("Dataset class names/order must match the existing manifest")
        if any(row.get("status") != "complete" for row in manifest["recordings"].values()):
            raise ValueError("Dataset contains an incomplete recording; recover it or use a new output directory")
    else:
        if (out_dir / "dataset.yaml").exists() or any((out_dir / "images").rglob("*.jpg")):
            raise ValueError("Existing dataset has no class/split manifest; use a new output directory")
        manifest = {"version": 1, "class_names": names, "groups": {}, "recordings": {}}
    video_path = getattr(source, "video_path", None)
    identity = recording_id or (str(Path(video_path).resolve()) if video_path else source.info.name)
    source_key = hashlib.sha256(identity.encode()).hexdigest()
    group_key = hashlib.sha256((split_group or identity).encode()).hexdigest()
    if source_key in manifest["recordings"]:
        raise ValueError("Recording already exists in this dataset; use a new output directory to relabel")
    existing_split = manifest["groups"].get(group_key)
    if split is not None and existing_split is not None and split != existing_split:
        raise ValueError("A recording group cannot be assigned to both train and val")
    # Stable across invocations; changing the sample count cannot move a frame
    # or a sibling recording into the other split.
    fraction = int(hashlib.sha256(f"{seed}:{group_key}".encode()).hexdigest()[:16], 16) / 2**64
    assigned_split = existing_split or split or ("val" if fraction < val_fraction else "train")
    for part in ("train", "val"):
        (out_dir / "images" / part).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / part).mkdir(parents=True, exist_ok=True)
    review_dir = out_dir / "review"
    if draw_review:
        review_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    raw_prefix = prefix or Path(source.info.name).stem or "clip"
    stem_prefix = re.sub(r"[^A-Za-z0-9_-]", "_", raw_prefix)[:80] + "_" + source_key[:16]
    stats = {"frames_seen": 0, "frames_kept": 0, "empty_kept": 0, "boxes": 0}
    per_class = {name: 0 for name in names}
    manifest["groups"][group_key] = assigned_split
    manifest["recordings"][source_key] = {"split": assigned_split, "group": group_key, "status": "incomplete"}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

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

            stem = f"{stem_prefix}_{frame.source_frame if frame.source_frame is not None else frame.index:07d}"
            image_path = out_dir / "images" / assigned_split / f"{stem}.jpg"
            label_path = out_dir / "labels" / assigned_split / f"{stem}.txt"
            if image_path.exists() or label_path.exists():
                raise FileExistsError(f"Refusing to overwrite dataset sample: {stem}")
            if not cv2.imwrite(str(image_path), frame.image):
                raise OSError(f"Could not write dataset image: {image_path}")
            label_path.write_text(
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
                if not cv2.imwrite(str(review_dir / f"{stem}.jpg"), canvas):
                    raise OSError("Could not write review image")
    finally:
        source.close()

    with (out_dir / "dataset.yaml").open("w", encoding="utf-8") as f:
        f.write(f"path: {json.dumps(out_dir.resolve().as_posix())}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(names)}\n")
        f.write("names:\n")
        for name in names:
            f.write(f"  - {json.dumps(name)}\n")

    manifest["recordings"][source_key]["status"] = "complete"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if not any((out_dir / "images" / "val").glob("*.jpg")):
        logger.warning("No validation images: add an independent recording group with split='val'")

    result: Dict[str, object] = {**stats, "boxes_per_class": per_class, "dataset": str(out_dir),
                               "split": assigned_split}
    logger.info("Auto-label result: %s", result)
    return result
