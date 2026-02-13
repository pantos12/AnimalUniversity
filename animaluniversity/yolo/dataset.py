from __future__ import annotations

import json
import logging
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def convert_coco_to_yolo(
    coco_json: str | Path,
    images_dir: str | Path,
    output_dir: str | Path,
    class_names: Optional[Iterable[str]] = None,
    train_split: float = 0.7,
    val_split: float = 0.2,
    test_split: float = 0.1,
    seed: int = 42,
    use_hardlinks: bool = True,
) -> Path:
    """
    Convert COCO detection annotations to YOLO format.

    Outputs:
    - output_dir/images/{train,val,test}
    - output_dir/labels/{train,val,test}
    - output_dir/dataset.yaml
    """
    coco_json = Path(coco_json)
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with coco_json.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    categories = {c["id"]: c["name"] for c in coco.get("categories", [])}
    if class_names:
        class_names = [c.lower() for c in class_names]
        cat_ids = {cid for cid, name in categories.items() if name.lower() in class_names}
        names = [name for cid, name in categories.items() if cid in cat_ids]
    else:
        cat_ids = set(categories.keys())
        names = [categories[cid] for cid in sorted(categories.keys())]

    if not names:
        raise ValueError("No categories matched the provided class_names filter.")

    cat_id_to_idx = {cid: idx for idx, cid in enumerate(sorted(cat_ids))}

    images = {img["id"]: img for img in coco.get("images", [])}
    annotations_by_image: dict[int, list[dict]] = {}
    for ann in coco.get("annotations", []):
        if ann.get("category_id") not in cat_ids:
            continue
        annotations_by_image.setdefault(ann["image_id"], []).append(ann)

    image_ids = list(images.keys())
    random.Random(seed).shuffle(image_ids)
    total = len(image_ids)
    train_end = int(total * train_split)
    val_end = train_end + int(total * val_split)
    splits = {
        "train": image_ids[:train_end],
        "val": image_ids[train_end:val_end],
        "test": image_ids[val_end:],
    }

    for split_name in splits.keys():
        (output_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    for split_name, ids in splits.items():
        for image_id in ids:
            img = images[image_id]
            img_file = images_dir / img["file_name"]
            if not img_file.exists():
                continue
            dest_img = output_dir / "images" / split_name / img_file.name
            if not dest_img.exists():
                try:
                    if use_hardlinks:
                        os.link(img_file, dest_img)
                    else:
                        shutil.copy2(img_file, dest_img)
                except OSError:
                    shutil.copy2(img_file, dest_img)

            label_path = output_dir / "labels" / split_name / f"{img_file.stem}.txt"
            width = float(img["width"])
            height = float(img["height"])
            anns = annotations_by_image.get(image_id, [])
            with label_path.open("w", encoding="utf-8") as lf:
                for ann in anns:
                    x, y, w, h = ann["bbox"]
                    xc = (x + w / 2.0) / width
                    yc = (y + h / 2.0) / height
                    wn = w / width
                    hn = h / height
                    cls = cat_id_to_idx[ann["category_id"]]
                    lf.write(f"{cls} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")

    dataset_yaml = output_dir / "dataset.yaml"
    with dataset_yaml.open("w", encoding="utf-8") as f:
        f.write(f"path: {output_dir.as_posix()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n")
        f.write(f"nc: {len(names)}\n")
        f.write("names:\n")
        for name in names:
            f.write(f"  - {name}\n")

    logger.info("COCO -> YOLO conversion done. Dataset: %s", dataset_yaml)
    return dataset_yaml


def list_coco_categories(coco_json: str | Path) -> list[str]:
    coco_json = Path(coco_json)
    with coco_json.open("r", encoding="utf-8") as f:
        coco = json.load(f)
    return [c["name"] for c in coco.get("categories", [])]


def convert_masks_to_yolo(
    masks_manifest: str | Path,
    images_dir: Optional[str | Path],
    output_dir: str | Path,
    mode: Literal["detect", "seg"] = "detect",
    class_name: str = "animal",
    train_split: float = 0.7,
    val_split: float = 0.2,
    test_split: float = 0.1,
    seed: int = 42,
    use_hardlinks: bool = True,
) -> Path:
    """Convert SAM2 mask manifest into a YOLO dataset."""
    if mode not in {"detect", "seg"}:
        raise ValueError("mode must be 'detect' or 'seg'")
    if train_split < 0 or val_split < 0 or test_split < 0:
        raise ValueError("split values must be non-negative")

    total_split = train_split + val_split + test_split
    if total_split <= 0:
        raise ValueError("At least one split must be > 0")

    masks_manifest = Path(masks_manifest)
    if not masks_manifest.exists():
        raise FileNotFoundError(f"Mask manifest not found: {masks_manifest}")

    images_dir_path = Path(images_dir) if images_dir else None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_frame: Dict[str, List[dict]] = defaultdict(list)
    with masks_manifest.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            frame_path = row.get("frame_path")
            if not frame_path:
                continue
            rows_by_frame[str(frame_path)].append(row)

    frame_paths = sorted(rows_by_frame.keys())
    if not frame_paths:
        raise ValueError(f"No records found in mask manifest: {masks_manifest}")

    random.Random(seed).shuffle(frame_paths)
    total = len(frame_paths)
    train_end = int(total * train_split / total_split)
    val_end = train_end + int(total * val_split / total_split)
    if total > 0 and train_split > 0 and train_end == 0:
        train_end = 1
        val_end = max(val_end, train_end)
    train_end = min(train_end, total)
    val_end = min(max(val_end, train_end), total)
    splits = {
        "train": frame_paths[:train_end],
        "val": frame_paths[train_end:val_end],
        "test": frame_paths[val_end:],
    }

    for split_name in splits:
        (output_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    for split_name, split_frames in splits.items():
        for frame_path_raw in split_frames:
            frame_path = Path(frame_path_raw)
            if not frame_path.is_absolute() and images_dir_path is not None:
                frame_path = images_dir_path / frame_path
            if not frame_path.exists():
                logger.warning("Skipping missing frame: %s", frame_path)
                continue

            dest_img = output_dir / "images" / split_name / frame_path.name
            if not dest_img.exists():
                try:
                    if use_hardlinks:
                        os.link(frame_path, dest_img)
                    else:
                        shutil.copy2(frame_path, dest_img)
                except OSError:
                    shutil.copy2(frame_path, dest_img)

            image = cv2.imread(str(frame_path))
            if image is None:
                logger.warning("Skipping unreadable frame: %s", frame_path)
                continue
            height, width = image.shape[:2]

            label_path = output_dir / "labels" / split_name / f"{frame_path.stem}.txt"
            anns = rows_by_frame[frame_path_raw]
            with label_path.open("w", encoding="utf-8") as lf:
                for ann in anns:
                    bbox = ann.get("bbox")
                    if bbox and len(bbox) == 4:
                        x1, y1, x2, y2 = [float(v) for v in bbox]
                    else:
                        mask_path = Path(ann.get("mask_path", ""))
                        if not mask_path.is_absolute() and images_dir_path is not None:
                            mask_path = images_dir_path / mask_path
                        if not mask_path.exists():
                            continue
                        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                        if mask is None:
                            continue
                        ys, xs = np.where(mask > 0)
                        if len(xs) == 0 or len(ys) == 0:
                            continue
                        x1, y1, x2, y2 = float(xs.min()), float(ys.min()), float(xs.max()), float(
                            ys.max()
                        )

                    if mode == "detect":
                        xc = ((x1 + x2) / 2.0) / float(width)
                        yc = ((y1 + y2) / 2.0) / float(height)
                        wn = (x2 - x1) / float(width)
                        hn = (y2 - y1) / float(height)
                        if wn <= 0 or hn <= 0:
                            continue
                        lf.write(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")
                    else:
                        contour = _mask_contour_for_yolo(
                            ann=ann,
                            images_dir=images_dir_path,
                            width=width,
                            height=height,
                        )
                        if contour:
                            lf.write("0 " + " ".join(f"{v:.6f}" for v in contour) + "\n")
                        else:
                            # bbox polygon fallback for segmentation mode
                            poly = _bbox_to_polygon(
                                x1=x1,
                                y1=y1,
                                x2=x2,
                                y2=y2,
                                width=width,
                                height=height,
                            )
                            lf.write("0 " + " ".join(f"{v:.6f}" for v in poly) + "\n")

    dataset_yaml = output_dir / "dataset.yaml"
    with dataset_yaml.open("w", encoding="utf-8") as f:
        f.write(f"path: {output_dir.as_posix()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n")
        f.write("nc: 1\n")
        f.write("names:\n")
        f.write(f"  - {class_name}\n")

    logger.info("Mask manifest -> YOLO conversion done. Dataset: %s", dataset_yaml)
    return dataset_yaml


def _mask_contour_for_yolo(
    ann: dict,
    images_dir: Optional[Path],
    width: int,
    height: int,
) -> List[float]:
    mask_path = Path(ann.get("mask_path", ""))
    if not mask_path.is_absolute() and images_dir is not None:
        mask_path = images_dir / mask_path
    if not mask_path.exists():
        return []

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []
    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        return []

    polygon: List[float] = []
    for point in contour:
        x, y = point[0]
        polygon.append(float(x) / float(width))
        polygon.append(float(y) / float(height))
    return polygon


def _bbox_to_polygon(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
) -> List[float]:
    return [
        x1 / float(width),
        y1 / float(height),
        x2 / float(width),
        y1 / float(height),
        x2 / float(width),
        y2 / float(height),
        x1 / float(width),
        y2 / float(height),
    ]
