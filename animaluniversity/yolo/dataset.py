from __future__ import annotations

import json
import logging
import os
import random
import shutil
from pathlib import Path
from typing import Iterable, Literal, Optional

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
    output_dir: str | Path,
    mode: Literal["detect", "seg"] = "detect",
    class_name: str = "animal",
    train_split: float = 0.7,
    val_split: float = 0.2,
    test_split: float = 0.1,
) -> None:
    """
    Convert mask manifests into a YOLO dataset.

    TODO: Implement mask -> bbox/seg conversion for SAM2 output.
    """
    raise NotImplementedError(
        "Mask-to-YOLO conversion not implemented yet. "
        "Implement in animaluniversity/yolo/dataset.py."
    )
