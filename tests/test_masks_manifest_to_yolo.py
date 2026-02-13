from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from animaluniversity.yolo.dataset import convert_masks_to_yolo


def test_convert_masks_to_yolo_detect(tmp_path: Path) -> None:
    frame_path = tmp_path / "frame_000001.jpg"
    mask_path = tmp_path / "mask_1.png"
    manifest_path = tmp_path / "masks_manifest.jsonl"

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert cv2.imwrite(str(frame_path), frame)

    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:61, 30:81] = 255
    assert cv2.imwrite(str(mask_path), mask)

    record = {
        "frame_path": str(frame_path),
        "frame_index": 0,
        "instance_id": 1,
        "mask_path": str(mask_path),
        "bbox": [30, 20, 80, 60],
        "area": int((mask > 0).sum()),
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    dataset_dir = tmp_path / "dataset"
    dataset_yaml = convert_masks_to_yolo(
        masks_manifest=manifest_path,
        images_dir=None,
        output_dir=dataset_dir,
        mode="detect",
        class_name="animal",
    )

    assert dataset_yaml.exists()
    train_label = dataset_dir / "labels" / "train" / "frame_000001.txt"
    assert train_label.exists()
    parts = train_label.read_text(encoding="utf-8").strip().split()
    assert len(parts) == 5
    assert parts[0] == "0"


def test_convert_masks_to_yolo_seg(tmp_path: Path) -> None:
    frame_path = tmp_path / "frame_000001.jpg"
    mask_path = tmp_path / "mask_1.png"
    manifest_path = tmp_path / "masks_manifest.jsonl"

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert cv2.imwrite(str(frame_path), frame)

    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(mask, (10, 10), (70, 80), color=255, thickness=-1)
    assert cv2.imwrite(str(mask_path), mask)

    record = {
        "frame_path": str(frame_path),
        "frame_index": 0,
        "instance_id": 1,
        "mask_path": str(mask_path),
        "bbox": [10, 10, 70, 80],
        "area": int((mask > 0).sum()),
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    dataset_dir = tmp_path / "dataset"
    convert_masks_to_yolo(
        masks_manifest=manifest_path,
        images_dir=None,
        output_dir=dataset_dir,
        mode="seg",
        class_name="animal",
    )

    train_label = dataset_dir / "labels" / "train" / "frame_000001.txt"
    assert train_label.exists()
    parts = train_label.read_text(encoding="utf-8").strip().split()
    assert len(parts) > 8
    assert parts[0] == "0"
