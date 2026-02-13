from __future__ import annotations

import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional


COCO_ANIMAL_CLASS_ID_TO_NAME: Dict[int, str] = {
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
}


def _split_items(items: List[Path], train: float, val: float, test: float) -> Dict[str, List[Path]]:
    if train < 0 or val < 0 or test < 0:
        raise ValueError("Split values must be non-negative.")
    total = train + val + test
    if total <= 0:
        raise ValueError("At least one split must be > 0.")

    n = len(items)
    train_end = int(n * train / total)
    val_end = train_end + int(n * val / total)
    if n > 0 and train > 0 and train_end == 0:
        train_end = 1
        val_end = max(val_end, train_end)
    train_end = min(train_end, n)
    val_end = min(max(val_end, train_end), n)

    return {
        "train": items[:train_end],
        "val": items[train_end:val_end],
        "test": items[val_end:],
    }


def _maybe_link(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_species_bootstrap_dataset(
    frame_dirs: Iterable[str | Path],
    output_dir: str | Path,
    weights: str | Path,
    class_ids: Optional[List[int]] = None,
    conf: float = 0.25,
    min_conf: float = 0.35,
    neg_keep_prob: float = 0.2,
    train_split: float = 0.7,
    val_split: float = 0.2,
    test_split: float = 0.1,
    seed: int = 42,
    imgsz: int = 640,
    batch_size: int = 64,
    device: str = "cpu",
) -> Dict[str, object]:
    from ultralytics import YOLO

    frame_paths: List[Path] = []
    for raw_dir in frame_dirs:
        d = Path(raw_dir)
        if not d.exists():
            continue
        frame_paths.extend(sorted(d.glob("frame_*.jpg")))
    if not frame_paths:
        raise ValueError("No frames found in provided frame_dirs.")

    selected_ids = class_ids[:] if class_ids else sorted(COCO_ANIMAL_CLASS_ID_TO_NAME.keys())
    unknown_ids = [cid for cid in selected_ids if cid not in COCO_ANIMAL_CLASS_ID_TO_NAME]
    if unknown_ids:
        raise ValueError(f"Unsupported class IDs (not in COCO animal set): {unknown_ids}")

    class_names = [COCO_ANIMAL_CLASS_ID_TO_NAME[cid] for cid in selected_ids]
    class_id_to_idx = {cid: i for i, cid in enumerate(selected_ids)}

    output_dir = Path(output_dir)
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    random.shuffle(frame_paths)
    split_map = _split_items(frame_paths, train_split, val_split, test_split)

    frame_to_split = {}
    for split, frames in split_map.items():
        for frame in frames:
            frame_to_split[frame] = split

    model = YOLO(str(weights))

    stats_by_split = {
        "train": {"kept": 0, "positive": 0, "negative": 0},
        "val": {"kept": 0, "positive": 0, "negative": 0},
        "test": {"kept": 0, "positive": 0, "negative": 0},
    }
    class_counter: Counter[str] = Counter()
    dropped_no_label = 0

    for i in range(0, len(frame_paths), batch_size):
        batch = frame_paths[i : i + batch_size]
        results = model.predict(
            source=[str(p) for p in batch],
            conf=conf,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )
        for frame_path, result in zip(batch, results):
            split = frame_to_split[frame_path]
            lines: List[str] = []

            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                xywhn = boxes.xywhn.detach().cpu().tolist()
                confs = boxes.conf.detach().cpu().tolist()
                cls_ids = boxes.cls.detach().cpu().tolist()
                for (x, y, w, h), score, raw_cls in zip(xywhn, confs, cls_ids):
                    cid = int(raw_cls)
                    if cid not in class_id_to_idx:
                        continue
                    if float(score) < min_conf:
                        continue
                    cls_idx = class_id_to_idx[cid]
                    lines.append(f"{cls_idx} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                    class_counter[class_names[cls_idx]] += 1

            is_positive = len(lines) > 0
            if (not is_positive) and (random.random() > neg_keep_prob):
                dropped_no_label += 1
                continue

            stem = f"{frame_path.parent.name}_{frame_path.stem}"
            dst_img = output_dir / "images" / split / f"{stem}{frame_path.suffix}"
            dst_lbl = output_dir / "labels" / split / f"{stem}.txt"
            _maybe_link(frame_path, dst_img)
            with dst_lbl.open("w", encoding="utf-8") as f:
                for line in lines:
                    f.write(line + "\n")

            stats_by_split[split]["kept"] += 1
            if is_positive:
                stats_by_split[split]["positive"] += 1
            else:
                stats_by_split[split]["negative"] += 1

    dataset_yaml = output_dir / "dataset.yaml"
    with dataset_yaml.open("w", encoding="utf-8") as f:
        f.write(f"path: {output_dir.as_posix()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write("names:\n")
        for name in class_names:
            f.write(f"  - {name}\n")

    summary = {
        "dataset_dir": str(output_dir),
        "dataset_yaml": str(dataset_yaml),
        "class_names": class_names,
        "class_counts": dict(class_counter),
        "stats_by_split": stats_by_split,
        "input_frames_total": len(frame_paths),
        "dropped_negative_frames": dropped_no_label,
    }
    summary_path = output_dir / "bootstrap_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
