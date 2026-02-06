from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional


def train_yolo(
    dataset_dir: str | Path,
    mode: Literal["detect", "seg"] = "detect",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    device: Optional[str] = None,
    weights: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
) -> Path:
    """
    Train YOLO on a dataset.

    Inputs:
    - dataset_dir: root dataset directory
    - mode: detect or seg
    - epochs/imgsz/batch/device: training params
    - output_dir: run directory

    Outputs:
    - path to run artifacts directory
    """
    from ultralytics import YOLO

    dataset_dir = Path(dataset_dir)
    if output_dir is None:
        output_dir = dataset_dir / "runs"

    if weights is None:
        raise FileNotFoundError(
            "weights is required. Place a model checkpoint under models/ and pass --weights."
        )

    weights = Path(weights)
    if not weights.exists():
        raise FileNotFoundError(f"weights not found: {weights}")

    model = YOLO(str(weights))
    result = model.train(
        data=str(dataset_dir / "dataset.yaml"),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device or "auto",
        project=str(output_dir),
        name=f"yolo_{mode}",
        verbose=True,
    )

    return Path(result.save_dir)
