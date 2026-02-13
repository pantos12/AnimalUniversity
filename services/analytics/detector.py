"""Detector selection and runtime inference helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

from services.analytics.tracker import Detection


@dataclass(frozen=True)
class RuntimeModel:
    model_choice: str
    weights_path: Path
    plan_note: str
    bootstrap_required: bool = False


def _first_existing(candidates: Iterable[Path]) -> Optional[Path]:
    for path in candidates:
        if path.exists():
            return path
    return None


def resolve_runtime_model(
    model_choice: str,
    weights: Optional[str | Path] = None,
    models_dir: str | Path = "models",
) -> RuntimeModel:
    """Resolve model choice to a concrete runtime checkpoint.

    Choices:
    - ena24: use ENA24-trained YOLO checkpoint
    - sam2: train-first workflow (SAM2 labels -> YOLO runtime)
    - sam3: train-first workflow (currently mapped to SAM2 workflow)
    - custom: user-provided YOLO checkpoint
    """
    models_dir = Path(models_dir)
    choice = model_choice.lower().strip()

    if choice == "custom":
        if weights is None:
            raise FileNotFoundError("--weights is required when --model-choice=custom")
        weights_path = Path(weights)
        if not weights_path.exists():
            raise FileNotFoundError(f"Custom weights not found: {weights_path}")
        return RuntimeModel(
            model_choice=choice,
            weights_path=weights_path,
            plan_note="Using custom YOLO checkpoint for live detection.",
            bootstrap_required=False,
        )

    if choice == "ena24":
        if weights is not None:
            weights_path = Path(weights)
        else:
            candidates = [
                models_dir / "yolo" / "ena24.pt",
                models_dir / "yolo" / "best.pt",
            ]
            weights_path = _first_existing(candidates) or candidates[0]
        if not weights_path.exists():
            raise FileNotFoundError(
                "ENA24 runtime checkpoint not found. Place weights at "
                f"{weights_path} or pass --weights."
            )
        return RuntimeModel(
            model_choice=choice,
            weights_path=weights_path,
            plan_note="Using ENA24/base YOLO checkpoint for immediate live detection.",
            bootstrap_required=False,
        )

    if choice in {"sam2", "sam3"}:
        if weights is not None:
            weights_path = Path(weights)
        else:
            candidates = [
                models_dir / "yolo" / "sam2_finetuned.pt",
                models_dir / "yolo" / "best.pt",
            ]
            weights_path = _first_existing(candidates) or candidates[0]

        if not weights_path.exists():
            sam_variant = "SAM3" if choice == "sam3" else "SAM2"
            raise FileNotFoundError(
                f"{sam_variant} is a train-first path for this pipeline. "
                "Run labeling + training first, then pass the produced YOLO weights "
                f"with --weights (expected default: {weights_path})."
            )

        note = (
            "SAM3 selection currently uses the SAM2-assisted training workflow, then "
            "runs the resulting YOLO checkpoint in production."
            if choice == "sam3"
            else "SAM2-assisted labels converted to YOLO for production runtime."
        )
        return RuntimeModel(
            model_choice=choice,
            weights_path=weights_path,
            plan_note=note,
            bootstrap_required=True,
        )

    raise ValueError(f"Unsupported model choice: {model_choice}")


class YoloDetector:
    """Thin wrapper around Ultralytics YOLO inference."""

    def __init__(
        self,
        weights_path: str | Path,
        conf_threshold: float = 0.25,
        device: Optional[str] = None,
        class_ids: Optional[List[int]] = None,
        imgsz: int = 640,
    ) -> None:
        from ultralytics import YOLO

        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(f"weights not found: {self.weights_path}")
        self._model = YOLO(str(self.weights_path))
        self.conf_threshold = conf_threshold
        self.device = device
        self.class_ids = class_ids
        self.imgsz = imgsz

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        results = self._model.predict(
            source=frame_bgr,
            conf=self.conf_threshold,
            device=self.device or "auto",
            classes=self.class_ids,
            imgsz=self.imgsz,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        boxes = result.boxes
        if boxes is None:
            return []

        names = result.names or {}
        detections: List[Detection] = []
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].detach().cpu().tolist()
            conf = float(boxes.conf[i].detach().cpu().item())
            cls_idx = int(boxes.cls[i].detach().cpu().item())
            label = str(names.get(cls_idx, cls_idx))
            detections.append(
                Detection(
                    bbox_xyxy=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                    confidence=conf,
                    label=label,
                )
            )
        return detections
