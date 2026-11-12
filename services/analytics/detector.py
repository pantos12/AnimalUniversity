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
        augment: bool = False,
        nms_iou: float = 0.7,
    ) -> None:
        from ultralytics import YOLO

        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(f"weights not found: {self.weights_path}")
        self._model = YOLO(str(self.weights_path))
        self.conf_threshold = conf_threshold
        self.device = device or _resolve_ultralytics_device()
        self.class_ids = class_ids
        self.imgsz = imgsz
        self.augment = augment
        self.nms_iou = nms_iou

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        results = self._model.predict(
            source=frame_bgr,
            conf=self.conf_threshold,
            iou=self.nms_iou,
            device=self.device,
            classes=self.class_ids,
            imgsz=self.imgsz,
            augment=self.augment,
            verbose=False,
        )
        if not results:
            return []
        return detections_from_result(results[0])


def _to_list(value) -> list:
    """Convert a torch tensor or numpy array to a plain list."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return value.tolist() if hasattr(value, "tolist") else list(value)


def detections_from_result(result) -> List[Detection]:
    """Convert one Ultralytics result into ``Detection`` objects."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    names = getattr(result, "names", None) or {}
    xyxy = _to_list(boxes.xyxy)
    conf = _to_list(boxes.conf)
    cls = _to_list(boxes.cls)
    detections: List[Detection] = []
    for box, score, cls_idx in zip(xyxy, conf, cls):
        cls_idx = int(cls_idx)
        detections.append(
            Detection(
                bbox_xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                confidence=float(score),
                label=str(names.get(cls_idx, cls_idx)),
            )
        )
    return detections


class OpenVocabDetector:
    """Zero-shot detector: find animals by *name*, with no training.

    Wraps Ultralytics YOLOE (open-vocabulary YOLO). Give it the species you care
    about, e.g. ``["tiger", "red panda", "keeper"]``, and it detects them
    directly. Uses:

    - Day-one results on a new enclosure before any site-specific model exists.
    - Automatic labeling of archived footage to build a fine-tuning dataset
      (see ``animaluniversity.labeling.autolabel``).

    A fine-tuned ``YoloDetector`` will still be faster and more accurate on your
    own cameras; this is the bootstrap, not the end state.

    Requires ``ultralytics>=8.4``. Weights download on first use, so the first
    run needs internet access (or place the file under ``models/``).
    """

    DEFAULT_WEIGHTS = "yoloe-26s-seg.pt"

    def __init__(
        self,
        classes: List[str],
        weights_path: Optional[str | Path] = None,
        conf_threshold: float = 0.25,
        device: Optional[str] = None,
        imgsz: int = 640,
        nms_iou: float = 0.7,
    ) -> None:
        classes = [c.strip() for c in classes if c and c.strip()]
        if not classes:
            raise ValueError("OpenVocabDetector needs at least one class name, e.g. ['tiger']")
        try:
            from ultralytics import YOLOE
        except ImportError as exc:
            raise RuntimeError(
                "Open-vocabulary detection needs ultralytics>=8.4 "
                "(pip install -U 'ultralytics>=8.4')."
            ) from exc

        self.classes = classes
        self.weights_path = Path(weights_path) if weights_path else Path(self.DEFAULT_WEIGHTS)
        self._model = YOLOE(str(self.weights_path))
        self._model.set_classes(classes)
        self.conf_threshold = conf_threshold
        self.device = device or _resolve_ultralytics_device()
        self.imgsz = imgsz
        self.nms_iou = nms_iou

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        results = self._model.predict(
            source=frame_bgr,
            conf=self.conf_threshold,
            iou=self.nms_iou,
            device=self.device,
            imgsz=self.imgsz,
            verbose=False,
        )
        if not results:
            return []
        return detections_from_result(results[0])


def _resolve_ultralytics_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"
