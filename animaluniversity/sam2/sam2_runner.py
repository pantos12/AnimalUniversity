from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _select_config_from_checkpoint(ckpt_path: Path) -> str:
    name = ckpt_path.name
    mapping = {
        "sam2.1_hiera_tiny.pt": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "sam2.1_hiera_small.pt": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "sam2.1_hiera_base_plus.pt": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "sam2.1_hiera_large.pt": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "sam2_hiera_tiny.pt": "configs/sam2/sam2_hiera_t.yaml",
        "sam2_hiera_small.pt": "configs/sam2/sam2_hiera_s.yaml",
        "sam2_hiera_base_plus.pt": "configs/sam2/sam2_hiera_b+.yaml",
        "sam2_hiera_large.pt": "configs/sam2/sam2_hiera_l.yaml",
    }
    if name not in mapping:
        raise ValueError(f"Unknown SAM2 checkpoint name: {name}")
    return mapping[name]


def _mask_to_bbox(mask: np.ndarray) -> Optional[list[int]]:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return [x1, y1, x2, y2]


class SAM2Runner:
    """
    Minimal SAM2 runner for auto-labeling with a single box prompt.
    """

    def __init__(self, model_path: str | Path, device: Optional[str] = None):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"SAM2 checkpoint not found: {self.model_path}")
        self.device = device or ("cuda" if _torch_cuda_available() else "cpu")

    def segment_frames(
        self,
        frame_paths: Iterable[str | Path],
        prompts: Optional[dict] = None,
        output_dir: Optional[str | Path] = None,
        max_frames: Optional[int] = None,
    ) -> List[Path]:
        """
        Segment frames using SAM2.

        Inputs:
        - frame_paths: iterable of frame image paths (sorted)
        - prompts: dict containing "box" for first-frame prompting
        - output_dir: directory to write masks
        - max_frames: optional cap on number of frames to process

        Outputs:
        - list of generated mask paths
        """
        try:
            from sam2.build_sam import build_sam2_video_predictor
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "SAM2 package not installed or import failed. Please install SAM2 from source."
            ) from exc

        if prompts is None or "box" not in prompts:
            raise ValueError("prompts must include a 'box' for initial segmentation")

        frame_paths = [Path(p) for p in frame_paths]
        frame_paths = sorted(frame_paths)
        if not frame_paths:
            raise ValueError("No frames provided")

        if max_frames is not None:
            frame_paths = frame_paths[:max_frames]

        output_dir = Path(output_dir) if output_dir else Path("data/masks")
        output_dir.mkdir(parents=True, exist_ok=True)

        config_name = _select_config_from_checkpoint(self.model_path)
        predictor = build_sam2_video_predictor(
            config_file=config_name,
            ckpt_path=str(self.model_path),
            device=self.device,
        )

        video_dir = frame_paths[0].parent
        state = predictor.init_state(
            video_path=str(video_dir),
            offload_video_to_cpu=(self.device == "cpu"),
            offload_state_to_cpu=(self.device == "cpu"),
        )

        box = prompts["box"]
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=0,
            obj_id=1,
            box=box,
        )

        manifest_path = output_dir / "masks_manifest.jsonl"
        mask_paths: List[Path] = []

        with manifest_path.open("w", encoding="utf-8") as mf:
            for frame_idx, obj_ids, masks in predictor.propagate_in_video(
                state, start_frame_idx=0, max_frame_num_to_track=len(frame_paths)
            ):
                frame_path = frame_paths[frame_idx]
                frame_id = frame_path.stem
                frame_out_dir = output_dir / frame_id
                frame_out_dir.mkdir(parents=True, exist_ok=True)

                for i, obj_id in enumerate(obj_ids):
                    mask_logits = masks[i][0].detach().cpu().numpy()
                    mask = mask_logits > 0
                    bbox = _mask_to_bbox(mask)
                    if bbox is None:
                        continue
                    area = int(mask.sum())

                    mask_img = (mask.astype(np.uint8) * 255)
                    mask_path = frame_out_dir / f"{obj_id}.png"
                    cv2.imwrite(str(mask_path), mask_img)
                    mask_paths.append(mask_path)

                    record = {
                        "frame_path": str(frame_path),
                        "frame_index": int(frame_idx),
                        "instance_id": int(obj_id),
                        "mask_path": str(mask_path),
                        "bbox": bbox,
                        "area": area,
                    }
                    mf.write(json.dumps(record) + "\n")

        logger.info("Wrote masks manifest: %s", manifest_path)
        return mask_paths


def _torch_cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False
