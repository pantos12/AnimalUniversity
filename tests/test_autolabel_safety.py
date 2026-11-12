import json
from pathlib import Path

import pytest

from animaluniversity.labeling import autolabel_video
from animaluniversity.pipeline import VideoFileSource
from tests.conftest import BrightBlobDetector
from animaluniversity.yolo.train import train_yolo


def label(clip, out, **kwargs):
    return autolabel_video(VideoFileSource(clip, target_fps=1, max_frames=6),
                           BrightBlobDetector(), ["lion"], out, draw_review=False, **kwargs)


def test_recording_does_not_leak_into_both_splits(pacing_video, tmp_path):
    out = tmp_path / "dataset"
    autolabel_video(VideoFileSource(pacing_video, target_fps=1), BrightBlobDetector(), ["lion"], out)
    train, val = list((out / "images/train").iterdir()), list((out / "images/val").iterdir())
    assert bool(train) != bool(val)


def test_extending_dataset_rejects_changed_class_order(pacing_video, tmp_path):
    out = tmp_path / "dataset"
    label(pacing_video, out)
    original = (out / "dataset.yaml").read_bytes()
    with pytest.raises(ValueError, match="class"):
        autolabel_video(VideoFileSource(pacing_video), BrightBlobDetector(), ["tiger", "lion"], out)
    assert (out / "dataset.yaml").read_bytes() == original


def test_recording_group_cannot_cross_split_boundaries(pacing_video, tmp_path):
    out = tmp_path / "dataset"
    label(pacing_video, out, split="train", split_group="camera01-day1", recording_id="clip1")
    with pytest.raises(ValueError, match="group"):
        label(pacing_video, out, split="val", split_group="camera01-day1", recording_id="clip2")


def test_duplicate_run_cannot_overwrite_existing_labels(pacing_video, tmp_path):
    out = tmp_path / "dataset"
    label(pacing_video, out)
    snapshot = {str(p): p.read_bytes() for p in out.rglob("*.txt")}
    with pytest.raises(ValueError, match="already"):
        label(pacing_video, out)
    assert snapshot == {str(p): p.read_bytes() for p in out.rglob("*.txt")}


def test_same_filename_from_different_recordings_does_not_collide(pacing_video, tmp_path):
    out = tmp_path / "dataset"
    label(pacing_video, out, recording_id="clip1", split="train")
    label(pacing_video, out, recording_id="clip2", split="val")
    stems = [p.stem for p in (out / "images").rglob("*.jpg")]
    assert len(stems) == len(set(stems)) == 12
    assert len(json.loads((out / "dataset_manifest.json").read_text())["recordings"]) == 2


def test_image_write_failure_is_not_reported_as_a_success(pacing_video, tmp_path, monkeypatch):
    import cv2

    monkeypatch.setattr(cv2, "imwrite", lambda *_: False)
    with pytest.raises(OSError, match="image"):
        label(pacing_video, tmp_path / "dataset")


@pytest.mark.parametrize("kwargs", [{"val_fraction": -0.1}, {"keep_empty_fraction": 2.0},
                                     {"min_conf": float("nan")}])
def test_invalid_fractions_rejected(pacing_video, tmp_path, kwargs):
    with pytest.raises(ValueError):
        label(pacing_video, tmp_path / "dataset", **kwargs)


def test_single_recording_is_not_enough_for_training(pacing_video, tmp_path):
    out = tmp_path / "dataset"
    label(pacing_video, out, split="train")
    with pytest.raises(ValueError, match="independent"):
        train_yolo(out, weights="not-needed.pt")


def test_incomplete_recording_blocks_training_and_append(pacing_video, tmp_path):
    out = tmp_path / "dataset"
    label(pacing_video, out)
    path = out / "dataset_manifest.json"
    manifest = json.loads(path.read_text())
    next(iter(manifest["recordings"].values()))["status"] = "incomplete"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="incomplete"):
        train_yolo(out, weights="not-needed.pt")
    with pytest.raises(ValueError, match="incomplete"):
        label(pacing_video, out, recording_id="second")
