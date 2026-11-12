import sys
from types import SimpleNamespace

import numpy as np
import pytest

from services.analytics.detector import OpenVocabDetector, YoloDetector


class FakeModel:
    def __init__(self, path):
        self.names = {0: "0"}
        self.prompts = 0
        self.calls = []

    def set_classes(self, classes):
        self.prompts += 1
        self.names = dict(enumerate(classes))

    def load_prompt_embeddings(self, path):
        self.names = {0: "lion"}

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        images = kwargs["source"]
        images = images if isinstance(images, list) else [images]
        return [SimpleNamespace(boxes=None) for _ in images]


def test_missing_weights_rejected_before_importing_model_package(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "ultralytics", None)
    with pytest.raises(FileNotFoundError, match="prepare"):
        OpenVocabDetector(["lion"], tmp_path / "missing.pt")


def test_offline_unprepared_prompts_are_rejected(tmp_path, monkeypatch):
    weights = tmp_path / "model.pt"
    weights.touch()
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLOE=FakeModel))
    with pytest.raises(ValueError, match="prompt"):
        OpenVocabDetector(["lion"], weights, device="cpu")


def test_cached_prompts_skip_text_encoder_and_support_batches(tmp_path, monkeypatch):
    weights, prompts = tmp_path / "model.pt", tmp_path / "classes.npz"
    weights.touch()
    prompts.touch()
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLOE=FakeModel))
    detector = OpenVocabDetector(["lion"], weights, device="cpu", prompt_embeddings=prompts)
    assert detector._model.prompts == 0
    assert detector.detect_batch([np.zeros((4, 4, 3))] * 3) == [[], [], []]
    assert len(detector._model.calls) == 1


def test_prompt_class_order_must_match_request(tmp_path, monkeypatch):
    weights, prompts = tmp_path / "model.pt", tmp_path / "classes.npz"
    weights.touch()
    prompts.touch()
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLOE=FakeModel))
    with pytest.raises(ValueError, match="classes"):
        OpenVocabDetector(["tiger"], weights, prompt_embeddings=prompts)


def test_trained_detector_batches_without_changing_single_frame_api(tmp_path, monkeypatch):
    weights = tmp_path / "model.onnx"
    weights.touch()
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeModel))
    detector = YoloDetector(weights, device="cpu")
    assert detector.detect(np.zeros((4, 4, 3))) == []
    assert detector.detect_batch([np.zeros((4, 4, 3))] * 2) == [[], []]
    assert detector.detect_batch([]) == []
