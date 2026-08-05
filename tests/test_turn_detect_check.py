"""Pure-logic tests for the Smart Turn real-voice check (no model/audio)."""
from __future__ import annotations

import pytest

from tools.turn_detect_check import (
    _production_detector,
    classify_separability,
    smart_turn_probability,
)


def test_probability_check_delegates_to_production_detector():
    import numpy as np

    class Detector:
        def __init__(self):
            self.calls = []

        def completion_score(self, text, *, samples, sample_rate):
            self.calls.append((text, samples.copy(), sample_rate))
            return 0.625

    detector = Detector()
    samples = np.array([0.1, -0.2, 0.3], dtype="float64")

    assert smart_turn_probability(detector, samples, sample_rate=8_000) == 0.625
    assert len(detector.calls) == 1
    text, passed_samples, sample_rate = detector.calls[0]
    assert text == ""
    assert passed_samples.dtype == np.dtype("float32")
    assert np.array_equal(passed_samples, samples.astype("float32"))
    assert sample_rate == 8_000


def test_production_detector_requires_setup_model(tmp_path):
    with pytest.raises(RuntimeError, match="tools.setup_models --turn-model"):
        _production_detector(str(tmp_path / "missing.onnx"))


def test_production_detector_eagerly_loads_shared_runtime(tmp_path, monkeypatch):
    import core.endpointing as endpointing

    model = tmp_path / "smart-turn.onnx"
    model.write_bytes(b"placeholder")
    instances = []

    class Detector:
        def __init__(self, model_path):
            self.model_path = model_path
            self.loaded = False
            instances.append(self)

        def load(self):
            self.loaded = True

    monkeypatch.setattr(endpointing, "ProsodyTurnCompletionDetector", Detector)

    detector = _production_detector(str(model))

    assert detector is instances[0]
    assert detector.model_path == str(model)
    assert detector.loaded is True


def test_separable_when_complete_all_above_incomplete():
    v = classify_separability([0.9, 0.92, 0.88], [0.4, 0.5, 0.45])
    assert v["verdict"] == "separable"
    assert v["margin"] > 0
    assert v["threshold"] is not None
    # threshold sits between the complete floor (0.88) and incomplete ceiling (0.5)
    assert 0.5 < v["threshold"] < 0.88


def test_partial_when_means_separate_but_groups_overlap():
    # complete mean 0.8, incomplete mean 0.6 (>0.05 apart) but ranges overlap.
    v = classify_separability([0.95, 0.65, 0.80], [0.70, 0.55, 0.55])
    assert v["verdict"] == "partial"
    assert v["margin"] <= 0  # overlap -> no clean gate
    assert v["threshold"] is None


def test_flat_when_means_within_tolerance():
    # The Smart-Turn-on-bad-input case: everything clusters ~0.97.
    v = classify_separability([0.97, 0.98, 0.965], [0.985, 0.96, 0.975])
    assert v["verdict"] == "flat"


def test_carries_group_stats():
    v = classify_separability([0.9, 0.8], [0.4, 0.3])
    assert v["complete_min"] == 0.8 and v["incomplete_max"] == 0.4
    assert v["complete_mean"] == 0.85 and v["incomplete_mean"] == 0.35
