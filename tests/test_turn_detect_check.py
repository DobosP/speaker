"""Pure-logic tests for the Smart Turn real-voice check (no model/audio)."""
from __future__ import annotations

from tools.turn_detect_check import classify_separability


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
