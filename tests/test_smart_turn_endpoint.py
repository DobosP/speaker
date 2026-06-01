"""Tests for the Smart Turn v3 endpoint seam (core/endpointing.py + the sherpa
config plumbing). The shape-adaptation + config tests need no model or deps and
always run; the real-inference test self-skips unless onnxruntime + transformers
+ the downloaded model are present (so CI / fresh checkouts stay green)."""
from __future__ import annotations

import os

import numpy as np
import pytest

from core.endpointing import (
    AdaptiveEndpointPolicy,
    EndpointConfig,
    match_onnx_input_shape,
)
from core.engines.sherpa import SherpaConfig


# --- shape adaptation (pure, no model) ---------------------------------------


def test_match_shape_noop_when_dynamic_batch_and_mels_first():
    # The real v3.1-cpu export declares ('s6', 80, 800): mels already last-1,
    # frames last -> our (1, 80, 800) matches, no transpose.
    feats = np.zeros((1, 80, 800), dtype=np.float32)
    out = match_onnx_input_shape(feats, ("s6", 80, 800))
    assert out.shape == (1, 80, 800)


def test_match_shape_transposes_when_export_wants_mels_last():
    feats = np.zeros((1, 80, 800), dtype=np.float32)
    out = match_onnx_input_shape(feats, (1, 800, 80))  # frames, then mels
    assert out.shape == (1, 800, 80)


def test_match_shape_leaves_alone_when_already_matching():
    feats = np.zeros((1, 800, 80), dtype=np.float32)
    out = match_onnx_input_shape(feats, (1, 800, 80))
    assert out.shape == (1, 800, 80)


def test_match_shape_ignores_non_3d():
    feats = np.zeros((80, 800), dtype=np.float32)
    assert match_onnx_input_shape(feats, (80, 800)).shape == (80, 800)


# --- config plumbing ---------------------------------------------------------


def test_sherpa_config_reads_smart_turn_fields():
    c = SherpaConfig.from_dict(
        {"smart_turn_enabled": True, "smart_turn_model": "/models/smart-turn.onnx"}
    )
    assert c.smart_turn_enabled is True
    assert c.smart_turn_model == "/models/smart-turn.onnx"


def test_sherpa_config_smart_turn_defaults_off():
    c = SherpaConfig.from_dict({})
    assert c.smart_turn_enabled is False
    assert c.smart_turn_model == ""


# --- adaptive policy uses a high prosodic score the same as a lexical one -----


def test_policy_commits_early_on_high_completion_score():
    # The detector's output (lexical OR Smart Turn) flows through the SAME policy,
    # so a confident-complete prosodic score commits early just like a lexical one.
    policy = AdaptiveEndpointPolicy(
        EndpointConfig(enabled=True, min_silence_sec=0.5, complete_threshold=0.6)
    )
    assert policy.decide(acoustic_endpoint=False, completion_score=0.98, silence_sec=0.6) is True
    assert policy.decide(acoustic_endpoint=False, completion_score=0.1, silence_sec=0.6) is False


# --- real inference (skips unless deps + model are present) -------------------

_MODEL = os.path.join("pretrained_models", "sherpa", "smart_turn", "smart-turn-v3.1-cpu.onnx")


@pytest.mark.skipif(
    not os.path.exists(_MODEL),
    reason="Smart Turn model not downloaded (run tools.setup_models --smart-turn-model)",
)
def test_smart_turn_inference_returns_probability():
    pytest.importorskip("onnxruntime")
    pytest.importorskip("transformers")
    from core.endpointing import SmartTurnCompletionDetector

    d = SmartTurnCompletionDetector(_MODEL)
    assert d.needs_audio is True
    audio = (0.1 * np.sin(2 * np.pi * 150 * np.linspace(0, 1.5, 24000))).astype(np.float32)
    score = d.completion_score("what is the capital of france", samples=audio, sample_rate=16000)
    assert 0.0 <= score <= 1.0
    assert d.completion_score("no audio", samples=None) == 0.5  # neutral fallback
