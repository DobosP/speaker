"""Tests for semantic turn-completion endpointing (ask #1: when to speak).

The shipped v1 is a lexical turn-completion detector + an adaptive policy layered
on the acoustic timer, integrated into the sherpa engine via a pure
``_decide_endpoint``. These pin the DECISION logic (no models / no audio); the
real-model latency win is validated on device.
"""
from __future__ import annotations
import statistics

from pathlib import Path

from core.endpointing import (
    AdaptiveEndpointPolicy,
    EndpointConfig,
    LexicalTurnCompletionDetector,
    ProsodyTurnCompletionDetector,
    ScriptedTurnCompletionDetector,
    TurnCompletionDetector,
    _slaney_mel_filters,
)

_SMART_TURN_MODEL = Path("pretrained_models/sherpa/turn/smart-turn-v3.2-cpu.onnx")


# --- prosody detector (Smart Turn) -- pure feature-extraction + edges ---------


def test_slaney_mel_filters_shape_and_nonneg():
    import numpy as np

    mel = _slaney_mel_filters()
    assert mel.shape == (201, 80)
    assert float(np.min(mel)) >= 0.0
    assert float(np.max(mel)) > 0.0  # not all-zero
    # deterministic
    assert np.array_equal(mel, _slaney_mel_filters())


def test_prosody_detector_needs_audio_and_neutral_on_thin_audio():
    import numpy as np

    d = ProsodyTurnCompletionDetector("/no/model.onnx")  # not loaded until first run
    assert d.needs_audio is True
    # No / too-little audio -> 0.5 (no opinion -> the policy uses the acoustic
    # decision). The model is never loaded, so a bad path doesn't matter here.
    assert d.completion_score("hi", samples=None) == 0.5
    assert d.completion_score("hi", samples=np.zeros(1600, dtype="float32")) == 0.5  # 0.1s < 0.3


def test_prosody_logmel_shape_is_whisper_input():
    import numpy as np

    d = ProsodyTurnCompletionDetector("/no/model.onnx")
    d._mel = _slaney_mel_filters()  # _logmel uses it without loading the ONNX
    feats = d._logmel(np.random.default_rng(0).standard_normal(2 * 16000).astype("float32"), 16000)
    assert feats.shape == (1, 80, 800)  # the model's input_features shape
    assert feats.dtype == np.dtype("float32")


# --- prosody detector -- real model (skipped if the ONNX isn't downloaded) ----


def test_prosody_score_in_unit_range_on_real_model():
    import pytest

    if not _SMART_TURN_MODEL.exists():
        pytest.skip("Smart Turn ONNX not downloaded (python -m tools.setup_models --turn-model)")
    import numpy as np

    d = ProsodyTurnCompletionDetector(str(_SMART_TURN_MODEL))
    # noise is enough audio to run; the score must be a valid probability.
    s = d.completion_score("", samples=(np.random.default_rng(0).standard_normal(2 * 16000) * 0.1).astype("float32"))
    assert 0.0 <= s <= 1.0


def test_prosody_separates_recorded_complete_vs_incomplete_if_present():
    import glob
    import os

    import pytest

    if not _SMART_TURN_MODEL.exists():
        pytest.skip("Smart Turn ONNX not downloaded")
    comp = sorted(glob.glob("logs/turn_detect/*/complete_*.npy"))
    inc = sorted(glob.glob("logs/turn_detect/*/incomplete_*.npy"))
    if not comp or not inc:
        pytest.skip("no recorded turn_detect clips (python -m tools.turn_detect_check record ...)")
    import numpy as np

    d = ProsodyTurnCompletionDetector(str(_SMART_TURN_MODEL))
    cs = [d.completion_score("", samples=np.load(p).astype("float32")) for p in comp]
    is_ = [d.completion_score("", samples=np.load(p).astype("float32")) for p in inc]
    # On real human voice the model separates complete (high) from incomplete (low).
    assert statistics.fmean(cs) > statistics.fmean(is_) + 0.1


# --- engine factory: lexical default, prosody fallback ------------------------


def test_engine_builds_lexical_by_default():
    from core.engines.sherpa import SherpaConfig, SherpaOnnxEngine

    d = SherpaOnnxEngine._build_turn_detector(SherpaConfig.from_dict({"endpoint_enabled": True}))
    assert isinstance(d, LexicalTurnCompletionDetector)


def test_engine_prosody_missing_model_falls_back_to_lexical():
    from core.engines.sherpa import SherpaConfig, SherpaOnnxEngine

    cfg = SherpaConfig.from_dict({
        "endpoint_enabled": True, "endpoint_detector": "prosody",
        "endpoint_prosody_model": "/does/not/exist.onnx",
    })
    d = SherpaOnnxEngine._build_turn_detector(cfg)
    assert isinstance(d, LexicalTurnCompletionDetector)  # graceful fallback, no crash


def test_engine_prosody_builds_detector_when_model_present():
    import pytest

    if not _SMART_TURN_MODEL.exists():
        pytest.skip("Smart Turn ONNX not downloaded")
    from core.engines.sherpa import SherpaConfig, SherpaOnnxEngine

    cfg = SherpaConfig.from_dict({
        "endpoint_enabled": True, "endpoint_detector": "prosody",
        "endpoint_prosody_model": str(_SMART_TURN_MODEL),
    })
    d = SherpaOnnxEngine._build_turn_detector(cfg)
    assert isinstance(d, ProsodyTurnCompletionDetector) and d.needs_audio is True


# --- lexical detector --------------------------------------------------------


def test_complete_partials_score_high():
    d = LexicalTurnCompletionDetector()
    for text in (
        "what time is it",
        "what is the weather today",
        "play some music please",
        "what are you waiting for",  # stranded preposition is still complete
        "because i already told you",
    ):
        assert d.completion_score(text) >= 0.6, text


def test_mid_phrase_partials_score_low():
    d = LexicalTurnCompletionDetector()
    for text in (
        "i want to go to the",   # ends in article
        "tell me a joke and",    # ends in conjunction
        "set a reminder but",
        "um",                    # filler
        "the forecast is going to be very nice because",  # subordinator
    ):
        assert d.completion_score(text) <= 0.3, text


def test_short_utterance_is_neutral():
    d = LexicalTurnCompletionDetector()
    assert 0.3 < d.completion_score("hello") < 0.6  # too short to be confident


def test_terminal_subordinators_are_not_flagged_incomplete():
    # 'while'/'though' are commonly terminal ("wait a while", "...though"); a
    # false-incomplete would wrongly EXTEND a finished turn (non-recoverable).
    d = LexicalTurnCompletionDetector()
    assert d.completion_score("wait here for a while") >= 0.6
    assert d.completion_score("i would like that though") >= 0.6


def test_empty_text_is_neutral():
    assert LexicalTurnCompletionDetector().completion_score("") == 0.5


def test_romanian_conjunction_ending_is_incomplete():
    d = LexicalTurnCompletionDetector()
    assert d.completion_score("vreau sa merg si") <= 0.3  # ends in 'si' (and)


def test_detector_satisfies_protocol():
    assert isinstance(LexicalTurnCompletionDetector(), TurnCompletionDetector)
    assert isinstance(ScriptedTurnCompletionDetector(), TurnCompletionDetector)


# --- adaptive policy ---------------------------------------------------------


def _policy(**kw):
    return AdaptiveEndpointPolicy(EndpointConfig(enabled=True, **kw))


def test_policy_shortens_on_confident_complete():
    p = _policy(min_silence_sec=0.4)
    # complete + at least min silence -> end early, even before the acoustic timer
    assert p.decide(acoustic_endpoint=False, completion_score=0.9, silence_sec=0.5) is True
    # ...but not before the minimum settle (min_silence guards against clipping)
    assert p.decide(acoustic_endpoint=False, completion_score=0.9, silence_sec=0.3) is False


def test_policy_high_confidence_floor_lets_complete_turn_commit_earlier():
    # Adaptive confidence-tiered floor: a HIGH-confidence completion (>= 0.75, the
    # lexical "normal ending word" bin) commits at the LOWER 0.55 floor; a medium-
    # confidence complete (0.6 <= score < 0.75) keeps the full 0.7 floor.
    p = _policy(min_silence_sec=0.7, complete_threshold=0.6,
                high_confidence_floor=0.55, high_confidence_score=0.75)
    # 0.75 commits at the lower 0.55 floor...
    assert p.decide(acoustic_endpoint=False, completion_score=0.75, silence_sec=0.55) is True
    # ...but not below it (still guards the decoder lookahead).
    assert p.decide(acoustic_endpoint=False, completion_score=0.75, silence_sec=0.54) is False
    # A medium-confidence complete keeps the 0.7 floor (no early commit at 0.55).
    assert p.decide(acoustic_endpoint=False, completion_score=0.65, silence_sec=0.55) is False
    assert p.decide(acoustic_endpoint=False, completion_score=0.65, silence_sec=0.7) is True


def test_policy_high_confidence_floor_zero_is_uniform_min_silence():
    # Disabled (0.0, the default): even a 0.75 turn must wait the full min_silence
    # floor -- byte-identical to the pre-feature behaviour.
    p = _policy(min_silence_sec=0.7, high_confidence_floor=0.0)
    assert p.decide(acoustic_endpoint=False, completion_score=0.75, silence_sec=0.55) is False
    assert p.decide(acoustic_endpoint=False, completion_score=0.75, silence_sec=0.7) is True


def test_policy_extends_bounded_on_mid_phrase():
    p = _policy(max_silence_sec=1.6)
    # acoustic wants to end, but the partial is mid-phrase + we haven't waited long
    assert p.decide(acoustic_endpoint=True, completion_score=0.1, silence_sec=1.0) is False
    # past the cap, the acoustic decision stands (hard backstop)
    assert p.decide(acoustic_endpoint=True, completion_score=0.1, silence_sec=1.7) is True


def test_policy_falls_back_to_acoustic_when_unsure():
    p = _policy()
    assert p.decide(acoustic_endpoint=True, completion_score=0.5, silence_sec=0.9) is True
    assert p.decide(acoustic_endpoint=False, completion_score=0.5, silence_sec=0.9) is False


def test_endpoint_config_from_sherpa():
    from core.engines.sherpa import SherpaConfig

    cfg = SherpaConfig(endpoint_enabled=True, endpoint_min_silence_sec=0.3)
    ep = EndpointConfig.from_sherpa(cfg)
    assert ep.enabled is True
    assert ep.min_silence_sec == 0.3


# --- engine integration (pure decision; no models / no audio) ----------------


def _engine(detector=None, **cfg_kw):
    from core.engines.sherpa import SherpaConfig, SherpaOnnxEngine

    return SherpaOnnxEngine(SherpaConfig(**cfg_kw), turn_detector=detector)


def test_engine_disabled_is_an_exact_acoustic_passthrough():
    e = _engine()  # endpoint_enabled=False (default)
    for acoustic in (True, False):
        for partial in ("", "i want to go to the", "what time is it"):
            assert e._decide_endpoint(
                acoustic_endpoint=acoustic, partial=partial, silence_sec=5.0
            ) is acoustic


def test_engine_shortens_on_complete_partial():
    det = ScriptedTurnCompletionDetector({"what time is it": 0.9})
    e = _engine(detector=det, endpoint_enabled=True, endpoint_min_silence_sec=0.3)
    assert e._decide_endpoint(
        acoustic_endpoint=False, partial="what time is it", silence_sec=0.4
    ) is True
    assert det.calls == ["what time is it"]


def test_engine_passes_audio_to_an_audio_detector():
    # The seam for a prosodic (Smart Turn) model: samples reach the detector.
    seen = {}

    class _AudioDetector:
        needs_audio = True

        def completion_score(self, text, *, samples=None, sample_rate=16000):
            seen["samples"] = samples
            seen["sr"] = sample_rate
            return 0.9

    e = _engine(detector=_AudioDetector(), endpoint_enabled=True, endpoint_min_silence_sec=0.3)
    e._decide_endpoint(
        acoustic_endpoint=False, partial="hello there", silence_sec=0.4, samples=[0.1, 0.2]
    )
    assert seen["samples"] == [0.1, 0.2]
    assert seen["sr"] == 16000


def test_engine_extends_then_backstops_on_incomplete_partial():
    det = ScriptedTurnCompletionDetector({"and then": 0.05})
    e = _engine(detector=det, endpoint_enabled=True, endpoint_max_silence_sec=1.6)
    assert e._decide_endpoint(acoustic_endpoint=True, partial="and then", silence_sec=1.0) is False
    assert e._decide_endpoint(acoustic_endpoint=True, partial="and then", silence_sec=2.0) is True


def test_engine_empty_partial_skips_the_detector():
    det = ScriptedTurnCompletionDetector(default=0.9)
    e = _engine(detector=det, endpoint_enabled=True)
    assert e._decide_endpoint(acoustic_endpoint=True, partial="   ", silence_sec=0.3) is True
    assert det.calls == []  # never consulted on an empty partial


def test_engine_detector_error_falls_back_to_acoustic():
    class _BoomDetector:
        def completion_score(self, text, *, samples=None, sample_rate=16000):
            raise RuntimeError("model exploded")

    e = _engine(detector=_BoomDetector(), endpoint_enabled=True)
    assert e._decide_endpoint(acoustic_endpoint=True, partial="hello there", silence_sec=0.3) is True


def test_sherpa_config_parses_endpoint_fields_and_ignores_comment():
    from core.engines.sherpa import SherpaConfig

    cfg = SherpaConfig.from_dict(
        {"endpoint_enabled": True, "endpoint_min_silence_sec": 0.25, "_endpoint_comment": "x"}
    )
    assert cfg.endpoint_enabled is True
    assert cfg.endpoint_min_silence_sec == 0.25
