"""Tests for semantic turn-completion endpointing (ask #1: when to speak).

The shipped v1 is a lexical turn-completion detector + an adaptive policy layered
on the acoustic timer, integrated into the sherpa engine via a pure
``_decide_endpoint``. These pin the DECISION logic (no models / no audio); the
real-model latency win is validated on device.
"""
from __future__ import annotations

from core.endpointing import (
    AdaptiveEndpointPolicy,
    EndpointConfig,
    LexicalTurnCompletionDetector,
    ScriptedTurnCompletionDetector,
    TurnCompletionDetector,
)


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
