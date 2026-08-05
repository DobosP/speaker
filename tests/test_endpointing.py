"""Tests for semantic turn-completion endpointing (ask #1: when to speak).

The shipped v1 is a lexical turn-completion detector + an adaptive policy layered
on the acoustic timer, integrated into the sherpa engine via a pure
``_decide_endpoint``. These pin the DECISION logic (no models / no audio); the
optional real-model contract has a separate marker, and live impact remains a
separate owner gate.
"""
from __future__ import annotations

import os
from pathlib import Path
import statistics
import warnings

import pytest

from core.endpointing import (
    AdaptiveEndpointPolicy,
    CompletionScoreState,
    EndpointConfig,
    LexicalTurnCompletionDetector,
    ProsodyTurnCompletionDetector,
    ScriptedTurnCompletionDetector,
    SessionPauseModel,
    TurnCompletionDetector,
    _slaney_mel_filters,
    evaluate_turn_completion,
    observe_turn_completion,
)

_SMART_TURN_MODEL = Path(
    os.environ.get(
        "SPEAKER_SMART_TURN_MODEL",
        "pretrained_models/sherpa/turn/smart-turn-v3.2-cpu.onnx",
    )
)


# --- prosody detector (Smart Turn) -- pure feature-extraction + edges ---------


def test_slaney_mel_filters_shape_and_nonneg():
    import numpy as np

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        mel = _slaney_mel_filters()
    assert mel.shape == (201, 80)
    assert float(np.min(mel)) >= 0.0
    assert float(np.max(mel)) > 0.0  # not all-zero
    # deterministic
    assert np.array_equal(mel, _slaney_mel_filters())


def _upstream_smart_turn_logmel(samples):
    """Independent Smart Turn/Transformers reference for deterministic parity.

    The ordering is pinned by ``audio_utils.py`` + ``inference.py`` at
    pipecat-ai/smart-turn@4786657e242dfe77dd138699ac564ee074a2a543 and the
    vendored numpy implementation at
    pipecat-ai/pipecat@0db3c9a0a8d4982c997afd073a3f6372e9d44515.
    """
    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view

    waveform = np.asarray(samples, dtype="float32").reshape(-1)[-128_000:]
    if waveform.size < 128_000:
        waveform = np.pad(waveform, (128_000 - waveform.size, 0))
    waveform = (waveform - waveform.mean()) / np.sqrt(waveform.var() + 1e-7)

    centered = np.pad(waveform.astype("float64"), (200, 200), mode="reflect")
    window = np.hanning(401)[:-1]
    frames = sliding_window_view(centered, 400)[::160]
    magnitudes = (np.abs(np.fft.rfft(frames * window, axis=-1)) ** 2).T
    mel = np.maximum(1e-10, _slaney_mel_filters().T @ magnitudes)
    log_mel = np.log10(mel)[:, :-1]
    log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
    return ((log_mel + 4.0) / 4.0)[None].astype("float32")


@pytest.mark.parametrize("duration_sec", [0.5, 2.0, 7.0, 8.0, 9.0])
def test_prosody_logmel_matches_upstream_end_aligned_contract(duration_sec):
    import numpy as np

    sample_count = int(duration_sec * 16_000)
    time = np.arange(sample_count, dtype="float32") / 16_000
    envelope = np.linspace(0.35, 1.0, sample_count, dtype="float32")
    samples = envelope * (
        0.18 * np.sin(2 * np.pi * 197 * time)
        + 0.04 * np.cos(2 * np.pi * 733 * time)
    )

    detector = ProsodyTurnCompletionDetector("/no/model.onnx")
    detector._mel = _slaney_mel_filters()
    actual = detector._logmel(samples, 16_000)
    expected = _upstream_smart_turn_logmel(samples)

    assert actual.shape == expected.shape == (1, 80, 800)
    assert np.allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_prosody_logmel_matches_fixed_official_transformers_probe():
    """Pin independent WhisperFeatureExtractor output without that dependency.

    The constants were generated from the official extractor using the exact
    Smart Turn invocation in upstream ``inference.py`` and this deterministic
    two-second fixture. In particular, the last-frame sum catches a regression
    that moves short-turn speech from the end to the beginning of the tensor.
    """
    import numpy as np

    sample_count = 2 * 16_000
    time = np.arange(sample_count, dtype="float32") / 16_000
    envelope = np.linspace(0.35, 1.0, sample_count, dtype="float32")
    samples = envelope * (
        0.18 * np.sin(2 * np.pi * 197 * time)
        + 0.04 * np.cos(2 * np.pi * 733 * time)
    )
    detector = ProsodyTurnCompletionDetector("/no/model.onnx")
    detector._mel = _slaney_mel_filters()
    features = detector._logmel(samples, 16_000)

    actual = np.array(
        [
            features[0, 0, 0],
            features[0, 17, 137],
            features[0, 79, 799],
            features[0, :, 0].sum(),
            features[0, :, 400].sum(),
            features[0, :, 799].sum(),
        ],
        dtype="float32",
    )
    expected = np.array(
        [-0.1325642, -0.1325642, -0.1325642, -10.605135, -10.605135, 30.5976505],
        dtype="float32",
    )
    assert np.allclose(actual, expected, rtol=0.0, atol=1e-4)


def test_prosody_detector_needs_audio_and_neutral_on_thin_audio():
    import numpy as np

    d = ProsodyTurnCompletionDetector("/no/model.onnx")  # not loaded until first run
    assert d.needs_audio is True
    # No / too-little audio -> 0.5 (no opinion -> the policy uses the acoustic
    # decision). The model is never loaded, so a bad path doesn't matter here.
    assert d.completion_score("hi", samples=None) == 0.5
    assert d.completion_score("hi", samples=np.zeros(1600, dtype="float32")) == 0.5  # 0.1s < 0.3


@pytest.mark.parametrize("samples", [None, [0.0]])
def test_prosody_rejects_non_16k_before_neutral_audio_return(samples):
    detector = ProsodyTurnCompletionDetector("/no/model.onnx")

    with pytest.raises(ValueError, match="caller-provided 16 kHz"):
        detector.completion_score("", samples=samples, sample_rate=48_000)


def test_prosody_logmel_shape_is_whisper_input():
    import numpy as np

    d = ProsodyTurnCompletionDetector("/no/model.onnx")
    d._mel = _slaney_mel_filters()  # _logmel uses it without loading the ONNX
    feats = d._logmel(np.random.default_rng(0).standard_normal(2 * 16000).astype("float32"), 16000)
    assert feats.shape == (1, 80, 800)  # the model's input_features shape
    assert feats.dtype == np.dtype("float32")


@pytest.mark.parametrize("probability", [0.0, 0.25, 1.0])
def test_prosody_accepts_one_finite_probability(probability):
    assert ProsodyTurnCompletionDetector._probability([[probability]]) == probability


@pytest.mark.parametrize(
    ("outputs", "message"),
    [
        ([], "exactly one output tensor"),
        ([[0.25], [0.75]], "exactly one output tensor"),
        ([[]], "exactly one probability"),
        ([[0.25, 0.75]], "exactly one probability"),
        ([[True]], "floating-point dtype"),
        ([[1]], "floating-point dtype"),
        ([["0.5"]], "floating-point dtype"),
        ([[0.5 + 0j]], "floating-point dtype"),
        ([[float("nan")]], "finite probability"),
        ([[float("inf")]], "finite probability"),
        ([[-0.01]], "finite probability"),
        ([[1.01]], "finite probability"),
    ],
)
def test_prosody_rejects_invalid_onnx_output(outputs, message):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match=message):
            ProsodyTurnCompletionDetector._probability(outputs)


def test_prosody_load_exercises_input_and_output_contract():
    import numpy as np

    class Session:
        def __init__(self):
            self.inputs = []

        def run(self, _outputs, inputs):
            self.inputs.append(inputs)
            return [[0.75]]

    detector = ProsodyTurnCompletionDetector("/no/model.onnx")
    detector._session = Session()
    detector._mel = _slaney_mel_filters()
    detector.load()

    assert len(detector._session.inputs) == 1
    features = detector._session.inputs[0]["input_features"]
    assert features.shape == (1, 80, 800)
    assert features.dtype == np.dtype("float32")
    assert np.count_nonzero(features) == 0


# --- prosody detector -- real model (skipped if the ONNX isn't downloaded) ----


@pytest.mark.real_model
def test_prosody_score_in_unit_range_on_real_model():
    if not _SMART_TURN_MODEL.exists():
        pytest.skip("Smart Turn ONNX not downloaded (python -m tools.setup_models --turn-model)")
    import numpy as np

    d = ProsodyTurnCompletionDetector(str(_SMART_TURN_MODEL))
    d.load()
    # noise is enough audio to run; the score must be a valid probability.
    s = d.completion_score("", samples=(np.random.default_rng(0).standard_normal(2 * 16000) * 0.1).astype("float32"))
    assert 0.0 <= s <= 1.0


@pytest.mark.real_model
def test_prosody_separates_recorded_complete_vs_incomplete_if_present():
    import glob

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


@pytest.mark.parametrize(
    "sample_rate",
    [48_000, True, "16000", "bad", None, float("nan"), float("inf"), 16_000.9],
)
def test_engine_invalid_prosody_rate_falls_back_before_load(
    tmp_path, monkeypatch, sample_rate
):
    from types import SimpleNamespace

    from core.engines.sherpa import SherpaOnnxEngine

    model = tmp_path / "smart-turn.onnx"
    model.write_bytes(b"test placeholder")
    loads = []
    monkeypatch.setattr(
        ProsodyTurnCompletionDetector,
        "load",
        lambda detector: loads.append(detector._model_path),
    )
    cfg = SimpleNamespace(
        sample_rate=sample_rate,
        endpoint_enabled=True,
        endpoint_detector="prosody",
        endpoint_prosody_model=str(model),
        endpoint_prosody_threads=1,
    )

    detector = SherpaOnnxEngine._build_turn_detector(cfg)

    assert isinstance(detector, LexicalTurnCompletionDetector)
    assert loads == []


@pytest.mark.parametrize("sample_rate", [16_000, 16_000.0])
def test_engine_prosody_eagerly_loads_valid_model(
    tmp_path, monkeypatch, sample_rate
):
    from types import SimpleNamespace

    from core.engines.sherpa import SherpaOnnxEngine

    model = tmp_path / "smart-turn.onnx"
    model.write_bytes(b"test placeholder")
    loads = []
    monkeypatch.setattr(
        ProsodyTurnCompletionDetector,
        "load",
        lambda detector: loads.append(detector._model_path),
    )
    cfg = SimpleNamespace(
        sample_rate=sample_rate,
        endpoint_enabled=True,
        endpoint_detector="prosody",
        endpoint_prosody_model=str(model),
        endpoint_prosody_threads=1,
    )

    detector = SherpaOnnxEngine._build_turn_detector(cfg)

    assert isinstance(detector, ProsodyTurnCompletionDetector)
    assert loads == [str(model)]


def test_engine_corrupt_prosody_falls_back_once_before_capture(tmp_path, monkeypatch):
    from core.engines.sherpa import SherpaConfig, SherpaOnnxEngine

    model = tmp_path / "corrupt.onnx"
    model.write_bytes(b"not an ONNX model")
    attempts = []

    def fail_load(detector):
        attempts.append(detector._model_path)
        raise ValueError("bad model")

    monkeypatch.setattr(ProsodyTurnCompletionDetector, "load", fail_load)
    cfg = SherpaConfig.from_dict(
        {
            "endpoint_enabled": True,
            "endpoint_detector": "prosody",
            "endpoint_prosody_model": str(model),
        }
    )

    detector = SherpaOnnxEngine._build_turn_detector(cfg)

    assert isinstance(detector, LexicalTurnCompletionDetector)
    assert attempts == [str(model)]


@pytest.mark.real_model
def test_engine_prosody_builds_detector_when_model_present():
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


def test_engine_never_early_endpoints_without_verified_vad_silence():
    """A stable decoder hypothesis is not acoustic silence."""
    det = ScriptedTurnCompletionDetector({"tell me a complete story": 0.9})
    e = _engine(detector=det, endpoint_enabled=True, endpoint_min_silence_sec=0.3)
    assert e._decide_endpoint(
        acoustic_endpoint=False,
        partial="tell me a complete story",
        silence_sec=5.0,
        allow_early=False,
    ) is False
    # A real acoustic endpoint can still pass through semantic accept/hold.
    assert e._decide_endpoint(
        acoustic_endpoint=True,
        partial="tell me a complete story",
        silence_sec=5.0,
        allow_early=False,
    ) is True


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


def test_engine_only_assembles_prosody_audio_inside_decision_window():
    class _AudioDetector:
        needs_audio = True

        def completion_score(self, text, *, samples=None, sample_rate=16000):
            return 0.9

    e = _engine(detector=_AudioDetector(), endpoint_enabled=True)
    floor = e._endpoint_prosody_min_silence
    assert not e._endpoint_audio_needed(
        acoustic_endpoint=False,
        partial="hello there",
        silence_sec=max(0.0, floor - 0.01),
        allow_early=True,
    )
    assert not e._endpoint_audio_needed(
        acoustic_endpoint=False,
        partial="hello there",
        silence_sec=floor + 1.0,
        allow_early=False,
    )
    assert e._endpoint_audio_needed(
        acoustic_endpoint=False,
        partial="hello there",
        silence_sec=floor,
        allow_early=True,
    )


def test_engine_extends_then_backstops_on_incomplete_partial():
    det = ScriptedTurnCompletionDetector({"and then": 0.05})
    e = _engine(detector=det, endpoint_enabled=True, endpoint_max_silence_sec=1.6)
    assert e._decide_endpoint(acoustic_endpoint=True, partial="and then", silence_sec=1.0) is False
    assert e._decide_endpoint(acoustic_endpoint=True, partial="and then", silence_sec=2.0) is True


def test_engine_active_vad_makes_acoustic_rule3_endpoint_a_hard_boundary():
    det = ScriptedTurnCompletionDetector({"and then": 0.05})
    e = _engine(detector=det, endpoint_enabled=True, endpoint_max_silence_sec=1.6)

    assert e._decide_endpoint(
        acoustic_endpoint=True,
        partial="and then",
        silence_sec=0.0,
        allow_early=False,
        vad_active=True,
    ) is True
    # The hard boundary bypasses semantic scoring entirely; it is not merely a
    # low-score HOLD that happens to time out later.
    assert det.calls == []


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


def test_shared_observation_helper_pins_every_engine_branch_state():
    class Detector:
        needs_audio = True

        def __init__(self, *, fail: bool = False) -> None:
            self.fail = fail

        def completion_score(self, text, *, samples=None, sample_rate=16000):
            if self.fail:
                raise RuntimeError("synthetic")
            return 0.8

    policy = AdaptiveEndpointPolicy(EndpointConfig(enabled=True))
    scored = observe_turn_completion(
        detector=Detector(),
        policy=policy,
        acoustic_endpoint=False,
        partial="available",
        silence_sec=0.2,
        samples=[0.0],
        audio_min_silence_sec=0.15,
        detector_needs_audio=True,
    )
    assert scored.completion_state is CompletionScoreState.SCORED
    assert scored.completion_score == 0.8

    cases = (
        ({"acoustic_endpoint": True, "vad_active": True}, CompletionScoreState.HARD_BOUNDARY),
        ({"policy": None}, CompletionScoreState.UNAVAILABLE),
        ({"partial": "  "}, CompletionScoreState.NO_PARTIAL),
        ({"allow_early": False}, CompletionScoreState.EARLY_GUARDED),
        ({"silence_sec": 0.1}, CompletionScoreState.AUDIO_WINDOW_PENDING),
        ({"detector": Detector(fail=True)}, CompletionScoreState.DETECTOR_ERROR),
    )
    defaults = {
        "detector": Detector(),
        "policy": policy,
        "acoustic_endpoint": False,
        "partial": "available",
        "silence_sec": 0.2,
        "samples": [0.0],
        "allow_early": True,
        "vad_active": False,
        "audio_min_silence_sec": 0.15,
        "detector_needs_audio": True,
    }
    for overrides, expected in cases:
        observation = observe_turn_completion(**(defaults | overrides))
        assert observation.completion_state is expected


def test_shared_observation_helper_preserves_exact_detector_arguments_and_guards():
    samples = [0.125, -0.25]
    calls = []

    class Detector:
        needs_audio = True

        def completion_score(self, text, *, samples=None, sample_rate=16000):
            calls.append((text, samples, sample_rate))
            return 0.8125

    detector = Detector()
    policy = AdaptiveEndpointPolicy(EndpointConfig(enabled=True))
    observation = observe_turn_completion(
        detector=detector,
        policy=policy,
        acoustic_endpoint=False,
        partial="  available now  ",
        silence_sec=0.2,
        samples=samples,
        sample_rate=22_050,
        audio_min_silence_sec=0.15,
        detector_needs_audio=True,
    )

    assert calls == [("available now", samples, 22_050)]
    assert calls[0][1] is samples
    assert observation.completion_state is CompletionScoreState.SCORED
    assert observation.completion_score == 0.8125

    guarded = (
        {"acoustic_endpoint": True, "vad_active": True},
        {"policy": None},
        {"detector": None},
        {"partial": "  "},
        {"allow_early": False},
        {"silence_sec": 0.1},
    )
    defaults = {
        "detector": detector,
        "policy": policy,
        "acoustic_endpoint": False,
        "partial": "available now",
        "silence_sec": 0.2,
        "samples": samples,
        "sample_rate": 22_050,
        "allow_early": True,
        "vad_active": False,
        "audio_min_silence_sec": 0.15,
        "detector_needs_audio": True,
    }
    for overrides in guarded:
        calls.clear()
        observe_turn_completion(**(defaults | overrides))
        assert calls == []


def test_shared_observation_helper_invokes_error_callback_inside_exception():
    import sys

    seen = []

    class Detector:
        needs_audio = False

        def completion_score(self, text, *, samples=None, sample_rate=16000):
            raise RuntimeError("synthetic")

    def on_error(error: Exception) -> None:
        seen.append((error, sys.exc_info()[1]))

    policy = AdaptiveEndpointPolicy(EndpointConfig(enabled=True))
    observation, decision = evaluate_turn_completion(
        detector=Detector(),
        policy=policy,
        acoustic_endpoint=True,
        partial="available",
        silence_sec=0.8,
        on_detector_error=on_error,
    )

    assert observation.completion_state is CompletionScoreState.DETECTOR_ERROR
    assert decision.commit is True
    assert len(seen) == 1
    assert seen[0][0] is seen[0][1]


def test_engine_uses_cached_audio_need_when_detector_attribute_mutates():
    class Detector:
        needs_audio = True

        def completion_score(self, text, *, samples=None, sample_rate=16000):
            return 0.9

    detector = Detector()
    engine = _engine(detector=detector, endpoint_enabled=True)
    detector.needs_audio = False

    observation = engine._observe_turn(
        acoustic_endpoint=False,
        partial="available",
        silence_sec=0.1,
        samples=[0.0],
    )

    assert observation.completion_state is CompletionScoreState.AUDIO_WINDOW_PENDING


def test_engine_cached_audio_need_false_overrides_mutated_detector_attribute():
    calls = []

    class Detector:
        needs_audio = False

        def completion_score(self, text, *, samples=None, sample_rate=16000):
            calls.append((text, samples, sample_rate))
            return 0.9

    detector = Detector()
    engine = _engine(detector=detector, endpoint_enabled=True)
    detector.needs_audio = True
    samples = [0.0]

    observation = engine._observe_turn(
        acoustic_endpoint=False,
        partial="available",
        silence_sec=0.1,
        samples=samples,
    )

    assert observation.completion_state is CompletionScoreState.SCORED
    assert calls == [("available", samples, engine.config.sample_rate)]
    assert calls[0][1] is samples


def test_shared_evaluation_is_differentially_identical_to_engine_turn_evaluation():
    class Detector:
        def __init__(self, *, needs_audio=False, score=0.8, error=None):
            self.needs_audio = needs_audio
            self.score = score
            self.error = error

        def completion_score(self, text, *, samples=None, sample_rate=16000):
            if self.error is not None:
                raise self.error
            return self.score

    cases = (
        {
            "detector": Detector(),
            "endpoint_enabled": False,
            "call": {"acoustic_endpoint": True, "partial": "available", "silence_sec": 0.8},
        },
        {
            "detector": Detector(),
            "endpoint_enabled": True,
            "call": {"acoustic_endpoint": True, "partial": " ", "silence_sec": 0.8},
        },
        {
            "detector": Detector(),
            "endpoint_enabled": True,
            "call": {
                "acoustic_endpoint": False,
                "partial": "available",
                "silence_sec": 0.8,
                "allow_early": False,
            },
        },
        {
            "detector": Detector(needs_audio=True),
            "endpoint_enabled": True,
            "call": {"acoustic_endpoint": False, "partial": "available", "silence_sec": 0.1},
        },
        {
            "detector": Detector(score=0.9),
            "endpoint_enabled": True,
            "call": {"acoustic_endpoint": False, "partial": "available", "silence_sec": 0.8},
        },
        {
            "detector": Detector(error=RuntimeError("synthetic differential")),
            "endpoint_enabled": True,
            "call": {"acoustic_endpoint": True, "partial": "available", "silence_sec": 0.8},
        },
        {
            "detector": Detector(error=AssertionError("must not score")),
            "endpoint_enabled": True,
            "call": {
                "acoustic_endpoint": True,
                "partial": "available",
                "silence_sec": 0.0,
                "vad_active": True,
            },
        },
    )

    for case in cases:
        detector = case["detector"]
        engine = _engine(
            detector=detector,
            endpoint_enabled=case["endpoint_enabled"],
        )
        call = case["call"]
        actual = engine._turn_evaluation(**call)
        expected = evaluate_turn_completion(
            detector=engine._turn_detector,
            policy=engine._endpoint_policy,
            samples=call.get("samples"),
            sample_rate=engine.config.sample_rate,
            audio_min_silence_sec=engine._endpoint_prosody_min_silence,
            detector_needs_audio=engine._endpoint_wants_audio,
            **{key: value for key, value in call.items() if key != "samples"},
        )
        assert actual == expected


def test_engine_detector_error_debug_log_retains_exact_exception(caplog):
    import logging

    error = RuntimeError("exact detector traceback")

    class Detector:
        needs_audio = False

        def completion_score(self, text, *, samples=None, sample_rate=16000):
            raise error

    engine = _engine(detector=Detector(), endpoint_enabled=True)
    with caplog.at_level(logging.DEBUG, logger="speaker.sherpa"):
        observation = engine._observe_turn(
            acoustic_endpoint=True,
            partial="available",
            silence_sec=0.8,
        )

    assert observation.completion_state is CompletionScoreState.DETECTOR_ERROR
    records = [
        record
        for record in caplog.records
        if record.getMessage()
        == "turn-completion detector failed; using acoustic endpoint"
    ]
    assert len(records) == 1
    assert records[0].exc_info is not None
    assert records[0].exc_info[1] is error


def test_sherpa_config_parses_endpoint_fields_and_ignores_comment():
    from core.engines.sherpa import SherpaConfig

    cfg = SherpaConfig.from_dict(
        {"endpoint_enabled": True, "endpoint_min_silence_sec": 0.25, "_endpoint_comment": "x"}
    )
    assert cfg.endpoint_enabled is True
    assert cfg.endpoint_min_silence_sec == 0.25


# --- FIX-4: session pause model (learned trailing-silence floor Lp) -----------


def test_session_pause_model_cold_start_below_min_samples():
    # Under min_samples the floor is the (clamped) cold-start value; once enough
    # pauses are seen it tracks the learned quantile (rises for a slow speaker).
    m = SessionPauseModel(min_samples=8, cold_start_sec=0.5, floor_lo=0.4, floor_hi=1.6)
    assert m.floor() == pytest.approx(0.5)  # no samples -> cold start
    for _ in range(7):
        m.observe_pause(0.9)
    assert m.floor() == pytest.approx(0.5)  # 7 < 8 -> still cold start
    m.observe_pause(0.9)  # 8th sample -> learned floor engages
    assert m.floor() == pytest.approx(0.9 * 1.15)  # 1.035, above the cold start


def test_session_pause_model_cold_start_is_clamped():
    # A cold-start above the physical ceiling is clamped to floor_hi (never a raw
    # number leaks past the bounds).
    m = SessionPauseModel(min_samples=4, cold_start_sec=2.0, floor_lo=0.3, floor_hi=1.6)
    assert m.floor() == pytest.approx(1.6)


def test_session_pause_model_tracks_learned_quantile():
    # rises toward the learned pause for a slow (~0.9s) speaker...
    slow = SessionPauseModel(min_samples=4, floor_lo=0.4, floor_hi=1.6, pause_margin=0.15)
    for _ in range(6):
        slow.observe_pause(0.9)
    assert slow.floor() == pytest.approx(0.9 * 1.15)  # 1.035
    assert slow.floor() > 0.4
    # ...and collapses to floor_lo for a terse (~0.3s) speaker.
    terse = SessionPauseModel(min_samples=4, floor_lo=0.4, floor_hi=1.6, pause_margin=0.15)
    for _ in range(6):
        terse.observe_pause(0.3)
    assert terse.floor() == pytest.approx(0.4)  # 0.3*1.15=0.345 clamped up to floor_lo


def test_session_pause_model_observe_filters_micro_and_oversized_gaps():
    # Sub-0.2s micro-pauses and gaps above floor_hi (end-of-turn silences) are
    # ignored so they can't drag or inflate the learned floor.
    m = SessionPauseModel(floor_hi=1.6)
    m.observe_pause(0.1)    # micro-pause < 0.2 -> ignored
    m.observe_pause(0.19)   # just under the 0.2 bound -> ignored
    m.observe_pause(1.9)    # > floor_hi -> ignored
    assert len(m._samples) == 0
    m.observe_pause(0.2)    # exactly the linguistic floor -> kept
    m.observe_pause(0.5)    # in range -> kept
    m.observe_pause(1.6)    # exactly floor_hi -> kept
    assert list(m._samples) == [0.2, 0.5, 1.6]


def test_session_pause_model_window_ages_out_old_pauses():
    # A bounded deque(maxlen=window): old pauses drop off so the floor tracks the
    # CURRENT speaker, not the whole session.
    m = SessionPauseModel(window=3, min_samples=1, floor_lo=0.0, floor_hi=1.6, pause_margin=0.0)
    for g in (0.3, 0.4, 0.5, 0.9, 0.9, 0.9):
        m.observe_pause(g)
    assert list(m._samples) == [0.9, 0.9, 0.9]
    assert m.floor() == pytest.approx(0.9)  # the early small pauses aged out


# --- FIX-4: adaptive-floor policy (Lp replaces the fixed min-silence) ----------

# The full existing decide() matrix, run with adaptive_floor OFF, must stay
# byte-identical to today (both the default and the explicit-False flag).
_BYTE_IDENTICAL_CASES = [
    (dict(min_silence_sec=0.4), False, 0.9, 0.5, True),
    (dict(min_silence_sec=0.4), False, 0.9, 0.3, False),
    (dict(min_silence_sec=0.7, complete_threshold=0.6,
          high_confidence_floor=0.55, high_confidence_score=0.75), False, 0.75, 0.55, True),
    (dict(min_silence_sec=0.7, complete_threshold=0.6,
          high_confidence_floor=0.55, high_confidence_score=0.75), False, 0.75, 0.54, False),
    (dict(min_silence_sec=0.7, complete_threshold=0.6,
          high_confidence_floor=0.55, high_confidence_score=0.75), False, 0.65, 0.55, False),
    (dict(min_silence_sec=0.7, complete_threshold=0.6,
          high_confidence_floor=0.55, high_confidence_score=0.75), False, 0.65, 0.7, True),
    (dict(min_silence_sec=0.7, high_confidence_floor=0.0), False, 0.75, 0.55, False),
    (dict(min_silence_sec=0.7, high_confidence_floor=0.0), False, 0.75, 0.7, True),
    (dict(max_silence_sec=1.6), True, 0.1, 1.0, False),
    (dict(max_silence_sec=1.6), True, 0.1, 1.7, True),
    (dict(), True, 0.5, 0.9, True),
    (dict(), False, 0.5, 0.9, False),
]


@pytest.mark.parametrize("cfg_kw,acoustic,score,silence,expected", _BYTE_IDENTICAL_CASES)
def test_policy_adaptive_floor_false_is_byte_identical(cfg_kw, acoustic, score, silence, expected):
    default = AdaptiveEndpointPolicy(EndpointConfig(enabled=True, **cfg_kw))
    explicit = AdaptiveEndpointPolicy(EndpointConfig(enabled=True, adaptive_floor=False, **cfg_kw))
    for p in (default, explicit):
        assert p.decide(
            acoustic_endpoint=acoustic, completion_score=score, silence_sec=silence
        ) is expected


def test_policy_adaptive_anti_chop_holds_a_mid_utterance_pause():
    # The fragmentation regression: "the mere story ... about ... my cat biki".
    # Learned Lp=0.9; a complete-scoring partial at 0.6s silence -- with the
    # acoustic timer ALREADY fired -- is HELD, so the resume lands as a new
    # partial instead of committing a fragment.
    p = _policy(adaptive_floor=True, complete_threshold=0.6, max_silence_sec=1.6,
                pause_margin=0.0, pause_min_samples=1)
    p.observe_pause(0.9)  # Lp = clamp(0.9 * 1.0, floor_lo=0.0, floor_hi=1.6) = 0.9
    assert p.decide(acoustic_endpoint=True, completion_score=0.75, silence_sec=0.6) is False
    # ...but once trailing silence reaches Lp, the complete turn commits.
    assert p.decide(acoustic_endpoint=True, completion_score=0.75, silence_sec=0.95) is True


def test_policy_adaptive_terse_speaker_commits_at_floor_lo():
    # A terse speaker with tiny pauses -> Lp collapses to floor_lo (the
    # high_confidence_floor decoder guard). A complete turn commits as soon as
    # trailing silence reaches floor_lo -- no slower than the fixed floor was.
    p = _policy(adaptive_floor=True, complete_threshold=0.6, max_silence_sec=1.6,
                high_confidence_floor=0.4, pause_min_samples=4)
    for _ in range(4):
        p.observe_pause(0.3)  # 0.3*1.15=0.345 -> clamped up to floor_lo 0.4
    assert p._pause.floor() == pytest.approx(0.4)
    assert p.decide(acoustic_endpoint=False, completion_score=0.9, silence_sec=0.4) is True
    assert p.decide(acoustic_endpoint=False, completion_score=0.9, silence_sec=0.39) is False


def test_policy_adaptive_incomplete_still_bounded_by_max_silence():
    # The incomplete-EXTEND behaviour is preserved and still hard-capped at
    # max_silence_sec (a mis-scored turn always commits eventually).
    p = _policy(adaptive_floor=True, incomplete_threshold=0.3, max_silence_sec=1.6,
                pause_margin=0.0, pause_min_samples=1)
    p.observe_pause(0.9)  # Lp = 0.9
    assert p.decide(acoustic_endpoint=True, completion_score=0.1, silence_sec=1.0) is False
    assert p.decide(acoustic_endpoint=True, completion_score=0.1, silence_sec=1.7) is True


def test_policy_observe_pause_delegates_to_the_session_model():
    p = _policy(adaptive_floor=True, pause_min_samples=8, pause_margin=0.15,
                high_confidence_floor=0.0, max_silence_sec=1.6)
    for _ in range(8):
        p.observe_pause(0.8)
    assert len(p._pause._samples) == 8
    assert p._pause.floor() == pytest.approx(min(max(0.8 * 1.15, 0.0), 1.6))
