"""Engine-level tests for ASR final post-processing and stream creation.

No sherpa-onnx, no models, no audio: the recognizer / punctuation objects are
fakes assigned onto the engine, exercising the casing + punctuation + hotword
wiring directly (the threaded capture loop is out of scope).
"""
from __future__ import annotations

from core.engines.sherpa import SherpaConfig, SherpaOnnxEngine


class _FakePunct:
    def __init__(self):
        self.seen = []

    def add_punctuation(self, text):
        self.seen.append(text)
        return text + "."


def test_postprocess_final_casing_only():
    eng = SherpaOnnxEngine(SherpaConfig(asr_restore_casing=True))
    # No punctuation model -> casing restoration only.
    assert eng._postprocess_final("HE MURMURED HIS MURDERING") == "He murmured his murdering"


def test_postprocess_final_casing_disabled_is_passthrough():
    eng = SherpaOnnxEngine(SherpaConfig(asr_restore_casing=False))
    assert eng._postprocess_final("HE MURMURED") == "HE MURMURED"


def test_postprocess_final_with_punctuation_then_casing():
    eng = SherpaOnnxEngine(SherpaConfig(asr_restore_casing=True))
    eng._punct = _FakePunct()
    # Punctuation runs first (sees raw text), then casing (force-lowercases the
    # all-caps source before re-capitalizing) -> readable sentence.
    out = eng._postprocess_final("WHAT TIME IS IT")
    assert eng._punct.seen == ["WHAT TIME IS IT"]
    assert out == "What time is it."


def test_postprocess_final_survives_punctuation_failure():
    class _Boom:
        def add_punctuation(self, text):
            raise RuntimeError("punct model died")

    eng = SherpaOnnxEngine(SherpaConfig(asr_restore_casing=True))
    eng._punct = _Boom()
    # Falls back to casing-only rather than losing the turn.
    assert eng._postprocess_final("SET A TIMER") == "Set a timer"


class _FakeRecognizer:
    def __init__(self):
        self.hotwords_arg = "UNSET"
        self.plain_calls = 0

    def create_stream(self, hotwords=None):
        self.hotwords_arg = hotwords
        return object()


class _NoHotwordRecognizer:
    """A build whose create_stream() rejects the hotwords kwarg."""

    def __init__(self):
        self.plain_calls = 0

    def create_stream(self):
        self.plain_calls += 1
        return object()


def test_new_asr_stream_passes_hotwords_with_beam_search():
    eng = SherpaOnnxEngine(
        SherpaConfig(asr_decoding_method="modified_beam_search", asr_hotwords="Flurry\nParis")
    )
    rec = _FakeRecognizer()
    eng._recognizer = rec
    eng._hotwords = ["Flurry", "Paris"]
    eng._new_asr_stream()
    assert rec.hotwords_arg == "Flurry\nParis"


def test_new_asr_stream_no_hotwords_under_greedy():
    eng = SherpaOnnxEngine(
        SherpaConfig(asr_decoding_method="greedy_search", asr_hotwords="Flurry")
    )
    rec = _FakeRecognizer()
    eng._recognizer = rec
    eng._hotwords = ["Flurry"]
    eng._new_asr_stream()
    # Greedy search must not pass hotwords (the recognizer wouldn't honor them).
    assert rec.hotwords_arg is None


def test_new_asr_stream_falls_back_when_build_rejects_hotwords():
    eng = SherpaOnnxEngine(
        SherpaConfig(asr_decoding_method="modified_beam_search", asr_hotwords="Flurry")
    )
    rec = _NoHotwordRecognizer()
    eng._recognizer = rec
    eng._hotwords = ["Flurry"]
    eng._new_asr_stream()  # TypeError on hotwords= -> retried plain
    assert rec.plain_calls == 1
