"""Two-pass ASR: the optional offline second-pass FINAL recognizer (SenseVoice/
Whisper) that re-transcribes the endpointed utterance for the text that reaches
the LLM. These pin the wiring/fallback logic with fakes -- no models/audio."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from core.engines._sherpa_models import build_final_recognizer
from core.engines.sherpa import SherpaConfig, SherpaOnnxEngine


class _FakeStream:
    def __init__(self, text):
        self.result = SimpleNamespace(text=text)

    def accept_waveform(self, sr, a):
        pass


class _FakeOffline:
    """A stand-in OfflineRecognizer: returns a fixed text for any audio."""

    def __init__(self, text):
        self._text = text

    def create_stream(self):
        return _FakeStream(self._text)

    def decode_stream(self, stream):
        pass


def _engine(**sherpa):
    return SherpaOnnxEngine(SherpaConfig.from_dict(sherpa))


# --- build_final_recognizer (fail-open) ---------------------------------------


def test_build_final_recognizer_none_without_backend():
    assert build_final_recognizer(SherpaConfig.from_dict({})) is None


def test_build_final_recognizer_none_when_model_missing():
    cfg = SherpaConfig.from_dict({"asr_final_backend": "sense_voice",
                                  "asr_final_model": "/does/not/exist.onnx"})
    assert build_final_recognizer(cfg) is None  # graceful, no crash


# --- _final_transcribe (second pass vs fallback) ------------------------------


def test_final_transcribe_uses_second_pass_when_present():
    eng = _engine()
    eng._final_recognizer = _FakeOffline("Hey, are you listening to me.")
    out = eng._final_transcribe(np.ones(16000, dtype="float32"), "HEY IRIC LISTENING TO ME")
    assert out == "Hey, are you listening to me."  # the clean second-pass text


def test_final_transcribe_falls_back_without_second_pass():
    eng = _engine()
    assert eng._final_recognizer is None
    # falls back to _postprocess_final (casing) of the streaming raw
    assert eng._final_transcribe(None, "hello world") == "Hello world"


def test_final_transcribe_empty_second_pass_falls_back():
    eng = _engine()
    eng._final_recognizer = _FakeOffline("")  # empty result -> use the streaming final
    assert eng._final_transcribe(np.ones(16000, dtype="float32"), "hello world") == "Hello world"


def test_final_transcribe_min_sec_skips_short_utterance():
    eng = _engine(asr_final_min_sec=2.0)
    eng._final_recognizer = _FakeOffline("SECOND PASS")
    # 0.5s of audio < the 2.0s floor -> skip the second pass, fall back.
    out = eng._final_transcribe(np.ones(8000, dtype="float32"), "hi there")
    assert out != "SECOND PASS"
    # ...but a long-enough utterance DOES use it.
    long_out = eng._final_transcribe(np.ones(2 * 16000, dtype="float32"), "hi there")
    assert long_out == "SECOND PASS"


def test_final_transcribe_recovers_from_second_pass_error():
    class _Boom:
        def create_stream(self):
            raise RuntimeError("model exploded")

    eng = _engine()
    eng._final_recognizer = _Boom()
    # a second-pass failure must never lose the turn -> the streaming final stands.
    assert eng._final_transcribe(np.ones(16000, dtype="float32"), "hello world") == "Hello world"


def test_config_parses_final_fields():
    c = SherpaConfig.from_dict({
        "asr_final_backend": "sense_voice", "asr_final_model": "/m.onnx",
        "asr_final_tokens": "/t.txt", "asr_final_use_itn": False, "asr_final_min_sec": 0.5,
    })
    assert c.asr_final_backend == "sense_voice" and c.asr_final_model == "/m.onnx"
    assert c.asr_final_use_itn is False and c.asr_final_min_sec == 0.5
