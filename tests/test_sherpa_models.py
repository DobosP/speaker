"""The shared sherpa-onnx builders construct the right objects from config.

A fake ``sherpa_onnx`` module is injected so this runs without the native
package (and without any model files), locking the extraction shared by the
local and LiveKit engines.
"""
import sys

import pytest

from core.engines._sherpa_models import build_recognizer, build_tts, build_vad
from core.engines.sherpa import SherpaConfig


class _FakeRecognizer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeOnlineRecognizer:
    @staticmethod
    def from_transducer(**kwargs):
        return _FakeRecognizer(**kwargs)


class _Silero:
    def __init__(self):
        self.model = None


class _FakeVadConfig:
    def __init__(self):
        self.silero_vad = _Silero()
        self.sample_rate = None
        self.num_threads = None
        self.provider = None


class _FakeVad:
    def __init__(self, config, buffer_size_in_seconds=None):
        self.config = config
        self.buffer = buffer_size_in_seconds


class _Vits:
    def __init__(self):
        self.model = None
        self.tokens = None
        self.data_dir = None


class _ModelConfig:
    def __init__(self):
        self.vits = _Vits()
        self.num_threads = None
        self.provider = None


class _FakeOfflineTtsConfig:
    def __init__(self):
        self.model = _ModelConfig()


class _FakeOfflineTts:
    def __init__(self, config):
        self.config = config


@pytest.fixture
def fake_sherpa(monkeypatch):
    import types

    mod = types.ModuleType("sherpa_onnx")
    mod.OnlineRecognizer = _FakeOnlineRecognizer
    mod.VadModelConfig = _FakeVadConfig
    mod.VoiceActivityDetector = _FakeVad
    mod.OfflineTtsConfig = _FakeOfflineTtsConfig
    mod.OfflineTts = _FakeOfflineTts
    monkeypatch.setitem(sys.modules, "sherpa_onnx", mod)
    return mod


def test_builders_return_none_when_unconfigured(fake_sherpa):
    c = SherpaConfig()  # all model paths empty
    assert build_recognizer(c) is None
    assert build_vad(c) is None
    assert build_tts(c) is None


def test_build_recognizer_passes_config(fake_sherpa):
    c = SherpaConfig(
        asr_tokens="tok",
        asr_encoder="enc",
        asr_decoder="dec",
        asr_joiner="joi",
        sample_rate=16000,
        provider="cpu",
        asr_num_threads=3,
    )
    rec = build_recognizer(c)
    assert isinstance(rec, _FakeRecognizer)
    assert rec.kwargs["encoder"] == "enc"
    assert rec.kwargs["num_threads"] == 3
    assert rec.kwargs["sample_rate"] == 16000
    assert rec.kwargs["enable_endpoint_detection"] is True


def test_build_vad_passes_config(fake_sherpa):
    c = SherpaConfig(vad_model="vad.onnx", sample_rate=16000, provider="cpu", asr_num_threads=2)
    vad = build_vad(c)
    assert isinstance(vad, _FakeVad)
    assert vad.config.silero_vad.model == "vad.onnx"
    assert vad.config.sample_rate == 16000
    assert vad.config.num_threads == 2
    assert vad.buffer == 30


def test_build_tts_passes_config(fake_sherpa):
    c = SherpaConfig(
        tts_model="tts.onnx",
        tts_tokens="tts_tokens.txt",
        tts_data_dir="data",
        provider="cpu",
        tts_num_threads=1,
    )
    tts = build_tts(c)
    assert isinstance(tts, _FakeOfflineTts)
    assert tts.config.model.vits.model == "tts.onnx"
    assert tts.config.model.vits.tokens == "tts_tokens.txt"
    assert tts.config.model.vits.data_dir == "data"
    assert tts.config.model.num_threads == 1
