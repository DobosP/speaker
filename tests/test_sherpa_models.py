"""The shared sherpa-onnx builders construct the right objects from config.

A fake ``sherpa_onnx`` module is injected so this runs without the native
package (and without any model files), locking the extraction shared by the
local and LiveKit engines.
"""
import sys

import pytest

from core.engines._sherpa_models import (
    _supported,
    build_punctuation,
    build_recognizer,
    build_tts,
    build_vad,
)
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
        self.noise_scale = 0.667
        self.noise_scale_w = 0.8


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


class _FakePunctModelConfig:
    def __init__(self, ct_transformer=None, num_threads=None, provider=None):
        self.ct_transformer = ct_transformer
        self.num_threads = num_threads
        self.provider = provider


class _FakePunctConfig:
    def __init__(self, model=None):
        self.model = model


class _FakePunctuation:
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
    mod.OfflinePunctuationConfig = _FakePunctConfig
    mod.OfflinePunctuationModelConfig = _FakePunctModelConfig
    mod.OfflinePunctuation = _FakePunctuation
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


def test_build_recognizer_wires_decoding_and_endpoint_rules(fake_sherpa):
    c = SherpaConfig(
        asr_encoder="enc",
        asr_tokens="tok",
        asr_decoding_method="modified_beam_search",
        asr_max_active_paths=6,
        asr_hotwords="Flurry\nParis",
        asr_hotwords_score=2.0,
        asr_rule2_min_trailing_silence=0.7,
    )
    rec = build_recognizer(c)
    assert rec.kwargs["decoding_method"] == "modified_beam_search"
    assert rec.kwargs["max_active_paths"] == 6
    assert rec.kwargs["rule2_min_trailing_silence"] == 0.7
    # Hotword score is passed only with beam search.
    assert rec.kwargs["hotwords_score"] == 2.0


def test_build_recognizer_no_hotword_score_under_greedy(fake_sherpa):
    c = SherpaConfig(
        asr_encoder="enc", asr_tokens="tok",
        asr_decoding_method="greedy_search", asr_hotwords="Flurry",
    )
    rec = build_recognizer(c)
    assert "hotwords_score" not in rec.kwargs


def test_supported_drops_unknown_kwargs_for_narrow_signature():
    def narrow(encoder, decoder):  # no **kwargs
        return None

    kept = _supported(narrow, {"encoder": 1, "decoder": 2, "rule2_min_trailing_silence": 0.7})
    assert kept == {"encoder": 1, "decoder": 2}


def test_supported_keeps_all_for_varkw_signature():
    def wide(**kwargs):
        return None

    payload = {"a": 1, "b": 2}
    assert _supported(wide, payload) == payload


def test_build_vad_passes_config(fake_sherpa):
    c = SherpaConfig(vad_model="vad.onnx", sample_rate=16000, provider="cpu", asr_num_threads=2)
    vad = build_vad(c)
    assert isinstance(vad, _FakeVad)
    assert vad.config.silero_vad.model == "vad.onnx"
    assert vad.config.sample_rate == 16000
    assert vad.config.num_threads == 2
    assert vad.buffer == 30


def test_build_punctuation_none_when_unconfigured(fake_sherpa):
    assert build_punctuation(SherpaConfig()) is None


def test_build_punctuation_passes_config(fake_sherpa):
    c = SherpaConfig(punct_model="punct.onnx", provider="cpu", asr_num_threads=2)
    punct = build_punctuation(c)
    assert isinstance(punct, _FakePunctuation)
    assert punct.config.model.ct_transformer == "punct.onnx"
    assert punct.config.model.num_threads == 2


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
    assert tts.config.model.vits.noise_scale == 0.667
    assert tts.config.model.vits.noise_scale_w == 0.8


def test_build_tts_deterministic_vits_zeros_only_generation_noise(fake_sherpa):
    c = SherpaConfig(
        tts_model="tts.onnx",
        tts_tokens="tts_tokens.txt",
        tts_data_dir="data",
    )

    tts = build_tts(c, deterministic_vits=True)

    assert isinstance(tts, _FakeOfflineTts)
    assert tts.config.model.vits.model == "tts.onnx"
    assert tts.config.model.vits.tokens == "tts_tokens.txt"
    assert tts.config.model.vits.data_dir == "data"
    assert tts.config.model.vits.noise_scale == 0.0
    assert tts.config.model.vits.noise_scale_w == 0.0
