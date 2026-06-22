"""Tests for the TTS backend selection in core/engines/_sherpa_models.build_tts.

sherpa_onnx is faked (no model files, no native runtime) so we assert ONLY the
config wiring: tts_voices present -> the Kokoro family branch; absent -> the
byte-identical VITS/Piper path. The real synth is covered by the manual A/B."""
from __future__ import annotations

import sys
import types

from core.engines._sherpa_models import build_tts
from core.engines.sherpa import SherpaConfig


def _fake_sherpa_onnx(captured):
    m = types.ModuleType("sherpa_onnx")

    class _Cfg:
        def __init__(self):
            self.model = types.SimpleNamespace(
                vits=types.SimpleNamespace(model="", tokens="", data_dir=""),
                kokoro=types.SimpleNamespace(model="", voices="", tokens="", data_dir="", lexicon=""),
                num_threads=0,
                provider="",
            )

    m.OfflineTtsConfig = _Cfg

    def _offline_tts(cfg):
        captured["cfg"] = cfg
        return object()

    m.OfflineTts = _offline_tts
    return m


def _build(monkeypatch, cfg):
    captured = {}
    monkeypatch.setitem(sys.modules, "sherpa_onnx", _fake_sherpa_onnx(captured))
    out = build_tts(cfg)
    return out, captured.get("cfg")


def test_build_tts_none_without_model(monkeypatch):
    out, _ = _build(monkeypatch, SherpaConfig(tts_model=""))
    assert out is None


def test_build_tts_vits_path_when_no_voices(monkeypatch):
    out, cfg = _build(monkeypatch, SherpaConfig(
        tts_model="/m/voice.onnx", tts_tokens="/m/tokens.txt", tts_data_dir="/m/espeak"))
    assert out is not None
    assert cfg.model.vits.model == "/m/voice.onnx"
    assert cfg.model.vits.tokens == "/m/tokens.txt"
    assert cfg.model.vits.data_dir == "/m/espeak"
    assert cfg.model.kokoro.model == ""          # Kokoro untouched -> VITS path


def test_build_tts_kokoro_path_when_voices_set(monkeypatch):
    out, cfg = _build(monkeypatch, SherpaConfig(
        tts_model="/k/model.int8.onnx", tts_voices="/k/voices.bin",
        tts_tokens="/k/tokens.txt", tts_data_dir="/k/espeak",
        tts_lexicon="/k/lexicon-us-en.txt"))
    assert out is not None
    assert cfg.model.kokoro.model == "/k/model.int8.onnx"
    assert cfg.model.kokoro.voices == "/k/voices.bin"
    assert cfg.model.kokoro.tokens == "/k/tokens.txt"
    assert cfg.model.kokoro.data_dir == "/k/espeak"
    assert cfg.model.kokoro.lexicon == "/k/lexicon-us-en.txt"
    assert cfg.model.vits.model == ""            # VITS untouched -> Kokoro path


def test_build_tts_kokoro_without_lexicon_leaves_it_empty(monkeypatch):
    out, cfg = _build(monkeypatch, SherpaConfig(
        tts_model="/k/model.int8.onnx", tts_voices="/k/voices.bin", tts_tokens="/k/tokens.txt"))
    assert cfg.model.kokoro.voices == "/k/voices.bin"
    assert cfg.model.kokoro.lexicon == ""        # optional -> not set
