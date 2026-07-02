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


def _kokoro_files(tmp_path):
    """Create the (empty) files a Kokoro config points at, so build_tts's
    existence guard admits them. Returns (model, voices, tokens) paths as str."""
    paths = []
    for name in ("model.int8.onnx", "voices.bin", "tokens.txt"):
        p = tmp_path / name
        p.write_bytes(b"x")
        paths.append(str(p))
    return paths


def test_build_tts_kokoro_path_when_voices_set(monkeypatch, tmp_path):
    model, voices, tokens = _kokoro_files(tmp_path)
    out, cfg = _build(monkeypatch, SherpaConfig(
        tts_model=model, tts_voices=voices, tts_tokens=tokens,
        tts_data_dir="/k/espeak", tts_lexicon="/k/lexicon-us-en.txt"))
    assert out is not None
    assert cfg.model.kokoro.model == model
    assert cfg.model.kokoro.voices == voices
    assert cfg.model.kokoro.tokens == tokens
    assert cfg.model.kokoro.data_dir == "/k/espeak"
    assert cfg.model.kokoro.lexicon == "/k/lexicon-us-en.txt"
    assert cfg.model.vits.model == ""            # VITS untouched -> Kokoro path


def test_build_tts_kokoro_without_lexicon_leaves_it_empty(monkeypatch, tmp_path):
    model, voices, tokens = _kokoro_files(tmp_path)
    out, cfg = _build(monkeypatch, SherpaConfig(
        tts_model=model, tts_voices=voices, tts_tokens=tokens))
    assert cfg.model.kokoro.voices == voices
    assert cfg.model.kokoro.lexicon == ""        # optional -> not set


def test_build_tts_kokoro_missing_files_returns_none(monkeypatch, caplog):
    # tts_voices set (Kokoro) but the package was never fetched: graceful None +
    # an actionable warning, instead of the native loader hard-aborting.
    import logging

    with caplog.at_level(logging.WARNING):
        out, _ = _build(monkeypatch, SherpaConfig(
            tts_model="/nope/model.onnx", tts_voices="/nope/voices.bin",
            tts_tokens="/nope/tokens.txt"))
    assert out is None
    assert any("Kokoro" in r.message and "missing" in r.message for r in caplog.records)


def test_build_tts_returns_none_on_build_error(monkeypatch):
    # Any other native build failure (corrupt model, etc.) also fails open to
    # no-TTS rather than crashing the capture thread. VITS branch (no existence
    # gate), so this exercises the try/except around OfflineTts().
    m = _fake_sherpa_onnx({})

    def _boom(cfg):
        raise RuntimeError("bad model")

    m.OfflineTts = _boom
    monkeypatch.setitem(sys.modules, "sherpa_onnx", m)
    assert build_tts(SherpaConfig(tts_model="/m/v.onnx", tts_tokens="/m/t.txt")) is None
