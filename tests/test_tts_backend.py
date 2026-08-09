"""Tests for the TTS backend selection in core/engines/_sherpa_models.build_tts.

sherpa_onnx is faked (no model files, no native runtime) so we assert ONLY the
config wiring: tts_voices present -> the Kokoro family branch; absent -> the
byte-identical VITS/Piper path. The real synth is covered by the manual A/B.

Also covers the family/model preflight (2026-07 incident): a half-finished
Kokoro switch (tts_voices = Kokoro's voices.bin, tts_model still the VITS
file) makes sherpa's native loader call C++ ``exit(-1)`` -- the interpreter
dies with rc 255 and zero output. ``_tts_family_preflight`` must turn exactly
that config into a readable RuntimeError BEFORE sherpa sees it, stay silent on
correct pairings, and fail open when the model's metadata can't be read."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from core.engines._sherpa_models import build_tts, read_onnx_custom_metadata
from core.engines.sherpa import SherpaConfig


def _fake_sherpa_onnx(captured):
    m = types.ModuleType("sherpa_onnx")

    class _Cfg:
        def __init__(self):
            self.model = types.SimpleNamespace(
                vits=types.SimpleNamespace(
                    model="", tokens="", data_dir="",
                    noise_scale=0.667, noise_scale_w=0.8,
                ),
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


def test_build_tts_deterministic_vits_is_explicit_and_default_preserving(monkeypatch):
    captured = {}
    monkeypatch.setitem(sys.modules, "sherpa_onnx", _fake_sherpa_onnx(captured))
    cfg = SherpaConfig(tts_model="/m/voice.onnx", tts_tokens="/m/tokens.txt")

    out = build_tts(cfg, deterministic_vits=True)

    assert out is not None
    assert captured["cfg"].model.vits.noise_scale == 0.0
    assert captured["cfg"].model.vits.noise_scale_w == 0.0


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


# --- family/model preflight (the exit(-1) class killer) ---------------------


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _stub_onnx(path: Path, meta: dict[str, str]) -> str:
    """Write a minimal-but-real ONNX ModelProto: ir_version, a graph blob big
    enough to prove the reader seeks past it, and the given metadata_props."""
    blob = b"\x08\x08"  # ir_version = 8
    graph = b"G" * 4096  # field 7 (graph), skipped wholesale by the reader
    blob += b"\x3a" + _varint(len(graph)) + graph
    for key, value in meta.items():
        k, v = key.encode(), value.encode()
        entry = b"\x0a" + _varint(len(k)) + k + b"\x12" + _varint(len(v)) + v
        blob += b"\x72" + _varint(len(entry)) + entry  # field 14: metadata_props
    path.write_bytes(blob)
    return str(path)


def test_read_onnx_custom_metadata_reads_stub(tmp_path):
    p = _stub_onnx(tmp_path / "m.onnx", {"model_type": "kokoro", "style_dim": "510,1,256"})
    assert read_onnx_custom_metadata(p) == {"model_type": "kokoro", "style_dim": "510,1,256"}


def test_preflight_kokoro_selected_but_vits_model_raises(monkeypatch, tmp_path):
    # THE incident config: tts_voices points at Kokoro's voices.bin while
    # tts_model is still the VITS export. sherpa would exit(-1) the whole
    # interpreter; the preflight must raise a readable error naming both keys.
    captured = {}
    monkeypatch.setitem(sys.modules, "sherpa_onnx", _fake_sherpa_onnx(captured))
    model = _stub_onnx(tmp_path / "en_US-libritts_r-medium.onnx", {"model_type": "vits"})
    voices = tmp_path / "voices.bin"
    voices.write_bytes(b"v")
    tokens = tmp_path / "tokens.txt"
    tokens.write_bytes(b"t")

    with pytest.raises(RuntimeError) as exc:
        build_tts(SherpaConfig(tts_model=model, tts_voices=str(voices), tts_tokens=str(tokens)))
    msg = str(exc.value)
    assert "tts_model" in msg and "tts_voices" in msg
    assert "exit(-1)" in msg
    assert "cfg" not in captured  # sherpa was never reached


def test_preflight_kokoro_model_without_voices_raises(monkeypatch, tmp_path):
    # The mirror image: a Kokoro export on the VITS path also dies natively.
    captured = {}
    monkeypatch.setitem(sys.modules, "sherpa_onnx", _fake_sherpa_onnx(captured))
    model = _stub_onnx(tmp_path / "model.int8.onnx", {"model_type": "kokoro"})

    with pytest.raises(RuntimeError) as exc:
        build_tts(SherpaConfig(tts_model=model, tts_tokens="/m/t.txt"))
    msg = str(exc.value)
    assert "tts_model" in msg and "tts_voices" in msg
    assert "cfg" not in captured


def test_preflight_style_dim_alone_fingerprints_kokoro(monkeypatch, tmp_path):
    # Older Kokoro exports may lack model_type; style_dim is Kokoro-only.
    monkeypatch.setitem(sys.modules, "sherpa_onnx", _fake_sherpa_onnx({}))
    model = _stub_onnx(tmp_path / "model.onnx", {"style_dim": "510,1,256"})
    with pytest.raises(RuntimeError):
        build_tts(SherpaConfig(tts_model=model, tts_tokens="/m/t.txt"))


def test_preflight_correct_pairings_build(monkeypatch, tmp_path):
    kok_model = _stub_onnx(tmp_path / "kokoro.onnx", {"model_type": "kokoro"})
    voices = tmp_path / "voices.bin"
    voices.write_bytes(b"v")
    tokens = tmp_path / "tokens.txt"
    tokens.write_bytes(b"t")
    out, cfg = _build(monkeypatch, SherpaConfig(
        tts_model=kok_model, tts_voices=str(voices), tts_tokens=str(tokens)))
    assert out is not None and cfg.model.kokoro.model == kok_model

    vits_model = _stub_onnx(tmp_path / "vits.onnx", {"model_type": "vits"})
    out, cfg = _build(monkeypatch, SherpaConfig(tts_model=vits_model, tts_tokens=str(tokens)))
    assert out is not None and cfg.model.vits.model == vits_model


def test_preflight_unreadable_metadata_fails_open(monkeypatch, tmp_path, caplog):
    # The preflight must never become its own blocker: an existing file whose
    # bytes aren't a parseable ModelProto -> warn and hand it to sherpa as-is.
    import logging

    model = tmp_path / "weird.onnx"
    model.write_bytes(b"\x0b\x00not-a-protobuf")  # wire type 3 -> unparseable
    with caplog.at_level(logging.WARNING):
        out, cfg = _build(monkeypatch, SherpaConfig(tts_model=str(model), tts_tokens="/m/t.txt"))
    assert out is not None
    assert any("preflight" in r.message for r in caplog.records)


def test_preflight_inconclusive_metadata_proceeds(monkeypatch, tmp_path):
    # A clean ModelProto with no family fingerprint (no model_type/style_dim):
    # inconclusive, so trust the config rather than block unknown exports.
    model = _stub_onnx(tmp_path / "plain.onnx", {"producer": "someone"})
    out, _ = _build(monkeypatch, SherpaConfig(tts_model=model, tts_tokens="/m/t.txt"))
    assert out is not None


# A plain checkout has pretrained_models/sherpa; task worktrees symlink the
# shared store one level deeper (pretrained_models/pretrained_models/sherpa).
_PM = Path(__file__).resolve().parent.parent / "pretrained_models"
_SHERPA_DIR = next(
    (p / "sherpa" for p in (_PM, _PM / "pretrained_models") if (p / "sherpa").is_dir()),
    _PM / "sherpa",
)


@pytest.mark.real_model
def test_preflight_reproduces_2026_07_incident_on_real_models(monkeypatch):
    """The actual 10-day blindness config, on the real files: Kokoro's
    voices.bin selected while tts_model still names the VITS export. Must be a
    readable RuntimeError, not a C++ exit(-1). sherpa_onnx is faked anyway so
    a preflight regression can't kill this pytest process natively."""
    kokoro_dir = _SHERPA_DIR / "tts_kokoro"
    vits = sorted((_SHERPA_DIR / "tts").glob("*.onnx")) if (_SHERPA_DIR / "tts").is_dir() else []
    if not (kokoro_dir / "voices.bin").exists() or not vits:
        pytest.skip("real Kokoro + VITS packages not on disk")
    monkeypatch.setitem(sys.modules, "sherpa_onnx", _fake_sherpa_onnx({}))

    with pytest.raises(RuntimeError) as exc:
        build_tts(SherpaConfig(
            tts_model=str(vits[0]),
            tts_voices=str(kokoro_dir / "voices.bin"),
            tts_tokens=str(kokoro_dir / "tokens.txt"),
        ))
    assert "tts_model" in str(exc.value) and "tts_voices" in str(exc.value)


@pytest.mark.real_model
def test_preflight_real_kokoro_model_without_voices_raises(monkeypatch):
    kokoro_model = _SHERPA_DIR / "tts_kokoro" / "model.int8.onnx"
    if not kokoro_model.exists():
        pytest.skip("real Kokoro package not on disk")
    monkeypatch.setitem(sys.modules, "sherpa_onnx", _fake_sherpa_onnx({}))
    with pytest.raises(RuntimeError):
        build_tts(SherpaConfig(tts_model=str(kokoro_model), tts_tokens="/m/t.txt"))


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
