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
            captured["config_calls"] = captured.get("config_calls", 0) + 1
            self.model = types.SimpleNamespace(
                vits=types.SimpleNamespace(
                    model="", tokens="", data_dir="",
                    noise_scale=0.667, noise_scale_w=0.8,
                ),
                kokoro=types.SimpleNamespace(model="", voices="", tokens="", data_dir="", lexicon=""),
                kitten=types.SimpleNamespace(model="", voices="", tokens="", data_dir=""),
                num_threads=0,
                provider="",
            )

    m.OfflineTtsConfig = _Cfg
    m.OfflineTtsKittenModelConfig = type("_KittenAPI", (), {})
    m.OfflineTtsModelConfig = type("_ModelAPI", (), {"kitten": None})

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


def _kitten_files(tmp_path, *, metadata=None):
    meta = {"model_type": "kitten-tts", "sample_rate": "24000", "n_speakers": "2",
            "has_espeak": "1", "voice": "en-us", "style_dim": "2,4", "version": "8",
            "speaker_speed_priors": "0.8,0.8"}
    if metadata is not None:
        meta = metadata
    model = _stub_onnx(tmp_path / "model.int8.onnx", meta)
    voices = tmp_path / "voices.bin"
    voices.write_bytes(b"\x00" * 64)
    tokens = tmp_path / "tokens.txt"
    tokens.write_text("a 0\nb 1\n")
    data = tmp_path / "espeak-ng-data"
    (data / "lang/gmw").mkdir(parents=True)
    for name in ("phontab", "phondata", "phonindex", "en_dict", "lang/gmw/en"):
        (data / name).write_bytes(b"synthetic bootstrap")
    return SherpaConfig(tts_backend="kitten", tts_model=model, tts_voices=str(voices),
                        tts_tokens=str(tokens), tts_data_dir=str(data), tts_num_threads=2)


def test_explicit_kitten_wins_over_voices_and_preserves_thread_voice_policy(monkeypatch, tmp_path):
    config = _kitten_files(tmp_path)
    config.tts_speaker_id = 1
    config.tts_lock_speaker_id = True
    config.tts_speaker_voices = {"other": 0}
    output, native = _build(monkeypatch, config)
    assert output is not None
    assert native.model.kitten.model == config.tts_model
    assert native.model.kitten.voices == config.tts_voices
    assert native.model.kitten.tokens == config.tts_tokens
    assert native.model.kitten.data_dir == config.tts_data_dir
    assert native.model.num_threads == 2
    assert native.model.provider == "cpu"
    assert native.model.kokoro.model == native.model.vits.model == ""
    assert config.tts_speaker_id == 1
    assert config.tts_lock_speaker_id is True
    from core.tts_markup import resolve_tts_params
    sid, _ = resolve_tts_params({"voice": "other"}, default_sid=config.tts_speaker_id,
                               default_speed=1.0, voice_map=config.tts_speaker_voices,
                               num_speakers=2, lock_speaker_id=config.tts_lock_speaker_id)
    assert sid == 1


def test_kitten_key_round_trips_config_without_changing_default():
    assert SherpaConfig().tts_backend == ""
    assert SherpaConfig.from_dict({"tts_backend": "kitten"}).tts_backend == "kitten"


@pytest.mark.parametrize("backend", ["unknown", "auto", "vits", "kokoro", None, 1])
def test_unknown_explicit_tts_backend_refuses_before_native_config(monkeypatch, backend):
    captured = {}
    monkeypatch.setitem(sys.modules, "sherpa_onnx", _fake_sherpa_onnx(captured))
    with pytest.raises(RuntimeError, match="Unsupported tts_backend"):
        build_tts(SherpaConfig(tts_backend=backend))
    assert captured == {}


@pytest.mark.parametrize("field", ["tts_model", "tts_voices", "tts_tokens", "tts_data_dir"])
@pytest.mark.parametrize("failure", ["empty", "missing", "wrong_type"])
def test_kitten_required_assets_fail_before_cpp(monkeypatch, tmp_path, field, failure):
    config = _kitten_files(tmp_path)
    if failure == "empty":
        setattr(config, field, "")
    elif failure == "missing":
        setattr(config, field, str(tmp_path / "absent"))
    elif field == "tts_data_dir":
        setattr(config, field, config.tts_tokens)
    else:
        setattr(config, field, config.tts_data_dir)
    captured = {}
    monkeypatch.setitem(sys.modules, "sherpa_onnx", _fake_sherpa_onnx(captured))
    with pytest.raises(RuntimeError, match="Kitten TTS"):
        build_tts(config)
    assert captured == {}


@pytest.mark.parametrize("field", ["tts_model", "tts_voices", "tts_tokens"])
def test_empty_kitten_asset_files_fail_before_cpp(monkeypatch, tmp_path, field):
    config = _kitten_files(tmp_path)
    Path(getattr(config, field)).write_bytes(b"")
    captured = {}
    monkeypatch.setitem(sys.modules, "sherpa_onnx", _fake_sherpa_onnx(captured))
    with pytest.raises(RuntimeError, match="Kitten TTS"):
        build_tts(config)
    assert captured == {}


def test_kitten_empty_phonemizer_directory_is_not_an_admission(monkeypatch, tmp_path):
    config = _kitten_files(tmp_path)
    Path(config.tts_data_dir, "phondata").unlink()
    captured = {}
    monkeypatch.setitem(sys.modules, "sherpa_onnx", _fake_sherpa_onnx(captured))
    with pytest.raises(RuntimeError, match="Kitten TTS"):
        build_tts(config)
    assert captured == {}


@pytest.mark.parametrize("metadata", [None, {}, {"model_type": "vits"}, {"model_type": "kokoro"}, {"model_type": "kitten"}])
def test_kitten_metadata_must_be_conclusive_before_cpp(monkeypatch, tmp_path, metadata):
    config = _kitten_files(tmp_path)
    if metadata is None:
        Path(config.tts_model).write_bytes(b"\x0b\x00invalid protobuf")
    else:
        _stub_onnx(Path(config.tts_model), metadata)
    captured = {}
    monkeypatch.setitem(sys.modules, "sherpa_onnx", _fake_sherpa_onnx(captured))
    with pytest.raises(RuntimeError, match="Kitten TTS"):
        build_tts(config)
    assert captured == {}


@pytest.mark.parametrize("key,value", [
    ("style_dim", "2,4,8"), ("style_dim", "0,4"), ("n_speakers", "0"),
    ("sample_rate", "nan"), ("has_espeak", "0"), ("voice", "unavailable"),
    ("speaker_speed_priors", "0.8"), ("speaker_speed_priors", "nan,0.8"),
    ("version", "9"), ("max_token_len", "zero"), ("end_id", "broken"),
    ("add_pad_after_end", "2"),
])
def test_kitten_loader_metadata_errors_are_refused_before_cpp(monkeypatch, tmp_path, key, value):
    config = _kitten_files(tmp_path)
    metadata = read_onnx_custom_metadata(config.tts_model)
    metadata[key] = value
    _stub_onnx(Path(config.tts_model), metadata)
    captured = {}
    monkeypatch.setitem(sys.modules, "sherpa_onnx", _fake_sherpa_onnx(captured))
    with pytest.raises(RuntimeError, match="Kitten TTS"):
        build_tts(config)
    assert captured == {}


def test_kitten_wrong_voices_shape_is_refused_before_cpp(monkeypatch, tmp_path):
    config = _kitten_files(tmp_path)
    Path(config.tts_voices).write_bytes(b"wrong-family voices")
    captured = {}
    monkeypatch.setitem(sys.modules, "sherpa_onnx", _fake_sherpa_onnx(captured))
    with pytest.raises(RuntimeError, match="voices.bin size"):
        build_tts(config)
    assert captured == {}


@pytest.mark.parametrize("sid", [-1, 2, True])
def test_kitten_default_sid_cannot_abort_later_synthesis(monkeypatch, tmp_path, sid):
    config = _kitten_files(tmp_path)
    config.tts_speaker_id = sid
    captured = {}
    monkeypatch.setitem(sys.modules, "sherpa_onnx", _fake_sherpa_onnx(captured))
    with pytest.raises(RuntimeError, match="tts_speaker_id"):
        build_tts(config)
    assert captured == {}


@pytest.mark.parametrize("api", ["OfflineTtsKittenModelConfig", "OfflineTtsModelConfig", "OfflineTts"])
def test_kitten_missing_native_api_cannot_fallback(monkeypatch, tmp_path, api):
    config = _kitten_files(tmp_path)
    captured = {}
    module = _fake_sherpa_onnx(captured)
    delattr(module, api)
    monkeypatch.setitem(sys.modules, "sherpa_onnx", module)
    with pytest.raises(RuntimeError, match="native API is unavailable"):
        build_tts(config)
    assert captured == {}


def test_kitten_native_build_error_refuses_without_muted_fallback(monkeypatch, tmp_path):
    config = _kitten_files(tmp_path)
    module = _fake_sherpa_onnx({})
    def failure(_config):
        raise ValueError("private native diagnostic")
    module.OfflineTts = failure
    monkeypatch.setitem(sys.modules, "sherpa_onnx", module)
    with pytest.raises(RuntimeError, match="^Kitten TTS native construction failed$"):
        build_tts(config)


@pytest.mark.parametrize("voices_present", [False, True])
def test_kitten_model_cannot_enter_legacy_vits_or_kokoro_path(monkeypatch, tmp_path, voices_present):
    config = _kitten_files(tmp_path)
    config.tts_backend = ""
    if not voices_present:
        config.tts_voices = ""
    captured = {}
    monkeypatch.setitem(sys.modules, "sherpa_onnx", _fake_sherpa_onnx(captured))
    with pytest.raises(RuntimeError, match="explicit tts_backend"):
        build_tts(config)
    assert captured == {}


def test_strict_kitten_metadata_rejects_duplicate_family_tags(monkeypatch, tmp_path):
    config = _kitten_files(tmp_path)
    blob = Path(config.tts_model).read_bytes()
    key, value = b"model_type", b"kitten-tts"
    entry = b"\x0a" + _varint(len(key)) + key + b"\x12" + _varint(len(value)) + value
    Path(config.tts_model).write_bytes(blob + b"\x72" + _varint(len(entry)) + entry)
    assert read_onnx_custom_metadata(config.tts_model)["model_type"] == "kitten-tts"
    assert read_onnx_custom_metadata(config.tts_model, strict=True) is None
    captured = {}
    monkeypatch.setitem(sys.modules, "sherpa_onnx", _fake_sherpa_onnx(captured))
    with pytest.raises(RuntimeError, match="conclusive"):
        build_tts(config)
    assert captured == {}


def test_strict_kitten_metadata_refuses_truncation_and_oversized_metadata(tmp_path):
    path = tmp_path / "truncated.onnx"
    path.write_bytes(b"\x72\x7fshort")
    assert read_onnx_custom_metadata(str(path), strict=True) is None
    path = tmp_path / "oversized.onnx"
    _stub_onnx(path, {"model_type": "kitten-tts", "extra": "x" * 65536})
    assert read_onnx_custom_metadata(str(path), strict=True) is None
