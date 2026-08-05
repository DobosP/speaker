"""Atomic final-STT profile, runtime-bound, and readiness parity contracts."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.config import (
    FINAL_STT_PROFILE_NAMES,
    apply_device_profile,
    apply_final_stt_profile,
    deep_merge,
)
from core.engines import _faster_whisper
from core.engine import EngineCallbacks
from core.engines._sherpa_models import (
    build_final_recognizer,
    build_final_verifier,
)
from core.engines.file_replay import FileReplayEngine
from core.engines.sherpa import SherpaConfig, SherpaOnnxEngine


ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize(
    ("name", "backend", "verifier", "threads"),
    [
        ("sense-voice", "sense_voice", "", 0),
        (
            "parakeet-faster-whisper",
            "nemo_transducer",
            "faster_whisper",
            1,
        ),
    ],
)
def test_committed_profiles_replace_all_final_fields_without_mutating_input(
    name, backend, verifier, threads
):
    config = _config()
    before = copy.deepcopy(config)

    effective, metadata = apply_final_stt_profile(config, name)

    assert config == before
    selected = config["final_stt_profiles"][name]
    assert {
        key: value
        for key, value in effective["sherpa"].items()
        if key.startswith("asr_final_")
    } == selected["sherpa"]
    assert effective["sherpa"]["asr_final_backend"] == backend
    assert effective["sherpa"]["asr_final_verifier_backend"] == verifier
    assert effective["sherpa"]["asr_final_verifier_cpu_threads"] == threads
    assert metadata.name == name
    assert metadata.schema_version == 1
    assert metadata.sha256 == _canonical_sha256(selected)
    assert len(metadata.sha256) == 64


def test_profile_map_is_wholesale_opaque_during_local_merge():
    base = _config()
    replacement = {
        "sense-voice": copy.deepcopy(base["final_stt_profiles"]["sense-voice"])
    }

    merged = deep_merge(base, {"final_stt_profiles": replacement})

    assert merged["final_stt_profiles"] == replacement
    assert "parakeet-faster-whisper" not in merged["final_stt_profiles"]
    assert base["final_stt_profiles"] != replacement


def test_profile_applied_after_device_profile_wins_every_final_field():
    config = _config()
    config["device_profiles"]["desktop"]["sherpa"][
        "asr_final_model"
    ] = "SENTINEL_DEVICE_MODEL"

    device_effective = apply_device_profile(config, "desktop", strict=True)
    effective, _metadata = apply_final_stt_profile(device_effective, "sense-voice")

    assert effective["sherpa"]["asr_final_model"].endswith(
        "sense_voice/model.int8.onnx"
    )


@pytest.mark.parametrize(
    ("name", "key", "value"),
    [
        ("sense-voice", "asr_final_use_itn", False),
        ("sense-voice", "asr_final_language", "auto"),
        ("sense-voice", "asr_final_hr_lexicon", "local.txt"),
        ("sense-voice", "asr_final_min_sec", 0.6),
        ("sense-voice", "asr_final_async", False),
        ("sense-voice", "asr_final_required", False),
        ("sense-voice", "asr_final_preroll_sec", 1.0),
        ("sense-voice", "asr_final_verifier_cpu_threads", 1),
        ("parakeet-faster-whisper", "asr_final_verifier_cpu_threads", 2),
    ],
)
def test_fixed_profile_identity_rejects_policy_retuning(name, key, value):
    config = _config()
    config["final_stt_profiles"][name]["sherpa"][key] = value

    with pytest.raises(ValueError, match=key):
        apply_final_stt_profile(config, name)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda config: config["final_stt_profiles"].pop("sense-voice"),
        lambda config: config["final_stt_profiles"]["sense-voice"].update(
            schema_version=1.0
        ),
        lambda config: config["final_stt_profiles"]["sense-voice"]["sherpa"].pop(
            "asr_final_rule_fsts"
        ),
        lambda config: config["final_stt_profiles"]["sense-voice"]["sherpa"].update(
            unexpected="value"
        ),
        lambda config: config["final_stt_profiles"]["sense-voice"]["sherpa"].update(
            asr_final_model=""
        ),
        lambda config: config["final_stt_profiles"][
            "parakeet-faster-whisper"
        ]["sherpa"].update(asr_final_verifier_backend=""),
    ],
)
def test_complete_profile_map_and_backend_artifacts_fail_closed(mutation):
    config = _config()
    mutation(config)

    with pytest.raises(ValueError):
        apply_final_stt_profile(config, "sense-voice")


def test_unknown_or_unsafe_profile_name_is_rejected():
    config = _config()
    for name in ("other", "sense-voice/../../other", "Sense-Voice"):
        with pytest.raises(ValueError, match="unknown final STT profile"):
            apply_final_stt_profile(config, name)
    assert FINAL_STT_PROFILE_NAMES == (
        "sense-voice",
        "parakeet-faster-whisper",
    )


def test_faster_whisper_explicit_cpu_bound_reaches_ctranslate_constructor(tmp_path):
    calls = []

    class Model:
        pass

    def factory(*args, **kwargs):
        calls.append((args, kwargs))
        return Model()

    recognizer = _faster_whisper.FasterWhisperEndpointRecognizer(
        tmp_path,
        cpu_threads=1,
        model_factory=factory,
    )
    recognizer.warm()

    assert calls[0][1]["cpu_threads"] == 1


@pytest.mark.parametrize("threads", [-1, 257, 1.5, True])
def test_faster_whisper_rejects_invalid_cpu_thread_bound(tmp_path, threads):
    with pytest.raises(ValueError, match="cpu_threads"):
        _faster_whisper.FasterWhisperEndpointRecognizer(
            tmp_path,
            cpu_threads=threads,
            model_factory=lambda *_args, **_kwargs: object(),
        )


@pytest.mark.parametrize(("threads", "expected"), [(0, {}), (1, {"cpu_threads": 1})])
def test_runtime_verifier_builder_omits_legacy_zero_and_forwards_profile_bound(
    tmp_path, monkeypatch, threads, expected
):
    calls = []

    class Verifier:
        def __init__(self, path, **kwargs):
            calls.append((path, kwargs))

    monkeypatch.setattr(
        _faster_whisper,
        "FasterWhisperEndpointRecognizer",
        Verifier,
    )
    config = SherpaConfig(
        asr_final_verifier_backend="faster_whisper",
        asr_final_verifier_model=str(tmp_path),
        asr_final_verifier_cpu_threads=threads,
    )

    assert isinstance(build_final_verifier(config), Verifier)
    assert calls == [(str(tmp_path), expected)]


def test_required_verifier_warm_failure_is_strict_but_optional_stays_lazy(
    tmp_path, monkeypatch
):
    events = []

    class Verifier:
        def __init__(self, path, **kwargs):
            events.append(("construct", path, kwargs))

        def warm(self):
            events.append(("warm",))
            raise RuntimeError("corrupt verifier model")

    monkeypatch.setattr(
        _faster_whisper,
        "FasterWhisperEndpointRecognizer",
        Verifier,
    )
    required = SherpaConfig(
        asr_final_verifier_backend="faster_whisper",
        asr_final_verifier_model=str(tmp_path),
        asr_final_verifier_cpu_threads=1,
        asr_final_required=True,
    )
    optional = SherpaConfig(
        asr_final_verifier_backend="faster_whisper",
        asr_final_verifier_model=str(tmp_path),
        asr_final_verifier_cpu_threads=1,
    )

    with pytest.raises(RuntimeError, match="required faster_whisper.*failed"):
        build_final_verifier(required)
    assert isinstance(build_final_verifier(optional), Verifier)
    assert events == [
        ("construct", str(tmp_path), {"cpu_threads": 1}),
        ("warm",),
        ("construct", str(tmp_path), {"cpu_threads": 1}),
    ]


def test_doctor_applies_profile_before_readiness_and_reports_safe_digest(monkeypatch):
    import tools.doctor as doctor

    observed = []

    def fake_runtime_checks(config, **kwargs):
        observed.append((config, kwargs))
        return [doctor.Check("runtime", True, "ready")]

    monkeypatch.setattr(doctor, "run_runtime_checks", fake_runtime_checks)

    checks = doctor.run_all(
        _config(),
        device="desktop",
        final_stt_profile="parakeet-faster-whisper",
    )

    assert observed[0][0]["sherpa"]["asr_final_backend"] == "nemo_transducer"
    assert observed[0][0]["sherpa"]["asr_final_verifier_cpu_threads"] == 1
    identity = next(check for check in checks if check.name == "final STT profile")
    assert "parakeet-faster-whisper; sha256=" in identity.detail
    assert "pretrained_models" not in identity.detail


def test_readiness_warm_probe_uses_same_explicit_cpu_bound(monkeypatch):
    from core import readiness

    calls = []

    class Recognizer:
        def __init__(self, path, **kwargs):
            calls.append((path, kwargs))

        def warm(self):
            calls.append(("warm", {}))

    class CTranslate2:
        @staticmethod
        def get_supported_compute_types(device, index):
            assert (device, index) == ("cuda", 0)
            return {"float16"}

    monkeypatch.setattr(
        _faster_whisper,
        "FasterWhisperEndpointRecognizer",
        Recognizer,
    )
    imports = {
        "faster_whisper": object(),
        "ctranslate2": CTranslate2(),
    }
    result = readiness.check_asr_final_verifier_runtime(
        {
            "sherpa": {
                "asr_final_verifier_backend": "faster_whisper",
                "asr_final_verifier_model": "/local/model",
                "asr_final_verifier_cpu_threads": 1,
            }
        },
        import_fn=imports.__getitem__,
        preload_fn=lambda: None,
    )

    assert result.ok
    assert calls == [("/local/model", {"cpu_threads": 1}), ("warm", {})]


def _failing_sense_voice_module(message: str = "corrupt SenseVoice model"):
    class OfflineRecognizer:
        @staticmethod
        def from_sense_voice(**_kwargs):
            raise RuntimeError(message)

    return SimpleNamespace(
        __version__="1.13.3",
        OfflineRecognizer=OfflineRecognizer,
    )


def _sense_voice_config(tmp_path, *, required: bool) -> SherpaConfig:
    model = tmp_path / "sense-model.onnx"
    tokens = tmp_path / "sense-tokens.txt"
    model.write_bytes(b"corrupt-model")
    tokens.write_text("token 0\n", encoding="utf-8")
    return SherpaConfig(
        endpoint_enabled=False,
        asr_final_backend="sense_voice",
        asr_final_model=str(model),
        asr_final_tokens=str(tokens),
        asr_final_required=required,
    )


def test_required_sense_voice_constructor_failure_is_strict_but_legacy_falls_back(
    tmp_path, monkeypatch
):
    monkeypatch.setitem(sys.modules, "sherpa_onnx", _failing_sense_voice_module())

    with pytest.raises(RuntimeError, match="required sense_voice.*failed to build"):
        build_final_recognizer(_sense_voice_config(tmp_path, required=True))

    assert build_final_recognizer(
        _sense_voice_config(tmp_path, required=False)
    ) is None


def test_required_sense_voice_readiness_probes_decode_and_fails_closed():
    from core import readiness

    config = {
        "sherpa": {
            "asr_final_backend": "sense_voice",
            "asr_final_required": True,
        }
    }
    observed = []
    selected = readiness.check_asr_final_runtime(
        config,
        import_fn=lambda _name: SimpleNamespace(__version__="1.13.3"),
        model_probe_fn=lambda sherpa: observed.append(dict(sherpa)),
    )
    failed = readiness.check_asr_final_runtime(
        config,
        import_fn=lambda _name: SimpleNamespace(__version__="1.13.3"),
        model_probe_fn=lambda _sherpa: (_ for _ in ()).throw(
            RuntimeError("corrupt constructor")
        ),
    )

    assert selected.ok
    assert observed == [config["sherpa"]]
    assert not failed.ok
    assert "corrupt constructor" in failed.detail


def test_required_sense_voice_readiness_rejects_corrupt_model_via_shared_builder(
    tmp_path, monkeypatch
):
    from core import readiness

    module = _failing_sense_voice_module()
    monkeypatch.setitem(sys.modules, "sherpa_onnx", module)
    result = readiness.check_asr_final_runtime(
        {"sherpa": vars(_sense_voice_config(tmp_path, required=True))},
        import_fn=lambda name: module if name == "sherpa_onnx" else object(),
    )

    assert not result.ok
    assert "required sense_voice final recognizer failed to build" in result.detail


def test_runtime_checks_routes_required_sense_voice_to_strict_probe(monkeypatch):
    from core import readiness

    observed = []

    def probe(config, *, import_fn):
        observed.append((config, import_fn))
        return readiness.Check("ASR final runtime", False, "strict probe ran")

    monkeypatch.setattr(readiness, "check_asr_final_runtime", probe)

    def import_fn(_name):
        return object()

    checks = readiness.run_runtime_checks(
        {
            "sherpa": {
                "asr_final_backend": "sense_voice",
                "asr_final_required": True,
            },
            "llm": {"backend": "ollama"},
        },
        resolved=True,
        llm_mode="echo",
        import_fn=import_fn,
        exists=lambda _path: True,
        platform="win32",
        include_speaker=False,
        require_audio_devices=False,
        require_os_echo_route=False,
    )

    assert len(observed) == 1
    assert observed[0][0]["sherpa"]["asr_final_required"] is True
    assert observed[0][1] is import_fn
    assert any(check.detail == "strict probe ran" for check in checks)


def test_unprofiled_sense_voice_keeps_legacy_readiness_fallback():
    from core import readiness

    result = readiness.check_asr_final_runtime(
        {"sherpa": {"asr_final_backend": "sense_voice"}},
        import_fn=lambda _name: pytest.fail("unprofiled SenseVoice was imported"),
        model_probe_fn=lambda _sherpa: pytest.fail("unprofiled SenseVoice was probed"),
    )

    assert result.ok
    assert result.detail == "NeMo transducer disabled"


def test_required_sense_voice_engine_fails_before_audio_device_query(
    tmp_path, monkeypatch
):
    import core.engines.sherpa as sherpa_engine

    events = []
    monkeypatch.setitem(sys.modules, "sherpa_onnx", _failing_sense_voice_module())
    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        SimpleNamespace(query_devices=lambda *_args, **_kwargs: events.append("device")),
    )
    monkeypatch.setattr(sherpa_engine, "build_recognizer", lambda _config: object())
    engine = SherpaOnnxEngine(_sense_voice_config(tmp_path, required=True))

    with pytest.raises(RuntimeError, match="required sense_voice.*failed to build"):
        engine.start(EngineCallbacks())

    assert events == []
    assert not engine._running.is_set()


def test_required_sense_voice_replay_fails_in_central_builder(
    tmp_path, monkeypatch
):
    import core.engines.file_replay as file_replay

    monkeypatch.setitem(sys.modules, "sherpa_onnx", _failing_sense_voice_module())
    monkeypatch.setattr(file_replay, "build_recognizer", lambda _config: object())
    engine = FileReplayEngine(
        _sense_voice_config(tmp_path, required=True),
        asr_only=True,
    )

    with pytest.raises(RuntimeError, match="required sense_voice.*failed to build"):
        engine.start(EngineCallbacks())

    assert not engine._started
