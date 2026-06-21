"""Tests for the install/preflight tooling (tools.setup_models + tools.doctor).

All pure / injected fakes -- no network, no audio, no Ollama, no models.
"""
from __future__ import annotations

import os

import pytest

from tools.doctor import (
    check_audio,
    check_audio_frontend,
    check_imports,
    check_ollama,
    check_platform,
    check_python,
    check_sherpa_models,
    check_speaker_id,
    profile_ollama_models,
    run_all,
    summarize,
)


class _FakeSD16k:
    def query_devices(self, kind):
        return {"name": "d", "default_samplerate": 16000.0}
from tools.setup_models import dest_for, extract_member, wire_sherpa_paths


def test_asr_and_tts_tokens_download_to_separate_dirs():
    # Both models ship a file literally named tokens.txt; they must not share a
    # folder or one clobbers the other (wrong ASR vocab -> garbage + crash).
    assert dest_for("/m", "asr_tokens") != dest_for("/m", "tts_tokens")
    assert dest_for("/m", "asr_encoder") == dest_for("/m", "asr_tokens")


def test_apply_accuracy_high_uses_fp32_weights():
    from tools.setup_models import apply_accuracy

    m = {
        "asr_encoder": {"file": "encoder-x.int8.onnx"},
        "asr_joiner": {"file": "joiner-x.int8.onnx"},
        "asr_decoder": {"file": "decoder-x.onnx"},
    }
    high = apply_accuracy({k: dict(v) for k, v in m.items()}, "high")
    assert high["asr_encoder"]["file"] == "encoder-x.onnx"
    assert high["asr_joiner"]["file"] == "joiner-x.onnx"

    fast = apply_accuracy({k: dict(v) for k, v in m.items()}, "fast")
    assert fast["asr_encoder"]["file"] == "encoder-x.int8.onnx"


# --- setup_models.wire_sherpa_paths -----------------------------------------


def test_wire_sherpa_paths_sets_makes_absolute_and_preserves():
    cfg = {"sherpa": {"sample_rate": 16000}, "llm": {"backend": "ollama"}}
    out = wire_sherpa_paths(
        cfg,
        {"asr_encoder": "m/enc.onnx", "tts_data_dir": ""},
        abspath=lambda p: "/abs/" + p,
    )
    assert out["sherpa"]["asr_encoder"] == "/abs/m/enc.onnx"
    assert out["sherpa"]["sample_rate"] == 16000  # untouched field preserved
    assert "tts_data_dir" not in out["sherpa"]  # empty path skipped
    assert out["llm"] == {"backend": "ollama"}  # other sections preserved


def test_wire_sherpa_paths_creates_sherpa_section_when_absent():
    cfg: dict = {}
    wire_sherpa_paths(cfg, {"asr_tokens": "t.txt"}, abspath=lambda p: p)
    assert cfg["sherpa"]["asr_tokens"] == "t.txt"


# --- setup_models.extract_member (punctuation archive unpack) ----------------


def test_extract_member_flattens_nested_model(tmp_path):
    import io
    import tarfile

    # A release-style archive: model.onnx nested inside a directory.
    archive = tmp_path / "punct.tar.bz2"
    payload = b"ONNXMODELBYTES"
    with tarfile.open(archive, "w:bz2") as tar:
        info = tarfile.TarInfo("sherpa-onnx-punct-xyz/model.onnx")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
        readme = tarfile.TarInfo("sherpa-onnx-punct-xyz/README.md")
        readme.size = 3
        tar.addfile(readme, io.BytesIO(b"hi\n"))

    out = extract_member(str(archive), "model.onnx", str(tmp_path / "out"))
    assert out.endswith("model.onnx")
    assert os.path.basename(out) == "model.onnx"  # flattened, no nested dir
    with open(out, "rb") as fh:
        assert fh.read() == payload


def test_extract_member_missing_raises(tmp_path):
    import io
    import tarfile

    archive = tmp_path / "empty.tar.bz2"
    with tarfile.open(archive, "w:bz2") as tar:
        info = tarfile.TarInfo("notes.txt")
        info.size = 3
        tar.addfile(info, io.BytesIO(b"hi\n"))

    with pytest.raises(FileNotFoundError):
        extract_member(str(archive), "model.onnx", str(tmp_path / "out"))


# --- doctor checks -----------------------------------------------------------


def test_check_python_version_gate():
    assert check_python((3, 11, 0)).ok
    assert not check_python((3, 8, 0)).ok


def test_check_platform_names_os_and_venv_state():
    # Always OK (informational); names the OS and venv state.
    win = check_platform("win32", in_venv=True)
    assert win.ok and "Windows" in win.detail and "venv=yes" in win.detail
    lin = check_platform("linux", in_venv=False)
    assert lin.ok and "Linux" in lin.detail and "venv=no" in lin.detail
    # Not-in-venv nudges toward the installer but never fails readiness.
    assert "install" in lin.hint
    assert check_platform("darwin", in_venv=True).detail.startswith("macOS")


def test_check_imports_flags_missing_with_pip_hint():
    def fake_import(name):
        if name == "numpy":
            return object()
        raise ImportError("nope")

    checks = check_imports(("numpy", "sherpa_onnx"), import_fn=fake_import)
    assert checks[0].ok
    assert not checks[1].ok
    assert "pip install sherpa-onnx" in checks[1].hint  # underscore -> dashed pip name


def test_check_sherpa_models_missing_then_present():
    bad = check_sherpa_models({"sherpa": {}})
    assert not bad.ok
    assert "setup_models" in bad.hint

    paths = {
        k: f"/m/{k}"
        for k in ("asr_tokens", "asr_encoder", "asr_decoder", "asr_joiner", "tts_model", "tts_tokens")
    }
    good = check_sherpa_models({"sherpa": paths}, exists=lambda p: True)
    assert good.ok


def test_check_sherpa_models_path_set_but_file_absent():
    paths = {
        k: f"/m/{k}"
        for k in ("asr_tokens", "asr_encoder", "asr_decoder", "asr_joiner", "tts_model", "tts_tokens")
    }
    res = check_sherpa_models({"sherpa": paths}, exists=lambda p: False)
    assert not res.ok
    assert "missing on disk" in res.detail


def test_check_speaker_id_unconfigured_is_ok_advisory():
    # Optional feature: unset model is OK (never blocks readiness), just advised.
    c = check_speaker_id({"sherpa": {}})
    assert c.ok
    assert "optional" in c.detail


def test_check_speaker_id_model_set_but_missing_is_fail():
    c = check_speaker_id(
        {"sherpa": {"speaker_embedding_model": "/m/spk.onnx"}}, exists=lambda p: False
    )
    assert not c.ok
    assert "setup_models" in c.hint


def test_check_speaker_id_model_present_not_enrolled_is_ok_with_nudge():
    c = check_speaker_id(
        {"sherpa": {"speaker_embedding_model": "/m/spk.onnx"}}, exists=lambda p: True
    )
    assert c.ok  # fail-open: not enrolled must not block readiness
    assert "enroll" in c.detail


def test_check_speaker_id_enrolled_is_ok():
    cfg = {
        "sherpa": {
            "speaker_embedding_model": "/m/spk.onnx",
            "speaker_enroll_embedding": "/m/enroll.json",
        }
    }
    c = check_speaker_id(cfg, exists=lambda p: True)
    assert c.ok
    assert "enrollment present" in c.detail


def test_check_ollama_models_present_and_missing():
    checks = check_ollama(("gemma3:4b", "gemma3:12b"), lister=lambda: ["gemma3:4b"])
    by_name = {c.name: c for c in checks}
    assert by_name["ollama"].ok
    assert by_name["ollama model gemma3:4b"].ok
    assert not by_name["ollama model gemma3:12b"].ok
    assert by_name["ollama model gemma3:12b"].hint == "ollama pull gemma3:12b"


def test_check_ollama_unreachable():
    def boom():
        raise RuntimeError("connection refused")

    checks = check_ollama(lister=boom)
    assert len(checks) == 1
    assert not checks[0].ok
    assert "serve" in checks[0].hint


def test_check_audio_with_fake_devices():
    class FakeSD:
        def query_devices(self, kind):
            return {"name": f"{kind}-device", "default_samplerate": 48000.0}

    checks = check_audio(sd=FakeSD())
    assert all(c.ok for c in checks)
    assert any("48000 Hz" in c.detail for c in checks)


def test_check_audio_no_device():
    class FakeSD:
        def query_devices(self, kind):
            raise RuntimeError("no default device")

    checks = check_audio(sd=FakeSD())
    assert all(not c.ok for c in checks)


def test_run_all_and_summarize_reports_not_ready():
    class FakeSD:
        def query_devices(self, kind):
            return {"name": "d", "default_samplerate": 44100.0}

    checks = run_all(
        {"sherpa": {}},
        sd=FakeSD(),
        ollama_lister=lambda: [],  # reachable but no models
        import_fn=lambda name: object(),  # all imports succeed
    )
    ready, text = summarize(checks)
    assert not ready  # sherpa models unset + ollama models missing
    assert "FAIL" in text


# --- check_audio_frontend: apm/livekit gating (must not false-fail clean clones) ---


def _no_livekit(name):
    if "livekit" in name:
        raise ImportError("simulated clean clone: livekit not installed")
    return object()


def _apm_check(checks):
    for c in checks:
        if "apm" in c.name.lower() or "livekit" in c.name.lower():
            return c
    return None


def test_audio_frontend_unselected_apm_profile_is_advisory_not_fail():
    """The committed open_speaker profile DEFINES aec_backend=apm, but the active
    (resolved) profile here is nlms. With livekit absent the apm check must be
    ADVISORY (ok=True) so READY is never blocked on a healthy default box."""
    config = {
        "device": "desktop",
        "sherpa": {"aec_backend": "nlms"},
        "device_profiles": {
            "desktop": {"sherpa": {"aec_backend": "nlms"}},
            "open_speaker": {"sherpa": {"aec_backend": "apm"}},
        },
    }
    checks = check_audio_frontend(config, import_fn=_no_livekit)
    apm = _apm_check(checks)
    assert apm is not None and apm.ok is True          # advisory, not a failure
    assert not any(not c.ok for c in checks)           # nothing here blocks READY


def test_audio_frontend_active_apm_backend_fails_without_livekit():
    """When the RESOLVED profile actually selects apm, livekit IS required: absent
    -> a real FAIL (AEC would silently fail open to no echo cancellation)."""
    config = {
        "device": "open_speaker",
        "sherpa": {"aec_backend": "nlms"},
        "device_profiles": {"open_speaker": {"sherpa": {"aec_backend": "apm"}}},
    }
    checks = check_audio_frontend(config, import_fn=_no_livekit)
    apm = _apm_check(checks)
    assert apm is not None and apm.ok is False


def test_audio_frontend_active_apm_ok_when_livekit_present():
    class _RTC:
        AudioProcessingModule = object()

    def have_livekit(name):
        return _RTC() if "livekit" in name else object()

    config = {
        "device": "open_speaker",
        "sherpa": {"aec_backend": "nlms"},
        "device_profiles": {"open_speaker": {"sherpa": {"aec_backend": "apm"}}},
    }
    apm = _apm_check(check_audio_frontend(config, import_fn=have_livekit))
    assert apm is not None and apm.ok is True


def test_audio_frontend_no_apm_anywhere_emits_no_livekit_check():
    config = {
        "device": "desktop",
        "sherpa": {"aec_backend": "nlms"},
        "device_profiles": {"desktop": {"sherpa": {"aec_backend": "nlms"}}},
    }
    assert _apm_check(check_audio_frontend(config, import_fn=_no_livekit)) is None


# --- doctor validates the SELECTED profile's ollama models (gemma3:1b gap) ------


def test_profile_ollama_models_picks_the_selected_profile():
    cfg = {
        "device": "auto",
        "llm": {"main_model": "gemma3:12b", "fast_model": "gemma3:4b"},
        "device_profiles": {
            "desktop": {"llm": {"main_model": "gemma3:12b", "fast_model": "gemma3:4b"}},
            "open_speaker": {"llm": {"main_model": "gemma3:4b", "fast_model": "gemma3:1b"}},
        },
    }
    assert set(profile_ollama_models(cfg, "open_speaker")) == {"gemma3:4b", "gemma3:1b"}


def test_profile_ollama_models_empty_for_non_ollama_backend():
    cfg = {
        "device": "phone",
        "llm": {"backend": "ollama", "main_model": "gemma3:12b", "fast_model": "gemma3:4b"},
        "device_profiles": {
            "phone": {"llm": {"backend": "llamacpp", "main_model": "a.gguf", "fast_model": "b.gguf"}}
        },
    }
    assert profile_ollama_models(cfg, "phone") == ()  # GGUF, not ollama-pulled


def test_run_all_catches_a_missing_profile_model():
    """`doctor --device open_speaker` on a box without gemma3:1b must FAIL on it --
    the gap that let the 2026-06-21 `gemma3:1b not found` 404 reach a live run."""
    cfg = {
        "device": "auto",
        "sherpa": {},
        "llm": {"main_model": "gemma3:12b", "fast_model": "gemma3:4b"},
        "device_profiles": {
            "open_speaker": {"llm": {"main_model": "gemma3:4b", "fast_model": "gemma3:1b"}}
        },
    }
    checks = run_all(
        cfg, sd=_FakeSD16k(),
        ollama_lister=lambda: ["gemma3:4b", "gemma3:12b"],  # NO gemma3:1b
        import_fn=lambda name: object(), exists=lambda p: True,
        device="open_speaker",
    )
    bad = [c for c in checks if c.name == "ollama model gemma3:1b"]
    assert bad and not bad[0].ok and "ollama pull gemma3:1b" in bad[0].hint


def test_run_all_ready_when_everything_passes():
    class FakeSD:
        def query_devices(self, kind):
            return {"name": "d", "default_samplerate": 16000.0}

    paths = {
        k: f"/m/{k}"
        for k in ("asr_tokens", "asr_encoder", "asr_decoder", "asr_joiner", "tts_model", "tts_tokens")
    }
    checks = run_all(
        {"sherpa": paths},
        sd=FakeSD(),
        ollama_lister=lambda: ["gemma3:12b", "gemma3:4b"],
        import_fn=lambda name: object(),
        exists=lambda p: True,
    )
    ready, _ = summarize(checks)
    assert ready
