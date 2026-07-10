"""Tests for the install/preflight tooling (tools.setup_models + tools.doctor).

All pure / injected fakes -- no network, no audio, no Ollama, no models.
"""
from __future__ import annotations

import os

import pytest

from tools.doctor import (
    PipeWireState,
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
    run_runtime_checks,
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
    assert "config.local.json" in bad.detail
    assert "config.local.json" in bad.hint

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
    assert "config.local.json" in res.detail


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


def test_audio_frontend_inert_apm_backend_never_requires_livekit():
    """The backend string is inert while AEC is disabled."""
    config = {"sherpa": {"aec_enabled": False, "aec_backend": "apm"}}
    checks = check_audio_frontend(config, import_fn=_no_livekit)
    assert _apm_check(checks) is None


def test_audio_frontend_active_apm_backend_fails_without_livekit():
    """When the RESOLVED profile actually selects apm, livekit IS required: absent
    -> a real FAIL (AEC would silently fail open to no echo cancellation)."""
    config = {"sherpa": {"aec_enabled": True, "aec_backend": "apm"}}
    checks = check_audio_frontend(config, import_fn=_no_livekit)
    apm = _apm_check(checks)
    assert apm is not None and apm.ok is False


def test_audio_frontend_active_apm_ok_when_livekit_present():
    class _RTC:
        AudioProcessingModule = object()

    def have_livekit(name):
        return _RTC() if "livekit" in name else object()

    config = {"sherpa": {"aec_enabled": True, "aec_backend": "apm"}}
    apm = _apm_check(check_audio_frontend(config, import_fn=have_livekit))
    assert apm is not None and apm.ok is True


def test_audio_frontend_no_apm_anywhere_emits_no_livekit_check():
    config = {
        "device": "desktop",
        "sherpa": {"aec_backend": "nlms"},
        "device_profiles": {"desktop": {"sherpa": {"aec_backend": "nlms"}}},
    }
    assert _apm_check(check_audio_frontend(config, import_fn=_no_livekit)) is None


# --- ADR-0013 word-cut EC preflight (run-20260706-231226 launched degraded with
# --- no module-echo-cancel loaded and nothing said so) ---------------------------


def _word_cut_route_check(checks):
    for c in checks:
        if "word-cut" in c.name.lower() and "route" in c.name.lower():
            return c
    return None


def _word_cut_vad_check(checks):
    for c in checks:
        if "word-cut vad" in c.name.lower():
            return c
    return None


_EC_ROUTED = PipeWireState(
    modules=(
        "42\tmodule-echo-cancel\taec_method=webrtc "
        "source_name=ec_source sink_name=ec_sink"
    ),
    sources="51\tec_source\tPipeWire\tfloat32le 1ch 16000Hz\tRUNNING",
    sinks="52\tec_sink\tPipeWire\tfloat32le 2ch 48000Hz\tRUNNING",
    default_source="ec_source",
    default_sink="ec_sink",
)


def test_word_cut_ec_check_fails_when_module_missing():
    config = {"sherpa": {
        "barge_word_cut_enabled": True,
        "aec_enabled": False,
        "vad_model": "/m/vad.onnx",
    }}
    checks = check_audio_frontend(
        config,
        import_fn=_no_livekit,
        exists=lambda _p: True,
        platform="linux",
        pipewire_state=PipeWireState(),
    )
    c = _word_cut_route_check(checks)
    assert c is not None and c.ok is False
    assert "module-echo-cancel" in c.hint  # the exact fix, not just a complaint


def test_word_cut_ec_check_passes_when_module_loaded():
    config = {"sherpa": {
        "barge_word_cut_enabled": True,
        "aec_enabled": False,
        "vad_model": "/m/vad.onnx",
    }}
    checks = check_audio_frontend(
        config,
        import_fn=_no_livekit,
        exists=lambda _p: True,
        platform="linux",
        pipewire_state=_EC_ROUTED,
    )
    c = _word_cut_route_check(checks)
    assert c is not None and c.ok is True


def test_word_cut_ec_module_loaded_but_raw_defaults_fails_route():
    state = PipeWireState(
        modules=_EC_ROUTED.modules,
        sources=_EC_ROUTED.sources,
        sinks=_EC_ROUTED.sinks,
        default_source="alsa_input.raw",
        default_sink="alsa_output.raw",
    )
    checks = check_audio_frontend(
        {"sherpa": {
            "barge_word_cut_enabled": True,
            "aec_enabled": False,
            "vad_model": "/m/vad.onnx",
        }},
        import_fn=_no_livekit,
        exists=lambda _p: True,
        platform="linux",
        pipewire_state=state,
    )
    route = _word_cut_route_check(checks)
    assert route is not None and not route.ok
    assert "capture" in route.detail and "playback" in route.detail


def test_word_cut_explicit_ec_routes_do_not_require_ec_defaults():
    state = PipeWireState(
        modules=_EC_ROUTED.modules,
        sources=_EC_ROUTED.sources,
        sinks=_EC_ROUTED.sinks,
        default_source="alsa_input.raw",
        default_sink="alsa_output.raw",
    )
    checks = check_audio_frontend(
        {"sherpa": {
            "barge_word_cut_enabled": True,
            "aec_enabled": False,
            "vad_model": "/m/vad.onnx",
            "input_device": "ec_source",
            "output_device": "ec_sink",
        }},
        import_fn=_no_livekit,
        exists=lambda _p: True,
        platform="linux",
        pipewire_state=state,
    )
    route = _word_cut_route_check(checks)
    assert route is not None and route.ok
    assert "capture=ec_source" in route.detail
    assert "playback=ec_sink" in route.detail


def test_word_cut_module_arguments_do_not_count_as_existing_nodes():
    checks = check_audio_frontend(
        {"sherpa": {
            "barge_word_cut_enabled": True,
            "aec_enabled": False,
            "vad_model": "/m/vad.onnx",
        }},
        import_fn=_no_livekit,
        exists=lambda _p: True,
        platform="linux",
        pipewire_state=PipeWireState(
            modules=_EC_ROUTED.modules,
            default_source="ec_source",
            default_sink="ec_sink",
        ),
    )
    route = _word_cut_route_check(checks)
    assert route is not None and not route.ok
    assert "nodes were not found" in route.detail


def test_word_cut_requires_configured_existing_vad_model():
    checks = check_audio_frontend(
        {"sherpa": {"barge_word_cut_enabled": True, "aec_enabled": False}},
        import_fn=_no_livekit,
        platform="win32",
    )
    vad = _word_cut_vad_check(checks)
    assert vad is not None and not vad.ok
    assert "vad_model" in vad.detail


def test_stale_word_cut_flag_is_inert_when_barge_is_globally_disabled():
    checks = check_audio_frontend(
        {"sherpa": {
            "barge_in_enabled": False,
            "barge_word_cut_enabled": True,
            "aec_enabled": False,
        }},
        import_fn=_no_livekit,
        platform="linux",
        pipewire_state=PipeWireState(),
    )
    assert _word_cut_vad_check(checks) is None
    assert _word_cut_route_check(checks) is None


def test_word_cut_ec_check_absent_when_path_not_selected():
    # In-app AEC on -> word-cut inert (ADR-0013 scoping) -> no check; flag off
    # -> no check. The preflight must never nag configs that don't use the path.
    for sherpa in (
        {"barge_word_cut_enabled": True, "aec_enabled": True},
        {"barge_word_cut_enabled": False, "aec_enabled": False},
    ):
        checks = check_audio_frontend(
            {"sherpa": sherpa}, import_fn=_no_livekit,
            platform="linux", pipewire_state=PipeWireState(),
        )
        assert _word_cut_route_check(checks) is None


def test_word_cut_ec_check_absent_off_linux():
    # Windows uses WASAPI communications capture, not PipeWire -- no pactl check.
    config = {"sherpa": {
        "barge_word_cut_enabled": True,
        "aec_enabled": False,
        "vad_model": "/m/vad.onnx",
    }}
    checks = check_audio_frontend(
        config,
        import_fn=_no_livekit,
        exists=lambda _p: True,
        platform="win32",
    )
    assert _word_cut_route_check(checks) is None


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


def test_run_all_device_override_drives_audio_frontend_too():
    """`doctor --device` must not select one profile for LLM and another for DSP."""
    paths = {
        key: f"/m/{key}"
        for key in (
            "asr_tokens", "asr_encoder", "asr_decoder", "asr_joiner",
            "tts_model", "tts_tokens",
        )
    }
    cfg = {
        "device": "desktop",
        "sherpa": {
            **paths,
            "aec_enabled": False,
            "aec_backend": "nlms",
            "barge_word_cut_enabled": True,
            "vad_model": "/m/vad.onnx",
        },
        "llm": {"backend": "ollama", "main_model": "gemma3:4b"},
        "device_profiles": {
            "desktop": {},
            "open_speaker": {
                "sherpa": {
                    "aec_enabled": True,
                    "aec_backend": "apm",
                    "barge_word_cut_enabled": True,
                }
            },
        },
    }
    checks = run_all(
        cfg,
        device="open_speaker",
        sd=_FakeSD16k(),
        ollama_lister=lambda: ["gemma3:4b"],
        import_fn=_no_livekit,
        exists=lambda _p: True,
        pipewire_state=PipeWireState(),
    )
    apm = _apm_check(checks)
    assert apm is not None and not apm.ok
    # The selected profile has in-app AEC, so word-cut is inert and OS EC/VAD
    # are not required. Seeing either would prove profile resolution drift.
    assert _word_cut_route_check(checks) is None
    assert _word_cut_vad_check(checks) is None


def test_doctor_invalid_device_is_a_clean_check_not_a_traceback(monkeypatch, capsys):
    import core.app as app
    import tools.doctor as doctor

    monkeypatch.setattr(app, "_load_config", lambda *args, **kwargs: {
        "device": "desktop",
        "device_profiles": {"desktop": {}},
        "sherpa": {},
    })
    assert doctor.main(["--device", "does-not-exist"]) == 1
    text = capsys.readouterr().out.lower()
    assert "[fail] device profile" in text
    assert "desktop" in text


def test_runtime_checks_echo_never_imports_or_contacts_ollama():
    paths = {
        key: f"/m/{key}"
        for key in (
            "asr_tokens", "asr_encoder", "asr_decoder", "asr_joiner",
            "tts_model", "tts_tokens",
        )
    }

    def imports(name):
        if name == "ollama":
            raise AssertionError("EchoLLM readiness touched Ollama")
        return object()

    def list_ollama():
        raise AssertionError("EchoLLM readiness contacted Ollama")

    checks = run_runtime_checks(
        {"sherpa": paths, "llm": {"backend": "ollama"}},
        resolved=True,
        llm_mode="echo",
        sd=_FakeSD16k(),
        ollama_lister=list_ollama,
        import_fn=imports,
        exists=lambda _p: True,
        platform="win32",
    )
    assert all(check.ok for check in checks)
    assert not any("ollama" in check.name for check in checks)


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
