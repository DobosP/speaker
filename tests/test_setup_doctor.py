"""Tests for the install/preflight tooling (tools.setup_models + tools.doctor).

All pure / injected fakes -- no network, no audio, no Ollama, no models.
"""
from __future__ import annotations

import os

import pytest

import core.readiness as readiness
from core.minicpm_identity import MINICPM_Q8_CONTRACT
from tools.doctor import (
    PipeWireState,
    check_audio,
    check_audio_frontend,
    check_imports,
    check_llamacpp_abort_runtime,
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


class _VerifiedLlamaCpp:
    __version__ = "0.3.33"
    ggml_abort_callback = staticmethod(lambda callback: callback)
    llama_set_abort_callback = staticmethod(lambda *_args: None)
    llama_get_memory = staticmethod(lambda *_args: object())
    llama_memory_clear = staticmethod(lambda *_args: None)


def test_check_llamacpp_abort_runtime_accepts_pinned_cpu_binding():
    check = check_llamacpp_abort_runtime(
        {"llm": {"backend": "llamacpp", "n_gpu_layers": 0}},
        import_fn=lambda name: _VerifiedLlamaCpp if name == "llama_cpp" else object(),
    )

    assert check.ok
    assert "llama-cpp-python==0.3.33" in check.detail
    assert "n_gpu_layers=0" in check.detail


def test_check_llamacpp_abort_runtime_rejects_version_drift_with_exact_fix():
    class WrongVersion(_VerifiedLlamaCpp):
        __version__ = "0.3.32"

    check = check_llamacpp_abort_runtime(
        {"llm": {"backend": "llamacpp", "n_gpu_layers": 0}},
        import_fn=lambda _name: WrongVersion,
    )

    assert not check.ok
    assert "requires llama-cpp-python==0.3.33" in check.detail
    assert "found 0.3.32" in check.detail
    assert "--force-reinstall llama-cpp-python==0.3.33" in check.hint


def test_check_llamacpp_abort_runtime_rejects_gpu_offload():
    check = check_llamacpp_abort_runtime(
        {"llm": {"backend": "llamacpp", "n_gpu_layers": 1}},
        import_fn=lambda _name: _VerifiedLlamaCpp,
    )

    assert not check.ok
    assert "abort is CPU-only" in check.detail
    assert "n_gpu_layers=1" in check.detail
    assert "set llm.n_gpu_layers=0" in check.hint


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


def _complete_sherpa_paths():
    return {
        key: f"/m/{key}"
        for key in (
            "asr_tokens", "asr_encoder", "asr_decoder", "asr_joiner",
            "tts_model", "tts_tokens",
        )
    }


@pytest.mark.parametrize("missing_key", ("tts_voices", "tts_data_dir", "tts_lexicon"))
def test_check_sherpa_models_validates_selected_kokoro_support(missing_key):
    paths = {
        **_complete_sherpa_paths(),
        "tts_voices": "/m/voices.bin",
        "tts_data_dir": "/m/espeak-ng-data",
        "tts_lexicon": "/m/lexicon.txt",
    }
    present = set(paths.values()) - {paths[missing_key]}
    result = check_sherpa_models(
        {"sherpa": paths}, exists=lambda path: path in present
    )
    assert not result.ok
    assert missing_key in result.detail


def test_check_sherpa_models_ignores_inactive_kokoro_lexicon():
    paths = {**_complete_sherpa_paths(), "tts_lexicon": "/stale/missing.txt"}
    present = set(_complete_sherpa_paths().values())
    assert check_sherpa_models(
        {"sherpa": paths}, exists=lambda path: path in present
    ).ok


def test_check_sherpa_models_validates_each_kokoro_lexicon_path():
    paths = {
        **_complete_sherpa_paths(),
        "tts_voices": "/m/voices.bin",
        "tts_lexicon": "/m/lexicon-en.txt, /m/lexicon-zh.txt",
    }
    present = {
        *set(_complete_sherpa_paths().values()),
        "/m/voices.bin",
        "/m/lexicon-en.txt",
        "/m/lexicon-zh.txt",
    }
    assert check_sherpa_models(
        {"sherpa": paths}, exists=lambda path: path in present
    ).ok

    present.remove("/m/lexicon-zh.txt")
    result = check_sherpa_models(
        {"sherpa": paths}, exists=lambda path: path in present
    )
    assert not result.ok
    assert "/m/lexicon-zh.txt" in result.detail


@pytest.mark.parametrize(
    ("backend", "missing_key"),
    (
        ("sense_voice", "asr_final_model"),
        ("sense_voice", "asr_final_tokens"),
        ("whisper", "asr_final_model"),
        ("whisper", "asr_final_tokens"),
        ("whisper", "asr_final_decoder"),
    ),
)
def test_check_sherpa_models_validates_selected_final_asr(backend, missing_key):
    paths = {
        **_complete_sherpa_paths(),
        "asr_final_backend": backend,
        "asr_final_model": "/m/final-model.onnx",
        "asr_final_tokens": "/m/final-tokens.txt",
        "asr_final_decoder": "/m/final-decoder.onnx",
    }
    present = {
        value for key, value in paths.items()
        if key != "asr_final_backend" and key != missing_key
    }
    result = check_sherpa_models(
        {"sherpa": paths}, exists=lambda path: path in present
    )
    assert not result.ok
    assert missing_key.removeprefix("asr_final_") in result.detail


def test_check_sherpa_models_accepts_complete_selected_final_asr():
    for backend in ("sense_voice", "whisper"):
        paths = {
            **_complete_sherpa_paths(),
            "asr_final_backend": backend,
            "asr_final_model": "/m/final-model.onnx",
            "asr_final_tokens": "/m/final-tokens.txt",
            "asr_final_decoder": "/m/final-decoder.onnx",
        }
        assert check_sherpa_models(
            {"sherpa": paths}, exists=lambda _path: True
        ).ok


@pytest.mark.parametrize(
    "missing_key",
    (
        "asr_final_hr_dict_dir",
        "asr_final_hr_lexicon",
        "asr_final_hr_rule_fsts",
        "asr_final_rule_fsts",
    ),
)
def test_check_sherpa_models_validates_selected_sensevoice_support(missing_key):
    paths = {
        **_complete_sherpa_paths(),
        "asr_final_backend": "sense_voice",
        "asr_final_model": "/m/final-model.onnx",
        "asr_final_tokens": "/m/final-tokens.txt",
        "asr_final_hr_dict_dir": "/m/hr-dict",
        "asr_final_hr_lexicon": "/m/hr-lexicon.txt",
        "asr_final_hr_rule_fsts": "/m/hr-one.fst",
        "asr_final_rule_fsts": "/m/rule-one.fst",
    }
    present = {
        value for key, value in paths.items()
        if key != "asr_final_backend" and key != missing_key
    }
    result = check_sherpa_models(
        {"sherpa": paths}, exists=lambda path: path in present
    )
    assert not result.ok
    assert missing_key in result.detail


@pytest.mark.parametrize(
    "key", ("asr_final_hr_rule_fsts", "asr_final_rule_fsts")
)
def test_check_sherpa_models_validates_each_final_rule_fst(key):
    paths = {
        **_complete_sherpa_paths(),
        "asr_final_backend": "sense_voice",
        "asr_final_model": "/m/final-model.onnx",
        "asr_final_tokens": "/m/final-tokens.txt",
        key: "/m/one.fst, /m/two.fst",
    }
    present = {
        *set(_complete_sherpa_paths().values()),
        "/m/final-model.onnx",
        "/m/final-tokens.txt",
        "/m/one.fst",
        "/m/two.fst",
    }
    assert check_sherpa_models(
        {"sherpa": paths}, exists=lambda path: path in present
    ).ok

    present.remove("/m/two.fst")
    result = check_sherpa_models(
        {"sherpa": paths}, exists=lambda path: path in present
    )
    assert not result.ok
    assert "/m/two.fst" in result.detail


def test_check_sherpa_models_ignores_sensevoice_support_for_whisper():
    paths = {
        **_complete_sherpa_paths(),
        "asr_final_backend": "whisper",
        "asr_final_model": "/m/final-model.onnx",
        "asr_final_tokens": "/m/final-tokens.txt",
        "asr_final_decoder": "/m/final-decoder.onnx",
        "asr_final_hr_dict_dir": "/stale/hr-dict",
        "asr_final_rule_fsts": "/stale/rules.fst",
    }
    present = {
        *set(_complete_sherpa_paths().values()),
        "/m/final-model.onnx",
        "/m/final-tokens.txt",
        "/m/final-decoder.onnx",
    }
    assert check_sherpa_models(
        {"sherpa": paths}, exists=lambda path: path in present
    ).ok


def test_check_sherpa_models_ignores_inactive_final_asr_artifacts():
    paths = {
        **_complete_sherpa_paths(),
        "asr_final_backend": "",
        "asr_final_model": "/stale/model.onnx",
        "asr_final_tokens": "/stale/tokens.txt",
        "asr_final_decoder": "/stale/decoder.onnx",
    }
    present = set(_complete_sherpa_paths().values())
    assert check_sherpa_models(
        {"sherpa": paths}, exists=lambda path: path in present
    ).ok


def test_check_sherpa_models_rejects_unknown_final_asr_backend():
    paths = {**_complete_sherpa_paths(), "asr_final_backend": "mystery"}
    result = check_sherpa_models({"sherpa": paths}, exists=lambda _path: True)
    assert not result.ok
    assert "unsupported" in result.detail


def test_check_sherpa_models_validates_configured_vad():
    paths = {**_complete_sherpa_paths(), "vad_model": "/m/vad.onnx"}
    present = set(_complete_sherpa_paths().values())
    result = check_sherpa_models(
        {"sherpa": paths}, exists=lambda path: path in present
    )
    assert not result.ok
    assert "vad_model missing" in result.detail


def test_check_sherpa_models_validates_configured_punctuation_model():
    paths = {**_complete_sherpa_paths(), "punct_model": "/m/punct.onnx"}
    present = set(_complete_sherpa_paths().values())
    result = check_sherpa_models(
        {"sherpa": paths}, exists=lambda path: path in present
    )
    assert not result.ok
    assert "punct_model missing" in result.detail


@pytest.mark.parametrize(
    "missing_key",
    ("kws_tokens", "kws_encoder", "kws_decoder", "kws_joiner", "kws_keywords_file"),
)
def test_check_sherpa_models_validates_selected_kws_group(missing_key):
    paths = {
        **_complete_sherpa_paths(),
        "kws_tokens": "/m/kws-tokens.txt",
        "kws_encoder": "/m/kws-encoder.onnx",
        "kws_decoder": "/m/kws-decoder.onnx",
        "kws_joiner": "/m/kws-joiner.onnx",
        "kws_keywords_file": "/m/keywords.txt",
    }
    # A configured-but-missing encoder selects KWS; it is not the same as an
    # empty encoder, which deliberately leaves the whole optional group inert.
    present = set(paths.values()) - {paths[missing_key]}
    result = check_sherpa_models(
        {"sherpa": paths}, exists=lambda path: path in present
    )
    assert not result.ok
    assert missing_key in result.detail


def test_check_sherpa_models_ignores_inactive_kws_fields():
    paths = {
        **_complete_sherpa_paths(),
        "kws_encoder": "",
        "kws_tokens": "/stale/tokens.txt",
        "kws_decoder": "/stale/decoder.onnx",
        "kws_joiner": "/stale/joiner.onnx",
        "kws_keywords_file": "/stale/keywords.txt",
    }
    present = set(_complete_sherpa_paths().values())
    assert check_sherpa_models(
        {"sherpa": paths}, exists=lambda path: path in present
    ).ok


def test_check_sherpa_models_requires_denoiser_only_when_enabled():
    base = _complete_sherpa_paths()
    present = set(base.values())
    disabled = {**base, "denoise_enabled": False, "denoise_model": "/stale.onnx"}
    assert check_sherpa_models(
        {"sherpa": disabled}, exists=lambda path: path in present
    ).ok

    enabled = {**base, "denoise_enabled": True, "denoise_model": "/missing.onnx"}
    result = check_sherpa_models(
        {"sherpa": enabled}, exists=lambda path: path in present
    )
    assert not result.ok
    assert "denoise_model missing" in result.detail

    unset = {**base, "denoise_enabled": True}
    result = check_sherpa_models(
        {"sherpa": unset}, exists=lambda path: path in present
    )
    assert not result.ok
    assert "denoise_model unset" in result.detail


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
    assert "enrollment file present" in c.detail
    assert "after the microphone opens" in c.detail


def test_check_speaker_id_is_required_by_active_word_cut():
    selected = {
        "barge_in_enabled": True,
        "barge_word_cut_enabled": True,
        "barge_word_cut_require_speaker": True,
        "aec_enabled": False,
    }
    missing_model = check_speaker_id({"sherpa": selected})
    assert not missing_model.ok
    assert "requires" in missing_model.detail

    missing_enrollment = check_speaker_id(
        {"sherpa": {**selected, "speaker_embedding_model": "/m/spk.onnx"}},
        exists=lambda path: path == "/m/spk.onnx",
    )
    assert not missing_enrollment.ok
    assert "enrollment" in missing_enrollment.detail

    enrolled = check_speaker_id(
        {
            "sherpa": {
                **selected,
                "speaker_embedding_model": "/m/spk.onnx",
                "speaker_enroll_embedding": "/m/enroll.json",
            }
        },
        exists=lambda _path: True,
    )
    assert enrolled.ok


def test_check_speaker_id_remains_advisory_when_word_cut_uses_in_app_aec():
    c = check_speaker_id(
        {
            "sherpa": {
                "barge_word_cut_enabled": True,
                "barge_word_cut_require_speaker": True,
                "aec_enabled": True,
            }
        }
    )
    assert c.ok
    assert "optional" in c.detail


def test_check_ollama_models_present_and_missing():
    checks = check_ollama(("gemma3:4b", "gemma3:12b"), lister=lambda: ["gemma3:4b"])
    by_name = {c.name: c for c in checks}
    assert by_name["ollama"].ok
    assert by_name["ollama model gemma3:4b"].ok
    assert not by_name["ollama model gemma3:12b"].ok
    assert by_name["ollama model gemma3:12b"].hint == "ollama pull gemma3:12b"


def test_check_ollama_minicpm_uses_template_aware_setup_hint():
    checks = check_ollama(("minicpm5-1b:q8",), lister=lambda: [])
    model = next(c for c in checks if c.name == "ollama model minicpm5-1b:q8")
    assert not model.ok
    assert model.hint == "python -m tools.setup_minicpm"


def _minicpm_show(*, digest=None, template=None, parameters=None, quantization="Q8_0"):
    digest = digest or MINICPM_Q8_CONTRACT.blob_sha256
    alias = {
        "modelfile": f"FROM /models/sha256-{digest}",
        "template": template or MINICPM_Q8_CONTRACT.template,
        "parameters": parameters or (
            'stop "<|im_end|>"\nstop "</s>"\n'
            "temperature 0.7\ntop_p 0.95\nnum_ctx 8192"
        ),
        "details": {"quantization_level": quantization},
    }
    source = {"modelfile": f"FROM /models/sha256-{MINICPM_Q8_CONTRACT.blob_sha256}"}
    return lambda model: alias if model == MINICPM_Q8_CONTRACT.alias else source


def test_check_ollama_minicpm_present_requires_canonical_identity():
    checks = check_ollama(
        (MINICPM_Q8_CONTRACT.alias,),
        lister=lambda: [MINICPM_Q8_CONTRACT.alias],
        show=_minicpm_show(),
    )
    model = next(c for c in checks if c.name.endswith(MINICPM_Q8_CONTRACT.alias))
    assert model.ok
    assert "pinned Q8_0" in model.detail
    assert model.hint == ""


@pytest.mark.parametrize(
    "show",
    (
        _minicpm_show(digest="a" * 64),
        _minicpm_show(template="{{ .Prompt }}"),
        _minicpm_show(quantization="Q4_K_M"),
        _minicpm_show(
            parameters=(
                'stop "<|im_end|>"\nstop "</s>"\nstop "extra"\n'
                "temperature 0.7\ntop_p 0.95\nnum_ctx 8192"
            )
        ),
    ),
)
def test_check_ollama_minicpm_identity_mismatch_fails_with_setup_hint(show):
    checks = check_ollama(
        (MINICPM_Q8_CONTRACT.alias,),
        lister=lambda: [MINICPM_Q8_CONTRACT.alias],
        show=show,
    )
    model = next(c for c in checks if c.name.endswith(MINICPM_Q8_CONTRACT.alias))
    assert not model.ok
    assert "identity mismatch" in model.detail
    assert model.hint == "python -m tools.setup_minicpm"


def test_check_ollama_minicpm_show_error_fails_closed():
    def boom(_model):
        raise RuntimeError("synthetic show failure")

    checks = check_ollama(
        (MINICPM_Q8_CONTRACT.alias,),
        lister=lambda: [MINICPM_Q8_CONTRACT.alias],
        show=boom,
    )
    model = next(c for c in checks if c.name.endswith(MINICPM_Q8_CONTRACT.alias))
    assert not model.ok
    assert "synthetic show failure" in model.detail


def test_check_ollama_rejects_noncanonical_minicpm_alias():
    checks = check_ollama(
        ("minicpm5-1b:latest",),
        lister=lambda: ["minicpm5-1b:latest"],
        show=lambda _model: pytest.fail("show was called"),
    )
    model = next(c for c in checks if c.name.endswith("minicpm5-1b:latest"))
    assert not model.ok
    assert MINICPM_Q8_CONTRACT.alias in model.detail
    assert model.hint == "python -m tools.setup_minicpm"


def test_generic_ollama_model_presence_does_not_call_identity_show():
    checks = check_ollama(
        ("gemma3:12b",),
        lister=lambda: ["gemma3:12b"],
        show=lambda _model: pytest.fail("show was called"),
    )
    assert all(check.ok for check in checks)


def test_check_ollama_uses_one_host_bound_client_for_list_and_show():
    calls = []

    class Client:
        def list(self):
            calls.append("list")
            return {"models": [{"model": MINICPM_Q8_CONTRACT.alias}]}

        def show(self, model):
            calls.append(("show", model))
            return _minicpm_show()(model)

    def factory(**kwargs):
        calls.append(("factory", kwargs))
        return Client()

    checks = check_ollama(
        (MINICPM_Q8_CONTRACT.alias,),
        host="http://ollama.test:11434",
        client_factory=factory,
    )

    assert all(check.ok for check in checks)
    assert calls[0] == (
        "factory",
        {"timeout": 5.0, "host": "http://ollama.test:11434"},
    )
    assert calls[1:] == [
        "list",
        ("show", MINICPM_Q8_CONTRACT.alias),
        ("show", MINICPM_Q8_CONTRACT.source),
    ]


def test_runtime_checks_forwards_the_resolved_ollama_host(monkeypatch):
    captured = {}

    def check(models, *, lister, show, host):
        captured.update(models=tuple(models), lister=lister, show=show, host=host)
        return []

    monkeypatch.setattr(readiness, "check_ollama", check)
    paths = {
        key: f"/m/{key}"
        for key in (
            "asr_tokens",
            "asr_encoder",
            "asr_decoder",
            "asr_joiner",
            "tts_model",
            "tts_tokens",
        )
    }
    run_runtime_checks(
        {
            "sherpa": paths,
            "llm": {
                "backend": "ollama",
                "host": "http://resolved-ollama.test:11434",
                "main_model": "gemma3:12b",
            },
        },
        resolved=True,
        import_fn=lambda _name: object(),
        exists=lambda _path: True,
        platform="win32",
        include_speaker=False,
        require_audio_devices=False,
    )

    assert captured == {
        "models": ("gemma3:12b",),
        "lister": None,
        "show": None,
        "host": "http://resolved-ollama.test:11434",
    }


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


def test_audio_frontend_accepts_scipy_antialias_fallback_without_soxr():
    def imports(name):
        if name == "soxr":
            raise ImportError("not installed")
        return object()

    checks = check_audio_frontend(
        {"sherpa": {}}, import_fn=imports, platform="test"
    )
    resampler = next(c for c in checks if "resampler" in c.name)
    assert resampler.ok
    assert "SciPy" in resampler.detail


def test_audio_frontend_rejects_only_when_no_antialias_backend_exists():
    checks = check_audio_frontend(
        {"sherpa": {}},
        import_fn=lambda _name: (_ for _ in ()).throw(ImportError("missing")),
        platform="test",
    )
    resampler = next(c for c in checks if "resampler" in c.name)
    assert not resampler.ok
    assert "neither soxr nor SciPy" in resampler.detail


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


def _aec_backend_check(checks, backend=None):
    for check in checks:
        if "aec backend" not in check.name.lower():
            continue
        if backend is None or backend in check.name.lower():
            return check
    return None


def test_audio_frontend_active_numpy_aec_backend_is_ready():
    for backend in ("nlms", "fdaf", "numpy"):
        check = _aec_backend_check(check_audio_frontend(
            {"sherpa": {"aec_enabled": True, "aec_backend": backend}},
            import_fn=lambda _name: object(),
        ), backend)
        assert check is not None and check.ok


def test_audio_frontend_unknown_active_aec_backend_fails():
    check = _aec_backend_check(check_audio_frontend(
        {"sherpa": {"aec_enabled": True, "aec_backend": "mystery"}},
        import_fn=lambda _name: object(),
    ))
    assert check is not None and not check.ok
    assert "unknown" in check.detail


def test_audio_frontend_unknown_inactive_aec_backend_is_ignored():
    checks = check_audio_frontend(
        {"sherpa": {"aec_enabled": False, "aec_backend": "mystery"}},
        import_fn=lambda _name: object(),
    )
    assert _aec_backend_check(checks) is None


def test_audio_frontend_dtln_requires_both_stages():
    model_dir = "/m/dtln"
    stage1 = f"{model_dir}/dtln_aec_stage1.onnx"
    check = _aec_backend_check(check_audio_frontend(
        {"sherpa": {
            "aec_enabled": True, "aec_backend": "dtln", "aec_model": model_dir,
        }},
        import_fn=lambda _name: object(),
        exists=lambda path: path == stage1,
    ), "dtln")
    assert check is not None and not check.ok
    assert "stage 2 missing" in check.detail
    assert "setup_models --aec-model" in check.hint


def test_audio_frontend_dtln_requires_onnxruntime():
    def imports(name):
        if name == "onnxruntime":
            raise ImportError("not installed")
        return object()

    check = _aec_backend_check(check_audio_frontend(
        {"sherpa": {
            "aec_enabled": True, "aec_backend": "dtln", "aec_model": "/m/dtln",
        }},
        import_fn=imports,
        exists=lambda _path: True,
    ), "dtln")
    assert check is not None and not check.ok
    assert "onnxruntime unavailable" in check.detail


@pytest.mark.parametrize(
    "model",
    ("/m/dtln", "/m/dtln_aec_stage1.onnx"),
)
def test_audio_frontend_dtln_ready_with_models_and_onnxruntime(model):
    check = _aec_backend_check(check_audio_frontend(
        {"sherpa": {
            "aec_enabled": True, "aec_backend": "dtln", "aec_model": model,
        }},
        import_fn=lambda _name: object(),
        exists=lambda _path: True,
    ), "dtln")
    assert check is not None and check.ok
    assert "onnxruntime available" in check.detail


def test_audio_frontend_dtln_rejects_ambiguous_direct_model_path():
    check = _aec_backend_check(check_audio_frontend(
        {"sherpa": {
            "aec_enabled": True, "aec_backend": "dtln", "aec_model": "/m/model.onnx",
        }},
        import_fn=lambda _name: object(),
        exists=lambda _path: True,
    ), "dtln")
    assert check is not None and not check.ok
    assert "stage-1" in check.detail


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


def _wasapi_route_check(checks):
    return next(
        (c for c in checks if "wasapi communications" in c.name.lower()), None
    )


def test_windows_word_cut_fails_without_voice_communications_request():
    checks = check_audio_frontend(
        {"sherpa": {
            "barge_word_cut_enabled": True,
            "aec_enabled": False,
            "vad_model": "/m/vad.onnx",
        }},
        import_fn=lambda _name: object(),
        exists=lambda _p: True,
        platform="win32",
    )
    route = _wasapi_route_check(checks)
    assert route is not None and not route.ok
    assert "capture_voice_comm=true" in route.detail


def test_windows_selected_voice_path_must_be_constructible():
    class FakeSD:
        class WasapiSettings:
            def __init__(self, **kwargs):
                if "communications" in kwargs:
                    raise TypeError("unexpected keyword argument 'communications'")

    checks = check_audio_frontend(
        {"sherpa": {"capture_voice_comm": True}},
        import_fn=lambda name: FakeSD if name == "sounddevice" else object(),
        platform="win32",
    )
    route = _wasapi_route_check(checks)
    assert route is not None and not route.ok
    assert "cannot request" in route.detail


def test_windows_selected_voice_path_passes_when_constructible():
    class FakeSD:
        class WasapiSettings:
            def __init__(self, *, communications):
                assert communications is True

    checks = check_audio_frontend(
        {"sherpa": {"capture_voice_comm": True}},
        import_fn=lambda name: FakeSD if name == "sounddevice" else object(),
        platform="win32",
    )
    route = _wasapi_route_check(checks)
    assert route is not None and route.ok


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

    def show_ollama(_model):
        raise AssertionError("EchoLLM readiness inspected Ollama")

    checks = run_runtime_checks(
        {"sherpa": paths, "llm": {"backend": "ollama"}},
        resolved=True,
        llm_mode="echo",
        sd=_FakeSD16k(),
        ollama_lister=list_ollama,
        ollama_show=show_ollama,
        import_fn=imports,
        exists=lambda _p: True,
        platform="win32",
    )
    assert all(check.ok for check in checks)
    assert not any("ollama" in check.name for check in checks)


def test_runtime_checks_selected_llamacpp_fails_closed_on_unverified_abort_abi():
    class WrongVersion(_VerifiedLlamaCpp):
        __version__ = "0.3.32"

    paths = _complete_sherpa_paths()

    def imports(name):
        return WrongVersion if name == "llama_cpp" else object()

    checks = run_runtime_checks(
        {
            "sherpa": paths,
            "llm": {
                "backend": "llamacpp",
                "main_model_path": "/m/minicpm.gguf",
                "n_gpu_layers": 0,
            },
        },
        resolved=True,
        import_fn=imports,
        exists=lambda _p: True,
        platform="win32",
        include_speaker=False,
        require_audio_devices=False,
    )

    check = next(c for c in checks if c.name == "llama.cpp CPU cancellation")
    assert not check.ok
    assert "found 0.3.32" in check.detail
    assert "llama-cpp-python==0.3.33" in check.hint
    assert not any(c.name == "ollama" for c in checks)


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
        ollama_lister=lambda: ["gemma3:12b", "minicpm5-1b:q8"],
        ollama_show=_minicpm_show(),
        import_fn=lambda name: object(),
        exists=lambda p: True,
    )
    ready, _ = summarize(checks)
    assert ready
