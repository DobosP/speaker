"""Tests for the install/preflight tooling (tools.setup_models + tools.doctor).

All pure / injected fakes -- no network, no audio, no Ollama, no models.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import core.readiness as readiness
import tools.setup_models as setup_models
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
from tools.setup_models import (
    FILE_KEYS,
    dest_for,
    extract_member,
    publish_config_atomic,
    required_selected_artifact_errors,
    wire_sherpa_paths,
)


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


def test_required_selected_artifacts_cover_shipped_profile(tmp_path):
    resolved = {}
    for key in (
        *FILE_KEYS,
        "speaker_embedding_model",
        "denoise_model",
        "asr_final_model",
        "asr_final_tokens",
        "tts_voices",
        "tts_lexicon",
    ):
        path = tmp_path / key
        path.write_bytes(b"model")
        resolved[key] = str(path)
    data_dir = tmp_path / "tts_data_dir"
    data_dir.mkdir()
    resolved["tts_data_dir"] = str(data_dir)

    assert required_selected_artifact_errors(
        resolved,
        speaker_model=True,
        denoise_model=True,
        sense_voice=True,
        kokoro=True,
    ) == []

    os.unlink(resolved["denoise_model"])
    errors = required_selected_artifact_errors(
        resolved,
        speaker_model=True,
        denoise_model=True,
        sense_voice=True,
        kokoro=True,
    )
    assert errors == ["selected GTCRN denoise model is missing on disk"]

    Path(resolved["denoise_model"]).write_bytes(b"model")
    os.unlink(resolved["speaker_embedding_model"])
    errors = required_selected_artifact_errors(
        resolved,
        speaker_model=True,
        denoise_model=True,
        sense_voice=True,
        kokoro=True,
    )
    assert errors == ["default speaker-ID model is missing on disk"]


def test_required_selected_artifacts_require_complete_parakeet_group(tmp_path):
    resolved = {}
    for key in (*FILE_KEYS, "asr_final_model", "asr_final_decoder", "asr_final_tokens"):
        path = tmp_path / key
        path.write_bytes(b"model")
        resolved[key] = str(path)
    resolved["asr_final_joiner"] = str(tmp_path / "missing-joiner.onnx")

    errors = required_selected_artifact_errors(
        resolved,
        speaker_model=False,
        denoise_model=False,
        sense_voice=False,
        kokoro=False,
        parakeet_final=True,
    )

    assert errors == ["selected Parakeet joiner is missing on disk"]


def test_atomic_config_publish_preserves_old_bytes_on_replace_failure(
    tmp_path, monkeypatch
):
    target = tmp_path / "config.local.json"
    original = b'{"existing": true}\n'
    target.write_bytes(original)

    def fail_replace(_source, _target):
        raise OSError("forced replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="forced replace failure"):
        publish_config_atomic({"new": True}, str(target))

    assert target.read_bytes() == original
    assert list(tmp_path.glob(".config.local.json.*.tmp")) == []


def _fake_selected_model_downloads(monkeypatch, *, fail_denoise: bool):
    import huggingface_hub
    import tools.bench.models as bench_models
    import tools.setup_models as setup_models

    manifest = {
        key: {"repo": "example/models", "file": f"{key}.onnx"}
        for key in FILE_KEYS
    }
    manifest["asr_encoder"]["file"] = "encoder.int8.onnx"
    manifest["asr_joiner"]["file"] = "joiner.int8.onnx"
    monkeypatch.setattr(bench_models, "load_manifest", lambda _path: manifest)

    def hf_download(*, filename, local_dir, **_kwargs):
        path = Path(local_dir) / Path(filename).name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"hf-model")
        return str(path)

    def snapshot_download(*, local_dir, repo_id=None, **_kwargs):
        root = Path(local_dir)
        if repo_id == setup_models.FASTER_WHISPER_VERIFIER_REPO:
            root.mkdir(parents=True, exist_ok=True)
            for name in setup_models.FASTER_WHISPER_REQUIRED_FILES:
                (root / name).write_bytes(b"faster-whisper")
        else:
            data = root / "espeak-ng-data"
            data.mkdir(parents=True, exist_ok=True)
        return str(local_dir)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_download)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)

    def direct_download(dest_dir, url, *, force=False):
        del force
        if fail_denoise and url == "fail-denoise":
            raise OSError("synthetic GTCRN failure")
        path = Path(dest_dir) / (Path(url).name or "asset.bin")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"direct-model")
        return str(path)

    def fake_extract(_archive, suffix, dest_dir):
        path = Path(dest_dir) / Path(suffix).name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"archive-model")
        return str(path)

    def fake_kokoro(dest_dir, _url, *, force=False):
        del force
        root = Path(dest_dir)
        root.mkdir(parents=True, exist_ok=True)
        files = {
            "tts_model": root / "model.int8.onnx",
            "tts_voices": root / "voices.bin",
            "tts_tokens": root / "tokens.txt",
            "tts_lexicon": root / "lexicon-us-en.txt",
        }
        for path in files.values():
            path.write_bytes(b"kokoro")
        data = root / "espeak-ng-data"
        data.mkdir()
        return {**{key: str(path) for key, path in files.items()}, "tts_data_dir": str(data)}

    monkeypatch.setattr(setup_models, "fetch_speaker_model", direct_download)
    monkeypatch.setattr(setup_models, "extract_member", fake_extract)
    monkeypatch.setattr(setup_models, "fetch_kokoro_package", fake_kokoro)


def _selected_setup_args(tmp_path):
    return [
        "--dest", str(tmp_path / "models"),
        "--config", str(tmp_path / "config.local.json"),
        "--speaker-model-url", "speaker.onnx",
        "--denoise-model", "--denoise-model-url", "fail-denoise",
        "--sense-voice", "--sense-voice-url", "sense-voice.tar.bz2",
        "--final-verifier", "faster-whisper-small",
        "--kokoro", "--require-selected",
    ]


def _write_faster_whisper_snapshot(path: Path, payload: bytes) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for name in setup_models.FASTER_WHISPER_REQUIRED_FILES:
        (path / name).write_bytes(payload)


def test_parakeet_setup_verifies_archive_and_publishes_complete_group(
    tmp_path,
    monkeypatch,
):
    import hashlib
    import io
    import tarfile

    archive = tmp_path / "candidate.tar.bz2"
    with tarfile.open(archive, "w:bz2") as tar:
        for index, name in enumerate(setup_models.PARAKEET_FINAL_FILES):
            payload = f"model-{index}".encode()
            member = tarfile.TarInfo(f"candidate/{name}")
            member.size = len(payload)
            tar.addfile(member, io.BytesIO(payload))
    expected = hashlib.sha256(archive.read_bytes()).hexdigest()

    monkeypatch.setattr(
        setup_models,
        "fetch_speaker_model",
        lambda _dest, _url, *, force=False: str(archive),
    )

    paths = setup_models.fetch_parakeet_final(
        str(tmp_path / "models"),
        url="https://example.invalid/model.tar.bz2",
        expected_sha256=expected,
    )

    assert set(paths) == {
        "asr_final_model",
        "asr_final_decoder",
        "asr_final_joiner",
        "asr_final_tokens",
    }
    assert all(Path(path).is_file() for path in paths.values())

    Path(paths["asr_final_model"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="does not match the verified archive"):
        setup_models.fetch_parakeet_final(
            str(tmp_path / "models"),
            url="https://example.invalid/model.tar.bz2",
            expected_sha256=expected,
        )
    restored = setup_models.fetch_parakeet_final(
        str(tmp_path / "models"),
        url="https://example.invalid/model.tar.bz2",
        expected_sha256=expected,
        force=True,
    )
    assert Path(restored["asr_final_model"]).read_bytes() == b"model-0"

    with pytest.raises(ValueError, match="checksum mismatch"):
        setup_models.fetch_parakeet_final(
            str(tmp_path / "other-models"),
            url="https://example.invalid/model.tar.bz2",
            expected_sha256="0" * 64,
        )


def test_faster_whisper_setup_pins_revision_and_stages_before_publish(tmp_path):
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        _write_faster_whisper_snapshot(Path(kwargs["local_dir"]), b"new")
        return kwargs["local_dir"]

    result = setup_models.fetch_faster_whisper_verifier(
        str(tmp_path),
        repo_id=setup_models.FASTER_WHISPER_VERIFIER_REPO,
        revision=setup_models.FASTER_WHISPER_VERIFIER_REVISION,
        token=None,
        force=False,
        snapshot_download_fn=download,
    )

    expected = tmp_path / (
        "faster_whisper_small-"
        + setup_models.FASTER_WHISPER_VERIFIER_REVISION[:12]
    )
    assert Path(result) == expected
    assert calls[0]["revision"] == setup_models.FASTER_WHISPER_VERIFIER_REVISION
    assert Path(calls[0]["local_dir"]) != expected
    assert (expected / "model.bin").read_bytes() == b"new"


def test_failed_forced_verifier_fetch_preserves_active_snapshot(tmp_path):
    destination = tmp_path / (
        "faster_whisper_small-"
        + setup_models.FASTER_WHISPER_VERIFIER_REVISION[:12]
    )
    _write_faster_whisper_snapshot(destination, b"old")

    def fail(**kwargs):
        staging = Path(kwargs["local_dir"])
        staging.mkdir(parents=True, exist_ok=True)
        (staging / "model.bin").write_bytes(b"partial")
        raise RuntimeError("download failed")

    with pytest.raises(RuntimeError, match="download failed"):
        setup_models.fetch_faster_whisper_verifier(
            str(tmp_path),
            repo_id=setup_models.FASTER_WHISPER_VERIFIER_REPO,
            revision=setup_models.FASTER_WHISPER_VERIFIER_REVISION,
            token=None,
            force=True,
            snapshot_download_fn=fail,
        )

    assert (destination / "model.bin").read_bytes() == b"old"


def test_verifier_setup_rejects_mutable_revision_before_download(tmp_path):
    with pytest.raises(ValueError, match="full commit hash"):
        setup_models.fetch_faster_whisper_verifier(
            str(tmp_path),
            repo_id=setup_models.FASTER_WHISPER_VERIFIER_REPO,
            revision="main",
            token=None,
            force=False,
            snapshot_download_fn=lambda **_kwargs: pytest.fail("downloaded"),
        )


def test_required_selected_failure_does_not_publish_partial_config(
    tmp_path, monkeypatch
):
    import tools.setup_models as setup_models

    _fake_selected_model_downloads(monkeypatch, fail_denoise=True)
    config = tmp_path / "config.local.json"
    original = b'{"unrelated": {"kept": true}}\n'
    config.write_bytes(original)

    assert setup_models.main(_selected_setup_args(tmp_path)) == 1
    assert config.read_bytes() == original


def test_require_selected_cannot_disable_default_speaker_model(tmp_path):
    import tools.setup_models as setup_models

    with pytest.raises(SystemExit) as raised:
        setup_models.main([
            "--dest", str(tmp_path / "models"),
            "--config", str(tmp_path / "config.local.json"),
            "--require-selected",
            "--no-speaker-model",
        ])

    assert raised.value.code == 2
    assert not (tmp_path / "config.local.json").exists()


def test_required_selected_success_publishes_one_complete_config(
    tmp_path, monkeypatch
):
    import tools.setup_models as setup_models

    _fake_selected_model_downloads(monkeypatch, fail_denoise=False)
    config = tmp_path / "config.local.json"
    config.write_text('{"unrelated": {"kept": true}}\n', encoding="utf-8")

    assert setup_models.main(_selected_setup_args(tmp_path)) == 0
    written = json.loads(config.read_text(encoding="utf-8"))
    sherpa = written["sherpa"]
    assert written["unrelated"] == {"kept": True}
    assert sherpa["asr_final_backend"] == "sense_voice"
    assert sherpa["asr_final_verifier_backend"] == "faster_whisper"
    assert all(
        sherpa[key]
        for key in (
            *FILE_KEYS,
            "speaker_embedding_model",
            "denoise_model",
            "asr_final_model",
            "asr_final_tokens",
            "asr_final_verifier_model",
            "tts_voices",
            "tts_data_dir",
            "tts_lexicon",
        )
    )
    assert list(tmp_path.glob(".config.local.json.*.tmp")) == []


def test_required_selected_parakeet_publishes_backend_and_four_paths(
    tmp_path,
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(setup_models.sys, "platform", "linux")
    _fake_selected_model_downloads(monkeypatch, fail_denoise=False)

    def fake_parakeet(destination, **_kwargs):
        root = Path(destination) / setup_models.PARAKEET_FINAL_DIR
        root.mkdir(parents=True, exist_ok=True)
        paths = {
            "asr_final_model": root / "encoder.int8.onnx",
            "asr_final_decoder": root / "decoder.int8.onnx",
            "asr_final_joiner": root / "joiner.int8.onnx",
            "asr_final_tokens": root / "tokens.txt",
        }
        for path in paths.values():
            path.write_bytes(b"parakeet")
        return {key: str(path) for key, path in paths.items()}

    monkeypatch.setattr(setup_models, "fetch_parakeet_final", fake_parakeet)
    args = _selected_setup_args(tmp_path)
    sense_index = args.index("--sense-voice")
    del args[sense_index : sense_index + 3]
    args.extend(("--final-asr", "parakeet-unified-en"))

    assert setup_models.main(args) == 0
    written = json.loads(
        (tmp_path / "config.local.json").read_text(encoding="utf-8")
    )
    sherpa = written["sherpa"]
    assert sherpa["asr_final_backend"] == "nemo_transducer"
    assert all(
        sherpa[key]
        for key in (
            "asr_final_model",
            "asr_final_decoder",
            "asr_final_joiner",
            "asr_final_tokens",
        )
    )
    output = capsys.readouterr().out
    assert "Now run:  ./live.sh" in output
    assert setup_models.normal_voice_entry("win32") == "python -m core --engine sherpa"


def test_setup_rejects_two_offline_final_asr_selections():
    with pytest.raises(SystemExit) as raised:
        setup_models.main(
            [
                "--sense-voice",
                "--final-asr",
                "parakeet-unified-en",
            ]
        )

    assert raised.value.code == 2


def test_required_selected_rejects_non_object_existing_config(
    tmp_path, monkeypatch
):
    import tools.setup_models as setup_models

    _fake_selected_model_downloads(monkeypatch, fail_denoise=False)
    config = tmp_path / "config.local.json"
    original = b"[]\n"
    config.write_bytes(original)

    assert setup_models.main(_selected_setup_args(tmp_path)) == 1
    assert config.read_bytes() == original


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
        ("nemo_transducer", "asr_final_model"),
        ("nemo_transducer", "asr_final_tokens"),
        ("nemo_transducer", "asr_final_decoder"),
        ("nemo_transducer", "asr_final_joiner"),
    ),
)
def test_check_sherpa_models_validates_selected_final_asr(backend, missing_key):
    paths = {
        **_complete_sherpa_paths(),
        "asr_final_backend": backend,
        "asr_final_model": "/m/final-model.onnx",
        "asr_final_tokens": "/m/final-tokens.txt",
        "asr_final_decoder": "/m/final-decoder.onnx",
        "asr_final_joiner": "/m/final-joiner.onnx",
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
    for backend in ("sense_voice", "whisper", "nemo_transducer"):
        paths = {
            **_complete_sherpa_paths(),
            "asr_final_backend": backend,
            "asr_final_model": "/m/final-model.onnx",
            "asr_final_tokens": "/m/final-tokens.txt",
            "asr_final_decoder": "/m/final-decoder.onnx",
            "asr_final_joiner": "/m/final-joiner.onnx",
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


def test_check_sherpa_models_validates_selected_faster_whisper_verifier():
    model = "/m/faster-whisper-small"
    paths = {
        **_complete_sherpa_paths(),
        "asr_final_verifier_backend": "faster_whisper",
        "asr_final_verifier_model": model,
    }
    present = {
        *set(_complete_sherpa_paths().values()),
        model,
        *(f"{model}/{name}" for name in readiness._FASTER_WHISPER_SNAPSHOT_FILES),
    }
    assert check_sherpa_models(
        {"sherpa": paths}, exists=lambda path: path in present
    ).ok

    present.remove(f"{model}/model.bin")
    result = check_sherpa_models(
        {"sherpa": paths}, exists=lambda path: path in present
    )
    assert not result.ok
    assert "verifier model.bin missing" in result.detail


def test_check_sherpa_models_ignores_inactive_verifier_artifact():
    paths = {
        **_complete_sherpa_paths(),
        "asr_final_verifier_backend": "",
        "asr_final_verifier_model": "/stale/missing",
    }
    present = set(_complete_sherpa_paths().values())

    assert check_sherpa_models(
        {"sherpa": paths}, exists=lambda path: path in present
    ).ok


def test_check_sherpa_models_rejects_unknown_verifier_backend():
    paths = {
        **_complete_sherpa_paths(),
        "asr_final_verifier_backend": "generative_rewriter",
    }
    result = check_sherpa_models({"sherpa": paths}, exists=lambda _path: True)

    assert not result.ok
    assert "asr_final_verifier_backend" in result.detail
    assert "unsupported" in result.detail


def test_selected_verifier_runtime_requires_cuda_float16():
    config = {
        "sherpa": {
            "asr_final_verifier_backend": "faster_whisper",
            "asr_final_verifier_model": "/model",
        }
    }

    class _CTranslate2:
        @staticmethod
        def get_supported_compute_types(device, index):
            assert (device, index) == ("cuda", 0)
            return {"float16", "int8_float16"}

    def available(name):
        return _CTranslate2 if name == "ctranslate2" else object()

    assert readiness.check_asr_final_verifier_runtime(
        config,
        import_fn=available,
        preload_fn=lambda: None,
        model_probe_fn=lambda path: path == "/model" or pytest.fail(path),
    ).ok

    class _CpuOnly:
        @staticmethod
        def get_supported_compute_types(_device, _index):
            return {"int8"}

    failed = readiness.check_asr_final_verifier_runtime(
        config,
        import_fn=lambda name: _CpuOnly if name == "ctranslate2" else object(),
        preload_fn=lambda: None,
        model_probe_fn=lambda _path: pytest.fail("CPU-only reached model probe"),
    )
    assert not failed.ok
    assert "FP16" in failed.detail


def test_selected_verifier_runtime_preloads_before_any_model_import():
    config = {
        "sherpa": {
            "asr_final_verifier_backend": "faster_whisper",
            "asr_final_verifier_model": "/model",
        }
    }
    events = []

    def fail_preload():
        events.append("preload")
        raise RuntimeError("safe bootstrap failure")

    result = readiness.check_asr_final_verifier_runtime(
        config,
        preload_fn=fail_preload,
        import_fn=lambda name: events.append(name),
        model_probe_fn=lambda _path: events.append("model"),
    )

    assert not result.ok
    assert events == ["preload"]


def test_selected_verifier_runtime_proves_exact_model_load():
    config = {
        "sherpa": {
            "asr_final_verifier_backend": "faster_whisper",
            "asr_final_verifier_model": "/model",
        }
    }

    class _CTranslate2:
        @staticmethod
        def get_supported_compute_types(_device, _index):
            return {"float16"}

    result = readiness.check_asr_final_verifier_runtime(
        config,
        preload_fn=lambda: None,
        import_fn=lambda name: _CTranslate2 if name == "ctranslate2" else object(),
        model_probe_fn=lambda _path: (_ for _ in ()).throw(
            RuntimeError("corrupt model")
        ),
    )

    assert not result.ok
    assert "corrupt model" in result.detail


def test_selected_nemo_runtime_requires_pinned_version_and_completed_decode():
    config = {"sherpa": {"asr_final_backend": "nemo_transducer"}}
    probed = []

    selected = readiness.check_asr_final_runtime(
        config,
        import_fn=lambda name: (
            type("Sherpa", (), {"__version__": "1.13.3"})
            if name == "sherpa_onnx"
            else pytest.fail(name)
        ),
        model_probe_fn=lambda sherpa: probed.append(dict(sherpa)),
    )
    assert selected.ok
    assert probed == [config["sherpa"]]

    wrong_version = readiness.check_asr_final_runtime(
        config,
        import_fn=lambda _name: type("Sherpa", (), {"__version__": "1.13.2"}),
        model_probe_fn=lambda _sherpa: pytest.fail("wrong version reached probe"),
    )
    assert not wrong_version.ok
    assert "1.13.3" in wrong_version.detail

    failed_decode = readiness.check_asr_final_runtime(
        config,
        import_fn=lambda _name: type("Sherpa", (), {"__version__": "1.13.3"}),
        model_probe_fn=lambda _sherpa: (_ for _ in ()).throw(
            RuntimeError("decode unavailable")
        ),
    )
    assert not failed_decode.ok
    assert "decode unavailable" in failed_decode.detail


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


def test_check_speaker_id_is_advisory_for_default_identity_free_word_cut():
    selected = {
        "barge_in_enabled": True,
        "barge_word_cut_enabled": True,
        "aec_enabled": False,
    }

    check = check_speaker_id({"sherpa": selected})

    assert check.ok
    assert "optional owner filtering off" in check.detail
    assert "lexical barge-in remains available" in check.detail


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


class _VirtualTopology:
    def __init__(self, ok=True, detail="exact run-owned topology"):
        self.ok = ok
        self.detail = detail

    def verify_topology(self):
        return self.ok, self.detail


def _virtual_route_check(checks):
    return next(
        (c for c in checks if "virtual delay ec topology" in c.name.lower()),
        None,
    )


def test_virtual_delay_topology_replaces_unrelated_host_route_check():
    checks = check_audio_frontend(
        {"sherpa": {
            "barge_word_cut_enabled": True,
            "aec_enabled": False,
            "vad_model": "/m/vad.onnx",
        }},
        import_fn=_no_livekit,
        exists=lambda _p: True,
        platform="linux",
        pipewire_state=PipeWireState(),
        virtual_audio_binder=_VirtualTopology(),
    )

    virtual = _virtual_route_check(checks)
    assert virtual is not None and virtual.ok
    assert _word_cut_route_check(checks) is None


def test_bad_virtual_delay_topology_cannot_fallback_to_valid_host_route():
    checks = check_audio_frontend(
        {"sherpa": {
            "barge_word_cut_enabled": True,
            "aec_enabled": False,
            "vad_model": "/m/vad.onnx",
        }},
        import_fn=_no_livekit,
        exists=lambda _p: True,
        platform="linux",
        pipewire_state=_EC_ROUTED,
        virtual_audio_binder=_VirtualTopology(False, "loopback master drifted"),
    )

    virtual = _virtual_route_check(checks)
    assert virtual is not None and not virtual.ok
    assert "drifted" in virtual.detail
    assert _word_cut_route_check(checks) is None


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


def test_doctor_defer_ollama_reports_only_base_ready(monkeypatch, capsys):
    import core.app as app
    import tools.doctor as doctor

    config = {
        "device": "desktop",
        "device_profiles": {},
        "llm": {"backend": "ollama"},
    }
    seen = {}
    monkeypatch.setattr(app, "_load_config", lambda *_args, **_kwargs: config)

    def fake_run_all(_config, **kwargs):
        seen.update(kwargs)
        return [doctor.Check("base speech runtime", True, "complete")]

    monkeypatch.setattr(doctor, "run_all", fake_run_all)

    assert doctor.main(["--defer-ollama"]) == 0
    assert seen["llm_mode"] == "echo"
    output = capsys.readouterr().out
    assert "BASE READY (Ollama deferred)" in output
    assert "READY -> python -m core" not in output


def test_doctor_defer_ollama_propagates_base_failure(monkeypatch, capsys):
    import core.app as app
    import tools.doctor as doctor

    monkeypatch.setattr(app, "_load_config", lambda *_args, **_kwargs: {
        "device": "desktop",
        "device_profiles": {},
        "llm": {"backend": "ollama"},
    })
    monkeypatch.setattr(
        doctor,
        "run_all",
        lambda *_args, **_kwargs: [doctor.Check("speech models", False, "missing")],
    )

    assert doctor.main(["--defer-ollama"]) == 1
    output = capsys.readouterr().out
    assert "BASE NOT READY" in output
    assert "BASE READY (Ollama deferred)" not in output


def test_doctor_refuses_to_defer_a_non_ollama_backend(monkeypatch, capsys):
    import core.app as app
    import tools.doctor as doctor

    monkeypatch.setattr(app, "_load_config", lambda *_args, **_kwargs: {
        "device": "phone",
        "device_profiles": {},
        "llm": {"backend": "llamacpp"},
    })
    monkeypatch.setattr(
        doctor,
        "run_all",
        lambda *_args, **_kwargs: pytest.fail("non-Ollama deferral ran checks"),
    )

    assert doctor.main(["--defer-ollama"]) == 1
    output = capsys.readouterr().out
    assert "selected LLM backend is 'llamacpp'" in output
    assert "BASE NOT READY" in output


def test_doctor_defer_llm_accepts_llamacpp_and_runs_only_base_checks(
    monkeypatch, capsys
):
    import core.app as app
    import tools.doctor as doctor

    config = {
        "device": "phone",
        "device_profiles": {},
        "llm": {"backend": "llamacpp"},
    }
    seen = {}
    monkeypatch.setattr(app, "_load_config", lambda *_args, **_kwargs: config)

    def fake_run_all(_config, **kwargs):
        seen.update(kwargs)
        return [doctor.Check("base speech runtime", True, "complete")]

    monkeypatch.setattr(doctor, "run_all", fake_run_all)

    assert doctor.main(["--defer-llm"]) == 0
    assert seen["llm_mode"] == "echo"
    output = capsys.readouterr().out
    assert "BASE READY (local LLM deferred)" in output
    assert "READY -> python -m core" not in output


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
