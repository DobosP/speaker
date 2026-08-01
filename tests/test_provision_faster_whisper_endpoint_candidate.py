from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import sys

import pytest

from tools import provision_faster_whisper_endpoint_candidate as provision
from tools import streaming_stt_eval
from tools.streaming_stt.manifest import (
    FASTER_WHISPER_ARTIFACT_NAMES,
    FASTER_WHISPER_ENDPOINT_ADAPTER,
    FasterWhisperEndpointConfig,
    ManifestError,
    load_worker_manifest,
)


def _write_tree_file(root: Path, relative: str, payload: bytes = b"fixture") -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _inputs(tmp_path: Path):
    venv = tmp_path / "candidate-venv"
    venv.mkdir(mode=0o700)
    python = venv / "bin" / "python"
    python.parent.mkdir(mode=0o700)
    python.symlink_to(Path(sys.executable).resolve(strict=True))
    (venv / "pyvenv.cfg").write_text(
        "home = /usr/bin\ninclude-system-site-packages = false\nversion = 3.12.3\n",
        encoding="utf-8",
    )
    site_packages = venv / "lib" / "python3.12" / "site-packages"
    site_packages.mkdir(parents=True)
    site_packages.chmod(0o700)
    for relative in sorted(provision._EXPECTED_RUNTIME_PATHS):
        _write_tree_file(site_packages, relative)
    _write_tree_file(
        site_packages,
        "ctranslate2/_ext.cpython-312-x86_64-linux-gnu.so",
    )

    model = tmp_path / "model"
    model.mkdir(mode=0o700)
    _write_tree_file(model, "config.json", b"{}")
    _write_tree_file(model, "model.bin", b"model-weights")
    _write_tree_file(model, "tokenizer.json", b"{}")
    _write_tree_file(model, "vocabulary.txt", b"token\n")
    return venv, site_packages, model


def test_provision_binds_runtime_and_model_trees_without_importing_candidate(
    tmp_path,
):
    venv, _site_packages, model = _inputs(tmp_path)
    output = tmp_path / "receipt"

    manifest = provision.provision_candidate(
        venv_root=venv,
        model_root=model,
        output_dir=output,
        model_id="faster-whisper-small-local",
        language="ro",
    )

    assert manifest.schema_version == 6
    assert manifest.adapter == FASTER_WHISPER_ENDPOINT_ADAPTER
    assert manifest.model_id == "faster-whisper-small-local"
    assert manifest.python.path == venv / "bin" / "python"
    assert tuple(manifest.artifact_by_name) == FASTER_WHISPER_ARTIFACT_NAMES
    assert isinstance(manifest.adapter_config, FasterWhisperEndpointConfig)
    assert manifest.adapter_config.language == "ro"
    assert manifest.adapter_config.execution_mode == "endpoint-final-only"
    assert manifest.adapter_config.partial_hypotheses is False
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert stat.S_IMODE(manifest.path.stat().st_mode) == 0o600
    assert all(
        stat.S_IMODE(manifest.artifact_by_name[name].path.stat().st_mode) == 0o600
        for name in ("runtime-receipt", "model-receipt")
    )
    assert "faster_whisper" not in sys.modules
    safe = provision._safe_result(manifest)
    assert set(safe) == {
        "ok",
        "adapter",
        "model_id",
        "manifest_sha256",
        "runtime_receipt_sha256",
        "model_receipt_sha256",
        "artifact_set_sha256",
    }
    assert str(tmp_path) not in str(safe)

    runtime_digest = streaming_stt_eval._verified_runtime_receipt_digest(manifest)
    model_digest = streaming_stt_eval._verified_model_receipt_digest(manifest)
    assert runtime_digest == manifest.artifact_by_name["runtime-receipt"].sha256
    assert model_digest == manifest.artifact_by_name["model-receipt"].sha256


def test_provision_is_no_overwrite_and_preserves_first_receipt(tmp_path):
    venv, _site_packages, model = _inputs(tmp_path)
    output = tmp_path / "receipt"
    manifest = provision.provision_candidate(
        venv_root=venv,
        model_root=model,
        output_dir=output,
    )
    before = manifest.path.read_bytes()

    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            venv_root=venv,
            model_root=model,
            output_dir=output,
        )

    assert manifest.path.read_bytes() == before


@pytest.mark.parametrize("location", ["venv", "site-packages", "model", "ancestor"])
def test_output_root_overlap_fails_before_mkdir(
    tmp_path,
    monkeypatch,
    location,
):
    venv, site_packages, model = _inputs(tmp_path)
    output = {
        "venv": venv / "receipt",
        "site-packages": site_packages / "receipt",
        "model": model / "receipt",
        "ancestor": tmp_path,
    }[location]
    mkdir_calls: list[tuple[object, ...]] = []

    def forbidden_mkdir(*args, **_kwargs):
        mkdir_calls.append(args)
        raise OSError

    monkeypatch.setattr(provision.os, "mkdir", forbidden_mkdir)

    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            venv_root=venv,
            model_root=model,
            output_dir=output,
        )

    assert mkdir_calls == []
    if location != "ancestor":
        assert not output.exists()


def test_nested_runtime_and_model_roots_fail_before_output_mutation(
    tmp_path,
    monkeypatch,
):
    venv, site_packages, _model = _inputs(tmp_path)
    nested_model = site_packages / "nested-model"
    nested_model.mkdir(mode=0o700)
    _write_tree_file(nested_model, "config.json", b"{}")
    _write_tree_file(nested_model, "model.bin", b"model-weights")
    _write_tree_file(nested_model, "tokenizer.json", b"{}")
    _write_tree_file(nested_model, "vocabulary.txt", b"token\n")
    output = tmp_path / "receipt"
    mkdir_calls: list[tuple[object, ...]] = []

    def forbidden_mkdir(*args, **_kwargs):
        mkdir_calls.append(args)
        raise OSError

    monkeypatch.setattr(provision.os, "mkdir", forbidden_mkdir)

    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            venv_root=venv,
            model_root=nested_model,
            output_dir=output,
        )

    assert mkdir_calls == []
    assert not output.exists()


@pytest.mark.parametrize("invalid_layout", ["python3.11", "marker-3.11.9"])
def test_wrong_python_layout_fails_before_output_mutation(
    tmp_path,
    monkeypatch,
    invalid_layout,
):
    venv, site_packages, model = _inputs(tmp_path)
    if invalid_layout == "python3.11":
        site_packages.parent.rename(site_packages.parent.parent / "python3.11")
    else:
        (venv / "pyvenv.cfg").write_text(
            "home = /usr/bin\ninclude-system-site-packages = false\nversion = 3.11.9\n",
            encoding="utf-8",
        )
    output = tmp_path / "receipt"
    mkdir_calls: list[tuple[object, ...]] = []

    def forbidden_mkdir(*args, **_kwargs):
        mkdir_calls.append(args)
        raise OSError

    monkeypatch.setattr(provision.os, "mkdir", forbidden_mkdir)

    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            venv_root=venv,
            model_root=model,
            output_dir=output,
        )

    assert mkdir_calls == []
    assert not output.exists()


def test_private_writer_rechecks_descriptor_identity_after_parent_fsync(
    tmp_path,
    monkeypatch,
):
    parent = tmp_path / "private-output"
    parent.mkdir(mode=0o700)
    target = parent / "worker-manifest.json"
    replacement = parent / "replacement"
    replacement.write_bytes(b"replacement")
    replacement.chmod(0o600)
    real_fsync = os.fsync
    fsync_calls = 0

    def replacing_fsync(descriptor):
        nonlocal fsync_calls
        real_fsync(descriptor)
        fsync_calls += 1
        if fsync_calls == 2:
            target.unlink()
            replacement.replace(target)

    monkeypatch.setattr(provision.os, "fsync", replacing_fsync)

    with pytest.raises(provision.ProvisionError):
        provision._write_new_private(target, b"bound-manifest")

    assert fsync_calls == 2
    assert target.read_bytes() == b"replacement"


def test_model_tree_change_fails_close_time_receipt_recheck(tmp_path):
    venv, _site_packages, model = _inputs(tmp_path)
    manifest = provision.provision_candidate(
        venv_root=venv,
        model_root=model,
        output_dir=tmp_path / "receipt",
    )
    (model / "model.bin").write_bytes(b"changed-weights")

    with pytest.raises(ValueError):
        streaming_stt_eval._verified_model_receipt_digest(manifest)


@pytest.mark.parametrize(
    "remove",
    ["model.bin", "tokenizer.json", "vocabulary.txt"],
)
def test_provision_rejects_incomplete_model_closure(tmp_path, remove):
    venv, _site_packages, model = _inputs(tmp_path)
    (model / remove).unlink()

    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            venv_root=venv,
            model_root=model,
            output_dir=tmp_path / "receipt",
        )


def test_provision_rejects_hugging_face_symlink_model_tree(tmp_path):
    venv, _site_packages, model = _inputs(tmp_path)
    outside = tmp_path / "blob-model.bin"
    outside.write_bytes(b"model-weights")
    (model / "model.bin").unlink()
    (model / "model.bin").symlink_to(outside)

    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            venv_root=venv,
            model_root=model,
            output_dir=tmp_path / "receipt",
        )


def test_provision_rejects_hard_linked_model_tree(tmp_path):
    venv, _site_packages, model = _inputs(tmp_path)
    outside = tmp_path / "shared-model.bin"
    outside.write_bytes(b"model-weights")
    (model / "model.bin").unlink()
    os.link(outside, model / "model.bin")

    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            venv_root=venv,
            model_root=model,
            output_dir=tmp_path / "receipt",
        )


def test_provision_rejects_non_private_model_root(tmp_path):
    venv, _site_packages, model = _inputs(tmp_path)
    model.chmod(0o755)

    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            venv_root=venv,
            model_root=model,
            output_dir=tmp_path / "receipt",
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "config",
        "artifact-order",
        "artifact-layout",
        "schema",
        "adapter",
    ],
)
def test_schema_v6_manifest_rejects_strict_negative_mutations(tmp_path, mutation):
    venv, _site_packages, model = _inputs(tmp_path)
    manifest = provision.provision_candidate(
        venv_root=venv,
        model_root=model,
        output_dir=tmp_path / "receipt",
    )
    payload = json.loads(manifest.path.read_text(encoding="utf-8"))
    if mutation == "config":
        payload["adapter_config"]["partial_hypotheses"] = True
    elif mutation == "artifact-order":
        payload["artifacts"][0], payload["artifacts"][1] = (
            payload["artifacts"][1],
            payload["artifacts"][0],
        )
    elif mutation == "artifact-layout":
        payload["artifacts"][0]["path"] = str(tmp_path / "wrong-receipt-name")
    elif mutation == "schema":
        payload["schema_version"] = 5
    else:
        payload["adapter"] = "fake-json-v1"
    manifest.path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ManifestError):
        load_worker_manifest(manifest.path)


def test_cli_failure_is_path_and_detail_free(tmp_path, capsys):
    result = provision.main(
        [
            "--venv-root",
            str(tmp_path / "private-missing-runtime"),
            "--model-root",
            str(tmp_path / "private-missing-model"),
            "--output-dir",
            str(tmp_path / "private-output"),
        ]
    )
    stdout = capsys.readouterr().out

    assert result == 2
    assert json.loads(stdout) == provision._SAFE_ERROR
    assert str(tmp_path) not in stdout
