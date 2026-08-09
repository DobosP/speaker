from __future__ import annotations

import builtins
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace

import pytest

from tools import provision_kyutai_stt_candidate as provision
from tools.streaming_stt import manifest as manifest_module
from tools.streaming_stt.manifest import (
    KYUTAI_ADAPTER,
    KYUTAI_ARTIFACT_NAMES,
)
from tools.streaming_stt.runtime_receipt import (
    KYUTAI_RUNTIME_TREE_LIMITS,
    generate_runtime_tree_receipt,
    load_runtime_tree_receipt,
)
from tools.streaming_stt.wheel_lock import (
    KYUTAI_WHEEL_ARCHIVE_POLICY,
    InstalledContentExpectation,
    WheelLockError,
)


_RUNTIME_CONSTANTS = {
    "KYUTAI_RUNTIME_CONTENT_SHA256": "content_digest",
    "KYUTAI_RUNTIME_FILE_COUNT": "file_count",
    "KYUTAI_RUNTIME_TOTAL_SIZE_BYTES": "total_size_bytes",
    "KYUTAI_RUNTIME_MAXIMUM_FILE_BYTES": "maximum_file_bytes",
}
_MODEL_PAYLOADS = {
    "model-config": b'{"fixture":"config"}\n',
    "model-weights": b"fixture-model-weights",
    "model-mimi": b"fixture-mimi-weights",
    "model-tokenizer": b"fixture-tokenizer",
}


def _create_runtime(venv: Path) -> Path:
    venv.mkdir(mode=0o700)
    python = venv / "bin" / "python"
    python.parent.mkdir(mode=0o700)
    python.symlink_to(Path(sys.executable).resolve(strict=True))
    marker = venv / "pyvenv.cfg"
    marker.write_text(
        "home = /usr/bin\n"
        "include-system-site-packages = false\n"
        "version = 3.12.3\n"
        "executable = /usr/bin/python3.12\n",
        encoding="utf-8",
    )
    marker.chmod(0o600)
    site_packages = venv / "lib" / "python3.12" / "site-packages"
    site_packages.mkdir(parents=True, mode=0o700)
    site_packages.chmod(0o700)
    package = site_packages / "fixture_runtime"
    package.mkdir()
    (package / "__init__.py").write_bytes(b"FIXTURE = True\n")
    return site_packages


def _inputs(tmp_path: Path, monkeypatch):
    venv = tmp_path / "candidate-venv"
    site_packages = _create_runtime(venv)
    fixture_receipt = generate_runtime_tree_receipt(
        site_packages,
        tmp_path / "fixture-runtime-receipt.json",
        limits=KYUTAI_RUNTIME_TREE_LIMITS,
    )
    runtime_values = {
        "content_digest": fixture_receipt.content_digest,
        "file_count": fixture_receipt.file_count,
        "total_size_bytes": fixture_receipt.total_size_bytes,
        "maximum_file_bytes": max(item.size_bytes for item in fixture_receipt.files),
    }
    for module in (provision, manifest_module):
        for constant, attribute in _RUNTIME_CONSTANTS.items():
            monkeypatch.setattr(module, constant, runtime_values[attribute])

    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir(mode=0o700)
    policies = []

    def verify_fixture_install(
        _lock,
        selected_wheelhouse,
        receipt,
        *,
        archive_policy,
    ):
        assert selected_wheelhouse == wheelhouse
        policies.append(archive_policy)
        return InstalledContentExpectation(files=receipt.files)

    monkeypatch.setattr(
        provision,
        "verify_locked_wheel_install",
        verify_fixture_install,
    )

    model = tmp_path / "model"
    model.mkdir(mode=0o700)
    model_receipts = {}
    for artifact_name, basename in provision._MODEL_FILES.items():
        path = model / basename
        path.write_bytes(_MODEL_PAYLOADS[artifact_name])
        path.chmod(0o600)
        model_receipts[artifact_name] = (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat().st_size,
        )
    monkeypatch.setattr(provision, "_MODEL_RECEIPTS", model_receipts)
    monkeypatch.setattr(manifest_module, "KYUTAI_MODEL_RECEIPTS", model_receipts)
    return venv, site_packages, model, wheelhouse, policies


def _provision(tmp_path: Path, monkeypatch):
    venv, site_packages, model, wheelhouse, policies = _inputs(
        tmp_path,
        monkeypatch,
    )
    manifest = provision.provision_candidate(
        venv_root=venv,
        model_root=model,
        wheelhouse=wheelhouse,
        output_dir=tmp_path / "receipt",
    )
    return manifest, venv, site_packages, model, wheelhouse, policies


def test_production_tuple_is_the_exact_candle_095e_closure():
    assert provision._MODEL_FILES == {
        "model-config": "config.json",
        "model-weights": "model.safetensors",
        "model-mimi": "mimi-pytorch-e351c8d8@125.safetensors",
        "model-tokenizer": "tokenizer_en_fr_audio_8000.model",
    }
    assert provision._MODEL_RECEIPTS == manifest_module.KYUTAI_MODEL_RECEIPTS
    assert provision._MODEL_RECEIPTS == {
        "model-config": (
            "a3f1c6f7a39fca1fb1bbff68eaabc560b8037d2cdc68aa1f489859949a4223de",
            1_315,
        ),
        "model-weights": (
            "b9e97c53229dce728d65c76bfa892f7b563c69d671899f0ebc6518582dddec6f",
            1_978_522_200,
        ),
        "model-mimi": (
            "09b782f0629851a271227fb9d36db65c041790365f11bbe5d3d59369cf863f50",
            384_644_900,
        ),
        "model-tokenizer": (
            "cd87dd5d17169151782ac700280ec057e5d658a9afbe238a048ea5ff318cce69",
            120_378,
        ),
    }
    lock_bytes = provision._REPOSITORY_WHEEL_LOCK.read_bytes()
    assert len(lock_bytes) == 25_818
    assert hashlib.sha256(lock_bytes).hexdigest() == (
        "8d0e41563bb5e91500af42a912c5e6f825f16a67537ddc350556303e88609e25"
    )
    assert manifest_module.KYUTAI_RUNTIME_CONTENT_SHA256 == (
        "9edc9c42b8c718d0e4b17d917c04c05acc0b5ecca4ec9504866ad34f1f62dbf0"
    )
    assert manifest_module.KYUTAI_RUNTIME_FILE_COUNT == 16_547
    assert manifest_module.KYUTAI_RUNTIME_TOTAL_SIZE_BYTES == 5_694_993_765
    assert manifest_module.KYUTAI_RUNTIME_MAXIMUM_FILE_BYTES == 984_633_129


def test_provision_binds_schema_v9_without_candidate_imports(
    tmp_path,
    monkeypatch,
):
    imported = []
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in {
            "huggingface_hub",
            "julius",
            "moshi",
            "safetensors",
            "sentencepiece",
            "torch",
        }:
            imported.append(name)
            raise AssertionError("candidate import attempted")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    manifest, venv, site_packages, model, _wheelhouse, policies = _provision(
        tmp_path,
        monkeypatch,
    )

    assert imported == []
    assert manifest.schema_version == 9
    assert manifest.adapter == KYUTAI_ADAPTER
    assert manifest.model_id == "kyutai-stt-1b-en-fr-candle-095e-cuda-bf16"
    assert manifest.python.path == venv / "bin" / "python"
    assert manifest.python.path.resolve(strict=True) == Path("/usr/bin/python3.12")
    assert tuple(manifest.artifact_by_name) == KYUTAI_ARTIFACT_NAMES
    assert {
        manifest.artifact_by_name[name].path.parent
        for name in KYUTAI_ARTIFACT_NAMES
        if name.startswith("model-")
    } == {model}
    for name in KYUTAI_ARTIFACT_NAMES:
        if not name.startswith("model-"):
            continue
        metadata = manifest.artifact_by_name[name].path.lstat()
        assert stat.S_IMODE(metadata.st_mode) == 0o600
        assert metadata.st_nlink == 1
        assert metadata.st_uid == os.geteuid()
    assert manifest.worker.path == provision._FIXED_WORKER
    assert manifest.limits.startup_timeout_sec == 300.0
    assert manifest.limits.case_timeout_sec == 900.0
    assert manifest.adapter_config is not None
    assert manifest.adapter_config.python_version == "3.12.3"
    assert manifest.adapter_config.moshi_version == "0.2.11"
    assert manifest.adapter_config.torch_version == "2.7.1+cu126"
    assert manifest.adapter_config.julius_version == "0.2.8"
    assert manifest.adapter_config.model_revision == (
        "095e38f6242006a93c2541149b181988397f5c7c"
    )
    assert manifest.adapter_config.resampling_mode == "whole-buffer-noncausal"
    assert manifest.adapter_config.semantic_head_policy == "diagnostic-finite-only"
    assert manifest.adapter_config.endpoint_owner == "none"
    assert manifest.adapter_config.early_stop is False
    assert manifest.adapter_config.torch_compile is False
    assert manifest.adapter_config.cuda_graph is False
    assert policies == [KYUTAI_WHEEL_ARCHIVE_POLICY]
    assert stat.S_IMODE(manifest.path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(manifest.path.stat().st_mode) == 0o600
    assert {path.name for path in manifest.path.parent.iterdir()} == (
        provision._OUTPUT_FILENAMES
    )
    assert all(
        stat.S_IMODE(manifest.artifact_by_name[name].path.stat().st_mode) == 0o600
        for name in ("runtime-receipt", "runtime-wheel-lock", "venv-marker")
    )
    receipt = load_runtime_tree_receipt(
        manifest.artifact_by_name["runtime-receipt"].path,
        limits=KYUTAI_RUNTIME_TREE_LIMITS,
    )
    assert receipt.root == site_packages
    receipt_artifact = manifest.artifact_by_name["runtime-receipt"]
    assert receipt_artifact.sha256 == receipt.digest
    assert receipt_artifact.size_bytes == receipt.receipt_path.stat().st_size
    assert (
        0
        < receipt_artifact.size_bytes
        <= KYUTAI_RUNTIME_TREE_LIMITS.maximum_receipt_bytes
    )
    safe = provision._safe_result(manifest)
    assert set(safe) == {
        "ok",
        "adapter",
        "model_id",
        "manifest_sha256",
        "runtime_receipt_sha256",
        "worker_sha256",
        "artifact_set_sha256",
    }
    assert str(tmp_path) not in json.dumps(safe)


def test_provision_is_no_overwrite_and_preserves_first_manifest(
    tmp_path,
    monkeypatch,
):
    manifest, venv, _site_packages, model, wheelhouse, _policies = _provision(
        tmp_path,
        monkeypatch,
    )
    before = manifest.path.read_bytes()

    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            venv_root=venv,
            model_root=model,
            wheelhouse=wheelhouse,
            output_dir=manifest.path.parent,
        )

    assert manifest.path.read_bytes() == before


@pytest.mark.parametrize("mutation", ("extra", "tamper", "symlink", "hardlink"))
def test_provision_rejects_changed_or_linked_model_before_output(
    tmp_path,
    monkeypatch,
    mutation,
):
    venv, _site_packages, model, wheelhouse, _policies = _inputs(
        tmp_path,
        monkeypatch,
    )
    selected = model / "config.json"
    if mutation == "extra":
        (model / "README.md").write_text("not staged", encoding="utf-8")
    elif mutation == "tamper":
        selected.write_bytes(b"tampered")
    else:
        payload = selected.read_bytes()
        selected.unlink()
        outside = tmp_path / f"{mutation}-target"
        outside.write_bytes(payload)
        if mutation == "symlink":
            selected.symlink_to(outside)
        else:
            os.link(outside, selected)
    output = tmp_path / "rejected"

    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            venv_root=venv,
            model_root=model,
            wheelhouse=wheelhouse,
            output_dir=output,
        )

    assert not output.exists()


@pytest.mark.parametrize("mode", (0o400, 0o640, 0o644, 0o660))
def test_provision_requires_mode_0600_model_artifacts(
    tmp_path,
    monkeypatch,
    mode,
):
    venv, _site_packages, model, wheelhouse, _policies = _inputs(
        tmp_path,
        monkeypatch,
    )
    (model / "config.json").chmod(mode)
    output = tmp_path / "wrong-model-mode"

    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            venv_root=venv,
            model_root=model,
            wheelhouse=wheelhouse,
            output_dir=output,
        )

    assert not output.exists()


def test_provision_requires_current_uid_model_artifacts(tmp_path, monkeypatch):
    venv, _site_packages, model, wheelhouse, _policies = _inputs(
        tmp_path,
        monkeypatch,
    )
    selected = model / "config.json"
    original_lstat = Path.lstat

    def lstat_with_wrong_owner(path):
        metadata = original_lstat(path)
        if path == selected:
            return SimpleNamespace(
                st_mode=metadata.st_mode,
                st_nlink=metadata.st_nlink,
                st_uid=metadata.st_uid + 1,
            )
        return metadata

    monkeypatch.setattr(Path, "lstat", lstat_with_wrong_owner)
    output = tmp_path / "wrong-model-owner"

    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            venv_root=venv,
            model_root=model,
            wheelhouse=wheelhouse,
            output_dir=output,
        )

    assert not output.exists()


@pytest.mark.parametrize(
    "marker_text",
    (
        (
            "home = /usr/bin\n"
            "include-system-site-packages = true\n"
            "version = 3.12.3\n"
            "executable = /usr/bin/python3.12\n"
        ),
        (
            "home = /usr/bin\n"
            "include-system-site-packages = false\n"
            "version = 3.12.4\n"
            "executable = /usr/bin/python3.12\n"
        ),
        ("home = /usr/bin\ninclude-system-site-packages = false\nversion = 3.12.3\n"),
        (
            "home = /usr/bin\n"
            "include-system-site-packages = false\n"
            "version = 3.12.3\n"
            "version = 3.12.3\n"
            "executable = /usr/bin/python3.12\n"
        ),
        (
            "home = /usr/bin\n"
            "include-system-site-packages = false\n"
            "version = 3.12.3\n"
            "executable = /tmp/python3.12\n"
        ),
    ),
)
def test_provision_requires_exact_private_python3123_marker(
    tmp_path,
    monkeypatch,
    marker_text,
):
    venv, _site_packages, model, wheelhouse, _policies = _inputs(
        tmp_path,
        monkeypatch,
    )
    marker = venv / "pyvenv.cfg"
    marker.write_text(marker_text, encoding="utf-8")
    marker.chmod(0o600)
    output = tmp_path / "bad-marker"

    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            venv_root=venv,
            model_root=model,
            wheelhouse=wheelhouse,
            output_dir=output,
        )

    assert not output.exists()


def test_provision_requires_private_marker_mode(tmp_path, monkeypatch):
    venv, _site_packages, model, wheelhouse, _policies = _inputs(
        tmp_path,
        monkeypatch,
    )
    (venv / "pyvenv.cfg").chmod(0o644)
    output = tmp_path / "public-marker"

    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            venv_root=venv,
            model_root=model,
            wheelhouse=wheelhouse,
            output_dir=output,
        )

    assert not output.exists()


@pytest.mark.parametrize("location", ("venv", "model", "wheelhouse", "ancestor"))
def test_output_overlap_fails_before_manifest_publication(
    tmp_path,
    monkeypatch,
    location,
):
    venv, _site_packages, model, wheelhouse, _policies = _inputs(
        tmp_path,
        monkeypatch,
    )
    output = {
        "venv": venv / "receipt",
        "model": model / "receipt",
        "wheelhouse": wheelhouse / "receipt",
        "ancestor": tmp_path,
    }[location]

    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            venv_root=venv,
            model_root=model,
            wheelhouse=wheelhouse,
            output_dir=output,
        )

    assert not (output / "worker-manifest.json").exists()


def test_relocated_identical_runtimes_bind_dynamic_receipt_identity(
    tmp_path,
    monkeypatch,
):
    first_venv, first_site_packages, model, wheelhouse, policies = _inputs(
        tmp_path,
        monkeypatch,
    )
    second_venv = tmp_path / "relocated-byte-identical-runtime"
    second_site_packages = _create_runtime(second_venv)

    first_manifest = provision.provision_candidate(
        venv_root=first_venv,
        model_root=model,
        wheelhouse=wheelhouse,
        output_dir=tmp_path / "first-receipt",
    )
    second_manifest = provision.provision_candidate(
        venv_root=second_venv,
        model_root=model,
        wheelhouse=wheelhouse,
        output_dir=tmp_path / "second-receipt",
    )
    first_receipt = load_runtime_tree_receipt(
        first_manifest.artifact_by_name["runtime-receipt"].path,
        limits=KYUTAI_RUNTIME_TREE_LIMITS,
    )
    second_receipt = load_runtime_tree_receipt(
        second_manifest.artifact_by_name["runtime-receipt"].path,
        limits=KYUTAI_RUNTIME_TREE_LIMITS,
    )

    assert first_receipt.root == first_site_packages
    assert second_receipt.root == second_site_packages
    assert first_receipt.files == second_receipt.files
    assert first_receipt.content_digest == second_receipt.content_digest
    assert first_receipt.digest != second_receipt.digest
    assert first_manifest.artifact_by_name["runtime-receipt"].sha256 == (
        first_receipt.digest
    )
    assert second_manifest.artifact_by_name["runtime-receipt"].sha256 == (
        second_receipt.digest
    )
    for manifest in (first_manifest, second_manifest):
        artifact = manifest.artifact_by_name["runtime-receipt"]
        assert (
            0
            < artifact.size_bytes
            <= (KYUTAI_RUNTIME_TREE_LIMITS.maximum_receipt_bytes)
        )
    assert policies == [KYUTAI_WHEEL_ARCHIVE_POLICY, KYUTAI_WHEEL_ARCHIVE_POLICY]


def test_runtime_metrics_are_manifest_bound_before_output(
    tmp_path,
    monkeypatch,
):
    venv, _site_packages, model, wheelhouse, _policies = _inputs(
        tmp_path,
        monkeypatch,
    )
    monkeypatch.setattr(
        provision,
        "KYUTAI_RUNTIME_FILE_COUNT",
        provision.KYUTAI_RUNTIME_FILE_COUNT + 1,
    )
    output = tmp_path / "wrong-runtime-metrics"

    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            venv_root=venv,
            model_root=model,
            wheelhouse=wheelhouse,
            output_dir=output,
        )

    assert not output.exists()


def test_wheelhouse_failure_under_kyutai_policy_never_publishes_manifest(
    tmp_path,
    monkeypatch,
):
    venv, _site_packages, model, wheelhouse, _policies = _inputs(
        tmp_path,
        monkeypatch,
    )

    def reject(*_args, **_kwargs):
        raise WheelLockError()

    monkeypatch.setattr(provision, "verify_locked_wheel_install", reject)
    output = tmp_path / "bad-wheelhouse"

    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            venv_root=venv,
            model_root=model,
            wheelhouse=wheelhouse,
            output_dir=output,
        )

    assert output.exists()
    assert not (output / "worker-manifest.json").exists()


def test_tampered_repository_lock_fails_before_output(tmp_path, monkeypatch):
    venv, _site_packages, model, wheelhouse, _policies = _inputs(
        tmp_path,
        monkeypatch,
    )
    bad_lock = tmp_path / "bad-repository-lock.json"
    bad_lock.write_bytes(provision._REPOSITORY_WHEEL_LOCK.read_bytes() + b" ")
    monkeypatch.setattr(provision, "_REPOSITORY_WHEEL_LOCK", bad_lock)
    output = tmp_path / "bad-lock"

    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            venv_root=venv,
            model_root=model,
            wheelhouse=wheelhouse,
            output_dir=output,
        )

    assert not output.exists()


def test_cli_failure_is_detail_free(tmp_path, monkeypatch, capsys):
    private = tmp_path / "private-runtime"

    def reject(**_kwargs):
        raise provision.ProvisionError(str(private))

    monkeypatch.setattr(provision, "provision_candidate", reject)
    code = provision.main(
        [
            "--venv-root",
            str(private),
            "--model-root",
            str(tmp_path / "model"),
            "--wheelhouse",
            str(tmp_path / "wheelhouse"),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )

    assert code == 2
    result = json.loads(capsys.readouterr().out)
    assert result == {
        "error": "kyutai_candidate_prerequisites_unavailable",
        "ok": False,
    }
    assert str(tmp_path) not in json.dumps(result)


def test_cli_help_promises_verification_only(capsys):
    with pytest.raises(SystemExit) as raised:
        provision.main(["--help"])

    assert raised.value.code == 0
    help_text = capsys.readouterr().out.casefold()
    for word in ("does not install", "download", "import", "execute", "load"):
        assert word in help_text
