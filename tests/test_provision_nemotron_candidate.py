from __future__ import annotations

import builtins
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

import pytest

from tools import provision_nemotron_candidate as provision
from tools.streaming_stt import manifest as manifest_module
from tools.streaming_stt.manifest import (
    NEMOTRON_ADAPTER,
    NEMOTRON_ARTIFACT_NAMES,
)
from tools.streaming_stt.runtime_receipt import (
    NEMOTRON_RUNTIME_TREE_LIMITS,
    generate_runtime_tree_receipt,
    load_runtime_tree_receipt,
)
from tools.streaming_stt.wheel_lock import InstalledContentExpectation


def _inputs(tmp_path: Path, monkeypatch):
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
    (site_packages / "candidate-runtime.py").write_bytes(b"candidate-runtime")
    fixture_receipt = generate_runtime_tree_receipt(
        site_packages,
        tmp_path / "fixture-runtime-receipt.json",
        limits=NEMOTRON_RUNTIME_TREE_LIMITS,
    )
    runtime_metrics = {
        "NEMOTRON_RUNTIME_CONTENT_SHA256": fixture_receipt.content_digest,
        "NEMOTRON_RUNTIME_FILE_COUNT": len(fixture_receipt.files),
        "NEMOTRON_RUNTIME_TOTAL_SIZE_BYTES": sum(
            item.size_bytes for item in fixture_receipt.files
        ),
        "NEMOTRON_RUNTIME_MAXIMUM_FILE_BYTES": max(
            item.size_bytes for item in fixture_receipt.files
        ),
    }
    for module in (provision, manifest_module):
        for name, value in runtime_metrics.items():
            monkeypatch.setattr(module, name, value)

    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir(mode=0o700)

    def verify_fixture_install(_lock, selected_wheelhouse, receipt):
        assert selected_wheelhouse == wheelhouse
        return InstalledContentExpectation(files=receipt.files)

    monkeypatch.setattr(
        provision,
        "verify_locked_wheel_install",
        verify_fixture_install,
    )

    model = tmp_path / "model"
    model.mkdir(mode=0o700)
    model_paths = {
        name: model / basename for name, basename in provision._MODEL_FILES.items()
    }
    for index, path in enumerate(model_paths.values(), start=1):
        path.write_bytes(bytes([index]))
    receipts = {
        name: (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat().st_size,
        )
        for name, path in model_paths.items()
    }
    monkeypatch.setattr(provision, "_MODEL_RECEIPTS", receipts)
    monkeypatch.setattr(manifest_module, "_NEMOTRON_MODEL_RECEIPTS", receipts)
    return venv, site_packages, model, wheelhouse


def _provision(
    tmp_path: Path,
    monkeypatch,
    **overrides,
):
    venv, site_packages, model, wheelhouse = _inputs(tmp_path, monkeypatch)
    arguments = {
        "venv_root": venv,
        "model_root": model,
        "wheelhouse": wheelhouse,
        "output_dir": tmp_path / "receipt",
    }
    arguments.update(overrides)
    manifest = provision.provision_candidate(**arguments)
    return manifest, venv, site_packages, model, wheelhouse


def test_provision_model_receipts_match_manifest_contract():
    assert provision._MODEL_RECEIPTS == manifest_module._NEMOTRON_MODEL_RECEIPTS


def test_provision_accepts_uv_python312_marker(tmp_path, monkeypatch):
    venv, _site_packages, model, wheelhouse = _inputs(tmp_path, monkeypatch)
    (venv / "pyvenv.cfg").write_text(
        "home = /usr/bin\n"
        "implementation = CPython\n"
        "uv = 0.11.23\n"
        "version_info = 3.12.3\n"
        "include-system-site-packages = false\n",
        encoding="utf-8",
    )

    manifest = provision.provision_candidate(
        venv_root=venv,
        model_root=model,
        wheelhouse=wheelhouse,
        output_dir=tmp_path / "uv-receipt",
    )

    assert manifest.schema_version == 3


def test_provision_binds_existing_runtime_without_candidate_imports(
    tmp_path,
    monkeypatch,
):
    imported: list[str] = []
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in {
            "torch",
            "transformers",
            "huggingface_hub",
            "safetensors",
        }:
            imported.append(name)
            raise AssertionError("candidate import attempted")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    manifest, venv, _site_packages, _model, _wheelhouse = _provision(
        tmp_path,
        monkeypatch,
    )

    assert imported == []
    assert manifest.schema_version == 3
    assert manifest.adapter == NEMOTRON_ADAPTER
    assert manifest.python.path == venv / "bin" / "python"
    assert tuple(manifest.artifact_by_name) == NEMOTRON_ARTIFACT_NAMES
    assert manifest.adapter_config is not None
    expected_config = manifest_module.NemotronConfig(
        runtime_content_sha256=manifest_module.NEMOTRON_RUNTIME_CONTENT_SHA256,
        runtime_file_count=manifest_module.NEMOTRON_RUNTIME_FILE_COUNT,
        runtime_total_size_bytes=manifest_module.NEMOTRON_RUNTIME_TOTAL_SIZE_BYTES,
        runtime_maximum_file_bytes=(
            manifest_module.NEMOTRON_RUNTIME_MAXIMUM_FILE_BYTES
        ),
    )
    assert manifest.adapter_config.as_dict() == expected_config.as_dict()
    assert stat.S_IMODE(manifest.path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(manifest.path.stat().st_mode) == 0o600
    receipt_path = manifest.artifact_by_name["runtime-receipt"].path
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o600
    receipt = load_runtime_tree_receipt(
        receipt_path,
        limits=NEMOTRON_RUNTIME_TREE_LIMITS,
    )
    assert receipt.root == venv / "lib" / "python3.12" / "site-packages"
    safe = provision._safe_result(manifest)
    assert set(safe) == {
        "ok",
        "adapter",
        "model_id",
        "manifest_sha256",
        "runtime_receipt_sha256",
        "artifact_set_sha256",
    }
    assert str(tmp_path) not in json.dumps(safe)


def test_provision_accepts_pinned_language_lookahead_and_runtime_lock(
    tmp_path,
    monkeypatch,
):
    venv, _site_packages, model, wheelhouse = _inputs(tmp_path, monkeypatch)

    manifest = provision.provision_candidate(
        venv_root=venv,
        model_root=model,
        wheelhouse=wheelhouse,
        output_dir=tmp_path / "locked",
        lookahead_tokens=13,
        language="ro-RO",
    )

    assert manifest.model_id.endswith("-ro-ro-la13-cuda")
    assert manifest.adapter_config is not None
    assert manifest.adapter_config.lookahead_tokens == 13
    assert manifest.adapter_config.language == "ro-RO"


def test_provision_is_no_overwrite_and_preserves_first_manifest(
    tmp_path,
    monkeypatch,
):
    manifest, venv, _site_packages, model, wheelhouse = _provision(
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


@pytest.mark.parametrize("extra_name", ["weights.nemo", "README.md", "nested"])
def test_provision_rejects_any_extra_model_entry_before_output(
    tmp_path,
    monkeypatch,
    extra_name,
):
    venv, _site_packages, model, wheelhouse = _inputs(tmp_path, monkeypatch)
    extra = model / extra_name
    if extra_name == "nested":
        extra.mkdir()
    else:
        extra.write_bytes(b"not-in-the-official-closure")
    output = tmp_path / "rejected"

    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            venv_root=venv,
            model_root=model,
            wheelhouse=wheelhouse,
            output_dir=output,
        )

    assert not output.exists()


def test_provision_rejects_tampered_model_before_output(tmp_path, monkeypatch):
    venv, _site_packages, model, wheelhouse = _inputs(tmp_path, monkeypatch)
    (model / "config.json").write_bytes(b"tampered")
    output = tmp_path / "tampered"

    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            venv_root=venv,
            model_root=model,
            wheelhouse=wheelhouse,
            output_dir=output,
        )

    assert not output.exists()


@pytest.mark.parametrize("mutation", ["symlink", "hardlink"])
def test_provision_rejects_linked_model_file_before_output(
    tmp_path,
    monkeypatch,
    mutation,
):
    venv, _site_packages, model, wheelhouse = _inputs(tmp_path, monkeypatch)
    selected = model / "config.json"
    payload = selected.read_bytes()
    selected.unlink()
    outside = tmp_path / f"{mutation}-target"
    outside.write_bytes(payload)
    if mutation == "symlink":
        selected.symlink_to(outside)
    else:
        os.link(outside, selected)
    output = tmp_path / mutation

    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            venv_root=venv,
            model_root=model,
            wheelhouse=wheelhouse,
            output_dir=output,
        )

    assert not output.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lookahead_tokens", 1),
        ("lookahead_tokens", True),
        ("language", "ro"),
        ("language", "auto"),
    ],
)
def test_provision_rejects_wrong_closed_config_before_output(
    tmp_path,
    monkeypatch,
    field,
    value,
):
    venv, _site_packages, model, wheelhouse = _inputs(tmp_path, monkeypatch)
    output = tmp_path / "wrong-config"
    arguments = {
        "venv_root": venv,
        "model_root": model,
        "wheelhouse": wheelhouse,
        "output_dir": output,
        field: value,
    }

    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(**arguments)

    assert not output.exists()


@pytest.mark.parametrize(
    "marker_text",
    [
        "include-system-site-packages = true\nversion = 3.12.3\n",
        "include-system-site-packages = false\nversion = 3.11.9\n",
        "include-system-site-packages = false\n",
        ("include-system-site-packages = false\nversion = 3.12.3\nversion = 3.12.4\n"),
        (
            "include-system-site-packages = false\n"
            "version = 3.12.3\n"
            "version_info = 3.12.3\n"
        ),
    ],
)
def test_provision_requires_exact_isolated_python312_marker(
    tmp_path,
    monkeypatch,
    marker_text,
):
    venv, _site_packages, model, wheelhouse = _inputs(tmp_path, monkeypatch)
    (venv / "pyvenv.cfg").write_text(marker_text, encoding="utf-8")
    output = tmp_path / "bad-marker"

    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            venv_root=venv,
            model_root=model,
            wheelhouse=wheelhouse,
            output_dir=output,
        )

    assert not output.exists()


def test_provision_rejects_non_python312_site_packages_layout(
    tmp_path,
    monkeypatch,
):
    venv, site_packages, model, wheelhouse = _inputs(tmp_path, monkeypatch)
    python_root = site_packages.parent
    python_root.rename(python_root.with_name("python3.11"))
    output = tmp_path / "wrong-layout"

    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            venv_root=venv,
            model_root=model,
            wheelhouse=wheelhouse,
            output_dir=output,
        )

    assert not output.exists()


@pytest.mark.parametrize("selected_path", ["venv", "model", "output"])
def test_provision_refuses_production_venv_and_paths_inside_it(
    tmp_path,
    monkeypatch,
    selected_path,
):
    venv, _site_packages, model, wheelhouse = _inputs(tmp_path, monkeypatch)
    production = tmp_path / "production-venv"
    production.mkdir(mode=0o700)
    monkeypatch.setattr(provision, "_PRODUCTION_VENV", production)
    output = tmp_path / "outside-output"
    if selected_path == "venv":
        production = venv
        monkeypatch.setattr(provision, "_PRODUCTION_VENV", production)
    elif selected_path == "model":
        nested_model = production / "models"
        model.rename(nested_model)
        model = nested_model
    else:
        output = production / "receipts"

    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            venv_root=venv,
            model_root=model,
            wheelhouse=wheelhouse,
            output_dir=output,
        )

    assert not output.exists()


def test_provision_refuses_a_candidate_root_containing_production_venv(
    tmp_path,
    monkeypatch,
):
    parent = tmp_path / "candidate-parent"
    parent.mkdir(mode=0o700)
    production = parent / ".venv"
    production.mkdir(mode=0o700)
    monkeypatch.setattr(provision, "_PRODUCTION_VENV", production)

    with pytest.raises(provision.ProvisionError):
        provision._require_outside_production(parent)


def test_provision_rejects_wrong_runtime_closure_without_manifest_publish(
    tmp_path,
    monkeypatch,
):
    venv, _site_packages, model, wheelhouse = _inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(
        provision,
        "NEMOTRON_RUNTIME_CONTENT_SHA256",
        "0" * 64,
    )
    monkeypatch.setattr(
        manifest_module,
        "NEMOTRON_RUNTIME_CONTENT_SHA256",
        "0" * 64,
    )
    output = tmp_path / "wrong-runtime-lock"

    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            venv_root=venv,
            model_root=model,
            wheelhouse=wheelhouse,
            output_dir=output,
        )

    assert output.is_dir()
    assert not (output / "worker-manifest.json").exists()


def test_cli_error_is_detail_free_and_does_not_print_paths(
    tmp_path,
    capsys,
):
    secret = tmp_path / "private-name-never-print"

    code = provision.main(
        [
            "--venv-root",
            str(secret),
            "--model-root",
            str(secret / "model"),
            "--wheelhouse",
            str(secret / "wheelhouse"),
            "--output-dir",
            str(secret / "output"),
        ]
    )

    captured = capsys.readouterr()
    assert code == 2
    assert captured.err == ""
    assert str(secret) not in captured.out
    assert json.loads(captured.out) == provision._SAFE_ERROR


def test_provision_help_states_no_install_download_or_candidate_import():
    help_text = " ".join(provision._parser().format_help().split())

    assert "does not install or download" in help_text
    assert "does not import the candidate runtime" in help_text
