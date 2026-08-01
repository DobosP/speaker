from __future__ import annotations

import builtins
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

import pytest

from tests.test_streaming_stt_wheel_lock import _entry, _wheel, _write_lock
from tools import provision_parakeet_realtime_eou_candidate as provision
from tools.streaming_stt import manifest as manifest_module
from tools.streaming_stt.manifest import (
    PARAKEET_REALTIME_EOU_ADAPTER,
    PARAKEET_REALTIME_EOU_ARTIFACT_NAMES,
)
from tools.streaming_stt.runtime_receipt import (
    PARAKEET_RUNTIME_TREE_LIMITS,
    generate_runtime_tree_receipt,
    load_runtime_tree_receipt,
)
from tools.streaming_stt.runtime_builder import (
    RuntimeBuildPolicy,
    prepare_locked_runtime,
)
from tools.streaming_stt.wheel_lock import (
    PARAKEET_WHEEL_ARCHIVE_POLICY,
    InstalledContentExpectation,
    load_wheel_lock,
)


_RUNTIME_CONSTANTS = {
    "PARAKEET_REALTIME_EOU_RUNTIME_CONTENT_SHA256": "content_digest",
    "PARAKEET_REALTIME_EOU_RUNTIME_FILE_COUNT": "file_count",
    "PARAKEET_REALTIME_EOU_RUNTIME_TOTAL_SIZE_BYTES": "total_size_bytes",
    "PARAKEET_REALTIME_EOU_RUNTIME_MAXIMUM_FILE_BYTES": "maximum_file_bytes",
}


def _inputs(tmp_path: Path, monkeypatch):
    venv = tmp_path / "candidate-venv"
    venv.mkdir(mode=0o700)
    python = venv / "bin" / "python"
    python.parent.mkdir(mode=0o700)
    python.symlink_to(Path(sys.executable).resolve(strict=True))
    (venv / "pyvenv.cfg").write_text(
        "home = /usr/bin\n"
        "include-system-site-packages = false\n"
        "version = 3.12.3\n"
        "executable = /usr/bin/python3.12\n",
        encoding="utf-8",
    )
    (venv / "pyvenv.cfg").chmod(0o600)
    site_packages = venv / "lib" / "python3.12" / "site-packages"
    site_packages.mkdir(parents=True)
    site_packages.chmod(0o700)
    (site_packages / "candidate-runtime.py").write_bytes(b"candidate-runtime")
    fixture_receipt = generate_runtime_tree_receipt(
        site_packages,
        tmp_path / "fixture-runtime-receipt.json",
        limits=PARAKEET_RUNTIME_TREE_LIMITS,
    )
    runtime_values = {
        "content_digest": fixture_receipt.content_digest,
        "file_count": fixture_receipt.file_count,
        "total_size_bytes": fixture_receipt.total_size_bytes,
        "maximum_file_bytes": max(
            item.size_bytes for item in fixture_receipt.files
        ),
    }
    for module in (provision, manifest_module):
        for constant, attribute in _RUNTIME_CONSTANTS.items():
            monkeypatch.setattr(module, constant, runtime_values[attribute])

    wheel = _wheel("candidate_runtime")
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir(mode=0o700)
    (wheelhouse / wheel[0]).write_bytes(wheel[1])
    repository_lock = _write_lock(
        tmp_path / "repository-wheel-lock.json",
        [_entry(wheel[0], wheel[1])],
    )
    lock_bytes = repository_lock.read_bytes()
    lock_digest = hashlib.sha256(lock_bytes).hexdigest()
    for module in (provision, manifest_module):
        monkeypatch.setattr(
            module,
            "PARAKEET_REALTIME_EOU_WHEEL_LOCK_SHA256",
            lock_digest,
        )
        monkeypatch.setattr(
            module,
            "PARAKEET_REALTIME_EOU_WHEEL_LOCK_SIZE_BYTES",
            len(lock_bytes),
        )
    monkeypatch.setattr(provision, "_REPOSITORY_WHEEL_LOCK", repository_lock)

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
    model_path = model / "parakeet_realtime_eou_120m-v1.nemo"
    model_path.write_bytes(b"exact-model-fixture")
    model_receipt = (
        hashlib.sha256(model_path.read_bytes()).hexdigest(),
        model_path.stat().st_size,
    )
    monkeypatch.setattr(
        provision,
        "PARAKEET_REALTIME_EOU_MODEL_RECEIPT",
        model_receipt,
    )
    monkeypatch.setattr(
        manifest_module,
        "PARAKEET_REALTIME_EOU_MODEL_RECEIPT",
        model_receipt,
    )
    return venv, site_packages, model, wheelhouse, repository_lock, policies


def _provision(tmp_path: Path, monkeypatch, **overrides):
    venv, site_packages, model, wheelhouse, lock, policies = _inputs(
        tmp_path,
        monkeypatch,
    )
    arguments = {
        "venv_root": venv,
        "model_root": model,
        "wheelhouse": wheelhouse,
        "output_dir": tmp_path / "receipt",
    }
    arguments.update(overrides)
    manifest = provision.provision_candidate(**arguments)
    return manifest, venv, site_packages, model, wheelhouse, lock, policies


def _assert_manifest_unpublished(output: Path) -> None:
    for path in (
        output / "worker-manifest.json",
        output / provision._PENDING_MANIFEST_FILENAME,
    ):
        try:
            path.lstat()
        except FileNotFoundError:
            continue
        raise AssertionError(f"manifest entry remains unpublished: {path.name}")


def test_unpublished_assertion_detects_dangling_symlink(tmp_path):
    output = tmp_path / "dangling-entry"
    output.mkdir()
    (output / provision._PENDING_MANIFEST_FILENAME).symlink_to(
        output / "missing-target"
    )

    with pytest.raises(AssertionError, match="manifest entry remains unpublished"):
        _assert_manifest_unpublished(output)


def test_provision_binds_offline_schema_v5_closure_without_candidate_imports(
    tmp_path,
    monkeypatch,
):
    imported = []
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in {
            "huggingface_hub",
            "lightning",
            "nemo",
            "numpy",
            "torch",
            "torchaudio",
        }:
            imported.append(name)
            raise AssertionError("candidate import attempted")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    manifest, venv, _site_packages, model, _wheelhouse, lock, policies = (
        _provision(tmp_path, monkeypatch)
    )

    assert imported == []
    assert manifest.schema_version == 5
    assert manifest.adapter == PARAKEET_REALTIME_EOU_ADAPTER
    assert manifest.model_id == "parakeet-realtime-eou-120m-v1-en-cuda"
    assert manifest.python.path == venv / "bin" / "python"
    assert manifest.python.path.resolve(strict=True) == provision._TRUSTED_INTERPRETER
    assert os.access(manifest.python.path, os.X_OK)
    assert manifest.limits.startup_timeout_sec == 300.0
    assert manifest.limits.case_timeout_sec == 900.0
    assert manifest.adapter_config is not None
    assert manifest.adapter_config.as_dict() == {
        "python_version": "3.12.3",
        "nemo_version": "2.7.3",
        "torch_version": "2.7.1+cu126",
        "cuda_version": "12.6",
        "numpy_version": "2.2.6",
        "model_repo_id": "nvidia/parakeet_realtime_eou_120m-v1",
        "model_revision": "a7e2b4629593dce0ec19f600e00e9904353fda2d",
        "model_filename": "parakeet_realtime_eou_120m-v1.nemo",
        "language": "en",
        "device": "cuda:0",
        "dtype": "float32",
        "sample_rate": 16_000,
        "native_chunk_samples": 1_280,
        "maximum_tail_padding_samples": 48_000,
        "attention_context_left": 70,
        "attention_context_right": 1,
        "batch_size": 1,
        "eou_token": "<EOU>",
        "eob_token": "<EOB>",
        "use_amp": False,
        "wheel_lock_sha256": provision.PARAKEET_REALTIME_EOU_WHEEL_LOCK_SHA256,
        "runtime_content_sha256": (
            provision.PARAKEET_REALTIME_EOU_RUNTIME_CONTENT_SHA256
        ),
        "runtime_file_count": provision.PARAKEET_REALTIME_EOU_RUNTIME_FILE_COUNT,
        "runtime_total_size_bytes": (
            provision.PARAKEET_REALTIME_EOU_RUNTIME_TOTAL_SIZE_BYTES
        ),
        "runtime_maximum_file_bytes": (
            provision.PARAKEET_REALTIME_EOU_RUNTIME_MAXIMUM_FILE_BYTES
        ),
    }
    assert tuple(manifest.artifact_by_name) == PARAKEET_REALTIME_EOU_ARTIFACT_NAMES
    assert manifest.artifact_by_name["model-nemo"].path.parent == model
    assert manifest.worker.path == provision._FIXED_WORKER
    assert manifest.worker.sha256 == hashlib.sha256(
        provision._FIXED_WORKER.read_bytes()
    ).hexdigest()
    assert policies == [PARAKEET_WHEEL_ARCHIVE_POLICY]
    assert stat.S_IMODE(manifest.path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(manifest.path.stat().st_mode) == 0o600
    assert set(path.name for path in manifest.path.parent.iterdir()) == {
        "parakeet-realtime-eou-runtime-wheels.lock.json",
        "runtime-receipt.json",
        "worker-manifest.json",
    }
    assert not (manifest.path.parent / provision._PENDING_MANIFEST_FILENAME).exists()
    copied_lock = manifest.artifact_by_name["runtime-wheel-lock"].path
    assert copied_lock.read_bytes() == lock.read_bytes()
    assert stat.S_IMODE(copied_lock.stat().st_mode) == 0o600
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


def test_provision_receipts_exact_private_runtime_tree(tmp_path, monkeypatch):
    manifest, _venv, site_packages, _model, _wheelhouse, _lock, _policies = (
        _provision(tmp_path, monkeypatch)
    )
    receipt_path = manifest.artifact_by_name["runtime-receipt"].path

    receipt = load_runtime_tree_receipt(
        receipt_path,
        limits=PARAKEET_RUNTIME_TREE_LIMITS,
    )

    assert receipt.root == site_packages
    assert receipt.content_digest == provision.PARAKEET_REALTIME_EOU_RUNTIME_CONTENT_SHA256
    assert receipt.file_count == provision.PARAKEET_REALTIME_EOU_RUNTIME_FILE_COUNT
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o600


def test_provision_is_no_overwrite_and_preserves_first_manifest(
    tmp_path,
    monkeypatch,
):
    manifest, venv, _site_packages, model, wheelhouse, _lock, _policies = (
        _provision(tmp_path, monkeypatch)
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
def test_provision_rejects_non_exact_single_model_before_output(
    tmp_path,
    monkeypatch,
    mutation,
):
    venv, _site_packages, model, wheelhouse, _lock, _policies = _inputs(
        tmp_path,
        monkeypatch,
    )
    selected = model / "parakeet_realtime_eou_120m-v1.nemo"
    if mutation == "extra":
        (model / "README.md").write_text("extra", encoding="utf-8")
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


@pytest.mark.parametrize(
    "marker",
    (
        "home = /usr/bin\n"
        "include-system-site-packages = true\n"
        "version = 3.12.3\n"
        "executable = /usr/bin/python3.12\n",
        "home = /usr/bin\n"
        "include-system-site-packages = false\n"
        "version = 3.11.9\n"
        "executable = /usr/bin/python3.12\n",
        "home = /usr/bin\n"
        "include-system-site-packages = false\n"
        "version = 3.12.3\n",
        "home = /usr/bin\n"
        "home = /usr/bin\n"
        "include-system-site-packages = false\n"
        "version = 3.12.3\n"
        "executable = /usr/bin/python3.12\n",
        "home = /usr/bin\n"
        "include-system-site-packages = false\n"
        "version = 3.12.3\n"
        "executable = /tmp/python3.12\n",
        "home = /usr/bin\n"
        "include-system-site-packages = false\n"
        "version = 3.12.3\n"
        "executable = /usr/bin/python3.12\n"
        "prompt = candidate\n",
    ),
)
def test_provision_rejects_non_exact_python_marker_before_output(
    tmp_path,
    monkeypatch,
    marker,
):
    venv, _site_packages, model, wheelhouse, _lock, _policies = _inputs(
        tmp_path,
        monkeypatch,
    )
    (venv / "pyvenv.cfg").write_text(marker, encoding="utf-8")
    output = tmp_path / "bad-marker"

    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            venv_root=venv,
            model_root=model,
            wheelhouse=wheelhouse,
            output_dir=output,
        )

    assert not output.exists()


@pytest.mark.parametrize("executable", (True, False))
def test_provision_rejects_arbitrary_or_non_executable_python_target(
    tmp_path,
    monkeypatch,
    executable,
):
    venv, _site_packages, model, wheelhouse, _lock, _policies = _inputs(
        tmp_path,
        monkeypatch,
    )
    python = venv / "bin" / "python"
    python.unlink()
    arbitrary = tmp_path / "arbitrary-python"
    arbitrary.write_bytes(b"#!/bin/sh\nexit 0\n")
    arbitrary.chmod(0o700 if executable else 0o600)
    python.symlink_to(arbitrary)
    output = tmp_path / "bad-python"

    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            venv_root=venv,
            model_root=model,
            wheelhouse=wheelhouse,
            output_dir=output,
        )

    assert not output.exists()


def test_provision_rejects_tampered_repository_lock_before_output(
    tmp_path,
    monkeypatch,
):
    venv, _site_packages, model, wheelhouse, lock, _policies = _inputs(
        tmp_path,
        monkeypatch,
    )
    lock.write_bytes(lock.read_bytes() + b" ")
    output = tmp_path / "bad-lock"

    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            venv_root=venv,
            model_root=model,
            wheelhouse=wheelhouse,
            output_dir=output,
        )

    assert not output.exists()


def test_provision_rejects_wrong_runtime_without_manifest_publish(
    tmp_path,
    monkeypatch,
):
    venv, site_packages, model, wheelhouse, _lock, _policies = _inputs(
        tmp_path,
        monkeypatch,
    )
    (site_packages / "unexpected.py").write_bytes(b"changed")
    output = tmp_path / "wrong-runtime"

    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            venv_root=venv,
            model_root=model,
            wheelhouse=wheelhouse,
            output_dir=output,
        )

    assert output.is_dir()
    _assert_manifest_unpublished(output)


@pytest.mark.parametrize("mutation", ("same-bytes-new-identity", "wrong-semantics"))
def test_marker_is_semantically_and_identity_revalidated_before_publication(
    tmp_path,
    monkeypatch,
    mutation,
):
    venv, _site_packages, model, wheelhouse, _lock, _policies = _inputs(
        tmp_path,
        monkeypatch,
    )
    marker = venv / "pyvenv.cfg"
    original_load = provision.load_worker_manifest

    def load_then_mutate(path):
        manifest = original_load(path)
        replacement = tmp_path / "replacement-marker"
        payload = marker.read_bytes()
        if mutation == "wrong-semantics":
            payload = payload.replace(
                b"executable = /usr/bin/python3.12",
                b"executable = /tmp/python3.12",
            )
        replacement.write_bytes(payload)
        replacement.chmod(0o600)
        os.replace(replacement, marker)
        return manifest

    monkeypatch.setattr(provision, "load_worker_manifest", load_then_mutate)
    output = tmp_path / "changed-marker"

    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            venv_root=venv,
            model_root=model,
            wheelhouse=wheelhouse,
            output_dir=output,
        )

    _assert_manifest_unpublished(output)


def test_model_root_closure_is_rechecked_after_pending_validation(
    tmp_path,
    monkeypatch,
):
    venv, _site_packages, model, wheelhouse, _lock, _policies = _inputs(
        tmp_path,
        monkeypatch,
    )
    original_load = provision.load_worker_manifest

    def load_then_add_file(path):
        manifest = original_load(path)
        (model / "unexpected.txt").write_text("changed", encoding="utf-8")
        return manifest

    monkeypatch.setattr(provision, "load_worker_manifest", load_then_add_file)
    output = tmp_path / "changed-model"

    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            venv_root=venv,
            model_root=model,
            wheelhouse=wheelhouse,
            output_dir=output,
        )

    _assert_manifest_unpublished(output)


def test_canonical_manifest_bytes_equal_the_validated_pending_bytes(
    tmp_path,
    monkeypatch,
):
    validated = []
    original_load = provision.load_worker_manifest

    def load_and_record(path):
        validated.append(path.read_bytes())
        return original_load(path)

    monkeypatch.setattr(provision, "load_worker_manifest", load_and_record)

    manifest, *_rest = _provision(tmp_path, monkeypatch)

    assert len(validated) == 1
    assert manifest.path.read_bytes() == validated[0]
    assert manifest.digest == hashlib.sha256(validated[0]).hexdigest()


def test_parsed_manifest_digest_must_match_locally_generated_bytes(
    tmp_path,
    monkeypatch,
):
    venv, _site_packages, model, wheelhouse, _lock, _policies = _inputs(
        tmp_path,
        monkeypatch,
    )
    original_load = provision.load_worker_manifest

    def load_with_wrong_digest(path):
        return replace(original_load(path), digest="0" * 64)

    monkeypatch.setattr(provision, "load_worker_manifest", load_with_wrong_digest)
    output = tmp_path / "wrong-parsed-digest"

    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            venv_root=venv,
            model_root=model,
            wheelhouse=wheelhouse,
            output_dir=output,
        )

    _assert_manifest_unpublished(output)


@pytest.mark.parametrize("replacement", ("same-bytes-new-inode", "dangling-symlink"))
def test_pending_path_replacement_after_validation_cannot_be_published(
    tmp_path,
    monkeypatch,
    replacement,
):
    venv, _site_packages, model, wheelhouse, _lock, _policies = _inputs(
        tmp_path,
        monkeypatch,
    )
    original_publish = provision._publish_pending_manifest
    replacement_identity = []

    def replace_then_publish(binding, canonical):
        replacement_path = tmp_path / "replacement-pending"
        if replacement == "same-bytes-new-inode":
            replacement_path.write_bytes(binding.data)
            replacement_path.chmod(0o600)
        else:
            replacement_path.symlink_to(tmp_path / "missing-target")
        os.replace(replacement_path, binding.path)
        replacement_identity.append(binding.path.lstat().st_ino)
        return original_publish(binding, canonical)

    monkeypatch.setattr(
        provision,
        "_publish_pending_manifest",
        replace_then_publish,
    )
    output = tmp_path / f"replaced-pending-{replacement}"

    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            venv_root=venv,
            model_root=model,
            wheelhouse=wheelhouse,
            output_dir=output,
        )

    assert not (output / "worker-manifest.json").exists()
    pending = output / provision._PENDING_MANIFEST_FILENAME
    assert pending.lstat().st_ino == replacement_identity[0]
    if replacement == "same-bytes-new-inode":
        assert pending.is_file()
        assert pending.read_bytes().endswith(b"\n")
    else:
        assert pending.is_symlink()
        assert os.readlink(pending) == str(tmp_path / "missing-target")


def test_publication_rejects_a_different_inode_even_with_identical_bytes(
    tmp_path,
    monkeypatch,
):
    venv, _site_packages, model, wheelhouse, _lock, _policies = _inputs(
        tmp_path,
        monkeypatch,
    )
    real_link = os.link
    substitute = tmp_path / "publication-substitute"

    def link_substitute(
        source,
        destination,
        *,
        src_dir_fd=None,
        dst_dir_fd=None,
        follow_symlinks=True,
    ):
        del src_dir_fd, follow_symlinks
        substitute.write_bytes(Path(source).read_bytes())
        substitute.chmod(0o600)
        return real_link(
            substitute,
            destination,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=False,
        )

    monkeypatch.setattr(provision.os, "link", link_substitute)
    output = tmp_path / "wrong-publication-inode"

    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            venv_root=venv,
            model_root=model,
            wheelhouse=wheelhouse,
            output_dir=output,
        )

    canonical = output / "worker-manifest.json"
    assert canonical.samefile(substitute)
    assert not (output / provision._PENDING_MANIFEST_FILENAME).exists()
    assert substitute.exists()


@pytest.mark.parametrize("raced_entry", ("regular", "dangling-symlink"))
def test_link_collision_preserves_a_raced_canonical_entry(
    tmp_path,
    monkeypatch,
    raced_entry,
):
    venv, _site_packages, model, wheelhouse, _lock, _policies = _inputs(
        tmp_path,
        monkeypatch,
    )
    real_link = os.link
    output = tmp_path / f"canonical-link-race-{raced_entry}"
    canonical = output / "worker-manifest.json"

    def precreate_then_link(
        source,
        destination,
        *,
        src_dir_fd=None,
        dst_dir_fd=None,
        follow_symlinks=True,
    ):
        if raced_entry == "regular":
            canonical.write_bytes(b"raced canonical")
        else:
            canonical.symlink_to(output / "missing-target")
        return real_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(provision.os, "link", precreate_then_link)

    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            venv_root=venv,
            model_root=model,
            wheelhouse=wheelhouse,
            output_dir=output,
        )

    assert not (output / provision._PENDING_MANIFEST_FILENAME).exists()
    if raced_entry == "regular":
        assert canonical.read_bytes() == b"raced canonical"
    else:
        assert canonical.is_symlink()
        assert os.readlink(canonical) == str(output / "missing-target")


def test_post_link_foreign_replacement_is_preserved_while_owned_link_rolls_back(
    tmp_path,
    monkeypatch,
):
    venv, _site_packages, model, wheelhouse, _lock, _policies = _inputs(
        tmp_path,
        monkeypatch,
    )
    output = tmp_path / "post-link-foreign-replacement"
    canonical = output / "worker-manifest.json"

    def replace_linked_entry():
        replacement = tmp_path / "foreign-canonical"
        replacement.write_bytes(b"foreign canonical")
        replacement.chmod(0o600)
        os.replace(replacement, canonical)
        raise RuntimeError("injected after replacement")

    monkeypatch.setattr(
        provision,
        "_post_validation_fault_point",
        replace_linked_entry,
    )

    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            venv_root=venv,
            model_root=model,
            wheelhouse=wheelhouse,
            output_dir=output,
        )

    assert canonical.read_bytes() == b"foreign canonical"
    assert not (output / provision._PENDING_MANIFEST_FILENAME).exists()


def test_post_link_content_mutation_rolls_back_both_manifest_names(
    tmp_path,
    monkeypatch,
):
    venv, _site_packages, model, wheelhouse, _lock, _policies = _inputs(
        tmp_path,
        monkeypatch,
    )
    output = tmp_path / "post-link-mutation"
    canonical = output / "worker-manifest.json"

    def mutate_linked_inode():
        size = canonical.stat().st_size
        canonical.write_bytes(b"x" * size)
        canonical.chmod(0o600)

    monkeypatch.setattr(
        provision,
        "_post_validation_fault_point",
        mutate_linked_inode,
    )

    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            venv_root=venv,
            model_root=model,
            wheelhouse=wheelhouse,
            output_dir=output,
        )

    _assert_manifest_unpublished(output)


def test_post_validation_publication_fault_rolls_back_canonical_manifest(
    tmp_path,
    monkeypatch,
):
    venv, _site_packages, model, wheelhouse, _lock, _policies = _inputs(
        tmp_path,
        monkeypatch,
    )

    def fail_after_link():
        raise RuntimeError("injected after validation")

    monkeypatch.setattr(
        provision,
        "_post_validation_fault_point",
        fail_after_link,
    )
    output = tmp_path / "publish-fault"

    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            venv_root=venv,
            model_root=model,
            wheelhouse=wheelhouse,
            output_dir=output,
        )

    _assert_manifest_unpublished(output)


def test_repository_lock_has_exact_published_identity():
    payload = provision._repository_wheel_lock_bytes()

    assert len(payload) == provision.PARAKEET_REALTIME_EOU_WHEEL_LOCK_SIZE_BYTES
    assert (
        hashlib.sha256(payload).hexdigest()
        == provision.PARAKEET_REALTIME_EOU_WHEEL_LOCK_SHA256
    )
    assert (
        load_wheel_lock(
            provision._REPOSITORY_WHEEL_LOCK,
            expected_digest=provision.PARAKEET_REALTIME_EOU_WHEEL_LOCK_SHA256,
        ).digest
        == provision.PARAKEET_REALTIME_EOU_WHEEL_LOCK_SHA256
    )


def test_real_small_lock_wheelhouse_runtime_chain_is_provisioned(
    tmp_path,
    monkeypatch,
):
    wheel = _wheel("candidate_runtime")
    wheelhouse = tmp_path / "real-wheelhouse"
    wheelhouse.mkdir(mode=0o700)
    (wheelhouse / wheel[0]).write_bytes(wheel[1])
    repository_lock = _write_lock(
        tmp_path / "real-repository-lock.json",
        [_entry(wheel[0], wheel[1])],
    )
    lock = load_wheel_lock(repository_lock)
    prepared = prepare_locked_runtime(
        lock,
        wheelhouse,
        system_python=provision._TRUSTED_INTERPRETER,
        output_root=tmp_path / "real-candidate-venv",
        receipt_path=tmp_path / "builder-runtime-receipt.json",
        policy=RuntimeBuildPolicy(
            python_version="3.12.3",
            python_directory="python3.12",
            wheel_archive_policy=PARAKEET_WHEEL_ARCHIVE_POLICY,
        ),
    )
    runtime_values = {
        "content_digest": prepared.expectation.content_digest,
        "file_count": prepared.expectation.file_count,
        "total_size_bytes": prepared.expectation.total_size_bytes,
        "maximum_file_bytes": prepared.expectation.maximum_file_bytes,
    }
    lock_bytes = repository_lock.read_bytes()
    for module in (provision, manifest_module):
        for constant, attribute in _RUNTIME_CONSTANTS.items():
            monkeypatch.setattr(module, constant, runtime_values[attribute])
        monkeypatch.setattr(
            module,
            "PARAKEET_REALTIME_EOU_WHEEL_LOCK_SHA256",
            lock.digest,
        )
        monkeypatch.setattr(
            module,
            "PARAKEET_REALTIME_EOU_WHEEL_LOCK_SIZE_BYTES",
            len(lock_bytes),
        )
    monkeypatch.setattr(provision, "_REPOSITORY_WHEEL_LOCK", repository_lock)

    model = tmp_path / "real-model"
    model.mkdir(mode=0o700)
    model_path = model / "parakeet_realtime_eou_120m-v1.nemo"
    model_path.write_bytes(b"real-chain-model-fixture")
    model_receipt = (
        hashlib.sha256(model_path.read_bytes()).hexdigest(),
        model_path.stat().st_size,
    )
    monkeypatch.setattr(
        provision,
        "PARAKEET_REALTIME_EOU_MODEL_RECEIPT",
        model_receipt,
    )
    monkeypatch.setattr(
        manifest_module,
        "PARAKEET_REALTIME_EOU_MODEL_RECEIPT",
        model_receipt,
    )

    manifest = provision.provision_candidate(
        venv_root=prepared.root,
        model_root=model,
        wheelhouse=wheelhouse,
        output_dir=tmp_path / "real-chain-output",
    )

    receipt = load_runtime_tree_receipt(
        manifest.artifact_by_name["runtime-receipt"].path,
        limits=PARAKEET_RUNTIME_TREE_LIMITS,
    )
    assert receipt.files == prepared.expectation.files
    assert receipt.content_digest == prepared.expectation.content_digest
    assert manifest.python.path.resolve(strict=True) == provision._TRUSTED_INTERPRETER


def test_cli_error_is_detail_free_and_help_states_offline_boundary(
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
    help_text = " ".join(provision._parser().format_help().split())
    for promise in ("resolve", "install", "download", "import", "load"):
        assert promise in help_text
