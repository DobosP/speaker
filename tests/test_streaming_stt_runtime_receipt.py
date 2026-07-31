from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from types import MappingProxyType

import pytest

from tools.streaming_stt import runtime_receipt as receipt_module
from tools.streaming_stt.runtime_receipt import (
    DEFAULT_RUNTIME_TREE_LIMITS,
    RuntimeTreeReceiptError,
    generate_runtime_tree_receipt,
    load_runtime_tree_receipt,
    verify_runtime_tree_receipt,
)


def _runtime_tree(tmp_path: Path) -> Path:
    root = tmp_path / "runtime"
    root.mkdir(mode=0o700)
    (root / "alpha.bin").write_bytes(b"alpha")
    package = root / "package"
    package.mkdir()
    (package / "beta.bin").write_bytes(b"beta")
    return root


def _generate(tmp_path: Path):
    root = _runtime_tree(tmp_path)
    receipt_path = tmp_path / "runtime-receipt.json"
    return root, receipt_path, generate_runtime_tree_receipt(root, receipt_path)


def _write_private_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    path.chmod(0o600)


def test_generator_writes_canonical_private_exact_receipt(tmp_path):
    root, receipt_path, receipt = _generate(tmp_path)
    raw = receipt_path.read_bytes()
    payload = json.loads(raw)

    assert set(payload) == {"schema_version", "root", "files"}
    assert payload["schema_version"] == 1
    assert payload["root"] == root.as_posix()
    assert payload["files"] == sorted(payload["files"], key=lambda item: item["path"])
    assert [item["path"] for item in payload["files"]] == [
        "alpha.bin",
        "package/beta.bin",
    ]
    assert receipt.digest == hashlib.sha256(raw).hexdigest()
    assert receipt.root == root
    assert receipt.file_count == 2
    assert receipt.total_size_bytes == 9
    assert receipt.file_by_path is receipt.mapping
    assert isinstance(receipt.mapping, MappingProxyType)
    assert receipt.mapping["alpha.bin"].path == "alpha.bin"
    assert receipt.mapping["alpha.bin"].sha256 == hashlib.sha256(b"alpha").hexdigest()
    assert receipt_path.stat().st_mode & 0o777 == 0o600
    assert str(root) not in repr(receipt)
    assert str(receipt_path) not in repr(receipt)
    with pytest.raises(TypeError):
        receipt.mapping["other.bin"] = receipt.files[0]  # type: ignore[index]

    verify_runtime_tree_receipt(receipt)
    assert (
        load_runtime_tree_receipt(
            receipt_path,
            expected_digest=receipt.digest,
        )
        == receipt
    )


def test_generator_never_overwrites_and_rejects_destination_inside_tree(tmp_path):
    root, receipt_path, _receipt = _generate(tmp_path)

    with pytest.raises(RuntimeTreeReceiptError):
        generate_runtime_tree_receipt(root, receipt_path)
    with pytest.raises(RuntimeTreeReceiptError):
        generate_runtime_tree_receipt(root, root / "receipt.json")

    assert receipt_path.is_file()
    assert not (root / "receipt.json").exists()


@pytest.mark.parametrize(
    "mutation",
    ("content", "extra-file", "missing-file", "empty-directory"),
)
def test_verifier_requires_exact_current_tree(tmp_path, mutation):
    root, _receipt_path, receipt = _generate(tmp_path)

    if mutation == "content":
        (root / "alpha.bin").write_bytes(b"ALPHA")
    elif mutation == "extra-file":
        (root / "extra.bin").write_bytes(b"extra")
    elif mutation == "missing-file":
        (root / "alpha.bin").unlink()
    else:
        (root / "unreceipted-empty-directory").mkdir()

    with pytest.raises(RuntimeTreeReceiptError) as caught:
        verify_runtime_tree_receipt(receipt)
    assert str(caught.value) == ""


@pytest.mark.parametrize(
    "unsafe_kind",
    ("hardlink", "file-symlink", "directory-symlink", "fifo"),
)
def test_generator_rejects_non_exact_or_unsafe_tree_entries(tmp_path, unsafe_kind):
    root = tmp_path / "runtime"
    root.mkdir(mode=0o700)
    external = tmp_path / "external"
    external.mkdir()
    (external / "payload.bin").write_bytes(b"payload")

    if unsafe_kind == "hardlink":
        source = root / "source"
        source.write_bytes(b"payload")
        os.link(source, root / "unsafe")
    elif unsafe_kind == "file-symlink":
        (root / "unsafe").symlink_to(external / "payload.bin")
    elif unsafe_kind == "directory-symlink":
        (root / "unsafe").symlink_to(external, target_is_directory=True)
    else:
        os.mkfifo(root / "unsafe")

    with pytest.raises(RuntimeTreeReceiptError):
        generate_runtime_tree_receipt(root, tmp_path / "receipt.json")


def test_generator_hashes_zero_byte_regular_package_markers(tmp_path):
    root = tmp_path / "runtime"
    root.mkdir(mode=0o700)
    (root / "py.typed").write_bytes(b"")

    receipt = generate_runtime_tree_receipt(root, tmp_path / "receipt.json")

    marker = receipt.mapping["py.typed"]
    assert marker.size_bytes == 0
    assert marker.sha256 == hashlib.sha256(b"").hexdigest()
    verify_runtime_tree_receipt(receipt)


def test_generator_requires_private_owned_canonical_root(tmp_path):
    root = _runtime_tree(tmp_path)
    root.chmod(0o755)

    with pytest.raises(RuntimeTreeReceiptError):
        generate_runtime_tree_receipt(root, tmp_path / "permissive.json")

    root.chmod(0o700)
    alias = tmp_path / "runtime-alias"
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(RuntimeTreeReceiptError):
        generate_runtime_tree_receipt(alias, tmp_path / "aliased.json")


@pytest.mark.parametrize(
    "mutation",
    (
        "relative-root",
        "unsorted-files",
        "duplicate-file",
        "bad-digest",
        "negative-size",
        "extra-field",
    ),
)
def test_loader_rejects_noncanonical_schema(tmp_path, mutation):
    _root, receipt_path, _receipt = _generate(tmp_path)
    payload = json.loads(receipt_path.read_bytes())
    malformed = copy.deepcopy(payload)

    if mutation == "relative-root":
        malformed["root"] = "runtime"
    elif mutation == "unsorted-files":
        malformed["files"].reverse()
    elif mutation == "duplicate-file":
        malformed["files"].append(copy.deepcopy(malformed["files"][-1]))
    elif mutation == "bad-digest":
        malformed["files"][0]["sha256"] = "A" * 64
    elif mutation == "negative-size":
        malformed["files"][0]["size_bytes"] = -1
    else:
        malformed["unexpected"] = True

    candidate = tmp_path / f"{mutation}.json"
    _write_private_json(candidate, malformed)
    with pytest.raises(RuntimeTreeReceiptError):
        load_runtime_tree_receipt(candidate)


def test_loader_rejects_duplicate_json_keys(tmp_path):
    _root, receipt_path, _receipt = _generate(tmp_path)
    raw = receipt_path.read_bytes()
    duplicate = b'{"schema_version":1,' + raw[1:]
    candidate = tmp_path / "duplicate-key.json"
    candidate.write_bytes(duplicate)
    candidate.chmod(0o600)

    with pytest.raises(RuntimeTreeReceiptError):
        load_runtime_tree_receipt(candidate)


def test_loader_requires_private_single_link_receipt_and_expected_digest(tmp_path):
    _root, receipt_path, receipt = _generate(tmp_path)

    with pytest.raises(RuntimeTreeReceiptError):
        load_runtime_tree_receipt(receipt_path, expected_digest="0" * 64)

    receipt_path.chmod(0o644)
    with pytest.raises(RuntimeTreeReceiptError):
        load_runtime_tree_receipt(receipt_path)
    receipt_path.chmod(0o600)

    linked = tmp_path / "linked-receipt.json"
    os.link(receipt_path, linked)
    with pytest.raises(RuntimeTreeReceiptError):
        load_runtime_tree_receipt(receipt_path)
    with pytest.raises(RuntimeTreeReceiptError):
        load_runtime_tree_receipt(linked)

    assert receipt.digest


@pytest.mark.parametrize(
    "limits",
    (
        replace(DEFAULT_RUNTIME_TREE_LIMITS, maximum_files=1),
        replace(DEFAULT_RUNTIME_TREE_LIMITS, maximum_directories=1),
        replace(DEFAULT_RUNTIME_TREE_LIMITS, maximum_file_bytes=4),
        replace(DEFAULT_RUNTIME_TREE_LIMITS, maximum_aggregate_bytes=8),
        replace(DEFAULT_RUNTIME_TREE_LIMITS, maximum_receipt_bytes=32),
        replace(DEFAULT_RUNTIME_TREE_LIMITS, maximum_relative_path_bytes=8),
        replace(DEFAULT_RUNTIME_TREE_LIMITS, maximum_depth=1),
    ),
)
def test_generator_enforces_each_resource_ceiling(tmp_path, limits):
    root = _runtime_tree(tmp_path)
    second_package = root / "second"
    second_package.mkdir()
    (second_package / "gamma.bin").write_bytes(b"g")

    with pytest.raises(RuntimeTreeReceiptError):
        generate_runtime_tree_receipt(
            root,
            tmp_path / "receipt.json",
            limits=limits,
        )


def test_verifier_detects_same_content_identity_swap_during_hash_open(
    tmp_path,
    monkeypatch,
):
    root, _receipt_path, receipt = _generate(tmp_path)
    target = root / "alpha.bin"
    original_open = receipt_module.os.open
    swapped = False

    def racing_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if path == "alpha.bin" and dir_fd is not None and not swapped:
            swapped = True
            target.unlink()
            target.write_bytes(b"alpha")
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(receipt_module.os, "open", racing_open)

    with pytest.raises(RuntimeTreeReceiptError):
        verify_runtime_tree_receipt(receipt)
    assert swapped is True


def test_verifier_binds_exact_receipt_bytes(tmp_path):
    _root, receipt_path, receipt = _generate(tmp_path)
    receipt_path.write_bytes(receipt_path.read_bytes() + b" ")
    receipt_path.chmod(0o600)

    with pytest.raises(RuntimeTreeReceiptError):
        verify_runtime_tree_receipt(receipt)


def test_venv_runtime_location_binds_exact_site_packages_root(tmp_path):
    from tools.streaming_stt.runtime_receipt import verify_venv_runtime_location

    venv = tmp_path / "candidate-venv"
    python = venv / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"python")
    runtime = venv / "lib" / "python3.12" / "site-packages"
    runtime.mkdir(parents=True)
    runtime.chmod(0o700)
    (runtime / "candidate.py").write_bytes(b"candidate")
    receipt = generate_runtime_tree_receipt(
        runtime,
        tmp_path / "venv-receipt.json",
    )

    verify_venv_runtime_location(receipt, python)

    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    (outside / "candidate.py").write_bytes(b"candidate")
    outside_receipt = generate_runtime_tree_receipt(
        outside,
        tmp_path / "outside-receipt.json",
    )
    with pytest.raises(RuntimeTreeReceiptError):
        verify_venv_runtime_location(outside_receipt, python)
