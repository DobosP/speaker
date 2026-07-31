from __future__ import annotations

import base64
import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import struct
from urllib.parse import quote
import zipfile

import pytest

from tools.streaming_stt import wheel_lock as wheel_lock_module
from tools.streaming_stt.runtime_receipt import (
    NEMOTRON_RUNTIME_TREE_LIMITS,
    generate_runtime_tree_receipt,
)
from tools.streaming_stt.wheel_lock import (
    InstallerFile,
    WheelLockError,
    generate_measured_wheel_lock,
    load_wheel_lock,
    verify_installed_wheels,
    verify_locked_wheel_install,
    verify_wheelhouse,
)


def _write_pylock(
    path: Path,
    wheels: list[tuple[str, bytes, dict[str, bytes]]],
    *,
    mutate=None,
) -> Path:
    packages: list[dict[str, object]] = []
    for filename, data, _contents in sorted(wheels):
        selected = _entry(filename, data)
        packages.append(
            {
                "name": selected["distribution"],
                "version": selected["version"],
                "wheels": [
                    {
                        "url": selected["source_url"],
                        "size": selected["size_bytes"],
                        "hashes": {"sha256": selected["sha256"]},
                    }
                ],
            }
        )
    if mutate is not None:
        mutate(packages)
    lines = [
        'lock-version = "1.0"',
        'created-by = "uv"',
        'requires-python = ">=3.12.3"',
        "",
    ]
    for package in packages:
        wheel = package["wheels"][0]
        wheel_fields = [
            f'url = "{wheel["url"]}"',
            f'hashes = {{ sha256 = "{wheel["hashes"]["sha256"]}" }}',
        ]
        if "size" in wheel:
            wheel_fields.insert(1, f"size = {json.dumps(wheel['size'])}")
        lines.extend(
            [
                "[[packages]]",
                f'name = "{package["name"]}"',
                f'version = "{package["version"]}"',
                f"wheels = [{{ {', '.join(wheel_fields)} }}]",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _record_hash(data: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest())
    return f"sha256={encoded.rstrip(b'=').decode('ascii')}"


def _record_bytes(rows: list[list[str]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _wheel(
    distribution: str,
    *,
    version: str = "1.0",
    package_path: str | None = None,
    package_bytes: bytes = b"VALUE = 1\n",
    additional_contents: tuple[tuple[str, bytes], ...] = (),
    extra_archive: list[tuple[str | zipfile.ZipInfo, bytes]] = (),
    record_mutation=None,
    metadata_name: str | None = None,
    metadata_version: str | None = None,
    internal_tag: str = "py3-none-any",
    root_is_purelib: str = "true",
) -> tuple[str, bytes, dict[str, bytes]]:
    filename = f"{distribution}-{version}-py3-none-any.whl"
    dist_info = f"{distribution}-{version}.dist-info"
    selected_package_path = package_path or f"{distribution}/__init__.py"
    contents: list[tuple[str | zipfile.ZipInfo, bytes]] = [
        (selected_package_path, package_bytes),
        (
            f"{dist_info}/METADATA",
            (
                "Metadata-Version: 2.1\n"
                f"Name: {metadata_name or distribution}\n"
                f"Version: {metadata_version or version}\n"
            ).encode(),
        ),
        (
            f"{dist_info}/WHEEL",
            (
                "Wheel-Version: 1.0\n"
                f"Root-Is-Purelib: {root_is_purelib}\n"
                f"Tag: {internal_tag}\n"
            ).encode(),
        ),
        *additional_contents,
    ]
    rows = [[str(path), _record_hash(data), str(len(data))] for path, data in contents]
    record_path = f"{dist_info}/RECORD"
    rows.append([record_path, "", ""])
    if record_mutation is not None:
        rows = record_mutation(copy.deepcopy(rows))
    record = _record_bytes(rows)
    entries = [*contents, (record_path, record), *extra_archive]
    target = io.BytesIO()
    with zipfile.ZipFile(
        target,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for path, data in entries:
            archive.writestr(path, data)

    installed: dict[str, bytes] = {}
    data_prefix = f"{distribution}-{version}.data/"
    for path, data in contents:
        assert isinstance(path, str)
        if path.startswith(f"{data_prefix}purelib/"):
            installed[path.removeprefix(f"{data_prefix}purelib/")] = data
        elif path.startswith(f"{data_prefix}platlib/"):
            installed[path.removeprefix(f"{data_prefix}platlib/")] = data
        else:
            installed[path] = data
    return filename, target.getvalue(), installed


def _entry(filename: str, data: bytes) -> dict[str, object]:
    stem = filename.removesuffix(".whl")
    prefix, python_tag, abi_tag, platform_tag = stem.rsplit("-", 3)
    distribution, version = prefix.rsplit("-", 1)
    return {
        "distribution": distribution.replace("_", "-"),
        "version": version,
        "filename": filename,
        "wheel_tag": f"{python_tag}-{abi_tag}-{platform_tag}",
        "source_url": (
            f"https://files.example.invalid/wheels/{quote(filename, safe='-_.+')}"
        ),
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _write_lock(
    path: Path,
    entries: list[dict[str, object]],
) -> Path:
    payload = {
        "schema_version": 1,
        "wheels": sorted(
            entries,
            key=lambda item: (
                item["distribution"],
                item["version"],
                item["filename"],
            ),
        ),
    }
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    path.chmod(0o600)
    return path


def _selection(
    tmp_path: Path,
    wheels: list[tuple[str, bytes, dict[str, bytes]]],
):
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir(mode=0o700)
    entries: list[dict[str, object]] = []
    installed: dict[str, bytes] = {}
    for filename, data, mapped in wheels:
        (wheelhouse / filename).write_bytes(data)
        entries.append(_entry(filename, data))
        installed.update(mapped)
    lock_path = _write_lock(tmp_path / "wheel-lock.json", entries)
    lock = load_wheel_lock(lock_path)
    verified = verify_wheelhouse(lock, wheelhouse)
    return lock, verified, wheelhouse, installed


def _encode_row(path: str, digest: str | None, size: int | None) -> list[str]:
    if digest is None:
        return [path, "", ""]
    encoded = base64.urlsafe_b64encode(bytes.fromhex(digest)).rstrip(b"=")
    return [path, f"sha256={encoded.decode('ascii')}", str(size)]


def _install(
    tmp_path: Path,
    verified,
    contents: dict[str, bytes],
    *,
    installer_files: tuple[InstallerFile, ...] = (),
    mutate=None,
):
    root = tmp_path / "venv" / "lib" / "python3.12" / "site-packages"
    root.mkdir(parents=True)
    root.chmod(0o700)
    for relative, data in contents.items():
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    installer_by_record: dict[str, list[InstallerFile]] = {
        wheel.record_path: [] for wheel in verified.wheels
    }
    for item in installer_files:
        destination = root / item.relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(item.content)
        dist_info = item.relative_path.rsplit("/", 1)[0]
        record = next(
            wheel.record_path
            for wheel in verified.wheels
            if wheel.dist_info_directory == dist_info
        )
        installer_by_record[record].append(item)
    for wheel in verified.wheels:
        rows = [
            _encode_row(row.installed_path, row.sha256, row.size_bytes)
            for row in wheel.release_rows
        ]
        rows.extend(
            [
                item.relative_path,
                _record_hash(item.content),
                str(len(item.content)),
            ]
            for item in installer_by_record[wheel.record_path]
        )
        rows.sort(key=lambda row: row[0])
        destination = root / wheel.record_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(_record_bytes(rows))
    if mutate is not None:
        mutate(root, verified)
    receipt_path = tmp_path / "runtime-receipt.json"
    receipt = generate_runtime_tree_receipt(
        root,
        receipt_path,
        limits=NEMOTRON_RUNTIME_TREE_LIMITS,
    )
    return root, receipt


def test_generator_uses_uv_context_but_measures_exact_private_wheels(tmp_path):
    alpha = _wheel("alpha")
    bravo = _wheel("bravo")
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir(mode=0o700)
    for filename, data, _contents in (alpha, bravo):
        (wheelhouse / filename).write_bytes(data)
    pylock = _write_pylock(tmp_path / "pylock.toml", [bravo, alpha])

    lock = generate_measured_wheel_lock(
        pylock,
        wheelhouse,
        tmp_path / "strict-lock.json",
    )

    assert [wheel.distribution for wheel in lock.wheels] == ["alpha", "bravo"]
    assert lock.total_size_bytes == len(alpha[1]) + len(bravo[1])
    assert lock.wheels[0].source_url == _entry(alpha[0], alpha[1])["source_url"]
    assert lock.wheels[0].sha256 == hashlib.sha256(alpha[1]).hexdigest()
    assert lock.path.stat().st_mode & 0o777 == 0o600
    assert load_wheel_lock(lock.path, expected_digest=lock.digest) == lock
    assert len(verify_wheelhouse(lock, wheelhouse).wheels) == 2


def test_generator_allows_absent_context_size_but_uses_measurement(tmp_path):
    alpha = _wheel("alpha")
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir(mode=0o700)
    (wheelhouse / alpha[0]).write_bytes(alpha[1])

    def omit_size(packages):
        del packages[0]["wheels"][0]["size"]

    pylock = _write_pylock(
        tmp_path / "pylock.toml",
        [alpha],
        mutate=omit_size,
    )
    lock = generate_measured_wheel_lock(
        pylock,
        wheelhouse,
        tmp_path / "strict-lock.json",
    )

    assert lock.wheels[0].size_bytes == len(alpha[1])


@pytest.mark.parametrize(
    "mutation",
    (
        "bad-size",
        "malformed-size",
        "bad-hash",
        "bad-source",
        "missing-package",
        "duplicate-package",
    ),
)
def test_generator_rejects_pylock_that_does_not_match_measured_closure(
    tmp_path,
    mutation,
):
    alpha = _wheel("alpha")
    bravo = _wheel("bravo")
    wheels = [alpha, bravo]
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir(mode=0o700)
    for filename, data, _contents in wheels:
        (wheelhouse / filename).write_bytes(data)

    def mutate(packages):
        if mutation == "bad-size":
            packages[0]["wheels"][0]["size"] += 1
        elif mutation == "malformed-size":
            packages[0]["wheels"][0]["size"] = False
        elif mutation == "bad-hash":
            packages[0]["wheels"][0]["hashes"]["sha256"] = "0" * 64
        elif mutation == "bad-source":
            packages[0]["wheels"][0]["url"] = (
                "https://files.example.invalid/wheels/other.whl"
            )
        elif mutation == "missing-package":
            packages.pop()
        else:
            packages.append(copy.deepcopy(packages[0]))

    pylock = _write_pylock(
        tmp_path / "pylock.toml",
        wheels,
        mutate=mutate,
    )
    destination = tmp_path / "strict-lock.json"

    with pytest.raises(WheelLockError):
        generate_measured_wheel_lock(pylock, wheelhouse, destination)

    assert not destination.exists()


def test_generator_is_no_overwrite_and_refuses_output_inside_wheelhouse(tmp_path):
    wheel = _wheel("alpha")
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir(mode=0o700)
    (wheelhouse / wheel[0]).write_bytes(wheel[1])
    pylock = _write_pylock(tmp_path / "pylock.toml", [wheel])
    destination = tmp_path / "strict-lock.json"
    first = generate_measured_wheel_lock(pylock, wheelhouse, destination)
    before = destination.read_bytes()

    with pytest.raises(WheelLockError):
        generate_measured_wheel_lock(pylock, wheelhouse, destination)
    with pytest.raises(WheelLockError):
        generate_measured_wheel_lock(
            pylock,
            wheelhouse,
            wheelhouse / "strict-lock.json",
        )

    assert destination.read_bytes() == before
    assert first.path == destination


def test_exact_wheelhouse_and_installed_union_produce_root_independent_summary(
    tmp_path,
):
    alpha = _wheel("alpha")
    bravo = _wheel(
        "bravo",
        package_path="bravo-1.0.data/purelib/bravo/__init__.py",
        package_bytes=b"BRAVO = 2\n",
    )
    lock, verified, wheelhouse, contents = _selection(
        tmp_path,
        [bravo, alpha],
    )
    installer_files = (
        InstallerFile("alpha-1.0.dist-info/INSTALLER", b"uv"),
        InstallerFile("alpha-1.0.dist-info/REQUESTED", b""),
        InstallerFile("bravo-1.0.dist-info/INSTALLER", b"uv"),
    )
    _root, receipt = _install(
        tmp_path,
        verified,
        contents,
        installer_files=installer_files,
    )

    expected = verify_installed_wheels(
        verified,
        receipt,
        installer_files=installer_files,
    )
    wrapped = verify_locked_wheel_install(
        lock,
        wheelhouse,
        receipt,
        installer_files=installer_files,
    )

    assert wrapped == expected
    assert expected.files == receipt.files
    assert expected.content_digest == receipt.content_digest
    assert expected.file_count == receipt.file_count
    assert expected.total_size_bytes == receipt.total_size_bytes
    assert expected.maximum_file_bytes == max(item.size_bytes for item in receipt.files)
    assert "bravo/__init__.py" in receipt.file_by_path
    assert all(".data/" not in path for path in receipt.file_by_path)
    assert lock.total_size_bytes == sum(len(item[1]) for item in (alpha, bravo))

    second = tmp_path / "second"
    second.mkdir()
    _lock2, verified2, _wheelhouse2, contents2 = _selection(
        second,
        [alpha, bravo],
    )
    _root2, receipt2 = _install(
        second,
        verified2,
        contents2,
        installer_files=installer_files,
    )
    expected2 = verify_installed_wheels(
        verified2,
        receipt2,
        installer_files=installer_files,
    )
    assert expected2.content_digest == expected.content_digest


def test_nemotron_profile_allows_only_exact_inert_setuptools_pth(tmp_path):
    pth = (
        b"import os; var = 'SETUPTOOLS_USE_DISTUTILS'; enabled = "
        b"os.environ.get(var, 'local') == 'local'; enabled and "
        b"__import__('_distutils_hack').add_shim(); \n"
    )
    wheel = _wheel(
        "setuptools",
        version="81.0.0",
        package_path="distutils-precedence.pth",
        package_bytes=pth,
    )
    _lock, verified, _wheelhouse, contents = _selection(tmp_path, [wheel])
    _root, receipt = _install(tmp_path, verified, contents)

    expected = verify_installed_wheels(verified, receipt)

    assert expected.files == receipt.files
    assert (
        receipt.file_by_path["distutils-precedence.pth"].sha256
        == hashlib.sha256(pth).hexdigest()
    )


def test_nemotron_profile_allows_only_exact_inert_cuda_redirector_pth(tmp_path):
    pth = (
        b"# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & "
        b"AFFILIATES. All rights reserved.\n"
        b"# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE\n\n"
        b"import _cuda_bindings_redirector\n"
    )
    wheel = _wheel(
        "cuda_bindings",
        version="12.9.7",
        package_path="_cuda_bindings_redirector.pth",
        package_bytes=pth,
    )
    _lock, verified, _wheelhouse, contents = _selection(tmp_path, [wheel])
    _root, receipt = _install(tmp_path, verified, contents)

    assert verify_installed_wheels(verified, receipt).files == receipt.files


def test_nemotron_profile_deduplicates_only_exact_shared_namespace_file(
    tmp_path,
    monkeypatch,
):
    alpha = _wheel("alpha", package_path="shared/__init__.py", package_bytes=b"")
    bravo = _wheel("bravo", package_path="shared/__init__.py", package_bytes=b"")
    monkeypatch.setitem(
        wheel_lock_module._NEMOTRON_SHARED_RUNTIME_FILES,
        "shared/__init__.py",
        {
            "receipt": (hashlib.sha256(b"").hexdigest(), 0),
            "owners": frozenset({("alpha", "1.0"), ("bravo", "1.0")}),
        },
    )
    _lock, verified, _wheelhouse, contents = _selection(
        tmp_path,
        [alpha, bravo],
    )
    _root, receipt = _install(tmp_path, verified, contents)

    expected = verify_installed_wheels(verified, receipt)

    assert expected.files == receipt.files
    assert (
        tuple(item.relative_path for item in expected.files).count("shared/__init__.py")
        == 1
    )


def test_nemotron_profile_binds_one_external_non_runtime_file(
    tmp_path,
    monkeypatch,
):
    archive_path = "sympy-1.14.0.data/data/share/man/man1/isympy.1"
    data = b"exact non-runtime manual fixture\n"
    monkeypatch.setitem(
        wheel_lock_module._NEMOTRON_EXTERNAL_RUNTIME_EXCLUSIONS,
        ("sympy", "1.14.0", archive_path),
        (hashlib.sha256(data).hexdigest(), len(data)),
    )
    wheel = _wheel(
        "sympy",
        version="1.14.0",
        package_path=archive_path,
        package_bytes=data,
    )
    _lock, verified, _wheelhouse, contents = _selection(tmp_path, [wheel])
    contents.pop(archive_path)
    _root, receipt = _install(tmp_path, verified, contents)

    expected = verify_installed_wheels(verified, receipt)

    assert expected.files == receipt.files
    assert archive_path not in receipt.file_by_path
    assert verified.wheels[0].excluded_external_files == (
        wheel_lock_module.RuntimeFileReceipt(
            relative_path=archive_path,
            sha256=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
        ),
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "distribution",
        "version",
        "filename",
        "wheel-tag",
        "http",
        "url-filename",
        "url-query",
        "zero-size",
        "uppercase-sha",
        "extra-field",
    ),
)
def test_lock_rejects_non_exact_wheel_selection(tmp_path, mutation):
    filename, data, _contents = _wheel("alpha")
    entry = _entry(filename, data)
    if mutation == "distribution":
        entry["distribution"] = "Alpha_Name"
    elif mutation == "version":
        entry["version"] = "01.0"
    elif mutation == "filename":
        entry["filename"] = "alpha-1.0.tar.gz"
    elif mutation == "wheel-tag":
        entry["wheel_tag"] = "cp312-none-any"
    elif mutation == "http":
        entry["source_url"] = str(entry["source_url"]).replace("https:", "http:")
    elif mutation == "url-filename":
        entry["source_url"] = "https://files.example.invalid/wheels/other.whl"
    elif mutation == "url-query":
        entry["source_url"] = f"{entry['source_url']}?credential=forbidden"
    elif mutation == "zero-size":
        entry["size_bytes"] = 0
    elif mutation == "uppercase-sha":
        entry["sha256"] = str(entry["sha256"]).upper()
    else:
        entry["editable"] = False
    path = _write_lock(tmp_path / f"{mutation}.json", [entry])

    with pytest.raises(WheelLockError):
        load_wheel_lock(path)


def test_lock_rejects_duplicate_keys_unordered_or_duplicate_distribution(tmp_path):
    alpha = _wheel("alpha")
    bravo = _wheel("bravo")
    entries = [_entry(alpha[0], alpha[1]), _entry(bravo[0], bravo[1])]
    path = _write_lock(tmp_path / "lock.json", entries)
    raw = path.read_bytes()
    path.write_bytes(b'{"schema_version":1,' + raw[1:])
    path.chmod(0o600)
    with pytest.raises(WheelLockError):
        load_wheel_lock(path)

    payload = {"schema_version": 1, "wheels": list(reversed(entries))}
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.chmod(0o600)
    with pytest.raises(WheelLockError):
        load_wheel_lock(path)

    duplicate = copy.deepcopy(entries[1])
    duplicate["distribution"] = "alpha"
    duplicate["filename"] = "alpha-2.0-py3-none-any.whl"
    duplicate["version"] = "2.0"
    duplicate["source_url"] = (
        "https://files.example.invalid/wheels/alpha-2.0-py3-none-any.whl"
    )
    _write_lock(path, [entries[0], duplicate])
    with pytest.raises(WheelLockError):
        load_wheel_lock(path)


def test_lock_requires_private_single_link_file_and_digest(tmp_path):
    filename, data, _contents = _wheel("alpha")
    path = _write_lock(tmp_path / "lock.json", [_entry(filename, data)])
    lock = load_wheel_lock(path)
    with pytest.raises(WheelLockError):
        load_wheel_lock(path, expected_digest="0" * 64)

    path.chmod(0o644)
    with pytest.raises(WheelLockError):
        load_wheel_lock(path)
    path.chmod(0o600)

    linked = tmp_path / "linked.json"
    os.link(path, linked)
    with pytest.raises(WheelLockError):
        load_wheel_lock(path)
    assert lock.wheels[0].filename == filename


def test_lock_privacy_and_content_are_checked_on_same_open_inode(
    tmp_path,
    monkeypatch,
):
    filename, data, _contents = _wheel("alpha")
    path = _write_lock(tmp_path / "lock.json", [_entry(filename, data)])
    replacement = path.read_bytes()
    original_open = wheel_lock_module.os.open
    swapped = False

    def racing_open(selected, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if selected == path.name and dir_fd is not None and not swapped:
            swapped = True
            # Keep the original inode linked so a fast tmpfs/overlayfs cannot
            # immediately recycle the inode number and make the race fixture
            # spuriously indistinguishable at nanosecond timestamp resolution.
            path.rename(path.with_name("original-lock.json"))
            path.write_bytes(replacement)
            path.chmod(0o600)
        return original_open(selected, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(wheel_lock_module.os, "open", racing_open)

    with pytest.raises(WheelLockError):
        load_wheel_lock(path)
    assert swapped is True


@pytest.mark.parametrize(
    "mutation",
    ("extra", "missing", "symlink", "hardlink", "public-root"),
)
def test_wheelhouse_requires_exact_private_regular_single_link_files(
    tmp_path,
    mutation,
):
    wheel = _wheel("alpha")
    lock, _verified, wheelhouse, _contents = _selection(tmp_path, [wheel])
    selected = wheelhouse / wheel[0]
    if mutation == "extra":
        (wheelhouse / "extra.whl").write_bytes(b"extra")
    elif mutation == "missing":
        selected.unlink()
    elif mutation == "symlink":
        selected.unlink()
        outside = tmp_path / "outside.whl"
        outside.write_bytes(wheel[1])
        selected.symlink_to(outside)
    elif mutation == "hardlink":
        linked = tmp_path / "linked.whl"
        os.link(selected, linked)
    else:
        wheelhouse.chmod(0o755)

    with pytest.raises(WheelLockError):
        verify_wheelhouse(lock, wheelhouse)


def test_wheelhouse_rechecks_lock_and_wheel_bytes(tmp_path):
    wheel = _wheel("alpha")
    lock, _verified, wheelhouse, _contents = _selection(tmp_path, [wheel])
    (wheelhouse / wheel[0]).write_bytes(wheel[1] + b"tampered")
    with pytest.raises(WheelLockError):
        verify_wheelhouse(lock, wheelhouse)

    (wheelhouse / wheel[0]).write_bytes(wheel[1])
    lock.path.write_bytes(lock.path.read_bytes() + b" ")
    lock.path.chmod(0o600)
    with pytest.raises(WheelLockError):
        verify_wheelhouse(lock, wheelhouse)


def _unsafe_wheel(
    kind: str,
) -> tuple[str, bytes, dict[str, bytes]]:
    if kind == "traversal":
        return _wheel("alpha", package_path="../escape.py")
    if kind == "backslash":
        return _wheel("alpha", package_path=r"alpha\escape.py")
    if kind == "pth":
        return _wheel("alpha", package_path="alpha-hook.pth")
    if kind == "shadow":
        return _wheel("alpha", package_path="json.py")
    if kind == "unrecorded":
        return _wheel(
            "alpha",
            extra_archive=[("alpha/hidden.py", b"hidden")],
        )
    if kind == "wrong-record-hash":

        def wrong_hash(rows):
            rows[0][1] = f"sha256={base64.urlsafe_b64encode(b'0' * 32).decode()}"
            return rows

        return _wheel("alpha", record_mutation=wrong_hash)
    if kind == "duplicate-record":

        def duplicate_record(rows):
            rows.append(copy.deepcopy(rows[0]))
            return rows

        return _wheel("alpha", record_mutation=duplicate_record)
    if kind == "noncanonical-size":

        def noncanonical_size(rows):
            rows[0][2] = f"0{rows[0][2]}"
            return rows

        return _wheel("alpha", record_mutation=noncanonical_size)
    if kind == "foreign-dist-info":
        return _wheel(
            "alpha",
            package_path="other-1.0.dist-info/METADATA",
            package_bytes=b"Name: other\nVersion: 1.0\n",
        )
    if kind == "metadata-name":
        return _wheel("alpha", metadata_name="other")
    if kind == "metadata-version":
        return _wheel("alpha", metadata_version="2.0")
    if kind == "wheel-tag":
        return _wheel("alpha", internal_tag="cp312-none-any")
    if kind == "wheel-root":
        return _wheel("alpha", root_is_purelib="maybe")
    if kind == "nul-original":

        def truncated_record(rows):
            rows[0][0] = "alpha/x.py"
            return rows

        filename, data, installed = _wheel(
            "alpha",
            package_path="alpha/x.pyXYZ",
            record_mutation=truncated_record,
        )
        original = b"alpha/x.pyXYZ"
        poisoned = b"alpha/x.py\x00YZ"
        assert data.count(original) == 2
        return filename, data.replace(original, poisoned), installed
    if kind == "symlink":
        info = zipfile.ZipInfo("alpha/link.py")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        return _wheel("alpha", extra_archive=[(info, b"../outside")])
    if kind == "zip-bomb":
        return _wheel(
            "alpha",
            package_bytes=b"\x00" * (2 * 1024 * 1024),
        )
    raise AssertionError(kind)


@pytest.mark.parametrize(
    "kind",
    (
        "traversal",
        "backslash",
        "pth",
        "shadow",
        "unrecorded",
        "wrong-record-hash",
        "duplicate-record",
        "noncanonical-size",
        "foreign-dist-info",
        "metadata-name",
        "metadata-version",
        "wheel-tag",
        "wheel-root",
        "nul-original",
        "symlink",
        "zip-bomb",
    ),
)
def test_archive_rejects_unsafe_paths_hooks_hidden_payloads_and_bombs(
    tmp_path,
    kind,
):
    wheel = _unsafe_wheel(kind)
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir(mode=0o700)
    (wheelhouse / wheel[0]).write_bytes(wheel[1])
    lock = load_wheel_lock(
        _write_lock(tmp_path / "lock.json", [_entry(wheel[0], wheel[1])])
    )

    with pytest.raises(WheelLockError):
        verify_wheelhouse(lock, wheelhouse)


def test_zip_central_directory_is_bounded_before_archive_allocation(tmp_path):
    filename, data, contents = _wheel("alpha")
    malformed = bytearray(data)
    eocd = malformed.rfind(b"PK\x05\x06")
    assert eocd >= 0
    struct.pack_into("<H", malformed, eocd + 10, 0xFFFF)
    wheel = filename, bytes(malformed), contents
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir(mode=0o700)
    (wheelhouse / filename).write_bytes(wheel[1])
    lock = load_wheel_lock(
        _write_lock(tmp_path / "lock.json", [_entry(filename, wheel[1])])
    )

    with pytest.raises(WheelLockError):
        verify_wheelhouse(lock, wheelhouse)


def test_nested_vendored_dist_info_record_is_owned_by_top_level_record(tmp_path):
    wheel = _wheel(
        "alpha",
        additional_contents=(
            (
                "alpha/_vendor/vendored-2.0.dist-info/METADATA",
                b"Name: vendored\nVersion: 2.0\n",
            ),
            (
                "alpha/_vendor/vendored-2.0.dist-info/RECORD",
                b"vendored.py,,\n",
            ),
        ),
    )
    lock, verified, _wheelhouse, _contents = _selection(tmp_path, [wheel])

    assert lock.wheels[0].distribution == "alpha"
    assert len(verified.wheels) == 1
    assert "alpha/_vendor/vendored-2.0.dist-info/RECORD" in {
        item.relative_path for item in verified.wheels[0].release_files
    }


def test_archive_rejects_overlapping_record_ownership_across_wheels(tmp_path):
    alpha = _wheel("alpha", package_path="shared/payload.py")
    bravo = _wheel("bravo", package_path="shared/payload.py")
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir(mode=0o700)
    entries = []
    for wheel in (alpha, bravo):
        (wheelhouse / wheel[0]).write_bytes(wheel[1])
        entries.append(_entry(wheel[0], wheel[1]))
    lock = load_wheel_lock(_write_lock(tmp_path / "lock.json", entries))

    with pytest.raises(WheelLockError):
        verify_wheelhouse(lock, wheelhouse)


@pytest.mark.parametrize("mutation", ("extra", "missing", "tampered"))
def test_install_rejects_extra_missing_or_tampered_files(tmp_path, mutation):
    wheel = _wheel("alpha")
    _lock, verified, _wheelhouse, contents = _selection(tmp_path, [wheel])

    def mutate(root, _verified):
        selected = root / "alpha/__init__.py"
        if mutation == "extra":
            (root / "extra.py").write_bytes(b"extra")
        elif mutation == "missing":
            selected.unlink()
            selected.parent.rmdir()
        else:
            selected.write_bytes(b"tampered")

    _root, receipt = _install(
        tmp_path,
        verified,
        contents,
        mutate=mutate,
    )
    with pytest.raises(WheelLockError):
        verify_installed_wheels(verified, receipt)


@pytest.mark.parametrize(
    "mutation",
    ("extra-row", "missing-row", "wrong-hash", "hashed-record"),
)
def test_install_rejects_record_drift(tmp_path, mutation):
    wheel = _wheel("alpha")
    _lock, verified, _wheelhouse, contents = _selection(tmp_path, [wheel])

    def mutate(root, current):
        record_path = root / current.wheels[0].record_path
        rows = list(
            csv.reader(
                io.StringIO(record_path.read_text(encoding="utf-8"), newline=""),
                strict=True,
            )
        )
        if mutation == "extra-row":
            (root / "extra.py").write_bytes(b"extra")
            rows.append(["extra.py", _record_hash(b"extra"), "5"])
        elif mutation == "missing-row":
            rows = [row for row in rows if row[0] != "alpha/__init__.py"]
        elif mutation == "wrong-hash":
            row = next(row for row in rows if row[0] == "alpha/__init__.py")
            row[1] = _record_hash(b"other")
        else:
            row = next(row for row in rows if row[0].endswith("/RECORD"))
            row[1:] = [_record_hash(b"record"), "6"]
        record_path.write_bytes(_record_bytes(rows))

    _root, receipt = _install(
        tmp_path,
        verified,
        contents,
        mutate=mutate,
    )
    with pytest.raises(WheelLockError):
        verify_installed_wheels(verified, receipt)


@pytest.mark.parametrize(
    "installer",
    (
        InstallerFile("alpha-1.0.dist-info/INSTALLER", b"pip\n"),
        InstallerFile("alpha-1.0.dist-info/REQUESTED", b"x"),
        InstallerFile("alpha-1.0.dist-info/direct_url.json", b"{}"),
        InstallerFile("other-1.0.dist-info/INSTALLER", b"uv"),
    ),
)
def test_installer_allowlist_is_tiny_explicit_and_deterministic(
    tmp_path,
    installer,
):
    wheel = _wheel("alpha")
    _lock, verified, _wheelhouse, contents = _selection(tmp_path, [wheel])
    _root, receipt = _install(tmp_path, verified, contents)

    with pytest.raises(WheelLockError):
        verify_installed_wheels(
            verified,
            receipt,
            installer_files=(installer,),
        )


def test_unlisted_installer_file_and_record_row_are_rejected(tmp_path):
    wheel = _wheel("alpha")
    _lock, verified, _wheelhouse, contents = _selection(tmp_path, [wheel])
    installer = InstallerFile("alpha-1.0.dist-info/INSTALLER", b"uv")
    _root, receipt = _install(
        tmp_path,
        verified,
        contents,
        installer_files=(installer,),
    )

    with pytest.raises(WheelLockError):
        verify_installed_wheels(verified, receipt)


def test_installer_iterable_is_consumed_only_through_fixed_ceiling(tmp_path):
    wheel = _wheel("alpha")
    _lock, verified, _wheelhouse, contents = _selection(tmp_path, [wheel])
    _root, receipt = _install(tmp_path, verified, contents)
    consumed = 0

    def unbounded():
        nonlocal consumed
        while True:
            consumed += 1
            yield InstallerFile(
                f"alpha-1.0.dist-info/INSTALLER-{consumed}",
                b"uv",
            )

    with pytest.raises(WheelLockError):
        verify_installed_wheels(
            verified,
            receipt,
            installer_files=unbounded(),
        )

    assert consumed == wheel_lock_module.MAXIMUM_INSTALLER_FILES + 1


def test_install_receipt_must_be_site_packages_and_current(tmp_path):
    wheel = _wheel("alpha")
    _lock, verified, _wheelhouse, contents = _selection(tmp_path, [wheel])
    root, receipt = _install(tmp_path, verified, contents)
    (root / "alpha/__init__.py").write_bytes(b"changed after receipt")
    with pytest.raises(WheelLockError):
        verify_installed_wheels(verified, receipt)

    other = tmp_path / "runtime"
    other.mkdir(mode=0o700)
    (other / "payload").write_bytes(b"payload")
    other_receipt = generate_runtime_tree_receipt(
        other,
        tmp_path / "other-receipt.json",
        limits=NEMOTRON_RUNTIME_TREE_LIMITS,
    )
    with pytest.raises(WheelLockError):
        verify_installed_wheels(verified, other_receipt)
