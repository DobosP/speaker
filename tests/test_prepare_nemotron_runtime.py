from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile

import pytest

from tools.prepare_nemotron_runtime import main
from tools.streaming_stt.runtime_builder import (
    RuntimeBuildError,
    prepare_locked_runtime,
)
from tools.streaming_stt.wheel_lock import load_wheel_lock


def _record_hash(data: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return f"sha256={encoded.decode('ascii')}"


def _wheel() -> tuple[str, bytes]:
    filename = "alpha-1.0-py3-none-any.whl"
    dist_info = "alpha-1.0.dist-info"
    contents = {
        "alpha-1.0.data/purelib/alpha/__init__.py": b"VALUE = 7\n",
        f"{dist_info}/METADATA": (
            b"Metadata-Version: 2.1\nName: alpha\nVersion: 1.0\n"
        ),
        f"{dist_info}/WHEEL": (
            b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        ),
        f"{dist_info}/entry_points.txt": b"[console_scripts]\nalpha=alpha:main\n",
    }
    rows = [
        [path, _record_hash(data), str(len(data))] for path, data in contents.items()
    ]
    record_path = f"{dist_info}/RECORD"
    rows.append([record_path, "", ""])
    output = io.StringIO(newline="")
    csv.writer(output, lineterminator="\n").writerows(rows)
    contents[record_path] = output.getvalue().encode("utf-8")
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, mode="w", compression=zipfile.ZIP_DEFLATED) as wheel:
        for path, data in contents.items():
            wheel.writestr(path, data)
    return filename, archive.getvalue()


def _selection(tmp_path: Path):
    filename, data = _wheel()
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir(mode=0o700)
    selected = wheelhouse / filename
    selected.write_bytes(data)
    selected.chmod(0o600)
    lock_payload = {
        "schema_version": 1,
        "wheels": [
            {
                "distribution": "alpha",
                "version": "1.0",
                "filename": filename,
                "wheel_tag": "py3-none-any",
                "source_url": f"https://files.example.invalid/{filename}",
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        ],
    }
    lock_path = tmp_path / "wheel-lock.json"
    lock_path.write_text(
        json.dumps(
            lock_payload,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    lock_path.chmod(0o600)
    return load_wheel_lock(lock_path), wheelhouse


def test_builder_creates_only_minimal_script_free_verified_runtime(tmp_path):
    lock, wheelhouse = _selection(tmp_path)
    output = tmp_path / "venv"
    receipt = tmp_path / "runtime-receipt.json"

    prepared = prepare_locked_runtime(
        lock,
        wheelhouse,
        system_python=Path("/usr/bin/python3.12"),
        output_root=output,
        receipt_path=receipt,
    )

    assert prepared.expectation.content_digest == prepared.receipt.content_digest
    assert prepared.expectation.files == prepared.receipt.files
    assert {path.name for path in output.iterdir()} == {
        "bin",
        "lib",
        "pyvenv.cfg",
    }
    assert {path.name for path in (output / "bin").iterdir()} == {"python"}
    assert not (output / "bin" / "alpha").exists()
    assert not list(prepared.site_packages.rglob("INSTALLER"))
    assert not list(prepared.site_packages.rglob("REQUESTED"))
    assert (prepared.site_packages / "alpha" / "__init__.py").is_file()
    assert not (prepared.site_packages / "alpha-1.0.data").exists()
    completed = subprocess.run(
        [
            str(prepared.python),
            "-I",
            "-S",
            "-B",
            "-c",
            (
                "import sys;"
                f"sys.path.insert(0,{str(prepared.site_packages)!r});"
                "import alpha;"
                "print(alpha.VALUE)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 0
    assert completed.stdout == "7\n"
    assert completed.stderr == ""


def test_builder_rejects_existing_or_nested_output(tmp_path):
    lock, wheelhouse = _selection(tmp_path)
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(RuntimeBuildError):
        prepare_locked_runtime(
            lock,
            wheelhouse,
            system_python=Path("/usr/bin/python3.12"),
            output_root=existing,
            receipt_path=tmp_path / "receipt.json",
        )
    with pytest.raises(RuntimeBuildError):
        prepare_locked_runtime(
            lock,
            wheelhouse,
            system_python=Path("/usr/bin/python3.12"),
            output_root=wheelhouse / "nested",
            receipt_path=tmp_path / "receipt-2.json",
        )
    with pytest.raises(RuntimeBuildError):
        prepare_locked_runtime(
            lock,
            wheelhouse,
            system_python=Path("/usr/bin/python3.12"),
            output_root=tmp_path / "venv-safe",
            receipt_path=wheelhouse / "receipt.json",
        )


def test_cli_returns_only_safe_aggregate_receipts(tmp_path, capsys):
    lock, wheelhouse = _selection(tmp_path)

    assert (
        main(
            [
                "--wheel-lock",
                str(lock.path),
                "--wheelhouse",
                str(wheelhouse),
                "--system-python",
                "/usr/bin/python3.12",
                "--output-root",
                str(tmp_path / "venv"),
                "--receipt",
                str(tmp_path / "receipt.json"),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is True
    assert result["wheel_lock_sha256"] == lock.digest
    assert result["runtime_file_count"] > 0
    assert "path" not in result


def test_cli_failure_is_path_free_and_does_not_mutate_wheelhouse(tmp_path, capsys):
    lock, wheelhouse = _selection(tmp_path)
    before = {
        path.relative_to(wheelhouse): path.read_bytes() for path in wheelhouse.iterdir()
    }

    assert (
        main(
            [
                "--wheel-lock",
                str(lock.path),
                "--wheelhouse",
                str(wheelhouse),
                "--system-python",
                "/usr/bin/python3.12",
                "--output-root",
                str(tmp_path / "venv"),
                "--receipt",
                str(wheelhouse / "receipt.json"),
            ]
        )
        == 2
    )

    result = json.loads(capsys.readouterr().out)
    assert result == {
        "error": "nemotron_runtime_prerequisites_unavailable",
        "ok": False,
    }
    assert str(tmp_path) not in json.dumps(result)
    assert {
        path.relative_to(wheelhouse): path.read_bytes() for path in wheelhouse.iterdir()
    } == before
    assert not (tmp_path / "venv").exists()
