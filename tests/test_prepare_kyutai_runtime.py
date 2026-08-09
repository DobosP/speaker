from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from tools import prepare_kyutai_runtime as prepare
from tools.streaming_stt.runtime_builder import RuntimeBuildError


def _arguments(tmp_path: Path) -> list[str]:
    return [
        "--wheel-lock",
        str(tmp_path / "wheel-lock.json"),
        "--wheelhouse",
        str(tmp_path / "wheelhouse"),
        "--system-python",
        "/usr/bin/python3.12",
        "--output-root",
        str(tmp_path / "runtime"),
        "--receipt",
        str(tmp_path / "runtime-receipt.json"),
    ]


def test_cli_reports_only_aggregate_runtime_receipts(tmp_path, monkeypatch, capsys):
    lock = SimpleNamespace(digest="1" * 64)
    expectation = SimpleNamespace(
        content_digest="2" * 64,
        file_count=17,
        total_size_bytes=23,
        maximum_file_bytes=7,
    )
    receipt = SimpleNamespace(digest="3" * 64)
    prepared = SimpleNamespace(
        wheel_lock_sha256=lock.digest,
        expectation=expectation,
        receipt=receipt,
    )
    monkeypatch.setattr(prepare, "load_wheel_lock", lambda path: lock)
    calls = []

    def prepare_fixture(*args, **kwargs):
        calls.append((args, kwargs))
        return prepared

    monkeypatch.setattr(prepare, "prepare_locked_runtime", prepare_fixture)

    assert prepare.main(_arguments(tmp_path)) == 0

    result = json.loads(capsys.readouterr().out)
    assert result == {
        "ok": True,
        "runtime_content_sha256": "2" * 64,
        "runtime_file_count": 17,
        "runtime_maximum_file_bytes": 7,
        "runtime_receipt_sha256": "3" * 64,
        "runtime_total_size_bytes": 23,
        "wheel_lock_sha256": "1" * 64,
    }
    assert str(tmp_path) not in json.dumps(result)
    assert len(calls) == 1
    assert calls[0][1]["policy"] is prepare._RUNTIME_POLICY


def test_runtime_policy_has_only_the_measured_kyutai_exceptions():
    policy = prepare._RUNTIME_POLICY.wheel_archive_policy

    assert set(policy.external_runtime_exclusions) == {
        (
            "sympy",
            "1.14.0",
            "sympy-1.14.0.data/data/share/man/man1/isympy.1",
        )
    }
    assert set(policy.inert_pth_files) == {
        ("setuptools", "84.0.0", "distutils-precedence.pth")
    }
    assert set(policy.shared_runtime_files) == {"nvidia/__init__.py"}
    assert policy.high_compression_members == {}


def test_cli_failure_is_detail_free(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(prepare, "load_wheel_lock", lambda path: object())

    def reject(*args, **kwargs):
        raise RuntimeBuildError()

    monkeypatch.setattr(prepare, "prepare_locked_runtime", reject)

    assert prepare.main(_arguments(tmp_path)) == 2
    result = json.loads(capsys.readouterr().out)
    assert result == {
        "error": "kyutai_runtime_prerequisites_unavailable",
        "ok": False,
    }
    assert str(tmp_path) not in json.dumps(result)
