from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.test_prepare_nemotron_runtime import _selection
from tests.test_streaming_stt_wheel_lock import _entry, _wheel, _write_lock
from tools.prepare_parakeet_runtime import _parser, _runtime_policy, main
from tools.streaming_stt.runtime_builder import RuntimeBuildError
from tools.streaming_stt.wheel_lock import PARAKEET_WHEEL_ARCHIVE_POLICY


def test_parakeet_prepare_uses_caller_version_and_closed_measured_policy(
    tmp_path,
    capsys,
):
    lock, wheelhouse = _selection(tmp_path)
    output = tmp_path / "runtime"
    receipt = tmp_path / "runtime-receipt.json"

    assert (
        main(
            [
                "--wheel-lock",
                str(lock.path),
                "--wheelhouse",
                str(wheelhouse),
                "--system-python",
                "/usr/bin/python3.12",
                "--python-version",
                "3.12.3",
                "--output-root",
                str(output),
                "--receipt",
                str(receipt),
            ]
        )
        == 0
    )

    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is True
    assert result["python_version"] == "3.12.3"
    assert result["wheel_lock_sha256"] == lock.digest
    assert result["runtime_file_count"] > 0
    assert "path" not in result
    assert (
        output / "pyvenv.cfg"
    ).read_text(encoding="utf-8") == (
        "home = /usr/bin\n"
        "include-system-site-packages = false\n"
        "version = 3.12.3\n"
        "executable = /usr/bin/python3.12\n"
    )
    policy = _runtime_policy("3.12.3")
    assert policy.wheel_archive_policy is PARAKEET_WHEEL_ARCHIVE_POLICY
    assert policy.wheel_archive_policy.external_runtime_exclusions
    assert policy.wheel_archive_policy.inert_pth_files
    assert policy.wheel_archive_policy.shared_runtime_files
    assert policy.wheel_archive_policy.high_compression_members


@pytest.mark.parametrize("version", ["3.12", "03.12.3", "3.12.03", "latest"])
def test_parakeet_prepare_rejects_non_exact_python_patch(version):
    with pytest.raises(RuntimeBuildError):
        _runtime_policy(version)


def test_parakeet_prepare_rejects_python_minor_mismatch_without_output(
    tmp_path,
    capsys,
):
    lock, wheelhouse = _selection(tmp_path)
    output = tmp_path / "runtime"

    assert (
        main(
            [
                "--wheel-lock",
                str(lock.path),
                "--wheelhouse",
                str(wheelhouse),
                "--system-python",
                "/usr/bin/python3.12",
                "--python-version",
                "3.11.9",
                "--output-root",
                str(output),
                "--receipt",
                str(tmp_path / "receipt.json"),
            ]
        )
        == 2
    )
    result = json.loads(capsys.readouterr().out)
    assert result == {
        "error": "parakeet_runtime_prerequisites_unavailable",
        "ok": False,
    }
    assert str(tmp_path) not in json.dumps(result)
    assert not output.exists()


def test_parakeet_prepare_rejects_unmeasured_pth_before_output(tmp_path, capsys):
    wheel = _wheel(
        "candidate_hooks",
        package_path="candidate-hooks.pth",
        package_bytes=b"import candidate_hooks\n",
    )
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir(mode=0o700)
    filename, data, _contents = wheel
    (wheelhouse / filename).write_bytes(data)
    lock_path = _write_lock(tmp_path / "wheel-lock.json", [_entry(filename, data)])
    output = tmp_path / "runtime"

    assert (
        main(
            [
                "--wheel-lock",
                str(lock_path),
                "--wheelhouse",
                str(wheelhouse),
                "--system-python",
                "/usr/bin/python3.12",
                "--python-version",
                "3.12.3",
                "--output-root",
                str(output),
                "--receipt",
                str(tmp_path / "receipt.json"),
            ]
        )
        == 2
    )
    assert json.loads(capsys.readouterr().out) == {
        "error": "parakeet_runtime_prerequisites_unavailable",
        "ok": False,
    }
    assert not output.exists()


def test_parakeet_prepare_help_promises_no_dependency_side_effects():
    help_text = " ".join(_parser().format_help().split())

    assert "No resolver, package installer, download, or candidate import" in help_text


def test_parakeet_prepare_source_does_not_freeze_unmeasured_candidate_versions():
    source = Path("tools/prepare_parakeet_runtime.py").read_text(encoding="utf-8")

    assert "nemo-toolkit==" not in source
    assert "parakeet_realtime_eou" not in source
