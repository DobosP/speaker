from __future__ import annotations

import json
import os
from pathlib import Path
import stat

import pytest

from core.metrics import ASR_FINAL, LLM_FIRST_TOKEN, TTS_FIRST_AUDIO, TurnRecord
from tools.bench import report, runner


def _run() -> runner.ProductionRun:
    sample = runner.TurnSample(
        name="private-case",
        expectation="transcript",
        transcript="recognized private words",
        responded=True,
        record=TurnRecord(
            stamps={ASR_FINAL: 1.0, LLM_FIRST_TOKEN: 1.1, TTS_FIRST_AUDIO: 1.2}
        ),
        reference="reference private words",
        assertion="transcript",
        replay_decode_seconds=0.2,
        replay_to_idle_seconds=0.4,
    )
    return runner.ProductionRun(
        samples=(sample,),
        device_profile="desktop_gpu_4090",
        final_stt_profile="sense-voice",
        final_stt_profile_sha256="1" * 64,
        final_stt_profile_schema_version=1,
        production_config_sha256="2" * 64,
        execution_config_sha256="3" * 64,
        replay_safety_schema_version=1,
        replay_safety_sha256="4" * 64,
        active_artifacts_sha256="5" * 64,
        active_artifact_roots=5,
        active_artifact_files=8,
        active_artifact_bytes=1024,
        source_revision="6" * 40,
        runtime_identities={"python": {"executable_sha256": "7" * 64}},
        ollama_identity_sha256="8" * 64,
        ollama_role_contracts={},
    )


def _files() -> dict[str, bytes]:
    return {
        "summary.json": b'{"safe":true}\n',
        "index.html": b"<!doctype html><p>safe aggregate</p>",
    }


def _staging_entries(parent: Path) -> list[Path]:
    return sorted(parent.glob(".speaker-production-report-*.tmp"))


def test_publication_commits_complete_private_directory_and_fsyncs(monkeypatch, tmp_path):
    destination = tmp_path / "report"
    original_rename = report._rename_directory_noreplace
    original_fsync = report.os.fsync
    fsync_kinds: list[str] = []
    commit_observations: list[tuple[bool, set[str]]] = []

    def observed_fsync(descriptor: int) -> None:
        metadata = os.fstat(descriptor)
        fsync_kinds.append("directory" if stat.S_ISDIR(metadata.st_mode) else "file")
        original_fsync(descriptor)

    def observed_rename(parent_fd: int, source: str, target: str) -> None:
        with pytest.raises(FileNotFoundError):
            os.stat(target, dir_fd=parent_fd, follow_symlinks=False)
        stage_fd = os.open(
            source,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        try:
            commit_observations.append((True, set(os.listdir(stage_fd))))
        finally:
            os.close(stage_fd)
        original_rename(parent_fd, source, target)

    monkeypatch.setattr(report.os, "fsync", observed_fsync)
    monkeypatch.setattr(report, "_rename_directory_noreplace", observed_rename)

    index, payload = report.write_production_reports(
        destination,
        _run(),
        {"schema_version": 2, "sha256": "9" * 64, "case_count": 1},
        {"samples": 1, "peak": {}},
        sample_interval_sec=1.0,
    )

    assert index == destination / "index.html"
    assert json.loads((destination / "summary.json").read_text()) == payload
    assert commit_observations == [(True, {"summary.json", "index.html"})]
    assert fsync_kinds.count("file") >= 2
    assert fsync_kinds.count("directory") >= 2
    assert stat.S_IMODE(destination.stat().st_mode) == 0o700
    assert all(
        stat.S_IMODE((destination / name).stat().st_mode) == 0o600
        for name in ("summary.json", "index.html")
    )
    assert not _staging_entries(tmp_path)


def test_destination_race_is_no_overwrite_and_cleans_only_owned_stage(
    monkeypatch, tmp_path
):
    destination = tmp_path / "report"
    original_rename = report._rename_directory_noreplace

    def install_foreign_then_commit(parent_fd: int, source: str, target: str) -> None:
        os.mkdir(target, mode=0o700, dir_fd=parent_fd)
        foreign_fd = os.open(
            target,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        try:
            marker = os.open(
                "foreign-marker",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=foreign_fd,
            )
            os.close(marker)
        finally:
            os.close(foreign_fd)
        original_rename(parent_fd, source, target)

    monkeypatch.setattr(
        report,
        "_rename_directory_noreplace",
        install_foreign_then_commit,
    )

    with pytest.raises(FileExistsError):
        report._publish_private_report_directory(destination, _files())

    assert (destination / "foreign-marker").is_file()
    assert not (destination / "summary.json").exists()
    assert not _staging_entries(tmp_path)


def test_failed_staging_write_is_detail_free_and_never_exposes_final_leaf(
    monkeypatch, tmp_path
):
    destination = tmp_path / "report"
    original_write = report._write_private_file_at

    def fail_second(descriptor: int, name: str, payload: bytes):
        if name == "index.html":
            raise OSError("PRIVATE_PATH_AND_TRANSCRIPT_CANARY")
        return original_write(descriptor, name, payload)

    monkeypatch.setattr(report, "_write_private_file_at", fail_second)

    with pytest.raises(OSError) as failure:
        report._publish_private_report_directory(destination, _files())

    assert str(failure.value) == ""
    assert not destination.exists()
    assert not _staging_entries(tmp_path)


def test_partial_file_write_failure_removes_file_and_staging(monkeypatch, tmp_path):
    destination = tmp_path / "report"
    original_write = report.os.write
    writes = 0

    def write_once_then_fail(descriptor: int, payload) -> int:
        nonlocal writes
        writes += 1
        if writes == 1:
            return original_write(descriptor, payload[:1])
        raise OSError("PRIVATE_PARTIAL_WRITE_CANARY")

    monkeypatch.setattr(report.os, "write", write_once_then_fail)

    with pytest.raises(OSError) as failure:
        report._publish_private_report_directory(destination, _files())

    assert str(failure.value) == ""
    assert not destination.exists()
    assert not _staging_entries(tmp_path)


def test_parent_fsync_failure_after_rename_removes_owned_final(monkeypatch, tmp_path):
    destination = tmp_path / "report"
    original_fsync = report.os.fsync
    directory_calls = 0
    failed = False

    def fail_commit_fsync_once(descriptor: int) -> None:
        nonlocal directory_calls, failed
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            directory_calls += 1
            if directory_calls == 2 and not failed:
                failed = True
                raise OSError("PRIVATE_FSYNC_CANARY")
        original_fsync(descriptor)

    monkeypatch.setattr(report.os, "fsync", fail_commit_fsync_once)

    with pytest.raises(OSError) as failure:
        report._publish_private_report_directory(destination, _files())

    assert str(failure.value) == ""
    assert failed is True
    assert not destination.exists()
    assert not _staging_entries(tmp_path)


def test_persistent_directory_fsync_failure_still_removes_owned_final(
    monkeypatch, tmp_path
):
    destination = tmp_path / "report"
    original_fsync = report.os.fsync
    directory_calls = 0

    def fail_from_commit_onward(descriptor: int) -> None:
        nonlocal directory_calls
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            directory_calls += 1
            if directory_calls >= 2:
                raise OSError("PRIVATE_PERSISTENT_FSYNC_CANARY")
        original_fsync(descriptor)

    monkeypatch.setattr(report.os, "fsync", fail_from_commit_onward)

    with pytest.raises(OSError) as failure:
        report._publish_private_report_directory(destination, _files())

    assert str(failure.value) == ""
    assert directory_calls >= 3
    assert not destination.exists()
    assert not _staging_entries(tmp_path)


def test_replaced_parent_chain_fails_without_following_symlink(monkeypatch, tmp_path):
    parent = tmp_path / "reports"
    parent.mkdir(mode=0o700)
    destination = parent / "report"
    moved = tmp_path / "moved-reports"
    foreign = tmp_path / "foreign"
    foreign.mkdir(mode=0o700)
    original_open_parent = report._open_report_parent
    calls = 0

    def replace_before_reopen(path: Path, *, create: bool):
        nonlocal calls
        calls += 1
        if calls == 2:
            parent.rename(moved)
            parent.symlink_to(foreign, target_is_directory=True)
        return original_open_parent(path, create=create)

    monkeypatch.setattr(report, "_open_report_parent", replace_before_reopen)

    with pytest.raises(OSError) as failure:
        report._publish_private_report_directory(destination, _files())

    assert str(failure.value) == ""
    assert not (moved / "report").exists()
    assert not (foreign / "report").exists()
    assert not _staging_entries(moved)


def test_preexisting_staging_collision_is_never_removed(monkeypatch, tmp_path):
    first_token = "a" * 24
    second_token = "b" * 24
    foreign_stage = (
        tmp_path / f".speaker-production-report-{os.getpid()}-{first_token}.tmp"
    )
    foreign_stage.mkdir(mode=0o700)
    (foreign_stage / "foreign-marker").write_bytes(b"keep")
    tokens = iter((first_token, second_token))
    monkeypatch.setattr(report.secrets, "token_hex", lambda _size: next(tokens))
    monkeypatch.setattr(
        report,
        "_write_private_file_at",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("PRIVATE_CANARY")),
    )

    with pytest.raises(OSError):
        report._publish_private_report_directory(tmp_path / "report", _files())

    assert (foreign_stage / "foreign-marker").read_bytes() == b"keep"
    assert _staging_entries(tmp_path) == [foreign_stage]
    assert not (tmp_path / "report").exists()
