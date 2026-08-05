from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.streaming_stt import cgroup_observation
from tools.streaming_stt.cgroup_observation import (
    CgroupObservationError,
    CgroupScopeObserver,
)
from tools.streaming_stt.manifest import PARAKEET_CPP_ADAPTER
from tools.streaming_stt.supervisor import StreamingWorker, WorkerError


def _write_counters(
    scope: Path,
    *,
    current: bytes = b"90\n",
    peak: bytes = b"100\n",
    cpu: bytes = (
        b"nr_periods 4\n"
        b"system_usec 10\n"
        b"usage_usec 30\n"
        b"user_usec 20\n"
        b"core_sched.force_idle_usec 0\n"
    ),
) -> None:
    (scope / "memory.current").write_bytes(current)
    (scope / "memory.peak").write_bytes(peak)
    (scope / "cpu.stat").write_bytes(cpu)


def _observer(tmp_path: Path) -> tuple[CgroupScopeObserver, Path, Path, bytes]:
    proc_path = tmp_path / "proc" / "424242" / "cgroup"
    proc_path.parent.mkdir(parents=True)
    payload = b"0::/user.slice/run-sensitive-scope.scope\n"
    proc_path.write_bytes(payload)
    scope = tmp_path / "cgroup" / "user.slice" / "run-sensitive-scope.scope"
    scope.mkdir(parents=True)
    _write_counters(scope)
    return (
        CgroupScopeObserver.bind(
            scope_path=scope,
            proc_cgroup_path=proc_path,
            expected_proc_cgroup=payload,
        ),
        scope,
        proc_path,
        payload,
    )


def test_pinned_observer_reads_keyed_counters_and_allows_current_to_fall(tmp_path):
    observer, scope, _proc_path, _payload = _observer(tmp_path)
    try:
        first = observer.sample()
        _write_counters(
            scope,
            current=b"80\n",
            peak=b"110\n",
            cpu=(
                b"user_usec 25\nusage_usec 40\nsystem_usec 15\nnew_kernel_counter 7\n"
            ),
        )
        second = observer.sample()
    finally:
        observer.close()

    assert first.as_dict() == {
        "memory_current_bytes": 90,
        "memory_peak_bytes": 100,
        "cpu_usage_usec": 30,
        "cpu_user_usec": 20,
        "cpu_system_usec": 10,
    }
    assert second.memory_current_bytes == 80
    assert second.memory_peak_bytes == 110
    assert second.cpu_usage_usec == 40
    encoded = json.dumps([first.as_dict(), second.as_dict()])
    assert "424242" not in encoded
    assert "sensitive" not in encoded


def test_bounded_reader_consumes_short_reads_instead_of_accepting_a_prefix(
    tmp_path,
    monkeypatch,
):
    observer, _scope, _proc_path, _payload = _observer(tmp_path)
    original = os.read

    def one_byte(descriptor: int, _size: int) -> bytes:
        return original(descriptor, 1)

    monkeypatch.setattr(cgroup_observation.os, "read", one_byte)
    try:
        assert observer.sample().cpu_usage_usec == 30
    finally:
        observer.close()


def test_observer_opens_every_scope_and_counter_descriptor_read_only(
    tmp_path,
    monkeypatch,
):
    original = os.open
    observed_flags: list[int] = []

    def read_only_open(path, flags, *args, **kwargs):
        observed_flags.append(flags)
        assert flags & (os.O_WRONLY | os.O_RDWR) == 0
        return original(path, flags, *args, **kwargs)

    monkeypatch.setattr(cgroup_observation.os, "open", read_only_open)
    observer, _scope, _proc_path, _payload = _observer(tmp_path)
    descriptor = observer._directory_descriptor  # noqa: SLF001 - close proof
    observer.sample()
    observer.close()
    observer.close()

    assert observed_flags
    with pytest.raises(OSError):
        os.fstat(descriptor)


@pytest.mark.parametrize(
    ("filename", "payload"),
    [
        ("memory.current", b"-1\n"),
        ("memory.current", b"1\r\n"),
        ("memory.current", b"9223372036854775808\n"),
        ("memory.current", b"1\n2\n"),
        ("memory.peak", b"max\n"),
        ("memory.peak", b"\xff\n"),
        ("cpu.stat", b"usage_usec 30\nuser_usec 20\n"),
        (
            "cpu.stat",
            b"usage_usec 30\nusage_usec 31\nuser_usec 20\nsystem_usec 10\n",
        ),
        (
            "cpu.stat",
            b"usage_usec 30 extra\nuser_usec 20\nsystem_usec 10\n",
        ),
    ],
)
def test_observer_rejects_malformed_or_incomplete_counters(
    tmp_path,
    filename,
    payload,
):
    observer, scope, _proc_path, _expected = _observer(tmp_path)
    (scope / filename).write_bytes(payload)
    try:
        with pytest.raises(CgroupObservationError, match="cgroup_observation_invalid"):
            observer.sample()
    finally:
        observer.close()


def test_observer_rejects_oversized_cpu_stat(tmp_path):
    observer, scope, _proc_path, _expected = _observer(tmp_path)
    (scope / "cpu.stat").write_bytes(
        b"usage_usec 30\nuser_usec 20\nsystem_usec 10\n" + b"padding_counter 1\n" * 300
    )
    try:
        with pytest.raises(CgroupObservationError, match="cgroup_observation_invalid"):
            observer.sample()
    finally:
        observer.close()


@pytest.mark.parametrize(
    ("second_current", "second_peak", "second_cpu"),
    [
        (b"80\n", b"99\n", b"usage_usec 40\nuser_usec 25\nsystem_usec 15\n"),
        (b"80\n", b"110\n", b"usage_usec 29\nuser_usec 25\nsystem_usec 15\n"),
        (b"80\n", b"110\n", b"usage_usec 40\nuser_usec 19\nsystem_usec 15\n"),
        (b"80\n", b"110\n", b"usage_usec 40\nuser_usec 25\nsystem_usec 9\n"),
    ],
)
def test_observer_rejects_cumulative_counter_regression(
    tmp_path,
    second_current,
    second_peak,
    second_cpu,
):
    observer, scope, _proc_path, _expected = _observer(tmp_path)
    try:
        observer.sample()
        _write_counters(
            scope,
            current=second_current,
            peak=second_peak,
            cpu=second_cpu,
        )
        with pytest.raises(CgroupObservationError, match="cgroup_observation_invalid"):
            observer.sample()
    finally:
        observer.close()


def test_observer_rejects_pid_migration(tmp_path):
    observer, _scope, proc_path, _expected = _observer(tmp_path)
    proc_path.write_bytes(b"0::/user.slice/run-different.scope\n")
    try:
        with pytest.raises(CgroupObservationError, match="cgroup_observation_invalid"):
            observer.sample()
    finally:
        observer.close()


def test_observer_rejects_scope_path_replacement(tmp_path):
    observer, scope, _proc_path, _expected = _observer(tmp_path)
    moved = scope.with_name("moved.scope")
    scope.rename(moved)
    scope.mkdir()
    _write_counters(scope)
    try:
        with pytest.raises(CgroupObservationError, match="cgroup_observation_invalid"):
            observer.sample()
    finally:
        observer.close()


def test_bind_rejects_scope_and_counter_symlinks(tmp_path):
    observer, scope, proc_path, payload = _observer(tmp_path)
    observer.close()
    linked_scope = scope.with_name("linked.scope")
    linked_scope.symlink_to(scope, target_is_directory=True)
    with pytest.raises(CgroupObservationError, match="cgroup_observation_invalid"):
        CgroupScopeObserver.bind(
            scope_path=linked_scope,
            proc_cgroup_path=proc_path,
            expected_proc_cgroup=payload,
        )

    (scope / "memory.current").unlink()
    (scope / "memory.current").symlink_to(scope / "memory.peak")
    observer = CgroupScopeObserver.bind(
        scope_path=scope,
        proc_cgroup_path=proc_path,
        expected_proc_cgroup=payload,
    )
    try:
        with pytest.raises(CgroupObservationError, match="cgroup_observation_invalid"):
            observer.sample()
    finally:
        observer.close()


def test_worker_records_exact_ordered_controller_boundaries(
    tmp_path,
    monkeypatch,
):
    observer, scope, _proc_path, _payload = _observer(tmp_path)
    evidence = {"verified": True}

    class Process:
        pid = 424242

        @staticmethod
        def poll():
            return None

    ticks = iter((10.1, 10.2, 10.3))
    worker = StreamingWorker.__new__(StreamingWorker)
    worker.manifest = SimpleNamespace(adapter=PARAKEET_CPP_ADAPTER)
    worker._process = Process()  # noqa: SLF001 - exact lifecycle seam
    worker._cgroup_evidence = evidence  # noqa: SLF001
    worker._cgroup_observer = observer  # noqa: SLF001
    worker._launch_started_monotonic = 10.0  # noqa: SLF001
    worker._resource_boundaries = []  # noqa: SLF001
    worker._completed_cases = 0  # noqa: SLF001
    worker._monotonic = lambda: next(ticks)  # noqa: SLF001
    monkeypatch.setattr(
        "tools.streaming_stt.supervisor._read_parakeet_cpp_cgroup_evidence",
        lambda _pid: evidence,
    )

    try:
        worker._record_resource_boundary(  # noqa: SLF001
            "startup",
            "validated_ready",
            completed_cases=0,
        )
        _write_counters(
            scope,
            current=b"95\n",
            peak=b"105\n",
            cpu=b"usage_usec 35\nuser_usec 23\nsystem_usec 12\n",
        )
        first_trace = object()
        assert worker._complete_trace(first_trace) is first_trace  # type: ignore[arg-type]  # noqa: SLF001
        first_use = worker._resource_boundaries[-1]  # noqa: SLF001
        _write_counters(
            scope,
            current=b"85\n",
            peak=b"115\n",
            cpu=b"usage_usec 50\nuser_usec 32\nsystem_usec 18\n",
        )
        for _ in range(6):
            worker._complete_trace(object())  # type: ignore[arg-type]  # noqa: SLF001
        assert len(worker._resource_boundaries) == 2  # noqa: SLF001
        assert worker._resource_boundaries[-1] is first_use  # noqa: SLF001
        worker._record_resource_boundary(  # noqa: SLF001
            "resident",
            "pre_shutdown_after_suite",
            completed_cases=7,
        )
        report = worker.resource_observations
    finally:
        observer.close()

    assert report is not None
    assert [item["phase"] for item in report["boundaries"]] == [
        "startup",
        "first_use",
        "resident",
    ]
    assert [item["completed_cases"] for item in report["boundaries"]] == [0, 1, 7]
    assert [item["wall_elapsed_ms"] for item in report["boundaries"]] == [
        100.0,
        200.0,
        300.0,
    ]
    assert report["os_cache_control"] == "uncontrolled"
    assert report["cache_eviction_performed"] is False
    encoded = json.dumps(report).lower()
    assert "cold" not in encoded
    assert "warm" not in encoded


def test_worker_refuses_reordered_or_duplicate_boundaries(tmp_path, monkeypatch):
    observer, _scope, _proc_path, _payload = _observer(tmp_path)
    worker = StreamingWorker.__new__(StreamingWorker)
    worker.manifest = SimpleNamespace(adapter=PARAKEET_CPP_ADAPTER)
    worker._process = SimpleNamespace(pid=1, poll=lambda: None)  # noqa: SLF001
    worker._cgroup_evidence = {"verified": True}  # noqa: SLF001
    worker._cgroup_observer = observer  # noqa: SLF001
    worker._launch_started_monotonic = 1.0  # noqa: SLF001
    worker._resource_boundaries = []  # noqa: SLF001
    worker._completed_cases = 0  # noqa: SLF001
    worker._monotonic = lambda: 1.1  # noqa: SLF001
    monkeypatch.setattr(
        "tools.streaming_stt.supervisor._read_parakeet_cpp_cgroup_evidence",
        lambda _pid: {"verified": True},
    )
    try:
        with pytest.raises(WorkerError, match="worker_prerequisite"):
            worker._record_resource_boundary(  # noqa: SLF001
                "first_use",
                "first_validated_terminal",
                completed_cases=1,
            )
        monkeypatch.setattr(
            "tools.streaming_stt.supervisor._read_parakeet_cpp_cgroup_evidence",
            lambda _pid: {"verified": False},
        )
        with pytest.raises(WorkerError, match="worker_prerequisite"):
            worker._record_resource_boundary(  # noqa: SLF001
                "startup",
                "validated_ready",
                completed_cases=0,
            )
        assert worker._resource_boundaries == []  # noqa: SLF001
    finally:
        observer.close()
