"""Control and scope checks for the synthetic reference microbenchmark."""

from __future__ import annotations

import pytest

from tools import reference_ring_benchmark as benchmark


@pytest.mark.parametrize(
    "kwargs",
    [
        {"blocks": True},
        {"blocks": 2},
        {"iterations": 99},
        {"iterations": 20001},
        {"cpu_budget": 0},
        {"cpu_budget": 5},
    ],
)
def test_rejects_unbounded_or_non_integer_budgets(kwargs):
    with pytest.raises(ValueError, match="reference_benchmark_invalid_budget"):
        benchmark.benchmark(**kwargs)


def _install_fake_scope(monkeypatch):
    scopes = []
    monkeypatch.setattr(
        benchmark.os, "sched_getaffinity", lambda _pid: {2, 3, 4}, raising=False
    )
    monkeypatch.setattr(
        benchmark.os,
        "sched_setaffinity",
        lambda _pid, cpus: scopes.append(set(cpus)),
        raising=False,
    )
    monkeypatch.setattr(
        benchmark, "_measure", lambda *_args: {"synthetic_test_measurement": 1}
    )
    return scopes


def test_report_keeps_narrow_scope_and_restores_affinity(monkeypatch):
    scopes = _install_fake_scope(monkeypatch)
    report = benchmark.benchmark(blocks=3, iterations=100)
    assert scopes == [{2, 3}, {2, 3, 4}]
    assert report["cpu_affinity_enforced"] is True
    assert len(report["cells"]) == 3
    assert all(cell["exact_equal"] for cell in report["cells"])
    assert all(
        set(cell["variants"]) == {"gather_v1", "bounded_copy"}
        for cell in report["cells"]
    )
    assert report["claims"]["synthetic_only"] is True
    assert all(
        value is False
        for key, value in report["claims"].items()
        if key != "synthetic_only"
    )


def test_source_change_refuses_result_and_restores_affinity(monkeypatch):
    scopes = _install_fake_scope(monkeypatch)
    hashes = iter([["before"], ["after"]])
    monkeypatch.setattr(benchmark, "_source_hashes", lambda: next(hashes))
    with pytest.raises(ValueError, match="reference_benchmark_source_changed"):
        benchmark.benchmark(blocks=3, iterations=100)
    assert scopes == [{2, 3}, {2, 3, 4}]


def test_failed_measurement_restores_affinity(monkeypatch):
    scopes = _install_fake_scope(monkeypatch)

    def fail(*_args):
        raise RuntimeError("synthetic_test_failure")

    monkeypatch.setattr(benchmark, "_measure", fail)
    with pytest.raises(RuntimeError, match="synthetic_test_failure"):
        benchmark.benchmark(blocks=3, iterations=100)
    assert scopes == [{2, 3}, {2, 3, 4}]
