"""Tests for compute telemetry (core.sysinfo).

Inject fake samplers / nvidia-smi runners -- no psutil, no GPU needed.
"""
from __future__ import annotations

from core.sysinfo import SystemMonitor, _deltas, _gpu_snapshot, snapshot


def test_snapshot_never_raises_and_has_timestamp():
    s = snapshot()  # psutil/nvidia-smi may be absent here; must degrade
    assert "t" in s
    assert "gpu" in s  # None when no nvidia-smi


def test_gpu_snapshot_parses_csv():
    class R:
        returncode = 0
        stdout = "37, 8100, 24576, 61\n"

    gpus = _gpu_snapshot(run=lambda *a, **k: R())
    assert gpus == [{"util_percent": 37.0, "mem_used_mb": 8100.0, "mem_total_mb": 24576.0, "temp_c": 61.0}]


def test_gpu_snapshot_none_when_no_smi():
    def boom(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    assert _gpu_snapshot(run=boom) is None


def test_deltas_computes_growth():
    before = {"ram_used_mb": 1000.0, "proc_rss_mb": 100.0, "gpu": [{"mem_used_mb": 500.0}]}
    after = {"ram_used_mb": 1500.0, "proc_rss_mb": 300.0, "gpu": [{"mem_used_mb": 9000.0}]}
    d = _deltas(before, after)
    assert d["ram_used_mb_delta"] == 500.0
    assert d["proc_rss_mb_delta"] == 200.0
    assert d["gpu_mem_used_mb_delta"] == 8500.0


def test_monitor_baseline_peak_final_and_summary():
    seq = [
        {"t": 1, "ram_used_mb": 1000.0, "proc_rss_mb": 100.0,
         "gpu": [{"util_percent": 10.0, "mem_used_mb": 500.0, "mem_total_mb": 24000.0}]},
        {"t": 2, "ram_used_mb": 1500.0, "proc_rss_mb": 300.0,
         "gpu": [{"util_percent": 80.0, "mem_used_mb": 9000.0, "mem_total_mb": 24000.0}]},
    ]
    it = iter(seq)

    class Summary:
        system = None

    summ = Summary()
    # Huge interval so the background thread never samples during the test.
    mon = SystemMonitor(summ, interval=10_000, sampler=lambda: next(it))
    mon.start()  # consumes seq[0] as baseline
    block = mon.stop()  # consumes seq[1] as final

    assert block["baseline"]["ram_used_mb"] == 1000.0
    assert block["final"]["ram_used_mb"] == 1500.0
    assert block["peak"]["ram_used_mb"] == 1500.0
    assert block["peak"]["proc_rss_mb"] == 300.0
    assert block["peak"]["gpu_mem_used_mb"] == 9000.0
    assert block["deltas"]["ram_used_mb_delta"] == 500.0
    assert block["deltas"]["gpu_mem_used_mb_delta"] == 8500.0
    assert summ.system is block  # folded into the run summary


def test_monitor_mark_records_named_snapshot():
    seq = [{"t": 1, "ram_used_mb": 100.0}, {"t": 2, "ram_used_mb": 200.0}, {"t": 3, "ram_used_mb": 150.0}]
    it = iter(seq)
    mon = SystemMonitor(interval=10_000, sampler=lambda: next(it))
    mon.start()  # baseline = seq[0]
    mon.mark("after_build")  # seq[1]
    block = mon.stop()  # final = seq[2]
    assert block["marks"]["after_build"]["ram_used_mb"] == 200.0
    assert block["peak"]["ram_used_mb"] == 200.0  # mark counted toward peak
