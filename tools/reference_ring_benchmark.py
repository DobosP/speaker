"""Synthetic paired-reference read microbenchmark; no audio devices or models.

Compare the pre-ADR0230 NumPy gather reader with the production bounded-copy
reader on identical retained samples. Times are distributions of block means,
not per-callback tail latency. Traced peaks are Python/NumPy allocations, not RSS.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import marshal
import os
from pathlib import Path
import statistics
import time
import tracemalloc

import numpy as np

from core.engines._aec import FarEndRing


class _GatherReader(FarEndRing):
    """The previous read algorithm, sharing production push and lock behavior."""

    def read_windows(self, n: int, delays) -> tuple[np.ndarray, ...]:
        n = int(n)
        normalized_delays = tuple(max(0, int(delay)) for delay in delays)
        if n <= 0:
            return tuple(np.zeros(0, dtype=np.float32) for _delay in normalized_delays)
        outputs = tuple(np.zeros(n, dtype=np.float32) for _delay in normalized_delays)
        with self._lock:
            written = self._written
            avail_lo = max(0, written - self._cap)
            for delay, out in zip(normalized_delays, outputs):
                hi = written - delay
                lo = hi - n
                if hi <= 0:
                    continue
                idx = np.arange(lo, hi)
                valid = (idx >= avail_lo) & (idx < written)
                if valid.any():
                    out[valid] = self._buf[idx[valid] % self._cap]
        return outputs


_CASES = (
    ("paired_10ms", 160, (0, 6400)),
    ("paired_100ms_wrapped", 1600, (0, 6400)),
    ("paired_100ms_partial_eviction", 1600, (0, 31000)),
)
_SOURCE_FILES = (
    Path(__file__).resolve(),
    Path(__file__).resolve().parents[1] / "core/engines/_aec.py",
)


def _source_hashes() -> list[str]:
    hashes = []
    for source in _SOURCE_FILES:
        with source.open("rb") as stream:
            data = stream.read(2 * 1024 * 1024 + 1)
        if len(data) > 2 * 1024 * 1024:
            raise ValueError("reference_benchmark_source_too_large")
        hashes.append(hashlib.sha256(data).hexdigest())
    return hashes


def _code_hash(function) -> str:
    return hashlib.sha256(marshal.dumps(function.__code__)).hexdigest()


def _prepare(kind: type[FarEndRing]) -> FarEndRing:
    ring = kind(32000)
    # Nonzero head and a wrap; these are public synthetic timeline indices.
    signal = np.arange(1, 32328, dtype=np.float32)
    for offset in range(0, len(signal), 257):
        ring.push(signal[offset : offset + 257])
    return ring


def _measure(
    ring: FarEndRing, length: int, delays: tuple[int, ...], blocks: int, iterations: int
) -> dict:
    for _ in range(100):
        ring.read_windows(length, delays)
    wall_means = []
    cpu_means = []
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(blocks):
            cpu_start = time.process_time_ns()
            wall_start = time.perf_counter_ns()
            for _ in range(iterations):
                ring.read_windows(length, delays)
            wall_means.append((time.perf_counter_ns() - wall_start) / iterations)
            cpu_means.append((time.process_time_ns() - cpu_start) / iterations)
    finally:
        if was_enabled:
            gc.enable()
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        for _ in range(10):
            ring.read_windows(length, delays)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return {
        "wall_ns_per_read_block_mean_median": statistics.median(wall_means),
        "wall_ns_per_read_block_mean_max": max(wall_means),
        "cpu_ns_per_read_block_mean_median": statistics.median(cpu_means),
        "traced_peak_bytes_ten_reads": peak,
    }


def benchmark(*, blocks: int = 7, iterations: int = 5000, cpu_budget: int = 2) -> dict:
    for value, low, high in (
        (blocks, 3, 15),
        (iterations, 100, 20000),
        (cpu_budget, 1, 4),
    ):
        if type(value) is not int or not low <= value <= high:
            raise ValueError("reference_benchmark_invalid_budget")
    source_hashes = _source_hashes()
    original_affinity = None
    affinity_enforced = False
    if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
        original_affinity = os.sched_getaffinity(0)
        os.sched_setaffinity(0, sorted(original_affinity)[:cpu_budget])
        affinity_enforced = True
    cells = []
    try:
        for case, length, delays in _CASES:
            baseline = _prepare(_GatherReader)
            current = _prepare(FarEndRing)
            expected = baseline.read_windows(length, delays)
            actual = current.read_windows(length, delays)
            if any(
                not np.array_equal(left, right) for left, right in zip(expected, actual)
            ):
                raise ValueError("reference_benchmark_output_mismatch")
            variants = {}
            # Alternate order by case to avoid always favoring the second path.
            entries = [("gather_v1", baseline), ("bounded_copy", current)]
            if len(cells) % 2:
                entries.reverse()
            for label, ring in entries:
                variants[label] = _measure(ring, length, delays, blocks, iterations)
            cells.append(
                {
                    "case": case,
                    "samples_per_window": length,
                    "windows_per_read": len(delays),
                    "exact_equal": True,
                    "variants": variants,
                }
            )
        if _source_hashes() != source_hashes:
            raise ValueError("reference_benchmark_source_changed")
    finally:
        if original_affinity is not None:
            os.sched_setaffinity(0, original_affinity)
    return {
        "schema_version": 1,
        "kind": "synthetic-reference-ring-microbenchmark",
        "numpy_version": np.__version__,
        "source_sha256": source_hashes,
        "loaded_reader_code_sha256": {
            "gather_v1": _code_hash(_GatherReader.read_windows),
            "bounded_copy": _code_hash(FarEndRing.read_windows),
        },
        "capacity_samples": 32000,
        "sample_rate_hz": 16000,
        "blocks": blocks,
        "iterations_per_block": iterations,
        "cpu_budget_requested": cpu_budget,
        "cpu_affinity_enforced": affinity_enforced,
        "claims": {
            "synthetic_only": True,
            "audio_device_opened": False,
            "model_loaded": False,
            "live_quality_validated": False,
            "phone_validated": False,
            "lock_hold_tail_latency_measured": False,
            "whole_assistant_performance_measured": False,
        },
        "cells": cells,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--cpu-budget", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        report = benchmark(
            blocks=args.blocks, iterations=args.iterations, cpu_budget=args.cpu_budget
        )
    except Exception:
        print(
            json.dumps(
                {"ok": False, "error": "reference_benchmark_failed"}, sort_keys=True
            )
        )
        return 2
    print(json.dumps(report, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
