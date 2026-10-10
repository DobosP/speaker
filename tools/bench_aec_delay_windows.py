"""Synthetic storage-only AEC history benchmark against pinned project baseline.

No model, microphone, device, private data or network is used. Results are scoped
maintenance latency and Python-traced allocations, not process RSS or audibility.
"""

from __future__ import annotations

import argparse
import ast
from collections import deque
from dataclasses import asdict
from hashlib import sha256
import inspect
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import tracemalloc

import numpy as np

from core.engines._aec import AecDelayCalibrator as Current, AecDelaySnapshot


def run_benchmark() -> dict:
    REPO = Path(__file__).resolve().parents[1]
    BASE_COMMIT = "70345ae309b7fd84c27d520d0bdaa972e112a2bd"
    module_source = subprocess.check_output(
        ["git", "show", f"{BASE_COMMIT}:core/engines/_aec.py"], cwd=REPO, text=True
    )
    tree = ast.parse(module_source)
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "AecDelayCalibrator"
    )
    source = ast.get_source_segment(module_source, node)
    namespace = {"np": np, "deque": deque, "AecDelaySnapshot": AecDelaySnapshot}
    exec(compile(source, "<pinned-calibrator>", "exec"), namespace)
    Baseline = namespace["AecDelayCalibrator"]

    current_source = inspect.getsource(Current)
    current_node = ast.parse(current_source).body[0]
    for method in (
        "_clamp",
        "_estimate_delay",
        "_recalc_now",
        "snapshot",
        "current_delay_samples",
    ):
        original = next(
            x for x in node.body if isinstance(x, ast.FunctionDef) and x.name == method
        )
        revised = next(
            x
            for x in current_node.body
            if isinstance(x, ast.FunctionDef) and x.name == method
        )
        assert ast.dump(original, include_attributes=False) == ast.dump(
            revised, include_attributes=False
        )
    Ring = Current

    rng = np.random.default_rng(421)
    window_cases = 0
    for window in (1, 7, 1600, 24000):
        before = Baseline(16000, 80, window_ms=window / 16, recalc_interval_ms=1e9)
        after = Ring(16000, 80, window_ms=window / 16, recalc_interval_ms=1e9)
        for n in (0, 1, 3, 1599, 1600, window - 1, window, window + 1, 2 * window + 7):
            n = max(0, n)
            mic = rng.standard_normal(n + 3).astype(np.float32)
            far = rng.standard_normal(n).astype(np.float32)
            before.observe(mic, far)
            after.observe(mic, far)
            assert before._mic.tobytes() == after._mic.tobytes()
            assert before._far.tobytes() == after._far.tobytes()
            assert before._since == after._since
            window_cases += 1
        before.reset_continuity()
        after.reset_continuity()
        assert before._mic.size == after._mic.size == 0
        assert before._far.size == after._far.size == 0
        assert before.snapshot() == after.snapshot()

    # The existing estimator sees byte-identical windows and returns exact same lag,
    # gate, operating delay, median and reset outcomes through silence/echo/drift.
    far = rng.normal(0, 0.08, 16000 * 4).astype(np.float32)
    mic = np.concatenate((np.zeros(384, np.float32), far[:-384]))
    mic += rng.normal(0, 0.001, mic.size).astype(np.float32)
    before = Baseline(16000, 80)
    after = Ring(16000, 80)
    for start in range(0, len(far), 1600):
        a, b = mic[start : start + 1600], far[start : start + 1600]
        before.observe(a, b)
        after.observe(a, b)
        assert before._mic.tobytes() == after._mic.tobytes()
        assert before._far.tobytes() == after._far.tobytes()
        assert before.snapshot() == after.snapshot()
        assert list(before._median) == list(after._median)
    assert before.current_delay_samples() == 384
    snapshot = asdict(before.snapshot())
    before.reset_continuity()
    after.reset_continuity()
    assert before.snapshot() == after.snapshot()
    before.reset()
    after.reset()
    assert before.snapshot() == after.snapshot()

    block = np.zeros(1600, np.float32)

    def ready(cls):
        obj = cls(16000, 80)
        for _ in range(30):
            obj.observe(block, block)
        return obj

    def timed_pair():
        objects = (ready(Baseline), ready(Ring))
        timings = ([], [])
        for iteration in range(4000):
            # Adjacent, alternating observations reduce CPU/load drift and
            # balance any benefit from being the second call in a pair.
            for index in (0, 1) if iteration % 2 == 0 else (1, 0):
                at = time.perf_counter_ns()
                objects[index].observe(block, block)
                timings[index].append(time.perf_counter_ns() - at)
        return tuple(
            {
                "p50_us": statistics.median(values) / 1000,
                "p95_us": sorted(values)[int(0.95 * len(values))] / 1000,
            }
            for values in timings
        )

    def allocated(cls):
        obj = ready(cls)
        tracemalloc.start()
        tracemalloc.reset_peak()
        for _ in range(200):
            obj.observe(block, block)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        if cls is Ring:
            retained = obj._mic_buffer.nbytes + obj._far_buffer.nbytes
        else:
            retained = sum(
                (x.base if isinstance(x.base, np.ndarray) else x).nbytes
                for x in (obj._mic, obj._far)
            )
        return {
            "incremental_traced_current_bytes": current,
            "incremental_traced_peak_bytes": peak,
            "retained_window_storage_bytes": retained,
        }

    def complete_peak(cls, energetic):
        # Include persistent ring/window allocations, unlike the separate
        # post-warm incremental-allocation experiment.
        tracemalloc.start()
        obj = ready(cls)
        tracemalloc.reset_peak()
        if energetic:
            for start in range(0, 6400, 1600):
                obj.observe(mic[start : start + 1600], far[start : start + 1600])
        else:
            for _ in range(200):
                obj.observe(block, block)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return {"traced_current_bytes": current, "traced_peak_bytes": peak}

    report = {
        "schema": 1,
        "baseline_source_commit": BASE_COMMIT,
        "baseline_class_sha256": sha256(source.encode()).hexdigest(),
        "current_class_sha256": sha256(current_source.encode()).hexdigest(),
        "synthetic_only": True,
        "native_model_or_device": False,
        "thread_environment_requested_one": all(
            os.environ.get(name) == "1"
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        ),
        "window_exact_cases": window_cases,
        "estimator_ast_unchanged": True,
        "synthetic_delay_snapshot": snapshot,
        "idle_100ms_observe": {},
    }
    for _round in range(2):
        for name, values in zip(("baseline", "ring"), timed_pair()):
            report["idle_100ms_observe"].setdefault(name, []).append(values)
    report["timing_method"] = "adjacent_alternating_pairs_unpaced"
    report["idle_incremental_allocation"] = {
        "baseline": allocated(Baseline),
        "ring": allocated(Ring),
    }
    report["complete_storage_and_working_peak"] = {
        "idle": {
            "baseline": complete_peak(Baseline, False),
            "ring": complete_peak(Ring, False),
        },
        "energetic_recalculation": {
            "baseline": complete_peak(Baseline, True),
            "ring": complete_peak(Ring, True),
        },
    }
    report["rolling_copy_bytes_full_idle_block"] = {
        "baseline": 2 * (24000 + 1600) * 4,
        "ring": 2 * 1600 * 4,
    }
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if any(
        os.environ.get(name) != "1"
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    ):
        print(
            "aec_window_benchmark_refused:thread_environment_required", file=sys.stderr
        )
        return 2
    try:
        report = run_benchmark()
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(report, output, indent=2)
            output.write("\n")
    except Exception:
        print("aec_window_benchmark_refused:benchmark_failed", file=sys.stderr)
        return 2
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
