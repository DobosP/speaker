"""Compare repeated watchdog scans against a pinned local Git baseline.

Synthetic metrics only; no inference, microphone, recordings or background daemon.
Run from a speaker checkout: python -m tools.bench_watchdog_history --baseline-ref 1bed9e2
The optional output is a compact reproducible JSON result, never raw user data.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import tracemalloc
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
FILES = ("core/metrics.py", "core/watchdog.py")


def load_pair(name, ref=None):
    package = ModuleType(name)
    package.__path__ = []
    sys.modules[name] = package
    modules, hashes = [], {}
    for path in FILES:
        source = (
            subprocess.check_output(["git", "show", f"{ref}:{path}"], cwd=ROOT)
            if ref
            else (ROOT / path).read_bytes()
        )
        hashes[path] = hashlib.sha256(source).hexdigest()
        key = name + "." + Path(path).stem
        module = ModuleType(key)
        module.__package__ = name
        sys.modules[key] = module
        exec(compile(source, path, "exec"), module.__dict__)
        modules.append(module)
    return (*modules, hashes)


def measure(metrics, watchdog, count, repeats):
    recorder = metrics.MetricsRecorder(clock=lambda: 1.0)
    for _ in range(count):
        recorder.mark(metrics.ASR_FINAL)
        recorder.mark(metrics.LLM_FIRST_TOKEN)
        recorder.mark(metrics.TTS_FIRST_AUDIO)
        recorder.close_turn()
    calls = [0]
    monitor = watchdog.StuckWatchdog(
        recorder, clock=lambda: 2.0, on_tick=lambda: calls.__setitem__(0, calls[0] + 1)
    )
    tracemalloc.start()
    monitor.tick()
    _, initial_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    gc.collect()
    timings = []
    for _ in range(repeats):
        before = time.perf_counter_ns()
        monitor.tick()
        timings.append((time.perf_counter_ns() - before) / 1000)
    tracemalloc.start()
    monitor.tick()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert calls[0] == repeats + 2 and len(recorder.records()) == count
    return {
        "turns": count,
        "repeats": repeats,
        "tick_us_p50": statistics.median(timings),
        "tick_us_max": max(timings),
        "incremental_peak_bytes": peak,
        "initial_discovery_peak_bytes": initial_peak,
        "history_preserved": True,
        "maintenance_calls_preserved": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="1bed9e2")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    base_ref = subprocess.check_output(
        ["git", "rev-parse", args.baseline_ref], cwd=ROOT, text=True
    ).strip()
    baseline = load_pair("_watchdog_baseline", base_ref)
    current = load_pair("_watchdog_current")
    result = {
        "schema_version": 1,
        "kind": "synthetic_settled_history_watchdog",
        "baseline_commit": base_ref,
        "physical_audio": False,
        "native_models": False,
        "source_sha256": {"baseline": baseline[2], "current": current[2]},
        "limits": "Repeated synthetic monitor work, not voice latency or total session RSS; initial history discovery excluded.",
        "cases": [],
    }
    for count in (1000, 50000):
        for iteration in range(2):
            order = (
                (("baseline", baseline), ("current", current))
                if iteration == 0
                else (("current", current), ("baseline", baseline))
            )
            row = {"turns": count, "order": [name for name, _ in order]}
            for name, modules in order:
                row[name] = measure(modules[0], modules[1], count, 30)
            result["cases"].append(row)
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
