"""Compare pinned/current bridge terminal waits using public local SDK fakes.

This measures final next() after response/client cleanup is already complete,
not model inference, whole-turn latency, playback or physical-device behavior.
No daemon, network, native model or private inputs are used.
"""

from __future__ import annotations

import argparse
import ast
from hashlib import sha256
import inspect
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import core.llm as implementation
from tests.test_ollama_async_cancel import (
    FakeAsyncChunks,
    FakeAsyncClientFactory,
    FakeSyncClient,
)


BASELINE = "1bed9e21df29c7885a88757b07e154e2ade972a3"
_CLASSES = ("_OllamaAsyncStreamProducer", "_OllamaAsyncTokenStream")


def run_benchmark(repeats: int = 5) -> dict:
    if type(repeats) is not int or not 3 <= repeats <= 20:
        raise ValueError("repeats_out_of_bounds")
    root = Path(__file__).resolve().parents[1]
    baseline = subprocess.check_output(
        ["git", "show", f"{BASELINE}:core/llm.py"],
        cwd=root,
        timeout=5,
    )
    tree = ast.parse(baseline)
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in _CLASSES
    ]
    if len(nodes) != 2:
        raise ValueError("baseline_classes_unavailable")
    namespace = dict(vars(implementation))
    # Execute only our two pinned repository classes, not third-party source.
    exec(
        compile(
            ast.Module(body=nodes, type_ignores=[]), "<pinned-ollama-bridge>", "exec"
        ),
        namespace,
    )
    baseline_stream = namespace[_CLASSES[1]]
    current_stream = implementation._OllamaAsyncTokenStream
    values = {"baseline": [], "current": []}
    for iteration in range(repeats):
        variants = (("baseline", baseline_stream), ("current", current_stream))
        if iteration % 2:
            variants = tuple(reversed(variants))
        for name, stream_class in variants:
            response = FakeAsyncChunks("Public reply.")
            factory = FakeAsyncClientFactory(response)
            owner = implementation.OllamaLLM(
                "public-local-fake",
                client=FakeSyncClient(),
                async_client_factory=factory,
            )
            stream = stream_class(owner, "Public prompt.", None, {}, None)
            try:
                if next(stream) != "Public reply." or not stream._producer.done.wait(
                    2.0
                ):
                    raise ValueError("fake_stream_incomplete")
                if not factory.clients[0].closed.is_set() or response.aclose_calls != 1:
                    raise ValueError("fake_cleanup_incomplete")
                started = time.perf_counter_ns()
                marker = object()
                if next(stream, marker) is not marker:
                    raise ValueError("fake_terminal_incomplete")
                values[name].append((time.perf_counter_ns() - started) / 1_000_000)
            finally:
                stream.close()
                stream._thread.join(2.0)
                if stream._thread.is_alive():
                    raise ValueError("fake_worker_retained")
    current_source = Path(implementation.__file__).read_bytes()
    current_classes = "\n".join(
        inspect.getsource(getattr(implementation, name)) for name in _CLASSES
    )
    baseline_classes = "\n".join(
        ast.get_source_segment(baseline.decode(), node) for node in nodes
    )
    return {
        "schema_version": 1,
        "scope": "public_fake_sdk_terminal_next_after_known_cleanup;not_model_turn_or_audio_latency",
        "synthetic_only": True,
        "network_or_native_model": False,
        "cpu_isolation_attested": False,
        "baseline_commit": BASELINE,
        "baseline_llm_source_sha256": sha256(baseline).hexdigest(),
        "current_llm_source_sha256": sha256(current_source).hexdigest(),
        "baseline_bridge_sha256": sha256(baseline_classes.encode()).hexdigest(),
        "current_bridge_sha256": sha256(current_classes.encode()).hexdigest(),
        "fixture_source_sha256": sha256(
            (root / "tests/test_ollama_async_cancel.py").read_bytes()
        ).hexdigest(),
        "benchmark_source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "method": "adjacent_alternating_variants;one_public_chunk;wait_known_done_before_terminal_next",
        "cases_per_variant": repeats,
        "terminal_next_ms": {
            name: {
                "p50": round(statistics.median(samples), 4),
                "min": round(min(samples), 4),
                "max": round(max(samples), 4),
            }
            for name, samples in values.items()
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args(argv)
    try:
        report = run_benchmark(args.repeats)
        payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(args.output, flags, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(payload)
    except Exception:
        print("ollama_terminal_benchmark_refused")
        return 2
    print(json.dumps(report, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
