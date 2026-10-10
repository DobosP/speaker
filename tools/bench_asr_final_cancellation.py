"""Pinned, synthetic final-ASR cancellation work/overhead benchmark.

Only project source and public arrays are used. Verifier scratch allocation is
an explicit fake workload, not a native inference/whole-agent performance claim.
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
import tracemalloc
from threading import Event
from types import SimpleNamespace

import numpy as np

from core.engines import sherpa


_BASE_COMMIT = "1bed9e21df29c7885a88757b07e154e2ade972a3"
_PHASES = ("punctuation", "create", "accept", "decode", "verifier")
_THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _baseline():
    repo = Path(__file__).resolve().parents[1]
    source = subprocess.check_output(
        ["git", "show", f"{_BASE_COMMIT}:core/engines/sherpa.py"],
        cwd=repo,
        text=True,
    )
    tree = ast.parse(source)
    functions = "\n\n".join(
        ast.get_source_segment(source, node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_postprocess_final_text", "_resolve_final_transcript"}
    )
    namespace = dict(vars(sherpa))
    exec(compile(functions, "<pinned-project-final-selector>", "exec"), namespace)
    return namespace["_resolve_final_transcript"], sha256(
        functions.encode()
    ).hexdigest()


class _Models:
    def __init__(
        self,
        cancel_at=None,
        *,
        scratch_samples=0,
        offline="public synthetic phrase",
        verifier="public synthetic phrase",
    ):
        self.cancel = Event()
        self.calls = []
        self.after_cancel = 0
        self.scratch_bytes = 0
        models = self

        def visit(phase):
            if models.cancel.is_set():
                models.after_cancel += 1
            models.calls.append(phase)
            if phase == cancel_at:
                models.cancel.set()

        class Stream:
            result = SimpleNamespace(text=offline)

            def accept_waveform(self, rate, pcm):
                visit("accept")

        class Offline:
            def create_stream(self):
                visit("create")
                return Stream()

            def decode_stream(self, stream):
                visit("decode")

        class Punctuation:
            def add_punctuation(self, text):
                visit("punctuation")
                return text

        class Verifier:
            def transcribe(self, pcm, rate):
                visit("verifier")
                if scratch_samples:
                    scratch = np.ones(scratch_samples, np.float32)
                    models.scratch_bytes += scratch.nbytes
                    assert float(np.sum(scratch)) == scratch_samples
                return SimpleNamespace(text=verifier)

        self.offline, self.punctuation, self.verifier = (
            Offline(),
            Punctuation(),
            Verifier(),
        )


def run_benchmark():
    if not all(os.environ.get(name) == "1" for name in _THREAD_ENV):
        raise ValueError("thread_environment_required")
    baseline, source_hash = _baseline()
    config = sherpa.SherpaConfig(asr_final_min_sec=0.1)
    pcm = np.ones(16000, np.float32)

    def run(resolver, models, raw="public synthetic phrase", **kwargs):
        if resolver is sherpa._resolve_final_transcript:
            kwargs["is_current"] = lambda: not models.cancel.is_set()
        return resolver(
            config,
            models.offline,
            models.punctuation,
            pcm,
            raw,
            final_verifier=models.verifier,
            log_exceptions=False,
            **kwargs,
        )

    equivalent = 0
    for backend in ("sense_voice", "nemo_transducer"):
        config.asr_final_backend = backend
        for raw in ("public synthetic phrase", "stop", ""):
            for offline in ("public synthetic phrase", ""):
                for verifier in ("public synthetic phrase", ""):
                    for allow in (False, True):
                        before = run(
                            baseline,
                            _Models(offline=offline, verifier=verifier),
                            raw,
                            allow_empty_streaming=allow,
                        )
                        after = run(
                            sherpa._resolve_final_transcript,
                            _Models(offline=offline, verifier=verifier),
                            raw,
                            allow_empty_streaming=allow,
                        )
                        assert before == after
                        equivalent += 1
    config.asr_final_backend = "sense_voice"
    cancellation = {}
    for phase in _PHASES:
        before, after = _Models(phase), _Models(phase)
        run(baseline, before)
        try:
            run(sherpa._resolve_final_transcript, after)
        except sherpa._FinalAsrCancelled:
            pass
        else:
            raise AssertionError("revocation_not_observed")
        assert after.calls == list(_PHASES[: _PHASES.index(phase) + 1])
        assert after.after_cancel == 0
        cancellation[phase] = {
            "baseline_calls_after_revocation": before.after_cancel,
            "current_calls_after_revocation": after.after_cancel,
        }

    def paired(*, cancel_at, scratch_samples, repetitions):
        values = ([], [])
        scratch = ([], [])
        for iteration in range(repetitions):
            for index in (0, 1) if iteration % 2 == 0 else (1, 0):
                model = _Models(cancel_at, scratch_samples=scratch_samples)
                at = time.perf_counter_ns()
                try:
                    run((baseline, sherpa._resolve_final_transcript)[index], model)
                except sherpa._FinalAsrCancelled:
                    pass
                values[index].append(time.perf_counter_ns() - at)
                scratch[index].append(model.scratch_bytes)
        return {
            name: {
                "p50_us": statistics.median(values[index]) / 1000,
                "p95_us": sorted(values[index])[int(0.95 * repetitions)] / 1000,
                "scratch_bytes_per_call": max(scratch[index]),
            }
            for index, name in enumerate(("baseline", "current"))
        }

    def allocation(resolver):
        model = _Models("decode", scratch_samples=262144)
        tracemalloc.start()
        try:
            run(resolver, model)
        except sherpa._FinalAsrCancelled:
            pass
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return {"traced_current_bytes": current, "traced_peak_bytes": peak}

    return {
        "schema": 1,
        "baseline_source_commit": _BASE_COMMIT,
        "baseline_helpers_sha256": source_hash,
        "current_helpers_sha256": sha256(
            "\n\n".join(
                inspect.getsource(helper)
                for helper in (
                    sherpa._FinalAsrCancelled,
                    sherpa._check_final_asr_current,
                    sherpa._postprocess_final_text,
                    sherpa._resolve_final_transcript,
                )
            ).encode()
        ).hexdigest(),
        "synthetic_only": True,
        "native_model_or_device": False,
        "thread_environment_requested_one": True,
        "cpu_isolation": False,
        "healthy_exact_decision_cases": equivalent,
        "cancellation_boundaries": cancellation,
        "healthy_no_cost_fake_paired": paired(
            cancel_at=None, scratch_samples=0, repetitions=400
        ),
        "revoked_after_offline_decode_fake_1mib_verifier_paired": paired(
            cancel_at="decode", scratch_samples=262144, repetitions=100
        ),
        "revoked_after_offline_decode_fake_allocation": {
            "baseline": allocation(baseline),
            "current": allocation(sherpa._resolve_final_transcript),
        },
        "limits": {
            "entered_native_call_preemption": False,
            "native_latency_or_rss_evidence": False,
            "whole_agent_or_physical_quality_evidence": False,
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_benchmark()
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
