"""Local CPU canary for reference prompts; no tools, audio or cloud calls.

Owner references remain private. These measurements exclude ASR errors and do
not qualify model tool use, memory, or live conversation behavior.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time

MODEL_HASHES = {
    "minicpm5-1b-q8": "0dc7638539067268774c275a14a6ec9c7e01f7eeb2cff606c8590361fa527e4c",
    "minicpm5-2b-q4km": "ec2d5801640099e97d8d7e8003ad4d81f336e757811f03a26173dddf386602fd",
}
SYSTEM = (
    "You are a local English voice assistant. Give only the spoken answer, "
    "briefly in at most two sentences. You have no access to live weather, "
    "device time, reminders, apps or conversation memory in this test."
)
CANARIES = (
    (
        "Compute twenty three plus nineteen. Reply only with the decimal number.",
        "arithmetic",
    ),
    ("What is the capital of France? Reply only with the city name.", "geography"),
    ("Spell the word necessary. Reply only with the word.", "spelling"),
    ("Respond with exactly the word READY.", "instruction"),
)
MAX_INPUT = 2 * 1024 * 1024


class CanaryError(RuntimeError):
    """Detail-free prerequisite or worker-contract failure."""


def _identity(metadata):
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_json(path: Path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise CanaryError()
            result[key] = value
        return result

    metadata = path.lstat()
    if (
        path.resolve(strict=True) != path
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or not 0 < metadata.st_size <= MAX_INPUT
    ):
        raise CanaryError()
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as source:
        if _identity(os.fstat(source.fileno())) != _identity(metadata):
            raise CanaryError()
        raw = source.read(MAX_INPUT + 1)
        if _identity(os.fstat(source.fileno())) != _identity(metadata):
            raise CanaryError()
    if len(raw) != metadata.st_size or _identity(path.lstat()) != _identity(metadata):
        raise CanaryError()
    return json.loads(
        raw,
        object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(CanaryError()),
    ), hashlib.sha256(raw).hexdigest()


def _model_hash(path: Path):
    metadata = path.lstat()
    if (
        path.resolve(strict=True) != path
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or not 0 < metadata.st_size <= 2 * 1024**3
    ):
        raise CanaryError()
    result = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    total = 0
    with os.fdopen(descriptor, "rb") as source:
        if _identity(os.fstat(source.fileno())) != _identity(metadata):
            raise CanaryError()
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            total += len(block)
            if total > metadata.st_size:
                raise CanaryError()
            result.update(block)
        if total != metadata.st_size or _identity(
            os.fstat(source.fileno())
        ) != _identity(metadata):
            raise CanaryError()
    if _identity(path.lstat()) != _identity(metadata):
        raise CanaryError()
    return result.hexdigest(), metadata.st_size


def reference_prompts(value):
    if (
        not isinstance(value, dict)
        or set(value) != {"clips"}
        or not isinstance(value["clips"], list)
        or not 1 <= len(value["clips"]) <= 256
    ):
        raise CanaryError()
    groups = {}
    for item in value["clips"]:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("text"), str)
            or not 0 < len(item["text"]) <= 1024
        ):
            raise CanaryError()
        group = item.get("group")
        if group not in {
            "questions",
            "commands",
            "long",
            "barge",
            "memory",
            "natural",
            "corrections",
        }:
            raise CanaryError()
        groups.setdefault(group, []).append(item["text"])
    selected = [groups[group][0] for group in sorted(groups)]
    for round_index in range(1, 8):
        for group in sorted(groups):
            if len(selected) >= 8:
                return selected[:8]
            if len(groups[group]) > round_index:
                selected.append(groups[group][round_index])
    return selected[:8]


def canary_passes(kind, response):
    if not isinstance(response, str):
        return False
    import re

    text = response.strip().lower()
    if kind == "arithmetic":
        return (
            re.fullmatch(r"(?:the answer is\s+)?(?:42|forty[- ]two)[.!]?", text)
            is not None
        )
    if kind == "geography":
        return (
            re.fullmatch(
                r"(?:(?:the capital of france is|it is|it's)\s+)?paris[.!]?", text
            )
            is not None
        )
    if kind == "spelling":
        return text in {
            "necessary",
            "necessary.",
            "n e c e s s a r y",
            "n-e-c-e-s-s-a-r-y",
        }
    if kind == "instruction":
        return response.strip() == "READY"

    raise CanaryError()


def percentile(values, quantile):
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise CanaryError()
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def _write(path: Path, value):
    payload = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    with path.open("xb") as destination:
        os.chmod(path, 0o600)
        destination.write(payload)


@contextmanager
def _quiet_native():
    saved = [os.dup(1), os.dup(2)]
    descriptor = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(descriptor, 1)
        os.dup2(descriptor, 2)
        yield
    finally:
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        for item in (*saved, descriptor):
            os.close(item)


def _deny_network(*_args, **_kwargs):
    raise CanaryError()


def _worker(request_path: Path, result_path: Path):
    with _quiet_native():
        try:
            request, _ = _read_json(request_path)
            if not isinstance(request, dict) or set(request) != {
                "model_id",
                "model_path",
                "manifest",
                "manifest_sha256",
                "threads",
                "repeats",
                "canaries_only",
            }:
                raise CanaryError()
            if (
                type(request["threads"]) is not int
                or not 1 <= request["threads"] <= 8
                or type(request["repeats"]) is not int
                or not 1 <= request["repeats"] <= 4
            ):
                raise CanaryError()
            if type(request["canaries_only"]) is not bool:
                raise CanaryError()
            model_id = request["model_id"]
            if model_id not in MODEL_HASHES:
                raise CanaryError()
            model_path = Path(request["model_path"])
            identity, size = _model_hash(model_path)
            if identity != MODEL_HASHES[model_id]:
                raise CanaryError()
            value, manifest_hash = _read_json(Path(request["manifest"]))
            selected = [] if request["canaries_only"] else reference_prompts(value)
            if manifest_hash != request["manifest_sha256"]:
                raise CanaryError()
            threads = request["threads"]
            affinity = False
            if hasattr(os, "sched_getaffinity"):
                os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:threads])
                affinity = True
            resource.setrlimit(resource.RLIMIT_CPU, (900, 901))
            resource.setrlimit(resource.RLIMIT_AS, (12 * 1024**3, 12 * 1024**3))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            socket.socket.connect = _deny_network
            socket.socket.connect_ex = _deny_network
            socket.create_connection = _deny_network
            socket.socket.sendto = _deny_network
            from importlib.metadata import version
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import Jinja2ChatFormatter

            started = time.perf_counter()
            llm = Llama(
                model_path=str(model_path),
                n_ctx=1024,
                n_threads=threads,
                n_threads_batch=threads,
                n_gpu_layers=0,
                seed=7,
                verbose=False,
            )
            load_ms = (time.perf_counter() - started) * 1000
            eos = llm.detokenize([llm.token_eos()], special=True).decode()
            bos = llm.detokenize([llm.token_bos()], special=True).decode()
            template = llm.metadata.get("tokenizer.chat_template")
            if not isinstance(template, str) or not template:
                raise CanaryError()
            formatter = Jinja2ChatFormatter(template, eos_token=eos, bos_token=bos)

            def generate(text):
                formatted = formatter(
                    messages=[
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": text},
                    ],
                    enable_thinking=False,
                )
                prompt_tokens = llm.tokenize(
                    formatted.prompt.encode(),
                    add_bos=not formatted.added_special,
                    special=True,
                )
                if len(prompt_tokens) > 512:
                    raise CanaryError()
                start = time.perf_counter()
                first_ms = None
                response = []
                truncated = False
                for event in llm.create_completion(
                    prompt_tokens,
                    stream=True,
                    max_tokens=64,
                    temperature=0.0,
                    min_p=0.0,
                    top_k=0,
                    top_p=1.0,
                    stop=formatted.stop,
                    seed=7,
                    stopping_criteria=formatted.stopping_criteria,
                ):
                    choice = event["choices"][0]
                    truncated |= choice.get("finish_reason") == "length"
                    piece = choice.get("text", "")
                    if piece:
                        if first_ms is None:
                            first_ms = (time.perf_counter() - start) * 1000
                        response.append(piece)
                elapsed_ms = (time.perf_counter() - start) * 1000
                output = "".join(response)
                count = len(llm.tokenize(output.encode(), add_bos=False, special=True))
                return output, first_ms, elapsed_ms, count, truncated

            _, cold_first, cold_total, _, _ = generate("Say hello briefly.")
            firsts, totals, token_counts = [], [], []
            empty = reasoning = truncated_count = 0
            canaries = {kind: {"attempts": 0, "passed": 0} for _, kind in CANARIES}
            prompts = [(text, None) for text in selected] + list(CANARIES)
            cpu_start = time.process_time()
            for _ in range(request["repeats"]):
                for text, kind in prompts:
                    output, first_ms, elapsed_ms, count, truncated = generate(text)
                    truncated_count += truncated
                    totals.append(elapsed_ms)
                    token_counts.append(count)
                    if first_ms is None:
                        empty += 1
                    else:
                        firsts.append(first_ms)
                    reasoning += "<think>" in output or "<analysis>" in output
                    if kind is not None:
                        canaries[kind]["attempts"] += 1
                        canaries[kind]["passed"] += canary_passes(kind, output)
            cpu_seconds = time.process_time() - cpu_start
            llm.close()
            if (
                _model_hash(model_path)[0] != identity
                or _read_json(Path(request["manifest"]))[1] != manifest_hash
            ):
                raise CanaryError()
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
                1 if sys.platform == "darwin" else 1024
            )
            _write(
                result_path,
                {
                    "model_id": model_id,
                    "status": "ok",
                    "model_sha256": identity,
                    "model_bytes": size,
                    "runtime_version": version("llama-cpp-python"),
                    "template_sha256": hashlib.sha256(template.encode()).hexdigest(),
                    "threads": threads,
                    "cpu_affinity_enforced": affinity,
                    "gpu_layers": 0,
                    "context_tokens": 1024,
                    "max_reply_tokens": 64,
                    "thinking_requested": False,
                    "min_p": 0.0,
                    "cold_first_visible_chunk_ms": cold_first,
                    "cold_completion_ms": cold_total,
                    "load_ms": load_ms,
                    "warm_first_visible_chunk_ms": {
                        "p50": percentile(firsts, 0.5),
                        "p95": percentile(firsts, 0.95),
                    }
                    if firsts
                    else None,
                    "warm_completion_ms": {
                        "p50": percentile(totals, 0.5),
                        "p95": percentile(totals, 0.95),
                    },
                    "retokenized_completion_tokens_per_second": sum(token_counts)
                    / (sum(totals) / 1000),
                    "decoding_mode": "greedy; min_p inactive",
                    "canary_scoring": "strict-format-proxy",
                    "warm_filesystem_model_load": True,
                    "truncated_outputs": truncated_count,
                    "empty_outputs": empty,
                    "reasoning_marker_outputs": reasoning,
                    "owner_reference_prompts": len(selected),
                    "generations": len(prompts) * request["repeats"],
                    "cpu_seconds": cpu_seconds,
                    "process_peak_rss_bytes": peak,
                    "canaries": canaries,
                },
            )
        except BaseException:
            if not result_path.exists():
                _write(
                    result_path,
                    {
                        "model_id": request.get("model_id", "unknown")
                        if "request" in locals()
                        else "unknown",
                        "status": "worker_failed",
                    },
                )


def validate_report(report, model_id, threads, repeats, owner_count):
    import re

    if report["model_sha256"] != MODEL_HASHES[model_id] or not re.fullmatch(
        r"[0-9a-f]{64}", report["template_sha256"]
    ):
        raise CanaryError()
    if not isinstance(report["runtime_version"], str) or not re.fullmatch(
        r"[0-9]+(?:\.[0-9]+){1,3}(?:[a-z]+[0-9]*)?", report["runtime_version"]
    ):
        raise CanaryError()
    fixed = {
        "threads": threads,
        "gpu_layers": 0,
        "context_tokens": 1024,
        "max_reply_tokens": 64,
        "owner_reference_prompts": owner_count,
        "generations": (owner_count + len(CANARIES)) * repeats,
    }
    for key, expected in fixed.items():
        if type(report[key]) is not int or report[key] != expected:
            raise CanaryError()
    for key in (
        "cpu_affinity_enforced",
        "thinking_requested",
        "warm_filesystem_model_load",
    ):
        if type(report[key]) is not bool:
            raise CanaryError()
    if (
        report["thinking_requested"]
        or not report["warm_filesystem_model_load"]
        or report["decoding_mode"] != "greedy; min_p inactive"
        or report["canary_scoring"] != "strict-format-proxy"
    ):
        raise CanaryError()
    for key in ("model_bytes", "process_peak_rss_bytes"):
        if type(report[key]) is not int or not 0 < report[key] <= 32 * 1024**3:
            raise CanaryError()
    for key in ("empty_outputs", "reasoning_marker_outputs", "truncated_outputs"):
        if (
            type(report[key]) is not int
            or not 0 <= report[key] <= report["generations"]
        ):
            raise CanaryError()
    for key in (
        "min_p",
        "cold_completion_ms",
        "load_ms",
        "retokenized_completion_tokens_per_second",
        "cpu_seconds",
    ):
        if (
            type(report[key]) not in (int, float)
            or not math.isfinite(report[key])
            or not 0 <= report[key] <= 10**7
        ):
            raise CanaryError()
    if report["min_p"] != 0.0:
        raise CanaryError()
    cold = report["cold_first_visible_chunk_ms"]
    if cold is not None and (
        type(cold) not in (int, float)
        or not math.isfinite(cold)
        or not 0 <= cold <= 10**7
    ):
        raise CanaryError()
    for key in ("warm_first_visible_chunk_ms", "warm_completion_ms"):
        value = report[key]
        if (
            value is None
            and key == "warm_first_visible_chunk_ms"
            and report["empty_outputs"] == report["generations"]
        ):
            continue
        if (
            not isinstance(value, dict)
            or set(value) != {"p50", "p95"}
            or any(
                type(v) not in (int, float)
                or not math.isfinite(v)
                or not 0 <= v <= 10**7
                for v in value.values()
            )
            or value["p50"] > value["p95"]
        ):
            raise CanaryError()
    canaries = report["canaries"]
    if not isinstance(canaries, dict) or set(canaries) != {
        kind for _, kind in CANARIES
    }:
        raise CanaryError()
    for value in canaries.values():
        if (
            not isinstance(value, dict)
            or set(value) != {"attempts", "passed"}
            or type(value["attempts"]) is not int
            or value["attempts"] != repeats
            or type(value["passed"]) is not int
            or not 0 <= value["passed"] <= repeats
        ):
            raise CanaryError()


def _terminate_worker(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def run(args):
    if (
        type(args.repeats) is not int
        or not 1 <= args.repeats <= 4
        or type(args.threads) is not int
        or not 1 <= args.threads <= 8
    ):
        raise CanaryError()
    manifest = Path(args.manifest).absolute()
    value, manifest_sha = _read_json(manifest)
    selected = [] if args.canaries_only else reference_prompts(value)
    configuration, _ = _read_json(Path(args.model_config).absolute())
    if (
        set(configuration) != {"models"}
        or not isinstance(configuration["models"], list)
        or len(configuration["models"]) != 2
    ):
        raise CanaryError()
    scratch = Path(args.scratch_root).absolute()
    scratch.mkdir(parents=True, exist_ok=True, mode=0o700)
    reports = []
    seen_models = set()
    for entry in configuration["models"]:
        if set(entry) != {"id", "path"} or entry["id"] not in MODEL_HASHES:
            raise CanaryError()
        if entry["id"] in seen_models:
            raise CanaryError()
        seen_models.add(entry["id"])
        model_path = Path(entry["path"]).absolute()
        if _model_hash(model_path)[0] != MODEL_HASHES[entry["id"]]:
            raise CanaryError()
        root = Path(tempfile.mkdtemp(prefix="llm-cell-", dir=scratch))
        request = root / "request.json"
        result = root / "result.json"
        _write(
            request,
            {
                "model_id": entry["id"],
                "model_path": str(model_path),
                "manifest": str(manifest),
                "manifest_sha256": manifest_sha,
                "threads": args.threads,
                "repeats": args.repeats,
                "canaries_only": args.canaries_only,
            },
        )
        environment = {
            "PATH": os.defpath,
            "LANG": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "CUDA_VISIBLE_DEVICES": "",
        }
        process = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-B",
                __file__,
                "--worker",
                str(request),
                "--result",
                str(result),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env=environment,
        )
        try:
            process.wait(timeout=900)
        except BaseException:
            _terminate_worker(process)
            raise
        report = (
            _read_json(result)[0]
            if result.exists() and process.returncode == 0
            else {"model_id": entry["id"], "status": "worker_failed"}
        )
        # Only the fixed aggregate worker fields can cross to a report.
        expected = {
            "model_id",
            "status",
            "model_sha256",
            "model_bytes",
            "runtime_version",
            "template_sha256",
            "threads",
            "cpu_affinity_enforced",
            "gpu_layers",
            "context_tokens",
            "max_reply_tokens",
            "thinking_requested",
            "min_p",
            "cold_first_visible_chunk_ms",
            "cold_completion_ms",
            "load_ms",
            "warm_first_visible_chunk_ms",
            "warm_completion_ms",
            "retokenized_completion_tokens_per_second",
            "decoding_mode",
            "canary_scoring",
            "warm_filesystem_model_load",
            "truncated_outputs",
            "empty_outputs",
            "reasoning_marker_outputs",
            "owner_reference_prompts",
            "generations",
            "cpu_seconds",
            "process_peak_rss_bytes",
            "canaries",
        }
        if (
            report.get("model_id") != entry["id"]
            or report.get("status") not in {"ok", "worker_failed"}
            or (report["status"] == "ok" and set(report) != expected)
            or (
                report["status"] == "worker_failed"
                and set(report) != {"model_id", "status"}
            )
        ):
            raise CanaryError()
        if report["status"] == "ok":
            validate_report(
                report, entry["id"], args.threads, args.repeats, len(selected)
            )
        if _model_hash(model_path)[0] != MODEL_HASHES[entry["id"]]:
            raise CanaryError()
        reports.append(report)
    if _read_json(manifest)[1] != manifest_sha:
        raise CanaryError()
    payload = {
        "schema_version": 1,
        "kind": "english-local-llm-public-canary-v1"
        if args.canaries_only
        else "english-local-llm-reference-canary-v1",
        "manifest_sha256": manifest_sha,
        "benchmark_code_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "source_reference_count": len(value["clips"]),
        "selected_owner_reference_count": len(selected),
        "repeats": args.repeats,
        "cells": reports,
        "evidence_scope": {
            "audio_executed": False,
            "asr_errors_included": False,
            "tools_executed": False,
            "quality_promotion": False,
            "phone_validation": False,
            "network_inference": False,
        },
    }
    if args.output:
        _write(Path(args.output).absolute(), payload)
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    return 0 if all(item["status"] == "ok" for item in reports) else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest")
    parser.add_argument("--model-config")
    parser.add_argument("--scratch-root")
    parser.add_argument("--output")
    parser.add_argument(
        "--canaries-only",
        action="store_true",
        help="Run only the four explicitly formatted public constraints.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.worker is not None and args.result is not None:
        _worker(args.worker, args.result)
        return 0
    try:
        if not args.manifest or not args.model_config or not args.scratch_root:
            raise CanaryError()
        return run(args)
    except BaseException:
        print(json.dumps({"ok": False, "error": "english_llm_benchmark_unavailable"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
