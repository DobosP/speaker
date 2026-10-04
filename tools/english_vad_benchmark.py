"""Local Silero VAD cost and descriptive segmentation; no labelled accuracy.

Uses only explicitly supplied recorded PCM and model files. Complete transcripts
stay in the corpus helper's private objects and are never scored or rendered.
One bounded offline worker runs per model; no live devices or agent are used.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import importlib.util
from importlib import metadata
import json
import math
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import tempfile
import time

MODEL_IDS = ("silero-v4-installed", "silero-v6.2-wheel")
PHASES = {"request", "bindings", "load", "frames", "final_bindings", "write"}
MODEL_LIMIT = 5 * 1024**2
REPORT_LIMIT = 65536
CONFIG = {
    "sample_rate": 16000,
    "window_size": 512,
    "threshold": 0.5,
    "min_speech_duration": 0.25,
    "min_silence_duration": 0.3,
    "max_speech_duration": 20.0,
    # Select the SDK's native default. The installed Python ABI has no setter.
    "neg_threshold": -1.0,
    "buffer_size_in_seconds": 31.0,
}
NEG_THRESHOLD_BINDING = "native-sdk-default-unsettable"
_DIGEST = re.compile(r"[a-f0-9]{64}\Z")
_HELPER = None


class VadBenchmarkError(RuntimeError):
    """Only a fixed error code crosses the process boundary."""


@dataclass(frozen=True)
class Model:
    id: str
    path: Path = field(repr=False)


def _helper():
    global _HELPER
    if _HELPER is None:
        path = Path(__file__).with_name("english_asr_benchmark.py")
        spec = importlib.util.spec_from_file_location("english_vad_corpus_helper", path)
        if spec is None or spec.loader is None:
            raise VadBenchmarkError()
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _HELPER = module
    return _HELPER


def _code_binding():
    helper = _helper()
    return {
        key: hashlib.sha256(helper._read(path, helper.MAX_MANIFEST_BYTES)).hexdigest()
        for key, path in {
            "benchmark_sha256": Path(__file__).resolve(),
            "corpus_helper_sha256": Path(helper.__file__).resolve(),
        }.items()
    }


def _model_binding(model):
    if model.id not in MODEL_IDS:
        raise VadBenchmarkError()
    digest, size = _helper()._file_digest(model.path, MODEL_LIMIT)
    return {"sha256": digest, "bytes": size}


def _runtime_binding():
    helper = _helper()
    digest, _size = helper._file_digest(
        Path(sys.executable).resolve(strict=True), 64 * 1024**2
    )
    result = {"executable_sha256": digest}
    marker = Path(sys.executable).parent.parent / "pyvenv.cfg"
    if marker.exists():
        result["venv_marker_sha256"] = hashlib.sha256(
            helper._read(marker, 65536)
        ).hexdigest()
    return result


def _native(model, threads):
    # Lazy native import occurs only in the output-suppressed CPU worker.
    import sherpa_onnx

    if metadata.version("sherpa-onnx") != "1.13.3":
        raise VadBenchmarkError()
    config = sherpa_onnx.VadModelConfig()
    if CONFIG["neg_threshold"] != -1.0 or hasattr(config.silero_vad, "neg_threshold"):
        # A different request or ABI requires explicit support and a new receipt.
        raise VadBenchmarkError()
    config.sample_rate = CONFIG["sample_rate"]
    config.num_threads = threads
    config.provider = "cpu"
    config.debug = False
    config.silero_vad.model = str(model.path)
    for key in (
        "window_size",
        "threshold",
        "min_speech_duration",
        "min_silence_duration",
        "max_speech_duration",
    ):
        setattr(config.silero_vad, key, CONFIG[key])
    return sherpa_onnx.VoiceActivityDetector(
        config, buffer_size_in_seconds=CONFIG["buffer_size_in_seconds"]
    )


def _segment_lengths(vad, frames):
    total = count = 0
    while not vad.empty():
        segment = vad.front
        start, size = segment.start, len(segment.samples)
        if type(start) is not int or not 0 <= start <= frames + CONFIG["window_size"]:
            raise VadBenchmarkError()
        if not 0 <= size <= frames + CONFIG["window_size"]:
            raise VadBenchmarkError()
        total += max(0, min(start + size, frames) - min(start, frames))
        count += 1
        if count > 2048:
            raise VadBenchmarkError()
        vad.pop()
    return total, count


def feed_clip(vad, samples):
    """Submit nonoverlapping 512-new-sample frames, reset and flush per clip."""
    import numpy as np

    if (
        type(samples) is not np.ndarray
        or samples.dtype != np.dtype("float32")
        or samples.ndim != 1
        or not 1 <= len(samples) <= 30 * 16000
        or not np.isfinite(samples).all()
        or (np.abs(samples) > 1.0).any()
    ):
        raise VadBenchmarkError()
    vad.reset()
    lengths = segments = active = calls = padding = 0
    times = []
    frame_cpu_seconds = 0.0
    for offset in range(0, len(samples), CONFIG["window_size"]):
        source = samples[offset : offset + CONFIG["window_size"]]
        frame = np.zeros(CONFIG["window_size"], dtype=np.float32)
        frame[: len(source)] = source
        padding += CONFIG["window_size"] - len(source)
        started = time.perf_counter()
        cpu_started = time.process_time()
        vad.accept_waveform(frame)
        state = vad.is_speech_detected()
        if type(state) is not bool:
            raise VadBenchmarkError()
        active += int(state)
        size, count = _segment_lengths(vad, len(samples))
        lengths += size
        segments += count
        times.append((time.perf_counter() - started) * 1000)
        frame_cpu_seconds += time.process_time() - cpu_started
        calls += 1
    flush_start = time.perf_counter()
    vad.flush()
    size, count = _segment_lengths(vad, len(samples))
    flush_ms = (time.perf_counter() - flush_start) * 1000
    lengths += size
    segments += count
    if not 0 <= lengths <= len(samples) or segments > 2048:
        raise VadBenchmarkError()
    return {
        "frames": calls,
        "active_frames": active,
        "padding_samples": padding,
        "segment_samples": lengths,
        "segments": segments,
        "frame_ms": times,
        "flush_ms": flush_ms,
        "frame_cpu_seconds": frame_cpu_seconds,
    }


def benchmark_cell(model, corpus, repeats, threads, *, factory=_native, phase=None):
    import numpy as np

    helper = _helper()
    binding = _model_binding(model)
    model_identity = helper._identity(model.path.lstat())
    runtime, code = _runtime_binding(), _code_binding()
    helper.verify_corpus(corpus)
    if phase:
        phase("load")
    wall, cpu = time.perf_counter(), time.process_time()
    load_start = time.perf_counter()
    vad = factory(model, threads)
    load_ms = (time.perf_counter() - load_start) * 1000
    if phase:
        phase("frames")
    totals = {
        key: 0
        for key in (
            "frames",
            "active_frames",
            "padding_samples",
            "segment_samples",
            "segments",
        )
    }
    timings, flush_times = [], []
    frame_cpu_seconds = 0.0
    for _repeat in range(repeats):
        for clip in corpus.clips:
            raw = helper._read(clip.path, helper.MAX_WAV_BYTES)
            if hashlib.sha256(raw).hexdigest() != clip.digest:
                raise VadBenchmarkError()
            pcm, _frames = helper._pcm(raw)
            samples = np.frombuffer(pcm, dtype="<i2").astype("float32") / 32768.0
            stats = feed_clip(vad, samples)
            for key in totals:
                totals[key] += stats[key]
            timings.extend(stats["frame_ms"])
            flush_times.append(stats["flush_ms"])
            frame_cpu_seconds += stats["frame_cpu_seconds"]
    if phase:
        phase("final_bindings")
    helper.verify_corpus(corpus)
    if (
        binding != _model_binding(model)
        or model_identity != helper._identity(model.path.lstat())
        or runtime != _runtime_binding()
        or code != _code_binding()
    ):
        raise VadBenchmarkError()
    del vad
    audio = corpus.seconds * repeats
    frame_seconds = sum(timings) / 1000
    flush_seconds = sum(flush_times) / 1000
    return {
        "model_id": model.id,
        "status": "complete",
        "error_count": 0,
        "corpus_sha256": corpus.digest,
        "model_binding": binding,
        "runtime_binding": runtime,
        "code_binding": code,
        "versions": {
            "python": sys.version.split()[0],
            "sherpa-onnx": metadata.version("sherpa-onnx"),
            "numpy": np.__version__,
        },
        "config": dict(CONFIG),
        "neg_threshold_binding": NEG_THRESHOLD_BINDING,
        "native_exit_threshold_attested": False,
        "clips": len(corpus.clips),
        "repeats": repeats,
        "threads": threads,
        "source_audio_seconds": corpus.seconds,
        "source_audio_seconds_across_calls": audio,
        "clip_calls": len(corpus.clips) * repeats,
        **totals,
        "complete_corpus_coverage": True,
        "model_load_ms": load_ms,
        "first_frame_ms": timings[0],
        "frame_p50_ms": helper._percentile(timings, 0.5),
        "frame_p95_ms": helper._percentile(timings, 0.95),
        "frame_seconds": frame_seconds,
        "flush_seconds": flush_seconds,
        "frame_cpu_seconds": frame_cpu_seconds,
        "frame_rtf": frame_seconds / audio,
        "frame_cpu_rtf": frame_cpu_seconds / audio,
        "worker_wall_seconds": time.perf_counter() - wall,
        "worker_cpu_seconds": time.process_time() - cpu,
        "peak_rss_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024)
        ),
        "active_frame_fraction": totals["active_frames"] / totals["frames"],
        "segment_seconds": totals["segment_samples"] / 16000,
    }


def _limits(repeats, threads, timeout):
    if (
        type(repeats) is not int
        or not 1 <= repeats <= 8
        or type(threads) is not int
        or not 1 <= threads <= 8
    ):
        raise VadBenchmarkError()
    if (
        type(timeout) not in (int, float)
        or not math.isfinite(timeout)
        or not 1 <= timeout <= 1800
    ):
        raise VadBenchmarkError()


def _write_new(path, value):
    raw = _helper()._canonical(value) + b"\n"
    if len(raw) > REPORT_LIMIT:
        raise VadBenchmarkError()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _worker(request_path):
    fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)
    request, phase = None, "request"

    def set_phase(value):
        nonlocal phase
        phase = value

    try:
        helper = _helper()
        request = helper._json(helper._read(request_path, REPORT_LIMIT))
        keys = {
            "manifest",
            "source_root",
            "model_id",
            "model",
            "corpus_sha256",
            "model_binding",
            "runtime_binding",
            "code_binding",
            "repeats",
            "threads",
            "timeout",
            "result",
        }
        if (
            type(request) is not dict
            or set(request) != keys
            or request["model_id"] not in MODEL_IDS
        ):
            raise VadBenchmarkError()
        _limits(request["repeats"], request["threads"], request["timeout"])
        resource.setrlimit(
            resource.RLIMIT_CPU,
            (math.ceil(request["timeout"]), math.ceil(request["timeout"]) + 1),
        )
        resource.setrlimit(resource.RLIMIT_AS, (12 * 1024**3, 12 * 1024**3))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        affinity = helper._cpu_affinity(request["threads"])
        if affinity != {"applied": True, "logical_cpus": request["threads"]}:
            raise VadBenchmarkError()
        helper._deny_network()
        phase = "bindings"
        corpus = helper.load_corpus(request["manifest"], request["source_root"])
        model = Model(request["model_id"], helper._absolute(request["model"]))
        if (
            corpus.digest != request["corpus_sha256"]
            or _model_binding(model) != request["model_binding"]
            or _runtime_binding() != request["runtime_binding"]
            or _code_binding() != request["code_binding"]
        ):
            raise VadBenchmarkError()
        result = benchmark_cell(
            model, corpus, request["repeats"], request["threads"], phase=set_phase
        )
        result["cpu_affinity"] = affinity
        phase = "write"
        _write_new(Path(request["result"]), result)
        return 0
    except BaseException:
        if (
            type(request) is dict
            and request.get("model_id") in MODEL_IDS
            and type(request.get("result")) is str
        ):
            try:
                _write_new(
                    Path(request["result"]),
                    {
                        "model_id": request["model_id"],
                        "status": "worker_failed",
                        "error_count": 1,
                        "phase": phase,
                    },
                )
            except BaseException:
                pass
        return 2


def _validate_result(result, model_id, corpus, repeats, threads):
    def number(value, low=0, high=None):
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or value < low
            or (high is not None and value > high)
        ):
            raise VadBenchmarkError()

    def integer(value, low=0, high=2**31):
        if type(value) is not int or not low <= value <= high:
            raise VadBenchmarkError()

    def digest(value):
        if type(value) is not str or not _DIGEST.fullmatch(value):
            raise VadBenchmarkError()

    if type(result) is not dict or result.get("model_id") != model_id:
        raise VadBenchmarkError()
    if result.get("status") == "worker_failed":
        if (
            set(result) != {"model_id", "status", "error_count", "phase"}
            or type(result["error_count"]) is not int
            or result["error_count"] != 1
            or result["phase"] not in PHASES
        ):
            raise VadBenchmarkError()
        return
    numeric = {
        "clips",
        "repeats",
        "threads",
        "source_audio_seconds",
        "source_audio_seconds_across_calls",
        "clip_calls",
        "frames",
        "active_frames",
        "padding_samples",
        "segment_samples",
        "segments",
        "model_load_ms",
        "first_frame_ms",
        "frame_p50_ms",
        "frame_p95_ms",
        "frame_seconds",
        "flush_seconds",
        "frame_cpu_seconds",
        "frame_rtf",
        "frame_cpu_rtf",
        "worker_wall_seconds",
        "worker_cpu_seconds",
        "peak_rss_bytes",
        "active_frame_fraction",
        "segment_seconds",
    }
    keys = numeric | {
        "model_id",
        "status",
        "error_count",
        "corpus_sha256",
        "model_binding",
        "runtime_binding",
        "code_binding",
        "versions",
        "config",
        "complete_corpus_coverage",
        "cpu_affinity",
        "neg_threshold_binding",
        "native_exit_threshold_attested",
    }
    if (
        set(result) != keys
        or result.get("status") != "complete"
        or type(result["error_count"]) is not int
        or result["error_count"] != 0
        or result["complete_corpus_coverage"] is not True
    ):
        raise VadBenchmarkError()
    if (
        result["neg_threshold_binding"] != NEG_THRESHOLD_BINDING
        or result["native_exit_threshold_attested"] is not False
    ):
        raise VadBenchmarkError()
    for key in numeric:
        number(result[key])
    for key in (
        "clips",
        "repeats",
        "threads",
        "clip_calls",
        "frames",
        "active_frames",
        "padding_samples",
        "segment_samples",
        "segments",
        "peak_rss_bytes",
    ):
        integer(result[key])
    expected_frames = (
        sum(math.ceil(clip.frames / 512) for clip in corpus.clips) * repeats
    )
    expected_padding = sum((-clip.frames) % 512 for clip in corpus.clips) * repeats
    expected = {
        "clips": len(corpus.clips),
        "repeats": repeats,
        "threads": threads,
        "clip_calls": len(corpus.clips) * repeats,
        "frames": expected_frames,
        "padding_samples": expected_padding,
    }
    if (
        any(result[key] != value for key, value in expected.items())
        or not 0 <= result["active_frames"] <= expected_frames
        or result["segment_samples"]
        > sum(clip.frames for clip in corpus.clips) * repeats
    ):
        raise VadBenchmarkError()
    if result["corpus_sha256"] != corpus.digest:
        raise VadBenchmarkError()
    digest(result["corpus_sha256"])
    if (
        type(result["config"]) is not dict
        or set(result["config"]) != set(CONFIG)
        or any(
            type(result["config"][key]) is not type(value)
            or result["config"][key] != value
            for key, value in CONFIG.items()
        )
    ):
        raise VadBenchmarkError()
    for key, value in {
        "source_audio_seconds": corpus.seconds,
        "source_audio_seconds_across_calls": corpus.seconds * repeats,
        "active_frame_fraction": result["active_frames"] / expected_frames,
        "segment_seconds": result["segment_samples"] / 16000,
        "frame_rtf": result["frame_seconds"] / (corpus.seconds * repeats),
        "frame_cpu_rtf": result["frame_cpu_seconds"] / (corpus.seconds * repeats),
    }.items():
        if not math.isclose(result[key], value, rel_tol=1e-12, abs_tol=1e-12):
            raise VadBenchmarkError()
    binding = result["model_binding"]
    if type(binding) is not dict or set(binding) != {"sha256", "bytes"}:
        raise VadBenchmarkError()
    digest(binding["sha256"])
    integer(binding["bytes"], 1, MODEL_LIMIT)
    for key, required, allowed in (
        (
            "runtime_binding",
            {"executable_sha256"},
            {"executable_sha256", "venv_marker_sha256"},
        ),
        (
            "code_binding",
            {"benchmark_sha256", "corpus_helper_sha256"},
            {"benchmark_sha256", "corpus_helper_sha256"},
        ),
    ):
        value = result[key]
        if type(value) is not dict or not required <= set(value) <= allowed:
            raise VadBenchmarkError()
        for item in value.values():
            digest(item)
    versions = result["versions"]
    if (
        type(versions) is not dict
        or set(versions) != {"python", "numpy", "sherpa-onnx"}
        or any(
            type(v) is not str or not re.fullmatch(r"[A-Za-z0-9.+_-]{1,64}", v)
            for v in versions.values()
        )
    ):
        raise VadBenchmarkError()
    if (
        result["cpu_affinity"] != {"applied": True, "logical_cpus": threads}
        or type(result["cpu_affinity"].get("applied")) is not bool
        or type(result["cpu_affinity"].get("logical_cpus")) is not int
    ):
        raise VadBenchmarkError()


def run_cell(model, corpus, scratch, repeats, threads, timeout):
    helper = _helper()
    binding, runtime, code = _model_binding(model), _runtime_binding(), _code_binding()
    with tempfile.TemporaryDirectory(prefix="vad-cell-", dir=scratch) as directory:
        result_path, request_path = (
            Path(directory) / "result.json",
            Path(directory) / "request.json",
        )
        _write_new(
            request_path,
            {
                "manifest": str(corpus.manifest),
                "source_root": str(corpus.source_root),
                "model_id": model.id,
                "model": str(model.path),
                "corpus_sha256": corpus.digest,
                "model_binding": binding,
                "runtime_binding": runtime,
                "code_binding": code,
                "repeats": repeats,
                "threads": threads,
                "timeout": timeout,
                "result": str(result_path),
            },
        )
        environment = {
            "PATH": os.defpath,
            "LANG": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": str(threads),
            "OPENBLAS_NUM_THREADS": str(threads),
            "MKL_NUM_THREADS": str(threads),
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }
        process = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-B",
                str(Path(__file__).resolve()),
                "--worker-request",
                str(request_path),
            ],
            cwd=directory,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            helper._stop_group(process)
            return {"model_id": model.id, "status": "worker_timeout", "error_count": 1}
        except BaseException:
            helper._stop_group(process)
            raise
        try:
            result = helper._json(helper._read(result_path, REPORT_LIMIT))
            _validate_result(result, model.id, corpus, repeats, threads)
            if result["status"] == "complete" and (
                process.returncode != 0
                or result["model_binding"] != binding
                or result["runtime_binding"] != runtime
                or result["code_binding"] != code
            ):
                raise VadBenchmarkError()
        except BaseException:
            return {"model_id": model.id, "status": "result_rejected", "error_count": 1}
    helper.verify_corpus(corpus)
    if (
        binding != _model_binding(model)
        or runtime != _runtime_binding()
        or code != _code_binding()
    ):
        raise VadBenchmarkError()
    return result


class _Parser(argparse.ArgumentParser):
    def error(self, _message):
        self.exit(2, '{"error":"english_vad_benchmark_invalid","ok":false}\n')


def main(argv=None):
    parser = _Parser(description=__doc__)
    for name in (
        "manifest",
        "source-root",
        "baseline-model",
        "candidate-model",
        "scratch-root",
        "output",
        "worker-request",
    ):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--timeout-sec", type=float, default=300)
    args = parser.parse_args(argv)
    if args.worker_request is not None:
        return _worker(args.worker_request)
    try:
        _limits(args.repeats, args.threads, args.timeout_sec)
        helper = _helper()
        if any(
            value is None
            for value in (
                args.manifest,
                args.baseline_model,
                args.candidate_model,
                args.scratch_root,
            )
        ):
            raise VadBenchmarkError()
        scratch = helper._absolute(args.scratch_root)
        if not scratch.is_dir() or scratch.stat().st_mode & 0o777 != 0o700:
            raise VadBenchmarkError()
        corpus = helper.load_corpus(args.manifest, args.source_root)
        models = [
            Model(identifier, helper._absolute(path))
            for identifier, path in zip(
                MODEL_IDS, (args.baseline_model, args.candidate_model), strict=True
            )
        ]
        config = {
            "models": [
                {"id": model.id, "binding": _model_binding(model)} for model in models
            ],
            "vad": CONFIG,
            "neg_threshold_binding": NEG_THRESHOLD_BINDING,
            "native_exit_threshold_attested": False,
            "threads": args.threads,
            "repeats": args.repeats,
        }
        cells = [
            run_cell(
                model, corpus, scratch, args.repeats, args.threads, args.timeout_sec
            )
            for model in models
        ]
        result = {
            "schema_version": 1,
            "benchmark": "silero-vad-cost-and-segments",
            "corpus_sha256": corpus.digest,
            "config_sha256": hashlib.sha256(helper._canonical(config)).hexdigest(),
            "config": CONFIG,
            "cells": cells,
            "no_accuracy_labels": True,
            "neg_threshold_binding": NEG_THRESHOLD_BINDING,
            "native_exit_threshold_attested": False,
            "native_inference_coverage_attested": False,
            "frame_metric_kind": "public-api-callback-not-model-kernel",
            "frame_policy": "512-new-samples-final-zero-pad-only",
            "state_policy": "reset-each-clip-flush-without-extra-audio",
            "fraction_kind": "hysteretic-detected-state-per-padded-frame",
            "segment_policy": "clamped-to-original-clip",
            "network_sandbox_attested": False,
            "network_policy": "local-model-constructor-python-connect-hooks-offline-env",
        }
        if args.output is not None:
            _write_new(args.output, result)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0 if all(cell["status"] == "complete" for cell in cells) else 1
    except BaseException:
        print('{"error":"english_vad_benchmark_invalid","ok":false}')
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
