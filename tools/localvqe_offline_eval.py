"""Pinned, offline synthetic AEC comparison; never a live/device quality gate."""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import stat
import subprocess
import sys
import time
import uuid

# -I workers explicitly load this reviewed repository, ignoring ambient paths.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from core.engines._aec import EchoCanceller, _FDAFAdaptiveFilter
from tools.audio_eval.metrics import (
    capped_db_ratio,
    projection_gain,
    si_sdr_db,
    signal_power,
)

try:
    import resource
except ImportError:  # Windows; RSS/rlimit availability is reported explicitly.
    resource = None

CODE_COMMIT = "f53063c9eb2a85f96479867d1dd911dc3bf6319b"
GGML_COMMIT = "c044a8eeae2591faa0950c8b5e514cbc4bbfc4ca"
WEIGHT_REVISION = "29ca38495cba9d6393a92a4dd890f28dd81f758d"
WEIGHTS = {
    "localvqe-aec-203k": (
        2924224,
        "b6e43138588a83bfe903ab5e143b4020b91c1e1629f5a575ac5855ff0003c731",
    ),
    "localvqe-aec-2.7k": (
        16864,
        "d79f824f6ee6f58b2a7108d1df90db51c42c15a7400a5addbd3be9972ba657a4",
    ),
}
MODELS = (*WEIGHTS, "apm-echo-only", "apm-ns-on", "nlms-default")
CASES = ("far_linear", "far_nonlinear", "far_drift", "near_only", "doubletalk")
SAMPLE_RATE = 16000
SAMPLES = 128000
CALLBACK_SAMPLES = 1280
SCORE_START = 64000
PHASES = {
    "request",
    "bindings",
    "load",
    "frames",
    "metrics",
    "close",
    "final_bindings",
    "timeout",
    "worker_result",
}
_NATIVE_NAME = re.compile(
    r"(?:lib)?(?:ggml|localvqe)[A-Za-z0-9_.-]*(?:\.so(?:\.\d+)*|\.dylib|\.dll)\Z"
)
_REPO = Path(__file__).resolve().parents[1]
_CODE = (
    "tools/localvqe_offline_eval.py",
    "tools/english_asr_benchmark.py",
    "core/engines/_aec.py",
    "core/engines/_apm.py",
    "tools/audio_eval/metrics.py",
)


class EvalError(RuntimeError):
    """Fixed-detail evaluation refusal."""


class _PortableIO:
    """Small strict file/JSON boundary; no platform-native imports."""

    _DIGEST = re.compile(r"[a-f0-9]{64}\Z")

    @staticmethod
    def _canonical(value):
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")

    @staticmethod
    def _json(raw):
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise EvalError()
                result[key] = value
            return result

        def invalid(_value):
            raise EvalError()

        return json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=invalid,
        )

    @staticmethod
    def _absolute(path):
        candidate = Path(os.path.abspath(path))
        if candidate.resolve(strict=True) != candidate:
            raise EvalError()
        return candidate

    @staticmethod
    def _identity(value):
        return (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_nlink,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    @classmethod
    def _file_digest(cls, path, maximum, *, allow_empty=False):
        path = cls._absolute(path)
        before = path.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not (0 if allow_empty else 1) <= before.st_size <= maximum
        ):
            raise EvalError()
        digest = hashlib.sha256()
        total = 0
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as stream:
            if cls._identity(os.fstat(stream.fileno())) != cls._identity(before):
                raise EvalError()
            while chunk := stream.read(1048576):
                total += len(chunk)
                if total > maximum:
                    raise EvalError()
                digest.update(chunk)
            if cls._identity(os.fstat(stream.fileno())) != cls._identity(before):
                raise EvalError()
        if total != before.st_size or cls._identity(path.lstat()) != cls._identity(
            before
        ):
            raise EvalError()
        return digest.hexdigest(), total

    @classmethod
    def _read(cls, path, maximum):
        path = cls._absolute(path)
        before = path.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 0 < before.st_size <= maximum
        ):
            raise EvalError()
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as stream:
            if cls._identity(os.fstat(stream.fileno())) != cls._identity(before):
                raise EvalError()
            raw = stream.read(maximum + 1)
            if cls._identity(os.fstat(stream.fileno())) != cls._identity(before):
                raise EvalError()
        if len(raw) != before.st_size or cls._identity(path.lstat()) != cls._identity(
            before
        ):
            raise EvalError()
        return raw

    @classmethod
    def _write_new(cls, path, value):
        raw = cls._canonical(value) + b"\n"
        if len(raw) > 512 * 1024:
            raise EvalError()
        fd = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())

    @staticmethod
    def _deny_network():
        def denied(*_args, **_kwargs):
            raise EvalError()

        socket.socket.connect = denied
        socket.socket.connect_ex = denied
        socket.socket.sendto = denied
        socket.create_connection = denied

    @staticmethod
    def _stop_group(process):
        if process.poll() is not None:
            return
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        process.wait(timeout=2)


io = _PortableIO


def _affinity():
    if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
        mask = os.sched_getaffinity(0)
        os.sched_setaffinity(0, {min(mask)})
        return True
    # No implicit platform dependency or pretend enforcement on other hosts.
    return False


def _hash(path, limit, *, allow_empty=False):
    return io._file_digest(Path(path), limit, allow_empty=allow_empty)


def load_assets(path: Path) -> dict:
    value = io._json(io._read(path, 65536))
    keys = {
        "schema_version",
        "localvqe_commit",
        "ggml_commit",
        "weight_revision",
        "native_dir",
        "library",
        "files",
        "models",
    }
    if (
        type(value) is not dict
        or set(value) != keys
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["localvqe_commit"] != CODE_COMMIT
        or value["ggml_commit"] != GGML_COMMIT
        or value["weight_revision"] != WEIGHT_REVISION
        or value["library"]
        not in {
            "liblocalvqe.so.0.1.0",
            "liblocalvqe.dylib",
            "localvqe.dll",
            "liblocalvqe.dll",
        }
        or type(value["models"]) is not dict
        or set(value["models"]) != set(WEIGHTS)
        or type(value["files"]) is not list
        or not 3 <= len(value["files"]) <= 64
    ):
        raise EvalError()
    directory = io._absolute(value["native_dir"])
    before = directory.lstat()
    names = set()
    total = 0
    for item in value["files"]:
        if (
            type(item) is not dict
            or set(item) != {"name", "bytes", "sha256"}
            or type(item["name"]) is not str
            or not _NATIVE_NAME.fullmatch(item["name"])
            or item["name"] in names
            or type(item["bytes"]) is not int
            or not 0 < item["bytes"] <= 64 * 1024**2
            or type(item["sha256"]) is not str
            or not io._DIGEST.fullmatch(item["sha256"])
        ):
            raise EvalError()
        names.add(item["name"])
        if _hash(directory / item["name"], 64 * 1024**2) != (
            item["sha256"],
            item["bytes"],
        ):
            raise EvalError()
        total += item["bytes"]
    actual_names = set()
    with os.scandir(directory) as entries:
        for entry in entries:
            actual_names.add(entry.name)
            if len(actual_names) > 64:
                raise EvalError()
    if (
        total > 256 * 1024**2
        or value["library"] not in names
        or actual_names != names
        or io._identity(before) != io._identity(directory.lstat())
    ):
        raise EvalError()
    for model_id, (size, digest) in WEIGHTS.items():
        if type(value["models"][model_id]) is not str or _hash(
            value["models"][model_id], 4 * 1024**2
        ) != (digest, size):
            raise EvalError()
    return value


def bindings(path: Path) -> dict:
    assets = load_assets(path)
    apm = []
    try:
        distribution = metadata.distribution("livekit")
        apm_version = distribution.version
        for entry in distribution.files or ():
            if str(entry).endswith((".so", ".dylib", ".dll", ".py")):
                digest, size = _hash(
                    distribution.locate_file(entry),
                    64 * 1024**2,
                    allow_empty=str(entry).endswith(".py"),
                )
                apm.append({"name": str(entry), "sha256": digest, "bytes": size})
                if len(apm) > 512 or sum(item["bytes"] for item in apm) > 128 * 1024**2:
                    raise EvalError()
    except metadata.PackageNotFoundError:
        apm_version = None
    return {
        "assets_sha256": _hash(path, 65536)[0],
        "native_files": assets["files"],
        "model_files": [
            {"id": key, "bytes": item[0], "sha256": item[1]}
            for key, item in WEIGHTS.items()
        ],
        "code": [
            {"name": name, "sha256": _hash(_REPO / name, 2 * 1024**2)[0]}
            for name in _CODE
        ],
        "numpy_version": np.__version__,
        "livekit_version": apm_version,
        "livekit_files": apm,
    }


@dataclass(frozen=True)
class SignalCase:
    id: str
    microphone: np.ndarray
    reference: np.ndarray
    near: np.ndarray


def signal_cases() -> tuple[SignalCase, ...]:
    """Deterministic harmonic/modulated signals, not recorded speech."""
    t = np.arange(SAMPLES, dtype=np.float64) / SAMPLE_RATE
    far_phase = 2 * np.pi * (143 * t + 4 * t * t)
    near_phase = 2 * np.pi * (217 * t - 3 * t * t)
    far = (
        0.16 * np.sin(far_phase)
        + 0.06 * np.sin(2.13 * far_phase)
        + 0.03 * np.sin(3.71 * far_phase)
    ) * (0.65 + 0.35 * np.sin(2 * np.pi * 2.7 * t) ** 2)
    near = (0.10 * np.sin(near_phase) + 0.045 * np.sin(2.47 * near_phase)) * (
        0.6 + 0.4 * np.sin(2 * np.pi * 3.3 * t) ** 2
    )
    index = np.arange(SAMPLES, dtype=np.float64)

    def echo(delay):
        result = np.zeros(SAMPLES, dtype=np.float64)
        for tail, gain in ((0, 0.6), (64, 0.2), (192, 0.1)):
            result += gain * np.interp(
                index - delay - tail, index, far, left=0, right=0
            )
        return result

    linear = echo(80)
    drift = echo(np.linspace(40, 160, SAMPLES))
    nonlinear = 0.16 * np.tanh(linear / 0.16)
    talking = near.copy()
    talking[: 2 * SAMPLE_RATE] = 0
    zero = np.zeros(SAMPLES, dtype=np.float32)

    def f32(x):
        return np.asarray(x, dtype=np.float32)

    return (
        SignalCase("far_linear", f32(linear), f32(far), zero.copy()),
        SignalCase("far_nonlinear", f32(nonlinear), f32(far), zero.copy()),
        SignalCase("far_drift", f32(drift), f32(far), zero.copy()),
        SignalCase("near_only", f32(near), zero.copy(), f32(near)),
        SignalCase("doubletalk", f32(linear + talking), f32(far), f32(talking)),
    )


def signal_binding(cases) -> str:
    digest = hashlib.sha256()
    for case in cases:
        digest.update(case.id.encode("ascii"))
        for array in (case.microphone, case.reference, case.near):
            digest.update(array.astype("<f4", copy=False).tobytes())
    return digest.hexdigest()


class LocalVqe:
    def __init__(self, assets, model_id):
        self.ctx = 0
        self.dll_directory = None
        if os.name == "nt" and hasattr(os, "add_dll_directory"):
            self.dll_directory = os.add_dll_directory(assets["native_dir"])
        try:
            self.library = ctypes.CDLL(
                str(Path(assets["native_dir"]) / assets["library"])
            )
        except BaseException:
            if self.dll_directory is not None:
                self.dll_directory.close()
            raise
        lib = self.library
        signatures = {
            "localvqe_options_new": (ctypes.c_size_t, []),
            "localvqe_options_free": (None, [ctypes.c_size_t]),
            "localvqe_options_set_model_path": (
                ctypes.c_int,
                [ctypes.c_size_t, ctypes.c_char_p],
            ),
            "localvqe_options_set_backend": (
                ctypes.c_int,
                [ctypes.c_size_t, ctypes.c_char_p],
            ),
            "localvqe_options_set_threads": (
                ctypes.c_int,
                [ctypes.c_size_t, ctypes.c_int],
            ),
            "localvqe_new_with_options": (ctypes.c_size_t, [ctypes.c_size_t]),
            "localvqe_free": (None, [ctypes.c_size_t]),
            "localvqe_reset": (None, [ctypes.c_size_t]),
            "localvqe_sample_rate": (ctypes.c_int, [ctypes.c_size_t]),
            "localvqe_hop_length": (ctypes.c_int, [ctypes.c_size_t]),
            "localvqe_fft_size": (ctypes.c_int, [ctypes.c_size_t]),
            "localvqe_set_noise_gate": (
                ctypes.c_int,
                [ctypes.c_size_t, ctypes.c_int, ctypes.c_float],
            ),
            "localvqe_process_frame_f32": (
                ctypes.c_int,
                [
                    ctypes.c_size_t,
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.c_int,
                    ctypes.POINTER(ctypes.c_float),
                ],
            ),
        }
        try:
            for name, (result, arguments) in signatures.items():
                function = getattr(lib, name)
                function.restype, function.argtypes = result, arguments
        except BaseException:
            self.close()
            raise
        options = lib.localvqe_options_new()
        if not options:
            self.close()
            raise EvalError()
        try:
            try:
                if (
                    lib.localvqe_options_set_model_path(
                        options, os.fsencode(assets["models"][model_id])
                    )
                    or lib.localvqe_options_set_backend(options, b"CPU")
                    or lib.localvqe_options_set_threads(options, 1)
                ):
                    raise EvalError()
                self.ctx = lib.localvqe_new_with_options(options)
            finally:
                lib.localvqe_options_free(options)
        except BaseException:
            self.close()
            raise
        if not self.ctx:
            self.close()
            raise EvalError()
        try:
            if (
                lib.localvqe_sample_rate(self.ctx) != SAMPLE_RATE
                or lib.localvqe_hop_length(self.ctx) != 256
                or lib.localvqe_fft_size(self.ctx) != 512
                or lib.localvqe_set_noise_gate(self.ctx, 0, -45.0)
            ):
                raise EvalError()
            lib.localvqe_reset(self.ctx)
        except BaseException:
            self.close()
            raise

    def process(self, microphone, reference):
        if len(microphone) != CALLBACK_SAMPLES or len(reference) != CALLBACK_SAMPLES:
            raise EvalError()
        output = np.empty(CALLBACK_SAMPLES, dtype=np.float32)
        pointer = ctypes.POINTER(ctypes.c_float)
        for offset in range(0, CALLBACK_SAMPLES, 256):
            if self.library.localvqe_process_frame_f32(
                self.ctx,
                microphone[offset:].ctypes.data_as(pointer),
                reference[offset:].ctypes.data_as(pointer),
                256,
                output[offset:].ctypes.data_as(pointer),
            ):
                raise EvalError()
        return output

    def close(self):
        if self.ctx:
            self.library.localvqe_free(self.ctx)
            self.ctx = 0
        if self.dll_directory is not None:
            self.dll_directory.close()
            self.dll_directory = None


class _ObservedImpl:
    def __init__(self, impl):
        self.impl, self.output = impl, None
        self.errors = self.guard_fallbacks = 0

    def process(self, near, far):
        try:
            self.output = self.impl.process(near, far)
            return self.output
        except Exception:
            self.errors += 1
            raise

    def reset(self):
        self.impl.reset()


class ProductionAec:
    def __init__(self, model_id):
        if model_id == "nlms-default":
            impl = _FDAFAdaptiveFilter(512, mu=0.3, leak=0.9999, doubletalk_freeze=True)
        else:
            from core.engines._apm import _WebRTCAPM

            impl = _WebRTCAPM(
                echo_cancellation=True,
                noise_suppression=model_id == "apm-ns-on",
                high_pass_filter=True,
                gain_control=False,
                stream_delay_ms=0,
                sample_rate=SAMPLE_RATE,
            )
        self.observed = _ObservedImpl(impl)
        self.engine = EchoCanceller(self.observed, sample_rate=SAMPLE_RATE)

    def process(self, near, far):
        output = self.engine.process_16k(near, far)
        if output is not self.observed.output:
            self.observed.guard_fallbacks += 1
        if self.observed.errors:
            raise EvalError()
        return output

    def close(self):
        # The shipped APM Python API exposes no explicit close receipt. Native
        # destruction/return is not attested; isolated process exit is the bound.
        self.engine = None
        self.observed.impl = None


def factory(assets, model_id):
    return (
        LocalVqe(assets, model_id) if model_id in WEIGHTS else ProductionAec(model_id)
    )


def feed_case(adapter, case, *, observe=lambda: None):
    pieces = []
    wall = cpu = 0.0
    for start in range(0, SAMPLES, CALLBACK_SAMPLES):
        near = np.array(case.microphone[start : start + CALLBACK_SAMPLES], copy=True)
        far = np.array(case.reference[start : start + CALLBACK_SAMPLES], copy=True)
        before_cpu, before_wall = time.process_time(), time.perf_counter()
        output = adapter.process(near, far)
        wall += time.perf_counter() - before_wall
        cpu += time.process_time() - before_cpu
        if (
            not isinstance(output, np.ndarray)
            or output.ndim != 1
            or output.dtype.kind != "f"
            or not np.all(np.isfinite(output))
            or len(output) > CALLBACK_SAMPLES + 512
        ):
            raise EvalError()
        pieces.append(np.array(output, dtype=np.float32, copy=True))
        observe()
    result = np.concatenate(pieces)
    if result.shape != case.microphone.shape:
        raise EvalError()
    return result, wall, cpu


def metrics(case, output):
    microphone = case.microphone[SCORE_START:]
    target = case.near[SCORE_START:]
    enhanced = output[SCORE_START:]
    near_active = bool(np.any(target))
    muted = bool(near_active and signal_power(enhanced) == 0)
    return {
        "far_only_attenuation_db": None
        if near_active
        else capped_db_ratio(signal_power(microphone), signal_power(enhanced)),
        "near_projection_gain": projection_gain(enhanced, target)
        if near_active
        else None,
        "near_si_sdr_db": si_sdr_db(enhanced, target)
        if near_active and not muted
        else None,
        "near_si_sdr_improvement_db": si_sdr_db(enhanced, target)
        - si_sdr_db(microphone, target)
        if case.id == "doubletalk" and not muted
        else None,
        "near_output_zero": muted,
        "output_peak": float(np.max(np.abs(output))),
        "output_fraction_over_one": float(np.mean(np.abs(output) > 1)),
    }


def benchmark_cell(
    assets,
    model_id,
    *,
    adapter_factory=factory,
    observe=lambda: None,
    phase=lambda _value: None,
):
    cases = signal_cases()
    cells = []
    for case in cases:
        phase("load")
        started = time.perf_counter()
        adapter = adapter_factory(assets, model_id)
        load = time.perf_counter() - started
        try:
            observe()
            phase("frames")
            output, wall, cpu = feed_case(adapter, case, observe=observe)
            phase("metrics")
            cell = {
                "case": case.id,
                "samples": SAMPLES,
                "output_samples": len(output),
                "load_seconds": load,
                "process_wall_seconds": wall,
                "process_cpu_seconds": cpu,
                "process_rtf": wall / (SAMPLES / SAMPLE_RATE),
                "guard_fallbacks": getattr(
                    getattr(adapter, "observed", None), "guard_fallbacks", 0
                ),
                "metrics": metrics(case, output),
            }
            cells.append(cell)
        except BaseException:
            try:
                adapter.close()
            except BaseException:
                pass
            raise
        else:
            phase("close")
            adapter.close()
    return {
        "model_id": model_id,
        "status": "complete",
        "signal_digest": signal_binding(cases),
        "cells": cells,
    }


def _worker(request_path):
    phase = "request"
    request = None
    try:
        request = io._json(io._read(Path(request_path), 65536))
        if (
            type(request) is not dict
            or set(request) != {"assets", "bindings", "model_id", "result", "timeout"}
            or request["model_id"] not in MODELS
            or type(request["timeout"]) is not int
            or not 10 <= request["timeout"] <= 90
        ):
            raise EvalError()
        if resource is not None:
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            resource.setrlimit(
                resource.RLIMIT_CPU, (request["timeout"], request["timeout"] + 1)
            )
            resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
        affinity_applied = _affinity()
        io._deny_network()
        phase = "bindings"
        assets_path = Path(request["assets"])
        if bindings(assets_path) != request["bindings"]:
            raise EvalError()
        assets = load_assets(assets_path)
        sampler = None
        if sys.platform.startswith("linux") and affinity_applied:
            from tools import english_asr_benchmark as affinity_io

            sampler = affinity_io._AffinitySamples(
                set(os.sched_getaffinity(0)), affinity_io._sample_thread_affinity
            )

        def update(value):
            nonlocal phase
            phase = value

        result = benchmark_cell(
            assets,
            request["model_id"],
            observe=sampler.observe if sampler is not None else lambda: None,
            phase=update,
        )
        phase = "final_bindings"
        if bindings(assets_path) != request["bindings"]:
            raise EvalError()
        receipt = sampler.receipt() if sampler is not None else None
        if receipt is not None:
            receipt["scope"] = "after_each_model_load_and_processing_callback"
        peak = None
        if resource is not None:
            peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            if sys.platform == "darwin":
                peak //= 1024
        result.update(
            {
                "bindings_sha256": hashlib.sha256(
                    io._canonical(request["bindings"])
                ).hexdigest(),
                "peak_rss_kib": peak,
                "cpu_affinity_applied": affinity_applied,
                "posix_resource_limits_applied": resource is not None,
                "thread_affinity": receipt,
            }
        )
        io._write_new(Path(request["result"]), result)
        return 0
    except BaseException:
        if type(request) is dict and type(request.get("result")) is str:
            try:
                io._write_new(
                    Path(request["result"]),
                    {
                        "model_id": request["model_id"],
                        "status": "failed",
                        "phase": phase,
                    },
                )
            except BaseException:
                pass
        return 2


def validate_result(value, model_id, expected_binding):
    if type(value) is not dict or value.get("model_id") != model_id:
        raise EvalError()
    if value.get("status") == "failed":
        if (
            set(value) != {"model_id", "status", "phase"}
            or value["phase"] not in PHASES
        ):
            raise EvalError()
        return
    if (
        set(value)
        != {
            "model_id",
            "status",
            "signal_digest",
            "cells",
            "bindings_sha256",
            "peak_rss_kib",
            "cpu_affinity_applied",
            "posix_resource_limits_applied",
            "thread_affinity",
        }
        or value["status"] != "complete"
        or value["signal_digest"] != signal_binding(signal_cases())
        or value["bindings_sha256"] != expected_binding
        or (
            value["peak_rss_kib"] is not None
            and (
                type(value["peak_rss_kib"]) is not int
                or not 1 <= value["peak_rss_kib"] <= 4 * 1024**2
            )
        )
        or type(value["cpu_affinity_applied"]) is not bool
        or type(value["posix_resource_limits_applied"]) is not bool
        or type(value["cells"]) is not list
        or len(value["cells"]) != len(CASES)
    ):
        raise EvalError()
    receipt = value["thread_affinity"]
    if receipt is not None:
        if (
            type(receipt) is not dict
            or not value["cpu_affinity_applied"]
            or receipt.get("scope") != "after_each_model_load_and_processing_callback"
            or receipt.get("caller_mask_logical_cpus") != 1
        ):
            raise EvalError()
        from tools import english_asr_benchmark as affinity_io

        legacy_receipt = dict(receipt, scope="after_model_load_and_each_decode_samples")
        affinity_io._validate_thread_affinity(
            legacy_receipt,
            len(CASES) * (1 + SAMPLES // CALLBACK_SAMPLES),
            complete=True,
        )
    elif sys.platform.startswith("linux"):
        raise EvalError()
    metric_keys = {
        "far_only_attenuation_db",
        "near_projection_gain",
        "near_si_sdr_db",
        "near_si_sdr_improvement_db",
        "near_output_zero",
        "output_peak",
        "output_fraction_over_one",
    }
    for case_id, cell in zip(CASES, value["cells"]):
        if (
            type(cell) is not dict
            or set(cell)
            != {
                "case",
                "samples",
                "output_samples",
                "load_seconds",
                "process_wall_seconds",
                "process_cpu_seconds",
                "process_rtf",
                "guard_fallbacks",
                "metrics",
            }
            or cell["case"] != case_id
            or type(cell["samples"]) is not int
            or cell["samples"] != SAMPLES
            or type(cell["output_samples"]) is not int
            or cell["output_samples"] != SAMPLES
            or type(cell["guard_fallbacks"]) is not int
            or not 0 <= cell["guard_fallbacks"] <= SAMPLES // CALLBACK_SAMPLES
            or type(cell["metrics"]) is not dict
            or set(cell["metrics"]) != metric_keys
        ):
            raise EvalError()
        for key in (
            "load_seconds",
            "process_wall_seconds",
            "process_cpu_seconds",
            "process_rtf",
        ):
            number = cell[key]
            if (
                type(number) not in {int, float}
                or not math.isfinite(number)
                or not 0 <= number <= 90
            ):
                raise EvalError()
        if not math.isclose(
            cell["process_rtf"],
            cell["process_wall_seconds"] / (SAMPLES / SAMPLE_RATE),
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise EvalError()
        for key, number in cell["metrics"].items():
            if key == "near_output_zero":
                if type(number) is not bool or (
                    number and case_id not in {"near_only", "doubletalk"}
                ):
                    raise EvalError()
                continue
            muted = cell["metrics"]["near_output_zero"] is True
            nullable = (
                (
                    key == "far_only_attenuation_db"
                    and case_id in {"near_only", "doubletalk"}
                )
                or (
                    key in {"near_projection_gain", "near_si_sdr_db"}
                    and case_id not in {"near_only", "doubletalk"}
                )
                or (key == "near_si_sdr_improvement_db" and case_id != "doubletalk")
                or (muted and key in {"near_si_sdr_db", "near_si_sdr_improvement_db"})
            )
            if nullable:
                if number is not None:
                    raise EvalError()
            elif (
                type(number) not in {int, float}
                or not math.isfinite(number)
                or abs(number) > 1000
            ):
                raise EvalError()
        if (
            not 0 <= cell["metrics"]["output_fraction_over_one"] <= 1
            or cell["metrics"]["output_peak"] < 0
        ):
            raise EvalError()


def run_cell(assets_path, model_id, scratch, timeout=45):
    if model_id not in MODELS or type(timeout) is not int or not 10 <= timeout <= 90:
        raise EvalError()
    bound = bindings(assets_path)
    digest = hashlib.sha256(io._canonical(bound)).hexdigest()
    directory = scratch / str(uuid.uuid4())
    directory.mkdir(mode=0o700)
    request = directory / "request.json"
    result = directory / "aggregate.json"
    io._write_new(
        request,
        {
            "assets": str(assets_path),
            "bindings": bound,
            "model_id": model_id,
            "result": str(result),
            "timeout": timeout,
        },
    )
    assets = load_assets(assets_path)
    environment = {
        "PATH": os.defpath,
        "LANG": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "GGML_NTHREADS": "1",
        "LOCALVQE_ALLOW_UNHASHED": "0",
        "LD_LIBRARY_PATH": assets["native_dir"],
        "DYLD_LIBRARY_PATH": assets["native_dir"],
        "GGML_BACKEND_PATH": assets["native_dir"],
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
    }
    if os.name == "nt" and "SYSTEMROOT" in os.environ:
        environment["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    child = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-B",
            str(Path(__file__).resolve()),
            "--worker-request",
            str(request),
        ],
        cwd=directory,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=os.name == "posix",
    )
    try:
        child.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        io._stop_group(child)
        return {"model_id": model_id, "status": "failed", "phase": "timeout"}
    if not result.exists():
        return {"model_id": model_id, "status": "failed", "phase": "worker_result"}
    value = io._json(io._read(result, 65536))
    validate_result(value, model_id, digest)
    if bindings(assets_path) != bound or (
        child.returncode != 0 and value["status"] == "complete"
    ):
        raise EvalError()
    return value


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise EvalError()


def main(argv=None):
    parser = _Parser(description=__doc__)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--scratch", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker-request", type=Path, help=argparse.SUPPRESS)
    try:
        args = parser.parse_args(argv)
        if args.worker_request:
            return _worker(args.worker_request)
        if args.assets is None or args.scratch is None or args.output is None:
            raise EvalError()
        scratch = io._absolute(args.scratch)
        bound = bindings(args.assets)
        results = [run_cell(args.assets, model_id, scratch) for model_id in MODELS]
        expected_binding = hashlib.sha256(io._canonical(bound)).hexdigest()
        if bindings(args.assets) != bound or any(
            result["status"] == "complete"
            and result["bindings_sha256"] != expected_binding
            for result in results
        ):
            raise EvalError()
        report = {
            "schema_version": 1,
            "kind": "synthetic-aec-candidate-comparison",
            "sample_rate_hz": SAMPLE_RATE,
            "callback_samples": CALLBACK_SAMPLES,
            "score_start_samples": SCORE_START,
            "signal_digest": signal_binding(signal_cases()),
            "bindings": bound,
            "native_localvqe_threads_requested": 1,
            "caller_affinity_cpus_requested": 1,
            "claims": {
                "synthetic_only": True,
                "raw_recordings_used": False,
                "audio_device_opened": False,
                "python_socket_guard": True,
                "os_network_sandbox_attested": False,
                "live_or_phone_quality_validated": False,
                "default_adoption": False,
                "apm_native_thread_count_controlled": False,
                "apm_native_destruction_attested": False,
            },
            "results": results,
        }
        io._write_new(args.output, report)
        print(
            json.dumps(
                {
                    "ok": all(cell["status"] == "complete" for cell in results),
                    "complete_cells": sum(
                        cell["status"] == "complete" for cell in results
                    ),
                    "total_cells": len(results),
                },
                sort_keys=True,
            )
        )
        return 0 if all(cell["status"] == "complete" for cell in results) else 2
    except Exception:
        print(
            json.dumps(
                {"ok": False, "error": "localvqe_offline_evaluation_failed"},
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
