"""Private paired AFTER-HOST AEC replay: execution/energy, never a quality gate.

Only complete schema-2 diagnostic bundles with contiguous, equal per-frame
pre-gain/reference ranges are accepted. No transcript, log, or source wave is
copied or printed. The synthetic evaluator's source and receipts stay separate.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import hashlib
import io as byte_io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
import wave

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from core.diagnostic_bundle import validate_manifest
from tools import localvqe_offline_eval as base
from tools.audio_eval.metrics import capped_db_ratio, signal_power, signal_rms

MAX_SAMPLES = 180 * 16000
MAX_TIMELINE = 16 * 1024**2
ROLES = ("model_pre_gain_tap", "playback_reference_reader_snapshot")
PHASES = {
    "request",
    "bindings",
    "pair",
    "load",
    "frames",
    "metrics",
    "close",
    "final_bindings",
    "timeout",
    "worker_result",
}
_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Pair:
    microphone: np.ndarray
    reference: np.ndarray
    ranges: tuple[tuple[int, int], ...]
    receipt: dict


def _bound_artifact(parent, entry, maximum):
    if (
        type(entry) is not dict
        or type(entry.get("file")) is not str
        or Path(entry["file"]).name != entry["file"]
    ):
        raise base.EvalError()
    raw = base.io._read(parent / entry["file"], maximum)
    if len(raw) != entry.get("bytes") or hashlib.sha256(raw).hexdigest() != entry.get(
        "sha256"
    ):
        raise base.EvalError()
    return raw


def _decode(raw, samples):
    with wave.open(byte_io.BytesIO(raw), "rb") as audio:
        if (
            audio.getnchannels() != 1
            or audio.getsampwidth() != 2
            or audio.getframerate() != 16000
            or audio.getcomptype() != "NONE"
            or audio.getnframes() != samples
        ):
            raise base.EvalError()
        pcm = audio.readframes(samples + 1)
    if len(pcm) != samples * 2:
        raise base.EvalError()
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def load_pair(manifest_path: Path, expected_sha: str) -> Pair:
    if type(expected_sha) is not str or not base.io._DIGEST.fullmatch(expected_sha):
        raise base.EvalError()
    raw = base.io._read(manifest_path, 256 * 1024)
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise base.EvalError()
    manifest = base.io._json(raw)
    if type(manifest) is not dict or type(manifest.get("tracks")) is not dict:
        raise base.EvalError()
    entries = list(manifest["tracks"].values())
    if len(entries) != 4:
        raise base.EvalError()
    limits = [(entry, MAX_SAMPLES * 2 + 65536) for entry in entries]
    limits.extend(
        (
            (manifest.get("timeline"), MAX_TIMELINE),
            (manifest.get("final_model_input"), 8 * 1024**2),
        )
    )
    for entry, maximum in limits:
        if (
            type(entry) is not dict
            or type(entry.get("bytes")) is not int
            or not 0 <= entry["bytes"] <= maximum
            or type(entry.get("file")) is not str
            or Path(entry["file"]).name != entry["file"]
            or (manifest_path.parent / entry["file"]).lstat().st_size != entry["bytes"]
        ):
            raise base.EvalError()
    if (
        type(manifest.get("frame_count")) is not int
        or not 0 < manifest["frame_count"] <= 10000
    ):
        raise base.EvalError()
    if (
        type(manifest) is not dict
        or manifest.get("schema_version") != 2
        or manifest.get("complete") is not True
        or manifest.get("clean_shutdown") is not True
        or manifest.get("failure_codes") != []
        or manifest.get("sample_rate_hz") != 16000
        or not validate_manifest(manifest_path)
    ):
        raise base.EvalError()
    provenance = manifest["provenance"]
    if (
        provenance.get("physical_raw_mic_present") is not False
        or provenance.get("native_host_pcm_present") is not False
        or provenance.get("model_pre_gain_tap_may_include_host_processing") is not True
        or provenance.get("model_pre_gain_tap_includes_application_resampling")
        is not True
        or provenance.get("playback_reference_preserves_reader_snapshot") is not True
    ):
        raise base.EvalError()
    tracks = [manifest["tracks"][role] for role in ROLES]
    samples = tracks[0]["samples"]
    if (
        type(samples) is not int
        or not 0 < samples <= MAX_SAMPLES
        or samples % 2560
        or tracks[1]["samples"] != samples
    ):
        raise base.EvalError()
    body = _bound_artifact(manifest_path.parent, manifest["timeline"], MAX_TIMELINE)
    ranges = []
    previous = None
    source_group = None
    cursor = 0
    records = body.splitlines()
    if len(records) > 50000:
        raise base.EvalError()
    for line in records:
        if not 0 < len(line) <= 65536:
            raise base.EvalError()
        record = base.io._json(line)
        if record.get("kind") != "frame":
            continue
        if record["frame_index"] != len(ranges):
            raise base.EvalError()
        left, right = (record["track_ranges"][role] for role in ROLES)
        if (
            left != right
            or left["sample_start"] != cursor
            or not 0 < left["sample_end"] - cursor <= 16000
        ):
            raise base.EvalError()
        coordinate = record["coordinate"]
        group = tuple(
            coordinate[key]
            for key in (
                "capture_epoch",
                "source_generation",
                "capture_generation",
                "sample_rate_hz",
            )
        )
        if source_group is None:
            source_group = group
        if group != source_group or coordinate.get("gap_reason") is not None:
            raise base.EvalError()
        if previous is not None and (
            coordinate["sequence"] != previous["sequence"] + 1
            or coordinate["source_sample_start"] != previous["source_sample_end"]
        ):
            raise base.EvalError()
        ranges.append((cursor, left["sample_end"]))
        cursor = left["sample_end"]
        previous = coordinate
    if cursor != samples or len(ranges) != manifest["frame_count"]:
        raise base.EvalError()
    microphone, reference = [
        _decode(
            _bound_artifact(manifest_path.parent, entry, MAX_SAMPLES * 2 + 65536),
            samples,
        )
        for entry in tracks
    ]
    # Reload after all reads; no result can bind a replaced source manifest.
    if base.io._read(manifest_path, 256 * 1024) != raw:
        raise base.EvalError()
    receipt = {
        "manifest_sha256": expected_sha,
        "timeline_sha256": manifest["timeline"]["sha256"],
        "pre_gain_sha256": tracks[0]["sha256"],
        "reader_reference_sha256": tracks[1]["sha256"],
        "samples": samples,
        "frame_count": len(ranges),
        "sample_rate_hz": 16000,
        "minimum_frame_samples": min(end - start for start, end in ranges),
        "maximum_frame_samples": max(end - start for start, end in ranges),
        "capture_source_groups": 1,
        "capture_gaps": 0,
    }
    return Pair(microphone, reference, tuple(ranges), receipt)


def pair_digest(pair):
    return hashlib.sha256(base.io._canonical(pair.receipt)).hexdigest()


def bindings(assets_path):
    return {
        "synthetic_components": base.bindings(assets_path),
        "recorded_tool_sha256": base._hash(Path(__file__).resolve(), 2 * 1024**2)[0],
        "diagnostic_verifier_sha256": base._hash(
            _ROOT / "core/diagnostic_bundle.py", 2 * 1024**2
        )[0],
    }


class BufferedLocalVqe(base.LocalVqe):
    """Preserve original capture calls; frame only the native 256-sample API."""

    def __init__(self, assets, model_id):
        super().__init__(assets, model_id)
        self.near = self.far = np.zeros(0, dtype=np.float32)

    def process(self, microphone, reference):
        self.near = np.concatenate((self.near, microphone))
        self.far = np.concatenate((self.far, reference))
        count = len(self.near) // 256 * 256
        output = np.empty(count, dtype=np.float32)
        pointer = ctypes.POINTER(ctypes.c_float)
        for offset in range(0, count, 256):
            if self.library.localvqe_process_frame_f32(
                self.ctx,
                self.near[offset:].ctypes.data_as(pointer),
                self.far[offset:].ctypes.data_as(pointer),
                256,
                output[offset:].ctypes.data_as(pointer),
            ):
                raise base.EvalError()
        self.near, self.far = self.near[count:].copy(), self.far[count:].copy()
        return output


def factory(assets, model_id):
    return (
        BufferedLocalVqe(assets, model_id)
        if model_id in base.WEIGHTS
        else base.ProductionAec(model_id)
    )


def benchmark_pair(
    assets,
    model_id,
    pair,
    *,
    adapter_factory=factory,
    observe=lambda: None,
    phase=lambda _value: None,
):
    phase("load")
    started = time.perf_counter()
    adapter = adapter_factory(assets, model_id)
    load = time.perf_counter() - started
    try:
        observe()
        phase("frames")
        pieces = []
        wall = cpu = 0.0
        for start, end in pair.ranges:
            near, far = (
                pair.microphone[start:end].copy(),
                pair.reference[start:end].copy(),
            )
            before_cpu, before_wall = time.process_time(), time.perf_counter()
            output = adapter.process(near, far)
            wall += time.perf_counter() - before_wall
            cpu += time.process_time() - before_cpu
            if (
                not isinstance(output, np.ndarray)
                or output.ndim != 1
                or output.dtype.kind != "f"
                or len(output) > len(near) + 512
                or not np.all(np.isfinite(output))
            ):
                raise base.EvalError()
            pieces.append(np.array(output, dtype=np.float32, copy=True))
            observe()
        output = np.concatenate(pieces)
        if output.shape != pair.microphone.shape:
            raise base.EvalError()
        phase("metrics")
        input_power, output_power = signal_power(pair.microphone), signal_power(output)
        measured = {
            "microphone_rms": signal_rms(pair.microphone),
            "reference_rms": signal_rms(pair.reference),
            "output_rms": signal_rms(output),
            "input_output_power_change_db": None
            if input_power == output_power == 0
            else capped_db_ratio(output_power, input_power),
            "output_peak": float(np.max(np.abs(output))),
            "output_fraction_over_one": float(np.mean(np.abs(output) > 1)),
            "reference_active_frame_fraction": sum(
                signal_rms(pair.reference[start:end]) > 1e-5
                for start, end in pair.ranges
            )
            / len(pair.ranges),
            "output_zero": output_power == 0,
        }
        result = {
            "model_id": model_id,
            "status": "complete",
            "input_binding": pair_digest(pair),
            "samples": len(output),
            "frame_count": len(pair.ranges),
            "load_seconds": load,
            "process_wall_seconds": wall,
            "process_cpu_seconds": cpu,
            "process_rtf": wall / (len(output) / 16000),
            "guard_fallbacks": getattr(
                getattr(adapter, "observed", None), "guard_fallbacks", 0
            ),
            "metrics": measured,
        }
    except BaseException:
        try:
            adapter.close()
        except BaseException:
            pass
        raise
    else:
        phase("close")
        adapter.close()
    return result


def _worker(request_path):
    request = None
    phase = "request"
    try:
        request = base.io._json(base.io._read(request_path, 256 * 1024))
        if (
            type(request) is not dict
            or set(request)
            != {
                "assets",
                "manifest",
                "manifest_sha256",
                "pair_binding",
                "bindings",
                "model_id",
                "result",
                "timeout",
            }
            or request["model_id"] not in base.MODELS
            or type(request["timeout"]) is not int
            or not 30 <= request["timeout"] <= 180
        ):
            raise base.EvalError()
        if base.resource is not None:
            for name, limits in (
                ("RLIMIT_CORE", (0, 0)),
                ("RLIMIT_CPU", (request["timeout"], request["timeout"] + 1)),
                ("RLIMIT_AS", (4 * 1024**3, 4 * 1024**3)),
            ):
                base.resource.setrlimit(getattr(base.resource, name), limits)
        affinity_applied = base._affinity()
        base.io._deny_network()
        phase = "bindings"
        assets_path = Path(request["assets"])
        if bindings(assets_path) != request["bindings"]:
            raise base.EvalError()
        assets = base.load_assets(assets_path)
        phase = "pair"
        pair = load_pair(Path(request["manifest"]), request["manifest_sha256"])
        if pair_digest(pair) != request["pair_binding"]:
            raise base.EvalError()
        sampler = None
        if sys.platform.startswith("linux") and affinity_applied:
            from tools import english_asr_benchmark as affinity_io

            sampler = affinity_io._AffinitySamples(
                set(os.sched_getaffinity(0)), affinity_io._sample_thread_affinity
            )

        def update(value):
            nonlocal phase
            phase = value

        result = benchmark_pair(
            assets,
            request["model_id"],
            pair,
            observe=sampler.observe if sampler else lambda: None,
            phase=update,
        )
        phase = "final_bindings"
        if (
            bindings(assets_path) != request["bindings"]
            or pair_digest(
                load_pair(Path(request["manifest"]), request["manifest_sha256"])
            )
            != request["pair_binding"]
        ):
            raise base.EvalError()
        receipt = sampler.receipt() if sampler else None
        if receipt:
            receipt["scope"] = "after_model_load_and_each_original_capture_frame"
        peak = None
        if base.resource is not None:
            peak = int(base.resource.getrusage(base.resource.RUSAGE_SELF).ru_maxrss)
            if sys.platform == "darwin":
                peak //= 1024
        result.update(
            {
                "bindings_sha256": hashlib.sha256(
                    base.io._canonical(request["bindings"])
                ).hexdigest(),
                "peak_rss_kib": peak,
                "cpu_affinity_applied": affinity_applied,
                "posix_resource_limits_applied": base.resource is not None,
                "thread_affinity": receipt,
            }
        )
        base.io._write_new(Path(request["result"]), result)
        return 0
    except BaseException:
        if type(request) is dict and type(request.get("result")) is str:
            try:
                base.io._write_new(
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


def validate_result(value, model_id, pair, expected_binding):
    if type(value) is not dict or value.get("model_id") != model_id:
        raise base.EvalError()
    if value.get("status") == "failed":
        if (
            set(value) != {"model_id", "status", "phase"}
            or value["phase"] not in PHASES
        ):
            raise base.EvalError()
        return
    keys = {
        "model_id",
        "status",
        "input_binding",
        "samples",
        "frame_count",
        "load_seconds",
        "process_wall_seconds",
        "process_cpu_seconds",
        "process_rtf",
        "guard_fallbacks",
        "metrics",
        "bindings_sha256",
        "peak_rss_kib",
        "cpu_affinity_applied",
        "posix_resource_limits_applied",
        "thread_affinity",
    }
    if (
        set(value) != keys
        or value["status"] != "complete"
        or value["input_binding"] != pair_digest(pair)
        or value["bindings_sha256"] != expected_binding
        or type(value["samples"]) is not int
        or value["samples"] != len(pair.microphone)
        or type(value["frame_count"]) is not int
        or value["frame_count"] != len(pair.ranges)
        or type(value["guard_fallbacks"]) is not int
        or not 0 <= value["guard_fallbacks"] <= len(pair.ranges)
        or type(value["cpu_affinity_applied"]) is not bool
        or type(value["posix_resource_limits_applied"]) is not bool
        or (
            value["peak_rss_kib"] is not None
            and (
                type(value["peak_rss_kib"]) is not int
                or not 0 < value["peak_rss_kib"] <= 4 * 1024**2
            )
        )
    ):
        raise base.EvalError()
    for key in (
        "load_seconds",
        "process_wall_seconds",
        "process_cpu_seconds",
        "process_rtf",
    ):
        number = value[key]
        if (
            type(number) not in {int, float}
            or not math.isfinite(number)
            or not 0 <= number <= 180
        ):
            raise base.EvalError()
    if not math.isclose(
        value["process_rtf"],
        value["process_wall_seconds"] / (value["samples"] / 16000),
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise base.EvalError()
    measured = value["metrics"]
    if (
        type(measured) is not dict
        or set(measured)
        != {
            "microphone_rms",
            "reference_rms",
            "output_rms",
            "input_output_power_change_db",
            "output_peak",
            "output_fraction_over_one",
            "reference_active_frame_fraction",
            "output_zero",
        }
        or type(measured["output_zero"]) is not bool
    ):
        raise base.EvalError()
    for key, number in measured.items():
        if key == "output_zero":
            continue
        if key == "input_output_power_change_db" and number is None:
            if measured["microphone_rms"] != 0 or measured["output_rms"] != 0:
                raise base.EvalError()
            continue
        if (
            type(number) not in {int, float}
            or not math.isfinite(number)
            or abs(number) > 1000
            or (key != "input_output_power_change_db" and number < 0)
        ):
            raise base.EvalError()
    if any(
        not 0 <= measured[key] <= 1
        for key in ("output_fraction_over_one", "reference_active_frame_fraction")
    ) or measured["output_zero"] != (measured["output_rms"] == 0):
        raise base.EvalError()
    receipt = value["thread_affinity"]
    if receipt is not None:
        if (
            type(receipt) is not dict
            or not value["cpu_affinity_applied"]
            or receipt.get("scope")
            != "after_model_load_and_each_original_capture_frame"
            or receipt.get("caller_mask_logical_cpus") != 1
        ):
            raise base.EvalError()
        from tools import english_asr_benchmark as affinity_io

        affinity_io._validate_thread_affinity(
            dict(receipt, scope="after_model_load_and_each_decode_samples"),
            len(pair.ranges) + 1,
            complete=True,
        )
    elif sys.platform.startswith("linux"):
        raise base.EvalError()


def run_cell(assets_path, manifest, expected_sha, pair, model_id, scratch, timeout=120):
    if (
        model_id not in base.MODELS
        or type(timeout) is not int
        or not 30 <= timeout <= 180
    ):
        raise base.EvalError()
    bound = bindings(assets_path)
    digest = hashlib.sha256(base.io._canonical(bound)).hexdigest()
    directory = scratch / str(uuid.uuid4())
    directory.mkdir(mode=0o700)
    request, result = directory / "request.json", directory / "aggregate.json"
    base.io._write_new(
        request,
        {
            "assets": str(assets_path),
            "manifest": str(manifest),
            "manifest_sha256": expected_sha,
            "pair_binding": pair_digest(pair),
            "bindings": bound,
            "model_id": model_id,
            "result": str(result),
            "timeout": timeout,
        },
    )
    native = base.load_assets(assets_path)["native_dir"]
    env = {
        "PATH": os.defpath,
        "LANG": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "GGML_NTHREADS": "1",
        "LOCALVQE_ALLOW_UNHASHED": "0",
        "LD_LIBRARY_PATH": native,
        "DYLD_LIBRARY_PATH": native,
        "GGML_BACKEND_PATH": native,
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
    }
    if os.name == "nt" and "SYSTEMROOT" in os.environ:
        env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
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
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=os.name == "posix",
    )
    try:
        child.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        base.io._stop_group(child)
        return {"model_id": model_id, "status": "failed", "phase": "timeout"}
    if not result.exists():
        return {"model_id": model_id, "status": "failed", "phase": "worker_result"}
    value = base.io._json(base.io._read(result, 65536))
    validate_result(value, model_id, pair, digest)
    if (
        bindings(assets_path) != bound
        or pair_digest(load_pair(manifest, expected_sha)) != pair_digest(pair)
        or (child.returncode != 0 and value["status"] == "complete")
    ):
        raise base.EvalError()
    return value


def main(argv=None):
    parser = base._Parser(description=__doc__)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--scratch", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker-request", type=Path, help=argparse.SUPPRESS)
    try:
        args = parser.parse_args(argv)
        if args.worker_request:
            return _worker(args.worker_request)
        if any(
            value is None
            for value in (
                args.assets,
                args.manifest,
                args.expected_manifest_sha256,
                args.scratch,
                args.output,
            )
        ):
            raise base.EvalError()
        scratch = base.io._absolute(args.scratch)
        pair = load_pair(args.manifest, args.expected_manifest_sha256)
        bound = bindings(args.assets)
        digest = hashlib.sha256(base.io._canonical(bound)).hexdigest()
        results = [
            run_cell(
                args.assets,
                args.manifest,
                args.expected_manifest_sha256,
                pair,
                model_id,
                scratch,
            )
            for model_id in base.MODELS
        ]
        if (
            bindings(args.assets) != bound
            or pair_digest(load_pair(args.manifest, args.expected_manifest_sha256))
            != pair_digest(pair)
            or any(
                result["status"] == "complete" and result["bindings_sha256"] != digest
                for result in results
            )
        ):
            raise base.EvalError()
        report = {
            "schema_version": 1,
            "kind": "private-after-host-paired-aec-replay",
            "input": pair.receipt,
            "bindings": bound,
            "native_localvqe_threads_requested": 1,
            "caller_affinity_cpus_requested": 1,
            "results": results,
            "claims": {
                "recorded_audio_used_locally": True,
                "physical_raw_mic_present": False,
                "native_host_pcm_present": False,
                "extra_reference_alignment_applied": False,
                "original_capture_frame_boundaries_preserved": True,
                "terminal_padding_applied": False,
                "audio_device_opened": False,
                "transcripts_or_logs_read": False,
                "waveforms_written": False,
                "near_end_labels_present": False,
                "energy_change_is_erle": False,
                "quality_or_stop_winner": False,
                "default_adoption": False,
                "python_socket_guard": True,
                "os_network_sandbox_attested": False,
                "apm_native_thread_count_controlled": False,
                "apm_native_destruction_attested": False,
            },
        }
        base.io._write_new(args.output, report)
        complete = sum(result["status"] == "complete" for result in results)
        print(
            json.dumps(
                {
                    "ok": complete == len(results),
                    "complete_cells": complete,
                    "total_cells": len(results),
                },
                sort_keys=True,
            )
        )
        return 0 if complete == len(results) else 2
    except Exception:
        print(
            json.dumps(
                {"ok": False, "error": "localvqe_recorded_evaluation_failed"},
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
