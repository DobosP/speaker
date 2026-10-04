"""Offline English TTS development measurements; no capture, playback or adoption.

Private manifest text travels only through a local worker's stdin. Workers keep
PCM in memory, suppress native output, and return aggregate numerical receipts.
Provisioning is explicit and uses the repository's bounded no-follow extractor.
"""

from __future__ import annotations

import argparse
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
import tarfile
import time
from typing import Callable
import urllib.request

CATALOG_PATH = Path(__file__).with_name("english_tts_candidates.json")
MAX_MANIFEST_BYTES = 256 * 1024
MAX_TEXT_BYTES = 2048
MAX_TEXTS = 128
MAX_REPORT_BYTES = 128 * 1024
MAX_ARTIFACT_BYTES = 1024 * 1024 * 1024
MAX_PACKAGE_BYTES = 2 * 1024 * 1024 * 1024
MAX_ARTIFACTS = 4096
MAX_CALLBACKS_PER_GENERATION = 4096
MAX_SAMPLES_PER_GENERATION = 50_000_000
ADDRESS_SPACE_LIMIT = 12 * 1024**3
KINDS = {
    "kokoro_v1_1": "kokoro",
    "vits_libritts_r_medium": "vits",
    "kitten_nano_v0_8_int8": "kitten",
    "supertonic3_int8": "supertonic",
}


class BenchmarkError(Exception):
    """Content-free failure code, safe to print beside aggregate results."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _regular(path: Path) -> None:
    try:
        info = path.lstat()
    except OSError:
        raise BenchmarkError("artifact_unavailable") from None
    if not stat.S_ISREG(info.st_mode):
        raise BenchmarkError("artifact_not_regular")


def file_digest(path: Path, *, ceiling: int = MAX_ARTIFACT_BYTES) -> tuple[str, int]:
    _regular(path)
    digest = hashlib.sha256()
    total = 0
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > ceiling:
                raise BenchmarkError("artifact_size_limit")
            while block := stream.read(1024 * 1024):
                total += len(block)
                if total > ceiling:
                    raise BenchmarkError("artifact_size_limit")
                digest.update(block)
            after = os.fstat(stream.fileno())
            if (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ) or total != before.st_size:
                raise BenchmarkError("artifact_changed")
    except OSError:
        raise BenchmarkError("artifact_read_failed") from None
    return digest.hexdigest(), total


def artifact_inventory(root: Path) -> list[dict]:
    if root.is_symlink() or not root.is_dir():
        raise BenchmarkError("package_unavailable")
    records = []
    total = 0
    for parent, directories, files in os.walk(root, followlinks=False):
        for name in directories:
            if (Path(parent) / name).is_symlink():
                raise BenchmarkError("package_symlink")
        for name in sorted(files):
            path = Path(parent) / name
            if name == "provision.json":
                continue
            digest, size = file_digest(path)
            total += size
            if len(records) >= MAX_ARTIFACTS or total > MAX_PACKAGE_BYTES:
                raise BenchmarkError("package_size_limit")
            records.append(
                {
                    "file": path.relative_to(root).as_posix(),
                    "bytes": size,
                    "sha256": digest,
                }
            )
    return sorted(records, key=lambda row: row["file"])


def load_owner_manifest(path: Path) -> tuple[list[str], dict]:
    _regular(path)
    try:
        with path.open("rb") as stream:
            raw = stream.read(MAX_MANIFEST_BYTES + 1)
        if len(raw) > MAX_MANIFEST_BYTES:
            raise BenchmarkError("manifest_size_limit")
        manifest = json.loads(raw)
        if type(manifest) is not dict or type(manifest.get("clips")) is not list:
            raise BenchmarkError("manifest_invalid")
        clips = manifest["clips"]
        if not 1 <= len(clips) <= MAX_TEXTS:
            raise BenchmarkError("manifest_count_limit")
        texts = []
        for clip in clips:
            if type(clip) is not dict or type(clip.get("text")) is not str:
                raise BenchmarkError("manifest_invalid")
            text = clip["text"]
            if not text.strip() or len(text.encode("utf-8")) > MAX_TEXT_BYTES:
                raise BenchmarkError("manifest_text_limit")
            texts.append(text)
        binding = hashlib.sha256(canonical_bytes(manifest)).hexdigest()
    except BenchmarkError:
        raise
    except Exception:
        raise BenchmarkError("manifest_invalid") from None
    return texts, {
        "manifest_sha256": binding,
        "manifest_binding": "sha256_canonical_sorted_compact_json_ascii",
        "reference_count": len(texts),
        "provenance": "owner_recording_unattested_development",
        "disjoint_holdout": False,
        "source_audio_consumed": False,
    }


def load_catalog() -> dict:
    return json.loads(CATALOG_PATH.read_text())


def _private_write(path: Path, value: object) -> None:
    payload = canonical_bytes(value) + b"\n"
    if len(payload) > MAX_REPORT_BYTES:
        raise BenchmarkError("report_size_limit")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError:
        raise BenchmarkError("report_write_failed") from None


def _safe_directory(path: Path) -> None:
    # Reuse the same descriptor-based no-follow path checks as model setup.
    from tools.setup_models import _ensure_extract_destination

    try:
        _ensure_extract_destination(str(path))
    except Exception:
        raise BenchmarkError("destination_unsafe") from None


def _download(candidate: dict, target: Path, *, deadline_seconds: float = 180) -> None:
    if target.exists():
        digest, _ = file_digest(target, ceiling=candidate["max_archive_bytes"])
        if digest != candidate["archive_sha256"]:
            raise BenchmarkError("archive_checksum_mismatch")
        return
    part = target.with_name(target.name + ".part")
    deadline = time.monotonic() + deadline_seconds
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(part, flags, 0o600)
        with os.fdopen(fd, "wb") as stream:
            with urllib.request.urlopen(candidate["url"], timeout=15) as response:
                total = 0
                digest = hashlib.sha256()
                while block := response.read(1024 * 1024):
                    total += len(block)
                    if (
                        total > candidate["max_archive_bytes"]
                        or time.monotonic() > deadline
                    ):
                        raise BenchmarkError("download_bound_exceeded")
                    stream.write(block)
                    digest.update(block)
                stream.flush()
                os.fsync(stream.fileno())
        if digest.hexdigest() != candidate["archive_sha256"]:
            raise BenchmarkError("archive_checksum_mismatch")
        # No-clobber publication. Failed partials stay available for diagnosis.
        os.link(part, target)
        part.unlink()
    except BenchmarkError:
        raise
    except Exception:
        raise BenchmarkError("download_failed") from None


def provision_candidate(model_id: str, root: Path) -> dict:
    candidate = load_catalog()["candidates"].get(model_id)
    if candidate is None:
        raise BenchmarkError("candidate_unknown")
    _safe_directory(root)
    archives = root / "archives"
    _safe_directory(archives)
    archive = archives / candidate["archive"]
    _download(candidate, archive)
    digest, archive_size = file_digest(archive, ceiling=candidate["max_archive_bytes"])
    if digest != candidate["archive_sha256"]:
        raise BenchmarkError("archive_checksum_mismatch")
    if "weights_license_sha256" in candidate:
        licenses = root / "licenses"
        _safe_directory(licenses)
        license_target = licenses / (model_id + "-weights-license.txt")
        _download(
            {
                "url": candidate["license_source"].replace("/blob/", "/resolve/"),
                "archive_sha256": candidate["weights_license_sha256"],
                "max_archive_bytes": 65536,
            },
            license_target,
        )
    package = root / model_id
    receipt_path = package / "provision.json"
    if receipt_path.exists():
        try:
            receipt = json.loads(receipt_path.read_bytes())
            if receipt["archive_sha256"] != digest or receipt[
                "artifacts"
            ] != artifact_inventory(package):
                raise BenchmarkError("provision_changed")
            return receipt
        except BenchmarkError:
            raise
        except Exception:
            raise BenchmarkError("provision_changed") from None
    if package.exists():
        raise BenchmarkError("provision_incomplete")
    # The shared extractor selects only regular files. Reject links and all
    # unusual types in the entire archive before it gets a chance to select.
    try:
        with tarfile.open(archive, "r:*") as tar:
            count = 0
            declared_bytes = 0
            destinations = set()
            for member in tar:
                count += 1
                declared_bytes += member.size if member.isfile() else 0
                parts = member.name.replace("\\", "/").split("/")
                if (
                    count > MAX_ARTIFACTS
                    or member.name.startswith(("/", "\\"))
                    or ".." in parts
                    or not (member.isfile() or member.isdir())
                    or member.size < 0
                    or member.size > MAX_ARTIFACT_BYTES
                    or declared_bytes > MAX_PACKAGE_BYTES
                ):
                    raise BenchmarkError("archive_unsafe")
                if member.isfile():
                    relative = "/".join(
                        part for part in parts[1:] if part and part != "."
                    )
                    if not relative or relative in destinations:
                        raise BenchmarkError("archive_unsafe")
                    destinations.add(relative)
        from tools.setup_models import _extract_tar_members

        _extract_tar_members(
            str(archive),
            str(package),
            select=lambda members: members,
            path_mode="strip_top_level",
            reject_parent_parts=True,
            unreadable_is_error=True,
        )
    except BenchmarkError:
        raise
    except Exception:
        raise BenchmarkError("archive_extract_failed") from None
    receipt = {
        "schema": 1,
        "model_id": model_id,
        "version": candidate["version"],
        "license": candidate["weights_license"],
        "engine_license": "Apache-2.0",
        "license_source": candidate["license_source"],
        "weights_license_sha256": candidate.get("weights_license_sha256"),
        "archive_sha256": digest,
        "archive_bytes": archive_size,
        "artifacts": artifact_inventory(package),
    }
    _private_write(receipt_path, receipt)
    return receipt


def _summary(values: list[float]) -> dict | None:
    if not values:
        return None
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": ordered[-1],
    }


def benchmark_model(
    generate: Callable,
    texts: list[str],
    *,
    repeats: int,
    clock: Callable[[], float] = time.perf_counter,
) -> dict:
    """Pure injectable reducer; retains no reference text, PCM or per-case rows."""
    import numpy as np

    if type(repeats) is not int or not 1 <= repeats <= 10:
        raise BenchmarkError("repeats_invalid")
    if not texts or len(texts) > MAX_TEXTS:
        raise BenchmarkError("text_count_invalid")
    first_callbacks = []
    first_pcm = []
    full_times = []
    durations = []
    rtf = []
    callback_counts = []
    early_callbacks = 0
    invalid_waves = 0
    zero_waves = 0
    full_scale_samples = 0
    samples_total = 0
    callback_missing = 0
    invalid_callback_chunks = 0
    # Repetition zero is a separately reported cold-first call, not discarded.
    cold_first = None
    for repeat in range(repeats):
        for text in texts:
            if (
                type(text) is not str
                or not text.strip()
                or len(text.encode("utf-8")) > MAX_TEXT_BYTES
            ):
                raise BenchmarkError("text_invalid")
            started = clock()
            first = None
            nonzero = None
            callback_count = 0
            callback_events = 0
            callback_invalid = False

            def callback(samples, _progress):
                nonlocal first, nonzero, callback_count, callback_events
                nonlocal callback_invalid, invalid_callback_chunks
                entered = clock()
                callback_events += 1
                if callback_events > MAX_CALLBACKS_PER_GENERATION:
                    callback_invalid = True
                    invalid_callback_chunks += 1
                    return 0
                values = np.asarray(samples)
                if (
                    values.ndim != 1
                    or values.dtype.kind != "f"
                    or values.size > MAX_SAMPLES_PER_GENERATION
                    or not np.all(np.isfinite(values))
                ):
                    callback_invalid = True
                    invalid_callback_chunks += 1
                    return 1
                if values.size:
                    callback_count += 1
                    if first is None:
                        first = entered - started
                    if nonzero is None and np.any(values != 0):
                        nonzero = entered - started
                return 1  # Native Sherpa callback: 1 continues, 0 cancels.

            try:
                audio = generate(text, callback)
                elapsed = clock() - started
                sample_rate = int(audio.sample_rate)
                values = np.asarray(audio.samples)
            except Exception:
                raise BenchmarkError("synthesis_failed") from None
            if (
                not math.isfinite(elapsed)
                or elapsed < 0
                or any(
                    value is not None
                    and (not math.isfinite(value) or value < 0 or value > elapsed)
                    for value in (first, nonzero)
                )
            ):
                raise BenchmarkError("clock_invalid")
            full_times.append(elapsed)
            callback_counts.append(callback_count)
            if first is None:
                callback_missing += 1
            else:
                first_callbacks.append(first)
            if nonzero is not None:
                first_pcm.append(nonzero)
            # A chunk available at least 1 ms before return is observable
            # inside-call delivery; this is not proof of audible playback.
            if first is not None and elapsed - first >= 0.001:
                early_callbacks += 1
            if (
                callback_invalid
                or values.ndim != 1
                or values.dtype.kind != "f"
                or values.size > MAX_SAMPLES_PER_GENERATION
                or sample_rate <= 0
                or values.size == 0
                or not np.all(np.isfinite(values))
            ):
                invalid_waves += 1
            else:
                duration = values.size / sample_rate
                durations.append(duration)
                rtf.append(elapsed / duration)
                zero_waves += int(not np.any(values != 0))
                full_scale_samples += int(np.count_nonzero(np.abs(values) >= 1.0))
                samples_total += int(values.size)
            if cold_first is None:
                cold_first = {
                    "full_synthesis_seconds": elapsed,
                    "first_callback_seconds": first,
                    "first_nonzero_pcm_seconds": nonzero,
                }
            # Drop the native result and every view before the next generation.
            del audio, values
    return {
        "generation_count": len(texts) * repeats,
        "repeats": repeats,
        "cold_first_generation": cold_first,
        "first_callback_seconds": _summary(first_callbacks),
        "first_nonzero_pcm_callback_seconds": _summary(first_pcm),
        "full_synthesis_seconds": _summary(full_times),
        "audio_seconds": _summary(durations),
        "rtf": _summary(rtf),
        "callback_chunks": _summary(callback_counts),
        "generations_without_callback": callback_missing,
        "generations_with_callback_before_return": early_callbacks,
        "invalid_waveforms": invalid_waves,
        "invalid_callback_chunks": invalid_callback_chunks,
        "all_zero_waveforms": zero_waves,
        "samples_at_or_above_full_scale": full_scale_samples,
        "samples_total": samples_total,
        "quality_authority": False,
        "audibility_authority": False,
        "live_authority": False,
        "phone_authority": False,
        "control_authority": False,
    }


def _load_native(model_id: str, root: Path, threads: int, sid: int = 0, steps: int = 8):
    import sherpa_onnx as s

    if s.__version__ != "1.13.3":
        raise BenchmarkError("runtime_version_mismatch")
    kind = KINDS[model_id]
    config = s.OfflineTtsConfig()
    config.model.num_threads = threads
    config.model.provider = "cpu"
    config.model.debug = False
    config.max_num_sentences = 1
    if kind in {"kokoro", "kitten", "vits"}:
        model = getattr(config.model, kind)
        model.model = str(
            root
            / ("en_US-libritts_r-medium.onnx" if kind == "vits" else "model.int8.onnx")
        )
        model.tokens = str(root / "tokens.txt")
        model.data_dir = str(root / "espeak-ng-data")
        if kind != "vits":
            model.voices = str(root / "voices.bin")
        if kind == "kokoro":
            model.lexicon = str(root / "lexicon-us-en.txt")
    else:
        model = config.model.supertonic
        for key, filename in {
            "duration_predictor": "duration_predictor.int8.onnx",
            "text_encoder": "text_encoder.int8.onnx",
            "vector_estimator": "vector_estimator.int8.onnx",
            "vocoder": "vocoder.int8.onnx",
            "tts_json": "tts.json",
            "unicode_indexer": "unicode_indexer.bin",
            "voice_style": "voice.bin",
        }.items():
            setattr(model, key, str(root / filename))
    if not config.validate():
        raise BenchmarkError("model_config_invalid")
    tts = s.OfflineTts(config)
    generation = s.GenerationConfig()
    generation.sid = sid
    generation.speed = 1.0
    generation.num_steps = steps
    generation.extra = {"lang": "en"}
    return (
        tts,
        lambda text, callback: tts.generate(text, config=generation, callback=callback),
        s.__version__,
    )


def _validate_payload(payload: dict) -> None:
    if type(payload) is not dict or set(payload) != {
        "model_id",
        "model_root",
        "texts",
        "threads",
        "repeats",
        "sid",
        "steps",
    }:
        raise BenchmarkError("worker_input_invalid")
    if (
        payload["model_id"] not in KINDS
        or type(payload["model_root"]) is not str
        or not 1 <= len(payload["model_root"]) <= 4096
    ):
        raise BenchmarkError("worker_input_invalid")
    for name, low, high in (
        ("threads", 1, 4),
        ("repeats", 1, 10),
        ("sid", 0, 1024),
        ("steps", 1, 16),
    ):
        if type(payload[name]) is not int or not low <= payload[name] <= high:
            raise BenchmarkError("worker_input_invalid")
    texts = payload["texts"]
    if type(texts) is not list or not 1 <= len(texts) <= MAX_TEXTS:
        raise BenchmarkError("worker_input_invalid")
    for text in texts:
        if (
            type(text) is not str
            or not text.strip()
            or len(text.encode("utf-8")) > MAX_TEXT_BYTES
        ):
            raise BenchmarkError("worker_input_invalid")


def _validate_worker_report(
    result: object, payload: dict, *, cpu_limit_seconds: int = 600
) -> None:
    if type(result) is not dict or set(result) != {
        "status",
        "model_id",
        "model_kind",
        "runtime",
        "execution",
        "model_artifacts_sha256",
        "model_artifact_bytes",
        "model_load_seconds",
        "peak_process_rss_kib",
        "metrics",
    }:
        raise BenchmarkError("tts_worker_protocol_failed")
    if (
        result["status"] != "ok"
        or result["model_id"] != payload["model_id"]
        or result["model_kind"] != KINDS[payload["model_id"]]
    ):
        raise BenchmarkError("tts_worker_protocol_failed")
    expected_runtime = {
        "sherpa_onnx": "1.13.3",
        "provider": "cpu",
        "configured_inference_threads": payload["threads"],
        "speaker_id": payload["sid"],
        "generation_steps": payload["steps"],
        "speed": 1.0,
        "postprocessing": "none",
    }
    if result["runtime"] != expected_runtime or any(
        type(result["runtime"].get(key)) is not type(value)
        for key, value in expected_runtime.items()
    ):
        raise BenchmarkError("tts_worker_protocol_failed")
    expected_execution = _execution_receipt(payload["threads"], cpu_limit_seconds)
    if result["execution"] != expected_execution or any(
        type(result["execution"].get(key)) is not type(value)
        for key, value in expected_execution.items()
    ):
        raise BenchmarkError("tts_worker_protocol_failed")
    digest = result["model_artifacts_sha256"]
    if (
        type(digest) is not str
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise BenchmarkError("tts_worker_protocol_failed")
    for key in ("model_artifact_bytes", "model_load_seconds", "peak_process_rss_kib"):
        value = result[key]
        if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
            raise BenchmarkError("tts_worker_protocol_failed")
    metrics = result["metrics"]
    expected_metric_keys = {
        "generation_count",
        "repeats",
        "cold_first_generation",
        "first_callback_seconds",
        "first_nonzero_pcm_callback_seconds",
        "full_synthesis_seconds",
        "audio_seconds",
        "rtf",
        "callback_chunks",
        "generations_without_callback",
        "generations_with_callback_before_return",
        "invalid_waveforms",
        "invalid_callback_chunks",
        "all_zero_waveforms",
        "samples_at_or_above_full_scale",
        "samples_total",
        "quality_authority",
        "audibility_authority",
        "live_authority",
        "phone_authority",
        "control_authority",
    }
    if type(metrics) is not dict or set(metrics) != expected_metric_keys:
        raise BenchmarkError("tts_worker_protocol_failed")

    expected_count = len(payload["texts"]) * payload["repeats"]
    case_counters = (
        "generation_count",
        "repeats",
        "generations_without_callback",
        "generations_with_callback_before_return",
        "invalid_waveforms",
        "all_zero_waveforms",
    )
    for key in case_counters:
        if type(metrics[key]) is not int or not 0 <= metrics[key] <= expected_count:
            raise BenchmarkError("tts_worker_protocol_failed")
    for key in ("samples_total", "samples_at_or_above_full_scale"):
        if (
            type(metrics[key]) is not int
            or not 0 <= metrics[key] <= expected_count * MAX_SAMPLES_PER_GENERATION
        ):
            raise BenchmarkError("tts_worker_protocol_failed")
    if metrics["samples_at_or_above_full_scale"] > metrics["samples_total"]:
        raise BenchmarkError("tts_worker_protocol_failed")
    if type(metrics["invalid_callback_chunks"]) is not int or not 0 <= metrics[
        "invalid_callback_chunks"
    ] <= expected_count * (MAX_CALLBACKS_PER_GENERATION + 1):
        raise BenchmarkError("tts_worker_protocol_failed")

    def finite_number(value):
        return type(value) in {int, float} and math.isfinite(value) and value >= 0

    for key in (
        "first_callback_seconds",
        "first_nonzero_pcm_callback_seconds",
        "full_synthesis_seconds",
        "audio_seconds",
        "rtf",
        "callback_chunks",
    ):
        stats = metrics[key]
        if stats is None:
            continue
        if type(stats) is not dict or set(stats) != {
            "count",
            "min",
            "p50",
            "p95",
            "max",
        }:
            raise BenchmarkError("tts_worker_protocol_failed")
        if (
            type(stats["count"]) is not int
            or not 0 < stats["count"] <= metrics["generation_count"]
        ):
            raise BenchmarkError("tts_worker_protocol_failed")
        if any(not finite_number(stats[key]) for key in ("min", "p50", "p95", "max")):
            raise BenchmarkError("tts_worker_protocol_failed")
        if not stats["min"] <= stats["p50"] <= stats["p95"] <= stats["max"]:
            raise BenchmarkError("tts_worker_protocol_failed")
    cold = metrics["cold_first_generation"]
    if type(cold) is not dict or set(cold) != {
        "full_synthesis_seconds",
        "first_callback_seconds",
        "first_nonzero_pcm_seconds",
    }:
        raise BenchmarkError("tts_worker_protocol_failed")
    if any(value is not None and not finite_number(value) for value in cold.values()):
        raise BenchmarkError("tts_worker_protocol_failed")
    if (
        metrics["generation_count"] != len(payload["texts"]) * payload["repeats"]
        or metrics["repeats"] != payload["repeats"]
    ):
        raise BenchmarkError("tts_worker_protocol_failed")
    for key in (
        "quality_authority",
        "audibility_authority",
        "live_authority",
        "phone_authority",
        "control_authority",
    ):
        if metrics[key] is not False:
            raise BenchmarkError("tts_worker_protocol_failed")


def _verify_model_identity(model_id: str, inventory: list[dict]) -> None:
    catalog = load_catalog()
    candidate = catalog["candidates"].get(model_id)
    if candidate is not None:
        actual = hashlib.sha256(canonical_bytes(inventory)).hexdigest()
        if actual != candidate.get("artifact_inventory_sha256"):
            raise BenchmarkError("model_identity_mismatch")
        return
    key = (
        "baseline_kokoro_identity"
        if model_id == "kokoro_v1_1"
        else "baseline_vits_identity"
    )
    expected = catalog.get(key)
    if type(expected) is not dict or not expected:
        raise BenchmarkError("model_identity_unproven")
    actual = {
        row["file"]: {"bytes": row["bytes"], "sha256": row["sha256"]}
        for row in inventory
    }
    if any(actual.get(name) != identity for name, identity in expected.items()):
        raise BenchmarkError("model_identity_mismatch")


def _worker_environment() -> dict[str, str]:
    # No inherited credentials, proxy settings, Python injection or SDK keys.
    # The executable and model paths are explicit; model inference needs no HOME.
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }
    env.update(
        {
            name: "1"
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "PYTHONDONTWRITEBYTECODE",
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
                "HF_HUB_DISABLE_TELEMETRY",
            )
        }
    )
    env["SPEAKER_TEST_LOG"] = "0"
    env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def _deny_worker_network(*_args, **_kwargs):
    raise BenchmarkError("worker_python_network_denied")


def _execution_receipt(threads: int, cpu_limit_seconds: int) -> dict:
    return {
        "cpu_affinity_applied": True,
        "logical_cpus": threads,
        "cpu_limit_seconds": cpu_limit_seconds,
        "cpu_hard_limit_seconds": cpu_limit_seconds + 1,
        "address_space_limit_bytes": ADDRESS_SPACE_LIMIT,
        "core_limit_bytes": 0,
        "network_python_hooks": True,
        "os_network_isolated": False,
        "sanitized_environment": True,
    }


def _apply_worker_guards(threads: int, cpu_limit_seconds: int) -> dict:
    if type(cpu_limit_seconds) is not int or not 1 <= cpu_limit_seconds <= 3600:
        raise BenchmarkError("worker_cpu_limit_invalid")
    try:
        allowed = sorted(os.sched_getaffinity(0))
        if len(allowed) < threads:
            raise BenchmarkError("worker_cpu_affinity_unavailable")
        selected = set(allowed[:threads])
        os.sched_setaffinity(0, selected)
        if set(os.sched_getaffinity(0)) != selected:
            raise BenchmarkError("worker_cpu_affinity_unavailable")
        resource.setrlimit(
            resource.RLIMIT_CPU, (cpu_limit_seconds, cpu_limit_seconds + 1)
        )
        resource.setrlimit(
            resource.RLIMIT_AS, (ADDRESS_SPACE_LIMIT, ADDRESS_SPACE_LIMIT)
        )
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        sanitized = _worker_environment()
        os.environ.clear()
        os.environ.update(sanitized)
        socket.socket.connect = _deny_worker_network
        socket.socket.connect_ex = _deny_worker_network
        socket.socket.sendto = _deny_worker_network
        socket.create_connection = _deny_worker_network
    except BenchmarkError:
        raise
    except Exception:
        raise BenchmarkError("worker_resource_guard_failed") from None
    return _execution_receipt(threads, cpu_limit_seconds)


def worker(payload: dict, *, execution: dict | None = None) -> dict:
    _validate_payload(payload)
    model_id = payload["model_id"]
    root = Path(payload["model_root"])
    threads = payload["threads"]
    if model_id not in KINDS or type(threads) is not int or not 1 <= threads <= 4:
        raise BenchmarkError("worker_input_invalid")
    before = artifact_inventory(root)
    _verify_model_identity(model_id, before)
    if execution is None:
        raise BenchmarkError("worker_guards_unproven")
    started = time.perf_counter()
    native, generate, version = _load_native(
        model_id, root, threads, payload["sid"], payload["steps"]
    )
    loaded = time.perf_counter() - started
    metrics = benchmark_model(generate, payload["texts"], repeats=payload["repeats"])
    del generate, native
    after = artifact_inventory(root)
    if after != before:
        raise BenchmarkError("model_artifacts_changed")
    return {
        "status": "ok",
        "model_id": model_id,
        "model_kind": KINDS[model_id],
        "runtime": {
            "sherpa_onnx": version,
            "provider": "cpu",
            "configured_inference_threads": threads,
            "speaker_id": payload["sid"],
            "generation_steps": payload["steps"],
            "speed": 1.0,
            "postprocessing": "none",
        },
        "execution": execution,
        "model_artifacts_sha256": hashlib.sha256(canonical_bytes(before)).hexdigest(),
        "model_artifact_bytes": sum(row["bytes"] for row in before),
        "model_load_seconds": loaded,
        "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "metrics": metrics,
    }


def _worker_entry() -> int:
    # Retain a dedicated report descriptor, then permanently silence Python and
    # native stdout/stderr before inspecting private text or importing Sherpa.
    report_fd = os.dup(1)
    null_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null_fd, 1)
    os.dup2(null_fd, 2)
    os.close(null_fd)
    try:
        raw = sys.stdin.buffer.read(MAX_MANIFEST_BYTES + 1)
        if len(raw) > MAX_MANIFEST_BYTES:
            raise BenchmarkError("worker_input_limit")
        job = json.loads(raw)
        if type(job) is not dict or set(job) != {"request", "cpu_limit_seconds"}:
            raise BenchmarkError("worker_input_invalid")
        request = job["request"]
        _validate_payload(request)
        execution = _apply_worker_guards(request["threads"], job["cpu_limit_seconds"])
        result = worker(request, execution=execution)
        exit_code = 0
    except BenchmarkError as error:
        result = {"status": "failed", "code": error.code}
        exit_code = 2
    except BaseException:
        result = {"status": "failed", "code": "tts_worker_failed"}
        exit_code = 2
    os.write(report_fd, canonical_bytes(result) + b"\n")
    os.close(report_fd)
    return exit_code


def _stop_and_reap_worker(process) -> None:
    # Every abnormal parent exit must retire its exact new-session worker.
    if process.poll() is not None:
        process.wait(timeout=5)
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.communicate(timeout=1)
        return
    except BaseException:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.communicate(timeout=5)
    except BaseException:
        try:
            process.kill()
            process.wait(timeout=5)
        except BaseException:
            raise BenchmarkError("tts_worker_cleanup_unproven") from None


def run_isolated(payload: dict, *, timeout: float = 600) -> dict:
    _validate_payload(payload)
    if not math.isfinite(timeout) or not 0 < timeout <= 3600:
        raise BenchmarkError("timeout_invalid")
    env = _worker_environment()
    cpu_limit_seconds = math.ceil(timeout)
    job = {"request": payload, "cpu_limit_seconds": cpu_limit_seconds}
    process = subprocess.Popen(
        [sys.executable, "-B", "-m", "tools.english_tts_benchmark", "--worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        env=env,
    )
    try:
        stdout, _ = process.communicate(canonical_bytes(job), timeout=timeout)
    except subprocess.TimeoutExpired:
        _stop_and_reap_worker(process)
        raise BenchmarkError("tts_worker_timeout") from None
    except BaseException:
        _stop_and_reap_worker(process)
        raise
    if len(stdout) > MAX_REPORT_BYTES:
        raise BenchmarkError("tts_worker_report_limit")
    try:
        result = json.loads(stdout)
    except Exception:
        raise BenchmarkError("tts_worker_protocol_failed") from None
    if (
        process.returncode != 0
        or type(result) is not dict
        or result.get("status") != "ok"
    ):
        # Worker messages, provider errors, and stderr never become observer data.
        raise BenchmarkError("tts_worker_failed")
    _validate_worker_report(result, payload, cpu_limit_seconds=cpu_limit_seconds)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model", choices=sorted(KINDS))
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--speaker-id", type=int, default=0)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--provision", type=Path, metavar="ROOT")
    args = parser.parse_args(argv)
    if args.worker:
        return _worker_entry()
    try:
        if args.model is None:
            raise BenchmarkError("model_required")
        if args.provision is not None:
            receipt = provision_candidate(args.model, args.provision)
            print(
                canonical_bytes(
                    {
                        "status": "provisioned",
                        "model_id": args.model,
                        "archive_sha256": receipt["archive_sha256"],
                        "artifact_count": len(receipt["artifacts"]),
                    }
                ).decode()
            )
            return 0
        if args.manifest is None or args.model_root is None:
            raise BenchmarkError("manifest_and_model_root_required")
        texts, source = load_owner_manifest(args.manifest)
        result = run_isolated(
            {
                "model_id": args.model,
                "model_root": str(args.model_root),
                "texts": texts,
                "threads": args.threads,
                "repeats": args.repeats,
                "sid": args.speaker_id,
                "steps": args.steps,
            },
            timeout=args.timeout,
        )
        report = {
            "schema": 1,
            "purpose": "offline_english_tts_development_only",
            "source": source,
            "result": result,
        }
        if args.output is not None:
            _private_write(args.output, report)
        print(canonical_bytes(report).decode())
        return 0
    except BenchmarkError as error:
        print(canonical_bytes({"status": "failed", "code": error.code}).decode())
        return 2
    except Exception:
        print('{"status":"failed","code":"tts_benchmark_failed"}')
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
