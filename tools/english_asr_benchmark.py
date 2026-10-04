"""CPU-only, aggregate-only English after-PCM model comparison.

The caller supplies scripted references; this development harness does not
create labels, attest human recording provenance, or promote a default. One
isolated child processes one model at a time. Existing live, verifier and
streaming-worker contracts are deliberately independent of this harness.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import hashlib
import io
import json
import math
from importlib import metadata as distribution_metadata
import platform
import socket
import os
from pathlib import Path
import re
import resource
import signal
import stat
import subprocess
import sys
import tempfile
import time
import wave

MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_WAV_BYTES = 30 * 16000 * 2 + 65536
MAX_CLIPS = 256
MAX_AUDIO_SECONDS = 600
MAX_REFERENCE_CHARS = 1024
MAX_HYPOTHESIS_CHARS = 4096
MAX_MODEL_FILES = 64
MAX_MODEL_BYTES = 8 * 1024**3
MAX_REPORT_BYTES = 512 * 1024
ROLES = frozenset({"round_trip", "speak", "barge", "command"})
MODEL_IDS = frozenset(
    {
        "zipformer-en",
        "zipformer-en-int8",
        "sensevoice-en",
        "parakeet-unified-en",
        "parakeet-tdt-v3",
        "faster-whisper-small",
        "moonshine-tiny-streaming-v015",
        "moonshine-small-streaming-v015",
    }
)
ZIPFORMER_IDS = frozenset({"zipformer-en", "zipformer-en-int8"})
ZIPFORMER_TAIL_PADDING_SAMPLES = 10560
# Official streaming-file example pads 0.66 s before input_finished():
# https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/online-decode-files.py
MOONSHINE_IDS = frozenset(
    {"moonshine-tiny-streaming-v015", "moonshine-small-streaming-v015"}
)
MOONSHINE_FILES = frozenset(
    {
        "adapter.ort",
        "cross_kv.ort",
        "decoder_kv.ort",
        "encoder.ort",
        "frontend.model.ort",
        "frontend.weights.ort",
        "streaming_config.json",
        "tokenizer.bin",
    }
)
_WORD = re.compile(r"[a-z0-9']+")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class BenchmarkError(RuntimeError):
    """Only fixed error codes may cross the worker boundary."""


@dataclass(frozen=True)
class Clip:
    path: Path = field(repr=False)
    reference: str = field(repr=False)
    role: str
    digest: str
    frames: int

    @property
    def duration(self) -> float:
        return self.frames / 16000


@dataclass(frozen=True)
class Corpus:
    manifest: Path = field(repr=False)
    clips: tuple[Clip, ...] = field(repr=False)
    digest: str
    manifest_digest: str
    source_root: Path = field(repr=False)
    kind: str

    @property
    def seconds(self) -> float:
        return sum(clip.duration for clip in self.clips)


@dataclass(frozen=True)
class Model:
    id: str
    artifacts: dict[str, Path] = field(repr=False)
    python: Path | None = field(default=None, repr=False)
    arch: str | None = None
    runtime_wheel: Path | None = field(default=None, repr=False)
    runtime_wheel_sha256: str | None = None
    ort_single_thread: bool = False


def _canonical(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _json(raw: bytes):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise BenchmarkError()
            result[key] = value
        return result

    def constant(_value):
        raise BenchmarkError()

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def _absolute(path: Path | str) -> Path:
    candidate = Path(os.path.abspath(path))
    # Reject any symlink in the supplied path, including parent components.
    if candidate.resolve(strict=True) != candidate:
        raise BenchmarkError()
    return candidate


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


def _read(path: Path, maximum: int) -> bytes:
    path = _absolute(path)
    before = path.lstat()
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or not 0 < before.st_size <= maximum
    ):
        raise BenchmarkError()
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as handle:
        opened = os.fstat(handle.fileno())
        if _identity(opened) != _identity(before):
            raise BenchmarkError()
        raw = handle.read(maximum + 1)
        after = os.fstat(handle.fileno())
    if (
        len(raw) != before.st_size
        or _identity(after) != _identity(before)
        or _identity(path.lstat()) != _identity(before)
    ):
        raise BenchmarkError()
    return raw


def _file_digest(path: Path, maximum: int) -> tuple[str, int]:
    path = _absolute(path)
    before = path.lstat()
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or not 0 < before.st_size <= maximum
    ):
        raise BenchmarkError()
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    digest = hashlib.sha256()
    size = 0
    with os.fdopen(descriptor, "rb") as handle:
        if _identity(os.fstat(handle.fileno())) != _identity(before):
            raise BenchmarkError()
        while block := handle.read(1024 * 1024):
            size += len(block)
            if size > maximum:
                raise BenchmarkError()
            digest.update(block)
        after = os.fstat(handle.fileno())
    if (
        size != before.st_size
        or _identity(after) != _identity(before)
        or _identity(path.lstat()) != _identity(before)
    ):
        raise BenchmarkError()
    return digest.hexdigest(), size


def _pcm(raw: bytes) -> tuple[bytes, int]:
    with wave.open(io.BytesIO(raw), "rb") as audio:
        frames = audio.getnframes()
        if (
            audio.getnchannels(),
            audio.getsampwidth(),
            audio.getframerate(),
            audio.getcomptype(),
        ) != (1, 2, 16000, "NONE"):
            raise BenchmarkError()
        if not 1 <= frames <= 30 * 16000:
            raise BenchmarkError()
        pcm = audio.readframes(frames + 1)
    if len(pcm) != frames * 2:
        raise BenchmarkError()
    return pcm, frames


def load_corpus(path: Path | str, source_root: Path | str | None = None) -> Corpus:
    manifest = _absolute(path)
    root = _absolute(source_root) if source_root is not None else manifest.parent
    if not root.is_dir():
        raise BenchmarkError()
    raw = _read(manifest, MAX_MANIFEST_BYTES)
    value = _json(raw)
    if not isinstance(value, dict) or "clips" not in value:
        raise BenchmarkError()
    legacy = "clip_dir" in value
    if not legacy and set(value) != {"clips"}:
        raise BenchmarkError()
    directory = manifest.parent
    if legacy:
        if not isinstance(value["clip_dir"], str) or not value["clip_dir"]:
            raise BenchmarkError()
        declared = Path(value["clip_dir"])
        directory = _absolute(declared if declared.is_absolute() else root / declared)
        # Legacy clip_dir is an explicit grant, independently of task cwd.
        if not directory.is_dir():
            raise BenchmarkError()
    rows = value["clips"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_CLIPS:
        raise BenchmarkError()
    clips, seen = [], set()
    digest = hashlib.sha256(raw)
    manifest_digest = digest.hexdigest()
    for row in rows:
        if not isinstance(row, dict):
            raise BenchmarkError()
        if legacy:
            if not {"id", "expected_text", "sha256"} <= set(row):
                raise BenchmarkError()
            identifier = row["id"]
            if not isinstance(identifier, str) or not identifier:
                raise BenchmarkError()
            name, reference, role = (
                identifier + ".wav",
                row["expected_text"],
                "round_trip",
            )
            expected_hash = row["sha256"]
            if not isinstance(expected_hash, str) or not _DIGEST.fullmatch(
                expected_hash
            ):
                raise BenchmarkError()
        else:
            if not {"file", "text", "role"} <= set(row) or set(row) - {
                "file",
                "text",
                "role",
                "group",
                "tag",
                "intent",
            }:
                raise BenchmarkError()
            name, reference, role = row["file"], row["text"], row["role"]
            expected_hash = None
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or name in {".", ".."}
            or "\x00" in name
            or name in seen
            or not isinstance(reference, str)
            or not 1 <= len(reference) <= MAX_REFERENCE_CHARS
            or not _WORD.findall(reference.lower())
            or len(_WORD.findall(reference.lower())) > 256
            or not isinstance(role, str)
            or role not in ROLES
        ):
            raise BenchmarkError()
        seen.add(name)
        clip_path = _absolute(directory / name)
        if clip_path.parent != directory:
            raise BenchmarkError()
        waveform = _read(clip_path, MAX_WAV_BYTES)
        _data, frames = _pcm(waveform)
        clip_digest = hashlib.sha256(waveform).hexdigest()
        if expected_hash is not None and expected_hash != clip_digest:
            raise BenchmarkError()
        digest.update(bytes.fromhex(clip_digest))
        clips.append(Clip(clip_path, reference, role, clip_digest, frames))
    corpus = Corpus(
        manifest,
        tuple(clips),
        digest.hexdigest(),
        manifest_digest,
        root,
        "recorded-legacy" if legacy else "owner-scripted",
    )
    if corpus.seconds > MAX_AUDIO_SECONDS:
        raise BenchmarkError()
    return corpus


def verify_corpus(corpus: Corpus) -> None:
    if load_corpus(corpus.manifest, corpus.source_root).digest != corpus.digest:
        raise BenchmarkError()


def _python_path(path: Path | str) -> Path:
    candidate = Path(os.path.abspath(path))
    if _absolute(candidate.parent) != candidate.parent:
        raise BenchmarkError()
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file() or not os.access(candidate, os.X_OK):
        raise BenchmarkError()
    return candidate


def _runtime_binding(model: Model) -> dict:
    python = model.python or Path(sys.executable)
    executable_digest, _size = _file_digest(
        python.resolve(strict=True), 64 * 1024 * 1024
    )
    result = {"executable_sha256": executable_digest}
    marker = python.parent.parent / "pyvenv.cfg"
    if marker.exists():
        result["venv_marker_sha256"] = hashlib.sha256(_read(marker, 65536)).hexdigest()
    if model.runtime_wheel is not None:
        digest, size = _file_digest(model.runtime_wheel, 512 * 1024 * 1024)
        if digest != model.runtime_wheel_sha256:
            raise BenchmarkError()
        result.update({"wheel_sha256": digest, "wheel_bytes": size})
    return result


def load_models(path: Path | str) -> tuple[tuple[Model, ...], str]:
    config = _absolute(path)
    raw = _read(config, MAX_MANIFEST_BYTES)
    value = _json(raw)
    if (
        not isinstance(value, dict)
        or set(value) != {"schema_version", "models"}
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
    ):
        raise BenchmarkError()
    entries = value["models"]
    if not isinstance(entries, list) or not 1 <= len(entries) <= len(MODEL_IDS):
        raise BenchmarkError()
    models = []
    seen = set()
    for entry in entries:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("id"), str)
            or entry["id"] not in MODEL_IDS
            or entry["id"] in seen
        ):
            raise BenchmarkError()
        identifier = entry["id"]
        if identifier in MOONSHINE_IDS:
            required = {"id", "kind", "model_dir", "arch", "python"}
            optional = {"runtime_wheel", "runtime_wheel_sha256", "ort_single_thread"}
            if (
                not required <= set(entry)
                or set(entry) - required - optional
                or entry["kind"] != "moonshine-native"
            ):
                raise BenchmarkError()
            expected_arch = (
                "tiny" if identifier == "moonshine-tiny-streaming-v015" else "small"
            )
            if entry["arch"] != expected_arch or any(
                not isinstance(entry[key], str) or not entry[key]
                for key in ("model_dir", "python")
            ):
                raise BenchmarkError()
            supplied = Path(entry["model_dir"])
            directory = _absolute(
                supplied if supplied.is_absolute() else config.parent / supplied
            )
            executable = Path(entry["python"])
            python = _python_path(
                executable if executable.is_absolute() else config.parent / executable
            )
            wheel = None
            wheel_digest = entry.get("runtime_wheel_sha256")
            if bool(entry.get("runtime_wheel")) != bool(wheel_digest):
                raise BenchmarkError()
            if wheel_digest is not None:
                if (
                    not isinstance(wheel_digest, str)
                    or not _DIGEST.fullmatch(wheel_digest)
                    or not isinstance(entry["runtime_wheel"], str)
                ):
                    raise BenchmarkError()
                supplied = Path(entry["runtime_wheel"])
                wheel = _absolute(
                    supplied if supplied.is_absolute() else config.parent / supplied
                )
            single_thread = entry.get("ort_single_thread", False)
            if type(single_thread) is not bool:
                raise BenchmarkError()
            models.append(
                Model(
                    identifier,
                    {"directory": directory},
                    python,
                    expected_arch,
                    wheel,
                    wheel_digest,
                    single_thread,
                )
            )
        else:
            if set(entry) != {"id", "artifacts"}:
                raise BenchmarkError()
            required = (
                {"directory"}
                if identifier == "faster-whisper-small"
                else {"model", "tokens"}
                if identifier == "sensevoice-en"
                else {"encoder", "decoder", "joiner", "tokens"}
            )
            artifacts = entry["artifacts"]
            if (
                not isinstance(artifacts, dict)
                or set(artifacts) != required
                or any(
                    not isinstance(item, str) or not item for item in artifacts.values()
                )
            ):
                raise BenchmarkError()
            resolved = {}
            for key, item in artifacts.items():
                supplied = Path(item)
                resolved[key] = _absolute(
                    supplied if supplied.is_absolute() else config.parent / supplied
                )
            models.append(Model(identifier, resolved))
        seen.add(identifier)
    return tuple(models), hashlib.sha256(raw).hexdigest()


def model_binding(model: Model) -> dict:
    leaves = []
    if model.id == "faster-whisper-small" or model.id in MOONSHINE_IDS:
        directory = model.artifacts["directory"]
        if not directory.is_dir():
            raise BenchmarkError()
        files, pending, entries_seen = [], [(directory, 0)], 0
        while pending:
            parent, depth = pending.pop()
            with os.scandir(parent) as entries:
                for entry in entries:
                    entries_seen += 1
                    if entries_seen > MAX_MODEL_FILES * 2 or entry.is_symlink():
                        raise BenchmarkError()
                    if entry.is_dir(follow_symlinks=False):
                        if depth >= 3:
                            raise BenchmarkError()
                        pending.append((Path(entry.path), depth + 1))
                    elif entry.is_file(follow_symlinks=False):
                        files.append(Path(entry.path))
                    else:
                        raise BenchmarkError()
        files.sort()
        names = {path.relative_to(directory).as_posix() for path in files}
        if model.id == "faster-whisper-small" and (
            not {"model.bin", "config.json", "tokenizer.json"} <= names
            or not names & {"vocabulary.txt", "vocabulary.json"}
        ):
            raise BenchmarkError()
        if model.id in MOONSHINE_IDS and names != MOONSHINE_FILES:
            raise BenchmarkError()
        leaves = [(path.relative_to(directory).as_posix(), path) for path in files]
    else:
        leaves = sorted(model.artifacts.items())
    if not 1 <= len(leaves) <= MAX_MODEL_FILES:
        raise BenchmarkError()
    binding = []
    size = 0
    for role, path in leaves:
        digest, length = _file_digest(path, MAX_MODEL_BYTES)
        size += length
        if size > MAX_MODEL_BYTES:
            raise BenchmarkError()
        binding.append({"role": role, "sha256": digest, "bytes": length})
    # File roles / names affect the digest but never leave the private worker.
    return {
        "sha256": hashlib.sha256(_canonical(binding)).hexdigest(),
        "files": len(binding),
        "bytes": size,
    }


def _distance(reference, hypothesis) -> int:
    previous = list(range(len(hypothesis) + 1))
    for index, ref in enumerate(reference, 1):
        current = [index]
        for other, hyp in enumerate(hypothesis, 1):
            current.append(
                min(
                    previous[other] + 1,
                    current[-1] + 1,
                    previous[other - 1] + (ref != hyp),
                )
            )
        previous = current
    return previous[-1]


def _accuracy_pairs(pairs) -> dict:
    totals = Counter(
        clips=0,
        exact=0,
        empty=0,
        word_errors=0,
        reference_words=0,
        character_errors=0,
        reference_characters=0,
    )
    for reference, hypothesis in pairs:
        ref, hyp = _WORD.findall(reference.lower()), _WORD.findall(hypothesis.lower())
        ref_chars, hyp_chars = "".join(ref), "".join(hyp)
        totals.update(
            {
                "clips": 1,
                "exact": int(ref == hyp),
                "empty": int(not hyp),
                "word_errors": _distance(ref, hyp),
                "reference_words": len(ref),
                "character_errors": _distance(ref_chars, hyp_chars),
                "reference_characters": len(ref_chars),
            }
        )
    return {
        **dict(totals),
        "wer": totals["word_errors"] / max(1, totals["reference_words"]),
        "cer": totals["character_errors"] / max(1, totals["reference_characters"]),
    }


def _percentile(values, percentile) -> float | None:
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(len(values) * percentile) - 1)]


def _decoder(model: Model, threads: int):
    paths = {key: str(path) for key, path in model.artifacts.items()}
    if model.id in MOONSHINE_IDS:
        import importlib.util

        adapter_path = Path(__file__).with_name("english_moonshine_adapter.py")
        module_spec = importlib.util.spec_from_file_location(
            "english_moonshine_adapter", adapter_path
        )
        if module_spec is None or module_spec.loader is None:
            raise BenchmarkError()
        adapter = importlib.util.module_from_spec(module_spec)
        sys.modules[module_spec.name] = adapter
        module_spec.loader.exec_module(adapter)
        return adapter.build_decoder(
            model.artifacts["directory"], model.arch + "-streaming", threads
        )
    if model.id == "faster-whisper-small":
        from faster_whisper import WhisperModel

        recognizer = WhisperModel(
            paths["directory"],
            device="cpu",
            compute_type="int8",
            cpu_threads=threads,
            num_workers=1,
            local_files_only=True,
        )

        def decode(samples):
            segments, _info = recognizer.transcribe(
                samples,
                language="en",
                task="transcribe",
                beam_size=5,
                temperature=0.0,
                vad_filter=False,
                condition_on_previous_text=False,
                without_timestamps=True,
                word_timestamps=False,
            )
            text = []
            for index, segment in enumerate(segments):
                if index >= 256 or not isinstance(segment.text, str):
                    raise BenchmarkError()
                text.append(segment.text)
                if sum(map(len, text)) > MAX_HYPOTHESIS_CHARS:
                    raise BenchmarkError()
            return " ".join(text).strip()

        return decode
    import sherpa_onnx

    if model.id in ZIPFORMER_IDS:
        recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            **paths,
            provider="cpu",
            num_threads=threads,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="modified_beam_search",
            max_active_paths=4,
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=2.4,
            rule2_min_trailing_silence=0.8,
            rule3_min_utterance_length=20.0,
        )

        def decode(samples):
            stream = recognizer.create_stream()
            for start in range(0, len(samples), 1600):
                stream.accept_waveform(16000, samples[start : start + 1600])
                while recognizer.is_ready(stream):
                    recognizer.decode_stream(stream)
            import numpy as np

            stream.accept_waveform(
                16000, np.zeros(ZIPFORMER_TAIL_PADDING_SAMPLES, dtype="float32")
            )
            stream.input_finished()
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)
            return recognizer.get_result(stream)

        return decode
    if model.id == "sensevoice-en":
        recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            **paths, provider="cpu", num_threads=threads, language="en", use_itn=True
        )
    else:
        recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            **paths,
            provider="cpu",
            num_threads=threads,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
            max_active_paths=4,
            model_type="nemo_transducer",
        )

    def decode(samples):
        stream = recognizer.create_stream()
        stream.accept_waveform(16000, samples)
        recognizer.decode_stream(stream)
        return stream.result.text

    return decode


def _versions(model: Model) -> dict:
    names = (
        ("moonshine-voice",)
        if model.id in MOONSHINE_IDS
        else (
            ("faster-whisper", "ctranslate2", "numpy")
            if model.id == "faster-whisper-small"
            else ("sherpa-onnx", "numpy")
        )
    )
    result = {"python": platform.python_version()}
    for name in names:
        try:
            version = distribution_metadata.version(name)
        except distribution_metadata.PackageNotFoundError:
            version = "unavailable"
        if not re.fullmatch(r"[A-Za-z0-9.+_-]{1,64}", version):
            raise BenchmarkError()
        result[name] = version
    return result


def _code_binding(model: Model) -> dict:
    paths = {"benchmark_sha256": Path(__file__).resolve()}
    if model.id in MOONSHINE_IDS:
        paths["moonshine_adapter_sha256"] = Path(__file__).with_name(
            "english_moonshine_adapter.py"
        )
    return {
        role: hashlib.sha256(_read(path, MAX_MANIFEST_BYTES)).hexdigest()
        for role, path in paths.items()
    }


def _cpu_affinity(threads: int) -> dict:
    try:
        available = sorted(os.sched_getaffinity(0))
        selected = set(available[:threads])
        if not selected:
            raise BenchmarkError()
        os.sched_setaffinity(0, selected)
        return {"applied": True, "logical_cpus": len(os.sched_getaffinity(0))}
    except (AttributeError, OSError):
        return {"applied": False, "logical_cpus": None}


def _deny_network() -> None:
    def deny(*_args, **_kwargs):
        raise BenchmarkError()

    socket.socket.connect = deny
    socket.socket.connect_ex = deny
    socket.socket.sendto = deny
    socket.create_connection = deny


def _process_threads() -> int | None:
    try:
        with open("/proc/self/status", "r", encoding="ascii") as handle:
            status = handle.read(65536)
        for line in status.splitlines():
            if line.startswith("Threads:"):
                count = int(line.split(":", 1)[1].strip())
                return count if 1 <= count <= 65536 else None
    except (OSError, ValueError):
        pass
    return None


def _thread_policy(model: Model, threads: int) -> dict:
    native_supported = model.id not in MOONSHINE_IDS
    value = os.environ.get("MOONSHINE_ORT_SINGLE_THREAD", "unset")
    if value not in {"unset", "0", "1"}:
        raise BenchmarkError()
    if model.id in MOONSHINE_IDS and value != ("1" if model.ort_single_thread else "0"):
        raise BenchmarkError()
    return {
        "requested_affinity_threads": threads,
        "native_thread_budget_supported": native_supported,
        "configured_native_threads": threads
        if native_supported
        else (1 if model.ort_single_thread else None),
        "ort_single_thread": model.ort_single_thread,
        "moonshine_single_thread_env": value,
    }


def _input_receipt(
    model_id: str, source_seconds: float, clips: int, repeats: int
) -> dict:
    padding = ZIPFORMER_TAIL_PADDING_SAMPLES if model_id in ZIPFORMER_IDS else 0
    return {
        "sample_rate": 16000,
        "tail_padding_samples_per_clip": padding,
        "tail_padding_policy": "sherpa_streaming_file_flush" if padding else "none",
        "input_finished": model_id in ZIPFORMER_IDS,
        "original_audio_seconds": source_seconds,
        "source_audio_seconds_across_calls": source_seconds * repeats,
        "total_model_input_seconds": (source_seconds + padding / 16000 * clips)
        * repeats,
    }


MAX_AFFINITY_THREADS = 4096
MAX_CPU_INDEX = 65535


class NativeAffinityViolation(BenchmarkError):
    def __init__(self, receipt: dict):
        super().__init__()
        self.receipt = receipt


def _parse_cpu_list(value: str) -> set[int]:
    if not value or not re.fullmatch(
        r"[0-9]+(?:-[0-9]+)?(?:,[0-9]+(?:-[0-9]+)?)*", value
    ):
        raise BenchmarkError()
    result = set()
    for item in value.split(","):
        bounds = item.split("-")
        low, high = int(bounds[0]), int(bounds[-1])
        if low > high or high > MAX_CPU_INDEX or high - low + 1 > MAX_AFFINITY_THREADS:
            raise BenchmarkError()
        result.update(range(low, high + 1))
        if len(result) > MAX_AFFINITY_THREADS:
            raise BenchmarkError()
    return result


def _sample_thread_affinity(caller_mask: set[int] | None) -> dict:
    sample = {
        "successful": False,
        "threads_observed": 0,
        "unknown_threads": 0,
        "read_failures": 0,
        "outside_caller": 0,
        "union_logical_cpus": 0,
    }
    if caller_mask is None:
        return sample
    union = set()
    try:
        with os.scandir("/proc/self/task") as entries:
            for entry in entries:
                if not entry.name.isdecimal():
                    continue
                sample["threads_observed"] += 1
                if sample["threads_observed"] > MAX_AFFINITY_THREADS:
                    raise BenchmarkError()
                try:
                    with open(
                        Path(entry.path) / "status", "r", encoding="ascii"
                    ) as handle:
                        raw = handle.read(65536)
                    values = [
                        line.split(":", 1)[1].strip()
                        for line in raw.splitlines()
                        if line.startswith("Cpus_allowed_list:")
                    ]
                    if len(values) != 1:
                        raise BenchmarkError()
                    mask = _parse_cpu_list(values[0])
                    union.update(mask)
                    sample["outside_caller"] += int(not mask <= caller_mask)
                except (OSError, UnicodeError):
                    sample["unknown_threads"] += 1
                    sample["read_failures"] += 1
                except BenchmarkError:
                    sample["unknown_threads"] += 1
        sample["successful"] = sample["threads_observed"] > 0
    except (OSError, BenchmarkError):
        sample["read_failures"] += 1
    sample["union_logical_cpus"] = len(union)
    return sample


class _AffinitySamples:
    def __init__(self, caller_mask: set[int] | None, sampler):
        self.caller_mask = caller_mask
        self.sampler = sampler
        self.samples = []

    def observe(self) -> None:
        self.samples.append(self.sampler(self.caller_mask))
        if self.samples[-1]["outside_caller"]:
            raise NativeAffinityViolation(self.receipt())

    def receipt(self) -> dict:
        samples = self.samples
        total = lambda key: sum(sample[key] for sample in samples)
        maximum = lambda key: max((sample[key] for sample in samples), default=0)
        successful = sum(sample["successful"] for sample in samples)
        compliant = bool(
            self.caller_mask is not None
            and samples
            and successful == len(samples)
            and total("unknown_threads")
            == total("read_failures")
            == total("outside_caller")
            == 0
        )
        return {
            "scope": "after_model_load_and_each_decode_samples",
            "continuous_observation": False,
            "cgroup_isolation": False,
            "caller_mask_logical_cpus": len(self.caller_mask)
            if self.caller_mask is not None
            else None,
            "sample_attempts": len(samples),
            "successful_samples": successful,
            "sampling_failures": len(samples) - successful,
            "threads_observed": total("threads_observed"),
            "unknown_thread_observations": total("unknown_threads"),
            "read_failures": total("read_failures"),
            "outside_caller_mask_observations": total("outside_caller"),
            "maximum_threads_observed": maximum("threads_observed"),
            "maximum_outside_caller_mask_observations": maximum("outside_caller"),
            "maximum_observed_logical_cpus": maximum("union_logical_cpus"),
            "violating_samples": sum(
                sample["outside_caller"] > 0 for sample in samples
            ),
            "sampled_compliant": compliant,
        }


def benchmark_cell(
    model: Model,
    corpus: Corpus,
    repeats: int,
    threads: int,
    *,
    decoder_factory=_decoder,
    caller_mask: set[int] | None = None,
    affinity_sampler=_sample_thread_affinity,
    affinity_failure_sink=None,
) -> dict:
    import numpy as np

    binding = model_binding(model)
    runtime_binding = _runtime_binding(model)
    code_binding = _code_binding(model)
    verify_corpus(corpus)
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    threads_before_load = _process_threads()
    load_start = time.perf_counter()
    decode = decoder_factory(model, threads)
    load_ms = (time.perf_counter() - load_start) * 1000
    threads_after_load = _process_threads()
    sampled_threads = [
        count
        for count in (threads_before_load, threads_after_load)
        if count is not None
    ]
    coverage = _AffinitySamples(caller_mask, affinity_sampler)
    times, first_texts = [], []
    inconsistent = 0
    try:
        coverage.observe()
        for repeat in range(repeats):
            for ordinal, clip in enumerate(corpus.clips):
                raw = _read(clip.path, MAX_WAV_BYTES)
                if hashlib.sha256(raw).hexdigest() != clip.digest:
                    raise BenchmarkError()
                pcm, _frames = _pcm(raw)
                samples = np.frombuffer(pcm, dtype="<i2").astype("float32") / 32768.0
                call_start = time.perf_counter()
                hypothesis = decode(samples)
                elapsed = (time.perf_counter() - call_start) * 1000
                coverage.observe()
                if (
                    not isinstance(hypothesis, str)
                    or len(hypothesis) > MAX_HYPOTHESIS_CHARS
                ):
                    raise BenchmarkError()
                times.append(elapsed)
                if repeat == 0:
                    first_texts.append(hypothesis)
                elif hypothesis != first_texts[ordinal]:
                    inconsistent += 1
    except NativeAffinityViolation as failure:
        if affinity_failure_sink is not None:
            affinity_failure_sink(failure.receipt)
        raise
    finally:
        close = getattr(decode, "close", None)
        if callable(close):
            close()
    pairs = [
        (clip.reference, hypothesis)
        for clip, hypothesis in zip(corpus.clips, first_texts, strict=True)
    ]
    accuracy = _accuracy_pairs(pairs)
    by_role = {
        role: _accuracy_pairs(
            [
                pair
                for clip, pair in zip(corpus.clips, pairs, strict=True)
                if clip.role == role
            ]
        )
        for role in sorted(ROLES)
        if any(clip.role == role for clip in corpus.clips)
    }
    verify_corpus(corpus)
    if (
        model_binding(model) != binding
        or _runtime_binding(model) != runtime_binding
        or _code_binding(model) != code_binding
    ):
        raise BenchmarkError()
    cpu_seconds = time.process_time() - start_cpu
    wall_seconds = time.perf_counter() - start_wall
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "model_id": model.id,
        "status": "complete",
        "error_count": 0,
        "model_tuple": binding,
        "model_input": _input_receipt(
            model.id, corpus.seconds, len(corpus.clips), repeats
        ),
        "runtime_binding": runtime_binding,
        "code_binding": code_binding,
        "versions": _versions(model),
        "clips": len(corpus.clips),
        "calls": len(times),
        "complete_corpus_coverage": len(first_texts) == len(corpus.clips),
        "accuracy": accuracy,
        "by_role": by_role,
        "model_load_ms": load_ms,
        "first_call_ms": times[0],
        "warm_p50_ms": _percentile(times[1:], 0.5),
        "warm_p95_ms": _percentile(times[1:], 0.95),
        "decode_seconds": sum(times) / 1000,
        "worker_wall_seconds": wall_seconds,
        "worker_cpu_seconds": cpu_seconds,
        "decode_rtf": sum(times) / 1000 / (corpus.seconds * repeats),
        "worker_wall_rtf": wall_seconds / (corpus.seconds * repeats),
        "peak_rss_bytes": int(rss * (1 if sys.platform == "darwin" else 1024)),
        "repeat_disagreements": inconsistent,
        "thread_policy": _thread_policy(model, threads),
        "thread_affinity": coverage.receipt(),
        "process_threads": {
            "before_model_load": threads_before_load,
            "after_model_load": threads_after_load,
            "maximum_sampled": max(sampled_threads) if sampled_threads else None,
        },
    }


def _write_new(path: Path, value) -> None:
    raw = _canonical(value) + b"\n"
    if len(raw) > MAX_REPORT_BYTES:
        raise BenchmarkError()
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _worker(request_path: Path) -> int:
    # Redirect OS descriptors before importing any native library. No native
    # exception, warning or decoded text can reach an inherited output stream.
    descriptor = os.open(os.devnull, os.O_WRONLY)
    os.dup2(descriptor, 1)
    os.dup2(descriptor, 2)
    os.close(descriptor)
    request = None
    try:
        request = _json(_read(request_path, MAX_REPORT_BYTES))
        if not isinstance(request, dict) or set(request) != {
            "manifest",
            "source_root",
            "model",
            "corpus_digest",
            "model_tuple",
            "runtime_binding",
            "code_binding",
            "repeats",
            "threads",
            "result",
            "timeout_sec",
        }:
            raise BenchmarkError()
        if not _DIGEST.fullmatch(request["corpus_digest"]):
            raise BenchmarkError()
        _validate_limits(request["repeats"], request["threads"], request["timeout_sec"])
        resource.setrlimit(
            resource.RLIMIT_CPU,
            (math.ceil(request["timeout_sec"]), math.ceil(request["timeout_sec"]) + 1),
        )
        resource.setrlimit(resource.RLIMIT_AS, (12 * 1024**3, 12 * 1024**3))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        affinity = _cpu_affinity(request["threads"])
        _deny_network()
        corpus = load_corpus(request["manifest"], request["source_root"])
        if corpus.digest != request["corpus_digest"]:
            raise BenchmarkError()
        model_raw = request["model"]
        if (
            not isinstance(model_raw, dict)
            or set(model_raw)
            != {
                "id",
                "artifacts",
                "python",
                "arch",
                "runtime_wheel",
                "runtime_wheel_sha256",
                "ort_single_thread",
            }
            or model_raw["id"] not in MODEL_IDS
        ):
            raise BenchmarkError()
        if type(model_raw["ort_single_thread"]) is not bool:
            raise BenchmarkError()
        model = Model(
            model_raw["id"],
            {key: _absolute(value) for key, value in model_raw["artifacts"].items()},
            _python_path(model_raw["python"]) if model_raw["python"] else None,
            model_raw["arch"],
            _absolute(model_raw["runtime_wheel"])
            if model_raw["runtime_wheel"]
            else None,
            model_raw["runtime_wheel_sha256"],
            model_raw["ort_single_thread"],
        )
        if (
            model_binding(model) != request["model_tuple"]
            or _runtime_binding(model) != request["runtime_binding"]
            or _code_binding(model) != request["code_binding"]
        ):
            raise BenchmarkError()
        caller_mask = set(os.sched_getaffinity(0)) if affinity["applied"] else None

        def affinity_failure(receipt):
            _write_new(
                Path(request["result"]),
                {
                    "model_id": model.id,
                    "status": "worker_failed",
                    "error_count": 1,
                    "complete_corpus_coverage": False,
                    "failure_code": "native_thread_affinity_violation",
                    "thread_affinity": receipt,
                },
            )

        result = benchmark_cell(
            model,
            corpus,
            request["repeats"],
            request["threads"],
            caller_mask=caller_mask,
            affinity_failure_sink=affinity_failure,
        )
        result["cpu_affinity"] = affinity
        _write_new(Path(request["result"]), result)
        return 0
    except NativeAffinityViolation:
        # The fixed aggregate violation marker is published before native close,
        # since a badly oversubscribed native destructor may itself stall.
        return 2
    except BaseException:
        if isinstance(request, dict) and isinstance(request.get("result"), str):
            try:
                _write_new(
                    Path(request["result"]),
                    {
                        "model_id": request["model"]["id"],
                        "status": "worker_failed",
                        "error_count": 1,
                        "complete_corpus_coverage": False,
                    },
                )
            except BaseException:
                pass
        return 2


def _validate_limits(repeats: int, threads: int, timeout_sec: float) -> None:
    if (
        type(repeats) is not int
        or not 1 <= repeats <= 8
        or type(threads) is not int
        or not 1 <= threads <= 8
        or isinstance(timeout_sec, bool)
        or not isinstance(timeout_sec, (int, float))
        or not math.isfinite(timeout_sec)
        or not 1 <= timeout_sec <= 3600
    ):
        raise BenchmarkError()


def _stop_group(process) -> None:
    if process.poll() is not None:
        return
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
        process.wait(timeout=2)


def run_cell(
    model: Model,
    corpus: Corpus,
    scratch: Path,
    repeats: int,
    threads: int,
    timeout_sec: float,
) -> dict:
    _validate_limits(repeats, threads, timeout_sec)
    binding = model_binding(model)
    runtime_binding = _runtime_binding(model)
    code_binding = _code_binding(model)
    with tempfile.TemporaryDirectory(prefix="cell-", dir=scratch) as raw_directory:
        directory = Path(raw_directory)
        result_path = directory / "aggregate.json"
        request_path = directory / "request.json"
        _write_new(
            request_path,
            {
                "manifest": str(corpus.manifest),
                "source_root": str(corpus.source_root),
                "model": {
                    "id": model.id,
                    "artifacts": {
                        key: str(path) for key, path in model.artifacts.items()
                    },
                    "python": str(model.python) if model.python else None,
                    "arch": model.arch,
                    "runtime_wheel": str(model.runtime_wheel)
                    if model.runtime_wheel
                    else None,
                    "runtime_wheel_sha256": model.runtime_wheel_sha256,
                    "ort_single_thread": model.ort_single_thread,
                },
                "corpus_digest": corpus.digest,
                "model_tuple": binding,
                "runtime_binding": runtime_binding,
                "code_binding": code_binding,
                "repeats": repeats,
                "threads": threads,
                "timeout_sec": timeout_sec,
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
            "NUMEXPR_NUM_THREADS": str(threads),
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "MOONSHINE_ORT_SINGLE_THREAD": "1" if model.ort_single_thread else "0",
        }
        started = time.perf_counter()
        process = subprocess.Popen(
            [
                str(model.python) if model.python else sys.executable,
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
            process.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            _stop_group(process)
            return {
                "model_id": model.id,
                "status": "worker_timeout",
                "error_count": 1,
                "complete_corpus_coverage": False,
            }
        except BaseException:
            _stop_group(process)
            raise
        elapsed = time.perf_counter() - started
        try:
            result = _json(_read(result_path, MAX_REPORT_BYTES))
            _validate_cell_result(
                result,
                model.id,
                len(corpus.clips),
                repeats,
                source_seconds=corpus.seconds,
                role_counts=Counter(clip.role for clip in corpus.clips),
                ort_single_thread=model.ort_single_thread,
            )
            if result["status"] == "complete" and process.returncode != 0:
                raise BenchmarkError()
            verify_corpus(corpus)
            if (
                model_binding(model) != binding
                or _runtime_binding(model) != runtime_binding
                or _code_binding(model) != code_binding
            ):
                raise BenchmarkError()
            if result["status"] == "complete" and (
                result["model_tuple"] != binding
                or result["runtime_binding"] != runtime_binding
                or result["code_binding"] != code_binding
            ):
                raise BenchmarkError()
        except BaseException:
            return {
                "model_id": model.id,
                "status": "worker_failed",
                "error_count": 1,
                "complete_corpus_coverage": False,
            }
        result["process_wall_seconds"] = elapsed
        result["process_wall_rtf"] = elapsed / (corpus.seconds * repeats)
        return result


def _validate_thread_affinity(receipt, max_samples: int, *, complete: bool) -> None:
    counts = {
        "sample_attempts",
        "successful_samples",
        "sampling_failures",
        "threads_observed",
        "unknown_thread_observations",
        "read_failures",
        "outside_caller_mask_observations",
        "maximum_threads_observed",
        "maximum_outside_caller_mask_observations",
        "maximum_observed_logical_cpus",
        "violating_samples",
    }
    keys = counts | {
        "scope",
        "continuous_observation",
        "cgroup_isolation",
        "caller_mask_logical_cpus",
        "sampled_compliant",
    }
    if type(receipt) is not dict or set(receipt) != keys:
        raise BenchmarkError()
    if (
        receipt["scope"] != "after_model_load_and_each_decode_samples"
        or receipt["continuous_observation"] is not False
        or receipt["cgroup_isolation"] is not False
        or type(receipt["sampled_compliant"]) is not bool
    ):
        raise BenchmarkError()
    for key in counts:
        value = receipt[key]
        if (
            type(value) is not int
            or not 0 <= value <= (MAX_AFFINITY_THREADS + 1) * max_samples
        ):
            raise BenchmarkError()
    attempts = receipt["sample_attempts"]
    if not 1 <= attempts <= max_samples or (complete and attempts != max_samples):
        raise BenchmarkError()
    if (
        receipt["successful_samples"] + receipt["sampling_failures"] != attempts
        or receipt["violating_samples"] > attempts
        or receipt["maximum_threads_observed"] > MAX_AFFINITY_THREADS + 1
        or receipt["maximum_outside_caller_mask_observations"]
        > receipt["maximum_threads_observed"]
        or receipt["outside_caller_mask_observations"] > receipt["threads_observed"]
        or receipt["unknown_thread_observations"] > receipt["threads_observed"]
        or receipt["maximum_observed_logical_cpus"] > MAX_CPU_INDEX + 1
    ):
        raise BenchmarkError()
    caller = receipt["caller_mask_logical_cpus"]
    if caller is not None and (type(caller) is not int or not 1 <= caller <= 8):
        raise BenchmarkError()
    compliant = bool(
        caller is not None
        and receipt["successful_samples"] == attempts
        and receipt["unknown_thread_observations"]
        == receipt["read_failures"]
        == receipt["outside_caller_mask_observations"]
        == 0
    )
    if receipt["sampled_compliant"] is not compliant:
        raise BenchmarkError()
    if complete and (
        receipt["violating_samples"] or receipt["outside_caller_mask_observations"]
    ):
        raise BenchmarkError()


def _validate_cell_result(
    result,
    model_id: str,
    clips: int,
    repeats: int,
    *,
    source_seconds: float | None = None,
    role_counts: dict | None = None,
    ort_single_thread: bool = False,
) -> None:
    def integer(value, low=0, high=None):
        if type(value) is not int or value < low or (high is not None and value > high):
            raise BenchmarkError()

    def number(value, *, nullable=False):
        if nullable and value is None:
            return
        if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
            raise BenchmarkError()

    def digest(value):
        if type(value) is not str or not _DIGEST.fullmatch(value):
            raise BenchmarkError()

    minimal = {"model_id", "status", "error_count", "complete_corpus_coverage"}
    timing_keys = {
        "model_load_ms",
        "first_call_ms",
        "warm_p50_ms",
        "warm_p95_ms",
        "decode_seconds",
        "worker_wall_seconds",
        "worker_cpu_seconds",
        "decode_rtf",
        "worker_wall_rtf",
    }
    complete = (
        minimal
        | timing_keys
        | {
            "model_tuple",
            "model_input",
            "clips",
            "calls",
            "accuracy",
            "by_role",
            "peak_rss_bytes",
            "repeat_disagreements",
            "runtime_binding",
            "code_binding",
            "versions",
            "cpu_affinity",
            "thread_policy",
            "thread_affinity",
            "process_threads",
        }
    )
    if (
        type(result) is not dict
        or result.get("model_id") != model_id
        or result.get("status") not in {"complete", "worker_failed"}
    ):
        raise BenchmarkError()
    integer(result.get("error_count"), 0, 1)
    if result["status"] != "complete":
        allowed = minimal | {"failure_code", "thread_affinity"}
        if (
            set(result) not in (minimal, allowed)
            or result["error_count"] != 1
            or result["complete_corpus_coverage"] is not False
        ):
            raise BenchmarkError()
        if set(result) == allowed:
            if result["failure_code"] != "native_thread_affinity_violation":
                raise BenchmarkError()
            _validate_thread_affinity(
                result["thread_affinity"], 1 + clips * repeats, complete=False
            )
            if not result["thread_affinity"]["outside_caller_mask_observations"]:
                raise BenchmarkError()
        return
    if (
        set(result) != complete
        or result["complete_corpus_coverage"] is not True
        or result["error_count"] != 0
    ):
        raise BenchmarkError()
    integer(result["clips"], clips, clips)
    integer(result["calls"], clips * repeats, clips * repeats)
    integer(result["peak_rss_bytes"], 1)
    integer(result["repeat_disagreements"], 0, clips * (repeats - 1))
    for key in timing_keys:
        number(
            result[key],
            nullable=key in {"warm_p50_ms", "warm_p95_ms"} and clips * repeats == 1,
        )
    if (
        result["warm_p50_ms"] is not None
        and result["warm_p95_ms"] < result["warm_p50_ms"]
    ):
        raise BenchmarkError()

    accuracy_counts = {
        "clips",
        "exact",
        "empty",
        "word_errors",
        "reference_words",
        "character_errors",
        "reference_characters",
    }

    def accuracy(value, expected_clips=None):
        if type(value) is not dict or set(value) != accuracy_counts | {"wer", "cer"}:
            raise BenchmarkError()
        for key in accuracy_counts:
            integer(value[key])
        if expected_clips is not None and value["clips"] != expected_clips:
            raise BenchmarkError()
        count = value["clips"]
        if (
            value["exact"] + value["empty"] > count
            or not count <= value["reference_words"] <= 256 * count
            or not value["reference_words"]
            <= value["reference_characters"]
            <= MAX_REFERENCE_CHARS * count
            or value["word_errors"]
            > value["reference_words"] + MAX_HYPOTHESIS_CHARS * count
            or value["character_errors"]
            > value["reference_characters"] + MAX_HYPOTHESIS_CHARS * count
        ):
            raise BenchmarkError()
        for metric, numerator, denominator in (
            ("wer", "word_errors", "reference_words"),
            ("cer", "character_errors", "reference_characters"),
        ):
            number(value[metric])
            if not math.isclose(
                value[metric],
                value[numerator] / max(1, value[denominator]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise BenchmarkError()

    accuracy(result["accuracy"], clips)
    roles = result["by_role"]
    if type(roles) is not dict or not roles or set(roles) - ROLES:
        raise BenchmarkError()
    if role_counts is not None and set(roles) != set(role_counts):
        raise BenchmarkError()
    for role, value in roles.items():
        accuracy(value, role_counts[role] if role_counts is not None else None)
    if any(
        sum(value[key] for value in roles.values()) != result["accuracy"][key]
        for key in accuracy_counts
    ):
        raise BenchmarkError()

    inputs = result["model_input"]
    input_keys = {
        "sample_rate",
        "tail_padding_samples_per_clip",
        "tail_padding_policy",
        "input_finished",
        "original_audio_seconds",
        "source_audio_seconds_across_calls",
        "total_model_input_seconds",
    }
    if type(inputs) is not dict or set(inputs) != input_keys:
        raise BenchmarkError()
    for key in (
        "original_audio_seconds",
        "source_audio_seconds_across_calls",
        "total_model_input_seconds",
    ):
        number(inputs[key])
    if not 0 < inputs["original_audio_seconds"] <= MAX_AUDIO_SECONDS:
        raise BenchmarkError()
    original = (
        inputs["original_audio_seconds"] if source_seconds is None else source_seconds
    )
    expected_input = _input_receipt(model_id, original, clips, repeats)
    for key, expected in expected_input.items():
        actual = inputs[key]
        if type(expected) is float:
            if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
                raise BenchmarkError()
        elif type(actual) is not type(expected) or actual != expected:
            raise BenchmarkError()
    for metric, numerator in (
        ("decode_rtf", "decode_seconds"),
        ("worker_wall_rtf", "worker_wall_seconds"),
    ):
        if not math.isclose(
            result[metric],
            result[numerator] / (original * repeats),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise BenchmarkError()

    binding = result["model_tuple"]
    if type(binding) is not dict or set(binding) != {"sha256", "files", "bytes"}:
        raise BenchmarkError()
    digest(binding["sha256"])
    integer(binding["files"], 1, MAX_MODEL_FILES)
    integer(binding["bytes"], 1, MAX_MODEL_BYTES)
    for name, required, allowed in (
        (
            "runtime_binding",
            {"executable_sha256"},
            {"executable_sha256", "venv_marker_sha256", "wheel_sha256", "wheel_bytes"},
        ),
        (
            "code_binding",
            {"benchmark_sha256"},
            {"benchmark_sha256", "moonshine_adapter_sha256"},
        ),
    ):
        value = result[name]
        if (
            type(value) is not dict
            or not required <= set(value)
            or set(value) - allowed
        ):
            raise BenchmarkError()
        for key, item in value.items():
            if key == "wheel_bytes":
                integer(item, 1, 512 * 1024 * 1024)
            else:
                digest(item)
        if name == "runtime_binding" and (
            ("wheel_sha256" in value) != ("wheel_bytes" in value)
        ):
            raise BenchmarkError()
    versions = result["versions"]
    expected_versions = (
        {"python", "moonshine-voice"}
        if model_id in MOONSHINE_IDS
        else {"python", "faster-whisper", "ctranslate2", "numpy"}
        if model_id == "faster-whisper-small"
        else {"python", "sherpa-onnx", "numpy"}
    )
    if (
        type(versions) is not dict
        or set(versions) != expected_versions
        or any(
            type(value) is not str or not re.fullmatch(r"[A-Za-z0-9.+_-]{1,64}", value)
            for value in versions.values()
        )
    ):
        raise BenchmarkError()
    policy = result["thread_policy"]
    if (
        type(policy) is not dict
        or set(policy)
        != {
            "requested_affinity_threads",
            "native_thread_budget_supported",
            "configured_native_threads",
            "moonshine_single_thread_env",
            "ort_single_thread",
        }
        or type(policy["native_thread_budget_supported"]) is not bool
        or type(policy["moonshine_single_thread_env"]) is not str
        or policy["moonshine_single_thread_env"] not in {"unset", "0", "1"}
    ):
        raise BenchmarkError()
    if (
        type(policy["ort_single_thread"]) is not bool
        or policy["ort_single_thread"] is not ort_single_thread
    ):
        raise BenchmarkError()
    integer(policy["requested_affinity_threads"], 1, 8)
    native_supported = model_id not in MOONSHINE_IDS
    if policy["native_thread_budget_supported"] is not native_supported:
        raise BenchmarkError()
    if native_supported:
        integer(
            policy["configured_native_threads"],
            policy["requested_affinity_threads"],
            policy["requested_affinity_threads"],
        )
    elif ort_single_thread:
        integer(policy["configured_native_threads"], 1, 1)
        if policy["moonshine_single_thread_env"] != "1":
            raise BenchmarkError()
    elif (
        policy["configured_native_threads"] is not None
        or policy["moonshine_single_thread_env"] != "0"
    ):
        raise BenchmarkError()
    affinity = result["cpu_affinity"]
    if (
        type(affinity) is not dict
        or set(affinity) != {"applied", "logical_cpus"}
        or type(affinity["applied"]) is not bool
    ):
        raise BenchmarkError()
    if affinity["applied"]:
        integer(affinity["logical_cpus"], 1, policy["requested_affinity_threads"])
    elif affinity["logical_cpus"] is not None:
        raise BenchmarkError()
    _validate_thread_affinity(
        result["thread_affinity"], 1 + clips * repeats, complete=True
    )
    coverage = result["thread_affinity"]
    if coverage["caller_mask_logical_cpus"] != affinity["logical_cpus"]:
        raise BenchmarkError()
    thread_fields = result["process_threads"]
    if type(thread_fields) is not dict or set(thread_fields) != {
        "before_model_load",
        "after_model_load",
        "maximum_sampled",
    }:
        raise BenchmarkError()
    for value in thread_fields.values():
        if value is not None:
            integer(value, 1, 65536)
    samples = [
        thread_fields[key]
        for key in ("before_model_load", "after_model_load")
        if thread_fields[key] is not None
    ]
    if thread_fields["maximum_sampled"] != (max(samples) if samples else None):
        raise BenchmarkError()


class _SafeParser(argparse.ArgumentParser):
    def error(self, _message):
        self.exit(2, '{"error":"english_asr_benchmark_arguments_invalid","ok":false}\n')


def _parser():
    parser = _SafeParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--timeout-sec", type=float, default=900)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker-request", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.worker_request is not None:
        return _worker(args.worker_request)
    try:
        _validate_limits(args.repeats, args.threads, args.timeout_sec)
        if (
            args.manifest is None
            or args.model_config is None
            or args.scratch_root is None
        ):
            raise BenchmarkError()
        corpus = load_corpus(args.manifest, args.source_root)
        models, config_digest = load_models(args.model_config)
        scratch = _absolute(args.scratch_root)
        if not scratch.is_dir() or stat.S_IMODE(scratch.lstat().st_mode) != 0o700:
            raise BenchmarkError()
        cells = [
            run_cell(
                model, corpus, scratch, args.repeats, args.threads, args.timeout_sec
            )
            for model in models
        ]
        verify_corpus(corpus)
        if (
            hashlib.sha256(
                _read(_absolute(args.model_config), MAX_MANIFEST_BYTES)
            ).hexdigest()
            != config_digest
        ):
            raise BenchmarkError()
        result = {
            "schema_version": 1,
            "ok": all(cell["status"] == "complete" for cell in cells),
            "scope": {
                "cpu_only": True,
                "after_pcm_only": True,
                "english_only": True,
                "reference_provenance": "caller_supplied_script"
                if corpus.kind == "owner-scripted"
                else "provided_legacy_labels",
                "human_recording_attested": False,
                "held_out": False,
                "live_latency": False,
                "endpoint_authority": False,
                "default_promotion": False,
                "phone_performance": False,
                "no_audio_device": True,
                "no_tts": True,
                "no_agent_control": True,
                "network_sandbox_attested": False,
            },
            "corpus": {
                "kind": corpus.kind,
                "sha256": corpus.digest,
                "manifest_sha256": corpus.manifest_digest,
                "clips": len(corpus.clips),
                "audio_seconds": corpus.seconds,
            },
            "config_sha256": config_digest,
            "execution": {
                "repeats": args.repeats,
                "threads": args.threads,
                "tail_padding_scope": "per_model_input_receipt",
                "streaming_input_finished_scope": "per_model_input_receipt",
                "timeout_seconds_per_model": args.timeout_sec,
                "accuracy_repeat": "first",
                "warm_latency_scope": "all_calls_except_first",
                "rss_scope": "isolated_process_high_water",
                "process_threads_scope": "before_and_after_model_load_only_not_continuous_peak",
                "all_thread_affinity_scope": "after_model_load_and_each_decode_samples_not_continuous_or_cgroup",
                "decode_rtf_scope": "decode_wall_over_repeated_source_audio",
                "process_wall_includes": "startup_load_decode_scoring_verification",
            },
            "cells": cells,
        }
        if args.output is not None:
            _write_new(Path(os.path.abspath(args.output)), result)
        print(_canonical(result).decode())
        return 0 if result["ok"] else 2
    except BaseException:
        print('{"error":"english_asr_benchmark_unavailable","ok":false}')
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
