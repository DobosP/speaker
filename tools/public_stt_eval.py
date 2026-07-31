"""Aggregate-only, whole-clip decoder smoke over public voice fixtures.

The report binds one validated manifest, one generated metadata file, every
prepared case hash, the complete local model directory tree, and the fixed
production decoder configuration.  Only ``transcript`` and
``empty_reference`` cases are decoded; every excluded assertion is still
validated, hashed, and counted.

This is deliberately not frame replay, endpointing, AEC, session-actor, tool,
owner-voice, held-out, or live-conversation latency evidence.  The production
recognizer contract is Linux with NVIDIA CUDA device 0 and float16.

Run it after preparing a public fixture suite::

    python -m tools.public_stt_eval \
        --manifest tests/fixtures/public_voice_manifest.v2.json \
        --metadata .cache/public_voice/v2/conversation/metadata.json \
        --model-path /path/to/local/model \
        --label large-v3-turbo
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import inspect
import io
import json
import math
import os
import re
import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath
from time import perf_counter_ns
from typing import Callable, Iterator, Mapping, Protocol, Sequence

_CPU_THREAD_DEFAULTS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
}
for _environment_name, _default_value in _CPU_THREAD_DEFAULTS.items():
    os.environ.setdefault(_environment_name, _default_value)

from core.engines._faster_whisper import (  # noqa: E402
    FasterWhisperEndpointRecognizer,
)
from core import wer as _wer_module  # noqa: E402
from tools.recorded_stt_eval import (  # noqa: E402
    AccuracyTotals,
    _measure,
    _sum_accuracy,
    _write_report,
)
from tools.public_voice_fixtures import (  # noqa: E402
    DEFAULT_MANIFEST,
    EVIDENCE_SCOPE,
    PublicFixtureError,
    PublicVoiceManifest,
    _clone_or_copy_file,
    _contains_token_subsequence,
    _normalised_word_tokens,
    _strict_json_loads,
    _unlock_private_tree,
    load_manifest,
    source_records_for_suite,
)

_SAFE_LABEL = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ERROR = {"ok": False, "error": "public_stt_prerequisites_unavailable"}
_MAX_CASE_AUDIO_BYTES = 8 * 1024 * 1024
_MAX_CORPUS_AUDIO_BYTES = 32 * 1024 * 1024
_METADATA_FIELDS = {
    "schema_version",
    "suite",
    "purpose",
    "evidence_scope",
    "audio_committed",
    "manifest_sha256",
    "selection_seed",
    "cases",
    "groups",
    "sources",
}
_CASE_FIELDS = {
    "name",
    "file",
    "expectation",
    "expected_text",
    "assertion",
    "tags",
    "sample_rate",
    "sample_count",
    "duration_sec",
    "sha256",
    "source_sha256",
    "selection",
}
_ELIGIBLE_ASSERTIONS = frozenset({"transcript", "empty_reference"})
_DECODER_IMPLEMENTATION = (
    "core.engines._faster_whisper.FasterWhisperEndpointRecognizer"
)
_MODEL_LOADER_CONFIG = {
    "device": "cuda",
    "device_index": 0,
    "compute_type": "float16",
    "num_workers": 1,
    "local_files_only": True,
}
_TRANSCRIBE_CONFIG = {
    "task": "transcribe",
    "language": "en",
    "beam_size": 5,
    "patience": 1.0,
    "temperature": 0.0,
    "compression_ratio_threshold": 2.4,
    "log_prob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    "vad_filter": False,
    "vad_parameters": None,
    "condition_on_previous_text": False,
    "initial_prompt": None,
    "prefix": None,
    "hotwords": None,
    "suppress_blank": True,
    "suppress_tokens": [-1],
    "without_timestamps": True,
    "word_timestamps": False,
}


class PublicSttPrerequisiteError(RuntimeError):
    """A deliberately detail-free corpus or model prerequisite failure."""


class _RecognitionResult(Protocol):
    text: str


class _Recognizer(Protocol):
    def warm(self) -> None: ...

    def transcribe(self, pcm: object, sample_rate: int) -> _RecognitionResult: ...


RecognizerFactory = Callable[[Path], _Recognizer]


@dataclass(frozen=True)
class _EvaluationCase:
    assertion: str
    reference: str = field(repr=False)
    audio_path: Path = field(repr=False)
    audio_bytes: bytes = field(repr=False)
    sha256: str = field(repr=False)
    sample_rate: int
    sample_count: int

    def __repr__(self) -> str:
        return (
            "_EvaluationCase("
            f"assertion={self.assertion!r}, sample_rate={self.sample_rate!r}, "
            f"sample_count={self.sample_count!r})"
        )


@dataclass(frozen=True)
class _LoadedCorpus:
    all_cases: tuple[_EvaluationCase, ...] = field(repr=False)
    eligible_cases: tuple[_EvaluationCase, ...] = field(repr=False)
    excluded_assertions: Mapping[str, int]
    binding: Mapping[str, object]


@dataclass(frozen=True)
class _DecodeRun:
    transcript: AccuracyTotals
    empty_reference: AccuracyTotals
    empty_reference_exact_raw: int
    empty_reference_nonempty_raw: int
    decode_latencies_ns: tuple[int, ...]
    audio_seconds: float
    decode_errors: int
    model_load_ns: int


def _fail() -> PublicSttPrerequisiteError:
    return PublicSttPrerequisiteError()


def _safe_label(value: object) -> str:
    if not isinstance(value, str) or _SAFE_LABEL.fullmatch(value) is None:
        raise _fail()
    return value


def _existing_file(value: object) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise _fail()
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        raise _fail() from None
    if not path.is_file():
        raise _fail()
    return path


def _local_directory(value: object) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise _fail()
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        raise _fail() from None
    if not path.is_dir():
        raise _fail()
    return path


def _positive_integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise _fail()
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    except OSError:
        raise _fail() from None
    return digest.hexdigest()


def _verified_file_bytes(path: Path, expected_hash: str, maximum: int) -> bytes:
    try:
        with path.open("rb") as source:
            payload = source.read(maximum + 1)
    except OSError:
        raise _fail() from None
    if len(payload) > maximum or hashlib.sha256(payload).hexdigest() != expected_hash:
        raise _fail()
    return payload


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _evaluator_binding(*, production_report: bool) -> dict[str, object]:
    repo_root = Path(__file__).resolve(strict=True).parents[1]
    candidates = {
        "tools/public_stt_eval.py": Path(__file__),
        "tools/recorded_stt_eval.py": Path(
            inspect.getsourcefile(_measure) or ""
        ),
        "core/wer.py": Path(
            inspect.getsourcefile(_wer_module.word_error_rate) or ""
        ),
    }
    files: dict[str, str] = {}
    for relative_path, candidate in candidates.items():
        try:
            actual = candidate.resolve(strict=True)
            expected = (repo_root / relative_path).resolve(strict=True)
        except (OSError, RuntimeError):
            raise _fail() from None
        if actual != expected or not actual.is_file():
            raise _fail()
        files[relative_path] = _sha256_file(actual)
    return {
        "schema_version": 1,
        "kind": "public_stt_aggregate_evaluator",
        "production_report": production_report,
        "files": files,
    }


def _strict_json_file(path: Path) -> tuple[bytes, Mapping[str, object]]:
    try:
        raw = path.read_bytes()
        payload = _strict_json_loads(raw)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        raise _fail() from None
    if not isinstance(payload, dict):
        raise _fail()
    return raw, payload


def _contained_audio_path(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise _fail()
    pure = PurePosixPath(value)
    if pure.is_absolute() or len(pure.parts) != 1 or pure.name != value:
        raise _fail()
    candidate = root / value
    if candidate.is_symlink():
        raise _fail()
    try:
        path = candidate.resolve(strict=True)
        path.relative_to(root)
    except (OSError, RuntimeError, ValueError):
        raise _fail() from None
    if not path.is_file() or path.is_symlink():
        raise _fail()
    return path


def _fixture_source_hashes(
    manifest: PublicVoiceManifest, fixture: Mapping[str, object]
) -> dict[str, str]:
    source_ids = [str(fixture["source_id"])] + [
        str(value) for value in fixture.get("required_source_ids", [])
    ]
    return {
        source_id: str(manifest.source_by_id[source_id]["sha256"])
        for source_id in source_ids
    }


def _valid_sha(value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise _fail()
    return value


def _validate_selection(
    manifest: PublicVoiceManifest,
    fixture: Mapping[str, object],
    row: Mapping[str, object],
) -> None:
    selection = row.get("selection")
    extract = fixture.get("extract")
    if not isinstance(selection, dict) or not isinstance(extract, dict):
        raise _fail()
    extractor = extract.get("type")
    if extractor == "speech_commands_tar":
        expected = {
            "archive_member",
            "archive_label",
            "speaker_id",
            "selection_rank_sha256",
        }
        if set(selection) != expected:
            raise _fail()
        member = selection.get("archive_member")
        label = selection.get("archive_label")
        speaker = selection.get("speaker_id")
        if (
            not isinstance(member, str)
            or not member
            or not isinstance(label, str)
            or not label
            or (speaker is not None and (not isinstance(speaker, str) or not speaker))
        ):
            raise _fail()
        member_path = PurePosixPath(member)
        if (
            member_path.is_absolute()
            or ".." in member_path.parts
            or "." in member_path.parts
            or len(member_path.parts) != 2
            or member_path.parts[0] != label
        ):
            raise _fail()
        expected_rank = hashlib.sha256(
            str(manifest.data["selection_seed"]).encode("utf-8")
            + b"\0"
            + member.encode("utf-8")
        ).hexdigest()
        if selection.get("selection_rank_sha256") != expected_rank:
            raise _fail()
        filename = member_path.parts[1]
        target = extract.get("target")
        if target == "_unknown_":
            if label in {
                "yes",
                "no",
                "up",
                "down",
                "left",
                "right",
                "on",
                "off",
                "stop",
                "go",
                "_silence_",
                "_background_noise_",
            }:
                raise _fail()
            if row.get("expected_text") != label:
                raise _fail()
        elif target == "_silence_":
            if label != "_silence_" or speaker is not None:
                raise _fail()
        elif label != target or speaker is None:
            raise _fail()
        if target != "_silence_":
            marker = "_nohash_"
            if marker not in filename or filename.split(marker, 1)[0] != speaker:
                raise _fail()
        return
    if extractor == "direct_wav":
        if selection:
            raise _fail()
        return
    if extractor == "tar_member":
        if selection != {"archive_member": extract.get("member")}:
            raise _fail()
        return
    if extractor != "wav_segment":
        raise _fail()
    required = {
        "start_sec",
        "end_sec",
        "active_speakers",
        "max_concurrent_speakers",
        "vocal_types",
        "annotation_tokens",
        "annotation_token_count",
        "annotation_transcript",
    }
    if set(selection) != required:
        raise _fail()
    if (
        selection.get("start_sec") != float(extract["start_sec"])
        or selection.get("end_sec") != float(extract["end_sec"])
        or type(selection.get("start_sec")) is not float
        or type(selection.get("end_sec")) is not float
    ):
        raise _fail()
    annotation = extract.get("annotation")
    tokens = selection.get("annotation_tokens")
    vocals = selection.get("vocal_types")
    if (
        not isinstance(annotation, dict)
        or not isinstance(tokens, list)
        or any(not isinstance(item, str) or not item for item in tokens)
        or selection.get("annotation_token_count") != len(tokens)
        or selection.get("annotation_transcript") != " ".join(tokens)
        or not isinstance(vocals, list)
        or vocals != sorted(set(vocals))
        or any(not isinstance(item, str) or not item for item in vocals)
    ):
        raise _fail()
    active = selection.get("active_speakers")
    concurrent = selection.get("max_concurrent_speakers")
    if (
        isinstance(active, bool)
        or not isinstance(active, int)
        or isinstance(concurrent, bool)
        or not isinstance(concurrent, int)
        or not int(annotation["min_active_speakers"])
        <= active
        <= int(annotation["max_active_speakers"])
        or concurrent < int(annotation["min_concurrent_speakers"])
        or concurrent > active
    ):
        raise _fail()
    required_vocal = annotation.get("required_vocal_type")
    if required_vocal is not None and required_vocal not in vocals:
        raise _fail()
    if fixture.get("assertion") == "transcript":
        expected_tokens = _normalised_word_tokens(str(fixture.get("expected_text") or ""))
        observed_tokens = _normalised_word_tokens(" ".join(tokens))
        if not _contains_token_subsequence(observed_tokens, expected_tokens):
            raise _fail()


def _validate_case_row(
    *,
    manifest: PublicVoiceManifest,
    fixture: Mapping[str, object],
    row: Mapping[str, object],
    root: Path,
    maximum_audio_bytes: int,
) -> _EvaluationCase:
    if set(row) != _CASE_FIELDS:
        raise _fail()
    fixture_id = str(fixture["id"])
    assertion = str(fixture["assertion"])
    if (
        row.get("name") != fixture_id
        or row.get("file") != f"{fixture_id}.npy"
        or row.get("assertion") != assertion
        or _canonical_bytes(row.get("tags"))
        != _canonical_bytes(list(fixture.get("tags", [])))
        or type(row.get("sample_rate")) is not int
        or row.get("sample_rate") != manifest.data["output_sample_rate"]
        or _canonical_bytes(row.get("source_sha256"))
        != _canonical_bytes(_fixture_source_hashes(manifest, fixture))
    ):
        raise _fail()
    expected = row.get("expected_text")
    if not isinstance(expected, str) or not isinstance(row.get("expectation"), str):
        raise _fail()
    if assertion == "transcript":
        if expected != fixture.get("expected_text") or row.get("expectation") != expected:
            raise _fail()
    elif assertion == "unknown_word":
        if not expected or row.get("expectation") != expected:
            raise _fail()
    else:
        if expected != "" or row.get("expectation") != assertion:
            raise _fail()
    sample_count = _positive_integer(row.get("sample_count"))
    duration = row.get("duration_sec")
    if (
        type(duration) is not float
        or not math.isfinite(duration)
        or duration
        != round(sample_count / int(manifest.data["output_sample_rate"]), 6)
    ):
        raise _fail()
    expected_hash = _valid_sha(row.get("sha256"))
    audio_path = _contained_audio_path(root, row.get("file"))
    audio_bytes = _verified_file_bytes(
        audio_path,
        expected_hash,
        maximum=min(
            sample_count * 4 + 64 * 1024,
            _MAX_CASE_AUDIO_BYTES,
            maximum_audio_bytes,
        ),
    )
    _validate_selection(manifest, fixture, row)
    return _EvaluationCase(
        assertion=assertion,
        reference=expected,
        audio_path=audio_path,
        audio_bytes=audio_bytes,
        sha256=expected_hash,
        sample_rate=int(manifest.data["output_sample_rate"]),
        sample_count=sample_count,
    )


def _validate_audio_file(case: _EvaluationCase) -> None:
    import numpy as np

    samples: object | None = None
    try:
        samples = np.load(io.BytesIO(case.audio_bytes), allow_pickle=False)
        if (
            not isinstance(samples, np.ndarray)
            or samples.ndim != 1
            or samples.dtype != np.float32
            or samples.size != case.sample_count
            or not np.isfinite(samples).all()
        ):
            raise _fail()
    except PublicSttPrerequisiteError:
        raise
    except Exception:
        raise _fail() from None
    finally:
        if samples is not None:
            _close_memmap(samples)


def _load_corpus(
    metadata_path: Path, manifest: PublicVoiceManifest
) -> _LoadedCorpus:
    raw, payload = _strict_json_file(metadata_path)
    if (
        set(payload) != _METADATA_FIELDS
        or type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != 2
        or payload.get("purpose") != manifest.data["purpose"]
        or _canonical_bytes(payload.get("evidence_scope"))
        != _canonical_bytes(EVIDENCE_SCOPE)
        or payload.get("audio_committed") is not False
        or payload.get("manifest_sha256") != manifest.digest
        or payload.get("selection_seed") != manifest.data["selection_seed"]
    ):
        raise _fail()
    suite = payload.get("suite")
    if not isinstance(suite, str):
        raise _fail()
    fixtures = tuple(
        fixture for fixture in manifest.fixtures if fixture["suite"] == suite
    )
    if not fixtures:
        raise _fail()
    if _canonical_bytes(payload.get("groups")) != _canonical_bytes(
        [group for group in manifest.groups if group.get("suite") == suite]
    ) or _canonical_bytes(payload.get("sources")) != _canonical_bytes(
        list(source_records_for_suite(manifest, suite))
    ):
        raise _fail()
    rows = payload.get("cases")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise _fail()
    names = [row.get("name") for row in rows]
    expected_names = [fixture["id"] for fixture in fixtures]
    if (
        any(not isinstance(name, str) for name in names)
        or len(set(names)) != len(names)
        or set(names) != set(expected_names)
        or len(rows) != len(fixtures)
    ):
        raise _fail()
    row_by_name = {str(row["name"]): row for row in rows}
    all_cases_list: list[_EvaluationCase] = []
    retained_audio_bytes = 0
    for fixture in fixtures:
        case = _validate_case_row(
            manifest=manifest,
            fixture=fixture,
            row=row_by_name[str(fixture["id"])],
            root=metadata_path.parent,
            maximum_audio_bytes=_MAX_CORPUS_AUDIO_BYTES - retained_audio_bytes,
        )
        retained_audio_bytes += len(case.audio_bytes)
        if retained_audio_bytes > _MAX_CORPUS_AUDIO_BYTES:
            raise _fail()
        all_cases_list.append(case)
    all_cases = tuple(all_cases_list)
    for case in all_cases:
        _validate_audio_file(case)
    eligible = tuple(
        case for case in all_cases if case.assertion in _ELIGIBLE_ASSERTIONS
    )
    if not eligible:
        raise _fail()
    excluded = Counter(
        case.assertion
        for case in all_cases
        if case.assertion not in _ELIGIBLE_ASSERTIONS
    )
    source_records = list(source_records_for_suite(manifest, suite))
    case_records = sorted(
        (
            {
                "assertion": case.assertion,
                "file": case.audio_path.name,
                "sample_count": case.sample_count,
                "sample_rate": case.sample_rate,
                "sha256": case.sha256,
            }
            for case in all_cases
        ),
        key=lambda item: str(item["file"]),
    )
    return _LoadedCorpus(
        all_cases=all_cases,
        eligible_cases=eligible,
        excluded_assertions=dict(sorted(excluded.items())),
        binding={
            "manifest_sha256": manifest.digest,
            "metadata_sha256": hashlib.sha256(raw).hexdigest(),
            "suite": suite,
            "source_set_sha256": _canonical_digest(source_records),
            "sources": len(source_records),
            "case_set_sha256": _canonical_digest(case_records),
            "cases": len(all_cases),
        },
    )


def _verify_case_snapshot(cases: Sequence[_EvaluationCase]) -> None:
    for case in cases:
        if _sha256_file(case.audio_path) != case.sha256:
            raise _fail()


def _fingerprint_model_tree(root: Path) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    files = 0
    directories = 0
    total_bytes = 0
    try:
        paths = sorted(root.rglob("*"), key=lambda path: path.relative_to(root).as_posix())
        for path in paths:
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                resolved = path.resolve(strict=True)
                if not resolved.is_file():
                    raise _fail()
                size = resolved.stat().st_size
                files += 1
                total_bytes += size
                entries.append(
                    {
                        "path": relative,
                        "type": "file",
                        "size": size,
                        "sha256": _sha256_file(resolved),
                    }
                )
            elif path.is_dir():
                directories += 1
                entries.append({"path": relative, "type": "directory"})
            elif path.is_file():
                size = path.stat().st_size
                files += 1
                total_bytes += size
                entries.append(
                    {
                        "path": relative,
                        "type": "file",
                        "size": size,
                        "sha256": _sha256_file(path),
                    }
                )
            else:
                raise _fail()
    except PublicSttPrerequisiteError:
        raise
    except OSError:
        raise _fail() from None
    if files == 0:
        raise _fail()
    return {
        "tree_sha256": _canonical_digest(entries),
        "files": files,
        "directories": directories,
        "bytes": total_bytes,
    }


@contextmanager
def _model_directory_snapshot(
    source_root: Path,
    source_binding: Mapping[str, object],
    *,
    snapshot_parent: Path,
) -> Iterator[Path]:
    """Yield a private read-only materialized copy of the bound model tree."""

    try:
        snapshot_parent = snapshot_parent.resolve(strict=True)
    except (OSError, RuntimeError):
        raise _fail() from None
    if not snapshot_parent.is_dir():
        raise _fail()
    try:
        temporary_root = Path(
            tempfile.mkdtemp(
                prefix=".speaker-public-stt-model-",
                dir=snapshot_parent,
            )
        )
    except OSError:
        raise _fail() from None
    snapshot_root = temporary_root / "model"
    try:
        snapshot_root.mkdir(mode=0o700)
        paths = sorted(
            source_root.rglob("*"),
            key=lambda path: path.relative_to(source_root).as_posix(),
        )
        for source_path in paths:
            relative = source_path.relative_to(source_root)
            destination = snapshot_root / relative
            if source_path.is_symlink():
                resolved = source_path.resolve(strict=True)
                if not resolved.is_file():
                    raise _fail()
                _clone_or_copy_file(resolved, destination)
            elif source_path.is_dir():
                destination.mkdir(mode=0o700)
            elif source_path.is_file():
                _clone_or_copy_file(source_path, destination)
            else:
                raise _fail()
        if _fingerprint_model_tree(snapshot_root) != dict(source_binding):
            raise _fail()
        directories = sorted(
            (path for path in snapshot_root.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        )
        for directory in directories:
            directory.chmod(0o500)
        snapshot_root.chmod(0o500)
        if _fingerprint_model_tree(snapshot_root) != dict(source_binding):
            raise _fail()
        yield snapshot_root
        if _fingerprint_model_tree(snapshot_root) != dict(source_binding):
            raise _fail()
    except PublicFixtureError:
        raise _fail() from None
    except PublicSttPrerequisiteError:
        raise
    except OSError:
        raise _fail() from None
    finally:
        _unlock_private_tree(temporary_root)
        shutil.rmtree(temporary_root, ignore_errors=True)


def _callable_source_binding(factory: RecognizerFactory) -> dict[str, object]:
    try:
        source_text = inspect.getsource(factory).encode("utf-8")
        source_file = inspect.getsourcefile(factory)
    except (OSError, TypeError):
        raise _fail() from None
    if source_file is None:
        raise _fail()
    return {
        "kind": "test_recognizer_override",
        "production": False,
        "callable": (
            f"{getattr(factory, '__module__', '<unknown>')}."
            f"{getattr(factory, '__qualname__', '<unknown>')}"
        ),
        "callable_source_sha256": hashlib.sha256(source_text).hexdigest(),
        "module_file_sha256": _sha256_file(Path(source_file)),
    }


def _decoder_binding(
    factory: RecognizerFactory,
    *,
    allow_test_recognizer_override: bool,
) -> dict[str, object]:
    if factory is not FasterWhisperEndpointRecognizer:
        if not allow_test_recognizer_override:
            raise _fail()
        return _callable_source_binding(factory)
    if not sys.platform.startswith("linux"):
        raise _fail()
    source = inspect.getsourcefile(FasterWhisperEndpointRecognizer)
    if source is None:
        raise _fail()
    try:
        packages = {
            "ctranslate2": version("ctranslate2"),
            "faster-whisper": version("faster-whisper"),
        }
    except PackageNotFoundError:
        raise _fail() from None
    return {
        "kind": "production_faster_whisper",
        "production": True,
        "implementation": _DECODER_IMPLEMENTATION,
        "implementation_sha256": _sha256_file(Path(source)),
        "packages": packages,
        "runtime": {
            "platform": "linux",
            "accelerator": "nvidia-cuda",
            "device_index": 0,
            "compute_type": "float16",
        },
        "model_loader": dict(_MODEL_LOADER_CONFIG),
        "transcribe": dict(_TRANSCRIBE_CONFIG),
    }


def _percentile_ms(values_ns: Sequence[int], percentile: float) -> float:
    if not values_ns:
        return 0.0
    ordered = sorted(values_ns)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return round(ordered[index] / 1_000_000, 3)


def _close_memmap(samples: object) -> None:
    mapped = getattr(samples, "_mmap", None)
    close = getattr(mapped, "close", None)
    if callable(close):
        close()


def _load_audio(case: _EvaluationCase) -> object:
    import numpy as np

    try:
        loaded = np.load(io.BytesIO(case.audio_bytes), allow_pickle=False)
        if (
            not isinstance(loaded, np.ndarray)
            or loaded.ndim != 1
            or loaded.dtype != np.float32
            or loaded.size != case.sample_count
            or not np.isfinite(loaded).all()
        ):
            raise _fail()
        samples = np.array(loaded, dtype=np.float32, order="C", copy=True)
        samples.setflags(write=False)
        return samples
    except PublicSttPrerequisiteError:
        raise
    except Exception:
        raise _fail() from None


def _add_measurement(
    totals: AccuracyTotals, reference: str, hypothesis: str
) -> AccuracyTotals:
    return _sum_accuracy(totals, _measure(((reference, hypothesis),)))


def _decode_corpus(
    corpus: _LoadedCorpus,
    model_snapshot: Path,
    *,
    recognizer_factory: RecognizerFactory,
    clock_ns: Callable[[], int],
) -> _DecodeRun:
    try:
        recognizer = recognizer_factory(model_snapshot)
        load_started = clock_ns()
        recognizer.warm()
        load_ns = max(0, clock_ns() - load_started)
    except Exception:
        raise _fail() from None

    transcript = AccuracyTotals()
    empty_reference = AccuracyTotals()
    empty_reference_exact_raw = 0
    empty_reference_nonempty_raw = 0
    decode_latencies_ns: list[int] = []
    audio_seconds = 0.0
    decode_errors = 0

    try:
        for case in corpus.eligible_cases:
            samples: object | None = None
            hypothesis = ""
            decoded = False
            started: int | None = None
            try:
                samples = _load_audio(case)
                audio_seconds += len(samples) / float(case.sample_rate)  # type: ignore[arg-type]
                started = clock_ns()
                result = recognizer.transcribe(samples, case.sample_rate)
                if not isinstance(result.text, str):
                    raise ValueError("recognizer returned invalid text")
                hypothesis = result.text
                decoded = True
            except PublicSttPrerequisiteError:
                raise
            except Exception:
                decode_errors += 1
            finally:
                if started is not None:
                    decode_latencies_ns.append(max(0, clock_ns() - started))
                if samples is not None:
                    _close_memmap(samples)

            if case.assertion == "transcript":
                transcript = _add_measurement(transcript, case.reference, hypothesis)
            else:
                empty_reference = _add_measurement(
                    empty_reference, case.reference, hypothesis
                )
                if decoded and hypothesis.strip():
                    empty_reference_nonempty_raw += 1
                elif decoded:
                    empty_reference_exact_raw += 1
    finally:
        del recognizer
        gc.collect()

    return _DecodeRun(
        transcript=transcript,
        empty_reference=empty_reference,
        empty_reference_exact_raw=empty_reference_exact_raw,
        empty_reference_nonempty_raw=empty_reference_nonempty_raw,
        decode_latencies_ns=tuple(decode_latencies_ns),
        audio_seconds=audio_seconds,
        decode_errors=decode_errors,
        model_load_ns=load_ns,
    )


def evaluate_public_stt(
    metadata_path: Path | str,
    model_path: Path | str,
    *,
    label: str,
    manifest_path: Path | str = DEFAULT_MANIFEST,
    snapshot_root: Path | str | None = None,
    recognizer_factory: RecognizerFactory = FasterWhisperEndpointRecognizer,
    allow_test_recognizer_override: bool = False,
    clock_ns: Callable[[], int] = perf_counter_ns,
) -> dict[str, object]:
    """Evaluate one exact local corpus/model/decoder tuple."""

    safe_label = _safe_label(label)
    metadata = _existing_file(metadata_path)
    local_model = _local_directory(model_path)
    try:
        manifest = load_manifest(_existing_file(manifest_path))
    except (PublicFixtureError, PublicSttPrerequisiteError):
        raise _fail() from None
    corpus = _load_corpus(metadata, manifest)
    model_binding = _fingerprint_model_tree(local_model)
    decoder_binding = _decoder_binding(
        recognizer_factory,
        allow_test_recognizer_override=allow_test_recognizer_override,
    )
    evaluator_binding = _evaluator_binding(
        production_report=bool(decoder_binding["production"])
    )
    snapshot_parent = (
        local_model.parent
        if snapshot_root is None
        else _local_directory(snapshot_root)
    )
    if snapshot_parent == local_model or snapshot_parent.is_relative_to(local_model):
        raise _fail()
    with _model_directory_snapshot(
        local_model,
        model_binding,
        snapshot_parent=snapshot_parent,
    ) as model_snapshot:
        run = _decode_corpus(
            corpus,
            model_snapshot,
            recognizer_factory=recognizer_factory,
            clock_ns=clock_ns,
        )
        _verify_case_snapshot(corpus.all_cases)
        if (
            _sha256_file(metadata) != corpus.binding["metadata_sha256"]
            or _sha256_file(manifest.path) != corpus.binding["manifest_sha256"]
            or _fingerprint_model_tree(local_model) != model_binding
            or _evaluator_binding(
                production_report=bool(decoder_binding["production"])
            )
            != evaluator_binding
        ):
            raise _fail()

    transcript = run.transcript
    empty_reference = run.empty_reference
    combined_clips = transcript.clips + empty_reference.clips
    combined_exact = transcript.exact + run.empty_reference_exact_raw
    decode_total_ns = sum(run.decode_latencies_ns)
    model_report = dict(model_binding)
    model_report["evaluation_input"] = {
        "kind": "private_materialized_snapshot",
        "copy_strategy": "reflink_first_file_copy_fallback",
        "logical_bytes": model_binding["bytes"],
    }
    return {
        "ok": run.decode_errors == 0,
        "label": safe_label,
        "evidence": {
            "kind": "whole_clip_decoder_smoke",
            "frame_replay": False,
            "aec_behavior": False,
            "live_hardware": False,
            "owner_voice": False,
            "held_out": False,
        },
        "corpus": dict(corpus.binding),
        "model": model_report,
        "decoder": decoder_binding,
        "evaluator": evaluator_binding,
        "cases": {
            "all": len(corpus.all_cases),
            "eligible": combined_clips,
            "transcript": transcript.clips,
            "empty_reference": empty_reference.clips,
            "excluded_assertions": dict(corpus.excluded_assertions),
        },
        "accuracy": {
            "transcript": transcript.as_dict(),
            "empty_reference": {
                "clips": empty_reference.clips,
                "exact_empty": run.empty_reference_exact_raw,
                "nonempty_transcripts": run.empty_reference_nonempty_raw,
                "wer": round(empty_reference.wer, 4),
                "cer": round(empty_reference.cer, 4),
                "word_errors": empty_reference.word_errors,
                "char_errors": empty_reference.char_edits,
            },
            "combined_exact": combined_exact,
            "combined_exact_rate": (
                round(combined_exact / combined_clips, 4) if combined_clips else 0.0
            ),
            "decode_errors": run.decode_errors,
        },
        "latency": {
            "model_load_ms": round(run.model_load_ns / 1_000_000, 3),
            "decode_p50_ms": _percentile_ms(run.decode_latencies_ns, 0.50),
            "decode_p95_ms": _percentile_ms(run.decode_latencies_ns, 0.95),
            "decode_max_ms": _percentile_ms(run.decode_latencies_ns, 1.0),
            "decode_total_ms": round(decode_total_ns / 1_000_000, 3),
            "audio_total_sec": round(run.audio_seconds, 3),
            "aggregate_rtf": (
                round((decode_total_ns / 1_000_000_000) / run.audio_seconds, 4)
                if run.audio_seconds
                else 0.0
            ),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate-only whole-clip decoder smoke for one exact public "
            "corpus/model/decoder tuple. Not frame replay, AEC, or live latency."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--snapshot-root",
        type=Path,
        help=(
            "existing on-disk directory for the temporary private model copy "
            "(defaults to the model directory's parent)"
        ),
    )
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="optional aggregate-only JSON report (atomically written mode 600)",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    recognizer_factory: RecognizerFactory = FasterWhisperEndpointRecognizer,
    allow_test_recognizer_override: bool = False,
    clock_ns: Callable[[], int] = perf_counter_ns,
) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = evaluate_public_stt(
            args.metadata,
            args.model_path,
            label=args.label,
            manifest_path=args.manifest,
            snapshot_root=args.snapshot_root,
            recognizer_factory=recognizer_factory,
            allow_test_recognizer_override=allow_test_recognizer_override,
            clock_ns=clock_ns,
        )
        if args.output is not None:
            _write_report(args.output, payload)
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        return 0 if payload["ok"] else 1
    except Exception:  # noqa: BLE001 - CLI failures must remain detail-free
        print(json.dumps(_SAFE_ERROR, sort_keys=True, separators=(",", ":")))
        return 2


if __name__ == "__main__":  # pragma: no cover - exercised through main tests
    raise SystemExit(main())
