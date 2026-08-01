"""Prepare a private Romanian-accented English VoxPopuli STT slice.

The command has no downloader or network client.  It requires the two exact
``en_accented`` test Parquet shards as explicit absolute local overrides and an
explicit isolated PyArrow interpreter.  A private worker selects 24 gold
Romanian-accented English utterances from distinct speakers, six in each of
four duration bands, and the parent publishes schema-v2 f32le replay through
the shared hardened corpus writer.  CLI output is aggregate-only.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
from typing import Callable, Iterator, Mapping, Sequence

from core.wer import normalize
from tools import public_voice_fixtures as public_fixtures
from tools import voxpopuli_ro_accent_parquet_worker as worker_contract
from tools.streaming_stt.bounded_io import BoundedReadError, hash_regular_bounded
from tools.streaming_stt.corpus import CorpusProvenance, LoadedCorpus
from tools.streaming_stt.corpus_writer import (
    CorpusWriteCase,
    CorpusWriterError,
    publish_private_corpus,
)
from tools.streaming_stt.protocol import MAX_CORPUS_BYTES, MAX_PCM_BYTES


DATASET_ID = "voxpopuli_en_accented_ro"
CONFIG = "en_accented"
SPLIT = "test"
SOURCE_REVISION = worker_contract.SOURCE_REVISION
SELECTION_SEED = worker_contract.SELECTION_SEED
SELECTION_COUNT = worker_contract.SELECTION_COUNT
SOURCE_TOTAL_ROWS = worker_contract.SOURCE_TOTAL_ROWS
PYARROW_VERSION = worker_contract.PYARROW_VERSION
PROTOCOL_VERSION = worker_contract.PROTOCOL_VERSION
SCHEMA_CONTRACT_SHA256 = worker_contract.SCHEMA_CONTRACT_SHA256
DURATION_BUCKETS = worker_contract.DURATION_BUCKETS

DATASET_CARD_URL = (
    "https://huggingface.co/datasets/facebook/voxpopuli/blob/main/README.md"
)
UPSTREAM_REPOSITORY_URL = "https://github.com/facebookresearch/voxpopuli"
PAPER_URL = "https://aclanthology.org/2021.acl-long.80"
CC0_URL = "https://creativecommons.org/publicdomain/zero/1.0/"
EP_MEDIA_URL = "https://multimedia.europarl.europa.eu/en/home"
EP_LEGAL_NOTICE_URL = "https://www.europarl.europa.eu/legal-notice/en/"
REQUIRED_TERMS = frozenset({"CC0-1.0", "EUROPEAN-PARLIAMENT-LEGAL-NOTICE"})


@dataclass(frozen=True, slots=True)
class SourceShard:
    shard_index: int
    filename: str
    size_bytes: int
    sha256: str
    download_url: str


SOURCE_SHARDS = tuple(
    SourceShard(
        shard_index=int(item["shard_index"]),
        filename=str(item["filename"]),
        size_bytes=int(item["size_bytes"]),
        sha256=str(item["sha256"]),
        download_url=(
            "https://huggingface.co/datasets/facebook/voxpopuli/resolve/"
            f"{SOURCE_REVISION}/en_accented/{item['filename']}"
        ),
    )
    for item in worker_contract.SOURCE_SHARDS
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PRIVATE_CACHE_ROOT = _REPO_ROOT / ".cache"
_WORKER_PATH = Path(worker_contract.__file__).resolve(strict=True)
_MAX_WORKER_BYTES = 512 * 1024
_MAX_WORKER_RESULT_BYTES = 256 * 1024
_MAX_AUDIO_BYTES = 16 * 1024 * 1024
_MAX_RECEIPT_BYTES = 256 * 1024
_WORKER_TIMEOUT_SEC = 900.0
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ID_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}\Z")
_SAFE_ERROR = {
    "ok": False,
    "error": "voxpopuli_ro_accent_prerequisites_unavailable",
}


class VoxPopuliPreparationError(RuntimeError):
    """A detail-free source, worker, selection, decode, or publication error."""


@dataclass(frozen=True, slots=True)
class TestWorkerInjection:
    """Test-only worker seam that cannot claim production PyArrow evidence."""

    run: Callable[[tuple[int, ...], Path], None] = field(repr=False)
    fixture_id: str


@dataclass(frozen=True, slots=True)
class PreparedVoxPopuliCorpus:
    corpus: LoadedCorpus
    receipt_sha256: str
    selection_recipe_sha256: str
    metadata_sha256: str


@dataclass(frozen=True, slots=True)
class _SourceHandle:
    shard: SourceShard
    path: Path = field(repr=False)
    descriptor: int = field(repr=False)
    identity: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _WorkerRun:
    output_dir: Path = field(repr=False)
    binding: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _ValidatedWorker:
    cases: tuple[CorpusWriteCase, ...] = field(repr=False)
    receipt_cases: tuple[Mapping[str, object], ...]
    metadata_binding: Mapping[str, object]
    source_rows: tuple[Mapping[str, object], ...]
    worker_summary: Mapping[str, object]


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, UnicodeError, ValueError, OverflowError):
        raise VoxPopuliPreparationError("receipt contract failed") from None


def _strict_json(raw: bytes) -> object:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise VoxPopuliPreparationError("worker result failed")
            result[key] = value
        return result

    try:
        return json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except VoxPopuliPreparationError:
        raise
    except (UnicodeError, ValueError, OverflowError):
        raise VoxPopuliPreparationError("worker result failed") from None


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _sha256(value: object) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise VoxPopuliPreparationError("worker result failed")
    return value


def _private_file_bytes(path: Path, *, maximum: int) -> bytes:
    try:
        with public_fixtures._stable_regular_reader(
            path,
            field="private worker output",
            maximum=maximum,
        ) as (handle, opened):
            if stat.S_IMODE(opened.st_mode) & 0o077:
                raise VoxPopuliPreparationError("worker result failed")
            payload = handle.read(opened.st_size + 1)
            if len(payload) != opened.st_size:
                raise VoxPopuliPreparationError("worker result failed")
            return payload
    except VoxPopuliPreparationError:
        raise
    except public_fixtures.PublicFixtureError:
        raise VoxPopuliPreparationError("worker result failed") from None


def _canonical_external_path(path: Path | str, *, must_exist: bool) -> Path:
    supplied = Path(path).expanduser()
    if not supplied.is_absolute():
        raise VoxPopuliPreparationError("path prerequisite failed")
    lexical = Path(os.path.abspath(supplied))
    try:
        if must_exist:
            resolved = lexical.resolve(strict=True)
        else:
            resolved_parent = lexical.parent.resolve(strict=True)
            resolved = resolved_parent / lexical.name
    except (OSError, RuntimeError, ValueError):
        raise VoxPopuliPreparationError("path prerequisite failed") from None
    inside_repo = lexical == _REPO_ROOT or _REPO_ROOT in lexical.parents
    inside_private_cache = (
        lexical == _PRIVATE_CACHE_ROOT or _PRIVATE_CACHE_ROOT in lexical.parents
    )
    if resolved != lexical or (inside_repo and not inside_private_cache):
        raise VoxPopuliPreparationError("path prerequisite failed")
    return lexical


def _validated_output_path(path: Path | str) -> Path:
    candidate = _canonical_external_path(path, must_exist=False)
    if (
        not candidate.name
        or candidate.name in {".", ".."}
        or candidate.exists()
        or candidate.is_symlink()
    ):
        raise VoxPopuliPreparationError("output prerequisite failed")
    return candidate


def _source_path(path: Path | str, shard: SourceShard) -> tuple[Path, os.stat_result]:
    candidate = _canonical_external_path(path, must_exist=True)
    try:
        metadata = candidate.lstat()
    except OSError:
        raise VoxPopuliPreparationError("source prerequisite failed") from None
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_size != shard.size_bytes
        or stat.S_IMODE(metadata.st_mode) & 0o077
        or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
    ):
        raise VoxPopuliPreparationError("source prerequisite failed")
    return candidate, metadata


@contextmanager
def _opened_sources(paths: Sequence[Path | str]) -> Iterator[tuple[_SourceHandle, ...]]:
    if len(paths) != len(SOURCE_SHARDS):
        raise VoxPopuliPreparationError("source prerequisite failed")
    handles: list[_SourceHandle] = []
    try:
        for shard, supplied in zip(SOURCE_SHARDS, paths, strict=True):
            path, before = _source_path(supplied, shard)
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, flags)
            opened = os.fstat(descriptor)
            if _identity(opened) != _identity(before):
                os.close(descriptor)
                raise VoxPopuliPreparationError("source prerequisite failed")
            handles.append(
                _SourceHandle(
                    shard=shard,
                    path=path,
                    descriptor=descriptor,
                    identity=_identity(opened),
                )
            )
        yield tuple(handles)
        _verify_sources(tuple(handles), hash_content=False)
    except VoxPopuliPreparationError:
        raise
    except (OSError, OverflowError, ValueError):
        raise VoxPopuliPreparationError("source prerequisite failed") from None
    finally:
        for handle in handles:
            try:
                os.close(handle.descriptor)
            except OSError:
                pass


def _descriptor_sha256(descriptor: int, expected_size: int) -> str:
    try:
        digest = hashlib.sha256()
        consumed = 0
        while consumed <= expected_size:
            chunk = os.pread(
                descriptor,
                min(1024 * 1024, expected_size + 1 - consumed),
                consumed,
            )
            if not chunk:
                break
            consumed += len(chunk)
            if consumed > expected_size:
                raise VoxPopuliPreparationError("source verification failed")
            digest.update(chunk)
        if consumed != expected_size:
            raise VoxPopuliPreparationError("source verification failed")
        return digest.hexdigest()
    except VoxPopuliPreparationError:
        raise
    except (OSError, OverflowError, ValueError):
        raise VoxPopuliPreparationError("source verification failed") from None


def _verify_sources(handles: tuple[_SourceHandle, ...], *, hash_content: bool) -> None:
    for handle in handles:
        try:
            opened = os.fstat(handle.descriptor)
            current = handle.path.lstat()
            if (
                _identity(opened) != handle.identity
                or _identity(current) != handle.identity
                or handle.path.resolve(strict=True) != handle.path
                or (
                    hash_content
                    and _descriptor_sha256(handle.descriptor, handle.shard.size_bytes)
                    != handle.shard.sha256
                )
            ):
                raise VoxPopuliPreparationError("source verification failed")
        except VoxPopuliPreparationError:
            raise
        except (OSError, RuntimeError, ValueError):
            raise VoxPopuliPreparationError("source verification failed") from None


def _write_private_json(path: Path, value: object) -> None:
    payload = _canonical_json_bytes(value)
    try:
        with path.open("xb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        path.chmod(0o400)
    except OSError:
        raise VoxPopuliPreparationError("worker prerequisite failed") from None


def _worker_environment(root: Path) -> dict[str, str]:
    return {
        "HOME": str(root),
        "TMPDIR": str(root),
        "PATH": os.defpath,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "ARROW_NUM_THREADS": "1",
    }


@contextmanager
def _run_worker(
    handles: tuple[_SourceHandle, ...],
    *,
    output_parent: Path,
    parquet_python: Path | str | None,
    test_worker_injection: TestWorkerInjection | None,
) -> Iterator[_WorkerRun]:
    root = Path(tempfile.mkdtemp(prefix=".speaker-voxpopuli-", dir=output_parent))
    root.chmod(0o700)
    output = root / "output"
    output.mkdir(mode=0o700)
    process: subprocess.Popen[bytes] | None = None
    group_reaped = False
    reap_attempted = False
    try:
        if test_worker_injection is not None:
            if (
                type(test_worker_injection) is not TestWorkerInjection
                or not callable(test_worker_injection.run)
                or _SAFE_ID_RE.fullmatch(test_worker_injection.fixture_id) is None
            ):
                raise VoxPopuliPreparationError("worker prerequisite failed")
            try:
                test_worker_injection.run(
                    tuple(handle.descriptor for handle in handles), output
                )
            except VoxPopuliPreparationError:
                raise
            except Exception:
                raise VoxPopuliPreparationError("worker failed") from None
            yield _WorkerRun(
                output_dir=output,
                binding={
                    "contract": "test-injected-voxpopuli-worker-v1",
                    "fixture_id": test_worker_injection.fixture_id,
                    "production_evidence": False,
                },
            )
            return

        try:
            interpreter = public_fixtures._validate_parquet_python(parquet_python)
        except public_fixtures.PublicFixtureError:
            raise VoxPopuliPreparationError("worker prerequisite failed") from None
        worker_payload = public_fixtures._read_stable_repo_worker(_WORKER_PATH)
        if not worker_payload or len(worker_payload) > _MAX_WORKER_BYTES:
            raise VoxPopuliPreparationError("worker prerequisite failed")
        worker_sha256 = hashlib.sha256(worker_payload).hexdigest()
        worker_snapshot = root / "voxpopuli_ro_accent_parquet_worker.py"
        public_fixtures._write_private_worker_snapshot(worker_snapshot, worker_payload)
        request_path = root / "request.json"
        request = {
            "schema_version": PROTOCOL_VERSION,
            "source_revision": SOURCE_REVISION,
            "selection_seed": SELECTION_SEED,
            "source_descriptors": [
                {
                    "shard_index": handle.shard.shard_index,
                    "descriptor": handle.descriptor,
                    "filename": handle.shard.filename,
                    "size_bytes": handle.shard.size_bytes,
                    "sha256": handle.shard.sha256,
                }
                for handle in handles
            ],
        }
        _write_private_json(request_path, request)
        try:
            process = subprocess.Popen(
                [
                    str(interpreter),
                    "-I",
                    "-B",
                    str(worker_snapshot),
                    "--request",
                    str(request_path),
                    "--output-dir",
                    str(output),
                ],
                cwd=root,
                env=_worker_environment(root),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                pass_fds=tuple(handle.descriptor for handle in handles),
                start_new_session=True,
            )
            return_code = process.wait(timeout=_WORKER_TIMEOUT_SEC)
        except subprocess.TimeoutExpired as exc:
            raise VoxPopuliPreparationError("worker timed out") from exc
        except OSError as exc:
            raise VoxPopuliPreparationError("worker failed") from exc
        reap_attempted = True
        public_fixtures._terminate_worker_process_group(process)
        group_reaped = True
        if return_code != 0:
            raise VoxPopuliPreparationError("worker failed")
        if (
            hashlib.sha256(
                public_fixtures._read_stable_worker_file(
                    worker_snapshot, maximum=_MAX_WORKER_BYTES
                )
            ).hexdigest()
            != worker_sha256
            or hashlib.sha256(
                public_fixtures._read_stable_repo_worker(_WORKER_PATH)
            ).hexdigest()
            != worker_sha256
        ):
            raise VoxPopuliPreparationError("worker changed")
        yield _WorkerRun(
            output_dir=output,
            binding={
                "contract": "isolated-pyarrow-voxpopuli-worker-v2",
                "protocol_version": PROTOCOL_VERSION,
                "worker_sha256": worker_sha256,
                "pyarrow_version": PYARROW_VERSION,
                "isolated_python": True,
                "single_thread_environment": True,
                "source_transport": "inherited-verified-file-descriptors",
                "production_evidence": True,
            },
        )
    except public_fixtures.PublicFixtureError:
        raise VoxPopuliPreparationError("worker failed") from None
    finally:
        if process is not None and not group_reaped and not reap_attempted:
            reap_attempted = True
            public_fixtures._terminate_worker_process_group(process)
            group_reaped = True
        if process is None or group_reaped:
            public_fixtures._unlock_private_tree(root)
            shutil.rmtree(root, ignore_errors=True)


def _wav_data(payload: bytes) -> tuple[bytes, int]:
    """Independently validate one selected exact IEEE-float32 WAV."""

    if (
        not isinstance(payload, bytes)
        or len(payload) < 58
        or len(payload) > _MAX_AUDIO_BYTES
        or payload[:4] != b"RIFF"
        or payload[8:12] != b"WAVE"
        or struct.unpack_from("<I", payload, 4)[0] + 8 != len(payload)
    ):
        raise VoxPopuliPreparationError("audio decode failed")
    fmt_size = struct.unpack_from("<I", payload, 16)[0]
    fmt = struct.unpack_from("<HHIIHH", payload, 20)
    fact_size = struct.unpack_from("<I", payload, 42)[0]
    fact_samples = struct.unpack_from("<I", payload, 46)[0]
    data_size = struct.unpack_from("<I", payload, 54)[0]
    if (
        payload[12:16] != b"fmt "
        or fmt_size != 18
        or fmt != (3, 1, 16_000, 64_000, 4, 32)
        or payload[36:38] != b"\x00\x00"
        or payload[38:42] != b"fact"
        or fact_size != 4
        or payload[50:54] != b"data"
        or data_size <= 0
        or data_size % 4
        or 58 + data_size != len(payload)
        or fact_samples != data_size // 4
    ):
        raise VoxPopuliPreparationError("audio decode failed")
    return payload[58:], data_size // 4


def _wav_to_f32le(payload: bytes, *, expected_samples: int) -> bytes:
    pcm, samples = _wav_data(payload)
    if samples != expected_samples or samples * 4 > MAX_PCM_BYTES:
        raise VoxPopuliPreparationError("audio decode failed")
    try:
        values = struct.iter_unpack("<f", pcm)
        if any(not math.isfinite(value) for (value,) in values):
            raise VoxPopuliPreparationError("audio decode failed")
    except struct.error:
        raise VoxPopuliPreparationError("audio decode failed") from None
    # IEEE float WAV does not impose a normative [-1, 1] magnitude bound.
    # Preserve every finite binary32 bit-pattern; never clip or normalize it.
    return bytes(pcm)


def _safe_text(value: object, *, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise VoxPopuliPreparationError("worker result failed")
    try:
        if len(value.encode("utf-8", errors="strict")) > maximum:
            raise VoxPopuliPreparationError("worker result failed")
    except UnicodeError:
        raise VoxPopuliPreparationError("worker result failed") from None
    return value


def _case_id(index: int, speaker_id: str, audio_id: str) -> tuple[str, str]:
    identity = worker_contract._rank("identity", speaker_id, audio_id)
    return f"vp-en-ro-{index:02d}-{identity[:12]}", identity


def _selection_recipe() -> dict[str, object]:
    minimum_samples = min(item[1] for item in DURATION_BUCKETS)
    maximum_samples = max(item[2] for item in DURATION_BUCKETS)
    return {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "revision": SOURCE_REVISION,
        "config": CONFIG,
        "split": SPLIT,
        "selection_seed": SELECTION_SEED,
        "filters": {
            "accent": "en_ro",
            "is_gold_transcript": True,
            "language_class_id": 16,
            "require_normalized_reference": True,
            "minimum_duration_samples_inclusive": minimum_samples,
            "maximum_duration_samples_inclusive": maximum_samples,
        },
        "duration_buckets": [
            {
                "id": name,
                "minimum_samples_inclusive": lower,
                "maximum_samples": upper,
                "maximum_inclusive": inclusive,
                "quota": quota,
            }
            for name, lower, upper, inclusive, quota in DURATION_BUCKETS
        ],
        "distinct_selected_speakers": True,
        "utterances_per_selected_speaker": 1,
        "row_rank": (
            "sha256(seed||NUL||revision||NUL||row||NUL||speaker_id||NUL||audio_id)"
        ),
        "speaker_bucket_rank": (
            "sha256(seed||NUL||revision||NUL||speaker-bucket||NUL||bucket||NUL||speaker_id)"
        ),
        "assignment": "deterministic-bipartite-augmenting-path-v1",
        "literal_normalized_text_preserved": True,
        "scoring_normalizer": "core.wer.normalize-v1",
    }


def _validate_worker_output(run: _WorkerRun) -> _ValidatedWorker:
    output = run.output_dir
    if output.is_symlink() or not output.is_dir():
        raise VoxPopuliPreparationError("worker result failed")
    expected_files = {"result.json"} | {
        f"case-{index:02d}.wav" for index in range(SELECTION_COUNT)
    }
    try:
        entries = tuple(output.iterdir())
    except OSError:
        raise VoxPopuliPreparationError("worker result failed") from None
    names = [entry.name for entry in entries]
    if (
        set(names) != expected_files
        or len(names) != len(expected_files)
        or len({name.casefold() for name in names}) != len(names)
    ):
        raise VoxPopuliPreparationError("worker result failed")
    raw_result = _private_file_bytes(
        output / "result.json", maximum=_MAX_WORKER_RESULT_BYTES
    )
    result = _strict_json(raw_result)
    if not isinstance(result, dict) or set(result) != worker_contract._RESULT_FIELDS:
        raise VoxPopuliPreparationError("worker result failed")
    bucket_counts = {name: quota for name, *_middle, quota in DURATION_BUCKETS}
    eligible_set_sha256 = _sha256(result.get("eligible_set_sha256"))
    eligible_rows = result.get("eligible_rows")
    eligible_speakers = result.get("eligible_speakers")
    if (
        result.get("schema_version") != PROTOCOL_VERSION
        or result.get("pyarrow_version") != PYARROW_VERSION
        or result.get("source_revision") != SOURCE_REVISION
        or result.get("selection_seed") != SELECTION_SEED
        or result.get("schema_contract_sha256") != SCHEMA_CONTRACT_SHA256
        or result.get("source_total_rows") != SOURCE_TOTAL_ROWS
        or result.get("target_accent_speakers")
        != worker_contract.EXPECTED_EN_RO_SPEAKERS
        or isinstance(eligible_rows, bool)
        or not isinstance(eligible_rows, int)
        or not SELECTION_COUNT <= eligible_rows <= worker_contract._MAX_TARGET_ROWS
        or isinstance(eligible_speakers, bool)
        or not isinstance(eligible_speakers, int)
        or not SELECTION_COUNT
        <= eligible_speakers
        <= worker_contract.EXPECTED_EN_RO_SPEAKERS
        or result.get("bucket_counts") != bucket_counts
    ):
        raise VoxPopuliPreparationError("worker provenance failed")

    raw_sources = result.get("source_shards")
    if not isinstance(raw_sources, list) or len(raw_sources) != len(SOURCE_SHARDS):
        raise VoxPopuliPreparationError("worker provenance failed")
    source_rows: list[Mapping[str, object]] = []
    total_rows = 0
    for shard, row in zip(SOURCE_SHARDS, raw_sources, strict=True):
        if (
            not isinstance(row, dict)
            or set(row) != worker_contract._RESULT_SOURCE_FIELDS
        ):
            raise VoxPopuliPreparationError("worker provenance failed")
        rows = row.get("rows")
        groups = row.get("row_groups")
        if (
            row.get("shard_index") != shard.shard_index
            or row.get("filename") != shard.filename
            or row.get("size_bytes") != shard.size_bytes
            or row.get("sha256") != shard.sha256
            or isinstance(rows, bool)
            or not isinstance(rows, int)
            or rows <= 0
            or rows > SOURCE_TOTAL_ROWS
            or isinstance(groups, bool)
            or not isinstance(groups, int)
            or groups <= 0
            or groups > rows
        ):
            raise VoxPopuliPreparationError("worker provenance failed")
        total_rows += rows
        source_rows.append(dict(row))
    if total_rows != SOURCE_TOTAL_ROWS:
        raise VoxPopuliPreparationError("worker provenance failed")

    raw_cases = result.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) != SELECTION_COUNT:
        raise VoxPopuliPreparationError("worker result failed")
    corpus_cases: list[CorpusWriteCase] = []
    receipt_cases: list[Mapping[str, object]] = []
    seen_audio_ids: set[str] = set()
    seen_speakers: set[str] = set()
    selected_by_bucket: dict[str, list[tuple[str, str]]] = {
        name: [] for name, *_rest in DURATION_BUCKETS
    }
    total_pcm = 0
    for index, row in enumerate(raw_cases):
        if not isinstance(row, dict) or set(row) != worker_contract._RESULT_CASE_FIELDS:
            raise VoxPopuliPreparationError("worker result failed")
        audio_id = _safe_text(row.get("audio_id"), maximum=512)
        speaker_id = _safe_text(row.get("speaker_id"), maximum=512)
        gender = row.get("gender")
        transcript = _safe_text(row.get("transcription"), maximum=4096)
        transcript_sha256 = hashlib.sha256(transcript.encode("utf-8")).hexdigest()
        shard_index = row.get("shard_index")
        row_index = row.get("row_index")
        samples = row.get("duration_samples")
        bucket = row.get("duration_bucket")
        audio_size = row.get("audio_size_bytes")
        audio_filename = f"case-{index:02d}.wav"
        row_rank = worker_contract.row_rank(speaker_id, audio_id)
        speaker_rank = worker_contract.speaker_bucket_rank(str(bucket), speaker_id)
        if (
            row.get("selection_index") != index
            or audio_id in seen_audio_ids
            or speaker_id in seen_speakers
            or not isinstance(gender, str)
            or "\x00" in gender
            or len(gender.encode("utf-8")) > 128
            or not normalize(transcript)
            or row.get("transcription_sha256") != transcript_sha256
            or isinstance(shard_index, bool)
            or not isinstance(shard_index, int)
            or not 0 <= shard_index < len(source_rows)
            or isinstance(row_index, bool)
            or not isinstance(row_index, int)
            or not 0 <= row_index < int(source_rows[shard_index]["rows"])
            or isinstance(samples, bool)
            or not isinstance(samples, int)
            or worker_contract.duration_bucket(samples) != bucket
            or row.get("row_rank_sha256") != row_rank
            or row.get("speaker_bucket_rank_sha256") != speaker_rank
            or row.get("audio_file") != audio_filename
            or isinstance(audio_size, bool)
            or not isinstance(audio_size, int)
            or not 0 < audio_size <= _MAX_AUDIO_BYTES
        ):
            raise VoxPopuliPreparationError("worker case failed")
        seen_audio_ids.add(audio_id)
        seen_speakers.add(speaker_id)
        _sha256(row.get("audio_sha256"))
        payload = _private_file_bytes(output / audio_filename, maximum=_MAX_AUDIO_BYTES)
        if (
            len(payload) != audio_size
            or hashlib.sha256(payload).hexdigest() != row["audio_sha256"]
        ):
            raise VoxPopuliPreparationError("worker case failed")
        raw_f32le = _wav_to_f32le(payload, expected_samples=samples)
        total_pcm += len(raw_f32le)
        if total_pcm > MAX_CORPUS_BYTES:
            raise VoxPopuliPreparationError("decoded corpus budget failed")
        case_id, identity_sha256 = _case_id(index, speaker_id, audio_id)
        pcm_sha256 = hashlib.sha256(raw_f32le).hexdigest()
        corpus_cases.append(
            CorpusWriteCase(
                case_id=case_id,
                audio_bytes=raw_f32le,
                reference=transcript,
                tags=(
                    "public-voice",
                    "voxpopuli",
                    "en-ro-accent",
                    "gold",
                    str(bucket),
                ),
            )
        )
        receipt_cases.append(
            {
                "selection_index": index,
                "case_id": case_id,
                "identity_sha256": identity_sha256,
                "speaker_sha256": worker_contract._rank("speaker", speaker_id),
                "duration_samples": samples,
                "duration_bucket": bucket,
                "transcription_sha256": transcript_sha256,
                "row_rank_sha256": row_rank,
                "speaker_bucket_rank_sha256": speaker_rank,
                "source_wav_sha256": row["audio_sha256"],
                "source_wav_bytes": audio_size,
                "pcm_sha256": pcm_sha256,
                "pcm_bytes": len(raw_f32le),
            }
        )
        selected_by_bucket[str(bucket)].append((speaker_rank, row_rank))
    if len(seen_audio_ids) != SELECTION_COUNT or len(seen_speakers) != SELECTION_COUNT:
        raise VoxPopuliPreparationError("worker selection failed")
    for name, *_middle, quota in DURATION_BUCKETS:
        ranks = selected_by_bucket[name]
        if len(ranks) != quota or ranks != sorted(ranks):
            raise VoxPopuliPreparationError("worker selection failed")

    metadata_binding = {
        "schema_contract_sha256": SCHEMA_CONTRACT_SHA256,
        "source_total_rows": SOURCE_TOTAL_ROWS,
        "source_shards": source_rows,
        "target_accent_speakers": worker_contract.EXPECTED_EN_RO_SPEAKERS,
        "eligible_rows": eligible_rows,
        "eligible_speakers": eligible_speakers,
        "eligible_set_sha256": eligible_set_sha256,
    }
    return _ValidatedWorker(
        cases=tuple(corpus_cases),
        receipt_cases=tuple(receipt_cases),
        metadata_binding=metadata_binding,
        source_rows=tuple(source_rows),
        worker_summary={
            "protocol_version": PROTOCOL_VERSION,
            "pyarrow_version": PYARROW_VERSION,
            "schema_contract_sha256": SCHEMA_CONTRACT_SHA256,
            "eligible_set_sha256": eligible_set_sha256,
            "eligible_rows": eligible_rows,
            "eligible_speakers": eligible_speakers,
        },
    )


def _module_digest(path: Path) -> str:
    try:
        resolved = path.resolve(strict=True)
        return hash_regular_bounded(
            resolved,
            maximum_bytes=8 * 1024 * 1024,
            expected_bytes=resolved.stat().st_size,
        ).sha256
    except (OSError, RuntimeError, ValueError, BoundedReadError):
        raise VoxPopuliPreparationError("preparer provenance failed") from None


def _preparer_provenance() -> dict[str, object]:
    files = (
        "core/wer.py",
        "tools/public_voice_fixtures.py",
        "tools/prepare_voxpopuli_ro_accent.py",
        "tools/voxpopuli_ro_accent_parquet_worker.py",
        "tools/streaming_stt/bounded_io.py",
        "tools/streaming_stt/corpus.py",
        "tools/streaming_stt/corpus_writer.py",
        "tools/streaming_stt/protocol.py",
    )
    return {
        "contract": "voxpopuli-en-ro-private-preparer-v2",
        "files": {
            relative: _module_digest(_REPO_ROOT / relative) for relative in files
        },
        "pcm_decoder": "exact-ieee-float32-wav-finite-identity-16000-v2",
        "amplitude_policy": "finite-binary32-no-bound-no-clipping-no-normalization",
        "python_implementation": sys.implementation.name,
        "single_thread_worker_environment": True,
    }


def prepare_voxpopuli_ro_accent(
    *,
    source_shard_0: Path | str,
    source_shard_1: Path | str,
    parquet_python: Path | str | None,
    output_dir: Path | str,
    accepted_terms: frozenset[str],
    test_worker_injection: TestWorkerInjection | None = None,
) -> PreparedVoxPopuliCorpus:
    """Verify exact local shards and publish the deterministic private slice."""

    if not REQUIRED_TERMS <= accepted_terms:
        raise VoxPopuliPreparationError("terms acknowledgement failed")
    destination = _validated_output_path(output_dir)
    preparer_binding = _preparer_provenance()
    recipe = _selection_recipe()
    recipe_sha256 = hashlib.sha256(_canonical_json_bytes(recipe)).hexdigest()
    with _opened_sources((source_shard_0, source_shard_1)) as handles:
        with _run_worker(
            handles,
            output_parent=destination.parent,
            parquet_python=parquet_python,
            test_worker_injection=test_worker_injection,
        ) as run:
            validated = _validate_worker_output(run)
            _verify_sources(handles, hash_content=True)

            metadata_sha256 = hashlib.sha256(
                _canonical_json_bytes(validated.metadata_binding)
            ).hexdigest()
            receipt = {
                "schema_version": 1,
                "kind": "voxpopuli-en-ro-accent-preparation",
                "dataset": {
                    "dataset_id": DATASET_ID,
                    "upstream_dataset": "facebook/voxpopuli",
                    "revision": SOURCE_REVISION,
                    "config": CONFIG,
                    "split": SPLIT,
                    "source_total_rows": SOURCE_TOTAL_ROWS,
                    "dataset_card_url": DATASET_CARD_URL,
                    "upstream_repository_url": UPSTREAM_REPOSITORY_URL,
                    "paper_url": PAPER_URL,
                    "source_artifacts": [
                        {
                            "shard_index": shard.shard_index,
                            "filename": shard.filename,
                            "size_bytes": shard.size_bytes,
                            "sha256": shard.sha256,
                            "download_url": shard.download_url,
                        }
                        for shard in SOURCE_SHARDS
                    ],
                },
                "license": {
                    "dataset_license_id": "CC0-1.0",
                    "dataset_license_url": CC0_URL,
                    "raw_source": "European Parliament event recordings 2009-2020",
                    "raw_source_url": EP_MEDIA_URL,
                    "raw_source_legal_notice_url": EP_LEGAL_NOTICE_URL,
                },
                "accepted_terms": sorted(REQUIRED_TERMS),
                "usage_constraints": [
                    "evaluation_only",
                    "private_cache_only",
                    "no_speaker_reidentification",
                    "no_identity_inference",
                    "public_official_speech",
                ],
                "selection": {
                    "recipe": recipe,
                    "recipe_sha256": recipe_sha256,
                    "eligible_set_sha256": validated.worker_summary[
                        "eligible_set_sha256"
                    ],
                    "eligible_rows": validated.worker_summary["eligible_rows"],
                    "eligible_speakers": validated.worker_summary["eligible_speakers"],
                    "selected_cases": SELECTION_COUNT,
                    "selected_distinct_speakers": SELECTION_COUNT,
                    "bucket_counts": {
                        name: quota for name, *_middle, quota in DURATION_BUCKETS
                    },
                },
                "metadata": dict(validated.metadata_binding),
                "worker": dict(run.binding),
                "preparer": preparer_binding,
                "cases": list(validated.receipt_cases),
                "total_pcm_bytes": sum(
                    int(item["pcm_bytes"]) for item in validated.receipt_cases
                ),
                "privacy": {
                    "receipt_contains_transcripts": False,
                    "receipt_contains_speaker_ids": False,
                    "receipt_contains_audio_ids": False,
                    "receipt_contains_local_paths": False,
                    "cli_is_aggregate_only": True,
                },
                "evidence_scope": {
                    "pcm_preparation_complete": True,
                    "stt_model_run": False,
                    "timing_scope": "prepared-pcm-only",
                    "formal_spontaneous_accent_diagnostic": True,
                    "domestic_conversation": False,
                    "commands": False,
                    "capture": False,
                    "vad": False,
                    "aec": False,
                    "endpoint_ground_truth": False,
                    "agent_tools": False,
                    "live_latency": False,
                    "model_training_disjoint": False,
                },
            }
            receipt_payload = _canonical_json_bytes(receipt)
            if not receipt_payload or len(receipt_payload) > _MAX_RECEIPT_BYTES:
                raise VoxPopuliPreparationError("receipt contract failed")
            receipt_sha256 = hashlib.sha256(receipt_payload).hexdigest()
            provenance = CorpusProvenance(
                kind="public-voice-v1",
                suite="voxpopuli-en-ro-accent",
                manifest_sha256=recipe_sha256,
                metadata_sha256=metadata_sha256,
                source_set_sha256=receipt_sha256,
            )
            try:
                corpus = publish_private_corpus(
                    cases=validated.cases,
                    provenance=provenance,
                    output_dir=destination,
                    purpose=(
                        "VoxPopuli Romanian-accented English gold spontaneous P0 slice"
                    ),
                    sidecars={"preparation-receipt.json": receipt_payload},
                )
            except (CorpusWriterError, OSError):
                raise VoxPopuliPreparationError("publication failed") from None
            if corpus.provenance != provenance:
                raise VoxPopuliPreparationError("publication failed")
        _verify_sources(handles, hash_content=False)
    if _preparer_provenance() != preparer_binding:
        raise VoxPopuliPreparationError("preparer provenance changed")
    return PreparedVoxPopuliCorpus(
        corpus=corpus,
        receipt_sha256=receipt_sha256,
        selection_recipe_sha256=recipe_sha256,
        metadata_sha256=metadata_sha256,
    )


def _safe_result(result: PreparedVoxPopuliCorpus) -> dict[str, object]:
    corpus = result.corpus
    if corpus.provenance is None:
        raise VoxPopuliPreparationError("publication failed")
    return {
        "ok": True,
        "dataset_id": DATASET_ID,
        "revision": SOURCE_REVISION,
        "cases": len(corpus.cases),
        "audio_bytes": corpus.audio_bytes,
        "corpus_sha256": corpus.digest,
        "receipt_sha256": result.receipt_sha256,
        "selection_recipe_sha256": result.selection_recipe_sha256,
        "metadata_sha256": result.metadata_sha256,
        "provenance": corpus.provenance.as_dict(),
    }


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        raise VoxPopuliPreparationError("argument contract failed")


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        description=(
            "Prepare the exact local VoxPopuli Romanian-accented English "
            "24-speaker private streaming corpus; no download, network, model, "
            "or audio-device execution."
        )
    )
    parser.add_argument("--source-shard-0", type=Path, required=True)
    parser.add_argument("--source-shard-1", type=Path, required=True)
    parser.add_argument("--parquet-python", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--accept-term",
        action="append",
        default=[],
        help=(
            "repeat for CC0-1.0 and EUROPEAN-PARLIAMENT-LEGAL-NOTICE after "
            "reviewing the official provenance links"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        result = prepare_voxpopuli_ro_accent(
            source_shard_0=args.source_shard_0,
            source_shard_1=args.source_shard_1,
            parquet_python=args.parquet_python,
            output_dir=args.output_dir,
            accepted_terms=frozenset(args.accept_term),
        )
        payload: Mapping[str, object] = _safe_result(result)
        code = 0
    except Exception:  # noqa: BLE001 - paths, IDs, and transcripts stay private
        payload = _SAFE_ERROR
        code = 2
    print(
        json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
