"""Create a deterministic DEMAND-noise derivative of a public STT corpus.

The command performs no download and runs no recognizer.  It accepts one
already verified schema-v2 public corpus plus the three exact 16 kHz DEMAND
domestic archives, reads only channel 01, and publishes one private schema-v2
corpus containing a kitchen/living/washing variant of every source case.

Raw paths, transcripts, and audio never appear in the preparation receipt or
CLI result.  The clean corpus remains separate so clean and noisy evidence
cannot be accidentally pooled into one headline number.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import stat
import sys
from typing import Mapping, Sequence
import wave
import zipfile

import numpy as np

from tools.public_voice_eval_matrix import dataset_by_id
from tools.streaming_stt.bounded_io import (
    BoundedReadError,
    hash_regular_bounded,
    opened_directory_nofollow,
)
from tools.streaming_stt.corpus import (
    CorpusError,
    CorpusProvenance,
    LoadedCorpus,
    load_corpus,
    verify_corpus_snapshot,
)
from tools.streaming_stt.corpus_writer import (
    CorpusWriteCase,
    CorpusWriterError,
    publish_private_corpus,
)
from tools.streaming_stt.protocol import MAX_CORPUS_BYTES, MAX_PCM_BYTES


DATASET_ID = "demand_domestic_16k"
DATASET_UPSTREAM_ID = "zenodo:1227121:DKITCHEN+DLIVING+DWASHING"
DATASET_DOI = "10.5281/zenodo.1227121"
DATASET_CANONICAL_URL = "https://zenodo.org/records/1227121"
LICENSE_ID = "CC-BY-SA-3.0"
LICENSE_URL = "https://creativecommons.org/licenses/by-sa/3.0/legalcode"
REQUIRED_TERMS = frozenset({LICENSE_ID})
SAMPLE_RATE_HZ = 16_000
CHANNEL = 1
SELECTION_SEED = "speaker-demand-domestic-v1-2026-08-01"
SNR_LEVELS_DB = (20, 10, 0)
_REPO_ROOT = Path(__file__).resolve().parents[1]
_MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
_MAX_MEMBER_BYTES = 16 * 1024 * 1024
_MAX_ARCHIVE_MEMBERS = 17
_MAX_CODE_BYTES = 8 * 1024 * 1024
_MAX_RUNTIME_BINARY_BYTES = 128 * 1024 * 1024
_MAX_RUNTIME_TREE_ENTRIES = 10_000
_MAX_RUNTIME_TREE_BYTES = 1024 * 1024 * 1024
_MAX_RUNTIME_TREE_DEPTH = 32
_SAFE_ID_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ERROR = {
    "ok": False,
    "error": "demand_noise_corpus_prerequisites_unavailable",
}


class DemandPreparationError(RuntimeError):
    """A detail-free source, transform, or publication failure."""


@dataclass(frozen=True, slots=True)
class DemandArchive:
    artifact_id: str
    filename: str
    root: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class TestDemandInjection:
    """Explicit test-only source contract; never reachable from the CLI."""

    archives: tuple[DemandArchive, ...]
    fixture_id: str


@dataclass(frozen=True, slots=True)
class _NoiseTrack:
    source: DemandArchive
    archive_sha256: str
    member_sha256: str
    source_path: Path = field(repr=False)
    source_identity: tuple[int, ...] = field(repr=False)
    samples: np.ndarray = field(repr=False)


@dataclass(frozen=True, slots=True)
class PreparedDemandCorpus:
    corpus: LoadedCorpus
    receipt_sha256: str
    source_set_sha256: str
    production_sources: bool


ARCHIVES = (
    DemandArchive(
        "dkitchen",
        "DKITCHEN_16k.zip",
        "DKITCHEN",
        110_501_049,
        "9d68e46d2709847c59580770e2f8aa160607ebff7475788174aa1080332cc845",
    ),
    DemandArchive(
        "dliving",
        "DLIVING_16k.zip",
        "DLIVING",
        80_170_627,
        "2b1726fe06e41551ce2397f2aaf3e4fb692c912d81914b04708df0bbb5252338",
    ),
    DemandArchive(
        "dwashing",
        "DWASHING_16k.zip",
        "DWASHING",
        102_343_250,
        "f687c0806b0e9731b71a9c01c6134a21330efc5fa4aa7ee97cf4df276ad1dae1",
    ),
)
_PUBLISHED_MD5 = {
    "dkitchen": "7ffbf52d7f4699f96927846103dc8788",
    "dliving": "46741384d9e434a0bd8b3ec1830b6052",
    "dwashing": "7e5ee9437ce9409c5f9a779b6212a240",
}
_STDLIB_RUNTIME_MODULES = (
    "_hashlib",
    "_io",
    "_json",
    "_sre",
    "_stat",
    "_struct",
    "argparse",
    "array",
    "binascii",
    "collections",
    "collections.abc",
    "contextlib",
    "dataclasses",
    "enum",
    "hashlib",
    "io",
    "json",
    "json.decoder",
    "json.encoder",
    "json.scanner",
    "math",
    "os",
    "pathlib",
    "platform",
    "posix",
    "re",
    "re._compiler",
    "re._constants",
    "re._parser",
    "stat",
    "struct",
    "sys",
    "typing",
    "urllib.parse",
    "wave",
    "zipfile",
    "zlib",
)


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
        raise DemandPreparationError() from None


def _digest_json(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        stat.S_IFMT(value.st_mode),
        value.st_nlink,
        value.st_uid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _git_marker_present(directory_fd: int) -> bool:
    marker_fd = -1
    try:
        try:
            marker = os.stat(".git", dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            return False
        if stat.S_ISREG(marker.st_mode):
            return marker.st_size > 0
        if stat.S_ISLNK(marker.st_mode) or not stat.S_ISDIR(marker.st_mode):
            return True
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        marker_fd = os.open(".git", flags, dir_fd=directory_fd)
        opened = os.fstat(marker_fd)
        if _identity(opened) != _identity(marker):
            raise BoundedReadError()
        try:
            os.stat("HEAD", dir_fd=marker_fd, follow_symlinks=False)
        except FileNotFoundError:
            return False
        return True
    except BoundedReadError:
        raise
    except (OSError, OverflowError, ValueError):
        raise BoundedReadError() from None
    finally:
        if marker_fd >= 0:
            try:
                os.close(marker_fd)
            except OSError:
                pass


def _has_git_ancestor(directory: Path) -> bool:
    candidate = Path(os.path.abspath(directory))
    if not candidate.is_absolute():
        raise BoundedReadError()
    for ancestor in (candidate, *candidate.parents):
        with opened_directory_nofollow(ancestor) as (_stable, descriptor):
            if _git_marker_present(descriptor):
                return True
    return False


def _validated_output_path(path: Path | str) -> Path:
    supplied = Path(path).expanduser()
    if not supplied.is_absolute():
        raise DemandPreparationError()
    candidate = Path(os.path.abspath(supplied))
    try:
        parent = candidate.parent.resolve(strict=True)
        inside_git = _has_git_ancestor(parent)
    except (OSError, RuntimeError, ValueError, BoundedReadError):
        raise DemandPreparationError() from None
    if (
        candidate.name in {"", ".", ".."}
        or candidate == _REPO_ROOT
        or _REPO_ROOT in candidate.parents
        or inside_git
        or parent != candidate.parent
        or candidate.exists()
        or candidate.is_symlink()
    ):
        raise DemandPreparationError()
    return candidate


def _module_digest(path: Path, *, maximum_bytes: int) -> str:
    try:
        resolved = path.resolve(strict=True)
        metadata = resolved.stat()
        return hash_regular_bounded(
            resolved,
            maximum_bytes=maximum_bytes,
            expected_bytes=metadata.st_size,
        ).sha256
    except (OSError, RuntimeError, ValueError, BoundedReadError):
        raise DemandPreparationError() from None


def _runtime_tree_digest(root: Path) -> Mapping[str, object]:
    """Bind one package tree without retaining its local path."""

    root_descriptor = -1
    try:
        stable_root = root.resolve(strict=True)
        if stable_root != root or not stable_root.is_dir():
            raise DemandPreparationError()
        root_before = stable_root.lstat()
        directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        directory_flags |= getattr(os, "O_DIRECTORY", 0)
        directory_flags |= getattr(os, "O_NOFOLLOW", 0)
        file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        file_flags |= getattr(os, "O_NOFOLLOW", 0)
        root_descriptor = os.open(stable_root, directory_flags)
        root_opened = os.fstat(root_descriptor)
        if not stat.S_ISDIR(root_opened.st_mode) or _identity(
            root_opened
        ) != _identity(root_before):
            raise DemandPreparationError()

        rows: list[Mapping[str, object]] = []
        total_bytes = 0
        entries_seen = 0

        def walk(directory_fd: int, parts: tuple[str, ...], depth: int) -> None:
            nonlocal entries_seen, total_bytes
            if depth > _MAX_RUNTIME_TREE_DEPTH:
                raise DemandPreparationError()
            names: list[str] = []
            with os.scandir(directory_fd) as entries:
                for entry in entries:
                    entries_seen += 1
                    if entries_seen > _MAX_RUNTIME_TREE_ENTRIES:
                        raise DemandPreparationError()
                    names.append(entry.name)
            names.sort()
            for name in names:
                metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                relative_parts = (*parts, name)
                if stat.S_ISLNK(metadata.st_mode):
                    raise DemandPreparationError()
                if stat.S_ISDIR(metadata.st_mode):
                    child = os.open(name, directory_flags, dir_fd=directory_fd)
                    try:
                        opened = os.fstat(child)
                        if _identity(opened) != _identity(metadata):
                            raise DemandPreparationError()
                        walk(child, relative_parts, depth + 1)
                        after = os.fstat(child)
                        current = os.stat(
                            name,
                            dir_fd=directory_fd,
                            follow_symlinks=False,
                        )
                        if _identity(after) != _identity(opened) or _identity(
                            current
                        ) != _identity(opened):
                            raise DemandPreparationError()
                    finally:
                        os.close(child)
                    continue
                if not stat.S_ISREG(metadata.st_mode):
                    raise DemandPreparationError()

                relative = PurePosixPath(*relative_parts).as_posix()
                if (
                    metadata.st_size == 0
                    or name.endswith(".pyc")
                    or "__pycache__" in relative_parts
                ):
                    continue
                if (
                    metadata.st_size < 0
                    or metadata.st_size > _MAX_RUNTIME_BINARY_BYTES
                    or total_bytes + metadata.st_size > _MAX_RUNTIME_TREE_BYTES
                ):
                    raise DemandPreparationError()

                descriptor = os.open(name, file_flags, dir_fd=directory_fd)
                try:
                    opened = os.fstat(descriptor)
                    if not stat.S_ISREG(opened.st_mode) or _identity(
                        opened
                    ) != _identity(metadata):
                        raise DemandPreparationError()
                    digest = hashlib.sha256()
                    consumed = 0
                    while consumed <= metadata.st_size:
                        chunk = os.read(
                            descriptor,
                            min(1024 * 1024, metadata.st_size + 1 - consumed),
                        )
                        if not chunk:
                            break
                        consumed += len(chunk)
                        if consumed > metadata.st_size:
                            raise DemandPreparationError()
                        digest.update(chunk)
                    after = os.fstat(descriptor)
                    current = os.stat(
                        name,
                        dir_fd=directory_fd,
                        follow_symlinks=False,
                    )
                    if (
                        consumed != metadata.st_size
                        or _identity(after) != _identity(opened)
                        or _identity(current) != _identity(opened)
                    ):
                        raise DemandPreparationError()
                finally:
                    os.close(descriptor)
                total_bytes += consumed
                rows.append(
                    {
                        "relative_name": relative,
                        "size_bytes": consumed,
                        "sha256": digest.hexdigest(),
                    }
                )

        walk(root_descriptor, (), 0)
        root_after = os.fstat(root_descriptor)
        root_current = stable_root.lstat()
        if (
            _identity(root_after) != _identity(root_opened)
            or _identity(root_current) != _identity(root_opened)
            or stable_root.resolve(strict=True) != stable_root
            or not rows
        ):
            raise DemandPreparationError()
        return {
            "tree_sha256": hashlib.sha256(_canonical_json_bytes(rows)).hexdigest(),
            "files": len(rows),
            "bytes": total_bytes,
        }
    except DemandPreparationError:
        raise
    except (OSError, RuntimeError, ValueError, BoundedReadError):
        raise DemandPreparationError() from None
    finally:
        if root_descriptor >= 0:
            try:
                os.close(root_descriptor)
            except OSError:
                pass


def _stdlib_module_provenance() -> Mapping[str, object]:
    bindings: dict[str, object] = {}
    try:
        for module_name in _STDLIB_RUNTIME_MODULES:
            module = importlib.import_module(module_name)
            path_value = getattr(module, "__file__", None)
            if isinstance(path_value, str) and path_value:
                bindings[module_name] = {
                    "implementation": "file",
                    "sha256": _module_digest(
                        Path(path_value),
                        maximum_bytes=_MAX_RUNTIME_BINARY_BYTES,
                    ),
                }
                continue
            origin = getattr(getattr(module, "__spec__", None), "origin", None)
            if origin not in {"built-in", "frozen"}:
                raise DemandPreparationError()
            bindings[module_name] = {
                "implementation": str(origin),
                "bound_by": "python_executable_sha256",
            }
        return bindings
    except DemandPreparationError:
        raise
    except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
        raise DemandPreparationError() from None


def _preparer_provenance() -> Mapping[str, object]:
    files = (
        "core/wer.py",
        "tools/prepare_demand_noise_streaming_stt_corpus.py",
        "tools/public_voice_eval_matrix.py",
        "tools/streaming_stt/bounded_io.py",
        "tools/streaming_stt/corpus.py",
        "tools/streaming_stt/corpus_writer.py",
        "tools/streaming_stt/protocol.py",
    )
    return {
        "contract": "demand-domestic-private-noise-preparer-v3",
        "files": {
            relative: _module_digest(
                _REPO_ROOT / relative,
                maximum_bytes=_MAX_CODE_BYTES,
            )
            for relative in files
        },
    }


def _runtime_contract() -> Mapping[str, object]:
    try:
        numpy_init = Path(str(np.__file__)).resolve(strict=True)
        numpy_root = numpy_init.parent
        numpy_libraries = numpy_root.parent / "numpy.libs"
        try:
            numpy_core_name = "numpy._core._multiarray_umath"
            numpy_core_module = importlib.import_module(numpy_core_name)
        except ImportError:
            numpy_core_name = "numpy.core._multiarray_umath"
            numpy_core_module = importlib.import_module(numpy_core_name)
        numpy_core = Path(str(numpy_core_module.__file__)).resolve(strict=True)
        if numpy_root not in numpy_core.parents:
            raise DemandPreparationError()
        python_executable = Path(sys.executable).resolve(strict=True)
        cache_tag = sys.implementation.cache_tag
        if not isinstance(cache_tag, str) or not cache_tag:
            raise DemandPreparationError()
        return {
            "contract": "cpython-stdlib-numpy-pcm16-f32le-noise-mix-v2",
            "python": {
                "implementation": sys.implementation.name,
                "version": platform.python_version(),
                "hexversion": sys.hexversion,
                "cache_tag": cache_tag,
                "byteorder": sys.byteorder,
                "executable_sha256": _module_digest(
                    python_executable,
                    maximum_bytes=_MAX_RUNTIME_BINARY_BYTES,
                ),
                "stdlib_modules": _stdlib_module_provenance(),
            },
            "numpy": {
                "version": str(np.__version__),
                "multiarray_umath_module": numpy_core_name,
                "module_sha256": _module_digest(
                    numpy_init,
                    maximum_bytes=_MAX_RUNTIME_BINARY_BYTES,
                ),
                "multiarray_umath_sha256": _module_digest(
                    numpy_core,
                    maximum_bytes=_MAX_RUNTIME_BINARY_BYTES,
                ),
                "package": _runtime_tree_digest(numpy_root),
                "bundled_libraries": (
                    {"present": True, **_runtime_tree_digest(numpy_libraries)}
                    if numpy_libraries.is_dir()
                    else {"present": False}
                ),
                "input_dtype": np.dtype("<i2").str,
                "working_dtype": np.dtype("<f8").str,
                "output_dtype": np.dtype("<f4").str,
            },
            "platform": {
                "system": platform.system(),
                "machine": platform.machine(),
            },
            "numeric_policy": (
                "little-endian-pcm16-input,float64-rms-and-mix,"
                "little-endian-float32-output"
            ),
        }
    except DemandPreparationError:
        raise
    except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
        raise DemandPreparationError() from None


def _execution_bindings() -> Mapping[str, object]:
    return {
        "preparer": _preparer_provenance(),
        "runtime": _runtime_contract(),
    }


def _validate_archive_contract(source: DemandArchive) -> None:
    if (
        type(source) is not DemandArchive
        or _SAFE_ID_RE.fullmatch(source.artifact_id) is None
        or not source.filename.endswith("_16k.zip")
        or not source.root.isupper()
        or "/" in source.filename
        or "/" in source.root
        or type(source.size_bytes) is not int
        or not 0 < source.size_bytes <= _MAX_ARCHIVE_BYTES
        or _SHA256_RE.fullmatch(source.sha256) is None
    ):
        raise DemandPreparationError()


def _validate_catalog_contract() -> Mapping[str, object]:
    try:
        dataset = dataset_by_id(DATASET_ID)
        by_id = {artifact.artifact_id: artifact for artifact in dataset.artifacts}
        valid = (
            dataset.dataset_id == DATASET_ID
            and dataset.upstream_id == DATASET_UPSTREAM_ID
            and dataset.revision == DATASET_DOI
            and dataset.canonical_url == DATASET_CANONICAL_URL
            and dataset.license.license_id == LICENSE_ID
            and dataset.license.license_url == LICENSE_URL
            and dataset.license.acceptance_ids == tuple(sorted(REQUIRED_TERMS))
            and dataset.license.redistribution == "cache-only"
            and tuple(item.value for item in dataset.usage_constraints)
            == ("evaluation_only", "private_cache_only")
            and set(by_id) == {source.artifact_id for source in ARCHIVES}
        )
        for source in ARCHIVES:
            artifact = by_id[source.artifact_id]
            valid = valid and (
                artifact.locator
                == (
                    "https://zenodo.org/api/records/1227121/files/"
                    f"{source.filename}/content"
                )
                and artifact.filename == source.filename
                and artifact.size_bytes == source.size_bytes
                and artifact.integrity.value == "pinned"
                and artifact.checksum is not None
                and artifact.checksum.algorithm.value == "sha256"
                and artifact.checksum.digest == source.sha256
                and artifact.checksum.source
                == (
                    "local-freeze-after-upstream-md5-"
                    f"{_PUBLISHED_MD5[source.artifact_id]}"
                )
            )
    except (AttributeError, KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise DemandPreparationError()
    return {
        "dataset_id": DATASET_ID,
        "upstream_id": DATASET_UPSTREAM_ID,
        "doi": DATASET_DOI,
        "canonical_url": DATASET_CANONICAL_URL,
        "license_id": LICENSE_ID,
        "license_url": LICENSE_URL,
        "accepted_terms": sorted(REQUIRED_TERMS),
    }


def _source_contract(
    injection: TestDemandInjection | None,
) -> tuple[tuple[DemandArchive, ...], bool, str | None]:
    if injection is None:
        sources = ARCHIVES
        production = True
        fixture_sha256 = None
    elif (
        type(injection) is TestDemandInjection
        and isinstance(injection.fixture_id, str)
        and injection.fixture_id.startswith("test-")
        and len(injection.fixture_id) <= 128
    ):
        sources = injection.archives
        production = False
        fixture_sha256 = hashlib.sha256(injection.fixture_id.encode()).hexdigest()
    else:
        raise DemandPreparationError()
    if (
        len(sources) != 3
        or tuple(source.artifact_id for source in sources)
        != ("dkitchen", "dliving", "dwashing")
        or len({source.root for source in sources}) != 3
    ):
        raise DemandPreparationError()
    for source in sources:
        _validate_archive_contract(source)
    return sources, production, fixture_sha256


def _absolute_private_file(path: Path | str, source: DemandArchive) -> Path:
    supplied = Path(path).expanduser()
    if not supplied.is_absolute():
        raise DemandPreparationError()
    lexical = Path(os.path.abspath(supplied))
    try:
        resolved = lexical.resolve(strict=True)
        metadata = lexical.lstat()
    except (OSError, RuntimeError, ValueError):
        raise DemandPreparationError() from None
    if (
        resolved != lexical
        or lexical.name != source.filename
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_size != source.size_bytes
        or stat.S_IMODE(metadata.st_mode) & 0o077
        or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
    ):
        raise DemandPreparationError()
    return lexical


def _hash_archive_descriptor(descriptor: int, source: DemandArchive) -> str:
    try:
        digest = hashlib.sha256()
        consumed = 0
        while consumed <= source.size_bytes:
            chunk = os.pread(
                descriptor,
                min(1024 * 1024, source.size_bytes + 1 - consumed),
                consumed,
            )
            if not chunk:
                break
            consumed += len(chunk)
            if consumed > source.size_bytes:
                raise DemandPreparationError()
            digest.update(chunk)
        if consumed != source.size_bytes:
            raise DemandPreparationError()
        return digest.hexdigest()
    except DemandPreparationError:
        raise
    except (OSError, OverflowError, ValueError):
        raise DemandPreparationError() from None


def _read_member_exact(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
) -> bytes:
    if not 0 < info.file_size <= _MAX_MEMBER_BYTES:
        raise DemandPreparationError()
    try:
        with archive.open(info, "r") as handle:
            chunks: list[bytes] = []
            consumed = 0
            while consumed <= info.file_size:
                chunk = handle.read(min(64 * 1024, info.file_size + 1 - consumed))
                if not chunk:
                    break
                consumed += len(chunk)
                if consumed > info.file_size:
                    raise DemandPreparationError()
                chunks.append(chunk)
            if consumed != info.file_size or handle.read(1):
                raise DemandPreparationError()
            return b"".join(chunks)
    except DemandPreparationError:
        raise
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile):
        raise DemandPreparationError() from None


def _decode_pcm16_wav(payload: bytes) -> np.ndarray:
    if not isinstance(payload, bytes) or len(payload) > _MAX_MEMBER_BYTES:
        raise DemandPreparationError()
    try:
        with wave.open(io.BytesIO(payload), "rb") as reader:
            frames = reader.getnframes()
            if (
                reader.getnchannels() != 1
                or reader.getsampwidth() != 2
                or reader.getframerate() != SAMPLE_RATE_HZ
                or reader.getcomptype() != "NONE"
                or not SAMPLE_RATE_HZ <= frames <= 10 * 60 * SAMPLE_RATE_HZ
            ):
                raise DemandPreparationError()
            raw = reader.readframes(frames + 1)
            if len(raw) != frames * 2 or reader.readframes(1):
                raise DemandPreparationError()
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32)
        samples *= np.float32(1.0 / 32768.0)
        if samples.size != frames or not np.isfinite(samples).all():
            raise DemandPreparationError()
        return samples
    except DemandPreparationError:
        raise
    except (EOFError, OSError, TypeError, ValueError, wave.Error):
        raise DemandPreparationError() from None


def _load_noise(path: Path | str, source: DemandArchive) -> _NoiseTrack:
    candidate = _absolute_private_file(path, source)
    descriptor = -1
    try:
        before = candidate.lstat()
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(candidate, flags)
        opened = os.fstat(descriptor)
        if _identity(opened) != _identity(before):
            raise DemandPreparationError()
        archive_sha256 = _hash_archive_descriptor(descriptor, source)
        if archive_sha256 != source.sha256:
            raise DemandPreparationError()
        os.lseek(descriptor, 0, os.SEEK_SET)
        with os.fdopen(os.dup(descriptor), "rb") as file_handle:
            with zipfile.ZipFile(file_handle, mode="r", allowZip64=False) as archive:
                infos = archive.infolist()
                expected = {
                    f"{source.root}/",
                    *(f"{source.root}/ch{index:02d}.wav" for index in range(1, 17)),
                }
                names = [info.filename for info in infos]
                if (
                    len(infos) != _MAX_ARCHIVE_MEMBERS
                    or len(names) != len(set(names))
                    or len({name.casefold() for name in names}) != len(names)
                    or set(names) != expected
                    or any(
                        info.flag_bits & 1
                        or info.compress_type
                        not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                        or info.file_size < 0
                        or info.compress_size < 0
                        or info.file_size > _MAX_MEMBER_BYTES
                        for info in infos
                    )
                ):
                    raise DemandPreparationError()
                member = archive.getinfo(f"{source.root}/ch{CHANNEL:02d}.wav")
                payload = _read_member_exact(archive, member)
        after = os.fstat(descriptor)
        named_after = candidate.lstat()
        if _identity(after) != _identity(opened) or _identity(named_after) != _identity(
            opened
        ):
            raise DemandPreparationError()
        return _NoiseTrack(
            source=source,
            archive_sha256=archive_sha256,
            member_sha256=hashlib.sha256(payload).hexdigest(),
            source_path=candidate,
            source_identity=_identity(opened),
            samples=_decode_pcm16_wav(payload),
        )
    except DemandPreparationError:
        raise
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile):
        raise DemandPreparationError() from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _recheck_noise_sources(noises: Sequence[_NoiseTrack]) -> None:
    for noise in noises:
        candidate = _absolute_private_file(noise.source_path, noise.source)
        descriptor = -1
        try:
            before = candidate.lstat()
            if _identity(before) != noise.source_identity:
                raise DemandPreparationError()
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(candidate, flags)
            opened = os.fstat(descriptor)
            if _identity(opened) != noise.source_identity:
                raise DemandPreparationError()
            if _hash_archive_descriptor(descriptor, noise.source) != noise.archive_sha256:
                raise DemandPreparationError()
            after = os.fstat(descriptor)
            named_after = candidate.lstat()
            if (
                _identity(after) != noise.source_identity
                or _identity(named_after) != noise.source_identity
            ):
                raise DemandPreparationError()
        except DemandPreparationError:
            raise
        except (OSError, RuntimeError, ValueError):
            raise DemandPreparationError() from None
        finally:
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass


def _source_samples(case: object) -> np.ndarray:
    raw = getattr(case, "audio_bytes", None)
    samples = getattr(case, "samples", None)
    if (
        not isinstance(raw, bytes)
        or type(samples) is not int
        or samples <= 0
        or len(raw) != samples * 4
    ):
        raise DemandPreparationError()
    result = np.frombuffer(raw, dtype="<f4")
    if result.size != samples or not np.isfinite(result).all():
        raise DemandPreparationError()
    return result


def _active_rms(samples: np.ndarray) -> tuple[float, int]:
    values = samples.astype(np.float64, copy=False)
    peak = float(np.max(np.abs(values), initial=0.0))
    threshold = max(1e-5, peak * 0.02)
    active = values[np.abs(values) >= threshold]
    if active.size < 160:
        raise DemandPreparationError()
    rms = float(np.sqrt(np.mean(np.square(active))))
    if not math.isfinite(rms) or rms <= 0.0:
        raise DemandPreparationError()
    return rms, int(active.size)


def _case_offsets(corpus: LoadedCorpus, noise: _NoiseTrack) -> Mapping[int, int]:
    total = sum(case.samples for case in corpus.cases)
    if total <= 0 or total > int(noise.samples.size):
        raise DemandPreparationError()
    ranked = sorted(
        range(len(corpus.cases)),
        key=lambda index: hashlib.sha256(
            (
                SELECTION_SEED
                + "\0"
                + noise.source.artifact_id
                + "\0"
                + corpus.cases[index].sha256
            ).encode("ascii")
        ).digest(),
    )
    available = int(noise.samples.size) - total
    base_digest = hashlib.sha256(
        (
            SELECTION_SEED
            + "\0"
            + corpus.digest
            + "\0"
            + noise.member_sha256
        ).encode("ascii")
    ).digest()
    base = int.from_bytes(base_digest[:8], "big") % (available + 1)
    cursor = base
    offsets: dict[int, int] = {}
    for index in ranked:
        offsets[index] = cursor
        cursor += corpus.cases[index].samples
    return offsets


def _unique_tags(values: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


def _mix_case(
    signal: np.ndarray,
    noise: np.ndarray,
    *,
    snr_db: int,
) -> tuple[bytes, Mapping[str, object]]:
    if signal.size != noise.size or snr_db not in SNR_LEVELS_DB:
        raise DemandPreparationError()
    signal64 = signal.astype(np.float64, copy=False)
    noise64 = noise.astype(np.float64, copy=True)
    noise64 -= float(np.mean(noise64))
    speech_rms, active_samples = _active_rms(signal)
    noise_rms = float(np.sqrt(np.mean(np.square(noise64))))
    if not math.isfinite(noise_rms) or noise_rms <= 0.0:
        raise DemandPreparationError()
    target_noise_rms = speech_rms / (10.0 ** (snr_db / 20.0))
    gain = target_noise_rms / noise_rms
    component = noise64 * gain
    measured_noise_rms = float(np.sqrt(np.mean(np.square(component))))
    measured_snr_db = 20.0 * math.log10(speech_rms / measured_noise_rms)
    mixed = signal64 + component
    peak = float(np.max(np.abs(mixed), initial=0.0))
    final_scale = 1.0 if peak <= 0.99 else 0.99 / peak
    mixed *= final_scale
    output = mixed.astype("<f4", copy=False)
    if (
        not np.isfinite(output).all()
        or output.size != signal.size
        or not all(
            math.isfinite(value)
            for value in (gain, measured_snr_db, final_scale)
        )
    ):
        raise DemandPreparationError()
    return output.tobytes(order="C"), {
        "active_speech_samples": active_samples,
        "target_snr_db": snr_db,
        "measured_snr_db": round(measured_snr_db, 6),
        "noise_gain": round(gain, 12),
        "final_scale": round(final_scale, 12),
    }


def _source_set_sha256(corpus: LoadedCorpus, noises: Sequence[_NoiseTrack]) -> str:
    return _digest_json(
        {
            "source_corpus_sha256": corpus.digest,
            "demand_archives": [
                {
                    "artifact_id": item.source.artifact_id,
                    "archive_sha256": item.archive_sha256,
                    "member_sha256": item.member_sha256,
                }
                for item in noises
            ],
        }
    )


def prepare_demand_noise_corpus(
    *,
    source_corpus: Path | str,
    archive_paths: Sequence[Path | str],
    output_dir: Path | str,
    accepted_terms: frozenset[str],
    test_injection: TestDemandInjection | None = None,
) -> PreparedDemandCorpus:
    """Publish three deterministic domestic-noise variants per source case."""

    if not REQUIRED_TERMS <= accepted_terms:
        raise DemandPreparationError()
    dataset_binding = _validate_catalog_contract()
    sources, production_sources, fixture_sha256 = _source_contract(test_injection)
    if len(archive_paths) != len(sources):
        raise DemandPreparationError()
    destination = _validated_output_path(output_dir)
    execution_bindings = _execution_bindings()
    try:
        corpus = load_corpus(source_corpus)
    except (CorpusError, OSError, RuntimeError, ValueError):
        raise DemandPreparationError() from None
    if (
        corpus.schema_version != 2
        or corpus.provenance is None
        or corpus.provenance.kind != "public-voice-v1"
        or any(case.assertion != "transcript" for case in corpus.cases)
    ):
        raise DemandPreparationError()
    verify_corpus_snapshot(corpus)
    noises = tuple(
        _load_noise(path, source)
        for path, source in zip(archive_paths, sources, strict=True)
    )
    projected_bytes = corpus.audio_bytes * len(noises)
    if projected_bytes <= 0 or projected_bytes > MAX_CORPUS_BYTES:
        raise DemandPreparationError()

    offsets_by_environment = tuple(_case_offsets(corpus, noise) for noise in noises)
    rows: list[CorpusWriteCase] = []
    receipt_cases: list[Mapping[str, object]] = []
    output_hashes: set[str] = set()
    for case_index, case in enumerate(corpus.cases):
        signal = _source_samples(case)
        for environment_index, (noise, offsets) in enumerate(
            zip(noises, offsets_by_environment, strict=True)
        ):
            offset = offsets[case_index]
            end = offset + case.samples
            segment = noise.samples[offset:end]
            if segment.size != case.samples:
                raise DemandPreparationError()
            snr_db = SNR_LEVELS_DB[(case_index + environment_index) % len(SNR_LEVELS_DB)]
            audio_bytes, measurements = _mix_case(signal, segment, snr_db=snr_db)
            if len(audio_bytes) > MAX_PCM_BYTES:
                raise DemandPreparationError()
            output_sha256 = hashlib.sha256(audio_bytes).hexdigest()
            if output_sha256 in output_hashes:
                raise DemandPreparationError()
            output_hashes.add(output_sha256)
            snr_tag = f"snr{snr_db}"
            case_id = f"noise-{case_index:03d}-{noise.source.artifact_id}-{snr_tag}"
            rows.append(
                CorpusWriteCase(
                    case_id=case_id,
                    audio_bytes=audio_bytes,
                    reference=case.expected_text,
                    tags=_unique_tags(
                        (*case.tags, "noise", "demand", noise.source.artifact_id, snr_tag)
                    ),
                    commands=case.commands,
                )
            )
            receipt_cases.append(
                {
                    "source_audio_sha256": case.sha256,
                    "source_samples": case.samples,
                    "environment": noise.source.artifact_id,
                    "noise_member_sha256": noise.member_sha256,
                    "noise_sample_offset": offset,
                    "output_sha256": output_sha256,
                    **measurements,
                }
            )

    source_set_sha256 = _source_set_sha256(corpus, noises)
    receipt: dict[str, object] = {
        "schema_version": 1,
        "kind": "demand-domestic-noise-transform-v1",
        "production_sources": production_sources,
        "dataset": dataset_binding,
        "accepted_terms": sorted(REQUIRED_TERMS),
        "selection_seed": SELECTION_SEED,
        "source_corpus": {
            "schema_version": corpus.schema_version,
            "manifest_sha256": corpus.digest,
            "provenance": corpus.provenance.as_dict(),
            "cases": len(corpus.cases),
            "audio_bytes": corpus.audio_bytes,
        },
        "sources": [
            {
                "artifact_id": noise.source.artifact_id,
                "archive_bytes": noise.source.size_bytes,
                "archive_sha256": noise.archive_sha256,
                "channel": CHANNEL,
                "member_sha256": noise.member_sha256,
                "member_samples": int(noise.samples.size),
            }
            for noise in noises
        ],
        "transform": {
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "snr_levels_db": list(SNR_LEVELS_DB),
            "snr_assignment": "(source_case_index+environment_index)%3",
            "noise_dc_removed": True,
            "speech_rms": "samples_at_or_above_max(1e-5,0.02*peak)",
            "non_overlapping_intervals_per_environment": True,
            "peak_limit": 0.99,
        },
        "preparer": execution_bindings["preparer"],
        "runtime": execution_bindings["runtime"],
        "source_set_sha256": source_set_sha256,
        "cases": receipt_cases,
        "privacy": {
            "receipt_contains_transcripts": False,
            "receipt_contains_local_paths": False,
            "cli_is_aggregate_only": True,
        },
    }
    if fixture_sha256 is not None:
        receipt["test_fixture_sha256"] = fixture_sha256
    receipt_payload = _canonical_json_bytes(receipt)
    receipt_sha256 = hashlib.sha256(receipt_payload).hexdigest()
    provenance = CorpusProvenance(
        kind="public-voice-v1",
        suite="demand-domestic-v1",
        manifest_sha256=corpus.digest,
        metadata_sha256=receipt_sha256,
        source_set_sha256=source_set_sha256,
    )
    try:
        published = publish_private_corpus(
            cases=tuple(rows),
            provenance=provenance,
            output_dir=destination,
            purpose="deterministic DEMAND domestic-noise streaming STT corpus v1",
            sidecars={"preparation-receipt.json": receipt_payload},
        )
    except (CorpusWriterError, OSError):
        raise DemandPreparationError() from None
    try:
        verify_corpus_snapshot(corpus)
        _recheck_noise_sources(noises)
        if (
            _validate_catalog_contract() != dataset_binding
            or _execution_bindings() != execution_bindings
        ):
            raise DemandPreparationError()
    except DemandPreparationError:
        raise
    except (CorpusError, OSError, RuntimeError, ValueError):
        raise DemandPreparationError() from None
    return PreparedDemandCorpus(
        corpus=published,
        receipt_sha256=receipt_sha256,
        source_set_sha256=source_set_sha256,
        production_sources=production_sources,
    )


def _safe_result(result: PreparedDemandCorpus) -> Mapping[str, object]:
    corpus = result.corpus
    if corpus.schema_version != 2 or corpus.provenance is None:
        raise DemandPreparationError()
    environment_counts: dict[str, int] = {}
    snr_counts: dict[str, int] = {}
    for case in corpus.cases:
        for tag in case.tags:
            if tag in {source.artifact_id for source in ARCHIVES}:
                environment_counts[tag] = environment_counts.get(tag, 0) + 1
            if tag in {"snr0", "snr10", "snr20"}:
                snr_counts[tag] = snr_counts.get(tag, 0) + 1
    return {
        "ok": True,
        "production_sources": result.production_sources,
        "corpus_sha256": corpus.digest,
        "receipt_sha256": result.receipt_sha256,
        "source_set_sha256": result.source_set_sha256,
        "cases": len(corpus.cases),
        "audio_bytes": corpus.audio_bytes,
        "environment_counts": dict(sorted(environment_counts.items())),
        "snr_counts": dict(sorted(snr_counts.items())),
        "provenance": corpus.provenance.as_dict(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a private deterministic DEMAND kitchen/living/washing "
            "noise derivative of one verified public schema-v2 STT corpus."
        )
    )
    parser.add_argument("--source-corpus", type=Path, required=True)
    parser.add_argument("--dkitchen", type=Path, required=True)
    parser.add_argument("--dliving", type=Path, required=True)
    parser.add_argument("--dwashing", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--accept-term", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = prepare_demand_noise_corpus(
            source_corpus=args.source_corpus,
            archive_paths=(args.dkitchen, args.dliving, args.dwashing),
            output_dir=args.output_dir,
            accepted_terms=frozenset(args.accept_term),
        )
        payload: Mapping[str, object] = _safe_result(result)
        code = 0
    except Exception:  # noqa: BLE001 - source paths and transcript rows stay private
        payload = _SAFE_ERROR
        code = 2
    print(_canonical_json_bytes(payload).decode("ascii").rstrip("\n"))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
