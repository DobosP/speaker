#!/usr/bin/env python3
"""Causal aggregate evaluator for the exact admitted LiveKit English shard.

Exact-source diagnostic mode is deliberately two-stage. An isolated PyArrow 25 worker
materializes private ordinal PCM and sample-domain spans without loading text;
the project runtime then evaluates recorded causal prefixes with Speaker's
production turn detector admission seam and adaptive endpoint policy. The
published report is aggregate-only and never contains paths, rows, IDs, PCM,
transcripts, messages, words, or per-span scores.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import subprocess
import sys
from typing import Callable, Iterator, Mapping, Optional, Sequence

from tools import inventory_livekit_eot_english as inventory_source
from tools import livekit_eot_parquet_worker as inventory_worker
from tools import public_voice_fixtures as public_fixtures
from tools import setup_models
from tools.streaming_stt.source_bundle import (
    SourceBundleError,
    load_source_bundle,
    verify_source_bundle,
)
from tools.streaming_stt.supervisor import sanitized_worker_environment


REPORT_SCHEMA_VERSION = 1
REPORT_KIND = "livekit-eot-speaker-causal-endpoint-aggregate-v1"
MATERIALIZATION_SCHEMA_VERSION = 1
MATERIALIZATION_KIND = "livekit-eot-causal-pcm-v1"
DATASET = "livekit/eot-bench-data"
DATASET_CONFIG = "en"
DATASET_SPLIT = "validation"
LICENSE = "CC-BY-4.0"
PARTIAL_ASSUMPTION = "assumed_nonempty_current_epoch_at_every_scored_span"
NO_PARTIAL_PROFILE = "no_current_epoch_partial_acoustic_fallback"
LABEL_CONVENTION = inventory_worker.LABEL_CONVENTION
SAMPLE_RATE_HZ = 16_000
GRID_SAMPLES = 1_600
INVENTORY_REPORT_SHA256 = (
    "4c6c53d6324741006fa34da7bc6ecb02069ca68e2d5efffd166661b8b2f610e5"
)
INVENTORY_SHA256 = (
    "693ce2c21acbde3f7b3751557262c87353e06fc293688bc2e46022bcbb99eb48"
)
AUDIO_SET_SHA256 = (
    "b15ca98e781f7347fe0439d750d240ef1cb5a3ed3936753b273cc6339df1a973"
)
SILENCE_SET_SHA256 = (
    "535c246fd73a13fcd5b19e9ce3da7e2324f184c6d43afdfeea97ecec103b5800"
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CAUSAL_WORKER_PATH = _REPO_ROOT / "tools/livekit_causal_endpoint_worker.py"
_MODEL_WORKER_PATH = _REPO_ROOT / "tools/livekit_causal_endpoint_model_worker.py"
_INVENTORY_WORKER_PATH = _REPO_ROOT / "tools/livekit_eot_parquet_worker.py"
_ENDPOINTING_PATH = _REPO_ROOT / "core/endpointing.py"
_ACOUSTIC_PATH = _REPO_ROOT / "always_on_agent/acoustic.py"
_TEXT_PATH = _REPO_ROOT / "always_on_agent/text.py"
_EXECUTION_CLOSURE_RELATIVE_PATHS = (
    "tools/livekit_causal_endpoint_eval.py",
    "tools/livekit_causal_endpoint_worker.py",
    "tools/livekit_causal_endpoint_model_worker.py",
    "tools/livekit_eot_parquet_worker.py",
    "tools/inventory_livekit_eot_english.py",
    "tools/public_voice_fixtures.py",
    "tools/streaming_stt/bounded_io.py",
    "tools/setup_models.py",
    "core/endpointing.py",
    "core/engines/sherpa.py",
    "always_on_agent/acoustic.py",
    "always_on_agent/text.py",
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_VERSION_RE = re.compile(r"[0-9A-Za-z][0-9A-Za-z.+_-]{0,63}\Z")
_MAX_PRIVATE_JSON_BYTES = 2 * 1024 * 1024
_MAX_MATERIALIZER_REQUEST_BYTES = 128 * 1024
_MAX_REPORT_BYTES = 128 * 1024
_MAX_PCM_BYTES = 32 * 1024 * 1024
_MAX_EXECUTION_CLOSURE_MEMBER_BYTES = 1024 * 1024
_WORKER_TIMEOUT_SECONDS = 900.0
_MODEL_WORKER_TIMEOUT_SECONDS = 2_400.0
_EXPECTED_ROWS = 400
_EXPECTED_SPANS = 1_250
_EXPECTED_HOLDS = 850
_EXPECTED_PCM_SAMPLES = 80_654_406
_SCRATCH_REGISTRY_MINIMUM_SPARE_FDS = 512
_SMART_TURN_MODEL_FILENAME = setup_models.SMART_TURN_MODEL_FILENAME
_SMART_TURN_MODEL_REVISION = setup_models.SMART_TURN_MODEL_REVISION
_SMART_TURN_MODEL_BYTES = setup_models.SMART_TURN_MODEL_BYTES
_SMART_TURN_MODEL_SHA256 = setup_models.SMART_TURN_MODEL_SHA256
_SMART_TURN_MODEL_LICENSE = setup_models.SMART_TURN_MODEL_LICENSE
_SAFE_ERROR = {
    "ok": False,
    "error": "livekit_causal_endpoint_evaluation_prerequisites_unavailable",
}


class LiveKitCausalEndpointError(RuntimeError):
    """A detail-free source, model, materialization, or report failure."""


@dataclass(frozen=True, slots=True)
class CausalSpan:
    start_sample: int
    end_sample: int

    @property
    def duration_samples(self) -> int:
        return self.end_sample - self.start_sample


@dataclass(frozen=True, slots=True)
class MaterializedRow:
    ordinal: int
    pcm_filename: str = field(repr=False)
    pcm_bytes: int
    pcm_samples: int
    pcm_sha256: str
    silence_spans: tuple[CausalSpan, ...]


@dataclass(frozen=True, slots=True)
class Materialization:
    directory: Path = field(repr=False)
    directory_identity: tuple[int, ...]
    manifest_sha256: str
    manifest: Mapping[str, object] = field(repr=False)
    rows: tuple[MaterializedRow, ...] = field(repr=False)
    leaf_identities: Mapping[str, tuple[int, ...]] = field(
        default_factory=dict,
        repr=False,
    )
    retained_directory_descriptor: int = field(default=-1, repr=False)


@dataclass(frozen=True, slots=True)
class EndpointPolicyContract:
    endpoint_config: Mapping[str, object]
    config_sha256: str
    sample_rate_hz: int
    grid_samples: int
    acoustic_rule2_samples: int
    prosody_min_samples: int
    max_wait_samples: int
    rule3_samples: int
    runtime_default_detector: str


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    report_path: Path = field(repr=False)
    report_sha256: str
    hold_denominator: int
    eot_denominator: int
    evidence_class: str
    production_runtime_evidence: bool


@dataclass(frozen=True, slots=True)
class ModelStageResult:
    candidate: Mapping[str, object]
    acoustic_no_partial_fallback: Mapping[str, object]
    execution_receipt: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class MaterializerStageResult:
    output: Path = field(repr=False)
    execution_receipt: Mapping[str, object]


@dataclass(slots=True)
class ScratchRegistry:
    """Retained descriptors for run-owned entries eligible for cleanup."""

    entries: dict[tuple[str, ...], "ScratchEntry"] = field(default_factory=dict)

    def close(self) -> None:
        for entry in self.entries.values():
            try:
                os.close(entry.descriptor)
            except OSError:
                pass
        self.entries.clear()


@dataclass(frozen=True, slots=True)
class ScratchEntry:
    descriptor: int = field(repr=False)
    is_directory: bool


@dataclass(frozen=True, slots=True)
class VenvLaunchSnapshot:
    argv0: Path = field(repr=False)
    root: Path = field(repr=False)
    root_snapshot: tuple[tuple[int, ...], Path, tuple[int, ...]] = field(repr=False)
    executable_snapshot: tuple[tuple[int, ...], Path, tuple[int, ...]] = field(
        repr=False
    )
    marker_path: Path = field(repr=False)
    marker_payload: bytes = field(repr=False)
    marker_identity: tuple[int, ...] = field(repr=False)


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
        raise LiveKitCausalEndpointError("report contract failed") from None


def _strict_json(raw: bytes) -> object:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise ValueError("duplicate")
            result[key] = value
        return result

    try:
        return json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except (UnicodeError, ValueError, OverflowError):
        raise LiveKitCausalEndpointError("private JSON contract failed") from None


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_uid,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _directory_entry_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        stat.S_IFMT(info.st_mode),
        stat.S_IMODE(info.st_mode),
        info.st_uid,
    )


def _inode_identity(info: os.stat_result) -> tuple[int, ...]:
    """Identity suitable for safe cleanup across writes and chmod."""

    return (
        info.st_dev,
        info.st_ino,
        stat.S_IFMT(info.st_mode),
        info.st_uid,
    )


def _read_descriptor(descriptor: int, expected_size: int) -> bytes:
    chunks = bytearray()
    try:
        while len(chunks) <= expected_size:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, expected_size + 1 - len(chunks)),
            )
            if not chunk:
                break
            chunks.extend(chunk)
    except OSError:
        raise LiveKitCausalEndpointError("private input changed") from None
    if len(chunks) != expected_size:
        raise LiveKitCausalEndpointError("private input changed")
    return bytes(chunks)


def _pread_descriptor(descriptor: int, expected_size: int) -> bytes:
    chunks = bytearray()
    try:
        while len(chunks) < expected_size:
            chunk = os.pread(
                descriptor,
                min(1024 * 1024, expected_size - len(chunks)),
                len(chunks),
            )
            if not chunk:
                break
            chunks.extend(chunk)
    except OSError:
        raise LiveKitCausalEndpointError("private input changed") from None
    if len(chunks) != expected_size:
        raise LiveKitCausalEndpointError("private input changed")
    return bytes(chunks)


def _open_verified_regular(
    path: Path,
    *,
    maximum: int,
    expected_sha256: str | None = None,
    private: bool,
) -> tuple[int, tuple[int, ...], str]:
    descriptor = -1
    try:
        before = path.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > maximum
            or (private and stat.S_IMODE(before.st_mode) & 0o077)
            or (private and hasattr(os, "geteuid") and before.st_uid != os.geteuid())
        ):
            raise LiveKitCausalEndpointError("private input contract failed")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if _identity(opened) != _identity(before):
            raise LiveKitCausalEndpointError("private input changed")
        digest = hashlib.sha256(
            _pread_descriptor(descriptor, opened.st_size)
        ).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise LiveKitCausalEndpointError("private input changed")
        if (
            _identity(os.fstat(descriptor)) != _identity(opened)
            or _identity(path.lstat()) != _identity(opened)
        ):
            raise LiveKitCausalEndpointError("private input changed")
        return descriptor, _identity(opened), digest
    except LiveKitCausalEndpointError:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    except OSError:
        if descriptor >= 0:
            os.close(descriptor)
        raise LiveKitCausalEndpointError("private input contract failed") from None


def _rebind_verified_regular(
    descriptor: int,
    path: Path,
    identity: tuple[int, ...],
    digest: str,
) -> None:
    try:
        current = os.fstat(descriptor)
        if (
            _identity(current) != identity
            or _identity(path.lstat()) != identity
            or hashlib.sha256(
                _pread_descriptor(descriptor, current.st_size)
            ).hexdigest()
            != digest
        ):
            raise LiveKitCausalEndpointError("private input changed")
    except LiveKitCausalEndpointError:
        raise
    except OSError:
        raise LiveKitCausalEndpointError("private input changed") from None


def _open_verified_private_directory(path: Path) -> tuple[int, tuple[int, ...]]:
    descriptor = -1
    try:
        before = path.lstat()
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if (
            _identity(opened) != _identity(before)
            or not stat.S_ISDIR(opened.st_mode)
            or stat.S_IMODE(opened.st_mode) != 0o700
            or (hasattr(os, "geteuid") and opened.st_uid != os.geteuid())
        ):
            raise LiveKitCausalEndpointError("private directory changed")
        return descriptor, _directory_entry_identity(opened)
    except LiveKitCausalEndpointError:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    except OSError:
        if descriptor >= 0:
            os.close(descriptor)
        raise LiveKitCausalEndpointError("private directory changed") from None


def _rebind_verified_directory(
    descriptor: int,
    path: Path,
    identity: tuple[int, ...],
) -> None:
    try:
        if (
            _directory_entry_identity(os.fstat(descriptor)) != identity
            or _directory_entry_identity(path.lstat()) != identity
        ):
            raise LiveKitCausalEndpointError("private directory changed")
    except LiveKitCausalEndpointError:
        raise
    except OSError:
        raise LiveKitCausalEndpointError("private directory changed") from None


def _read_stable_regular(
    path: Path,
    *,
    maximum: int,
    private: bool,
) -> tuple[bytes, tuple[int, ...]]:
    descriptor = -1
    try:
        before = path.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > maximum
            or (private and stat.S_IMODE(before.st_mode) & 0o077)
            or (private and hasattr(os, "geteuid") and before.st_uid != os.geteuid())
        ):
            raise LiveKitCausalEndpointError("private input contract failed")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if _identity(opened) != _identity(before):
            raise LiveKitCausalEndpointError("private input changed")
        payload = _read_descriptor(descriptor, opened.st_size)
        after_fd = os.fstat(descriptor)
        after_path = path.lstat()
        if _identity(after_fd) != _identity(opened) or _identity(after_path) != _identity(opened):
            raise LiveKitCausalEndpointError("private input changed")
        return payload, _identity(opened)
    except LiveKitCausalEndpointError:
        raise
    except OSError:
        raise LiveKitCausalEndpointError("private input contract failed") from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _read_stable_execution_closure_member(path: Path) -> bytes:
    """Read one allowlisted repo member through a retained, rebound descriptor."""

    try:
        if not isinstance(path, Path) or not path.is_absolute():
            raise LiveKitCausalEndpointError("execution closure failed")
        lexical = Path(os.path.abspath(path))
        relative = lexical.relative_to(_REPO_ROOT).as_posix()
        if (
            lexical != path
            or lexical != _REPO_ROOT / relative
            or relative not in _EXECUTION_CLOSURE_RELATIVE_PATHS
            or lexical.resolve(strict=True) != lexical
        ):
            raise LiveKitCausalEndpointError("execution closure failed")
        payload, _identity_snapshot = _read_stable_regular(
            lexical,
            maximum=_MAX_EXECUTION_CLOSURE_MEMBER_BYTES,
            private=False,
        )
        if not payload:
            raise LiveKitCausalEndpointError("execution closure failed")
        return payload
    except LiveKitCausalEndpointError:
        raise LiveKitCausalEndpointError("execution closure failed") from None
    except (OSError, ValueError):
        raise LiveKitCausalEndpointError("execution closure failed") from None


def _snapshot_execution_closure() -> dict[str, str]:
    closure: dict[str, str] = {}
    try:
        for relative_path in _EXECUTION_CLOSURE_RELATIVE_PATHS:
            source = _REPO_ROOT / relative_path
            if source.resolve(strict=True) != source:
                raise LiveKitCausalEndpointError("execution closure failed")
            payload = _read_stable_execution_closure_member(source)
            closure[relative_path] = hashlib.sha256(payload).hexdigest()
    except LiveKitCausalEndpointError:
        raise
    except Exception:
        raise LiveKitCausalEndpointError("execution closure failed") from None
    if tuple(closure) != _EXECUTION_CLOSURE_RELATIVE_PATHS or any(
        _SHA256_RE.fullmatch(value) is None for value in closure.values()
    ):
        raise LiveKitCausalEndpointError("execution closure failed")
    return closure


def _rebind_execution_closure(expected: Mapping[str, str]) -> None:
    if not isinstance(expected, dict) or tuple(expected) != _EXECUTION_CLOSURE_RELATIVE_PATHS:
        raise LiveKitCausalEndpointError("execution closure changed")
    if _snapshot_execution_closure() != expected:
        raise LiveKitCausalEndpointError("execution closure changed")


def _create_private_scratch(
    path: Path | str,
) -> tuple[Path, int, tuple[int, ...], ScratchRegistry]:
    supplied = Path(path).expanduser()
    if not supplied.is_absolute():
        raise LiveKitCausalEndpointError("scratch prerequisite failed")
    lexical = Path(os.path.abspath(supplied))
    if not lexical.name or lexical.name in {".", "..", ".git"}:
        raise LiveKitCausalEndpointError("scratch prerequisite failed")
    parent_descriptor = -1
    scratch_descriptor = -1
    created = False
    succeeded = False
    try:
        parent = lexical.parent.resolve(strict=True)
        parent_info = parent.lstat()
        if (
            parent != lexical.parent
            or not inventory_source._outside_repo(parent)
            or inventory_source._has_git_ancestor(parent)
            or not stat.S_ISDIR(parent_info.st_mode)
            or stat.S_IMODE(parent_info.st_mode) & 0o077
            or (hasattr(os, "geteuid") and parent_info.st_uid != os.geteuid())
        ):
            raise LiveKitCausalEndpointError("scratch prerequisite failed")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        parent_descriptor = os.open(parent, flags)
        if _directory_entry_identity(os.fstat(parent_descriptor)) != _directory_entry_identity(parent_info):
            raise LiveKitCausalEndpointError("scratch prerequisite failed")
        os.mkdir(lexical.name, 0o700, dir_fd=parent_descriptor)
        created = True
        scratch_descriptor = os.open(
            lexical.name,
            flags,
            dir_fd=parent_descriptor,
        )
        current = lexical.lstat()
        opened = os.fstat(scratch_descriptor)
        if (
            _directory_entry_identity(opened) != _directory_entry_identity(current)
            or
            not stat.S_ISDIR(current.st_mode)
            or stat.S_IMODE(current.st_mode) != 0o700
            or (hasattr(os, "geteuid") and current.st_uid != os.geteuid())
            or lexical.resolve(strict=True) != lexical
        ):
            raise LiveKitCausalEndpointError("scratch prerequisite failed")
        identity = _directory_entry_identity(opened)
        os.fsync(parent_descriptor)
        succeeded = True
        return lexical, scratch_descriptor, identity, ScratchRegistry()
    except LiveKitCausalEndpointError:
        raise
    except (OSError, RuntimeError, ValueError, inventory_source.LiveKitEotInventoryError):
        raise LiveKitCausalEndpointError("scratch prerequisite failed") from None
    finally:
        if created and not succeeded:
            try:
                if scratch_descriptor < 0 or os.listdir(scratch_descriptor):
                    raise OSError
                current = os.stat(
                    lexical.name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
                if (
                    _directory_entry_identity(current)
                    != _directory_entry_identity(os.fstat(scratch_descriptor))
                ):
                    raise OSError
                os.rmdir(lexical.name, dir_fd=parent_descriptor)
            except OSError:
                pass
            if scratch_descriptor >= 0:
                os.close(scratch_descriptor)
        if parent_descriptor >= 0:
            os.close(parent_descriptor)


def _require_scratch_registry_capacity() -> None:
    """Fail before creation unless all run-owned leaves can retain descriptors."""

    try:
        import resource

        soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft == resource.RLIM_INFINITY:
            return
        opened = len(os.listdir("/proc/self/fd"))
        if soft - opened < _SCRATCH_REGISTRY_MINIMUM_SPARE_FDS:
            raise LiveKitCausalEndpointError("scratch descriptor capacity failed")
    except LiveKitCausalEndpointError:
        raise
    except (AttributeError, OSError, ValueError):
        raise LiveKitCausalEndpointError("scratch descriptor capacity failed") from None


def _assert_private_scratch_boundary(
    path: Path,
    descriptor: int,
    identity: tuple[int, ...],
) -> None:
    _rebind_verified_directory(descriptor, path, identity)
    try:
        if (
            path.resolve(strict=True) != path
            or not inventory_source._outside_repo(path)
            or inventory_source._has_git_ancestor(path)
        ):
            raise LiveKitCausalEndpointError("scratch boundary changed")
    except LiveKitCausalEndpointError:
        raise
    except (OSError, RuntimeError, ValueError, inventory_source.LiveKitEotInventoryError):
        raise LiveKitCausalEndpointError("scratch boundary changed") from None


def _validate_scratch_relative_path(relative_path: tuple[str, ...]) -> None:
    if not relative_path or any(
        not item or item in {".", ".."} or "/" in item or "\\" in item
        for item in relative_path
    ):
        raise LiveKitCausalEndpointError("scratch registry failed")


def _scratch_entry_matches(
    entry: ScratchEntry,
    info: os.stat_result,
) -> bool:
    try:
        opened = os.fstat(entry.descriptor)
    except OSError:
        return False
    if entry.is_directory:
        return stat.S_ISDIR(info.st_mode) and (
            _directory_entry_identity(opened) == _directory_entry_identity(info)
        )
    return stat.S_ISREG(info.st_mode) and _identity(opened) == _identity(info)


def _register_open_scratch_descriptor(
    registry: ScratchRegistry,
    relative_path: tuple[str, ...],
    descriptor: int,
    *,
    is_directory: bool,
    path_info: os.stat_result,
) -> None:
    _validate_scratch_relative_path(relative_path)
    candidate = ScratchEntry(descriptor=descriptor, is_directory=is_directory)
    if not _scratch_entry_matches(candidate, path_info):
        raise LiveKitCausalEndpointError("scratch registry changed")
    previous = registry.entries.get(relative_path)
    if previous is not None:
        try:
            if (
                previous.is_directory != is_directory
                or not _scratch_entry_matches(previous, path_info)
            ):
                raise LiveKitCausalEndpointError("scratch registry changed")
        finally:
            os.close(descriptor)
        return
    registry.entries[relative_path] = candidate


def _register_scratch_descriptor_copy(
    registry: ScratchRegistry,
    relative_path: tuple[str, ...],
    descriptor: int,
    *,
    is_directory: bool,
    parent_descriptor: int,
    name: str,
) -> None:
    copied = os.dup(descriptor)
    try:
        info = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        _register_open_scratch_descriptor(
            registry,
            relative_path,
            copied,
            is_directory=is_directory,
            path_info=info,
        )
        copied = -1
    finally:
        if copied >= 0:
            os.close(copied)


def _register_scratch_tree(
    parent_descriptor: int,
    name: str,
    registry: ScratchRegistry,
    *,
    prefix: tuple[str, ...] = (),
) -> None:
    """Register a completed private tree by no-follow descriptor traversal."""

    relative = (*prefix, name)
    try:
        info = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        if not _scratch_entry_allowed(relative, info):
            raise LiveKitCausalEndpointError("scratch registry failed")
        if stat.S_ISDIR(info.st_mode):
            if (
                stat.S_IMODE(info.st_mode) != 0o700
                or (hasattr(os, "geteuid") and info.st_uid != os.geteuid())
            ):
                raise LiveKitCausalEndpointError("scratch registry failed")
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(name, flags, dir_fd=parent_descriptor)
            try:
                _register_open_scratch_descriptor(
                    registry,
                    relative,
                    descriptor,
                    is_directory=True,
                    path_info=info,
                )
                descriptor = -1
                retained = registry.entries[relative].descriptor
                for child in os.listdir(retained):
                    _register_scratch_tree(
                        retained,
                        child,
                        registry,
                        prefix=relative,
                    )
                current = os.stat(
                    name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
                if not _scratch_entry_matches(registry.entries[relative], current):
                    raise LiveKitCausalEndpointError("scratch registry changed")
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
            return
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) & 0o077
            or (hasattr(os, "geteuid") and info.st_uid != os.geteuid())
        ):
            raise LiveKitCausalEndpointError("scratch registry failed")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
        try:
            _register_open_scratch_descriptor(
                registry,
                relative,
                descriptor,
                is_directory=False,
                path_info=info,
            )
            descriptor = -1
        finally:
            if descriptor >= 0:
                os.close(descriptor)
    except LiveKitCausalEndpointError:
        raise
    except OSError:
        raise LiveKitCausalEndpointError("scratch registry failed") from None


def _make_private_subdirectory(
    parent: Path,
    name: str,
    *,
    parent_descriptor: int | None = None,
    registry: ScratchRegistry | None = None,
    registry_prefix: tuple[str, ...] = (),
) -> Path:
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise LiveKitCausalEndpointError("scratch contract failed")
    target = parent / name
    opened_parent = -1
    directory_descriptor = -1
    created = False
    created_identity: tuple[int, ...] | None = None
    try:
        if parent_descriptor is None:
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_DIRECTORY", 0) | getattr(
                os,
                "O_NOFOLLOW",
                0,
            )
            opened_parent = os.open(parent, flags)
            parent_descriptor = opened_parent
        os.mkdir(name, 0o700, dir_fd=parent_descriptor)
        created = True
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        # Capture the new directory before any path-based inspection.  Every
        # later operation either uses this descriptor or proves the path still
        # names this exact inode.
        directory_descriptor = os.open(name, flags, dir_fd=parent_descriptor)
        opened = os.fstat(directory_descriptor)
        created_identity = _identity(opened)
        current = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            _identity(opened) != created_identity
            or _identity(current) != created_identity
            or not stat.S_ISDIR(opened.st_mode)
            or stat.S_IMODE(opened.st_mode) != 0o700
            or os.listdir(directory_descriptor)
            or target.resolve(strict=True) != target
            or (hasattr(os, "geteuid") and opened.st_uid != os.geteuid())
        ):
            raise LiveKitCausalEndpointError("scratch contract failed")
        current = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if _directory_entry_identity(opened) != _directory_entry_identity(current):
            raise LiveKitCausalEndpointError("scratch contract failed")
        if registry is not None:
            _register_open_scratch_descriptor(
                registry,
                (*registry_prefix, name),
                directory_descriptor,
                is_directory=True,
                path_info=current,
            )
            directory_descriptor = -1
        os.fsync(parent_descriptor)
        return target
    except LiveKitCausalEndpointError:
        raise
    except (OSError, RuntimeError, ValueError):
        raise LiveKitCausalEndpointError("scratch contract failed") from None
    finally:
        if directory_descriptor >= 0:
            if created:
                try:
                    opened = os.fstat(directory_descriptor)
                    current = os.stat(
                        name,
                        dir_fd=parent_descriptor,
                        follow_symlinks=False,
                    )
                    if (
                        created_identity is not None
                        and _identity(opened) == created_identity
                        and _identity(current) == created_identity
                        and not os.listdir(directory_descriptor)
                    ):
                        os.rmdir(name, dir_fd=parent_descriptor)
                        if os.fstat(directory_descriptor).st_nlink != 0:
                            raise OSError
                except OSError:
                    pass
            os.close(directory_descriptor)
        if opened_parent >= 0:
            os.close(opened_parent)


def _create_filled_private_file_at(
    directory_descriptor: int,
    name: str,
    payload: bytes,
    *,
    maximum: int,
    mode: int = 0o600,
) -> tuple[int, tuple[int, ...]]:
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise LiveKitCausalEndpointError("private output contract failed")
    if not payload or len(payload) > maximum or mode not in {0o400, 0o600}:
        raise LiveKitCausalEndpointError("private output contract failed")
    descriptor = -1
    created = False
    created_identity: tuple[int, ...] | None = None
    try:
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(name, flags, 0o600, dir_fd=directory_descriptor)
        created = True
        opened = os.fstat(descriptor)
        created_identity = _inode_identity(opened)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_size != 0
            or stat.S_IMODE(opened.st_mode) != 0o600
            or (hasattr(os, "geteuid") and opened.st_uid != os.geteuid())
        ):
            raise LiveKitCausalEndpointError("private output contract failed")
        offset = 0
        while offset < len(payload):
            written = os.pwrite(descriptor, payload[offset:], offset)
            if written <= 0:
                raise LiveKitCausalEndpointError("private output contract failed")
            offset += written
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
        after_fd = os.fstat(descriptor)
        current = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if (
            _identity(after_fd) != _identity(current)
            or _inode_identity(after_fd) != created_identity
            or after_fd.st_nlink != 1
            or stat.S_IMODE(after_fd.st_mode) != mode
            or after_fd.st_size != len(payload)
            or _pread_descriptor(descriptor, len(payload)) != payload
        ):
            raise LiveKitCausalEndpointError("private output contract failed")
        return descriptor, created_identity
    except LiveKitCausalEndpointError:
        _unlink_private_leaf_if_bound(
            directory_descriptor,
            name,
            descriptor=descriptor,
            created=created,
            created_identity=created_identity,
        )
        if descriptor >= 0:
            os.close(descriptor)
            descriptor = -1
        raise
    except OSError:
        _unlink_private_leaf_if_bound(
            directory_descriptor,
            name,
            descriptor=descriptor,
            created=created,
            created_identity=created_identity,
        )
        if descriptor >= 0:
            os.close(descriptor)
            descriptor = -1
        raise LiveKitCausalEndpointError("private output contract failed") from None


def _write_new_private_file_at(
    directory_descriptor: int,
    name: str,
    payload: bytes,
    *,
    maximum: int,
    mode: int = 0o600,
) -> tuple[int, ...]:
    descriptor, identity = _create_filled_private_file_at(
        directory_descriptor,
        name,
        payload,
        maximum=maximum,
        mode=mode,
    )
    try:
        return identity
    finally:
        os.close(descriptor)


def _unlink_private_leaf_if_bound(
    directory_descriptor: int,
    name: str,
    *,
    descriptor: int,
    created: bool,
    created_identity: tuple[int, ...] | None,
) -> None:
    """Remove only this call's created inode; preserve any raced replacement."""

    if not created or created_identity is None or descriptor < 0:
        return
    try:
        opened = os.fstat(descriptor)
        current = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        if (
            _inode_identity(opened) == created_identity
            and _identity(current) == _identity(opened)
        ):
            os.unlink(name, dir_fd=directory_descriptor)
            if os.fstat(descriptor).st_nlink != 0:
                raise OSError
    except OSError:
        pass


def _write_new_private_file(
    path: Path,
    payload: bytes,
    *,
    maximum: int,
    mode: int = 0o600,
) -> tuple[int, ...]:
    parent_descriptor = -1
    try:
        parent_before = path.parent.lstat()
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        parent_descriptor = os.open(path.parent, flags)
        if _identity(os.fstat(parent_descriptor)) != _identity(parent_before):
            raise LiveKitCausalEndpointError("private output contract failed")
        identity = _write_new_private_file_at(
            parent_descriptor,
            path.name,
            payload,
            maximum=maximum,
            mode=mode,
        )
        if _identity(os.fstat(parent_descriptor)) != _identity(path.parent.lstat()):
            raise LiveKitCausalEndpointError("private output contract failed")
        return identity
    except LiveKitCausalEndpointError:
        raise
    except OSError:
        raise LiveKitCausalEndpointError("private output contract failed") from None
    finally:
        if parent_descriptor >= 0:
            os.close(parent_descriptor)


def _validate_inventory_receipt(path: Path | str) -> Mapping[str, object]:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        raise LiveKitCausalEndpointError("inventory receipt prerequisite failed")
    lexical = Path(os.path.abspath(candidate))
    try:
        if lexical.resolve(strict=True) != lexical:
            raise LiveKitCausalEndpointError("inventory receipt prerequisite failed")
        payload, _identity_value = _read_stable_regular(
            lexical,
            maximum=64 * 1024,
            private=True,
        )
    except LiveKitCausalEndpointError:
        raise
    except (OSError, RuntimeError, ValueError):
        raise LiveKitCausalEndpointError("inventory receipt prerequisite failed") from None
    if hashlib.sha256(payload).hexdigest() != INVENTORY_REPORT_SHA256:
        raise LiveKitCausalEndpointError("inventory receipt mismatch")
    value = _strict_json(payload)
    if not isinstance(value, dict):
        raise LiveKitCausalEndpointError("inventory receipt mismatch")
    inventory = value.get("inventory")
    worker = value.get("worker")
    if (
        value.get("schema_version") != 1
        or value.get("kind") != "livekit-eot-english-aggregate-inventory-v1"
        or value.get("dataset") != DATASET
        or value.get("config") != DATASET_CONFIG
        or value.get("split") != DATASET_SPLIT
        or value.get("license") != LICENSE
        or value.get("source_revision") != inventory_worker.SOURCE_REVISION
        or value.get("source_size_bytes") != inventory_worker.SOURCE_SIZE_BYTES
        or value.get("source_sha256") != inventory_worker.SOURCE_SHA256
        or value.get("schema_contract_sha256") != inventory_worker.SCHEMA_CONTRACT_SHA256
        or value.get("production_evidence") is not True
        or not isinstance(inventory, dict)
        or not isinstance(worker, dict)
        or worker.get("pyarrow_version") != inventory_worker.PYARROW_VERSION
        or worker.get("production_evidence") is not True
        or inventory.get("row_count") != _EXPECTED_ROWS
        or inventory.get("silence_span_count") != _EXPECTED_SPANS
        or inventory.get("hold_label_count") != _EXPECTED_HOLDS
        or inventory.get("eot_label_count") != _EXPECTED_ROWS
        or inventory.get("audio_samples_total") != _EXPECTED_PCM_SAMPLES
        or inventory.get("audio_set_sha256") != AUDIO_SET_SHA256
        or inventory.get("silence_set_sha256") != SILENCE_SET_SHA256
        or inventory.get("inventory_sha256") != INVENTORY_SHA256
        or inventory.get("duration_exact_sample_count") != 394
        or inventory.get("duration_microsecond_quantized_count") != 6
        or inventory.get("final_gap_zero_sample_count") != 394
        or inventory.get("final_gap_one_sample_count") != 6
    ):
        raise LiveKitCausalEndpointError("inventory receipt mismatch")
    return value


def _exact_samples(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LiveKitCausalEndpointError(f"{name} must be numeric")
    try:
        scaled = Decimal(str(value)) * Decimal(SAMPLE_RATE_HZ)
    except (InvalidOperation, ValueError):
        raise LiveKitCausalEndpointError(f"{name} is invalid") from None
    integral = scaled.to_integral_value()
    if not scaled.is_finite() or scaled != integral or integral <= 0:
        raise LiveKitCausalEndpointError(f"{name} is not exact in samples")
    return int(integral)


def _load_policy_contract(config_path: Path | str) -> EndpointPolicyContract:
    candidate = Path(config_path).expanduser()
    expected = _REPO_ROOT / "config.json"
    if not candidate.is_absolute():
        raise LiveKitCausalEndpointError("config prerequisite failed")
    lexical = Path(os.path.abspath(candidate))
    try:
        if lexical.resolve(strict=True) != expected or lexical != expected:
            raise LiveKitCausalEndpointError("config prerequisite failed")
        payload = public_fixtures._read_stable_repo_worker(lexical)
    except LiveKitCausalEndpointError:
        raise
    except Exception:
        raise LiveKitCausalEndpointError("config prerequisite failed") from None
    value = _strict_json(payload)
    if not isinstance(value, dict) or not isinstance(value.get("sherpa"), dict):
        raise LiveKitCausalEndpointError("config prerequisite failed")
    sherpa = value["sherpa"]

    def exact_bool(name: str, default: bool) -> bool:
        item = sherpa.get(name, default)
        if not isinstance(item, bool):
            raise LiveKitCausalEndpointError("endpoint config prerequisite failed")
        return item

    def exact_int(name: str, default: int) -> int:
        item = sherpa.get(name, default)
        if isinstance(item, bool) or not isinstance(item, int):
            raise LiveKitCausalEndpointError("endpoint config prerequisite failed")
        return item

    def exact_float(name: str, default: float) -> float:
        item = sherpa.get(name, default)
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise LiveKitCausalEndpointError("endpoint config prerequisite failed")
        result = float(item)
        if not math.isfinite(result):
            raise LiveKitCausalEndpointError("endpoint config prerequisite failed")
        return result

    endpoint: dict[str, object] = {
        "enabled": exact_bool("endpoint_enabled", False),
        "min_silence_sec": exact_float("endpoint_min_silence_sec", 0.5),
        "max_silence_sec": exact_float("endpoint_max_silence_sec", 1.6),
        "complete_threshold": exact_float("endpoint_complete_threshold", 0.6),
        "incomplete_threshold": exact_float("endpoint_incomplete_threshold", 0.3),
        "high_confidence_floor": exact_float(
            "endpoint_high_confidence_floor", 0.0
        ),
        "high_confidence_score": exact_float(
            "endpoint_high_confidence_score", 0.75
        ),
        "adaptive_floor": exact_bool("endpoint_adaptive_floor", False),
        "pause_window": exact_int("endpoint_pause_window", 64),
        "pause_quantile": exact_float("endpoint_pause_quantile", 0.85),
        "pause_margin": exact_float("endpoint_pause_margin", 0.15),
        "pause_min_samples": exact_int("endpoint_pause_min_samples", 8),
    }
    sample_rate = exact_int("sample_rate", SAMPLE_RATE_HZ)
    endpoint_detector = sherpa.get("endpoint_detector", "lexical")
    endpoint_model = sherpa.get("endpoint_prosody_model", "")
    if not isinstance(endpoint_detector, str) or not isinstance(endpoint_model, str):
        raise LiveKitCausalEndpointError("endpoint config prerequisite failed")
    if (
        sample_rate != SAMPLE_RATE_HZ
        or endpoint["enabled"] is not True
        or endpoint_detector != "lexical"
        or endpoint_model
        or endpoint["adaptive_floor"] is not True
    ):
        raise LiveKitCausalEndpointError("endpoint config prerequisite failed")
    acoustic_rule2 = _exact_samples(
        sherpa.get("asr_rule2_min_trailing_silence", 0.8),
        name="acoustic rule2",
    )
    prosody_min = _exact_samples(
        sherpa.get("endpoint_prosody_min_silence", 0.15),
        name="prosody decision window",
    )
    max_wait = _exact_samples(endpoint["max_silence_sec"], name="maximum wait")
    rule3 = _exact_samples(
        sherpa.get("asr_rule3_min_utterance_length", 20.0),
        name="acoustic rule3",
    )
    if (
        acoustic_rule2 != 12_800
        or prosody_min != 2_400
        or max_wait != 25_600
        or rule3 != 320_000
        or GRID_SAMPLES != 1_600
    ):
        raise LiveKitCausalEndpointError("endpoint config prerequisite failed")
    return EndpointPolicyContract(
        endpoint_config=endpoint,
        config_sha256=hashlib.sha256(payload).hexdigest(),
        sample_rate_hz=SAMPLE_RATE_HZ,
        grid_samples=GRID_SAMPLES,
        acoustic_rule2_samples=acoustic_rule2,
        prosody_min_samples=prosody_min,
        max_wait_samples=max_wait,
        rule3_samples=rule3,
        runtime_default_detector=endpoint_detector,
    )


def _snapshot_exact_model(
    source: Path | str,
    scratch: Path,
    *,
    scratch_descriptor: int | None = None,
    scratch_registry: ScratchRegistry | None = None,
) -> tuple[Path, Mapping[str, object]]:
    candidate = Path(source).expanduser()
    if not candidate.is_absolute():
        raise LiveKitCausalEndpointError("model prerequisite failed")
    lexical = Path(os.path.abspath(candidate))
    try:
        if lexical.resolve(strict=True) != lexical:
            raise LiveKitCausalEndpointError("model prerequisite failed")
        payload, _source_identity = _read_stable_regular(
            lexical,
            maximum=_SMART_TURN_MODEL_BYTES,
            private=False,
        )
    except LiveKitCausalEndpointError:
        raise
    except Exception:
        raise LiveKitCausalEndpointError("model prerequisite failed") from None
    if (
        len(payload) != _SMART_TURN_MODEL_BYTES
        or hashlib.sha256(payload).hexdigest() != _SMART_TURN_MODEL_SHA256
    ):
        raise LiveKitCausalEndpointError("model prerequisite failed")
    snapshot = scratch / _SMART_TURN_MODEL_FILENAME
    if scratch_descriptor is None:
        _write_new_private_file(
            snapshot,
            payload,
            maximum=_SMART_TURN_MODEL_BYTES,
            mode=0o400,
        )
    else:
        snapshot_descriptor = -1
        try:
            snapshot_descriptor, _created_identity = _create_filled_private_file_at(
                scratch_descriptor,
                _SMART_TURN_MODEL_FILENAME,
                payload,
                maximum=_SMART_TURN_MODEL_BYTES,
                mode=0o400,
            )
            if scratch_registry is not None:
                info = os.stat(
                    _SMART_TURN_MODEL_FILENAME,
                    dir_fd=scratch_descriptor,
                    follow_symlinks=False,
                )
                _register_open_scratch_descriptor(
                    scratch_registry,
                    (_SMART_TURN_MODEL_FILENAME,),
                    snapshot_descriptor,
                    is_directory=False,
                    path_info=info,
                )
                snapshot_descriptor = -1
        finally:
            if snapshot_descriptor >= 0:
                os.close(snapshot_descriptor)
    try:
        snapshot_payload, _ = _read_stable_regular(
            snapshot,
            maximum=_SMART_TURN_MODEL_BYTES,
            private=True,
        )
    except (LiveKitCausalEndpointError, OSError):
        raise LiveKitCausalEndpointError("model snapshot failed") from None
    if snapshot_payload != payload:
        raise LiveKitCausalEndpointError("model snapshot failed")
    return snapshot, {
        "model_id": "pipecat-ai/smart-turn-v3",
        "revision": _SMART_TURN_MODEL_REVISION,
        "filename": _SMART_TURN_MODEL_FILENAME,
        "size_bytes": _SMART_TURN_MODEL_BYTES,
        "sha256": _SMART_TURN_MODEL_SHA256,
        "license": _SMART_TURN_MODEL_LICENSE,
        "provider": "CPUExecutionProvider",
        "threads": 1,
    }


def _worker_environment(root: Path) -> dict[str, str]:
    return {
        "HOME": str(root),
        "TMPDIR": str(root),
        "PATH": os.defpath,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "ARROW_NUM_THREADS": "1",
    }


def _snapshot_model_venv_launch() -> VenvLaunchSnapshot:
    argv0 = Path(os.path.abspath(Path(sys.executable)))
    root = argv0.parent.parent
    try:
        root_snapshot = public_fixtures._snapshot_venv_path(root, directory=True)
        executable_snapshot = public_fixtures._snapshot_venv_path(
            argv0,
            directory=False,
        )
        marker_path = public_fixtures._effective_venv_marker(argv0, root)
        marker_payload, marker_identity = public_fixtures._read_valid_venv_marker(
            marker_path
        )
        if (
            root_snapshot[1] != Path(sys.prefix).resolve(strict=True)
            or executable_snapshot[1] != argv0.resolve(strict=True)
        ):
            raise LiveKitCausalEndpointError("model-stage Python prerequisite failed")
    except LiveKitCausalEndpointError:
        raise
    except Exception:
        raise LiveKitCausalEndpointError("model-stage Python prerequisite failed") from None
    return VenvLaunchSnapshot(
        argv0=argv0,
        root=root,
        root_snapshot=root_snapshot,
        executable_snapshot=executable_snapshot,
        marker_path=marker_path,
        marker_payload=marker_payload,
        marker_identity=marker_identity,
    )


def _rebind_model_venv_launch(snapshot: VenvLaunchSnapshot) -> None:
    try:
        if (
            public_fixtures._snapshot_venv_path(snapshot.root, directory=True)
            != snapshot.root_snapshot
            or public_fixtures._snapshot_venv_path(
                snapshot.argv0,
                directory=False,
            )
            != snapshot.executable_snapshot
            or public_fixtures._effective_venv_marker(
                snapshot.argv0,
                snapshot.root,
            )
            != snapshot.marker_path
            or public_fixtures._read_valid_venv_marker(snapshot.marker_path)
            != (snapshot.marker_payload, snapshot.marker_identity)
        ):
            raise LiveKitCausalEndpointError("model-stage Python changed")
    except LiveKitCausalEndpointError:
        raise
    except Exception:
        raise LiveKitCausalEndpointError("model-stage Python changed") from None


def _snapshot_materializer_venv_launch(argv0: Path) -> VenvLaunchSnapshot:
    lexical = Path(os.path.abspath(argv0))
    root = lexical.parent.parent
    try:
        root_snapshot = public_fixtures._snapshot_venv_path(root, directory=True)
        executable_snapshot = public_fixtures._snapshot_venv_path(
            lexical,
            directory=False,
        )
        marker_path = public_fixtures._effective_venv_marker(lexical, root)
        marker_payload, marker_identity = public_fixtures._read_valid_venv_marker(
            marker_path
        )
    except Exception:
        raise LiveKitCausalEndpointError("materializer prerequisite failed") from None
    return VenvLaunchSnapshot(
        argv0=lexical,
        root=root,
        root_snapshot=root_snapshot,
        executable_snapshot=executable_snapshot,
        marker_path=marker_path,
        marker_payload=marker_payload,
        marker_identity=marker_identity,
    )


def _rebind_materializer_venv_launch(snapshot: VenvLaunchSnapshot) -> None:
    try:
        if (
            public_fixtures._snapshot_venv_path(snapshot.root, directory=True)
            != snapshot.root_snapshot
            or public_fixtures._snapshot_venv_path(
                snapshot.argv0,
                directory=False,
            )
            != snapshot.executable_snapshot
            or public_fixtures._effective_venv_marker(
                snapshot.argv0,
                snapshot.root,
            )
            != snapshot.marker_path
            or public_fixtures._read_valid_venv_marker(snapshot.marker_path)
            != (snapshot.marker_payload, snapshot.marker_identity)
        ):
            raise LiveKitCausalEndpointError("materializer changed")
    except LiveKitCausalEndpointError:
        raise
    except Exception:
        raise LiveKitCausalEndpointError("materializer changed") from None


def _materialize_source(
    source_handle: object,
    *,
    scratch: Path,
    scratch_descriptor: int,
    scratch_identity: tuple[int, ...],
    parquet_python: Path | str,
    execution_closure: Mapping[str, str],
    scratch_registry: ScratchRegistry,
) -> MaterializerStageResult:
    _assert_private_scratch_boundary(scratch, scratch_descriptor, scratch_identity)
    worker_root = _make_private_subdirectory(
        scratch,
        "worker",
        parent_descriptor=scratch_descriptor,
        registry=scratch_registry,
    )
    output = _make_private_subdirectory(
        scratch,
        "materialized",
        parent_descriptor=scratch_descriptor,
        registry=scratch_registry,
    )
    try:
        interpreter = public_fixtures._validate_parquet_python(parquet_python)
        if not interpreter.is_absolute():
            raise LiveKitCausalEndpointError("materializer prerequisite failed")
        venv_launch = _snapshot_materializer_venv_launch(interpreter)
        validated_snapshot = public_fixtures._validate_parquet_python(
            venv_launch.argv0
        )
        if Path(os.path.abspath(validated_snapshot)) != venv_launch.argv0:
            raise LiveKitCausalEndpointError("materializer prerequisite failed")
        _rebind_materializer_venv_launch(venv_launch)
        interpreter = venv_launch.argv0
        interpreter_resolved = venv_launch.executable_snapshot[1]
        causal_payload = _read_stable_execution_closure_member(_CAUSAL_WORKER_PATH)
        inventory_payload = _read_stable_execution_closure_member(
            _INVENTORY_WORKER_PATH
        )
    except Exception:
        raise LiveKitCausalEndpointError("materializer prerequisite failed") from None
    causal_sha256 = hashlib.sha256(causal_payload).hexdigest()
    inventory_sha256 = hashlib.sha256(inventory_payload).hexdigest()
    if (
        causal_sha256 != execution_closure["tools/livekit_causal_endpoint_worker.py"]
        or inventory_sha256 != execution_closure["tools/livekit_eot_parquet_worker.py"]
    ):
        raise LiveKitCausalEndpointError("execution closure changed")
    causal_snapshot = worker_root / "livekit_causal_endpoint_worker.py"
    inventory_snapshot = worker_root / "livekit_eot_parquet_worker.py"
    process: subprocess.Popen[bytes] | None = None
    group_reaped = False
    causal_descriptor = -1
    inventory_descriptor = -1
    python_descriptor = -1
    request_descriptor = -1
    output_descriptor = -1
    worker_root_descriptor = -1
    try:
        worker_root_descriptor = os.dup(
            scratch_registry.entries[("worker",)].descriptor
        )
        worker_root_identity = _directory_entry_identity(
            os.fstat(worker_root_descriptor)
        )
        _rebind_verified_directory(
            worker_root_descriptor,
            worker_root,
            worker_root_identity,
        )
        causal_descriptor, _created_identity = _create_filled_private_file_at(
            worker_root_descriptor,
            causal_snapshot.name,
            causal_payload,
            maximum=1024 * 1024,
            mode=0o400,
        )
        causal_identity = _identity(os.fstat(causal_descriptor))
        _register_scratch_descriptor_copy(
            scratch_registry,
            ("worker", causal_snapshot.name),
            causal_descriptor,
            is_directory=False,
            parent_descriptor=worker_root_descriptor,
            name=causal_snapshot.name,
        )
        inventory_descriptor, _created_identity = _create_filled_private_file_at(
            worker_root_descriptor,
            inventory_snapshot.name,
            inventory_payload,
            maximum=1024 * 1024,
            mode=0o400,
        )
        inventory_identity = _identity(os.fstat(inventory_descriptor))
        _register_scratch_descriptor_copy(
            scratch_registry,
            ("worker", inventory_snapshot.name),
            inventory_descriptor,
            is_directory=False,
            parent_descriptor=worker_root_descriptor,
            name=inventory_snapshot.name,
        )
        python_descriptor, python_identity, python_sha256 = _open_verified_regular(
            interpreter_resolved,
            maximum=64 * 1024 * 1024,
            private=False,
        )
        output_descriptor = os.dup(
            scratch_registry.entries[("materialized",)].descriptor
        )
        output_identity = _directory_entry_identity(os.fstat(output_descriptor))
        _rebind_verified_directory(output_descriptor, output, output_identity)
        output_leaf_identities = _precreate_materialization_leaves(
            output_descriptor,
            scratch_registry,
        )
        request = worker_root / "request.json"
        request_payload = _canonical_json_bytes(
            {
                "schema_version": MATERIALIZATION_SCHEMA_VERSION,
                "source_descriptor": source_handle.descriptor,
                "source_revision": inventory_worker.SOURCE_REVISION,
                "source_size_bytes": inventory_worker.SOURCE_SIZE_BYTES,
                "source_sha256": inventory_worker.SOURCE_SHA256,
                "causal_worker_descriptor": causal_descriptor,
                "inventory_worker_descriptor": inventory_descriptor,
                "causal_worker_sha256": causal_sha256,
                "inventory_worker_sha256": inventory_sha256,
                "output_leaf_identities": output_leaf_identities,
            }
        )
        request_descriptor, _created_identity = _create_filled_private_file_at(
            worker_root_descriptor,
            request.name,
            request_payload,
            maximum=_MAX_MATERIALIZER_REQUEST_BYTES,
        )
        request_identity = _identity(os.fstat(request_descriptor))
        request_sha256 = hashlib.sha256(request_payload).hexdigest()
        _register_scratch_descriptor_copy(
            scratch_registry,
            ("worker", request.name),
            request_descriptor,
            is_directory=False,
            parent_descriptor=worker_root_descriptor,
            name=request.name,
        )
        _rebind_materializer_venv_launch(venv_launch)
        _assert_private_scratch_boundary(scratch, scratch_descriptor, scratch_identity)
        worker_root_fd_path = Path(f"/proc/self/fd/{worker_root_descriptor}")
        process = subprocess.Popen(
            [
                str(interpreter),
                "-I",
                "-B",
                f"/proc/self/fd/{causal_descriptor}",
                "--request-fd",
                str(request_descriptor),
                "--output-dir-fd",
                str(output_descriptor),
            ],
            executable=f"/proc/self/fd/{python_descriptor}",
            cwd=worker_root_fd_path,
            env=_worker_environment(worker_root_fd_path),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            pass_fds=(
                source_handle.descriptor,
                python_descriptor,
                causal_descriptor,
                inventory_descriptor,
                request_descriptor,
                output_descriptor,
                worker_root_descriptor,
            ),
            start_new_session=True,
        )
        return_code = process.wait(timeout=_WORKER_TIMEOUT_SECONDS)
        public_fixtures._terminate_worker_process_group(process)
        group_reaped = True
        if return_code != 0:
            raise LiveKitCausalEndpointError("materializer failed")
        causal_after = public_fixtures._read_stable_worker_file(
            causal_snapshot,
            maximum=1024 * 1024,
        )
        inventory_after = public_fixtures._read_stable_worker_file(
            inventory_snapshot,
            maximum=1024 * 1024,
        )
        if (
            hashlib.sha256(causal_after).hexdigest() != causal_sha256
            or hashlib.sha256(inventory_after).hexdigest() != inventory_sha256
        ):
            raise LiveKitCausalEndpointError("materializer changed")
        _rebind_verified_regular(
            causal_descriptor,
            causal_snapshot,
            causal_identity,
            causal_sha256,
        )
        _rebind_verified_regular(
            inventory_descriptor,
            inventory_snapshot,
            inventory_identity,
            inventory_sha256,
        )
        _rebind_verified_regular(
            python_descriptor,
            interpreter_resolved,
            python_identity,
            python_sha256,
        )
        _rebind_verified_regular(
            request_descriptor,
            request,
            request_identity,
            request_sha256,
        )
        _rebind_verified_directory(
            output_descriptor,
            output,
            output_identity,
        )
        _rebind_verified_directory(
            worker_root_descriptor,
            worker_root,
            worker_root_identity,
        )
        if set(os.listdir(worker_root_descriptor)) != {
            causal_snapshot.name,
            inventory_snapshot.name,
            request.name,
        }:
            raise LiveKitCausalEndpointError("materializer changed")
        _rebind_materializer_venv_launch(venv_launch)
        _rebind_execution_closure(execution_closure)
        return MaterializerStageResult(
            output=output,
            execution_receipt={
                "python_executable_sha256": python_sha256,
                "lexical_venv_argv0_preserved": True,
                "venv_marker_sha256": hashlib.sha256(
                    venv_launch.marker_payload
                ).hexdigest(),
                "pyarrow_version": inventory_worker.PYARROW_VERSION,
                "wall_timeout_seconds": _WORKER_TIMEOUT_SECONDS,
                "requested_cpu_soft_limit_ceiling_seconds": 600,
                "requested_address_space_soft_limit_ceiling_bytes": (
                    2 * 1024 * 1024 * 1024
                ),
                "requested_file_size_soft_limit_ceiling_bytes": 32 * 1024 * 1024,
                "requested_file_descriptor_soft_limit_ceiling": 64,
                "inherited_hard_limits_may_reduce_ceilings": True,
                "worker_threads": 1,
                "offline_environment": True,
                "network_namespace_isolation": False,
                "cgroup_scope": False,
            },
        )
    except subprocess.TimeoutExpired:
        raise LiveKitCausalEndpointError("materializer timed out") from None
    except LiveKitCausalEndpointError:
        raise
    except Exception:
        raise LiveKitCausalEndpointError("materializer failed") from None
    finally:
        if process is not None and not group_reaped:
            try:
                public_fixtures._terminate_worker_process_group(process)
            except Exception:
                pass
        for descriptor in (
            causal_descriptor,
            inventory_descriptor,
            python_descriptor,
            request_descriptor,
            output_descriptor,
            worker_root_descriptor,
        ):
            if descriptor >= 0:
                os.close(descriptor)


_MATERIALIZATION_FIELDS = {
    "schema_version",
    "kind",
    "pyarrow_version",
    "source_revision",
    "source_size_bytes",
    "source_sha256",
    "schema_contract_sha256",
    "inventory_report_sha256",
    "inventory_sha256",
    "sample_rate_hz",
    "pcm_contract",
    "label_convention",
    "grid_samples",
    "causal_worker_sha256",
    "inventory_worker_sha256",
    "row_count",
    "row_group_count",
    "audio_bytes_total",
    "pcm_bytes_total",
    "pcm_samples_total",
    "silence_span_count",
    "hold_label_count",
    "eot_label_count",
    "off_grid_span_count",
    "duration_exact_sample_count",
    "duration_microsecond_quantized_count",
    "final_gap_zero_sample_count",
    "final_gap_one_sample_count",
    "pcm_set_sha256",
    "materialization_sha256",
    "rows",
}
_MATERIALIZED_ROW_FIELDS = {
    "ordinal",
    "pcm_filename",
    "pcm_bytes",
    "pcm_samples",
    "pcm_sha256",
    "silence_spans",
}


def _read_private_regular_at(
    directory_descriptor: int,
    name: str,
    *,
    maximum: int,
    expected_stable_identity: tuple[int, ...] | None = None,
) -> tuple[bytes, tuple[int, ...]]:
    descriptor = -1
    try:
        before = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > maximum
            or stat.S_IMODE(before.st_mode) & 0o077
            or (hasattr(os, "geteuid") and before.st_uid != os.geteuid())
            or (
                expected_stable_identity is not None
                and _stable_file_identity(before) != expected_stable_identity
            )
        ):
            raise LiveKitCausalEndpointError("materialization contract failed")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(name, flags, dir_fd=directory_descriptor)
        opened = os.fstat(descriptor)
        if (
            _identity(opened) != _identity(before)
            or (
                expected_stable_identity is not None
                and _stable_file_identity(opened) != expected_stable_identity
            )
        ):
            raise LiveKitCausalEndpointError("materialization contract failed")
        payload = _pread_descriptor(descriptor, opened.st_size)
        after = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        if (
            _identity(os.fstat(descriptor)) != _identity(opened)
            or _identity(after) != _identity(opened)
            or (
                expected_stable_identity is not None
                and _stable_file_identity(after) != expected_stable_identity
            )
        ):
            raise LiveKitCausalEndpointError("materialization contract failed")
        return payload, _identity(opened)
    except LiveKitCausalEndpointError:
        raise
    except OSError:
        raise LiveKitCausalEndpointError("materialization contract failed") from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _rebind_materialization_leaf_identities(
    directory_descriptor: int,
    expected: Mapping[str, tuple[int, ...]],
) -> None:
    if not expected:
        return
    try:
        if set(os.listdir(directory_descriptor)) != set(expected):
            raise LiveKitCausalEndpointError("materialization changed")
        for name, identity in expected.items():
            info = os.stat(
                name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            if _stable_file_identity(info) != identity:
                raise LiveKitCausalEndpointError("materialization changed")
    except LiveKitCausalEndpointError:
        raise
    except OSError:
        raise LiveKitCausalEndpointError("materialization changed") from None


def _load_materialization(
    directory: Path,
    execution_closure: Mapping[str, str],
    *,
    retained_directory_descriptor: int = -1,
    retained_leaf_entries: Mapping[str, ScratchEntry] | None = None,
) -> Materialization:
    descriptor = -1
    try:
        resolved = directory.resolve(strict=True)
        before = directory.lstat()
        if retained_directory_descriptor >= 0:
            descriptor = os.dup(retained_directory_descriptor)
        else:
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(directory, flags)
        info = os.fstat(descriptor)
    except (OSError, RuntimeError, ValueError):
        raise LiveKitCausalEndpointError("materialization contract failed") from None
    if (
        resolved != directory
        or _identity(info) != _identity(before)
        or not stat.S_ISDIR(info.st_mode)
        or stat.S_IMODE(info.st_mode) != 0o700
        or (hasattr(os, "geteuid") and info.st_uid != os.geteuid())
    ):
        os.close(descriptor)
        raise LiveKitCausalEndpointError("materialization contract failed")
    try:
        entries = set(os.listdir(descriptor))
        entry_stats = {
            name: os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            for name in entries
        }
        leaf_identities: dict[str, tuple[int, ...]] = {}
        if retained_leaf_entries is not None:
            if set(retained_leaf_entries) != entries:
                raise LiveKitCausalEndpointError("materialization contract failed")
            for name, entry in retained_leaf_entries.items():
                info_at_path = entry_stats[name]
                if (
                    entry.is_directory
                    or not _scratch_entry_matches(entry, info_at_path)
                ):
                    raise LiveKitCausalEndpointError(
                        "materialization contract failed"
                    )
                retained_info = os.fstat(entry.descriptor)
                stable_identity = _stable_file_identity(retained_info)
                if (
                    not stat.S_ISREG(retained_info.st_mode)
                    or retained_info.st_nlink != 1
                    or stat.S_IMODE(retained_info.st_mode) != 0o600
                    or _stable_file_identity(info_at_path) != stable_identity
                ):
                    raise LiveKitCausalEndpointError(
                        "materialization contract failed"
                    )
                leaf_identities[name] = stable_identity
        payload, _manifest_identity = _read_private_regular_at(
            descriptor,
            "manifest.json",
            maximum=_MAX_PRIVATE_JSON_BYTES,
            expected_stable_identity=leaf_identities.get("manifest.json"),
        )
        if (
            _identity(os.fstat(descriptor)) != _identity(info)
            or _identity(directory.lstat()) != _identity(info)
        ):
            raise LiveKitCausalEndpointError("materialization contract failed")
    except LiveKitCausalEndpointError:
        raise
    except OSError:
        raise LiveKitCausalEndpointError("materialization contract failed") from None
    finally:
        os.close(descriptor)
        descriptor = -1
    value = _strict_json(payload)
    if not isinstance(value, dict) or set(value) != _MATERIALIZATION_FIELDS:
        raise LiveKitCausalEndpointError("materialization contract failed")
    schema_version = value.get("schema_version")
    off_grid_span_count = value.get("off_grid_span_count")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != MATERIALIZATION_SCHEMA_VERSION
        or isinstance(off_grid_span_count, bool)
        or not isinstance(off_grid_span_count, int)
        or off_grid_span_count < 0
        or value.get("kind") != MATERIALIZATION_KIND
        or value.get("pyarrow_version") != inventory_worker.PYARROW_VERSION
        or value.get("source_revision") != inventory_worker.SOURCE_REVISION
        or value.get("source_size_bytes") != inventory_worker.SOURCE_SIZE_BYTES
        or value.get("source_sha256") != inventory_worker.SOURCE_SHA256
        or value.get("schema_contract_sha256") != inventory_worker.SCHEMA_CONTRACT_SHA256
        or value.get("inventory_report_sha256") != INVENTORY_REPORT_SHA256
        or value.get("inventory_sha256") != INVENTORY_SHA256
        or value.get("sample_rate_hz") != SAMPLE_RATE_HZ
        or value.get("pcm_contract") != "signed-pcm16-little-endian-mono-16000hz"
        or value.get("label_convention") != LABEL_CONVENTION
        or value.get("grid_samples") != GRID_SAMPLES
        or value.get("causal_worker_sha256")
        != execution_closure["tools/livekit_causal_endpoint_worker.py"]
        or value.get("inventory_worker_sha256")
        != execution_closure["tools/livekit_eot_parquet_worker.py"]
        or value.get("row_count") != _EXPECTED_ROWS
        or value.get("row_group_count") != 2
        or value.get("audio_bytes_total") != 161_326_412
        or value.get("pcm_samples_total") != _EXPECTED_PCM_SAMPLES
        or value.get("pcm_bytes_total") != _EXPECTED_PCM_SAMPLES * 2
        or value.get("silence_span_count") != _EXPECTED_SPANS
        or value.get("hold_label_count") != _EXPECTED_HOLDS
        or value.get("eot_label_count") != _EXPECTED_ROWS
        or value.get("duration_exact_sample_count") != 394
        or value.get("duration_microsecond_quantized_count") != 6
        or value.get("final_gap_zero_sample_count") != 394
        or value.get("final_gap_one_sample_count") != 6
        or _SHA256_RE.fullmatch(str(value.get("pcm_set_sha256"))) is None
        or _SHA256_RE.fullmatch(str(value.get("materialization_sha256"))) is None
    ):
        raise LiveKitCausalEndpointError("materialization contract failed")
    raw_rows = value.get("rows")
    if not isinstance(raw_rows, list) or len(raw_rows) != _EXPECTED_ROWS:
        raise LiveKitCausalEndpointError("materialization contract failed")
    rows: list[MaterializedRow] = []
    pcm_total = 0
    sample_total = 0
    span_total = 0
    hold_total = 0
    off_grid_total = 0
    pcm_set = hashlib.sha256()
    materialization_digest = hashlib.sha256()
    expected_names = {"manifest.json"}
    for ordinal, raw in enumerate(raw_rows):
        if not isinstance(raw, dict) or set(raw) != _MATERIALIZED_ROW_FIELDS:
            raise LiveKitCausalEndpointError("materialization contract failed")
        filename = raw.get("pcm_filename")
        pcm_bytes = raw.get("pcm_bytes")
        pcm_samples = raw.get("pcm_samples")
        pcm_sha256 = raw.get("pcm_sha256")
        raw_spans = raw.get("silence_spans")
        raw_ordinal = raw.get("ordinal")
        if (
            isinstance(raw_ordinal, bool)
            or not isinstance(raw_ordinal, int)
            or raw_ordinal != ordinal
            or filename != f"row-{ordinal:04d}.pcm"
            or not isinstance(filename, str)
            or not isinstance(pcm_bytes, int)
            or isinstance(pcm_bytes, bool)
            or not 0 < pcm_bytes <= _MAX_PCM_BYTES
            or pcm_bytes % 2
            or not isinstance(pcm_samples, int)
            or isinstance(pcm_samples, bool)
            or pcm_samples != pcm_bytes // 2
            or not isinstance(pcm_sha256, str)
            or _SHA256_RE.fullmatch(pcm_sha256) is None
            or not isinstance(raw_spans, list)
            or not 1 <= len(raw_spans) <= 64
        ):
            raise LiveKitCausalEndpointError("materialization contract failed")
        spans: list[CausalSpan] = []
        packed_spans = bytearray()
        prior_end = -1
        for raw_span in raw_spans:
            if (
                not isinstance(raw_span, list)
                or len(raw_span) != 2
                or any(isinstance(item, bool) or not isinstance(item, int) for item in raw_span)
            ):
                raise LiveKitCausalEndpointError("materialization contract failed")
            start, end = raw_span
            if (
                start < 0
                or end <= start
                or start < prior_end
                or end > pcm_samples
                or end - start < 1_600
                or end - start > 80_000
            ):
                raise LiveKitCausalEndpointError("materialization contract failed")
            spans.append(CausalSpan(start, end))
            packed_spans.extend(struct.pack(">QQ", start, end))
            if (end - start) % GRID_SAMPLES:
                off_grid_total += 1
            prior_end = end
        final_gap = pcm_samples - spans[-1].end_sample
        if spans[-1].duration_samples < 3_200 or final_gap not in {0, 1}:
            raise LiveKitCausalEndpointError("materialization contract failed")
        expected_names.add(filename)
        pcm_info = entry_stats.get(filename)
        if (
            pcm_info is None
            or not stat.S_ISREG(pcm_info.st_mode)
            or pcm_info.st_nlink != 1
            or stat.S_IMODE(pcm_info.st_mode) != 0o600
            or pcm_info.st_size != pcm_bytes
            or (hasattr(os, "geteuid") and pcm_info.st_uid != os.geteuid())
        ):
            raise LiveKitCausalEndpointError("materialization contract failed")
        pcm_total += pcm_bytes
        sample_total += pcm_samples
        span_total += len(spans)
        hold_total += len(spans) - 1
        pcm_digest = bytes.fromhex(pcm_sha256)
        pcm_set.update(pcm_digest)
        materialization_digest.update(pcm_digest)
        materialization_digest.update(bytes(packed_spans))
        materialization_digest.update(
            struct.pack(">QQQ", ordinal, pcm_bytes, len(spans))
        )
        rows.append(
            MaterializedRow(
                ordinal=ordinal,
                pcm_filename=filename,
                pcm_bytes=pcm_bytes,
                pcm_samples=pcm_samples,
                pcm_sha256=pcm_sha256,
                silence_spans=tuple(spans),
            )
        )
    if (
        entries != expected_names
        or pcm_total != value["pcm_bytes_total"]
        or sample_total != value["pcm_samples_total"]
        or span_total != value["silence_span_count"]
        or hold_total != value["hold_label_count"]
        or off_grid_total != value["off_grid_span_count"]
        or pcm_set.hexdigest() != value["pcm_set_sha256"]
        or materialization_digest.hexdigest() != value["materialization_sha256"]
    ):
        raise LiveKitCausalEndpointError("materialization contract failed")
    if leaf_identities:
        _rebind_materialization_leaf_identities(
            retained_directory_descriptor,
            leaf_identities,
        )
    return Materialization(
        directory=directory,
        directory_identity=_identity(info),
        manifest_sha256=hashlib.sha256(payload).hexdigest(),
        manifest=value,
        rows=tuple(rows),
        leaf_identities=leaf_identities,
        retained_directory_descriptor=retained_directory_descriptor,
    )


@contextmanager
def _opened_materialization(materialization: Materialization) -> Iterator[int]:
    descriptor = -1
    try:
        if materialization.retained_directory_descriptor >= 0:
            descriptor = os.dup(materialization.retained_directory_descriptor)
        else:
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(materialization.directory, flags)
        opened = os.fstat(descriptor)
        if (
            _identity(opened) != materialization.directory_identity
            or _identity(materialization.directory.lstat()) != materialization.directory_identity
        ):
            raise LiveKitCausalEndpointError("materialization changed")
        _rebind_materialization_leaf_identities(
            descriptor,
            materialization.leaf_identities,
        )
        yield descriptor
        _rebind_materialization_leaf_identities(
            descriptor,
            materialization.leaf_identities,
        )
        if (
            _identity(os.fstat(descriptor)) != materialization.directory_identity
            or _identity(materialization.directory.lstat()) != materialization.directory_identity
        ):
            raise LiveKitCausalEndpointError("materialization changed")
    except LiveKitCausalEndpointError:
        raise
    except OSError:
        raise LiveKitCausalEndpointError("materialization changed") from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _read_materialized_pcm(
    directory_descriptor: int,
    row: MaterializedRow,
    *,
    expected_stable_identity: tuple[int, ...] | None = None,
) -> bytes:
    descriptor = -1
    try:
        before = os.stat(
            row.pcm_filename,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_size != row.pcm_bytes
            or (hasattr(os, "geteuid") and before.st_uid != os.geteuid())
            or (
                expected_stable_identity is not None
                and _stable_file_identity(before) != expected_stable_identity
            )
        ):
            raise LiveKitCausalEndpointError("materialized PCM changed")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(row.pcm_filename, flags, dir_fd=directory_descriptor)
        opened = os.fstat(descriptor)
        if (
            _identity(opened) != _identity(before)
            or (
                expected_stable_identity is not None
                and _stable_file_identity(opened) != expected_stable_identity
            )
        ):
            raise LiveKitCausalEndpointError("materialized PCM changed")
        payload = _read_descriptor(descriptor, row.pcm_bytes)
        after = os.stat(
            row.pcm_filename,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if (
            _identity(os.fstat(descriptor)) != _identity(opened)
            or _identity(after) != _identity(opened)
            or hashlib.sha256(payload).hexdigest() != row.pcm_sha256
            or (
                expected_stable_identity is not None
                and _stable_file_identity(after) != expected_stable_identity
            )
        ):
            raise LiveKitCausalEndpointError("materialized PCM changed")
        return payload
    except LiveKitCausalEndpointError:
        raise
    except OSError:
        raise LiveKitCausalEndpointError("materialized PCM changed") from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _rebind_materialization(materialization: Materialization) -> None:
    with _opened_materialization(materialization) as descriptor:
        payload, _identity_value = _read_private_regular_at(
            descriptor,
            "manifest.json",
            maximum=_MAX_PRIVATE_JSON_BYTES,
            expected_stable_identity=materialization.leaf_identities.get(
                "manifest.json"
            ),
        )
        if (
            hashlib.sha256(payload).hexdigest() != materialization.manifest_sha256
            or set(os.listdir(descriptor))
            != {"manifest.json", *(row.pcm_filename for row in materialization.rows)}
        ):
            raise LiveKitCausalEndpointError("materialization changed")
        for row in materialization.rows:
            _read_materialized_pcm(
                descriptor,
                row,
                expected_stable_identity=materialization.leaf_identities.get(
                    row.pcm_filename
                ),
            )


def _materialized_rows_payload(
    rows: Sequence[MaterializedRow],
) -> list[Mapping[str, object]]:
    return [
        {
            "ordinal": row.ordinal,
            "pcm_filename": row.pcm_filename,
            "pcm_bytes": row.pcm_bytes,
            "pcm_samples": row.pcm_samples,
            "pcm_sha256": row.pcm_sha256,
            "silence_spans": [
                [span.start_sample, span.end_sample] for span in row.silence_spans
            ],
        }
        for row in rows
    ]


def _create_empty_private_file_at(
    directory_descriptor: int,
    name: str,
) -> tuple[int, tuple[int, ...]]:
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise LiveKitCausalEndpointError("private output contract failed")
    descriptor = -1
    created = False
    created_identity: tuple[int, ...] | None = None
    try:
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(name, flags, 0o600, dir_fd=directory_descriptor)
        created = True
        opened = os.fstat(descriptor)
        created_identity = _inode_identity(opened)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_size != 0
            or stat.S_IMODE(opened.st_mode) != 0o600
            or (hasattr(os, "geteuid") and opened.st_uid != os.geteuid())
        ):
            raise LiveKitCausalEndpointError("private output contract failed")
        return descriptor, _directory_entry_identity(opened) + (opened.st_nlink,)
    except LiveKitCausalEndpointError:
        _unlink_private_leaf_if_bound(
            directory_descriptor,
            name,
            descriptor=descriptor,
            created=created,
            created_identity=created_identity,
        )
        if descriptor >= 0:
            os.close(descriptor)
        raise
    except OSError:
        _unlink_private_leaf_if_bound(
            directory_descriptor,
            name,
            descriptor=descriptor,
            created=created,
            created_identity=created_identity,
        )
        if descriptor >= 0:
            os.close(descriptor)
        raise LiveKitCausalEndpointError("private output contract failed") from None


def _precreate_materialization_leaves(
    directory_descriptor: int,
    registry: ScratchRegistry,
) -> Mapping[str, list[int]]:
    """Own every deterministic worker leaf before any private row is decoded."""

    identities: dict[str, list[int]] = {}
    names = (
        *(f"row-{ordinal:04d}.pcm" for ordinal in range(_EXPECTED_ROWS)),
        "manifest.json",
    )
    for name in names:
        descriptor = -1
        try:
            descriptor, stable_identity = _create_empty_private_file_at(
                directory_descriptor,
                name,
            )
            info = os.stat(
                name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            _register_open_scratch_descriptor(
                registry,
                ("materialized", name),
                descriptor,
                is_directory=False,
                path_info=info,
            )
            descriptor = -1
            identities[name] = list(stable_identity)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
    if len(identities) != _EXPECTED_ROWS + 1:
        raise LiveKitCausalEndpointError("materialization ownership failed")
    return identities


def _stable_file_identity(info: os.stat_result) -> tuple[int, ...]:
    return _directory_entry_identity(info) + (info.st_nlink,)


def _validate_metric_summary(
    value: object,
    *,
    expected_count: int,
    maximum_samples: int,
    maximum_is_strict: bool = False,
) -> None:
    if not isinstance(value, dict) or set(value) != {"samples", "milliseconds"}:
        raise LiveKitCausalEndpointError("model-stage result contract failed")
    samples = value.get("samples")
    milliseconds = value.get("milliseconds")
    sample_fields = {"count", "sum", "min", "p50_nearest_rank", "p95_nearest_rank", "max"}
    millisecond_fields = sample_fields - {"count"}
    if (
        not isinstance(samples, dict)
        or set(samples) != sample_fields
        or not isinstance(milliseconds, dict)
        or set(milliseconds) != millisecond_fields
        or isinstance(samples.get("count"), bool)
        or not isinstance(samples.get("count"), int)
        or samples.get("count") != expected_count
    ):
        raise LiveKitCausalEndpointError("model-stage result contract failed")
    for name in sample_fields - {"count"}:
        item = samples.get(name)
        if item is not None and (
            isinstance(item, bool) or not isinstance(item, int) or item < 0
        ):
            raise LiveKitCausalEndpointError("model-stage result contract failed")
    if expected_count == 0:
        if samples != {
            "count": 0,
            "sum": 0,
            "min": None,
            "p50_nearest_rank": None,
            "p95_nearest_rank": None,
            "max": None,
        }:
            raise LiveKitCausalEndpointError("model-stage result contract failed")
    else:
        if (
            samples["min"] is None
            or samples["max"] is None
            or samples["p50_nearest_rank"] is None
            or samples["p95_nearest_rank"] is None
            or samples["sum"] < samples["min"] * expected_count
            or samples["sum"] > samples["max"] * expected_count
            or not samples["min"] <= samples["p50_nearest_rank"] <= samples["max"]
            or not samples["min"] <= samples["p95_nearest_rank"] <= samples["max"]
            or samples["p50_nearest_rank"] > samples["p95_nearest_rank"]
        ):
            raise LiveKitCausalEndpointError("model-stage result contract failed")
        if (
            samples["max"] >= maximum_samples
            if maximum_is_strict
            else samples["max"] > maximum_samples
        ):
            raise LiveKitCausalEndpointError("model-stage result contract failed")
    for name, item in milliseconds.items():
        sample_name = name
        expected = (
            None
            if samples[sample_name] is None
            else samples[sample_name] * 1000.0 / SAMPLE_RATE_HZ
        )
        if expected is None:
            valid = item is None
        else:
            valid = (
                not isinstance(item, bool)
                and isinstance(item, (int, float))
                and math.isfinite(float(item))
                and item == expected
            )
        if not valid:
            raise LiveKitCausalEndpointError("model-stage result contract failed")


def _validate_profile_result(
    value: object,
    *,
    expected_partial_state: str,
    expected_holds: int,
    expected_eot: int,
) -> None:
    fields = {
        "partial_state",
        "hold",
        "eot",
        "recorded_prefix_decision_count",
        "completion_state_counts",
        "decision_basis_counts",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise LiveKitCausalEndpointError("model-stage result contract failed")
    hold = value.get("hold")
    eot = value.get("eot")
    states = value.get("completion_state_counts")
    bases = value.get("decision_basis_counts")
    decisions = value.get("recorded_prefix_decision_count")
    if (
        value.get("partial_state") != expected_partial_state
        or not isinstance(hold, dict)
        or set(hold)
        != {"denominator", "early_cut_count", "early_cut_rate", "rows_with_any_early_cut"}
        or isinstance(hold.get("denominator"), bool)
        or not isinstance(hold.get("denominator"), int)
        or hold.get("denominator") != expected_holds
        or not isinstance(eot, dict)
        or set(eot)
        != {
            "denominator",
            "publisher_labelled_commit_count",
            "right_censored_at_publisher_labelled_silence_end_count",
            "observed_publisher_labelled_silence_delay",
            "publisher_labelled_censored_delay_lower_bound",
            "conservative_all_row_delay_with_censored_at_max_wait",
        }
        or isinstance(eot.get("denominator"), bool)
        or not isinstance(eot.get("denominator"), int)
        or eot.get("denominator") != expected_eot
        or isinstance(decisions, bool)
        or not isinstance(decisions, int)
        or decisions < 0
        or not isinstance(states, dict)
        or not isinstance(bases, dict)
    ):
        raise LiveKitCausalEndpointError("model-stage result contract failed")
    early = hold.get("early_cut_count")
    rows_with_cut = hold.get("rows_with_any_early_cut")
    rate = hold.get("early_cut_rate")
    if (
        isinstance(early, bool)
        or not isinstance(early, int)
        or not 0 <= early <= expected_holds
        or isinstance(rows_with_cut, bool)
        or not isinstance(rows_with_cut, int)
        or not 0 <= rows_with_cut <= min(_EXPECTED_ROWS, early)
        or (
            rate is not None
            and (
                isinstance(rate, bool)
                or not isinstance(rate, (int, float))
                or not math.isfinite(float(rate))
            )
        )
        or rate != (early / expected_holds if expected_holds else None)
    ):
        raise LiveKitCausalEndpointError("model-stage result contract failed")
    committed = eot.get("publisher_labelled_commit_count")
    censored = eot.get(
        "right_censored_at_publisher_labelled_silence_end_count"
    )
    if (
        isinstance(committed, bool)
        or not isinstance(committed, int)
        or isinstance(censored, bool)
        or not isinstance(censored, int)
        or committed < 0
        or censored < 0
        or committed + censored != expected_eot
    ):
        raise LiveKitCausalEndpointError("model-stage result contract failed")
    observed_summary = eot["observed_publisher_labelled_silence_delay"]
    conservative_summary = eot[
        "conservative_all_row_delay_with_censored_at_max_wait"
    ]
    _validate_metric_summary(
        observed_summary,
        expected_count=committed,
        maximum_samples=25_600,
    )
    _validate_metric_summary(
        eot["publisher_labelled_censored_delay_lower_bound"],
        expected_count=censored,
        maximum_samples=25_600,
        maximum_is_strict=True,
    )
    _validate_metric_summary(
        conservative_summary,
        expected_count=expected_eot,
        maximum_samples=25_600,
    )
    observed_samples = observed_summary["samples"]
    conservative_samples = conservative_summary["samples"]
    if censored == 0:
        if conservative_summary != observed_summary:
            raise LiveKitCausalEndpointError("model-stage result contract failed")
    else:
        expected_min = 25_600 if committed == 0 else observed_samples["min"]
        if (
            conservative_samples["sum"]
            != observed_samples["sum"] + censored * 25_600
            or conservative_samples["min"] != expected_min
            or conservative_samples["max"] != 25_600
            or (
                committed > 0
                and (
                    conservative_samples["p50_nearest_rank"]
                    < observed_samples["p50_nearest_rank"]
                    or conservative_samples["p95_nearest_rank"]
                    < observed_samples["p95_nearest_rank"]
                )
            )
        ):
            raise LiveKitCausalEndpointError("model-stage result contract failed")
        conservative_ranks = {
            "p50_nearest_rank": (expected_eot + 1) // 2,
            "p95_nearest_rank": (95 * expected_eot + 99) // 100,
        }
        if any(
            rank > committed and conservative_samples[name] != 25_600
            for name, rank in conservative_ranks.items()
        ):
            raise LiveKitCausalEndpointError("model-stage result contract failed")
    for counter in (states, bases):
        if any(
            not isinstance(key, str)
            or isinstance(item, bool)
            or not isinstance(item, int)
            or item <= 0
            for key, item in counter.items()
        ) or sum(counter.values()) != decisions:
            raise LiveKitCausalEndpointError("model-stage result contract failed")
    allowed_states = (
        {"audio_window_pending", "scored"}
        if expected_partial_state == PARTIAL_ASSUMPTION
        else {"no_partial"}
    )
    allowed_bases = (
        {"acoustic", "semantic_early", "semantic_hold", "semantic_wait"}
        if expected_partial_state == PARTIAL_ASSUMPTION
        else {"acoustic"}
    )
    if (
        not set(states) <= allowed_states
        or not set(bases) <= allowed_bases
        or decisions > (expected_holds + expected_eot) * 16
    ):
        raise LiveKitCausalEndpointError("model-stage result contract failed")


def _fully_before_rule3_denominators(
    rows: Sequence[MaterializedRow],
    rule3_samples: int,
) -> tuple[int, int]:
    holds = 0
    eot = 0
    for row in rows:
        for index, span in enumerate(row.silence_spans):
            if span.end_sample < rule3_samples:
                if index == len(row.silence_spans) - 1:
                    eot += 1
                else:
                    holds += 1
    return holds, eot


def _stage_registered_model_source_bundle(
    relative_sources: Mapping[str, Path],
    *,
    scratch: Path,
    scratch_descriptor: int,
    scratch_registry: ScratchRegistry,
    required_files: Sequence[str],
    execution_closure: Mapping[str, str],
):
    """Stage the four-file model worker closure with immediate fd ownership."""

    root = _make_private_subdirectory(
        scratch,
        "model-source-bundle",
        parent_descriptor=scratch_descriptor,
        registry=scratch_registry,
    )
    root_entry = scratch_registry.entries[("model-source-bundle",)]
    root_descriptor = root_entry.descriptor
    directories = sorted({relative.split("/", 1)[0] for relative in required_files})
    for directory in directories:
        try:
            _make_private_subdirectory(
                root,
                directory,
                parent_descriptor=root_descriptor,
                registry=scratch_registry,
                registry_prefix=("model-source-bundle",),
            )
        except LiveKitCausalEndpointError:
            raise SourceBundleError() from None

    file_records: list[Mapping[str, object]] = []
    for relative in required_files:
        source = relative_sources.get(relative)
        if source is None or relative.count("/") != 1:
            raise SourceBundleError()
        parent_name, filename = relative.split("/", 1)
        parent_entry = scratch_registry.entries[
            ("model-source-bundle", parent_name)
        ]
        try:
            payload = _read_stable_execution_closure_member(source)
        except Exception:
            raise SourceBundleError() from None
        digest = hashlib.sha256(payload).hexdigest()
        if digest != execution_closure.get(relative):
            raise SourceBundleError()
        descriptor = -1
        try:
            descriptor, _created_identity = _create_filled_private_file_at(
                parent_entry.descriptor,
                filename,
                payload,
                maximum=1024 * 1024,
                mode=0o400,
            )
            info = os.stat(
                filename,
                dir_fd=parent_entry.descriptor,
                follow_symlinks=False,
            )
            _register_open_scratch_descriptor(
                scratch_registry,
                ("model-source-bundle", parent_name, filename),
                descriptor,
                is_directory=False,
                path_info=info,
            )
            descriptor = -1
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        file_records.append(
            {"path": relative, "sha256": digest, "size_bytes": len(payload)}
        )

    manifest_payload = _canonical_json_bytes(
        {
            "schema_version": 1,
            "worker": "tools/livekit_causal_endpoint_model_worker.py",
            "files": file_records,
        }
    )
    manifest_descriptor = -1
    try:
        manifest_descriptor, _created_identity = _create_filled_private_file_at(
            root_descriptor,
            "source-bundle.json",
            manifest_payload,
            maximum=128 * 1024,
            mode=0o400,
        )
        info = os.stat(
            "source-bundle.json",
            dir_fd=root_descriptor,
            follow_symlinks=False,
        )
        _register_open_scratch_descriptor(
            scratch_registry,
            ("model-source-bundle", "source-bundle.json"),
            manifest_descriptor,
            is_directory=False,
            path_info=info,
        )
        manifest_descriptor = -1
    finally:
        if manifest_descriptor >= 0:
            os.close(manifest_descriptor)
    tree_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    bundle = load_source_bundle(
        root / "source-bundle.json",
        expected_tree_sha256=tree_sha256,
        required_files=required_files,
    )
    verify_source_bundle(bundle, required_files=required_files)
    return bundle


def _run_model_stage(
    *,
    materialization: Materialization,
    model_snapshot: Path,
    policy: EndpointPolicyContract,
    execution_closure: Mapping[str, str],
    scratch: Path,
    scratch_descriptor: int,
    scratch_identity: tuple[int, ...],
    scratch_registry: ScratchRegistry,
) -> ModelStageResult:
    """Run all ONNX scoring/reduction in one bounded byte-bound subprocess."""

    expected_materialization_leaves = {
        "manifest.json",
        *(row.pcm_filename for row in materialization.rows),
    }
    if (
        set(materialization.leaf_identities) != expected_materialization_leaves
        or any(
            len(identity) != 6
            or any(isinstance(item, bool) or not isinstance(item, int) for item in identity)
            for identity in materialization.leaf_identities.values()
        )
    ):
        raise LiveKitCausalEndpointError("materialization ownership failed")
    relative_sources = {
        "tools/livekit_causal_endpoint_model_worker.py": _MODEL_WORKER_PATH,
        "core/endpointing.py": _ENDPOINTING_PATH,
        "always_on_agent/acoustic.py": _ACOUSTIC_PATH,
        "always_on_agent/text.py": _TEXT_PATH,
    }
    required_files = tuple(sorted(relative_sources))
    _assert_private_scratch_boundary(scratch, scratch_descriptor, scratch_identity)
    try:
        bundle = _stage_registered_model_source_bundle(
            relative_sources,
            scratch=scratch,
            scratch_descriptor=scratch_descriptor,
            scratch_registry=scratch_registry,
            required_files=required_files,
            execution_closure=execution_closure,
        )
        verify_source_bundle(bundle, required_files=required_files)
    except SourceBundleError:
        raise LiveKitCausalEndpointError("model-stage source bundle failed") from None
    _assert_private_scratch_boundary(scratch, scratch_descriptor, scratch_identity)
    for item in bundle.files:
        if execution_closure.get(item.relative_path) != item.sha256:
            raise LiveKitCausalEndpointError("execution closure changed")

    code_paths = {item.relative_path: bundle.root / item.relative_path for item in bundle.files}
    descriptors: list[int] = []
    process: subprocess.Popen[bytes] | None = None
    group_reaped = False
    result_payload = b""
    result_value: object = None
    try:
        code_bound: dict[str, tuple[int, tuple[int, ...], str]] = {}
        for relative in required_files:
            entry = scratch_registry.entries[
                ("model-source-bundle", *relative.split("/"))
            ]
            descriptor = os.dup(entry.descriptor)
            descriptors.append(descriptor)
            opened = os.fstat(descriptor)
            payload = _pread_descriptor(descriptor, opened.st_size)
            digest = hashlib.sha256(payload).hexdigest()
            path_info = code_paths[relative].lstat()
            if (
                entry.is_directory
                or not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or stat.S_IMODE(opened.st_mode) & 0o077
                or opened.st_size <= 0
                or opened.st_size > 1024 * 1024
                or _identity(opened) != _identity(path_info)
                or digest != execution_closure[relative]
            ):
                raise LiveKitCausalEndpointError("model-stage source bundle changed")
            identity = _identity(opened)
            code_bound[relative] = (descriptor, identity, digest)

        model_descriptor, model_identity, model_sha256 = _open_verified_regular(
            model_snapshot,
            maximum=_SMART_TURN_MODEL_BYTES,
            expected_sha256=_SMART_TURN_MODEL_SHA256,
            private=True,
        )
        descriptors.append(model_descriptor)
        venv_launch = _snapshot_model_venv_launch()
        python_argv0 = venv_launch.argv0
        python_path = venv_launch.executable_snapshot[1]
        python_descriptor, python_identity, python_sha256 = _open_verified_regular(
            python_path,
            maximum=64 * 1024 * 1024,
            private=False,
        )
        descriptors.append(python_descriptor)

        with _opened_materialization(materialization) as materialization_descriptor:
            materialization_identity = _directory_entry_identity(
                os.fstat(materialization_descriptor)
            )
            result_descriptor, result_identity = _create_empty_private_file_at(
                scratch_descriptor,
                "model-result.json",
            )
            descriptors.append(result_descriptor)
            _register_scratch_descriptor_copy(
                scratch_registry,
                ("model-result.json",),
                result_descriptor,
                is_directory=False,
                parent_descriptor=scratch_descriptor,
                name="model-result.json",
            )

            worker_descriptor = code_bound[
                "tools/livekit_causal_endpoint_model_worker.py"
            ][0]
            endpointing_descriptor = code_bound["core/endpointing.py"][0]
            acoustic_descriptor = code_bound["always_on_agent/acoustic.py"][0]
            text_descriptor = code_bound["always_on_agent/text.py"][0]
            request_value = {
                "schema_version": 1,
                "worker_descriptor": worker_descriptor,
                "endpointing_descriptor": endpointing_descriptor,
                "acoustic_descriptor": acoustic_descriptor,
                "text_descriptor": text_descriptor,
                "model_descriptor": model_descriptor,
                "materialization_descriptor": materialization_descriptor,
                "result_descriptor": result_descriptor,
                "code_sha256": {
                    "worker": execution_closure[
                        "tools/livekit_causal_endpoint_model_worker.py"
                    ],
                    "endpointing": execution_closure["core/endpointing.py"],
                    "acoustic": execution_closure["always_on_agent/acoustic.py"],
                    "text": execution_closure["always_on_agent/text.py"],
                },
                "config_sha256": policy.config_sha256,
                "model_size_bytes": _SMART_TURN_MODEL_BYTES,
                "model_sha256": _SMART_TURN_MODEL_SHA256,
                "materialization_manifest_sha256": materialization.manifest_sha256,
                "materialization_sha256": materialization.manifest[
                    "materialization_sha256"
                ],
                "pcm_set_sha256": materialization.manifest["pcm_set_sha256"],
                "materialization_directory_identity": list(materialization_identity),
                "materialization_leaf_identities": {
                    name: list(identity)
                    for name, identity in sorted(
                        materialization.leaf_identities.items()
                    )
                },
                "endpoint_config": dict(policy.endpoint_config),
                "acoustic_rule2_samples": policy.acoustic_rule2_samples,
                "prosody_min_samples": policy.prosody_min_samples,
                "max_wait_samples": policy.max_wait_samples,
                "rule3_samples": policy.rule3_samples,
                "expected_hold_denominator": _EXPECTED_HOLDS,
                "expected_eot_denominator": _EXPECTED_ROWS,
                "rows": _materialized_rows_payload(materialization.rows),
            }
            request_payload = _canonical_json_bytes(request_value)
            request_descriptor, _created_identity = _create_filled_private_file_at(
                scratch_descriptor,
                "model-request.json",
                request_payload,
                maximum=_MAX_PRIVATE_JSON_BYTES,
            )
            descriptors.append(request_descriptor)
            _register_scratch_descriptor_copy(
                scratch_registry,
                ("model-request.json",),
                request_descriptor,
                is_directory=False,
                parent_descriptor=scratch_descriptor,
                name="model-request.json",
            )
            request_path = scratch / "model-request.json"
            request_identity = _identity(os.fstat(request_descriptor))
            request_sha256 = hashlib.sha256(request_payload).hexdigest()

            scratch_fd_path = Path(f"/proc/self/fd/{scratch_descriptor}")
            environment = sanitized_worker_environment(scratch_fd_path)
            environment.update(
                {
                    "PATH": os.defpath,
                    "CUDA_VISIBLE_DEVICES": "",
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                }
            )
            _assert_private_scratch_boundary(
                scratch,
                scratch_descriptor,
                scratch_identity,
            )
            process = subprocess.Popen(
                [
                    str(python_argv0),
                    "-I",
                    "-B",
                    f"/proc/self/fd/{worker_descriptor}",
                    "--request-fd",
                    str(request_descriptor),
                ],
                executable=f"/proc/self/fd/{python_descriptor}",
                cwd=scratch_fd_path,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                pass_fds=(
                    python_descriptor,
                    worker_descriptor,
                    endpointing_descriptor,
                    acoustic_descriptor,
                    text_descriptor,
                    model_descriptor,
                    materialization_descriptor,
                    result_descriptor,
                    request_descriptor,
                    scratch_descriptor,
                ),
                start_new_session=True,
            )
            try:
                return_code = process.wait(timeout=_MODEL_WORKER_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                raise LiveKitCausalEndpointError("model stage timed out") from None
            finally:
                public_fixtures._terminate_worker_process_group(process)
                group_reaped = True
            if return_code != 0:
                raise LiveKitCausalEndpointError("model stage failed")

            _rebind_model_venv_launch(venv_launch)
            for relative, (descriptor, identity, digest) in code_bound.items():
                _rebind_verified_regular(
                    descriptor,
                    code_paths[relative],
                    identity,
                    digest,
                )
            _rebind_verified_regular(
                model_descriptor,
                model_snapshot,
                model_identity,
                model_sha256,
            )
            _rebind_verified_regular(
                python_descriptor,
                python_path,
                python_identity,
                python_sha256,
            )
            _rebind_verified_regular(
                request_descriptor,
                request_path,
                request_identity,
                request_sha256,
            )
            result_info = os.fstat(result_descriptor)
            if (
                _stable_file_identity(result_info) != result_identity
                or not 0 < result_info.st_size <= _MAX_REPORT_BYTES
            ):
                raise LiveKitCausalEndpointError("model-stage result contract failed")
            result_payload = _pread_descriptor(result_descriptor, result_info.st_size)
            result_value = _strict_json(result_payload)
        verify_source_bundle(bundle, required_files=required_files)
        _rebind_model_venv_launch(venv_launch)
    except LiveKitCausalEndpointError:
        raise
    except SourceBundleError:
        raise LiveKitCausalEndpointError("model-stage source bundle changed") from None
    except Exception:
        raise LiveKitCausalEndpointError("model stage failed") from None
    finally:
        if process is not None and not group_reaped:
            try:
                public_fixtures._terminate_worker_process_group(process)
            except Exception:
                pass
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass

    expected_contract = {
        "request_sha256": hashlib.sha256(request_payload).hexdigest(),
        "worker_sha256": execution_closure[
            "tools/livekit_causal_endpoint_model_worker.py"
        ],
        "endpointing_sha256": execution_closure["core/endpointing.py"],
        "acoustic_sha256": execution_closure["always_on_agent/acoustic.py"],
        "text_sha256": execution_closure["always_on_agent/text.py"],
        "model_sha256": _SMART_TURN_MODEL_SHA256,
        "config_sha256": policy.config_sha256,
        "materialization_manifest_sha256": materialization.manifest_sha256,
        "materialization_sha256": materialization.manifest["materialization_sha256"],
        "pcm_set_sha256": materialization.manifest["pcm_set_sha256"],
    }
    result_schema_version = (
        result_value.get("schema_version")
        if isinstance(result_value, dict)
        else None
    )
    if (
        not isinstance(result_value, dict)
        or set(result_value)
        != {
            "schema_version",
            "evaluation_contract",
            "candidate",
            "acoustic_no_partial_fallback",
            "runtime_receipt",
        }
        or isinstance(result_schema_version, bool)
        or not isinstance(result_schema_version, int)
        or result_schema_version != 1
        or result_value.get("evaluation_contract") != expected_contract
    ):
        raise LiveKitCausalEndpointError("model-stage result contract failed")
    runtime_receipt = result_value.get("runtime_receipt")
    if (
        not isinstance(runtime_receipt, dict)
        or set(runtime_receipt)
        != {"python_version", "numpy_version", "onnxruntime_version", "provider"}
        or runtime_receipt.get("provider") != "CPUExecutionProvider"
        or any(
            not isinstance(runtime_receipt.get(name), str)
            or _VERSION_RE.fullmatch(runtime_receipt[name]) is None
            for name in ("python_version", "numpy_version", "onnxruntime_version")
        )
    ):
        raise LiveKitCausalEndpointError("model-stage result contract failed")
    fully_before_holds, fully_before_eot = _fully_before_rule3_denominators(
        materialization.rows,
        policy.rule3_samples,
    )
    for key, partial_state in (
        ("candidate", PARTIAL_ASSUMPTION),
        ("acoustic_no_partial_fallback", NO_PARTIAL_PROFILE),
    ):
        profiles = result_value.get(key)
        if not isinstance(profiles, dict) or set(profiles) != {
            "publisher_labels_fully_before_rule3",
            "semantic_counterfactual_full_source",
        }:
            raise LiveKitCausalEndpointError("model-stage result contract failed")
        _validate_profile_result(
            profiles["publisher_labels_fully_before_rule3"],
            expected_partial_state=partial_state,
            expected_holds=fully_before_holds,
            expected_eot=fully_before_eot,
        )
        _validate_profile_result(
            profiles["semantic_counterfactual_full_source"],
            expected_partial_state=partial_state,
            expected_holds=_EXPECTED_HOLDS,
            expected_eot=_EXPECTED_ROWS,
        )
    return ModelStageResult(
        candidate=result_value["candidate"],
        acoustic_no_partial_fallback=result_value[
            "acoustic_no_partial_fallback"
        ],
        execution_receipt={
            "source_bundle_tree_sha256": bundle.tree_sha256,
            "python_executable_sha256": python_sha256,
            "lexical_venv_argv0_preserved": True,
            "venv_marker_sha256": hashlib.sha256(
                venv_launch.marker_payload
            ).hexdigest(),
            "wall_timeout_seconds": _MODEL_WORKER_TIMEOUT_SECONDS,
            "requested_cpu_soft_limit_ceiling_seconds": 1_800,
            "requested_address_space_soft_limit_ceiling_bytes": (
                4 * 1024 * 1024 * 1024
            ),
            "requested_file_size_soft_limit_ceiling_bytes": _MAX_REPORT_BYTES,
            "requested_file_descriptor_soft_limit_ceiling": 64,
            "inherited_hard_limits_may_reduce_ceilings": True,
            "worker_threads": 1,
            "network_namespace_isolation": False,
            "offline_environment": True,
            "cgroup_scope": False,
            "runtime_receipt": dict(runtime_receipt),
        },
    )


def _cleanup_child_tree(
    parent_descriptor: int,
    name: str,
    *,
    allowed: Callable[[tuple[str, ...], os.stat_result], bool],
    prefix: tuple[str, ...],
    registry: ScratchRegistry,
) -> None:
    try:
        before = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except OSError:
        raise LiveKitCausalEndpointError("scratch cleanup failed") from None
    current_prefix = (*prefix, name)
    entry = registry.entries.get(current_prefix)
    if (
        not allowed(current_prefix, before)
        or entry is None
        or not _scratch_entry_matches(entry, before)
    ):
        raise LiveKitCausalEndpointError("scratch cleanup failed")
    if stat.S_ISDIR(before.st_mode):
        if (
            stat.S_IMODE(before.st_mode) != 0o700
            or (hasattr(os, "geteuid") and before.st_uid != os.geteuid())
        ):
            raise LiveKitCausalEndpointError("scratch cleanup failed")
        if not entry.is_directory:
            raise LiveKitCausalEndpointError("scratch cleanup failed")
        descriptor = entry.descriptor
        try:
            stable = _directory_entry_identity(os.fstat(descriptor))
            if stable != _directory_entry_identity(before):
                raise LiveKitCausalEndpointError("scratch cleanup failed")
            for child in os.listdir(descriptor):
                _cleanup_child_tree(
                    descriptor,
                    child,
                    allowed=allowed,
                    prefix=current_prefix,
                    registry=registry,
                )
            after = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
            if (
                os.listdir(descriptor)
                or _directory_entry_identity(os.fstat(descriptor)) != stable
                or _directory_entry_identity(after) != stable
            ):
                raise LiveKitCausalEndpointError("scratch cleanup failed")
            os.rmdir(name, dir_fd=parent_descriptor)
            if os.fstat(descriptor).st_nlink != 0:
                raise LiveKitCausalEndpointError("scratch cleanup failed")
        except LiveKitCausalEndpointError:
            raise
        except OSError:
            raise LiveKitCausalEndpointError("scratch cleanup failed") from None
        return
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) & 0o077
        or (hasattr(os, "geteuid") and before.st_uid != os.geteuid())
    ):
        raise LiveKitCausalEndpointError("scratch cleanup failed")
    if entry.is_directory or not _scratch_entry_matches(entry, before):
        raise LiveKitCausalEndpointError("scratch cleanup failed")
    try:
        current = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        if not _scratch_entry_matches(entry, current):
            raise LiveKitCausalEndpointError("scratch cleanup failed")
        os.unlink(name, dir_fd=parent_descriptor)
        if os.fstat(entry.descriptor).st_nlink != 0:
            raise LiveKitCausalEndpointError("scratch cleanup failed")
    except LiveKitCausalEndpointError:
        raise
    except OSError:
        raise LiveKitCausalEndpointError("scratch cleanup failed") from None


def _scratch_entry_allowed(path: tuple[str, ...], info: os.stat_result) -> bool:
    top = {
        _SMART_TURN_MODEL_FILENAME,
        "worker",
        "materialized",
        "model-source-bundle",
        "model-result.json",
        "model-request.json",
    }
    if len(path) == 1:
        return path[0] in top
    root = path[0]
    if root == "worker":
        return len(path) == 2 and path[1] in {
            "livekit_causal_endpoint_worker.py",
            "livekit_eot_parquet_worker.py",
            "request.json",
        }
    if root == "materialized":
        match = re.fullmatch(r"row-([0-9]{4})\.pcm", path[1]) if len(path) == 2 else None
        return len(path) == 2 and (
            path[1] == "manifest.json"
            or (match is not None and int(match.group(1)) < _EXPECTED_ROWS)
        )
    if root != "model-source-bundle":
        return False
    allowed_bundle = {
        ("model-source-bundle", "source-bundle.json"),
        ("model-source-bundle", "tools"),
        ("model-source-bundle", "tools", "livekit_causal_endpoint_model_worker.py"),
        ("model-source-bundle", "core"),
        ("model-source-bundle", "core", "endpointing.py"),
        ("model-source-bundle", "always_on_agent"),
        ("model-source-bundle", "always_on_agent", "acoustic.py"),
        ("model-source-bundle", "always_on_agent", "text.py"),
    }
    return path in allowed_bundle


def _validate_registered_scratch_tree(
    parent_descriptor: int,
    *,
    prefix: tuple[str, ...],
    registry: ScratchRegistry,
    seen: set[tuple[str, ...]],
) -> None:
    try:
        children = os.listdir(parent_descriptor)
    except OSError:
        raise LiveKitCausalEndpointError("scratch cleanup failed") from None
    for name in children:
        relative = (*prefix, name)
        try:
            info = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        except OSError:
            raise LiveKitCausalEndpointError("scratch cleanup failed") from None
        entry = registry.entries.get(relative)
        if (
            not _scratch_entry_allowed(relative, info)
            or entry is None
            or not _scratch_entry_matches(entry, info)
        ):
            raise LiveKitCausalEndpointError("scratch cleanup failed")
        seen.add(relative)
        if stat.S_ISDIR(info.st_mode):
            if (
                stat.S_IMODE(info.st_mode) != 0o700
                or (hasattr(os, "geteuid") and info.st_uid != os.geteuid())
            ):
                raise LiveKitCausalEndpointError("scratch cleanup failed")
            if not entry.is_directory:
                raise LiveKitCausalEndpointError("scratch cleanup failed") from None
            descriptor = entry.descriptor
            try:
                if not _scratch_entry_matches(entry, info):
                    raise LiveKitCausalEndpointError("scratch cleanup failed")
                _validate_registered_scratch_tree(
                    descriptor,
                    prefix=relative,
                    registry=registry,
                    seen=seen,
                )
                current = os.stat(
                    name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
                if not _scratch_entry_matches(entry, current):
                    raise LiveKitCausalEndpointError("scratch cleanup failed")
            except OSError:
                raise LiveKitCausalEndpointError("scratch cleanup failed") from None
        elif (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) & 0o077
            or (hasattr(os, "geteuid") and info.st_uid != os.geteuid())
        ):
            raise LiveKitCausalEndpointError("scratch cleanup failed")
        elif entry.is_directory:
            raise LiveKitCausalEndpointError("scratch cleanup failed")


def _cleanup_private_scratch(
    path: Path,
    descriptor: int,
    identity: tuple[int, ...],
    registry: ScratchRegistry,
) -> None:
    _assert_private_scratch_boundary(path, descriptor, identity)
    try:
        names = set(os.listdir(descriptor))
    except OSError:
        raise LiveKitCausalEndpointError("scratch cleanup failed") from None
    if not names <= {
        _SMART_TURN_MODEL_FILENAME,
        "worker",
        "materialized",
        "model-source-bundle",
        "model-result.json",
        "model-request.json",
    }:
        raise LiveKitCausalEndpointError("scratch cleanup failed")
    seen: set[tuple[str, ...]] = set()
    _validate_registered_scratch_tree(
        descriptor,
        prefix=(),
        registry=registry,
        seen=seen,
    )
    if seen != set(registry.entries):
        raise LiveKitCausalEndpointError("scratch cleanup failed")
    for name in sorted(names):
        _cleanup_child_tree(
            descriptor,
            name,
            allowed=_scratch_entry_allowed,
            prefix=(),
            registry=registry,
        )
    try:
        if os.listdir(descriptor):
            raise LiveKitCausalEndpointError("scratch cleanup failed")
        parent = path.parent
        parent_info = parent.lstat()
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        parent_descriptor = os.open(parent, flags)
        try:
            if (
                _directory_entry_identity(os.fstat(parent_descriptor))
                != _directory_entry_identity(parent_info)
                or _directory_entry_identity(
                    os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
                )
                != identity
                or _directory_entry_identity(os.fstat(descriptor)) != identity
            ):
                raise LiveKitCausalEndpointError("scratch cleanup failed")
            os.rmdir(path.name, dir_fd=parent_descriptor)
            os.fsync(parent_descriptor)
            if os.fstat(descriptor).st_nlink != 0:
                raise LiveKitCausalEndpointError("scratch cleanup failed")
        finally:
            os.close(parent_descriptor)
    except LiveKitCausalEndpointError:
        raise
    except OSError:
        raise LiveKitCausalEndpointError("scratch cleanup failed") from None


def _publisher_rule3_label_scope(
    rows: Sequence[MaterializedRow],
    rule3_samples: int,
) -> Mapping[str, object]:
    crossing_rows: set[int] = set()
    at_or_beyond_rows: set[int] = set()
    crossing_holds = 0
    crossing_eot = 0
    at_or_beyond_holds = 0
    at_or_beyond_eot = 0
    for row in rows:
        for index, span in enumerate(row.silence_spans):
            is_eot = index == len(row.silence_spans) - 1
            if span.start_sample < rule3_samples <= span.end_sample:
                crossing_rows.add(row.ordinal)
                if is_eot:
                    crossing_eot += 1
                else:
                    crossing_holds += 1
            elif span.start_sample >= rule3_samples:
                at_or_beyond_rows.add(row.ordinal)
                if is_eot:
                    at_or_beyond_eot += 1
                else:
                    at_or_beyond_holds += 1
    fully_before_holds, fully_before_eot = _fully_before_rule3_denominators(
        rows,
        rule3_samples,
    )
    return {
        "rule3_samples": rule3_samples,
        "subset_semantics": (
            "publisher labels whose silence end is strictly before rule3; "
            "not production or tick-level rule3 execution"
        ),
        "crossing_rule3": {
            "row_count": len(crossing_rows),
            "hold_label_count": crossing_holds,
            "eot_label_count": crossing_eot,
        },
        "starting_at_or_beyond_rule3": {
            "row_count": len(at_or_beyond_rows),
            "hold_label_count": at_or_beyond_holds,
            "eot_label_count": at_or_beyond_eot,
        },
        "publisher_labels_fully_before_rule3": {
            "hold_denominator": fully_before_holds,
            "eot_denominator": fully_before_eot,
        },
        "crossing_labels_may_contain_earlier_decision_ticks": True,
    }


def _rebind_config_source(config_path: Path | str, expected_sha256: str) -> None:
    current = _load_policy_contract(config_path)
    if current.config_sha256 != expected_sha256:
        raise LiveKitCausalEndpointError("config changed")


def _rebind_source_path(source_path: Path | str) -> None:
    try:
        with inventory_source._opened_source(source_path) as source:
            inventory_source._verify_source(source, hash_content=True)
    except inventory_source.LiveKitEotInventoryError:
        raise LiveKitCausalEndpointError("source changed") from None


def _publish_report(
    destination: Path,
    parent_info: os.stat_result,
    payload: bytes,
    *,
    rebind: Callable[[], None],
) -> str:
    if not payload or len(payload) > _MAX_REPORT_BYTES:
        raise LiveKitCausalEndpointError("report contract failed")
    parent_descriptor = -1
    temporary_descriptor = -1
    temporary_name: Optional[str] = None
    temporary_identity: tuple[int, ...] | None = None
    published_created = False
    published_succeeded = False
    published_identity: tuple[int, ...] | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        parent_descriptor = os.open(destination.parent, flags)
        if (
            _directory_entry_identity(os.fstat(parent_descriptor))
            != _directory_entry_identity(parent_info)
            or inventory_source._has_git_ancestor(destination.parent)
        ):
            raise LiveKitCausalEndpointError("report publication failed")
        try:
            os.stat(destination.name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise LiveKitCausalEndpointError("report publication failed")
        temporary_name = f".{destination.name}.{os.urandom(12).hex()}.part"
        temporary_descriptor, temporary_identity = _create_filled_private_file_at(
            parent_descriptor,
            temporary_name,
            payload,
            maximum=_MAX_REPORT_BYTES,
        )
        info = os.fstat(temporary_descriptor)
        temporary_payload = _pread_descriptor(temporary_descriptor, len(payload))
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_size != len(payload)
            or temporary_payload != payload
            or (hasattr(os, "geteuid") and info.st_uid != os.geteuid())
            or _inode_identity(info) != temporary_identity
        ):
            raise LiveKitCausalEndpointError("report publication failed")
        rebind()
        try:
            current_parent = destination.parent.lstat()
            os.stat(
                destination.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        except OSError:
            raise LiveKitCausalEndpointError("report publication failed") from None
        else:
            raise LiveKitCausalEndpointError("report publication failed")
        if (
            _directory_entry_identity(os.fstat(parent_descriptor))
            != _directory_entry_identity(parent_info)
            or _directory_entry_identity(current_parent)
            != _directory_entry_identity(parent_info)
            or inventory_source._has_git_ancestor(destination.parent)
        ):
            raise LiveKitCausalEndpointError("report publication failed")
        current_temporary = os.stat(
            temporary_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if _inode_identity(current_temporary) != temporary_identity:
            raise LiveKitCausalEndpointError("report publication failed")
        # Link the object held by the retained descriptor, not the mutable
        # temporary directory entry.  AT_SYMLINK_FOLLOW on Linux procfs
        # resolves this magic link to that exact open inode; if an attacker
        # unlinks the inode first, the link fails instead of publishing a
        # replacement installed at ``temporary_name``.
        published_identity = temporary_identity
        published_created = True
        os.link(
            f"/proc/self/fd/{temporary_descriptor}",
            destination.name,
            dst_dir_fd=parent_descriptor,
            follow_symlinks=True,
        )
        linked = os.stat(
            destination.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if _inode_identity(linked) != published_identity:
            raise LiveKitCausalEndpointError("report publication failed")
        current_temporary = os.stat(
            temporary_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if _inode_identity(current_temporary) != temporary_identity:
            raise LiveKitCausalEndpointError("report publication failed")
        os.unlink(temporary_name, dir_fd=parent_descriptor)
        temporary_name = None
        if os.fstat(temporary_descriptor).st_nlink != 1:
            raise LiveKitCausalEndpointError("report publication failed")
        os.fsync(parent_descriptor)
        published = os.fstat(temporary_descriptor)
        published_payload = _pread_descriptor(temporary_descriptor, len(payload))
        current = os.stat(
            destination.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            _identity(published) != _identity(current)
            or not stat.S_ISREG(published.st_mode)
            or published.st_nlink != 1
            or stat.S_IMODE(published.st_mode) != 0o600
            or published_payload != payload
            or (hasattr(os, "geteuid") and published.st_uid != os.geteuid())
        ):
            raise LiveKitCausalEndpointError("report publication failed")
        rebind()
        published_succeeded = True
        return hashlib.sha256(payload).hexdigest()
    except LiveKitCausalEndpointError:
        raise
    except (OSError, ValueError, inventory_source.LiveKitEotInventoryError):
        raise LiveKitCausalEndpointError("report publication failed") from None
    finally:
        cleanup_failed = False
        if parent_descriptor >= 0:
            if (
                published_created
                and not published_succeeded
                and published_identity is not None
            ):
                try:
                    current = os.stat(
                        destination.name,
                        dir_fd=parent_descriptor,
                        follow_symlinks=False,
                    )
                    if _inode_identity(current) == published_identity:
                        os.unlink(destination.name, dir_fd=parent_descriptor)
                        os.fsync(parent_descriptor)
                except OSError:
                    cleanup_failed = True
            if temporary_name is not None:
                try:
                    current = os.stat(
                        temporary_name,
                        dir_fd=parent_descriptor,
                        follow_symlinks=False,
                    )
                    if (
                        temporary_identity is not None
                        and _inode_identity(current) == temporary_identity
                    ):
                        os.unlink(temporary_name, dir_fd=parent_descriptor)
                except OSError:
                    cleanup_failed = True
            if temporary_descriptor >= 0 and not published_succeeded:
                try:
                    if os.fstat(temporary_descriptor).st_nlink != 0:
                        cleanup_failed = True
                    os.fsync(parent_descriptor)
                except OSError:
                    cleanup_failed = True
            if temporary_descriptor >= 0:
                os.close(temporary_descriptor)
            os.close(parent_descriptor)
        if cleanup_failed:
            raise LiveKitCausalEndpointError("report publication failed")


def evaluate_livekit_causal_endpoint(
    *,
    source_parquet: Path | str,
    inventory_report: Path | str,
    parquet_python: Path | str,
    model: Path | str,
    config: Path | str,
    scratch_root: Path | str,
    output: Path | str,
    accepted_terms: frozenset[str],
    accept_partial_assumption: bool,
) -> EvaluationResult:
    """Run the exact private causal evaluator and publish one aggregate report."""

    if not isinstance(accepted_terms, frozenset) or LICENSE not in accepted_terms:
        raise LiveKitCausalEndpointError("license acknowledgement failed")
    if accept_partial_assumption is not True:
        raise LiveKitCausalEndpointError("partial availability assumption not accepted")
    execution_closure = _snapshot_execution_closure()
    _validate_inventory_receipt(inventory_report)
    policy = _load_policy_contract(config)
    try:
        destination, parent_info = inventory_source._output_path(output)
    except inventory_source.LiveKitEotInventoryError:
        raise LiveKitCausalEndpointError("report prerequisite failed") from None
    _require_scratch_registry_capacity()
    (
        scratch,
        scratch_descriptor,
        scratch_identity,
        scratch_registry,
    ) = _create_private_scratch(scratch_root)
    try:
        if (
            destination == scratch
            or scratch in destination.parents
            or destination in scratch.parents
        ):
            raise LiveKitCausalEndpointError("private output paths overlap")
        _assert_private_scratch_boundary(
            scratch,
            scratch_descriptor,
            scratch_identity,
        )
        model_snapshot, model_binding = _snapshot_exact_model(
            model,
            scratch,
            scratch_descriptor=scratch_descriptor,
            scratch_registry=scratch_registry,
        )
        with inventory_source._opened_source(source_parquet) as source:
            try:
                inventory_source._verify_source(source, hash_content=True)
                materializer_stage = _materialize_source(
                    source,
                    scratch=scratch,
                    scratch_descriptor=scratch_descriptor,
                    scratch_identity=scratch_identity,
                    parquet_python=parquet_python,
                    execution_closure=execution_closure,
                    scratch_registry=scratch_registry,
                )
                inventory_source._verify_source(source, hash_content=True)
            except inventory_source.LiveKitEotInventoryError:
                raise LiveKitCausalEndpointError("source verification failed") from None
        materialization = _load_materialization(
            materializer_stage.output,
            execution_closure,
            retained_directory_descriptor=scratch_registry.entries[
                ("materialized",)
            ].descriptor,
            retained_leaf_entries={
                relative[-1]: entry
                for relative, entry in scratch_registry.entries.items()
                if len(relative) == 2 and relative[0] == "materialized"
            },
        )
        stage = _run_model_stage(
            materialization=materialization,
            model_snapshot=model_snapshot,
            policy=policy,
            execution_closure=execution_closure,
            scratch=scratch,
            scratch_descriptor=scratch_descriptor,
            scratch_identity=scratch_identity,
            scratch_registry=scratch_registry,
        )
        _rebind_materialization(materialization)
        _rebind_execution_closure(execution_closure)
        materialization_binding = {
            "schema_version": MATERIALIZATION_SCHEMA_VERSION,
            "kind": MATERIALIZATION_KIND,
            "manifest_sha256": materialization.manifest_sha256,
            "materialization_sha256": materialization.manifest[
                "materialization_sha256"
            ],
            "pcm_set_sha256": materialization.manifest["pcm_set_sha256"],
            "row_count": materialization.manifest["row_count"],
            "silence_span_count": materialization.manifest["silence_span_count"],
            "hold_label_count": materialization.manifest["hold_label_count"],
            "eot_label_count": materialization.manifest["eot_label_count"],
            "off_grid_span_count": materialization.manifest["off_grid_span_count"],
            "final_gap_zero_sample_count": materialization.manifest[
                "final_gap_zero_sample_count"
            ],
            "final_gap_one_sample_count": materialization.manifest[
                "final_gap_one_sample_count"
            ],
        }
        rule3_scope = _publisher_rule3_label_scope(
            materialization.rows,
            policy.rule3_samples,
        )
        candidate = stage.candidate
        no_partial = stage.acoustic_no_partial_fallback
        model_stage_receipt = stage.execution_receipt
        materializer_stage_receipt = materializer_stage.execution_receipt
    finally:
        if scratch_descriptor >= 0:
            try:
                _cleanup_private_scratch(
                    scratch,
                    scratch_descriptor,
                    scratch_identity,
                    scratch_registry,
                )
            finally:
                scratch_registry.close()
                os.close(scratch_descriptor)

    policy_snapshot = dict(policy.endpoint_config)
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "dataset": DATASET,
        "config": DATASET_CONFIG,
        "split": DATASET_SPLIT,
        "license": LICENSE,
        "attribution": {
            "creator": "LiveKit",
            "source_url": "https://huggingface.co/datasets/livekit/eot-bench-data",
            "license_url": "https://creativecommons.org/licenses/by/4.0/legalcode",
        },
        "evaluation_scope": "endpoint-early-cut-and-policy-delay-only-no-wer",
        "source_binding": {
            "revision": inventory_worker.SOURCE_REVISION,
            "size_bytes": inventory_worker.SOURCE_SIZE_BYTES,
            "sha256": inventory_worker.SOURCE_SHA256,
            "schema_contract_sha256": inventory_worker.SCHEMA_CONTRACT_SHA256,
            "inventory_report_sha256": INVENTORY_REPORT_SHA256,
            "inventory_sha256": INVENTORY_SHA256,
            "audio_set_sha256": AUDIO_SET_SHA256,
            "silence_set_sha256": SILENCE_SET_SHA256,
        },
        "model_binding": dict(model_binding),
        "runtime_dependencies": {
            "pyarrow": inventory_worker.PYARROW_VERSION,
            "onnxruntime": model_stage_receipt["runtime_receipt"][
                "onnxruntime_version"
            ],
            "numpy": model_stage_receipt["runtime_receipt"]["numpy_version"],
            "python": model_stage_receipt["runtime_receipt"]["python_version"],
            "provider": model_stage_receipt["runtime_receipt"]["provider"],
            "scope": "recorded-runtime-versions-not-byte-exact-runtime-tree-receipts",
        },
        "config_binding": {
            "sha256": policy.config_sha256,
            "runtime_default_detector": policy.runtime_default_detector,
            "evaluation_detector": "exact-opt-in-smart-turn-v3.2-cpu",
            "endpoint_policy": policy_snapshot,
            "acoustic_rule2_samples": policy.acoustic_rule2_samples,
            "prosody_min_samples": policy.prosody_min_samples,
            "max_wait_samples": policy.max_wait_samples,
            "rule3_samples": policy.rule3_samples,
        },
        "execution_binding": {
            "supervisor_source_state_sha256": dict(execution_closure),
            "model_stage": dict(model_stage_receipt),
            "materializer_stage": dict(materializer_stage_receipt),
            "parent_scratch_registry_minimum_spare_file_descriptors": (
                _SCRATCH_REGISTRY_MINIMUM_SPARE_FDS
            ),
            "model_stage_code_and_model_consumed_via_verified_descriptors": True,
            "materializer_code_and_source_consumed_via_verified_descriptors": True,
            "sherpa_integration_evidence": "source-state-plus-headless-differential-parity",
        },
        "materialization_binding": materialization_binding,
        "protocol": {
            "label_convention": LABEL_CONVENTION,
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "decision_grid_samples": GRID_SAMPLES,
            "decision_grid_origin": "span_start",
            "eot_off_grid_publisher_labelled_silence_end_included": True,
            "hold_resume_boundary_excluded": True,
            "hold_cutoff_definition": "first_commit_delay_strictly_less_than_hold_length",
            "eot_delay_definition": (
                "first_publisher_bounded_causal_prefix_commit_minus_"
                "final_silence_start"
            ),
            "eot_censoring": (
                "stop_model_scoring_at_publisher_labelled_final_silence_end; "
                "report labelled-silence-duration lower bound and conservative "
                "production-max-wait upper substitution"
            ),
            "publisher_labelled_final_silence_to_audio_residual_samples": {
                "zero_sample_row_count": 394,
                "one_sample_row_count": 6,
                "unlabelled_residual_is_never_scored": True,
            },
            "partial_assumption": PARTIAL_ASSUMPTION,
            "partial_assumption_explicitly_accepted": True,
            "partial_text_value_source": "fixed-opaque-sentinel-not-dataset-text",
            "acoustic_endpoint_input": "rule2-sample-threshold-surrogate-not-recognizer-vad-execution",
            "vad_quiet_assumed": True,
            "early_endpoint_allowed_assumed": True,
            "adaptive_state": "fresh-per-row; only prior publisher-HOLD pauses after oracle resume",
            "later_spans_after_predicted_cut": "counterfactual-independent-benchmark-branches",
            "source_columns_scanned": ["audio", "language", "duration", "silence_spans"],
            "forbidden_value_columns": ["id", "words", "messages"],
            "audio_path_label": "validated-bounded-flat-label-never-opened-or-published",
        },
        "candidate": candidate,
        "acoustic_no_partial_fallback": no_partial,
        "publisher_rule3_label_scope": rule3_scope,
        "limits": [
            "conditional on a nonempty same-epoch streaming partial at every scored span",
            "raw dataset PCM is not the production post-front-end capture domain",
            "does not execute Sherpa recognizer, VAD, rule3, capture, recovery, or device paths",
            "policy delay is not model inference time, wall latency, or live latency",
            "publisher silences are endpoint labels; words and messages are not STT truth",
            "six rows retain one unlabelled audio sample after the final-silence label; that residual is not scored",
            "no microphone, AEC, barge-in, owner voice, enrollment, tools, LLM, TTS, or live A/B",
            "owner recordings and bare-speaker live validation remain unrun",
            "worker network namespace isolation was not available; offline environment only",
            "workers request RLIMIT soft ceilings but do not return applied values; no cgroup scope or receipt exists",
        ],
        "quality_verdict": "diagnostic_only",
        "evidence_class": "real-source-offline-endpoint-counterfactual",
        "production_runtime_evidence": False,
    }
    payload = _canonical_json_bytes(report)

    def rebind() -> None:
        _rebind_execution_closure(execution_closure)
        _validate_inventory_receipt(inventory_report)
        _rebind_config_source(config, policy.config_sha256)
        _rebind_source_path(source_parquet)

    report_sha256 = _publish_report(
        destination,
        parent_info,
        payload,
        rebind=rebind,
    )
    return EvaluationResult(
        report_path=destination,
        report_sha256=report_sha256,
        hold_denominator=int(
            candidate["semantic_counterfactual_full_source"]["hold"]["denominator"]
        ),
        eot_denominator=int(
            candidate["semantic_counterfactual_full_source"]["eot"]["denominator"]
        ),
        evidence_class="real-source-offline-endpoint-counterfactual",
        production_runtime_evidence=False,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the exact private causal LiveKit endpoint diagnostic."
    )
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--inventory-report", type=Path, required=True)
    parser.add_argument("--parquet-python", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accept-license", action="append", default=[])
    parser.add_argument(
        "--accept-partial-assumption",
        action="store_true",
        help=(
            "explicitly accept that every scored span is conditional on an "
            "available nonempty same-epoch streaming partial"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        os.umask(0o077)
        args = _parser().parse_args(argv)
        result = evaluate_livekit_causal_endpoint(
            source_parquet=args.source_parquet,
            inventory_report=args.inventory_report,
            parquet_python=args.parquet_python,
            model=args.model,
            config=args.config,
            scratch_root=args.scratch_root,
            output=args.output,
            accepted_terms=frozenset(args.accept_license),
            accept_partial_assumption=args.accept_partial_assumption,
        )
        print(
            json.dumps(
                {
                    "ok": True,
                    "report_sha256": result.report_sha256,
                    "hold_denominator": result.hold_denominator,
                    "eot_denominator": result.eot_denominator,
                    "evidence_class": result.evidence_class,
                    "production_runtime_evidence": (
                        result.production_runtime_evidence
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    except Exception:  # noqa: BLE001 - never expose private paths/content
        print(json.dumps(_SAFE_ERROR, sort_keys=True, separators=(",", ":")))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
