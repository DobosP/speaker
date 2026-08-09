"""Private publication contract for the fixed guided STT capture plan."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Callable, Mapping

from core.guided_stt_plan import (
    GUIDED_STT_CAPTURE_PROTOCOL,
    built_in_guided_stt_plan,
    canonical_guided_stt_plan_bytes,
    guided_stt_route_availability_sha256,
    guided_stt_summary_contract_sha256,
)


_SAFE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PLAN_FILENAME = "reference-plan.json"
_CONTRACT_FILENAME = "capture-contract.json"
_RECEIPT_FILENAME = "capture-bundle-receipt.json"
_CONTRACT_SCHEMA_VERSION = 1
_RECEIPT_SCHEMA_VERSION = 1
_RECEIPT_KIND = "guided-stt-capture-bundle-v1"
_MAX_CONTRACT_BYTES = 64 << 10
_MAX_PLAN_BYTES = 64 << 10
_MAX_RECEIPT_BYTES = 64 << 10
_MAX_SUMMARY_BYTES = 4 << 20
_MAX_MANIFEST_BYTES = 1 << 20
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SUMMARY_PATH_KEYS = (
    "recording",
    "pre_dsp_reference",
    "playback_reference",
    "asr_input_reference",
    "final_model_input",
    "diagnostic_timeline",
)
_SIDE_EFFECT_POLICY = {
    "capabilities": False,
    "control_commands": False,
    "keyword_spotter": False,
    "llm": False,
    "memory": False,
    "output_device": False,
    "owner_authority": False,
    "playback": False,
    "speaker_verification": False,
    "tts": False,
}


class GuidedCapturePublicationError(RuntimeError):
    """A detail-free private publication or binding failure."""

    def __init__(self) -> None:
        super().__init__("guided capture prerequisites unavailable")


@dataclass(frozen=True, slots=True)
class _PublishedFile:
    path: Path = field(repr=False)
    sha256: str
    inode: tuple[int, int] = field(repr=False)
    identity: tuple[int, ...] = field(repr=False)
    binding_identity: tuple[int, ...] = field(repr=False)
    parent_identity: tuple[int, ...] = field(repr=False)
    size: int


@dataclass(frozen=True, slots=True)
class GuidedCapturePublication:
    plan_path: Path = field(repr=False)
    contract_path: Path = field(repr=False)
    plan_sha256: str
    contract_sha256: str
    case_count: int
    _run_dir_identity: tuple[int, ...] = field(repr=False)
    _plan_identity: tuple[int, ...] = field(repr=False)
    _contract_identity: tuple[int, ...] = field(repr=False)
    _plan_bytes: bytes = field(repr=False)
    _contract_bytes: bytes = field(repr=False)


@dataclass(frozen=True, slots=True)
class GuidedCaptureBundlePublication:
    receipt_path: Path = field(repr=False)
    receipt_sha256: str
    summary_contract_sha256: str
    _receipt_identity: tuple[int, ...] = field(repr=False)
    _receipt_bytes: bytes = field(repr=False)
    verified_bundle: VerifiedGuidedCaptureBundle = field(repr=False)


@dataclass(frozen=True, slots=True)
class VerifiedGuidedCaptureBundle:
    run_dir: Path = field(repr=False)
    _run_dir_identity: tuple[int, ...] = field(repr=False)
    receipt_path: Path = field(repr=False)
    receipt_sha256: str
    _receipt_identity: tuple[int, ...] = field(repr=False)
    plan_path: Path = field(repr=False)
    plan_sha256: str
    _plan_identity: tuple[int, ...] = field(repr=False)
    contract_path: Path = field(repr=False)
    contract_sha256: str
    _contract_identity: tuple[int, ...] = field(repr=False)
    summary_path: Path = field(repr=False)
    summary_sha256: str
    _summary_identity: tuple[int, ...] = field(repr=False)
    summary_contract_sha256: str
    diagnostic_manifest_path: Path = field(repr=False)
    diagnostic_manifest_sha256: str
    _diagnostic_manifest_identity: tuple[int, ...] = field(repr=False)
    final_stt_profile: str
    final_stt_profile_sha256: str
    final_stt_profile_schema_version: int
    capture_config_sha256: str
    effective_sherpa_sha256: str
    device_profile: str
    effective_input_gain: float
    case_count: int
    case_order_sha256: str


@dataclass(frozen=True, slots=True)
class _PrivateSnapshot:
    path: Path = field(repr=False)
    raw: bytes = field(repr=False)
    sha256: str
    inode: tuple[int, int] = field(repr=False)
    identity: tuple[int, ...] = field(repr=False)
    binding_identity: tuple[int, ...] = field(repr=False)
    size: int


def _canonical_json(payload: object) -> bytes:
    try:
        return (
            json.dumps(
                payload,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
    except (MemoryError, TypeError, UnicodeError, ValueError, OverflowError):
        raise GuidedCapturePublicationError() from None


def _strict_json(raw: bytes) -> object:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise GuidedCapturePublicationError()
            result[key] = value
        return result

    def reject_constant(_value: str) -> object:
        raise GuidedCapturePublicationError()

    try:
        return json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except GuidedCapturePublicationError:
        raise
    except (UnicodeError, ValueError, OverflowError):
        raise GuidedCapturePublicationError() from None


def _file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        stat.S_IMODE(metadata.st_mode),
        metadata.st_uid,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _binding_file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    """Identity stable across the terminal hard-link commit.

    Linking an anonymous receipt changes only link count and ctime.  The retained
    identity still binds the inode, type, permissions, owner, size, and content
    mtime, while every live snapshot separately requires one link.
    """

    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        stat.S_IMODE(metadata.st_mode),
        metadata.st_uid,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


def _directory_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        stat.S_IMODE(metadata.st_mode),
        metadata.st_uid,
    )


def _snapshot_private_file(path: Path, *, maximum_bytes: int) -> _PrivateSnapshot:
    parent_fd = descriptor = -1
    try:
        if (
            not isinstance(path, Path)
            or not path.is_absolute()
            or path.name in {"", ".", ".."}
            or type(maximum_bytes) is not int
            or maximum_bytes <= 0
        ):
            raise GuidedCapturePublicationError()
        parent_fd, parent_before = _directory_fd(path.parent)
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path.name, flags, dir_fd=parent_fd)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_uid != os.getuid()
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > maximum_bytes
        ):
            raise GuidedCapturePublicationError()
        raw = bytearray()
        while len(raw) <= maximum_bytes:
            chunk = os.read(
                descriptor,
                min(65536, maximum_bytes + 1 - len(raw)),
            )
            if not chunk:
                break
            raw.extend(chunk)
        after = os.fstat(descriptor)
        path_after = os.stat(
            path.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        parent_after = os.fstat(parent_fd)
        identity = _file_identity(before)
        if (
            len(raw) != before.st_size
            or identity != _file_identity(after)
            or identity != _file_identity(path_after)
            or parent_before.st_dev != parent_after.st_dev
            or parent_before.st_ino != parent_after.st_ino
            or parent_before.st_mode != parent_after.st_mode
        ):
            raise GuidedCapturePublicationError()
        payload = bytes(raw)
        return _PrivateSnapshot(
            path=path,
            raw=payload,
            sha256=hashlib.sha256(payload).hexdigest(),
            inode=(before.st_dev, before.st_ino),
            identity=identity,
            binding_identity=_binding_file_identity(before),
            size=before.st_size,
        )
    except GuidedCapturePublicationError:
        raise
    except (OSError, TypeError, ValueError):
        raise GuidedCapturePublicationError() from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if parent_fd >= 0:
            try:
                os.close(parent_fd)
            except OSError:
                pass


def _snapshot_matches(first: _PrivateSnapshot, second: _PrivateSnapshot) -> bool:
    return bool(
        first.path == second.path
        and first.raw == second.raw
        and first.sha256 == second.sha256
        and first.identity == second.identity
        and first.size == second.size
    )


def _sha256(value: object) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise GuidedCapturePublicationError()
    return value


def _safe_file(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or Path(value).name != value
        or value in {".", ".."}
    ):
        raise GuidedCapturePublicationError()
    return value


def _directory_fd(path: Path) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or metadata.st_uid != os.getuid()
    ):
        os.close(descriptor)
        raise GuidedCapturePublicationError()
    return descriptor, metadata


def _publish_at(
    parent: Path,
    name: str,
    payload: bytes,
    *,
    expected_parent_identity: tuple[int, ...] | None = None,
) -> _PublishedFile:
    parent_fd = descriptor = -1
    try:
        parent_fd, descriptor, staged = _stage_terminal_file(
            parent,
            name,
            payload,
            expected_parent_identity=expected_parent_identity,
        )
        parent_identity = _directory_identity(os.fstat(parent_fd))
        _link_staged_terminal(parent_fd, descriptor, name)
        os.fsync(parent_fd)
        published = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        parent_after = os.fstat(parent_fd)
        if (
            _directory_identity(parent_after) != parent_identity
            or not stat.S_ISREG(published.st_mode)
            or stat.S_IMODE(published.st_mode) != 0o600
            or published.st_uid != os.getuid()
            or published.st_nlink != 1
            or (published.st_dev, published.st_ino) != staged.inode
            or _binding_file_identity(published) != staged.binding_identity
            or published.st_size != len(payload)
        ):
            raise GuidedCapturePublicationError()
        return _PublishedFile(
            path=parent / name,
            sha256=hashlib.sha256(payload).hexdigest(),
            inode=(published.st_dev, published.st_ino),
            identity=_file_identity(published),
            binding_identity=_binding_file_identity(published),
            parent_identity=_directory_identity(parent_after),
            size=published.st_size,
        )
    except GuidedCapturePublicationError:
        raise
    except (OSError, TypeError, ValueError):
        raise GuidedCapturePublicationError() from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if parent_fd >= 0:
            # A completed final file is evidence and is intentionally preserved,
            # including when publication of the second file later fails.
            try:
                os.close(parent_fd)
            except OSError:
                pass


def _stage_terminal_file(
    parent: Path,
    name: str,
    payload: bytes,
    *,
    expected_parent_identity: tuple[int, ...] | None,
) -> tuple[int, int, _PrivateSnapshot]:
    """Prepare an unnamed, fully verified inode for one no-clobber link."""

    parent_fd = descriptor = -1
    prepared = False
    try:
        parent_fd, parent_metadata = _directory_fd(parent)
        if (
            expected_parent_identity is not None
            and _directory_identity(parent_metadata) != expected_parent_identity
        ):
            raise GuidedCapturePublicationError()
        try:
            os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise GuidedCapturePublicationError()

        temporary_flag = getattr(os, "O_TMPFILE", 0)
        if not temporary_flag:
            raise GuidedCapturePublicationError()
        flags = os.O_RDWR | temporary_flag | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(".", flags, 0o600, dir_fd=parent_fd)
        os.fchmod(descriptor, 0o600)
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise GuidedCapturePublicationError()
            written += count
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_IMODE(opened.st_mode) != 0o600
            or opened.st_uid != os.getuid()
            or opened.st_nlink != 0
            or opened.st_size != len(payload)
        ):
            raise GuidedCapturePublicationError()
        os.lseek(descriptor, 0, os.SEEK_SET)
        raw = bytearray()
        while len(raw) <= len(payload):
            chunk = os.read(
                descriptor,
                min(65536, len(payload) + 1 - len(raw)),
            )
            if not chunk:
                break
            raw.extend(chunk)
        after = os.fstat(descriptor)
        if bytes(raw) != payload or _file_identity(after) != _file_identity(opened):
            raise GuidedCapturePublicationError()
        snapshot = _PrivateSnapshot(
            path=parent / name,
            raw=payload,
            sha256=hashlib.sha256(payload).hexdigest(),
            inode=(opened.st_dev, opened.st_ino),
            identity=_file_identity(opened),
            binding_identity=_binding_file_identity(opened),
            size=opened.st_size,
        )
        prepared = True
        return parent_fd, descriptor, snapshot
    except GuidedCapturePublicationError:
        raise
    except (OSError, TypeError, ValueError):
        raise GuidedCapturePublicationError() from None
    finally:
        if not prepared and descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if not prepared and parent_fd >= 0:
            try:
                os.close(parent_fd)
            except OSError:
                pass


def _link_staged_terminal(parent_fd: int, descriptor: int, name: str) -> None:
    """Make an O_TMPFILE inode visible with one no-clobber syscall."""

    try:
        os.link(
            f"/proc/self/fd/{descriptor}",
            name,
            dst_dir_fd=parent_fd,
            follow_symlinks=True,
        )
    except (OSError, TypeError, ValueError):
        raise GuidedCapturePublicationError() from None


def _case_order_sha256() -> str:
    plan = built_in_guided_stt_plan()
    return hashlib.sha256(
        b"speaker-guided-stt-case-order-v1\0"
        + _canonical_json([case.case_id for case in plan.cases])
    ).hexdigest()


def _route_gate_profile_sha256() -> str:
    from tools.tool_route_gate import ToolRouteGateProfile, profile_digest

    return profile_digest(
        ToolRouteGateProfile(
            vault_enabled=True,
            reminders_enabled=True,
            app_aliases=("obsidian",),
        )
    )


def _load_contract_and_plan(
    run_dir: Path,
    *,
    expected_contract_sha256: str | None = None,
    expected_plan_sha256: str | None = None,
) -> tuple[_PrivateSnapshot, _PrivateSnapshot, dict[str, object]]:
    plan_snapshot = _snapshot_private_file(
        run_dir / _PLAN_FILENAME,
        maximum_bytes=_MAX_PLAN_BYTES,
    )
    contract_snapshot = _snapshot_private_file(
        run_dir / _CONTRACT_FILENAME,
        maximum_bytes=_MAX_CONTRACT_BYTES,
    )
    plan = built_in_guided_stt_plan()
    payload = _strict_json(contract_snapshot.raw)
    if (
        plan_snapshot.raw != canonical_guided_stt_plan_bytes()
        or plan_snapshot.sha256 != plan.sha256
        or (
            expected_plan_sha256 is not None
            and plan_snapshot.sha256 != _sha256(expected_plan_sha256)
        )
        or (
            expected_contract_sha256 is not None
            and contract_snapshot.sha256 != _sha256(expected_contract_sha256)
        )
        or not isinstance(payload, dict)
        or _canonical_json(payload) != contract_snapshot.raw
        or set(payload)
        != {
            "schema_version",
            "capture_protocol",
            "execution_purpose",
            "side_effect_policy",
            "speaker_identity_policy",
            "route_availability_sha256",
            "tool_route_profile_sha256",
            "capture_environment",
            "plan",
            "final_stt_profile",
        }
        or payload.get("schema_version") != _CONTRACT_SCHEMA_VERSION
        or payload.get("capture_protocol") != GUIDED_STT_CAPTURE_PROTOCOL
        or payload.get("execution_purpose") != "stt_capture_only"
        or payload.get("side_effect_policy") != _SIDE_EFFECT_POLICY
        or payload.get("speaker_identity_policy") != "off_for_session"
        or payload.get("route_availability_sha256")
        != guided_stt_route_availability_sha256()
        or payload.get("tool_route_profile_sha256") != _route_gate_profile_sha256()
    ):
        raise GuidedCapturePublicationError()

    environment = payload.get("capture_environment")
    plan_binding = payload.get("plan")
    profile = payload.get("final_stt_profile")
    if (
        not isinstance(environment, dict)
        or set(environment)
        != {
            "capture_config_sha256",
            "device_profile",
            "effective_input_gain",
            "effective_sherpa_sha256",
        }
        or _SHA256_RE.fullmatch(str(environment.get("capture_config_sha256", "")))
        is None
        or not isinstance(environment.get("device_profile"), str)
        or _SAFE_NAME_RE.fullmatch(str(environment["device_profile"])) is None
        or isinstance(environment.get("effective_input_gain"), bool)
        or not isinstance(environment.get("effective_input_gain"), (int, float))
        or not math.isfinite(float(environment["effective_input_gain"]))
        or float(environment["effective_input_gain"]) <= 0.0
        or _SHA256_RE.fullmatch(str(environment.get("effective_sherpa_sha256", "")))
        is None
        or not isinstance(plan_binding, dict)
        or set(plan_binding)
        != {
            "case_count",
            "case_order_sha256",
            "file",
            "id",
            "schema_version",
            "sha256",
        }
        or plan_binding.get("case_count") != plan.case_count
        or plan_binding.get("case_order_sha256") != _case_order_sha256()
        or plan_binding.get("file") != _PLAN_FILENAME
        or plan_binding.get("id") != plan.plan_id
        or plan_binding.get("schema_version") != plan.schema_version
        or plan_binding.get("sha256") != plan.sha256
        or not isinstance(profile, dict)
        or set(profile) != {"name", "schema_version", "sha256"}
        or profile.get("name") not in {"sense-voice", "parakeet-faster-whisper"}
        or type(profile.get("schema_version")) is not int
        or int(profile["schema_version"]) <= 0
        or _SHA256_RE.fullmatch(str(profile.get("sha256", ""))) is None
    ):
        raise GuidedCapturePublicationError()
    return plan_snapshot, contract_snapshot, payload


def _unique_summary_path(run_dir: Path) -> Path:
    try:
        candidates = tuple(
            path
            for path in run_dir.iterdir()
            if path.name.startswith("run-") and path.name.endswith(".summary.json")
        )
    except OSError:
        raise GuidedCapturePublicationError() from None
    if len(candidates) != 1:
        raise GuidedCapturePublicationError()
    return candidates[0]


def _summary_binding(
    run_dir: Path,
    snapshot: _PrivateSnapshot,
    contract: Mapping[str, object],
    contract_sha256: str,
) -> tuple[str, str, str, Path, frozenset[str]]:
    summary = _strict_json(snapshot.raw)
    if not isinstance(summary, dict):
        raise GuidedCapturePublicationError()
    run_id = summary.get("run_id")
    log_path_value = summary.get("log_path")
    meta = summary.get("meta")
    counts = summary.get("counts")
    llm = summary.get("llm")
    if (
        not isinstance(run_id, str)
        or _RUN_ID_RE.fullmatch(run_id) is None
        or snapshot.path.name != f"run-{run_id}.summary.json"
        or not isinstance(log_path_value, str)
        or Path(log_path_value) != run_dir / f"run-{run_id}.txt"
        or not isinstance(meta, dict)
        or not isinstance(counts, dict)
        or not isinstance(llm, dict)
        or summary.get("transcript") != []
        or summary.get("turns") != []
        or llm.get("requests") != []
        or counts.get("llm_requests") != 0
        or counts.get("turns") != 0
        or counts.get("transcript_entries") != 0
        or counts.get("errors") != 0
    ):
        raise GuidedCapturePublicationError()
    stored_summary_contract = _sha256(meta.get("capture_summary_contract_sha256"))
    expected_manifest_sha256 = _sha256(meta.get("diagnostic_manifest_sha256"))
    try:
        computed_summary_contract = guided_stt_summary_contract_sha256(meta)
    except Exception:
        raise GuidedCapturePublicationError() from None
    environment = contract["capture_environment"]
    plan = contract["plan"]
    profile = contract["final_stt_profile"]
    assert isinstance(environment, dict)
    assert isinstance(plan, dict)
    assert isinstance(profile, dict)
    if (
        not hmac.compare_digest(
            stored_summary_contract,
            computed_summary_contract,
        )
        or meta.get("capture_only_contract_sha256") != contract_sha256
        or meta.get("final_stt_profile") != profile["name"]
        or meta.get("final_stt_profile_sha256") != profile["sha256"]
        or meta.get("final_stt_profile_schema_version") != profile["schema_version"]
        or meta.get("capture_only_profile_sha256") != profile["sha256"]
        or meta.get("capture_only_sherpa_sha256")
        != environment["effective_sherpa_sha256"]
        or meta.get("device") != environment["device_profile"]
        or meta.get("capture_plan_sha256") != plan["sha256"]
    ):
        raise GuidedCapturePublicationError()

    expected_path_names = {
        "recording": f"run-{run_id}.wav",
        "pre_dsp_reference": f"run-{run_id}.pre-dsp.wav",
        "playback_reference": f"run-{run_id}.ref.wav",
        "asr_input_reference": f"run-{run_id}.asr.wav",
        "final_model_input": f"run-{run_id}.final-input.f32le",
        "diagnostic_timeline": f"run-{run_id}.timeline.jsonl",
    }
    if tuple(expected_path_names) != _SUMMARY_PATH_KEYS:
        raise GuidedCapturePublicationError()
    expected_names: set[str] = set()
    for key, expected_name in expected_path_names.items():
        value = meta.get(key)
        if not isinstance(value, str):
            raise GuidedCapturePublicationError()
        path = Path(value)
        if (
            not path.is_absolute()
            or path.parent != run_dir
            or path.name != expected_name
            or path.name in expected_names
        ):
            raise GuidedCapturePublicationError()
        expected_names.add(path.name)
    manifest_value = meta.get("diagnostic_manifest")
    if not isinstance(manifest_value, str):
        raise GuidedCapturePublicationError()
    manifest_path = Path(manifest_value)
    if (
        not manifest_path.is_absolute()
        or manifest_path.parent != run_dir
        or manifest_path.name != f"run-{run_id}.diagnostic.json"
        or manifest_path.name in expected_names
    ):
        raise GuidedCapturePublicationError()
    return (
        run_id,
        stored_summary_contract,
        expected_manifest_sha256,
        manifest_path,
        frozenset(expected_names),
    )


def _validated_manifest_snapshot(
    path: Path,
    *,
    expected_final_count: int,
    expected_artifact_files: frozenset[str],
) -> _PrivateSnapshot:
    from core.diagnostic_bundle import validate_manifest

    before = _snapshot_private_file(path, maximum_bytes=_MAX_MANIFEST_BYTES)
    if not validate_manifest(
        path,
        expected_final_model_input_count=expected_final_count,
        expected_artifact_files=expected_artifact_files,
    ):
        raise GuidedCapturePublicationError()
    after = _snapshot_private_file(path, maximum_bytes=_MAX_MANIFEST_BYTES)
    if not _snapshot_matches(before, after):
        raise GuidedCapturePublicationError()
    return before


def _artifact_payload(snapshot: _PrivateSnapshot) -> dict[str, object]:
    return {
        "file": snapshot.path.name,
        "bytes": snapshot.size,
        "sha256": snapshot.sha256,
    }


def _verified_bundle_value(
    *,
    run_dir: Path,
    run_dir_identity: tuple[int, ...],
    receipt: _PrivateSnapshot,
    plan: _PrivateSnapshot,
    contract: _PrivateSnapshot,
    contract_payload: Mapping[str, object],
    summary: _PrivateSnapshot,
    summary_contract_sha256: str,
    manifest: _PrivateSnapshot,
) -> VerifiedGuidedCaptureBundle:
    try:
        environment = contract_payload["capture_environment"]
        profile = contract_payload["final_stt_profile"]
        plan_binding = contract_payload["plan"]
        if not all(
            isinstance(value, Mapping) for value in (environment, profile, plan_binding)
        ):
            raise GuidedCapturePublicationError()
        return VerifiedGuidedCaptureBundle(
            run_dir=run_dir,
            _run_dir_identity=run_dir_identity,
            receipt_path=receipt.path,
            receipt_sha256=receipt.sha256,
            _receipt_identity=receipt.binding_identity,
            plan_path=plan.path,
            plan_sha256=plan.sha256,
            _plan_identity=plan.identity,
            contract_path=contract.path,
            contract_sha256=contract.sha256,
            _contract_identity=contract.identity,
            summary_path=summary.path,
            summary_sha256=summary.sha256,
            _summary_identity=summary.identity,
            summary_contract_sha256=summary_contract_sha256,
            diagnostic_manifest_path=manifest.path,
            diagnostic_manifest_sha256=manifest.sha256,
            _diagnostic_manifest_identity=manifest.identity,
            final_stt_profile=str(profile["name"]),
            final_stt_profile_sha256=str(profile["sha256"]),
            final_stt_profile_schema_version=int(profile["schema_version"]),
            capture_config_sha256=str(environment["capture_config_sha256"]),
            effective_sherpa_sha256=str(environment["effective_sherpa_sha256"]),
            device_profile=str(environment["device_profile"]),
            effective_input_gain=float(environment["effective_input_gain"]),
            case_count=int(plan_binding["case_count"]),
            case_order_sha256=str(plan_binding["case_order_sha256"]),
        )
    except GuidedCapturePublicationError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError):
        raise GuidedCapturePublicationError() from None


def publish_guided_capture_contract(
    run_dir: Path | str,
    *,
    final_stt_profile: str,
    final_stt_profile_sha256: str,
    final_stt_profile_schema_version: int,
    effective_device: str,
    effective_input_gain: float,
    capture_config_sha256: str,
    effective_sherpa_sha256: str,
) -> GuidedCapturePublication:
    """Publish the immutable plan and aggregate-only binding before capture."""

    try:
        parent = Path(run_dir)
        if (
            not isinstance(final_stt_profile, str)
            or _SAFE_NAME_RE.fullmatch(final_stt_profile) is None
            or not isinstance(final_stt_profile_sha256, str)
            or _SHA256_RE.fullmatch(final_stt_profile_sha256) is None
            or type(final_stt_profile_schema_version) is not int
            or final_stt_profile_schema_version <= 0
            or not isinstance(effective_device, str)
            or _SAFE_NAME_RE.fullmatch(effective_device) is None
            or isinstance(effective_input_gain, bool)
            or not isinstance(effective_input_gain, (int, float))
            or not math.isfinite(float(effective_input_gain))
            or float(effective_input_gain) <= 0.0
            or not isinstance(capture_config_sha256, str)
            or _SHA256_RE.fullmatch(capture_config_sha256) is None
            or not isinstance(effective_sherpa_sha256, str)
            or _SHA256_RE.fullmatch(effective_sherpa_sha256) is None
        ):
            raise GuidedCapturePublicationError()
        plan = built_in_guided_stt_plan()
        plan_raw = canonical_guided_stt_plan_bytes()
        plan_file = _publish_at(parent, _PLAN_FILENAME, plan_raw)
        if plan_file.sha256 != plan.sha256:
            raise GuidedCapturePublicationError()
        contract_raw = _canonical_json(
            {
                "schema_version": _CONTRACT_SCHEMA_VERSION,
                "capture_protocol": GUIDED_STT_CAPTURE_PROTOCOL,
                "execution_purpose": "stt_capture_only",
                "side_effect_policy": dict(_SIDE_EFFECT_POLICY),
                "speaker_identity_policy": "off_for_session",
                "route_availability_sha256": (guided_stt_route_availability_sha256()),
                "tool_route_profile_sha256": _route_gate_profile_sha256(),
                "capture_environment": {
                    "capture_config_sha256": capture_config_sha256,
                    "device_profile": effective_device,
                    "effective_input_gain": float(effective_input_gain),
                    "effective_sherpa_sha256": effective_sherpa_sha256,
                },
                "plan": {
                    "case_count": plan.case_count,
                    "case_order_sha256": _case_order_sha256(),
                    "file": _PLAN_FILENAME,
                    "id": plan.plan_id,
                    "schema_version": plan.schema_version,
                    "sha256": plan.sha256,
                },
                "final_stt_profile": {
                    "name": final_stt_profile,
                    "schema_version": final_stt_profile_schema_version,
                    "sha256": final_stt_profile_sha256,
                },
            }
        )
        contract_file = _publish_at(
            parent,
            _CONTRACT_FILENAME,
            contract_raw,
            expected_parent_identity=plan_file.parent_identity,
        )
        publication = GuidedCapturePublication(
            plan_path=plan_file.path,
            contract_path=contract_file.path,
            plan_sha256=plan_file.sha256,
            contract_sha256=contract_file.sha256,
            case_count=plan.case_count,
            _run_dir_identity=plan_file.parent_identity,
            _plan_identity=plan_file.identity,
            _contract_identity=contract_file.identity,
            _plan_bytes=plan_raw,
            _contract_bytes=contract_raw,
        )
        verify_guided_capture_publication(publication)
        return publication
    except GuidedCapturePublicationError:
        raise
    except Exception:
        raise GuidedCapturePublicationError() from None


def _verify_file_at(
    parent_fd: int,
    name: str,
    identity: tuple[int, ...],
    expected: bytes,
) -> None:
    descriptor = -1
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(name, flags, dir_fd=parent_fd)
        metadata = os.fstat(descriptor)
        if (
            _file_identity(metadata) != identity
            or not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_uid != os.getuid()
            or metadata.st_nlink != 1
            or metadata.st_size != len(expected)
        ):
            raise GuidedCapturePublicationError()
        before_identity = (
            metadata.st_dev,
            metadata.st_ino,
            stat.S_IFMT(metadata.st_mode),
            stat.S_IMODE(metadata.st_mode),
            metadata.st_uid,
            metadata.st_nlink,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )
        raw = bytearray()
        while len(raw) <= len(expected):
            chunk = os.read(descriptor, min(8192, len(expected) + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
        after = os.fstat(descriptor)
        path_after = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        final_fd = os.fstat(descriptor)
        after_identity = (
            after.st_dev,
            after.st_ino,
            stat.S_IFMT(after.st_mode),
            stat.S_IMODE(after.st_mode),
            after.st_uid,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        path_identity = (
            path_after.st_dev,
            path_after.st_ino,
            stat.S_IFMT(path_after.st_mode),
            stat.S_IMODE(path_after.st_mode),
            path_after.st_uid,
            path_after.st_nlink,
            path_after.st_size,
            path_after.st_mtime_ns,
            path_after.st_ctime_ns,
        )
        final_identity = (
            final_fd.st_dev,
            final_fd.st_ino,
            stat.S_IFMT(final_fd.st_mode),
            stat.S_IMODE(final_fd.st_mode),
            final_fd.st_uid,
            final_fd.st_nlink,
            final_fd.st_size,
            final_fd.st_mtime_ns,
            final_fd.st_ctime_ns,
        )
        if (
            bytes(raw) != expected
            or before_identity != after_identity
            or after_identity != path_identity
            or path_identity != final_identity
        ):
            raise GuidedCapturePublicationError()
    except GuidedCapturePublicationError:
        raise
    except (OSError, TypeError, ValueError):
        raise GuidedCapturePublicationError() from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def verify_guided_capture_publication(
    publication: GuidedCapturePublication,
) -> None:
    if not isinstance(publication, GuidedCapturePublication):
        raise GuidedCapturePublicationError()
    if (
        publication.plan_path.parent != publication.contract_path.parent
        or publication.plan_path.name != _PLAN_FILENAME
        or publication.contract_path.name != _CONTRACT_FILENAME
    ):
        raise GuidedCapturePublicationError()
    parent_fd = -1
    try:
        parent = publication.plan_path.parent
        parent_fd, metadata = _directory_fd(parent)
        if _directory_identity(metadata) != publication._run_dir_identity:
            raise GuidedCapturePublicationError()
        _verify_file_at(
            parent_fd,
            publication.plan_path.name,
            publication._plan_identity,
            publication._plan_bytes,
        )
        _verify_file_at(
            parent_fd,
            publication.contract_path.name,
            publication._contract_identity,
            publication._contract_bytes,
        )
        after = os.fstat(parent_fd)
        lexical = parent.lstat()
        if (
            _directory_identity(after) != publication._run_dir_identity
            or _directory_identity(lexical) != publication._run_dir_identity
            or parent.resolve(strict=True) != parent
        ):
            raise GuidedCapturePublicationError()
    finally:
        if parent_fd >= 0:
            try:
                os.close(parent_fd)
            except OSError:
                pass


def publish_guided_capture_bundle_receipt(
    publication: GuidedCapturePublication,
    *,
    commit_guard: Callable[[], bool] | None = None,
) -> GuidedCaptureBundlePublication:
    """Publish the terminal root with the final no-clobber link as commit."""

    parent_fd = receipt_fd = -1
    try:
        if commit_guard is not None and not callable(commit_guard):
            raise GuidedCapturePublicationError()
        verify_guided_capture_publication(publication)
        run_dir = publication.plan_path.parent
        plan, contract, contract_payload = _load_contract_and_plan(
            run_dir,
            expected_contract_sha256=publication.contract_sha256,
            expected_plan_sha256=publication.plan_sha256,
        )
        if (
            plan.identity != publication._plan_identity
            or contract.identity != publication._contract_identity
        ):
            raise GuidedCapturePublicationError()
        summary = _snapshot_private_file(
            _unique_summary_path(run_dir),
            maximum_bytes=_MAX_SUMMARY_BYTES,
        )
        (
            run_id,
            summary_contract_sha256,
            expected_manifest_sha256,
            manifest_path,
            expected_artifact_files,
        ) = _summary_binding(
            run_dir,
            summary,
            contract_payload,
            contract.sha256,
        )
        manifest = _validated_manifest_snapshot(
            manifest_path,
            expected_final_count=publication.case_count,
            expected_artifact_files=expected_artifact_files,
        )
        if manifest.sha256 != expected_manifest_sha256:
            raise GuidedCapturePublicationError()
        receipt_raw = _canonical_json(
            {
                "schema_version": _RECEIPT_SCHEMA_VERSION,
                "kind": _RECEIPT_KIND,
                "capture_protocol": GUIDED_STT_CAPTURE_PROTOCOL,
                "run_id": run_id,
                "reference_plan": _artifact_payload(plan),
                "capture_contract": _artifact_payload(contract),
                "run_summary": {
                    **_artifact_payload(summary),
                    "contract_sha256": summary_contract_sha256,
                },
                "diagnostic_manifest": _artifact_payload(manifest),
            }
        )
        parent_fd, receipt_fd, receipt = _stage_terminal_file(
            run_dir,
            _RECEIPT_FILENAME,
            receipt_raw,
            expected_parent_identity=publication._run_dir_identity,
        )

        # Staging is invisible. Reopen every retained DAG node after it and bind
        # the still-live lexical run directory to the held descriptor. Nothing
        # fallible follows the guard except the single no-clobber final link.
        verify_guided_capture_publication(publication)
        for retained, maximum in (
            (plan, _MAX_PLAN_BYTES),
            (contract, _MAX_CONTRACT_BYTES),
            (summary, _MAX_SUMMARY_BYTES),
            (manifest, _MAX_MANIFEST_BYTES),
        ):
            if not _snapshot_matches(
                retained,
                _snapshot_private_file(retained.path, maximum_bytes=maximum),
            ):
                raise GuidedCapturePublicationError()
        opened_parent = os.fstat(parent_fd)
        lexical_parent = run_dir.lstat()
        if (
            _directory_identity(opened_parent) != publication._run_dir_identity
            or _directory_identity(lexical_parent) != publication._run_dir_identity
            or run_dir.resolve(strict=True) != run_dir
        ):
            raise GuidedCapturePublicationError()

        verified = _verified_bundle_value(
            run_dir=run_dir,
            run_dir_identity=publication._run_dir_identity,
            receipt=receipt,
            plan=plan,
            contract=contract,
            contract_payload=contract_payload,
            summary=summary,
            summary_contract_sha256=summary_contract_sha256,
            manifest=manifest,
        )
        value = GuidedCaptureBundlePublication(
            receipt_path=verified.receipt_path,
            receipt_sha256=verified.receipt_sha256,
            summary_contract_sha256=summary_contract_sha256,
            _receipt_identity=receipt.binding_identity,
            _receipt_bytes=receipt_raw,
            verified_bundle=verified,
        )
        if commit_guard is not None and commit_guard() is not True:
            raise GuidedCapturePublicationError()
        _link_staged_terminal(parent_fd, receipt_fd, _RECEIPT_FILENAME)
        return value
    except GuidedCapturePublicationError:
        raise
    except Exception:
        raise GuidedCapturePublicationError() from None
    finally:
        if receipt_fd >= 0:
            try:
                os.close(receipt_fd)
            except OSError:
                pass
        if parent_fd >= 0:
            try:
                os.close(parent_fd)
            except OSError:
                pass


def _receipt_artifact(
    value: object,
    *,
    summary: bool = False,
) -> tuple[str, int, str, str | None]:
    expected = {"file", "bytes", "sha256"}
    if summary:
        expected.add("contract_sha256")
    if not isinstance(value, dict) or set(value) != expected:
        raise GuidedCapturePublicationError()
    filename = _safe_file(value.get("file"))
    size = value.get("bytes")
    if type(size) is not int or size <= 0:
        raise GuidedCapturePublicationError()
    digest = _sha256(value.get("sha256"))
    contract_digest = _sha256(value.get("contract_sha256")) if summary else None
    return filename, size, digest, contract_digest


def load_verified_guided_capture_bundle(
    run_dir: Path | str,
) -> VerifiedGuidedCaptureBundle:
    """Strictly reopen one terminal receipt and every transitive binding."""

    try:
        parent = Path(run_dir)
        if (
            not parent.is_absolute()
            or Path(os.path.abspath(parent)) != parent
            or parent.resolve(strict=True) != parent
        ):
            raise GuidedCapturePublicationError()
        parent_fd = -1
        try:
            parent_fd, parent_before = _directory_fd(parent)
            run_dir_identity = _directory_identity(parent_before)
        finally:
            if parent_fd >= 0:
                os.close(parent_fd)
        receipt = _snapshot_private_file(
            parent / _RECEIPT_FILENAME,
            maximum_bytes=_MAX_RECEIPT_BYTES,
        )
        payload = _strict_json(receipt.raw)
        if (
            not isinstance(payload, dict)
            or _canonical_json(payload) != receipt.raw
            or set(payload)
            != {
                "schema_version",
                "kind",
                "capture_protocol",
                "run_id",
                "reference_plan",
                "capture_contract",
                "run_summary",
                "diagnostic_manifest",
            }
            or payload.get("schema_version") != _RECEIPT_SCHEMA_VERSION
            or payload.get("kind") != _RECEIPT_KIND
            or payload.get("capture_protocol") != GUIDED_STT_CAPTURE_PROTOCOL
            or not isinstance(payload.get("run_id"), str)
            or _RUN_ID_RE.fullmatch(str(payload["run_id"])) is None
        ):
            raise GuidedCapturePublicationError()
        plan_entry = _receipt_artifact(payload.get("reference_plan"))
        contract_entry = _receipt_artifact(payload.get("capture_contract"))
        summary_entry = _receipt_artifact(payload.get("run_summary"), summary=True)
        manifest_entry = _receipt_artifact(payload.get("diagnostic_manifest"))
        filenames = (
            plan_entry[0],
            contract_entry[0],
            summary_entry[0],
            manifest_entry[0],
            _RECEIPT_FILENAME,
        )
        if (
            plan_entry[0] != _PLAN_FILENAME
            or contract_entry[0] != _CONTRACT_FILENAME
            or len(filenames) != len(set(filenames))
        ):
            raise GuidedCapturePublicationError()

        plan, contract, contract_payload = _load_contract_and_plan(
            parent,
            expected_contract_sha256=contract_entry[2],
            expected_plan_sha256=plan_entry[2],
        )
        if plan.size != plan_entry[1] or contract.size != contract_entry[1]:
            raise GuidedCapturePublicationError()
        summary = _snapshot_private_file(
            parent / summary_entry[0],
            maximum_bytes=_MAX_SUMMARY_BYTES,
        )
        if summary.size != summary_entry[1] or summary.sha256 != summary_entry[2]:
            raise GuidedCapturePublicationError()
        (
            run_id,
            summary_contract_sha256,
            expected_manifest_sha256,
            manifest_path,
            expected_artifact_files,
        ) = _summary_binding(
            parent,
            summary,
            contract_payload,
            contract.sha256,
        )
        if (
            run_id != payload["run_id"]
            or summary_contract_sha256 != summary_entry[3]
            or manifest_path.name != manifest_entry[0]
        ):
            raise GuidedCapturePublicationError()
        manifest = _validated_manifest_snapshot(
            manifest_path,
            expected_final_count=int(contract_payload["plan"]["case_count"]),
            expected_artifact_files=expected_artifact_files,
        )
        if (
            manifest.size != manifest_entry[1]
            or manifest.sha256 != manifest_entry[2]
            or manifest.sha256 != expected_manifest_sha256
        ):
            raise GuidedCapturePublicationError()

        for prior, maximum in (
            (plan, _MAX_PLAN_BYTES),
            (contract, _MAX_CONTRACT_BYTES),
            (summary, _MAX_SUMMARY_BYTES),
            (manifest, _MAX_MANIFEST_BYTES),
            (receipt, _MAX_RECEIPT_BYTES),
        ):
            if not _snapshot_matches(
                prior,
                _snapshot_private_file(prior.path, maximum_bytes=maximum),
            ):
                raise GuidedCapturePublicationError()
        parent_fd = -1
        try:
            parent_fd, parent_after = _directory_fd(parent)
            if _directory_identity(parent_after) != run_dir_identity:
                raise GuidedCapturePublicationError()
        finally:
            if parent_fd >= 0:
                os.close(parent_fd)

        return _verified_bundle_value(
            run_dir=parent,
            run_dir_identity=run_dir_identity,
            receipt=receipt,
            plan=plan,
            contract=contract,
            contract_payload=contract_payload,
            summary=summary,
            summary_contract_sha256=summary_contract_sha256,
            manifest=manifest,
        )
    except GuidedCapturePublicationError:
        raise
    except Exception:
        raise GuidedCapturePublicationError() from None


def verify_guided_capture_bundle(bundle: VerifiedGuidedCaptureBundle) -> None:
    """Reopen a previously verified bundle and require exact typed equality."""

    if not isinstance(bundle, VerifiedGuidedCaptureBundle):
        raise GuidedCapturePublicationError()
    if load_verified_guided_capture_bundle(bundle.run_dir) != bundle:
        raise GuidedCapturePublicationError()


__all__ = [
    "GuidedCaptureBundlePublication",
    "GuidedCapturePublication",
    "GuidedCapturePublicationError",
    "VerifiedGuidedCaptureBundle",
    "load_verified_guided_capture_bundle",
    "publish_guided_capture_bundle_receipt",
    "publish_guided_capture_contract",
    "verify_guided_capture_bundle",
    "verify_guided_capture_publication",
]
