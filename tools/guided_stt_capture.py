"""Private publication contract for the fixed guided STT capture plan."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat

from core.guided_stt_plan import (
    GUIDED_STT_CAPTURE_PROTOCOL,
    built_in_guided_stt_plan,
    canonical_guided_stt_plan_bytes,
    guided_stt_route_availability_sha256,
)


_SAFE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PLAN_FILENAME = "reference-plan.json"
_CONTRACT_FILENAME = "capture-contract.json"
_CONTRACT_SCHEMA_VERSION = 1


class GuidedCapturePublicationError(RuntimeError):
    """A detail-free private publication or binding failure."""

    def __init__(self) -> None:
        super().__init__("guided capture prerequisites unavailable")


@dataclass(frozen=True, slots=True)
class _PublishedFile:
    path: Path = field(repr=False)
    sha256: str
    inode: tuple[int, int] = field(repr=False)
    size: int


@dataclass(frozen=True, slots=True)
class GuidedCapturePublication:
    plan_path: Path = field(repr=False)
    contract_path: Path = field(repr=False)
    plan_sha256: str
    contract_sha256: str
    case_count: int
    _plan_inode: tuple[int, int] = field(repr=False)
    _contract_inode: tuple[int, int] = field(repr=False)
    _plan_bytes: bytes = field(repr=False)
    _contract_bytes: bytes = field(repr=False)


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


def _publish_at(parent: Path, name: str, payload: bytes) -> _PublishedFile:
    parent_fd = descriptor = -1
    temporary = f".guided-capture-{secrets.token_hex(12)}.tmp"
    temporary_exists = False
    try:
        parent_fd, parent_before = _directory_fd(parent)
        try:
            os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise GuidedCapturePublicationError()

        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(temporary, flags, 0o600, dir_fd=parent_fd)
        temporary_exists = True
        os.fchmod(descriptor, 0o600)
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise GuidedCapturePublicationError()
            written += count
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_uid != os.getuid()
            or metadata.st_nlink != 1
            or metadata.st_size != len(payload)
        ):
            raise GuidedCapturePublicationError()

        os.link(
            temporary,
            name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        os.unlink(temporary, dir_fd=parent_fd)
        temporary_exists = False
        os.fsync(parent_fd)
        published = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        parent_after = os.fstat(parent_fd)
        if (
            parent_after.st_dev != parent_before.st_dev
            or parent_after.st_ino != parent_before.st_ino
            or not stat.S_ISREG(published.st_mode)
            or stat.S_IMODE(published.st_mode) != 0o600
            or published.st_uid != os.getuid()
            or published.st_nlink != 1
            or published.st_dev != metadata.st_dev
            or published.st_ino != metadata.st_ino
            or published.st_size != len(payload)
        ):
            raise GuidedCapturePublicationError()
        return _PublishedFile(
            path=parent / name,
            sha256=hashlib.sha256(payload).hexdigest(),
            inode=(published.st_dev, published.st_ino),
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
            if temporary_exists:
                try:
                    os.unlink(temporary, dir_fd=parent_fd)
                except OSError:
                    pass
            # A completed final file is evidence and is intentionally preserved,
            # including when publication of the second file later fails.
            try:
                os.close(parent_fd)
            except OSError:
                pass


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
                "side_effect_policy": {
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
                },
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
        contract_file = _publish_at(parent, _CONTRACT_FILENAME, contract_raw)
        publication = GuidedCapturePublication(
            plan_path=plan_file.path,
            contract_path=contract_file.path,
            plan_sha256=plan_file.sha256,
            contract_sha256=contract_file.sha256,
            case_count=plan.case_count,
            _plan_inode=plan_file.inode,
            _contract_inode=contract_file.inode,
            _plan_bytes=plan_raw,
            _contract_bytes=contract_raw,
        )
        verify_guided_capture_publication(publication)
        return publication
    except GuidedCapturePublicationError:
        raise
    except Exception:
        raise GuidedCapturePublicationError() from None


def _verify_file(path: Path, inode: tuple[int, int], expected: bytes) -> None:
    descriptor = -1
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        metadata = os.fstat(descriptor)
        if (
            (metadata.st_dev, metadata.st_ino) != inode
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
        path_after = os.stat(path, follow_symlinks=False)
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
    if publication.plan_path.parent != publication.contract_path.parent:
        raise GuidedCapturePublicationError()
    parent_fd = -1
    try:
        parent_fd, _metadata = _directory_fd(publication.plan_path.parent)
    finally:
        if parent_fd >= 0:
            try:
                os.close(parent_fd)
            except OSError:
                pass
    _verify_file(
        publication.plan_path,
        publication._plan_inode,
        publication._plan_bytes,
    )
    _verify_file(
        publication.contract_path,
        publication._contract_inode,
        publication._contract_bytes,
    )


__all__ = [
    "GuidedCapturePublication",
    "GuidedCapturePublicationError",
    "publish_guided_capture_contract",
    "verify_guided_capture_publication",
]
