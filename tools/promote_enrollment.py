"""Promote one accepted isolated v5 speaker enrollment without opening audio.

Preparation and capture deliberately leave the historical reference active.
This command is the explicit, device-free commit step after the operator has
accepted the live gate. It publishes a private independent copy of the feature
candidate, then atomically changes only the primary config's enrollment pointer.

No embedding, digest, secret value, or other file content is printed. Success
prints the explicitly supplied config and enrollment paths for operator review.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import stat
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

from core.enroll import (
    ENROLLMENT_FRONTEND_VERSION,
    ENROLLMENT_PREPARATION_KEY,
    ENROLLMENT_PREPARATION_SCHEMA,
    EnrollmentFrontendProvenance,
)
from tools.prepare_enrollment import PreparationError, _write_all


class PromotionError(RuntimeError):
    """The requested promotion was refused before a safe staging commit."""


class PromotionStagedError(PromotionError):
    """Exact accepted bytes are durable and the old config is still active."""


class PromotionAmbiguousError(PromotionError):
    """A filesystem commit may have happened, so the operator must inspect."""


@dataclass(frozen=True)
class PromotionResult:
    primary_config: Path
    accepted_enrollment: Path
    previous_enrollment: Path
    rollback_backup: Path
    source_candidate: Path
    accepted_was_adopted: bool = False


@dataclass(frozen=True)
class _DirectorySnapshot:
    path: str
    dev: int
    ino: int
    mode: int
    uid: int
    gid: int


@dataclass(frozen=True)
class _BoundRegularPathSnapshot:
    """Identity and content obtained through one opened regular-file inode."""

    path: str
    label: str
    dev: int
    ino: int
    size: int
    mtime_ns: int
    ctime_ns: int
    mode: int
    uid: int
    gid: int
    nlink: int
    content_sha256: bytes
    ancestors: tuple[_DirectorySnapshot, ...]


_CANDIDATE_NAME = re.compile(
    r"enrollment\.v5-[A-Za-z0-9][A-Za-z0-9._-]*\.json\Z"
)
_FINGERPRINT = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SHA256_HEX = re.compile(r"[0-9a-f]{64}\Z")
_MARKER_CONTRACT_FIELDS = (
    "dev",
    "ino",
    "size",
    "mtime_ns",
    "ctime_ns",
    "mode",
    "uid",
    "gid",
    "nlink",
)


def _absolute(path: str | os.PathLike[str], *, label: str) -> Path:
    value = Path(path)
    if not value.is_absolute():
        raise PromotionError(f"{label} must be an absolute path")
    return Path(os.path.abspath(os.fspath(value)))


def _directory_chain(path: Path) -> tuple[_DirectorySnapshot, ...]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    snapshots: list[_DirectorySnapshot] = []
    for part in absolute.parts[1:]:
        current /= part
        info = os.lstat(current)
        if stat.S_ISLNK(info.st_mode):
            raise PromotionError(f"refusing symlink directory: {current}")
        if not stat.S_ISDIR(info.st_mode):
            raise PromotionError(f"path component is not a directory: {current}")
        snapshots.append(
            _DirectorySnapshot(
                path=str(current),
                dev=info.st_dev,
                ino=info.st_ino,
                mode=stat.S_IMODE(info.st_mode),
                uid=info.st_uid,
                gid=info.st_gid,
            )
        )
    return tuple(snapshots)


def _require_owned_directory(path: Path, *, label: str) -> None:
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise PromotionError(f"{label} is not a regular directory: {path}")
    if info.st_uid != os.getuid():
        raise PromotionError(f"{label} is not owned by the current user: {path}")
    if stat.S_IMODE(info.st_mode) & 0o022:
        raise PromotionError(f"{label} is group/world writable: {path}")


def _metadata_signature(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
        stat.S_IMODE(info.st_mode),
        info.st_uid,
        info.st_gid,
        info.st_nlink,
    )


def _read_bound_bytes(
    path: Path, *, label: str
) -> tuple[_BoundRegularPathSnapshot, bytes]:
    """Read bytes and all guard metadata from the same no-follow file handle."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    ancestors = _directory_chain(absolute.parent)
    try:
        before = os.lstat(absolute)
    except FileNotFoundError as exc:
        raise PromotionError(f"missing {label}: {absolute}") from exc
    if stat.S_ISLNK(before.st_mode):
        raise PromotionError(f"refusing to follow symlink for {label}: {absolute}")
    if not stat.S_ISREG(before.st_mode):
        raise PromotionError(f"refusing non-regular {label}: {absolute}")

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(absolute, flags)
    except OSError as exc:
        raise PromotionError(f"could not safely open {label}: {absolute}") from exc
    try:
        opened = os.fstat(fd)
        if (
            not stat.S_ISREG(opened.st_mode)
            or _metadata_signature(opened) != _metadata_signature(before)
        ):
            raise PromotionError(f"{label} changed while opening: {absolute}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)

    raw = b"".join(chunks)
    try:
        current = os.lstat(absolute)
        current_ancestors = _directory_chain(absolute.parent)
    except (FileNotFoundError, OSError) as exc:
        raise PromotionError(f"{label} changed while reading: {absolute}") from exc
    if (
        not stat.S_ISREG(current.st_mode)
        or _metadata_signature(after) != _metadata_signature(opened)
        or _metadata_signature(current) != _metadata_signature(after)
        or current_ancestors != ancestors
        or len(raw) != after.st_size
    ):
        raise PromotionError(f"{label} changed while reading: {absolute}")

    return (
        _BoundRegularPathSnapshot(
            path=str(absolute),
            label=label,
            dev=after.st_dev,
            ino=after.st_ino,
            size=after.st_size,
            mtime_ns=after.st_mtime_ns,
            ctime_ns=after.st_ctime_ns,
            mode=stat.S_IMODE(after.st_mode),
            uid=after.st_uid,
            gid=after.st_gid,
            nlink=after.st_nlink,
            content_sha256=hashlib.sha256(raw).digest(),
            ancestors=ancestors,
        ),
        raw,
    )


def _read_bound_json(
    path: Path, *, label: str
) -> tuple[_BoundRegularPathSnapshot, dict[str, object], bytes]:
    snapshot, raw = _read_bound_bytes(path, label=label)
    try:
        loaded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PromotionError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(loaded, dict):
        raise PromotionError(f"{label} must contain a JSON object")
    return snapshot, loaded, raw


def _revalidate_bound_path(snapshot: _BoundRegularPathSnapshot) -> None:
    current, _ = _read_bound_bytes(Path(snapshot.path), label=snapshot.label)
    if current != snapshot:
        raise PromotionError(f"{snapshot.label} changed: {snapshot.path}")


def _require_private_current_user(
    snapshot: _BoundRegularPathSnapshot, *, label: str
) -> None:
    if snapshot.uid != os.getuid():
        raise PromotionError(f"{label} is not owned by the current user")
    if snapshot.mode != 0o600:
        raise PromotionError(f"{label} is not mode 600")
    if snapshot.nlink != 1:
        raise PromotionError(f"{label} does not have exactly one link")


def _require_disjoint_inodes(
    snapshots: tuple[_BoundRegularPathSnapshot, ...],
) -> None:
    seen: dict[tuple[int, int], str] = {}
    for snapshot in snapshots:
        identity = (snapshot.dev, snapshot.ino)
        previous = seen.get(identity)
        if previous is not None:
            raise PromotionError(
                f"protected state aliases one inode: {previous} and {snapshot.label}"
            )
        seen[identity] = snapshot.label


def _required_mapping(
    data: Mapping[str, object], key: str, *, label: str
) -> Mapping[str, object]:
    value = data.get(key)
    if not isinstance(value, Mapping):
        raise PromotionError(f"{label} has no {key} JSON object")
    return value


def _marker_int(marker: Mapping[str, object], key: str) -> int:
    value = marker.get(key)
    if type(value) is not int or value < 0:
        raise PromotionError(f"prepared marker has invalid {key}")
    return value


def _marker_contract(
    marker: Mapping[str, object], key: str
) -> tuple[tuple[int, ...], str]:
    metadata = tuple(
        _marker_int(marker, f"{key}_{field}")
        for field in _MARKER_CONTRACT_FIELDS
    )
    digest = marker.get(f"{key}_sha256")
    if not isinstance(digest, str) or not _SHA256_HEX.fullmatch(digest):
        raise PromotionError(f"prepared marker has invalid {key}_sha256")
    return metadata, digest


def _snapshot_marker_contract(
    snapshot: _BoundRegularPathSnapshot,
) -> tuple[tuple[int, ...], str]:
    return (
        (
            snapshot.dev,
            snapshot.ino,
            snapshot.size,
            snapshot.mtime_ns,
            snapshot.ctime_ns,
            snapshot.mode,
            snapshot.uid,
            snapshot.gid,
            snapshot.nlink,
        ),
        snapshot.content_sha256.hex(),
    )


def _require_marker_path(
    marker: Mapping[str, object], key: str, expected: Path
) -> None:
    if marker.get(key) != str(expected):
        raise PromotionError(f"prepared marker {key} does not match the explicit path")


def _configured_absolute(
    config: Mapping[str, object], key: str, *, label: str
) -> Path:
    value = config.get(key)
    if not isinstance(value, str) or not value or not os.path.isabs(value):
        raise PromotionError(f"{label} has no absolute {key}")
    return Path(os.path.abspath(value))


def _configured_sample_rate(config: Mapping[str, object], *, label: str) -> int:
    value = config.get("sample_rate", 16000)
    if type(value) is not int or value <= 0:
        raise PromotionError(f"{label} has an invalid sample_rate")
    return value


def _historical_dimension(
    data: Mapping[str, object], *, expected_model: Path
) -> int:
    model = data.get("model")
    if not isinstance(model, str) or not os.path.isabs(model):
        raise PromotionError("historical enrollment lacks absolute model provenance")
    if Path(os.path.abspath(model)) != expected_model:
        raise PromotionError("historical enrollment was produced by a different model")
    embedding = data.get("embedding")
    dim = data.get("dim")
    if (
        not isinstance(embedding, list)
        or not embedding
        or type(dim) is not int
        or dim != len(embedding)
    ):
        raise PromotionError("historical enrollment has invalid dimension evidence")
    return dim


def _validate_candidate_payload(
    data: Mapping[str, object],
    *,
    expected_model: Path,
    expected_sample_rate: int,
    historical_dim: int,
) -> None:
    embedding = data.get("embedding")
    if not isinstance(embedding, list) or not embedding:
        raise PromotionError("v5 candidate has no non-empty embedding")
    values: list[float] = []
    for value in embedding:
        if type(value) not in (int, float):
            raise PromotionError("v5 candidate embedding contains a non-number")
        try:
            number = float(value)
        except OverflowError as exc:
            raise PromotionError(
                "v5 candidate embedding contains an out-of-range number"
            ) from exc
        if not math.isfinite(number):
            raise PromotionError("v5 candidate embedding contains a non-finite value")
        values.append(number)
    norm = math.hypot(*values)
    if not math.isfinite(norm) or not math.isclose(
        norm, 1.0, rel_tol=1e-4, abs_tol=1e-4
    ):
        raise PromotionError("v5 candidate embedding is not a finite unit vector")

    dim = data.get("dim")
    if type(dim) is not int or dim != len(values):
        raise PromotionError("v5 candidate dimension does not match its embedding")
    if dim != historical_dim:
        raise PromotionError(
            "v5 candidate dimension does not match the historical same-model enrollment"
        )
    passes = data.get("passes")
    if type(passes) is not int or passes < 3:
        raise PromotionError("v5 candidate requires at least three enrollment passes")
    sample_rate = data.get("sample_rate")
    if type(sample_rate) is not int or sample_rate != expected_sample_rate:
        raise PromotionError("v5 candidate sample rate does not match the config")

    model = data.get("model")
    if not isinstance(model, str) or not model or not os.path.isabs(model):
        raise PromotionError("v5 candidate has no absolute model provenance")
    if Path(os.path.abspath(model)) != expected_model:
        raise PromotionError("v5 candidate model does not match the prepared config")

    frontend_data = data.get("frontend")
    if not isinstance(frontend_data, Mapping):
        raise PromotionError("v5 candidate has no front-end provenance")
    if type(frontend_data.get("version")) is not int or (
        frontend_data.get("version") != ENROLLMENT_FRONTEND_VERSION
    ):
        raise PromotionError("candidate provenance is not exact front-end v5")
    if not isinstance(frontend_data.get("summary"), str) or not str(
        frontend_data.get("summary")
    ).strip():
        raise PromotionError("v5 candidate has invalid front-end provenance")
    if type(frontend_data.get("raw_baseline")) is not bool:
        raise PromotionError("v5 candidate has invalid front-end provenance")
    try:
        frontend = EnrollmentFrontendProvenance.from_dict(frontend_data)
    except ValueError as exc:
        raise PromotionError("v5 candidate has invalid front-end provenance") from exc
    if not _FINGERPRINT.fullmatch(frontend.fingerprint):
        raise PromotionError("v5 candidate has an invalid front-end fingerprint")
    if frontend.raw_baseline:
        raise PromotionError("raw-baseline provenance is not a live v5 capture")


def _strict_fsync_directory(path: Path, *, label: str) -> None:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISDIR(info.st_mode):
            raise OSError(f"{label} is not a directory")
        os.fsync(fd)
    finally:
        os.close(fd)


def _strict_fsync_bound_file(snapshot: _BoundRegularPathSnapshot) -> None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(snapshot.path, flags)
    try:
        opened = os.fstat(fd)
        if _metadata_signature(opened) != (
            snapshot.dev,
            snapshot.ino,
            snapshot.size,
            snapshot.mtime_ns,
            snapshot.ctime_ns,
            snapshot.mode,
            snapshot.uid,
            snapshot.gid,
            snapshot.nlink,
        ):
            raise PromotionError(f"{snapshot.label} changed before file sync")
        os.fsync(fd)
    finally:
        os.close(fd)
    _revalidate_bound_path(snapshot)


@contextmanager
def _config_lock(primary_config: Path) -> Iterator[tuple[int, Path]]:
    """Hold the stable advisory lock used by every cooperating promoter."""

    parent = primary_config.parent
    _require_owned_directory(parent, label="primary config directory")
    lock_path = parent / f".{primary_config.name}.enrollment-promotion.lock"
    base_flags = (
        os.O_RDWR | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        fd = os.open(lock_path, base_flags | os.O_CREAT | os.O_EXCL, 0o600)
        created = True
    except FileExistsError:
        try:
            fd = os.open(lock_path, base_flags)
            created = False
        except OSError as exc:
            raise PromotionError("could not safely open the promotion lock") from exc
    except OSError as exc:
        raise PromotionError("could not safely open the promotion lock") from exc
    try:
        if created:
            os.fchmod(fd, 0o600)
            os.fsync(fd)
            _strict_fsync_directory(parent, label="promotion lock directory")
        opened = os.fstat(fd)
        current = os.lstat(lock_path)
        if (
            not stat.S_ISREG(opened.st_mode)
            or _metadata_signature(opened) != _metadata_signature(current)
            or stat.S_IMODE(opened.st_mode) != 0o600
            or opened.st_uid != os.getuid()
            or opened.st_nlink != 1
        ):
            raise PromotionError("promotion lock is not a private current-user file")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PromotionError("another enrollment promotion holds the config lock") from exc
        try:
            yield fd, lock_path
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                # Closing the descriptor also releases flock. Unlock failure is
                # not a filesystem commit result and must not turn success into
                # a false refusal after the config directory is durable.
                pass
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def _validate_existing_accepted(
    payload: bytes,
    destination: Path,
    *,
    guards: tuple[_BoundRegularPathSnapshot, ...],
) -> _BoundRegularPathSnapshot:
    snapshot, existing = _read_bound_bytes(
        destination, label="existing accepted enrollment"
    )
    _require_private_current_user(snapshot, label="existing accepted enrollment")
    if existing != payload:
        raise PromotionError(
            "existing accepted enrollment does not exactly match the candidate"
        )
    _require_disjoint_inodes((*guards, snapshot))
    for guard in guards:
        _revalidate_bound_path(guard)
    _revalidate_bound_path(snapshot)
    try:
        _strict_fsync_bound_file(snapshot)
        _strict_fsync_directory(destination.parent, label="accepted enrollment directory")
    except OSError as exc:
        raise PromotionAmbiguousError(
            "accepted orphan is exact but its directory durability is ambiguous"
        ) from exc
    return snapshot


def _publish_or_adopt_verified_copy(
    payload: bytes,
    destination: Path,
    *,
    guards: tuple[_BoundRegularPathSnapshot, ...],
) -> tuple[_BoundRegularPathSnapshot, bool]:
    """Exclusively publish or safely adopt one exact independent copy."""

    if os.path.lexists(destination):
        return _validate_existing_accepted(
            payload, destination, guards=guards
        ), True
    _require_owned_directory(destination.parent, label="accepted enrollment directory")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary: Path | None = Path(temporary_name)
    publication_started = False
    adoption_confirmed = False
    adoption_ambiguous = False
    try:
        os.fchmod(fd, 0o600)
        _write_all(fd, payload)
        os.fsync(fd)
        os.close(fd)
        fd = -1
        assert temporary is not None
        temporary_snapshot, copied = _read_bound_bytes(
            temporary, label="temporary accepted enrollment"
        )
        _require_private_current_user(
            temporary_snapshot, label="temporary accepted enrollment"
        )
        if copied != payload:
            raise PromotionError("accepted enrollment copy verification failed")
        for guard in guards:
            _revalidate_bound_path(guard)

        try:
            os.link(temporary, destination)
        except FileExistsError:
            try:
                adopted = _validate_existing_accepted(
                    payload, destination, guards=guards
                )
            except PromotionAmbiguousError:
                adoption_ambiguous = True
                raise
            adoption_confirmed = True
            return adopted, True
        publication_started = True
        try:
            os.unlink(temporary)
            temporary = None
            _strict_fsync_directory(
                destination.parent, label="accepted enrollment directory"
            )
            accepted_snapshot, accepted_bytes = _read_bound_bytes(
                destination, label="published accepted enrollment"
            )
            _require_private_current_user(
                accepted_snapshot, label="published accepted enrollment"
            )
            if accepted_bytes != payload:
                raise PromotionError("published accepted enrollment changed")
            _require_disjoint_inodes((*guards, accepted_snapshot))
        except (OSError, PromotionError) as exc:
            raise PromotionAmbiguousError(
                "accepted enrollment publication may have committed ambiguously"
            ) from exc
        return accepted_snapshot, False
    finally:
        if fd >= 0:
            os.close(fd)
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                if publication_started or adoption_confirmed or adoption_ambiguous:
                    raise PromotionAmbiguousError(
                        "accepted enrollment is durable but temporary-link cleanup failed"
                    ) from exc
                raise
        # Once published, cleanup is complete before the strict directory sync.
        # Never perform a second best-effort sync here.


def _encode_config(data: Mapping[str, object]) -> bytes:
    return json.dumps(data, indent=2).encode("utf-8")


def _atomic_replace_config(
    path: Path,
    payload: bytes,
    *,
    expected: _BoundRegularPathSnapshot,
    guards: tuple[_BoundRegularPathSnapshot, ...],
) -> None:
    """Replace the config and strictly sync its directory.

    Callers must hold ``_config_lock``. An error after ``os.replace`` is an
    ambiguous commit, while any earlier error is safe for staged confirmation.
    """

    parent = path.parent
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=parent
    )
    temporary: Path | None = Path(temporary_name)
    try:
        os.fchmod(fd, 0o600)
        _write_all(fd, payload)
        os.fsync(fd)
        os.close(fd)
        fd = -1
        assert temporary is not None
        temporary_snapshot, copied = _read_bound_bytes(
            temporary, label="temporary primary config"
        )
        _require_private_current_user(
            temporary_snapshot, label="temporary primary config"
        )
        if copied != payload:
            raise PromotionError("temporary primary config verification failed")
        for guard in guards:
            _revalidate_bound_path(guard)
        _revalidate_bound_path(expected)
        os.replace(temporary, path)
        temporary = None
        try:
            _strict_fsync_directory(parent, label="primary config directory")
        except OSError as exc:
            raise PromotionAmbiguousError(
                "primary config replace completed but directory sync failed"
            ) from exc
    finally:
        if fd >= 0:
            os.close(fd)
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _confirm_staged_and_inactive(
    *,
    accepted: Path,
    candidate_bytes: bytes,
    primary_config: Path,
    original_primary: _BoundRegularPathSnapshot,
    guards: tuple[_BoundRegularPathSnapshot, ...],
    source: Path,
) -> bool:
    try:
        accepted_snapshot, accepted_bytes = _read_bound_bytes(
            accepted, label="staged accepted enrollment"
        )
        _require_private_current_user(
            accepted_snapshot, label="staged accepted enrollment"
        )
        if accepted_bytes != candidate_bytes:
            return False
        _require_disjoint_inodes((*guards, accepted_snapshot))
        for guard in guards:
            _revalidate_bound_path(guard)
        _revalidate_bound_path(accepted_snapshot)
        primary_snapshot, primary_data, _ = _read_bound_json(
            primary_config, label=original_primary.label
        )
        if primary_snapshot != original_primary:
            return False
        sherpa = _required_mapping(primary_data, "sherpa", label="primary config")
        if (
            _configured_absolute(
                sherpa, "speaker_enroll_embedding", label="primary config"
            )
            != source
        ):
            return False
        # The primary read above is fallible and adversarially observable. Prove
        # the full six-object state once more after it before claiming exit 3.
        for guard in guards:
            _revalidate_bound_path(guard)
        _revalidate_bound_path(accepted_snapshot)
        _revalidate_bound_path(primary_snapshot)
        return True
    except (OSError, PromotionError):
        return False


def _promote_enrollment(
    *,
    worktree: str | os.PathLike[str],
    primary_config: str | os.PathLike[str],
    expected_candidate: str | os.PathLike[str],
    expected_source_enrollment: str | os.PathLike[str],
    expected_backup: str | os.PathLike[str],
    accepted_enrollment: str | os.PathLike[str],
    accept_live_gate: bool,
) -> PromotionResult:
    if not accept_live_gate:
        raise PromotionError("promotion requires explicit live-gate acceptance")

    worktree_path = _absolute(worktree, label="feature worktree")
    feature_config = worktree_path / "config.local.json"
    primary_config_path = _absolute(primary_config, label="primary config")
    candidate = _absolute(expected_candidate, label="expected candidate")
    source = _absolute(expected_source_enrollment, label="expected source enrollment")
    backup = _absolute(expected_backup, label="expected backup")
    accepted = _absolute(accepted_enrollment, label="accepted enrollment")

    expected_candidate_parent = (
        worktree_path / "pretrained_models" / "sherpa" / "speaker"
    )
    if candidate.parent != expected_candidate_parent:
        raise PromotionError("expected candidate is outside the feature speaker directory")
    if not _CANDIDATE_NAME.fullmatch(candidate.name):
        raise PromotionError("expected candidate is not a v5 candidate name")
    if backup.parent != source.parent:
        raise PromotionError("rollback backup is not adjacent to the historical enrollment")
    if accepted.parent != source.parent:
        raise PromotionError("accepted enrollment must be adjacent to the historical enrollment")
    expected_accepted_name = f"{candidate.stem}-accepted.json"
    if accepted.name != expected_accepted_name:
        raise PromotionError(f"accepted enrollment must be named {expected_accepted_name}")
    protected_paths = (
        feature_config,
        primary_config_path,
        candidate,
        source,
        backup,
        accepted,
    )
    if len(set(protected_paths)) != len(protected_paths):
        raise PromotionError("both configs and all four enrollment paths must differ")
    _require_owned_directory(worktree_path, label="feature worktree")
    _require_owned_directory(candidate.parent, label="candidate directory")
    _require_owned_directory(source.parent, label="historical enrollment directory")

    with _config_lock(primary_config_path) as (_lock_fd, lock_path):
        if lock_path in protected_paths:
            raise PromotionError("promotion lock aliases protected state")
        feature_snapshot, feature_data, _ = _read_bound_json(
            feature_config, label="isolated local config"
        )
        primary_snapshot, primary_data, _ = _read_bound_json(
            primary_config_path, label="primary local config"
        )
        candidate_snapshot, candidate_data, candidate_bytes = _read_bound_json(
            candidate, label="populated v5 candidate"
        )
        source_snapshot, source_data, source_bytes = _read_bound_json(
            source, label="historical enrollment"
        )
        backup_snapshot, _, backup_bytes = _read_bound_json(
            backup, label="rollback enrollment backup"
        )
        state_snapshots = (
            feature_snapshot,
            primary_snapshot,
            candidate_snapshot,
            source_snapshot,
            backup_snapshot,
        )
        _require_disjoint_inodes(state_snapshots)
        _require_private_current_user(feature_snapshot, label="isolated local config")
        _require_private_current_user(candidate_snapshot, label="populated v5 candidate")
        _require_private_current_user(backup_snapshot, label="rollback enrollment backup")
        if source_bytes != backup_bytes:
            raise PromotionError("rollback backup no longer matches the historical enrollment")
        if ENROLLMENT_PREPARATION_KEY in primary_data:
            raise PromotionError(
                "primary config ambiguously carries a prepared enrollment marker"
            )

        marker = feature_data.get(ENROLLMENT_PREPARATION_KEY)
        if not isinstance(marker, Mapping):
            raise PromotionError("isolated config has no prepared enrollment marker")
        if _marker_int(marker, "schema") != ENROLLMENT_PREPARATION_SCHEMA:
            raise PromotionError("prepared marker schema is unsupported")
        if _marker_int(marker, "frontend_version") != ENROLLMENT_FRONTEND_VERSION:
            raise PromotionError("prepared marker is not bound to front-end v5")
        _require_marker_path(marker, "worktree", worktree_path)
        _require_marker_path(marker, "primary_config", primary_config_path)
        _require_marker_path(marker, "candidate", candidate)
        _require_marker_path(marker, "source_enrollment", source)
        _require_marker_path(marker, "backup", backup)
        if _marker_contract(marker, "primary") != _snapshot_marker_contract(
            primary_snapshot
        ):
            raise PromotionError("primary config changed since preparation")
        if _marker_contract(marker, "source") != _snapshot_marker_contract(source_snapshot):
            raise PromotionError("historical enrollment changed since preparation")
        if _marker_contract(marker, "backup") != _snapshot_marker_contract(backup_snapshot):
            raise PromotionError("rollback backup changed since preparation")
        reserved_metadata, reserved_digest = _marker_contract(marker, "candidate")
        if reserved_metadata[2] != 0 or reserved_digest != hashlib.sha256(b"").hexdigest():
            raise PromotionError("prepared marker does not describe an empty reservation")
        if (
            reserved_metadata[5] != 0o600
            or reserved_metadata[6] != os.getuid()
            or reserved_metadata[8] != 1
        ):
            raise PromotionError("prepared candidate reservation was not private")
        if (reserved_metadata[0], reserved_metadata[1]) == (
            candidate_snapshot.dev,
            candidate_snapshot.ino,
        ):
            raise PromotionError("prepared candidate was not atomically populated")
        if candidate_snapshot.size <= 0:
            raise PromotionError("prepared candidate is still empty")

        feature_sherpa = _required_mapping(
            feature_data, "sherpa", label="isolated config"
        )
        primary_sherpa = _required_mapping(
            primary_data, "sherpa", label="primary config"
        )
        if _configured_absolute(
            feature_sherpa, "speaker_enroll_embedding", label="isolated config"
        ) != candidate:
            raise PromotionError("isolated config no longer points at the prepared candidate")
        if _configured_absolute(
            primary_sherpa, "speaker_enroll_embedding", label="primary config"
        ) != source:
            raise PromotionError("primary config no longer points at the historical enrollment")
        feature_model = _configured_absolute(
            feature_sherpa, "speaker_embedding_model", label="isolated config"
        )
        primary_model = _configured_absolute(
            primary_sherpa, "speaker_embedding_model", label="primary config"
        )
        if primary_model != feature_model:
            raise PromotionError("primary and isolated configs name different speaker models")
        feature_sample_rate = _configured_sample_rate(
            feature_sherpa, label="isolated config"
        )
        primary_sample_rate = _configured_sample_rate(
            primary_sherpa, label="primary config"
        )
        if primary_sample_rate != feature_sample_rate:
            raise PromotionError("primary and isolated configs use different sample rates")
        historical_dim = _historical_dimension(
            source_data, expected_model=feature_model
        )
        _validate_candidate_payload(
            candidate_data,
            expected_model=feature_model,
            expected_sample_rate=feature_sample_rate,
            historical_dim=historical_dim,
        )

        accepted_snapshot, accepted_was_adopted = _publish_or_adopt_verified_copy(
            candidate_bytes,
            accepted,
            guards=state_snapshots,
        )
        updated_config = dict(primary_data)
        updated_sherpa = dict(primary_sherpa)
        updated_sherpa["speaker_enroll_embedding"] = str(accepted)
        updated_config["sherpa"] = updated_sherpa
        try:
            _atomic_replace_config(
                primary_config_path,
                _encode_config(updated_config),
                expected=primary_snapshot,
                guards=(*state_snapshots, accepted_snapshot),
            )
        except PromotionAmbiguousError:
            raise
        except (OSError, PromotionError, TypeError, ValueError) as exc:
            if _confirm_staged_and_inactive(
                accepted=accepted,
                candidate_bytes=candidate_bytes,
                primary_config=primary_config_path,
                original_primary=primary_snapshot,
                guards=state_snapshots,
                source=source,
            ):
                raise PromotionStagedError(
                    f"accepted v5 is exactly staged at {accepted}; "
                    f"primary config remains inactive: {exc}"
                ) from exc
            raise PromotionAmbiguousError(
                "promotion failed after staging and exact inactive state could not be confirmed"
            ) from exc

        return PromotionResult(
            primary_config=primary_config_path,
            accepted_enrollment=accepted,
            previous_enrollment=source,
            rollback_backup=backup,
            source_candidate=candidate,
            accepted_was_adopted=accepted_was_adopted,
        )


def promote_enrollment(
    *,
    worktree: str | os.PathLike[str],
    primary_config: str | os.PathLike[str],
    expected_candidate: str | os.PathLike[str],
    expected_source_enrollment: str | os.PathLike[str],
    expected_backup: str | os.PathLike[str],
    accepted_enrollment: str | os.PathLike[str],
    accept_live_gate: bool = False,
) -> PromotionResult:
    """Promote a live-accepted candidate, normalizing low-level failures."""

    try:
        return _promote_enrollment(
            worktree=worktree,
            primary_config=primary_config,
            expected_candidate=expected_candidate,
            expected_source_enrollment=expected_source_enrollment,
            expected_backup=expected_backup,
            accepted_enrollment=accepted_enrollment,
            accept_live_gate=accept_live_gate,
        )
    except PromotionError:
        raise
    except (PreparationError, OSError, TypeError, ValueError) as exc:
        raise PromotionError(f"promotion failed safely: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tools.promote_enrollment",
        description=(
            "Atomically activate one accepted isolated v5 enrollment "
            "(no audio devices)."
        ),
    )
    parser.add_argument("--worktree", required=True)
    parser.add_argument("--primary-config", required=True)
    parser.add_argument("--expected-candidate", required=True)
    parser.add_argument("--expected-source-enrollment", required=True)
    parser.add_argument("--expected-backup", required=True)
    parser.add_argument("--accepted-enrollment", required=True)
    parser.add_argument(
        "--accept-live-gate",
        action="store_true",
        help="confirm that the candidate passed the required manual live gate",
    )
    args = parser.parse_args(argv)
    try:
        result = promote_enrollment(
            worktree=args.worktree,
            primary_config=args.primary_config,
            expected_candidate=args.expected_candidate,
            expected_source_enrollment=args.expected_source_enrollment,
            expected_backup=args.expected_backup,
            accepted_enrollment=args.accepted_enrollment,
            accept_live_gate=args.accept_live_gate,
        )
    except PromotionAmbiguousError as exc:
        print(f"promotion outcome ambiguous: {exc}", file=sys.stderr)
        return 4
    except PromotionStagedError as exc:
        print(f"promotion staged but not activated: {exc}", file=sys.stderr)
        return 3
    except PromotionError as exc:
        print(f"promotion refused: {exc}", file=sys.stderr)
        return 2

    print("enrollment promotion complete (no audio opened; no contents printed)")
    print(f"  active config:     {result.primary_config}")
    accepted_state = "verified existing" if result.accepted_was_adopted else "published"
    print(
        f"  accepted v5:       {result.accepted_enrollment} "
        f"(mode 600, {accepted_state})"
    )
    print(f"  historical source: {result.previous_enrollment} (retained)")
    print(f"  rollback backup:   {result.rollback_backup} (retained)")
    print(f"  isolated candidate: {result.source_candidate} (retained)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
