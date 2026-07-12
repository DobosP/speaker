"""Prepare an isolated, fail-closed speaker-enrollment destination.

This command is deliberately device-free: it never imports ``sounddevice`` or
opens an audio stream.  It exists for feature worktrees whose ignored
``config.local.json`` was symlinked to the primary checkout.  Enrollment must
not follow that link and overwrite the primary config/reference while a live
candidate is still under test.

The preparation transaction:

* verifies the feature config link points at the explicitly named primary file;
* makes a mode-600, verified, no-clobber backup of the current enrollment;
* reserves a unique feature-local v5 candidate file; and
* atomically replaces the feature symlink, as the final publish, with a
  mode-600 regular config clone already wired to that candidate.

No config value, enrollment byte, embedding, or digest is printed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from core.enroll import (
    ENROLLMENT_FRONTEND_VERSION,
    ENROLLMENT_PREPARATION_KEY,
    ENROLLMENT_PREPARATION_SCHEMA,
    _fsync_directory,
)


class PreparationError(RuntimeError):
    """The requested preparation was ambiguous or unsafe."""


@dataclass(frozen=True)
class PreparationResult:
    config_local: Path
    backup: Path
    candidate: Path


_CANDIDATE_NAME = re.compile(r"enrollment\.v5-[A-Za-z0-9][A-Za-z0-9._-]*\.json\Z")


def _mode(path: Path) -> int:
    return stat.S_IMODE(os.lstat(path).st_mode)


def _require_absolute(path: str | os.PathLike[str], *, label: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise PreparationError(f"{label} must be an absolute path")
    return candidate


def _regular_stat(path: Path, *, label: str) -> os.stat_result:
    try:
        info = os.lstat(path)
    except FileNotFoundError as exc:
        raise PreparationError(f"missing {label}: {path}") from exc
    if stat.S_ISLNK(info.st_mode):
        raise PreparationError(f"refusing symlink for {label}: {path}")
    if not stat.S_ISREG(info.st_mode):
        raise PreparationError(f"{label} is not a regular file: {path}")
    return info


def _directory_stat(path: Path, *, label: str) -> os.stat_result:
    try:
        info = os.lstat(path)
    except FileNotFoundError as exc:
        raise PreparationError(f"missing {label}: {path}") from exc
    if stat.S_ISLNK(info.st_mode):
        raise PreparationError(f"refusing symlink for {label}: {path}")
    if not stat.S_ISDIR(info.st_mode):
        raise PreparationError(f"{label} is not a directory: {path}")
    return info


def _open_regular(path: Path, *, label: str) -> tuple[int, os.stat_result]:
    before = _regular_stat(path, label=label)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise PreparationError(f"could not safely open {label}: {path}") from exc
    opened = os.fstat(fd)
    if (
        not stat.S_ISREG(opened.st_mode)
        or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
    ):
        os.close(fd)
        raise PreparationError(f"{label} changed while opening: {path}")
    return fd, opened


def _read_regular_bytes(path: Path, *, label: str) -> tuple[bytes, os.stat_result]:
    fd, opened = _open_regular(path, label=label)
    try:
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    current = _regular_stat(path, label=label)
    if (
        (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        != (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
        or (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino)
    ):
        raise PreparationError(f"{label} changed while reading: {path}")
    return b"".join(chunks), opened


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("short write")
        view = view[written:]


def _verified_exclusive_backup(source: Path, backup: Path) -> None:
    """Publish a verified backup without replacing any existing path."""
    if os.path.lexists(backup):
        raise PreparationError(f"backup path already exists: {backup}")
    _directory_stat(backup.parent, label="backup directory")

    source_bytes, source_info = _read_regular_bytes(
        source, label="existing enrollment"
    )
    source_digest = hashlib.sha256(source_bytes).digest()
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{backup.name}.", suffix=".tmp", dir=backup.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, 0o600)
        _write_all(fd, source_bytes)
        os.fsync(fd)
        os.close(fd)
        fd = -1

        copied, _ = _read_regular_bytes(temporary, label="temporary backup")
        if hashlib.sha256(copied).digest() != source_digest:
            raise PreparationError("backup verification failed before publish")
        current = _regular_stat(source, label="existing enrollment")
        if (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns) != (
            source_info.st_dev,
            source_info.st_ino,
            source_info.st_size,
            source_info.st_mtime_ns,
        ):
            raise PreparationError("existing enrollment changed during backup")

        try:
            os.link(temporary, backup)
        except FileExistsError as exc:
            raise PreparationError(f"backup path already exists: {backup}") from exc
        published, _ = _read_regular_bytes(backup, label="published backup")
        if hashlib.sha256(published).digest() != source_digest:
            raise PreparationError("published backup verification failed")
        _fsync_directory(str(backup.parent))
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _resolved_link_target(link: Path) -> Path:
    raw = os.readlink(link)
    target = Path(raw)
    if not target.is_absolute():
        target = link.parent / target
    try:
        return target.resolve(strict=True)
    except FileNotFoundError as exc:
        raise PreparationError(f"config symlink target is missing: {link}") from exc


def _publish_isolated_config(
    link: Path,
    *,
    link_info: os.stat_result,
    expected_target: Path,
    target_info: os.stat_result,
    payload: bytes,
) -> None:
    """Atomically replace only ``link`` with the verified, already-wired config."""
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{link.name}.", suffix=".tmp", dir=link.parent
    )
    temporary: Path | None = Path(temporary_name)
    payload_digest = hashlib.sha256(payload).digest()
    try:
        os.fchmod(fd, 0o600)
        _write_all(fd, payload)
        os.fsync(fd)
        os.close(fd)
        fd = -1
        assert temporary is not None
        copied, _ = _read_regular_bytes(temporary, label="temporary config copy")
        if hashlib.sha256(copied).digest() != payload_digest:
            raise PreparationError("isolated config byte-copy verification failed")

        current_link = os.lstat(link)
        if (
            not stat.S_ISLNK(current_link.st_mode)
            or (current_link.st_dev, current_link.st_ino)
            != (link_info.st_dev, link_info.st_ino)
            or _resolved_link_target(link) != expected_target
        ):
            raise PreparationError("feature config symlink changed before publish")
        current_target = _regular_stat(expected_target, label="primary local config")
        if (
            current_target.st_dev,
            current_target.st_ino,
            current_target.st_size,
            current_target.st_mtime_ns,
        ) != (
            target_info.st_dev,
            target_info.st_ino,
            target_info.st_size,
            target_info.st_mtime_ns,
        ):
            raise PreparationError("primary local config changed during preparation")

        os.replace(temporary, link)
        temporary = None
        isolated, _ = _read_regular_bytes(link, label="isolated local config")
        if hashlib.sha256(isolated).digest() != payload_digest:
            raise PreparationError("isolated config verification failed after publish")
        _fsync_directory(str(link.parent))
    finally:
        if fd >= 0:
            os.close(fd)
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _ensure_candidate_parent(worktree: Path, parent: Path) -> None:
    try:
        relative = parent.relative_to(worktree)
    except ValueError as exc:
        raise PreparationError("candidate parent escaped the feature worktree") from exc
    current = worktree
    for part in relative.parts:
        current = current / part
        if os.path.lexists(current):
            _directory_stat(current, label="candidate directory")
        else:
            os.mkdir(current, 0o700)


def _reserve_candidate(path: Path) -> None:
    if os.path.lexists(path):
        raise PreparationError(f"candidate path already exists: {path}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise PreparationError(f"candidate path already exists: {path}") from exc
    try:
        os.fchmod(fd, 0o600)
        os.fsync(fd)
    finally:
        os.close(fd)
    _fsync_directory(str(path.parent))


def prepare_enrollment(
    *,
    worktree: str | os.PathLike[str],
    expected_config_target: str | os.PathLike[str],
    expected_enrollment: str | os.PathLike[str],
    backup: str | os.PathLike[str],
    candidate_name: str,
) -> PreparationResult:
    """Prepare isolated enrollment state without opening an audio device."""
    worktree_path = Path(os.path.abspath(os.fspath(worktree)))
    _directory_stat(worktree_path, label="feature worktree")
    worktree_path = worktree_path.resolve(strict=True)
    config_link = worktree_path / "config.local.json"
    try:
        link_info = os.lstat(config_link)
    except FileNotFoundError as exc:
        raise PreparationError(f"missing feature config symlink: {config_link}") from exc
    if not stat.S_ISLNK(link_info.st_mode):
        raise PreparationError(
            f"feature config must be an unprepared symlink: {config_link}"
        )

    config_target = _require_absolute(
        expected_config_target, label="expected config target"
    )
    target_info = _regular_stat(config_target, label="primary local config")
    config_target = config_target.resolve(strict=True)
    if _resolved_link_target(config_link) != config_target:
        raise PreparationError("feature config symlink does not match expected target")

    config_bytes, reread_target_info = _read_regular_bytes(
        config_target, label="primary local config"
    )
    if (target_info.st_dev, target_info.st_ino) != (
        reread_target_info.st_dev,
        reread_target_info.st_ino,
    ):
        raise PreparationError("primary local config changed during validation")
    try:
        config_data = json.loads(config_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreparationError("primary local config is not valid UTF-8 JSON") from exc
    if not isinstance(config_data, dict):
        raise PreparationError("primary local config must contain a JSON object")
    sherpa = config_data.get("sherpa")
    if not isinstance(sherpa, dict):
        raise PreparationError("primary local config has no sherpa JSON object")

    enrollment_path = _require_absolute(
        expected_enrollment, label="expected enrollment"
    )
    _regular_stat(enrollment_path, label="existing enrollment")
    enrollment_path = enrollment_path.resolve(strict=True)
    enrollment_bytes, source_info = _read_regular_bytes(
        enrollment_path, label="existing enrollment"
    )
    try:
        enrollment_data = json.loads(enrollment_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreparationError("existing enrollment is not valid UTF-8 JSON") from exc
    if (
        not isinstance(enrollment_data, dict)
        or not isinstance(enrollment_data.get("embedding"), list)
        or not enrollment_data["embedding"]
    ):
        raise PreparationError("existing enrollment has no non-empty embedding")
    configured_enrollment = sherpa.get("speaker_enroll_embedding")
    if not isinstance(configured_enrollment, str) or not configured_enrollment:
        raise PreparationError("primary config has no speaker enrollment path")
    configured_path = Path(configured_enrollment)
    if not configured_path.is_absolute() or configured_path != enrollment_path:
        raise PreparationError(
            "configured speaker enrollment does not match the expected path"
        )

    backup_path = _require_absolute(backup, label="backup")
    try:
        backup_parent = backup_path.parent.resolve(strict=True)
    except FileNotFoundError as exc:
        raise PreparationError("backup directory is missing") from exc
    backup_path = backup_parent / backup_path.name
    if backup_path.parent != enrollment_path.parent:
        raise PreparationError("backup must be adjacent to the existing enrollment")
    if backup_path == enrollment_path:
        raise PreparationError("backup path must differ from the existing enrollment")
    if os.path.lexists(backup_path):
        raise PreparationError(f"backup path already exists: {backup_path}")
    _directory_stat(backup_path.parent, label="backup directory")

    if not _CANDIDATE_NAME.fullmatch(candidate_name):
        raise PreparationError(
            "candidate name must match enrollment.v5-<unique-id>.json"
        )
    candidate = (
        worktree_path
        / "pretrained_models"
        / "sherpa"
        / "speaker"
        / candidate_name
    )
    if os.path.lexists(candidate):
        raise PreparationError(f"candidate path already exists: {candidate}")
    # Validate every already-existing ancestor before creating anything.
    current = worktree_path
    for part in candidate.parent.relative_to(worktree_path).parts:
        current = current / part
        if os.path.lexists(current):
            _directory_stat(current, label="candidate directory")

    # The config link is the safety boundary. Publish it LAST and already wired:
    # every earlier failure leaves run_enrollment's symlink guard in place, while
    # every later failure leaves a regular config naming only the isolated file.
    _verified_exclusive_backup(enrollment_path, backup_path)
    current_source = _regular_stat(
        enrollment_path, label="existing enrollment"
    )
    if (
        current_source.st_dev,
        current_source.st_ino,
        current_source.st_size,
        current_source.st_mtime_ns,
    ) != (
        source_info.st_dev,
        source_info.st_ino,
        source_info.st_size,
        source_info.st_mtime_ns,
    ):
        raise PreparationError("existing enrollment changed before final publish")
    _ensure_candidate_parent(worktree_path, candidate.parent)
    _reserve_candidate(candidate)
    candidate_info = _regular_stat(candidate, label="reserved v5 candidate")
    backup_info = _regular_stat(backup_path, label="published backup")
    wired_config = dict(config_data)
    wired_sherpa = dict(sherpa)
    wired_sherpa["speaker_enroll_embedding"] = str(candidate)
    wired_config["sherpa"] = wired_sherpa
    wired_config[ENROLLMENT_PREPARATION_KEY] = {
        "schema": ENROLLMENT_PREPARATION_SCHEMA,
        "frontend_version": ENROLLMENT_FRONTEND_VERSION,
        "worktree": str(worktree_path),
        "candidate": str(candidate),
        "backup": str(backup_path),
        "source_enrollment": str(enrollment_path),
        "candidate_dev": candidate_info.st_dev,
        "candidate_ino": candidate_info.st_ino,
        "candidate_size": candidate_info.st_size,
        "candidate_mtime_ns": candidate_info.st_mtime_ns,
        "backup_dev": backup_info.st_dev,
        "backup_ino": backup_info.st_ino,
        "backup_size": backup_info.st_size,
        "backup_mtime_ns": backup_info.st_mtime_ns,
        "source_dev": source_info.st_dev,
        "source_ino": source_info.st_ino,
        "source_size": source_info.st_size,
        "source_mtime_ns": source_info.st_mtime_ns,
    }
    wired_bytes = json.dumps(wired_config, indent=2).encode("utf-8")
    _publish_isolated_config(
        config_link,
        link_info=link_info,
        expected_target=config_target,
        target_info=target_info,
        payload=wired_bytes,
    )

    if config_link.is_symlink() or not config_link.is_file() or _mode(config_link) != 0o600:
        raise PreparationError("isolated config did not finish as a mode-600 regular file")
    if candidate.is_symlink() or not candidate.is_file() or _mode(candidate) != 0o600:
        raise PreparationError("candidate reservation is not a mode-600 regular file")
    if candidate.stat().st_size != 0:
        raise PreparationError("new candidate reservation was unexpectedly non-empty")
    if _mode(backup_path) != 0o600:
        raise PreparationError("verified backup is not mode 600")

    try:
        isolated_bytes, _ = _read_regular_bytes(
            config_link, label="isolated local config"
        )
        isolated = json.loads(isolated_bytes.decode("utf-8"))
        wired = isolated["sherpa"]["speaker_enroll_embedding"]
    except (
        OSError,
        KeyError,
        TypeError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise PreparationError("could not verify isolated config wiring") from exc
    if wired != str(candidate):
        raise PreparationError("isolated config did not retain the candidate path")

    return PreparationResult(
        config_local=config_link,
        backup=backup_path,
        candidate=candidate,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tools.prepare_enrollment",
        description="Prepare an isolated v5 enrollment destination (no audio devices).",
    )
    parser.add_argument("--worktree", default=".")
    parser.add_argument("--expected-config-target", required=True)
    parser.add_argument("--expected-enrollment", required=True)
    parser.add_argument("--backup", required=True)
    parser.add_argument("--candidate-name", required=True)
    args = parser.parse_args(argv)
    try:
        result = prepare_enrollment(
            worktree=args.worktree,
            expected_config_target=args.expected_config_target,
            expected_enrollment=args.expected_enrollment,
            backup=args.backup,
            candidate_name=args.candidate_name,
        )
    except (PreparationError, OSError, ValueError) as exc:
        print(f"preparation refused: {exc}", file=sys.stderr)
        return 2

    print("enrollment preparation complete (no audio opened; no contents printed)")
    print(f"  isolated config: {result.config_local} (regular, mode 600)")
    print(f"  verified backup: {result.backup} (exclusive, mode 600)")
    print(f"  v5 candidate:    {result.candidate} (reserved, mode 600)")
    command = (
        f"cd {shlex.quote(str(result.config_local.parent))} && "
        f"{shlex.quote(sys.executable)} -m core --engine sherpa "
        "--enroll --require-prepared-enrollment "
        "--enroll-seconds 12 --enroll-passes 3"
    )
    print("  next (opens the microphone):")
    print(f"    {command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
