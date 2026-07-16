#!/usr/bin/env python3
"""Configure optional assistant capabilities without inspecting user data.

This command only records explicit, machine-local grants in
``config.local.json``.  It validates an Obsidian vault as a directory but never
enumerates or reads its notes, and it records exact desktop application IDs
without launching them.

Examples::

    python -m tools.setup_assistant --obsidian-vault ~/dobo-brain/paul-brain
    python -m tools.setup_assistant --enable-reminders
    python -m tools.setup_assistant --trust-app obsidian=md.obsidian.Obsidian.desktop
    python -m tools.setup_assistant --disable-obsidian --untrust-app obsidian
"""
from __future__ import annotations

import argparse
import errno
import json
import os
from pathlib import Path
import re
import secrets
import stat
import sys
from dataclasses import dataclass
from typing import Any, Mapping


DEFAULT_CONFIG = "config.local.json"

_ALIAS_RE = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
_DESKTOP_ID_RE = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]{0,254}\.desktop\Z"
)


class SetupError(RuntimeError):
    """A safe setup transaction could not be completed."""


@dataclass(frozen=True)
class TrustedApp:
    """One exact, setup-approved desktop application."""

    alias: str
    desktop_id: str


@dataclass(frozen=True)
class SetupRequest:
    """Capability changes requested for one atomic publication."""

    obsidian_vault: str | None = None
    disable_obsidian: bool = False
    reminders_enabled: bool | None = None
    trust_apps: tuple[TrustedApp, ...] = ()
    untrust_apps: tuple[str, ...] = ()

    @property
    def has_changes(self) -> bool:
        return bool(
            self.obsidian_vault is not None
            or self.disable_obsidian
            or self.reminders_enabled is not None
            or self.trust_apps
            or self.untrust_apps
        )


def _validate_alias(value: str) -> str:
    if not _ALIAS_RE.fullmatch(value):
        raise SetupError(
            "app alias must start with a lowercase letter and contain only "
            "lowercase letters, digits, '_' or '-' (maximum 64 characters)"
        )
    return value


def validate_desktop_id(value: str) -> str:
    """Return a safe, exact freedesktop desktop ID.

    A desktop ID is a basename, not a path or shell command.  Setup deliberately
    does not probe or launch it; the runtime connector remains responsible for
    resolving the exact allowlisted ID when the user asks to open the app.
    """
    if (
        not _DESKTOP_ID_RE.fullmatch(value)
        or ".." in value
        or "/" in value
        or "\\" in value
    ):
        raise SetupError(
            "desktop ID must be an exact safe basename ending in '.desktop'"
        )
    return value


def parse_trusted_app(value: str) -> TrustedApp:
    """Parse ``alias=desktop-id`` without performing application I/O."""
    if value.count("=") != 1:
        raise SetupError("trusted app must use alias=desktop-id")
    alias, desktop_id = value.split("=", 1)
    return TrustedApp(
        alias=_validate_alias(alias),
        desktop_id=validate_desktop_id(desktop_id),
    )


def validate_vault_root(value: str) -> str:
    """Validate a vault directory and return its canonical absolute path.

    The check opens only the directory itself.  It never lists entries, looks
    for Obsidian metadata, or reads any note.
    """
    expanded = os.path.abspath(os.path.expanduser(value))
    try:
        canonical = os.path.realpath(expanded, strict=True)
    except (OSError, RuntimeError) as exc:
        raise SetupError(f"Obsidian vault is not accessible: {value}") from exc

    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise SetupError("safe vault validation is not supported on this platform")

    flags = os.O_RDONLY | nofollow | directory | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(canonical, flags)
    except OSError as exc:
        raise SetupError(f"Obsidian vault is not a safe readable directory: {value}") from exc
    try:
        if not stat.S_ISDIR(os.fstat(fd).st_mode):
            raise SetupError(f"Obsidian vault is not a directory: {value}")
    finally:
        os.close(fd)
    return canonical


def _copy_object(value: Any, *, section: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SetupError(f"existing '{section}' configuration must be a JSON object")
    return dict(value)


def apply_setup(
    config: Mapping[str, Any],
    request: SetupRequest,
    *,
    validated_vault: str | None = None,
) -> dict[str, Any]:
    """Merge requested changes while preserving every omitted key and section."""
    if not isinstance(config, Mapping):
        raise SetupError("existing configuration must be a JSON object")
    if request.obsidian_vault is not None and request.disable_obsidian:
        raise SetupError("cannot enable and disable Obsidian in the same request")

    result = dict(config)

    if request.obsidian_vault is not None or request.disable_obsidian:
        obsidian = _copy_object(result.get("obsidian", {}), section="obsidian")
        if request.obsidian_vault is not None:
            vault = validated_vault or validate_vault_root(request.obsidian_vault)
            obsidian["enabled"] = True
            obsidian["vault_root"] = vault
        else:
            obsidian["enabled"] = False
        result["obsidian"] = obsidian

    if request.reminders_enabled is not None:
        reminders = _copy_object(result.get("reminders", {}), section="reminders")
        reminders["enabled"] = request.reminders_enabled
        result["reminders"] = reminders

    if request.trust_apps or request.untrust_apps:
        trusted = _copy_object(result.get("trusted_apps", {}), section="trusted_apps")
        apps = _copy_object(trusted.get("apps", {}), section="trusted_apps.apps")
        for alias in request.untrust_apps:
            apps.pop(alias, None)
        for app in request.trust_apps:
            apps[app.alias] = {
                "connector": "desktop_launch",
                "desktop_id": app.desktop_id,
                "operations": ["open"],
            }
        trusted["apps"] = apps
        if request.trust_apps:
            trusted["enabled"] = True
        elif not apps:
            trusted["enabled"] = False
        result["trusted_apps"] = trusted

    return result


def _signature(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _open_parent(path: str) -> tuple[int, str, str]:
    """Open and pin ``path``'s parent without following any symlink component."""
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise SetupError("safe config publication is not supported on this platform")

    absolute = os.path.abspath(os.path.expanduser(path))
    parsed = Path(absolute)
    name = parsed.name
    if not name or name in {".", ".."}:
        raise SetupError("config path must name a file")

    flags = os.O_RDONLY | nofollow | directory | getattr(os, "O_CLOEXEC", 0)
    parts = parsed.parts
    if not parts or parts[0] != os.path.sep:
        raise SetupError("config path could not be made absolute")

    try:
        current = os.open(os.path.sep, flags)
    except OSError as exc:
        raise SetupError("could not open the config path root safely") from exc
    try:
        for component in parts[1:-1]:
            try:
                next_fd = os.open(component, flags, dir_fd=current)
            except OSError as exc:
                raise SetupError(
                    "config parent must exist and contain no symlink components"
                ) from exc
            os.close(current)
            current = next_fd
        return current, name, absolute
    except BaseException:
        os.close(current)
        raise


def _read_existing(parent_fd: int, name: str) -> tuple[dict[str, Any], tuple[int, ...] | None]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(name, flags, dir_fd=parent_fd)
    except FileNotFoundError:
        return {}, None
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.EMLINK}:
            raise SetupError("config path must not be a symlink") from exc
        raise SetupError("existing config could not be opened safely") from exc

    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise SetupError("config path must be a regular file")
        try:
            with os.fdopen(fd, "r", encoding="utf-8", closefd=False) as stream:
                loaded = json.load(stream)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SetupError("existing config is not valid UTF-8 JSON") from exc
        after = os.fstat(fd)
        if _signature(before) != _signature(after):
            raise SetupError("existing config changed while it was being read")
        if not isinstance(loaded, dict):
            raise SetupError("existing configuration must be a JSON object")
        return loaded, _signature(after)
    finally:
        os.close(fd)


def _current_signature(parent_fd: int, name: str) -> tuple[int, ...] | None:
    try:
        info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise SetupError("could not verify the config publication target") from exc
    if not stat.S_ISREG(info.st_mode):
        raise SetupError("config publication target must be a regular file")
    return _signature(info)


_SIGNATURE_NOT_SUPPLIED = object()


def publish_config_atomic(
    config: Mapping[str, Any],
    path: str = DEFAULT_CONFIG,
    *,
    _expected_signature: object = _SIGNATURE_NOT_SUPPLIED,
) -> str:
    """Publish a mode-0600 regular JSON file without following symlinks."""
    parent_fd, name, absolute = _open_parent(path)
    temp_name: str | None = None
    temp_fd: int | None = None
    try:
        _old, original_signature = _read_existing(parent_fd, name)
        if (
            _expected_signature is not _SIGNATURE_NOT_SUPPLIED
            and original_signature != _expected_signature
        ):
            raise SetupError("config changed concurrently; no changes were published")
        temp_name = f".{name}.{secrets.token_hex(8)}.tmp"
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            temp_fd = os.open(temp_name, flags, 0o600, dir_fd=parent_fd)
            os.fchmod(temp_fd, 0o600)
            with os.fdopen(temp_fd, "w", encoding="utf-8", closefd=False) as stream:
                json.dump(config, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(temp_fd)
        except (OSError, TypeError, ValueError) as exc:
            raise SetupError("could not serialize the new config safely") from exc
        finally:
            if temp_fd is not None:
                os.close(temp_fd)
                temp_fd = None

        if _current_signature(parent_fd, name) != original_signature:
            raise SetupError("config changed concurrently; no changes were published")
        try:
            os.replace(
                temp_name,
                name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
        except OSError as exc:
            raise SetupError("atomic config publication failed") from exc
        temp_name = None
        try:
            os.fsync(parent_fd)
        except OSError:
            # The complete regular file is already published.  Some filesystems
            # reject directory fsync, so this cannot safely be reported as a
            # failed (and supposedly non-mutating) transaction.
            pass
        return absolute
    finally:
        if temp_fd is not None:
            os.close(temp_fd)
        if temp_name is not None:
            try:
                os.unlink(temp_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        os.close(parent_fd)


def configure_assistant(request: SetupRequest, *, config_path: str = DEFAULT_CONFIG) -> str:
    """Validate, merge, and atomically publish one setup request."""
    if not request.has_changes:
        raise SetupError("no capability changes were requested")

    vault = None
    if request.obsidian_vault is not None:
        vault = validate_vault_root(request.obsidian_vault)

    parent_fd, name, _absolute = _open_parent(config_path)
    try:
        existing, original_signature = _read_existing(parent_fd, name)
    finally:
        os.close(parent_fd)
    merged = apply_setup(existing, request, validated_vault=vault)
    return publish_config_atomic(
        merged,
        config_path,
        _expected_signature=original_signature,
    )


def _request_from_args(args: argparse.Namespace) -> SetupRequest:
    trusted = tuple(parse_trusted_app(value) for value in args.trust_app)
    untrusted = tuple(_validate_alias(value) for value in args.untrust_app)

    trusted_aliases = [app.alias for app in trusted]
    if len(trusted_aliases) != len(set(trusted_aliases)):
        raise SetupError("each app alias may be trusted only once per setup command")
    if len(untrusted) != len(set(untrusted)):
        raise SetupError("each app alias may be untrusted only once per setup command")
    overlap = set(trusted_aliases).intersection(untrusted)
    if overlap:
        raise SetupError(
            f"cannot trust and untrust the same app alias: {sorted(overlap)[0]}"
        )

    reminders_enabled: bool | None = None
    if args.enable_reminders:
        reminders_enabled = True
    elif args.disable_reminders:
        reminders_enabled = False

    return SetupRequest(
        obsidian_vault=args.obsidian_vault,
        disable_obsidian=args.disable_obsidian,
        reminders_enabled=reminders_enabled,
        trust_apps=trusted,
        untrust_apps=untrusted,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Configure optional capabilities for the unified voice assistant",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"machine-local config path (default: {DEFAULT_CONFIG})",
    )
    obsidian = parser.add_mutually_exclusive_group()
    obsidian.add_argument(
        "--obsidian-vault",
        metavar="PATH",
        help="enable Obsidian access for this exact vault directory",
    )
    obsidian.add_argument(
        "--disable-obsidian",
        action="store_true",
        help="disable Obsidian access while preserving its other settings",
    )
    reminders = parser.add_mutually_exclusive_group()
    reminders.add_argument(
        "--enable-reminders",
        action="store_true",
        help="enable the reminders capability",
    )
    reminders.add_argument(
        "--disable-reminders",
        action="store_true",
        help="disable the reminders capability",
    )
    parser.add_argument(
        "--trust-app",
        action="append",
        default=[],
        metavar="ALIAS=DESKTOP-ID",
        help="allow the assistant to open one exact desktop app (repeatable)",
    )
    parser.add_argument(
        "--untrust-app",
        action="append",
        default=[],
        metavar="ALIAS",
        help="remove one trusted desktop app alias (repeatable)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        request = _request_from_args(args)
    except SetupError as exc:
        parser.error(str(exc))

    if not request.has_changes:
        print("No capability options supplied; config.local.json was not touched.")
        return 0

    try:
        published = configure_assistant(request, config_path=args.config)
    except SetupError as exc:
        print(f"Assistant setup failed: {exc}", file=sys.stderr)
        return 1

    enabled: list[str] = []
    if request.obsidian_vault is not None:
        enabled.append("Obsidian")
    if request.reminders_enabled is True:
        enabled.append("reminders")
    if request.trust_apps:
        enabled.append("trusted apps")
    print(f"Assistant capability config published safely: {published}")
    if enabled:
        print(f"Enabled/updated: {', '.join(enabled)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
