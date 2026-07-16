"""One-command private Linux live-session launcher.

``./live.sh`` calls this module.  It keeps platform setup out of the portable
``core`` entry point while making the normal on-machine workflow one command:

* reuse a healthy Ollama server, or start one for this session;
* reuse the canonical PipeWire echo-cancel nodes, or create them temporarily;
* select the echo-cancel source/sink and require the shared doctor to pass;
* run sherpa with DEBUG, mic recording, and an aligned playback reference;
* restore defaults and stop/unload only resources created by this launcher.

The orchestration is deliberately dependency-injected so its failure and
cleanup paths are covered without opening an audio device or contacting Ollama.
"""
from __future__ import annotations

import argparse
import ipaddress
import os
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import IO, Mapping, Protocol, Sequence
from urllib.parse import urlsplit

try:  # Imported lazily in effect: the module still gives a useful error off Linux.
    import fcntl
except ImportError:  # pragma: no cover - exercised on non-POSIX hosts
    fcntl = None  # type: ignore[assignment]

try:
    import pwd
except ImportError:  # pragma: no cover - Windows help/import path
    pwd = None  # type: ignore[assignment]

_EC_SOURCE = "echo-cancel-source"
_EC_SINK = "echo-cancel-sink"
_OLLAMA_START_TIMEOUT_SEC = 15.0
_OLLAMA_STOP_TIMEOUT_SEC = 5.0
_ROUTE_START_TIMEOUT_SEC = 5.0
_DOCTOR_TIMEOUT_SEC = 60.0
_VOICE_INTERRUPT_GRACE_SEC = 20.0
_VOICE_TERM_GRACE_SEC = 10.0
_VOICE_KILL_GRACE_SEC = 2.0


class LauncherError(RuntimeError):
    """Fail-closed setup error with an operator-facing message."""


class LauncherInterrupted(LauncherError):
    def __init__(self, signum: int):
        super().__init__(f"received signal {signum}")
        self.signum = signum


class ChildProcess(Protocol):
    pid: int

    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    def send_signal(self, sig: int) -> None: ...


class Operations(Protocol):
    platform: str

    def which(self, name: str) -> str | None: ...

    def run(
        self,
        command: Sequence[str],
        *,
        capture: bool = False,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]: ...

    def popen(
        self,
        command: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        stdout: int | IO[bytes] | IO[str] | None = None,
        stderr: int | IO[bytes] | IO[str] | None = None,
        start_new_session: bool = False,
        child_signal_mask: set[int] | None = None,
    ) -> ChildProcess: ...

    def sleep(self, seconds: float) -> None: ...

    def signal_group(self, process: ChildProcess, signum: int) -> None: ...


class RealOperations:
    platform = sys.platform

    def which(self, name: str) -> str | None:
        return shutil.which(name)

    def run(
        self,
        command: Sequence[str],
        *,
        capture: bool = False,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            list(command),
            check=False,
            text=True,
            capture_output=capture,
            env=dict(env) if env is not None else None,
            timeout=timeout,
        )

    def popen(
        self,
        command: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        stdout: int | IO[bytes] | IO[str] | None = None,
        stderr: int | IO[bytes] | IO[str] | None = None,
        start_new_session: bool = False,
        child_signal_mask: set[int] | None = None,
    ) -> ChildProcess:
        preexec_fn = None
        if child_signal_mask is not None and hasattr(signal, "pthread_sigmask"):
            inherited_mask = set(child_signal_mask)

            def restore_child_signal_mask() -> None:
                signal.pthread_sigmask(signal.SIG_SETMASK, inherited_mask)

            preexec_fn = restore_child_signal_mask
        return subprocess.Popen(
            list(command),
            env=dict(env) if env is not None else None,
            stdout=stdout,
            stderr=stderr,
            start_new_session=start_new_session,
            preexec_fn=preexec_fn,
        )

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def signal_group(self, process: ChildProcess, signum: int) -> None:
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass


def _lifecycle_signals() -> set[int]:
    signals = {signal.SIGINT, signal.SIGTERM}
    if hasattr(signal, "SIGHUP"):
        signals.add(signal.SIGHUP)
    return signals


def _block_lifecycle_signals():
    if not hasattr(signal, "pthread_sigmask"):
        return None
    try:
        return signal.pthread_sigmask(signal.SIG_BLOCK, _lifecycle_signals())
    except (OSError, ValueError):
        return None


def _restore_signal_mask(old_mask) -> None:
    if old_mask is not None:
        signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)


@dataclass
class LiveSessionLock:
    handle: IO[str]

    @classmethod
    def acquire(cls, path: Path) -> "LiveSessionLock":
        if fcntl is None:
            raise LauncherError("the live-session lock requires Linux/POSIX flock")
        handle: IO[str] | None = None
        fd: int | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(path, flags, 0o600)
            metadata = os.fstat(fd)
            if not stat.S_ISREG(metadata.st_mode):
                raise LauncherError(f"live-session lock is not a regular file: {path}")
            os.fchmod(fd, 0o600)
            handle = os.fdopen(fd, "a+", encoding="utf-8")
            fd = None
        except OSError as exc:
            if handle is not None:
                handle.close()
            elif fd is not None:
                os.close(fd)
            raise LauncherError(f"could not open live-session lock {path}: {exc}") from exc
        except BaseException:
            if handle is not None:
                handle.close()
            elif fd is not None:
                os.close(fd)
            raise
        assert handle is not None
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise LauncherError(
                "another ./live.sh session already owns the audio setup"
            ) from exc
        except BaseException:
            handle.close()
            raise
        return cls(handle)

    def close(self) -> None:
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()


def _default_lock_path() -> Path:
    if pwd is not None and hasattr(os, "getuid"):
        home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    else:  # pragma: no cover - launcher rejects non-Linux before host mutation
        home = Path.home()
    return home / ".local/state/speaker/live-session.lock"


def _checked_text(
    ops: Operations,
    command: Sequence[str],
    *,
    label: str,
    allow_empty: bool = False,
) -> str:
    try:
        result = ops.run(command, capture=True, timeout=5.0)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LauncherError(f"could not inspect {label}: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "command failed").strip()
        raise LauncherError(f"could not inspect {label}: {detail}")
    value = (result.stdout or "").strip()
    if not value and not allow_empty:
        raise LauncherError(f"could not inspect {label}: empty result")
    return value


def _run_checked(
    ops: Operations, command: Sequence[str], *, label: str
) -> subprocess.CompletedProcess[str]:
    try:
        result = ops.run(command, capture=True, timeout=5.0)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LauncherError(f"{label} failed: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "command failed").strip()
        raise LauncherError(f"{label} failed: {detail}")
    return result


def _short_names(text: str) -> set[str]:
    names: set[str] = set()
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 2:
            names.add(fields[1])
    return names


def _short_name_count(text: str, name: str) -> int:
    return sum(
        1
        for line in text.splitlines()
        if len((fields := line.split())) >= 2 and fields[1] == name
    )


def _looks_like_echo_cancel(name: object) -> bool:
    value = str(name or "").strip().lower().replace("_", "-").replace(".", "-")
    return (
        ("echo" in value and "cancel" in value)
        or value.startswith(("ec-source", "ec-sink", "aec-source", "aec-sink"))
    )


@dataclass(frozen=True)
class _EchoModule:
    module_id: str
    arguments: Mapping[str, str]


def _parse_module_arguments(raw: str) -> dict[str, str]:
    try:
        tokens = shlex.split(raw)
    except ValueError as exc:
        raise LauncherError(f"malformed module-echo-cancel arguments: {exc}") from exc
    parsed: dict[str, str] = {}
    for token in tokens:
        key, separator, value = token.partition("=")
        if not separator or not key:
            raise LauncherError(
                f"malformed module-echo-cancel argument token {token!r}"
            )
        # PipeWire-Pulse removes quoting around space-separated aec_args in
        # `list short modules`. Only dotted WebRTC keys are continuations.
        if key.startswith("webrtc.") and "aec_args" in parsed:
            parsed["aec_args"] += f" {token}"
            continue
        if key in parsed:
            raise LauncherError(
                f"duplicate module-echo-cancel argument {key!r}"
            )
        parsed[key] = value
    return parsed


def _parse_echo_modules(text: str) -> list[_EchoModule]:
    modules: list[_EchoModule] = []
    seen_ids: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        fields = raw_line.split("\t", 3)
        if len(fields) < 2:
            fields = raw_line.split(maxsplit=2)
        if len(fields) < 2 or fields[1].strip() != "module-echo-cancel":
            continue
        module_id = fields[0].strip()
        if not module_id.isdigit():
            raise LauncherError(
                f"malformed module-echo-cancel id {module_id!r}"
            )
        if module_id in seen_ids:
            raise LauncherError(f"duplicate PipeWire module id {module_id}")
        seen_ids.add(module_id)
        arguments = fields[2].strip() if len(fields) >= 3 else ""
        modules.append(_EchoModule(module_id, _parse_module_arguments(arguments)))
    return modules


def _all_module_ids(text: str) -> set[str]:
    found: set[str] = set()
    for raw_line in text.splitlines():
        fields = raw_line.split("\t", 1)
        if len(fields) < 2:
            fields = raw_line.split(maxsplit=1)
        if fields and fields[0].strip().isdigit():
            found.add(fields[0].strip())
    return found


def _parse_aec_args(raw: str) -> dict[str, str]:
    try:
        tokens = shlex.split(raw)
    except ValueError as exc:
        raise LauncherError(f"malformed WebRTC aec_args: {exc}") from exc
    parsed: dict[str, str] = {}
    for token in tokens:
        key, separator, value = token.partition("=")
        if not separator or not key.startswith("webrtc."):
            raise LauncherError(f"malformed WebRTC aec_args token {token!r}")
        if key in parsed:
            raise LauncherError(f"duplicate WebRTC aec_args key {key!r}")
        parsed[key] = value.lower()
    return parsed


def _matches_owned_module_contract(
    module: _EchoModule, *, source_master: str, sink_master: str
) -> bool:
    expected = {
        "aec_method": "webrtc",
        "source_name": _EC_SOURCE,
        "sink_name": _EC_SINK,
        "source_master": source_master,
        "sink_master": sink_master,
        "aec_args": module.arguments.get("aec_args", ""),
    }
    if dict(module.arguments) != expected:
        return False
    return _parse_aec_args(module.arguments["aec_args"]) == {
        "webrtc.noise_suppression": "false",
        "webrtc.gain_control": "false",
    }


def _canonical_module(
    modules: Sequence[_EchoModule],
    *,
    original_source: str,
    original_sink: str,
    source_names: set[str],
    sink_names: set[str],
) -> _EchoModule:
    claimants = [
        module
        for module in modules
        if module.arguments.get("source_name") == _EC_SOURCE
        or module.arguments.get("sink_name") == _EC_SINK
    ]
    if len(claimants) != 1:
        raise LauncherError(
            "the canonical echo-cancel nodes must belong to exactly one "
            f"module-echo-cancel (found {len(claimants)} claimants)"
        )
    module = claimants[0]
    if (
        module.arguments.get("source_name") != _EC_SOURCE
        or module.arguments.get("sink_name") != _EC_SINK
    ):
        raise LauncherError(
            "one module-echo-cancel claims only part of the canonical route"
        )
    expected_keys = {
        "aec_method",
        "source_name",
        "sink_name",
        "source_master",
        "sink_master",
        "aec_args",
    }
    if set(module.arguments) != expected_keys:
        raise LauncherError(
            "canonical module-echo-cancel arguments differ from the exact live contract"
        )
    if module.arguments["aec_method"].lower() != "webrtc":
        raise LauncherError("canonical echo cancellation is not using WebRTC")
    expected_aec = {
        "webrtc.noise_suppression": "false",
        "webrtc.gain_control": "false",
    }
    if _parse_aec_args(module.arguments["aec_args"]) != expected_aec:
        raise LauncherError(
            "canonical echo cancellation must keep noise suppression and gain control off"
        )

    source_master = module.arguments["source_master"]
    sink_master = module.arguments["sink_master"]
    if (
        not source_master
        or not sink_master
        or _looks_like_echo_cancel(source_master)
        or _looks_like_echo_cancel(sink_master)
    ):
        raise LauncherError("canonical echo cancellation has an unsafe echo-cancel master")
    if source_master not in source_names or sink_master not in sink_names:
        raise LauncherError("canonical echo cancellation refers to a missing master node")

    defaults_are_canonical = (
        original_source == _EC_SOURCE and original_sink == _EC_SINK
    )
    if not defaults_are_canonical and (
        source_master != original_source or sink_master != original_sink
    ):
        raise LauncherError(
            "canonical echo cancellation targets different masters than the current "
            "raw audio defaults"
        )
    return module


def _is_local_pulse_server(value: str) -> bool:
    server = value.strip()
    if (
        not server
        or server.startswith("{")
        or "," in server
        or any(character.isspace() for character in server)
    ):
        return False
    if server.startswith("unix:"):
        server = server[len("unix:"):]
        if server.startswith("path="):
            server = server[len("path="):]
    return Path(server).is_absolute()


def _require_local_pulse_server(ops: Operations) -> None:
    ambient = os.environ.get("PULSE_SERVER", "").strip()
    if ambient and not _is_local_pulse_server(ambient):
        raise LauncherError(
            "./live.sh refuses a non-local PULSE_SERVER; raw audio must stay "
            "on this machine"
        )
    info = _checked_text(ops, ["pactl", "info"], label="Pulse server identity")
    server = ""
    for line in info.splitlines():
        label, separator, value = line.partition(":")
        if separator and label.strip().lower() == "server string":
            server = value.strip()
            break
    if not _is_local_pulse_server(server):
        raise LauncherError(
            "pactl is not connected through a proven local Unix audio server; "
            "refusing host-route mutation"
        )


@dataclass
class EchoRouteLease:
    ops: Operations
    original_source: str
    original_sink: str
    module_id: str | None = None
    unknown_module_loaded: bool = False
    active: bool = True

    @classmethod
    def prepare(cls, ops: Operations) -> "EchoRouteLease":
        if not ops.platform.startswith("linux"):
            raise LauncherError(
                "automatic live setup currently supports Linux/PipeWire only; "
                "use python -m core on this platform"
            )
        if ops.which("pactl") is None:
            raise LauncherError("pactl is required for the Linux echo-cancel route")
        _require_local_pulse_server(ops)

        original_source = _checked_text(
            ops, ["pactl", "get-default-source"], label="default audio source"
        )
        original_sink = _checked_text(
            ops, ["pactl", "get-default-sink"], label="default audio sink"
        )
        lease = cls(ops, original_source, original_sink)

        try:
            modules_before = _checked_text(
                ops,
                ["pactl", "list", "short", "modules"],
                label="PipeWire modules",
                allow_empty=True,
            )
            source_inventory = _checked_text(
                ops, ["pactl", "list", "short", "sources"], label="audio sources"
            )
            sink_inventory = _checked_text(
                ops, ["pactl", "list", "short", "sinks"], label="audio sinks"
            )
            sources = _short_names(source_inventory)
            sinks = _short_names(sink_inventory)
            modules = _parse_echo_modules(modules_before)
            echo_source_names = {
                module.arguments.get("source_name", "") for module in modules
            }
            echo_sink_names = {
                module.arguments.get("sink_name", "") for module in modules
            }

            originals_are_canonical = (
                original_source == _EC_SOURCE and original_sink == _EC_SINK
            )
            if (original_source == _EC_SOURCE) != (original_sink == _EC_SINK):
                raise LauncherError(
                    "only one canonical echo-cancel node is the current default; "
                    "refusing the partial route"
                )
            if not originals_are_canonical and (
                original_source in echo_source_names
                or original_sink in echo_sink_names
                or _looks_like_echo_cancel(original_source)
                or _looks_like_echo_cancel(original_sink)
            ):
                raise LauncherError(
                    "the current audio default is already an echo-cancel node; "
                    "refusing to stack echo cancellation"
                )
            if (
                _short_name_count(source_inventory, original_source) != 1
                or _short_name_count(sink_inventory, original_sink) != 1
            ):
                raise LauncherError(
                    "the current audio defaults must each identify one active node"
                )

            source_exists = _EC_SOURCE in sources
            sink_exists = _EC_SINK in sinks
            if source_exists != sink_exists:
                raise LauncherError(
                    "partial echo-cancel route found; both canonical nodes must exist "
                    "before they can be reused"
                )

            if source_exists:
                if (
                    _short_name_count(source_inventory, _EC_SOURCE) != 1
                    or _short_name_count(sink_inventory, _EC_SINK) != 1
                ):
                    raise LauncherError(
                        "canonical echo-cancel source/sink names must each be unique"
                    )
                _canonical_module(
                    modules,
                    original_source=original_source,
                    original_sink=original_sink,
                    source_names=sources,
                    sink_names=sinks,
                )
                print("[live-setup] reusing existing PipeWire echo-cancel nodes")
            else:
                canonical_before = [
                    module
                    for module in modules
                    if module.arguments.get("source_name") == _EC_SOURCE
                    or module.arguments.get("sink_name") == _EC_SINK
                ]
                if canonical_before:
                    raise LauncherError(
                        "a canonical echo-cancel module exists without both nodes; "
                        "refusing to stack another module on the partial route"
                    )
                before_ids = {module.module_id for module in modules}
                old_mask = _block_lifecycle_signals()
                try:
                    lease.unknown_module_loaded = True
                    load_error: BaseException | None = None
                    result: subprocess.CompletedProcess[str] | None = None
                    try:
                        result = _run_checked(
                            ops,
                            [
                                "pactl", "load-module", "module-echo-cancel",
                                "aec_method=webrtc",
                                f"source_name={_EC_SOURCE}",
                                f"sink_name={_EC_SINK}",
                                f"source_master={original_source}",
                                f"sink_master={original_sink}",
                                "aec_args=webrtc.noise_suppression=false "
                                "webrtc.gain_control=false",
                            ],
                            label="loading PipeWire echo cancellation",
                        )
                    except BaseException as exc:
                        load_error = exc

                    modules_after = _checked_text(
                        ops,
                        ["pactl", "list", "short", "modules"],
                        label="new PipeWire module",
                        allow_empty=True,
                    )
                    parsed_after = _parse_echo_modules(modules_after)
                    new_modules = [
                        module
                        for module in parsed_after
                        if module.module_id not in before_ids
                    ]
                    owned_candidates = [
                        module
                        for module in new_modules
                        if _matches_owned_module_contract(
                            module,
                            source_master=original_source,
                            sink_master=original_sink,
                        )
                    ]

                    raw_id = (result.stdout or "").strip() if result else ""
                    if result is not None and raw_id.isdigit():
                        if (
                            len(owned_candidates) != 1
                            or owned_candidates[0].module_id != raw_id
                        ):
                            raise LauncherError(
                                "PipeWire returned a module id that does not match "
                                "the newly verified echo-cancel contract"
                            )
                        lease.module_id = owned_candidates[0].module_id
                        lease.unknown_module_loaded = False
                    elif len(owned_candidates) == 1:
                        lease.module_id = owned_candidates[0].module_id
                        lease.unknown_module_loaded = False
                    elif result is not None:
                        raise LauncherError(
                            "PipeWire loaded an echo-cancel module but returned "
                            "no unique safe module id for cleanup"
                        )
                    elif not new_modules:
                        # A post-failure inventory proves this attempt published
                        # no echo module, so there is no ambiguous owned resource.
                        lease.unknown_module_loaded = False

                    if load_error is not None:
                        if lease.unknown_module_loaded:
                            raise LauncherError(
                                f"{load_error}; module load outcome is ambiguous"
                            ) from load_error
                        raise load_error
                finally:
                    _restore_signal_mask(old_mask)
                print(
                    "[live-setup] created temporary PipeWire echo-cancel nodes "
                    f"(module {lease.module_id})"
                )

                waited = 0.0
                while waited < _ROUTE_START_TIMEOUT_SEC:
                    source_inventory = _checked_text(
                        ops,
                        ["pactl", "list", "short", "sources"],
                        label="echo-cancel source",
                    )
                    sink_inventory = _checked_text(
                        ops,
                        ["pactl", "list", "short", "sinks"],
                        label="echo-cancel sink",
                    )
                    sources = _short_names(source_inventory)
                    sinks = _short_names(sink_inventory)
                    if _EC_SOURCE in sources and _EC_SINK in sinks:
                        break
                    ops.sleep(0.1)
                    waited += 0.1
                else:
                    raise LauncherError(
                        "PipeWire did not publish both echo-cancel nodes in time"
                    )
                if (
                    _short_name_count(source_inventory, _EC_SOURCE) != 1
                    or _short_name_count(sink_inventory, _EC_SINK) != 1
                ):
                    raise LauncherError(
                        "created echo-cancel source/sink names are not unique"
                    )

                verified_modules = _parse_echo_modules(_checked_text(
                    ops,
                    ["pactl", "list", "short", "modules"],
                    label="created PipeWire module",
                ))
                verified = _canonical_module(
                    verified_modules,
                    original_source=original_source,
                    original_sink=original_sink,
                    source_names=sources,
                    sink_names=sinks,
                )
                if verified.module_id != lease.module_id:
                    raise LauncherError(
                        "the created canonical route does not match the owned module id"
                    )

            _run_checked(
                ops,
                ["pactl", "set-default-source", _EC_SOURCE],
                label="selecting echo-cancel source",
            )
            _run_checked(
                ops,
                ["pactl", "set-default-sink", _EC_SINK],
                label="selecting echo-cancel sink",
            )
            print(
                f"[live-setup] routed capture={_EC_SOURCE}; playback={_EC_SINK}"
            )
            return lease
        except BaseException as exc:
            cleanup_errors = lease.close()
            if cleanup_errors:
                if isinstance(exc, LauncherInterrupted):
                    for error in cleanup_errors:
                        print(f"[live] cleanup warning: {error}", file=sys.stderr)
                    raise
                detail = "; ".join(cleanup_errors)
                if isinstance(exc, LauncherError):
                    raise LauncherError(f"{exc}; cleanup incomplete: {detail}") from exc
                raise LauncherError(
                    f"audio-route setup failed: {exc}; cleanup incomplete: {detail}"
                ) from exc
            raise

    def close(self) -> list[str]:
        if not self.active:
            return []
        self.active = False
        errors: list[str] = []

        def restore_if_owned(kind: str, selected: str, original: str) -> None:
            get_command = ["pactl", f"get-default-{kind}"]
            try:
                current_result = self.ops.run(
                    get_command, capture=True, timeout=5.0
                )
                if current_result.returncode != 0:
                    errors.append(
                        f"{kind} restore: could not verify current default"
                    )
                    return
                current = (current_result.stdout or "").strip()
                if not current:
                    errors.append(
                        f"{kind} restore: current default probe returned empty"
                    )
                    return
                if current != selected:
                    # Another operator/process changed the route during this run.
                    # Do not overwrite that newer choice.
                    return
                result = self.ops.run(
                    ["pactl", f"set-default-{kind}", original],
                    capture=True,
                    timeout=5.0,
                )
                if result.returncode != 0:
                    errors.append(
                        f"{kind} restore: {(result.stderr or '').strip()}"
                    )
            except Exception as exc:  # noqa: BLE001 - cleanup must continue
                errors.append(f"{kind} restore: {exc}")

        restore_if_owned("source", _EC_SOURCE, self.original_source)
        restore_if_owned("sink", _EC_SINK, self.original_sink)

        if self.unknown_module_loaded:
            errors.append(
                "unidentified owned module retained: PipeWire returned no unique "
                "module id safe to unload"
            )
        if self.module_id is not None:
            still_selected = False
            for kind, selected in (("source", _EC_SOURCE), ("sink", _EC_SINK)):
                try:
                    result = self.ops.run(
                        ["pactl", f"get-default-{kind}"],
                        capture=True,
                        timeout=5.0,
                    )
                    if result.returncode != 0:
                        errors.append(
                            f"module {self.module_id} retained: cannot verify "
                            f"the default {kind}"
                        )
                        still_selected = True
                    else:
                        current = (result.stdout or "").strip()
                        if not current:
                            errors.append(
                                f"module {self.module_id} retained: empty default "
                                f"{kind} probe"
                            )
                            still_selected = True
                        elif current == selected:
                            errors.append(
                                f"module {self.module_id} retained: {selected} is "
                                "still the active default"
                            )
                            still_selected = True
                except Exception as exc:  # noqa: BLE001
                    errors.append(
                        f"module {self.module_id} retained: {kind} probe: {exc}"
                    )
                    still_selected = True
            if still_selected:
                return errors
            try:
                inventory = self.ops.run(
                    ["pactl", "list", "short", "modules"],
                    capture=True,
                    timeout=5.0,
                )
                if inventory.returncode != 0:
                    errors.append(
                        f"module {self.module_id} retained: cannot verify module ownership"
                    )
                    return errors
                inventory_text = inventory.stdout or ""
                all_ids = _all_module_ids(inventory_text)
                echo_modules = _parse_echo_modules(inventory_text)
                identified = [
                    module
                    for module in echo_modules
                    if module.module_id == self.module_id
                ]
                if not identified:
                    if self.module_id in all_ids:
                        errors.append(
                            f"module {self.module_id} retained: numeric id now belongs "
                            "to a different module"
                        )
                    # Otherwise an external actor already removed our module.
                    return errors
                if len(identified) != 1 or not _matches_owned_module_contract(
                    identified[0],
                    source_master=self.original_source,
                    sink_master=self.original_sink,
                ):
                    errors.append(
                        f"module {self.module_id} retained: exact ownership contract "
                        "no longer matches"
                    )
                    return errors
                result = self.ops.run(
                    ["pactl", "unload-module", self.module_id],
                    capture=True,
                    timeout=5.0,
                )
                if result.returncode != 0:
                    errors.append(
                        f"module unload {self.module_id}: "
                        f"{(result.stderr or '').strip()}"
                    )
            except Exception as exc:  # noqa: BLE001 - report exact retained state
                errors.append(f"module unload {self.module_id}: {exc}")
        if not errors:
            print("[live-setup] restored the original audio defaults")
        return errors


def _ollama_healthy(ops: Operations, env: Mapping[str, str]) -> bool:
    try:
        result = ops.run(
            ["ollama", "list"], capture=True, env=env, timeout=2.0
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


@dataclass
class OllamaLease:
    ops: Operations
    process: ChildProcess | None = None
    log_handle: IO[bytes] | None = None
    active: bool = True

    @classmethod
    def prepare(
        cls,
        ops: Operations,
        run_dir: Path,
        env: Mapping[str, str],
        *,
        child_signal_mask: set[int] | None = None,
    ) -> "OllamaLease":
        lease = cls(ops)
        if _ollama_healthy(ops, env):
            print("[live-setup] reusing the running Ollama server")
            return lease
        if ops.which("ollama") is None:
            raise LauncherError("Ollama is not running and the ollama command is missing")

        log_path = run_dir / "ollama-serve.log"
        lease.log_handle = log_path.open("ab")
        try:
            old_mask = _block_lifecycle_signals()
            try:
                try:
                    lease.process = ops.popen(
                        ["ollama", "serve"],
                        env=env,
                        stdout=lease.log_handle,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        child_signal_mask=(
                            child_signal_mask
                            if child_signal_mask is not None
                            else old_mask
                        ),
                    )
                except OSError as exc:
                    raise LauncherError(
                        f"could not start temporary Ollama: {exc}"
                    ) from exc
            finally:
                _restore_signal_mask(old_mask)
            print(f"[live-setup] starting temporary Ollama; log={log_path}")
            waited = 0.0
            while waited < _OLLAMA_START_TIMEOUT_SEC:
                if _ollama_healthy(ops, env):
                    return lease
                if lease.process.poll() is not None:
                    break
                ops.sleep(0.25)
                waited += 0.25
            raise LauncherError(
                f"temporary Ollama did not become ready; inspect {log_path}"
            )
        except BaseException as exc:
            cleanup_errors = lease.close()
            if cleanup_errors:
                if isinstance(exc, LauncherInterrupted):
                    for error in cleanup_errors:
                        print(f"[live] cleanup warning: {error}", file=sys.stderr)
                    raise
                detail = "; ".join(cleanup_errors)
                if isinstance(exc, LauncherError):
                    raise LauncherError(f"{exc}; cleanup incomplete: {detail}") from exc
                raise LauncherError(
                    f"Ollama setup failed: {exc}; cleanup incomplete: {detail}"
                ) from exc
            raise

    def close(self) -> list[str]:
        if not self.active:
            return []
        self.active = False
        errors: list[str] = []
        if self.process is not None and self.process.poll() is None:
            try:
                self.ops.signal_group(self.process, signal.SIGTERM)
            except Exception as exc:  # noqa: BLE001 - escalate below
                errors.append(f"temporary Ollama terminate: {exc}")
            try:
                self.process.wait(timeout=_OLLAMA_STOP_TIMEOUT_SEC)
            except subprocess.TimeoutExpired:
                try:
                    self.ops.signal_group(self.process, signal.SIGKILL)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"temporary Ollama kill: {exc}")
                try:
                    self.process.wait(timeout=2.0)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"temporary Ollama stop: {exc}")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"temporary Ollama stop: {exc}")
        if self.log_handle is not None:
            try:
                self.log_handle.close()
            except OSError as exc:
                errors.append(f"temporary Ollama log close: {exc}")
        if self.process is not None and not errors:
            print("[live-setup] stopped the temporary Ollama server")
        return errors


@dataclass(frozen=True)
class _SelectedLiveConfig:
    llm_backend: str
    ollama_host: str | None


def _selected_live_config(
    root: Path, requested_device: str | None
) -> _SelectedLiveConfig:
    """Resolve and validate the profile core will use without constructing it."""
    config_path = root / "config.json"
    if not config_path.exists():
        return _SelectedLiveConfig("ollama", None)
    try:
        from core.config import apply_device_profile, load_config, resolve_device

        config = load_config(
            str(config_path), local=str(root / "config.local.json")
        )
        requested = requested_device or config.get("device", "auto")
        device, _rationale = resolve_device(config, requested)
        config = apply_device_profile(config, device, strict=True)
    except (OSError, ValueError) as exc:
        raise LauncherError(f"could not resolve the selected device profile: {exc}") from exc
    llm = config.get("llm", {}) or {}
    backend = str(llm.get("backend", "ollama") or "ollama").lower()
    if backend not in {"ollama", "llamacpp"}:
        raise LauncherError(f"unsupported local LLM backend {backend!r}")
    configured_host = str(llm.get("host", "") or "").strip() or None

    sherpa = config.get("sherpa", {}) or {}
    if bool(sherpa.get("aec_enabled", False)):
        raise LauncherError(
            "the selected config enables in-app AEC; ./live.sh already uses OS "
            "echo cancellation, so select a profile with sherpa.aec_enabled=false"
        )
    selectors = (
        ("input_device", _EC_SOURCE),
        ("output_device", _EC_SINK),
    )
    defaultish = {"", "default", "pipewire", "pulse"}
    for key, canonical in selectors:
        value = str(sherpa.get(key, "") or "").strip()
        if value.lower() not in defaultish and value != canonical:
            raise LauncherError(
                f"the selected sherpa.{key} bypasses the launcher-owned OS-EC "
                f"route; use the system default or {canonical!r}"
            )
    return _SelectedLiveConfig(backend, configured_host)


def _loopback_ollama_host(configured: str | None) -> str:
    raw = (configured or "http://127.0.0.1:11434").strip()
    try:
        parsed = urlsplit(raw if "://" in raw else f"//{raw}")
    except ValueError as exc:
        raise LauncherError("invalid Ollama host endpoint") from exc
    if parsed.username or parsed.password:
        raise LauncherError("the live launcher refuses credentials in llm.host")
    try:
        host = parsed.hostname or ""
        # Accessing .port performs its validation.
        _port = parsed.port
    except ValueError as exc:
        raise LauncherError("invalid Ollama host endpoint") from exc
    if parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
        raise LauncherError("the live launcher requires a bare loopback Ollama endpoint")
    if parsed.scheme and parsed.scheme.lower() not in {"http", "https"}:
        raise LauncherError("the live launcher requires an HTTP(S) Ollama endpoint")
    local = host.lower() == "localhost"
    if not local:
        try:
            local = ipaddress.ip_address(host).is_loopback
        except ValueError:
            local = False
    if not local:
        raise LauncherError(
            f"the live launcher refuses non-loopback Ollama host {raw!r}; "
            "private vault prompts must stay on this machine"
        )
    return raw


def _bypass_proxies_for_ollama(env: dict[str, str], endpoint: str) -> None:
    parsed = urlsplit(endpoint if "://" in endpoint else f"//{endpoint}")
    host = parsed.hostname
    if not host:  # The endpoint was already validated; keep this fail closed.
        raise LauncherError("could not derive the loopback Ollama hostname")
    additions = [host]
    if ":" in host:
        additions.append(f"[{host}]")
    for key in ("NO_PROXY", "no_proxy"):
        values = [value.strip() for value in env.get(key, "").split(",")]
        values = [value for value in values if value]
        existing = {value.lower() for value in values}
        for value in additions:
            if value.lower() not in existing:
                values.append(value)
                existing.add(value.lower())
        env[key] = ",".join(values)


def _safe_label(value: str) -> str:
    label = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-_")
    if not label:
        raise LauncherError("--run-label must contain a letter or number")
    return label[:48]


def create_run_dir(
    root: Path, label: str, *, now: datetime | None = None
) -> Path:
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    base = f"{_safe_label(label)}-{stamp}"
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    opened: list[int] = []
    try:
        current_fd = os.open(root, directory_flags)
        opened.append(current_fd)
        for name in ("logs", "live"):
            try:
                os.mkdir(name, mode=0o700, dir_fd=current_fd)
            except FileExistsError:
                pass
            child_fd = os.open(name, directory_flags, dir_fd=current_fd)
            if not stat.S_ISDIR(os.fstat(child_fd).st_mode):
                os.close(child_fd)
                raise LauncherError(f"private run parent {name!r} is not a directory")
            opened.append(child_fd)
            current_fd = child_fd

        for suffix in range(100):
            name = base if suffix == 0 else f"{base}-{suffix:02d}"
            try:
                os.mkdir(name, mode=0o700, dir_fd=current_fd)
            except FileExistsError:
                continue
            leaf_fd = os.open(name, directory_flags, dir_fd=current_fd)
            try:
                os.fchmod(leaf_fd, 0o700)
            finally:
                os.close(leaf_fd)
            return root / "logs" / "live" / name
    finally:
        for descriptor in reversed(opened):
            os.close(descriptor)
    raise LauncherError("could not allocate a unique private live-run directory")


def _run_voice_child(
    ops: Operations, command: Sequence[str], env: Mapping[str, str]
) -> int:
    process: ChildProcess | None = None
    old_mask = None
    try:
        # On the advertised POSIX path, block lifecycle signals across Popen so
        # a pending signal cannot land after setsid() but before we own the PID.
        old_mask = _block_lifecycle_signals()
        try:
            process = ops.popen(
                command,
                env=env,
                start_new_session=True,
                child_signal_mask=old_mask,
            )
        except OSError as exc:
            raise LauncherError(f"could not start the voice runtime: {exc}") from exc
        finally:
            if old_mask is not None:
                _restore_signal_mask(old_mask)
                old_mask = None

        while True:
            if process.poll() is not None:
                return int(process.poll() or 0)
            try:
                return process.wait(timeout=0.25)
            except subprocess.TimeoutExpired:
                continue
    except (LauncherInterrupted, KeyboardInterrupt) as exc:
        signum = exc.signum if isinstance(exc, LauncherInterrupted) else signal.SIGINT
        if process is not None:
            child_signal = (
                signal.SIGTERM
                if hasattr(signal, "SIGHUP") and signum == signal.SIGHUP
                else signum
            )
            _stop_voice_child(ops, process, child_signal)
        if isinstance(exc, LauncherInterrupted):
            raise
        raise LauncherInterrupted(signal.SIGINT) from exc
    except BaseException:
        if process is not None:
            _stop_voice_child(ops, process, signal.SIGTERM)
        raise
    finally:
        if old_mask is not None:
            _restore_signal_mask(old_mask)


def _stop_voice_child(
    ops: Operations, process: ChildProcess, first_signal: int
) -> None:
    stages = (
        (first_signal, _VOICE_INTERRUPT_GRACE_SEC),
        (signal.SIGTERM, _VOICE_TERM_GRACE_SEC),
        (signal.SIGKILL, _VOICE_KILL_GRACE_SEC),
    )
    for signum, timeout in stages:
        if process.poll() is not None:
            return
        if signum == signal.SIGKILL:
            print(
                "[live] voice runtime did not finalize after grace periods; "
                "forcing stop (the recording tail may be truncated)",
                file=sys.stderr,
            )
        try:
            ops.signal_group(process, signum)
        except Exception as exc:  # noqa: BLE001 - still attempt bounded wait/escalation
            print(f"[live] voice process-group signal failed: {exc}", file=sys.stderr)
        try:
            process.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            continue
        except Exception as exc:  # noqa: BLE001 - advance to stronger bounded stop
            print(f"[live] voice shutdown wait failed: {exc}", file=sys.stderr)
    if process.poll() is None:
        print(
            "[live] cleanup warning: voice process group could not be verified stopped",
            file=sys.stderr,
        )


def _print_artifacts(run_dir: Path) -> None:
    files = sorted(path for path in run_dir.iterdir() if path.is_file())
    print(f"[live] private local bundle: {run_dir}")
    for path in files:
        print(f"[live]   {path.name} ({path.stat().st_size} bytes)")
    print("[live] keep this directory local; do not commit, push, or upload it")


_VALUE_OPTIONS = (
    ("--llm", "llm"),
    ("--model", "model"),
    ("--fast-model", "fast_model"),
    ("--asr-final", "asr_final"),
    ("--device", "device"),
    ("--mode", "mode"),
    ("--input-gain", "input_gain"),
)
_FLAG_OPTIONS = (
    ("--agent", "agent"),
    ("--gui-actions", "gui_actions"),
    ("--planner", "planner"),
    ("--stream-tts", "stream_tts"),
)


def _live_parser() -> argparse.ArgumentParser:
    from always_on_agent.events import Mode

    parser = argparse.ArgumentParser(
        description=(
            "Prepare a reversible Linux open-speaker route and capture one "
            "private replayable voice session"
        ),
        epilog=(
            "Supported runtime options are passed safely to `python -m core`; "
            "for example: ./live.sh --llm echo"
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--run-label", default="manual", help="safe label for logs/live/<label>-<time>"
    )
    parser.add_argument("--llm", choices=["echo", "ollama"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--fast-model", dest="fast_model", default=None)
    parser.add_argument(
        "--asr-final",
        dest="asr_final",
        choices=["streaming", "sense_voice", "whisper"],
        default=None,
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--mode", choices=[mode.value for mode in Mode], default=None)
    parser.add_argument("--input-gain", dest="input_gain", type=float, default=None)
    parser.add_argument("--agent", action="store_true")
    parser.add_argument("--gui-actions", dest="gui_actions", action="store_true")
    parser.add_argument("--planner", action="store_true")
    parser.add_argument("--stream-tts", dest="stream_tts", action="store_true")
    return parser


def _parse_live_arguments(
    argv: Sequence[str],
) -> tuple[argparse.ArgumentParser, argparse.Namespace, list[str]]:
    parser = _live_parser()
    supported = {
        "--run-label",
        *(option for option, _destination in _VALUE_OPTIONS),
        *(option for option, _destination in _FLAG_OPTIONS),
    }
    counts = {option: 0 for option in supported}
    for token in argv:
        name = token.split("=", 1)[0]
        if name in counts:
            counts[name] += 1
    duplicate = next(
        (name for name in sorted(counts) if counts[name] > 1), None
    )
    if duplicate is not None:
        parser.error(f"duplicate option {duplicate} is not allowed")

    args = parser.parse_args(list(argv))
    if str(args.device or "").lower() == "open_speaker":
        parser.error(
            "./live.sh uses the host PipeWire echo-cancel path; omit "
            "--device open_speaker (the in-app APM fallback)"
        )

    core_args: list[str] = []
    for option, destination in _VALUE_OPTIONS:
        value = getattr(args, destination)
        if value is not None:
            core_args.extend([option, str(value)])
    for option, destination in _FLAG_OPTIONS:
        if getattr(args, destination):
            core_args.append(option)
    return parser, args, core_args


def run_live_session(
    argv: Sequence[str] | None = None,
    *,
    ops: Operations | None = None,
    root: Path | None = None,
    lock_path: Path | None = None,
    now: datetime | None = None,
) -> int:
    _parser, args, core_args = _parse_live_arguments(list(argv or ()))

    ops = ops or RealOperations()
    root = (root or Path(__file__).resolve().parents[1]).resolve()
    prior_umask = os.umask(0o077)
    live_lock: LiveSessionLock | None = None
    run_dir: Path | None = None
    route: EchoRouteLease | None = None
    ollama: OllamaLease | None = None
    result_code = 2
    cleanup_errors: list[str] = []
    signal_state = {"signum": 0, "cleaning": False}
    saved_signals: dict[int, object] = {}

    def request_shutdown(signum, _frame):
        if signal_state["signum"]:
            return
        signal_state["signum"] = int(signum)
        if not signal_state["cleaning"]:
            raise LauncherInterrupted(int(signum))

    try:
        try:
            for signum in sorted(_lifecycle_signals()):
                previous = signal.getsignal(signum)
                signal.signal(signum, request_shutdown)
                saved_signals[signum] = previous
        except (ValueError, OSError):
            for signum, handler in saved_signals.items():
                try:
                    signal.signal(signum, handler)
                except (ValueError, OSError):
                    pass
            saved_signals.clear()
        old_mask = _block_lifecycle_signals()
        try:
            live_lock = LiveSessionLock.acquire(lock_path or _default_lock_path())
        finally:
            _restore_signal_mask(old_mask)
        try:
            run_dir = create_run_dir(root, args.run_label, now=now)
        except OSError as exc:
            raise LauncherError(f"could not create the private live-run bundle: {exc}") from exc
        env = dict(os.environ)
        env["SPEAKER_RUN_LOG_DIR"] = str(run_dir)
        env["SPEAKER_KEEP_RUNS"] = "0"
        # Per-client audio selectors/configs override the server defaults that
        # this launcher proved/owns, so ambient raw/virtual pins cannot bypass EC.
        for unsafe_audio_env in (
            "PULSE_SOURCE",
            "PULSE_SINK",
            "ALSA_CONFIG_PATH",
        ):
            env.pop(unsafe_audio_env, None)

        llm_mode = str(args.llm or "ollama").lower()
        selected = _selected_live_config(root, args.device)
        if llm_mode != "echo" and selected.llm_backend == "ollama":
            ollama_host = _loopback_ollama_host(selected.ollama_host)
            env["OLLAMA_HOST"] = ollama_host
            _bypass_proxies_for_ollama(env, ollama_host)
            old_mask = _block_lifecycle_signals()
            try:
                ollama = OllamaLease.prepare(
                    ops,
                    run_dir,
                    env,
                    child_signal_mask=old_mask,
                )
            finally:
                _restore_signal_mask(old_mask)
        old_mask = _block_lifecycle_signals()
        try:
            route = EchoRouteLease.prepare(ops)
        finally:
            _restore_signal_mask(old_mask)

        doctor = [sys.executable, "-m", "tools.doctor"]
        if args.device:
            doctor.extend(["--device", args.device])
        if llm_mode == "echo":
            doctor.append(
                "--defer-ollama"
                if selected.llm_backend == "ollama"
                else "--defer-llm"
            )
        print("[live-setup] running the shared readiness doctor")
        try:
            doctor_result = ops.run(
                doctor, env=env, timeout=_DOCTOR_TIMEOUT_SEC
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise LauncherError(f"readiness doctor could not complete: {exc}") from exc
        if doctor_result.returncode != 0:
            raise LauncherError(
                f"doctor reported NOT READY (exit {doctor_result.returncode}); "
                "the microphone was not opened"
            )

        command = [
            sys.executable,
            "-m",
            "core",
            "--engine",
            "sherpa",
            "--debug",
            "--record",
            "--record-playback-reference",
            *core_args,
        ]
        print(f"[live] recording locally in {run_dir}")
        print("[live] stay silent during calibration; use one Ctrl-C to finish")
        result_code = _run_voice_child(ops, command, env)
    except LauncherInterrupted as exc:
        print(f"[live] interrupted by signal {exc.signum}; cleaning up", file=sys.stderr)
        result_code = 128 + exc.signum
    except LauncherError as exc:
        print(f"[live] setup failed: {exc}", file=sys.stderr)
        result_code = 2
    except KeyboardInterrupt:
        signal_state["signum"] = signal.SIGINT
        print("[live] interrupted; cleaning up", file=sys.stderr)
        result_code = 128 + signal.SIGINT
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"[live] setup failed: {exc}", file=sys.stderr)
        result_code = 2
    finally:
        signal_state["cleaning"] = True
        if route is not None:
            cleanup_errors.extend(route.close())
        if ollama is not None:
            cleanup_errors.extend(ollama.close())
        if live_lock is not None:
            try:
                live_lock.close()
            except OSError as exc:
                cleanup_errors.append(f"live-session lock release: {exc}")
        for signum, handler in saved_signals.items():
            try:
                signal.signal(signum, handler)
            except (ValueError, OSError) as exc:
                cleanup_errors.append(f"signal-handler restore {signum}: {exc}")
        os.umask(prior_umask)

    if signal_state["signum"]:
        result_code = 128 + signal_state["signum"]

    if cleanup_errors:
        for error in cleanup_errors:
            print(f"[live] cleanup warning: {error}", file=sys.stderr)
        if result_code == 0:
            result_code = 2
    if run_dir is not None:
        _print_artifacts(run_dir)
    return result_code


def main(argv: Sequence[str] | None = None) -> int:
    return run_live_session(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    raise SystemExit(main())
