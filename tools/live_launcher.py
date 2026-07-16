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
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import IO, Mapping, Protocol, Sequence

try:  # Imported lazily in effect: the module still gives a useful error off Linux.
    import fcntl
except ImportError:  # pragma: no cover - exercised on non-POSIX hosts
    fcntl = None  # type: ignore[assignment]

_EC_SOURCE = "echo-cancel-source"
_EC_SINK = "echo-cancel-sink"
_OLLAMA_START_TIMEOUT_SEC = 15.0
_OLLAMA_STOP_TIMEOUT_SEC = 5.0
_ROUTE_START_TIMEOUT_SEC = 5.0
_VOICE_INTERRUPT_GRACE_SEC = 20.0
_VOICE_TERM_GRACE_SEC = 10.0


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
    ) -> ChildProcess:
        return subprocess.Popen(
            list(command),
            env=dict(env) if env is not None else None,
            stdout=stdout,
            stderr=stderr,
            start_new_session=start_new_session,
        )

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def signal_group(self, process: ChildProcess, signum: int) -> None:
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass


@dataclass
class LiveSessionLock:
    handle: IO[str]

    @classmethod
    def acquire(cls, path: Path) -> "LiveSessionLock":
        if fcntl is None:
            raise LauncherError("the live-session lock requires Linux/POSIX flock")
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        handle = path.open("a+", encoding="utf-8")
        path.chmod(0o600)
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
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR", "").strip()
    parent = Path(runtime_dir) if runtime_dir else Path.home() / ".local/state/speaker"
    return parent / "live-session.lock"


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


def _matching_module_ids(text: str) -> set[str]:
    found: set[str] = set()
    for line in text.splitlines():
        low = line.lower()
        if (
            "module-echo-cancel" in low
            and f"source_name={_EC_SOURCE}" in line
            and f"sink_name={_EC_SINK}" in line
        ):
            fields = line.split()
            if fields and fields[0].isdigit():
                found.add(fields[0])
    return found


@dataclass
class EchoRouteLease:
    ops: Operations
    original_source: str
    original_sink: str
    module_id: str | None = None
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
            sources = _short_names(_checked_text(
                ops, ["pactl", "list", "short", "sources"], label="audio sources"
            ))
            sinks = _short_names(_checked_text(
                ops, ["pactl", "list", "short", "sinks"], label="audio sinks"
            ))
            source_exists = _EC_SOURCE in sources
            sink_exists = _EC_SINK in sinks
            if source_exists != sink_exists:
                raise LauncherError(
                    "partial echo-cancel route found; both canonical nodes must exist "
                    "before they can be reused"
                )

            if source_exists:
                if not _matching_module_ids(modules_before):
                    raise LauncherError(
                        "echo-cancel nodes exist without an owned module-echo-cancel "
                        "record; refusing to alter the ambiguous route"
                    )
                print("[live-setup] reusing existing PipeWire echo-cancel nodes")
            else:
                before_ids = _matching_module_ids(modules_before)
                if before_ids:
                    raise LauncherError(
                        "a canonical echo-cancel module exists without both nodes; "
                        "refusing to stack another module on the partial route"
                    )
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
                raw_id = (result.stdout or "").strip()
                if raw_id.isdigit():
                    lease.module_id = raw_id
                else:
                    modules_after = _checked_text(
                        ops,
                        ["pactl", "list", "short", "modules"],
                        label="new PipeWire module",
                        allow_empty=True,
                    )
                    new_ids = _matching_module_ids(modules_after) - before_ids
                    if len(new_ids) != 1:
                        raise LauncherError(
                            "PipeWire loaded an echo-cancel module but returned no "
                            "unique safe module id for cleanup"
                        )
                    lease.module_id = new_ids.pop()
                print(
                    "[live-setup] created temporary PipeWire echo-cancel nodes "
                    f"(module {lease.module_id})"
                )

                waited = 0.0
                while waited < _ROUTE_START_TIMEOUT_SEC:
                    sources = _short_names(_checked_text(
                        ops,
                        ["pactl", "list", "short", "sources"],
                        label="echo-cancel source",
                    ))
                    sinks = _short_names(_checked_text(
                        ops,
                        ["pactl", "list", "short", "sinks"],
                        label="echo-cancel sink",
                    ))
                    if _EC_SOURCE in sources and _EC_SINK in sinks:
                        break
                    ops.sleep(0.1)
                    waited += 0.1
                else:
                    raise LauncherError(
                        "PipeWire did not publish both echo-cancel nodes in time"
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
        except BaseException:
            lease.close()
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
                    elif (result.stdout or "").strip() == selected:
                        errors.append(
                            f"module {self.module_id} retained: {selected} is still "
                            "the active default"
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
        cls, ops: Operations, run_dir: Path, env: Mapping[str, str]
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
            lease.process = ops.popen(
                ["ollama", "serve"],
                env=env,
                stdout=lease.log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
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
        except BaseException:
            lease.close()
            raise

    def close(self) -> list[str]:
        if not self.active:
            return []
        self.active = False
        errors: list[str] = []
        if self.process is not None and self.process.poll() is None:
            self.ops.signal_group(self.process, signal.SIGTERM)
            try:
                self.process.wait(timeout=_OLLAMA_STOP_TIMEOUT_SEC)
            except subprocess.TimeoutExpired:
                self.ops.signal_group(self.process, signal.SIGKILL)
                try:
                    self.process.wait(timeout=2.0)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"temporary Ollama stop: {exc}")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"temporary Ollama stop: {exc}")
        if self.log_handle is not None:
            self.log_handle.close()
        if self.process is not None and not errors:
            print("[live-setup] stopped the temporary Ollama server")
        return errors


def _option_value(arguments: Sequence[str], name: str) -> str | None:
    for index, value in enumerate(arguments):
        if value == name and index + 1 < len(arguments):
            return arguments[index + 1]
        prefix = name + "="
        if value.startswith(prefix):
            return value[len(prefix):]
    return None


def _has_option(arguments: Sequence[str], name: str) -> bool:
    return any(value == name or value.startswith(name + "=") for value in arguments)


def _selected_llm_backend(root: Path, arguments: Sequence[str]) -> str:
    """Resolve the same local backend/profile as core without constructing it."""
    config_path = root / "config.json"
    if not config_path.exists():
        return "ollama"
    try:
        from core.config import apply_device_profile, load_config, resolve_device

        config = load_config(
            str(config_path), local=str(root / "config.local.json")
        )
        requested = _option_value(arguments, "--device") or config.get("device", "auto")
        device, _rationale = resolve_device(config, requested)
        config = apply_device_profile(config, device, strict=True)
    except (OSError, ValueError) as exc:
        raise LauncherError(f"could not resolve the selected device profile: {exc}") from exc
    return str((config.get("llm", {}) or {}).get("backend", "ollama")).lower()


def _safe_label(value: str) -> str:
    label = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-_")
    if not label:
        raise LauncherError("--run-label must contain a letter or number")
    return label[:48]


def create_run_dir(
    root: Path, label: str, *, now: datetime | None = None
) -> Path:
    parent = root / "logs" / "live"
    parent.mkdir(parents=True, exist_ok=True)
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    base = f"{_safe_label(label)}-{stamp}"
    for suffix in range(100):
        name = base if suffix == 0 else f"{base}-{suffix:02d}"
        path = parent / name
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            continue
        path.chmod(0o700)
        return path
    raise LauncherError("could not allocate a unique private live-run directory")


def _run_voice_child(
    ops: Operations, command: Sequence[str], env: Mapping[str, str]
) -> int:
    process = ops.popen(command, env=env, start_new_session=True)
    saved: dict[int, object] = {}

    def forward(signum, _frame):
        if process.poll() is None:
            process.send_signal(signum)

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            saved[signum] = signal.getsignal(signum)
            signal.signal(signum, forward)
    except (ValueError, OSError):
        saved.clear()
    try:
        return process.wait()
    except KeyboardInterrupt:
        process.send_signal(signal.SIGINT)
        return process.wait()
    finally:
        for signum, handler in saved.items():
            signal.signal(signum, handler)


def _print_artifacts(run_dir: Path) -> None:
    files = sorted(path for path in run_dir.iterdir() if path.is_file())
    print(f"[live] private local bundle: {run_dir}")
    for path in files:
        print(f"[live]   {path.name} ({path.stat().st_size} bytes)")
    print("[live] keep this directory local; do not commit, push, or upload it")


def run_live_session(
    argv: Sequence[str] | None = None,
    *,
    ops: Operations | None = None,
    root: Path | None = None,
    now: datetime | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a reversible Linux open-speaker route and capture one "
            "private replayable voice session"
        ),
        epilog=(
            "Unknown arguments pass to `python -m core`, for example: "
            "./live.sh --llm echo"
        ),
    )
    parser.add_argument(
        "--run-label", default="manual", help="safe label for logs/live/<label>-<time>"
    )
    args, core_args = parser.parse_known_args(list(argv or ()))
    if core_args[:1] == ["--"]:
        core_args = core_args[1:]
    reserved = {
        "--engine",
        "--replay-dir",
        "--debug",
        "--record",
        "--record-playback-reference",
        "--list-devices",
        "--input-device",
        "--output-device",
        "--enroll",
        "--enroll-seconds",
        "--enroll-passes",
        "--replace-enrollment",
        "--require-prepared-enrollment",
        "--autotest-virtual-delay-contract",
    }
    conflict = next(
        (name for name in sorted(reserved) if _has_option(core_args, name)), None
    )
    if conflict:
        parser.error(f"./live.sh owns {conflict}; omit that option")
    if (_option_value(core_args, "--device") or "").lower() == "open_speaker":
        parser.error(
            "./live.sh uses the host PipeWire echo-cancel path; omit "
            "--device open_speaker (the in-app APM fallback)"
        )

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
        signal_state["signum"] = int(signum)
        if not signal_state["cleaning"]:
            raise LauncherInterrupted(int(signum))

    try:
        try:
            for signum in (signal.SIGINT, signal.SIGTERM):
                saved_signals[signum] = signal.getsignal(signum)
                signal.signal(signum, request_shutdown)
        except (ValueError, OSError):
            saved_signals.clear()
        live_lock = LiveSessionLock.acquire(root)
        run_dir = create_run_dir(root, args.run_label, now=now)
        env = dict(os.environ)
        env["SPEAKER_RUN_LOG_DIR"] = str(run_dir)
        env["SPEAKER_KEEP_RUNS"] = "0"

        llm_mode = (_option_value(core_args, "--llm") or "ollama").lower()
        llm_backend = _selected_llm_backend(root, core_args)
        if llm_mode != "echo" and llm_backend == "ollama":
            ollama = OllamaLease.prepare(ops, run_dir, env)
        route = EchoRouteLease.prepare(ops)

        doctor = [sys.executable, "-m", "tools.doctor"]
        device = _option_value(core_args, "--device")
        if device:
            doctor.extend(["--device", device])
        if llm_mode == "echo":
            doctor.append("--defer-ollama")
        print("[live-setup] running the shared readiness doctor")
        doctor_result = ops.run(doctor, env=env, timeout=None)
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
    finally:
        signal_state["cleaning"] = True
        if route is not None:
            cleanup_errors.extend(route.close())
        if ollama is not None:
            cleanup_errors.extend(ollama.close())
        if live_lock is not None:
            live_lock.close()
        for signum, handler in saved_signals.items():
            signal.signal(signum, handler)
        os.umask(prior_umask)

    if signal_state["signum"] and result_code == 0:
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
