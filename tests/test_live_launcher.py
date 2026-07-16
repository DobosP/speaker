"""Headless contract tests for the one-command live-session launcher."""
from __future__ import annotations

import json
import os
import signal
import stat
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

from tools.live_launcher import (
    LiveSessionLock,
    create_run_dir,
    run_live_session,
)


def _completed(
    command, returncode: int = 0, stdout: str = "", stderr: str = ""
):
    return subprocess.CompletedProcess(command, returncode, stdout, stderr)


class _FakeProcess:
    _next_pid = 2000

    def __init__(self, kind: str, *, returncode: int = 0, on_wait=None):
        self.kind = kind
        self.returncode = returncode
        self.on_wait = on_wait
        self.running = True
        self.terminated = False
        self.killed = False
        self.signals: list[int] = []
        self.pid = _FakeProcess._next_pid
        _FakeProcess._next_pid += 1

    def poll(self):
        return None if self.running else self.returncode

    def wait(self, timeout=None):
        if self.on_wait is not None:
            callback, self.on_wait = self.on_wait, None
            callback()
        self.running = False
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.running = False

    def kill(self):
        self.killed = True
        self.running = False

    def send_signal(self, sig):
        self.signals.append(sig)
        self.running = False


class _FakeOps:
    platform = "linux"

    def __init__(
        self,
        *,
        nodes: bool = False,
        module_loaded: bool | None = None,
        ollama_healthy: bool = True,
        doctor_rc: int = 0,
        voice_rc: int = 0,
    ):
        self.default_source = "raw-source"
        self.default_sink = "raw-sink"
        self.nodes = nodes
        self.module_loaded = nodes if module_loaded is None else module_loaded
        self.ollama_healthy = ollama_healthy
        self.doctor_rc = doctor_rc
        self.voice_rc = voice_rc
        self.calls: list[tuple[str, tuple[str, ...]]] = []
        self.popen_calls: list[tuple[tuple[str, ...], dict]] = []
        self.ollama_process: _FakeProcess | None = None
        self.voice_process: _FakeProcess | None = None
        self.ollama_lists_after_spawn = 0
        self.ollama_ready_after = 1
        self.fail_set_sink = False
        self.fail_restore_source = False
        self.invalid_module_output = False
        self.voice_wait_callback = None
        self.last_voice_env = None

    def which(self, name):
        return f"/usr/bin/{name}" if name in {"pactl", "ollama"} else None

    def run(self, command, *, capture=False, env=None, timeout=None):
        command = tuple(command)
        self.calls.append(("run", command))
        if command == ("ollama", "list"):
            if self.ollama_process is not None:
                self.ollama_lists_after_spawn += 1
                if self.ollama_lists_after_spawn >= self.ollama_ready_after:
                    self.ollama_healthy = True
            return _completed(command, 0 if self.ollama_healthy else 1)
        if command == ("pactl", "get-default-source"):
            return _completed(command, stdout=self.default_source + "\n")
        if command == ("pactl", "get-default-sink"):
            return _completed(command, stdout=self.default_sink + "\n")
        if command == ("pactl", "list", "short", "modules"):
            text = "1\tmodule-always-sink\n"
            if self.module_loaded:
                text += (
                    "77\tmodule-echo-cancel\t"
                    "source_name=echo-cancel-source "
                    "sink_name=echo-cancel-sink\n"
                )
            return _completed(command, stdout=text)
        if command == ("pactl", "list", "short", "sources"):
            text = "10\traw-source\tPipeWire\n"
            if self.nodes:
                text += "11\techo-cancel-source\tPipeWire\n"
            return _completed(command, stdout=text)
        if command == ("pactl", "list", "short", "sinks"):
            text = "20\traw-sink\tPipeWire\n"
            if self.nodes:
                text += "21\techo-cancel-sink\tPipeWire\n"
            return _completed(command, stdout=text)
        if command[:3] == ("pactl", "load-module", "module-echo-cancel"):
            self.module_loaded = True
            self.nodes = True
            return _completed(
                command, stdout="not-an-id\n" if self.invalid_module_output else "77\n"
            )
        if command[:2] == ("pactl", "set-default-source"):
            target = command[2]
            if target == "raw-source" and self.fail_restore_source:
                return _completed(command, 1, stderr="forced source restore failure")
            self.default_source = target
            return _completed(command)
        if command[:2] == ("pactl", "set-default-sink"):
            target = command[2]
            if target == "echo-cancel-sink" and self.fail_set_sink:
                return _completed(command, 1, stderr="forced sink selection failure")
            self.default_sink = target
            return _completed(command)
        if command[:2] == ("pactl", "unload-module"):
            self.module_loaded = False
            self.nodes = False
            return _completed(command)
        if len(command) >= 3 and command[1:3] == ("-m", "tools.doctor"):
            return _completed(command, self.doctor_rc)
        return _completed(command)

    def popen(
        self,
        command,
        *,
        env=None,
        stdout=None,
        stderr=None,
        start_new_session=False,
    ):
        command = tuple(command)
        kwargs = {
            "env": env,
            "stdout": stdout,
            "stderr": stderr,
            "start_new_session": start_new_session,
        }
        self.popen_calls.append((command, kwargs))
        if command == ("ollama", "serve"):
            self.ollama_process = _FakeProcess("ollama")
            return self.ollama_process
        self.last_voice_env = env
        self.voice_process = _FakeProcess(
            "voice", returncode=self.voice_rc, on_wait=self.voice_wait_callback
        )
        return self.voice_process

    def sleep(self, _seconds):
        return None


_NOW = datetime(2026, 7, 16, 12, 34, 56)


def _commands(ops: _FakeOps) -> list[tuple[str, ...]]:
    return [command for _kind, command in ops.calls]


def _voice_command(ops: _FakeOps) -> tuple[str, ...]:
    return next(command for command, _kwargs in ops.popen_calls if "core" in command)


def test_one_command_creates_private_recorded_session_and_cleans_owned_resources(
    tmp_path, capsys
):
    ops = _FakeOps(nodes=False, ollama_healthy=False)

    rc = run_live_session(
        ["--run-label", "vault test", "--mode", "assistant"],
        ops=ops,
        root=tmp_path,
        now=_NOW,
    )

    assert rc == 0
    run_dir = tmp_path / "logs/live/vault-test-20260716-123456"
    assert stat.S_IMODE(run_dir.stat().st_mode) == 0o700
    assert ops.default_source == "raw-source"
    assert ops.default_sink == "raw-sink"
    assert ops.module_loaded is False
    assert ops.ollama_process is not None and ops.ollama_process.terminated
    command = _voice_command(ops)
    assert command[:7] == (
        sys.executable,
        "-m",
        "core",
        "--engine",
        "sherpa",
        "--debug",
        "--record",
    )
    assert "--record-playback-reference" in command
    assert command[-2:] == ("--mode", "assistant")
    assert ops.last_voice_env["SPEAKER_RUN_LOG_DIR"] == str(run_dir)
    assert ops.last_voice_env["SPEAKER_KEEP_RUNS"] == "0"
    assert "do not commit, push, or upload" in capsys.readouterr().out


def test_existing_ollama_and_echo_route_are_reused_not_destroyed(tmp_path):
    ops = _FakeOps(nodes=True, module_loaded=True, ollama_healthy=True)

    assert run_live_session([], ops=ops, root=tmp_path, now=_NOW) == 0

    commands = _commands(ops)
    assert not any(command[:2] == ("pactl", "load-module") for command in commands)
    assert not any(command[:2] == ("pactl", "unload-module") for command in commands)
    assert all(command != ("ollama", "serve") for command, _ in ops.popen_calls)
    assert ops.module_loaded is True
    assert ops.default_source == "raw-source"
    assert ops.default_sink == "raw-sink"


def test_echo_llm_skips_ollama_and_uses_deferred_doctor(tmp_path):
    ops = _FakeOps(nodes=True, ollama_healthy=False)

    assert run_live_session(
        ["--llm", "echo"], ops=ops, root=tmp_path, now=_NOW
    ) == 0

    assert all(command != ("ollama", "serve") for command, _ in ops.popen_calls)
    doctor = next(
        command for command in _commands(ops) if command[1:3] == ("-m", "tools.doctor")
    )
    assert doctor[-1] == "--defer-ollama"


def test_llamacpp_profile_does_not_start_unneeded_ollama(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({
            "device": "local_cpu",
            "device_profiles": {
                "local_cpu": {"llm": {"backend": "llamacpp"}},
            },
            "llm": {"backend": "ollama"},
        }),
        encoding="utf-8",
    )
    ops = _FakeOps(nodes=True, ollama_healthy=False)

    assert run_live_session([], ops=ops, root=tmp_path, now=_NOW) == 0

    assert ("ollama", "list") not in _commands(ops)
    assert all(command != ("ollama", "serve") for command, _ in ops.popen_calls)


def test_doctor_failure_never_opens_microphone_and_cleans_up(tmp_path):
    ops = _FakeOps(nodes=False, ollama_healthy=True, doctor_rc=9)

    assert run_live_session([], ops=ops, root=tmp_path, now=_NOW) == 2

    assert not any("core" in command for command, _ in ops.popen_calls)
    assert ops.default_source == "raw-source"
    assert ops.default_sink == "raw-sink"
    assert ops.module_loaded is False


def test_partial_route_failure_restores_source_and_unloads_owned_module(tmp_path):
    ops = _FakeOps(nodes=False, ollama_healthy=True)
    ops.fail_set_sink = True

    assert run_live_session([], ops=ops, root=tmp_path, now=_NOW) == 2

    assert ops.default_source == "raw-source"
    assert ops.default_sink == "raw-sink"
    assert ops.module_loaded is False
    assert not any("core" in command for command, _ in ops.popen_calls)


def test_cleanup_preserves_newer_external_default_changes(tmp_path):
    ops = _FakeOps(nodes=False, ollama_healthy=True)

    def external_change():
        ops.default_source = "external-source"
        ops.default_sink = "external-sink"

    ops.voice_wait_callback = external_change
    assert run_live_session([], ops=ops, root=tmp_path, now=_NOW) == 0

    assert ops.default_source == "external-source"
    assert ops.default_sink == "external-sink"
    assert ops.module_loaded is False


def test_failed_restore_retains_module_and_makes_result_red(tmp_path, capsys):
    ops = _FakeOps(nodes=False, ollama_healthy=True)
    ops.fail_restore_source = True

    assert run_live_session([], ops=ops, root=tmp_path, now=_NOW) == 2

    assert ops.default_source == "echo-cancel-source"
    assert ops.default_sink == "raw-sink"
    assert ops.module_loaded is True
    assert not any(
        command[:2] == ("pactl", "unload-module") for command in _commands(ops)
    )
    assert "retained" in capsys.readouterr().err


def test_core_failure_propagates_after_cleanup_and_keeps_run_directory(tmp_path):
    ops = _FakeOps(nodes=False, ollama_healthy=True, voice_rc=7)

    assert run_live_session([], ops=ops, root=tmp_path, now=_NOW) == 7

    assert (tmp_path / "logs/live/manual-20260716-123456").is_dir()
    assert ops.module_loaded is False
    assert ops.default_source == "raw-source"
    assert ops.default_sink == "raw-sink"


def test_invalid_module_stdout_is_recovered_from_exact_new_module_identity(tmp_path):
    ops = _FakeOps(nodes=False, ollama_healthy=True)
    ops.invalid_module_output = True

    assert run_live_session([], ops=ops, root=tmp_path, now=_NOW) == 0
    assert ops.module_loaded is False


def test_unique_run_directories_are_never_reused(tmp_path):
    first = create_run_dir(tmp_path, "manual", now=_NOW)
    marker = first / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")
    second = create_run_dir(tmp_path, "manual", now=_NOW)

    assert first.name == "manual-20260716-123456"
    assert second.name == "manual-20260716-123456-01"
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_concurrent_launcher_is_rejected_without_touching_audio(tmp_path):
    lock = LiveSessionLock.acquire(tmp_path)
    try:
        ops = _FakeOps()
        assert run_live_session([], ops=ops, root=tmp_path, now=_NOW) == 2
        assert ops.calls == []
        assert ops.popen_calls == []
    finally:
        lock.close()


@pytest.mark.parametrize(
    "arguments",
    [
        ["--engine", "sherpa"],
        ["--record"],
        ["--input-device", "raw-source"],
        ["--enroll"],
        ["--device", "open_speaker"],
    ],
)
def test_launcher_rejects_options_that_break_its_evidence_contract(arguments):
    with pytest.raises(SystemExit) as exc:
        run_live_session(arguments, ops=_FakeOps(), root=Path("/unused"))
    assert exc.value.code == 2


def test_live_shell_is_a_thin_helpable_entrypoint(tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, SPEAKER_PYTHON=sys.executable)
    result = subprocess.run(
        ["bash", "live.sh", "--help"],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 0
    assert "reversible Linux open-speaker route" in result.stdout
    assert "live_launcher" not in result.stderr
