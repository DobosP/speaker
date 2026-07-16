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
    RealOperations,
    create_run_dir,
    run_live_session,
)


def _completed(
    command, returncode: int = 0, stdout: str = "", stderr: str = ""
):
    return subprocess.CompletedProcess(command, returncode, stdout, stderr)


class _FakeProcess:
    _next_pid = 2000

    def __init__(
        self,
        kind: str,
        *,
        returncode: int = 0,
        on_wait=None,
        hang_until_kill: bool = False,
    ):
        self.kind = kind
        self.returncode = returncode
        self.on_wait = on_wait
        self.hang_until_kill = hang_until_kill
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
        if self.running and self.hang_until_kill:
            raise subprocess.TimeoutExpired(self.kind, timeout)
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
        self.run_envs: list[tuple[tuple[str, ...], dict[str, str] | None]] = []
        self.popen_calls: list[tuple[tuple[str, ...], dict]] = []
        self.events: list[tuple[object, ...]] = []
        self.ollama_process: _FakeProcess | None = None
        self.voice_process: _FakeProcess | None = None
        self.ollama_lists_after_spawn = 0
        self.ollama_ready_after = 1
        self.fail_set_sink = False
        self.fail_restore_source = False
        self.invalid_module_output = False
        self.load_module_stdout: str | None = None
        self.load_module_error_after_publish: BaseException | None = None
        self.extra_module_arguments: list[str] = []
        self.extra_sources: set[str] = set()
        self.extra_sinks: set[str] = set()
        self.missing_commands: set[str] = set()
        self.doctor_error: BaseException | None = None
        self.voice_popen_error: BaseException | None = None
        self.ollama_popen_error: BaseException | None = None
        self.sleep_callback = None
        self.ollama_hang_until_kill = False
        self.ollama_exit_immediately = False
        self.voice_hang_until_kill = False
        self.voice_wait_callback = None
        self.set_sink_callback = None
        self.last_voice_env = None
        self.empty_default_probes = False
        self.pulse_server = "/run/user/1000/pulse/native"
        self.module_name = "module-echo-cancel"
        self.module_arguments = (
            "aec_method=webrtc "
            "source_name=echo-cancel-source "
            "sink_name=echo-cancel-sink "
            "source_master=raw-source sink_master=raw-sink "
            "aec_args=webrtc.noise_suppression=false "
            "webrtc.gain_control=false"
        )

    def which(self, name):
        if name in self.missing_commands:
            return None
        return f"/usr/bin/{name}" if name in {"pactl", "ollama"} else None

    def run(self, command, *, capture=False, env=None, timeout=None):
        command = tuple(command)
        self.calls.append(("run", command))
        self.run_envs.append((command, dict(env) if env is not None else None))
        self.events.append(("run", *command))
        if command == ("ollama", "list"):
            if self.ollama_process is not None:
                self.ollama_lists_after_spawn += 1
                if self.ollama_lists_after_spawn >= self.ollama_ready_after:
                    self.ollama_healthy = True
            return _completed(command, 0 if self.ollama_healthy else 1)
        if command == ("pactl", "info"):
            return _completed(
                command,
                stdout=f"Server String: {self.pulse_server}\n",
            )
        if command == ("pactl", "get-default-source"):
            stdout = "" if self.empty_default_probes else self.default_source + "\n"
            return _completed(command, stdout=stdout)
        if command == ("pactl", "get-default-sink"):
            stdout = "" if self.empty_default_probes else self.default_sink + "\n"
            return _completed(command, stdout=stdout)
        if command == ("pactl", "list", "short", "modules"):
            text = "1\tmodule-always-sink\n"
            if self.module_loaded:
                text += (
                    f"77\t{self.module_name}\t"
                    f"{self.module_arguments}\n"
                )
                for offset, arguments in enumerate(
                    self.extra_module_arguments, start=78
                ):
                    text += f"{offset}\tmodule-echo-cancel\t{arguments}\n"
            return _completed(command, stdout=text)
        if command == ("pactl", "list", "short", "sources"):
            text = "10\traw-source\tPipeWire\n"
            if self.nodes:
                text += "11\techo-cancel-source\tPipeWire\n"
            for offset, name in enumerate(sorted(self.extra_sources), start=12):
                text += f"{offset}\t{name}\tPipeWire\n"
            return _completed(command, stdout=text)
        if command == ("pactl", "list", "short", "sinks"):
            text = "20\traw-sink\tPipeWire\n"
            if self.nodes:
                text += "21\techo-cancel-sink\tPipeWire\n"
            for offset, name in enumerate(sorted(self.extra_sinks), start=22):
                text += f"{offset}\t{name}\tPipeWire\n"
            return _completed(command, stdout=text)
        if command[:3] == ("pactl", "load-module", "module-echo-cancel"):
            self.module_loaded = True
            self.nodes = True
            if self.load_module_error_after_publish is not None:
                raise self.load_module_error_after_publish
            if self.load_module_stdout is not None:
                return _completed(command, stdout=self.load_module_stdout)
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
            if target == "echo-cancel-sink" and self.set_sink_callback is not None:
                callback, self.set_sink_callback = self.set_sink_callback, None
                callback()
            return _completed(command)
        if command[:2] == ("pactl", "unload-module"):
            self.module_loaded = False
            self.nodes = False
            return _completed(command)
        if len(command) >= 3 and command[1:3] == ("-m", "tools.doctor"):
            if self.doctor_error is not None:
                raise self.doctor_error
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
        child_signal_mask=None,
    ):
        command = tuple(command)
        kwargs = {
            "env": env,
            "stdout": stdout,
            "stderr": stderr,
            "start_new_session": start_new_session,
            "child_signal_mask": child_signal_mask,
        }
        self.popen_calls.append((command, kwargs))
        self.events.append(("popen", *command))
        if command == ("ollama", "serve"):
            if self.ollama_popen_error is not None:
                raise self.ollama_popen_error
            self.ollama_process = _FakeProcess(
                "ollama", hang_until_kill=self.ollama_hang_until_kill
            )
            if self.ollama_exit_immediately:
                self.ollama_process.running = False
                self.ollama_process.returncode = 1
            return self.ollama_process
        if self.voice_popen_error is not None:
            raise self.voice_popen_error
        self.last_voice_env = env
        self.voice_process = _FakeProcess(
            "voice",
            returncode=self.voice_rc,
            on_wait=self.voice_wait_callback,
            hang_until_kill=self.voice_hang_until_kill,
        )
        return self.voice_process

    def sleep(self, _seconds):
        self.events.append(("sleep", _seconds))
        if self.sleep_callback is not None:
            callback, self.sleep_callback = self.sleep_callback, None
            callback()
        return None

    def signal_group(self, process, signum):
        self.events.append(("signal-group", process.kind, signum))
        process.signals.append(signum)
        if signum == signal.SIGTERM:
            process.terminated = True
        if signum == signal.SIGKILL:
            process.killed = True
        if not process.hang_until_kill or signum == signal.SIGKILL:
            process.running = False


_NOW = datetime(2026, 7, 16, 12, 34, 56)


def _commands(ops: _FakeOps) -> list[tuple[str, ...]]:
    return [command for _kind, command in ops.calls]


def _voice_command(ops: _FakeOps) -> tuple[str, ...]:
    return next(command for command, _kwargs in ops.popen_calls if "core" in command)


def _run_session(arguments, *, ops: _FakeOps, root: Path, now=_NOW) -> int:
    return run_live_session(
        arguments,
        ops=ops,
        root=root,
        lock_path=root / "host-live.lock",
        now=now,
    )


def test_one_command_creates_private_recorded_session_and_cleans_owned_resources(
    tmp_path, capsys
):
    ops = _FakeOps(nodes=False, ollama_healthy=False)

    rc = _run_session(
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
    assert "--record-pre-dsp-reference" in command
    assert "--record-playback-reference" in command
    assert command[-2:] == ("--mode", "assistant")
    assert ops.last_voice_env["SPEAKER_RUN_LOG_DIR"] == str(run_dir)
    assert ops.last_voice_env["SPEAKER_KEEP_RUNS"] == "0"
    assert "do not commit, push, or upload" in capsys.readouterr().out


def test_existing_ollama_and_echo_route_are_reused_not_destroyed(tmp_path):
    ops = _FakeOps(nodes=True, module_loaded=True, ollama_healthy=True)

    assert _run_session([], ops=ops, root=tmp_path) == 0

    commands = _commands(ops)
    assert not any(command[:2] == ("pactl", "load-module") for command in commands)
    assert not any(command[:2] == ("pactl", "unload-module") for command in commands)
    assert all(command != ("ollama", "serve") for command, _ in ops.popen_calls)
    assert ops.module_loaded is True
    assert ops.default_source == "raw-source"
    assert ops.default_sink == "raw-sink"


def test_existing_route_accepts_quoted_reordered_exact_module_arguments(tmp_path):
    ops = _FakeOps(nodes=True, module_loaded=True, ollama_healthy=True)
    ops.module_arguments = (
        "sink_master=raw-sink source_master=raw-source "
        'aec_args="webrtc.gain_control=false '
        'webrtc.noise_suppression=false" '
        "sink_name=echo-cancel-sink aec_method=webrtc "
        "source_name=echo-cancel-source"
    )

    assert _run_session([], ops=ops, root=tmp_path) == 0
    assert ops.module_loaded is True


def test_existing_canonical_defaults_reuse_safe_raw_masters(tmp_path):
    ops = _FakeOps(nodes=True, module_loaded=True, ollama_healthy=True)
    ops.default_source = "echo-cancel-source"
    ops.default_sink = "echo-cancel-sink"

    assert _run_session([], ops=ops, root=tmp_path) == 0

    assert ops.default_source == "echo-cancel-source"
    assert ops.default_sink == "echo-cancel-sink"
    assert ops.module_loaded is True
    assert not any(
        command[:2] == ("pactl", "unload-module") for command in _commands(ops)
    )


def test_existing_route_with_wrong_masters_is_rejected_before_microphone(tmp_path):
    ops = _FakeOps(nodes=True, module_loaded=True, ollama_healthy=True)
    ops.extra_sources.add("other-source")
    ops.module_arguments = ops.module_arguments.replace(
        "source_master=raw-source", "source_master=other-source"
    )

    assert _run_session([], ops=ops, root=tmp_path) == 2

    assert not any("core" in command for command, _ in ops.popen_calls)
    assert ops.module_loaded is True
    assert ops.default_source == "raw-source"
    assert ops.default_sink == "raw-sink"


@pytest.mark.parametrize(
    "module_arguments",
    [
        (
            "aec_method=webrtc source_name=echo-cancel-source "
            "sink_name=echo-cancel-sink source_master=raw-source "
            "sink_master=raw-sink "
            "aec_args=webrtc.noise_suppression=true "
            "webrtc.gain_control=false"
        ),
        (
            "aec_method=webrtc source_name=echo-cancel-source "
            "sink_name=echo-cancel-sink source_master=raw-source "
            "sink_master=raw-sink "
            "aec_args=webrtc.noise_suppression=false"
        ),
    ],
)
def test_existing_route_with_noncanonical_aec_args_is_rejected(
    tmp_path, module_arguments
):
    ops = _FakeOps(nodes=True, module_loaded=True, ollama_healthy=True)
    ops.module_arguments = module_arguments

    assert _run_session([], ops=ops, root=tmp_path) == 2
    assert not any("core" in command for command, _ in ops.popen_calls)


def test_duplicate_canonical_echo_modules_are_rejected(tmp_path):
    ops = _FakeOps(nodes=True, module_loaded=True, ollama_healthy=True)
    ops.extra_module_arguments.append(ops.module_arguments)

    assert _run_session([], ops=ops, root=tmp_path) == 2

    assert not any("core" in command for command, _ in ops.popen_calls)
    assert not any(
        command[:2] == ("pactl", "unload-module") for command in _commands(ops)
    )


def test_split_canonical_node_claimants_are_rejected_without_mutation(tmp_path):
    ops = _FakeOps(nodes=True, module_loaded=True, ollama_healthy=True)
    ops.module_arguments = ops.module_arguments.replace(
        "sink_name=echo-cancel-sink", "sink_name=other-sink"
    )
    ops.extra_module_arguments.append(
        ops.module_arguments.replace(
            "source_name=echo-cancel-source", "source_name=other-source"
        ).replace("sink_name=other-sink", "sink_name=echo-cancel-sink")
    )

    assert _run_session([], ops=ops, root=tmp_path) == 2

    assert ops.default_source == "raw-source"
    assert ops.default_sink == "raw-sink"
    assert ops.module_loaded is True
    assert not any(
        command[:2] in {
            ("pactl", "set-default-source"),
            ("pactl", "set-default-sink"),
            ("pactl", "unload-module"),
        }
        for command in _commands(ops)
    )
    assert not any("core" in command for command, _ in ops.popen_calls)


def test_noncanonical_echo_cancel_defaults_are_rejected_without_stacking(tmp_path):
    ops = _FakeOps(nodes=True, module_loaded=True, ollama_healthy=True)
    ops.default_source = "ec_source"
    ops.default_sink = "ec_sink"

    assert _run_session([], ops=ops, root=tmp_path) == 2

    assert not any(
        command[:2] == ("pactl", "load-module") for command in _commands(ops)
    )
    assert not any("core" in command for command, _ in ops.popen_calls)


def test_echo_llm_skips_ollama_and_uses_deferred_doctor(tmp_path):
    ops = _FakeOps(nodes=True, ollama_healthy=False)

    assert _run_session(["--llm", "echo"], ops=ops, root=tmp_path) == 0

    assert all(command != ("ollama", "serve") for command, _ in ops.popen_calls)
    doctor = next(
        command for command in _commands(ops) if command[1:3] == ("-m", "tools.doctor")
    )
    assert doctor[-1] == "--defer-ollama"


@pytest.mark.parametrize("requested_device", [None, "unsafe"])
def test_selected_default_or_requested_profile_with_in_app_aec_is_rejected(
    tmp_path, requested_device
):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "device": "safe" if requested_device else "unsafe",
                "device_profiles": {
                    "safe": {"sherpa": {"aec_enabled": False}},
                    "unsafe": {"sherpa": {"aec_enabled": True}},
                },
                "sherpa": {"aec_enabled": False},
                "llm": {"backend": "llamacpp"},
            }
        ),
        encoding="utf-8",
    )
    ops = _FakeOps(nodes=False, ollama_healthy=False)
    arguments = ["--device", requested_device] if requested_device else []

    assert _run_session(arguments, ops=ops, root=tmp_path) == 2

    assert ops.calls == []
    assert ops.popen_calls == []


def test_machine_local_override_with_in_app_aec_is_rejected(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("SPEAKER_NO_LOCAL_CONFIG", raising=False)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "device": "safe",
                "device_profiles": {
                    "safe": {"sherpa": {"aec_enabled": False}},
                },
                "sherpa": {"aec_enabled": False},
                "llm": {"backend": "llamacpp"},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "config.local.json").write_text(
        json.dumps(
            {
                "device_profiles": {
                    "safe": {"sherpa": {"aec_enabled": True}},
                }
            }
        ),
        encoding="utf-8",
    )
    ops = _FakeOps(nodes=False, ollama_healthy=False)

    assert _run_session([], ops=ops, root=tmp_path) == 2

    assert ops.calls == []
    assert ops.popen_calls == []


@pytest.mark.parametrize(
    ("selector", "raw_device"),
    [
        ("input_device", "raw-source"),
        ("output_device", "raw-sink"),
    ],
)
def test_selected_profile_cannot_bypass_os_ec_with_a_raw_device(
    tmp_path, selector, raw_device
):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "device": "unsafe",
                "device_profiles": {
                    "unsafe": {
                        "sherpa": {
                            "aec_enabled": False,
                            selector: raw_device,
                        }
                    }
                },
                "sherpa": {"aec_enabled": False},
                "llm": {"backend": "llamacpp"},
            }
        ),
        encoding="utf-8",
    )
    ops = _FakeOps(nodes=False, ollama_healthy=False)

    assert _run_session([], ops=ops, root=tmp_path) == 2

    assert ops.calls == []
    assert ops.popen_calls == []


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

    assert _run_session([], ops=ops, root=tmp_path) == 0

    assert ("ollama", "list") not in _commands(ops)
    assert all(command != ("ollama", "serve") for command, _ in ops.popen_calls)


def test_llamacpp_profile_with_echo_uses_full_local_llm_deferral(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "device": "local_cpu",
                "device_profiles": {
                    "local_cpu": {"llm": {"backend": "llamacpp"}},
                },
                "llm": {"backend": "ollama"},
            }
        ),
        encoding="utf-8",
    )
    ops = _FakeOps(nodes=True, ollama_healthy=False)

    assert _run_session(["--llm", "echo"], ops=ops, root=tmp_path) == 0

    doctor = next(
        command for command in _commands(ops) if command[1:3] == ("-m", "tools.doctor")
    )
    assert doctor[-1] == "--defer-llm"
    assert "--defer-ollama" not in doctor
    assert ("ollama", "list") not in _commands(ops)


def test_ambient_remote_ollama_host_is_overridden_with_loopback(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("OLLAMA_HOST", "http://remote.example:11434")
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8080")
    monkeypatch.setenv("ALL_PROXY", "socks5://proxy.example:1080")
    monkeypatch.setenv("NO_PROXY", "unrelated.example")
    monkeypatch.setenv("no_proxy", "another.example")
    ops = _FakeOps(nodes=True, ollama_healthy=True)

    assert _run_session([], ops=ops, root=tmp_path) == 0

    ollama_envs = [
        env for command, env in ops.run_envs if command == ("ollama", "list")
    ]
    assert ollama_envs
    assert {env["OLLAMA_HOST"] for env in ollama_envs if env is not None} == {
        "http://127.0.0.1:11434"
    }
    assert ops.last_voice_env["OLLAMA_HOST"] == "http://127.0.0.1:11434"
    assert "127.0.0.1" in ops.last_voice_env["NO_PROXY"].split(",")
    assert "127.0.0.1" in ops.last_voice_env["no_proxy"].split(",")
    assert ops.last_voice_env["HTTP_PROXY"] == "http://proxy.example:8080"
    assert ops.last_voice_env["HTTPS_PROXY"] == "http://proxy.example:8080"
    assert ops.last_voice_env["ALL_PROXY"] == "socks5://proxy.example:1080"


def test_ambient_audio_client_overrides_are_removed_from_doctor_and_core(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PULSE_SOURCE", "raw-source")
    monkeypatch.setenv("PULSE_SINK", "raw-sink")
    monkeypatch.setenv("ALSA_CONFIG_PATH", "/tmp/hostile-virtual-asound.conf")
    ops = _FakeOps(nodes=True, ollama_healthy=True)

    assert _run_session([], ops=ops, root=tmp_path) == 0

    doctor_env = next(
        env
        for command, env in ops.run_envs
        if len(command) >= 3 and command[1:3] == ("-m", "tools.doctor")
    )
    assert doctor_env is not None
    for name in ("PULSE_SOURCE", "PULSE_SINK", "ALSA_CONFIG_PATH"):
        assert name not in doctor_env
        assert name not in ops.last_voice_env


def test_configured_remote_ollama_host_is_rejected_before_host_setup(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "device": "local_cpu",
                "device_profiles": {
                    "local_cpu": {
                        "llm": {
                            "backend": "ollama",
                            "host": "http://remote.example:11434",
                        }
                    }
                },
                "llm": {
                    "backend": "ollama",
                    "host": "http://remote.example:11434",
                }
            }
        ),
        encoding="utf-8",
    )
    ops = _FakeOps(nodes=False, ollama_healthy=True)

    assert _run_session([], ops=ops, root=tmp_path) == 2

    assert ops.calls == []
    assert ops.popen_calls == []


def test_bad_port_with_embedded_credentials_never_echoes_the_secret(
    tmp_path, capsys
):
    secret = "vault-token-should-never-print"
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "device": "local_cpu",
                "device_profiles": {
                    "local_cpu": {"sherpa": {"aec_enabled": False}},
                },
                "llm": {
                    "backend": "ollama",
                    "host": f"http://paul:{secret}@127.0.0.1:not-a-port",
                },
            }
        ),
        encoding="utf-8",
    )
    ops = _FakeOps(nodes=False, ollama_healthy=False)

    assert _run_session([], ops=ops, root=tmp_path) == 2

    output = capsys.readouterr()
    assert secret not in output.out
    assert secret not in output.err
    assert "traceback" not in output.err.lower()
    assert ops.calls == []
    assert ops.popen_calls == []


def test_configured_loopback_ollama_host_is_preserved(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "device": "local_cpu",
                "device_profiles": {
                    "local_cpu": {
                        "llm": {
                            "backend": "ollama",
                            "host": "http://127.0.0.2:11434",
                        }
                    }
                },
                "llm": {
                    "backend": "ollama",
                    "host": "http://127.0.0.2:11434",
                }
            }
        ),
        encoding="utf-8",
    )
    ops = _FakeOps(nodes=True, ollama_healthy=True)

    assert _run_session([], ops=ops, root=tmp_path) == 0

    ollama_env = next(
        env for command, env in ops.run_envs if command == ("ollama", "list")
    )
    assert ollama_env is not None
    assert ollama_env["OLLAMA_HOST"] == "http://127.0.0.2:11434"


def test_doctor_failure_never_opens_microphone_and_cleans_up(tmp_path):
    ops = _FakeOps(nodes=False, ollama_healthy=True, doctor_rc=9)

    assert _run_session([], ops=ops, root=tmp_path) == 2

    assert not any("core" in command for command, _ in ops.popen_calls)
    assert ops.default_source == "raw-source"
    assert ops.default_sink == "raw-sink"
    assert ops.module_loaded is False


@pytest.mark.parametrize("platform", ["darwin", "win32"])
def test_non_linux_platform_fails_before_audio_commands(tmp_path, platform):
    ops = _FakeOps(ollama_healthy=False)
    ops.platform = platform

    assert _run_session(["--llm", "echo"], ops=ops, root=tmp_path) == 2

    assert not any(command[0] == "pactl" for command in _commands(ops))
    assert ops.popen_calls == []


def test_missing_pactl_fails_before_audio_commands(tmp_path):
    ops = _FakeOps(ollama_healthy=False)
    ops.missing_commands.add("pactl")

    assert _run_session(["--llm", "echo"], ops=ops, root=tmp_path) == 2

    assert not any(command[0] == "pactl" for command in _commands(ops))
    assert ops.popen_calls == []


def test_remote_ambient_pulse_server_is_rejected_before_any_audio_probe(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PULSE_SERVER", "tcp:audio.example:4713")
    ops = _FakeOps(nodes=False, ollama_healthy=False)

    assert _run_session(["--llm", "echo"], ops=ops, root=tmp_path) == 2

    assert not any(command[0] == "pactl" for command in _commands(ops))
    assert ops.popen_calls == []


def test_remote_pactl_server_identity_is_rejected_before_route_mutation(tmp_path):
    ops = _FakeOps(nodes=False, ollama_healthy=False)
    ops.pulse_server = "tcp:audio.example:4713"

    assert _run_session(["--llm", "echo"], ops=ops, root=tmp_path) == 2

    pactl_commands = [
        command for command in _commands(ops) if command and command[0] == "pactl"
    ]
    assert pactl_commands == [("pactl", "info")]
    assert not any("core" in command for command, _ in ops.popen_calls)


def test_doctor_oserror_cleans_owned_route_without_opening_microphone(tmp_path):
    ops = _FakeOps(nodes=False, ollama_healthy=True)
    ops.doctor_error = OSError("doctor executable unavailable")

    assert _run_session([], ops=ops, root=tmp_path) == 2

    assert ops.default_source == "raw-source"
    assert ops.default_sink == "raw-sink"
    assert ops.module_loaded is False
    assert not any("core" in command for command, _ in ops.popen_calls)


def test_voice_popen_oserror_cleans_all_owned_host_resources(tmp_path):
    ops = _FakeOps(nodes=False, ollama_healthy=False)
    ops.voice_popen_error = OSError("voice executable unavailable")

    assert _run_session([], ops=ops, root=tmp_path) == 2

    assert ops.default_source == "raw-source"
    assert ops.default_sink == "raw-sink"
    assert ops.module_loaded is False
    assert ops.ollama_process is not None and ops.ollama_process.terminated


def test_ollama_popen_oserror_never_touches_audio_route(tmp_path):
    ops = _FakeOps(nodes=False, ollama_healthy=False)
    ops.ollama_popen_error = OSError("ollama executable unavailable")

    assert _run_session([], ops=ops, root=tmp_path) == 2

    assert not any(command[0] == "pactl" for command in _commands(ops))
    assert not any("core" in command for command, _ in ops.popen_calls)


def test_ollama_startup_timeout_stops_owned_group_without_touching_audio(tmp_path):
    ops = _FakeOps(nodes=False, ollama_healthy=False)
    ops.ollama_ready_after = 1000

    assert _run_session([], ops=ops, root=tmp_path) == 2

    assert ops.ollama_process is not None
    assert ops.ollama_process.signals == [signal.SIGTERM]
    assert not any(command[0] == "pactl" for command in _commands(ops))
    assert not any("core" in command for command, _ in ops.popen_calls)


def test_ollama_early_exit_never_touches_audio_route(tmp_path):
    ops = _FakeOps(nodes=False, ollama_healthy=False)
    ops.ollama_ready_after = 1000
    ops.ollama_exit_immediately = True

    assert _run_session([], ops=ops, root=tmp_path) == 2

    assert ops.ollama_process is not None
    assert ops.ollama_process.poll() == 1
    assert ops.ollama_process.signals == []
    assert not any(command[0] == "pactl" for command in _commands(ops))
    assert not any("core" in command for command, _ in ops.popen_calls)


def test_partial_route_failure_restores_source_and_unloads_owned_module(tmp_path):
    ops = _FakeOps(nodes=False, ollama_healthy=True)
    ops.fail_set_sink = True

    assert _run_session([], ops=ops, root=tmp_path) == 2

    assert ops.default_source == "raw-source"
    assert ops.default_sink == "raw-sink"
    assert ops.module_loaded is False
    assert not any("core" in command for command, _ in ops.popen_calls)


def test_setup_and_restore_failures_are_both_reported_and_module_is_retained(
    tmp_path, capsys
):
    ops = _FakeOps(nodes=False, ollama_healthy=True)
    ops.fail_set_sink = True
    ops.fail_restore_source = True

    assert _run_session([], ops=ops, root=tmp_path) == 2

    assert ops.default_source == "echo-cancel-source"
    assert ops.default_sink == "raw-sink"
    assert ops.module_loaded is True
    stderr = capsys.readouterr().err
    assert "forced sink selection failure" in stderr
    assert "forced source restore failure" in stderr
    assert "retained" in stderr


def test_cleanup_preserves_newer_external_default_changes(tmp_path):
    ops = _FakeOps(nodes=False, ollama_healthy=True)

    def external_change():
        ops.default_source = "external-source"
        ops.default_sink = "external-sink"

    ops.voice_wait_callback = external_change
    assert _run_session([], ops=ops, root=tmp_path) == 0

    assert ops.default_source == "external-source"
    assert ops.default_sink == "external-sink"
    assert ops.module_loaded is False


def test_failed_restore_retains_module_and_makes_result_red(tmp_path, capsys):
    ops = _FakeOps(nodes=False, ollama_healthy=True)
    ops.fail_restore_source = True

    assert _run_session([], ops=ops, root=tmp_path) == 2

    assert ops.default_source == "echo-cancel-source"
    assert ops.default_sink == "raw-sink"
    assert ops.module_loaded is True
    assert not any(
        command[:2] == ("pactl", "unload-module") for command in _commands(ops)
    )
    assert "retained" in capsys.readouterr().err


def test_core_failure_propagates_after_cleanup_and_keeps_run_directory(tmp_path):
    ops = _FakeOps(nodes=False, ollama_healthy=True, voice_rc=7)

    assert _run_session([], ops=ops, root=tmp_path) == 7

    assert (tmp_path / "logs/live/manual-20260716-123456").is_dir()
    assert ops.module_loaded is False
    assert ops.default_source == "raw-source"
    assert ops.default_sink == "raw-sink"


def test_invalid_module_stdout_is_recovered_from_exact_new_module_identity(tmp_path):
    ops = _FakeOps(nodes=False, ollama_healthy=True)
    ops.invalid_module_output = True

    assert _run_session([], ops=ops, root=tmp_path) == 0
    assert ops.module_loaded is False


def test_wrong_numeric_load_id_is_never_used_for_unload(tmp_path, capsys):
    ops = _FakeOps(nodes=False, ollama_healthy=True)
    ops.load_module_stdout = "999\n"

    assert _run_session([], ops=ops, root=tmp_path) == 2

    unloads = [
        command
        for command in _commands(ops)
        if command[:2] == ("pactl", "unload-module")
    ]
    assert unloads == []
    assert ops.module_loaded is True
    stderr = capsys.readouterr().err
    assert "does not match" in stderr
    assert "unidentified owned module retained" in stderr


def test_timed_out_load_that_published_exact_module_is_reconciled_and_unloaded(
    tmp_path, capsys
):
    ops = _FakeOps(nodes=False, ollama_healthy=True)
    ops.load_module_error_after_publish = subprocess.TimeoutExpired(
        ("pactl", "load-module", "module-echo-cancel"), 5.0
    )

    assert _run_session([], ops=ops, root=tmp_path) == 2

    assert ("pactl", "unload-module", "77") in _commands(ops)
    assert ops.module_loaded is False
    assert "timed out" in capsys.readouterr().err.lower()


def test_invalid_module_stdout_with_ambiguous_new_ids_is_retained_and_red(
    tmp_path, capsys
):
    ops = _FakeOps(nodes=False, ollama_healthy=True)
    ops.invalid_module_output = True
    ops.extra_module_arguments.append(ops.module_arguments)

    assert _run_session([], ops=ops, root=tmp_path) == 2

    assert ops.module_loaded is True
    assert not any(
        command[:2] == ("pactl", "unload-module") for command in _commands(ops)
    )
    stderr = capsys.readouterr().err
    assert "no unique safe module id" in stderr
    assert "unidentified owned module retained" in stderr


def test_cleanup_never_unloads_a_reused_numeric_module_id(tmp_path, capsys):
    ops = _FakeOps(nodes=False, ollama_healthy=True)

    def replace_owned_module():
        ops.module_name = "module-null-sink"
        ops.nodes = False

    ops.voice_wait_callback = replace_owned_module

    assert _run_session([], ops=ops, root=tmp_path) == 2

    assert ops.module_loaded is True
    assert not any(
        command[:2] == ("pactl", "unload-module") for command in _commands(ops)
    )
    assert "numeric id now belongs to a different module" in capsys.readouterr().err


def test_empty_successful_default_probes_retain_owned_module_and_fail_red(
    tmp_path, capsys
):
    ops = _FakeOps(nodes=False, ollama_healthy=True)
    ops.voice_wait_callback = lambda: setattr(ops, "empty_default_probes", True)

    assert _run_session([], ops=ops, root=tmp_path) == 2

    assert ops.module_loaded is True
    assert not any(
        command[:2] == ("pactl", "unload-module") for command in _commands(ops)
    )
    stderr = capsys.readouterr().err.lower()
    assert "empty default source probe" in stderr
    assert "empty default sink probe" in stderr


def test_setup_sigterm_cleans_owned_ollama_and_restores_handlers(tmp_path):
    ops = _FakeOps(nodes=False, ollama_healthy=False)
    ops.ollama_ready_after = 1000
    ops.sleep_callback = lambda: signal.raise_signal(signal.SIGTERM)
    original_handler = signal.getsignal(signal.SIGTERM)

    assert _run_session([], ops=ops, root=tmp_path) == 128 + signal.SIGTERM

    assert signal.getsignal(signal.SIGTERM) is original_handler
    assert ops.ollama_process is not None
    assert ops.ollama_process.signals == [signal.SIGTERM]
    assert not any(command[0] == "pactl" for command in _commands(ops))
    assert not any("core" in command for command, _ in ops.popen_calls)


@pytest.mark.skipif(not hasattr(signal, "SIGHUP"), reason="SIGHUP is POSIX-only")
def test_sighup_maps_to_child_sigterm_then_cleans_route(tmp_path):
    ops = _FakeOps(nodes=False, ollama_healthy=True)
    ops.voice_wait_callback = lambda: signal.raise_signal(signal.SIGHUP)

    assert _run_session([], ops=ops, root=tmp_path) == 128 + signal.SIGHUP

    assert ops.voice_process is not None
    assert ops.voice_process.signals == [signal.SIGTERM]
    assert ops.default_source == "raw-source"
    assert ops.default_sink == "raw-sink"
    assert ops.module_loaded is False


@pytest.mark.skipif(
    not hasattr(signal, "pthread_sigmask"),
    reason="pending-signal route handoff is POSIX-only",
)
def test_pending_sigterm_at_final_route_store_still_cleans_owned_route(tmp_path):
    ops = _FakeOps(nodes=False, ollama_healthy=True)
    ops.set_sink_callback = lambda: signal.raise_signal(signal.SIGTERM)

    assert _run_session([], ops=ops, root=tmp_path) == 128 + signal.SIGTERM

    assert ops.default_source == "raw-source"
    assert ops.default_sink == "raw-sink"
    assert ops.module_loaded is False
    assert ("pactl", "unload-module", "77") in _commands(ops)
    assert not any("core" in command for command, _ in ops.popen_calls)


@pytest.mark.skipif(
    not hasattr(signal, "pthread_sigmask"),
    reason="child signal-mask proof is POSIX-only",
)
@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_real_child_restores_parent_preblock_mask_and_handles_signal(signum):
    original_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signum})
    if signum in original_mask:
        signal.pthread_sigmask(signal.SIG_SETMASK, original_mask)
        pytest.skip(f"signal {signum} was already blocked by the test host")
    process = None
    try:
        process = RealOperations().popen(
            [
                sys.executable,
                "-c",
                (
                    "import signal,sys; "
                    "stop=lambda *_: sys.exit(0); "
                    "signal.signal(signal.SIGINT,stop); "
                    "signal.signal(signal.SIGTERM,stop); "
                    "blocked=signal.pthread_sigmask(signal.SIG_BLOCK,[]); "
                    "sys.exit(9) if (signal.SIGINT in blocked or "
                    "signal.SIGTERM in blocked) else None; "
                    "print('ready',flush=True); signal.pause()"
                ),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            child_signal_mask=original_mask,
        )
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, original_mask)

    try:
        assert process.stdout is not None
        assert process.stdout.readline() == b"ready\n"
        os.killpg(process.pid, signum)
        assert process.wait(timeout=2.0) == 0
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=2.0)


def test_hung_voice_group_escalates_to_sigkill_and_route_still_cleans(tmp_path):
    ops = _FakeOps(nodes=False, ollama_healthy=True)
    ops.voice_hang_until_kill = True
    ops.voice_wait_callback = lambda: signal.raise_signal(signal.SIGINT)

    assert _run_session([], ops=ops, root=tmp_path) == 128 + signal.SIGINT

    assert ops.voice_process is not None
    assert ops.voice_process.signals == [
        signal.SIGINT,
        signal.SIGTERM,
        signal.SIGKILL,
    ]
    assert ops.voice_process.killed
    voice_kwargs = next(
        kwargs for command, kwargs in ops.popen_calls if "core" in command
    )
    assert voice_kwargs["start_new_session"] is True
    assert ops.default_source == "raw-source"
    assert ops.default_sink == "raw-sink"
    assert ops.module_loaded is False


def test_hung_owned_ollama_group_escalates_to_sigkill(tmp_path):
    ops = _FakeOps(nodes=True, ollama_healthy=False)
    ops.ollama_hang_until_kill = True

    assert _run_session([], ops=ops, root=tmp_path) == 0

    assert ops.ollama_process is not None
    assert ops.ollama_process.signals == [signal.SIGTERM, signal.SIGKILL]
    assert ops.ollama_process.killed


def test_setup_doctor_voice_and_cleanup_order_is_explicit(tmp_path):
    ops = _FakeOps(nodes=False, ollama_healthy=True)

    assert _run_session([], ops=ops, root=tmp_path) == 0

    events = ops.events
    loaded = events.index(
        (
            "run",
            "pactl",
            "load-module",
            "module-echo-cancel",
            "aec_method=webrtc",
            "source_name=echo-cancel-source",
            "sink_name=echo-cancel-sink",
            "source_master=raw-source",
            "sink_master=raw-sink",
            "aec_args=webrtc.noise_suppression=false webrtc.gain_control=false",
        )
    )
    selected_source = events.index(
        ("run", "pactl", "set-default-source", "echo-cancel-source")
    )
    selected_sink = events.index(
        ("run", "pactl", "set-default-sink", "echo-cancel-sink")
    )
    doctor = next(
        index
        for index, event in enumerate(events)
        if event[:4] == ("run", sys.executable, "-m", "tools.doctor")
    )
    voice = next(
        index
        for index, event in enumerate(events)
        if event[:4] == ("popen", sys.executable, "-m", "core")
    )
    restored_source = events.index(
        ("run", "pactl", "set-default-source", "raw-source")
    )
    restored_sink = events.index(
        ("run", "pactl", "set-default-sink", "raw-sink")
    )
    unloaded = events.index(("run", "pactl", "unload-module", "77"))
    assert (
        loaded
        < selected_source
        < selected_sink
        < doctor
        < voice
        < restored_source
        < restored_sink
        < unloaded
    )


def test_unique_run_directories_are_never_reused(tmp_path):
    first = create_run_dir(tmp_path, "manual", now=_NOW)
    marker = first / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")
    second = create_run_dir(tmp_path, "manual", now=_NOW)

    assert first.name == "manual-20260716-123456"
    assert second.name == "manual-20260716-123456-01"
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_live_session_lock_symlink_is_rejected_before_host_setup(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    target = tmp_path / "lock-target"
    target.write_text("unrelated", encoding="utf-8")
    (root / "host-live.lock").symlink_to(target)
    ops = _FakeOps()

    assert _run_session([], ops=ops, root=root) == 2

    assert target.read_text(encoding="utf-8") == "unrelated"
    assert ops.calls == []
    assert ops.popen_calls == []


@pytest.mark.parametrize("symlink_parent", ["logs", "live"])
def test_private_run_parent_symlink_is_rejected_before_host_setup(
    tmp_path, symlink_parent
):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    if symlink_parent == "logs":
        (root / "logs").symlink_to(outside, target_is_directory=True)
    else:
        (root / "logs").mkdir()
        (root / "logs" / "live").symlink_to(outside, target_is_directory=True)
    ops = _FakeOps()

    assert _run_session([], ops=ops, root=root) == 2

    assert list(outside.iterdir()) == []
    assert ops.calls == []
    assert ops.popen_calls == []


def test_concurrent_launcher_is_rejected_without_touching_audio(tmp_path):
    lock_path = tmp_path / "global-live.lock"
    lock = LiveSessionLock.acquire(lock_path)
    try:
        ops = _FakeOps()
        assert run_live_session(
            [], ops=ops, root=tmp_path / "other-worktree", lock_path=lock_path, now=_NOW
        ) == 2
        assert ops.calls == []
        assert ops.popen_calls == []
        assert not (tmp_path / "other-worktree").exists()
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
        ["--agent"],
        ["--gui-actions"],
        ["--planner"],
    ],
)
def test_launcher_rejects_options_that_break_its_evidence_contract(arguments):
    with pytest.raises(SystemExit) as exc:
        run_live_session(arguments, ops=_FakeOps(), root=Path("/unused"))
    assert exc.value.code == 2


@pytest.mark.parametrize(
    "arguments",
    [
        ["--model", "first", "--model", "second"],
        ["--model=first", "--model=second"],
        ["--model", "first", "--model=second"],
        ["--run-label", "first", "--run-label=second"],
    ],
)
def test_launcher_rejects_duplicate_runtime_options(arguments):
    with pytest.raises(SystemExit) as exc:
        run_live_session(arguments, ops=_FakeOps(), root=Path("/unused"))
    assert exc.value.code == 2


@pytest.mark.parametrize("option", ["--dev", "--mod", "--run-lab"])
def test_launcher_rejects_abbreviated_options(option):
    with pytest.raises(SystemExit) as exc:
        run_live_session([option, "value"], ops=_FakeOps(), root=Path("/unused"))
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
