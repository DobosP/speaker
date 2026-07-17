"""Tests for the cross-platform installer helpers (tools.install).

Pure / injected -- no venv is created, no subprocess runs. The OS is passed
explicitly so both the Windows and POSIX paths are exercised on Linux CI.
"""
from __future__ import annotations

import argparse

from tools.install import (
    FINAL_VERIFIER_RUNTIME_DEPS,
    RUNTIME_DEPS,
    SELECTED_MODEL_ARGS,
    activation_hint,
    capability_setup_args,
    final_verifier_supported,
    install_plan,
    is_windows,
    main,
    needs_fresh_venv,
    normal_voice_entry,
    portaudio_hint,
    runtime_deps,
    selected_model_args,
    venv_python_path,
)


def test_is_windows_by_platform_string():
    assert is_windows("win32")
    assert not is_windows("linux")


def test_normal_voice_entry_uses_linux_session_wrapper():
    assert normal_voice_entry("linux") == "./live.sh"
    assert normal_voice_entry("win32") == "python -m core --engine sherpa"
    assert not is_windows("darwin")


def test_venv_python_path_per_os():
    # Windows: Scripts\python.exe; POSIX: bin/python.
    win = venv_python_path(".venv", system="win32")
    assert win.endswith("python.exe")
    assert "Scripts" in win
    posix = venv_python_path(".venv", system="linux")
    assert posix.endswith("python")
    assert "bin" in posix


def test_activation_hint_per_os():
    assert "Activate.ps1" in activation_hint(".venv", system="win32")
    assert activation_hint(".venv", system="linux") == "source .venv/bin/activate"


def test_portaudio_hint_per_os():
    assert "wheel" in portaudio_hint(system="win32").lower()
    assert "brew" in portaudio_hint(system="darwin").lower()
    assert "apt-get" in portaudio_hint(system="linux")


def test_needs_fresh_venv_when_interpreter_missing():
    assert needs_fresh_venv(".venv", exists=lambda p: False) is True


def test_needs_fresh_venv_when_pip_broken():
    # Interpreter present but pip is broken (the conda-mix failure) -> rebuild.
    assert needs_fresh_venv(".venv", exists=lambda p: True, pip_ok=lambda p: False) is True
    # Interpreter present and pip works -> reuse.
    assert needs_fresh_venv(".venv", exists=lambda p: True, pip_ok=lambda p: True) is False
    # Present, no pip probe given -> assume usable (don't needlessly rebuild).
    assert needs_fresh_venv(".venv", exists=lambda p: True) is False


def _args(**over):
    base = dict(
        venv=".venv",
        recreate=False,
        skip_models=False,
        final_asr=None,
        final_verifier=None,
        obsidian_vault=None,
        disable_obsidian=False,
        enable_reminders=False,
        disable_reminders=False,
        trust_app=[],
        untrust_app=[],
    )
    base.update(over)
    return argparse.Namespace(**base)


def test_install_plan_includes_deps_and_models():
    plan = install_plan(_args(), system="linux")
    text = "\n".join(plan)
    assert any(dep in text for dep in RUNTIME_DEPS)
    assert "setup_models" in text
    assert all(flag in text for flag in SELECTED_MODEL_ARGS)
    assert "doctor --defer-ollama" in text


def test_lean_runtime_deps_include_required_signal_resamplers():
    assert "scipy>=1.13" in RUNTIME_DEPS
    assert "soxr>=0.3" in RUNTIME_DEPS


def test_final_verifier_is_explicit_linux_x86_64_opt_in():
    default = _args()
    selected = _args(final_verifier="faster-whisper-small")

    assert not any(dep in runtime_deps(default) for dep in FINAL_VERIFIER_RUNTIME_DEPS)
    assert all(dep in runtime_deps(selected) for dep in FINAL_VERIFIER_RUNTIME_DEPS)
    assert selected_model_args(default) == SELECTED_MODEL_ARGS
    assert selected_model_args(selected)[-2:] == (
        "--final-verifier",
        "faster-whisper-small",
    )
    assert final_verifier_supported(system="linux", machine="x86_64")
    assert not final_verifier_supported(system="win32", machine="AMD64")
    assert not final_verifier_supported(system="linux", machine="aarch64")


def test_parakeet_final_asr_replaces_fresh_install_sensevoice_selection():
    selected = selected_model_args(_args(final_asr="parakeet-unified-en"))

    assert "--sense-voice" not in selected
    assert selected[-2:] == ("--final-asr", "parakeet-unified-en")


def test_install_plan_includes_selected_final_verifier_only_when_requested():
    default = "\n".join(install_plan(_args(), system="linux"))
    selected = "\n".join(
        install_plan(
            _args(final_verifier="faster-whisper-small"),
            system="linux",
        )
    )

    assert "faster-whisper==1.2.1" not in default
    assert "--final-verifier" not in default
    assert "faster-whisper==1.2.1" in selected
    assert "--final-verifier faster-whisper-small" in selected


def test_install_plan_respects_skip_models():
    plan = install_plan(_args(skip_models=True), system="linux")
    text = "\n".join(plan)
    assert "skip model download" in text
    assert "incomplete" in text
    assert "tools.setup_models" not in text
    assert "tools.doctor" not in text


def test_install_plan_uses_windows_interpreter_path():
    plan = install_plan(_args(), system="win32")
    assert any("python.exe" in step for step in plan)


def test_capability_setup_args_preserve_repeated_app_options():
    args = capability_setup_args(
        _args(
            obsidian_vault="/vault with spaces",
            enable_reminders=True,
            trust_app=["notes=notes.desktop", "browser=browser.desktop"],
            untrust_app=["old-notes", "old-browser"],
        )
    )
    assert args == [
        "--obsidian-vault", "/vault with spaces",
        "--enable-reminders",
        "--trust-app", "notes=notes.desktop",
        "--trust-app", "browser=browser.desktop",
        "--untrust-app", "old-notes",
        "--untrust-app", "old-browser",
    ]


def test_install_plan_places_capability_setup_between_models_and_doctor():
    plan = install_plan(
        _args(enable_reminders=True, trust_app=["notes=notes.desktop"]),
        system="linux",
    )
    model_index = next(i for i, step in enumerate(plan) if "tools.setup_models" in step)
    setup_index = next(i for i, step in enumerate(plan) if "tools.setup_assistant" in step)
    doctor_index = next(i for i, step in enumerate(plan) if "tools.doctor" in step)
    assert model_index < setup_index < doctor_index
    assert "--enable-reminders" in plan[setup_index]
    assert "notes=notes.desktop" in plan[setup_index]


def _run_installer(monkeypatch, responses, *argv):
    import tools.install as installer

    calls: list[list[str]] = []
    remaining = iter(responses)
    monkeypatch.setattr(
        installer,
        "create_venv",
        lambda *_args, **_kwargs: "/venv/bin/python",
    )

    def fake_run(command, *, dry_run):
        assert dry_run is False
        calls.append(command)
        return next(remaining)

    monkeypatch.setattr(installer, "_run", fake_run)
    return main(["--venv", "/venv", *argv]), calls


def test_installer_propagates_selected_model_setup_failure(monkeypatch):
    rc, calls = _run_installer(monkeypatch, (0, 7))

    assert rc == 7
    assert calls[1] == [
        "/venv/bin/python", "-m", "tools.setup_models", *SELECTED_MODEL_ARGS,
    ]
    assert all("tools.doctor" not in command for command in calls)


def test_installer_final_verifier_installs_pins_then_forwards_setup(monkeypatch):
    rc, calls = _run_installer(
        monkeypatch,
        (0, 0, 0),
        "--final-verifier",
        "faster-whisper-small",
    )

    assert rc == 0
    assert all(dep in calls[0] for dep in FINAL_VERIFIER_RUNTIME_DEPS)
    assert calls[1][-2:] == ["--final-verifier", "faster-whisper-small"]


def test_installer_forwards_parakeet_final_asr_selection(monkeypatch):
    rc, calls = _run_installer(
        monkeypatch,
        (0, 0, 0),
        "--final-asr",
        "parakeet-unified-en",
    )

    assert rc == 0
    assert "--sense-voice" not in calls[1]
    assert calls[1][-2:] == ["--final-asr", "parakeet-unified-en"]
    assert "sherpa-onnx==1.13.3" in calls[0]


def test_installer_propagates_deferred_doctor_failure(monkeypatch):
    rc, calls = _run_installer(monkeypatch, (0, 0, 9))

    assert rc == 9
    assert calls[-1] == [
        "/venv/bin/python", "-m", "tools.doctor", "--defer-ollama",
    ]


def test_installer_skip_models_is_explicitly_incomplete(monkeypatch):
    rc, calls = _run_installer(monkeypatch, (0,), "--skip-models")

    assert rc == 2
    assert len(calls) == 1
    assert calls[0][2:4] == ["pip", "install"]


def test_installer_success_stops_at_base_ready(monkeypatch, capsys):
    import tools.install as installer

    monkeypatch.setattr(installer.sys, "platform", "linux")
    rc, _calls = _run_installer(monkeypatch, (0, 0, 0))

    assert rc == 0
    output = capsys.readouterr().out
    assert "Base speech runtime installed" in output
    assert "python -m tools.doctor" in output
    assert "Only the final doctor READY result" in output
    assert "    ./live.sh" in output


def test_installer_runs_capability_setup_only_after_model_publication(monkeypatch):
    rc, calls = _run_installer(
        monkeypatch,
        (0, 0, 0, 0),
        "--obsidian-vault", "/vault",
        "--enable-reminders",
        "--trust-app", "notes=notes.desktop",
        "--trust-app", "browser=browser.desktop",
        "--untrust-app", "old-notes",
    )

    assert rc == 0
    assert calls[1][:3] == ["/venv/bin/python", "-m", "tools.setup_models"]
    assert calls[2] == [
        "/venv/bin/python", "-m", "tools.setup_assistant",
        "--obsidian-vault", "/vault",
        "--enable-reminders",
        "--trust-app", "notes=notes.desktop",
        "--trust-app", "browser=browser.desktop",
        "--untrust-app", "old-notes",
    ]
    assert calls[3] == [
        "/venv/bin/python", "-m", "tools.doctor", "--defer-ollama",
    ]


def test_installer_capability_setup_failure_stops_before_doctor(monkeypatch):
    rc, calls = _run_installer(
        monkeypatch,
        (0, 0, 6),
        "--disable-obsidian",
    )

    assert rc == 6
    assert calls[-1] == [
        "/venv/bin/python", "-m", "tools.setup_assistant", "--disable-obsidian",
    ]
    assert all("tools.doctor" not in command for command in calls)


def test_skip_models_never_runs_capability_setup(monkeypatch):
    rc, calls = _run_installer(
        monkeypatch,
        (0,),
        "--skip-models",
        "--enable-reminders",
    )

    assert rc == 2
    assert len(calls) == 1
    assert all("tools.setup_assistant" not in command for command in calls)
