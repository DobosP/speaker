"""Tests for the cross-platform installer helpers (tools.install).

Pure / injected -- no venv is created, no subprocess runs. The OS is passed
explicitly so both the Windows and POSIX paths are exercised on Linux CI.
"""
from __future__ import annotations

import argparse

from tools.install import (
    RUNTIME_DEPS,
    activation_hint,
    install_plan,
    is_windows,
    needs_fresh_venv,
    portaudio_hint,
    venv_python_path,
)


def test_is_windows_by_platform_string():
    assert is_windows("win32")
    assert not is_windows("linux")
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
    base = dict(venv=".venv", recreate=False, skip_models=False)
    base.update(over)
    return argparse.Namespace(**base)


def test_install_plan_includes_deps_and_models():
    plan = install_plan(_args(), system="linux")
    text = "\n".join(plan)
    assert any(dep in text for dep in RUNTIME_DEPS)
    assert "setup_models" in text
    assert "doctor" in text


def test_install_plan_respects_skip_models():
    plan = install_plan(_args(skip_models=True), system="linux")
    text = "\n".join(plan)
    assert "skip model download" in text
    assert "tools.setup_models" not in text


def test_install_plan_uses_windows_interpreter_path():
    plan = install_plan(_args(), system="win32")
    assert any("python.exe" in step for step in plan)
