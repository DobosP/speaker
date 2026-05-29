"""Real-process end-to-end test for the CLI entrypoint.

Every other test exercises the runtime *in-process* (scripted engine, fake LLM,
direct ``VoiceRuntime`` construction) and so never runs the actual command the
docs tell users to run. This test subprocesses the real entrypoint --
``python -m core --engine console --llm echo`` -- and drives it over stdin, to
pin the whole boot path as a process:

    app.main -> _load_config -> _apply_device_profile -> _build_llms (EchoLLM)
    -> _build_engine (ScriptedEngine) -> _build_memory -> attach_web_search
    -> VoiceRuntime -> _run_console loop

It is marked ``slow`` (it spawns and tears down a Python interpreter) and
``e2e`` (real-process boundary). It is built to be deterministic and to *never*
hang CI: stdin is closed after a fixed script so the console loop hits EOF and
exits on its own, and the subprocess is additionally guarded by a bounded
timeout that kills the child and fails with its captured output.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.e2e]

# Repo root: this file is <root>/tests/test_cli_e2e.py.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Hard upper bound on the child process. The echo LLM answers instantly and EOF
# closes the loop, so a healthy run finishes in well under a second; this only
# trips if the process genuinely wedges. Kept under pytest.ini's per-test
# --timeout so *this* test's kill+capture path reports the failure (with output)
# rather than the bare pytest-timeout teardown.
SUBPROCESS_TIMEOUT_S = 45.0

# A short, deterministic script: greet (assistant reply), switch to research
# mode, ask a research question, then stop. Newline-terminated; the trailing
# newline + closed stdin gives the console loop its EOF.
STDIN_SCRIPT = "assistant please help me\nresearch mode\nthe history of rome\nstop\n"


def test_console_cli_end_to_end():
    """Subprocess the real entrypoint and assert the console path works."""
    env = dict(os.environ)
    # PYTHONPATH so `python -m core` resolves regardless of the child's cwd
    # handling; SPEAKER_NO_LOCAL_CONFIG so a dev box's real model paths in
    # config.local.json can't redirect the run; PYTHONIOENCODING so the child's
    # stdout never dies with UnicodeEncodeError on a cp1252 Windows console.
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["SPEAKER_NO_LOCAL_CONFIG"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    proc = subprocess.Popen(
        [sys.executable, "-m", "core", "--engine", "console", "--llm", "echo"],
        cwd=str(REPO_ROOT),
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    try:
        stdout, stderr = proc.communicate(input=STDIN_SCRIPT, timeout=SUBPROCESS_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        # Drain whatever the child managed to emit so the failure is debuggable.
        stdout, stderr = proc.communicate()
        pytest.fail(
            f"CLI did not exit within {SUBPROCESS_TIMEOUT_S}s (killed).\n"
            f"--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
        )

    # Diagnostic context attached to every assertion failure below.
    detail = f"\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"

    assert proc.returncode == 0, f"non-zero exit {proc.returncode}{detail}"
    # Console banner from _run_console.
    assert "[console] mode=assistant" in stdout, f"missing console banner{detail}"
    # EchoLLM reply to the first utterance (capabilities -> TTS -> console echo).
    assert "You said: assistant please help me" in stdout, f"missing assistant reply{detail}"
    # The 'research mode' utterance flipped the control plane into research mode.
    assert "[mode=research]" in stdout, f"never entered research mode{detail}"
