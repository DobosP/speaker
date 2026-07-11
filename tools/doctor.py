#!/usr/bin/env python3
"""Preflight report for the native (``--engine sherpa``) voice runtime.

The production readiness contract lives in :mod:`core.readiness`; this module
adds doctor-only Python/platform presentation and the CLI. Readiness functions
are re-exported for backwards-compatible tests and tool callers.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from typing import Callable, Iterable, Optional

from core.readiness import (
    DEFAULT_OLLAMA_MODELS,
    SHERPA_REQUIRED,
    Check,
    PipeWireState,
    check_audio,
    check_audio_frontend,
    check_imports,
    check_llamacpp_abort_runtime,
    check_llamacpp_models,
    check_ollama,
    check_sherpa_models,
    check_speaker_id,
    probe_pipewire_state,
    profile_ollama_models,
    resolve_check_config,
    run_runtime_checks,
)

# Historical public name retained for callers that inspect the doctor surface.
REQUIRED_IMPORTS = ("numpy", "scipy", "sounddevice", "sherpa_onnx", "ollama")


def check_python(version=sys.version_info) -> Check:
    ok = (version[0], version[1]) >= (3, 10)
    return Check(
        "python",
        ok,
        f"{version[0]}.{version[1]}.{version[2]}",
        "" if ok else "Python 3.10+ required",
    )


def check_platform(
    platform: str = sys.platform, *, in_venv: Optional[bool] = None
) -> Check:
    """Report OS and venv state (informational, never a readiness failure)."""
    name = {"win32": "Windows", "darwin": "macOS"}.get(
        platform,
        "Linux" if platform.startswith("linux") else platform,
    )
    if in_venv is None:
        in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    detail = f"{name}; venv={'yes' if in_venv else 'no'}"
    hint = (
        ""
        if in_venv
        else "not in a venv -- run the installer (install.sh / install.ps1)"
    )
    return Check("platform", True, detail, hint)


def run_all(
    config: dict,
    *,
    sd=None,
    ollama_lister: Optional[Callable[[], Iterable[str]]] = None,
    import_fn: Callable[[str], object] = importlib.import_module,
    exists: Callable[[str], bool] = os.path.exists,
    models_needed: Optional[Iterable[str]] = None,
    device=None,
    llm_mode: str = "configured",
    platform: str = sys.platform,
    pipewire_state: Optional[PipeWireState] = None,
    pipewire_probe: Callable[[], Optional[PipeWireState]] = probe_pipewire_state,
) -> list[Check]:
    """Resolve/apply one profile, then run every check over that merged view."""
    merged, _ = resolve_check_config(config, device)
    return [check_python(), check_platform()] + run_runtime_checks(
        merged,
        resolved=True,
        llm_mode=llm_mode,
        sd=sd,
        ollama_lister=ollama_lister,
        import_fn=import_fn,
        exists=exists,
        models_needed=models_needed,
        platform=platform,
        pipewire_state=pipewire_state,
        pipewire_probe=pipewire_probe,
    )


def summarize(checks: Iterable[Check]) -> tuple[bool, str]:
    ready = True
    lines: list[str] = []
    for check in checks:
        if not check.ok:
            ready = False
        mark = "OK  " if check.ok else "FAIL"
        line = f"[{mark}] {check.name}: {check.detail}".rstrip()
        if check.hint and not check.ok:
            line += f"\n        fix: {check.hint}"
        lines.append(line)
    return ready, "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Preflight check for the native voice runtime"
    )
    parser.add_argument("--config", default="config.json")
    parser.add_argument(
        "--device",
        default=None,
        help="check a device profile (e.g. open_speaker); default = config.device",
    )
    args = parser.parse_args(argv)
    try:
        from core.app import _load_config

        config = _load_config(args.config)
    except Exception:
        config = {}
    try:
        checks = run_all(config, device=args.device)
    except ValueError as exc:
        checks = [Check(
            "device profile", False, str(exc), "use a listed --device profile"
        )]
    ready, text = summarize(checks)
    print(text)
    print()
    if ready:
        print("READY -> python -m core --engine sherpa")
        return 0
    print("NOT READY -- fix the FAIL lines above, then re-run `python -m tools.doctor`")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
