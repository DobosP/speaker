#!/usr/bin/env python3
"""Preflight report for the native (``--session local``) voice runtime.

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

from core.config import (
    FINAL_STT_PROFILE_NAMES,
    SpeakerIdentityPolicyError,
    apply_final_stt_profile,
    apply_no_speaker_enrollment,
)
from core.readiness import (
    DEFAULT_OLLAMA_MODELS,
    SHERPA_REQUIRED,
    Check,
    PipeWireState,
    check_audio,
    check_audio_frontend,
    check_asr_final_runtime,
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
    ollama_show: Optional[Callable[[str], object]] = None,
    import_fn: Callable[[str], object] = importlib.import_module,
    exists: Callable[[str], bool] = os.path.exists,
    models_needed: Optional[Iterable[str]] = None,
    device=None,
    final_stt_profile: str | None = None,
    no_speaker_enrollment: bool = False,
    llm_mode: str = "configured",
    platform: str = sys.platform,
    pipewire_state: Optional[PipeWireState] = None,
    pipewire_probe: Callable[[], Optional[PipeWireState]] = probe_pipewire_state,
    capture_only: bool = False,
) -> list[Check]:
    """Resolve/apply one profile, then run every check over that merged view."""
    merged, _ = resolve_check_config(config, device)
    prefix = [check_python(), check_platform()]
    if final_stt_profile:
        merged, metadata = apply_final_stt_profile(merged, final_stt_profile)
        prefix.append(Check(
            "final STT profile",
            True,
            f"{metadata.name}; sha256={metadata.sha256}",
        ))
    if no_speaker_enrollment:
        merged = apply_no_speaker_enrollment(merged)
        detail = (
            "off for this capture-only process; configured references and files "
            "are unchanged; assistant, tool, control, TTS, KWS, speaker-"
            "verification, and playback effects are unavailable"
            if capture_only
            else "off for this process only; configured references and files are "
            "unchanged; speaker verification is unavailable; other audible "
            "speakers may converse and use read or direct-live-confirmed tools"
        )
        prefix.append(Check("speaker identity policy", True, detail))
    runtime_checks = run_runtime_checks(
        merged,
        resolved=True,
        llm_mode=llm_mode,
        sd=sd,
        ollama_lister=ollama_lister,
        ollama_show=ollama_show,
        import_fn=import_fn,
        exists=exists,
        models_needed=models_needed,
        platform=platform,
        pipewire_state=pipewire_state,
        pipewire_probe=pipewire_probe,
        include_speaker=not capture_only,
        include_tts=not capture_only,
        include_kws=not capture_only,
        audio_device_kinds=("input",) if capture_only else ("input", "output"),
    )
    if no_speaker_enrollment:
        # Avoid the normal "run --enroll" nudge when references are
        # intentionally masked.  Keep a failing speaker-ID check: it is the
        # fail-closed signal for an explicit identity-required word-cut policy.
        runtime_checks = [
            check
            for check in runtime_checks
            if check.name != "speaker-ID" or not check.ok
        ]
    return prefix + runtime_checks


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
        description="Preflight check for the native voice runtime",
        allow_abbrev=False,
    )
    parser.add_argument("--config", default="config.json")
    parser.add_argument(
        "--device",
        default=None,
        help="check a device profile (e.g. open_speaker); default = config.device",
    )
    parser.add_argument(
        "--final-stt-profile",
        choices=FINAL_STT_PROFILE_NAMES,
        default=None,
        help=(
            "apply the same complete session-scoped final-STT profile that core "
            "will use before checking models and runtime readiness"
        ),
    )
    parser.add_argument(
        "--no-speaker-enrollment",
        action="store_true",
        help=(
            "check the same process-local enrollment-free configuration used "
            "by core; persisted enrollment remains unchanged"
        ),
    )
    parser.add_argument(
        "--defer-ollama",
        action="store_true",
        help=(
            "check only the selected base speech runtime, without importing or "
            "contacting Ollama; success is reported as BASE READY, never READY"
        ),
    )
    parser.add_argument(
        "--defer-llm",
        action="store_true",
        help=(
            "check only the selected base speech runtime without constructing "
            "the configured local LLM; success is BASE READY, never READY"
        ),
    )
    parser.add_argument(
        "--capture-only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    if args.defer_ollama and args.defer_llm:
        parser.error("choose only one LLM deferral option")
    if args.capture_only:
        if not args.defer_llm:
            parser.error("--capture-only requires --defer-llm")
        if args.defer_ollama:
            parser.error("--capture-only does not accept --defer-ollama")
        if not args.final_stt_profile:
            parser.error("--capture-only requires --final-stt-profile")
        if not args.no_speaker_enrollment:
            parser.error("--capture-only requires --no-speaker-enrollment")
    try:
        from core.app import _load_config

        config = _load_config(args.config)
    except Exception:
        config = {}
    try:
        if args.defer_llm:
            checks = run_all(
                config,
                device=args.device,
                final_stt_profile=args.final_stt_profile,
                no_speaker_enrollment=args.no_speaker_enrollment,
                llm_mode="echo",
                capture_only=args.capture_only,
            )
        elif args.defer_ollama:
            merged, _ = resolve_check_config(config, args.device)
            backend = str(
                ((merged.get("llm", {}) or {}).get("backend", "ollama") or "ollama")
            ).lower()
            if backend != "ollama":
                checks = [Check(
                    "Ollama deferral",
                    False,
                    f"selected LLM backend is {backend!r}, not 'ollama'",
                    "omit --defer-ollama and satisfy the selected backend checks",
                )]
            else:
                checks = run_all(
                    config,
                    device=args.device,
                    final_stt_profile=args.final_stt_profile,
                    no_speaker_enrollment=args.no_speaker_enrollment,
                    llm_mode="echo",
                )
        else:
            checks = run_all(
                config,
                device=args.device,
                final_stt_profile=args.final_stt_profile,
                no_speaker_enrollment=args.no_speaker_enrollment,
            )
    except SpeakerIdentityPolicyError as exc:
        checks = [Check(
            "speaker identity policy",
            False,
            str(exc),
            "select an identity-free word-cut profile or omit "
            "--no-speaker-enrollment",
        )]
    except ValueError as exc:
        label = "final STT profile" if args.final_stt_profile else "device profile"
        checks = [Check(
            label,
            False,
            str(exc),
            (
                "use the complete configured --final-stt-profile map"
                if args.final_stt_profile
                else "use a listed --device profile"
            ),
        )]
    ready, text = summarize(checks)
    print(text)
    print()
    deferred = args.defer_ollama or args.defer_llm
    if ready:
        if args.capture_only:
            print(
                "CAPTURE READY -- assistant, tool, control, TTS, KWS, and "
                "playback effects are unavailable"
            )
            return 0
        if deferred:
            label = "Ollama deferred" if args.defer_ollama else "local LLM deferred"
            print(
                f"BASE READY ({label}) -- install/provision the selected "
                "local models, then run `python -m tools.doctor` for full READY"
            )
            return 0
        print("READY -> python -m core --session local")
        return 0
    if args.capture_only:
        print(
            "CAPTURE NOT READY -- fix the FAIL lines above, then re-run the "
            "capture-only doctor"
        )
    elif deferred:
        flag = "--defer-ollama" if args.defer_ollama else "--defer-llm"
        print(
            "BASE NOT READY -- fix the FAIL lines above, then re-run "
            f"`python -m tools.doctor {flag}`"
        )
    else:
        print("NOT READY -- fix the FAIL lines above, then re-run `python -m tools.doctor`")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
