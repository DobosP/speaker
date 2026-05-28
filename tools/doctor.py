#!/usr/bin/env python3
"""Preflight 'doctor' for the native (`--engine sherpa`) voice runtime.

Runs a set of independent checks -- Python version, required Python packages,
sherpa model paths, Ollama reachability + models, and audio devices -- and
prints a READY / NOT READY report with the exact command to fix each failure.

    python -m tools.doctor

Every check is a small pure function with injectable dependencies (the import
function, the Ollama lister, the sounddevice module, a path-exists probe), so
the whole thing is unit-testable without audio hardware, Ollama, or models.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

REQUIRED_IMPORTS = ("numpy", "sounddevice", "sherpa_onnx", "ollama")
# The ASR + TTS artifacts the engine must have to hear and speak (VAD is
# optional -- it only gates barge-in).
SHERPA_REQUIRED = (
    "asr_tokens",
    "asr_encoder",
    "asr_decoder",
    "asr_joiner",
    "tts_model",
    "tts_tokens",
)
DEFAULT_OLLAMA_MODELS = ("gemma3:12b", "gemma3:4b")
_PIP_NAME = {"sherpa_onnx": "sherpa-onnx"}


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    hint: str = ""


def check_python(version=sys.version_info) -> Check:
    ok = (version[0], version[1]) >= (3, 10)
    return Check(
        "python",
        ok,
        f"{version[0]}.{version[1]}.{version[2]}",
        "" if ok else "Python 3.10+ required",
    )


def check_imports(
    modules: Iterable[str] = REQUIRED_IMPORTS,
    import_fn: Callable[[str], object] = importlib.import_module,
) -> list[Check]:
    out: list[Check] = []
    for mod in modules:
        try:
            import_fn(mod)
            out.append(Check(f"import {mod}", True))
        except Exception as exc:  # noqa: BLE001 - any import failure is a fail
            out.append(
                Check(
                    f"import {mod}",
                    False,
                    str(exc),
                    f"python -m pip install {_PIP_NAME.get(mod, mod)}",
                )
            )
    return out


def check_sherpa_models(
    config: dict, exists: Callable[[str], bool] = os.path.exists
) -> Check:
    sherpa = (config or {}).get("sherpa", {}) or {}
    problems: list[str] = []
    for key in SHERPA_REQUIRED:
        path = sherpa.get(key, "")
        if not path:
            problems.append(f"{key} unset")
        elif not exists(path):
            problems.append(f"{key} missing on disk")
    if problems:
        return Check("sherpa models", False, "; ".join(problems), "python -m tools.setup_models")
    return Check("sherpa models", True, "ASR + TTS paths set")


def check_speaker_id(config: dict, exists: Callable[[str], bool] = os.path.exists) -> Check:
    """Advisory check for the speaker-ID gate (barge-in + input gating).

    Speaker-ID is optional: the assistant runs without it (fail-open), so a
    missing model or enrollment is never a hard FAIL that blocks readiness --
    it's surfaced as an OK line with a nudge. The one genuine failure is a
    configured model path that isn't on disk (a broken setup)."""
    sherpa = (config or {}).get("sherpa", {}) or {}
    model = sherpa.get("speaker_embedding_model", "")
    if not model:
        return Check(
            "speaker-ID", True,
            "not configured (optional; barge-in/input gating off) -- "
            "enable with `python -m tools.setup_models`",
        )
    if not exists(model):
        return Check(
            "speaker-ID", False, "model path set but missing on disk",
            "python -m tools.setup_models",
        )
    enroll_emb = sherpa.get("speaker_enroll_embedding", "")
    enroll_wav = sherpa.get("speaker_enroll_wav", "")
    if (enroll_emb and exists(enroll_emb)) or (enroll_wav and exists(enroll_wav)):
        return Check("speaker-ID", True, "model + enrollment present")
    return Check(
        "speaker-ID", True,
        "model present but not enrolled -- gate is fail-open; "
        "run `python -m core --enroll` to enroll your voice",
    )


def _default_ollama_lister() -> list[str]:
    import ollama

    data = ollama.list()
    models = data.get("models", []) if isinstance(data, dict) else getattr(data, "models", [])
    names: list[str] = []
    for m in models:
        if isinstance(m, dict):
            name = m.get("model") or m.get("name")
        else:
            name = getattr(m, "model", None) or getattr(m, "name", None)
        if name:
            names.append(name)
    return names


def check_ollama(
    models_needed: Iterable[str] = DEFAULT_OLLAMA_MODELS,
    lister: Optional[Callable[[], Iterable[str]]] = None,
) -> list[Check]:
    lister = lister or _default_ollama_lister
    try:
        available = set(lister())
    except Exception as exc:  # noqa: BLE001 - daemon down / not installed
        return [
            Check(
                "ollama",
                False,
                f"not reachable: {exc}",
                "start it with `ollama serve` (install: https://ollama.com)",
            )
        ]
    out = [Check("ollama", True, f"{len(available)} model(s) installed")]
    for need in models_needed:
        ok = need in available
        out.append(
            Check(f"ollama model {need}", ok, "" if ok else "not pulled",
                  "" if ok else f"ollama pull {need}")
        )
    return out


def check_audio(sd=None) -> list[Check]:
    if sd is None:
        try:
            import sounddevice as sd  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            return [Check("audio", False, str(exc), "python -m pip install sounddevice")]
    out: list[Check] = []
    for kind in ("input", "output"):
        try:
            dev = sd.query_devices(kind=kind)
            name = dev.get("name", "?") if isinstance(dev, dict) else getattr(dev, "name", "?")
            rate = (
                dev.get("default_samplerate", 0)
                if isinstance(dev, dict)
                else getattr(dev, "default_samplerate", 0)
            )
            out.append(Check(f"audio {kind}", True, f"{name} @ {int(rate)} Hz"))
        except Exception as exc:  # noqa: BLE001 - no default device
            out.append(
                Check(f"audio {kind}", False, str(exc), "run `python -m sounddevice` to list devices")
            )
    return out


def run_all(
    config: dict,
    *,
    sd=None,
    ollama_lister: Optional[Callable[[], Iterable[str]]] = None,
    import_fn: Callable[[str], object] = importlib.import_module,
    exists: Callable[[str], bool] = os.path.exists,
    models_needed: Iterable[str] = DEFAULT_OLLAMA_MODELS,
) -> list[Check]:
    checks = [check_python()]
    checks += check_imports(import_fn=import_fn)
    checks.append(check_sherpa_models(config, exists=exists))
    checks.append(check_speaker_id(config, exists=exists))
    checks += check_ollama(models_needed=models_needed, lister=ollama_lister)
    checks += check_audio(sd=sd)
    return checks


def summarize(checks: Iterable[Check]) -> tuple[bool, str]:
    ready = True
    lines: list[str] = []
    for c in checks:
        if not c.ok:
            ready = False
        mark = "OK  " if c.ok else "FAIL"
        line = f"[{mark}] {c.name}: {c.detail}".rstrip()
        if c.hint and not c.ok:
            line += f"\n        fix: {c.hint}"
        lines.append(line)
    return ready, "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preflight check for the native voice runtime")
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args(argv)
    try:
        # Use the app loader so config.local.json (the machine-local model paths
        # written by tools.setup_models) is merged in.
        from core.app import _load_config

        config = _load_config(args.config)
    except Exception:
        config = {}
    ready, text = summarize(run_all(config))
    print(text)
    print()
    if ready:
        print("READY -> python -m core --engine sherpa")
        return 0
    print("NOT READY -- fix the FAIL lines above, then re-run `python -m tools.doctor`")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
