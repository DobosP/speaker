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

# scipy backs the coherence barge-in detector (the default barge path). Without
# it the engine still runs but silently falls back to the legacy level gate, so
# treat it as required -- a missing scipy is exactly the stale-venv gap that made
# barge-in degrade on a real machine (2026-06-02).
REQUIRED_IMPORTS = ("numpy", "scipy", "sounddevice", "sherpa_onnx", "ollama")
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


def check_platform(platform: str = sys.platform, *, in_venv: Optional[bool] = None) -> Check:
    """Report the OS + whether we're inside a venv (always OK; informational).

    Surfaces the two cross-platform gotchas at a glance: which OS we detected
    (Linux/Windows/macOS) and whether the interpreter is the project's venv --
    a 'no' here is the usual cause of the conda/venv 'No module named pip' mess
    the installer fixes, so we nudge toward it without failing readiness."""
    name = {"win32": "Windows", "darwin": "macOS"}.get(platform, "Linux"
            if platform.startswith("linux") else platform)
    if in_venv is None:
        in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    detail = f"{name}; venv={'yes' if in_venv else 'no'}"
    hint = "" if in_venv else "not in a venv -- run the installer (install.sh / install.ps1)"
    return Check("platform", True, detail, hint)


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
        return Check(
            "sherpa models",
            False,
            "; ".join(problems) + " (config.json/config.local.json)",
            "python -m tools.setup_models  # writes config.local.json",
        )
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
        except OSError as exc:  # noqa: BLE001 - imported but native PortAudio missing
            # Distinct from a missing pip package: the wheel is there but its
            # native PortAudio lib isn't (common on a bare Linux box).
            from tools.install import portaudio_hint

            return [Check("audio", False, f"PortAudio not loadable: {exc}", portaudio_hint())]
        except Exception as exc:  # noqa: BLE001 - package not installed
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


def _pipewire_echo_cancel_loaded(modules_text: Optional[str] = None) -> Optional[bool]:
    """True/False if a PipeWire/Pulse ``module-echo-cancel`` (the OS voice-comm /
    Teams-equivalent capture path) is loaded; ``None`` when ``pactl`` is
    unavailable (so the check is skipped rather than failed)."""
    if modules_text is None:
        import shutil
        import subprocess

        if shutil.which("pactl") is None:
            return None
        try:
            res = subprocess.run(
                ["pactl", "list", "short", "modules"],
                capture_output=True, text=True, timeout=3,
            )
            modules_text = res.stdout
        except Exception:  # noqa: BLE001 - pactl absent / errored -> skip
            return None
    return "echo-cancel" in (modules_text or "")


def check_audio_frontend(
    config: dict,
    *,
    import_fn: Callable[[str], object] = importlib.import_module,
    platform: str = sys.platform,
    modules_text: Optional[str] = None,
) -> list[Check]:
    """Capture/playback front-end deps: the anti-alias resampler, the WebRTC APM
    (only when a profile selects it), and the OS echo-cancel source on Linux."""
    out: list[Check] = []
    # soxr: the stateful anti-alias resampler. Without it the capture path falls
    # back to a stateless per-block polyphase filter (a seam every 0.1 s) -> a real
    # WER hit on sibilants. Advisory: capture still works, just lower quality.
    try:
        import_fn("soxr")
        out.append(Check("soxr anti-alias resampler", True))
    except Exception:  # noqa: BLE001
        out.append(Check(
            "soxr anti-alias resampler", False,
            "not installed -> per-block resampling aliases into the speech band",
            "python -m pip install soxr",
        ))
    # livekit (the WebRTC APM) must be importable ONLY when the config actually
    # SELECTS the apm backend -- i.e. the ACTIVE resolved profile, not merely any
    # profile that DEFINES it. The committed `open_speaker` profile selects apm,
    # but it is opt-in (--device open_speaker); livekit is a remote-only optional
    # dependency, so demanding it on every clean clone (which resolves to a non-apm
    # profile) would wrongly flip READY=False on a perfectly healthy default box.
    # Mirror the capture_voice_comm rule below: an unselected apm profile is
    # ADVISORY (never blocks READY).
    sherpa = config.get("sherpa", {}) if isinstance(config, dict) else {}
    active_sherpa = sherpa
    active_backend = str(sherpa.get("aec_backend", "")).lower()
    try:  # resolve the device the way core.app does; best-effort
        from core.config import apply_device_profile, resolve_device
        dev, _rationale = resolve_device(config, config.get("device", "auto"))
        active_sherpa = apply_device_profile(config, dev).get("sherpa", {}) or sherpa
        active_backend = str(
            active_sherpa.get("aec_backend", active_backend)
        ).lower()
    except Exception:  # noqa: BLE001 - fall back to the base backend
        pass
    defined_backends = {active_backend}
    for prof in (config.get("device_profiles", {}) or {}).values():
        defined_backends.add(str((prof.get("sherpa", {}) or {}).get("aec_backend", "")).lower())

    def _have_apm() -> Optional[bool]:
        try:
            rtc = import_fn("livekit.rtc")
            return bool(hasattr(rtc, "AudioProcessingModule"))
        except Exception:  # noqa: BLE001
            return None  # livekit not importable at all

    if active_backend in ("apm", "webrtc"):
        have = _have_apm()
        if have:
            out.append(Check("livekit WebRTC APM (aec_backend=apm)", True, "", ""))
        elif have is False:
            out.append(Check(
                "livekit WebRTC APM (aec_backend=apm)", False,
                "livekit present but exposes no AudioProcessingModule",
                "python -m pip install -U livekit",
            ))
        else:
            out.append(Check(
                "livekit WebRTC APM (aec_backend=apm)", False,
                "aec_backend='apm' selected but livekit isn't installed (AEC fails open to none)",
                "python -m pip install livekit",
            ))
    elif defined_backends & {"apm", "webrtc"}:
        # An UNSELECTED profile (e.g. open_speaker) uses apm. Advisory only --
        # ok=True so it never flips READY; just tells the user what they'd need.
        have = _have_apm()
        out.append(Check(
            "livekit WebRTC APM (optional; an unselected profile uses aec_backend=apm)",
            True,
            "livekit available" if have
            else "not installed; only needed if you select an apm profile (e.g. --device open_speaker)",
            "" if have else "python -m pip install livekit",
        ))
    # Linux + the user opted into the OS voice-comm path (capture_voice_comm): is
    # the PipeWire/Pulse echo-cancel source actually loaded? Only checked when
    # requested -- it is an ALTERNATIVE to the in-app APM, not required, so it must
    # never block READY for someone using the APM (or no AEC at all).
    wants_voice_comm = bool(sherpa.get("capture_voice_comm", False)) or any(
        (prof.get("sherpa", {}) or {}).get("capture_voice_comm", False)
        for prof in (config.get("device_profiles", {}) or {}).values()
    )
    if platform.startswith("linux") and wants_voice_comm:
        loaded = _pipewire_echo_cancel_loaded(modules_text)
        if loaded is not None:
            out.append(Check(
                "OS echo-cancel source (PipeWire)", loaded,
                "module-echo-cancel loaded" if loaded else "not loaded (raw mic; no OS AEC/NS/AGC)",
                "" if loaded
                else "pactl load-module module-echo-cancel aec_method=webrtc  (see docs/audio_pipeline.md)",
            ))
    # ADR-0013 word-cut barge preflight: the word-cut path (barge_word_cut_enabled
    # with the in-app AEC OFF) is only viable when the OS canceller cleans the
    # near-end during playback -- without it the recognizer sees raw echo with the
    # user drowned underneath, the word gate has nothing to cut on, and the run
    # degrades SILENTLY (exactly how the 2026-07-06 live batch shipped:
    # run-20260706-231226 launched with no module-echo-cancel check anywhere).
    # Only checked when the ACTIVE resolved config selects that path on Linux.
    wants_word_cut = bool(active_sherpa.get("barge_word_cut_enabled", False)) and not bool(
        active_sherpa.get("aec_enabled", False)
    )
    if platform.startswith("linux") and wants_word_cut:
        loaded = _pipewire_echo_cancel_loaded(modules_text)
        if loaded is not None:
            out.append(Check(
                "OS echo-cancel for word-cut barge (ADR-0013)", loaded,
                "module-echo-cancel loaded" if loaded
                else "barge_word_cut_enabled=true + aec_enabled=false but NO "
                "module-echo-cancel is loaded: the near-end user won't survive "
                "playback, so the word-cut has nothing to cut on",
                "" if loaded else (
                    "pactl load-module module-echo-cancel aec_method=webrtc "
                    "source_name=echo-cancel-source sink_name=echo-cancel-sink "
                    'aec_args="webrtc.noise_suppression=false webrtc.gain_control=false"'
                    " then set both echo-cancel nodes as defaults (ADR-0013)"
                ),
            ))
    return out


def profile_ollama_models(config: dict, device=None) -> tuple:
    """The ollama models the ACTIVE (resolved) device profile actually needs --
    main_model + fast_model + router_model from the device-profile-merged llm block.
    This lets doctor catch a profile that selects a model the box never pulled (e.g.
    `--device open_speaker` -> gemma3:1b), instead of only the hardcoded defaults
    (the gap that let the 2026-06-21 `gemma3:1b not found` 404 ship). Empty for a
    non-ollama backend (llamacpp uses GGUF files, not ollama-pulled models)."""
    try:
        from core.config import apply_device_profile, resolve_device
        dev, _ = resolve_device(config, device or config.get("device", "auto"))
        merged = apply_device_profile(config, dev)
    except Exception:  # noqa: BLE001 - best-effort; fall back to the base config
        merged = config if isinstance(config, dict) else {}
    llm = merged.get("llm", {}) if isinstance(merged, dict) else {}
    backend = str(llm.get("backend", "") or "").lower()
    if backend and backend != "ollama":
        return ()
    seen = []
    for key in ("main_model", "fast_model", "router_model"):
        v = llm.get(key)
        if v and str(v) not in seen:
            seen.append(str(v))
    return tuple(seen)


def run_all(
    config: dict,
    *,
    sd=None,
    ollama_lister: Optional[Callable[[], Iterable[str]]] = None,
    import_fn: Callable[[str], object] = importlib.import_module,
    exists: Callable[[str], bool] = os.path.exists,
    models_needed: Optional[Iterable[str]] = None,
    device=None,
) -> list[Check]:
    # When models_needed isn't pinned by the caller, check the ACTIVE profile's
    # models (so a missing profile model is caught), falling back to the defaults.
    if models_needed is None:
        models_needed = profile_ollama_models(config, device) or DEFAULT_OLLAMA_MODELS
    checks = [check_python(), check_platform()]
    checks += check_imports(import_fn=import_fn)
    checks.append(check_sherpa_models(config, exists=exists))
    checks.append(check_speaker_id(config, exists=exists))
    checks += check_ollama(models_needed=models_needed, lister=ollama_lister)
    checks += check_audio(sd=sd)
    checks += check_audio_frontend(config)
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
    parser.add_argument("--device", default=None,
                        help="check the models/config a specific device profile needs "
                             "(e.g. open_speaker); default = config.device")
    args = parser.parse_args(argv)
    try:
        # Use the app loader so config.local.json (the machine-local model paths
        # written by tools.setup_models) is merged in.
        from core.app import _load_config

        config = _load_config(args.config)
    except Exception:
        config = {}
    ready, text = summarize(run_all(config, device=args.device))
    print(text)
    print()
    if ready:
        print("READY -> python -m core --engine sherpa")
        return 0
    print("NOT READY -- fix the FAIL lines above, then re-run `python -m tools.doctor`")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
