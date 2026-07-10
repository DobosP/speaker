"""Shared selected-profile readiness for the native on-device voice runtime.

This module is production-owned: ``core.app`` consumes it directly, while
``tools.doctor`` and ``tools.live_session`` present the same checks through
their CLIs. Every external dependency is injectable so route/device decisions
remain deterministic in headless tests.
"""
from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Optional

RUNTIME_IMPORTS = ("numpy", "scipy", "sounddevice", "sherpa_onnx")
SHERPA_REQUIRED = (
    "asr_tokens",
    "asr_encoder",
    "asr_decoder",
    "asr_joiner",
    "tts_model",
    "tts_tokens",
)
DEFAULT_OLLAMA_MODELS = ("gemma3:12b", "gemma3:4b")
_PIP_NAME = {
    "sherpa_onnx": "sherpa-onnx",
    "llama_cpp": "llama-cpp-python",
}


@dataclass(frozen=True)
class PipeWireState:
    """Minimum injectable ``pactl`` state needed to prove EC routing."""

    modules: str = ""
    sources: str = ""
    sinks: str = ""
    default_source: str = ""
    default_sink: str = ""


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    hint: str = ""


def check_imports(
    modules: Iterable[str] = RUNTIME_IMPORTS,
    import_fn: Callable[[str], object] = importlib.import_module,
) -> list[Check]:
    out: list[Check] = []
    for mod in modules:
        try:
            import_fn(mod)
            out.append(Check(f"import {mod}", True))
        except Exception as exc:  # noqa: BLE001 - any import failure is a fail
            out.append(Check(
                f"import {mod}",
                False,
                str(exc),
                f"python -m pip install {_PIP_NAME.get(mod, mod)}",
            ))
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


def check_speaker_id(
    config: dict, exists: Callable[[str], bool] = os.path.exists
) -> Check:
    """Advisory speaker-ID check; a configured missing model is a real failure."""
    sherpa = (config or {}).get("sherpa", {}) or {}
    model = sherpa.get("speaker_embedding_model", "")
    if not model:
        return Check(
            "speaker-ID",
            True,
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
        "speaker-ID",
        True,
        "model present but not enrolled -- gate is fail-open; "
        "run `python -m core --enroll` to enroll your voice",
    )


def _default_ollama_lister() -> list[str]:
    import ollama

    data = ollama.list()
    models = data.get("models", []) if isinstance(data, dict) else getattr(data, "models", [])
    names: list[str] = []
    for model in models:
        if isinstance(model, dict):
            name = model.get("model") or model.get("name")
        else:
            name = getattr(model, "model", None) or getattr(model, "name", None)
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
        return [Check(
            "ollama",
            False,
            f"not reachable: {exc}",
            "start it with `ollama serve` (install: https://ollama.com)",
        )]
    out = [Check("ollama", True, f"{len(available)} model(s) installed")]
    for needed in models_needed:
        ok = needed in available
        out.append(Check(
            f"ollama model {needed}",
            ok,
            "" if ok else "not pulled",
            "" if ok else f"ollama pull {needed}",
        ))
    return out


def _sounddevice_selector(value):
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return value


def check_audio(config: Optional[dict] = None, sd=None) -> list[Check]:
    """Verify selected input/output devices, not merely sounddevice import."""
    if sd is None:
        try:
            import sounddevice as sd  # noqa: PLC0415
        except OSError as exc:
            return [Check(
                "audio",
                False,
                f"PortAudio not loadable: {exc}",
                "install PortAudio (apt install libportaudio2 / brew install portaudio)",
            )]
        except Exception as exc:  # noqa: BLE001
            return [Check("audio", False, str(exc), "python -m pip install sounddevice")]
    sherpa = (config or {}).get("sherpa", {}) or {}
    out: list[Check] = []
    for kind in ("input", "output"):
        try:
            selected = _sounddevice_selector(sherpa.get(f"{kind}_device"))
            if selected is None or selected == "":
                device = sd.query_devices(kind=kind)
            else:
                device = sd.query_devices(selected, kind=kind)
            name = (
                device.get("name", "?")
                if isinstance(device, dict)
                else getattr(device, "name", "?")
            )
            rate = (
                device.get("default_samplerate", 0)
                if isinstance(device, dict)
                else getattr(device, "default_samplerate", 0)
            )
            route = f"selected={selected!r}; " if selected not in (None, "") else ""
            out.append(Check(f"audio {kind}", True, f"{route}{name} @ {int(rate)} Hz"))
        except Exception as exc:  # noqa: BLE001 - unusable route is not ready
            out.append(Check(
                f"audio {kind}",
                False,
                str(exc),
                "run `python -m sounddevice` to list devices",
            ))
    return out


def probe_pipewire_state() -> Optional[PipeWireState]:
    """Read the minimum PipeWire/Pulse state needed to prove EC routing."""
    import shutil
    import subprocess

    if shutil.which("pactl") is None:
        return None

    def _run(*args: str) -> Optional[str]:
        try:
            result = subprocess.run(
                ["pactl", *args], capture_output=True, text=True, timeout=3,
            )
        except Exception:  # noqa: BLE001 - diagnostic state is unverifiable
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    modules = _run("list", "short", "modules")
    sources = _run("list", "short", "sources")
    sinks = _run("list", "short", "sinks")
    if modules is None or sources is None or sinks is None:
        return None
    default_source = _run("get-default-source")
    default_sink = _run("get-default-sink")
    if default_source is None or default_sink is None:
        info = _run("info") or ""
        for line in info.splitlines():
            key, _, value = line.partition(":")
            if key.strip().lower() == "default source" and default_source is None:
                default_source = value.strip()
            elif key.strip().lower() == "default sink" and default_sink is None:
                default_sink = value.strip()
    return PipeWireState(
        modules=modules,
        sources=sources,
        sinks=sinks,
        default_source=default_source or "",
        default_sink=default_sink or "",
    )


def _short_node_names(text: str) -> set[str]:
    names: set[str] = set()
    for line in (text or "").splitlines():
        fields = line.split()
        if len(fields) >= 2:
            names.add(fields[1].strip())
    return names


def _module_named_nodes(modules: str, key: str) -> set[str]:
    import re

    names: set[str] = set()
    pattern = re.compile(
        rf"(?:^|\s){re.escape(key)}=(?:\"([^\"]+)\"|'([^']+)'|([^\s]+))"
    )
    for line in (modules or "").splitlines():
        if "module-echo-cancel" not in line.lower():
            continue
        for match in pattern.finditer(line):
            names.add(next(value for value in match.groups() if value))
    return names


def _looks_like_echo_cancel(name: object) -> bool:
    value = str(name or "").strip().lower().replace("_", "-").replace(".", "-")
    return (
        ("echo" in value and "cancel" in value)
        or value.startswith(("ec-source", "ec-sink", "aec-source", "aec-sink"))
    )


def _echo_cancel_nodes(state: PipeWireState, kind: str) -> set[str]:
    text = state.sources if kind == "source" else state.sinks
    actual = _short_node_names(text)
    configured = _module_named_nodes(
        state.modules, "source_name" if kind == "source" else "sink_name"
    )
    discovered = {
        name for name in actual if _looks_like_echo_cancel(name)
    }
    # A module argument is only evidence of intent. Require the named node to
    # exist in pactl's active source/sink inventory before accepting the route.
    return (configured & actual) | discovered


def _route_uses_echo_cancel(selector, default_node: str, nodes: set[str]) -> bool:
    value = str(selector or "").strip()
    defaultish = not value or value.lower() in {"default", "pipewire", "pulse"}
    if defaultish:
        return bool(default_node) and default_node in nodes
    if value.isdigit():
        return False
    if value in nodes:
        return True
    low = value.lower()
    if any(node.lower() in low or low in node.lower() for node in nodes):
        return True
    return bool(nodes) and _looks_like_echo_cancel(value)


def _active_route_name(selector, default_node: str) -> str:
    value = str(selector or "").strip()
    if not value or value.lower() in {"default", "pipewire", "pulse"}:
        return default_node
    return value


def _check_pipewire_echo_route(
    sherpa: Mapping[str, object], state: Optional[PipeWireState]
) -> tuple[bool, str]:
    if state is None:
        return False, "could not inspect PipeWire/Pulse routing with pactl"
    if "module-echo-cancel" not in state.modules.lower():
        return False, "module-echo-cancel is not loaded"
    sources = _echo_cancel_nodes(state, "source")
    sinks = _echo_cancel_nodes(state, "sink")
    if not sources or not sinks:
        return False, "echo-cancel module is loaded but its source/sink nodes were not found"
    capture_ok = _route_uses_echo_cancel(
        sherpa.get("input_device"), state.default_source, sources
    )
    playback_ok = _route_uses_echo_cancel(
        sherpa.get("output_device"), state.default_sink, sinks
    )
    if not capture_ok or not playback_ok:
        missing: list[str] = []
        if not capture_ok:
            missing.append("capture is not routed through the echo-cancel source")
        if not playback_ok:
            missing.append("playback is not routed through the echo-cancel sink")
        return False, "; ".join(missing)
    return True, (
        f"capture={_active_route_name(sherpa.get('input_device'), state.default_source)}; "
        f"playback={_active_route_name(sherpa.get('output_device'), state.default_sink)}"
    )


def check_audio_frontend(
    config: dict,
    *,
    import_fn: Callable[[str], object] = importlib.import_module,
    exists: Callable[[str], bool] = os.path.exists,
    platform: str = sys.platform,
    pipewire_state: Optional[PipeWireState] = None,
    pipewire_probe: Callable[[], Optional[PipeWireState]] = probe_pipewire_state,
    require_os_echo_route: bool = True,
) -> list[Check]:
    """Check active resampler, APM, VAD, and OS echo-route prerequisites."""
    out: list[Check] = []
    try:
        import_fn("soxr")
        out.append(Check("soxr anti-alias resampler", True))
    except Exception:
        out.append(Check(
            "soxr anti-alias resampler",
            False,
            "not installed -> per-block resampling aliases into the speech band",
            "python -m pip install soxr",
        ))

    sherpa = config.get("sherpa", {}) if isinstance(config, dict) else {}
    backend = str(sherpa.get("aec_backend", "")).lower()
    aec_enabled = bool(sherpa.get("aec_enabled", False))
    if aec_enabled and backend in ("apm", "webrtc"):
        try:
            rtc = import_fn("livekit.rtc")
            have_apm = bool(hasattr(rtc, "AudioProcessingModule"))
        except Exception:  # noqa: BLE001
            have_apm = None
        if have_apm:
            out.append(Check("livekit WebRTC APM (aec_backend=apm)", True))
        elif have_apm is False:
            out.append(Check(
                "livekit WebRTC APM (aec_backend=apm)",
                False,
                "livekit present but exposes no AudioProcessingModule",
                "python -m pip install -U livekit",
            ))
        else:
            out.append(Check(
                "livekit WebRTC APM (aec_backend=apm)",
                False,
                "AEC+APM selected but livekit is not installed",
                "python -m pip install livekit",
            ))

    wants_word_cut = (
        bool(sherpa.get("barge_in_enabled", True))
        and bool(sherpa.get("barge_word_cut_enabled", False))
        and not aec_enabled
    )
    if wants_word_cut:
        vad_model = str(sherpa.get("vad_model", "") or "")
        vad_ok = bool(vad_model) and exists(vad_model)
        out.append(Check(
            "word-cut VAD model (ADR-0013)",
            vad_ok,
            vad_model if vad_ok else (
                "active word-cut requires a configured, existing vad_model"
            ),
            "" if vad_ok else "python -m tools.setup_models",
        ))

    wants_linux_ec = platform.startswith("linux") and (
        wants_word_cut or bool(sherpa.get("capture_voice_comm", False))
    )
    if wants_linux_ec and require_os_echo_route:
        state = pipewire_state if pipewire_state is not None else pipewire_probe()
        route_ok, detail = _check_pipewire_echo_route(sherpa, state)
        name = (
            "OS echo-cancel route for word-cut barge (ADR-0013)"
            if wants_word_cut
            else "OS echo-cancel route (PipeWire)"
        )
        out.append(Check(
            name,
            route_ok,
            detail,
            "" if route_ok else (
                "pactl load-module module-echo-cancel aec_method=webrtc "
                "source_name=echo-cancel-source sink_name=echo-cancel-sink "
                'aec_args="webrtc.noise_suppression=false webrtc.gain_control=false"'
                " then set both echo-cancel nodes as defaults (ADR-0013)"
            ),
        ))
    return out


def resolve_check_config(config: dict, device=None) -> tuple[dict, str]:
    """Resolve/apply one requested profile for every downstream readiness check."""
    from .config import apply_device_profile, resolve_device

    requested = device if device is not None else config.get("device", "auto")
    resolved, _ = resolve_device(config, requested)
    merged = apply_device_profile(
        config, resolved, strict=bool(config.get("device_profiles"))
    )
    return merged, resolved


def _ollama_models_from_resolved(config: dict) -> tuple[str, ...]:
    llm = config.get("llm", {}) if isinstance(config, dict) else {}
    backend = str(llm.get("backend", "") or "").lower()
    if backend and backend != "ollama":
        return ()
    seen: list[str] = []
    for key in ("main_model", "fast_model", "router_model"):
        value = llm.get(key)
        if value and str(value) not in seen:
            seen.append(str(value))
    return tuple(seen) or DEFAULT_OLLAMA_MODELS


def profile_ollama_models(config: dict, device=None) -> tuple[str, ...]:
    try:
        merged, _ = resolve_check_config(config, device)
    except Exception:  # noqa: BLE001 - public diagnostic remains best-effort
        merged = config if isinstance(config, dict) else {}
    return _ollama_models_from_resolved(merged)


def check_llamacpp_models(
    config: dict, exists: Callable[[str], bool] = os.path.exists
) -> list[Check]:
    llm = (config or {}).get("llm", {}) or {}
    paths = [llm.get("main_model_path")]
    if llm.get("fast_model_path"):
        paths.append(llm.get("fast_model_path"))
    missing = [
        str(path or "<unset>") for path in paths if not path or not exists(str(path))
    ]
    return [Check(
        "llama.cpp models",
        not missing,
        "configured model files exist" if not missing else f"missing: {', '.join(missing)}",
        "" if not missing else "python -m tools.setup_models --gguf",
    )]


def run_runtime_checks(
    config: dict,
    *,
    device=None,
    resolved: bool = False,
    llm_mode: str = "configured",
    sd=None,
    ollama_lister: Optional[Callable[[], Iterable[str]]] = None,
    import_fn: Callable[[str], object] = importlib.import_module,
    exists: Callable[[str], bool] = os.path.exists,
    models_needed: Optional[Iterable[str]] = None,
    platform: str = sys.platform,
    pipewire_state: Optional[PipeWireState] = None,
    pipewire_probe: Callable[[], Optional[PipeWireState]] = probe_pipewire_state,
    include_speaker: bool = True,
    require_audio_devices: bool = True,
    require_os_echo_route: bool = True,
) -> list[Check]:
    """Run one shared readiness contract over one selected device profile."""
    merged = config if resolved else resolve_check_config(config, device)[0]
    llm = (merged.get("llm", {}) or {}) if isinstance(merged, dict) else {}
    backend = str(llm.get("backend", "ollama") or "ollama").lower()

    imports = list(RUNTIME_IMPORTS)
    if llm_mode != "echo":
        imports.append("llama_cpp" if backend == "llamacpp" else "ollama")
    checks = check_imports(imports, import_fn=import_fn)
    checks.append(check_sherpa_models(merged, exists=exists))
    if include_speaker:
        checks.append(check_speaker_id(merged, exists=exists))

    if llm_mode != "echo":
        if backend == "llamacpp":
            checks += check_llamacpp_models(merged, exists=exists)
        else:
            needed = (
                tuple(models_needed)
                if models_needed is not None
                else _ollama_models_from_resolved(merged)
            )
            checks += check_ollama(needed, lister=ollama_lister)

    if require_audio_devices:
        checks += check_audio(merged, sd=sd)
    checks += check_audio_frontend(
        merged,
        import_fn=import_fn,
        exists=exists,
        platform=platform,
        pipewire_state=pipewire_state,
        pipewire_probe=pipewire_probe,
        require_os_echo_route=require_os_echo_route,
    )
    return checks
