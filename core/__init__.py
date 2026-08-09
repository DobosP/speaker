"""Lean runtime for the voice assistant (the post-refactor core).

This package replaces the role of the monolithic ``main.py`` + the hand-rolled
``utils/audio.py`` stack. It owns no DSP and no model internals:

- ``engine``      : the AudioEngine seam (STT in, TTS out).
- ``engines``     : concrete engines (sherpa-onnx for production, scripted for tests).
- ``llm``         : thin local-LLM client (Ollama) behind a protocol.
- ``capabilities``: wires the brain's capability registry to a real LLM.
- ``runtime``     : VoiceRuntime, the thin orchestrator (engine <-> brain <-> TTS).

The control-plane "brain" lives in ``always_on_agent/`` and is reused as-is.

Public compatibility exports are resolved lazily so importing a narrow helper
such as :mod:`core.wer` does not initialize the audio runtime or control plane.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .engine import AudioEngine, EngineCallbacks
    from .runtime import VoiceRuntime

__all__ = ["AudioEngine", "EngineCallbacks", "VoiceRuntime"]


def __getattr__(name: str) -> object:
    if name in {"AudioEngine", "EngineCallbacks"}:
        from .engine import AudioEngine, EngineCallbacks

        value = {
            "AudioEngine": AudioEngine,
            "EngineCallbacks": EngineCallbacks,
        }[name]
    elif name == "VoiceRuntime":
        from .runtime import VoiceRuntime

        value = VoiceRuntime
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value
