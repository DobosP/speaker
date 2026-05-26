"""Lean runtime for the voice assistant (the post-refactor core).

This package replaces the role of the monolithic ``main.py`` + the hand-rolled
``utils/audio.py`` stack. It owns no DSP and no model internals:

- ``engine``      : the AudioEngine seam (STT in, TTS out).
- ``engines``     : concrete engines (sherpa-onnx for production, scripted for tests).
- ``llm``         : thin local-LLM client (Ollama) behind a protocol.
- ``capabilities``: wires the brain's capability registry to a real LLM.
- ``runtime``     : VoiceRuntime, the thin orchestrator (engine <-> brain <-> TTS).

The control-plane "brain" lives in ``always_on_agent/`` and is reused as-is.
"""

from .engine import AudioEngine, EngineCallbacks
from .runtime import VoiceRuntime

__all__ = ["AudioEngine", "EngineCallbacks", "VoiceRuntime"]
