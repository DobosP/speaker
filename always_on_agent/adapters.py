from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IntegrationTarget:
    name: str
    role: str
    python_package: str | None
    notes: str


INTEGRATION_TARGETS = (
    IntegrationTarget(
        name="Moonshine",
        role="low-latency edge ASR",
        python_package="useful-moonshine-onnx",
        notes="Use as a partial/final STT provider when installed; keep behind SpeechToTextAdapter.",
    ),
    IntegrationTarget(
        name="Pipecat",
        role="frame pipeline inspiration / optional transport pipeline",
        python_package="pipecat-ai",
        notes="Map AgentEvent to frame processors if the project adopts Pipecat later.",
    ),
    IntegrationTarget(
        name="LiveKit Agents",
        role="WebRTC session and production voice-agent runtime",
        python_package="livekit-agents",
        notes="Use for browser/phone sessions; keep local runtime compatible through adapter events.",
    ),
    IntegrationTarget(
        name="Wyoming",
        role="local voice service boundary",
        python_package="wyoming",
        notes="Use for Home Assistant-style wake/STT/TTS services on LAN devices.",
    ),
)


class SpeechToTextAdapter:
    """Boundary for Moonshine, Whisper, whisper.cpp, or remote STT providers."""

    def transcribe_partial(self, audio) -> str:
        raise NotImplementedError

    def transcribe_final(self, audio) -> str:
        raise NotImplementedError


class MoonshineAdapter(SpeechToTextAdapter):
    """
    Lazy adapter for useful-moonshine-onnx.

    This file intentionally does not import Moonshine at module import time, so
    tests and control-plane work do not require model downloads.
    """

    def __init__(self, model_id: str = "moonshine:tiny"):
        self.model_id = model_id
        self._model = None

    def _load(self):
        if self._model is None:
            from utils.stt import get_stt_model

            self._model = get_stt_model(self.model_id)
        return self._model

    def transcribe_partial(self, audio) -> str:
        return str(self._load().transcribe(audio))

    def transcribe_final(self, audio) -> str:
        return str(self._load().transcribe(audio))
