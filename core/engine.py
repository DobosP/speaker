from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional


def _noop() -> None:
    pass


def _noop_text(_text: str) -> None:
    pass


@dataclass
class EngineCallbacks:
    """Events an :class:`AudioEngine` raises into the runtime.

    All callbacks may be invoked from an audio/worker thread, so handlers must
    be cheap and thread-safe (the runtime just publishes onto the event bus).
    """

    on_partial: Callable[[str], None] = _noop_text
    on_final: Callable[[str], None] = _noop_text
    on_barge_in: Callable[[], None] = _noop
    on_speech_start: Callable[[], None] = _noop
    on_speech_end: Callable[[], None] = _noop


class AudioEngine(ABC):
    """Boundary between the audio/STT/TTS stack and the control-plane brain.

    An engine owns everything real-time and acoustic: capture, VAD, streaming
    ASR, endpointing, barge-in detection, and TTS playback. It turns microphone
    audio into ``on_partial``/``on_final`` text and plays text via ``speak``.

    This is the seam that replaces the hand-rolled ``utils/audio.py``. The
    production implementation is :class:`core.engines.sherpa.SherpaOnnxEngine`;
    tests use :class:`core.engines.scripted.ScriptedEngine`.
    """

    @abstractmethod
    def start(self, callbacks: EngineCallbacks) -> None:
        """Begin capturing and recognizing; wire callbacks for transcripts."""

    @abstractmethod
    def stop(self) -> None:
        """Stop capture/playback and release devices."""

    @abstractmethod
    def speak(self, text: str, on_done: Optional[Callable[[], None]] = None) -> None:
        """Synthesize and play ``text``. Non-blocking; ``on_done`` fires when done."""

    @abstractmethod
    def stop_speaking(self) -> None:
        """Immediately halt playback (used on confirmed barge-in)."""

    @property
    def is_speaking(self) -> bool:
        return False
