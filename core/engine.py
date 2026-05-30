from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional


def _noop(*_args, **_kwargs) -> None:
    pass


def _noop_text(*_args, **_kwargs) -> None:
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
    # Command fast-path: a spotted control keyword (e.g. "stop") that the brain
    # should act on directly, skipping ASR-text -> analyzer -> LLM. The argument
    # is the matched keyword phrase; the runtime maps it to a control event.
    on_command: Callable[[str], None] = _noop_text
    # Latency instrumentation: the engine reports stage boundaries it alone can
    # time precisely (user speech end, first TTS audio, barge-in stop) by name;
    # the runtime forwards them to its MetricsRecorder. See ``core.metrics``.
    # Accepts an optional ``at=<perf_counter>`` kwarg so an engine can stamp a
    # stage at a known earlier instant (e.g. SPEECH_END at the true silence
    # onset, before the endpointer's trailing-silence wait elapses).
    on_metric: Callable[..., None] = _noop_text
    # Liveness signal: the engine fires this from its capture loop on its
    # existing heartbeat cadence. The runtime's watchdog uses it to detect a
    # crashed/stalled capture thread. Engines without an audio loop leave the
    # default no-op (the watchdog then silently skips its silence check).
    on_heartbeat: Callable[[], None] = _noop
    # Capture-stream lifecycle signal: ``state`` is one of "open" / "recovering"
    # / "fatal" (mirroring :class:`core.engines._recovering_input.StreamState`).
    # The runtime publishes it as an AgentEvent so the brain knows a silent
    # gap is a device error, not a user pause; the watchdog uses it to
    # suppress its "audio thread stalled" warning during legitimate reopens.
    on_capture_state: Callable[[str, str], None] = lambda state, message: None


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

    def warm(self) -> None:
        """Exercise the engine's models once so the first turn isn't cold.

        Optional (default no-op; the production engine overrides it). Called off
        the hot path -- the runtime's background warm thread, after ``start()``
        has built the models -- so a slow JIT pass never blocks bring-up."""
