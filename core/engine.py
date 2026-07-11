from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


def _noop(*_args, **_kwargs) -> None:
    pass


def _noop_text(*_args, **_kwargs) -> None:
    pass


class PlaybackOutcome(str, Enum):
    """Terminal result for one opt-in tracked speech fragment."""

    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    DROPPED = "dropped"
    FAILED = "failed"


@dataclass(frozen=True)
class TrackedSpeech:
    """One engine-independent text fragment whose playback is tracked.

    ``fragment_id`` is an opaque, runtime-owned identifier.  Engines must copy
    it unchanged into the corresponding :class:`PlaybackReceipt`; it must not
    contain transcript text or other user data.
    """

    fragment_id: str
    text: str


@dataclass(frozen=True)
class PlaybackReceipt:
    """Engine attestation of the terminal playback state of one fragment.

    ``safe_text_prefix`` is text the engine can prove reached its playback sink,
    after any engine-side sanitization.  It is deliberately text, rather than a
    character count into :class:`TrackedSpeech`, because the queued input is not
    proof of what played.  An engine without text/audio alignment must report the
    full sanitized fragment only on completion and an empty prefix on a partial
    interruption; callers must never infer a prefix from the sample ratio.

    Sample counts are optional because deterministic/no-audio engines can attest
    text completion without inventing an acoustic sample rate.  When present,
    both counts use ``output_sample_rate`` and ``played_samples`` never exceeds
    ``total_samples``.
    """

    fragment_id: str
    outcome: PlaybackOutcome
    safe_text_prefix: str = ""
    played_samples: Optional[int] = None
    total_samples: Optional[int] = None
    output_sample_rate: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.fragment_id:
            raise ValueError("playback receipt requires a fragment_id")
        for name, value in (
            ("played_samples", self.played_samples),
            ("total_samples", self.total_samples),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")
        if (
            self.played_samples is not None
            and self.total_samples is not None
            and self.played_samples > self.total_samples
        ):
            raise ValueError("played_samples cannot exceed total_samples")
        if self.output_sample_rate is not None and self.output_sample_rate <= 0:
            raise ValueError("output_sample_rate must be positive")

    @property
    def completed(self) -> bool:
        return self.outcome is PlaybackOutcome.COMPLETED

    @property
    def interrupted(self) -> bool:
        return self.outcome is PlaybackOutcome.INTERRUPTED


@dataclass(frozen=True)
class PlaybackCapabilities:
    """Opt-in playback facts an :class:`AudioEngine` can guarantee.

    The all-false value is the backwards-compatible default.  Runtime code must
    capability-check instead of fabricating receipts from legacy ``on_done``
    callbacks, whose meaning differs among existing adapters.
    """

    tracked_terminal: bool = False
    exact_started: bool = False
    sample_counts: bool = False


NO_PLAYBACK_CAPABILITIES = PlaybackCapabilities()


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

    @property
    def playback_capabilities(self) -> PlaybackCapabilities:
        """Playback attestations implemented by this engine.

        Kept non-abstract so every existing engine remains source-compatible.
        Engines that opt in must give every accepted ``speak_tracked`` call one
        and only one terminal callback, including rejection, interruption,
        queue eviction, shutdown, and failure paths.  ``on_started`` is optional
        and fires at most once, before the terminal callback, only after playback
        reaches the engine's sink.  Either callback may run synchronously; a
        hardware engine must not invoke them from its hard real-time audio
        callback.
        """

        return NO_PLAYBACK_CAPABILITIES

    def speak_tracked(
        self,
        speech: TrackedSpeech,
        *,
        on_terminal: Callable[[PlaybackReceipt], None],
        on_started: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Play an opt-in tracked fragment.

        Legacy engines intentionally raise instead of adapting ``speak`` or
        ``on_done`` into a receipt: queue admission and synthesis completion are
        not evidence that audio played.  Callers must check
        :attr:`playback_capabilities` first.
        """

        raise NotImplementedError("this audio engine does not support tracked speech")

    def warm(self) -> None:
        """Exercise the engine's models once so the first turn isn't cold.

        Optional (default no-op; the production engine overrides it). Called off
        the hot path -- the runtime's background warm thread, after ``start()``
        has built the models -- so a slow JIT pass never blocks bring-up."""
