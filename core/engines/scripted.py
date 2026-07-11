from __future__ import annotations

import threading
from typing import Callable, Optional

from ..engine import (
    AudioEngine,
    EngineCallbacks,
    NO_PLAYBACK_CAPABILITIES,
    PlaybackCapabilities,
    PlaybackOutcome,
    PlaybackReceipt,
    TrackedSpeech,
)
from ..metrics import TTS_FIRST_AUDIO


_SCRIPTED_PLAYBACK_CAPABILITIES = PlaybackCapabilities(
    tracked_terminal=True,
    exact_started=True,
    sample_counts=False,
    speech_style_hints=True,
)


class _ScriptedTrackedPlayback:
    """Serialize one scripted fragment's start and terminal callbacks."""

    def __init__(
        self,
        speech: TrackedSpeech,
        on_terminal: Callable[[PlaybackReceipt], None],
        on_started: Optional[Callable[[str], None]],
    ) -> None:
        self.speech = speech
        self._on_terminal = on_terminal
        self._on_started = on_started
        # The callbacks may synchronously call back into the engine (notably an
        # on_started test can request stop), so this must be re-entrant.  Holding
        # it across callback dispatch also serializes a stop from another thread:
        # that stop cannot deliver terminal before an already-claimed start.
        self._lock = threading.RLock()
        self._started = False
        self._terminal = False

    def start(self, callbacks: EngineCallbacks) -> bool:
        with self._lock:
            if self._started or self._terminal:
                return False
            self._started = True
            callbacks.on_speech_start()
            callbacks.on_metric(TTS_FIRST_AUDIO)
            if self._on_started is not None:
                self._on_started(self.speech.fragment_id)
            return True

    def finish(self, outcome: PlaybackOutcome) -> bool:
        with self._lock:
            if self._terminal:
                return False
            self._terminal = True
            self._on_terminal(
                PlaybackReceipt(
                    fragment_id=self.speech.fragment_id,
                    outcome=outcome,
                    safe_text_prefix=(
                        self.speech.text
                        if outcome is PlaybackOutcome.COMPLETED
                        else ""
                    ),
                )
            )
            return True


class ScriptedEngine(AudioEngine):
    """In-memory engine for tests, replay, and a no-audio console demo.

    There is no hardware here. Callers drive recognition by calling
    :meth:`partial` / :meth:`final` / :meth:`barge_in`, and read :attr:`spoken`
    to assert what the assistant tried to say.

    By default ``speak`` completes immediately. With ``hold_speech=True`` it
    stays "speaking" until :meth:`finish_speaking` (or ``stop_speaking``) is
    called, which lets tests inject a barge-in mid-utterance.
    """

    def __init__(self, hold_speech: bool = False):
        self._cb = EngineCallbacks()
        self.spoken: list[str] = []
        self._lock = threading.Lock()
        self._speaking = False
        self._hold = hold_speech
        self._pending_done: Optional[Callable[[], None]] = None
        self._pending_tracked: Optional[_ScriptedTrackedPlayback] = None

    # --- AudioEngine ---
    def start(self, callbacks: EngineCallbacks) -> None:
        self._cb = callbacks

    def stop(self) -> None:
        self.stop_speaking()

    def speak(self, text: str, on_done: Optional[Callable[[], None]] = None) -> None:
        with self._lock:
            replaced_tracked = self._pending_tracked
            self._pending_tracked = None
            self.spoken.append(text)
            self._speaking = True
            self._pending_done = on_done
        if replaced_tracked is not None:
            replaced_tracked.finish(PlaybackOutcome.INTERRUPTED)
        self._cb.on_speech_start()
        self._cb.on_metric(TTS_FIRST_AUDIO)
        if not self._hold:
            self._finish()

    @property
    def playback_capabilities(self) -> PlaybackCapabilities:
        # Backward compatibility for downstream test engines that historically
        # customized ScriptedEngine by overriding only speak().  Silently routing
        # those subclasses around their override through our inherited
        # speak_tracked() would change their behavior.  They stay legacy unless
        # they explicitly override the tracked seam (or capability property).
        if (
            type(self).speak is not ScriptedEngine.speak
            and type(self).speak_tracked is ScriptedEngine.speak_tracked
        ):
            return NO_PLAYBACK_CAPABILITIES
        return _SCRIPTED_PLAYBACK_CAPABILITIES

    def speak_tracked(
        self,
        speech: TrackedSpeech,
        *,
        on_terminal: Callable[[PlaybackReceipt], None],
        on_started: Optional[Callable[[str], None]] = None,
    ) -> None:
        # ScriptedEngine's sink is deterministic but has no sample domain.  It
        # can attest the whole requested text on normal completion; while held,
        # an interruption has no alignment evidence and therefore attests no
        # prefix.  Replacing an already-held tracked fragment interrupts that
        # older fragment so every invocation still has exactly one terminal.
        pending = _ScriptedTrackedPlayback(speech, on_terminal, on_started)
        with self._lock:
            replaced = self._pending_tracked
            self.spoken.append(speech.text)
            self._speaking = True
            self._pending_done = None
            self._pending_tracked = pending
        if replaced is not None:
            replaced.finish(PlaybackOutcome.INTERRUPTED)
        started = pending.start(self._cb)
        if started and not self._hold:
            # Finish only the fragment THIS call installed.  A concurrent
            # speak_tracked may replace it while on_started is in flight; the
            # old caller must never grab and complete that newer fragment
            # before its own start callback (receipt-review race).
            self._finish_tracked(expected=pending)

    def stop_speaking(self) -> None:
        with self._lock:
            was_speaking = self._speaking
            self._speaking = False
            self._pending_done = None
            tracked = self._pending_tracked
            self._pending_tracked = None
        if was_speaking:
            self._cb.on_speech_end()
        if tracked is not None:
            tracked.finish(PlaybackOutcome.INTERRUPTED)

    @property
    def is_speaking(self) -> bool:
        with self._lock:
            return self._speaking

    # --- test/console drivers ---
    def partial(self, text: str) -> None:
        self._cb.on_partial(text)

    def final(self, text: str) -> None:
        self._cb.on_final(text)

    def barge_in(self) -> None:
        self._cb.on_barge_in()

    def command(self, keyword: str) -> None:
        """Simulate the keyword spotter firing a control phrase."""
        self._cb.on_command(keyword)

    def finish_speaking(self) -> None:
        with self._lock:
            tracked = self._pending_tracked is not None
        if tracked:
            self._finish_tracked()
        else:
            self._finish()

    def _finish(self) -> None:
        with self._lock:
            done = self._pending_done
            self._speaking = False
            self._pending_done = None
        self._cb.on_speech_end()
        if done is not None:
            done()

    def _finish_tracked(
        self, expected: Optional[_ScriptedTrackedPlayback] = None
    ) -> None:
        with self._lock:
            tracked = self._pending_tracked
            if tracked is None or (expected is not None and tracked is not expected):
                return
            self._pending_tracked = None
            self._speaking = False
        self._cb.on_speech_end()
        tracked.finish(PlaybackOutcome.COMPLETED)
