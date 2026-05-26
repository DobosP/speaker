from __future__ import annotations

import threading
from typing import Callable, Optional

from ..engine import AudioEngine, EngineCallbacks


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

    # --- AudioEngine ---
    def start(self, callbacks: EngineCallbacks) -> None:
        self._cb = callbacks

    def stop(self) -> None:
        self.stop_speaking()

    def speak(self, text: str, on_done: Optional[Callable[[], None]] = None) -> None:
        with self._lock:
            self.spoken.append(text)
            self._speaking = True
            self._pending_done = on_done
        self._cb.on_speech_start()
        if not self._hold:
            self._finish()

    def stop_speaking(self) -> None:
        with self._lock:
            was_speaking = self._speaking
            self._speaking = False
            self._pending_done = None
        if was_speaking:
            self._cb.on_speech_end()

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

    def finish_speaking(self) -> None:
        self._finish()

    def _finish(self) -> None:
        with self._lock:
            done = self._pending_done
            self._speaking = False
            self._pending_done = None
        self._cb.on_speech_end()
        if done is not None:
            done()
