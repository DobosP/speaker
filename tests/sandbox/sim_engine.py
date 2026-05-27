from __future__ import annotations

import threading
import time
from typing import Callable, Optional

from core.engine import AudioEngine, EngineCallbacks
from core.metrics import BARGE_IN_STOP, SPEECH_END, TTS_FIRST_AUDIO

from .profiles import DeviceProfile


class SimulatedEngine(AudioEngine):
    """An :class:`AudioEngine` that imitates a real device's timing.

    Recognition: :meth:`say` emits incremental partials ("what" -> "what time"
    -> "what time is it") spaced by ``stt_partial_interval_sec``, waits
    ``stt_endpoint_delay_sec`` (the trailing silence), then fires the final.

    Playback: :meth:`speak` models synthesis + playback duration on a worker
    thread and can be cut off mid-utterance by :meth:`stop_speaking` or a
    :meth:`user_barge_in`, so barge-in *timing* is meaningful.

    Everything runs on real threads, so the brain's concurrency is exercised
    exactly as in production.
    """

    def __init__(self, profile: DeviceProfile):
        self.profile = profile
        self._cb = EngineCallbacks()
        self.spoken: list[str] = []
        self._lock = threading.Lock()
        self._speaking = threading.Event()
        self._interrupt = threading.Event()

    # --- AudioEngine ---
    def start(self, callbacks: EngineCallbacks) -> None:
        self._cb = callbacks

    def stop(self) -> None:
        self._interrupt.set()
        self._speaking.clear()

    def speak(self, text: str, on_done: Optional[Callable[[], None]] = None) -> None:
        self._interrupt.clear()
        self._speaking.set()
        with self._lock:
            self.spoken.append(text)
        threading.Thread(target=self._play, args=(text, on_done), daemon=True).start()

    def stop_speaking(self) -> None:
        self._interrupt.set()

    @property
    def is_speaking(self) -> bool:
        return self._speaking.is_set()

    def _play(self, text: str, on_done: Optional[Callable[[], None]]) -> None:
        self._cb.on_speech_start()
        time.sleep(self.profile.tts_ttfa_sec)
        self._cb.on_metric(TTS_FIRST_AUDIO)
        duration = max(1, len(text.split())) * self.profile.tts_realtime_factor
        end = time.time() + duration
        interrupted = False
        while time.time() < end:
            if self._interrupt.is_set():
                interrupted = True
                self._cb.on_metric(BARGE_IN_STOP)
                break
            time.sleep(0.005)
        self._speaking.clear()
        self._cb.on_speech_end()
        if on_done is not None and not interrupted:
            on_done()

    # --- scenario drivers (called from the test thread) ---
    def say(self, text: str, *, incremental: bool = True) -> None:
        """Speak an utterance to the assistant with realistic STT timing."""
        words = text.split()
        if incremental and words:
            accumulated: list[str] = []
            for word in words:
                accumulated.append(word)
                self._cb.on_partial(" ".join(accumulated))
                time.sleep(self.profile.stt_partial_interval_sec)
        self._cb.on_metric(SPEECH_END)
        time.sleep(self.profile.stt_endpoint_delay_sec)
        self._cb.on_final(text)

    def user_barge_in(self) -> None:
        """Simulate the user starting to talk over the assistant."""
        self._cb.on_barge_in()
