"""
Hardware-free test harness for AudioRecorder barge-in scenarios.

Key components
--------------
MockWakewordService
    A drop-in replacement for ``BaseWakewordService`` that can emit a wakeword
    detection event on demand via ``arm()``.  Inject it into ``AudioRecorder``
    by assigning it to ``recorder._wakeword_service`` after construction.

AudioHarness
    Wraps an ``AudioRecorder`` instance and replaces PortAudio / sounddevice
    with a no-op mock so the worker thread runs without any audio hardware.
    Audio frames are injected directly into ``recorder._audio_queue`` via
    ``inject()``.

make_recorder(...)
    Factory that creates an ``AudioRecorder`` pre-configured for testing:
    tuned thresholds, no real device, barge-in delays set to zero so tests
    are fast and deterministic.

within_ms
    Context-manager SLO gate.  Fails the test if the enclosed block takes
    longer than the requested millisecond budget.

Usage example::

    from tests.harness import AudioHarness, MockWakewordService, make_recorder, within_ms
    from tests.fixtures import voiced_speech, tts_echo, silence

    def test_human_voice_triggers_bargein():
        interrupt_events = []
        rec = make_recorder(on_interrupt=lambda info: interrupt_events.append(info))
        ww  = MockWakewordService()
        rec._wakeword_service = ww

        with AudioHarness(rec) as h:
            h.set_tts_speaking()
            with within_ms(1500):
                h.inject(voiced_speech(1.0))
                h.drain()
        assert interrupt_events, "expected barge-in to fire"
"""

from __future__ import annotations

import queue
import sys
import os
import time
import threading
from typing import Callable, Optional
from unittest.mock import MagicMock, patch

import numpy as np

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.audio import AudioRecorder, ListenerState
from utils.wakeword_service import BaseWakewordService, WakewordEvent


# ── Mock wakeword service ────────────────────────────────────────────────────

class MockWakewordService(BaseWakewordService):
    """
    Controllable wakeword service for tests.

    Call ``arm()`` to inject a detection event that the ``AudioRecorder``
    will pick up on the next ``_update_wakeword_state`` call.
    """

    mode = "mock"

    def __init__(self):
        self._events: queue.Queue = queue.Queue()
        self._available = True

    @property
    def available(self) -> bool:
        return self._available

    @property
    def labels(self) -> list[str]:
        return ["hey_test"]

    @property
    def last_score(self) -> float:
        return 0.99

    def arm(self, label: str = "hey_test", score: float = 0.99):
        """Inject a wakeword-detected event."""
        self._events.put(
            WakewordEvent(
                detected=True,
                score=score,
                label=label,
                timestamp=time.time(),
            )
        )

    def submit_audio(self, audio_chunk: np.ndarray, sample_rate: int):
        pass  # Events are added manually via arm()

    def poll_event(self) -> Optional[WakewordEvent]:
        try:
            return self._events.get_nowait()
        except queue.Empty:
            return None

    def start(self):
        pass

    def stop(self):
        pass


# ── Audio harness ────────────────────────────────────────────────────────────

class AudioHarness:
    """
    Wraps an ``AudioRecorder`` for hardware-free testing.

    The harness patches ``sounddevice.InputStream`` with a no-op mock so that
    ``recorder.start()`` succeeds even without any audio hardware.  Audio is
    fed into the recorder's processing queue via ``inject()``.

    Can be used as a context manager::

        with AudioHarness(recorder) as h:
            h.set_tts_speaking()
            h.inject(voiced_speech(1.0))
            h.drain()
    """

    CHUNK_SIZE = 1024  # frames per injection chunk (matches recorder default)

    def __init__(self, recorder: AudioRecorder):
        self.recorder = recorder
        self._mock_stream: Optional[MagicMock] = None
        self._patches: list = []

    # ── lifecycle ────────────────────────────────────────────────────────

    def start(self):
        """Start the recorder with a mocked audio stream."""
        if self.recorder.is_recording:
            return

        mock_stream = MagicMock()
        mock_stream.start = MagicMock()
        mock_stream.stop = MagicMock()
        mock_stream.close = MagicMock()
        self._mock_stream = mock_stream

        p = patch("utils.audio.sd.InputStream", return_value=mock_stream)
        self._patches.append(p)
        p.start()

        self.recorder.start()

    def stop(self):
        """Stop the recorder and clean up patches."""
        try:
            self.recorder.stop()
        except Exception:
            pass
        for p in self._patches:
            try:
                p.stop()
            except Exception:
                pass
        self._patches.clear()

    def __enter__(self) -> "AudioHarness":
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

    # ── audio injection ──────────────────────────────────────────────────

    def inject(
        self,
        audio: np.ndarray,
        chunk_size: int = CHUNK_SIZE,
        inter_chunk_delay: float = 0.0,
    ):
        """
        Slice *audio* into chunks of *chunk_size* samples and push each chunk
        directly into ``recorder._audio_queue``.

        The last chunk is zero-padded to *chunk_size* if shorter.

        Args:
            audio: Float32 audio array at the recorder's device sample rate.
            chunk_size: Samples per queue entry (should match recorder blocksize).
            inter_chunk_delay: Optional sleep between chunks (seconds).  Use
                               0.0 for maximum injection speed in unit tests.
        """
        audio = audio.flatten().astype(np.float32)
        n = len(audio)
        offset = 0
        while offset < n:
            chunk = audio[offset : offset + chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            self.recorder._audio_queue.put(chunk)
            offset += chunk_size
            if inter_chunk_delay > 0.0:
                time.sleep(inter_chunk_delay)

    def drain(self, timeout: float = 4.0, settle: float = 0.08):
        """
        Block until ``_audio_queue`` is empty and processing settles.

        Args:
            timeout: Maximum time to wait (seconds).
            settle: Extra sleep after queue drains to let the last frame finish.
        """
        deadline = time.time() + timeout
        while not self.recorder._audio_queue.empty():
            if time.time() > deadline:
                break
            time.sleep(0.01)
        time.sleep(settle)

    # ── TTS state helpers ────────────────────────────────────────────────

    def set_tts_speaking(
        self,
        audio_ref: Optional[np.ndarray] = None,
        zero_delays: bool = True,
    ):
        """
        Tell the recorder the assistant has started speaking.

        Args:
            audio_ref: If provided, feed this audio as the AEC reference so the
                       NLMS echo canceller can learn to cancel it from mic input.
            zero_delays: If True, immediately zero the TTS start time so that
                         barge-in timing guards fire on the very first frame
                         (useful in tests where we do not want to wait 500 ms).
        """
        self.recorder.set_assistant_speaking(True)
        if audio_ref is not None:
            sr = self.recorder.device_sample_rate
            self.recorder.set_echo_reference(audio_ref, sr)
        if zero_delays:
            # Back-date TTS start so min-delay guards pass immediately
            self.recorder._tts_start_time = time.time() - 2.0
            self.recorder._aec_ref_set_time = time.time() - 2.0

    def stop_tts(self):
        """Tell the recorder the assistant has stopped speaking."""
        self.recorder.set_assistant_speaking(False)

    # ── convenience ─────────────────────────────────────────────────────

    def listener_state(self) -> ListenerState:
        """Return the current ``ListenerState`` enum value."""
        return self.recorder._listener_state

    def wait_for_state(
        self,
        target: ListenerState,
        timeout: float = 3.0,
        poll: float = 0.01,
    ) -> bool:
        """
        Poll until the recorder enters *target* state.

        Returns True if the state was reached within *timeout* seconds.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.recorder._listener_state == target:
                return True
            time.sleep(poll)
        return False


# ── SLO timing gate ──────────────────────────────────────────────────────────

class within_ms:
    """
    Context manager that asserts the enclosed block completes within a time budget.

    Example::

        with within_ms(800):
            h.inject(voiced_speech(1.0))
            h.drain()
    """

    def __init__(self, limit_ms: float):
        self.limit_ms = limit_ms
        self._t0: float = 0.0

    def __enter__(self) -> "within_ms":
        self._t0 = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return False  # don't suppress existing exceptions
        elapsed_ms = (time.time() - self._t0) * 1000.0
        assert elapsed_ms < self.limit_ms, (
            f"SLO violated: block took {elapsed_ms:.1f} ms "
            f"(budget: {self.limit_ms:.0f} ms)"
        )
        return False


# ── Recorder factory ─────────────────────────────────────────────────────────

def make_recorder(
    callback: Optional[Callable] = None,
    on_interrupt: Optional[Callable] = None,
    wakeword_enabled: bool = False,
    wakeword_policy: str = "strict_required",
    wakeword_miss_limit: int = 3,
    wakeword_recovery_window_sec: float = 2.0,
    vad_threshold: float = 0.01,
    barge_in_min_delay_sec: float = 0.0,
    barge_in_min_delay_after_ref_sec: float = 0.0,
    barge_in_cooldown_sec: float = 0.0,
    barge_in_min_speech_sec: float = 0.2,
    echo_corr_threshold: float = 0.45,
    silence_duration: float = 0.4,
    aec_enabled: bool = True,
    aec_strength: float = 0.5,
    aec_filter_ms: float = 50.0,
    **extra,
) -> AudioRecorder:
    """
    Create an ``AudioRecorder`` with test-friendly defaults.

    During construction, sounddevice is mocked to report a 16 000 Hz device so
    that all duration calculations (min_speech_samples, noise_floor, etc.) use
    the same sample rate as the synthetic audio fixtures.  This avoids the
    mismatch that would occur when real hardware runs at 44 100 Hz.

    All timing guards are set to zero so scenarios produce observable outcomes
    quickly and deterministically.  ``wakeword_service_mode`` is forced to
    ``"local"`` so no subprocess is spawned.  Replace
    ``recorder._wakeword_service`` with a :class:`MockWakewordService` to
    control wakeword detection from the test.

    Args:
        callback: Called when a complete utterance is captured.
        on_interrupt: Called when barge-in is detected during TTS.
        wakeword_enabled: Whether the wakeword gate is active.
        wakeword_policy: One of ``strict_required``, ``hybrid_recovery``,
                         ``legacy_compatible``.
        **extra: Any additional ``AudioRecorder`` keyword arguments.
    """
    if callback is None:
        callback = lambda audio: None  # noqa: E731

    # Force the recorder to believe the device is a 16 000 Hz input so that
    # injected synthetic audio (also at 16 kHz) is interpreted with the
    # correct timing.  Without this, hardware at 44 100 Hz causes duration
    # and min_speech_samples calculations to be 2.76× too large.
    _fake_device = {
        "name": "Mock 16kHz Device",
        "max_input_channels": 1,
        "default_samplerate": 16000.0,
    }
    with patch("utils.audio.sd.query_devices", return_value=_fake_device):
        with patch("utils.audio.sd.default") as _sd_default:
            _sd_default.device = (0, 0)
            return AudioRecorder(
                callback=callback,
                on_interrupt=on_interrupt,
                vad_threshold=vad_threshold,
                silence_duration=silence_duration,
                barge_in_min_delay_sec=barge_in_min_delay_sec,
                barge_in_min_delay_after_ref_sec=barge_in_min_delay_after_ref_sec,
                barge_in_cooldown_sec=barge_in_cooldown_sec,
                barge_in_min_speech_sec=barge_in_min_speech_sec,
                echo_corr_threshold=echo_corr_threshold,
                wakeword_enabled=wakeword_enabled,
                wakeword_policy=wakeword_policy,
                wakeword_miss_limit=wakeword_miss_limit,
                wakeword_recovery_window_sec=wakeword_recovery_window_sec,
                aec_enabled=aec_enabled,
                aec_strength=aec_strength,
                aec_filter_ms=aec_filter_ms,
                wakeword_service_mode="local",  # never start a subprocess in tests
                **extra,
            )
