"""Frame-accurate replay through the production capture-session seam."""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Callable

import numpy as np

from core.media_session import CaptureBlockContext, CapturedBlock

from .corpus import FRAME_SAMPLES, SAMPLE_RATE_HZ, ReplayCase, ReplayTrack


_REFERENCE_ACTIVE_FLOOR = 1e-8


def _bounded_nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _clock_value(clock: Callable[[], float]) -> float:
    value = float(clock())
    if not math.isfinite(value) or value < 0.0:
        raise RuntimeError("capture replay clock returned an invalid value")
    return value


def _readonly_slice(samples: np.ndarray, start: int, end: int) -> np.ndarray:
    owned = np.array(samples[start:end], dtype="float32", copy=True).reshape(-1)
    owned.setflags(write=False)
    return owned


def _decode(track: ReplayTrack) -> np.ndarray:
    decoded = np.frombuffer(track.pcm_bytes, dtype="<f4")
    if decoded.size != track.samples:
        raise ValueError("capture replay track size changed")
    owned = np.array(decoded, dtype="float32", copy=True).reshape(-1)
    owned.setflags(write=False)
    return owned


class _ReplayMailbox:
    """Small compatibility view used by ``SherpaOnnxEngine._capture_loop``."""

    def __init__(self, owner: "ReplayCaptureSession") -> None:
        self._owner = owner

    @property
    def closed(self) -> bool:
        with self._owner._condition:
            return self._owner._closed

    @property
    def depth(self) -> int:
        with self._owner._condition:
            return int(not self._owner._closed)

    @property
    def buffered_ms(self) -> float:
        return 100.0 * float(self.depth)

    @property
    def generation(self) -> int:
        return self._owner.generation


class ReplayCaptureSession:
    """A paced, duck-typed replacement for ``CaptureMediaSession``.

    The first explicit :meth:`start` (or first :meth:`take`) records a
    ``perf_counter``-domain base.  A frame becomes available only once its
    source-sample end coordinate is causal in that clock domain, so endpoint and
    VAD code observe the same 100 ms progression as live capture.

    ``on_exhausted`` is the runner's normal-completion bridge.  A Sherpa runner
    should pass ``engine._running.clear`` (or an equivalent public lifecycle
    callback); it is invoked exactly once before ``mailbox.closed`` becomes
    visible.  This distinguishes expected corpus exhaustion from a failed native
    capture reader.
    """

    def __init__(
        self,
        case: ReplayCase,
        *,
        clock: Callable[[], float] = time.perf_counter,
        on_exhausted: Callable[[], None] | None = None,
        capture_epoch: int = 0,
        source_generation: int = 1,
        capture_generation: int = 1,
        source_device: Any = "capture-replay",
        source_domain: Any = "capture-replay",
    ) -> None:
        if (
            case.sample_rate_hz != SAMPLE_RATE_HZ
            or case.frame_samples != FRAME_SAMPLES
            or case.samples <= 0
            or case.samples % FRAME_SAMPLES
        ):
            raise ValueError("capture replay case has an incompatible frame contract")
        if not callable(clock):
            raise TypeError("clock must be callable")
        if on_exhausted is not None and not callable(on_exhausted):
            raise TypeError("on_exhausted must be callable")
        self.case = case
        self._clock = clock
        self._on_exhausted = on_exhausted
        self._capture_epoch = _bounded_nonnegative_int(
            capture_epoch,
            field="capture_epoch",
        )
        self._source_generation = _bounded_nonnegative_int(
            source_generation,
            field="source_generation",
        )
        self._capture_generation = _bounded_nonnegative_int(
            capture_generation,
            field="capture_generation",
        )
        self._source_device = source_device
        self._source_domain = source_domain
        self._condition = threading.Condition(threading.RLock())
        self._started = False
        self._closed = False
        self._naturally_exhausted = False
        self._base: float | None = None
        self._next_sample = 0
        self._sequence = 0
        self._error: BaseException | None = None

        self._mic = _decode(case.mic)
        zero_track = case.track("far_reference_zero")
        delayed_track = case.track("far_reference_delayed")
        coherence_track = case.track("coherence_reference")
        self._far_zero = (
            _decode(zero_track)
            if zero_track is not None
            else np.zeros(case.samples, dtype="float32")
        )
        if delayed_track is not None:
            self._far_delayed = _decode(delayed_track)
        else:
            derived = np.zeros(case.samples, dtype="float32")
            delay = case.aec_delay_samples
            if zero_track is not None:
                if delay == 0:
                    derived[:] = self._far_zero
                elif delay < case.samples:
                    derived[delay:] = self._far_zero[: case.samples - delay]
            derived.setflags(write=False)
            self._far_delayed = derived
        self._coherence = (
            _decode(coherence_track)
            if coherence_track is not None
            else self._far_zero
        )
        if self._far_zero.flags.writeable:
            self._far_zero.setflags(write=False)
        if self._coherence.flags.writeable:
            self._coherence.setflags(write=False)

        active = np.flatnonzero(np.abs(self._far_zero) > _REFERENCE_ACTIVE_FLOOR)
        self._reference_active_samples = active
        self._first_reference_sample = int(active[0]) if active.size else None
        self._last_reference_sample = int(active[-1]) if active.size else None
        self.mailbox = _ReplayMailbox(self)

    @property
    def error(self) -> BaseException | None:
        with self._condition:
            return self._error

    @property
    def generation(self) -> int:
        return self._capture_generation

    @property
    def is_alive(self) -> bool:
        with self._condition:
            return self._started and not self._closed

    @property
    def timeline_base(self) -> float | None:
        with self._condition:
            return self._base

    @property
    def frames_total(self) -> int:
        return self.case.samples // self.case.frame_samples

    @property
    def frames_taken(self) -> int:
        with self._condition:
            return self._sequence

    @property
    def naturally_exhausted(self) -> bool:
        with self._condition:
            return self._naturally_exhausted

    def start(self) -> None:
        """Anchor the replay timeline without opening an audio device."""

        with self._condition:
            if self._started:
                raise RuntimeError("capture replay session already started")
            if self._closed:
                raise RuntimeError("capture replay session is closed")
            try:
                self._base = _clock_value(self._clock)
            except Exception as exc:
                self._fail_locked(exc)
                raise
            self._started = True

    def _start_if_needed_locked(self) -> None:
        if self._started:
            return
        if self._closed:
            return
        self._base = _clock_value(self._clock)
        self._started = True

    def _fail_locked(self, exc: BaseException) -> None:
        self._error = exc
        self._closed = True
        self._condition.notify_all()

    def _context(
        self,
        *,
        start: int,
        end: int,
        captured_at: float,
    ) -> CaptureBlockContext:
        assert self._base is not None
        far_zero = _readonly_slice(self._far_zero, start, end)
        far_delayed = _readonly_slice(self._far_delayed, start, end)
        coherence = _readonly_slice(self._coherence, start, end)
        playback_level = (
            float(np.sqrt(np.mean(far_zero.astype("float64") ** 2)))
            if far_zero.size
            else 0.0
        )
        speaking = bool(np.any(np.abs(far_zero) > _REFERENCE_ACTIVE_FLOOR))
        onset_at = 0.0
        last_at = 0.0
        if (
            self._first_reference_sample is not None
            and self._first_reference_sample < end
        ):
            onset_at = self._base + (
                float(self._first_reference_sample) / float(self.case.sample_rate_hz)
            )
        if self._last_reference_sample is not None:
            prior_count = int(
                np.searchsorted(
                    self._reference_active_samples,
                    end,
                    side="left",
                )
            )
            if prior_count:
                last_at = self._base + (
                    float(int(self._reference_active_samples[prior_count - 1]) + 1)
                    / float(self.case.sample_rate_hz)
                )
        reference_generation = int(self._first_reference_sample is not None)
        return CaptureBlockContext(
            source_domain=self._source_domain,
            authority_source_generation=self._source_generation,
            authority_source_device=self._source_device,
            os_echo_route_verified=False,
            word_cut_route_verified=False,
            speaker_authority_available=False,
            speaking=speaking,
            barge_watch_active=speaking,
            speak_generation=reference_generation,
            playback_generation=reference_generation,
            playback_level=playback_level,
            playback_onset_at=onset_at,
            last_playback_at=min(last_at, captured_at),
            aec_delay_samples=self.case.aec_delay_samples,
            far_reference_zero=far_zero,
            far_reference_delayed=far_delayed,
            coherence_reference=coherence,
        )

    def _make_block_locked(self, start: int, end: int) -> CapturedBlock:
        assert self._base is not None
        self._sequence += 1
        captured_started_at = self._base + (
            float(start) / float(self.case.sample_rate_hz)
        )
        captured_at = self._base + (
            float(end) / float(self.case.sample_rate_hz)
        )
        pcm = np.array(self._mic[start:end], dtype="float32", copy=True).reshape(-1)
        return CapturedBlock(
            pcm=pcm,
            sample_rate_hz=self.case.sample_rate_hz,
            sequence=self._sequence,
            captured_at=captured_at,
            capture_epoch=self._capture_epoch,
            source_generation=self._source_generation,
            capture_generation=self._capture_generation,
            captured_started_at=captured_started_at,
            source_device=self._source_device,
            source_sample_start=start,
            source_sample_end=end,
            context=self._context(
                start=start,
                end=end,
                captured_at=captured_at,
            ),
            gap_before=None,
        )

    def take(self, timeout: float | None = None) -> CapturedBlock | None:
        """Return the next due 100 ms frame, a timeout, or clean exhaustion."""

        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or float(timeout) < 0.0
        ):
            raise ValueError("timeout must be a finite non-negative number")
        with self._condition:
            if self._closed:
                return None
            try:
                self._start_if_needed_locked()
                assert self._base is not None
                if self._next_sample == self.case.samples:
                    callback = self._on_exhausted
                    self._on_exhausted = None
                    if callback is not None:
                        callback()
                    self._naturally_exhausted = True
                    self._closed = True
                    self._condition.notify_all()
                    return None
                now = _clock_value(self._clock)
                deadline = None if timeout is None else now + float(timeout)
                end = min(
                    self.case.samples,
                    self._next_sample + self.case.frame_samples,
                )
                due_at = self._base + (
                    float(end) / float(self.case.sample_rate_hz)
                )
                while not self._closed:
                    now = _clock_value(self._clock)
                    if now + 1e-12 >= due_at:
                        break
                    wait_seconds = due_at - now
                    if deadline is not None:
                        remaining = deadline - now
                        if remaining <= 0.0:
                            return None
                        wait_seconds = min(wait_seconds, remaining)
                    self._condition.wait(wait_seconds)
                if self._closed:
                    return None
                block = self._make_block_locked(self._next_sample, end)
                self._next_sample = end
                return block
            except Exception as exc:
                self._fail_locked(exc)
                return None

    def request_stop(self) -> None:
        """Stop pacing immediately and retire every remaining replay frame."""

        self.close()

    def close(self) -> None:
        """Idempotently close the session without reporting natural exhaustion."""

        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def join(self, timeout: float | None = None) -> None:
        """Compatibility no-op: replay owns no reader thread."""

        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or float(timeout) < 0.0
        ):
            raise ValueError("timeout must be a finite non-negative number")
