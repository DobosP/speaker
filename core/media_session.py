"""Bounded capture transport between a native audio reader and media processing.

The native reader must never wait for VAD, ASR, speaker evidence, callbacks, or
any other downstream work.  :class:`CaptureMailbox` therefore has a fixed time
and frame budget and a non-blocking producer.  When that budget is exceeded it
discards stale queued PCM, keeps the newest admissible block, and attaches one
coalesced discontinuity to the next block consumed.

``CaptureMediaSession`` owns the small reader pump used by a live engine.  It
does not interpret speech and deliberately knows nothing about enrollment,
barge-in authority, recognizers, or playback.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import math
import threading
import time
from typing import Any, Callable, Optional

import numpy as np


class CaptureGapReason(str, Enum):
    """Why contiguous captured PCM was lost or retired."""

    DEVICE_OVERFLOW = "device_overflow"
    MEDIA_BACKPRESSURE = "media_backpressure"
    CAPTURE_RECOVERY = "capture_recovery"


@dataclass(frozen=True)
class CaptureStamp:
    """Immutable source-clock coordinate produced by the native reader."""

    sample_rate_hz: int
    sample_count: int
    captured_started_at: float
    captured_at: float
    capture_epoch: int
    source_generation: int
    source_device: Any
    source_sample_start: int
    source_sample_end: int


@dataclass(frozen=True)
class CaptureBlockContext:
    """Reader-time playout and authority facts used to process one block.

    Reference arrays are model-rate snapshots owned by the block.  They must not
    be replaced with a later read from a shared playback ring: a bounded media
    backlog intentionally decouples capture time from processing time.
    """

    source_domain: Any = None
    authority_source_generation: int = -1
    authority_source_device: Any = None
    os_echo_route_verified: bool = False
    word_cut_route_verified: bool = False
    speaker_authority_available: bool = False
    speaking: bool = False
    barge_watch_active: bool = False
    speak_generation: int = 0
    playback_generation: int = 0
    playback_level: float = 0.0
    playback_onset_at: float = 0.0
    last_playback_at: float = 0.0
    aec_delay_samples: int = 0
    far_reference_zero: Optional[np.ndarray] = None
    far_reference_delayed: Optional[np.ndarray] = None
    coherence_reference: Optional[np.ndarray] = None


@dataclass(frozen=True)
class CaptureGap:
    """One coalesced discontinuity delivered before post-gap PCM."""

    reason: CaptureGapReason
    prior_generation: int
    generation: int
    first_dropped_sequence: Optional[int]
    last_dropped_sequence: Optional[int]
    dropped_frames: int
    dropped_samples: int


@dataclass(frozen=True)
class CapturedBlock:
    """One owned mono PCM block plus its capture lineage."""

    pcm: np.ndarray
    sample_rate_hz: int
    sequence: int
    captured_at: float
    capture_epoch: int
    source_generation: int
    capture_generation: int
    captured_started_at: float = 0.0
    source_device: Any = None
    source_sample_start: int = 0
    source_sample_end: int = 0
    context: Optional[CaptureBlockContext] = None
    gap_before: Optional[CaptureGap] = None

    @property
    def duration_ms(self) -> float:
        return 1000.0 * float(self.pcm.size) / float(self.sample_rate_hz)


@dataclass(frozen=True)
class CaptureOfferResult:
    """Result of one non-blocking producer offer."""

    accepted: bool
    sequence: int
    capture_generation: int
    depth: int
    buffered_ms: float
    dropped_frames: int = 0
    dropped_samples: int = 0


@dataclass(frozen=True)
class _QueuedBlock:
    pcm: np.ndarray
    sample_rate_hz: int
    sequence: int
    captured_at: float
    capture_epoch: int
    source_generation: int
    capture_generation: int
    captured_started_at: float
    source_device: Any
    source_sample_start: int
    source_sample_end: int
    context: Optional[CaptureBlockContext]


_GAP_PRIORITY = {
    CaptureGapReason.MEDIA_BACKPRESSURE: 0,
    CaptureGapReason.DEVICE_OVERFLOW: 1,
    CaptureGapReason.CAPTURE_RECOVERY: 2,
}


@dataclass(frozen=True)
class CaptureControlEvent:
    """One recovery lifecycle transition, fenced to a capture epoch."""

    state: str
    message: str
    capture_epoch: int
    source_generation: int
    source_device: Any
    occurred_at: float
    sequence: int


_CONTROL_PRIORITY = {
    "open": 0,
    "recovering": 1,
    "fatal": 2,
}


class CaptureControlLane:
    """Bounded non-blocking recovery-event lane.

    The recovering input wrapper calls :meth:`offer` from the native reader
    thread.  Offers perform only a short lock/deque mutation; runtime callbacks
    and metrics are deliberately left to the media processor.
    """

    def __init__(
        self,
        *,
        max_events: int = 16,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if isinstance(max_events, bool) or not isinstance(max_events, int):
            raise TypeError("max_events must be an integer")
        if max_events <= 0:
            raise ValueError("max_events must be positive")
        self._max_events = max_events
        self._clock = clock
        self._lock = threading.Lock()
        self._queue: deque[CaptureControlEvent] = deque()
        self._next_sequence = 1
        self._closed = False

    @property
    def depth(self) -> int:
        with self._lock:
            return len(self._queue)

    def offer(
        self,
        state: Any,
        message: str,
        *,
        capture_epoch: int,
        source_generation: int,
        source_device: Any = None,
        occurred_at: Optional[float] = None,
    ) -> bool:
        """Publish without blocking on runtime work or consumer capacity."""

        state_str = str(getattr(state, "value", state))
        at = self._clock() if occurred_at is None else float(occurred_at)
        with self._lock:
            if self._closed:
                return False
            event = CaptureControlEvent(
                state=state_str,
                message=str(message),
                capture_epoch=int(capture_epoch),
                source_generation=int(source_generation),
                source_device=source_device,
                occurred_at=at,
                sequence=self._next_sequence,
            )
            self._next_sequence += 1
            if len(self._queue) >= self._max_events:
                incoming_priority = _CONTROL_PRIORITY.get(state_str, 1)
                # Admit by deterministic severity. Retire the oldest event at
                # the lowest priority no greater than the incoming event:
                # FATAL can replace anything, RECOVERING can replace OPEN or
                # RECOVERING, and OPEN can replace only OPEN. A low-priority
                # notification is rejected when every retained transition is
                # more important, so saturation can never erase recovery/fatal
                # state merely because a device later emits OPEN.
                admissible = tuple(
                    (
                        index,
                        _CONTROL_PRIORITY.get(queued.state, 1),
                    )
                    for index, queued in enumerate(self._queue)
                    if _CONTROL_PRIORITY.get(queued.state, 1) <= incoming_priority
                )
                if not admissible:
                    return False
                victim_priority = min(priority for _index, priority in admissible)
                victim = next(
                    index
                    for index, priority in admissible
                    if priority == victim_priority
                )
                del self._queue[victim]
            self._queue.append(event)
            return True

    def drain(self) -> tuple[CaptureControlEvent, ...]:
        with self._lock:
            events = tuple(self._queue)
            self._queue.clear()
            return events

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._queue.clear()


class CaptureMailbox:
    """A non-blocking producer / blocking consumer PCM mailbox.

    ``buffer_ms`` and ``max_frames`` are simultaneous hard limits.  The
    producer is copied into contiguous mailbox-owned storage. A single block
    larger than the time budget is rejected; its discontinuity remains pending
    and is attached to the next admissible block.
    """

    def __init__(
        self,
        *,
        buffer_ms: float = 300.0,
        max_frames: int = 8,
        source_generation: int = 0,
        source_sample_rate_hz: Optional[int] = None,
        source_device: Any = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if (
            isinstance(buffer_ms, bool)
            or not isinstance(buffer_ms, (int, float))
            or not math.isfinite(float(buffer_ms))
            or float(buffer_ms) <= 0.0
        ):
            raise ValueError("buffer_ms must be a finite positive number")
        if isinstance(max_frames, bool) or not isinstance(max_frames, int):
            raise TypeError("max_frames must be an integer")
        if max_frames <= 0:
            raise ValueError("max_frames must be positive")
        if (
            isinstance(source_generation, bool)
            or not isinstance(source_generation, int)
            or source_generation < 0
        ):
            raise ValueError("source_generation must be a non-negative integer")
        if source_sample_rate_hz is not None and source_sample_rate_hz <= 0:
            raise ValueError("source_sample_rate_hz must be positive")

        self._buffer_seconds = float(buffer_ms) / 1000.0
        self._max_frames = max_frames
        self._source_generation = source_generation
        self._source_sample_rate_hz = source_sample_rate_hz
        self._source_device = source_device
        self._capture_generation = source_generation
        self._clock = clock
        self._condition = threading.Condition()
        self._queue: deque[_QueuedBlock] = deque()
        self._queued_seconds = 0.0
        self._pending_gap: Optional[CaptureGap] = None
        self._next_sequence = 1
        self._closed = False

    @property
    def generation(self) -> int:
        with self._condition:
            return self._capture_generation

    @property
    def source_generation(self) -> int:
        with self._condition:
            return self._source_generation

    @property
    def depth(self) -> int:
        with self._condition:
            return len(self._queue)

    @property
    def buffered_ms(self) -> float:
        with self._condition:
            return 1000.0 * self._queued_seconds

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed

    def offer(
        self,
        pcm: Any,
        *,
        sample_rate_hz: int,
        source_generation: int,
        capture_epoch: int = 0,
        captured_at: Optional[float] = None,
        captured_started_at: Optional[float] = None,
        source_device: Any = None,
        source_sample_start: int = 0,
        source_sample_end: Optional[int] = None,
        context: Optional[CaptureBlockContext] = None,
        device_overflow: bool = False,
    ) -> CaptureOfferResult:
        """Offer one block without ever waiting for consumer capacity."""

        if (
            isinstance(sample_rate_hz, bool)
            or not isinstance(sample_rate_hz, int)
            or sample_rate_hz <= 0
        ):
            raise ValueError("sample_rate_hz must be a positive integer")
        if (
            isinstance(source_generation, bool)
            or not isinstance(source_generation, int)
            or source_generation < 0
        ):
            raise ValueError("source_generation must be a non-negative integer")
        if (
            isinstance(capture_epoch, bool)
            or not isinstance(capture_epoch, int)
            or capture_epoch < 0
        ):
            raise ValueError("capture_epoch must be a non-negative integer")

        source_view = np.asarray(pcm)
        sample_count = int(source_view.size)
        duration = float(sample_count) / float(sample_rate_hz)
        oversize = duration > self._buffer_seconds + 1e-12
        # Native wrappers and test doubles may recycle their read buffer after
        # this call. Own a contiguous snapshot before cross-thread publication.
        # An invalid oversized block is never copied into mailbox storage.
        samples = (
            None
            if oversize
            else np.array(
                source_view,
                dtype="float32",
                copy=True,
                order="C",
            ).reshape(-1)
        )
        at = self._clock() if captured_at is None else float(captured_at)
        if not math.isfinite(at) or at < 0.0:
            raise ValueError("captured_at must be a finite non-negative clock")
        started_at = (
            max(0.0, at - duration)
            if captured_started_at is None
            else float(captured_started_at)
        )
        if not math.isfinite(started_at) or started_at < 0.0 or started_at > at:
            raise ValueError(
                "captured_started_at must be finite, non-negative, and <= captured_at"
            )
        sample_start = int(source_sample_start)
        sample_end = (
            sample_start + sample_count
            if source_sample_end is None
            else int(source_sample_end)
        )
        if (
            sample_start < 0
            or sample_end < sample_start
            or sample_end - sample_start != sample_count
        ):
            raise ValueError(
                "source sample coordinates must exactly span the PCM block"
            )

        with self._condition:
            sequence = self._next_sequence
            self._next_sequence += 1
            if self._closed:
                return CaptureOfferResult(
                    accepted=False,
                    sequence=sequence,
                    capture_generation=self._capture_generation,
                    depth=len(self._queue),
                    buffered_ms=1000.0 * self._queued_seconds,
                )

            source_changed = bool(
                source_generation != self._source_generation
                or source_device != self._source_device
                or (
                    self._source_sample_rate_hz is not None
                    and sample_rate_hz != self._source_sample_rate_hz
                )
            )
            self._source_generation = source_generation
            self._source_sample_rate_hz = sample_rate_hz
            self._source_device = source_device
            over_capacity = bool(
                not source_changed
                and not device_overflow
                and (
                    len(self._queue) >= self._max_frames
                    or self._queued_seconds + duration > self._buffer_seconds + 1e-12
                )
            )

            gap_reason: Optional[CaptureGapReason] = None
            if source_changed:
                gap_reason = CaptureGapReason.CAPTURE_RECOVERY
            elif device_overflow:
                gap_reason = CaptureGapReason.DEVICE_OVERFLOW
            elif over_capacity or oversize:
                gap_reason = CaptureGapReason.MEDIA_BACKPRESSURE

            dropped = tuple(self._queue) if gap_reason is not None else ()
            if gap_reason is not None:
                self._queue.clear()
                self._queued_seconds = 0.0
            rejected = oversize
            dropped_sequences = [block.sequence for block in dropped]
            dropped_samples = sum(block.pcm.size for block in dropped)
            if rejected:
                dropped_sequences.append(sequence)
                dropped_samples += sample_count
            if gap_reason is not None:
                self._record_gap(
                    gap_reason,
                    dropped_sequences=dropped_sequences,
                    dropped_samples=dropped_samples,
                )

            if not rejected:
                assert samples is not None
                self._queue.append(
                    _QueuedBlock(
                        pcm=samples,
                        sample_rate_hz=sample_rate_hz,
                        sequence=sequence,
                        captured_at=at,
                        capture_epoch=capture_epoch,
                        source_generation=source_generation,
                        capture_generation=self._capture_generation,
                        captured_started_at=started_at,
                        source_device=source_device,
                        source_sample_start=sample_start,
                        source_sample_end=sample_end,
                        context=context,
                    )
                )
                self._queued_seconds += duration
                self._condition.notify()

            return CaptureOfferResult(
                accepted=not rejected,
                sequence=sequence,
                capture_generation=self._capture_generation,
                depth=len(self._queue),
                buffered_ms=1000.0 * self._queued_seconds,
                dropped_frames=len(dropped_sequences),
                dropped_samples=dropped_samples,
            )

    def take(self, timeout: Optional[float] = None) -> Optional[CapturedBlock]:
        """Take the next block, or ``None`` after a closed mailbox drains."""

        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or float(timeout) < 0.0
        ):
            raise ValueError("timeout must be a finite non-negative number")
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        with self._condition:
            while not self._queue and not self._closed:
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._condition.wait(remaining)
            if not self._queue:
                return None

            queued = self._queue.popleft()
            self._queued_seconds = max(
                0.0,
                self._queued_seconds
                - float(queued.pcm.size) / float(queued.sample_rate_hz),
            )
            gap = None
            if (
                self._pending_gap is not None
                and queued.capture_generation == self._pending_gap.generation
            ):
                gap = self._pending_gap
                self._pending_gap = None
            return CapturedBlock(
                pcm=queued.pcm,
                sample_rate_hz=queued.sample_rate_hz,
                sequence=queued.sequence,
                captured_at=queued.captured_at,
                capture_epoch=queued.capture_epoch,
                source_generation=queued.source_generation,
                capture_generation=queued.capture_generation,
                captured_started_at=queued.captured_started_at,
                source_device=queued.source_device,
                source_sample_start=queued.source_sample_start,
                source_sample_end=queued.source_sample_end,
                context=queued.context,
                gap_before=gap,
            )

    def close(self) -> None:
        """Reject offers, retire queued PCM, and wake every consumer."""

        with self._condition:
            self._closed = True
            self._queue.clear()
            self._queued_seconds = 0.0
            self._pending_gap = None
            self._condition.notify_all()

    def _record_gap(
        self,
        reason: CaptureGapReason,
        *,
        dropped_sequences: list[int],
        dropped_samples: int,
    ) -> None:
        prior_generation = self._capture_generation
        self._capture_generation += 1
        existing = self._pending_gap
        first = min(dropped_sequences) if dropped_sequences else None
        last = max(dropped_sequences) if dropped_sequences else None
        dropped_frames = len(dropped_sequences)
        if existing is not None:
            if _GAP_PRIORITY[existing.reason] > _GAP_PRIORITY[reason]:
                reason = existing.reason
            prior_generation = existing.prior_generation
            if existing.first_dropped_sequence is not None:
                first = (
                    existing.first_dropped_sequence
                    if first is None
                    else min(existing.first_dropped_sequence, first)
                )
            if existing.last_dropped_sequence is not None:
                last = (
                    existing.last_dropped_sequence
                    if last is None
                    else max(existing.last_dropped_sequence, last)
                )
            dropped_frames += existing.dropped_frames
            dropped_samples += existing.dropped_samples
        self._pending_gap = CaptureGap(
            reason=reason,
            prior_generation=prior_generation,
            generation=self._capture_generation,
            first_dropped_sequence=first,
            last_dropped_sequence=last,
            dropped_frames=dropped_frames,
            dropped_samples=dropped_samples,
        )


class CaptureMediaSession:
    """Own a native capture reader and its bounded mailbox."""

    def __init__(
        self,
        source: Any,
        *,
        block_seconds: float,
        buffer_ms: float = 300.0,
        max_frames: int = 8,
        capture_epoch: int = 0,
        clock: Callable[[], float] = time.perf_counter,
        context_provider: Optional[
            Callable[[CaptureStamp], CaptureBlockContext]
        ] = None,
        thread_name: str = "speaker-capture-reader",
    ) -> None:
        if (
            isinstance(block_seconds, bool)
            or not isinstance(block_seconds, (int, float))
            or not math.isfinite(float(block_seconds))
            or float(block_seconds) <= 0.0
        ):
            raise ValueError("block_seconds must be a finite positive number")
        source_generation = int(getattr(source, "generation", 0) or 0)
        source_sample_rate_hz = int(source.actual_samplerate)
        try:
            source_device = source.actual_device
        except (AttributeError, RuntimeError):
            source_device = None
        self.mailbox = CaptureMailbox(
            buffer_ms=buffer_ms,
            max_frames=max_frames,
            source_generation=source_generation,
            source_sample_rate_hz=source_sample_rate_hz,
            source_device=source_device,
            clock=clock,
        )
        if float(buffer_ms) + 1e-9 < 1000.0 * float(block_seconds):
            raise ValueError(
                "buffer_ms must hold at least one configured capture block"
            )
        self._source = source
        self._block_seconds = float(block_seconds)
        self._buffer_seconds = float(buffer_ms) / 1000.0
        self._capture_epoch = capture_epoch
        self._clock = clock
        self._context_provider = context_provider
        self._stop = threading.Event()
        self._error: Optional[BaseException] = None
        self._started = False
        self._source_sample_cursor = 0
        self._cursor_source_generation = source_generation
        self._cursor_sample_rate_hz = source_sample_rate_hz
        self._cursor_source_device = source_device
        self._thread = threading.Thread(
            target=self._reader_loop,
            name=thread_name,
            daemon=True,
        )

    @property
    def reader_thread(self) -> threading.Thread:
        return self._thread

    @property
    def error(self) -> Optional[BaseException]:
        return self._error

    @property
    def generation(self) -> int:
        return self.mailbox.generation

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def start(self) -> None:
        if self._started:
            raise RuntimeError("capture media session already started")
        self._thread.start()
        self._started = True

    def take(self, timeout: Optional[float] = None) -> Optional[CapturedBlock]:
        return self.mailbox.take(timeout)

    def request_stop(self) -> None:
        self._stop.set()
        self.mailbox.close()

    def join(self, timeout: Optional[float] = None) -> None:
        if self._started:
            self._thread.join(timeout)

    def _reader_loop(self) -> None:
        try:
            while not self._stop.is_set():
                sample_rate_hz = int(self._source.actual_samplerate)
                frames = max(1, int(sample_rate_hz * self._block_seconds))
                pcm, overflowed = self._source.read(frames)
                if self._stop.is_set():
                    break
                sample_rate_hz = int(self._source.actual_samplerate)
                source_generation = int(getattr(self._source, "generation", 0) or 0)
                try:
                    source_device = self._source.actual_device
                except (AttributeError, RuntimeError):
                    source_device = None
                captured_at = self._clock()
                sample_count = int(np.asarray(pcm).size)
                if (
                    source_generation != self._cursor_source_generation
                    or sample_rate_hz != self._cursor_sample_rate_hz
                    or source_device != self._cursor_source_device
                ):
                    self._source_sample_cursor = 0
                    self._cursor_source_generation = source_generation
                    self._cursor_sample_rate_hz = sample_rate_hz
                    self._cursor_source_device = source_device
                sample_start = self._source_sample_cursor
                sample_end = sample_start + sample_count
                self._source_sample_cursor = sample_end
                captured_started_at = max(
                    0.0,
                    captured_at - float(sample_count) / float(sample_rate_hz),
                )
                stamp = CaptureStamp(
                    sample_rate_hz=sample_rate_hz,
                    sample_count=sample_count,
                    captured_started_at=captured_started_at,
                    captured_at=captured_at,
                    capture_epoch=self._capture_epoch,
                    source_generation=source_generation,
                    source_device=source_device,
                    source_sample_start=sample_start,
                    source_sample_end=sample_end,
                )
                context = (
                    self._context_provider(stamp)
                    if (
                        self._context_provider is not None
                        and float(sample_count) / float(sample_rate_hz)
                        <= self._buffer_seconds + 1e-12
                    )
                    else None
                )
                result = self.mailbox.offer(
                    pcm,
                    sample_rate_hz=sample_rate_hz,
                    source_generation=source_generation,
                    capture_epoch=self._capture_epoch,
                    captured_at=captured_at,
                    captured_started_at=captured_started_at,
                    source_device=source_device,
                    source_sample_start=sample_start,
                    source_sample_end=sample_end,
                    context=context,
                    device_overflow=bool(overflowed),
                )
                if not result.accepted and self.mailbox.closed:
                    break
        except BaseException as exc:  # reader failure belongs to the consumer
            if not self._stop.is_set():
                self._error = exc
        finally:
            self.mailbox.close()
