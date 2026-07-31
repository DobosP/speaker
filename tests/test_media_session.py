from __future__ import annotations

import threading

import numpy as np
import pytest

from core.media_session import (
    CaptureBlockContext,
    CaptureControlLane,
    CaptureGapReason,
    CaptureMailbox,
    CaptureMediaSession,
)


def _block(value: float, samples: int = 100) -> np.ndarray:
    return np.full(samples, value, dtype="float32")


def test_mailbox_preserves_fifo_and_owns_source_storage() -> None:
    mailbox = CaptureMailbox(
        buffer_ms=300.0,
        max_frames=8,
        source_generation=4,
        source_sample_rate_hz=1000,
        clock=lambda: 12.5,
    )
    source = _block(1.0)

    first = mailbox.offer(
        source,
        sample_rate_hz=1000,
        source_generation=4,
        capture_epoch=7,
    )
    mailbox.offer(
        _block(2.0),
        sample_rate_hz=1000,
        source_generation=4,
        capture_epoch=7,
    )
    source.fill(99.0)

    one = mailbox.take(timeout=0.0)
    two = mailbox.take(timeout=0.0)

    assert first.accepted
    assert one is not None and two is not None
    assert (one.sequence, two.sequence) == (1, 2)
    assert one.capture_epoch == 7
    assert one.capture_generation == 4
    assert one.captured_at == 12.5
    assert one.gap_before is None
    assert np.all(one.pcm == 1.0)
    assert np.all(two.pcm == 2.0)
    assert mailbox.depth == 0
    assert mailbox.buffered_ms == 0.0


def test_backpressure_flushes_stale_pcm_and_coalesces_one_gap() -> None:
    mailbox = CaptureMailbox(
        buffer_ms=300.0,
        max_frames=8,
        source_generation=1,
        source_sample_rate_hz=1000,
    )

    for value in range(1, 8):
        result = mailbox.offer(
            _block(float(value)),
            sample_rate_hz=1000,
            source_generation=1,
        )
        assert result.depth <= 3
        assert result.buffered_ms <= 300.0 + 1e-6

    newest = mailbox.take(timeout=0.0)

    assert newest is not None
    assert newest.sequence == 7
    assert np.all(newest.pcm == 7.0)
    assert newest.gap_before is not None
    assert newest.gap_before.reason is CaptureGapReason.MEDIA_BACKPRESSURE
    assert newest.gap_before.prior_generation == 1
    assert newest.gap_before.generation == 3
    assert newest.gap_before.first_dropped_sequence == 1
    assert newest.gap_before.last_dropped_sequence == 6
    assert newest.gap_before.dropped_frames == 6
    assert newest.gap_before.dropped_samples == 600


def test_frame_cap_is_hard_even_when_time_budget_has_room() -> None:
    mailbox = CaptureMailbox(
        buffer_ms=1000.0,
        max_frames=2,
        source_generation=0,
        source_sample_rate_hz=1000,
    )
    for value in (1.0, 2.0, 3.0):
        mailbox.offer(
            _block(value, samples=10),
            sample_rate_hz=1000,
            source_generation=0,
        )

    newest = mailbox.take(timeout=0.0)

    assert mailbox.depth == 0
    assert newest is not None and newest.sequence == 3
    assert newest.gap_before is not None
    assert newest.gap_before.dropped_frames == 2


def test_oversize_block_is_rejected_and_gap_waits_for_admissible_pcm() -> None:
    mailbox = CaptureMailbox(
        buffer_ms=100.0,
        max_frames=8,
        source_generation=2,
        source_sample_rate_hz=1000,
    )

    rejected = mailbox.offer(
        _block(1.0, samples=101),
        sample_rate_hz=1000,
        source_generation=2,
    )
    accepted = mailbox.offer(
        _block(2.0, samples=50),
        sample_rate_hz=1000,
        source_generation=2,
    )
    current = mailbox.take(timeout=0.0)

    assert not rejected.accepted
    assert accepted.accepted
    assert current is not None and current.sequence == 2
    assert current.gap_before is not None
    assert current.gap_before.reason is CaptureGapReason.MEDIA_BACKPRESSURE
    assert current.gap_before.first_dropped_sequence == 1
    assert current.gap_before.last_dropped_sequence == 1
    assert current.gap_before.dropped_frames == 1
    assert current.gap_before.dropped_samples == 101


def test_device_overflow_and_source_recovery_rotate_distinct_generations() -> None:
    mailbox = CaptureMailbox(
        buffer_ms=300.0,
        max_frames=8,
        source_generation=4,
        source_sample_rate_hz=1000,
    )
    mailbox.offer(
        _block(1.0),
        sample_rate_hz=1000,
        source_generation=4,
        device_overflow=True,
    )
    overflow = mailbox.take(timeout=0.0)
    mailbox.offer(
        _block(2.0),
        sample_rate_hz=2000,
        source_generation=5,
    )
    recovery = mailbox.take(timeout=0.0)

    assert overflow is not None and overflow.gap_before is not None
    assert overflow.gap_before.reason is CaptureGapReason.DEVICE_OVERFLOW
    assert overflow.capture_generation == 5
    assert recovery is not None and recovery.gap_before is not None
    assert recovery.gap_before.reason is CaptureGapReason.CAPTURE_RECOVERY
    assert recovery.capture_generation == 6
    assert recovery.source_generation == 5
    assert recovery.sample_rate_hz == 2000


def test_close_wakes_blocked_take_and_rejects_late_offer() -> None:
    mailbox = CaptureMailbox()
    entered = threading.Event()
    result: list[object] = []

    def _take() -> None:
        entered.set()
        result.append(mailbox.take())

    thread = threading.Thread(target=_take)
    thread.start()
    assert entered.wait(1.0)
    mailbox.close()
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert result == [None]
    late = mailbox.offer(
        _block(1.0),
        sample_rate_hz=1000,
        source_generation=0,
    )
    assert not late.accepted
    assert mailbox.depth == 0


def test_close_retires_buffered_pcm_instead_of_draining_it() -> None:
    mailbox = CaptureMailbox()
    mailbox.offer(
        _block(1.0),
        sample_rate_hz=1000,
        source_generation=0,
    )

    mailbox.close()

    assert mailbox.depth == 0
    assert mailbox.buffered_ms == 0.0
    assert mailbox.take(timeout=0.0) is None


class _FastThenBlockedSource:
    actual_samplerate = 1000
    generation = 1

    def __init__(self, count: int = 20) -> None:
        self.count = count
        self.reads = 0
        self.produced = threading.Event()
        self.release = threading.Event()

    def read(self, frames: int):
        self.reads += 1
        if self.reads > self.count:
            self.produced.set()
            self.release.wait(2.0)
        return _block(float(self.reads), frames), False


def test_reader_keeps_native_capture_moving_while_consumer_is_stalled() -> None:
    source = _FastThenBlockedSource()
    session = CaptureMediaSession(
        source,
        block_seconds=0.1,
        buffer_ms=300.0,
        max_frames=8,
    )
    session.start()
    try:
        assert source.produced.wait(1.0)
        assert source.reads >= 21
        assert session.mailbox.depth <= 3
        assert session.mailbox.buffered_ms <= 300.0 + 1e-6

        drained = []
        while True:
            block = session.take(timeout=0.0)
            if block is None:
                break
            drained.append(block)
        assert drained
        assert drained[-1].sequence == 20
        gaps = [block.gap_before for block in drained if block.gap_before]
        assert len(gaps) == 1
        assert gaps[0].reason is CaptureGapReason.MEDIA_BACKPRESSURE
    finally:
        session.request_stop()
        source.release.set()
        session.join(timeout=1.0)
    assert not session.is_alive


class _FailingSource:
    actual_samplerate = 16000
    generation = 1

    def read(self, _frames: int):
        raise RuntimeError("scripted native failure")


def test_reader_failure_is_retained_and_closes_consumer_side() -> None:
    session = CaptureMediaSession(_FailingSource(), block_seconds=0.1)
    session.start()
    session.join(timeout=1.0)

    assert not session.is_alive
    assert isinstance(session.error, RuntimeError)
    assert session.take(timeout=0.0) is None


def test_media_session_rejects_budget_smaller_than_one_native_block() -> None:
    with pytest.raises(ValueError, match="at least one"):
        CaptureMediaSession(
            _FailingSource(),
            block_seconds=0.1,
            buffer_ms=99.0,
        )


class _BlockedSource:
    actual_samplerate = 16000
    generation = 1

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def read(self, frames: int):
        self.entered.set()
        self.release.wait(2.0)
        return np.ones(frames, dtype="float32"), False


def test_reader_does_not_publish_a_native_block_returned_after_stop() -> None:
    source = _BlockedSource()
    session = CaptureMediaSession(source, block_seconds=0.1)
    session.start()
    assert source.entered.wait(1.0)

    session.request_stop()
    source.release.set()
    session.join(timeout=1.0)

    assert not session.is_alive
    assert session.error is None
    assert session.mailbox.depth == 0
    assert session.take(timeout=0.0) is None


class _StampedSource:
    actual_samplerate = 1000
    generation = 4
    actual_device = "mic-a"

    def __init__(self) -> None:
        self.reads = 0
        self.release = threading.Event()

    def read(self, frames: int):
        self.reads += 1
        if self.reads > 1:
            self.release.wait(2.0)
        return _block(float(self.reads), frames), False


def test_reader_binds_source_clock_device_and_immutable_context() -> None:
    source = _StampedSource()
    stamps = []

    def context_provider(stamp):
        stamps.append(stamp)
        return CaptureBlockContext(
            speaking=True,
            speak_generation=8,
            playback_generation=3,
        )

    session = CaptureMediaSession(
        source,
        block_seconds=0.1,
        clock=lambda: 10.0,
        context_provider=context_provider,
    )
    session.start()
    try:
        block = session.take(timeout=1.0)
        assert block is not None
        assert block.captured_at == 10.0
        assert block.captured_started_at == 9.9
        assert block.source_device == "mic-a"
        assert (block.source_sample_start, block.source_sample_end) == (0, 100)
        assert block.context is not None
        assert block.context.speaking
        assert block.context.playback_generation == 3
        assert len(stamps) == 1  # provider ran exactly once for the published block
        assert stamps[0].source_device == "mic-a"
        assert stamps[0].source_sample_end == 100
    finally:
        session.request_stop()
        source.release.set()
        session.join(timeout=1.0)


def test_control_lane_is_bounded_and_retains_recovery_priority() -> None:
    lane = CaptureControlLane(max_events=2, clock=lambda: 1.0)
    lane.offer(
        "open",
        "initial",
        capture_epoch=2,
        source_generation=1,
    )
    lane.offer(
        "open",
        "duplicate",
        capture_epoch=2,
        source_generation=1,
    )
    lane.offer(
        "recovering",
        "lost",
        capture_epoch=2,
        source_generation=1,
    )
    lane.offer(
        "fatal",
        "gone",
        capture_epoch=2,
        source_generation=1,
    )

    events = lane.drain()

    assert lane.depth == 0
    assert [event.state for event in events] == ["recovering", "fatal"]
    assert [event.sequence for event in events] == [3, 4]


def test_control_lane_rejects_lower_priority_offer_when_saturated() -> None:
    lane = CaptureControlLane(max_events=2, clock=lambda: 1.0)
    assert lane.offer(
        "fatal",
        "device gone",
        capture_epoch=2,
        source_generation=1,
    )
    assert lane.offer(
        "recovering",
        "retrying",
        capture_epoch=2,
        source_generation=1,
    )

    assert not lane.offer(
        "open",
        "late open",
        capture_epoch=2,
        source_generation=2,
    )
    events = lane.drain()

    assert [event.state for event in events] == ["fatal", "recovering"]
    assert [event.sequence for event in events] == [1, 2]


def test_control_lane_never_replaces_fatal_with_recovering() -> None:
    lane = CaptureControlLane(max_events=1, clock=lambda: 1.0)
    assert lane.offer(
        "fatal",
        "device gone",
        capture_epoch=2,
        source_generation=1,
    )

    assert not lane.offer(
        "recovering",
        "retrying",
        capture_epoch=2,
        source_generation=2,
    )
    assert [event.state for event in lane.drain()] == ["fatal"]
