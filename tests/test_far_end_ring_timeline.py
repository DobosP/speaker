"""Played-reference timeline tests; no device, model, or private recording."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from core.engines._aec import FarEndRing


def _oracle(history: list[float], capacity: int, length: int, delay: int) -> np.ndarray:
    end = len(history) - max(0, delay)
    start = end - length
    retained_start = max(0, len(history) - capacity)
    return np.asarray(
        [
            history[i] if retained_start <= i < len(history) else 0.0
            for i in range(start, end)
        ],
        dtype=np.float32,
    )


@pytest.mark.parametrize("prefix,count", [(0, 9), (3, 8), (3, 10), (5, 29)])
def test_whole_ring_replacement_preserves_absolute_positions(prefix: int, count: int):
    capacity = 8
    ring = FarEndRing(capacity)
    history = list(range(1, prefix + count + 1))
    ring.push(np.asarray(history[:prefix], dtype=np.float32))
    ring.push(np.asarray(history[prefix:], dtype=np.float32))

    for length in (1, 8, 13):
        for delay in (0, 1, 5, 10, 100):
            np.testing.assert_array_equal(
                ring.read(length, delay), _oracle(history, capacity, length, delay)
            )


@pytest.mark.parametrize("capacity", [1, 3, 8, 31])
def test_arbitrary_chunks_and_delays_match_retained_timeline(capacity: int):
    rng = np.random.default_rng(20261009 + capacity)
    ring = FarEndRing(capacity)
    history: list[float] = []
    for _ in range(100):
        count = int(rng.integers(0, capacity * 4 + 1))
        samples = rng.standard_normal(count).astype(np.float32)
        history.extend(samples.tolist())
        ring.push(samples)
        length = int(rng.integers(0, capacity * 3 + 1))
        delays = (0, -5, capacity // 2, capacity + 1, len(history) + capacity)
        windows = ring.read_windows(length, delays)
        for delay, window in zip(delays, windows):
            np.testing.assert_array_equal(
                window, _oracle(history, capacity, length, delay)
            )


def test_read_results_do_not_alias_ring_or_each_other():
    ring = FarEndRing(8)
    ring.push(np.arange(1, 9, dtype=np.float32))
    first, second = ring.read_windows(8, (0, 0))
    first[:] = -1
    np.testing.assert_array_equal(second, np.arange(1, 9, dtype=np.float32))
    ring.clear()
    np.testing.assert_array_equal(second, np.arange(1, 9, dtype=np.float32))
    assert not ring.read(8, 0).any()


def test_multi_window_snapshot_stays_consistent_during_wrapping_pushes():
    ring = FarEndRing(64)
    ring.push(np.arange(1, 65, dtype=np.float32))
    start = threading.Event()
    finished = threading.Event()
    failures: list[BaseException] = []

    def producer():
        try:
            start.wait(timeout=2)
            cursor = 65
            for count in (3, 64, 101, 7) * 100:
                ring.push(np.arange(cursor, cursor + count, dtype=np.float32))
                cursor += count
        except BaseException as error:
            failures.append(error)
        finally:
            finished.set()

    worker = threading.Thread(target=producer)
    worker.start()
    start.set()
    try:
        for _ in range(1000):
            recent, delayed = ring.read_windows(16, (0, 7))
            end = int(recent[-1]) + 1
            np.testing.assert_array_equal(
                recent, np.arange(end - 16, end, dtype=np.float32)
            )
            np.testing.assert_array_equal(
                delayed, np.arange(end - 23, end - 7, dtype=np.float32)
            )
    finally:
        worker.join(timeout=3)
    assert finished.is_set()
    assert not worker.is_alive()
    assert not failures
