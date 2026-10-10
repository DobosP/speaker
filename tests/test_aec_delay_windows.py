"""Exact paired AEC history under wrap, resets, bursts and hostile partitions."""

import numpy as np
import pytest

from core.engines._aec import AecDelayCalibrator


@pytest.mark.parametrize("capacity", [1, 7, 1600, 24000])
def test_window_matches_concatenate_oracle_for_arbitrary_chunks(capacity):
    cal = AecDelayCalibrator(16000, 80, window_ms=capacity / 16, recalc_interval_ms=1e9)
    rng = np.random.default_rng(421)
    expected_mic = np.empty(0, np.float32)
    expected_far = np.empty(0, np.float32)
    for n in (
        0,
        1,
        3,
        1599,
        1600,
        capacity - 1,
        capacity,
        capacity + 1,
        2 * capacity + 7,
    ):
        n = max(0, n)
        mic = rng.standard_normal(n + 3).astype(np.float32)
        far = rng.standard_normal(n).astype(np.float32)
        expected_mic = np.concatenate((expected_mic, mic[:n]))[-capacity:]
        expected_far = np.concatenate((expected_far, far))[-capacity:]
        cal.observe(mic, far)
        assert cal._mic.tobytes() == expected_mic.tobytes()
        assert cal._far.tobytes() == expected_far.tobytes()
        assert cal._mic.dtype == cal._far.dtype == np.float32
        assert 0 <= cal._window_pos < capacity
        assert cal._window_count <= capacity


def test_observe_owns_samples_and_snapshots_do_not_alias_future_history():
    cal = AecDelayCalibrator(16000, 80, window_ms=1, recalc_interval_ms=1e9)
    mic = np.arange(10, dtype=np.float32)
    far = mic + 20
    cal.observe(mic, far)
    snapshot_mic, snapshot_far = cal._mic, cal._far
    mic.fill(-10)
    far.fill(-20)
    np.testing.assert_array_equal(cal._mic, np.arange(10, dtype=np.float32))
    np.testing.assert_array_equal(cal._far, np.arange(10, dtype=np.float32) + 20)
    cal.observe(np.ones(12, np.float32), np.ones(12, np.float32))
    np.testing.assert_array_equal(snapshot_mic, np.arange(10, dtype=np.float32))
    np.testing.assert_array_equal(snapshot_far, np.arange(10, dtype=np.float32) + 20)


@pytest.mark.parametrize("operation", ["reset", "reset_continuity"])
def test_reset_discards_only_history_and_reuses_bounded_storage(operation):
    cal = AecDelayCalibrator(16000, 80, window_ms=1, recalc_interval_ms=1e9)
    cal.observe(np.ones(23, np.float32), np.ones(23, np.float32))
    cal._operating = 321
    cal._acquired = True
    cal._median.extend((300, 321))
    storage = (id(cal._mic_buffer), id(cal._far_buffer))
    getattr(cal, operation)()
    assert cal._mic.size == cal._far.size == 0
    assert cal._since == 0
    assert (id(cal._mic_buffer), id(cal._far_buffer)) == storage
    assert not cal._mic_buffer.any() and not cal._far_buffer.any()
    if operation == "reset_continuity":
        assert cal.current_delay_samples() == 321
        assert cal._acquired and list(cal._median) == [300, 321]
    else:
        assert cal.current_delay_samples() == 80
        assert not cal._acquired and list(cal._median) == []
    cal.observe(np.arange(3, dtype=np.float32), np.arange(3, dtype=np.float32) + 5)
    np.testing.assert_array_equal(cal._mic, np.arange(3, dtype=np.float32))
    np.testing.assert_array_equal(cal._far, np.arange(3, dtype=np.float32) + 5)


def test_oversized_chunk_cannot_retain_an_oversized_history_base():
    cal = AecDelayCalibrator(16000, 80, window_ms=1, recalc_interval_ms=1e9)
    chunk = np.arange(50000, dtype=np.float32)
    cal.observe(chunk, chunk)
    assert cal._mic_buffer.nbytes + cal._far_buffer.nbytes == 2 * 16 * 4
    np.testing.assert_array_equal(cal._mic, chunk[-16:])
    assert cal._mic.base is None and cal._far.base is None


def test_strided_nonfinite_data_and_empty_partner_preserve_exact_bytes():
    cal = AecDelayCalibrator(16000, 80, window_ms=1, recalc_interval_ms=1e9)
    values = np.array([0.0, np.nan, np.inf, -np.inf, -0.0, 0.5], np.float32)[::-1]
    cal.observe(values, np.ones(values.size, np.float32))
    assert cal._mic.tobytes() == values.tobytes()
    before = cal._mic.tobytes(), cal._far.tobytes(), cal._since
    cal.observe(np.ones(20, np.float32), np.empty(0, np.float32))
    assert (cal._mic.tobytes(), cal._far.tobytes(), cal._since) == before


def test_estimator_receives_exact_chronology_and_original_energy_clock():
    cal = AecDelayCalibrator(16000, 80, window_ms=1, recalc_interval_ms=0.5)
    windows = []
    cal._estimate_delay = lambda far, mic: (
        windows.append((far.copy(), mic.copy())) or (321, 1.0)
    )
    cal.observe(np.arange(5, dtype=np.float32), np.zeros(5, np.float32))
    assert cal._since == 0 and windows == []
    cal.observe(np.arange(5, 13, dtype=np.float32), np.ones(8, np.float32))
    assert cal._since == 0 and len(windows) == 1
    np.testing.assert_array_equal(windows[0][1], np.arange(13, dtype=np.float32))
    np.testing.assert_array_equal(
        windows[0][0], np.concatenate((np.zeros(5), np.ones(8))).astype(np.float32)
    )
    assert cal.current_delay_samples() == 321
