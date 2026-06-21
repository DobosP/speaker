"""PlaybackFIFO barge-in fade-out (core.engines._aec.PlaybackFIFO.flush).

A hard cut zeroes the playback queue in one sample -- a step discontinuity that
clicks on every barge-in. ``flush(fade_samples)`` keeps a short head and ramps it
to silence instead. These pin the buffer behaviour with no audio device.
"""
import numpy as np

from core.engines._aec import PlaybackFIFO


def _drain(fifo, n):
    out = np.empty(n, dtype="float32")
    fifo.read_into(out)
    return out


def test_hard_flush_drops_everything():
    f = PlaybackFIFO(capacity=4000)
    f.write(np.ones(1000, dtype="float32"), lambda: False)
    assert f.count() == 1000
    f.flush()                       # legacy hard cut
    assert f.count() == 0
    assert np.all(_drain(f, 100) == 0.0)


def test_fade_keeps_and_ramps_tail():
    f = PlaybackFIFO(capacity=4000)
    f.write(np.ones(1000, dtype="float32"), lambda: False)
    f.flush(fade_samples=64)
    assert f.count() == 64
    tail = _drain(f, 64)
    # Monotonic non-increasing ramp from ~full to 0 (raised cosine), ending at 0.
    assert tail[0] > 0.9
    assert tail[-1] < 1e-3
    assert np.all(np.diff(tail) <= 1e-6)
    # After the faded tail, the FIFO is empty -> silence.
    assert np.all(_drain(f, 64) == 0.0)


def test_fade_caps_at_available_count():
    f = PlaybackFIFO(capacity=4000)
    f.write(np.ones(10, dtype="float32"), lambda: False)
    f.flush(fade_samples=500)       # asked for more than queued
    assert f.count() == 10          # only what was there


def test_fade_no_click_at_boundary():
    # The last real played sample (~1.0) glides toward 0 without a jump back up.
    f = PlaybackFIFO(capacity=8000)
    f.write(np.ones(2000, dtype="float32"), lambda: False)
    f.flush(fade_samples=128)
    tail = _drain(f, 128)
    assert float(np.max(np.abs(np.diff(tail)))) < 0.1   # smooth, no step


def test_fade_injects_far_less_boundary_transient_than_a_hard_cut():
    """Pins the de-click EFFECT (not just the ramp shape): the step discontinuity a
    HARD cut tees into the echo reference -- the thing that can nudge a false
    self-interrupt -- is materially smaller with the fade. Drain some played
    full-scale samples, flush, then drain the following silence, and compare the
    playback->silence boundary transient."""
    def boundary_jump(fade):
        f = PlaybackFIFO(capacity=8000)
        f.write(np.ones(2000, dtype="float32"), lambda: False)
        played = _drain(f, 500)                 # 500 full-scale samples already played
        f.flush(fade)                           # barge-in cut
        after = _drain(f, 600)                  # faded tail (if any) + silence
        stream = np.concatenate([played, after])
        return float(np.max(np.abs(np.diff(stream))))

    hard = boundary_jump(0)
    faded = boundary_jump(64)                   # ~4 ms @ 16 kHz
    assert hard > 0.9                           # the hard cut IS a full-scale step
    assert faded < hard / 5                     # the fade glides instead
