"""Acoustic-artifact + hot-mic diagnostics in the sherpa engine.

The hard-real-time output callback (_audio_cb) can't log, so it COUNTS blocks
where the FIFO ran dry mid-block while speaking -- the gap PortAudio zero-fills
into a buzzy/glitchy artifact. The playback loop reports the per-reply delta
off-thread. No models, no audio device -- a fake FIFO drives the callback.
"""
from __future__ import annotations

import numpy as np

from core.engines.sherpa import SherpaConfig, SherpaOnnxEngine


class _Fifo:
    """Stand-in playback FIFO: fills the first ``n`` samples (real audio) and
    zero-fills the rest, returning ``n`` -- exactly read_into's underrun contract."""

    def __init__(self, n: int):
        self.n = n

    def read_into(self, view) -> int:
        view[: self.n] = 0.1
        view[self.n :] = 0.0
        return self.n


def _cb(eng, *, fill, frames=64):
    eng._fifo = _Fifo(fill)
    eng._audio_cb(np.zeros((frames, 1), dtype="float32"), frames, None, None)


def test_audio_cb_counts_only_mid_block_underruns_while_speaking():
    eng = SherpaOnnxEngine(SherpaConfig())
    eng._speaking.set()

    _cb(eng, fill=20)                       # partial fill mid-reply -> a gap
    assert eng._underrun_blocks == 1

    _cb(eng, fill=64)                       # full block -> not an underrun
    assert eng._underrun_blocks == 1

    _cb(eng, fill=0)                        # fully empty (normal end-of-utterance)
    assert eng._underrun_blocks == 1

    eng._speaking.clear()
    _cb(eng, fill=20)                       # partial while NOT speaking -> ignored
    assert eng._underrun_blocks == 1


def test_underrun_counter_is_per_reply_baselined():
    # The playback loop snapshots _underrun_at_reply_start on silent->speaking so
    # it reports only THIS reply's gaps, not the cumulative total.
    eng = SherpaOnnxEngine(SherpaConfig())
    eng._speaking.set()
    for _ in range(3):
        _cb(eng, fill=10)
    assert eng._underrun_blocks == 3
    eng._underrun_at_reply_start = eng._underrun_blocks  # mimic the next reply's start
    _cb(eng, fill=10)
    assert eng._underrun_blocks - eng._underrun_at_reply_start == 1
