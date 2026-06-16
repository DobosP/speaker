"""audio-bargein-8: moving the coherence reference ingest off the real-time
audio callback (lock-free deque -> capture thread) must not change WHAT the
detector decides -- only which thread feeds it.

This pins the behaviour-preserving property: feeding the same played blocks
"inline" (old: note_playback on the playback/callback thread) vs "drained from a
queue right before decide" (new: capture thread) yields identical verdicts and
incoherent fractions. The detector logic itself is untouched.

NOTE: the real-time benefit (no audio-callback stall on the coherence lock when
coherence_ring_ms is raised, no underrun on small low-spec buffers) is a live
property -- it is validated on device, not here. This test guards the logical
equivalence; the open-speaker A/B guards the acoustics.

Tier 0: pure numpy/scipy, no audio device, no models.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from core.engines.echo_coherence import EchoCoherenceDetector


def _played_and_mic(seed: int, n: int = 12, size: int = 1600):
    rng = np.random.default_rng(seed)
    t = np.arange(size) / 16000.0
    played, mics = [], []
    for i in range(n):
        ref = (0.3 * np.sin(2 * np.pi * 440 * t + 0.5 * i)).astype("float32")
        noise = (0.05 * rng.standard_normal(size)).astype("float32")
        played.append(ref)
        mics.append((ref + noise).astype("float32"))  # echo + uncorrelated noise
    return played, mics


def test_coherence_ingest_order_is_behaviour_preserving():
    inline = EchoCoherenceDetector(16000)
    drained = EchoCoherenceDetector(16000)
    if not inline.available:  # scipy missing on this install
        pytest.skip("scipy unavailable")

    played, mics = _played_and_mic(seed=1)
    q: deque = deque()
    out_inline, out_drained = [], []
    for pb, mic in zip(played, mics):
        # OLD model: note_playback inline, then decide.
        inline.note_playback(pb, 16000)
        out_inline.append(inline.decide(mic))
        # NEW model (audio-bargein-8): the callback would copy+queue the block;
        # the capture thread drains it into note_playback right before decide.
        q.append(np.array(pb, dtype=np.float32, copy=True))
        while q:
            drained.note_playback(q.popleft(), 16000)
        out_drained.append(drained.decide(mic))

    assert out_inline == out_drained, "thread-placement move changed the verdicts"
    assert abs(
        inline.last_incoherent_fraction - drained.last_incoherent_fraction
    ) < 1e-9


def test_queued_block_copy_survives_source_buffer_mutation():
    """`_audio_cb` appends np.array(played, copy=True); `played` is a view into
    the device buffer PortAudio reuses, so the queued copy must be independent."""
    src = np.ones(16, dtype=np.float32)  # stand-in for the reused output view
    q: deque = deque()
    q.append(np.array(src, dtype=np.float32, copy=True))  # exactly what _audio_cb does
    src[:] = 0.0  # PortAudio overwrites the buffer for the next block
    assert np.all(q[0] == 1.0), "queued reference block was aliased to the device buffer"
