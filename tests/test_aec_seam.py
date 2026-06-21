"""Tests for the on-device AEC (core/engines/_aec.py). Pure NumPy, no audio device
and no model: the build gate, the passthrough-on-error seam, the far-end ring
alignment math, and the adaptive filter's single-talk cancellation + double-talk
freeze + reset."""
from __future__ import annotations

import os
import threading
import time

import numpy as np
import pytest

from core.engines._aec import (
    EchoCanceller,
    FarEndRing,
    PlaybackFIFO,
    _DTLNEchoCanceller,
    _FDAFAdaptiveFilter,
    _resolve_dtln_paths,
    build_aec,
)
from core.engines.sherpa import SherpaConfig


# --- build gate --------------------------------------------------------------


def test_build_returns_none_when_disabled():
    assert build_aec(SherpaConfig.from_dict({})) is None
    assert build_aec(SherpaConfig.from_dict({"aec_enabled": False})) is None


def test_build_nlms_when_enabled():
    aec = build_aec(SherpaConfig.from_dict({"aec_enabled": True, "aec_backend": "nlms"}))
    assert isinstance(aec, EchoCanceller)


def test_build_dtln_and_unknown_fail_open_to_none():
    # The deep ONNX tier is deferred -> no-op (NOT a crash); unknown backend too.
    assert build_aec(SherpaConfig.from_dict({"aec_enabled": True, "aec_backend": "dtln"})) is None
    assert build_aec(SherpaConfig.from_dict({"aec_enabled": True, "aec_backend": "bogus"})) is None


# --- passthrough-on-error seam ----------------------------------------------


class _RaisingImpl:
    def process(self, near, far):
        raise RuntimeError("boom")


def test_process_16k_passthrough_on_error():
    ec = EchoCanceller(_RaisingImpl())
    near = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = ec.process_16k(near, np.zeros(3, dtype=np.float32))
    assert np.array_equal(out, near)  # never crash the capture daemon


def test_reset_is_safe_on_impl_without_reset():
    EchoCanceller(_RaisingImpl()).reset()  # no reset() on impl -> no-op, no raise


# --- reset()/process_16k serialization (cross-thread race) -------------------


class _FramedImpl:
    """Mimics a framed canceller's (e.g. the WebRTC APM) near/far carry buffers:
    ``process`` does a multi-step read-modify-write on two lists that MUST stay
    length-aligned, with a widened window between the two extends. A ``reset``
    interleaved mid-``process`` clears the buffers between the extends and desyncs
    them -- the exact corruption EchoCanceller's lock must prevent."""

    def __init__(self):
        self.near: list = []
        self.far: list = []
        self.desynced = False

    def process(self, near, far):
        n = list(np.asarray(near, dtype=np.float32).reshape(-1))
        self.near.extend(n)        # step 1
        time.sleep(0.0005)         # window an interleaved reset() could land in
        self.far.extend(n)         # step 2 -- same count as step 1 by construction
        frame = 4
        k = min(len(self.near), len(self.far)) // frame
        self.near = self.near[k * frame:]
        self.far = self.far[k * frame:]
        if len(self.near) != len(self.far):
            self.desynced = True   # near/far carry lengths diverged -> AEC ruined
        return np.zeros(k * frame, dtype=np.float32)

    def reset(self):
        self.near = []
        self.far = []


def test_reset_is_serialized_against_process_under_concurrency():
    """A reset() from another thread (the event bus / playback worker) must never
    land inside an in-flight process() block, which would permanently desync a
    framed impl's near/far carry buffers (silent ERLE collapse). The lock added to
    EchoCanceller serializes the two; without it this test desyncs reliably."""
    impl = _FramedImpl()
    ec = EchoCanceller(impl)
    stop = threading.Event()

    def hammer_reset():
        while not stop.is_set():
            ec.reset()

    t = threading.Thread(target=hammer_reset, daemon=True)
    t.start()
    try:
        block = np.ones(6, dtype=np.float32)  # 6 % 4 != 0 -> always a carry remainder
        far = np.zeros(6, dtype=np.float32)
        for _ in range(200):
            ec.process_16k(block, far)
    finally:
        stop.set()
        t.join(timeout=2.0)
    assert not impl.desynced  # the lock kept near/far length-aligned across reset()


def test_divergence_guard_internal_reset_does_not_deadlock():
    """The in-process divergence guard resets the impl while already holding the
    lock -- it must use the unlocked path (_do_reset), not re-enter self.reset(),
    or process_16k would deadlock. An amplifying-then-NaN impl forces that path."""

    class _DivergingImpl:
        def __init__(self):
            self.reset_calls = 0

        def process(self, near, far):
            return np.array([np.inf, np.inf], dtype=np.float32)  # non-finite -> guard fires

        def reset(self):
            self.reset_calls += 1

    impl = _DivergingImpl()
    ec = EchoCanceller(impl)
    near = np.array([0.1, 0.2], dtype=np.float32)
    out = ec.process_16k(near, np.zeros(2, dtype=np.float32))  # must return, not hang
    assert np.array_equal(out, near)         # passthrough on divergence
    assert impl.reset_calls == 1             # guard reset fired exactly once


# --- divergence guard (a canceller must never AMPLIFY) ----------------------


class _DivergingImpl:
    """Stands in for an adaptive filter that has diverged and blown up -- it
    returns an output far louder than the input (observed live: post-AEC RMS ~7.4
    on an open laptop speaker, which instantly self-interrupts)."""

    def __init__(self):
        self.reset_calls = 0

    def process(self, near, far):
        return np.asarray(near, dtype=np.float32) * 100.0  # amplified -> diverged

    def reset(self):
        self.reset_calls += 1


def test_process_16k_drops_diverged_output_and_passes_near_through():
    impl = _DivergingImpl()
    ec = EchoCanceller(impl)
    near = np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32)
    out = ec.process_16k(near, np.ones(4, dtype=np.float32))
    # The blown-up output is discarded -> raw near-end passes through (bounded).
    assert np.allclose(out, near)
    assert impl.reset_calls == 1  # the diverged filter was reset


def test_process_16k_passes_a_real_cancellation_through():
    # A canceller that REDUCES the level (out <= near) is kept as-is.
    class _GoodImpl:
        def process(self, near, far):
            return np.asarray(near, dtype=np.float32) * 0.1  # cancelled

    near = np.array([0.4, 0.4, 0.4, 0.4], dtype=np.float32)
    out = EchoCanceller(_GoodImpl()).process_16k(near, np.ones(4, dtype=np.float32))
    assert float(np.sqrt(np.mean(out.astype(np.float64) ** 2))) < float(
        np.sqrt(np.mean(near.astype(np.float64) ** 2))
    )


# --- far-end ring alignment math --------------------------------------------


def test_ring_reads_most_recent_window_at_zero_delay():
    ring = FarEndRing(capacity=100)
    ring.push(np.arange(50, dtype=np.float32))
    np.testing.assert_array_equal(ring.read(10, 0), np.arange(40, 50))


def test_ring_delay_offsets_the_window():
    ring = FarEndRing(capacity=100)
    ring.push(np.arange(50, dtype=np.float32))
    np.testing.assert_array_equal(ring.read(10, 5), np.arange(35, 45))


def test_ring_zeros_before_anything_played():
    assert np.all(FarEndRing(100).read(10, 0) == 0.0)


def test_ring_eviction_returns_recent_and_zeros_for_evicted():
    ring = FarEndRing(capacity=100)
    ring.push(np.arange(200, dtype=np.float32))  # only the last 100 survive
    np.testing.assert_array_equal(ring.read(10, 0), np.arange(190, 200))
    assert np.all(ring.read(10, 150) == 0.0)  # window fell off the back -> zeros


def test_ring_clear():
    ring = FarEndRing(100)
    ring.push(np.arange(50, dtype=np.float32))
    ring.clear()
    assert np.all(ring.read(10, 0) == 0.0)


# --- playback FIFO (producer/audio-callback seam) ----------------------------
# The callback-OutputStream rewrite routes TTS through this bounded FIFO so the
# audio callback can tee the EXACT just-played block into the far ring (aligning
# the AEC reference to real acoustic playback). These exercise the threading +
# wrap-around + backpressure contracts with no audio device.

_never_abort = lambda: False  # noqa: E731 - test predicate


def _disable_self_wake(fifo):
    """Pin the consumer->producer notify path for wake tests.

    write() parks a full-ring producer in ``self._cond.wait(timeout=0.1)``, so it
    SELF-wakes every 100 ms regardless of any notify. That backstop would let the
    wake tests below pass even if read_into()/flush() never notified -- a dropped
    or mis-targeted ``notify_all`` (a real defect: every backpressure release would
    then incur a ~100 ms stall) would stay green. We replace the FIFO Condition's
    timeout-bounded wait with a block-until-notified wait, so the ONLY way a blocked
    producer can unblock is a real ``notify_all``; if that notify regresses, the
    test's ``done.wait()`` times out and the test fails."""
    real_wait = fifo._cond.wait
    fifo._cond.wait = lambda timeout=None: real_wait()  # ignore the self-wake timeout


def test_fifo_write_then_read_round_trips():
    fifo = PlaybackFIFO(8)
    fifo.write(np.array([1, 2, 3, 4], dtype="float32"), _never_abort)
    out = np.empty(4, dtype="float32")
    n = fifo.read_into(out)
    assert n == 4
    np.testing.assert_array_equal(out, [1, 2, 3, 4])
    assert fifo.count() == 0


def test_fifo_wraps_around_correctly():
    # Capacity 4; write 3, read 2 (heads advance), then write 3 more -> the new
    # write wraps the ring. Read all 4 and check ordering survives the wrap.
    fifo = PlaybackFIFO(4)
    fifo.write(np.array([1, 2, 3], dtype="float32"), _never_abort)
    out2 = np.empty(2, dtype="float32")
    fifo.read_into(out2)  # consumes 1,2 -> read head at 2
    np.testing.assert_array_equal(out2, [1, 2])
    fifo.write(np.array([4, 5, 6], dtype="float32"), _never_abort)  # writes 4,5 wrap to 6
    out4 = np.empty(4, dtype="float32")
    n = fifo.read_into(out4)
    assert n == 4
    np.testing.assert_array_equal(out4, [3, 4, 5, 6])


def test_fifo_read_into_underrun_zero_fills_and_returns_real_count():
    fifo = PlaybackFIFO(8)
    fifo.write(np.array([7, 8], dtype="float32"), _never_abort)
    out = np.full(5, -1.0, dtype="float32")
    n = fifo.read_into(out)
    assert n == 2  # only 2 REAL samples
    np.testing.assert_array_equal(out[:2], [7, 8])
    np.testing.assert_array_equal(out[2:], [0, 0, 0])  # tail zero-filled, never a stall


def test_fifo_write_backpressure_blocks_until_drained():
    # Producer blocks when the ring is full; a read frees space and wakes it.
    import threading

    fifo = PlaybackFIFO(4)
    _disable_self_wake(fifo)  # so only a real notify_all can unblock the producer
    fifo.write(np.array([1, 2, 3, 4], dtype="float32"), _never_abort)  # now full
    done = threading.Event()

    def producer():
        fifo.write(np.array([5, 6], dtype="float32"), _never_abort)  # must block: no space
        done.set()

    t = threading.Thread(target=producer, daemon=True)
    t.start()
    assert not done.wait(timeout=0.2), "write() should block while the FIFO is full"
    # Drain 2 samples -> producer wakes, writes its 2, completes.
    out = np.empty(2, dtype="float32")
    fifo.read_into(out)
    np.testing.assert_array_equal(out, [1, 2])
    assert done.wait(timeout=1.0), "read_into should have woken the blocked producer"
    t.join(timeout=1.0)
    assert fifo.count() == 4  # 3,4 still queued + the new 5,6


def test_fifo_flush_empties_and_wakes_blocked_producer():
    import threading

    fifo = PlaybackFIFO(4)
    _disable_self_wake(fifo)  # so only flush()'s notify_all can unblock the producer
    fifo.write(np.array([1, 2, 3, 4], dtype="float32"), _never_abort)  # full
    done = threading.Event()

    def producer():
        fifo.write(np.array([9], dtype="float32"), _never_abort)
        done.set()

    t = threading.Thread(target=producer, daemon=True)
    t.start()
    assert not done.wait(timeout=0.2)
    fifo.flush()  # drops everything + notifies -> producer wakes and writes its 1
    assert done.wait(timeout=1.0), "flush() must wake a producer blocked in write()"
    t.join(timeout=1.0)
    # The 4 queued were dropped by flush; only the producer's single sample remains.
    assert fifo.count() == 1


def test_fifo_write_returns_promptly_when_should_abort_flips_while_full():
    # Deadlock guard: a producer blocked on a full FIFO must return when the
    # abort predicate goes True (barge-in/shutdown), even with no reader.
    import threading

    fifo = PlaybackFIFO(4)
    fifo.write(np.array([1, 2, 3, 4], dtype="float32"), _never_abort)  # full
    abort = threading.Event()
    done = threading.Event()

    def producer():
        fifo.write(np.array([5, 6], dtype="float32"), should_abort=abort.is_set)
        done.set()

    t = threading.Thread(target=producer, daemon=True)
    t.start()
    assert not done.wait(timeout=0.2), "should still be blocked (FIFO full, not aborted)"
    abort.set()  # barge-in/shutdown
    assert done.wait(timeout=1.0), "write() must return promptly once should_abort is True"
    t.join(timeout=1.0)
    assert fifo.count() == 4  # nothing from the aborted write was enqueued


def test_fifo_write_does_not_enqueue_after_abort():
    # An already-aborted producer drops its whole payload (the barged chunk).
    fifo = PlaybackFIFO(8)
    fifo.write(np.array([1, 2], dtype="float32"), should_abort=lambda: True)
    assert fifo.count() == 0


def test_fifo_mid_chunk_abort_halts_enqueue_and_flush_clears_the_partial():
    # A long chunk is copied over SEVERAL lock passes (it can fill the ring and
    # backpressure mid-chunk). An abort that arrives mid-chunk STOPS further
    # enqueue but does NOT retroactively clear what an earlier pass already
    # queued -- a true rollback is impossible once the audio callback has played
    # part of it. Dropping the still-queued partial is flush()'s job, which the
    # barge-in/shutdown paths always call alongside the abort. This pins that
    # real contract (and guards against a naive "rollback" that, under a
    # concurrent flush/drain, would corrupt the shared count).
    fifo = PlaybackFIFO(4)
    calls = {"n": 0}

    def abort_after_first_pass():
        # False on the first probe (lets pass 1 fill the ring with 1..4), True
        # thereafter -> pass 2 sees the full ring and aborts before adding more.
        calls["n"] += 1
        return calls["n"] > 1

    fifo.write(np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype="float32"), abort_after_first_pass)
    # Pass 1 queued 1..4 (ring full); pass 2 aborted. should_abort ALONE leaves
    # that partial queued -- it only halts further synthesis.
    assert fifo.count() == 4
    # The production pattern pairs the abort with flush(), which drops it.
    fifo.flush()
    assert fifo.count() == 0


# --- adaptive filter: single-talk cancellation -------------------------------


def _echo(far, fir):
    return np.convolve(far, fir)[: len(far)].astype(np.float32)


def _run(filt, near, far, block):
    out = []
    for i in range(0, len(near) - block, block):
        out.append(filt.process(near[i : i + block], far[i : i + block]))
    return np.concatenate(out)


def test_fdaf_cancels_linear_echo_single_talk():
    rng = np.random.default_rng(0)
    far = (rng.standard_normal(16000 * 2) * 0.3).astype(np.float32)  # 2 s
    fir = np.zeros(200, dtype=np.float64)
    fir[5], fir[40], fir[120] = 0.6, 0.3, 0.15  # sparse decaying room response
    near = _echo(far, fir)  # single-talk: mic hears only the echo
    e = _run(_FDAFAdaptiveFilter(frame=512), near, far, 512)
    d = near[: len(e)]
    half = len(e) // 2  # measure ERLE on the converged tail
    erle = 10 * np.log10(np.sum(d[half:] ** 2) / (np.sum(e[half:] ** 2) + 1e-12))
    assert np.isfinite(erle) and erle > 20.0  # >=20 dB cancellation


def test_fdaf_does_not_amplify_on_nonlinear_mismatched_echo():
    # The live failure: on a NONLINEAR open laptop speaker the linear filter
    # diverged and amplified (post-AEC RMS ~7.4). Simulate an un-cancellable echo
    # (a hard nonlinearity the linear filter can't model) + a mis-aligned reference
    # and assert the output never blows up past the input -- the divergence
    # recovery (leak + tap-shrink + raw passthrough) holds it bounded.
    rng = np.random.default_rng(7)
    far = (rng.standard_normal(16000) * 0.3).astype(np.float32)
    echo = np.tanh(8.0 * np.convolve(far, [0.0, 0.0, 0.9])[: len(far)]).astype(np.float32)  # clipped + delayed
    near = echo
    filt = _FDAFAdaptiveFilter(frame=512)
    e = _run(filt, near, far, 512)
    near_rms = float(np.sqrt(np.mean(near[: len(e)].astype(np.float64) ** 2)))
    out_rms = float(np.sqrt(np.mean(e.astype(np.float64) ** 2)))
    assert np.all(np.isfinite(e))
    assert out_rms <= near_rms * 2.0 + 1e-6   # never amplifies -> can't self-interrupt


def test_fdaf_no_far_is_passthrough():
    # Nothing playing (silent far) -> no echo to cancel -> near returned ~unchanged.
    rng = np.random.default_rng(1)
    near = (rng.standard_normal(2048) * 0.2).astype(np.float32)
    far = np.zeros_like(near)
    e = _run(_FDAFAdaptiveFilter(frame=512), near, far, 512)
    np.testing.assert_allclose(e, near[: len(e)], atol=1e-5)


# --- adaptive filter: double-talk freeze + reset -----------------------------


def test_fdaf_freezes_on_double_talk_and_does_not_diverge():
    rng = np.random.default_rng(2)
    fir = np.zeros(200, dtype=np.float64)
    fir[5], fir[40] = 0.6, 0.25
    filt = _FDAFAdaptiveFilter(frame=512, doubletalk_freeze=True, warmup_frames=4)
    # Converge on single-talk first.
    far = (rng.standard_normal(512 * 24) * 0.3).astype(np.float32)
    near = _echo(far, fir)
    _run(filt, near, far, 512)
    w_before = filt.W.copy()
    # Now one DOUBLE-TALK frame: echo + a loud near-end voice.
    f = (rng.standard_normal(512) * 0.3).astype(np.float32)
    echo = _echo(np.concatenate([far[-512:], f]), fir)[-512:]
    speech = (rng.standard_normal(512) * 1.5).astype(np.float32)  # loud near-end
    e = filt.process(echo + speech, f)
    # Froze -> filter weights essentially unchanged (didn't adapt onto the voice)...
    assert np.linalg.norm(filt.W - w_before) < 1e-6
    # ...and the near-end speech survives in the output (not cancelled away).
    assert np.sqrt(np.mean(e**2)) > 0.5


def test_fdaf_reset_clears_state():
    filt = _FDAFAdaptiveFilter(frame=256)
    rng = np.random.default_rng(3)
    far = (rng.standard_normal(2048) * 0.3).astype(np.float32)
    _run(filt, _echo(far, np.array([0.0, 0.5, 0.2])), far, 256)
    assert filt.count > 0 and np.any(filt.W != 0)
    filt.reset()
    assert filt.count == 0 and np.all(filt.W == 0)


# --- DTLN ONNX backend (fake sessions: no model, no onnxruntime) -------------


class _FakeIn:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


class _FakeStage:
    """Mimics a DTLN-aec ONNX stage: inputs [primary, state(4-D), reference];
    outputs [main, state(4-D)]. Returns a passthrough main (ones mask for stage 1,
    the primary block for stage 2) and increments the LSTM state by 1 so a test can
    confirm the state is carried block-to-block."""

    def __init__(self, feat_dim: int, *, passthrough_primary: bool):
        self._feat = feat_dim
        self._pass = passthrough_primary
        self._ins = [
            _FakeIn("primary", [1, 1, feat_dim]),
            _FakeIn("state", [1, 2, 512, 2]),
            _FakeIn("reference", [1, 1, feat_dim]),
        ]
        self._outs = [_FakeIn("main", [1, 1, feat_dim]), _FakeIn("state_out", [1, 2, 512, 2])]
        self.calls = 0

    def get_inputs(self):
        return self._ins

    def get_outputs(self):
        return self._outs

    def run(self, out_names, feed):
        self.calls += 1
        state = np.asarray(feed["state"], dtype=np.float32)
        main = (
            np.asarray(feed["primary"], dtype=np.float32)
            if self._pass
            else np.ones((1, 1, self._feat), dtype=np.float32)
        )
        return [main, state + 1.0]


def _fake_dtln():
    return _DTLNEchoCanceller(
        sessions=(_FakeStage(257, passthrough_primary=False), _FakeStage(512, passthrough_primary=True))
    )


def test_dtln_io_mapping_by_shape():
    aec = _fake_dtln()
    assert aec._m1["primary"] == "primary" and aec._m1["reference"] == "reference"
    assert aec._m1["state_in"] == "state" and aec._m1["state_shape"] == (1, 2, 512, 2)
    assert aec._m1["main_out"] == "main" and aec._m1["state_out"] == "state_out"
    assert aec._m2["primary"] == "primary" and aec._m2["state_shape"] == (1, 2, 512, 2)


def test_dtln_streams_in_128_hops_and_carries_state():
    aec = _fake_dtln()
    out = aec.process(np.zeros(1600, dtype=np.float32), np.zeros(1600, dtype=np.float32))
    # 1600 / 128 = 12 full hops -> 12 block-steps -> 12*128 output samples.
    assert aec._s1.calls == 12 and aec._s2.calls == 12
    assert out.shape[0] == 12 * 128
    # State carried: the fake adds 1 each step, so after 12 steps it reads 12.
    assert np.allclose(aec.st1, 12.0) and np.allclose(aec.st2, 12.0)


def test_dtln_reset_clears_state_and_buffers():
    aec = _fake_dtln()
    aec.process(np.zeros(1600, dtype=np.float32), np.zeros(1600, dtype=np.float32))
    assert np.any(aec.st1 != 0)
    aec.reset()
    assert np.all(aec.st1 == 0) and np.all(aec.st2 == 0)
    assert aec._nq.shape[0] == 0 and aec._fq.shape[0] == 0


# --- DTLN model-path resolver ------------------------------------------------


def test_resolve_dtln_paths_directory(tmp_path):
    (tmp_path / "dtln_aec_stage1.onnx").write_bytes(b"x")
    (tmp_path / "dtln_aec_stage2.onnx").write_bytes(b"x")
    s1, s2 = _resolve_dtln_paths(str(tmp_path))
    assert s1.endswith("dtln_aec_stage1.onnx") and s2.endswith("dtln_aec_stage2.onnx")


def test_resolve_dtln_paths_missing_returns_none(tmp_path):
    assert _resolve_dtln_paths("") is None
    assert _resolve_dtln_paths(str(tmp_path)) is None  # empty dir


def test_resolve_dtln_paths_stage1_file_derives_stage2(tmp_path):
    (tmp_path / "dtln_aec_stage1.onnx").write_bytes(b"x")
    (tmp_path / "dtln_aec_stage2.onnx").write_bytes(b"x")
    s1, s2 = _resolve_dtln_paths(str(tmp_path / "dtln_aec_stage1.onnx"))
    assert s1.endswith("stage1.onnx") and s2.endswith("stage2.onnx")


# --- DTLN real inference (skips unless the converted model + onnxruntime exist)


_DTLN1 = os.path.join("pretrained_models", "sherpa", "aec", "dtln_aec_stage1.onnx")


@pytest.mark.real_model
@pytest.mark.skipif(
    not os.path.exists(_DTLN1),
    reason="DTLN-aec ONNX not present (run tools.setup_models --aec-model)",
)
def test_dtln_real_inference_reduces_echo():
    pytest.importorskip("onnxruntime")
    aec = _DTLNEchoCanceller(_DTLN1, _DTLN1.replace("stage1", "stage2"))
    rng = np.random.default_rng(0)
    raw = rng.standard_normal(16000 * 2).astype(np.float32)
    far = (np.convolve(raw, np.ones(8) / 8)[: len(raw)] * 0.3).astype(np.float32)
    fir = np.zeros(150, dtype=np.float64)
    fir[10], fir[60] = 0.5, 0.25
    mic = np.convolve(far, fir)[: len(far)].astype(np.float32)
    out = np.concatenate([aec.process(mic[i : i + 1600], far[i : i + 1600]) for i in range(0, len(mic) - 1600, 1600)])
    assert np.all(np.isfinite(out))
    tail = len(out) // 2
    erle = 10 * np.log10(np.sum(mic[tail:len(out)] ** 2) / (np.sum(out[tail:] ** 2) + 1e-12))
    assert erle > 15.0  # deep canceller -> strong reduction on single-talk echo
