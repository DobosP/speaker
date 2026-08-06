"""asr-tts-2: the offline second pass runs on a dedicated worker thread, off the
real-time capture loop. These pin the finalize/dispatch logic and the worker's
ordering + resilience with fakes -- no models, no audio, no capture thread.

Why this exists: a SYNCHRONOUS second-pass decode on the capture thread stalled
the one loop that also reads the mic + services barge-in (measured in
run-20260617-103630 as white-noise TTS output + false-barge self-interrupts).
The fix keeps EXACTLY ONE final per utterance (the runtime is one-final-per-
utterance: newest-input-wins is a cancel, not an upgrade) but produces it off
the hot path."""

from __future__ import annotations

import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from always_on_agent.acoustic import AcousticLineage, AcousticSpan
from core.engine import EngineCallbacks, TranscriptAbortReason
from core.engines.sherpa import SherpaConfig, SherpaOnnxEngine
from core.engines.speaker_gate import SpeakerGate
from core.metrics import SPEECH_END
from core.realtime_media_stage import RealtimeMediaStage


class _FakeStream:
    def __init__(self, text):
        self.result = SimpleNamespace(text=text)

    def accept_waveform(self, sr, a):
        pass


class _FakeOffline:
    """A stand-in OfflineRecognizer: returns a fixed text for any audio."""

    def __init__(self, text):
        self._text = text

    def create_stream(self):
        return _FakeStream(self._text)

    def decode_stream(self, stream):
        pass


class _Rec:
    """Records the dispatched finals + metrics and the thread they fired on."""

    def __init__(self):
        self.finals: list[str] = []
        self.metrics: list[tuple] = []
        self.final_threads: list[int] = []
        self.aborts = []

    def callbacks(self) -> EngineCallbacks:
        return EngineCallbacks(
            on_final=self._on_final,
            on_metric=self._on_metric,
            on_transcript_abort=self.aborts.append,
        )

    def _on_final(self, text):
        self.finals.append(text)
        self.final_threads.append(threading.get_ident())

    def _on_metric(self, name, **kw):
        self.metrics.append((name, kw))


def _engine(rec: _Rec | None = None, **sherpa) -> SherpaOnnxEngine:
    eng = SherpaOnnxEngine(SherpaConfig.from_dict(sherpa))
    if rec is not None:
        eng._cb = rec.callbacks()
    return eng


def _acoustic(turn: int = 1) -> AcousticLineage:
    return AcousticLineage.single(
        AcousticSpan(
            stream_id="test-stream",
            utterance_id=f"u{turn}",
            turn_index=turn,
        )
    )


def _install_stage(
    eng: SherpaOnnxEngine,
    max_queued: int = 8,
) -> RealtimeMediaStage:
    stage = RealtimeMediaStage(max_queued=max_queued)
    eng._final_stage = stage
    return stage


def _wait_for(predicate, *, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.002)
    return bool(predicate())


# --- the config knob ----------------------------------------------------------


def test_async_is_the_default():
    assert SherpaConfig().asr_final_async is True
    assert SherpaConfig.from_dict({"asr_final_async": False}).asr_final_async is False


# --- the async gate (_maybe_setup_async_final): worker-vs-inline decision ------


def test_gate_no_queue_without_a_second_pass_recognizer():
    eng = _engine()  # asr_final_async default True, but no recognizer built
    assert eng._final_recognizer is None
    eng._maybe_setup_async_final()
    assert eng._final_stage is None  # no recognizer -> inline, byte-identical


def test_gate_no_queue_when_async_off_even_with_a_recognizer():
    # The "off = legacy inline, byte-identical" invariant rests on this gate.
    eng = _engine(asr_final_async=False)
    eng._final_recognizer = _FakeOffline("x")
    eng._maybe_setup_async_final()
    assert eng._final_stage is None


def test_gate_creates_queue_when_async_on_and_recognizer_present():
    eng = _engine(asr_final_async=True)
    eng._final_recognizer = _FakeOffline("x")
    eng._maybe_setup_async_final()
    assert isinstance(eng._final_stage, RealtimeMediaStage)
    snapshot = eng._final_stage.snapshot()
    assert snapshot.max_queued == 8
    assert not snapshot.closed and snapshot.queued == 0


def test_gate_creates_queue_when_only_verifier_is_present():
    eng = _engine(asr_final_async=True)
    eng._final_verifier = object()

    eng._maybe_setup_async_final()

    assert isinstance(eng._final_stage, RealtimeMediaStage)


def test_gate_replaces_closed_stage_and_clears_it_when_ineligible():
    eng = _engine(asr_final_async=True)
    eng._final_recognizer = _FakeOffline("x")
    eng._maybe_setup_async_final()
    first = eng._final_stage
    assert first is not None
    first.close()

    eng._maybe_setup_async_final()
    second = eng._final_stage
    assert second is not None and second is not first
    assert not second.snapshot().closed

    eng._final_recognizer = None
    eng._final_verifier = None
    eng._maybe_setup_async_final()
    assert eng._final_stage is None
    assert second.snapshot().closed


# --- _finalize_and_dispatch: the one shared finalize path ---------------------


def test_finalize_dispatches_upgraded_final_and_backdated_speech_end():
    rec = _Rec()
    eng = _engine(rec)
    eng._final_recognizer = _FakeOffline("Helen, how are you.")
    seg = np.ones(2 * 16000, dtype="float32")  # 2.0s
    # The 2nd pass provides the punctuation/casing upgrade of an AGREEING streaming
    # final -- the common real correction the agreement_guard admits (word overlap).
    eng._finalize_and_dispatch(seg, "helen how are you", 123.5)
    assert rec.finals == [
        "Helen, how are you."
    ]  # the clean 2nd-pass text reached the LLM
    # SPEECH_END is stamped with the captured perf_counter, however late this ran.
    assert (SPEECH_END, {"at": 123.5}) in rec.metrics


def test_finalize_keeps_streaming_when_second_pass_disagrees_on_long_clip():
    # Guard the guard: a 2nd pass that INVENTS a different sentence with near-zero
    # overlap on a long clip is indistinguishable from a hallucination, so the
    # fail-closed agreement_guard keeps the (post-processed) streaming final rather
    # than trusting the 2nd pass. Locks the core/asr_text.py hardening
    # (commits 65bc852 / 5bc03cb) at the finalize/dispatch integration level.
    rec = _Rec()
    eng = _engine(rec)
    eng._final_recognizer = _FakeOffline("Helen, how are you.")
    seg = np.ones(2 * 16000, dtype="float32")  # 2.0s -> not short, but no overlap
    eng._finalize_and_dispatch(seg, "WHOLE HOLLO", 123.5)
    assert rec.finals, "a final was still dispatched"
    kept = rec.finals[0].lower()
    assert (
        "whole" in kept and "helen" not in kept
    )  # streaming kept, hallucination rejected


def test_finalize_without_second_pass_dispatches_streaming_final():
    rec = _Rec()
    eng = _engine(rec)
    assert eng._final_recognizer is None
    eng._finalize_and_dispatch(None, "hello world", 1.0)
    assert rec.finals == ["Hello world"]  # _postprocess_final casing, byte-identical


def test_finalize_2nd_pass_reads_asr_seg_gates_keep_seg():
    """fix 2: the offline 2nd pass decodes the NS-OFF ``asr_seg`` (so the LLM-facing
    final keeps the near-end words NS erased), while the echo-floor + speaker-ID
    gates keep the NS-ON ``seg`` (the louder NS-off signal would clear the learned
    floor too easily and re-open the echo-final cascade). asr_seg=None (every
    non-apm-owns-ns path) decodes ``seg`` -> byte-identical."""
    rec = _Rec()
    eng = _engine(rec)
    seg = np.ones(16000, dtype="float32")  # NS-on
    asr_seg = np.full(16000, 2.0, dtype="float32")  # NS-off, distinct object
    got: dict = {}
    eng._final_transcribe = lambda s, raw: (got.__setitem__("transcribe", s), "final")[
        1
    ]
    eng._final_above_floor = lambda s: (got.__setitem__("floor", s), True)[1]
    original_speaker_decision = eng._speaker_decision_for_final
    eng._speaker_decision_for_final = lambda s: (
        got.__setitem__("speaker", s),
        original_speaker_decision(s),
    )[1]

    eng._finalize_and_dispatch(seg, "raw", 1.0, asr_seg)
    assert got["transcribe"] is asr_seg  # 2nd pass reads the NS-off audio
    assert got["floor"] is seg  # gates keep the NS-on audio
    assert got["speaker"] is seg

    got.clear()
    eng._finalize_and_dispatch(seg, "raw", 1.0)  # asr_seg=None
    assert got["transcribe"] is seg  # falls back to seg (byte-identical)


def test_finalize_drops_below_echo_floor():
    rec = _Rec()
    eng = _engine(rec)
    eng._final_recognizer = _FakeOffline("are you there")
    eng._final_above_floor = lambda seg: False  # L1: echo/ambient, not speech
    eng._finalize_and_dispatch(np.ones(2 * 16000, dtype="float32"), "Ario der", 2.0)
    assert rec.finals == []
    assert ("echo_floor_rejected_final", {}) in rec.metrics
    # a dropped final never stamps SPEECH_END (no turn was delivered)
    assert all(name != SPEECH_END for name, _ in rec.metrics)


def test_finalize_drop_terminalizes_typed_partial_lineage():
    rec = _Rec()
    eng = _engine(rec)
    eng._final_above_floor = lambda _seg: False

    eng._finalize_and_dispatch(
        np.ones(16000, dtype="float32"),
        "echo text",
        2.0,
        acoustic=_acoustic(),
        revision=1,
    )

    assert rec.finals == []
    assert len(rec.aborts) == 1
    assert rec.aborts[0].reason is TranscriptAbortReason.ECHO_REJECTED
    assert rec.aborts[0].revision == 1
    assert rec.aborts[0].acoustic.spans[0].key == _acoustic().spans[0].key


def test_attested_short_interrupt_repair_still_obeys_echo_floor_gate():
    rec = _Rec()
    eng = _engine(rec, asr_final_backend="sense_voice")
    eng._final_recognizer = _FakeOffline("Cancel that.")
    eng._final_above_floor = lambda seg: False
    # The repair happens inside transcription only.  The common finalizer must
    # still reject a below-floor carrier before the recovered control reaches
    # the runtime.
    eng._finalize_and_dispatch(
        np.ones(2 * 16000, dtype="float32"),
        "CASTLE DEATH",
        2.0,
        speech_sec=0.6,
    )
    assert rec.finals == []
    assert ("echo_floor_rejected_final", {}) in rec.metrics


def test_finalize_drops_on_speaker_reject():
    rec = _Rec()
    eng = _engine(rec)
    eng._final_recognizer = _FakeOffline("are you there")
    eng._final_above_floor = lambda seg: True
    eng._speaker_gate = SpeakerGate(
        threshold=0.5,
        embed_fn=lambda _samples, _sample_rate: [0.0, 1.0],
    )
    eng._speaker_gate.enroll_embedding([1.0, 0.0])
    eng._finalize_and_dispatch(np.ones(2 * 16000, dtype="float32"), "Ario der", 2.0)
    assert rec.finals == []
    assert ("speaker_rejected_final", {}) in rec.metrics


# --- _final_worker: ordering, off-thread, resilience --------------------------


def _run_worker(eng: SherpaOnnxEngine) -> threading.Thread:
    stage = eng._final_stage
    assert stage is not None
    eng._running.set()
    t = threading.Thread(target=eng._final_worker, args=(stage,), daemon=True)
    t.start()
    return t


def _drain_raw_finals(stage: RealtimeMediaStage) -> list[str]:
    drained = []
    while (item := stage.take(timeout=0.0)) is not None:
        drained.append(item.payload.raw_final)
        assert stage.finish(item)
    return drained


def test_worker_dispatches_in_capture_order_off_thread():
    rec = _Rec()
    eng = _engine(rec)  # no 2nd pass -> _final_transcribe = post-processed streaming
    stage = _install_stage(eng)
    t = _run_worker(eng)
    for raw in ("one", "two", "three"):
        eng._enqueue_final(np.ones(16000, dtype="float32"), raw, 0.0)
    assert _wait_for(lambda: len(rec.finals) == 3)
    stage.close()
    t.join(timeout=3.0)
    assert not t.is_alive()
    assert rec.finals == ["One", "Two", "Three"]  # single consumer -> capture order
    # every dispatch happened on the worker, NOT the test thread (that's the point)
    assert rec.final_threads and all(
        tid != threading.get_ident() for tid in rec.final_threads
    )


def test_final_delivery_observer_brackets_terminal_work_and_finished_ownership():
    events = []
    observer_finished = threading.Event()
    eng = _engine()
    stage = _install_stage(eng)
    acoustic = _acoustic(4)

    class _Observer:
        def begin_worker_item(self, sequence, observed_acoustic, revision):
            assert stage.snapshot().in_flight == 1
            assert observed_acoustic is acoustic
            events.append(("begin", sequence, revision))

        def finish_worker_item(self, sequence, finish_succeeded):
            snapshot = stage.snapshot()
            assert snapshot.in_flight == 0
            assert snapshot.finished == 1
            events.append(("finish", sequence, finish_succeeded))
            observer_finished.set()

    eng._cb = EngineCallbacks(
        on_final=lambda text: events.append(("callback", text)),
    )
    eng._running.set()
    worker = threading.Thread(
        target=eng._final_worker,
        args=(stage,),
        kwargs={"final_delivery_observer": _Observer()},
        daemon=True,
    )
    worker.start()
    eng._enqueue_final(
        np.ones(16000, dtype="float32"),
        "observed final",
        0.0,
        acoustic=acoustic,
        revision=3,
    )

    assert observer_finished.wait(timeout=1.0)
    stage.close()
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert events == [
        ("begin", 1, 3),
        ("callback", "Observed final"),
        ("finish", 1, True),
    ]


def test_final_delivery_observer_failures_do_not_change_default_worker_delivery(
    caplog,
):
    rec = _Rec()
    eng = _engine(rec)
    stage = _install_stage(eng)

    class _FailingObserver:
        def begin_worker_item(self, *_args):
            raise RuntimeError("fixed begin failure")

        def finish_worker_item(self, *_args):
            raise RuntimeError("fixed finish failure")

    eng._running.set()
    worker = threading.Thread(
        target=eng._final_worker,
        args=(stage,),
        kwargs={"final_delivery_observer": _FailingObserver()},
        daemon=True,
    )
    worker.start()
    for raw in ("one", "two"):
        eng._enqueue_final(np.ones(16000, dtype="float32"), raw, 0.0)

    assert _wait_for(lambda: stage.snapshot().finished == 2)
    stage.close()
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert rec.finals == ["One", "Two"]
    assert rec.aborts == []
    assert caplog.messages.count(
        "final-delivery observer begin_worker_item failed"
    ) == 2
    assert caplog.messages.count(
        "final-delivery observer finish_worker_item failed"
    ) == 2


def test_worker_survives_a_finalize_exception():
    rec = _Rec()
    eng = _engine(rec)
    stage = _install_stage(eng)

    boom = {"first": True}

    def _flaky_floor(seg):
        if boom["first"]:
            boom["first"] = False
            raise RuntimeError("transient")
        return True

    eng._final_above_floor = _flaky_floor
    t = _run_worker(eng)
    eng._enqueue_final(
        np.ones(16000, dtype="float32"),
        "first turn",
        0.0,
        acoustic=_acoustic(1),
        revision=1,
    )
    eng._enqueue_final(
        np.ones(16000, dtype="float32"),
        "second turn",
        0.0,
        acoustic=_acoustic(2),
        revision=1,
    )
    assert _wait_for(lambda: len(rec.finals) == 1 and len(rec.aborts) == 1)
    stage.close()
    t.join(timeout=3.0)
    assert not t.is_alive()
    assert rec.finals == ["Second turn"]  # the bad turn dropped; the worker kept going
    assert rec.aborts[0].reason is TranscriptAbortReason.INTERNAL_ERROR
    assert rec.aborts[0].acoustic.spans[0].key == _acoustic(1).spans[0].key


def test_worker_does_not_abort_after_final_callback_was_attempted():
    attempts = []
    aborts = []
    eng = _engine()

    def _raising_final(result):
        attempts.append(result)
        raise RuntimeError("consumer failed after accepting final")

    eng._cb = EngineCallbacks(
        on_final_result=_raising_final,
        on_transcript_abort=aborts.append,
    )
    stage = _install_stage(eng)
    t = _run_worker(eng)
    eng._enqueue_final(
        np.ones(16000, dtype="float32"),
        "one terminal",
        0.0,
        acoustic=_acoustic(),
        revision=1,
    )
    assert _wait_for(lambda: len(attempts) == 1)
    stage.close()
    t.join(timeout=3.0)

    assert not t.is_alive()
    assert len(attempts) == 1
    assert aborts == []


def test_enqueue_drops_oldest_on_overflow_preserving_capture_order():
    # Overflow must keep the NEWEST utterances in capture order (not reorder),
    # because the runtime supersede is newest-arrival-wins: a stale final arriving
    # after a newer one would wrongly cancel the newer turn. No worker draining ->
    # the bound is hit deterministically.
    rec = _Rec()
    eng = _engine(rec)
    stage = _install_stage(eng, max_queued=2)
    inline_calls = []
    eng._finalize_and_dispatch = lambda *_args, **_kwargs: inline_calls.append(True)
    seg = np.ones(16000, dtype="float32")
    for raw in ("a", "b", "c", "d"):
        eng._enqueue_final(seg, raw, 0.0)
    drained = _drain_raw_finals(stage)
    assert drained == ["c", "d"]  # two oldest dropped; order preserved, FIFO
    # each drop is observable in the run bundle (two oldest were dropped)
    overflow = [
        m for m, _ in rec.metrics if m == "second_pass_queue_overflow_dropped_final"
    ]
    assert len(overflow) == 2
    assert inline_calls == []


def test_enqueue_overflow_terminalizes_each_dropped_typed_turn():
    rec = _Rec()
    eng = _engine(rec)
    _install_stage(eng, max_queued=1)
    seg = np.ones(16000, dtype="float32")

    eng._enqueue_final(seg, "old", 0.0, acoustic=_acoustic(1), revision=1)
    eng._enqueue_final(seg, "new", 0.0, acoustic=_acoustic(2), revision=1)

    assert len(rec.aborts) == 1
    assert rec.aborts[0].reason is TranscriptAbortReason.BACKPRESSURE
    assert rec.aborts[0].acoustic.spans[0].key == _acoustic(1).spans[0].key


def test_enqueue_no_drop_when_room():
    eng = _engine()
    stage = _install_stage(eng)
    seg = np.ones(16000, dtype="float32")
    for raw in ("a", "b", "c"):
        eng._enqueue_final(seg, raw, 0.0)
    assert _drain_raw_finals(stage) == ["a", "b", "c"]


def test_enqueue_owns_read_only_pcm_and_explicit_capture_scope():
    eng = _engine()
    stage = _install_stage(eng)
    seg = np.arange(8, dtype="float32")
    asr_seg = seg + 10.0

    eng._enqueue_final(
        seg,
        "owned",
        1.0,
        asr_seg=asr_seg,
        capture_epoch=7,
        capture_generation=11,
    )
    seg[:] = -1.0
    asr_seg[:] = -2.0

    item = stage.take(timeout=0.0)
    assert item is not None
    assert item.scope.capture_epoch == 7
    assert item.scope.capture_generation == 11
    np.testing.assert_array_equal(item.payload.seg, np.arange(8, dtype="float32"))
    np.testing.assert_array_equal(
        item.payload.asr_seg,
        np.arange(8, dtype="float32") + 10.0,
    )
    assert item.payload.seg.flags.owndata
    assert item.payload.asr_seg.flags.owndata
    assert not item.payload.seg.flags.writeable
    assert not item.payload.asr_seg.flags.writeable
    assert stage.finish(item)


def test_closed_stage_rejects_late_final_without_inline_decode():
    rec = _Rec()
    eng = _engine(rec)
    stage = _install_stage(eng)
    stage.close()
    inline_calls = []
    eng._finalize_and_dispatch = lambda *_args, **_kwargs: inline_calls.append(True)

    eng._enqueue_final(
        np.ones(16000, dtype="float32"),
        "late",
        0.0,
        capture_epoch=0,
        capture_generation=2,
        acoustic=_acoustic(1),
        revision=1,
    )

    assert inline_calls == []
    assert stage.snapshot().rejected_closed == 1
    assert len(rec.aborts) == 1
    assert rec.aborts[0].reason is TranscriptAbortReason.SHUTDOWN


def test_stage_close_wakes_worker_even_while_running_stays_set():
    eng = _engine()
    stage = _install_stage(eng)
    worker = _run_worker(eng)

    stage.close()
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert eng._running.is_set()
    eng._running.clear()


def test_stop_releases_queued_final_and_fences_external_shutdown_abort():
    rec = _Rec()
    eng = _engine(rec)
    stage = _install_stage(eng)
    eng._enqueue_final(
        np.ones(16000, dtype="float32"),
        "queued",
        0.0,
        acoustic=_acoustic(1),
        revision=1,
    )
    assert stage.snapshot().queued == 1
    assert eng._diagnostic_bundle is None

    eng.stop()

    snapshot = stage.snapshot()
    assert snapshot.closed
    assert snapshot.queued == 0
    assert snapshot.close_released == 1
    assert rec.aborts == []


def test_worker_exits_when_running_clears_without_sentinel():
    rec = _Rec()
    eng = _engine(rec)
    stage = _install_stage(eng)
    t = _run_worker(eng)
    eng._running.clear()  # the loop guard -> it exits on the next 0.1s poll
    t.join(timeout=3.0)
    assert not t.is_alive()
    stage.close()


def test_slow_async_final_cannot_callback_after_capture_stop_epoch():
    """A >stop-join second pass is fenced again at the actual callback seam."""
    rec = _Rec()
    eng = _engine(rec)
    stage = _install_stage(eng)
    decode_started = threading.Event()
    release_decode = threading.Event()

    def _slow_final(_seg, _raw, **_kwargs):
        decode_started.set()
        assert release_decode.wait(timeout=3.0)
        return "Late final"

    eng._final_transcribe = _slow_final
    eng._final_above_floor = lambda _seg: True
    t = _run_worker(eng)
    eng._final_thread = t
    with eng._capture_effect_condition:
        epoch = eng._capture_epoch
    eng._enqueue_final(
        np.ones(16000, dtype="float32"),
        "late final",
        1.0,
        capture_epoch=epoch,
    )
    assert decode_started.wait(timeout=1.0)

    eng.stop()  # bounded final-worker join returns while decode remains blocked
    assert stage.snapshot().closed
    assert t.is_alive()
    release_decode.set()
    t.join(timeout=1.0)

    assert not t.is_alive()
    assert rec.finals == []
    assert rec.metrics == []


def test_callback_lease_finishing_during_worker_join_releases_retained_resources():
    """The post-join pass collects resources retained by the first snapshot."""
    callback_started = threading.Event()
    release_callback = threading.Event()
    finals = []

    def _blocking_metric(name, **_kwargs):
        if name == SPEECH_END:
            callback_started.set()
            assert release_callback.wait(timeout=2.0)

    class _Input:
        def __init__(self):
            self.request_close_calls = 0
            self.close_calls = 0

        def request_close(self):
            self.request_close_calls += 1

        def close(self, *, teardown_timeout):
            assert teardown_timeout == 1.0
            self.close_calls += 1
            return True

    class _Output:
        def __init__(self):
            self.stop_calls = 0
            self.close_calls = 0

        def stop(self):
            self.stop_calls += 1

        def close(self):
            self.close_calls += 1

    class _Recorder:
        seconds = 0.0
        path = "retained.wav"

        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    class _ReleaseOnJoin:
        def __init__(self):
            self.join_timeouts = []

        def join(self, timeout):
            self.join_timeouts.append(timeout)
            release_callback.set()

        def is_alive(self):
            return False

    eng = _engine()
    eng._cb = EngineCallbacks(
        on_final=finals.append,
        on_metric=_blocking_metric,
    )
    stage = _install_stage(eng)
    eng._final_transcribe = lambda _seg, _raw, **_kwargs: "Final"
    eng._final_above_floor = lambda _seg: True
    capture_input = _Input()
    output = _Output()
    recorder = _Recorder()
    pre_dsp_recorder = _Recorder()
    eng._stream_in = capture_input
    eng._out_stream = output
    eng._recorder = recorder
    eng._pre_dsp_recorder = pre_dsp_recorder
    eng._play_thread = _ReleaseOnJoin()
    final_thread = _run_worker(eng)
    eng._final_thread = final_thread
    with eng._capture_effect_condition:
        epoch = eng._capture_epoch
    eng._enqueue_final(
        np.ones(16000, dtype="float32"),
        "final",
        1.0,
        capture_epoch=epoch,
    )
    assert callback_started.wait(timeout=1.0)
    assert eng._capture_effects == 1

    eng.stop()

    assert not final_thread.is_alive()
    assert stage.snapshot().closed
    assert capture_input.request_close_calls == 1
    assert capture_input.close_calls == 1
    assert eng._stream_in is None
    assert output.stop_calls == 1
    assert output.close_calls == 1
    assert eng._out_stream is None
    assert recorder.close_calls == 1
    assert pre_dsp_recorder.close_calls == 1
    assert eng._recorder is None
    assert eng._pre_dsp_recorder is None
    assert not eng._capture_resource_hold.is_set()
    assert eng._capture_effects == 0
    assert finals == []


def test_final_callback_stop_defers_cleanup_then_reaches_restart_build(monkeypatch):
    stopped = threading.Event()

    class _Input:
        def __init__(self):
            self.close_calls = 0

        def request_close(self):
            return None

        def close(self, *, teardown_timeout):
            assert teardown_timeout == 1.0
            self.close_calls += 1
            return True

    class _Output:
        def __init__(self):
            self.stop_calls = 0
            self.close_calls = 0

        def stop(self):
            self.stop_calls += 1

        def close(self):
            self.close_calls += 1

    class _Recorder:
        seconds = 0.0
        path = "deferred.wav"

        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    eng = _engine()
    stage = _install_stage(eng)
    capture_input = _Input()
    output = _Output()
    recorder = _Recorder()
    pre_dsp_recorder = _Recorder()
    eng._stream_in = capture_input
    eng._out_stream = output
    eng._recorder = recorder
    eng._pre_dsp_recorder = pre_dsp_recorder
    eng._final_transcribe = lambda _seg, _raw, **_kwargs: "Final"
    eng._final_above_floor = lambda _seg: True

    def _stop_from_final(_text):
        eng.stop()
        stopped.set()

    eng._cb = EngineCallbacks(on_final=_stop_from_final)
    final_thread = _run_worker(eng)
    eng._final_thread = final_thread
    eng._enqueue_final(
        np.ones(16000, dtype="float32"),
        "final",
        1.0,
        capture_epoch=0,
        capture_generation=3,
    )

    assert stopped.wait(timeout=2.0)
    final_thread.join(timeout=2.0)
    assert not final_thread.is_alive()
    assert stage.snapshot().closed
    assert not eng._capture_resource_hold.is_set()
    assert eng._stream_in is None
    assert capture_input.close_calls == 1
    assert eng._out_stream is None
    assert output.stop_calls == 1
    assert output.close_calls == 1
    assert eng._recorder is None
    assert eng._pre_dsp_recorder is None
    assert recorder.close_calls == 1
    assert pre_dsp_recorder.close_calls == 1

    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace())

    def _restart_reached_build():
        raise RuntimeError("restart reached build")

    monkeypatch.setattr(eng, "_build", _restart_reached_build)
    with pytest.raises(RuntimeError, match="restart reached build"):
        eng.start(EngineCallbacks())
    eng._running.clear()


# --- real-model smoke: the REAL SenseVoice second pass runs on the worker ------
# Proves the wiring the fakes can't: the actual offline recognizer plugs into
# _finalize_and_dispatch and dispatches exactly one final per utterance off the
# calling thread. Self-skips without the downloaded model (CI-safe).


@pytest.mark.real_model
def test_real_sense_voice_second_pass_dispatches_one_final_off_thread():
    import os

    from core.engines._sherpa_models import build_final_recognizer

    model = "pretrained_models/sherpa/sense_voice/model.int8.onnx"
    tokens = "pretrained_models/sherpa/sense_voice/tokens.txt"
    if not (os.path.exists(model) and os.path.exists(tokens)):
        pytest.skip("SenseVoice not downloaded (python -m tools.setup_models)")

    cfg = SherpaConfig.from_dict(
        {
            "asr_final_backend": "sense_voice",
            "asr_final_model": model,
            "asr_final_tokens": tokens,
            "asr_final_language": "en",
            "asr_final_async": True,
        }
    )
    rec_model = build_final_recognizer(cfg)
    if rec_model is None:
        pytest.skip("SenseVoice failed to build")

    rec = _Rec()
    eng = SherpaOnnxEngine(cfg)
    eng._cb = rec.callbacks()
    eng._final_recognizer = rec_model
    stage = _install_stage(eng)

    samples = (
        np.load("tests/fixture_audio/recorded_ocean_utterance.npy")
        .astype("float32")
        .reshape(-1)
    )
    t = _run_worker(eng)
    eng._enqueue_final(samples, "raw streaming text", 1.0)
    assert _wait_for(
        lambda: stage.snapshot().queued == 0 and stage.snapshot().in_flight == 0,
        timeout=15.0,
    )
    stage.close()
    t.join(timeout=15.0)
    assert not t.is_alive()
    # Exactly one terminal outcome (dispatched, or a floor/speaker drop) -- the
    # one-final-per-utterance invariant -- produced by the real decode, off-thread.
    drops = [
        m
        for m, _ in rec.metrics
        if m in ("echo_floor_rejected_final", "speaker_rejected_final")
    ]
    assert len(rec.finals) + len(drops) == 1
    if rec.finals:
        assert rec.final_threads[0] != threading.get_ident()
