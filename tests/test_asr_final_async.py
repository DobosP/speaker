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

import queue
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from core.engine import EngineCallbacks
from core.engines.sherpa import SherpaConfig, SherpaOnnxEngine
from core.metrics import SPEECH_END


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

    def callbacks(self) -> EngineCallbacks:
        return EngineCallbacks(on_final=self._on_final, on_metric=self._on_metric)

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


# --- the config knob ----------------------------------------------------------


def test_async_is_the_default():
    assert SherpaConfig().asr_final_async is True
    assert SherpaConfig.from_dict({"asr_final_async": False}).asr_final_async is False


# --- the async gate (_maybe_setup_async_final): worker-vs-inline decision ------


def test_gate_no_queue_without_a_second_pass_recognizer():
    eng = _engine()  # asr_final_async default True, but no recognizer built
    assert eng._final_recognizer is None
    eng._maybe_setup_async_final()
    assert eng._final_q is None  # no recognizer -> inline, byte-identical


def test_gate_no_queue_when_async_off_even_with_a_recognizer():
    # The "off = legacy inline, byte-identical" invariant rests on this gate.
    eng = _engine(asr_final_async=False)
    eng._final_recognizer = _FakeOffline("x")
    eng._maybe_setup_async_final()
    assert eng._final_q is None


def test_gate_creates_queue_when_async_on_and_recognizer_present():
    eng = _engine(asr_final_async=True)
    eng._final_recognizer = _FakeOffline("x")
    eng._maybe_setup_async_final()
    assert isinstance(eng._final_q, queue.Queue)


# --- _finalize_and_dispatch: the one shared finalize path ---------------------


def test_finalize_dispatches_upgraded_final_and_backdated_speech_end():
    rec = _Rec()
    eng = _engine(rec)
    eng._final_recognizer = _FakeOffline("Helen, how are you.")
    seg = np.ones(2 * 16000, dtype="float32")  # 2.0s -> not short -> trust 2nd pass
    eng._finalize_and_dispatch(seg, "WHOLE HOLLO", 123.5)
    assert rec.finals == ["Helen, how are you."]  # the clean 2nd-pass text reached the LLM
    # SPEECH_END is stamped with the captured perf_counter, however late this ran.
    assert (SPEECH_END, {"at": 123.5}) in rec.metrics


def test_finalize_without_second_pass_dispatches_streaming_final():
    rec = _Rec()
    eng = _engine(rec)
    assert eng._final_recognizer is None
    eng._finalize_and_dispatch(None, "hello world", 1.0)
    assert rec.finals == ["Hello world"]  # _postprocess_final casing, byte-identical


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


def test_finalize_drops_on_speaker_reject():
    rec = _Rec()
    eng = _engine(rec)
    eng._final_recognizer = _FakeOffline("are you there")
    eng._final_above_floor = lambda seg: True
    eng._should_act_on_final = lambda seg: False  # not the enrolled user
    eng._finalize_and_dispatch(np.ones(2 * 16000, dtype="float32"), "Ario der", 2.0)
    assert rec.finals == []
    assert ("speaker_rejected_final", {}) in rec.metrics


# --- _final_worker: ordering, off-thread, resilience --------------------------


def _run_worker(eng: SherpaOnnxEngine) -> threading.Thread:
    eng._running.set()
    t = threading.Thread(target=eng._final_worker, daemon=True)
    t.start()
    return t


def test_worker_dispatches_in_capture_order_off_thread():
    rec = _Rec()
    eng = _engine(rec)  # no 2nd pass -> _final_transcribe = post-processed streaming
    eng._final_q = queue.Queue(maxsize=8)
    t = _run_worker(eng)
    for raw in ("one", "two", "three"):
        eng._final_q.put((np.ones(16000, dtype="float32"), raw, 0.0))
    eng._final_q.put(None)  # sentinel
    t.join(timeout=3.0)
    assert not t.is_alive()
    assert rec.finals == ["One", "Two", "Three"]  # single consumer -> capture order
    # every dispatch happened on the worker, NOT the test thread (that's the point)
    assert rec.final_threads and all(tid != threading.get_ident() for tid in rec.final_threads)


def test_worker_survives_a_finalize_exception():
    rec = _Rec()
    eng = _engine(rec)
    eng._final_q = queue.Queue(maxsize=8)

    boom = {"first": True}

    def _flaky_floor(seg):
        if boom["first"]:
            boom["first"] = False
            raise RuntimeError("transient")
        return True

    eng._final_above_floor = _flaky_floor
    t = _run_worker(eng)
    eng._final_q.put((np.ones(16000, dtype="float32"), "first turn", 0.0))   # raises -> dropped
    eng._final_q.put((np.ones(16000, dtype="float32"), "second turn", 0.0))  # must still dispatch
    eng._final_q.put(None)
    t.join(timeout=3.0)
    assert not t.is_alive()
    assert rec.finals == ["Second turn"]  # the bad turn dropped; the worker kept going


def test_enqueue_drops_oldest_on_overflow_preserving_capture_order():
    # Overflow must keep the NEWEST utterances in capture order (not reorder),
    # because the runtime supersede is newest-arrival-wins: a stale final arriving
    # after a newer one would wrongly cancel the newer turn. No worker draining ->
    # the bound is hit deterministically.
    rec = _Rec()
    eng = _engine(rec)
    eng._final_q = queue.Queue(maxsize=2)
    seg = np.ones(16000, dtype="float32")
    for raw in ("a", "b", "c", "d"):
        eng._enqueue_final(seg, raw, 0.0)
    drained = []
    while True:
        try:
            drained.append(eng._final_q.get_nowait()[1])
        except queue.Empty:
            break
    assert drained == ["c", "d"]  # two oldest dropped; order preserved, FIFO
    # each drop is observable in the run bundle (two oldest were dropped)
    overflow = [m for m, _ in rec.metrics if m == "second_pass_queue_overflow_dropped_final"]
    assert len(overflow) == 2


def test_enqueue_no_drop_when_room():
    eng = _engine()
    eng._final_q = queue.Queue(maxsize=8)
    seg = np.ones(16000, dtype="float32")
    for raw in ("a", "b", "c"):
        eng._enqueue_final(seg, raw, 0.0)
    drained = []
    while True:
        try:
            drained.append(eng._final_q.get_nowait()[1])
        except queue.Empty:
            break
    assert drained == ["a", "b", "c"]


def test_worker_exits_when_running_clears_without_sentinel():
    rec = _Rec()
    eng = _engine(rec)
    eng._final_q = queue.Queue(maxsize=8)
    t = _run_worker(eng)
    eng._running.clear()  # the loop guard -> it exits on the next 0.1s poll
    t.join(timeout=3.0)
    assert not t.is_alive()


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

    cfg = SherpaConfig.from_dict({
        "asr_final_backend": "sense_voice", "asr_final_model": model,
        "asr_final_tokens": tokens, "asr_final_language": "en", "asr_final_async": True,
    })
    rec_model = build_final_recognizer(cfg)
    if rec_model is None:
        pytest.skip("SenseVoice failed to build")

    rec = _Rec()
    eng = SherpaOnnxEngine(cfg)
    eng._cb = rec.callbacks()
    eng._final_recognizer = rec_model
    eng._final_q = queue.Queue(maxsize=8)

    samples = np.load("tests/fixture_audio/recorded_ocean_utterance.npy").astype("float32").reshape(-1)
    t = _run_worker(eng)
    eng._final_q.put((samples, "raw streaming text", 1.0))
    eng._final_q.put(None)
    t.join(timeout=15.0)
    assert not t.is_alive()
    # Exactly one terminal outcome (dispatched, or a floor/speaker drop) -- the
    # one-final-per-utterance invariant -- produced by the real decode, off-thread.
    drops = [m for m, _ in rec.metrics if m in ("echo_floor_rejected_final", "speaker_rejected_final")]
    assert len(rec.finals) + len(drops) == 1
    if rec.finals:
        assert rec.final_threads[0] != threading.get_ident()
