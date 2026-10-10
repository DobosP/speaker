"""Exact final-work cancellation between model calls, without devices/models."""

from __future__ import annotations

from itertools import product
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from always_on_agent.acoustic import AcousticLineage, AcousticSpan
from core.engine import EngineCallbacks
from core.engines.sherpa import (
    _FinalAsrCancelled,
    _resolve_final_transcript,
    SherpaConfig,
    SherpaOnnxEngine,
)
from core.realtime_media_stage import CaptureScope, RealtimeMediaStage


_PHASES = ("punctuation", "create", "accept", "decode", "verifier")


class _Models:
    def __init__(self, *, cancel_at=None, cancel=None, error=False):
        self.cancel_at = cancel_at
        self.cancel = threading.Event() if cancel is None else cancel
        self.error = error
        self.calls = []
        self.pcm = []
        self.streams_released = []
        models = self

        class Stream:
            result = SimpleNamespace(text="public synthetic phrase")

            def accept_waveform(self, rate, pcm):
                models.pcm.append(("offline", pcm, rate))
                models.visit("accept")

            def __del__(self):
                models.streams_released.append(threading.get_ident())

        class Offline:
            def create_stream(self):
                models.visit("create")
                return Stream()

            def decode_stream(self, stream):
                models.visit("decode")

        class Punctuation:
            def add_punctuation(self, text):
                models.visit("punctuation")
                return text

        class Verifier:
            def transcribe(self, pcm, rate):
                models.pcm.append(("verifier", pcm, rate))
                models.visit("verifier")
                return SimpleNamespace(text="public synthetic phrase")

        self.offline = Offline()
        self.punctuation = Punctuation()
        self.verifier = Verifier()

    def visit(self, phase):
        self.calls.append(phase)
        if phase == self.cancel_at:
            self.cancel.set()
            if self.error:
                raise RuntimeError("synthetic failure after revocation")


def _engine(models):
    engine = SherpaOnnxEngine(SherpaConfig(asr_final_min_sec=0.1))
    engine._punct = models.punctuation
    engine._final_recognizer = models.offline
    engine._final_verifier = models.verifier
    engine._capture_callback_context.media_cancel_event = models.cancel
    return engine


@pytest.mark.parametrize("phase", _PHASES)
@pytest.mark.parametrize("error", [False, True])
def test_revocation_at_each_model_boundary_admits_no_later_model_or_fallback(
    phase, error, caplog
):
    models = _Models(cancel_at=phase, error=error)
    with pytest.raises(_FinalAsrCancelled):
        _resolve_final_transcript(
            SherpaConfig(asr_final_min_sec=0.1),
            models.offline,
            models.punctuation,
            np.ones(16000, np.float32),
            "public synthetic phrase",
            final_verifier=models.verifier,
            is_current=lambda: not models.cancel.is_set(),
        )
    assert models.calls == list(_PHASES[: _PHASES.index(phase) + 1])
    assert "using raw text" not in caplog.text
    assert "using streaming final" not in caplog.text
    assert "using established ASR final" not in caplog.text
    if phase in {"accept", "decode", "verifier"}:
        assert models.streams_released == [threading.get_ident()]


@pytest.mark.parametrize("phase", _PHASES)
@pytest.mark.parametrize("error", [False, True])
def test_live_selection_cancellation_keeps_verifier_health_and_aborts_observation(
    phase, error
):
    models = _Models(cancel_at=phase, error=error)
    engine = _engine(models)
    finals, metrics, aborted = [], [], []
    engine._cb = EngineCallbacks(
        on_final=finals.append,
        on_metric=lambda *args, **kwargs: metrics.append(args),
    )
    engine._diagnostic_final_aborted = lambda *args: aborted.append(args)
    engine._finalize_and_dispatch(
        np.ones(16000, np.float32),
        "public synthetic phrase",
        123.0,
        revision=7,
    )
    assert models.calls == list(_PHASES[: _PHASES.index(phase) + 1])
    assert finals == metrics == []
    assert aborted == [(None, 7, "stale_fenced")]
    assert engine._final_verifier is models.verifier


def test_already_revoked_exact_work_enters_no_punctuation_or_model():
    models = _Models()
    models.cancel.set()
    engine = _engine(models)
    engine._finalize_and_dispatch(np.ones(16000, np.float32), "public phrase", 0.0)
    assert models.calls == []
    assert engine._final_verifier is models.verifier


@pytest.mark.parametrize(
    "backend,raw,offline,verifier,short,allow_empty",
    tuple(
        product(
            ("sense_voice", "nemo_transducer"),
            ("public synthetic phrase", "stop", ""),
            ("public synthetic phrase", ""),
            ("public synthetic phrase", ""),
            (False, True),
            (False, True),
        )
    ),
)
def test_healthy_explicit_predicate_preserves_exact_decision_and_pcm(
    backend,
    raw,
    offline,
    verifier,
    short,
    allow_empty,
):
    config = SherpaConfig(asr_final_backend=backend, asr_final_min_sec=0.35)
    pcm = np.arange(1600 if short else 16000, dtype=np.float32)

    def resolve(predicate):
        models = _Models()
        models.offline.create_stream = lambda: SimpleNamespace(
            accept_waveform=lambda rate, samples: models.pcm.append(
                ("offline", samples, rate)
            ),
            result=SimpleNamespace(text=offline),
        )
        models.verifier.transcribe = lambda samples, rate: (
            models.pcm.append(("verifier", samples, rate))
            or SimpleNamespace(text=verifier)
        )
        decision = _resolve_final_transcript(
            config,
            models.offline,
            models.punctuation,
            pcm,
            raw,
            final_verifier=models.verifier,
            allow_empty_streaming=allow_empty,
            is_current=predicate,
        )
        return decision, models

    original, before = resolve(None)
    guarded, after = resolve(lambda: True)
    assert guarded == original
    assert before.calls == after.calls
    assert [p[0] for p in before.pcm] == [p[0] for p in after.pcm]
    assert all(samples is pcm and rate == 16000 for _, samples, rate in after.pcm)


@pytest.mark.parametrize("phase", _PHASES)
@pytest.mark.parametrize("error", [False, True])
def test_async_retirement_waits_entered_call_then_frees_worker_for_successor(
    phase, error
):
    models = _Models()
    entered, release, successor = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    original_visit = models.visit
    blocked = False

    def visit(at):
        nonlocal blocked
        original_visit(at)
        if at == phase and not blocked:
            blocked = True
            entered.set()
            assert release.wait(3.0)
            if error:
                raise RuntimeError("synthetic retired call failure")

    models.visit = visit
    engine = _engine(models)
    # The worker's exact stage item installs its own cancel event/thread context.
    stage = engine._final_stage = RealtimeMediaStage(max_queued=2)
    finals = []
    aborted = []
    engine._cb = EngineCallbacks(
        on_final=lambda text: (finals.append(text), successor.set())
    )
    engine._final_above_floor = lambda pcm: True
    engine._diagnostic_final_aborted = lambda *args: aborted.append(args)
    engine._running.set()
    epoch = engine._capture_epoch
    old_acoustic = AcousticLineage.single(
        AcousticSpan(
            stream_id="synthetic-stream",
            utterance_id="old",
            turn_index=1,
        )
    )
    new_acoustic = AcousticLineage.single(
        AcousticSpan(
            stream_id="synthetic-stream",
            utterance_id="new",
            turn_index=2,
        )
    )
    worker = threading.Thread(target=engine._final_worker, args=(stage,), daemon=True)
    engine._enqueue_final(
        np.ones(16000, np.float32),
        "public synthetic phrase",
        1.0,
        capture_epoch=epoch,
        capture_generation=2,
        acoustic=old_acoustic,
        revision=1,
    )
    worker.start()
    try:
        assert entered.wait(3.0)
        retired = stage.retire_scope(CaptureScope(epoch, 2))
        assert retired.in_flight_cancelled == 1 and retired.queued_released == 0
        engine._enqueue_final(
            np.ones(16000, np.float32),
            "public synthetic phrase",
            2.0,
            capture_epoch=epoch,
            capture_generation=3,
            acoustic=new_acoustic,
            revision=2,
        )
        snapshot = stage.snapshot()
        assert snapshot.in_flight == snapshot.queued == 1
        assert not successor.is_set()
        assert models.calls == list(_PHASES[: _PHASES.index(phase) + 1])
        release.set()
        assert successor.wait(3.0)
        assert stage.wait_idle(3.0)
        assert stage.snapshot().finished == 2
        assert models.calls == list(_PHASES[: _PHASES.index(phase) + 1]) + list(_PHASES)
        assert finals == ["public synthetic phrase"]
        assert aborted == [(old_acoustic, 1, "stale_fenced")]
        assert engine._final_verifier is models.verifier
        expected_releases = (
            1 if phase == "punctuation" or (phase == "create" and error) else 2
        )
        assert len(models.streams_released) == expected_releases
        assert all(owner == worker.ident for owner in models.streams_released)
    finally:
        release.set()
        stage.close()
        engine._running.clear()
        worker.join(3.0)
    assert not worker.is_alive()
    assert stage.snapshot().in_flight == 0


def test_foreign_scope_retirement_does_not_cancel_current_final():
    models = _Models()
    engine = _engine(models)
    stage = RealtimeMediaStage(max_queued=1)
    current = stage.offer_nowait(CaptureScope(engine._capture_epoch, 3), None).item
    assert stage.take(timeout=0.0) is current
    foreign = stage.retire_scope(CaptureScope(engine._capture_epoch, 2))
    assert foreign.in_flight_cancelled == foreign.queued_released == 0
    engine._capture_callback_context.media_cancel_event = current.cancel_event
    finals = []
    engine._cb = EngineCallbacks(on_final=finals.append)
    engine._final_above_floor = lambda pcm: True
    engine._finalize_and_dispatch(
        np.ones(16000, np.float32), "public synthetic phrase", time.perf_counter()
    )
    assert not current.cancelled
    assert models.calls == list(_PHASES)
    assert finals == ["public synthetic phrase"]
    assert stage.finish(current)
    assert stage.wait_idle(0.0)
