"""Exact runtime receipt observations; no models, audio devices or raw audio."""

from __future__ import annotations

import pytest

from always_on_agent.events import AgentEvent, EventKind
from core.engine import PlaybackCapabilities, PlaybackOutcome, PlaybackReceipt
from core.llm import EchoLLM
from core.metrics import (
    ASR_FINAL,
    LLM_FIRST_TOKEN,
    LLM_REQUESTED,
    MetricsRecorder,
    TTS_ADMITTED,
    TTS_FIRST_AUDIO,
    TTS_RENDER_START_OBSERVED,
    TTS_TEXT_READY,
)
from core.runtime import VoiceRuntime
from tests.test_engine_playback_receipts import _LegacyEngine
from tests.test_playback_receipts import _ControlledReceiptEngine, _task


def _runtime(engine=None):
    engine = engine or _ControlledReceiptEngine()
    runtime = VoiceRuntime(engine, EchoLLM())
    now = [10.0]
    runtime.metrics = MetricsRecorder(clock=lambda: now[0])
    runtime.start(run_bus=False)
    runtime.metrics.mark(ASR_FINAL)
    token = runtime.metrics.current_turn_token()
    now[0] = 10.1
    runtime.metrics.mark(LLM_REQUESTED, turn_token=token)
    now[0] = 10.2
    runtime.metrics.mark(LLM_FIRST_TOKEN, turn_token=token)
    now[0] = 10.3
    runtime.metrics.mark(TTS_TEXT_READY, turn_token=token)
    return runtime, engine, now, _task(runtime), token


def _enqueue(runtime, task, token, **extra):
    payload = {
        "task_id": task.task_id,
        "epoch": task.speech_epoch,
        "text": "Public answer.",
        "streaming": True,
        "input_generation": 0,
        "metrics_turn_token": token,
        **task.identity.to_payload(),
        **extra,
    }
    runtime.bus.publish(AgentEvent(EventKind.TTS_REQUEST, payload))
    runtime.bus.drain()


def test_exact_onset_observation_is_separate_from_global_audio_and_terminal():
    runtime, engine, now, task, token = _runtime()
    try:
        now[0] = 10.5
        _enqueue(runtime, task, token)
        assert len(runtime._playback_metric_turns) == 1
        assert runtime.metrics.records()[0].stamps[TTS_ADMITTED] == 10.5
        now[0] = 10.6
        engine._cb.on_metric(TTS_FIRST_AUDIO)
        runtime._on_playback_started("foreign-fragment")
        assert TTS_RENDER_START_OBSERVED not in runtime.metrics.records()[0].stamps
        now[0] = 10.8
        engine.start_fragment(0)
        assert runtime._playback_metric_turns == {}
        assert (
            runtime.metrics.stage_breakdowns()[0]["tts_admission_to_render_observed"]
            == 0.3
        )
        assert runtime.metrics.records()[0].stamps[TTS_FIRST_AUDIO] == 10.6
        engine.terminal(0, PlaybackOutcome.COMPLETED, safe_text_prefix="Public answer.")
        runtime._on_playback_started(engine.fragments[0].speech.fragment_id)
        assert runtime.metrics.records()[0].stamps[TTS_RENDER_START_OBSERVED] == 10.8
    finally:
        runtime.stop()


def test_old_receipt_after_new_metric_turn_cannot_fill_successor_observation():
    runtime, engine, now, task, token = _runtime()
    try:
        _enqueue(runtime, task, token)
        now[0] = 11.0
        runtime.metrics.mark(ASR_FINAL)
        engine.start_fragment(0)
        engine.terminal(0, PlaybackOutcome.COMPLETED, safe_text_prefix="Public answer.")
        assert runtime._playback_metric_turns == {}
        assert all(
            TTS_RENDER_START_OBSERVED not in record.stamps
            for record in runtime.metrics.records()
        )
    finally:
        runtime.stop()


@pytest.mark.parametrize(
    "outcome",
    [PlaybackOutcome.DROPPED, PlaybackOutcome.FAILED, PlaybackOutcome.INTERRUPTED],
)
def test_terminal_before_start_retires_observation_without_inventing_onset(outcome):
    runtime, engine, _now, task, token = _runtime()
    try:
        _enqueue(runtime, task, token)
        engine.terminal(0, outcome)
        runtime._on_playback_started(engine.fragments[0].speech.fragment_id)
        assert runtime._playback_metric_turns == {}
        assert TTS_RENDER_START_OBSERVED not in runtime.metrics.records()[0].stamps
    finally:
        runtime.stop()


def test_global_interruption_forgets_map_before_late_started_callback():
    runtime, engine, _now, task, token = _runtime()
    try:
        _enqueue(runtime, task, token)
        runtime._interrupt_playback_history()
        engine.start_fragment(0)
        assert runtime._playback_metric_turns == {}
        assert TTS_RENDER_START_OBSERVED not in runtime.metrics.records()[0].stamps
    finally:
        runtime.stop()


@pytest.mark.parametrize("token_value", [None, True, 0, "1"])
def test_missing_or_invalid_metric_binding_does_not_infer_current_turn(token_value):
    runtime, engine, _now, task, _token = _runtime()
    try:
        _enqueue(runtime, task, token_value)
        engine.start_fragment(0)
        assert runtime._playback_metric_turns == {}
        assert TTS_ADMITTED not in runtime.metrics.records()[0].stamps
        assert TTS_RENDER_START_OBSERVED not in runtime.metrics.records()[0].stamps
    finally:
        runtime.stop()


@pytest.mark.parametrize("latency_ack", [False, True])
def test_auxiliary_output_cannot_fill_normal_reply_stage_observations(latency_ack):
    runtime, engine, _now, _task_value, token = _runtime()
    try:
        runtime._publish_auxiliary_event(
            AgentEvent(
                EventKind.TTS_REQUEST,
                {
                    "task_id": "public-auxiliary",
                    "text": "Public acknowledgement.",
                    "latency_ack": latency_ack,
                    "metrics_turn_token": token,
                },
            )
        )
        runtime.bus.drain()
        engine.start_fragment(0)
        assert runtime._playback_metric_turns == {}
        assert TTS_ADMITTED not in runtime.metrics.records()[0].stamps
        assert TTS_RENDER_START_OBSERVED not in runtime.metrics.records()[0].stamps
    finally:
        runtime.stop()


@pytest.mark.parametrize(
    "engine",
    [
        _LegacyEngine(),
        type(
            "NoExactOnset",
            (_ControlledReceiptEngine,),
            {
                "_CAPS": PlaybackCapabilities(
                    tracked_terminal=True, exact_started=False
                ),
            },
        )(),
    ],
)
def test_untracked_or_nonexact_output_leaves_new_output_stages_unknown(engine):
    runtime, engine, _now, task, token = _runtime(engine)
    try:
        _enqueue(runtime, task, token)
        if isinstance(engine, _ControlledReceiptEngine):
            engine.start_fragment(0)
        assert runtime._playback_metric_turns == {}
        assert TTS_ADMITTED not in runtime.metrics.records()[0].stamps
        assert TTS_RENDER_START_OBSERVED not in runtime.metrics.records()[0].stamps
    finally:
        runtime.stop()


def test_map_is_bounded_by_receipt_capacity_and_cleared_by_shutdown():
    runtime, engine, _now, task, token = _runtime()
    try:
        for _ in range(65):
            _enqueue(runtime, task, token)
        assert len(engine.fragments) == 64
        assert len(runtime._playback_metric_turns) == 64
        runtime._on_playback_terminal(
            PlaybackReceipt("foreign", PlaybackOutcome.FAILED)
        )
        assert len(runtime._playback_metric_turns) == 64
    finally:
        runtime.stop()
    assert runtime._playback_metric_turns == {}


def test_failed_sink_handoff_removes_observation_binding(monkeypatch):
    runtime, engine, _now, task, token = _runtime()

    def failed(*args, **kwargs):
        raise RuntimeError("public injected handoff failure")

    monkeypatch.setattr(engine, "speak_tracked", failed)
    try:
        _enqueue(runtime, task, token)
        assert runtime._playback_metric_turns == {}
        assert TTS_RENDER_START_OBSERVED not in runtime.metrics.records()[0].stamps
    finally:
        runtime.stop()


@pytest.mark.parametrize("kind", [EventKind.TASK_CANCELLED, EventKind.TASK_FAILED])
def test_exact_task_terminal_retires_pending_observation_before_late_start(kind):
    runtime, engine, _now, task, token = _runtime()
    try:
        _enqueue(runtime, task, token)
        runtime.bus.publish(
            AgentEvent(
                kind,
                {
                    "task_id": task.task_id,
                    "error": "public failure",
                    "epoch": task.speech_epoch,
                    **task.identity.to_payload(),
                },
            )
        )
        runtime.bus.drain()
        assert runtime._playback_metric_turns == {}
        engine.start_fragment(0)
        assert TTS_RENDER_START_OBSERVED not in runtime.metrics.records()[0].stamps
    finally:
        runtime.stop()


def test_foreign_task_terminal_cannot_remove_an_owned_observation():
    runtime, engine, _now, task, token = _runtime()
    try:
        _enqueue(runtime, task, token)
        other = _task(runtime)
        runtime.bus.publish(
            AgentEvent(
                EventKind.TASK_FAILED,
                {
                    "task_id": other.task_id,
                    "error": "public failure",
                    "epoch": other.speech_epoch,
                    **other.identity.to_payload(),
                },
            )
        )
        runtime.bus.drain()
        assert len(runtime._playback_metric_turns) == 1
        engine.start_fragment(0)
        assert TTS_RENDER_START_OBSERVED in runtime.metrics.records()[0].stamps
    finally:
        runtime.stop()


def test_registered_fragment_rollback_cannot_leave_an_observation_orphan(monkeypatch):
    runtime, engine, _now, task, token = _runtime()

    def failed(**_kwargs):
        assert runtime._playback_metric_turns
        raise RuntimeError("public injected observation failure")

    monkeypatch.setattr(runtime, "_observe_playback_diagnostic", failed)
    try:
        _enqueue(runtime, task, token)
        assert runtime._playback_metric_turns == {}
        assert runtime.supervisor.session_actor.active_playback_count == 0
        assert engine.fragments == []
    finally:
        runtime.stop()
