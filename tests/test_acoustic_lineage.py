from __future__ import annotations

import math
import threading
import time

import pytest

from always_on_agent.acoustic import (
    AcousticLineage,
    AcousticSource,
    AcousticSpan,
    EndpointReason,
)
from always_on_agent.events import AgentEvent, EventKind, Mode
from core.addressing import ACT, INGEST, ScriptedAddressingClassifier
from core.engine import (
    AcousticSignal,
    CommandDetection,
    EngineCallbacks,
    FinalTranscript,
    OwnerVerification,
    PartialTranscript,
    TranscriptAbort,
    TranscriptAbortReason,
)
from core.engines.scripted import ScriptedEngine
from core.engines._acoustic_turn import AcousticTurnTracker
from core.input_lineage import AcousticRevisionGate
from core.llm import EchoLLM
from core.metrics import ASR_FINAL, LLM_FIRST_TOKEN, TTS_REQUESTED
from core.runtime import VoiceRuntime


def _span(
    utterance_id: str = "turn-1",
    *,
    turn_index: int | None = None,
    speech_end_at: float | None = None,
    endpoint_reason: EndpointReason = EndpointReason.UNKNOWN,
) -> AcousticSpan:
    return AcousticSpan(
        stream_id="stream-1",
        utterance_id=utterance_id,
        turn_index=turn_index,
        capture_epoch=2,
        capture_generation=3,
        source=AcousticSource.LIVE_CAPTURE,
        speech_start_at=10.0,
        speech_end_at=speech_end_at,
        endpoint_committed_at=(
            speech_end_at + 0.2 if speech_end_at is not None else None
        ),
        emitted_at=(speech_end_at + 0.3 if speech_end_at is not None else 10.1),
        sample_rate_hz=16000,
        owned_sample_count=3200,
        endpoint_reason=endpoint_reason,
    )


def _lineage(
    utterance_id: str = "turn-1",
    *,
    turn_index: int | None = None,
    final: bool = False,
) -> AcousticLineage:
    return AcousticLineage.single(
        _span(
            utterance_id,
            turn_index=turn_index,
            speech_end_at=10.8 if final else None,
            endpoint_reason=(
                EndpointReason.VAD_SILENCE if final else EndpointReason.UNKNOWN
            ),
        )
    )


def test_legacy_agent_event_payloads_are_unchanged() -> None:
    assert AgentEvent.partial("hello").payload == {
        "text": "hello",
        "is_final": False,
    }
    assert AgentEvent.final("hello").payload == {
        "text": "hello",
        "is_final": True,
        "owner_verified": False,
        "origin": "unknown",
    }


def test_acoustic_lineage_round_trips_as_json_safe_metadata() -> None:
    lineage = _lineage(final=True)
    payload = lineage.to_payload()

    assert AcousticLineage.from_payload(payload) == lineage
    assert AcousticLineage.try_from_payload(payload) == lineage
    assert AcousticLineage.try_from_payload({"spans": "not-a-list"}) is None
    assert "text" not in repr(payload).lower()
    assert all(
        not hasattr(value, "shape")
        for span in payload["spans"]
        for value in span.values()
    )


@pytest.mark.parametrize(
    "changes, error",
    [
        ({"stream_id": ""}, ValueError),
        ({"turn_index": -1}, ValueError),
        ({"capture_epoch": -1}, ValueError),
        ({"capture_generation": True}, TypeError),
        ({"sample_rate_hz": 0}, ValueError),
        ({"speech_start_at": math.inf}, ValueError),
        ({"speech_end_at": 9.0}, ValueError),
        (
            {
                "speech_end_at": None,
                "endpoint_committed_at": 9.0,
            },
            ValueError,
        ),
        ({"emitted_at": 11.0}, ValueError),
    ],
)
def test_acoustic_span_rejects_invalid_metadata(changes, error) -> None:
    fields = {
        "stream_id": "stream-1",
        "utterance_id": "turn-1",
        "turn_index": 1,
        "capture_epoch": 0,
        "capture_generation": 0,
        "source": AcousticSource.LIVE_CAPTURE,
        "speech_start_at": 10.0,
        "speech_end_at": 11.0,
        "endpoint_committed_at": 11.1,
        "emitted_at": 11.2,
        "sample_rate_hz": 16000,
        "owned_sample_count": 3200,
        "endpoint_reason": EndpointReason.ASR,
    }
    fields.update(changes)
    with pytest.raises(error):
        AcousticSpan(**fields)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: PartialTranscript("hello", revision=1),
        lambda: FinalTranscript("hello", revision=1),
        lambda: AcousticSignal(revision=1),
        lambda: CommandDetection("stop", revision=1),
        lambda: AgentEvent.partial("hello", revision=1),
        lambda: AgentEvent.final("hello", revision=1),
    ],
)
def test_revision_without_acoustic_identity_is_rejected(factory) -> None:
    with pytest.raises(ValueError, match="requires acoustic"):
        factory()


def test_lineage_combine_replaces_same_turn_and_preserves_distinct_order() -> None:
    partial = _lineage()
    final = _lineage(final=True)
    second = _lineage("turn-2", final=True)

    combined = AcousticLineage.combine((partial, second, final))

    assert combined is not None
    assert [span.utterance_id for span in combined.spans] == [
        "turn-1",
        "turn-2",
    ]
    assert combined.spans[0].speech_end_at == 10.8


def test_span_identity_includes_capture_domain() -> None:
    first = _span(turn_index=1)
    replacement_domain = AcousticSpan(
        **{
            **first.__dict__,
            "capture_generation": first.capture_generation + 1,
        }
    )

    assert first.key != replacement_domain.key
    combined = AcousticLineage.combine(
        (
            AcousticLineage.single(first),
            AcousticLineage.single(replacement_domain),
        )
    )
    assert combined is not None
    assert len(combined.spans) == 2


def test_revision_gate_rejects_stale_duplicate_and_post_final_updates() -> None:
    gate = AcousticRevisionGate(max_turns=2)
    lineage = _lineage()

    assert gate.accept(lineage, 0, final=False)
    assert not gate.accept(lineage, 0, final=False)
    assert gate.accept(_lineage(final=True), 1, final=True)
    assert not gate.accept(_lineage(final=True), 2, final=True)
    assert not gate.accept(lineage, 3, final=False)
    assert gate.accept(_lineage("turn-2"), 0, final=False)


def test_revision_gate_rejects_late_final_after_newer_turn_partial() -> None:
    gate = AcousticRevisionGate()
    first = _lineage("turn-1", turn_index=1)
    second = _lineage("turn-2", turn_index=2)

    assert gate.accept(first, 0, final=False)
    assert gate.accept(second, 0, final=False)
    assert not gate.accept(
        _lineage("turn-1", turn_index=1, final=True),
        1,
        final=True,
    )


def test_stream_high_water_survives_per_turn_lru_eviction() -> None:
    gate = AcousticRevisionGate(max_turns=2)
    assert gate.accept(_lineage("turn-1", turn_index=1), 0, final=True)
    assert gate.accept(_lineage("turn-2", turn_index=2), 0, final=True)
    assert gate.accept(_lineage("turn-3", turn_index=3), 0, final=True)

    assert not gate.accept(
        _lineage("turn-1", turn_index=1, final=True),
        1,
        final=True,
    )


def test_acoustic_turn_tracker_rotates_domain_and_closes_one_revision() -> None:
    tracker = AcousticTurnTracker("stream-a")
    tracker.rotate_capture(capture_epoch=4, capture_generation=7)

    partial, partial_revision = tracker.partial(
        emitted_at=20.2,
        speech_start_at=20.0,
    )
    final, final_revision = tracker.close(
        speech_start_at=20.0,
        speech_end_at=20.8,
        endpoint_committed_at=21.0,
        emitted_at=21.1,
        sample_rate_hz=16000,
        owned_sample_count=17600,
        endpoint_reason=EndpointReason.VAD_SILENCE,
    )

    assert partial_revision == 0
    assert final_revision == 1
    assert partial.spans[0].key == final.spans[0].key
    assert partial.spans[0].turn_index == 1
    assert final.spans[0].capture_epoch == 4
    assert final.spans[0].capture_generation == 7
    assert not tracker.active

    tracker.rotate_capture(capture_epoch=5, capture_generation=8)
    replacement, replacement_revision = tracker.partial(
        emitted_at=30.0,
        speech_start_at=29.9,
    )
    assert replacement_revision == 0
    assert replacement.spans[0].key != final.spans[0].key
    assert replacement.spans[0].turn_index == 2
    assert replacement.spans[0].capture_generation == 8


def test_terminal_command_closes_pcm_turn_before_new_onset() -> None:
    tracker = AcousticTurnTracker("stream-a")
    tracker.rotate_capture(capture_epoch=1, capture_generation=1)

    command, command_revision = tracker.terminal(emitted_at=10.4)
    assert command_revision == 0
    assert command.spans[0].endpoint_reason is EndpointReason.COMMAND
    assert command.spans[0].endpoint_committed_at == 10.4
    assert not tracker.active

    next_turn, _ = tracker.partial(
        speech_start_at=21.0,
        emitted_at=21.1,
    )
    assert command.spans[0].key != next_turn.spans[0].key
    assert next_turn.spans[0].turn_index == 2


def test_tracker_abort_is_terminal_and_does_not_create_an_idle_turn() -> None:
    tracker = AcousticTurnTracker("stream-a")
    tracker.rotate_capture(capture_epoch=1, capture_generation=1)
    assert tracker.abort(emitted_at=1.0) is None
    tracker.partial(speech_start_at=2.0, emitted_at=2.1)

    aborted = tracker.abort(emitted_at=2.3)

    assert aborted is not None
    lineage, revision = aborted
    assert revision == 1
    assert lineage.spans[0].endpoint_reason is EndpointReason.ABORTED
    assert not tracker.active


def test_typed_transcript_callbacks_are_preferred_exactly_once() -> None:
    engine = ScriptedEngine()
    legacy: list[str] = []
    typed: list[object] = []
    engine.start(
        EngineCallbacks(
            on_partial=legacy.append,
            on_final=legacy.append,
            on_barge_in=lambda: legacy.append("barge"),
            on_command=legacy.append,
            on_partial_result=typed.append,
            on_final_result=typed.append,
            on_barge_in_result=typed.append,
            on_command_result=typed.append,
        )
    )
    lineage = _lineage()
    updates = [
        PartialTranscript("hel", lineage, 0),
        FinalTranscript("hello", acoustic=_lineage(final=True), revision=1),
        AcousticSignal(lineage, detected_at=10.1),
        CommandDetection("stop", lineage),
    ]

    engine.partial_result(updates[0])
    engine.final_result(updates[1])
    engine.barge_in_result(updates[2])
    engine.command_result(updates[3])

    assert typed == updates
    assert legacy == []


def test_typed_runtime_propagates_lineage_without_minting_authority() -> None:
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        EchoLLM(reply="ack"),
        start_mode=Mode.ASSISTANT,
    )
    runtime.start(run_bus=False)
    try:
        engine.partial_result(PartialTranscript("hello", _lineage(), 0))
        engine.final_result(
            FinalTranscript(
                "hello there",
                acoustic=_lineage(final=True),
                revision=1,
            )
        )
        assert runtime.wait_idle()

        finals = [
            event
            for event in runtime.supervisor.state.event_log
            if event.kind is EventKind.STT_FINAL
        ]
        assert len(finals) == 1
        assert AcousticLineage.from_payload(finals[0].payload["acoustic"]) == _lineage(
            final=True
        )
        assert finals[0].payload["revision"] == 1
        assert finals[0].payload["owner_verified"] is False
        assert finals[0].payload["origin"] == "unknown"

        # A duplicate native callback is rejected before it can allocate a new
        # runtime input generation or publish a second task.
        generation = runtime.supervisor.latest_arrival_generation
        engine.final_result(
            FinalTranscript(
                "duplicate",
                acoustic=_lineage(final=True),
                revision=1,
            )
        )
        assert runtime.wait_idle()
        assert runtime.supervisor.latest_arrival_generation == generation
    finally:
        runtime.stop()


def test_typed_command_terminalizes_lineage_before_control_side_effects() -> None:
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        EchoLLM(reply="ack"),
        start_mode=Mode.ASSISTANT,
    )
    runtime.start(run_bus=False)
    try:
        command = CommandDetection("stop", _lineage(), revision=0)
        engine.command_result(command)
        engine.command_result(command)
        engine.final_result(
            FinalTranscript(
                "stop",
                acoustic=_lineage(final=True),
                revision=1,
            )
        )
        assert runtime.wait_idle()

        stops = [
            event
            for event in runtime.supervisor.state.event_log
            if event.kind is EventKind.CONTROL_STOP
            and event.payload.get("reason") == "command"
        ]
        assert len(stops) == 1
        assert stops[0].payload["revision"] == 0
        assert not any(
            event.kind is EventKind.STT_FINAL
            for event in runtime.supervisor.state.event_log
        )
    finally:
        runtime.stop()


def test_duplicate_typed_barge_is_rejected_before_cancellation() -> None:
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        EchoLLM(reply="ack"),
        start_mode=Mode.ASSISTANT,
    )
    runtime.start(run_bus=False)
    try:
        barge = AcousticSignal(
            _lineage("turn-1", turn_index=1),
            revision=0,
            detected_at=10.0,
        )
        initial_input_epoch = runtime.supervisor.input_epoch

        engine.barge_in_result(barge)
        accepted_input_epoch = runtime.supervisor.input_epoch
        engine.barge_in_result(barge)

        assert accepted_input_epoch == initial_input_epoch + 1
        assert runtime.supervisor.input_epoch == accepted_input_epoch
        assert runtime.supervisor.state.spoken_outputs.count("[cancelled]") == 1
    finally:
        runtime.stop()


def test_mapped_command_acceptance_and_effect_are_one_atomic_boundary(
    monkeypatch,
) -> None:
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        EchoLLM(reply="ack"),
        start_mode=Mode.ASSISTANT,
    )
    runtime.start(run_bus=False)
    entered_effect = threading.Event()
    release_effect = threading.Event()
    partial_started = threading.Event()
    partial_done = threading.Event()
    original_invalidate = runtime._post_barge_response.invalidate

    def blocking_invalidate() -> None:
        entered_effect.set()
        assert release_effect.wait(timeout=1.0)
        original_invalidate()

    monkeypatch.setattr(
        runtime._post_barge_response,
        "invalidate",
        blocking_invalidate,
    )
    command_thread = threading.Thread(
        target=lambda: engine.command_result(
            CommandDetection(
                "stop",
                _lineage("turn-1", turn_index=1),
                revision=0,
            )
        )
    )

    def publish_newer_partial() -> None:
        partial_started.set()
        engine.partial_result(
            PartialTranscript(
                "newer turn",
                _lineage("turn-2", turn_index=2),
                revision=0,
            )
        )
        partial_done.set()

    partial_thread = threading.Thread(target=publish_newer_partial)
    try:
        command_thread.start()
        assert entered_effect.wait(timeout=1.0)
        partial_thread.start()
        assert partial_started.wait(timeout=1.0)
        assert not partial_done.wait(timeout=0.05)

        release_effect.set()
        command_thread.join(timeout=1.0)
        partial_thread.join(timeout=1.0)
        assert not command_thread.is_alive()
        assert not partial_thread.is_alive()
        assert partial_done.is_set()
    finally:
        release_effect.set()
        command_thread.join(timeout=1.0)
        partial_thread.join(timeout=1.0)
        runtime.stop()


def test_stale_rejected_final_cannot_invalidate_newer_post_barge_turn() -> None:
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        EchoLLM(reply="ack"),
        start_mode=Mode.ASSISTANT,
    )
    runtime.start(run_bus=False)
    try:
        input_epoch = runtime.supervisor.input_epoch
        runtime._post_barge_response.arm(input_epoch)
        engine.partial_result(
            PartialTranscript(
                "newer turn",
                _lineage("turn-2", turn_index=2),
                revision=0,
            )
        )

        engine.final_result(
            FinalTranscript(
                "stale rejected voice",
                owner_verification=OwnerVerification.REJECTED,
                origin="live_audio",
                acoustic=_lineage(
                    "turn-1",
                    turn_index=1,
                    final=True,
                ),
                revision=1,
            )
        )

        assert runtime._post_barge_response.is_armed(input_epoch)
        assert not any(
            event.kind is EventKind.STT_FINAL
            for event in runtime.supervisor.state.event_log
        )
    finally:
        runtime.stop()


def test_late_final_cannot_preempt_a_newer_stream_turn_partial() -> None:
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        EchoLLM(reply="ack"),
        start_mode=Mode.ASSISTANT,
    )
    runtime.start(run_bus=False)
    try:
        engine.partial_result(
            PartialTranscript(
                "old turn",
                _lineage("turn-1", turn_index=1),
                0,
            )
        )
        engine.partial_result(
            PartialTranscript(
                "new turn",
                _lineage("turn-2", turn_index=2),
                0,
            )
        )
        generation = runtime.supervisor.latest_arrival_generation
        engine.final_result(
            FinalTranscript(
                "old final",
                acoustic=_lineage(
                    "turn-1",
                    turn_index=1,
                    final=True,
                ),
                revision=1,
            )
        )
        assert runtime.wait_idle()
        assert runtime.supervisor.latest_arrival_generation == generation
        assert not any(
            event.kind is EventKind.STT_FINAL
            for event in runtime.supervisor.state.event_log
        )
    finally:
        runtime.stop()


def test_transcript_abort_retires_partial_fence_without_answering() -> None:
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        EchoLLM(reply="ack"),
        start_mode=Mode.ASSISTANT,
    )
    runtime.start(run_bus=False)
    try:
        partial_lineage = _lineage(turn_index=1)
        final_lineage = _lineage(turn_index=1, final=True)
        engine.partial_result(PartialTranscript("maybe speech", partial_lineage, 0))
        assert runtime._partial_fence_active

        engine.transcript_abort(
            TranscriptAbort(
                acoustic=final_lineage,
                revision=1,
                reason=TranscriptAbortReason.ECHO_REJECTED,
            )
        )
        assert runtime.wait_idle()

        assert not runtime._partial_fence_active
        aborted = [
            event
            for event in runtime.supervisor.state.event_log
            if event.kind is EventKind.STT_ABORTED
        ]
        assert len(aborted) == 1
        assert aborted[0].payload["reason"] == "echo_rejected"
        assert not any(
            event.kind is EventKind.STT_FINAL
            for event in runtime.supervisor.state.event_log
        )
    finally:
        runtime.stop()


def test_no_partial_abort_cannot_cancel_an_unrelated_valid_final() -> None:
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        EchoLLM(reply="ack"),
        start_mode=Mode.ASSISTANT,
    )
    runtime.start(run_bus=False)
    try:
        engine.final_result(
            FinalTranscript(
                "valid first turn",
                acoustic=_lineage(
                    "turn-1",
                    turn_index=1,
                    final=True,
                ),
                revision=0,
            )
        )
        engine.transcript_abort(
            TranscriptAbort(
                acoustic=_lineage(
                    "turn-2",
                    turn_index=2,
                    final=True,
                ),
                revision=0,
                reason=TranscriptAbortReason.INPUT_REJECTED,
            )
        )
        assert runtime.wait_idle(timeout=3.0)

        finals = [
            event
            for event in runtime.supervisor.state.event_log
            if event.kind is EventKind.STT_FINAL
        ]
        assert [event.payload["text"] for event in finals] == ["valid first turn"]
        assert engine.spoken == ["ack"]
    finally:
        runtime.stop()


def test_matching_partial_abort_restores_cancelled_valid_final() -> None:
    class _BlockingFirstAddressing:
        def __init__(self) -> None:
            self.started = threading.Event()
            self.release = threading.Event()
            self.calls = 0

        def classify(self, _text, *, recent=()):
            self.calls += 1
            if self.calls == 1:
                self.started.set()
                assert self.release.wait(timeout=2.0)
            return ACT

    addressing = _BlockingFirstAddressing()
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        EchoLLM(reply="ack"),
        start_mode=Mode.ASSISTANT,
        addressing=addressing,
    )
    runtime.start(run_bus=False)
    try:
        engine.final_result(
            FinalTranscript(
                "valid first turn",
                acoustic=_lineage(
                    "turn-1",
                    turn_index=1,
                    final=True,
                ),
                revision=0,
            )
        )
        assert addressing.started.wait(timeout=1.0)

        second_partial = _lineage("turn-2", turn_index=2)
        engine.partial_result(
            PartialTranscript("rejected continuation", second_partial, 0)
        )
        engine.transcript_abort(
            TranscriptAbort(
                acoustic=_lineage(
                    "turn-2",
                    turn_index=2,
                    final=True,
                ),
                revision=1,
                reason=TranscriptAbortReason.INPUT_REJECTED,
            )
        )
        addressing.release.set()
        assert runtime.wait_idle(timeout=3.0)

        finals = [
            event
            for event in runtime.supervisor.state.event_log
            if event.kind is EventKind.STT_FINAL
        ]
        assert [event.payload["text"] for event in finals] == ["valid first turn"]
        assert engine.spoken == ["ack"]
        assert addressing.calls == 2
    finally:
        addressing.release.set()
        runtime.stop()


def test_matching_partial_abort_restores_committed_unheard_final_once() -> None:
    addressing = ScriptedAddressingClassifier(default=ACT)
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        EchoLLM(reply="ack"),
        start_mode=Mode.ASSISTANT,
        addressing=addressing,
    )
    runtime.start(run_bus=False)
    try:
        engine.final_result(
            FinalTranscript(
                "valid first turn",
                acoustic=_lineage(
                    "turn-1",
                    turn_index=1,
                    final=True,
                ),
                revision=0,
            )
        )
        deadline = time.monotonic() + 1.0
        while not runtime._published_unheard_finals:
            if time.monotonic() >= deadline:
                pytest.fail("first final did not reach the event bus")
            time.sleep(0.005)

        [before] = runtime.metrics.records()
        assert ASR_FINAL in before.stamps
        second_partial = _lineage("turn-2", turn_index=2)
        engine.partial_result(
            PartialTranscript("rejected continuation", second_partial, 0)
        )
        engine.transcript_abort(
            TranscriptAbort(
                acoustic=_lineage(
                    "turn-2",
                    turn_index=2,
                    final=True,
                ),
                revision=1,
                reason=TranscriptAbortReason.INPUT_REJECTED,
            )
        )
        assert runtime.wait_idle(timeout=3.0)

        finals = [
            event
            for event in runtime.supervisor.state.event_log
            if event.kind is EventKind.STT_FINAL
        ]
        assert [event.payload["text"] for event in finals] == ["valid first turn"]
        assert engine.spoken == ["ack"]
        assert [text for text, _recent in addressing.calls] == ["valid first turn"]
        assert [item.text for item in runtime.memory.all() if "user" in item.tags] == [
            "valid first turn"
        ]
        [after] = runtime.metrics.records()
        assert {ASR_FINAL, LLM_FIRST_TOKEN, TTS_REQUESTED} <= set(after.stamps)
        assert runtime._published_unheard == {}
        assert runtime._published_unheard_finals == {}
    finally:
        runtime.stop()


def test_matching_partial_abort_reissues_a_claimed_final_once(
    monkeypatch,
) -> None:
    addressing = ScriptedAddressingClassifier(default=ACT)
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        EchoLLM(reply="ack"),
        start_mode=Mode.ASSISTANT,
        addressing=addressing,
    )
    first_analysis_started = threading.Event()
    release_first_analysis = threading.Event()
    original_observe = runtime.supervisor.analyzer.observe

    def blocking_observe(text, *, is_final):
        if (
            is_final
            and text == "valid first turn"
            and not first_analysis_started.is_set()
        ):
            first_analysis_started.set()
            assert release_first_analysis.wait(timeout=2.0)
        return original_observe(text, is_final=is_final)

    monkeypatch.setattr(
        runtime.supervisor.analyzer,
        "observe",
        blocking_observe,
    )
    runtime.start(run_bus=True)
    try:
        engine.final_result(
            FinalTranscript(
                "valid first turn",
                acoustic=_lineage(
                    "turn-1",
                    turn_index=1,
                    final=True,
                ),
                revision=0,
            )
        )
        assert first_analysis_started.wait(timeout=1.0)

        second_partial = _lineage("turn-2", turn_index=2)
        engine.partial_result(
            PartialTranscript("rejected continuation", second_partial, 0)
        )
        engine.transcript_abort(
            TranscriptAbort(
                acoustic=_lineage(
                    "turn-2",
                    turn_index=2,
                    final=True,
                ),
                revision=1,
                reason=TranscriptAbortReason.INPUT_REJECTED,
            )
        )
        release_first_analysis.set()
        assert runtime.wait_idle(timeout=3.0)

        assert runtime.supervisor.state.transcript_log == ["valid first turn"]
        assert engine.spoken == ["ack"]
        assert [text for text, _recent in addressing.calls] == ["valid first turn"]
        assert [item.text for item in runtime.memory.all() if "user" in item.tags] == [
            "valid first turn"
        ]
        [record] = runtime.metrics.records()
        assert {ASR_FINAL, LLM_FIRST_TOKEN, TTS_REQUESTED} <= set(record.stamps)
        assert runtime._published_unheard == {}
        assert runtime._published_unheard_finals == {}
    finally:
        release_first_analysis.set()
        runtime.stop()


def test_matching_partial_abort_restores_after_stale_original_resolves() -> None:
    addressing = ScriptedAddressingClassifier(default=ACT)
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        EchoLLM(reply="ack"),
        start_mode=Mode.ASSISTANT,
        addressing=addressing,
    )
    runtime.start(run_bus=False)
    try:
        engine.final_result(
            FinalTranscript(
                "valid first turn",
                acoustic=_lineage(
                    "turn-1",
                    turn_index=1,
                    final=True,
                ),
                revision=0,
            )
        )
        deadline = time.monotonic() + 1.0
        while not runtime._published_unheard_finals:
            if time.monotonic() >= deadline:
                pytest.fail("first final did not reach the event bus")
            time.sleep(0.005)

        second_partial = _lineage("turn-2", turn_index=2)
        engine.partial_result(
            PartialTranscript("rejected continuation", second_partial, 0)
        )
        # Resolve the now-stale original before the recognizer reports B's
        # abort. The partial fence, not the live bus map, must retain rollback
        # ownership across this gap.
        runtime.bus.drain()
        assert runtime._published_unheard_finals == {}
        assert runtime.supervisor.state.transcript_log == []
        assert engine.spoken == []

        engine.transcript_abort(
            TranscriptAbort(
                acoustic=_lineage(
                    "turn-2",
                    turn_index=2,
                    final=True,
                ),
                revision=1,
                reason=TranscriptAbortReason.INPUT_REJECTED,
            )
        )
        assert runtime.wait_idle(timeout=3.0)

        assert runtime.supervisor.state.transcript_log == ["valid first turn"]
        assert engine.spoken == ["ack"]
        assert [text for text, _recent in addressing.calls] == ["valid first turn"]
        assert [item.text for item in runtime.memory.all() if "user" in item.tags] == [
            "valid first turn"
        ]
        [record] = runtime.metrics.records()
        assert {ASR_FINAL, LLM_FIRST_TOKEN, TTS_REQUESTED} <= set(record.stamps)
    finally:
        runtime.stop()


@pytest.mark.parametrize(
    ("final_utterance", "answer_expected"),
    [("turn-1", True), ("turn-2", False)],
)
def test_post_barge_response_grant_requires_matching_acoustic_turn(
    final_utterance: str,
    answer_expected: bool,
) -> None:
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        EchoLLM(reply="ack"),
        start_mode=Mode.ASSISTANT,
        addressing=ScriptedAddressingClassifier(default=INGEST),
        unsure_acts=False,
    )
    runtime.start(run_bus=False)
    try:
        engine.barge_in_result(
            AcousticSignal(
                _lineage("turn-1", turn_index=1),
                detected_at=10.0,
            )
        )
        engine.final_result(
            FinalTranscript(
                "ambient-looking continuation",
                origin="live_audio",
                acoustic=_lineage(
                    final_utterance,
                    turn_index=1 if answer_expected else 2,
                    final=True,
                ),
                revision=1,
            )
        )
        assert runtime.wait_idle(timeout=3.0)

        finals = [
            event
            for event in runtime.supervisor.state.event_log
            if event.kind is EventKind.STT_FINAL
        ]
        if answer_expected:
            assert engine.spoken == ["ack"]
            assert len(finals) == 1
            assert finals[0].payload["metadata"]["post_barge_response_only"] is True
        else:
            assert engine.spoken == []
            assert finals == []
            assert any(
                item.text == "ambient-looking continuation" and "ingested" in item.tags
                for item in runtime.memory.all()
            )
    finally:
        runtime.stop()


def test_typed_abort_retires_matching_post_barge_grant() -> None:
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        EchoLLM(reply="ack"),
        start_mode=Mode.ASSISTANT,
    )
    runtime.start(run_bus=False)
    try:
        lineage = _lineage("turn-1", turn_index=1)
        engine.barge_in_result(AcousticSignal(lineage, detected_at=10.0))
        assert runtime._post_barge_response.is_armed(runtime.supervisor.input_epoch)

        engine.transcript_abort(
            TranscriptAbort(
                acoustic=_lineage(
                    "turn-1",
                    turn_index=1,
                    final=True,
                ),
                revision=1,
                reason=TranscriptAbortReason.EMPTY_FINAL,
            )
        )

        assert not runtime._post_barge_response.is_armed(runtime.supervisor.input_epoch)
    finally:
        runtime.stop()
