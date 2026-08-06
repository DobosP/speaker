"""Pure single-owner logical-turn lifecycle contracts."""

from __future__ import annotations

import pytest

from always_on_agent.logical_turn import (
    LogicalTurnAbortReason,
    LogicalTurnBoundary,
    LogicalTurnCoordinator,
    LogicalTurnPhase,
    LogicalTurnRejected,
    LogicalTurnTransition,
    LogicalTurnTransitionKind,
)


def _started(
    **limits: int,
) -> tuple[LogicalTurnCoordinator, int]:
    coordinator = LogicalTurnCoordinator(**limits)
    started = coordinator.start(native_epoch=7, evidence_sequence=1)
    return coordinator, started.turn_id


def _ready(
    coordinator: LogicalTurnCoordinator,
    turn_id: int,
    *,
    evidence_sequence: int = 2,
    reason: LogicalTurnBoundary = LogicalTurnBoundary.SEMANTIC_COMPLETE,
) -> LogicalTurnTransition:
    return coordinator.mark_inference_ready(
        turn_id,
        reason=reason,
        expected_evidence_sequence=evidence_sequence,
    )


def test_one_native_final_reaches_exactly_one_commit():
    coordinator, turn_id = _started()
    coordinator.observe_final(
        turn_id,
        native_epoch=7,
        native_sequence=1,
        evidence_sequence=2,
        text="Can you hear me?",
    )
    ready = _ready(coordinator, turn_id)
    committed = coordinator.commit(
        turn_id,
        readiness_token=ready.readiness_token,
    )

    assert ready.kind is LogicalTurnTransitionKind.INFERENCE_READY
    assert committed.transition.kind is LogicalTurnTransitionKind.COMMITTED
    assert committed.text == "Can you hear me?"
    assert [ready.sequence, committed.transition.sequence] == [2, 3]
    assert committed.transition.readiness_token == ready.readiness_token
    assert committed.transition.reason == ready.reason
    assert coordinator.snapshot().active_turn_id is None

    with pytest.raises(LogicalTurnRejected, match="no logical turn"):
        coordinator.commit(
            turn_id,
            readiness_token=ready.readiness_token,
        )


def test_native_reset_epochs_accumulate_inside_one_logical_turn():
    coordinator, turn_id = _started()
    coordinator.observe_final(
        turn_id,
        native_epoch=7,
        native_sequence=1,
        evidence_sequence=2,
        text="A long story about",
    )
    first_ready = _ready(
        coordinator,
        turn_id,
        reason=LogicalTurnBoundary.NATIVE_FINAL,
    )
    coordinator.resume_speech(
        turn_id,
        native_epoch=8,
        evidence_sequence=3,
    )
    assert coordinator.snapshot().phase is LogicalTurnPhase.COLLECTING
    coordinator.observe_final(
        turn_id,
        native_epoch=8,
        native_sequence=1,
        evidence_sequence=4,
        text="a lighthouse keeper",
    )
    ready = _ready(coordinator, turn_id, evidence_sequence=4)
    committed = coordinator.commit(
        turn_id,
        readiness_token=ready.readiness_token,
    )

    assert first_ready.readiness_token != ready.readiness_token
    assert ready.turn_id == turn_id
    assert ready.native_epoch_count == 2
    assert ready.native_final_count == 2
    assert committed.text == "A long story about a lighthouse keeper"


def test_higher_current_epoch_revision_replaces_instead_of_appending():
    coordinator, turn_id = _started()
    coordinator.observe_final(
        turn_id,
        native_epoch=7,
        native_sequence=1,
        evidence_sequence=2,
        text="find in my fault",
    )
    coordinator.observe_final(
        turn_id,
        native_epoch=7,
        native_sequence=2,
        evidence_sequence=3,
        text="find in my vault",
    )
    ready = _ready(
        coordinator,
        turn_id,
        evidence_sequence=3,
        reason=LogicalTurnBoundary.NATIVE_FINAL,
    )
    result = coordinator.commit(
        turn_id,
        readiness_token=ready.readiness_token,
    )

    assert result.text == "find in my vault"
    assert result.transition.native_epoch_count == 1
    assert result.transition.native_final_count == 2


def test_stale_readiness_cannot_commit_after_resumed_speech():
    coordinator, turn_id = _started()
    coordinator.observe_final(
        turn_id,
        native_epoch=7,
        native_sequence=1,
        evidence_sequence=2,
        text="tell me about",
    )
    first_ready = _ready(
        coordinator,
        turn_id,
        reason=LogicalTurnBoundary.NATIVE_FINAL,
    )

    coordinator.resume_speech(
        turn_id,
        native_epoch=8,
        evidence_sequence=3,
    )
    coordinator.observe_final(
        turn_id,
        native_epoch=8,
        native_sequence=1,
        evidence_sequence=4,
        text="Romanian history",
    )
    second_ready = _ready(
        coordinator,
        turn_id,
        evidence_sequence=4,
        reason=LogicalTurnBoundary.SILENCE_LIMIT,
    )
    before = coordinator.snapshot()

    with pytest.raises(LogicalTurnRejected, match="stale readiness"):
        coordinator.commit(
            turn_id,
            readiness_token=first_ready.readiness_token,
        )
    assert coordinator.snapshot() == before

    result = coordinator.commit(
        turn_id,
        readiness_token=second_ready.readiness_token,
    )
    assert result.text == "tell me about Romanian history"
    assert result.transition.reason == "silence_limit"


def test_duplicate_resume_and_final_after_readiness_fail_without_mutation():
    coordinator, turn_id = _started()
    coordinator.observe_final(
        turn_id,
        native_epoch=7,
        native_sequence=1,
        evidence_sequence=2,
        text="complete",
    )
    _ready(coordinator, turn_id)
    before = coordinator.snapshot()

    with pytest.raises(LogicalTurnRejected, match="stale or duplicate"):
        coordinator.resume_speech(
            turn_id,
            native_epoch=7,
            evidence_sequence=2,
        )
    assert coordinator.snapshot() == before

    with pytest.raises(LogicalTurnRejected, match="explicit resumed speech"):
        coordinator.observe_final(
            turn_id,
            native_epoch=7,
            native_sequence=2,
            evidence_sequence=3,
            text="late correction",
        )
    assert coordinator.snapshot() == before


def test_old_epoch_revision_is_rejected_after_new_epoch():
    coordinator, turn_id = _started()
    coordinator.observe_final(
        turn_id,
        native_epoch=7,
        native_sequence=1,
        evidence_sequence=2,
        text="first",
    )
    coordinator.resume_speech(
        turn_id,
        native_epoch=8,
        evidence_sequence=3,
    )
    coordinator.observe_final(
        turn_id,
        native_epoch=8,
        native_sequence=1,
        evidence_sequence=4,
        text="second",
    )
    before = coordinator.snapshot()

    with pytest.raises(LogicalTurnRejected, match="current epoch"):
        coordinator.observe_final(
            turn_id,
            native_epoch=7,
            native_sequence=2,
            evidence_sequence=5,
            text="late first",
        )
    assert coordinator.snapshot() == before


def test_stale_boundary_and_native_sequences_fail_without_mutation():
    coordinator, turn_id = _started()
    coordinator.observe_final(
        turn_id,
        native_epoch=7,
        native_sequence=3,
        evidence_sequence=2,
        text="latest",
    )
    before = coordinator.snapshot()

    with pytest.raises(LogicalTurnRejected, match="stale or duplicate native"):
        coordinator.observe_final(
            turn_id,
            native_epoch=7,
            native_sequence=3,
            evidence_sequence=3,
            text="duplicate",
        )
    with pytest.raises(LogicalTurnRejected, match="readiness evidence is stale"):
        coordinator.mark_inference_ready(
            turn_id,
            reason=LogicalTurnBoundary.NATIVE_FINAL,
            expected_evidence_sequence=1,
        )
    with pytest.raises(LogicalTurnRejected, match="foreign"):
        coordinator.observe_final(
            turn_id + 1,
            native_epoch=7,
            native_sequence=4,
            evidence_sequence=3,
            text="foreign",
        )

    assert coordinator.snapshot() == before


@pytest.mark.parametrize(
    "reason",
    [
        LogicalTurnAbortReason.TRANSCRIPT_ABORT,
        LogicalTurnAbortReason.CONTROL,
        LogicalTurnAbortReason.STOP,
        LogicalTurnAbortReason.SHUTDOWN,
    ],
)
def test_abort_is_terminal_and_next_id_is_monotonic(reason):
    coordinator, turn_id = _started()
    terminal = coordinator.abort(
        turn_id,
        reason=reason,
        evidence_sequence=2,
    )

    assert terminal.kind is LogicalTurnTransitionKind.ABORTED
    assert terminal.reason == reason.value
    assert terminal.readiness_token is None
    assert coordinator.snapshot().last_terminal_turn_id == turn_id
    with pytest.raises(LogicalTurnRejected, match="no logical turn"):
        coordinator.observe_final(
            turn_id,
            native_epoch=7,
            native_sequence=1,
            evidence_sequence=3,
            text="late",
        )

    next_turn = coordinator.start(native_epoch=0, evidence_sequence=1)
    assert next_turn.turn_id == turn_id + 1


def test_hard_limit_is_explicit_and_bound_to_commit():
    coordinator, turn_id = _started()
    coordinator.observe_final(
        turn_id,
        native_epoch=7,
        native_sequence=1,
        evidence_sequence=2,
        text="bounded text",
    )
    ready = _ready(
        coordinator,
        turn_id,
        reason=LogicalTurnBoundary.HARD_LIMIT,
    )
    committed = coordinator.commit(
        turn_id,
        readiness_token=ready.readiness_token,
    )

    assert ready.reason == committed.transition.reason == "hard_limit"


def test_transcript_is_absent_from_transitions_snapshots_and_repr():
    coordinator, turn_id = _started()
    secret = "private spoken words"
    coordinator.observe_final(
        turn_id,
        native_epoch=7,
        native_sequence=1,
        evidence_sequence=2,
        text=secret,
    )
    ready = _ready(
        coordinator,
        turn_id,
        reason=LogicalTurnBoundary.NATIVE_FINAL,
    )
    committed = coordinator.commit(
        turn_id,
        readiness_token=ready.readiness_token,
    )

    assert secret not in repr(ready)
    assert secret not in repr(coordinator.snapshot())
    assert secret not in repr(committed)
    assert committed.text == secret


def test_malformed_or_empty_evidence_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        LogicalTurnCoordinator(max_native_epochs=0)
    coordinator = LogicalTurnCoordinator()
    with pytest.raises(TypeError, match="integer"):
        coordinator.start(native_epoch=True, evidence_sequence=1)
    with pytest.raises(ValueError, match="non-negative"):
        coordinator.start(native_epoch=-1, evidence_sequence=1)

    started = coordinator.start(native_epoch=0, evidence_sequence=1)
    with pytest.raises(LogicalTurnRejected, match="already active"):
        coordinator.start(native_epoch=1, evidence_sequence=2)
    with pytest.raises(LogicalTurnRejected, match="empty"):
        coordinator.observe_final(
            started.turn_id,
            native_epoch=0,
            native_sequence=1,
            evidence_sequence=2,
            text="  ",
        )
    with pytest.raises(LogicalTurnRejected, match="requires a native final"):
        coordinator.mark_inference_ready(
            started.turn_id,
            reason=LogicalTurnBoundary.NATIVE_FINAL,
            expected_evidence_sequence=1,
        )


def test_epoch_final_and_joined_text_bounds_fail_atomically():
    coordinator = LogicalTurnCoordinator(
        max_native_epochs=2,
        max_native_finals=2,
        max_text_chars=9,
    )
    started = coordinator.start(native_epoch=0, evidence_sequence=1)
    coordinator.observe_final(
        started.turn_id,
        native_epoch=0,
        native_sequence=1,
        evidence_sequence=2,
        text="four",
    )
    coordinator.resume_speech(
        started.turn_id,
        native_epoch=1,
        evidence_sequence=3,
    )
    before = coordinator.snapshot()

    with pytest.raises(LogicalTurnRejected, match="epoch limit"):
        coordinator.resume_speech(
            started.turn_id,
            native_epoch=2,
            evidence_sequence=4,
        )
    assert coordinator.snapshot() == before

    coordinator.observe_final(
        started.turn_id,
        native_epoch=1,
        native_sequence=1,
        evidence_sequence=4,
        text="1234",
    )
    before = coordinator.snapshot()
    with pytest.raises(LogicalTurnRejected, match="final limit"):
        coordinator.observe_final(
            started.turn_id,
            native_epoch=1,
            native_sequence=2,
            evidence_sequence=5,
            text="1234",
        )
    assert coordinator.snapshot() == before

    text_limited, limited_id = _started(max_text_chars=8)
    text_limited.observe_final(
        limited_id,
        native_epoch=7,
        native_sequence=1,
        evidence_sequence=2,
        text="four",
    )
    text_limited.resume_speech(
        limited_id,
        native_epoch=8,
        evidence_sequence=3,
    )
    before = text_limited.snapshot()
    with pytest.raises(LogicalTurnRejected, match="text limit"):
        text_limited.observe_final(
            limited_id,
            native_epoch=8,
            native_sequence=1,
            evidence_sequence=4,
            text="1234",
        )
    assert text_limited.snapshot() == before


def test_public_transition_rejects_forged_shape():
    with pytest.raises(ValueError, match="readiness token"):
        LogicalTurnTransition(
            kind=LogicalTurnTransitionKind.STARTED,
            turn_id=1,
            sequence=1,
            native_epoch_count=1,
            native_final_count=0,
            evidence_sequence=1,
            readiness_token=1,
            reason="speech",
        )
