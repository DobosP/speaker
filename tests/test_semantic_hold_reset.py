"""Pure reset-and-accumulate endpoint regressions (no model or audio device)."""
from __future__ import annotations

import pytest

from core.engines._asr_segment import ASRSegment
from core.engines._semantic_hold import (
    HeldASRText,
    resolve_semantic_endpoint_boundary,
)


def test_disabled_text_state_is_exact_passthrough_and_inert() -> None:
    state = HeldASRText(enabled=False)
    native = "  NATIVE TEXT  "

    assert state.compose(native, capture_epoch=9, capture_generation=4) is native
    assert state.active is False
    assert state.reset_count == 0


def test_held_native_pieces_compose_once_without_exposing_text_in_repr() -> None:
    state = HeldASRText(enabled=True)
    state.bind(capture_epoch=2, capture_generation=7)

    assert state.hold(
        "  PRIVATE FIRST  ", capture_epoch=2, capture_generation=7
    ) == 1
    assert state.compose(
        " PRIVATE SECOND ", capture_epoch=2, capture_generation=7
    ) == "PRIVATE FIRST PRIVATE SECOND"
    assert "PRIVATE" not in repr(state)
    assert state.hold(
        " PRIVATE SECOND ", capture_epoch=2, capture_generation=7
    ) == 2

    assert state.consume(
        " THIRD ", capture_epoch=2, capture_generation=7
    ) == "PRIVATE FIRST PRIVATE SECOND THIRD"
    assert state.active is False
    assert state.consume("", capture_epoch=2, capture_generation=7) == ""


def test_scope_rotation_and_segment_reset_clear_held_text_without_leak() -> None:
    segment = ASRSegment(
        sample_rate=100,
        pre_roll_sec=0.2,
        max_utterance_sec=2.0,
        vad_available=True,
        block_sec=0.1,
        semantic_hold_reset_enabled=True,
    )
    segment.bind_capture_scope(capture_epoch=3, capture_generation=5)
    segment.hold_native_endpoint(
        "OLD PRIVATE", capture_epoch=3, capture_generation=5
    )

    with pytest.raises(RuntimeError, match="outside its capture scope"):
        segment.logical_text("NEW", capture_epoch=3, capture_generation=6)

    segment.bind_capture_scope(capture_epoch=3, capture_generation=6)
    assert segment.logical_text(
        "NEW", capture_epoch=3, capture_generation=6
    ) == "NEW"
    segment.hold_native_endpoint("NEW", capture_epoch=3, capture_generation=6)
    segment.reset()
    assert segment.logical_text(
        "AFTER", capture_epoch=3, capture_generation=6
    ) == "AFTER"


def test_total_quiet_backstop_uses_total_silence_after_native_reset() -> None:
    boundary = resolve_semantic_endpoint_boundary(
        enabled=True,
        native_acoustic_endpoint=False,
        semantic_hold_active=True,
        vad_active=False,
        trailing_silence_sec=1.6,
        utterance_elapsed_sec=3.0,
        max_silence_sec=1.6,
        rule3_min_utterance_length_sec=20.0,
    )

    assert boundary.max_silence_backstop is True
    assert boundary.hard_boundary is False
    assert boundary.effective_acoustic_endpoint is True


def test_resumed_vad_disarms_old_quiet_deadline_and_rule3_rearms_hard_boundary() -> None:
    resumed = resolve_semantic_endpoint_boundary(
        enabled=True,
        native_acoustic_endpoint=False,
        semantic_hold_active=True,
        vad_active=True,
        trailing_silence_sec=0.0,
        utterance_elapsed_sec=19.9,
        max_silence_sec=1.6,
        rule3_min_utterance_length_sec=20.0,
    )
    hard = resolve_semantic_endpoint_boundary(
        enabled=True,
        native_acoustic_endpoint=False,
        semantic_hold_active=True,
        vad_active=True,
        trailing_silence_sec=0.0,
        utterance_elapsed_sec=20.0,
        max_silence_sec=1.6,
        rule3_min_utterance_length_sec=20.0,
    )

    assert resumed.effective_acoustic_endpoint is False
    assert hard.hard_boundary is True
    assert hard.effective_acoustic_endpoint is True


def test_disabled_boundary_preserves_only_current_native_endpoint() -> None:
    boundary = resolve_semantic_endpoint_boundary(
        enabled=False,
        native_acoustic_endpoint=False,
        semantic_hold_active=True,
        vad_active=False,
        trailing_silence_sec=99.0,
        utterance_elapsed_sec=99.0,
        max_silence_sec=1.6,
        rule3_min_utterance_length_sec=20.0,
    )

    assert boundary.semantic_hold_active is False
    assert boundary.max_silence_backstop is False
    assert boundary.hard_boundary is False
    assert boundary.effective_acoustic_endpoint is False
