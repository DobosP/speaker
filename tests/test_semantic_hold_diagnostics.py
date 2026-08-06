"""Transcript-free endpoint replay v2 coverage for native HOLD resets."""
from __future__ import annotations

from copy import deepcopy

from core import diagnostic_bundle
from core.diagnostic_bundle import DiagnosticStage, EndpointReplayConfig


_STREAM = "sherpa-11111111111111111111111111111111"


def _config() -> EndpointReplayConfig:
    return EndpointReplayConfig(
        algorithm="adaptive_endpoint_v2",
        enabled=True,
        min_silence_sec=0.7,
        max_silence_sec=1.6,
        complete_threshold=0.6,
        incomplete_threshold=0.3,
        semantic_hold_reset_enabled=True,
        rule3_min_utterance_length_sec=20.0,
    )


def _base(stage: DiagnosticStage, at: int) -> dict[str, object]:
    return {
        "kind": "observation",
        "stage": stage.value,
        "monotonic_ns": at,
        "frame_index": at,
        "bundle_sample_start": at * 1600,
        "bundle_sample_end": (at + 1) * 1600,
        "stream_id": _STREAM,
        "utterance_id": "u1",
    }


def _hold_evaluation() -> dict[str, object]:
    record = _base(DiagnosticStage.ENDPOINT_EVALUATED, 1)
    record.update(
        reason="unknown",
        basis="semantic_hold",
        completion_state="scored",
        numeric_value=0.05,
        acoustic_endpoint=True,
        native_acoustic_endpoint=True,
        semantic_hold_active=False,
        hard_boundary=False,
        vad_active=False,
        early_endpoint_allowed=True,
        trailing_silence_sec=0.8,
        utterance_elapsed_sec=1.0,
        boolean_value=False,
    )
    return record


def _held_reset() -> dict[str, object]:
    record = _base(DiagnosticStage.ENDPOINT_HELD_RESET, 1)
    record.update(ordinal=1, outcome="reset")
    return record


def _max_evaluation() -> dict[str, object]:
    record = _base(DiagnosticStage.ENDPOINT_EVALUATED, 2)
    record.update(
        reason="asr",
        basis="acoustic",
        completion_state="scored",
        numeric_value=0.05,
        acoustic_endpoint=True,
        native_acoustic_endpoint=False,
        semantic_hold_active=True,
        hard_boundary=False,
        vad_active=False,
        early_endpoint_allowed=True,
        trailing_silence_sec=1.6,
        utterance_elapsed_sec=1.8,
        boolean_value=True,
    )
    return record


def _commit() -> dict[str, object]:
    record = _base(DiagnosticStage.ENDPOINT_COMMITTED, 2)
    record.update(
        revision=1,
        reason="asr",
        basis="acoustic",
        completion_state="scored",
        numeric_value=0.05,
    )
    return record


def test_endpoint_replay_v2_round_trips_without_changing_v1_payload() -> None:
    config = _config()
    assert EndpointReplayConfig.from_payload(config.to_payload()) == config
    assert config.to_payload()["semantic_hold_reset_enabled"] is True

    v1 = EndpointReplayConfig()
    assert EndpointReplayConfig.from_payload(v1.to_payload()) == v1
    assert "semantic_hold_reset_enabled" not in v1.to_payload()
    assert "rule3_min_utterance_length_sec" not in v1.to_payload()


def test_endpoint_replay_v2_pairs_hold_reset_with_later_commit() -> None:
    records = [_hold_evaluation(), _held_reset(), _max_evaluation(), _commit()]

    assert diagnostic_bundle._validate_endpoint_replay(records, _config())


def test_endpoint_replay_v2_rejects_missing_or_tampered_reset() -> None:
    valid = [_hold_evaluation(), _held_reset(), _max_evaluation(), _commit()]
    assert not diagnostic_bundle._validate_endpoint_replay(
        [valid[0], *valid[2:]], _config()
    )

    for field, value in (
        ("ordinal", 2),
        ("outcome", "failed"),
        ("utterance_id", "u2"),
    ):
        tampered = deepcopy(valid)
        tampered[1][field] = value
        assert not diagnostic_bundle._validate_endpoint_replay(tampered, _config())


def test_endpoint_replay_v2_rejects_forged_effective_boundary() -> None:
    records = [_hold_evaluation(), _held_reset(), _max_evaluation(), _commit()]
    records[2]["acoustic_endpoint"] = False
    records[2]["boolean_value"] = False
    records[2]["reason"] = "unknown"
    records[2]["basis"] = "semantic_wait"

    assert not diagnostic_bundle._validate_endpoint_replay(records, _config())


def test_endpoint_replay_v2_accepts_abort_after_successful_reset() -> None:
    aborted = _base(DiagnosticStage.FINAL_ABORTED, 3)
    aborted.update(revision=1, reason="backpressure")

    assert diagnostic_bundle._validate_endpoint_replay(
        [_hold_evaluation(), _held_reset(), aborted], _config()
    )
