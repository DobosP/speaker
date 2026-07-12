from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from tests.replay_voice_driver import (
    BargeResult,
    RECORDED_BARGE_DRAIN_GRACE_SEC,
    RECORDED_OWNER_TO_STOP_MAX_SEC,
    causal_barge_cut,
    load_manifest,
    recorded_floor_control_deadline,
    require_recorded_prerequisite,
    validate_manifest,
)


def _valid() -> BargeResult:
    return BargeResult(
        asr_final="Tell me a long story.",
        response="A deliberately long response.",
        target_turn_token=7,
        metric_turn_token=7,
        base_fully_consumed_at=1.0,
        tts_first_audio_at=2.0,
        barge_first_consumed_at=3.0,
        barge_fully_consumed_at=5.0,
        barge_in_at=3.2,
        stop_call_at=3.3,
        barge_in_stop_at=3.31,
        floor_control_clean=True,
        speaking_before_barge=True,
    )


def test_causal_barge_evidence_accepts_one_ordered_same_turn_cut():
    evidence = _valid()
    assert causal_barge_cut(evidence) is True
    assert evidence.barged is True
    assert evidence.barge_in_latency == pytest.approx(0.11)


def test_causal_barge_evidence_enforces_verifier_owned_time_bounds():
    owner_boundary = replace(
        _valid(),
        barge_first_consumed_at=3.0,
        barge_fully_consumed_at=4.0,
        barge_in_at=3.2,
        stop_call_at=3.9,
        barge_in_stop_at=3.0 + RECORDED_OWNER_TO_STOP_MAX_SEC,
    )
    assert causal_barge_cut(owner_boundary) is True
    assert causal_barge_cut(
        replace(
            owner_boundary,
            barge_in_stop_at=(
                3.0 + RECORDED_OWNER_TO_STOP_MAX_SEC + 1e-6
            ),
        )
    ) is False

    drain_boundary = replace(
        _valid(),
        barge_fully_consumed_at=3.2,
        stop_call_at=3.3,
        barge_in_stop_at=3.2 + RECORDED_BARGE_DRAIN_GRACE_SEC,
    )
    assert causal_barge_cut(drain_boundary) is True
    assert causal_barge_cut(
        replace(
            drain_boundary,
            barge_in_stop_at=(
                3.2 + RECORDED_BARGE_DRAIN_GRACE_SEC + 1e-6
            ),
        )
    ) is False


@pytest.mark.parametrize(
    ("observed", "expected"),
    [
        (20.2, 21.0),  # onset grace dominates
        (20.6, 21.2),  # late true first-audio observation dominates
    ],
)
def test_recorded_floor_control_observes_full_armed_window(observed, expected):
    assert recorded_floor_control_deadline(
        first_audio_observed_at=observed,
        playback_onset_at=20.0,
        onset_grace_sec=0.4,
        sustain_window_sec=0.5,
        block_sec=0.1,
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    "changes",
    [
        {"metric_turn_token": 6},  # stale record from an older turn
        {"barge_in_stop_at": None},  # BARGE_IN without a FIFO cut
        {"barge_in_at": None},  # a STOP stamp cannot stand in for BARGE_IN
        {"stop_call_at": None},  # FIFO stamp without the observed engine call
        {"barge_in_at": 2.8, "barge_in_stop_at": 2.9},  # before owner PCM
        {"barge_fully_consumed_at": 2.9},  # impossible drain before first sample
        {"stop_call_at": 4.1, "barge_in_stop_at": 4.2},  # unusably late cut
        {"barge_in_stop_at": 5.3},  # after this buffer/grace
        {"floor_control_clean": False},  # self-cut before owner injection
        {"speaking_before_barge": False},  # stop was a no-op
    ],
)
def test_causal_barge_evidence_rejects_false_green_shapes(changes):
    assert causal_barge_cut(replace(_valid(), **changes)) is False


def test_recorded_manifest_pins_unique_waveforms_and_same_run_cases():
    manifest = load_manifest()
    lookup = validate_manifest(manifest)
    assert len(lookup) == 8
    assert len(manifest["clips"]) == 6
    assert len(manifest["barge"]) == 2
    assert len(manifest["barge_cases"]) == 2


def test_required_recorded_mode_turns_missing_prerequisite_into_failure(
    monkeypatch,
):
    monkeypatch.setenv("SPEAKER_REQUIRE_RECORDED", "1")
    with pytest.raises(pytest.fail.Exception, match="required recorded gate"):
        require_recorded_prerequisite(False, "fixture missing")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda data: data["clips"][0].update(sha256=""),
        lambda data: data["clips"][1].update(id=data["clips"][0]["id"]),
        lambda data: data["barge"][0].update(
            sha256=data["clips"][0]["sha256"]
        ),
        lambda data: data["barge_cases"][0].update(run="wrong-run"),
        lambda data: data.update(barge_cases=[]),
        lambda data: data["barge_cases"][1].update(
            base_id=data["barge_cases"][0]["base_id"],
            barge_id=data["barge_cases"][0]["barge_id"],
            run=data["barge_cases"][0]["run"],
        ),
        lambda data: data["clips"].pop(),
        lambda data: data["clips"][0].update(expected_text=""),
        lambda data: data["clips"][0].update(start_sec=float("nan")),
        lambda data: data["clips"][0].update(end_sec=0.0),
        lambda data: data["clips"][2].update(speech_sec=2.1),
    ],
)
def test_recorded_manifest_rejects_unpinned_or_ambiguous_provenance(mutate):
    manifest = deepcopy(load_manifest())
    mutate(manifest)
    with pytest.raises(ValueError):
        validate_manifest(manifest)
