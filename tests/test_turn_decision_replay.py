"""Deterministic, model-free replay tests for the typed endpoint seam."""

from __future__ import annotations

from dataclasses import dataclass, fields

import pytest

from always_on_agent.acoustic import EndpointReason
from core.endpointing import (
    AdaptiveEndpointPolicy,
    CompletionScoreState,
    EndpointConfig,
    TurnDecision,
    TurnDecisionBasis,
    TurnObservation,
    decide_turn,
)


def _policy(**changes) -> AdaptiveEndpointPolicy:
    return AdaptiveEndpointPolicy(EndpointConfig(enabled=True, **changes))


def _scored(
    *,
    acoustic_endpoint: bool,
    completion_score: float,
    trailing_silence_sec: float,
    vad_active: bool = False,
) -> TurnObservation:
    return TurnObservation(
        acoustic_endpoint=acoustic_endpoint,
        vad_active=vad_active,
        early_endpoint_allowed=not vad_active,
        trailing_silence_sec=trailing_silence_sec,
        completion_state=CompletionScoreState.SCORED,
        completion_score=completion_score,
    )


@pytest.mark.parametrize(
    "state",
    [
        CompletionScoreState.UNAVAILABLE,
        CompletionScoreState.NO_PARTIAL,
        CompletionScoreState.EARLY_GUARDED,
        CompletionScoreState.AUDIO_WINDOW_PENDING,
        CompletionScoreState.DETECTOR_ERROR,
    ],
)
@pytest.mark.parametrize("acoustic_endpoint", [False, True])
def test_unscored_observations_are_exact_acoustic_passthrough(
    state: CompletionScoreState,
    acoustic_endpoint: bool,
) -> None:
    observation = TurnObservation(
        acoustic_endpoint=acoustic_endpoint,
        vad_active=False,
        early_endpoint_allowed=True,
        trailing_silence_sec=0.9,
        completion_state=state,
        # A stale value must not become authoritative when the state says that
        # no score was admitted for this observation.
        completion_score=0.99,
    )

    assert decide_turn(observation, policy=_policy()) == TurnDecision(
        commit=acoustic_endpoint,
        endpoint_reason=(
            EndpointReason.ASR
            if acoustic_endpoint
            else EndpointReason.UNKNOWN
        ),
        basis=TurnDecisionBasis.ACOUSTIC,
    )


def test_vad_active_acoustic_endpoint_is_the_hard_max_wait_boundary() -> None:
    observation = TurnObservation(
        acoustic_endpoint=True,
        vad_active=True,
        early_endpoint_allowed=False,
        trailing_silence_sec=0.0,
        completion_state=CompletionScoreState.HARD_BOUNDARY,
        completion_score=0.01,
    )

    # The low score would make the ordinary policy hold. The hard ownership
    # boundary must bypass that result exactly as the live engine does today.
    assert decide_turn(observation, policy=_policy()) == TurnDecision(
        commit=True,
        endpoint_reason=EndpointReason.MAX_WAIT,
        basis=TurnDecisionBasis.MAX_WAIT,
    )


def test_early_guard_cannot_be_bypassed_by_an_inconsistent_score() -> None:
    observation = TurnObservation(
        acoustic_endpoint=False,
        vad_active=False,
        early_endpoint_allowed=False,
        trailing_silence_sec=5.0,
        completion_state=CompletionScoreState.SCORED,
        completion_score=0.99,
    )

    assert decide_turn(observation, policy=_policy()) == TurnDecision(
        False,
        EndpointReason.UNKNOWN,
        TurnDecisionBasis.ACOUSTIC,
    )


@pytest.mark.parametrize(
    "observation, expected",
    [
        (
            _scored(
                acoustic_endpoint=False,
                completion_score=0.9,
                trailing_silence_sec=0.5,
            ),
            TurnDecision(
                True,
                EndpointReason.SEMANTIC,
                TurnDecisionBasis.SEMANTIC_EARLY,
            ),
        ),
        (
            _scored(
                acoustic_endpoint=True,
                completion_score=0.1,
                trailing_silence_sec=1.0,
            ),
            TurnDecision(
                False,
                EndpointReason.UNKNOWN,
                TurnDecisionBasis.SEMANTIC_HOLD,
            ),
        ),
        (
            _scored(
                acoustic_endpoint=True,
                completion_score=0.5,
                trailing_silence_sec=0.9,
            ),
            TurnDecision(
                True,
                EndpointReason.ASR,
                TurnDecisionBasis.ACOUSTIC,
            ),
        ),
        (
            _scored(
                acoustic_endpoint=False,
                completion_score=0.5,
                trailing_silence_sec=0.9,
            ),
            TurnDecision(
                False,
                EndpointReason.UNKNOWN,
                TurnDecisionBasis.SEMANTIC_WAIT,
            ),
        ),
        (
            _scored(
                acoustic_endpoint=True,
                completion_score=0.1,
                trailing_silence_sec=1.7,
            ),
            TurnDecision(
                True,
                EndpointReason.ASR,
                TurnDecisionBasis.ACOUSTIC,
            ),
        ),
    ],
)
def test_scored_observations_retain_policy_result_and_explain_its_basis(
    observation: TurnObservation,
    expected: TurnDecision,
) -> None:
    assert decide_turn(
        observation,
        policy=_policy(min_silence_sec=0.4, max_silence_sec=1.6),
    ) == expected


@pytest.mark.parametrize(
    "config, acoustic_endpoint, score, silence_sec",
    [
        ({"min_silence_sec": 0.4}, False, 0.9, 0.5),
        ({"min_silence_sec": 0.4}, False, 0.9, 0.3),
        (
            {
                "min_silence_sec": 0.7,
                "complete_threshold": 0.6,
                "high_confidence_floor": 0.55,
                "high_confidence_score": 0.75,
            },
            False,
            0.75,
            0.55,
        ),
        ({"max_silence_sec": 1.6}, True, 0.1, 1.0),
        ({"max_silence_sec": 1.6}, True, 0.1, 1.7),
        ({}, True, 0.5, 0.9),
        ({}, False, 0.5, 0.9),
    ],
)
def test_typed_commit_is_exactly_the_existing_policy_boolean(
    config: dict[str, float],
    acoustic_endpoint: bool,
    score: float,
    silence_sec: float,
) -> None:
    legacy = _policy(**config).decide(
        acoustic_endpoint=acoustic_endpoint,
        completion_score=score,
        silence_sec=silence_sec,
    )
    typed = decide_turn(
        _scored(
            acoustic_endpoint=acoustic_endpoint,
            completion_score=score,
            trailing_silence_sec=silence_sec,
        ),
        policy=_policy(**config),
    )

    assert typed.commit is legacy


def test_missing_policy_or_missing_score_falls_back_to_acoustic() -> None:
    scored = _scored(
        acoustic_endpoint=True,
        completion_score=0.01,
        trailing_silence_sec=1.0,
    )
    missing_score = TurnObservation(
        acoustic_endpoint=False,
        vad_active=False,
        early_endpoint_allowed=True,
        trailing_silence_sec=1.0,
        completion_state=CompletionScoreState.SCORED,
        completion_score=None,
    )

    assert decide_turn(scored, policy=None) == TurnDecision(
        True,
        EndpointReason.ASR,
        TurnDecisionBasis.ACOUSTIC,
    )
    assert decide_turn(missing_score, policy=_policy()) == TurnDecision(
        False,
        EndpointReason.UNKNOWN,
        TurnDecisionBasis.ACOUSTIC,
    )


@dataclass(frozen=True)
class _Pause:
    seconds: float


_ADAPTIVE_TRACE: tuple[_Pause | TurnObservation, ...] = (
    _scored(
        acoustic_endpoint=False,
        completion_score=0.9,
        trailing_silence_sec=0.5,
    ),
    _Pause(0.9),
    _scored(
        acoustic_endpoint=True,
        completion_score=0.75,
        trailing_silence_sec=0.6,
    ),
    _scored(
        acoustic_endpoint=True,
        completion_score=0.75,
        trailing_silence_sec=0.95,
    ),
    _scored(
        acoustic_endpoint=False,
        completion_score=0.75,
        trailing_silence_sec=0.95,
    ),
)


def _replay_adaptive_trace() -> tuple[TurnDecision, ...]:
    policy = _policy(
        adaptive_floor=True,
        min_silence_sec=0.5,
        max_silence_sec=1.6,
        high_confidence_floor=0.4,
        pause_margin=0.0,
        pause_min_samples=1,
    )
    decisions: list[TurnDecision] = []
    for step in _ADAPTIVE_TRACE:
        if isinstance(step, _Pause):
            policy.observe_pause(step.seconds)
        else:
            decisions.append(decide_turn(step, policy=policy))
    return tuple(decisions)


def test_ordered_adaptive_trace_replays_deterministically() -> None:
    expected = (
        TurnDecision(
            True,
            EndpointReason.SEMANTIC,
            TurnDecisionBasis.SEMANTIC_EARLY,
        ),
        TurnDecision(
            False,
            EndpointReason.UNKNOWN,
            TurnDecisionBasis.SEMANTIC_HOLD,
        ),
        TurnDecision(
            True,
            EndpointReason.ASR,
            TurnDecisionBasis.ACOUSTIC,
        ),
        TurnDecision(
            True,
            EndpointReason.SEMANTIC,
            TurnDecisionBasis.SEMANTIC_EARLY,
        ),
    )

    assert _replay_adaptive_trace() == expected
    assert _replay_adaptive_trace() == expected


def test_endpoint_policy_rejects_nonfinite_unreplayable_configuration() -> None:
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="must be finite"):
            AdaptiveEndpointPolicy(
                EndpointConfig(enabled=True, min_silence_sec=value)
            )


def test_pause_snapshot_restore_is_exact_and_atomic() -> None:
    policy = _policy(
        adaptive_floor=True,
        max_silence_sec=1.6,
        pause_window=2,
    )
    policy.restore_replay_pause_samples((0.3, 0.4))
    assert policy.replay_pause_samples() == (0.3, 0.4)

    with pytest.raises(ValueError, match="not representable"):
        policy.restore_replay_pause_samples((0.1, 0.5))
    assert policy.replay_pause_samples() == (0.3, 0.4)


def test_turn_observation_schema_contains_no_text_pcm_identity_or_clock() -> None:
    assert {item.name for item in fields(TurnObservation)} == {
        "acoustic_endpoint",
        "vad_active",
        "early_endpoint_allowed",
        "trailing_silence_sec",
        "completion_state",
        "completion_score",
    }
