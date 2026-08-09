"""Exhaustive contracts for the inactive semantic-interruption policy."""

from __future__ import annotations

from dataclasses import asdict, fields
import math

import pytest

from always_on_agent.interruption_policy import (
    PROTOCOL_VERSION,
    InterruptionDecision,
    InterruptionDecisionBasis,
    InterruptionDisposition,
    InterruptionObservation,
    InterruptionPolicyConfig,
    SemanticScoreState,
    decide_interruption,
)


CONFIG = InterruptionPolicyConfig(
    non_interrupting_max=0.2,
    take_floor_min=0.8,
)
CURRENT_TOKEN = 7


@pytest.mark.parametrize(
    ("score", "disposition"),
    [
        (0.0, InterruptionDisposition.NON_INTERRUPTING),
        (0.2, InterruptionDisposition.NON_INTERRUPTING),
        (math.nextafter(0.2, 1.0), InterruptionDisposition.ABSTAIN),
        (0.5, InterruptionDisposition.ABSTAIN),
        (math.nextafter(0.8, 0.0), InterruptionDisposition.ABSTAIN),
        (0.8, InterruptionDisposition.TAKE_FLOOR),
        (1.0, InterruptionDisposition.TAKE_FLOOR),
    ],
)
def test_scored_candidate_uses_closed_threshold_boundaries(
    score: float,
    disposition: InterruptionDisposition,
) -> None:
    decision = decide_interruption(
        InterruptionObservation(
            playback_active=True,
            acoustic_confirmed=True,
            candidate_token=CURRENT_TOKEN,
            score_state=SemanticScoreState.SCORED,
            detector_candidate_token=CURRENT_TOKEN,
            takeover_score=score,
        ),
        CONFIG,
    )

    assert decision == InterruptionDecision(
        disposition=disposition,
        basis=InterruptionDecisionBasis.SEMANTIC_SCORE,
    )


@pytest.mark.parametrize(
    ("playback_active", "acoustic_confirmed"),
    [(False, False), (False, True), (True, False)],
)
@pytest.mark.parametrize(
    ("score_state", "score"),
    [
        (SemanticScoreState.UNAVAILABLE, None),
        (SemanticScoreState.DETECTOR_ERROR, None),
        (SemanticScoreState.SCORED, 0.0),
        (SemanticScoreState.SCORED, 1.0),
    ],
)
def test_non_candidate_always_abstains_even_if_a_score_arrived(
    playback_active: bool,
    acoustic_confirmed: bool,
    score_state: SemanticScoreState,
    score: float | None,
) -> None:
    decision = decide_interruption(
        InterruptionObservation(
            playback_active=playback_active,
            acoustic_confirmed=acoustic_confirmed,
            candidate_token=CURRENT_TOKEN,
            score_state=score_state,
            detector_candidate_token=(
                None
                if score_state is SemanticScoreState.UNAVAILABLE
                else CURRENT_TOKEN + 1
            ),
            takeover_score=score,
        ),
        CONFIG,
    )

    assert decision == InterruptionDecision(
        disposition=InterruptionDisposition.ABSTAIN,
        basis=InterruptionDecisionBasis.NOT_A_CANDIDATE,
    )


@pytest.mark.parametrize(
    ("score_state", "basis"),
    [
        (SemanticScoreState.UNAVAILABLE, InterruptionDecisionBasis.NO_SCORE),
        (
            SemanticScoreState.DETECTOR_ERROR,
            InterruptionDecisionBasis.DETECTOR_ERROR,
        ),
    ],
)
def test_unscored_candidate_abstains_with_exact_basis(
    score_state: SemanticScoreState,
    basis: InterruptionDecisionBasis,
) -> None:
    assert decide_interruption(
        InterruptionObservation(
            playback_active=True,
            acoustic_confirmed=True,
            candidate_token=CURRENT_TOKEN,
            score_state=score_state,
            detector_candidate_token=(
                None if score_state is SemanticScoreState.UNAVAILABLE else CURRENT_TOKEN
            ),
        ),
        CONFIG,
    ) == InterruptionDecision(InterruptionDisposition.ABSTAIN, basis)


@pytest.mark.parametrize("name", ["playback_active", "acoustic_confirmed"])
@pytest.mark.parametrize("value", [0, 1, None, "true", object()])
def test_observation_requires_exact_bools(name: str, value: object) -> None:
    kwargs = {
        "playback_active": True,
        "acoustic_confirmed": True,
        "candidate_token": CURRENT_TOKEN,
        "score_state": SemanticScoreState.UNAVAILABLE,
    }
    kwargs[name] = value
    with pytest.raises(TypeError, match=rf"{name} must be a bool"):
        InterruptionObservation(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", ["scored", None, 1, object()])
def test_observation_requires_typed_score_state(value: object) -> None:
    with pytest.raises(TypeError, match="score_state must be"):
        InterruptionObservation(
            playback_active=True,
            acoustic_confirmed=True,
            candidate_token=CURRENT_TOKEN,
            score_state=value,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "score",
    [None, True, False, 0, 1, -0.01, 1.01, math.nan, math.inf, -math.inf],
)
def test_scored_observation_requires_one_finite_float_score(
    score: object,
) -> None:
    expected = ValueError if score is None or type(score) is float else TypeError
    with pytest.raises(expected):
        InterruptionObservation(
            playback_active=True,
            acoustic_confirmed=True,
            candidate_token=CURRENT_TOKEN,
            score_state=SemanticScoreState.SCORED,
            detector_candidate_token=CURRENT_TOKEN,
            takeover_score=score,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "score_state",
    [SemanticScoreState.UNAVAILABLE, SemanticScoreState.DETECTOR_ERROR],
)
def test_unscored_observation_rejects_a_score(
    score_state: SemanticScoreState,
) -> None:
    with pytest.raises(ValueError, match="unscored observation"):
        InterruptionObservation(
            playback_active=True,
            acoustic_confirmed=True,
            candidate_token=CURRENT_TOKEN,
            score_state=score_state,
            detector_candidate_token=(
                None if score_state is SemanticScoreState.UNAVAILABLE else CURRENT_TOKEN
            ),
            takeover_score=0.5,
        )


@pytest.mark.parametrize("name", ["candidate_token", "detector_candidate_token"])
@pytest.mark.parametrize(
    ("value", "error"),
    [
        (0, ValueError),
        (-1, ValueError),
        (True, TypeError),
        (1.0, TypeError),
        ("1", TypeError),
    ],
)
def test_observation_requires_positive_exact_candidate_tokens(
    name: str,
    value: object,
    error: type[Exception],
) -> None:
    kwargs = {
        "playback_active": True,
        "acoustic_confirmed": True,
        "candidate_token": CURRENT_TOKEN,
        "score_state": SemanticScoreState.SCORED,
        "detector_candidate_token": CURRENT_TOKEN,
        "takeover_score": 0.5,
    }
    kwargs[name] = value
    with pytest.raises(error, match=rf"{name} must be"):
        InterruptionObservation(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("score_state", "detector_candidate_token", "match"),
    [
        (
            SemanticScoreState.UNAVAILABLE,
            CURRENT_TOKEN,
            "unavailable observation",
        ),
        (SemanticScoreState.SCORED, None, "detector result requires"),
        (SemanticScoreState.DETECTOR_ERROR, None, "detector result requires"),
    ],
)
def test_detector_token_presence_matches_score_state(
    score_state: SemanticScoreState,
    detector_candidate_token: int | None,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        InterruptionObservation(
            playback_active=True,
            acoustic_confirmed=True,
            candidate_token=CURRENT_TOKEN,
            score_state=score_state,
            detector_candidate_token=detector_candidate_token,
            takeover_score=(0.5 if score_state is SemanticScoreState.SCORED else None),
        )


@pytest.mark.parametrize(
    ("score_state", "score"),
    [
        (SemanticScoreState.SCORED, 0.0),
        (SemanticScoreState.SCORED, 1.0),
        (SemanticScoreState.DETECTOR_ERROR, None),
    ],
)
def test_stale_detector_result_abstains_before_score_or_error(
    score_state: SemanticScoreState,
    score: float | None,
) -> None:
    decision = decide_interruption(
        InterruptionObservation(
            playback_active=True,
            acoustic_confirmed=True,
            candidate_token=CURRENT_TOKEN,
            score_state=score_state,
            detector_candidate_token=CURRENT_TOKEN - 1,
            takeover_score=score,
        ),
        CONFIG,
    )

    assert decision == InterruptionDecision(
        InterruptionDisposition.ABSTAIN,
        InterruptionDecisionBasis.STALE_DETECTOR_RESULT,
    )


def test_never_reused_tokens_prevent_an_aba_result_from_reviving() -> None:
    first_a_token = 41
    _intervening_b_token = 42
    current_a_token = 43

    assert decide_interruption(
        InterruptionObservation(
            playback_active=True,
            acoustic_confirmed=True,
            candidate_token=current_a_token,
            score_state=SemanticScoreState.SCORED,
            detector_candidate_token=first_a_token,
            takeover_score=1.0,
        ),
        CONFIG,
    ) == InterruptionDecision(
        InterruptionDisposition.ABSTAIN,
        InterruptionDecisionBasis.STALE_DETECTOR_RESULT,
    )


@pytest.mark.parametrize(
    ("non_interrupting_max", "take_floor_min", "error"),
    [
        (0.5, 0.5, ValueError),
        (0.6, 0.5, ValueError),
        (0.5, math.nextafter(0.5, 1.0), ValueError),
        (-0.1, 0.8, ValueError),
        (0.2, 1.1, ValueError),
        (math.nan, 0.8, ValueError),
        (0.2, math.inf, ValueError),
        (0, 0.8, TypeError),
        (0.2, 1, TypeError),
        (False, 0.8, TypeError),
        (0.2, True, TypeError),
    ],
)
def test_config_rejects_malformed_or_overlapping_thresholds(
    non_interrupting_max: object,
    take_floor_min: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        InterruptionPolicyConfig(
            non_interrupting_max=non_interrupting_max,  # type: ignore[arg-type]
            take_floor_min=take_floor_min,  # type: ignore[arg-type]
        )


def test_config_accepts_a_representable_dead_band_and_extremes() -> None:
    middle = math.nextafter(0.5, 1.0)
    upper = math.nextafter(middle, 1.0)

    assert InterruptionPolicyConfig(0.5, upper) == InterruptionPolicyConfig(
        non_interrupting_max=0.5,
        take_floor_min=upper,
    )
    assert InterruptionPolicyConfig(0.0, 1.0) == InterruptionPolicyConfig(
        non_interrupting_max=0.0,
        take_floor_min=1.0,
    )
    assert decide_interruption(
        InterruptionObservation(
            playback_active=True,
            acoustic_confirmed=True,
            candidate_token=CURRENT_TOKEN,
            score_state=SemanticScoreState.SCORED,
            detector_candidate_token=CURRENT_TOKEN,
            takeover_score=middle,
        ),
        InterruptionPolicyConfig(0.5, upper),
    ) == InterruptionDecision(
        InterruptionDisposition.ABSTAIN,
        InterruptionDecisionBasis.SEMANTIC_SCORE,
    )


@pytest.mark.parametrize("disposition", list(InterruptionDisposition))
@pytest.mark.parametrize("basis", list(InterruptionDecisionBasis))
def test_decision_has_exact_closed_cartesian_algebra(
    disposition: InterruptionDisposition,
    basis: InterruptionDecisionBasis,
) -> None:
    is_valid = disposition is InterruptionDisposition.ABSTAIN or (
        basis is InterruptionDecisionBasis.SEMANTIC_SCORE
    )
    if is_valid:
        assert InterruptionDecision(disposition, basis) == InterruptionDecision(
            disposition=disposition,
            basis=basis,
        )
        return
    with pytest.raises(ValueError, match="inconsistent"):
        InterruptionDecision(disposition, basis)


@pytest.mark.parametrize(
    ("disposition", "basis"),
    [
        ("take_floor", InterruptionDecisionBasis.SEMANTIC_SCORE),
        (InterruptionDisposition.TAKE_FLOOR, "semantic_score"),
    ],
)
def test_decision_requires_exact_enum_types(
    disposition: object,
    basis: object,
) -> None:
    with pytest.raises(TypeError):
        InterruptionDecision(disposition, basis)  # type: ignore[arg-type]


def test_decide_requires_exact_contract_objects() -> None:
    observation = InterruptionObservation(
        playback_active=True,
        acoustic_confirmed=True,
        candidate_token=CURRENT_TOKEN,
        score_state=SemanticScoreState.UNAVAILABLE,
    )
    with pytest.raises(TypeError, match="observation must be"):
        decide_interruption(asdict(observation), CONFIG)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="config must be"):
        decide_interruption(observation, asdict(CONFIG))  # type: ignore[arg-type]


def test_contract_is_deterministic_effect_free_and_transcript_free() -> None:
    observation = InterruptionObservation(
        playback_active=True,
        acoustic_confirmed=True,
        candidate_token=CURRENT_TOKEN,
        score_state=SemanticScoreState.SCORED,
        detector_candidate_token=CURRENT_TOKEN,
        takeover_score=0.1,
    )
    before = asdict(observation)

    first = decide_interruption(observation, CONFIG)
    second = decide_interruption(observation, CONFIG)

    assert first == second
    assert asdict(observation) == before
    assert PROTOCOL_VERSION == "semantic-interruption-policy-v1"
    assert {item.name for item in fields(InterruptionObservation)} == {
        "playback_active",
        "acoustic_confirmed",
        "candidate_token",
        "score_state",
        "detector_candidate_token",
        "takeover_score",
    }
    assert {item.name for item in fields(InterruptionPolicyConfig)} == {
        "non_interrupting_max",
        "take_floor_min",
    }
    assert {item.name for item in fields(InterruptionDecision)} == {
        "disposition",
        "basis",
    }
    assert [item.value for item in SemanticScoreState] == [
        "unavailable",
        "scored",
        "detector_error",
    ]
    assert [item.value for item in InterruptionDisposition] == [
        "take_floor",
        "non_interrupting",
        "abstain",
    ]
    assert [item.value for item in InterruptionDecisionBasis] == [
        "semantic_score",
        "not_a_candidate",
        "no_score",
        "stale_detector_result",
        "detector_error",
    ]
    serialized = repr(
        (
            CONFIG,
            observation,
            first,
            asdict(CONFIG),
            asdict(observation),
            asdict(first),
        )
    )
    for forbidden in (
        "transcript",
        "speaker",
        "audio",
        "pcm",
        "lineage",
        "text",
        "path",
        "file",
        "command",
        "tool",
        "identity",
    ):
        assert forbidden not in serialized.lower()
