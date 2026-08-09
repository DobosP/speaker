"""Pure, inactive semantic-interruption decision contract.

This module classifies already-produced semantic score evidence.  It owns no
detector, text, PCM, speaker identity, callback, clock, thread, or effect.  In
particular, ``NON_INTERRUPTING`` is evidence only: it does not authorize a
runtime to continue playback, resume a reply, suppress an exact STOP, or grant
tool authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


PROTOCOL_VERSION = "semantic-interruption-policy-v1"


class SemanticScoreState(str, Enum):
    """Availability of one bounded semantic takeover score."""

    UNAVAILABLE = "unavailable"
    SCORED = "scored"
    DETECTOR_ERROR = "detector_error"


class InterruptionDisposition(str, Enum):
    """Effect-free interpretation of one candidate observation."""

    TAKE_FLOOR = "take_floor"
    NON_INTERRUPTING = "non_interrupting"
    ABSTAIN = "abstain"


class InterruptionDecisionBasis(str, Enum):
    """Closed reason for the returned disposition."""

    SEMANTIC_SCORE = "semantic_score"
    NOT_A_CANDIDATE = "not_a_candidate"
    NO_SCORE = "no_score"
    STALE_DETECTOR_RESULT = "stale_detector_result"
    DETECTOR_ERROR = "detector_error"


def _exact_bool(name: str, value: bool) -> None:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a bool")


def _positive_int(name: str, value: int) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an int")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _score(name: str, value: float) -> None:
    if type(value) is not float:
        raise TypeError(f"{name} must be a float")
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and within [0, 1]")


@dataclass(frozen=True)
class InterruptionPolicyConfig:
    """Explicit dead-band thresholds for one semantic score protocol."""

    non_interrupting_max: float
    take_floor_min: float

    def __post_init__(self) -> None:
        _score("non_interrupting_max", self.non_interrupting_max)
        _score("take_floor_min", self.take_floor_min)
        if math.nextafter(self.non_interrupting_max, 1.0) >= self.take_floor_min:
            raise ValueError(
                "thresholds must leave a representable score in the dead band"
            )


@dataclass(frozen=True)
class InterruptionObservation:
    """Transcript-free facts supplied to the pure policy.

    ``candidate_token`` identifies the current physical overlap candidate.  The
    caller must allocate a positive token that is never reused within its
    process.  A scored or failed detector result carries the token it evaluated;
    a mismatch is stale and abstains, including an A -> B -> A-shaped race.
    """

    playback_active: bool
    acoustic_confirmed: bool
    candidate_token: int
    score_state: SemanticScoreState
    detector_candidate_token: int | None = None
    takeover_score: float | None = None

    def __post_init__(self) -> None:
        _exact_bool("playback_active", self.playback_active)
        _exact_bool("acoustic_confirmed", self.acoustic_confirmed)
        _positive_int("candidate_token", self.candidate_token)
        if type(self.score_state) is not SemanticScoreState:
            raise TypeError("score_state must be a SemanticScoreState")
        if self.score_state is SemanticScoreState.UNAVAILABLE:
            if self.detector_candidate_token is not None:
                raise ValueError(
                    "unavailable observation cannot carry a detector candidate token"
                )
        else:
            if self.detector_candidate_token is None:
                raise ValueError("detector result requires detector_candidate_token")
            _positive_int("detector_candidate_token", self.detector_candidate_token)
        if self.score_state is SemanticScoreState.SCORED:
            if self.takeover_score is None:
                raise ValueError("scored observation requires takeover_score")
            _score("takeover_score", self.takeover_score)
        elif self.takeover_score is not None:
            raise ValueError("unscored observation cannot carry a score")


@dataclass(frozen=True)
class InterruptionDecision:
    """Closed, effect-free output of :func:`decide_interruption`."""

    disposition: InterruptionDisposition
    basis: InterruptionDecisionBasis

    def __post_init__(self) -> None:
        if type(self.disposition) is not InterruptionDisposition:
            raise TypeError("disposition must be an InterruptionDisposition")
        if type(self.basis) is not InterruptionDecisionBasis:
            raise TypeError("basis must be an InterruptionDecisionBasis")
        scored_disposition = self.disposition in {
            InterruptionDisposition.TAKE_FLOOR,
            InterruptionDisposition.NON_INTERRUPTING,
        }
        if scored_disposition != (
            self.basis is InterruptionDecisionBasis.SEMANTIC_SCORE
            and self.disposition is not InterruptionDisposition.ABSTAIN
        ):
            raise ValueError("disposition and basis are inconsistent")


def decide_interruption(
    observation: InterruptionObservation,
    config: InterruptionPolicyConfig,
) -> InterruptionDecision:
    """Return one deterministic disposition without performing an effect."""

    if type(observation) is not InterruptionObservation:
        raise TypeError("observation must be an InterruptionObservation")
    if type(config) is not InterruptionPolicyConfig:
        raise TypeError("config must be an InterruptionPolicyConfig")
    if not observation.playback_active or not observation.acoustic_confirmed:
        return InterruptionDecision(
            InterruptionDisposition.ABSTAIN,
            InterruptionDecisionBasis.NOT_A_CANDIDATE,
        )
    if observation.score_state is SemanticScoreState.UNAVAILABLE:
        return InterruptionDecision(
            InterruptionDisposition.ABSTAIN,
            InterruptionDecisionBasis.NO_SCORE,
        )
    if observation.detector_candidate_token != observation.candidate_token:
        return InterruptionDecision(
            InterruptionDisposition.ABSTAIN,
            InterruptionDecisionBasis.STALE_DETECTOR_RESULT,
        )
    if observation.score_state is SemanticScoreState.DETECTOR_ERROR:
        return InterruptionDecision(
            InterruptionDisposition.ABSTAIN,
            InterruptionDecisionBasis.DETECTOR_ERROR,
        )

    score = observation.takeover_score
    if score is None:  # Constructor validation makes this unreachable.
        raise ValueError("scored observation requires takeover_score")
    if score <= config.non_interrupting_max:
        return InterruptionDecision(
            InterruptionDisposition.NON_INTERRUPTING,
            InterruptionDecisionBasis.SEMANTIC_SCORE,
        )
    if score >= config.take_floor_min:
        return InterruptionDecision(
            InterruptionDisposition.TAKE_FLOOR,
            InterruptionDecisionBasis.SEMANTIC_SCORE,
        )
    return InterruptionDecision(
        InterruptionDisposition.ABSTAIN,
        InterruptionDecisionBasis.SEMANTIC_SCORE,
    )


__all__ = [
    "PROTOCOL_VERSION",
    "InterruptionDecision",
    "InterruptionDecisionBasis",
    "InterruptionDisposition",
    "InterruptionObservation",
    "InterruptionPolicyConfig",
    "SemanticScoreState",
    "decide_interruption",
]
