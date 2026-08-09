"""Transcript-free two-utterance overlap word-error accounting.

The protocol is deliberately narrower than speaker-attributed or time-constrained
multi-talker metrics.  It compares one hypothesis with both concatenation orders
of exactly two references and retains the lower word-edit count.  This is an
utterance-order-independent diagnostic for a single-output recognizer; it is not
ORC-WER, tcpWER, or tcORC.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final, Literal, Sequence

from core.wer import WerResult, word_error_rate


PROTOCOL_VERSION: Final = "two-utterance-min-order-wer-v1"

SelectedOrder = Literal[
    "role_a_then_role_b",
    "role_b_then_role_a",
]

_ROLE_A_THEN_ROLE_B: Final = "role_a_then_role_b"
_ROLE_B_THEN_ROLE_A: Final = "role_b_then_role_a"


@dataclass(frozen=True)
class OverlapMetricInput:
    """One private pair of references and its single-stream hypothesis."""

    reference_a: str = field(repr=False)
    reference_b: str = field(repr=False)
    hypothesis: str = field(repr=False)

    def __post_init__(self) -> None:
        if any(
            type(value) is not str
            for value in (self.reference_a, self.reference_b, self.hypothesis)
        ):
            raise ValueError("invalid overlap metric input")

    def __repr__(self) -> str:
        return "OverlapMetricInput(<private>)"


def _expected_wer(
    *,
    word_errors: int,
    reference_words: int,
    hypothesis_words: int,
) -> float:
    if reference_words == 0:
        return 0.0 if hypothesis_words == 0 else 1.0
    try:
        value = word_errors / reference_words
    except (OverflowError, ValueError):
        raise ValueError("invalid overlap metric counts") from None
    if not math.isfinite(value):
        raise ValueError("invalid overlap metric counts")
    return value


def _validate_counts(
    *,
    substitutions: object,
    insertions: object,
    deletions: object,
    word_errors: object,
    reference_words: object,
    hypothesis_words: object,
    wer: object,
) -> None:
    integers = (
        substitutions,
        insertions,
        deletions,
        word_errors,
        reference_words,
        hypothesis_words,
    )
    if any(type(value) is not int or value < 0 for value in integers):
        raise ValueError("invalid overlap metric counts")

    # The checks below bind the summary to a possible edit path.  In
    # particular, they keep callers from smuggling an independently averaged
    # or non-finite WER into an aggregate.
    if (
        word_errors != substitutions + insertions + deletions
        or deletions > reference_words
        or insertions > hypothesis_words
        or substitutions > reference_words - deletions
        or substitutions > hypothesis_words - insertions
        or hypothesis_words != reference_words - deletions + insertions
        or type(wer) is not float
        or not math.isfinite(wer)
    ):
        raise ValueError("invalid overlap metric counts")
    expected = _expected_wer(
        word_errors=word_errors,
        reference_words=reference_words,
        hypothesis_words=hypothesis_words,
    )
    if wer != expected:
        raise ValueError("invalid overlap metric counts")


@dataclass(frozen=True)
class OverlapMetricResult:
    """Aggregate-safe result for one two-reference/single-output case."""

    protocol: str
    selected_order: SelectedOrder
    substitutions: int
    insertions: int
    deletions: int
    word_errors: int
    reference_words: int
    hypothesis_words: int
    wer: float

    def __post_init__(self) -> None:
        if (
            type(self.protocol) is not str
            or self.protocol != PROTOCOL_VERSION
            or type(self.selected_order) is not str
            or self.selected_order not in {_ROLE_A_THEN_ROLE_B, _ROLE_B_THEN_ROLE_A}
        ):
            raise ValueError("invalid overlap metric result")
        _validate_counts(
            substitutions=self.substitutions,
            insertions=self.insertions,
            deletions=self.deletions,
            word_errors=self.word_errors,
            reference_words=self.reference_words,
            hypothesis_words=self.hypothesis_words,
            wer=self.wer,
        )

    def as_dict(self) -> dict[str, int | float | str]:
        """Return only protocol metadata and aggregate-safe counters."""

        return {
            "protocol": self.protocol,
            "selected_order": self.selected_order,
            "substitutions": self.substitutions,
            "insertions": self.insertions,
            "deletions": self.deletions,
            "word_errors": self.word_errors,
            "reference_words": self.reference_words,
            "hypothesis_words": self.hypothesis_words,
            "wer": self.wer,
        }


@dataclass(frozen=True)
class OverlapMetricAggregate:
    """Micro-aggregate of raw integer counts across scored cases."""

    protocol: str
    cases: int
    substitutions: int
    insertions: int
    deletions: int
    word_errors: int
    reference_words: int
    hypothesis_words: int
    wer: float

    def __post_init__(self) -> None:
        if (
            type(self.protocol) is not str
            or self.protocol != PROTOCOL_VERSION
            or type(self.cases) is not int
            or self.cases < 0
            or (
                self.cases == 0
                and any(
                    value != 0
                    for value in (
                        self.substitutions,
                        self.insertions,
                        self.deletions,
                        self.word_errors,
                        self.reference_words,
                        self.hypothesis_words,
                    )
                )
            )
        ):
            raise ValueError("invalid overlap metric aggregate")
        _validate_counts(
            substitutions=self.substitutions,
            insertions=self.insertions,
            deletions=self.deletions,
            word_errors=self.word_errors,
            reference_words=self.reference_words,
            hypothesis_words=self.hypothesis_words,
            wer=self.wer,
        )

    def as_dict(self) -> dict[str, int | float | str]:
        """Return a transcript-free JSON-compatible summary."""

        return {
            "protocol": self.protocol,
            "cases": self.cases,
            "substitutions": self.substitutions,
            "insertions": self.insertions,
            "deletions": self.deletions,
            "word_errors": self.word_errors,
            "reference_words": self.reference_words,
            "hypothesis_words": self.hypothesis_words,
            "wer": self.wer,
        }


def _word_errors(result: WerResult) -> int:
    return result.substitutions + result.insertions + result.deletions


def _result_from_wer(
    measured: WerResult,
    *,
    selected_order: SelectedOrder,
) -> OverlapMetricResult:
    errors = _word_errors(measured)
    return OverlapMetricResult(
        protocol=PROTOCOL_VERSION,
        selected_order=selected_order,
        substitutions=measured.substitutions,
        insertions=measured.insertions,
        deletions=measured.deletions,
        word_errors=errors,
        reference_words=measured.ref_words,
        hypothesis_words=measured.hyp_words,
        wer=_expected_wer(
            word_errors=errors,
            reference_words=measured.ref_words,
            hypothesis_words=measured.hyp_words,
        ),
    )


def score_overlap_metric(value: OverlapMetricInput) -> OverlapMetricResult:
    """Score one validated private input without retaining its text."""

    if type(value) is not OverlapMetricInput:
        raise ValueError("invalid overlap metric input")

    role_a_first = word_error_rate(
        f"{value.reference_a} {value.reference_b}",
        value.hypothesis,
    )
    role_b_first = word_error_rate(
        f"{value.reference_b} {value.reference_a}",
        value.hypothesis,
    )

    # Both orders contain the same reference words, so minimizing the integer
    # edit count is exactly equivalent to minimizing WER.  A tie always selects
    # A -> B, independent of the edit backtrace's S/I/D decomposition.
    if _word_errors(role_b_first) < _word_errors(role_a_first):
        return _result_from_wer(
            role_b_first,
            selected_order=_ROLE_B_THEN_ROLE_A,
        )
    return _result_from_wer(
        role_a_first,
        selected_order=_ROLE_A_THEN_ROLE_B,
    )


def score_two_utterance_min_order_wer(
    reference_a: str,
    reference_b: str,
    hypothesis: str,
) -> OverlapMetricResult:
    """Score both reference orders and deterministically retain the best one."""

    return score_overlap_metric(
        OverlapMetricInput(
            reference_a=reference_a,
            reference_b=reference_b,
            hypothesis=hypothesis,
        )
    )


def aggregate_two_utterance_min_order_wer(
    results: Sequence[OverlapMetricResult],
) -> OverlapMetricAggregate:
    """Sum integer edit counts first, then derive the micro-averaged WER."""

    if isinstance(results, (str, bytes)):
        raise ValueError("invalid overlap metric results")
    try:
        selected = tuple(results)
    except (TypeError, ValueError):
        raise ValueError("invalid overlap metric results") from None
    if any(type(result) is not OverlapMetricResult for result in selected):
        raise ValueError("invalid overlap metric results")

    substitutions = sum(result.substitutions for result in selected)
    insertions = sum(result.insertions for result in selected)
    deletions = sum(result.deletions for result in selected)
    reference_words = sum(result.reference_words for result in selected)
    hypothesis_words = sum(result.hypothesis_words for result in selected)
    word_errors = substitutions + insertions + deletions
    return OverlapMetricAggregate(
        protocol=PROTOCOL_VERSION,
        cases=len(selected),
        substitutions=substitutions,
        insertions=insertions,
        deletions=deletions,
        word_errors=word_errors,
        reference_words=reference_words,
        hypothesis_words=hypothesis_words,
        wer=_expected_wer(
            word_errors=word_errors,
            reference_words=reference_words,
            hypothesis_words=hypothesis_words,
        ),
    )
