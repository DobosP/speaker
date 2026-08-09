"""Pure four-way reduction for the locked Anyreach synthetic shadow slice.

The reducer accepts only aggregate-safe score events.  It does not retain or
emit messages, row IDs, per-row logits, local paths, policy scores, or effects.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final, Sequence

from tools.semantic_interruption.anyreach import (
    AnyreachAction,
    BenchmarkSlot,
    CANDIDATE_ID,
)
from tools.semantic_interruption.protocol import (
    ArgmaxDecision,
    MODEL_ACTION_ORDER,
    ProtocolError,
    ScoreEvent,
    deterministic_argmax,
    stable_four_way_softmax,
)


SHADOW_SUMMARY_KIND: Final = "anyreach-four-way-shadow-summary-v1"
EXPECTED_ACTION_ORDER: Final = (
    AnyreachAction.START_LISTENING,
    AnyreachAction.CONTINUE_SPEAKING,
)
PREDICTION_ACTION_ORDER: Final = MODEL_ACTION_ORDER
SELECTED_ROWS: Final = 24


class ShadowContractError(RuntimeError):
    """A locked-slot result set failed closed without private detail."""


def _expected_slots() -> tuple[BenchmarkSlot, ...]:
    return tuple(
        BenchmarkSlot(
            ordinal=index,
            source_row=36 + index,
            row_id=(
                f"manual_sli_{index + 1:02d}"
                if index < 12
                else f"manual_cs_{index - 11:02d}"
            ),
            action_index=2 if index < 12 else 3,
            action=(
                AnyreachAction.START_LISTENING
                if index < 12
                else AnyreachAction.CONTINUE_SPEAKING
            ),
        )
        for index in range(SELECTED_ROWS)
    )


EXACT_SYNTHETIC24_SLOTS: Final = _expected_slots()


def _count(value: object, *, maximum: int = SELECTED_ROWS) -> int:
    if type(value) is not int or value < 0 or value > maximum:
        raise ShadowContractError()
    return value


@dataclass(frozen=True)
class ShadowSummary:
    """One aggregate-only 2x4 result; ties are outside the confusion cells."""

    confusion: tuple[
        tuple[int, int, int, int],
        tuple[int, int, int, int],
    ]
    correct: int
    cross_selected_wrong: int
    out_of_slice: int
    ties: int
    selected_rows: int = SELECTED_ROWS

    def __post_init__(self) -> None:
        if (
            type(self.selected_rows) is not int
            or self.selected_rows != SELECTED_ROWS
            or type(self.confusion) is not tuple
            or len(self.confusion) != len(EXPECTED_ACTION_ORDER)
            or any(type(row) is not tuple or len(row) != 4 for row in self.confusion)
        ):
            raise ShadowContractError()
        for row in self.confusion:
            for value in row:
                _count(value)
        _count(self.correct)
        _count(self.cross_selected_wrong)
        _count(self.out_of_slice)
        _count(self.ties)
        confusion_total = sum(sum(row) for row in self.confusion)
        if (
            confusion_total + self.ties != SELECTED_ROWS
            or self.correct != self.confusion[0][2] + self.confusion[1][3]
            or self.cross_selected_wrong != self.confusion[0][3] + self.confusion[1][2]
            or self.out_of_slice != sum(row[0] + row[1] for row in self.confusion)
            or self.correct + self.cross_selected_wrong + self.out_of_slice + self.ties
            != SELECTED_ROWS
        ):
            raise ShadowContractError()

    def as_dict(self) -> dict[str, object]:
        """Return the complete safe output surface: counts and ordering only."""

        ShadowSummary.__post_init__(self)
        return {
            "kind": SHADOW_SUMMARY_KIND,
            "candidate_id": CANDIDATE_ID,
            "selected_rows": self.selected_rows,
            "expected_action_order": [action.value for action in EXPECTED_ACTION_ORDER],
            "prediction_action_order": [
                action.value for action in PREDICTION_ACTION_ORDER
            ],
            "confusion": [list(row) for row in self.confusion],
            "correct": self.correct,
            "cross_selected_wrong": self.cross_selected_wrong,
            "out_of_slice": self.out_of_slice,
            "ties": self.ties,
        }


def _exact_slots(slots: Sequence[BenchmarkSlot]) -> tuple[BenchmarkSlot, ...]:
    if type(slots) not in {tuple, list} or len(slots) != SELECTED_ROWS:
        raise ShadowContractError()
    retained = tuple(slots)
    expected = _expected_slots()
    for slot, locked in zip(retained, expected, strict=True):
        if (
            type(slot) is not BenchmarkSlot
            or type(slot.ordinal) is not int
            or slot.ordinal != locked.ordinal
            or type(slot.source_row) is not int
            or slot.source_row != locked.source_row
            or type(slot.row_id) is not str
            or slot.row_id != locked.row_id
            or type(slot.action_index) is not int
            or slot.action_index != locked.action_index
            or type(slot.action) is not AnyreachAction
            or slot.action is not locked.action
        ):
            raise ShadowContractError()
    return retained


def _exact_results(results: Sequence[ScoreEvent]) -> tuple[ScoreEvent, ...]:
    if type(results) not in {tuple, list} or len(results) != SELECTED_ROWS:
        raise ShadowContractError()
    retained = tuple(results)
    if any(type(result) is not ScoreEvent for result in retained):
        raise ShadowContractError()
    return retained


def reduce_shadow_results(
    slots: Sequence[BenchmarkSlot],
    results: Sequence[ScoreEvent],
) -> ShadowSummary:
    """Reduce exactly one ordered four-way result for each locked slot."""

    locked_slots = _exact_slots(slots)
    score_events = _exact_results(results)
    request_ids: set[str] = set()
    confusion = [[0, 0, 0, 0], [0, 0, 0, 0]]
    ties = 0

    for slot, event in zip(locked_slots, score_events, strict=True):
        try:
            rebound = ScoreEvent(
                request_id=event.request_id,
                ordinal=event.ordinal,
                logits=event.logits,
                protocol_version=event.protocol_version,
            )
            if (
                type(event.probabilities) is not tuple
                or len(event.probabilities) != len(MODEL_ACTION_ORDER)
                or any(
                    type(value) is not float or not math.isfinite(value)
                    for value in event.probabilities
                )
                or type(event.argmax) is not ArgmaxDecision
            ):
                raise ProtocolError()
            ArgmaxDecision.__post_init__(event.argmax)
        except ProtocolError:
            raise ShadowContractError() from None
        if (
            event.probabilities != rebound.probabilities
            or event.argmax != rebound.argmax
            or rebound.ordinal != slot.ordinal
            or rebound.request_id in request_ids
            or slot.action not in EXPECTED_ACTION_ORDER
        ):
            raise ShadowContractError()
        request_ids.add(rebound.request_id)

        # Recompute both derived values from all four raw logits.  This rejects
        # a forged or stale derived dataclass and makes pair renormalization
        # structurally unavailable to the reducer.
        probabilities = stable_four_way_softmax(rebound.logits)
        argmax = deterministic_argmax(rebound.logits)
        if event.probabilities != probabilities or event.argmax != argmax:
            raise ShadowContractError()
        if argmax.is_tie:
            ties += 1
            continue
        predicted = argmax.unique_action
        if predicted is None:
            raise ShadowContractError()
        expected_index = EXPECTED_ACTION_ORDER.index(slot.action)
        prediction_index = PREDICTION_ACTION_ORDER.index(predicted)
        confusion[expected_index][prediction_index] += 1

    closed_confusion = (
        (
            confusion[0][0],
            confusion[0][1],
            confusion[0][2],
            confusion[0][3],
        ),
        (
            confusion[1][0],
            confusion[1][1],
            confusion[1][2],
            confusion[1][3],
        ),
    )
    return ShadowSummary(
        confusion=closed_confusion,
        correct=closed_confusion[0][2] + closed_confusion[1][3],
        cross_selected_wrong=closed_confusion[0][3] + closed_confusion[1][2],
        out_of_slice=sum(row[0] + row[1] for row in closed_confusion),
        ties=ties,
    )
