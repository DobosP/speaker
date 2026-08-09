"""Exact four-way aggregate tests for the Anyreach synthetic-24 reducer."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tools.semantic_interruption import anyreach
from tools.semantic_interruption import protocol
from tools.semantic_interruption import shadow


_SLOTS = anyreach.benchmark_slots(anyreach.load_source_lock())


def _winning_logits(
    action: anyreach.AnyreachAction,
) -> tuple[float, float, float, float]:
    values = [-5.0, -5.0, -5.0, -5.0]
    values[protocol.MODEL_ACTION_ORDER.index(action)] = 5.0
    return (values[0], values[1], values[2], values[3])


def _results_for_predictions(
    predictions: tuple[anyreach.AnyreachAction, ...],
) -> tuple[protocol.ScoreEvent, ...]:
    assert len(predictions) == shadow.SELECTED_ROWS
    return tuple(
        protocol.ScoreEvent(
            request_id=f"case-{slot.ordinal:04d}",
            ordinal=slot.ordinal,
            logits=_winning_logits(prediction),
        )
        for slot, prediction in zip(_SLOTS, predictions, strict=True)
    )


def _uniform_results(
    action: anyreach.AnyreachAction,
) -> tuple[protocol.ScoreEvent, ...]:
    return _results_for_predictions((action,) * shadow.SELECTED_ROWS)


def _correct_results() -> tuple[protocol.ScoreEvent, ...]:
    return _results_for_predictions(tuple(slot.action for slot in _SLOTS))


def test_reducer_binds_the_exact_locked_synthetic24_slots() -> None:
    assert shadow.EXACT_SYNTHETIC24_SLOTS == _SLOTS
    assert len(shadow.EXACT_SYNTHETIC24_SLOTS) == 24
    assert [slot.ordinal for slot in shadow.EXACT_SYNTHETIC24_SLOTS] == list(range(24))
    assert [slot.source_row for slot in shadow.EXACT_SYNTHETIC24_SLOTS] == list(
        range(36, 60)
    )
    assert [slot.action for slot in shadow.EXACT_SYNTHETIC24_SLOTS] == [
        anyreach.AnyreachAction.START_LISTENING
    ] * 12 + [anyreach.AnyreachAction.CONTINUE_SPEAKING] * 12


def test_all_correct_results_form_the_exact_two_by_four_confusion() -> None:
    summary = shadow.reduce_shadow_results(_SLOTS, _correct_results())

    assert shadow.EXPECTED_ACTION_ORDER == (
        anyreach.AnyreachAction.START_LISTENING,
        anyreach.AnyreachAction.CONTINUE_SPEAKING,
    )
    assert shadow.PREDICTION_ACTION_ORDER == (
        anyreach.AnyreachAction.CONTINUE_LISTENING,
        anyreach.AnyreachAction.START_SPEAKING,
        anyreach.AnyreachAction.START_LISTENING,
        anyreach.AnyreachAction.CONTINUE_SPEAKING,
    )
    assert summary.confusion == ((0, 0, 12, 0), (0, 0, 0, 12))
    assert summary.correct == 24
    assert summary.cross_selected_wrong == 0
    assert summary.out_of_slice == 0
    assert summary.ties == 0


@pytest.mark.parametrize(
    ("prediction", "confusion", "correct", "cross_wrong", "out_of_slice"),
    [
        (
            anyreach.AnyreachAction.CONTINUE_LISTENING,
            ((12, 0, 0, 0), (12, 0, 0, 0)),
            0,
            0,
            24,
        ),
        (
            anyreach.AnyreachAction.START_SPEAKING,
            ((0, 12, 0, 0), (0, 12, 0, 0)),
            0,
            0,
            24,
        ),
        (
            anyreach.AnyreachAction.START_LISTENING,
            ((0, 0, 12, 0), (0, 0, 12, 0)),
            12,
            12,
            0,
        ),
        (
            anyreach.AnyreachAction.CONTINUE_SPEAKING,
            ((0, 0, 0, 12), (0, 0, 0, 12)),
            12,
            12,
            0,
        ),
    ],
)
def test_each_four_way_prediction_column_is_counted_without_collapsing(
    prediction: anyreach.AnyreachAction,
    confusion: tuple[tuple[int, int, int, int], tuple[int, int, int, int]],
    correct: int,
    cross_wrong: int,
    out_of_slice: int,
) -> None:
    summary = shadow.reduce_shadow_results(_SLOTS, _uniform_results(prediction))

    assert summary.confusion == confusion
    assert summary.correct == correct
    assert summary.cross_selected_wrong == cross_wrong
    assert summary.out_of_slice == out_of_slice
    assert summary.ties == 0


def test_selected_pair_is_never_renormalized_away_from_four_way_argmax() -> None:
    # Within the selected pair, SLi beats CS.  Across the real four-way output,
    # CL is the unique winner and must remain an out-of-slice error.
    logits = (100.0, -100.0, 10.0, 9.0)
    results = tuple(
        protocol.ScoreEvent(f"case-{slot.ordinal:04d}", slot.ordinal, logits)
        for slot in _SLOTS
    )

    summary = shadow.reduce_shadow_results(_SLOTS, results)

    assert summary.confusion == ((12, 0, 0, 0), (12, 0, 0, 0))
    assert summary.out_of_slice == 24
    assert summary.correct == 0
    assert summary.cross_selected_wrong == 0


def test_exact_ties_are_explicit_and_excluded_from_confusion_cells() -> None:
    results = tuple(
        protocol.ScoreEvent(
            f"case-{slot.ordinal:04d}",
            slot.ordinal,
            (1.0, 1.0, 1.0, 1.0),
        )
        for slot in _SLOTS
    )

    summary = shadow.reduce_shadow_results(_SLOTS, results)

    assert summary.confusion == ((0, 0, 0, 0), (0, 0, 0, 0))
    assert summary.correct == 0
    assert summary.cross_selected_wrong == 0
    assert summary.out_of_slice == 0
    assert summary.ties == 24


def test_mixed_unique_and_tied_results_close_to_exactly_24() -> None:
    results = list(_correct_results())
    results[0] = protocol.ScoreEvent("case-0000", 0, (8.0, 8.0, 0.0, 0.0))
    results[12] = protocol.ScoreEvent("case-0012", 12, (8.0, 0.0, 8.0, 0.0))

    summary = shadow.reduce_shadow_results(_SLOTS, results)

    assert summary.confusion == ((0, 0, 11, 0), (0, 0, 0, 11))
    assert summary.correct == 22
    assert summary.ties == 2
    assert (
        summary.correct
        + summary.cross_selected_wrong
        + summary.out_of_slice
        + summary.ties
        == 24
    )


def test_extreme_finite_logits_reduce_without_overflow() -> None:
    results = tuple(
        protocol.ScoreEvent(
            f"case-{slot.ordinal:04d}",
            slot.ordinal,
            (-1.0e308, -1.0e308, 1.0e308, -1.0e308),
        )
        for slot in _SLOTS
    )

    summary = shadow.reduce_shadow_results(_SLOTS, results)

    assert summary.confusion == ((0, 0, 12, 0), (0, 0, 12, 0))
    assert summary.correct == 12
    assert summary.cross_selected_wrong == 12


@pytest.mark.parametrize(
    "field", ["ordinal", "source_row", "row_id", "action_index", "action"]
)
def test_any_locked_slot_change_fails_closed(field: str) -> None:
    slots = list(_SLOTS)
    replacements: dict[str, object] = {
        "ordinal": 1,
        "source_row": 35,
        "row_id": "other",
        "action_index": 3,
        "action": anyreach.AnyreachAction.CONTINUE_SPEAKING,
    }
    slots[0] = replace(slots[0], **{field: replacements[field]})

    with pytest.raises(shadow.ShadowContractError):
        shadow.reduce_shadow_results(slots, _correct_results())


def test_coercively_equal_locked_slot_field_types_fail_closed() -> None:
    slots = list(_SLOTS)
    slots[0] = replace(
        slots[0],
        ordinal=False,
        source_row=36.0,
        action_index=2.0,
        action="start_listening",
    )

    with pytest.raises(shadow.ShadowContractError):
        shadow.reduce_shadow_results(slots, _correct_results())


def test_missing_extra_or_reordered_results_fail_closed() -> None:
    results = list(_correct_results())

    with pytest.raises(shadow.ShadowContractError):
        shadow.reduce_shadow_results(_SLOTS, results[:-1])
    with pytest.raises(shadow.ShadowContractError):
        shadow.reduce_shadow_results(_SLOTS, [*results, results[-1]])
    results[0], results[1] = results[1], results[0]
    with pytest.raises(shadow.ShadowContractError):
        shadow.reduce_shadow_results(_SLOTS, results)


def test_duplicate_request_id_and_wrong_ordinal_fail_closed() -> None:
    duplicate_ids = list(_correct_results())
    duplicate_ids[1] = replace(duplicate_ids[1], request_id=duplicate_ids[0].request_id)
    wrong_ordinal = list(_correct_results())
    wrong_ordinal[1] = replace(wrong_ordinal[1], ordinal=0)

    with pytest.raises(shadow.ShadowContractError):
        shadow.reduce_shadow_results(_SLOTS, duplicate_ids)
    with pytest.raises(shadow.ShadowContractError):
        shadow.reduce_shadow_results(_SLOTS, wrong_ordinal)


def test_reducer_recomputes_and_rejects_forged_derived_score_state() -> None:
    results = list(_correct_results())
    object.__setattr__(results[0], "probabilities", (0.25, 0.25, 0.25, 0.25))

    with pytest.raises(shadow.ShadowContractError):
        shadow.reduce_shadow_results(_SLOTS, results)


def test_reducer_rejects_coercively_equal_derived_score_types() -> None:
    integer_probabilities = list(_correct_results())
    object.__setattr__(integer_probabilities[0], "probabilities", (0, 0, 1, 0))
    string_argmax = list(_correct_results())
    object.__setattr__(
        string_argmax[0].argmax,
        "unique_action",
        "start_listening",
    )

    with pytest.raises(shadow.ShadowContractError):
        shadow.reduce_shadow_results(_SLOTS, integer_probabilities)
    with pytest.raises(shadow.ShadowContractError):
        shadow.reduce_shadow_results(_SLOTS, string_argmax)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("request_id", "PRIVATE INPUT"),
        ("ordinal", True),
        ("logits", (float("inf"), 0.0, 0.0, 0.0)),
        ("protocol_version", 2),
    ],
)
def test_reducer_revalidates_every_forged_score_event_field(
    field: str,
    value: object,
) -> None:
    results = list(_correct_results())
    object.__setattr__(results[0], field, value)

    with pytest.raises(shadow.ShadowContractError):
        shadow.reduce_shadow_results(_SLOTS, results)


def test_reducer_rejects_unbounded_iterable_inputs_before_iteration() -> None:
    class Endless:
        def __iter__(self):  # type: ignore[no-untyped-def]
            while True:
                yield _SLOTS[0]

    with pytest.raises(shadow.ShadowContractError):
        shadow.reduce_shadow_results(Endless(), _correct_results())  # type: ignore[arg-type]
    with pytest.raises(shadow.ShadowContractError):
        shadow.reduce_shadow_results(_SLOTS, Endless())  # type: ignore[arg-type]


def test_summary_surface_is_aggregate_only() -> None:
    private_fragments = {
        "manual_sli_01",
        "manual_cs_12",
        "case-0000",
        "/home/paul/private",
        "private benchmark text",
    }
    summary = shadow.reduce_shadow_results(_SLOTS, _correct_results())
    value = summary.as_dict()
    rendered = repr(value)

    assert set(value) == {
        "kind",
        "candidate_id",
        "selected_rows",
        "expected_action_order",
        "prediction_action_order",
        "confusion",
        "correct",
        "cross_selected_wrong",
        "out_of_slice",
        "ties",
    }
    assert value["kind"] == shadow.SHADOW_SUMMARY_KIND
    assert value["candidate_id"] == anyreach.CANDIDATE_ID
    assert not any(fragment in rendered for fragment in private_fragments)
    assert "logit" not in rendered
    assert "probabil" not in rendered
    assert "takeover" not in rendered


def test_summary_constructor_rejects_nonclosing_counts() -> None:
    with pytest.raises(shadow.ShadowContractError):
        shadow.ShadowSummary(
            confusion=((0, 0, 12, 0), (0, 0, 0, 12)),
            correct=23,
            cross_selected_wrong=0,
            out_of_slice=0,
            ties=0,
        )


def test_summary_serialization_revalidates_forged_counts() -> None:
    summary = shadow.reduce_shadow_results(_SLOTS, _correct_results())
    object.__setattr__(summary, "correct", 23)
    object.__setattr__(summary, "__post_init__", lambda: None)

    with pytest.raises(shadow.ShadowContractError):
        summary.as_dict()


def test_reducer_imports_no_policy_runtime_or_effect_module() -> None:
    source = Path(shadow.__file__).read_text(encoding="utf-8")

    assert "always_on_agent" not in source
    assert "interruption_policy" not in source
    assert "onnxruntime" not in source
    assert "tokenizers" not in source
    assert "pyarrow" not in source
