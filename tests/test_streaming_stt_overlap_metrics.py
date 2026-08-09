from __future__ import annotations

import json
from dataclasses import replace

import pytest

from tools.streaming_stt.overlap_metrics import (
    PROTOCOL_VERSION,
    OverlapMetricAggregate,
    OverlapMetricInput,
    aggregate_two_utterance_min_order_wer,
    score_overlap_metric,
    score_two_utterance_min_order_wer,
)


def test_order_inversion_selects_reverse_reference_order_without_text_output():
    reference_a = "SENTINEL_PRIVATE_ALPHA,"
    reference_b = "SENTINEL_PRIVATE_BRAVO!"
    hypothesis = "sentinel_private_bravo sentinel_private_alpha"

    result = score_two_utterance_min_order_wer(
        reference_a,
        reference_b,
        hypothesis,
    )
    encoded = json.dumps(result.as_dict(), sort_keys=True)

    assert result.protocol == PROTOCOL_VERSION
    assert result.selected_order == "role_b_then_role_a"
    assert result.word_errors == 0
    assert result.reference_words == 6
    assert result.hypothesis_words == 6
    assert result.wer == 0.0
    assert reference_a not in encoded
    assert reference_b not in encoded
    assert hypothesis not in encoded
    assert "reference_a" not in encoded
    assert "reference_b" not in encoded


def test_one_role_omission_is_counted_as_deletions():
    result = score_two_utterance_min_order_wer(
        "alpha beta",
        "gamma delta",
        "alpha beta",
    )

    assert result.selected_order == "role_a_then_role_b"
    assert result.substitutions == 0
    assert result.insertions == 0
    assert result.deletions == 2
    assert result.word_errors == 2
    assert result.reference_words == 4
    assert result.hypothesis_words == 2
    assert result.wer == 0.5


def test_equal_edit_cost_tie_deterministically_selects_role_a_first():
    result = score_two_utterance_min_order_wer("one", "two", "three")

    assert result.selected_order == "role_a_then_role_b"
    assert result.word_errors == 2
    assert result.reference_words == 2
    assert result.hypothesis_words == 1
    assert result.wer == 1.0


def test_empty_references_and_hypothesis_have_zero_wer():
    result = score_two_utterance_min_order_wer("", "", "")
    aggregate = aggregate_two_utterance_min_order_wer(())

    assert result.selected_order == "role_a_then_role_b"
    assert result.as_dict() == {
        "protocol": PROTOCOL_VERSION,
        "selected_order": "role_a_then_role_b",
        "substitutions": 0,
        "insertions": 0,
        "deletions": 0,
        "word_errors": 0,
        "reference_words": 0,
        "hypothesis_words": 0,
        "wer": 0.0,
    }
    assert aggregate.as_dict() == {
        "protocol": PROTOCOL_VERSION,
        "cases": 0,
        "substitutions": 0,
        "insertions": 0,
        "deletions": 0,
        "word_errors": 0,
        "reference_words": 0,
        "hypothesis_words": 0,
        "wer": 0.0,
    }


def test_empty_reference_with_words_uses_core_wer_insertion_semantics():
    result = score_two_utterance_min_order_wer("", "", "unexpected words")

    assert result.substitutions == 0
    assert result.insertions == 2
    assert result.deletions == 0
    assert result.reference_words == 0
    assert result.hypothesis_words == 2
    assert result.wer == 1.0


def test_aggregate_sums_integer_counts_before_deriving_wer():
    exact_two_words = score_two_utterance_min_order_wer(
        "one two",
        "",
        "one two",
    )
    one_substitution = score_two_utterance_min_order_wer(
        "three",
        "",
        "wrong",
    )

    aggregate = aggregate_two_utterance_min_order_wer(
        (exact_two_words, one_substitution)
    )

    assert aggregate.cases == 2
    assert aggregate.substitutions == 1
    assert aggregate.insertions == 0
    assert aggregate.deletions == 0
    assert aggregate.word_errors == 1
    assert aggregate.reference_words == 3
    assert aggregate.hypothesis_words == 3
    assert aggregate.wer == 1 / 3
    assert aggregate.wer != (exact_two_words.wer + one_substitution.wer) / 2


def test_private_input_repr_is_redacted_and_convenience_paths_match():
    value = OverlapMetricInput(
        reference_a="SENTINEL_PRIVATE_A",
        reference_b="SENTINEL_PRIVATE_B",
        hypothesis="SENTINEL_PRIVATE_HYPOTHESIS",
    )

    assert repr(value) == "OverlapMetricInput(<private>)"
    assert score_overlap_metric(value) == score_two_utterance_min_order_wer(
        value.reference_a,
        value.reference_b,
        value.hypothesis,
    )


@pytest.mark.parametrize(
    ("reference_a", "reference_b", "hypothesis"),
    [
        (None, "valid", "valid"),
        ("valid", 1, "valid"),
        ("valid", "valid", b"invalid"),
        ("valid", "valid", float("nan")),
    ],
)
def test_invalid_input_types_fail_without_echoing_values(
    reference_a: object,
    reference_b: object,
    hypothesis: object,
):
    with pytest.raises(ValueError, match="^invalid overlap metric input$"):
        score_two_utterance_min_order_wer(  # type: ignore[arg-type]
            reference_a,
            reference_b,
            hypothesis,
        )


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_result_wer_is_rejected(nonfinite: float):
    result = score_two_utterance_min_order_wer("one", "two", "one two")

    with pytest.raises(ValueError, match="^invalid overlap metric counts$"):
        replace(result, wer=nonfinite)


def test_invalid_result_shapes_and_aggregate_items_are_rejected():
    result = score_two_utterance_min_order_wer("one", "two", "one two")

    with pytest.raises(ValueError, match="^invalid overlap metric counts$"):
        replace(result, substitutions=True)
    with pytest.raises(ValueError, match="^invalid overlap metric counts$"):
        replace(result, word_errors=1)
    with pytest.raises(ValueError, match="^invalid overlap metric result$"):
        replace(result, selected_order="unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="^invalid overlap metric results$"):
        aggregate_two_utterance_min_order_wer([object()])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="^invalid overlap metric results$"):
        aggregate_two_utterance_min_order_wer("invalid")  # type: ignore[arg-type]


def test_zero_case_aggregate_cannot_carry_forged_counts():
    with pytest.raises(ValueError, match="^invalid overlap metric aggregate$"):
        OverlapMetricAggregate(
            protocol=PROTOCOL_VERSION,
            cases=0,
            substitutions=1,
            insertions=0,
            deletions=0,
            word_errors=1,
            reference_words=1,
            hypothesis_words=1,
            wer=1.0,
        )
