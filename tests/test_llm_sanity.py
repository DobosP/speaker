"""Fast unit tests for the LLM output-degeneracy heuristics.

These pin the detection logic with no model (the real fast Gemma run on CPU lives
in tools/llm_sanity.main, exercised by the llm-sanity workflow)."""
from __future__ import annotations

from tools.llm_sanity import _check, looks_degenerate, outputs_distinct


def test_flags_single_token_spam():
    assert looks_degenerate("okay okay okay okay okay okay")


def test_flags_phrase_loop():
    assert looks_degenerate("let's do this let's do this let's do this")


def test_flags_empty():
    assert looks_degenerate("")


def test_accepts_normal_answer():
    assert not looks_degenerate("The capital of France is Paris.")


def test_accepts_short_nonempty_answer():
    assert not looks_degenerate("Four.")


def test_outputs_distinct_true_for_varied_answers():
    assert outputs_distinct(["Paris.", "Here's a joke.", "Four."])


def test_outputs_distinct_false_when_all_identical():
    assert not outputs_distinct(
        ["Okay, let's do this!", "okay, let's do this!", "Okay, let's do this!"]
    )


def test_check_flags_canned_repeated_answers():
    pairs = [("q1", "Okay, let's do this!"), ("q2", "Okay, let's do this!")]
    assert _check(pairs)  # all-same answers -> flagged


def test_check_passes_healthy_answers():
    pairs = [
        ("capital of France", "The capital of France is Paris."),
        ("2+2", "Two plus two is four."),
    ]
    assert _check(pairs) == []
