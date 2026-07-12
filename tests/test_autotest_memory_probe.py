"""Pure contract tests for the autonomous memory probe's semantic grader."""
from __future__ import annotations

import pytest

from tools.autotest.memory_probe import _uses_recalled_fact, run_memory_probe


@pytest.mark.parametrize(
    "answer",
    (
        "Teal!",
        "It's teal.",
        "It is TEAL",
        "Your favorite color is teal.",
        "You said your favorite color was teal.",
        "Teal, as you told me.",
        "I remember that your favorite color is teal.",
        "Not blue -- teal.",
    ),
)
def test_recalled_fact_grader_accepts_confident_user_or_scalar_answers(answer):
    assert _uses_recalled_fact(answer, "teal")


@pytest.mark.parametrize(
    "answer",
    (
        "",
        "Your favorite color is blue.",
        "Your favorite color is tealish.",
        "I don't know whether your favorite color is teal.",
        "I cannot confirm teal.",
        "I have no idea; teal?",
        "I don't have a favorite color, but teal is nice.",
        "I do not possess personal tastes; teal is a pleasant choice.",
        "As an AI, I have no personal preferences, though teal sounds good.",
        "I'm an assistant without a favorite; teal is attractive.",
        "I can't be certain it is teal.",
        "Maybe teal.",
        "It might be teal.",
        "I think it's teal.",
        "I doubt it's teal.",
        "I'm not sure, perhaps teal.",
        "Your favorite color is not teal.",
        "It isn't teal.",
        "Teal is not your favorite color.",
        "Teal isn't your favorite color.",
        "Teal? No.",
        "My favorite color is teal.",
        "My favorite color: teal.",
        "My favorite color? Teal.",
        "Teal is my favorite color.",
        "I prefer teal.",
        "I would choose teal.",
        "For me, teal.",
    ),
)
def test_recalled_fact_grader_rejects_missing_uncertain_or_misattributed_answers(answer):
    assert not _uses_recalled_fact(answer, "teal")


def test_recalled_fact_grader_supports_multiword_keywords():
    assert _uses_recalled_fact("Your project codename is Blue Heron.", "blue heron")
    assert not _uses_recalled_fact("Your project codename is blue-heron.", "blue heron")


def test_memory_probe_reports_controller_and_retrieval_separately():
    result = run_memory_probe(llm_kind="echo")

    assert result.ok is True
    assert result.recall_available is True
    assert result.recall_injected is False
    assert result.answer_uses_fact is True
    assert result.answer_model == "control"
    assert result.controller_answer is True
