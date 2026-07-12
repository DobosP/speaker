"""Pure contract tests for the autonomous memory probe's semantic grader."""
from __future__ import annotations

import pytest

from always_on_agent.memory import MemoryItem
from always_on_agent.sqlite_memory import SqliteVecMemory
import tools.autotest.memory_probe as memory_probe
from tools.autotest.memory_probe import (
    _uses_recalled_canary,
    _uses_recalled_fact,
    run_memory_probe,
)


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


@pytest.mark.parametrize(
    "answer",
    (
        "Amber Finch.",
        "It was Amber Finch.",
        "Your lighthouse project codename was Amber Finch.",
        "Amber Finch was the lighthouse project codename.",
    ),
)
def test_cross_session_canary_grader_accepts_scalar_or_grounded_subject(answer):
    assert _uses_recalled_canary(
        answer, "Amber Finch", ("project", "codename")
    )


@pytest.mark.parametrize(
    "answer",
    (
        "The assistant's lighthouse project codename is Amber Finch.",
        "Another project codename is Amber Finch.",
        "Amber Finch is a nice bird.",
        "The lighthouse project codename is Blue Heron.",
    ),
)
def test_cross_session_canary_grader_rejects_wrong_subject_or_value(answer):
    assert not _uses_recalled_canary(
        answer, "Amber Finch", ("project", "codename")
    )


def test_memory_probe_reports_fenced_cross_session_main_route_separately():
    result = run_memory_probe(llm_kind="echo")

    assert result.ok is False
    assert result.plumbing_ok is True
    assert result.complete is False
    assert result.outcome == "diagnostic_pass"
    assert result.recall_available is True
    assert result.recall_injected is True
    assert result.recall_fenced is True
    assert result.answer_uses_fact is None
    assert result.answer_model == "echo"
    assert result.answer_route == "main"
    assert result.topology_valid is True
    assert result.recent_history_clean is True
    assert result.controller_answer is False
    assert result.cross_session is True


def test_memory_probe_fails_if_reopened_rows_leak_into_native_history(monkeypatch):
    def leaky_all(self):
        return [
            MemoryItem(text, tags, ts)
            for text, tags, ts, _embedding in self._recent_rows()
        ]

    monkeypatch.setattr(SqliteVecMemory, "all", leaky_all)
    result = run_memory_probe(llm_kind="echo")

    assert result.ok is False
    assert result.plumbing_ok is False
    assert result.outcome == "fail"
    assert result.recent_history_clean is False
    assert result.cross_session is False


class _CanaryLLM:
    def __init__(self, *, model, **_kwargs):
        self.model = model

    def stream(self, prompt, *, system=None, images=None, history=None):
        if "what was my lighthouse project codename" in prompt.lower():
            yield "Amber Finch."
        else:
            yield "Okay."

    def generate(self, prompt, *, system=None, images=None, history=None):
        return "".join(
            self.stream(prompt, system=system, images=images, history=history)
        )


def test_real_memory_probe_requires_grounded_answer_and_distinct_roles(monkeypatch):
    monkeypatch.setattr(memory_probe, "OllamaLLM", _CanaryLLM)

    result = run_memory_probe(
        llm_kind="ollama",
        model="minicpm-test",
        main_model="gemma-test",
    )
    assert result.ok is True
    assert result.complete is True
    assert result.outcome == "pass"
    assert result.answer_uses_fact is True
    assert result.topology_valid is True

    invalid = run_memory_probe(
        llm_kind="ollama",
        model="minicpm-test",
        main_model="minicpm-test",
    )
    assert invalid.ok is False
    assert invalid.complete is True
    assert invalid.outcome == "fail"
    assert invalid.topology_valid is False
