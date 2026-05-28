"""Tests for core.sensitivity: heuristic data-sensitivity classifier.

The classifier maps a query (+ optional Mode and IntentKind) to one of
``private`` / ``code`` / ``public``; the LLM router uses this to pick a
cloud-provider chain (US-only for private, CN-OK for public).

Order of evaluation is critical -- personal-data markers must override
both code AND public markers, so the tests assert each branch in
isolation plus a few combinations."""
from __future__ import annotations

from always_on_agent.events import Mode
from always_on_agent.models import IntentKind

from core.sensitivity import CODE, PRIVATE, PUBLIC, classify_sensitivity


# --- defaults --------------------------------------------------------------


def test_empty_query_defaults_to_private():
    assert classify_sensitivity("") == PRIVATE
    assert classify_sensitivity("   ") == PRIVATE


def test_unknown_text_defaults_to_private():
    assert classify_sensitivity("blah blah") == PRIVATE
    assert classify_sensitivity("the cat sat on the mat") == PRIVATE


# --- public branch ---------------------------------------------------------


def test_factual_openers_are_public():
    assert classify_sensitivity("What is the boiling point of water?") == PUBLIC
    assert classify_sensitivity("Who is the president of France?") == PUBLIC
    assert classify_sensitivity("When was the Eiffel Tower built?") == PUBLIC
    assert classify_sensitivity("How does photosynthesis work?") == PUBLIC
    assert classify_sensitivity("Explain quantum entanglement briefly") == PUBLIC


def test_public_with_personal_marker_falls_to_private():
    """Public-style opener referencing 'my X' is still private."""
    assert classify_sensitivity("What is in my notes about the meeting") == PRIVATE
    assert classify_sensitivity("When was my last doctor appointment") == PRIVATE


# --- code branch -----------------------------------------------------------


def test_code_keywords_route_to_code():
    assert classify_sensitivity("Refactor this function") == CODE
    assert classify_sensitivity("Debug the stacktrace I just pasted") == CODE
    assert classify_sensitivity("Write a Python class for this") == CODE
    assert classify_sensitivity("Fix the bug in this code") == CODE


def test_code_with_personal_marker_falls_to_private():
    """A coding request that mentions 'my password' (etc.) is still private."""
    assert classify_sensitivity("Refactor my password manager script") == PRIVATE


# --- private explicit ------------------------------------------------------


def test_personal_markers_force_private():
    assert classify_sensitivity("Remind me to buy milk") == PRIVATE
    assert classify_sensitivity("Add this to my notes") == PRIVATE
    assert classify_sensitivity("Email my wife about dinner") == PRIVATE
    assert classify_sensitivity("Read me my schedule") == PRIVATE
    assert classify_sensitivity("What's in my inbox") == PRIVATE


# --- mode + intent overrides -----------------------------------------------


def test_command_intent_routes_private_regardless_of_text():
    # COMMAND mode often controls personal devices; always private.
    assert classify_sensitivity(
        "Open the door", intent_kind=IntentKind.COMMAND
    ) == PRIVATE
    # Even a factual-looking command stays private:
    assert classify_sensitivity(
        "What time is it", intent_kind=IntentKind.COMMAND
    ) == PRIVATE


def test_dictation_intent_routes_private():
    assert classify_sensitivity(
        "How does this work", intent_kind=IntentKind.DICTATION
    ) == PRIVATE


def test_meeting_mode_routes_private():
    # Meeting notes are PII even if they contain code or factual content.
    assert classify_sensitivity(
        "What is the agenda", mode=Mode.MEETING
    ) == PRIVATE


def test_research_intent_does_not_change_classification():
    # RESEARCH intent affects tier (main vs fast) but not sensitivity --
    # research on "what is climate change" can use the public chain.
    assert classify_sensitivity(
        "What is climate change", intent_kind=IntentKind.RESEARCH
    ) == PUBLIC


# --- API completeness ------------------------------------------------------


def test_classify_returns_valid_sensitivity_strings():
    """The output is always one of the documented constants."""
    valid = {PRIVATE, CODE, PUBLIC}
    samples = [
        "",
        "what is x",
        "refactor this function",
        "my email",
        "blah blah",
    ]
    for s in samples:
        assert classify_sensitivity(s) in valid
