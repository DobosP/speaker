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

from core.sensitivity import (
    CODE,
    PRIVATE,
    PUBLIC,
    classify_sensitivity,
    may_leave_device,
)


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


def test_public_economics_queries_stay_public():
    """Generic economics/finance questions with no person attached stay
    public -- the name+money PII rule must not swallow encyclopedic facts."""
    assert classify_sensitivity("What is the GDP of France in dollars") == PUBLIC
    assert classify_sensitivity("How many dollars is a Bitcoin worth") == PUBLIC


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


# --- security-5: hardened PII gate (golden cases) --------------------------
# docs/archive/review_ultracode.md security-5: the prior gate matched only a *fixed*
# noun list after "my", so these PII phrasings escaped to the cheap public
# (PRC-hosted) chain. Each must now classify PRIVATE, failing safe per §9.7.


def test_coworker_salary_is_private():
    """The canonical leak from the finding: a possessive + name + money
    phrasing must never reach a public chain, in any of its phrasings."""
    assert classify_sensitivity("my coworker John's salary") == PRIVATE
    assert classify_sensitivity("What is my coworker John salary") == PRIVATE
    assert classify_sensitivity("How much does my coworker John make") == PRIVATE
    # Even with no possessive at all -- name adjacent to money is PII.
    assert classify_sensitivity("What is John's salary") == PRIVATE


def test_possessive_plus_arbitrary_noun_is_private():
    """ANY first/second-person possessive + noun is private, not just the
    historical fixed noun list -- this is the core of the security-5 fix."""
    assert classify_sensitivity("What is my coworker working on") == PRIVATE
    assert classify_sensitivity("Tell me about my landlord") == PRIVATE
    assert classify_sensitivity("Who is my dentist") == PRIVATE
    assert classify_sensitivity("Summarize my mortgage paperwork") == PRIVATE
    # Second-person possessive too.
    assert classify_sensitivity("What is your salary") == PRIVATE
    assert classify_sensitivity("What is in your inbox") == PRIVATE


def test_name_plus_money_is_private():
    """A proper name adjacent to a monetary amount is PII even with no
    possessive (third-party financial info)."""
    assert classify_sensitivity("Jane earns 90000 dollars a year") == PRIVATE
    assert classify_sensitivity("What is the bonus for Michael") == PRIVATE
    assert classify_sensitivity("Pay Sarah $500 next week") == PRIVATE


def test_addresses_are_private():
    assert classify_sensitivity("My home address is 123 Main Street") == PRIVATE
    assert classify_sensitivity("Send the package to 42 Oak Ave") == PRIVATE
    assert classify_sensitivity("What is the zip code on file") == PRIVATE


def test_health_data_is_private():
    assert classify_sensitivity("What was my diagnosis last week") == PRIVATE
    assert classify_sensitivity("List the medication I take") == PRIVATE
    assert classify_sensitivity("Look up the symptoms I described") == PRIVATE


def test_credentials_are_private():
    assert classify_sensitivity("What is the password for the router") == PRIVATE
    assert classify_sensitivity("Store this api key for me") == PRIVATE
    assert classify_sensitivity("What is my credit card number") == PRIVATE
    assert classify_sensitivity("Remember my social security number") == PRIVATE


def test_compensation_terms_fail_safe_private():
    """Bare compensation terms with no public-fact framing fail safe to
    private (uncertainty resolves toward private per the finding)."""
    assert classify_sensitivity("What is the average salary in Germany") == PRIVATE
    assert classify_sensitivity("What is the minimum wage") == PRIVATE


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


# --- §9.7 egress gate (may_leave_device) -----------------------------------
# BR3 (LOCKED, "block PII only"): the web-search surface egress predicate.
# Any PII/personal/possessive query => False (corpus only, no egress); a
# plain non-PII public lookup => True (reaches self-hosted SearXNG). PII
# precedence wins even for CODE-with-credential queries (BR5). MEETING mode
# and COMMAND/DICTATION/MEETING_NOTE intents also block egress.


def test_pii_possessive_query_may_not_leave_device():
    """The canonical leak: a possessive + name + money phrasing is blocked
    from egress, never reaching the web-search backend (BR3)."""
    assert may_leave_device("my coworker John's salary") is False


def test_plain_public_query_may_leave_device():
    """A plain non-PII public lookup is permitted to egress to the
    self-hosted SearXNG backend (BR3 block-PII-only)."""
    assert may_leave_device("weather in Berlin") is True
    assert may_leave_device("who won the 2022 world cup") is True


def test_code_with_credential_may_not_leave_device():
    """PII precedence wins (BR5): a CODE query carrying a credential phrase
    is blocked even though CODE != PRIVATE -- ``_is_personal`` fires first."""
    assert may_leave_device("debug this, the api key is sk-abc123") is False


def test_meeting_mode_may_not_leave_device():
    assert may_leave_device("what is the agenda", mode=Mode.MEETING) is False


def test_command_intent_may_not_leave_device():
    assert (
        may_leave_device("what time is it", intent_kind=IntentKind.COMMAND)
        is False
    )


def test_dictation_and_meeting_note_intents_may_not_leave_device():
    assert (
        may_leave_device("how does this work", intent_kind=IntentKind.DICTATION)
        is False
    )
    assert (
        may_leave_device(
            "summarize the discussion", intent_kind=IntentKind.MEETING_NOTE
        )
        is False
    )


def test_research_intent_does_not_block_public_egress():
    """RESEARCH/SEARCH intent signals the user wants an external lookup, so a
    non-PII query under RESEARCH still egresses (it is not a blocked intent)."""
    assert (
        may_leave_device("what is climate change", intent_kind=IntentKind.RESEARCH)
        is True
    )
