"""Classify the data-sensitivity of a user turn so the LLM router can pick
the right cloud-provider chain.

The thinking tier (`docs/target_architecture.md` §9.7) may use cloud LLMs,
but **which** cloud matters: US-hosted providers (Cerebras, Groq) keep data
under US legal jurisdiction; Chinese-hosted providers (DeepSeek, Moonshot)
are cheaper but route post-ASR text through PRC servers. Most personal
queries should never see the cheap-and-Chinese chain; coding tasks and
general-knowledge lookups usually can.

This module produces a `Sensitivity` tag from the user's text + the
existing intent/mode signals. Plain regex heuristics -- safe-by-default
(unknown → ``private``). A learned classifier is a follow-up only if the
heuristic ever mis-routes in practice.
"""
from __future__ import annotations

import re
from typing import Optional

from always_on_agent.events import Mode
from always_on_agent.models import IntentKind

# Sensitivity tag values. Strings rather than an enum so they round-trip
# cleanly through the brain's existing capability ``context`` dict (which is
# already plain dict[str, object]).
PRIVATE = "private"   # default; references personal data, OS actions, dictation
CODE = "code"         # programming / refactoring (code is typically non-PII)
PUBLIC = "public"     # general-knowledge / encyclopedic lookups

Sensitivity = str  # one of PRIVATE / CODE / PUBLIC

# Most-restrictive-wins ordering for egress: PRIVATE never leaves the device, so
# it dominates; CODE may use a coding cloud; PUBLIC is the most permissive.
_SENSITIVITY_RANK = {PUBLIC: 0, CODE: 1, PRIVATE: 2}


def most_sensitive(*levels: Sensitivity) -> Sensitivity:
    """The most-private (most-restrictive) of the given levels.

    Used to float a turn's sensitivity to cover everything in its prompt -- e.g.
    a public current query whose recent-conversation context includes a private
    prior turn must route as private, so the prior turn can't ride to a public
    (possibly cloud) chain (§9.7)."""
    best = PUBLIC
    for level in levels:
        if _SENSITIVITY_RANK.get(level, _SENSITIVITY_RANK[PRIVATE]) > _SENSITIVITY_RANK[best]:
            best = level if level in _SENSITIVITY_RANK else PRIVATE
    return best

# Words that imply the query is about programming work. Code itself rarely
# contains personal info; routing it to a coding-tuned cloud (Cerebras
# qwen-3-coder-480b, Groq Kimi K2) is the cost+quality sweet spot.
_CODE_MARKERS = re.compile(
    r"\b(function|class|def\b|method|module|library|api|"
    r"refactor|debug|implement|compile|build error|stack ?trace|"
    r"unit test|regex|algorithm|data ?structure|"
    r"syntax|typescript|javascript|python|rust|golang|c\+\+|java\b|"
    r"this code|fix (?:this|the) (?:code|bug|function))\b",
    re.IGNORECASE,
)

# Encyclopedic/factual openers — the typical "Wikipedia answer" question
# pattern. Routing these to the cheapest fast cloud (DeepSeek V4-Flash) is
# usually fine because the content isn't about the user.
_PUBLIC_MARKERS = re.compile(
    r"^(what is|what's|who is|who was|when was|when did|"
    r"where is|where was|why does|why is|how does|how is|how many|"
    r"define |explain |tell me about |describe )",
    re.IGNORECASE,
)

# Signals that the query is about the user's personal data even if it
# otherwise looks public. Always wins (drops to PRIVATE).
#
# security-5 (docs/review_ultracode.md): the earlier gate only matched a
# *fixed* noun list after "my", so phrasings like "my coworker John's
# salary" or "what is my coworker John salary" slipped through to the
# cheap PRC-hosted public chain. The §9.7 boundary requires failing safe
# toward PRIVATE, so we now treat *any* first/second-person possessive
# followed (within a few words) by a noun-like token as personal data,
# regardless of which noun it is.

# First/second-person possessives that mark the query as being about the
# speaker's (or the addressed party's) own data. Third-person ("his",
# "her", "their") is intentionally excluded -- it does not by itself imply
# the *speaker's* PII -- but third-person PII is still caught by the
# explicit category markers below (e.g. a name + money amount).
_POSSESSIVE = r"my|mine|our|ours|your|yours"

# A "noun-like" tail: a plain word (optionally possessive, e.g. "John's")
# that is not one of a few function words which would make the match a
# false positive ("my and", "your or"). We allow up to three intervening
# words so naturally-phrased queries still match ("my last doctor
# appointment", "my old email inbox", "my coworker John's salary").
_NOUN_TAIL = r"(?!and\b|or\b|but\b|the\b|a\b|to\b|of\b|is\b|are\b)[a-z]+(?:'s)?"
_POSSESSIVE_NOUN = re.compile(
    rf"\b(?:{_POSSESSIVE})(?:\s+\w+){{0,3}}\s+{_NOUN_TAIL}\b",
    re.IGNORECASE,
)

# Imperative "save it to my world" phrasings that imply storing personal
# data even when no explicit possessive+noun pair is present.
_PERSONAL_ACTIONS = re.compile(
    r"\b(remind me|remember to|add to my|save to|note that)\b",
    re.IGNORECASE,
)

# Explicit PII categories -- fire even without a first/second-person
# possessive, so third-party PII ("John's salary", "Jane lives at ...")
# never reaches a public chain.
#
# Names + money: a Capitalized given name adjacent to a currency/amount or
# the words salary/wage/income/pay. Matched on the ORIGINAL-case text so we
# can use the capitalization signal for a proper name.
_NAME = r"[A-Z][a-z]+(?:'s)?"
_MONEY = (
    r"\$\s?\d|\d+\s?(?:dollars?|euros?|pounds?|usd|eur|gbp)\b|"
    r"\b(?:salary|salaries|wage|wages|income|paycheck|pay ?stub|net worth|bonus)\b"
)
_NAME_AND_MONEY = re.compile(
    rf"(?:{_NAME}(?:\s+\w+){{0,4}}\s+(?:{_MONEY})|"
    rf"(?:{_MONEY})(?:\s+\w+){{0,4}}\s+{_NAME})",
)

# Other obvious PII categories (case-insensitive). Addresses, health, and
# credentials each independently force PRIVATE.
_PII_CATEGORY = re.compile(
    r"\b("
    # credentials / secrets
    r"password|passphrase|passcode|api ?key|secret ?key|access ?token|"
    r"credit ?card|debit ?card|card number|cvv|pin number|ssn|"
    r"social security|bank account|routing number|iban|account number|"
    # addresses / location
    r"home address|street address|mailing address|zip ?code|postal code|"
    r"phone number|date of birth|birthday|"
    # health
    r"diagnosis|diagnosed|prescription|medication|medical record|"
    r"health record|blood pressure|symptoms?|illness|disease"
    r")\b",
    re.IGNORECASE,
)

# Street-address shape: a number followed by a street-type word
# ("123 Main Street", "42 Oak Ave"). Case-insensitive on the street type.
_ADDRESS = re.compile(
    r"\b\d{1,6}\s+\w+(?:\s+\w+)?\s+"
    r"(?:street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr|"
    r"court|ct|place|pl|way|terrace)\b",
    re.IGNORECASE,
)


def _is_personal(text: str) -> bool:
    """Return True when ``text`` references personal/PII data and must be
    kept off any public (potentially PRC-hosted) cloud chain.

    Fail-safe: any of the personal/PII signals below is sufficient.
    """
    if _POSSESSIVE_NOUN.search(text):
        return True
    if _PERSONAL_ACTIONS.search(text):
        return True
    if _NAME_AND_MONEY.search(text):
        return True
    if _PII_CATEGORY.search(text):
        return True
    if _ADDRESS.search(text):
        return True
    return False


def classify_sensitivity(
    query: str,
    *,
    mode: Optional[Mode] = None,
    intent_kind: Optional[IntentKind] = None,
) -> Sensitivity:
    """Return the sensitivity tag for ``query``.

    Order of evaluation (first match wins):

    1. Personal-data / PII markers → ``private`` (overrides everything).
       This includes any first/second-person possessive + noun
       ("my <noun>", "your <noun>") and explicit PII categories
       (name+money, addresses, health, credentials). See ``_is_personal``.
    2. Intent/mode that always implies personal data → ``private``
       (``COMMAND``, ``DICTATION``, ``MEETING_NOTE``, mode=``MEETING``).
    3. Code markers → ``code``.
    4. Public/encyclopedic openers → ``public``.
    5. Otherwise → ``private`` (safe default).

    Fails safe toward ``private`` on any uncertainty so PII never reaches
    a cheap public (potentially PRC-hosted) chain (security-5, §9.7).
    """
    text = (query or "").strip()
    if not text:
        return PRIVATE

    if _is_personal(text):
        return PRIVATE

    if intent_kind in {IntentKind.COMMAND, IntentKind.DICTATION, IntentKind.MEETING_NOTE}:
        return PRIVATE
    if mode == Mode.MEETING:
        return PRIVATE

    if _CODE_MARKERS.search(text):
        return CODE

    if _PUBLIC_MARKERS.match(text):
        return PUBLIC

    return PRIVATE


def may_leave_device(
    query: str,
    *,
    mode: Optional[Mode] = None,
    intent_kind: Optional[IntentKind] = None,
) -> bool:
    """Return True when ``query`` is permitted to egress to the web-search
    surface (the self-hosted SearXNG backend), per the §9.7 data boundary.

    Decision BR3 (LOCKED, "block PII only"): a plain non-PII lookup such as
    "weather in Berlin" may reach external (user-controlled) infra because a
    SEARCH/RESEARCH intent signals the user wants an external lookup, but any
    PII/personal/possessive query is hard-blocked to corpus-only. The PII
    check is ``_is_personal`` -- the P0-hardened detector (security-5) that
    also runs first inside ``classify_sensitivity`` -- so CODE-with-credential
    queries ("debug this, the api key is sk-...") fail closed via the same
    PII precedence (BR5): ``_is_personal`` fires before any CODE marker, so a
    credential phrase never egresses even though CODE != PRIVATE.

    Mode/intent that always implies personal data also blocks egress
    (``Mode.MEETING`` and the ``COMMAND``/``DICTATION``/``MEETING_NOTE``
    intents) -- these are exactly the signals that force PRIVATE in
    ``classify_sensitivity`` above.

    Cross-reference: this gate and ``core.routing.ChainSelector.choose_chain``
    are the two consumers of this module's sensitivity signals -- the gate is
    the binary §9.7 egress decision (leave the device or not), while
    ChainSelector picks *which* cloud chain a permitted turn uses from the
    ``classify_sensitivity`` tag. They share the same PII precedence (run
    ``_is_personal`` first) so they MUST NOT drift: any new PII signal added
    here belongs in ``_is_personal`` so both consumers see it. Callers pass a
    raw ``query`` (NOT a trusted ``context['sensitivity']`` tag, which is only
    set on the assistant path); enum coercion for ``mode``/``intent_kind`` is
    the caller's responsibility and must fail safe to ``None`` (BR2).
    """
    if _is_personal(query or ""):
        return False
    if mode == Mode.MEETING:
        return False
    if intent_kind in {
        IntentKind.COMMAND,
        IntentKind.DICTATION,
        IntentKind.MEETING_NOTE,
    }:
        return False
    return True


__all__ = [
    "PRIVATE",
    "CODE",
    "PUBLIC",
    "Sensitivity",
    "classify_sensitivity",
    "may_leave_device",
]
