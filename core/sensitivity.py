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
# otherwise looks public. Always wins (drops to PRIVATE). Allows up to
# three intervening words ("my last doctor appointment", "my old email
# inbox") so naturally-phrased queries still match.
_PERSONAL_NOUNS = (
    r"notes?|memos?|calendar|schedule|appointment|family|wife|husband|"
    r"kids?|children|email|inbox|messages?|texts?|chats?|password|"
    r"account|bank|address|home|location|health|doctor|medical"
)
_PERSONAL_MARKERS = re.compile(
    rf"\bmy(?:\s+\w+){{0,3}}\s+(?:{_PERSONAL_NOUNS})\b|"
    rf"\b(remind me|remember to|add to my|save to|note that)\b",
    re.IGNORECASE,
)


def classify_sensitivity(
    query: str,
    *,
    mode: Optional[Mode] = None,
    intent_kind: Optional[IntentKind] = None,
) -> Sensitivity:
    """Return the sensitivity tag for ``query``.

    Order of evaluation (first match wins):

    1. Personal-data markers → ``private`` (overrides everything).
    2. Intent/mode that always implies personal data → ``private``
       (``COMMAND``, ``DICTATION``, ``MEETING_NOTE``, mode=``MEETING``).
    3. Code markers → ``code``.
    4. Public/encyclopedic openers → ``public``.
    5. Otherwise → ``private`` (safe default).
    """
    text = (query or "").strip()
    if not text:
        return PRIVATE

    if _PERSONAL_MARKERS.search(text):
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


__all__ = ["PRIVATE", "CODE", "PUBLIC", "Sensitivity", "classify_sensitivity"]
