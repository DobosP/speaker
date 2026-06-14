"""Procedural memory: persisted, user-taught BEHAVIOR rules.

The third memory class alongside semantic (the user-profile facts) and episodic
(messages/summaries): standing directives the user teaches the assistant about HOW
to behave -- "always answer in one sentence", "call me Sam", "never read long URLs
aloud". Unlike episodic recall (query-gated, retrieved by relevance), procedural
rules are FEW, DURABLE, and injected on EVERY turn as a small instruction block.

This module is stdlib-only + backend-neutral (importable by the brain and core,
like ``recall.py``/``untrusted.py``):

* :func:`extract_rule` -- a high-precision, deterministic detector that turns an
  explicit teach utterance ("from now on, X" / "always X" / "never X" / "call me
  X") into a normalized rule string, or ``None``. Tight patterns + a min-content
  guard keep ordinary speech from being mistaken for a directive.
* :func:`render_rules` -- the bounded, ALWAYS-injected instruction block (trusted:
  the user authored these, so unlike recalled/screen/web content they are NOT
  spotlighted as untrusted).
"""
from __future__ import annotations

import re
from typing import Optional, Sequence

from .text import normalize_text

PROCEDURAL_HEADER = "Standing user preferences (always follow these):"

_MIN_RULE_WORDS = 2

# Capture requires an EXPLICIT teaching FRAME (not a bare "always X") so ordinary
# speech can't become a durable rule -- "always sunny in philadelphia" / "never
# gonna give you up" have no frame and are ignored; "from now on keep it short" /
# "please always answer briefly" / "I want you to use metric" do.
_NAME_RE = re.compile(
    r"^(?:call me|address me as|i'?d? ?(?:would )?like to be called)\s+(.+)", re.IGNORECASE
)
# A directive body (group 1) carried by an unambiguous instruction frame. Optional
# trailing words use the LEADING-space form ``(?: word)?`` so they never consume the
# separator that the ``[,:]?\s+`` after the frame needs.
_FRAME_RE = re.compile(
    r"^(?:"
    r"from now on|going forward|from here on|in future|"
    r"remember to(?: always)?|make sure (?:to|you)(?: always)?|be sure to(?: always)?|"
    r"i (?:want|need|expect) you to(?: always)?|i'?d? ?(?:would )?like you to(?: always)?|"
    r"(?:can|could|would) you(?: please)?(?: always)?|please (?:can|could) you(?: always)?"
    r")[,:]?\s+(?:please\s+)?(.+)",
    re.IGNORECASE,
)
# "please always/never X" -> force the Always/Never prefix on the body.
_PLEASE_AN_RE = re.compile(r"^please\s+(always|never)\s+(.+)", re.IGNORECASE)

# A directive that STARTS with one of these is a statement of fact / a content
# request, not a behavior rule -- reject so "from now on I have a dentist
# appointment" or "I want you to know the news" never becomes a rule.
_STATEMENT_OPENERS = frozenset({
    "i", "i'm", "im", "we", "he", "she", "it", "they", "you", "there",
    "the", "a", "an", "my", "your", "our", "this", "that", "these", "those",
})
_CONTENT_VERBS = frozenset({
    "tell", "show", "give", "find", "search", "look", "get", "remind", "list",
    "play", "send", "order", "buy", "book", "fetch", "open", "read",  # content requests, not HOW rules
    "know", "remember", "watch", "check", "monitor", "track", "learn",  # info/recall requests
})
# Non-name continuations of "call me ..." (idioms / actions, never a name).
_NAME_DENY = frozenset({
    "later", "back", "soon", "now", "an", "a", "the", "when", "if", "maybe",
    "asap", "tomorrow", "tonight", "today", "that", "this", "crazy", "up",
    "down", "in", "out", "please", "right", "after", "before", "once",
})


def _clean(text: str) -> str:
    return " ".join((text or "").split()).strip(" .,:;!?")


def _finalize(rule: str) -> str:
    return rule.rstrip(".") + "." if rule and not rule.endswith(".") else rule


def _directive_rule(body: str, *, force: Optional[str] = None) -> Optional[str]:
    """Normalize a frame's directive body into an imperative rule, or ``None`` if it
    is a statement/content-request rather than a behavior directive."""
    body = _clean(body)
    words = body.split()
    if len(words) < _MIN_RULE_WORDS:
        return None
    head = words[0].lower().strip(".,!?'")
    if head in _STATEMENT_OPENERS:
        return None  # a statement of fact, not a HOW directive
    if head in _CONTENT_VERBS:
        return None  # a content/recall request ("tell me the weather"), not a behavior rule
    if force == "always":
        return _finalize(f"Always {body}")
    if force == "never":
        return _finalize(f"Never {body}")
    return _finalize(body[0].upper() + body[1:])


def extract_rule(text: str) -> Optional[str]:
    """Return a normalized procedural rule from an EXPLICIT teach utterance, or
    ``None`` when the text is not a clear behavior directive.

    Deliberately high-precision (a missed rule is re-teachable; a wrongly-captured
    one pollutes the durable instruction block forever): it requires a teaching
    FRAME and rejects statement/content-request bodies and non-name "call me ..."
    continuations.

    Examples -> rule: "from now on keep answers short" -> "Keep answers short.";
    "please always answer in one sentence" -> "Always answer in one sentence.";
    "I want you to use metric units" -> "Use metric units."; "call me Sam" ->
    "Address the user as Sam.". Ignored: "always sunny in philadelphia", "never
    gonna give you up", "from now on I have a meeting", "call me an ambulance"."""
    if not text:
        return None
    raw = text.strip()

    m = _NAME_RE.match(raw)
    if m:
        rest = _clean(m.group(1)).split()
        if not rest:
            return None
        name = rest[0].strip(".,!?'")
        if not name or not name[0].isalpha() or name.lower() in _NAME_DENY:
            return None  # an idiom/action ("call me later/an ambulance"), not a name
        return f"Address the user as {name[:1].upper() + name[1:]}."

    m = _PLEASE_AN_RE.match(raw)
    if m:
        return _directive_rule(m.group(2), force=m.group(1).lower())

    m = _FRAME_RE.match(raw)
    if m:
        low = raw.lower()
        force = "always" if " always" in low[: m.start(1)] or low.startswith("always") else None
        return _directive_rule(m.group(1), force=force)
    return None


def render_rules(rules: Sequence[str], *, max_rules: int = 12) -> str:
    """A bounded, deduped, ALWAYS-injected instruction block, or ``''`` when there
    are no rules. The MOST RECENT rules win (rules are passed most-recent-first);
    duplicates (normalized) are dropped. Trusted -- the user authored these."""
    seen: set[str] = set()
    kept: list[str] = []
    for r in rules:
        r = " ".join((r or "").split())  # collapse whitespace; keep the rule's own punctuation
        if not r:
            continue
        key = normalize_text(r)  # dedup is punctuation/case-insensitive
        if key in seen:
            continue
        seen.add(key)
        kept.append(r)
        if len(kept) >= max_rules:
            break
    if not kept:
        return ""
    return PROCEDURAL_HEADER + "\n" + "\n".join(f"- {r}" for r in kept)
