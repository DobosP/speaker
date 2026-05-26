"""Pure predicates for voice-assistant response quality.

These encode the assistant's stated output contract (LocalLLM.SYSTEM_PROMPT:
"1-2 sentences max, no emoji, no filler") as deterministic checks that can be
applied to real model output. Being pure, they are also unit-testable on
fabricated strings with no model in the loop.
"""
from __future__ import annotations

import re

_EMOJI_RE = re.compile(
    "["
    "\U0001f600-\U0001f64f"
    "\U0001f300-\U0001f5ff"
    "\U0001f680-\U0001f6ff"
    "\U0001f1e0-\U0001f1ff"
    "\U00002702-\U000027b0"
    "\U000024c2-\U0001f251"
    "]+",
    flags=re.UNICODE,
)

# Filler openers LocalLLM._postprocess is meant to strip.
_BANNED_PREFIXES = (
    "ah, ",
    "i see! ",
    "great! ",
    "of course! ",
    "well, ",
    "you know, ",
    "basically, ",
)


def has_emoji(text: str) -> bool:
    return bool(_EMOJI_RE.search(text or ""))


def count_sentences(text: str) -> int:
    parts = [p for p in re.split(r"[.!?]+", text or "") if p.strip()]
    return len(parts)


def starts_with_filler(text: str) -> bool:
    low = (text or "").lower()
    return any(low.startswith(p) for p in _BANNED_PREFIXES)


def voice_format_violations(
    text: str, *, max_sentences: int = 3, max_chars: int = 240
) -> list[str]:
    """Return a list of contract violations (empty list == compliant).

    Bounds are generous on purpose: the goal is to catch gross violations
    (emoji, essays, filler), not to penalize a model for an extra clause.
    """
    issues: list[str] = []
    t = (text or "").strip()
    if not t:
        issues.append("empty")
        return issues
    if has_emoji(t):
        issues.append("emoji")
    if starts_with_filler(t):
        issues.append("filler_prefix")
    if count_sentences(t) > max_sentences:
        issues.append(f"too_many_sentences({count_sentences(t)})")
    if len(t) > max_chars:
        issues.append(f"too_long({len(t)})")
    return issues


def is_voice_format_ok(text: str, **kwargs) -> bool:
    return not voice_format_violations(text, **kwargs)
