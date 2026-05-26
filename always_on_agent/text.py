from __future__ import annotations

import re


_WORD_RE = re.compile(r"[a-z0-9]+")


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    cleaned = text.lower().strip().replace("'", "")
    return " ".join(_WORD_RE.findall(cleaned))


def keywords(text: str, limit: int = 8) -> tuple[str, ...]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "me",
        "of",
        "on",
        "or",
        "please",
        "the",
        "to",
        "what",
        "with",
        "you",
    }
    words = [w for w in normalize_text(text).split() if w not in stopwords]
    result: list[str] = []
    for word in words:
        if word not in result:
            result.append(word)
    return tuple(result[:limit])


def detect_language(text: str) -> str:
    normalized = normalize_text(text)
    romanian_markers = {
        "vreau",
        "cauta",
        "cautare",
        "cerceteaza",
        "pentru",
        "asistent",
        "mod",
        "opreste",
        "anuleaza",
        "scrie",
    }
    english_markers = {
        "assistant",
        "search",
        "research",
        "stop",
        "cancel",
        "dictate",
        "meeting",
        "mode",
    }
    words = set(normalized.split())
    ro_score = len(words & romanian_markers)
    en_score = len(words & english_markers)
    if ro_score > en_score:
        return "ro"
    if en_score > 0:
        return "en"
    return "unknown"
