"""Transcript cleanup: rewrites a raw ASR final into the user's *intended*
sentence -- disfluencies removed, self-corrections resolved -- so the brain
acts on what the user meant, not on every "um" and false start.

The technique is the post-ASR LLM pass that voice products like Claude's
voice mode, Voicebox, and assorted research tooling settle on (see Shing
Lyu's "Using LLM to get cleaner voice transcriptions", 2024; arXiv
2506.18510 "Smooth Operators"; cs/0008016 "Processing Self Corrections in
a speech to speech system"). Linguistically the relevant signals are
*editing terms* ("I mean", "no wait"), *word-level repairs*, and the
**word-repeat-at-end** pattern -- the user said it, heard the recognizer
catch it wrong, and repeated the last word as a correction marker. The
prompt explicitly enumerates these so a small fast-tier model
(gemma3:1b-4b class) can apply them.

The runtime calls :class:`TranscriptCleaner` after the addressing gate
(``core/addressing.py``) says ACT and before publishing to the brain bus,
so an ingested ambient utterance never pays the cleanup cost.

UX policy (decided 2026-05): the raw final AND the cleaned final are
both logged at INFO; the run bundle's transcript stores both entries
side-by-side so the user can audit every rewrite.
"""
from __future__ import annotations

import logging
import re
from typing import Iterable, Optional, Protocol, runtime_checkable

from .llm import LLMClient

log = logging.getLogger("speaker.cleanup")


@runtime_checkable
class TranscriptCleaner(Protocol):
    """Return a cleaned version of ``text``; equal to ``text`` if untouched."""

    def clean(self, text: str, recent: Iterable[str] = ()) -> str: ...


def rewrite_is_overreach(raw: str, cleaned: str, *, max_extra_words: int = 2) -> bool:
    """True iff the cleaner INVENTED content instead of cleaning it.

    A cleanup removes disfluencies and resolves self-corrections -- it can only
    SHRINK or lightly reshape the utterance. A rewrite that turns a tiny
    fragment into a much longer sentence is hallucination: live
    (run-20260610-132603) the fast-tier cleaner rewrote the noise fragment
    'Well' into the assistant's own prior sentence 'What would you like to know
    about your place?' (it sits in the cleaner's recent-context), manufacturing
    a phantom user turn the assistant then answered. Rule: a raw of N words may
    grow by at most ``max_extra_words`` -- 'Ario der' -> 'are you there' (2->3)
    survives, 'Well' -> a 9-word sentence does not. A same-length rewrite that
    quietly DROPS most of the raw's content and INVENTS new words ('TEND ME A
    LONG STORY ABOUT HER' -> 'A long story about the ceiling') slips past the
    length bound, so it is caught separately by ``drops_and_invents``. Pure +
    deterministic."""
    from always_on_agent.text import normalize_text

    from .asr_text import drops_and_invents

    raw_n = len(normalize_text(raw).split())
    cleaned_n = len(normalize_text(cleaned).split())
    if cleaned_n > raw_n + max_extra_words:
        return True
    if drops_and_invents(raw, cleaned):
        return True
    return False


_SYSTEM_PROMPT = """You are a transcript cleaner for a voice assistant. The
microphone produced a raw automatic speech recognition (ASR) final. Your
job is to return the user's INTENDED sentence -- disfluencies removed and
self-corrections resolved -- and NOTHING else.

Rules:
- Remove filler words when used as fillers (not when they carry meaning):
  um, uh, er, ah, like, you know, sort of, kind of, I mean (as a filler).
- If the user self-corrects with an editing term -- "I mean", "I meant",
  "no wait", "actually no", "scratch that" -- KEEP ONLY the corrected
  version, drop the original.
- WORD REPEAT AT THE END: if the user repeats the last word (e.g. "tell me
  about Paris Paris", "go to the store store"), treat the repeat as a
  correction marker and keep only one copy of the word. Same for the last
  two-word phrase ("the model the model" -> "the model").
- If a word is restated with a different one right after ("the algorithm,
  no, the model"), keep the second.
- Preserve meaning, punctuation, and casing. If nothing needs cleaning,
  return the input UNCHANGED.

Output ONLY the cleaned utterance. No quotes, no preamble, no explanation,
no trailing punctuation that the user didn't say."""


class LLMTranscriptCleaner:
    """Fast-tier LLM cleaner. Runs heuristic signal detection first
    (word-repeat-at-end, editing-term presence) and annotates the prompt
    so a small model has fewer ambiguous cases to resolve."""

    _EDITING_TERMS = (
        "i mean", "i meant", "no wait", "actually no",
        "scratch that", "let me restart", "let me rephrase",
    )

    def __init__(self, llm: LLMClient, *, max_context: int = 3) -> None:
        self._llm = llm
        self._max_context = max_context

    def clean(self, text: str, recent: Iterable[str] = ()) -> str:
        if not text.strip():
            return text
        signals = detect_signals(text)
        prompt = self._build_prompt(text, recent, signals)
        try:
            reply = self._llm.generate(prompt, system=_SYSTEM_PROMPT)
        except Exception:  # noqa: BLE001
            log.exception("cleanup LLM call failed; passing raw text through")
            return text
        return _sanitize_reply(reply, fallback=text)

    def _build_prompt(self, text: str, recent: Iterable[str], signals: list[str]) -> str:
        parts: list[str] = []
        context = [line for line in list(recent)[-self._max_context:] if line]
        if context:
            parts.append("Recent utterances (most recent last):")
            parts.extend(f"  - {line}" for line in context)
            parts.append("")
        if signals:
            parts.append("Detected signals in the raw text:")
            parts.extend(f"  - {s}" for s in signals)
            parts.append("")
        parts.append(f"Raw ASR text: {text}")
        parts.append("")
        parts.append("Cleaned:")
        return "\n".join(parts)


def detect_signals(text: str) -> list[str]:
    """Heuristic flags that get folded into the cleanup prompt. Surface-only:
    catches the obvious patterns so the LLM has a head start; the model
    still does the actual rewrite."""
    found: list[str] = []
    if has_trailing_repeat(text):
        found.append("word-repeat-at-end (last token or last bigram repeats; treat as correction marker)")
    lower = text.lower()
    for term in LLMTranscriptCleaner._EDITING_TERMS:
        if term in lower:
            found.append(f"editing term present: {term!r}")
            break
    return found


_WORD_RE = re.compile(r"\b[\w']+\b", flags=re.UNICODE)


def has_trailing_repeat(text: str) -> bool:
    """True if the utterance ends with a duplicated 1-, 2-, or 3-word phrase
    (case-insensitive). Examples: 'go go', 'the model the model',
    'we should go we should go'."""
    words = [w.lower() for w in _WORD_RE.findall(text)]
    for n in (1, 2, 3):
        if len(words) >= 2 * n and words[-n:] == words[-2 * n:-n]:
            return True
    return False


def _sanitize_reply(reply: str, *, fallback: str) -> str:
    """LLM cleanup sometimes wraps the answer in quotes or adds a leading
    'Cleaned:' label. Strip those; on a completely empty reply, fall back
    to the raw text so an LLM glitch never sends "" to the brain."""
    cleaned = (reply or "").strip()
    if not cleaned:
        return fallback
    for prefix in ("cleaned:", "output:", "result:"):
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix):].lstrip()
    # Strip a single matched wrapping quote pair (LLMs love to quote).
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in ('"', "'"):
        cleaned = cleaned[1:-1].strip()
    return cleaned or fallback


class ScriptedTranscriptCleaner:
    """Test fake: raw text -> cleaned text. Anything not in the map is
    returned unchanged (so cleaner is a no-op for non-scripted inputs)."""

    def __init__(self, mapping: Optional[dict[str, str]] = None) -> None:
        self._mapping = dict(mapping or {})
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def clean(self, text: str, recent: Iterable[str] = ()) -> str:
        self.calls.append((text, tuple(recent)))
        return self._mapping.get(text, text)
