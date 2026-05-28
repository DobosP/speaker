"""Addressing classifier: decides whether an ASR final is addressed to the
assistant or just ambient speech that should be remembered silently.

The pipeline used to send every clean ASR final straight to the LLM, which
meant the assistant answered everything it heard -- see
``logs/runs/run-20260528-004726.summary.json`` for the symptom (four
nonsense transcripts answered as polite "I don't know that name" replies).
This module is the gate: a thin pre-LLM classification step that returns
``ACT`` for utterances genuinely directed at the assistant and ``INGEST``
for ambient speech (background noise, the user thinking out loud, reading
aloud, another person in the room). ``UNSURE`` is the safe fallback when
the classifier can't decide; the caller picks a policy.

See ``docs/target_architecture.md`` §9.8 for the design decision and
``PROJECT_KICKOFF.md`` §2 for the product intent. Speaker-ID gating (so
*only the enrolled user's voice* is even considered) is a separate layer
landing in a follow-up PR.
"""
from __future__ import annotations

import logging
from typing import Iterable, Optional, Protocol, runtime_checkable

from .llm import LLMClient

log = logging.getLogger("speaker.addressing")

ACT = "ACT"
INGEST = "INGEST"
UNSURE = "UNSURE"

_VALID = (ACT, INGEST, UNSURE)


@runtime_checkable
class AddressingClassifier(Protocol):
    """Returns one of :data:`ACT` / :data:`INGEST` / :data:`UNSURE`."""

    def classify(self, text: str, recent: Iterable[str] = ()) -> str: ...


_SYSTEM_PROMPT = """You are an addressing classifier for a voice assistant.
The microphone is always on. Audio from the room reaches you the same way
whether the user is speaking TO the assistant, thinking out loud, reading
something aloud, or another person is talking nearby.

Given the latest utterance plus a few recent ones for context, decide if
THE LATEST utterance is addressed to the assistant (the user expects a
response).

Reply with exactly one word:
- ACT     -- clearly addressed to the assistant; it should respond.
- INGEST  -- not addressed; remember silently, do not respond.
- UNSURE  -- truly ambiguous.

Do not explain. Do not add punctuation. One word."""


class LLMAddressingClassifier:
    """LLM-backed addressing classifier.

    Calls the fast-tier LLM with a fixed system prompt and parses the first
    word of the reply. Anything that doesn't parse to ``ACT``/``INGEST``/
    ``UNSURE`` becomes ``UNSURE`` -- a hiccup in the LLM never silently
    flips behavior; the caller's policy decides what UNSURE means.
    """

    def __init__(self, llm: LLMClient, *, max_context: int = 4) -> None:
        self._llm = llm
        self._max_context = max_context

    def classify(self, text: str, recent: Iterable[str] = ()) -> str:
        prompt = self._build_prompt(text, recent)
        try:
            reply = self._llm.generate(prompt, system=_SYSTEM_PROMPT)
        except Exception:  # noqa: BLE001
            log.exception("addressing classifier LLM call failed; defaulting to UNSURE")
            return UNSURE
        return _parse_decision(reply)

    def _build_prompt(self, text: str, recent: Iterable[str]) -> str:
        context = [line for line in list(recent)[-self._max_context:] if line]
        parts: list[str] = []
        if context:
            parts.append("Recent utterances (most recent last):")
            parts.extend(f"  - {line}" for line in context)
            parts.append("")
        parts.append(f'Latest utterance: "{text}"')
        return "\n".join(parts)


def _parse_decision(reply: str) -> str:
    """Pluck the first decision word out of an LLM reply. Anything malformed
    is :data:`UNSURE` so an LLM hiccup never causes a spurious ACT."""
    stripped = (reply or "").strip()
    if not stripped:
        return UNSURE
    first = stripped.split()[0].strip(".,!?:;\"'").upper()
    if first in _VALID:
        return first
    log.warning("addressing classifier returned %r; defaulting to UNSURE", reply)
    return UNSURE


class ScriptedAddressingClassifier:
    """Test fake: maps utterance text -> decision. Anything not in the map
    is ``default`` (defaults to :data:`UNSURE`)."""

    def __init__(self, mapping: Optional[dict[str, str]] = None, *, default: str = UNSURE) -> None:
        self._mapping = dict(mapping or {})
        self._default = default
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def classify(self, text: str, recent: Iterable[str] = ()) -> str:
        self.calls.append((text, tuple(recent)))
        return self._mapping.get(text, self._default)
