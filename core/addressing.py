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

from .llm import LLMCallCancelled, LLMClient, collect_llm_text

log = logging.getLogger("speaker.addressing")

ACT = "ACT"
INGEST = "INGEST"
UNSURE = "UNSURE"

_VALID = (ACT, INGEST, UNSURE)


@runtime_checkable
class AddressingClassifier(Protocol):
    """Returns one of :data:`ACT` / :data:`INGEST` / :data:`UNSURE`."""

    def classify(self, text: str, recent: Iterable[str] = ()) -> str: ...


# NOTE: the framing is "is this a QUESTION/REQUEST/COMMAND for the assistant?",
# NOT "is this addressed to the assistant?". The latter (the original prompt) made
# the small fast model drop clear questions as ambient -- live, gemma3:4b INGESTed
# "how do I make pasta", "why is the sky blue", "what about Italy" (missed-question
# bug). Reframing around question/request/command + explicit examples fixed the
# questions (10/10 novel) while still INGESTing plain statements / talk-to-another /
# reading-aloud. With unsure_acts=True the borderline cases lean to ACT, which is
# the right trade (a missed question is worse than answering an occasional aside).
_SYSTEM_PROMPT = """You are the addressing gate for an always-on voice assistant.
The microphone hears the whole room. Decide whether THE LATEST utterance is the
user asking or telling the ASSISTANT to do something.

ACT if it is a QUESTION, a REQUEST, or a COMMAND the user wants the assistant to
answer or do -- even short, casual, or with no wake word ("what time is it",
"how do I make pasta", "play some music", "what about Italy").

INGEST if it is NOT a question/request/command aimed at the assistant:
- a plain statement or comment with no request ("I think I left the stove on")
- talking to another person ("no, I told you yesterday")
- reading aloud or quoting
- a mid-thought fragment or muttering ("um, so anyway, where was I")
- RECOGNIZER NOISE: garbled, nonsensical, or disjointed text that does not
  read as a coherent question, request, or command. The microphone hears
  household noise and half-words; these come out as word salad. When the
  utterance does not parse as something a person would deliberately say to
  an assistant, it is noise -> INGEST, never ACT.

If genuinely torn between ACT and INGEST, answer UNSURE.
Reply with exactly one word: ACT, INGEST, or UNSURE. No punctuation, no explanation.

"why is the sky blue" -> ACT
"set a timer for five minutes" -> ACT
"what about Italy" -> ACT
"can you help me" -> ACT
"I think I left the stove on" -> INGEST
"no I already told you that yesterday" -> INGEST
"um so anyway where was I" -> INGEST
"N Sanos you know" -> INGEST
"I just kind cast brand" -> INGEST
"ca chap for to" -> INGEST"""


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
            reply = collect_llm_text(
                self._llm,
                prompt,
                system=_SYSTEM_PROMPT,
            )
        except LLMCallCancelled:
            raise
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
