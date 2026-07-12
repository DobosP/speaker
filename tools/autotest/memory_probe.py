"""Autonomous memory tier: does the assistant store a fact and *use* it later?

Drives the real ``assistant.answer`` capability (the same provider the runtime
uses) across a temporary SQLite close/reopen with recall enabled -- no Postgres,
no audio. Two levels of proof:

* **plumbing** -- the recall block injected into the model's ``system`` on the
  recalling turn contains the stored fact (keyword-overlap recall found it);
* **semantic** -- the model's actual answer contains the fact (it used the
  recalled context). Needs a real small LLM; ``--llm echo`` skips this level.

A fact is stated in turn 1, buried under unrelated distractor turns, then asked
for again -- the canonical "remembered across turns" check.
"""
from __future__ import annotations

import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Sequence

from always_on_agent.capabilities import CapabilityRegistry
from always_on_agent.sqlite_memory import SqliteVecMemory
from always_on_agent.untrusted import fenced_data_contains
from core.capabilities import RecallConfig, attach_llm_capabilities
from core.conversation import RecentContextConfig
from core.llm import EchoLLM, OllamaLLM
from .verdicts import DIAGNOSTIC_PASS, FAIL, PASS


def _uses_recalled_fact(answer: str, keyword: str) -> bool:
    """Return whether *answer* confidently attributes *keyword* to the user.

    A bare keyword check can turn a denial (``"I don't know; maybe teal"``) or
    an assistant-persona claim (``"My favorite color is teal"``) into a false
    green.  This deliberately small semantic grader accepts concise scalar and
    user-attributed answers, while failing closed on common denial, hedging,
    negation, and first-person self-attribution forms.
    """
    text = re.sub(r"\s+", " ", (answer or "").replace("’", "'")).strip()
    wanted = re.sub(r"\s+", " ", (keyword or "")).strip()
    if not text or not wanted:
        return False

    term = r"\s+".join(re.escape(part) for part in wanted.split())
    bounded_term = rf"(?<!\w){term}(?!\w)"
    if re.search(bounded_term, text, flags=re.IGNORECASE) is None:
        return False

    # Knowledge denials and hedges are not evidence of successful recall even
    # when they repeat the expected value as a possibility.
    uncertainty_patterns = (
        r"\b(?:maybe|perhaps|possibly|probably|presumably)\b",
        r"\b(?:may|might|could)\s+(?:well\s+)?be\b",
        r"\b(?:i\s+)?(?:do(?:n't| not)|can(?:not|'t))\s+"
        r"(?:know|recall|remember|confirm|determine|say|tell|answer|verify)\b",
        r"\b(?:i\s+)?can(?:not|'t)\s+be\s+(?:sure|certain)\b",
        r"\b(?:i(?:'m| am)?\s+)?(?:not sure|unsure)\b",
        r"\b(?:i\s+have\s+)?no\s+(?:idea|information|memory|record|way\s+to\s+know)\b",
        r"\b(?:i|we)\s+(?:do(?:n't| not)|can(?:not|'t))\s+"
        r"(?:have|possess)\b[^.!?;\n]{0,80}\b(?:favorite|favourite|preference|"
        r"taste|opinion)s?\b",
        r"\bno\s+(?:personal\s+)?(?:favorite|favourite|preference|taste|opinion)s?\b",
        r"\b(?:as\s+an?\s+|i(?:'m| am)\s+an?\s+)(?:ai|assistant|language\s+model)\b",
        r"\b(?:unknown|unclear|uncertain)\b",
        r"\b(?:i\s+)?(?:think|guess|suspect|doubt)\b",
        r"\b(?:i\s+)?(?:would|have)\s+to\s+guess\b",
        r"\b(?:seems|appears)\s+(?:to\s+be\s+)?",
    )
    if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in uncertainty_patterns):
        return False

    # Reject direct negation of the expected value, without rejecting useful
    # contrasts such as "not blue -- teal".
    negated_term_patterns = (
        rf"\b(?:isn't|isnt|wasn't|wasnt|aren't|arent|weren't|werent|not|never)\s+"
        rf"(?:actually\s+)?{bounded_term}",
        rf"{bounded_term}\s+(?:(?:is|was|are|were)\s+not|isn't|isnt|wasn't|wasnt|"
        rf"aren't|arent|weren't|werent)\b",
        rf"{bounded_term}\s*\?\s*no\b",
    )
    if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in negated_term_patterns):
        return False

    # Explicit second-person attribution wins over incidental first-person
    # framing (for example, "I remember that your favorite color is teal").
    user_attribution_patterns = (
        rf"\b(?:your|yours)\b[^.!?;\n]{{0,100}}{bounded_term}",
        rf"\byou\s+(?:said|mentioned|stated|told\s+me|prefer|preferred|like|liked|"
        rf"love|loved|chose|picked|selected)\b[^.!?;\n]{{0,100}}{bounded_term}",
        rf"{bounded_term}[^.!?;\n]{{0,80}}\b(?:is\s+yours|was\s+yours|as\s+you\s+"
        rf"(?:said|mentioned|told\s+me))\b",
    )
    if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in user_attribution_patterns):
        return True

    # Do not credit the model for assigning the remembered fact to itself.
    assistant_attribution_patterns = (
        rf"\bmy\s+[^.!?;\n]{{0,80}}(?:\b(?:is|was|are|were)\b|'s|:)\s+"
        rf"(?:definitely\s+|certainly\s+)?{bounded_term}",
        rf"\bmy\s+[^.!?;\n]{{0,80}}\?\s*{bounded_term}",
        rf"{bounded_term}[^.!?;\n]{{0,60}}\b(?:is|was)\s+(?:one\s+of\s+)?my\b",
        rf"\bmine\b[^.!?;\n]{{0,60}}{bounded_term}",
        rf"{bounded_term}[^.!?;\n]{{0,60}}\b(?:is|was)\s+mine\b",
        rf"\bi\s+(?:personally\s+)?(?:(?:would\s+)?(?:prefer|like|love|choose)|"
        rf"chose|picked|selected|"
        rf"said|mentioned|stated)\b[^.!?;\n]{{0,80}}{bounded_term}",
        rf"{bounded_term}[^.!?;\n]{{0,60}}\bi\s+(?:prefer|like|love|chose|choose|"
        rf"picked|selected)\b",
        rf"\b(?:for\s+me|personally)\b[^.!?;\n]{{0,60}}{bounded_term}",
    )
    if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in assistant_attribution_patterns):
        return False

    # Scalar ("Teal!"), copular ("It's teal"), and other neutral factual
    # phrasings are valid once the false-positive forms above are excluded.
    return True


def _uses_recalled_canary(
    answer: str,
    keyword: str,
    subject_terms: Sequence[str],
) -> bool:
    """Fail-closed grounding for the probe's synthetic non-personal fact."""
    if not _uses_recalled_fact(answer, keyword):
        return False
    normalized = re.sub(r"[^a-z0-9\s]", " ", answer.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    wanted = re.sub(r"[^a-z0-9\s]", " ", keyword.lower())
    wanted = re.sub(r"\s+", " ", wanted).strip()
    # A concise scalar is unambiguous because the prompt asks for exactly one
    # synthetic value.
    if normalized in {wanted, f"it is {wanted}", f"it was {wanted}"}:
        return True
    terms = [term.lower().strip() for term in subject_terms if term.strip()]
    if not terms or any(re.search(rf"\b{re.escape(term)}\b", normalized) is None for term in terms):
        return False
    # Reject assigning the canary to the assistant or an explicitly different
    # subject.  Neutral/the-project and second-person attribution remain valid.
    if re.search(
        r"\b(?:assistant(?:s)?|another|other|their|his|her)\b"
        r"[^.!?;\n]{0,80}\bproject\b",
        answer.replace("’", "'").lower(),
    ):
        return False
    return True


class _CapturingLLM:
    """Wraps a real LLM, recording the ``system`` prompt + output of each call
    so the probe can assert on both the injected recall block and the answer."""

    def __init__(self, inner):
        self.inner = inner
        self.systems: list[str] = []
        self.outputs: list[str] = []
        self.histories: list[list[dict]] = []

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        images=None,
        history=None,
    ) -> str:
        self.systems.append(system or "")
        self.histories.append(list(history or []))
        out = self.inner.generate(
            prompt, system=system, images=images, history=history
        )
        self.outputs.append(out)
        return out

    def stream(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        images=None,
        history=None,
    ) -> Iterator[str]:
        self.systems.append(system or "")
        self.histories.append(list(history or []))
        chunks: list[str] = []
        for tok in self.inner.stream(
            prompt, system=system, images=images, history=history
        ):
            chunks.append(tok)
            yield tok
        self.outputs.append("".join(chunks))


@dataclass
class MemoryResult:
    ok: bool
    plumbing_ok: bool
    complete: bool
    outcome: str
    fact: str
    keyword: str
    recall_available: bool         # plumbing: selector returned the fact
    recall_injected: bool          # model path only: fact appeared in its prompt
    recall_fenced: bool            # persisted fact stayed inside untrusted-data boundary
    answer_uses_fact: Optional[bool]  # semantic: answer grounds the fact (None for echo)
    answer: str
    answer_model: str
    answer_route: str
    topology_valid: bool
    recent_history_clean: bool
    controller_answer: bool
    cross_session: bool
    recall_block_preview: str
    llm_label: str
    turns: int
    duration_sec: float
    detail: list[str] = field(default_factory=list)


def run_memory_probe(
    *,
    llm_kind: str = "ollama",
    model: str = "minicpm5-1b:q8",
    main_model: Optional[str] = "gemma3:12b",
    fact: str = "my lighthouse project codename was Amber Finch",
    keyword: str = "Amber Finch",
    question: str = "what was my lighthouse project codename?",
    answer_subject_terms: Sequence[str] = ("project", "codename"),
    distractors: Sequence[str] = (
        "what time is it",
        "tell me a fun fact about the ocean",
        "how many days are in a week",
    ),
) -> MemoryResult:
    """Run a fact -> restart -> fenced strong-recall flow and return a verdict."""
    t0 = time.monotonic()
    detail: list[str] = []

    if llm_kind == "echo":
        base = EchoLLM()
        label = "echo"
        fast_llm = main_llm = _CapturingLLM(base)
        topology_valid = True  # deterministic plumbing mode has no model roles
    else:
        fast_llm = _CapturingLLM(
            OllamaLLM(model=model, keep_alive="5m", timeout=120.0, think=False)
        )
        resolved_main = main_model or model
        main_llm = (
            fast_llm
            if resolved_main == model
            else _CapturingLLM(
                OllamaLLM(
                    model=resolved_main,
                    keep_alive="5m",
                    timeout=120.0,
                    think=False,
                )
            )
        )
        label = f"ollama/fast={model},main={resolved_main}"
        topology_valid = main_llm is not fast_llm

    with tempfile.TemporaryDirectory(prefix="speaker-memory-probe-") as tmp:
        path = str(Path(tmp) / "memory.db")
        memory = SqliteVecMemory(path)
        registry = CapabilityRegistry()
        attach_llm_capabilities(
            registry,
            main_llm,
            fast_llm=fast_llm,
            memory=memory,
            recall=RecallConfig(enabled=True, max_chars=600),
            recent_context=RecentContextConfig(enabled=False),
        )

        # Process/session 1: establish and bury the fact through the real
        # assistant capability, then close the backend to erase every controller
        # cache and trusted recent-history ring.
        try:
            registry.invoke("assistant.answer", fact, {})
            detail.append(f"turn 1 (state fact): {fact!r}")
            for i, d in enumerate(distractors, start=2):
                registry.invoke("assistant.answer", d, {})
                detail.append(f"turn {i} (distractor): {d!r}")
        finally:
            memory.close()

        # Process/session 2: a fresh backend + registry may see old rows only
        # through persistent recall, never native recent history or controller
        # state.  Inspect only this final model call.
        for captured in (fast_llm, main_llm):
            captured.systems.clear()
            captured.outputs.clear()
            captured.histories.clear()
        memory = SqliteVecMemory(path)
        registry = CapabilityRegistry()
        attach_llm_capabilities(
            registry,
            main_llm,
            fast_llm=fast_llm,
            memory=memory,
            recall=RecallConfig(enabled=True, max_chars=600),
            recent_context=RecentContextConfig(enabled=True, as_messages=True),
        )
        try:
            reopened_recent_empty = memory.all() == []
            retrieval_block = memory.context_for_llm(question)
            result = registry.invoke("assistant.answer", question, {})
        finally:
            memory.close()
    n_turns = 1 + len(distractors) + 1
    detail.append(f"turn {n_turns} (recall): {question!r}")

    controller_answer = bool(result.data.get("recalled_self_fact"))
    answered_by = main_llm if main_llm.outputs else fast_llm
    system_used = "" if controller_answer else (
        answered_by.systems[-1] if answered_by.systems else ""
    )
    answer = result.text or (answered_by.outputs[-1] if answered_by.outputs else "")
    answer_route = str(result.data.get("route", ""))
    final_history = answered_by.histories[-1] if answered_by.histories else []
    if controller_answer:
        answer_model = "control"
    elif llm_kind == "echo":
        answer_model = "echo"
    else:
        answer_model = (main_model or model) if answered_by is main_llm else model
    recall_available = keyword.lower() in retrieval_block.lower()
    recall_injected = keyword.lower() in system_used.lower()
    recall_fenced = recall_injected and fenced_data_contains(system_used, keyword)
    recent_history_clean = reopened_recent_empty and not final_history
    cross_session = recall_available and recent_history_clean

    if controller_answer:
        answer_uses_fact = _uses_recalled_fact(answer or "", keyword)
    elif llm_kind == "echo":
        answer_uses_fact: Optional[bool] = None  # echo can't reason; plumbing only
    else:
        answer_uses_fact = _uses_recalled_canary(
            answer or "", keyword, answer_subject_terms
        )
    plumbing_ok = (
        recall_available
        and recall_injected
        and recall_fenced
        and answer_route == "main"
        and not controller_answer
        and recent_history_clean
        and cross_session
    )
    complete = llm_kind != "echo"
    if not plumbing_ok or (complete and not topology_valid):
        outcome = FAIL
    elif not complete:
        outcome = DIAGNOSTIC_PASS
    else:
        outcome = PASS if bool(answer_uses_fact) else FAIL
    ok = outcome == PASS

    # a compact preview of where the recall block sits in the system prompt
    preview_source = system_used or retrieval_block
    lo = preview_source.lower()
    pos = lo.find(keyword.lower())
    preview = (
        preview_source[max(0, pos - 40): pos + 60]
        if pos >= 0 else preview_source[:120]
    ).strip()

    return MemoryResult(
        ok=ok,
        plumbing_ok=plumbing_ok,
        complete=complete,
        outcome=outcome,
        fact=fact,
        keyword=keyword,
        recall_available=recall_available,
        recall_injected=recall_injected,
        recall_fenced=recall_fenced,
        answer_uses_fact=answer_uses_fact,
        answer=answer.strip(),
        answer_model=answer_model,
        answer_route=answer_route,
        topology_valid=topology_valid,
        recent_history_clean=recent_history_clean,
        controller_answer=controller_answer,
        cross_session=cross_session,
        recall_block_preview=preview,
        llm_label=label,
        turns=n_turns,
        duration_sec=round(time.monotonic() - t0, 2),
        detail=detail,
    )
