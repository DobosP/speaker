from __future__ import annotations

import os
import re
from threading import Event
from typing import Callable, Iterator, Mapping, Optional

from always_on_agent.capabilities import CapabilityRegistry, CapabilityResult

from .llm import LLMClient
from .metrics import MetricsRecorder, mark_first_token
from .routing import HeuristicRouter, Router

# Predicate deciding whether an ASSISTANT-mode query should escalate to the
# ReAct planner instead of a one-shot reply.
EscalatePredicate = Callable[[str, Mapping[str, object]], bool]

DEFAULT_SYSTEM = (
    "You are a local, on-device voice assistant. Reply in one or two short, "
    "natural spoken sentences. Do not use markdown, lists, headings, or "
    "preambles like 'Sure'. If you don't know, say so briefly."
)


def _collect(tokens: Iterator[str], cancel: Optional[Event]) -> tuple[str, bool]:
    """Drain a token stream, stopping early if ``cancel`` fires.

    Returns ``(text, cancelled)``. Streaming (rather than a blocking
    ``generate``) is what makes a slow local model interruptible: barge-in cuts
    generation off mid-stream instead of waiting for the whole answer."""
    parts: list[str] = []
    cancelled = False
    for token in tokens:
        if cancel is not None and cancel.is_set():
            cancelled = True
            break
        parts.append(token)
    return "".join(parts).strip(), cancelled


# A sentence terminator followed by whitespace marks a chunk safe to speak
# while the rest of the answer is still being generated.
_SENTENCE_END = re.compile(r"[.!?]\s")


def _stream_and_speak(
    tokens: Iterator[str], cancel: Optional[Event], emit: Callable[[str], None]
) -> tuple[str, bool]:
    """Drain a token stream, speaking each complete sentence as it lands.

    This is the latency win: playback of sentence one starts while the model is
    still generating sentence two. Returns the full ``(text, cancelled)`` so the
    caller can still log/remember the whole answer."""
    parts: list[str] = []
    buffer = ""
    cancelled = False
    for token in tokens:
        if cancel is not None and cancel.is_set():
            cancelled = True
            break
        parts.append(token)
        buffer += token
        while True:
            match = _SENTENCE_END.search(buffer)
            if not match:
                break
            sentence = buffer[: match.start() + 1].strip()
            buffer = buffer[match.end():]
            if sentence:
                emit(sentence)
    tail = buffer.strip()
    if tail and not cancelled:
        emit(tail)
    return "".join(parts).strip(), cancelled


def attach_llm_capabilities(
    registry: CapabilityRegistry,
    llm: LLMClient,
    *,
    fast_llm: Optional[LLMClient] = None,
    system: str = DEFAULT_SYSTEM,
    router: Optional[Router] = None,
    escalate: Optional[EscalatePredicate] = None,
    agent_capability: str = "agent.react",
    recorder: Optional[MetricsRecorder] = None,
) -> CapabilityRegistry:
    """Replace the brain's stub providers with real LLM-backed ones.

    ``create_default_capabilities`` registers offline stubs (a tiny hardcoded
    corpus). Here we override the two that should reason with the model:
    ``assistant.answer`` (direct replies) and ``research.local`` (synthesis of
    the gathered search/scope steps). Local-only providers such as
    ``search.local`` and ``meeting.note`` are left untouched.

    Two-model split: ``assistant.answer`` picks its model per query via
    ``router`` (a small ``fast_llm`` for snappy replies, the larger ``llm`` for
    reasoning-heavy asks) while ``research.local`` always runs on ``llm``. When
    ``fast_llm`` is omitted both tiers collapse to ``llm`` (routing is a no-op).
    """

    fast = fast_llm or llm
    router = router or HeuristicRouter()
    debug_routing = bool(os.environ.get("SPEAKER_DEBUG_ROUTING"))

    def assistant(query: str, context: dict[str, object]) -> CapabilityResult:
        cancel = context.get("cancel_event")
        # Smart-mode escalation: hand reasoning/gathering queries to the ReAct
        # planner (when registered) instead of a one-shot reply.
        if (
            escalate is not None
            and agent_capability in registry.names()
            and escalate(query, context)
        ):
            return registry.invoke(agent_capability, query, context)
        tier = router.choose(query, context)
        model = llm if tier == "main" else fast
        if debug_routing:
            print(f"[route] {tier} <- {query!r}", flush=True)
        emit = context.get("emit_speech")
        tokens = mark_first_token(model.stream(query, system=system), recorder)
        if callable(emit):
            text, cancelled = _stream_and_speak(tokens, cancel, emit)  # type: ignore[arg-type]
            return CapabilityResult(
                True,
                text or "Sorry, I don't have an answer for that.",
                data={"route": tier, "streamed": True, "cancelled": cancelled},
            )
        text, cancelled = _collect(tokens, cancel)  # type: ignore[arg-type]
        if cancelled:
            return CapabilityResult(True, text, data={"cancelled": True, "route": tier})
        return CapabilityResult(
            True,
            text or "Sorry, I don't have an answer for that.",
            data={"route": tier},
        )

    def research_synth(query: str, context: dict[str, object]) -> CapabilityResult:
        previous = context.get("previous_steps", [])
        gathered = " ".join(
            str(step.get("text", ""))
            for step in previous
            if isinstance(step, dict) and step.get("text")
        )
        prompt = (
            f"Question: {query}\n"
            f"Local findings: {gathered or '(none)'}\n"
            "Give a brief spoken-style summary and one concrete recommendation."
        )
        cancel = context.get("cancel_event")
        emit = context.get("emit_speech")
        tokens = mark_first_token(llm.stream(prompt, system=system), recorder)
        if callable(emit):
            text, cancelled = _stream_and_speak(tokens, cancel, emit)  # type: ignore[arg-type]
            return CapabilityResult(True, text, data={"cancelled": cancelled, "streamed": True})
        text, cancelled = _collect(tokens, cancel)  # type: ignore[arg-type]
        return CapabilityResult(True, text, data={"cancelled": cancelled})

    registry.register("assistant.answer", assistant)
    registry.register("research.local", research_synth)
    return registry
