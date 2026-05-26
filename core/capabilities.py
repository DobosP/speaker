from __future__ import annotations

from threading import Event
from typing import Iterator, Optional

from always_on_agent.capabilities import CapabilityRegistry, CapabilityResult

from .llm import LLMClient

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


def attach_llm_capabilities(
    registry: CapabilityRegistry,
    llm: LLMClient,
    *,
    fast_llm: Optional[LLMClient] = None,
    system: str = DEFAULT_SYSTEM,
) -> CapabilityRegistry:
    """Replace the brain's stub providers with real LLM-backed ones.

    ``create_default_capabilities`` registers offline stubs (a tiny hardcoded
    corpus). Here we override the two that should reason with the model:
    ``assistant.answer`` (direct replies) and ``research.local`` (synthesis of
    the gathered search/scope steps). Local-only providers such as
    ``search.local`` and ``meeting.note`` are left untouched.

    Two-model split: ``assistant.answer`` runs on ``fast_llm`` (a small, snappy
    model for short spoken replies) while ``research.local`` runs on ``llm`` (the
    larger, multimodal model). When ``fast_llm`` is omitted both use ``llm``.
    """

    quick = fast_llm or llm

    def assistant(query: str, context: dict[str, object]) -> CapabilityResult:
        cancel = context.get("cancel_event")
        text, cancelled = _collect(quick.stream(query, system=system), cancel)  # type: ignore[arg-type]
        if cancelled:
            return CapabilityResult(True, text, data={"cancelled": True})
        return CapabilityResult(True, text or "Sorry, I don't have an answer for that.")

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
        text, cancelled = _collect(llm.stream(prompt, system=system), cancel)  # type: ignore[arg-type]
        return CapabilityResult(True, text, data={"cancelled": cancelled})

    registry.register("assistant.answer", assistant)
    registry.register("research.local", research_synth)
    return registry
