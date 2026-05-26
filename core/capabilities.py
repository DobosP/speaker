from __future__ import annotations

from always_on_agent.capabilities import CapabilityRegistry, CapabilityResult

from .llm import LLMClient

DEFAULT_SYSTEM = (
    "You are a local, on-device voice assistant. Reply in one or two short, "
    "natural spoken sentences. Do not use markdown, lists, headings, or "
    "preambles like 'Sure'. If you don't know, say so briefly."
)


def attach_llm_capabilities(
    registry: CapabilityRegistry,
    llm: LLMClient,
    *,
    system: str = DEFAULT_SYSTEM,
) -> CapabilityRegistry:
    """Replace the brain's stub providers with real LLM-backed ones.

    ``create_default_capabilities`` registers offline stubs (a tiny hardcoded
    corpus). Here we override the two that should reason with the model:
    ``assistant.answer`` (direct replies) and ``research.local`` (synthesis of
    the gathered search/scope steps). Local-only providers such as
    ``search.local`` and ``meeting.note`` are left untouched.
    """

    def assistant(query: str, context: dict[str, object]) -> CapabilityResult:
        text = llm.generate(query, system=system).strip()
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
        return CapabilityResult(True, llm.generate(prompt, system=system).strip())

    registry.register("assistant.answer", assistant)
    registry.register("research.local", research_synth)
    return registry
