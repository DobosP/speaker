"""Autonomous memory tier: does the assistant store a fact and *use* it later?

Drives the real ``assistant.answer`` capability (the same provider the runtime
uses) over an in-RAM :class:`SessionMemory` with recall enabled -- no Postgres,
no audio. Two levels of proof:

* **plumbing** -- the recall block injected into the model's ``system`` on the
  recalling turn contains the stored fact (keyword-overlap recall found it);
* **semantic** -- the model's actual answer contains the fact (it used the
  recalled context). Needs a real small LLM; ``--llm echo`` skips this level.

A fact is stated in turn 1, buried under unrelated distractor turns, then asked
for again -- the canonical "remembered across turns" check.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator, Optional, Sequence

from always_on_agent.capabilities import CapabilityRegistry
from always_on_agent.memory import SessionMemory
from core.capabilities import RecallConfig, attach_llm_capabilities
from core.conversation import RecentContextConfig
from core.llm import EchoLLM, OllamaLLM


class _CapturingLLM:
    """Wraps a real LLM, recording the ``system`` prompt + output of each call
    so the probe can assert on both the injected recall block and the answer."""

    def __init__(self, inner):
        self.inner = inner
        self.systems: list[str] = []
        self.outputs: list[str] = []

    def generate(self, prompt: str, *, system: Optional[str] = None, images=None) -> str:
        self.systems.append(system or "")
        out = self.inner.generate(prompt, system=system, images=images)
        self.outputs.append(out)
        return out

    def stream(self, prompt: str, *, system: Optional[str] = None, images=None) -> Iterator[str]:
        self.systems.append(system or "")
        chunks: list[str] = []
        for tok in self.inner.stream(prompt, system=system, images=images):
            chunks.append(tok)
            yield tok
        self.outputs.append("".join(chunks))


@dataclass
class MemoryResult:
    ok: bool
    fact: str
    keyword: str
    recall_injected: bool          # plumbing: fact appeared in the recall block
    answer_uses_fact: Optional[bool]  # semantic: answer contains the fact (None for echo)
    answer: str
    recall_block_preview: str
    llm_label: str
    turns: int
    duration_sec: float
    detail: list[str] = field(default_factory=list)


def run_memory_probe(
    *,
    llm_kind: str = "ollama",
    model: str = "gemma3:4b",
    fact: str = "my favorite color is teal",
    keyword: str = "teal",
    question: str = "what is my favorite color?",
    distractors: Sequence[str] = (
        "what time is it",
        "tell me a fun fact about the ocean",
        "how many days are in a week",
    ),
) -> MemoryResult:
    """Run the fact->distractors->recall flow and return a verdict."""
    t0 = time.monotonic()
    detail: list[str] = []

    if llm_kind == "echo":
        base = EchoLLM()
        label = "echo"
    else:
        base = OllamaLLM(model=model, keep_alive="5m", timeout=120.0, think=False)
        label = f"ollama/{model}"
    llm = _CapturingLLM(base)

    memory = SessionMemory()
    registry = CapabilityRegistry()
    attach_llm_capabilities(
        registry,
        llm,
        memory=memory,
        recall=RecallConfig(enabled=True, max_chars=600),
        recent_context=RecentContextConfig(enabled=False),
    )

    # Turn 1: establish the fact.
    registry.invoke("assistant.answer", fact, {})
    detail.append(f"turn 1 (state fact): {fact!r}")

    # Turns 2..n: unrelated distractors so recall must actually search, not echo
    # the immediately-preceding turn.
    for i, d in enumerate(distractors, start=2):
        registry.invoke("assistant.answer", d, {})
        detail.append(f"turn {i} (distractor): {d!r}")

    # Final turn: ask for the fact back. Clear captured systems so we inspect
    # only the recalling turn.
    llm.systems.clear()
    llm.outputs.clear()
    result = registry.invoke("assistant.answer", question, {})
    n_turns = 1 + len(distractors) + 1
    detail.append(f"turn {n_turns} (recall): {question!r}")

    system_used = llm.systems[-1] if llm.systems else ""
    answer = result.text or (llm.outputs[-1] if llm.outputs else "")
    recall_injected = keyword.lower() in system_used.lower()

    if llm_kind == "echo":
        answer_uses_fact: Optional[bool] = None  # echo can't reason; plumbing only
        ok = recall_injected
    else:
        answer_uses_fact = keyword.lower() in (answer or "").lower()
        ok = recall_injected and bool(answer_uses_fact)

    # a compact preview of where the recall block sits in the system prompt
    lo = system_used.lower()
    pos = lo.find(keyword.lower())
    preview = (system_used[max(0, pos - 40): pos + 60] if pos >= 0 else system_used[:120]).strip()

    return MemoryResult(
        ok=ok,
        fact=fact,
        keyword=keyword,
        recall_injected=recall_injected,
        answer_uses_fact=answer_uses_fact,
        answer=answer.strip(),
        recall_block_preview=preview,
        llm_label=label,
        turns=n_turns,
        duration_sec=round(time.monotonic() - t0, 2),
        detail=detail,
    )
