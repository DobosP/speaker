"""Continuity-completion slice -- capability-level injection behavior (no DB).

Covers two of the three slice items at the prompt-assembly seam
(``core.capabilities.attach_llm_capabilities``), using a recording LLM + an
in-RAM :class:`SessionMemory` subclass so no PostgreSQL is needed:

* **lm-2 Wire 3** -- the one-shot "Last session" recap is injected on the FIRST
  one-shot answer turn only (a process-start latch), ahead of recall, and floats
  the turn's sensitivity over its (private) content.
* **Recall-B** -- durable PROFILE facts inject even when ``recall_enabled`` is
  OFF (decoupled), without double-injecting when recall is ON, and float
  sensitivity too.

The Postgres-tier producers (snapshot, ``get_profile_context``, assistant-final
persistence + recall) live in ``test_memory_lm5_pg.py`` (self-skips w/o psycopg).
"""
from __future__ import annotations

from always_on_agent.capabilities import CapabilityRegistry, CapabilityResult
from always_on_agent.memory import SessionMemory

from core.capabilities import DEFAULT_SYSTEM, RecallConfig, attach_llm_capabilities
from core.conversation import RecentContextConfig
from core.sensitivity import PRIVATE, classify_sensitivity

_NO_RECENT = RecentContextConfig(enabled=False)


class _RecordingLLM:
    def __init__(self) -> None:
        self.systems: list = []

    def generate(self, prompt, *, system=None, images=None):
        self.systems.append(system)
        return "ok"

    def stream(self, prompt, *, system=None, images=None):
        self.systems.append(system)
        yield "ok"


def _assistant(llm, memory, **kw):
    reg = CapabilityRegistry()
    attach_llm_capabilities(reg, llm, memory=memory, recent_context=_NO_RECENT, **kw)
    return reg


# --- lm-2 Wire 3: one-shot "Last session" block -----------------------------


class _HeadMemory(SessionMemory):
    """In-RAM memory that also surfaces a prior-session head (lm-2 Wire 3)."""

    def __init__(self, head: str = "", **kw):
        super().__init__(**kw)
        self._head = head

    def last_session_summary(self) -> str:
        return self._head


def test_last_session_block_injected_once_then_latched():
    llm = _RecordingLLM()
    mem = _HeadMemory(head="We discussed your trip to Berlin and your dog Rex.")
    reg = _assistant(llm, mem, recall=RecallConfig(enabled=False))

    reg.invoke("assistant.answer", "what should I pack", {})
    reg.invoke("assistant.answer", "and the weather", {})

    sys1 = llm.systems[0]
    assert "=== Last Session ===" in sys1
    assert "Berlin" in sys1
    assert DEFAULT_SYSTEM in sys1  # the recap rides AHEAD of the stable system
    # One-shot: the second turn is latched -> no recap.
    assert "=== Last Session ===" not in llm.systems[1]


def test_last_session_block_absent_when_head_empty_is_byte_identical():
    llm = _RecordingLLM()
    mem = _HeadMemory(head="")  # continuity off / no prior summary
    reg = _assistant(llm, mem, recall=RecallConfig(enabled=False))
    reg.invoke("assistant.answer", "hello", {})
    assert llm.systems == [DEFAULT_SYSTEM]


def test_last_session_block_floats_sensitivity_to_private():
    # §9.7 (lm-3): a private fact in the recap floats a PUBLIC turn to PRIVATE.
    llm = _RecordingLLM()
    mem = _HeadMemory(head="The user's salary is $95,000 at Acme Corp.")
    reg = _assistant(llm, mem, recall=RecallConfig(enabled=False))
    ctx: dict = {"intent_kind": "assistant"}
    reg.invoke("assistant.answer", "what is the capital of france", ctx)
    assert ctx["sensitivity"] == PRIVATE


def test_last_session_block_compressed_to_budget():
    # A pathologically long head is bounded (whole-word) to the recall budget so
    # it can't blow up TTFT; some content still rides.
    long_head = " ".join(f"Point {i} about the trip." for i in range(200))
    llm = _RecordingLLM()
    mem = _HeadMemory(head=long_head)
    reg = _assistant(llm, mem, recall=RecallConfig(enabled=False, max_tokens=40))
    reg.invoke("assistant.answer", "tell me about the trip", {})
    sys1 = llm.systems[0]
    assert "=== Last Session ===" in sys1
    # bounded: the recap line is far shorter than the raw head.
    recap = sys1.split("=== Last Session ===", 1)[1]
    assert len(recap) < len(long_head)


# --- Recall-B: profile injection decoupled from recall_enabled --------------


class _ProfileMemory(SessionMemory):
    def __init__(self, block: str = "", **kw):
        super().__init__(**kw)
        self._block = block

    def profile_block(self) -> str:
        return self._block


def test_profile_block_injects_with_recall_off():
    llm = _RecordingLLM()
    mem = _ProfileMemory(block="=== User Profile ===\n- name: Alice\n- city: Berlin")
    reg = _assistant(llm, mem, recall=RecallConfig(enabled=False))
    reg.invoke("assistant.answer", "what's my name", {})
    sys = llm.systems[-1]
    assert "=== User Profile ===" in sys
    assert "Alice" in sys
    assert DEFAULT_SYSTEM in sys


def test_profile_block_empty_keeps_default_system():
    llm = _RecordingLLM()
    mem = _ProfileMemory(block="")  # profiles disabled / no facts
    reg = _assistant(llm, mem, recall=RecallConfig(enabled=False))
    reg.invoke("assistant.answer", "hi", {})
    assert llm.systems == [DEFAULT_SYSTEM]


def test_profile_block_not_called_when_recall_on():
    # Recall ON already includes the profile sub-pass inside context_for_llm;
    # profile_block() must NOT also run (no duplicate block, no double work).
    calls = {"profile": 0}

    class _Mem(SessionMemory):
        def context_for_llm(self, query):
            return "=== Past Conversations ===\nUser: hi"

        def profile_block(self):
            calls["profile"] += 1
            return "=== User Profile ===\n- name: Bob"

    llm = _RecordingLLM()
    reg = _assistant(llm, _Mem(), recall=RecallConfig(enabled=True))
    reg.invoke("assistant.answer", "q", {})
    assert calls["profile"] == 0
    assert "=== Past Conversations ===" in llm.systems[-1]


def test_profile_block_floats_sensitivity_to_private():
    llm = _RecordingLLM()
    mem = _ProfileMemory(block="=== User Profile ===\n- salary: $95,000 at Acme")
    reg = _assistant(llm, mem, recall=RecallConfig(enabled=False))
    ctx: dict = {"intent_kind": "assistant"}
    reg.invoke("assistant.answer", "what is the capital of france", ctx)
    assert ctx["sensitivity"] == PRIVATE


def test_recall_off_and_no_profile_is_byte_identical():
    # The decoupling must not change the default: recall off + no profile tier
    # (plain SessionMemory) -> exactly the base system prompt.
    llm = _RecordingLLM()
    mem = SessionMemory()
    mem.add("my favorite color is teal", tags=("user",))
    reg = _assistant(llm, mem, recall=RecallConfig(enabled=False))
    reg.invoke("assistant.answer", "what is my favorite color", {})
    assert llm.systems == [DEFAULT_SYSTEM]


# --- §9.7 combined-block sensitivity float (review-driven coverage) ----------


def test_recall_on_floats_over_private_last_session_head():
    # recall ON + a BENIGN recall block, but a PRIVATE last-session head must
    # still float the turn to PRIVATE -- the head, not just the recall block,
    # drives the combined float (a regression to floating recall_block alone
    # would leak the private head onto a public chain).
    class _Mem(_HeadMemory):
        def context_for_llm(self, query):
            return "=== Past Conversations ===\nUser: what's the weather"

    llm = _RecordingLLM()
    mem = _Mem(head="The user's salary is $95,000 at Acme Corp.")
    reg = _assistant(llm, mem, recall=RecallConfig(enabled=True))
    ctx: dict = {"intent_kind": "assistant"}
    reg.invoke("assistant.answer", "what is the capital of france", ctx)
    assert ctx["sensitivity"] == PRIVATE
    assert "=== Last Session ===" in llm.systems[-1]
    assert "Past Conversations" in llm.systems[-1]


def test_vision_label_forces_private_through_combined_block():
    # A benign-TEXT screen recall (classifies as CODE/PUBLIC, not PRIVATE) must
    # STILL force the turn PRIVATE via the VISION_LABEL guard over the combined
    # block -- the §9.7 invariant this diff moved onto the combined path.
    benign = "a python tutorial about refactor and debug"
    assert classify_sensitivity(benign) != PRIVATE  # premise: text alone isn't private

    class _Mem(SessionMemory):
        def context_for_llm(self, query):
            return f"=== Screen Memory ===\nScreen: {benign}"

    llm = _RecordingLLM()
    reg = _assistant(llm, _Mem(), recall=RecallConfig(enabled=True))
    ctx: dict = {"intent_kind": "assistant"}
    reg.invoke("assistant.answer", "what is the capital of france", ctx)
    assert ctx["sensitivity"] == PRIVATE


# --- escalation-path exclusion + latch robustness (review-driven) ------------


def test_escalated_first_turn_does_not_consume_or_inject_recap():
    # An escalated (ReAct) first turn returns BEFORE the one-shot injection, so
    # it must neither inject the recap into the planner nor burn the one-shot
    # latch -- the recap must survive to the first one-shot turn.
    seen = {"planner_calls": 0}

    def _planner(query, context):
        seen["planner_calls"] += 1
        return CapabilityResult(True, "planned")

    llm = _RecordingLLM()
    mem = _HeadMemory(head="We discussed your trip to Berlin.")
    reg = CapabilityRegistry()
    reg.register("agent.react", _planner)
    escalated = {"on": True}
    attach_llm_capabilities(
        reg, llm, memory=mem, recent_context=_NO_RECENT,
        escalate=lambda q, ctx: escalated["on"],
    )

    reg.invoke("assistant.answer", "research the trip in depth", {})  # escalates
    assert seen["planner_calls"] == 1
    assert llm.systems == []  # planner stub never touched the LLM; nothing injected

    escalated["on"] = False
    reg.invoke("assistant.answer", "what should I pack", {})  # one-shot
    assert "=== Last Session ===" in llm.systems[-1]
    assert "Berlin" in llm.systems[-1]


def test_recap_latch_retries_when_head_raises_on_first_turn():
    # A transient failure fetching the head on turn 1 must NOT burn the one-shot
    # latch (the recap is retried next turn), since the latch now flips only once
    # the recap is actually built.
    calls = {"n": 0}

    class _FlakyHead(SessionMemory):
        def last_session_summary(self):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient head fetch failure")
            return "We discussed Berlin."

    llm = _RecordingLLM()
    reg = _assistant(llm, _FlakyHead(), recall=RecallConfig(enabled=False))

    reg.invoke("assistant.answer", "turn one", {})  # head raises -> swallowed
    assert "=== Last Session ===" not in (llm.systems[-1] or "")
    reg.invoke("assistant.answer", "turn two", {})  # retried -> recap appears
    assert "=== Last Session ===" in llm.systems[-1]
