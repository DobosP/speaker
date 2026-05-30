"""Tests for recent-conversation context (ask #2: the model's short-term memory).

The answering model now sees the last few user/assistant turns, so it can resolve
references ("its", "the second one") to what was just said. These pin the builder
and the assistant() injection (with a recording LLM; no real model needed).
"""
from __future__ import annotations

from always_on_agent.capabilities import CapabilityRegistry
from always_on_agent.memory import SessionMemory

from core.capabilities import DEFAULT_SYSTEM, attach_llm_capabilities
from core.conversation import RecentContextConfig, build_recent_context


# --- the builder -------------------------------------------------------------


def _convo() -> SessionMemory:
    m = SessionMemory()
    m.add("what is the capital of france", tags=("user",))
    m.add("Paris.", tags=("assistant_output",))
    m.add("what is its population", tags=("user",))
    m.add("About 2.1 million.", tags=("assistant_output",))
    return m


def test_builds_block_from_user_and_assistant_turns_in_order():
    block = build_recent_context(_convo(), RecentContextConfig())
    assert "Recent conversation" in block
    assert block.index("what is the capital of france") < block.index("Paris.")
    assert block.index("Paris.") < block.index("what is its population")
    assert "User: what is the capital of france" in block
    assert "You: Paris." in block


def test_ambient_and_other_tags_are_excluded():
    m = SessionMemory()
    m.add("the user mumbling to themselves", tags=("ingested",))
    m.add("a stored meeting note", tags=("meeting", "note"))
    m.add("real question", tags=("user",))
    block = build_recent_context(m, RecentContextConfig())
    assert "real question" in block
    assert "mumbling" not in block
    assert "meeting note" not in block


def test_empty_or_disabled_or_none_memory_returns_empty():
    assert build_recent_context(SessionMemory(), RecentContextConfig()) == ""
    assert build_recent_context(_convo(), RecentContextConfig(enabled=False)) == ""
    assert build_recent_context(None, RecentContextConfig()) == ""


def test_max_turns_keeps_the_most_recent():
    block = build_recent_context(_convo(), RecentContextConfig(max_turns=2))
    assert "what is its population" in block
    assert "About 2.1 million." in block
    assert "capital of france" not in block  # oldest dropped


def test_max_chars_drops_oldest_until_it_fits():
    block = build_recent_context(_convo(), RecentContextConfig(max_chars=70))
    assert len(block) <= 70 + len("About 2.1 million.")  # header + at least the newest turn
    assert "About 2.1 million." in block  # newest kept
    assert "capital of france" not in block  # oldest dropped


def test_per_turn_truncation():
    m = SessionMemory()
    m.add("x" * 500, tags=("user",))
    block = build_recent_context(m, RecentContextConfig(per_turn_chars=50, max_chars=400))
    assert "x" * 50 in block
    assert "x" * 51 not in block


def test_config_from_dict_defaults_on():
    assert RecentContextConfig.from_dict(None).enabled is True
    cfg = RecentContextConfig.from_dict(
        {"recent_context_enabled": False, "recent_context_turns": 3}
    )
    assert cfg.enabled is False
    assert cfg.max_turns == 3


# --- assistant() injection ---------------------------------------------------


class _RecordingLLM:
    def __init__(self):
        self.systems: list = []

    def generate(self, prompt, *, system=None, images=None):
        self.systems.append(system)
        return "ok"

    def stream(self, prompt, *, system=None, images=None):
        self.systems.append(system)
        yield "ok"


def _assistant(llm, memory, **kw):
    reg = CapabilityRegistry()
    attach_llm_capabilities(reg, llm, memory=memory, **kw)
    return reg


def test_assistant_injects_recent_conversation_on_a_later_turn():
    llm = _RecordingLLM()
    mem = SessionMemory()
    reg = _assistant(llm, mem, recent_context=RecentContextConfig(enabled=True))

    reg.invoke("assistant.answer", "what is the capital of france", {})
    mem.add("Paris.", tags=("assistant_output",))  # the supervisor remembers replies
    reg.invoke("assistant.answer", "what is its population", {})

    # Turn 1 had nothing prior -> exactly the base system.
    assert "Recent conversation" not in llm.systems[0]
    assert llm.systems[0] == DEFAULT_SYSTEM
    # Turn 2 sees the prior turn so "its" has a referent. The stable system
    # prompt stays FIRST (cacheable prefix); the volatile block is appended.
    sys2 = llm.systems[-1]
    assert sys2.startswith(DEFAULT_SYSTEM)
    assert "Recent conversation" in sys2
    assert "what is the capital of france" in sys2
    assert "Paris." in sys2


def test_assistant_disabled_recent_context_is_unchanged():
    llm = _RecordingLLM()
    mem = SessionMemory()
    reg = _assistant(llm, mem, recent_context=RecentContextConfig(enabled=False))

    reg.invoke("assistant.answer", "first question", {})
    mem.add("an answer.", tags=("assistant_output",))
    reg.invoke("assistant.answer", "second question", {})

    assert all(s == DEFAULT_SYSTEM for s in llm.systems)


def test_assistant_suppresses_recent_context_on_a_continuation_turn():
    # A continuation turn's synthetic prompt already embeds the prior context, so
    # the recent block is suppressed (no double-injection).
    llm = _RecordingLLM()
    mem = SessionMemory()
    mem.add("what is the capital of france", tags=("user",))
    mem.add("Paris.", tags=("assistant_output",))
    reg = _assistant(llm, mem, recent_context=RecentContextConfig(enabled=True))

    reg.invoke("assistant.answer", "and also Germany", {"metadata": {"skip_user_memory": True}})
    assert "Recent conversation" not in llm.systems[-1]


def test_assistant_floats_sensitivity_over_a_private_prior_turn():
    # §9.7: a public current query must route private when a prior turn was private.
    from core.sensitivity import PRIVATE, PUBLIC

    llm = _RecordingLLM()
    mem = SessionMemory()
    mem.add("my bank account number is 12345", tags=("user",))  # private prior turn
    mem.add("Noted.", tags=("assistant_output",))
    reg = _assistant(llm, mem, recent_context=RecentContextConfig(enabled=True))

    ctx: dict = {"intent_kind": "assistant"}  # truthy so invoke passes it through
    reg.invoke("assistant.answer", "what is the capital of france", ctx)  # public current
    assert ctx["sensitivity"] == PRIVATE  # floated up by the private prior turn


def test_abstain_and_apology_replies_are_excluded_from_context():
    mem = SessionMemory()
    mem.add("real question", tags=("user",))
    mem.add("Sorry, I don't have an answer for that.", tags=("assistant_output",))
    mem.add("Sorry, that took too long -- let's try again.", tags=("assistant_output",))
    block = build_recent_context(mem, RecentContextConfig())
    assert "real question" in block
    assert "Sorry," not in block  # placeholder/abstain replies carry no content
