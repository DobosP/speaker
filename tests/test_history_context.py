"""R11: prior turns as role-structured chat messages.

Covers the shared history helper, the LLM message-builders that insert history
between the system prompt and the current turn, and the end-to-end capabilities
path -- configured messages mode passes ``history`` to the model and drops the
pasted text block, while the programmatic fallback stays compatible with text.
"""
from __future__ import annotations

import json
from pathlib import Path

from always_on_agent.capabilities import create_default_capabilities
from always_on_agent.capabilities import CapabilityRegistry, CapabilityResult
from always_on_agent.memory import SessionMemory
from always_on_agent.recall import estimate_tokens

from core.capabilities import RecallConfig, attach_llm_capabilities
from core.conversation import RecentContextConfig, history_messages
from core.llm import EchoLLM, OllamaLLM, _history_messages, _openai_messages
from core.persona import build_system_prompt
from core.websearch import WebSearchConfig, attach_web_search_capability


# --- the shared normalizer + role mapping ------------------------------------


def test_history_messages_maps_and_drops_empties():
    turns = [("User", "hi there"), ("You", "hello"), ("User", "   "), ("You", "")]
    assert history_messages(turns) == [
        {"role": "user", "content": "hi there"},
        {"role": "assistant", "content": "hello"},
    ]


def test_history_messages_normalizer_filters_bad_roles():
    raw = [
        {"role": "user", "content": "a"},
        {"role": "system", "content": "nope"},   # only user/assistant allowed
        {"role": "assistant", "content": ""},      # empty dropped
        {"role": "assistant", "content": "b"},
    ]
    assert _history_messages(raw) == [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
    ]


# --- LLM message builders insert history between system + current turn --------


def test_ollama_messages_inserts_history():
    llm = OllamaLLM("m")  # constructor only; _messages needs no client
    msgs = llm._messages(
        "now", "SYS", None,
        [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}],
    )
    assert msgs == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "now"},
    ]


def test_ollama_messages_no_history_is_byte_identical():
    llm = OllamaLLM("m")
    assert llm._messages("now", "SYS", None) == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "now"},
    ]


def test_openai_messages_inserts_history():
    msgs = _openai_messages("now", "SYS", None, [{"role": "user", "content": "a"}])
    assert msgs == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "a"},
        {"role": "user", "content": "now"},
    ]


def test_echo_reflects_history_but_is_identical_without_it():
    assert EchoLLM().generate("hi") == "You said: hi"  # default -> byte-identical
    out = EchoLLM().generate("hi", history=[{"role": "user", "content": "x"}])
    assert "[+1 prior turn(s)]" in out


# --- end-to-end through the capability layer ---------------------------------


class _HistoryRecordingLLM:
    def __init__(self) -> None:
        self.systems: list = []
        self.histories: list = []

    def generate(self, prompt, *, system=None, images=None, history=None):
        self.systems.append(system)
        self.histories.append(history)
        return "ok"

    def stream(self, prompt, *, system=None, images=None, history=None):
        self.systems.append(system)
        self.histories.append(history)
        yield "ok"


def _mem_with_prior_turns() -> SessionMemory:
    mem = SessionMemory()
    mem.add("tell me a story", tags=("user",))
    mem.add("Once upon a time there was a dragon.", tags=("assistant_output",))
    return mem


def test_messages_mode_passes_history_and_drops_text_block():
    llm = _HistoryRecordingLLM()
    mem = _mem_with_prior_turns()
    reg = CapabilityRegistry()
    attach_llm_capabilities(
        reg, llm, memory=mem,
        recent_context=RecentContextConfig(as_messages=True),
    )
    reg.invoke("assistant.answer", "continue it", {})
    # The PRIOR turns reach the model as role-structured messages (the current
    # query is the prompt, not part of history).
    assert llm.histories[-1] == [
        {"role": "user", "content": "tell me a story"},
        {"role": "assistant", "content": "Once upon a time there was a dragon."},
    ]
    # ...and the recent-conversation TEXT block is NOT pasted into the system prompt.
    assert "Recent conversation" not in (llm.systems[-1] or "")


def test_messages_mode_still_supplies_text_context_to_escalated_planner():
    llm = _HistoryRecordingLLM()
    mem = _mem_with_prior_turns()
    reg = CapabilityRegistry()
    seen: dict = {}
    reg.register(
        "agent.react",
        lambda _query, context: seen.update(context)
        or CapabilityResult(True, "planned"),
    )
    attach_llm_capabilities(
        reg,
        llm,
        memory=mem,
        recent_context=RecentContextConfig(as_messages=True),
        escalate=lambda _query, _context: True,
    )

    result = reg.invoke("assistant.answer", "continue it", {})

    assert result.text == "planned"
    assert "tell me a story" in seen["recent_conversation"]
    assert "dragon" in seen["recent_conversation"]


def test_messages_mode_honors_recent_token_reserve():
    llm = _HistoryRecordingLLM()
    mem = SessionMemory()
    mem.add("x" * 60, tags=("user",))
    mem.add("y" * 60, tags=("assistant_output",))
    reg = CapabilityRegistry()
    attach_llm_capabilities(
        reg,
        llm,
        memory=mem,
        recent_context=RecentContextConfig(
            as_messages=True,
            reserve_tokens=25,
        ),
    )

    reg.invoke("assistant.answer", "continue", {})

    history = llm.histories[-1]
    assert history == [{"role": "assistant", "content": "y" * 60}]
    rendered = "\n".join(
        f"{item['role']}: {item['content']}" for item in history
    )
    assert estimate_tokens(rendered) <= 25


def test_programmatic_default_keeps_legacy_text_mode():
    assert RecentContextConfig().as_messages is False


def test_shipped_history_budget_leaves_headroom_on_smallest_profile():
    config = json.loads(
        (Path(__file__).parents[1] / "config.json").read_text(encoding="utf-8")
    )
    memory = config["memory"]
    phone_lite = config["device_profiles"]["phone_lite"]["llm"]
    registry = create_default_capabilities(SessionMemory())
    attach_web_search_capability(registry, WebSearchConfig())
    system_tokens = estimate_tokens(
        build_system_prompt(registry, web_enabled=False)
    )

    committed = (
        system_tokens
        + int(memory["recall_max_tokens"])
        + int(memory["recall_recent_reserve_tokens"])
        + int(phone_lite["options"]["num_predict"])
    )

    assert int(phone_lite["n_ctx"]) - committed >= 300


def test_explicit_text_mode_pastes_block_and_passes_no_history():
    llm = _HistoryRecordingLLM()
    mem = _mem_with_prior_turns()
    reg = CapabilityRegistry()
    attach_llm_capabilities(
        reg, llm, memory=mem,
        recent_context=RecentContextConfig(as_messages=False),
    )
    reg.invoke("assistant.answer", "continue it", {})
    assert llm.histories[-1] is None  # no structured history in text mode
    assert "Recent conversation" in (llm.systems[-1] or "")  # pasted into system
