"""R11: prior turns as role-structured chat messages (opt-in ``as_messages``).

Covers the shared history helper, the LLM message-builders that insert history
between the system prompt and the current turn, and the end-to-end capabilities
path -- messages mode passes ``history`` to the model and drops the pasted text
block, while the default (text) mode stays byte-identical.
"""
from __future__ import annotations

from always_on_agent.capabilities import CapabilityRegistry
from always_on_agent.memory import SessionMemory

from core.capabilities import RecallConfig, attach_llm_capabilities
from core.conversation import RecentContextConfig, history_messages
from core.llm import EchoLLM, OllamaLLM, _history_messages, _openai_messages


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


def test_text_mode_default_pastes_block_and_passes_no_history():
    llm = _HistoryRecordingLLM()
    mem = _mem_with_prior_turns()
    reg = CapabilityRegistry()
    attach_llm_capabilities(
        reg, llm, memory=mem,
        recent_context=RecentContextConfig(),  # default: text mode
    )
    reg.invoke("assistant.answer", "continue it", {})
    assert llm.histories[-1] is None  # no structured history in text mode
    assert "Recent conversation" in (llm.systems[-1] or "")  # pasted into system
