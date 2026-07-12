"""Tests for recent-conversation context (ask #2: the model's short-term memory).

The answering model now sees the last few user/assistant turns, so it can resolve
references ("its", "the second one") to what was just said. These pin the builder
and the assistant() injection (with a recording LLM; no real model needed).
"""
from __future__ import annotations

from always_on_agent.capabilities import CapabilityRegistry
from always_on_agent.memory import SessionMemory

from core.capabilities import DEFAULT_SYSTEM, attach_llm_capabilities
from core.conversation import (
    RecentContextConfig,
    build_recent_context,
    collect_recent_turns,
    is_topic_reset,
)


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
    assert RecentContextConfig.from_dict(None).as_messages is False
    cfg = RecentContextConfig.from_dict(
        {"recent_context_enabled": False, "recent_context_turns": 3}
    )
    assert cfg.enabled is False
    assert cfg.max_turns == 3


# --- topic reset (2026-06-10 plan step 5) -------------------------------------


def test_is_topic_reset_matches_short_reset_utterances():
    cfg = RecentContextConfig()
    assert is_topic_reset("never mind", cfg)
    assert is_topic_reset("Forget it", cfg)
    assert is_topic_reset("OK, new topic", cfg)
    assert is_topic_reset("can we change the subject", cfg)  # contained phrase
    assert is_topic_reset("alt subiect", cfg)  # Romanian
    assert is_topic_reset("Las-o baltă", cfg)  # accented input normalizes


def test_is_topic_reset_rejects_long_or_unrelated_utterances():
    cfg = RecentContextConfig()
    # A long sentence merely MENTIONING a phrase is not a reset.
    assert not is_topic_reset(
        "tell me the story where the hero says never mind and walks away", cfg
    )
    assert not is_topic_reset("what is the capital of france", cfg)
    assert not is_topic_reset("", cfg)
    # Phrase words present but NOT contiguous.
    assert not is_topic_reset("forget the engine that", cfg)
    # Resume-like phrases are NOT resets (owner 2026-06-10: "start again" after
    # a cut means CONTINUE the interrupted reply -- core/resume.py owns it).
    assert not is_topic_reset("start again", cfg)
    assert not is_topic_reset("start over", cfg)
    assert not is_topic_reset("continue", cfg)


def test_is_topic_reset_disabled_never_matches():
    cfg = RecentContextConfig(reset_enabled=False)
    assert not is_topic_reset("never mind", cfg)


def _stale_then_reset_then_new() -> SessionMemory:
    """The live failure shape: a stale topic, an explicit reset, a new turn."""
    m = SessionMemory()
    m.add("no i was referring to our interaction the volume", tags=("user",))
    m.add("I apologize for the misunderstanding! ... the volume?", tags=("assistant_output",))
    m.add("Never mind", tags=("user",))
    m.add("Okay, fresh start!", tags=("assistant_output",))
    m.add("tell me a story about a lighthouse", tags=("user",))
    return m


def test_reset_turn_in_memory_cuts_the_block_at_the_reset():
    turns = collect_recent_turns(_stale_then_reset_then_new(), RecentContextConfig())
    texts = [t for _, t in turns]
    # Everything at/before "Never mind" is gone; the post-reset turns survive.
    assert "Okay, fresh start!" in texts
    assert "tell me a story about a lighthouse" in texts
    assert not any("volume" in t for t in texts)
    assert not any("Never mind" in t for t in texts)


def test_reset_current_query_suppresses_the_block_for_its_own_turn():
    m = SessionMemory()
    m.add("no i was referring to our interaction the volume", tags=("user",))
    m.add("I apologize ... the volume?", tags=("assistant_output",))
    # The reset turn ITSELF must answer fresh -- the live failure replied with
    # the stale volume apology because the block still carried it.
    assert collect_recent_turns(m, RecentContextConfig(), current_query="Never mind") == []
    # A non-reset query still sees the thread.
    assert collect_recent_turns(m, RecentContextConfig(), current_query="and the volume?") != []


def test_reset_disabled_keeps_the_old_behaviour():
    cfg = RecentContextConfig(reset_enabled=False)
    turns = collect_recent_turns(_stale_then_reset_then_new(), cfg)
    texts = [t for _, t in turns]
    assert any("volume" in t for t in texts)  # stale topic kept (old behaviour)
    assert collect_recent_turns(
        _stale_then_reset_then_new(), cfg, current_query="Never mind"
    ) != []


def test_reset_config_from_dict():
    cfg = RecentContextConfig.from_dict(
        {
            "recent_context_reset_enabled": False,
            "recent_context_reset_max_words": 4,
            "recent_context_reset_phrases": ["wipe it"],
        }
    )
    assert cfg.reset_enabled is False
    assert cfg.reset_max_words == 4
    assert cfg.reset_phrases == ("wipe it",)
    # Defaults: on, 8 words, the shipped phrase table.
    dflt = RecentContextConfig.from_dict(None)
    assert dflt.reset_enabled is True
    assert dflt.reset_max_words == 8
    assert "never mind" in dflt.reset_phrases
    # Resume-like phrases must NOT be in the default reset table (owner
    # 2026-06-10: they mean CONTINUE the interrupted reply).
    assert "start again" not in dflt.reset_phrases
    assert "start over" not in dflt.reset_phrases


def test_assistant_reset_query_answers_without_stale_context():
    """End-to-end through assistant(): 'Never mind' must NOT see the old topic."""
    llm = _RecordingLLM()
    mem = SessionMemory()
    mem.add("no i was referring to our interaction the volume", tags=("user",))
    mem.add("I apologize for the misunderstanding! ... the volume?", tags=("assistant_output",))
    reg = _assistant(
        llm,
        mem,
        recent_context=RecentContextConfig(enabled=True, as_messages=True),
    )

    reg.invoke("assistant.answer", "Never mind", {})
    assert "Recent conversation" not in llm.systems[-1]
    assert "volume" not in llm.systems[-1]
    assert llm.histories[-1] is None

    # And the NEXT turn does not resurrect the pre-reset topic either.
    mem.add("Okay, fresh start!", tags=("assistant_output",))
    reg.invoke("assistant.answer", "tell me a story about a lighthouse", {})
    assert "volume" not in llm.systems[-1]
    assert llm.histories[-1] == [
        {"role": "assistant", "content": "Okay, fresh start!"},
    ]


# --- assistant() injection ---------------------------------------------------


class _RecordingLLM:
    def __init__(self):
        self.systems: list = []
        self.histories: list = []
        self.prompts: list[str] = []

    def generate(self, prompt, *, system=None, images=None, history=None):
        self.prompts.append(prompt)
        self.systems.append(system)
        self.histories.append(history)
        return "ok"

    def stream(self, prompt, *, system=None, images=None, history=None):
        self.prompts.append(prompt)
        self.systems.append(system)
        self.histories.append(history)
        yield "ok"


def _assistant(llm, memory, **kw):
    reg = CapabilityRegistry()
    attach_llm_capabilities(reg, llm, memory=memory, **kw)
    return reg


def test_session_fact_command_uses_fixed_ack_and_role_history():
    llm = _RecordingLLM()
    mem = SessionMemory()
    reg = _assistant(llm, mem, recent_context=RecentContextConfig())
    query = "Remember for this conversation that the project codename is Orion."
    spoken: list[str] = []

    result = reg.invoke("assistant.answer", query, {"emit_speech": spoken.append})

    assert result.ok is True
    assert result.data["route"] == "control"
    assert spoken == ["Okay, I'll remember that for this conversation."]
    assert llm.systems == []
    assert any(
        item.text == query and "user" in item.tags
        for item in mem.all()
    )

    mem.add(result.text, tags=("assistant_output",))
    followup = reg.invoke(
        "assistant.answer",
        "What is the project codename?",
        {"emit_speech": spoken.append},
    )
    assert followup.text == "Orion."
    assert followup.data["route"] == "control"
    assert followup.data["session_fact"] is True
    assert spoken[-1] == "Orion."
    assert llm.systems == []


def test_session_fact_does_not_swallow_compound_followup():
    llm = _RecordingLLM()
    mem = SessionMemory()
    reg = _assistant(llm, mem)
    reg.invoke(
        "assistant.answer",
        "Remember for this conversation that the project codename is Orion.",
        {},
    )

    compound = (
        "What is the project codename? Also what is France's capital?"
    )
    result = reg.invoke("assistant.answer", compound, {})

    assert result.text == "ok"
    assert llm.prompts[-1] == compound


def test_session_fact_keys_preserve_meaningful_punctuation():
    llm = _RecordingLLM()
    mem = SessionMemory()
    reg = _assistant(llm, mem)
    reg.invoke(
        "assistant.answer",
        "Remember for this conversation that the C++ version is twenty.",
        {},
    )
    reg.invoke(
        "assistant.answer",
        "Remember for this conversation that the C# version is twelve.",
        {},
    )

    assert reg.invoke("assistant.answer", "What is the C++ version?", {}).text == "twenty."
    assert reg.invoke("assistant.answer", "What is the C# version?", {}).text == "twelve."


def test_session_fact_expires_with_memory_and_cache_is_bounded():
    llm = _RecordingLLM()
    mem = SessionMemory(max_items=64)
    reg = _assistant(llm, mem)
    for index in range(17):
        reg.invoke(
            "assistant.answer",
            f"Remember for this conversation that item {index} is value{index}.",
            {},
        )

    assert reg.invoke("assistant.answer", "What is item 0?", {}).text == "ok"
    assert reg.invoke("assistant.answer", "What is item 16?", {}).text == "value16."

    evicting_memory = SessionMemory(max_items=2)
    evicting_llm = _RecordingLLM()
    evicting = _assistant(evicting_llm, evicting_memory)
    evicting.invoke(
        "assistant.answer",
        "Remember for this conversation that the codename is Orion.",
        {},
    )
    evicting_memory.add("newer one", tags=("user",))
    evicting_memory.add("newer two", tags=("assistant_output",))

    assert evicting.invoke(
        "assistant.answer", "What is the codename?", {}
    ).text == "ok"


def test_response_only_exact_word_is_controller_bounded():
    llm = _RecordingLLM()
    spoken: list[str] = []
    reg = _assistant(llm, SessionMemory())

    result = reg.invoke(
        "assistant.answer",
        "Actually, respond with only the word Tokyo.",
        {
            "emit_speech": spoken.append,
            "metadata": {"post_barge_response_only": True},
        },
    )

    assert result.text == "Tokyo."
    assert result.data["handled_local"] is True
    assert spoken == ["Tokyo."]
    assert llm.systems == []


def test_exact_one_word_and_repeat_are_controller_bounded():
    llm = _RecordingLLM()
    memory = SessionMemory()
    spoken: list[str] = []
    reg = _assistant(llm, memory)

    first = reg.invoke(
        "assistant.answer",
        "Say exactly one word: Orion.",
        {"emit_speech": spoken.append},
    )
    memory.add(first.text, tags=("assistant_output",))
    second = reg.invoke(
        "assistant.answer",
        "Repeat your previous answer exactly.",
        {"emit_speech": spoken.append},
    )

    assert first.text == second.text == "Orion."
    assert first.data["exact_word"] is True
    assert second.data["repeat_previous"] is True
    assert first.data["route"] == second.data["route"] == "control"
    assert spoken == ["Orion.", "Orion."]
    assert llm.systems == []


def test_repeat_without_a_previous_answer_falls_through_to_model():
    llm = _RecordingLLM()
    memory = SessionMemory()
    memory.add("A stale answer from an earlier session.", tags=("assistant_output",))
    reg = _assistant(llm, memory)

    result = reg.invoke(
        "assistant.answer",
        "Repeat your previous answer exactly.",
        {},
    )

    assert result.text == "ok"
    assert llm.prompts == ["Repeat your previous answer exactly."]


def test_assistant_injects_recent_conversation_on_a_later_turn():
    llm = _RecordingLLM()
    mem = SessionMemory()
    reg = _assistant(
        llm,
        mem,
        recent_context=RecentContextConfig(enabled=True, as_messages=True),
    )

    reg.invoke("assistant.answer", "what is the capital of france", {})
    mem.add("Paris.", tags=("assistant_output",))  # the supervisor remembers replies
    reg.invoke("assistant.answer", "what is its population", {})

    # Turn 1 had nothing prior -> exactly the base system.
    assert "Recent conversation" not in llm.systems[0]
    assert llm.systems[0] == DEFAULT_SYSTEM
    # Turn 2 sees the prior turn so "its" has a referent. The stable system
    # prompt remains unchanged while role-structured history carries the thread.
    sys2 = llm.systems[-1]
    assert sys2 == DEFAULT_SYSTEM
    assert llm.histories[-1] == [
        {"role": "user", "content": "what is the capital of france"},
        {"role": "assistant", "content": "Paris."},
    ]


def test_assistant_disabled_recent_context_is_unchanged():
    llm = _RecordingLLM()
    mem = SessionMemory()
    reg = _assistant(
        llm,
        mem,
        recent_context=RecentContextConfig(enabled=False, as_messages=True),
    )

    reg.invoke("assistant.answer", "first question", {})
    mem.add("an answer.", tags=("assistant_output",))
    reg.invoke("assistant.answer", "second question", {})

    assert all(s == DEFAULT_SYSTEM for s in llm.systems)
    assert all(history is None for history in llm.histories)


def test_assistant_suppresses_recent_context_on_a_continuation_turn():
    # A continuation turn's synthetic prompt already embeds the prior context, so
    # the recent block is suppressed (no double-injection).
    llm = _RecordingLLM()
    mem = SessionMemory()
    mem.add("what is the capital of france", tags=("user",))
    mem.add("Paris.", tags=("assistant_output",))
    reg = _assistant(
        llm,
        mem,
        recent_context=RecentContextConfig(enabled=True, as_messages=True),
    )

    reg.invoke("assistant.answer", "and also Germany", {"metadata": {"skip_user_memory": True}})
    assert "Recent conversation" not in llm.systems[-1]
    assert llm.histories[-1] is None


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


def test_assistant_floats_sensitivity_over_a_private_recall_block():
    # §9.7 (review lm-3): a PUBLIC current query whose RECALL block surfaces
    # private remembered facts must route the turn on the private chain. The
    # recall path previously skipped the sensitivity float that recent-turns
    # and images get.
    from core.capabilities import RecallConfig
    from core.sensitivity import PRIVATE

    class _RecallingMemory(SessionMemory):
        def context_for_llm(self, query: str) -> str:
            return "Remembered: the user's salary is $95,000 at Acme Corp."

    llm = _RecordingLLM()
    mem = _RecallingMemory()
    reg = _assistant(
        llm, mem,
        recall=RecallConfig(enabled=True),
        recent_context=RecentContextConfig(enabled=False),
    )

    ctx: dict = {"intent_kind": "assistant"}
    reg.invoke("assistant.answer", "what is the capital of france", ctx)
    assert ctx["sensitivity"] == PRIVATE  # floated by the private recall block
    # The block itself still reaches the prompt (contract unchanged).
    assert "Remembered:" in llm.systems[-1]


def test_abstain_and_apology_replies_are_excluded_from_context():
    mem = SessionMemory()
    mem.add("real question", tags=("user",))
    mem.add("Sorry, I don't have an answer for that.", tags=("assistant_output",))
    mem.add("Sorry, that took too long -- let's try again.", tags=("assistant_output",))
    block = build_recent_context(mem, RecentContextConfig())
    assert "real question" in block
    assert "Sorry," not in block  # placeholder/abstain replies carry no content
