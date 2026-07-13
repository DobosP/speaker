"""Tests for the input-gate addressing classifier (core.addressing) and its
integration with VoiceRuntime.

Two layers:

* unit tests for :class:`LLMAddressingClassifier` parsing / fallback behavior
  (driven by a hand-coded LLM stub, no real model);
* end-to-end tests that drive a final into a real :class:`VoiceRuntime` with
  a :class:`ScriptedAddressingClassifier` and assert the brain *did* or *did
  not* see the utterance (mirrors the existing scripted-engine test style in
  ``test_core_runtime.py``).
"""
from __future__ import annotations

from typing import Iterator, Optional, Sequence

from always_on_agent.events import Mode

from core.addressing import (
    ACT,
    INGEST,
    UNSURE,
    LLMAddressingClassifier,
    ScriptedAddressingClassifier,
    _parse_decision,
)
from core.engines.scripted import ScriptedEngine
from core.llm import EchoLLM
from core.runtime import VoiceRuntime


# --- LLMAddressingClassifier unit tests --------------------------------------

class _StubLLM:
    """Minimal LLMClient stub: returns the next reply from a queue, captures
    every prompt+system it was called with."""

    def __init__(self, replies: Sequence[str]) -> None:
        self._replies = list(replies)
        self.calls: list[tuple[str, Optional[str]]] = []

    def generate(self, prompt: str, *, system: Optional[str] = None, images=None) -> str:
        self.calls.append((prompt, system))
        if not self._replies:
            raise AssertionError("StubLLM ran out of scripted replies")
        return self._replies.pop(0)

    def stream(self, prompt: str, *, system=None, images=None) -> Iterator[str]:
        yield self.generate(prompt, system=system, images=images)


class _RaisingLLM:
    def generate(self, prompt, *, system=None, images=None):
        raise RuntimeError("ollama is down")

    def stream(self, prompt, *, system=None, images=None):
        raise RuntimeError("ollama is down")


def test_parse_decision_handles_clean_replies():
    assert _parse_decision("ACT") == ACT
    assert _parse_decision("INGEST") == INGEST
    assert _parse_decision("UNSURE") == UNSURE


def test_parse_decision_strips_punctuation_and_case():
    assert _parse_decision("Act.") == ACT
    assert _parse_decision("  ingest!\n") == INGEST
    assert _parse_decision('"UNSURE"') == UNSURE


def test_parse_decision_accepts_bounded_action_alias_only():
    assert _parse_decision("ACTION") == ACT
    assert _parse_decision("Action.") == ACT
    assert _parse_decision("ACTIVE") == ACT
    assert _parse_decision("active.") == ACT
    assert _parse_decision("ACTIONABLE") == UNSURE
    assert _parse_decision("ACTIVELY") == UNSURE
    assert _parse_decision("ACTIVITY") == UNSURE
    assert _parse_decision("INACTIVE") == UNSURE


def test_parse_decision_defaults_to_unsure_for_garbage():
    assert _parse_decision("") == UNSURE
    assert _parse_decision("   ") == UNSURE
    assert _parse_decision("hmm let me think about that") == UNSURE
    assert _parse_decision("the assistant should respond") == UNSURE  # not first word


def test_classifier_returns_unsure_when_llm_raises():
    cls = LLMAddressingClassifier(_RaisingLLM())
    assert cls.classify("what time is it") == UNSURE


def test_classifier_normalizes_exact_active_alias():
    llm = _StubLLM(["ACTIVE"])
    cls = LLMAddressingClassifier(llm)

    assert cls.classify("What is the capital of France?") == ACT
    assert len(llm.calls) == 1


def test_classifier_uses_system_prompt_and_includes_context():
    llm = _StubLLM(["ACT"])
    cls = LLMAddressingClassifier(llm)
    decision = cls.classify("what time is it", recent=["hello there", "any updates?"])
    assert decision == ACT
    prompt, system = llm.calls[0]
    assert system is not None and "addressing gate" in system.lower()
    assert "what time is it" in prompt
    # Recent context is included so the LLM can disambiguate.
    assert "hello there" in prompt
    assert "any updates?" in prompt


def test_classifier_truncates_context_to_max():
    llm = _StubLLM(["INGEST"])
    cls = LLMAddressingClassifier(llm, max_context=2)
    cls.classify("end", recent=["alpha", "bravo", "charlie", "delta", "echo"])
    prompt, _ = llm.calls[0]
    # Only the last two recents should appear.
    assert "delta" in prompt and "echo" in prompt
    assert "alpha" not in prompt
    assert "bravo" not in prompt
    assert "charlie" not in prompt


def test_classifier_short_circuits_high_precision_imperatives():
    llm = _StubLLM([])
    classifier = LLMAddressingClassifier(llm)

    for text in (
        "Remember for this conversation that the codename is Orion.",
        "Look up Pipecat using your tools.",
        "Search for current Pipecat releases with your tool.",
        "Research Pipecat and LiveKit using your tools.",
        "Please search for Pipecat.",
        "Please research Pipecat and LiveKit.",
        "Repeat your previous answer exactly.",
        "Say exactly three short sentences: Blue. White. Red.",
        "What is the project codename? Answer with the codename.",
        "What is the capital of France, I mean Japan?",
        "Which country contains the city you just named?",
    ):
        assert classifier.classify(text) == ACT

    assert llm.calls == []


def test_classifier_keeps_ambiguous_statements_on_the_learned_gate():
    statements = (
        "I remember the old kitchen table.",
        "I heard him say hello.",
        "I heard a question? Answer was no.",
        "Research shows the result was negative.",
        "Search results were inconclusive.",
        "Name is only a label.",
        "Open source software is useful.",
        "Set theory is abstract.",
        "Resume formatting matters.",
        "Please is a polite word.",
        "What a lovely day.",
        "What time is dinner?",
        "Look up is a phrasal verb.",
        "Look up tables improve database joins.",
        "She said, look up Pipecat using your tools.",
        "Our guide explains how to research Pipecat with your tools.",
        "Research ethics matter.",
        "The quiz read: what is two plus two? Answer with a number.",
        "What is two plus two? Answer with a number, she read aloud.",
        'He asked, "Which country contains the city you just named?"',
    )
    llm = _StubLLM(["INGEST"] * len(statements))
    classifier = LLMAddressingClassifier(llm)

    assert all(classifier.classify(text) == INGEST for text in statements)
    assert len(llm.calls) == len(statements)


# --- VoiceRuntime integration tests ------------------------------------------

def _runtime_with_gate(
    decisions: Optional[dict[str, str]] = None,
    *,
    default: str = UNSURE,
    unsure_acts: bool = True,
    reply: str = "an answer",
):
    """Build a runtime with a scripted classifier, ready to receive finals."""
    engine = ScriptedEngine()
    gate = ScriptedAddressingClassifier(decisions or {}, default=default)
    runtime = VoiceRuntime(
        engine,
        EchoLLM(reply=reply),
        start_mode=Mode.ASSISTANT,
        addressing=gate,
        unsure_acts=unsure_acts,
    )
    runtime.start(run_bus=False)
    return runtime, engine, gate


def test_act_lets_the_utterance_through_to_the_brain():
    runtime, engine, _ = _runtime_with_gate({"what time is it": ACT})
    engine.final("what time is it")
    assert runtime.wait_idle()
    assert engine.spoken == ["an answer"]


def test_ingest_drops_the_utterance_and_writes_memory():
    runtime, engine, gate = _runtime_with_gate(
        {"HE MURMURED HIS MURDERING": INGEST}
    )
    engine.final("HE MURMURED HIS MURDERING")
    assert runtime.wait_idle()
    assert engine.spoken == []  # the brain never produced a reply
    # The utterance still landed in memory (so the next ACT decision sees it).
    mem_texts = [item.text for item in runtime.memory.all()]
    assert "HE MURMURED HIS MURDERING" in mem_texts
    # And memory carries the 'ingested' tag for downstream tooling.
    ingested = [item for item in runtime.memory.all() if "ingested" in item.tags]
    assert any(item.text == "HE MURMURED HIS MURDERING" for item in ingested)
    # The classifier saw the call.
    assert gate.calls and gate.calls[0][0] == "HE MURMURED HIS MURDERING"


def test_unsure_acts_by_default():
    """The default policy treats UNSURE as ACT (don't drop real queries on
    classifier uncertainty)."""
    runtime, engine, _ = _runtime_with_gate(default=UNSURE)
    engine.final("uh hmm yeah")
    assert runtime.wait_idle()
    assert engine.spoken == ["an answer"]


def test_unsure_ingests_when_policy_is_conservative():
    runtime, engine, _ = _runtime_with_gate(default=UNSURE, unsure_acts=False)
    engine.final("uh hmm yeah")
    assert runtime.wait_idle()
    assert engine.spoken == []  # silenced; conservative policy
    assert any(
        item.text == "uh hmm yeah" and "ingested" in item.tags
        for item in runtime.memory.all()
    )


def test_active_alias_acts_when_policy_is_conservative():
    engine = ScriptedEngine()
    gate = LLMAddressingClassifier(_StubLLM(["ACTIVE"]))
    runtime = VoiceRuntime(
        engine,
        EchoLLM(reply="Paris."),
        start_mode=Mode.ASSISTANT,
        addressing=gate,
        unsure_acts=False,
    )
    runtime.start(run_bus=False)

    engine.final("What is the capital of France?")

    assert runtime.wait_idle()
    assert engine.spoken == ["Paris."]


def test_no_classifier_preserves_legacy_behavior():
    """When no addressing classifier is wired in, the runtime behaves exactly
    as before -- every clean final reaches the brain."""
    engine = ScriptedEngine()
    runtime = VoiceRuntime(engine, EchoLLM(reply="legacy"), start_mode=Mode.ASSISTANT)
    runtime.start(run_bus=False)
    engine.final("anything goes")
    assert runtime.wait_idle()
    assert engine.spoken == ["legacy"]


def test_ingested_utterance_does_not_open_a_metrics_turn():
    """A turn with no LLM/TTS stamps would trip the stuck watchdog later, so
    INGEST'd utterances must not call metrics.mark(ASR_FINAL)."""
    runtime, engine, _ = _runtime_with_gate({"background noise here": INGEST})
    engine.final("background noise here")
    assert runtime.wait_idle()
    assert runtime.metrics.records() == []  # no turn opened


def test_ingested_utterance_marks_engine_opened_turn_handled_local():
    """Production engines stamp SPEECH_END before the addressing gate. If that
    text is INGESTed, mark the open turn locally handled so watchdog summaries
    do not report a fake stuck turn."""
    from core.metrics import HANDLED_LOCAL, SPEECH_END

    runtime, engine, _ = _runtime_with_gate({"background noise here": INGEST})
    runtime.metrics.mark(SPEECH_END)
    engine.final("background noise here")
    assert runtime.wait_idle()
    [record] = runtime.metrics.records()
    assert HANDLED_LOCAL in record.stamps


def test_classifier_sees_recent_memory_as_context():
    """The runtime feeds the last few memory items to the classifier so it can
    disambiguate. Ingested utterances flow into memory via the runtime; later
    finals see them in their ``recent`` context."""
    runtime, engine, gate = _runtime_with_gate(
        {"BACKGROUND ONE": INGEST, "BACKGROUND TWO": INGEST},
    )
    engine.final("BACKGROUND ONE")
    assert runtime.wait_idle()
    engine.final("BACKGROUND TWO")
    assert runtime.wait_idle()
    assert len(gate.calls) == 2
    _, recent_for_second = gate.calls[1]
    assert "BACKGROUND ONE" in recent_for_second


def test_system_prompt_frames_questions_as_act():
    """Pin the addressing-prompt intent: the fast model decides ACT by asking
    'is this a QUESTION/REQUEST/COMMAND for the assistant?', not the vaguer 'is
    this addressed to me?' that made gemma3:4b drop clear questions as ambient
    (the missed-question bug). Validated live: 10/10 novel questions -> ACT while
    plain statements / talk-to-another / reading-aloud still INGEST. This guards
    against a silent revert to the over-INGEST framing."""
    from core.addressing import _SYSTEM_PROMPT

    up = _SYSTEM_PROMPT.upper()
    assert "QUESTION" in up and "REQUEST" in up and "COMMAND" in up
    assert "ACT" in up and "INGEST" in up
    # Worked examples for BOTH directions keep the small model calibrated.
    assert "-> ACT" in _SYSTEM_PROMPT and "-> INGEST" in _SYSTEM_PROMPT
