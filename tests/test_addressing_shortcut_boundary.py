"""Paired synthetic tests for complete-turn addressing shortcuts.

Prepared during the resource pause; qualification is separate. All fixtures are
invented text. This covers a lexical bypass, not model or acoustic quality.
"""
from __future__ import annotations

import pytest

from always_on_agent.events import Mode
from core.addressing import ACT, INGEST, UNSURE, LLMAddressingClassifier
from core.engines.scripted import ScriptedEngine
from core.llm import EchoLLM
from core.runtime import VoiceRuntime


_GENUINE_OPEN_REQUESTS = (
    "Remember for this conversation that the codename is Orion.",
    "Look up Pipecat using your tools.",
    "Search for current Pipecat releases with your tool.",
    "Research Pipecat and LiveKit using your tools.",
    "Please search for Pipecat.",
    "Please research Pipecat and LiveKit.",
    "Say exactly three short sentences: Blue. White. Red.",
    "Repeat your previous answer exactly, with no omissions.",
    "Repeat your previous answer exactly. Include all the details.",
    "Please research Pipecat and LiveKit in detail.",
    "Research the phrase 'she read aloud' using your tools.",
    "Say exactly one word: 'hello'.",
)
_NARRATED_REQUESTS = (
    "Repeat your previous answer exactly, she read aloud.",
    "Remember for this conversation is the phrase printed in the guide.",
    "Please research shows the result was negative.",
    "Look up Pipecat using your tools, she read aloud.",
    "Search for current Pipecat releases with your tool, the manual says.",
    "Research Pipecat and LiveKit using your tools. He read from the script.",
    "Please search for Pipecat, the manual says.",
    "Please research Pipecat and LiveKit, she read aloud.",
    "Remember for this conversation that the codename is Orion, she read aloud.",
    "Say exactly three short sentences is the line in the guide.",
    "Say exactly three short sentences: Blue. White. Red. She read from the script.",
    "Repeat your previous answer exactly. She read aloud.",
    '"Repeat your previous answer exactly."',
    "'Repeat your previous answer exactly.'",
    "The script said: repeat your previous answer exactly.",
)


class _VerdictLLM:
    def __init__(self, verdict):
        self.verdict = verdict
        self.calls = []

    def generate(self, prompt, *, system=None, images=None):
        self.calls.append((prompt, system))
        return self.verdict

    def stream(self, prompt, *, system=None, images=None):
        yield self.generate(prompt, system=system, images=images)


def _assert_full_turn_classified(llm, text):
    assert len(llm.calls) == 1
    prompt, system = llm.calls[0]
    assert prompt == f'Latest utterance: "{text}"'
    assert "addressing gate" in system


@pytest.mark.parametrize(
    "text",
    (
        "Repeat your previous answer exactly.",
        "repeat your previous answer exactly",
        "  REPEAT   YOUR PREVIOUS ANSWER EXACTLY! \n",
        "Repeat your previous answer exactly?",
    ),
)
def test_complete_fixed_repeat_request_keeps_shortcut(text):
    llm = _VerdictLLM(INGEST)
    assert LLMAddressingClassifier(llm).classify(text) == ACT
    assert llm.calls == []


@pytest.mark.parametrize("text", _GENUINE_OPEN_REQUESTS)
def test_genuine_open_request_keeps_full_text_and_honors_learned_act(text):
    llm = _VerdictLLM(ACT)
    assert LLMAddressingClassifier(llm).classify(text) == ACT
    _assert_full_turn_classified(llm, text)


@pytest.mark.parametrize("text", _NARRATED_REQUESTS)
@pytest.mark.parametrize(
    ("reply", "expected"),
    ((INGEST, INGEST), (UNSURE, UNSURE), ("ACT because it is a request.", UNSURE)),
)
def test_narrated_or_quoted_prefix_uses_learned_verdict_for_complete_turn(
    text, reply, expected
):
    llm = _VerdictLLM(reply)
    assert LLMAddressingClassifier(llm).classify(text) == expected
    _assert_full_turn_classified(llm, text)


class _RecordingAnswerLLM(EchoLLM):
    def __init__(self):
        super().__init__(reply="This answer must not run.")
        self.calls = []

    def generate(self, prompt, **kwargs):
        self.calls.append(prompt)
        return super().generate(prompt, **kwargs)


@pytest.mark.parametrize("text", _NARRATED_REQUESTS[:3])
@pytest.mark.parametrize("reply", (INGEST, UNSURE, "ACT because it is a request."))
def test_reported_prefix_counterexamples_cannot_answer_under_conservative_policy(
    text, reply
):
    engine = ScriptedEngine()
    gate_llm = _VerdictLLM(reply)
    answer_llm = _RecordingAnswerLLM()
    runtime = VoiceRuntime(
        engine,
        answer_llm,
        start_mode=Mode.ASSISTANT,
        addressing=LLMAddressingClassifier(gate_llm),
        unsure_acts=False,
    )
    invocations = []
    unsubscribe = runtime.supervisor.capabilities.observe_invocations(invocations.append)
    runtime.start(run_bus=False)
    try:
        engine.final(text)
        assert runtime.wait_idle()
        _assert_full_turn_classified(gate_llm, text)
        assert answer_llm.calls == []
        assert invocations == []
        assert engine.spoken == []
        assert any(
            item.text == text and "ingested" in item.tags
            for item in runtime.memory.all()
        )
    finally:
        unsubscribe()
        runtime.stop()
