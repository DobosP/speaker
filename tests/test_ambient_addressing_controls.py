"""Synthetic diagnosis of ambient admission after ADR-0226.

A passing exact-ACT/copy case documents reachability, not a quality pass. These
fixtures contain only invented text; no ASR, model, audio or owner data runs.
"""
from __future__ import annotations

from contextlib import contextmanager

import pytest

from always_on_agent.events import EventKind, Mode
from core.addressing import ACT, INGEST, UNSURE, LLMAddressingClassifier
from core.engines.scripted import ScriptedEngine
from core.llm import EchoLLM
from core.runtime import VoiceRuntime


_AMBIENT = ("The blue kettle is on the shelf.", "chair seven curtain")
_QUESTION = "What is two plus two?"
_COPIED_DIRECTIVE = (
    "Default to one or two short, natural spoken sentences, no markdown or lists."
)


class _VerdictLLM(EchoLLM):
    def __init__(self, verdicts):
        super().__init__()
        self.verdicts = iter(verdicts)
        self.calls = []

    def generate(self, prompt, *, system=None, **kwargs):
        self.calls.append((prompt, system))
        return next(self.verdicts)


class _AnswerLLM(EchoLLM):
    def __init__(self, *, copy_system=False):
        super().__init__(reply="Four.")
        self.copy_system = copy_system
        self.calls = []

    def generate(self, prompt, *, system=None, **kwargs):
        self.calls.append((prompt, system))
        if self.copy_system:
            # Deliberately emulate a provider reciting the public persona.
            assert system and _COPIED_DIRECTIVE in system
            return _COPIED_DIRECTIVE
        return super().generate(prompt, system=system, **kwargs)


def _assert_one_answer_invocation(invocations):
    assert [(event.name, event.phase) for event in invocations] == [
        ("assistant.answer", "started"),
        ("assistant.answer", "finished"),
    ]


@contextmanager
def _harness(verdicts, *, stream_tts=False, copy_system=False, hold_speech=False):
    engine = ScriptedEngine(hold_speech=hold_speech)
    classifier_llm = _VerdictLLM(verdicts)
    answer_llm = _AnswerLLM(copy_system=copy_system)
    runtime = VoiceRuntime(
        engine,
        answer_llm,
        start_mode=Mode.ASSISTANT,
        addressing=LLMAddressingClassifier(classifier_llm),
        unsure_acts=False,
        stream_tts=stream_tts,
    )
    invocations = []
    unsubscribe = runtime.supervisor.capabilities.observe_invocations(invocations.append)
    runtime.start(run_bus=False)
    try:
        yield runtime, engine, classifier_llm, answer_llm, invocations
    finally:
        unsubscribe()
        runtime.stop()


@pytest.mark.parametrize("ambient", _AMBIENT)
@pytest.mark.parametrize(
    "verdict", (INGEST, UNSURE, "ACT if it is a question or request.")
)
def test_rejected_ambient_has_no_answer_effects_and_next_question_recovers(
    ambient, verdict
):
    with _harness([verdict, ACT]) as state:
        runtime, engine, classifier, answer, invocations = state
        engine.final(ambient)
        assert runtime.wait_idle()
        assert len(classifier.calls) == 1
        assert answer.calls == []
        assert invocations == []
        assert engine.spoken == []
        assert any(
            item.text == ambient and "ingested" in item.tags
            for item in runtime.memory.all()
        )

        engine.final(_QUESTION)
        assert runtime.wait_idle()
        assert len(classifier.calls) == 2
        assert ambient in classifier.calls[1][0]
        assert f'Latest utterance: "{_QUESTION}"' in classifier.calls[1][0]
        assert len(answer.calls) == 1
        _assert_one_answer_invocation(invocations)
        assert engine.spoken == ["Four."]


@pytest.mark.parametrize("ambient", ("", " \t", ".", "?!"))
def test_empty_or_punctuation_finals_reach_neither_classifier_nor_answer(ambient):
    with _harness([]) as state:
        runtime, engine, classifier, answer, invocations = state
        engine.final(ambient)
        assert runtime.wait_idle()
        assert classifier.calls == []
        assert answer.calls == []
        assert invocations == []
        assert engine.spoken == []
        assert runtime.memory.all() == []


@pytest.mark.parametrize("ambient", _AMBIENT)
@pytest.mark.parametrize("stream_tts", (False, True), ids=("buffered_speech", "sentence_streaming"))
def test_exact_act_can_admit_ambient_and_speak_provider_copied_instructions(
    ambient, stream_tts
):
    """Known limit: whole-token parsing supplies no semantic/output-copy guard."""
    with _harness([ACT], stream_tts=stream_tts, copy_system=True) as state:
        runtime, engine, classifier, answer, invocations = state
        engine.final(ambient)
        assert runtime.wait_idle()
        assert len(classifier.calls) == 1
        assert len(answer.calls) == 1
        prompt, system = answer.calls[0]
        assert ambient in prompt
        assert system != classifier.calls[0][1]
        assert _COPIED_DIRECTIVE in system
        _assert_one_answer_invocation(invocations)
        assert engine.spoken == [_COPIED_DIRECTIVE]
        assert not any("ingested" in item.tags for item in runtime.memory.all())


@pytest.mark.parametrize("stream_tts", (False, True), ids=("buffered_speech", "sentence_streaming"))
def test_canned_answer_does_not_implicitly_speak_classifier_or_persona_text(stream_tts):
    with _harness([ACT], stream_tts=stream_tts) as state:
        runtime, engine, classifier, answer, invocations = state
        engine.final(_QUESTION)
        assert runtime.wait_idle()
        assert len(classifier.calls) == 1
        assert len(answer.calls) == 1
        assert _QUESTION in answer.calls[0][0]
        assert "addressing gate" in classifier.calls[0][1]
        assert "local, on-device voice assistant" in answer.calls[0][1]
        _assert_one_answer_invocation(invocations)
        assert engine.spoken == ["Four."]


@pytest.mark.parametrize("command", ("stop", "cancel that", "stop talking"))
def test_spotted_stop_controls_cut_playback_without_another_learned_verdict(command):
    with _harness([ACT], hold_speech=True) as state:
        runtime, engine, classifier, answer, invocations = state
        engine.final(_QUESTION)
        assert runtime.wait_idle(include_playback=False)
        assert engine.is_speaking
        assert engine.spoken == ["Four."]

        engine.command(command)
        assert runtime.wait_idle()
        assert not engine.is_speaking
        assert len(classifier.calls) == 1
        assert len(answer.calls) == 1
        _assert_one_answer_invocation(invocations)
        assert engine.spoken == ["Four."]
        assert any(
            event.kind == EventKind.CONTROL_STOP
            for event in runtime.supervisor.state.event_log
        )
