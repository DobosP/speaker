from __future__ import annotations

import math
import time
import pytest

from always_on_agent.conversation_admission import (
    ConversationAdmission,
    ConversationAdmissionConfig,
)
from always_on_agent.events import AgentEvent, Mode
from core.addressing import ACT, ScriptedAddressingClassifier
from core.engine import FinalTranscript
from core.engines.scripted import ScriptedEngine
from core.llm import EchoLLM
from core.runtime import VoiceRuntime


@pytest.mark.parametrize(
    "text",
    [
        "What is the capital of France?",
        "What's the time?",
        "Who wrote Frankenstein?",
        "Which country is Paris in?",
        "How tall is Everest?",
        "Can you help me?",
        "Tell me a story about a dragon",
        "Explain photosynthesis",
        "I need help with my code",
        "Is Python faster than Rust?",
        "Are elephants mammals?",
        "Does Paris have a metro?",
        "I'd like a short explanation",
        "Convert ten miles to kilometers",
        "Reset the timer",
        "Weather in Bucharest",
        "A recipe for pancakes please",
        "Please summarize this",
        "What about Italy?",
        "Make it shorter",
        "Go on",
    ],
)
def test_implicit_questions_and_requests_remain_candidates(text):
    gate = ConversationAdmission(ConversationAdmissionConfig())
    assert gate.inspect(text, input_epoch=1).candidate


@pytest.mark.parametrize(
    "text",
    [
        "N Sanos you know",
        "ca chap for to",
        "I just kind cast brand",
        "um hmm",
        "I think I left the stove on",
        "No I told you yesterday",
        "She asked what is the capital of France",
        "What a lovely day",
        "Paris",
    ],
)
def test_ambient_text_cannot_open_or_extend_conversation(text):
    gate = ConversationAdmission(ConversationAdmissionConfig())
    assert not gate.inspect(text, input_epoch=1).candidate
    assert not gate.inspect("Rome", input_epoch=1).candidate


def test_only_current_heard_question_opens_bounded_answer_window():
    clock = [10.0]
    gate = ConversationAdmission(
        ConversationAdmissionConfig(answer_window_sec=8), clock=lambda: clock[0]
    )
    gate.note_admitted(input_epoch=1, input_generation=3)
    assert not gate.inspect("Paris", input_epoch=1).candidate
    gate.note_rendered("Which city?", input_epoch=1, input_generation=2)
    assert not gate.inspect("Paris", input_epoch=1).candidate
    gate.note_rendered("Which city?", input_epoch=1, input_generation=3)
    assert gate.inspect("Paris", input_epoch=1).reason == "heard_question"
    assert not gate.inspect("Paris", input_epoch=2).candidate
    assert not gate.inspect("um hmm", input_epoch=1).candidate
    clock[0] = 18.0
    assert not gate.inspect("Paris", input_epoch=1).candidate
    gate.note_rendered("Which city?", input_epoch=1, input_generation=3)
    gate.invalidate()
    assert not gate.inspect("Paris", input_epoch=1).candidate


def test_name_requires_whole_prefix_not_ambient_mention():
    gate = ConversationAdmission(ConversationAdmissionConfig(assistant_name="Nova"))
    assert gate.inspect("Hey Nova, what do you think?", input_epoch=1).candidate
    assert not gate.inspect("I saw a supernova yesterday", input_epoch=1).candidate
    assert not gate.inspect("I told Nova yesterday", input_epoch=1).candidate


@pytest.mark.parametrize("value", [True, -1, 31, math.nan, math.inf, "8"])
def test_answer_window_rejects_unbounded_configuration(value):
    with pytest.raises(ValueError):
        ConversationAdmissionConfig(answer_window_sec=value)


def runtime(reply="Four.", hold=False):
    engine = ScriptedEngine(hold_speech=hold)
    classifier = ScriptedAddressingClassifier(default=ACT)
    rt = VoiceRuntime(
        engine,
        EchoLLM(reply=reply),
        start_mode=Mode.ASSISTANT,
        addressing=classifier,
        unsure_acts=False,
        conversation_admission=ConversationAdmissionConfig(),
    )
    rt.start(run_bus=False)
    return rt, engine, classifier


def live(engine, text):
    engine.final_result(FinalTranscript(text, origin="live_audio"))


def test_wrong_act_model_cannot_admit_idle_room_noise_or_call_tools():
    rt, engine, classifier = runtime()
    calls = []
    unsubscribe = rt.supervisor.capabilities.observe_invocations(calls.append)
    try:
        live(engine, "N Sanos you know")
        assert rt.wait_idle()
        assert classifier.calls == []
        assert calls == []
        assert engine.spoken == []
        assert any(
            x.text == "N Sanos you know" and "ingested" in x.tags
            for x in rt.memory.all()
        )
        live(engine, "What is two plus two?")
        assert rt.wait_idle()
        assert len(classifier.calls) == 1
        assert engine.spoken == ["Four."]
    finally:
        unsubscribe()
        rt.stop()


def test_ambient_partial_and_final_do_not_retire_playing_reply():
    rt, engine, classifier = runtime(hold=True)
    try:
        live(engine, "Tell me a story")
        deadline = time.monotonic() + 2
        while not engine.is_speaking and time.monotonic() < deadline:
            rt.bus.drain()
            time.sleep(0.001)
        assert engine.is_speaking
        generation = rt.supervisor.latest_arrival_generation
        engine.partial("random room noise")
        live(engine, "random room noise")
        rt.bus.drain()
        assert rt.supervisor.latest_arrival_generation == generation
        assert engine.is_speaking
        assert len(classifier.calls) == 1
    finally:
        rt.stop()


def test_played_question_allows_answer_and_stop_closes_window():
    rt, engine, classifier = runtime(reply="Which city?")
    try:
        live(engine, "Can you help me plan a trip?")
        assert rt.wait_idle()
        live(engine, "Paris")
        assert rt.wait_idle()
        assert len(classifier.calls) == 2
        engine.command("stop")
        rt.bus.drain()
        live(engine, "Rome")
        assert rt.wait_idle()
        assert len(classifier.calls) == 2
    finally:
        rt.stop()


def test_typed_console_input_and_dictation_keep_existing_semantics():
    rt, engine, classifier = runtime()
    try:
        engine.final("hello there")
        assert rt.wait_idle()
        assert len(classifier.calls) == 1
        rt.bus.publish(AgentEvent.mode(Mode.DICTATION))
        rt.bus.drain()
        assert rt._has_conversation_cue("ordinary dictation statement")
    finally:
        rt.stop()


def test_missing_addressing_model_cannot_become_implicit_act():
    from core.addressing import UnavailableAddressingClassifier, INGEST

    assert UnavailableAddressingClassifier().classify("What time is it?") == INGEST


def test_ambient_flood_leaves_control_capacity_and_never_faults():
    from always_on_agent.event_bus import EventBus
    from always_on_agent.events import EventKind

    bus = EventBus(max_events=8, control_reserve=2)
    seen = []
    bus.subscribe(seen.append)
    for _ in range(100):
        bus.publish(AgentEvent(EventKind.AMBIENT_TRANSCRIPT, {"text": "room fragment"}))
    bus.publish(AgentEvent.stop())
    bus.drain()
    assert any(event.kind == EventKind.CONTROL_STOP for event in seen)
    assert not any(event.kind == EventKind.MAILBOX_FAULT for event in seen)


def test_failure_is_not_semantic_ambiguity_even_when_unsure_acts():
    from core.addressing import UnavailableAddressingClassifier

    engine = ScriptedEngine()
    rt = VoiceRuntime(
        engine,
        EchoLLM(reply="must not speak"),
        addressing=UnavailableAddressingClassifier(),
        unsure_acts=True,
        conversation_admission=ConversationAdmissionConfig(),
    )
    rt.start(run_bus=False)
    try:
        live(engine, "What is two plus two?")
        assert rt.wait_idle()
        assert engine.spoken == []
    finally:
        rt.stop()


def _acoustic(name):
    from always_on_agent.acoustic import AcousticLineage, AcousticSource, AcousticSpan

    return AcousticLineage.single(
        AcousticSpan(
            stream_id="admission-test",
            utterance_id=name,
            source=AcousticSource.SCRIPTED,
        )
    )


def test_partial_cue_survives_window_expiry_only_for_exact_utterance():
    clock = [10.0]
    gate = ConversationAdmission(ConversationAdmissionConfig(), clock=lambda: clock[0])
    gate.note_admitted(input_epoch=1, input_generation=1)
    gate.note_rendered("Which city?", input_epoch=1, input_generation=1)
    own = _acoustic("answer")
    foreign = _acoustic("noise")
    assert gate.observe("Paris", input_epoch=1, acoustic=own, partial=True).candidate
    gate.note_arrival(input_epoch=1, input_generation=2)
    clock[0] = 20
    assert not gate.observe("garbled words", input_epoch=1, acoustic=foreign).candidate
    assert gate.observe("Paris", input_epoch=1, acoustic=own, partial=True).candidate
    assert gate.observe("Paris", input_epoch=1, acoustic=own).candidate
    assert not gate.observe("Paris", input_epoch=1, acoustic=own).candidate


def test_old_reply_cannot_open_window_after_new_arrival_or_abort():
    gate = ConversationAdmission(ConversationAdmissionConfig())
    gate.note_admitted(input_epoch=1, input_generation=1)
    gate.note_arrival(input_epoch=1, input_generation=2)
    gate.note_rendered("Which city?", input_epoch=1, input_generation=1)
    assert not gate.inspect("Paris", input_epoch=1).candidate
    gate.observe(
        "What time", input_epoch=1, acoustic=_acoustic("question"), partial=True
    )
    gate.abandon(_acoustic("question"))
    assert not gate.observe(
        "fragment", input_epoch=1, acoustic=_acoustic("question")
    ).candidate


def test_auxiliary_question_cannot_open_answer_window():
    rt, engine, classifier = runtime(reply="Four.")
    try:
        live(engine, "What is two plus two?")
        assert rt.wait_idle()
        rt._speak_local_intent("Which city?")
        assert rt.wait_idle()
        live(engine, "Paris")
        assert rt.wait_idle()
        assert len(classifier.calls) == 1
    finally:
        rt.stop()


def test_carried_partials_cannot_renew_deadline_or_override_hard_refusal():
    clock = [0.0]
    gate = ConversationAdmission(ConversationAdmissionConfig(), clock=lambda: clock[0])
    own = _acoustic("request")
    assert gate.observe(
        "What time", input_epoch=1, acoustic=own, partial=True
    ).candidate
    clock[0] = 29
    assert gate.observe("fragment", input_epoch=1, acoustic=own, partial=True).candidate
    clock[0] = 30
    assert not gate.observe("fragment", input_epoch=1, acoustic=own).candidate
    assert gate.observe(
        "What time", input_epoch=1, acoustic=own, partial=True
    ).candidate
    assert not gate.observe("x" * 8193, input_epoch=1, acoustic=own).candidate
    assert gate.observe(
        "What time", input_epoch=1, acoustic=own, partial=True
    ).candidate
    assert not gate.observe("um hmm", input_epoch=1, acoustic=own).candidate


def test_exact_restoration_preserves_question_followup_without_new_authority():
    gate = ConversationAdmission(ConversationAdmissionConfig())
    gate.note_admitted(input_epoch=1, input_generation=1)
    gate.note_arrival(input_epoch=1, input_generation=2)
    gate.transfer_admitted(input_epoch=1, previous_generation=99, input_generation=2)
    gate.note_rendered("Which city?", input_epoch=1, input_generation=2)
    assert not gate.inspect("Paris", input_epoch=1).candidate
    gate.transfer_admitted(input_epoch=1, previous_generation=1, input_generation=2)
    gate.note_rendered("Which city?", input_epoch=1, input_generation=2)
    assert gate.inspect("Paris", input_epoch=1).candidate


def test_capture_cue_never_resolves_device_tools_or_calls_unknown_classifier():
    rt, engine, _ = runtime()

    class ForbiddenDispatcher:
        def match(self, text):
            raise AssertionError("device DB/manager accessed on capture thread")

    class ForbiddenContinuation:
        realtime_safe = False

        def classify(self, *args):
            raise AssertionError("blocking classifier accessed on capture thread")

    try:
        rt._device_tool_dispatcher = ForbiddenDispatcher()
        rt.supervisor._continuation = ForbiddenContinuation()
        assert rt._has_conversation_cue("Cancel my next reminder")
        assert rt._has_conversation_cue("Launch calculator")
        assert not rt._has_conversation_cue("random room fragments")
    finally:
        rt._device_tool_dispatcher = None
        rt.stop()
