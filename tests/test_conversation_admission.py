from __future__ import annotations

import math
import time
import pytest

from always_on_agent.conversation_admission import (
    ConversationAdmission,
    ConversationAdmissionConfig,
)
from always_on_agent.events import AgentEvent, EventKind, Mode
from always_on_agent.models import SYNTHETIC_RESUME_TAIL_METADATA_KEY
from core.addressing import ACT, ScriptedAddressingClassifier
from core.engine import FinalTranscript
from core.engines.scripted import ScriptedEngine
from core.llm import EchoLLM
from core.runtime import VoiceRuntime
from core.resume import ResumeConfig


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


@pytest.mark.parametrize("explicit,expected", [(False, []), (True, ["Echo reply."])])
def test_factory_console_preserved_while_missing_speech_classifier_refuses(
    explicit, expected
):
    from core.app import build_runtime

    engine = ScriptedEngine()
    rt = build_runtime(
        {
            "input_gate": {"enabled": True, "unsure_acts": True},
            "warm_on_start": False,
            "memory": {"backend": "session"},
        },
        engine=engine,
        llm=EchoLLM(reply="Echo reply."),
        fast_llm=None,
        router=None,
        start_mode=Mode.ASSISTANT,
        explicit_text_input=explicit,
    )
    rt.start(run_bus=False)
    try:
        engine.final("What is two plus two?")
        assert rt.wait_idle()
        assert engine.spoken == expected
    finally:
        rt.stop()


def test_heard_resumed_question_admits_short_answer_without_resume_authority():
    engine = ScriptedEngine(hold_speech=True)
    classifier = ScriptedAddressingClassifier(default=ACT)
    rt = VoiceRuntime(
        engine,
        EchoLLM(reply="Which city?"),
        start_mode=Mode.ASSISTANT,
        addressing=classifier,
        unsure_acts=False,
        conversation_admission=ConversationAdmissionConfig(),
        resume_config=ResumeConfig(enabled=True, echo_guard_enabled=True),
    )
    rt.start(run_bus=False)
    events = []
    rt.bus.subscribe(events.append)

    def wait_for_speech():
        deadline = time.monotonic() + 2
        while not engine.is_speaking and time.monotonic() < deadline:
            rt.bus.drain()
            time.sleep(0.001)
        assert engine.is_speaking

    try:
        live(engine, "Can you help plan a trip?")
        wait_for_speech()
        engine.finish_speaking()
        assert rt.wait_idle()
        engine.barge_in()
        rt.bus.drain()
        assert rt._resume.preview_resume_prompt("continue") is not None

        live(engine, "continue")
        wait_for_speech()
        resumed = [
            event
            for event in events
            if event.kind == EventKind.STT_FINAL
            and event.payload.get("metadata", {}).get(
                SYNTHETIC_RESUME_TAIL_METADATA_KEY
            )
        ]
        assert len(resumed) == 1
        assert resumed[0].payload["origin"] == "unknown"
        assert resumed[0].payload["owner_verified"] is False
        assert resumed[0].payload["metadata"]["post_barge_response_only"] is True
        assert resumed[0].payload["metadata"]["skip_user_memory"] is True
        assert not rt._conversation_admission.inspect(
            "Paris", input_epoch=rt.supervisor.input_epoch
        ).candidate  # Admission alone cannot stand in for rendered speech.
        engine.finish_speaking()
        assert rt.wait_idle()

        live(engine, "Paris")
        deadline = time.monotonic() + 2
        while len(classifier.calls) < 2 and time.monotonic() < deadline:
            rt.bus.drain()
            time.sleep(0.001)
        assert [call[0] for call in classifier.calls] == [
            "Can you help plan a trip?",
            "Paris",
        ]
        wait_for_speech()
        engine.finish_speaking()
        assert rt.wait_idle()
        assert engine.spoken == ["Which city?"] * 3
    finally:
        rt.stop()


@pytest.mark.parametrize("text", [
    "Run a harmless test command.", "Execute a harmless test command.",
    "Dictate these public notes", "Cauta voce", "Cerceteaza voce", "Scrie text",
    "Assistant, tell me a story", "assistant tell me a story",
    "Hey, could you explain photosynthesis?", "Hello assistant, explain photosynthesis",
    "Hi, assistant: what is two plus two?", "Okay, please execute a harmless command",
])
def test_canonical_requests_and_bounded_vocatives_remain_candidates(text):
    gate = ConversationAdmission(ConversationAdmissionConfig())
    assert gate.inspect(text, input_epoch=1).candidate


@pytest.mark.parametrize("text", [
    '"Run a harmless test command."', "'Execute a harmless command please'",
    'Assistant, "run a harmless command"', 'Hey, "execute a harmless command"',
    "He said run a harmless command", "She asked the assistant to execute a command",
    "Someone said hey could you explain photosynthesis", "The assistant told me a story",
    "Assistant went home yesterday", "Hey, she said execute a command",
])
def test_quoted_or_reported_commands_and_greetings_stay_ambient(text):
    rt, engine, classifier = runtime()
    try:
        live(engine, text)
        assert rt.wait_idle()
        assert classifier.calls == []
        assert engine.spoken == []
        assert rt.supervisor.state.pending_confirmations == {}
    finally:
        rt.stop()


@pytest.mark.parametrize("text", [
    "Run a harmless test command.", "Execute a harmless test command.",
])
def test_canonical_staged_commands_still_need_confirmation_before_provider(text):
    rt, engine, classifier = runtime()
    calls = []
    unsubscribe = rt.supervisor.capabilities.observe_invocations(calls.append)
    try:
        live(engine, text)
        assert rt.wait_idle()
        assert len(rt.supervisor.state.pending_confirmations) == 1
        assert engine.spoken  # existing prompt reached the scripted receipt sink
        assert calls == []  # candidate admission is never action permission
        assert len(classifier.calls) == 1  # semantic admission remains separate
        assert classifier.calls[0][0] == text
    finally:
        unsubscribe()
        rt.stop()


@pytest.mark.parametrize("text", [
    "Assistant, tell me a story", "Hey, could you explain photosynthesis?",
])
def test_vocative_inspection_preserves_immutable_addressing_input(text):
    rt, engine, classifier = runtime()
    try:
        live(engine, text)
        assert rt.wait_idle()
        assert classifier.calls[0][0] == text
        assert engine.spoken == ["Four."]
    finally:
        rt.stop()


@pytest.mark.parametrize("text", [
    "Browse my notes about travel.", "Consult my vault about travel.",
    "Query my notes for travel.", "Go into my notes to find travel.",
    "Kindly browse my notes about travel.", "Computer search my notes for travel.",
    "Jarvis search my notes for travel.", "Asistent search my notes for travel.",
    "Please kindly search my notes for travel.",
])
def test_existing_vault_request_grammar_has_a_cheap_candidate_cue(text):
    from always_on_agent.speech_analyzer import is_vault_lookup_request
    assert is_vault_lookup_request(text)  # audit actual shipped parser forms
    gate = ConversationAdmission(ConversationAdmissionConfig())
    assert gate.inspect(text, input_epoch=1).candidate


@pytest.mark.parametrize("text", [
    '"Browse my notes about travel."', 'Kindly "query my notes for travel"',
    "He said consult my vault about travel", "Computer said the room was quiet",
    "Jarvis went home", "She asked the computer to browse my notes",
])
def test_vault_courtesy_words_do_not_admit_quoted_or_reported_requests(text):
    gate = ConversationAdmission(ConversationAdmissionConfig())
    assert not gate.inspect(text, input_epoch=1).candidate



def test_quoted_short_answer_requires_an_actually_heard_question():
    gate = ConversationAdmission(ConversationAdmissionConfig())
    assert not gate.inspect('"Paris"', input_epoch=1).candidate
    gate.note_admitted(input_epoch=1, input_generation=1)
    gate.note_rendered("Which city?", input_epoch=1, input_generation=1)
    assert gate.inspect('"Paris"', input_epoch=1).reason == "heard_question"
    assert not gate.inspect('"Paris"', input_epoch=2).candidate


def test_exact_heard_answer_ticket_carries_quote_after_arrival_closes_window():
    clock = [10.0]
    gate = ConversationAdmission(ConversationAdmissionConfig(), clock=lambda: clock[0])
    own = _acoustic("quoted-answer")
    gate.note_admitted(input_epoch=1, input_generation=1)
    gate.note_rendered("Which city?", input_epoch=1, input_generation=1)
    assert gate.observe("Paris", input_epoch=1, acoustic=own, partial=True).reason == "heard_question"
    gate.note_arrival(input_epoch=1, input_generation=2)
    clock[0] += 1
    assert gate.observe('"Paris"', input_epoch=1, acoustic=own).reason == "same_utterance"
    assert not gate.observe('"Paris"', input_epoch=1, acoustic=_acoustic("other-answer")).candidate


def test_request_ticket_cannot_turn_idle_quoted_command_into_a_request():
    gate = ConversationAdmission(ConversationAdmissionConfig())
    own = _acoustic("quoted-command")
    assert gate.observe("Run a harmless command", input_epoch=1, acoustic=own, partial=True).candidate
    assert not gate.observe('"Run a harmless command"', input_epoch=1, acoustic=own).candidate
    assert gate._partial_cue is None


@pytest.mark.parametrize("changed", ["epoch", "keys", "expiry"])
def test_quoted_heard_answer_ticket_keeps_original_identity_and_expiry(changed):
    clock = [10.0]
    gate = ConversationAdmission(ConversationAdmissionConfig(), clock=lambda: clock[0])
    own = _acoustic("quoted-bounded-answer")
    gate.note_admitted(input_epoch=1, input_generation=1)
    gate.note_rendered("Which city?", input_epoch=1, input_generation=1)
    gate.observe("Paris", input_epoch=1, acoustic=own, partial=True)
    gate.note_arrival(input_epoch=1, input_generation=2)
    epoch = 2 if changed == "epoch" else 1
    acoustic = _acoustic("other-bounded-answer") if changed == "keys" else own
    if changed == "expiry":
        clock[0] += 30
    assert not gate.observe('"Paris"', input_epoch=epoch, acoustic=acoustic).candidate
