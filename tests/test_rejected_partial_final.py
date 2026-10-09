"""Typed negative finals retire only their exact provisional input lifecycle."""
import pytest

from always_on_agent.continuation import ContinuationConfig
from always_on_agent.conversation_admission import ConversationAdmissionConfig
from core.addressing import ACT, ScriptedAddressingClassifier
from core.engine import FinalTranscript, PartialTranscript
from core.engines.scripted import ScriptedEngine
from core.llm import EchoLLM
from core.runtime import VoiceRuntime
from tests.test_conversation_admission import _acoustic
from tests.test_final_preprocessing_cancel import _FirstPretokenBlockedEcho


@pytest.mark.parametrize("successor", ["Make it shorter", "What is two plus two?"])
def test_rejected_typed_final_retires_reserved_unheard_continuation_without_replay(successor):
    model = _FirstPretokenBlockedEcho()
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine, model, addressing=ScriptedAddressingClassifier(default=ACT),
        unsure_acts=False, continuation_config=ContinuationConfig(enabled=True),
        conversation_admission=ConversationAdmissionConfig(),
    )
    runtime.start(run_bus=True)
    owned = _acoustic("rejected-owned")
    try:
        engine.final_result(FinalTranscript(
            "Explain the moon phases", origin="live_audio", acoustic=_acoustic("old-unheard"), revision=1,
        ))
        assert model.first_started.wait(timeout=1)
        engine.partial_result(PartialTranscript("Run a harmless command", owned, 0))
        generation = runtime._partial_fence_generation
        assert runtime._partial_fence_active
        assert runtime._latest_arrival_continuation() is not None

        engine.final_result(FinalTranscript(
            '"Run a harmless command"', origin="live_audio", acoustic=owned, revision=1,
        ))
        assert not runtime._partial_fence_active
        assert runtime._latest_arrival_continuation() is None
        assert runtime.supervisor.latest_arrival_generation == generation
        model.release_first.set()
        assert runtime.wait_idle(timeout=2)
        assert len(model.prompts) == 1  # no automatic restoration/reissue
        assert engine.spoken == []

        engine.final_result(FinalTranscript(
            successor, origin="live_audio", acoustic=_acoustic("valid-successor"), revision=1,
        ))
        assert runtime.wait_idle(timeout=2)
        assert len(model.prompts) == 2
        assert "moon phases" not in model.prompts[-1].lower()
        assert successor.lower() in model.prompts[-1].lower()
        assert len(engine.spoken) == 1
    finally:
        model.release_first.set()
        runtime.stop()


@pytest.mark.parametrize("foreign", [None, _acoustic("foreign-final")])
def test_unmatched_or_unkeyed_ambient_final_cannot_retire_another_partial(foreign):
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine, EchoLLM(reply="ack"), addressing=ScriptedAddressingClassifier(default=ACT),
        unsure_acts=False, conversation_admission=ConversationAdmissionConfig(),
    )
    runtime.start(run_bus=True)
    owned = _acoustic("still-owned")
    try:
        engine.partial_result(PartialTranscript("Run a harmless command", owned, 0))
        generation = runtime._partial_fence_generation
        keys = runtime._partial_fence_acoustic_keys
        engine.final_result(FinalTranscript(
            '"Run a harmless command"', origin="live_audio", acoustic=foreign,
            revision=1 if foreign is not None else 0,
        ))
        assert runtime._partial_fence_active
        assert runtime._partial_fence_generation == generation
        assert runtime._partial_fence_acoustic_keys == keys
        assert runtime.supervisor.latest_arrival_generation == generation
        engine.final_result(FinalTranscript(
            "Run a harmless command", origin="live_audio", acoustic=owned, revision=1,
        ))
        assert runtime.wait_idle(timeout=2)
        assert not runtime._partial_fence_active
        assert len(runtime.supervisor.state.pending_confirmations) == 1
    finally:
        runtime.stop()


def test_typed_quoted_heard_answer_still_admits_across_partial_arrival():
    engine = ScriptedEngine()
    addressing = ScriptedAddressingClassifier(default=ACT)
    runtime = VoiceRuntime(
        engine, EchoLLM(reply="Which city?"), addressing=addressing,
        unsure_acts=False, conversation_admission=ConversationAdmissionConfig(),
    )
    runtime.start(run_bus=True)
    answer = _acoustic("heard-quoted-answer")
    try:
        engine.final_result(FinalTranscript(
            "Can you help plan a trip?", origin="live_audio", acoustic=_acoustic("question"), revision=1,
        ))
        assert runtime.wait_idle(timeout=2)
        assert engine.spoken == ["Which city?"]
        engine.partial_result(PartialTranscript("Paris", answer, 0))
        assert runtime._partial_fence_active
        engine.final_result(FinalTranscript('"Paris"', origin="live_audio", acoustic=answer, revision=1))
        assert runtime.wait_idle(timeout=2)
        assert len(addressing.calls) == 2
        assert addressing.calls[-1][0] == '"Paris"'
        assert engine.spoken == ["Which city?", "Which city?"]
        assert not runtime._partial_fence_active
    finally:
        runtime.stop()
