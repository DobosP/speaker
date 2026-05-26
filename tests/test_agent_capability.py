"""Tests for the action-brain capability provider (callback bridge)."""
from utils.agent_brain import AgentEvent
from utils.agent_capability import create_agent_provider
from utils.capabilities import CapabilityRequest


class FakeBrain:
    def __init__(self, events, raise_exc=None):
        self._events = events
        self._raise = raise_exc
        self.closed = False

    def stream_run(self, instruction, should_cancel=None, on_confirm=None):
        if self._raise:
            raise self._raise
        try:
            for event in self._events:
                yield event
        finally:
            self.closed = True


def _req(instruction):
    return CapabilityRequest(name="agent.execute", payload={"instruction": instruction})


def test_provider_speaks_each_phrase_and_marks_streamed():
    spoken = []
    brain = FakeBrain([AgentEvent("speak", text="Hello."), AgentEvent("result", text="done")])
    resp = create_agent_provider(brain, speak_cb=spoken.append)(_req("do it"))
    assert spoken == ["Hello.", "done"]
    assert resp.ok and resp.data["streamed"] and resp.data["spoke"]


def test_provider_empty_instruction_fails():
    resp = create_agent_provider(FakeBrain([]), speak_cb=lambda t: None)(_req("   "))
    assert not resp.ok and "empty" in resp.error


def test_provider_speaks_fallback_on_error_event():
    spoken = []
    resp = create_agent_provider(
        FakeBrain([AgentEvent("error", text="boom")]), speak_cb=spoken.append
    )(_req("do it"))
    assert spoken and "couldn't complete" in spoken[0].lower()
    assert resp.data["error"] == "boom"


def test_provider_handles_brain_exception_without_double_speak():
    spoken = []
    brain = FakeBrain([], raise_exc=RuntimeError("not installed"))
    resp = create_agent_provider(brain, speak_cb=spoken.append)(_req("do it"))
    # streamed marker is still set so main does not speak a second time
    assert resp.ok and resp.data["streamed"]
    assert spoken and "couldn't complete" in spoken[0].lower()
    assert "not installed" in resp.data["error"]


def test_provider_cancel_breaks_and_closes_stream():
    spoken = []
    brain = FakeBrain([AgentEvent("speak", text="one"), AgentEvent("speak", text="two")])
    cancel_state = {"v": False}

    def speak(text):
        spoken.append(text)
        cancel_state["v"] = True  # request cancel after the first phrase

    resp = create_agent_provider(
        brain, speak_cb=speak, cancel_cb=lambda: cancel_state["v"]
    )(_req("do it"))
    assert spoken == ["one"]  # stopped before the second phrase
    assert brain.closed  # generator .close() ran in finally
    assert resp.data["streamed"]
