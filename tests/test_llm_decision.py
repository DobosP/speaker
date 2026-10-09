"""Native-free decision budgets, isolation, exact output and cancel ownership."""
from __future__ import annotations

import asyncio
import sys
import threading
from types import SimpleNamespace

import pytest

from always_on_agent.models import CLOUD_EGRESS_SCOPE_CONTEXT_KEY, CloudEgressScope
from core.llm import LLMCallCancelled, LlamaCppLLM, OllamaLLM, capability_context, collect_llm_decision
from core.llm_decision import DecisionRequest, decision_request

CHOICES = ("ACT", "INGEST", "UNSURE", "ACTION", "ACTIVE")


class Script:
    def __init__(self, *parts):
        self.parts = parts
        self.closed = False
        self.context = None
        self.request = None

    def stream(self, prompt, *, system=None, images=None, history=None):
        self.context = capability_context.get()
        self.request = decision_request.get()
        try:
            yield from self.parts
        finally:
            self.closed = True


def decide(client, **kwargs):
    return collect_llm_decision(client, "public question", system="choose", choices=CHOICES, **kwargs)


@pytest.mark.parametrize("reply,want", [
    ('"ACT"', "ACT"), ("act.", "ACT"), ("ACTION", "ACTION"),
    ("'INGEST'", "INGEST"), ("UNSURE", "UNSURE"),
    ("ACT if the user asks", None), ("ACT, INGEST, or UNSURE", None),
    ('["ACT"]', None), ('{"label":"ACT"}', None), ('"ACT', None),
    ('"ACT" trailing', None), ("<think>ACT</think>", None), ("", None),
])
def test_only_complete_decision_is_returned(reply, want):
    script = Script(reply)
    assert decide(script) == want
    assert script.closed
    assert decision_request.get() is None


def test_output_is_bounded_and_iterator_closed_without_collecting_tail():
    class TooLong(Script):
        def stream(self, *args, **kwargs):
            try:
                yield "x" * 257
                pytest.fail("overflow must stop before reading another chunk")
            finally:
                self.closed = True
    script = TooLong()
    assert decide(script) is None
    assert script.closed


@pytest.mark.parametrize("kwargs", [
    {"max_tokens": 17}, {"max_tokens": 0}, {"max_tokens": True},
    {"timeout_sec": 3.01}, {"timeout_sec": 0}, {"timeout_sec": True},
    {"timeout_sec": float("nan")}, {"timeout_sec": float("inf")},
])
def test_invalid_budgets_refuse_before_model(kwargs):
    script = Script("ACT")
    with pytest.raises(ValueError):
        decide(script, **kwargs)
    assert script.context is None


@pytest.mark.parametrize("choices", [("ACT", "ACT"), (), ["ACT"], ("a",), ("A\nB",), ("A" * 33,)])
def test_choice_contract_is_bounded(choices):
    with pytest.raises(ValueError):
        DecisionRequest(choices)


@pytest.mark.parametrize("scope", [CloudEgressScope.LOCAL_ONLY, CloudEgressScope.CURRENT_TURN_ONLY])
def test_local_scope_only_narrows_and_context_restores(scope):
    original = {CLOUD_EGRESS_SCOPE_CONTEXT_KEY: scope, "owner_verified": False, "private": "fixture"}
    token = capability_context.set(original)
    try:
        script = Script("ACT")
        assert decide(script) == "ACT"
        assert script.context[CLOUD_EGRESS_SCOPE_CONTEXT_KEY] is CloudEgressScope.LOCAL_ONLY
        assert script.context["owner_verified"] is False
        assert capability_context.get() is original
        assert "cancel_event" not in original
    finally:
        capability_context.reset(token)


def test_external_cancellation_beats_completed_label():
    event = threading.Event()
    class CancelAfter(Script):
        def stream(self, *args, **kwargs):
            yield "ACT"
            event.set()
    with pytest.raises(LLMCallCancelled):
        decide(CancelAfter(), cancel_event=event)
    assert decision_request.get() is None


def test_cooperative_expiration_refuses_even_complete_label(monkeypatch):
    from core import llm_decision
    now = [0.0]
    monkeypatch.setattr(llm_decision.time, "monotonic", lambda: now[0])
    class Expired(Script):
        def stream(self, *args, **kwargs):
            yield "ACT"
            now[0] = 4.0
    assert decide(Expired()) is None


class SyncClient:
    def __init__(self):
        self.calls = []
    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["stream"]:
            return iter([{"message": {"content": '"ACT"'}}])
        return {"message": {"content": "answer"}}


def test_ollama_request_overrides_do_not_mutate_answers_or_model_options():
    native = SyncClient()
    options = {"num_ctx": 8192, "num_predict": 512, "num_gpu": 999, "temperature": 0.6}
    client = OllamaLLM("same-model", options=options, keep_alive=-1, think=True, client=native)
    assert decide(client) == "ACT"
    decision = native.calls[0]
    assert decision["options"] == {**options, "num_predict": 16, "temperature": 0.0}
    assert decision["format"] == {"type": "string", "enum": list(CHOICES)}
    assert decision["think"] is False
    assert decision["keep_alive"] == -1
    assert client.generate("answer question") == "answer"
    answer = native.calls[1]
    assert answer["options"] == options
    assert "format" not in answer
    assert answer["think"] is True
    assert options["num_predict"] == 512


def test_existing_smaller_token_cap_is_never_raised():
    native = SyncClient()
    client = OllamaLLM(options={"num_predict": 4}, client=native)
    assert decide(client) == "ACT"
    assert native.calls[0]["options"]["num_predict"] == 4


def test_llamacpp_captures_decision_grammar_and_restores_answer_budget(monkeypatch):
    grammar = object()
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(
        LlamaGrammar=SimpleNamespace(from_string=lambda value, verbose: grammar)))
    class Native:
        def __init__(self):
            self.calls = []
        def create_chat_completion(self, **kwargs):
            self.calls.append(kwargs)
            return iter([{"choices": [{"delta": {"content": "ACT"}}]}])
    native = Native()
    client = LlamaCppLLM("fixture.gguf", options={"num_predict": 512, "temperature": 0.7}, client=native)
    assert decide(client) == "ACT"
    assert native.calls[0]["max_tokens"] == 16
    assert native.calls[0]["grammar"] is grammar
    assert native.calls[0]["temperature"] == 0.0
    assert list(client.stream("answer")) == ["ACT"]
    assert native.calls[1]["max_tokens"] == 512
    assert "grammar" not in native.calls[1]


def test_ollama_deadline_cancels_blocked_pre_first_token_provider():
    entered = threading.Event()
    cancelled = threading.Event()
    closed = threading.Event()
    class Native:
        async def chat(self, **kwargs):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
        async def close(self):
            closed.set()
    client = OllamaLLM(async_client_factory=lambda **kwargs: Native())
    result = []
    thread = threading.Thread(target=lambda: result.append(decide(client, timeout_sec=0.1)))
    thread.start()
    assert entered.wait(1)
    thread.join(1)
    assert not thread.is_alive()
    assert result == [None]
    assert cancelled.wait(1)
    assert closed.wait(1)


def test_direct_cloud_client_is_refused_before_transport():
    from core.llm import OpenAICompatLLM
    class NoCall:
        def __getattr__(self, name):
            pytest.fail("direct compatible-API decision must not touch transport")
    client = OpenAICompatLLM("public-cloud", base_url="https://example.invalid/v1", client=NoCall())
    assert decide(client) is None


def test_factory_hybrid_decision_never_starts_cloud():
    from core.llm import HedgeLLM
    class NoCloud(Script):
        def stream(self, *args, **kwargs):
            pytest.fail("decision must not launch a cloud source")
    local = Script("ACT")
    client = HedgeLLM(local=local, cloud=NoCloud(), hedge_delay_ms=0)
    token = capability_context.set({CLOUD_EGRESS_SCOPE_CONTEXT_KEY: CloudEgressScope.CURRENT_TURN_ONLY})
    try:
        assert decide(client) == "ACT"
    finally:
        capability_context.reset(token)


def test_factory_hybrid_preserves_native_decision_options_in_source_owner():
    from core.llm import HedgeLLM
    native = SyncClient()
    local = OllamaLLM(options={"num_predict": 512}, client=native)
    hybrid = HedgeLLM(local=local, cloud=Script("must not start"), hedge_delay_ms=0)
    assert decide(hybrid) == "ACT"
    assert native.calls[0]["options"]["num_predict"] == 16
    assert native.calls[0]["format"]["enum"] == list(CHOICES)


def test_pre_cancelled_direct_compatible_client_preserves_cancellation():
    from core.llm import OpenAICompatLLM
    event = threading.Event()
    event.set()
    client = OpenAICompatLLM("public-cloud", client=object())
    with pytest.raises(LLMCallCancelled):
        decide(client, cancel_event=event)


def test_empty_piece_storm_does_not_grow_the_collected_parts():
    import tracemalloc
    class EmptyStorm:
        def stream(self, *args, **kwargs):
            for _ in range(100_000):
                yield ""
            yield "ACT"
    tracemalloc.start()
    try:
        assert decide(EmptyStorm()) == "ACT"
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 128 * 1024
