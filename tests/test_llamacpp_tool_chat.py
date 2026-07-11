"""Headless tests for the scoped MiniCPM special-token provider boundary."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event

import pytest

from core.llm import (
    LLAMACPP_TOOL_FORMAT_MINICPM5,
    LLMCallCancelled,
    LLMProviderOutputError,
    LlamaCppLLM,
)


_TEMPLATE = """
{% if tools %}<tools></tools>{% endif %}
<function name="x"><param name="query">x</param></function>
{% if message.tool_calls %}{{ tool_call.arguments.items() }}{% endif %}
{% if message.role == "tool" %}<tool_response></tool_response>{% endif %}
{% if enable_thinking %}<think></think>{% endif %}
"""

_PIECES = {
    18: b"<function",
    19: b"</function>",
    20: b"<param",
    21: b"</param>",
    100: b' name="web.search">',
    101: b' name="query">latest news',
    102: b"ordinary answer",
    103: b"late content",
}


@dataclass
class _Plan:
    items: list[tuple[int | None, str | None]]
    cancel: Event | None = None
    cancel_at: int | None = None
    close_error: BaseException | None = None


class _LazyIterator:
    def __init__(self, client, plan: _Plan):
        self.client = client
        self.plan = plan
        self.index = 0
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.plan.items):
            raise StopIteration
        token, finish = self.plan.items[self.index]
        if self.plan.cancel_at == self.index and self.plan.cancel is not None:
            self.plan.cancel.set()
        self.index += 1
        content = None
        if token is not None:
            content = self.client.detokenize([token], prev_tokens=[]).decode()
        return {
            "choices": [
                {"delta": {"content": content}, "finish_reason": finish}
            ]
        }

    def close(self):
        self.closed = True
        # Teardown must still be inside the special-preserving scope.
        self.client.close_special_flags.append(
            bool(self.client.detokenize([18], prev_tokens=[]))
        )
        if self.plan.close_error is not None:
            raise self.plan.close_error


class _LazyClient:
    def __init__(self, *plans: _Plan, template: str = _TEMPLATE):
        self.metadata = {"tokenizer.chat_template": template}
        self.plans = list(plans)
        self.detokenize_special: list[bool] = []
        self.close_special_flags: list[bool] = []
        self.calls: list[dict] = []
        self.iterators: list[_LazyIterator] = []
        self.reset_calls = 0

    def detokenize(self, tokens, prev_tokens=None, special=False):
        del prev_tokens
        self.detokenize_special.append(bool(special))
        token = tokens[0]
        if token in (18, 19, 20, 21) and not special:
            return b""
        return _PIECES[token]

    def create_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        if not kwargs.get("stream"):
            return {
                "choices": [
                    {"message": {"content": "healthy"}, "finish_reason": "stop"}
                ]
            }
        iterator = _LazyIterator(self, self.plans.pop(0))
        self.iterators.append(iterator)
        return iterator

    def reset(self):
        self.reset_calls += 1


def _call_xml() -> _Plan:
    return _Plan(
        [
            (18, None),
            (100, None),
            (20, None),
            (101, None),
            (21, None),
            (19, None),
            (None, "stop"),
        ]
    )


def _llm(client, *, options=None, think=False):
    return LlamaCppLLM(
        "fake.gguf",
        client=client,
        think=think,
        tool_format=LLAMACPP_TOOL_FORMAT_MINICPM5,
        options=options,
    )


def _complete(llm, hook=None, cancel_event=None):
    return llm.complete_minicpm_tool_chat(
        messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "q"}],
        tools=[{"type": "function", "function": {"name": "web.search"}}],
        first_token_hook=hook,
        cancel_event=cancel_event,
    )


def test_special_tokens_survive_only_inside_buffered_tool_completion():
    client = _LazyClient(_call_xml())
    llm = _llm(client)
    hooks: list[str] = []

    result = _complete(llm, lambda: hooks.append("first"))

    assert result.text == (
        '<function name="web.search"><param name="query">latest news'
        "</param></function>"
    )
    assert result.finish_reason == "stop"
    assert hooks == ["first"]
    assert client.detokenize_special and all(client.detokenize_special)
    assert client.close_special_flags == [True]
    assert "detokenize" not in client.__dict__
    # Ordinary detokenization is back to the class method/default-false behavior.
    assert client.detokenize([18]) == b""
    assert client.detokenize_special[-1] is False


def test_tool_completion_restores_an_existing_instance_override_exactly():
    client = _LazyClient(_call_xml())
    class_method = client.detokenize

    def instance_override(tokens, prev_tokens=None, special=False):
        return class_method(tokens, prev_tokens=prev_tokens, special=special)

    client.detokenize = instance_override
    original = client.__dict__["detokenize"]
    llm = _llm(client)

    _complete(llm)
    assert client.__dict__["detokenize"] is original


class _MutateThenRaiseClient(_LazyClient):
    def __init__(self, *plans):
        super().__init__(*plans)
        self.fail_next_detokenize_set = True

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        if (
            name == "detokenize"
            and getattr(self, "fail_next_detokenize_set", False)
        ):
            object.__setattr__(self, "fail_next_detokenize_set", False)
            raise RuntimeError("mutated before synthetic setattr failure")


def test_mutate_then_raise_during_patch_still_restores_inherited_descriptor():
    client = _MutateThenRaiseClient(_call_xml())
    llm = _llm(client)
    with pytest.raises(RuntimeError, match="mutated before"):
        _complete(llm)
    assert "detokenize" not in client.__dict__
    assert client.detokenize([18]) == b""
    assert not llm._lock.locked()


def test_tool_completion_uses_a_protocol_option_allowlist():
    client = _LazyClient(_call_xml())
    llm = _llm(
        client,
        options={
            "temperature": 0.9,
            "top_p": 0.8,
            "max_tokens": 999,
            "stop": "</function>",
            "response_format": {"type": "json_object"},
            "grammar": object(),
            "tool_choice": "none",
            "messages": ["bad"],
            "stream": False,
        },
    )

    _complete(llm)
    call = client.calls[0]
    assert call["temperature"] == 0.0
    assert call["top_p"] == 0.8
    assert call["max_tokens"] == 256
    assert call["stream"] is True
    for forbidden in ("stop", "response_format", "grammar", "tool_choice"):
        assert forbidden not in call


def test_cancel_between_lazy_chunks_restores_detokenizer_resets_and_reuses_context():
    cancel = Event()
    cancelled = _call_xml()
    cancelled.cancel = cancel
    cancelled.cancel_at = 2
    client = _LazyClient(cancelled, _Plan([(102, None), (None, "stop")]))
    llm = _llm(client)
    with pytest.raises(LLMCallCancelled):
        _complete(llm, cancel_event=cancel)

    assert client.reset_calls == 1
    assert "detokenize" not in client.__dict__
    assert llm._context_poisoned is False
    recovery = _complete(llm)
    assert recovery.text == "ordinary answer"
    assert recovery.finish_reason == "stop"


def test_cancel_on_terminal_chunk_wins_natural_completion():
    cancel = Event()
    plan = _call_xml()
    plan.cancel = cancel
    plan.cancel_at = len(plan.items) - 1
    client = _LazyClient(plan)
    llm = _llm(client)
    with pytest.raises(LLMCallCancelled):
        _complete(llm, cancel_event=cancel)
    assert client.reset_calls == 1
    assert "detokenize" not in client.__dict__


@pytest.mark.parametrize(
    "items",
    [
        [(102, "stop")],
        [(102, None), (None, "stop"), (103, None)],
        [(102, None), (None, "stop"), (None, "length")],
    ],
)
def test_ambiguous_or_post_terminal_stream_shape_fails_closed(items):
    client = _LazyClient(_Plan(items))
    llm = _llm(client)
    with pytest.raises(LLMProviderOutputError):
        _complete(llm)
    assert "detokenize" not in client.__dict__
    assert not llm._lock.locked()


class _SyntheticControlFlow(BaseException):
    pass


def test_foreign_close_control_flow_propagates_after_restore_and_unlock():
    plan = _call_xml()
    plan.close_error = _SyntheticControlFlow("close control flow")
    client = _LazyClient(plan)
    llm = _llm(client)
    with pytest.raises(_SyntheticControlFlow):
        _complete(llm)
    assert "detokenize" not in client.__dict__
    assert not llm._lock.locked()


class _NoDetokenizeClient(_LazyClient):
    detokenize = None


def test_missing_detokenizer_and_near_miss_template_fail_before_generation():
    missing = _NoDetokenizeClient(_call_xml())
    llm = _llm(missing)
    with pytest.raises(RuntimeError, match="detokenize seam"):
        _complete(llm)
    assert missing.calls == []
    assert not llm._lock.locked()

    near_miss = _LazyClient(_call_xml(), template="{% if tools %}<tools></tools>{% endif %}")
    llm = _llm(near_miss)
    with pytest.raises(RuntimeError, match="verified embedded template"):
        _complete(llm)
    assert near_miss.calls == []
    assert not llm._lock.locked()


def test_native_tool_completion_refuses_think_mode_even_with_matching_template():
    client = _LazyClient(_call_xml())
    llm = _llm(client, think=True)
    with pytest.raises(RuntimeError, match="think=False"):
        _complete(llm)
    assert client.calls == []
