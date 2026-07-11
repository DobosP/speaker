"""MiniCPM phone voice output: no generated/spoken reasoning preamble."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

import pytest

from core.capabilities import _stream_and_speak
from core.llm import (
    LLAMACPP_PINNED_VERSION,
    LLMCallCancelled,
    LLMProviderOutputError,
    LlamaCppLLM,
    _LlamaCppAbortAPI,
    _ReasoningTagFilter,
    _require_safe_reasoning_output,
    capability_context,
)
from core.metrics import ASR_FINAL, LLM_FIRST_TOKEN, MetricsRecorder, mark_first_token


def _filter_chunks(chunks, *, initial_reasoning_closer: str | None = None):
    parser = _ReasoningTagFilter(
        initial_reasoning_closer=initial_reasoning_closer,
    )
    visible: list[str] = []
    for chunk in chunks:
        visible.extend(parser.feed(chunk))
    visible.extend(parser.finish())
    _require_safe_reasoning_output(parser)
    return "".join(visible), parser


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Before <think>private.\nreasoning!</think> After.", "Before  After."),
        (
            "A<|thought_begin|>hidden<|thought_end|>B",
            "AB",
        ),
        ("Hello </think>world.", "Hello world."),
        ("Ordinary <this is visible> text.", "Ordinary <this is visible> text."),
    ],
)
def test_reasoning_filter_handles_every_two_chunk_split(raw, expected):
    for split in range(len(raw) + 1):
        text, _parser = _filter_chunks((raw[:split], raw[split:]))
        assert text == expected, f"split={split}"


def test_reasoning_filter_handles_character_chunks_case_and_attributes():
    raw = "<THINK mode='deep'>Never speak this.</ThInK>Final answer."
    text, parser = _filter_chunks(raw)

    assert text == "Final answer."
    assert parser.suppressed_blocks == 1
    assert parser.suppressed_chars >= len("Never speak this.")


@pytest.mark.parametrize(
    ("closer", "generated_close"),
    [("</think", "</think>"), ("<|thought_end|>", "<|thought_end|>")],
)
def test_prefilled_think_mode_starts_hidden_until_generated_close(
    closer,
    generated_close,
):
    text, parser = _filter_chunks(
        ("private reasoning", generated_close, "Safe answer."),
        initial_reasoning_closer=closer,
    )

    assert text == "Safe answer."
    assert parser.suppressed_chars == len("private reasoning")


@pytest.mark.parametrize(
    "raw",
    [
        "<think>outer <think>inner</think>still outer</think>Safe answer.",
        (
            "<|thought_begin|>outer <|thought_begin|>inner"
            "<|thought_end|>still outer<|thought_end|>Safe answer."
        ),
    ],
)
def test_nested_reasoning_blocks_never_expose_outer_tail(raw):
    for split in range(len(raw) + 1):
        text, parser = _filter_chunks((raw[:split], raw[split:]))
        assert text == "Safe answer.", f"split={split}"
        assert parser.suppressed_blocks == 2


def test_whitespace_deformed_reasoning_markers_fail_closed_to_safe_answer():
    raw = "< think>SECRET< /think>Safe answer."

    for split in range(len(raw) + 1):
        text, _parser = _filter_chunks((raw[:split], raw[split:]))
        assert text == "Safe answer.", f"split={split}"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("<thinking>ordinary</thinking> safe", "<thinking>ordinary</thinking> safe"),
        ("ordinary <thinkable> text", "ordinary <thinkable> text"),
        ("ordinary </thinkable> text", "ordinary </thinkable> text"),
        (
            "<think>hidden </thinkable> outer tail</think>Safe answer.",
            "Safe answer.",
        ),
    ],
)
def test_think_prefixed_ordinary_words_are_chunk_invariant(raw, expected):
    text, _parser = _filter_chunks(list(raw))
    assert text == expected
    for split in range(len(raw) + 1):
        text, _parser = _filter_chunks((raw[:split], raw[split:]))
        assert text == expected, f"split={split}"


@pytest.mark.parametrize(
    "raw",
    [
        "<think>SECRET</think attr=y>Safe answer.",
        "</think attr=y>Safe answer.",
    ],
)
def test_closing_tag_attributes_are_safe_across_chunk_boundaries(raw):
    text, _parser = _filter_chunks(list(raw))
    assert text == "Safe answer."
    for split in range(len(raw) + 1):
        text, _parser = _filter_chunks((raw[:split], raw[split:]))
        assert text == "Safe answer.", f"split={split}"


@pytest.mark.parametrize(
    "raw",
    [
        "<think PRIVATE</think>Safe answer.",
        "<|thought_begin|PRIVATE<|thought_end|>Safe answer.",
        "<|thought_beginPRIVATE<|thought_end|>Safe answer.",
        "<|thought_begiPRIVATE<|thought_end|>Safe answer.",
        "<|thought_bPRIVATE<|thought_end|>Safe answer.",
        "<|thought_PRIVATE<|thought_end|>Safe answer.",
        "<|thoughPRIVATE<|thought_end|>Safe answer.",
        "< |thought_beginPRIVATE<|thought_end|>Safe answer.",
    ],
)
def test_malformed_opener_recovery_never_exposes_hidden_text(raw):
    text, _parser = _filter_chunks((raw,))
    assert text == "Safe answer."
    text, _parser = _filter_chunks(list(raw))
    assert text == "Safe answer."
    for split in range(len(raw) + 1):
        text, _parser = _filter_chunks((raw[:split], raw[split:]))
        assert text == "Safe answer.", f"split={split}"


def test_malformed_special_closer_never_becomes_visible():
    parser = _ReasoningTagFilter()

    visible = []
    for char in "<|thought_end|PRIVATE":
        visible.extend(parser.feed(char))
    assert visible == []
    parser.finish()
    with pytest.raises(LLMProviderOutputError, match="malformed/unclosed"):
        _require_safe_reasoning_output(parser)


def test_unknown_special_control_token_fails_closed():
    parser = _ReasoningTagFilter()
    visible = []
    for char in "<|eot_id|>PRIVATE":
        visible.extend(parser.feed(char))

    assert visible == []
    parser.finish()
    with pytest.raises(LLMProviderOutputError, match="malformed/unclosed"):
        _require_safe_reasoning_output(parser)


@pytest.mark.parametrize(
    "raw",
    [
        "<thi",
        "<THINK mode='deep'",
        "Answer first. <think>unfinished private text",
        "<|thought_begin|>unfinished",
    ],
)
def test_partial_or_unclosed_reasoning_fails_closed(raw):
    parser = _ReasoningTagFilter()
    parser.feed(raw)
    parser.finish()

    with pytest.raises(LLMProviderOutputError, match="malformed/unclosed"):
        _require_safe_reasoning_output(parser)


def test_closed_reasoning_without_final_answer_is_provider_error():
    parser = _ReasoningTagFilter()
    parser.feed("<think>private</think>   ")
    parser.finish()

    with pytest.raises(LLMProviderOutputError, match="no final answer"):
        _require_safe_reasoning_output(parser)


def test_malformed_reasoning_opener_cannot_grow_stream_buffer():
    parser = _ReasoningTagFilter()
    assert parser.feed("<think " + ("private " * 2000)) == []
    assert len(parser._buffer) <= len("</think")

    visible = parser.feed("</think>Safe answer.")
    visible.extend(parser.finish())
    _require_safe_reasoning_output(parser)
    assert "".join(visible) == "Safe answer."


def test_reasoning_nesting_depth_is_bounded_and_fails_closed():
    parser = _ReasoningTagFilter()
    parser.feed("<think>" * 1000)

    assert len(parser._active_closers) <= parser._MAX_NESTING
    parser.finish()
    with pytest.raises(LLMProviderOutputError, match="malformed/unclosed"):
        _require_safe_reasoning_output(parser)


class _ResponseClient:
    def __init__(
        self,
        response: str,
        *,
        chunks: list[str] | None = None,
        finish_reason: str = "stop",
    ) -> None:
        self.response = response
        self.chunks = chunks
        self.finish_reason = finish_reason

    def create_chat_completion(self, *, stream=False, **_kwargs):
        if stream:
            pieces = self.chunks if self.chunks is not None else [self.response]
            return iter(
                {
                    "choices": [
                        {
                            "delta": {"content": piece},
                            "finish_reason": (
                                self.finish_reason if index == len(pieces) - 1 else None
                            ),
                        }
                    ]
                }
                for index, piece in enumerate(pieces)
            )
        return {
            "choices": [
                {
                    "message": {"content": self.response},
                    "finish_reason": self.finish_reason,
                }
            ]
        }


def test_generate_removes_reasoning_before_return():
    client = _ResponseClient("<think>secret plan</think>The answer is four.")
    llm = LlamaCppLLM("fake.gguf", client=client)

    assert llm.generate("2+2") == "The answer is four."
    assert llm.last_reasoning_chars == len("secret plan")
    assert llm.last_reasoning_blocks == 1
    assert llm.last_finish_reason == "stop"


def test_injected_nonreasoning_think_opt_in_does_not_hide_direct_answer():
    client = _ResponseClient("Direct answer.", chunks=["Direct ", "answer."])
    llm = LlamaCppLLM("fake.gguf", client=client, think=True)

    assert llm.generate("hello") == "Direct answer."
    assert list(llm.stream("hello")) == ["Direct ", "answer."]
    assert llm.last_reasoning_chars == 0


def test_stream_filters_before_sentence_emission():
    client = _ResponseClient(
        "",
        chunks=["<thi", "nk>Private sentence.\n", "Still private!</think>", "Safe answer."],
    )
    llm = LlamaCppLLM("fake.gguf", client=client)
    spoken: list[str] = []

    text, cancelled = _stream_and_speak(llm.stream("hello"), None, spoken.append)

    assert not cancelled
    assert text == "Safe answer."
    assert spoken == ["Safe answer."]


def test_stream_drops_post_reasoning_whitespace_before_first_safe_piece():
    client = _ResponseClient(
        "",
        chunks=["<think>Private.</think>", "\n\n", "Safe answer."],
    )
    llm = LlamaCppLLM("fake.gguf", client=client)

    assert list(llm.stream("hello")) == ["Safe answer."]
    assert llm.last_finish_reason == "stop"


def test_reasoning_only_whitespace_never_stamps_first_token():
    recorder = MetricsRecorder()
    recorder.mark(ASR_FINAL)
    client = _ResponseClient("", chunks=["<think>Private.</think>", "\n\n"])
    llm = LlamaCppLLM("fake.gguf", client=client)

    with pytest.raises(LLMProviderOutputError, match="no final answer"):
        list(mark_first_token(llm.stream("hello"), recorder))

    [record] = recorder.records()
    assert LLM_FIRST_TOKEN not in record.stamps


def test_consumer_close_releases_context_for_same_client_reuse():
    client = _ResponseClient("", chunks=["First ", "answer."])
    llm = LlamaCppLLM("fake.gguf", client=client)
    first = llm.stream("first")

    assert next(first) == "First "
    first.close()

    assert list(llm.stream("second")) == ["First ", "answer."]


def test_stream_unclosed_reasoning_raises_instead_of_silent_success():
    client = _ResponseClient("", chunks=["<think>Private sentence."])
    llm = LlamaCppLLM("fake.gguf", client=client)
    spoken: list[str] = []

    with pytest.raises(LLMProviderOutputError, match="malformed/unclosed"):
        _stream_and_speak(llm.stream("hello"), None, spoken.append)
    assert spoken == []


@dataclass
class _FakeContext:
    callback: object | None = None
    callback_data: object | None = None
    memory: object = object()


class _AbortAPI:
    @staticmethod
    def callback_type(callback):
        return callback

    @staticmethod
    def set_callback(ctx, callback, data):
        ctx.callback = callback
        ctx.callback_data = data

    @staticmethod
    def get_memory(ctx):
        return ctx.memory

    @staticmethod
    def clear_memory(_memory, _data):
        return None

    @classmethod
    def api(cls):
        return _LlamaCppAbortAPI(
            callback_type=cls.callback_type,
            set_callback=cls.set_callback,
            get_memory=cls.get_memory,
            clear_memory=cls.clear_memory,
        )


class _BlockedReasoningIterator:
    def __init__(self, ctx: _FakeContext) -> None:
        self.ctx = ctx
        self.index = 0
        self.blocked = threading.Event()
        self.closed = threading.Event()

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == 0:
            self.index += 1
            return {"choices": [{"delta": {"content": "<think>private."}}]}
        self.blocked.set()
        while True:
            callback = self.ctx.callback
            if callable(callback) and callback(self.ctx.callback_data):
                raise RuntimeError("synthetic native abort")
            self.closed.wait(0.005)

    def close(self):
        self.closed.set()


class _CancelThenHealthyClient:
    def __init__(self) -> None:
        self.ctx = _FakeContext()
        self.iterator = _BlockedReasoningIterator(self.ctx)
        self.calls = 0
        self.reset_calls = 0

    def create_chat_completion(self, *, stream=False, **_kwargs):
        assert stream
        self.calls += 1
        if self.calls == 1:
            return self.iterator
        return iter([{"choices": [{"delta": {"content": "Healthy answer."}}]}])

    def reset(self):
        self.reset_calls += 1


def test_cancel_while_reasoning_emits_nothing_and_next_call_recovers():
    client = _CancelThenHealthyClient()
    llm = LlamaCppLLM("fake.gguf", client=client, abort_api=_AbortAPI.api())
    cancel = threading.Event()
    outcome: dict[str, object] = {}

    def consume():
        token = capability_context.set({"cancel_event": cancel})
        try:
            outcome["pieces"] = list(llm.stream("first"))
        except BaseException as exc:
            outcome["error"] = exc
        finally:
            capability_context.reset(token)

    worker = threading.Thread(target=consume, daemon=True)
    worker.start()
    assert client.iterator.blocked.wait(2.0)
    cancel.set()
    worker.join(2.0)

    assert not worker.is_alive()
    assert isinstance(outcome.get("error"), LLMCallCancelled)
    assert outcome.get("pieces") is None
    assert client.reset_calls == 1
    assert list(llm.stream("healthy")) == ["Healthy answer."]


def _hybrid_template(*, supports_flag: bool = True) -> str:
    flag = "{% if enable_thinking is false %}" if supports_flag else "{% if true %}"
    return f"{flag}<think></think>{{% endif %}}"


def _handler_client(template: str, *, chat_format: str = "chat_template.default"):
    recorded: list[dict] = []

    def base_handler(*_args, **kwargs):
        recorded.append(kwargs)
        return "formatted"

    client = SimpleNamespace(
        metadata={"tokenizer.chat_template": template},
        chat_handler=None,
        chat_format=chat_format,
        _chat_handlers={"chat_template.default": base_handler},
    )
    return client, recorded


@pytest.mark.parametrize(("think", "expected"), [(False, False), (True, True)])
def test_chat_handler_receives_real_boolean_think_mode(think, expected):
    client, recorded = _handler_client(_hybrid_template())
    llm = LlamaCppLLM("fake.gguf", client=object(), think=think)

    closer = llm._configure_chat_template(client)
    assert client.chat_handler(messages=[], llama=object()) == "formatted"
    assert recorded[0]["enable_thinking"] is expected
    assert closer == ("</think" if think else None)


def test_special_reasoning_template_reports_matching_prefill_closer():
    client, _recorded = _handler_client(
        "{% if enable_thinking %}<|thought_begin|>{% endif %}"
    )
    llm = LlamaCppLLM("fake.gguf", client=object(), think=True)

    assert llm._configure_chat_template(client) == "<|thought_end|>"


def test_reasoning_template_without_control_fails_closed():
    client, _recorded = _handler_client(_hybrid_template(supports_flag=False))
    llm = LlamaCppLLM("fake.gguf", client=object())

    with pytest.raises(RuntimeError, match="lacks.*enable_thinking"):
        llm._configure_chat_template(client)


def test_reasoning_template_rejects_implicit_think_mode():
    client, _recorded = _handler_client(_hybrid_template())
    llm = LlamaCppLLM("fake.gguf", client=object(), think=None)

    with pytest.raises(RuntimeError, match="explicit think boolean"):
        llm._configure_chat_template(client)


def test_explicit_nonembedded_chat_format_fails_closed():
    client, _recorded = _handler_client(_hybrid_template(), chat_format="chatml")
    llm = LlamaCppLLM("fake.gguf", client=object())

    with pytest.raises(RuntimeError, match="chat_template.default"):
        llm._configure_chat_template(client)


def test_existing_custom_chat_handler_fails_closed():
    client, _recorded = _handler_client(_hybrid_template())
    client.chat_handler = lambda **_kwargs: None
    llm = LlamaCppLLM("fake.gguf", client=object())

    with pytest.raises(RuntimeError, match="chat_template.default"):
        llm._configure_chat_template(client)


def test_nonreasoning_template_needs_no_wrapper():
    client, recorded = _handler_client("{{ messages }}")
    llm = LlamaCppLLM("fake.gguf", client=object())

    llm._configure_chat_template(client)
    assert client.chat_handler is None
    assert recorded == []


def _verified_fake_module(monkeypatch, llama_type):
    import sys

    module = ModuleType("llama_cpp")
    module.__version__ = LLAMACPP_PINNED_VERSION
    module.Llama = llama_type
    module.ggml_abort_callback = lambda callback: callback
    module.llama_set_abort_callback = lambda *_args: None
    module.llama_get_memory = lambda _ctx: object()
    module.llama_memory_clear = lambda *_args: None
    monkeypatch.setitem(sys.modules, "llama_cpp", module)


def test_model_construction_installs_no_think_handler(monkeypatch):
    recorded: list[dict] = []

    def base_handler(*_args, **kwargs):
        recorded.append(kwargs)
        return "formatted"

    class FakeLlama:
        def __init__(self, **_kwargs):
            self.metadata = {"tokenizer.chat_template": _hybrid_template()}
            self.chat_handler = None
            self.chat_format = "chat_template.default"
            self._chat_handlers = {"chat_template.default": base_handler}

    _verified_fake_module(monkeypatch, FakeLlama)
    llm = LlamaCppLLM("fake.gguf")

    client = llm._ensure()
    assert client.chat_handler(messages=[], llama=object()) == "formatted"
    assert recorded[0]["enable_thinking"] is False


def test_production_nonreasoning_model_think_true_keeps_direct_output(monkeypatch):
    class FakeLlama:
        def __init__(self, **_kwargs):
            self.metadata = {"tokenizer.chat_template": "{{ messages }}"}
            self.chat_handler = None
            self.chat_format = "chat_template.default"
            self._chat_handlers = {}
            self.ctx = object()

        def create_chat_completion(self, *, stream=False, **_kwargs):
            if stream:
                return iter(
                    [
                        {
                            "choices": [
                                {
                                    "delta": {"content": "Direct answer."},
                                    "finish_reason": "stop",
                                }
                            ]
                        }
                    ]
                )
            return {
                "choices": [
                    {
                        "message": {"content": "Direct answer."},
                        "finish_reason": "stop",
                    }
                ]
            }

        def reset(self):
            return None

    _verified_fake_module(monkeypatch, FakeLlama)
    llm = LlamaCppLLM("fake.gguf", think=True)

    assert llm.generate("hello") == "Direct answer."
    assert list(llm.stream("hello")) == ["Direct answer."]
    assert llm._reasoning_prefill_closer is None


def test_model_construction_closes_and_stays_unset_on_unsafe_template(monkeypatch):
    closed = threading.Event()

    class FakeLlama:
        def __init__(self, **_kwargs):
            self.metadata = {
                "tokenizer.chat_template": _hybrid_template(supports_flag=False)
            }
            self.chat_handler = None
            self.chat_format = "chat_template.default"
            self._chat_handlers = {"chat_template.default": lambda **_kwargs: None}

        def close(self):
            closed.set()

    _verified_fake_module(monkeypatch, FakeLlama)
    llm = LlamaCppLLM("fake.gguf")

    with pytest.raises(RuntimeError, match="lacks.*enable_thinking"):
        llm._ensure()
    assert closed.is_set()
    assert llm._client is None


def test_model_construction_closes_on_custom_chat_format(monkeypatch):
    closed = threading.Event()

    class FakeLlama:
        def __init__(self, **_kwargs):
            self.metadata = {"tokenizer.chat_template": _hybrid_template()}
            self.chat_handler = None
            self.chat_format = "chatml"
            self._chat_handlers = {"chat_template.default": lambda **_kwargs: None}

        def close(self):
            closed.set()

    _verified_fake_module(monkeypatch, FakeLlama)
    llm = LlamaCppLLM("fake.gguf")

    with pytest.raises(RuntimeError, match="chat_template.default"):
        llm._ensure()
    assert closed.is_set()
    assert llm._client is None
