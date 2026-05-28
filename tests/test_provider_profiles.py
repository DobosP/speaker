"""Tests for ProviderProfile + OpenAICompatLLM per-vendor quirks.

Each cloud sharing the ``/v1/chat/completions`` shape still has a small
quirk that, ignored, causes silent failures:

- Moonshot Kimi rejects custom ``temperature``/``top_p``/``n``
- Cerebras wants non-standard params via ``extra_body=``
- Cerebras free tier caps ``max_tokens=8192``
- DeepSeek V4-Pro streams CoT in ``delta.reasoning_content``
- Groq gpt-oss-120b streams reasoning in ``delta.reasoning`` AND rejects ``n != 1``

These are exactly the bugs the May-2026 audit found in `config.json`. Each
test pins one quirk so adding a new provider can't quietly regress them.
"""
from __future__ import annotations

import types
from typing import Iterator

from core.llm import (
    PROVIDER_PROFILES,
    OpenAICompatLLM,
    ProviderProfile,
)


# --- registry --------------------------------------------------------------


def test_registered_profiles_cover_every_shipped_provider():
    """Every preset config.json ships ought to have a matching profile.

    Doesn't test the config itself (that's test_cloud_providers.py); just
    asserts the named profiles exist."""
    expected = {
        "openai_compat", "cerebras", "groq",
        "deepseek", "deepseek_reasoning", "moonshot",
    }
    assert expected.issubset(PROVIDER_PROFILES.keys())


def test_unknown_profile_name_falls_back_to_openai_compat():
    llm = OpenAICompatLLM("m", profile="not_a_real_profile")
    assert llm.profile is PROVIDER_PROFILES["openai_compat"]


def test_explicit_profile_instance_is_used_as_is():
    custom = ProviderProfile(name="x", max_tokens_cap=42)
    llm = OpenAICompatLLM("m", profile=custom)
    assert llm.profile is custom


def test_none_profile_defaults_to_openai_compat():
    llm = OpenAICompatLLM("m", profile=None)
    assert llm.profile.name == "openai_compat"


# --- _create_kwargs quirk handling ----------------------------------------


def test_moonshot_strips_forbidden_params():
    """Moonshot Kimi rejects custom temperature/top_p/n; OpenAICompatLLM must
    strip them BEFORE the request lands at the wire."""
    llm = OpenAICompatLLM(
        "kimi-k2.6", profile="moonshot",
        options={"temperature": 0.7, "top_p": 0.9, "n": 2, "max_tokens": 100},
    )
    kw = llm._create_kwargs("hi", system=None, images=None, stream=True)
    assert "temperature" not in kw
    assert "top_p" not in kw
    assert "n" not in kw
    # Non-forbidden keys still pass through.
    assert kw["max_tokens"] == 100


def test_groq_strips_n_but_keeps_others():
    """Groq accepts only n=1; strip it. Other params (temperature, max_tokens)
    pass through normally."""
    llm = OpenAICompatLLM(
        "openai/gpt-oss-120b", profile="groq",
        options={"n": 4, "temperature": 0.3, "max_tokens": 200},
    )
    kw = llm._create_kwargs("hi", system=None, images=None, stream=True)
    assert "n" not in kw
    assert kw["temperature"] == 0.3
    assert kw["max_tokens"] == 200


def test_cerebras_routes_unknown_params_to_extra_body():
    """Cerebras returns 400 if non-OpenAI params land in top-level kwargs;
    they must be routed through extra_body=."""
    llm = OpenAICompatLLM(
        "gpt-oss-120b", profile="cerebras",
        options={"clear_thinking": True, "reasoning_effort": "high", "temperature": 0.5},
    )
    kw = llm._create_kwargs("hi", system=None, images=None, stream=True)
    assert kw.get("extra_body") == {"clear_thinking": True, "reasoning_effort": "high"}
    assert "clear_thinking" not in kw
    assert "reasoning_effort" not in kw
    # Standard OpenAI params (temperature) stay top-level.
    assert kw["temperature"] == 0.5


def test_cerebras_caps_max_tokens_at_8192():
    """Cerebras free tier rejects max_tokens > 8192; enforce locally."""
    llm = OpenAICompatLLM(
        "gpt-oss-120b", profile="cerebras",
        options={"max_tokens": 99999},
    )
    kw = llm._create_kwargs("hi", system=None, images=None, stream=True)
    assert kw["max_tokens"] == 8192


def test_cerebras_default_max_tokens_when_caller_omits():
    """If the caller doesn't set max_tokens at all and the profile has a cap,
    the cap is applied as the floor so requests don't accidentally accept the
    server default (which may differ across plans)."""
    llm = OpenAICompatLLM("gpt-oss-120b", profile="cerebras")
    kw = llm._create_kwargs("hi", system=None, images=None, stream=True)
    assert kw["max_tokens"] == 8192


def test_extra_body_merges_with_caller_supplied():
    """If the caller already passed extra_body=, the profile's extra_body
    keys must merge, not clobber."""
    llm = OpenAICompatLLM(
        "gpt-oss-120b", profile="cerebras",
        options={"clear_thinking": True, "extra_body": {"foo": "bar"}},
    )
    kw = llm._create_kwargs("hi", system=None, images=None, stream=True)
    assert kw["extra_body"] == {"foo": "bar", "clear_thinking": True}


# --- stream() reasoning-channel handling ----------------------------------


def _delta(*, content=None, reasoning=None, reasoning_content=None):
    """Build a fake stream chunk with optional reasoning-channel fields."""
    fields: dict = {}
    if content is not None:
        fields["content"] = content
    if reasoning is not None:
        fields["reasoning"] = reasoning
    if reasoning_content is not None:
        fields["reasoning_content"] = reasoning_content
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(delta=types.SimpleNamespace(**fields))]
    )


class _FakeOpenAI:
    """Minimal OpenAI client stub: stream() yields the canned chunks."""

    def __init__(self, chunks):
        self._chunks = chunks
        self.last_kwargs: dict = {}
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        self.last_kwargs = kwargs
        if kwargs.get("stream"):
            return iter(self._chunks)
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=""))]
        )


def test_deepseek_reasoning_content_consumed_but_not_yielded_by_default():
    """DeepSeek V4-Pro streams `delta.reasoning_content` BEFORE delta.content.
    The voice assistant must not speak the CoT (suppress) but must count it
    for the run-summary observability metric."""
    chunks = [
        _delta(reasoning_content="Let me think... "),
        _delta(reasoning_content="The answer is four. "),
        _delta(content="The answer is four."),
    ]
    fake = _FakeOpenAI(chunks)
    llm = OpenAICompatLLM(
        "deepseek-v4-pro", profile="deepseek_reasoning", client=fake,
    )
    out = list(llm.stream("hi"))
    # Only content yielded, not reasoning.
    assert "".join(out) == "The answer is four."
    # Reasoning was observed and counted.
    assert llm.last_reasoning_chars == len("Let me think... ") + len("The answer is four. ")


def test_groq_gpt_oss_reasoning_field_consumed_and_counted():
    """Groq's gpt-oss-120b puts reasoning in delta.reasoning (not
    delta.reasoning_content -- different from DeepSeek). Profile field
    selects which one to consume."""
    chunks = [
        _delta(reasoning="thinking..."),
        _delta(content="4"),
    ]
    fake = _FakeOpenAI(chunks)
    llm = OpenAICompatLLM(
        "openai/gpt-oss-120b", profile="groq", client=fake,
    )
    out = list(llm.stream("hi"))
    assert "".join(out) == "4"
    assert llm.last_reasoning_chars == len("thinking...")


def test_reasoning_chars_resets_per_stream_call():
    """The metric must reset between turns -- a stale count from a previous
    call would mis-attribute reasoning load to the wrong turn."""
    chunks_1 = [_delta(reasoning_content="cot1"), _delta(content="A")]
    chunks_2 = [_delta(reasoning_content="x"), _delta(content="B")]
    fake = _FakeOpenAI(chunks_1)
    llm = OpenAICompatLLM("m", profile="deepseek_reasoning", client=fake)
    list(llm.stream("q1"))
    assert llm.last_reasoning_chars == 4
    fake._chunks = chunks_2
    list(llm.stream("q2"))
    assert llm.last_reasoning_chars == 1


def test_profile_without_reasoning_field_does_not_track():
    """Plain chat profiles (e.g. deepseek for V4-Flash) leave the counter at 0."""
    chunks = [_delta(content="hello")]
    fake = _FakeOpenAI(chunks)
    llm = OpenAICompatLLM("deepseek-v4-flash", profile="deepseek", client=fake)
    out = list(llm.stream("q"))
    assert "".join(out) == "hello"
    assert llm.last_reasoning_chars == 0


def test_suppress_reasoning_false_yields_reasoning_too():
    """For debugging the CoT, the profile field is overridable. When
    suppress=False, reasoning_content surfaces as a yielded token too."""
    chunks = [_delta(reasoning_content="cot"), _delta(content=" answer")]
    fake = _FakeOpenAI(chunks)
    custom = ProviderProfile(
        name="dbg", reasoning_field="reasoning_content",
        suppress_reasoning_in_stream=False,
    )
    llm = OpenAICompatLLM("m", profile=custom, client=fake)
    out = list(llm.stream("q"))
    assert "".join(out) == "cot answer"


def test_empty_delta_chunks_are_skipped_gracefully():
    """Some providers send keepalive / non-content chunks; OpenAICompatLLM
    must skip them rather than crashing."""
    chunks = [
        types.SimpleNamespace(choices=[]),                                  # no choices
        _delta(),                                                            # delta with no fields
        _delta(content=""),                                                  # empty content
        _delta(content="hello"),
    ]
    fake = _FakeOpenAI(chunks)
    llm = OpenAICompatLLM("m", profile="openai_compat", client=fake)
    out = list(llm.stream("q"))
    assert "".join(out) == "hello"
