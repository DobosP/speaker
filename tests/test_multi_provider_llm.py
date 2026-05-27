"""Optional cloud LLM tier: OpenAI-compatible client + local/cloud hedge.

All fakes -- no `openai` package, no network. Covers message formatting, stream
delta extraction, the staggered-hedge / fallback strategies, and the rule that
any cloud failure falls back to local (fully-local stays the safety net).
"""
from __future__ import annotations

import time
import types
from typing import Iterator

from core.app import _wrap_cloud
from core.llm import HedgeLLM, OpenAICompatLLM


# --- OpenAICompatLLM ---------------------------------------------------------


def _resp(content):
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))]
    )


def _chunk(content):
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content=content))]
    )


class FakeOpenAI:
    def __init__(self, reply="hi", stream_tokens=("he", "llo", None)):
        self.reply = reply
        self.stream_tokens = stream_tokens
        self.calls: list[dict] = []
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return iter([_chunk(t) for t in self.stream_tokens])
        return _resp(self.reply)


def test_openai_generate_formats_messages_and_returns_content():
    fake = FakeOpenAI(reply="answer")
    llm = OpenAICompatLLM("llama-3.3-70b", client=fake, options={"temperature": 0.2})
    assert llm.generate("question", system="be brief") == "answer"
    call = fake.calls[0]
    assert call["model"] == "llama-3.3-70b"
    assert call["temperature"] == 0.2
    assert call["messages"][0] == {"role": "system", "content": "be brief"}
    assert call["messages"][-1] == {"role": "user", "content": "question"}


def test_openai_stream_yields_nonempty_deltas():
    fake = FakeOpenAI(stream_tokens=("he", "llo", None, ""))
    out = list(OpenAICompatLLM("m", client=fake).stream("hi"))
    assert out == ["he", "llo"]  # None and "" filtered
    assert fake.calls[0]["stream"] is True


def test_openai_multimodal_content_for_images():
    fake = FakeOpenAI()
    OpenAICompatLLM("m", client=fake).generate("describe", images=["/tmp/a.png", b"raw"])
    content = fake.calls[0]["messages"][-1]["content"]
    assert content[0] == {"type": "text", "text": "describe"}
    assert content[1]["image_url"]["url"] == "/tmp/a.png"
    assert content[2]["image_url"]["url"].startswith("data:image/png;base64,")


def test_openai_resolves_api_key_from_env(monkeypatch):
    monkeypatch.setenv("MY_KEY", "secret-123")
    llm = OpenAICompatLLM("m", api_key_env="MY_KEY")
    assert llm._api_key == "secret-123"


# --- HedgeLLM ----------------------------------------------------------------


class FakeStreamLLM:
    def __init__(self, tokens, *, first_token_delay=0.0, error=None):
        self.tokens = list(tokens)
        self.first_token_delay = first_token_delay
        self.error = error
        self.started = False

    def stream(self, prompt, *, system=None, images=None) -> Iterator[str]:
        self.started = True
        if self.error:
            raise RuntimeError(self.error)
        for i, tok in enumerate(self.tokens):
            if i == 0 and self.first_token_delay:
                time.sleep(self.first_token_delay)
            yield tok

    def generate(self, prompt, *, system=None, images=None) -> str:
        return "".join(self.tokens)


def test_hedge_local_fast_wins_and_cloud_never_starts():
    local = FakeStreamLLM(["a", "b"])
    cloud = FakeStreamLLM(["X"])
    h = HedgeLLM(local=local, cloud=cloud, strategy="hedge", hedge_delay_ms=300)
    assert "".join(h.stream("q")) == "ab"
    assert cloud.started is False  # local answered within the hedge delay


def test_hedge_local_slow_lets_cloud_win():
    local = FakeStreamLLM(["a"], first_token_delay=0.6)
    cloud = FakeStreamLLM(["X", "Y"])
    h = HedgeLLM(local=local, cloud=cloud, strategy="hedge", hedge_delay_ms=40)
    assert "".join(h.stream("q")) == "XY"
    assert cloud.started is True


def test_hedge_falls_back_to_local_when_cloud_errors():
    local = FakeStreamLLM(["a", "b"], first_token_delay=0.2)
    cloud = FakeStreamLLM([], error="rate limited")
    h = HedgeLLM(local=local, cloud=cloud, strategy="hedge", hedge_delay_ms=20)
    assert "".join(h.stream("q")) == "ab"


def test_fallback_strategy_prefers_cloud_when_healthy():
    local = FakeStreamLLM(["L"])
    cloud = FakeStreamLLM(["C1", "C2"])
    h = HedgeLLM(local=local, cloud=cloud, strategy="fallback", ttft_deadline_ms=500)
    assert "".join(h.stream("q")) == "C1C2"
    assert local.started is False


def test_fallback_strategy_uses_local_when_cloud_errors():
    local = FakeStreamLLM(["L1", "L2"])
    cloud = FakeStreamLLM([], error="boom")
    h = HedgeLLM(local=local, cloud=cloud, strategy="fallback", ttft_deadline_ms=500)
    assert "".join(h.stream("q")) == "L1L2"


def test_hedge_with_no_cloud_is_passthrough():
    local = FakeStreamLLM(["only", "local"])
    h = HedgeLLM(local=local, cloud=None)
    assert "".join(h.stream("q")) == "onlylocal"


# --- wiring ------------------------------------------------------------------


def test_wrap_cloud_disabled_returns_local_unchanged():
    local = FakeStreamLLM(["x"])
    assert _wrap_cloud(local, {"cloud": {"enabled": False}}) is local
    assert _wrap_cloud(local, {}) is local  # no cloud block at all


def test_wrap_cloud_enabled_builds_hedge():
    local = FakeStreamLLM(["x"])
    wrapped = _wrap_cloud(
        local,
        {"cloud": {"enabled": True, "model": "llama-3.3-70b", "strategy": "hedge"}},
    )
    assert isinstance(wrapped, HedgeLLM)
    assert wrapped.local is local
    assert isinstance(wrapped.cloud, OpenAICompatLLM)
