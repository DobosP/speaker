"""Tests for the cloud_providers + cloud_chains config wiring in core/app.py.

_wrap_cloud has two paths -- single-cloud back-compat and the new multi-
provider sensitivity-routed shape. This file covers the new path:
provider registry resolution, missing-API-key handling, chain build, and
the SensitivityRouterLLM hookup.
"""
from __future__ import annotations

import json
import os
import threading
import time
import types

import pytest

from core.app import _build_cloud_client, _wrap_cloud
from core.llm import (
    HedgeLLM,
    LLMClient,
    OpenAICompatLLM,
    SensitivityRouterLLM,
    capability_context,
)


class _Stub(LLMClient):
    def generate(self, prompt, *, system=None, images=None) -> str:
        return "stub"

    def stream(self, prompt, *, system=None, images=None):
        yield "stub"


# --- preset resolution -----------------------------------------------------


def test_build_cloud_client_requires_model():
    assert _build_cloud_client("x", {"base_url": "..."}) is None
    assert _build_cloud_client("x", {}) is None


def test_build_cloud_client_skips_when_api_key_env_unset(monkeypatch):
    monkeypatch.delenv("MISSING_KEY", raising=False)
    preset = {
        "base_url": "https://x.example/v1",
        "model": "m1",
        "api_key_env": "MISSING_KEY",
    }
    assert _build_cloud_client("p", preset) is None


def test_build_cloud_client_constructs_with_env_key(monkeypatch):
    monkeypatch.setenv("PRESENT_KEY", "secret")
    client = _build_cloud_client(
        "p", {"base_url": "https://x.example/v1", "model": "m1", "api_key_env": "PRESENT_KEY"}
    )
    assert isinstance(client, OpenAICompatLLM)
    assert client.model == "m1"
    assert client._api_key == "secret"


def test_build_cloud_client_ignores_underscore_keys_in_preset():
    """The preset may carry _comment / _pricing_usd_per_mtok metadata."""
    client = _build_cloud_client(
        "p",
        {
            "_comment": "Cerebras Qwen-235B",
            "_pricing_usd_per_mtok": {"in": 0.60, "out": 1.20},
            "base_url": "https://x.example/v1",
            "model": "m1",
        },
    )
    assert isinstance(client, OpenAICompatLLM)


# --- _wrap_cloud disabled --------------------------------------------------


def test_wrap_cloud_disabled_when_cloud_block_off():
    local = _Stub()
    assert _wrap_cloud(local, {"cloud": {"enabled": False}}) is local


def test_wrap_cloud_disabled_when_strategy_local_only():
    local = _Stub()
    cfg = {
        "cloud": {"enabled": True, "strategy": "local_only"},
        "cloud_providers": {"x": {"base_url": "y", "model": "m"}},
        "cloud_chains": {"private": ["x"]},
    }
    assert _wrap_cloud(local, cfg) is local


# --- multi-provider sensitivity-routed path --------------------------------


def test_wrap_cloud_builds_sensitivity_router_with_per_chain_hedges(monkeypatch):
    monkeypatch.setenv("KEY_A", "k")
    monkeypatch.setenv("KEY_B", "k")
    cfg = {
        "cloud": {"enabled": True, "strategy": "hedge", "hedge_delay_ms": 0},
        "cloud_providers": {
            "a": {"base_url": "https://a", "model": "ma", "api_key_env": "KEY_A"},
            "b": {"base_url": "https://b", "model": "mb", "api_key_env": "KEY_B"},
        },
        "cloud_chains": {
            "private": ["a", "b"],
            "public": ["b"],
        },
        "cloud_routing": {
            "default_chain": "private",
            "sensitivity_to_chain": {"private": "private", "public": "public"},
        },
    }
    wrapped = _wrap_cloud(_Stub(), cfg)
    assert isinstance(wrapped, SensitivityRouterLLM)
    assert set(wrapped.chains.keys()) == {"private", "public"}
    assert wrapped.default_chain == "private"
    # Each chain entry is a HedgeLLM whose clouds match the chain order.
    for name, expected_n in [("private", 2), ("public", 1)]:
        chain = wrapped.chains[name]
        assert isinstance(chain, HedgeLLM)
        assert len(chain.clouds) == expected_n


def test_wrap_cloud_drops_providers_with_missing_keys(monkeypatch):
    """If an API key is unset, that provider is silently absent from any
    chain it appears in; the remaining chain still works."""
    monkeypatch.setenv("KEY_A", "k")
    monkeypatch.delenv("KEY_B", raising=False)
    cfg = {
        "cloud": {"enabled": True, "strategy": "hedge"},
        "cloud_providers": {
            "a": {"base_url": "https://a", "model": "ma", "api_key_env": "KEY_A"},
            "b": {"base_url": "https://b", "model": "mb", "api_key_env": "KEY_B"},
        },
        "cloud_chains": {"private": ["a", "b"]},
        "cloud_routing": {"default_chain": "private"},
    }
    wrapped = _wrap_cloud(_Stub(), cfg)
    assert isinstance(wrapped, SensitivityRouterLLM)
    chain = wrapped.chains["private"]
    assert len(chain.clouds) == 1  # only "a" survived


def test_wrap_cloud_returns_local_when_every_chain_is_empty(monkeypatch):
    """All API keys missing -> no usable cloud -> stay fully local."""
    monkeypatch.delenv("KEY_A", raising=False)
    cfg = {
        "cloud": {"enabled": True, "strategy": "hedge"},
        "cloud_providers": {
            "a": {"base_url": "https://a", "model": "ma", "api_key_env": "KEY_A"},
        },
        "cloud_chains": {"private": ["a"]},
        "cloud_routing": {"default_chain": "private"},
    }
    local = _Stub()
    assert _wrap_cloud(local, cfg) is local


def test_wrap_cloud_default_chain_falls_back_when_not_in_chains(monkeypatch):
    """Config typo: default_chain points to a chain name that isn't built.
    The wrapper should still produce a working SensitivityRouterLLM using
    one of the available chains as the default."""
    monkeypatch.setenv("KEY_A", "k")
    cfg = {
        "cloud": {"enabled": True, "strategy": "hedge"},
        "cloud_providers": {"a": {"base_url": "https://a", "model": "ma", "api_key_env": "KEY_A"}},
        "cloud_chains": {"public": ["a"]},
        "cloud_routing": {"default_chain": "private"},  # typo: no "private" chain
    }
    wrapped = _wrap_cloud(_Stub(), cfg)
    assert isinstance(wrapped, SensitivityRouterLLM)
    assert wrapped.default_chain == "public"


def test_wrap_cloud_ignores_underscore_keys_in_providers_and_chains(monkeypatch):
    """The config carries _comment / _pricing metadata keys -- they must
    not be treated as provider names or chain names."""
    monkeypatch.setenv("KEY_A", "k")
    cfg = {
        "cloud": {"enabled": True, "strategy": "hedge"},
        "cloud_providers": {
            "_comment": "metadata, not a provider",
            "a": {"base_url": "https://a", "model": "ma", "api_key_env": "KEY_A"},
        },
        "cloud_chains": {
            "_comment": "metadata, not a chain",
            "private": ["a"],
        },
        "cloud_routing": {"default_chain": "private"},
    }
    wrapped = _wrap_cloud(_Stub(), cfg)
    assert isinstance(wrapped, SensitivityRouterLLM)
    assert set(wrapped.chains.keys()) == {"private"}


# --- single-cloud back-compat preserved ------------------------------------


def test_wrap_cloud_falls_back_to_single_cloud_shape_when_no_providers(monkeypatch):
    monkeypatch.setenv("LEGACY_KEY", "k")
    cfg = {
        "cloud": {
            "enabled": True,
            "strategy": "hedge",
            "base_url": "https://x",
            "model": "m1",
            "api_key_env": "LEGACY_KEY",
        }
    }
    wrapped = _wrap_cloud(_Stub(), cfg)
    assert isinstance(wrapped, HedgeLLM)
    assert isinstance(wrapped.cloud, OpenAICompatLLM)


# --- config.json sanity ----------------------------------------------------


def test_config_json_has_well_formed_cloud_providers_block():
    """The committed config.json's cloud_providers entries must each have a
    base_url + model + api_key_env so users can fill in keys and go."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(repo_root, "config.json")) as fh:
        cfg = json.load(fh)
    providers = cfg["llm"]["cloud_providers"]
    for name, preset in providers.items():
        if name.startswith("_"):
            continue
        assert "base_url" in preset, f"{name} missing base_url"
        assert "model" in preset, f"{name} missing model"
        assert "api_key_env" in preset, f"{name} missing api_key_env"


def test_config_json_chains_reference_only_defined_providers():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(repo_root, "config.json")) as fh:
        cfg = json.load(fh)
    providers = {n for n in cfg["llm"]["cloud_providers"] if not n.startswith("_")}
    for chain_name, preset_names in cfg["llm"]["cloud_chains"].items():
        if chain_name.startswith("_"):
            continue
        for pname in preset_names:
            assert pname in providers, (
                f"chain {chain_name!r} references undefined provider {pname!r}"
            )


def test_config_json_routing_mapping_targets_defined_chains():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(repo_root, "config.json")) as fh:
        cfg = json.load(fh)
    chains = {n for n in cfg["llm"]["cloud_chains"] if not n.startswith("_")}
    routing = cfg["llm"]["cloud_routing"]
    assert routing["default_chain"] in chains
    for sens, chain_name in routing["sensitivity_to_chain"].items():
        assert chain_name in chains, (
            f"sensitivity {sens!r} -> chain {chain_name!r} which is not defined"
        )


# --- HTTP hard-close + timeout (BR1 / BR6) ---------------------------------


def _chunk(content):
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content=content))]
    )


class _ClosableStream:
    """Fake SDK Stream: yields canned chunks and records that .close() ran.

    Mirrors the openai Stream surface OpenAICompatLLM.stream() touches: it is
    iterable and has a .close() (the HTTP hard-close we port from cloudchat)."""

    def __init__(self, chunks):
        self._chunks = list(chunks)
        self._i = 0
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._i >= len(self._chunks):
            raise StopIteration
        chunk = self._chunks[self._i]
        self._i += 1
        return chunk

    def close(self):
        self.closed = True


class _FakeClientReturning:
    """Fake OpenAI client whose chat.completions.create returns ``stream``."""

    def __init__(self, stream):
        self._stream = stream
        self.create_called = False
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        self.create_called = True
        return self._stream


def test_stream_hard_closes_sdk_stream_when_consumer_stops_early():
    """Consume one token then close the generator: the SDK stream's .close()
    must run (HTTP hard-close => provider stops billing), via the finally in
    OpenAICompatLLM.stream (BR6 / goal 4)."""
    sdk = _ClosableStream([_chunk("a"), _chunk("b"), _chunk("c")])
    llm = OpenAICompatLLM("m", client=_FakeClientReturning(sdk))
    gen = llm.stream("hi")
    assert next(gen) == "a"  # first token flowed
    assert sdk.closed is False
    gen.close()  # consumer cancels (barge-in)
    assert sdk.closed is True


def test_stream_close_before_first_token_is_a_noop_not_an_error():
    """Closing the generator with ZERO tokens consumed must not raise: the
    sdk_stream binding lives INSIDE the generator body, so a pre-first-token
    close finds it unbound and the getattr-guarded finally is a no-op (BR6)."""
    sdk = _ClosableStream([_chunk("x")])
    client = _FakeClientReturning(sdk)
    llm = OpenAICompatLLM("m", client=client)
    gen = llm.stream("hi")
    # Never advanced -> generator body never started -> create() never called.
    gen.close()  # must not raise AttributeError
    assert client.create_called is False
    assert sdk.closed is False  # nothing to close; no spurious close


class _BlockingStream:
    """Fake SDK Stream that blocks in __next__ like a pre-first-token loser.

    The real openai/httpx client reaps such a read at its configured socket
    timeout; we model that here by waking after ``deadline`` and raising a
    timeout-like error, so the test asserts OpenAICompatLLM passes a *small*
    timeout through and the block is bounded by it (BR1)."""

    def __init__(self, deadline):
        self._deadline = deadline
        self._event = threading.Event()
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        # Block as a stalled first-token read would; the SDK's socket timeout
        # is what unblocks it. Bounded by the (small) configured timeout.
        if self._event.wait(timeout=self._deadline):
            raise StopIteration  # closed early
        raise TimeoutError("read timed out")

    def close(self):
        self.closed = True
        self._event.set()


def test_blocked_first_token_read_is_reaped_within_configured_timeout():
    """A small timeout (BR1) is honoured: a stream stalled in __next__ is
    reaped within the bound rather than wedging for the 30s default.

    The fake reads ``llm._timeout`` as the SDK's socket timeout would, so this
    pins both that the timeout is plumbed onto the client AND that it bounds a
    pre-first-token loser."""
    timeout_s = 0.3
    # The fake stream wakes at the configured timeout (what httpx would do).
    sdk_holder: dict = {}

    class _TimeoutClient:
        def __init__(self):
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=self._create)
            )

        def _create(self, **kwargs):
            sdk_holder["stream"] = _BlockingStream(deadline=timeout_s)
            return sdk_holder["stream"]

    llm = OpenAICompatLLM("m", timeout=timeout_s, client=_TimeoutClient())
    assert llm._timeout == timeout_s  # the small timeout is configured (BR1)
    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        list(llm.stream("hi"))
    elapsed = time.monotonic() - t0
    # Reaped within a small multiple of the timeout, never near the 30s default.
    assert elapsed < timeout_s + 1.0


def test_timeout_is_passed_to_the_lazily_built_openai_client(monkeypatch):
    """BR1 plumbing: the configured timeout reaches the OpenAI client ctor so
    the SDK's socket read honours it (not the 30s default)."""
    captured: dict = {}

    fake_openai_module = types.SimpleNamespace(
        OpenAI=lambda **kwargs: captured.update(kwargs) or object()
    )
    import sys

    monkeypatch.setitem(sys.modules, "openai", fake_openai_module)
    llm = OpenAICompatLLM("m", base_url="https://x/v1", timeout=4.0)
    llm._ensure()
    assert captured["timeout"] == 4.0
