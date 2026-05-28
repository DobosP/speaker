"""Tests for HedgeLLM chain support (multi-cloud failover).

The single-cloud back-compat is already covered by test_multi_provider_llm.py.
This file exercises the new code path: ``cloud`` is a list of clients, the
hedger races them in sequence with local always as the final safety net.
"""
from __future__ import annotations

import time
from typing import Iterator

from core.llm import HedgeLLM, SensitivityRouterLLM, capability_context
from core.routing import ChainSelector


class FakeStreamLLM:
    """Reuse the same fake used in test_multi_provider_llm.py."""

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


# --- HedgeLLM chain --------------------------------------------------------


def test_chain_cloud_attribute_returns_first_cloud_for_backcompat():
    """Existing tests read ``hedge.cloud`` -- it must still return the first
    cloud when a list is provided."""
    c1 = FakeStreamLLM(["a"])
    c2 = FakeStreamLLM(["b"])
    h = HedgeLLM(local=FakeStreamLLM([]), cloud=[c1, c2])
    assert h.cloud is c1
    assert h.clouds == [c1, c2]


def test_chain_empty_list_is_same_as_no_cloud():
    local = FakeStreamLLM(["L"])
    h = HedgeLLM(local=local, cloud=[])
    assert h.cloud is None
    assert "".join(h.stream("q")) == "L"


def test_fallback_chain_advances_past_failing_cloud():
    """cloud_0 errors -> cloud_1 takes over within the same fallback strategy."""
    local = FakeStreamLLM(["L"], first_token_delay=2.0)  # slow local; shouldn't matter
    c0 = FakeStreamLLM([], error="rate limited")
    c1 = FakeStreamLLM(["X", "Y"])
    h = HedgeLLM(local=local, cloud=[c0, c1], strategy="fallback", ttft_deadline_ms=500)
    out = "".join(h.stream("q"))
    assert out == "XY"
    assert c0.started is True
    assert c1.started is True


def test_fallback_chain_falls_back_to_local_when_all_clouds_fail():
    local = FakeStreamLLM(["L1", "L2"])
    c0 = FakeStreamLLM([], error="boom")
    c1 = FakeStreamLLM([], error="kapow")
    h = HedgeLLM(local=local, cloud=[c0, c1], strategy="fallback", ttft_deadline_ms=200)
    out = "".join(h.stream("q"))
    assert out == "L1L2"
    assert c0.started and c1.started


def test_hedge_chain_advances_past_failing_cloud_keeping_local_race():
    """In hedge strategy: cloud_0 errors -> cloud_1 is brought up; whichever
    of {local, cloud_1} produces first wins."""
    # Local takes 600ms; cloud_0 errors fast; cloud_1 produces immediately.
    local = FakeStreamLLM(["L"], first_token_delay=0.6)
    c0 = FakeStreamLLM([], error="500")
    c1 = FakeStreamLLM(["X"])
    h = HedgeLLM(local=local, cloud=[c0, c1], strategy="hedge", hedge_delay_ms=20)
    out = "".join(h.stream("q"))
    assert out == "X"
    assert c1.started is True


def test_hedge_chain_local_wins_when_all_clouds_error():
    local = FakeStreamLLM(["L1", "L2"])
    c0 = FakeStreamLLM([], error="x")
    c1 = FakeStreamLLM([], error="y")
    h = HedgeLLM(local=local, cloud=[c0, c1], strategy="hedge", hedge_delay_ms=10)
    out = "".join(h.stream("q"))
    assert out == "L1L2"


def test_chain_returns_empty_when_local_and_every_cloud_yield_nothing():
    """No source produces a token -> empty stream (not a hang)."""
    local = FakeStreamLLM([])
    c0 = FakeStreamLLM([])
    h = HedgeLLM(local=local, cloud=[c0], strategy="hedge", hedge_delay_ms=10)
    assert "".join(h.stream("q")) == ""


def test_chain_normalizes_none_cloud_to_empty_list():
    """Passing cloud=None must keep working (back-compat)."""
    local = FakeStreamLLM(["L"])
    h = HedgeLLM(local=local, cloud=None)
    assert h.clouds == []
    assert h.cloud is None
    assert "".join(h.stream("q")) == "L"


# --- SensitivityRouterLLM --------------------------------------------------


def test_router_picks_chain_by_context_sensitivity():
    private_llm = FakeStreamLLM(["P", "P"])
    code_llm = FakeStreamLLM(["C", "C"])
    public_llm = FakeStreamLLM(["U", "U"])
    selector = ChainSelector(
        {"private": "private", "code": "code", "public": "public"},
        default_chain="private",
    )
    r = SensitivityRouterLLM(
        {"private": private_llm, "code": code_llm, "public": public_llm},
        selector=selector,
        default_chain="private",
    )
    # No context -> defaults to private.
    out = "".join(r.stream("hello"))
    assert out == "PP"

    # Set context: code.
    token = capability_context.set({"sensitivity": "code"})
    try:
        assert "".join(r.stream("hello")) == "CC"
    finally:
        capability_context.reset(token)

    # Set context: public.
    token = capability_context.set({"sensitivity": "public"})
    try:
        assert "".join(r.stream("hello")) == "UU"
    finally:
        capability_context.reset(token)


def test_router_default_chain_used_when_sensitivity_is_unknown():
    private_llm = FakeStreamLLM(["P"])
    code_llm = FakeStreamLLM(["C"])
    selector = ChainSelector(
        {"private": "private", "code": "code"}, default_chain="private"
    )
    r = SensitivityRouterLLM(
        {"private": private_llm, "code": code_llm},
        selector=selector,
        default_chain="private",
    )
    token = capability_context.set({"sensitivity": "alien"})  # unknown
    try:
        assert "".join(r.stream("hello")) == "P"
    finally:
        capability_context.reset(token)


def test_router_default_chain_used_when_resolved_chain_is_missing():
    """Selector returns a chain name that's not in chains -> fall back."""
    only_private = FakeStreamLLM(["P"])
    # The selector says map private->private, but ChainSelector only knows that
    # if asked with sensitivity="private". An unknown sensitivity returns
    # default_chain="public", which isn't in chains -> falls through to
    # default_chain of SensitivityRouterLLM.
    selector = ChainSelector({"private": "private"}, default_chain="public")
    r = SensitivityRouterLLM(
        {"private": only_private}, selector=selector, default_chain="private"
    )
    token = capability_context.set({"sensitivity": "anything"})
    try:
        # Selector returns "public" (default), which isn't a key -> falls back
        # to the router's default_chain ("private") -> private_llm.
        assert "".join(r.stream("q")) == "P"
    finally:
        capability_context.reset(token)


def test_router_requires_default_chain_to_exist():
    selector = ChainSelector({}, default_chain="private")
    import pytest

    with pytest.raises(ValueError):
        SensitivityRouterLLM({"code": FakeStreamLLM([])},
                             selector=selector, default_chain="private")


def test_router_generate_also_routes_by_context():
    private_llm = FakeStreamLLM(["P"])
    code_llm = FakeStreamLLM(["C"])
    selector = ChainSelector(
        {"private": "private", "code": "code"}, default_chain="private"
    )
    r = SensitivityRouterLLM(
        {"private": private_llm, "code": code_llm},
        selector=selector, default_chain="private",
    )
    token = capability_context.set({"sensitivity": "code"})
    try:
        assert r.generate("hi") == "C"
    finally:
        capability_context.reset(token)
