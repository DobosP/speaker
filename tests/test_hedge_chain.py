"""Tests for HedgeLLM chain support (multi-cloud failover).

The single-cloud back-compat is already covered by test_multi_provider_llm.py.
This file exercises the new code path: ``cloud`` is a list of clients, the
hedger races them in sequence with local always as the final safety net.
"""
from __future__ import annotations

import threading
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


class HangingLLM:
    """A source that blocks BEFORE its first token and never yields, errors, or
    completes on its own -- models an in-process ``LlamaCppLLM`` whose native
    call wedges (no socket timeout reaps it). Unblocks only when the consumer
    stops it via the worker's stop event closing the generator (GeneratorExit),
    or when ``release`` is set, so the test process never leaks a live thread."""

    def __init__(self):
        self.started = False
        self.released = threading.Event()
        self.closed = False

    def stream(self, prompt, *, system=None, images=None) -> Iterator[str]:
        self.started = True
        try:
            # Block as a hung pre-first-token native call would. Never yields.
            self.released.wait(timeout=30.0)
            return
            yield  # pragma: no cover -- make this a generator
        finally:
            self.closed = True

    def generate(self, prompt, *, system=None, images=None) -> str:  # pragma: no cover
        return ""


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


# --- pre-first-token wall-clock budget (P1 low) ----------------------------


def _budget(seconds: float) -> dict:
    """Class-constant overrides that shrink the wall-clock budget for a fast
    test (the production floor is 30s). Applied per-instance."""
    return {
        "WINNER_SELECT_BUDGET_FLOOR": seconds,
        "WINNER_SELECT_TTFT_BUDGET_MULT": 0,  # budget := the floor alone
    }


def test_hung_local_pre_first_token_is_reaped_within_wall_clock_budget():
    """A local source that blocks before any token (no error, no completion)
    must NOT wedge the turn forever: the bounded wall-clock winner-selection
    budget reaps it and the stream ends cleanly, riding whatever cloud has."""
    hung_local = HangingLLM()
    cloud = FakeStreamLLM([], error="cloud also down")  # nothing usable
    h = HedgeLLM(local=hung_local, cloud=[cloud], strategy="hedge", hedge_delay_ms=0)
    # Shrink the budget so the test is fast (production floor is 30s).
    h.WINNER_SELECT_BUDGET_FLOOR = 0.3
    h.WINNER_SELECT_TTFT_BUDGET_MULT = 0
    t0 = time.monotonic()
    out = "".join(h.stream("q"))
    elapsed = time.monotonic() - t0
    # No token produced; ended cleanly (empty), not wedged.
    assert out == ""
    assert hung_local.started is True
    # Reaped within a small multiple of the budget, never the 30s default.
    assert elapsed < 2.0, f"hung pre-first-token wedged the turn for {elapsed:.2f}s"
    hung_local.released.set()  # let the daemon worker exit promptly


def test_wall_clock_budget_does_not_truncate_a_healthy_slow_winner():
    """The budget bounds only the PRE-first-token wait: a winner that takes a
    while to produce its first token but then streams fine must NOT be cut by
    the winner-selection budget (that's the POST-first-token DRAIN_IDLE_TIMEOUT
    job). Here local's first token lands inside the budget, then it streams."""
    local = FakeStreamLLM(["a", "b", "c"], first_token_delay=0.2)
    cloud = FakeStreamLLM([], error="down")
    h = HedgeLLM(local=local, cloud=[cloud], strategy="hedge", hedge_delay_ms=0)
    h.WINNER_SELECT_BUDGET_FLOOR = 0.5  # first token (0.2s) lands inside it
    h.WINNER_SELECT_TTFT_BUDGET_MULT = 0
    assert "".join(h.stream("q")) == "abc"


def test_winner_select_budget_is_finite_and_scales_with_chain_length():
    """The budget is always bounded (never inf) and grows with the number of
    sources so a healthy fallback chain that retires each cloud at its own
    ttft_deadline is never tripped."""
    # ttft_deadline large enough that the scaling term clears the 30s floor.
    one = HedgeLLM(local=FakeStreamLLM(["x"]), cloud=[FakeStreamLLM(["c"])],
                   ttft_deadline_ms=10000)
    many = HedgeLLM(
        local=FakeStreamLLM(["x"]),
        cloud=[FakeStreamLLM(["c"]) for _ in range(3)],
        ttft_deadline_ms=10000,
    )
    b1 = one._winner_select_budget()
    b3 = many._winner_select_budget()
    assert b1 != float("inf") and b3 != float("inf")
    assert b3 > b1  # more sources -> more pre-first-token headroom


# --- per-call hedge_delay_ms override (PINNED CONTRACT) --------------------


def test_hedge_delay_ms_override_makes_local_lose_when_zero():
    """With a large constructor hedge_delay the cloud would never get a chance
    against an instant local; a per-call hedge_delay_ms=0 overrides that for the
    turn, launching the cloud immediately so it can win against a slow local."""
    local = FakeStreamLLM(["L"], first_token_delay=0.3)
    cloud = FakeStreamLLM(["C"])
    h = HedgeLLM(local=local, cloud=[cloud], strategy="hedge", hedge_delay_ms=5000)
    # Override the 5s constructor delay down to 0 for this turn -> cloud races.
    out = "".join(h.stream("q", hedge_delay_ms=0))
    assert out == "C"
    assert cloud.started is True


def test_hedge_delay_ms_override_none_is_byte_identical_default():
    """Default (None) leaves the constructor hedge_delay untouched: a large
    constructor delay still suppresses the cloud, so local wins."""
    local = FakeStreamLLM(["L"])
    cloud = FakeStreamLLM(["C"], first_token_delay=0.05)
    h = HedgeLLM(local=local, cloud=[cloud], strategy="hedge", hedge_delay_ms=5000)
    # No override -> 5s hedge delay -> local (instant) wins before cloud starts.
    assert "".join(h.stream("q")) == "L"
    assert cloud.started is False


def test_hedge_delay_ms_override_does_not_mutate_constructor_value():
    """The per-call override is per-turn only: the instance's hedge_delay is
    unchanged after a call that overrode it."""
    h = HedgeLLM(
        local=FakeStreamLLM(["L"]),
        cloud=[FakeStreamLLM(["C"])],
        strategy="hedge",
        hedge_delay_ms=150,
    )
    before = h.hedge_delay
    "".join(h.stream("q", hedge_delay_ms=0))
    assert h.hedge_delay == before == 0.15


def test_hedge_delay_ms_override_ignored_by_fallback_strategy():
    """fallback uses ttft_deadline, not a hedge delay; passing hedge_delay_ms
    must be harmless (no crash, normal advancement)."""
    local = FakeStreamLLM(["L"], first_token_delay=2.0)
    c0 = FakeStreamLLM([], error="boom")
    c1 = FakeStreamLLM(["X"])
    h = HedgeLLM(local=local, cloud=[c0, c1], strategy="fallback", ttft_deadline_ms=300)
    out = "".join(h.stream("q", hedge_delay_ms=0))
    assert out == "X"


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


def test_router_forwards_hedge_delay_ms_to_a_hedge_backend():
    """The router is a transparent dispatcher: a per-turn hedge_delay_ms must
    reach the chosen HedgeLLM chain (PINNED CONTRACT -- routing passes it)."""
    # A large constructor hedge delay would let local win; the per-call override
    # to 0 races the cloud, which then wins against a slow local.
    chain = HedgeLLM(
        local=FakeStreamLLM(["L"], first_token_delay=0.3),
        cloud=[FakeStreamLLM(["C"])],
        strategy="hedge",
        hedge_delay_ms=5000,
    )
    r = SensitivityRouterLLM({"private": chain}, selector=None, default_chain="private")
    assert "".join(r.stream("q", hedge_delay_ms=0)) == "C"


def test_router_hedge_delay_ms_is_dropped_for_non_hedge_backend():
    """A non-HedgeLLM backing client (plain LLMClient) doesn't accept the kwarg;
    the router must not pass it through (default behaviour, no crash)."""
    plain = FakeStreamLLM(["P"])  # FakeStreamLLM.stream has no hedge_delay_ms
    r = SensitivityRouterLLM({"private": plain}, selector=None, default_chain="private")
    # Passing the override must not raise even though plain ignores it.
    assert "".join(r.stream("q", hedge_delay_ms=0)) == "P"
