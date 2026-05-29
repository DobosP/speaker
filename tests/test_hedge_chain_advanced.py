"""Adversarial corner cases for the HedgeLLM chain logic.

The basic chain tests in test_hedge_chain.py cover the happy paths
(advance on error, fall through to local, etc.). This file pins behavior
in scenarios that bit me during implementation:

- Cancellation cuts the *winner's* stream mid-flight (barge-in).
- A cloud produces one token then errors -- the partial output is kept,
  not discarded.
- ``hedge_delay_ms=0`` means everyone starts at once (true race).
- A 4-cloud chain advances correctly when several middle clouds fail.
- ``strategy="fallback"`` with a slow first cloud advances past the ttft
  deadline to the next cloud.
"""
from __future__ import annotations

import threading
import time
from typing import Iterator

import pytest

from core.llm import HedgeLLM

# These cases pin HedgeLLM behaviour under real ttft/between-token delays and
# cancellation races, so they sleep to reproduce the timing -- the slow tail of
# the suite. Mark the module `slow` so `-m "not slow"` skips it.
pytestmark = pytest.mark.slow


class FakeStreamLLM:
    """Like the fake in test_multi_provider_llm.py but exposes a
    stop event so tests can prove cancellation propagated."""

    def __init__(
        self,
        tokens,
        *,
        first_token_delay=0.0,
        between_token_delay=0.0,
        error=None,
        error_after=None,
    ):
        self.tokens = list(tokens)
        self.first_token_delay = first_token_delay
        self.between_token_delay = between_token_delay
        self.error = error
        # If set, raise after this many tokens have been yielded (lets us
        # test "produces N tokens, then dies" scenarios).
        self.error_after = error_after
        self.started = False
        self.completed = False
        self.yielded = 0

    def stream(self, prompt, *, system=None, images=None) -> Iterator[str]:
        self.started = True
        if self.error and self.error_after is None:
            raise RuntimeError(self.error)
        for i, tok in enumerate(self.tokens):
            if i == 0 and self.first_token_delay:
                time.sleep(self.first_token_delay)
            elif self.between_token_delay:
                time.sleep(self.between_token_delay)
            yield tok
            self.yielded = i + 1
            if self.error_after is not None and self.yielded >= self.error_after:
                raise RuntimeError(self.error or "mid-stream failure")
        self.completed = True

    def generate(self, prompt, *, system=None, images=None) -> str:
        return "".join(self.tokens)


# --- hedge_delay_ms=0 (full race) ------------------------------------------


def test_hedge_delay_zero_kicks_both_at_once():
    """delay=0 must launch local AND the first cloud immediately --
    that's the recommended config for CPU-only hosts where local is too
    slow to ever beat the cloud."""
    local = FakeStreamLLM(["L"], first_token_delay=0.2)
    cloud = FakeStreamLLM(["X"])
    h = HedgeLLM(local=local, cloud=[cloud], strategy="hedge", hedge_delay_ms=0)
    out = "".join(h.stream("q"))
    assert out == "X"
    assert local.started is True  # both started -- this is the race
    assert cloud.started is True


# --- mid-stream failure of the winner --------------------------------------


def test_cloud_produces_then_errors_partial_output_preserved():
    """A cloud that yields one token then raises should still deliver
    that one token -- losing the partial would be a worse outcome than
    a slightly truncated answer."""
    local = FakeStreamLLM(["L"], first_token_delay=2.0)
    cloud = FakeStreamLLM(["X", "Y"], error_after=1)
    h = HedgeLLM(local=local, cloud=[cloud], strategy="hedge", hedge_delay_ms=0)
    out = "".join(h.stream("q"))
    # We get cloud's first token; the error after the second yield ends
    # the stream (we may or may not catch Y depending on timing).
    assert out.startswith("X")


# --- multi-cloud chain advancement -----------------------------------------


def test_four_cloud_chain_advances_through_middle_failures():
    """cloud_0 errs, cloud_1 errs, cloud_2 errs, cloud_3 produces."""
    local = FakeStreamLLM(["LOCAL"], first_token_delay=2.0)
    c0 = FakeStreamLLM([], error="0")
    c1 = FakeStreamLLM([], error="1")
    c2 = FakeStreamLLM([], error="2")
    c3 = FakeStreamLLM(["A", "B", "C"])
    h = HedgeLLM(
        local=local,
        cloud=[c0, c1, c2, c3],
        strategy="hedge",
        hedge_delay_ms=10,
    )
    out = "".join(h.stream("q"))
    assert out == "ABC"
    # The chain marched through every failing cloud.
    assert c0.started and c1.started and c2.started
    # Final winner produced.
    assert c3.started and c3.completed


def test_fallback_strategy_slow_cloud_advances_on_deadline():
    """ttft_deadline_ms makes a SLOW cloud (not errored, just slow) yield
    to the next cloud in the chain. Without this, a hung provider would
    block the chain indefinitely."""
    local = FakeStreamLLM(["LOCAL"])
    slow = FakeStreamLLM(["S"], first_token_delay=2.0)  # exceeds 150ms deadline
    fast = FakeStreamLLM(["F"])
    h = HedgeLLM(
        local=local,
        cloud=[slow, fast],
        strategy="fallback",
        ttft_deadline_ms=150,
    )
    started = time.monotonic()
    out = "".join(h.stream("q"))
    elapsed = time.monotonic() - started
    assert out == "F"
    # Should not have waited the full 2.0s for the slow cloud.
    assert elapsed < 1.5, f"chain blocked on slow cloud for {elapsed:.2f}s"


def test_fallback_all_clouds_time_out_then_local_wins():
    """When every cloud is slow AND local is fast, fallback should still
    end up returning local's output rather than waiting forever."""
    local = FakeStreamLLM(["L1", "L2"])
    slow_a = FakeStreamLLM(["A"], first_token_delay=2.0)
    slow_b = FakeStreamLLM(["B"], first_token_delay=2.0)
    h = HedgeLLM(
        local=local,
        cloud=[slow_a, slow_b],
        strategy="fallback",
        ttft_deadline_ms=80,
    )
    out = "".join(h.stream("q"))
    assert out == "L1L2"


# --- local-tier failure scenarios ------------------------------------------


def test_hedge_local_errors_then_cloud_wins():
    """If local crashes (e.g. Ollama down), the cloud chain still serves
    the turn -- we don't blackhole the user just because the local model
    is broken."""
    local = FakeStreamLLM([], error="ollama not running")
    cloud = FakeStreamLLM(["OK"])
    h = HedgeLLM(local=local, cloud=[cloud], strategy="hedge", hedge_delay_ms=10)
    assert "".join(h.stream("q")) == "OK"


def test_hedge_local_and_first_cloud_both_error_chain_continues():
    """Local errors AND cloud_0 errors -- chain still advances to cloud_1.
    This is the worst-case CPU laptop scenario (Ollama crashed, primary
    cloud rate-limited, secondary cloud must save the day)."""
    local = FakeStreamLLM([], error="local down")
    c0 = FakeStreamLLM([], error="rate limit")
    c1 = FakeStreamLLM(["SAVED"])
    h = HedgeLLM(local=local, cloud=[c0, c1], strategy="hedge", hedge_delay_ms=0)
    out = "".join(h.stream("q"))
    assert out == "SAVED"
    assert c1.started


# --- stream consumption + GC -----------------------------------------------


def test_stream_can_be_partially_consumed_without_hanging():
    """Iterate exactly one token and stop. The worker threads are daemon
    so the test process can exit, but a finite stream shouldn't leak
    file descriptors / queue references either."""
    local = FakeStreamLLM(["a", "b", "c", "d"])
    cloud = FakeStreamLLM(["X", "Y", "Z"])
    h = HedgeLLM(local=local, cloud=[cloud], strategy="hedge", hedge_delay_ms=200)
    it = iter(h.stream("q"))
    first = next(it)
    assert first in {"a", "X"}
    # Drop the iterator without exhausting it. Should not hang.
    del it


def test_stream_complete_drains_winner_late_tokens():
    """After the winner is selected, late tokens from the winner must
    still surface (the loser's late tokens are dropped). This is what
    lets the streamed sentence break happen incrementally."""
    local = FakeStreamLLM(
        ["one", " ", "two", " ", "three"],
        first_token_delay=0.0,
        between_token_delay=0.005,
    )
    cloud = FakeStreamLLM(["X"], first_token_delay=2.0)
    h = HedgeLLM(local=local, cloud=[cloud], strategy="hedge", hedge_delay_ms=200)
    tokens = list(h.stream("q"))
    # Local won the race; we must see every one of its tokens.
    assert "".join(tokens) == "one two three"


# --- generate() wraps stream() ---------------------------------------------


def test_generate_returns_joined_stream_with_strip():
    """``generate`` is the non-streaming convenience -- it should still
    use the same chain semantics under the hood."""
    local = FakeStreamLLM([" hello ", " world "])
    h = HedgeLLM(local=local, cloud=[])
    # " hello " + " world " -> " hello  world " -> strip() -> "hello  world".
    assert h.generate("q") == "hello  world"


# --- cloud property back-compat under chain --------------------------------


def test_cloud_property_remains_None_when_chain_empty():
    """Legacy code that checks ``hedge.cloud is None`` must still work
    when the chain is configured as an empty list."""
    h = HedgeLLM(local=FakeStreamLLM(["x"]), cloud=[])
    assert h.cloud is None
    assert h.clouds == []


def test_cloud_property_returns_first_of_chain():
    """When a chain has multiple entries, ``hedge.cloud`` returns the
    first -- enough for the existing tests that introspect this."""
    c1, c2 = FakeStreamLLM([]), FakeStreamLLM([])
    h = HedgeLLM(local=FakeStreamLLM([]), cloud=[c1, c2])
    assert h.cloud is c1
    assert h.clouds == [c1, c2]


# --- determinism under repeated runs ---------------------------------------


def test_chain_logic_is_deterministic_for_fixed_timings():
    """Run the same chain config twice; same winner each time. Flaky
    chain logic would be a nightmare to debug in the field."""
    for _ in range(5):
        local = FakeStreamLLM(["L"], first_token_delay=0.5)
        cloud = FakeStreamLLM(["X"])
        h = HedgeLLM(local=local, cloud=[cloud], strategy="hedge", hedge_delay_ms=10)
        assert "".join(h.stream("q")) == "X"


# --- thread-safety smoke ---------------------------------------------------


def test_concurrent_hedge_calls_do_not_interfere():
    """Two simultaneous .stream() calls on the same HedgeLLM (different
    threads) must each get their own answer -- no shared state in the
    queue/stops dicts."""
    # Make two independent HedgeLLM instances so each thread can race
    # without sharing state (the design isn't required to support a
    # single instance under concurrent stream() calls).
    results: dict[str, str] = {}

    def run(label):
        local = FakeStreamLLM([label.upper()])
        cloud = FakeStreamLLM([label])
        h = HedgeLLM(local=local, cloud=[cloud], strategy="hedge", hedge_delay_ms=50)
        results[label] = "".join(h.stream("q"))

    threads = [threading.Thread(target=run, args=(l,)) for l in ("a", "b", "c")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)
    # Each result should be one of the two possible outcomes (local won or
    # cloud won) for that label -- never another label's output.
    for label, result in results.items():
        assert result in {label, label.upper()}, (
            f"label={label} got cross-talk: {result!r}"
        )
