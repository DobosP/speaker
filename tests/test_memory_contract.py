"""Contract tests for the P2 ``Memory`` seam.

Exercises BOTH backends behind the one Protocol -- the in-RAM
:class:`SessionMemory` and a no-DB :class:`MemoryManagerAdapter` -- so they
behave identically at the seam:

* ``isinstance(x, Memory)`` (the runtime-checkable Protocol);
* add / search / all / context_for_llm / prune / close all work;
* ``all()`` PRESERVES tags on both backends (R3 -- the adapter's own in-RAM
  ring buffer, not ``recent_messages`` which drops tags);
* a "fact stated in turn 1 is recalled at turn N" flow on an in-RAM backend
  with recall enabled (R1 -- the answered query must be ingested so recall has
  something to find);
* a DEFAULT-OFF neutrality test: with ``recall_enabled=false`` the assistant
  capability calls ``model.stream`` with a ``system`` byte-identical to the
  DEFAULT, proving the gate short-circuits before touching the prompt.

The adapter is built with ``enable_embeddings=False`` + an injected
``pool_factory`` (the fake pool from ``test_memory_pool``) so it never needs a
real PostgreSQL -- mirroring ``tests/test_memory_pool.py``.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

import pytest

# The adapter backend is constructed at collection time in @parametrize(_backends())
# below, which lazily pulls utils.memory -> numpy/psycopg. Skip the whole module
# cleanly in a thin env that lacks those optional deps instead of erroring at
# collection time.
pytest.importorskip("numpy")
pytest.importorskip("psycopg")

from always_on_agent.capabilities import CapabilityRegistry
from always_on_agent.memory import (
    Memory,
    MemoryItem,
    MemoryManagerAdapter,
    SessionMemory,
)
from core.capabilities import DEFAULT_SYSTEM, RecallConfig, attach_llm_capabilities
from core.conversation import RecentContextConfig

# These tests pin the RECALL contract specifically; the orthogonal default-on
# recent-conversation context (core/conversation.py) would otherwise append its
# own block, so it's disabled here. Recent-context has its own tests
# (tests/test_conversation.py).
_NO_RECENT = RecentContextConfig(enabled=False)


# --- a minimal no-DB connection pool ----------------------------------------
#
# Self-contained (no sibling-test import) so a single-file run works. With
# ``enable_embeddings=False`` the adapter never runs a real query: search short-
# circuits, and the only DB touches are the idempotent demo-schema bootstrap +
# the warm-start recent-messages SELECT, both satisfied by an empty cursor.


class _NullCursor:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=()):
        return None

    def fetchall(self):
        return []

    def fetchone(self):
        return None


class _NullConn:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self, *, row_factory=None):
        return _NullCursor()


class _NullPool:
    def __init__(self, conninfo=None, *, min_size, max_size, kwargs=None, open=True):
        self.closed = False

    @contextmanager
    def connection(self):
        yield _NullConn()

    def close(self):
        self.closed = True


# --- backend builders -------------------------------------------------------


def _make_adapter() -> MemoryManagerAdapter:
    """A no-DB adapter: null pool + embeddings off (no real Postgres)."""

    def factory(*, conninfo, min_size, max_size, kwargs):
        return _NullPool(
            conninfo, min_size=min_size, max_size=max_size, kwargs=kwargs, open=True
        )

    return MemoryManagerAdapter(
        db_url="postgresql://fake",
        enable_embeddings=False,
        smart_save=False,
        pool_min_size=1,
        pool_max_size=5,
        pool_factory=factory,
    )


def _backends() -> list[Memory]:
    return [SessionMemory(), _make_adapter()]


# --- a recording fake LLM ---------------------------------------------------


class _RecordingLLM:
    """Captures every ``system`` it is streamed with so a test can assert the
    exact prompt the model received."""

    def __init__(self, reply: str = "an answer") -> None:
        self.reply = reply
        self.systems: list[Optional[str]] = []

    def generate(self, prompt: str, *, system: Optional[str] = None, images=None) -> str:
        self.systems.append(system)
        return self.reply

    def stream(self, prompt: str, *, system: Optional[str] = None, images=None) -> Iterator[str]:
        self.systems.append(system)
        yield self.reply


# --- Protocol conformance ---------------------------------------------------


@pytest.mark.parametrize("mem", _backends())
def test_isinstance_memory(mem: Memory):
    assert isinstance(mem, Memory)
    try:
        # Every verb is present and callable on both backends.
        mem.add("hello world this is a test", tags=("user",))
        assert isinstance(mem.search("hello"), (list, tuple))
        assert isinstance(mem.all(), (list, tuple))
        assert isinstance(mem.context_for_llm("hello"), str)
        assert mem.prune() == 0
    finally:
        mem.close()


@pytest.mark.parametrize("mem", _backends())
def test_all_preserves_tags(mem: Memory):
    """R3: ``all()`` returns the raw (text, tags) handed to ``add()`` on BOTH
    backends -- the adapter from its own ring buffer, not ``recent_messages``
    (which keeps only role + junk-filters and would drop the 'ingested' tag
    that ``test_addressing`` relies on)."""
    try:
        mem.add("HE MURMURED HIS MURDERING", tags=("ingested",))
        items = mem.all()
        assert any(
            isinstance(i, MemoryItem)
            and i.text == "HE MURMURED HIS MURDERING"
            and "ingested" in i.tags
            for i in items
        )
    finally:
        mem.close()


@pytest.mark.parametrize("mem", _backends())
def test_close_is_idempotent_enough(mem: Memory):
    # close() must not raise on either backend (flush + release).
    mem.add("something to flush here", tags=("user",))
    mem.close()


# --- recall flow (R1) -------------------------------------------------------


def test_fact_stated_turn_1_recalled_at_turn_n():
    """R1 on the in-RAM backend: the assistant capability ingests each answered
    query, so a fact established in turn 1 is retrievable at turn N when recall
    is enabled. The answer for the recalling turn carries the recalled block in
    its ``system``."""
    memory = SessionMemory()
    registry = CapabilityRegistry()
    llm = _RecordingLLM()
    attach_llm_capabilities(
        registry,
        llm,
        memory=memory,
        recall=RecallConfig(enabled=True, max_chars=600),
        recent_context=_NO_RECENT,
    )

    # Turn 1: establish a fact (the query itself is ingested by the closure).
    registry.invoke("assistant.answer", "my favorite color is teal", {})
    # A few unrelated turns.
    for _ in range(5):
        registry.invoke("assistant.answer", "what time is it", {})
    llm.systems.clear()
    # Turn N: a query overlapping the turn-1 fact should recall it.
    registry.invoke("assistant.answer", "remind me my favorite color", {})

    assert llm.systems, "model.stream was never called"
    system_used = llm.systems[-1]
    assert "favorite color is teal" in system_used
    # The default system is still appended after the recall block.
    assert system_used.endswith(DEFAULT_SYSTEM)


def test_adapter_recall_degrades_to_empty_without_db():
    """The adapter's recall is the Postgres semantic path
    (``get_context_for_llm`` -> ``search_memory``), gated on embeddings + a
    live DB. With neither (the logic-suite no-DB build), recall returns ``''``
    -- so even with ``recall_enabled=true`` the prompt is left at the DEFAULT.
    This is the safe construct-without-DB degradation (no Postgres-ism leaks,
    no crash). The DB-backed turn-1 -> turn-N recall is asserted in the
    Postgres-marked integration suite (R8), excluded from this logic run."""
    memory = _make_adapter()
    assert memory.context_for_llm("my favorite color") == ""
    registry = CapabilityRegistry()
    llm = _RecordingLLM()
    attach_llm_capabilities(
        registry,
        llm,
        memory=memory,
        recall=RecallConfig(enabled=True, max_chars=600),
        recent_context=_NO_RECENT,
    )
    try:
        registry.invoke("assistant.answer", "my favorite color is teal", {})
        llm.systems.clear()
        registry.invoke("assistant.answer", "remind me my favorite color", {})
        assert llm.systems == [DEFAULT_SYSTEM]
        # But the answered query WAS ingested into the tag-faithful ring (R1):
        # recall would find it once a real DB + embeddings are present.
        assert any("teal" in i.text for i in memory.all())
    finally:
        memory.close()


# --- default-off neutrality -------------------------------------------------


@pytest.mark.parametrize("mem", _backends())
def test_recall_default_off_is_byte_identical(mem: Memory):
    """With ``recall_enabled=false`` (the default), the assistant capability
    must call ``model.stream`` with a ``system`` byte-identical to the DEFAULT
    -- the config gate short-circuits before any recall block is built, so the
    prompt is provably unchanged even when memory already holds related facts.
    """
    registry = CapabilityRegistry()
    llm = _RecordingLLM()
    # Default RecallConfig() is OFF.
    attach_llm_capabilities(registry, llm, memory=mem, recall=RecallConfig(), recent_context=_NO_RECENT)
    try:
        # Pre-load a clearly-relevant fact so a leak would be visible.
        mem.add("my favorite color is teal", tags=("user",))
        llm.systems.clear()
        registry.invoke("assistant.answer", "what is my favorite color", {})
        assert llm.systems == [DEFAULT_SYSTEM]
    finally:
        mem.close()


def test_recall_off_when_memory_absent_is_default_system():
    """No memory wired at all -> the assistant still answers with the DEFAULT
    system (back-compat: the new params default to None)."""
    registry = CapabilityRegistry()
    llm = _RecordingLLM()
    attach_llm_capabilities(registry, llm)
    registry.invoke("assistant.answer", "hello there", {})
    assert llm.systems == [DEFAULT_SYSTEM]


# --- max_chars cap ----------------------------------------------------------


def test_recall_block_is_capped_by_max_chars():
    """The prepended recall block is truncated to ``max_chars`` so a large
    history can't blow up TTFT."""
    memory = SessionMemory()
    registry = CapabilityRegistry()
    llm = _RecordingLLM()
    attach_llm_capabilities(
        registry, llm, memory=memory, recall=RecallConfig(enabled=True, max_chars=20),
        recent_context=_NO_RECENT,
    )
    # A fact long enough that the recall block exceeds the cap.
    memory.add("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda", tags=("user",))
    llm.systems.clear()
    registry.invoke("assistant.answer", "alpha beta gamma delta epsilon", {})
    system_used = llm.systems[-1]
    # system = recall[:max_chars] + "\n\n" + DEFAULT_SYSTEM
    prefix, _, tail = system_used.partition("\n\n" + DEFAULT_SYSTEM)
    assert tail == ""  # the default system is the suffix
    assert len(prefix) <= 20
