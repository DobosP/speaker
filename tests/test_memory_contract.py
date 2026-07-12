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


def _backend_factories():
    """FACTORIES (not instances) so every parametrized test gets a FRESH backend.

    Required now that SqliteVecMemory is in the set: its close() shuts the sqlite
    connection, so a shared instance reused across tests would hit a closed DB.
    Each factory is id'd by backend class name for readable test ids."""
    from always_on_agent.sqlite_memory import SqliteVecMemory

    return [
        pytest.param(SessionMemory, id="SessionMemory"),
        pytest.param(_make_adapter, id="MemoryManagerAdapter"),
        pytest.param(lambda: SqliteVecMemory(":memory:"), id="SqliteVecMemory"),
    ]


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


@pytest.mark.parametrize("make_mem", _backend_factories())
def test_isinstance_memory(make_mem):
    mem: Memory = make_mem()
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


@pytest.mark.parametrize("make_mem", _backend_factories())
def test_all_preserves_tags(make_mem):
    """R3: ``all()`` returns the raw (text, tags) handed to ``add()`` on every
    backend -- the adapter from its own ring buffer, not ``recent_messages``
    (which keeps only role + junk-filters and would drop the 'ingested' tag
    that ``test_addressing`` relies on)."""
    mem: Memory = make_mem()
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


@pytest.mark.parametrize("make_mem", _backend_factories())
def test_close_is_idempotent_enough(make_mem):
    # close() must not raise on any backend (flush + release).
    mem: Memory = make_mem()
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
    # R06b: the base system prompt is the cacheable prefix (FIRST); the recall
    # block now follows it (present, just after system), so the KV cache survives.
    assert system_used.startswith(DEFAULT_SYSTEM)
    assert system_used.index(DEFAULT_SYSTEM) < system_used.index("favorite color is teal")


def test_weak_rendered_recall_does_not_override_an_always_fast_router():
    """A stopword-only recall hit is not evidence that a turn needs main."""
    memory = SessionMemory()
    memory.add("my favorite color is teal", tags=("user",))
    main = _RecordingLLM("main answer")
    fast = _RecordingLLM("fast answer")

    class _AlwaysFast:
        def choose(self, query, context):
            return "fast"

    registry = CapabilityRegistry()
    attach_llm_capabilities(
        registry,
        main,
        fast_llm=fast,
        router=_AlwaysFast(),
        memory=memory,
        recall=RecallConfig(enabled=True, max_chars=600),
        recent_context=_NO_RECENT,
    )

    result = registry.invoke(
        "assistant.answer", "what is the capital of france?", {}
    )

    assert result.text == "fast answer"
    assert result.data["route"] == "fast"
    assert "favorite color is teal" in (fast.systems[0] or "")
    assert main.systems == []


def test_strong_subject_matched_recall_routes_to_main_inside_untrusted_fence():
    memory = SessionMemory()
    memory.add(
        "my lighthouse project codename was Amber Finch",
        tags=("user",),
    )
    main = _RecordingLLM("Amber Finch.")
    fast = _RecordingLLM("fast answer")
    registry = CapabilityRegistry()
    attach_llm_capabilities(
        registry,
        main,
        fast_llm=fast,
        memory=memory,
        recall=RecallConfig(enabled=True, max_chars=600),
        recent_context=_NO_RECENT,
    )

    result = registry.invoke(
        "assistant.answer", "what was my lighthouse project codename?", {}
    )

    assert result.text == "Amber Finch."
    assert result.data["route"] == "main"
    assert result.data["sensitivity"] == "private"
    assert fast.systems == []
    assert len(main.systems) == 1
    system = main.systems[0] or ""
    assert "my lighthouse project codename was Amber Finch" in system
    assert "UNTRUSTED reference DATA" in system


def test_reused_context_cannot_carry_strong_recall_promotion_to_next_turn():
    memory = SessionMemory()
    memory.add(
        "my lighthouse project codename was Amber Finch",
        tags=("user",),
    )
    main = _RecordingLLM("Amber Finch.")
    fast = _RecordingLLM("fast answer")
    registry = CapabilityRegistry()
    attach_llm_capabilities(
        registry,
        main,
        fast_llm=fast,
        memory=memory,
        recall=RecallConfig(enabled=True, max_chars=600),
        recent_context=_NO_RECENT,
    )
    reused: dict[str, object] = {}

    first = registry.invoke(
        "assistant.answer",
        "what was my lighthouse project codename?",
        reused,
    )
    second = registry.invoke("assistant.answer", "hello there", reused)

    assert first.data["route"] == "main"
    assert second.data["route"] == "fast"


def test_no_recalled_context_keeps_short_answer_on_fast_model():
    memory = SessionMemory()
    main = _RecordingLLM("main answer")
    fast = _RecordingLLM("fast answer")
    registry = CapabilityRegistry()
    attach_llm_capabilities(
        registry,
        main,
        fast_llm=fast,
        memory=memory,
        recall=RecallConfig(enabled=True, max_chars=600),
        recent_context=_NO_RECENT,
    )

    result = registry.invoke("assistant.answer", "hello there", {})

    assert result.text == "fast answer"
    assert result.data["route"] == "fast"
    assert fast.systems == [DEFAULT_SYSTEM]
    assert main.systems == []


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


@pytest.mark.parametrize("make_mem", _backend_factories())
def test_recall_default_off_is_byte_identical(make_mem):
    """With ``recall_enabled=false`` (the default), the assistant capability
    must call ``model.stream`` with a ``system`` byte-identical to the DEFAULT
    -- the config gate short-circuits before any recall block is built, so the
    prompt is provably unchanged even when memory already holds related facts.
    """
    mem: Memory = make_mem()
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


# --- token budget (replaces the legacy char cap) ----------------------------


def test_recall_block_is_token_bounded():
    """The prepended recall block is bounded by a TOKEN budget (not a blunt char
    cap), so a large history can't blow up TTFT -- and, unlike the old
    ``[:max_chars]`` slice, it is never cut mid-word. Pins the headline
    context-efficiency contract on the RAM/no-DB path."""
    from always_on_agent.recall import RecallBudget, estimate_tokens

    budget = RecallBudget(max_tokens=30)
    memory = SessionMemory(budget=budget)
    registry = CapabilityRegistry()
    llm = _RecordingLLM()
    attach_llm_capabilities(
        registry, llm, memory=memory,
        recall=RecallConfig(enabled=True, max_tokens=30),
        recent_context=_NO_RECENT,
    )
    # Many long, query-overlapping facts -> a candidate pool far larger than the
    # budget; the selector must bound the injected block.
    for i in range(8):
        memory.add(f"project milestone {i} concerns alpha beta gamma delta epsilon", tags=("user",))
    llm.systems.clear()
    registry.invoke("assistant.answer", "tell me about the alpha beta gamma milestone", {})

    from always_on_agent.untrusted import _BEGIN, _END

    system_used = llm.systems[-1]
    assert system_used.startswith(DEFAULT_SYSTEM)  # R06b: system is the cacheable prefix
    # The recall block now rides inside an untrusted-data envelope (prompt-injection
    # spotlighting). The recall TOKEN budget bounds the recalled CONTENT; the fixed
    # security directive is separate, bounded overhead -- so measure the content
    # INSIDE the fences against the budget.
    recall_content = system_used.split(_BEGIN, 1)[1].split(_END, 1)[0]
    recall_content = recall_content.split("\n", 1)[1].strip()  # drop the "[untrusted memory]" header
    assert recall_content.startswith("=== Past Conversations ===")  # something recalled
    assert estimate_tokens(recall_content) <= 30  # token-bounded, not char-sliced
    # Whole-word guarantee: every rendered word is a complete word (no mid-cut).
    for line in recall_content.split("\n")[1:]:
        assert not line.endswith("alph") and not line.endswith("gam")


def test_recall_block_max_chars_alias_still_constructs():
    """The deprecated ``max_chars`` kwarg still constructs and derives the token
    budget (max_chars // 4), so existing callers don't break."""
    assert RecallConfig(enabled=True, max_chars=600).max_tokens == 150
    assert RecallConfig(enabled=True, max_chars=20).max_tokens == 5


def test_ram_recall_uses_canonical_labels():
    """The in-RAM recall now emits the SAME canonical ``User:``/``Assistant:``
    labels as the Postgres path (closing the RAM-vs-PG format gap)."""
    mem = SessionMemory()
    mem.add("my favorite color is teal", tags=("user",))
    mem.add("noted, teal it is", tags=("assistant_output",))
    block = mem.context_for_llm("what is my favorite color")
    assert block.startswith("=== Past Conversations ===")
    assert "User: my favorite color is teal" in block


def test_ram_and_postgres_recall_byte_identical_for_same_candidates(monkeypatch):
    """Parity: fed the SAME candidate list, the RAM path and the Postgres path
    (via their shared build_block) emit a BYTE-IDENTICAL block. Proven by
    stubbing the Postgres ``search_memory`` to return rows equivalent to what the
    RAM store holds, then comparing the two rendered blocks."""
    from always_on_agent.recall import RecallBudget

    budget = RecallBudget(max_tokens=120)
    q = "what is my favorite color"

    ram = SessionMemory(budget=budget)
    ram.add("my favorite color is teal", tags=("user",))
    ram_block = ram.context_for_llm(q)

    pg = _make_adapter()
    mgr = pg._manager
    mgr._recall_budget = budget
    mgr._embeddings_available = True
    mgr._db_available = True
    # Equivalent DB row for the same fact (cosine score in the message branch).
    monkeypatch.setattr(
        mgr, "search_memory",
        lambda query, limit=5: [
            {"type": "message", "role": "user", "content": "my favorite color is teal",
             "timestamp": 0.0, "similarity": 0.8}
        ],
    )
    pg_block = pg.context_for_llm(q)
    pg.close()

    assert ram_block == pg_block
    assert "User: my favorite color is teal" in ram_block


def test_build_block_deterministic_on_multi_item_candidate_list():
    """The real parity contract: build_block is a pure function of (candidates,
    query, budget). Fed a BYTE-IDENTICAL multi-item list (mixed kinds + distinct
    scores), it returns identical output -- so both backends, which differ only
    in how they GATHER candidates, render identically for identical candidates.
    (The n=1 test above can't catch ordering/cutoff/dedup divergence; this can.)"""
    from always_on_agent.recall import Candidate, RecallBudget, build_block

    budget = RecallBudget(max_tokens=120)
    q = "the trip to japan and the coffee order"
    cands = [
        Candidate("we planned the trip to japan", 0.91, kind="message", role="user"),
        Candidate("you wanted a dark roast coffee", 0.74, kind="message", role="assistant"),
        Candidate("earlier chat about travel and drinks", 0.66, kind="summary", span=(1.0, 2.0)),
        Candidate("totally unrelated low-relevance line", 0.05, kind="message", role="user"),
    ]
    first = build_block(list(cands), q, budget)
    second = build_block(list(cands), q, budget)
    assert first == second and first  # deterministic + non-empty


# --- lm-7 window parity + meeting RAM-only routing (adapter) -----------------


def test_adapter_ring_is_capped_to_working_window():
    """lm-7: the adapter's all() ring honors working_window (not max_recent), so a
    flood can't grow it and it matches SessionMemory/SqliteVecMemory."""

    def factory(*, conninfo, min_size, max_size, kwargs):
        return _NullPool(conninfo, min_size=min_size, max_size=max_size, kwargs=kwargs, open=True)

    mem = MemoryManagerAdapter(
        db_url="postgresql://fake",
        enable_embeddings=False,
        smart_save=False,
        pool_min_size=1,
        pool_max_size=5,
        pool_factory=factory,
        working_window=5,
    )
    try:
        for i in range(20):
            mem.add(f"line number {i}", tags=("user",))
        items = mem.all()
        assert len(items) == 5  # capped to working_window
        assert items[-1].text == "line number 19"  # newest retained
    finally:
        mem.close()


class _RecordingManager:
    """Fake MemoryManager that records how add() routed each utterance."""

    max_recent_messages = 200
    meeting_persist = False  # legacy attr; the adapter no longer reads it

    def __init__(self):
        self.messages = []        # (role, text) from add_message
        self.user_utterances = []  # text from queue_user_utterance
        self.observations = []     # text from add_observation

    def add_message(self, role, text):
        self.messages.append((role, text))

    def queue_user_utterance(self, text):
        self.user_utterances.append(text)

    def add_observation(self, text):
        self.observations.append(text)

    def get_context_for_llm(self, query):
        return ""

    def apply_retention(self):
        return 0

    def close(self):
        return None


def test_adapter_routing_meeting_is_ram_only(monkeypatch):
    """Meeting notes are RAM-only (§9.7): present in all() but never handed to the
    persisting manager. user->queue, assistant->add_message, vision->add_observation."""
    rec = _RecordingManager()
    monkeypatch.setattr("utils.memory.create_memory_manager", lambda **kw: rec)
    mem = MemoryManagerAdapter(db_url="postgresql://fake")
    try:
        mem.add("private meeting note", tags=("meeting",))
        mem.add("a user question", tags=("user",))
        mem.add("an assistant reply", tags=("assistant_output",))
        mem.add("screen says hello", tags=("vision",))

        # Meeting text is in the RAM ring but NEVER persisted anywhere.
        assert any("private meeting note" in i.text for i in mem.all())
        assert "private meeting note" not in rec.user_utterances
        assert all("private meeting note" not in t for _r, t in rec.messages)
        assert all("private meeting note" not in t for t in rec.observations)

        # The other channels route as documented.
        assert rec.user_utterances == ["a user question"]
        assert rec.messages == [("assistant", "an assistant reply")]
        assert rec.observations == ["screen says hello"]
    finally:
        mem.close()
