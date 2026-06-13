"""
Multi-layer memory system with PostgreSQL and vector search.

Architecture:
- Layer 1: Recent messages (short-term, last N messages -- in-process)
- Layer 2: Conversation summaries (medium-term, condensed history -- DB)
- Layer 3: Vector embeddings (long-term, semantic search via pgvector -- DB)

May-2026 audit fixes (PR-2):

- **Thread-safe connections.** Replaced a single shared ``psycopg2.connection``
  (not thread-safe; corrupted under the ``MemoryWriter`` Timer thread + the
  request thread) with a ``psycopg_pool.ConnectionPool``. Every DB call site
  uses ``with self._pool.connection() as conn:`` to acquire a per-call
  connection from the pool.

- **Per-row embedding dimension.** ``embedding`` is now the unconstrained
  pgvector ``vector`` type with paired ``embedding_dim INTEGER`` +
  ``embedder_id VARCHAR(64)`` columns and CHECK constraints. Swapping
  embedders (e.g. all-MiniLM-L6-v2 -> bge-small) no longer requires
  destroying the table; old rows coexist with new ones, queries filter
  on embedder_id before the similarity scan.

- **HNSW partial indexes per embedder** (not IVFFlat on empty table).
  See migrations/004_hnsw_indexes.sql for the index DDL.

Schema is managed by ``yoyo-migrations`` under ``migrations/``; run
``python tools/migrate.py apply`` once per deploy. The Python layer
performs the same idempotent CREATE TABLE IF NOT EXISTS on first
connect for the demo / dev path (so ``python utils/memory.py`` still
works without running migrations), but production should use the
yoyo path.
"""
import hashlib
import os
import re
import threading
from dataclasses import dataclass, replace as _dc_replace
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from utils.memory_config import MemoryWriterConfig, config_from_dict
from utils.memory_writer import MemoryWriter, is_junk_stt_text

# The shared, backend-neutral recall selector. Postgres-isms (SQL, pgvector,
# pool, embedder_id) stay HERE; the selection intelligence (token budget,
# adaptive cutoff, dedup, compression) lives in the brain module so the RAM and
# Postgres paths emit byte-identical blocks. recall.py imports nothing from
# utils, so this is a one-way dependency (no cycle).
from always_on_agent.recall import Candidate, RecallBudget, build_block, estimate_tokens

# Uniform relevance score for profile candidates in the recall sub-pass. The
# exact value is irrelevant -- profile rows are ranked only against each other in
# their OWN build_block pass (never against cosine scores, per the Candidate
# invariant), and a uniform score makes the adaptive cutoff keep them all and the
# token budget bound the volume. (The stored per-row confidence is a WRITE-side
# floor, deliberately not used as a recall score.)
_PROFILE_RECALL_SCORE = 1.0


def _finite(value, default: float = 0.0) -> float:
    """Coerce a similarity to a finite float (NaN/inf/garbage -> ``default``).

    A NaN cosine score would sort unpredictably and poison the adaptive cutoff's
    gap math, so sanitize at the candidate boundary."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if f != f or f == float("inf") or f == float("-inf"):  # NaN or +/-inf
        return default
    return f


def _to_epoch(value) -> float:
    """Coerce a timestamp (``datetime`` | epoch number | ``None``) to epoch secs.

    Postgres ``TIMESTAMPTZ`` rows come back as ``datetime`` while the in-RAM path
    uses epoch floats; the recall selector only needs a consistent orderable
    number for recency/span checks, so normalize to seconds (``0.0`` if missing)."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value.timestamp())
    except Exception:  # noqa: BLE001 - a malformed timestamp must not break recall
        return 0.0

# Database (psycopg3 + connection pool). Both are optional so the rest of
# the runtime + test suite work without a Postgres install -- ``MemoryManager``
# silently degrades to in-process-only when the import fails.
try:
    import psycopg  # type: ignore
    from psycopg.rows import dict_row  # type: ignore
    from psycopg_pool import ConnectionPool  # type: ignore
    POSTGRES_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by import-smoke
    POSTGRES_AVAILABLE = False

# Embeddings (sentence-transformers). Also optional.
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:  # pragma: no cover
    EMBEDDINGS_AVAILABLE = False


# Default embedder identity. Pinned here so the same string winds up in
# ``embedder_id`` on every row, partial-HNSW indexes match it, and the
# audit-time backfill of legacy rows knows what to fill in.
DEFAULT_EMBEDDER_ID = "all-MiniLM-L6-v2"
DEFAULT_EMBEDDER_DIM = 384


# Idempotent schema for the demo / fresh-DB path. Production deploys run
# ``python tools/migrate.py apply`` instead; this lets a developer poke at
# the memory layer without learning yoyo first. Kept in sync with
# migrations/001_init.sql + 002_constraints.sql + 004_hnsw_indexes.sql.
_DEMO_SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS messages (
    id              BIGSERIAL PRIMARY KEY,
    session_id      VARCHAR(64) NOT NULL,
    role            VARCHAR(20) NOT NULL,
    content         TEXT NOT NULL,
    timestamp       TIMESTAMPTZ DEFAULT NOW(),
    embedding       vector,
    embedding_dim   INTEGER,
    embedder_id     VARCHAR(64),
    raw_text        TEXT,
    cleaned_text    TEXT,
    source          VARCHAR(32) DEFAULT 'user_final',
    confidence      FLOAT DEFAULT 1.0,
    saved_at        TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS summaries (
    id              BIGSERIAL PRIMARY KEY,
    session_id      VARCHAR(64),
    summary         TEXT NOT NULL,
    topics          TEXT[],
    user_preferences TEXT[],
    start_time      TIMESTAMPTZ,
    end_time        TIMESTAMPTZ,
    message_count   INT,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    embedding       vector,
    embedding_dim   INTEGER,
    embedder_id     VARCHAR(64)
);

CREATE TABLE IF NOT EXISTS user_profile (
    id              BIGSERIAL PRIMARY KEY,
    key             VARCHAR(255) UNIQUE NOT NULL,
    value           TEXT NOT NULL,
    confidence      FLOAT DEFAULT 1.0,
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_messages_session   ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_messages_embedder  ON messages(embedder_id);
CREATE INDEX IF NOT EXISTS idx_summaries_session  ON summaries(session_id);
"""


# Constraints are split out so they can fail loud if the schema has been
# touched by hand (or by an older deployment); we run them in a separate
# DO block with a "constraint already exists" tolerance.
_DEMO_CONSTRAINTS_SQL = """
DO $$ BEGIN
    BEGIN
        ALTER TABLE messages ADD CONSTRAINT embedding_dim_matches
            CHECK (embedding IS NULL OR vector_dims(embedding) = embedding_dim);
    EXCEPTION WHEN duplicate_object THEN NULL; END;
    BEGIN
        ALTER TABLE messages ADD CONSTRAINT embedder_present_with_embedding
            CHECK ((embedding IS NULL) = (embedder_id IS NULL));
    EXCEPTION WHEN duplicate_object THEN NULL; END;
    BEGIN
        ALTER TABLE summaries ADD CONSTRAINT sum_embedding_dim_matches
            CHECK (embedding IS NULL OR vector_dims(embedding) = embedding_dim);
    EXCEPTION WHEN duplicate_object THEN NULL; END;
    BEGIN
        ALTER TABLE summaries ADD CONSTRAINT sum_embedder_present_with_embedding
            CHECK ((embedding IS NULL) = (embedder_id IS NULL));
    EXCEPTION WHEN duplicate_object THEN NULL; END;
END $$;
"""


_DEMO_HNSW_SQL = """
DROP INDEX IF EXISTS idx_messages_embedding;
DROP INDEX IF EXISTS idx_summaries_embedding;

CREATE INDEX IF NOT EXISTS idx_messages_emb_{embedder_safe}
    ON messages USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
    WHERE embedder_id = '{embedder_id}';

CREATE INDEX IF NOT EXISTS idx_summaries_emb_{embedder_safe}
    ON summaries USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
    WHERE embedder_id = '{embedder_id}';
"""


# Deterministic high-signal profile patterns (R8). Each maps a user phrase to a
# ``user_profile`` (key, value) at a confidence FLOOR of >= 0.9 -- only crisp,
# self-reported facts, never a fuzzy LLM guess (that path runs separately on the
# writer thread). ``call me X`` and ``my name is X`` both populate ``name``; the
# value is captured up to sentence-ending punctuation.
_PROFILE_PATTERNS: tuple[tuple[str, "re.Pattern[str]"], ...] = (
    ("name", re.compile(r"\bmy name is\s+(?P<v>[^.,!?;]+)", re.IGNORECASE)),
    ("name", re.compile(r"\bcall me\s+(?P<v>[^.,!?;]+)", re.IGNORECASE)),
    ("location", re.compile(r"\bi live in\s+(?P<v>[^.,!?;]+)", re.IGNORECASE)),
    ("preference", re.compile(r"\bi prefer\s+(?P<v>[^.,!?;]+)", re.IGNORECASE)),
)
_PROFILE_CONFIDENCE = 0.9


@dataclass
class Message:
    """A single message in the conversation."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_dict(self) -> dict:
        return {
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class ConversationSummary:
    """Summary of a conversation segment."""
    summary: str
    topics: List[str]
    user_preferences: List[str]
    start_time: datetime
    end_time: datetime
    message_count: int


class MemoryManager:
    """
    Multi-layer memory manager with PostgreSQL backend.

    Connection management is a thread-safe ``psycopg_pool.ConnectionPool``
    (psycopg3); every DB call site acquires a connection via
    ``with self._pool.connection()`` and releases it on the way out, so the
    background writer thread + the request thread + the summary thread
    each get their own short-lived connection rather than fighting over a
    shared ``psycopg2.connection`` (the audited bug).
    """

    def __init__(
        self,
        db_url: str = None,
        session_id: str = None,
        max_recent_messages: int = 20,
        max_context_tokens: int = 2000,
        embedding_model: str = DEFAULT_EMBEDDER_ID,
        enable_embeddings: bool = True,
        smart_save: bool = True,
        persist_roles: tuple = ("user",),
        flush_interval_sec: float = 240.0,
        min_user_words: int = 3,
        memory_config: Optional[Dict[str, Any]] = None,
        memory_writer_config: Optional[MemoryWriterConfig] = None,
        text_cleaner: Optional[Callable[[str, str], Optional[str]]] = None,
        pool_min_size: int = 2,
        pool_max_size: int = 5,
        pool_factory: Optional[Callable] = None,
        summarizer: Optional[Callable[[str], str]] = None,
        profile_enabled: bool = False,
        episodic_ttl_days: int = 90,
        summary_ttl_days: int = 365,
        recall_budget: Optional[RecallBudget] = None,
    ):
        """Initialize the memory manager.

        Args:
            db_url:               PostgreSQL connection URL (or ``$DATABASE_URL``).
            session_id:           Unique session identifier (auto-generated if None).
            max_recent_messages:  Cap on Layer-1 short-term memory.
            max_context_tokens:   Approx token budget before summarizing.
            embedding_model:      sentence-transformer model name (also the
                                  embedder_id stored on each row).
            pool_min_size:        Pool low-water mark. 2 is enough for 1 reader
                                  + 1 background writer; covers warm-pool latency.
            pool_max_size:        Pool ceiling. 5 covers the realistic burst of
                                  (reader + writer + summary thread + admin).
            pool_factory:         Injectable factory ``(conninfo, min_size,
                                  max_size, kwargs) -> pool``. Tests pass a fake
                                  with a ``BoundedSemaphore`` to exercise the
                                  concurrency contract without a real Postgres.
            summarizer:           Fast-tier LLM callable ``str -> str`` for the
                                  rolling summary (R2). Runs OFF the bus thread
                                  on the writer background thread; keyword
                                  fallback when ``None``.
            profile_enabled:      Gate (default OFF, Postgres-only, R8) for the
                                  ingest-time user-profile extractor.
            episodic_ttl_days:    Age TTL for ``messages`` -- summarize-then-evict
                                  past this age in ``apply_retention``.
            summary_ttl_days:     Age TTL for ``summaries`` (long; profile never
                                  TTL'd).
        """
        self.db_url = db_url or os.getenv('DATABASE_URL', 'postgresql:///voice_assistant')
        self.session_id = session_id or self._generate_session_id()
        self.max_recent_messages = max_recent_messages
        self.max_context_tokens = max_context_tokens
        self.enable_embeddings = enable_embeddings
        self.smart_save = smart_save
        self.persist_roles = tuple(persist_roles)
        self.min_user_words = max(1, int(min_user_words))
        self._writer_config = memory_writer_config or config_from_dict(memory_config)
        if flush_interval_sec is not None:
            self._writer_config.save_interval_sec = max(30.0, float(flush_interval_sec))
        self._text_cleaner = text_cleaner
        self._writer: Optional[MemoryWriter] = None
        self._last_assistant_text = ""

        # P2b producers (R2/R8). The summarizer + profile LLM work runs OFF the
        # bus thread; these knobs are wired through create_memory_manager.
        self._summarizer = summarizer
        self.profile_enabled = bool(profile_enabled)
        self.episodic_ttl_days = max(0, int(episodic_ttl_days))
        self.summary_ttl_days = max(0, int(summary_ttl_days))
        # Guards _check_and_summarize against scheduling a second summary job
        # while one is already in flight on the background thread.
        self._summary_in_flight = False
        self._summary_lock = threading.Lock()

        # Recall budget: the shared token-budget contract for get_context_for_llm.
        # The candidate POOL we over-fetch from the DB is derived from the budget
        # (clamped) so the adaptive selector -- not a fixed SQL LIMIT -- decides
        # what is injected. NOTE: this does NOT touch _estimate_tokens / the
        # summarize trigger, which keep their own (words*1.3) estimator.
        self._recall_budget = recall_budget or RecallBudget()
        self._recall_pool = max(6, min(16, self._recall_budget.max_tokens // 20))
        # Rolling summary head: the prior summary folded into the next one so
        # the layer-2 record accumulates rather than fragmenting (R2).
        self._summary_head = ""

        # In-memory recent messages (Layer 1)
        self.recent_messages: List[Message] = []

        # Connection pool + embedder identity.
        self._pool = None
        self._pool_min_size = pool_min_size
        self._pool_max_size = pool_max_size
        self._pool_factory = pool_factory
        self._db_available = False

        # Embedder identity. Tracked so every row carries the embedder_id +
        # embedding_dim that produced it -- enabling per-embedder partial
        # HNSW indexes and zero-downtime model swaps.
        self.embedder_id = embedding_model
        self.embedder_dim = DEFAULT_EMBEDDER_DIM  # updated on _init_embeddings
        self.embedder = None
        self._embeddings_available = False

        # Initialize
        self._init_database()
        if self.enable_embeddings:
            self._init_embeddings(embedding_model)
        else:
            print("Memory embeddings: disabled (faster smart-save mode).")
        self._load_recent_messages()

    # --- lifecycle ---------------------------------------------------------

    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        timestamp = datetime.now().isoformat()
        return hashlib.sha256(timestamp.encode()).hexdigest()[:16]

    def _schedule_background(self, fn: Callable[[], None]) -> None:
        """Run ``fn`` off the caller's thread (R2 -- never on the bus thread).

        Prefers the ``MemoryWriter`` background worker so producer jobs share
        the existing off-hot-path machinery; when no writer exists (no-DB /
        smart-save off) it falls back to a one-off daemon thread. Either way the
        summarizer/profile LLM call NEVER runs synchronously inside
        ``add_message``/``queue_user_utterance`` on the single bus thread."""
        if self._writer is not None and self._writer.schedule_job(fn):
            return
        worker = threading.Thread(target=self._run_background_job, args=(fn,), daemon=True)
        worker.start()

    @staticmethod
    def _run_background_job(fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - background jobs must not crash the thread
            print(f"[warn] Memory background job failed: {exc}")

    def _init_database(self):
        """Open the connection pool + ensure schema exists (demo/dev path)."""
        # An injected ``pool_factory`` is a DI seam (tests / a custom pool) and
        # must work even without the optional psycopg driver -- only fall back to
        # in-memory when there's no driver AND no factory.
        if not POSTGRES_AVAILABLE and self._pool_factory is None:
            print("[warn] psycopg + psycopg_pool not available. Memory will be in-memory only.")
            print("   Install with: pip install 'psycopg[binary,pool]'")
            return

        try:
            if self._pool_factory is not None:
                self._pool = self._pool_factory(
                    conninfo=self.db_url,
                    min_size=self._pool_min_size,
                    max_size=self._pool_max_size,
                    kwargs={"autocommit": True},
                )
            else:
                self._pool = ConnectionPool(
                    conninfo=self.db_url,
                    min_size=self._pool_min_size,
                    max_size=self._pool_max_size,
                    kwargs={"autocommit": True},
                    open=True,
                )
            # First-connect schema ensure -- production should run
            # ``python tools/migrate.py apply`` instead; this is for the
            # ``python utils/memory.py`` demo path.
            self._ensure_demo_schema()

            self._db_available = True
            if self.smart_save and self._writer_config.enabled:
                self._writer = MemoryWriter(
                    config=self._writer_config,
                    persist_fn=self._persist_user_message,
                    text_cleaner=self._text_cleaner,
                )
            print(f"[ok] Database connected (session: {self.session_id[:8]}...)")
        except Exception as e:
            print(f"[warn] Database connection failed: {e}")
            print("   Memory will be in-memory only for this session.")
            self._pool = None

    def _ensure_demo_schema(self):
        """Idempotent schema bootstrap for the demo/dev path."""
        safe = re.sub(r"[^a-zA-Z0-9]+", "_", self.embedder_id).strip("_").lower()
        hnsw_sql = _DEMO_HNSW_SQL.format(embedder_safe=safe, embedder_id=self.embedder_id)
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(_DEMO_SCHEMA_SQL)
                try:
                    cur.execute(_DEMO_CONSTRAINTS_SQL)
                except Exception as exc:
                    # On a legacy DB this can fail if existing rows violate
                    # the new CHECK; the right path is to run the yoyo
                    # 003_backfill_dim_embedder migration first. Log and
                    # continue -- queries still work, just without constraints.
                    print(f"[warn] constraints not added: {exc}; run "
                          "`python tools/migrate.py apply` to backfill.")
                try:
                    cur.execute(hnsw_sql)
                except Exception as exc:
                    # Index DDL is the most likely to fail on older pgvector
                    # versions (HNSW arrived in pgvector 0.5). Don't crash;
                    # search will fall back to a seq scan.
                    print(f"[warn] HNSW index DDL skipped: {exc}")

    def _init_embeddings(self, model_name: str):
        """Initialize the embedding model."""
        if not EMBEDDINGS_AVAILABLE:
            print("[warn] sentence-transformers not available. Semantic search disabled.")
            print("   Install with: pip install sentence-transformers")
            return

        try:
            print(f"[..] Loading embedding model: {model_name}...")
            self.embedder = SentenceTransformer(model_name)
            # Get the real dim from the model so we don't shadow-assume 384.
            try:
                self.embedder_dim = int(self.embedder.get_sentence_embedding_dimension())
            except Exception:
                self.embedder_dim = DEFAULT_EMBEDDER_DIM
            self._embeddings_available = True
            print(f"[ok] Embedding model loaded (dim={self.embedder_dim})!")
        except Exception as e:
            print(f"[warn] Failed to load embedding model: {e}")

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding vector for text. Returns ``None`` on failure so
        downstream INSERTs go through the NULL-embedding branch (no
        embedder_id stored, no CHECK violation)."""
        if not self._embeddings_available or not self.embedder:
            return None
        try:
            return self.embedder.encode(text, convert_to_numpy=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] Embedding failed: {exc}")
            return None

    def _load_recent_messages(self):
        """Load recent messages from database (Layer 1 warm start)."""
        if not self._db_available:
            return
        try:
            with self._pool.connection() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(
                        """
                        SELECT role, content, timestamp
                        FROM messages
                        WHERE session_id = %s AND role = 'user'
                        ORDER BY COALESCE(saved_at, timestamp) DESC
                        LIMIT %s
                        """,
                        (self.session_id, self.max_recent_messages),
                    )
                    rows = cur.fetchall()
            self.recent_messages = [
                Message(role=row['role'], content=row['content'], timestamp=row['timestamp'])
                for row in reversed(rows)  # oldest first
            ]
        except Exception as e:
            print(f"[warn] Failed to load recent messages: {e}")

    # --- message ingestion (the writer thread call site) -------------------

    def add_message(self, role: str, content: str, *, persist: bool = True) -> Optional[Message]:
        """Add a message to in-session memory.

        User speech is queued for debounced PostgreSQL persistence; assistant
        replies stay in RAM for conversation context only."""
        cleaned = self._basic_clean_text(content)
        if not cleaned:
            return None
        if role == "user" and self._is_obvious_junk(cleaned):
            return None

        message = Message(role=role, content=cleaned)
        self.recent_messages.append(message)
        if len(self.recent_messages) > self.max_recent_messages:
            self.recent_messages = self.recent_messages[-self.max_recent_messages:]

        if role == "assistant":
            self._last_assistant_text = cleaned
        elif (
            persist
            and role == "user"
            and self._db_available
            and role in self.persist_roles
        ):
            self._queue_user_for_persist(cleaned, raw_text=content.strip())

        self._check_and_summarize()
        return message

    def queue_user_utterance(
        self, text: str, *, source: str = "user_final", confidence: float = 1.0,
    ) -> bool:
        """Queue user speech; also updates short-term history when substantive."""
        msg = self.add_message("user", text, persist=False)
        if msg is None:
            return False
        return self._queue_user_for_persist(
            msg.content, raw_text=text.strip(), source=source, confidence=confidence,
        )

    def _queue_user_for_persist(
        self, cleaned: str, *, raw_text: str,
        source: str = "user_final", confidence: float = 1.0,
    ) -> bool:
        if not self._db_available or "user" not in self.persist_roles:
            return False
        # Profile producer (R8): default-off, Postgres-only. The regex match runs
        # inline (cheap); the profile DB write is scheduled off the bus thread
        # inside _extract_profile so it never inflates TTFT.
        if self.profile_enabled:
            self._extract_profile(cleaned)
        if self.smart_save and self._writer:
            return self._writer.enqueue(
                raw_text,
                source=source,
                confidence=confidence,
                last_assistant_text=self._last_assistant_text,
            )
        if not self._is_user_memory_worthy(cleaned):
            return False
        message = Message(role="user", content=cleaned)
        embedding = self._get_embedding(cleaned)
        self._save_message_to_db(
            message, embedding,
            raw_text=raw_text, cleaned_text=cleaned,
            source=source, confidence=confidence,
        )
        return True

    def add_observation(self, text: str) -> bool:
        """Persist a VISUAL (screen) observation as a recallable memory row.

        The dedicated ingest path for visual memory: stores the caption+OCR trace
        with ``role='observation'`` / ``source='vision'`` so it is (a) retrievable
        by :meth:`search_memory` alongside user messages, (b) NEVER routed through
        ``queue_user_utterance``/``_extract_profile`` (so OCR'd screen text can't
        spawn bogus ``user_profile`` rows), and (c) excluded from the recent-
        conversation block (its tag is not user/assistant). Called from the visual
        memorizer's BACKGROUND worker, never the bus thread. Returns True on write."""
        cleaned = (text or "").strip()
        if not cleaned or not self._db_available:
            return False
        message = Message(role="observation", content=cleaned)
        embedding = self._get_embedding(cleaned)
        self._save_message_to_db(
            message, embedding, raw_text=cleaned, cleaned_text=cleaned,
            source="vision", confidence=1.0,
        )
        return True

    def clear_observations(self) -> int:
        """Purge all persisted visual observations (owner privacy control, §9.7)."""
        if not self._db_available:
            return 0
        try:
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM messages WHERE source = %s", ("vision",))
                    return cur.rowcount or 0
        except Exception as e:
            print(f"[warn] Failed to clear observations: {e}")
            return 0

    def set_text_cleaner(self, cleaner):
        """Optional hook: overrides built-in Ollama cleanup when set."""
        self._text_cleaner = cleaner
        if self._writer:
            self._writer.set_text_cleaner(cleaner)

    def _persist_user_message(
        self, *, raw_text: str, cleaned_text: str,
        source: str, confidence: float, captured_at: datetime, reason: str = "",
    ) -> None:
        if not self._is_user_memory_worthy(cleaned_text):
            return
        message = Message(role="user", content=cleaned_text, timestamp=captured_at)
        embedding = self._get_embedding(cleaned_text)
        self._save_message_to_db(
            message, embedding,
            raw_text=raw_text, cleaned_text=cleaned_text,
            source=source, confidence=confidence,
        )

    def _is_obvious_junk(self, text: str) -> bool:
        return is_junk_stt_text(text)

    @staticmethod
    def _basic_clean_text(content: str) -> str:
        text = (content or "").strip()
        if not text:
            return ""
        text = re.sub(r"\s+", " ", text)
        text = text.replace(">>", "").replace("<<", "")
        text = re.sub(r"\s+([,.!?])", r"\1", text)
        text = re.sub(r"([.!?]){2,}", r"\1", text)
        text = text.strip(" -")
        return text.strip()

    def _is_user_memory_worthy(self, text: str) -> bool:
        normalized = re.sub(r"[^a-z0-9\s]", " ", text.lower())
        normalized = re.sub(r"\s+", " ", normalized).strip()
        if len(normalized.split()) < self.min_user_words:
            return False
        junk_markers = (
            "thank you thank you", "thanks for watching", "subscribe", "hario",
            "blank audio", "birds chirping", "music",
        )
        if any(marker in normalized for marker in junk_markers):
            return False
        filler_only = {
            "thank you very much", "i think ill be right back",
            "i am saying something", "im saying something",
            "i was saying something right",
        }
        if normalized in filler_only:
            return False
        unique_words = set(normalized.split())
        if len(unique_words) <= 2 and len(normalized.split()) >= 5:
            return False
        return True

    def flush_pending(self) -> int:
        """Force flush buffered user utterances to PostgreSQL."""
        if self._writer:
            saved = self._writer.flush(force=True)
            if saved:
                print(f"Memory: flushed {saved} smart-saved message(s) to Postgres")
            return saved
        return 0

    # --- persistence -------------------------------------------------------

    def _check_embedding_dim(self, embedding: Optional[np.ndarray]) -> None:
        """Reject embeddings whose dimension doesn't match the loaded model.

        Surfaces a clear Python-side error before the DB CHECK constraint
        would fire -- saves the operator chasing a ``CheckViolation`` from
        an opaque pgvector message."""
        if embedding is None:
            return
        actual = int(np.asarray(embedding).shape[-1])
        if actual != self.embedder_dim:
            raise ValueError(
                f"Embedding dimension mismatch: got {actual}, expected "
                f"{self.embedder_dim} for embedder {self.embedder_id!r}"
            )

    def _save_message_to_db(
        self, message: Message, embedding: Optional[np.ndarray], *,
        raw_text: Optional[str] = None, cleaned_text: Optional[str] = None,
        source: str = "user_final", confidence: float = 1.0,
    ):
        """Save a single message via a short-lived pool connection."""
        self._check_embedding_dim(embedding)
        saved_at = datetime.now()
        content = cleaned_text or message.content
        raw = raw_text if raw_text is not None else message.content
        try:
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    if embedding is not None:
                        cur.execute(
                            """
                            INSERT INTO messages (
                                session_id, role, content, timestamp,
                                embedding, embedding_dim, embedder_id,
                                raw_text, cleaned_text, source, confidence, saved_at
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                self.session_id, message.role, content, message.timestamp,
                                embedding.tolist(), int(embedding.shape[-1]), self.embedder_id,
                                raw, content, source, confidence, saved_at,
                            ),
                        )
                    else:
                        cur.execute(
                            """
                            INSERT INTO messages (
                                session_id, role, content, timestamp,
                                raw_text, cleaned_text, source, confidence, saved_at
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                self.session_id, message.role, content, message.timestamp,
                                raw, content, source, confidence, saved_at,
                            ),
                        )
        except Exception as e:
            print(f"[warn] Failed to save message: {e}")

    def _estimate_tokens(self, text: str) -> int:
        """Rough estimate of token count (words * 1.3)."""
        return int(len(text.split()) * 1.3)

    def _check_and_summarize(self):
        """TRIGGER a rolling summary when context grows too long (R2).

        Runs on the single bus thread (via ``add_message`` ->
        ``_handle_task_completed``), so it MUST return promptly: it only
        snapshots the older half, trims ``recent_messages`` synchronously
        (cheap, no I/O), and hands the LLM summarize + DB persist to the
        background writer thread. The summarizer call itself NEVER runs here."""
        total_tokens = sum(self._estimate_tokens(m.content) for m in self.recent_messages)
        if total_tokens <= self.max_context_tokens or len(self.recent_messages) <= 10:
            return

        # One summary in flight at a time -- avoid scheduling a second job while
        # the first is still folding/persisting on the background thread.
        with self._summary_lock:
            if self._summary_in_flight:
                return
            self._summary_in_flight = True

        split = len(self.recent_messages) // 2
        messages_to_summarize = self.recent_messages[:split]
        # Trim synchronously so the working window shrinks immediately; the LLM
        # call + persist happen off-thread against this snapshot.
        self.recent_messages = self.recent_messages[split:]
        self._schedule_background(lambda: self._create_summary(messages_to_summarize))

    def _create_summary(self, messages_to_summarize: List["Message"]):
        """Roll the older messages into the accumulating summary (R2).

        Runs on the ``MemoryWriter`` background thread (scheduled by
        ``_check_and_summarize``), never on the bus thread. Uses the injected
        fast-tier ``self._summarizer`` and FOLDS the prior summary head in so
        the layer-2 record accumulates (rolling, not fragmented); falls back to
        the legacy keyword/topic body when no summarizer is wired. Persists via
        the unchanged ``_save_summary_to_db``."""
        try:
            if not messages_to_summarize:
                return
            conversation_text = "\n".join(
                f"{m.role}: {m.content}" for m in messages_to_summarize
            )
            # Guard the rolling-head read/write with the lock (never held across
            # the LLM call) so a concurrent apply_retention fold can't interleave.
            with self._summary_lock:
                prior = self._summary_head
            summary_text = self._summarize_text(prior, conversation_text)
            with self._summary_lock:
                self._summary_head = summary_text
            topics = self._extract_topics(conversation_text)
            summary = ConversationSummary(
                summary=summary_text,
                topics=topics, user_preferences=[],
                start_time=messages_to_summarize[0].timestamp,
                end_time=messages_to_summarize[-1].timestamp,
                message_count=len(messages_to_summarize),
            )
            if self._db_available:
                self._save_summary_to_db(summary)
            print(f"[mem] Created summary of {summary.message_count} messages")
        finally:
            with self._summary_lock:
                self._summary_in_flight = False

    def _summarize_text(self, prior: str, conversation_text: str) -> str:
        """Fold the prior summary head + new turns into one rolling summary.

        No lock is held across this call -- it may invoke the fast LLM. When no
        summarizer is injected, falls back to the legacy keyword/topic body so
        the layer-2 path still exercises end-to-end."""
        if self._summarizer is not None:
            parts = []
            if prior:
                parts.append(f"Summary so far:\n{prior}")
            parts.append(f"New conversation:\n{conversation_text}")
            try:
                rolled = self._summarizer("\n\n".join(parts))
                if rolled and rolled.strip():
                    return rolled.strip()
            except Exception as exc:  # noqa: BLE001 - fall back, never crash the thread
                print(f"[warn] Summarizer failed, using keyword fallback: {exc}")
        # Keyword fallback (R2): legacy topic-frequency body, folded onto prior.
        topics = self._extract_topics(conversation_text)
        new_body = (
            f"Conversation about: {', '.join(topics[:5])}" if topics
            else "Conversation segment."
        )
        return f"{prior}\n{new_body}".strip() if prior else new_body

    def _extract_topics(self, text: str) -> List[str]:
        """Simple keyword extraction (placeholder for the LLM summary path)."""
        words = text.lower().split()
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'i', 'you', 'he', 'she', 'it',
            'we', 'they', 'my', 'your', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'as', 'by', 'about', 'what', 'how', 'when', 'where',
            'why', 'who', 'this', 'that', 'these', 'those', 'can', 'could', 'would',
            'should', 'will', 'have', 'has', 'had', 'do', 'does', 'did', 'be', 'been',
            'being', 'user:', 'assistant:',
        }
        filtered = [w for w in words if w not in stopwords and len(w) > 3]
        freq: dict[str, int] = {}
        for w in filtered:
            freq[w] = freq.get(w, 0) + 1
        sorted_topics = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [t[0] for t in sorted_topics[:10]]

    def _save_summary_to_db(self, summary: ConversationSummary):
        """Save summary via a short-lived pool connection."""
        try:
            embedding = self._get_embedding(summary.summary)
            self._check_embedding_dim(embedding)
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    if embedding is not None:
                        cur.execute(
                            """
                            INSERT INTO summaries
                            (session_id, summary, topics, user_preferences,
                             start_time, end_time, message_count,
                             embedding, embedding_dim, embedder_id)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                self.session_id, summary.summary,
                                summary.topics, summary.user_preferences,
                                summary.start_time, summary.end_time, summary.message_count,
                                embedding.tolist(), int(embedding.shape[-1]), self.embedder_id,
                            ),
                        )
                    else:
                        cur.execute(
                            """
                            INSERT INTO summaries
                            (session_id, summary, topics, user_preferences,
                             start_time, end_time, message_count)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                self.session_id, summary.summary,
                                summary.topics, summary.user_preferences,
                                summary.start_time, summary.end_time, summary.message_count,
                            ),
                        )
        except Exception as e:
            print(f"[warn] Failed to save summary: {e}")

    # --- retrieval ---------------------------------------------------------

    def search_memory(
        self, query: str, limit: int = 5, *, ef_search: int = 40,
    ) -> List[Dict[str, Any]]:
        """Search memory using semantic similarity.

        Uses ``SET LOCAL hnsw.ef_search`` for per-query recall tuning;
        defaults to 40 (pgvector default). Filters on ``embedder_id`` so
        vectors from different models are never compared."""
        if not self._db_available or not self._embeddings_available:
            return []

        query_embedding = self._get_embedding(query)
        if query_embedding is None:
            return []
        self._check_embedding_dim(query_embedding)

        results: list[dict] = []
        try:
            with self._pool.connection() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(f"SET LOCAL hnsw.ef_search = {int(ef_search)}")
                    # Messages
                    cur.execute(
                        """
                        SELECT role, content, timestamp,
                               1 - (embedding <=> %s::vector) AS similarity
                        FROM messages
                        WHERE embedding IS NOT NULL
                          AND embedder_id = %s
                          AND embedding_dim = %s
                          AND role = 'user'
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (
                            query_embedding.tolist(),
                            self.embedder_id, int(query_embedding.shape[-1]),
                            query_embedding.tolist(), limit,
                        ),
                    )
                    for row in cur.fetchall():
                        results.append({
                            'type': 'message',
                            'role': row['role'],
                            'content': row['content'],
                            'timestamp': row['timestamp'],
                            'similarity': float(row['similarity'] or 0.0),
                        })
                    # Summaries
                    cur.execute(
                        """
                        SELECT summary, topics, start_time, end_time,
                               1 - (embedding <=> %s::vector) AS similarity
                        FROM summaries
                        WHERE embedding IS NOT NULL
                          AND embedder_id = %s
                          AND embedding_dim = %s
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (
                            query_embedding.tolist(),
                            self.embedder_id, int(query_embedding.shape[-1]),
                            query_embedding.tolist(), limit,
                        ),
                    )
                    for row in cur.fetchall():
                        results.append({
                            'type': 'summary',
                            'content': row['summary'],
                            'topics': row['topics'],
                            'start_time': row['start_time'],
                            'end_time': row['end_time'],
                            'similarity': float(row['similarity'] or 0.0),
                        })
        except Exception as e:
            print(f"[warn] Search failed: {e}")

        results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        return results[:limit]

    def _search_observations(self, query: str, limit: int, *, ef_search: int = 40) -> List[Dict[str, Any]]:
        """Semantic search over VISUAL (screen) observations only.

        Kept SEPARATE from :meth:`search_memory` (which is user-conversation +
        summaries) so visual memories get their OWN bounded recall sub-pass and
        can never crowd user-message recall out of a single shared pool, and so
        the generic ``Memory.search`` seam never surfaces screen rows."""
        if not self._db_available or not self._embeddings_available:
            return []
        q = self._get_embedding(query)
        if q is None:
            return []
        self._check_embedding_dim(q)
        out: list[dict] = []
        try:
            with self._pool.connection() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(f"SET LOCAL hnsw.ef_search = {int(ef_search)}")
                    cur.execute(
                        """
                        SELECT content, timestamp,
                               1 - (embedding <=> %s::vector) AS similarity
                        FROM messages
                        WHERE embedding IS NOT NULL
                          AND embedder_id = %s
                          AND embedding_dim = %s
                          AND role = 'observation'
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (q.tolist(), self.embedder_id, int(q.shape[-1]), q.tolist(), limit),
                    )
                    for row in cur.fetchall():
                        out.append({
                            'type': 'vision',
                            'content': row['content'],
                            'timestamp': row['timestamp'],
                            'similarity': float(row['similarity'] or 0.0),
                        })
        except Exception as e:
            print(f"[warn] Observation search failed: {e}")
        return out

    def get_context_for_llm(self, current_query: str = None) -> str:
        """Formatted context string for the LLM (token-budgeted, deduped).

        THREE independent, separately-budgeted blocks (concatenated), each run
        through the shared :func:`build_block` (adaptive cutoff + dedup + token
        budget + compression) so their score scales never mix in one ranked list:

        1. **Recall** -- conversation hits (user messages + summaries) gated on a
           query + live embeddings + DB.
        2. **Vision** -- persisted visual (screen) observations, fetched by a
           SEPARATE query so they never crowd conversation recall out of a shared
           pool, with their OWN reserved floor (and vice versa).
        3. **Profile** -- gated ONLY on ``profile_enabled`` (default OFF),
           independent of DB/embeddings.

        All three SHARE one ``max_tokens`` budget so their concatenation cannot
        exceed it: each present source gets a reserved floor (so none can be
        evicted by another), and unused budget flows down to the next pass.
        Combined block tokens <= the single ``recall_budget.max_tokens``."""
        budget = self._recall_budget
        total = budget.max_tokens
        cpt = budget.chars_per_token
        have_db = bool(current_query and self._embeddings_available and self._db_available)

        # Fetch each source up front so a floor is reserved ONLY for a source that
        # is present (absent source -> no reservation -> its share flows to others).
        conv_rows = self.search_memory(current_query, limit=self._recall_pool) if have_db else []
        vis_rows = self._search_observations(current_query, limit=self._recall_pool) if have_db else []
        profile = self.get_user_profile() if self.profile_enabled else {}

        prof_floor = (total // 4) if profile else 0      # durable facts: >= a quarter
        vis_floor = (total // 4) if vis_rows else 0       # screen memory: >= a quarter
        recall_cap = max(total - prof_floor - vis_floor, 0)

        parts: list[str] = []
        used = 0

        def _emit(cands: list, cap: int, header: Optional[str] = None) -> None:
            nonlocal used
            if cap <= 0 or not cands:
                return
            b = _dc_replace(budget, max_tokens=cap, header=header) if header \
                else _dc_replace(budget, max_tokens=cap)
            block = build_block(cands, current_query or "", b)
            if block:
                parts.append(block)
                used += estimate_tokens(block, cpt)

        # 1. Conversation recall (messages + summaries).
        conv_cands: list[Candidate] = []
        for r in conv_rows:
            if r.get('type') == 'summary':
                span = (_to_epoch(r.get('start_time')), _to_epoch(r.get('end_time')))
                conv_cands.append(Candidate(
                    str(r.get('content', '')), _finite(r.get('similarity')),
                    kind='summary', timestamp=_to_epoch(r.get('end_time')), span=span,
                ))
            else:
                conv_cands.append(Candidate(
                    str(r.get('content', '')), _finite(r.get('similarity')),
                    kind='message', role=r.get('role'), timestamp=_to_epoch(r.get('timestamp')),
                ))
        _emit(conv_cands, recall_cap)

        # 2. Vision (screen) memory -- reserved floor + whatever recall left unused.
        vis_cands = [
            Candidate(str(r.get('content', '')), _finite(r.get('similarity')),
                      kind='vision', timestamp=_to_epoch(r.get('timestamp')))
            for r in vis_rows
        ]
        _emit(vis_cands, vis_floor + max(recall_cap - used, 0), header="=== Screen Memory ===")

        # 3. Profile -- claims whatever the prior passes left (>= prof_floor).
        if profile:
            prof_cands = [Candidate(f"{k}: {v}", _PROFILE_RECALL_SCORE, kind='profile') for k, v in profile.items()]
            _emit(prof_cands, max(total - used, 0), header="=== User Profile ===")

        return "\n\n".join(parts)

    def get_chat_history(self) -> List[Dict[str, str]]:
        """Get recent messages as list of dicts for LLM."""
        return [{'role': m.role, 'content': m.content} for m in self.recent_messages]

    def update_user_profile(self, key: str, value: str, confidence: float = 1.0):
        """Update a user profile entry."""
        if not self._db_available:
            return
        try:
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO user_profile (key, value, confidence, updated_at)
                        VALUES (%s, %s, %s, NOW())
                        ON CONFLICT (key) DO UPDATE SET
                            value = EXCLUDED.value,
                            confidence = EXCLUDED.confidence,
                            updated_at = NOW()
                        """,
                        (key, value, confidence),
                    )
        except Exception as e:
            print(f"[warn] Failed to update profile: {e}")

    def _extract_profile(self, text: str) -> None:
        """Ingest-time user-profile producer (R8): default-off, Postgres-only,
        confidence-floored.

        A deterministic regex pass over high-signal phrases ('my name is X',
        'call me Z', 'I live in Y', 'I prefer ...') writes durable
        ``user_profile`` rows at a confidence FLOOR of ``_PROFILE_CONFIDENCE``
        (>= 0.9). The regex match is cheap and runs inline, but the matched rows
        are a Postgres write and this path is reached from the answer/task thread
        (the assistant capability ingests the query before ``model.stream``), so
        the write is scheduled OFF-thread to avoid inflating TTFT."""
        if not self.profile_enabled or not self._db_available or not text:
            return
        seen: set[str] = set()
        matches: list[tuple[str, str]] = []
        for key, pattern in _PROFILE_PATTERNS:
            match = pattern.search(text)
            if not match:
                continue
            value = match.group("v").strip(" -\"'")
            if not value or key in seen:
                continue
            seen.add(key)
            matches.append((key, value))
        if matches:
            def _write_profile() -> None:
                for k, v in matches:
                    self.update_user_profile(k, v, confidence=_PROFILE_CONFIDENCE)
            self._schedule_background(_write_profile)
        # An optional fuzzy LLM extraction would be scheduled here via
        # ``self._schedule_background(...)`` so it runs off the bus thread; the
        # deterministic pass above is the only confidence-floored writer this
        # cycle.

    def get_user_profile(self) -> Dict[str, str]:
        """Get all user profile entries."""
        if not self._db_available:
            return {}
        try:
            with self._pool.connection() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(
                        "SELECT key, value FROM user_profile ORDER BY confidence DESC"
                    )
                    return {row['key']: row['value'] for row in cur.fetchall()}
        except Exception as e:
            print(f"[warn] Failed to get profile: {e}")
            return {}

    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get statistics about stored conversations."""
        stats = {
            'session_id': self.session_id,
            'recent_messages': len(self.recent_messages),
            'pending_db_messages': self._writer.pending_count if self._writer else 0,
            'db_available': self._db_available,
            'embeddings_available': self._embeddings_available,
            'smart_save': self.smart_save,
            'persist_roles': self.persist_roles,
        }
        if self._db_available:
            try:
                with self._pool.connection() as conn:
                    with conn.cursor(row_factory=dict_row) as cur:
                        cur.execute("SELECT COUNT(*) AS count FROM messages")
                        stats['total_messages'] = cur.fetchone()['count']
                        cur.execute("SELECT COUNT(*) AS count FROM summaries")
                        stats['total_summaries'] = cur.fetchone()['count']
                        cur.execute("SELECT COUNT(*) AS count FROM user_profile")
                        stats['profile_entries'] = cur.fetchone()['count']
            except Exception:
                pass
        return stats

    def apply_retention(self, now=None) -> int:
        """Age-TTL retention pass (R6/§6); returns the number of rows removed.

        - No-op (returns 0) without a live DB (guarded by ``_db_available``).
        - Episodic ``messages`` older than ``episodic_ttl_days`` are
          SUMMARIZED-THEN-EVICTED: their text is folded into the rolling summary
          (persisted to ``summaries``) before the rows are deleted, so recall
          keeps a condensed trace instead of losing the history outright.
        - ``summaries`` older than ``summary_ttl_days`` are deleted.
        - ``user_profile`` is durable and NEVER TTL'd.

        ``now`` is injectable for tests; defaults to ``datetime.now()``. A TTL of
        0 disables that tier's eviction (keep-forever)."""
        if not self._db_available:
            return 0
        now = now or datetime.now()
        removed = 0
        try:
            with self._pool.connection() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    if self.episodic_ttl_days > 0:
                        # Summarize-then-evict: read the expiring episodic rows,
                        # fold them into the rolling summary, then DELETE.
                        cur.execute(
                            """
                            SELECT role, content, timestamp
                            FROM messages
                            WHERE COALESCE(saved_at, timestamp)
                                  < %s - (%s || ' days')::interval
                            ORDER BY COALESCE(saved_at, timestamp)
                            """,
                            (now, str(self.episodic_ttl_days)),
                        )
                        expiring = cur.fetchall()
                        if expiring:
                            conversation_text = "\n".join(
                                f"{r['role']}: {r['content']}" for r in expiring
                            )
                            with self._summary_lock:
                                prior_head = self._summary_head
                            summary_text = self._summarize_text(
                                prior_head, conversation_text
                            )
                            with self._summary_lock:
                                self._summary_head = summary_text
                            # NOTE: _save_summary_to_db acquires its own pool
                            # connection while this outer one is held -> requires
                            # pool_max_size >= 2 (default 5). A future polish could
                            # de-nest by persisting outside this connection block.
                            self._save_summary_to_db(
                                ConversationSummary(
                                    summary=summary_text,
                                    topics=self._extract_topics(conversation_text),
                                    user_preferences=[],
                                    start_time=expiring[0]["timestamp"],
                                    end_time=expiring[-1]["timestamp"],
                                    message_count=len(expiring),
                                )
                            )
                            cur.execute(
                                """
                                DELETE FROM messages
                                WHERE COALESCE(saved_at, timestamp)
                                      < %s - (%s || ' days')::interval
                                """,
                                (now, str(self.episodic_ttl_days)),
                            )
                            removed += cur.rowcount or 0
                    if self.summary_ttl_days > 0:
                        cur.execute(
                            """
                            DELETE FROM summaries
                            WHERE created_at < %s - (%s || ' days')::interval
                            """,
                            (now, str(self.summary_ttl_days)),
                        )
                        removed += cur.rowcount or 0
        except Exception as e:
            print(f"[warn] Retention pass failed: {e}")
        return removed

    def clear_session(self):
        """Clear current session's messages (but keep summaries)."""
        self.recent_messages = []
        if self._db_available:
            try:
                with self._pool.connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "DELETE FROM messages WHERE session_id = %s",
                            (self.session_id,),
                        )
            except Exception as e:
                print(f"[warn] Failed to clear session: {e}")

    def close(self):
        """Flush pending user memory and close the pool."""
        try:
            if self._writer:
                saved = self._writer.close()
                if saved:
                    print(f"Memory: flushed {saved} user utterance(s) on shutdown")
            else:
                self.flush_pending()
        except Exception as e:
            print(f"[warn] Failed to flush pending memory: {e}")
        self._writer = None
        if self._pool is not None:
            try:
                self._pool.close()
            except Exception:
                pass
            self._pool = None


def create_memory_manager(**kwargs) -> MemoryManager:
    """Create a memory manager instance."""
    return MemoryManager(**kwargs)


# Test
if __name__ == "__main__":
    print("Testing Memory Manager...")
    print(f"PostgreSQL available: {POSTGRES_AVAILABLE}")
    print(f"Embeddings available: {EMBEDDINGS_AVAILABLE}")

    memory = MemoryManager(db_url="postgresql://localhost/nonexistent")
    memory.add_message("user", "Hello, my name is John")
    memory.add_message("assistant", "Nice to meet you, John!")
    memory.add_message("user", "I like programming in Python")

    print(f"\nRecent messages: {len(memory.recent_messages)}")
    print(f"\nContext for LLM:")
    print(memory.get_context_for_llm("What's my name?"))
    print(f"\nStats: {memory.get_conversation_stats()}")
