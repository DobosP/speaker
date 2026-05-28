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
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from utils.memory_config import MemoryWriterConfig, config_from_dict
from utils.memory_writer import MemoryWriter, is_junk_stt_text

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

    def _init_database(self):
        """Open the connection pool + ensure schema exists (demo/dev path)."""
        if not POSTGRES_AVAILABLE:
            print("⚠️  psycopg + psycopg_pool not available. Memory will be in-memory only.")
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
            print(f"✅ Database connected (session: {self.session_id[:8]}...)")
        except Exception as e:
            print(f"⚠️  Database connection failed: {e}")
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
                    print(f"⚠️  constraints not added: {exc}; run "
                          "`python tools/migrate.py apply` to backfill.")
                try:
                    cur.execute(hnsw_sql)
                except Exception as exc:
                    # Index DDL is the most likely to fail on older pgvector
                    # versions (HNSW arrived in pgvector 0.5). Don't crash;
                    # search will fall back to a seq scan.
                    print(f"⚠️  HNSW index DDL skipped: {exc}")

    def _init_embeddings(self, model_name: str):
        """Initialize the embedding model."""
        if not EMBEDDINGS_AVAILABLE:
            print("⚠️  sentence-transformers not available. Semantic search disabled.")
            print("   Install with: pip install sentence-transformers")
            return

        try:
            print(f"🔄 Loading embedding model: {model_name}...")
            self.embedder = SentenceTransformer(model_name)
            # Get the real dim from the model so we don't shadow-assume 384.
            try:
                self.embedder_dim = int(self.embedder.get_sentence_embedding_dimension())
            except Exception:
                self.embedder_dim = DEFAULT_EMBEDDER_DIM
            self._embeddings_available = True
            print(f"✅ Embedding model loaded (dim={self.embedder_dim})!")
        except Exception as e:
            print(f"⚠️  Failed to load embedding model: {e}")

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding vector for text. Returns ``None`` on failure so
        downstream INSERTs go through the NULL-embedding branch (no
        embedder_id stored, no CHECK violation)."""
        if not self._embeddings_available or not self.embedder:
            return None
        try:
            return self.embedder.encode(text, convert_to_numpy=True)
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️  Embedding failed: {exc}")
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
            print(f"⚠️  Failed to load recent messages: {e}")

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
            print(f"⚠️  Failed to save message: {e}")

    def _estimate_tokens(self, text: str) -> int:
        """Rough estimate of token count (words * 1.3)."""
        return int(len(text.split()) * 1.3)

    def _check_and_summarize(self):
        """Check if context is too long and create summary if needed."""
        total_tokens = sum(self._estimate_tokens(m.content) for m in self.recent_messages)
        if total_tokens > self.max_context_tokens and len(self.recent_messages) > 10:
            self._create_summary()

    def _create_summary(self):
        """Create a summary of older messages.

        NOTE: still the legacy keyword-frequency 'summary' from before the
        audit -- a real LLM summary lands in Phase 4. Left here so the
        layer-2 storage path exercises end-to-end (and future-replace is
        a single function swap)."""
        if len(self.recent_messages) < 10:
            return
        messages_to_summarize = self.recent_messages[: len(self.recent_messages) // 2]
        conversation_text = "\n".join(
            f"{m.role}: {m.content}" for m in messages_to_summarize
        )
        topics = self._extract_topics(conversation_text)
        summary = ConversationSummary(
            summary=f"Conversation with {len(messages_to_summarize)} messages about: {', '.join(topics[:5])}",
            topics=topics, user_preferences=[],
            start_time=messages_to_summarize[0].timestamp,
            end_time=messages_to_summarize[-1].timestamp,
            message_count=len(messages_to_summarize),
        )
        if self._db_available:
            self._save_summary_to_db(summary)
        self.recent_messages = self.recent_messages[len(messages_to_summarize):]
        print(f"📝 Created summary of {summary.message_count} messages")

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
            print(f"⚠️  Failed to save summary: {e}")

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
            print(f"⚠️  Search failed: {e}")

        results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        return results[:limit]

    def get_context_for_llm(self, current_query: str = None) -> str:
        """Formatted context string for the LLM."""
        context_parts: list[str] = []
        if current_query and self._embeddings_available and self._db_available:
            relevant = self.search_memory(current_query, limit=5)
            high_relevance = [item for item in relevant if item.get('similarity', 0) > 0.6]
            if high_relevance:
                context_parts.append("=== Past Conversations ===")
                for item in high_relevance[:3]:
                    if item['type'] == 'message':
                        role_label = "User" if item['role'] == 'user' else "Assistant"
                        content = item['content'][:150]
                        context_parts.append(f"{role_label}: {content}")
                    elif item['type'] == 'summary':
                        context_parts.append(f"Summary: {item['content'][:150]}")
                context_parts.append("")
        profile = self.get_user_profile()
        if profile:
            context_parts.append("=== User Profile ===")
            for key, value in profile.items():
                context_parts.append(f"- {key}: {value}")
            context_parts.append("")
        return "\n".join(context_parts)

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
            print(f"⚠️  Failed to update profile: {e}")

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
            print(f"⚠️  Failed to get profile: {e}")
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
                print(f"⚠️  Failed to clear session: {e}")

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
            print(f"⚠️  Failed to flush pending memory: {e}")
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
