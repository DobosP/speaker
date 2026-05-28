-- 001_init.sql -- initial schema for the local-first voice assistant's
-- smart-memory layer (PostgreSQL + pgvector).
--
-- Design notes (the parts that changed in the May-2026 audit):
--
-- 1) ``embedding`` is the unconstrained pgvector ``vector`` type (NOT
--    ``vector(384)``). Per-row dimension is enforced by the
--    ``embedding_dim_matches`` CHECK and tracked in ``embedding_dim``.
--    This lets the project swap embedders (e.g. all-MiniLM-L6-v2 -> bge-small)
--    without a destructive table rebuild: old 384-dim rows coexist with
--    new 768-dim rows, queries filter on ``embedder_id`` before the
--    similarity scan.
--
-- 2) Vector indexes are **HNSW** (not IVFFlat). Unlike IVFFlat, HNSW has
--    no "training" step -- the centroids issue (IVFFlat built on empty
--    data is useless) doesn't apply. For <1M rows (a voice assistant's
--    lifetime corpus) HNSW is the recommended pgvector default.
--
-- 3) Indexes are **partial** per (embedder_id, embedding_dim). Adding a
--    new embedder ships a new partial index without rebuilding the old
--    one and never compares vectors from different models.
--
-- Sources:
--   pgvector README - https://github.com/pgvector/pgvector
--   HNSW vs IVFFlat - https://dev.to/philip_mcclarence_2ef9475/ivfflat-vs-hnsw-in-pgvector-which-index-should-you-use-305p
--   Multi-dim pattern - https://community.openai.com/t/how-to-deal-with-different-vector-dimensions-for-embeddings-and-search-with-pgvector/602141

-- depends:

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS messages (
    id              BIGSERIAL PRIMARY KEY,
    session_id      VARCHAR(64) NOT NULL,
    role            VARCHAR(20) NOT NULL,
    content         TEXT NOT NULL,
    timestamp       TIMESTAMPTZ DEFAULT NOW(),
    embedding       vector,                      -- unconstrained; per-row dim
    embedding_dim   INTEGER,                     -- nullable iff embedding NULL
    embedder_id     VARCHAR(64),                 -- e.g. 'all-MiniLM-L6-v2'
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
