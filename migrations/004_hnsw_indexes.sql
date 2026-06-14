-- 004_hnsw_indexes.sql -- partial HNSW indexes per embedder.
--
-- Why HNSW (not IVFFlat): HNSW has no "training" step, so it builds
-- correctly on an empty table (the audited bug -- IVFFlat centroids on
-- empty data produce useless results). At our scale (10k-100k rows) HNSW
-- is the pgvector default.
--
-- Why partial (WHERE embedder_id = ...): adding a new embedder ships a
-- new partial index without dropping/rebuilding the old one, and the
-- WHERE clause prevents cross-model similarity queries from ever
-- evaluating.
--
-- ``cosine_ops``: sentence-transformer embeddings are typically unit-
-- normalized; cosine and inner-product give identical ranking and we
-- keep cosine as the safer default. Per-query ``SET LOCAL hnsw.ef_search``
-- is tunable in the application layer.

-- depends: 001_init

SET maintenance_work_mem = '512MB';
SET LOCAL statement_timeout = 0;

-- Drop the legacy IVFFlat index from the deleted schema if present (idempotent).
DROP INDEX IF EXISTS idx_messages_embedding;
DROP INDEX IF EXISTS idx_summaries_embedding;

-- The ``embedding`` column is the UNCONSTRAINED ``vector`` type (001 keeps it
-- multi-dim on purpose), and pgvector refuses to build an HNSW index on a
-- column with no declared dimension. This partial index is already scoped to a
-- single 384-dim embedder, so cast the indexed expression to ``vector(384)`` to
-- give the index a concrete dimension. The explicit index NAME is unchanged, so
-- the rollback's ``DROP INDEX`` still matches.
CREATE INDEX IF NOT EXISTS idx_messages_emb_minilm_l6
    ON messages USING hnsw ((embedding::vector(384)) vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
    WHERE embedder_id = 'all-MiniLM-L6-v2';

CREATE INDEX IF NOT EXISTS idx_summaries_emb_minilm_l6
    ON summaries USING hnsw ((embedding::vector(384)) vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
    WHERE embedder_id = 'all-MiniLM-L6-v2';
