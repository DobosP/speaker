-- 002_constraints.sql -- CHECK constraints that enforce the per-row
-- embedding-dimension contract introduced in 001.
--
-- Split out from 001 because adding constraints on an empty table is
-- trivial but adding them on a populated table requires a backfill first
-- (see 003_backfill_dim_embedder.sql for the upgrade path on a legacy
-- 384-only DB).

-- depends: 001_init

ALTER TABLE messages
    ADD CONSTRAINT IF NOT EXISTS embedding_dim_matches
    CHECK (embedding IS NULL OR vector_dims(embedding) = embedding_dim);

ALTER TABLE messages
    ADD CONSTRAINT IF NOT EXISTS embedder_present_with_embedding
    CHECK ((embedding IS NULL) = (embedder_id IS NULL));

ALTER TABLE summaries
    ADD CONSTRAINT IF NOT EXISTS sum_embedding_dim_matches
    CHECK (embedding IS NULL OR vector_dims(embedding) = embedding_dim);

ALTER TABLE summaries
    ADD CONSTRAINT IF NOT EXISTS sum_embedder_present_with_embedding
    CHECK ((embedding IS NULL) = (embedder_id IS NULL));
