-- 003_backfill_dim_embedder.sql -- legacy-DB upgrade hook.
--
-- Repos that were running the previous schema (``vector(384)`` hardcoded,
-- no ``embedding_dim`` / ``embedder_id`` columns) need their existing
-- 384-d rows backfilled with the default embedder identity before the
-- 002 CHECK constraints can be added without rejecting them.
--
-- On a fresh DB created by 001, this migration is a no-op (every row
-- already has embedder_id set by the application layer when INSERTed).

-- depends: 001_init

UPDATE messages
SET embedding_dim = vector_dims(embedding),
    embedder_id   = 'all-MiniLM-L6-v2'
WHERE embedding IS NOT NULL
  AND (embedder_id IS NULL OR embedding_dim IS NULL);

UPDATE summaries
SET embedding_dim = vector_dims(embedding),
    embedder_id   = 'all-MiniLM-L6-v2'
WHERE embedding IS NOT NULL
  AND (embedder_id IS NULL OR embedding_dim IS NULL);
