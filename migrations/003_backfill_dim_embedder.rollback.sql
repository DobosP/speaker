-- Rollback for 003 -- not strictly invertible (we'd lose embedder identity
-- without a separate audit trail). Best-effort: NULL out both columns so
-- the schema returns to pre-3 state. Run only with intent.

-- depends: 001_init

UPDATE messages
SET embedding_dim = NULL,
    embedder_id   = NULL;

UPDATE summaries
SET embedding_dim = NULL,
    embedder_id   = NULL;
