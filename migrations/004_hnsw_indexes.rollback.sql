-- Rollback for 004 -- drops the HNSW partial indexes (does NOT restore
-- the legacy IVFFlat ones; those were always incorrect on an empty
-- table and we don't want to recreate the bug).

DROP INDEX IF EXISTS idx_messages_emb_minilm_l6;
DROP INDEX IF EXISTS idx_summaries_emb_minilm_l6;
