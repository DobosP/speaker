-- Rollback for 002_constraints.sql

ALTER TABLE messages  DROP CONSTRAINT IF EXISTS embedding_dim_matches;
ALTER TABLE messages  DROP CONSTRAINT IF EXISTS embedder_present_with_embedding;
ALTER TABLE summaries DROP CONSTRAINT IF EXISTS sum_embedding_dim_matches;
ALTER TABLE summaries DROP CONSTRAINT IF EXISTS sum_embedder_present_with_embedding;
