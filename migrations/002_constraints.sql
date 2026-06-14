-- 002_constraints.sql -- CHECK constraints that enforce the per-row
-- embedding-dimension contract introduced in 001.
--
-- Split out from 001 because adding constraints on an empty table is
-- trivial but adding them on a populated table requires a backfill first
-- (see 003_backfill_dim_embedder.sql for the upgrade path on a legacy
-- 384-only DB).

-- depends: 001_init

-- ``ALTER TABLE ... ADD CONSTRAINT`` has no ``IF NOT EXISTS`` clause in
-- PostgreSQL (unlike ``DROP CONSTRAINT IF EXISTS``) -- the previous form was a
-- syntax error that broke ``tools/migrate.py apply`` on the 002 step. For
-- idempotent re-runs we wrap each ADD in its own sub-block and swallow
-- ``duplicate_object`` -- the same pattern proven in utils/memory.py
-- (_DEMO_CONSTRAINTS_SQL). Constraint names are unchanged so the rollback's
-- ``DROP CONSTRAINT IF EXISTS`` (002_constraints.rollback.sql) still matches.
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
