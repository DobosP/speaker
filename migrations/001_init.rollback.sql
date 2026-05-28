-- Rollback for 001_init.sql -- drops the three core tables. The pgvector
-- extension is NOT dropped (other databases on the cluster may use it).

DROP TABLE IF EXISTS user_profile;
DROP TABLE IF EXISTS summaries;
DROP TABLE IF EXISTS messages;
