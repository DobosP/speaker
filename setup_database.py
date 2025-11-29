#!/usr/bin/env python3
"""
Database Setup Script for Voice Assistant Memory

This script sets up the PostgreSQL database with pgvector extension
for the voice assistant's multi-layer memory system.

Requirements:
- PostgreSQL 14+ with pgvector extension
- psycopg2-binary

Usage:
    # With default settings (localhost)
    python setup_database.py
    
    # With custom database URL
    python setup_database.py --db-url "postgresql://user:pass@host/dbname"
    
    # Create new database
    python setup_database.py --create-db --db-name voice_assistant
"""
import argparse
import sys
import os

try:
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
except ImportError:
    print("❌ psycopg2 not installed. Install with: pip install psycopg2-binary")
    sys.exit(1)


# SQL to create tables (same as in memory.py)
CREATE_TABLES_SQL = """
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Messages table (all conversations)
CREATE TABLE IF NOT EXISTS messages (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL,
    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    embedding vector(384)  -- for all-MiniLM-L6-v2
);

-- Summaries table (condensed history)
CREATE TABLE IF NOT EXISTS summaries (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(64),
    summary TEXT NOT NULL,
    topics TEXT[],
    user_preferences TEXT[],
    start_time TIMESTAMPTZ,
    end_time TIMESTAMPTZ,
    message_count INT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    embedding vector(384)
);

-- User profile (learned preferences)
CREATE TABLE IF NOT EXISTS user_profile (
    id SERIAL PRIMARY KEY,
    key VARCHAR(255) UNIQUE NOT NULL,
    value TEXT NOT NULL,
    confidence FLOAT DEFAULT 1.0,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for fast lookup
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_summaries_session ON summaries(session_id);
"""

# Vector indexes (created after data exists)
CREATE_VECTOR_INDEXES_SQL = """
-- Vector indexes for semantic search (using IVFFlat for speed)
-- Note: These indexes work best with some data already in the tables
CREATE INDEX IF NOT EXISTS idx_messages_embedding ON messages 
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_summaries_embedding ON summaries 
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
"""


def create_database(host: str, user: str, password: str, db_name: str):
    """Create a new PostgreSQL database."""
    try:
        # Connect to default postgres database
        conn = psycopg2.connect(
            host=host,
            user=user,
            password=password,
            database="postgres"
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        
        with conn.cursor() as cur:
            # Check if database exists
            cur.execute(
                "SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s",
                (db_name,)
            )
            exists = cur.fetchone()
            
            if exists:
                print(f"⚠️  Database '{db_name}' already exists")
            else:
                cur.execute(f'CREATE DATABASE "{db_name}"')
                print(f"✅ Created database: {db_name}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Failed to create database: {e}")
        return False


def setup_tables(db_url: str):
    """Create tables and indexes in the database."""
    try:
        conn = psycopg2.connect(db_url)
        conn.autocommit = True
        
        with conn.cursor() as cur:
            # Create tables
            print("📦 Creating tables...")
            cur.execute(CREATE_TABLES_SQL)
            print("✅ Tables created")
            
            # Check if pgvector is available
            cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            if cur.fetchone():
                print("✅ pgvector extension enabled")
            else:
                print("⚠️  pgvector extension not available")
                print("   Install with: CREATE EXTENSION vector;")
                print("   Or install pgvector: https://github.com/pgvector/pgvector")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Failed to setup tables: {e}")
        return False


def verify_setup(db_url: str):
    """Verify the database setup is correct."""
    try:
        conn = psycopg2.connect(db_url)
        
        with conn.cursor() as cur:
            # Check tables
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                AND table_name IN ('messages', 'summaries', 'user_profile')
            """)
            tables = [row[0] for row in cur.fetchall()]
            
            print("\n📊 Database Status:")
            print(f"   Tables found: {', '.join(tables) if tables else 'None'}")
            
            # Check message count
            cur.execute("SELECT COUNT(*) FROM messages")
            msg_count = cur.fetchone()[0]
            print(f"   Total messages: {msg_count}")
            
            # Check summary count
            cur.execute("SELECT COUNT(*) FROM summaries")
            sum_count = cur.fetchone()[0]
            print(f"   Total summaries: {sum_count}")
            
            # Check pgvector
            cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            has_vector = cur.fetchone() is not None
            print(f"   pgvector: {'✅ Enabled' if has_vector else '❌ Not installed'}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Setup PostgreSQL database for Voice Assistant memory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Peer authentication (no password)
  python setup_database.py --db-url "postgresql:///voice_assistant"
  
  # Password authentication
  python setup_database.py --db-url "postgresql://user:pass@localhost/voice_assistant"
  
  # Create new database
  python setup_database.py --create-db --db-name voice_assistant
  
  # Verify existing setup
  python setup_database.py --verify-only
        """
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=os.getenv("DATABASE_URL", "postgresql://localhost/voice_assistant"),
        help="PostgreSQL connection URL (default: from DATABASE_URL env or localhost/voice_assistant)"
    )
    parser.add_argument(
        "--create-db",
        action="store_true",
        help="Create the database if it doesn't exist"
    )
    parser.add_argument(
        "--db-name",
        type=str,
        default="voice_assistant",
        help="Database name to create (used with --create-db)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Database host (used with --create-db)"
    )
    parser.add_argument(
        "--user",
        type=str,
        default=os.getenv("PGUSER", os.getenv("USER", "postgres")),
        help="Database user (used with --create-db)"
    )
    parser.add_argument(
        "--password",
        type=str,
        default=os.getenv("PGPASSWORD", ""),
        help="Database password (used with --create-db)"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing setup, don't create anything"
    )
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("🗄️  Voice Assistant Database Setup")
    print("=" * 50)
    
    if args.verify_only:
        verify_setup(args.db_url)
        return
    
    # Create database if requested
    if args.create_db:
        print(f"\n📦 Creating database: {args.db_name}")
        if not create_database(args.host, args.user, args.password, args.db_name):
            return
        # Update URL to use new database
        args.db_url = f"postgresql://{args.user}:{args.password}@{args.host}/{args.db_name}"
    
    # Setup tables
    print(f"\n📦 Setting up tables in: {args.db_url.split('@')[-1] if '@' in args.db_url else args.db_url}")
    if not setup_tables(args.db_url):
        return
    
    # Verify setup
    verify_setup(args.db_url)
    
    print("\n" + "=" * 50)
    print("✅ Database setup complete!")
    print("=" * 50)
    print(f"\nTo use this database, set the environment variable:")
    print(f"  export DATABASE_URL=\"{args.db_url}\"")
    print("\nOr pass it to the voice assistant:")
    print(f"  python main.py --db-url \"{args.db_url}\"")


if __name__ == "__main__":
    main()

