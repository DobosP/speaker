#!/usr/bin/env python3
"""
Database Setup Script for Voice Assistant Memory — thin wrapper.

The schema is owned by the canonical migrations path
(``python -m tools.migrate apply``); this script no longer carries any
``CREATE TABLE`` SQL. It is reduced to two convenience helpers around that
path:

- ``--create-db``  : create the PostgreSQL *database* (role/db bootstrap) so
  the migrations have somewhere to run.
- ``--verify-only``: a small read-only health check of the resulting schema.

The default ``main()`` flow shells the schema step to ``tools.migrate`` and
then verifies — there is exactly one place the schema is defined.

Requirements:
- PostgreSQL 14+ with pgvector extension
- psycopg (psycopg3) + yoyo-migrations (both declared in requirements.txt)

Usage:
    # Apply migrations against the default local socket DB
    python setup_database.py

    # Against a custom database URL
    python setup_database.py --db-url "postgresql://user:pass@host/dbname"

    # Create the database first, then apply migrations
    python setup_database.py --create-db --db-name voice_assistant

    # Only verify an existing setup
    python setup_database.py --verify-only
"""
import argparse
import sys
import os
from urllib.parse import urlsplit, urlunsplit


def _redact_db_url(db_url: str) -> str:
    """Return ``db_url`` with any password component masked.

    Never print a raw ``DATABASE_URL`` to stdout — it commonly carries a
    password. This replaces the password with ``***`` while keeping the rest
    of the URL legible (user, host, path) for setup hints.
    """
    try:
        parts = urlsplit(db_url)
    except Exception:
        # If we can't parse it, fall back to host-only (drop any userinfo).
        return db_url.split("@")[-1] if "@" in db_url else db_url
    if parts.password is None:
        return db_url
    user = parts.username or ""
    host = parts.hostname or ""
    netloc = f"{user}:***@{host}" if user else f"***@{host}"
    if parts.port:
        netloc = f"{netloc}:{parts.port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


try:
    import psycopg  # psycopg3
except ImportError:
    print("❌ psycopg (psycopg3) not installed. Install with: pip install 'psycopg[binary,pool]'")
    sys.exit(1)


def create_database(host: str, user: str, password: str, db_name: str):
    """Create a new PostgreSQL database (psycopg3)."""
    try:
        # Connect to the default ``postgres`` database in autocommit mode --
        # CREATE DATABASE cannot run inside a transaction block.
        conn = psycopg.connect(
            host=host,
            user=user,
            password=password,
            dbname="postgres",
            autocommit=True,
        )
        try:
            with conn.cursor() as cur:
                # Check if database exists
                cur.execute(
                    "SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s",
                    (db_name,),
                )
                exists = cur.fetchone()

                if exists:
                    print(f"⚠️  Database '{db_name}' already exists")
                else:
                    cur.execute(f'CREATE DATABASE "{db_name}"')
                    print(f"✅ Created database: {db_name}")
        finally:
            conn.close()
        return True

    except Exception as e:
        print(f"❌ Failed to create database: {e}")
        return False


def verify_setup(db_url: str):
    """Verify the database setup is correct (psycopg3, read-only)."""
    try:
        conn = psycopg.connect(db_url)
        try:
            with conn.cursor() as cur:
                # Check tables
                cur.execute(
                    """
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name IN ('messages', 'summaries', 'user_profile')
                    """
                )
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
        finally:
            conn.close()
        return True

    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False


def _apply_migrations(db_url: str) -> bool:
    """Apply the canonical schema via ``tools.migrate apply``.

    Wrapped in try/except so a missing optional dep (yoyo-migrations) or a
    migration failure surfaces a friendly hint rather than a raw traceback.
    Never echoes the password — only a redacted URL appears in messages.
    """
    try:
        from tools.migrate import main as migrate_main
    except ImportError:
        print("❌ Could not import tools.migrate. Install schema deps with:")
        print("      pip install yoyo-migrations 'psycopg[binary,pool]'")
        return False
    try:
        rc = migrate_main(["apply", "--database-url", db_url])
    except ModuleNotFoundError as exc:
        # yoyo (or its driver) is imported lazily inside tools.migrate.
        print(f"❌ Schema migration dependency missing ({exc.name}). Install with:")
        print("      pip install yoyo-migrations 'psycopg[binary,pool]'")
        return False
    except Exception as exc:
        print(f"❌ Schema migration failed: {exc}")
        print("   Run the canonical path directly to see details:")
        print(f"      python -m tools.migrate apply --database-url \"{_redact_db_url(db_url)}\"")
        return False
    return rc == 0


def main():
    parser = argparse.ArgumentParser(
        description="Setup PostgreSQL database for Voice Assistant memory (thin migrate wrapper)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Peer authentication (no password)
  python setup_database.py --db-url "postgresql:///voice_assistant"

  # Password authentication
  python setup_database.py --db-url "postgresql://user:pass@localhost/voice_assistant"

  # Create new database, then apply migrations
  python setup_database.py --create-db --db-name voice_assistant

  # Verify existing setup
  python setup_database.py --verify-only

Note: the schema itself lives in the canonical migrations path
(`python -m tools.migrate apply`); this script just creates the database
(--create-db) and verifies the result (--verify-only).
        """
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=os.getenv("DATABASE_URL", "postgresql:///voice_assistant"),
        help="PostgreSQL connection URL (default: from DATABASE_URL env or local socket voice_assistant)"
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
    print("Schema is owned by the canonical migrations path:")
    print("      python -m tools.migrate apply")
    print("This script creates the database (--create-db) and verifies it")
    print("(--verify-only); the schema step below defers to tools.migrate.")

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

    # Schema step: defer to the canonical migrations path.
    print(f"\n📦 Applying migrations to: {_redact_db_url(args.db_url)}")
    if not _apply_migrations(args.db_url):
        return

    # Verify setup
    verify_setup(args.db_url)

    print("\n" + "=" * 50)
    print("✅ Database setup complete!")
    print("=" * 50)
    # NOTE: never echo the full DATABASE_URL — it may contain a password.
    # Show a password-redacted form for the hints below.
    redacted_url = _redact_db_url(args.db_url)
    print(f"\nTo use this database, set the environment variable:")
    print(f"  export DATABASE_URL=\"{redacted_url}\"")
    print("  (password redacted above — substitute your real credentials)")
    print("\nOr pass it to the voice assistant:")
    print(f"  python main.py --db-url \"{redacted_url}\"")


if __name__ == "__main__":
    main()
