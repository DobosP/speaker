"""Apply / rollback / inspect smart-memory PostgreSQL migrations.

Thin wrapper around yoyo-migrations. The migrations live in
``migrations/*.sql`` next to this script's project root; each ``<id>.sql``
file has a matching ``<id>.rollback.sql``. Yoyo tracks state in a
``_yoyo_migration`` table inside the target database.

Usage:

    python tools/migrate.py status                  # show applied / pending
    python tools/migrate.py apply                   # apply all pending
    python tools/migrate.py apply --dry-run         # show what would apply
    python tools/migrate.py rollback                # rollback the most recent
    python tools/migrate.py rollback --count 2      # rollback two

The database URL defaults to ``$DATABASE_URL``; override via
``--database-url``. The migrations themselves are PostgreSQL-only (they
use the pgvector ``vector`` type) -- on SQLite the schema will need a
parallel set of migrations; today we only ship the Postgres path.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _migrations_dir() -> Path:
    return _project_root() / "migrations"


def _backend(database_url: str):
    """Lazy import so the module is importable without yoyo + psycopg
    installed (matches the codebase convention for optional deps)."""
    from yoyo import get_backend  # type: ignore
    # yoyo accepts both ``postgresql://`` and ``postgres://`` -- normalize
    # so users who set DATABASE_URL=postgres:// (Heroku-style) work.
    if database_url.startswith("postgres://"):
        database_url = "postgresql://" + database_url[len("postgres://"):]
    if not database_url.startswith("postgresql+psycopg://") and database_url.startswith("postgresql://"):
        # Yoyo's psycopg3 driver tag.
        database_url = "postgresql+psycopg://" + database_url[len("postgresql://"):]
    return get_backend(database_url)


def _read_migrations(directory: Path):
    from yoyo import read_migrations  # type: ignore
    return read_migrations(str(directory))


def _redact_db_url(db_url: str) -> str:
    """Mask any password in a DATABASE_URL for safe logging/printing.

    Self-contained on purpose: it must NOT import ``setup_database`` (that module
    runs ``sys.exit(1)`` at import time when psycopg is absent -- a SystemExit,
    not an Exception, so it would crash this redaction on a psycopg-less box /
    CI). Masks both the userinfo password (``user:pass@host``) and a libpq
    query-param password (``?password=...`` / ``?sslpassword=...``). Redaction
    must never itself raise or leak.
    """
    try:
        import re
        from urllib.parse import urlsplit
        out = db_url
        parts = urlsplit(db_url)
        # Mask the userinfo password (user:pass@host) via a targeted replace so
        # the rest of the DSN (scheme, ///, path, query) is preserved exactly.
        if parts.password is not None and parts.netloc:
            user = parts.username or ""
            host = parts.hostname or ""
            masked = f"{user}:***@{host}" if user else f"***@{host}"
            if parts.port:
                masked = f"{masked}:{parts.port}"
            out = out.replace(parts.netloc, masked, 1)
        # Mask a libpq query-param password (?password=... / ?sslpassword=...).
        out = re.sub(
            r"((?:password|sslpassword)=)[^&\s]*", r"\1***", out, flags=re.IGNORECASE
        )
        return out
    except Exception:  # noqa: BLE001 - redaction must never raise / leak
        return db_url.split("@")[-1] if "@" in db_url else db_url


def _cmd_status(args: argparse.Namespace) -> int:
    backend = _backend(args.database_url)
    all_migrations = _read_migrations(_migrations_dir())
    with backend.lock():
        applied = backend.to_apply(all_migrations)  # the ones still pending
    pending_ids = {m.id for m in applied}
    print(f"Migrations directory: {_migrations_dir()}")
    print(f"Database:             {_redact_db_url(args.database_url)}")
    print()
    print(f"{'STATUS':<10} {'ID'}")
    for m in all_migrations:
        status = "PENDING" if m.id in pending_ids else "APPLIED"
        print(f"{status:<10} {m.id}")
    return 0


def _cmd_apply(args: argparse.Namespace) -> int:
    backend = _backend(args.database_url)
    all_migrations = _read_migrations(_migrations_dir())
    with backend.lock():
        to_apply = backend.to_apply(all_migrations)
        if not to_apply:
            print("Nothing to apply -- database is up to date.")
            return 0
        if args.dry_run:
            print(f"Would apply {len(to_apply)} migration(s):")
            for m in to_apply:
                print(f"  - {m.id}")
            return 0
        print(f"Applying {len(to_apply)} migration(s)...")
        backend.apply_migrations(to_apply)
    print("Done.")
    return 0


def _cmd_rollback(args: argparse.Namespace) -> int:
    backend = _backend(args.database_url)
    all_migrations = _read_migrations(_migrations_dir())
    with backend.lock():
        to_rollback = backend.to_rollback(all_migrations)
        target = to_rollback[: args.count] if args.count else to_rollback[:1]
        if not target:
            print("Nothing to rollback.")
            return 0
        if args.dry_run:
            print(f"Would rollback {len(target)} migration(s):")
            for m in target:
                print(f"  - {m.id}")
            return 0
        print(f"Rolling back {len(target)} migration(s)...")
        backend.rollback_migrations(target)
    print("Done.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply / rollback smart-memory PostgreSQL migrations (yoyo).",
    )
    parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL", "postgresql:///voice_assistant"),
        help="Postgres URL (default: $DATABASE_URL or postgresql:///voice_assistant)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    s_status = sub.add_parser("status", help="Show applied / pending migrations.")
    s_status.set_defaults(func=_cmd_status)

    s_apply = sub.add_parser("apply", help="Apply all pending migrations.")
    s_apply.add_argument("--dry-run", action="store_true", help="Show what would be applied; don't write.")
    s_apply.set_defaults(func=_cmd_apply)

    s_rollback = sub.add_parser("rollback", help="Rollback the most recent migration(s).")
    s_rollback.add_argument("--count", type=int, default=1, help="How many to rollback (default 1).")
    s_rollback.add_argument("--dry-run", action="store_true", help="Show what would be rolled back; don't write.")
    s_rollback.set_defaults(func=_cmd_rollback)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
