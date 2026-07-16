"""Typed, durable local reminders for the voice-agent capability layer.

Reminder text is kept in a private SQLite database.  The production scheduler
receives only an opaque reminder id and invokes a fixed local helper; reminder
text therefore never appears in the transient systemd unit command line.

This module deliberately does not decide whether a transcript is an action.
The controller must mark a mutation as a direct, confirmed live-audio
instruction.  Read-only listing remains available without action authority.
"""
from __future__ import annotations

import os
import re
import secrets
import sqlite3
import stat
import subprocess
import sys
import threading
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Callable, Mapping, Optional, Protocol, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from always_on_agent.capabilities import (
    CapabilityRegistry,
    CapabilityResult,
    CapabilitySpec,
)
from always_on_agent.origin import Origin


_UTC = timezone.utc
_OPAQUE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_IDEMPOTENCY_MAX_CHARS = 240
_MESSAGE_MAX_CHARS = 500
_DEFAULT_STORE = "~/.local/state/speaker/reminders.sqlite3"
_ACTIVE_STATUSES = ("pending", "scheduled")
_DELIVERY_LEASE_SEC = 60.0
_VOICE_CLAIM_LEASE_SEC = 60.0
log = logging.getLogger("speaker.reminders")


class ReminderError(ValueError):
    """A reminder request is invalid or conflicts with a previous request."""


class ReminderBackendError(RuntimeError):
    """The local scheduling or notification backend failed."""


@dataclass(frozen=True)
class ReminderConfig:
    """Machine-local reminder settings with bounded parsing horizons."""

    enabled: bool = False
    store_path: str = _DEFAULT_STORE
    min_delay_sec: float = 5.0
    max_horizon_days: int = 366
    timezone_name: str = "local"

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, object]]) -> "ReminderConfig":
        data = data if isinstance(data, Mapping) else {}
        try:
            min_delay = float(data.get("min_delay_sec", 5.0))
        except (TypeError, ValueError, OverflowError):
            min_delay = 5.0
        try:
            max_days = int(data.get("max_horizon_days", 366))
        except (TypeError, ValueError, OverflowError):
            max_days = 366
        timezone_name = str(data.get("timezone", "local") or "local").strip()
        if len(timezone_name) > 128 or any(ord(char) < 32 for char in timezone_name):
            raise ReminderError("reminder timezone name is invalid")
        return cls(
            enabled=data.get("enabled") is True,
            store_path=str(data.get("store_path", _DEFAULT_STORE) or _DEFAULT_STORE),
            min_delay_sec=max(1.0, min(3600.0, min_delay)),
            max_horizon_days=max(1, min(3660, max_days)),
            timezone_name=timezone_name,
        )


@dataclass(frozen=True)
class ParsedReminder:
    message: str
    due_at: datetime


@dataclass(frozen=True)
class ReminderRecord:
    reminder_id: str
    idempotency_key: str
    message: str
    due_at: datetime
    status: str
    created_at: datetime
    delivered_at: datetime | None = None
    cancelled_at: datetime | None = None
    delivery_started_at: datetime | None = None
    delivery_claim_token: str = ""
    voice_claimed_at: datetime | None = None
    voice_claim_token: str = ""
    voice_announced_at: datetime | None = None


@dataclass(frozen=True)
class ReminderCreateResult:
    reminder: ReminderRecord
    created: bool


@dataclass(frozen=True)
class ReminderCancelResult:
    reminder: ReminderRecord | None
    cancelled: bool


_NUMBER_WORDS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}

_PREFIX = r"(?:please\s+)?remind\s+me(?:\s+to)?\s+"
_CREATE_ALIAS_RE = re.compile(
    r"^(?:please\s+)?(?:set|add|create)\s+(?:me\s+)?(?:a\s+)?reminder"
    r"(?:\s+to)?\s+(?P<body>.+)$",
    re.IGNORECASE,
)
_AMOUNT = r"(?:\d{1,6}|[a-z]+(?:[-\s][a-z]+)?)"
_RELATIVE_AFTER_RE = re.compile(
    rf"^{_PREFIX}(?P<message>.+?)\s+in\s+(?P<amount>{_AMOUNT})\s+"
    r"(?P<unit>seconds?|secs?|minutes?|mins?|hours?|hrs?|days?)\s*[.!?]?\s*$",
    re.IGNORECASE,
)
_RELATIVE_BEFORE_RE = re.compile(
    rf"^(?:please\s+)?in\s+(?P<amount>{_AMOUNT})\s+"
    r"(?P<unit>seconds?|secs?|minutes?|mins?|hours?|hrs?|days?),?\s+"
    r"remind\s+me(?:\s+to)?\s+(?P<message>.+?)\s*[.!?]?\s*$",
    re.IGNORECASE,
)
_DATE_AT_RE = re.compile(
    rf"^{_PREFIX}(?P<message>.+?)\s+on\s+(?P<date>\d{{4}}-\d{{2}}-\d{{2}})"
    r"\s+at\s+(?P<clock>\d{1,2}(?::\d{2})?\s*(?:am|pm)?)"
    r"(?:\s+(?P<zone>UTC|Z|[+-]\d{2}:\d{2}))?\s*[.!?]?\s*$",
    re.IGNORECASE,
)
_ISO_AT_RE = re.compile(
    rf"^{_PREFIX}(?P<message>.+?)\s+(?:at|on)\s+"
    r"(?P<stamp>\d{4}-\d{2}-\d{2}[T\s]\d{1,2}:\d{2}"
    r"(?::\d{2})?(?:Z|[+-]\d{2}:\d{2})?)\s*[.!?]?\s*$",
    re.IGNORECASE,
)
_TIME_ONLY_RE = re.compile(
    rf"^{_PREFIX}(?P<message>.+?)\s+at\s+"
    r"(?P<clock>\d{1,2}(?::\d{2})?\s*(?:am|pm))\s*[.!?]?\s*$|"
    rf"^{_PREFIX}(?P<message24>.+?)\s+at\s+"
    r"(?P<clock24>\d{1,2}:\d{2})\s*[.!?]?\s*$",
    re.IGNORECASE,
)


def _number(value: str) -> int | None:
    text = " ".join((value or "").casefold().replace("-", " ").split())
    if text.isdigit():
        return int(text)
    if text in _NUMBER_WORDS:
        return _NUMBER_WORDS[text]
    parts = text.split()
    if len(parts) == 2:
        tens = _NUMBER_WORDS.get(parts[0])
        ones = _NUMBER_WORDS.get(parts[1])
        if tens in (20, 30, 40, 50, 60, 70, 80, 90) and ones is not None and 0 < ones < 10:
            return tens + ones
    return None


def _clean_message(value: str) -> str:
    message = " ".join((value or "").split()).strip(" \t\r\n")
    if message.endswith((".", "!")):
        message = message[:-1].rstrip()
    if not message:
        raise ReminderError("reminder message is empty")
    if len(message) > _MESSAGE_MAX_CHARS:
        raise ReminderError("reminder message is too long")
    if any(ord(char) < 32 and char not in "\t\r\n" for char in message):
        raise ReminderError("reminder message contains control characters")
    return message


def _clock_parts(value: str) -> tuple[int, int]:
    match = re.fullmatch(
        r"\s*(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*(?P<ampm>am|pm)?\s*",
        value or "",
        re.IGNORECASE,
    )
    if match is None:
        raise ReminderError("invalid reminder time")
    hour = int(match.group("hour"))
    minute = int(match.group("minute") or 0)
    ampm = (match.group("ampm") or "").casefold()
    if minute > 59:
        raise ReminderError("invalid reminder time")
    if ampm:
        if hour < 1 or hour > 12:
            raise ReminderError("invalid reminder time")
        hour = hour % 12 + (12 if ampm == "pm" else 0)
    elif hour > 23:
        raise ReminderError("invalid reminder time")
    return hour, minute


def _zone(value: str | None, fallback: tzinfo) -> tzinfo:
    token = (value or "").strip().upper()
    if not token:
        return fallback
    if token in {"UTC", "Z"}:
        return _UTC
    match = re.fullmatch(r"(?P<sign>[+-])(?P<hour>\d{2}):(?P<minute>\d{2})", token)
    if match is None:
        raise ReminderError("invalid reminder timezone")
    hour = int(match.group("hour"))
    minute = int(match.group("minute"))
    if hour > 23 or minute > 59:
        raise ReminderError("invalid reminder timezone")
    offset = timedelta(hours=hour, minutes=minute)
    if match.group("sign") == "-":
        offset = -offset
    return timezone(offset)


def _configured_zone(value: str) -> tzinfo:
    """Resolve an explicit IANA zone or the host's IANA ``local`` zone."""

    name = str(value or "").strip()
    if name.casefold() != "local":
        return ZoneInfo(name)

    candidates: list[str] = []
    env_zone = os.environ.get("TZ", "").strip()
    if env_zone and not env_zone.startswith((":", "/")):
        candidates.append(env_zone)
    try:
        system_zone = Path("/etc/timezone").read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        system_zone = ""
    if system_zone:
        candidates.append(system_zone)
    try:
        localtime = Path("/etc/localtime").resolve(strict=True)
        marker = "/zoneinfo/"
        rendered = str(localtime)
        if marker in rendered:
            candidates.append(rendered.split(marker, 1)[1])
    except OSError:
        pass
    for candidate in candidates:
        if len(candidate) > 128 or any(ord(char) < 32 for char in candidate):
            continue
        try:
            return ZoneInfo(candidate)
        except (ZoneInfoNotFoundError, ValueError):
            continue
    raise ZoneInfoNotFoundError("system IANA timezone is unavailable")


class ReminderParser:
    """Small deterministic grammar for bounded relative and absolute reminders."""

    def __init__(
        self,
        *,
        clock: Callable[[], datetime] | None = None,
        local_timezone: tzinfo | None = None,
        min_delay_sec: float = 5.0,
        max_horizon_days: int = 366,
    ) -> None:
        self._clock = clock or (lambda: datetime.now().astimezone())
        self._local_timezone = local_timezone
        self.min_delay_sec = max(1.0, float(min_delay_sec))
        self.max_horizon = timedelta(days=max(1, int(max_horizon_days)))

    @property
    def local_timezone(self) -> tzinfo | None:
        return self._local_timezone

    def now_utc(self) -> datetime:
        now = self._clock()
        if not isinstance(now, datetime):
            raise ReminderError("reminder clock returned an invalid value")
        local = self._local_timezone or now.tzinfo or datetime.now().astimezone().tzinfo
        assert local is not None
        if now.tzinfo is None:
            now = now.replace(tzinfo=local)
        return now.astimezone(_UTC)

    def validate_due(self, due_at: datetime, *, now: datetime | None = None) -> datetime:
        if not isinstance(due_at, datetime):
            raise ReminderError("reminder due time is invalid")
        local = self._local_timezone or self._clock().tzinfo or datetime.now().astimezone().tzinfo
        assert local is not None
        if due_at.tzinfo is None:
            due_at = due_at.replace(tzinfo=local)
        due_utc = due_at.astimezone(_UTC)
        base = now.astimezone(_UTC) if now is not None else self.now_utc()
        delay = due_utc - base
        if delay.total_seconds() < self.min_delay_sec:
            raise ReminderError("reminder time must be in the future")
        if delay > self.max_horizon:
            raise ReminderError("reminder time is beyond the configured horizon")
        return due_utc

    def parse(self, text: str) -> ParsedReminder:
        query = (text or "").strip()
        alias = _CREATE_ALIAS_RE.fullmatch(query)
        if alias is not None:
            query = "remind me to " + alias.group("body").strip()
        now_utc = self.now_utc()
        local = self._local_timezone or self._clock().tzinfo or datetime.now().astimezone().tzinfo
        assert local is not None
        now_local = now_utc.astimezone(local)

        relative = _RELATIVE_AFTER_RE.match(query) or _RELATIVE_BEFORE_RE.match(query)
        if relative is not None:
            amount = _number(relative.group("amount"))
            if amount is None or amount <= 0:
                raise ReminderError("invalid reminder duration")
            unit = relative.group("unit").casefold()
            if unit.startswith(("sec",)):
                delta = timedelta(seconds=amount)
            elif unit.startswith(("min",)):
                delta = timedelta(minutes=amount)
            elif unit.startswith(("hour", "hr")):
                delta = timedelta(hours=amount)
            else:
                delta = timedelta(days=amount)
            due = self.validate_due(now_utc + delta, now=now_utc)
            return ParsedReminder(_clean_message(relative.group("message")), due)

        date_at = _DATE_AT_RE.match(query)
        if date_at is not None:
            try:
                year, month, day = (int(part) for part in date_at.group("date").split("-"))
                hour, minute = _clock_parts(date_at.group("clock"))
                due = datetime(
                    year,
                    month,
                    day,
                    hour,
                    minute,
                    tzinfo=_zone(date_at.group("zone"), local),
                )
            except ValueError as exc:
                raise ReminderError("invalid reminder date") from exc
            return ParsedReminder(
                _clean_message(date_at.group("message")),
                self.validate_due(due, now=now_utc),
            )

        iso_at = _ISO_AT_RE.match(query)
        if iso_at is not None:
            stamp = iso_at.group("stamp").replace(" ", "T", 1)
            if stamp.endswith(("Z", "z")):
                stamp = stamp[:-1] + "+00:00"
            try:
                due = datetime.fromisoformat(stamp)
            except ValueError as exc:
                raise ReminderError("invalid reminder date") from exc
            if due.tzinfo is None:
                due = due.replace(tzinfo=local)
            return ParsedReminder(
                _clean_message(iso_at.group("message")),
                self.validate_due(due, now=now_utc),
            )

        time_only = _TIME_ONLY_RE.match(query)
        if time_only is not None:
            message = time_only.group("message") or time_only.group("message24") or ""
            clock_text = time_only.group("clock") or time_only.group("clock24") or ""
            hour, minute = _clock_parts(clock_text)
            due_local = now_local.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if (due_local - now_local).total_seconds() < self.min_delay_sec:
                due_local += timedelta(days=1)
            return ParsedReminder(
                _clean_message(message),
                self.validate_due(due_local, now=now_utc),
            )

        raise ReminderError("unsupported reminder phrasing")


def _prepare_private_db(path: str) -> str:
    if path == ":memory:":
        return path
    requested = Path(os.path.expanduser(path))
    if not requested.name:
        raise ReminderError("reminder store path is invalid")
    requested.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    parent = requested.parent.resolve(strict=True)
    target = parent / requested.name
    try:
        info = os.lstat(target)
    except FileNotFoundError:
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(target, flags, 0o600)
        os.close(fd)
        info = os.lstat(target)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ReminderError("reminder store must be a regular file")
    os.chmod(target, 0o600, follow_symlinks=False)
    return str(target)


class ReminderStore:
    """Thread-safe private SQLite persistence shared by scheduler and helper."""

    def __init__(self, path: str = _DEFAULT_STORE) -> None:
        self.path = _prepare_private_db(path)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            self.path,
            timeout=5.0,
            isolation_level=None,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.execute("PRAGMA journal_mode=DELETE")
            self._conn.execute("PRAGMA synchronous=FULL")
            # The voice runtime and systemd helper can open the same database
            # during an upgrade. Serialize schema inspection + ALTERs under one
            # write transaction so two legacy openers cannot add the same column.
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS reminders (
                        reminder_id TEXT PRIMARY KEY,
                        idempotency_key TEXT NOT NULL UNIQUE,
                        message TEXT NOT NULL,
                        due_at REAL NOT NULL,
                        status TEXT NOT NULL CHECK (
                            status IN (
                                'pending', 'scheduled', 'delivering',
                                'delivered', 'delivery_failed', 'cancelled'
                            )
                        ),
                        created_at REAL NOT NULL,
                        delivered_at REAL,
                        cancelled_at REAL,
                        delivery_started_at REAL,
                        delivery_claim_token TEXT,
                        voice_claimed_at REAL,
                        voice_claim_token TEXT,
                        voice_announced_at REAL
                    )
                    """
                )
                self._conn.execute(
                    "CREATE INDEX IF NOT EXISTS reminders_due_idx "
                    "ON reminders(status, due_at)"
                )
                columns = {
                    str(row[1])
                    for row in self._conn.execute("PRAGMA table_info(reminders)")
                }
                migrations = (
                    ("voice_announced_at", "REAL"),
                    ("delivery_started_at", "REAL"),
                    ("voice_claimed_at", "REAL"),
                    ("delivery_claim_token", "TEXT"),
                    ("voice_claim_token", "TEXT"),
                )
                for name, sql_type in migrations:
                    if name not in columns:
                        self._conn.execute(
                            f"ALTER TABLE reminders ADD COLUMN {name} {sql_type}"
                        )
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
        if self.path != ":memory:":
            os.chmod(self.path, 0o600, follow_symlinks=False)

    @staticmethod
    def _record(row: sqlite3.Row | None) -> ReminderRecord | None:
        if row is None:
            return None
        return ReminderRecord(
            reminder_id=str(row["reminder_id"]),
            idempotency_key=str(row["idempotency_key"]),
            message=str(row["message"]),
            due_at=datetime.fromtimestamp(float(row["due_at"]), tz=_UTC),
            status=str(row["status"]),
            created_at=datetime.fromtimestamp(float(row["created_at"]), tz=_UTC),
            delivered_at=(
                datetime.fromtimestamp(float(row["delivered_at"]), tz=_UTC)
                if row["delivered_at"] is not None
                else None
            ),
            cancelled_at=(
                datetime.fromtimestamp(float(row["cancelled_at"]), tz=_UTC)
                if row["cancelled_at"] is not None
                else None
            ),
            delivery_started_at=(
                datetime.fromtimestamp(
                    float(row["delivery_started_at"]), tz=_UTC
                )
                if "delivery_started_at" in row.keys()
                and row["delivery_started_at"] is not None
                else None
            ),
            delivery_claim_token=(
                str(row["delivery_claim_token"] or "")
                if "delivery_claim_token" in row.keys()
                else ""
            ),
            voice_claimed_at=(
                datetime.fromtimestamp(float(row["voice_claimed_at"]), tz=_UTC)
                if "voice_claimed_at" in row.keys()
                and row["voice_claimed_at"] is not None
                else None
            ),
            voice_claim_token=(
                str(row["voice_claim_token"] or "")
                if "voice_claim_token" in row.keys()
                else ""
            ),
            voice_announced_at=(
                datetime.fromtimestamp(
                    float(row["voice_announced_at"]), tz=_UTC
                )
                if "voice_announced_at" in row.keys()
                and row["voice_announced_at"] is not None
                else None
            ),
        )

    def get(self, reminder_id: str) -> ReminderRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM reminders WHERE reminder_id = ?", (reminder_id,)
            ).fetchone()
        return self._record(row)

    def get_by_idempotency(self, key: str) -> ReminderRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM reminders WHERE idempotency_key = ?", (key,)
            ).fetchone()
        return self._record(row)

    def insert_pending(
        self,
        *,
        reminder_id: str,
        idempotency_key: str,
        message: str,
        due_at: datetime,
        created_at: datetime,
    ) -> tuple[ReminderRecord, bool]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    "SELECT * FROM reminders WHERE idempotency_key = ?",
                    (idempotency_key,),
                ).fetchone()
                if row is not None:
                    self._conn.execute("COMMIT")
                    record = self._record(row)
                    assert record is not None
                    return record, False
                self._conn.execute(
                    """
                    INSERT INTO reminders (
                        reminder_id, idempotency_key, message, due_at, status, created_at
                    ) VALUES (?, ?, ?, ?, 'pending', ?)
                    """,
                    (
                        reminder_id,
                        idempotency_key,
                        message,
                        due_at.timestamp(),
                        created_at.timestamp(),
                    ),
                )
                row = self._conn.execute(
                    "SELECT * FROM reminders WHERE reminder_id = ?", (reminder_id,)
                ).fetchone()
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
        record = self._record(row)
        assert record is not None
        return record, True

    def mark_scheduled(self, reminder_id: str) -> ReminderRecord:
        with self._lock:
            self._conn.execute(
                "UPDATE reminders SET status = 'scheduled', "
                "delivery_started_at = NULL, delivery_claim_token = NULL "
                "WHERE reminder_id = ? AND status = 'pending'",
                (reminder_id,),
            )
        record = self.get(reminder_id)
        if record is None:
            raise ReminderError("reminder disappeared while scheduling")
        return record

    def list(self, *, include_inactive: bool = False) -> tuple[ReminderRecord, ...]:
        with self._lock:
            if include_inactive:
                rows = self._conn.execute(
                    "SELECT * FROM reminders ORDER BY due_at, created_at"
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM reminders WHERE status IN ('pending', 'scheduled') "
                    "ORDER BY due_at, created_at"
                ).fetchall()
        return tuple(record for row in rows if (record := self._record(row)) is not None)

    def mark_cancelled(self, reminder_id: str, *, at: datetime) -> ReminderRecord | None:
        with self._lock:
            self._conn.execute(
                """
                UPDATE reminders
                SET status = 'cancelled', cancelled_at = ?,
                    delivery_started_at = NULL, delivery_claim_token = NULL
                WHERE reminder_id = ? AND status IN (
                    'pending', 'scheduled', 'delivering', 'delivery_failed'
                )
                """,
                (at.timestamp(), reminder_id),
            )
        return self.get(reminder_id)

    def claim_delivery(
        self,
        reminder_id: str,
        *,
        at: datetime | None = None,
    ) -> ReminderRecord | None:
        """Atomically claim one scheduled/pending reminder; repeats do nothing.

        ``pending`` is accepted deliberately: if systemd created the timer and
        the process crashed before ``mark_scheduled``, the opaque-id helper must
        still deliver instead of silently dropping the reminder.
        """

        claim_at = at or datetime.now(_UTC)
        if claim_at.tzinfo is None:
            claim_at = claim_at.replace(tzinfo=_UTC)
        claim_at = claim_at.astimezone(_UTC)
        claim_token = secrets.token_hex(16)
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    "SELECT * FROM reminders WHERE reminder_id = ?", (reminder_id,)
                ).fetchone()
                if row is None or row["status"] not in _ACTIVE_STATUSES:
                    self._conn.execute("COMMIT")
                    return None
                self._conn.execute(
                    "UPDATE reminders SET status = 'delivering', "
                    "delivery_started_at = ?, delivery_claim_token = ? "
                    "WHERE reminder_id = ?",
                    (claim_at.timestamp(), claim_token, reminder_id),
                )
                row = self._conn.execute(
                    "SELECT * FROM reminders WHERE reminder_id = ?", (reminder_id,)
                ).fetchone()
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
        return self._record(row)

    def recover_stale_deliveries(
        self,
        *,
        stale_before: datetime,
    ) -> tuple[ReminderRecord, ...]:
        """Make only expired delivery claims retryable.

        A second process may be actively calling the desktop notifier while the
        voice runtime starts. Its fresh lease must not be demoted or scheduled a
        second time. Legacy rows without a lease are treated as stale.
        """

        if stale_before.tzinfo is None:
            stale_before = stale_before.replace(tzinfo=_UTC)
        cutoff = stale_before.astimezone(_UTC).timestamp()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                rows = self._conn.execute(
                    """
                    SELECT * FROM reminders
                    WHERE status = 'delivering'
                      AND (delivery_started_at IS NULL OR delivery_started_at <= ?)
                    ORDER BY due_at, created_at
                    """,
                    (cutoff,),
                ).fetchall()
                reminder_ids = tuple(str(row["reminder_id"]) for row in rows)
                if reminder_ids:
                    placeholders = ", ".join("?" for _ in reminder_ids)
                    self._conn.execute(
                        "UPDATE reminders SET status = 'pending', "
                        "delivery_started_at = NULL, delivery_claim_token = NULL "
                        "WHERE status = 'delivering' "
                        f"AND reminder_id IN ({placeholders})",
                        reminder_ids,
                    )
                    rows = self._conn.execute(
                        "SELECT * FROM reminders WHERE reminder_id IN ("
                        f"{placeholders}) ORDER BY due_at, created_at",
                        reminder_ids,
                    ).fetchall()
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
        return tuple(
            record
            for row in rows
            if (record := self._record(row)) is not None
        )

    def claim_voice_announcements(
        self,
        *,
        since: datetime,
        at: datetime,
        claim_stale_before: datetime,
        limit: int = 10,
    ) -> tuple[ReminderRecord, ...]:
        """Atomically claim recently delivered reminders for in-agent speech.

        The standalone timer helper remains responsible for durable delivery.
        A running voice agent polls this marker and speaks each recent result at
        most once.  Only opaque ids cross the update boundary; reminder text is
        never written to logs or scheduler arguments here.
        """

        bounded_limit = max(1, min(100, int(limit)))
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                rows = self._conn.execute(
                    """
                    SELECT * FROM reminders
                    WHERE status IN ('delivered', 'delivery_failed')
                      AND delivered_at IS NOT NULL
                      AND delivered_at >= ?
                      AND voice_announced_at IS NULL
                      AND (voice_claimed_at IS NULL OR voice_claimed_at <= ?)
                    ORDER BY delivered_at, due_at, created_at
                    LIMIT ?
                    """,
                    (
                        since.timestamp(),
                        claim_stale_before.timestamp(),
                        bounded_limit,
                    ),
                ).fetchall()
                reminder_ids = tuple(str(row["reminder_id"]) for row in rows)
                if reminder_ids:
                    placeholders = ", ".join("?" for _ in reminder_ids)
                    for reminder_id in reminder_ids:
                        self._conn.execute(
                            "UPDATE reminders SET voice_claimed_at = ?, "
                            "voice_claim_token = ? WHERE reminder_id = ? "
                            "AND voice_announced_at IS NULL",
                            (
                                at.timestamp(),
                                secrets.token_hex(16),
                                reminder_id,
                            ),
                        )
                    rows = self._conn.execute(
                        "SELECT * FROM reminders WHERE reminder_id IN ("
                        f"{placeholders}) ORDER BY delivered_at, due_at, created_at",
                        reminder_ids,
                    ).fetchall()
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
        return tuple(
            record
            for row in rows
            if (record := self._record(row)) is not None
        )

    def renew_voice_claim(
        self,
        reminder_id: str,
        *,
        claim_token: str,
        at: datetime,
    ) -> bool:
        """Fence and renew exactly the claim about to be handed to TTS."""

        if (
            _OPAQUE_ID_RE.fullmatch(reminder_id or "") is None
            or _OPAQUE_ID_RE.fullmatch(claim_token or "") is None
        ):
            raise ReminderError("invalid opaque reminder id")
        with self._lock:
            cursor = self._conn.execute(
                """
                UPDATE reminders
                SET voice_claimed_at = ?
                WHERE reminder_id = ?
                  AND voice_announced_at IS NULL
                  AND voice_claim_token = ?
                """,
                (at.timestamp(), reminder_id, claim_token),
            )
        return int(cursor.rowcount) == 1

    def mark_voice_announced(
        self,
        reminder_id: str,
        *,
        claim_token: str,
        at: datetime,
    ) -> bool:
        """Commit exactly the claim whose TTS handoff succeeded."""

        with self._lock:
            cursor = self._conn.execute(
                "UPDATE reminders SET voice_announced_at = ?, "
                "voice_claimed_at = NULL, voice_claim_token = NULL "
                "WHERE reminder_id = ? AND voice_announced_at IS NULL "
                "AND voice_claim_token = ?",
                (at.timestamp(), reminder_id, claim_token),
            )
        return int(cursor.rowcount) == 1

    def release_voice_announcement(
        self,
        reminder_id: str,
        *,
        claim_token: str,
    ) -> bool:
        """Make a committed claim retryable after a failed TTS handoff."""

        with self._lock:
            cursor = self._conn.execute(
                "UPDATE reminders SET voice_announced_at = NULL, "
                "voice_claimed_at = NULL, voice_claim_token = NULL "
                "WHERE reminder_id = ? AND voice_claim_token = ?",
                (reminder_id, claim_token),
            )
        return int(cursor.rowcount) == 1

    def finish_delivery(
        self,
        reminder_id: str,
        *,
        claim_token: str,
        at: datetime,
        ok: bool,
    ) -> bool:
        status = "delivered" if ok else "delivery_failed"
        if _OPAQUE_ID_RE.fullmatch(claim_token or "") is None:
            return False
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE reminders SET status = ?, delivered_at = ?, "
                "delivery_started_at = NULL, delivery_claim_token = NULL "
                "WHERE reminder_id = ? AND status = 'delivering' "
                "AND delivery_claim_token = ?",
                (status, at.timestamp(), reminder_id, claim_token),
            )
        return int(cursor.rowcount) == 1

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class ReminderScheduler(Protocol):
    def schedule(self, reminder_id: str, due_at: datetime) -> None: ...

    def cancel(self, reminder_id: str) -> None: ...


Runner = Callable[..., object]


class SystemdReminderScheduler:
    """Schedule opaque ids with a transient user timer and fixed helper argv."""

    def __init__(
        self,
        store_path: str,
        *,
        runner: Runner = subprocess.run,
        systemd_run: str = "/usr/bin/systemd-run",
        systemctl: str = "/usr/bin/systemctl",
        helper_command: Sequence[str] | None = None,
        timeout_sec: float = 8.0,
    ) -> None:
        self.store_path = os.path.abspath(os.path.expanduser(store_path))
        self._runner = runner
        self._systemd_run = str(systemd_run)
        self._systemctl = str(systemctl)
        helper = helper_command or (
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "tools" / "reminder_notify.py"),
        )
        if not helper or any(not str(part) for part in helper):
            raise ReminderError("reminder notify helper command is invalid")
        self._helper = tuple(str(part) for part in helper)
        self._timeout = max(1.0, min(30.0, float(timeout_sec)))

    @staticmethod
    def unit_name(reminder_id: str) -> str:
        if _OPAQUE_ID_RE.fullmatch(reminder_id or "") is None:
            raise ReminderError("invalid opaque reminder id")
        return f"speaker-reminder-{reminder_id}"

    def schedule_argv(self, reminder_id: str, due_at: datetime) -> tuple[str, ...]:
        unit = self.unit_name(reminder_id)
        if due_at.tzinfo is None:
            raise ReminderError("scheduled reminder time must include a timezone")
        calendar = due_at.astimezone(_UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        return (
            self._systemd_run,
            "--user",
            "--no-ask-password",
            "--quiet",
            "--collect",
            f"--unit={unit}",
            f"--on-calendar={calendar}",
            "--",
            *self._helper,
            "--store",
            self.store_path,
            "--reminder-id",
            reminder_id,
        )

    def cancel_argv(self, reminder_id: str) -> tuple[str, ...]:
        unit = self.unit_name(reminder_id)
        return (
            self._systemctl,
            "--user",
            "--no-ask-password",
            "stop",
            f"{unit}.timer",
            f"{unit}.service",
        )

    def loaded_argv(self, reminder_id: str) -> tuple[str, ...]:
        unit = self.unit_name(reminder_id)
        return (
            self._systemctl,
            "--user",
            "--no-ask-password",
            "show",
            "--property=LoadState",
            "--value",
            f"{unit}.timer",
        )

    def _run(self, argv: Sequence[str], action: str) -> None:
        try:
            result = self._runner(
                list(argv),
                capture_output=True,
                text=True,
                timeout=self._timeout,
                check=False,
                shell=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ReminderBackendError(f"local reminder {action} failed") from exc
        if int(getattr(result, "returncode", 1)) != 0:
            # Do not include stderr: unit diagnostics can contain helper arguments.
            raise ReminderBackendError(f"local reminder {action} failed")

    def schedule(self, reminder_id: str, due_at: datetime) -> None:
        try:
            self._run(self.schedule_argv(reminder_id, due_at), "scheduling")
        except ReminderBackendError as scheduling_error:
            # ``systemd-run`` may have created the timer immediately before its
            # client timed out or lost the reply.  Resolve that ambiguity using
            # the opaque unit name: a loaded timer means the idempotent schedule
            # already exists and must not be duplicated under a fresh id.
            try:
                result = self._runner(
                    list(self.loaded_argv(reminder_id)),
                    capture_output=True,
                    text=True,
                    timeout=self._timeout,
                    check=False,
                    shell=False,
                )
            except (OSError, subprocess.SubprocessError):
                raise scheduling_error
            if (
                int(getattr(result, "returncode", 1)) == 0
                and str(getattr(result, "stdout", "") or "").strip() == "loaded"
            ):
                return
            raise scheduling_error

    def cancel(self, reminder_id: str) -> None:
        self._run(self.cancel_argv(reminder_id), "cancellation")


def _idempotency_key(value: object) -> str:
    key = str(value or "").strip()
    if not key or len(key) > _IDEMPOTENCY_MAX_CHARS:
        raise ReminderError("a bounded idempotency key is required")
    if any(ord(char) < 32 for char in key):
        raise ReminderError("idempotency key contains control characters")
    return key


class ReminderManager:
    """Coordinates parsing, durable idempotency, scheduling, listing, and cancel."""

    def __init__(
        self,
        store: ReminderStore,
        scheduler: ReminderScheduler,
        *,
        parser: ReminderParser | None = None,
        id_factory: Callable[[], str] | None = None,
        delivery_lease_sec: float = _DELIVERY_LEASE_SEC,
    ) -> None:
        self.store = store
        self.scheduler = scheduler
        self.parser = parser or ReminderParser()
        self._id_factory = id_factory or (lambda: secrets.token_hex(16))
        self._delivery_lease_sec = max(
            10.0,
            min(3600.0, float(delivery_lease_sec)),
        )
        self._schedule_retry_sec = 5.0
        self._schedule_retry_at: dict[str, datetime] = {}
        self._mutation_lock = threading.Lock()

    def create_from_text(self, text: str, *, idempotency_key: str) -> ReminderCreateResult:
        key = _idempotency_key(idempotency_key)
        # A retry must reuse the originally persisted absolute due time. Re-
        # parsing "in ten minutes" here would move the deadline and conflict
        # with the idempotency row after even a short backend timeout.
        with self._mutation_lock:
            existing = self.store.get_by_idempotency(key)
            if existing is not None:
                if existing.status == "pending":
                    self.scheduler.schedule(existing.reminder_id, existing.due_at)
                    existing = self.store.mark_scheduled(existing.reminder_id)
                    self._schedule_retry_at.pop(existing.reminder_id, None)
                return ReminderCreateResult(existing, False)
        parsed = self.parser.parse(text)
        return self.create(
            parsed.message,
            parsed.due_at,
            idempotency_key=key,
        )

    def create(
        self,
        message: str,
        due_at: datetime,
        *,
        idempotency_key: str,
    ) -> ReminderCreateResult:
        clean_message = _clean_message(message)
        now = self.parser.now_utc()
        due = self.parser.validate_due(due_at, now=now)
        key = _idempotency_key(idempotency_key)
        with self._mutation_lock:
            existing = self.store.get_by_idempotency(key)
            if existing is not None:
                if (
                    existing.message != clean_message
                    or abs((existing.due_at - due).total_seconds()) > 0.5
                ):
                    raise ReminderError("idempotency key was already used for another reminder")
                if existing.status != "pending":
                    return ReminderCreateResult(existing, False)
                record = existing
                inserted = False
            else:
                reminder_id = str(self._id_factory()).casefold()
                if _OPAQUE_ID_RE.fullmatch(reminder_id) is None:
                    raise ReminderError("reminder id factory returned an invalid opaque id")
                record, inserted = self.store.insert_pending(
                    reminder_id=reminder_id,
                    idempotency_key=key,
                    message=clean_message,
                    due_at=due,
                    created_at=now,
                )
            # A pending record is deliberately retained on an ambiguous backend
            # failure. Retrying the same idempotency key reuses the same opaque id
            # and therefore the same systemd unit name rather than duplicating it.
            self.scheduler.schedule(record.reminder_id, record.due_at)
            record = self.store.mark_scheduled(record.reminder_id)
            self._schedule_retry_at.pop(record.reminder_id, None)
            return ReminderCreateResult(record, inserted)

    def list(self, *, include_inactive: bool = False) -> tuple[ReminderRecord, ...]:
        return self.store.list(include_inactive=include_inactive)

    def resolve_id(self, reference: str) -> str:
        """Resolve a full id or an unambiguous eight-or-more-character prefix.

        Spoken/listed references never become scheduler arguments directly: the
        resolved, store-backed opaque id is passed to :meth:`cancel` instead.
        """

        token = str(reference or "").strip().casefold()
        if re.fullmatch(r"[0-9a-f]{8,32}", token) is None:
            raise ReminderError("invalid reminder reference")
        if len(token) == 32:
            record = self.store.get(token)
            if record is None or record.status not in (*_ACTIVE_STATUSES, "delivery_failed"):
                raise ReminderError("active reminder was not found")
            return token
        matches = [
            record.reminder_id
            for record in self.store.list()
            if record.reminder_id.startswith(token)
        ]
        if not matches:
            raise ReminderError("active reminder was not found")
        if len(matches) != 1:
            raise ReminderError("reminder reference is ambiguous")
        return matches[0]

    def cancel(self, reminder_id: str) -> ReminderCancelResult:
        if _OPAQUE_ID_RE.fullmatch(reminder_id or "") is None:
            raise ReminderError("invalid opaque reminder id")
        with self._mutation_lock:
            record = self.store.get(reminder_id)
            if record is None:
                return ReminderCancelResult(None, False)
            if record.status not in (*_ACTIVE_STATUSES, "delivering", "delivery_failed"):
                return ReminderCancelResult(record, False)
            prior_status = record.status
            # The durable state is the delivery guard. Flip it first so a due
            # helper racing systemctl observes cancelled and performs no action.
            record = self.store.mark_cancelled(reminder_id, at=self.parser.now_utc())
            if record is None or record.status != "cancelled":
                return ReminderCancelResult(record, False)
            self._schedule_retry_at.pop(reminder_id, None)
            if prior_status in (*_ACTIVE_STATUSES, "delivering"):
                try:
                    self.scheduler.cancel(reminder_id)
                except ReminderBackendError:
                    # A missing/finished transient unit is harmless once SQLite
                    # is cancelled; never resurrect the reminder on this error.
                    log.warning(
                        "reminder transient unit could not be stopped after durable cancel"
                    )
            return ReminderCancelResult(record, True)

    def reconcile(self) -> None:
        """Best-effort recreation of timers after an interrupted process."""

        now = self.parser.now_utc()
        self.store.recover_stale_deliveries(
            stale_before=now - timedelta(seconds=self._delivery_lease_sec)
        )
        self._schedule_records(self.store.list(), now=now)

    def reconcile_stale_deliveries(self) -> None:
        """Retry expired helper claims and pending scheduler publications."""

        now = self.parser.now_utc()
        self.store.recover_stale_deliveries(
            stale_before=now - timedelta(seconds=self._delivery_lease_sec)
        )
        records = tuple(
            record
            for record in self.store.list()
            if (
                record.status == "pending"
                and self._schedule_retry_at.get(record.reminder_id, now) <= now
            )
            or (
                record.status == "scheduled"
                and record.reminder_id in self._schedule_retry_at
                and self._schedule_retry_at[record.reminder_id] <= now
            )
        )
        self._schedule_records(records, now=now)

    def _schedule_records(
        self,
        records: Sequence[ReminderRecord],
        *,
        now: datetime,
    ) -> None:
        for record in records:
            schedule_at = record.due_at
            if schedule_at <= now:
                schedule_at = now + timedelta(
                    seconds=max(1.0, self.parser.min_delay_sec)
                )
            try:
                self.scheduler.schedule(record.reminder_id, schedule_at)
                self.store.mark_scheduled(record.reminder_id)
                self._schedule_retry_at.pop(record.reminder_id, None)
            except ReminderBackendError:
                self._schedule_retry_at[record.reminder_id] = now + timedelta(
                    seconds=self._schedule_retry_sec
                )
                log.warning("reminder timer reconciliation failed for an opaque id")

    def claim_voice_announcements(
        self,
        *,
        max_age_sec: float = 600.0,
        limit: int = 10,
    ) -> tuple[ReminderRecord, ...]:
        """Claim recent durable deliveries for speech by a running agent."""

        now = self.parser.now_utc()
        age = max(1.0, min(3600.0, float(max_age_sec)))
        return self.store.claim_voice_announcements(
            since=now - timedelta(seconds=age),
            at=now,
            claim_stale_before=now - timedelta(seconds=_VOICE_CLAIM_LEASE_SEC),
            limit=limit,
        )

    def renew_voice_claim(
        self,
        reminder_id: str,
        *,
        claim_token: str,
    ) -> bool:
        return self.store.renew_voice_claim(
            reminder_id,
            claim_token=claim_token,
            at=self.parser.now_utc(),
        )

    def mark_voice_announced(
        self,
        reminder_id: str,
        *,
        claim_token: str,
    ) -> bool:
        return self.store.mark_voice_announced(
            reminder_id,
            claim_token=claim_token,
            at=self.parser.now_utc(),
        )

    def release_voice_announcement(
        self,
        reminder_id: str,
        *,
        claim_token: str,
    ) -> bool:
        return self.store.release_voice_announcement(
            reminder_id,
            claim_token=claim_token,
        )

    def close(self) -> None:
        self.store.close()


def build_reminder_manager(
    config: ReminderConfig,
    *,
    scheduler: ReminderScheduler | None = None,
    parser: ReminderParser | None = None,
    runner: Runner = subprocess.run,
) -> ReminderManager | None:
    """Build the manager only for an explicit machine-local feature opt-in."""

    if not config.enabled:
        return None
    store = ReminderStore(config.store_path)
    if parser is None:
        local_timezone: tzinfo | None = None
        if config.timezone_name:
            try:
                local_timezone = _configured_zone(config.timezone_name)
            except (ZoneInfoNotFoundError, ValueError) as exc:
                store.close()
                raise ReminderError("configured reminder timezone is unavailable") from exc
        effective_parser = ReminderParser(
            local_timezone=local_timezone,
            min_delay_sec=config.min_delay_sec,
            max_horizon_days=config.max_horizon_days,
        )
    else:
        effective_parser = parser
    backend = scheduler or SystemdReminderScheduler(store.path, runner=runner)
    manager = ReminderManager(store, backend, parser=effective_parser)
    manager.reconcile()
    return manager


def _authorized_mutation(context: Mapping[str, object]) -> bool:
    origin = context.get("origin", Origin.UNKNOWN)
    if isinstance(origin, Origin):
        origin = origin.value
    return bool(
        origin == Origin.LIVE_AUDIO.value
        and context.get("direct_user_instruction") is True
        and context.get("confirmed") is True
    )


def _blocked_mutation() -> CapabilityResult:
    return CapabilityResult(
        True,
        "I need a direct, confirmed live request before changing reminders.",
        data={"executed": False, "blocked": "action_authorization"},
    )


_CANCEL_ID_RE = re.compile(
    r"^cancel\s+(?:the\s+)?reminder\s+(?P<id>[0-9a-f]{32})\s*[.!?]?\s*$",
    re.IGNORECASE,
)


def attach_reminder_capabilities(
    registry: CapabilityRegistry,
    manager: ReminderManager,
) -> CapabilityRegistry:
    """Attach typed reminder create/list/cancel providers to ``registry``."""

    def spoken_due(due_at: datetime) -> str:
        local_zone = (
            manager.parser.local_timezone
            or datetime.now().astimezone().tzinfo
            or _UTC
        )
        local = due_at.astimezone(local_zone)
        hour = local.strftime("%I").lstrip("0") or "12"
        zone = local.tzname() or "local time"
        return (
            f"{local:%Y-%m-%d} at {hour}:{local:%M} "
            f"{local:%p} {zone}"
        )

    def create_provider(query: str, context: dict[str, object]) -> CapabilityResult:
        if not _authorized_mutation(context):
            return _blocked_mutation()
        key = context.get("idempotency_key") or context.get("action_id") or context.get("task_id")
        try:
            request = context.get("reminder_request")
            if isinstance(request, ParsedReminder):
                result = manager.create(
                    request.message,
                    request.due_at,
                    idempotency_key=_idempotency_key(key),
                )
            else:
                result = manager.create_from_text(
                    query,
                    idempotency_key=_idempotency_key(key),
                )
        except (ReminderError, ReminderBackendError) as exc:
            return CapabilityResult(False, "", error=str(exc))
        record = result.reminder
        return CapabilityResult(
            True,
            f"Reminder set for {spoken_due(record.due_at)}.",
            data={
                "executed": result.created,
                "idempotent_replay": not result.created,
                "reminder_id": record.reminder_id,
                "due_at": record.due_at.isoformat(),
            },
        )

    def list_provider(_query: str, _context: dict[str, object]) -> CapabilityResult:
        reminders = manager.list()
        if not reminders:
            return CapabilityResult(True, "You have no active reminders.", data={"count": 0})
        rendered = "; ".join(
            f"{record.message} at {spoken_due(record.due_at)} "
            f"({record.reminder_id[:8]})"
            for record in reminders[:10]
        )
        return CapabilityResult(
            True,
            "Active reminders: " + rendered,
            data={
                "count": len(reminders),
                "reminders": [
                    {
                        "reminder_id": record.reminder_id,
                        "message": record.message,
                        "due_at": record.due_at.isoformat(),
                        "status": record.status,
                    }
                    for record in reminders[:10]
                ],
            },
        )

    def cancel_provider(query: str, context: dict[str, object]) -> CapabilityResult:
        if not _authorized_mutation(context):
            return _blocked_mutation()
        reminder_id = str(context.get("reminder_id", "") or "")
        if not reminder_id:
            match = _CANCEL_ID_RE.fullmatch(query or "")
            reminder_id = match.group("id").casefold() if match is not None else ""
        if not reminder_id:
            return CapabilityResult(
                True,
                "I couldn't find that reminder.",
                data={"executed": False, "missing": True},
            )
        try:
            result = manager.cancel(reminder_id)
        except (ReminderError, ReminderBackendError) as exc:
            return CapabilityResult(False, "", error=str(exc))
        if result.reminder is None:
            return CapabilityResult(
                True,
                "I couldn't find that reminder.",
                data={"executed": False, "missing": True},
            )
        return CapabilityResult(
            True,
            "Reminder cancelled." if result.cancelled else "That reminder was already inactive.",
            data={
                "executed": result.cancelled,
                "reminder_id": result.reminder.reminder_id,
            },
        )

    registry.register(
        "reminder.create",
        create_provider,
        spec=CapabilitySpec(
            name="reminder.create",
            summary="set a durable local reminder after confirmation",
            when_to_use="when the user directly asks to set a reminder",
            egress="local",
            speaks=True,
            side_effecting=True,
            planner_tool=False,
            # Mutations are controller-handled and must not be advertised to
            # the answering model, which could otherwise falsely claim it set
            # a reminder on a garbled phrase that missed deterministic routing.
            user_facing=False,
            authority="direct_live",
            requires_confirmation=True,
        ),
    )
    registry.register(
        "reminder.list",
        list_provider,
        spec=CapabilitySpec(
            name="reminder.list",
            summary="list active local reminders",
            when_to_use="when the user asks which reminders are active",
            egress="local",
            speaks=True,
            side_effecting=False,
            planner_tool=False,
            user_facing=True,
            authority="none",
        ),
    )
    registry.register(
        "reminder.cancel",
        cancel_provider,
        spec=CapabilitySpec(
            name="reminder.cancel",
            summary="cancel a durable local reminder after confirmation",
            when_to_use="when the user directly asks to cancel a reminder",
            egress="local",
            speaks=True,
            side_effecting=True,
            planner_tool=False,
            user_facing=False,
            authority="direct_live",
            requires_confirmation=True,
        ),
    )
    return registry


# Singular spelling for callers that treat the reminder integration as one tool.
attach_reminder_capability = attach_reminder_capabilities


__all__ = [
    "ParsedReminder",
    "ReminderBackendError",
    "ReminderCancelResult",
    "ReminderConfig",
    "ReminderCreateResult",
    "ReminderError",
    "ReminderManager",
    "ReminderParser",
    "ReminderRecord",
    "ReminderScheduler",
    "ReminderStore",
    "SystemdReminderScheduler",
    "attach_reminder_capabilities",
    "attach_reminder_capability",
    "build_reminder_manager",
]
