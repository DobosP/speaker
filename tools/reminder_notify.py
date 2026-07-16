#!/usr/bin/env python3
"""Deliver one opaque-id reminder from the private local reminder store.

The transient scheduler invokes this fixed helper with a store path and opaque
id.  Reminder text is read only after this process starts and is never accepted
as a command-line argument from the scheduler.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol, Sequence


# An absolute script path is used by systemd, so make the checkout/package root
# importable without relying on a login shell's working directory or PYTHONPATH.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.reminders import (  # noqa: E402  (path bootstrap is intentional)
    ReminderBackendError,
    ReminderStore,
)


class Notifier(Protocol):
    def notify(self, message: str) -> None: ...


Runner = Callable[..., object]


class NotifySendNotifier:
    """Send one local desktop notification through fixed argv and no shell."""

    def __init__(
        self,
        *,
        runner: Runner = subprocess.run,
        executable: str = "/usr/bin/notify-send",
        timeout_sec: float = 8.0,
    ) -> None:
        self._runner = runner
        self._executable = str(executable)
        self._timeout = max(1.0, min(30.0, float(timeout_sec)))

    def argv(self, message: str) -> tuple[str, ...]:
        text = str(message or "")
        if not text or len(text) > 500 or "\x00" in text:
            raise ReminderBackendError("stored reminder message is invalid")
        return (
            self._executable,
            "--app-name=Speaker",
            "--urgency=normal",
            "--",
            "Speaker reminder",
            text,
        )

    def notify(self, message: str) -> None:
        try:
            result = self._runner(
                list(self.argv(message)),
                capture_output=True,
                text=True,
                timeout=self._timeout,
                check=False,
                shell=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ReminderBackendError("local reminder notification failed") from exc
        if int(getattr(result, "returncode", 1)) != 0:
            raise ReminderBackendError("local reminder notification failed")


def _now_utc(clock: Callable[[], datetime]) -> datetime:
    now = clock()
    if not isinstance(now, datetime):
        raise ReminderBackendError("reminder helper clock returned an invalid value")
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)


def deliver_reminder(
    store: ReminderStore,
    reminder_id: str,
    notifier: Notifier,
    *,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> bool:
    """Atomically deliver at most once; return false for repeats/inactive ids."""

    record = store.claim_delivery(
        str(reminder_id or ""),
        at=_now_utc(clock),
    )
    if record is None:
        return False
    try:
        notifier.notify(record.message)
    except Exception as exc:
        store.finish_delivery(
            record.reminder_id,
            claim_token=record.delivery_claim_token,
            at=_now_utc(clock),
            ok=False,
        )
        if isinstance(exc, ReminderBackendError):
            raise
        raise ReminderBackendError("local reminder notification failed") from exc
    return store.finish_delivery(
        record.reminder_id,
        claim_token=record.delivery_claim_token,
        at=_now_utc(clock),
        ok=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deliver one local Speaker reminder")
    parser.add_argument("--store", required=True)
    parser.add_argument("--reminder-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    store: ReminderStore | None = None
    try:
        store = ReminderStore(args.store)
        deliver_reminder(store, args.reminder_id, NotifySendNotifier())
    except Exception:
        # Do not print reminder content, process diagnostics, or paths.  The
        # scheduler needs only a generic failure status for local troubleshooting.
        print("speaker reminder delivery failed", file=sys.stderr)
        return 1
    finally:
        if store is not None:
            store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
