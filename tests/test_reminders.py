from __future__ import annotations

import os
import sqlite3
import stat
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from always_on_agent.capabilities import CapabilityRegistry
from always_on_agent.origin import Origin
from core.reminders import (
    ReminderBackendError,
    ReminderConfig,
    ReminderError,
    ReminderManager,
    ReminderParser,
    ReminderStore,
    SystemdReminderScheduler,
    attach_reminder_capabilities,
    build_reminder_manager,
)
from tools.reminder_notify import NotifySendNotifier, deliver_reminder


NOW = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
REMINDER_ID = "0123456789abcdef0123456789abcdef"


class FakeScheduler:
    def __init__(self, *, fail_schedule_once: bool = False) -> None:
        self.schedule_calls: list[tuple[str, datetime]] = []
        self.cancel_calls: list[str] = []
        self.fail_schedule_once = fail_schedule_once

    def schedule(self, reminder_id: str, due_at: datetime) -> None:
        self.schedule_calls.append((reminder_id, due_at))
        if self.fail_schedule_once:
            self.fail_schedule_once = False
            raise ReminderBackendError("local reminder scheduling failed")

    def cancel(self, reminder_id: str) -> None:
        self.cancel_calls.append(reminder_id)


def parser() -> ReminderParser:
    return ReminderParser(
        clock=lambda: NOW,
        local_timezone=timezone.utc,
        min_delay_sec=1,
        max_horizon_days=30,
    )


def manager(tmp_path, scheduler: FakeScheduler | None = None) -> ReminderManager:
    return ReminderManager(
        ReminderStore(str(tmp_path / "private" / "reminders.sqlite3")),
        scheduler or FakeScheduler(),
        parser=parser(),
        id_factory=lambda: REMINDER_ID,
    )


@pytest.mark.parametrize(
    ("text", "message", "expected"),
    [
        ("remind me to drink water in 10 minutes?", "drink water", NOW + timedelta(minutes=10)),
        ("in twenty one minutes, remind me to stretch", "stretch", NOW + timedelta(minutes=21)),
        ("remind me to call Paul on 2026-07-17 at 14:30 UTC", "call Paul", datetime(2026, 7, 17, 14, 30, tzinfo=timezone.utc)),
        ("remind me to close the window at 6 pm", "close the window", datetime(2026, 7, 16, 18, 0, tzinfo=timezone.utc)),
        ("remind me to test at 2026-07-17T15:20:00+03:00", "test", datetime(2026, 7, 17, 12, 20, tzinfo=timezone.utc)),
    ],
)
def test_parser_accepts_bounded_relative_and_absolute_phrases(text, message, expected):
    result = parser().parse(text)
    assert result.message == message
    assert result.due_at == expected


@pytest.mark.parametrize("verb", ("set", "add", "create"))
def test_parser_accepts_common_create_reminder_synonyms(verb):
    result = parser().parse(f"{verb} a reminder to hydrate in ten minutes")
    assert result.message == "hydrate"
    assert result.due_at == NOW + timedelta(minutes=10)


def test_parser_rejects_unsupported_past_and_out_of_horizon_requests():
    with pytest.raises(ReminderError, match="unsupported"):
        parser().parse("remember that I like coffee")
    with pytest.raises(ReminderError, match="future"):
        parser().parse("remind me to test on 2026-07-15 at 10:00 UTC")
    with pytest.raises(ReminderError, match="horizon"):
        parser().parse("remind me to test in 31 days")


def test_configured_iana_timezone_is_applied_by_the_manager_builder(tmp_path):
    config = ReminderConfig.from_dict(
        {
            "enabled": True,
            "store_path": str(tmp_path / "timezone.sqlite3"),
            "timezone": "Europe/Bucharest",
        }
    )
    built = build_reminder_manager(config, scheduler=FakeScheduler())
    assert built is not None
    assert built.parser.local_timezone is not None
    assert built.parser.local_timezone.tzname(datetime(2026, 7, 16)) == "EEST"
    built.close()


def test_invalid_configured_timezone_fails_closed_and_closes_the_store(tmp_path):
    config = ReminderConfig(
        enabled=True,
        store_path=str(tmp_path / "invalid-timezone.sqlite3"),
        timezone_name="Nowhere/Imaginary",
    )
    with pytest.raises(ReminderError, match="unavailable"):
        build_reminder_manager(config, scheduler=FakeScheduler())


def test_create_is_private_durable_and_idempotent_across_restart(tmp_path):
    scheduler = FakeScheduler()
    first = manager(tmp_path, scheduler)
    result = first.create_from_text(
        "remind me to test backups in 10 minutes",
        idempotency_key="turn-1",
    )
    replay = first.create_from_text(
        "remind me to test backups in 10 minutes",
        idempotency_key="turn-1",
    )

    assert result.created is True
    assert replay.created is False
    assert replay.reminder.reminder_id == REMINDER_ID
    assert len(scheduler.schedule_calls) == 1
    path = first.store.path
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    first.close()

    restarted = ReminderManager(ReminderStore(path), FakeScheduler(), parser=parser())
    assert restarted.list()[0].message == "test backups"
    assert restarted.list()[0].status == "scheduled"
    restarted.close()


def test_ambiguous_schedule_failure_retries_same_opaque_id_without_duplicate(tmp_path):
    scheduler = FakeScheduler(fail_schedule_once=True)
    reminders = manager(tmp_path, scheduler)
    with pytest.raises(ReminderBackendError):
        reminders.create_from_text(
            "remind me to stretch in 10 minutes",
            idempotency_key="turn-retry",
        )
    pending = reminders.list()[0]
    assert pending.status == "pending"

    result = reminders.create_from_text(
        "remind me to stretch in 10 minutes",
        idempotency_key="turn-retry",
    )
    assert result.created is False
    assert result.reminder.status == "scheduled"
    assert [call[0] for call in scheduler.schedule_calls] == [REMINDER_ID, REMINDER_ID]
    assert len(reminders.list()) == 1
    reminders.close()


def test_cancel_and_prefix_resolution_are_idempotent(tmp_path):
    scheduler = FakeScheduler()
    reminders = manager(tmp_path, scheduler)
    reminders.create_from_text(
        "remind me to stop in 10 minutes", idempotency_key="turn-cancel"
    )
    assert reminders.resolve_id(REMINDER_ID[:8]) == REMINDER_ID

    first = reminders.cancel(REMINDER_ID)
    second = reminders.cancel(REMINDER_ID)
    assert first.cancelled is True
    assert second.cancelled is False
    assert scheduler.cancel_calls == [REMINDER_ID]
    reminders.close()


def test_durable_cancel_wins_even_when_transient_unit_is_missing(tmp_path):
    class MissingUnitScheduler(FakeScheduler):
        def cancel(self, reminder_id: str) -> None:
            super().cancel(reminder_id)
            raise ReminderBackendError("unit missing")

    scheduler = MissingUnitScheduler()
    reminders = manager(tmp_path, scheduler)
    reminders.create_from_text(
        "remind me to stop in 10 minutes", idempotency_key="cancel-missing"
    )

    result = reminders.cancel(REMINDER_ID)

    assert result.cancelled is True
    assert reminders.store.get(REMINDER_ID).status == "cancelled"
    assert deliver_reminder(
        reminders.store, REMINDER_ID, FakeNotifier(), clock=lambda: NOW
    ) is False
    reminders.close()


def test_builder_reconciles_pending_and_interrupted_delivery_rows(tmp_path):
    path = str(tmp_path / "recover.sqlite3")
    failing = FakeScheduler(fail_schedule_once=True)
    first = ReminderManager(
        ReminderStore(path), failing, parser=parser(), id_factory=lambda: REMINDER_ID
    )
    with pytest.raises(ReminderBackendError):
        first.create_from_text(
            "remind me to recover in 10 minutes", idempotency_key="recover"
        )
    first.close()

    recovered_scheduler = FakeScheduler()
    recovered = build_reminder_manager(
        ReminderConfig(
            enabled=True,
            store_path=path,
            timezone_name="UTC",
            min_delay_sec=1,
        ),
        scheduler=recovered_scheduler,
    )
    assert recovered is not None
    assert recovered.store.get(REMINDER_ID).status == "scheduled"
    assert [call[0] for call in recovered_scheduler.schedule_calls] == [REMINDER_ID]

    assert recovered.store.claim_delivery(
        REMINDER_ID,
        at=NOW - timedelta(seconds=61),
    ) is not None
    recovered.close()

    interrupted_scheduler = FakeScheduler()
    restarted = build_reminder_manager(
        ReminderConfig(
            enabled=True,
            store_path=path,
            timezone_name="UTC",
            min_delay_sec=1,
        ),
        scheduler=interrupted_scheduler,
        parser=parser(),
    )
    assert restarted is not None
    assert restarted.store.get(REMINDER_ID).status == "scheduled"
    assert [call[0] for call in interrupted_scheduler.schedule_calls] == [REMINDER_ID]
    restarted.close()


def test_fresh_delivery_lease_is_not_recovered_but_stale_claim_retries_once(tmp_path):
    path = str(tmp_path / "delivery-lease.sqlite3")
    creator = ReminderManager(
        ReminderStore(path),
        FakeScheduler(),
        parser=parser(),
        id_factory=lambda: REMINDER_ID,
    )
    creator.create_from_text(
        "remind me to lease in 10 minutes",
        idempotency_key="delivery-lease",
    )
    first_claim = creator.store.claim_delivery(REMINDER_ID, at=NOW)
    assert first_claim is not None
    assert first_claim.delivery_claim_token

    current = [NOW]
    scheduler = FakeScheduler()
    recovery = ReminderManager(
        ReminderStore(path),
        scheduler,
        parser=ReminderParser(
            clock=lambda: current[0],
            local_timezone=timezone.utc,
            min_delay_sec=1,
        ),
        delivery_lease_sec=60,
    )
    recovery.reconcile_stale_deliveries()
    assert recovery.store.get(REMINDER_ID).status == "delivering"
    assert scheduler.schedule_calls == []

    current[0] = NOW + timedelta(seconds=61)
    recovery.reconcile_stale_deliveries()
    assert recovery.store.get(REMINDER_ID).status == "scheduled"
    assert [call[0] for call in scheduler.schedule_calls] == [REMINDER_ID]

    second_claim = recovery.store.claim_delivery(REMINDER_ID, at=current[0])
    assert second_claim is not None
    assert second_claim.delivery_claim_token != first_claim.delivery_claim_token
    assert creator.store.finish_delivery(
        REMINDER_ID,
        claim_token=first_claim.delivery_claim_token,
        at=current[0],
        ok=True,
    ) is False
    assert recovery.store.get(REMINDER_ID).status == "delivering"
    assert recovery.store.finish_delivery(
        REMINDER_ID,
        claim_token=second_claim.delivery_claim_token,
        at=current[0],
        ok=True,
    ) is True

    recovery.reconcile_stale_deliveries()
    assert [call[0] for call in scheduler.schedule_calls] == [REMINDER_ID]
    recovery.close()
    creator.close()


def test_pending_timer_publication_retries_with_bounded_backoff(tmp_path):
    current = [NOW]
    scheduler = FakeScheduler(fail_schedule_once=True)
    reminders = ReminderManager(
        ReminderStore(str(tmp_path / "pending-retry.sqlite3")),
        scheduler,
        parser=ReminderParser(
            clock=lambda: current[0],
            local_timezone=timezone.utc,
            min_delay_sec=1,
        ),
    )
    reminders.store.insert_pending(
        reminder_id=REMINDER_ID,
        idempotency_key="pending-retry",
        message="retry timer",
        due_at=NOW + timedelta(minutes=10),
        created_at=NOW,
    )

    reminders.reconcile_stale_deliveries()
    assert reminders.store.get(REMINDER_ID).status == "pending"
    assert len(scheduler.schedule_calls) == 1
    reminders.reconcile_stale_deliveries()
    assert len(scheduler.schedule_calls) == 1

    current[0] += timedelta(seconds=5)
    reminders.reconcile_stale_deliveries()
    assert reminders.store.get(REMINDER_ID).status == "scheduled"
    assert len(scheduler.schedule_calls) == 2
    reminders.close()


def test_failed_scheduled_timer_recreation_retries_after_backoff(tmp_path):
    path = str(tmp_path / "scheduled-retry.sqlite3")
    creator = ReminderManager(
        ReminderStore(path),
        FakeScheduler(),
        parser=parser(),
        id_factory=lambda: REMINDER_ID,
    )
    creator.create_from_text(
        "remind me to retry startup in 10 minutes",
        idempotency_key="scheduled-retry",
    )
    creator.close()

    current = [NOW]
    scheduler = FakeScheduler(fail_schedule_once=True)
    restarted = ReminderManager(
        ReminderStore(path),
        scheduler,
        parser=ReminderParser(
            clock=lambda: current[0],
            local_timezone=timezone.utc,
            min_delay_sec=1,
        ),
    )
    restarted.reconcile()
    assert restarted.store.get(REMINDER_ID).status == "scheduled"
    assert len(scheduler.schedule_calls) == 1
    restarted.reconcile_stale_deliveries()
    assert len(scheduler.schedule_calls) == 1

    current[0] += timedelta(seconds=5)
    restarted.reconcile_stale_deliveries()
    assert len(scheduler.schedule_calls) == 2
    restarted.reconcile_stale_deliveries()
    assert len(scheduler.schedule_calls) == 2
    restarted.close()


@dataclass
class Completed:
    returncode: int = 0
    stdout: str = ""


def test_systemd_scheduler_uses_fixed_argv_and_never_receives_reminder_text(tmp_path):
    calls: list[tuple[list[str], dict[str, object]]] = []

    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        return Completed()

    scheduler = SystemdReminderScheduler(
        str(tmp_path / "reminders.sqlite3"),
        runner=runner,
        systemd_run="systemd-run",
        systemctl="systemctl",
        helper_command=("python", "/fixed/reminder_notify.py"),
    )
    scheduler.schedule(REMINDER_ID, NOW + timedelta(minutes=10))
    scheduler.cancel(REMINDER_ID)

    schedule_argv = calls[0][0]
    assert schedule_argv == [
        "systemd-run",
        "--user",
        "--no-ask-password",
        "--quiet",
        "--collect",
        f"--unit=speaker-reminder-{REMINDER_ID}",
        "--on-calendar=2026-07-16 12:10:00 UTC",
        "--",
        "python",
        "/fixed/reminder_notify.py",
        "--store",
        str(tmp_path / "reminders.sqlite3"),
        "--reminder-id",
        REMINDER_ID,
    ]
    assert "private reminder text" not in " ".join(schedule_argv)
    assert calls[1][0][-2:] == [
        f"speaker-reminder-{REMINDER_ID}.timer",
        f"speaker-reminder-{REMINDER_ID}.service",
    ]
    assert all(call[1]["check"] is False for call in calls)
    assert all(call[1]["shell"] is False for call in calls)


def test_systemd_scheduler_resolves_ambiguous_failure_by_opaque_unit_id(tmp_path):
    calls: list[list[str]] = []

    def runner(argv, **_kwargs):
        calls.append(argv)
        if argv[0] == "systemd-run":
            return Completed(returncode=1)
        return Completed(stdout="loaded\n")

    scheduler = SystemdReminderScheduler(
        str(tmp_path / "reminders.sqlite3"),
        runner=runner,
        systemd_run="systemd-run",
        systemctl="systemctl",
        helper_command=("python", "/fixed/reminder_notify.py"),
    )
    scheduler.schedule(REMINDER_ID, NOW + timedelta(minutes=10))

    assert calls[1] == [
        "systemctl",
        "--user",
        "--no-ask-password",
        "show",
        "--property=LoadState",
        "--value",
        f"speaker-reminder-{REMINDER_ID}.timer",
    ]


def test_capabilities_enforce_direct_confirmed_live_mutations_but_list_is_read_only(tmp_path):
    scheduler = FakeScheduler()
    reminders = manager(tmp_path, scheduler)
    registry = CapabilityRegistry()
    attach_reminder_capabilities(registry, reminders)

    create_spec = registry.spec("reminder.create")
    list_spec = registry.spec("reminder.list")
    cancel_spec = registry.spec("reminder.cancel")
    assert create_spec is not None and create_spec.authority == "direct_live"
    assert create_spec.requires_confirmation is True and create_spec.planner_tool is False
    assert list_spec is not None and list_spec.authority == "none"
    assert cancel_spec is not None and cancel_spec.authority == "direct_live"

    blocked = registry.invoke(
        "reminder.create",
        "remind me to test in 10 minutes",
        {"origin": Origin.WEB, "direct_user_instruction": True, "confirmed": True, "task_id": "x"},
    )
    assert blocked.data["executed"] is False
    assert scheduler.schedule_calls == []

    allowed_context = {
        "origin": Origin.LIVE_AUDIO,
        "direct_user_instruction": True,
        "confirmed": True,
        "task_id": "voice-turn-1",
    }
    created = registry.invoke(
        "reminder.create", "remind me to test in 10 minutes", allowed_context
    )
    assert created.ok is True and created.data["executed"] is True
    listed = registry.invoke("reminder.list", "list my reminders", {})
    assert listed.ok is True and listed.data["count"] == 1

    blocked_cancel = registry.invoke(
        "reminder.cancel", "", {"reminder_id": REMINDER_ID}
    )
    assert blocked_cancel.data["executed"] is False
    cancelled = registry.invoke(
        "reminder.cancel", "", {**allowed_context, "reminder_id": REMINDER_ID}
    )
    assert cancelled.data["executed"] is True
    reminders.close()


def test_reminder_speech_renders_the_configured_local_timezone(tmp_path):
    reminders = ReminderManager(
        ReminderStore(str(tmp_path / "local-speech.sqlite3")),
        FakeScheduler(),
        parser=ReminderParser(
            clock=lambda: NOW,
            local_timezone=ZoneInfo("Europe/Bucharest"),
            min_delay_sec=1,
        ),
        id_factory=lambda: REMINDER_ID,
    )
    registry = CapabilityRegistry()
    attach_reminder_capabilities(registry, reminders)
    context = {
        "origin": Origin.LIVE_AUDIO,
        "direct_user_instruction": True,
        "confirmed": True,
        "task_id": "local-time",
    }

    created = registry.invoke(
        "reminder.create",
        "remind me to eat dinner at 6 pm",
        context,
    )
    listed = registry.invoke("reminder.list", "which reminders are set?", {})

    assert "6:00 PM EEST" in created.text
    assert "+00:00" not in created.text
    assert "6:00 PM EEST" in listed.text
    assert created.data["due_at"] == "2026-07-16T15:00:00+00:00"
    reminders.close()


class FakeNotifier:
    def __init__(self, *, fail: bool = False) -> None:
        self.messages: list[str] = []
        self.fail = fail

    def notify(self, message: str) -> None:
        self.messages.append(message)
        if self.fail:
            raise RuntimeError("desktop unavailable")


def test_notify_helper_claims_and_delivers_once(tmp_path):
    reminders = manager(tmp_path)
    reminders.create_from_text(
        "remind me to hydrate in 10 minutes", idempotency_key="notify-once"
    )
    notifier = FakeNotifier()

    assert deliver_reminder(reminders.store, REMINDER_ID, notifier, clock=lambda: NOW) is True
    assert deliver_reminder(reminders.store, REMINDER_ID, notifier, clock=lambda: NOW) is False
    assert notifier.messages == ["hydrate"]
    assert reminders.store.get(REMINDER_ID).status == "delivered"
    announced = reminders.claim_voice_announcements()
    assert [record.message for record in announced] == ["hydrate"]
    assert announced[0].voice_claimed_at == NOW
    assert announced[0].voice_claim_token
    assert announced[0].voice_announced_at is None
    assert reminders.claim_voice_announcements() == ()
    assert reminders.renew_voice_claim(
        REMINDER_ID,
        claim_token=announced[0].voice_claim_token,
    ) is True
    assert reminders.mark_voice_announced(
        REMINDER_ID,
        claim_token=announced[0].voice_claim_token,
    ) is True
    assert reminders.store.get(REMINDER_ID).voice_announced_at == NOW
    assert reminders.mark_voice_announced(
        REMINDER_ID,
        claim_token=announced[0].voice_claim_token,
    ) is False
    reminders.close()


def test_unadmitted_voice_claim_expires_and_can_be_retried(tmp_path):
    reminders = manager(tmp_path)
    reminders.create_from_text(
        "remind me to retry speech in 10 minutes",
        idempotency_key="voice-lease",
    )
    assert deliver_reminder(
        reminders.store,
        REMINDER_ID,
        FakeNotifier(),
        clock=lambda: NOW,
    ) is True

    first = reminders.store.claim_voice_announcements(
        since=NOW - timedelta(minutes=10),
        at=NOW,
        claim_stale_before=NOW - timedelta(seconds=60),
    )
    fresh = reminders.store.claim_voice_announcements(
        since=NOW - timedelta(minutes=10),
        at=NOW + timedelta(seconds=59),
        claim_stale_before=NOW - timedelta(seconds=1),
    )
    retried = reminders.store.claim_voice_announcements(
        since=NOW - timedelta(minutes=10),
        at=NOW + timedelta(seconds=61),
        claim_stale_before=NOW + timedelta(seconds=1),
    )

    assert [record.reminder_id for record in first] == [REMINDER_ID]
    assert fresh == ()
    assert [record.reminder_id for record in retried] == [REMINDER_ID]
    assert first[0].voice_claim_token != retried[0].voice_claim_token
    assert reminders.store.renew_voice_claim(
        REMINDER_ID,
        claim_token=first[0].voice_claim_token,
        at=NOW + timedelta(seconds=61),
    ) is False
    assert reminders.store.renew_voice_claim(
        REMINDER_ID,
        claim_token=retried[0].voice_claim_token,
        at=NOW + timedelta(seconds=61),
    ) is True
    assert reminders.store.mark_voice_announced(
        REMINDER_ID,
        claim_token=retried[0].voice_claim_token,
        at=NOW + timedelta(seconds=61),
    ) is True
    reminders.close()


def test_store_migrates_existing_private_database_for_voice_announcements(tmp_path):
    path = tmp_path / "legacy-reminders.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE reminders (
            reminder_id TEXT PRIMARY KEY,
            idempotency_key TEXT NOT NULL UNIQUE,
            message TEXT NOT NULL,
            due_at REAL NOT NULL,
            status TEXT NOT NULL,
            created_at REAL NOT NULL,
            delivered_at REAL,
            cancelled_at REAL
        )
        """
    )
    connection.commit()
    connection.close()

    store = ReminderStore(str(path))
    columns = {
        str(row[1]) for row in store._conn.execute("PRAGMA table_info(reminders)")
    }
    assert {
        "delivery_started_at",
        "delivery_claim_token",
        "voice_claimed_at",
        "voice_claim_token",
        "voice_announced_at",
    } <= columns
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    store.close()


def test_concurrent_legacy_database_open_serializes_schema_migration(tmp_path):
    path = tmp_path / "concurrent-legacy.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE reminders (
            reminder_id TEXT PRIMARY KEY,
            idempotency_key TEXT NOT NULL UNIQUE,
            message TEXT NOT NULL,
            due_at REAL NOT NULL,
            status TEXT NOT NULL,
            created_at REAL NOT NULL,
            delivered_at REAL,
            cancelled_at REAL
        )
        """
    )
    connection.commit()
    connection.close()

    worker_count = 8
    barrier = threading.Barrier(worker_count)
    errors: list[BaseException] = []
    error_lock = threading.Lock()

    def opener() -> None:
        try:
            barrier.wait(timeout=2.0)
            opened = ReminderStore(str(path))
            opened.close()
        except BaseException as exc:  # collected for a deterministic assertion
            with error_lock:
                errors.append(exc)

    threads = [threading.Thread(target=opener) for _ in range(worker_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []


def test_notify_failure_is_generic_and_not_replayed(tmp_path):
    reminders = manager(tmp_path)
    reminders.create_from_text(
        "remind me to private text in 10 minutes", idempotency_key="notify-fail"
    )
    notifier = FakeNotifier(fail=True)
    with pytest.raises(ReminderBackendError, match="notification failed") as caught:
        deliver_reminder(reminders.store, REMINDER_ID, notifier, clock=lambda: NOW)
    assert "private text" not in str(caught.value)
    assert reminders.store.get(REMINDER_ID).status == "delivery_failed"
    assert deliver_reminder(reminders.store, REMINDER_ID, notifier, clock=lambda: NOW) is False
    reminders.close()


def test_notify_send_uses_argv_without_shell():
    calls: list[tuple[list[str], dict[str, object]]] = []

    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        return Completed()

    notifier = NotifySendNotifier(runner=runner, executable="notify-send")
    notifier.notify("literal ; $(not a shell)")
    assert calls[0][0][-1] == "literal ; $(not a shell)"
    assert calls[0][1]["shell"] is False
