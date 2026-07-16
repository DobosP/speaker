from __future__ import annotations

from datetime import datetime, timezone

from core.engine import FinalTranscript
from core.engines.scripted import ScriptedEngine
from core.reminders import (
    ReminderConfig,
    ReminderManager,
    ReminderParser,
    ReminderStore,
)
from core.runtime import VoiceRuntime
from core.trusted_apps import (
    TrustedAppManager,
    TrustedAppsConfig,
)
from tools.reminder_notify import deliver_reminder


class _RecordingScheduler:
    def __init__(self) -> None:
        self.scheduled: list[tuple[str, datetime]] = []
        self.cancelled: list[str] = []

    def schedule(self, reminder_id: str, due_at: datetime) -> None:
        self.scheduled.append((reminder_id, due_at))

    def cancel(self, reminder_id: str) -> None:
        self.cancelled.append(reminder_id)


class _RecordingLauncher:
    def __init__(self) -> None:
        self.desktop_ids: list[str] = []

    def open(self, desktop_id: str) -> None:
        self.desktop_ids.append(desktop_id)


class _RecordingNotifier:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def notify(self, message: str) -> None:
        self.messages.append(message)


class _AlwaysIngestAddressing:
    def classify(self, text: str, *, recent=()):
        from core.addressing import INGEST

        return INGEST


class _PunctuationOnlyCleaner:
    def clean(self, text: str, *, recent=()):
        return "Yes." if text.casefold() == "yes" else text


class _ActionRewritingCleaner:
    def clean(self, text: str, *, recent=()):
        if text == "open the obsidian app":
            return "open obsidian"
        return text


class _ActionErasingCleaner:
    def clean(self, text: str, *, recent=()):
        if text == "open obsidian":
            return ""
        return text


class _ActionManufacturingCleaner:
    def clean(self, text: str, *, recent=()):
        if text == '"open obsidian"':
            return "open obsidian"
        return text


def _live(runtime: VoiceRuntime, text: str) -> None:
    runtime._on_final_result(FinalTranscript(text=text, origin="live_audio"))
    assert runtime.wait_idle()


def test_normal_runtime_stages_and_executes_typed_reminder_tools(
    tmp_path, monkeypatch
):
    scheduler = _RecordingScheduler()
    now = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)

    def build(config: ReminderConfig):
        return ReminderManager(
            ReminderStore(config.store_path),
            scheduler,
            parser=ReminderParser(
                clock=lambda: now,
                local_timezone=timezone.utc,
                min_delay_sec=1,
            ),
            id_factory=lambda: "a" * 32,
        )

    monkeypatch.setattr("core.runtime.build_reminder_manager", build)
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        reminder_config=ReminderConfig(
            enabled=True,
            store_path=str(tmp_path / "reminders.sqlite3"),
            min_delay_sec=1,
        ),
    )
    runtime.start(run_bus=False)
    try:
        _live(runtime, "remind me to call Ana in ten minutes")
        assert scheduler.scheduled == []
        assert len(runtime.supervisor.state.pending_confirmations) == 1
        assert any(
            text.startswith("Confirm command: remind me") for text in engine.spoken
        )

        _live(runtime, "confirm")
        assert len(scheduler.scheduled) == 1
        assert scheduler.scheduled[0][1] == datetime(
            2026, 7, 16, 12, 10, tzinfo=timezone.utc
        )

        # Listing is read-only: it runs immediately through the same normal
        # chatbot/controller without opening another confirmation.
        _live(runtime, "which reminders are set?")
        assert not runtime.supervisor.state.pending_confirmations
        assert any(text.startswith("Active reminders:") for text in engine.spoken)
        assert "list active local reminders" in runtime._system_prompt
        assert "set a durable local reminder" not in runtime._system_prompt

        # The durable helper owns delivery even if the voice process is down.
        # When it is running, the watchdog claims that delivery exactly once
        # and speaks it through the normal cancellable auxiliary-TTS path.
        notifier = _RecordingNotifier()
        assert deliver_reminder(
            runtime._reminder_manager.store,
            "a" * 32,
            notifier,
            clock=lambda: now,
        ) is True
        runtime._on_watchdog_tick()
        assert runtime.wait_idle()
        assert notifier.messages == ["call Ana"]
        assert engine.spoken.count("Reminder: call Ana") == 1
        runtime._on_watchdog_tick()
        assert runtime.wait_idle()
        assert engine.spoken.count("Reminder: call Ana") == 1
    finally:
        runtime.stop()


def test_cancelled_due_voice_request_retries_after_its_claim_lease(
    tmp_path, monkeypatch
):
    scheduler = _RecordingScheduler()
    current = [datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)]

    def build(config: ReminderConfig):
        return ReminderManager(
            ReminderStore(config.store_path),
            scheduler,
            parser=ReminderParser(
                clock=lambda: current[0],
                local_timezone=timezone.utc,
                min_delay_sec=1,
            ),
            id_factory=lambda: "b" * 32,
        )

    monkeypatch.setattr("core.runtime.build_reminder_manager", build)
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        reminder_config=ReminderConfig(
            enabled=True,
            store_path=str(tmp_path / "retry-reminders.sqlite3"),
            min_delay_sec=1,
        ),
    )
    runtime.start(run_bus=False)
    try:
        runtime._reminder_manager.create(
            "retry voice",
            current[0].replace(minute=10),
            idempotency_key="retry-voice",
        )
        assert deliver_reminder(
            runtime._reminder_manager.store,
            "b" * 32,
            _RecordingNotifier(),
            clock=lambda: current[0],
        ) is True

        runtime._on_watchdog_tick()
        runtime.supervisor.cancel_pending_aux_tts()
        assert runtime.wait_idle()
        assert "Reminder: retry voice" not in engine.spoken
        assert (
            runtime._reminder_manager.store.get("b" * 32).voice_announced_at
            is None
        )

        current[0] = current[0].replace(minute=1, second=1)
        runtime._on_watchdog_tick()
        assert runtime.wait_idle()
        assert engine.spoken.count("Reminder: retry voice") == 1
        assert (
            runtime._reminder_manager.store.get("b" * 32).voice_announced_at
            == current[0]
        )
    finally:
        runtime.stop()


def test_normal_runtime_opens_only_setup_allowlisted_app_after_confirmation(
    monkeypatch,
):
    config = TrustedAppsConfig.from_dict(
        {
            "enabled": True,
            "apps": {
                "obsidian": {
                    "connector": "desktop_launch",
                    "desktop_id": "obsidian.desktop",
                    "operations": ["open"],
                }
            },
        }
    )
    launcher = _RecordingLauncher()
    manager = TrustedAppManager(config, launcher)
    monkeypatch.setattr(
        "core.runtime.build_trusted_app_manager", lambda _config: manager
    )
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        trusted_apps_config=config,
        addressing=_AlwaysIngestAddressing(),
        unsure_acts=False,
        cleaner=_PunctuationOnlyCleaner(),
    )
    runtime.start(run_bus=False)
    try:
        _live(runtime, "open obsidian?")
        assert launcher.desktop_ids == []
        assert len(runtime.supervisor.state.pending_confirmations) == 1

        _live(runtime, "yes")
        assert launcher.desktop_ids == ["obsidian.desktop"]
        assert "open an explicitly allowlisted desktop app" not in runtime._system_prompt
        assert "configured local tools listed above" in runtime._system_prompt

        _live(runtime, "open calculator")
        _live(runtime, "confirm")
        assert launcher.desktop_ids == ["obsidian.desktop"]
    finally:
        runtime.stop()


def test_live_kws_confirmation_can_approve_a_direct_typed_tool(monkeypatch):
    config = TrustedAppsConfig.from_dict(
        {
            "enabled": True,
            "apps": {
                "obsidian": {
                    "connector": "desktop_launch",
                    "desktop_id": "obsidian.desktop",
                    "operations": ["open"],
                }
            },
        }
    )
    launcher = _RecordingLauncher()
    manager = TrustedAppManager(config, launcher)
    monkeypatch.setattr(
        "core.runtime.build_trusted_app_manager", lambda _config: manager
    )
    engine = ScriptedEngine()
    runtime = VoiceRuntime(
        engine,
        trusted_apps_config=config,
        command_map={"yes do it": "confirm"},
    )
    runtime.start(run_bus=False)
    try:
        _live(runtime, "open obsidian")
        assert len(runtime.supervisor.state.pending_confirmations) == 1

        engine.command("yes do it")
        assert runtime.wait_idle()
        assert launcher.desktop_ids == ["obsidian.desktop"]
        assert not runtime.supervisor.state.pending_confirmations
    finally:
        runtime.stop()


def test_model_token_rewrite_cannot_retain_device_action_authority(monkeypatch):
    config = TrustedAppsConfig.from_dict(
        {
            "enabled": True,
            "apps": {
                "obsidian": {
                    "connector": "desktop_launch",
                    "desktop_id": "obsidian.desktop",
                    "operations": ["open"],
                }
            },
        }
    )
    launcher = _RecordingLauncher()
    manager = TrustedAppManager(config, launcher)
    monkeypatch.setattr(
        "core.runtime.build_trusted_app_manager", lambda _config: manager
    )
    runtime = VoiceRuntime(
        ScriptedEngine(),
        trusted_apps_config=config,
        addressing=_AlwaysIngestAddressing(),
        unsure_acts=False,
        cleaner=_ActionRewritingCleaner(),
    )
    runtime.start(run_bus=False)
    try:
        _live(runtime, "open the obsidian app")
        assert len(runtime.supervisor.state.pending_confirmations) == 1
        _live(runtime, "yes")
        assert launcher.desktop_ids == []
    finally:
        runtime.stop()


def test_model_empty_rewrite_cannot_retain_device_action_authority(monkeypatch):
    config = TrustedAppsConfig.from_dict(
        {
            "enabled": True,
            "apps": {
                "obsidian": {
                    "connector": "desktop_launch",
                    "desktop_id": "obsidian.desktop",
                    "operations": ["open"],
                }
            },
        }
    )
    launcher = _RecordingLauncher()
    manager = TrustedAppManager(config, launcher)
    monkeypatch.setattr(
        "core.runtime.build_trusted_app_manager", lambda _config: manager
    )
    runtime = VoiceRuntime(
        ScriptedEngine(),
        trusted_apps_config=config,
        cleaner=_ActionErasingCleaner(),
    )
    runtime.start(run_bus=False)
    try:
        _live(runtime, "open obsidian")
        assert len(runtime.supervisor.state.pending_confirmations) == 1
        _live(runtime, "yes")
        assert launcher.desktop_ids == []
    finally:
        runtime.stop()


def test_cleaner_cannot_manufacture_an_exact_device_command(monkeypatch):
    config = TrustedAppsConfig.from_dict(
        {
            "enabled": True,
            "apps": {
                "obsidian": {
                    "connector": "desktop_launch",
                    "desktop_id": "obsidian.desktop",
                    "operations": ["open"],
                }
            },
        }
    )
    launcher = _RecordingLauncher()
    manager = TrustedAppManager(config, launcher)
    monkeypatch.setattr(
        "core.runtime.build_trusted_app_manager", lambda _config: manager
    )
    runtime = VoiceRuntime(
        ScriptedEngine(),
        trusted_apps_config=config,
        cleaner=_ActionManufacturingCleaner(),
    )
    runtime.start(run_bus=False)
    try:
        _live(runtime, '"open obsidian"')
        assert len(runtime.supervisor.state.pending_confirmations) == 1
        _live(runtime, "yes")
        assert launcher.desktop_ids == []
    finally:
        runtime.stop()


def test_reminder_phrase_is_ordinary_chat_when_tool_was_not_enabled():
    engine = ScriptedEngine()
    runtime = VoiceRuntime(engine)
    runtime.start(run_bus=False)
    try:
        _live(runtime, "remind me to stretch in ten minutes")
        assert not runtime.supervisor.state.pending_confirmations
        assert "set a durable local reminder" not in runtime._system_prompt
    finally:
        runtime.stop()
