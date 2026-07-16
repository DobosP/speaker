from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from always_on_agent.capabilities import (
    CapabilityRegistry,
    create_default_capabilities,
)
from always_on_agent.origin import Origin
from core.reminders import (
    ReminderManager,
    ReminderParser,
    ReminderStore,
    attach_reminder_capabilities,
)
from core.trusted_apps import (
    AppLaunchError,
    DeviceToolCommandDispatcher,
    GtkAppLauncher,
    TrustedAppError,
    TrustedAppManager,
    TrustedAppsConfig,
    attach_device_tool_command_dispatcher,
    attach_trusted_app_capability,
)


NOW = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
REMINDER_ID = "fedcba9876543210fedcba9876543210"


def config() -> TrustedAppsConfig:
    return TrustedAppsConfig.from_dict(
        {
            "enabled": True,
            "apps": {
                "browser": {
                    "connector": "desktop_launch",
                    "desktop_id": "firefox.desktop",
                    "operations": ["open"],
                },
                "notes": {
                    "connector": "desktop_launch",
                    "desktop_id": "md.obsidian.Obsidian.desktop",
                    "operations": ["open"],
                },
            },
        }
    )


@pytest.mark.parametrize(
    "entry",
    [
        {"connector": "shell", "desktop_id": "firefox.desktop", "operations": ["open"]},
        {"connector": "desktop_launch", "desktop_id": "/tmp/firefox.desktop", "operations": ["open"]},
        {"connector": "desktop_launch", "desktop_id": "firefox.desktop --url", "operations": ["open"]},
        {"connector": "desktop_launch", "desktop_id": "https:example.desktop", "operations": ["open"]},
        {"connector": "desktop_launch", "desktop_id": "firefox.desktop", "operations": ["open", "uri"]},
    ],
)
def test_config_rejects_non_desktop_connectors_paths_args_uris_and_extra_operations(entry):
    with pytest.raises(TrustedAppError):
        TrustedAppsConfig.from_dict(
            {"enabled": True, "apps": {"browser": entry}}
        )


def test_config_is_explicit_opt_in_and_normalizes_only_the_exact_alias():
    assert TrustedAppsConfig.from_dict(None).enabled is False
    parsed = config()
    assert parsed.enabled is True
    assert [app.alias for app in parsed.apps] == ["browser", "notes"]
    with pytest.raises(TrustedAppError, match="at least one"):
        TrustedAppsConfig.from_dict({"enabled": True, "apps": {}})


@dataclass
class Completed:
    returncode: int = 0


def test_gtk_launcher_passes_exactly_one_validated_desktop_id_without_shell():
    calls: list[tuple[list[str], dict[str, object]]] = []

    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        return Completed()

    launcher = GtkAppLauncher(runner=runner, executable="gtk-launch")
    launcher.open("firefox.desktop")
    assert calls == [
        (
            ["gtk-launch", "firefox.desktop"],
            {
                "capture_output": True,
                "text": True,
                "timeout": 8.0,
                "check": False,
                "shell": False,
            },
        )
    ]
    with pytest.raises(TrustedAppError):
        launcher.open("firefox.desktop https://example.com")


class FakeLauncher:
    def __init__(self, *, fail: bool = False) -> None:
        self.desktop_ids: list[str] = []
        self.fail = fail

    def open(self, desktop_id: str) -> None:
        self.desktop_ids.append(desktop_id)
        if self.fail:
            raise AppLaunchError("trusted app launch failed")


def test_manager_requires_exact_alias_and_suppresses_same_turn_replay():
    launcher = FakeLauncher()
    apps = TrustedAppManager(config(), launcher)
    assert apps.aliases == ("browser", "notes")
    assert apps.match_open_command("open the browser app") == "browser"
    assert apps.match_open_command("open browser?") == "browser"
    assert apps.match_open_command("launch notes") == "notes"
    assert apps.match_open_command("open browser https://example.com") is None
    assert apps.match_open_command("open web browser") is None

    first = apps.open("BROWSER", idempotency_key="turn-1")
    replay = apps.open("browser", idempotency_key="turn-1")
    assert first.launched is True and replay.launched is False
    assert launcher.desktop_ids == ["firefox.desktop"]
    with pytest.raises(TrustedAppError, match="another app"):
        apps.open("notes", idempotency_key="turn-1")


def live_context(task_id: str = "voice-1") -> dict[str, object]:
    return {
        "origin": Origin.LIVE_AUDIO,
        "direct_user_instruction": True,
        "confirmed": True,
        "task_id": task_id,
    }


def test_app_capability_is_non_planner_and_requires_direct_confirmed_live_audio():
    launcher = FakeLauncher()
    apps = TrustedAppManager(config(), launcher)
    registry = CapabilityRegistry()
    attach_trusted_app_capability(registry, apps)

    spec = registry.spec("app.open")
    assert spec is not None
    assert spec.side_effecting is True and spec.planner_tool is False
    assert spec.authority == "direct_live" and spec.requires_confirmation is True

    for context in (
        {},
        {"origin": Origin.WEB, "direct_user_instruction": True, "confirmed": True},
        {"origin": Origin.LIVE_AUDIO, "direct_user_instruction": True, "confirmed": False},
    ):
        result = registry.invoke("app.open", "open notes", context)
        assert result.data["executed"] is False
    assert launcher.desktop_ids == []

    result = registry.invoke("app.open", "open notes", live_context())
    assert result.ok is True and result.data["executed"] is True
    assert launcher.desktop_ids == ["md.obsidian.Obsidian.desktop"]


class FakeScheduler:
    def __init__(self) -> None:
        self.scheduled: list[str] = []
        self.cancelled: list[str] = []

    def schedule(self, reminder_id, _due_at):
        self.scheduled.append(reminder_id)

    def cancel(self, reminder_id):
        self.cancelled.append(reminder_id)


def reminder_manager(tmp_path) -> ReminderManager:
    return ReminderManager(
        ReminderStore(str(tmp_path / "reminders.sqlite3")),
        FakeScheduler(),
        parser=ReminderParser(
            clock=lambda: NOW,
            local_timezone=timezone.utc,
            min_delay_sec=1,
            max_horizon_days=30,
        ),
        id_factory=lambda: REMINDER_ID,
    )


def test_typed_dispatcher_routes_only_reminders_and_exact_allowlisted_app_aliases(tmp_path):
    reminders = reminder_manager(tmp_path)
    launcher = FakeLauncher()
    apps = TrustedAppManager(config(), launcher)
    registry = CapabilityRegistry()
    attach_reminder_capabilities(registry, reminders)
    attach_trusted_app_capability(registry, apps)
    dispatcher = attach_device_tool_command_dispatcher(
        registry,
        reminder_manager=reminders,
        trusted_app_manager=apps,
    )
    assert isinstance(dispatcher, DeviceToolCommandDispatcher)
    command_spec = registry.spec("device.command")
    assert command_spec is not None and command_spec.authority == "none"
    assert command_spec.planner_tool is False
    assert dispatcher.match("open calculator") is None
    assert dispatcher.match("open browser https://example.com") is None
    for phrase in (
        "show me the reminders",
        "which reminders are set?",
        "what reminders are active?",
        "do I have any reminders?",
    ):
        command = dispatcher.match(phrase)
        assert command is not None
        assert command.capability_name == "reminder.list"

    created = registry.invoke(
        "device.command",
        "remind me to hydrate in 10 minutes",
        live_context("voice-create"),
    )
    assert created.ok is True and created.data["reminder_id"] == REMINDER_ID

    listed = registry.invoke("device.command", "what reminders do I have", {})
    assert listed.ok is True and listed.data["count"] == 1

    opened = registry.invoke(
        "device.command", "please open the browser app", live_context("voice-open")
    )
    assert opened.ok is True and launcher.desktop_ids == ["firefox.desktop"]

    unsafe = registry.invoke(
        "device.command",
        "open browser https://example.com",
        live_context("voice-unsafe"),
    )
    arbitrary = registry.invoke(
        "device.command", "delete all my files", live_context("voice-arbitrary")
    )
    assert unsafe.ok is False and unsafe.data["executed"] is False
    assert arbitrary.ok is False and arbitrary.data["handled"] is False
    assert launcher.desktop_ids == ["firefox.desktop"]

    cancelled = registry.invoke(
        "device.command", "cancel my next reminder", live_context("voice-cancel")
    )
    assert cancelled.ok is True and cancelled.data["executed"] is True
    assert reminders.list() == ()

    missing = registry.invoke(
        "device.command", "cancel my next reminder", live_context("voice-missing")
    )
    assert missing.ok is True
    assert missing.data == {"executed": False, "missing": True}
    reminders.close()


def test_dispatcher_is_not_attached_when_no_device_tools_are_enabled():
    registry = CapabilityRegistry()
    result = attach_device_tool_command_dispatcher(registry)
    assert result is None
    assert "device.command" not in registry.names()


def test_attaching_device_dispatcher_does_not_replace_generic_command_stage(tmp_path):
    reminders = reminder_manager(tmp_path)
    registry = create_default_capabilities()
    attach_reminder_capabilities(registry, reminders)

    attach_device_tool_command_dispatcher(
        registry,
        reminder_manager=reminders,
    )

    generic = registry.invoke("command.stage", "existing workflow", {})
    assert generic.ok is True
    assert generic.data == {"requires_confirmation": True}
    assert registry.spec("command.stage") is not None
    assert registry.spec("device.command") is not None
    reminders.close()
