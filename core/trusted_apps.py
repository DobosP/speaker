"""Allowlisted desktop-app launch tools for the unified voice agent.

Only an exact configured spoken alias can select an application.  The alias is
resolved to a setup-time ``.desktop`` id, and the launcher receives exactly one
argument: no URI, filename, option, shell fragment, or model-produced trailing
text can reach the process boundary.

This module also contains the deliberately small internal ``device.command``
dispatcher used by the normal controller.  It recognizes only typed reminder
and trusted-app commands, then delegates through :class:`CapabilityRegistry` so
the target capability's central authority policy remains the enforcement
boundary.  The historical ``command.stage`` capability remains untouched.
"""
from __future__ import annotations

import re
import subprocess
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Literal, Mapping, Optional, Protocol, Sequence

from always_on_agent.capabilities import (
    CapabilityRegistry,
    CapabilityResult,
    CapabilitySpec,
)
from always_on_agent.origin import Origin

from .reminders import ParsedReminder, ReminderError, ReminderManager


_ALIAS_MAX_CHARS = 64
_IDEMPOTENCY_MAX_CHARS = 240
_ALIAS_RE = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_DESKTOP_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,254}\.desktop$")
_OPEN_PREFIX_RE = re.compile(
    r"^(?:please\s+)?(?:open|launch|start)\s+(?P<target>.+?)\s*[.!?]?\s*$",
    re.IGNORECASE,
)
_LIST_REMINDERS_RE = re.compile(
    r"^(?:please\s+)?(?:"
    r"(?:list|show)(?:\s+me)?\s+(?:(?:my|the|active)\s+){0,2}reminders?"
    r"|(?:what|which)\s+reminders?\s+(?:do\s+i\s+have|are\s+(?:active|set))"
    r"|do\s+i\s+have\s+any\s+reminders?"
    r")\s*[.!?]?\s*$",
    re.IGNORECASE,
)
_CANCEL_REMINDER_RE = re.compile(
    r"^(?:please\s+)?(?:cancel|delete|remove)\s+(?:my\s+|the\s+)?reminder\s+"
    r"(?P<reference>[0-9a-f]{8,32})\s*[.!?]?\s*$",
    re.IGNORECASE,
)
_CANCEL_NEXT_RE = re.compile(
    r"^(?:please\s+)?(?:cancel|delete|remove)\s+(?:my\s+|the\s+)?"
    r"(?:next|first)\s+reminder\s*[.!?]?\s*$",
    re.IGNORECASE,
)


class TrustedAppError(ValueError):
    """A trusted-app configuration or request is invalid."""


class AppLaunchError(RuntimeError):
    """The allowlisted local app launcher failed."""


def _normal_alias(value: object) -> str:
    alias = " ".join(str(value or "").casefold().split())
    if not alias or len(alias) > _ALIAS_MAX_CHARS or _ALIAS_RE.fullmatch(alias) is None:
        raise TrustedAppError("trusted app alias is invalid")
    return alias


def _desktop_id(value: object) -> str:
    desktop_id = str(value or "").strip()
    if (
        not desktop_id
        or len(desktop_id) > 263
        or desktop_id.startswith("-")
        or ".." in desktop_id
        or "/" in desktop_id
        or "\\" in desktop_id
        or _DESKTOP_ID_RE.fullmatch(desktop_id) is None
    ):
        raise TrustedAppError("trusted app desktop_id is invalid")
    return desktop_id


def _idempotency_key(value: object) -> str:
    key = str(value or "").strip()
    if not key or len(key) > _IDEMPOTENCY_MAX_CHARS:
        raise TrustedAppError("a bounded idempotency key is required")
    if any(ord(char) < 32 for char in key):
        raise TrustedAppError("idempotency key contains control characters")
    return key


@dataclass(frozen=True)
class TrustedApp:
    alias: str
    desktop_id: str
    connector: str = "desktop_launch"
    operations: tuple[str, ...] = ("open",)

    @classmethod
    def from_entry(cls, alias: object, data: object) -> "TrustedApp":
        if not isinstance(data, Mapping):
            raise TrustedAppError("trusted app entry must be an object")
        connector = str(data.get("connector", "") or "")
        if connector != "desktop_launch":
            raise TrustedAppError("trusted app connector must be desktop_launch")
        raw_operations = data.get("operations", ())
        if (
            isinstance(raw_operations, (str, bytes))
            or not isinstance(raw_operations, Sequence)
        ):
            raise TrustedAppError("trusted app operations must be a list")
        operations = tuple(str(item) for item in raw_operations)
        if operations != ("open",):
            raise TrustedAppError("trusted app operations must contain only open")
        return cls(
            alias=_normal_alias(alias),
            desktop_id=_desktop_id(data.get("desktop_id")),
            connector=connector,
            operations=operations,
        )


@dataclass(frozen=True)
class TrustedAppsConfig:
    """Explicit setup-time allowlist; missing/false means no launch surface."""

    enabled: bool = False
    apps: tuple[TrustedApp, ...] = ()

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, object]]) -> "TrustedAppsConfig":
        data = data if isinstance(data, Mapping) else {}
        raw_apps = data.get("apps", {})
        if not isinstance(raw_apps, Mapping):
            raise TrustedAppError("trusted_apps.apps must be an object")
        apps: list[TrustedApp] = []
        aliases: set[str] = set()
        for raw_alias, entry in raw_apps.items():
            app = TrustedApp.from_entry(raw_alias, entry)
            if app.alias in aliases:
                raise TrustedAppError("trusted app aliases must be unique")
            aliases.add(app.alias)
            apps.append(app)
        enabled = data.get("enabled") is True
        if enabled and not apps:
            raise TrustedAppError("enabled trusted apps require at least one app")
        return cls(enabled=enabled, apps=tuple(apps))


@dataclass(frozen=True)
class AppLaunchResult:
    alias: str
    desktop_id: str
    launched: bool


class AppLauncher(Protocol):
    def open(self, desktop_id: str) -> None: ...


Runner = Callable[..., object]


class GtkAppLauncher:
    """Launch one validated desktop id through a fixed argv, never a shell."""

    def __init__(
        self,
        *,
        runner: Runner = subprocess.run,
        executable: str = "/usr/bin/gtk-launch",
        timeout_sec: float = 8.0,
    ) -> None:
        self._runner = runner
        self._executable = str(executable)
        self._timeout = max(1.0, min(30.0, float(timeout_sec)))

    def argv(self, desktop_id: str) -> tuple[str, str]:
        return self._executable, _desktop_id(desktop_id)

    def open(self, desktop_id: str) -> None:
        try:
            result = self._runner(
                list(self.argv(desktop_id)),
                capture_output=True,
                text=True,
                timeout=self._timeout,
                check=False,
                shell=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise AppLaunchError("trusted app launch failed") from exc
        if int(getattr(result, "returncode", 1)) != 0:
            raise AppLaunchError("trusted app launch failed")


class TrustedAppManager:
    """Resolve exact aliases and suppress duplicate launches within one process."""

    def __init__(
        self,
        config: TrustedAppsConfig,
        launcher: AppLauncher,
        *,
        max_idempotency_entries: int = 512,
    ) -> None:
        if not config.enabled:
            raise TrustedAppError("trusted app manager requires an enabled config")
        validated: dict[str, TrustedApp] = {}
        for raw_app in config.apps:
            app = TrustedApp.from_entry(
                raw_app.alias,
                {
                    "connector": raw_app.connector,
                    "desktop_id": raw_app.desktop_id,
                    "operations": raw_app.operations,
                },
            )
            if app.alias in validated:
                raise TrustedAppError("trusted app aliases must be unique")
            validated[app.alias] = app
        if not validated:
            raise TrustedAppError("trusted app manager requires at least one app")
        self._apps = validated
        self._launcher = launcher
        self._max_entries = max(1, min(4096, int(max_idempotency_entries)))
        self._completed: OrderedDict[str, str] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def aliases(self) -> tuple[str, ...]:
        return tuple(sorted(self._apps))

    def resolve(self, alias: object) -> TrustedApp:
        normalized = _normal_alias(alias)
        app = self._apps.get(normalized)
        if app is None:
            raise TrustedAppError("app alias is not in the trusted allowlist")
        return app

    def match_open_command(self, text: str) -> str | None:
        """Return the configured alias only for an exact bounded open command."""

        match = _OPEN_PREFIX_RE.fullmatch(text or "")
        if match is None:
            return None
        target = " ".join(match.group("target").casefold().split())
        candidates = [target]
        if target.startswith("the "):
            candidates.append(target[4:])
        for candidate in tuple(candidates):
            if candidate.endswith(" app"):
                candidates.append(candidate[:-4].rstrip())
            if candidate.endswith(" application"):
                candidates.append(candidate[:-12].rstrip())
        for candidate in candidates:
            if candidate in self._apps:
                return candidate
        return None

    def open(self, alias: object, *, idempotency_key: object) -> AppLaunchResult:
        app = self.resolve(alias)
        key = _idempotency_key(idempotency_key)
        with self._lock:
            previous = self._completed.get(key)
            if previous is not None:
                if previous != app.alias:
                    raise TrustedAppError(
                        "idempotency key was already used for another app launch"
                    )
                self._completed.move_to_end(key)
                return AppLaunchResult(app.alias, app.desktop_id, False)
            self._launcher.open(app.desktop_id)
            self._completed[key] = app.alias
            while len(self._completed) > self._max_entries:
                self._completed.popitem(last=False)
        return AppLaunchResult(app.alias, app.desktop_id, True)


def build_trusted_app_manager(
    config: TrustedAppsConfig,
    *,
    launcher: AppLauncher | None = None,
    runner: Runner = subprocess.run,
) -> TrustedAppManager | None:
    if not config.enabled:
        return None
    return TrustedAppManager(config, launcher or GtkAppLauncher(runner=runner))


def _authorized_mutation(context: Mapping[str, object]) -> bool:
    origin = context.get("origin", Origin.UNKNOWN)
    if isinstance(origin, Origin):
        origin = origin.value
    return bool(
        origin == Origin.LIVE_AUDIO.value
        and context.get("direct_user_instruction") is True
        and context.get("confirmed") is True
    )


def attach_trusted_app_capability(
    registry: CapabilityRegistry,
    manager: TrustedAppManager,
) -> CapabilityRegistry:
    """Attach the single side-effecting ``app.open`` provider."""

    def provider(query: str, context: dict[str, object]) -> CapabilityResult:
        if not _authorized_mutation(context):
            return CapabilityResult(
                True,
                "I need a direct, confirmed live request before opening an app.",
                data={"executed": False, "blocked": "action_authorization"},
            )
        alias = context.get("app_alias")
        if alias is None:
            alias = manager.match_open_command(query)
        if alias is None:
            return CapabilityResult(
                False,
                "",
                data={"executed": False},
                error="unsupported trusted app command",
            )
        key = context.get("idempotency_key") or context.get("action_id") or context.get("task_id")
        try:
            result = manager.open(alias, idempotency_key=key)
        except (TrustedAppError, AppLaunchError) as exc:
            return CapabilityResult(
                False,
                "",
                data={"executed": False},
                error=str(exc),
            )
        return CapabilityResult(
            True,
            f"Opening {result.alias}." if result.launched else f"{result.alias} is already opening.",
            data={
                "executed": result.launched,
                "idempotent_replay": not result.launched,
                "app_alias": result.alias,
                "desktop_id": result.desktop_id,
            },
        )

    registry.register(
        "app.open",
        provider,
        spec=CapabilitySpec(
            name="app.open",
            summary="open an explicitly allowlisted desktop app after confirmation",
            when_to_use="when the user directly asks to open a configured trusted app",
            egress="local",
            speaks=True,
            side_effecting=True,
            planner_tool=False,
            # Execution acknowledgements come only from this provider.  Keep
            # the mutation out of assistant.answer's self-description so a
            # model cannot fabricate "Opening ..." on an unroutable transcript.
            user_facing=False,
            authority="direct_live",
            requires_confirmation=True,
        ),
    )
    return registry


# Plural spelling for callers that group integrations by feature family.
attach_trusted_app_capabilities = attach_trusted_app_capability


@dataclass(frozen=True)
class DeviceToolCommand:
    """One strict device-command match and its typed provider arguments."""

    capability_name: Literal["reminder.create", "reminder.list", "reminder.cancel", "app.open"]
    query: str
    reminder_request: ParsedReminder | None = None
    reminder_id: str = ""
    app_alias: str = ""

    def context_updates(self) -> dict[str, object]:
        updates: dict[str, object] = {}
        if self.reminder_request is not None:
            updates["reminder_request"] = self.reminder_request
        if self.reminder_id:
            updates["reminder_id"] = self.reminder_id
        if self.app_alias:
            updates["app_alias"] = self.app_alias
        return updates


class DeviceToolCommandDispatcher:
    """Match only supported device-tool commands and invoke their typed provider."""

    def __init__(
        self,
        registry: CapabilityRegistry,
        *,
        reminder_manager: ReminderManager | None = None,
        trusted_app_manager: TrustedAppManager | None = None,
    ) -> None:
        if reminder_manager is None and trusted_app_manager is None:
            raise ValueError("device tool dispatcher requires an enabled tool")
        self.registry = registry
        self.reminder_manager = reminder_manager
        self.trusted_app_manager = trusted_app_manager

    def match(self, query: str) -> DeviceToolCommand | None:
        text = (query or "").strip()
        if self.reminder_manager is not None:
            if _LIST_REMINDERS_RE.fullmatch(text):
                return DeviceToolCommand("reminder.list", text)

            cancel = _CANCEL_REMINDER_RE.fullmatch(text)
            if cancel is not None:
                try:
                    reminder_id = self.reminder_manager.resolve_id(
                        cancel.group("reference").casefold()
                    )
                except ReminderError:
                    reminder_id = ""
                return DeviceToolCommand(
                    "reminder.cancel", text, reminder_id=reminder_id
                )

            if _CANCEL_NEXT_RE.fullmatch(text):
                active = self.reminder_manager.list()
                return DeviceToolCommand(
                    "reminder.cancel",
                    text,
                    reminder_id=active[0].reminder_id if active else "",
                )

            try:
                parsed = self.reminder_manager.parser.parse(text)
            except ReminderError:
                parsed = None
            if parsed is not None:
                return DeviceToolCommand(
                    "reminder.create", text, reminder_request=parsed
                )

        if self.trusted_app_manager is not None:
            alias = self.trusted_app_manager.match_open_command(text)
            if alias is not None:
                return DeviceToolCommand(
                    "app.open", text, app_alias=alias
                )
        return None

    def dispatch(self, query: str, context: dict[str, object]) -> CapabilityResult:
        metadata = context.get("metadata")
        prepared = (
            metadata.get("prepared_device_command")
            if isinstance(metadata, Mapping)
            else None
        )
        command = (
            prepared
            if isinstance(prepared, DeviceToolCommand) and prepared.query == query
            else self.match(query)
        )
        if command is None:
            return CapabilityResult(
                False,
                "",
                data={"handled": False, "executed": False},
                error="unsupported device tool command",
            )
        delegated = dict(context)
        delegated.update(command.context_updates())
        return self.registry.invoke(command.capability_name, command.query, delegated)


def attach_device_tool_command_dispatcher(
    registry: CapabilityRegistry,
    *,
    reminder_manager: ReminderManager | None = None,
    trusted_app_manager: TrustedAppManager | None = None,
    capability_name: str = "device.command",
) -> DeviceToolCommandDispatcher | None:
    """Attach the internal dispatcher only when at least one tool is enabled.

    The dispatcher itself has no action authority.  It cannot perform an action;
    it only delegates a strict typed match to target providers whose specs apply
    the appropriate direct-live/confirmation policy.
    """

    if reminder_manager is None and trusted_app_manager is None:
        return None
    dispatcher = DeviceToolCommandDispatcher(
        registry,
        reminder_manager=reminder_manager,
        trusted_app_manager=trusted_app_manager,
    )
    registry.register(
        capability_name,
        dispatcher.dispatch,
        spec=CapabilitySpec(
            name=capability_name,
            summary="route confirmed reminder and allowlisted-app voice commands",
            when_to_use="when a typed reminder or configured trusted-app command is staged",
            egress="local",
            speaks=True,
            side_effecting=True,
            planner_tool=False,
            user_facing=False,
            authority="none",
        ),
    )
    return dispatcher


__all__ = [
    "AppLaunchError",
    "AppLaunchResult",
    "AppLauncher",
    "DeviceToolCommand",
    "DeviceToolCommandDispatcher",
    "GtkAppLauncher",
    "TrustedApp",
    "TrustedAppError",
    "TrustedAppManager",
    "TrustedAppsConfig",
    "attach_device_tool_command_dispatcher",
    "attach_trusted_app_capabilities",
    "attach_trusted_app_capability",
    "build_trusted_app_manager",
]
