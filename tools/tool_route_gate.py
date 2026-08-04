"""Pure, aggregate-only recognized-text routing gate.

The gate reuses the production deterministic analyzer, planner, and strict
device-command matcher.  It deliberately has no capability invocation seam:
reminders use a fixed clock and empty read-only state, and the app launcher and
capability registry are sentinels that raise if a future change tries to act.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from types import MappingProxyType
from typing import Mapping, Sequence


EXPECTED_ROUTES = (
    "none",
    "vault.search",
    "web.search",
    "reminder.create",
    "reminder.list",
    "reminder.cancel",
    "app.open",
)
_EXPECTED_ROUTE_SET = frozenset(EXPECTED_ROUTES)
_POSITIVE_ROUTES = frozenset(EXPECTED_ROUTES[1:])
_DEVICE_ROUTES = frozenset(
    {"reminder.create", "reminder.list", "reminder.cancel", "app.open"}
)
_ACTUAL_ROUTES = frozenset(
    (*EXPECTED_ROUTES, "control", "generic_action", "other_action")
)
_EXPECTED_PREFIX = "expected-tool."
_CONTROL_KINDS = frozenset({"stop", "confirm", "deny", "mode_switch"})
_PROFILE_DOMAIN = b"speaker-recorded-tool-route-gate-profile-v1\0"


class ToolRouteGateError(RuntimeError):
    """A deliberately detail-free route-gate prerequisite failure."""

    def __init__(self) -> None:
        super().__init__("tool route gate prerequisites unavailable")


@dataclass(frozen=True)
class ToolRouteGateProfile:
    """Explicit synthetic availability; aliases never enter reports or reprs."""

    vault_enabled: bool = False
    reminders_enabled: bool = False
    app_aliases: tuple[str, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        try:
            if (
                type(self.vault_enabled) is not bool
                or type(self.reminders_enabled) is not bool
            ):
                raise ToolRouteGateError()
            if isinstance(self.app_aliases, (str, bytes)):
                raise ToolRouteGateError()
            aliases = tuple(self.app_aliases)
            if any(type(alias) is not str for alias in aliases) or len(
                set(aliases)
            ) != len(aliases):
                raise ToolRouteGateError()
            object.__setattr__(self, "app_aliases", aliases)
        except ToolRouteGateError:
            raise
        except Exception:
            raise ToolRouteGateError() from None


class _NoOpenLauncher:
    def open(self, _desktop_id: str) -> None:
        raise ToolRouteGateError()


class _DeterministicReminderAdapter:
    def __init__(self) -> None:
        from core.reminders import ReminderParser

        fixed_now = datetime(2030, 1, 1, 12, 0, tzinfo=timezone.utc)
        self.parser = ReminderParser(
            clock=lambda: fixed_now,
            local_timezone=timezone.utc,
        )

    def resolve_id(self, _reference: str) -> str:
        from core.reminders import ReminderError

        raise ReminderError("unavailable")

    def list(self) -> tuple[object, ...]:
        return ()


def _trusted_app_manager(aliases: Sequence[str]):
    if not aliases:
        return None
    try:
        from core.trusted_apps import (
            TrustedApp,
            TrustedAppManager,
            TrustedAppsConfig,
        )

        apps = tuple(
            TrustedApp.from_entry(
                alias,
                {
                    "connector": "desktop_launch",
                    "desktop_id": f"route-gate-{index}.desktop",
                    "operations": ["open"],
                },
            )
            for index, alias in enumerate(aliases)
        )
        return TrustedAppManager(
            TrustedAppsConfig(enabled=True, apps=apps),
            _NoOpenLauncher(),
        )
    except Exception:
        raise ToolRouteGateError() from None


class ToolRouteClassifier:
    """Production-shaped deterministic routing with no execution authority."""

    def __init__(self, profile: ToolRouteGateProfile) -> None:
        try:
            from always_on_agent.capabilities import CapabilityRegistry
            from always_on_agent.planner import TaskPlanner
            from always_on_agent.speech_analyzer import LiveSpeechAnalyzer, ModePolicy
            from core.trusted_apps import DeviceToolCommandDispatcher

            if not isinstance(profile, ToolRouteGateProfile):
                raise ToolRouteGateError()

            class _NoInvokeRegistry(CapabilityRegistry):
                def invoke(self, *_args, **_kwargs):
                    raise ToolRouteGateError()

            reminder_manager = (
                _DeterministicReminderAdapter() if profile.reminders_enabled else None
            )
            app_manager = _trusted_app_manager(profile.app_aliases)
            self._dispatcher = (
                DeviceToolCommandDispatcher(
                    _NoInvokeRegistry(),
                    reminder_manager=reminder_manager,  # type: ignore[arg-type]
                    trusted_app_manager=app_manager,
                )
                if reminder_manager is not None or app_manager is not None
                else None
            )
            self._analyzer = LiveSpeechAnalyzer(
                ModePolicy(
                    vault_search_enabled=profile.vault_enabled,
                    device_tools_enabled=self._dispatcher is not None,
                    device_tool_matcher=(
                        self._dispatcher.match if self._dispatcher is not None else None
                    ),
                )
            )
            self._planner = TaskPlanner()
        except ToolRouteGateError:
            raise
        except Exception:
            raise ToolRouteGateError() from None

    def __repr__(self) -> str:
        return "ToolRouteClassifier(<private>)"

    def classify(self, text: str) -> str:
        try:
            from always_on_agent.events import Mode
            from always_on_agent.models import IntentKind
            from core.trusted_apps import DeviceToolCommand

            if type(text) is not str or not text.strip():
                raise ToolRouteGateError()
            observation = self._analyzer.observe(text, is_final=True)
            decision = self._analyzer.decide(observation, Mode.ASSISTANT)
            if decision.kind.value in _CONTROL_KINDS:
                return "control"
            if decision.kind == IntentKind.IGNORE:
                return "none"

            plan = self._planner.plan(decision)
            capabilities = tuple(step.capability for step in plan.steps)
            if decision.kind == IntentKind.COMMAND:
                prepared = decision.metadata.get("prepared_device_command")
                if (
                    isinstance(prepared, DeviceToolCommand)
                    and prepared.query == decision.text
                    and capabilities == ("device.command",)
                    and prepared.capability_name in _DEVICE_ROUTES
                ):
                    return prepared.capability_name
                if capabilities == ("command.stage",):
                    return "generic_action"
                return "other_action"
            if decision.kind in {IntentKind.SEARCH, IntentKind.RESEARCH}:
                routed = tuple(
                    capability
                    for capability in capabilities
                    if capability in {"vault.search", "web.search"}
                )
                if len(routed) == 1:
                    return routed[0]
                if len(routed) > 1:
                    raise ToolRouteGateError()
                return "other_action"
            if decision.kind == IntentKind.ASSISTANT and capabilities == (
                "assistant.answer",
            ):
                return "none"
            if decision.kind in {IntentKind.DICTATION, IntentKind.MEETING_NOTE}:
                return "other_action"
            return "other_action"
        except ToolRouteGateError:
            raise
        except Exception:
            raise ToolRouteGateError() from None


def expected_route(tags: Sequence[str]) -> str:
    """Parse exactly one closed expected-tool annotation without echoing it."""

    try:
        if isinstance(tags, (str, bytes)) or not isinstance(tags, Sequence):
            raise ToolRouteGateError()
        found: list[str] = []
        for tag in tags:
            if type(tag) is not str:
                raise ToolRouteGateError()
            if tag.startswith(_EXPECTED_PREFIX):
                route = tag.removeprefix(_EXPECTED_PREFIX)
                if route not in _EXPECTED_ROUTE_SET:
                    raise ToolRouteGateError()
                found.append(route)
        if len(found) != 1:
            raise ToolRouteGateError()
        return found[0]
    except ToolRouteGateError:
        raise
    except Exception:
        raise ToolRouteGateError() from None


@dataclass(frozen=True)
class ToolRouteGateTotals:
    annotated_cases: int
    decisions: int
    single_decision_cases: int
    empty_decisions: int
    expected_positive_cases: int
    expected_none_cases: int
    exact_cases: int
    misses: int
    wrong_tool: int
    unexpected_tool: int
    unexpected_control: int
    unexpected_action: int
    multi_decision_cases: int
    attempts: Mapping[str, int] = field(repr=False)
    hits: Mapping[str, int] = field(repr=False)

    def __post_init__(self) -> None:
        try:
            counter_names = (
                "annotated_cases",
                "decisions",
                "single_decision_cases",
                "empty_decisions",
                "expected_positive_cases",
                "expected_none_cases",
                "exact_cases",
                "misses",
                "wrong_tool",
                "unexpected_tool",
                "unexpected_control",
                "unexpected_action",
                "multi_decision_cases",
            )
            if any(
                type(getattr(self, name)) is not int or getattr(self, name) < 0
                for name in counter_names
            ):
                raise ToolRouteGateError()
            attempts = _closed_route_counts(self.attempts)
            hits = _closed_route_counts(self.hits)
            if (
                self.expected_positive_cases + self.expected_none_cases
                != self.annotated_cases
                or sum(attempts.values()) != self.annotated_cases
                or attempts["none"] != self.expected_none_cases
                or sum(attempts[route] for route in EXPECTED_ROUTES[1:])
                != self.expected_positive_cases
                or sum(hits.values()) != self.exact_cases
                or any(hits[route] > attempts[route] for route in EXPECTED_ROUTES)
                or self.single_decision_cases + self.multi_decision_cases
                > self.annotated_cases
                or self.empty_decisions > self.decisions
                or self.exact_cases > self.annotated_cases
                or any(
                    value > self.annotated_cases
                    for value in (
                        self.misses,
                        self.wrong_tool,
                        self.unexpected_tool,
                        self.unexpected_control,
                        self.unexpected_action,
                    )
                )
            ):
                raise ToolRouteGateError()
            object.__setattr__(self, "attempts", MappingProxyType(attempts))
            object.__setattr__(self, "hits", MappingProxyType(hits))
        except ToolRouteGateError:
            raise
        except Exception:
            raise ToolRouteGateError() from None

    @property
    def complete(self) -> bool:
        return (
            self.expected_positive_cases > 0
            and self.expected_none_cases > 0
            and self.decisions == self.annotated_cases
            and self.single_decision_cases == self.annotated_cases
            and self.empty_decisions == 0
            and self.exact_cases == self.annotated_cases
            and self.misses == 0
            and self.wrong_tool == 0
            and self.unexpected_tool == 0
            and self.unexpected_control == 0
            and self.unexpected_action == 0
            and self.multi_decision_cases == 0
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "annotated_cases": self.annotated_cases,
            "decisions": self.decisions,
            "single_decision_cases": self.single_decision_cases,
            "empty_decisions": self.empty_decisions,
            "expected_positive_cases": self.expected_positive_cases,
            "expected_none_cases": self.expected_none_cases,
            "exact_cases": self.exact_cases,
            "misses": self.misses,
            "wrong_tool": self.wrong_tool,
            "unexpected_tool": self.unexpected_tool,
            "unexpected_control": self.unexpected_control,
            "unexpected_action": self.unexpected_action,
            "multi_decision_cases": self.multi_decision_cases,
            "per_expected": [
                {
                    "route": route,
                    "attempts": int(self.attempts.get(route, 0)),
                    "hits": int(self.hits.get(route, 0)),
                }
                for route in EXPECTED_ROUTES
            ],
            "complete": self.complete,
        }


class ToolRouteGateAccumulator:
    """Consume private terminal strings case-by-case and retain counters only."""

    def __init__(self, classifier: ToolRouteClassifier) -> None:
        if not isinstance(classifier, ToolRouteClassifier):
            raise ToolRouteGateError()
        self._classifier = classifier
        self._counts: Counter[str] = Counter()
        self._attempts: Counter[str] = Counter()
        self._hits: Counter[str] = Counter()

    def add_case(self, tags: Sequence[str], selected: Sequence[str]) -> None:
        try:
            expected = expected_route(tags)
            if isinstance(selected, (str, bytes)) or not isinstance(selected, Sequence):
                raise ToolRouteGateError()
            decisions = tuple(selected)
            if any(type(value) is not str for value in decisions):
                raise ToolRouteGateError()
            self._counts["annotated_cases"] += 1
            self._counts["decisions"] += len(decisions)
            self._attempts[expected] += 1
            self._counts[
                "expected_none_cases"
                if expected == "none"
                else "expected_positive_cases"
            ] += 1
            self._counts["empty_decisions"] += sum(
                not value.strip() for value in decisions
            )
            if len(decisions) == 1:
                self._counts["single_decision_cases"] += 1
            elif len(decisions) > 1:
                self._counts["multi_decision_cases"] += 1
            if len(decisions) != 1 or not decisions[0].strip():
                if expected != "none":
                    self._counts["misses"] += 1
                return

            actual = self._classifier.classify(decisions[0])
            if actual not in _ACTUAL_ROUTES:
                raise ToolRouteGateError()
            if actual == expected:
                self._counts["exact_cases"] += 1
                self._hits[expected] += 1
                return
            if expected != "none":
                self._counts["misses"] += 1
                if actual in _POSITIVE_ROUTES:
                    self._counts["wrong_tool"] += 1
            if expected == "none" and actual in _POSITIVE_ROUTES:
                self._counts["unexpected_tool"] += 1
            if actual == "control":
                self._counts["unexpected_control"] += 1
            if actual in {"generic_action", "other_action"}:
                self._counts["unexpected_action"] += 1
        except ToolRouteGateError:
            raise
        except Exception:
            raise ToolRouteGateError() from None

    def totals(self) -> ToolRouteGateTotals:
        values = {
            name: int(self._counts[name])
            for name in (
                "annotated_cases",
                "decisions",
                "single_decision_cases",
                "empty_decisions",
                "expected_positive_cases",
                "expected_none_cases",
                "exact_cases",
                "misses",
                "wrong_tool",
                "unexpected_tool",
                "unexpected_control",
                "unexpected_action",
                "multi_decision_cases",
            )
        }
        return ToolRouteGateTotals(
            **values,
            attempts={route: int(self._attempts[route]) for route in EXPECTED_ROUTES},
            hits={route: int(self._hits[route]) for route in EXPECTED_ROUTES},
        )


def _closed_route_counts(value: Mapping[str, int]) -> dict[str, int]:
    try:
        if not isinstance(value, Mapping):
            raise ToolRouteGateError()
        copied = dict(value.items())
        if set(copied) != _EXPECTED_ROUTE_SET or any(
            type(route) is not str or type(count) is not int or count < 0
            for route, count in copied.items()
        ):
            raise ToolRouteGateError()
        return {route: copied[route] for route in EXPECTED_ROUTES}
    except ToolRouteGateError:
        raise
    except Exception:
        raise ToolRouteGateError() from None


def profile_digest(profile: ToolRouteGateProfile) -> str:
    """Bind the inert availability profile without disclosing private aliases."""

    try:
        if not isinstance(profile, ToolRouteGateProfile):
            raise ToolRouteGateError()
        app_manager = _trusted_app_manager(profile.app_aliases)
        normalized_aliases = app_manager.aliases if app_manager is not None else ()
        payload = {
            "app_aliases": sorted(normalized_aliases),
            "fixed_clock": "2030-01-01T12:00:00Z",
            "reminders_enabled": profile.reminders_enabled,
            "vault_enabled": profile.vault_enabled,
            "version": 1,
            "web_planning": True,
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        return hashlib.sha256(_PROFILE_DOMAIN + encoded).hexdigest()
    except ToolRouteGateError:
        raise
    except Exception:
        raise ToolRouteGateError() from None


def no_regression(
    baseline: ToolRouteGateTotals,
    candidate: ToolRouteGateTotals,
) -> bool:
    """Require identical annotations and monotonic aggregate route safety."""

    if not isinstance(baseline, ToolRouteGateTotals) or not isinstance(
        candidate, ToolRouteGateTotals
    ):
        return False
    return (
        candidate.annotated_cases == baseline.annotated_cases
        and candidate.expected_positive_cases == baseline.expected_positive_cases
        and candidate.expected_none_cases == baseline.expected_none_cases
        and candidate.single_decision_cases >= baseline.single_decision_cases
        and candidate.empty_decisions <= baseline.empty_decisions
        and candidate.exact_cases >= baseline.exact_cases
        and candidate.misses <= baseline.misses
        and candidate.wrong_tool <= baseline.wrong_tool
        and candidate.unexpected_tool <= baseline.unexpected_tool
        and candidate.unexpected_control <= baseline.unexpected_control
        and candidate.unexpected_action <= baseline.unexpected_action
        and candidate.multi_decision_cases <= baseline.multi_decision_cases
        and all(
            candidate.attempts.get(route, -1) == baseline.attempts.get(route, -2)
            and candidate.hits.get(route, -1) >= baseline.hits.get(route, -2)
            for route in EXPECTED_ROUTES
        )
    )


__all__ = [
    "EXPECTED_ROUTES",
    "ToolRouteClassifier",
    "ToolRouteGateAccumulator",
    "ToolRouteGateError",
    "ToolRouteGateProfile",
    "ToolRouteGateTotals",
    "expected_route",
    "no_regression",
    "profile_digest",
]
