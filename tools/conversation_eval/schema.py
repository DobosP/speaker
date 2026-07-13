from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TraceEvent:
    """One ordered observation in a conversation evaluation trace."""

    sequence: int
    elapsed_ms: float
    kind: str
    task_id: str = ""
    turn_id: int | None = None
    payload: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    expected: object
    actual: object

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TurnResult:
    index: int
    input_text: str
    attempted_output: str
    # Text terminally attested at the output sink. This is not proof that a
    # human heard acoustic playback; the evaluator opens no speaker device.
    sink_attested_output: str
    grade: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ScenarioResult:
    scenario_id: str
    description: str
    model: str
    run_index: int
    passed: bool
    duration_ms: float
    turns: tuple[TurnResult, ...]
    checks: tuple[CheckResult, ...]
    trace: tuple[TraceEvent, ...]
    memory_before: tuple[dict[str, object], ...]
    memory_after: tuple[dict[str, object], ...]
    metrics: tuple[dict[str, object], ...]
    model_calls: tuple[dict[str, object], ...] = ()
    error: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "description": self.description,
            "model": self.model,
            "run_index": self.run_index,
            "passed": self.passed,
            "duration_ms": self.duration_ms,
            "turns": [turn.as_dict() for turn in self.turns],
            "checks": [check.as_dict() for check in self.checks],
            "trace": [event.as_dict() for event in self.trace],
            "memory_before": list(self.memory_before),
            "memory_after": list(self.memory_after),
            "metrics": list(self.metrics),
            "model_calls": list(self.model_calls),
            "error": self.error,
        }


@dataclass(frozen=True)
class ModelSummary:
    model: str
    runs: int
    scenarios: int
    passed_results: int
    total_results: int
    pass_at_1: bool
    pass_power_k: bool
    scenario_reliability: dict[str, dict[str, object]]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


ScenarioKind = Literal[
    "conversation",
    "barge_stop",
    "barge_redirect",
    "tool_failure_recovery",
    "mid_tool_barge",
    "untrusted_tool_result",
]
AnswerRoute = Literal["fast", "main"]


@dataclass(frozen=True)
class TurnSpec:
    text: str
    expect: tuple[str, ...] = ()
    forbid: tuple[str, ...] = ()
    exact_response: str = ""
    max_sentences: int | None = None
    max_words: int | None = None


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    description: str
    kind: ScenarioKind
    turns: tuple[TurnSpec, ...]
    expected_tools: tuple[str, ...] = ()
    expected_tool_query_terms: tuple[tuple[str, ...], ...] = ()
    expected_tool_ok: tuple[bool, ...] = ()
    expected_capabilities: tuple[str, ...] | None = None
    expected_task_terminals: tuple[str, ...] | None = None
    forbidden_tools: tuple[str, ...] = ()
    required_events: tuple[str, ...] = ()
    forbidden_events: tuple[str, ...] = ()
    expected_cancel_reason: str = ""
    expected_answer_routes: tuple[AnswerRoute, ...] | None = None
    require_no_stale_output: bool = False
    timeout_sec: float = 12.0
