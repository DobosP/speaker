"""Pure schema and report contracts for the headless conversation-flow gate.

This module deliberately carries no runtime, model, audio, or filesystem work.
It grades the completeness and integrity of already-produced scripted-runtime
flow results without turning fixture transcripts into speech-recognition
evidence.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
import math
from numbers import Real
from typing import Mapping, Sequence

from .schema import TraceEvent


FLOW_SCHEMA_VERSION = 1
FLOW_SCENARIO_SET_VERSION = 1
FLOW_REQUIRED_RUNS = 3
FLOW_PROFILE = "headless_scripted_runtime"
FLOW_SCENARIO_IDS = (
    "turn_incremental",
    "barge_recovery",
    "slow_tool_complete_followup",
    "stop_restart_fence",
)


_FLOW_EVIDENCE_ROWS = (
    (
        "headless_runtime_plumbing",
        "COVERED",
        "Scripted transcript-to-runtime lifecycle, ownership, and teardown evidence.",
    ),
    (
        "typed_turn_endpoint_policy",
        "COVERED",
        "A pure endpoint-policy decision plus typed partial/final revision handling.",
    ),
    (
        "scripted_streaming_playback",
        "COVERED",
        "Streaming text reaches a tracked scripted sink with terminal receipts.",
    ),
    (
        "interruption_recovery",
        "COVERED",
        "Task, generation, and playback ownership are checked across interruption.",
    ),
    (
        "read_tool_lifecycle",
        "COVERED",
        "A deterministic read-only tool completes before its answer and follow-up.",
    ),
    (
        "synthetic_action_tool_lifecycle",
        "COVERED",
        "Confirmation and actor-owned action-tool identities use a no-op "
        "in-memory provider; no external effect occurs.",
    ),
    (
        "fresh_session_fencing",
        "COVERED",
        "Bounded stop and a newly assembled session are checked for cross-session leakage.",
    ),
    (
        "causal_headless_timing",
        "COVERED",
        "One monotonic test clock proves finite causal ordering, not performance.",
    ),
    (
        "fixture_transcripts",
        "plumbing_only",
        "Known text drives control-plane plumbing and is never STT-quality evidence.",
    ),
    (
        "microphone_native_reader",
        "NOT_COVERED",
        "No microphone or production native-reader input is opened.",
    ),
    (
        "vad_capture",
        "NOT_COVERED",
        "No capture-domain VAD, policy-to-engine callback wiring, resampling, "
        "or PCM continuity is exercised.",
    ),
    (
        "actual_stt_wer",
        "NOT_COVERED",
        "No recognizer decodes audio and no WER, CER, or partial quality is measured.",
    ),
    (
        "real_tts_audio_pacing",
        "NOT_COVERED",
        "No real synthesis, PCM output, buffer pacing, or audible continuity is measured.",
    ),
    (
        "aec_room_open_speaker",
        "NOT_COVERED",
        "No echo cancellation, room acoustics, or open-speaker path is exercised.",
    ),
    (
        "physical_audibility",
        "NOT_COVERED",
        "A scripted sink receipt is not proof that a person heard audio.",
    ),
    (
        "live_perceived_latency",
        "NOT_COVERED",
        "Headless timestamps prove causal plumbing only, not live or perceived latency.",
    ),
    (
        "external_effects",
        "NOT_COVERED",
        "No reminder, trusted-app, network, or other external mutation is performed.",
    ),
)


def _json_safe(value: object) -> object:
    """Return a deterministic value accepted by strict ``json.dumps``.

    Invalid evidence still belongs in a red report.  Non-finite numbers are
    therefore represented by stable diagnostic strings rather than leaking
    JSON's non-standard NaN/Infinity tokens.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if math.isnan(value):
            return "nonfinite:nan"
        return "nonfinite:+inf" if value > 0 else "nonfinite:-inf"
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        converted = [_json_safe(item) for item in value]
        return sorted(converted, key=lambda item: repr(item))
    if is_dataclass(value):
        return _json_safe(asdict(value))
    return f"<{type(value).__name__}>"


def _finite_nonnegative(value: object) -> bool:
    return bool(
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


@dataclass(frozen=True)
class FlowCheck:
    name: str
    ok: bool
    expected: object
    actual: object

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "ok": self.ok,
            "expected": _json_safe(self.expected),
            "actual": _json_safe(self.actual),
        }


@dataclass(frozen=True)
class FlowScenarioResult:
    scenario_id: str
    description: str
    run_index: int
    passed: bool
    duration_ms: float
    checks: tuple[FlowCheck, ...]
    trace: tuple[TraceEvent, ...]
    error: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "description": self.description,
            "run_index": self.run_index,
            "passed": self.passed,
            "duration_ms": _json_safe(self.duration_ms),
            "checks": [check.as_dict() for check in self.checks],
            "trace": [_json_safe(event.as_dict()) for event in self.trace],
            "error": self.error,
        }


def _result_sort_key(result: FlowScenarioResult) -> tuple[object, ...]:
    scenario_order = {
        scenario_id: index for index, scenario_id in enumerate(FLOW_SCENARIO_IDS)
    }
    run_index = result.run_index
    valid_run_index = isinstance(run_index, int) and not isinstance(run_index, bool)
    return (
        scenario_order.get(result.scenario_id, len(scenario_order)),
        str(result.scenario_id),
        int(run_index) if valid_run_index else FLOW_REQUIRED_RUNS + 1,
        repr(run_index),
    )


def _coverage(
    results: tuple[FlowScenarioResult, ...],
    requested_runs: int,
) -> dict[str, object]:
    required_keys = tuple(
        (scenario_id, run_index)
        for scenario_id in FLOW_SCENARIO_IDS
        for run_index in range(1, FLOW_REQUIRED_RUNS + 1)
    )
    counts = Counter((result.scenario_id, result.run_index) for result in results)
    observed: dict[str, list[object]] = defaultdict(list)
    for result in results:
        observed[str(result.scenario_id)].append(_json_safe(result.run_index))

    unknown_ids = sorted(
        {
            str(result.scenario_id)
            for result in results
            if result.scenario_id not in FLOW_SCENARIO_IDS
        }
    )
    invalid_indices = sorted(
        (
            {
                "scenario_id": str(result.scenario_id),
                "run_index": _json_safe(result.run_index),
            }
            for result in results
            if not (
                isinstance(result.run_index, int)
                and not isinstance(result.run_index, bool)
                and 1 <= result.run_index <= FLOW_REQUIRED_RUNS
            )
        ),
        key=lambda row: (str(row["scenario_id"]), repr(row["run_index"])),
    )
    duplicates = [
        {"scenario_id": str(scenario_id), "run_index": _json_safe(run_index)}
        for (scenario_id, run_index), count in counts.items()
        if count > 1
    ]
    duplicates.sort(
        key=lambda row: (
            FLOW_SCENARIO_IDS.index(row["scenario_id"])
            if row["scenario_id"] in FLOW_SCENARIO_IDS
            else len(FLOW_SCENARIO_IDS),
            str(row["scenario_id"]),
            repr(row["run_index"]),
        )
    )
    missing = [
        {"scenario_id": scenario_id, "run_index": run_index}
        for scenario_id, run_index in required_keys
        if counts.get((scenario_id, run_index), 0) == 0
    ]
    requested_runs_exact = bool(
        isinstance(requested_runs, int)
        and not isinstance(requested_runs, bool)
        and requested_runs == FLOW_REQUIRED_RUNS
    )
    exact = bool(
        requested_runs_exact
        and len(results) == len(required_keys)
        and not unknown_ids
        and not invalid_indices
        and not duplicates
        and not missing
    )
    observed_rows = {
        scenario_id: sorted(
            observed.get(scenario_id, []),
            key=repr,
        )
        for scenario_id in FLOW_SCENARIO_IDS
    }
    for scenario_id in sorted(set(observed) - set(FLOW_SCENARIO_IDS)):
        observed_rows[scenario_id] = sorted(observed[scenario_id], key=repr)
    return {
        "required_scenario_ids": list(FLOW_SCENARIO_IDS),
        "required_run_indices": list(range(1, FLOW_REQUIRED_RUNS + 1)),
        "expected_result_count": len(required_keys),
        "observed_result_count": len(results),
        "observed": observed_rows,
        "unknown_scenario_ids": unknown_ids,
        "invalid_run_indices": invalid_indices,
        "duplicate_scenario_runs": duplicates,
        "missing_scenario_runs": missing,
        "requested_runs_exact": requested_runs_exact,
        "exact": exact,
    }


def build_flow_report(
    results: Sequence[FlowScenarioResult],
    requested_runs: int,
) -> dict[str, object]:
    """Build a strict aggregate report over exactly four scenarios x three runs."""

    ordered_results = tuple(sorted(tuple(results), key=_result_sort_key))
    coverage = _coverage(ordered_results, requested_runs)
    results_passed = bool(
        ordered_results and all(result.passed is True for result in ordered_results)
    )
    checks_present = bool(
        ordered_results and all(bool(result.checks) for result in ordered_results)
    )
    checks_passed = bool(
        checks_present
        and all(
            check.ok is True
            for result in ordered_results
            for check in result.checks
        )
    )
    errors_empty = bool(
        ordered_results and all(result.error == "" for result in ordered_results)
    )
    durations_valid = bool(
        ordered_results
        and all(_finite_nonnegative(result.duration_ms) for result in ordered_results)
    )
    traces_present = bool(
        ordered_results and all(bool(result.trace) for result in ordered_results)
    )
    timestamps_valid = bool(
        traces_present
        and all(
            _finite_nonnegative(event.elapsed_ms)
            for result in ordered_results
            for event in result.trace
        )
    )
    trace_sequences_valid = bool(
        traces_present
        and all(
            tuple(event.sequence for event in result.trace)
            == tuple(range(1, len(result.trace) + 1))
            for result in ordered_results
        )
    )
    trace_elapsed_monotonic = bool(
        timestamps_valid
        and all(
            all(
                before.elapsed_ms <= after.elapsed_ms
                for before, after in zip(result.trace, result.trace[1:])
            )
            for result in ordered_results
        )
    )
    trace_within_duration = bool(
        timestamps_valid
        and durations_valid
        and all(
            max(event.elapsed_ms for event in result.trace)
            <= float(result.duration_ms) + 0.001
            for result in ordered_results
        )
    )
    trace_result_binding = bool(
        traces_present
        and all(
            len(terminals) == 1
            and result.trace[-1] is terminals[0]
            and terminals[0].turn_id == result.run_index
            and terminals[0].payload.get("scenario_id") == result.scenario_id
            and terminals[0].payload.get("checks_ok") is result.passed
            and terminals[0].payload.get("error") == result.error
            for result in ordered_results
            for terminals in (
                tuple(event for event in result.trace if event.kind == "flow.complete"),
            )
        )
    )
    passed = bool(
        coverage["exact"]
        and results_passed
        and checks_passed
        and errors_empty
        and durations_valid
        and timestamps_valid
        and trace_sequences_valid
        and trace_elapsed_monotonic
        and trace_within_duration
        and trace_result_binding
    )
    strict_result_passes = tuple(
        bool(
            result.passed is True
            and result.checks
            and all(check.ok is True for check in result.checks)
            and result.error == ""
            and _finite_nonnegative(result.duration_ms)
            and result.trace
            and all(_finite_nonnegative(event.elapsed_ms) for event in result.trace)
            and tuple(event.sequence for event in result.trace)
            == tuple(range(1, len(result.trace) + 1))
            and all(
                before.elapsed_ms <= after.elapsed_ms
                for before, after in zip(result.trace, result.trace[1:])
            )
            and max(event.elapsed_ms for event in result.trace)
            <= float(result.duration_ms) + 0.001
            and len(
                terminals := tuple(
                    event for event in result.trace if event.kind == "flow.complete"
                )
            )
            == 1
            and result.trace[-1] is terminals[0]
            and terminals[0].turn_id == result.run_index
            and terminals[0].payload.get("scenario_id") == result.scenario_id
            and terminals[0].payload.get("checks_ok") is result.passed
            and terminals[0].payload.get("error") == result.error
        )
        for result in ordered_results
    )
    pass_at_1 = bool(
        coverage["exact"]
        and all(
            strict_result_passes[index]
            for index, result in enumerate(ordered_results)
            if result.run_index == 1
        )
    )
    summary = {
        "scenarios": len(FLOW_SCENARIO_IDS),
        "runs_per_scenario": FLOW_REQUIRED_RUNS,
        "passed_results": sum(strict_result_passes),
        "total_results": len(ordered_results),
        "pass_at_1": pass_at_1,
        "pass_power_3": passed,
    }
    evidence_matrix = {
        name: {"status": status, "scope": scope}
        for name, status, scope in _FLOW_EVIDENCE_ROWS
    }
    gate = {
        "exact_coverage": bool(coverage["exact"]),
        "results_passed": results_passed,
        "checks_present": checks_present,
        "checks_passed": checks_passed,
        "errors_empty": errors_empty,
        "durations_finite_nonnegative": durations_valid,
        "traces_present": traces_present,
        "timestamps_finite_nonnegative": timestamps_valid,
        "trace_sequences_contiguous": trace_sequences_valid,
        "trace_elapsed_monotonic": trace_elapsed_monotonic,
        "trace_within_duration": trace_within_duration,
        "trace_result_binding": trace_result_binding,
        "passed": passed,
    }
    return {
        "schema_version": FLOW_SCHEMA_VERSION,
        "scenario_set_version": FLOW_SCENARIO_SET_VERSION,
        "profile": FLOW_PROFILE,
        "scope": (
            "Deterministic headless control-plane lifecycle only; fixture text "
            "and scripted sink receipts are not acoustic quality evidence."
        ),
        "required_runs": FLOW_REQUIRED_RUNS,
        "requested_runs": _json_safe(requested_runs),
        "fixture_transcripts": {
            "classification": "plumbing_only",
            "stt_quality_evidence": False,
        },
        "evidence_matrix": evidence_matrix,
        "summary": summary,
        "coverage": coverage,
        "results": [result.as_dict() for result in ordered_results],
        "gate": gate,
        "passed": passed,
    }
